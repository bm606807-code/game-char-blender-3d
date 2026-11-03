"""STAGES 4-7 - Build the character as real 3D geometry and assemble it in Blender.

Phase A (NumPy/SciPy/scikit-image): sculpt body, head, hands, boots and costume shells as
signed-distance fields and polygonise them with marching cubes. The raw meshes are cached
untouched in build/cache/raw_meshes.npz (the "original mesh" backup).

Phase B (Blender / bpy): import, clean and decimate the meshes; add UVs, PBR materials,
hard-surface parts (backpack, pouches, goggles, buckles), strand geometry (hair, brows,
lashes, laces, piping) and the rock pedestal; organise collections; save the .blend.

Usage:
  python3 scripts/03_build_character.py            # full build
  python3 scripts/03_build_character.py --reuse    # reuse the cached Phase A meshes
"""
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.ndimage import gaussian_filter
from charlib import sdf as S
from charlib import anatomy as A
from charlib.anatomy import SK, V, lerp

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "build" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
RAW = CACHE / "raw_meshes.npz"
BLEND_DIR = ROOT / "output" / "blend"
BLEND_DIR.mkdir(parents=True, exist_ok=True)

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def xform(v, M):
    return (np.c_[v, np.ones(len(v))] @ M.T)[:, :3].astype(np.float32)


# =============================================================================== PHASE A
def phase_a():
    meshes = {}  # name -> (verts, faces, material_key, collection_key)

    def put(name, vf, mat, coll):
        v, f = vf
        if len(f):
            meshes[name] = (v, f, mat, coll)
            log(f"  {name}: {len(f)} tris")

    # ------------------------------------------------------------------ body + costume shells
    log("body field (2 mm)")
    g = S.Grid((-0.33, -0.46, 0.03), (0.47, 0.20, 1.50), 0.002)
    A.body_field(g)
    put("Body_Bodysuit", S.extract(g.d, g.lo, g.h), "suit", "body")
    D = g.d
    Ds = gaussian_filter(D, 2.0).astype(np.float32)  # 4 mm blur: cloth bridges small concavities
    log("soft field ready")

    def torso_prox(X, Y, Z, rx=0.205, ry=0.135):
        return np.sqrt((X / rx) ** 2 + ((Y - 0.01) / ry) ** 2) - 1.0

    # ---- jacket (cropped, open front, sleeves rolled to the forearm)
    def jacket_region(X, Y, Z):
        zz = np.clip(Z, 1.165, 1.46)
        w = np.interp(zz, [1.165, 1.30, 1.46], [0.088, 0.080, 0.050])
        opening = np.maximum(np.abs(X) - w, Y + 0.01)
        torso = np.maximum(np.maximum(1.165 - Z, Z - 1.50), np.abs(X) - 0.19)
        torso = np.maximum(torso, -opening)
        r = torso
        for side in "RL":
            Sh, E, Wr = A.arm_pts(side)
            u = (Wr - E) / np.linalg.norm(Wr - E)
            Cf = lerp(E, Wr, 0.30)
            arm = np.minimum(S.round_cone(X, Y, Z, Sh - (E - Sh) * 0.1, E, 0.068, 0.062),
                             S.round_cone(X, Y, Z, E, Cf + u * 0.08, 0.062, 0.060))
            arm = np.maximum(arm, S.plane(X, Y, Z, Cf, u))
            r = np.minimum(r, arm)
        return r

    put("Jacket_Grey", S.extract_sub(
        g, lambda X, Y, Z, sl: np.maximum(S.shell(Ds[sl], 0.0035, 0.0055), jacket_region(X, Y, Z)),
        (-0.33, -0.42, 1.08), (0.47, 0.18, 1.50)), "jacket", "clothing")

    # ---- orange trim along the jacket front edge
    def trim_region(X, Y, Z):
        zz = np.clip(Z, 1.165, 1.46)
        w = np.interp(zz, [1.165, 1.30, 1.46], [0.088, 0.080, 0.050])
        opening = np.maximum(np.abs(X) - w, Y + 0.01)
        band = np.abs(opening) - 0.011
        band = np.maximum(band, np.maximum(1.165 - Z, Z - 1.47))
        return np.maximum(band, Y - 0.0)

    put("Jacket_Trim_Orange", S.extract_sub(
        g, lambda X, Y, Z, sl: np.maximum(S.shell(Ds[sl], 0.0075, 0.0045), trim_region(X, Y, Z)),
        (-0.2, -0.2, 1.14), (0.2, 0.05, 1.49)), "orange_fabric", "clothing")

    # ---- rolled sleeve cuffs (two folds)
    for side in "RL":
        Sh, E, Wr = A.arm_pts(side)
        u = (Wr - E) / np.linalg.norm(Wr - E)
        Cf = lerp(E, Wr, 0.30)
        R = S.frame_from_axis(u).T

        def cuff(X, Y, Z, sl, Cf=Cf, u=u, R=R):
            d1 = S.torus(X, Y, Z, Cf + u * 0.004, 0.044, 0.0125, R)
            d2 = S.torus(X, Y, Z, Cf - u * 0.016, 0.042, 0.0105, R)
            d = S.smin(d1, d2, 0.004)
            return np.maximum(d, -(D[sl] - 0.002))

        put(f"Sleeve_Cuff_{side}", S.extract_sub(g, cuff, Cf - 0.08, Cf + 0.08), "jacket_inner", "clothing")

    # ---- hood bunched around the neck/shoulders
    def hood(X, Y, Z, sl):
        phis = np.linspace(-np.pi + 0.62, np.pi - 0.62, 40)  # 0 = back, gap at the front
        pts = [V(0.108 * np.sin(p), 0.022 + 0.088 * np.cos(p), 1.438 + 0.030 * np.cos(p)) for p in phis]
        rad = [0.026 + 0.032 * (1 + np.cos(p)) / 2 for p in phis]
        d = S.chain(X, Y, Z, pts, rad)
        d = S.smin(d, S.ellipsoid(X, Y, Z, (0, 0.112, 1.470), (0.105, 0.058, 0.060)), 0.025)
        for s in (-1, 1):
            d = S.smin(d, S.ellipsoid(X, Y, Z, (s * 0.085, 0.090, 1.455), (0.050, 0.050, 0.045)), 0.02)
            d = S.smin(d, S.ellipsoid(X, Y, Z, (s * 0.110, 0.012, 1.436), (0.034, 0.048, 0.032)), 0.02)
            d = S.smin(d, S.ellipsoid(X, Y, Z, (s * 0.088, -0.060, 1.420), (0.030, 0.034, 0.026)), 0.02)
        # fabric folds
        d = d + 0.003 * np.sin(X * 140 + Z * 60) * np.sin(Y * 120 - Z * 40)
        return np.maximum(d, -(Ds[sl] - 0.007))

    put("Hood_Orange", S.extract_sub(g, hood, (-0.20, -0.13, 1.35), (0.20, 0.20, 1.55)), "hood", "clothing")

    # ---- corset / waist armour band
    def corset(X, Y, Z, sl):
        r = np.maximum(np.maximum(1.0 - Z, Z - 1.168), torso_prox(X, Y, Z, 0.19, 0.13))
        return np.maximum(S.shell(Ds[sl], 0.0015, 0.0045), r)

    put("Corset_Charcoal", S.extract_sub(g, corset, (-0.2, -0.15, 0.96), (0.2, 0.15, 1.2)), "leather_dark", "armour")

    # ---- belts
    def belt(X, Y, Z, sl):
        r = np.maximum(np.abs(Z - 0.948) - 0.023, torso_prox(X, Y, Z))
        return np.maximum(S.shell(Ds[sl], 0.003, 0.009), r)

    put("Belt_Main_Brown", S.extract_sub(g, belt, (-0.22, -0.15, 0.90), (0.22, 0.15, 1.0)), "leather_brown", "accessories")

    def belt2(X, Y, Z, sl):
        pl = S.plane(X, Y, Z, (0, 0, 0.895), (0.13, 0.0, 1.0))
        r = np.maximum(np.abs(pl) - 0.011, torso_prox(X, Y, Z))
        return np.maximum(S.shell(Ds[sl], 0.0025, 0.006), r)

    put("Belt_Strap_Orange", S.extract_sub(g, belt2, (-0.22, -0.15, 0.82), (0.22, 0.15, 0.97)), "orange_leather", "accessories")

    # ---- right-thigh holster straps
    def thigh_straps(X, Y, Z, sl):
        band = np.minimum(np.abs(Z - 0.705) - 0.011, np.abs(Z - 0.605) - 0.010)
        leg = np.sqrt((X + 0.095) ** 2 + (Y + 0.0) ** 2) - 0.13
        return np.maximum(S.shell(Ds[sl], 0.0015, 0.005), np.maximum(band, leg))

    put("Thigh_Straps_R", S.extract_sub(g, thigh_straps, (-0.25, -0.13, 0.57), (0.03, 0.13, 0.74)), "leather_brown", "accessories")

    # ---- knee pads + straps
    for side, s in (("R", -1), ("L", 1)):
        H, K, An = A.leg_pts(side)

        def pad(X, Y, Z, sl, K=K):
            e = S.ellipsoid(X, Y, Z, K + V(0, -0.020, 0.004), (0.051, 0.043, 0.070))
            d = S.shell(e, 0.0, 0.006)
            d = S.smax(d, Y - (K[1] - 0.004), 0.003)
            return np.maximum(d, -(D[sl] - 0.001))

        put(f"KneePad_{side}", S.extract_sub(g, pad, K - 0.09, K + 0.09), "metal", "armour")

        def kstrap(X, Y, Z, sl, K=K):
            band = np.minimum(np.abs(Z - (K[2] + 0.062)) - 0.007, np.abs(Z - (K[2] - 0.066)) - 0.007)
            leg = np.sqrt((X - K[0]) ** 2 + (Y - K[1]) ** 2) - 0.09
            return np.maximum(S.shell(Ds[sl], 0.001, 0.004), np.maximum(band, leg))

        put(f"KneeStraps_{side}", S.extract_sub(g, kstrap, K - 0.1, K + 0.1), "leather_dark", "armour")

        # ---- orange leg panels (raised appliqué)
        def panels(X, Y, Z, sl, H=H, K=K, An=An, s=s):
            shin = S.ellipsoid(X, Y, Z, lerp(K, An, 0.40) + V(s * 0.012, -0.040, 0), (0.022, 0.050, 0.095),
                               S.rot_xyz(0, s * 8, 0))
            thigh = S.ellipsoid(X, Y, Z, lerp(H, K, 0.55) + V(s * 0.070, -0.035, 0), (0.030, 0.060, 0.115),
                                S.rot_xyz(0, -s * 14, 0))
            hip = S.ellipsoid(X, Y, Z, V(s * 0.150, -0.040, 0.985), (0.035, 0.050, 0.040))
            r = np.minimum(np.minimum(shin, thigh), hip)
            return np.maximum(S.shell(Ds[sl], 0.0005, 0.004), r)

        put(f"Leg_Panels_Orange_{side}", S.extract_sub(g, panels, (H[0] - 0.2, -0.2, 0.15), (H[0] + 0.2, 0.15, 1.05)),
            "orange_leather", "armour")

    # ---- backpack harness straps over the shoulders
    def straps(X, Y, Z, sl):
        r = None
        for s in (-1, 1):
            path = [V(s * 0.070, 0.11, 1.24), V(s * 0.090, 0.075, 1.42), V(s * 0.100, 0.00, 1.468),
                    V(s * 0.105, -0.070, 1.40), V(s * 0.100, -0.098, 1.30), V(s * 0.098, -0.085, 1.19)]
            pts = S.catmull(path, 6)
            c = S.chain(X, Y, Z, pts, [0.021] * len(pts))
            r = c if r is None else np.minimum(r, c)
        return np.maximum(S.shell(Ds[sl], 0.0095, 0.0060), r)

    put("Harness_Straps", S.extract_sub(g, straps, (-0.2, -0.16, 1.15), (0.2, 0.16, 1.50)), "leather_dark", "backpack")

    del D, Ds, g

    # ------------------------------------------------------------------ head (0.9 mm)
    log("head field (0.9 mm)")
    hg = S.Grid((-0.105, -0.145, 1.415), (0.105, 0.150, 1.835), 0.0009)
    A.head_field(hg)
    put("Head_Skin", S.extract(hg.d, hg.lo, hg.h), "skin", "head")
    Dh = hg.d
    c = SK["head_c"]
    Rh = A.HEAD_R

    def head_local(X, Y, Z):
        return S.local(X, Y, Z, c, Rh)

    def hair_region(X, Y, Z):
        x, y, z = head_local(X, Y, Z)
        th = np.abs(np.arctan2(x, -y))  # 0 = front, pi = back
        zl = np.interp(th, [0.0, 0.25, 0.55, 0.95, 1.30, 1.62, 2.05, 2.5, np.pi],
                       [0.071, 0.069, 0.050, 0.040, 0.034, 0.030, -0.020, -0.060, -0.072])
        return zl - z

    def hair_cap(X, Y, Z, sl):
        hr = hair_region(X, Y, Z)
        thick = np.clip(-hr * 0.45, 0.0006, 0.0105)  # hair thins out toward the hairline
        d = np.maximum(np.maximum(Dh[sl] - thick, -(Dh[sl] + 0.0015)), hr)
        x, y, z = head_local(X, Y, Z)
        # combed-up grooves toward the bun
        d = d + 0.0009 * np.sin(np.arctan2(x, y + 0.03) * 70)
        return d

    put("Hair_Cap", S.extract_sub(hg, hair_cap, hg.lo, hg.lo + (hg.n - 1) * hg.h, 0.0), "hair", "hair")

    def bun(X, Y, Z, sl):
        x, y, z = head_local(X, Y, Z)
        bc = V(0.0, 0.030, 0.140)
        x, y, z = [bc[i] + (q - bc[i]) / 1.32 for i, q in enumerate(np.broadcast_arrays(x, y, z))]
        d = S.ellipsoid(x, y, z, bc, (0.034, 0.032, 0.026))
        rng = np.random.default_rng(5)
        for i in range(7):
            a = i / 7 * 2 * np.pi + rng.random() * 0.4
            p = bc + V(0.030 * np.cos(a), 0.028 * np.sin(a), -0.004 + rng.random() * 0.012)
            R = S.rot_xyz(rng.random() * 40 - 20, rng.random() * 40 - 20, np.degrees(a) + 90)
            d = S.smin(d, S.ellipsoid(x, y, z, p, (0.026, 0.014, 0.015), R), 0.008)
        d = S.smin(d, S.ellipsoid(x, y, z, bc + V(0.004, -0.002, 0.020), (0.022, 0.020, 0.014)), 0.01)
        d = S.smin(d, S.ellipsoid(x, y, z, V(0, 0.030, 0.112), (0.036, 0.034, 0.016)), 0.012)  # base
        d = d + 0.0012 * np.sin(np.arctan2(y - bc[1], x) * 26 + z * 300)
        return d * 1.32

    put("Hair_Bun", S.extract_sub(hg, bun, (-0.09, -0.06, 1.70), (0.09, 0.13, 1.83)), "hair", "hair")

    def goggle_strap(X, Y, Z, sl):
        x, y, z = head_local(X, Y, Z)
        pl = S.plane(x, y, z, (0, -0.09, 0.074), (0, 0.42, 1.0))
        return np.maximum(S.shell(Dh[sl], 0.0095, 0.0045), np.abs(pl) - 0.0105)

    put("Goggles_Strap", S.extract_sub(hg, goggle_strap, hg.lo, hg.lo + (hg.n - 1) * hg.h, 0.0), "leather_dark", "accessories")
    del Dh, hg

    # ------------------------------------------------------------------ hands (0.6 mm, hand-local)
    for side, pose in (("R", "relaxed"), ("L", "open_up")):
        log(f"hand {side} ({pose})")
        hgd = S.Grid((-0.075, -0.075, -0.06), (0.205, 0.085, 0.05), 0.0006)
        A.hand_field(hgd, pose)
        M = A.hand_matrix(side)
        flip = np.linalg.det(M[:3, :3]) < 0
        Dd = hgd.d

        def tw(vf):
            v, f = vf
            return xform(v, M), (f[:, ::-1] if flip else f)

        put(f"Hand_Skin_{side}", tw(S.extract(Dd, hgd.lo, hgd.h)), "skin", "hands")
        put(f"Glove_{side}", tw(S.extract_sub(
            hgd, lambda X, Y, Z, sl: np.maximum(S.shell(Dd[sl], 0.0004, 0.0021), A.glove_region(X, Y, Z, pose)),
            hgd.lo, hgd.lo + (hgd.n - 1) * hgd.h, 0.0)), "leather_dark", "hands")
        put(f"Glove_Knuckle_{side}", tw(S.extract_sub(
            hgd, lambda X, Y, Z, sl: np.maximum(S.shell(Dd[sl], 0.0020, 0.0016),
                                                np.maximum(np.maximum(0.058 - X, X - 0.098),
                                                           np.maximum(0.004 - Z, np.abs(Y) - 0.036))),
            (0.04, -0.06, -0.02), (0.11, 0.06, 0.04))), "metal_dark", "hands")
        put(f"Glove_WristStrap_{side}", tw(S.extract_sub(
            hgd, lambda X, Y, Z, sl: np.maximum(S.shell(Dd[sl], 0.0024, 0.0030), np.abs(X + 0.024) - 0.0065),
            (-0.04, -0.06, -0.04), (-0.01, 0.06, 0.04))), "orange_leather", "hands")
        del Dd, hgd

    # ------------------------------------------------------------------ boots (1.2 mm, boot-local)
    for side in "RL":
        log(f"boot {side}")
        bg = S.Grid((-0.11, -0.075, -0.002), (0.225, 0.075, 0.275), 0.0012)
        bg.apply(A.boot_upper, bg.lo, bg.lo + (bg.n - 1) * bg.h, op="set", margin=0)
        Db = bg.d.copy()
        M = A.boot_matrix(side)
        lat = 1 if side == "L" else -1
        tw = lambda vf: (xform(vf[0], M), vf[1])
        full = (bg.lo, bg.lo + (bg.n - 1) * bg.h)
        put(f"Boot_Upper_{side}", tw(S.extract(Db, bg.lo, bg.h)), "leather_dark", "boots")
        sole = S.Grid((-0.11, -0.075, -0.002), (0.225, 0.075, 0.06), 0.0012)
        sole.apply(A.boot_sole, sole.lo, sole.lo + (sole.n - 1) * sole.h, op="set", margin=0)
        Dso = sole.d
        put(f"Boot_Outsole_{side}", tw(S.extract_sub(sole, lambda X, Y, Z, sl: np.maximum(Dso[sl], Z - 0.012),
                                                     sole.lo, sole.lo + (sole.n - 1) * sole.h, 0.0)), "rubber", "boots")
        put(f"Boot_Midsole_{side}", tw(S.extract_sub(sole, lambda X, Y, Z, sl: np.maximum(Dso[sl], 0.012 - Z),
                                                     sole.lo, sole.lo + (sole.n - 1) * sole.h, 0.0)), "orange_bright", "boots")
        put(f"Boot_ToeCap_{side}", tw(S.extract_sub(bg, lambda X, Y, Z, sl: np.maximum(A.boot_toecap(X, Y, Z), -(Db[sl] + 0.0005)),
                                                    *full, 0.0)), "rubber", "boots")

        def bpanel(X, Y, Z, sl):
            r = np.minimum(S.ellipsoid(X, Y, Z, (0.040, 0.0, 0.062), (0.070, 0.080, 0.022), S.rot_xyz(0, -12, 0)),
                           S.ellipsoid(X, Y, Z, (-0.040, 0.0, 0.080), (0.030, 0.080, 0.050)))
            r = np.maximum(r, -Y * lat - 0.0) if False else r
            return np.maximum(S.shell(Db[sl], 0.0, 0.0035), r)

        put(f"Boot_Panels_Orange_{side}", tw(S.extract_sub(bg, bpanel, *full, 0.0)), "orange_leather", "boots")

        def bstraps(X, Y, Z, sl):
            band = np.minimum(np.abs(Z - 0.175) - 0.009, np.abs(Z - 0.222) - 0.009)
            return np.maximum(S.shell(Db[sl], 0.0005, 0.0045), np.maximum(band, X - 0.03))

        put(f"Boot_Straps_{side}", tw(S.extract_sub(bg, bstraps, *full, 0.0)), "leather_dark2", "boots")
        put(f"Boot_Collar_{side}", tw(S.extract_sub(
            bg, lambda X, Y, Z, sl: S.torus(X, Y, Z, (-0.002, 0.0, 0.250), 0.050, 0.010),
            (-0.07, -0.07, 0.22), (0.07, 0.07, 0.275))), "orange_fabric", "boots")
        del Db, Dso, bg, sole

    np.savez_compressed(RAW, **{f"{k}__v": v for k, (v, f, m, c) in meshes.items()},
                        **{f"{k}__f": f for k, (v, f, m, c) in meshes.items()},
                        meta=np.array([[k, m, c] for k, (v, f, m, c) in meshes.items()]))
    log(f"phase A done: {len(meshes)} meshes cached -> {RAW.relative_to(ROOT)}")


# =============================================================================== PHASE B
def phase_b():
    import bpy
    from mathutils import Vector, Matrix
    from charlib import bl

    log("phase B: Blender assembly")
    sc = bl.reset_scene()
    C = {k: bl.collection(n) for k, n in [
        ("root", "CHARACTER")]}
    for k, n in [("body", "CHAR_Body"), ("head", "CHAR_Head_Face"), ("hair", "CHAR_Hair"),
                 ("clothing", "CHAR_Clothing"), ("armour", "CHAR_Armour"), ("accessories", "CHAR_Accessories"),
                 ("hands", "CHAR_Hands_Gloves"), ("boots", "CHAR_Boots"), ("backpack", "CHAR_Backpack")]:
        C[k] = bl.collection(n, C["root"])
    C["pedestal"] = bl.collection("PEDESTAL")

    # ------------------------------------------------------------------ materials
    MAT = {
        "skin": bl.pbr("M_Skin", (1, 1, 1, 1), rough=0.5, albedo_map="skin_albedo.png", normal_map="skin_normal.png",
                       normal_strength=0.35, rough_map="skin_roughness.png", sss=0.15, sss_radius=(1.0, 0.45, 0.3)),
        "lips": bl.pbr("M_Lips", "#A2564E", rough=0.38, normal_map="skin_normal.png", normal_strength=0.3, sss=0.25),
        "suit": bl.pbr("M_Bodysuit_Graphite", "#303238", rough=0.42, albedo_map="leather_albedo_mask.png",
                       albedo_is_mask=True, normal_map="leather_normal.png", normal_strength=0.45,
                       rough_map="leather_roughness.png", sheen=0.15),
        "jacket": bl.pbr("M_Jacket_Grey", "#85888D", rough=0.85, albedo_map="fabric_albedo_mask.png",
                         albedo_is_mask=True, normal_map="fabric_normal.png", normal_strength=0.7,
                         rough_map="fabric_roughness.png", sheen=0.05),
        "jacket_inner": bl.pbr("M_Jacket_Cuff", "#9C9FA3", rough=0.85, albedo_map="fabric_albedo_mask.png",
                               albedo_is_mask=True, normal_map="fabric_normal.png", normal_strength=0.8, sheen=0.05),
        "hood": bl.pbr("M_Hood_Orange", "#D9661C", rough=0.9, albedo_map="fabric_albedo_mask.png", albedo_is_mask=True,
                       normal_map="fabric_normal.png", normal_strength=1.0, sheen=0.08),
        "orange_fabric": bl.pbr("M_Orange_Fabric", "#F07824", rough=0.8, albedo_map="fabric_albedo_mask.png",
                                albedo_is_mask=True, normal_map="fabric_normal.png", normal_strength=0.8, sheen=0.05),
        "orange_leather": bl.pbr("M_Orange_Leather", "#F07824", rough=0.5, albedo_map="leather_albedo_mask.png",
                                 albedo_is_mask=True, normal_map="leather_normal.png", normal_strength=0.4,
                                 rough_map="leather_roughness.png"),
        "orange_bright": bl.pbr("M_Orange_Bright", "#FF761A", rough=0.55, normal_map="leather_normal.png",
                                normal_strength=0.3),
        "leather_dark": bl.pbr("M_Leather_Charcoal", "#202329", rough=0.5, albedo_map="leather_albedo_mask.png",
                               albedo_is_mask=True, normal_map="leather_normal.png", normal_strength=0.55,
                               rough_map="leather_roughness.png"),
        "leather_dark2": bl.pbr("M_Leather_Graphite", "#2A2C31", rough=0.55, albedo_map="leather_albedo_mask.png",
                                albedo_is_mask=True, normal_map="leather_normal.png", normal_strength=0.55),
        "leather_brown": bl.pbr("M_Leather_Brown", "#5B4232", rough=0.6, albedo_map="leather_albedo_mask.png",
                                albedo_is_mask=True, normal_map="leather_normal.png", normal_strength=0.7,
                                rough_map="leather_roughness.png"),
        "metal": bl.pbr("M_Metal_Brushed", "#8E9298", rough=0.48, metal=0.85, normal_map="metal_normal.png",
                        normal_strength=0.3, rough_map="metal_roughness.png"),
        "metal_dark": bl.pbr("M_Metal_Gunmetal", "#45484E", rough=0.38, metal=1.0, normal_map="metal_normal.png",
                             normal_strength=0.3, rough_map="metal_roughness.png"),
        "plastic_dark": bl.pbr("M_HardPlastic_Charcoal", "#26282D", rough=0.4, normal_map="metal_normal.png",
                               normal_strength=0.15, coat=0.2),
        "rubber": bl.pbr("M_Rubber_Black", "#17181A", rough=0.8, normal_map="leather_normal.png", normal_strength=0.3),
        "cyan": bl.pbr("M_Cyan_LED", "#00DDF5", rough=0.2, emission="#00DDF5", emission_strength=8.0),
        "hair": bl.pbr("M_Hair_DarkBrown", (1, 1, 1, 1), rough=0.42, albedo_map="hair_albedo.png",
                       normal_map="hair_normal.png", normal_strength=0.8, aniso=0.6),
        "hair_strand": bl.pbr("M_Hair_Strands", "#2A1A12", rough=0.38, aniso=0.7),
        "brow": bl.pbr("M_Eyebrow", "#1E140F", rough=0.5),
        "eye": bl.pbr("M_Eye", (1, 1, 1, 1), rough=0.05, albedo_map="eye_iris.png", coat=1.0, sss=0.05),
        "lens": bl.pbr("M_Goggle_Lens", "#FF7A1A", rough=0.05, coat=1.0, transmission=0.35, spec=0.8),
        "rock": bl.pbr("M_Rock_Pedestal", (1, 1, 1, 1), rough=0.8, albedo_map="rock_albedo.png",
                       normal_map="rock_normal.png", normal_strength=1.2, rough_map="rock_roughness.png"),
        "jewel": bl.pbr("M_Jewellery_Silver", "#C9CCD2", rough=0.2, metal=1.0),
    }
    MAT["eye"].node_tree.nodes["Image Texture"].extension = "EXTEND"

    # ------------------------------------------------------------------ import raw meshes
    Z = np.load(RAW)
    meta = Z["meta"]
    # target edge lengths (m) for decimation, by collection
    EDGE = {"body": 0.0042, "clothing": 0.0045, "armour": 0.0040, "accessories": 0.0040, "backpack": 0.0045,
            "head": 0.0017, "hair": 0.0030, "hands": 0.0013, "boots": 0.0028}
    OB = {}
    for name, mkey, ckey in meta:
        v, f = Z[f"{name}__v"], Z[f"{name}__f"]
        ob = bl.mesh_from_arrays(name, v, f, C[ckey], MAT[mkey])
        # decimate to a target density
        area = sum(p.area for p in ob.data.polygons)
        target = 2.0 * area / EDGE[ckey] ** 2
        ratio = float(min(1.0, target / len(f)))
        bl.cleanup(ob, smooth_iter=2 if ckey not in ("hands",) else 1, smooth_factor=0.5, decimate=ratio)
        OB[name] = ob
        log(f"  {name}: {len(f)} -> {len(ob.data.polygons)} faces")

    # cloth wrinkles on the jacket and hood (real displacement, applied)
    for nm, strength, size in (("Jacket_Grey", 0.0022, 0.035), ("Hood_Orange", 0.004, 0.03)):
        tex = bpy.data.textures.new(nm + "_folds", "STUCCI")
        tex.noise_scale = size
        tex.turbulence = 3.0
        bl.add_mod(OB[nm], "DISPLACE", texture=tex, strength=strength, mid_level=0.5, texture_coords="GLOBAL")
        bl.apply_mods(OB[nm])

    # ------------------------------------------------------------------ material regions
    body = OB["Body_Bodysuit"]
    fc = bl.face_centers(body)
    skin_mask = np.zeros(len(fc), bool)
    for side in "RL":
        Sh, E, Wr = A.arm_pts(side)
        Cf = lerp(E, Wr, 0.30)
        u = (Wr - E) / np.linalg.norm(Wr - E)
        t = (fc - Cf) @ u
        dist = np.linalg.norm((fc - Cf) - np.outer(t, u), axis=1)
        skin_mask |= (t > -0.01) & (dist < 0.046)  # bare forearm between rolled cuff and glove
    bl.assign_by_mask(body, skin_mask, MAT["skin"])
    # orange accent stripes on the bodysuit (vertical side piping bands on the torso)
    accent = (np.abs(np.abs(fc[:, 0]) - 0.115) < 0.006) & (fc[:, 1] < -0.04) & (fc[:, 2] > 1.0) & (fc[:, 2] < 1.17)
    bl.assign_by_mask(body, accent, MAT["orange_leather"])

    head = OB["Head_Skin"]
    fc = bl.face_centers(head)
    loc = (fc - SK["head_c"]) @ A.HEAD_R  # world -> head-local
    lips = ((((loc[:, 0] / 0.0235) ** 2 + ((loc[:, 1] + 0.088) / 0.014) ** 2 + ((loc[:, 2] + 0.0556) / 0.0128) ** 2) < 1.0)
            & (loc[:, 1] < -0.080))
    bl.assign_by_mask(head, lips, MAT["lips"])

    # ------------------------------------------------------------------ UVs (consistent texel density)
    for name, ob in OB.items():
        tile = 0.06 if ob.users_collection[0].name in ("CHAR_Head_Face", "CHAR_Hands_Gloves") else 0.20
        bl.world_uv(ob, tile)
    log("UVs done")

    # ------------------------------------------------------------------ eyes
    for s in (-1, 1):
        p = A.head_to_world(A.EYE_LOCAL[s])
        eye = bl.uvsphere(f"Eye_{'L' if s > 0 else 'R'}", C["head"], A.EYE_R, tuple(p), mat=MAT["eye"], seg=48, rings=32)
        me = eye.data
        uv = me.uv_layers.new(name="UVMap")
        Rinv = A.HEAD_R.T
        S_ = 0.0122
        for poly in me.polygons:
            for li in poly.loop_indices:
                co = np.array(me.vertices[me.loops[li].vertex_index].co)
                # sphere created unrotated in world; express in head-local to face forward
                l = Rinv @ co
                uv.data[li].uv = (l[0] / S_ + 0.5, l[2] / S_ + 0.5) if l[1] < 0 else (0.0, 0.0)

    # ------------------------------------------------------------------ strands: hair, brows, lashes
    rng = np.random.default_rng(7)
    hc = SK["head_c"]
    Rh = A.HEAD_R
    W = lambda p: hc + Rh @ np.asarray(p, float)
    cap_tree = bl.bvh(OB["Hair_Cap"])
    bun_tree = bl.bvh(OB["Hair_Bun"])
    head_tree = bl.bvh(head)
    bun_c = W((0.0, 0.030, 0.136))

    # combed scalp strands: from points on the cap toward the bun base
    capv = np.array([v.co for v in OB["Hair_Cap"].data.vertices])
    capn = np.array([v.normal for v in OB["Hair_Cap"].data.vertices])
    loc_c = (capv - hc) @ Rh
    sel = np.where((loc_c[:, 2] < 0.10) & (((capv - hc) * capn).sum(1) > 0))[0]
    starts = capv[rng.choice(sel, size=min(1800, len(sel)), replace=False)]
    lines = []
    for p0 in starts:
        pts = [Vector(p0)]
        for i in range(12):
            p = pts[-1]
            to = Vector(bun_c) - p
            if to.length < 0.03:
                break
            step = to.normalized() * min(0.012, to.length * 0.35)
            q = p + step + Vector(rng.normal(0, 0.0012, 3))
            hit = cap_tree.find_nearest(q)
            if hit[0] is None:
                break
            pts.append(hit[0] + hit[1] * (0.0006 + 0.0006 * rng.random()))
        if len(pts) >= 3:
            lines.append([tuple(p) for p in pts])
    bl.strands_object("Hair_Strands_Scalp", C["hair"], lines, 0.00045, MAT["hair_strand"])

    # bun wraps
    lines = []
    for i in range(420):
        ax = Vector(rng.normal(0, 1, 3)).normalized()
        ax.z = abs(ax.z) + 0.6
        ax.normalize()
        u = ax.orthogonal().normalized()
        w = ax.cross(u)
        r = 0.052 + 0.014 * rng.random()
        a0 = rng.random() * 2 * np.pi
        span = 1.2 + rng.random() * 2.5
        pts = []
        for a in np.linspace(a0, a0 + span, 14):
            q = Vector(bun_c) + Vector((0, 0, 0.004)) + (u * math.cos(a) + w * math.sin(a)) * r
            hit = bun_tree.find_nearest(q)
            if hit[0] is not None:
                pts.append(tuple(hit[0] + hit[1] * (0.0008 + 0.002 * rng.random())))
        if len(pts) > 3:
            lines.append(pts)
    # messy loose loops sticking out of the bun
    for i in range(14):
        d = Vector(rng.normal(0, 1, 3)); d.z = abs(d.z) + 0.3; d.normalize()
        base = bun_tree.find_nearest(Vector(bun_c) + d * 0.065)[0]
        side = d.orthogonal().normalized()
        L = 0.025 + 0.035 * rng.random()
        pts = [tuple(base + d * L * t * 0.6 + side * math.sin(t * math.pi) * 0.018
                     + Vector((0, 0, -0.01 * t * t))) for t in np.linspace(0, 1, 8)]
        lines.append(pts)
    bl.strands_object("Hair_Strands_Bun", C["hair"], lines, 0.0005, MAT["hair_strand"])

    # loose face-framing strands (temples / in front of the ears)
    lines = []
    for s in (-1, 1):
        for i in range(10):
            a = rng.random()
            start = W((s * (0.050 + 0.016 * a), -0.050 - 0.020 * rng.random(), 0.058 - 0.012 * a))
            L = 0.07 + 0.08 * rng.random()
            pts = []
            for t in np.linspace(0, 1, 12):
                p = start + Rh @ np.array([s * (0.010 * t + 0.003 * math.sin(t * 4 + i)), -0.010 * t,
                                           -L * t + 0.002 * math.sin(t * 5 + i)])
                hit = head_tree.find_nearest(Vector(p))
                q = Vector(p)
                if hit[0] is not None and (q - hit[0]).length < 0.003:
                    q = hit[0] + hit[1] * 0.003
                pts.append(tuple(q))
            lines.append(pts)
    bl.strands_object("Hair_Strands_Loose", C["hair"], lines, 0.00055, MAT["hair_strand"])

    # eyebrows
    lines = []
    for s in (-1, 1):
        arc = [np.array((s * 0.012, -0.0905, 0.0255)), np.array((s * 0.030, -0.088, 0.0325)),
               np.array((s * 0.049, -0.074, 0.0285))]
        for i in range(110):
            t = rng.random()
            p = (1 - t) ** 2 * arc[0] + 2 * (1 - t) * t * arc[1] + t ** 2 * arc[2]
            p = p + np.array([0, 0, (rng.random() - 0.5) * 0.004 * (1.2 - t)])
            tang = (arc[2] - arc[0]); tang /= np.linalg.norm(tang)
            dirv = tang * 0.8 + np.array([0, 0, 0.6 if t < 0.25 else 0.15])
            dirv /= np.linalg.norm(dirv)
            L = 0.006 + 0.004 * rng.random()
            raw = [W(p + dirv * L * k) for k in np.linspace(0, 1, 4)]
            lines.append(bl.project_polyline(head_tree, raw, lift=0.0004 + 0.0004 * np.linspace(0, 1, 4).mean()))
    bl.strands_object("Eyebrows", C["head"], lines, 0.00022, MAT["brow"])

    # upper lashes
    lines = []
    for s in (-1, 1):
        ec = A.EYE_LOCAL[s]
        for i in range(70):
            x = (rng.random() * 2 - 1) * 0.0105
            z = 0.0031 + 0.0012 * (1 - (x / 0.0105) ** 2)
            y = -math.sqrt(max(0.01325 ** 2 - x * x - z * z, 0))
            p = ec + np.array([x, y, z])
            L = 0.006 + 0.003 * (1 - abs(x) / 0.0105) + (0.002 if s * x > 0 else 0)
            pts = [W(p), W(p + np.array([s * 0.0006, -L * 0.55, L * 0.25])), W(p + np.array([s * 0.001, -L * 0.8, L * 0.7]))]
            lines.append([tuple(q) for q in pts])
    bl.strands_object("Eyelashes", C["head"], lines, 0.00011, MAT["brow"])
    log("strands done")

    # ------------------------------------------------------------------ goggles, earrings, choker
    for s in (-1, 1):
        p = W((s * 0.031, -0.087, 0.080))
        lens = bl.uvsphere(f"Goggles_Lens_{'L' if s > 0 else 'R'}", C["accessories"], 1.0, tuple(p),
                           scale=(0.025, 0.011, 0.0165), rot=(-28, 0, 6 + s * 22), mat=MAT["lens"], seg=40, rings=20)
        rim = bl.torus_obj(f"Goggles_Frame_{'L' if s > 0 else 'R'}", C["accessories"], 1.0, 0.13, tuple(p),
                           rot=(62, 0, 6 + s * 22), mat=MAT["plastic_dark"], major=40, minor=10)
        rim.scale = (0.026, 0.0175, 0.026)
    bl.box("Goggles_Bridge", C["accessories"], (0.016, 0.008, 0.008), tuple(W((0, -0.094, 0.079))),
           rot=(-28, 0, 6), mat=MAT["plastic_dark"], bevel=0.002)
    for s in (-1, 1):
        bl.box(f"Goggles_Hinge_{'L' if s > 0 else 'R'}", C["accessories"], (0.010, 0.016, 0.012),
               tuple(W((s * 0.062, -0.066, 0.074))), rot=(-15, 0, 6 + s * 55), mat=MAT["orange_bright"], bevel=0.002)
        bl.torus_obj(f"Earring_{'L' if s > 0 else 'R'}", C["accessories"], 0.0045, 0.0007,
                     tuple(W((s * 0.0735, -0.001, -0.026))), rot=(0, 90, 0), mat=MAT["jewel"], major=24, minor=6)
    ch = bl.torus_obj("Choker", C["accessories"], 1.0, 0.035, tuple(W((0, 0.004, -0.122))), rot=(8, 0, 6),
                      mat=MAT["leather_dark"], major=64, minor=8)
    ch.scale = (0.0545, 0.0525, 0.0545)
    bl.box("Choker_Tag", C["accessories"], (0.010, 0.003, 0.012), tuple(W((0, -0.054, -0.128))),
           rot=(10, 0, 6), mat=MAT["jewel"], bevel=0.001)

    # ------------------------------------------------------------------ backpack (hard surface)
    bp = C["backpack"]
    bl.box("Pack_Housing", bp, (0.20, 0.075, 0.23), (0, 0.142, 1.315), rot=(-6, 0, 0), mat=MAT["plastic_dark"], bevel=0.014)
    bl.box("Pack_Yoke", bp, (0.30, 0.062, 0.060), (0, 0.128, 1.420), rot=(-12, 0, 0), mat=MAT["leather_dark2"], bevel=0.012)
    for s in (-1, 1):
        bl.box(f"Pack_Wing_{'L' if s > 0 else 'R'}", bp, (0.070, 0.065, 0.150), (s * 0.118, 0.128, 1.330),
               rot=(-6, 0, s * 18), mat=MAT["leather_dark2"], bevel=0.010)
        bl.box(f"Pack_Wing_Panel_{'L' if s > 0 else 'R'}", bp, (0.040, 0.010, 0.095), (s * 0.130, 0.162, 1.335),
               rot=(-6, 0, s * 18), mat=MAT["orange_leather"], bevel=0.003)
        bl.box(f"Pack_Wing_LED_{'L' if s > 0 else 'R'}", bp, (0.006, 0.028, 0.050), (s * 0.158, 0.120, 1.345),
               rot=(-6, 0, s * 18), mat=MAT["cyan"], bevel=0.002)
        for k in range(2):
            bl.box(f"Pack_Vent_{s}_{k}", bp, (0.030, 0.006, 0.006), (s * 0.045, 0.181, 1.395 - k * 0.014),
                   rot=(-6, 0, 0), mat=MAT["rubber"], bevel=0.002)
        bl.box(f"Pack_LED_Upper_{s}", bp, (0.007, 0.006, 0.045), (s * 0.030, 0.183, 1.335), rot=(-6, 0, 0),
               mat=MAT["cyan"], bevel=0.002)
        for k, z in enumerate((1.262, 1.240)):
            bl.box(f"Pack_Chevron_{s}_{k}", bp, (0.050, 0.006, 0.0075), (s * 0.023, 0.184, z), rot=(-6, s * 28, 0),
                   mat=MAT["cyan"], bevel=0.002)
        for z in (1.21, 1.42):
            bl.cylinder(f"Pack_Screw_{s}_{z}", bp, 0.0045, 0.006, (s * 0.085, 0.178 if z < 1.3 else 0.160, z),
                        rot=(90, 0, 0), mat=MAT["metal"], verts=16)
    bl.box("Pack_CenterPlate", bp, (0.120, 0.012, 0.160), (0, 0.176, 1.310), rot=(-6, 0, 0), mat=MAT["leather_dark"], bevel=0.006)
    bl.box("Pack_Orange_V", bp, (0.090, 0.008, 0.024), (0, 0.181, 1.215), rot=(-6, 0, 0), mat=MAT["orange_leather"], bevel=0.004)
    # LED badges on the front harness straps
    for s in (-1, 1):
        bl.box(f"Strap_LED_{s}", bp, (0.012, 0.006, 0.030), (s * 0.104, -0.112, 1.33), rot=(12, 0, 0),
               mat=MAT["cyan"], bevel=0.002)
        bl.box(f"Strap_Buckle_{s}", bp, (0.030, 0.008, 0.020), (s * 0.100, -0.112, 1.255), rot=(8, 0, 0),
               mat=MAT["metal_dark"], bevel=0.002)

    # ------------------------------------------------------------------ pouches, buckles
    ac = C["accessories"]

    def pouch(name, size, loc, rot, led=True):
        bl.box(name, ac, size, loc, rot=rot, mat=MAT["leather_brown"], bevel=0.007)
        sx, sy, sz = size
        R = Matrix.Rotation(math.radians(rot[2]), 3, "Z") @ Matrix.Rotation(math.radians(rot[1]), 3, "Y") @ \
            Matrix.Rotation(math.radians(rot[0]), 3, "X")
        off = lambda v: tuple(Vector(loc) + R @ Vector(v))
        bl.box(name + "_Flap", ac, (sx + 0.004, sy + 0.006, sz * 0.32), off((0, -0.002, sz * 0.36)), rot=rot,
               mat=MAT["leather_brown"], bevel=0.005)
        bl.box(name + "_Clasp", ac, (0.012, 0.005, 0.016), off((0, -sy / 2 - 0.004, sz * 0.22)), rot=rot,
               mat=MAT["metal"], bevel=0.0015)
        if led:
            bl.box(name + "_LED", ac, (0.006, 0.004, sz * 0.40), off((sx * 0.30, -sy / 2 - 0.002, -sz * 0.08)),
                   rot=rot, mat=MAT["cyan"], bevel=0.0015)

    pouch("Pouch_Thigh_R", (0.050, 0.040, 0.115), (-0.186, -0.040, 0.655), (0, 4, -28))
    pouch("Pouch_Thigh_R2", (0.040, 0.030, 0.080), (-0.150, -0.088, 0.665), (0, 0, -12), led=False)
    pouch("Pouch_Hip_R", (0.042, 0.050, 0.070), (-0.188, 0.005, 0.905), (0, 6, -10), led=False)
    pouch("Pouch_Hip_L", (0.045, 0.028, 0.050), (0.135, -0.090, 0.915), (0, 0, 22), led=False)
    bl.box("Belt_Buckle", ac, (0.055, 0.012, 0.038), (0, -0.098, 0.948), mat=MAT["metal_dark"], bevel=0.003)
    bl.box("Belt_Buckle_Inset", ac, (0.036, 0.006, 0.020), (0, -0.105, 0.948), mat=MAT["orange_bright"], bevel=0.002)
    for s in (-1, 1):
        bl.box(f"Belt_Ring_{s}", ac, (0.020, 0.010, 0.026), (s * 0.150, -0.075, 0.915), rot=(0, 0, s * 35),
               mat=MAT["metal"], bevel=0.002)

    # boots: ankle discs + eyelets + laces
    for side in "RL":
        M = A.boot_matrix(side)
        Mw = Matrix([list(r) for r in M])
        bc = C["boots"]
        for sgn in (-1, 1):
            p = Mw @ Vector((0.0, sgn * 0.056, 0.095))
            yaw = math.degrees(math.atan2(M[1, 0], M[0, 0]))
            bl.cylinder(f"Boot_AnkleDisc_{side}_{sgn}", bc, 0.022, 0.008, tuple(p), rot=(90, 0, yaw),
                        mat=MAT["metal_dark"], verts=32)
            bl.torus_obj(f"Boot_AnkleRing_{side}_{sgn}", bc, 0.022, 0.0028, tuple(p), rot=(90, 0, yaw),
                         mat=MAT["orange_bright"], major=32, minor=8)
        # lace zig-zag on the instep / shaft front
        up_tree = bl.bvh(OB[f"Boot_Upper_{side}"])
        rows = [(0.105, 0.060), (0.085, 0.074), (0.065, 0.090), (0.048, 0.108), (0.040, 0.128),
                (0.040, 0.150), (0.042, 0.172), (0.044, 0.196), (0.046, 0.220)]
        lace = []
        eyel = []
        for i, (x, z) in enumerate(rows):
            for sgn in ((-1, 1) if i % 2 == 0 else (1, -1)):
                q = Mw @ Vector((x + 0.03, sgn * 0.018, z))
                hit = up_tree.find_nearest(q)
                p = hit[0] + hit[1] * 0.002
                lace.append(tuple(p))
                eyel.append((p, hit[1]))
        bl.strands_object(f"Boot_Laces_{side}", bc, [lace], 0.0016, MAT["orange_bright"], radius_taper=False, resolution=2)
        for k, (p, n) in enumerate(eyel):
            e = bl.torus_obj(f"Boot_Eyelet_{side}_{k}", bc, 0.0028, 0.0009, tuple(p), mat=MAT["metal"], major=12, minor=6)
            e.rotation_mode = "QUATERNION"
            e.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(n)

    # orange piping lines projected onto the bodysuit
    body_tree = bl.bvh(body)
    piping = []
    for s in (-1, 1):
        for path in ([(s * 0.172, -0.01, 0.95), (s * 0.165, -0.03, 0.84), (s * 0.150, -0.045, 0.72), (s * 0.130, -0.055, 0.60), (s * 0.120, -0.050, 0.54)],
                     [(s * 0.060, -0.090, 0.86), (s * 0.090, -0.095, 0.76), (s * 0.115, -0.080, 0.66), (s * 0.122, -0.060, 0.56)],
                     [(s * 0.070, -0.080, 0.42), (s * 0.095, -0.060, 0.33), (s * 0.110, -0.050, 0.25)],
                     [(s * 0.140, 0.040, 0.42), (s * 0.130, 0.060, 0.32), (s * 0.115, 0.055, 0.24)],
                     [(s * 0.085, 0.100, 0.86), (s * 0.130, 0.080, 0.80), (s * 0.160, 0.040, 0.76)]):
            pts = S.catmull([np.array(p) for p in path], 10)
            piping.append(bl.project_polyline(body_tree, [tuple(p) for p in pts], lift=0.0012))
    bl.strands_object("Piping_Orange", C["armour"], piping, 0.0016, MAT["orange_bright"], radius_taper=False, resolution=2)

    # ------------------------------------------------------------------ pedestal (faceted rock)
    import bmesh
    from mathutils import noise
    bm = bmesh.new()
    NA, NZ = 256, 26
    ring = []
    for j in range(NZ + 1):
        t = j / NZ
        z = -0.185 + 0.185 * t
        rb = 0.330 - 0.025 * t - 0.03 * max(0, t - 0.8) / 0.2
        ring.append([bm.verts.new((rb * math.cos(2 * math.pi * i / NA), rb * math.sin(2 * math.pi * i / NA), z)) for i in range(NA)])
    rad_top = 0.330 - 0.025 - 0.03
    for k in range(1, 22):
        r = rad_top * (1 - k / 22)
        if r < 0.004:
            break
        ring.append([bm.verts.new((r * math.cos(2 * math.pi * i / NA), r * math.sin(2 * math.pi * i / NA), 0.0)) for i in range(NA)])
    for j in range(len(ring) - 1):
        for i in range(NA):
            a, b = ring[j][i], ring[j][(i + 1) % NA]
            c2, d2 = ring[j + 1][(i + 1) % NA], ring[j + 1][i]
            bm.faces.new((a, b, c2, d2))
    bm.faces.new(ring[-1][::-1])
    bm.faces.new(ring[0])
    for vtx in bm.verts:
        p = vtx.co.copy()
        r = math.hypot(p.x, p.y)
        ang = math.atan2(p.y, p.x)
        q = Vector((p.x, p.y, p.z)) * 9.0
        dist, pts = noise.voronoi(q, distance_metric="DISTANCE")[:2] if hasattr(noise, "voronoi") else ((0, 0), [q])
        cell = pts[0]
        h = (math.sin(cell.x * 12.9898 + cell.y * 78.233 + cell.z * 37.719) * 43758.5453) % 1.0
        gx = ((math.sin(cell.x * 93.9 + cell.y * 67.3) * 15731.0) % 1.0) - 0.5
        gz = ((math.sin(cell.z * 41.1 + cell.x * 27.7) * 789221.0) % 1.0) - 0.5
        facet = 0.014 * h + 0.16 * (gx * (p.x - cell.x / 9.0) + gz * (p.z - cell.z / 9.0))
        crack = -0.006 * max(0.0, 1.0 - (dist[1] - dist[0]) * 12.0)
        outline = 0.035 * noise.noise(Vector((math.cos(ang) * 2.2, math.sin(ang) * 2.2, 0.3)))
        top = p.z > -0.001
        edge_w = min(1.0, max(0.0, (r - 0.20) / 0.07))
        if top:
            vtx.co.z += (facet * 0.5 + crack) * edge_w + 0.002 * noise.noise(q * 0.5) - 0.006 * edge_w ** 2
            vtx.co.x *= 1 + outline
            vtx.co.y *= 1 + outline
        else:
            k = 1 + outline + facet * 3.0 + crack * 2.0
            vtx.co.x *= k
            vtx.co.y *= k
            vtx.co.z += 0.004 * noise.noise(q)
    me = bpy.data.meshes.new("Pedestal_Rock")
    bm.to_mesh(me)
    bm.free()
    ped = bpy.data.objects.new("Pedestal_Rock", me)
    C["pedestal"].objects.link(ped)
    me.materials.append(MAT["rock"])
    bl.recalc_normals(ped)
    bl.add_mod(ped, "SUBSURF", levels=1, render_levels=1)
    bl.apply_mods(ped)
    me.shade_smooth()
    bl.add_mod(ped, "WEIGHTED_NORMAL", keep_sharp=True)
    bl.world_uv(ped, 0.35)

    # ------------------------------------------------------------------ finish
    # convert strand curves to meshes so every export format carries them
    for ob in [o for o in bpy.data.objects if o.type == "CURVE"]:
        bl.activate(ob)
        bpy.ops.object.convert(target="MESH")
    # apply the bevel / weighted-normal modifiers of hard-surface parts
    for ob in [o for o in bpy.data.objects if o.type == "MESH" and o.modifiers]:
        bl.apply_mods(ob)
    for ob in [o for o in bpy.data.objects if o.type == "MESH" and not o.data.uv_layers]:
        bl.world_uv(ob, 0.20)

    sc.render.engine = "CYCLES"
    out = BLEND_DIR / "character_model.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True, relative_remap=True)
    tris = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
    log(f"saved {out.relative_to(ROOT)}  objects={len(bpy.data.objects)}  faces={tris}")


if __name__ == "__main__":
    if "--reuse" not in sys.argv or not RAW.exists():
        phase_a()
    phase_b()
