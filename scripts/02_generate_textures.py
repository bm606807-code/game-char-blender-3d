"""STAGE 6a - Generate tileable PBR texture maps (albedo / normal / roughness) with NumPy.

All maps are periodic (FFT-filtered noise + periodic patterns) so they tile without seams.
They are written to assets/textures/ and referenced by the Blender materials, so the same
maps travel with the GLB / OBJ+MTL exports.

Usage: python3 scripts/02_generate_textures.py
"""
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path("assets/textures")
OUT.mkdir(parents=True, exist_ok=True)
N = 1024
rng = np.random.default_rng(1751)


def noise(scale_px, n=N, octaves=1, persistence=0.5, seed=None):
    """Periodic band-limited noise in [0,1]; scale_px = feature size in pixels."""
    g = np.random.default_rng(seed) if seed is not None else rng
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    out = np.zeros((n, n))
    amp = 1.0
    s = scale_px
    for _ in range(octaves):
        spec = np.fft.fft2(g.standard_normal((n, n)))
        filt = np.exp(-(f * s) ** 2 * 4.0)
        layer = np.real(np.fft.ifft2(spec * filt))
        layer /= layer.std() + 1e-9
        out += amp * layer
        amp *= persistence
        s /= 2.0
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out


def worley(cells, n=N, seed=0):
    """Periodic F2-F1 Worley noise -> crisp cell edges (rock facets)."""
    from scipy.spatial import cKDTree
    g = np.random.default_rng(seed)
    pts = g.random((cells, 2)) * n
    tree = cKDTree(pts, boxsize=n)  # periodic box -> seamless tiling
    yy, xx = np.mgrid[0:n, 0:n]
    q = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64)
    d, _ = tree.query(q, k=2, workers=-1)
    return d[:, 0].reshape(n, n), d[:, 1].reshape(n, n)


def normal_from_height(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    nz = np.ones_like(h)
    l = np.sqrt(dx ** 2 + dy ** 2 + nz ** 2)
    # OpenGL convention (+Y up), as used by Blender and glTF
    nrm = np.dstack([-dx / l, dy / l, nz / l])
    return ((nrm * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def save_gray(a, name):
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L").save(OUT / name)


def save_rgb(a, name):
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB").save(OUT / name)


def tint(gray, c0, c1):
    c0 = np.array(c0, float) / 255
    c1 = np.array(c1, float) / 255
    return c0 + (c1 - c0) * gray[..., None]


yy, xx = np.mgrid[0:N, 0:N] / N

# ---------------------------------------------------------------- fabric (twill weave)
period = 64  # threads per tile
t = 2 * np.pi * period
warp = np.sin(t * (xx + yy * 0.5)) * 0.5 + 0.5
weft = np.sin(t * (yy - xx * 0.5)) * 0.5 + 0.5
weave = 0.6 * warp * (np.sin(t * yy / 2) > 0) + 0.6 * weft * (np.sin(t * yy / 2) <= 0)
fuzz = noise(3, octaves=2)
mott = noise(120, octaves=3)
h_fabric = 0.7 * weave + 0.3 * fuzz
save_rgb(np.dstack([0.82 + 0.18 * (0.6 * h_fabric + 0.4 * mott)] * 3), "fabric_albedo_mask.png")
Image.fromarray(normal_from_height(h_fabric, 2.2)).save(OUT / "fabric_normal.png")
save_gray(0.78 + 0.15 * fuzz - 0.08 * mott, "fabric_roughness.png")

# ---------------------------------------------------------------- leather / synthetic suit grain
grain_d1, grain_d2 = worley(900, seed=3)
grain = np.clip((grain_d2 - grain_d1) / 14.0, 0, 1)
wear = noise(140, octaves=4, seed=7)
creases = noise(18, octaves=2, seed=8)
creases = np.abs(creases - 0.5) * 2
h_leather = 0.55 * grain + 0.25 * (1 - creases) + 0.2 * noise(4, seed=9)
Image.fromarray(normal_from_height(h_leather, 3.0)).save(OUT / "leather_normal.png")
save_gray(0.42 + 0.28 * wear - 0.10 * grain, "leather_roughness.png")
# scuffed variation mask (light = worn edges / dust)
save_rgb(np.dstack([0.80 + 0.20 * wear + 0.06 * (grain - 0.5)] * 3), "leather_albedo_mask.png")

# ---------------------------------------------------------------- skin (pores, subtle blotches)
pores_d1, _ = worley(5000, seed=11)
pores = 1 - np.clip(pores_d1 / 6.0, 0, 1)
blotch = noise(90, octaves=4, seed=12)
fine = noise(2, seed=13)
skin_base = np.array([0.66, 0.50, 0.41])  # warm light-tan (linear-ish sRGB values)
skin_red = np.array([0.64, 0.44, 0.37])
skin_alb = skin_base * (1 - 0.35 * blotch[..., None]) + skin_red * 0.35 * blotch[..., None]
skin_alb *= (0.97 + 0.03 * fine)[..., None]
skin_alb *= (1 - 0.05 * pores)[..., None]
save_rgb(skin_alb, "skin_albedo.png")
Image.fromarray(normal_from_height(0.6 * fine - 0.8 * pores, 1.4)).save(OUT / "skin_normal.png")
save_gray(0.48 + 0.10 * blotch + 0.08 * pores, "skin_roughness.png")

# ---------------------------------------------------------------- rock pedestal (faceted basalt)
r_d1, r_d2 = worley(70, seed=21)
facets = np.clip((r_d2 - r_d1) / 40.0, 0, 1)
r_mid = noise(40, octaves=4, seed=22)
r_fine = noise(3, octaves=2, seed=23)
h_rock = 0.55 * facets + 0.30 * r_mid + 0.15 * r_fine
cracks = (1 - np.clip((r_d2 - r_d1) / 4.0, 0, 1))  # 1 inside crack lines
rock_alb = tint(0.35 * r_mid + 0.25 * r_fine + 0.4 * facets, (24, 24, 27), (78, 74, 72))
rust = np.clip(cracks * 1.2 + (noise(25, seed=24) - 0.62) * 3.0, 0, 1)[..., None]
rock_alb = rock_alb * (1 - rust * 0.85) + np.array([0.45, 0.24, 0.12]) * rust * 0.85
save_rgb(rock_alb, "rock_albedo.png")
Image.fromarray(normal_from_height(h_rock - 0.6 * cracks, 6.0)).save(OUT / "rock_normal.png")
save_gray(0.70 + 0.25 * r_fine - 0.25 * rust[..., 0], "rock_roughness.png")
save_gray(h_rock, "rock_height.png")

# ---------------------------------------------------------------- brushed / scratched metal
brushed = noise(1.5, seed=31)
brushed = np.mean([np.roll(brushed, k, 1) for k in range(0, 40, 2)], 0)  # horizontal streaks
scratch = np.zeros((N, N))
for _ in range(260):
    x0, y0 = rng.random(2) * N
    ang = rng.random() * np.pi
    L = rng.random() * 120 + 20
    for s in np.linspace(0, L, int(L)):
        scratch[int(y0 + s * np.sin(ang)) % N, int(x0 + s * np.cos(ang)) % N] = 1
save_gray(0.28 + 0.25 * (brushed - brushed.mean()) / (brushed.std() * 4 + 1e-9) + 0.25 * scratch
          + 0.15 * noise(120, octaves=3, seed=32), "metal_roughness.png")
Image.fromarray(normal_from_height(brushed * 0.3 - scratch * 0.5, 2.0)).save(OUT / "metal_normal.png")

# ---------------------------------------------------------------- iris (radial, brown-hazel)
cy = cx = N / 2
Y, X = np.mgrid[0:N, 0:N]
r = np.hypot(X - cx, Y - cy) / (N / 2)
th = np.arctan2(Y - cy, X - cx)
fibres = noise(2, seed=41)
radial = 0.5 + 0.5 * np.sin(th * 90 + 6 * fibres)
iris = np.zeros((N, N, 3))
base = np.array([0.16, 0.09, 0.045]) * (0.75 + 0.5 * radial[..., None] * (1 - r[..., None]))
collar = np.exp(-((r - 0.42) / 0.12) ** 2)[..., None] * np.array([0.14, 0.10, 0.04])
iris = base + collar
limbal = np.clip((r - 0.82) / 0.1, 0, 1)[..., None]
iris = iris * (1 - limbal) + np.array([0.04, 0.03, 0.02]) * limbal
pupil = (r < 0.30)[..., None]
iris = np.where(pupil, 0.01, iris)
sclera = (r > 0.95)[..., None]
iris = np.where(sclera, np.array([0.86, 0.83, 0.80]) - 0.04 * noise(30, seed=42)[..., None], iris)
save_rgb(iris, "eye_iris.png")

# ---------------------------------------------------------------- hair strand flow (anisotropic)
strands = noise(1.2, seed=51)
strands = np.mean([np.roll(strands, k, 0) for k in range(0, 60, 3)], 0)  # vertical streaks
strands = (strands - strands.min()) / (strands.max() - strands.min())
hair_alb = tint(0.6 * strands + 0.4 * noise(60, seed=52), (12, 8, 6), (58, 38, 26))
save_rgb(hair_alb, "hair_albedo.png")
Image.fromarray(normal_from_height(strands, 4.0)).save(OUT / "hair_normal.png")

print("textures written:", sorted(p.name for p in OUT.iterdir()))
