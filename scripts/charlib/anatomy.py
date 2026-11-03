"""Character definition: skeleton landmarks + SDF sculpts of body, head, hands and boots.

Canonical reference = view 01 (front): LEFT arm extended forward / slightly raised, palm up;
RIGHT arm hanging; utility pouch on the RIGHT thigh. Character faces -Y, her left is +X.
Height: 1.75 m from boot sole (z=0) to the top of the skull; the hair bun sits above that.
"""
import numpy as np
from . import sdf as S

V = lambda *a: np.array(a, float)

# ----------------------------------------------------------------------------- skeleton (metres)
SK = {
    "head_c": V(0.0, -0.004, 1.632),
    "neck_base": V(0.0, 0.012, 1.455),
    # arms
    "R_sh": V(-0.164, 0.010, 1.394), "R_el": V(-0.214, 0.040, 1.128), "R_wr": V(-0.236, 0.006, 0.898),
    "L_sh": V(0.164, 0.010, 1.394), "L_el": V(0.292, -0.112, 1.212), "L_wr": V(0.372, -0.338, 1.252),
    # legs
    "R_hip": V(-0.088, 0.008, 0.865), "R_kn": V(-0.100, -0.006, 0.492), "R_an": V(-0.112, 0.022, 0.088),
    "L_hip": V(0.088, 0.008, 0.865), "L_kn": V(0.100, -0.006, 0.492), "L_an": V(0.112, 0.022, 0.088),
}
FOOT_YAW = {"R": -9.0, "L": 9.0}  # toes turned slightly outward (deg about Z)

TORSO_TABLE = [
    # z,     hw,    front, back,  yc,    exponent
    (0.780, 0.148, 0.080, 0.084, 0.010, 2.2),
    (0.840, 0.167, 0.085, 0.094, 0.012, 2.3),
    (0.900, 0.176, 0.086, 0.098, 0.012, 2.4),
    (0.960, 0.165, 0.084, 0.088, 0.010, 2.4),
    (1.020, 0.144, 0.080, 0.076, 0.006, 2.3),
    (1.080, 0.127, 0.077, 0.070, 0.004, 2.2),
    (1.140, 0.132, 0.083, 0.074, 0.002, 2.2),
    (1.200, 0.145, 0.090, 0.080, 0.000, 2.3),
    (1.260, 0.151, 0.093, 0.084, 0.000, 2.4),
    (1.320, 0.156, 0.087, 0.088, 0.004, 2.5),
    (1.370, 0.164, 0.074, 0.086, 0.008, 2.7),
    (1.405, 0.150, 0.062, 0.074, 0.012, 2.6),
    (1.440, 0.088, 0.052, 0.058, 0.016, 2.1),
    (1.500, 0.056, 0.047, 0.050, 0.018, 2.0),
]
NECK_TOP = 1.478  # body (bodysuit collar) ends here; the head sculpt carries the skin neck


def arm_pts(side):
    return SK[f"{side}_sh"], SK[f"{side}_el"], SK[f"{side}_wr"]


def leg_pts(side):
    return SK[f"{side}_hip"], SK[f"{side}_kn"], SK[f"{side}_an"]


def lerp(a, b, t):
    return a + (b - a) * t


# ============================================================================= BODY
def body_field(grid):
    g = grid
    # --- torso
    g.apply(lambda X, Y, Z: S.loft_z(X, Y, Z, TORSO_TABLE, 0.79, NECK_TOP, 0.03),
            (-0.2, -0.12, 0.76), (0.2, 0.12, NECK_TOP + 0.01), op="union")
    torso_bits = []
    for s in (-1, 1):
        # glutes, bust, shoulder blades, obliques
        torso_bits += [
            (lambda X, Y, Z, s=s: S.ellipsoid(X, Y, Z, (s * 0.071, 0.066, 0.858), (0.078, 0.062, 0.088),
                                               S.rot_xyz(0, s * 8, 0)), 0.035),
            (lambda X, Y, Z, s=s: S.ellipsoid(X, Y, Z, (s * 0.077, -0.070, 1.252), (0.064, 0.054, 0.058),
                                               S.rot_xyz(-8, 0, s * 14)), 0.03),
            (lambda X, Y, Z, s=s: S.ellipsoid(X, Y, Z, (s * 0.070, 0.074, 1.300), (0.055, 0.022, 0.070)), 0.03),
            (lambda X, Y, Z, s=s: S.round_cone(X, Y, Z, (s * 0.020, -0.042, 1.428), (s * 0.150, -0.018, 1.418),
                                                0.010, 0.012), 0.018),  # clavicle
        ]
    torso_bits.append((lambda X, Y, Z: S.ellipsoid(X, Y, Z, (0, 0.022, 1.438), (0.125, 0.050, 0.038)), 0.035))
    for fn, k in torso_bits:
        g.apply(fn, (-0.25, -0.20, 0.70), (0.25, 0.20, NECK_TOP), op="union", k=k)

    # --- legs
    for side, s in (("R", -1), ("L", 1)):
        H, K, A = leg_pts(side)
        top = H + V(-s * 0.004, 0.0, 0.03)

        def leg(X, Y, Z, H=H, K=K, A=A, top=top, s=s):
            d = S.round_cone(X, Y, Z, top, K, 0.084, 0.054)
            d = S.smin(d, S.round_cone(X, Y, Z, K, A, 0.050, 0.031), 0.02)
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(H, K, 0.15) + V(0, -0.028, 0), lerp(H, K, 0.92) + V(0, -0.03, 0), 0.058, 0.048), 0.03)  # quads
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(H, K, 0.1) + V(s * 0.036, 0.0, 0), lerp(H, K, 0.75) + V(s * 0.03, 0, 0), 0.042, 0.055), 0.03)  # vastus lateralis
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(H, K, 0.1) + V(0, 0.032, 0), lerp(H, K, 0.85) + V(0, 0.03, 0), 0.050, 0.045), 0.03)  # hamstrings
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(H, K, 0.35) + V(-s * 0.040, -0.005, 0), lerp(H, K, 0.95) + V(-s * 0.025, 0, 0), 0.036, 0.040), 0.03)  # adductors
            d = S.smin(d, S.sphere(X, Y, Z, K + V(0, -0.006, 0), 0.043), 0.03)
            d = S.smin(d, S.ellipsoid(X, Y, Z, K + V(0, -0.044, 0.004), (0.023, 0.013, 0.027)), 0.012)  # patella
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(K, A, 0.08) + V(0, 0.036, 0), lerp(K, A, 0.55) + V(0, 0.026, 0), 0.040, 0.042), 0.025)  # calf
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(K, A, 0.12) + V(-s * 0.018, 0.024, 0), lerp(K, A, 0.48) + V(-s * 0.016, 0.016, 0), 0.032), 0.025)
            return d

        g.apply(leg, (H[0] - 0.16, -0.16, 0.04), (H[0] + 0.16, 0.16, 0.99), op="union", k=0.03)

    # --- arms
    for side, s in (("R", -1), ("L", 1)):
        Sh, E, W = arm_pts(side)

        def arm(X, Y, Z, Sh=Sh, E=E, W=W, s=s):
            d = S.round_cone(X, Y, Z, Sh, E, 0.045, 0.033)
            d = S.smin(d, S.round_cone(X, Y, Z, E, W, 0.035, 0.0245), 0.015)
            d = S.smin(d, S.ellipsoid(X, Y, Z, Sh + V(s * 0.004, 0.0, -0.016), (0.047, 0.052, 0.056)), 0.035)  # deltoid
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(Sh, E, 0.25), lerp(Sh, E, 0.85), 0.034, 0.036), 0.02)  # biceps/triceps
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(E, W, 0.02), lerp(E, W, 0.55), 0.037, 0.033), 0.02)  # forearm flexors
            d = S.smin(d, S.ellipsoid_along(X, Y, Z, lerp(E, W, 0.85), W + (W - E) * 0.06, 0.028, 0.019), 0.01)  # flat wrist
            return d

        lo = np.minimum(np.minimum(Sh, E), W) - 0.08
        hi = np.maximum(np.maximum(Sh, E), W) + 0.08
        g.apply(arm, lo, hi, op="union", k=0.02)

    # flatten the neck top (bodysuit collar edge) and the very bottom
    g.apply(lambda X, Y, Z: Z - NECK_TOP, (-0.1, -0.1, NECK_TOP - 0.03), (0.1, 0.1, NECK_TOP + 0.05),
            op="intersect", k=0.004)


# ============================================================================= HEAD
HEAD_R = S.rot_xyz(0, 0, 6.0)  # slight turn toward the gesturing hand (her left)


def head_field(grid):
    c = SK["head_c"]
    R = HEAD_R

    def W(p):  # head-local -> world
        return c + R @ V(*p)

    # build everything in head-local space by transforming the grid coordinates
    def F(X, Y, Z):
        x, y, z = S.local(X, Y, Z, c, R)
        L = lambda fn, *a, **k: fn(x, y, z, *a, **k)
        d = L(S.ellipsoid, (0, 0.010, 0.022), (0.0745, 0.094, 0.090))                      # cranium
        d = S.smin(d, L(S.ellipsoid, (0, -0.034, -0.020), (0.065, 0.058, 0.070)), 0.04)    # face mass
        d = S.smin(d, L(S.ellipsoid, (0, -0.030, -0.052), (0.057, 0.052, 0.042)), 0.035)    # lower face
        for s in (-1, 1):
            d = S.smin(d, L(S.round_cone, (s * 0.058, 0.002, -0.032), (s * 0.019, -0.066, -0.076), 0.011, 0.011), 0.03)  # jaw line
            d = S.smin(d, L(S.ellipsoid, (s * 0.045, -0.058, -0.002), (0.021, 0.016, 0.015)), 0.02)  # cheekbone
            d = S.smin(d, L(S.ellipsoid, (s * 0.040, -0.061, -0.026), (0.022, 0.018, 0.021)), 0.022)  # cheek fat
            d = S.smin(d, L(S.ellipsoid, (s * 0.030, -0.081, 0.029), (0.022, 0.010, 0.008), S.rot_xyz(0, s * -8, 0)), 0.014)  # brow
        d = S.smin(d, L(S.ellipsoid, (0, -0.078, -0.081), (0.020, 0.015, 0.016)), 0.014)    # chin
        d = S.smin(d, L(S.ellipsoid, (0, -0.052, 0.046), (0.056, 0.034, 0.046)), 0.03)     # forehead
        # eye sockets
        for s in (-1, 1):
            d = S.smax(d, -L(S.ellipsoid, (s * 0.031, -0.090, 0.011), (0.0175, 0.0135, 0.0115)), 0.007)
        # eyelids: shells hugging the eyeball, opened by two tilted half-spaces
        for s in (-1, 1):
            ec = V(s * 0.031, -0.0775, 0.011)
            lid = L(S.sphere, ec, 0.0132)
            upper = S.smax(lid, -L(S.plane, ec + V(0, 0, 0.0030), V(0, 0.25, 1.0)), 0.0015)
            lower = S.smax(lid, L(S.plane, ec + V(0, 0, -0.0044), V(0, -0.2, 1.0)), 0.0015)
            crease = L(S.ellipsoid, ec + V(0, -0.002, 0.0085), (0.016, 0.010, 0.0042))
            d = S.smin(d, S.smin(upper, lower, 0.001), 0.004)
            d = S.smin(d, crease, 0.003)
        # nose
        d = S.smin(d, L(S.round_cone, (0, -0.089, 0.018), (0, -0.104, -0.020), 0.0062, 0.0086), 0.008)
        d = S.smin(d, L(S.sphere, (0, -0.1055, -0.025), 0.0105), 0.006)
        for s in (-1, 1):
            d = S.smin(d, L(S.ellipsoid, (s * 0.0118, -0.0935, -0.0285), (0.0082, 0.0090, 0.0072)), 0.004)  # alae
            d = S.smax(d, -L(S.ellipsoid, (s * 0.0060, -0.0985, -0.0372), (0.0032, 0.0045, 0.0020),
                              S.rot_xyz(0, 0, s * 20)), 0.0015)  # nostrils
        # lips
        d = S.smin(d, L(S.ellipsoid, (0, -0.0870, -0.0510), (0.0215, 0.0092, 0.0068), S.rot_xyz(-12, 0, 0)), 0.005)
        d = S.smin(d, L(S.ellipsoid, (0, -0.0855, -0.0602), (0.0190, 0.0098, 0.0072), S.rot_xyz(10, 0, 0)), 0.005)
        d = S.smin(d, L(S.ellipsoid, (0, -0.0905, -0.0478), (0.0045, 0.0040, 0.0060)), 0.003)  # cupid's bow
        d = S.smax(d, -L(S.ellipsoid, (0, -0.0985, -0.0560), (0.0215, 0.0065, 0.0009)), 0.0012)  # mouth line
        # ears
        for s in (-1, 1):
            Re = S.rot_xyz(0, 0, s * 4) @ S.rot_xyz(0, s * -6, 0)
            ear = L(S.ellipsoid, (s * 0.0715, 0.010, 0.002), (0.0070, 0.0170, 0.028), Re)
            ear = S.smax(ear, -L(S.ellipsoid, (s * 0.0770, 0.008, 0.000), (0.0040, 0.0110, 0.019), Re), 0.002)
            d = S.smin(d, ear, 0.004)
        # neck (skin)
        d = S.smin(d, L(S.round_cone, (0, 0.010, -0.050), (0, 0.016, -0.205), 0.050, 0.055), 0.025)
        for s in (-1, 1):  # sternocleidomastoid
            d = S.smin(d, L(S.round_cone, (s * 0.048, 0.012, -0.045), (s * 0.016, -0.036, -0.185), 0.012, 0.010), 0.018)
        d = S.smax(d, (1.425 - Z), 0.004)  # cut below the bodysuit collar
        return d

    grid.apply(F, grid.lo, grid.lo + (grid.n - 1) * grid.h, op="set", margin=0.0)


EYE_LOCAL = {s: V(s * 0.031, -0.0775, 0.011) for s in (-1, 1)}
EYE_R = 0.0118


def head_to_world(p):
    return SK["head_c"] + HEAD_R @ V(*p)


# ============================================================================= HANDS
# hand-local frame: +x toward fingers, +y toward thumb, +z = back of hand (right hand)
FINGERS = {
    #          base (x, y, z),            lengths,                 r0,     yaw
    "index": ((0.090, 0.0265, 0.002), (0.043, 0.026, 0.020), 0.0094, 6.0),
    "middle": ((0.094, 0.0075, 0.003), (0.048, 0.029, 0.021), 0.0098, 0.0),
    "ring": ((0.091, -0.0115, 0.002), (0.045, 0.027, 0.020), 0.0091, -5.0),
    "pinky": ((0.083, -0.0290, 0.000), (0.034, 0.021, 0.018), 0.0080, -11.0),
}
POSES = {
    # curl per joint (deg, flexion toward palm), thumb curls, thumb abduction
    "open_up": {"index": (8, 12, 8), "middle": (10, 14, 9), "ring": (14, 18, 10), "pinky": (18, 22, 12),
                "thumb": (10, 12), "thumb_abd": 42},
    "relaxed": {"index": (22, 30, 18), "middle": (26, 34, 20), "ring": (30, 38, 22), "pinky": (34, 40, 24),
                "thumb": (18, 20), "thumb_abd": 28},
}


def finger_joints(base, lengths, yaw, curls):
    pts = [V(*base)]
    pitch = 0.0
    for L, c in zip(lengths, curls):
        pitch += np.radians(c)
        dirv = V(np.cos(np.radians(yaw)) * np.cos(pitch), np.sin(np.radians(yaw)) * np.cos(pitch), -np.sin(pitch))
        pts.append(pts[-1] + dirv * L)
    return pts


def thumb_joints(pose):
    abd = np.radians(POSES[pose]["thumb_abd"])
    c1, c2 = (np.radians(a) for a in POSES[pose]["thumb"])
    base = V(0.022, 0.020, -0.008)
    d0 = V(np.cos(abd), np.sin(abd), -0.40); d0 /= np.linalg.norm(d0)
    p1 = base + d0 * 0.034
    def bend(d, a):
        # flex around an axis roughly perpendicular to the thumb, toward the palm
        ax = np.cross(d, V(0, 0, 1)); ax /= np.linalg.norm(ax)
        return d * np.cos(a) + np.cross(ax, d) * np.sin(a) * -1 + ax * np.dot(ax, d) * (1 - np.cos(a))
    d1 = bend(d0, c1); p2 = p1 + d1 * 0.029
    d2 = bend(d1, c2); p3 = p2 + d2 * 0.023
    return [base, p1, p2, p3]


def hand_field(grid, pose):
    P = POSES[pose]

    def F(X, Y, Z):
        d = S.round_box(X, Y, Z, (0.050, 0.0, 0.001), (0.044, 0.038, 0.0125), 0.010)  # palm
        d = S.smin(d, S.ellipsoid(X, Y, Z, (0.030, 0.022, -0.007), (0.030, 0.017, 0.013)), 0.008)  # thenar
        d = S.smin(d, S.ellipsoid(X, Y, Z, (0.045, -0.026, -0.006), (0.034, 0.013, 0.010)), 0.008)  # hypothenar
        d = S.smin(d, S.ellipsoid(X, Y, Z, (-0.012, 0.0, 0.0), (0.035, 0.027, 0.0185)), 0.012)    # wrist
        for name, (base, lens, r0, yaw) in FINGERS.items():
            pts = finger_joints(base, lens, yaw, P[name])
            radii = [r0, r0 * 0.92, r0 * 0.84, r0 * 0.74]
            f = S.chain(X, Y, Z, pts, radii)
            for j in (1, 2):  # knuckle bulges
                f = S.smin(f, S.sphere(X, Y, Z, pts[j] + V(0, 0, 0.0012), radii[j] * 1.06), 0.003)
            d = S.smin(d, f, 0.006)
        tp = thumb_joints(pose)
        d = S.smin(d, S.chain(X, Y, Z, tp, [0.0135, 0.0112, 0.0102, 0.0094]), 0.008)
        return d

    grid.apply(F, grid.lo, grid.lo + (grid.n - 1) * grid.h, op="set", margin=0.0)


def glove_region(X, Y, Z, pose):
    """Negative where the fingerless glove covers the hand."""
    P = POSES[pose]
    r = X - 0.088  # palm & back up to the knuckles
    for name, (base, lens, r0, yaw) in FINGERS.items():
        pts = finger_joints(base, lens, yaw, P[name])
        end = pts[0] + (pts[1] - pts[0]) * 0.55
        r = np.minimum(r, S.round_cone(X, Y, Z, pts[0] - (pts[1] - pts[0]) * 0.2, end, r0 + 0.006, r0 + 0.006))
    tp = thumb_joints(pose)
    r = np.minimum(r, S.round_cone(X, Y, Z, tp[0], tp[1], 0.019, 0.016))
    r = np.maximum(r, -0.045 - X)  # cuff ends on the forearm
    return r


def hand_matrix(side):
    """4x4 matrix: hand-local (right-hand convention) -> world, mirrored for the left hand."""
    Sh, E, W = arm_pts(side)
    fwd = W - E
    fwd /= np.linalg.norm(fwd)
    if side == "L":   # palm up, fingers forward and slightly raised, thumb outward (+X)
        fwd = fwd + V(0.0, 0.0, 0.10); fwd /= np.linalg.norm(fwd)
        dorsal = V(0.15, 0.05, -1.0)
    else:             # hanging, palm toward the thigh, back of hand outward
        fwd = fwd + V(0.0, -0.04, 0.0); fwd /= np.linalg.norm(fwd)
        dorsal = V(-1.0, -0.25, 0.0)
    dorsal = dorsal - fwd * np.dot(dorsal, fwd); dorsal /= np.linalg.norm(dorsal)
    lat = np.cross(dorsal, fwd)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = fwd, lat, dorsal
    if side == "L":
        M[:3, 1] *= -1  # mirror: left hand
    M[:3, 3] = W + fwd * 0.004
    return M


# ============================================================================= BOOTS
# boot-local frame: origin on the ground under the ankle, +x toward the toe, +y = medial(-)/lateral, +z up
def boot_matrix(side):
    A = SK[f"{side}_an"]
    yaw = np.radians(FOOT_YAW[side])
    fwd = V(np.sin(yaw), -np.cos(yaw), 0.0)
    up = V(0, 0, 1.0)
    lat = np.cross(up, fwd)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = fwd, lat, up
    M[:3, 3] = V(A[0], A[1] + 0.012, 0.0)
    return M


def boot_upper(X, Y, Z):
    d = S.round_cone(X, Y, Z, (-0.004, 0, 0.085), (0.0, 0.0, 0.250), 0.053, 0.051)                # shaft
    d = S.smin(d, S.round_box(X, Y, Z, (0.050, 0.0, 0.058), (0.095, 0.046, 0.040), 0.032), 0.03)  # foot
    d = S.smin(d, S.ellipsoid(X, Y, Z, (0.120, 0.0, 0.050), (0.060, 0.047, 0.036)), 0.02)         # toe box
    d = S.smin(d, S.ellipsoid(X, Y, Z, (-0.035, 0.0, 0.060), (0.040, 0.044, 0.045)), 0.02)        # heel
    d = S.smax(d, 0.026 - Z, 0.004)  # sits on the midsole
    return d


def boot_sole(X, Y, Z):
    d = S.round_box(X, Y, Z, (0.055, 0.0, 0.019), (0.128, 0.054, 0.019), 0.012)
    d = S.smin(d, S.ellipsoid(X, Y, Z, (0.150, 0.0, 0.022), (0.045, 0.053, 0.020)), 0.01)
    # tread lugs
    lugs = np.abs(((X + 0.3) % 0.020) - 0.010) - 0.0035
    d = S.smax(d, -S.smax(lugs, Z - 0.004, 0.0005), 0.001)
    return d


def boot_toecap(X, Y, Z):
    e = S.ellipsoid(X, Y, Z, (0.122, 0.0, 0.048), (0.066, 0.052, 0.042))
    return S.smax(S.shell(e, -0.001, 0.004), 0.09 - X, 0.002)
