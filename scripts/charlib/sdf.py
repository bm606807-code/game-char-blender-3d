"""Signed-distance-field sculpting toolkit (NumPy) + marching-cubes surface extraction.

Conventions: metres, Blender world axes (Z up), the character faces -Y.
Negative distance = inside. All primitive functions take broadcastable coordinate
arrays X (nx,1,1), Y (1,ny,1), Z (1,1,nz), which keeps memory low on big grids.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from skimage.measure import marching_cubes

BIG = np.float32(1.0)


# --------------------------------------------------------------------------- blending
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


# --------------------------------------------------------------------------- helpers
def _v(p):
    return np.asarray(p, dtype=np.float64)


def frame_from_axis(axis, up=(0, 0, 1)):
    """Orthonormal rows (u, v, w) with w along axis."""
    w = _v(axis); w = w / np.linalg.norm(w)
    up = _v(up)
    if abs(np.dot(up, w)) > 0.95:
        up = _v((1, 0, 0)) if abs(w[0]) < 0.9 else _v((0, 1, 0))
    u = np.cross(up, w); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.stack([u, v, w])


def rot_xyz(rx=0.0, ry=0.0, rz=0.0):
    """Rotation matrix from Euler angles in degrees (applied X, then Y, then Z)."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def local(X, Y, Z, c, R=None):
    """Coordinates of the grid in a local frame centred at c with rotation R (local->world)."""
    x, y, z = X - c[0], Y - c[1], Z - c[2]
    if R is None:
        return x, y, z
    # p_local = R^T (p - c)
    return (R[0, 0] * x + R[1, 0] * y + R[2, 0] * z,
            R[0, 1] * x + R[1, 1] * y + R[2, 1] * z,
            R[0, 2] * x + R[1, 2] * y + R[2, 2] * z)


# --------------------------------------------------------------------------- primitives
def sphere(X, Y, Z, c, r):
    return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r


def ellipsoid(X, Y, Z, c, radii, R=None):
    x, y, z = local(X, Y, Z, c, R)
    rx, ry, rz = radii
    k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2 + (z / rz ** 2) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def ellipsoid_along(X, Y, Z, a, b, r_perp, r_side=None, up=(0, 0, 1)):
    """Ellipsoid spanning segment a-b (muscle bellies)."""
    a, b = _v(a), _v(b)
    F = frame_from_axis(b - a, up)
    R = F.T  # columns u, v, w
    L = np.linalg.norm(b - a) / 2
    return ellipsoid(X, Y, Z, (a + b) / 2, (r_perp, r_side or r_perp, L), R)


def round_cone(X, Y, Z, a, b, ra, rb):
    a, b = _v(a), _v(b)
    ba = b - a
    l2 = np.dot(ba, ba)
    px, py, pz = X - a[0], Y - a[1], Z - a[2]
    t = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / l2, 0.0, 1.0)
    dx, dy, dz = px - ba[0] * t, py - ba[1] * t, pz - ba[2] * t
    return np.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * t)


def seg_param(X, Y, Z, a, b):
    a, b = _v(a), _v(b)
    ba = b - a
    l2 = np.dot(ba, ba)
    t = ((X - a[0]) * ba[0] + (Y - a[1]) * ba[1] + (Z - a[2]) * ba[2]) / l2
    return t


def chain(X, Y, Z, pts, radii):
    d = None
    for i in range(len(pts) - 1):
        c = round_cone(X, Y, Z, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = c if d is None else np.minimum(d, c)
    return d


def catmull(pts, n_per=6):
    """Catmull-Rom resample through control points (endpoints kept)."""
    P = [_v(p) for p in pts]
    P = [P[0] + (P[0] - P[1])] + P + [P[-1] + (P[-1] - P[-2])]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for s in np.linspace(0, 1, n_per, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3))
    out.append(P[-2])
    return np.array(out)


def round_box(X, Y, Z, c, half, r, R=None):
    x, y, z = local(X, Y, Z, c, R)
    qx, qy, qz = np.abs(x) - half[0] + r, np.abs(y) - half[1] + r, np.abs(z) - half[2] + r
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def torus(X, Y, Z, c, R_major, r_minor, R=None, scale_xy=(1.0, 1.0)):
    """Torus around local Z axis; scale_xy makes an elliptical ring."""
    x, y, z = local(X, Y, Z, c, R)
    q = np.sqrt((x / scale_xy[0]) ** 2 + (y / scale_xy[1]) ** 2) - R_major
    return np.sqrt(q * q + z * z) - r_minor


def plane(X, Y, Z, p, n):
    n = _v(n) / np.linalg.norm(n)
    return (X - p[0]) * n[0] + (Y - p[1]) * n[1] + (Z - p[2]) * n[2]


def loft_z(X, Y, Z, table, z0, z1, kcap=0.02):
    """Generalised cylinder along Z with asymmetric super-elliptic cross-sections.

    table rows: (z, half_width, front_depth, back_depth, y_centre, exponent)
    """
    t = np.asarray(table, float)
    zz = Z.ravel()
    cols = [PchipInterpolator(t[:, 0], t[:, i])(np.clip(zz, t[0, 0], t[-1, 0])) for i in range(1, 6)]
    shp = Z.shape
    hw, df, db, yc, ex = (c.reshape(shp) for c in cols)
    dy = Y - yc
    d = np.where(dy < 0, df, db)
    u = np.abs(X) / hw
    v = np.abs(dy) / d
    r = (u ** ex + v ** ex) ** (1.0 / ex)
    dist = (r - 1.0) * np.minimum(hw, d)
    dist = smax(dist, z0 - Z, kcap)
    dist = smax(dist, Z - z1, kcap)
    return dist


# --------------------------------------------------------------------------- grid / field
class Grid:
    def __init__(self, lo, hi, h):
        self.lo = _v(lo); self.h = float(h)
        self.n = np.ceil((_v(hi) - self.lo) / h).astype(int) + 1
        self.x = (self.lo[0] + np.arange(self.n[0]) * h).astype(np.float32)
        self.y = (self.lo[1] + np.arange(self.n[1]) * h).astype(np.float32)
        self.z = (self.lo[2] + np.arange(self.n[2]) * h).astype(np.float32)
        self.d = np.full(tuple(self.n), BIG, np.float32)

    def slices(self, bmin, bmax, margin=0.03):
        i0 = np.clip(np.floor((_v(bmin) - margin - self.lo) / self.h).astype(int), 0, self.n - 1)
        i1 = np.clip(np.ceil((_v(bmax) + margin - self.lo) / self.h).astype(int) + 1, 1, self.n)
        return tuple(slice(a, b) for a, b in zip(i0, i1))

    def coords(self, sl):
        return (self.x[sl[0]][:, None, None], self.y[sl[1]][None, :, None], self.z[sl[2]][None, None, :])

    def apply(self, fn, bmin, bmax, op="union", k=0.0, margin=0.03, target=None):
        target = self.d if target is None else target
        sl = self.slices(bmin, bmax, margin)
        X, Y, Z = self.coords(sl)
        v = fn(X, Y, Z).astype(np.float32)
        cur = target[sl]
        if op == "union":
            target[sl] = smin(cur, v, k)
        elif op == "subtract":
            target[sl] = smax(cur, -v, k)
        elif op == "intersect":
            target[sl] = smax(cur, v, k)
        elif op == "set":
            target[sl] = v
        return sl


def extract(vol, origin, h, level=0.0):
    """Marching cubes -> (verts Nx3 float32, faces Mx3 int32) in world space."""
    if vol.min() >= level or vol.max() <= level:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    v, f, _, _ = marching_cubes(vol, level=level, spacing=(h, h, h), allow_degenerate=False)
    v = v + _v(origin)
    # skimage returns faces wound for the positive-gradient side; flip so normals point outward
    f = f[:, ::-1]
    return v.astype(np.float32), f.astype(np.int32)


def extract_sub(grid, vol_fn, bmin, bmax, margin=0.01):
    """Evaluate vol_fn(X,Y,Z,sl) on a sub-box of grid and polygonise it."""
    sl = grid.slices(bmin, bmax, margin)
    X, Y, Z = grid.coords(sl)
    vol = vol_fn(X, Y, Z, sl).astype(np.float32)
    # close the box so pieces touching the bounds are watertight
    vol[[0, -1], :, :] = np.maximum(vol[[0, -1], :, :], 1e-3)
    vol[:, [0, -1], :] = np.maximum(vol[:, [0, -1], :], 1e-3)
    vol[:, :, [0, -1]] = np.maximum(vol[:, :, [0, -1]], 1e-3)
    origin = (grid.x[sl[0]][0], grid.y[sl[1]][0], grid.z[sl[2]][0])
    return extract(vol, origin, grid.h)


def shell(d, offset, thickness):
    """Thin shell between distance offset and offset+thickness from a base field."""
    return np.abs(d - (offset + thickness / 2)) - thickness / 2
