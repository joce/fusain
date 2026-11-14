"""A portrait of a young woman, to be drawn in ballpoint pen.

Stage 1: the head, designed and lit as a plain shaded study.

The face is built on MediaPipe's canonical face model (Apache License 2.0),
``examples/data/face_mesh.obj``: a neutral face surface in centimetres, x to
the sitter's left, y up, z towards the viewer.  It is reshaped here into a
young woman's face, its eyes are opened and given eyeballs, and a neck,
shoulders and hair are built round it.  Everything is rendered from one
camera, with a soft key light and a weak fill, into a grey study, a study in
flat colours and the maps the pen work will follow.

    uv run python examples/portrait.py [output folder]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from murex import filters, write_png
from numba import njit

HERE = Path(__file__).parent
MESH = HERE / "data" / "face_mesh.obj"
WIDTH, HEIGHT = 1100, 1400

# Regions of the picture, as stored in the region map.
BACKGROUND, SKIN, LIPS, EYE_WHITE, IRIS, PUPIL, BROW, HAIR, GARMENT = range(9)
REGION_NAMES = (
    "background skin lips eye_white iris pupil brow hair garment".split()
)

# Kinds of surface in the scene, one per triangle.
K_FACE, K_MOUTH, K_HOLE, K_BALL, K_BODY, K_SKULL, K_HAIR, K_LID = range(8)

# Landmarks of the face mesh (vertex indices).
LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             409, 270, 269, 267, 0, 37, 39, 40, 185]  # fmt: skip
LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             415, 310, 311, 312, 13, 82, 81, 80, 191]  # fmt: skip
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133,
         173, 157, 158, 159, 160, 161, 246]  # fmt: skip
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362,
         398, 384, 385, 386, 387, 388, 466]  # fmt: skip
BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109]  # fmt: skip
NOSE_TIP, NOSE_BRIDGE, CHIN = 1, 168, 152


# ----------------------------------------------------------------- the mesh


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of a Wavefront file (polygons are fanned)."""
    vs, fs = [], []
    for line in path.read_text().splitlines():
        p = line.split()
        if p and p[0] == "v":
            vs.append([float(c) for c in p[1:4]])
        elif p and p[0] == "f":
            idx = [int(c.split("/")[0]) - 1 for c in p[1:]]
            fs += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(vs), np.array(fs, dtype=np.int64)


def subdivide(v, f, kind):
    """Midpoint subdivision; each new triangle keeps its parent's kind.  The
    old vertices keep their indices, so landmarks stay valid."""
    m = len(f)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    ab, bc, ca = (len(v) + inv.reshape(3, m))
    a, b, c = f.T
    v2 = np.vstack([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2])
    f2 = np.concatenate(
        [np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
    )
    return v2, f2, np.tile(kind, 4)


def neighbour_mean(v, f):
    acc = np.zeros_like(v)
    cnt = np.zeros(len(v))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        np.add.at(acc, f[:, i], v[f[:, j]])
        np.add.at(acc, f[:, j], v[f[:, i]])
        np.add.at(cnt, f[:, i], 1)
        np.add.at(cnt, f[:, j], 1)
    return acc / np.maximum(cnt, 1)[:, None]


def smooth(v, f, rounds=1, lam=0.5, mu=None, weight=None):
    """Laplacian smoothing; with ``mu`` (negative), Taubin's, which does not
    shrink.  ``weight`` scales the step per vertex."""
    w = np.ones(len(v)) if weight is None else weight
    for _ in range(rounds):
        for k in (lam,) if mu is None else (lam, mu):
            v = v + k * w[:, None] * (neighbour_mean(v, f) - v)
    return v


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Turn by ``yaw`` (about y), nod by ``pitch`` (about x, positive lifts
    the chin), tilt by ``roll`` (about z); degrees."""
    a, b, c = (math.radians(x) for x in (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ rx @ ry


def polygon_inside(px, py, poly):
    """Even-odd test of points against a closed polygon (k, 2)."""
    inside = np.zeros(np.shape(px), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        cross = (b > py) != (d > py)
        t = (py - b) / np.where(d == b, 1e-12, d - b)
        inside ^= cross & (px < a + t * (c - a))
    return inside


# ------------------------------------------------------- reshaping the face


@dataclass
class Bump:
    """A smooth displacement: ``move`` at ``at``, falling off as a Gaussian of
    radius ``r``; mirrored to the other side when ``at`` is off the middle."""

    at: tuple
    r: float
    move: tuple


def apply_bumps(v, bumps):
    out = v.copy()
    for b in bumps:
        at, mv = np.array(b.at, float), np.array(b.move, float)
        sides = [(at, mv)]
        if abs(at[0]) > 1e-6:
            sides.append((at * [-1, 1, 1], mv * [-1, 1, 1]))
        for c, d in sides:
            w = np.exp(-np.sum((v - c) ** 2, axis=1) / (2 * b.r**2))
            out += w[:, None] * d
    return out


def taper_jaw(v):
    """Narrow the lower face towards a small chin, more the lower it is."""
    y = v[:, 1]
    s = np.clip((-0.5 - y) / 7.5, 0, 1)
    s = s * s * (3 - 2 * s)
    out = v.copy()
    out[:, 0] *= 0.965 - 0.10 * s
    return out


FACE_BUMPS = [
    # jaw: the angle softened, raised and drawn in; the chin smaller
    Bump((5.9, -6.2, -0.6), 2.2, (-0.25, 0.35, 0.0)),
    Bump((3.2, -8.3, 2.8), 1.6, (-0.15, 0.2, -0.1)),
    Bump((0.0, -9.3, 4.2), 1.5, (0.0, 0.3, 0.05)),
    # forehead: no brow ridge, a gently rounded brow
    Bump((0.0, 4.6, 5.6), 1.3, (0.0, 0.0, -0.25)),
    Bump((2.6, 5.1, 5.0), 1.3, (0.0, 0.0, -0.18)),
    Bump((0.0, 7.3, 4.8), 2.2, (0.0, 0.0, 0.18)),
    # nose: slimmer, shorter, slightly turned up, narrow wings
    Bump((0.0, -0.7, 7.4), 0.9, (0.0, 0.22, -0.4)),
    Bump((0.0, -1.9, 6.1), 0.7, (0.0, 0.15, -0.1)),
    Bump((1.75, -1.0, 4.9), 0.8, (-0.28, 0.08, -0.05)),
    Bump((0.0, 1.2, 6.5), 0.9, (0.0, 0.0, -0.12)),
    # cheekbones higher and fuller, the cheeks below them smooth
    Bump((4.6, 1.0, 4.0), 1.5, (0.08, 0.15, 0.12)),
    Bump((4.2, -3.0, 3.8), 1.6, (-0.12, 0.0, -0.05)),
    # lips: fuller, a defined bow, the corners lifted a touch
    Bump((0.0, -3.7, 5.8), 0.7, (0.0, 0.05, 0.22)),
    Bump((0.65, -3.35, 5.8), 0.35, (0.0, 0.14, 0.05)),
    Bump((0.0, -5.1, 5.5), 0.8, (0.0, -0.12, 0.18)),
    Bump((2.45, -4.35, 4.3), 0.5, (-0.05, 0.12, 0.0)),
]


def eye_centre(v, ring):
    return v[ring].mean(axis=0)


def open_eyes(v):
    """Make the eye openings larger: the upper lid higher, the corners
    further apart, a smooth falloff round them."""
    out = v.copy()
    for ring, side in ((EYE_R, -1), (EYE_L, 1)):
        c = eye_centre(v, ring)
        d = v - c
        w = np.exp(-(d[:, 0] ** 2 / (2 * 1.6**2) + d[:, 1] ** 2 / (2 * 0.55**2)))
        w *= np.exp(-np.clip(np.abs(d[:, 2]) - 1.0, 0, None) ** 2)
        up = d[:, 1] > 0
        out[:, 1] += w * np.where(up, 0.32, 0.12) * np.tanh(d[:, 1] / 0.25)
        out[:, 0] += w * 0.09 * np.tanh(d[:, 0] / 0.6)
        del side
    return out


UPPER_INNER = [191, 80, 81, 82, 13, 312, 311, 310, 415]
LOWER_INNER = [95, 88, 178, 87, 14, 317, 402, 318, 324]


def close_mouth(v):
    """Bring the lips together: the parted inner lip lines meet halfway,
    the lips round them following with a soft falloff."""
    up, lo = v[UPPER_INNER], v[LOWER_INNER]
    mid = (up + lo) / 2
    out = v.copy()
    for c, half in ((up, mid - up), (lo, mid - lo)):
        for p, d in zip(c, half):
            w = np.exp(-np.sum((v[:, :2] - p[:2]) ** 2, axis=1) / (2 * 0.32**2))
            out += w[:, None] * d * [1, 0.97, 1]
    return out


def slim_nose(v):
    """Narrow the bridge and the tip of the nose towards the middle."""
    x, y, z = v.T
    band = np.exp(-(((y - 0.8) / 2.6) ** 2)) * np.clip((z - 4.3) / 1.2, 0, 1)
    out = v.copy()
    out[:, 0] -= 0.2 * x * np.exp(-(x**2) / (2 * 1.1**2)) * band
    return out


def feminise(v):
    v = close_mouth(v)
    v = taper_jaw(v)
    v = slim_nose(v)
    v = apply_bumps(v, FACE_BUMPS)
    return open_eyes(v)


def face_kinds(v, f):
    """Kind of each triangle of the base mesh: face, mouth (inside the inner
    lip line) or a hole (inside an eye opening)."""
    cen = v[f].mean(axis=1)
    kind = np.full(len(f), K_FACE)
    mouth = polygon_inside(cen[:, 0], cen[:, 1], v[LIP_INNER, :2])
    kind[mouth] = K_MOUTH
    for ring in (EYE_R, EYE_L):
        hole = polygon_inside(cen[:, 0], cen[:, 1], v[ring, :2])
        kind[hole & (cen[:, 2] > v[ring, 2].min() - 0.5)] = K_HOLE
    return kind


def build_face(levels=3):
    v, f = load_obj(MESH)
    v = feminise(v)
    kind = face_kinds(v, f)
    for _ in range(levels):
        v, f, kind = subdivide(v, f, kind)
        v = smooth(v, f, 1, 0.5)
    v = smooth(v, f, 3, 0.5, -0.53)
    return v, f, kind


# ----------------------------------------------------------- rasterising


@njit(cache=False)
def _raster(px, py, pz, tris, depth, tid, bary):  # pragma: no cover
    h, w = depth.shape
    for t in range(tris.shape[0]):
        i, j, k = tris[t, 0], tris[t, 1], tris[t, 2]
        x0, x1, x2 = px[i], px[j], px[k]
        y0, y1, y2 = py[i], py[j], py[k]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xa = max(int(math.floor(min(x0, x1, x2))), 0)
        xb = min(int(math.ceil(max(x0, x1, x2))), w - 1)
        ya = max(int(math.floor(min(y0, y1, y2))), 0)
        yb = min(int(math.ceil(max(y0, y1, y2))), h - 1)
        for yy in range(ya, yb + 1):
            cy = yy + 0.5
            for xx in range(xa, xb + 1):
                cx = xx + 0.5
                l0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
                l1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
                l2 = 1.0 - l0 - l1
                if min(l0, l1, l2) < -1e-9:
                    continue
                z = l0 * pz[i] + l1 * pz[j] + l2 * pz[k]
                if z > depth[yy, xx]:
                    depth[yy, xx] = z
                    tid[yy, xx] = t
                    bary[yy, xx, 0] = l0
                    bary[yy, xx, 1] = l1
                    bary[yy, xx, 2] = l2


@dataclass
class View:
    """An orthographic view: ``scale`` pixels per centimetre, the origin at
    pixel ``centre``, of an image ``size`` (height, width)."""

    size: tuple
    scale: float
    centre: tuple


@dataclass
class Frame:
    """What one raster pass sees at each pixel."""

    depth: np.ndarray
    tid: np.ndarray
    bary: np.ndarray
    seen: np.ndarray = field(init=False)

    def __post_init__(self):
        self.seen = self.tid >= 0


def raster(v, f, view: View) -> Frame:
    h, w = view.size
    px = view.centre[0] + v[:, 0] * view.scale
    py = view.centre[1] - v[:, 1] * view.scale
    depth = np.full((h, w), -np.inf)
    tid = np.full((h, w), -1, dtype=np.int64)
    bary = np.zeros((h, w, 3))
    _raster(px, py, v[:, 2].copy(), f, depth, tid, bary)
    return Frame(depth, tid, bary)


def interpolate(attr, f, frame: Frame):
    """Per-pixel value of a per-vertex attribute (n, k)."""
    t = np.where(frame.seen, frame.tid, 0)
    corners = attr[f[t]]  # h, w, 3, k
    return np.einsum("hwc,hwck->hwk", frame.bary, corners)


# ------------------------------------------------------------------ the eyes

EYE_RADIUS = 1.2
IRIS_RADIUS = 0.56
PUPIL_RADIUS = 0.2
#: Upper and lower lid points in the middle of each opening.
LID_MID = {"r": (159, 145), "l": (386, 374)}


@dataclass
class Eye:
    centre: np.ndarray  # head space
    ring: list


def place_eye(v, ring, mids):
    """An eyeball behind an opening, its front a little ahead of the lid
    margins, the iris under the upper lid by a fifth of its height."""
    top, bot = v[mids[0]], v[mids[1]]
    cx = (top[0] + bot[0]) / 2
    cy = top[1] - 0.6 * IRIS_RADIUS - 0.03
    cz = (top[2] + bot[2]) / 2 + 0.12 - EYE_RADIUS
    return Eye(np.array([cx, cy, cz]), ring)


def wrap_lids(v, f, eyes, gap=0.07):
    """Lift the lids round each eyeball so that they lie over it."""
    lift = np.zeros(len(v))
    for e in eyes:
        d2 = np.sum((v[:, :2] - e.centre[:2]) ** 2, axis=1)
        r2 = (EYE_RADIUS + gap) ** 2
        zs = e.centre[2] + np.sqrt(np.clip(r2 - d2, 0, None))
        need = np.where(d2 < r2, np.clip(zs - v[:, 2], 0, None), 0)
        lift = np.maximum(lift, need)
    spread = lift[:, None]
    for _ in range(6):
        spread = np.maximum(neighbour_mean(spread, f), 0.85 * spread)
    out = v.copy()
    out[:, 2] += np.maximum(spread[:, 0], lift)
    return out


def hole_loop(f, kind):
    """The boundary of the holes: edges of hole triangles shared with the
    face, as a mapping from each vertex to the next one round the loop."""
    hole = f[kind == K_HOLE]
    face = f[kind != K_HOLE]
    fe = {tuple(sorted(e)) for t in face for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    nxt = {}
    for t in hole:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if tuple(sorted((a, b))) in fe:
                nxt[b] = a  # reversed: the face's orientation
    return nxt


def loops(nxt):
    seen, out = set(), []
    for s in nxt:
        if s in seen:
            continue
        loop = [s]
        seen.add(s)
        while nxt[loop[-1]] not in seen:
            loop.append(nxt[loop[-1]])
            seen.add(loop[-1])
        out.append(loop)
    return out


def strip(n, r0, r1):
    """Triangles joining two closed rings of n points starting at r0, r1."""
    i = np.arange(n)
    j = np.roll(i, -1)
    return np.concatenate(
        [np.stack([r0 + i, r0 + j, r1 + i], 1), np.stack([r0 + j, r1 + j, r1 + i], 1)]
    )


def lid_margin(v, loop, eye: Eye):
    """A thick lid edge: from the rim of the opening, a ring turning in
    towards the eyeball, then one inside it."""
    rim = v[loop]
    d = rim - eye.centre
    unit = d / np.linalg.norm(d, axis=1, keepdims=True)
    ring1 = rim - unit * 0.06 + np.array([0, 0, -0.07])
    ring2 = eye.centre + unit * (0.8 * EYE_RADIUS)
    n = len(loop)
    pts = np.vstack([rim, ring1, ring2])
    return pts, np.concatenate([strip(n, 0, n), strip(n, n, 2 * n)])


def uv_sphere(c, r, nu=96, nv=48):
    th = np.linspace(0, np.pi, nv + 1)
    ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    t, p = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(t) * np.sin(p), np.cos(t), np.sin(t) * np.cos(p)], axis=-1)
    v = c + r * d.reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nv), np.arange(nu), indexing="ij")
    a = i * nu + j
    b = i * nu + (j + 1) % nu
    f = np.concatenate(
        [np.stack([a, a + nu, b], -1).reshape(-1, 3), np.stack([b, a + nu, b + nu], -1).reshape(-1, 3)]
    )
    return v, f


# ------------------------------------------------------------------ the scene


@dataclass
class Pose:
    """The body turned by ``body``, the head turned on the neck by ``head``
    about ``pivot`` (in head space), both rotation matrices."""

    body: np.ndarray
    head: np.ndarray
    pivot: np.ndarray

    def of_head(self, p):
        return ((p - self.pivot) @ self.head.T + self.pivot) @ self.body.T

    def of_body(self, p):
        return p @ self.body.T

    def head_dir(self, d):
        return d @ self.head.T @ self.body.T


@dataclass
class Scene:
    """Triangles of every surface, in camera space, with their kind, the
    vertex normals of each part, and each vertex's place before the pose."""

    v: list = field(default_factory=list)
    f: list = field(default_factory=list)
    kind: list = field(default_factory=list)
    n: list = field(default_factory=list)
    local: list = field(default_factory=list)
    count: int = 0

    def add(self, local, posed, f, kind):
        self.v.append(posed)
        self.local.append(local)
        self.n.append(vertex_normals(posed, f))
        self.f.append(f + self.count)
        self.kind.append(np.broadcast_to(kind, (len(f),)).astype(np.int64))
        self.count += len(posed)

    def merged(self):
        return (
            np.vstack(self.v),
            np.vstack(self.f),
            np.concatenate(self.kind),
            np.vstack(self.n),
            np.vstack(self.local),
        )


def add_face(scene, pose, face):
    v, f, kind = face
    keep = kind != K_HOLE
    eyes = [place_eye(v, EYE_R, LID_MID["r"]), place_eye(v, EYE_L, LID_MID["l"])]
    v = wrap_lids(v, f, eyes)
    scene.add(v, pose.of_head(v), f[keep], kind[keep])
    for loop in loops(hole_loop(f, kind)):
        c = v[loop].mean(axis=0)
        eye = min(eyes, key=lambda e: np.linalg.norm(e.centre[:2] - c[:2]))
        pts, tri = lid_margin(v, loop, eye)
        scene.add(pts, pose.of_head(pts), tri, K_LID)
    for eye in eyes:
        pts, tri = uv_sphere(eye.centre, EYE_RADIUS)
        scene.add(pts, pose.of_head(pts), tri, K_BALL)
    return eyes


# ------------------------------------------------- neck, shoulders and skull


def catmull_rom(pts, n):
    """n points along a Catmull-Rom curve through ``pts`` (k, d), spaced
    evenly along its length."""
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    t = np.linspace(0, 1, 40)[:-1, None]
    seg = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        seg.append(
            0.5
            * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
               + (3 * p1 - p0 - 3 * p2 + p3) * t**3)  # fmt: skip
        )
    seg.append(pts[-1:])
    dense = np.vstack(seg)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, dense[:, k]) for k in range(dense.shape[1])], axis=1)


#: Half the width of the body against height, from inside the head down the
#: neck, over the shoulders and down the arms: (half width, height, half depth).
BODY_PROFILE = np.array([
    (4.9, -1.0, 5.1), (4.95, -8.0, 5.2), (5.2, -12.6, 5.35), (6.0, -14.6, 5.6),
    (8.3, -15.9, 5.9), (12.4, -17.3, 6.3), (15.6, -18.6, 6.6), (17.6, -20.2, 7.0),
    (18.4, -22.8, 7.6), (18.5, -27.0, 8.2), (18.2, -46.0, 8.6),
])  # fmt: skip


def body_axis_z(y):
    """The neck leans forward: its axis comes forward as it rises."""
    t = np.clip((y + 15.0) / 12.0, 0, 1)
    return -6.6 + 2.6 * t * t * (3 - 2 * t) + 0.0 * y


def body_surface(n_rows=260, n_cols=200):
    prof = catmull_rom(BODY_PROFILE, n_rows)
    w, y, d = prof[:, 0:1], prof[:, 1:2], prof[:, 2:3]
    th = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)[None, :]
    expo = 2.0 + 0.6 * np.clip((-14.0 - y) / 6.0, 0, 1)
    c, s = np.cos(th), np.sin(th)
    x = w * np.sign(s) * np.abs(s) ** (2 / expo)
    z = d * np.sign(c) * np.abs(c) ** (2 / expo) + body_axis_z(y)
    v = np.stack([x, np.broadcast_to(y, x.shape), z], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_rows - 1), np.arange(n_cols), indexing="ij")
    a = i * n_cols + j
    b = i * n_cols + (j + 1) % n_cols
    f = np.concatenate(
        [np.stack([a, b, a + n_cols], -1).reshape(-1, 3),
         np.stack([b, b + n_cols, a + n_cols], -1).reshape(-1, 3)]  # fmt: skip
    )
    return v, f


def segment_distance(p, a, b):
    ab = np.asarray(b, float) - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


#: Forms on the neck and chest, as raised (positive) or sunk ridges along
#: segments: (from, to, width, height).
BODY_FORMS = [
    ((1.6, -16.6, 0.0), (8.0, -17.0, 0.0), 0.55, 0.22),  # collarbones
    ((8.0, -17.0, 0.0), (13.5, -17.6, 0.0), 0.6, 0.18),
    ((2.5, -15.4, 0.0), (9.0, -15.6, 0.0), 0.7, -0.18),  # hollows above them
    ((0.0, -15.8, 0.0), (0.0, -16.3, 0.0), 0.9, -0.42),  # the notch
    ((4.2, -6.5, 0.0), (1.4, -15.6, 0.0), 0.8, 0.22),  # the long neck muscles
]


def shape_body(v, f):
    """Collarbones, the notch between them and the neck's muscles, laid on
    the front of the body."""
    n = vertex_normals(v, f)
    front = np.clip(v[:, 2] - body_axis_z(v[:, 1]), 0, None)
    front = np.clip(front / 2.5, 0, 1)
    out = v.copy()
    q = v * [1, 1, 0]
    for a, b, wd, h in BODY_FORMS:
        for sx in (1, -1):
            pa, pb = np.array(a) * [sx, 1, 1], np.array(b) * [sx, 1, 1]
            dist = segment_distance(q, pa, pb)
            out += (h * front * np.exp(-0.5 * (dist / wd) ** 2))[:, None] * n
    return out


SKULL_CENTRE = np.array([0.0, 3.6, -4.3])
SKULL_RADII = np.array([7.2, 10.0, 9.2])


def skull():
    v, f = uv_sphere(np.zeros(3), 1.0, 96, 64)
    return v * SKULL_RADII + SKULL_CENTRE, f


# ------------------------------------------------------------------ the hair

#: The shell the hair lies on: centre and radii (x, y, z) at its widest.
HAIR_CENTRE = np.array([0.0, 3.4, -4.3])
HAIR_RADII = np.array([8.55, 11.2, 10.5])
#: Where the side curtains begin, either side of the face (radians from the
#: front; the sitter's left, the near side, first).
CURTAIN_EDGE = (0.86, -1.12)
PARTING = -0.2  # azimuth of the parting, a little to the sitter's right
FRINGE_END = 5.55  # height of the fringe's ends over the brows
CURTAIN_END = -12.6


def lock_tips(phi, width, seed):
    """Uneven ends: a lock every ``width`` radians, each with its own length
    and a rounded point; returns 0 at a tip, rising to 1 between locks."""
    rng = np.random.default_rng(seed)
    k = np.floor(phi / width)
    jitter = rng.uniform(-0.5, 0.5, 4096)
    idx = (k.astype(np.int64) + 2048) % 4096
    frac = phi / width - k - 0.5 - 0.3 * jitter[idx]
    length = rng.uniform(0, 1, 4096)[idx]
    return np.clip(np.abs(frac) * 2, 0, 1) ** 1.6, length


def hair_ends(phi):
    """Height where the hair ends at each azimuth: the fringe in front, the
    curtains at the sides and back, a little uneven."""
    tip, ln = lock_tips(phi, 0.16, 3)
    fringe = FRINGE_END + 0.32 * tip + 0.28 * ln + 0.18 * np.tanh(phi / 0.5) + 0.25 * (phi / 0.9) ** 2
    tip2, ln2 = lock_tips(phi + 0.05, 0.22, 7)
    curtain = CURTAIN_END + 0.9 * tip2 + 0.6 * ln2 + 0.7 * (np.abs(phi) - 1.0).clip(0) / 2.1
    edge = np.where(phi > 0, CURTAIN_EDGE[0], -CURTAIN_EDGE[1])
    t = np.clip((np.abs(phi) - edge) / 0.07 + 0.5, 0, 1)
    t = t * t * (3 - 2 * t)
    return fringe * (1 - t) + curtain * t, t


def profile_length(y_end):
    b = HAIR_RADII[1]
    c = HAIR_CENTRE[1]
    return np.where(
        y_end >= c, np.arccos(np.clip((y_end - c) / b, -1, 1)), np.pi / 2 + (c - y_end) / b
    )


def shell_radius(phi):
    ax, az = HAIR_RADII[0], HAIR_RADII[2]
    return 1 / np.sqrt(np.sin(phi) ** 2 / ax**2 + np.cos(phi) ** 2 / az**2)


@dataclass
class HairGrid:
    phi: np.ndarray  # (rows, cols) azimuth of each column's root
    s: np.ndarray  # profile parameter
    u: np.ndarray  # 0 at the crown, 1 at the ends
    side: np.ndarray  # 0 in the fringe, 1 in the curtains


def hair_grid(rows=240, cols=900):
    phi = np.linspace(-np.pi, np.pi, cols, endpoint=False) + np.pi / cols
    y_end, side = hair_ends(phi)
    s_end = profile_length(y_end)
    u = np.linspace(0, 1, rows)[:, None] ** 0.9
    s = u * s_end[None, :]
    shape = (rows, cols)
    return HairGrid(
        np.broadcast_to(phi, shape), s, np.broadcast_to(u, shape), np.broadcast_to(side, shape)
    )


def hair_profile(g: HairGrid):
    """Horizontal distance from the axis and height of each grid point."""
    b, yc = HAIR_RADII[1], HAIR_CENTRE[1]
    a = shell_radius(g.phi)
    cap = g.s <= np.pi / 2
    alpha = np.minimum(g.s, np.pi / 2)
    rho = a * np.sin(alpha)
    crown = 1 + 0.05 * np.cos(alpha) ** 2
    y = np.where(cap, yc + b * np.cos(alpha) * crown, yc - (g.s - np.pi / 2) * b)
    # The curtains turn in at their ends and come a little forward.
    fall = np.clip((g.s - np.pi / 2) * b, 0, None)
    end_len = (1 - g.u) * (g.s / np.maximum(g.u, 1e-6)) * b
    turn = np.exp(-np.clip(end_len, 0, None) / 2.2) * np.clip(fall / 4, 0, 1)
    rho = rho - 0.9 * turn * g.side
    return rho, y, fall
