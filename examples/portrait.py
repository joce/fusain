"""A portrait of a young woman, drawn in blue, black and red ballpoint.

Run from the repository root: ``python examples/portrait.py``. Writes
``examples/output/portrait.png``; ``python examples/portrait.py FOLDER``
also writes the shaded studies of the head and the maps the pen follows to
FOLDER.

The head is modelled first, then drawn.  The face is built on MediaPipe's
canonical face model (Apache License 2.0), ``examples/data/face_mesh.obj``:
a neutral face surface in centimetres, x to the sitter's left, y up, z
towards the viewer.  It is reshaped into a young woman's face, its eyes are
opened and given eyeballs, and a neck, shoulders and hair laid in locks are
built round it.  One camera and a soft light give, per pixel, what is seen,
how it is lit, which way its surface runs and which way the hair flows.

The pen then works as a hand does, patch by patch: for each ink and layer
it finds where the most tone is still wanting, lays there a bundle of
near-parallel arcs that follow the surface, each tapering at both ends and
keeping its distance from the others of its layer, and repeats until the
tone is reached.  Red is the flesh, blue woven through it makes the
shadows violet, black is kept for the darkest places; the hair is drawn
lock by lock along its flow, and a few living lines and accents come last.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from murex import filters, write_png
from numba import njit

from fusain import Sheet, papers, pens

HERE = Path(__file__).parent
MESH = HERE / "data" / "face_mesh.obj"
WIDTH, HEIGHT = 1100, 1400

# Regions of the picture, as stored in the region map.
BACKGROUND, SKIN, LIPS, EYE_WHITE, IRIS, PUPIL, BROW, HAIR, GARMENT = range(9)
REGION_NAMES = [
    "background",
    "skin",
    "lips",
    "eye_white",
    "iris",
    "pupil",
    "brow",
    "hair",
    "garment",
]

# Kinds of surface in the scene, one per triangle.
K_FACE, K_MOUTH, K_HOLE, K_BALL, K_BODY, K_SKULL, K_HAIR, K_LID, K_ROLL = range(9)

# Landmarks of the face mesh (vertex indices).
LIP_OUTER = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
]
LIP_INNER = [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
NOSE_TIP, NOSE_BRIDGE, CHIN = 1, 168, 152


# ----------------------------------------------------------------- the mesh


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of a Wavefront file (polygons are fanned).

    Used for the face mesh: MediaPipe's canonical face model, under the
    Apache License 2.0."""
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
    ab, bc, ca = len(v) + inv.reshape(3, m)
    a, b, c = f.T
    v2 = np.vstack([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2])
    f2 = np.concatenate(
        [
            np.stack(t, axis=1)
            for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
        ]
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
    ry = np.array(
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    )
    rx = np.array(
        [[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]]
    )
    rz = np.array(
        [[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]]
    )
    return rz @ rx @ ry


def polygon_inside(px, py, poly):
    """Even-odd test of points against a closed polygon (k, 2)."""
    inside = np.zeros(np.shape(px), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1, strict=True):
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
    cheeks = np.exp(-0.5 * ((y - 1.0) / 2.5) ** 2)
    out = v.copy()
    out[:, 0] *= 0.965 + 0.035 * cheeks - 0.1 * s
    return out


FACE_BUMPS = [
    # jaw: the angle softened, raised and drawn in; the chin smaller
    Bump((5.9, -6.2, -0.6), 2.4, (-0.4, 0.6, 0.0)),
    Bump((3.2, -8.3, 2.8), 1.6, (-0.15, 0.2, -0.1)),
    Bump((0.0, -9.3, 4.2), 2.2, (0.0, 0.55, 0.1)),
    # forehead: no brow ridge, a gently rounded brow
    Bump((0.0, 4.6, 5.6), 1.3, (0.0, 0.0, -0.25)),
    Bump((2.6, 5.1, 5.0), 1.3, (0.0, 0.0, -0.18)),
    Bump((0.0, 7.3, 4.8), 2.2, (0.0, 0.0, 0.18)),
    # nose: slimmer, shorter, slightly turned up, narrow wings, a low root
    Bump((0.0, -0.7, 7.4), 1.0, (0.0, 0.25, -0.25)),
    Bump((0.0, 2.7, 5.7), 0.9, (0.0, 0.0, -0.22)),
    Bump((0.0, -1.9, 6.1), 0.7, (0.0, 0.15, -0.1)),
    Bump((1.75, -1.0, 4.9), 0.8, (-0.32, 0.1, -0.08)),
    Bump((0.0, 1.2, 6.5), 0.9, (0.0, 0.0, -0.12)),
    # young cheeks: full below the eye, flowing without a break to the jaw
    Bump((4.6, 1.0, 4.0), 1.5, (0.08, 0.15, 0.12)),
    Bump((3.0, -1.0, 5.0), 1.6, (0.12, 0.0, 0.28)),
    Bump((3.9, -4.0, 3.9), 1.8, (0.16, 0.0, 0.2)),
    Bump((2.1, -2.6, 5.2), 0.7, (0.0, 0.0, 0.12)),
    Bump((1.4, -8.4, 4.0), 1.0, (-0.22, 0.05, 0.0)),
    # lips: at rest, a gentle bow, the corners level and barely lifted
    Bump((0.0, -4.3, 5.6), 1.1, (0.0, 0.0, -0.16)),
    Bump((0.0, -3.7, 5.8), 0.7, (0.0, -0.06, -0.03)),
    Bump((0.0, -3.4, 5.9), 0.3, (0.0, -0.06, 0.0)),
    Bump((0.65, -3.35, 5.8), 0.35, (0.0, 0.09, 0.04)),
    Bump((0.75, -1.55, 5.4), 0.4, (0.0, -0.05, 0.12)),
    Bump((0.0, -5.1, 5.5), 0.9, (0.0, -0.2, 0.1)),
    Bump((2.45, -4.35, 4.3), 0.5, (-0.18, 0.12, 0.0)),
]


def eye_centre(v, ring):
    return v[ring].mean(axis=0)


def open_eyes(v):
    """Make the eye openings larger: the upper lid higher, the corners
    further apart, a smooth falloff round them."""
    out = v.copy()
    for ring in (EYE_R, EYE_L):
        c = eye_centre(v, ring)
        d = v - c
        w = np.exp(-(d[:, 0] ** 2 / (2 * 1.6**2) + d[:, 1] ** 2 / (2 * 0.55**2)))
        w *= np.exp(-np.clip(np.abs(d[:, 2]) - 1.0, 0, None) ** 2)
        up = d[:, 1] > 0
        out[:, 1] += w * np.where(up, 0.64, 0.22) * np.tanh(d[:, 1] / 0.25)
        out[:, 0] += w * 0.28 * np.tanh(d[:, 0] / 0.6)
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
        for p, d in zip(c, half, strict=True):
            w = np.exp(-np.sum((v[:, :2] - p[:2]) ** 2, axis=1) / (2 * 0.32**2))
            out += w[:, None] * d * [1, 0.97, 1]
    return out


def small_nose(v, k=0.86):
    """Shrink the nose towards its root: shorter, less projecting, narrower."""
    root = np.array([0.0, 2.4, 5.5])
    d = v - [0.0, -0.4, 6.4]
    w = np.exp(
        -0.5 * ((d[:, 0] / 1.5) ** 2 + (d[:, 1] / 2.2) ** 2 + (d[:, 2] / 1.6) ** 2)
    )
    w *= np.clip((v[:, 2] - 4.2) / 0.8, 0, 1)
    return v + w[:, None] * (root + k * (v - root) - v)


def slim_nose(v):
    """Narrow the bridge and the tip of the nose towards the middle."""
    x, y, z = v.T
    band = np.exp(-(((y - 0.8) / 2.6) ** 2)) * np.clip((z - 4.3) / 1.2, 0, 1)
    out = v.copy()
    out[:, 0] -= 0.3 * x * np.exp(-(x**2) / (2 * 1.1**2)) * band
    return out


#: Heights in the face moved, to bring the eyes, nose and mouth closer
#: together: (height, shift), from the top down.
FACE_HEIGHTS = np.array(
    [
        (9.5, 0.0),
        (5.5, -0.25),
        (2.6, -0.4),
        (-1.9, 0.18),
        (-4.3, 0.45),
        (-8.6, 0.8),
        (-12.0, 0.8),
    ]
)


def compact(v):
    out = v.copy()
    ys, dy = FACE_HEIGHTS[::-1, 0], FACE_HEIGHTS[::-1, 1]
    out[:, 1] += np.interp(v[:, 1], ys, dy)
    return out


def feminise(v):
    v = close_mouth(v)
    v = taper_jaw(v)
    v = slim_nose(v)
    v = apply_bumps(v, FACE_BUMPS)
    v = small_nose(v)
    return compact(open_eyes(v))


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


def outer_rim(f):
    """The mask's boundary, in the order of its triangles' edges."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    nxt = dict(e[cnt[inv] == 1].tolist())
    loop = [next(iter(nxt))]
    while nxt[loop[-1]] != loop[0]:
        loop.append(nxt[loop[-1]])
    return np.array(loop)


def roll_under(v, f, kind, r=0.35, steps=8, deep=0.9):
    """Turn the mask's edge under, so that the jaw and the sides of the face
    end in a rounded edge rather than a cut."""
    rim = outer_rim(f)
    n = vertex_normals(v, f)[rim]
    t = np.roll(v[rim], -1, axis=0) - np.roll(v[rim], 1, axis=0)
    out = unit_rows(np.cross(t, n))
    out *= np.sign(np.sum(out * (v[rim] - v.mean(axis=0)), axis=1))[:, None]
    rings = [rim]
    pts = [v]
    count = len(v)
    for k in range(1, steps + 1):
        th = np.radians(115) * k / steps
        p = v[rim] + r * np.sin(th) * out - deep * (1 - np.cos(th)) * n
        if k == steps:
            p = p - 0.8 * n + 0.3 * out
        pts.append(p)
        rings.append(np.arange(count, count + len(rim)))
        count += len(rim)
    tris = [f]
    for a, b in itertools.pairwise(rings):
        a1, b1 = np.roll(a, -1), np.roll(b, -1)
        tris += [np.stack([a1, a, b], 1), np.stack([a1, b, b1], 1)]
    new = sum(len(t) for t in tris[1:])
    return np.vstack(pts), np.vstack(tris), np.concatenate([kind, np.full(new, K_ROLL)])


def build_face(levels=3):
    v, f = load_obj(MESH)
    v = feminise(v)
    kind = face_kinds(v, f)
    for _ in range(levels):
        v, f, kind = subdivide(v, f, kind)
        v = smooth(v, f, 1, 0.5)
    v = smooth(v, f, 3, 0.5, -0.53)
    v = smooth(v, f, 10, 0.5, -0.53, fold_weight(v))
    return roll_under(v, f, kind)


#: Where the skin is smoothed further: no folds from the nose to the mouth,
#: no hollows under the eyes.  (x, y, radius), mirrored.
SMOOTH_SPOTS = [(2.2, -2.3, 1.1), (3.0, 1.0, 0.9), (2.7, -4.6, 0.6), (0.9, 4.7, 1.0)]


def fold_weight(v):
    w = np.zeros(len(v))
    for x, y, r in SMOOTH_SPOTS:
        for sx in (x, -x):
            d2 = (v[:, 0] - sx) ** 2 + (v[:, 1] - y) ** 2
            w = np.maximum(w, np.exp(-0.5 * d2 / r**2))
    return w * (v[:, 2] > 2.0)


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
IRIS_RADIUS = 0.62
PUPIL_RADIUS = 0.17
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
    cy = bot[1] + IRIS_RADIUS - 0.03
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
    fe = {
        tuple(sorted(e))
        for t in face
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))
    }
    nxt = {}
    for t in hole:
        nxt.update(
            {
                b: a
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))
                if tuple(sorted((a, b))) in fe
            }
        )  # reversed: the face's orientation
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
        [
            np.stack([a, a + nu, b], -1).reshape(-1, 3),
            np.stack([b, a + nu, b + nu], -1).reshape(-1, 3),
        ]
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
        return (p + [0.0, BODY_LIFT, 0.0]) @ self.body.T

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
    attrs: list = field(default_factory=list)
    count: int = 0

    def add(self, local, posed, f, kind, attrs=None, soften=0):
        self.v.append(posed)
        self.local.append(local)
        n = vertex_normals(posed, f)
        for _ in range(soften):
            n = unit_rows(neighbour_mean(n, f) + n)
        self.n.append(n)
        self.f.append(f + self.count)
        self.kind.append(np.broadcast_to(kind, (len(f),)).astype(np.int64))
        self.attrs.append(attrs or {})
        self.count += len(posed)

    def attr(self, name, width):
        """One per-vertex attribute over all parts, zero where a part has none."""
        return np.vstack(
            [
                a.get(name, np.zeros((len(v), width)))
                for a, v in zip(self.attrs, self.v, strict=True)
            ]
        )

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
    face[0][:] = v
    scene.add(v, pose.of_head(v), f[keep], kind[keep], soften=3)
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
            * (
                2 * p1
                + (p2 - p0) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (3 * p1 - p0 - 3 * p2 + p3) * t**3
            )
        )
    seg.append(pts[-1:])
    dense = np.vstack(seg)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    u = np.linspace(0, s[-1], n)
    return np.stack(
        [np.interp(u, s, dense[:, k]) for k in range(dense.shape[1])], axis=1
    )


#: Half the width of the body against height, from inside the head down the
#: neck, over the shoulders and down the arms: (half width, height, half depth).
BODY_PROFILE = np.array(
    [
        (4.6, -1.0, 4.9),
        (4.65, -8.0, 5.0),
        (4.95, -12.6, 5.2),
        (6.0, -14.6, 5.6),
        (8.3, -15.9, 5.9),
        (12.4, -17.3, 6.3),
        (15.6, -18.6, 6.6),
        (17.6, -20.2, 7.0),
        (18.5, -22.8, 7.6),
        (18.9, -27.0, 8.2),
        (19.8, -46.0, 8.6),
    ]
)


def body_axis_z(y):
    """The neck leans forward: its axis comes forward as it rises."""
    t = np.clip((y + 15.0) / 12.0, 0, 1)
    return -6.6 + 2.6 * t * t * (3 - 2 * t)


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
        [
            np.stack([a, b, a + n_cols], -1).reshape(-1, 3),
            np.stack([b, b + n_cols, a + n_cols], -1).reshape(-1, 3),
        ]
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
SKULL_RADII = np.array([7.1, 9.0, 9.2])


def skull():
    """The back and top of the head; its front is left to the face."""
    v, f = uv_sphere(np.zeros(3), 1.0, 96, 64)
    v = v * SKULL_RADII + SKULL_CENTRE
    c = v[f].mean(axis=1)
    return v, f[(c[:, 2] < -0.5) | (c[:, 1] > 9.0)]


# ------------------------------------------------------------------ the hair

#: The shell the hair lies on: centre and radii (x, y, z) at its widest.
HAIR_CENTRE = np.array([0.0, 3.2, -4.4])
HAIR_RADII = np.array([8.1, 10.1, 9.7])
#: Where the side curtains begin, either side of the face (radians from the
#: front; the sitter's left, the near side, first).
CURTAIN_EDGE = (0.76, -1.12)
PARTING = -0.2  # azimuth of the parting, a little to the sitter's right
FRINGE_END = 4.45  # height of the fringe's ends over the brows
CURTAIN_END = -12.4
#: Where the mass under the locks ends: at the hairline in front (the
#: forehead shows between the locks of the fringe), short of the ends behind.
BASE_ENDS = (7.5, -11.6)


def lock_tips(phi, width, seed):
    """Uneven ends: locks of irregular width round the head, each with its
    own length and a soft point.  Returns, at each azimuth, 0 at a lock's
    tip rising to 1 between locks, and the lock's length (0 to 1)."""
    rng = np.random.default_rng(seed)
    edges = np.cumsum(rng.uniform(0.55, 1.45, 200) * width) - np.pi - 1.0
    k = np.clip(np.searchsorted(edges, phi) - 1, 0, len(edges) - 2)
    frac = (phi - edges[k]) / (edges[k + 1] - edges[k])
    centre = rng.uniform(0.35, 0.65, 200)[k]
    side = np.where(frac < centre, frac / centre, (1 - frac) / (1 - centre))
    return (1 - side) ** 1.4, rng.uniform(0, 1, 200)[k]


def hair_ends(phi, ends=(FRINGE_END, CURTAIN_END)):
    """Height where the hair ends at each azimuth: the fringe in front, the
    curtains at the sides and back, a little uneven."""
    tip, ln = lock_tips(phi, 0.13, 3)
    fringe = (
        ends[0]
        + 0.25 * tip**1.5
        + 0.25 * ln
        + 0.15 * np.tanh(phi / 0.5)
        + 0.3 * (phi / 0.9) ** 2
    )
    tip2, ln2 = lock_tips(phi + 0.05, 0.2, 7)
    curtain = (
        ends[1] + 0.35 * tip2**2 + 0.45 * ln2 + 0.9 * (np.abs(phi) - 1.0).clip(0) / 2.1
    )
    edge = np.where(phi > 0, CURTAIN_EDGE[0], -CURTAIN_EDGE[1])
    t = np.clip((np.abs(phi) - edge) / 0.07 + 0.5, 0, 1)
    t = t * t * (3 - 2 * t)
    # single strands stop short of their lock's end: the ends are frayed
    fray = np.random.default_rng(5).random(len(phi)) ** 3
    return fringe * (1 - t) + curtain * t + fray * (0.35 + 0.35 * t), t


def profile_length(y_end):
    b = HAIR_RADII[1]
    c = HAIR_CENTRE[1]
    return np.where(
        y_end >= c,
        np.arccos(np.clip((y_end - c) / b, -1, 1)),
        np.pi / 2 + (c - y_end) / b,
    )


def shell_radius(phi):
    """The shell's horizontal radius at its widest, deeper in front (over
    the forehead) than behind."""
    ax = HAIR_RADII[0]
    az = HAIR_RADII[2] + 0.9 * np.clip(np.cos(phi), 0, 1)
    return 1 / np.sqrt(np.sin(phi) ** 2 / ax**2 + np.cos(phi) ** 2 / az**2)


@dataclass
class HairGrid:
    phi: np.ndarray  # (rows, cols) azimuth of each column's root
    s: np.ndarray  # profile parameter
    u: np.ndarray  # 0 at the crown, 1 at the ends
    side: np.ndarray  # 0 in the fringe, 1 in the curtains


HAIR_GRID = (240, 900)
LOCK_GRID = (70, 7)


def hair_grid(rows=HAIR_GRID[0], cols=HAIR_GRID[1], ends=BASE_ENDS):
    phi = np.linspace(-np.pi, np.pi, cols, endpoint=False) + np.pi / cols
    y_end, side = hair_ends(phi, ends)
    s_end = profile_length(y_end)
    u = np.linspace(0, 1, rows)[:, None] ** 0.9
    s = u * s_end[None, :]
    shape = (rows, cols)
    return HairGrid(
        np.broadcast_to(phi, shape),
        s,
        np.broadcast_to(u, shape),
        np.broadcast_to(side, shape),
    )


def hair_profile(g: HairGrid):
    """Horizontal distance from the axis and height of each grid point."""
    b, yc = HAIR_RADII[1], HAIR_CENTRE[1]
    a = shell_radius(g.phi)
    cap = g.s <= np.pi / 2
    alpha = np.minimum(g.s, np.pi / 2)
    rho = a * np.sin(alpha) ** 0.9
    y = np.where(cap, yc + b * np.cos(alpha), yc - (g.s - np.pi / 2) * b)
    # The curtains swell a little by the cheeks, narrow towards the neck
    # and turn in at their ends.
    fall = np.clip((g.s - np.pi / 2) * b, 0, None)
    rho = rho - 0.35 * np.exp(-0.5 * ((alpha - 1.2) / 0.25) ** 2) * g.side
    rho = rho + 0.25 * np.exp(-0.5 * ((fall - 4.0) / 3.0) ** 2) - 0.06 * fall
    end_len = (1 - g.u) * (g.s / np.maximum(g.u, 1e-6)) * b
    turn = np.exp(-np.clip(end_len, 0, None) / 2.2) * np.clip(fall / 4, 0, 1)
    rho = rho - 0.9 * turn * g.side
    return rho, y, fall


#: The map of how far the face stands from the hair's axis, by azimuth and
#: height: (pixels per radian, pixels per centimetre, top height).
CYL = (120.0, 12.0, 16.0)


def face_clearance(v, f):
    """Distance of the face from the hair's axis at each azimuth and height
    (a cylindrical z-buffer), spread a little so that the hair clears it
    smoothly."""
    zc = HAIR_CENTRE[2]
    f = f[v[f].min(axis=1)[:, 2] > -1.0]  # the face, not its edge by the ears
    phi = np.arctan2(v[:, 0], v[:, 2] - zc)
    rho = np.hypot(v[:, 0], v[:, 2] - zc)
    cyl = np.stack([phi, v[:, 1], rho], axis=1)
    h, w = int(36 * CYL[1]), int(2 * np.pi * CYL[0])
    view = View((h, w), 1.0, (0.0, 0.0))
    px = np.stack(
        [(cyl[:, 0] + np.pi) * CYL[0], (CYL[2] - cyl[:, 1]) * CYL[1], cyl[:, 2]], 1
    )
    px[:, 1] *= -1  # raster() flips y
    fr = raster(px, f, view)
    out = np.where(fr.seen, fr.depth, 0.0)
    # above the top of the face, let its distance fall away gently
    fall = 1.8 / CYL[1]
    for i in range(out.shape[0] - 2, -1, -1):
        out[i] = np.maximum(out[i], out[i + 1] - fall)
    for _ in range(2):
        out = np.maximum(out, filters.box_blur(out, 5))
    return filters.box_blur(out, 4)


def clearance_at(cmap, phi, y):
    i = np.clip(((CYL[2] - y) * CYL[1]).astype(np.int64), 0, cmap.shape[0] - 1)
    j = np.clip(((phi + np.pi) * CYL[0]).astype(np.int64), 0, cmap.shape[1] - 1)
    return cmap[i, j]


def locks(phi, s, u, seed=11):
    """Gentle clumping of the hair into locks, stronger towards the ends."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(phi)
    for k in (5, 7, 11, 17, 26):
        a, b, c = rng.uniform(0, 2 * np.pi, 3)
        amp = rng.uniform(0.4, 1.0)
        out += (
            amp
            * np.sin(
                k * phi + a + 0.9 * np.sin(1.3 * s + b) + 0.5 * np.sin(2 * phi + c)
            )
            / k**0.7
        )
    return out * 0.05


def soft_max(a, b, k):
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0, 1)
    return b + (a - b) * h + k * h * (1 - h)


def hair_points(g: HairGrid, cmap):
    rho, y, fall = hair_profile(g)
    edge = np.where(g.phi > 0, CURTAIN_EDGE[0], -CURTAIN_EDGE[1])
    near_edge = np.exp(-np.clip(np.abs(g.phi) - edge, 0, None) / 0.35)
    sway = (
        0.22 * np.exp(-0.5 * ((fall - 6.5) / 3.5) ** 2)
        + 0.08 * np.clip(fall / 14, 0, 1) ** 2
    )
    sweep = sway * near_edge * g.side
    phi = g.phi - np.sign(g.phi) * sweep
    rho = rho + locks(phi, g.s, g.u)
    rho = soft_max(rho, clearance_at(cmap, phi, y) + 0.3, 0.3)
    # the parting: a narrow groove from the crown forward
    alpha = np.minimum(g.s, np.pi / 2)
    win = np.clip((alpha - 0.12) / 0.15, 0, 1) * np.clip((0.85 - alpha) / 0.2, 0, 1)
    dphi = np.angle(np.exp(1j * (phi - PARTING)))
    groove = 0.16 * win * np.exp(-0.5 * (dphi * rho / 0.35) ** 2)
    rho = rho - groove
    y = y - groove * np.cos(alpha)
    zc = HAIR_CENTRE[2]
    return np.stack([rho * np.sin(phi), y, zc + rho * np.cos(phi)], axis=-1), dphi


def grid_triangles(rows, cols, wrap=True):
    i, j = np.meshgrid(
        np.arange(rows - 1), np.arange(cols if wrap else cols - 1), indexing="ij"
    )
    a = i * cols + j
    b = i * cols + (j + 1) % cols
    return np.concatenate(
        [
            np.stack([a, b, a + cols], -1).reshape(-1, 3),
            np.stack([b, b + cols, a + cols], -1).reshape(-1, 3),
        ]
    )


def hair_flow(p, dphi, g: HairGrid):
    """Direction of the hair at each point, root to tip: down the profile,
    turned away from the parting near it."""
    ds = np.gradient(p, axis=0)
    dp = np.gradient(p, axis=1)
    ds /= np.maximum(np.linalg.norm(ds, axis=-1, keepdims=True), 1e-9)
    dp /= np.maximum(np.linalg.norm(dp, axis=-1, keepdims=True), 1e-9)
    alpha = np.minimum(g.s, np.pi / 2)
    away = (
        np.sign(dphi) * np.exp(-np.abs(dphi) / 0.5) * np.clip((1.0 - alpha) / 0.5, 0, 1)
    )
    flow = ds + 0.8 * away[..., None] * dp
    return flow / np.maximum(np.linalg.norm(flow, axis=-1, keepdims=True), 1e-9)


@dataclass
class Lock:
    """One lock of hair laid over the mass: where its root is round the
    head, where it ends, how wide it is and how it falls."""

    phi: float
    y_end: float
    width: float
    lift: float  # how far it lies off the mass
    wave: float  # amplitude of its sideways wave (cm)
    phase: float
    turn: float  # its end turns out (positive) or in (negative), cm
    sway: float  # extra turn towards the face as it falls (radians)
    shine: float


def lock_specs(seed=17):
    """Two layers of locks all round, the outer one over the gaps of the
    inner; the fringe's locks finer and of more different lengths."""
    rng = np.random.default_rng(seed)
    out = []
    for layer, n in ((0, 46), (1, 40)):
        for i in range(n):
            phi = -np.pi + (i + 0.5 * layer + rng.uniform(-0.3, 0.3)) * 2 * np.pi / n
            out.append(one_lock(phi, layer, rng))
    # one lock from the near temple falls forward across the cheek
    out.append(Lock(0.88, -9.4, 1.1, 0.35, 0.15, 1.0, 0.4, 0.13, 1.2))
    return out


def one_lock(phi, layer, rng):
    edge = CURTAIN_EDGE[0] if phi > 0 else -CURTAIN_EDGE[1]
    lift = 0.1 + 0.2 * layer + rng.uniform(0, 0.08)
    if layer and rng.uniform() < 0.15:
        lift += 0.35  # a lock that stands off the mass
    shine = rng.uniform(0.5, 1.3)
    if abs(phi) < edge - 0.06:
        y_end = FRINGE_END + rng.uniform(-0.15, 0.55) + 0.3 * (phi / 0.9) ** 2
        return Lock(phi, y_end, rng.uniform(1.2, 1.8), lift, 0.06, 0.0, 0.0, 0.0, shine)
    # the curtains' locks keep clear of the face
    phi = np.sign(phi) * max(abs(phi), edge + 0.07)
    y_end = CURTAIN_END + rng.uniform(-0.6, 1.3) + 0.9 * max(abs(phi) - 1.0, 0) / 2.1
    turn = rng.choice([-1, -1, 1]) * rng.uniform(0.3, 0.8)
    if 1.2 < abs(phi) < 2.0 and rng.uniform() < 0.25:
        y_end, turn = CURTAIN_END - 0.9, 0.7  # ends that reach the shoulder
    near = abs(phi) - edge < 0.35  # by the face: finer, flatter
    wave = rng.uniform(0.15, 0.4) * (0.4 if near else 1.0)
    return Lock(
        phi,
        y_end,
        rng.uniform(1.0, 1.4) if near else rng.uniform(1.4, 2.2),
        lift,
        wave,
        rng.uniform(0, 6.3),
        turn,
        0.0,
        shine,
    )


def lock_frame(c, cmap, phi, s, u, side):
    """Unit vectors along a lock's centre line, across it on the mass, and
    out of the mass."""
    t = unit_rows(np.gradient(c, axis=0))
    nb, _ = hair_points(HairGrid(phi + 0.01, s, u, side), cmap)
    b = nb - c
    b = unit_rows(b - np.sum(b * t, axis=1, keepdims=True) * t)
    n = np.cross(t, b)
    out = c - HAIR_CENTRE
    n *= np.sign(np.sum(n * out, axis=1, keepdims=True))
    return t, b, n


def lock_path(lock: Lock, u):
    """Profile parameter and azimuth along a lock.  Its root is on the
    parting (further forward for the locks that fall nearer the front), from
    where it turns aside to fall at its own azimuth; the cheek lock swings
    forward below the eye."""
    turn = np.angle(np.exp(1j * (lock.phi - PARTING)))
    s_root = 0.62 * np.clip(1 - abs(turn) / 2.6, 0, 1)
    s_end = profile_length(np.array(lock.y_end))
    s = s_root + u * (s_end - s_root)
    aside = np.clip((s - s_root) / 0.55, 0, 1)
    aside = aside * aside * (3 - 2 * aside)
    late = np.clip((u - 0.45) / 0.5, 0, 1)
    sway = np.sign(lock.phi) * lock.sway * late * late * (3 - 2 * late)
    return s, PARTING + turn * aside - sway


def lock_ribbon(k, lock: Lock, cmap, rows=LOCK_GRID[0], cols=LOCK_GRID[1]):
    """A lock as a ribbon rounded across, lying on the mass: waving a
    little, tapering to a point, its end turned in or out."""
    side = 0.0 if lock.y_end > HAIR_CENTRE[1] else 1.0
    u = np.linspace(0, 1, rows) ** 0.85
    s, phi = lock_path(lock, u)
    c, _ = hair_points(HairGrid(phi, s, u, np.full(rows, side)), cmap)
    _, b, n = lock_frame(c, cmap, phi, s, u, np.full(rows, side))
    fall = np.clip(HAIR_CENTRE[1] - c[:, 1], 0, None)
    left = (s[-1] - s) * HAIR_RADII[1]
    end = np.exp(-left / 2.5) * np.clip(fall / 5, 0, 1)
    wave = lock.wave * np.sin(fall / 2.2 + lock.phase) * np.clip(fall / 4, 0, 1)
    c = c + b * wave[:, None] + n * (lock.turn * end)[:, None]
    rho = np.hypot(c[:, 0], c[:, 2] - HAIR_CENTRE[2])
    tip = np.clip((1 - u) / (0.3 - 0.16 * side), 0, 1) ** 0.55
    w = np.minimum(lock.width, 0.12 * rho + 0.05) * (0.15 + 0.85 * tip)
    v = np.linspace(-1, 1, cols)
    lift = lock.lift * np.clip(u / 0.3, 0.3, 1)  # the root lies on the mass
    bulge = (1 - v**2)[None, :] * (0.22 * w + 0.7 * lift)[:, None] + 0.3 * lift[:, None]
    pts = (
        c[:, None]
        + b[:, None] * (v[None, :] * w[:, None] / 2)[..., None]
        + n[:, None] * bulge[..., None]
    )
    tan = unit_rows(np.gradient(pts, axis=0))
    strand = 1000.0 * (k + 1) + (v[None, :] * w[:, None] / 0.06)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    remain = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    attrs = {
        "flow": tan.reshape(-1, 3),
        "hair_u": np.broadcast_to(u[:, None], (rows, cols)).reshape(-1, 1),
        "strand": strand.reshape(-1, 1),
        "lock": np.full((rows * cols, 1), k + 1.0),
        "shine": np.full((rows * cols, 1), lock.shine),
        "remain": np.broadcast_to(remain[:, None], (rows, cols)).reshape(-1, 1),
    }
    return pts.reshape(-1, 3), grid_triangles(rows, cols, wrap=False), attrs


def build_hair(face_v, face_f):
    g = hair_grid()
    cmap = face_clearance(face_v, face_f)
    p, dphi = hair_points(g, cmap)
    flow = hair_flow(p, dphi, g)
    rows, cols = g.phi.shape
    f = grid_triangles(rows, cols)
    strand = np.broadcast_to(np.arange(cols, dtype=float), (rows, cols))
    attrs = {
        "flow": flow.reshape(-1, 3),
        "hair_u": g.u.reshape(-1, 1),
        "strand": strand.reshape(-1, 1),
    }
    attrs["shine"] = np.full((rows * cols, 1), 0.6)
    ribbons = [lock_ribbon(k, lock, cmap) for k, lock in enumerate(lock_specs())]
    return p.reshape(-1, 3), f, attrs, ribbons


# ------------------------------------------------------- posing and seeing

HEAD_TURN = (-27.0, 3.0, -3.0)  # yaw, pitch (positive: face down), roll
BODY_TURN = (-38.0, 0.0, 1.5)
PIVOT = np.array([0.0, -3.5, -4.5])
#: Picture scale (pixels per centimetre) and where the head-space origin
#: (between the eyes, a little behind) lands.
VIEW = View((HEIGHT, WIDTH), 34.0, (545.0, 565.0))
#: The body sits this much higher than it was modelled: a shorter neck.
BODY_LIFT = 2.5


def make_pose():
    body = rotation(*BODY_TURN)
    head = body.T @ rotation(*HEAD_TURN)
    return Pose(body, head, PIVOT)


@dataclass
class Model:
    scene: Scene
    eyes: list
    pose: Pose
    face: tuple
    #: The hair as strands to draw along, posed: grids (rows root to tip,
    #: columns across), the mass first, then each lock.
    strands: list = field(default_factory=list)


def build_model() -> Model:
    pose = make_pose()
    face = build_face()
    scene = Scene()
    eyes = add_face(scene, pose, face)
    sv, sf = skull()
    scene.add(sv, pose.of_head(sv), sf, K_SKULL)
    bv, bf = body_surface()
    bv = shape_body(bv, bf)
    scene.add(bv, pose.of_body(bv), bf, K_BODY)
    hv, hf, attrs, locks = build_hair(face[0], face[1][face[2] == K_FACE])
    strands = []
    for (pv, pf, pa), shape in zip(
        [(hv, hf, attrs)] + locks, [HAIR_GRID] + [LOCK_GRID] * len(locks), strict=True
    ):
        pa["flow"] = pose.head_dir(pa["flow"])
        scene.add(pv, pose.of_head(pv), pf, K_HAIR, pa)
        strands.append(pose.of_head(pv).reshape(*shape, 3))
    return Model(scene, eyes, pose, face, strands)


# ------------------------------------------------------------------ the light


def unit(*v):
    a = np.array(v, float)
    return a / np.linalg.norm(a)


#: Towards the key light (above, to the side she faces, in front) and the
#: fill (weak, from the other side, low).
KEY = unit(-0.5, 0.55, 0.67)
FILL = unit(0.75, 0.2, 0.65)
VIEWER = np.array([0.0, 0.0, 1.0])


def light_basis(w):
    u = np.cross([0.0, 1.0, 0.0], w)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(w, u), w])


@njit(cache=False)
def _blocked(depth, x, y, z, bias):  # pragma: no cover
    h, w = depth.shape
    xi, yi = int(x), int(y)
    if 0 <= xi < w and 0 <= yi < h:
        return depth[yi, xi] - z - bias
    return -1.0


@njit(cache=False)
def _soft_shadow(sx, sy, sz, grad, depth, disc, search, angle, out):  # pragma: no cover
    """Percentage-closer soft shadows: find the blockers round each point,
    then filter over a penumbra that widens with their distance.  Each
    sample is compared with the plane of the receiving surface, ``grad``
    being its slope in depth per pixel."""
    k = disc.shape[0]
    for i in range(sx.shape[0]):
        bias = 0.05 + 0.8 * (abs(grad[i, 0]) + abs(grad[i, 1]))
        nb, acc = 0, 0.0
        for j in range(k):
            dx, dy = disc[j, 0] * search, disc[j, 1] * search
            z = sz[i] + grad[i, 0] * dx + grad[i, 1] * dy
            d = _blocked(depth, sx[i] + dx, sy[i] + dy, z, bias)
            if d > 0:
                nb += 1
                acc += d
        if nb == 0:
            out[i] = 1.0
            continue
        pen = max(acc / nb * angle, 1.0)
        lit = 0
        for j in range(k):
            dx, dy = disc[j, 0] * pen, disc[j, 1] * pen
            z = sz[i] + grad[i, 0] * dx + grad[i, 1] * dy
            if _blocked(depth, sx[i] + dx, sy[i] + dy, z, bias) <= 0:
                lit += 1
        out[i] = lit / k


def disc_samples(k=48):
    i = np.arange(k) + 0.5
    r = np.sqrt(i / k)
    a = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def shadows(v, f, pos, normal, seen, lift, key=KEY, scale=36.0, softness=0.14):
    """How much of the key light reaches each pixel's point (0 to 1); each
    point is first moved ``lift`` towards the light (a bias against
    surfaces shadowing themselves)."""
    basis = light_basis(key)
    lv = v @ basis.T
    lo, hi = lv[:, :2].min(0) - 1, lv[:, :2].max(0) + 1
    size = np.ceil((hi - lo) * scale).astype(int) + 1
    view = View((size[1], size[0]), scale, (-lo[0] * scale, hi[1] * scale))
    depth = raster(lv, f, view).depth
    p = pos[seen] @ basis.T
    sx = view.centre[0] + p[:, 0] * scale
    sy = view.centre[1] - p[:, 1] * scale
    nl = normal[seen] @ basis.T
    nw = np.clip(nl[:, 2], 0.25, None)
    # depth of the receiving plane per pixel step (x right, y down)
    grad = np.stack([-nl[:, 0] / nw, nl[:, 1] / nw], axis=1) / scale
    out = np.ones(len(p))
    _soft_shadow(
        sx,
        sy,
        p[:, 2] + lift[seen],
        grad,
        depth,
        disc_samples(),
        0.5 * scale,
        softness * scale,
        out,
    )
    full = np.ones(seen.shape)
    full[seen] = out
    return full


def occlusion(depth, seen, scale):
    """Ambient occlusion from the depth buffer: how much nearer surface
    there is round each point, over a few distances."""
    z = np.where(seen, depth, depth[seen].min() - 10.0)
    occ = np.zeros_like(z)
    for r_cm, wgt in ((0.12, 0.5), (0.35, 0.6), (1.0, 0.6), (2.6, 0.5)):
        b = filters.box_blur(z, max(int(r_cm * scale), 1))
        d = np.clip(b - z, 0, None)
        occ += (
            wgt
            * np.clip(d / r_cm, 0, 1)
            * np.exp(-np.clip(d - r_cm, 0, None) / (2 * r_cm))
        )
    return np.clip(1 - 0.45 * occ, 0.15, 1)


# --------------------------------------------------------- what is seen where


@dataclass
class Buffers:
    """Per pixel: what is seen, where it is (camera space and before the
    pose), which way it faces, and the hair's direction."""

    seen: np.ndarray
    kind: np.ndarray
    depth: np.ndarray
    pos: np.ndarray
    local: np.ndarray
    normal: np.ndarray
    inside: np.ndarray
    flow: np.ndarray
    hair_u: np.ndarray
    strand: np.ndarray
    lock: np.ndarray
    shine: np.ndarray


def unit_rows(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


#: How far from a lock's end its strands may stop (cm).
FRAY = 1.4


def fray_ends(model: Model, v, f, fr: Frame, view: View) -> Frame:
    """Strands stop at different lengths near a lock's end: where one has
    stopped, see past the lock's end to what lies behind."""
    scene = model.scene
    left = scene.attr("remain", 1)[:, 0]
    tip = (scene.attr("lock", 1)[f[:, 0], 0] > 0) & (left[f].min(axis=1) < FRAY)
    remain = interpolate(left[:, None], f, fr)[..., 0]
    strand = interpolate(scene.attr("strand", 1), f, fr)[..., 0]
    fray = value_noise(strand * 0.8, 21) ** 1.5
    gone = fr.seen & tip[np.maximum(fr.tid, 0)] & (remain < 0.05 + FRAY * fray)
    keep = np.nonzero(~tip)[0]
    behind = raster(v, f[keep], view)
    tid = np.where(
        gone, np.where(behind.seen, keep[np.maximum(behind.tid, 0)], -1), fr.tid
    )
    depth = np.where(gone, behind.depth, fr.depth)
    bary = np.where(gone[..., None], behind.bary, fr.bary)
    return Frame(depth, tid, bary)


def see(model: Model, view: View) -> Buffers:
    v, f, kind, vn, loc = model.scene.merged()
    fr = fray_ends(model, v, f, raster(v, f, view), view)
    pos = interpolate(v, f, fr)
    n = unit_rows(interpolate(vn, f, fr))
    k = np.where(fr.seen, kind[np.maximum(fr.tid, 0)], -1)
    centre = model.pose.of_head(HAIR_CENTRE)
    outward = np.sign(np.sum(n * (pos - centre), axis=-1))
    inside = (k == K_HAIR) & (outward * n[..., 2] < 0)
    n = n * np.where(n[..., 2:3] < 0, -1, 1)
    flow = interpolate(model.scene.attr("flow", 3), f, fr)
    flow = unit_rows(flow - np.sum(flow * n, axis=-1, keepdims=True) * n)
    hair_u = interpolate(model.scene.attr("hair_u", 1), f, fr)[..., 0]
    strand = interpolate(model.scene.attr("strand", 1), f, fr)[..., 0]
    tri = f[np.maximum(fr.tid, 0), 0]
    lock = np.where(fr.seen, model.scene.attr("lock", 1)[tri, 0], 0).astype(np.int32)
    shine = interpolate(model.scene.attr("shine", 1), f, fr)[..., 0]
    local = interpolate(loc, f, fr)
    return Buffers(
        fr.seen, k, fr.depth, pos, local, n, inside, flow, hair_u, strand, lock, shine
    )


# ------------------------------------------------- features drawn on the face

#: The rims of the eye openings, inner corner to outer: upper, lower.
RIM_R = (
    [133, 173, 157, 158, 159, 160, 161, 246, 33],
    [133, 155, 154, 153, 145, 144, 163, 7, 33],
)
RIM_L = tuple([EYE_L[EYE_R.index(i)] for i in rim] for rim in RIM_R)


def closed_curve(pts, n=200):
    return catmull_rom(np.vstack([pts, pts[:1]]), n)


def polyline_distance(q, line):
    """Distance of points q (m, 2) to a polyline (k, 2), and the position
    along it (0 to 1) of the nearest point."""
    best = np.full(len(q), np.inf)
    where = np.zeros(len(q))
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    acc = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-12)
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        ab = b - a
        t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.linalg.norm(q - (a + t[:, None] * ab), axis=1)
        closer = d < best
        best[closer] = d[closer]
        where[closer] = (acc[i] + t * (acc[i + 1] - acc[i]))[closer]
    return best, where


def stroke(q, line, width, soft):
    """Coverage of points q by a line of varying width (a function of the
    position along it), with soft edges."""
    d, t = polyline_distance(q, line)
    return np.clip((width(t) / 2 - d) / soft + 0.5, 0, 1)


@dataclass
class Features:
    """Shapes on the face, in head space (x, y), applied as decals."""

    lips: np.ndarray
    mouth: np.ndarray
    brows: list
    upper: list
    lower: list
    crease: list


def brow_shape(v, idx, tilt=0.0):
    """A young woman's brow: a soft head by the nose, a gentle arch two
    thirds of the way out, a fine tail."""
    lower, upper = v[idx[:5], :2][::-1], v[idx[5:], :2][::-1]
    mid = catmull_rom((lower + upper) / 2, 40)[:36]
    t = np.linspace(0, 1, 36)[:, None]
    arch = np.exp(-0.5 * ((t - 0.66) / 0.2) ** 2) - 0.5 * t**3
    line = mid * [1, 0] + [0, 1] * (
        mid[:, 1:].mean() + 0.0 * (mid[:, 1:] - mid[:, 1:].mean())
    )
    # ``tilt`` raises the end by the nose, to set the brow level in the
    # picture against the turn and tilt of the head
    return line + [0, 0.08] * arch + [0, 1] * (tilt * (1 - t)) - [0, 0.28]


def brow_width(t):
    return (0.44 - 0.32 * t**1.6) * (0.8 + 0.2 * np.sqrt(np.clip(t * 5, 0, 1)))


def brow_strength(t):
    head = np.clip(t / 0.35, 0, 1)
    tail = np.clip((1 - t) / 0.2, 0, 1)
    return (0.25 + 0.75 * head * head * (3 - 2 * head)) * (0.5 + 0.5 * tail)


def lid_crease(v, upper):
    pts = v[upper]
    t = np.linspace(0, 1, len(upper))[:, None]
    lift = 0.07 + 0.3 * np.sin(np.pi * t) ** 0.8
    return catmull_rom(pts[:, :2] + np.hstack([np.zeros_like(t), lift]), 40)


def face_features(v) -> Features:
    up = v[[78] + UPPER_INNER + [308], :2]
    lo = v[[78] + LOWER_INNER + [308], :2]
    rims = (RIM_R, RIM_L)
    return Features(
        lips=closed_curve(v[LIP_OUTER, :2]),
        mouth=catmull_rom((up + lo) / 2, 60),
        brows=[brow_shape(v, BROW_R, 0.08), brow_shape(v, BROW_L, -0.02)],
        upper=[catmull_rom(v[r[0], :2], 60) for r in rims],
        lower=[catmull_rom(v[r[1], :2], 60) for r in rims],
        crease=[lid_crease(v, r[0]) for r in rims],
    )


# --------------------------------------------------------------- the shading


def to_linear(hex_colour):
    c = np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


COLOURS = {
    BACKGROUND: "#fbfaf7",
    SKIN: "#eec9b3",
    LIPS: "#c46a6e",
    EYE_WHITE: "#eae6e2",
    IRIS: "#4e3122",
    PUPIL: "#141010",
    BROW: "#2e1f1b",
    HAIR: "#201716",
    GARMENT: "#3d5488",
}
LASH = to_linear("#1c1515")
#: The colour of the light in the shadows of the skin, which has passed
#: through it.
WARM = np.array([1.25, 0.82, 0.75])


@dataclass
class Eyes:
    """Per pixel on the eyeballs: iris and pupil masks, the iris's own
    pattern, and the catchlight."""

    white: np.ndarray
    iris: np.ndarray
    pupil: np.ndarray
    pattern: np.ndarray
    glint: np.ndarray


def gaze(pose):
    """The eyes look at the viewer, a touch less turned than that."""
    return unit_rows(0.3 * pose.head_dir(np.array([0.0, 0.0, 1.0])) + 0.7 * VIEWER)


def iris_pattern(q, g, cos_i):
    """Radial fibres, a dark rim and a lighter lower side (the light
    passing through the cornea comes out opposite it)."""
    up = unit_rows(np.cross(np.cross(g, [0.0, 1.0, 0.0]), g))
    side = np.cross(g, up)
    a = np.arctan2(q @ side, q @ up)
    r = np.sqrt(np.clip(1 - (q @ g) ** 2, 0, 1)) / math.sqrt(1 - cos_i**2)
    fib = 1 + 0.12 * np.sin(23 * a) * np.sin(7 * a + 1) + 0.08 * np.sin(41 * a + 2)
    rim = 1 - 0.55 * np.clip((r - 0.78) / 0.22, 0, 1) ** 1.5
    inner = 1 - 0.25 * np.clip((0.42 - r) / 0.2, 0, 1)
    low = 1 + 0.9 * np.clip(-(q @ up) / math.sqrt(1 - cos_i**2), 0, 1) * (r > 0.35)
    return fib * rim * inner * low


def see_eyes(b: Buffers, model: Model) -> Eyes:
    shape = b.seen.shape
    no = np.zeros(shape, dtype=bool)
    out = Eyes(no.copy(), no.copy(), no.copy(), np.ones(shape), np.zeros(shape))
    g = gaze(model.pose)
    cos_i = math.sqrt(1 - (IRIS_RADIUS / EYE_RADIUS) ** 2)
    cos_p = math.sqrt(1 - (PUPIL_RADIUS / EYE_RADIUS) ** 2)
    half = unit_rows(KEY + VIEWER)
    for eye in model.eyes:
        c = model.pose.of_head(eye.centre)
        ball = (b.kind == K_BALL) & (
            np.linalg.norm(b.pos - c, axis=-1) < EYE_RADIUS + 0.02
        )
        q = unit_rows(b.pos[ball] - c)
        ca = q @ g
        iris, pupil = ca > cos_i, ca > cos_p
        out.white[ball] = ~iris
        out.iris[ball] = iris & ~pupil
        out.pupil[ball] = pupil
        out.pattern[ball] = np.where(iris, iris_pattern(q, g, cos_i), 1.0)
        # the catchlight, off a cornea bulging from the ball
        rc = 0.8
        cc = c + g * (EYE_RADIUS - rc + 0.07)
        dxy = (b.pos[ball] - cc)[:, :2]
        nz = np.sqrt(np.clip(rc**2 - np.sum(dxy**2, axis=1), 0, None))
        cn = unit_rows(np.hstack([dxy, nz[:, None]]))
        out.glint[ball] = np.clip((cn @ half - 0.9978) / 0.0006, 0, 1) * iris
    return out


def neckline(x):
    """Height of a soft scoop neckline across the body (body space)."""
    t = np.clip(np.abs(x) / 10.8, 0, 1)
    return np.where(np.abs(x) < 10.8, -21.8 + 5.0 * t**2.2, 99.0)


def decal_points(b: Buffers):
    m = np.isin(b.kind, (K_FACE, K_LID, K_BALL, K_MOUTH))
    return m, b.local[m][:, :2]


def flick(line, length=0.22, rise=0.1):
    """The outer end of a lash line, carried on and up a little."""
    d = unit_rows(line[-1] - line[-4])
    tip = line[-1] + d * length + np.array([0.0, rise])
    return np.vstack([line, line[-1] + (tip - line[-1]) * 0.5, tip])


def lashes(q, line, count=34):
    """A soft fringe of lashes over the upper lid line, longer and denser
    towards the outer corner, each leaning outwards."""
    d, t = polyline_distance(q, line)
    k = np.clip(t * (len(line) - 1), 0, len(line) - 1).astype(np.int64)
    above = q[:, 1] > line[k, 1]
    length = 0.1 + 0.22 * t**1.2
    u = np.clip(d / length, 0, 1)
    comb = 0.5 + 0.5 * np.cos(2 * np.pi * (t * count - 0.8 * u * (0.5 + t)))
    taper = (1 - u) ** 1.2 * (d < length)
    return np.clip(
        2.2 * above * taper * (0.4 + 0.6 * comb**2) * (0.45 + 0.55 * t), 0, 1
    )


def face_marks(b: Buffers, feat: Features):
    """Darkenings drawn on the face: lashes, lid creases, the line between
    the lips.  Returns (strength 0 to 1, colour) per face pixel."""
    m, q = decal_points(b)
    dark = np.zeros(len(q))
    for up, lo, cr in zip(feat.upper, feat.lower, feat.crease, strict=True):
        dark = np.maximum(
            dark, 0.92 * stroke(q, flick(up), lambda t: 0.05 + 0.12 * t**1.3, 0.025)
        )
        dark = np.maximum(dark, 0.85 * lashes(q, up))
        dark = np.maximum(dark, 0.3 * stroke(q, lo, lambda t: 0.015 + 0.035 * t, 0.02))
        dark = np.maximum(dark, 0.5 * stroke(q, cr, lambda t: 0.06 + 0.0 * t, 0.05))
    mouth = stroke(
        q, feat.mouth, lambda t: 0.025 + 0.05 * np.sin(np.pi * t).clip(0) ** 0.6, 0.02
    )
    dark = np.maximum(dark, 0.75 * mouth)
    out = np.zeros(b.seen.shape)
    out[m] = dark
    out[b.kind == K_MOUTH] = 0.8
    return out


def regions(b: Buffers, feat: Features, eyes: Eyes):
    """The region of each pixel, and how strongly the brows cover it."""
    region = np.where(b.seen, SKIN, BACKGROUND)
    region[b.kind == K_HAIR] = HAIR
    garment = (b.kind == K_BODY) & (b.local[..., 1] < neckline(b.local[..., 0]))
    region[garment] = GARMENT
    face = b.kind == K_FACE
    x, y = b.local[..., 0], b.local[..., 1]
    region[face & polygon_inside(x, y, feat.lips)] = LIPS
    brow = np.zeros(b.seen.shape)
    q = b.local[face][:, :2]
    for line in feat.brows:
        d, t = polyline_distance(q, line)
        cov = np.clip((brow_width(t) / 2 - d) / 0.12 + 0.5, 0, 1)
        cov *= 0.75 + 0.25 * value_noise(t * 90 + 7 * d, 9)
        brow[face] = np.maximum(brow[face], cov * brow_strength(t))
    region[brow > 0.3] = BROW
    region[eyes.white] = EYE_WHITE
    region[eyes.iris] = IRIS
    region[eyes.pupil] = PUPIL
    return region, brow


def albedo(region, brow, eyes: Eyes, marks):
    out = np.zeros(region.shape + (3,))
    for r, hexc in COLOURS.items():
        out[region == r] = to_linear(hexc)
    out *= np.where(region == IRIS, eyes.pattern, 1.0)[..., None]
    a = np.clip(0.8 * brow * np.isin(region, (SKIN, BROW)), 0, 0.8)[..., None]
    out = np.where(
        a > 0, (1 - a) * to_linear(COLOURS[SKIN]) + a * to_linear(COLOURS[BROW]), out
    )
    m = marks[..., None]
    return out * (1 - m) + LASH * m


@dataclass
class Lighting:
    diffuse: np.ndarray  # light falling on each point, before its colour
    spec: np.ndarray  # light reflected off it
    shadow: np.ndarray
    ao: np.ndarray
    soft: np.ndarray  # the part of ``diffuse`` from the fill and the room


def value_noise(x, seed):
    table = np.random.default_rng(seed).random(4096)
    i = np.floor(x).astype(np.int64)
    t = x - i
    t = t * t * (3 - 2 * t)
    return table[i % 4096] * (1 - t) + table[(i + 1) % 4096] * t


def strands(b: Buffers):
    """A texture of strands and clumps across the hair (0 to about 1)."""
    c = b.strand
    fine = value_noise(c * 3.1, 1) * 0.5 + value_noise(c * 7.3, 2) * 0.3
    clump = value_noise(c / 5.0, 3)
    return 0.35 * fine + 0.65 * clump * (0.6 + 0.4 * fine)


def hair_light(b: Buffers, shadow, ao):
    """Diffuse plus two highlights across the strands (shifted along the
    normal, as off the scales of a hair)."""
    t = b.flow
    n = b.normal
    half = unit_rows(KEY + VIEWER)
    kk = np.sqrt(np.clip(1 - np.sum(t * KEY, axis=-1) ** 2, 0, 1))
    ndl = np.clip(n @ KEY, 0, 1)
    diff = (0.55 * ndl + 0.45 * kk * np.clip(n @ KEY + 0.4, 0, 1)) * shadow
    spec = np.zeros(ndl.shape)
    for shift, power, k in ((-0.1, 160, 0.3), (0.15, 40, 0.04)):
        ts = unit_rows(t + shift * n)
        th = np.sum(ts * half, axis=-1)
        spec += k * np.sqrt(np.clip(1 - th**2, 0, 1)) ** power
    tex = strands(b)
    spec *= shadow * np.clip(ndl * 2.5, 0, 1) * ao * (0.25 + 1.3 * tex) * b.shine
    # a faint sheen from the fill, on the side away from the key
    th = np.sum(unit_rows(t - 0.1 * n) * unit_rows(FILL + VIEWER), axis=-1)
    sheen = (
        0.012 * np.sqrt(np.clip(1 - th**2, 0, 1)) ** 60 * np.clip(n @ FILL, 0, 1) * ao
    )
    return diff * (0.55 + 0.9 * tex), spec + sheen * (0.3 + tex)


def lighting(b: Buffers, region, shadow, ao) -> Lighting:
    n = b.normal
    ndl = n @ KEY
    wrap = np.clip((ndl + 0.3) / 1.3, 0, 1) ** 1.2
    # light passes partly through hair, and young skin takes soft shadows
    skin = np.isin(region, (SKIN, LIPS, BROW)) & (b.kind != K_BODY)
    shadow = np.where(skin, 0.4 + 0.6 * shadow, shadow)
    key = np.where(region == GARMENT, np.clip(ndl, 0, 1), wrap) * shadow
    fill = 0.2 * np.clip(n @ FILL, 0, 1) * ao
    amb = 0.1 * ao * (0.6 + 0.4 * n[..., 1])
    diff = key + fill + amb
    half = unit_rows(KEY + VIEWER)
    ndh = np.clip(n @ half, 0, 1)
    gloss = {
        SKIN: (0.035, 18),
        LIPS: (0.16, 50),
        EYE_WHITE: (0.08, 40),
        BROW: (0.01, 10),
    }
    spec = np.zeros(ndl.shape)
    for r, (k, p) in gloss.items():
        spec = np.where(region == r, k * ndh**p * shadow, spec)
    hd, hs = hair_light(b, shadow, ao)
    hair = region == HAIR
    diff = np.where(hair, hd + fill + amb, diff)
    spec = np.where(hair, hs, spec)
    inside = b.inside
    diff = np.where(inside, 0.35 * diff, diff)
    spec = np.where(inside, 0.0, spec)
    # the turn of the jaw and the chin underneath, in soft shadow
    dim = np.where(hair, 1.0, ao**0.5) * np.where(b.kind == K_ROLL, 0.85, 1.0)
    return Lighting(diff * dim, spec, shadow, ao, np.where(hair, 0, fill + amb) * dim)


# --------------------------------------------------------------- the studies


@dataclass
class Picture:
    b: Buffers
    region: np.ndarray
    light: Lighting
    colour: np.ndarray  # linear RGB


def picture(model: Model, view: View) -> Picture:
    b = see(model, view)
    feat = face_features(model.face[0])
    eyes = see_eyes(b, model)
    v, f, *_ = model.scene.merged()
    lift = np.where(b.kind == K_HAIR, 0.12, 0.0)
    shadow = shadows(v, f, b.pos, b.normal, b.seen, lift)
    shadow = filters.box_blur(shadow, max(1, int(view.scale / 40)))
    ao = occlusion(b.depth, b.seen, view.scale)
    region, brow = regions(b, feat, eyes)
    marks = face_marks(b, feat)
    lt = lighting(b, region, shadow, ao)
    warm = 1 + (WARM - 1) * (lt.soft / np.maximum(lt.diffuse, 1e-6))[..., None]
    warm = np.where(
        (region == SKIN)[..., None] | (region == LIPS)[..., None], warm, 1.0
    )
    colour = (
        albedo(region, brow, eyes, marks) * lt.diffuse[..., None] * warm
        + lt.spec[..., None]
    )
    colour += 1.6 * eyes.glint[..., None] * (shadow[..., None] > 0.3)
    colour[~b.seen] = to_linear(COLOURS[BACKGROUND])
    return Picture(b, region, lt, colour)


def scaled(view: View, k: int) -> View:
    return View(
        (view.size[0] * k, view.size[1] * k),
        view.scale * k,
        (view.centre[0] * k, view.centre[1] * k),
    )


def shrink(img, k):
    h, w = img.shape[0] // k, img.shape[1] // k
    return img[: h * k, : w * k].reshape(h, k, w, k, *img.shape[2:]).mean(axis=(1, 3))


def as_png(lin):
    return np.round(to_srgb(lin) * 255).astype(np.uint8)


def grey(lin):
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    return np.repeat(y[..., None], 3, axis=-1)


def flat_colours(region):
    out = np.zeros(region.shape + (3,))
    for r, hexc in COLOURS.items():
        out[region == r] = to_linear(hexc)
    return out


def crop_around(img, pts, view: View, k, pad):
    """A crop of a k-times render round head-space points."""
    x = (view.centre[0] + pts[:, 0] * view.scale) * k
    y = (view.centre[1] - pts[:, 1] * view.scale) * k
    x0, x1 = int(x.min() - pad * k), int(x.max() + pad * k)
    y0, y1 = int(y.min() - pad * k), int(y.max() + pad * k)
    return img[max(y0, 0) : y1, max(x0, 0) : x1]


def write_studies(model: Model, out: Path, ss=2):
    """The grey and colour studies (rendered at ``ss`` times the size and
    shrunk), the regions in flat colours, and enlarged crops."""
    pic = picture(model, scaled(VIEW, ss))
    colour = shrink(pic.colour, ss)
    write_png(str(out / "study_grey.png"), as_png(grey(colour)))
    write_png(str(out / "study_colour.png"), as_png(colour))
    write_png(str(out / "study_flat.png"), as_png(flat_colours(pic.region[::ss, ::ss])))
    v = model.face[0]
    eyes = model.pose.of_head(v[EYE_R + EYE_L])
    mouth = model.pose.of_head(v[LIP_OUTER])
    write_png(
        str(out / "eyes.png"), as_png(crop_around(pic.colour, eyes, VIEW, ss, 40))
    )
    write_png(
        str(out / "mouth.png"), as_png(crop_around(pic.colour, mouth, VIEW, ss, 40))
    )


# ------------------------------------------------- what the pen work follows


def to_pixels(view: View, p):
    return np.stack(
        [view.centre[0] + p[:, 0] * view.scale, view.centre[1] - p[:, 1] * view.scale],
        1,
    )


def visible(view: View, depth, p, tol=0.08):
    """Which camera-space points are not hidden behind something else."""
    xy = to_pixels(view, p).astype(np.int64)
    h, w = depth.shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    out = np.zeros(len(p), dtype=bool)
    xi, yi = xy[inside, 0], xy[inside, 1]
    out[inside] = p[inside, 2] >= depth[yi, xi] - tol
    return out


def lift_to_face(model: Model, line):
    """Head-space points on the face under a line drawn in (x, y)."""
    v = model.face[0]
    front = v[:, 2] > 0
    out = []
    for q in line:
        d = np.sum((v[front, :2] - q) ** 2, axis=1)
        near = np.argsort(d)[:4]
        wgt = 1 / (d[near] + 1e-4)
        out.append([q[0], q[1], (v[front][near, 2] * wgt).sum() / wgt.sum()])
    return np.array(out)


def iris_rims(model: Model):
    g = gaze(model.pose)
    up = unit_rows(np.cross(np.cross(g, [0.0, 1.0, 0.0]), g))
    side = np.cross(g, up)
    sin_i = IRIS_RADIUS / EYE_RADIUS
    th = np.linspace(0, 2 * np.pi, 73)[:, None]
    ring = math.sqrt(1 - sin_i**2) * g + sin_i * (np.cos(th) * up + np.sin(th) * side)
    return [model.pose.of_head(e.centre) + EYE_RADIUS * ring for e in model.eyes]


def contours(model: Model, b: Buffers, view: View):
    """The features' lines as polylines in picture pixels, each point with
    a flag saying whether it is seen: {name: (k, 3) array of x, y, seen}."""
    feat = face_features(model.face[0])
    lines = {"lips": feat.lips, "mouth": feat.mouth}
    for side, i in (("right", 0), ("left", 1)):
        lines[f"brow_{side}"] = feat.brows[i]
        lines[f"lid_upper_{side}"] = feat.upper[i]
        lines[f"lid_lower_{side}"] = feat.lower[i]
        lines[f"crease_{side}"] = feat.crease[i]
    out = {}
    for name, line in lines.items():
        p = model.pose.of_head(lift_to_face(model, line))
        out[name] = np.hstack(
            [to_pixels(view, p), visible(view, b.depth, p, 0.2)[:, None]]
        )
    oval = closed_curve(model.face[0][OVAL], 240)
    for name, p in [("face_oval", model.pose.of_head(oval))] + [
        (f"iris_{s}", r)
        for s, r in zip(("right", "left"), iris_rims(model), strict=True)
    ]:
        out[name] = np.hstack(
            [to_pixels(view, p), visible(view, b.depth, p, 0.15)[:, None]]
        )
    return out


#: The eight neighbours of a pixel, clockwise from the west: (dx, dy).
RING = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


def trace(mask, start, limit=200000):
    """The boundary of the shape in ``mask`` through ``start`` (a pixel
    whose west neighbour is outside), followed clockwise (Moore)."""
    h, w = mask.shape
    path = [start]
    p, back = start, 0  # we came from the west
    for _ in range(limit):
        for i in range(1, 9):
            k = (back + i) % 8
            x, y = p[0] + RING[k][0], p[1] + RING[k][1]
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                back = (k + 4) % 8  # look round from where we came
                p = (x, y)
                break
        else:
            return path
        if p == start:
            return path
        path.append(p)
    return path


def outlines(mask, min_len=80, most=12):
    """The boundaries of the larger shapes of a mask, as polylines (x, y)."""
    starts = mask & ~np.roll(mask, 1, axis=1)
    starts[:, 0] = mask[:, 0]
    done = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(starts)
    for x, y in zip(xs, ys, strict=True):
        if done[y, x]:
            continue
        path = np.array(trace(mask, (int(x), int(y))))
        done[path[:, 1], path[:, 0]] = True
        if len(path) >= min_len:
            out.append(path[::2].astype(float) + 0.5)
    out.sort(key=len, reverse=True)
    return out[:most]


def silhouettes(b: Buffers, region):
    """Outer edges of the face, the neck and shoulders, the hair and the
    garment, as polylines in picture pixels (all seen)."""
    head = np.isin(b.kind, (K_FACE, K_LID, K_BALL, K_MOUTH, K_ROLL))
    masks = {
        "outline_face": head,
        "outline_neck": (b.kind == K_BODY) & (region == SKIN),
        "outline_hair": region == HAIR,
        "outline_garment": region == GARMENT,
    }
    out = {}
    for name, m in masks.items():
        for i, line in enumerate(outlines(m)):
            out[f"{name}_{i}"] = np.hstack([line, np.ones((len(line), 1))])
    return out


def image_flow(b: Buffers):
    """The hair's direction in the picture, as unit (dx, dy) in pixels (y
    down); zero off the hair."""
    d = np.stack([b.flow[..., 0], -b.flow[..., 1]], axis=-1)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    return np.where((b.kind == K_HAIR)[..., None], d, 0.0)


def write_maps(model: Model, out: Path):
    """Per-pixel maps at the picture's size, and the feature lines."""
    pic = picture(model, VIEW)
    b = pic.b
    np.savez_compressed(
        out / "maps.npz",
        region=pic.region.astype(np.uint8),
        region_names=np.array(REGION_NAMES),
        tone=(pic.colour @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32),
        light=np.clip(pic.light.diffuse + pic.light.spec, 0, 2).astype(np.float32),
        shadow=pic.light.shadow.astype(np.float32),
        normal=np.where(b.seen[..., None], b.normal, 0).astype(np.float32),
        flow=image_flow(b).astype(np.float32),
        hair_u=b.hair_u.astype(np.float32),
        depth=np.where(b.seen, b.depth, np.nan).astype(np.float32),
    )
    lines = contours(model, b, VIEW) | silhouettes(b, pic.region)
    np.savez_compressed(out / "contours.npz", **lines)


# ------------------------------------------------------------ the pen work

PAPER = papers.BRISTOL.with_(color="#fbfaf7")
RED = pens.ballpoint(pens.RED.ink.hex, width=2.0)
BLUE = pens.ballpoint(pens.BLUE.ink.hex, width=2.0)
BLACK = pens.ballpoint(pens.BLACK.ink.hex, width=2.0)
BOLD = pens.BLACK  # the 3 px ball, for the strongest lines
#: What the drawing stops at, by region: each pixel's part.
P_PAPER, P_FACE, P_NECK, P_LIPS, P_WHITE, P_IRIS, P_HAIR, P_GARMENT = range(8)
#: Down-sampling of the maps that keep count of the ink laid.
CELL = 4


@dataclass
class Maps:
    """What the pen work reads off the model, per pixel of the picture."""

    part: np.ndarray  # int8, P_*
    lock: np.ndarray  # int32: the lock of hair seen, 0 where none
    shade: np.ndarray  # 0 in full light to 1 in the deepest shadow
    tone: np.ndarray  # the shaded study's luminance (linear)
    spec: np.ndarray  # the highlight's strength
    local: np.ndarray  # (h, w, 3) the point before the pose
    around: np.ndarray  # (h, w, 3) the surface's direction round its axis
    along: np.ndarray  # (h, w, 3) and along it
    flow: np.ndarray  # (h, w, 2) the hair's direction in the picture
    brow: np.ndarray  # bool: under the brows
    facing: np.ndarray  # how squarely the surface faces the viewer (0 to 1)
    lines: dict  # polylines in pixels: (k, 3) x, y, seen
    eyes: list  # per eye: (pupil centre, iris radius) in pixels
    form: np.ndarray  # shade under the key alone: 0 lit to 1 turned away
    depth: np.ndarray  # towards the viewer, of what is seen
    strands: list  # posed grids of hair (rows root to tip, columns)
    nose: dict  # head points of the nose, in pixels


def part_map(pic: Picture, eyes: Eyes) -> np.ndarray:
    r, k = pic.region, pic.b.kind
    part = np.full(r.shape, P_PAPER, np.int8)
    part[np.isin(r, (SKIN, BROW))] = P_FACE
    part[(r == SKIN) & (k == K_BODY)] = P_NECK
    part[r == LIPS] = P_LIPS
    part[r == EYE_WHITE] = P_WHITE
    part[np.isin(r, (IRIS, PUPIL))] = P_IRIS
    part[r == HAIR] = P_HAIR
    part[r == GARMENT] = P_GARMENT
    part[eyes.glint > 0.3] = P_PAPER  # the catchlights stay paper
    return part


def surface_frame(model: Model, b: Buffers):
    """Unit tangents of the surface: round the head's (or the body's) up
    axis, and along it."""
    up_head = model.pose.head_dir(np.array([0.0, 1.0, 0.0]))
    up_body = model.pose.of_body(np.array([0.0, 1.0, 0.0])) - model.pose.of_body(
        np.zeros(3)
    )
    up = np.where((b.kind == K_BODY)[..., None], up_body, up_head)
    around = unit_rows(np.cross(up, b.normal))
    along = unit_rows(np.cross(b.normal, around))
    return around, along


def pupils(model: Model, glint):
    """Per eye: the iris's centre and radius in pixels, and where its
    catchlight is."""
    g = gaze(model.pose)
    ys, xs = np.nonzero(glint > 0.3)
    out = []
    for e in model.eyes:
        c = model.pose.of_head(e.centre)
        front = c + g * math.sqrt(EYE_RADIUS**2 - IRIS_RADIUS**2)
        xy = to_pixels(VIEW, front[None])[0]
        r = IRIS_RADIUS * VIEW.scale
        near = (xs - xy[0]) ** 2 + (ys - xy[1]) ** 2 < r * r
        spot = np.array([xs[near].mean(), ys[near].mean()]) if near.any() else xy
        out.append((xy, r, spot))
    return out


#: Where the nose is, in head space: |x| below, y between, z above.
NOSE_BOX = (2.2, -2.4, 3.4, 4.2)


def contour_segments(v, f, g):
    """Segments where ``g`` (per vertex) changes sign across triangles,
    each end keyed by the edge it lies on."""
    segs = []
    for tri in f:
        ends = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if (g[a] > 0) != (g[b] > 0):
                t = g[a] / (g[a] - g[b])
                ends.append(((min(a, b), max(a, b)), v[a] + t * (v[b] - v[a])))
        if len(ends) == 2:
            segs.append(ends)
    return segs


def chain(segs):
    """Join segments that share an edge into polylines."""
    by_edge = {}
    for i, sg in enumerate(segs):
        for key, _ in sg:
            by_edge.setdefault(key, []).append(i)
    used, out = set(), []
    for i in range(len(segs)):
        if i in used:
            continue
        used.add(i)
        line = [segs[i][0], segs[i][1]]
        for end in (1, 0):
            while True:
                key = line[-1][0] if end else line[0][0]
                nxt = [j for j in by_edge[key] if j not in used]
                if not nxt:
                    break
                j = nxt[0]
                used.add(j)
                a, b = segs[j]
                new = b if a[0] == key else a
                if end:
                    line.append(new)
                else:
                    line.insert(0, new)
        out.append(np.array([p for _, p in line]))
    return out


def nose_lines(model: Model, b: Buffers):
    """The nose's outline where it turns away from the viewer, as seen."""
    sc = model.scene
    v, n, f, loc = sc.v[0], sc.n[0], sc.f[0], sc.local[0]
    c = loc[f].mean(axis=1)
    bx, y0, y1, z0 = NOSE_BOX
    keep = (np.abs(c[:, 0]) < bx) & (c[:, 1] > y0) & (c[:, 1] < y1) & (c[:, 2] > z0)
    out = {}
    lines = [ln for ln in chain(contour_segments(v, f[keep], n[:, 2])) if len(ln) > 4]
    for k, ln in enumerate(lines):
        seen = visible(VIEW, b.depth, ln, 0.12)
        out[f"nose_{k}"] = np.hstack([to_pixels(VIEW, ln), seen[:, None]])
    return out


def pen_maps(model: Model) -> Maps:
    pic = picture(model, VIEW)
    b = pic.b
    eyes = see_eyes(b, model)
    lit = pic.light.diffuse
    shade = np.clip(1.0 - lit / 1.05, 0, 1)
    around, along = surface_frame(model, b)
    lines = contours(model, b, VIEW) | silhouettes(b, pic.region)
    lines |= nose_lines(model, b)
    lines["subnasale"] = np.hstack(
        [to_pixels(VIEW, model.pose.of_head(model.face[0][[2]])), [[1.0]]]
    )
    return Maps(
        part_map(pic, eyes),
        b.lock,
        shade.astype(np.float32),
        (pic.colour @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32),
        pic.light.spec.astype(np.float32),
        b.local.astype(np.float32),
        around.astype(np.float32),
        along.astype(np.float32),
        image_flow(b).astype(np.float32),
        pic.region == BROW,
        np.clip(b.normal[..., 2], 0, 1).astype(np.float32),
        lines,
        pupils(model, eyes.glint),
        form_shade(b, pic.light.shadow),
        np.where(b.seen, b.depth, -1e9).astype(np.float32),
        model.strands,
        nose_points(model),
    )


def form_shade(b: Buffers, shadow):
    """How far each point turns from the key light, or lies in its shadow."""
    lit = np.clip(b.normal @ KEY, 0, 1) * shadow
    return (1 - lit).astype(np.float32)


#: Points of the nose: its wings (sitter's right, left), and under it.
NOSE_POINTS = {"wing_r": 129, "wing_l": 358, "under": 2, "tip": 4, "bridge": 6}


def nose_points(model: Model):
    """Pixel places of points of the nose, and the creases round its wings
    as lines (x, y, seen) for the contours."""
    v = model.face[0]
    out = {}
    for k, i in NOSE_POINTS.items():
        out[k] = to_pixels(VIEW, model.pose.of_head(v[[i]]))[0]
    out["stomion_y"] = float(v[[13, 14], 1].mean())
    for name, i in (("wing_r", 129), ("wing_l", 358)):
        c = v[i, :2] + [math.copysign(0.12, v[i, 0]), 0.1]
        side = math.copysign(1, v[i, 0])
        a = np.radians(np.linspace(-30, 110, 24))
        arc = c + 0.5 * np.stack([side * np.cos(a), np.sin(a)], 1)
        out[name + "_crease"] = model.pose.of_head(lift_to_face(model, arc))
    return out


@njit(cache=False)
def _trace(
    x, y, dx, dy, field, part, want, sub, target, tmin, turn, step, nmax, out, full
):
    """Follow the direction field from (x, y), setting off along (dx, dy)
    and turned by ``turn`` (radians) from the field, while on the part
    ``want``, where the target is at least ``tmin`` and no other line of
    the layer is near (``full`` is 0).  A change of ``sub``
    (the lock of hair) ends it after a few steps.  Writes the points to
    ``out`` and returns how many."""  # pragma: no cover
    h, w = part.shape
    c, s = math.cos(turn), math.sin(turn)
    first = sub[min(max(int(y), 0), h - 1), min(max(int(x), 0), w - 1)]
    strays, crowd, n = 0, 0, 0
    for _ in range(nmax):
        xi, yi = int(x), int(y)
        if xi < 0 or yi < 0 or xi >= w or yi >= h or part[yi, xi] != want:
            break
        if target[yi, xi] < tmin:
            break
        crowd = crowd + 1 if full[yi, xi] else 0
        strays = strays + 1 if sub[yi, xi] != first else 0
        if strays > 4 or crowd > 6:
            break
        out[n, 0], out[n, 1] = x, y
        n += 1
        fx, fy = field[yi, xi, 0], field[yi, xi, 1]
        gx, gy = fx * c - fy * s, fx * s + fy * c
        if gx * dx + gy * dy < 0:
            gx, gy = -gx, -gy
        dx, dy = 0.75 * dx + 0.25 * gx, 0.75 * dy + 0.25 * gy
        norm = math.sqrt(dx * dx + dy * dy) + 1e-12
        dx, dy = dx / norm, dy / norm
        x += dx * step
        y += dy * step
    return n


@njit(cache=False)
def _stamp(occ, xs, ys, radius):  # pragma: no cover
    """Mark the pixels within ``radius`` of the points: the room a line
    keeps from the next one of its layer."""
    h, w = occ.shape
    r = int(math.ceil(radius))
    r2 = radius * radius
    for i in range(xs.shape[0]):
        cx, cy = xs[i], ys[i]
        for y in range(max(int(cy) - r, 0), min(int(cy) + r + 1, h)):
            for x in range(max(int(cx) - r, 0), min(int(cx) + r + 1, w)):
                if (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r2:
                    occ[y, x] = 1


def line_weight(p):
    """How much a line of the 2 px ball darkens, times its width (px), at
    pressure ``p``: measured on hatching."""
    return 1.25 * np.asarray(p) ** 0.55


@dataclass
class Pass:
    """One layer of hatching: an ink over a part, at an angle to the
    surface's round direction, up to ``cap`` of coverage beyond ``floor``."""

    pen: pens.Pen
    part: int
    angle: float  # degrees from the direction round the form
    floor: float
    cap: float
    pressure: float
    length: tuple = (60, 200)
    count: tuple = (5, 14)
    spacing: tuple = (3.0, 13.0)
    shortest: int = 22  # points of 2 px


@dataclass
class Hatcher:
    """Lays hatching patch by patch on a sheet, keeping count of the ink."""

    sheet: Sheet
    maps: Maps
    rng: np.random.Generator
    strokes: int = 0
    full: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.uint8))

    def field(self, angle):
        """The direction in the picture of the surface tangent ``angle``
        degrees from the one round the form (for the hair: its flow)."""
        m = self.maps
        a = math.radians(angle)
        t = math.cos(a) * m.around + math.sin(a) * m.along
        d = np.stack([t[..., 0], -t[..., 1]], axis=-1)
        d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
        hair = (m.part == P_HAIR)[..., None]
        c, s = math.cos(a), math.sin(a)
        f = np.stack(
            [
                m.flow[..., 0] * c - m.flow[..., 1] * s,
                m.flow[..., 0] * s + m.flow[..., 1] * c,
            ],
            axis=-1,
        )
        return np.where(hair, f, d).astype(np.float32)

    def lay(self, ps: Pass, target, limit=4000):
        """Hatch until what is laid comes near ``target`` (coverage) between
        the pass's floor and floor + cap."""
        goal = np.clip(target - ps.floor, 0, ps.cap)
        goal = np.where(self.maps.part == ps.part, goal, 0).astype(np.float32)
        field = self.field(ps.angle)
        h, w = goal.shape
        coarse = shrink(goal[: h // CELL * CELL, : w // CELL * CELL], CELL)
        laid = np.zeros_like(coarse)
        self.full = np.zeros(goal.shape, np.uint8)
        sub = self.maps.lock if ps.part == P_HAIR else np.zeros_like(self.maps.lock)
        for _ in range(limit):
            deficit = coarse - laid
            pick = deficit * self.rng.random(deficit.shape) ** 0.35
            i = int(np.argmax(pick))
            cy, cx = divmod(i, coarse.shape[1])
            if deficit[cy, cx] < 0.3 * ps.cap:
                break
            centre = (np.array([cx, cy]) + self.rng.random(2)) * CELL
            if not self.bundle(ps, centre, goal, field, sub, laid):
                laid[cy, cx] += ps.cap  # nothing could be laid here

    def bundle(self, ps: Pass, centre, goal, field, sub, laid):
        """A bundle of near-parallel strokes across the patch at ``centre``."""
        m, rng = self.maps, self.rng
        x, y = int(centre[0]), int(centre[1])
        if m.part[y, x] != ps.part:
            return False
        turn = math.radians(rng.normal(0, 6))
        fan = math.radians(rng.uniform(-1.5, 1.5))
        d = field[y, x]
        d = np.array(
            [
                d[0] * math.cos(turn) - d[1] * math.sin(turn),
                d[0] * math.sin(turn) + d[1] * math.cos(turn),
            ]
        )
        across = np.array([-d[1], d[0]])
        p = ps.pressure * rng.uniform(0.85, 1.15)
        gap = float(np.clip(line_weight(p) / max(goal[y, x], 1e-3), *ps.spacing))
        n = int(rng.integers(*ps.count))
        length = rng.uniform(*ps.length)
        laid_any = False
        for k in range(n):
            off = (k - (n - 1) / 2) * gap * rng.uniform(0.85, 1.15)
            start = centre + across * off + d * rng.normal(0, 0.15 * length)
            spread = turn + fan * (k - (n - 1) / 2)
            pts = self.trace(ps, start, d, spread, field, goal, sub, length)
            if pts is not None:
                self.draw(ps, pts, p, goal, laid)
                _stamp(self.full, pts[:, 0], pts[:, 1], 0.62 * gap)
                laid_any = True
        return laid_any

    def trace(self, ps: Pass, start, d, turn, field, goal, sub, length):
        m = self.maps
        tmin = 0.12 * ps.cap
        half = [length * self.rng.uniform(0.35, 0.65)]
        half.append(length - half[0])
        out = np.zeros((400, 2))
        ways = []
        for sign, ln in zip((1, -1), half, strict=True):
            k = _trace(
                start[0],
                start[1],
                sign * d[0],
                sign * d[1],
                field,
                m.part,
                ps.part,
                sub,
                goal,
                tmin,
                turn,
                2.0,
                int(ln / 2),
                out,
                self.full,
            )
            ways.append(out[:k].copy())
        pts = np.vstack([ways[1][::-1], ways[0][1:]])
        return pts if len(pts) >= ps.shortest else None

    def draw(self, ps: Pass, pts, p, goal, laid):
        """One stroke, tapering at both ends and following the target."""
        u = np.linspace(0, 1, len(pts))
        taper = np.clip(np.sin(np.pi * u) * 1.6, 0, 1) ** 0.7
        xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
        here = goal[yi, xi] / max(ps.cap, 1e-6)
        press = p * taper * np.clip(0.55 + 0.6 * here, 0.5, 1.15)
        keep = slice(None, None, 3)
        path = pts[keep]
        if len(path) < 3:
            return
        self.sheet.stroke([tuple(q) for q in path], ps.pen, list(press[keep] + 0.01))
        self.strokes += 1
        cy = np.minimum(yi // CELL, laid.shape[0] - 1)
        cx = np.minimum(xi // CELL, laid.shape[1] - 1)
        np.add.at(laid, (cy, cx), line_weight(press) * 2.0 / CELL**2)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


#: Where the flesh flushes: (x, y, radius, strength) in head space, mirrored.
BLUSH = [
    (3.3, -1.6, 1.4, 0.07),
    (0.0, -0.7, 0.6, 0.06),
    (1.2, -1.1, 0.45, 0.05),
    (0.0, -7.4, 0.9, 0.04),
    (2.9, 2.6, 0.9, 0.03),
]
#: Where the shadow is cool, a violet: (x, y, radius, strength), mirrored.
COOL = [
    (2.7, 2.7, 0.8, 0.045),
    (1.15, 0.6, 0.5, 0.05),
    (0.0, -5.6, 0.8, 0.07),
    (5.6, 2.5, 1.2, 0.05),
]
#: Round the mouth and chin the hatching stays open and warm.
MOUTH_ZONE = (0.0, -5.2, 2.4)


def spots(local, table):
    out = np.zeros(local.shape[:2], np.float32)
    for x, y, r, k in table:
        for sx in {x, -x}:
            d2 = (local[..., 0] - sx) ** 2 + (local[..., 1] - y) ** 2
            out += k * np.exp(-0.5 * d2 / r**2)
    return out


def skin_value(m: Maps):
    """The value to model the skin by: mostly how the surface turns from the
    key light, with a little of the soft study's shading.  On the neck it is
    softened, so that the head's shadow is graded."""
    v = np.clip(0.65 * m.form + 0.35 * m.shade, 0, 1)
    neck = m.part == P_NECK
    soft = filters.box_blur(np.where(neck, v, 0.0), 10)
    weight = filters.box_blur(neck.astype(float), 10)
    return np.where(neck, soft / np.maximum(weight, 1e-6), v)


def skin_targets(m: Maps):
    """Coverage wanted from each ink over the skin of the face and neck: in
    steps, so that three or four values read: paper in the lights, one open
    red set in the light half-tones, more red and blue in the half-tones,
    closer red, blue and a little black in the shadows."""
    V = skin_value(m)
    face = m.part == P_FACE
    skin = face | (m.part == P_NECK)
    light = smoothstep(0.07, 0.14, V - 3.0 * np.clip(m.spec - 0.02, 0, 0.05))
    neck = m.part == P_NECK
    light = np.where(neck, np.maximum(smoothstep(0.0, 0.08, V), 0.45), light)
    blush = np.where(face, spots(m.local, BLUSH), 0)
    mid = np.where(
        neck, 0.12 + 0.88 * smoothstep(0.1, 0.4, V), smoothstep(0.3, 0.42, V)
    )
    red = 0.04 * light + 0.06 * mid + 0.09 * smoothstep(0.52, 0.66, V)
    red = red + blush
    x, y = m.local[..., 0], m.local[..., 1]
    near_mouth = np.exp(-0.5 * (x**2 + (y - MOUTH_ZONE[1]) ** 2) / MOUTH_ZONE[2] ** 2)
    calm = np.where(face, 1 - 0.7 * near_mouth, 1)
    cool = np.where(face, spots(m.local, COOL), 0) * light
    blue = 0.05 * smoothstep(0.38, 0.5, V) + 0.07 * smoothstep(0.56, 0.7, V)
    blue = (blue + 0.07 * smoothstep(0.76, 0.9, V)) * calm + cool + 0.06 * m.brow
    black = np.where(face, 0.08 * smoothstep(0.85, 0.97, V), 0)
    return {
        k: np.where(skin, v, 0).astype(np.float32)
        for k, v in (("red", red), ("blue", blue), ("black", black))
    }


def skin_passes(part):
    long = (110, 260)
    return [
        ("red", Pass(RED, part, 12, 0.0, 0.04, 0.2, long, spacing=(6.0, 13.0))),
        ("red", Pass(RED, part, 62, 0.04, 0.06, 0.22, long)),
        ("red", Pass(RED, part, -42, 0.1, 0.1, 0.26)),
        ("blue", Pass(BLUE, part, -50, 0.0, 0.05, 0.2, long)),
        ("blue", Pass(BLUE, part, 30, 0.05, 0.07, 0.24)),
        ("blue", Pass(BLUE, part, 95, 0.12, 0.08, 0.28)),
        ("black", Pass(BLACK, part, 50, 0.0, 0.08, 0.28)),
    ]


def draw_skin(h: Hatcher):
    t = skin_targets(h.maps)
    for part in (P_FACE, P_NECK):
        for ink, ps in skin_passes(part):
            h.lay(ps, t[ink])


def feature_passes():
    lips = {"length": (25, 90), "count": (4, 10), "spacing": (2.2, 6.0), "shortest": 6}
    white = {"length": (14, 40), "count": (3, 6), "spacing": (3.0, 7.0), "shortest": 4}
    return [
        ("red", Pass(RED, P_LIPS, 0, 0.0, 0.16, 0.45, **lips)),
        ("red", Pass(RED, P_LIPS, 25, 0.16, 0.16, 0.45, **lips)),
        ("blue", Pass(BLUE, P_LIPS, 35, 0.0, 0.06, 0.25, **lips)),
        ("red", Pass(RED, P_WHITE, 0, 0.0, 0.08, 0.15, **white)),
        ("blue", Pass(BLUE, P_WHITE, 40, 0.0, 0.08, 0.18, **white)),
    ]


def feature_targets(m: Maps):
    S, lips, white = m.shade, m.part == P_LIPS, m.part == P_WHITE
    bare = 1 - smoothstep(0.03, 0.09, m.spec)
    upper = lips & (m.local[..., 1] > m.nose["stomion_y"])
    red = np.where(lips, (0.1 + 0.12 * S) * (0.25 + 0.75 * bare), 0) + 0.025 * upper
    red = red + np.where(white, 0.03 + 0.08 * smoothstep(0.15, 0.6, S), 0)
    blue = np.where(lips, 0.06 * smoothstep(0.45, 0.9, S), 0)
    blue = blue + np.where(white, 0.08 * smoothstep(0.3, 0.8, S), 0)
    black = np.where(lips, 0.06 * smoothstep(0.7, 1.0, S), 0)
    return {"red": red, "blue": blue, "black": black}


def draw_features(h: Hatcher):
    t = feature_targets(h.maps)
    for ink, ps in feature_passes():
        h.lay(ps, t[ink].astype(np.float32), limit=60 if ps.part == P_LIPS else 400)


def seen_runs(line, min_len=3):
    """The runs of a polyline (k, 3: x, y, seen) that are seen."""
    seen = line[:, 2] > 0.5
    out, start = [], None
    for i, ok in enumerate(np.append(seen, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                out.append(line[start:i, :2])
            start = None
    return out


def pen_line(sheet, pts, pen, pressure, restate=1, wobble=0.6, rng=None):
    """A drawn line along ``pts``, its pressure a function of u (0 to 1),
    restated ``restate`` times a little off the first."""
    rng = rng or np.random.default_rng(len(pts))
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return
    for k in range(restate):
        off = rng.normal(0, wobble, 2) if k else np.zeros(2)
        u0, u1 = (
            (0.0, 1.0) if k == 0 else sorted(rng.uniform(0, 1, 2) * [0.4, 1] + [0, 0.1])
        )
        n = len(pts)
        a, b = int(u0 * (n - 1)), max(int(u1 * (n - 1)) + 1, int(u0 * (n - 1)) + 2)
        seg = pts[a:b] + off
        u = np.linspace(u0, u1, len(seg))
        sheet.stroke([tuple(q) for q in seg], pen, [float(pressure(x)) for x in u])


def swell(lo, hi, peak=0.5, ends=0.15):
    """A pressure that rises from nothing to ``hi`` at ``peak`` and falls."""

    def f(u):
        rise = min(u / ends, 1.0, (1 - u) / ends)
        bump = math.exp(-0.5 * ((u - peak) / 0.3) ** 2)
        return max(rise, 0.0) ** 0.7 * (lo + (hi - lo) * bump)

    return f


def resample(pts, step=3.0):
    pts = np.asarray(pts, float)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = max(int(d[-1] / step), 2)
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], 1)


def draw_eye_lines(h: Hatcher, side):
    sheet, lines = h.sheet, h.maps.lines
    for run in seen_runs(lines[f"lid_upper_{side}"]):
        run = resample(run, 2.0)
        lid_shadow(h, run)
        u = np.linspace(0, 1, len(run))[:, None]
        for lift in (0.0, 1.3, 2.6):  # a thick lid, thickest outwards
            thick = run - np.array([0, 1]) * lift * u
            pen_line(
                sheet,
                thick,
                BOLD,
                swell(0.4, 0.95, 0.75, 0.1),
                1 + (lift == 0),
                0.5,
                h.rng,
            )
        pen_line(sheet, run - [0, 3.5], RED, swell(0.15, 0.35, 0.4), 1)
        draw_lashes(h, run)
    for run in seen_runs(lines[f"crease_{side}"]):
        run = resample(run, 2.0)
        n = len(run)
        pen_line(sheet, run[n // 6 : -n // 8], BLUE, swell(0.1, 0.25, 0.5), 1)
        pen_line(sheet, run[n // 5 : -n // 5], RED, swell(0.1, 0.22, 0.5), 1)
    for run in seen_runs(lines[f"lid_lower_{side}"]):
        run = resample(run, 2.0)
        n = len(run)
        pen_line(sheet, run[n // 4 :] + [0, 1.5], RED, swell(0.08, 0.22, 0.7), 1)
        pen_line(sheet, run[n // 2 :] + [0, 1.0], BLACK, swell(0.05, 0.18, 0.85), 1)


def draw_lashes(h: Hatcher, run, count=7):
    """A few lashes curving out and up from the outer third of the lid."""
    n = len(run)
    for k in range(count):
        i = int(n * (0.62 + 0.36 * k / (count - 1)))
        i = min(i, n - 2)
        p = run[i]
        t = run[i + 1] - run[i - 1]
        t = t / (np.linalg.norm(t) + 1e-9)
        up = np.array([t[1], -t[0]])
        if up[1] > 0:
            up = -up
        out = 0.4 + 0.6 * k / (count - 1)
        d = unit_rows(up + t * out)
        ln = 7 + 7 * k / (count - 1) + h.rng.uniform(-1.5, 1.5)
        pts = [p, p + d * ln * 0.5 + t * ln * 0.12, p + d * ln + t * ln * 0.35]
        h.sheet.stroke([tuple(q) for q in pts], BLACK, swell(0.35, 0.6, 0.2, 0.3))


def draw_iris(h: Hatcher, centre, r, spot):
    """Dark brown: radial strokes of red, black over them in the upper half
    and at the rim, the lower half lighter; a smaller black pupil; the
    catchlight left as a clean disc of paper, a warm fleck opposite it."""
    m, sheet, rng = h.maps, h.sheet, h.rng
    yy, xx = np.mgrid[0 : m.part.shape[0], 0 : m.part.shape[1]]
    glint = (xx - spot[0]) ** 2 + (yy - spot[1]) ** 2 < (0.2 * r) ** 2
    inside = (m.part == P_IRIS) & ~glint
    for a in np.arange(0, 2 * np.pi, np.radians(5.0)):
        low = max(math.sin(a), 0.0)  # 1 straight down (y grows downward)
        rays = [
            (RED, 0.3, 0.97, 0.35),
            (BLACK, 0.62 + 0.15 * low, 0.98, 0.55),
            (BLACK, 0.3, 0.98 - 0.4 * low, 0.5 - 0.2 * low),
            (BLACK, 0.3, 0.9 - 0.5 * low, 0.4),
        ]
        for pen, r0, r1, p in rays:
            b = a + rng.normal(0, 0.05)
            d = np.array([math.cos(b), math.sin(b)])
            r0, r1 = r0 + rng.uniform(-0.04, 0.06), r1 - rng.uniform(0, 0.06)
            if r1 > r0:
                clip_stroke(
                    sheet,
                    centre + np.outer(np.linspace(r0, r1, 8) * r, d),
                    inside,
                    pen,
                    p,
                )
    ring = np.linspace(0, 6.3, 60)
    rim = centre + r * 0.94 * np.stack([np.cos(ring), np.sin(ring)], 1)
    clip_stroke(sheet, rim, inside, BOLD, 0.55)
    pupil = 0.36 * r
    for k in np.arange(-pupil, pupil, 0.9):
        for ang in (0.5, 2.1):
            d = np.array([math.cos(ang), math.sin(ang)])
            q = np.array([-d[1], d[0]])
            half = math.sqrt(max(pupil**2 - k**2, 0))
            pts = [centre + q * k + d * t for t in np.linspace(-half, half, 4)]
            clip_stroke(sheet, np.array(pts), inside, BLACK, 0.8)
    fleck = centre - (spot - centre) * 0.8
    clip_stroke(
        sheet, np.array([fleck - [1.5, 0], fleck + [1.5, 0.5]]), inside, RED, 0.3
    )


def lid_shadow(h: Hatcher, run):
    """The upper lid's shadow on the top of the eye: short arcs under it."""
    m = h.maps
    eye = np.isin(m.part, (P_WHITE, P_IRIS))
    for k, off in enumerate(np.arange(1.0, 7.0, 1.6)):
        pts = run + [0, off]
        p = 0.35 - 0.05 * k
        clip_stroke(h.sheet, pts, eye, BLUE if k % 2 else BLACK, p)


def clip_stroke(sheet, pts, mask, pen, p):
    """Stroke the parts of a path that lie on ``mask``."""
    pts = resample(pts, 1.0)
    xi = np.clip(pts[:, 0].astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, mask.shape[0] - 1)
    line = np.hstack([pts, mask[yi, xi][:, None]])
    for run in seen_runs(line, 2):
        sheet.stroke([tuple(q) for q in run], pen, p)


def draw_brow(h: Hatcher, line, scale):
    """Fine strokes in the direction of growth: up and out at the head,
    then lying along the brow towards its tail."""
    rng = h.rng
    pts = resample(line[:, :2], 1.0)
    n = len(pts)
    for _ in range(420):
        t = rng.uniform(0, 1) ** 0.9
        i = min(int(t * (n - 1)), n - 2)
        along = unit_rows(pts[i + 1] - pts[i])
        if along[0] * (pts[-1, 0] - pts[0, 0]) < 0:
            along = -along
        up = np.array([along[1], -along[0]])
        if up[1] > 0:
            up = -up
        half = brow_width(t) * scale / 2
        start_side = rng.uniform(-1.0, 0.8)
        start = pts[i] + up * start_side * half
        lean = 0.35 * max(0.0, 1 - t / 0.2) + 0.2
        d = unit_rows(along + up * lean)
        ln = rng.uniform(6, 12) * (1 - 0.35 * t)
        k = brow_strength(t)
        pen = BLACK if rng.uniform() < 0.45 else (RED if rng.uniform() < 0.6 else BLUE)
        edge = 1 - abs(start_side) ** 2
        p = (0.05 + 0.2 * k) * (0.5 + 0.5 * edge) * (0.55 + 0.45 * min(t / 0.3, 1))
        mid = start + d * ln / 2 + up * ln * 0.06
        h.sheet.stroke(
            [tuple(start), tuple(mid), tuple(start + d * ln)],
            pen,
            swell(0.3 * p, p, 0.4, 0.3),
        )


def draw_eyes(h: Hatcher):
    for side in ("right", "left"):
        draw_eye_lines(h, side)
    for centre, r, spot in h.maps.eyes:
        draw_iris(h, centre, r, spot)
    for side in ("right", "left"):
        draw_brow(h, h.maps.lines[f"brow_{side}"], 0.9 * VIEW.scale)


def draw_mouth(h: Hatcher):
    for run in seen_runs(h.maps.lines["mouth"]):
        run = resample(run, 2.0)
        u = np.linspace(-1, 1, len(run))
        run = run - np.stack([0 * u, 3.0 * np.abs(u) ** 4], 1)  # the corners lift
        pen_line(h.sheet, run, BLACK, swell(0.15, 0.6, 0.45, 0.2), 2, 0.5, h.rng)
        pen_line(h.sheet, run + [0, -0.8], RED, swell(0.2, 0.5, 0.5, 0.15), 1)


def hair_dark(m: Maps):
    """How dark the hair is at each pixel: 0 in the shine, 1 in the depths."""
    return 1 - np.clip(m.tone / 0.2, 0, 1) ** 0.45


@dataclass
class StrandInk:
    """Strokes of one ink along the strands: how many per pixel of a lock's
    width, the threshold of darkness each needs, and its pressure."""

    pen: pens.Pen
    density: float
    wants: Callable  # darkness -> how likely a stroke is drawn there (0 to 1)
    pressure: tuple  # at darkness 0 and 1


STRAND_INKS = [
    StrandInk(
        RED,
        0.4,
        lambda d: 0.15 + 0.85 * np.exp(-0.5 * ((d - 0.5) / 0.2) ** 2),
        (0.25, 0.4),
    ),
    StrandInk(BLACK, 1.3, lambda d: smoothstep(0.2, 0.9, d) ** 1.2, (0.25, 0.7)),
    StrandInk(BLUE, 0.25, lambda d: smoothstep(0.78, 1.0, d), (0.3, 0.5)),
]


def strand_lines(grid, count, rng):
    """``count`` strands across a grid (rows root to tip, columns across),
    each at its own place across it and stopping a little short of the end."""
    rows, cols = grid.shape[:2]
    out = []
    for j in range(count):
        c = (j + rng.uniform(0.1, 0.9)) / count * (cols - 1)
        i = min(int(c), cols - 2)
        f = c - i
        line = grid[:, i] * (1 - f) + grid[:, i + 1] * f
        end = int(rows * rng.uniform(0.86, 1.0)) if cols < 20 else rows
        out.append(line[:end])
    return out


def upsample(line, step=2.5):
    """A 3D polyline resampled to about ``step`` pixels in the picture."""
    xy = to_pixels(VIEW, line)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    n = max(int(d[-1] / step), 2)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, line[:, k]) for k in range(3)], 1)


def strand_strokes(h: Hatcher, line, ink: StrandInk, dark, lock):
    """Draw one strand where it is seen and dark enough for it: long
    strokes swelling in the middle and thinning to root and tip."""
    m = h.maps
    p3 = upsample(line)
    xy = to_pixels(VIEW, p3)
    xi = np.clip(xy[:, 0].astype(int), 0, WIDTH - 1)
    yi = np.clip(xy[:, 1].astype(int), 0, HEIGHT - 1)
    seen = (m.lock[yi, xi] == lock) & (m.part[yi, xi] == P_HAIR)
    d = dark[yi, xi]
    gate = h.rng.uniform()
    on = seen & (ink.wants(d) > gate)
    w = np.where(on, 1.0, 0.0)
    lo, hi = ink.pressure
    for run, _ in weighted_runs(xy, w, 0.5, 8):
        n = len(run)
        u = np.linspace(0, 1, n)
        dd = sample(dark, run)
        press = (lo + (hi - lo) * dd) * np.sin(np.pi * u).clip(0) ** 0.6 + 0.01
        h.sheet.stroke([tuple(q) for q in run[::3]], ink.pen, list(press[::3]))
        h.strokes += 1


def draw_hair(h: Hatcher):
    """Every lock, and the mass under them, as long strokes along its
    strands: packed and black where dark, spaced with red between in the
    middles, a few in the shine."""
    m, rng = h.maps, h.rng
    dark = hair_dark(m)
    for k, grid in enumerate(m.strands):
        mid = to_pixels(VIEW, grid[grid.shape[0] // 2])
        width = np.linalg.norm(mid[-1] - mid[0]) if k else grid.shape[1] * 2.0
        if k == 0:
            width = 2.0 * grid.shape[1]
        for ink in STRAND_INKS:
            count = max(int(width * ink.density), 2)
            for line in strand_lines(grid, count, rng):
                strand_strokes(h, line, ink, dark, k)


def garment_targets(m: Maps):
    y = np.arange(m.part.shape[0])[:, None]
    dark = np.clip((0.1 - m.tone) / 0.035, 0, 1)
    fade = 1 - smoothstep(1080, 1370, y)
    blue = (0.05 + 0.15 * dark + folds(m.part.shape)) * fade
    return np.where(m.part == P_GARMENT, blue, 0).astype(np.float32)


#: Folds of the top, in the picture: (x0, y0, x1, y1, width), pixels.
FOLDS = [
    (230, 1110, 330, 1330, 22),
    (980, 1090, 900, 1300, 26),
    (640, 1230, 760, 1390, 18),
    (420, 1240, 470, 1400, 16),
]


def folds(shape):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    q = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1).astype(float)
    out = np.zeros(xx.size)
    for x0, y0, x1, y1, w in FOLDS:
        d = segment_distance(q, np.array([x0, y0, 0.0]), np.array([x1, y1, 0.0]))
        out = np.maximum(out, np.exp(-0.5 * (d / w) ** 2))
    return 0.09 * out.reshape(shape)


def loose_ends(h: Hatcher, count=36):
    """The top ends in long loose strokes of blue, falling out of the
    lattice towards the bottom of the sheet."""
    m, rng = h.maps, h.rng
    field = h.field(20)
    for _ in range(count):
        x, y = rng.uniform(170, 1090), rng.uniform(1150, 1330)
        if m.part[int(y), int(x)] != P_GARMENT:
            continue
        d = field[int(y), int(x)].astype(float)
        d = d if d[0] > 0 else -d
        ln = rng.uniform(160, 380)
        sag = rng.uniform(-0.15, 0.25)
        u = np.linspace(0, 1, 10)[:, None]
        pts = np.array([x, y]) + d * ln * u + np.array([0, 1]) * sag * ln * u**2
        pts = resample(pts, 3.0)
        ok = np.isin(sample(m.part, pts), (P_GARMENT, P_PAPER))
        for run in seen_runs(np.hstack([pts, ok[:, None]]), 8):
            h.sheet.stroke(
                [tuple(q) for q in run[::3]], BLUE, swell(0.12, 0.3, 0.4, 0.3)
            )


def draw_garment(h: Hatcher):
    t = garment_targets(h.maps)
    kw = {"length": (260, 620), "count": (4, 9), "spacing": (6.0, 18.0)}
    for a, lo, cap, p in (
        (20, 0.0, 0.05, 0.3),
        (-50, 0.05, 0.06, 0.32),
        (80, 0.11, 0.1, 0.35),
    ):
        h.lay(Pass(BLUE, P_GARMENT, a, lo, cap, p, **kw), t)
    h.lay(Pass(BLACK, P_GARMENT, 55, 0.2, 0.06, 0.3, **kw), t)
    loose_ends(h)


def sample(a, pts):
    xi = np.clip(np.asarray(pts)[:, 0].astype(int), 0, a.shape[1] - 1)
    yi = np.clip(np.asarray(pts)[:, 1].astype(int), 0, a.shape[0] - 1)
    return a[yi, xi]


def outward(line, inside):
    """Unit normals of a polyline pointing out of the shape ``inside``."""
    t = unit_rows(np.gradient(line, axis=0))
    n = np.stack([t[:, 1], -t[:, 0]], 1)
    flip = sample(inside, line + 3 * n)
    return np.where(flip[:, None], -n, n)


def weighted_runs(line, weight, low=0.08, min_len=6):
    """The runs of a line where ``weight`` is above ``low``."""
    on = np.append(weight > low, False)
    out, start = [], None
    for i, ok in enumerate(on):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                out.append((line[start:i], weight[start:i]))
            start = None
    return out


def draw_contour(h: Hatcher, line, weight, pens_):
    """A living line: its pressure follows ``weight`` along it, swelling and
    thinning, broken where the weight falls; restated where strong."""
    for run, w in weighted_runs(line, weight):
        n = len(run)
        u = np.linspace(0, 1, n)
        taper = np.clip(np.minimum(u, 1 - u) / 0.12, 0, 1) ** 0.7
        noise = 0.85 + 0.3 * value_noise(np.arange(n) / 9.0 + h.rng.uniform(0, 99), 4)
        press = w * taper * noise
        for k, pen in enumerate(pens_):
            if k and press.max() < 0.3:
                break
            off = h.rng.normal(0, 0.6, 2) if k else 0
            h.sheet.stroke(
                [tuple(q) for q in run[::2] + off],
                pen,
                list(press[::2] * (1 - 0.25 * k) + 0.005),
            )


HEAD_PARTS = (P_FACE, P_LIPS, P_WHITE, P_IRIS)


def face_outline(h: Hatcher):
    """Firm where the face overlaps the neck or the far hair; none on the
    near cheek; lighter where lit."""
    m = h.maps
    head = np.isin(m.part, HEAD_PARTS)
    centre = m.lines["subnasale"][0, 0]
    for name, line in m.lines.items():
        if not name.startswith("outline_face"):
            continue
        pts = resample(line[:, :2], 2.0)
        n = outward(pts, head)
        other = sample(m.part, pts + 5 * n)
        dark = sample(m.shade, pts - 4 * n)
        far = (other == P_HAIR) & (pts[:, 0] > centre + 40)
        w = np.where(other == P_NECK, 0.55, np.where(far, 0.45, 0.0))
        w = w * (0.45 + 0.8 * dark)
        draw_contour(h, pts, w, (BLACK, BLUE))


def body_outlines(h: Hatcher):
    m = h.maps
    for name, line in m.lines.items():
        part = {"outline_neck": P_NECK, "outline_garment": P_GARMENT}.get(name[:-2])
        if part is None:
            continue
        pts = resample(line[:, :2], 2.0)
        n = outward(pts, m.part == part)
        other = sample(m.part, pts + 5 * n)
        dark = sample(m.shade, pts - 4 * n)
        if part == P_NECK:
            w = np.where(other == P_GARMENT, 0.5, np.where(other == P_PAPER, 0.35, 0))
            gaps = smoothstep(0.25, 0.45, value_noise(np.arange(len(pts)) / 25.0, 8))
            draw_contour(h, pts, w * (0.15 + 1.1 * dark) * gaps, (BLACK, BLUE))
        else:
            w = np.where(other == P_PAPER, 0.35, 0.0) * (pts[:, 1] < 1250)
            draw_contour(h, pts, w * (0.5 + 0.8 * dark), (BLUE,))


def nose_marks(h: Hatcher):
    """A light, partial outline of the nose on its shaded side, and the
    nostrils as small dark accents."""
    m = h.maps
    for name, line in m.lines.items():
        if not name.startswith("nose_"):
            continue
        for run in seen_runs(line, 4):
            pts = resample(run, 2.0)
            dark = sample(m.shade, pts)
            draw_contour(h, pts, 0.1 + 0.4 * smoothstep(0.2, 0.7, dark), (BLUE, BLACK))
    sx, sy = m.lines["subnasale"][0, :2]
    span = 1.6 * VIEW.scale
    win = m.shade[int(sy - 12) : int(sy + 4), int(sx - span) : int(sx + span)]
    for half in (slice(0, int(span) - 6), slice(int(span) + 6, None)):
        part = win[:, half]
        i = int(np.argmax(part))
        y, x = divmod(i, part.shape[1])
        x += (half.start or 0) + int(sx - span)
        y += int(sy - 12)
        p = np.array([x, y], float)
        pts = [p + [-5, 1], p + [0, -1.5], p + [5, 0.5]]
        for k in range(3):
            q = [tuple(r + [0, 0.8 * k]) for r in pts]
            h.sheet.stroke(
                q, BLACK if k < 2 else RED, swell(0.3, 0.7 - 0.15 * k, 0.5, 0.3)
            )


def nose_wings(h: Hatcher):
    """The creases round the wings of the nose: a light, broken line."""
    m = h.maps
    for name in ("wing_r_crease", "wing_l_crease"):
        p = m.nose[name]
        xy = to_pixels(VIEW, p)
        seen = visible(VIEW, m.depth, p, 0.15)
        for run in seen_runs(np.hstack([xy, seen[:, None]]), 4):
            pts = resample(run, 2.0)
            dark = sample(m.form, pts)
            draw_contour(h, pts, 0.15 + 0.4 * smoothstep(0.3, 0.8, dark), (BLUE, BLACK))


def hair_strays(h: Hatcher, count=9):
    """A few loose hairs leaving the silhouette in long, curling strokes."""
    m, rng = h.maps, h.rng
    hair = m.part == P_HAIR
    line = max((v for k, v in m.lines.items() if k.startswith("outline_hair")), key=len)
    pts = resample(line[:, :2], 2.0)
    n = outward(pts, hair)
    edge = sample(m.part, pts + 6 * n) == P_PAPER
    idx = np.nonzero(edge)[0]
    for i in rng.choice(idx, size=min(count, len(idx)), replace=False):
        p0 = pts[i] - 10 * n[i]
        f = m.flow[int(p0[1]), int(p0[0])]
        if not f.any():
            continue
        f = f if f[1] > 0 else -f  # hair falls
        ln = rng.uniform(40, 110)
        bend = n[i] * rng.uniform(6, 22) * rng.choice([1, 1, -0.5])
        u = np.linspace(0, 1, 9)[:, None]
        path = (
            p0 + f * ln * u + bend * u**2 + n[i] * 4 * np.sin(u * 5 + rng.uniform(0, 6))
        )
        pen = BLACK if rng.uniform() < 0.7 else RED
        path = resample(path, 2.0)
        ok = np.isin(sample(m.part, path), (P_HAIR, P_PAPER))
        for run in seen_runs(np.hstack([path, ok[:, None]]), 8)[:1]:
            h.sheet.stroke(
                [tuple(q) for q in run[::3]], pen, swell(0.1, 0.35, 0.3, 0.25)
            )


def draw_lines(h: Hatcher):
    hair_strays(h)
    nose_wings(h)
    face_outline(h)
    body_outlines(h)
    nose_marks(h)


def drawing(m: Maps, seed=7) -> Sheet:
    sheet = Sheet(WIDTH, HEIGHT, paper=PAPER, seed=seed)
    h = Hatcher(sheet, m, np.random.default_rng(seed))
    draw_skin(h)
    draw_features(h)
    draw_hair(h)
    draw_garment(h)
    draw_eyes(h)
    draw_mouth(h)
    draw_lines(h)
    return sheet


def main(studies: Path | None = None):
    """Draw the portrait to ``examples/output/portrait.png``; with a folder,
    also write the shaded studies and the maps there."""
    model = build_model()
    if studies is not None:
        studies.mkdir(parents=True, exist_ok=True)
        write_studies(model, studies)
        write_maps(model, studies)
    sheet = drawing(pen_maps(model))
    out = HERE / "output" / "portrait.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
