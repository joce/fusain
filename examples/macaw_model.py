"""The great green macaw of ``macaw.py``, built as solids and lit.

Imported by ``macaw.py``, which lays the coloured pencil over it: this
module gives, for every pixel of the sheet, which part of the bird is seen
there, where on the bird it lies, how much light it takes, and which way
the feathers run; and it lays the lines it knows (the eye's lids, the
face's rows of feathers, the bill's edges, the wing's feather edges, the
tail's shafts) on the bird's surfaces through any object with a ``line``.

The measurements (landmarks read off fine grids laid over eleven reference
photographs, CC BY / CC0; nothing sampled from them). The head, from seven
near-profile heads, in units of the distance from the eye to the base of
the culmen (the "unit" throughout), the bird facing +x, y up:

* the bill's chord (culmen base to the hook's tip) is 1.26 long (1.17 over
  all seven, 1.26 without the two strongly three-quarter ones, which shorten
  it) and turns 102 degrees down from the line eye-to-culmen-base (sd 2.5):
  the tip hangs almost under the base;
* the culmen stands forward of the chord by 0.26 of its length, highest
  0.43 of the way down; the upper mandible's base sweeps back from the
  culmen to the gape, 0.31 in front of the eye and 0.62 below it;
* the bare face reaches 0.16 above the eye, 0.46 behind it and 0.95 below,
  its feather lines (eight to nine rows, about 0.07 apart) curving about a
  point above the bill's base; the red forehead runs from 0.3 in front of
  the eye to the bill, 0.35 above the eye's level;
* the crown is only 0.35 above the eye: a flat head, the eye set high; the
  iris about 0.06 across its radius.

The whole bird, from three whole perched birds, in eye-to-bill-tip lengths
(1.45 units): shoulder 1.2 from the eye; folded wing 4.1-4.4 from shoulder
to tip, its greater coverts ending 42% of the way; breast 1.6 below the eye;
feet 2.5; tail tip 8.1-9.1, hanging within 5-10 degrees of the vertical.

The model: the feathered head and body are ellipsoids melted together (a
smooth maximum of their depths); each mandible is a surface inflated from
its measured outline, full width along its base and thin at its cutting
edge; the folded wing is a shell lying on the body, its lower edge standing
proud; the tail is six stacked blades; toes and claws wrap a branch. Every
pixel seen knows where it lies on the bird, so the face, the brow, the eye,
the face's feather rows and the bill's lines are laid on the surface and
turn with it. The feathers are relief on the model (covert rows growing in
size down the wing, overlapping flight feathers, small tiles over head and
body that follow the way the feathers run), shown only as much as the light
rakes them.
"""

# cspell:words Catmull culmen headness interp keepdims lexsort nanmin prio tomial tomium

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from itertools import pairwise
from typing import Protocol

import numpy as np
from murex import noise
from murex.filters import box_blur, smoothstep

WIDTH, HEIGHT = 1100, 1540
#: Pixels to one unit of the head (eye to the base of the culmen).
UNIT = 134.0
#: Where the eye lands on the sheet.
EYE_AT = (330.0, 225.0)
SEED = 11
#: How much the light is opened up, so that the sunlit green reads light
#: and the bare face and catchlight go to the paper.
EXPOSURE = 2.3
#: Toward the light (x right, y up, z toward the viewer): above, front-left.
LIGHT = np.array([-0.72, 0.6, 0.34]) / np.linalg.norm([-0.72, 0.6, 0.34])

# ----------------------------------------------------------------------
# the measured head (units: eye to culmen base; facing +x, y up)

#: The bill's chord: from the culmen's base, 1.38 long, turned 102 degrees
#: down from the line eye-to-base.
CHORD, HINGE = 1.26, 102.0
#: The culmen, in the bill's own frame (along the chord, across it: negative
#: is forward of it), averaged over seven photographs.
CULMEN = [
    (0.0, 0.0), (0.03, -0.09), (0.10, -0.156), (0.19, -0.215), (0.28, -0.25),
    (0.43, -0.262), (0.55, -0.255), (0.64, -0.238), (0.74, -0.205),
    (0.82, -0.16), (0.9, -0.1), (0.96, -0.045), (1.0, 0.0),
]  # fmt: skip
#: The upper mandible's back edge: its base against the face (from the
#: culmen's base to the gape), then its cutting edge, with the tomial notch,
#: forward to the hook's point.
BILL_BASE = [(0.0, 0.0), (0.05, 0.14), (0.124, 0.272), (0.25, 0.35), (0.38, 0.40),
             (0.5, 0.43), (0.58, 0.42)]  # fmt: skip
#: (measured on the open bill of one photograph, where the inside of the
#: hook shows: from the gape the edge runs forward, nearly level, then down
#: the inside of the hook, which lies forward of the chord)
TOMIUM = [(0.58, 0.42), (0.55, 0.3), (0.56, 0.17), (0.6, 0.08), (0.63, 0.05),
          (0.66, 0.045), (0.73, -0.02), (0.84, -0.045), (0.93, -0.03),
          (1.0, 0.0)]  # fmt: skip
#: The lower mandible, closed: from the gape, under the upper's edge, out
#: from inside the hook and round its chin and back.
LOWER = [(0.58, 0.42), (0.6, 0.2), (0.68, 0.02), (0.8, -0.02), (0.9, 0.0), (0.97, 0.08),
         (1.03, 0.22), (1.08, 0.36), (1.08, 0.49), (1.01, 0.63), (0.89, 0.74),
         (0.75, 0.75), (0.64, 0.63)]  # fmt: skip
#: The bare face's border, the red forehead's, in the skull frame (eye at
#: the origin, the culmen's base at (1, 0)).
FACE_EDGE = [(0.30, 0.23), (0.16, 0.20), (-0.02, 0.16), (-0.2, 0.13), (-0.34, 0.07),
             (-0.44, -0.08), (-0.46, -0.2), (-0.43, -0.46), (-0.36, -0.62),
             (-0.27, -0.76), (-0.15, -0.94)]  # fmt: skip
BROW_EDGE = [(0.32, 0.30), (0.52, 0.34), (0.71, 0.35), (0.88, 0.30), (0.97, 0.24),
             (1.06, 0.12), (1.05, 0.02), (0.92, -0.04), (0.74, -0.06), (0.55, 0.0),
             (0.4, 0.1), (0.3, 0.2)]  # fmt: skip
#: The eye: the iris's radius, the pupil's, and the bare ring round it.
IRIS, PUPIL, RING = 0.085, 0.036, 0.17


def bill_point(b) -> np.ndarray:
    """Points of the bill's own frame, (N, 2), in the skull frame."""
    a = -math.radians(HINGE)
    u = np.array([math.cos(a), math.sin(a)])
    n = np.array([-u[1], u[0]]) * -1.0  # (backward, up)
    b = np.asarray(b, dtype=np.float64)
    return np.array([1.0, 0.0]) + CHORD * (b[..., :1] * u + b[..., 1:2] * n)


# ----------------------------------------------------------------------
# solids


@dataclass
class Blob:
    """An ellipsoid: its centre, its axes (columns of ``frame``), its radii;
    the part it belongs to, and the way feathers or grain run on it."""

    centre: np.ndarray
    frame: np.ndarray
    radii: np.ndarray
    part: str
    flow: np.ndarray


def axes(along, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """An orthonormal frame (3, 3) whose first axis is ``along`` and second
    as near ``up`` as it can be."""
    a = np.asarray(along, dtype=np.float64)
    a = a / np.linalg.norm(a)
    u = np.asarray(up, dtype=np.float64) - a * np.dot(a, up)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(a, [0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u)
    return np.stack([a, u, np.cross(a, u)], axis=1)


def blob(centre, radii, part, along=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), flow=None):
    """An ellipsoid of the bird."""
    frame = axes(along, up)
    f = frame[:, 0] if flow is None else np.asarray(flow, dtype=np.float64)
    return Blob(np.asarray(centre, float), frame, np.asarray(radii, float), part, f)


def in_plane(angle_deg: float) -> np.ndarray:
    """The direction at ``angle_deg`` (counter-clockwise from forward) in the
    bird's middle plane."""
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


def chain(spine, heights, widths, part, side=(0.0, 0.0, 1.0), overlap=1.6):
    """Ellipsoids along ``spine`` (N, 3): a tube, ``heights`` across it in
    the plane and ``widths`` toward ``side``, each as long as the spacing
    times ``overlap``."""
    spine = np.asarray(spine, dtype=np.float64)
    tangent = np.gradient(spine, axis=0)
    step = np.linalg.norm(tangent, axis=1)
    out = []
    for p, t, h, w, d in zip(spine, tangent, heights, widths, step, strict=True):
        frame = axes(t, np.cross(side, t))
        long = max(d * overlap, 0.6 * min(h, w), 1e-3)
        radii = np.array([long, max(h, 1e-3), max(w, 1e-3)])
        out.append(Blob(p, frame, radii, part, frame[:, 0]))
    return out


def resample(points, n: int) -> np.ndarray:
    """``n`` points evenly along the polyline ``points``."""
    p = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(p.shape[1])], axis=1)


def smooth_curve(points, n: int) -> np.ndarray:
    """A smooth curve through ``points`` (Catmull-Rom), ``n`` points evenly."""
    p = np.asarray(points, dtype=np.float64)
    p = np.concatenate([p[:1] * 2 - p[1:2], p, p[-1:] * 2 - p[-2:-1]])
    out = []
    for i in range(1, len(p) - 2):
        t = np.linspace(0.0, 1.0, 12, endpoint=False)[:, None]
        a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(
            0.5
            * (
                2 * b
                + (-a + c) * t
                + (2 * a - 5 * b + 4 * c - d) * t**2
                + (-a + 3 * b - 3 * c + d) * t**3
            )
        )
    out.append(p[-2:-1])
    return resample(np.concatenate(out), n)


@dataclass
class Patch:
    """A surface given by many points: where they are, the surface's normal
    at each, the way feathers or grain run there; the part it belongs to."""

    points: np.ndarray
    normals: np.ndarray
    flow: np.ndarray
    part: str
    #: The height grid it was made from: (x0, y0, step, heights), in the
    #: plane of ``basis`` (origin, x, y, z directions).
    grid: tuple | None = None
    basis: tuple | None = None

    def lift(self, xy) -> np.ndarray:
        """Points (N, 2) of its plane, lifted onto its near surface (N, 3),
        in the frame it was made in."""
        assert self.grid is not None and self.basis is not None
        x0, y0, step, h = self.grid
        xy = np.asarray(xy, dtype=np.float64)
        fx = np.clip((xy[:, 0] - x0) / step, 0, h.shape[1] - 1.001)
        fy = np.clip((xy[:, 1] - y0) / step, 0, h.shape[0] - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        ax, ay = fx - ix, fy - iy
        z = (
            h[iy, ix] * (1 - ax) * (1 - ay)
            + h[iy, ix + 1] * ax * (1 - ay)
            + h[iy + 1, ix] * (1 - ax) * ay
            + h[iy + 1, ix + 1] * ax * ay
        )
        o, ex, ey, ez = (np.asarray(v, dtype=np.float64) for v in self.basis)
        return o + np.outer(xy[:, 0], ex) + np.outer(xy[:, 1], ey) + np.outer(z, ez)


IDENTITY = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def inside(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which of the points (x, y) lie inside the closed polygon ``poly``."""
    out = np.zeros(x.shape, bool)
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        crosses = (ay > y) != (by > y)
        cx = ax + (y - ay) * (bx - ax) / np.where(by != ay, by - ay, 1.0)
        out ^= crosses & (x < cx)
    return out


def distance_to(curve: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance from each point (x, y) to the polyline ``curve``."""
    best = np.full(x.shape, np.inf)
    for (ax, ay), (bx, by) in pairwise(curve):
        ex, ey = bx - ax, by - ay
        t = np.clip(
            ((x - ax) * ex + (y - ay) * ey) / max(ex * ex + ey * ey, 1e-12), 0, 1
        )
        np.minimum(best, np.hypot(x - ax - t * ex, y - ay - t * ey), out=best)
    return best


def smooth_interp(t, values) -> np.ndarray:
    """``values`` (evenly spaced over 0..1) at ``t``, smoothly between."""
    n = len(values)
    knots = np.stack([np.linspace(0.0, 1.0, n), np.asarray(values, float)], 1)
    dense = smooth_curve(knots, 400)
    return np.interp(t, dense[:, 0], dense[:, 1])


def rounded(d, radius):
    """A quarter circle: how high a rounded edge stands ``d`` in from it."""
    u = np.clip(np.asarray(d) / radius, 0.0, 1.0)
    return np.sqrt(1.0 - (1.0 - u) ** 2)


def grid_inside(poly: np.ndarray, step: float):
    """The points of a square grid of ``step`` inside ``poly``."""
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    gx, gy = np.meshgrid(xs, ys)
    keep = inside(poly, gx, gy)
    return gx, gy, keep


def relief(height: np.ndarray, gx, gy, keep, step, base=None):
    """Points and normals of the surface standing ``height`` above the
    plane (plus ``base``, another such height, if given) over the grid."""
    h = height if base is None else height + base
    dy, dx = np.gradient(h, step)
    n = np.stack([-dx, -dy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    pts = np.stack([gx, gy, h], -1)
    return pts[keep], n[keep]


def mirror_z(points: np.ndarray, normals: np.ndarray):
    """The same surface on the other side of the middle plane."""
    flip = np.array([1.0, 1.0, -1.0])
    return np.concatenate([points, points * flip]), np.concatenate(
        [normals, normals * flip]
    )


def mandible(outline, free, ridge, widths, part, rounding, step=0.004):
    """A mandible of horn from its outline in profile (skull frame): as
    thick as ``widths`` (from its root to its point, along its chord),
    rounded over ``ridge`` (the culmen or the chin) and thin at its ``free``
    edges (``rounding`` = how broadly at the ridge and at the edge), its
    base set square into the face."""
    round_ridge, round_edge = rounding
    ring = smooth_curve(np.concatenate([outline, outline[:1]]), 500)
    gx, gy, keep = grid_inside(ring, step)
    edge = distance_to(smooth_curve(free, 300), gx, gy)
    crest = distance_to(smooth_curve(ridge, 200), gx, gy)
    b0, tip = np.array([1.0, 0.0]), bill_point([(1.0, 0.0)])[0]
    chord = tip - b0
    # (how far from its base toward its point: it is as wide all along its
    # base, which sweeps back from the forehead to the gape)
    base = np.asarray(outline)[np.linalg.norm(np.asarray(outline) - tip, axis=1) > 0]
    d_base = distance_to(base_of(outline), gx, gy)
    d_tip = np.hypot(gx - point_of(outline)[0], gy - point_of(outline)[1])
    t = d_base / np.maximum(d_base + d_tip, 1e-9)
    del base
    w = smooth_interp(t, widths)
    near_ridge = np.clip(crest / np.maximum(edge + crest, 1e-9), 0, 1)
    radius = round_ridge * (1 - near_ridge) + round_edge * near_ridge
    height = w * rounded(edge, radius) * np.where(keep, 1.0, 0.0)
    pts, nrm = relief(height, gx, gy, keep, step)
    pts, nrm = mirror_z(pts, nrm)
    flow = np.tile(np.append(chord / np.linalg.norm(chord), 0.0), (len(pts), 1))
    return Patch(pts, nrm, flow, part, (gx[0, 0], gy[0, 0], step, height), IDENTITY)


@cache
def head_blobs() -> list[Blob]:
    """The head, in the skull frame (eye at the origin, the culmen's base at
    (1, 0), y up, z toward the bird's right): the feathered skull, forehead,
    cheeks and nape; the eye; the two mandibles."""
    back = (-1.0, 0.0, 0.0)
    plume = [
        blob((-0.16, -0.02, 0.0), (0.72, 0.38, 0.46), "plumage", flow=back),
        blob((0.4, 0.14, 0.0), (0.55, 0.24, 0.26), "plumage", flow=back),
        blob((0.0, -0.42, 0.0), (0.55, 0.52, 0.4), "plumage", flow=(-0.3, -1.0, 0)),
        blob((-0.5, -0.32, 0.0), (0.6, 0.68, 0.45), "plumage", flow=(-0.5, -1.0, 0)),
        blob((0.36, -0.3, 0.0), (0.28, 0.28, 0.24), "plumage", flow=back),
    ]
    eye = [blob((0.0, 0.0, 0.36), (0.1, 0.1, 0.1), "eye")]
    return plume + eye


def base_of(outline) -> np.ndarray:
    """The base of a mandible's outline: where it is set into the face (the
    upper's from the forehead to the gape, the lower's from the gape back)."""
    o = np.asarray(outline)
    if np.allclose(o[0], bill_point(LOWER[:1])[0]):
        return np.vstack([o[-1], o[0]])
    return bill_point(BILL_BASE)


def point_of(outline) -> np.ndarray:
    """The point of a mandible: the hook's tip, or the front of the chin."""
    o = np.asarray(outline)
    if np.allclose(o[0], bill_point(LOWER[:1])[0]):
        return bill_point([(1.06, 0.3)])[0]
    return bill_point([(1.0, 0.0)])[0]


@cache
def bill_patches() -> list[Patch]:
    """The two mandibles, as horn."""
    upper = np.concatenate(
        [
            bill_point(CULMEN),
            bill_point(TOMIUM[::-1][1:]),
            bill_point(BILL_BASE[::-1][1:-1]),
        ]
    )
    lower = bill_point(LOWER)
    chin = bill_point(LOWER[4:10])
    upper_free = np.concatenate([bill_point(CULMEN), bill_point(TOMIUM[::-1][1:])])
    lower_free = bill_point(LOWER[:-1])
    return [
        mandible(
            upper,
            upper_free,
            bill_point(CULMEN),
            [0.28, 0.27, 0.23, 0.16, 0.08, 0.025],
            "upper",
            (0.3, 0.035),
        ),
        mandible(
            lower,
            lower_free,
            chin,
            [0.3, 0.3, 0.28, 0.24, 0.16, 0.08],
            "lower",
            (0.3, 0.08),
        ),
    ]


# ----------------------------------------------------------------------
# the bird (units: eye to culmen base; the eye at the origin, facing +x,
# y up, z toward the side the viewer sees)

#: The head's pitch (the line eye-to-culmen-base, degrees above forward).
HEAD_PITCH = 4.0
#: The wrist of the folded wing and the tip of its longest primary.
SHOULDER, WING_TIP = (-1.5, -0.86), (-4.42, -6.85)
#: The root of the tail and its direction (degrees from straight down,
#: backward), how long its middle feathers are, how far its plane is rolled
#: to show its upper side to the viewer: it hangs steeper than the wing,
#: so that the wing's tips lie on it.
TAIL_ROOT, TAIL_HANG, TAIL_LENGTH, TAIL_ROLL = (-3.5, -4.75), 17.0, 7.6, 62.0
#: The branch: a point on its axis under the feet, its direction (it runs
#: along the bird, as macaws often stand), its radius; where along it from
#: that point each foot grips.
BRANCH_AT, BRANCH_DIR, BRANCH_R = (-1.65, -4.32, 0.0), (0.97, -0.2, 0.06), 0.34
FEET = (-0.42, 0.34)

#: The feathered body: centre (x, y), radii (along, across, sideways) and
#: the direction (degrees) its long axis runs, down from the neck: the
#: neck, the breast, the belly, the rump and vent, the back.
TORSO = [
    ((-0.62, -1.0), (0.95, 0.85, 0.54), -112.0),
    ((-1.55, -1.9), (1.45, 1.22, 1.06), -126.0),
    ((-2.05, -2.75), (1.2, 1.1, 1.02), -128.0),
    ((-2.3, -3.35), (1.15, 1.0, 0.96), -132.0),
    ((-3.2, -4.35), (1.0, 0.65, 0.68), -130.0),
    ((-2.6, -2.4), (1.9, 0.62, 0.8), -130.0),
]
#: The feathered thighs, one over each foot, coming down over its top.
THIGHS = [
    ((-2.0, -3.78, 0.16), (0.5, 0.46, 0.4)),
    ((-1.38, -3.7, -0.04), (0.46, 0.44, 0.38)),
]


@cache
def body_blobs() -> list[Blob]:
    """The neck, breast, belly, back, vent and thighs: one soft mass."""
    out = [
        blob((x, y, 0.0), radii, "plumage", in_plane(deg), flow=in_plane(deg))
        for (x, y), radii, deg in TORSO
    ]
    out += [
        blob(c, radii, "plumage", in_plane(-100.0), flow=in_plane(-80.0))
        for c, radii in THIGHS
    ]
    return out


def soft_extent(blobs: list[Blob], x, y, k: float = 0.08) -> np.ndarray:
    """How far toward +z the soft union of ``blobs`` reaches over the points
    (x, y) of the bird's middle plane (-inf where it does not)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    origin = np.stack([x, y, np.zeros_like(x)], -1)
    top = np.full(x.shape, -np.inf)
    for b in blobs:
        top, _ = smooth_max(top, ray_hit(b, origin, np.array([0.0, 0.0, 1.0])), k)
    return top


def ray_hit(b: Blob, origin: np.ndarray, d: np.ndarray) -> np.ndarray:
    """How far along ``d`` from each ``origin`` (..., 3) the far side of the
    ellipsoid ``b`` lies (-inf where the ray misses)."""
    q0 = (origin - b.centre) @ b.frame / b.radii
    dd = (b.frame.T @ d) / b.radii
    a = dd @ dd
    bb = 2.0 * q0 @ dd
    c = np.sum(q0 * q0, axis=-1) - 1.0
    disc = bb * bb - 4 * a * c
    return np.where(disc >= 0, (-bb + np.sqrt(np.maximum(disc, 0))) / (2 * a), -np.inf)


def plate(outline, height_of, basis, part, step=0.004, both=True) -> Patch:
    """A surface standing over the plane of ``basis`` (origin, x, y, z
    directions in the bird's frame): over the inside of ``outline`` (in
    the plane's x, y), ``height_of(gx, gy, edge)`` high (``edge`` the
    distance in from the outline); and its mirror image under the plane."""
    ring = np.asarray(outline, dtype=np.float64)
    gx, gy, keep = grid_inside(ring, step)
    edge = distance_to(np.concatenate([ring, ring[:1]]), gx, gy)
    height = np.where(keep, height_of(gx, gy, edge), 0.0)
    pts, nrm = relief(height, gx, gy, keep, step)
    if both:
        pts, nrm = mirror_z(pts, nrm)
    o, ex, ey, ez = (np.asarray(v, dtype=np.float64) for v in basis)
    m = np.stack([ex, ey, ez], axis=1)
    flow = np.tile(ex, (len(pts), 1))
    return Patch(
        o + pts @ m.T, nrm @ m.T, flow, part, (gx[0, 0], gy[0, 0], step, height), basis
    )


def wing_frame():
    """The folded wing's start (the wrist), its unit direction along it, and
    across it (toward the front and down), and its length."""
    a, b = np.array(SHOULDER), np.array(WING_TIP)
    length = float(np.linalg.norm(b - a))
    along = (b - a) / length
    return a, along, np.array([-along[1], along[0]]), length


#: The folded wing's outline in the bird's middle plane: from the wrist its
#: upper edge runs back along the line of the back to the tip, then its
#: lower edge comes forward across the flank, back to the wrist.
WING_OUTLINE = [
    (-1.5, -0.86), (-1.95, -0.93), (-2.45, -1.38), (-2.95, -2.18), (-3.4, -3.08),
    (-3.75, -3.98), (-4.05, -4.88), (-4.25, -5.78), (-4.38, -6.48), (-4.42, -6.85),
    (-4.28, -6.62), (-3.95, -5.85), (-3.45, -5.02), (-2.82, -4.25), (-2.15, -3.62),
    (-1.6, -3.0), (-1.25, -2.35), (-1.08, -1.7), (-1.12, -1.2), (-1.3, -0.93),
]  # fmt: skip


def wing_uv(x, y):
    """The wing's own (u, v) at points (x, y) of the bird's middle plane: u
    0 at the wrist to 1 at the tip, v across (toward the lower edge)."""
    a, along, across, length = wing_frame()
    dx, dy = x - a[0], y - a[1]
    return (dx * along[0] + dy * along[1]) / length, dx * across[0] + dy * across[1]


def wing_patch(body: list[Blob]) -> Patch:
    """The folded wing, a shell lying on the flank and over the back: thick
    over the coverts, thin over the long flight feathers, its lower edge
    standing proud of the flank."""
    ring = smooth_curve(np.vstack([WING_OUTLINE, WING_OUTLINE[:1]]), 400)
    k = BLEND["plumage"] / UNIT
    step = 0.004

    def height(gx, gy, edge):
        u, v = wing_uv(gx, gy)
        body_z = soft_extent(body, gx, gy, k)
        base = np.where(np.isfinite(body_z), body_z, 0.1).astype(np.float32)
        r = int(0.12 / step)
        base = box_blur(box_blur(base, r), r).astype(np.float64)
        base = np.where(np.isfinite(body_z), np.maximum(base, body_z), base)
        thick = np.interp(u, [0.0, 0.3, 0.45, 0.7, 1.0], [0.3, 0.27, 0.17, 0.09, 0.05])
        lower = np.clip(v / 1.0, 0.0, 1.0)
        radius = 0.3 * (1 - lower) + 0.07 * lower
        return base - 0.06 + (thick + 0.08) * rounded(edge, radius)

    basis = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    patch = plate(ring, height, basis, "wing", step=step, both=False)
    _, along, _, _ = wing_frame()
    patch.flow = np.tile([along[0], along[1], 0.0], (len(patch.points), 1))
    return patch


def tail_axes():
    """The tail's direction, its upper side's normal and its sideways axis,
    the plane rolled toward the viewer by TAIL_ROLL."""
    h = math.radians(TAIL_HANG)
    d = np.array([-math.sin(h), -math.cos(h), 0.0])
    up = np.array([-math.cos(h), math.sin(h), 0.0])
    r = math.radians(TAIL_ROLL)
    n = math.cos(r) * up + math.sin(r) * np.array([0.0, 0.0, 1.0])
    return d, n, np.cross(n, d)


#: The tail feathers seen, from the uppermost (the middle pair) down: how
#: far sideways each lies, how long it is (of TAIL_LENGTH), how far it fans
#: (degrees) and how wide it is.
BLADES = [(0.05, 1.0, 1.0, 0.25), (-0.07, 0.95, -1.2, 0.25), (0.2, 0.8, 3.5, 0.24),
          (-0.21, 0.76, -3.5, 0.24), (0.33, 0.6, 6.0, 0.23),
          (-0.34, 0.56, -6.0, 0.23)]  # fmt: skip


def blade_half(t, half: float) -> np.ndarray:
    """How wide a tail feather is (half its width) at ``t`` along it (0 at
    the root, 1 at the tip): narrowing a little, its end rounded."""
    t = np.asarray(t, dtype=np.float64)
    w = half * np.where(
        t < 0.72, 1.0 - 0.22 * t, 0.84 * np.sqrt(np.clip((1 - t) / 0.28, 0, 1))
    )
    return np.maximum(w, 0.004)


def blade_outline(length: float, half: float) -> np.ndarray:
    """A long tail feather in its own plane: from its root, narrowing
    a little, its end rounded to a blunt point."""
    t = np.linspace(0.0, 1.0, 60)
    w = blade_half(t, half)
    top = np.stack([t * length, w], 1)
    return np.vstack([top, top[::-1] * np.array([1.0, -1.0])])


def blade_basis(k: int) -> tuple:
    """Tail feather ``k``'s plane: its root, its direction, across it and
    its upper side's normal."""
    d, n, lateral = tail_axes()
    off, _, fan, _ = BLADES[k]
    f = math.radians(fan)
    dk = math.cos(f) * d + math.sin(f) * lateral
    origin = np.array([*TAIL_ROOT, 0.1]) + lateral * off - n * (0.045 * k)
    return origin, dk, np.cross(n, dk), n


def tail_patches() -> list[Patch]:
    """The long tail feathers: thin blades from the root, stacked."""
    out = []
    for k, (_, frac, _, half) in enumerate(BLADES):
        outline = blade_outline(TAIL_LENGTH * frac, half)

        def height(gx, gy, edge):
            return 0.022 * rounded(edge, 0.03) + 0.012 * (1 - np.abs(gy) / 0.3)

        out.append(plate(outline, height, blade_basis(k), f"tail{k}", step=0.006))
    return out


def branch_frame():
    """The branch's axis direction and, across it, up and toward the viewer's
    side."""
    b = np.asarray(BRANCH_DIR, dtype=np.float64)
    b /= np.linalg.norm(b)
    up = np.array([0.0, 1.0, 0.0]) - b * b[1]
    up /= np.linalg.norm(up)
    side = np.cross(b, up)
    if side[2] < 0:
        side = -side
    return b, up, side


def branch_blobs() -> list[Blob]:
    """The branch: a long round stick, a little bent."""
    b, up, side = branch_frame()
    s = np.linspace(-11.0, 11.0, 300)
    spine = np.asarray(BRANCH_AT) + np.outer(s, b) + np.outer(0.006 * s**2, up)
    r = BRANCH_R * (1.0 - 0.01 * s)
    return chain(spine, r, r, "branch", side=side, overlap=6.0)


def toe(centre, turn, along, thick, part="toes"):
    """A toe round the branch: ``turn`` (from, to) in degrees round its
    section (90 is the top, 0 the side toward the viewer), ``along`` (from,
    to) the branch's axis, ``thick`` (from, to) its radius."""
    b, up, side = branch_frame()
    a = np.radians(np.linspace(turn[0], turn[1], 16))
    r = BRANCH_R + np.linspace(thick[0], thick[1], 16) * 0.8
    spine = (
        centre
        + np.outer(np.cos(a) * r, side)
        + np.outer(np.sin(a) * r, up)
        + np.outer(np.linspace(along[0], along[1], 16), b)
    )
    radius = np.linspace(thick[0], thick[1], 16)
    return chain(spine, radius, radius, part, side=b, overlap=3.0)


#: The toes of a foot: round the branch from the top (degrees), sideways
#: along it, thickness: two over the near side, two over the far side.
TOES = ((100, 5, 0.1, 0.14), (100, 12, -0.09, 0.14), (95, 215, 0.14, 0.125),
        (95, 205, -0.16, 0.1))  # fmt: skip


def feet_blobs() -> list[Blob]:
    """The two feet gripping the branch, two toes on either side of it, each
    with its claw, and the short scaly leg from the thigh."""
    b, up, side = branch_frame()
    out = []
    for f in FEET:
        c = np.asarray(BRANCH_AT) + b * f
        top = c + up * (BRANCH_R + 0.08)
        leg = np.linspace(top + up * 0.22 + side * 0.04, top, 6)
        out += chain(leg, [0.15] * 6, [0.14] * 6, "toes", side=b, overlap=3.0)
        for start, stop, along, th in TOES:
            out += toe(c, (start, stop), (0.0, along), (th, th * 0.7))
            end = stop + (-58 if stop < 90 else 58)
            out += toe(
                c, (stop, end), (along, along * 1.12), (th * 0.62, 0.018), "claws"
            )
    return out


def bird_blobs() -> tuple[list[Blob], list[Blob]]:
    """All of the bird's ellipsoids, in its own frame: those of the head
    (which turn with it) and the rest."""
    return head_blobs(), body_blobs() + feet_blobs() + branch_blobs()


@cache
def bird_patches() -> tuple[list[Patch], list[Patch]]:
    """All of the bird's surfaces of points: the head's and the rest."""
    return bill_patches(), [wing_patch(body_blobs()), *tail_patches()]


# ----------------------------------------------------------------------
# the pose and the view

#: How far the bird turns its front toward the viewer (degrees); how its
#: head turns further, cocks (rolls) and nods, about the top of the neck.
TURN = 14.0
HEAD_TURN, HEAD_COCK, HEAD_NOD = -16.0, 6.0, -4.0
NECK_PIVOT = np.array([-0.6, -0.8, 0.0])


def rot(axis: int, degrees: float) -> np.ndarray:
    """The rotation (3, 3) by ``degrees`` about the x, y or z axis."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m = np.eye(3)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


@dataclass
class Pose:
    """Where the bird's own frame lands on the sheet: x right, y down, z
    toward the viewer, in pixels."""

    view: np.ndarray = field(
        default_factory=lambda: np.diag([1.0, -1.0, -1.0]) @ rot(1, 180.0 - TURN)
    )
    head: np.ndarray = field(
        default_factory=lambda: rot(1, HEAD_TURN) @ rot(0, HEAD_COCK) @ rot(2, HEAD_NOD)
    )
    skull: np.ndarray = field(default_factory=lambda: rot(2, HEAD_PITCH))

    def to_bird(self, p, head: bool) -> np.ndarray:
        """Points (..., 3) of the skull frame (head) or the bird's frame,
        in the bird's frame as posed."""
        p = np.asarray(p, dtype=np.float64)
        if not head:
            return p
        return (p @ self.skull.T - NECK_PIVOT) @ self.head.T + NECK_PIVOT

    def linear(self, head: bool) -> np.ndarray:
        """The rotation from the skull frame (head) or the bird's to the
        sheet."""
        return self.view @ self.head @ self.skull if head else self.view

    def to_sheet(self, p, head: bool = False) -> np.ndarray:
        """Points (..., 3) of the skull or bird frame on the sheet (pixels)."""
        q = self.to_bird(p, head) @ self.view.T * UNIT
        return q + np.array([EYE_AT[0], EYE_AT[1], 0.0])

    def from_sheet(self, p, head: bool = False) -> np.ndarray:
        """Points (..., 3) of the sheet back in the skull or bird frame."""
        q = (np.asarray(p, float) - np.array([EYE_AT[0], EYE_AT[1], 0.0])) / UNIT
        q = q @ self.view
        if not head:
            return q
        return ((q - NECK_PIVOT) @ self.head + NECK_PIVOT) @ self.skull

    def place(self, blobs: list[Blob], head: bool) -> list[Blob]:
        """The blobs on the sheet."""
        m = self.linear(head)
        return [
            Blob(
                self.to_sheet(b.centre, head),
                m @ b.frame,
                b.radii * UNIT,
                b.part,
                m @ b.flow,
            )
            for b in blobs
        ]


@dataclass
class Scene:
    """The solids on the sheet: ellipsoids and surfaces of points."""

    blobs: list[Blob]
    patches: list[Patch]

    def turned(self, m: np.ndarray, shift=(0.0, 0.0, 0.0)) -> Scene:
        """The scene turned by the rotation ``m`` (and moved by -``shift``)."""
        sh = np.asarray(shift, dtype=np.float64)
        blobs = [
            Blob(m @ b.centre - sh, m @ b.frame, b.radii, b.part, b.flow)
            for b in self.blobs
        ]
        patches = [
            Patch(p.points @ m.T - sh, p.normals @ m.T, p.flow @ m.T, p.part)
            for p in self.patches
        ]
        return Scene(blobs, patches)


def place_patch(pose: Pose, patch: Patch, head: bool) -> Patch:
    """A surface on the sheet."""
    m = pose.linear(head)
    return Patch(
        pose.to_sheet(patch.points, head),
        patch.normals @ m.T,
        patch.flow @ m.T,
        patch.part,
        patch.grid,
        patch.basis,
    )


def posed(pose: Pose) -> Scene:
    """All of the bird's solids, on the sheet."""
    head, rest = bird_blobs()
    blobs = pose.place(head, True) + pose.place(rest, False)
    head_p, rest_p = bird_patches()
    patches = [place_patch(pose, p, True) for p in head_p]
    patches += [place_patch(pose, p, False) for p in rest_p]
    return Scene(blobs, patches)


# ----------------------------------------------------------------------
# rendering the solids

#: How softly the solids of each part melt into one another (pixels).
BLEND = {"plumage": 15.0, "upper": 3.0, "lower": 3.0, "wing": 4.0, "wing_far": 4.0,
         "branch": 4.0, "toes": 2.0, "claws": 1.0, "eye": 0.5}  # fmt: skip


@dataclass
class Render:
    """What the eye meets at each pixel: how near (z), which part, the
    surface's normal and the way its feathers run (both on the sheet)."""

    z: np.ndarray
    part: np.ndarray
    normal: np.ndarray
    flow: np.ndarray
    names: list[str]

    def mask(self, *names: str) -> np.ndarray:
        """Where any of the parts ``names`` is seen."""
        ids = [self.names.index(n) for n in names if n in self.names]
        return np.isin(self.part, ids)


def extent(b: Blob) -> np.ndarray:
    """How far the ellipsoid reaches from its centre along x, y, z."""
    return np.sqrt(((b.frame * b.radii) ** 2).sum(axis=1))


def ellipsoid(b: Blob, box):
    """The near surface of ``b`` over the pixels of ``box`` (x0, y0, x1, y1):
    its depth (-inf off it) and its normal."""
    x0, y0, x1, y1 = box
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64) + 0.5
    p = np.stack([xs, ys, np.zeros_like(xs)], axis=-1) - b.centre
    q0 = (p @ b.frame) / b.radii
    d = b.frame[2] / b.radii
    a = d @ d
    bb = 2.0 * (q0 @ d)
    c = np.sum(q0 * q0, axis=-1) - 1.0
    disc = bb * bb - 4.0 * a * c
    hit = disc >= 0
    root = np.sqrt(np.maximum(disc, 0.0))
    z = np.where(hit, (-bb + root) / (2.0 * a), -np.inf)
    q = q0 + np.where(hit, z, 0.0)[..., None] * d
    n = (q / b.radii) @ b.frame.T
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    return z, n, root / a


def box_of(b: Blob, shape) -> tuple[int, int, int, int] | None:
    """The pixels ``b`` may cover, or None if none of them is on the sheet."""
    e = extent(b)
    h, w = shape
    x0, y0 = (
        int(max(math.floor(b.centre[0] - e[0]), 0)),
        int(max(math.floor(b.centre[1] - e[1]), 0)),
    )
    x1, y1 = (
        int(min(math.ceil(b.centre[0] + e[0]) + 1, w)),
        int(min(math.ceil(b.centre[1] + e[1]) + 1, h)),
    )
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def smooth_max(old, new, k):
    """The smooth maximum of two depths (-inf where absent) and how much of
    the new one is in it (0..1): equal to the larger far from where they
    meet, rounded over ``k`` where they do."""
    have_old, have_new = np.isfinite(old), np.isfinite(new)
    o = np.where(have_old, old, 0.0)
    n = np.where(have_new, new, 0.0)
    h = np.clip(0.5 + 0.5 * (n - o) / k, 0.0, 1.0)
    h = np.where(have_old & have_new, h, np.where(have_new, 1.0, 0.0))
    z = o * (1 - h) + n * h + k * h * (1 - h) * (have_old & have_new)
    return np.where(have_old | have_new, z, -np.inf), h


def render_part(blobs: list[Blob], shape, k: float):
    """One part: its solids melted together (a smooth maximum of their
    depths), their normals and flows mixed where they meet."""
    top = np.full(shape, -np.inf)
    nrm = np.zeros((*shape, 3))
    flo = np.zeros((*shape, 3))
    for b in blobs:
        box = box_of(b, shape)
        if box is None:
            continue
        z, n, thick = ellipsoid(b, box)
        sl = (slice(box[1], box[3]), slice(box[0], box[2]))
        new, h = smooth_max(top[sl], z, k)
        # (a solid counts for less toward its own silhouette, so that the
        # surface it melts into turns smoothly there)
        cover = np.clip(thick / (2.0 * k), 0.0, 1.0)
        hn = np.where(np.isfinite(top[sl]), h * cover, h)[..., None]
        nrm[sl] = nrm[sl] * (1 - hn) + n * hn
        flo[sl] = flo[sl] * (1 - hn) + b.flow * hn
        top[sl] = new
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    return top, nrm, flo


def splat(patches: list[Patch], shape):
    """Surfaces of points: at each pixel the nearest point's depth, normal
    and flow."""
    pts = np.concatenate([p.points for p in patches])
    nrm = np.concatenate([p.normals for p in patches])
    flo = np.concatenate([p.flow for p in patches])
    h, w = shape
    # (each point covers a 2 x 2 footprint, so that no hole opens where the
    # surface turns steeply away)
    pts = np.concatenate(
        [pts + [dx, dy, 0.0] for dx in (0.0, 0.7) for dy in (0.0, 0.7)]
    )
    nrm, flo = np.tile(nrm, (4, 1)), np.tile(flo, (4, 1))
    ix, iy = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    key, z = iy[ok] * w + ix[ok], pts[ok, 2]
    order = np.lexsort((z, key))
    ks = key[order]
    last = order[np.append(ks[1:] != ks[:-1], True)]
    zz = np.full(h * w, -np.inf)
    nn, ff = np.zeros((h * w, 3)), np.zeros((h * w, 3))
    zz[key[last]] = z[last]
    nn[key[last]] = nrm[ok][last]
    ff[key[last]] = flo[ok][last]
    return zz.reshape(shape), nn.reshape(*shape, 3), ff.reshape(*shape, 3)


def render_any(scene: Scene, name: str, shape):
    """One part of the scene, whether solids or surfaces."""
    blobs = [b for b in scene.blobs if b.part == name]
    if blobs:
        return render_part(blobs, shape, BLEND.get(name, 1.0))
    return splat([p for p in scene.patches if p.part == name], shape)


def render(scene: Scene, shape=(HEIGHT, WIDTH)) -> Render:
    """The solids as the viewer sees them."""
    names = sorted({b.part for b in scene.blobs} | {p.part for p in scene.patches})
    out = Render(
        np.full(shape, -np.inf),
        np.full(shape, -1, np.int16),
        np.zeros((*shape, 3)),
        np.zeros((*shape, 3)),
        names,
    )
    for i, name in enumerate(names):
        z, n, f = render_any(scene, name, shape)
        front = z > out.z
        out.z[front], out.part[front] = z[front], i
        out.normal[front], out.flow[front] = n[front], f[front]
    return out


def light_frame() -> np.ndarray:
    """The rotation that turns the light's direction (on the sheet) into +z."""
    sun = SUN
    e1 = np.cross([0.0, 1.0, 0.0], sun)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(sun, e1), sun])


def shadowed(scene: Scene, seen: Render, bias: float = 3.0) -> np.ndarray:
    """Where the seen surfaces lie in the shadow of others (0 or 1)."""
    m = light_frame()
    turned = scene.turned(m)
    lo = np.min([b.centre - extent(b) for b in turned.blobs], axis=0)
    hi = np.max([b.centre + extent(b) for b in turned.blobs], axis=0)
    shift = np.array([lo[0] - 2, lo[1] - 2, 0.0])
    turned = scene.turned(m, m.T @ shift * 0 + shift)
    shape = (int(hi[1] - lo[1]) + 5, int(hi[0] - lo[0]) + 5)
    depth = np.full(shape, -np.inf)
    names = {b.part for b in turned.blobs} | {p.part for p in turned.patches}
    for name in sorted(names):
        z, _, _ = render_any(turned, name, shape)
        np.maximum(depth, z, out=depth)
    ys, xs = np.nonzero(np.isfinite(seen.z))
    p = np.stack([xs + 0.5, ys + 0.5, seen.z[ys, xs]], -1) + 2.5 * seen.normal[ys, xs]
    p = p @ m.T - shift
    ix = np.clip(p[:, 0].astype(int), 0, shape[1] - 1)
    iy = np.clip(p[:, 1].astype(int), 0, shape[0] - 1)
    out = np.zeros(seen.z.shape)
    out[ys, xs] = (depth[iy, ix] - p[:, 2] > bias).astype(float)
    return out


# ----------------------------------------------------------------------
# where each seen pixel lies on the bird


@dataclass
class Model:
    """The bird as seen: the render, the cast shadow, and for each pixel
    where it lies in the skull's frame and in the bird's."""

    pose: Pose
    scene: Scene
    seen: Render
    shadow: np.ndarray
    skull: np.ndarray
    bird: np.ndarray
    headness: np.ndarray
    #: The normals with the feathers' relief on them (H, W, 3); and the
    #: relief itself, in pixels.
    normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    relief: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def part(self, *names: str) -> np.ndarray:
        """Where any of these parts is seen."""
        return self.seen.mask(*names)


def implicit(blobs: list[Blob], p: np.ndarray) -> np.ndarray:
    """How far inside the nearest of ``blobs`` each point (N, 3) is: below 1
    inside one."""
    best = np.full(len(p), np.inf)
    for b in blobs:
        q = (p - b.centre) @ b.frame / b.radii
        np.minimum(best, np.sqrt(np.sum(q * q, axis=1)), out=best)
    return best


def build_model() -> Model:
    """The bird modelled, posed, seen and lit."""
    pose = Pose()
    scene = posed(pose)
    seen = render(scene)
    shadow = box_blur(box_blur(shadowed(scene, seen).astype(np.float32), 2), 1)
    ys, xs = np.nonzero(np.isfinite(seen.z))
    p = np.stack([xs + 0.5, ys + 0.5, seen.z[ys, xs]], -1)
    skull = np.full((*seen.z.shape, 3), np.nan)
    bird = np.full((*seen.z.shape, 3), np.nan)
    skull[ys, xs] = pose.from_sheet(p, head=True)
    bird[ys, xs] = pose.from_sheet(p, head=False)
    head, rest = bird_blobs()
    near_head = implicit(head[:5], skull[ys, xs])
    near_body = implicit(rest[: len(TORSO) + len(THIGHS)], bird[ys, xs])
    headness = np.zeros(seen.z.shape)
    headness[ys, xs] = smoothstep(0.25, -0.25, near_head - near_body)
    headness = np.where(seen.mask("upper", "lower", "eye"), 1.0, headness)
    model = Model(pose, scene, seen, shadow, skull, bird, headness)
    model.relief = np.zeros(seen.z.shape)
    model.normal = seen.normal.copy()
    return model


def rake(model: Model) -> np.ndarray:
    """How much the light rakes each surface, 0..1: little where it falls
    full on it or where the surface is deep in shadow, most where the form
    turns from light to shadow."""
    n = model.seen.normal
    facing = n @ SUN
    turning = smoothstep(0.95, 0.45, facing) * smoothstep(-0.25, 0.2, facing)
    return np.clip(0.12 + 0.88 * turning * (1.0 - 0.6 * model.shadow), 0.0, 1.0)


def emboss(model: Model, height: np.ndarray, where: np.ndarray) -> None:
    """Raise the relief ``height`` (pixels) over ``where`` on the model's
    surfaces: its slopes tilt their normals, as much as the light rakes
    them there (so that the feathers are seen only where it does)."""
    h = np.where(where, height, 0.0) * rake(model)
    model.relief += h
    gy, gx = np.gradient(h)
    gx, gy = gx * where, gy * where
    n = model.normal - np.stack([gx, gy, np.zeros_like(gx)], -1)
    model.normal = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


# ----------------------------------------------------------------------
# lifting lines drawn on the bird onto its surfaces


def head_surface(xy) -> np.ndarray:
    """Points (N, 2) of the skull frame's side, lifted onto the near side
    of the head (N, 3): onto the eye where it stands out, else the head."""
    xy = np.asarray(xy, dtype=np.float64)
    head = head_blobs()
    z = soft_extent(
        [b for b in head if b.part == "plumage"],
        xy[:, 0],
        xy[:, 1],
        BLEND["plumage"] / UNIT,
    )
    eye = next(b for b in head if b.part == "eye")
    r2 = np.sum((xy - eye.centre[:2]) ** 2, axis=1)
    on_eye = eye.centre[2] + np.sqrt(np.clip(eye.radii[0] ** 2 - r2, 0, None))
    z = np.where(np.isfinite(z), z, 0.0)
    return np.concatenate(
        [xy, np.maximum(z, np.where(r2 < eye.radii[0] ** 2, on_eye, -9))[:, None]], 1
    )


def on_head(model: Model, xy) -> np.ndarray:
    """Points (N, 2) of the skull frame's side, on the sheet (N, 3)."""
    return model.pose.to_sheet(head_surface(xy), head=True)


def on_bill(model: Model, which: int, xy, side: float = 1.0) -> np.ndarray:
    """Points (N, 2) of the skull frame on mandible ``which`` (0 upper, 1
    lower), on its near side (or its middle, ``side`` 0), on the sheet."""
    patch = bill_patches()[which]
    p = patch.lift(xy)
    p[:, 2] *= side
    return model.pose.to_sheet(p, head=True)


def dashes(rng, line: np.ndarray, dash, gap) -> list[np.ndarray]:
    """A polyline broken into dashes (lengths ``dash`` and gaps ``gap``,
    each a (low, high) range, in its own units)."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out, t = [], rng.uniform(0, gap[1])
    while t < s[-1]:
        n = rng.uniform(*dash)
        ts = np.linspace(t, min(t + n, s[-1]), 6)
        out.append(
            np.stack([np.interp(ts, s, line[:, k]) for k in range(line.shape[1])], 1)
        )
        t += n + rng.uniform(*gap)
    return out


# ----------------------------------------------------------------------
# the head's features


PLUME = ("plumage",)


#: The centre the rows of tiny feathers across the bare face curve about
#: (skull frame), and the first and last row's radius: nine rows, as
#: counted on the photographs, closest under the eye.
FACE_CENTRE, FACE_ROWS = (0.78, 0.38), (0.98, 1.6)


def draw_face_lines(model: Model, ink: Marks, rng) -> None:
    """The rows of tiny dark feathers across the bare face: arcs about a
    point above the bill's base, from the face's back edge forward and down
    toward the gape, each a broken line of tiny tufts, uneven, and fading
    toward the bill."""
    ring = smooth_curve(np.vstack([face_ring(), face_ring()[:1]]), 300)
    cx, cy = FACE_CENTRE
    radius = FACE_ROWS[0]
    while radius < FACE_ROWS[1]:
        t = np.linspace(math.radians(195), math.radians(265), 160)
        wav = 0.012 * np.sin(t * rng.uniform(25, 45) + radius * 9) + 0.006 * np.sin(
            t * rng.uniform(70, 110)
        )
        arc = np.stack(
            [cx + (radius + wav) * np.cos(t), cy + (radius + wav) * np.sin(t)], 1
        )
        keep = inside(ring, arc[:, 0], arc[:, 1]) & (
            np.hypot(arc[:, 0], arc[:, 1]) > IRIS + 0.07
        )
        for run in runs(arc, keep):
            for dash in dashes(rng, run, (0.008, 0.045), (0.004, 0.04)):
                crowd = face_crowd(dash.mean(axis=0))
                if rng.random() < 0.5 * (1.0 - crowd):
                    continue
                d = rng.uniform(0.3, 0.85) * (0.6 + 0.4 * crowd)
                draw3d(
                    ink,
                    model,
                    on_head(model, dash),
                    rng.uniform(1.4, 2.0),
                    d,
                    6.0,
                    PLUME,
                )
        radius += rng.uniform(0.055, 0.09)


def face_crowd(p) -> float:
    """How densely the face's tiny feathers crowd at a point of the skull
    frame, 0..1: most near the bill's base and under the eye."""
    x, y = float(p[0]), float(p[1])
    base = float(smoothstep(0.15, 0.4, x))
    under = math.exp(-((math.hypot(x + 0.02, y + 0.3) / 0.2) ** 2))
    return max(base, under)


def runs(points: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
    """The unbroken stretches of ``points`` where ``keep`` holds."""
    out, start = [], None
    for i, k in enumerate(np.append(keep, False)):
        if k and start is None:
            start = i
        elif not k and start is not None:
            if i - start > 1:
                out.append(points[start:i])
            start = None
    return out


def draw_bill(model: Model, ink: Marks, rng) -> None:
    """The bill's drawing: the cutting edge with its notch, dark; the base
    where it meets the face; the line of the culmen; the growth lines and
    the grain of the horn, faint, along its arch."""
    upper = ("upper",)
    tomium = smooth_curve(bill_point(TOMIUM), 60)
    tomium = tomium + grain_offset(tomium, 0.012)
    draw3d(
        ink,
        model,
        on_bill(model, 0, tomium),
        2.4,
        np.linspace(0.85, 0.6, 60),
        5.0,
        upper,
    )
    base = smooth_curve(bill_point(BILL_BASE), 50)
    base = base + grain_offset(base, 0.012)
    draw3d(ink, model, on_bill(model, 0, base), 2.2, 0.55, 5.0, upper)
    culmen = smooth_curve(bill_point(CULMEN), 80)
    for k in range(7):
        grain = culmen + grain_offset(culmen, -(0.04 + 0.05 * k))
        for dash in dashes(rng, grain, (0.12, 0.4), (0.04, 0.15)):
            draw3d(
                ink,
                model,
                on_bill(model, 0, dash),
                1.0,
                rng.uniform(0.08, 0.18),
                4.0,
                upper,
            )


def grain_offset(curve: np.ndarray, off: float) -> np.ndarray:
    """How far to move a curve's points to run ``off`` inside it (toward
    the bill's back)."""
    t = np.gradient(curve, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return np.stack([t[:, 1], -t[:, 0]], 1) * -off


# ----------------------------------------------------------------------
# light

#: Toward the light, on the sheet (x right, y down, z toward the viewer).
SUN = LIGHT * np.array([1.0, -1.0, 1.0])


def occlusion(model: Model) -> np.ndarray:
    """How much each surface is shut in by those near it, 0..1: crevices,
    the slot under the wing, the neck under the head."""
    z = np.where(np.isfinite(model.seen.z), model.seen.z, np.nan)
    near = np.nan_to_num(z, nan=np.nanmin(z) - 80.0).astype(np.float32)
    out = np.zeros(z.shape)
    for r, w in ((4, 0.5), (12, 0.35), (30, 0.25)):
        around = box_blur(box_blur(near, r), r)
        out += w * np.clip((around - near) / (1.6 * r), 0.0, 1.0)
    return np.where(np.isfinite(model.seen.z), np.clip(out, 0.0, 1.0), 0.0)


def illumination(model: Model) -> np.ndarray:
    """How brightly each seen surface is lit, about 0..1.1: the sun (wrapped
    a little on the soft plumage), cast shadows, the light thrown back up
    from below into the shadows, and occlusion."""
    n = model.normal
    soft_parts = model.part("plumage", "wing")
    wrap = np.where(soft_parts, 0.3, 0.08)
    sun = np.clip((n @ SUN + wrap) / (1.0 + wrap), 0.0, 1.0) ** 1.15
    sun *= 1.0 - 0.95 * model.shadow
    bounce = np.clip(n @ np.array([0.15, 0.85, 0.5]), 0.0, 1.0) * 0.12
    sky = 0.06 + 0.06 * np.clip(-n[..., 1], 0.0, 1.0)
    occ = occlusion(model)
    return (sky + bounce + sun) * (1.0 - 0.75 * occ) * EXPOSURE


def specular(model: Model, power: float) -> np.ndarray:
    """The sheen of a hard surface (horn, claw, eye), 0..1."""
    h = SUN + np.array([0.0, 0.0, 1.0])
    h /= np.linalg.norm(h)
    return np.clip(model.normal @ h, 0.0, 1.0) ** power * (1.0 - model.shadow)


# ----------------------------------------------------------------------
# the wing's feathers

#: The covert rows' coordinate: how far back along the wing (units) the
#: rows' tips lie, a row being tilted back toward the lower edge and bowed
#: toward the tip in the middle.
ROW_SLANT, ROW_BOW, ROW_MID = 0.35, 0.25, 0.07
#: Where the greater coverts' tips lie (units back from the wrist): 42% of
#: the way to the wing's tip, as measured.
GREATER = 0.42


def covert_rows(rng: np.random.Generator) -> list[dict]:
    """The rows of coverts, from the wrist back: each row's tip position
    (units, in the row coordinate), its feathers (where across the wing
    each begins, so each one's width), and how deep their rounded tips are.
    The feathers grow down the wing, the greater coverts largest."""
    _, _, _, length = wing_frame()
    rows, d, w = [], 0.34, 0.15
    end = GREATER * length - 0.8
    while d < end:
        rows.append(row_of(rng, d, w, 0.8))
        d += 0.8 * w * rng.uniform(0.85, 1.15)
        w = min(w * 1.28, 0.44)
    rows.append(row_of(rng, GREATER * length, 0.44, 0.7))
    return rows


def row_of(rng: np.random.Generator, d: float, w: float, depth: float) -> dict:
    """One row of coverts: tips about ``d`` back, feathers about ``w`` wide,
    neither all of one width nor evenly set."""
    edges = [-1.0 + rng.uniform(0.0, w)]
    while edges[-1] < 1.2:
        edges.append(edges[-1] + w * rng.uniform(0.65, 1.35))
    n = len(edges)
    jitter = rng.normal(0.0, 0.16 * w, n)
    depths = depth * rng.uniform(0.6, 1.25, n)
    skew = rng.uniform(-0.3, 0.3, n)
    return {
        "d": d,
        "edges": np.array(edges),
        "jitter": jitter,
        "depth": depths,
        "skew": skew,
        "w": w,
    }


def row_coordinate(u, v):
    """The covert rows' coordinate at wing coordinates (u, v)."""
    _, _, _, length = wing_frame()
    return u * length - ROW_SLANT * (v - ROW_MID) + ROW_BOW * (v - ROW_MID) ** 2


def row_tip(row: dict, v) -> np.ndarray:
    """Where the tips of a row lie (row coordinate) at each ``v`` across the
    wing: rounded tips, one per feather, a little uneven."""
    e = row["edges"]
    j = np.clip(np.searchsorted(e, v) - 1, 0, len(e) - 2)
    a, b = e[j], e[j + 1]
    half = 0.53 * (b - a)
    theta = np.clip((v - 0.5 * (a + b)) / half + row["skew"][j], -1.0, 1.0)
    return (
        row["d"]
        + row["jitter"][j]
        - row["depth"][j] * (b - a) * (1 - np.sqrt(1 - theta**2))
    )


def covert_relief(u, v, rows):
    """Over the coverts: which row each place lies in (-1 past them), how
    far from the row above's tips to its own (0..1), and where across its
    own feather (0 at the edge the next one covers, 1 at its free edge)."""
    d = row_coordinate(u, v)
    which = np.full(u.shape, -1)
    q = np.zeros(u.shape)
    side = np.zeros(u.shape)
    prev = np.full(u.shape, -1.0)
    for k, row in enumerate(rows):
        tip = row_tip(row, v)
        here = (which < 0) & (d <= tip)
        which = np.where(here, k, which)
        q = np.where(here, np.clip((d - prev) / np.maximum(tip - prev, 1e-6), 0, 1), q)
        e = row["edges"]
        j = np.clip(np.searchsorted(e, v) - 1, 0, len(e) - 2)
        side = np.where(here, np.clip((v - e[j]) / (e[j + 1] - e[j]), 0, 1), side)
        prev = tip
    return which, q, side


#: The flight feathers seen below the greater coverts, from the top (the
#: tertials, over the back) down to the primaries at the lower edge: where
#: each one's free lower edge lies across the wing (0 at its upper edge, 1
#: at its lower) and how far along the wing it reaches.
FLIGHT = [(0.16, 0.6), (0.3, 0.63), (0.42, 0.66), (0.53, 0.69), (0.63, 0.72),
          (0.72, 0.76), (0.8, 0.86), (0.87, 0.93), (0.935, 0.97),
          (1.0, 1.0)]  # fmt: skip


def wing_edges(u):
    """Where the wing's upper and lower edges lie across it (v) at ``u``."""
    a, along, across, length = wing_frame()
    ring = smooth_curve(np.vstack([WING_OUTLINE, WING_OUTLINE[:1]]), 400)
    ru, rv = wing_uv(ring[:, 0], ring[:, 1])
    tip = int(np.argmax(ru))
    top, bottom = slice(0, tip + 1), slice(tip, None)
    order_t, order_b = np.argsort(ru[top]), np.argsort(ru[bottom])
    up = np.interp(u, ru[top][order_t], rv[top][order_t])
    low = np.interp(u, ru[bottom][order_b], rv[bottom][order_b])
    del a, along, across, length
    return np.minimum(up, low), np.maximum(up, low)


def flight_relief(u, v):
    """Over the flight feathers: which one is seen at each place (the
    uppermost that reaches it), and how far across from the edge of the
    one above to its own free edge (0..1)."""
    up, low = wing_edges(u)
    f = (v - up) / np.maximum(low - up, 1e-6)
    which = np.full(u.shape, -1)
    q = np.zeros(u.shape)
    prev = 0.0
    for j, (edge, reach) in enumerate(FLIGHT):
        top = edge - 0.24
        theta = np.clip((f - (edge - 0.12)) / 0.12, -1, 1)
        end = reach - 0.05 * (1 - np.sqrt(1 - theta**2))
        here = (which < 0) & (f <= edge) & (f >= top) & (u <= end)
        which = np.where(here, j, which)
        q = np.where(
            here,
            np.clip((f - max(prev, top)) / max(edge - max(prev, top), 1e-6), 0, 1),
            q,
        )
        prev = edge
    return which, q


def feather_wing(model: Model, rng: np.random.Generator) -> dict:
    """The wing's feathers on the model: its relief (each feather rising to
    its free edge, then a step down to the next), and for each place which
    feather it is on."""
    wing = model.part("wing")
    b = np.nan_to_num(model.bird)
    u, v = wing_uv(b[..., 0], b[..., 1])
    rows = covert_rows(rng)
    row, q, side = covert_relief(u, v, rows)
    blade, qf = flight_relief(u, v)
    covert = wing & (row >= 0)
    flight = wing & (row < 0)
    size = np.array([r["w"] for r in rows])[np.clip(row, 0, len(rows) - 1)]
    lift = np.where(covert, 9.0 * size**1.3 / 0.38**0.3, 2.4)
    height = (
        np.where(covert, 0.75 * q**0.8 + 0.25 * side, np.clip(qf, 0, 1) ** 0.8) * lift
    )
    emboss(model, box_blur(height.astype(np.float32), 1), wing)
    return {
        "u": u,
        "v": v,
        "rows": rows,
        "row": np.where(wing, row, -2),
        "q": q,
        "blade": np.where(flight, blade, -2),
        "qf": qf,
    }


# ----------------------------------------------------------------------
# the small feathers of the head and body


def flow2d(model: Model):
    """The way the feathers run at each pixel, on the sheet: a unit vector
    (H, W, 2) and how much it is foreshortened (0..1)."""
    f = model.seen.flow[..., :2]
    size = np.linalg.norm(f, axis=-1)
    return f / np.maximum(size, 1e-9)[..., None], np.clip(size, 0.15, 1.0)


def feather_size(model: Model) -> np.ndarray:
    """How long the small feathers are at each pixel (units): tiny on the
    face's edge and forehead, small over the crown, larger down the neck,
    largest over the breast and belly, small again on the thighs."""
    b = np.nan_to_num(model.bird)
    head = model.headness
    down = smoothstep(-0.8, -3.2, b[..., 1])
    body = 0.09 + 0.08 * down
    thigh = smoothstep(0.75, 0.4, implicit_map(model, THIGHS))
    body = body * (1 - 0.4 * thigh)
    return head * 0.055 + (1 - head) * body


def implicit_map(model: Model, ellipsoids) -> np.ndarray:
    """How far inside the nearest of these ellipsoids (centre, radii) each
    pixel's point of the bird lies (below 1 inside)."""
    b = model.bird
    best = np.full(b.shape[:2], np.inf)
    for c, r in ellipsoids:
        q = (np.nan_to_num(b, nan=99.0) - np.asarray(c)) / np.asarray(r)
        np.minimum(best, np.sqrt(np.sum(q * q, axis=-1)), out=best)
    return best


def scatter(
    where: np.ndarray, spacing: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Points (N, 2) over ``where``, about ``spacing`` (pixels, a map) apart:
    none closer than that, none of the region far from one."""
    ys, xs = np.nonzero(where)
    if not len(xs):
        return np.zeros((0, 2))
    r = spacing[ys, xs]
    count = int(np.sum(1.0 / r**2) * 2.2)
    pick = rng.choice(len(xs), size=min(count, len(xs)), replace=False)
    cell = float(r.min()) / 1.5
    grid: dict[tuple[int, int], list[int]] = {}
    pts = []
    for i in pick:
        x, y, ri = xs[i] + rng.random(), ys[i] + rng.random(), r[i]
        cx, cy = int(x / cell), int(y / cell)
        reach = int(math.ceil(ri / cell))
        if any(
            (pts[k][0] - x) ** 2 + (pts[k][1] - y) ** 2 < (0.5 * (ri + pts[k][2])) ** 2
            for gx in range(cx - reach, cx + reach + 1)
            for gy in range(cy - reach, cy + reach + 1)
            for k in grid.get((gx, gy), ())
        ):
            continue
        grid.setdefault((cx, cy), []).append(len(pts))
        pts.append((x, y, ri))
    return np.array(pts)[:, :2]


@dataclass
class Tiles:
    """Feathers laid as overlapping tiles on the sheet: which one is seen at
    each pixel, how far toward its tip (-1..0.5 of its length), across it
    (-1..1), and its seeds."""

    which: np.ndarray
    along: np.ndarray
    across: np.ndarray
    seeds: np.ndarray


def lay_tiles(model: Model, where: np.ndarray, size_px: np.ndarray, rng) -> Tiles:
    """Small feathers over ``where``, each ``size_px`` long, as the way the
    feathers run lays them: each lying over the base of the next, so at any
    place the one seen is the one nearest its own tip there."""
    direction, short = flow2d(model)
    seeds = scatter(where, np.maximum(0.62 * size_px * np.sqrt(short), 2.5), rng)
    h, w = where.shape
    best = np.full((h, w), -9.0)
    which = np.full((h, w), -1)
    along = np.zeros((h, w))
    across = np.zeros((h, w))
    for k, (x, y) in enumerate(seeds):
        ix, iy = int(x), int(y)
        d = direction[iy, ix]
        length = size_px[iy, ix] * short[iy, ix] * rng.uniform(0.85, 1.15)
        width = size_px[iy, ix] * rng.uniform(0.75, 0.95)
        r = int(length + width) + 2
        x0, x1, y0, y1 = (
            max(ix - r, 0),
            min(ix + r + 1, w),
            max(iy - r, 0),
            min(iy + r + 1, h),
        )
        gy, gx = np.mgrid[y0:y1, x0:x1] + 0.5
        sa = ((gx - x) * d[0] + (gy - y) * d[1]) / length
        ta = ((gx - x) * -d[1] + (gy - y) * d[0]) / (0.5 * width)
        tip = 0.5 * np.sqrt(np.clip(1 - ta**2, 0, 1))
        cover = (sa >= -1.0) & (sa <= tip) & (np.abs(ta) <= 1) & where[y0:y1, x0:x1]
        prio = sa + 0.02 * rng.random()
        take = cover & (prio > best[y0:y1, x0:x1])
        best[y0:y1, x0:x1] = np.where(take, prio, best[y0:y1, x0:x1])
        which[y0:y1, x0:x1] = np.where(take, k, which[y0:y1, x0:x1])
        along[y0:y1, x0:x1] = np.where(take, sa, along[y0:y1, x0:x1])
        across[y0:y1, x0:x1] = np.where(take, ta, across[y0:y1, x0:x1])
    return Tiles(which, along, across, seeds)


def feather_plumage(model: Model, rng: np.random.Generator) -> Tiles:
    """The small feathers of the head, neck, breast, belly and thighs, laid
    as tiles in relief: each rising toward its rounded tip, a step down to
    the next."""
    where = model.part("plumage") & (face_mask(model) < 0.5) & (brow_mask(model) < 0.5)
    size = feather_size(model) * UNIT
    tiles = lay_tiles(model, where, size, rng)
    lift = 0.05 * size
    height = np.where(
        tiles.which >= 0, (0.35 + 0.65 * np.clip(tiles.along + 1.0, 0, 1.5) / 1.5), 0
    )
    height = height * lift - 0.08 * lift * np.abs(tiles.across)
    soft = box_blur(box_blur(height.astype(np.float32), 1), 1)
    emboss(model, soft, where & (tiles.which >= 0))
    return tiles


# ----------------------------------------------------------------------
# lines of the wing, tail and feet


def on_wing(model: Model, uv) -> np.ndarray:
    """Points (u, v) of the folded wing, on its surface, on the sheet."""
    uv = np.asarray(uv, dtype=np.float64)
    a, along, across, length = wing_frame()
    xy = a + np.outer(uv[:, 0] * length, along) + np.outer(uv[:, 1], across)
    return model.pose.to_sheet(bird_patches()[1][0].lift(xy))


def shade_at(model: Model, pts: np.ndarray) -> np.ndarray:
    """How far into shadow each point (N, 3) of the sheet lies, 0..1, and how
    much the light rakes it there, as weights for a line."""
    h, w = model.seen.z.shape
    ix = np.clip(pts[:, 0].astype(int), 0, w - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, h - 1)
    return rake(model)[iy, ix]


def draw_flight(model: Model, ink: Marks, rng) -> None:
    """The flight feathers: the free edge of each, a crisp dark line where it
    lies over the next, and its rounded end; firm where the light rakes."""
    for edge, reach in FLIGHT[:-1]:
        u = np.linspace(GREATER - 0.02, reach - 0.02, 60)
        up, low = wing_edges(u)
        v = up + edge * (low - up)
        pts = on_wing(model, np.stack([u, v], 1))
        dark = (0.35 + 0.45 * shade_at(model, pts)) * np.sin(
            np.linspace(0.15, 2.9, 60)
        ) ** 0.4
        draw3d(ink, model, pts, rng.uniform(1.6, 2.2), dark, 4.0, ("wing",))
        theta = np.linspace(-1.0, 1.0, 16)
        f = edge - 0.12 + 0.12 * theta
        uu = reach - 0.05 * (1 - np.sqrt(1 - theta**2))
        up, low = wing_edges(uu)
        end = on_wing(model, np.stack([uu, up + f * (low - up)], 1))
        draw3d(
            ink,
            model,
            end,
            1.8,
            0.55 * np.sin(np.linspace(0.2, 2.9, 16)),
            4.0,
            ("wing",),
        )


def draw_coverts(model: Model, ink: Marks, wing: dict, rng, count: int = 4) -> None:
    """The tips of the larger coverts, each a soft dark line along its edge,
    stated where the light rakes the wing and left out elsewhere."""
    rows = wing["rows"]
    for row in rows[-count:]:
        e = row["edges"]
        for a, b in pairwise(e):
            v = np.linspace(a, b, 14)
            d = row_tip(row, v)
            u = (
                d + ROW_SLANT * (v - ROW_MID) - ROW_BOW * (v - ROW_MID) ** 2
            ) / wing_frame()[3]
            pts = on_wing(model, np.stack([u + 0.004, v], 1))
            weight = shade_at(model, pts)
            if rng.random() > 0.45 + 0.5 * float(weight.mean()):
                continue
            dark = (0.3 + 0.45 * weight) * np.sin(np.linspace(0.1, 3.0, 14))
            draw3d(ink, model, pts, rng.uniform(1.4, 2.4), dark, 4.0, ("wing",))


def draw_tail(model: Model, ink: Marks, rng) -> None:
    """The tail feathers' shafts, fine and broken, along each one's middle."""
    for k, (_, frac, _, _) in enumerate(BLADES):
        origin, dk, side, n = blade_basis(k)
        t = np.linspace(0.03, 0.93, 90) * TAIL_LENGTH * frac
        patch = bird_patches()[1][1 + k]
        local = np.stack([t, np.zeros_like(t)], 1)
        pts = model.pose.to_sheet(patch.lift(local) + n * 0.002)
        lift = np.convolve(rng.random(90) < 0.12, np.ones(5), mode="same") > 0
        dark = np.where(lift, 0.0, 0.35 * (1.0 - 0.5 * t / t[-1]))
        draw3d(ink, model, pts, 1.3, dark, 3.0, (f"tail{k}",))
        del origin, dk, side


def draw_feet(model: Model, ink: Marks) -> None:
    """The scales across the toes, as rings; the claws, dark."""
    b, up, side = branch_frame()
    for f in FEET:
        c = np.asarray(BRANCH_AT) + b * f
        for start, stop, along, th in TOES:
            for a in np.linspace(start, stop, 8)[1:-1]:
                ang = math.radians(a)
                frac = (a - start) / (stop - start)
                r = BRANCH_R + th * (1 - 0.35 * frac) * 0.8
                centre = (
                    c
                    + side * r * math.cos(ang)
                    + up * r * math.sin(ang)
                    + b * along * frac
                )
                radial = side * math.cos(ang) + up * math.sin(ang)
                ring = [
                    centre + radial * th * (1 - 0.35 * frac) * q + b * th * 0.9 * w
                    for q, w in ((1.0, -1.0), (1.25, 0.0), (1.0, 1.0))
                ]
                pts = model.pose.to_sheet(np.array(ring))
                draw3d(ink, model, pts, 1.3, [0.1, 0.5, 0.1], 4.0, ("toes",))


# ----------------------------------------------------------------------
# where the bare face, the red forehead and the tail's parts lie


class Marks(Protocol):
    """Whatever lays a line on the sheet: through ``pts`` (N, 2), ``width``
    and ``dark`` (each a number or one per point) along it."""

    def line(self, pts, width, dark) -> None: ...


def face_mask(model: Model) -> np.ndarray:
    """Where the bare skin of the face is seen (0..1): inside its border on
    either side of the head, behind the bill."""
    sk = model.skull
    x, y, z = sk[..., 0], sk[..., 1], sk[..., 2]
    ring = smooth_curve(np.vstack([face_ring(), face_ring()[:1]]), 300)
    ok = np.isfinite(x)
    out = np.zeros(x.shape)
    inner = inside(ring, np.where(ok, x, 9.0), np.where(ok, y, 9.0))
    out[inner & (np.abs(np.nan_to_num(z)) > 0.1)] = 1.0
    return ragged(out * model.part("plumage") * (model.headness > 0.15), 3.0, 41)


def face_ring() -> np.ndarray:
    """The border of the bare face (skull frame), its front edge a little
    under the bill's base."""
    front = bill_point(BILL_BASE[::-1]) + np.array([0.08, 0.0])
    return np.vstack([FACE_EDGE, bill_point(LOWER[-1:])[0] + [0.05, 0.0], front])


def brow_mask(model: Model) -> np.ndarray:
    """Where the red feathers of the forehead are seen."""
    sk = model.skull
    ring = smooth_curve(np.vstack([BROW_EDGE, BROW_EDGE[:1]]), 300)
    ok = np.isfinite(sk[..., 0])
    inner = inside(ring, np.where(ok, sk[..., 0], 9.0), np.where(ok, sk[..., 1], 9.0))
    return ragged(inner * model.part("plumage") * (model.headness > 0.5) * 1.0, 2.5, 43)


def ragged(mask: np.ndarray, cell: float, seed: int) -> np.ndarray:
    """``mask`` (0..1) with its edge broken, as where small feathers lie
    over the edge of a patch: breaks about ``cell`` pixels across."""
    rng = np.random.default_rng(seed)
    grain = noise.value_noise_2d(mask.shape[0], mask.shape[1], cell, rng) - 0.5
    soft = box_blur(box_blur(mask.astype(np.float32), 2), 2)
    return ((soft + 0.9 * grain > 0.5) & (soft > 0.05)).astype(np.float64)


def blade_coords(model: Model, k: int):
    """How far along tail feather ``k`` (0..1) and across it (-1..1) each
    pixel lies."""
    origin, dk, side, _ = blade_basis(k)
    _, frac, _, half = BLADES[k]
    rel = np.nan_to_num(model.bird) - origin
    return (rel @ dk) / (TAIL_LENGTH * frac), (rel @ side) / half


def visible(model: Model, pts: np.ndarray, tol: float = 3.0, parts=None) -> np.ndarray:
    """Which of the points (N, 3) on the sheet are not hidden behind a
    nearer surface (and lie on one of ``parts``, if given)."""
    h, w = model.seen.z.shape
    ix = np.clip(pts[:, 0].astype(int), 0, w - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, h - 1)
    z = model.seen.z[iy, ix]
    on = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    on &= np.isfinite(z) & (pts[:, 2] >= z - tol)
    if parts is not None:
        on &= model.part(*parts)[iy, ix]
    return on


def draw3d(ink: Marks, model: Model, pts3, width, dark, tol=3.0, parts=None) -> None:
    """A line through points (N, 3) on the sheet, lifted where hidden."""
    pts3 = np.asarray(pts3, dtype=np.float64)
    seen = visible(model, pts3, tol, parts)
    d = np.broadcast_to(np.asarray(dark, float), len(pts3)) * seen
    ink.line(pts3[:, :2], width, d)


def branch_fade(model: Model) -> np.ndarray:
    """How much of the branch is drawn: all of it under the feet, fading out
    either side, so that the bird is the subject."""
    x, y, _ = model.pose.to_sheet(np.asarray(BRANCH_AT))
    h, w = model.seen.z.shape
    yy, xx = np.mgrid[:h, :w]
    far = np.hypot(xx - x - 70.0, (yy - y) * 1.5)
    fade = np.exp(-((far / 300.0) ** 3))
    near = box_blur(model.part("branch").astype(np.float32), 3) > 0.01
    bird = (
        box_blur(
            (np.isfinite(model.seen.z) & ~model.part("branch")).astype(np.float32), 2
        )
        > 0.01
    )
    return np.where(near & ~bird, fade, 1.0)
