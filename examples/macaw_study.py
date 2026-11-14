"""A structural study of a great green macaw: the bird built as solids.

Run from the repository root: ``python examples/macaw_study.py OUTDIR``.
Writes ``study_grey.png``, ``study_colour.png``, ``head.png`` and the
renders the silhouette check is made from into OUTDIR.

This is the under-drawing of ``macaw.py``, not a drawing in pencil: the bird
modelled from measurements, lit, and its feather tracts and features drawn
on the model in line and tone.

The measurements (eleven reference photographs, CC BY / CC0, read off on
grids; nothing sampled from them). Head, in units of the distance from the
eye to the base of the culmen, the bird facing +x, y up:

* the bill's chord (base of culmen to the hook's tip) is 1.38 long and
  turns 102 degrees down from the line eye-to-culmen-base (seven photos,
  sd 2.5 degrees): the tip hangs almost under the base;
* the culmen bulges forward of that chord by 0.26 of its length, the bulge
  highest at 0.43 of the way down, leaving the base steeply;
* the upper mandible's base runs back from the culmen to the gape, which
  lies 0.62 along the chord and 0.47 behind it; the lower mandible's chin
  reaches 1.1 chords down and 0.5 behind;
* the bare face reaches 0.16 above the eye, 0.46 behind it and 0.96 below;
  the red forehead runs from 0.32 in front of the eye to the bill, its top
  0.35 above the eye level, and laps over the culmen's base;
* the crown is only 0.35 above the eye: a flat head, the eye set high.

Whole bird, in units of the eye-to-bill-tip distance (1.53 of the above):
shoulder 1.2 from the eye; folded wing 4.1-4.4 from shoulder to tip, its
greater coverts ending 42% of the way; breast front 1.6 below and a little
behind the eye; feet 2.5; tail tip 8.1-9.1; the tail hangs within 5-10
degrees of vertical even when the body leans.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from murex import write_png
from murex.filters import box_blur, smoothstep

WIDTH, HEIGHT = 1100, 1540
#: Pixels to one unit of the head (eye to the base of the culmen).
UNIT = 124.0
#: Where the eye lands on the sheet.
EYE_AT = (300.0, 215.0)
SEED = 11
#: Toward the light (x right, y up, z toward the viewer): above, front-left.
LIGHT = np.array([-0.5, 0.62, 0.6]) / np.linalg.norm([-0.5, 0.62, 0.6])

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
          (0.66, 0.045), (0.73, -0.02), (0.84, -0.045), (0.93, -0.03), (1.0, 0.0)]  # fmt: skip
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
IRIS, PUPIL, RING = 0.075, 0.034, 0.17


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
        x0, y0, step, h = self.grid
        xy = np.asarray(xy, dtype=np.float64)
        fx = np.clip((xy[:, 0] - x0) / step, 0, h.shape[1] - 1.001)
        fy = np.clip((xy[:, 1] - y0) / step, 0, h.shape[0] - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        ax, ay = fx - ix, fy - iy
        z = (h[iy, ix] * (1 - ax) * (1 - ay) + h[iy, ix + 1] * ax * (1 - ay)
             + h[iy + 1, ix] * (1 - ax) * ay + h[iy + 1, ix + 1] * ax * ay)
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
    for (ax, ay), (bx, by) in zip(curve[:-1], curve[1:], strict=True):
        ex, ey = bx - ax, by - ay
        t = np.clip(((x - ax) * ex + (y - ay) * ey) / max(ex * ex + ey * ey, 1e-12), 0, 1)
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
    return np.concatenate([points, points * flip]), np.concatenate([normals, normals * flip])


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


def section_cuts(outline: np.ndarray, spine: np.ndarray):
    """Where the line across ``spine`` at each of its points meets the
    closed ``outline`` (both (N, 2)): the two ends of each cross-section."""
    tangent = np.gradient(spine, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    a, b = outline, np.roll(outline, -1, axis=0)
    ends = []
    for p, n in zip(spine, normal, strict=True):
        # p + s n = a + t (b - a), for every edge at once
        e = b - a
        det = n[0] * -e[:, 1] + n[1] * e[:, 0]
        ok = np.abs(det) > 1e-12
        d = a - p
        s = np.where(ok, (d[:, 0] * -e[:, 1] + d[:, 1] * e[:, 0]) / np.where(ok, det, 1), 0)
        t = np.where(ok, (n[0] * d[:, 1] - n[1] * d[:, 0]) / np.where(ok, det, 1), -1)
        hit = ok & (t >= 0) & (t <= 1)
        below, above = s[hit & (s <= 0)], s[hit & (s >= 0)]
        lo = below.max() if below.size else 0.0
        hi = above.min() if above.size else 0.0
        ends.append((p + lo * n, p + hi * n))
    return np.array(ends)


def horn(outline, spine, widths, part, n=90, flow=None):
    """A solid of horn (a mandible) from its outline in profile (skull
    frame): ellipsoids across it along ``spine``, as wide as ``widths``
    (from its root to its end)."""
    spine2 = smooth_curve(spine, n)
    ring = np.asarray(outline, dtype=np.float64)
    ring = smooth_curve(np.concatenate([ring, ring[:1]]), 400)
    ends = section_cuts(ring, spine2)
    mid = 0.5 * (ends[:, 0] + ends[:, 1])
    half = 0.5 * np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    w = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(widths)), widths)
    spine3 = np.concatenate([mid, np.zeros((n, 1))], axis=1)
    blobs = chain(spine3, half, w, part)
    if flow is not None:
        for b in blobs:
            b.flow = np.asarray(flow, dtype=np.float64)
    return blobs


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
    upper = np.concatenate([bill_point(CULMEN), bill_point(TOMIUM[::-1][1:]),
                            bill_point(BILL_BASE[::-1][1:-1])])
    upper_spine = bill_point([(0.1, 0.04), (0.3, 0.06), (0.47, 0.03), (0.6, -0.08),
                              (0.75, -0.12), (0.88, -0.08), (0.97, -0.02), (0.995, 0.0)])
    lower = bill_point(LOWER)
    lower_spine = bill_point([(0.66, 0.6), (0.82, 0.58), (0.96, 0.46), (1.02, 0.3),
                              (0.97, 0.14)])
    del upper_spine, lower_spine
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


def bill_patches() -> list[Patch]:
    """The two mandibles, as horn."""
    upper = np.concatenate([bill_point(CULMEN), bill_point(TOMIUM[::-1][1:]),
                            bill_point(BILL_BASE[::-1][1:-1])])
    lower = bill_point(LOWER)
    chin = bill_point(LOWER[4:10])
    upper_free = np.concatenate([bill_point(CULMEN), bill_point(TOMIUM[::-1][1:])])
    lower_free = bill_point(LOWER[:-1])
    return [
        mandible(upper, upper_free, bill_point(CULMEN), [0.28, 0.27, 0.23, 0.16, 0.08, 0.025],
                 "upper", (0.3, 0.1)),
        mandible(lower, lower_free, chin, [0.3, 0.3, 0.28, 0.24, 0.16, 0.08], "lower",
                 (0.3, 0.08)),
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
    ((-1.6, -1.85), (1.4, 1.25, 1.08), -128.0),
    ((-2.25, -3.25), (1.3, 1.05, 1.0), -130.0),
    ((-3.2, -4.35), (1.0, 0.65, 0.68), -130.0),
    ((-2.6, -2.4), (1.9, 0.62, 0.8), -130.0),
]
THIGHS = [((-1.9, -3.6, z), (0.52, 0.46, 0.4)) for z in (0.42, -0.42)]


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
    return Patch(o + pts @ m.T, nrm @ m.T, flow, part, (gx[0, 0], gy[0, 0], step, height), basis)


def wing_frame():
    """The folded wing's start (the wrist), its unit direction along it, and
    across it (toward the front and down), and its length."""
    a, b = np.array(SHOULDER), np.array(WING_TIP)
    length = float(np.linalg.norm(b - a))
    along = (b - a) / length
    return a, along, np.array([-along[1], along[0]]) * -1.0, length


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
          (-0.21, 0.76, -3.5, 0.24), (0.33, 0.6, 6.0, 0.23), (-0.34, 0.56, -6.0, 0.23)]  # fmt: skip


def blade_outline(length: float, half: float) -> np.ndarray:
    """A long tail feather in its own plane: from its root, narrowing
    a little, its end rounded to a blunt point."""
    t = np.linspace(0.0, 1.0, 60)
    w = half * np.where(t < 0.72, 1.0 - 0.22 * t, 0.84 * np.sqrt(np.clip((1 - t) / 0.28, 0, 1)))
    w = np.maximum(w, 0.004)
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
    a = np.radians(np.linspace(*turn, 16))
    r = BRANCH_R + np.linspace(*thick, 16) * 0.8
    spine = (
        centre + np.outer(np.cos(a) * r, side) + np.outer(np.sin(a) * r, up)
        + np.outer(np.linspace(*along, 16), b)
    )
    radius = np.linspace(*thick, 16)
    return chain(spine, radius, radius, part, side=b, overlap=3.0)


#: The toes of a foot: round the branch from the top (degrees), sideways
#: along it, thickness: two over the near side, two over the far side.
TOES = ((95, -35, 0.2, 0.105), (95, -25, -0.12, 0.105), (95, 215, 0.14, 0.1),
        (95, 205, -0.16, 0.1))  # fmt: skip


def feet_blobs() -> list[Blob]:
    """The two feet gripping the branch, two toes on either side of it, each
    with its claw, and the short scaly leg from the thigh."""
    b, up, side = branch_frame()
    out = []
    for f in FEET:
        c = np.asarray(BRANCH_AT) + b * f
        top = c + up * (BRANCH_R + 0.08)
        leg = np.linspace(top + up * 0.5 + side * 0.08, top, 8)
        out += chain(leg, [0.14] * 8, [0.13] * 8, "toes", side=b, overlap=3.0)
        for start, stop, along, th in TOES:
            out += toe(c, (start, stop), (0.0, along), (th, th * 0.65))
            end = stop + (-42 if stop < 90 else 42)
            out += toe(c, (stop, end), (along, along * 1.1), (th * 0.55, 0.015), "claws")
    return out


def bird_blobs() -> tuple[list[Blob], list[Blob]]:
    """All of the bird's ellipsoids, in its own frame: those of the head
    (which turn with it) and the rest."""
    return head_blobs(), body_blobs() + feet_blobs() + branch_blobs()


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
            Blob(self.to_sheet(b.centre, head), m @ b.frame, b.radii * UNIT, b.part, m @ b.flow)
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
        blobs = [Blob(m @ b.centre - sh, m @ b.frame, b.radii, b.part, b.flow) for b in self.blobs]
        patches = [Patch(p.points @ m.T - sh, p.normals @ m.T, p.flow @ m.T, p.part)
                   for p in self.patches]
        return Scene(blobs, patches)


def place_patch(pose: Pose, patch: Patch, head: bool) -> Patch:
    """A surface on the sheet."""
    m = pose.linear(head)
    return Patch(pose.to_sheet(patch.points, head), patch.normals @ m.T, patch.flow @ m.T,
                 patch.part, patch.grid, patch.basis)


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
    x0, y0 = int(max(math.floor(b.centre[0] - e[0]), 0)), int(max(math.floor(b.centre[1] - e[1]), 0))
    x1, y1 = int(min(math.ceil(b.centre[0] + e[0]) + 1, w)), int(min(math.ceil(b.centre[1] + e[1]) + 1, h))
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
    out = Render(np.full(shape, -np.inf), np.full(shape, -1, np.int16),
                 np.zeros((*shape, 3)), np.zeros((*shape, 3)), names)
    for i, name in enumerate(names):
        z, n, f = render_any(scene, name, shape)
        front = z > out.z
        out.z[front], out.part[front] = z[front], i
        out.normal[front], out.flow[front] = n[front], f[front]
    return out


def light_frame() -> np.ndarray:
    """The rotation that turns the light's direction (on the sheet) into +z."""
    l = LIGHT * np.array([1.0, -1.0, 1.0])
    e1 = np.cross([0.0, 1.0, 0.0], l)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(l, e1), l])


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
    normal: np.ndarray = None
    relief: np.ndarray = None

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


def emboss(model: Model, height: np.ndarray, where: np.ndarray) -> None:
    """Raise the relief ``height`` (pixels) over ``where`` on the model's
    surfaces: its slopes tilt their normals."""
    h = np.where(where, height, 0.0)
    model.relief += h
    gy, gx = np.gradient(h)
    gx, gy = gx * where, gy * where
    n = model.normal - np.stack([gx, gy, np.zeros_like(gx)], -1)
    model.normal = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


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
    sun *= 1.0 - 0.92 * model.shadow
    bounce = np.clip(n @ np.array([0.15, 0.85, 0.5]), 0.0, 1.0) * 0.16
    sky = 0.16 + 0.08 * np.clip(-n[..., 1], 0.0, 1.0)
    occ = occlusion(model)
    return (sky + bounce + 0.9 * sun) * (1.0 - 0.7 * occ)


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
ROW_SLANT, ROW_BOW = 0.35, 0.25
#: Where the greater coverts' tips lie (units back from the wrist): 42% of
#: the way to the wing's tip, as measured.
GREATER = 0.42


def covert_rows(rng: np.random.Generator) -> list[dict]:
    """The rows of coverts, from the wrist back: each row's tip position
    (units, in the row coordinate), its feathers (where across the wing
    each begins, so each one's width), and how deep their rounded tips are.
    The feathers grow down the wing, the greater coverts largest."""
    _, _, _, length = wing_frame()
    rows, d, w = [], 0.32, 0.12
    end = GREATER * length - 0.95
    while d < end:
        rows.append(row_of(rng, d, w, 0.42))
        d += 0.68 * w * rng.uniform(0.9, 1.1)
        w = min(w * 1.17, 0.36)
    rows.append(row_of(rng, GREATER * length, 0.38, 0.55))
    return rows


def row_of(rng: np.random.Generator, d: float, w: float, depth: float) -> dict:
    """One row of coverts: tips about ``d`` back, feathers about ``w`` wide,
    neither all of one width nor evenly set."""
    edges = [-0.6 + rng.uniform(0.0, w)]
    while edges[-1] < 2.0:
        edges.append(edges[-1] + w * rng.uniform(0.8, 1.2))
    jitter = rng.normal(0.0, 0.08 * w, len(edges))
    return {"d": d, "edges": np.array(edges), "jitter": jitter, "depth": depth, "w": w}


def row_coordinate(u, v):
    """The covert rows' coordinate at wing coordinates (u, v)."""
    _, _, _, length = wing_frame()
    return u * length - ROW_SLANT * (v - 0.6) + ROW_BOW * (v - 0.6) ** 2


def row_tip(row: dict, v) -> np.ndarray:
    """Where the tips of a row lie (row coordinate) at each ``v`` across the
    wing: rounded tips, one per feather, a little uneven."""
    e = row["edges"]
    j = np.clip(np.searchsorted(e, v) - 1, 0, len(e) - 2)
    a, b = e[j], e[j + 1]
    half = 0.58 * (b - a)
    theta = np.clip((v - 0.5 * (a + b)) / half, -1.0, 1.0)
    return row["d"] + row["jitter"][j] - row["depth"] * (b - a) * (1 - np.sqrt(1 - theta**2))


def covert_relief(u, v, rows):
    """Over the coverts: which row each place lies in (-1 past them), how
    far from the row above's tips to its own (0..1), and where across its
    own feather (-1..1)."""
    d = row_coordinate(u, v)
    which = np.full(u.shape, -1)
    q = np.zeros(u.shape)
    prev = np.full(u.shape, -1.0)
    for k, row in enumerate(rows):
        tip = row_tip(row, v)
        here = (which < 0) & (d <= tip)
        which = np.where(here, k, which)
        q = np.where(here, np.clip((d - prev) / np.maximum(tip - prev, 1e-6), 0, 1), q)
        prev = tip
    return which, q


#: The flight feathers seen below the greater coverts, from the top (the
#: tertials, over the back) down to the primaries at the lower edge: where
#: each one's free lower edge lies across the wing (0 at its upper edge, 1
#: at its lower) and how far along the wing it reaches.
FLIGHT = [(0.16, 0.6), (0.3, 0.63), (0.42, 0.66), (0.53, 0.69), (0.63, 0.72),
          (0.72, 0.76), (0.8, 0.86), (0.87, 0.93), (0.935, 0.97), (1.0, 1.0)]  # fmt: skip


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
        q = np.where(here, np.clip((f - max(prev, top)) / max(edge - max(prev, top), 1e-6), 0, 1), q)
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
    row, q = covert_relief(u, v, rows)
    blade, qf = flight_relief(u, v)
    covert = wing & (row >= 0)
    flight = wing & (row < 0)
    lift = np.where(covert, 2.2 + 0.5 * np.minimum(row, 8), 2.6)
    height = np.where(covert, q**0.6, np.clip(qf, 0, 1) ** 0.8) * lift
    emboss(model, box_blur(height.astype(np.float32), 1), wing)
    return {"u": u, "v": v, "rows": rows, "row": np.where(wing, row, -2), "q": q,
            "blade": np.where(flight, blade, -2), "qf": qf}


# ----------------------------------------------------------------------
# local colour

#: The local colours (sRGB 0..1).
COLOURS = {
    "green": (0.36, 0.56, 0.15), "lime": (0.6, 0.71, 0.17), "olive": (0.42, 0.5, 0.16),
    "teal": (0.16, 0.5, 0.48), "blue": (0.16, 0.38, 0.66), "deep_blue": (0.13, 0.25, 0.52),
    "sky": (0.55, 0.76, 0.86), "rust": (0.76, 0.3, 0.14), "red": (0.8, 0.1, 0.12),
    "skin": (0.94, 0.8, 0.8), "horn": (0.24, 0.25, 0.28), "horn_tip": (0.36, 0.36, 0.38),
    "jaw": (0.17, 0.17, 0.2), "foot": (0.4, 0.4, 0.42), "claw": (0.13, 0.13, 0.15),
    "bark": (0.56, 0.5, 0.43), "eye": (0.9, 0.86, 0.66),
}  # fmt: skip


def paint(where, colour, amount=1.0):
    """A layer of local colour: where (0..1, (H, W)) and what."""
    return np.asarray(where, dtype=np.float64) * amount, np.array(COLOURS[colour])


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
    return out * model.part("plumage") * (model.headness > 0.15)


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
    return inner * model.part("plumage") * (model.headness > 0.5) * 1.0


def wing_colours(model: Model, wing: dict):
    """The wing's colours: green coverts, yellower toward the shoulder and
    on the greater coverts' tips; the tertials and secondaries green going
    to blue down the wing; the primaries' blue along the lower edge; blue
    toward the tips."""
    seen = model.part("wing")
    u, row, blade, qf = wing["u"], wing["row"], wing["blade"], wing["qf"]
    rows = len(wing["rows"])
    covert = seen & (row >= 0)
    flight = seen & (row < 0) & (blade >= 0)
    lime = covert * (0.45 + 0.4 * smoothstep(rows - 2.5, rows - 1, row) * wing["q"])
    teal = flight * smoothstep(1.5, 3.5, blade) * smoothstep(0.4, 0.62, u)
    blue = flight * np.clip(smoothstep(4.5, 6.5, blade) + smoothstep(0.62, 0.8, u)
                            + 0.4 * smoothstep(0.6, 1.0, qf) * (blade >= 3), 0, 1)
    deep = seen * smoothstep(0.85, 1.0, u)
    return [paint(seen, "green"), paint(lime, "lime"), paint(teal, "teal", 0.7),
            paint(blue, "blue"), paint(deep, "deep_blue", 0.5)]


def local_colour(model: Model, wing: dict) -> np.ndarray:
    """The local colour of everything seen (H, W, 3), before the light."""
    h, w = model.seen.z.shape
    out = np.ones((h, w, 3))
    b = model.bird
    plume = model.part("plumage")
    crown = smoothstep(0.2, 0.9, model.headness) * smoothstep(-0.6, 0.3, np.nan_to_num(model.skull[..., 1]))
    rump = smoothstep(-3.2, -3.6, np.nan_to_num(b[..., 0])) * smoothstep(-4.3, -4.7, np.nan_to_num(b[..., 1]))
    layers = [
        paint(plume, "green"),
        paint(plume * crown, "lime", 0.55),
        paint(plume * rump, "sky", 0.8),
        *wing_colours(model, wing),
        paint(face_mask(model), "skin"),
        paint(brow_mask(model), "red"),
        paint(model.part("upper"), "horn"),
        paint(model.part("lower"), "jaw"),
        paint(model.part("toes"), "foot"),
        paint(model.part("claws"), "claw"),
        paint(model.part("branch"), "bark"),
        paint(model.part("eye"), "eye"),
    ]
    layers += tail_colours(model)
    for amount, colour in layers:
        a = np.clip(amount, 0.0, 1.0)[..., None]
        out = out * (1 - a) + colour * a
    return out


def tail_colours(model: Model):
    """The tail feathers: rust down the middle pair, going to blue toward the
    tips and along the outer webs, the outer feathers bluer, the tips pale."""
    out = []
    for k in range(len(BLADES)):
        seen = model.part(f"tail{k}")
        if not seen.any():
            continue
        t, across = blade_coords(model, k)
        blue = np.clip(smoothstep(0.55, 0.85, t) + 0.6 * smoothstep(0.45, 0.95, np.abs(across))
                       + 0.25 * (k >= 4), 0.0, 1.0)
        out += [paint(seen, "rust"), paint(seen * blue, "blue"),
                paint(seen * smoothstep(0.88, 0.99, t), "sky", 0.7)]
    return out


def blade_coords(model: Model, k: int):
    """How far along tail feather ``k`` (0..1) and across it (-1..1) each
    pixel lies."""
    origin, dk, side, _ = blade_basis(k)
    _, frac, _, half = BLADES[k]
    rel = np.nan_to_num(model.bird) - origin
    return (rel @ dk) / (TAIL_LENGTH * frac), (rel @ side) / half


def lit(colour: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Local colour under the light (in linear light, back to sRGB)."""
    return np.clip(np.clip(colour, 0, 1) ** 2.2 * light[..., None], 0.0, 1.0) ** (1 / 2.2)


def grey(colour: np.ndarray) -> np.ndarray:
    """How light a colour is, 0..1."""
    return colour @ np.array([0.3, 0.59, 0.11])


# ----------------------------------------------------------------------
# the hand: lines on the sheet


class Ink:
    """Marks laid on the sheet: how dark (0..1) each pixel is made."""

    def __init__(self, shape):
        self.dark = np.zeros(shape)

    def line(self, pts, width, dark) -> None:
        """A line through ``pts`` (N, 2) on the sheet, ``width`` and ``dark``
        (each a number or one per point) along it, its edges soft."""
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 2:
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 0.3:
            return
        n = max(int(s[-1] / 0.45) + 1, 2)
        t = np.linspace(0.0, s[-1], n)
        x, y = np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])
        w = np.interp(t, s, np.broadcast_to(np.asarray(width, float), len(pts)))
        d = np.interp(t, s, np.broadcast_to(np.asarray(dark, float), len(pts)))
        for k in range(0, n, 40):
            self.dab(x[k : k + 41], y[k : k + 41], w[k : k + 41], d[k : k + 41])

    def dab(self, x, y, w, d) -> None:
        """Part of a line: discs along it, the darkest of them at each pixel."""
        if not (d > 0.003).any():
            return
        h, wd = self.dark.shape
        r = float(w.max()) / 2 + 1.5
        x0, x1 = int(max(math.floor(x.min() - r), 0)), int(min(math.ceil(x.max() + r) + 1, wd))
        y0, y1 = int(max(math.floor(y.min() - r), 0)), int(min(math.ceil(y.max() + r) + 1, h))
        if x1 <= x0 or y1 <= y0:
            return
        gy, gx = np.mgrid[y0:y1, x0:x1] + 0.5
        dist = np.hypot(gx[..., None] - x, gy[..., None] - y) - w / 2
        cover = (np.clip(0.5 - dist, 0.0, 1.0) * d).max(axis=-1)
        region = self.dark[y0:y1, x0:x1]
        region[:] = 1.0 - (1.0 - region) * (1.0 - cover)


def visible(model: Model, pts: np.ndarray, tol: float = 3.0) -> np.ndarray:
    """Which of the points (N, 3) on the sheet are not hidden behind a
    nearer surface."""
    h, w = model.seen.z.shape
    ix = np.clip(pts[:, 0].astype(int), 0, w - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, h - 1)
    z = model.seen.z[iy, ix]
    on = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    return on & (pts[:, 2] >= z - tol)


def draw3d(ink: Ink, model: Model, pts3, width, dark, tol=3.0) -> None:
    """A line through points (N, 3) on the sheet, lifted where hidden."""
    pts3 = np.asarray(pts3, dtype=np.float64)
    seen = visible(model, pts3, tol)
    d = np.broadcast_to(np.asarray(dark, float), len(pts3)) * seen
    ink.line(pts3[:, :2], width, d)


def contours(model: Model, ink: Ink, outline: np.ndarray) -> None:
    """The edges where one surface passes in front of another or of the
    paper: firm on the side away from the light and where the form turns
    under, lost where the light takes the edge."""
    z = np.where(np.isfinite(model.seen.z), model.seen.z, -1e4)
    jump = np.zeros(z.shape)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        nb = np.roll(np.roll(z, dy, 0), dx, 1)
        jump = np.maximum(jump, np.clip((z - nb - 6.0) / 10.0, 0.0, 1.0))
    jump *= np.isfinite(model.seen.z)
    n = model.seen.normal
    away = np.clip(0.55 - (n @ SUN), 0.0, 1.0) / 0.55
    weight = jump * (0.35 + 0.65 * away) * outline
    soft = box_blur(weight.astype(np.float32), 1)
    ink.dark = 1.0 - (1.0 - ink.dark) * (1.0 - np.clip(1.4 * soft, 0.0, 0.85))


def finish(model: Model, ink: Ink, colour: np.ndarray, light: np.ndarray, spec: np.ndarray):
    """The study in grey and in colour, as images (H, W, 3) 0..1."""
    seen = np.isfinite(model.seen.z)
    value = grey(colour)
    tone = np.clip(value ** 2.2 * light, 0.0, 1.0) ** (1 / 2.2) + spec
    tone = np.where(seen, np.clip(tone, 0.0, 1.0), 1.0)
    paper = np.array([0.985, 0.98, 0.965])
    grey_img = (tone * (1.0 - ink.dark))[..., None] * paper
    col = np.where(seen[..., None], np.clip(lit(colour, light) + spec[..., None], 0, 1), paper)
    col = col * (1.0 - ink.dark)[..., None]
    return grey_img, col


def save(path: Path, img: np.ndarray) -> None:
    """Write an image (H, W, 3) 0..1."""
    write_png(str(path), np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))


def enlarge(img: np.ndarray, box, factor: int) -> np.ndarray:
    """A part of an image (x0, y0, x1, y1), enlarged ``factor`` times."""
    x0, y0, x1, y1 = box
    return np.kron(img[y0:y1, x0:x1], np.ones((factor, factor, 1)))


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    model = build_model()
    rng = np.random.default_rng(SEED)
    wing = feather_wing(model, rng)
    light = illumination(model)
    colour = local_colour(model, wing)
    hard = model.part("upper", "lower", "claws", "eye")
    spec = 0.35 * specular(model, 18.0) * hard
    ink = Ink(model.seen.z.shape)
    contours(model, ink, np.ones(model.seen.z.shape))
    grey_img, col = finish(model, ink, colour, light, spec)
    save(out / "study_grey.png", grey_img)
    save(out / "study_colour.png", col)
    x, y = int(EYE_AT[0]), int(EYE_AT[1])
    box = (max(x - 200, 0), max(y - 130, 0), x + 220, y + 250)
    save(out / "head.png", enlarge(grey_img, box, 2))
    save(out / "head_colour.png", enlarge(col, box, 2))


if __name__ == "__main__":
    main()
