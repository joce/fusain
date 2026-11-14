"""A great green macaw, in coloured pencil on white paper.

Run from the repository root: ``python examples/macaw.py``. Writes
``examples/output/macaw.png``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from murex import CatmullRom, noise, write_png
from murex.filters import bilinear, box_blur, max_filter, smoothstep

from fusain import Medium, Paper, Sheet, pencils, tools

WIDTH, HEIGHT = 1100, 1460
SEED = 7
#: Toward the light: from the upper left, in front of the sheet.
LIGHT = np.array([-0.62, -0.62, 0.48]) / math.sqrt(0.62**2 + 0.62**2 + 0.48**2)

#: The bird is laid out in its own coordinates, then posed and set on the
#: sheet: its head is cocked by HEAD_TILT degrees about the top of the neck
#: (the neck bending with it), the whole bird leans forward by TURN degrees
#: about its feet (ANCHOR), which land at PLACE, SCALE times larger.
NECK, HEAD_TILT = (520.0, 430.0), -10.0
ANCHOR, PLACE, SCALE, TURN = (600.0, 975.0), (708.0, 1175.0), 1.36, -8.0

#: The outlines of the bird, each a closed curve through these points.
HEAD_BODY = [
    (352, 236), (370, 203), (402, 181), (448, 170), (497, 176), (537, 198),
    (564, 233), (590, 292), (604, 352), (622, 425), (660, 535), (695, 660),
    (722, 780), (742, 900), (752, 985), (742, 1030), (700, 1042), (640, 1018),
    (585, 978), (540, 935), (512, 895), (482, 850), (446, 778), (416, 690),
    (398, 600), (394, 525), (388, 466), (380, 430), (370, 380), (360, 300),
]  # fmt: skip
HEAD = [
    (352, 236), (370, 203), (402, 181), (448, 170), (497, 176), (537, 198),
    (564, 233), (586, 290), (592, 350), (565, 410), (505, 452), (440, 470),
    (398, 456), (380, 430), (370, 380), (360, 300),
]  # fmt: skip
WING = [
    (478, 520), (500, 465), (545, 425), (600, 415), (640, 450), (670, 530),
    (700, 640), (722, 760), (740, 880), (758, 1000), (778, 1110), (800, 1235),
    (784, 1226), (752, 1110), (712, 1000), (665, 900), (612, 800), (560, 700),
    (515, 620), (488, 570),
]  # fmt: skip
FACE = [
    (362, 250), (395, 234), (436, 240), (472, 266), (489, 308), (481, 352),
    (458, 386), (422, 402), (392, 398), (382, 368), (378, 320), (369, 280),
]  # fmt: skip
BROW = [
    (344, 242), (346, 211), (362, 188), (392, 176), (424, 178), (440, 192),
    (430, 214), (408, 232), (386, 244), (362, 250),
]  # fmt: skip
#: The upper bill: its ridge from the forehead round to the hook's point,
#: then back along its cutting edge (with the notch) to the gape.
CULMEN = [
    (360, 230), (318, 233), (283, 252), (258, 290), (247, 336), (249, 382),
    (258, 420), (271, 452), (285, 474), (296, 486),
]  # fmt: skip
TOMIUM = [
    (296, 486), (299, 476), (301, 456), (304, 432), (309, 414), (317, 409),
    (322, 398), (340, 384), (364, 374), (388, 368),
]  # fmt: skip
UPPER_BILL = CULMEN + TOMIUM[1:] + [(386, 320), (375, 282), (367, 250)]
LOWER_BILL = [
    (390, 358), (396, 398), (384, 428), (358, 444), (332, 441), (314, 428),
    (309, 410), (322, 390), (346, 372),
]  # fmt: skip
#: The rounded lower edge of the lower bill.
GONYS = [(310, 412), (316, 430), (334, 442), (358, 444), (384, 428), (396, 398)]
EYE = (426.0, 290.0)
IRIS, PUPIL = 14.5, 6.5
#: The folded wing's axis, from the shoulder to the tip.
WING_AXIS = ((560.0, 430.0), (797.0, 1232.0))
#: The feathered thigh, coming down over the tops of the feet.
THIGH = [
    (548, 920), (598, 908), (648, 922), (676, 950), (668, 978), (640, 986),
    (606, 984), (574, 978), (552, 958),
]  # fmt: skip
#: The branch the bird grips: its middle line and half its thickness.
BRANCH = [(-100, 960), (200, 975), (420, 984), (600, 992), (800, 999), (1000, 1007), (1300, 1020)]
BRANCH_HALF = 22.0
#: The near foot's toes (two forward over the branch, then the tips of the
#: two back ones under it), and the far foot's, each from its root to its
#: claw, and how thick each is at its root.
TOES = [
    ([(652, 972), (646, 984), (646, 998), (652, 1010)], 8.0),
    ([(668, 972), (674, 986), (683, 998), (692, 1006)], 8.0),
    ([(592, 968), (578, 978), (566, 994), (562, 1010), (566, 1020)], 10.0),
    ([(608, 970), (604, 988), (606, 1004), (614, 1016), (624, 1020)], 10.0),
    ([(626, 1012), (640, 1018), (652, 1020)], 8.0),
]
#: The long tail feathers, from the outermost (underneath) to the middle
#: pair (on top): sideways offset of the root, direction (degrees), length
#: and half-width.
BLADES = [
    (40.0, 70.0, 420.0, 20.0),
    (-38.0, 90.0, 450.0, 20.0),
    (25.0, 74.0, 600.0, 21.0),
    (-24.0, 86.0, 630.0, 21.0),
    (9.0, 77.0, 800.0, 22.0),
    (-9.0, 82.0, 830.0, 22.0),
]
TAIL_ROOT = (738.0, 1005.0)


# ----------------------------------------------------------------------
# geometry


def headness(y):
    """How much a point of the bird turns with the head, from its height."""
    return smoothstep(520.0, 390.0, y)


def bend(points) -> np.ndarray:
    """Points of the bird, (..., 2), with the head cocked on the neck."""
    p = np.asarray(points, dtype=np.float64)
    a = math.radians(HEAD_TILT) * headness(p[..., 1])
    dx, dy = p[..., 0] - NECK[0], p[..., 1] - NECK[1]
    c, s = np.cos(a), np.sin(a)
    return np.stack([NECK[0] + dx * c - dy * s, NECK[1] + dx * s + dy * c], -1)


def to_sheet(points) -> np.ndarray:
    """Points of the bird, (..., 2), where they land on the sheet."""
    p = bend(points) - np.array(ANCHOR)
    c, s = math.cos(math.radians(TURN)), math.sin(math.radians(TURN))
    x = (p[..., 0] * c - p[..., 1] * s) * SCALE + PLACE[0]
    y = (p[..., 0] * s + p[..., 1] * c) * SCALE + PLACE[1]
    return np.stack([x, y], axis=-1)


@cache
def grid() -> tuple[np.ndarray, np.ndarray]:
    """Where on the bird each pixel of the sheet lies: x and y, (H, W) each."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float64)
    c, s = math.cos(math.radians(TURN)), math.sin(math.radians(TURN))
    dx, dy = (xx - PLACE[0]) / SCALE, (yy - PLACE[1]) / SCALE
    px, py = dx * c + dy * s + ANCHOR[0], -dx * s + dy * c + ANCHOR[1]
    bx, by = px, py
    for _ in range(4):  # (undo the cock of the head, which depends on where)
        a = -math.radians(HEAD_TILT) * headness(by)
        ca, sa = np.cos(a), np.sin(a)
        ex, ey = px - NECK[0], py - NECK[1]
        bx, by = NECK[0] + ex * ca - ey * sa, NECK[1] + ex * sa + ey * ca
    return bx, by


@cache
def turning() -> np.ndarray:
    """How far each pixel's directions are turned from the bird's own, (H, W)
    in radians: the lean of the bird and the cock of its head."""
    _, yy = grid()
    return math.radians(TURN) + math.radians(HEAD_TILT) * headness(yy)


def pixel(x: float, y: float) -> tuple[int, int]:
    """The pixel (column, row) a point of the bird lands on, on the sheet."""
    px, py = to_sheet([x, y])
    return min(max(int(px), 0), WIDTH - 1), min(max(int(py), 0), HEIGHT - 1)


def outline(points, spacing: float = 2.0, closed: bool = True) -> np.ndarray:
    """A smooth curve through ``points``, sampled every ``spacing``."""
    curve = CatmullRom(np.asarray(points, dtype=np.float64), closed=closed)
    return curve.sample(spacing)[0]


def fill_polygon(poly: np.ndarray) -> np.ndarray:
    """Raster mask (H, W) of the inside of a polygon on the sheet."""
    mask = np.zeros((HEIGHT, WIDTH), bool)
    x0, y0 = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
    x1, y1 = min(x1, WIDTH), min(y1, HEIGHT)
    if x1 <= x0 or y1 <= y0:
        return mask
    ys = np.arange(y0, y1) + 0.5
    xs = np.arange(x0, x1) + 0.5
    inside = np.zeros((y1 - y0, x1 - x0), bool)
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        rows = (ys >= min(ay, by)) & (ys < max(ay, by))
        if ay == by or not rows.any():
            continue
        cx = ax + (ys[rows] - ay) * (bx - ax) / (by - ay)
        inside[rows] ^= xs[None, :] < cx[:, None]
    mask[y0:y1, x0:x1] = inside
    return mask


def fill(points) -> np.ndarray:
    """Raster mask (H, W) of the inside of the closed curve through
    ``points`` of the bird."""
    return fill_polygon(to_sheet(outline(points)))


def soft(mask: np.ndarray, r: int) -> np.ndarray:
    """``mask`` as floats, blurred by about ``r`` pixels."""
    out = mask.astype(np.float32)
    for _ in range(2):
        out = box_blur(out, r)
    return out


def grow(mask: np.ndarray, r: int) -> np.ndarray:
    """``mask`` grown by ``r`` pixels."""
    return max_filter(mask.astype(np.float32), r) > 0.5


def shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``mask`` moved by (dx, dy) pixels, empty where it came from."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = mask[
        max(-dy, 0) : h - max(dy, 0), max(-dx, 0) : w - max(dx, 0)
    ]
    return out


def lambert(height: np.ndarray) -> np.ndarray:
    """How squarely a surface of this height faces the light, 0..1."""
    gy, gx = np.gradient(height)
    n = np.stack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=0)
    return np.clip(np.tensordot(LIGHT, n, axes=1), 0.0, 1.0)


def dome(mask: np.ndarray, r: int, depth: float, power: float = 0.6) -> np.ndarray:
    """A rounded form rising from the edge of ``mask``, ``depth`` high."""
    return np.clip(soft(mask, r), 0.0, 1.0) ** power * depth


def stretch(tone: np.ndarray, where: np.ndarray) -> np.ndarray:
    """``tone`` spread so that over ``where`` it runs from 0 to 1."""
    a, b = np.percentile(tone[where], [2, 98])
    return np.clip((tone - a) / max(b - a, 1e-6), 0.0, 1.0)


def band(x, lo: float, hi: float):
    """0 below ``lo``, 1 above ``hi``, smooth between (or the other way round
    if ``lo`` > ``hi``)."""
    return smoothstep(lo, hi, x)


def trapezoid(s, edges) -> np.ndarray:
    """1 between ``b`` and ``c`` of ``edges`` (a, b, c, d), 0 outside ``a``
    and ``d``, smooth between."""
    a, b, c, d = edges
    return band(s, a, b) * (1.0 - band(s, c, d))


def flow(anchors) -> np.ndarray:
    """A field of directions (H, W) on the sheet, in radians: the directions
    at the anchors ``(x, y, degrees)`` of the bird, blended between them."""
    xx, yy = (g[::4, ::4] for g in grid())
    cx = np.zeros_like(xx)
    cy = np.zeros_like(xx)
    for x, y, deg in anchors:
        w = 1.0 / ((xx - x) ** 2 + (yy - y) ** 2 + 400.0) ** 1.5
        cx += w * math.cos(math.radians(deg))
        cy += w * math.sin(math.radians(deg))
    small = np.arctan2(cy, cx)
    big = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)[:HEIGHT, :WIDTH]
    return big + turning()


def distance_to(points, near: np.ndarray) -> np.ndarray:
    """Distance (in the bird's units) from each pixel of ``near`` to the
    curve through ``points``; elsewhere a lot."""
    out = np.full((HEIGHT, WIDTH), 1e6)
    rows, cols = np.nonzero(near)
    xx, yy = (g[rows, cols] for g in grid())
    line = outline(points, 1.5, closed=False)
    best = np.full(rows.size, 1e6)
    for x, y in line:
        np.minimum(best, np.hypot(xx - x, yy - y), out=best)
    out[rows, cols] = best
    return out


def frame(axis, half=None) -> tuple[np.ndarray, np.ndarray]:
    """How far along ``axis`` (two points of the bird) each pixel lies, 0 at
    its start and 1 at its end, and how far across it, in the bird's units
    (positive to the right of the axis, looking along it) or in its
    half-widths ``half(t)`` if given."""
    xx, yy = grid()
    a, b = np.array(axis[0]), np.array(axis[1])
    length = float(np.linalg.norm(b - a))
    ax, ay = (b - a) / length
    t = ((xx - a[0]) * ax + (yy - a[1]) * ay) / length
    across = (xx - a[0]) * -ay + (yy - a[1]) * ax
    if half is not None:
        across = across / half(np.clip(t, 0.0, 1.0))
    return t, across


def on_axis(axis, t: float, across: float) -> tuple[float, float]:
    """The point of the bird ``t`` along ``axis`` and ``across`` it."""
    a, b = np.array(axis[0]), np.array(axis[1])
    length = float(np.linalg.norm(b - a))
    ax, ay = (b - a) / length
    x = a[0] + t * length * ax - across * ay
    y = a[1] + t * length * ay + across * ax
    return float(x), float(y)


def circle(centre, radius: float, n: int = 36, start: float = 0.0, turn=2 * math.pi):
    """Points round a circle (or an arc of it)."""
    a = np.linspace(start, start + turn, n)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)], 1)


def blade(offset: float, degrees: float, length: float, half: float) -> np.ndarray:
    """The outline of a long tail feather, (N, 2) points of the bird: a
    narrow blade from the tail's root, rounded and a little pointed."""
    d = np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])
    n = np.array([-d[1], d[0]])
    s = np.linspace(0.0, 1.0, 40)
    width = half * np.where(s < 0.8, 0.85 + 0.15 * s / 0.8, np.sqrt(np.clip((1.0 - s) / 0.2, 0.0, 1.0)))
    root = np.array(TAIL_ROOT) + n * offset
    middle = root + np.outer(s * length, d)
    return np.concatenate([middle + n * width[:, None], (middle - n * width[:, None])[::-1]])


# ----------------------------------------------------------------------
# the hand


@dataclass
class Hand:
    """The sheet and the hand that draws on it."""

    sheet: Sheet
    rng: np.random.Generator
    box: dict[str, Medium] = field(default_factory=dict)
    #: How far each place is carried (H, W), 0..1: the fewer strokes there.
    finish: np.ndarray | None = None

    def pencil(self, colour: str) -> Medium:
        """The pencil of that colour, from the box."""
        if colour not in self.box:
            self.box[colour] = pencils.coloured(colour)
        return self.box[colour]

    def seeds(self, weight: np.ndarray, count: int) -> np.ndarray:
        """``count`` points of the sheet, more of them where ``weight`` is."""
        flat = np.clip(weight, 0.0, None).ravel().astype(np.float64)
        total = flat.sum()
        if total <= 0 or count <= 0:
            return np.zeros((0, 2))
        idx = self.rng.choice(flat.size, size=count, p=flat / total)
        y, x = np.divmod(idx, WIDTH)
        return np.stack([x, y], axis=1) + self.rng.uniform(0.0, 1.0, (count, 2))

    def trace(self, start, angle, steps: int, step: float, bend: float, spread):
        """Paths (N, steps + 1, 2) from ``start`` along the field ``angle``,
        each with its own slight curve and waver."""
        n = len(start)
        path = np.empty((n, steps + 1, 2))
        path[:, 0] = start
        turn = self.rng.normal(0.0, bend, n)
        offset = self.rng.normal(0.0, spread, n)
        for k in range(steps):
            ix = np.clip(path[:, k, 0].astype(int), 0, WIDTH - 1)
            iy = np.clip(path[:, k, 1].astype(int), 0, HEIGHT - 1)
            a = angle[iy, ix] + offset + turn * k
            path[:, k + 1, 0] = path[:, k, 0] + step * np.cos(a)
            path[:, k + 1, 1] = path[:, k, 1] + step * np.sin(a)
        return path

    def layer(self, colour: str, weight, angle, firm: float, **how) -> None:
        """One layer of strokes of one pencil: where ``weight`` > 0, more and
        firmer where it is higher, each running along the field ``angle``
        from the root of a feather toward its tip, kept inside ``clip``.
        ``cover`` is how many times over they would cover the area at full
        weight; ``tilt`` lays them with the side of the lead."""
        length, width = how.get("length", 30.0), how.get("width", 4.0)
        step, tilt = how.get("step", 4.0), how.get("tilt", 0.0)
        clip = how.get("clip", weight > 0.02)
        dense = weight if self.finish is None else weight * self.finish
        broad = width * (1.0 + 2.0 * tilt)
        count = int(how.get("cover", 1.5) * float(dense.sum()) / (broad * length))
        seed = self.seeds(dense, count)
        if not len(seed):
            return
        sx = np.clip(seed[:, 0].astype(int), 0, WIDTH - 1)
        sy = np.clip(seed[:, 1].astype(int), 0, HEIGHT - 1)
        at, own = angle[sy, sx], np.clip(weight[sy, sx], 0.0, 1.0)
        # (the stroke starts a little before the seed, so that it runs
        # across the place where one part meets the next)
        back = self.rng.uniform(0.0, 0.45 * length, len(seed))
        start = seed - back[:, None] * np.stack([np.cos(at), np.sin(at)], 1)
        steps = max(int(length * 2.0 / step), 2)
        bend, spread = how.get("bend", 0.02), how.get("spread", 0.12)
        paths = self.trace(start, angle, steps, step, bend, spread)
        lengths = self.rng.lognormal(math.log(length), 0.35, len(seed))
        medium = self.pencil(colour)
        for path, ln, w0 in zip(paths, lengths, own, strict=True):
            path = inside_run(path[: min(max(int(ln / step), 1), steps) + 1], clip)
            if len(path) < 2:
                continue
            ix = np.clip(path[:, 0].astype(int), 0, WIDTH - 1)
            iy = np.clip(path[:, 1].astype(int), 0, HEIGHT - 1)
            local = np.clip(weight[iy, ix], 0.0, 1.0)
            u = np.linspace(0.0, 1.0, len(path))
            bear = firm * np.sqrt(0.5 * w0 + 0.5 * local) * flick(u)
            bear *= self.rng.uniform(0.8, 1.15)
            self.sheet.stroke(path.tolist(), medium, bear.tolist(), width, tilt)

    def tone(self, colour: str, weight: np.ndarray, firm: float, angles, width=3.0):
        """Even hatching of one pencil over all of ``weight`` > 0, as firm as
        it asks, once at each of ``angles``: a closed, waxy tone."""
        rows, cols = np.nonzero(weight > 0.01)
        if not rows.size:
            return
        box = (cols.min() - 4, rows.min() - 4, cols.max() + 5, rows.max() + 5)
        w = weight.astype(np.float64)

        def pressure(x, y):
            xi = np.clip(np.asarray(x, dtype=np.float64), 0.0, WIDTH - 1.001)
            yi = np.clip(np.asarray(y, dtype=np.float64), 0.0, HEIGHT - 1.001)
            return firm * bilinear(w, yi, xi)

        for angle in angles:
            self.sheet.shade(box, self.pencil(colour), pressure, width, angle=angle)

    def erase(self, points, tool, pressure, width: float) -> None:
        """One pass of an eraser through ``points`` of the bird."""
        self.sheet.erase(to_sheet(points).tolist(), tool, pressure, width)

    def line(self, colour: str, points, pressure, width=2.5, wobble=0.5, within=None):
        """One stroke through ``points`` of the bird, as a hand follows them;
        lifted wherever it leaves ``within``, if given."""
        pts = to_sheet(points)
        if len(pts) < 2:
            return
        pts = pts + self.rng.normal(0.0, wobble, pts.shape)
        bear = np.interp(
            np.linspace(0.0, 1.0, len(pts)),
            np.linspace(0.0, 1.0, np.size(pressure)),
            np.ravel(pressure),
        )
        if within is not None:
            ix = np.clip(pts[:, 0].astype(int), 0, WIDTH - 1)
            iy = np.clip(pts[:, 1].astype(int), 0, HEIGHT - 1)
            bear = np.where(within[iy, ix], bear, 0.0)
        self.sheet.stroke(pts.tolist(), self.pencil(colour), bear.tolist(), width)


def inside_run(path: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """The first stretch of ``path`` that lies inside ``clip``."""
    ix = np.clip(path[:, 0].astype(int), 0, WIDTH - 1)
    iy = np.clip(path[:, 1].astype(int), 0, HEIGHT - 1)
    inside = clip[iy, ix]
    if not inside.any():
        return path[:0]
    first = int(np.argmax(inside))
    after = np.nonzero(~inside[first:])[0]
    last = first + int(after[0]) if after.size else len(path)
    return path[first:last]


def flick(u: np.ndarray) -> np.ndarray:
    """How the pressure runs along a stroke (``u`` 0 at its start, 1 at its
    end): it lands firmly and lifts away, as a feather is stroked in."""
    return np.minimum(u * 5.0 + 0.35, 1.0) * (1.0 - 0.8 * u**1.6)


# ----------------------------------------------------------------------
# the bird


@dataclass
class Bird:
    """Where each part of the bird is seen (``part``), where its strokes may
    run (``clip``), how strongly it is drawn at its edge (``edge``: lost
    where the light takes the silhouette), how dark it falls (``shade``:
    0 in the full light to 1 in the deepest shadow) and where light thrown
    back from below warms the edge of a shadow (``glow``)."""

    part: dict[str, np.ndarray]
    clip: dict[str, np.ndarray]
    edge: np.ndarray
    shade: dict[str, np.ndarray]
    glow: np.ndarray
    mottle: np.ndarray


def toe_outline(toe, root: float) -> np.ndarray:
    """The outline of a toe round its middle line, (N, 2) points of the bird,
    thick at its root and narrowing to the claw."""
    pts = outline(toe, 1.5, closed=False)
    d = np.gradient(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    half = np.linspace(root, root * 0.55, len(pts))[:, None]
    return np.concatenate([pts - n * half, (pts + n * half)[::-1]])


def parts() -> dict[str, np.ndarray]:
    """The parts of the bird, each where it is seen in front of the others."""
    xx, yy = grid()
    line = outline(BRANCH, 2.0, closed=False)
    across = (yy - np.interp(xx, line[:, 0], line[:, 1])) / BRANCH_HALF
    blades = [fill_polygon(to_sheet(blade(*b))) for b in BLADES]
    p = {
        "whole": fill(HEAD_BODY),
        "skull": fill(HEAD),
        "wing_all": fill(WING),
        "face": fill(FACE),
        "brow": fill(BROW),
        "upper": fill(UPPER_BILL),
        "lower": fill(LOWER_BILL),
        "thigh": fill(THIGH),
        "branch": np.abs(across) < 1.0,
        "toes": np.zeros((HEIGHT, WIDTH), bool),
    }
    for toe, root in TOES:
        p["toes"] |= fill_polygon(to_sheet(toe_outline(toe, root)))
    p["tail_all"] = np.logical_or.reduce(blades)
    bills = p["upper"] | p["lower"]
    p["lower"] &= ~p["upper"]
    p["brow"] &= ~bills
    p["face"] &= ~bills & ~p["brow"]
    front = bills | p["face"] | p["brow"]
    p["thigh"] &= ~p["wing_all"]
    p["toes"] &= ~p["thigh"]
    p["branch"] &= ~p["toes"] & ~p["thigh"]
    p["head"] = p["skull"] & p["whole"] & ~front
    p["wing"] = p["wing_all"] & ~p["branch"]
    hidden = p["head"] | front | p["wing_all"] | p["branch"] | p["thigh"] | p["toes"]
    p["body"] = p["whole"] & ~hidden
    behind = p["whole"] | p["wing_all"] | p["branch"] | p["thigh"] | p["toes"]
    over = np.zeros((HEIGHT, WIDTH), bool)
    for k in range(len(blades) - 1, -1, -1):
        p[f"blade{k}"] = blades[k] & ~over & ~behind
        over |= blades[k]
    p["tail"] = p["tail_all"] & ~behind
    return p


def body_shading(p: dict[str, np.ndarray]) -> np.ndarray:
    """How dark the head and body fall: each a rounded form under the light,
    lit and shadow sides parted at a clear turn; darker down the bird; in
    the shadow the head and bill throw on the throat and breast, in the
    slot between the folded wing and the flank, under the belly and on the
    far thigh."""
    xx, yy = grid()
    height = dome(p["whole"], 80, 420.0, 0.5) + dome(p["skull"], 50, 160.0, 0.6)
    form = 1.0 - lambert(height)
    shade = 0.5 * form + 0.4 * band(form, 0.45, 0.75)
    shade += 0.4 * band(yy, 420, 1000) + 0.15 * band(xx, 480, 720)
    head = p["skull"] | p["face"] | p["brow"] | p["upper"] | p["lower"]
    cast = soft(shift(head, int(14 * SCALE), int(34 * SCALE)), 22) * (1.0 - soft(head, 6))
    cast = cast * band(xx, 480.0, 400.0)
    t, _ = frame(WING_AXIS)
    slot = soft(p["wing_all"], 16) * ~p["wing_all"] * band(t, 0.15, 0.35)
    under = band(soft(p["whole"], 14), 0.95, 0.5) * band(yy, 640, 900)
    shade += 0.8 * cast + 0.9 * slot + 0.45 * band(yy, 840, 990) + 0.6 * under
    shade += 0.5 * soft(p["thigh"] | p["toes"], 10) * ~p["thigh"]
    return stretch(shade, p["body"] | p["head"])

def shading(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """How dark each part falls under the light from the upper left."""
    _, yy = grid()
    body = body_shading(p)
    t, _ = frame(WING_AXIS)
    wing = 0.75 * (1.0 - lambert(dome(p["wing_all"], 30, 90.0) + dome(p["whole"], 70, 120.0)))
    wing += 0.35 * band(yy, 450, 1150) + wing_feathering()
    wing = band(stretch(wing, p["wing"]), 0.0, 1.0)
    u, _ = frame(TAIL_AXIS)
    tail = 1.0 - lambert(dome(p["tail_all"], 10, 25.0))
    near = soft(p["whole"] | p["wing_all"] | p["thigh"], 26)
    tail = stretch(tail + 1.4 * near + 0.4 * band(u, 0.4, 0.0), p["tail"])
    thigh = 1.0 - lambert(dome(p["thigh"], 16, 50.0))
    thigh = 0.3 + 0.6 * stretch(thigh + 0.5 * band(grid()[0], 590, 680), p["thigh"])
    return {"body": body, "wing": wing, "tail": tail, "thigh": thigh}


#: The tail's middle line, from its root to past the end of its longest
#: feathers.
TAIL_AXIS = (TAIL_ROOT, (TAIL_ROOT[0] + 300.0, TAIL_ROOT[1] + 760.0))


def build(rng: np.random.Generator) -> Bird:
    """The bird as it is to be drawn."""
    p = parts()
    silhouette = p["whole"] | p["wing_all"] | p["tail_all"] | p["thigh"]
    seen = silhouette & ~p["branch"] & ~p["toes"]
    shade = shading(p)
    lit_side = 1.0 - band(shade["body"], 0.15, 0.45)
    edge = (soft(seen, 2) * (1.0 - lit_side) + soft(seen, 6) * lit_side) * seen
    inner = soft(silhouette, 7)
    glow = band(inner, 0.45, 0.6) * band(inner, 0.95, 0.8) * band(shade["body"], 0.55, 0.8)
    glow = glow * (p["body"] | p["head"])
    shade["body"] = np.clip(shade["body"] - 0.3 * glow, 0.0, 1.0)
    names = ["body", "head", "wing", "tail", "face", "brow", "upper", "lower", "thigh"]
    clip = {name: grow(p[name], 3) & seen for name in names}
    mottle = noise.fractal_noise_2d(HEIGHT, WIDTH, 40.0, rng, octaves=4)
    mottle = (mottle - mottle.min()) / (np.ptp(mottle) + 1e-9)
    return Bird(p, clip, edge, shade, glow, mottle)


# ----------------------------------------------------------------------
# the wing's feathers

#: The rows of coverts across the wing, from the shoulder down: how far
#: along the wing each row's tips lie, how wide its feathers are, and how
#: long their rounded tips (in widths).
COVERT_ROWS = (
    (0.075, 14.0, 0.8), (0.115, 15.0, 0.8), (0.155, 16.0, 0.8), (0.195, 17.0, 0.8),
    (0.24, 19.0, 0.85), (0.315, 25.0, 1.0), (0.425, 31.0, 1.25),
)  # fmt: skip
#: Where the folded flight feathers' edges lie across the wing (where they
#: leave the greater coverts; they close up toward the tip), front first,
#: and how far along the wing each feather between them reaches.
FLIGHT = (64.0, 46.0, 27.0, 8.0, -11.0, -30.0, -50.0)
FLIGHT_ENDS = (1.0, 0.96, 0.9, 0.82, 0.76, 0.7)
WING_LENGTH = 824.0


def covert_phase(across, size: float, k: int):
    """Where across its own feather each ``across`` of row ``k`` lies, 0..1:
    the feathers of a row neither all of a width nor evenly spaced."""
    warp = 0.24 * size * np.sin(across / (1.7 * size) + 2.1 * k)
    warp = warp + 0.11 * size * np.sin(across / (0.67 * size) + 0.9 * k)
    return ((across + warp + 0.5 * size * (k % 2)) / size) % 1.0


def wing_tips(across, t0: float, size: float, long: float, k: int):
    """Where along the wing row ``k``'s feather tips end, at each ``across``:
    a scallop of rounded tips ``size`` wide and ``long`` widths long."""
    phase = covert_phase(across, size, k)
    depth = long * size / WING_LENGTH
    wave = 0.008 * np.sin(across / 37.0 + 1.7 * k) + 0.005 * np.sin(across / 13.0 + k)
    return t0 + wave + depth * np.sqrt(np.clip(1.0 - (2.0 * phase - 1.0) ** 2, 0.0, 1.0))


def flight_place(t, across):
    """Which flight feather (fractional, 0 at the front edge) is at each
    place, and how far across it (0 at its front edge, 1 at its back)."""
    reach = np.clip(1.0 - 0.85 * (t - 0.44) / 0.56, 0.15, 1.0)
    q = (across - 4.0) / reach
    k = np.interp(-q, [-a for a in FLIGHT], np.arange(len(FLIGHT)))
    return k, k % 1.0


def wing_feathering() -> np.ndarray:
    """How much darker each place on the wing is for the feathers over it:
    each row of coverts dark just under the row above, lighter toward its
    own rounded tips; each flight feather dark beside the edge of the next."""
    t, across = frame(WING_AXIS)
    mod = np.zeros_like(t)
    prev = np.full_like(t, 0.02)
    for k, (t0, size, long) in enumerate(COVERT_ROWS):
        edge = wing_tips(across, t0, size, long, k)
        q = np.clip((t - prev) / np.maximum(edge - prev, 1e-3), 0.0, 1.0)
        inside = (t >= prev) & (t < edge)
        own = 0.05 * np.sin(np.floor((across + 0.5 * size * (k % 2)) / size) * 12.99 + k * 7.2)
        mod += inside * (0.22 * (1.0 - q) ** 3 - 0.08 * q + own)
        prev = edge
    _, phase = flight_place(t, across)
    mod += (t >= prev) * (0.45 * (1.0 - phase) ** 2.5 - 0.2 * band(phase, 0.35, 0.85))
    return mod


# ----------------------------------------------------------------------
# the box of pencils

LEMON, CHARTREUSE, OLIVE, MOSS = "#e2dc6a", "#b4c044", "#7c9032", "#58742c"
LEAF, PINE, DEEP, NIGHT, INDIGO = "#3c6a2e", "#284834", "#1b3a3c", "#16283a", "#1d2148"
GRASS, SPRUCE, UMBRA = "#5c9a34", "#1f4038", "#172a28"
OCHRE, STRAW = "#c8a040", "#eee4a0"
SKY, TURQ, COBALT, ULTRA, NAVY = "#8cc4e2", "#3ab0bf", "#2a6db8", "#2b3c94", "#1d2452"
ORANGE, VERMILION, MADDER, OXBLOOD = "#e8823a", "#d04a2c", "#982a26", "#5a1e1e"
BLUSH, ROSE, MAUVE, COOL = "#f4d6ce", "#e8a6a2", "#b6a0aa", "#a9a8b8"
LINE, WINE = "#241216", "#55202a"
SCARLET, CRIMSON, FLAME = "#e3352c", "#a51b26", "#f2683a"
HORN, SILVER, GREY, SLATE, CHAR, BLACK = (
    "#c4b9a6", "#a9a9ae", "#76767e", "#474a55", "#202026", "#121214",
)  # fmt: skip
UMBER, SEPIA, WARM_GREY = "#7a6450", "#4e3c30", "#a59a8c"

#: Layers of colour, laid in this order, the darks first while the tooth
#: still takes them: the pencil; where on the scale of shade it goes on
#: (rising from a to b, falling from c to d); how firmly; how many times
#: over it covers; the width of the point; how far over the pencil is held
#: (1: the side of the lead, for the broad lights); and how firmly it is
#: also hatched in, closed and waxy (0: left open and grainy).
GREENS = [
    (UMBRA, (0.8, 0.93, 2.0, 2.1), 0.85, 1.4, 3.0, 0.0, 0.7),
    (SPRUCE, (0.66, 0.82, 2.0, 2.1), 0.8, 1.6, 3.5, 0.0, 0.6),
    (DEEP, (0.52, 0.7, 0.95, 1.05), 0.72, 1.6, 4.0, 0.0, 0.5),
    (PINE, (0.4, 0.56, 0.82, 0.94), 0.62, 1.6, 4.0, 0.0, 0.4),
    (LEAF, (0.28, 0.42, 0.68, 0.82), 0.56, 1.6, 4.0, 0.0, 0.3),
    (GRASS, (0.12, 0.25, 0.6, 0.75), 0.52, 1.8, 4.0, 0.0, 0.3),
    (MOSS, (0.17, 0.3, 0.55, 0.7), 0.46, 1.2, 4.0, 0.0, 0.0),
    (OLIVE, (0.08, 0.18, 0.42, 0.56), 0.46, 1.5, 4.0, 0.0, 0.0),
    (CHARTREUSE, (-0.1, 0.03, 0.3, 0.44), 0.44, 1.4, 4.0, 0.0, 0.3),
    (LEMON, (-0.1, 0.0, 0.14, 0.28), 0.34, 1.0, 4.0, 1.0, 0.2),
]
BLUES = [
    (INDIGO, (0.8, 0.93, 2.0, 2.1), 0.85, 1.4, 3.0, 0.0, 0.6),
    (NAVY, (0.62, 0.8, 2.0, 2.1), 0.75, 1.5, 3.5, 0.0, 0.5),
    (ULTRA, (0.42, 0.62, 0.9, 1.0), 0.62, 1.5, 4.0, 0.0, 0.4),
    (COBALT, (0.18, 0.38, 0.7, 0.86), 0.55, 1.6, 4.0, 0.0, 0.0),
    (TURQ, (-0.1, 0.05, 0.4, 0.6), 0.46, 1.3, 4.0, 0.0, 0.0),
]
RUSTS = [
    (OXBLOOD, (0.72, 0.9, 2.0, 2.1), 0.7, 1.3, 3.5, 0.0, 0.4),
    (MADDER, (0.45, 0.66, 0.95, 1.05), 0.6, 1.5, 4.0, 0.0, 0.0),
    (VERMILION, (0.1, 0.3, 0.75, 0.9), 0.55, 1.7, 4.0, 0.0, 0.0),
    (ORANGE, (-0.1, 0.0, 0.3, 0.5), 0.42, 1.2, 4.0, 0.0, 0.0),
]
INKS = [
    (BLACK, (0.8, 0.92, 2.0, 2.1), 0.85, 1.6, 3.0, 0.0, 0.2),
    (CHAR, (0.58, 0.76, 0.95, 1.02), 0.72, 1.8, 3.0, 0.0, 0.0),
    (SLATE, (0.36, 0.54, 0.78, 0.88), 0.62, 1.8, 3.0, 0.0, 0.0),
    (GREY, (0.16, 0.32, 0.54, 0.68), 0.52, 1.6, 3.0, 0.0, 0.0),
    (SILVER, (-0.1, 0.05, 0.28, 0.42), 0.4, 1.4, 3.0, 0.0, 0.0),
]

BODY_FLOW = [
    (420, 470, 82), (420, 600, 78), (470, 780, 62), (560, 930, 45),
    (700, 600, 72), (760, 900, 80), (610, 380, 66), (560, 520, 74),
]  # fmt: skip
HEAD_FLOW = [
    (372, 208, -35), (440, 178, -2), (520, 200, 30), (575, 270, 62),
    (600, 380, 75), (430, 440, 70), (390, 430, 85), (510, 330, 50),
    (500, 255, 22), (545, 420, 72),
]  # fmt: skip
WING_FLOW = [
    (520, 470, 48), (600, 430, 62), (600, 560, 68), (680, 650, 73),
    (700, 900, 75), (775, 1150, 80), (560, 620, 64),
]  # fmt: skip
BROW_FLOW = [(350, 238, -80), (362, 205, -48), (398, 182, -12), (434, 194, 20), (404, 226, -20)]
THIGH_FLOW = [(560, 930, 70), (620, 930, 90), (665, 950, 110)]


def build_up(hand: Hand, layers, where, s, angle, **how) -> None:
    """Lay ``layers`` over ``where``, each where the shade ``s`` calls for it,
    in strokes ``length`` long (and hatched where the layer says)."""
    vary, scale = how.pop("vary", None), how.pop("scale", 1.0)
    angles, more = how.pop("angles", (62.0, 118.0)), how.pop("hatch", 0.0)
    for k, (colour, edges, firm, cover, width, tilt, hatch) in enumerate(layers):
        w = where * trapezoid(s, edges)
        if vary is not None:
            w = w * vary
        if hatch + more > 0.0:
            closed = w if hand.finish is None else w * hand.finish
            angle_k = (angles[k % len(angles)],)
            hand.tone(colour, closed, firm * min(hatch + more, 0.8), angle_k, 4.0)
        wide = max(width * scale, 2.6)
        hand.layer(colour, w, angle, firm, cover=cover, width=wide, tilt=tilt, **how)


# ----------------------------------------------------------------------
# the drawing


def draw_sketch(hand: Hand) -> None:
    """The first lines, light and searching, which the colour mostly covers."""
    shapes = [HEAD_BODY, WING, UPPER_BILL, LOWER_BILL, FACE] + [blade(*b) for b in BLADES]
    for points in shapes:
        curve = outline(points, 6.0)
        start, n = 0, len(curve)
        while start < n - 4:
            run = int(hand.rng.integers(10, 36))
            seg = curve[start : min(start + run, n)]
            if len(seg) > 2 and hand.rng.random() < 0.75:
                hand.line("#b0ad98", seg, [0.1, 0.22, 0.14], 2.0, 0.9)
            start += run - int(hand.rng.integers(0, 4))


def draw_rows(hand: Hand, where: np.ndarray, s: np.ndarray, rows) -> None:
    """Rows of small overlapping feather tips, each a soft curved touch at
    its lower edge: across the band ``rows`` = (axis, first, last, apart,
    size, how light the touch); only where the shade turns."""
    axis, first, last, apart, size, light = rows
    a, b = np.array(axis[0]), np.array(axis[1])
    along = (b - a) / np.linalg.norm(b - a)
    side = np.array([-along[1], along[0]])
    k = np.linspace(0.0, 1.0, 7)
    for d in np.arange(first, last, apart):
        across = -200.0 + hand.rng.uniform(0.0, size)
        while across < 200.0:
            wide = size * hand.rng.uniform(0.8, 1.25)
            bow = 0.3 * wide * np.sin(np.pi * k)
            pts = a + np.outer(d + bow, along) + np.outer(across + k * wide, side)
            x, y = pixel(*pts[3])
            show = trapezoid(float(s[y, x]), (0.3, 0.45, 0.68, 0.82))
            if where[y, x] and hand.rng.random() < 0.7 * show:
                firm = light * hand.rng.uniform(0.7, 1.1)
                hand.line(PINE, pts, [0.0, firm, firm, 0.6 * firm, 0.0], 3.0, 0.4, where)
            across += wide * hand.rng.uniform(0.85, 1.3)


def draw_plumage(hand: Hand, bird: Bird) -> None:
    """The head, neck, breast and belly: the small feathers stated in a few
    rows where the light turns, then the colour, darks first, in strokes
    from the root of each feather toward its tip: short on the head, longer
    on the body; the warm light thrown back into the shadow at its edge."""
    s, vary = bird.shade["body"], 0.8 + 0.4 * bird.mottle
    clip = bird.clip["body"] | bird.clip["head"]
    nape = (bird.part["head"] | bird.part["body"]) & (grid()[0] > 455) & (grid()[1] < 500)
    draw_rows(hand, nape, s, (((520, 290), (610, 480)), 20.0, 200.0, 9.0, 9.0, 0.32))
    draw_rows(hand, bird.part["body"], s, (((440, 470), (640, 1000)), 40.0, 520.0, 18.0, 20.0, 0.36))
    hand.layer(OCHRE, bird.glow, flow(BODY_FLOW), 0.42, length=12, cover=1.4, clip=clip)
    for name, anchors, length, scale in (("body", BODY_FLOW, 20.0, 1.0), ("head", HEAD_FLOW, 9.0, 0.75)):
        where = bird.part[name] * bird.edge
        angle = flow(anchors)
        build_up(hand, GREENS, where, s, angle, vary=vary, length=length, scale=scale, clip=clip, spread=0.2)


def draw_coverts(hand: Hand, bird: Bird) -> None:
    """The rows of coverts across the wing, each feather stated only by a
    dark accent in the shadow its tip throws on the feather below, on the
    side away from the light; some left unstated, more of them in the light
    where the light takes them, all lost in the wing's shadowed lower part."""
    s = bird.shade["wing"]
    wing = bird.part["wing"]
    across = np.arange(-120.0, 120.0, 1.0)
    for k, (t0, size, long) in enumerate(COVERT_ROWS):
        phase = covert_phase(across, size, k)
        ts = wing_tips(across, t0, size, long, k)
        starts = np.nonzero(np.diff(phase) < -0.5)[0] + 1
        for a, b in zip(starts[:-1], starts[1:], strict=True):
            reach = hand.rng.uniform(0.35, 0.7)
            seg = np.arange(a, b)[phase[a:b] < reach]
            if len(seg) < 3:
                continue
            x, y = pixel(*on_axis(WING_AXIS, float(ts[seg[len(seg) // 2]]), float(across[seg[len(seg) // 2]])))
            lit = 1.0 - band(float(s[y, x]), 0.35, 0.85)
            if not wing[y, x] or hand.rng.random() > 0.3 + 0.55 * lit:
                continue
            firm = (0.22 + 0.4 * lit) * hand.rng.uniform(0.7, 1.15) * (0.75 + 0.05 * k)
            bear = firm * np.sin(np.linspace(0.25, 2.9, len(seg)))
            lag = 0.006 + 0.002 * hand.rng.random()
            under = [on_axis(WING_AXIS, float(ts[i]) + lag, float(across[i])) for i in seg]
            hand.line(SPRUCE, under, bear.tolist(), 3.0 + 0.3 * k, 0.4, wing)
            if hand.rng.random() < 0.5:
                tip = [on_axis(WING_AXIS, float(ts[i]) + 0.001, float(across[i])) for i in seg]
                hand.line(UMBRA, tip, (0.8 * bear).tolist(), 1.8, 0.3, wing)

def draw_flight_edges(hand: Hand, bird: Bird) -> None:
    """The long flight feathers lying like closed blades: a crisp dark line
    where each overlaps the next, and the rounded ends of the shorter ones
    against the longer ones below."""
    s = bird.shade["wing"]
    wing = bird.part["wing"]
    for j, a0 in enumerate(FLIGHT[1:]):
        end = FLIGHT_ENDS[j] - 0.01
        ts = np.linspace(0.43, end, 40)
        reach = np.clip(1.0 - 0.85 * (ts - 0.44) / 0.56, 0.15, 1.0)
        pts = [on_axis(WING_AXIS, float(t), a0 * r + 4.0) for t, r in zip(ts, reach, strict=True)]
        x, y = pixel(*pts[20])
        firm = 0.7 - 0.25 * band(float(s[y, x]), 0.6, 0.95)
        bear = firm * (0.55 + 0.45 * np.sin(np.linspace(0.2, 2.9, 40))) * (1.0 - 0.4 * band(ts, 0.8, 1.0))
        hand.line(NIGHT, pts, bear.tolist(), 2.4, 0.25, wing)
        shadow = [(px, py) for px, py in pts]
        hand.line(NAVY, shadow, (bear * 0.5).tolist(), 5.0, 0.6, wing)
    for j in range(3, len(FLIGHT_ENDS)):  # the ends of the secondaries
        t1 = FLIGHT_ENDS[j]
        a_front, a_back = FLIGHT[j], FLIGHT[j + 1]
        reach = 1.0 - 0.85 * (t1 - 0.44) / 0.56
        k = np.linspace(0.0, 1.0, 9)
        ends = [on_axis(WING_AXIS, t1 + 0.02 * math.sin(math.pi * q), (a_front + (a_back - a_front) * q) * reach + 4.0) for q in k]
        hand.line(NIGHT, ends, [0.0, 0.5, 0.6, 0.5, 0.0], 2.4, 0.3, wing)


def draw_wing(hand: Hand, bird: Bird) -> None:
    """The folded wing: its structure first, while the tooth still takes a
    dark line; then the green shoulder and coverts, the flight feathers
    going from green to blue, and the blue primaries at the back."""
    draw_coverts(hand, bird)
    draw_flight_edges(hand, bird)
    t, across = frame(WING_AXIS)
    k, _ = flight_place(t, across)
    s = bird.shade["wing"]
    where = bird.part["wing"] * bird.edge
    clip = bird.clip["wing"]
    angle = flow(WING_FLOW)
    greater = wing_tips(across, *COVERT_ROWS[-1], len(COVERT_ROWS) - 1)
    covert = t < greater
    secondary = band(k, 2.6, 3.4) * band(t, 0.66, 0.5)
    green = where * np.clip(covert + (~covert) * secondary, 0.0, 1.0)
    teal = where * (~covert) * band(k, 2.6, 3.4) * trapezoid(t, (0.46, 0.54, 0.64, 0.72))
    teal += where * covert * band(t, greater - 0.03, greater) * band(across, 20.0, -20.0)
    hand.layer(TURQ, teal, angle, 0.5, length=16, cover=1.5, clip=clip)
    vary = 0.8 + 0.4 * bird.mottle
    build_up(hand, GREENS, green, s, angle, vary=vary, length=13, scale=0.9, clip=clip, spread=0.1)
    blue = where * (1.0 - green / np.maximum(where, 1e-6))
    deeper = np.clip(s + 0.25 * band(t, 0.75, 1.0), 0.0, 1.0)
    build_up(hand, BLUES, blue, deeper, angle, length=40, clip=clip, spread=0.05, bend=0.005)
    # the light along the bend of the wing, lifted out
    bend = [(548, 431), (508, 464), (487, 510), (490, 562), (514, 610), (545, 660)]
    hand.erase(bend, tools.KNEADED, [0.0, 0.5, 0.6, 0.5, 0.3, 0.0], 5.0)


def draw_edges(hand: Hand, bird: Bird) -> None:
    """The edges that are found: the dark under the wing's lower edge, where
    it lies over the flank."""
    lower = outline(WING, 3.0)
    t, _ = frame(WING_AXIS)
    pts = np.array([p for p in lower if 495 < p[0] < 760 and p[1] > 560 and p[0] < 0.42 * p[1] + 380])
    del t
    start = 0
    while start < len(pts) - 3:
        run = int(hand.rng.integers(6, 18))
        seg = pts[start : start + run] + np.array([-2.5, 0.0])
        if len(seg) > 2 and hand.rng.random() < 0.8:
            firm = hand.rng.uniform(0.3, 0.6)
            bear = firm * np.sin(np.linspace(0.2, 2.9, len(seg)))
            hand.line(SPRUCE, seg, bear.tolist(), hand.rng.uniform(3.0, 6.0), 0.8)
        start += run + int(hand.rng.integers(0, 6))


def draw_thigh(hand: Hand, bird: Bird) -> None:
    """The feathered thigh coming down over the tops of the feet, in shadow
    under the belly, its ruffled lower edge over the toes."""
    s = bird.shade["thigh"] + 0.15 * bird.mottle
    where = bird.part["thigh"] * 1.0
    angle = flow(THIGH_FLOW)
    clip = grow(bird.part["thigh"], 3) & ~bird.part["wing_all"]
    build_up(hand, GREENS[:7], where, s, angle, length=9, scale=0.75, clip=clip)
    for y0 in (930.0, 948.0, 964.0):
        for x0 in np.arange(560.0, 670.0, 13.0) + hand.rng.uniform(0, 8):
            if hand.rng.random() < 0.55:
                arc = [(x0 + 11.0 * q, y0 + 3.5 * math.sin(math.pi * q) + hand.rng.normal(0, 0.6)) for q in np.linspace(0, 1, 6)]
                hand.line(SPRUCE, arc, [0.0, 0.45, 0.45, 0.0], 2.4, 0.3, bird.part["thigh"])
    edge = outline(THIGH, 3.0)
    for x, y in edge[edge[:, 1] > 955]:
        a = math.radians(95.0 + hand.rng.normal(0.0, 15.0))
        tick = [(x, y - 4.0), (x + 9.0 * math.cos(a), y - 4.0 + 9.0 * math.sin(a))]
        hand.line(PINE, tick, [0.55, 0.2], 2.6, 0.3)


def draw_tail(hand: Hand, bird: Bird) -> None:
    """The long tail: narrow blades, graduated, the outer ones under and
    shorter; each stroked along its shaft, rust red above going to blue and
    to pale blue at its tip, darker where the next lies over it; a dark
    shaft here and there; the whole loosening toward the end."""
    covered = np.zeros((HEIGHT, WIDTH), bool)
    for k in range(len(BLADES) - 1, -1, -1):
        offset, degrees, length, half = BLADES[k]
        seen = bird.part[f"blade{k}"]
        root = np.array(TAIL_ROOT)
        d = np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])
        axis = (tuple(root + np.array([-d[1], d[0]]) * offset), tuple(root + np.array([-d[1], d[0]]) * offset + d * length))
        u, v = frame(axis)
        s = np.clip(bird.shade["tail"] + 0.45 * soft(covered, 5) - 0.15 * band(np.abs(v), 6.0, 0.0), 0.0, 1.0)
        draw_blade(hand, bird, seen, s, (u, v, half, degrees, k))
        covered |= fill_polygon(to_sheet(blade(*BLADES[k])))


def draw_blade(hand: Hand, bird: Bird, seen: np.ndarray, s: np.ndarray, where) -> None:
    """One long tail feather, ``where`` = (along, across, half-width,
    direction, which)."""
    u, v, half, degrees, k = where
    w = seen * bird.edge
    clip = grow(seen, 2)
    angle = np.full((HEIGHT, WIDTH), math.radians(degrees)) + turning()
    outer = band(np.abs(v), 0.4 * half, 0.9 * half) * (0.6 if k < 2 else 0.25)
    blue = np.clip(band(u, 0.62, 0.8) + outer * band(u, 0.2, 0.5), 0.0, 1.0)
    tip = band(u, 0.86, 0.97)
    on = -(degrees + TURN)
    how = {"clip": clip, "spread": 0.03, "bend": 0.003, "angles": (on, on + 6.0), "hatch": 0.45}
    build_up(hand, RUSTS, w * (1.0 - blue), s, angle, length=60, **how)
    build_up(hand, BLUES[1:], w * blue * (1.0 - tip), s, angle, length=56, **how)
    hand.tone(SKY, w * tip, 0.45, (on,), 4.0)
    hand.layer(SKY, w * tip, angle, 0.5, length=40, cover=1.8, clip=clip)
    if k % 2 == 1 or k == 4:
        ts = np.linspace(0.04, 0.9, 50)
        d = np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])
        a = np.array(BLADES[k][:1])  # (unused)
        del a
        root = np.array(TAIL_ROOT) + np.array([-d[1], d[0]]) * BLADES[k][0]
        shaft = root + np.outer(ts * BLADES[k][2], d)
        lift = np.convolve(hand.rng.random(50) < 0.2, np.ones(4), mode="same") > 0
        bear = np.where(lift, 0.0, 0.45 * (1.0 - 0.5 * ts))
        hand.line(OXBLOOD, shaft, bear.tolist(), 1.8, 0.3, seen)


def draw_face(hand: Hand, bird: Bird) -> None:
    """The bare face: off-white, modelled as a rounded cheek, cooler and
    greyer where it turns under and back, flushed pink toward the bill; the
    eye's place left clean for it."""
    xx, yy = grid()
    face = bird.part["face"] & (np.hypot(xx - EYE[0], yy - EYE[1]) > IRIS + 1.5)
    w = face * 1.0
    cheek = 1.0 - lambert(dome(bird.part["face"], 22, 45.0, 0.8))
    turn = np.clip(band(cheek, 0.2, 0.6) + 0.5 * band(yy, 330, 400), 0.0, 1.0)
    flush = band(xx, 450, 385) * (0.5 + 0.5 * band(yy, 260, 360))
    draw_face_lines(hand, bird)
    hand.tone(COOL, w * turn**1.5, 0.26, (25,), 2.6)
    hand.tone(MAUVE, w * turn**3, 0.22, (70,), 2.6)
    hand.tone(ROSE, w * flush, 0.24, (-20,), 2.6)
    hand.tone(BLUSH, w * (0.4 + 0.6 * flush), 0.2, (10,), 2.6)


def draw_face_lines(hand: Hand, bird: Bird) -> None:
    """Rows of tiny dark feathers across the bare skin, the lowest along the
    jaw, the higher ones flatter, up past the eye: each a broken line of
    tiny tufts of unequal length and spacing, denser near the eye and the
    base of the bill."""
    face = bird.part["face"]
    bx, by = np.meshgrid(np.arange(350.0, 500.0), np.arange(220.0, 420.0))
    rows = [pixel(x, y) for x, y in zip(bx.ravel(), by.ravel(), strict=True)]
    on = np.array([face[r, c] for c, r in rows]).reshape(bx.shape)
    seen = on.any(axis=0)
    cols = bx[0, seen]
    jaw = by[on.shape[0] - 1 - np.argmax(on[::-1, seen], axis=0), 0]
    jaw = np.convolve(np.pad(jaw, 12, mode="edge"), np.ones(25) / 25.0, mode="valid")
    level = float(np.median(jaw))
    ex, ey = EYE
    d = 6.0
    while d < 170.0:
        flat = min(d / 150.0, 1.0)
        row = (1.0 - flat) * jaw + flat * level - d
        draw_face_row(hand, face, cols, row, hand.rng.uniform(0.55, 1.0))
        d += hand.rng.uniform(8.0, 14.0) * (0.8 + 0.4 * flat)
    del ex, ey


def draw_face_row(hand: Hand, face: np.ndarray, cols, row, dark: float) -> None:
    """One row of tiny feathers across the face along ``row`` (y at each of
    ``cols``): little tufts, each two or three short ticks lying back along
    the row, in clusters and gaps of their own."""
    ex, ey = EYE
    phase = hand.rng.uniform(0.0, 6.3)
    x = float(cols[0]) + hand.rng.uniform(0.0, 6.0)
    while x < cols[-1] - 3:
        y = float(np.interp(x, cols, row)) + 1.4 * math.sin(x / 8.0 + phase)
        near = math.exp(-((math.hypot(x - ex, y - ey) - 24.0) / 28.0) ** 2)
        crowd = max(near, math.exp(-(((x - 385.0) / 20.0) ** 2)))
        slope = float(np.interp(x + 2.0, cols, row) - np.interp(x - 2.0, cols, row)) / 4.0
        a = math.atan2(slope, 1.0)
        far = math.hypot(x - ex, y - ey) > IRIS + 8
        if far and face[pixel(x, y)[::-1]] and hand.rng.random() < 0.5 + 0.45 * crowd:
            for _ in range(int(hand.rng.integers(1, 4))):
                tx, ty = x + hand.rng.uniform(-2.0, 2.0), y + hand.rng.normal(0.0, 0.8)
                ln = hand.rng.uniform(2.5, 6.0)
                b = a - 0.3 + hand.rng.normal(0.0, 0.1)
                tick = [(tx, ty), (tx + ln * math.cos(b), ty + ln * math.sin(b))]
                firm = dark * hand.rng.uniform(0.45, 0.85)
                colour = LINE if hand.rng.random() < 0.7 else WINE
                hand.line(colour, tick, [firm, firm * 0.4], hand.rng.uniform(1.8, 2.4), 0.2, face)
        x += hand.rng.uniform(2.5, 6.0) * (1.3 - 0.45 * crowd)

def draw_fringe(hand: Hand, bird: Bird) -> None:
    """The small green feathers at the edge of the bare face and of the
    bill, lying over it irregularly."""
    face, head = bird.part["face"], bird.part["head"]
    _, yy = grid()
    near = grow(face | bird.part["upper"], 7) & head
    angle = flow(HEAD_FLOW)
    s = bird.shade["body"]
    for x, y in hand.seeds(near * 1.0, 260):
        a = float(angle[int(y), int(x)]) + hand.rng.normal(0.0, 0.3)
        size = hand.rng.uniform(6.0, 15.0)
        start = (x - 0.4 * size * math.cos(a), y - 0.4 * size * math.sin(a))
        tip = (x + size * math.cos(a), y + size * math.sin(a))
        shade = float(s[min(int(y), HEIGHT - 1), min(int(x), WIDTH - 1)])
        colour = OLIVE if shade < 0.3 else LEAF if shade < 0.6 else PINE
        hand.sheet.stroke([start, tip], hand.pencil(colour), [0.2, 0.6, 0.15], 2.4)
    del yy


def draw_nape(hand: Hand, bird: Bird) -> None:
    """A few small feathers standing off the back of the head and the nape,
    so that the silhouette there is ruffled, not cut."""
    back = outline(HEAD_BODY, 2.0)
    back = back[(back[:, 0] > 470) & (back[:, 1] < 420)]
    angle = flow(HEAD_FLOW)
    for _ in range(26):
        x, y = back[int(hand.rng.integers(0, len(back)))]
        px, py = pixel(x, y)
        a = float(angle[py, px]) + hand.rng.normal(-0.25, 0.2) - math.radians(TURN)
        size = hand.rng.uniform(8.0, 16.0)
        root = (x - 0.6 * size * math.cos(a), y - 0.6 * size * math.sin(a))
        tip = (x + 0.5 * size * math.cos(a), y + 0.5 * size * math.sin(a))
        colour = LEAF if float(bird.shade["body"][py, px]) < 0.5 else PINE
        hand.line(colour, [root, tip], [0.55, 0.15], 2.4, 0.3)


def draw_eye(hand: Hand) -> None:
    """The small pale eye, round and calm, set in a ring of finely wrinkled
    bare skin: straw-yellow iris with a darker outer ring, black pupil, one
    catchlight left as paper, the lid's edge over its top and the shadow of
    the brow above, so that it sits in the head."""
    xx, yy = grid()
    r = np.hypot(xx - EYE[0], yy - EYE[1])
    lid = EYE[1] - 0.86 * IRIS + 0.002 * (xx - EYE[0]) ** 2
    glint = (EYE[0] - 3.5, EYE[1] - 4.0)
    spared = np.hypot(xx - glint[0], yy - glint[1]) > 2.3
    for k in range(9):
        radius = IRIS + 3.0 + 1.6 * k + hand.rng.normal(0.0, 0.5)
        ring = circle(EYE, radius, 30, hand.rng.uniform(0, 6.3), hand.rng.uniform(1.2, 3.0))
        firm = (0.36 - 0.03 * k) * hand.rng.uniform(0.7, 1.1)
        hand.line(MAUVE if k % 3 else COOL, ring, [0.0, firm, firm, 0.0], 1.8, 0.5)
    above = (r > IRIS) & (r < IRIS + 16.0) & (yy < EYE[1] - 2.0)
    hand.tone(COOL, above * band(yy, EYE[1] - 2.0, EYE[1] - IRIS) * 1.0, 0.32, (30,), 2.4)
    iris = (r < IRIS) & (yy > lid) & spared
    rim = band(r, IRIS - 3.5, IRIS - 0.8) * iris
    hood = band(yy, lid + 6.0, lid) * iris
    hand.tone(STRAW, iris * 1.0, 0.52, (10, 100), 2.4)
    hand.tone(OCHRE, np.clip(rim + 0.7 * hood, 0.0, 1.0), 0.55, (40, -50), 2.4)
    hand.tone(UMBER, rim * 0.6 + hood * 0.5, 0.4, (20,), 2.4)
    pupil = (r < PUPIL) & spared
    hand.tone(BLACK, pupil * 1.0, 1.0, (20, -40, 70, 120, 160), 2.2)
    upper = circle(EYE, IRIS + 1.2, 24, math.pi * 1.08, math.pi * 0.84)
    hand.line(LINE, upper, [0.35, 0.7, 0.8, 0.7, 0.4], 2.6, 0.15)
    lower = circle(EYE, IRIS + 1.0, 24, math.pi * 0.1, math.pi * 0.8)
    hand.line(MAUVE, lower, [0.2, 0.45, 0.45, 0.2], 2.0, 0.15)
    for _ in range(3):
        hand.erase([(glint[0] - 0.8, glint[1]), (glint[0] + 0.8, glint[1])], tools.ERASER, 1.0, 3.0)


def draw_brow(hand: Hand, bird: Bird) -> None:
    """The scarlet band over the bill: velvety, in short dense strokes, deep
    crimson in its shadow and scarlet in the light; its back edge a little
    raised over the green of the crown, which it shades."""
    angle = flow(BROW_FLOW)
    brow = bird.part["brow"]
    w = soft(brow, 3) * grow(brow, 2)
    clip = grow(brow, 4) & bird.part["whole"] & ~bird.part["upper"] & ~bird.part["face"]
    xx, yy = grid()
    shade = np.clip(band(yy, 200, 245) + band(xx, 395, 438), 0.0, 1.0)
    edge = grow(brow, 6) & ~brow & bird.part["head"] & (xx > 390)
    hand.layer(DEEP, soft(edge, 2) * edge, flow(HEAD_FLOW), 0.5, length=6, width=3.0, cover=1.4, clip=edge)
    hand.layer(OXBLOOD, w * shade**2, angle, 0.6, length=5, width=2.6, cover=1.8, clip=clip)
    hand.layer(CRIMSON, w * shade, angle, 0.62, length=6, width=2.6, cover=2.0, clip=clip)
    hand.layer(SCARLET, w, angle, 0.6, length=6, width=2.8, cover=2.6, clip=clip)
    hand.layer(FLAME, w * (1.0 - shade), angle, 0.45, length=6, width=2.8, cover=1.2, clip=clip)


def arch_flow(points) -> np.ndarray:
    """A smooth field of directions (H, W) that runs along the curve through
    ``points`` of the bird and parallel to it nearby."""
    line = outline(points, 12.0, closed=False)
    tangent = np.gradient(line, axis=0)
    degrees = np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0]))
    return flow([(x, y, a) for (x, y), a in zip(line, degrees, strict=True)])


def bill_tone(bird: Bird) -> np.ndarray:
    """How dark the bill is, 0..1: the upper bill blackish, lit along the
    top of its arch, darkening toward its cutting edge and toward the face,
    paler and horny toward the hook's point; the lower bill nearly black."""
    upper, lower = bird.part["upper"], bird.part["lower"]
    xx, yy = grid()
    ridge = distance_to(CULMEN, upper)
    edge = distance_to(TOMIUM, upper)
    up = 0.6 + 0.18 * band(ridge, 20.0, 80.0) + 0.1 * band(xx, 330.0, 385.0)
    up -= 0.32 * band(ridge, 26.0, 16.0) * band(yy, 450.0, 320.0)
    up += 0.1 * band(edge, 30.0, 8.0)
    up -= 0.3 * band(yy, 400.0, 470.0) * band(xx, 305.0, 268.0)
    low = 0.86 + 0.14 * band(distance_to(GONYS, lower), 3.0, 14.0)
    low -= 0.3 * band(distance_to(GONYS, lower), 6.0, 2.0)
    return np.clip(np.where(upper, up, np.where(lower, low, 0.0)), 0.0, 1.0)


def draw_bill(hand: Hand, bird: Bird) -> None:
    """The great hooked bill as horn: built in layered strokes that follow
    its arch, the highlight left as lighter pencil with one crisp lower
    edge, growth lines near its base, the cutting edge with its notch, and
    the small lower bill, nearly black, tucked under."""
    s = bill_tone(bird)
    upper, lower = bird.part["upper"], bird.part["lower"]
    xx, yy = grid()
    arch = arch_flow(CULMEN)
    low_arch = arch_flow(GONYS[::-1])
    tip = band(yy, 400.0, 465.0) * band(xx, 310.0, 275.0)
    for mask, angle, length in ((upper, arch, 26.0), (lower, low_arch, 14.0)):
        how = {"clip": mask, "spread": 0.03, "bend": 0.0, "hatch": 0.55, "angles": (55.0, 125.0, 95.0)}
        build_up(hand, INKS, mask * 1.0, s, angle, length=length, **how)
        hand.tone(HORN, mask * tip, 0.4, (60.0, 120.0), 3.0)
        hand.layer(HORN, mask * tip, angle, 0.4, length=14, width=3.0, cover=1.4, clip=mask)
    draw_bill_lines(hand, bird)


def draw_bill_lines(hand: Hand, bird: Bird) -> None:
    """The cutting edge and its notch, the bill's base against the face,
    growth lines near the base, and the light along the ridge."""
    upper = bird.part["upper"]
    hand.line(BLACK, TOMIUM, [0.0, 0.35, 0.7, 0.9, 0.9, 0.9, 0.8, 0.6], 2.6, 0.2)
    hand.line(SLATE, CULMEN[1:], [0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.6], 2.0, 0.2)
    base = [(362, 236), (370, 266), (378, 300), (385, 336), (389, 366)]
    hand.line(CHAR, base, [0.3, 0.7, 0.7, 0.6, 0.4], 2.6, 0.3)
    for k in range(5):
        off = 8.0 + 7.0 * k + hand.rng.normal(0.0, 1.5)
        growth = [(x - off, y + hand.rng.normal(0.0, 0.6)) for x, y in base[1:]]
        firm = hand.rng.uniform(0.2, 0.35)
        hand.line(SLATE if k % 2 else GREY, growth, [0.0, firm, firm, 0.0], 1.8, 0.4, upper)
    ridge = [(352, 242), (322, 245), (296, 260), (276, 285), (264, 318), (259, 350)]
    hand.erase(ridge, tools.KNEADED, [0.0, 0.3, 0.4, 0.35, 0.25, 0.0], 6.0)


def draw_feet(hand: Hand, bird: Bird) -> None:
    """The feet gripping the branch: thick dark grey toes, rounded under the
    light, ringed with scales, and long dark curved claws."""
    for toe, root in TOES:
        mask = fill_polygon(to_sheet(toe_outline(toe, root)))
        pts = outline(toe, 1.5, closed=False)
        d = np.gradient(pts, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        half = np.linspace(root, root * 0.55, len(pts))[:, None]
        under = fill_polygon(to_sheet(np.concatenate([pts + n * half * 0.1, (pts + n * half)[::-1]])))
        hand.tone(SLATE, mask * 1.0, 0.65, (30, -50), 2.4)
        hand.tone(CHAR, under * 1.0, 0.7, (70, -20), 2.4)
        hand.tone(GREY, mask * 1.0, 0.45, (110,), 2.4)
        for k in range(3, len(pts) - 2, 3):
            ring = [pts[k] - n[k] * half[k, 0], pts[k] + d[k] * 1.6, pts[k] + n[k] * half[k, 0]]
            hand.line(CHAR, ring, [0.0, 0.5, 0.0], 1.6, 0.3, mask)
        end = pts[-1]
        claw = [end - 2 * d[-1], end + 7 * d[-1] - 1 * n[-1], end + 13 * d[-1] - 6 * n[-1], end + 14 * d[-1] - 12 * n[-1]]
        hand.line(BLACK, claw, [0.95, 0.85, 0.6, 0.2], 4.6, 0.15)
    del bird


def draw_branch(hand: Hand, bird: Bird) -> None:
    """The branch, drawn only where the bird holds it: round, light on top
    and dark beneath, the bird's shadow across it, the grain of its bark in
    long broken lines, fading out either side."""
    xx, yy = grid()
    line = outline(BRANCH, 3.0, closed=False)
    v = (yy - np.interp(xx, line[:, 0], line[:, 1])) / BRANCH_HALF
    near = np.exp(-(((xx - 640.0) / 260.0) ** 2))
    shadow = band(xx, 560.0, 620.0) * band(xx, 840.0, 760.0)
    w = bird.part["branch"] * near
    angle = np.full((HEIGHT, WIDTH), math.radians(1.5)) + turning()
    dark = np.clip(band(v, 0.0, 0.9) + 0.7 * shadow, 0.0, 1.0)
    hand.tone(SEPIA, w * dark, 0.6, (-5, 30), 2.6)
    hand.layer(UMBER, w * band(v, -0.6, 0.3), angle, 0.45, length=34, width=4.0, cover=1.6)
    hand.layer(WARM_GREY, w, angle, 0.32, length=40, width=5.0, cover=1.0, tilt=0.6)
    for off in hand.rng.uniform(-0.8, 0.9, 12):
        pts = line + np.array([0.0, off * BRANCH_HALF])
        fade = np.exp(-(((pts[:, 0] - 640.0) / 230.0) ** 2))
        grain = fade * hand.rng.uniform(0.25, 0.5) * (0.5 + 0.5 * np.sin(pts[:, 0] / 17.0 + off * 9))
        keep = fade > 0.05
        hand.line(SEPIA, pts[keep], grain[keep].tolist(), 2.0, 0.8, ~bird.part["toes"])


def main() -> None:
    rng = np.random.default_rng(SEED)
    paper = Paper("wove", grain=2.0, depth=0.4, bite=0.5, give=0.4, color="#fbfaf6")
    sheet = Sheet(WIDTH, HEIGHT, paper=paper, seed=SEED)
    hand = Hand(sheet, rng)
    bird = build(rng)
    xx, yy = grid()
    rows = np.mgrid[:HEIGHT, :WIDTH][0]
    hand.finish = 1.0 - 0.35 * band(np.hypot(xx - EYE[0], yy - EYE[1]), 380.0, 1400.0)
    hand.finish *= 1.0 - 0.6 * band(rows, HEIGHT - 260.0, HEIGHT - 10.0)
    draw_sketch(hand)
    draw_bill(hand, bird)
    draw_tail(hand, bird)
    draw_edges(hand, bird)
    draw_plumage(hand, bird)
    draw_thigh(hand, bird)
    draw_wing(hand, bird)
    draw_face(hand, bird)
    draw_fringe(hand, bird)
    draw_nape(hand, bird)
    draw_brow(hand, bird)
    draw_eye(hand)
    draw_feet(hand, bird)
    draw_branch(hand, bird)
    out = Path(__file__).parent / "output" / "macaw.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
