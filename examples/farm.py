"""A joyful farm on a hill, in wax crayon: naive art in bold colours.

Run from the repository root: ``python examples/farm.py``. With
``python examples/farm.py --study FOLDER`` it writes the flat design study
(cut-paper shapes, flat colour and outlines) to FOLDER instead.

A red barn and a little white house with a cat on its roof sit on a round
green hill under a big sun; behind them a farther hill is a patchwork of
fields, with a windmill. In front, a meadow with a round blue pond and its
ducks, a spotted cow, a pig, hens and a rooster; sheep graze on the hill,
apple trees and lollipop trees stand about, sunflowers grow by the barn.

Everything is a flat shape with a wobbling outline, stacked as cut paper is,
nearer below, farther above.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from murex import CatmullRom, noise, write_png

WIDTH, HEIGHT = 1400, 1050
SEED = 11
PAPER_HEX = "#f8f6ef"

SKY = "#3a86d6"
SUN = "#ffd21f"
ORANGE = "#f7841f"
RED = "#d9261c"
PINK = "#f59bb3"
ROSE = "#ec6a8c"
GREEN = "#36a23a"
LIME = "#a8cf2c"
LEAF = "#1f7a3a"
BROWN = "#8b4a22"
NAVY = "#244fa8"
POND = "#1f6fcc"
BLACK = "#1c1a1a"
SMOKE = "#8a8a8a"
CREAM = "#f3e7b0"

OUTLINE = 11.0  # width of a black outline, in pixels

# ----------------------------------------------------------------------
# the design: flat shapes


@dataclass
class Mark:
    """A line laid as it is (a stripe, a stalk, a tail, a bird); ``clip``
    keeps it inside a part. ``colour`` None keeps the paper bare there."""

    path: np.ndarray
    colour: str | None
    width: float = 10.0
    clip: Part | None = None


@dataclass
class Part:
    """A closed shape of one flat colour (None: the bare paper), filled with
    strokes running at ``angle`` degrees, outlined ``line`` pixels wide (0:
    not outlined) in ``ink``, inside ``clip`` if given."""

    outline: np.ndarray
    colour: str | None
    angle: float = 30.0
    line: float = OUTLINE
    clip: Part | None = None
    name: str = ""
    ink: str = BLACK


Item = Part | Mark


class Place:
    """Local coordinates of a figure (facing right, in pixels) to the sheet:
    scaled, mirrored to face left, turned and moved."""

    def __init__(self, x, y, scale=1.0, left=False, turn=0.0):
        self.at = np.array([x, y], float)
        self.scale = np.array([-scale if left else scale, scale])
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        self.rot = np.array([[c, -s], [s, c]])

    def __call__(self, pts) -> np.ndarray:
        return (np.asarray(pts, float) * self.scale) @ self.rot.T + self.at


class Hand:
    """Draws shapes as a hand does: through a few points, never exactly."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def u(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def jitter(self, pts, amount: float) -> np.ndarray:
        pts = np.asarray(pts, float)
        return pts + self.rng.normal(0.0, amount, pts.shape)

    def _wobble(self, pts, tan, amount: float, cell: float) -> np.ndarray:
        n = len(pts)
        wob = (noise.value_noise_1d(n, cell, self.rng) - 0.5) * 2.0 * amount
        normal = np.stack([-tan[:, 1], tan[:, 0]], 1)
        return pts + normal * wob[:, None]

    def closed(self, ctrl, jitter=1.5, wobble=1.2) -> np.ndarray:
        """A closed curve through ``ctrl`` (each point a little off)."""
        curve = CatmullRom(self.jitter(ctrl, jitter), closed=True)
        pts, tan, _, _ = curve.sample(2.0)
        return self._wobble(pts, tan, wobble, 14.0)

    def line(self, ctrl, jitter=1.0, wobble=1.0) -> np.ndarray:
        """An open curve through ``ctrl``."""
        pts, tan, _, _ = CatmullRom(self.jitter(ctrl, jitter)).sample(2.0)
        return self._wobble(pts, tan, wobble, 14.0)

    def ellipse(self, cx, cy, rx, ry, n=9, irregular=0.05, turn=0.0) -> np.ndarray:
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + self.u(0, 2 * np.pi)
        r = 1.0 + self.rng.normal(0.0, irregular, n)
        x, y = rx * r * np.cos(a), ry * r * np.sin(a)
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)

    def blob(self, cx, cy, rx, ry, n=9, irregular=0.05, turn=0.0) -> np.ndarray:
        return self.closed(self.ellipse(cx, cy, rx, ry, n, irregular, turn), 0.5)

    def corners(self, pts, step=45.0, near=7.0) -> np.ndarray:
        """Control points for a polygon drawn freehand: its corners, a point
        close to each side of them (so that they round only a little) and a
        point every ``step`` pixels along its sides."""
        pts = np.asarray(pts, float)
        out = []
        for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=True):
            length = float(np.linalg.norm(b - a))
            d = (b - a) / max(length, 1e-9)
            out.append(a)
            k = max(int(length // step), 0)
            if length > 3 * near:
                out.append(a + d * near)
                out += [a + (b - a) * t for t in np.linspace(0, 1, k + 2)[1:-1]]
                out.append(b - d * near)
        return np.array(out)

    def poly(self, pts, jitter=1.5, wobble=1.0) -> np.ndarray:
        return self.closed(self.corners(pts), jitter, wobble)

    def scallop(self, cx, cy, rx, ry, lobes=7, depth=0.16, flat=0.5) -> np.ndarray:
        """A cloud's outline: round lobes all round, flatter underneath."""
        n = lobes * 8
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        phase = self.u(0, np.pi)
        bump = np.abs(np.sin(lobes * a / 2 + phase)) ** 0.7
        size = (
            1.0 + self.rng.normal(0, 0.04, lobes)[(a / (2 * np.pi) * lobes).astype(int)]
        )
        r = (1.0 - depth + depth * bump) * size
        y = np.sin(a)
        y = np.where(y > 0, y * flat, y)
        return self.closed(np.stack([cx + rx * r * np.cos(a), cy + ry * r * y], 1), 0.6)


# ----------------------------------------------------------------------
# the sky


def sky(hand: Hand) -> list[Item]:
    box = hand.corners([(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)], 200)
    return [Part(box, SKY, angle=12.0, line=0.0, name="sky")]


def sun(hand: Hand, cx=1165.0, cy=168.0, r=86.0) -> list[Item]:
    items: list[Item] = []
    n = 13
    start = hand.u(0, 2 * np.pi)
    for k in range(n):
        a = start + 2 * np.pi * k / n + hand.u(-0.06, 0.06)
        long = r + (62 if k % 2 else 44) + hand.u(-6, 6)
        half = (0.13 if k % 2 else 0.17) + hand.u(-0.02, 0.02)
        base = r + 14
        pts = [
            (cx + base * math.cos(a - half), cy + base * math.sin(a - half)),
            (cx + long * math.cos(a), cy + long * math.sin(a)),
            (cx + base * math.cos(a + half), cy + base * math.sin(a + half)),
        ]
        items.append(Part(hand.poly(pts, 1.0, 0.6), ORANGE, math.degrees(a), 9.0))
    items.append(Part(hand.blob(cx, cy, r, r * 0.97, 11, 0.025), SUN, 60.0, 12.0))
    return items


def clouds(hand: Hand) -> list[Item]:
    items: list[Item] = []
    for cx, cy, rx, ry, lobes in (
        (165, 108, 118, 52, 7),
        (640, 175, 135, 56, 8),
        (905, 88, 92, 40, 6),
    ):
        items.append(Part(hand.scallop(cx, cy, rx, ry, lobes), None, name="cloud"))
    return items


def birds(hand: Hand) -> list[Item]:
    items: list[Item] = []
    for x, y, s in (
        (418, 262, 1.0),
        (470, 228, 0.85),
        (775, 118, 0.9),
        (1010, 300, 0.75),
    ):
        w = 26 * s
        d = hand.u(-3, 3)
        path = [
            (x - w, y - 8 * s + d),
            (x - w / 2, y - 12 * s),
            (x, y),
            (x + w / 2, y - 13 * s),
            (x + w, y - 7 * s - d),
        ]
        items.append(Mark(hand.line(path, 0.8, 0.5), BLACK, 7.0))
    return items


# ----------------------------------------------------------------------
# the hills and their fields


def silhouette(hand: Hand, ctrl) -> tuple[np.ndarray, Callable]:
    """The top line of a hill, and its height as a function of x."""
    top = hand.line(ctrl, 2.0, 1.5)
    order = np.argsort(top[:, 0])
    xs, ys = top[order, 0], top[order, 1]

    def height(x):
        return np.interp(x, xs, ys)

    return top, height


def band(top: np.ndarray, bottom: float) -> np.ndarray:
    """The region under the line ``top`` down to ``bottom``."""
    return np.vstack([top, [(top[-1, 0], bottom), (top[0, 0], bottom)]])


def dividers(hand: Hand, height, cuts, bottom) -> list[np.ndarray]:
    """The lines between fields: each from above the hilltop at x down to
    ``bottom``, slanting by s (x moves s pixels for each pixel down)."""
    lines = []
    for x, s in cuts:
        y0 = float(height(min(max(x, 0.0), WIDTH)))
        ys = np.linspace(y0 - 60, bottom, 10)
        pts = np.stack([x + s * (ys - y0), ys], 1)
        inside = 0 < x < WIDTH
        lines.append(hand.line(pts, 2.0, 1.2) if inside else pts)
    return lines


def fields(hand: Hand, height, cuts, bottom, hill: Part, looks):
    """The fields of a hill, side by side between its dividers, and the
    dividers as lines."""
    lines = dividers(hand, height, cuts, bottom)
    patches = []
    for left, right, (colour, angle) in zip(lines[:-1], lines[1:], looks, strict=True):
        poly = np.vstack([left[::-1], right])
        patches.append(Part(poly, colour, angle, 0.0, hill, "field"))
    edges = [Mark(line, BLACK, 9.0, hill) for line in lines[1:-1]]
    return patches, edges


def contour_stripes(hand, height, x0, x1, start, gap, count, colour, width, clip):
    """Furrows: lines that follow the hill's curve, ``gap`` apart."""
    items: list[Item] = []
    xs = np.linspace(x0, x1, max(int((x1 - x0) / 60), 3))
    for k in range(count):
        d = start + k * gap * hand.u(0.9, 1.1)
        pts = np.stack([xs, height(xs) + d + hand.rng.normal(0, 2.0, len(xs))], 1)
        items.append(Mark(hand.line(pts, 1.0, 1.0), colour, width, clip))
    return items


def slant_stripes(hand, box, angle, gap, colour, width, clip) -> list[Item]:
    """Stripes straight across ``box`` at ``angle`` degrees, ``gap`` apart."""
    x0, y0, x1, y1 = box
    t = math.radians(angle)
    along = np.array([math.cos(t), -math.sin(t)])
    across = np.array([math.sin(t), math.cos(t)])
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    reach = math.hypot(x1 - x0, y1 - y0) / 2
    items: list[Item] = []
    for off in np.arange(-reach, reach, gap):
        mid = centre + across * (off + hand.u(-0.1, 0.1) * gap)
        pts = [mid + along * s for s in np.linspace(-reach, reach, 6)]
        items.append(Mark(hand.line(pts, 1.5, 1.0), colour, width, clip))
    return items


def dot_rows(hand, height, x0, x1, start, gap, rows, step, colour, r, clip):
    """Cabbages: rows of round dots along the hill's curve."""
    items: list[Item] = []
    for k in range(rows):
        shift = hand.u(0, step)
        for x in np.arange(x0 + shift, x1, step):
            x = x + hand.u(-0.15, 0.15) * step
            y = float(height(x)) + start + k * gap + hand.u(-3, 3)
            rr = r * hand.u(0.8, 1.15)
            items.append(
                Part(
                    hand.blob(x, y, rr, rr * 0.85, 6, 0.1),
                    colour,
                    hand.u(0, 180),
                    0.0,
                    clip,
                )
            )
    return items


FAR_CUTS = [
    (-60, 0.0),
    (175, -0.35),
    (455, -0.25),
    (655, 0.05),
    (870, 0.3),
    (1190, 0.4),
    (1470, 0.0),
]


def far_hills(hand: Hand):
    ctrl = [
        (-40, 428),
        (120, 384),
        (320, 352),
        (500, 374),
        (650, 398),
        (830, 372),
        (1050, 330),
        (1250, 350),
        (1440, 408),
    ]
    top, height = silhouette(hand, ctrl)
    hill = Part(band(top, 780), None, line=0.0, name="far hill")
    looks = [
        (GREEN, -30.0),
        (SUN, 70.0),
        (LIME, 5.0),
        (ORANGE, -10.0),
        (GREEN, 20.0),
        (LIME, 60.0),
    ]
    patches, edges = fields(hand, height, FAR_CUTS, 800, hill, looks)
    items: list[Item] = [hill, *patches]
    items += far_patterns(hand, height, patches)
    items += edges
    items.append(Mark(top, BLACK, OUTLINE))
    return items, height


def far_patterns(hand, height, patches) -> list[Item]:
    xs = [x for x, _ in FAR_CUTS]
    p = patches
    items: list[Item] = []
    items += slant_stripes(
        hand, (xs[0] - 40, 360, xs[1] + 120, 760), 55, 34, LEAF, 9.0, p[0]
    )
    items += wheat(hand, height, xs[1] - 60, xs[2] + 120, p[1])
    items += contour_stripes(
        hand, height, xs[2] - 80, xs[3] + 80, 24, 26, 12, GREEN, 8.0, p[2]
    )
    items += contour_stripes(
        hand, height, xs[3] - 80, xs[4] + 120, 22, 24, 14, RED, 8.0, p[3]
    )
    items += dot_rows(
        hand, height, xs[4] - 60, xs[5] + 160, 30, 34, 9, 40, LEAF, 11.0, p[4]
    )
    items += slant_stripes(
        hand, (xs[5] - 60, 330, xs[6] + 80, 760), -60, 40, GREEN, 9.0, p[5]
    )
    return items


def wheat(hand, height, x0, x1, clip) -> list[Item]:
    """A yellow field: short orange dashes in loose rows."""
    items: list[Item] = []
    for k in range(10):
        shift = hand.u(0, 30)
        for x in np.arange(x0 + shift, x1, 30):
            y = float(height(x)) + 22 + 26 * k + hand.u(-3, 3)
            lean = hand.u(-4, 4)
            items.append(
                Mark(
                    hand.line([(x + lean, y - 9), (x - lean, y + 9)], 0.8, 0.3),
                    ORANGE,
                    8.0,
                    clip,
                )
            )
    return items


def middle_hill(hand: Hand) -> list[Item]:
    ctrl = [
        (-40, 612),
        (150, 562),
        (420, 508),
        (700, 480),
        (980, 494),
        (1250, 532),
        (1440, 578),
    ]
    top, height = silhouette(hand, ctrl)
    hill = Part(band(top, 900), None, line=0.0, name="hill")
    cuts = [(-60, 0.0), (330, 0.45), (1470, 0.0)]
    looks = [(LIME, 15.0), (GREEN, -8.0)]
    patches, edges = fields(hand, height, cuts, 900, hill, looks)
    return [hill, *patches, *edges, Mark(top, BLACK, OUTLINE)]


def meadow(hand: Hand):
    """The meadow in front, and the height of its top edge at x."""
    ctrl = [(-40, 768), (250, 744), (560, 754), (860, 772), (1150, 748), (1440, 738)]
    top, height = silhouette(hand, ctrl)
    part = Part(band(top, HEIGHT + 40), GREEN, 18.0, 0.0, name="meadow")
    return [part, Mark(top, BLACK, OUTLINE)], height


# ----------------------------------------------------------------------
# buildings


def windmill(hand: Hand, x=318.0, base=372.0) -> list[Item]:
    items: list[Item] = []
    tower = [(x - 34, base), (x + 34, base), (x + 21, base - 112), (x - 21, base - 112)]
    items.append(Part(hand.poly(tower), None, 90.0, 10.0))
    items.append(
        Part(
            hand.poly(
                [
                    (x - 12, base),
                    (x + 12, base),
                    (x + 12, base - 30),
                    (x, base - 38),
                    (x - 12, base - 30),
                ],
                1.0,
                0.5,
            ),
            NAVY,
            90.0,
            8.0,
        )
    )
    cap = [
        (x - 30, base - 106),
        (x + 30, base - 106),
        (x + 22, base - 132),
        (x, base - 150),
        (x - 22, base - 132),
    ]
    items.append(Part(hand.poly(cap), RED, 30.0, 10.0))
    hub = np.array([x, base - 116.0])
    start = hand.u(38, 50)
    for k in range(4):
        a = math.radians(start + 90 * k + hand.u(-4, 4))
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        r0, r1, w = 16.0, 92.0 + hand.u(-5, 5), 11.0
        pts = [
            hub + d * r0,
            hub + d * r0 + n * w * 2,
            hub + d * r1 + n * w * 2,
            hub + d * r1,
        ]
        items.append(Part(hand.poly(pts, 1.0, 0.6), None, 0.0, 8.0))
        for t in (0.35, 0.65):
            rung = [
                hub + d * (r0 + t * (r1 - r0)),
                hub + d * (r0 + t * (r1 - r0)) + n * w * 2,
            ]
            items.append(Mark(hand.line(rung, 0.5, 0.3), BLACK, 6.0))
    items.append(Part(hand.blob(hub[0], hub[1], 9, 9, 6, 0.05), BLACK, 0.0, 6.0))
    return items


def house(hand: Hand, cx=468.0, base=656.0) -> list[Item]:
    items: list[Item] = []
    chimney = [
        (cx + 44, base - 160),
        (cx + 72, base - 160),
        (cx + 72, base - 246),
        (cx + 44, base - 246),
    ]
    items.append(Part(hand.poly(chimney), RED, 90.0, 10.0))
    items += smoke(hand, cx + 60, base - 262)
    walls = [
        (cx - 110, base),
        (cx + 110, base),
        (cx + 108, base - 128),
        (cx - 108, base - 128),
    ]
    items.append(Part(hand.poly(walls), None, 90.0))
    roof = [
        (cx - 132, base - 122),
        (cx + 132, base - 122),
        (cx + 92, base - 206),
        (cx - 92, base - 206),
    ]
    items.append(Part(hand.poly(roof), ORANGE, 20.0))
    for wx in (cx - 72, cx + 52):
        win = [
            (wx - 22, base - 100),
            (wx + 22, base - 100),
            (wx + 22, base - 56),
            (wx - 22, base - 56),
        ]
        items.append(Part(hand.poly(win, 1.0, 0.6), SUN, 45.0, 9.0))
        items.append(
            Mark(hand.line([(wx, base - 100), (wx, base - 56)], 0.5, 0.3), BLACK, 7.0)
        )
        items.append(
            Mark(
                hand.line([(wx - 22, base - 78), (wx + 22, base - 78)], 0.5, 0.3),
                BLACK,
                7.0,
            )
        )
    door = [
        (cx - 24, base),
        (cx + 22, base),
        (cx + 22, base - 80),
        (cx - 24, base - 80),
    ]
    items.append(Part(hand.poly(door, 1.0, 0.6), GREEN, 90.0, 9.0))
    items.append(Part(hand.blob(cx + 11, base - 40, 4.5, 4.5, 5), SUN, 0.0, 0.0))
    items += cat(hand, Place(cx - 58, base - 204, 0.85))
    return items


def smoke(hand: Hand, x, y) -> list[Item]:
    """Smoke from the chimney: three puffs, growing as they rise and drift."""
    items: list[Item] = []
    for k, r in enumerate((10.0, 14.0, 19.0)):
        cx, cy = x + 16 * k + 6 * k * k, y - 30 * k - 4 * k * k
        items.append(
            Part(
                hand.scallop(cx, cy, r * 1.25, r, 5, 0.18, 0.9),
                None,
                0.0,
                6.0,
                ink=SMOKE,
            )
        )
    return items


def barn(hand: Hand, cx=925.0, base=700.0, s=1.08) -> list[Item]:
    def at(pts):
        return [(cx + s * x, base + s * y) for x, y in pts]

    items: list[Item] = []
    gable = [(-135, -182), (-100, -268), (0, -322), (100, -268), (135, -182)]
    outer = [(-156, -168), (-116, -282), (0, -346), (116, -282), (156, -168)]
    roof = np.vstack([at(outer), np.array(at(gable))[::-1] + [0, 4]])
    items.append(Part(hand.poly(roof, 1.0, 0.8), NAVY, 30.0))
    items.append(
        Part(hand.poly(at([(-135, 0), (135, 0), *gable[::-1]]), 1.5, 1.0), RED, 80.0)
    )
    (x0, y0), (x1, y1) = at([(-66, -150), (66, 0)])
    items += cross_door(hand, x0, y0, x1, y1)
    loft = at([(-36, -258), (36, -258), (36, -196), (-36, -196)])
    items.append(Part(hand.poly(loft, 1.0, 0.6), None, 0.0, 9.0))
    hay = at([(-24, -246), (24, -246), (26, -207), (-25, -207)])
    items.append(Part(hand.poly(hay, 1.0, 0.6), SUN, 70.0, 7.0))
    return items


def cross_door(hand: Hand, x0, y0, x1, y1) -> list[Item]:
    """A big door: a white frame and a white cross-brace, red between."""
    items: list[Item] = [
        Part(hand.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), None, 0.0)
    ]
    i, d = 15.0, 15.0
    a, b, c, e = x0 + i, y0 + i, x1 - i, y1 - i
    mx, my = (a + c) / 2, (b + e) / 2
    k = (e - b) / (c - a)
    for tri, angle in (
        ([(a + d, b), (c - d, b), (mx, my - d * k)], 0.0),
        ([(a + d, e), (c - d, e), (mx, my + d * k)], 0.0),
        ([(a, b + d * k), (a, e - d * k), (mx - d, my)], 90.0),
        ([(c, b + d * k), (c, e - d * k), (mx + d, my)], 90.0),
    ):
        items.append(Part(hand.poly(tri, 1.0, 0.5), RED, angle, 7.0))
    return items


def fence(hand: Hand, height_at, x0=-10.0, x1=560.0) -> list[Item]:
    items: list[Item] = []
    xs = [x0]
    while xs[-1] < x1:
        xs.append(xs[-1] + hand.u(46, 58))
    for a, b in zip(xs, xs[1:], strict=False):
        for dy in (24, 50):
            ya, yb = height_at(a) - dy + hand.u(-2, 2), height_at(b) - dy + hand.u(
                -2, 2
            )
            rail = [(a, ya - 6), (b, yb - 6), (b, yb + 6), (a, ya + 6)]
            items.append(Part(hand.poly(rail, 0.8, 0.5), None, 0.0, 7.0))
    for x in xs:
        y = height_at(x) + 8
        h = hand.u(66, 76)
        lean = hand.u(-2.5, 2.5)
        post = [
            (x - 8, y),
            (x + 8, y),
            (x + 8 + lean, y - h),
            (x + lean, y - h - 12),
            (x - 8 + lean, y - h),
        ]
        items.append(Part(hand.poly(post, 0.8, 0.5), None, 90.0, 8.0))
    return items


# ----------------------------------------------------------------------
# trees and flowers


def lollipop(hand: Hand, x, base, h, r, colour=GREEN) -> list[Item]:
    lean = hand.u(-4, 4)
    w = max(5.0, r * 0.22)
    trunk = [
        (x - w, base),
        (x + w, base),
        (x + w * 0.8 + lean, base - h),
        (x - w * 0.8 + lean, base - h),
    ]
    crown = hand.blob(x + lean, base - h - r * 0.75, r * hand.u(0.95, 1.05), r, 9, 0.04)
    line = 9.0 if r > 30 else 7.0
    return [
        Part(hand.poly(trunk), BROWN, 90.0, line),
        Part(crown, colour, hand.u(20, 70), line),
    ]


def scatter(
    hand: Hand, count, x0, y0, x1, y1, apart, avoid=()
) -> list[tuple[float, float]]:
    """``count`` points in the box, at least ``apart`` from one another and
    outside the ``avoid`` boxes, as a hand spaces dots by eye."""
    out: list[tuple[float, float]] = []
    tries = 0
    while len(out) < count and tries < 4000:
        tries += 1
        x, y = hand.u(x0, x1), hand.u(y0, y1)
        if any(a <= x <= c and b <= y <= d for a, b, c, d in avoid):
            continue
        if all((x - px) ** 2 + (y - py) ** 2 >= apart**2 for px, py in out):
            out.append((x, y))
    return out


def apple_tree(hand: Hand, x, base, h, r) -> list[Item]:
    lean = hand.u(-4, 4)
    trunk = [
        (x - 13, base),
        (x + 13, base),
        (x + 9 + lean, base - h),
        (x - 9 + lean, base - h),
    ]
    cy = base - h - r * 0.7
    crown = hand.scallop(x + lean, cy, r, r * 0.85, 9, 0.12, 0.9)
    items: list[Item] = [
        Part(hand.poly(trunk), BROWN, 90.0, 10.0),
        Part(crown, LEAF, hand.u(30, 60)),
    ]
    box = (x + lean - r * 0.68, cy - r * 0.58, x + lean + r * 0.68, cy + r * 0.55)
    for px, py in scatter(hand, 8, *box, 26):
        rr = hand.u(9, 11)
        items.append(
            Part(hand.blob(px, py, rr, rr * 0.95, 6, 0.05), RED, hand.u(0, 180), 0.0)
        )
    return items


def sunflower(hand: Hand, x, base, h, r) -> list[Item]:
    lean = hand.u(-8, 8)
    head = np.array([x + lean, base - h])
    stalk = [(x, base), (x + lean * 0.3, base - h * 0.5), head]
    items: list[Item] = [Mark(hand.line(stalk, 1.0, 1.0), LEAF, 12.0)]
    for side, t in ((-1, hand.u(0.3, 0.4)), (1, hand.u(0.5, 0.6))):
        sx, sy = x + lean * t, base - h * t
        size = hand.u(0.85, 1.1)
        leaf = [
            (sx, sy),
            (sx + side * 22 * size, sy - 20 * size),
            (sx + side * 44 * size, sy - 14 * size),
            (sx + side * 24 * size, sy + 2),
        ]
        items.append(Part(hand.closed(leaf, 1.0, 0.6), LEAF, 0.0, 7.0))
    n = int(hand.u(11, 14))
    start = hand.u(0, 2 * np.pi)
    for k in range(n):
        a = start + 2 * np.pi * k / n + hand.u(-0.08, 0.08)
        d = r * hand.u(0.64, 0.7)
        px, py = head[0] + d * math.cos(a), head[1] + d * math.sin(a)
        petal = hand.blob(
            px, py, r * hand.u(0.3, 0.36), r * 0.17, 7, 0.04, math.degrees(a)
        )
        items.append(Part(petal, SUN, math.degrees(a), 5.0, ink=ORANGE))
    items.append(
        Part(hand.blob(head[0], head[1], r * 0.4, r * 0.4, 7, 0.05), BROWN, 45.0, 8.0)
    )
    return items


def flower(hand: Hand, x, y, colour) -> list[Item]:
    items: list[Item] = []
    n = int(hand.u(5, 7))
    start = hand.u(0, 2 * np.pi)
    size = hand.u(0.85, 1.15)
    for k in range(n):
        a = start + 2 * np.pi * k / n + hand.u(-0.15, 0.15)
        px, py = x + 10 * size * math.cos(a), y + 10 * size * math.sin(a)
        items.append(
            Part(
                hand.blob(px, py, 7 * size, 7 * size, 5, 0.1),
                colour,
                hand.u(0, 180),
                0.0,
            )
        )
    middle = SUN if colour != SUN else ORANGE
    items.append(
        Part(hand.blob(x, y, 5.5 * size, 5.5 * size, 5, 0.1), middle, 0.0, 0.0)
    )
    return items


# ----------------------------------------------------------------------
# animals, drawn facing right in local pixels and then placed


def dot(hand: Hand, place: Place, x, y, r, colour=BLACK) -> Part:
    cx, cy = place([(x, y)])[0]
    r *= abs(place.scale[1])
    return Part(hand.blob(cx, cy, r, r, 5, 0.05), colour, 0.0, 0.0)


def shape(
    hand: Hand, place: Place, pts, colour, angle=30.0, line=OUTLINE, closed=True
) -> Part:
    pts = place(pts)
    curve = hand.closed(pts, 1.0, 0.8) if closed else hand.poly(pts, 1.0, 0.8)
    return Part(curve, colour, angle, line)


def oval(cx, cy, rx, ry, n=10, turn=0.0) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    x, y = rx * np.cos(a), ry * np.sin(a)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def leg(x, top, bottom, w) -> list[tuple[float, float]]:
    return [(x - w, top), (x + w, top), (x + w * 0.9, bottom), (x - w * 0.9, bottom)]


def cow(hand: Hand, place: Place) -> list[Item]:
    items: list[Item] = []
    for x, dx in ((-80, -2), (-54, 3), (56, -3), (84, 2)):
        items.append(
            shape(hand, place, leg(x + dx, 20, 108, 11), None, 90.0, 9.0, False)
        )
        items.append(
            shape(hand, place, leg(x + dx, 96, 110, 10.5), BLACK, 0.0, 6.0, False)
        )
    items.append(shape(hand, place, oval(36, 50, 22, 15, 8), PINK, 0.0, 9.0))
    tail = place([(-100, -40), (-122, -22), (-130, 10), (-126, 34)])
    items.append(Mark(hand.line(tail, 0.8, 0.6), BLACK, 8.0))
    items.append(shape(hand, place, oval(-126, 42, 8, 12, 7, 10), BLACK, 90.0, 6.0))
    body = [
        (-104, -40),
        (-92, -52),
        (40, -54),
        (96, -50),
        (106, -30),
        (104, 38),
        (90, 50),
        (-90, 48),
        (-104, 34),
    ]
    body_part = shape(hand, place, body, None, 0.0, OUTLINE)
    items.append(body_part)
    for sx, sy, rx, ry in (
        (-55, -12, 28, 21),
        (25, -30, 22, 15),
        (58, 18, 22, 17),
        (-88, 26, 13, 14),
        (-12, 30, 15, 11),
    ):
        spot = shape(
            hand, place, oval(sx, sy, rx, ry, 8, hand.u(-30, 30)), BLACK, 30.0, 0.0
        )
        spot.clip = body_part
        items.append(spot)
    items += cow_head(hand, place)
    return items


def cow_head(hand: Hand, place: Place) -> list[Item]:
    items: list[Item] = []
    items.append(
        shape(
            hand, place, [(104, -58), (86, -84), (76, -76), (92, -52)], None, 0.0, 8.0
        )
    )
    for hx in (116, 138):
        horn = [
            (hx - 5, -76),
            (hx - 9, -94),
            (hx - 2, -104),
            (hx + 4, -92),
            (hx + 5, -76),
        ]
        items.append(shape(hand, place, horn, CREAM, 90.0, 7.0))
    items.append(shape(hand, place, oval(128, -52, 33, 30, 9, 10), None, 0.0))
    items.append(shape(hand, place, oval(151, -31, 24, 18, 9, -10), PINK, 0.0, 10.0))
    items.append(dot(hand, place, 122, -62, 5.5))
    items.append(dot(hand, place, 146, -31, 3.5))
    items.append(dot(hand, place, 160, -29, 3.5))
    return items


def pig(hand: Hand, place: Place) -> list[Item]:
    items: list[Item] = []
    for x, dx in ((-55, -2), (-30, 2), (34, -2), (58, 2)):
        items.append(
            shape(hand, place, leg(x + dx, 25, 64, 10), PINK, 90.0, 9.0, False)
        )
    t = np.linspace(0, 3.2 * np.pi, 14)
    curl = np.stack(
        [-92 - (4 + 2.2 * t) * np.cos(t), -18 - (4 + 2.2 * t) * np.sin(t)], 1
    )
    items.append(Mark(hand.line(place(curl), 0.5, 0.4), ROSE, 8.0))
    body = [
        (-88, 0),
        (-70, -42),
        (-10, -58),
        (55, -52),
        (92, -24),
        (96, 14),
        (60, 50),
        (-10, 56),
        (-72, 40),
    ]
    items.append(shape(hand, place, body, PINK, 20.0))
    items.append(
        shape(hand, place, [(48, -42), (72, -80), (84, -40)], PINK, 70.0, 9.0, False)
    )
    items.append(shape(hand, place, oval(104, -2, 13, 19, 8), ROSE, 90.0, 9.0))
    items.append(dot(hand, place, 101, -8, 3.0))
    items.append(dot(hand, place, 102, 6, 3.0))
    items.append(dot(hand, place, 74, -22, 5.5))
    return items


def bird_legs(hand: Hand, place: Place, xs, top, bottom) -> list[Item]:
    items: list[Item] = []
    for x in xs:
        items.append(
            Mark(hand.line(place([(x, top), (x - 2, bottom)]), 0.5, 0.3), ORANGE, 8.0)
        )
        foot = place([(x - 14, bottom + 2), (x - 2, bottom), (x + 12, bottom + 2)])
        items.append(Mark(hand.line(foot, 0.5, 0.3), ORANGE, 7.0))
    return items


def tail_feather(k: int) -> np.ndarray:
    """The k-th of a rooster's tail feathers: long ovals fanning up and back
    from its rump."""
    a = math.radians(96 + 28 * k)
    d = np.array([math.cos(a), -math.sin(a)])
    length = 46 - 4 * k
    centre = np.array([-40.0, -24.0]) + d * length
    return oval(centre[0], centre[1], length, 13, 10, -math.degrees(a))


HEN_BODY = [
    (42, -26),
    (50, -4),
    (40, 18),
    (16, 32),
    (-18, 32),
    (-42, 18),
    (-54, -4),
    (-58, -26),
    (-60, -46),
    (-46, -42),
    (-32, -28),
    (-12, -22),
    (8, -24),
    (24, -34),
]


def hen(hand: Hand, place: Place, colour=None, rooster=False) -> list[Item]:
    items: list[Item] = bird_legs(hand, place, (-8, 12), 28, 54)
    if rooster:
        for k, col in enumerate((LEAF, NAVY, RED)):
            items.append(shape(hand, place, tail_feather(k), col, 100.0 + 26 * k, 8.0))
    items.append(shape(hand, place, HEN_BODY, colour, 30.0, 9.0))
    wing = [(-24, -2), (-4, -6), (12, 2), (-6, 12)]
    items.append(shape(hand, place, wing, colour, 20.0, 7.0))
    comb = [(24, -50), (26, -70), (36, -62), (42, -76), (50, -62), (60, -68), (56, -46)]
    items.append(shape(hand, place, comb, RED, 0.0, 7.0))
    items.append(
        shape(hand, place, [(54, -44), (72, -39), (54, -33)], SUN, 0.0, 6.0, False)
    )
    items.append(shape(hand, place, oval(52, -24, 5, 8, 6), RED, 90.0, 6.0))
    items.append(shape(hand, place, oval(40, -40, 18, 17, 8), colour, 0.0, 9.0))
    items.append(dot(hand, place, 45, -44, 4.0))
    return items


def sheep(hand: Hand, place: Place) -> list[Item]:
    items: list[Item] = []
    for x, dx in ((-42, 0), (-20, 3), (24, -2), (46, 2)):
        items.append(
            shape(hand, place, leg(x + dx, 20, 60, 6.5), BLACK, 90.0, 6.0, False)
        )
    cx, cy = place([(0, 0)])[0]
    scale = abs(place.scale[0])
    items.append(
        Part(
            hand.scallop(cx, cy, 74 * scale, 48 * scale, 9, 0.14, 0.75), None, 0.0, 10.0
        )
    )
    items.append(shape(hand, place, [(56, -28), (40, -42), (46, -24)], BLACK, 0.0, 7.0))
    items.append(shape(hand, place, oval(76, -10, 22, 29, 9, -25), BLACK, 60.0, 8.0))
    items.append(dot(hand, place, 80, -17, 4.5, None))
    return items


DUCK_BODY = [
    (-58, -30),
    (-46, -8),
    (-28, 8),
    (4, 14),
    (34, 10),
    (48, -2),
    (44, -16),
    (26, -22),
    (-4, -16),
    (-34, -18),
]


def duck(hand: Hand, place: Place, colour) -> list[Item]:
    items: list[Item] = []
    items.append(shape(hand, place, DUCK_BODY, colour, 10.0, 9.0))
    items.append(
        shape(hand, place, [(-22, -6), (-6, -8), (8, -2), (-8, 3)], colour, 10.0, 6.0)
    )
    items.append(
        shape(
            hand,
            place,
            [(42, -42), (66, -38), (64, -31), (42, -30)],
            ORANGE,
            0.0,
            7.0,
            False,
        )
    )
    items.append(shape(hand, place, oval(30, -38, 18, 17, 8), colour, 0.0, 9.0))
    items.append(dot(hand, place, 35, -42, 3.8))
    return items


def cat(hand: Hand, place: Place) -> list[Item]:
    items: list[Item] = []
    tail = place([(20, -6), (40, -14), (46, -38), (36, -60), (42, -74)])
    items.append(Mark(hand.line(tail, 0.6, 0.5), BLACK, 11.0))
    items.append(
        shape(
            hand,
            place,
            [(-26, 0), (26, 0), (22, -40), (10, -64), (-12, -64), (-24, -40)],
            BLACK,
            90.0,
            8.0,
        )
    )
    items.append(shape(hand, place, oval(0, -80, 22, 20, 9), BLACK, 0.0, 8.0))
    for side in (-1, 1):
        ear = [(side * 6, -94), (side * 20, -112), (side * 21, -86)]
        items.append(shape(hand, place, ear, BLACK, 0.0, 6.0, False))
        items.append(dot(hand, place, side * 8, -82, 3.5, SUN))
    return items


# ----------------------------------------------------------------------
# the picture


def flowers(hand: Hand) -> list[Item]:
    avoid = [
        (60, 760, 410, 990),
        (380, 890, 530, 1050),
        (460, 790, 860, 1010),
        (890, 820, 1110, 990),
        (1090, 820, 1400, 1050),
    ]
    items: list[Item] = []
    colours = (RED, SUN, None, RED, SUN, None, ORANGE)
    for k, (x, y) in enumerate(scatter(hand, 16, 15, 790, 1385, 1030, 105, avoid)):
        items += flower(hand, x, y, colours[k % len(colours)])
    return items


def foreground(hand: Hand) -> list[Item]:
    items: list[Item] = flowers(hand)
    items += pond(hand)
    items += cow(hand, Place(228, 866, 1.0))
    items += hen(hand, Place(436, 985, 0.95, left=True), None)
    items += pig(hand, Place(1012, 918, 0.9, left=True))
    items += hen(hand, Place(1196, 922, 1.2, left=True), ORANGE, rooster=True)
    items += hen(hand, Place(1338, 982, 1.0, left=True), None)
    return items


def pond(hand: Hand, cx=660.0, cy=905.0) -> list[Item]:
    items: list[Item] = [Part(hand.blob(cx, cy, 178, 96, 11, 0.04), POND, 5.0)]
    for x, y, w in ((-110, 52, 40), (40, 62, 46), (120, -52, 34), (-120, -40, 30)):
        ripple = [
            (cx + x - w / 2, cy + y),
            (cx + x, cy + y - 6),
            (cx + x + w / 2, cy + y),
        ]
        items.append(Mark(hand.line(ripple, 0.5, 0.4), None, 7.0))
    for x, n in ((cx - 196, 3), (cx + 166, 2)):
        for k in range(n):
            reed = [
                (x + 11 * k, cy + 6),
                (x + 11 * k + hand.u(-6, 6), cy - 40 - hand.u(0, 30)),
            ]
            items.append(Mark(hand.line(reed, 0.5, 0.5), LEAF, 8.0))
    items += duck(hand, Place(cx - 34, cy + 12, 1.25, left=True), None)
    items += duck(hand, Place(cx + 112, cy - 14, 0.9, left=True), SUN)
    return items


def middle(hand: Hand) -> list[Item]:
    items: list[Item] = []
    items += sheep(hand, Place(112, 606, 0.8, left=True, turn=3))
    items += sheep(hand, Place(252, 572, 0.78, left=True, turn=-2))
    items += apple_tree(hand, 690, 660, 64, 66)
    items += house(hand)
    items += barn(hand)
    return items


def far_trees(hand: Hand, height) -> list[Item]:
    items: list[Item] = []
    for x, h, r, colour in (
        (556, 26, 19, LEAF),
        (598, 32, 23, GREEN),
        (640, 24, 17, LEAF),
        (1336, 28, 20, GREEN),
    ):
        items += lollipop(hand, x, float(height(x)) + 8, h, r, colour)
    return items


def scene() -> list[Item]:
    """Every shape of the picture, from the back to the front."""
    hand = Hand(SEED)
    items: list[Item] = []
    items += sky(hand)
    items += sun(hand)
    items += clouds(hand)
    items += birds(hand)
    far, far_top = far_hills(hand)
    items += far
    items += windmill(hand, 318.0, float(far_top(318.0)) + 20)
    items += far_trees(hand, far_top)
    items += middle_hill(hand)
    items += middle(hand)
    front, height = meadow(hand)
    items += fence(hand, lambda x: float(height(x)) - 4)
    for x, h, r in ((1160, 168, 40), (1232, 206, 44), (1302, 160, 38), (1370, 196, 42)):
        items += sunflower(
            hand, x + hand.u(-6, 6), float(height(x)) + 8, h + hand.u(-6, 6), r
        )
    items += front
    items += foreground(hand)
    return items


# ----------------------------------------------------------------------
# the flat study


def fill_mask(poly: np.ndarray) -> np.ndarray:
    """The pixels inside the closed polygon ``poly`` (even-odd), (H, W)."""
    mask = np.zeros((HEIGHT, WIDTH), bool)
    y0 = max(int(np.floor(poly[:, 1].min())), 0)
    y1 = min(int(np.ceil(poly[:, 1].max())) + 1, HEIGHT)
    if y1 <= y0:
        return mask
    rows = np.arange(y0, y1) + 0.5
    p, q = poly, np.roll(poly, -1, axis=0)
    lo, hi = np.minimum(p[:, 1], q[:, 1]), np.maximum(p[:, 1], q[:, 1])
    r, e = np.nonzero((rows[:, None] >= lo) & (rows[:, None] < hi))
    t = (rows[r] - p[e, 1]) / (q[e, 1] - p[e, 1])
    xc = p[e, 0] + t * (q[e, 0] - p[e, 0])
    xi = np.clip(np.ceil(xc - 0.5).astype(int), 0, WIDTH)
    toggles = np.zeros((y1 - y0, WIDTH + 1), np.int32)
    np.add.at(toggles, (r, xi), 1)
    mask[y0:y1] = (np.cumsum(toggles, axis=1)[:, :WIDTH] % 2) == 1
    return mask


_DISCS: dict[int, np.ndarray] = {}


def disc(r: int) -> np.ndarray:
    if r not in _DISCS:
        y, x = np.mgrid[-r : r + 1, -r : r + 1]
        _DISCS[r] = x * x + y * y <= r * r + r * 0.6
    return _DISCS[r]


def line_mask(path: np.ndarray, width: float | np.ndarray) -> np.ndarray:
    """The pixels a line of ``width`` along ``path`` covers, (H, W)."""
    mask = np.zeros((HEIGHT, WIDTH), bool)
    widths = np.broadcast_to(np.asarray(width, float), (len(path),))
    for (x, y), w in zip(path, widths, strict=True):
        r = max(int(round(w / 2)), 1)
        xi, yi = int(round(x)), int(round(y))
        x0, x1, y0, y1 = xi - r, xi + r + 1, yi - r, yi + r + 1
        if x1 <= 0 or y1 <= 0 or x0 >= WIDTH or y0 >= HEIGHT:
            continue
        k = disc(r)[
            max(-y0, 0) : k_end(y1, HEIGHT, 2 * r + 1),
            max(-x0, 0) : k_end(x1, WIDTH, 2 * r + 1),
        ]
        mask[max(y0, 0) : min(y1, HEIGHT), max(x0, 0) : min(x1, WIDTH)] |= k
    return mask


def k_end(end: int, size: int, n: int) -> int:
    return n - max(end - size, 0)


def outline_path(hand: Hand, ring: np.ndarray) -> list[np.ndarray]:
    """How a hand goes round a closed outline: from anywhere, a little past
    where it began, sometimes in two goes with a small gap."""
    ring = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    n = len(ring)
    start = int(hand.rng.integers(n))
    over = int(n * hand.u(0.02, 0.06)) + 2
    loop = np.roll(ring, -start, axis=0)
    loop = np.vstack([loop, loop[:over]])
    if n > 120 and hand.u(0, 1) < 0.35:
        cut = int(hand.u(0.3, 0.7) * n)
        gap = int(hand.u(2, 4))
        return [loop[:cut], loop[cut + gap :]]
    return [loop]


def paint(img: np.ndarray, mask: np.ndarray, colour: str | None) -> None:
    img[mask] = _rgb(PAPER_HEX if colour is None else colour)


def _rgb(hexc: str) -> np.ndarray:
    h = hexc.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def study(items: Sequence[Item]) -> np.ndarray:
    """The design as flat colour, cut paper laid back to front."""
    hand = Hand(SEED + 1)
    img = np.empty((HEIGHT, WIDTH, 3))
    img[:] = _rgb(PAPER_HEX)
    masks: dict[int, np.ndarray] = {}
    for item in items:
        if isinstance(item, Part):
            mask = fill_mask(item.outline)
            if item.clip is not None:
                mask &= masks[id(item.clip)]
            masks[id(item)] = mask
            paint(img, mask, item.colour)
            if item.line > 0:
                for path in outline_path(hand, item.outline):
                    w = item.line * (
                        0.85 + 0.3 * noise.value_noise_1d(len(path), 30.0, hand.rng)
                    )
                    paint(img, line_mask(path, w), item.ink)
        else:
            mask = line_mask(item.path, item.width)
            if item.clip is not None:
                mask &= masks[id(item.clip)]
            paint(img, mask, item.colour)
    return img


def save(img: np.ndarray, path: Path) -> None:
    write_png(str(path), np.round(np.clip(img, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--study":
        folder = Path(sys.argv[2])
        folder.mkdir(parents=True, exist_ok=True)
        img = study(scene())
        save(img, folder / "study_flat.png")
        for name, (x0, y0, x1, y1) in CROPS.items():
            crop = np.repeat(np.repeat(img[y0:y1, x0:x1], 2, 0), 2, 1)
            save(crop, folder / f"study_{name}.png")
        print(f"wrote the study to {folder}")
        return
    raise SystemExit("stage 2 (the crayon) is not written yet: use --study FOLDER")


CROPS = {
    "barn": (620, 300, 1180, 720),
    "house": (0, 160, 620, 760),
    "animals": (0, 720, 560, 1050),
    "pond": (460, 720, 1400, 1050),
}


if __name__ == "__main__":
    main()
