"""A joyful farm on a hill, in wax crayon: naive art in bold colours.

Run from the repository root: ``python examples/farm.py``. Writes
``examples/output/farm.png`` (in under a minute); ``python examples/farm.py
FILE`` writes it to FILE instead, and ``python examples/farm.py --study
FOLDER`` writes the flat design study (cut-paper shapes in flat colour, with
their outlines) and enlarged crops of it to FOLDER.

A red barn and a little white house with a black cat on its roof sit on a
round green hill under a big sun; behind them a farther hill is a patchwork
of fields (stripes, furrows, cabbages, wheat), with a windmill. Sheep graze
on the hill, sunflowers grow by the barn, an apple tree stands between. In
front, a meadow with flowers, a round blue pond with two ducks, a spotted
cow, a pig, two hens and a rooster.

The picture is first designed as cut paper: every element a closed shape
with a wobbling outline, a flat colour and a place in the stack, nearer
below and farther above, as naive art stacks things. The crayon then works
out what each shape shows once those in front are cut away, and lays it as
wax crayon must be laid, since a crayon does not cover what is under it:
the black outlines first, on bare paper, in pieces of a hand's reach, some
traced twice; then the patterns (furrows, stripes, dots, dashes) that a
field's colour goes over; then every shape filled with back-and-forth
strokes in its own direction, up to its lines, the light colours first, a
second looser pass across where it should be solid, big shapes patch by
patch. White is the paper, kept: clouds, walls, the cow, the sheep. A ragged
margin of bare paper is left all round.
"""

# cspell:words catmull interp

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
from murex import CatmullRom, filters, noise, write_png

from fusain import Sheet, crayons, papers

HERE = Path(__file__).parent
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
    under: bool = False  # a pattern that its shape's colour goes over


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
    under: bool = False  # a pattern that the colour behind goes over
    passes: float = 2.0  # passes of colour (a fraction: in some patches)
    flow: Callable | None = None  # the direction of the strokes at x
    drift: float = 10.0  # how far that wanders from patch to patch
    tint: str | None = None  # a lighter colour laid first in some patches
    radial: bool = False  # filled with strokes radiating from its middle
    weight: float = 1.0  # how firmly it is filled, against its colour's way


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

    def scallop(
        self, cx, cy, rx, ry, lobes=7, depth=0.16, flat=0.5, lean=0.0
    ) -> np.ndarray:
        """A cloud's outline: round lobes all round, flatter underneath, its
        top pushed up on one side by ``lean``."""
        n = lobes * 8
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        phase = self.u(0, np.pi)
        bump = np.abs(np.sin(lobes * a / 2 + phase)) ** 0.7
        size = (
            1.0 + self.rng.normal(0, 0.04, lobes)[(a / (2 * np.pi) * lobes).astype(int)]
        )
        r = (1.0 - depth + depth * bump) * size
        y = np.sin(a)
        y = np.where(y > 0, y * flat, y * (1.0 - lean * np.cos(a)))
        return self.closed(np.stack([cx + rx * r * np.cos(a), cy + ry * r * y], 1), 0.6)


# ----------------------------------------------------------------------
# the sky


def sky(hand: Hand) -> list[Item]:
    box = hand.corners([(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)], 200)
    return [Part(box, SKY, 30.0, 0.0, name="sky", passes=1.6, drift=20.0)]


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
        items.append(Part(hand.poly(pts, 1.0, 0.6), ORANGE, math.degrees(a), 6.5))
    disc = Part(hand.blob(cx, cy, r, r * 0.97, 11, 0.025), SUN, 60.0, 12.0)
    disc.radial = True
    items.append(disc)
    return items


def clouds(hand: Hand) -> list[Item]:
    items: list[Item] = []
    for cx, cy, rx, ry, lobes, depth, flat, lean in (
        (175, 104, 128, 44, 10, 0.12, 0.3, 0.25),
        (640, 168, 108, 72, 6, 0.2, 0.55, -0.3),
        (912, 84, 76, 40, 4, 0.24, 0.45, 0.4),
    ):
        outline = hand.scallop(cx, cy, rx, ry, lobes, depth, flat, lean)
        items.append(Part(outline, None, line=10.0, name="cloud"))
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


def slope(height) -> Callable:
    """The direction of the hill's top line at x, in degrees."""

    def at(x):
        return -math.degrees(math.atan((height(x + 30) - height(x - 30)) / 60.0))

    return at


def fields(hand: Hand, height, cuts, bottom, hill: Part, looks):
    """The fields of a hill, side by side between its dividers, and the
    dividers as lines. Each is filled round the hill's curve (and ``looks``
    turns that by an angle)."""
    lines = dividers(hand, height, cuts, bottom)
    patches = []
    for left, right, (colour, angle) in zip(lines[:-1], lines[1:], looks, strict=True):
        poly = np.vstack([left[::-1], right])
        field_ = Part(poly, colour, angle, 0.0, hill, "field", passes=1.25)
        field_.flow = slope(height)
        patches.append(field_)
    edges = [Mark(line, BLACK, 9.0, hill) for line in lines[1:-1]]
    return patches, edges


def contour_stripes(hand, height, x0, x1, start, gap, count, colour, width, clip):
    """Furrows: lines that follow the hill's curve, ``gap`` apart."""
    items: list[Item] = []
    xs = np.linspace(x0, x1, max(int((x1 - x0) / 60), 3))
    for k in range(count):
        d = start + k * gap * hand.u(0.9, 1.1)
        pts = np.stack([xs, height(xs) + d + hand.rng.normal(0, 2.0, len(xs))], 1)
        items.append(Mark(hand.line(pts, 1.0, 1.0), colour, width, clip, True))
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
        items.append(Mark(hand.line(pts, 1.5, 1.0), colour, width, clip, True))
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
            dot_ = Part(hand.blob(x, y, rr, rr * 0.85, 6, 0.1), colour, 0.0, 0.0, clip)
            dot_.under = True
            items.append(dot_)
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
        (GREEN, 0.0),
        (SUN, 65.0),
        (LIME, 0.0),
        (ORANGE, 0.0),
        (LIME, 15.0),
        (GREEN, -20.0),
    ]
    patches, edges = fields(hand, height, FAR_CUTS, 800, hill, looks)
    items: list[Item] = [hill, *patches]
    for k in (0, 2, 4, 5):  # (the green fields, whose patterns are dark)
        patches[k].weight = 0.82
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
        hand, (xs[5] - 60, 330, xs[6] + 80, 760), -60, 40, LEAF, 9.0, p[5]
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
                    True,
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
    looks = [(LIME, 10.0), (GREEN, -5.0)]
    patches, edges = fields(hand, height, cuts, 900, hill, looks)
    patches[1].tint = LIME
    return [hill, *patches, *edges, Mark(top, BLACK, OUTLINE)]


def meadow(hand: Hand):
    """The meadow in front, and the height of its top edge at x."""
    ctrl = [(-40, 768), (250, 744), (560, 754), (860, 772), (1150, 748), (1440, 738)]
    top, height = silhouette(hand, ctrl)
    part = Part(band(top, HEIGHT + 40), GREEN, 0.0, 0.0, name="meadow", passes=1.35)
    part.tint = LIME
    part.flow = slope(height)
    return [part, Mark(top, BLACK, OUTLINE)], height


# ----------------------------------------------------------------------
# buildings


def windmill(hand: Hand, x=318.0, base=372.0) -> list[Item]:
    items: list[Item] = []
    tower = [(x - 34, base), (x + 34, base), (x + 21, base - 108)]
    tower += [(x + 10, base - 124), (x - 10, base - 124), (x - 21, base - 108)]
    items.append(Part(hand.poly(tower), None, 90.0, 10.0))
    door = [(x - 12, base), (x + 12, base), (x + 12, base - 30), (x, base - 38)]
    items.append(
        Part(hand.poly([*door, (x - 12, base - 30)], 1.0, 0.5), NAVY, 90.0, 8.0)
    )
    hub = np.array([x + 2, base - 112.0])
    start = hand.u(43, 47)
    for k in range(4):
        a = math.radians(start + 90 * k + hand.u(-3, 3))
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        r1, w = 88.0 + hand.u(-6, 6), 9.0 + hand.u(-1, 1)
        pts = [hub + d * 12 - n * w, hub + d * r1 - n * w, hub + d * r1 + n * w]
        items.append(
            Part(hand.poly([*pts, hub + d * 12 + n * w], 1.0, 0.6), None, 0.0, 8.0)
        )
    items.append(Part(hand.blob(hub[0], hub[1], 8, 8, 6, 0.05), BLACK, 0.0, 0.0))
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
        items.append(Part(hand.poly(win, 1.0, 0.6), SUN, 45.0, 8.0))
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
    for k, r in enumerate((13.0, 18.0, 24.0)):
        cx, cy = x + 20 * k + 7 * k * k, y - 36 * k - 5 * k * k
        items.append(
            Part(
                hand.scallop(cx, cy, r * 1.25, r, 5, 0.18, 0.9),
                None,
                0.0,
                8.0,
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
    items.extend(
        (
            Part(hand.poly(roof, 1.0, 0.8), NAVY, 30.0),
            Part(
                hand.poly(at([(-135, 0), (135, 0), *gable[::-1]]), 1.5, 1.0), RED, 80.0
            ),
        )
    )
    (x0, y0), (x1, y1) = at([(-66, -150), (66, 0)])
    items += cross_door(hand, x0, y0, x1, y1)
    loft = at([(-36, -258), (36, -258), (36, -196), (-36, -196)])
    items.append(Part(hand.poly(loft, 1.0, 0.6), None, 0.0, 9.0))
    hay = at([(-24, -246), (24, -246), (26, -207), (-25, -207)])
    items.append(Part(hand.poly(hay, 1.0, 0.6), SUN, 70.0, 0.0))
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


def fence(hand: Hand, height_at, x0=48.0, x1=560.0) -> list[Item]:
    items: list[Item] = []
    xs = [x0]
    while xs[-1] < x1 - 40:
        xs.append(xs[-1] + hand.u(70, 84))
    for a, b in pairwise(xs):
        for dy in (22, 48):
            ya, yb = height_at(a) - dy + hand.u(-2, 2), height_at(b) - dy + hand.u(
                -2, 2
            )
            rail = [(a, ya - 6), (b, yb - 6), (b, yb + 6), (a, ya + 6)]
            items.append(Part(hand.poly(rail, 0.8, 0.5), None, 0.0, 7.0))
    for x in xs:
        y = height_at(x) + 8
        h = hand.u(64, 76)
        lean = hand.u(-3, 3)
        w = hand.u(9, 11)
        post = [
            (x - w, y),
            (x + w, y),
            (x + w + lean, y - h),
            (x + lean, y - h - 14),
            (x - w + lean, y - h),
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
        rr = hand.u(10, 12)
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
            (sx + side * 26 * size, sy - 26 * size),
            (sx + side * 54 * size, sy - 18 * size),
            (sx + side * 30 * size, sy + 4),
        ]
        items.append(Part(hand.closed(leaf, 1.0, 0.6), LEAF, side * 30.0, 6.0))
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
        Part(hand.blob(head[0], head[1], r * 0.4, r * 0.4, 7, 0.05), BROWN, 45.0, 7.0)
    )
    return items


def flower(hand: Hand, x, y, colour) -> list[Item]:
    """A meadow flower: a coloured one is a loop of petals drawn in one go
    round a dab of yellow; a white one is petals of bare paper."""
    n = int(hand.u(5, 7))
    start = hand.u(0, 2 * np.pi)
    size = hand.u(0.85, 1.15)
    middle = SUN if colour != SUN else ORANGE
    items: list[Item] = []
    if colour is None:
        for k in range(n):
            a = start + 2 * np.pi * k / n + hand.u(-0.15, 0.15)
            px, py = x + 12 * size * math.cos(a), y + 12 * size * math.sin(a)
            petal = hand.blob(px, py, 8 * size, 6.5 * size, 5, 0.1, math.degrees(a))
            items.append(Part(petal, None, 0.0, 0.0))
    else:
        a = start + np.linspace(0, 2 * np.pi * 1.04, 12 * n)
        r = 15 * size * (0.45 + 0.55 * np.abs(np.cos(n * (a - start) / 2)))
        loop = np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1)
        items.append(Mark(hand.line(loop, 0.4, 0.4), colour, 7.0))
    items.append(Part(hand.blob(x, y, 6 * size, 6 * size, 5, 0.1), middle, 0.0, 0.0))
    return items


# ----------------------------------------------------------------------
# animals, drawn facing right in local pixels and then placed


def dot(hand: Hand, place: Place, x, y, r, colour: str | None = BLACK) -> Part:
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
        items.extend(
            (
                shape(hand, place, leg(x + dx, 20, 108, 11), None, 90.0, 9.0, False),
                shape(hand, place, leg(x + dx, 96, 110, 10.5), BLACK, 0.0, 6.0, False),
            )
        )
    items.append(shape(hand, place, oval(36, 50, 22, 15, 8), PINK, 0.0, 9.0))
    tail = place([(-100, -40), (-122, -22), (-130, 10), (-126, 34)])
    items.extend(
        (
            Mark(hand.line(tail, 0.8, 0.6), BLACK, 8.0),
            shape(hand, place, oval(-126, 42, 8, 12, 7, 10), BLACK, 90.0, 6.0),
        )
    )
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
        items.append(shape(hand, place, horn, CREAM, 90.0, 6.0))
    items.extend(
        (
            shape(hand, place, oval(128, -52, 33, 30, 9, 10), None, 0.0),
            shape(hand, place, oval(151, -31, 24, 18, 9, -10), PINK, 0.0, 8.0),
            dot(hand, place, 122, -62, 5.5),
            dot(hand, place, 146, -31, 3.5),
            dot(hand, place, 160, -29, 3.5),
        )
    )
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
    items.extend(
        (
            shape(hand, place, body, PINK, 20.0),
            shape(
                hand, place, [(48, -42), (72, -80), (84, -40)], PINK, 70.0, 9.0, False
            ),
            shape(hand, place, oval(104, -2, 13, 19, 8), ROSE, 90.0, 7.0),
            dot(hand, place, 101, -8, 3.0),
            dot(hand, place, 102, 6, 3.0),
            dot(hand, place, 74, -22, 5.5),
        )
    )
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
    items.extend(
        (
            shape(hand, place, comb, RED, 0.0, 7.0),
            shape(hand, place, [(54, -44), (72, -39), (54, -33)], SUN, 0.0, 6.0, False),
            shape(hand, place, oval(52, -24, 5, 8, 6), RED, 90.0, 6.0),
            shape(hand, place, oval(40, -40, 18, 17, 8), colour, 0.0, 9.0),
            dot(hand, place, 45, -44, 2.8),
        )
    )
    return items


def sheep(hand: Hand, place: Place) -> list[Item]:
    items: list[Item] = []
    for x, dx in ((-42, 0), (-20, 3), (24, -2), (46, 2)):
        items.append(
            shape(hand, place, leg(x + dx, 20, 60, 6.5), BLACK, 90.0, 6.0, False)
        )
    cx, cy = place([(0, 0)])[0]
    scale = abs(place.scale[0])
    items.extend(
        (
            Part(
                hand.scallop(cx, cy, 74 * scale, 48 * scale, 9, 0.14, 0.75),
                None,
                0.0,
                10.0,
            ),
            shape(hand, place, [(84, -30), (64, -46), (74, -22)], BLACK, 0.0, 6.0),
            shape(hand, place, oval(102, -8, 19, 26, 9, -25), BLACK, 60.0, 6.0),
            dot(hand, place, 106, -16, 5.0, None),
        )
    )
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
    wing = place([(-26, -10), (-10, -2), (10, -4)])
    items.extend(
        (
            Mark(hand.line(wing, 0.5, 0.4), BLACK, 6.0),
            shape(
                hand,
                place,
                [(42, -42), (66, -38), (64, -31), (42, -30)],
                ORANGE,
                0.0,
                7.0,
                False,
            ),
            shape(hand, place, oval(30, -38, 18, 17, 8), colour, 0.0, 9.0),
            dot(hand, place, 35, -42, 3.8),
        )
    )
    return items


def cat(hand: Hand, place: Place) -> list[Item]:
    items: list[Item] = []
    tail = place([(20, -6), (40, -14), (46, -38), (36, -60), (42, -74)])
    items.extend(
        (
            Mark(hand.line(tail, 0.6, 0.5), BLACK, 11.0),
            shape(
                hand,
                place,
                [(-26, 0), (26, 0), (22, -40), (10, -64), (-12, -64), (-24, -40)],
                BLACK,
                90.0,
                8.0,
            ),
            shape(hand, place, oval(0, -80, 22, 20, 9), BLACK, 0.0, 8.0),
        )
    )
    for side in (-1, 1):
        ear = [(side * 6, -94), (side * 20, -112), (side * 21, -86)]
        items.extend(
            (
                shape(hand, place, ear, BLACK, 0.0, 6.0, False),
                dot(hand, place, side * 9, -82, 4.5, SUN),
            )
        )
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
    for k, (x, y) in enumerate(scatter(hand, 15, 50, 790, 1350, 1012, 105, avoid)):
        items += flower(hand, x, y, colours[k % len(colours)])
    return items


def foreground(hand: Hand) -> list[Item]:
    items: list[Item] = flowers(hand)
    items += pond(hand)
    items += cow(hand, Place(228, 866, 1.0))
    items += hen(hand, Place(436, 985, 0.95, left=True), None)
    items += pig(hand, Place(1000, 918, 0.9, left=True))
    items += hen(hand, Place(1190, 922, 1.2, left=True), ORANGE, rooster=True)
    items += hen(hand, Place(1320, 996, 0.85, left=True), None)
    return items


def pond(hand: Hand, cx=660.0, cy=905.0) -> list[Item]:
    items: list[Item] = [Part(hand.blob(cx, cy, 178, 96, 11, 0.04), POND, 5.0)]
    for x, y, w in ((-110, 52, 40), (40, 62, 46), (120, -52, 34), (-120, -40, 30)):
        ripple = [
            (cx + x - w / 2, cy + y),
            (cx + x, cy + y - 6),
            (cx + x + w / 2, cy + y),
        ]
        items.append(Mark(hand.line(ripple, 0.5, 0.4), None, 10.0))
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
    items += sheep(hand, Place(148, 624, 0.8, left=True, turn=3))
    items += sheep(hand, Place(292, 590, 0.78, left=True, turn=-2))
    items += apple_tree(hand, 690, 660, 64, 66)
    items += house(hand)
    items += barn(hand)
    return items


def far_trees(hand: Hand, height) -> list[Item]:
    items: list[Item] = []
    for x, h, r, colour in (
        (668, 26, 19, LEAF),
        (708, 32, 23, GREEN),
        (96, 28, 20, LEAF),
        (1322, 28, 20, GREEN),
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
    for x, h, r in ((1158, 172, 40), (1232, 206, 44), (1300, 164, 40)):
        items += sunflower(
            hand, x + hand.u(-6, 6), float(height(x)) + 8, h + hand.u(-6, 6), r
        )
    items += front
    items += foreground(hand)
    return items


# ----------------------------------------------------------------------
# the flat study


_DISCS: dict[int, np.ndarray] = {}


def disc(r: int) -> np.ndarray:
    if r not in _DISCS:
        y, x = np.mgrid[-r : r + 1, -r : r + 1]
        _DISCS[r] = x * x + y * y <= r * r + r * 0.6
    return _DISCS[r]


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


def _rgb(code: str) -> np.ndarray:
    h = code.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def study(items: Sequence[Item]) -> np.ndarray:
    """The design as flat colour, cut paper laid back to front."""
    hand = Hand(SEED + 1)
    img = np.empty((HEIGHT, WIDTH, 3))
    img[:] = _rgb(PAPER_HEX)
    own = own_regions(items)
    for item in items:
        reg = own[id(item)]
        if reg.m.size:
            img[reg.box][reg.m] = _rgb(
                PAPER_HEX if item.colour is None else item.colour
            )
        if isinstance(item, Part) and item.line > 0:
            for path in outline_path(hand, item.outline):
                line = line_region(path, item.line * hand.u(0.9, 1.1))
                if line.m.size:
                    img[line.box][line.m] = _rgb(item.ink)
    return img


# ----------------------------------------------------------------------
# the crayon: where each shape shows


#: White drawing paper with a clear tooth, so that the grain sparkles.
PAPER = papers.CARTRIDGE.with_(grain=3.5, depth=1.1, color=PAPER_HEX, name="drawing")

#: The crayon taken for each colour of the design.
CRAYON = {
    SKY: "#2066c4",
    SUN: "#ffcf12",
    ORANGE: "#f57a12",
    RED: "#cc180f",
    PINK: "#f492ad",
    ROSE: "#e8517f",
    GREEN: "#27922c",
    LIME: "#a4cc1e",
    LEAF: "#0f6131",
    BROWN: "#7c3f1a",
    NAVY: "#1f45a0",
    POND: "#1659b8",
    BLACK: "#141212",
    SMOKE: "#7a7a7a",
    CREAM: "#efd98c",
}

#: The colours are filled in this order: the light ones first.
ORDER = [BLACK, SUN, CREAM, LIME, PINK, ORANGE, ROSE, GREEN, RED, SKY, POND, BROWN]
ORDER += [LEAF, NAVY, SMOKE]

#: A black line is drawn this much wider than the design's: its grainy
#: edges read thinner than a flat one.
BOLD = 1.15
FILL_WIDTH = 12.0  # the width of a crayon filling a shape, in pixels
FILL_GAP = 1.2  # how far apart its strokes are, in widths
REACH = 420.0  # a shape larger than this is filled patch by patch
MARGIN = 27.0  # the bare paper left round the picture, in pixels


@dataclass
class Region:
    """A boolean mask over the box of the sheet that starts at (y0, x0)."""

    y0: int
    x0: int
    m: np.ndarray

    @property
    def box(self) -> tuple[slice, slice]:
        h, w = self.m.shape
        return slice(self.y0, self.y0 + h), slice(self.x0, self.x0 + w)

    def crop(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """This mask over another box (False outside it)."""
        out = np.zeros((h, w), bool)
        ya, yb = max(y0, self.y0), min(y0 + h, self.y0 + self.m.shape[0])
        xa, xb = max(x0, self.x0), min(x0 + w, self.x0 + self.m.shape[1])
        if ya < yb and xa < xb:
            out[ya - y0 : yb - y0, xa - x0 : xb - x0] = self.m[
                ya - self.y0 : yb - self.y0, xa - self.x0 : xb - self.x0
            ]
        return out


EMPTY = Region(0, 0, np.zeros((0, 0), bool))


def poly_region(poly: np.ndarray) -> Region:
    """The pixels inside the closed polygon ``poly`` (even-odd)."""
    y0 = max(int(np.floor(poly[:, 1].min())), 0)
    y1 = min(int(np.ceil(poly[:, 1].max())) + 1, HEIGHT)
    x0 = max(int(np.floor(poly[:, 0].min())), 0)
    x1 = min(int(np.ceil(poly[:, 0].max())) + 1, WIDTH)
    if y1 <= y0 or x1 <= x0:
        return EMPTY
    rows = np.arange(y0, y1) + 0.5
    p, q = poly, np.roll(poly, -1, axis=0)
    lo, hi = np.minimum(p[:, 1], q[:, 1]), np.maximum(p[:, 1], q[:, 1])
    r, e = np.nonzero((rows[:, None] >= lo) & (rows[:, None] < hi))
    t = (rows[r] - p[e, 1]) / (q[e, 1] - p[e, 1])
    xc = p[e, 0] + t * (q[e, 0] - p[e, 0])
    xi = np.clip(np.ceil(xc - 0.5).astype(int) - x0, 0, x1 - x0)
    toggles = np.zeros((y1 - y0, x1 - x0 + 1), np.int32)
    np.add.at(toggles, (r, xi), 1)
    return Region(y0, x0, (np.cumsum(toggles, axis=1)[:, : x1 - x0] % 2) == 1)


def line_region(path: np.ndarray, width: float) -> Region:
    """The pixels a line ``width`` wide along ``path`` covers."""
    r = max(int(round(width / 2)), 1)
    pts = np.round(path).astype(int)
    ya, xa = pts[:, 1].min() - r, pts[:, 0].min() - r
    m = np.zeros((pts[:, 1].max() + r + 1 - ya, pts[:, 0].max() + r + 1 - xa), bool)
    k, n = disc(r), 2 * r + 1
    for x, y in pts - [xa, ya]:
        m[y - r : y - r + n, x - r : x - r + n] |= k
    y0, x0 = max(ya, 0), max(xa, 0)
    y1, x1 = min(ya + m.shape[0], HEIGHT), min(xa + m.shape[1], WIDTH)
    if y1 <= y0 or x1 <= x0:
        return EMPTY
    return Region(y0, x0, m[y0 - ya : y1 - ya, x0 - xa : x1 - xa])


def erode(m: np.ndarray, r: int) -> np.ndarray:
    """``m`` shrunk by about ``r`` pixels all round."""
    if r <= 0:
        return m
    padded = np.pad(m, r + 1)
    out = filters.max_filter(~padded, r) < 0.5
    return out[r + 1 : -r - 1, r + 1 : -r - 1]


def own_regions(items: Sequence[Item]) -> dict[int, Region]:
    """Where each item lies, inside the item it is clipped to."""
    own: dict[int, Region] = {}
    for item in items:
        if isinstance(item, Part):
            reg = poly_region(item.outline)
        else:
            reg = line_region(item.path, item.width)
        if item.clip is not None and reg.m.size:
            reg.m &= own[id(item.clip)].crop(reg.y0, reg.x0, *reg.m.shape)
        own[id(item)] = reg
    return own


def sheet_edge(hand: Hand, inset: float) -> np.ndarray:
    """The picture's ragged edge: the inside of a closed line wandering
    ``inset`` pixels (give or take a few) in from the sheet's edges."""
    pts = []
    for (ax, ay), (bx, by) in pairwise(
        [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT), (0, 0)]
    ):
        n = int(math.hypot(bx - ax, by - ay) / 6)
        t = np.linspace(0, 1, n, endpoint=False)
        side = np.array([by - ay, ax - bx]) / math.hypot(bx - ax, by - ay)
        d = inset + 16 * (noise.fractal_noise_2d(1, n, 40.0, hand.rng, 3)[0] - 0.5)
        line = np.stack([ax + (bx - ax) * t, ay + (by - ay) * t], 1)
        pts.append(line - side * d[:, None])
    inside = np.zeros((HEIGHT, WIDTH), bool)
    reg = poly_region(np.vstack(pts))
    inside[reg.box] = reg.m
    return inside


# ----------------------------------------------------------------------
# the crayon: the plan of strokes


@dataclass
class Line:
    """One stroke of crayon along ``path`` (control points)."""

    path: np.ndarray
    colour: str
    width: float
    pressure: list[float]
    under: bool = False


@dataclass
class Plan:
    """What the crayon will do: lines first, then fills."""

    ink: list[Line] = field(default_factory=list)  # black, first
    lines: list[Line] = field(default_factory=list)  # coloured lines, next
    fills: list[tuple[Part, Region]] = field(default_factory=list)


def runs(keep: np.ndarray) -> list[tuple[int, int]]:
    """The stretches (start, end) where ``keep`` is True."""
    d = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
    starts, ends = np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()
    return list(zip(starts, ends, strict=True))


def pieces(hand: Hand, path: np.ndarray, short=70, long=140) -> list[np.ndarray]:
    """A long line in the hand's reach: pieces of ``short`` to ``long``
    points, each starting a few points before the last ended."""
    out, i = [], 0
    while True:
        n = int(hand.u(short, long))
        if len(path) - i < n + short // 2:
            out.append(path[i:])
            return out
        out.append(path[i : i + n])
        i += n - int(hand.u(3, 8))


def visible_runs(path, covered, edge, within: Region | None = None):
    """The parts of ``path`` that nothing in front hides, inside the edge
    (and inside ``within``)."""
    xi = np.clip(np.round(path[:, 0]).astype(int), 0, WIDTH - 1)
    yi = np.clip(np.round(path[:, 1]).astype(int), 0, HEIGHT - 1)
    on = ~covered[yi, xi] & edge[yi, xi]
    if within is not None:
        ly, lx = yi - within.y0, xi - within.x0
        h, w = within.m.shape
        ok = (ly >= 0) & (lx >= 0) & (ly < h) & (lx < w)
        on[ok] &= within.m[ly[ok], lx[ok]]
        on &= ok
    return [path[a:b] for a, b in runs(on) if b - a >= 4]


def firm_along(hand: Hand, n: int, base: float) -> list[float]:
    """A pressure along a line of ``n`` points: about ``base``, swelling and
    easing, lighter where it lands and lifts."""
    t = np.linspace(0, 1, n)
    swell = 0.07 * np.sin(2 * np.pi * (t * hand.u(0.6, 1.6) + hand.u(0, 1)))
    ends = 1.0 - 0.22 * (1.0 - np.sqrt(np.sin(np.pi * np.clip(t, 0.03, 0.97))))
    return list(np.clip(base * hand.u(0.92, 1.05) * (1 + swell) * ends, 0.1, 1.0))


def line_strokes(
    hand, path, colour, width, covered, edge, base=0.88, under=False, within=None
) -> list[Line]:
    """A line drawn as a hand draws it: in pieces of its reach, where it
    shows, and the outline traced again in part, a little off."""
    out = []
    for run in visible_runs(path, covered, edge, within):
        for piece in pieces(hand, run):
            pts = piece[:: max(1, len(piece) // 30)]
            if len(pts) < 2:
                pts = piece[[0, -1]]
            bold = BOLD if colour == BLACK and width >= 10 else 1.0
            w = width * hand.u(0.92, 1.08) * bold
            out.append(Line(pts, colour, w, firm_along(hand, len(pts), base), under))
            if (under or (width >= 7 and hand.u(0, 1) < 0.6)) and len(pts) > 4:
                out.append(retrace(hand, pts, colour, w, 0.9 if under else 0.68))
    return out


def retrace(hand: Hand, pts: np.ndarray, colour: str, width: float, firm) -> Line:
    """Part of a line gone over again, a pixel or so off."""
    n = len(pts)
    a = int(hand.u(0, 0.4) * n)
    b = max(a + 3, int(hand.u(0.6, 1.0) * n))
    sub = pts[a:b]
    tan = np.gradient(sub, axis=0)
    normal = (
        tan[:, ::-1] * [1, -1] / np.maximum(np.linalg.norm(tan, axis=1), 1e-9)[:, None]
    )
    sub = sub + normal * hand.u(-1.6, 1.6)
    return Line(sub, colour, width * 0.9, firm_along(hand, len(sub), firm))


def plan(items: Sequence[Item], hand: Hand) -> tuple[Plan, np.ndarray]:
    """The strokes of the picture, worked out from the front back: what
    each shape shows of itself once the shapes in front are cut away."""
    own = own_regions(items)
    edge = sheet_edge(hand, MARGIN)
    loose = grow(edge, 6, False)  # lines stop a little further out
    covered = np.zeros((HEIGHT, WIDTH), bool)
    work = Plan()
    for item in reversed(items):
        reg = own[id(item)]
        if isinstance(item, Part):
            plan_part(hand, work, item, reg, covered, loose, edge)
        else:
            plan_mark(hand, work, item, covered, loose, reg)
        if not getattr(item, "under", False) and reg.m.size:
            covered[reg.box] |= reg.m
    return work, edge


def plan_part(hand, work: Plan, part: Part, reg: Region, covered, loose, edge):
    if reg.m.size:
        vis = reg.m & ~covered[reg.box] & edge[reg.box]
        if part.colour is not None and vis.any():
            work.fills.append((part, Region(reg.y0, reg.x0, vis)))
    if part.line > 0:
        target = work.ink if part.ink == BLACK else work.lines
        for loop in outline_path(hand, part.outline):
            target += line_strokes(hand, loop, part.ink, part.line, covered, loose)


def plan_mark(hand, work: Plan, mark: Mark, covered, loose, reg: Region):
    if mark.colour is None:
        return
    target = work.ink if mark.colour == BLACK else work.lines
    firm = mark.under or mark.colour == LEAF
    base = 0.97 if firm or (mark.colour == BLACK and mark.width < 9) else 0.85
    within = reg if mark.clip is not None else None
    target += line_strokes(
        hand,
        mark.path,
        mark.colour,
        mark.width,
        covered,
        loose,
        base,
        mark.under,
        within,
    )


def ink_mask(lines: Sequence[Line]) -> np.ndarray:
    """Where the lines lie, a little narrower than they are drawn."""
    out = np.zeros((HEIGHT, WIDTH), bool)
    for line in lines:
        dense, *_ = CatmullRom(line.path).sample(2.0)
        reg = line_region(dense, line.width * 0.7)
        if reg.m.size:
            out[reg.box] |= reg.m
    return out


# ----------------------------------------------------------------------
# the crayon: filling a shape as a hand fills it


def axes(angle: float) -> tuple[np.ndarray, np.ndarray]:
    t = math.radians(angle)
    return np.array([math.cos(t), -math.sin(t)]), np.array([math.sin(t), math.cos(t)])


class Frame:
    """Which way the strokes of a pass run: straight, at ``angle`` degrees,
    or radiating from ``centre`` (local pixels), when an offset across them
    is an arc length at radius ``ref``."""

    def __init__(self, angle: float, centre=None, ref: float = 1.0):
        self.along, self.across = axes(angle)
        self.turn = math.radians(angle)
        self.centre = None if centre is None else np.asarray(centre, float)
        self.ref = ref

    def coords(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """How far along and how far across the strokes each point lies."""
        if self.centre is None:
            return pts @ self.along, pts @ self.across
        d = pts - self.centre
        turn = (np.arctan2(d[:, 1], d[:, 0]) - self.turn) % (2 * np.pi)
        return np.hypot(d[:, 0], d[:, 1]), turn * self.ref

    def point(self, off: float, s) -> np.ndarray:
        """The point (or points) ``s`` along the stroke at ``off`` across."""
        if self.centre is None:
            return off * self.across + np.multiply.outer(s, self.along)
        t = off / self.ref + self.turn
        ray = np.array([math.cos(t), math.sin(t)])
        return self.centre + np.multiply.outer(s, ray)


def scan(m: np.ndarray, frame: Frame, gap: float, hand: Hand):
    """Lines across the mask, about ``gap`` apart, running as ``frame`` says:
    for each, its offset and the stretches of it inside the mask, as (start,
    end) along it."""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    s, o = frame.coords(np.stack([xs, ys], 1) + 0.5)
    s_line = np.arange(s.min() - 1.5, s.max() + 1.5, 1.0)
    out = []
    off = o.min() + gap * hand.u(0.2, 0.6)
    while off < o.max():
        p = frame.point(off, s_line)
        xi, yi = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < m.shape[1]) & (yi < m.shape[0])
        hit = np.zeros(len(p), bool)
        hit[ok] = m[yi[ok], xi[ok]]
        out.append((off, [(s_line[a], s_line[b - 1]) for a, b in runs(hit)]))
        off += gap * hand.u(0.88, 1.12)
    return out


class Run:
    """Legs of a scribble that one stroke goes through, to and fro."""

    def __init__(self, limit: int):
        self.legs: list[tuple[float, float, float]] = []
        self.limit = limit


def link(scans, hand: Hand) -> list[Run]:
    """Join the stretches of successive lines into back-and-forth strokes:
    a stroke goes on to the next line where that overlaps its last leg,
    for a handful of legs."""
    done: list[Run] = []
    open_: list[tuple[Run, float, float]] = []
    for off, stretches in scans:
        nxt = []
        for s0, s1 in stretches:
            match = None
            for k, (run, a, b) in enumerate(open_):
                overlap = min(b, s1) - max(a, s0)
                if overlap > 0.4 * min(b - a, s1 - s0) and len(run.legs) < run.limit:
                    match = k
                    break
            if match is None:
                run = Run(int(hand.u(4, 10)))
                done.append(run)
            else:
                run = open_.pop(match)[0]
            run.legs.append((off, s0, s1))
            nxt.append((run, s0, s1))
        open_ = nxt
    return done


@dataclass
class Touch:
    """How a pass of filling goes: its direction, its width, how firm, how
    far apart, how neat."""

    angle: float
    width: float = FILL_WIDTH
    firm: float = 0.78
    gap: float = FILL_GAP
    loose: float = 1.0
    skip: float = 0.0  # the share of strokes the hand leaves out
    centre: tuple[float, float] | None = None  # strokes radiate from here
    patchy: bool = False  # the strokes come in clusters, gaps between
    even: bool = False  # every leg as firm as the next


def zigzag(run: Run, touch: Touch, frame: Frame, origin, hand: Hand, rim):
    """The path of one back-and-forth stroke through the legs of ``run``, and
    how hard the hand bears at each of its points: each leg clearly more or
    less than the last, and drifting along its length."""
    pts, bear = [], []
    for j, (off, s0, s1) in enumerate(run.legs):
        a = s0 - end_play(hand, touch)
        b = s1 + end_play(hand, touch)
        if j % 2:
            a, b = b, a
        n = max(int(abs(b - a) // 50), 0)
        ease = 1.0 - (1.0 - EASE) * min(abs(b - a) / (4.0 * touch.width), 1.0)
        firm, drift = (1.0, 0.0) if touch.even else leg_weight(hand)
        wob = hand.rng.normal(0, 0.7 * touch.loose, n + 2)
        for t, w in zip(np.linspace(0, 1, n + 2), wob, strict=True):
            pts.append(origin + frame.point(off + w, a + (b - a) * t))
            # (the hand bears on mid-leg, and eases where it turns)
            weight = firm * (1.0 + drift * (2.0 * t - 1.0))
            bear.append(weight * (ease if t in (0.0, 1.0) else 1.0))
    return stray(np.array(pts), hand, rim), np.array(bear)


def leg_weight(hand: Hand) -> tuple[float, float]:
    """How hard one leg is pressed against the pass's pressure, and how much
    that drifts from its start to its end."""
    return hand.u(LEG_LIGHT, LEG_HEAVY), hand.u(-0.22, 0.22)


def end_play(hand: Hand, touch: Touch) -> float:
    """How far past its stretch a leg runs: mostly about to the line, now
    and then short of it (a hairline of paper) or over it."""
    play = hand.rng.normal(0.3, 1.6 * touch.loose)
    if hand.u(0, 1) < 0.07:
        play += hand.u(2.0, 4.0)
    return float(play)


def stray(path: np.ndarray, hand: Hand, rim: np.ndarray) -> np.ndarray:
    """Where a stroke turns at the picture's edge, it sometimes runs on a
    little into the margin."""
    out = path.copy()
    for k in (0, len(path) - 1):
        x, y = np.clip(np.round(path[k]).astype(int), 0, [WIDTH - 1, HEIGHT - 1])
        if rim[y, x] and hand.u(0, 1) < 0.35 and len(path) > 1:
            d = path[k] - path[1 if k == 0 else -2]
            out[k] = path[k] + d / max(np.linalg.norm(d), 1e-9) * hand.u(4, 14)
    return out


def grow(m: np.ndarray, r: int, beyond: bool) -> np.ndarray:
    """``m`` grown by about ``r`` pixels all round (``beyond``: what lies
    outside the box)."""
    if r <= 0:
        return m
    padded = np.pad(m, r + 1, constant_values=beyond)
    return (filters.max_filter(padded, r) > 0.5)[r + 1 : -r - 1, r + 1 : -r - 1]


def fit_width(m: np.ndarray, widest: float) -> float:
    """The crayon for the shape: up to ``widest``, and narrower in a narrow
    shape, so that a few strokes go across it."""
    for r in range(12, 1, -1):
        if (m & ~grow(~m, r, True)).any():
            return min(widest, max(1.2 * r, 5.0))
    return 5.0


#: A stroke stops short of a line by its own half width; where the shape
#: meets another with no line between them, it runs this much further.
SOFT = 1


def scribble(sheet, hand: Hand, m, hard, y0, x0, colour, touch: Touch, rim, again=1):
    """Fill the mask (at (y0, x0) on the sheet) with back-and-forth strokes,
    up to its lines (``hard``) and a little over its other edges; then go
    back (``again`` times) into what a narrow neck of it left bare."""
    width = fit_width(m, touch.width)
    reach = int(round(width / 2 - 1.5))
    # (black keeps clear of everything it does not own, as of a cat's eyes,
    # and so does a crown round its apples)
    soft = reach if colour in (BLACK, LEAF) else reach - SOFT
    centre = m & ~grow(hard & ~m, reach, True) & ~grow(~m & ~hard, soft, False)
    if not centre.any():
        dab(sheet, hand, m, y0, x0, colour, touch)
        return
    medium = crayons.wax(CRAYON[colour])
    origin = np.array([x0, y0], float)
    # (a narrow shape is filled closer: its few strokes must cover it)
    gap = touch.gap if width > 0.75 * FILL_WIDTH else min(touch.gap, 0.65)
    plain = Touch(touch.angle, width, touch.firm, gap, touch.loose, touch.skip)
    plain.centre, plain.even, plain.patchy = touch.centre, touch.even, touch.patchy
    touch = plain
    frame = frame_for(touch, centre)
    reached = np.zeros_like(m)
    lines = scan(centre, frame, touch.gap * width, hand)
    if touch.patchy:
        lines = clusters(lines, hand)
    for run in link(lines, hand):
        if hand.u(0, 1) < touch.skip:
            continue
        path, bear = zigzag(run, touch, frame, origin, hand, rim)
        if len(path) < 2:
            continue
        cap = 0.98 if colour == BLACK else 0.82
        pressure = list(
            np.clip(firm_along(hand, len(path), touch.firm) * bear, 0.1, cap)
        )
        sheet.stroke(path, medium, pressure, width * hand.u(0.92, 1.06))
        mark_reached(reached, path - origin, width - 2.0)
    left = m & ~reached
    if again > 0 and (left & ~grow(~left, 2, True)).any():
        narrow = Touch(touch.angle + hand.u(-30, 30), width, touch.firm, touch.gap)
        narrow.even = touch.even
        scribble(sheet, hand, left, hard, y0, x0, colour, narrow, rim, again - 1)


def clusters(lines, hand: Hand):
    """Keep the lines of a pass in clusters of a few, with gaps of a few
    between them, and only part of each line: the patches a hand goes over
    again."""
    if not lines:
        return lines
    n = len(lines)
    across = noise.value_noise_1d(n, hand.u(3.0, 5.0), hand.rng)
    length = max((b for _, st in lines for _, b in st), default=0.0)
    start = min((a for _, st in lines for a, _ in st), default=0.0)
    out = []
    for k, (off, stretches) in enumerate(lines):
        if across[k] < 0.55:
            continue
        lo = start + (length - start) * hand.u(-0.3, 0.4)
        hi = lo + (length - start) * hand.u(0.5, 0.9)
        cut = [
            (max(a, lo), min(b, hi))
            for a, b in stretches
            if min(b, hi) - max(a, lo) > 8
        ]
        out.append((off, cut))
    return out


def frame_for(touch: Touch, m: np.ndarray) -> Frame:
    """Straight strokes, or strokes radiating from the touch's centre."""
    if touch.centre is None:
        return Frame(touch.angle)
    ys, xs = np.nonzero(m)
    far = np.hypot(xs - touch.centre[0], ys - touch.centre[1]).max()
    return Frame(touch.angle, touch.centre, 0.6 * max(far, 1.0))


def mark_reached(reached: np.ndarray, path: np.ndarray, width: float) -> None:
    """Note where a stroke along ``path`` (local) has laid colour."""
    dense, *_ = CatmullRom(path).sample(2.0)
    reg = line_region(dense, width)
    if not reg.m.size:
        return
    h, w = reached.shape
    y1, x1 = min(reg.y0 + reg.m.shape[0], h), min(reg.x0 + reg.m.shape[1], w)
    if y1 > reg.y0 and x1 > reg.x0:
        reached[reg.y0 : y1, reg.x0 : x1] |= reg.m[: y1 - reg.y0, : x1 - reg.x0]


def dab(sheet: Sheet, hand: Hand, m, y0, x0, colour, touch: Touch):
    """A shape too small to scribble: one firm dab, as wide as it is."""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return
    size = 2.0 * math.sqrt(len(xs) / math.pi)
    cx, cy = x0 + xs.mean() + 0.5, y0 + ys.mean() + 0.5
    width = min(max(size * 0.8, 4.0), touch.width)
    along, _ = axes(hand.u(0, 180))
    reach = max((size - width) / 2, size * 0.15) * along
    path = [(cx, cy) - reach, (cx, cy) + reach]
    medium = crayons.wax(CRAYON[colour])
    sheet.stroke(path, medium, max(min(touch.firm, 0.86), 0.66), width)


# ----------------------------------------------------------------------
# the crayon: patches of a large shape


def ellipse_mask(h, w, cx, cy, rx, ry, turn) -> np.ndarray:
    """A tilted ellipse over an (h, w) box."""
    out = np.zeros((h, w), bool)
    r = max(rx, ry)
    y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, h)
    x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, w)
    if y1 <= y0 or x1 <= x0:
        return out
    y, x = np.mgrid[y0:y1, x0:x1] + 0.5
    c, s = math.cos(turn), math.sin(turn)
    u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
    out[y0:y1, x0:x1] = (u / rx) ** 2 + (v / ry) ** 2 <= 1.0
    return out


def patches(m: np.ndarray, hand: Hand, size: float) -> list[tuple[np.ndarray, float]]:
    """A large mask cut into overlapping patches of a hand's reach, each with
    the x of its middle; together they cover all of it."""
    h, w = m.shape
    left = m.copy()
    out = []
    step = size * 0.75
    for y in np.arange(step * 0.4, h + step * 0.3, step * 0.8):
        for x in np.arange(step * 0.4 + hand.u(0, step * 0.4), w + step * 0.3, step):
            cx, cy = x + hand.u(-0.2, 0.2) * step, y + hand.u(-0.2, 0.2) * step
            patch = m & ellipse_mask(h, w, cx, cy, *patch_size(hand, size))
            if (patch & left).sum() > 0.05 * size * size:
                out.append((patch, cx))
                left &= ~patch
    while left.sum() > 0.002 * m.sum() + 30:
        ys, xs = np.nonzero(left)
        k = int(hand.rng.integers(len(xs)))
        cx, cy = xs[k] + 0.5, ys[k] + 0.5
        patch = m & ellipse_mask(h, w, cx, cy, *patch_size(hand, size * 0.7))
        out.append((patch, cx))
        left &= ~patch
    return out


def patch_size(hand: Hand, size: float) -> tuple[float, float, float]:
    return size * hand.u(0.5, 0.68), size * hand.u(0.4, 0.55), hand.u(-0.6, 0.6)


def fill(sheet: Sheet, hand: Hand, part: Part, m: np.ndarray, hard, y0, x0, rim):
    """Fill one shape, in one go or patch by patch, with a second, looser
    pass across the first where the shape should be solid."""
    h, w = m.shape
    parts = [(m, x0 + w / 2)] if max(h, w) <= REACH else patches(m, hand, 260.0)
    for patch, cx in parts:
        angle = part_angle(part, cx) + hand.u(-part.drift, part.drift)
        firm = FIRM.get(part.colour or "", 0.86) * hand.u(0.92, 1.05) * part.weight
        if part.tint is not None and hand.u(0, 1) < 0.4:
            under = Touch(angle - hand.u(20, 40), firm=0.6, gap=FILL_GAP * 1.2)
            scribble(sheet, hand, patch, hard, y0, x0, part.tint, under, rim, 0)
            firm *= 0.85
        centre = radial_centre(part, patch)
        for touch in passes(hand, part, angle, firm, centre):
            scribble(sheet, hand, patch, hard, y0, x0, part.colour, touch, rim)


def passes(hand: Hand, part: Part, angle: float, firm: float, centre) -> list[Touch]:
    """How a patch is filled: one pass with its legs about a width apart,
    and where the shape should be solid a second, patchy one across it;
    black is pressed solid, in two even passes."""
    turn = hand.u(45, 70) * hand.rng.choice([-1, 1])
    if part.colour == BLACK:
        first = Touch(angle, firm=0.95, gap=0.5)
        cross = Touch(angle + turn, firm=0.95, gap=0.5)
        first.even = cross.even = True
        return [first, cross]
    first = Touch(angle, firm=firm)
    first.centre = centre
    out = [first]
    if hand.u(0, 1) < part.passes - 1:
        cross = Touch(angle + turn, firm=firm * 0.95, gap=0.9)
        cross.loose, cross.patchy = 1.6, True
        out.append(cross)
    return out


def radial_centre(part: Part, m: np.ndarray):
    """Where a shape filled radiating (the sun) radiates from: its middle."""
    if not part.radial:
        return None
    ys, xs = np.nonzero(m)
    return (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)


def part_angle(part: Part, x: float) -> float:
    if part.flow is None:
        return part.angle
    return part.flow(x) + part.angle


#: Where a stroke turns, at a shape's edge, it bears this much as hard as
#: mid-leg: under the pressure at which wax smears what it passes over.
EASE = 0.8
#: How hard a leg of a pass is pressed, against the pass's own pressure:
#: from a skim to a firm one.
LEG_LIGHT, LEG_HEAVY = 0.45, 1.15

#: How firmly each colour is pressed when filling.
FIRM = {SUN: 0.86, LIME: 0.74, GREEN: 0.78, SKY: 0.82, CREAM: 0.66, PINK: 0.78}
FIRM[BLACK] = 0.95


def draw(items: Sequence[Item]) -> Sheet:
    """Lay the whole picture in crayon: the black lines first, on bare paper;
    then the coloured lines and patterns; then every shape filled, the light
    colours first, each up to its lines."""
    hand = Hand(SEED + 2)
    work, edge = plan(items, hand)
    sheet = Sheet(WIDTH, HEIGHT, paper=PAPER, seed=SEED)
    for line in work.ink + work.lines:
        medium = crayons.wax(CRAYON[line.colour])
        sheet.stroke(line.path, medium, line.pressure, line.width)
    ink = ink_mask([ln for ln in work.ink + work.lines if not ln.under])
    rim = edge & (erode(edge, 10) == 0)
    rank = {colour: k for k, colour in enumerate(ORDER)}

    def order(f):
        """Patterns first, then small dabs, then the light colours."""
        return (not f[0].under, f[1].m.sum() > 400, rank[f[0].colour])

    for part, reg in sorted(work.fills, key=order):
        hard = ink[reg.box] | ~edge[reg.box]
        fill(sheet, hand, part, reg.m & ~ink[reg.box], hard, reg.y0, reg.x0, rim)
    return sheet


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
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "output" / "farm.png"
    save(draw(scene()).render(), out)
    print(f"wrote {out}")


CROPS = {
    "sun": (1000, 0, 1400, 330),
    "barn": (620, 300, 1180, 720),
    "house": (0, 160, 620, 760),
    "animals": (0, 720, 560, 1050),
    "pond": (460, 720, 1400, 1050),
    "flowers": (1080, 480, 1400, 760),
}


if __name__ == "__main__":
    main()
