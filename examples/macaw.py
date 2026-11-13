"""A great green macaw banking in flight, seen from above, in coloured pencil
on white paper.

Run from the repository root: ``python examples/macaw.py``. Writes
``examples/output/macaw.png``.

The bird is built from its anatomy, not copied from a picture. Each wing is
a skeleton (shoulder, elbow, wrist, hand) that carries ten primaries on the
hand and the secondaries on the forearm; over their bases lie the greater
coverts, then the median and lesser coverts in rows toward the leading
edge, the alula at the wrist and the scapulars over the shoulder, each
feather overlapping the next like a roof tile. The tail is twelve graduated
feathers under its coverts; the back, rump and head are small contour
feathers lying along their tracts. The bird glances up, so its head shows in
profile: the hooked beak, the scarlet band above it, the bare face with its
lines of tiny feathers, the eye.

Every feather is drawn as a feather: barbs leaving the shaft toward the tip
on both vanes, a darker band where the feather above casts its shadow, then
the shaft. Colours are those of the species: yellow-green coverts, blue
flight feathers, a turquoise rump, a rust-red tail tipped with blue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from murex import CatmullRom, filters, write_png

from fusain import Paper, Sheet, pencils, tools

WIDTH, HEIGHT = 1400, 1150
#: Where the bird is on the sheet: the point between its shoulders, the way
#: it is heading (radians, y down) and its size.
CENTRE = np.array([640.0, 560.0])
HEADING = math.radians(222.0)
SCALE = 1.12
AHEAD = np.array([math.cos(HEADING), math.sin(HEADING)])
RIGHT = np.array([-AHEAD[1], AHEAD[0]])  # the bird's right, seen from above


def place(x: float, y: float) -> np.ndarray:
    """The sheet point of a point of the bird: ``x`` ahead of its shoulders,
    ``y`` to its right, in the bird's own measure."""
    return CENTRE + SCALE * (x * AHEAD + y * RIGHT)


def aim(angle: float, side: float = 1.0) -> np.ndarray:
    """The bird's-measure direction ``angle`` degrees back from straight out
    to the side (0 points out along the wing, 90 straight back)."""
    turn = math.radians(angle)
    return np.array([-math.sin(turn), side * math.cos(turn)])


@dataclass
class Feather:
    """One feather, in the bird's measure: its shaft from ``base`` toward
    ``way`` for ``length``, bowed sideways by ``bow`` (a share of its
    length); the widths of its two vanes (``lead`` on the side ``leading``
    points to); how pointed its tip is (0 round to 1 pointed); its plumage,
    which names its colours; and its rank: a higher rank lies over a lower."""

    base: np.ndarray
    way: np.ndarray
    length: float
    lead: float
    trail: float
    leading: np.ndarray
    plumage: str
    rank: int
    point: float = 0.3
    bow: float = 0.0

    def shaft(self, u) -> np.ndarray:
        """Points of the shaft at ``u`` (0 base to 1 tip), on the sheet."""
        u = np.asarray(u, dtype=np.float64)[..., None]
        side = self.leading - (self.leading @ self.way) * self.way
        bend = self.bow * self.length * np.sin(np.pi * u) * side
        at = self.base + u * self.length * self.way + bend
        return CENTRE + SCALE * (at[..., 0:1] * AHEAD + at[..., 1:2] * RIGHT)

    def breadth(self, u) -> np.ndarray:
        """How wide a vane is at ``u``, as a share of its widest."""
        u = np.asarray(u, dtype=np.float64)
        rise = np.minimum(0.3 + 2.4 * u, 1.0)
        start = 0.8 - 0.35 * self.point
        tip = np.clip((u - start) / (1.0 - start), 0.0, 1.0)
        return rise * (1.0 - tip ** (2.0 - self.point)) ** (0.5 + 0.5 * self.point)

    def outline(self, count: int = 24) -> np.ndarray:
        """The feather's outline on the sheet, (2 * count, 2)."""
        u = np.linspace(0.0, 1.0, count)
        spine = self.shaft(u)
        ahead = np.gradient(spine, axis=0)
        ahead /= np.maximum(np.hypot(ahead[:, 0], ahead[:, 1]), 1e-9)[:, None]
        across = self.across(ahead)
        wide = self.breadth(u)[:, None] * SCALE
        one = spine + across * wide * self.lead
        other = spine - across * wide * self.trail
        return np.concatenate([one, other[::-1]])

    def across(self, ahead: np.ndarray) -> np.ndarray:
        """Unit vectors across the shaft on the sheet, toward the leading
        vane, where the shaft runs along ``ahead`` (..., 2)."""
        normal = np.stack([-ahead[..., 1], ahead[..., 0]], axis=-1)
        lead = self.leading[0] * AHEAD + self.leading[1] * RIGHT
        return normal * np.sign(normal @ lead + 1e-12)[..., None]


# --------------------------------------------------------------------------
# the wing

#: The bones of the right wing, in the bird's measure: shoulder, elbow,
#: wrist and the tip of the hand; and the leading edge at shoulder and wrist.
SHOULDER = np.array([70.0, 42.0])
ELBOW = SHOULDER + 92.0 * aim(28.0)
WRIST = ELBOW + 150.0 * aim(-20.0)
HAND = WRIST + 100.0 * aim(12.0)
FRONT = (np.array([104.0, 50.0]), WRIST + np.array([22.0, 6.0]))
PRIMARIES = (235, 250, 265, 280, 295, 305, 310, 305, 290, 262)


def lerp(a, b, t: float):
    return np.asarray(a, dtype=np.float64) * (1.0 - t) + np.asarray(b, dtype=np.float64) * t


def mirrored(point: np.ndarray, side: float, squash: float) -> np.ndarray:
    return np.array([point[0], side * squash * point[1]])


def arm(s: float) -> np.ndarray:
    """The point of the arm's bones at ``s``: 0 at the shoulder, 0.38 at the
    elbow, 1 at the wrist."""
    if s < 0.38:
        return lerp(SHOULDER, ELBOW, s / 0.38)
    return lerp(ELBOW, WRIST, (s - 0.38) / 0.62)


def remiges(side: float, squash: float, sweep: float) -> list[Feather]:
    """The flight feathers of one wing and their greater coverts: primaries
    from the outermost in, then secondaries from the wrist in, each lying
    over the one before; the coverts over them all."""
    quills = []
    for k in range(9, -1, -1):  # primaries, outermost first
        base = lerp(WRIST, HAND, k / 9)
        angle = lerp(58.0, 6.0, (k / 9) ** 0.9) + sweep
        quills.append((base, angle, PRIMARIES[k], 9.0 + 0.5 * (9 - k), 37.0 - 1.4 * k, 0.4, "primary"))
    for j in range(12):  # secondaries, then the tertials by the body
        if j < 10:
            base, angle = lerp(WRIST, ELBOW, (j + 0.7) / 10), lerp(64.0, 90.0, j / 9) + sweep
            length = lerp(232.0, 205.0, j / 9)
        else:
            base, angle = lerp(ELBOW, SHOULDER, (j - 9.3) / 4), 98.0 + 10.0 * (j - 10) + sweep
            length = 170.0 - 35.0 * (j - 10)
        quills.append((base, angle, length, 14.0, 34.0, 0.2, "secondary"))
    feathers = []
    for rank, (base, angle, length, lead, trail, point, plumage) in enumerate(quills):
        way, leading = aim(angle, side), aim(angle - 90.0, side)
        root = mirrored(base, side, squash)
        feathers.append(Feather(root, way, length, lead, trail, leading, plumage, 100 + rank, point, 0.02))
        short = 0.36 if plumage == "primary" else 0.5
        cover = root - 8.0 * way
        feathers.append(
            Feather(cover, way, short * length + 8.0, 11.0, 17.0, leading, plumage + " covert", 200 + rank, 0.2)
        )
    return feathers


def coverts(side: float, squash: float, sweep: float) -> list[Feather]:
    """The median and lesser coverts of the arm, in rows toward the leading
    edge, each row over the one behind it; those of the hand; the alula."""
    feathers = []
    # (how far ahead of the bones each row starts, its feathers' length and
    # half-width: the median coverts reach furthest back)
    rows = ((0.0, 78.0, 14.0, "median covert"), (0.3, 58.0, 12.0, "lesser covert"),
            (0.62, 46.0, 11.0, "lesser covert"), (0.92, 36.0, 10.0, "lesser covert"))  # fmt: skip
    for row, (forward, length, wide, plumage) in enumerate(rows):
        count = 13 - row
        for k in range(count):
            s = (k + 0.5 + 0.5 * (row % 2)) / (count + 0.5)
            base = lerp(arm(s), lerp(*FRONT, s), forward)
            angle = lerp(96.0, 62.0, s) + sweep
            feathers.append(
                Feather(mirrored(base, side, squash), aim(angle, side), length, wide, wide,
                        aim(angle - 90.0, side), plumage, 300 + 20 * row + k, 0.15)  # fmt: skip
            )
    for k in range(7):  # along the hand
        base = lerp(WRIST, HAND, (k + 0.3) / 7) + 7.0 * aim(-70.0)
        angle = lerp(52.0, 14.0, k / 6) + sweep
        feathers.append(
            Feather(mirrored(base, side, squash), aim(angle, side), 40.0, 8.0, 9.0,
                    aim(angle - 90.0, side), "hand covert", 290 - k, 0.2)  # fmt: skip
        )
    for k, length in enumerate((64.0, 52.0, 40.0)):  # the alula, at the wrist
        base = WRIST + np.array([8.0 - 3.0 * k, 6.0 + 7.0 * k])
        feathers.append(
            Feather(mirrored(base, side, squash), aim(14.0 + sweep, side), length, 6.0, 9.0,
                    aim(-76.0 + sweep, side), "alula", 380 + k, 0.5)  # fmt: skip
        )
    return feathers


def scapulars(side: float, squash: float) -> list[Feather]:
    """The scapulars: long feathers from the shoulder back along the body,
    over the inner wing."""
    feathers = []
    for k in range(9):
        base = lerp((66.0, 36.0), (-46.0, 40.0), k / 8) + np.array([0.0, 12.0 * math.sin(k)])
        angle = 97.0 + 1.5 * k
        feathers.append(
            Feather(mirrored(base, side, squash), aim(angle, side), 112.0 + 3.0 * k, 16.0, 18.0,
                    aim(angle - 90.0, side), "scapular", 430 - k, 0.25, 0.03)  # fmt: skip
        )
    return feathers


# --------------------------------------------------------------------------
# tail and body

RECTRICES = ((1.5, 470.0), (4.5, 430.0), (8.0, 370.0), (12.0, 305.0), (16.5, 245.0), (21.0, 190.0))
TAIL = np.array([-215.0, 0.0])


def tail() -> list[Feather]:
    """The twelve tail feathers, the outer ones lowest, and the coverts that
    lie over their bases."""
    feathers = []
    for side in (1.0, -1.0):
        for k, (angle, length) in enumerate(RECTRICES):
            turn = math.radians(angle + (3.0 if side > 0 else 0.0))
            way = np.array([-math.cos(turn), side * math.sin(turn)])
            base = TAIL + np.array([0.0, side * 3.5 * (k + 0.5)])
            rank = 60 - 2 * k - (1 if side < 0 else 0)
            leading = np.array([-way[1], way[0]]) * side
            feathers.append(Feather(base, way, length, 14.0 - 0.4 * k, 15.0 - 0.4 * k, leading,
                                    "tail" if k < 3 else "outer tail", rank, 0.75, 0.02))  # fmt: skip
    for k in range(11):
        y = (k - 5) * 7.5
        turn = math.radians(2.5 * (k - 5))
        way = np.array([-math.cos(turn), math.sin(turn)])
        base = np.array([-150.0 - 4.0 * abs(k - 5), y])
        feathers.append(Feather(base, way, 118.0 - 5.0 * abs(k - 5), 9.0, 9.0,
                                np.array([0.0, 1.0]), "tail covert", 70 + 5 - abs(k - 5), 0.4))  # fmt: skip
    return feathers


def half_width(x: float) -> float:
    """Half the width of the body, seen from above, ``x`` ahead of the
    shoulders."""
    xs = (-225.0, -150.0, -60.0, 40.0, 120.0, 190.0, 240.0)
    wide = (20.0, 36.0, 50.0, 57.0, 55.0, 49.0, 45.0)
    return float(np.interp(x, xs, wide))


def body(rng) -> list[Feather]:
    """The contour feathers of rump, back, mantle and neck: rows from the
    tail forward, each row over the one behind."""
    feathers = []
    x, row = -215.0, 0
    while x < 232.0:
        size = float(np.interp(x, (-215.0, 60.0, 232.0), (40.0, 36.0, 24.0)))
        wide = half_width(x)
        count = max(int(2 * wide / (0.62 * size)), 2)
        for k in range(count):
            y = wide * (2 * (k + 0.5 * (row % 2) + rng.uniform(-0.2, 0.2)) / count - 1.0) * 0.96
            if abs(y) > wide:
                continue
            splay = math.radians(14.0 * y / wide + rng.normal(0.0, 5.0))
            way = np.array([-math.cos(splay), math.sin(splay)])
            plumage = "rump" if x < -95.0 else ("back" if x < -30.0 else ("mantle" if x < 150.0 else "nape"))
            feathers.append(Feather(np.array([x + rng.uniform(-3.0, 3.0), y]), way, size, 0.36 * size, 0.36 * size,
                                    np.array([0.0, 1.0]), plumage, 500 + row, 0.1))  # fmt: skip
        x += 0.42 * size
        row += 1
    return feathers


# --------------------------------------------------------------------------
# the head, in profile: the bird glances up, one eye to the sky

#: Where the middle of the skull is, in the bird's measure, and how far the
#: head is turned from the way the bird is heading (degrees).
SKULL = np.array([270.0, 0.0])
NOD = -14.0
#: The head's own measure: x toward the beak, y toward the crown.
UPPER_BEAK = ((38, 34), (62, 41), (86, 32), (103, 10), (109, -18), (103, -44), (88, -64),
              (84, -42), (74, -27), (58, -20), (46, -18))  # fmt: skip
LOWER_BEAK = ((48, -20), (70, -30), (80, -46), (74, -63), (58, -69), (44, -61), (40, -40))
FACE = ((38, 32), (18, 37), (-8, 27), (-23, 3), (-19, -26), (2, -46), (30, -57), (42, -51),
        (40, -40), (46, -18))  # fmt: skip
BROW = ((38, 34), (41, 46), (30, 57), (9, 59), (1, 48), (16, 38))
EYE = (10.0, 12.0)


def head_point(x: float, y: float) -> np.ndarray:
    """The bird's-measure point of a point of the head's own measure."""
    turn = math.radians(NOD)
    return SKULL + np.array([x * math.cos(turn) - y * math.sin(turn), x * math.sin(turn) + y * math.cos(turn)])


def head_shape(points) -> np.ndarray:
    """A shape of the head, on the sheet: a smooth closed curve through
    ``points`` of the head's own measure."""
    corners = np.array([place(*head_point(x, y)) for x, y in points])
    return CatmullRom(corners, closed=True).sample(2.0)[0]


def crest(rng) -> list[Feather]:
    """The small feathers of crown, nape, cheek and throat, lying back from
    the face, those nearer the beak over those behind; and the scarlet ones
    of the brow."""
    feathers = []
    for ring, radius in enumerate(np.arange(78.0, 1.0, -5.5)):
        count = max(int(2 * math.pi * radius / 9.0), 6)
        for k in range(count):
            turn = 2 * math.pi * (k + 0.5 * (ring % 2)) / count
            x, y = radius * math.cos(turn) - 22.0, 0.82 * radius * math.sin(turn) + 2.0
            away = np.array([x - 40.0, y - 5.0])
            way = away / max(float(np.hypot(*away)), 1e-6) + np.array([-0.7, 0.0])
            way = way / float(np.hypot(*way)) + rng.normal(0.0, 0.06, 2)
            base = head_point(x, y)
            ahead = head_point(x + way[0], y + way[1]) - base
            ahead /= float(np.hypot(*ahead))
            red = -2.0 < x < 44.0 and 30.0 < y < 64.0
            feathers.append(Feather(base, ahead, 12.0 if red else 15.0, 4.6, 4.6, np.array([-ahead[1], ahead[0]]),
                                    "brow" if red else ("nape" if x < -40.0 else "crown"), 600 + ring, 0.1))  # fmt: skip
    return feathers


def plumage(rng) -> list[Feather]:
    """Every feather of the bird, lowest first. The left wing is seen a
    little from its edge, as the bird banks."""
    feathers = tail()
    for side, squash, sweep in ((1.0, 1.0, 0.0), (-1.0, 0.88, 5.0)):
        feathers += remiges(side, squash, sweep)
        feathers += coverts(side, squash, sweep)
        feathers += scapulars(side, squash)
    feathers += body(rng) + crest(rng)
    return sorted(feathers, key=lambda feather: feather.rank)


# --------------------------------------------------------------------------
# which feather shows where


def fill(owner: np.ndarray, polygon: np.ndarray, mark: int) -> None:
    """Mark the pixels inside ``polygon`` in ``owner``."""
    x0, y0 = np.floor(polygon.min(axis=0)).astype(int)
    x1, y1 = np.ceil(polygon.max(axis=0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, owner.shape[1]), min(y1, owner.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1] + 0.5
    inside = np.zeros(xs.shape, bool)
    ax, ay = polygon[-1]
    for bx, by in polygon:
        if ay != by:
            crosses = (ys >= min(ay, by)) & (ys < max(ay, by))
            inside ^= crosses & (xs < ax + (ys - ay) * (bx - ax) / (by - ay))
        ax, ay = bx, by
    owner[y0:y1, x0:x1][inside] = mark


def shadows(owner: np.ndarray) -> np.ndarray:
    """How deep each pixel lies in the shadow of a feather above it, 0..1:
    the light comes from the upper left, so a feather is shaded along the
    edge of the one that overlaps it on that side."""
    shade = np.zeros(owner.shape)
    for step in range(1, 17):
        toward = np.roll(owner, (round(0.8 * step), round(0.6 * step)), axis=(0, 1))
        shade += ((toward > owner) & (owner >= 0)) / step**0.5
    return filters.box_blur(np.clip(shade / 4.5, 0.0, 1.0), 1)


# --------------------------------------------------------------------------
# drawing

PAPER = Paper("wove", grain=2.0, depth=0.4, bite=0.5, give=0.4, color="#fbfaf6")
#: A barb is drawn every so many pixels along the shaft, with a point this
#: wide.
BARBS, POINT = 2.4, 2.6


def pencil_set(*colours: str) -> tuple:
    return tuple(pencils.coloured(colour) for colour in colours)


#: The pencils for each plumage: a light one, the main one and a deep one.
GREEN = pencil_set("#c2cc46", "#7fa22c", "#35622a")
OLIVE = pencil_set("#d0cc4e", "#9aa030", "#55651f")
TEAL = pencil_set("#86cdbd", "#22888a", "#164e5a")
BLUE = pencil_set("#4f93c2", "#1a5684", "#0f2c4c")
NAVY = pencil_set("#2f6090", "#153a60", "#0b1424")
MARINE = pencil_set("#52a7ae", "#1b7486", "#0f4352")
LIME = pencil_set("#d9d954", "#b0bd34", "#647f28")
TURQUOISE = pencil_set("#c6e9e3", "#7fcfcc", "#35909a")
RUST = pencil_set("#ee8a3a", "#d0442a", "#84261a")
SKY = pencil_set("#86c2e2", "#3083bf", "#1b4a88")
SCARLET = pencil_set("#ee5a4a", "#d81e2b", "#8a141c")
QUILL = pencils.coloured("#16161c")
HORN, SLATE, JET = pencil_set("#a9a69a", "#55565c", "#141518")
SKIN, BLUSH, WINE = pencil_set("#f3ddd4", "#e3a9a4", "#5e1b24")
IRIS = pencils.coloured("#e7cf6a")


def colours(feather: Feather, u: float, lead: bool, rng) -> tuple:
    """The light, main and deep pencils for a barb at ``u`` along a feather,
    on its leading or its trailing vane."""
    kind = feather.plumage
    if kind in ("primary", "secondary", "alula"):
        if u > 0.8 + 0.12 * rng.random():
            return NAVY  # the dusky tip
        if kind == "secondary" and u < 0.3 + 0.25 * rng.random():
            return MARINE  # greener toward the base
        return BLUE if lead else (NAVY if rng.random() < 0.6 else BLUE)
    if kind in ("primary covert", "secondary covert", "hand covert"):
        return TEAL if u > 0.25 + 0.3 * rng.random() else GREEN
    if kind in ("tail", "outer tail"):
        change = 0.6 if kind == "tail" else 0.32
        if kind == "outer tail" and lead:
            change = 0.12
        return SKY if u > change + 0.2 * rng.random() else RUST
    if kind in ("rump", "tail covert"):
        return TURQUOISE
    if kind == "brow":
        return SCARLET
    if kind == "nape":
        return LIME if rng.random() < 0.65 else GREEN
    if kind == "back":
        return TURQUOISE if rng.random() < 0.5 else GREEN
    return OLIVE if kind in ("scapular", "median covert") and rng.random() < 0.6 else GREEN


def barbs(feather: Feather, rng, apart: float = BARBS):
    """The barbs of a feather, ``apart`` pixels from one another: for each,
    the points it runs through on the sheet (root on the shaft to tip at the
    edge of the vane), how far along the shaft it leaves, and whether it is
    on the leading vane."""
    length = feather.length * SCALE
    count = max(int(length / apart), 3)
    out = []
    for lead, wide in ((True, feather.lead), (False, feather.trail)):
        u = (np.arange(count) + rng.uniform(0.2, 0.8, count)) / count
        root = feather.shaft(u)
        ahead = feather.shaft(np.minimum(u + 0.02, 1.0)) - feather.shaft(np.maximum(u - 0.02, 0.0))
        ahead /= np.maximum(np.hypot(ahead[:, 0], ahead[:, 1]), 1e-9)[:, None]
        across = feather.across(ahead) * (1.0 if lead else -1.0)
        slant = np.radians(44.0 - 18.0 * u + rng.normal(0.0, 1.2, count))
        # (a barb ends further along than it starts: the vane's width there)
        along = wide * SCALE * feather.breadth(u) / np.tan(slant)
        reach = wide * SCALE * feather.breadth(np.minimum(u + 0.7 * along / length, 1.0))
        reach = np.maximum(reach - 1.3, 0.0)  # (the stroke has a width of its own)
        tip = root + ahead * (reach / np.tan(slant))[:, None] + across * reach[:, None]
        for k in range(count):
            if reach[k] < 1.2:
                continue
            bend = 0.1 * reach[k] * ahead[k]
            path = np.array([root[k], lerp(root[k], tip[k], 0.4) + bend, lerp(root[k], tip[k], 0.75) + bend, tip[k]])
            out.append((path, float(u[k]), lead))
    return out


def showing(owner: np.ndarray, index: int, path: np.ndarray) -> np.ndarray:
    """1 where the points of a path belong to the feather of that index."""
    col = np.clip(path[:, 0].astype(np.int64), 0, owner.shape[1] - 1)
    row = np.clip(path[:, 1].astype(np.int64), 0, owner.shape[0] - 1)
    return (owner[row, col] == index) * 1.0


class Field:
    """A patch of values, its corner at (``left``, ``top``) on the sheet, as
    a function of x and y (0 outside it): a shading's pressure."""

    def __init__(self, values: np.ndarray, left: int = 0, top: int = 0):
        self.values, self.left, self.top = values, left, top

    def __call__(self, x, y):
        height, width = self.values.shape
        col = np.asarray(x, dtype=np.int64) - self.left
        row = np.asarray(y, dtype=np.int64) - self.top
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        return np.where(inside, self.values[np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)], 0.0)


def window(shape: np.ndarray) -> tuple[slice, slice]:
    """The rows and columns of the sheet that hold a shape."""
    x0, y0 = np.floor(shape.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(shape.max(axis=0)).astype(int) + 3
    return slice(max(y0, 0), min(y1, HEIGHT)), slice(max(x0, 0), min(x1, WIDTH))


def tone_at(path, u: float, lead: bool, shade, rng) -> float:
    """How dark a barb is, 0..1: the trailing vane darker than the leading
    one, the tip dusky, and darker still in the shadow of the feather above."""
    col = min(max(int(path[2, 0]), 0), shade.shape[1] - 1)
    row = min(max(int(path[2, 1]), 0), shade.shape[0] - 1)
    tone = (0.22 if lead else 0.42) + 0.9 * shade[row, col] + 0.35 * max(u - 0.78, 0.0) / 0.22
    return float(min(max(tone + rng.normal(0.0, 0.06), 0.0), 1.0))


#: Toward the light, on the sheet and in the bird's measure.
LIGHT = np.array([-0.6, -0.8])
SIDE_LIGHT = float(LIGHT @ RIGHT)
BODY = ("rump", "back", "mantle", "nape", "tail covert")


def turned(feather: Feather) -> float:
    """How far a feather of the body is turned from the light, 0..1: the
    body is round, and darker down the side away from the lamp."""
    if feather.plumage not in BODY:
        return 0.0
    aside = feather.base[1] / half_width(feather.base[0])
    return float(np.clip(-0.45 * aside * np.sign(SIDE_LIGHT), 0.0, 0.45))


def draw_feather(sheet: Sheet, feather: Feather, index: int, owner, shade, rng) -> None:
    """One feather, where it shows: the shadow of the feather over it, its
    colour laid solid along its length, its barbs one by one, its margin
    and its shaft."""
    shape = feather.outline()
    rows, cols = window(shape)
    mine = (owner[rows, cols] == index) * 1.0
    if not mine.any():
        return
    small = feather.length < 60.0
    dark = np.clip((0.5 if small else 1.0) * shade[rows, cols] + turned(feather), 0.0, 1.0)
    # (a stroke has a width: keep its middle that far inside the feather)
    core = (filters.box_blur(mine, 1) > 0.9) * 1.0
    rim = core
    along = feather.shaft(1.0) - feather.shaft(0.0)
    angle = -math.degrees(math.atan2(along[1], along[0]))
    deep = colours(feather, 0.5, True, rng)[2]
    sheet.shade(shape, deep, Field(rim * np.clip(1.5 * dark, 0.0, 0.9), cols.start, rows.start), 3.0, angle + 20.0, hand=0.4)
    ys, xs = np.mgrid[rows, cols]
    middle = feather.shaft(0.55)
    tipward = ((xs - middle[0]) * along[0] + (ys - middle[1]) * along[1]) > 0
    lead_way = feather.across(along / float(np.hypot(*along)))
    leading = ((xs - middle[0]) * lead_way[0] + (ys - middle[1]) * lead_way[1]) > 0
    for stretch, u in ((False, 0.3), (True, 0.9)):  # the colour, laid solid, vane by vane
        for lead in (True, False):
            # (the leading vane catches the light)
            part = core * (tipward == stretch) * (leading == lead) * ((0.36 if lead else 0.58) + 0.22 * dark)
            pencil = colours(feather, u, lead, rng)[1]
            sheet.shade(shape, pencil, Field(part, cols.start, rows.start), 3.2, angle, hand=0.4)
    for path, u, lead in barbs(feather, rng):  # the barbs, one by one
        seen = showing(owner, index, path)
        if not seen.any():
            continue
        tone = min(tone_at(path, u, lead, shade, rng) * (0.6 if small else 1.0) + turned(feather), 1.0)
        set_, pick = colours(feather, u, lead, rng), rng.random()
        pencil = set_[0] if pick < 0.12 - 0.12 * tone else (set_[2] if pick > 0.88 - 0.5 * tone else set_[1])
        sheet.stroke(path, pencil, ((0.3 + 0.42 * tone) * seen).tolist(), 2.2)
    edge = shape + 1.4 * (feather.shaft(0.5) - shape) / np.maximum(np.hypot(*(feather.shaft(0.5) - shape).T), 1e-9)[:, None]
    seen = showing(owner, index, edge)
    if seen.sum() >= 3:  # its margin, a clean line
        sheet.stroke(edge, colours(feather, 0.6, True, rng)[1], (0.32 * seen).tolist(), 2.0)
    spine = feather.shaft(np.linspace(0.0, 1.0, 14))
    seen = showing(owner, index, spine)
    if seen.sum() >= 2:
        firm = 0.85 if feather.length > 80 else 0.25
        sheet.stroke(spine, QUILL, (firm * seen * np.linspace(1.0, 0.4, 14)).tolist(), 2.2)


# --------------------------------------------------------------------------
# beak, face and eye


def head_line(points, count: int = 14) -> np.ndarray:
    """A smooth open curve through points of the head's own measure, as
    ``count`` points on the sheet evenly along it."""
    corners = np.array([place(*head_point(x, y)) for x, y in points])
    dense = CatmullRom(corners).sample(1.0)[0]
    return dense[np.linspace(0, len(dense) - 1, count).astype(int)]


def draw_horn(sheet: Sheet, index: int, owner, top, bottom, dark: float) -> None:
    """A mandible: strokes that run its length, from its base to its tip,
    between its upper edge and its lower; black where it turns from the
    light, a band of light left along its ridge, paler toward its tip."""
    upper, lower = head_line(top, 18), head_line(bottom, 18)
    tipward = np.linspace(0.0, 1.0, 18)
    for pencil in (JET, SLATE, JET, HORN):
        for t in np.linspace(0.03, 0.97, 30):
            path = upper * (1.0 - t) + lower * t
            ridge = math.exp(-(((t - 0.2) / 0.09) ** 2))  # the light along the ridge
            depth = np.clip(dark + 0.7 * t - 0.75 * ridge - 0.3 * tipward**2, 0.0, 1.0)
            if pencil is JET:
                touch = 1.6 * (depth - 0.3)
            elif pencil is SLATE:
                touch = 0.35 + 0.5 * depth
            else:
                touch = 0.45 * (1.0 - depth)
            touch = np.clip(touch, 0.0, 1.0) * showing(owner, index, path)
            if touch.max() > 0.08:
                sheet.stroke(path, pencil, touch.tolist(), 3.0)


def face_row(row: float, count: int = 16) -> np.ndarray:
    """A line across the bare face, in the head's own measure: from the base
    of the beak back toward the ear, sagging round under the eye. ``row`` is
    0 at the brow and 1 at the jaw."""
    t = np.linspace(0.0, 1.0, count)
    x = 40.0 - (56.0 + 6.0 * row) * t
    y = 30.0 - 84.0 * row + (18.0 * row + 4.0) * np.sin(np.pi * t) * (2.0 * row - 0.9) * -1.0
    return np.stack([x, y], axis=1)


def draw_face(sheet: Sheet, index: int, owner, rng) -> None:
    """The bare face: pale skin in strokes that run back from the beak, a
    little warmer toward the jaw, and its lines of tiny dark feathers."""
    for row in np.linspace(0.0, 1.0, 34):
        line = np.array([place(*head_point(x, y)) for x, y in face_row(row)])
        seen = showing(owner, index, line)
        if seen.any():
            sheet.stroke(line, SKIN, (0.36 * seen).tolist(), 3.2)
            sheet.stroke(line, BLUSH, ((0.06 + 0.24 * row) * seen).tolist(), 3.2)
    for row in np.linspace(0.3, 0.97, 8):  # the lines of tiny feathers
        line = np.array([place(*head_point(x, y)) for x, y in face_row(row + rng.normal(0.0, 0.01), 40)])
        k = int(rng.integers(0, 3))
        while k < len(line) - 3:
            long = int(rng.integers(2, 4))
            dash = line[k : k + long + 1]
            clear = np.hypot(*(dash - place(*head_point(*EYE))).T).min() > 12.0 * SCALE
            if clear and showing(owner, index, dash).all():
                sheet.stroke(dash, WINE, float(rng.uniform(0.55, 0.85)), 2.0)
            k += long + int(rng.integers(2, 4))


def draw_eye(sheet: Sheet) -> None:
    """The eye: a dark rim, a pale yellow iris, the pupil, a point of light."""
    def ring(radius: float) -> np.ndarray:
        return np.array([place(*head_point(EYE[0] + radius * math.cos(t), EYE[1] + radius * math.sin(t)))
                         for t in np.linspace(0.0, 2 * math.pi, 18)])  # fmt: skip

    sheet.stroke(ring(8.5), SLATE, 0.7, 2.2)
    for radius in (6.6, 5.2):
        sheet.stroke(ring(radius), IRIS, 0.85, 2.4)
    for radius in (3.2, 1.6):
        for _ in range(2):
            sheet.stroke(ring(radius), JET, 1.0, 2.6)
    sheet.erase([place(*head_point(EYE[0] + 2.0, EYE[1] + 2.4))], tools.ERASER, 1.0, 2.5)


def draw_head(sheet: Sheet, owner, first: int, rng) -> None:
    """Beak, bare face and eye, over the feathers of the head."""
    draw_face(sheet, first, owner, rng)
    draw_horn(sheet, first + 1, owner, LOWER_BEAK[:4], (LOWER_BEAK[6], LOWER_BEAK[5], LOWER_BEAK[4], LOWER_BEAK[3]), 0.7)
    draw_horn(sheet, first + 2, owner, UPPER_BEAK[:7], UPPER_BEAK[:6:-1], 0.5)
    for points in (UPPER_BEAK, LOWER_BEAK):
        sheet.stroke(CatmullRom([place(*head_point(x, y)) for x, y in points], closed=True), JET, 0.6, 2.0)
    nostril = [place(*head_point(x, y)) for x, y in ((44, 24), (47, 21), (45, 17))]
    sheet.stroke(nostril, JET, 0.9, 2.4)
    draw_eye(sheet)


def main() -> None:
    rng = np.random.default_rng(7)
    feathers = plumage(rng)
    # put the bird in the middle of the sheet
    corners = np.concatenate([feather.outline(6) for feather in feathers] + [head_shape(UPPER_BEAK)])
    CENTRE[:] += 0.5 * np.array([WIDTH, HEIGHT]) - 0.5 * (corners.min(axis=0) + corners.max(axis=0))
    print("the bird spans", np.round(corners.max(axis=0) - corners.min(axis=0)))
    owner = np.full((HEIGHT, WIDTH), -1, np.int32)
    for index, feather in enumerate(feathers):
        fill(owner, feather.outline(), index)
    first = len(feathers)
    for k, points in enumerate((FACE, LOWER_BEAK, UPPER_BEAK)):
        fill(owner, head_shape(points), first + k)
    shade = shadows(owner)
    sheet = Sheet(WIDTH, HEIGHT, paper=PAPER, seed=7)
    for index, feather in enumerate(feathers):
        draw_feather(sheet, feather, index, owner, shade, rng)
    draw_head(sheet, owner, first, rng)
    out = Path(__file__).parent / "output" / "macaw.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}: {len(feathers)} feathers")


if __name__ == "__main__":
    main()
