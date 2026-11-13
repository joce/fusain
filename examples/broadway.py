"""Broadway in SoHo on an autumn afternoon, in soft pastel on toned paper.

Run from the repository root: ``python examples/broadway.py``. Writes
``examples/output/broadway.png``.

The street is built in space and then looked at; nothing is a shape to
fill. The eye stands in the roadway at a crossing and looks uptown, so the
street vanishes at one point. Each side is a row of cast-iron loft buildings,
each of its own width, height and colour: a tall ground floor of shop windows
between piers under a cornice; over it, storeys of tall windows between
slender columns, with a belt course at each floor; a frieze, and a heavy
cornice standing out over the street. Fire escapes zigzag down some fronts,
a wooden water tank shows over a roof, and where a building stands higher
than its neighbour its blank party wall shows. Trees grow from pits at the
kerb: a trunk, limbs that divide, a mass of leaves at the end of each, and
leaves fallen under them. Cabs come down the street; people walk on the
pavements and cross between the cars.

The sun is low, behind the left shoulder. From every point a ray is followed
toward it, past the buildings and through the foliage, and that tells what
is lit: the fronts on the right; on the left only what reaches above the
shadow of its own side (the edge of a cornice, a party wall, the top of a
tree); the roadway, but for the shadow of the tree at the corner behind the
eye and for the shadows the left side lays across it further on.

It is drawn as a pastellist works. The walls, the ground and the sky are
blocked in with the side of the stick, the sky and the far end of the street
are rubbed with a finger, and the sheet is fixed. Then each front is drawn
as it is built, in strokes that run with the perspective, its windows as
dabs in their rows; then the roadway, in strokes that run down the street,
and what is painted on it and lies on it; then the trees, in broken dabs,
the deep ones first and the light ones over them, the cars and the people,
the farthest first; and the brightest touches last. Every stroke takes the
colour of the light where it lands, and far things are paler and bluer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path

import numpy as np
from murex import Polyline, filters, noise, write_png

from fusain import Medium, Sheet, papers, pastels, tools

WIDTH, HEIGHT = 1000, 1180
SEED = 1873

#: The eye: where it stands across the street and how high, in metres (x is
#: across the street from its middle, y up, z along it away from the eye);
#: where on the sheet the street vanishes; the focal length in pixels.
EYE = (1.0, 1.65)
VANISH = (462.0, 800.0)
FOCAL = 900.0

#: The street, in metres: from its middle to the building line and to the
#: kerb, how far back a building goes, the length of a block and the width of
#: a cross street, where the first block starts, and the line of the trees.
HALF, KERB, DEEP = 12.0, 6.6, 30.0
BLOCK, CROSS, START = 115.0, 16.0, 3.0
TREE_LINE = 7.4
#: A front: the height of the ground floor, of a storey and of the frieze and
#: cornice; a window's share of its bay; how far under a floor line a window
#: ends and how far over it the next one starts.
GROUND_FLOOR, STOREY, ATTIC = 5.0, 3.9, 1.5
WINDOW, HEAD, SILL = 0.62, 0.55, 0.75

#: The sun: how far its light is turned from the line of the street and how
#: high it stands.
SUN_TURN, SUN_RISE = math.radians(18.0), math.radians(27.0)
#: What a ray gains for each metre it travels over the ground toward the sun
#: (x, y, z), and the unit vector toward the sun.
TO_SUN = np.array([-math.sin(SUN_TURN), math.tan(SUN_RISE), -math.cos(SUN_TURN)])
SUN = TO_SUN / np.linalg.norm(TO_SUN)
#: Where the shadow of a point one metre up falls on the ground (x, y, z).
CAST = np.array([-TO_SUN[0], 0.0, -TO_SUN[2]]) / TO_SUN[1]

#: What is seen at a pixel that no building covers.
SKY, ROAD, WALK = 0, 1, 2
GROUND = (ROAD, WALK)
#: The widest mark of the end of a stick, in pixels, and the narrowest; the
#: side of a stick marks a band this many times wider than its end.
STICK, THIN, SIDE = 13.0, 2.2, 5.0
#: Points lit at a time (bounds the working memory).
CHUNK = 8000

PAPER = papers.TONED_GREY.with_(color="#8f8274", name="toned umber")

Colour = np.ndarray
Tones = tuple[Colour, Colour]


def rgb(code: str) -> Colour:
    """The colour ``#rrggbb`` as three numbers in 0..1."""
    return np.array([int(code[k : k + 2], 16) for k in (1, 3, 5)]) / 255.0


def tones(sun: str, shade: str) -> Tones:
    """A thing's colour in the sun and in the shade."""
    return rgb(sun), rgb(shade)


def mix(a: Colour, b: Colour, share: float) -> Colour:
    return (1.0 - share) * a + share * b


def both(pair: Tones, other: Colour, share: float) -> Tones:
    """Both colours of a pair, mixed with another."""
    return mix(pair[0], other, share), mix(pair[1], other, share)


#: The colour things fade to with distance, and how fast (metres).
HAZE, REACH = rgb("#cbd0dc"), 430.0
INK, WHITE = rgb("#23202c"), rgb("#fbf4e2")

#: Walls, in the sun and in the shade.
WALLS = {
    "cream": tones("#f7d99c", "#6b7399"),
    "white": tones("#f8e8c4", "#7a82aa"),
    "grey": tones("#d8c2a0", "#565e7e"),
    "brick": tones("#dc6a3a", "#5e3c50"),
    "ochre": tones("#eaa844", "#6e6562"),
    "rose": tones("#e69a70", "#6a5068"),
    "stone": tones("#c48e5e", "#50465c"),
}
PARTY_WALL = tones("#d67c4c", "#5e4252")
#: What the first tone of a wall leans to: deeper than it will end.
UNDER = tones("#c8844a", "#343a6c")
#: Glass on each side of the street, and what it mirrors: on the shaded side
#: the lit fronts opposite or the sky, on the lit side the sky or the shade.
GLASS = {-1: tones("#343850", "#22263e"), 1: tones("#464c62", "#2c3046")}
MIRRORED = {
    -1: (tones("#f0c08c", "#e6b284"), tones("#c4d2e2", "#a4b6d0")),
    1: (tones("#bed0e2", "#8e9cba"), tones("#77829e", "#566080")),
}
BLIND = tones("#eadcc0", "#8a90aa")
SHOP_LIGHT = tones("#e8c078", "#eab868")
STREET_MIRRORED = tones("#b49c80", "#4e5a7c")
#: The boards over the shop windows, and the cloth of the awnings.
FASCIAS = (
    tones("#3a3a40", "#25253a"),
    tones("#5c7460", "#34464c"),
    tones("#a8443a", "#5c2c40"),
    tones("#e8dcc0", "#8288a6"),
    tones("#40507c", "#283258"),
)
IRON = tones("#4a4040", "#1f1e2c")
ROADWAY = tones("#d0b78e", "#575f88")
PAVEMENT = tones("#ecd9b0", "#6f77a0")
PAINT = tones("#f6ecd6", "#9aa0c6")
KERBSTONE = tones("#f4e8cc", "#464c74")
EARTH = tones("#6e543a", "#3c3444")
BARK = tones("#a48c6c", "#3a3442")
TANK = tones("#b47e50", "#5c4c58")
SKIN = tones("#e8b890", "#8a6c74")
HYDRANT = tones("#a83c30", "#5e2c3c")
SIGNAL = tones("#f0c020", "#a08a3c")
#: The sky, from overhead down to the end of the street, and its clouds.
SKY_HIGH, SKY_MID, SKY_LOW = rgb("#b2c7dc"), rgb("#d3dde2"), rgb("#f4e9cf")
CLOUD = rgb("#f8f2e4")

#: Leaves: light, middle and deep, in the sun and then in the shade.
LEAVES = {
    "gold": ("#f9d640", "#e6ac20", "#9c6c1c", "#b8a04c", "#8a7a3e", "#565036"),
    "orange": ("#f8a630", "#e4721e", "#9e3e1a", "#b06c42", "#884e3c", "#58383a"),
    "red": ("#ec5c30", "#c63220", "#7e1e1c", "#9c4046", "#742c3e", "#4a2436"),
    "russet": ("#d0b440", "#a08c30", "#5e5a26", "#80845a", "#5e664a", "#3e463c"),
}
#: Cars and coats, in the sun and in the shade.
PAINTS = {
    "cab": tones("#f6c518", "#a48a3c"),
    "black": tones("#4c4848", "#24243a"),
    "silver": tones("#dcd8cc", "#7880a0"),
    "red": tones("#b8402c", "#683044"),
    "white": tones("#f4eee0", "#8e96b8"),
    "blue": tones("#46628e", "#2a3860"),
    "green": tones("#4e6c54", "#2c4046"),
}
COATS = {
    "red": tones("#cc382c", "#702c40"),
    "navy": tones("#3e4c70", "#242a46"),
    "camel": tones("#d4a670", "#7a6668"),
    "black": tones("#3e3a3c", "#201e30"),
    "charcoal": tones("#5c5a5c", "#2e2e3e"),
    "olive": tones("#76703e", "#3c4034"),
    "brown": tones("#8a5a3a", "#463242"),
    "plum": tones("#7a3c58", "#3e2444"),
    "teal": tones("#3c7c84", "#28485e"),
    "cream": tones("#ece0c4", "#868aa6"),
    "mustard": tones("#d29a34", "#76603e"),
}
#: Trousers, skirts and stockings; hair; skin; what people carry.
LEGS = {
    "dark": tones("#3e3e48", "#20202e"),
    "black": tones("#2e2c30", "#18182a"),
    "jeans": tones("#5a74a0", "#303c60"),
    "khaki": tones("#c4ac84", "#666274"),
    "grey": tones("#9c9894", "#545870"),
}
HAIR = {
    "black": tones("#2c2628", "#16141e"),
    "dark": tones("#4a3428", "#221c24"),
    "brown": tones("#86562e", "#3e2c2e"),
    "fair": tones("#dcb470", "#7c6c5a"),
    "grey": tones("#c4beb4", "#6c6c7c"),
}
SKINS = {
    "fair": tones("#ecbc94", "#8c6e76"),
    "olive": tones("#cc9468", "#6e5058"),
    "brown": tones("#8e5c40", "#4a343e"),
}
LEATHER = tones("#7a4a2c", "#3a2a32")
CANVAS = tones("#eadcbc", "#878aa4")
SHOE = rgb("#2a2228")


def far(colour: Colour, z: float) -> Colour:
    """A colour seen ``z`` metres away: paler and bluer with distance."""
    return mix(colour, HAZE, 1.0 - math.exp(-max(z - 30.0, 0.0) / REACH))


def distant(pair: Tones, z: float) -> Tones:
    return far(pair[0], z), far(pair[1], z)


def shadowed(pair: Tones) -> Tones:
    """The colours of a wall where a ledge, a cornice or a balcony close by
    shades it: on a sunlit wall a warm half-tone, on a shaded one deeper."""
    return mix(pair[1], pair[0], 0.3), mix(pair[1], INK, 0.3)


def raised(pair: Tones, share: float) -> Tones:
    """The colours of an edge that catches more light than the wall it is
    on: the sun's if the sun reaches it, the sky's if not."""
    return mix(pair[0], WHITE, share), mix(pair[1], HAZE, 0.5 * share)


@cache
def _stick(code: int, hard: bool) -> Medium:
    name = f"#{code:06x}"
    return pastels.hard(name) if hard else pastels.soft(name)


def stick(colour: Colour, hard: bool = False) -> Medium:
    """The pastel of the box nearest to a colour (the box holds 43 steps of
    each of red, green and blue)."""
    r, g, b = (int(round(float(c) * 42.5)) * 6 for c in np.clip(colour, 0.0, 1.0))
    return _stick((r << 16) | (g << 8) | b, hard)


# --------------------------------------------------------------------------
# the eye


def project(points) -> np.ndarray:
    """Where on the sheet the eye sees points of the street, (..., 3) to
    (..., 2)."""
    p = np.asarray(points, dtype=np.float64)
    z = np.maximum(p[..., 2], 0.5)
    x = VANISH[0] + FOCAL * (p[..., 0] - EYE[0]) / z
    y = VANISH[1] - FOCAL * (p[..., 1] - EYE[1]) / z
    return np.stack([x, y], axis=-1)


def rise(z: float) -> float:
    """How many pixels a metre is, ``z`` metres away."""
    return FOCAL / z


#: How far along the street each building line enters the sheet.
NEAR = {
    -1: 0.96 * FOCAL * (HALF + EYE[0]) / VANISH[0],
    1: 0.96 * FOCAL * (HALF - EYE[0]) / (WIDTH - VANISH[0]),
}


def level(x0: float, x1: float, y: float, z: float) -> tuple:
    """The two ends of a level line that runs across the street."""
    return (x0, y, z), (x1, y, z)


def lengthwise(x: float, y: float, z0: float, z1: float) -> tuple:
    """The two ends of a level line that runs along the street."""
    return (x, y, z0), (x, y, z1)


# --------------------------------------------------------------------------
# the street


@dataclass(frozen=True)
class Building:
    """A loft building: which ``side`` of the street it is on (-1 left, 1
    right), from where to where along it, its storeys, the colour of its
    wall, how many bays wide it is, and its ``marks``: ``a`` for arched
    windows, ``e`` for a fire escape, ``t`` for a water tank on the roof,
    ``w`` for awnings."""

    side: int
    z0: float
    z1: float
    storeys: int
    wall: str
    bays: int
    marks: str = ""

    @property
    def height(self) -> float:
        return GROUND_FLOOR + (self.storeys - 1) * STOREY + ATTIC

    @property
    def bay(self) -> float:
        return (self.z1 - self.z0) / self.bays

    @property
    def box(self) -> tuple[float, float, float, float, float]:
        """Its volume: x from, x to, z from, z to, height."""
        x = sorted((self.side * HALF, self.side * (HALF + DEEP)))
        return (x[0], x[1], self.z0, self.z1, self.height)

    @property
    def span(self) -> tuple[float, float]:
        """The stretch of its front that is on the sheet."""
        return (max(self.z0, NEAR[self.side]), self.z1)

    @property
    def on_corner(self) -> bool:
        """Whether it is the first of its block, on a cross street."""
        return (self.z0 - START) % (BLOCK + CROSS) < 1.0

    def floors(self) -> list[float]:
        """The heights of its floor lines, from the top of the ground floor
        to the foot of the frieze."""
        return [GROUND_FLOOR + k * STOREY for k in range(self.storeys)]

    def point(self, z: float, y: float, out: float = 0.0) -> np.ndarray:
        """The point of its front ``z`` along the street and ``y`` up,
        standing ``out`` metres proud of the wall."""
        return np.array([self.side * (HALF - out), y, z])

    def edge(self, near: float, away: float, y: float, out: float = 0.0) -> tuple:
        """The two ends of a level line along its front."""
        return self.point(near, y, out), self.point(away, y, out)

    def upright(self, z: float, low: float, high: float, out: float = 0.0) -> tuple:
        """The two ends of an upright line on its front, the top first."""
        return self.point(z, high, out), self.point(z, low, out)

    def run(self, z: float) -> float:
        """How many pixels a metre along its front is, at ``z``."""
        return FOCAL * abs(self.side * HALF - EYE[0]) / z**2


#: The first block on each side, from the corner on: width in metres,
#: storeys, wall, bays and marks.
LEFT_BLOCK = (
    (27.0, 4, "brick", 10, ""),
    (15.0, 6, "cream", 6, "a"),
    (15.0, 5, "grey", 6, "e"),
    (22.0, 6, "ochre", 9, "at"),
    (18.0, 5, "brick", 7, "e"),
    (18.0, 6, "white", 7, "a"),
)
RIGHT_BLOCK = (
    (30.0, 6, "cream", 12, "aw"),
    (15.0, 5, "brick", 6, "e"),
    (22.0, 6, "white", 9, "at"),
    (15.0, 7, "ochre", 6, "w"),
    (18.0, 5, "grey", 7, "e"),
    (15.0, 6, "rose", 6, "a"),
)
#: Widths a further block is made of, in metres.
PLOTS = (23.0, 15.0, 15.0, 8.0, 23.0, 16.0, 15.0)


def terrace(side: int, start: float, plan) -> list[Building]:
    """The buildings of one block, side by side from ``start``."""
    row = []
    for wide, storeys, wall, bays, marks in plan:
        row.append(Building(side, start, start + wide, storeys, wall, bays, marks))
        start += wide
    return row


def further_block(rng: np.random.Generator) -> list[tuple]:
    """The plan of a block further up the street: lofts of four to seven
    storeys, now and then a taller one."""
    walls = list(WALLS)
    plan = []
    for wide in rng.permutation(PLOTS):
        storeys = int(rng.choice((4, 5, 5, 6, 6, 6, 7, 11)))
        wall = walls[int(rng.integers(len(walls)))]
        plan.append((float(wide), storeys, wall, max(int(wide / 2.5), 2), ""))
    return plan


def street(rng: np.random.Generator) -> list[Building]:
    """Every building of the street: the blocks ahead on both sides, and the
    two corners behind the eye, which only cast shadows (the left one low,
    so that the sun comes over it)."""
    rows = []
    for side, first in ((-1, LEFT_BLOCK), (1, RIGHT_BLOCK)):
        rows += terrace(side, START, first)
        start = START + BLOCK + CROSS
        while start < 1300.0:
            rows += terrace(side, start, further_block(rng))
            start += BLOCK + CROSS
        behind = START - CROSS
        storeys = 2 if side < 0 else 6
        rows.append(Building(side, behind - BLOCK, behind, storeys, "brick", 40))
    return rows


def crossing(z) -> np.ndarray:
    """Whether a cross street's roadway runs ``z`` along the street."""
    along = (np.asarray(z) - START) % (BLOCK + CROSS)
    return (along > BLOCK + 3.5) & (along < BLOCK + CROSS - 3.5)


@dataclass(frozen=True)
class Tree:
    """A street tree: its side of the street, where along it, its height in
    metres, the colour its leaves have turned and the seed it grew from."""

    side: int
    z: float
    tall: float
    leaf: str
    seed: int

    @property
    def x(self) -> float:
        return self.side * TREE_LINE


#: A tree behind the eye, at the corner: only its shadow and its fallen
#: leaves are in the picture.
OFFSTAGE = Tree(-1, -14.0, 13.0, "gold", 5)
#: The trees of the first block: where along the street, height and leaves.
LEFT_TREES = (
    (24.0, 12.5, "gold"),
    (37.0, 11.5, "orange"),
    (50.0, 11.0, "red"),
    (63.0, 12.0, "gold"),
    (77.0, 10.0, "russet"),
    (91.0, 11.0, "orange"),
    (105.0, 10.0, "gold"),
)
RIGHT_TREES = (
    (30.0, 9.5, "orange"),
    (43.5, 8.5, "gold"),
    (57.0, 10.0, "red"),
    (71.0, 9.0, "gold"),
    (85.0, 9.5, "orange"),
    (99.0, 9.0, "russet"),
    (111.0, 9.5, "red"),
)


def avenue(rng: np.random.Generator) -> list[Tree]:
    """The trees along both kerbs: those of the first block, then one every
    dozen metres or so up the street, none in a crossing."""
    trees = []
    kinds = ("gold", "gold", "orange", "orange", "red", "russet")
    for side, first in ((-1, LEFT_TREES), (1, RIGHT_TREES)):
        for z, tall, leaf in first:
            trees.append(Tree(side, z, tall, leaf, int(rng.integers(1 << 30))))
        z = START + BLOCK + CROSS + 6.0
        while z < 700.0:
            if (z - START) % (BLOCK + CROSS) < BLOCK - 4.0:
                kind = kinds[int(rng.integers(len(kinds)))]
                tall = float(rng.uniform(8.5, 12.0))
                trees.append(Tree(side, z, tall, kind, int(rng.integers(1 << 30))))
            z += float(rng.uniform(10.0, 16.0))
    return trees


def aim(turn: float, lift: float) -> np.ndarray:
    """The unit vector turned ``turn`` round the upright and lifted ``lift``
    from the level (radians)."""
    flat = math.cos(lift)
    return np.array([flat * math.cos(turn), math.sin(lift), flat * math.sin(turn)])


def grow(tree: Tree) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], np.ndarray]:
    """A tree as it grew: its limbs (from, to, thickness in metres), the
    trunk first, and the masses of its leaves (x, y, z, radius), one at the
    end of each limb and one at the top."""
    rng = np.random.default_rng(tree.seed)
    tall = tree.tall
    squeeze = np.array([0.7, 1.0, 0.7])  # (a street tree is pruned narrow)
    base = np.array([tree.x, 0.0, tree.z])
    fork = base + np.array([rng.normal(0.0, 0.12), 0.32 * tall, rng.normal(0.0, 0.12)])
    limbs = [(base, fork, 0.03 * tall)]
    masses = [(*(fork + np.array([0.0, 0.5 * tall, 0.0])), 0.17 * tall)]
    first = rng.uniform(0.0, 2.0 * math.pi)
    for k in range(3):
        turn = first + 2.0 * math.pi * k / 3.0 + rng.normal(0.0, 0.3)
        reach = 0.3 * tall * aim(turn, math.radians(rng.uniform(55.0, 70.0)))
        tip = fork + squeeze * reach
        limbs.append((fork, tip, 0.016 * tall))
        masses.append((*tip, 0.16 * tall))
        for way in (-1.0, 1.0):
            off = turn + way * rng.uniform(0.5, 1.0)
            reach = 0.2 * tall * aim(off, math.radians(rng.uniform(30.0, 55.0)))
            end = tip + squeeze * reach
            limbs.append((tip, end, 0.009 * tall))
            masses.append((*end, rng.uniform(0.14, 0.19) * tall))
    return limbs, np.array(masses)


# --------------------------------------------------------------------------
# the light


def sunlit(boxes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """1 where the sun reaches each of ``points`` (N, 3) and 0 where a
    building stands in its way. ``boxes`` are the buildings' volumes, (B, 5)."""
    across, along = -TO_SUN[0], -TO_SUN[2]
    out = np.empty(len(points))
    for start in range(0, len(points), CHUNK):
        p = points[start : start + CHUNK, None, :]
        # how far toward the sun the ray enters each volume, and leaves it
        enter = np.maximum((p[..., 0] - boxes[:, 1]) / across, 0.0)
        enter = np.maximum(enter, (p[..., 2] - boxes[:, 3]) / along)
        leave = (p[..., 0] - boxes[:, 0]) / across
        leave = np.minimum(leave, (p[..., 2] - boxes[:, 2]) / along)
        under = p[..., 1] + enter * TO_SUN[1] < boxes[:, 4]
        out[start : start + CHUNK] = ~((enter < leave - 1e-6) & under).any(axis=1)
    return out


def leaf_shade(masses: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The share of the sun that foliage keeps from each of ``points``
    (N, 3), 0..1. ``masses`` are balls of leaves, (M, 4): x, y, z, radius."""
    through = np.ones(len(points))
    for start in range(0, len(points), CHUNK):
        to = masses[None, :, :3] - points[start : start + CHUNK, None, :]
        along = to @ SUN
        aside = np.einsum("nmk,nmk->nm", to, to) - along**2
        dense = np.clip(1.0 - aside / masses[:, 3] ** 2, 0.0, 1.0) * (along > 0)
        through[start : start + CHUNK] = np.prod(1.0 - 0.8 * np.sqrt(dense), axis=1)
    return 1.0 - through


def front(index: int) -> int:
    """What the scene calls the street front of a building."""
    return 10 + 2 * index


def gable(index: int) -> int:
    """What the scene calls the wall of a building that faces down the
    street: a party wall above a lower neighbour, or a front on a cross
    street."""
    return 11 + 2 * index


class Scene:
    """What the eye sees at each pixel of the sheet (``seen``: sky, roadway,
    pavement, or the front or the gable of a building), how far away it is
    (``depth``) and how much of the sun reaches it (``lit``, 0..1)."""

    def __init__(self, buildings: list[Building], masses, rng: np.random.Generator):
        self.buildings = buildings
        self.boxes = np.array([b.box for b in buildings])
        col, row = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
        self.u, self.v = (col - VANISH[0]) / FOCAL, (VANISH[1] - row) / FOCAL
        self.seen = np.full((HEIGHT, WIDTH), SKY, np.int16)
        self.depth = np.full((HEIGHT, WIDTH), np.inf)
        self._fronts()
        self._gables()
        self._ground()
        self.lit = self._light(masses, rng)

    def _fronts(self) -> None:
        for side in (-1, 1):
            facing = side * self.u > 0
            reach = (side * HALF - EYE[0]) / np.where(facing, self.u, 1.0)
            reach = np.where(facing, reach, np.inf)
            height = EYE[1] + reach * self.v
            for k, b in enumerate(self.buildings):
                if b.side != side or b.z1 <= 1.0:
                    continue
                hit = (reach >= b.z0) & (reach < b.z1) & (height <= b.height)
                self.seen[hit] = front(k)
                self.depth[hit] = reach[hit]

    def _gables(self) -> None:
        for k, b in enumerate(self.buildings):
            if b.z0 <= 1.0:
                continue
            x = b.side * (EYE[0] + b.z0 * self.u)
            height = EYE[1] + b.z0 * self.v
            hit = (x >= HALF) & (x <= HALF + DEEP) & (height >= 0.0)
            hit &= (height <= b.height) & (self.depth > b.z0)
            self.seen[hit] = gable(k)
            self.depth[hit] = b.z0

    def _ground(self) -> None:
        reach = np.where(self.v < 0, -EYE[1] / np.minimum(self.v, -1e-9), np.inf)
        hit = reach < self.depth
        x = EYE[0] + reach * self.u
        road = (np.abs(x) < KERB) | crossing(np.where(hit, reach, 0.0))
        self.seen[hit] = np.where(road, ROAD, WALK)[hit]
        self.depth[hit] = reach[hit]

    def _light(self, masses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        solid = self.seen != SKY
        depth = self.depth[solid]
        x, y = EYE[0] + depth * self.u[solid], EYE[1] + depth * self.v[solid]
        points = np.stack([x, np.maximum(y, 0.0), depth], axis=1)
        sun = sunlit(self.boxes, points)
        under = (sun > 0) & (depth < 160.0)
        spot = points[under]
        # (leaves let flecks of sun through, drawn out the way the light runs)
        across = (spot[:, 0] + 0.6 * spot[:, 1]) / 0.3
        along = (spot[:, 2] + 1.9 * spot[:, 1]) / 0.7
        flecks = filters.smoothstep(0.22, 0.5, noise.lattice_noise(across, along, rng))
        sun[under] *= 1.0 - leaf_shade(masses, spot) * flecks
        lit = np.zeros((HEIGHT, WIDTH))
        lit[solid] = sun
        return filters.box_blur(lit, 1)


# --------------------------------------------------------------------------
# what stands and moves in the street


@dataclass(frozen=True)
class Car:
    """A car coming down the street: where its middle is across the street,
    where its nose is along it, its paint, whether it is a cab, its size."""

    x: float
    z: float
    paint: str
    cab: bool = False
    tall: float = 1.45
    wide: float = 1.85
    long: float = 4.7


@dataclass(frozen=True)
class Figure:
    """Someone in the street: where they stand (``x`` across the street,
    ``z`` along it), what they wear (the ``coat``, its ``hem`` as a share of
    their height, the ``legs``, a ``scarf``), which way they face
    (``toward`` the eye, ``away`` from it, or crossing to its ``left`` or
    ``right``), where they are in a stride (``step``, -1 to 1: its sign
    says which leg is ahead, 0 is standing on both), how tall and how broad
    (``build``), their ``hair`` and its ``cut``, their ``skin``, how far
    head and shoulders lean (``lean``, metres, toward the eye's right if
    more than 0), and what they are doing with their hands: ``swing``,
    ``pockets``, a ``bag`` across the body, a ``tote`` in a hand, a
    ``phone`` at the ear, or holding a ``child`` by the hand."""

    x: float
    z: float
    coat: str
    legs: str = "dark"
    facing: str = "toward"
    step: float = 0.6
    tall: float = 1.74
    build: float = 1.0
    hem: float = 0.42
    hair: str = "dark"
    cut: str = "short"
    skin: str = "fair"
    hands: str = "swing"
    lean: float = 0.0
    scarf: str = ""

    @property
    def across(self) -> float:
        """1 if they walk to the eye's right, -1 to its left, 0 if they come
        toward it or go away from it."""
        return {"left": -1.0, "right": 1.0}.get(self.facing, 0.0)


#: A standing adult is this many heads tall.
HEADS = 7.5

Point = tuple[float, float]
Limb = tuple[Point, Point, Point]


@dataclass(frozen=True)
class Frame:
    """Where the parts of a figure are, in metres, in the upright plane it
    stands in: ``u`` across the sheet from its feet, ``y`` up. The head is
    an oval whose top is at ``top``, ``head`` high, its middle at ``face``;
    the collar is at ``neck``; the coat is ``chest`` wide at the shoulders
    and ``skirt`` wide at its ``hem``; each limb runs from where it joins
    the body through its joint to its end (hip, knee, ankle; shoulder,
    elbow, wrist), the one behind first."""

    top: float
    head: float
    face: float
    neck: float
    chest: float
    hem: float
    skirt: float
    legs: tuple[Limb, ...]
    arms: tuple[Limb, ...]


def front_legs(fig: Figure) -> tuple[Limb, ...]:
    """The legs of someone seen from in front or behind: the one ahead
    planted straight, the other behind with its heel up, so that its foot
    shows a little higher and nearer the middle."""
    t, b = fig.tall, fig.build
    legs = []
    for side in (-1.0, 1.0):
        lift = 0.0 if side * fig.step >= 0.0 else abs(fig.step)
        hip = (side * 0.075 * b, 0.53 * t)
        knee = (side * (0.085 * b - 0.02 * lift), 0.28 * t + 0.05 * lift)
        foot = (side * (0.1 * b - 0.04 * lift), 0.03 + 0.08 * lift)
        legs.append((hip, knee, foot))
    return tuple(sorted(legs, key=lambda leg: -leg[2][1]))


def side_legs(fig: Figure) -> tuple[Limb, ...]:
    """The legs of someone seen from the side: a scissor, the leg ahead
    reaching forward to the heel, the one behind pushing off the toe; or,
    standing, one foot a little before the other."""
    t, way = fig.tall, fig.across
    reach = 0.2 * t * abs(fig.step) + 0.05
    hip = (0.0, 0.53 * t)
    ahead = (hip, (way * 0.5 * reach, 0.28 * t), (way * reach, 0.03))
    lift = abs(fig.step)
    knee = (-way * 0.2 * reach, 0.28 * t + 0.02 * lift)
    behind = (hip, knee, (-way * reach, 0.03 + 0.09 * lift))
    return (behind, ahead)


def front_arms(fig: Figure, chest: float) -> tuple[Limb, ...]:
    """The arms of someone seen from in front or behind: hanging at the
    sides and swinging against the legs, hands in the pockets, one hand on
    the strap of a bag, at the ear, or reaching down to a child's."""
    t = fig.tall
    arms = []
    for side in (-1.0, 1.0):
        root = (side * (chest - 0.06), fig.tall - 1.55 * t / HEADS)
        swing = abs(fig.step) * (side * fig.step < 0.0)
        elbow = (side * (chest + 0.015), 0.63 * t + 0.03 * swing)
        hand = (side * (chest - 0.01 - 0.05 * swing), 0.45 * t + 0.08 * swing)
        held = fig.hands if side < 0 or fig.hands == "pockets" else "swing"
        if held == "pockets":
            elbow, hand = (side * (chest + 0.04), 0.64 * t), (side * 0.15, 0.5 * t)
        elif held == "bag":
            elbow, hand = (side * (chest + 0.03), 0.66 * t), (side * 0.12, 0.7 * t)
        elif held == "phone":
            elbow = (side * (chest + 0.05), 0.7 * t)
            hand = (side * 0.1, t - 0.65 * t / HEADS)
        elif held == "child":
            elbow, hand = (side * (chest + 0.04), 0.62 * t), (side * (chest + 0.1), 0.44 * t)
        arms.append((root, elbow, hand))
    return tuple(arms)


def side_arms(fig: Figure, chest: float) -> tuple[Limb, ...]:
    """The arms of someone seen from the side: one swinging forward, the
    other back; a hand in a pocket stays by the hip."""
    t, way = fig.tall, fig.across
    root = (0.0, t - 1.6 * t / HEADS)
    swing = 0.12 * t * abs(fig.step)
    arms = []
    for k in (-1.0, 1.0):
        if fig.hands == "pockets":
            elbow, hand = (-way * (chest + 0.02), 0.64 * t), (0.0, 0.5 * t)
        else:
            elbow = (k * way * 0.4 * swing, 0.63 * t)
            hand = (k * way * (swing + 0.02 * k), 0.45 * t + 0.25 * swing)
        arms.append((root, elbow, hand))
    return tuple(arms)


def frame(fig: Figure) -> Frame:
    """The body of a figure, from its height, its build and its pose: seven
    and a half heads tall, the shoulders two heads wide, the legs from half
    the height."""
    t, b = fig.tall, fig.build
    head = t / HEADS
    if fig.across:
        chest, skirt = 0.13 * b, 0.15 * b
        legs, arms = side_legs(fig), side_arms(fig, chest)
    else:
        chest, skirt = 0.22 * b, 0.18 * b
        legs, arms = front_legs(fig), front_arms(fig, chest)
    face = fig.lean + 0.025 * fig.across
    return Frame(t, head, face, t - 1.2 * head, chest, fig.hem * t, skirt, legs, arms)


@dataclass(frozen=True)
class Lamp:
    """A street lamp: a post at the kerb with an arm over the roadway."""

    side: int
    z: float
    tall: float = 8.5


@dataclass(frozen=True)
class Hydrant:
    """A fire hydrant on the pavement."""

    x: float
    z: float


@dataclass(frozen=True)
class Signal:
    """A traffic signal at a crossing: a post at the corner, an arm over the
    roadway and the yellow box of lamps hung from it."""

    side: int
    z: float


CARS = (
    Car(2.8, 15.0, "cab", True),
    Car(0.0, 33.0, "black"),
    Car(-3.0, 38.0, "cab", True),
    Car(2.9, 52.0, "cab", True),
    Car(-0.1, 66.0, "silver"),
    Car(-2.8, 80.0, "cab", True),
    Car(2.9, 92.0, "white", tall=2.4),
    Car(0.0, 124.0, "cab", True),
    Car(-5.4, 19.5, "blue"),
    Car(-5.4, 27.0, "silver"),
    Car(-5.4, 42.0, "red"),
    Car(-5.4, 55.0, "black"),
    Car(-5.4, 68.0, "white"),
    Car(5.4, 25.5, "silver"),
    Car(5.4, 37.0, "green"),
    Car(5.4, 48.0, "white", tall=2.3),
    Car(5.4, 62.0, "black"),
    Car(5.4, 75.0, "red"),
)
#: The people: in the roadway, a woman in red crossing with a man a step
#: behind; on the sunny pavement a couple coming down the street, someone
#: waiting at the kerb, two talking, someone with a child; in the shade of
#: the other pavement, people coming and going, someone on the telephone;
#: and a few far off, crossing at the next corner.
CROWD = (
    Figure(0.75, 21.3, "red", "black", "right", 0.8, 1.7, 0.95, 0.4, "black", "long", hands="bag"),
    Figure(-0.3, 22.3, "navy", "grey", "right", -0.7, 1.82, 1.05, 0.47, "dark"),
    Figure(7.9, 17.6, "navy", "jeans", "toward", 0.7, 1.8, 1.05, 0.52, "dark", hands="pockets"),
    Figure(8.55, 18.2, "camel", "dark", "toward", -0.6, 1.66, 0.92, 0.34, "brown", "long", hands="bag", scarf="#b8402c"),
    Figure(6.95, 24.0, "olive", "khaki", "left", 0.0, 1.76, hands="pockets", hem=0.5),
    Figure(10.15, 27.0, "black", "grey", "right", 0.0, 1.68, 0.92, 0.34, "fair", "long", lean=0.03, hands="tote"),
    Figure(10.75, 27.2, "brown", "jeans", "left", 0.0, 1.82, 1.05, 0.48, "grey", lean=-0.03),
    Figure(9.4, 35.5, "plum", "black", "away", 0.7, 1.68, 0.92, 0.38, "dark", "long", hands="tote"),
    Figure(8.45, 44.0, "teal", "dark", "toward", 0.5, 1.74, hem=0.4, hands="child"),
    Figure(7.95, 44.1, "mustard", "jeans", "toward", -0.5, 1.12, 0.8, 0.45, "fair"),
    Figure(9.6, 53.0, "cream", "dark", "away", 0.6, 1.7, 0.95, 0.36, "brown", "long"),
    Figure(10.2, 53.6, "charcoal", "jeans", "away", -0.6, 1.8, 1.05, 0.48, "black"),
    Figure(8.9, 66.0, "red", "dark", "toward", 0.6, 1.7, hem=0.38),
    Figure(9.4, 80.0, "navy", "khaki", "away", 0.6),
    Figure(10.4, 84.0, "camel", "dark", "toward", -0.6),
    Figure(8.8, 96.0, "black", "jeans", "toward", 0.6),
    Figure(9.8, 104.0, "cream", "dark", "away", 0.6),
    Figure(-8.7, 23.0, "charcoal", "dark", "toward", 0.6, 1.8, 1.05, 0.45, "black", hands="pockets"),
    Figure(-9.3, 31.0, "camel", "black", "away", 0.6, 1.68, 0.92, 0.36, "fair", "long", hands="bag"),
    Figure(-9.9, 31.6, "navy", "dark", "away", -0.6, 1.8, 1.05, 0.46, "dark"),
    Figure(-7.6, 39.5, "cream", "grey", "right", 0.0, 1.72, hem=0.34, hair="brown", cut="long", hands="phone"),
    Figure(-9.6, 46.5, "olive", "jeans", "toward", 0.6, 1.78, hem=0.48),
    Figure(-8.8, 58.5, "black", "dark", "away", 0.6, 1.76),
    Figure(-9.4, 59.2, "teal", "black", "away", -0.6, 1.66, 0.92, 0.36, cut="long"),
    Figure(-10.2, 72.0, "navy", "khaki", "toward", 0.6),
    Figure(-8.4, 88.0, "camel", "dark", "toward", -0.6),
    Figure(-9.6, 100.0, "black", "dark", "away", 0.6),
    Figure(-3.0, 121.0, "black", "dark", "right", 0.6),
    Figure(1.5, 122.0, "camel", "dark", "left", 0.6),
    Figure(4.0, 121.5, "navy", "dark", "left", -0.6),
)
#: (The first lamp stands behind the eye: only its shadow is in the picture.)
LAMPS = (Lamp(-1, 1.5), Lamp(1, 20.5), Lamp(-1, 58.0), Lamp(1, 78.0))
HYDRANTS = (Hydrant(7.5, 14.0), Hydrant(-7.4, 31.0))
SIGNALS = (Signal(-1, 135.5), Signal(1, 119.0))


# --------------------------------------------------------------------------
# drawing


def inside(path: np.ndarray) -> np.ndarray:
    """1 where the points of a path are on the sheet."""
    x, y = path[:, 0], path[:, 1]
    return ((x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)) * 1.0


def reaches(lo: float, hi: float, rng: np.random.Generator) -> list[tuple]:
    """The stretch of street from ``lo`` to ``hi`` cut into lengths that each
    look about the same size from end to end, so that a stroke along one can
    keep one width; no two cuts quite alike."""
    count = max(math.ceil(math.log(hi / lo) / math.log(1.28)), 1)
    steps = (np.arange(count + 1) + rng.uniform(-0.25, 0.25, count + 1)) / count
    steps[0], steps[-1] = 0.0, 1.0
    return list(pairwise((lo * (hi / lo) ** steps).tolist()))


def sampler(image: np.ndarray, firm: float, left: int = 0, top: int = 0):
    """The pressure a shading takes from ``image``, a patch of the sheet
    whose corner is at (``left``, ``top``): ``firm`` times its value where
    the stick is, and nothing outside it."""
    high, wide = image.shape

    def pressure(x, y):
        col = np.rint(np.asarray(x)).astype(np.intp) - left
        row = np.rint(np.asarray(y)).astype(np.intp) - top
        on = (col >= 0) & (col < wide) & (row >= 0) & (row < high)
        value = image[np.clip(row, 0, high - 1), np.clip(col, 0, wide - 1)]
        return np.where(on, firm * value, 0.0)

    return pressure


class Easel:
    """The sheet, the street as the eye sees it, and the strokes everything
    is drawn with: each is given in the street's own measure, and is laid in
    the colour of the light that falls where it lands."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(SEED)
        #: (the people have their own, so that redrawing them leaves the
        #: trees as they were)
        self.crowd = np.random.default_rng(SEED + 1)
        self.sheet = Sheet(WIDTH, HEIGHT, paper=PAPER, seed=SEED)
        self.buildings = street(self.rng)
        self.trees = avenue(self.rng)
        self.crowns = {tree: grow(tree) for tree in (*self.trees, OFFSTAGE)}
        near = [mass for tree, (_, mass) in self.crowns.items() if tree.z < 150.0]
        self.scene = Scene(self.buildings, np.concatenate(near), self.rng)

    def vary(self, colour: Colour, by: float = 0.03) -> Colour:
        """The stick next to that one in the box: no two strokes of a
        pastel are quite one colour."""
        return colour + self.rng.normal(0.0, by, 3)

    def nearby(self, colours, by: float = 0.018):
        """:meth:`vary`, for one colour or for a pair: the one for the shade
        strays further, where colours are less sure."""
        if isinstance(colours, tuple):
            sun, shade = colours
            sun = None if sun is None else self.vary(sun, by)
            return sun, None if shade is None else self.vary(shade, 1.7 * by)
        return self.vary(colours, by)

    def at(self, field: np.ndarray, path: np.ndarray) -> np.ndarray:
        """The values of a map of the sheet, (H, W), at the points of a path."""
        col = np.clip(path[:, 0].astype(np.intp), 0, WIDTH - 1)
        row = np.clip(path[:, 1].astype(np.intp), 0, HEIGHT - 1)
        return field[row, col]

    def on(self, path: np.ndarray, face) -> np.ndarray:
        """1 where the points of a path lie on ``face`` (one or several of
        the things the scene sees), as the eye sees it."""
        return np.isin(self.at(self.scene.seen, path), face) * inside(path)

    def mark(self, path, colour, press, width, tilt=0.0, hard=False) -> None:
        """One stroke along a path of the sheet, ``width`` pixels wide,
        pressed as ``press`` says at each of its points; nothing if it would
        hardly touch."""
        press = np.broadcast_to(np.asarray(press, dtype=np.float64), (len(path),))
        if press.max() < 0.06:
            return
        tip = float(max(width, THIN) / (1.0 + (SIDE - 1.0) * tilt))
        chalk = stick(colour, hard)
        self.sheet.stroke(Polyline(path), chalk, press.tolist(), tip, tilt)

    def rail(
        self,
        ends,
        colours,
        firm: float,
        width: float,
        face=None,
        *,
        tilt: float = 0.0,
        hard: bool = False,
        worn: float = 0.0,
        square: bool = False,
    ) -> None:
        """One straight stroke between two points of the street. ``colours``
        is one colour, or two: the one for where the sun falls and the one
        for the shade (either may be None, to leave that part alone). With a
        ``face``, the stroke is only laid where the eye sees that face, and
        takes its light from the scene; without, it is a thing standing in
        the street, lit unless a building shades it. ``worn`` breaks it up;
        ``square`` stops its round ends short, at the two points."""
        ends = np.array(ends, dtype=np.float64)
        long = float(np.hypot(*np.diff(project(ends), axis=0)[0]))
        inset = min(0.5 * width / max(long, 1e-6), 0.4) if square else 0.0
        u = np.linspace(0.0, 1.0, int(np.clip(long / 4.0, 2, 60)) + 1)
        share = inset + (1.0 - 2.0 * inset) * u
        points = ends[0] + (ends[1] - ends[0]) * share[:, None]
        path = project(points)
        # (lighter where the stick lands and where it lifts)
        landing = 0.15 if square else 0.4
        ease = np.sqrt(np.sin(np.pi * np.clip(u, 0.05, 0.95)))
        press = firm * (1.0 - landing * (1.0 - ease))
        if worn:
            press = press * (1.0 - worn * self.rng.random(len(u)))
        if face is not None:
            press = press * self.on(path, face)
        if not isinstance(colours, tuple):
            self.mark(path, colours, press, width, tilt, hard)
            return
        if face is None:
            lit = sunlit(self.scene.boxes, points)
        else:
            lit = self.at(self.scene.lit, path)
        for colour, part in zip(colours, (lit, 1.0 - lit), strict=True):
            if colour is not None:
                self.mark(path, colour, press * part, width, tilt, hard)

    def panel(self, low, high, colours, firm: float, face=None) -> None:
        """A flat piece of something between two of its edges, each given by
        its two ends: strokes that run with the edges and stop square at the
        sides, as many side by side as the piece asks for."""
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        tall = float(np.hypot(*(project(high) - project(low)).T).mean())
        rows = max(math.ceil(tall / STICK), 1)
        for r in range(rows):
            ends = low + (r + 0.5) / rows * (high - low)
            self.rail(ends, colours, firm, 1.12 * tall / rows, face, square=True)

    def slab(self, x, y, z: float, colours, firm: float) -> None:
        """An upright piece of something that faces the eye, ``z`` away:
        from ``x[0]`` to ``x[1]`` across and from ``y[0]`` to ``y[1]`` up."""
        low, high = level(x[0], x[1], y[0], z), level(x[0], x[1], y[1], z)
        self.panel(low, high, colours, firm)

    def cast(self, foot, tall: float, thick: float, rng=None) -> None:
        """The shadow a post, a trunk or a person ``thick`` metres wide lays
        on sunlit ground, from its foot to where the shadow of its top
        falls."""
        foot = np.asarray(foot, dtype=np.float64)
        tip = foot + tall * CAST
        lo = max(float(foot[2]), 3.3)
        if tip[2] <= lo:
            return
        for near, away in reaches(lo, float(tip[2]), self.rng if rng is None else rng):
            share = (np.array([near, away]) - foot[2]) / (tip[2] - foot[2])
            ends = foot + share[:, None] * (tip - foot)
            middle = math.sqrt(near * away)
            for face, tone in ((ROAD, ROADWAY[1]), (WALK, PAVEMENT[1])):
                colours = (far(tone, middle), None)
                self.rail(ends, colours, 0.7, thick * rise(middle), face)


class Blocking(Easel):
    """The first stage: every large shape in its tone, with the side of the
    stick; the sky and the far end of the street rubbed with a finger."""

    def block_in(self) -> None:
        for k, b in self.far_first():
            self.wash_wall(b, k)
        self.wash_ground()
        self.sky()
        self.distance()
        self.sheet.fix()

    def far_first(self) -> list[tuple[int, Building]]:
        """The buildings ahead of the eye with their numbers, the farthest
        first."""
        ahead = [(k, b) for k, b in enumerate(self.buildings) if b.z1 > NEAR[b.side]]
        return sorted(ahead, key=lambda pair: -pair[1].z0)

    def sky(self) -> None:
        """The sky: a pale blue overhead that warms and pales toward the end
        of the street, laid with the side of the stick, rubbed smooth, then
        gone over lightly in strokes that are left as they are."""
        open_sky = (self.scene.seen == SKY) * 1.0
        clear = (filters.box_blur(open_sky, 14) > 0.75) * 1.0
        low = np.clip(np.arange(HEIGHT) / VANISH[1], 0.0, 1.0)[:, None] ** 1.6
        foot = VANISH[1] + 8.0
        wedge = [(-150, -40), (WIDTH + 100, -40), (VANISH[0] + 60, foot)]
        wedge.append((VANISH[0] - 60, foot))
        shade = self.sheet.shade
        parts = ((SKY_HIGH, 1.0 - low, 12.0), (SKY_MID, 4.0 * low * (1.0 - low), -10.0))
        for colour, part, angle in (*parts, (SKY_LOW, low, 4.0)):
            for turn in (0.0, 33.0):
                press = sampler(clear * part, 0.72)
                shade(wedge, stick(colour), press, 13.0, angle + turn, tilt=1.0)
        # (its edges, where the side of the stick would go over the roofs)
        end = (VANISH[0] - 170, VANISH[1] - 260, VANISH[0] + 170, foot)
        shade(end, stick(SKY_LOW), sampler(open_sky * low, 0.5), 5.0, 30.0)
        top = sampler(open_sky * (1.0 - low), 0.45)
        shade((0, 0, WIDTH, 150), stick(SKY_HIGH), top, 6.0, 20.0)
        for row in range(8, int(VANISH[1]), 10):
            cols = np.nonzero(open_sky[row] > 0)[0]
            if len(cols) > 50:
                ends = [(cols[0] + 12.0, row), (cols[-1] - 12.0, row + 3.0)]
                self.sheet.smudge(ends[:: 1 if row % 20 else -1], tools.FINGER, 0.5)
        edge = np.stack([np.linspace(0.0, WIDTH, 60), np.full(60, 4.0)], axis=1)
        self.mark(edge, SKY_HIGH, 0.6 * self.on(edge, SKY), 12.0)
        veil = sampler(clear * (0.2 + 0.25 * low), 1.0)
        chalk = stick(mix(SKY_MID, WHITE, 0.5))
        shade(wedge, chalk, veil, 12.0, 17.0, spacing=64.0, tilt=1.0)
        self.clouds()

    def clouds(self) -> None:
        """A few thin clouds high up: light strokes of the side of the
        stick."""
        places = ((300.0, 170.0, 220.0), (540.0, 330.0, 150.0), (120.0, 45.0, 180.0))
        u = np.linspace(0.0, 1.0, 14)[:, None]
        for x, y, long in (*places, (640.0, 90.0, 130.0), (400.0, 470.0, 90.0)):
            ends = np.array([[x, y], [x + long, y - 0.1 * long]])
            for k in range(2):
                shift = np.array([18.0 * k, self.rng.uniform(-7.0, 7.0)])
                path = ends[0] + shift + (ends[1] - ends[0]) * (1.0 - 0.2 * k) * u
                press = 0.36 * np.sin(np.pi * u[:, 0]) * self.on(path, SKY)
                self.mark(path, CLOUD, press, 30.0, tilt=1.0)

    def wash_wall(self, b: Building, k: int) -> None:
        """The first tone of a building's front and of its gable: its colour
        in the sun where the sun falls, in the shade elsewhere, a little
        deeper than it will end."""
        self.wash(front(k), distant(WALLS[b.wall], b.z0))
        back = WALLS[b.wall] if b.on_corner else PARTY_WALL
        self.wash(gable(k), distant(back, b.z0))

    def wash(self, face: int, pair: Tones) -> None:
        """The first tone of one face, where the eye sees it: upright strokes
        of the side of the stick (of its end, if the face is narrow), in two
        sticks that are not quite one colour."""
        rows, cols = np.nonzero(self.scene.seen == face)
        if len(rows) < 12:
            return
        left, top = int(cols.min()), int(rows.min())
        patch = (slice(top, int(rows.max()) + 1), slice(left, int(cols.max()) + 1))
        shape = (self.scene.seen[patch] == face) * 1.0
        high, wide = shape.shape
        area = (left - 2, top - 2, left + wide + 2, top + high + 2)
        broad = wide > 45
        width = float(np.clip(wide / 7.0, 3.0, 11.0))
        if broad:  # (the side of the stick marks a band: keep it inside)
            shape = (filters.box_blur(shape, int(1.6 * width)) > 0.8) * 1.0
        lit = self.scene.lit[patch]
        under = (mix(pair[0], UNDER[0], 0.25), mix(pair[1], UNDER[1], 0.35))
        images = (shape * lit, shape * (1.0 - lit))
        for colour, image in zip(under, images, strict=True):
            if image.sum() < 12:
                continue
            for angle in (84.0, 97.0) if wide > 12 else (90.0,):
                press = sampler(image, 0.45, left, top)
                chalk = stick(self.vary(colour, 0.035))
                self.sheet.shade(area, chalk, press, width, angle, tilt=1.0 * broad)

    def wash_ground(self) -> None:
        """The first tone of the roadway and the pavements, in strokes that
        run down the street, broader toward the eye."""
        edges = 3.2 * 1.42 ** np.arange(18)
        for near, away in pairwise(edges.tolist()):
            size = rise(math.sqrt(near * away))
            width = float(np.clip(0.7 * size, 12.0, 60.0))
            for sx in np.arange(-0.25 * width, WIDTH + width, 0.52 * width):
                sx += self.rng.uniform(-0.15, 0.15) * width
                x = EYE[0] + (sx - VANISH[0]) / size
                self.ground_stroke(x, near, away, width, 0.5, 1.0)

    def ground_stroke(self, x, near, away, width, firm, tilt=0.0) -> None:
        """One stroke down the street on the ground, ``x`` across it: in the
        colours of the roadway where it runs on the roadway, of the pavement
        where it runs on a pavement; a little firmer in the shade."""
        middle = math.sqrt(near * away)
        ends = lengthwise(x, 0.0, near, away)
        for face, pair in ((ROAD, ROADWAY), (WALK, PAVEMENT)):
            sun, shade = distant(pair, middle)
            sun, shade = self.vary(sun, 0.025), self.vary(shade, 0.035)
            self.rail(ends, (sun, None), firm, width, face, tilt=tilt)
            self.rail(ends, (None, shade), 1.3 * firm, width, face, tilt=tilt)

    def distance(self) -> None:
        """The far end of the street, rubbed with a finger: haze. And where
        the street bends, a mile on, the spire of the church that closes it,
        pale over the last roofs."""
        for k in range(9):
            y = VANISH[1] - 96.0 + 12.0 * k
            ends = [(VANISH[0] - 62.0, y), (VANISH[0] + 52.0, y + 2.0)]
            self.sheet.smudge(ends[:: 1 if k % 2 else -1], tools.FINGER, 0.4, 16.0)
        z = 1320.0
        stone = mix(HAZE, rgb("#8a90a8"), 0.55)
        for foot, top, wide in ((0.0, 38.0, 8.0), (38.0, 54.0, 5.0), (54.0, 70.0, 3.0)):
            ends = ((0.0, foot, z), (0.0, top, z))
            self.rail(ends, stone, 0.55, wide * rise(z), SKY)


class Fronts(Blocking):
    """The second stage: the buildings, each as it is built."""

    def architecture(self) -> None:
        """Each building, the farthest first, then fixative."""
        for k, b in self.far_first():
            self.gable_wall(b, k)
            if b.span[1] > b.span[0]:
                self.facade(b, k)
        self.sheet.fix()

    def band(self, b, heights, colours, firm, face=None, span=None, **how) -> None:
        """A band along a front between two heights (a spandrel, a frieze, a
        belt course, a cornice): strokes that run with the perspective, as
        many side by side as its height asks for, no two of quite one colour
        (``loose`` says how far apart). ``out`` stands it proud of the wall."""
        low, high = heights
        lo, hi = span or b.span
        out, loose = how.pop("out", 0.0), how.pop("loose", 0.018)
        if hi <= lo:
            return
        for near, away in reaches(lo, hi, self.rng):
            tall = (high - low) * rise(math.sqrt(near * away))
            rows = max(math.ceil(tall / STICK), 1)
            for r in range(rows):
                ends = b.edge(near, away, low + (r + 0.5) / rows * (high - low), out)
                tints = self.nearby(colours, loose)
                self.rail(ends, tints, firm, 1.15 * tall / rows, face, **how)

    def post(self, b, z, heights, colours, firm, wide, face=None, arch=0.0) -> None:
        """An upright on a front (a window, a column, a pier), ``wide``
        metres along the street: strokes from its top to its foot, side by
        side; under an arch the middle ones start higher."""
        low, high = heights
        across = wide * b.run(z)
        count = max(math.ceil(across / STICK), 1)
        tints = self.nearby(colours, 0.012)
        for k in range(count):
            share = (k + 0.5) / count
            top = high - arch * (1.0 - math.sin(math.pi * share)) * (count > 1)
            ends = b.upright(z + (share - 0.5) * wide, low, top)
            self.rail(ends, tints, firm, 1.15 * across / count, face)

    def facade(self, b: Building, k: int) -> None:
        """One front, as it is built: the storeys from the ground floor up,
        the cornice over them, a fire escape hung on it, a tank on its
        roof."""
        face = front(k)
        fine = rise(b.span[0])
        pair = distant(WALLS[b.wall], 0.5 * sum(b.span))
        self.shopfronts(b, face, pair, fine)
        for low, high in pairwise(b.floors()):
            self.storey(b, face, pair, (low, high), fine)
        self.cornice(b, face, pair, fine)
        if "e" in b.marks and fine > 6.0:
            self.fire_escape(b, pair)
        if "t" in b.marks:
            self.water_tank(b)

    def storey(self, b, face, pair, floor, fine) -> None:
        """One storey over the ground floor: the band of wall at its floor
        line, its row of windows, the columns between them, the belt
        course."""
        low, high = floor
        wall = (low - HEAD, low + SILL)
        glazed = (low + SILL, high - HEAD)
        arched = "a" in b.marks
        soft = 0.3 if fine > 12.0 else 0.0
        if arched:
            self.band(b, wall, pair, 0.55, face, tilt=soft)
        self.windows(b, face, pair, glazed)
        if not arched:  # (a flat lintel cuts the heads of the windows square)
            self.band(b, wall, pair, 0.55, face, tilt=soft)
        if fine > 9.0:
            self.columns(b, face, pair, glazed)
            self.band(b, (low - 0.5, low - 0.12), shadowed(pair), 0.5, face)
            self.band(b, (low - 0.1, low + 0.14), raised(pair, 0.6), 0.7, face)

    def windows(self, b, face, pair, heights) -> None:
        """A row of windows: a dab for each as long as they can be told
        apart, then one broken stroke for the rest of the row."""
        lo, hi = b.span
        glass = distant(GLASS[b.side], 0.5 * (lo + hi))
        arch = 0.4 if "a" in b.marks else 0.0
        wide = WINDOW * b.bay
        for j in range(b.bays):
            z = b.z0 + (j + 0.5) * b.bay
            if z < lo - 0.3 * b.bay:
                continue
            across = wide * b.run(z)
            if across < 2.6:
                rest = (z - 0.5 * b.bay, hi)
                self.band(b, heights, glass, 0.45, face, rest)
                self.glints(b, face, heights, rest)
                return
            self.post(b, z, heights, glass, 0.75, wide, face, arch)
            self.glint(b, face, z, (heights[0], heights[1] - arch * (across > STICK)))
            if across > 9.0:  # the bar where the two sashes meet
                bar = (mix(glass[0], pair[0], 0.4), mix(glass[1], pair[1], 0.5))
                middle = 0.5 * sum(heights)
                reach = (z - 0.45 * wide, z + 0.45 * wide)
                self.band(b, (middle - 0.04, middle + 0.04), bar, 0.45, face, reach)

    def mirrored(self, b, z, high) -> Tones | None:
        """What a window mirrors, if it mirrors anything: seen along the
        street glass is a mirror, the more the farther and the higher. Or
        it has a blind half drawn."""
        chance = 0.25 + 0.3 * min(z / 90.0, 1.0) + 0.2 * min(high / 25.0, 1.0)
        pick = self.rng.random()
        if pick > chance:
            return None
        if pick < 0.07:
            return distant(BLIND, z)
        warm, cool = MIRRORED[b.side]
        return distant(warm if self.rng.random() < 0.6 else cool, z)

    def glint(self, b, face, z, heights) -> None:
        """The upper part of a window, mirroring the other side or the sky."""
        low, high = heights
        seen = self.mirrored(b, z, high)
        if seen is not None:
            foot = high - self.rng.uniform(0.3, 0.65) * (high - low)
            wide = 0.7 * WINDOW * b.bay
            self.post(b, z, (foot, high - 0.15), seen, 0.5, wide, face)

    def glints(self, b, face, heights, span) -> None:
        """Mirrored light along the far part of a row of windows: short
        strokes, here and there."""
        low, high = heights
        z = span[0]
        while z < span[1]:
            long = self.rng.uniform(1.0, 3.5)
            seen = self.mirrored(b, z, high)
            if seen is not None:
                reach = (z, min(z + long, span[1]))
                self.band(b, (0.5 * (low + high), high - 0.1), seen, 0.45, face, reach)
            z += long + self.rng.uniform(1.0, 5.0)

    def columns(self, b, face, pair, heights) -> None:
        """The columns between the windows of a storey, where they can be
        told apart."""
        bright = (mix(pair[0], WHITE, 0.35), mix(pair[1], HAZE, 0.12))
        wide = (1.0 - WINDOW) * b.bay
        reach = (heights[0] - 0.1, heights[1] + 0.1)
        for j in range(b.bays + 1):
            z = b.z0 + j * b.bay
            if z >= b.span[0] and wide * b.run(z) >= 2.6:
                self.post(b, z, reach, bright, 0.6, 0.8 * wide, face)

    def shopfronts(self, b, face, pair, fine) -> None:
        """The ground floor: plate glass between piers on a low base, a
        fascia over each shop, a cornice over the whole."""
        lo, _ = b.span
        top = GROUND_FLOOR
        glass = distant(GLASS[b.side], lo)
        dark = (mix(glass[0], INK, 0.15), glass[1])
        self.band(b, (0.45, 3.45), dark, 0.8, face, loose=0.004)
        shops = max(round(b.bays / 3), 1)
        wide = (b.z1 - b.z0) / shops
        for j in range(shops):
            span = (max(b.z0 + j * wide, lo), b.z0 + (j + 1) * wide)
            if span[1] > lo:
                self.shop(b, face, span, fine)
        for j in range(shops + 1):  # the piers
            z = b.z0 + j * wide
            if z >= lo and fine > 5.0:
                self.post(b, z, (0.0, top - 0.5), pair, 0.65, 0.6, face)
        self.band(b, (0.0, 0.45), both(pair, INK, 0.3), 0.55, face)
        self.band(b, (top - 0.95, top - 0.55), shadowed(pair), 0.5, face)
        self.band(b, (top - 0.5, top - 0.05), raised(pair, 0.5), 0.7, face, out=0.3)

    def shop(self, b, face, span, fine) -> None:
        """One shop: its fascia; in its window, upright shapes half lost in
        the dark of the glass: the street mirrored, the far side of it, the
        lamps lit inside; an awning."""
        z = 0.5 * sum(span)
        fascia = distant(FASCIAS[int(self.rng.integers(len(FASCIAS)))], z)
        self.band(b, (3.55, 4.15), fascia, 0.65, face, (span[0] + 0.3, span[1] - 0.3))
        if fine < 5.0 or span[1] - span[0] < 2.0:
            return
        glass = distant(GLASS[b.side], z)
        seen = (STREET_MIRRORED, MIRRORED[b.side][0 if b.side < 0 else 1], SHOP_LIGHT)
        at = span[0] + 0.5
        while at < span[1] - 0.6:
            wide = self.rng.uniform(0.5, 1.5)
            pick = int(self.rng.integers(3))
            clear = self.rng.uniform(0.3, 0.6) + 0.25 * (pick == 2 and b.side < 0)
            tint = distant(seen[pick], z)
            tint = (mix(glass[0], tint[0], clear), mix(glass[1], tint[1], clear))
            heights = (self.rng.uniform(0.6, 1.3), 3.3)
            self.post(b, at + 0.5 * wide, heights, tint, 0.7, wide, face)
            at += wide + self.rng.uniform(0.3, 1.5)
        if "w" in b.marks and self.rng.random() < 0.7:
            self.awning(b, span)

    def awning(self, b, span) -> None:
        """An awning over a shop window: cloth sloping out over the pavement,
        its edge hanging down."""
        cloth = distant(FASCIAS[int(self.rng.integers(1, len(FASCIAS)))], span[0])
        cloth = (mix(cloth[0], WHITE, 0.25), cloth[1])
        near, away = span[0] + 0.5, span[1] - 0.5
        self.panel(b.edge(near, away, 3.4), b.edge(near, away, 2.75, 1.3), cloth, 0.75)
        hem = (mix(cloth[1], cloth[0], 0.4), mix(cloth[1], INK, 0.3))
        self.rail(b.edge(near, away, 2.6, 1.3), hem, 0.7, 0.3 * rise(near))

    def cornice(self, b, face, pair, fine) -> None:
        """The top of a front: the frieze, in the shadow of the cornice; the
        cornice itself, standing out over the street: its underside, seen
        from below, dark; the brackets that carry it; its edge in the
        light."""
        top = b.height
        frieze = (mix(pair[1], pair[0], 0.35), mix(pair[1], INK, 0.1))
        self.band(b, (top - ATTIC, top - 0.7), frieze, 0.6, face)
        under = mix(mix(pair[1], INK, 0.3), pair[0], 0.12 * (b.side > 0))
        lo, hi = b.span
        for near, away in reaches(lo, hi, self.rng):
            lip = b.edge(near, away, top - 0.5, 0.9)
            tint = self.vary(under, 0.015)
            self.panel(b.edge(near, away, top - 0.5), lip, tint, 0.7)
        for j in range(b.bays + 1):  # the brackets
            z = b.z0 + j * b.bay
            if z > b.span[0] and fine > 12.0:
                ends = b.upright(z, top - 1.4, top - 0.5, 0.3)
                self.rail(ends, mix(under, INK, 0.2), 0.6, 0.3 * b.run(z))
        self.band(b, (top - 0.5, top + 0.15), raised(pair, 0.55), 0.8, out=0.9)

    def fire_escape(self, b: Building, pair: Tones) -> None:
        """A fire escape: an iron balcony at each floor two bays wide,
        stairs from each to the next, turn and turn about, a ladder that
        stops short of the pavement; and on a sunlit wall, their shadows."""
        first = max(b.bays // 2 - 1, 0)
        near, away = b.z0 + first * b.bay + 0.3, b.z0 + (first + 2) * b.bay - 0.3
        if near < b.span[0]:
            return
        heights = [at + 0.15 for at in b.floors()[:-1]]
        thick = max(0.08 * rise(near), THIN)
        iron = distant(IRON, near)
        if b.side > 0:
            self.escape_shadow(b, pair, (near, away), heights)
        for i, y in enumerate(heights):
            self.balcony(b, (near, away), y, iron, thick)
            if i + 1 < len(heights):
                foot, head = (near + 0.4, away - 0.4)[:: 1 if i % 2 else -1]
                stair = (b.point(foot, y, 0.7), b.point(head, heights[i + 1], 0.7))
                self.rail(stair, iron, 0.85, 1.3 * thick)
        ladder = b.upright(near + 0.5, 2.9, heights[0], 1.2)
        self.rail(ladder, iron, 0.75, thick, hard=True)

    def balcony(self, b, span, y, iron, thick) -> None:
        """One balcony of a fire escape: its floor, seen from below, a dark
        slab standing out of the wall; its rail; its bars."""
        near, away = span
        self.panel(b.edge(near, away, y), b.edge(near, away, y, 1.25), iron, 0.7)
        self.rail(b.edge(near, away, y + 1.0, 1.25), iron, 0.8, thick, hard=True)
        for z in (near, away):
            ends = (b.point(z, y + 1.0), b.point(z, y + 1.0, 1.25))
            self.rail(ends, iron, 0.8, thick, hard=True)
        for z in np.arange(near, away + 0.1, 0.7):
            self.rail(b.upright(z, y, y + 1.0, 1.25), iron, 0.6, thick, hard=True)

    def escape_shadow(self, b, pair, span, heights) -> None:
        """The shadows a fire escape lays on a sunlit wall: each balcony's
        falls further up the street and lower, as the sun stands."""
        fronts = [front(k) for k, other in enumerate(self.buildings) if other.side > 0]
        shade = (shadowed(pair)[0], None)
        along = 1.25 / math.tan(SUN_TURN)
        down = 1.25 * TO_SUN[1] / math.sin(SUN_TURN)
        for y in heights:
            fallen = b.edge(span[0] + along, span[1] + along, y - down)
            self.panel(b.edge(*span, y), fallen, shade, 0.5, fronts)

    def water_tank(self, b: Building) -> None:
        """A wooden water tank on the roof, set back from the front: staves,
        hoops and a cone on iron legs, in the sun; only what shows over the
        roofs."""
        z, back = b.z0 + 3.4, -3.6
        if z < b.span[0] + 2.0:
            return
        foot, body = b.height + 2.4, b.height + 6.0
        wood = distant(TANK, z)
        size = rise(z)
        for off in np.linspace(-1.5, 1.5, 7):  # the staves, upright
            turned = float(np.clip(0.35 - 0.3 * b.side * off, 0.0, 0.85))
            stave = self.vary(mix(wood[0], wood[1], turned), 0.02)
            self.rail(
                b.upright(z, foot, body, back + off), stave, 0.75, 0.62 * size, SKY
            )
        dark = mix(wood[1], INK, 0.3)
        for y in (foot + 0.5, foot + 1.9, foot + 3.1):  # the hoops
            ends = (b.point(z, y, back - 1.8), b.point(z, y, back + 1.8))
            self.rail(ends, dark, 0.6, THIN, SKY, hard=True)
        peak = b.point(z, body + 1.2, back)
        for off in (-1.5, -0.5, 0.5, 1.5):  # the cone
            turned = float(np.clip(0.5 - 0.25 * b.side * off, 0.0, 1.0))
            ends = (b.point(z, body + 0.1, back + off), peak)
            self.rail(ends, mix(wood[0], wood[1], turned), 0.75, 0.5 * size, SKY)
        for off in (-1.3, 1.3):  # the legs
            ends = b.upright(z, b.height, foot, back + off)
            self.rail(ends, far(IRON[1], z), 0.7, THIN, SKY, hard=True)

    def gable_wall(self, b: Building, k: int) -> None:
        """The wall of a building that faces down the street, where it shows:
        brick courses run level across it; on a cross street it has its
        rows of windows too."""
        face = gable(k)
        rows, cols = np.nonzero(self.scene.seen == face)
        if len(rows) < 30:
            return
        size = rise(b.z0)
        x = EYE[0] + (np.array([cols.min(), cols.max() + 1.0]) - VANISH[0]) / size
        pair = distant(WALLS[b.wall] if b.on_corner else PARTY_WALL, b.z0)
        step = float(np.clip(0.8 * size, 3.0, STICK)) / size
        for y in np.arange(0.5 * step, b.height, 0.85 * step):
            ends = level(x[0], x[1], y, b.z0)
            self.rail(ends, self.nearby(pair, 0.025), 0.55, step * size, face)
        if not b.on_corner or size * STOREY < 8.0:
            return
        glass = distant(GLASS[1], b.z0)
        lo, hi = sorted(abs(v) for v in x)
        for low, high in pairwise(b.floors()):
            for across in np.arange(HALF + 1.6, hi, 2.6):
                if across >= lo:
                    at = b.side * across
                    ends = ((at, high - HEAD, b.z0), (at, low + SILL, b.z0))
                    self.rail(ends, glass, 0.7, 1.5 * size, face)


class Ground(Fronts):
    """The third stage: the roadway and the pavements, with what is painted
    on them and what lies on them."""

    def pavement(self) -> None:
        self.road_texture()
        self.kerbs()
        self.road_paint()
        self.ironwork()
        for tree in self.trees:
            if tree.z < 200.0:  # the square of earth it stands in
                x = sorted((tree.side * (KERB + 0.2), tree.side * (KERB + 1.6)))
                self.ground_patch(x, (tree.z - 0.9, tree.z + 0.9), EARTH, 0.6)
        self.fallen_leaves()
        self.sheet.fix()

    def road_texture(self) -> None:
        """Worn lanes and patches: strokes of the end of the stick down the
        street, over the first tone."""
        edges = 3.2 * 1.5 ** np.arange(12)
        for near, away in pairwise(edges.tolist()):
            middle = math.sqrt(near * away)
            width = float(np.clip(0.07 * rise(middle), THIN, 9.0))
            for _ in range(int(np.clip(900.0 / middle, 14, 70))):
                x = self.rng.uniform(-13.0, 13.0)
                end = away * self.rng.uniform(0.8, 1.1)
                self.ground_stroke(x, near, end, width, 0.42)

    def kerbs(self) -> None:
        """The kerbs: the right one faces the sun, the left one is a dark
        line; and the gutter along each."""
        for side in (-1, 1):
            for near, away in reaches(NEAR[side] * 0.5, 600.0, self.rng):
                middle = math.sqrt(near * away)
                stone = distant(KERBSTONE, middle)
                colours = stone if side > 0 else (stone[1], stone[1])
                width = max(0.16 * rise(middle), THIN)
                ends = lengthwise(side * KERB, 0.0, near, away)
                self.rail(ends, colours, 0.7, width, GROUND)
                dark = both(distant(ROADWAY, middle), INK, 0.3)
                ends = lengthwise(side * (KERB - 0.25), 0.0, near, away)
                self.rail(ends, dark, 0.4, width, ROAD)

    def road_paint(self) -> None:
        """What is painted on the roadway: the bars of the crossing at the
        eye's feet, the stop line beyond it, the dashes between the lanes,
        the far crossings."""
        for x in np.arange(-6.3, 6.4, 1.2):
            self.ground_patch((x - 0.3, x + 0.3), (5.2, 8.6), PAINT, 0.7, 0.6)
        self.cross_patch((-4.2, 4.2), (10.4, 10.85), PAINT, 0.65)
        for x in (-1.43, 1.43):
            for z in np.arange(13.0, 112.0, 9.0):
                self.ground_patch((x - 0.07, x + 0.07), (z, z + 3.0), PAINT, 0.7, 0.3)
        for start in np.arange(START + BLOCK, 700.0, BLOCK + CROSS):
            for z in (start + 0.5, start + CROSS - 3.5):
                self.cross_patch((-6.3, 6.3), (z, z + 3.0), PAINT, 0.5)

    def ground_patch(self, across, along, pair, firm, worn=0.0) -> None:
        """A patch of the ground in strokes that run down the street."""
        middle = math.sqrt(along[0] * along[1])
        wide = across[1] - across[0]
        count = max(math.ceil(wide * rise(along[0]) / STICK), 1)
        colours = distant(pair, middle)
        width = 1.1 * wide * rise(middle) / count
        for k in range(count):
            ends = lengthwise(across[0] + (k + 0.5) / count * wide, 0.0, *along)
            self.rail(ends, colours, firm, width, GROUND, worn=worn)

    def cross_patch(self, across, along, pair, firm) -> None:
        """A patch of the ground in strokes that run across the street."""
        near, away = 1.0 / along[0], 1.0 / along[1]
        tall = FOCAL * EYE[1] * (near - away)
        count = max(math.ceil(tall / STICK), 1)
        colours = distant(pair, along[0])
        for k in range(count):
            z = 1.0 / (near + (k + 0.5) / count * (away - near))
            ends = level(across[0], across[1], 0.0, z)
            self.rail(ends, colours, firm, 1.1 * tall / count, GROUND, worn=0.4)

    def ironwork(self) -> None:
        """Tar seams down the roadway, and the covers of manholes: rings of
        iron flat in the road, darker than the tar round them."""
        tar = both(ROADWAY, INK, 0.35)
        for x, start, end in ((-3.7, 3.4, 30.0), (1.2, 3.4, 22.0), (4.5, 9.0, 60.0)):
            for near, away in reaches(start, end, self.rng):
                middle = math.sqrt(near * away)
                width = max(0.035 * rise(middle), THIN)
                ends = lengthwise(x + self.rng.normal(0.0, 0.03), 0.0, near, away)
                self.rail(ends, distant(tar, middle), 0.5, width, GROUND, worn=0.6)
        turn = np.linspace(0.0, 2.0 * math.pi, 25)
        for x, z in ((3.0, 4.9), (-0.3, 12.5), (2.2, 31.0), (-2.4, 20.0)):
            iron = distant(both(ROADWAY, INK, 0.5), z)
            ring = [x + 0.3 * np.cos(turn), 0.0 * turn, z + 0.3 * np.sin(turn)]
            path = project(np.stack(ring, axis=1))
            lit = self.at(self.scene.lit, path)
            for colour, part in zip(iron, (lit, 1.0 - lit), strict=True):
                self.mark(path, colour, 0.75 * part, 0.06 * rise(z))
            for off in (-0.14, 0.0, 0.14):
                ends = level(x - 0.24, x + 0.24, 0.0, z + off)
                self.rail(ends, iron, 0.4, 0.07 * rise(z), GROUND)

    def fallen_leaves(self) -> None:
        """Leaves on the ground: under each tree, along the gutters, and
        blown across the crossing from the tree at the corner."""
        rng = self.rng
        for tree in self.trees:
            if tree.z > 260.0:
                continue
            count = int(np.clip(6000.0 / tree.z, 10, 300))
            x = tree.x + rng.normal(0.0, 1.7, count)
            self.scatter(tree.leaf, x, tree.z + rng.normal(2.0, 4.5, count))
            gutter = tree.side * (KERB - rng.uniform(0.05, 0.5, count // 2))
            self.scatter(tree.leaf, gutter, tree.z + rng.uniform(-7.0, 7.0, count // 2))
        for leaf, count in (("gold", 200), ("orange", 50), ("russet", 40)):
            x, z = rng.normal(-2.5, 3.2, count), rng.uniform(3.6, 17.0, count)
            self.scatter(leaf, x, z)

    def scatter(self, leaf: str, x, z) -> None:
        """Fallen leaves at those places: flecks lying flat, turned any
        way."""
        light, middle, _, shade, deep, _ = (rgb(code) for code in LEAVES[leaf])
        for across, away in zip(x, z, strict=True):
            turn = self.rng.uniform(0.0, math.pi)
            half = self.rng.uniform(0.03, 0.065)
            reach = half * np.array([math.cos(turn), 0.0, math.sin(turn)])
            spot = np.array([across, 0.0, max(away, 3.3)])
            pick = self.rng.random() < 0.5
            colours = (light if pick else middle, shade if pick else deep)
            width = float(np.clip(0.055 * rise(spot[2]), THIN, 7.0))
            firm = float(self.rng.uniform(0.5, 0.8))
            ends = (spot - reach, spot + reach)
            self.rail(ends, distant(colours, spot[2]), firm, width, GROUND)


class Life(Ground):
    """The fourth stage: everything that stands or moves in the street."""

    def life(self) -> None:
        """Trees, cars, people, lamps, hydrants and signals, the farthest
        first; fixative now and then, so that what stands in front covers
        what is behind it."""
        draw = {
            Tree: self.tree,
            Car: self.car,
            Walker: self.walker,
            Lamp: self.lamp,
            Hydrant: self.hydrant,
            Signal: self.signal,
        }
        things = [*self.trees, *CARS, *WALKERS, *LAMPS, *HYDRANTS, *SIGNALS]
        for count, thing in enumerate(sorted(things, key=lambda thing: -thing.z)):
            draw[type(thing)](thing)
            if thing.z < 130.0 and count % 4 == 0:
                self.sheet.fix()
        self.sheet.fix()

    def tree(self, tree: Tree) -> None:
        """One tree: the shadow of its trunk, the trunk and limbs, its
        leaves, then the limbs again where they show between the leaves."""
        limbs, masses = self.crowns[tree]
        size = rise(tree.z)
        bark = distant(BARK, tree.z)
        base, fork, thick = limbs[0]
        self.cast(base, fork[1], thick)
        paths = []
        for start, end, girth in limbs:
            bend = 0.5 * (start + end) + self.rng.normal(0.0, 0.03 * tree.tall, 3)
            paths.append((project(np.array([start, bend, end])).tolist(), girth * size))
            self.sheet.stroke(
                paths[-1][0], stick(bark[1]), 0.8, max(paths[-1][1], THIN)
            )
        if size * thick > 4.0:  # (the side of the trunk that faces the sun)
            aside = np.array([-0.3 * thick, 0.0, 0.0])
            ends = (base + aside, fork + aside)
            self.rail(ends, (bark[0], None), 0.7, 0.4 * thick * size)
        self.foliage(tree, masses)
        for path, width in paths[1:]:
            if width > 2.5:
                gaps = (self.rng.random(3) < 0.6) * 0.75
                self.sheet.stroke(path, stick(bark[1]), gaps.tolist(), 0.8 * width)

    def foliage(self, tree: Tree, masses: np.ndarray) -> None:
        """The leaves of a tree: dabs clustered in its masses, the deep ones
        first, the light ones over them on the side the sun is; where a
        building shades them they keep to the colours of the shade."""
        rng = self.rng
        size = rise(tree.z)
        dab = float(np.clip(0.42 * size, 3.0, STICK))
        spots, tone = [], []
        for x, y, z, radius in masses:
            count = int(np.clip(1.25 * math.pi * (radius * size / dab) ** 2, 3, 320))
            facing = rng.normal(0.0, 1.0, (count, 3))
            facing /= np.linalg.norm(facing, axis=1)[:, None]
            facing[:, 2] = -np.abs(facing[:, 2])  # (the side the eye sees)
            deep = rng.uniform(0.3, 1.0, count) ** 0.5
            spots.append(np.array([x, y, z]) + radius * deep[:, None] * facing)
            tone.append(facing @ SUN - 0.7 * (1.0 - deep) + rng.normal(0.0, 0.2, count))
        spots, tone = np.concatenate(spots), np.concatenate(tone)
        lit = sunlit(self.scene.boxes, spots)
        colours = [far(rgb(code), tree.z) for code in LEAVES[tree.leaf]]
        places = project(spots)
        for k in np.argsort(tone):
            step = 2 if tone[k] < 0.05 else (1 if tone[k] < 0.55 else 0)
            colour = self.vary(colours[step + (0 if lit[k] else 3)], 0.02)
            turn = rng.uniform(0.0, math.pi)
            reach = 0.5 * dab * rng.uniform(0.6, 1.6)
            reach = reach * np.array([math.cos(turn), math.sin(turn)])
            path = [places[k] - reach, places[k] + reach]
            self.sheet.stroke(path, stick(colour), float(rng.uniform(0.5, 0.85)), dab)

    def car(self, car: Car) -> None:
        """A car, nose toward the eye: its shadow, the flank the eye sees,
        its front, then bonnet, windscreen and roof."""
        pair = distant(PAINTS[car.paint], car.z)
        sides = (car.x - 0.5 * car.wide, car.x + 0.5 * car.wide)
        self.car_shadow(car, sides)
        if car.x > EYE[0]:
            self.flank(car, sides[0], pair)
        else:  # (its other flank is turned from the sun)
            self.flank(car, sides[1], (pair[1], pair[1]))
        self.nose(car, pair, sides)
        self.cabin(car, pair, sides)

    def car_shadow(self, car: Car, sides) -> None:
        """The shadow a car lays on the sunlit roadway, beside it and beyond
        it, and the dark under it."""
        left, right = sides
        back = car.z + car.long
        reach = car.tall * CAST
        dark = mix(far(ROADWAY[1], car.z), INK, 0.2)
        inner = lengthwise(right - 0.1, 0.0, car.z + 0.3, back + reach[2])
        start = car.z + 0.6 * reach[2]
        outer = lengthwise(right + reach[0], 0.0, start, back + reach[2])
        self.panel(inner, outer, (dark, None), 0.7, GROUND)
        self.slab((left + 0.1, right - 0.1), (0.0, 0.28), car.z, far(INK, car.z), 0.85)

    def flank(self, car: Car, x: float, pair: Tones) -> None:
        """The side of a car: doors, the windows over them, the wheels."""
        z, back, belt = car.z, car.z + car.long, 0.62 * car.tall
        glass = distant(GLASS[1], z)
        self.panel(lengthwise(x, 0.2, z, back), lengthwise(x, belt, z, back), pair, 0.8)
        roof = lengthwise(x, car.tall, z + 1.7, back - 0.4)
        self.panel(lengthwise(x, belt, z + 1.2, back - 0.2), roof, pair, 0.8)
        sill = lengthwise(x, belt + 0.06, z + 1.5, back - 0.5)
        head = lengthwise(x, car.tall - 0.1, z + 1.9, back - 0.7)
        self.panel(sill, head, glass, 0.8)
        for at in (z + 0.5, back - 1.25):
            tyre = lengthwise(x, 0.0, at, at + 0.7)
            self.panel(tyre, lengthwise(x, 0.62, at, at + 0.7), far(INK, z), 0.85)

    def nose(self, car: Car, pair: Tones, sides) -> None:
        """The front of a car, from the road up: tyres, bumper, grille and
        headlamps, the edge of the bonnet."""
        left, right = sides
        z, belt = car.z, 0.62 * car.tall
        dark = far(INK, z)
        for x in ((left, left + 0.26), (right - 0.26, right)):
            self.slab(x, (0.0, 0.5), z, dark, 0.85)
        bumper = (mix(pair[0], INK, 0.5), mix(pair[1], INK, 0.45))
        self.slab(sides, (0.2, 0.42), z, bumper, 0.8)
        self.slab(sides, (0.42, belt), z, pair, 0.85)
        self.slab((car.x - 0.5, car.x + 0.5), (0.46, 0.66), z, dark, 0.8)
        lamp = distant((WHITE, HAZE), z)
        for x in ((left + 0.08, left + 0.4), (right - 0.4, right - 0.08)):
            self.slab(x, (0.52, 0.7), z, lamp, 0.85)

    def cabin(self, car: Car, pair: Tones, sides) -> None:
        """The top of a car, as the eye looks down on it: the bonnet, the
        windscreen between its pillars with the sky in it, the roof; a cab's
        roof light."""
        left, right = sides
        z, belt, tall = car.z, 0.62 * car.tall, car.tall
        top = raised(pair, 0.35)
        glass = distant(GLASS[1], z)
        sky = (mix(glass[0], HAZE, 0.45), mix(glass[1], HAZE, 0.25))

        def across(inset: float, y: float, behind: float) -> tuple:
            return level(left + inset, right - inset, y, z + behind)

        hood, brow = across(0.0, belt + 0.08, 1.1), across(0.12, tall, 1.9)
        self.panel(across(0.0, belt, 0.0), hood, top, 0.85)
        self.panel(across(0.04, belt + 0.08, 1.1), brow, pair, 0.8)
        middle = across(0.19, 0.5 * (belt + tall) + 0.06, 1.5)
        self.panel(across(0.16, belt + 0.14, 1.15), middle, glass, 0.85)
        self.panel(middle, across(0.22, tall - 0.06, 1.85), sky, 0.85)
        self.panel(brow, across(0.12, tall + 0.02, 3.4), top, 0.85)
        for x in ((left - 0.16, left - 0.02), (right + 0.02, right + 0.16)):
            self.slab(x, (belt + 0.06, belt + 0.2), z + 1.2, far(INK, z), 0.8)
        if car.cab:
            lamp = distant((WHITE, HAZE), z)
            light = (car.x - 0.27, car.x + 0.27)
            self.slab(light, (tall + 0.02, tall + 0.17), z + 2.3, lamp, 0.9)

    def walker(self, walker: Walker) -> None:
        """Someone walking: shadow, legs in mid-stride, coat and sleeves,
        head and hair."""
        x, z, tall = walker.x, walker.z, walker.tall
        size = rise(z)
        self.cast((x, 0.0, z), tall, 0.3)
        coat = distant(COATS[walker.coat], z)
        legs = distant(COATS["navy" if walker.coat == "black" else "black"], z)
        hip = 0.47 * tall
        for way in (-1.0, 1.0):
            foot = (x + way * (0.05 + walker.stride), 0.0, z)
            self.rail(((x + 0.07 * way, hip + 0.05, z), foot), legs, 0.8, 0.14 * size)
        self.rail(((x, 0.85 * tall, z), (x, hip - 0.08, z)), coat, 0.85, 0.4 * size)
        sleeve = (mix(coat[0], coat[1], 0.6), mix(coat[1], INK, 0.2))
        ends = ((x + 0.14, 0.83 * tall, z), (x + 0.14, hip, z))
        self.rail(ends, sleeve, 0.7, 0.13 * size)
        for way in (-1.0, 1.0):
            ends = ((x + 0.2 * way, 0.8 * tall, z), (x + 0.25 * way, 0.5 * tall, z))
            self.rail(ends, sleeve, 0.65, 0.09 * size)
        head = ((x, 0.965 * tall, z), (x, 0.9 * tall, z))
        self.rail(head, distant(SKIN, z), 0.85, 0.19 * size)
        self.rail(level(x - 0.05, x + 0.07, tall, z), far(INK, z), 0.8, 0.1 * size)

    def lamp(self, lamp: Lamp) -> None:
        """A street lamp: its shadow, the post, the arm and the lantern."""
        x, z = lamp.side * (KERB + 0.5), lamp.z
        self.cast((x, 0.0, z), lamp.tall, 0.18)
        if z < 3.3:
            return
        iron = distant(IRON, z)
        width = max(0.16 * rise(z), THIN)
        self.rail(((x, 0.0, z), (x, lamp.tall, z)), iron, 0.85, width, hard=True)
        reach = (x - lamp.side * 2.4, lamp.tall + 0.5, z)
        self.rail(((x, lamp.tall, z), reach), iron, 0.8, 0.7 * width, hard=True)
        head = (reach[0] - lamp.side * 0.5, reach[1] - 0.05, z)
        self.rail((reach, head), raised(iron, 0.4), 0.85, 1.6 * width)

    def hydrant(self, hydrant: Hydrant) -> None:
        """A fire hydrant: its shadow, a stout post with a cap and a nozzle
        each side."""
        x, z = hydrant.x, hydrant.z
        size = rise(z)
        self.cast((x, 0.0, z), 0.75, 0.28)
        body = distant(HYDRANT, z)
        self.rail(((x, 0.0, z), (x, 0.62, z)), body, 0.85, 0.26 * size)
        self.rail(level(x - 0.2, x + 0.2, 0.42, z), body, 0.8, 0.14 * size)
        cap = ((x - 0.05, 0.7, z), (x + 0.02, 0.74, z))
        self.rail(cap, raised(body, 0.45), 0.85, 0.2 * size)

    def signal(self, signal: Signal) -> None:
        """A traffic signal: the post at the corner, the arm over the
        roadway, the yellow box that hangs from it and the lamp that is lit."""
        x, z = signal.side * (KERB + 0.6), signal.z
        size = rise(z)
        iron = distant(IRON, z)
        post = ((x, 0.0, z), (x, 6.8, z))
        self.rail(post, iron, 0.8, max(0.2 * size, THIN), hard=True)
        end = x - signal.side * 5.5
        self.rail(((x, 6.6, z), (end, 7.1, z)), iron, 0.8, THIN, hard=True)
        box = ((end, 7.0, z), (end, 6.0, z))
        self.rail(box, distant(SIGNAL, z), 0.85, max(0.4 * size, 3.0))
        red = ((end, 6.75, z), (end, 6.65, z))
        self.rail(red, far(rgb("#e2402c"), z), 0.9, max(0.22 * size, THIN))


class Picture(Life):
    """The whole drawing, stage by stage."""

    def accents(self) -> None:
        """The last stage: the brightest touches, with a firm hand: the sun
        on the bonnets of the cabs, on the paint of the crossing, on leaves
        still falling."""
        rng = self.rng
        glare = rgb("#fdea90")
        for car in CARS:
            if car.cab and car.z < 60.0:
                y = 0.62 * car.tall + 0.05
                reach = 0.5 * car.wide - 0.25
                ends = level(car.x - reach, car.x + reach - 0.25, y, car.z + 0.6)
                self.rail(ends, (glare, None), 0.9, 0.07 * rise(car.z))
        for x in np.arange(-6.3, 6.4, 1.2):
            for _ in range(3):
                at = x + rng.uniform(-0.2, 0.2)
                near = rng.uniform(5.3, 7.8)
                ends = lengthwise(at, 0.0, near, near + rng.uniform(0.3, 0.8))
                self.rail(ends, (WHITE, None), 0.9, 9.0, GROUND, worn=0.3)
        for tree in self.trees:
            if tree.side > 0 and tree.z < 75.0:
                self.falling(tree)

    def falling(self, tree: Tree) -> None:
        """A few leaves coming down from a tree, in the air."""
        light, middle = (far(rgb(code), tree.z) for code in LEAVES[tree.leaf][:2])
        size = float(np.clip(0.1 * rise(tree.z), THIN, 5.0))
        for _ in range(5):
            x = tree.x + self.rng.normal(-1.0, 2.2)
            y = self.rng.uniform(0.6, 0.45 * tree.tall)
            z = tree.z + self.rng.normal(0.0, 1.5)
            self.rail(((x, y, z), (x + 0.07, y - 0.06, z)), (light, middle), 0.85, size)


def main() -> None:
    art = Picture()
    art.block_in()
    art.architecture()
    art.pavement()
    art.life()
    art.accents()
    out = Path(__file__).parent / "output" / "broadway.png"
    write_png(str(out), np.round(art.sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
