"""A sketch of an art deco tower, in charcoal on a cream charcoal paper.

Run from the repository root: ``python examples/deco.py``. Writes
``examples/output/deco.png``.

An imagined setback skyscraper of the 1930s, seen from the street, corner on,
so that it towers. It is built as its architect would build it, in three
dimensions, and only then looked at: a base with an arcade and a door; a
shaft that steps back tier by tier; out of the middle of each face a bay, and
out of the bay a frontispiece, each rising past the setback behind it; a
ledge round the top of every one of them; a stepped crown and a mast. A small
perspective camera tells which face is seen at each pixel and the point of
the building seen there, and a ray from that point toward the sun tells
whether a bay or a ledge stands in its light.

Every mark is laid through that knowledge, along the form it describes. The
faces turned from the sun are toned with strokes up the face, as the piers
run, and rubbed with a stump; the shadows that bays and ledges cast are
hatched the way the light falls across the wall; each strip of windows is
drawn up its recess between two piers, then its windows floor by floor; each
ledge gets the dark of its underside along itself; each edge is drawn only as
firmly as the tones on its two sides leave it in need of a line. Ornament is
where such a building carries it: a rising sun over the door, a fan of rays
up the bays of the crown, a band of chevrons above them. The sky is vine
charcoal rubbed with a finger, a bank of cloud heavy behind the lit side and
clearing behind the shadow side, so that light stands against dark all the
way up; the buildings are lifted out of it with a kneaded eraser, so that the
dark comes right up to their edges.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from murex import filters, noise, write_png

from fusain import Medium, Sheet, charcoal, papers, tools

WIDTH, HEIGHT = 900, 1300
#: Charcoal paper: a felted tooth with no pattern to it, on a thick, soft
#: sheet that lets the stick down into it, so that the darks go deep and the
#: grain shows in a tone as flecks of paper; warm cream.
PAPER = papers.CARTRIDGE.with_(
    depth=0.9, bite=0.55, give=0.6, color="#eee8d7", name="charcoal paper"
)

#: The camera: where the eye is (x and y on the ground, z up, in metres),
#: what it looks at, its focal length in pixels and where on the sheet the
#: point it looks at lands.
EYE = np.array([-262.0, -205.0, 2.0])
TARGET = np.array([0.0, 0.0, 170.0])
FOCAL = 1340.0
CENTRE = (432.0, 580.0)
#: Toward the sun: from the left, a little behind the eye and fairly high, so
#: that the faces turned to -x are lit and those turned to -y are not.
SUN = np.array([-0.62, 0.5, 0.6]) / math.sqrt(0.62**2 + 0.5**2 + 0.6**2)

#: The tower, tier by tier from the street: half its width, the height of its
#: top, and the bay that stands out of the middle of each face (half its
#: width, half the width of the frontispiece in its middle, how far it stands
#: out). A bay rises past the top of its tier, the frontispiece higher still.
TIERS = (
    (43.0, 22.0, 0.0, 0.0, 0.0),
    (36.0, 126.0, 18.0, 9.0, 4.0),
    (28.0, 184.0, 13.5, 6.5, 3.4),
    (21.5, 228.0, 10.5, 0.0, 2.8),
    (15.5, 266.0, 10.0, 0.0, 2.2),
    (11.0, 292.0, 0.0, 0.0, 0.0),
    (7.0, 312.0, 0.0, 0.0, 0.0),
    (4.0, 328.0, 0.0, 0.0, 0.0),
)
#: How far up the next tier a bay and a frontispiece rise, and how far the
#: frontispiece stands out of its bay.
BAY_RISE, FRONT_RISE, FRONT_OUT = 0.4, 0.72, 1.5
#: The tier the crown starts at: its bays carry a fan of rays, not windows.
CROWN = 4
#: The mast: the heights of its foot and of its tip, and half its width at
#: the foot.
MAST = (328.0, 372.0, 1.5)
#: A ledge runs round the top of every tier and bay: how far it stands out
#: and how deep it is.
LEDGE_OUT, LEDGE_DEEP = 0.9, 1.6
#: A floor, a bay between two piers, the solid corner of a face and the
#: parapet over the top windows, in metres.
FLOOR, PITCH, CORNER, PARAPET = 4.0, 4.6, 1.4, 3.2
#: The lower blocks round the foot of the tower: (x from, to), (y from, to)
#: and height.
BLOCKS = (
    ((-178.0, -128.0), (-62.0, -12.0), 26.0),
    ((75.0, 135.0), (-70.0, 0.0), 44.0),
    ((120.0, 190.0), (-150.0, -90.0), 20.0),
    ((-230.0, -180.0), (60.0, 120.0), 44.0),
)

#: What is seen at a pixel that no face covers.
SKY, GROUND, NEEDLE = 0, -1, -2
#: The way the clouds drift across the sheet, and the strokes of the sky with
#: them, in degrees.
SLANT = 58.0
#: Straight up a face, as its piers run.
UP = (0.0, 1.0)


def _basis() -> np.ndarray:
    """The camera's right, up and forward, as the rows of a matrix."""
    forward = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(right, forward), forward])


BASIS = _basis()


def project(points) -> np.ndarray:
    """Where on the sheet the camera sees ``points`` (..., 3): (..., 2)."""
    seen = (np.asarray(points, dtype=np.float64) - EYE) @ BASIS.T
    across, up, depth = np.moveaxis(seen, -1, 0)
    x, y = CENTRE[0] + FOCAL * across / depth, CENTRE[1] - FOCAL * up / depth
    return np.stack([x, y], axis=-1)


# --------------------------------------------------------------------------
# the building


@dataclass(frozen=True)
class Box:
    """A block of the building: its low and high corners (x, y, z), what it
    is, the tier it belongs to and, for its face turned to -x and for the one
    turned to -y, the stretches that hold windows: (from, to) along the face
    and the height the windows start at."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    part: str
    tier: int = 0
    bays: tuple[tuple, tuple] = ((), ())


@dataclass(frozen=True)
class Face:
    """An upright face of a box, turned to -x (``axis`` 0, lit) or to -y
    (1, in shadow): from ``lo`` to ``hi`` along the other axis and from
    ``z0`` to ``z1`` in height. ``ident`` is what the map of what is seen
    holds where it shows."""

    ident: int
    axis: int
    box: Box

    @property
    def lo(self) -> float:
        return self.box.lo[1 - self.axis]

    @property
    def hi(self) -> float:
        return self.box.hi[1 - self.axis]

    @property
    def z0(self) -> float:
        return self.box.lo[2]

    @property
    def z1(self) -> float:
        return self.box.hi[2]

    @property
    def lit(self) -> bool:
        return self.axis == 0

    @property
    def wide(self) -> float:
        """How wide it is on the sheet, in pixels, half way up."""
        z = 0.5 * (self.z0 + self.z1)
        return float(np.hypot(*(self.point(self.hi, z) - self.point(self.lo, z))))

    def point(self, along, z) -> np.ndarray:
        """Where on the sheet the point ``along`` the face at height ``z`` is."""
        along, z = np.broadcast_arrays(
            np.asarray(along, dtype=np.float64), np.asarray(z, dtype=np.float64)
        )
        at = np.full(along.shape, self.box.lo[self.axis])
        order = (at, along, z) if self.axis == 0 else (along, at, z)
        return project(np.stack(order, axis=-1))

    def lines(self, way, gap: float) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Lines over the face, as their two ends on the sheet: they run
        along ``way`` (so far along the face, so far up it) and are about
        ``gap`` pixels apart. Straight on the building, so straight on the
        sheet, and closing up with distance as the face does."""
        way = np.asarray(way, dtype=np.float64) / np.hypot(*way)
        across = np.array([-way[1], way[0]])
        low, high = np.array([self.lo, self.z0]), np.array([self.hi, self.z1])
        middle = 0.5 * (low + high)
        scale = np.hypot(*(self.point(*(middle + across)) - self.point(*middle)))
        corners = [(s, z) for s in (self.lo, self.hi) for z in (self.z0, self.z1)]
        reach = [s * across[0] + z * across[1] for s, z in corners]
        step = gap / max(float(scale), 1e-6)
        for offset in np.arange(min(reach) + 0.5 * step, max(reach), step):
            base = offset * across
            ends = _clip(base, way, low, high)
            if ends is not None:
                enter, leave = (self.point(*(base + t * way)) for t in ends)
                yield enter, leave


def _clip(base, way, low, high) -> tuple[float, float] | None:
    """How far along ``way`` from ``base`` a line enters and leaves the
    rectangle from ``low`` to ``high``, or None if it misses it."""
    enter, leave = -np.inf, np.inf
    for k in (0, 1):
        if abs(way[k]) < 1e-9:
            if not low[k] <= base[k] <= high[k]:
                return None
            continue
        one, other = (low[k] - base[k]) / way[k], (high[k] - base[k]) / way[k]
        enter, leave = max(enter, min(one, other)), min(leave, max(one, other))
    return (enter, leave) if leave > enter else None


def ident(index: int, axis: int) -> int:
    """What the map of what is seen holds for a face of the box ``index``:
    ``axis`` 0 for the one turned to -x, 1 to -y, 2 for its underside."""
    return 3 * index + axis + 1


def cross(half: float, wide: float, heights, tier: int, bays) -> list[Box]:
    """The two boxes of a cross in plan: arms ``wide`` either side of the
    axes, reaching ``half`` from the middle, between two ``heights``.
    ``bays`` are the windows of the front of an arm and of its side."""
    (z0, z1), (front, side) = heights, bays
    return [
        Box((-half, -wide, z0), (half, wide, z1), "bay", tier, (front, side)),
        Box((-wide, -half, z0), (wide, half, z1), "bay", tier, (side, front)),
    ]


def ledge(box: Box) -> Box:
    """The ledge round the top of a box."""
    (x0, y0, _), (x1, y1, top), out = box.lo, box.hi, LEDGE_OUT
    low, high = (x0 - out, y0 - out, top - LEDGE_DEEP), (x1 + out, y1 + out, top)
    return Box(low, high, "ledge", box.tier)


def tier_boxes(k: int) -> list[Box]:
    """The boxes of one tier: its body, its bay and its frontispiece, each
    with the stretches of its faces that hold windows. The piers at the
    corners are solid."""
    half, top, wide, inner, out = TIERS[k]
    below = TIERS[k - 1][1] if k else 0.0
    if not wide:
        span = ((-half + CORNER, half - CORNER, below),)
        return [Box((-half, -half, below), (half, half, top), "tier", k, (span, span))]
    wings = ((-half + CORNER, -wide, below), (wide, half - CORNER, below))
    boxes = [Box((-half, -half, below), (half, half, top), "tier", k, (wings, wings))]
    step, behind = TIERS[k + 1][1] - top, TIERS[k + 1][0]
    # (the side of a bay shows above the setback, back to the next tier)
    side = ((-half - out + CORNER, -behind, top),)
    if inner:
        front = ((-wide + CORNER, -inner, below), (inner, wide - CORNER, below))
    elif k == CROWN:
        front = ()
    else:
        front = ((-wide + CORNER, wide - CORNER, below),)
    rise = top + BAY_RISE * step
    boxes += cross(half + out, wide, (below, rise), k, (front, side))
    if inner:
        front = ((-inner + CORNER, inner - CORNER, below),)
        rise = top + FRONT_RISE * step
        boxes += cross(half + out + FRONT_OUT, inner, (below, rise), k, (front, ()))
    return boxes


def tower() -> list[Box]:
    """The boxes of the tower, then the ledges round their tops, in the same
    order."""
    boxes = [box for k in range(len(TIERS)) for box in tier_boxes(k)]
    return boxes + [ledge(box) for box in boxes]


def neighbours() -> list[Box]:
    """The lower blocks around the foot of the tower."""
    return [Box((x[0], y[0], 0.0), (x[1], y[1], top), "block") for x, y, top in BLOCKS]


def mast() -> tuple[np.ndarray, np.ndarray, float]:
    """The mast on the sheet: its foot, its tip and half its width at the
    foot."""
    foot, tip = project((0.0, 0.0, MAST[0])), project((0.0, 0.0, MAST[1]))
    scale = np.linalg.norm(project(np.array([0.0, 0.0, MAST[0]]) + BASIS[0]) - foot)
    return foot, tip, float(MAST[2] * scale)


# --------------------------------------------------------------------------
# what is seen where, and how the light falls on it


def hits(start: np.ndarray, rays: np.ndarray, box: Box):
    """Where rays meet a box: how far along each ray (infinite if it misses)
    and the axis of the face it enters by."""
    with np.errstate(divide="ignore", invalid="ignore"):
        one = (np.asarray(box.lo) - start) / rays
        other = (np.asarray(box.hi) - start) / rays
    near = np.nan_to_num(np.minimum(one, other), nan=-np.inf)
    far = np.nan_to_num(np.maximum(one, other), nan=np.inf)
    enter, leave = near.max(axis=-1), far.min(axis=-1)
    return np.where((enter < leave) & (enter > 0), enter, np.inf), near.argmax(axis=-1)


def cast(boxes: list[Box]) -> tuple[np.ndarray, np.ndarray]:
    """What the eye sees at each pixel, (H, W): the ``ident`` of a face,
    ``SKY`` or ``GROUND``; and the point of the scene it sees there."""
    px, py = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    across, up = (px - CENTRE[0]) / FOCAL, (CENTRE[1] - py) / FOCAL
    rays = across[..., None] * BASIS[0] + up[..., None] * BASIS[1] + BASIS[2]
    down = rays[..., 2] < 0
    seen = np.where(down, GROUND, SKY).astype(np.int32)
    depth = np.where(down, -EYE[2] / np.minimum(rays[..., 2], -1e-9), np.inf)
    for index, box in enumerate(boxes):
        reach, axis = hits(EYE, rays, box)
        nearer = reach < depth
        depth[nearer] = reach[nearer]
        seen[nearer] = ident(index, 0) + axis[nearer]
    return seen, EYE + np.where(np.isfinite(depth), depth, 0.0)[..., None] * rays


def shadows(boxes: list[Box], seen: np.ndarray, where: np.ndarray):
    """Which pixels show a face turned from the sun, and which a face turned
    to it that a box stands in the light of: two maps, (H, W)."""
    axis = (seen - 1) % 3
    normal = np.zeros(where.shape)
    for k in range(3):  # (the eye sees the faces turned to -x, to -y and down)
        normal[..., k] = np.where((seen > 0) & (axis == k), -1.0, 0.0)
    normal[seen == GROUND] = (0.0, 0.0, 1.0)
    turned = (normal @ SUN <= 0) & (seen != SKY)
    start = where + 0.02 * normal
    rays = np.broadcast_to(SUN, where.shape)
    hidden = np.zeros(seen.shape, bool)
    for box in boxes:
        hidden |= np.isfinite(hits(start, rays, box)[0])
    return turned, hidden & ~turned & (seen != SKY)


def fill(seen: np.ndarray, polygon: np.ndarray, mark: int) -> None:
    """Mark the pixels inside ``polygon`` in ``seen``."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH] + 0.5
    inside = np.zeros(xs.shape, bool)
    ax, ay = polygon[-1]
    for bx, by in polygon:
        if ay != by:
            crosses = (ys >= min(ay, by)) & (ys < max(ay, by))
            inside ^= crosses & (xs < ax + (ys - ay) * (bx - ax) / (by - ay))
        ax, ay = bx, by
    seen[inside] = mark


def inset(mask: np.ndarray, by: int) -> np.ndarray:
    """``mask`` (0 or 1) without the pixels nearer than ``by`` to its edge:
    where the middle of a tip that wide may go."""
    return 1.0 - filters.max_filter(1.0 - mask, by) if by > 0 else mask


def margin(x: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """How much of a tone is laid near the edges of the sheet, 0..1, (H, W):
    a sketch stops short of them, raggedly."""
    edge = np.minimum(np.minimum(x, WIDTH - x), np.minimum(y, HEIGHT - y))
    ragged = noise.value_noise_2d(HEIGHT, WIDTH, 70.0, rng) - 0.5
    return filters.smoothstep(4.0, 30.0, edge + 44.0 * ragged)


def sky_tone(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How dark the sky is wanted, 0..1, (H, W): a heavy bank of cloud
    behind the lit side of the tower, clearing behind its shadow side, paler
    toward the street, in long streaks that drift on the slant."""
    axis, tip = project((0.0, 0.0, MAST[1]))
    rng = np.random.default_rng(3)
    streaks = noise.aniso_noise(HEIGHT, WIDTH, 420.0, 55.0, SLANT, rng)
    streaks = (streaks - streaks.mean()) / streaks.std()
    # (the edge of the bank passes behind the mast, so that the needle is
    # white against the dark, then behind the crown, whose shadow side is
    # dark against the clear sky)
    foot = project((0.0, 0.0, MAST[0]))[1]
    lean = 37.0 * (1.0 - filters.smoothstep(foot - 10.0, foot + 40.0, y))
    edge = axis - 12.0 + lean + 0.6 * np.maximum(tip + 40.0 - y, 0.0) + 14.0 * streaks
    clear = filters.smoothstep(-10.0, 60.0, x - edge)
    glow = np.hypot(x - axis - 140.0, (y - 520.0) * 0.55) / 420.0
    right = 0.14 + 0.5 * filters.smoothstep(0.35, 1.4, glow) + 0.06 * streaks
    left = 0.95 - 0.35 * filters.smoothstep(700.0, 1250.0, y)
    left *= 0.84 + 0.16 * filters.smoothstep(-430.0, -150.0, x - axis) + 0.07 * streaks
    return np.clip(left * (1.0 - clear) + right * clear, 0.0, 1.0)


# --------------------------------------------------------------------------
# strokes


@dataclass(frozen=True)
class Touch:
    """What a stroke is made with and how: a stick of charcoal or a tool
    that rubs, how firmly it is pressed and how wide its tip is."""

    what: Medium | tools.Tool
    firm: float
    width: float


def vine(firm: float, width: float) -> Touch:
    return Touch(charcoal.VINE, firm, width)


def black(firm: float, width: float) -> Touch:
    return Touch(charcoal.COMPRESSED, firm, width)


def pencil(firm: float, width: float) -> Touch:
    return Touch(charcoal.PENCIL, firm, width)


def stump(firm: float, width: float = 9.0) -> Touch:
    return Touch(tools.STUMP, firm, width)


def finger(firm: float, width: float = 22.0) -> Touch:
    return Touch(tools.FINGER, firm, width)


def sample(image: np.ndarray, x, y) -> np.ndarray:
    """What ``image``, (H, W), holds at points of the sheet, and nothing off
    it."""
    col = np.floor(np.asarray(x)).astype(np.intp)
    row = np.floor(np.asarray(y)).astype(np.intp)
    inside = (col >= 0) & (col < WIDTH) & (row >= 0) & (row < HEIGHT)
    value = image[np.clip(row, 0, HEIGHT - 1), np.clip(col, 0, WIDTH - 1)]
    return np.where(inside, value, 0.0)


def along(image: np.ndarray | None, start, end, firm: float, lift: float) -> Callable:
    """The pressure of a straight stroke from ``start`` to ``end`` as a
    function of u: ``firm`` times what ``image`` holds where the tip is,
    lighter by ``lift`` where it lands and where it leaves the paper."""

    def pressure(u):
        u = np.asarray(u, dtype=np.float64)
        ends = np.sqrt(np.sin(np.pi * np.clip(u, 0.03, 0.97)))
        wanted = firm * (1.0 - lift + lift * ends)
        if image is None:
            return wanted
        x, y = (start[k] + (end[k] - start[k]) * u for k in (0, 1))
        return wanted * sample(image, x, y)

    return pressure


def runs(image: np.ndarray | None, start: np.ndarray, end: np.ndarray):
    """The stretches of the line from ``start`` to ``end`` where ``image``
    holds something: the tip is lifted over the rest."""
    if image is None:
        yield start, end
        return
    length = float(np.hypot(*(end - start)))
    u = np.linspace(0.0, 1.0, max(int(length / 1.5), 2))
    x, y = (start[k] + (end[k] - start[k]) * u for k in (0, 1))
    live = np.concatenate([[0], (sample(image, x, y) > 0.04).astype(np.int8), [0]])
    turns = np.flatnonzero(np.diff(live))
    for first, last in zip(turns[0::2], turns[1::2] - 1, strict=True):
        yield start + (end - start) * u[first], start + (end - start) * u[last]


def pieces(start, end, rng, reach: float = 240.0):
    """A long line as the shorter strokes a hand draws it in, each begun a
    little before the last one ended."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    count = max(int(np.hypot(*(end - start)) / reach + rng.uniform(0.2, 1.0)), 1)
    cuts = np.linspace(0.0, 1.0, count + 1)
    cuts[1:-1] += rng.uniform(-0.25, 0.25, count - 1) / count
    for a, b in itertools.pairwise(cuts):
        yield start + (end - start) * max(a - 0.04, 0.0), start + (end - start) * b


def wavering(start: np.ndarray, end: np.ndarray, rng, wobble: float) -> np.ndarray:
    """The points of a stroke from ``start`` to ``end`` as a hand draws it:
    nearly straight."""
    knots = max(int(np.hypot(*(end - start)) / 80.0), 1) + 1
    path = start + (end - start) * np.linspace(0.0, 1.0, knots)[:, None]
    path[1:-1] += rng.normal(0.0, wobble, (knots - 2, 2))
    return path


class Scene:
    """The sheet, the boxes of the buildings, which face is seen where and
    how the light falls on it."""

    def __init__(self, seed: int = 1931):
        self.sheet = Sheet(WIDTH, HEIGHT, paper=PAPER, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.boxes = tower()
        self.ledges = len(self.boxes) // 2  # (the first of the ledges)
        self.blocks = len(self.boxes)  # (the first of the blocks round about)
        self.boxes += neighbours()
        self.seen, self.where = cast(self.boxes)
        self.turned, self.hidden = shadows(self.boxes, self.seen, self.where)
        foot, tip, half = mast()
        fill(self.seen, np.array([foot - (half, 0), tip, foot + (half, 0)]), NEEDLE)
        self.block = self.seen >= ident(self.blocks, 0)
        self.sky = (self.seen == SKY) * 1.0
        x, y = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
        self.margin = margin(x, y, self.rng)
        self.air = sky_tone(x, y) * self.margin
        # how dark a face turned from the sun is wanted: the street sends
        # some light back up, so it deepens with height
        deep = 0.52 + 0.4 * filters.smoothstep(0.0, 300.0, self.where[..., 2])
        self.gloom = np.where(self.block, 0.9 * self.margin, deep)

    def faces(self, *parts: str) -> Iterator[tuple[Face, np.ndarray]]:
        """The upright faces of the boxes that are one of ``parts`` and that
        show, each with the map of where it shows (1 there, else 0)."""
        for index, box in enumerate(self.boxes):
            for axis in (0, 1):
                mine = (self.seen == ident(index, axis)) * 1.0
                if box.part in parts and mine.sum() >= 12:
                    yield Face(ident(index, axis), axis, box), mine

    def stroke(self, start, end, touch: Touch, image=None, wobble: float = 0.5):
        """One stroke from ``start`` to ``end`` on the sheet, pressed as
        firmly as ``touch`` says times what ``image`` holds along it, and
        lifted where that is nothing."""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        rubs = isinstance(touch.what, tools.Tool)
        for a, b in runs(image, start, end):
            if np.hypot(*(b - a)) < 1.0:
                continue
            bearing = along(image, a, b, touch.firm, 0.0 if rubs else 0.3)
            path = wavering(a, b, self.rng, wobble)
            if isinstance(touch.what, tools.Tool):
                self.sheet.smudge(path, touch.what, bearing, touch.width)
            else:
                self.sheet.stroke(path, touch.what, bearing, touch.width)

    def strokes(self, lines, touch: Touch, image, reach: float, wobble: float):
        """A stroke along each of ``lines``, in pieces no longer than about
        ``reach``, no two pressed quite alike."""
        for start, end in lines:
            for a, b in pieces(start, end, self.rng, reach):
                firm = touch.firm * self.rng.uniform(0.85, 1.1)
                self.stroke(a, b, Touch(touch.what, firm, touch.width), image, wobble)

    def hatch(self, face: Face, way, image, touch: Touch, gap: float) -> None:
        """Strokes over a face, along ``way`` on it and ``gap`` pixels apart,
        pressed as ``image`` says."""
        self.strokes(face.lines(way, gap), touch, image, 240.0, 0.5)

    def sweep(self, angle: float, gap: float, touch: Touch, image) -> None:
        """Strokes right across the sheet at ``angle`` degrees (0 runs across
        it, 90 up it), ``gap`` apart, pressed as ``image`` says."""
        turn = math.radians(angle)
        way = np.array([math.cos(turn), -math.sin(turn)])
        across = np.array([math.sin(turn), math.cos(turn)])
        middle = np.array([0.5 * WIDTH, 0.5 * HEIGHT])
        span = 0.5 * math.hypot(WIDTH, HEIGHT)
        offsets = np.arange(-span, span, gap)
        offsets += self.rng.normal(0.0, 0.15 * gap, len(offsets))
        lines = [
            (middle + o * across - span * way, middle + o * across + span * way)
            for o in offsets
        ]
        self.strokes(lines, touch, image, 420.0, 2.0)


# --------------------------------------------------------------------------
# the tones: sky, shade and cast shadow


def draw_sky(scene: Scene) -> None:
    """The sky: vine charcoal in long strokes on the slant of the clouds and
    across it, rubbed smooth with a finger; then compressed charcoal where
    the cloud is heavy, left as it falls."""
    dark, open_ = scene.air, scene.sky
    heavy = np.clip((dark - 0.5) * 2.2, 0.0, 1.0)
    scene.sweep(SLANT, 3.4, vine(1.0, 8.0), (0.06 + 0.8 * dark) * inset(open_, 1))
    scene.sweep(SLANT - 28.0, 3.4, vine(1.0, 8.0), 0.8 * dark * inset(open_, 1))
    scene.sweep(-12.0, 9.0, finger(0.8), inset(open_, 3))
    scene.sweep(100.0, 9.0, finger(0.8), inset(open_, 3))
    scene.sweep(SLANT + 5.0, 3.3, black(0.75, 5.0), heavy * inset(open_, 1))
    scene.sweep(SLANT - 6.0, 9.0, stump(0.4, 12.0), heavy * inset(open_, 3))
    scene.sweep(SLANT + 8.0, 5.0, black(0.7, 4.0), heavy**2 * inset(open_, 1))
    lift_out(scene)


def lift_out(scene: Scene) -> None:
    """What the sky's strokes ran over of the buildings, lifted back out to
    the paper with the point of a kneaded eraser, so that the sky's dark
    comes right up to their edges."""
    solid = 1.0 - scene.sky - (scene.seen == GROUND)
    rows, cols = np.nonzero(solid)
    area = (cols.min() - 4, rows.min() - 4, cols.max() + 5, rows.max() + 5)
    where = inset(solid, 2)

    def pressure(x, y):
        return 0.9 * sample(where, x, y)

    for angle in (90.0, 0.0):
        scene.sheet.rub(area, tools.KNEADED, pressure, 5.0, angle, 2.0, hand=0.0)


def draw_shade(scene: Scene) -> None:
    """The faces of the tower turned from the sun: vine charcoal in strokes
    up each face, as its piers run, rubbed in with a stump; then compressed
    charcoal over it high up, where less light comes back from the street."""
    high = np.clip((scene.gloom - 0.62) * 3.0, 0.0, 1.0)
    for face, mine in scene.faces("tier", "bay"):
        if face.lit:
            continue
        for _ in range(2):
            scene.hatch(face, UP, inset(mine, 2), vine(0.9, 6.0), 3.4)
        if face.wide > 24.0:
            scene.hatch(face, UP, inset(mine, 4), stump(0.7), 5.0)
        scene.hatch(face, UP, high * inset(mine, 1), black(0.8, 4.0), 3.2)
        if face.wide > 24.0:
            scene.hatch(face, UP, high * inset(mine, 4), stump(0.5), 6.0)


def draw_cast(scene: Scene) -> None:
    """The shadows that bays, ledges and the tower itself cast on the lit
    faces: hatched the way the light falls across each face."""
    for face, mine in scene.faces("tier", "bay", "block"):
        if not face.lit:
            continue
        near = face.box.part == "block"
        mine = mine * scene.hidden * (scene.margin if near else 1.0)
        fall = (SUN[1], SUN[2])
        scene.hatch(face, fall, mine, black(0.48 if near else 0.45, 3.4), 3.0)
        scene.hatch(face, UP, inset(mine, 1), vine(0.5, 4.0), 4.0)
        if face.wide > 12.0:
            scene.hatch(face, fall, inset(mine, 3), stump(0.35, 7.0), 5.0)


def draw_haze(scene: Scene) -> None:
    """The lit faces of the tower near the street: a breath of tone, rubbed
    smooth, so that the light is whitest high up."""
    haze = 0.5 * (1.0 - filters.smoothstep(22.0, 170.0, scene.where[..., 2]))
    for face, mine in scene.faces("tier", "bay"):
        if face.lit:
            mine = mine * (1.0 - scene.hidden)
            scene.hatch(face, UP, haze * inset(mine, 2), vine(0.8, 5.0), 4.0)
            scene.hatch(
                face, UP, (haze > 0.05) * inset(mine, 5), finger(0.6, 14.0), 9.0
            )


def draw_blocks(scene: Scene) -> None:
    """The blocks round about: dark where they are turned from the sun, a
    quiet half tone where they face it, so that the tower keeps the light;
    and their tops against the sky, each drawn along itself."""
    for face, mine in scene.faces("block"):
        if face.lit:
            mine = mine * (1.0 - scene.hidden)
            scene.hatch(face, UP, inset(mine, 2) * scene.margin, vine(0.8, 5.0), 4.0)
            scene.hatch(face, UP, inset(mine, 4), stump(0.6), 7.0)
            continue
        scene.hatch(face, UP, scene.gloom * inset(mine, 2), black(0.7, 5.0), 3.6)
        scene.hatch(face, UP, inset(mine, 4), stump(0.6), 6.0)
        scene.hatch(face, UP, scene.gloom * inset(mine, 1), black(0.7, 4.0), 3.2)
    rim = filters.max_filter(scene.sky, 3) * filters.max_filter(scene.block * 1.0, 3)
    for box in scene.boxes[scene.blocks :]:
        (x0, y0, _), (x1, y1, top) = box.lo, box.hi
        for end in ((x0, y1, top), (x1, y0, top)):
            ends = project((x0, y0, top)), project(end)
            scene.stroke(*ends, black(0.8, 5.0), rim * scene.margin)


def draw_ground(scene: Scene) -> None:
    """The street: level strokes, dark."""
    ground = (scene.seen == GROUND) * scene.margin
    scene.sweep(1.5, 3.0, black(0.8, 4.0), ground)


# --------------------------------------------------------------------------
# piers, windows, ledges and edges


def strips(face: Face) -> Iterator[tuple[float, float, float, float]]:
    """The strips of windows between the piers of a face: for each, where
    along the face its middle is, how wide it is and the heights it starts
    and ends at."""
    top = face.z1 - PARAPET
    for start, end, foot in face.box.bays[face.axis]:
        count = max(round((end - start) / PITCH), 1)
        pitch = (end - start) / count
        for k in range(count):
            yield start + (k + 0.5) * pitch, 0.42 * pitch, foot, top


def draw_strip(scene: Scene, face: Face, mine: np.ndarray, strip) -> None:
    """One strip of windows, from the foot of its face to the parapet: the
    recess between two piers, glass and spandrels, drawn up the face; on a
    lit face the shadow that the pier on the side of the sun throws into it;
    then its windows floor by floor, a dab each, fainter in the haze of the
    street. High in the crown the strips are slits."""
    at, wide, foot, top = strip
    rng, middle = scene.rng, 0.5 * (foot + top)
    span = face.point(at + 0.5 * wide, middle) - face.point(at - 0.5 * wide, middle)
    tip = float(np.clip(0.8 * np.hypot(*span), 2.4, 6.5))
    stick = pencil if face.lit else black
    if face.box.tier >= CROWN:
        scene.stroke(
            face.point(at, foot + 1.0), face.point(at, top), stick(0.85, tip), mine
        )
        return
    clear = 0.55 + 0.45 * float(filters.smoothstep(-20.0, 120.0, middle))
    recess = [(face.point(at, foot), face.point(at, top))]
    scene.strokes(
        recess, stick((0.22 if face.lit else 0.7) * clear, tip), mine, 150.0, 0.3
    )
    if face.lit:
        jamb = at + 0.5 * wide * np.sign(SUN[1])
        shadow = [(face.point(jamb, foot), face.point(jamb, top))]
        scene.strokes(shadow, stick(0.42 * clear, 2.0), mine, 200.0, 0.3)
    for floor in np.arange(math.ceil(foot / FLOOR) * FLOOR, top - 2.5, FLOOR):
        clear = 0.55 + 0.45 * float(filters.smoothstep(-20.0, 120.0, floor))
        blind = (
            0.4 if rng.random() < 0.1 else 1.0
        )  # (a blind drawn, a pane that glints)
        firm = 0.8 * clear * rng.uniform(0.75, 1.1) * blind
        ends = face.point(at, floor + 1.3), face.point(at, floor + 2.8)
        scene.stroke(*ends, stick(firm, 0.85 * tip), mine, wobble=0.0)


def draw_windows(scene: Scene) -> None:
    """The strips of windows of every face of the tower above its base."""
    for face, mine in scene.faces("tier", "bay"):
        if face.box.tier not in (0, CROWN + 1):  # (those carry ornament instead)
            for strip in strips(face):
                draw_strip(scene, face, mine, strip)


def draw_ledges(scene: Scene) -> None:
    """The ledges: seen from the street, each shows its underside, dark; it
    throws a line of shadow on the lit wall under it, and is dark itself
    where it is turned from the sun. Each line runs along its ledge."""
    for index in range(scene.ledges, scene.blocks):
        box = scene.boxes[index]
        soffit = filters.max_filter((scene.seen == ident(index, 2)) * 1.0, 1)
        for axis in (0, 1):
            face = Face(ident(index, axis), axis, box)
            a, b = face.point(face.lo, face.z0), face.point(face.hi, face.z0)
            scene.stroke(a + (0.0, 1.0), b + (0.0, 1.0), black(0.9, 2.8), soffit)
            if face.lit:
                wall = Face(0, axis, scene.boxes[index - scene.ledges])
                a, b = (wall.point(s, face.z0 - 0.5) for s in (wall.lo, wall.hi))
                scene.stroke(a, b, pencil(0.8, 3.0), scene.hidden * 1.0)
            else:
                middle = 0.5 * (face.z0 + face.z1)
                a, b = (face.point(s, middle) for s in (face.lo, face.hi))
                scene.stroke(a, b, black(0.7, 3.5), (scene.seen == face.ident) * 1.0)


def edges(scene: Scene) -> Iterator[tuple[tuple, tuple]]:
    """The edges of the boxes that the eye may see, as their two ends in the
    scene: the three upright ones of each box, up to its ledge, and the top
    of each ledge and of each block."""
    for index, box in enumerate(scene.boxes):
        (x0, y0, z0), (x1, y1, z1) = box.lo, box.hi
        if box.part != "ledge":
            top = z1 - (LEDGE_DEEP if index < scene.ledges else 0.0)
            for x, y in ((x0, y1), (x0, y0), (x1, y0)):
                yield (x, y, z0), (x, y, top)
        if box.part in ("ledge", "block"):
            yield (x0, y0, z1), (x0, y1, z1)
            yield (x0, y0, z1), (x1, y0, z1)


def draw_edges(scene: Scene) -> None:
    """The edges of the boxes, each drawn along itself: firmly where the
    tones on its two sides are alike and only a line can tell them apart,
    hardly at all where light meets dark."""
    value = np.where(scene.sky > 0, scene.air, 0.05)  # (about how dark each pixel is)
    value[scene.hidden] = 0.55
    value[scene.turned] = scene.gloom[scene.turned]
    value[scene.seen == GROUND] = 0.8
    seen = scene.seen.astype(np.float64)
    border = filters.max_filter(seen, 1) != -filters.max_filter(-seen, 1)
    contrast = filters.max_filter(value, 2) + filters.max_filter(-value, 2)
    need = border * np.clip(1.0 - 2.5 * contrast, 0.15, 1.0)
    for a, b in edges(scene):
        scene.stroke(project(a), project(b), pencil(0.7, 2.2), need)


def draw_mast(scene: Scene) -> None:
    """The mast: a needle left white against the dark of the sky, with a
    line down the side turned from the sun and two collars near its foot."""
    foot, tip, half = mast()
    for share in (0.12, 0.26):
        at = foot + (tip - foot) * share
        wide = half * (1.0 - share) + 1.2
        collar = [at - (wide, -0.4), at + (wide, 0.4)]
        scene.sheet.stroke(collar, charcoal.PENCIL, 0.8, 2.0)
    side = [foot + (0.6 * half, 0.0), tip]
    scene.sheet.stroke(side, charcoal.PENCIL, [0.9, 0.5], 2.4)


# --------------------------------------------------------------------------
# the base and the ornament


def draw_opening(scene: Scene, face: Face, mine, at: float, half: float, top: float):
    """An opening of the arcade: dark, in strokes from its sill to its head."""
    span = face.point(at + half, 0.5 * top) - face.point(at - half, 0.5 * top)
    count = max(int(np.hypot(*span) / 2.6), 2)
    touch = black(0.75 if face.lit else 0.9, 3.2)
    for s in np.linspace(at - half, at + half, count):
        scene.stroke(face.point(s, 0.8), face.point(s, top), touch, mine, wobble=0.2)


def draw_sunburst(scene: Scene, face: Face, mine, low: float, reach: float) -> None:
    """A rising sun over the door in the middle of a face: rays from a point
    ``low`` above the street, and the arcs they start from and end at."""
    turns = np.radians(np.linspace(14.0, 166.0, 9))
    for turn in turns:
        way = np.array([math.cos(turn), math.sin(turn)])
        a, b = (face.point(r * way[0], low + r * way[1]) for r in (0.25 * reach, reach))
        scene.stroke(a, b, pencil(0.75, 2.0), mine, wobble=0.0)
    for radius in (0.25 * reach, 1.06 * reach):
        arc = face.point(radius * np.cos(turns), low + radius * np.sin(turns))
        scene.sheet.stroke(arc, charcoal.PENCIL, 0.7, 2.0)


def draw_base(scene: Scene) -> None:
    """The base: an arcade of tall openings between its piers, small windows
    over them and, in the middle of each face, the door under a rising sun."""
    count = 9
    for face, mine in scene.faces("tier"):
        if face.box.tier != 0:
            continue
        pitch = (face.hi - face.lo - 2.0 * CORNER) / count
        for k in range(count):
            at = face.lo + CORNER + (k + 0.5) * pitch
            if k == count // 2:
                draw_opening(scene, face, mine, at, 0.3 * pitch, 11.5)
                continue
            draw_opening(scene, face, mine, at, 0.32 * pitch, 11.0)
            for s in (at - 0.19 * pitch, at + 0.19 * pitch):
                touch = pencil(0.7, 3.4) if face.lit else black(0.7, 3.4)
                scene.stroke(face.point(s, 14.6), face.point(s, 17.6), touch, mine, 0.0)
        if face.lit:
            draw_sunburst(scene, face, mine, 12.6, 6.4)


def draw_fans(scene: Scene) -> None:
    """The bays of the crown: a fan of rays up each front, rising from behind
    the bay below."""
    below, top = TIERS[CROWN - 1][1], TIERS[CROWN][1]
    low = below + BAY_RISE * (top - below) - 3.0
    for face, mine in scene.faces("bay"):
        # (the front of a bay is at the end of its arm of the cross)
        front = 0 if face.box.lo[0] < face.box.lo[1] else 1
        if face.box.tier != CROWN or face.axis != front:
            continue
        high = face.z1 - LEDGE_DEEP - 0.4
        stick = pencil if face.lit else black
        apex = face.point(0.0, low)
        for s in np.linspace(face.lo + 1.2, face.hi - 1.2, 7):
            scene.stroke(apex, face.point(s, high), stick(0.95, 2.4), mine, 0.0)


def draw_chevrons(scene: Scene) -> None:
    """The tier over the crown's bays: a band of chevrons under its ledge,
    the teeth that hang from the ledge dark."""
    for face, mine in scene.faces("tier"):
        if face.box.tier != CROWN + 1:
            continue
        high = face.z1 - LEDGE_DEEP - 0.4
        teeth = np.linspace(face.lo + 1.0, face.hi - 1.0, 9)
        for k in range(1, 9, 2):
            tip = face.point(teeth[k], high - 7.0)
            for s in np.linspace(teeth[k - 1], teeth[k + 1], 5)[1:-1]:
                scene.stroke(tip, face.point(s, high), pencil(0.7, 2.4), mine, 0.0)
        path = face.point(teeth, high - 7.0 * (np.arange(9) % 2))
        for a, b in itertools.pairwise(path):
            scene.stroke(a, b, pencil(0.95, 2.4), mine, wobble=0.0)


def main() -> None:
    scene = Scene()
    draw_sky(scene)
    draw_shade(scene)
    draw_cast(scene)
    draw_haze(scene)
    draw_blocks(scene)
    draw_ground(scene)
    scene.sheet.fix()  # (the tones are set: the lines go over them, crisp)
    draw_windows(scene)
    draw_base(scene)
    draw_ledges(scene)
    draw_edges(scene)
    draw_fans(scene)
    draw_chevrons(scene)
    draw_mast(scene)
    out = Path(__file__).parent / "output" / "deco.png"
    write_png(str(out), np.round(scene.sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
