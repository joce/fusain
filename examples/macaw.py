"""A great green macaw, in coloured pencil on white paper.

Run from the repository root: ``python examples/macaw.py``. Writes
``examples/output/macaw.png`` (in about a minute and a quarter).

A great green macaw (Ara ambiguus) on a short length of branch, its body
turned a little toward us and leaning forward, its head turned further and
cocked, the long tail hanging down to the lower right.

The bird is built, not guessed: ``macaw_model.py`` holds a solid model of
it whose shapes were measured on reference photographs, posed and lit from
the upper left. Every pixel of the sheet knows where it lies on the bird,
how much light it takes and which way its feathers run.

It is drawn feather by feather, as a finished coloured pencil piece is:
each feather a small shaded form, laid in barb strokes from its shaft out
toward its edge and tip. What is laid first holds, so each feather is built
from the dark down: first its base and the shadow of the feather lying over
it, the crisp contact line under that feather's edge, its shaft; then the
light toward its tip and along its rim where the form is lit; then its own
saturated colour over its middle, firmly, until the tooth is closed. The
colours are a mosaic in which neighbours differ: yellow-green where the
light falls, through green and emerald to teal and cobalt in the shadows,
turquoise toward the rump, cobalt flight feathers, the tail olive at its
root, rust along its length and blue toward the tips. The head is a velvet
of tiny strokes graded by the light; the bill built in strokes along its
arch, its highlight laid first; the bare face with its rows of tiny dark
red feathers; the eye; the red brow. The branch is closed in layers of bark
and grain, a few tufts of moss on its top, broken off either side.
"""

# cspell:words ambiguus culmen headness keepdims lognormal tomium TURQ

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import macaw_model as mm
import numpy as np
from murex import write_png
from murex.filters import bilinear, box_blur, smoothstep

from fusain import Medium, Paper, Sheet, pencils, tools

WIDTH, HEIGHT = mm.WIDTH, mm.HEIGHT
SEED = 7

# ----------------------------------------------------------------------
# small tools


def soft(mask: np.ndarray, r: int) -> np.ndarray:
    """``mask`` as floats, blurred by about ``r`` pixels."""
    out = np.asarray(mask, dtype=np.float32)
    for _ in range(2):
        out = box_blur(out, r)
    return out.astype(np.float64)


def band(x, lo, hi):
    """0 below ``lo``, 1 above ``hi``, smooth between (or the other way round
    if ``lo`` > ``hi``)."""
    return smoothstep(lo, hi, x)


def trapezoid(s, edges) -> np.ndarray:
    """1 between ``b`` and ``c`` of ``edges`` (a, b, c, d), 0 outside ``a``
    and ``d``, smooth between."""
    a, b, c, d = edges
    return band(s, a, b) * (1.0 - band(s, c, d))


def stretch(tone: np.ndarray, where: np.ndarray) -> np.ndarray:
    """``tone`` spread so that over ``where`` it runs from 0 to 1."""
    a, b = np.percentile(tone[where], [2, 98])
    return np.clip((tone - a) / max(b - a, 1e-6), 0.0, 1.0)


# ----------------------------------------------------------------------
# the hand

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

    def line(self, colour: str, pts, pressure, width=2.5, wobble=0.3) -> None:
        """One stroke through ``pts`` (N, 2) of the sheet, as a hand follows
        them: ``pressure`` a number or one per point."""
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 2:
            return
        pts = pts + self.rng.normal(0.0, wobble, pts.shape)
        bear = np.broadcast_to(np.asarray(pressure, dtype=np.float64), len(pts))
        if float(bear.max()) < 0.01:
            return
        self.sheet.stroke(pts.tolist(), self.pencil(colour), bear.tolist(), width)

    def erase(self, pts, tool, pressure, width: float) -> None:
        """One pass of an eraser through ``pts`` (N, 2) of the sheet."""
        self.sheet.erase(
            np.asarray(pts, dtype=np.float64).tolist(), tool, pressure, width
        )


@dataclass
class Pencil:
    """A pencil held to follow the model's lines (its ``line`` takes the
    lines the model lays on the bird's surfaces): ``gain`` turns their
    darkness into pressure."""

    hand: Hand
    colour: str
    gain: float = 1.0
    widen: float = 1.0

    def line(self, pts, width, dark) -> None:
        """One stroke, lifted wherever the model hides the line."""
        pts = np.asarray(pts, dtype=np.float64)
        d = np.broadcast_to(np.asarray(dark, dtype=np.float64), len(pts))
        w = max(float(np.mean(width)) * self.widen, 1.8)
        for a, b in spans(d > 0.01):
            bear = np.clip(self.gain * d[a:b], 0.0, 1.0)
            self.hand.line(self.colour, pts[a:b], bear, w, 0.15)


def spans(keep: np.ndarray) -> list[tuple[int, int]]:
    """The (start, stop) of each unbroken run of ``keep``, two long or more."""
    edges = np.diff(np.concatenate([[0], keep.astype(int), [0]]))
    starts, stops = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True) if b - a > 1]


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


def swell(n: int) -> np.ndarray:
    """A sure line's pressure along it: it lands, swells and tapers."""
    return np.sin(np.linspace(0.12, math.pi - 0.12, n)) ** 0.7


# ----------------------------------------------------------------------
# the bird, as the model gives it

# ----------------------------------------------------------------------
# the bird, as the model gives it


@dataclass
class Bird:
    """The modelled bird on the sheet: where each part is seen (``part``),
    how dark it falls under the light (``shade``, 0 in the full light to 1
    in the deepest shadow), which way its feathers run (``angle``, radians
    on the sheet), the model itself and where on its wing each pixel is."""

    model: mm.Model
    wing: dict
    part: dict[str, np.ndarray]
    shade: np.ndarray
    angle: np.ndarray


def directions(model: mm.Model) -> np.ndarray:
    """Which way the feathers run at each pixel (radians on the sheet),
    smoothed a little; along the bill's arch on the bill."""
    f = model.seen.flow[..., :2].astype(np.float32)
    fx, fy = (box_blur(box_blur(f[..., k], 2), 2) for k in (0, 1))
    angle = np.arctan2(fy, fx).astype(np.float64)
    bill = model.part("upper", "lower")
    return np.where(bill, arch(model), angle)


def arch(model: mm.Model) -> np.ndarray:
    """Directions along the bill's arch (parallel to the culmen), on the
    sheet."""
    culmen = mm.smooth_curve(mm.bill_point(mm.CULMEN), 60)
    tangent = np.gradient(culmen, axis=0)
    m = model.pose.linear(True)
    t3 = np.concatenate([tangent, np.zeros((60, 1))], 1) @ m.T
    sk = np.nan_to_num(model.skull[..., :2], nan=99.0)
    best = np.zeros(sk.shape[:2], dtype=int)
    dist = np.full(sk.shape[:2], np.inf)
    for k, p in enumerate(culmen):
        d = np.hypot(sk[..., 0] - p[0], sk[..., 1] - p[1])
        best = np.where(d < dist, k, best)
        dist = np.minimum(d, dist)
    return np.arctan2(t3[best, 1], t3[best, 0])


def shading(model: mm.Model, part: dict[str, np.ndarray]) -> np.ndarray:
    """How dark everything falls, 0..1: the light on the model, deepened
    where an artist deepens it: the flank under the wing, the belly toward
    the perch, the body's shadow across the perch, the throat under the
    bill."""
    light = mm.illumination(model) / mm.EXPOSURE
    s = 1.0 - np.clip(light, 0.0, 1.0) ** 0.8
    by = np.nan_to_num(model.bird[..., 1])
    plume = part["plumage"]
    under_wing = soft(part["wing"], 14) * plume * ~part["wing"]
    s += 0.5 * under_wing + 0.3 * band(by, -2.8, -4.0) * plume
    chin = soft(part["upper"] | part["lower"], 12) * plume * ~part["face"]
    s += 0.35 * chin * band(by, -0.2, -0.8)
    body = np.isfinite(model.seen.z) & ~part["branch"]
    s += 0.55 * soft(np.roll(np.roll(body, 26, 0), 18, 1), 14) * part["branch"]
    return np.clip(s, 0.0, 1.0)


def build(rng: np.random.Generator) -> Bird:
    """The bird as it is to be drawn."""
    model = mm.build_model()
    wing = wing_maps(model, rng)
    names = ["plumage", "wing", "upper", "lower", "toes", "claws", "branch"]
    part = {name: model.part(name) for name in names}
    part["face"] = mm.face_mask(model) > 0.5
    part["brow"] = (mm.brow_mask(model) > 0.5) & ~part["face"]
    part["feathers"] = part["plumage"] & ~part["face"] & ~part["brow"]
    for k in range(len(mm.BLADES)):
        part[f"tail{k}"] = model.part(f"tail{k}")
    part["tail"] = np.logical_or.reduce([part[f"tail{k}"] for k in range(6)])
    shade = shading(model, part)
    feathered = part["feathers"] | part["wing"] | part["tail"]
    shade = np.where(feathered, stretch(shade, feathered), shade)
    return Bird(model, wing, part, beyond(shade, np.isfinite(model.seen.z)),
                directions(model))  # fmt: skip


def beyond(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """``values`` carried a little way out past where they are ``known``."""
    num = soft(np.where(known, values, 0.0), 4)
    den = soft(known, 4)
    return np.where(known, values, num / np.maximum(den, 1e-6))


# ----------------------------------------------------------------------
# the box of pencils


LEMON, CHARTREUSE, OLIVE, MOSS = "#dcd66a", "#acb544", "#7a8c30", "#55702a"
LEAF, PINE, DEEP, NIGHT, INDIGO = "#3c5a2c", "#2a4634", "#1e3836", "#17262a", "#1d2148"
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


# ----------------------------------------------------------------------
# the feathers, each a small shaded form

#: The kinds of feather, by tract.
HEAD, BODY, COVERT, GREATER, FLIGHT, TAIL = range(6)

#: Each feather's colours: the dark of its base and of the shadows on it,
#: its own saturated colour, and the light toward its tip.
PALETTES = {
    "lime": ("#14302a", "#5fb02e", "#f2e94a"),
    "green": ("#0b2a2a", "#22963e", "#e4e23e"),
    "emerald": ("#08261c", "#108a5a", "#bfe04a"),
    "teal": ("#0b2230", "#0f8a86", "#9fe36a"),
    "turquoise": ("#0b2236", "#14a0b0", "#8fe6e0"),
    "cobalt": ("#0a1838", "#1f5fb5", "#5fd3e6"),
    "rust": ("#2e100c", "#a8442a", "#e0a03c"),
    "olive": ("#1c2410", "#7a8a2e", "#e0d860"),
    "sky": ("#14304a", "#4a9ad0", "#c8ecf6"),
    "deep": ("#06180f", "#0f6a38", "#6fb84a"),
    "pine": ("#061a10", "#176236", "#62ae58"),
}
NAMES = list(PALETTES)
#: The greens of the body, from the light side round to the shadow side.
GREEN_RUN = ["lime", "green", "emerald", "deep", "pine"]


@dataclass
class Plumage:
    """Every feather seen, on the sheet: which one owns each pixel
    (``owner``, -1 for none); how far across its visible part from where
    the feather over it covers it to its own free edge (``a``, 0..1); how
    far from its shaft to either side (``c``, -1..1); which way its tip
    and its side lie (``along``, ``across``, unit vectors (H, W, 2)); its
    kind, its colours (``palette``, an index into NAMES), and whether its
    barbs go to one side only (``vane``: the flight feathers)."""

    owner: np.ndarray
    a: np.ndarray
    c: np.ndarray
    along: np.ndarray
    across: np.ndarray
    kind: np.ndarray
    palette: np.ndarray
    vane: np.ndarray
    #: A second colour some feathers turn to along their length, and how far.
    palette2: np.ndarray
    mix: np.ndarray


def empty_plumage() -> Plumage:
    """A plumage with no feathers yet."""
    shape = (HEIGHT, WIDTH)
    return Plumage(
        np.full(shape, -1, np.int32),
        np.zeros(shape),
        np.zeros(shape),
        np.zeros((*shape, 2)),
        np.zeros((*shape, 2)),
        np.full(shape, -1, np.int8),
        np.zeros(shape, np.int16),
        np.zeros(shape, bool),
        np.zeros(shape, np.int16),
        np.zeros(shape),
    )


def on_sheet(model: mm.Model, vector) -> np.ndarray:
    """A direction of the bird's frame (3,) as a unit vector on the sheet."""
    v = model.pose.linear(False) @ np.asarray(vector, dtype=np.float64)
    return v[:2] / max(float(np.linalg.norm(v[:2])), 1e-9)


# ---- the small feathers of the head and body, laid as tiles


@dataclass
class Tiling:
    """Small feathers laid as overlapping tiles: which one is seen at each
    pixel, how far toward its tip (``along``, -1 at its base) and across it,
    and where its free edge is there (``tip``)."""

    which: np.ndarray
    along: np.ndarray
    across: np.ndarray
    tip: np.ndarray
    #: Which way each tile's own tip lies (H, W, 2).
    way: np.ndarray


def tile_one(t: Tiling, best, seed, frame, where, rng) -> None:
    """Lay one small feather: ``seed`` (x, y, k), ``frame`` (direction,
    length, width); it shows where it is the nearest its own tip."""
    x, y, k = seed
    d, length, width = frame
    h, w = where.shape
    r = int(length + width) + 2
    ix, iy = int(x), int(y)
    y0, y1, x0, x1 = (
        max(iy - r, 0),
        min(iy + r + 1, h),
        max(ix - r, 0),
        min(ix + r + 1, w),
    )
    gy, gx = np.mgrid[y0:y1, x0:x1] + 0.5
    sa = ((gx - x) * d[0] + (gy - y) * d[1]) / length
    ta = ((gx - x) * -d[1] + (gy - y) * d[0]) / (0.5 * width)
    fray = 0.07 * np.sin(ta * rng.uniform(7, 12) + rng.uniform(0, 6.3))
    tip = 0.5 * np.sqrt(np.clip(1 - ta**2, 0, 1)) + fray
    cover = (sa >= -1.0) & (sa <= tip) & (np.abs(ta) <= 1) & where[y0:y1, x0:x1]
    take = cover & (sa > best[y0:y1, x0:x1])
    sl = (slice(y0, y1), slice(x0, x1))
    best[sl] = np.where(take, sa, best[sl])
    t.which[sl] = np.where(take, k, t.which[sl])
    t.along[sl] = np.where(take, sa, t.along[sl])
    t.across[sl] = np.where(take, ta, t.across[sl])
    t.tip[sl] = np.where(take, tip, t.tip[sl])
    t.way[sl] = np.where(take[..., None], d, t.way[sl])


def tiling(model: mm.Model, where: np.ndarray, reach: np.ndarray, size, rng) -> Tiling:
    """Small feathers over ``where``, each ``size`` pixels long, as the way
    the feathers run lays them, their tips a little frayed; free to reach
    over the edge of ``where`` into ``reach``, so that at the bird's edge
    the tips make the outline."""
    direction, short = mm.flow2d(model)
    seeds = mm.scatter(where, np.maximum(0.6 * size * np.sqrt(short), 2.5), rng)
    shape = where.shape
    t = Tiling(np.full(shape, -1), np.zeros(shape), np.zeros(shape), np.ones(shape),
               np.zeros((*shape, 2)))  # fmt: skip
    best = np.full(shape, -9.0)
    for k, (x, y) in enumerate(seeds):
        ix, iy = int(x), int(y)
        length = size[iy, ix] * short[iy, ix] * rng.uniform(0.75, 1.3)
        width = size[iy, ix] * rng.uniform(0.8, 1.1)
        tile_one(t, best, (x, y, k), (direction[iy, ix], length, width), reach, rng)
    return t


def depth_of_tiles(t: Tiling) -> np.ndarray:
    """How far across each tile's visible part (0 where the tile over it
    covers it, 1 at its own free edge)."""
    seen = t.which >= 0
    n = int(t.which.max()) + 1 if seen.any() else 0
    lo = np.full(n, np.inf)
    np.minimum.at(lo, t.which[seen], t.along[seen])
    base = np.where(seen, lo[np.maximum(t.which, 0)], 0.0)
    return np.where(
        seen, np.clip((t.along - base) / np.maximum(t.tip - base, 1e-3), 0, 1), 0
    )


def feather_size(model: mm.Model) -> np.ndarray:
    """How long the small feathers are (pixels): tiny on the head, small on
    the neck and mantle, larger and looser down the breast and belly, small
    again on the thighs."""
    b = np.nan_to_num(model.bird)
    down = smoothstep(-0.8, -3.0, b[..., 1])
    body = 0.2 + 0.11 * down
    thigh = smoothstep(0.8, 0.45, mm.implicit_map(model, mm.THIGHS))
    body = body * (1.0 - 0.35 * thigh)
    head = smoothstep(0.3, 0.8, model.headness)
    return (head * 0.1 + (1.0 - head) * body) * mm.UNIT


def outward(bird: Bird, where: np.ndarray) -> np.ndarray:
    """``where``, with a margin over the paper beside it, into which its
    feathers' tips may reach."""
    drawn = np.isfinite(bird.model.seen.z)
    return where | ((soft(where, 5) > 0.04) & ~drawn)


def body_tract(pl: Plumage, bird: Bird, rng) -> None:
    """The small feathers of the neck, breast, belly and thighs, and of the
    back of the head; the front of the head a velvet."""
    model = bird.model
    where = bird.part["feathers"]
    t = tiling(model, where, outward(bird, where), feather_size(model), rng)
    seen = t.which >= 0
    pl.owner[seen] = t.which[seen]
    pl.a[seen] = depth_of_tiles(t)[seen]
    pl.c[seen] = t.across[seen]
    pl.along[seen] = t.way[seen]
    pl.across[seen] = np.stack([-t.way[..., 1], t.way[..., 0]], -1)[seen]
    pl.kind[seen] = BODY
    # (the front of the head: a velvet of tiny feathers that barely
    # separate, modelled only by the light on its form; at the nape and the
    # cheek's edge they resolve into rows of small stated feathers)
    front = np.nan_to_num(model.skull[..., 0], nan=-9.0) > -0.3
    direction, _ = mm.flow2d(model)
    head = where & (((model.headness > 0.6) & front) | (pl.owner < 0))
    pl.owner[head] = 400000
    pl.a[head] = 0.62
    pl.c[head] = np.where(pl.c[head] < 0, -0.5, 0.5)
    pl.along[head] = direction[head]
    pl.across[head] = np.stack([-direction[..., 1], direction[..., 0]], -1)[head]
    pl.kind[head] = HEAD


# ---- the wing and the tail


def wing_maps(model: mm.Model, rng) -> dict:
    """Where on the folded wing each pixel lies, and which covert or flight
    feather it is on (without the model's relief)."""
    b = np.nan_to_num(model.bird)
    u, v = mm.wing_uv(b[..., 0], b[..., 1])
    rows = mm.covert_rows(rng)
    row, q, side = mm.covert_relief(u, v, rows)
    blade, qf = mm.flight_relief(u, v)
    return {"u": u, "v": v, "rows": rows, "row": row, "q": q, "side": side,
            "blade": blade, "qf": qf}  # fmt: skip


def covert_index(wing: dict, covert: np.ndarray) -> np.ndarray:
    """Which feather of its row each covert pixel is on."""
    out = np.zeros(covert.shape, int)
    for k, r in enumerate(wing["rows"]):
        here = covert & (wing["row"] == k)
        e = r["edges"]
        out[here] = np.clip(np.searchsorted(e, wing["v"][here]) - 1, 0, len(e) - 2)
    return out


def wing_tract(pl: Plumage, bird: Bird) -> None:
    """The coverts in rows growing down the wing, the greater coverts, and
    the long flight feathers lying one over the next."""
    model, wing = bird.model, bird.wing
    seen = bird.part["wing"]
    covert = seen & (wing["row"] >= 0)
    flight = seen & (wing["row"] < 0) & (wing["blade"] >= 0)
    last = len(wing["rows"]) - 1
    j = covert_index(wing, covert)
    pl.owner[covert] = 100000 + wing["row"][covert] * 500 + j[covert]
    pl.owner[flight] = 200000 + wing["blade"][flight]
    pl.a[covert], pl.a[flight] = wing["q"][covert], wing["qf"][flight]
    pl.c[covert] = 2.0 * wing["side"][covert] - 1.0
    pl.c[flight] = 1.0
    _, along, across, _ = mm.wing_frame()
    pl.along[seen] = on_sheet(model, [along[0], along[1], 0.0])
    pl.across[seen] = on_sheet(model, [across[0], across[1], 0.0])
    pl.kind[covert] = np.where(wing["row"][covert] == last, GREATER, COVERT)
    pl.kind[flight] = FLIGHT
    pl.vane[flight] = True


def tail_tract(pl: Plumage, bird: Bird) -> None:
    """The long tail feathers: each a blade, its shaft down its middle, its
    side toward the light lit, dark where the blade over it lies."""
    model = bird.model
    above = bird.part["plumage"] | bird.part["wing"]
    for k in range(len(mm.BLADES)):
        seen = bird.part[f"tail{k}"]
        _, dk, side, _ = mm.blade_basis(k)
        _, across = mm.blade_coords(model, k)
        covered = np.clip(2.5 * soft(above, 4), 0.0, 1.0)
        a = np.minimum(0.5 + 0.5 * np.clip(-across, -1, 1), 1.0 - covered)
        pl.owner[seen] = 300000 + k
        pl.a[seen] = a[seen]
        pl.c[seen] = np.clip(across[seen], -1.0, 1.0)
        pl.along[seen] = on_sheet(model, dk)
        pl.across[seen] = on_sheet(model, side)
        pl.kind[seen] = TAIL
        above = above | seen


# ---- the colours of each feather


def per_feather(pl: Plumage, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean of ``values`` over each feather, as a map (and the compact
    feather index of each pixel)."""
    seen = pl.owner >= 0
    ids, inv = np.unique(pl.owner[seen], return_inverse=True)
    mean = np.bincount(inv, values[seen]) / np.maximum(np.bincount(inv), 1)
    out = np.zeros(pl.owner.shape)
    out[seen] = mean[inv]
    index = np.full(pl.owner.shape, -1)
    index[seen] = inv
    del ids
    return out, index


def assign_colours(pl: Plumage, bird: Bird, rng) -> None:
    """Each feather's colour: yellow-green in the light, through emerald and
    teal to cobalt in the shadow, as a mosaic in which neighbours differ;
    turquoise toward the rump, cobalt on the flight feathers and the wing's
    tip, rust and blue on the tail."""
    s, index = per_feather(pl, bird.shade)
    n = int(index.max()) + 1
    jitter = rng.normal(0.0, 0.11, n)[np.maximum(index, 0)]
    b = np.nan_to_num(bird.model.bird)
    lower_wing = (pl.kind == GREATER) * 0.15 + (pl.kind == COVERT) * 0.04
    score = np.clip(0.1 + 0.65 * s + jitter + lower_wing, 0.0, 0.999)
    pick = np.array([NAMES.index(g) for g in GREEN_RUN])
    pl.palette[:] = pick[(score * len(pick)).astype(int).clip(0, len(pick) - 1)]
    accent = rng.random(n)[np.maximum(index, 0)] < 0.04
    shadowed = score > 0.45
    pl.palette[accent & shadowed] = NAMES.index("teal")
    rump = band(b[..., 0], -3.0, -3.5) * band(b[..., 1], -4.0, -4.6)
    turq = (pl.kind == BODY) & (rump > 0.4)
    pl.palette[turq] = NAMES.index("turquoise")
    head_colours(pl, bird)
    flight_colours(pl, bird)
    tail_colours(pl, bird)


def head_colours(pl: Plumage, bird: Bird) -> None:
    """The head's velvet, graded by the light: yellow-green on the lit
    crown, through green and emerald to teal in its shadows."""
    head = pl.kind == HEAD
    score = np.clip(0.05 + 0.85 * bird.shade, 0.0, 0.999) * (len(GREEN_RUN) - 2)
    low = np.floor(score).astype(int)
    pick = np.array([NAMES.index(g) for g in GREEN_RUN])
    pl.palette[head] = pick[low][head]
    pl.palette2[head] = pick[low + 1][head]
    pl.mix[head] = (score - low)[head]


def flight_colours(pl: Plumage, bird: Bird) -> None:
    """Teal secondaries going to cobalt, cobalt primaries, the wing's tip
    blue."""
    blade, u = bird.wing["blade"], bird.wing["u"]
    flight = pl.kind == FLIGHT
    pl.palette[flight & (blade < 2)] = NAMES.index("emerald")
    pl.palette[flight & (blade >= 2) & (blade < 4)] = NAMES.index("deep")
    pl.palette[flight & (blade >= 4)] = NAMES.index("cobalt")
    wing = (pl.kind >= COVERT) & (pl.kind <= FLIGHT)
    pl.palette2[wing] = NAMES.index("cobalt")
    pl.mix[wing] = band(u, 0.66, 0.82)[wing]


def tail_colours(pl: Plumage, bird: Bird) -> None:
    """The tail: olive near the body, rust, blue toward the tips and along
    the outer feathers, pale blue at the very tips."""
    for k in range(len(mm.BLADES)):
        seen = bird.part[f"tail{k}"]
        t, across = mm.blade_coords(bird.model, k)
        start = 0.55 - 0.07 * (k // 2)
        base = t < 0.2
        blue = band(t, start, start + 0.25)
        blue = np.maximum(
            blue, (0.85 if k >= 2 else 0.3) * band(np.abs(across), 0.45, 0.95)
        )
        tip = np.where(t > 0.9, NAMES.index("sky"), NAMES.index("cobalt"))
        rust, olive = NAMES.index("rust"), NAMES.index("olive")
        pl.palette[seen] = np.where(base, olive, rust)[seen]
        pl.palette2[seen] = np.where(base, rust, tip)[seen]
        mix = np.where(base, band(t, 0.02, 0.2), np.where(t > 0.9, 1.0, blue))
        pl.mix[seen] = mix[seen]


# ---- laying the feathers in pencil


def boundaries(pl: Plumage) -> np.ndarray:
    """Where one feather meets another or the paper."""
    o = pl.owner
    edge = np.zeros(o.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        edge |= np.roll(np.roll(o, dy, 0), dx, 1) != o
    return edge & (o >= 0)


@dataclass
class Zones:
    """How each pixel of a feather is to be laid, 0..1: its dark (the base
    and the shadow cast by the feather over it, and the large form's
    shadow), its light toward the tip, its own colour, the pale rim at its
    free edge, the crisp contact line under the free edge of the feather
    over it, and its shaft."""

    dark: np.ndarray
    light: np.ndarray
    mid: np.ndarray
    rim: np.ndarray
    contact: np.ndarray
    shaft: np.ndarray


def zones(pl: Plumage, shade: np.ndarray) -> Zones:
    """The zones of every feather, from where on it each pixel lies and how
    the large forms are lit."""
    a, kind = pl.a, pl.kind
    soft_kind = np.where(kind == HEAD, 0.0, np.where(kind == BODY, 0.75, 1.0))
    deep = (1.0 - smoothstep(0.0, 0.36, a)) ** 1.2 * soft_kind
    shadow = band(shade, 0.45, 0.95)
    edge = soft(boundaries(pl), 1)
    lit = 1.0 - band(shade, 0.25, 0.75)
    rim = np.clip(2.5 * edge, 0, 1) * smoothstep(0.6, 0.85, a) * (0.35 + 0.65 * lit)
    dark = np.clip(
        0.95 * deep + 0.75 * shadow * (1.0 - 0.85 * smoothstep(0.55, 0.95, a)), 0, 1
    )
    # (in the shadow each feather still catches a little reflected light at
    # its tip and rim, which keeps it apart from the next)
    reflected = 0.45 * smoothstep(0.7, 0.97, a) * (1.0 - lit) * (kind <= FLIGHT)
    light = 0.85 * smoothstep(0.3, 0.8, a) * lit * (1.0 - 0.4 * rim) + reflected
    mid = (1.0 - 0.75 * smoothstep(0.55, 0.97, a) * lit) * (1.0 - 0.4 * rim)
    contact = np.clip(3.0 * edge, 0, 1) * (a < 0.12) * (kind >= COVERT)
    stiff = (kind == COVERT) | (kind == GREATER) | (kind == TAIL)
    shaft = np.exp(-((pl.c / 0.07) ** 2)) * trapezoid(a, (0.1, 0.3, 0.85, 0.97)) * stiff
    shaft = np.where(kind == FLIGHT, trapezoid(a, (0.03, 0.06, 0.1, 0.13)), shaft)
    seen = pl.owner >= 0
    return Zones(
        *(np.where(seen, m, 0.0) for m in (dark, light, mid, rim, contact, shaft))
    )


@dataclass
class Barbs:
    """How a layer of barbs is laid: their length (a share of the barb
    length of each kind of feather) and width (pixels), how many times over
    they cover, and how far they lean toward the tip (``slant``; None to run
    along the shaft)."""

    length: float = 1.0
    width: float = 2.2
    cover: float = 2.5
    slant: float | None = 0.8


#: How long a barb is on each kind of feather (pixels): head, body, covert,
#: greater covert, flight feather, tail; and how far it leans toward the
#: feather's tip (the head's run nearly with the lie of the feathers, a
#: velvet; the coverts' fan out from the shaft).
BARB = np.array([7.0, 15.0, 17.0, 22.0, 20.0, 18.0])
SLANT = np.array([4.0, 2.0, 0.8, 0.8, 0.9, 1.2])


def grid_seeds(rng, weight: np.ndarray, spacing: float) -> np.ndarray:
    """Points over ``weight`` > 0 on a jittered grid ``spacing`` apart, kept
    as often as the weight asks: an even cover, without clumps or holes."""
    ys, xs = np.mgrid[0:HEIGHT:spacing, 0:WIDTH:spacing]
    pts = np.stack([xs.ravel(), ys.ravel()], 1) + rng.uniform(0, spacing, (xs.size, 2))
    ix = np.clip(pts[:, 0].astype(int), 0, WIDTH - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, HEIGHT - 1)
    keep = rng.random(len(pts)) < np.clip(weight[iy, ix] * 1.6, 0.0, 1.0)
    return pts[keep]


def trace_barbs(pl: Plumage, seed: np.ndarray, how: Barbs, rng) -> np.ndarray:
    """Paths (N, steps + 1, 2) from ``seed`` out from the shaft toward the
    feather's edge and its tip."""
    n = len(seed)
    steps = max(int(how.length * BARB.max() / 1.6), 2)
    step = how.length * BARB.max() / steps
    sx = np.clip(seed[:, 0].astype(int), 0, WIDTH - 1)
    sy = np.clip(seed[:, 1].astype(int), 0, HEIGHT - 1)
    sign = np.where(pl.c[sy, sx] < 0, -1.0, 1.0)
    loose = pl.kind[sy, sx] == BODY
    sign = np.where(loose, rng.choice([-1.0, 1.0], n), sign)
    sign = np.where(pl.vane[sy, sx], 1.0, sign)
    lean = SLANT[np.clip(pl.kind[sy, sx], 0, len(SLANT) - 1)] * (how.slant or 0.0) / 0.8
    curl = rng.normal(0.0, 0.08, n)
    path = np.empty((n, steps + 1, 2))
    path[:, 0] = seed
    for k in range(steps):
        ix = np.clip(path[:, k, 0].astype(int), 0, WIDTH - 1)
        iy = np.clip(path[:, k, 1].astype(int), 0, HEIGHT - 1)
        if how.slant is None:
            d = pl.along[iy, ix]
        else:
            d = (
                lean[:, None] * pl.along[iy, ix]
                + (sign + curl * k)[:, None] * pl.across[iy, ix]
            )
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        path[:, k + 1] = path[:, k] + step * d
    return path


def barb_layer(hand: Hand, colour: str, weight: np.ndarray, pl: Plumage, firm: float,
               how: Barbs) -> None:  # fmt: skip
    """One layer of barb strokes of one pencil over ``weight`` > 0, each
    kept on the feather it starts on, as firm as ``weight`` asks along it."""
    dense = weight if hand.finish is None else weight * hand.finish
    spacing = math.sqrt(how.width * 10.0 / how.cover)
    seed = grid_seeds(hand.rng, dense, spacing)
    if not len(seed):
        return
    paths = trace_barbs(pl, seed, how, hand.rng)
    ix = np.clip(paths[..., 0].astype(int), 0, WIDTH - 1)
    iy = np.clip(paths[..., 1].astype(int), 0, HEIGHT - 1)
    own = pl.owner[iy, ix] == pl.owner[iy[:, :1], ix[:, :1]]
    kind = np.clip(pl.kind[iy[:, 0], ix[:, 0]], 0, len(BARB) - 1)
    reach = how.length * BARB[kind] * hand.rng.uniform(0.7, 1.2, len(paths))
    reach = (reach / (how.length * BARB.max() / (paths.shape[1] - 1))).astype(int) + 1
    keep = np.minimum(np.cumprod(own, axis=1).sum(axis=1) + 2, reach)
    keep = np.minimum(keep, paths.shape[1])
    bear = firm * np.clip(weight[iy, ix], 0.0, 1.0)
    taper = np.minimum(np.linspace(0.6, 1.6, paths.shape[1]), 1.0)
    medium = hand.pencil(colour)
    jitter = hand.rng.uniform(0.85, 1.1, len(paths))
    for path, n, b, j in zip(paths, keep, bear, jitter, strict=True):
        if n >= 2 and b[:n].max() > 0.03:
            hand.sheet.stroke(
                path[:n].tolist(), medium, (b[:n] * taper[:n] * j).tolist(), how.width
            )


def each_palette(pl: Plumage, weight: np.ndarray, which: int):
    """For each palette in use: its pencil (0 dark, 1 mid, 2 light) and the
    weight over its feathers (shared where a feather turns from one colour
    to another)."""
    for k, name in enumerate(NAMES):
        here = weight * (
            (pl.palette == k) * (1.0 - pl.mix) + (pl.palette2 == k) * pl.mix
        )
        if here.max() > 0.02:
            yield PALETTES[name][which], here


def lay_feathers(hand: Hand, pl: Plumage, zone: Zones) -> None:
    """Every feather in pencil, as such a piece is built: first the darks
    (its base and the shadows on it, the contact line under the feather
    over it, its shaft), then its light toward the tip, then its own colour
    over its middle, firmly, until the tooth is closed."""
    for colour, w in each_palette(pl, zone.dark, 0):
        barb_layer(hand, colour, w, pl, 0.95, Barbs(0.8, 2.2, 3.0))
    barb_layer(hand, BLACK, zone.contact, pl, 0.95, Barbs(0.4, 1.8, 2.0, 0.0))
    barb_layer(hand, NIGHT, zone.shaft, pl, 0.75, Barbs(0.7, 1.5, 1.2, None))
    light = np.clip(zone.light + 0.7 * zone.rim, 0.0, 1.0)
    for colour, w in each_palette(pl, light, 2):
        barb_layer(hand, colour, w, pl, 0.85, Barbs(0.8, 2.2, 3.0))
    for firm in (0.85, 0.75, 0.6):
        for colour, w in each_palette(pl, zone.mid, 1):
            barb_layer(hand, colour, w, pl, firm, Barbs(1.0, 2.4, 2.5))


def build_plumage(bird: Bird, rng) -> Plumage:
    """All the feathers seen, and their colours."""
    pl = empty_plumage()
    body_tract(pl, bird, rng)
    wing_tract(pl, bird)
    tail_tract(pl, bird)
    assign_colours(pl, bird, rng)
    return pl


# ----------------------------------------------------------------------
# the bill


#: The pencils that build the horn, darks first: colour, where on the scale
#: of darkness it goes on, how firmly, how wide.
HORN_LAYERS = [
    (BLACK, (0.74, 0.92, 2.0, 2.1), 0.8, 3.6),
    (CHAR, (0.5, 0.72, 2.0, 2.1), 0.72, 3.6),
    (SLATE, (0.28, 0.5, 0.82, 0.92), 0.72, 3.4),
    (GREY, (0.0, 0.2, 0.55, 0.7), 0.7, 3.4),
]


def pieces(rng, n: int, short: int, long: int) -> list[tuple[int, int]]:
    """A line of ``n`` points broken into overlapping strokes of ``short``
    to ``long`` points, as a hand lays a long line."""
    out, a = [], int(rng.integers(0, max(short // 2, 1)))
    while a < n - 2:
        b = min(a + int(rng.integers(short, long + 1)), n)
        out.append((a, b))
        a = b - int(rng.integers(1, 3))
    return out


def arch_family(chain_a, chain_b, count: int, base=None, n: int = 60):
    """Curves (count, n, 2) of the skull frame between two chains of the
    bill's own frame that meet at the point: they follow its arch. If a
    ``base`` chain is given, the curves start along it (where the horn
    grows from the face), not on the straight line between the chains."""
    a = mm.smooth_curve(mm.bill_point(chain_a), n)
    b = mm.smooth_curve(mm.bill_point(chain_b), n)
    f = (np.arange(count) + 0.5) / count
    curves = a[None] * (1 - f[:, None, None]) + b[None] * f[:, None, None]
    if base is not None:
        root = mm.smooth_curve(mm.bill_point(base), count)
        miss = root - curves[:, 0]
        t = np.linspace(0.0, 1.0, n)
        curves = curves + miss[:, None, :] * ((1.0 - t) ** 2)[None, :, None]
    return f, curves


@dataclass
class Horn:
    """One mandible to be laid in strokes along its arch: its curves (in the
    skull frame), where across it each lies (0..1), how dark the horn is at
    each point and how much its highlight takes from that (``dark``, a
    function of across, along and the shade)."""

    which: int
    part: str
    f: np.ndarray
    curves: np.ndarray
    dark: Callable[..., tuple[np.ndarray, np.ndarray]]


#: The pencils that build the horn, darks first: colour, where on the scale
#: of darkness it goes on, how firmly, how wide.


def upper_dark(f, t, s):
    """The upper mandible: blackish grey, paler along the ridge and toward
    the hook, darker toward the cutting edge; and its one highlight band,
    following the arch a little below the ridge."""
    d = 0.42 + 0.38 * s + 0.18 * band(f, 0.6, 0.95) - 0.25 * band(t, 0.62, 0.95)
    d = d - 0.15 * band(f, 0.2, 0.0)
    light = np.exp(-(((f - 0.33) / 0.1) ** 2)) * trapezoid(t, (0.05, 0.25, 0.72, 0.9))
    return np.clip(d, 0.0, 1.0), light


def lower_dark(f, t, s):
    """The lower mandible: darker, with its own small light on its rounded
    side."""
    d = 0.66 + 0.34 * s
    light = (
        0.8
        * np.exp(-(((f - 0.42) / 0.12) ** 2))
        * trapezoid(t, (0.35, 0.55, 0.8, 0.95))
    )
    return np.clip(d, 0.0, 1.0), light


def horn_lines(bird: Bird, horn: Horn):
    """Each of a mandible's curves on the sheet, which of its points are
    seen, and where along it each lies."""
    for f, curve in zip(horn.f, horn.curves, strict=True):
        pts = mm.on_bill(bird.model, horn.which, curve)
        seen = mm.visible(bird.model, pts, 1.5, (horn.part,))
        yield f, pts, seen, np.linspace(0.0, 1.0, len(curve))


def lay_horn(hand: Hand, bird: Bird, horn: Horn) -> None:
    """Lay one mandible, every stroke along its arch: its highlight first in
    a pale grey (what is laid first holds), then the darks, darkest first,
    each kept off the highlight."""
    rng = hand.rng
    step = 0.5 / len(horn.f)
    for f, pts, seen, t in horn_lines(bird, horn):
        _, light = horn.dark(f, t, sample(bird.shade, pts))
        bear = 0.75 * light * seen
        if bear.max() > 0.05:
            hand.line(SILVER, pts[:, :2], bear, 3.0, 0.05)
    for colour, edges, firm, width in HORN_LAYERS:
        for f, pts, seen, t in horn_lines(bird, horn):
            d, light = horn.dark(f + rng.normal(0.0, step), t, sample(bird.shade, pts))
            bear = firm * trapezoid(d, edges) * (1.0 - 0.8 * light) * seen
            for a, b in pieces(rng, len(pts), 9, 18):
                if bear[a:b].max() > 0.03:
                    w = bear[a:b] * swell(b - a) ** 0.3
                    hand.line(colour, pts[a:b, :2], w, width, 0.08)


def sample(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """The values of a map (H, W) under points (N, 2+) of the sheet."""
    ix = np.clip(pts[:, 0].astype(int), 0, WIDTH - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, HEIGHT - 1)
    return field[iy, ix]


def draw_bill(hand: Hand, bird: Bird) -> None:
    """The great hooked bill as one massive form of horn: every stroke along
    its arch, blackish grey, paler and a little warm along the ridge and
    toward the hook, one soft highlight following the arch; a crisp cutting
    edge with a thin light under it, the hooked point clearly over the
    lower mandible; the lower mandible darker, with its own small light."""
    upper_f, upper_c = arch_family(mm.CULMEN, mm.TOMIUM, 96, mm.BILL_BASE)
    lower_f, lower_c = arch_family(mm.LOWER[:7], mm.LOWER[12:5:-1], 64)
    lower = Horn(1, "lower", lower_f, lower_c, lower_dark)
    upper = Horn(0, "upper", upper_f, upper_c, upper_dark)
    lay_horn(hand, bird, lower)
    lay_horn(hand, bird, upper)
    burnish(hand, bird, lower, CHAR)
    burnish(hand, bird, upper, SLATE)
    warm_tip(hand, bird, upper_c)
    bill_edges(hand, bird)


def burnish(hand: Hand, bird: Bird, horn: Horn, colour: str) -> None:
    """Press one grey firmly over all of a mandible but its highlight, along
    the arch, so that the horn closes smooth and glossy."""
    for f, pts, seen, t in horn_lines(bird, horn):
        _, light = horn.dark(f, t, sample(bird.shade, pts))
        bear = 0.62 * (1.0 - band(light, 0.15, 0.5)) * seen
        if bear.max() > 0.05:
            hand.line(colour, pts[:, :2], bear, 3.4, 0.04)


def warm_tip(hand: Hand, bird: Bird, curves: np.ndarray) -> None:
    """A warm horn grey along the ridge and toward the hook's point."""
    t = np.linspace(0.0, 1.0, curves.shape[1])
    for k, curve in enumerate(curves[: len(curves) // 2]):
        pts = mm.on_bill(bird.model, 0, curve)
        seen = mm.visible(bird.model, pts, 5.0, ("upper",))
        bear = 0.35 * band(t, 0.55, 0.9) * (1.0 - k / (len(curves) // 2)) * seen
        if bear.max() > 0.03:
            hand.line(HORN, pts[:, :2], bear, 3.0, 0.1)


def bill_edges(hand: Hand, bird: Bird) -> None:
    """The bill's sure lines: the cutting edge, crisp and dark, with a thin
    light along it on the lower mandible; the hooked point's outline over
    the lower mandible; the gape; the base against the face."""
    model = bird.model
    tomium = mm.smooth_curve(mm.bill_point(mm.TOMIUM), 50)
    edge = mm.on_bill(model, 0, tomium + mm.grain_offset(tomium, 0.008))
    bear = 0.95 * np.linspace(0.7, 1.0, 50)
    hand.line(BLACK, edge[:, :2], bear, 3.0, 0.08)
    hook = np.concatenate(
        [mm.bill_point(mm.CULMEN[-4:]), mm.bill_point(mm.TOMIUM[::-1][1:4])]
    )
    pts = mm.on_bill(model, 0, mm.smooth_curve(hook, 24))
    hand.line(BLACK, pts[:, :2], (0.9 * swell(24)).tolist(), 2.6, 0.06)
    gape = mm.on_bill(
        model, 1, mm.smooth_curve(mm.bill_point(mm.LOWER[-2:] + mm.LOWER[:1]), 16), 0.8
    )
    hand.line(BLACK, gape[:, :2], (0.9 * swell(16)).tolist(), 3.2, 0.1)
    culmen = mm.smooth_curve(mm.bill_point(mm.CULMEN), 60)
    ridge = mm.on_bill(model, 0, culmen + mm.grain_offset(culmen, -0.006))
    seen = mm.visible(model, ridge, 5.0, ("upper",))
    hand.line(CHAR, ridge[:, :2], 0.8 * swell(60) * seen, 2.6, 0.03)
    base = mm.smooth_curve(mm.bill_point(mm.BILL_BASE), 40)
    pts = mm.on_bill(model, 0, base + mm.grain_offset(base, 0.012))
    seen = mm.visible(model, pts, 5.0, ("upper",))
    hand.line(CHAR, pts[:, :2], 0.7 * swell(40) * seen, 2.6, 0.1)


#: The pencils of the tail, darks first: colour, which of its colours it
#: lays (olive near the body, rust, blue toward the tip, pale blue at the
#: very tip), where on the scale of darkness, how firmly.

# ----------------------------------------------------------------------
# the head, the feet and the branch


def draw_face(hand: Hand, bird: Bird) -> None:
    """The bare face: off-white, modelled as a rounded cheek, cooler where
    it turns away, flushed pink toward the bill, with its rows of tiny dark
    red feathers; the eye's place left clean."""
    face = bird.part["face"]
    sk = np.nan_to_num(bird.model.skull)
    clear = np.hypot(sk[..., 0], sk[..., 1]) > mm.IRIS + 0.03
    w = (face & clear) * 1.0
    turn = band(bird.shade, 0.2, 0.75)
    flush = band(sk[..., 0], 0.0, 0.5) * (0.5 + 0.5 * band(-sk[..., 1], 0.1, 0.6))
    mm.draw_face_lines(bird.model, Pencil(hand, LINE, 1.0), hand.rng)
    back = band(-sk[..., 0], 0.0, 0.45) * band(-sk[..., 1], 0.0, 0.8)
    hand.tone(COOL, w * np.clip(turn + 0.6 * back, 0, 1), 0.36, (25,), 2.6)
    hand.tone(MAUVE, w * np.clip(turn**2 + 0.4 * back, 0, 1), 0.3, (70,), 2.6)
    hand.tone(ROSE, w * (0.3 + 0.7 * flush), 0.3, (-20,), 2.6)
    hand.tone(BLUSH, w, 0.28, (10, 80), 2.6)


def draw_fringe(hand: Hand, bird: Bird) -> None:
    """The small green feathers at the edge of the bare face and the brow,
    lying over it irregularly."""
    face = bird.part["face"] | bird.part["brow"]
    near = (soft(face, 4) > 0.03) & bird.part["feathers"]
    sk = np.nan_to_num(bird.model.skull)
    lower = band(-sk[..., 1], 0.45, 0.75) * (soft(bird.part["face"], 3) > 0.03)
    for x, y in hand.seeds(near * 1.0 + 2.0 * lower, 360):
        ix, iy = min(int(x), WIDTH - 1), min(int(y), HEIGHT - 1)
        a = float(bird.angle[iy, ix]) + hand.rng.normal(0.0, 0.3)
        size = hand.rng.uniform(5.0, 12.0)
        start = (x - 0.4 * size * math.cos(a), y - 0.4 * size * math.sin(a))
        tip = (x + size * math.cos(a), y + size * math.sin(a))
        shade = float(bird.shade[iy, ix])
        colour = OLIVE if shade < 0.3 else LEAF if shade < 0.6 else PINE
        hand.sheet.stroke([start, tip], hand.pencil(colour), [0.2, 0.6, 0.15], 2.4)


def draw_brow(hand: Hand, bird: Bird) -> None:
    """The scarlet band over the bill: velvety, in short dense strokes lying
    back from the bill, deep crimson in its shadow and scarlet in the light;
    its back edge broken over the green of the crown."""
    brow = bird.part["brow"]
    w = soft(brow, 2) * (soft(brow, 2) > 0.2)
    clip = soft(brow, 3) > 0.15
    s = band(bird.shade, 0.1, 0.8)
    angle = bird.angle
    hand.tone(CRIMSON, w * (0.5 + 0.5 * s), 0.45, (-30,), 2.6)
    hand.tone(SCARLET, w, 0.5, (40,), 2.6)
    for colour, weight, firm, cover in (
        (OXBLOOD, w * s**2, 0.6, 1.8),
        (CRIMSON, w * (0.4 + 0.6 * s), 0.62, 2.2),
        (SCARLET, w, 0.64, 3.0),
        (FLAME, w * (1.0 - s) * 0.5, 0.4, 0.8),
    ):
        hand.layer(
            colour, weight, angle, firm, length=5, width=2.6, cover=cover, clip=clip
        )


def eye_masks(bird: Bird):
    """Where the iris, the pupil and the catchlight are seen, and the
    shadow of the upper lid over the iris."""
    sk = np.nan_to_num(bird.model.skull, nan=9.0)
    near = sk[..., 2] > 0.2
    r = np.hypot(sk[..., 0], sk[..., 1])
    iris = near & (r < mm.IRIS)
    pupil = near & (r < mm.PUPIL)
    glint = np.array([0.4, 0.45]) * mm.IRIS
    spared = np.hypot(sk[..., 0] - glint[0], sk[..., 1] - glint[1]) > 0.22 * mm.IRIS
    hood = iris * band(sk[..., 1], 0.1 * mm.IRIS, 0.85 * mm.IRIS)
    rim = iris * band(r, 0.7 * mm.IRIS, 0.95 * mm.IRIS)
    return iris & spared, pupil & spared, hood, rim, glint


def draw_eye(hand: Hand, bird: Bird) -> None:
    """The pale eye in its ring of wrinkled bare skin: straw-yellow iris
    with a darker rim, black pupil, the heavy upper lid and its shadow, the
    lower lid, one catchlight toward the light left as paper."""
    model = bird.model
    iris, pupil, hood, rim, glint = eye_masks(bird)
    draw_eye_skin(hand, bird)
    hand.tone(STRAW, iris * 1.0, 0.55, (10, 100), 2.4)
    hand.tone(OCHRE, np.clip(rim + 0.7 * hood, 0.0, 1.0) * iris, 0.55, (40, -50), 2.4)
    hand.tone(UMBER, (rim * 0.6 + hood * 0.6) * iris, 0.45, (20,), 2.4)
    hand.tone(BLACK, pupil * 1.0, 1.0, (20, -40, 70, 120, 160), 2.2)
    t = np.linspace(0.08 * math.pi, 0.97 * math.pi, 30)
    upper = np.stack([np.cos(t), np.sin(t)], 1) * (mm.IRIS + 0.008)
    pts = mm.on_head(model, upper)
    hand.line(LINE, pts[:, :2], (0.9 * swell(30) + 0.1).tolist(), 3.2, 0.1)
    t = np.linspace(1.12 * math.pi, 1.9 * math.pi, 24)
    lower = np.stack([np.cos(t), np.sin(t)], 1) * (mm.IRIS + 0.01)
    hand.line(
        MAUVE, mm.on_head(model, lower)[:, :2], (0.5 * swell(24)).tolist(), 2.2, 0.1
    )
    centre = mm.on_head(model, [glint])[0, :2]
    for _ in range(3):
        hand.erase([centre - [0.8, 0.0], centre + [0.8, 0.0]], tools.ERASER, 1.0, 3.2)


def draw_eye_skin(hand: Hand, bird: Bird) -> None:
    """The bare skin round the eye: fine wrinkles, short and uneven, lying
    round the eye but never in rings."""
    rng = hand.rng
    for _ in range(46):
        r = rng.uniform(mm.IRIS + 0.015, mm.IRIS + 0.11)
        a = rng.uniform(0.0, 2.0 * math.pi)
        span = rng.uniform(0.25, 0.8)
        t = np.linspace(a, a + span, 8)
        wob = r * (1.0 + 0.08 * np.sin(np.linspace(0, 3, 8) * rng.uniform(1, 3)))
        tilt = rng.normal(0.0, 0.15)
        arc = np.stack([np.cos(t + tilt) * wob, np.sin(t) * wob * 0.9], 1)
        pts = mm.on_head(bird.model, arc)
        seen = mm.visible(bird.model, pts, 6.0, ("plumage", "eye"))
        if seen.all():
            firm = rng.uniform(0.15, 0.32)
            hand.line(
                MAUVE if rng.random() < 0.6 else COOL,
                pts[:, :2],
                (firm * swell(8)).tolist(),
                1.8,
                0.1,
            )


def draw_feet(hand: Hand, bird: Bird) -> None:
    """The feet gripping the branch: thick dark grey toes, rounded under the
    light, ringed with scales; long dark curved claws."""
    toes, claws = bird.part["toes"], bird.part["claws"]
    s = bird.shade
    n = bird.model.seen.normal
    top = toes * np.clip(n @ mm.SUN, 0.0, 1.0) ** 1.5
    mm.draw_feet(bird.model, Pencil(hand, BLACK, 1.6))
    hand.tone(SILVER, top, 0.6, (70, -20), 2.2)
    hand.tone(CHAR, toes * band(s, 0.3, 0.75) * (1 - top), 0.78, (70, -20), 2.4)
    hand.tone(SLATE, toes * (1 - 0.6 * top), 0.6, (30, -50), 2.4)
    hand.tone(GREY, toes * (1 - band(s, 0.3, 0.6)), 0.45, (110,), 2.4)
    lit = toes * (1.0 - band(s, 0.2, 0.55))
    for x, y in hand.seeds(lit, 60):
        a = float(bird.angle[int(y), int(x)])
        stroke = [
            (x - 3 * math.cos(a), y - 3 * math.sin(a)),
            (x + 3 * math.cos(a), y + 3 * math.sin(a)),
        ]
        hand.erase(stroke, tools.KNEADED, [0.2, 0.55, 0.2], 3.0)
    hand.tone(BLACK, claws * 1.0, 0.9, (30, -40, 100), 2.2)
    hand.tone(CHAR, claws * 1.0, 0.8, (75,), 2.2)


def branch_reach(bird: Bird) -> np.ndarray:
    """How much of the branch is drawn at each place, 0..1: all of it under
    the bird, fading out either side."""
    b, _, _ = mm.branch_frame()
    rel = np.nan_to_num(bird.model.bird) - np.asarray(mm.BRANCH_AT)
    s = rel @ b
    return bird.part["branch"] * band(s, -3.6, -2.4) * band(s, 2.9, 1.9)


def draw_branch(hand: Hand, bird: Bird) -> None:
    """The branch, as densely as the bird: long grain strokes of the light
    bark along its top first (so that they hold), then the bark closed in
    layers, darkest beneath and where the bird's shadow crosses it, a crisp
    dark edge underneath; the closed tone gives out first toward the ends,
    where only loosening grain strokes run out into the paper; a few tufts
    of moss."""
    fade = branch_reach(bird)
    where = fade > 0.02
    s, angle = bird.shade, bird.angle
    closed = fade**2
    on = -float(np.degrees(np.median(angle[where])))
    grain = {"clip": where, "spread": 0.03, "bend": 0.003, "width": 1.8}
    lit = 1.0 - band(s, 0.3, 0.7)
    hand.layer(HORN, fade * lit, angle, 0.7, length=46, cover=0.6, **grain)
    hand.layer(WARM_GREY, fade * band(s, 0.2, 0.6), angle, 0.6, length=46, cover=0.4,
               **grain)  # fmt: skip
    hand.tone(CHAR, closed * band(s, 0.82, 1.0), 0.6, (on, on + 25), 3.0)
    hand.tone(SEPIA, closed * band(s, 0.3, 0.7), 0.66, (on - 20, on + 10), 3.0)
    hand.tone(
        UMBER, closed * trapezoid(s, (0.0, 0.25, 0.65, 0.85)), 0.62, (on + 5,), 3.0
    )
    hand.tone(WARM_GREY, closed * lit, 0.55, (on, on + 15), 3.0)
    hand.layer(SEPIA, fade, angle, 0.72, length=50, cover=0.9, **grain)
    hand.layer(
        CHAR, fade * band(s, 0.4, 0.8), angle, 0.7, length=44, cover=0.5, **grain
    )
    edge = (fade > 0.3) & ~(soft(where, 1) > 0.99)
    hand.tone(BLACK, edge * (s > 0.5) * 1.0, 0.8, (on,), 2.0)
    draw_moss(hand, bird)


#: Where along the branch (from under the feet) the tufts of moss grow.
MOSS = (-2.3, -1.15, 1.25)


def draw_moss(hand: Hand, bird: Bird) -> None:
    """Tufts of moss on top of the branch, stippled: a dark heart, bright
    yellow-green tips catching the light."""
    rng = hand.rng
    b, up, side = mm.branch_frame()
    branch = bird.part["branch"]
    for along in MOSS:
        p = (
            np.asarray(mm.BRANCH_AT)
            + b * along
            + mm.BRANCH_R * (0.92 * up + 0.4 * side)
        )
        x, y, _ = bird.model.pose.to_sheet(p)
        for colour, spread, n, firm in ((PINE, 5.0, 90, 0.9), (GRASS, 7.0, 150, 0.85),
                                        (CHARTREUSE, 7.0, 120, 0.8),
                                        (LEMON, 5.0, 40, 0.7)):  # fmt: skip
            dots = np.array([x, y]) + rng.normal(0, spread, (n, 2)) * [1.8, 0.6]
            for dx, dy in dots:
                ix, iy = int(dx), int(dy)
                if 0 <= ix < WIDTH and 0 <= iy < HEIGHT and branch[iy, ix]:
                    a = rng.uniform(-2.0, -1.1)
                    pts = [(dx, dy), (dx + 3.0 * math.cos(a), dy + 3.0 * math.sin(a))]
                    hand.sheet.stroke(pts, hand.pencil(colour), [firm, 0.4], 2.0)


def finish_map(bird: Bird) -> np.ndarray:
    """How far each place is carried: furthest at the head, less down the
    body, the tail loosening toward its end."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    eye = np.hypot(xx - mm.EYE_AT[0], yy - mm.EYE_AT[1])
    out = 1.0 - 0.3 * band(eye, 350.0, 1300.0)
    return out * (1.0 - 0.45 * band(yy, HEIGHT - 420.0, HEIGHT - 10.0))


def main() -> None:
    rng = np.random.default_rng(SEED)
    paper = Paper("wove", grain=2.0, depth=0.4, bite=0.5, give=0.4, color="#fbfaf6")
    sheet = Sheet(WIDTH, HEIGHT, paper=paper, seed=SEED)
    hand = Hand(sheet, rng)
    bird = build(rng)
    hand.finish = finish_map(bird)
    plumage = build_plumage(bird, rng)
    draw_bill(hand, bird)
    lay_feathers(hand, plumage, zones(plumage, bird.shade))
    draw_face(hand, bird)
    draw_fringe(hand, bird)
    draw_brow(hand, bird)
    draw_eye(hand, bird)
    draw_feet(hand, bird)
    draw_branch(hand, bird)
    out = Path(__file__).parent / "output" / "macaw.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
