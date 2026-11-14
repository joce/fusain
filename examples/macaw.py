"""A great green macaw, in coloured pencil on white paper.

Run from the repository root: ``python examples/macaw.py``. Writes
``examples/output/macaw.png`` (in about two minutes).

A great green macaw (Ara ambiguus) on a branch, its body turned a little
toward us and leaning forward, its head turned further and cocked, the long
tail hanging down to the lower right and out of the sheet.

The bird is not drawn from guessed outlines but built: ``macaw_model.py``
holds a solid model of it whose shapes were measured on reference
photographs (the bill's arch and hook, the bare face and its rows of tiny
feathers, the red forehead, the eye set high on a flat head, the folded
wing and where its coverts end, the tail), posed and lit from the upper
left, with its cast shadows. Every pixel of the sheet knows where it lies
on the bird: how much light it takes, what colour the bird is there, and
which way its feathers run.

The colour is laid as coloured pencil is, and as the model asks: darks
first while the tooth takes them, each colour hatched lightly closed and
then stroked along the lie of the feathers, short on the head, longer on
the body, long along the flight and tail feathers. The feathers' edges are
stated only where the light rakes them; the edges of the forms are found
dark where one form lies over another in shadow and lost in the light. The
head is carried furthest: the bill built in strokes along its arch, the
eye, the bare face with its broken lines of tiny dark feathers, the velvet
red brow. The drawing loosens toward the end of the tail.
"""

# cspell:words ambiguus culmen headness lognormal tomium TURQ

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import macaw_model as mm
import numpy as np
from murex import noise, write_png
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


@dataclass
class Bird:
    """The modelled bird on the sheet: where each part is seen (``part``),
    how dark it falls under the light (``shade``, 0 in the full light to 1
    in the deepest shadow), which way its feathers run (``angle``, radians
    on the sheet), and the model itself, its wing and small feathers."""

    model: mm.Model
    wing: dict
    tiles: mm.Tiles
    part: dict[str, np.ndarray]
    shade: np.ndarray
    angle: np.ndarray
    mottle: np.ndarray


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
    wing = mm.feather_wing(model, rng)
    tiles = mm.feather_plumage(model, rng)
    names = ["plumage", "wing", "upper", "lower", "eye", "toes", "claws", "branch"]
    part = {name: model.part(name) for name in names}
    part["eye"] = part["eye"] & False
    part["face"] = mm.face_mask(model) > 0.5
    part["brow"] = (mm.brow_mask(model) > 0.5) & ~part["face"]
    part["feathers"] = part["plumage"] & ~part["face"] & ~part["brow"]
    for k in range(len(mm.BLADES)):
        part[f"tail{k}"] = model.part(f"tail{k}")
    part["tail"] = np.logical_or.reduce([part[f"tail{k}"] for k in range(6)])
    shade = shading(model, part)
    feathered = part["feathers"] | part["wing"]
    shade = np.where(feathered, stretch(shade, feathered) ** 1.35, shade)
    shade = np.clip(shade + feather_shadows(wing, part["wing"]), 0.0, 1.0)
    mottle = noise.fractal_noise_2d(HEIGHT, WIDTH, 40.0, rng, octaves=4)
    mottle = (mottle - mottle.min()) / (np.ptp(mottle) + 1e-9)
    return Bird(model, wing, tiles, part, shade, directions(model), mottle)


def feather_shadows(wing: dict, seen: np.ndarray) -> np.ndarray:
    """How much darker each place of the wing is for the feather over it:
    each covert dark just under the tips of the row above and lighter toward
    its own tip; each flight feather dark beside the edge of the one over
    it."""
    covert = seen & (wing["row"] >= 0)
    flight = seen & (wing["row"] < 0) & (wing["blade"] >= 0)
    q, qf = wing["q"], wing["qf"]
    last = wing["row"] == len(wing["rows"]) - 1
    out = covert * (0.22 * (1.0 - q) ** 3 - 0.06 * q + 0.2 * q * last)
    lit = np.where(wing["blade"] < 4, 0.2, 0.08) * band(qf, 0.35, 0.85)
    return out + flight * (0.36 * (1.0 - qf) ** 2.5 - lit)


# ----------------------------------------------------------------------
# the box of pencils (as in the first state of this drawing)

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

#: Layers of colour, laid in this order, the darks first while the tooth
#: still takes them: the pencil; where on the scale of shade it goes on
#: (rising from a to b, falling from c to d); how firmly; how many times
#: over it covers; the width of the point; how far over the pencil is held;
#: and how firmly it is also hatched in, closed and waxy.
GREENS = [
    (NIGHT, (0.8, 0.95, 2.0, 2.1), 0.8, 1.3, 4.0, 0.0, 0.4),
    (DEEP, (0.62, 0.85, 2.0, 2.1), 0.75, 1.45, 4.0, 0.0, 0.4),
    (PINE, (0.45, 0.68, 2.0, 2.1), 0.64, 1.45, 4.0, 0.0, 0.4),
    (LEAF, (0.3, 0.5, 0.9, 1.05), 0.55, 1.45, 4.0, 0.0, 0.4),
    (MOSS, (0.15, 0.35, 0.7, 0.9), 0.52, 1.5, 4.0, 0.0, 0.4),
    (OLIVE, (0.0, 0.2, 0.5, 0.7), 0.5, 1.5, 4.0, 0.0, 0.4),
    (CHARTREUSE, (-0.1, 0.05, 0.35, 0.5), 0.44, 1.35, 4.0, 0.0, 0.4),
    (LEMON, (-0.2, -0.1, 0.18, 0.35), 0.36, 1.1, 4.0, 0.0, 0.4),
]
BLUES = [
    (INDIGO, (0.8, 0.93, 2.0, 2.1), 0.85, 1.3, 3.0, 0.0, 0.5),
    (NAVY, (0.62, 0.8, 2.0, 2.1), 0.75, 1.4, 3.5, 0.0, 0.45),
    (ULTRA, (0.42, 0.62, 0.9, 1.0), 0.62, 1.4, 4.0, 0.0, 0.4),
    (COBALT, (0.18, 0.38, 0.7, 0.86), 0.55, 1.4, 4.0, 0.0, 0.4),
    (TURQ, (-0.1, 0.05, 0.4, 0.6), 0.46, 1.2, 4.0, 0.0, 0.4),
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
HORN_LAYERS = [
    (BLACK, (0.74, 0.92, 2.0, 2.1), 0.8, 3.6),
    (CHAR, (0.5, 0.72, 2.0, 2.1), 0.72, 3.6),
    (SLATE, (0.28, 0.5, 0.82, 0.92), 0.62, 3.4),
    (GREY, (0.1, 0.28, 0.55, 0.7), 0.5, 3.4),
]


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


def lay_horn(hand: Hand, bird: Bird, horn: Horn) -> None:
    """Lay one mandible: every stroke along its arch, the darks first; the
    highlight left lighter and touched with a pale grey."""
    model, rng = bird.model, hand.rng
    step = 0.5 / len(horn.f)
    for colour, edges, firm, width in HORN_LAYERS:
        for f, curve in zip(horn.f, horn.curves, strict=True):
            pts = mm.on_bill(model, horn.which, curve)
            seen = mm.visible(model, pts, 5.0, (horn.part,))
            s = sample(bird.shade, pts)
            t = np.linspace(0.0, 1.0, len(curve))
            fj = f + rng.normal(0.0, step)
            d, light = horn.dark(fj, t, s)
            bear = firm * trapezoid(d, edges) * (1.0 - 0.8 * light) * seen
            for a, b in pieces(rng, len(pts), 9, 18):
                if bear[a:b].max() > 0.03:
                    hand.line(
                        colour,
                        pts[a:b, :2],
                        bear[a:b] * swell(b - a) ** 0.3,
                        width,
                        0.12,
                    )
    for f, curve in zip(horn.f, horn.curves, strict=True):
        pts = mm.on_bill(model, horn.which, curve)
        seen = mm.visible(model, pts, 5.0, (horn.part,))
        t = np.linspace(0.0, 1.0, len(curve))
        _, light = horn.dark(f, t, sample(bird.shade, pts))
        bear = 0.55 * light * seen
        if bear.max() > 0.05:
            hand.line(SILVER, pts[:, :2], bear, 3.0, 0.1)


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
    upper_f, upper_c = arch_family(mm.CULMEN, mm.TOMIUM, 72, mm.BILL_BASE)
    lower_f, lower_c = arch_family(mm.LOWER[:7], mm.LOWER[12:5:-1], 52)
    for name, colour, firm in (("lower", CHAR, 0.55), ("upper", SLATE, 0.45)):
        mask = bird.part[name]
        hand.layer(
            colour,
            mask * 1.0,
            bird.angle,
            firm,
            length=16,
            width=3.0,
            cover=1.4,
            clip=mask,
            spread=0.02,
            bend=0.0,
        )
    lay_horn(hand, bird, Horn(1, "lower", lower_f, lower_c, lower_dark))
    lay_horn(hand, bird, Horn(0, "upper", upper_f, upper_c, upper_dark))
    warm_tip(hand, bird, upper_c)
    bill_edges(hand, bird)


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
    below = mm.on_bill(model, 1, tomium + mm.grain_offset(tomium, -0.03), 1.4)
    for a, b in spans(mm.visible(model, below, 8.0)):
        hand.erase(below[a:b, :2], tools.KNEADED, (0.45 * swell(b - a)).tolist(), 2.2)
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
    base = mm.smooth_curve(mm.bill_point(mm.BILL_BASE), 40)
    pts = mm.on_bill(model, 0, base + mm.grain_offset(base, 0.012))
    seen = mm.visible(model, pts, 5.0, ("upper",))
    hand.line(CHAR, pts[:, :2], 0.7 * swell(40) * seen, 2.6, 0.1)


#: The pencils of the tail, darks first: colour, which of its colours it
#: lays (olive near the body, rust, blue toward the tip, pale blue at the
#: very tip), where on the scale of darkness, how firmly.
TAIL_LAYERS = [
    (PINE, 0, (0.55, 0.8, 2.0, 2.1), 0.6),
    (OXBLOOD, 1, (0.55, 0.8, 2.0, 2.1), 0.62),
    (NAVY, 2, (0.55, 0.8, 2.0, 2.1), 0.68),
    (MOSS, 0, (0.2, 0.45, 0.8, 0.95), 0.5),
    (MADDER, 1, (0.2, 0.45, 0.85, 1.0), 0.56),
    (ULTRA, 2, (0.3, 0.55, 0.85, 1.0), 0.55),
    (OLIVE, 0, (-0.1, 0.0, 0.4, 0.6), 0.4),
    (VERMILION, 1, (-0.1, 0.05, 0.45, 0.65), 0.32),
    (UMBER, 1, (0.05, 0.25, 0.8, 0.95), 0.4),
    (OCHRE, 1, (-0.1, 0.0, 0.3, 0.5), 0.3),
    (COBALT, 2, (-0.1, 0.05, 0.5, 0.7), 0.48),
    (SKY, 3, (-0.1, 0.0, 1.5, 2.0), 0.45),
]


def tail_colours(t, across, k: int) -> list[np.ndarray]:
    """How much of each of the tail's colours there is along a feather:
    olive-rust near the body, rust in the middle, blue toward the tip (and
    along the outer web of the outer feathers), pale blue at the tip."""
    start = 0.62 - 0.07 * (k // 2)
    outer = (0.4 if k >= 2 else 0.12) * band(np.abs(across), 0.45, 0.95)
    blue = np.clip(band(t, start, start + 0.2) + outer + 0.2 * (k >= 4), 0.0, 1.0)
    sky = band(t, 0.88, 0.98)
    olive = band(t, 0.3, 0.08) * (1.0 - blue)
    return [olive, np.clip(1.0 - blue - 0.6 * olive, 0.0, 1.0), blue * (1 - sky), sky]


def blade_lines(bird: Bird, k: int, count: int = 22, n: int = 160):
    """The lines along tail feather ``k``, on the sheet (count, n, 3), with
    where along (n,) and across (count,) each lies, and which of their
    points are seen."""
    _, frac, _, half = mm.BLADES[k]
    length = mm.TAIL_LENGTH * frac
    patch = mm.bird_patches()[1][1 + k]
    t = np.linspace(0.01, 0.995, n)
    across = np.linspace(-0.92, 0.92, count)
    width = mm.blade_half(t, half)
    out = np.empty((count, n, 3))
    for i, a in enumerate(across):
        local = np.stack([t * length, a * width], 1)
        out[i] = bird.model.pose.to_sheet(patch.lift(local))
    seen = np.array([mm.visible(bird.model, p, 3.0, (f"tail{k}",)) for p in out])
    return out, t, across, seen


def overlap_shadow(bird: Bird, k: int) -> np.ndarray:
    """The shadow the feathers lying over feather ``k`` throw on it, along
    their edges."""
    above = np.zeros((HEIGHT, WIDTH), bool)
    for j in range(k):
        above |= bird.part[f"tail{j}"]
    above |= bird.part["plumage"] | bird.part["wing"]
    return soft(above, 4) * ~above


def draw_blade(hand: Hand, bird: Bird, k: int) -> None:
    """One long tail feather in long strokes along it, darks first: its
    colour changing along its length, light catching one side, the shadow
    of the feathers over it, loosening and paling toward its end."""
    lines, t, across, seen = blade_lines(bird, k)
    shadow = overlap_shadow(bird, k)
    rng = hand.rng
    for colour, which, edges, firm in TAIL_LAYERS:
        for pts, a, vis in zip(lines, across, seen, strict=True):
            d = sample(bird.shade, pts) + 0.5 * sample(shadow, pts)
            d = np.clip(d - 0.2 * band(a, 0.25, 0.8) + 0.1 * band(a, -0.3, -0.9), 0, 1)
            weight = tail_colours(t, a, k)[which]
            loose = 1.0 - 0.6 * band(t, 0.6, 1.0)
            bear = firm * trapezoid(d, edges) * weight * loose * vis
            for i, j in pieces(rng, len(t), 14, 30):
                if rng.random() < 0.6 * float(band(t[i], 0.65, 1.0)):
                    continue
                if bear[i:j].max() > 0.03:
                    hand.line(
                        colour,
                        pts[i:j, :2],
                        bear[i:j] * flick(np.linspace(0, 1, j - i)) ** 0.4,
                        3.4,
                        0.2,
                    )
    shaft(hand, bird, k)


def shaft(hand: Hand, bird: Bird, k: int) -> None:
    """A tail feather's shaft: one dark line along it, a little off its
    middle, broken here and there, fading toward the tip."""
    lines, t, _, seen = blade_lines(bird, k, count=1)
    pts, vis = lines[0], seen[0]
    blue = tail_colours(t, np.zeros_like(t), k)[2]
    for colour, w in ((OXBLOOD, 1.0 - blue), (NAVY, blue)):
        bear = 0.7 * w * (1.0 - 0.6 * t) * vis
        for i, j in pieces(hand.rng, len(t), 18, 40):
            if hand.rng.random() < 0.15 or bear[i:j].max() < 0.05:
                continue
            hand.line(colour, pts[i:j, :2], bear[i:j], 1.8, 0.1)


def draw_tail(hand: Hand, bird: Bird) -> None:
    """The long tail: narrow feathers stepped in length, the outer ones under
    and shorter, each laid as its own blade."""
    for k in range(len(mm.BLADES) - 1, -1, -1):
        if bird.part[f"tail{k}"].any():
            draw_blade(hand, bird, k)


def found_edge(hand: Hand, bird: Bird, pts3: np.ndarray, colour: str) -> None:
    """A found edge: a sure stroke, dark where the form lies over another in
    shadow and lost where the light takes it."""
    seen = mm.visible(bird.model, pts3, 4.0)
    ix = np.clip(pts3[:, 0].astype(int), 0, WIDTH - 1)
    iy = np.clip(pts3[:, 1].astype(int), 0, HEIGHT - 1)
    dark = band(bird.shade[iy, ix], 0.25, 0.7) * seen
    for a, b in spans(dark > 0.05):
        bear = 0.85 * dark[a:b] * swell(b - a)
        hand.line(colour, pts3[a:b, :2], bear, hand.rng.uniform(3.0, 4.5), 0.4)


def draw_edges(hand: Hand, bird: Bird) -> None:
    """The edges that are found: the wing's lower edge over the flank, the
    lower bill against the throat, the belly over the perch."""
    model = bird.model
    lower = np.asarray(mm.WING_OUTLINE[10:] + mm.WING_OUTLINE[:1])
    edge = mm.smooth_curve(lower, 120)
    wing = mm.bird_patches()[1][0]
    inward = edge + np.array([0.0, 0.03])
    found_edge(hand, bird, model.pose.to_sheet(wing.lift(inward)), SPRUCE)
    chin = mm.smooth_curve(mm.bill_point(mm.LOWER[5:11]), 40)
    found_edge(hand, bird, mm.on_bill(model, 1, chin, 0.6), BLACK)


def draw_plumage(hand: Hand, bird: Bird) -> None:
    """The head, neck, breast, belly and thighs: the colour, darks first, in
    strokes from the root of each feather toward its tip, short on the head
    and longer on the body; a hint of turquoise on the rump."""
    s, vary = bird.shade, 0.8 + 0.4 * bird.mottle
    feathers = bird.part["feathers"]
    head = bird.model.headness > 0.5
    clip = soft(feathers, 2) > 0.3
    b = np.nan_to_num(bird.model.bird)
    rump = band(b[..., 0], -3.0, -3.5) * band(b[..., 1], -4.0, -4.6) * feathers
    hand.layer(TURQ, rump * 0.8, bird.angle, 0.45, length=14, cover=1.4, clip=clip)
    for where, length, scale in (
        (feathers & ~head, 18.0, 1.1),
        (feathers & head, 8.0, 0.85),
    ):
        build_up(
            hand,
            GREENS,
            where * 1.0,
            s,
            bird.angle,
            vary=vary,
            length=length,
            scale=scale,
            clip=clip,
            spread=0.22,
        )


def draw_wing(hand: Hand, bird: Bird) -> None:
    """The folded wing: its structure first, while the tooth still takes a
    dark line (the coverts' tips where the light rakes them, the flight
    feathers' edges); then the green coverts, the secondaries going from
    green to blue, the blue primaries along the lower edge and the tip."""
    model, wing = bird.model, bird.wing
    covert_accents(hand, bird, PINE, 1.1)
    mm.draw_flight(model, Pencil(hand, NIGHT, 1.1), hand.rng)
    seen = bird.part["wing"]
    blade, u = wing["blade"], wing["u"]
    covert = seen & (wing["row"] >= 0)
    flight = seen & ~covert
    blue = np.clip(band(blade, 3.5, 5.5) + band(u, 0.7, 0.85), 0.0, 1.0) * flight
    teal = band(blade, 1.5, 3.0) * flight * (1.0 - blue)
    green = np.clip(covert + flight * (1.0 - blue - teal), 0.0, 1.0)
    angle, s = bird.angle, bird.shade
    hand.layer(TURQ, teal, angle, 0.5, length=26, cover=1.5, clip=seen)
    vary = 0.75 + 0.5 * bird.mottle
    build_up(
        hand,
        GREENS,
        green * covert,
        s,
        angle,
        vary=vary,
        length=8,
        clip=seen,
        spread=0.1,
    )
    build_up(
        hand,
        GREENS,
        green * flight,
        s,
        angle,
        vary=vary,
        length=22,
        clip=seen,
        spread=0.04,
    )
    deeper = np.clip(s + 0.25 * band(u, 0.75, 1.0), 0.0, 1.0)
    build_up(
        hand, BLUES, blue + 0.5 * teal, deeper, angle, length=40, clip=seen, spread=0.05
    )
    covert_accents(hand, bird, DEEP, 0.9, 0.5)
    mm.draw_flight(model, Pencil(hand, NIGHT, 0.9), hand.rng)
    lift_flight_edges(hand, bird)


def covert_arc(bird: Bird, row: dict, a: float, b: float) -> np.ndarray:
    """The rounded tip of one covert (from ``a`` to ``b`` across the wing),
    just below its edge, on the sheet."""
    v = np.linspace(a, b, 14)
    d = mm.row_tip(row, v)
    u = d + mm.ROW_SLANT * (v - mm.ROW_MID) - mm.ROW_BOW * (v - mm.ROW_MID) ** 2
    return mm.on_wing(bird.model, np.stack([u / mm.wing_frame()[3] + 0.004, v], 1))


def covert_accents(hand: Hand, bird: Bird, colour: str, gain: float, least=0.0):
    """The coverts' tips stated as a painter states them: firmly where the
    light rakes the wing and at a few overlaps, often broken, elsewhere
    left to the soft step in tone; the lowest row hardly at all, so that it
    merges into the shadow of the flight feathers."""
    rows, rake, rng = bird.wing["rows"], mm.rake(bird.model), hand.rng
    for k, row in enumerate(rows):
        last = k == len(rows) - 1
        for a, b in pairwise(row["edges"]):
            pts = covert_arc(bird, row, a, b)
            seen = mm.visible(bird.model, pts, 4.0, ("wing",))
            if seen.sum() < 4:
                continue
            r = float(sample(rake, pts[seen]).mean())
            chance = max(r**2, 0.08) * (0.2 if last else 1.0)
            if r < least or rng.random() > chance:
                continue
            i, j = int(rng.integers(0, 4)), 14 - int(rng.integers(0, 5))
            bear = gain * (0.35 + 0.65 * r) * swell(j - i) * seen[i:j]
            width = 1.8 + 6.0 * row["w"]
            hand.line(colour, pts[i:j, :2], bear, width, 0.3)


def lift_flight_edges(hand: Hand, bird: Bird) -> None:
    """The light caught along each flight feather's free edge, lifted out
    with the kneaded eraser just inside its dark line."""
    for j, (edge, reach) in enumerate(mm.FLIGHT[:-1]):
        u = np.linspace(mm.GREATER + 0.02, reach - 0.04, 40)
        up, low = mm.wing_edges(u)
        v = up + (edge - 0.03) * (low - up)
        pts = mm.on_wing(bird.model, np.stack([u, v], 1))
        seen = mm.visible(bird.model, pts, 4.0, ("wing",))
        firm, width = (0.75, 5.0) if j < 4 else (0.5, 4.0)
        for a, b in spans(seen):
            bear = firm * swell(b - a)
            hand.erase(pts[a:b, :2], tools.KNEADED, bear.tolist(), width)


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


def draw_ticks(hand: Hand, bird: Bird) -> None:
    """The small feathers stated where the light rakes them: a crisp dark
    touch under each one's rounded tip, tiny on the head and neck, larger
    down the breast; left out in the full light and lost in the shadow."""
    model, tiles = bird.model, bird.tiles
    direction, short = mm.flow2d(model)
    size = mm.feather_size(model) * mm.UNIT
    rake = mm.rake(model)
    for x, y in tiles.seeds:
        ix, iy = int(x), int(y)
        r = float(rake[iy, ix])
        turn = float(trapezoid(bird.shade[iy, ix], (0.25, 0.4, 0.62, 0.78)))
        if not bird.part["feathers"][iy, ix] or hand.rng.random() > 0.85 * r * turn:
            continue
        d = direction[iy, ix]
        length, width = size[iy, ix] * short[iy, ix], 0.85 * size[iy, ix]
        t = np.linspace(-0.8, 0.8, 7)
        along = 0.5 * length * np.sqrt(1 - t**2) + 0.8
        pts = (
            np.array([x, y])
            + np.outer(along, d)
            + np.outer(t * width / 2, [-d[1], d[0]])
        )
        colour = PINE if bird.shade[iy, ix] < 0.55 else DEEP
        firm = (0.25 + 0.4 * r) * hand.rng.uniform(0.7, 1.1)
        hand.line(colour, pts, (firm * swell(7)).tolist(), 2.0 if length < 12 else 2.6)


def draw_thigh(hand: Hand, bird: Bird) -> None:
    """The thigh's feathers breaking its edge and falling over the top of
    the foot, so that the leg is not a lump on a peg."""
    feathers, toes = bird.part["feathers"], bird.part["toes"] | bird.part["claws"]
    bird_mask = np.isfinite(bird.model.seen.z) & ~bird.part["branch"]
    over_foot = feathers & (soft(toes, 8) > 0.02)
    edge = feathers & (soft(~bird_mask, 3) > 0.05)
    near_thigh = mm.implicit_map(bird.model, mm.THIGHS) < 1.2
    for where, count, reach in (
        (over_foot, 70, (8.0, 16.0)),
        (edge & near_thigh, 60, (5.0, 10.0)),
    ):
        for x, y in hand.seeds(where * 1.0, count):
            ix, iy = int(x), int(y)
            a = float(bird.angle[iy, ix]) + hand.rng.normal(0.0, 0.25)
            size = hand.rng.uniform(*reach)
            start = (x - 0.3 * size * math.cos(a), y - 0.3 * size * math.sin(a))
            tip = (x + size * math.cos(a), y + size * math.sin(a))
            shade = float(bird.shade[iy, ix])
            colour = MOSS if shade < 0.35 else LEAF if shade < 0.6 else PINE
            mid = (0.5 * (start[0] + tip[0]), 0.5 * (start[1] + tip[1]))
            hand.line(colour, [start, mid, tip], [0.55, 0.45, 0.1], 2.6, 0.2)


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
    hand.tone(CHAR, toes * band(s, 0.45, 0.85), 0.72, (70, -20), 2.4)
    hand.tone(SLATE, toes * 1.0, 0.58, (30, -50), 2.4)
    hand.tone(GREY, toes * (1 - band(s, 0.3, 0.6)), 0.45, (110,), 2.4)
    mm.draw_feet(bird.model, Pencil(hand, CHAR, 1.7))
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


def draw_branch(hand: Hand, bird: Bird) -> None:
    """The branch, drawn only where the bird holds it: round, light on top
    and dark beneath, the bird's shadow across it, the grain of its bark in
    long broken lines, fading out either side."""
    model = bird.model
    fade = mm.branch_fade(model)
    w = bird.part["branch"] * fade
    b, up, _ = mm.branch_frame()
    s = bird.shade
    hand.tone(SEPIA, w * band(fade, 0.4, 0.8) * band(s, 0.35, 0.9), 0.6, (-5, 30), 2.6)
    clip = bird.part["branch"] & (fade > 0.05)
    hand.layer(
        UMBER,
        w * band(s, 0.15, 0.5),
        bird.angle,
        0.45,
        length=34,
        width=4.0,
        cover=1.6,
        clip=clip,
    )
    hand.layer(
        WARM_GREY, w, bird.angle, 0.32, length=40, width=4.0, cover=1.0, clip=clip
    )
    for off in hand.rng.uniform(-0.8, 0.9, 12):
        along = np.linspace(-4.0, 4.0, 80)
        axis = np.asarray(mm.BRANCH_AT) + np.outer(along, b) + up * off * mm.BRANCH_R
        axis = axis + np.cross(b, up) * mm.BRANCH_R * math.sqrt(max(1 - off**2, 0.0))
        pts = model.pose.to_sheet(axis)
        ix = np.clip(pts[:, 0].astype(int), 0, WIDTH - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, HEIGHT - 1)
        seen = mm.visible(model, pts, 4.0, ("branch",))
        grain = fade[iy, ix] * seen * hand.rng.uniform(0.25, 0.5)
        grain *= 0.5 + 0.5 * np.sin(pts[:, 0] / 17.0 + off * 9)
        for a, c in spans(grain > 0.05):
            hand.line(SEPIA, pts[a:c, :2], grain[a:c].tolist(), 2.0, 0.6)


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
    draw_bill(hand, bird)
    draw_tail(hand, bird)
    draw_edges(hand, bird)
    draw_plumage(hand, bird)
    draw_wing(hand, bird)
    draw_face(hand, bird)
    draw_fringe(hand, bird)
    draw_ticks(hand, bird)
    draw_brow(hand, bird)
    draw_eye(hand, bird)
    draw_feet(hand, bird)
    draw_thigh(hand, bird)
    draw_branch(hand, bird)
    out = Path(__file__).parent / "output" / "macaw.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
