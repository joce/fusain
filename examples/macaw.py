"""A great green macaw, in coloured pencil on white paper.

Run from the repository root: ``python examples/macaw.py``. Writes
``examples/output/macaw.png``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from murex import CatmullRom, noise, write_png
from murex.filters import box_blur, max_filter, smoothstep

from fusain import Medium, Paper, Sheet, pencils, tools

WIDTH, HEIGHT = 1100, 1450
SEED = 7
#: Toward the light: from the upper left, in front of the sheet.
LIGHT = np.array([-0.45, -0.6, 0.66]) / math.sqrt(0.45**2 + 0.6**2 + 0.66**2)

#: The outlines of the bird, each a closed curve through these points.
HEAD_BODY = [
    (352, 236), (372, 205), (405, 183), (450, 173), (497, 180), (535, 202),
    (560, 235), (588, 300), (606, 370), (628, 440), (662, 540), (695, 660),
    (722, 780), (742, 900), (752, 985), (742, 1030), (700, 1040), (640, 1018),
    (585, 978), (536, 925), (494, 860), (456, 780), (428, 690), (412, 600),
    (404, 520), (393, 462), (378, 420), (366, 370), (356, 300),
]  # fmt: skip
HEAD = [
    (352, 236), (372, 205), (405, 183), (450, 173), (497, 180), (535, 202),
    (560, 235), (582, 290), (590, 350), (560, 410), (500, 450), (440, 470),
    (396, 452), (380, 420), (366, 370), (356, 300),
]  # fmt: skip
WING = [
    (478, 520), (500, 465), (545, 425), (600, 415), (640, 450), (670, 530),
    (700, 640), (722, 760), (740, 880), (758, 1000), (778, 1110), (796, 1222),
    (781, 1212), (752, 1110), (712, 1000), (665, 900), (612, 800), (560, 700),
    (515, 620), (488, 570),
]  # fmt: skip
TAIL = [
    (692, 1010), (752, 1000), (810, 1065), (885, 1185), (958, 1305), (1015, 1405),
    (1030, 1446), (1003, 1440), (935, 1350), (855, 1245), (770, 1140), (708, 1068),
]  # fmt: skip
FACE = [
    (362, 250), (395, 236), (436, 243), (470, 268), (486, 308), (478, 350),
    (455, 384), (420, 398), (392, 396), (378, 366), (374, 320), (366, 280),
]  # fmt: skip
BROW = [
    (346, 240), (350, 214), (370, 195), (400, 186), (420, 198), (408, 222),
    (388, 238), (364, 247),
]  # fmt: skip
UPPER_BILL = [
    (358, 234), (318, 238), (285, 256), (262, 290), (252, 332), (255, 374),
    (267, 412), (281, 438), (292, 453), (299, 446), (302, 428), (310, 400),
    (330, 378), (358, 366), (382, 362), (380, 320), (371, 285), (364, 252),
]  # fmt: skip
LOWER_BILL = [
    (388, 352), (395, 405), (380, 440), (352, 462), (322, 458), (304, 440),
    (300, 420), (310, 392), (330, 372), (358, 358),
]  # fmt: skip
EYE = (424.0, 292.0)
IRIS, PUPIL = 14.5, 6.5
#: The folded wing's axis, from the shoulder to the tip.
WING_AXIS = ((560.0, 430.0), (795.0, 1220.0))
#: The branch the bird grips: its middle line and half its thickness.
BRANCH = [(120, 1074), (360, 1022), (560, 985), (760, 952), (930, 928), (1100, 905)]
BRANCH_HALF = 21.0
#: The toes that show, each from the ankle to the claw.
TOES = [
    [(592, 958), (580, 970), (570, 986), (566, 1000)],
    [(603, 960), (601, 976), (604, 992), (610, 1003)],
    [(655, 950), (649, 964), (650, 980)],
    [(667, 951), (670, 966), (676, 979)],
]


# ----------------------------------------------------------------------
# geometry


def outline(points, spacing: float = 2.0) -> np.ndarray:
    """A closed smooth curve through ``points``, sampled every ``spacing``."""
    curve = CatmullRom(np.asarray(points, dtype=np.float64), closed=True)
    return curve.sample(spacing)[0]


def fill(points) -> np.ndarray:
    """Raster mask (H, W) of the inside of the closed curve through ``points``."""
    poly = outline(points)
    mask = np.zeros((HEIGHT, WIDTH), bool)
    x0, y0 = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
    x1, y1 = min(x1, WIDTH), min(y1, HEIGHT)
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


def grid() -> tuple[np.ndarray, np.ndarray]:
    """The x and the y of every pixel, (H, W) each."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float64)
    return xx, yy


def across_branch() -> np.ndarray:
    """Where each pixel lies across the branch: -1 on its top edge, 1 on its
    bottom edge (beyond either, it is off the branch)."""
    line = CatmullRom(np.asarray(BRANCH, dtype=np.float64)).sample(2.0)[0]
    xx, yy = grid()
    middle = np.interp(xx[0], line[:, 0], line[:, 1])
    return (yy - middle[None, :]) / BRANCH_HALF


def soft(mask: np.ndarray, r: int) -> np.ndarray:
    """``mask`` as floats, blurred by about ``r`` pixels."""
    out = mask.astype(np.float32)
    for _ in range(2):
        out = box_blur(out, r)
    return out


def grow(mask: np.ndarray, r: int) -> np.ndarray:
    """``mask`` grown by ``r`` pixels."""
    return max_filter(mask.astype(np.float32), r) > 0.5


def lambert(height: np.ndarray) -> np.ndarray:
    """How squarely a surface of this height faces the light, 0..1."""
    gy, gx = np.gradient(height)
    n = np.stack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=0)
    return np.clip(np.tensordot(LIGHT, n, axes=1), 0.0, 1.0)


def dome(mask: np.ndarray, r: int, depth: float, power: float = 0.6) -> np.ndarray:
    """A rounded form rising from the edge of ``mask``, ``depth`` high."""
    return np.clip(soft(mask, r), 0.0, 1.0) ** power * depth


def stretch(tone: np.ndarray, where: np.ndarray, lo=0.04, hi=0.96) -> np.ndarray:
    """``tone`` spread so that over ``where`` it runs from ``lo`` to ``hi``."""
    a, b = np.percentile(tone[where], [3, 97])
    return np.clip(lo + (hi - lo) * (tone - a) / max(b - a, 1e-6), 0.0, 1.0)


def flow(anchors) -> np.ndarray:
    """A field of directions (H, W) in radians: the directions at the anchors
    ``(x, y, degrees)``, blended between them by inverse distance."""
    yy, xx = np.mgrid[0:HEIGHT:4, 0:WIDTH:4].astype(np.float64)
    cx = np.zeros_like(xx)
    cy = np.zeros_like(xx)
    for x, y, deg in anchors:
        w = 1.0 / ((xx - x) ** 2 + (yy - y) ** 2 + 400.0) ** 1.5
        cx += w * math.cos(math.radians(deg))
        cy += w * math.sin(math.radians(deg))
    small = np.arctan2(cy, cx)
    big = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
    return big[:HEIGHT, :WIDTH]


def band(x, lo: float, hi: float):
    """0 below ``lo``, 1 above ``hi``, smooth between (or the other way round
    if ``lo`` > ``hi``)."""
    return smoothstep(lo, hi, x)


def wing_frame() -> tuple[np.ndarray, np.ndarray]:
    """How far along the folded wing each pixel lies (0 at the shoulder, 1 at
    the tip), and how far across it in pixels (negative toward its leading
    edge, the breast)."""
    xx, yy = grid()
    a, b = np.array(WING_AXIS[0]), np.array(WING_AXIS[1])
    length = float(np.linalg.norm(b - a))
    ax, ay = (b - a) / length
    t = ((xx - a[0]) * ax + (yy - a[1]) * ay) / length
    across = (xx - a[0]) * ay - (yy - a[1]) * ax
    return t, across


def on_wing(t: float, across: float) -> tuple[float, float]:
    """The point of the sheet at ``t`` along the wing and ``across`` it."""
    a, b = np.array(WING_AXIS[0]), np.array(WING_AXIS[1])
    length = float(np.linalg.norm(b - a))
    ax, ay = (b - a) / length
    x = a[0] + t * length * ax + across * ay
    y = a[1] + t * length * ay - across * ax
    return float(x), float(y)


# ----------------------------------------------------------------------
# the hand


@dataclass
class Hand:
    """The sheet and the hand that draws on it."""

    sheet: Sheet
    rng: np.random.Generator
    box: dict[str, Medium] = field(default_factory=dict)

    def pencil(self, colour: str) -> Medium:
        """The pencil of that colour, from the box."""
        if colour not in self.box:
            self.box[colour] = pencils.coloured(colour)
        return self.box[colour]

    def seeds(self, weight: np.ndarray, count: int) -> np.ndarray:
        """``count`` points, more of them where ``weight`` is higher."""
        flat = np.clip(weight, 0.0, None).ravel().astype(np.float64)
        total = flat.sum()
        if total <= 0 or count <= 0:
            return np.zeros((0, 2))
        idx = self.rng.choice(flat.size, size=count, p=flat / total)
        y, x = np.divmod(idx, WIDTH)
        jitter = self.rng.uniform(0.0, 1.0, (count, 2))
        return np.stack([x, y], axis=1) + jitter

    def trace(self, start, angle: np.ndarray, steps: int, step: float, bend: float):
        """Paths (N, steps + 1, 2) from ``start`` along the field ``angle``,
        each with its own slight curve and waver."""
        n = len(start)
        path = np.empty((n, steps + 1, 2))
        path[:, 0] = start
        turn = self.rng.normal(0.0, bend, n)
        offset = self.rng.normal(0.0, 0.1, n)
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
        and kept inside ``clip``. ``cover`` is how many times over the
        strokes would cover the area at full weight."""
        length, width = how.get("length", 30.0), how.get("width", 4.0)
        step = how.get("step", 4.0)
        clip = how.get("clip", weight > 0.02)
        count = int(how.get("cover", 1.5) * float(weight.sum()) / (width * length))
        seed = self.seeds(weight, count)
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
        paths = self.trace(start, angle, steps, step, how.get("bend", 0.02))
        lengths = self.rng.lognormal(math.log(length), 0.35, len(seed))
        medium, shape = self.pencil(colour), how.get("shape", "flick")
        for path, ln, w0 in zip(paths, lengths, own, strict=True):
            path = inside_run(path[: min(max(int(ln / step), 1), steps) + 1], clip)
            if len(path) < 2:
                continue
            ix = np.clip(path[:, 0].astype(int), 0, WIDTH - 1)
            iy = np.clip(path[:, 1].astype(int), 0, HEIGHT - 1)
            local = np.clip(weight[iy, ix], 0.0, 1.0)
            u = np.linspace(0.0, 1.0, len(path))
            bear = firm * np.sqrt(0.5 * w0 + 0.5 * local) * envelope(shape, u)
            bear *= self.rng.uniform(0.8, 1.15)
            self.sheet.stroke(path.tolist(), medium, bear.tolist(), width)

    def burnish(self, weight: np.ndarray, angle: np.ndarray, count: int) -> None:
        """Short firm strokes of the colourless blender where ``weight`` is,
        along ``angle``: they press the colour into the tooth and polish it."""
        for x, y in self.seeds(weight, count):
            a = float(angle[int(y), int(x)])
            d = np.array([math.cos(a), math.sin(a)]) * self.rng.uniform(5.0, 10.0)
            path = [(x - d[0], y - d[1]), (x, y), (x + d[0], y + d[1])]
            self.sheet.stroke(path, pencils.BLENDER, [0.3, 0.6, 0.3], 5.0)

    def line(self, colour: str, points, pressure, width=2.5, wobble=0.5, within=None):
        """One stroke through ``points``, as a hand follows them; lifted
        wherever it leaves ``within``, if given."""
        pts = np.asarray(points, dtype=np.float64)
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


def envelope(shape: str, u: np.ndarray) -> np.ndarray:
    """How the pressure runs along a stroke: ``flick`` lands firmly and lifts
    away, ``swell`` comes and goes, ``even`` keeps on."""
    if shape == "flick":
        return np.minimum(u * 5.0 + 0.35, 1.0) * (1.0 - 0.8 * u**1.6)
    if shape == "swell":
        return 0.25 + 0.75 * np.sin(np.pi * u) ** 0.7
    return np.minimum(u * 6.0 + 0.4, 1.0) * np.minimum((1.0 - u) * 6.0 + 0.4, 1.0)


# ----------------------------------------------------------------------
# the bird


@dataclass
class Bird:
    """Where each part of the bird is seen (``part``), where its strokes may
    run (``clip``), how strongly it is drawn at its edge (``edge``: lost
    where the light takes the silhouette), and how dark it falls (``shade``:
    0 in the full light to 1 in the deepest shadow)."""

    part: dict[str, np.ndarray]
    clip: dict[str, np.ndarray]
    edge: np.ndarray
    shade: dict[str, np.ndarray]
    mottle: np.ndarray


def parts() -> dict[str, np.ndarray]:
    """The parts of the bird, each where it is seen in front of the others."""
    whole = fill(HEAD_BODY)
    v = across_branch()
    p = {
        "whole": whole,
        "skull": fill(HEAD),
        "wing_all": fill(WING),
        "tail_all": fill(TAIL),
        "face": fill(FACE),
        "brow": fill(BROW),
        "upper": fill(UPPER_BILL),
        "lower": fill(LOWER_BILL),
        "branch": np.abs(v) < 1.0,
    }
    bills = p["upper"] | p["lower"]
    p["lower"] = p["lower"] & ~p["upper"]
    p["brow"] = p["brow"] & ~bills
    p["face"] = p["face"] & ~bills & ~p["brow"]
    front = bills | p["face"] | p["brow"]
    p["head"] = p["skull"] & whole & ~front
    p["wing"] = p["wing_all"] & ~p["branch"]
    p["body"] = whole & ~p["head"] & ~p["wing_all"] & ~front & ~p["branch"]
    p["tail"] = p["tail_all"] & ~whole & ~p["wing_all"] & ~p["branch"]
    return p


def shading(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """How dark each part falls: its own roundness under the light, the fall
    of the light down and across the bird, and the shadow of what stands
    over it."""
    xx, yy = grid()
    fall = 0.55 * band(yy, 250, 1100) + 0.25 * band(xx, 430, 800)
    height = dome(p["whole"], 60, 200.0) + dome(p["skull"], 40, 60.0, 1.0)
    contact = soft(p["wing_all"], 12) * ~p["wing_all"]
    chin = soft(p["lower"] | p["face"], 16) * (~p["lower"] & ~p["face"])
    body = 0.75 * (1.0 - lambert(height)) + fall + 0.6 * contact + 0.35 * chin
    wing = 0.8 * (1.0 - lambert(dome(p["wing_all"], 30, 90.0) + 0.5 * height))
    wing = wing + fall
    tail = 1.0 - lambert(dome(p["tail_all"], 16, 30.0))
    tail = tail + 0.8 * soft(p["whole"] | p["wing_all"], 22) + 0.3 * band(yy, 1300, 1050)
    bills = p["upper"] | p["lower"]
    bill = 1.0 - lambert(dome(bills, 16, 60.0))
    return {
        "body": stretch(body, p["body"] | p["head"]),
        "wing": stretch(wing, p["wing"]),
        "tail": stretch(tail, p["tail"]),
        "bill": stretch(bill, bills, 0.0, 1.0),
    }


def build(rng: np.random.Generator) -> Bird:
    """The bird as it is to be drawn."""
    p = parts()
    silhouette = p["whole"] | p["wing_all"] | p["tail_all"]
    seen = silhouette & ~p["branch"]
    edge = soft(seen, 3) * seen
    names = ("body", "head", "wing", "tail", "face", "brow", "upper", "lower")
    clip = {name: grow(p[name], 3) & seen for name in names}
    mottle = noise.fractal_noise_2d(HEIGHT, WIDTH, 40.0, rng, octaves=4)
    mottle = (mottle - mottle.min()) / (np.ptp(mottle) + 1e-9)
    return Bird(p, clip, edge, shading(p), mottle)


# ----------------------------------------------------------------------
# the drawing

LEMON, CHARTREUSE, OLIVE, MOSS = "#ebe46e", "#c3cf48", "#90a83c", "#5f8c36"
LEAF, PINE, DEEP, OCHRE = "#3f7c3c", "#2b5c42", "#203c46", "#c49a40"
PALE_TURQ, TURQ, COBALT = "#92d5d6", "#3fb2c2", "#2c6fb8"
ULTRA, NAVY = "#2c3f94", "#1e2650"
ORANGE, VERMILION, MADDER, OXBLOOD = "#e8873a", "#cf4a2c", "#9a2a26", "#5c201e"
SKY = "#8cc2e0"
BLUSH, ROSE, MAUVE, LINE, WINE = "#f2d0c8", "#e3a19f", "#b8949e", "#3e1a20", "#74262e"
SCARLET, CRIMSON, FLAME = "#e3352c", "#a91d26", "#f06a3a"
SILVER, GREY, SLATE, CHAR = "#a8a8ae", "#77777f", "#4a4d58", "#232328"
BLUEGREY, HORN, STRAW = "#64708a", "#bdb6aa", "#ece2a2"
UMBER, SEPIA, WARM_GREY = "#7a6450", "#4e3c30", "#a59a8c"

BODY_FLOW = [
    (420, 470, 80), (430, 600, 76), (480, 780, 62), (570, 940, 40),
    (700, 600, 70), (780, 900, 78), (600, 360, 62), (560, 520, 72),
]  # fmt: skip
HEAD_FLOW = [
    (372, 210, -30), (450, 182, 2), (530, 212, 35), (578, 290, 72),
    (430, 440, 62), (385, 425, 82), (510, 330, 45), (500, 260, 25),
    (540, 400, 70),
]  # fmt: skip
WING_FLOW = [
    (540, 450, 40), (610, 430, 60), (600, 560, 66), (680, 650, 72),
    (700, 900, 75), (770, 1150, 80), (560, 600, 62),
]  # fmt: skip
TAIL_FLOW = [(730, 1030, 50), (880, 1200, 54), (1010, 1400, 58)]
BILL_FLOW = [
    (355, 242, 182), (300, 248, 160), (264, 300, 112), (256, 370, 88),
    (280, 430, 62), (330, 380, 200), (360, 300, 120),
]  # fmt: skip
LOWER_FLOW = [(390, 380, 100), (360, 450, 160), (315, 440, 230)]

#: Layers of colour, each a pencil, where on the scale of shade it goes on
#: (rising from a to b, falling from c to d), how firmly, and how many times
#: over it covers; laid in this order, the darks first, while the tooth
#: still takes them.
GREENS = [
    (DEEP, (0.72, 0.9, 1.1, 1.2), 0.6, 1.3),
    (PINE, (0.55, 0.75, 1.1, 1.2), 0.56, 1.4),
    (LEAF, (0.4, 0.6, 0.85, 1.0), 0.52, 1.5),
    (MOSS, (0.25, 0.45, 0.7, 0.9), 0.5, 1.6),
    (OLIVE, (0.08, 0.3, 0.55, 0.75), 0.46, 1.6),
    (CHARTREUSE, (-0.1, 0.1, 0.4, 0.6), 0.42, 1.5),
    (LEMON, (-0.2, -0.1, 0.25, 0.45), 0.36, 1.3),
]
BLUES = [
    (NAVY, (0.7, 0.9, 1.1, 1.2), 0.6, 1.2),
    (ULTRA, (0.45, 0.7, 1.1, 1.2), 0.55, 1.4),
    (COBALT, (0.15, 0.4, 0.75, 0.95), 0.5, 1.6),
    (TURQ, (-0.2, -0.1, 0.4, 0.65), 0.45, 1.5),
]
RUSTS = [
    (OXBLOOD, (0.7, 0.9, 1.1, 1.2), 0.55, 1.1),
    (MADDER, (0.4, 0.65, 1.1, 1.2), 0.52, 1.4),
    (VERMILION, (0.0, 0.25, 0.75, 0.95), 0.5, 1.6),
    (ORANGE, (-0.2, -0.1, 0.3, 0.55), 0.42, 1.3),
]
INKS = [
    (CHAR, (0.3, 0.6, 1.1, 1.2), 0.66, 1.8),
    (SLATE, (0.1, 0.3, 0.75, 0.95), 0.56, 1.8),
    (GREY, (-0.1, 0.05, 0.4, 0.6), 0.48, 1.6),
    (SILVER, (-0.2, -0.1, 0.15, 0.3), 0.36, 1.2),
]


def trapezoid(s, edges) -> np.ndarray:
    """1 between ``b`` and ``c`` of ``edges`` (a, b, c, d), 0 outside ``a``
    and ``d``, smooth between."""
    a, b, c, d = edges
    return band(s, a, b) * (1.0 - band(s, c, d))


def build_up(hand: Hand, layers, where, s, angle, **how) -> None:
    """Lay ``layers`` over ``where``, each where the shade ``s`` calls for it."""
    vary = how.pop("vary", None)
    for colour, edges, firm, cover in layers:
        w = where * trapezoid(s, edges)
        if vary is not None:
            w = w * vary
        hand.layer(colour, w, angle, firm, cover=cover, **how)


def draw_sketch(hand: Hand) -> None:
    """The first lines, light and searching, which the colour covers."""
    for points in (HEAD_BODY, WING, TAIL, UPPER_BILL, LOWER_BILL, FACE):
        curve = outline(points, 6.0)
        n = len(curve)
        start = 0
        while start < n - 4:
            run = int(hand.rng.integers(12, 40))
            seg = curve[start : min(start + run, n)]
            if len(seg) > 2 and hand.rng.random() < 0.8:
                hand.line("#c4c2ae", seg, [0.1, 0.18, 0.14], 2.2, 0.8)
            start += run - int(hand.rng.integers(0, 4))


def draw_plumage(hand: Hand, bird: Bird) -> None:
    """The green of the head, breast, belly and back, along the lie of the
    feathers: fine on the head, broader on the body."""
    s, vary = bird.shade["body"], 0.75 + 0.5 * bird.mottle
    clip = bird.clip["body"] | bird.clip["head"]
    for name, anchors, fine in (("body", BODY_FLOW, 1.0), ("head", HEAD_FLOW, 0.55)):
        where = bird.part[name] * bird.edge
        ln, wd = 24.0 * fine, max(4.5 * fine, 3.4)
        angle = flow(anchors)
        build_up(hand, GREENS, where, s, angle, vary=vary, length=ln, width=wd, clip=clip)


def wing_tips(across: np.ndarray, t0: float, size: float, depth: float) -> np.ndarray:
    """Where along the wing a row of feather tips ends, at each ``across``:
    a scallop of rounded tips ``size`` wide, ``depth`` deep (in t)."""
    phase = (across / size) % 1.0
    return t0 + depth * np.sqrt(np.clip(1.0 - (2.0 * phase - 1.0) ** 2, 0.0, 1.0))


def draw_wing(hand: Hand, bird: Bird) -> None:
    """The folded wing: green coverts over blue flight feathers."""
    angle = flow(WING_FLOW)
    t, across = wing_frame()
    s = bird.shade["wing"]
    where = bird.part["wing"] * bird.edge
    clip = bird.clip["wing"]
    edge = wing_tips(across + 7.0, 0.405, 27.0, 0.035)
    green = where * (t < edge)
    blue = where * (t > edge - 0.012)
    vary = 0.75 + 0.5 * bird.mottle
    build_up(hand, GREENS[1:], green, s, angle, vary=vary, length=24, width=4.5, clip=clip)
    darker = np.clip(s + 0.3 * band(t, 0.6, 1.0), 0.0, 1.0)
    build_up(hand, BLUES, blue, darker, angle, length=50, width=4.5, clip=clip)
    draw_coverts(hand, bird, t, across)
    draw_flight_feathers(hand, bird)


def draw_coverts(hand: Hand, bird: Bird, t, across) -> None:
    """Rows of covert tips across the wing: found in the shade, lost in the
    light, and the shadow each row throws on the next."""
    s = bird.shade["wing"]
    wing = bird.part["wing"]
    for t0, size in ((0.17, 14.0), (0.25, 18.0), (0.33, 23.0), (0.405, 27.0)):
        a = -110.0 + hand.rng.uniform(0, size)
        while a < 110.0:
            tt = t0 + hand.rng.normal(0.0, 0.006)
            centre = on_wing(tt, a + size / 2)
            x, y = int(centre[0]), int(centre[1])
            inside = 0 <= y < HEIGHT and 0 <= x < WIDTH and wing[y, x]
            if inside and hand.rng.random() < 0.85:
                shade = float(s[y, x])
                arc = [on_wing(tt + 0.03 * math.sin(k / 6 * math.pi) * size / 27.0, a + k / 6 * size) for k in range(7)]
                firm = 0.22 + 0.45 * shade
                hand.line(PINE if t0 < 0.4 else DEEP, arc, [firm * 0.4, firm, firm, firm * 0.5], 2.6, 0.5)
                under = [(px + 1.5, py + 3.0) for px, py in arc[1:-1]]
                hand.line(DEEP, under, [0.0, firm * 0.6, firm * 0.6, 0.0], 4.0, 0.6)
            a += size * hand.rng.uniform(0.9, 1.1)
    del t, across


def draw_flight_feathers(hand: Hand, bird: Bird) -> None:
    """The edges of the folded flight feathers, converging on the wing's
    tip, each with its pale outer edge beside its dark shadow."""
    s = bird.shade["wing"]
    for k, a0 in enumerate(np.linspace(-62.0, 58.0, 7)):
        end = 0.86 + 0.13 * (1.0 - abs(a0 - 30.0) / 95.0) + hand.rng.normal(0, 0.01)
        ts = np.linspace(0.45 + hand.rng.uniform(0, 0.04), end, 24)
        pts = [on_wing(float(tt), a0 * (1.0 - 0.85 * (tt - 0.45) / 0.55) + 6.0) for tt in ts]
        x, y = (int(c) for c in pts[len(pts) // 2])
        firm = 0.3 + 0.35 * float(s[min(y, HEIGHT - 1), min(x, WIDTH - 1)])
        bear = (firm * (0.5 + 0.5 * np.sin(np.linspace(0.2, 2.9, 24)))).tolist()
        wing = bird.part["wing"]
        hand.line(NAVY, pts, bear, 2.6, 0.4, within=wing)
        light = [(px - 3.0, py + 1.0) for px, py in pts]
        hand.line(PALE_TURQ if k % 2 else TURQ, light, 0.3, 2.4, 0.4, within=wing)


def draw_back(hand: Hand, bird: Bird) -> None:
    """The lower back and rump, showing turquoise past the folded wing."""
    _, yy = grid()
    back = bird.part["body"] * bird.edge * band(yy, 820, 930) * (yy < 1000)
    angle = flow(BODY_FLOW)
    hand.layer(TURQ, back, angle, 0.45, length=20, cover=1.4, clip=bird.clip["body"])
    hand.layer(PALE_TURQ, back, angle, 0.35, length=16, cover=1.0, clip=bird.clip["body"])


def draw_tail(hand: Hand, bird: Bird) -> None:
    """The long tail, rust above, its outer feathers edged blue, loosening
    and paling toward its blue tip."""
    angle = flow(TAIL_FLOW)
    s = bird.shade["tail"]
    xx, yy = grid()
    fade = 1.0 - 0.55 * band(yy, 1150, 1440)
    tip = band(yy, 1320, 1400)
    w = bird.part["tail"] * bird.edge * fade
    clip = bird.clip["tail"]
    build_up(hand, RUSTS, w * (1.0 - tip), s, angle, length=60, width=5, clip=clip)
    hand.layer(SKY, w * tip, angle, 0.4, length=50, width=5, cover=1.3, clip=clip)
    hand.layer(COBALT, w * tip * band(s, 0.3, 0.8), angle, 0.4, length=40, clip=clip)
    # the outer feathers' blue edges, along both sides
    side = soft(bird.part["tail_all"], 6)
    rim = w * band(side, 0.75, 0.35) * (1.0 - tip)
    hand.layer(COBALT, rim, angle, 0.42, length=50, width=4, cover=1.2, clip=clip)
    del xx


def draw_face(hand: Hand, bird: Bird) -> None:
    """The bare face, pale and flushed, crossed by lines of tiny feathers."""
    face = bird.part["face"]
    xx, yy = grid()
    angle = np.full((HEIGHT, WIDTH), math.radians(-10.0))
    w = soft(face, 4) * face
    clip = bird.clip["face"]
    rim = band(soft(face, 10), 0.9, 0.4)
    flush = (0.3 + 0.7 * band(yy, 300, 390)) * (0.4 + 0.6 * rim)
    shadow = 0.5 * band(xx, 440, 485) + 0.6 * band(yy, 270, 245)
    hand.layer(MAUVE, w * shadow, angle, 0.24, length=12, cover=0.8, shape="swell", clip=clip)
    hand.layer(ROSE, w * flush, angle, 0.28, length=12, cover=1.0, shape="swell", clip=clip)
    hand.layer(BLUSH, w, angle, 0.3, length=14, cover=1.2, shape="swell", clip=clip)
    draw_face_lines(hand, face)


def draw_face_lines(hand: Hand, face: np.ndarray) -> None:
    """Rows of tiny dark feathers across the bare skin, following its shape
    from the brow down to the jaw: each a fine line of tiny tufts, beaded
    and broken, darker in some rows than in others."""
    cols = np.nonzero(face.any(axis=0))[0]
    top = np.array([np.argmax(face[:, c]) for c in cols], dtype=np.float64)
    bottom = np.array([HEIGHT - 1 - np.argmax(face[::-1, c]) for c in cols], dtype=np.float64)
    ex, ey = EYE
    for v in np.linspace(0.12, 0.92, 10) + hand.rng.normal(0.0, 0.012, 10):
        dark = hand.rng.uniform(0.7, 1.1)
        x = float(cols[0]) + hand.rng.uniform(4.0, 10.0)
        while x < cols[-1] - 6:
            dash = hand.rng.uniform(4.0, 9.0)
            xs = np.array([x, x + dash / 2, x + dash])
            ys = np.interp(xs, cols, top + v * (bottom - top))
            pts = np.stack([xs, ys], 1)
            far = math.hypot(x - ex, ys[0] - ey) > IRIS + 10
            if far and face[int(ys[1]), int(xs[1])]:
                colour = LINE if hand.rng.random() < 0.6 else WINE
                firm = dark * hand.rng.uniform(0.3, 0.6)
                hand.line(colour, pts, [firm * 0.7, firm, firm * 0.6], 2.2, 0.35, face)
            x += dash + hand.rng.uniform(0.5, 2.5)


def circle(centre, radius: float, n: int = 36, start: float = 0.0, turn=2 * math.pi):
    """Points round a circle (or an arc of it)."""
    a = np.linspace(start, start + turn, n)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)], 1)


def draw_eye(hand: Hand) -> None:
    """The pale eye in its ring of skin, its black pupil, the shadow of the
    lid across the top of it and the glint of the light."""
    hand.line(MAUVE, circle(EYE, IRIS + 5.0), 0.38, 5.0, 0.3)
    hand.line(ROSE, circle(EYE, IRIS + 8.5, 40, math.pi * 0.1, math.pi * 0.8), 0.3, 4.0, 0.4)
    for r in np.arange(1.5, IRIS, 2.0):
        hand.line(STRAW, circle(EYE, r, 30), 0.5, 3.0, 0.2)
    lid = circle(EYE, IRIS - 3.0, 20, math.pi * 1.1, math.pi * 0.8)
    hand.line(OCHRE, lid, [0.2, 0.5, 0.5, 0.2], 5.0, 0.2)
    rim = circle(EYE, IRIS, 46, -0.2)
    hand.line(LINE, rim, [0.45, 0.65, 0.8, 0.75, 0.5, 0.4, 0.5], 2.6, 0.2)
    upper = circle(EYE, IRIS + 1.5, 24, math.pi * 1.05, math.pi * 0.9)
    hand.line(LINE, upper, [0.3, 0.7, 0.7, 0.3], 3.2, 0.2)
    for r in np.arange(0.5, PUPIL + 0.5, 1.2):
        hand.line(CHAR, circle(EYE, r, 24), 0.85, 2.5, 0.1)
    glint = (EYE[0] - 4.5, EYE[1] - 5.0)
    hand.sheet.erase([glint, (glint[0] + 0.8, glint[1] + 0.5)], tools.ERASER, 1.0, 4.0)


def draw_brow(hand: Hand, bird: Bird) -> None:
    """The scarlet band over the bill: its small feathers lie back over the
    forehead, and its edge breaks into the green of the crown."""
    angle = flow([(350, 235, -60), (380, 205, -30), (410, 195, -10)])
    brow = bird.part["brow"]
    w = soft(brow, 5) * grow(brow, 3)
    clip = grow(brow, 5) & ~bird.part["upper"] & ~bird.part["face"]
    _, yy = grid()
    low = w * band(yy, 210, 245)
    hand.layer(CRIMSON, low, angle, 0.5, length=9, width=3, cover=1.4, clip=clip)
    hand.layer(SCARLET, w, angle, 0.55, length=10, width=3.5, cover=2.2, clip=clip)
    hand.layer(FLAME, w * band(yy, 225, 195), angle, 0.4, length=10, width=3.5, cover=1.0, clip=clip)


def draw_bill(hand: Hand, bird: Bird) -> None:
    """The great hooked bill, blackish grey, paler toward the tip of the
    hook, burnished to its horny sheen."""
    s = bird.shade["bill"]
    upper, lower = bird.part["upper"], bird.part["lower"]
    xx, yy = grid()
    tip = band(yy, 395, 448) * band(xx, 318, 285)
    for mask, anchors in ((lower, LOWER_FLOW), (upper, BILL_FLOW)):
        angle = flow(anchors)
        w = soft(mask, 2) * mask * (1.0 - 0.6 * tip)
        build_up(hand, INKS, w, s, angle, length=9, width=4, step=3.0, clip=mask)
        hand.layer(HORN, soft(mask, 2) * mask * tip, angle, 0.34, length=12, clip=mask)
        hand.layer(BLUEGREY, w * trapezoid(s, (0.3, 0.5, 0.7, 0.85)), angle, 0.3, length=12, clip=mask)
        hand.burnish(w, angle, 900)
    draw_bill_edges(hand)


def draw_bill_edges(hand: Hand) -> None:
    """The cutting edge where the hook closes over the lower bill, the
    nostril's fold at the base, and the light along the ridge."""
    tomium = [(299, 444), (303, 425), (311, 401), (331, 380), (358, 368), (382, 364)]
    hand.line(CHAR, tomium, [0.6, 0.85, 0.85, 0.8, 0.7, 0.5], 3.0, 0.3)
    base = [(360, 240), (366, 270), (372, 305), (378, 340), (381, 362)]
    hand.line(CHAR, base, [0.3, 0.6, 0.6, 0.5, 0.4], 2.6, 0.4)
    ridge = [(352, 243), (322, 246), (296, 258), (276, 280), (264, 310)]
    hand.sheet.erase(ridge, tools.ERASER, [0.0, 0.6, 0.7, 0.5, 0.0], 4.0)
    hook = [(259, 345), (262, 380), (272, 410)]
    hand.sheet.erase(hook, tools.KNEADED, [0.0, 0.6, 0.0], 6.0)


def draw_branch(hand: Hand, bird: Bird) -> None:
    """A hint of the branch: firm by the feet, fading away either side."""
    v = across_branch()
    xx, _ = grid()
    near = np.exp(-(((xx - 610.0) / 240.0) ** 2))
    w = bird.part["branch"] * near
    angle = np.full((HEIGHT, WIDTH), math.radians(-9.0))
    hand.layer(SEPIA, w * band(v, 0.4, 0.95), angle, 0.45, length=30, width=4, cover=1.0)
    hand.layer(UMBER, w * band(v, -0.2, 0.7), angle, 0.4, length=36, width=4.5, cover=1.2)
    hand.layer(WARM_GREY, w, angle, 0.3, length=40, width=5, cover=1.0)


def draw_feet(hand: Hand) -> None:
    """The dark grey toes over the branch, two forward, and their claws."""
    for toe in TOES:
        pts = CatmullRom(np.asarray(toe, dtype=np.float64)).sample(2.0)[0]
        for colour, off, wd, firm in ((SLATE, 2.5, 8.0, 0.6), (GREY, -1.0, 12.0, 0.5)):
            shifted = pts + np.array([off, off * 0.5])
            hand.line(colour, shifted, firm, wd, 0.3)
        for k in range(2, len(pts) - 3, 4):
            p, q = pts[k], pts[k + 1]
            d = (q - p) / max(float(np.linalg.norm(q - p)), 1e-6)
            n = np.array([-d[1], d[0]])
            hand.line(CHAR, [p - 4 * n, p + 4 * n], 0.4, 1.6, 0.2)
        end, d = pts[-1], pts[-1] - pts[-3]
        d = d / max(float(np.linalg.norm(d)), 1e-6)
        claw = [end, end + 6 * d, end + 9 * d + np.array([-d[1], d[0]]) * 3]
        hand.line(CHAR, claw, [0.75, 0.65, 0.3], 3.2, 0.2)


def main() -> None:
    rng = np.random.default_rng(SEED)
    paper = Paper("wove", grain=2.0, depth=0.4, bite=0.5, give=0.4, color="#fbfaf6")
    sheet = Sheet(WIDTH, HEIGHT, paper=paper, seed=SEED)
    hand = Hand(sheet, rng)
    bird = build(rng)
    draw_sketch(hand)
    draw_tail(hand, bird)
    draw_plumage(hand, bird)
    draw_back(hand, bird)
    draw_wing(hand, bird)
    draw_face(hand, bird)
    draw_brow(hand, bird)
    draw_bill(hand, bird)
    draw_eye(hand)
    draw_branch(hand, bird)
    draw_feet(hand)
    out = Path(__file__).parent / "output" / "macaw.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
