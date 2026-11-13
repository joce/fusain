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
    (560, 235), (585, 290), (605, 345), (640, 410), (690, 500), (735, 610),
    (772, 740), (798, 870), (808, 960), (795, 1015), (745, 1035), (690, 1035),
    (630, 1012), (585, 968), (535, 900), (488, 820), (448, 730), (420, 640),
    (402, 560), (395, 500), (390, 452), (378, 420), (366, 370), (356, 300),
]  # fmt: skip
HEAD = [
    (352, 236), (372, 205), (405, 183), (450, 173), (497, 180), (535, 202),
    (560, 235), (582, 290), (590, 350), (560, 410), (500, 450), (440, 470),
    (396, 452), (380, 420), (366, 370), (356, 300),
]  # fmt: skip
WING = [
    (500, 472), (528, 428), (575, 402), (625, 414), (668, 468), (705, 568),
    (735, 700), (757, 840), (772, 980), (787, 1110), (795, 1200), (777, 1184),
    (745, 1084), (702, 962), (648, 852), (594, 752), (548, 652), (512, 568),
]  # fmt: skip
TAIL = [
    (700, 1010), (760, 995), (820, 1062), (895, 1180), (970, 1305), (1030, 1410),
    (1040, 1440), (1010, 1436), (940, 1345), (860, 1240), (775, 1135), (712, 1065),
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
IRIS, PUPIL = 12.5, 5.5
#: The folded wing's axis, from the shoulder to the tip.
WING_AXIS = ((560.0, 420.0), (795.0, 1200.0))
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
        start = self.seeds(weight, count)
        steps = max(int(length * 2.0 / step), 2)
        paths = self.trace(start, angle, steps, step, how.get("bend", 0.02))
        lengths = self.rng.lognormal(math.log(length), 0.35, len(start))
        medium, shape = self.pencil(colour), how.get("shape", "flick")
        for path, ln in zip(paths, lengths, strict=True):
            path = path[: min(max(int(ln / step), 1), steps) + 1]
            ix = np.clip(path[:, 0].astype(int), 0, WIDTH - 1)
            iy = np.clip(path[:, 1].astype(int), 0, HEIGHT - 1)
            out = np.nonzero(~clip[iy, ix])[0]
            if out.size:
                path, ix, iy = path[: out[0]], ix[: out[0]], iy[: out[0]]
            if len(path) < 2:
                continue
            u = np.linspace(0.0, 1.0, len(path))
            bear = firm * np.clip(weight[iy, ix], 0.0, 1.0) ** 0.5 * envelope(shape, u)
            bear *= self.rng.uniform(0.8, 1.15)
            self.sheet.stroke(path.tolist(), medium, bear.tolist(), width)

    def line(self, colour: str, points, pressure, width: float = 2.5, wobble=0.5):
        """One stroke through ``points``, as a hand follows them."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            return
        pts = pts + self.rng.normal(0.0, wobble, pts.shape)
        self.sheet.stroke(pts.tolist(), self.pencil(colour), pressure, width)


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

#: The layers of green plumage, light to dark: the pencil, where on the
#: scale of shade it goes on (from, full), how firmly, and how many times
#: over it covers.
GREENS = [
    (LEMON, (1.4, -0.1), 0.34, 1.4),
    (CHARTREUSE, (-0.15, 0.25), 0.4, 1.6),
    (OLIVE, (0.05, 0.4), 0.44, 1.7),
    (MOSS, (0.25, 0.55), 0.48, 1.7),
    (LEAF, (0.4, 0.7), 0.5, 1.6),
    (PINE, (0.55, 0.85), 0.54, 1.4),
    (DEEP, (0.72, 1.0), 0.58, 1.2),
]


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


def draw_greens(hand: Hand, bird: Bird, name: str, angle, fine: float, s=None):
    """Green plumage, built light to dark along the lie of the feathers;
    ``fine`` scales the strokes (1 for the body, smaller for the head)."""
    s = bird.shade["body"] if s is None else s
    where = bird.part[name] * bird.edge
    clip = bird.clip[name]
    ln, wd = 34.0 * fine, 4.5 * max(fine, 0.75)
    for colour, (lo, hi), firm, cover in GREENS[::-1]:
        w = where * band(s, lo, hi) * (0.75 + 0.5 * bird.mottle)
        hand.layer(colour, w, angle, firm, length=ln, width=wd, cover=cover, clip=clip)


def draw_body(hand: Hand, bird: Bird) -> None:
    """The breast, belly and back."""
    draw_greens(hand, bird, "body", flow(BODY_FLOW), 1.0)


def draw_head(hand: Hand, bird: Bird) -> None:
    """The feathered crown, nape and cheek."""
    draw_greens(hand, bird, "head", flow(HEAD_FLOW), 0.55)


def draw_wing(hand: Hand, bird: Bird) -> None:
    """The folded wing: green coverts over blue flight feathers."""
    angle = flow(WING_FLOW)
    t, _ = wing_frame()
    s = bird.shade["wing"]
    where = bird.part["wing"] * bird.edge
    clip = bird.clip["wing"]
    green = where * (1.0 - band(t, 0.4, 0.46))
    for colour, (lo, hi), firm, cover in GREENS[-2::-1]:
        w = green * band(s, lo, hi) * (0.75 + 0.5 * bird.mottle)
        hand.layer(colour, w, angle, firm, length=24, width=4.5, cover=cover, clip=clip)
    blue = where * band(t, 0.38, 0.44)
    blues = [
        (TURQ, (1.0, 0.2), 0.4, 1.6),
        (COBALT, (0.1, 0.5), 0.42, 1.6),
        (ULTRA, (0.4, 0.8), 0.45, 1.4),
        (NAVY, (0.7, 1.0), 0.5, 1.1),
    ]
    for colour, (lo, hi), firm, cover in blues[::-1]:
        w = blue * band(s + 0.25 * band(t, 0.6, 1.0), lo, hi)
        hand.layer(colour, w, angle, firm, length=50, width=4.5, cover=cover, clip=clip)


def draw_tail(hand: Hand, bird: Bird) -> None:
    """The long tail, rust above, loosening toward its end."""
    angle = flow(TAIL_FLOW)
    s = bird.shade["tail"]
    _, yy = grid()
    fade = 1.0 - 0.55 * band(yy, 1150, 1440)
    tip = band(yy, 1320, 1400)
    w = bird.part["tail"] * bird.edge * fade
    clip = bird.clip["tail"]
    rust = [
        (ORANGE, (1.0, 0.3), 0.38, 1.4),
        (VERMILION, (0.0, 0.4), 0.45, 1.6),
        (MADDER, (0.4, 0.8), 0.48, 1.4),
        (OXBLOOD, (0.7, 1.0), 0.5, 1.0),
    ]
    for colour, (lo, hi), firm, cover in rust[::-1]:
        layer = w * band(s, lo, hi) * (1.0 - tip)
        hand.layer(colour, layer, angle, firm, length=60, width=5, cover=cover, clip=clip)
    hand.layer(SKY, w * tip, angle, 0.38, length=50, width=5, cover=1.3, clip=clip)
    hand.layer(COBALT, w * tip * band(s, 0.3, 0.8), angle, 0.38, length=40, clip=clip)


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
    hand.layer(BLUSH, w, angle, 0.3, length=14, cover=1.4, shape="swell", clip=clip)
    hand.layer(ROSE, w * flush, angle, 0.26, length=12, cover=1.0, shape="swell", clip=clip)
    hand.layer(MAUVE, w * shadow, angle, 0.22, length=12, cover=0.8, shape="swell", clip=clip)
    draw_face_lines(hand, face)


def draw_face_lines(hand: Hand, face: np.ndarray) -> None:
    """Rows of tiny dark feathers across the bare skin, bowing under the eye."""
    ex, ey = EYE
    centre = (ex + 25.0, ey - 140.0)
    for radius in np.arange(118.0, 260.0, 13.0):
        radius += hand.rng.normal(0.0, 1.5)
        a = math.radians(150.0) + hand.rng.uniform(0.0, 0.02)
        while a > math.radians(30.0):
            arc = hand.rng.uniform(3.0, 7.0) / radius
            x0, y0 = centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a)
            x1 = centre[0] + radius * math.cos(a - arc)
            y1 = centre[1] + radius * math.sin(a - arc)
            ok = all(
                0 <= int(y) < HEIGHT and 0 <= int(x) < WIDTH and face[int(y), int(x)]
                for x, y in ((x0, y0), (x1, y1))
            )
            if ok and math.hypot(x0 - ex, y0 - ey) > IRIS + 8:
                colour = LINE if hand.rng.random() < 0.65 else WINE
                firm = hand.rng.uniform(0.35, 0.7)
                hand.line(colour, [(x0, y0), (x1, y1)], [firm, firm * 0.6], 2.4, 0.4)
            a -= arc + hand.rng.uniform(1.0, 3.5) / radius


def circle(centre, radius: float, n: int = 36, start: float = 0.0, turn=2 * math.pi):
    """Points round a circle (or an arc of it)."""
    a = np.linspace(start, start + turn, n)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)], 1)


def draw_eye(hand: Hand) -> None:
    """The pale eye, its black pupil and the glint of the light in it."""
    hand.line(MAUVE, circle(EYE, IRIS + 4.5), 0.35, 5.0, 0.3)
    for r in np.arange(1.5, IRIS, 2.0):
        hand.line(STRAW, circle(EYE, r, 30), 0.45, 3.0, 0.2)
    for a0 in np.linspace(math.pi * 1.05, math.pi * 1.95, 7):
        spoke = [EYE + np.array([math.cos(a0), math.sin(a0)]) * r for r in (IRIS - 1, PUPIL + 1)]
        hand.line(OCHRE, spoke, [0.45, 0.2], 2.0, 0.3)
    rim = circle(EYE, IRIS, 46, -0.2)
    hand.line(LINE, rim, [0.5, 0.7, 0.75, 0.6, 0.5, 0.45, 0.55], 2.4, 0.2)
    for r in np.arange(0.5, PUPIL + 0.5, 1.2):
        hand.line(CHAR, circle(EYE, r, 24), 0.8, 2.5, 0.1)
    glint = (EYE[0] - 4.0, EYE[1] - 4.5)
    hand.sheet.erase([glint, (glint[0] + 0.6, glint[1] + 0.4)], tools.ERASER, 0.9, 3.5)


def draw_brow(hand: Hand, bird: Bird) -> None:
    """The scarlet band over the bill."""
    angle = flow([(350, 235, -60), (380, 205, -30), (410, 195, -10)])
    w = soft(bird.part["brow"], 3) * bird.part["brow"]
    clip = bird.clip["brow"]
    _, yy = grid()
    hand.layer(FLAME, w, angle, 0.35, length=10, width=3.5, cover=1.4, clip=clip)
    hand.layer(SCARLET, w, angle, 0.5, length=10, width=3.5, cover=2.0, clip=clip)
    low = w * band(yy, 215, 245)
    hand.layer(CRIMSON, low, angle, 0.45, length=8, width=3, cover=1.2, clip=clip)


def draw_bill(hand: Hand, bird: Bird) -> None:
    """The great hooked bill, dark grey, paler toward its tip."""
    s = bird.shade["bill"]
    upper, lower = bird.part["upper"], bird.part["lower"]
    xx, yy = grid()
    tip = band(yy, 390, 445) * band(xx, 320, 285)
    ink = [
        (SILVER, (-0.2, 0.1), 0.35),
        (GREY, (0.0, 0.2), 0.45),
        (SLATE, (0.12, 0.4), 0.52),
        (CHAR, (0.3, 0.7), 0.62),
    ]
    for mask, anchors in ((lower, LOWER_FLOW), (upper, BILL_FLOW)):
        angle = flow(anchors)
        w = soft(mask, 2) * mask
        for colour, (lo, hi), firm in ink[::-1]:
            layer = w * band(s, lo, hi) * (1.0 - 0.6 * tip)
            hand.layer(colour, layer, angle, firm, length=18, cover=1.8, bend=0.03)
    hand.layer(HORN, soft(upper, 2) * upper * tip, flow(BILL_FLOW), 0.3, length=14)


def draw_branch(hand: Hand, bird: Bird) -> None:
    """A hint of the branch: firm by the feet, fading away either side."""
    v = across_branch()
    xx, _ = grid()
    near = np.exp(-(((xx - 610.0) / 240.0) ** 2))
    on = bird.part["branch"] & ~grow(bird.part["whole"] & ~bird.part["branch"], 0)
    w = on * near
    angle = np.full((HEIGHT, WIDTH), math.radians(-9.0))
    hand.layer(WARM_GREY, w, angle, 0.3, length=40, width=5, cover=1.0)
    hand.layer(UMBER, w * band(v, -0.2, 0.7), angle, 0.4, length=36, width=4.5, cover=1.2)
    hand.layer(SEPIA, w * band(v, 0.4, 0.95), angle, 0.45, length=30, width=4, cover=1.0)


def draw_feet(hand: Hand) -> None:
    """The dark grey toes over the branch, two forward, and their claws."""
    for toe in TOES:
        pts = CatmullRom(np.asarray(toe, dtype=np.float64)).sample(2.0)[0]
        for colour, off, wd, firm in ((GREY, 0.0, 9.0, 0.5), (SLATE, 2.5, 5.0, 0.55)):
            shifted = pts + np.array([off, off * 0.5])
            hand.line(colour, shifted, firm, wd, 0.3)
        for k in range(2, len(pts) - 3, 4):
            p, q = pts[k], pts[k + 1]
            d = (q - p) / max(float(np.linalg.norm(q - p)), 1e-6)
            n = np.array([-d[1], d[0]])
            hand.line(CHAR, [p - 4 * n, p + 4 * n], 0.35, 1.6, 0.2)
        end, d = pts[-1], pts[-1] - pts[-3]
        d = d / max(float(np.linalg.norm(d)), 1e-6)
        claw = [end, end + 6 * d, end + 9 * d + np.array([-d[1], d[0]]) * 3]
        hand.line(CHAR, claw, [0.7, 0.6, 0.3], 3.0, 0.2)


def main() -> None:
    rng = np.random.default_rng(SEED)
    paper = Paper("wove", grain=2.0, depth=0.4, bite=0.5, give=0.4, color="#fbfaf6")
    sheet = Sheet(WIDTH, HEIGHT, paper=paper, seed=SEED)
    hand = Hand(sheet, rng)
    bird = build(rng)
    draw_sketch(hand)
    draw_tail(hand, bird)
    draw_body(hand, bird)
    draw_wing(hand, bird)
    draw_head(hand, bird)
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
