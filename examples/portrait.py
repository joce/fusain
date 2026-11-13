"""A young woman with dark shoulder-length hair and a fringe, in ballpoint
pen: black, blue and red.

Run from the repository root: ``python examples/portrait.py``. Writes
``examples/output/portrait.png``.

The head is built, not copied. Its proportions are the classical ones (eyes
at half the height of the head, an eye's width between them, the nose
halfway from there to the chin). Its surface is a relief: the egg of the
skull, with the brow, the eye sockets, the nose, the cheeks, the lips and
the chin modelled on it, and the neck under it. A lamp up and to the left
lights that relief, and casts the shadows of the nose, the lips, the fringe
and the jaw. The head is then turned a little to one side.

The pen follows the forms. The skin is hatched in blue, in lines that bend
over the relief as an engraver's do, pressed harder where the form turns
from the light; the lips are stroked in red across their length, the way
their folds run; the hair is drawn strand by strand in black, from the
crown down, lighter where it rounds the skull and catches the lamp.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from murex import CatmullRom, filters, write_png

from fusain import Sheet, papers, pens

WIDTH, HEIGHT = 1000, 1300
#: Where the middle of the face is (between the eyes), and the size of the
#: head: sheet pixels per unit of the head's own measure.
CX, CY, SIZE = 500.0, 540.0, 1.25
#: How far the head is turned to the side and tipped (radians).
TURN, TIP = math.radians(11.0), math.radians(-4.0)
#: Toward the lamp: left, up and out of the sheet.
LAMP = np.array([-0.34, -0.36, 0.87])

BLUE = pens.ballpoint("#22357e", width=2.0)
BLACK = pens.ballpoint("#1d1d21", width=2.0)
RED = pens.ballpoint("#b3262b", width=2.0)

# --------------------------------------------------------------------------
# the relief of the head, facing the sheet, in the head's own measure (x to
# its left as we see it, y down, the origin between the eyes)

GRID = np.mgrid[-470:620, -400:400].astype(np.float64)  # y, x of every point
YS, XS = GRID[0], GRID[1]


def bump(x: float, y: float, wide: float, tall: float, high: float) -> np.ndarray:
    """A rounded rise of the surface (or a hollow, if ``high`` is negative)."""
    return high * np.exp(-(((XS - x) / wide) ** 2) - ((YS - y) / tall) ** 2)


def both(x: float, y: float, wide: float, tall: float, high: float) -> np.ndarray:
    return bump(x, y, wide, tall, high) + bump(-x, y, wide, tall, high)


def relief() -> np.ndarray:
    """How far every point of head, neck and shoulders stands toward us."""
    egg = 1.0 - (XS / 168.0) ** 2 - ((YS + 8.0) / 232.0) ** 2
    jaw = np.clip((YS - 30.0) / 185.0, 0.0, 1.0)  # the face narrows to the chin
    jaw = jaw * jaw * (3.0 - 2.0 * jaw)
    egg -= jaw * (XS / 150.0) ** 2 * 0.9
    head = 150.0 * np.clip(egg, 0.0, None) ** 0.6 - 40.0
    neck = 70.0 * np.clip(1.0 - (XS / 80.0) ** 2, 0.0, 1.0) ** 0.6 - 40.0
    neck = np.where((YS > 150.0) & (YS < 300.0), neck, -40.0)
    slope = 266.0 + 0.16 * np.abs(XS)  # the line of the shoulders
    chest = np.clip((YS - slope) / 40.0, 0.0, 1.0) * (64.0 - 0.0003 * XS**2) - 40.0
    surface = np.maximum(np.maximum(head, neck), chest)
    surface += both(78.0, 62.0, 52.0, 44.0, 12.0)  # cheeks
    surface += bump(0.0, 40.0, 13.0, 48.0, 15.0) + bump(0.0, 86.0, 21.0, 22.0, 30.0)  # nose
    surface += both(21.0, 96.0, 11.0, 10.0, 9.0)  # its wings
    surface += both(68.0, -2.0, 36.0, 20.0, -15.0) + both(68.0, 0.0, 22.0, 12.0, 8.0)  # eyes
    surface += both(62.0, -40.0, 46.0, 12.0, 5.0)  # brows
    surface += bump(0.0, 141.0, 36.0, 8.0, 7.0) + bump(0.0, 161.0, 31.0, 10.0, 9.0)  # lips
    surface += bump(0.0, 122.0, 9.0, 12.0, -3.0) + both(47.0, 150.0, 9.0, 9.0, -4.0)
    surface += bump(0.0, 177.0, 26.0, 7.0, -5.0) + bump(0.0, 196.0, 27.0, 18.0, 10.0)  # chin
    for _ in range(3):
        surface = filters.box_blur(surface, 3)
    return surface


def lit(surface: np.ndarray) -> np.ndarray:
    """How dark every point of the relief is under the lamp, 0..1: turned
    from it, or in the shadow another part casts."""
    gy, gx = np.gradient(surface)
    facing = (-gx * LAMP[0] - gy * LAMP[1] + LAMP[2]) / np.sqrt(gx**2 + gy**2 + 1.0)
    tone = np.clip(0.95 * (0.95 - facing), 0.0, 1.0)
    shadow = np.zeros_like(surface)
    reach = np.hypot(LAMP[0], LAMP[1])
    for step in range(2, 60, 2):  # look toward the lamp for something higher
        dx, dy = round(step * LAMP[0] / reach), round(step * LAMP[1] / reach)
        above = np.roll(surface, (-dy, -dx), axis=(0, 1)) - surface - step * LAMP[2] / reach
        shadow = np.maximum(shadow, np.clip(above / 12.0, 0.0, 1.0))
    return np.clip(tone + 0.3 * filters.box_blur(shadow, 5), 0.0, 1.0)


# --------------------------------------------------------------------------
# from the head's measure to the sheet


class Pose:
    """The head turned and tipped, and put on the sheet."""

    def __init__(self, surface: np.ndarray):
        self.surface = surface
        shift = XS * math.cos(TURN) + (surface - 80.0) * math.sin(TURN)
        self.shift = np.maximum.accumulate(shift, axis=1)

    def height(self, x, y) -> np.ndarray:
        col = np.clip(np.asarray(x, dtype=np.int64) + 400, 0, 799)
        row = np.clip(np.asarray(y, dtype=np.int64) + 470, 0, 1089)
        return self.surface[row, col]

    def sheet(self, points, lift: float = 0.0) -> np.ndarray:
        """The sheet points of points of the head's measure (N, 2), each
        ``lift`` above the relief."""
        points = np.asarray(points, dtype=np.float64)
        x, y = points[..., 0], points[..., 1]
        z = self.height(x, y) + lift - 80.0
        x = x * math.cos(TURN) + z * math.sin(TURN)
        tipped_x = x * math.cos(TIP) - y * math.sin(TIP)
        tipped_y = x * math.sin(TIP) + y * math.cos(TIP)
        return np.stack([CX + SIZE * tipped_x, CY + SIZE * tipped_y], axis=-1)

    def field(self, values: np.ndarray) -> np.ndarray:
        """A map of the head's measure, as a map of the sheet (HEIGHT, WIDTH)."""
        ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        x, y = (xs - CX) / SIZE, (ys - CY) / SIZE
        back_x = x * math.cos(TIP) + y * math.sin(TIP)
        back_y = -x * math.sin(TIP) + y * math.cos(TIP)
        row = np.clip(back_y.astype(np.int64) + 470, 0, 1089)
        out = np.zeros((HEIGHT, WIDTH))
        for r in np.unique(row):  # (the turn moves points along their row only)
            here = row == r
            out[here] = np.interp(back_x[here], self.shift[r], values[r])
        return out


# --------------------------------------------------------------------------
# hair: strands from the crown, in the head's own measure


def strand(points, rng, wobble: float = 3.0, count: int = 26) -> np.ndarray:
    """A strand through ``points``, a little unsteady: ``count`` points."""
    knots = np.asarray(points, dtype=np.float64)
    knots[1:] += rng.normal(0.0, wobble, knots[1:].shape)
    dense = CatmullRom(knots).sample(4.0)[0]
    return dense[np.linspace(0, len(dense) - 1, count).astype(int)]


def side_hair(side: float, depth: float, rng) -> np.ndarray:
    """A strand of the hair that falls beside the face: ``depth`` is 0 next
    to the face and 1 at the outside of the hair. It leaves the parting,
    rounds the skull, hangs to the shoulder and turns in at its end."""
    q = depth
    return strand(
        [(side * (6 + 30 * q) + 10, -236 + 6 * q), (side * (92 + 62 * q), -196 + 30 * q),
         (side * (134 + 46 * q), -70 + 20 * q), (side * (136 + 50 * q), 60),
         (side * (124 + 56 * q), 180), (side * (108 + 62 * q), 262 + rng.normal(0.0, 6.0)),
         (side * (88 + 60 * q), 300 + 14 * q + rng.normal(0.0, 8.0))],
        rng,
    )  # fmt: skip


def fringe(end: float, rng) -> np.ndarray:
    """A strand of the fringe: from the crown forward and down over the
    forehead, to end over the brow at ``end`` across the face."""
    low = -52.0 + 5.0 * math.cos(end / 40.0) + rng.normal(0.0, 4.0)
    return strand([(10 + 0.2 * end, -234), (0.72 * end + 4, -178), (0.96 * end, -110), (end, low)], rng, 2.0, 18)


def hair_mask(pose: Pose) -> np.ndarray:
    """Where the hair hides the skin, on the sheet: 1 under the hair."""
    covered = np.zeros((HEIGHT, WIDTH))
    rng = np.random.default_rng(1)
    for side in (1.0, -1.0):
        inner = pose.sheet(side_hair(side, 0.0, rng), 10.0)
        outer = pose.sheet(side_hair(side, 1.0, rng), 10.0)
        fill(covered, np.concatenate([inner, outer[::-1]]))
    brow = pose.sheet(np.array([fringe(end, rng)[-1] for end in np.linspace(-132, 132, 30)]), 10.0)
    top = pose.sheet(np.array([(x, -262 + 0.0006 * x * x) for x in np.linspace(150, -150, 12)]), 10.0)
    fill(covered, np.concatenate([brow, top]))
    return covered


def fill(mask: np.ndarray, polygon: np.ndarray) -> None:
    """Set the pixels inside ``polygon`` to 1 in ``mask``."""
    x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int), 0)
    x1, y1 = np.ceil(polygon.max(axis=0)).astype(int) + 1
    x1, y1 = min(x1, mask.shape[1]), min(y1, mask.shape[0])
    ys, xs = np.mgrid[y0:y1, x0:x1] + 0.5
    inside = np.zeros(xs.shape, bool)
    ax, ay = polygon[-1]
    for bx, by in polygon:
        if ay != by:
            crosses = (ys >= min(ay, by)) & (ys < max(ay, by))
            inside ^= crosses & (xs < ax + (ys - ay) * (bx - ax) / (by - ay))
        ax, ay = bx, by
    mask[y0:y1, x0:x1][inside] = 1.0


def draw_hair(sheet: Sheet, pose: Pose, rng) -> None:
    """The hair, strand by strand: blue first, then black over it; pressed
    lightly where it rounds the skull and catches the lamp, hard in its
    depths: next to the face, under the fringe, toward its ends."""
    for pen, count in ((BLUE, 150), (BLACK, 400)):
        for k in range(count):
            side = 1.0 if k % 2 else -1.0
            depth = float(rng.random())
            path = side_hair(side, depth, rng)
            y = path[:, 1]
            sheen = np.exp(-(((y + 150.0 - 40.0 * depth) / 46.0) ** 2))  # the light round the skull
            lower = np.exp(-(((y - 150.0) / 70.0) ** 2)) * 0.35  # and where it turns at the cheek
            deep = 0.85 - 0.3 * depth * (side < 0) + 0.1 * (1.0 - depth)
            touch = np.clip(deep * (1.0 - 0.9 * sheen - lower) + rng.normal(0.0, 0.1), 0.04, 1.0)
            touch[-3:] *= (0.8, 0.55, 0.3)  # the end of a hair is fine
            sheet.stroke(pose.sheet(path, 10.0), pen, touch.tolist())
    for pen, count in ((BLUE, 70), (BLACK, 300)):
        for end in rng.uniform(-134.0, 134.0, count):
            path = fringe(float(end), rng)
            y = path[:, 1]
            sheen = np.exp(-(((y + 128.0) / 34.0) ** 2))
            touch = np.clip(0.8 * (1.0 - 0.75 * sheen) + rng.normal(0.0, 0.08), 0.05, 1.0)
            touch[-2:] *= (0.7, 0.4)
            sheet.stroke(pose.sheet(path, 12.0), pen, touch.tolist())


# --------------------------------------------------------------------------
# skin: hatching that bends over the relief


def hatch(sheet: Sheet, pen, pressure: np.ndarray, height: np.ndarray, angle: float, apart: float, rng) -> None:
    """Hatch wherever ``pressure`` (a map of the sheet) asks, in lines
    ``apart`` pixels from one another at ``angle`` degrees, each bent
    sideways by the relief under it (``height``), as an engraver's lines
    bend over a form."""
    turn = math.radians(angle)
    along = np.array([math.cos(turn), -math.sin(turn)])
    across = np.array([math.sin(turn), math.cos(turn)])
    reach = math.hypot(WIDTH, HEIGHT)
    for offset in np.arange(-reach, reach, apart):
        t = np.arange(-reach, reach, 3.0)
        line = np.array([0.5 * WIDTH, 0.5 * HEIGHT]) + (offset + rng.uniform(-0.5, 0.5)) * across + t[:, None] * along
        inside = (line[:, 0] >= 1) & (line[:, 0] < WIDTH - 1) & (line[:, 1] >= 1) & (line[:, 1] < HEIGHT - 1)
        line = line[inside]
        if len(line) < 3:
            continue
        col, row = line[:, 0].astype(int), line[:, 1].astype(int)
        line = line + 0.22 * height[row, col][:, None] * across
        col = np.clip(line[:, 0].astype(int), 0, WIDTH - 1)
        row = np.clip(line[:, 1].astype(int), 0, HEIGHT - 1)
        touch = pressure[row, col]
        marks = np.nonzero(touch > 0.03)[0]
        if marks.size < 2:
            continue
        for run in np.split(marks, np.nonzero(np.diff(marks) > 1)[0] + 1):
            if run.size >= 3:
                ends = np.ones(run.size)
                ends[0] = ends[-1] = 0.5
                sheet.stroke(line[run], pen, (touch[run] * ends).tolist())


def draw_skin(sheet: Sheet, tone: np.ndarray, skin: np.ndarray, height: np.ndarray, rng) -> None:
    """The modelling of face, neck and shoulders: three directions of blue
    hatching, each starting at a deeper tone than the last."""
    for angle, starts, firm, apart in ((48.0, 0.06, 0.3, 3.6), (-24.0, 0.34, 0.4, 4.0), (82.0, 0.58, 0.5, 4.2)):
        pressure = skin * np.clip(firm * (tone - starts) / (1.0 - starts), 0.0, 0.45)
        hatch(sheet, BLUE, pressure, height, angle, apart, rng)


# --------------------------------------------------------------------------
# the features, in the head's own measure


def line(pose: Pose, points, count: int = 24, lift: float = 0.0) -> np.ndarray:
    """A smooth curve through points of the head's measure, on the sheet."""
    dense = CatmullRom(np.asarray(points, dtype=np.float64)).sample(1.0)[0]
    return pose.sheet(dense[np.linspace(0, len(dense) - 1, count).astype(int)], lift)


def draw_eye(sheet: Sheet, pose: Pose, side: float, rng) -> None:
    """An eye: the iris and pupil under the upper lid, a point of light, the
    lash line, the fold of the lid, the lashes."""
    def at(x: float, y: float) -> tuple[float, float]:
        return (side * x, y)

    centre = np.array(at(68.0, -5.0))
    upper = [at(34, 7), at(49, -9), at(67, -16), at(88, -14), at(105, -6)]
    lower = [at(105, -6), at(91, 7), at(71, 12), at(52, 11), at(34, 7)]
    lid = CatmullRom(np.array(upper)).sample(1.0)[0]
    lid = lid[np.argsort(lid[:, 0])]

    def open_at(points: np.ndarray) -> np.ndarray:
        """1 where points lie below the upper lid: where the eye is open."""
        return (points[:, 1] > np.interp(points[:, 0], lid[:, 0], lid[:, 1]) + 1.5) * 1.0

    light = centre + np.array([-6.0, -5.0])

    def ring(radius: float, count: int = 30) -> np.ndarray:
        return centre + radius * np.array([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, count)])

    for radius in np.arange(17.0, 7.0, -2.0):  # the iris, dark, a little lighter low down
        around = ring(radius)
        shown = open_at(around) * (np.hypot(*(around - light).T) > 4.0)
        low = np.clip((around[:, 1] - centre[1]) / 17.0, 0.0, 1.0)
        sheet.stroke(pose.sheet(around), BLACK, ((0.75 - 0.3 * low) * shown).tolist())
    for k in range(40):  # its rays, from the pupil out
        turn = 2 * math.pi * (k + rng.random()) / 40
        ray = centre + np.outer(np.linspace(7.0, 17.5, 5), [math.cos(turn), math.sin(turn)])
        shown = open_at(ray) * (np.hypot(*(ray - light).T) > 4.0)
        sheet.stroke(pose.sheet(ray), BLUE if k % 3 else BLACK, (0.5 * shown).tolist())
    for radius, firm in ((17.8, 0.85), (7.0, 1.0), (5.2, 1.0), (3.4, 1.0), (1.6, 1.0)):  # its rim, and the pupil
        around = ring(radius)
        shown = open_at(around) * (np.hypot(*(around - light).T) > 3.4)
        sheet.stroke(pose.sheet(around), BLACK, (firm * shown).tolist())
    under = np.array([at(x, y) for x, y in ((44, -3), (58, -10), (74, -11), (92, -8))])
    sheet.stroke(line(pose, under, 14), BLUE, 0.3)  # the lid's shadow on the eye
    lash_line = line(pose, upper, 30)
    sheet.stroke(lash_line, BLACK, np.linspace(0.6, 1.0, 30).tolist())
    sheet.stroke(lash_line - np.array([0.0, 1.6]), BLACK, np.linspace(0.2, 0.9, 30).tolist())
    sheet.stroke(line(pose, [at(40, -12), at(58, -28), at(80, -28), at(102, -14)]), BLUE, 0.35)
    sheet.stroke(line(pose, lower), BLUE, np.r_[np.linspace(0.45, 0.2, 12), np.full(12, 0.12)].tolist())
    for t in np.linspace(0.15, 1.0, 17):  # the lashes, longer toward the outer corner
        root = lid[int(t * (len(lid) - 1))] if side > 0 else lid[int((1.0 - t) * (len(lid) - 1))]
        out = np.array([side * (0.25 + 0.9 * t), -1.0])
        out = out / float(np.hypot(*out)) * (5.0 + 7.5 * t)
        lash = np.array([root, root + 0.6 * out + np.array([0.0, 1.0]), root + out])
        sheet.stroke(pose.sheet(lash), BLACK, [0.75, 0.5, 0.2])


def draw_brow(sheet: Sheet, pose: Pose, side: float, rng) -> None:
    """An eyebrow, hair by hair: the hairs rise at its inner end and lie
    along it toward its tail, where it thins."""
    spine = CatmullRom(np.array([(26, -33), (46, -41), (70, -46), (92, -44), (114, -36)], float)).sample(1.0)[0]
    for _ in range(95):
        t = float(rng.random()) ** 0.85
        root = spine[int(t * (len(spine) - 1))] + np.array([0.0, rng.uniform(-0.5, 0.5) * (11.0 - 8.0 * t)])
        lean = math.radians(-68.0 + 82.0 * t + rng.normal(0.0, 8.0))
        tip = root + rng.uniform(8.0, 13.0) * np.array([math.cos(lean), math.sin(lean)])
        hair = np.array([root, 0.5 * (root + tip), tip]) * np.array([side, 1.0])
        sheet.stroke(pose.sheet(hair), BLACK, (np.array([0.6, 0.5, 0.2]) * rng.uniform(0.6, 1.0)).tolist())


def draw_nose(sheet: Sheet, pose: Pose) -> None:
    """The nose: its nostrils, the turn of its wings, the edge under its tip."""
    for side in (1.0, -1.0):
        for firm in (0.8, 0.6):
            sheet.stroke(line(pose, [(side * 7, 104), (side * 14, 100), (side * 22, 103)], 8), BLACK, firm)
        sheet.stroke(line(pose, [(side * 26, 84), (side * 31, 94), (side * 26, 105)], 10), BLUE, 0.3 if side < 0 else 0.45)
    sheet.stroke(line(pose, [(-9, 107), (0, 110), (9, 107)], 8), BLUE, 0.3)


def draw_lips(sheet: Sheet, pose: Pose, rng) -> None:
    """The lips, in red: strokes across them, the way their folds run; the
    upper lip in its own shadow, a light left on the lower; then the line
    between them."""
    xs = np.linspace(-46.0, 46.0, 60)
    part = np.interp(xs, (-46, -30, -12, 0, 12, 30, 46), (147.5, 150.5, 149, 152, 149, 150.5, 147.5))
    top = np.interp(xs, (-46, -28, -12, 0, 12, 28, 46), (147.5, 139, 133, 137, 133, 139, 147.5))
    low = np.interp(xs, (-46, -30, 0, 30, 46), (147.5, 165, 172.5, 165, 147.5))
    for k in range(len(xs)):
        x = xs[k] + rng.normal(0.0, 0.4)
        lean = 0.12 * x  # the folds fan out from the middle
        if part[k] - top[k] > 1.5:
            fold = np.array([(x, part[k] - 0.5), (x + 0.3 * lean, 0.5 * (part[k] + top[k])), (x + 0.5 * lean, top[k] + 0.8)])
            sheet.stroke(pose.sheet(fold), RED, [0.7, 0.6, 0.4])
        if low[k] - part[k] > 1.5:
            fold = np.array([(x, part[k] + 0.8), (x + 0.3 * lean, part[k] + 0.4 * (low[k] - part[k])),
                             (x + 0.45 * lean, part[k] + 0.7 * (low[k] - part[k])), (x + 0.5 * lean, low[k] - 0.8)])  # fmt: skip
            shine = math.exp(-((x / 20.0) ** 2))  # the light on the lower lip
            sheet.stroke(pose.sheet(fold), RED, [0.55, 0.42 - 0.34 * shine, 0.45 - 0.2 * shine, 0.6])
    between = pose.sheet(np.stack([xs, part], axis=1))
    sheet.stroke(between, BLACK, (0.35 + 0.4 * np.abs(np.cos(xs / 46.0 * math.pi))).tolist())
    sheet.stroke(pose.sheet(np.stack([xs, top - 0.5], axis=1)), RED, 0.3)
    sheet.stroke(pose.sheet(np.stack([xs, low + 0.5], axis=1)), RED, 0.35)


def draw_lines(sheet: Sheet, pose: Pose, covered: np.ndarray) -> None:
    """The lines of jaw, neck, collarbones and shoulders, where the hair
    does not hide them; lighter on the side the lamp is on."""
    def shown(path: np.ndarray) -> np.ndarray:
        col = np.clip(path[:, 0].astype(int), 0, WIDTH - 1)
        row = np.clip(path[:, 1].astype(int), 0, HEIGHT - 1)
        return 1.0 - covered[row, col]

    for side in (1.0, -1.0):
        firm = 0.5 if side > 0 else 0.3
        down = np.linspace(-20.0, 222.0, 50)
        narrow = np.clip((down - 30.0) / 185.0, 0.0, 1.0)
        narrow = narrow * narrow * (3.0 - 2.0 * narrow)
        wide = np.sqrt(np.clip(1.0 - ((down + 8.0) / 232.0) ** 2, 0.0, 1.0) / (1 / 168.0**2 + 0.9 * narrow / 150.0**2))
        jaw = pose.sheet(np.stack([side * (wide - 3.0), down], axis=1))
        sheet.stroke(jaw, BLUE, (firm * shown(jaw)).tolist())
        neck = line(pose, [(side * 78, 205), (side * 76, 236), (side * 86, 266), (side * 150, 292), (side * 300, 334)], 40)
        sheet.stroke(neck, BLUE, (firm * shown(neck) * np.linspace(1.0, 0.25, 40)).tolist())
        bone = line(pose, [(side * 18, 304), (side * 70, 296), (side * 150, 300)], 20, 2.0)
        sheet.stroke(bone, BLUE, (0.3 * np.sin(np.linspace(0.2, math.pi, 20))).tolist())
    top = line(pose, [(-210, 356), (-110, 404), (0, 424), (110, 404), (210, 356)], 50)
    sheet.stroke(top, BLACK, (0.55 * np.sin(np.linspace(0.25, math.pi - 0.25, 50))).tolist())


def draw_blush(sheet: Sheet, pose: Pose, skin: np.ndarray, rng) -> None:
    """A little red in the cheeks, very lightly."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    warmth = np.zeros((HEIGHT, WIDTH))
    for side in (1.0, -1.0):
        cx, cy = pose.sheet(np.array([side * 86.0, 72.0]))
        warmth += np.exp(-(((xs - cx) / 52.0) ** 2) - ((ys - cy) / 38.0) ** 2)
    hatch(sheet, RED, 0.1 * warmth * skin * (warmth > 0.3), np.zeros((HEIGHT, WIDTH)), 40.0, 4.4, rng)


def main() -> None:
    rng = np.random.default_rng(11)
    surface = relief()
    pose = Pose(surface)
    tone = pose.field(lit(surface))
    height = pose.field(surface)
    inside = pose.field((surface > -38.0) * 1.0)
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    # (the drawing fades out below and to the sides of the shoulders)
    fade = np.clip((1120.0 - ys) / 200.0, 0.0, 1.0) * np.clip(1.6 - np.abs(xs - CX) / 260.0, 0.0, 1.0)
    covered = hair_mask(pose)
    skin = (inside > 0.5) * (1.0 - covered) * fade
    sheet = Sheet(WIDTH, HEIGHT, paper=papers.BRISTOL.with_(color="#fbfaf7"), seed=11)
    draw_skin(sheet, tone, skin, height, rng)
    draw_blush(sheet, pose, skin, rng)
    draw_lines(sheet, pose, covered)
    draw_nose(sheet, pose)
    draw_lips(sheet, pose, rng)
    for side in (1.0, -1.0):
        draw_eye(sheet, pose, side, rng)
        draw_brow(sheet, pose, side, rng)
    draw_hair(sheet, pose, rng)
    out = Path(__file__).parent / "output" / "portrait.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
