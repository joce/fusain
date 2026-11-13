"""A portrait of a young woman, in blue, red and black ballpoint pen.

Run from the repository root: ``python examples/portrait.py``. Writes
``examples/output/portrait.png``.

Drawn in the manner of a pen draughtsman who builds everything out of
hatching, as an engraver does: sets of long, evenly spaced strokes that sweep
across a form following its roundness, crossed by a second set and a third as
the form turns into shadow; the skin built of red, its shadows crossed with
blue to violet, the hair black over red, with black kept for the deepest
accents.

She is invented, and built before she is drawn: a small head, neck and
shoulders, and the hair round them, are modelled in three dimensions as
distance fields, turned three-quarters to the left and tilted a little, and
lit by one soft light from above on the left. Raymarching them tells, at each
point of the sheet, what is seen there, the point of the head it is and how
much light it gets. The hatching follows that knowledge: each set of strokes
runs along the sections of the head by a family of parallel planes, which
curve over the brow, the cheek and the nose as the lines of an engraving do,
and each is laid only where the tone needs it, thinning out toward the light.
The hair is drawn in long strokes along the way it falls from the crown,
massed into locks, with the shine left as paper. The eyes, the brows, the
nose and the mouth are designed face on, laid on the surface of the head and
seen through the same camera, so that they turn with it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from murex import filters, write_png

from fusain import Sheet, papers, pens

WIDTH, HEIGHT = 1100, 1400
#: Pixels per centimetre of the sitter, and where the centre of her head
#: lands on the sheet.
SCALE = 26.0
ORIGIN = (610.0, 450.0)
#: How her head is turned (to her right), bowed and tilted, in degrees.
YAW, PITCH, ROLL = -27.0, -4.0, 5.0
#: Toward the light: from the upper left and in front.
LIGHT = np.array([-0.5, 0.55, 0.67]) / math.sqrt(0.25 + 0.3025 + 0.4489)
#: The raymarch runs at a coarser grid than the sheet: one ray per STEP
#: pixels each way.
STEP = 2
SEED = 11

Vec = tuple[np.ndarray, np.ndarray, np.ndarray]
Point = tuple[float, float, float]

RED = pens.RED
BLUE = pens.BLUE
BLACK = pens.BLACK
FINE_RED = pens.RED.with_(width=2.0)
FINE_BLUE = pens.BLUE.with_(width=2.0)
FINE_BLACK = pens.BLACK.with_(width=2.0)


# ----------------------------------------------------------------------
# The camera


def rotation() -> np.ndarray:
    """From head space (x to her left, y up, z out of her face) to the
    camera's (x right, y up, z toward the eye)."""
    yaw, pitch, roll = (math.radians(a) for a in (YAW, PITCH, ROLL))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    turn = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    bow = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    tilt = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return tilt @ bow @ turn


ROT = rotation()


def project(p) -> np.ndarray:
    """Points of head space (..., 3) to the sheet (..., 2) and depth."""
    v = np.asarray(p, dtype=np.float64) @ ROT.T
    return np.stack(
        [ORIGIN[0] + SCALE * v[..., 0], ORIGIN[1] - SCALE * v[..., 1], v[..., 2]],
        axis=-1,
    )


def to_head(v: Vec) -> Vec:
    """Points of camera space to head space."""
    r = ROT.T
    return (
        r[0, 0] * v[0] + r[0, 1] * v[1] + r[0, 2] * v[2],
        r[1, 0] * v[0] + r[1, 1] * v[1] + r[1, 2] * v[2],
        r[2, 0] * v[0] + r[2, 1] * v[1] + r[2, 2] * v[2],
    )


# ----------------------------------------------------------------------
# Distance fields (in centimetres of head space)


def ellipsoid(p: Vec, c: Point, r: Point) -> np.ndarray:
    q = [(p[i] - c[i]) / r[i] for i in range(3)]
    k0 = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    k1 = np.sqrt((q[0] / r[0]) ** 2 + (q[1] / r[1]) ** 2 + (q[2] / r[2]) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def sphere(p: Vec, c: Point, r: float) -> np.ndarray:
    return np.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2) - r


def capsule(p: Vec, a: Point, b: Point, r: float) -> np.ndarray:
    pa = [p[i] - a[i] for i in range(3)]
    ba = [b[i] - a[i] for i in range(3)]
    h = sum(pa[i] * ba[i] for i in range(3)) / sum(v * v for v in ba)
    h = np.clip(h, 0.0, 1.0)
    return np.sqrt(sum((pa[i] - ba[i] * h) ** 2 for i in range(3))) - r


def cone(p: Vec, a: Point, b: Point, r1: float, r2: float) -> np.ndarray:
    """A capsule whose radius goes from r1 at a to r2 at b."""
    ba = [b[i] - a[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in ba))
    h = sum((p[i] - a[i]) * ba[i] for i in range(3)) / (length * length)
    h = np.clip(h, 0.0, 1.0)
    d = np.sqrt(sum((p[i] - a[i] - ba[i] * h) ** 2 for i in range(3)))
    return d - (r1 + (r2 - r1) * h)


def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -smin(-a, -b, k)


def skull(p: Vec) -> np.ndarray:
    d = ellipsoid(p, (0, 2.0, -0.8), (7.1, 8.4, 9.6))
    return smin(d, ellipsoid(p, (0, 3.0, 3.0), (5.6, 5.0, 5.9)), 2.0)


def face(p: Vec) -> np.ndarray:
    """The skull and the face."""
    m = (np.abs(p[0]), p[1], p[2])
    d = skull(p)
    d = smin(d, ellipsoid(p, (0, -2.5, 2.5), (6.0, 5.0, 6.2)), 2.5)
    d = smin(d, ellipsoid(p, (0, -6.6, 2.2), (4.1, 4.9, 6.5)), 2.5)
    d = smin(d, cone(m, (4.5, -7.6, -0.8), (0.9, -10.5, 6.8), 0.8, 1.0), 2.5)
    d = smin(d, ellipsoid(p, (0, -10.1, 7.6), (1.7, 1.5, 1.6)), 2.0)  # chin
    d = smin(d, ellipsoid(p, (0, -6.8, 5.2), (3.1, 3.0, 4.2)), 1.8)  # muzzle
    d = smin(d, ellipsoid(m, (3.4, -4.4, 5.8), (2.3, 2.4, 2.7)), 2.2)  # cheeks
    d = smin(d, ellipsoid(m, (4.3, -2.6, 5.6), (1.8, 1.2, 1.6)), 1.8)
    d = smin(d, ellipsoid(m, (2.8, 0.7, 6.9), (2.5, 0.8, 1.3)), 1.6)  # brows
    d = smax(d, -ellipsoid(m, (3.1, -0.9, 8.75), (1.7, 1.1, 1.3)), 1.6)
    d = smin(d, ellipsoid(m, (3.1, -0.95, 6.7), (1.4, 1.05, 1.2)), 0.8)  # eyes
    return d


def features(p: Vec) -> np.ndarray:
    """The nose and the lips."""
    m = (np.abs(p[0]), p[1], p[2])
    d = cone(p, (0, -0.3, 8.1), (0, -4.0, 10.1), 0.5, 0.7)
    d = smin(d, sphere(p, (0, -4.35, 10.35), 0.85), 0.6)
    d = smin(d, ellipsoid(m, (1.1, -4.85, 9.1), (0.7, 0.55, 0.75)), 0.6)
    d = np.minimum(d, ellipsoid(p, (0, -6.45, 5.3), (2.4, 0.6, 4.45)))
    return np.minimum(d, ellipsoid(p, (0, -7.5, 5.2), (2.0, 0.68, 4.35)))


def skin(p: Vec) -> np.ndarray:
    """Head, neck and shoulders."""
    m = (np.abs(p[0]), p[1], p[2])
    d = face(p)
    f = features(p)
    d = smin(d, f, np.where(p[1] < -6.0, 0.5, 1.0))
    d = smin(d, capsule(p, (0, -4.0, -2.0), (0, -20.0, -3.0), 4.9), 1.0)  # neck
    d = smin(d, capsule(m, (3.5, -17.5, -3.0), (15.0, -21.5, -2.5), 3.0), 4.0)
    return smin(d, ellipsoid(p, (0, -26.0, -2.0), (17.5, 6.5, 8.5)), 5.0)


def hair(p: Vec) -> np.ndarray:
    """A cap over the skull, a fringe, and a bob to the shoulders."""
    x, y, z = p
    cap = skull(p) - 1.0
    fall = np.clip(-(y + 2.0) / 15.0, 0.0, 1.0)
    rx = 8.4 - 0.9 * fall**2
    rz = 9.6 - 1.0 * fall**2
    col = (np.sqrt((x / rx) ** 2 + ((z + 1.4) / rz) ** 2) - 1.0) * rx
    col = smax(col, -(y + 17.0) - 0.5 * np.cos(0.5 * x), 1.5)
    col = smax(col, y - 3.0, 2.0)
    d = smin(cap, col, 2.5)
    # the opening for the face: below the fringe, in front of the ears
    w = np.maximum(6.5 - 0.012 * (y + 3.0) ** 2, 5.2)
    fringe = 1.35 + 0.035 * x * x + 0.18 * np.sin(2.3 * x + 1.0)
    hole = smax(np.abs(x) - w, y - fringe, 0.4)
    hole = smax(hole, 0.8 - z, 0.5)
    return smax(d, -hole, 0.5)


def scene(p: Vec) -> np.ndarray:
    return np.minimum(skin(p), hair(p))


# ----------------------------------------------------------------------
# Raymarching


def march(xs: np.ndarray, ys: np.ndarray, near: float = 30.0) -> np.ndarray:
    """Orthographic rays toward -z from camera-space (xs, ys, near): the
    depth reached (camera z), or -inf where nothing is hit."""
    t = np.zeros(xs.size)
    alive = np.ones(xs.size, bool)
    hit = np.zeros(xs.size, bool)
    for _ in range(110):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        d = scene(to_head((xs[idx], ys[idx], near - t[idx])))
        t[idx] += 0.7 * d
        done = d < 0.008
        hit[idx[done]] = True
        alive[idx[done | (t[idx] > 2.2 * near)]] = False
    return np.where(hit, near - t, -np.inf)


def gradient(f: Callable[[Vec], np.ndarray], p: Vec, e: float = 0.02) -> np.ndarray:
    g = []
    for i in range(3):
        a, b = list(p), list(p)
        a[i] = a[i] + e
        b[i] = b[i] - e
        g.append(f((a[0], a[1], a[2])) - f((b[0], b[1], b[2])))
    out = np.array(g)
    return out / (np.linalg.norm(out, axis=0) + 1e-9)


def shadow(p: Vec) -> np.ndarray:
    """Soft shadow toward the light: 1 in full light."""
    toward = ROT.T @ LIGHT
    res = np.ones(p[0].size)
    t = np.full(p[0].size, 0.12)
    for _ in range(30):
        d = scene((p[0] + toward[0] * t, p[1] + toward[1] * t, p[2] + toward[2] * t))
        res = np.minimum(res, 5.0 * d / t)
        t += np.clip(d, 0.08, 1.5)
    return np.clip(res, 0.0, 1.0)


def occlusion(p: Vec, n: np.ndarray) -> np.ndarray:
    occ = np.zeros(p[0].size)
    for i in range(1, 6):
        h = 0.3 * i
        d = scene((p[0] + n[0] * h, p[1] + n[1] * h, p[2] + n[2] * h))
        occ += (h - d) / 2.0**i
    return np.clip(1.0 - 1.1 * occ, 0.0, 1.0)


@dataclass
class Model:
    """What the camera sees, at each pixel of the sheet."""

    skin: np.ndarray  # bool: skin seen here
    hair: np.ndarray  # bool: hair seen here
    pos: np.ndarray  # (3, H, W) the point seen, in head space
    normal: np.ndarray  # (3, H, W) its normal, in head space
    light: np.ndarray  # direct light, 0..1, shadows included
    ao: np.ndarray  # ambient occlusion
    depth: np.ndarray  # camera z, -inf off the figure


def sample(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``a`` read bilinearly at (x, y), in its own pixels (centres at +0.5)."""
    h, w = a.shape
    fx = np.clip(np.asarray(x, dtype=np.float64) - 0.5, 0.0, w - 1.001)
    fy = np.clip(np.asarray(y, dtype=np.float64) - 0.5, 0.0, h - 1.001)
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    tx, ty = fx - ix, fy - iy
    top = a[iy, ix] * (1 - tx) + a[iy, ix + 1] * tx
    bottom = a[iy + 1, ix] * (1 - tx) + a[iy + 1, ix + 1] * tx
    return top * (1 - ty) + bottom * ty


def extend(a: np.ndarray, mask: np.ndarray, reach: int) -> np.ndarray:
    """``a`` carried ``reach`` cells out of ``mask``, so that it stays
    smooth across the edge."""
    a = np.where(mask, a, 0.0)
    m = mask.astype(np.float64)
    for _ in range(reach):
        num = filters.box_blur(a * m, 1)
        den = filters.box_blur(m, 1)
        a = np.where(m > 0, a, num / np.maximum(den, 1e-9))
        m = np.where(den > 0, 1.0, 0.0)
    return a


def upsample(a: np.ndarray) -> np.ndarray:
    """A coarse grid to the sheet, bilinearly."""
    xs = (np.arange(WIDTH) + 0.5) / STEP
    ys = (np.arange(HEIGHT) + 0.5) / STEP
    gx, gy = np.meshgrid(xs, ys)
    return sample(a.astype(np.float64), gx, gy)


def build_model() -> Model:
    gx = (np.arange(0, WIDTH, STEP) + 0.5 * STEP - ORIGIN[0]) / SCALE
    gy = -(np.arange(0, HEIGHT, STEP) + 0.5 * STEP - ORIGIN[1]) / SCALE
    shape = (gy.size, gx.size)
    xs, ys = (a.ravel() for a in np.meshgrid(gx, gy))
    depth = march(xs, ys)
    idx = np.nonzero(np.isfinite(depth))[0]
    p = to_head((xs[idx], ys[idx], depth[idx]))
    n = gradient(scene, p)
    lit = np.clip(n.T @ (ROT.T @ LIGHT), -1.0, 1.0)
    def grid(values, fill=0.0) -> np.ndarray:
        out = np.full(xs.size, fill, dtype=np.float64)
        out[idx] = values
        return out.reshape(shape)

    seen = grid(np.ones(idx.size)) > 0
    on_hair = grid(hair(p) < skin(p)) > 0
    light = np.clip((lit + 0.3) / 1.3, 0.0, 1.0) * shadow(p)

    def smooth(values) -> np.ndarray:
        return upsample(extend(grid(values), seen, 4))

    return Model(
        skin=upsample(grid(np.ones(idx.size)) * ~on_hair) > 0.5,
        hair=upsample(on_hair.astype(np.float64)) > 0.5,
        pos=np.stack([smooth(c) for c in p]),
        normal=np.stack([smooth(c) for c in n]),
        light=smooth(light),
        ao=smooth(occlusion(p, n)),
        depth=upsample(np.where(seen, grid(depth[idx]), -1e3)),
    )


# ----------------------------------------------------------------------
# Strokes


def smoothstep(lo: float, hi: float, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def envelope(n: int, entry: float = 0.12, flick: float = 0.35) -> np.ndarray:
    """How hard a hand presses along a stroke of n points: it comes down
    quickly and lifts off slowly, so the stroke tapers out at its end."""
    u = np.linspace(0.0, 1.0, n)
    return smoothstep(0.0, entry, u + 0.02) * (1.0 - smoothstep(1.0 - flick, 1.0, u))


@dataclass
class Field:
    """A direction to draw in at each pixel, and where drawing may go."""

    dx: np.ndarray
    dy: np.ndarray
    ok: np.ndarray  # float: 1 where a stroke may run, 0 where it stops


def trace(field: Field, x0: np.ndarray, y0: np.ndarray, steps: int, h: float):
    """Follow ``field`` from the seeds, ``steps`` steps of ``h`` pixels
    (negative ``h`` runs against it): positions (steps + 1, n) and whether
    each is still on the field."""
    xs = np.empty((steps + 1, x0.size))
    ys = np.empty_like(xs)
    live = np.empty(xs.shape, bool)
    x, y = x0.astype(np.float64), y0.astype(np.float64)
    on = sample(field.ok, x, y) > 0.5
    xs[0], ys[0], live[0] = x, y, on
    for i in range(1, steps + 1):
        # midpoint rule
        mx = x + 0.5 * h * sample(field.dx, x, y)
        my = y + 0.5 * h * sample(field.dy, x, y)
        ux, uy = sample(field.dx, mx, my), sample(field.dy, mx, my)
        norm = np.hypot(ux, uy) + 1e-9
        x, y = x + h * ux / norm, y + h * uy / norm
        on &= sample(field.ok, x, y) > 0.5
        xs[i], ys[i], live[i] = x, y, on
    return xs, ys, live


def draw_traced(
    sheet: Sheet,
    pen,
    xs: np.ndarray,
    ys: np.ndarray,
    firm: np.ndarray,
    every: int = 3,
) -> int:
    """Draw each column of (xs, ys) as a stroke, ``firm`` its pressure at
    each point (0 where it is not to draw): the stroke is cut where the
    pressure is nothing for long, and tapered."""
    count = 0
    for j in range(xs.shape[1]):
        p = firm[:, j]
        on = np.nonzero(p > 0.004)[0]
        if on.size < 4:
            continue
        a, b = on[0], on[-1] + 1
        sel = np.arange(a, b, every)
        if sel.size < 2:
            continue
        pts = list(zip(xs[sel, j].tolist(), ys[sel, j].tolist(), strict=True))
        sheet.stroke(pts, pen, (p[sel] * envelope(sel.size)).tolist())
        count += 1
    return count


def jittered_seeds(mask: np.ndarray, gap: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Points about ``gap`` apart, at random within each cell, inside
    ``mask``."""
    gx = np.arange(0.0, WIDTH, gap)
    gy = np.arange(0.0, HEIGHT, gap)
    x, y = (a.ravel() for a in np.meshgrid(gx, gy))
    x = x + rng.uniform(0, gap, x.size)
    y = y + rng.uniform(0, gap, y.size)
    keep = sample(mask.astype(np.float64), x, y) > 0.5
    return x[keep], y[keep]


@dataclass
class Hatch:
    """One set of hatching on the skin: strokes along the sections of the
    head by planes square to ``axis`` (head space), ``gap`` cm apart, about
    ``length`` pixels long, laid where ``tone`` is above ``lo`` and pressed
    up to ``firm`` where it reaches ``hi``."""

    pen: object
    axis: Point
    gap: float
    length: float
    lo: float
    hi: float
    firm: float


def section_field(model: Model, axis: Point, mask: np.ndarray):
    """The level of each pixel's plane, and a field along the sections."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    phi = np.tensordot(a, model.pos, axes=1)
    gy, gx = np.gradient(phi)
    norm2 = gx * gx + gy * gy + 1e-12
    norm = np.sqrt(norm2)
    field = Field(-gy / norm, gx / norm, mask.astype(np.float64))
    return phi, gx / norm2, gy / norm2, field


def hatch(sheet: Sheet, model: Model, tone: np.ndarray, mask, spec: Hatch, rng):
    """Lay one set of hatching (see :class:`Hatch`) over ``tone``."""
    phi, nx, ny, field = section_field(model, spec.axis, mask)
    where = mask & (tone > spec.lo)
    # seeds, each moved onto the nearest of the planes (which are not quite
    # evenly spaced, as a hand spaces them)
    x, y = jittered_seeds(where, math.sqrt(spec.length * spec.gap * SCALE), rng)
    offset = rng.uniform(0, 1)
    for _ in range(2):
        f = sample(phi, x, y)
        level = spec.gap * (np.round(f / spec.gap - offset) + offset)
        level += rng.normal(0.0, 0.12 * spec.gap, x.size)
        x = x + (level - f) * sample(nx, x, y)
        y = y + (level - f) * sample(ny, x, y)
    half = rng.uniform(0.6, 1.4, x.size) * spec.length / 2.0
    steps = int(math.ceil(spec.length * 0.7 / 2.0))
    fx, fy, fl = trace(field, x, y, steps, 2.0)
    bx, by, bl = trace(field, x, y, steps, -2.0)
    xs = np.concatenate([bx[::-1], fx[1:]])
    ys = np.concatenate([by[::-1], fy[1:]])
    live = np.concatenate([bl[::-1], fl[1:]])
    reach = np.abs(np.arange(-steps, steps + 1) * 2.0)[:, None] <= half[None, :]
    t = sample(tone, xs, ys)
    firm = spec.firm * smoothstep(spec.lo, spec.hi, t) * live * reach
    if rng.uniform() < 0.5:  # strokes all one way, as a hand draws them
        xs, ys, firm = xs[::-1], ys[::-1], firm[::-1]
    return draw_traced(sheet, spec.pen, xs, ys, firm)


# ----------------------------------------------------------------------
# The values


@dataclass
class Values:
    """How dark each ink is to be laid, 0 (paper) to 1."""

    red: np.ndarray
    blue: np.ndarray
    black: np.ndarray


def skin_values(model: Model) -> Values:
    lit = model.light * (0.55 + 0.45 * model.ao)
    shade = np.clip(1.0 - lit, 0.0, 1.0)
    red = np.where(model.skin, 0.15 + 0.8 * shade, 0.0)
    blue = np.where(model.skin, smoothstep(0.25, 0.9, shade), 0.0)
    black = np.where(model.skin, smoothstep(0.75, 1.0, shade), 0.0)
    return Values(red, blue, black)


SKIN_HATCHES = (
    Hatch(RED, (0.0, 1.0, 0.15), 0.2, 70.0, 0.1, 0.5, 0.22),
    Hatch(RED, (0.6, 0.8, 0.0), 0.2, 70.0, 0.35, 0.8, 0.25),
    Hatch(BLUE, (-0.7, 0.7, 0.0), 0.2, 70.0, 0.1, 0.5, 0.22),
    Hatch(BLUE, (0.3, 0.95, 0.0), 0.2, 70.0, 0.4, 0.8, 0.3),
    Hatch(BLUE, (1.0, 0.2, 0.0), 0.2, 60.0, 0.65, 0.95, 0.3),
)


def draw_skin(sheet: Sheet, model: Model, values: Values, rng) -> int:
    count = 0
    for spec in SKIN_HATCHES:
        tone = values.red if spec.pen is RED else values.blue
        count += hatch(sheet, model, tone, model.skin, spec, rng)
    return count
