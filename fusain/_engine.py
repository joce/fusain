"""The compiled kernels that lay a medium on the paper, and their constants.

Kernels are ``@njit(cache=True, error_model="numpy")``, without ``fastmath``
and without ``parallel``, so that a drawing repeats exactly.
"""

from __future__ import annotations

import math

import numpy as np
from murex.spectral import N_BANDS
from numba import njit

#: Pigment is packed per pixel as [K (N bands), S, volume], all extensive
#: (summed over the pigment present); the held layer has a fourth part, the
#: volume of wax in it.
N = N_BANDS
IS, IV, IW = N, N + 1, N + 2
LOOSE, HELD = N + 2, N + 3

#: How far down the tooth a tip reaches at full pressure, as a share of a
#: tooth of depth 1: the hardest lead, and what the softest adds to it. Only
#: the softest leads pressed hard reach every valley: a firm HB line still
#: has white specks in it.
REACH_HARD = 0.26
REACH_SOFT = 1.13
#: A light touch reaches further than in proportion (the paper gives and the
#: tip rides on its fibres): the reach goes as the pressure to this power.
REACH_POWER = 0.6
#: Share of that reach a paper with no give leaves, and what full give adds.
GIVE_NONE = 0.6
GIVE_FULL = 0.8
#: A pixel is not level: its surface lies within this much of its height
#: (in tooth units), so a tip that reaches near it touches part of it.
MICRO = 0.2
#: The tip bears hardest on what stands highest: ground this far (in tooth
#: units) above the level the tip reaches gets a full pass's worth, lower
#: ground in proportion. That is the grain inside a line.
SPAN = 0.8
#: No two passes meet the paper alike: each shifts the level it reaches at
#: each pixel by up to half of this, either way.
JITTER = 0.3
#: A tip is flat across, and rounded off over this share of its radius (and
#: half a pixel) at the rim.
RIM = 0.35
#: A pencil held on its side marks with the side of its lead: at full tilt
#: the mark is this many times wider than the point's, no more.
SIDE = 5.0
#: Its weight is then spread over more paper: the share of its reach into
#: the tooth it loses at full tilt, and how much more of the tip (the share
#: of what is not rim already) is rounded off, which softens its edges.
TILT_REACH = 0.25
TILT_RIM = 0.7
#: A lead is not even: across the tip it bears up to this much more or less,
#: in bands about STREAK_WIDTH pixels wide that run the length of the stroke.
STREAK = 0.35
STREAK_WIDTH = 1.3
#: Thickness at which the tooth of a paper of depth 1 and bite 1 is full of
#: the softest lead pressed hard, and the share of it the hardest fills.
FILL = 2.6
FILL_HARD = 0.03
#: Thickness the softest lead lays in one pass at full pressure on an empty
#: tooth, on a paper of bite 0.5. A paper's tooth sets how much it holds,
#: not how much a pass lays, so a toothier paper takes more passes to fill.
LAY = 1.7
#: A light touch fills the tooth less: the share of the full thickness that
#: no pressure at all would reach.
FILL_LIGHT = 0.5
#: A paper with hardly any tooth still holds this share of a full one's.
FILL_FLAT = 0.45
#: Share of the dust under a tip that one pass at full pressure presses into
#: the paper, where it is held.
PRESS_IN = 0.15


@njit(cache=True, error_model="numpy")
def unit_noise(ix: int, iy: int, salt: int) -> float:
    """A number in 0..1 for this pixel and this pass, always the same one."""
    h = (ix * 73856093) ^ (iy * 19349663) ^ (salt * 83492791)
    h &= 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True, error_model="numpy")
def heading(xs, ys, i):
    """Which way the tip is going at sample ``i``, as a unit vector."""
    count = xs.shape[0]
    ahead, behind = min(i + 1, count - 1), max(i - 1, 0)
    tx, ty = xs[ahead] - xs[behind], ys[ahead] - ys[behind]
    going = math.sqrt(tx * tx + ty * ty)
    if going > 1e-9:
        return tx / going, ty / going
    return 1.0, 0.0


@njit(cache=True, error_model="numpy")
def streak_at(dx, dy, tx, ty, salt):
    """How much more or less than its share (1) the tip bears at the pixel
    ``dx``, ``dy`` from its centre: by how far across the tip that is, the
    same all along the stroke."""
    across = (dy * tx - dx * ty) / STREAK_WIDTH + 64.0
    band = int(math.floor(across))
    blend = across - band
    streak = unit_noise(band, 0, salt + 1) * (1.0 - blend)
    streak += unit_noise(band + 1, 0, salt + 1) * blend
    return 1.0 + STREAK * (2.0 * streak - 1.0)


@njit(cache=True, error_model="numpy")
def lay_stroke(
    tooth,
    cover,
    held,
    loose,
    xs,
    ys,
    pressures,
    steps,
    radius,
    length,
    tilt,
    k,
    s,
    softness,
    binding,
    depth,
    bite,
    give,
    salt,
):
    """Draw one pass of a tip through the points (``xs``, ``ys``), bearing
    with ``pressures`` (0..1) and moving ``steps`` pixels at each. The tip is
    an ellipse, ``radius`` pixels across the stroke and ``length`` along it
    (both half-widths; equal for a point), and ``tilt`` (0..1) says how far
    over on its side the stick is held. Lay the medium (absorption ``k`` per
    band and scattering ``s`` per unit volume, ``softness`` and ``binding``
    0..1) on the paper (``depth``, ``bite``, ``give``) where the tip touches
    its tooth.

    Returns the volume laid, which is the volume worn off the tip.
    """
    height, width = tooth.shape
    reach = (REACH_HARD + REACH_SOFT * softness) * (GIVE_NONE + GIVE_FULL * give)
    reach *= 1.0 - TILT_REACH * tilt
    # (a level paper is touched all over: any pressure reaches all of it)
    reach = reach / depth if depth > 1e-6 else 1e9
    shape = FILL_HARD + (1.0 - FILL_HARD) * softness**1.5
    lay = LAY * shape * (0.5 + bite)
    full = (
        FILL
        * shape
        * (FILL_FLAT + (1.0 - FILL_FLAT) * min(depth, 2.0))
        * (0.5 + 0.5 * bite)
    )
    wide, long = radius + 0.5, length + 0.5
    bound = max(wide, long)
    rounded = RIM + (1.0 - RIM) * TILT_RIM * tilt
    # a pixel on the path of the tip's centre gets one pass's worth in all
    per_pass = 1.0 / (2.0 * long - (rounded * long + 0.5))
    total = 0.0
    for i in range(xs.shape[0]):
        pressure = pressures[i]
        step = steps[i]
        if pressure <= 0.0 or step <= 0.0:
            continue
        x, y = xs[i], ys[i]
        tx, ty = heading(xs, ys, i)
        x0 = max(int(math.floor(x - bound)), 0)
        x1 = min(int(math.ceil(x + bound)), width)
        y0 = max(int(math.floor(y - bound)), 0)
        y1 = min(int(math.ceil(y + bound)), height)
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dx, dy = ix + 0.5 - x, iy + 0.5 - y
                # how far out in the tip this is (1 at its edge), and how
                # far in from the edge, in pixels
                forward, sideways = dx * tx + dy * ty, dy * tx - dx * ty
                out = math.sqrt((forward / long) ** 2 + (sideways / wide) ** 2)
                if out >= 1.0:
                    continue
                d = math.sqrt(dx * dx + dy * dy)
                across = d / out if out > 1e-9 else min(wide, long)
                inside = across - d
                edge = min(inside, 1.0)
                local = pressure * min(inside / (rounded * across + 0.5), 1.0)
                local *= streak_at(dx, dy, tx, ty, salt)
                level = 1.0 - reach * local**REACH_POWER
                level += JITTER * (unit_noise(ix, iy, salt) - 0.5)
                above = tooth[iy, ix] + MICRO - level
                touch = min(max(above / (2.0 * MICRO), 0.0), 1.0) * edge
                if touch <= 0.0:
                    continue
                covered = max(cover[iy, ix], touch)
                cover[iy, ix] = covered
                thick = (held[iy, ix, IV] + loose[iy, ix, IV]) / covered
                bearing = min(local, 1.0)
                # the tip presses some of the dust under it into the paper
                pressed = min(PRESS_IN * bearing * per_pass * step, 1.0) * touch
                for part in range(LOOSE):
                    moved = loose[iy, ix, part] * pressed
                    loose[iy, ix, part] -= moved
                    held[iy, ix, part] += moved
                room = full * (FILL_LIGHT + (1.0 - FILL_LIGHT) * bearing) - thick
                if room <= 0.0:
                    continue
                # (a light pass takes less, but not in proportion: the lead
                # still slides over all it touches)
                dv = lay * math.sqrt(bearing * min(above / SPAN, 1.0)) * (room / full)
                dv = min(dv * per_pass * step, room) * touch
                fixed, dust = dv * binding, dv * (1.0 - binding)
                for band_k in range(N):
                    held[iy, ix, band_k] += fixed * k[band_k]
                    loose[iy, ix, band_k] += dust * k[band_k]
                held[iy, ix, IS] += fixed * s
                held[iy, ix, IV] += fixed
                loose[iy, ix, IS] += dust * s
                loose[iy, ix, IV] += dust
                total += dv
    return total


@njit(cache=True, error_model="numpy")
def rub_stroke(
    cover,
    held,
    loose,
    xs,
    ys,
    pressures,
    steps,
    radius,
    lift_loose,
    lift_held,
    lay,
    press,
    smear,
    salt,
):
    """Rub one pass of a round tool of ``radius`` pixels through the points
    (``xs``, ``ys``), bearing with ``pressures`` (0..1) and moving ``steps``
    pixels at each. Where it bears fully, one pass lifts the shares
    ``lift_loose`` of the dust and ``lift_held`` of the held pigment onto the
    tool, presses the share ``press`` of the dust into the paper and spreads
    the share ``smear`` of it over the pixel; and over each of its own widths
    the tool lays the share ``lay`` of what it carries back down, as dust.

    Returns what the tool carries at the end: [K (N bands), S, volume].
    """
    height, width = cover.shape
    bound = radius + 0.5
    rim = RIM * bound + 0.5
    per_pass = 1.0 / (2.0 * bound - rim)
    # of what it carries, the share the tool lays on a pixel it bears fully
    # on, per pixel moved
    per_pixel = lay / (2.0 * bound * math.pi * bound * bound)
    load = np.zeros(LOOSE)
    for i in range(xs.shape[0]):
        pressure = pressures[i]
        step = steps[i]
        if pressure <= 0.0 or step <= 0.0:
            continue
        x, y = xs[i], ys[i]
        tx, ty = heading(xs, ys, i)
        x0 = max(int(math.floor(x - bound)), 0)
        x1 = min(int(math.ceil(x + bound)), width)
        y0 = max(int(math.floor(y - bound)), 0)
        y1 = min(int(math.ceil(y + bound)), height)
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dx, dy = ix + 0.5 - x, iy + 0.5 - y
                d = math.sqrt(dx * dx + dy * dy)
                edge = min(bound - d, 1.0)
                if edge <= 0.0:
                    continue
                local = pressure * min((bound - d) / rim, 1.0) * edge
                local *= streak_at(dx, dy, tx, ty, salt)
                share = local * per_pass * step  # of one pass at full pressure
                # lift
                up_loose = min(lift_loose * share, 1.0)
                up_held = min(lift_held * share, 1.0)
                for part in range(LOOSE):
                    taken = loose[iy, ix, part] * up_loose
                    loose[iy, ix, part] -= taken
                    load[part] += taken
                    taken = held[iy, ix, part] * up_held
                    held[iy, ix, part] -= taken
                    load[part] += taken
                held[iy, ix, IW] -= held[iy, ix, IW] * up_held
                # lay, as dust all over the pixel
                before = held[iy, ix, IV] + loose[iy, ix, IV]
                down = min(per_pixel * local * step, 1.0)
                laid = load[IV] * down
                for part in range(LOOSE):
                    given = load[part] * down
                    load[part] -= given
                    loose[iy, ix, part] += given
                after = before + laid
                if after <= 1e-9:
                    cover[iy, ix] = 0.0  # rubbed clean
                    continue
                bare = 1.0 - cover[iy, ix]
                cover[iy, ix] += bare * (laid / after)
                # press in, and push what is loose into the low ground
                pressed = min(press * share, 1.0)
                for part in range(LOOSE):
                    moved = loose[iy, ix, part] * pressed
                    loose[iy, ix, part] -= moved
                    held[iy, ix, part] += moved
                dusty = loose[iy, ix, IV] / after
                cover[iy, ix] += (1.0 - cover[iy, ix]) * min(smear * share, 1.0) * dusty
    return load


def as_kernel_arrays(points: np.ndarray, pressure: np.ndarray, spacing: float):
    """The arguments :func:`lay_stroke` takes for a sampled path: contiguous
    float64 x, y, pressure, and the distance moved at each sample (half a
    spacing at either end; one spacing for a single point, a dot)."""
    xs = np.ascontiguousarray(points[:, 0], dtype=np.float64)
    ys = np.ascontiguousarray(points[:, 1], dtype=np.float64)
    steps = np.full(xs.shape, float(spacing))
    if xs.size > 1:
        steps[0] = steps[-1] = 0.5 * spacing
    return xs, ys, np.ascontiguousarray(pressure, dtype=np.float64), steps
