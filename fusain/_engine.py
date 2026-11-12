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
#: tooth of depth 1: the hardest lead, and what the softest adds to it.
REACH_HARD = 0.4
REACH_SOFT = 2.2
#: Share of that reach a paper with no give leaves, and what full give adds.
GIVE_NONE = 0.6
GIVE_FULL = 0.8
#: A pixel is not level: its surface lies within this much of its height
#: (in tooth units), so a tip that reaches near it touches part of it.
MICRO = 0.2
#: No two passes meet the paper alike: each shifts the level it reaches at
#: each pixel by up to half of this, either way.
JITTER = 0.3
#: Thickness at which the tooth of a paper of depth 1 and bite 1 is full of
#: the softest lead pressed hard, and the share of it the hardest fills.
FILL = 2.1
FILL_HARD = 0.03
#: Thickness the softest lead lays in one pass at full pressure on an empty
#: tooth, on a paper of bite 0.5. A paper's tooth sets how much it holds,
#: not how much a pass lays, so a toothier paper takes more passes to fill.
LAY = 1.1
#: A light touch fills the tooth less: the share of the full thickness that
#: no pressure at all would reach.
FILL_LIGHT = 0.5
#: A paper with hardly any tooth still holds this share of a full one's.
FILL_FLAT = 0.45


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
    k,
    s,
    softness,
    binding,
    depth,
    bite,
    give,
    salt,
):
    """Draw one pass of a round tip of ``radius`` pixels through the points
    (``xs``, ``ys``), bearing with ``pressures`` (0..1) and moving ``steps``
    pixels at each: lay the medium (absorption ``k`` per band and scattering
    ``s`` per unit volume, ``softness`` and ``binding`` 0..1) on the paper
    (``depth``, ``bite``, ``give``) where the tip touches its tooth.

    Returns the volume laid, which is the volume worn off the tip.
    """
    height, width = tooth.shape
    reach = (REACH_HARD + REACH_SOFT * softness) * (GIVE_NONE + GIVE_FULL * give)
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
    bound = radius + 0.5
    # a pixel on the path of the tip's centre gets one pass's worth in all
    per_pass = 2.0 / (math.pi * bound)
    total = 0.0
    for i in range(xs.shape[0]):
        pressure = pressures[i]
        step = steps[i]
        if pressure <= 0.0 or step <= 0.0:
            continue
        x, y = xs[i], ys[i]
        x0, x1 = (
            max(int(math.floor(x - bound)), 0),
            min(int(math.ceil(x + bound)), width),
        )
        y0, y1 = (
            max(int(math.floor(y - bound)), 0),
            min(int(math.ceil(y + bound)), height),
        )
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dx, dy = ix + 0.5 - x, iy + 0.5 - y
                d = math.sqrt(dx * dx + dy * dy)
                edge = min(bound - d, 1.0)
                if edge <= 0.0:
                    continue
                # a rounded tip bears hardest at its centre
                local = pressure * math.sqrt(max(1.0 - (d / bound) ** 2, 0.0))
                level = 1.0 - local * reach
                level += JITTER * (unit_noise(ix, iy, salt) - 0.5)
                touch = (tooth[iy, ix] + MICRO - level) / (2.0 * MICRO)
                touch = min(max(touch, 0.0), 1.0) * edge
                if touch <= 0.0:
                    continue
                covered = max(cover[iy, ix], touch)
                thick = (held[iy, ix, IV] + loose[iy, ix, IV]) / covered
                room = full * (FILL_LIGHT + (1.0 - FILL_LIGHT) * local) - thick
                if room <= 0.0:
                    cover[iy, ix] = covered
                    continue
                # (a light pass takes less, but not in proportion: the lead
                # still slides over all it touches)
                dv = lay * math.sqrt(local) * (room / full) * per_pass * step
                dv = min(dv, room) * touch
                cover[iy, ix] = covered
                fixed, dust = dv * binding, dv * (1.0 - binding)
                for band in range(N):
                    held[iy, ix, band] += fixed * k[band]
                    loose[iy, ix, band] += dust * k[band]
                held[iy, ix, IS] += fixed * s
                held[iy, ix, IV] += fixed
                loose[iy, ix, IS] += dust * s
                loose[iy, ix, IV] += dust
                total += dv
    return total


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
