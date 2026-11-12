"""Shading: filling an area with strokes side by side, as a hand does."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np

#: How far apart the strokes of a shading are, in widths of the tip.
SPACING = 0.55
#: What an unsteady hand (``hand=1``) does to each stroke: how far it lands
#: from its place across the strokes and how far its ends stray along it
#: (in widths of the tip), how much each point of it wanders (the same), and
#: by how much its pressure and its width differ from the next one's.
STRAY_ACROSS = 0.15
STRAY_ALONG = 0.5
WANDER = 0.1
FIRMNESS = 0.15
WIDTH = 0.1
#: A stroke is lighter where the pencil lands and where it lifts: the share
#: of its pressure it has lost at either end.
LANDING = 0.45
#: A stroke bends about once in this many pixels.
BEND = 25.0

#: The pressure of a shading: a number, or a function of where on the sheet
#: (called with arrays of x and of y).
Pressure = float | Callable[[np.ndarray, np.ndarray], np.ndarray]


def as_outline(area) -> np.ndarray:
    """``area`` as the corners of a polygon, (N, 2): a rectangle given as
    (x0, y0, x1, y1), or three or more points."""
    try:
        points = np.asarray(area, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(_AREA.format(area)) from None
    if points.shape == (4,):
        x0, y0, x1, y1 = points
        points = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(_AREA.format(area))
    if not np.isfinite(points).all():
        raise ValueError(f"the corners of an area must be finite numbers: {area!r}")
    return points


_AREA = "an area is a rectangle (x0, y0, x1, y1) or three or more points, got {!r:.80}"


def _crossings(outline: np.ndarray, across: np.ndarray, offset: float) -> np.ndarray:
    """Where the line ``offset`` along ``across`` cuts the outline: the points,
    (M, 2), M even, in the order they come along the line."""
    here = outline @ across - offset
    there = np.roll(here, -1)
    cut = (here < 0) != (there < 0)
    share = (here[cut] / (here[cut] - there[cut]))[:, None]
    points = outline[cut] + share * (np.roll(outline, -1, axis=0)[cut] - outline[cut])
    along = np.array([across[1], -across[0]])
    return points[np.argsort(points @ along)]


def strokes(
    outline: np.ndarray,
    angle: float,
    width: float,
    spacing: float,
    hand: float,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, float, float]]:
    """The strokes that shade the polygon ``outline``: for each, its path
    (points, (K, 2)), how firm it is against the others (about 1) and the
    width of the tip for it."""
    turn = math.radians(angle)
    across = np.array([math.sin(turn), math.cos(turn)])
    along = np.array([math.cos(turn), -math.sin(turn)])
    reach = outline @ across
    for offset in np.arange(reach.min() + 0.5 * spacing, reach.max(), spacing):
        offset += rng.normal(0.0, STRAY_ACROSS * width * hand)
        cuts = _crossings(outline, across, offset)
        for start, end in zip(cuts[0::2], cuts[1::2], strict=True):
            start = start + rng.uniform(-1.0, 1.0) * STRAY_ALONG * width * hand * along
            end = end + rng.uniform(-1.0, 1.0) * STRAY_ALONG * width * hand * along
            length = float((end - start) @ along)
            if length < 0.5 * width:
                continue
            knots = max(int(length / BEND), 2)
            path = start + (end - start) * np.linspace(0.0, 1.0, knots + 1)[:, None]
            path += rng.normal(0.0, WANDER * width * hand, path.shape)
            firm = rng.uniform(1.0 - FIRMNESS * hand, 1.0 + FIRMNESS * hand)
            tip = width * rng.uniform(1.0 - WIDTH * hand, 1.0 + WIDTH * hand)
            yield path, firm, tip


def bearing(pressure: Pressure, path: np.ndarray, firm: float, hand: float):
    """The pressure along one stroke of a shading, as a function of u (0 at
    its start, 1 at its end): what the shading asks for where the stroke is,
    as firm as this stroke is, and lighter at either end."""
    start, end = path[0], path[-1]

    def along(u):
        u = np.asarray(u, dtype=np.float64)
        if callable(pressure):
            x, y = (start[k] + (end[k] - start[k]) * u for k in (0, 1))
            wanted = np.asarray(pressure(x, y), dtype=np.float64)
        else:
            wanted = float(pressure)
        lift = 1.0 - np.sqrt(np.sin(np.pi * np.clip(u, 0.02, 0.98)))
        return wanted * firm * (1.0 - LANDING * min(hand, 1.0) * lift)

    return along
