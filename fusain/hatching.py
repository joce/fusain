"""Shading and rubbing: filling an area with strokes side by side, or
working a tool over it in passes, as a hand does."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np
from murex import spline

#: How far apart the strokes of a shading are, in widths of the tip.
SPACING = 0.55
#: What an unsteady hand (``hand=1``) does to each stroke: how far it lands
#: from its place across the strokes and how far its ends stray along it
#: (in widths of the tip), how much each point of it wanders (the same), and
#: by how much its pressure and its width differ from the next one's.
STRAY_ACROSS = 0.15
STRAY_ALONG = 0.5
WANDER = 0.1
#: Neighbouring passes of a rub are no further apart than this (in widths
#: of the tool), however the hand strays, so that the tool covers the paper
#: between them with more than its rim: two passes that strayed a width
#: apart left a band between them that only the rims reached, rubbed a
#: fifth as much as the rest.
SPREAD = 0.6
#: A stroke does not wander by more than this share of its own length, so
#: that a stub cut short by a corner of the area stays near the hatching's
#: direction (a stroke of two widths or more wanders as far as ever).
WANDER_SHARE = 0.05
#: A hand keeps its firmness from one stroke to the next better than along
#: one, where it lands and lifts (:data:`LANDING`): each stroke is firm by
#: FIRMNESS either way. Near where a tip begins to polish a little
#: firmness is a lot of polish: at 15%, the firmest strokes of a layer at
#: 0.7 polished it four times as far as the lightest, and one layer of oil
#: pastel was uneven by 0.67 of its mean over eight sheets; at 5%, 1.6
#: times as far, and 0.54 (fusain-zhr).
FIRMNESS = 0.05
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
    broad: float | None = None,
    inset: tuple[float, float] = (0.0, 0.0),
    lined: float | None = None,
) -> Iterator[tuple[np.ndarray, float, float]]:
    """The strokes that shade the polygon ``outline``: for each, its path
    (points, (K, 2)), how firm it is against the others (about 1) and the
    width of the tip for it. ``width`` is the width of the tip's point and
    ``broad`` that of its mark across the stroke, wider if the stick is
    tilted: what runs along a stroke goes by the first, where a stroke lands
    among the others by the second. ``inset`` is how far each stroke is
    shortened at its start and at its end, from where the hand puts it (in
    widths of the tip): half a width keeps the round end of a tip inside it.
    Neither end is shortened by more than a quarter of the stroke.
    ``lined`` is what the hand lines each stroke up by against the last, and
    strays by a share of (``broad`` by default): the edge of the band of a
    stick laid flat, which it lays beside the last band, is the stick's
    thickness, and so is how far the band's length varies from stroke to
    stroke (a stray of the band's own length would part the bands or pile
    them up, where they meet by only a share of that thickness)."""
    broad = width if broad is None else broad
    share = 1.0 if lined is None else lined / broad
    turn = math.radians(angle)
    across = np.array([math.sin(turn), math.cos(turn)])
    along = np.array([math.cos(turn), -math.sin(turn)])
    reach = outline @ across
    for offset in np.arange(reach.min() + 0.5 * spacing, reach.max(), spacing):
        offset += rng.normal(0.0, STRAY_ACROSS * broad * share * hand)
        cuts = _crossings(outline, across, offset)
        for start, end in zip(cuts[0::2], cuts[1::2], strict=True):
            start = start + rng.uniform(-1.0, 1.0) * STRAY_ALONG * width * hand * along
            end = end + rng.uniform(-1.0, 1.0) * STRAY_ALONG * width * hand * along
            length = float((end - start) @ along)
            if length < 0.5 * width:
                continue
            start = start + min(inset[0] * width, 0.25 * length) * along
            end = end - min(inset[1] * width, 0.25 * length) * along
            knots = max(int(length / BEND), 2)
            path = start + (end - start) * np.linspace(0.0, 1.0, knots + 1)[:, None]
            # (a short stroke wanders less, from the same draws)
            wander = min(
                WANDER * width * hand, WANDER_SHARE * float((end - start) @ along)
            )
            path += rng.normal(0.0, wander, path.shape)
            firm = rng.uniform(1.0 - FIRMNESS * hand, 1.0 + FIRMNESS * hand)
            # (however unsteady the hand, a tip keeps some width)
            vary = WIDTH * hand * share
            tip = width * max(rng.uniform(1.0 - vary, 1.0 + vary), 0.25)
            yield path, firm, tip


def passes(
    outline: np.ndarray,
    angle: float,
    width: float,
    spacing: float,
    hand: float,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, float]]:
    """The passes of a tool (a stump, a finger, an eraser) ``width`` wide
    rubbed over the polygon ``outline``: for each, its path (points, (K, 2))
    and how firm it is against the others (about 1).

    Someone rubbing a tone with a tool works it up to the edge of the tone
    and turns there, where they can see it; they do not sweep past the edge,
    as a hand hatching with a pencil overshoots its strokes. So the middle
    of the tool keeps half its width inside the outline, and its edge comes
    to the area's edge and no further: every pass runs from that line to
    that line, the outermost passes run along it, and the others lie evenly
    between, no more than ``spacing`` apart. In the middle of the area the
    hand is as unsteady as in a shading (each pass strays across and wanders
    as a stroke does), but it keeps each pass within ``SPREAD`` of a width
    of the one before (or as far apart as they are meant to be, if that is
    further), so that it leaves no band between two passes that only the
    rim of the tool reaches; where the tool meets the edge it is guided by it, as
    steady as anywhere along a pass: the ends of a pass and the passes along
    the edge stray only as a point of a pass wanders (``WANDER``, a fifth of
    how far the ends of a pencil's strokes stray), and only inwards. Where
    the area is narrower than the tool, the tool goes down its middle."""
    margin = 0.5 * width
    sides = _Sides(outline)
    turn = math.radians(angle)
    across = np.array([math.sin(turn), math.cos(turn)])
    along = np.array([math.cos(turn), -math.sin(turn)])
    reach = outline @ across
    lo, hi = reach.min() + margin, reach.max() - margin
    if hi < lo:
        lo = hi = 0.5 * (lo + hi)
    count = int(math.ceil((hi - lo) / spacing)) + 1
    # (no two neighbours further apart than this, nor than they are meant
    # to be, if that is further)
    apart = max(SPREAD * width, (hi - lo) / max(count - 1, 1))
    last = lo
    for k, offset in enumerate(np.linspace(lo, hi, count)):
        if 0 < k < count - 1:  # (the outermost passes go along the edge)
            offset += rng.normal(0.0, STRAY_ACROSS * width * hand)
            # (it strays no further from the pass before it than ``apart``,
            # and leaves the passes after it near enough to reach the last)
            offset = min(offset, hi, last + apart)
            offset = max(offset, lo, hi - (count - 1 - k) * apart)
        last = offset
        cuts = _crossings(outline, across, offset)
        for start, end in zip(cuts[0::2], cuts[1::2], strict=True):
            length = float((end - start) @ along)
            if length < 0.5 * width:
                continue
            start = start + min(margin, 0.25 * length) * along
            end = end - min(margin, 0.25 * length) * along
            knots = max(int(length / BEND), 2)
            path = start + (end - start) * np.linspace(0.0, 1.0, knots + 1)[:, None]
            path += rng.normal(0.0, WANDER * width * hand, path.shape)
            firm = rng.uniform(1.0 - FIRMNESS * hand, 1.0 + FIRMNESS * hand)
            # (the points the hand goes through are kept inside, and so is
            # the curve it draws through them, which can bulge past them)
            path = _kept_inside(path, sides, margin)
            path = spline.as_path(path).sample(max(1.0, width / 8.0))[0]
            yield _kept_inside(path, sides, margin), firm


def _kept_inside(points: np.ndarray, sides: _Sides, margin: float):
    """``points`` moved, each as little as it takes, to ``margin`` or more
    inside the polygon of ``sides``: off the nearest point of its outline,
    and into the corner where it is too near two of its sides. A point that
    cannot be (the polygon is narrower than twice ``margin`` there) stays
    where it was."""
    # (a point is no nearer the outline than the nearest line of a side,
    # which is quick to tell)
    near = sides.lines(points) < margin - 1e-9
    if not near.any():
        return points
    moved = points[near]  # (only those near the outline need to move)
    last = np.full(len(moved), -1)
    for _ in range(3):
        which, depth, normal = sides.nearest(moved)
        short = depth < margin - 1e-9
        if not short.any():
            break
        corner = short & (last >= 0) & (last != which)
        moved[short & ~corner] += ((margin - depth)[:, None] * normal)[short & ~corner]
        moved[corner] = sides.meet(last[corner], which[corner], margin, moved[corner])
        last = np.where(short, which, last)
    else:
        depth = sides.nearest(moved)[1]
    fits = depth >= margin - 1e-6
    kept = points.copy()
    kept[near] = np.where(fits[:, None], moved, points[near])
    return kept


class _Sides:
    """The sides of a polygon, and how far inside it points are."""

    def __init__(self, outline: np.ndarray):
        self.corner = outline
        self.side = np.roll(outline, -1, axis=0) - outline
        side = self.side
        turning = float(np.sum(outline[:, 0] * side[:, 1] - outline[:, 1] * side[:, 0]))
        inward = math.copysign(1.0, turning) * np.stack([-side[:, 1], side[:, 0]], 1)
        self.inward = inward / np.maximum(np.hypot(*inward.T), 1e-12)[:, None]
        self.long = np.maximum((side * side).sum(1), 1e-12)
        self.level = (self.inward * outline).sum(1)

    def lines(self, at: np.ndarray) -> np.ndarray:
        """For each point, how far inside the nearest of the lines the sides
        lie on it is (less than nothing outside one): never more than how
        far inside the outline it is."""
        return (at @ self.inward.T - self.level[None]).min(axis=1)

    def nearest(self, at: np.ndarray):
        """For each point: its nearest side, how far inside the outline it
        is (less than nothing outside), and the way in from the nearest
        point of the outline (a unit vector)."""
        rel = at[:, None, :] - self.corner[None]
        t = np.clip((rel * self.side).sum(2) / self.long, 0.0, 1.0)
        off = rel - t[..., None] * self.side
        gap = np.hypot(off[..., 0], off[..., 1])
        which = np.argmin(gap, axis=1)
        rows = np.arange(len(at))
        away, gap = off[rows, which], gap[rows, which]
        # (even-odd: inside if a ray to the right crosses the outline an odd
        # number of times)
        y0, y1 = self.corner[:, 1], self.corner[:, 1] + self.side[:, 1]
        cut = (y0[None] > at[:, 1:2]) != (y1[None] > at[:, 1:2])
        share = (at[:, 1:2] - y0[None]) / np.where(cut, self.side[:, 1][None], 1.0)
        right = self.corner[:, 0][None] + share * self.side[:, 0][None] > at[:, 0:1]
        sign = np.where((cut & right).sum(1) % 2 == 1, 1.0, -1.0)
        normal = np.where(
            (gap > 1e-9)[:, None],
            sign[:, None] * away / np.maximum(gap, 1e-12)[:, None],
            self.inward[which],
        )
        return which, sign * gap, normal

    def meet(self, one, other, margin: float, at: np.ndarray) -> np.ndarray:
        """Where the lines ``margin`` inside the sides ``one`` and ``other``
        cross (``at``, where they are parallel)."""
        a, b = self.inward[one], self.inward[other]
        ca = (a * self.corner[one]).sum(1) + margin
        cb = (b * self.corner[other]).sum(1) + margin
        det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        ok = np.abs(det) > 1e-9
        safe = np.where(ok, det, 1.0)
        x = (ca * b[:, 1] - cb * a[:, 1]) / safe
        y = (a[:, 0] * cb - b[:, 0] * ca) / safe
        return np.where(ok[:, None], np.stack([x, y], 1), at)


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
