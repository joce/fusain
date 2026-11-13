"""The compiled kernels that lay a medium on the paper, and their constants.

Kernels are ``@njit(cache=True, error_model="numpy")``, without ``fastmath``
and without ``parallel``, so that a drawing repeats exactly.
"""

from __future__ import annotations

import math

import numpy as np
from murex.spectral import N_BANDS
from numba import njit

#: Pigment is packed per pixel as [K (N bands), S, volume, sheen], all
#: extensive (summed over the pigment present; the sheen is the volume times
#: the sheen of its medium); the held layer has a fifth part, the volume of
#: wax in it. There are three layers: what the paper holds, and
#: over it the dust, in two layers (``under`` and, on top, ``loose``) so that
#: what was laid last is seen over what was laid before.
N = N_BANDS
IS, IV, IX, IW = N, N + 1, N + 2, N + 3
LOOSE, HELD = N + 3, N + 4

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
#: A point wears to a flat as it draws, and its line widens: the cube of the
#: flat's width, in pixels, grows by this much for each unit of volume the
#: tip lays (a cone loses volume as the cube of where it is cut).
WEAR = 0.08
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
#: Dust piles up where a film would not: a medium that is all dust fills the
#: tooth with this much more again than one the paper holds entirely.
PILE = 1.0
#: A paper with hardly any tooth still holds this share of a full one's.
FILL_FLAT = 0.45
#: Share of the dust under a tip that one pass at full pressure presses into
#: the paper, where it is held.
PRESS_IN = 0.15
#: A pass lays its dust on top of the dust that was there, which goes under
#: it: all but a little of it (the tip stirs some up) once the new dust is
#: this thick, in proportion if it is thinner and does not cover it all.
COVERS = 0.2
BURY = 0.985
#: Wax films over the tooth: a film this thick is as slick as it gets, and a
#: pass of a waxy medium then lays only this much less than on bare paper
#: (resist).
WAX_FULL = 0.35
RESIST = 0.85
#: Dust that fixative has set takes only this share of the room in the tooth
#: that it took loose: a fixed layer has some tooth of its own.
FIXED_ROOM = 0.5
#: A dry medium, which the tooth has to scrape off, hardly takes on wax at
#: all; a medium with this share of wax or more counts as waxy.
RESIST_DRY = 0.97
STICKS = 0.3
#: A waxy medium softer than FIRM is a greasy paste, which sticks to wax: the
#: softest loses this share of its resist.
FIRM = 0.6
GREASE = 0.8
#: Pressed harder than BURNISH_FROM, a tip polishes a waxy layer flat: one
#: pass at full pressure on a film as slick as it gets raises ``press`` by
#: BURNISH, and a pressed tooth is level, so the white specks close up.
BURNISH_FROM = 0.6
BURNISH = 1.5
#: Polishing also smears the waxy layer: of what the paper holds there, the
#: tip drags along this share at full pressure in one pass, and lays this
#: many times what it drags back down over each of its own widths, valleys
#: included: within a pixel or two, so that it closes the white specks next
#: to it without carrying colour far.
BURNISH_LIFT = 0.6
BURNISH_LAY = 4.0
#: A ballpoint pen. Its ball is hard and small: at full pressure it reaches
#: this far down the tooth (times what the paper gives, as a tip does), and
#: a light touch reaches nearly as far (far more than in proportion): a
#: light line is thinner and paler, but it is still a line.
BALL_REACH = 3.0
BALL_POWER = 0.3
#: A ball is round: from its middle to its rim its surface rises this far
#: off the paper, in depths of the tooth, so a light line is narrower than
#: the ball.
BALL_ROUND = 2.5
#: Pass after pass the ball meets the paper more alike than a stick does:
#: the share of a stick's JITTER it has.
BALL_JITTER = 0.5
#: The ball lays its ink at a steady rate: one pass at full pressure leaves a
#: film this thick where it touches, and the lightest pass this share of it.
BALL_INK = 0.03
BALL_LIGHT = 0.15
#: The ink does not come quite evenly: by this share either way, over
#: lengths of about BALL_SURGE pixels.
BALL_UNEVEN = 0.2
BALL_SURGE = 12.0
#: The ball skids on a film of pigment: it lays less from the first of these
#: thicknesses (per covered area) and nothing from the second. On wax it
#: writes as badly as a dry medium does (RESIST_DRY).
SKID_FROM = 0.25
SKID_FULL = 1.0
#: A tool that rubs also spreads the dust under it from grain to grain: what
#: it takes onto its face that way (as strongly as it smears) it lays back
#: this many times over each of its own widths, so within a few pixels. That
#: is what evens a rubbed tone before the dust is worked into the paper.
SPREAD = 4.0
#: It spreads the dust this many times as readily as it stirs it.
SPREADS = 3.0
#: A tool that rubs reaches the dust underneath less than the dust on top:
#: the share of its lift it has there.
UNDER_LIFT = 0.7


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
def tip_bearing(dx, dy, tx, ty, wide, long, rounded):
    """How a tip bears on the pixel ``dx``, ``dy`` from its centre, going the
    way (``tx``, ``ty``): the share of its pressure it has there, and how far
    inside its edge that is, in pixels up to 1. The tip is an ellipse,
    ``wide`` across the stroke and ``long`` along it (half-widths), level but
    for the share ``rounded`` of it at the rim. (0, 0) outside it."""
    forward, sideways = dx * tx + dy * ty, dy * tx - dx * ty
    out = math.sqrt((forward / long) ** 2 + (sideways / wide) ** 2)
    if out >= 1.0:
        return 0.0, 0.0
    d = math.sqrt(dx * dx + dy * dy)
    across = d / out if out > 1e-9 else min(wide, long)
    inside = across - d
    return min(inside / (rounded * across + 0.5), 1.0), min(inside, 1.0)


@njit(cache=True, error_model="numpy")
def free_share(fixed, layer, iy, ix, which):
    """The share of a layer of dust at a pixel that fixative has not set.
    ``fixed`` holds, for each pixel, the volume it has set in the dust
    underneath (``which`` 0) and in the dust on top (1)."""
    there = layer[iy, ix, IV]
    if fixed[iy, ix, which] <= 0.0 or there <= 0.0:
        return 1.0
    return 1.0 - min(fixed[iy, ix, which] / there, 1.0)


@njit(cache=True, error_model="numpy")
def press_dust(held, under, loose, fixed, cover, iy, ix, share):
    """Press the ``share`` of the dust at a pixel into the paper: of the
    dust that fixative has not set (``fixed``), and less of it the more a
    layer that is set lies between it and the paper."""
    if loose[iy, ix, IV] <= 0.0 and under[iy, ix, IV] <= 0.0:
        return  # (no dust here: most pixels of a first pass)
    barrier = fixed[iy, ix, 0] + fixed[iy, ix, 1]
    if barrier > 0.0:
        share *= 1.0 - min(barrier / max(cover[iy, ix], 1e-6) / COVERS, 1.0)
    from_loose = share * free_share(fixed, loose, iy, ix, 1)
    from_under = share * free_share(fixed, under, iy, ix, 0)
    for part in range(LOOSE):
        moved = loose[iy, ix, part] * from_loose
        loose[iy, ix, part] -= moved
        held[iy, ix, part] += moved
        moved = under[iy, ix, part] * from_under
        under[iy, ix, part] -= moved
        held[iy, ix, part] += moved


@njit(cache=True, error_model="numpy")
def burnish(held, press, drag, iy, ix, polish, lay):
    """Polish a pixel by ``polish`` (the share of one full pass on a film as
    slick as it gets): press its tooth flat, take that share of the lift of
    what it holds onto the tip (``drag``, packed as the held layer is), and
    lay the share ``lay`` of what the tip drags back down on it."""
    press[iy, ix] = min(press[iy, ix] + BURNISH * polish, 1.0)
    up = min(BURNISH_LIFT * polish, 1.0)
    down = min(lay, 1.0)
    for part in range(HELD):
        taken = held[iy, ix, part] * up
        drag[part] += taken
        given = drag[part] * down
        drag[part] -= given
        held[iy, ix, part] += given - taken


@njit(cache=True, error_model="numpy")
def set_down(held, cover, drag, x, y, tx, ty, wide, long, rounded):
    """Where the tip lifts off, at (``x``, ``y``), lay what it still drags
    (``drag``, packed as the held layer is) on the marked paper under it.
    Returns the volume it could not lay: all of it, if no marked paper of
    the sheet is under the tip."""
    if drag[IV] <= 0.0:
        return 0.0
    height, width = cover.shape
    bound = max(wide, long)
    x0 = max(int(math.floor(x - bound)), 0)
    x1 = min(int(math.ceil(x + bound)), width)
    y0 = max(int(math.floor(y - bound)), 0)
    y1 = min(int(math.ceil(y + bound)), height)
    under = 0.0
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            if cover[iy, ix] > 0.0:
                under += tip_bearing(
                    ix + 0.5 - x, iy + 0.5 - y, tx, ty, wide, long, rounded
                )[1]
    if under <= 0.0:
        return drag[IV]
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            if cover[iy, ix] > 0.0:
                edge = tip_bearing(
                    ix + 0.5 - x, iy + 0.5 - y, tx, ty, wide, long, rounded
                )[1]
                for part in range(HELD):
                    held[iy, ix, part] += drag[part] * edge / under
    return 0.0


@njit(cache=True, error_model="numpy")
def settle_dust(fresh, cover, fixed, under, loose, k, s, sheen, xs, ys, bound):
    """At the end of a pass, put its dust (``fresh``, a volume per pixel) on
    top of the dust that was there, which goes under it, and leave ``fresh``
    empty again; the dust that goes under takes with it what fixative has
    set of it (``fixed``). Only where the tip went: within ``bound`` of the points
    (``xs``, ``ys``)."""
    height, width = fresh.shape
    for i in range(xs.shape[0]):
        x0 = max(int(math.floor(xs[i] - bound)), 0)
        x1 = min(int(math.ceil(xs[i] + bound)), width)
        y0 = max(int(math.floor(ys[i] - bound)), 0)
        y1 = min(int(math.ceil(ys[i] + bound)), height)
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dust = fresh[iy, ix]
                if dust <= 0.0:
                    continue
                fresh[iy, ix] = 0.0  # settled: the next point skips it
                buried = BURY * min(dust / cover[iy, ix] / COVERS, 1.0)
                for part in range(LOOSE):
                    moved = loose[iy, ix, part] * buried
                    loose[iy, ix, part] -= moved
                    under[iy, ix, part] += moved
                # (what fixative has set of it goes under with it)
                moved = fixed[iy, ix, 1] * buried
                fixed[iy, ix, 1] -= moved
                fixed[iy, ix, 0] += moved
                for band in range(N):
                    loose[iy, ix, band] += dust * k[band]
                loose[iy, ix, IS] += dust * s
                loose[iy, ix, IV] += dust
                loose[iy, ix, IX] += dust * sheen


@njit(cache=True, error_model="numpy")
def resist_of(softness, wax):
    """How much less of a medium a film of wax as slick as it gets takes than
    bare paper does."""
    waxy = min(wax / STICKS, 1.0)
    greasy = min(max((softness - FIRM) / (1.0 - FIRM), 0.0), 1.0)
    return RESIST_DRY * (1.0 - waxy) + RESIST * (1.0 - GREASE * greasy) * waxy


@njit(cache=True, error_model="numpy")
def lay_stroke(
    tooth,
    press,
    cover,
    held,
    under,
    loose,
    fresh,
    fixed,
    xs,
    ys,
    pressures,
    steps,
    tip,
    k,
    medium,
    paper,
    salt,
):
    """Draw one pass of a tip through the points (``xs``, ``ys``), bearing
    with ``pressures`` (0..1) and moving ``steps`` pixels at each.

    ``tip`` is [the width of its flat, the full width of the lead (the flat
    never gets wider), how fast it wears (:data:`WEAR`, or 0 for a tip that
    keeps its width), the volume it has worn since it was sharpened, how many
    times wider than its flat it marks (1 for a point, more on its side), its
    tilt (0..1)]. It is an ellipse, as long along the stroke as its flat is
    wide.

    ``medium`` is [scattering per unit volume, softness, binding, the share
    of wax in it, its sheen] (its absorption per band is ``k``) and ``paper`` is
    [depth, bite, give, hold]. The medium is laid where the tip touches the
    tooth.
    What the paper does not hold of it is dust, gathered in ``fresh`` (empty
    before and after) and laid at the end of the pass on top of the dust that
    was there (``loose``), which goes ``under`` it. Dust that fixative has set
    (``fixed``, its volume in each of the two layers) is not pressed in, and
    takes less of the room in the tooth. Wax films over the tooth,
    which then takes less (:func:`resist_of`); pressed hard, the tip polishes
    it flat (``press``) and smears it along.

    Returns the volume the tip has lost: what it laid, less what it has
    dragged off the sheet.
    """
    height, width = tooth.shape
    flat, lead, wear, worn, spread, tilt = (
        tip[0],
        tip[1],
        tip[2],
        tip[3],
        tip[4],
        tip[5],
    )
    s, softness, binding, wax = medium[0], medium[1], medium[2], medium[3]
    sheen = medium[4]
    depth, bite, give = paper[0], paper[1], paper[2]
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
        * (1.0 + PILE * (1.0 - binding))
        * paper[3]
    )
    rounded = RIM + (1.0 - RIM) * TILT_RIM * tilt
    resist = resist_of(softness, wax)
    start = flat**3
    reached = 0.0  # the furthest the tip has been from its path
    # what a tip that polishes drags along
    drag = np.zeros(HELD)
    total = 0.0
    # where the tip last bore, and how wide it was there
    at_x, at_y, at_tx, at_ty = xs[0], ys[0], 1.0, 0.0
    wide, long = 0.5 * flat * spread + 0.5, 0.5 * flat + 0.5
    for i in range(xs.shape[0]):
        pressure = pressures[i]
        step = steps[i]
        if pressure <= 0.0 or step <= 0.0:
            continue
        x, y = xs[i], ys[i]
        tx, ty = heading(xs, ys, i)
        at_x, at_y, at_tx, at_ty = x, y, tx, ty
        if wear > 0.0:  # the flat grows with what the tip has laid
            flat = min((start + wear * max(worn + total, 0.0)) ** (1.0 / 3.0), lead)
        wide, long = 0.5 * flat * spread + 0.5, 0.5 * flat + 0.5
        bound = max(wide, long)
        reached = max(reached, bound)
        # a pixel on the path of the tip's centre gets one pass's worth in
        # all; and of what it drags, the tip lays this share on a pixel it
        # bears on, per pixel moved
        per_pass = 1.0 / (2.0 * long - (rounded * long + 0.5))
        dragged = BURNISH_LAY / (2.0 * wide * math.pi * wide * long)
        x0 = max(int(math.floor(x - bound)), 0)
        x1 = min(int(math.ceil(x + bound)), width)
        y0 = max(int(math.floor(y - bound)), 0)
        y1 = min(int(math.ceil(y + bound)), height)
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dx, dy = ix + 0.5 - x, iy + 0.5 - y
                share, edge = tip_bearing(dx, dy, tx, ty, wide, long, rounded)
                if edge <= 0.0:
                    continue
                local = pressure * share * streak_at(dx, dy, tx, ty, salt)
                level = 1.0 - reach * local**REACH_POWER
                level += JITTER * (unit_noise(ix, iy, salt) - 0.5)
                # (a tooth pressed flat is level with its peaks)
                ground = tooth[iy, ix] + (1.0 - tooth[iy, ix]) * press[iy, ix]
                above = ground + MICRO - level
                touch = min(max(above / (2.0 * MICRO), 0.0), 1.0) * edge
                if touch <= 0.0:
                    continue
                covered = max(cover[iy, ix], touch)
                cover[iy, ix] = covered
                there = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
                thick = (there + fresh[iy, ix]) / covered
                thick -= (
                    (1.0 - FIXED_ROOM) * (fixed[iy, ix, 0] + fixed[iy, ix, 1]) / covered
                )
                bearing = min(local, 1.0)
                passing = per_pass * step * touch
                waxy = min(held[iy, ix, IW] / covered / WAX_FULL, 1.0)
                if bearing > BURNISH_FROM:
                    firm = (bearing - BURNISH_FROM) / (1.0 - BURNISH_FROM)
                    polish, lay_back = firm * waxy * passing, dragged * step * touch
                    burnish(held, press, drag, iy, ix, polish, lay_back)
                # the tip presses some of the dust under it into the paper
                pressed = min(PRESS_IN * bearing, 1.0) * passing
                press_dust(held, under, loose, fixed, cover, iy, ix, pressed)
                room = full * (FILL_LIGHT + (1.0 - FILL_LIGHT) * bearing) - thick
                if room <= 0.0:
                    continue
                # (a light pass takes less, but not in proportion: the lead
                # still slides over all it touches)
                dv = lay * math.sqrt(bearing * min(above / SPAN, 1.0)) * (room / full)
                dv *= 1.0 - resist * waxy
                dv = min(dv * per_pass * step, room) * touch
                kept = dv * binding
                for band in range(N):
                    held[iy, ix, band] += kept * k[band]
                held[iy, ix, IS] += kept * s
                held[iy, ix, IV] += kept
                held[iy, ix, IX] += kept * sheen
                held[iy, ix, IW] += dv * wax
                fresh[iy, ix] += dv - kept
                total += dv
    if binding < 1.0:  # (else there is no dust to settle)
        settle_dust(fresh, cover, fixed, under, loose, k, s, sheen, xs, ys, reached)
    # what the tip still drags it leaves where it lifts off; only off the
    # sheet does it leave the paper with it
    return total - set_down(
        held, cover, drag, at_x, at_y, at_tx, at_ty, wide, long, rounded
    )


@njit(cache=True, error_model="numpy")
def spread_dust(loose, near, iy, ix, take, back):
    """Take the share ``take`` of the dust on top at a pixel onto the face of
    the tool (``near``, packed as dust is), and lay the share ``back`` of
    what is on its face down on the pixel. Returns the volumes laid and
    taken."""
    given, taken = 0.0, 0.0
    for part in range(LOOSE):
        up = loose[iy, ix, part] * take
        down = (near[part] + up) * back
        near[part] += up - down
        loose[iy, ix, part] += down - up
        if part == IV:
            given, taken = down, up
    return given, taken


@njit(cache=True, error_model="numpy")
def rub_stroke(
    cover,
    held,
    under,
    loose,
    fixed,
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
    ``lift_loose`` of the dust (less of the dust underneath) and
    ``lift_held`` of the held pigment onto the tool, presses the share
    ``press`` of the dust into the paper, and stirs the share ``smear`` of
    the dust underneath up into the dust on top and spreads it over the
    pixel; and over each of its own widths the tool lays the share ``lay`` of
    what it carries back down, as dust on top. As strongly as it smears, it
    also spreads the dust on top over the pixels around. Dust that fixative
    has set
    (``fixed``, its volume in each of the two layers) is lifted only as held
    pigment is, and is neither pressed in nor stirred.

    Returns what the tool carries at the end: [K (N bands), S, volume, sheen].
    """
    height, width = cover.shape
    bound = radius + 0.5
    rim = RIM * bound + 0.5
    per_pass = 1.0 / (2.0 * bound - rim)
    # of what it carries, the share the tool lays on a pixel it bears fully
    # on, per pixel moved
    per_pixel = lay / (2.0 * bound * math.pi * bound * bound)
    per_near = SPREAD / (2.0 * bound * math.pi * bound * bound)
    load = np.zeros(LOOSE)
    near = np.zeros(LOOSE)  # the dust on the face of the tool, soon laid back
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
                up_held = min(lift_held * share, 1.0)
                free = free_share(fixed, loose, iy, ix, 1)
                up_loose = min(lift_loose * share, 1.0) * free
                up_loose += up_held * (1.0 - free)
                free = free_share(fixed, under, iy, ix, 0)
                up_under = min(UNDER_LIFT * lift_loose * share, 1.0) * free
                up_under += up_held * (1.0 - free)
                fixed[iy, ix, 0] -= fixed[iy, ix, 0] * up_held
                fixed[iy, ix, 1] -= fixed[iy, ix, 1] * up_held
                for part in range(LOOSE):
                    taken = loose[iy, ix, part] * up_loose
                    loose[iy, ix, part] -= taken
                    load[part] += taken
                    taken = under[iy, ix, part] * up_under
                    under[iy, ix, part] -= taken
                    load[part] += taken
                    taken = held[iy, ix, part] * up_held
                    held[iy, ix, part] -= taken
                    load[part] += taken
                held[iy, ix, IW] -= held[iy, ix, IW] * up_held
                # lay, as dust all over the pixel
                before = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
                down = min(per_pixel * local * step, 1.0)
                laid = load[IV] * down
                for part in range(LOOSE):
                    given = load[part] * down
                    load[part] -= given
                    loose[iy, ix, part] += given
                # spread, from grain to grain
                take = min(SPREADS * smear * share, 1.0)
                take *= free_share(fixed, loose, iy, ix, 1)
                back = min(per_near * local * step, 1.0)
                given, taken = spread_dust(loose, near, iy, ix, take, back)
                laid += given
                after = before + laid - taken
                if after <= 1e-9:
                    cover[iy, ix] = 0.0  # rubbed clean
                    continue
                bare = 1.0 - cover[iy, ix]
                cover[iy, ix] += bare * min(laid / after, 1.0)
                # press in; stir up what is underneath; and push what is
                # loose into the low ground
                spent = min(press * share, 1.0)
                press_dust(held, under, loose, fixed, cover, iy, ix, spent)
                stirred = min(smear * share, 1.0)
                moving = stirred * free_share(fixed, under, iy, ix, 0)
                for part in range(LOOSE):
                    moved = under[iy, ix, part] * moving
                    under[iy, ix, part] -= moved
                    loose[iy, ix, part] += moved
                free = under[iy, ix, IV] + loose[iy, ix, IV]
                free -= fixed[iy, ix, 0] + fixed[iy, ix, 1]
                dusty = max(free, 0.0) / after
                cover[iy, ix] += (1.0 - cover[iy, ix]) * stirred * dusty
    for part in range(LOOSE):  # (what is still on its face it carries off)
        load[part] += near[part]
    return load


@njit(cache=True, error_model="numpy")
def ball_grip(cover, held, under, loose, iy, ix):
    """The share of its ink a ball lays at a pixel: all of it on bare paper,
    less where it skids on a film of pigment or of wax."""
    covered = cover[iy, ix]
    if covered <= 0.0:
        return 1.0
    thick = (held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]) / covered
    skid = min(max((thick - SKID_FROM) / (SKID_FULL - SKID_FROM), 0.0), 1.0)
    waxy = min(held[iy, ix, IW] / covered / WAX_FULL, 1.0)
    # (only the covered share of the pixel is slick)
    return (1.0 - skid * covered) * (1.0 - RESIST_DRY * waxy * covered)


@njit(cache=True, error_model="numpy")
def surge_at(travel, salt):
    """How much more or less ink than its share (1) the ball gives after
    rolling ``travel`` pixels."""
    along = travel / BALL_SURGE
    band = int(math.floor(along))
    blend = along - band
    surge = unit_noise(band, 1, salt + 2) * (1.0 - blend)
    surge += unit_noise(band + 1, 1, salt + 2) * blend
    return 1.0 + BALL_UNEVEN * (2.0 * surge - 1.0)


@njit(cache=True, error_model="numpy")
def roll_stroke(
    tooth,
    press,
    cover,
    held,
    under,
    loose,
    ink,
    inked,
    xs,
    ys,
    pressures,
    steps,
    radius,
    k,
    pen,
    paper,
    salt,
):
    """Roll one pass of a ballpoint pen, its ball of ``radius`` pixels,
    through the points (``xs``, ``ys``), bearing with ``pressures`` (0..1)
    and moving ``steps`` pixels at each.

    ``pen`` is [the scattering of its ink per unit volume, how freely it
    flows] (the ink's absorption per band is ``k``) and ``paper`` is [depth,
    bite, give, hold]. The ball lays ink at a steady rate wherever it touches
    the
    paper: how much does not depend on the tooth, as it does for a stick,
    and only a little on the pressure, which sets how much of the tooth the
    ball reaches. The ink is a stain in the paper (``ink``, packed as dust
    is, on the share ``inked`` of each pixel), under whatever the paper
    carries; where that is thick or waxy, the ball skids and lays less.

    Returns the volume of ink laid.
    """
    height, width = tooth.shape
    s, flow = pen[0], pen[1]
    depth, give = paper[0], paper[2]
    reach = BALL_REACH * (GIVE_NONE + GIVE_FULL * give)
    reach = reach / depth if depth > 1e-6 else 1e9
    rise = BALL_ROUND / depth if depth > 1e-6 else 0.0
    bound = radius + 0.5
    per_pass = 1.0 / (2.0 * bound - 1.0)
    travel = 0.0
    total = 0.0
    for i in range(xs.shape[0]):
        pressure = pressures[i]
        step = steps[i]
        travel += step
        if pressure <= 0.0 or step <= 0.0:
            continue
        x, y = xs[i], ys[i]
        rate = BALL_INK * flow * (BALL_LIGHT + (1.0 - BALL_LIGHT) * pressure)
        rate *= surge_at(travel, salt) * per_pass * step
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
                level = 1.0 - reach * pressure**BALL_POWER + rise * (d / bound) ** 2
                level += BALL_JITTER * JITTER * (unit_noise(ix, iy, salt) - 0.5)
                ground = tooth[iy, ix] + (1.0 - tooth[iy, ix]) * press[iy, ix]
                above = ground + MICRO - level
                touch = min(max(above / (2.0 * MICRO), 0.0), 1.0) * edge
                if touch <= 0.0:
                    continue
                dv = rate * touch * ball_grip(cover, held, under, loose, iy, ix)
                if dv <= 0.0:
                    continue  # (it skids: no ink, and no more paper inked)
                inked[iy, ix] = max(inked[iy, ix], touch)
                for band in range(N):
                    ink[iy, ix, band] += dv * k[band]
                ink[iy, ix, IS] += dv * s
                ink[iy, ix, IV] += dv
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
