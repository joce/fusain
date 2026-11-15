"""The compiled kernels that lay a medium on the paper, and their constants.

Kernels are ``@njit(cache=True, error_model="numpy")``, without ``fastmath``
and without ``parallel``, so that a drawing repeats exactly.

What a kernel calls for every pixel is compiled into its caller
(``inline="always"``): numba otherwise counts a reference to each array such
a helper is given, and back, at every call, which took a third of the time
of a stroke and half of that of a rub. It changes no result;
``tools/benchmark.py`` measures what it is worth.

Speed must not change the picture: ``tools/fingerprint.py`` tells. What is
exact here and must stay so: a table holds values worked out by the very
expression it replaces (:func:`streaks_of`); a loop skips only pixels that
it would have left as they were (:func:`row_span`, :func:`settle_dust`);
and no sum is taken in another order.
"""

from __future__ import annotations

import math

import numpy as np
from murex.spectral import N_BANDS
from numba import njit

#: What a kernel calls for every pixel (see above).
_inlined = njit(cache=True, error_model="numpy", inline="always")

#: Pigment is packed per pixel as [K (N bands), S, volume, sheen], all
#: extensive (summed over the pigment present; the sheen is the volume times
#: the sheen of its medium); the held layer has a fifth part, the volume of
#: wax at the pixel. There are three layers: what the paper holds, and
#: over it what lies on that, in two layers (``under`` and, on top,
#: ``loose``) so that what was laid last is seen over what was laid before.
N = N_BANDS
IS, IV, IX, IW = N, N + 1, N + 2, N + 3
LOOSE, HELD = N + 3, N + 4
#: Most of what lies in the two layers is loose dust, but not all. For each
#: of the two (0 underneath, 1 on top) the sheet keeps the volume that is
#: bonded, of three kinds: dust that is set, by fixative or by a film laid
#: over it; a film, the share of a waxy medium that would have been held had
#: it met the paper and not what lay on it; and paste, what the paper does
#: not hold of a greasy medium. A tool lifts what is set and a film only as
#: it lifts what the paper holds, and neither stirs them nor presses them in;
#: paste it moves as far as it smears.
SET, FILM, PASTE = 0, 1, 2
KINDS = 3
#: What a tool carries is packed as dust is, with a fifth part: the volume of
#: it that is paste.
IP = N + 3

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
#: the mark is as wide as the length of lead that lies on the paper. For the
#: exposed lead of a sharpened pencil that is this many times its point; a
#: medium says its own (:attr:`fusain.Medium.side`), and a stick in the hand
#: can give the length in pixels.
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
#: The bands of a pass are numbered across the tip, the one under its middle
#: being STREAK_MIDDLE; a pass looks up the first STREAK_BANDS of them in a
#: table (enough for a tip some 160 pixels wide).
STREAK_MIDDLE = 64.0
STREAK_BANDS = 128
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
#: A waxy medium is laid as a film, which lies on what it meets: on the
#: paper, where the paper holds it, or on what lies over the paper, as far
#: as that covers it (all of it once it is COVERS thick). There it stays on
#: top, a film, and bonds this share of the loose dust beneath it (where the
#: film and the paste laid with it are themselves COVERS thick; in proportion
#: if thinner): the dust is set under a film, and turns to paste under paste.
BINDS = 0.7
#: Wax films over the tooth: a film this thick is as slick as it gets to a
#: waxy medium, and a pass of one then lays only this much less than on bare
#: paper (resist). A dry medium needs the tooth to scrape it off, and has
#: none once the wax covers it: to a dry medium a film of wax is as slick as
#: it gets when it is COVERS thick, so one firm line of crayon is enough.
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
#: What the paper does not hold of a greasy medium is paste, not dust: all of
#: it for a medium this greasy (as a share of the softest), in proportion
#: for one less greasy. Paste smears under a finger and does not come off on
#: a kneaded eraser.
PASTY = 0.5
#: A greasy tip ploughs into what it is laid over: of the dust and the paste
#: on top that a pass of paste would bury, this share stays on top and mixes
#: with it, so a light oil pastel only half covers a dark one.
PLOUGH = 0.12
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
#: A colourless blender melts into the film (:attr:`fusain.Medium.melts`):
#: it polishes a thin film as if it were slick, takes up MELT_LIFT times its
#: polish of what the paper holds (all of it at most) and lays MELT_LAY
#: times what it drags back down over each of its own widths. One firm layer
#: of it so stirs about half of what lies under it into an even film, which
#: closes the white specks; it lays it back within the width of its tip or
#: so, further along its way than across it, so an edge it is drawn across
#: is dragged out a little.
MELT_LIFT = 1.5
MELT_LAY = 6.0
#: A firm dry tip flattens the tooth itself: pressed harder than
#: BURNISH_FROM, the hardest lead raises ``press`` by FLATTEN in one pass at
#: full pressure. A softer tip gives before the paper does (it flattens as
#: the cube of its hardness), and one that crumbles to dust bears on the
#: paper only with what holds together (as the square of its binding). So a
#: hard pencil pressed firmly levels the tooth, on which its graphite then
#: lies flat and shines evenly; a soft one hardly does, and charcoal and
#: pastel next to not at all. (A waxy tip polishes its film, as above.)
FLATTEN = 5.0
#: The tooth is what scrapes a dry medium off its stick, and a tooth pressed
#: flat bites less: a dry medium lays this much less on it in a pass. It
#: lays it evenly, though: all of a flattened tooth is touched.
FLAT_BITE = 0.7
#: A hard round point pressed into the paper leaves a groove. ``dent`` is
#: how far below the peaks of its tooth the paper has been pushed down, in
#: depths of the tooth of a paper of depth 1: nothing stands higher there,
#: so a tip wider than the groove, which rides on the paper either side,
#: skips it unless it reaches that far down. A stylus at full pressure
#: indents the paper INDENT deep under its middle, on a paper of middling
#: give (times what the paper gives, as a tip's reach), less in proportion
#: at a lighter pressure and towards its rim, since it is round. The ball
#: of a ballpoint, pressed as hard, BALL_DENT.
INDENT = 2.0
BALL_DENT = 0.8
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
#: The ball gathers ink: this share of what it lays stays on its rim as it
#: rolls, up to a drop BALL_BLOB firm passes thick over the square of its
#: radius. A turn brings it round onto the paper, all of it when the line
#: turns by BALL_TURN radians, so a sharp turn leaves a blob and a gentle
#: curve nothing to see. A pen put to the paper has some on it already,
#: which it leaves where it lands: at most the share BALL_START of that,
#: and mostly next to none (a number drawn between 0 and 1, to the power
#: BALL_SELDOM, of it), so that one line in four or so starts with a dot
#: that shows.
BALL_GATHER = 0.05
BALL_BLOB = 4.0
BALL_TURN = 2.0
BALL_START = 0.8
BALL_SELDOM = 6.0
#: A drop of ink is left as a round dot BLOB_WIDE times as wide as the ball,
#: thickest at its middle: a little wider than the line, and darker.
BLOB_WIDE = 1.2
#: The ball skids on a film of pigment: it lays less from the first of these
#: thicknesses (per covered area) and nothing from the second. On wax it
#: writes as badly as a dry medium does (RESIST_DRY, on a film COVERS thick).
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
#: Wax holds what it binds, and an eraser slides on it. Of what the paper
#: holds, of a film and of what is set, an eraser lifts WAX_HOLDS less where
#: the wax is WAX_FAST thick or more (a third of what makes it as slick as
#: it gets to a pencil) and has kept its tooth, and POLISH_HOLDS less where
#: it has been polished flat: a light layer of coloured pencil comes off
#: fairly well, a dense one badly, a burnished one hardly. A blade or grit
#: does not slide: as far as a tool scrapes, it takes what it would take of
#: a medium without wax.
WAX_FAST = 0.1
WAX_HOLDS = 0.6
POLISH_HOLDS = 0.9
#: Graphite slides. A tool that smears moves, in one pass at full pressure,
#: this share (times how strongly it smears) of the graphite the paper
#: holds; the clay of a lead holds the rest back, and the less graphite a
#: lead has, the less of it slides (its share raised to this power). So a
#: stump evens a soft pencil's tone and drags it out, and hardly moves a hard
#: one's. Set dust lying over it keeps it where it is.
SLIDE = 1.0
SLIDE_POWER = 5.0
#: Of the graphite that slides, the share that comes loose as dust, which the
#: tool can carry off; the rest is pushed from grain to grain and stays held.
SLIDE_LOOSE = 0.15


@_inlined
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
def streaks_of(salt):
    """How hard each band of a tip bears in this pass, for the bands that
    :func:`streak_at` looks up: a number in 0..1 for each, and for the one
    after the last."""
    streaks = np.empty(STREAK_BANDS + 1)
    for band in range(STREAK_BANDS + 1):
        streaks[band] = unit_noise(band, 0, salt + 1)
    return streaks


@_inlined
def streak_at(dx, dy, tx, ty, streaks, salt):
    """How much more or less than its share (1) the tip bears at the pixel
    ``dx``, ``dy`` from its centre: by how far across the tip that is, the
    same all along the stroke. ``streaks`` is :func:`streaks_of` this pass."""
    across = (dy * tx - dx * ty) / STREAK_WIDTH + STREAK_MIDDLE
    band = int(math.floor(across))
    blend = across - band
    if 0 <= band < STREAK_BANDS:
        low, high = streaks[band], streaks[band + 1]
    else:  # (a tip too wide for the table)
        low = unit_noise(band, 0, salt + 1)
        high = unit_noise(band + 1, 0, salt + 1)
    streak = low * (1.0 - blend)
    streak += high * blend
    return 1.0 + STREAK * (2.0 * streak - 1.0)


@_inlined
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


@_inlined
def tip_extent(tx, ty, wide, long):
    """Where the tip of :func:`tip_bearing` lies, row by row: on the row
    ``dy`` below its centre it covers the pixels within ``half`` times the
    square root of ``top - dy * dy`` of the point ``lean * dy`` to the right
    of its centre, and none if that is negative (``top`` is the square of
    how far above and below its centre the tip reaches). Returns ``lean``,
    ``half`` and ``top``. (Exact but for rounding: allow a pixel more.)"""
    top = long * long * ty * ty + wide * wide * tx * tx
    lean = (long * long - wide * wide) * tx * ty / top
    return lean, long * wide / top, top


@_inlined
def row_span(x, dy, lean, half, top, x0, x1):
    """The pixels that a tip centred at ``x`` can cover on the row ``dy``
    below its centre, among those from ``x0`` to ``x1``: from the first
    returned to the one before the second, with a pixel to spare either
    way. ``lean``, ``half`` and ``top`` are :func:`tip_extent`; for a round
    tip of radius r, 0, 1 and r * r. The rest of the square around a tip is
    a fifth of it for a round one, and most of it for a stick on its side."""
    middle = x + lean * dy
    reach = half * math.sqrt(max(top - dy * dy, 0.0)) + 1.0
    return max(int(math.floor(middle - reach)), x0), min(
        int(math.ceil(middle + reach)), x1
    )


@_inlined
def free_share(bonded, layer, iy, ix, which):
    """The share of a layer at a pixel (``which``: 0 underneath, 1 on top)
    that can still be moved: loose dust and paste, which is neither set nor
    a film (``bonded`` holds the volume of each kind in each layer)."""
    there = layer[iy, ix, IV]
    hard = bonded[iy, ix, which, SET] + bonded[iy, ix, which, FILM]
    if hard <= 0.0 or there <= 0.0:
        return 1.0
    return 1.0 - min(hard / there, 1.0)


@_inlined
def hard_at(bonded, iy, ix):
    """The volume at a pixel, in the two layers together, that is set or a
    film: what lies between a tool and whatever is under it."""
    hard = bonded[iy, ix, 0, SET] + bonded[iy, ix, 0, FILM]
    return hard + bonded[iy, ix, 1, SET] + bonded[iy, ix, 1, FILM]


@_inlined
def press_dust(held, under, loose, bonded, cover, iy, ix, share):
    """Press the ``share`` of the dust and of the paste at a pixel into the
    paper: not what is set or a film (``bonded``), and less of the rest the
    more of those lies between it and the paper."""
    if loose[iy, ix, IV] <= 0.0 and under[iy, ix, IV] <= 0.0:
        return  # (no dust here: most pixels of a first pass)
    from_loose, from_under = share, share
    barrier = hard_at(bonded, iy, ix)
    if barrier > 0.0:
        share *= 1.0 - min(barrier / max(cover[iy, ix], 1e-6) / COVERS, 1.0)
        from_loose = share * free_share(bonded, loose, iy, ix, 1)
        from_under = share * free_share(bonded, under, iy, ix, 0)
    for part in range(LOOSE):
        moved = loose[iy, ix, part] * from_loose
        loose[iy, ix, part] -= moved
        held[iy, ix, part] += moved
        moved = under[iy, ix, part] * from_under
        under[iy, ix, part] -= moved
        held[iy, ix, part] += moved
    if bonded[iy, ix, 1, PASTE] > 0.0:
        bonded[iy, ix, 1, PASTE] -= bonded[iy, ix, 1, PASTE] * share
    if bonded[iy, ix, 0, PASTE] > 0.0:
        bonded[iy, ix, 0, PASTE] -= bonded[iy, ix, 0, PASTE] * share


@_inlined
def burnish(held, press, drag, iy, ix, polish, lift, lay):
    """Polish a pixel by ``polish`` (the share of one full pass on a film as
    slick as it gets): press its tooth flat, take ``lift`` times that share
    of what it holds onto the tip (``drag``, packed as the held layer is),
    and lay the share ``lay`` of what the tip drags back down on it."""
    press[iy, ix] = min(press[iy, ix] + BURNISH * polish, 1.0)
    up = min(lift * polish, 1.0)
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


@_inlined
def bond_dust(bonded, layer, iy, ix, which, share, film):
    """Bond the ``share`` of the loose dust in one layer at a pixel: the
    share ``film`` of that is set, as under a film, and the rest turns to
    paste."""
    there = bonded[iy, ix, which, SET] + bonded[iy, ix, which, FILM]
    there += bonded[iy, ix, which, PASTE]
    caught = max(layer[iy, ix, IV] - there, 0.0) * share
    bonded[iy, ix, which, SET] += caught * film
    bonded[iy, ix, which, PASTE] += caught * (1.0 - film)


@_inlined
def settle_pixel(fresh, cover, bonded, under, loose, pasty, iy, ix):
    """Make room at a pixel for what a pass has left to settle there
    (``fresh``: what the paper did not hold of it, and its film): what lay on
    top goes under, with what is bonded of it, and the film and the paste
    that come bond part of the loose dust. Returns the volumes of dust and
    paste, and of film, to lay on top; ``fresh`` is left empty."""
    rest, film = fresh[iy, ix, 0], fresh[iy, ix, 1]
    fresh[iy, ix, 0] = 0.0
    fresh[iy, ix, 1] = 0.0
    thin = 1.0 / (cover[iy, ix] * COVERS)
    buried = BURY * min((rest + film) * thin, 1.0)
    # (a greasy tip ploughs up part of what can be moved, which stays on top)
    free = free_share(bonded, loose, iy, ix, 1)
    going = buried * (1.0 - PLOUGH * pasty * free)
    for part in range(LOOSE):
        moved = loose[iy, ix, part] * going
        loose[iy, ix, part] -= moved
        under[iy, ix, part] += moved
    for kind in range(KINDS):
        moved = bonded[iy, ix, 1, kind] * buried
        if kind == PASTE:
            moved *= 1.0 - PLOUGH * pasty
        bonded[iy, ix, 1, kind] -= moved
        bonded[iy, ix, 0, kind] += moved
    paste = rest * pasty
    binder = film + paste
    if binder > 0.0:
        share = BINDS * min(binder * thin, 1.0)
        bond_dust(bonded, under, iy, ix, 0, share, film / binder)
        bond_dust(bonded, loose, iy, ix, 1, share, film / binder)
    bonded[iy, ix, 1, FILM] += film
    bonded[iy, ix, 1, PASTE] += paste
    return rest, film


@_inlined
def settle_row(fresh, cover, bonded, under, loose, k, s, sheen, pasty, iy, x0, x1):
    """Settle what a pass has left on the pixels ``x0`` to ``x1`` of a row
    (:func:`settle_dust`)."""
    for ix in range(x0, x1):
        if fresh[iy, ix, 0] <= 0.0 and fresh[iy, ix, 1] <= 0.0:
            continue  # (nothing, or settled already)
        rest, film = settle_pixel(fresh, cover, bonded, under, loose, pasty, iy, ix)
        new = rest + film
        for band in range(k.shape[0]):
            loose[iy, ix, band] += new * k[band]
        loose[iy, ix, IS] += new * s
        loose[iy, ix, IV] += new
        loose[iy, ix, IX] += new * sheen


@njit(cache=True, error_model="numpy")
def settle_dust(fresh, cover, bonded, under, loose, k, s, sheen, pasty, xs, ys, bound):
    """At the end of a pass, put what it left to lie on the sheet (``fresh``,
    two volumes per pixel: what the paper does not hold of the medium, of
    which the share ``pasty`` is paste and the rest dust, and its film) on
    top of what lay there, which goes under it with what is bonded of it
    (``bonded``), and leave ``fresh`` empty again. The film and the paste
    bond part of the loose dust they come to lie on. Only where the tip
    went: within ``bound`` of the points (``xs``, ``ys``)."""
    height, width = cover.shape
    # (the square around the point before: settled already)
    px0, px1, py0, py1 = 0, 0, 0, 0
    for i in range(xs.shape[0]):
        x0 = max(int(math.floor(xs[i] - bound)), 0)
        x1 = min(int(math.ceil(xs[i] + bound)), width)
        y0 = max(int(math.floor(ys[i] - bound)), 0)
        y1 = min(int(math.ceil(ys[i] + bound)), height)
        for iy in range(y0, y1):
            start = x0
            if py0 <= iy < py1:  # (only what lies to either side of it)
                settle_row(
                    fresh,
                    cover,
                    bonded,
                    under,
                    loose,
                    k,
                    s,
                    sheen,
                    pasty,
                    iy,
                    x0,
                    min(x1, px0),
                )
                start = max(x0, px1)
            settle_row(
                fresh, cover, bonded, under, loose, k, s, sheen, pasty, iy, start, x1
            )
        px0, px1, py0, py1 = x0, x1, y0, y1


@_inlined
def resist_dry(wax):
    """How much less of a medium with the share ``wax`` of wax a film of wax
    takes than bare paper does because the medium is dry: the tooth has to
    scrape it off."""
    return RESIST_DRY * (1.0 - min(wax / STICKS, 1.0))


@_inlined
def resist_waxy(softness, wax):
    """How much less of it the film takes because the medium is waxy: wax
    slides on wax, unless it is a greasy paste, which sticks."""
    greasy = min(max((softness - FIRM) / (1.0 - FIRM), 0.0), 1.0)
    return RESIST * (1.0 - GREASE * greasy) * min(wax / STICKS, 1.0)


@_inlined
def resist_of(softness, wax):
    """How much less of a medium a film of wax as slick as it gets takes than
    bare paper does."""
    return resist_dry(wax) + resist_waxy(softness, wax)


@_inlined
def paste_of(softness, wax):
    """The share that is paste, not dust, of what the paper does not hold of
    a medium: none of a dry medium or of a firm waxy one, all of a greasy
    one."""
    greasy = min(max((softness - FIRM) / (1.0 - FIRM), 0.0), 1.0)
    return min(wax / STICKS, 1.0) * min(greasy / PASTY, 1.0)


@_inlined
def lying_share(under, loose, bonded, covered, iy, ix):
    """The share of the marked part of a pixel on which something lies over
    what the paper holds: dust, set or not, or paste. (A film lies on those,
    and covers no more of the paper than they do.)"""
    lying = under[iy, ix, IV] + loose[iy, ix, IV]
    lying -= bonded[iy, ix, 0, FILM] + bonded[iy, ix, 1, FILM]
    return min(max(lying, 0.0) / covered / COVERS, 1.0)


@_inlined
def deposit(held, fresh, iy, ix, dv, binding, over, k, s, sheen, wax):
    """Lay the volume ``dv`` of a medium at a pixel: the paper holds the
    share ``binding`` of it, but for the share ``over`` of that, a film on
    what lies there; that film and what is not held are left in ``fresh``,
    to settle on top at the end of the pass. Returns the volume left
    there."""
    film = dv * binding * over
    kept = dv * binding - film
    # (over the bands of k, which are the N of the sheet: with a count it
    # cannot see, the compiler keeps this a loop, four or eight bands at a
    # time, where it would unroll N steps one band at a time)
    for band in range(k.shape[0]):
        held[iy, ix, band] += kept * k[band]
    held[iy, ix, IS] += kept * s
    held[iy, ix, IV] += kept
    held[iy, ix, IX] += kept * sheen
    held[iy, ix, IW] += dv * wax
    fresh[iy, ix, 0] += dv - dv * binding
    fresh[iy, ix, 1] += film
    return dv - kept


@_inlined
def within_ends(px, py, ends):
    """How much of the pixel centred at (``px``, ``py``) a stroke cut square
    at its ends leaves: 1 but behind the line across its start or beyond the
    line across its end, where it is cut. ``ends`` is [x, y, and heading
    (x, y) where it starts, 1 if it is cut square there, the same where it
    ends]."""
    share = 1.0
    if ends[4] > 0.0:
        ahead = (px - ends[0]) * ends[2] + (py - ends[1]) * ends[3]
        share = min(max(ahead + 0.5, 0.0), 1.0)
    if ends[9] > 0.0:
        short = (ends[5] - px) * ends[7] + (ends[6] - py) * ends[8]
        share = min(share, min(max(short + 0.5, 0.0), 1.0))
    return share if share > 1e-6 else 0.0  # (what rounding leaves past it)


@_inlined
def bear_on_row(row, now, iy, xa, xb, ends, streaks, salt):
    """How a tip at ``now`` (see :func:`bear`) bears on the pixels ``xa`` to
    ``xb`` of the row ``iy``, which depends on nothing the sheet holds: for
    each, in the three rows of ``row`` (counted from ``xa``), how far inside
    the edge of the tip it is (0 if the tip does not mark it), how hard the
    tip bears on it, and where it does mark it, the level in the tooth it
    reaches there in this pass.

    It is worked out for the whole row before anything is laid, in loops
    that do one thing each: the processor then works on several pixels at
    once, where one loop that also laid the medium went a pixel at a time
    (each has its divisions, its square roots and a power to wait for)."""
    x, y, tx, ty, wide, long, rounded = (
        now[0],
        now[1],
        now[2],
        now[3],
        now[4],
        now[5],
        now[6],
    )
    reach, pressure, at_x, at_y = now[7], now[8], now[11], now[12]
    dy = iy + 0.5 - y
    for ix in range(xa, xb):
        dx = ix + 0.5 - x
        share, edge = tip_bearing(dx, dy, tx, ty, wide, long, rounded)
        if edge > 0.0 and at_x == at_x:
            # (landing or lifting: only where it is put)
            ax, ay = ix + 0.5 - at_x, iy + 0.5 - at_y
            if tip_bearing(ax, ay, tx, ty, wide, long, rounded)[1] <= 0.0:
                edge = 0.0
        if edge > 0.0:
            edge *= within_ends(ix + 0.5, iy + 0.5, ends)
        row[0, ix - xa] = edge
        row[1, ix - xa] = pressure * share * streak_at(dx, dy, tx, ty, streaks, salt)
    for ix in range(xa, xb):
        if row[0, ix - xa] > 0.0:
            level = 1.0 - reach * row[1, ix - xa] ** REACH_POWER
            level += JITTER * (unit_noise(ix, iy, salt) - 0.5)
            row[2, ix - xa] = level


@njit(cache=True, error_model="numpy")
def bear(
    tooth,
    press,
    dent,
    cover,
    held,
    under,
    loose,
    fresh,
    bonded,
    drag,
    k,
    law,
    now,
    step,
    ends,
    streaks,
    row,
    salt,
):
    """Lay what a tip leaves as it slides ``step`` pixels over the paper
    under it. ``now`` is [x, y, heading (x, y), half-widths across and along, the
    share of it that is rounded, how far down the tooth it reaches at full
    pressure, its pressure, the share of a pass it lays on a pixel it bears on
    for each pixel it slides, the share of what it drags it lays back there
    for each pixel it slides, and, for a tip landing or
    lifting, the x and y of where it does so (else NaN): it marks then only
    the paper it covers there]. ``law`` is [what one pass lays, the
    thickness that fills the tooth, how much less a film of wax takes of it
    because it is dry and because it is waxy, how far what the paper holds
    of it is a film, binding, scattering, sheen, wax, how many depths of
    this paper's tooth one of a paper of depth 1 is, how far one pass of it
    at full pressure flattens the tooth, how much less of it a flattened
    tooth takes, how far it melts into a film of wax it polishes]. ``ends``
    is as for :func:`within_ends` and ``streaks`` as for :func:`streak_at`;
    ``row`` is room to work in, three rows as wide as the sheet
    (:func:`bear_on_row`).

    Returns the volume laid and what of it is left to settle on top."""
    height, width = tooth.shape
    x, y, tx, ty, wide, long = now[0], now[1], now[2], now[3], now[4], now[5]
    per_pass, dragged = now[9] * step, now[10] * step
    lay, full, dry, slides, filmy = law[0], law[1], law[2], law[3], law[4]
    binding, s, sheen, wax = law[5], law[6], law[7], law[8]
    sunk, crush, smooth, melts = law[9], law[10], law[11], law[12]
    # (a tip that melts into the film stirs more of it, and lays it sooner)
    stir = BURNISH_LIFT + (MELT_LIFT - BURNISH_LIFT) * melts
    dragged *= 1.0 + (MELT_LAY / BURNISH_LAY - 1.0) * melts
    bound = max(wide, long)
    x0 = max(int(math.floor(x - bound)), 0)
    x1 = min(int(math.ceil(x + bound)), width)
    y0 = max(int(math.floor(y - bound)), 0)
    y1 = min(int(math.ceil(y + bound)), height)
    lean, half, top = tip_extent(tx, ty, wide, long)
    laid = 0.0
    lying = 0.0
    for iy in range(y0, y1):
        xa, xb = row_span(x, iy + 0.5 - y, lean, half, top, x0, x1)
        # first how the tip bears on each pixel of the row, which is its
        # shape and nothing else; then what that does to the paper
        bear_on_row(row, now, iy, xa, xb, ends, streaks, salt)
        for ix in range(xa, xb):
            edge = row[0, ix - xa]
            if edge <= 0.0:
                continue
            local = row[1, ix - xa]
            level = row[2, ix - xa]
            # (a tooth pressed flat is level with its peaks, and nothing
            # in a groove stands higher than its floor)
            ground = tooth[iy, ix] + (1.0 - tooth[iy, ix]) * press[iy, ix]
            ground = min(ground, 1.0 - dent[iy, ix] * sunk)
            above = ground + MICRO - level
            touch = min(max(above / (2.0 * MICRO), 0.0), 1.0) * edge
            if touch <= 0.0:
                continue
            covered = max(cover[iy, ix], touch)
            cover[iy, ix] = covered
            there = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
            thick = (there + fresh[iy, ix, 0] + fresh[iy, ix, 1]) / covered
            was_set = bonded[iy, ix, 0, SET] + bonded[iy, ix, 1, SET]
            thick -= (1.0 - FIXED_ROOM) * was_set / covered
            bearing = min(local, 1.0)
            passing = per_pass * touch
            waxed = held[iy, ix, IW] / covered
            waxy = min(waxed / WAX_FULL, 1.0)
            if bearing > BURNISH_FROM:
                firm = (bearing - BURNISH_FROM) / (1.0 - BURNISH_FROM)
                # (a hard tip crushes the tooth, whatever lies on it)
                press[iy, ix] = min(press[iy, ix] + crush * firm * passing, 1.0)
                # (a blender melts into the film, however thin it is)
                slick = waxy + (1.0 - waxy) * melts
                polish, lay_back = firm * slick * passing, dragged * touch
                burnish(held, press, drag, iy, ix, polish, stir, lay_back)
            # the tip presses some of the dust under it into the paper
            pressed = min(PRESS_IN * bearing, 1.0) * passing
            press_dust(held, under, loose, bonded, cover, iy, ix, pressed)
            room = full * (FILL_LIGHT + (1.0 - FILL_LIGHT) * bearing) - thick
            if room <= 0.0:
                continue
            # (a light pass takes less, but not in proportion: the lead
            # still slides over all it touches)
            dv = lay * math.sqrt(bearing * min(above / SPAN, 1.0)) * (room / full)
            # (to a dry medium wax is slick as soon as it covers the tooth)
            dv *= 1.0 - dry * min(waxed / COVERS, 1.0) - slides * waxy
            # (and a flattened tooth bites it less)
            dv *= 1.0 - smooth * press[iy, ix]
            dv = min(dv * per_pass, room) * touch
            over = 0.0  # the share of what is held that lies on dust
            if filmy > 0.0:
                over = filmy * lying_share(under, loose, bonded, covered, iy, ix)
            lying += deposit(held, fresh, iy, ix, dv, binding, over, k, s, sheen, wax)
            laid += dv
    return laid, lying


@njit(cache=True, error_model="numpy")
def tip_at(now, shape, flat, reach, i, xs, ys, pressures):
    """Fill ``now`` (see :func:`bear`) for a tip ``flat`` wide at sample
    ``i``, of ``shape`` [its width, tilt, length of its side]. Returns how
    far from its centre it reaches."""
    tilt, side = shape[1], shape[2]
    wide = 0.5 * (flat + max(side - flat, 0.0) * tilt) + 0.5
    long = 0.5 * flat + 0.5
    rounded = RIM + (1.0 - RIM) * TILT_RIM * tilt
    now[0], now[1] = xs[i], ys[i]
    now[2], now[3] = heading(xs, ys, i)
    now[4], now[5], now[6] = wide, long, rounded
    now[7] = reach * (1.0 - TILT_REACH * tilt)
    now[8] = pressures[i]
    # a pixel on the path of the tip's centre gets one pass's worth in all;
    # and of what it drags, the tip lays this share on a pixel it bears on,
    # per pixel moved
    now[9] = 1.0 / (2.0 * long - (rounded * long + 0.5))
    now[10] = BURNISH_LAY / (2.0 * wide * math.pi * wide * long)
    now[11], now[12] = math.nan, math.nan
    return max(wide, long)


@njit(cache=True, error_model="numpy")
def put_down(
    tooth,
    press,
    dent,
    cover,
    held,
    under,
    loose,
    fresh,
    bonded,
    drag,
    k,
    law,
    now,
    ends,
    streaks,
    row,
    salt,
    spacing,
    way,
):
    """Where a tip lands (``way`` -1) or lifts (``way`` 1), at ``now`` (see
    :func:`bear`): lay on the paper it covers there what it would have laid
    had it slid onto it from behind (off it ahead), so that all of that gets
    a full pass, as under the middle of a stroke, and not only the paper
    ahead of the middle of the tip (behind it). It lays it as it lays a
    stroke, ``spacing`` pixels at a time. Returns the volume laid and what of
    it is left to settle."""
    x, y, tx, ty = now[0], now[1], now[2], now[3]
    count = int(math.ceil(2.0 * now[5] / spacing))
    now[11], now[12] = x, y
    laid, lying = 0.0, 0.0
    for j in range(1, count + 1):
        now[0], now[1] = x + way * j * spacing * tx, y + way * j * spacing * ty
        dv, rest = bear(
            tooth,
            press,
            dent,
            cover,
            held,
            under,
            loose,
            fresh,
            bonded,
            drag,
            k,
            law,
            now,
            spacing,
            ends,
            streaks,
            row,
            salt,
        )
        laid += dv
        lying += rest
    now[0], now[1], now[11], now[12] = x, y, math.nan, math.nan
    return laid, lying


@njit(cache=True, error_model="numpy")
def law_of(medium, paper):
    """What :func:`bear` takes as ``law`` for a medium [scattering per unit
    volume, softness, binding, the share of wax in it, its sheen, how far
    it melts into a film of wax] on a paper [depth, bite, give, hold]; and
    how far down the tooth it reaches at full pressure, held upright."""
    s, softness, binding, wax = medium[0], medium[1], medium[2], medium[3]
    depth, bite, give = paper[0], paper[1], paper[2]
    reach = (REACH_HARD + REACH_SOFT * softness) * (GIVE_NONE + GIVE_FULL * give)
    # (a level paper is touched all over: any pressure reaches all of it,
    # short of the bottom of a groove)
    reach = reach / depth if depth > 1e-6 else 1e9 * reach
    shape = FILL_HARD + (1.0 - FILL_HARD) * softness**1.5
    full = (
        FILL
        * shape
        * (FILL_FLAT + (1.0 - FILL_FLAT) * min(depth, 2.0))
        * (0.5 + 0.5 * bite)
        * (1.0 + PILE * (1.0 - binding))
        * paper[3]
    )
    law = np.empty(13)
    law[0], law[1] = LAY * shape * (0.5 + bite), full
    law[2], law[3] = resist_dry(wax), resist_waxy(softness, wax)
    law[4] = min(wax / STICKS, 1.0)  # how far what the paper holds is a film
    law[5], law[6], law[7], law[8] = binding, s, medium[4], wax
    law[9] = 1.0 / depth if depth > 1e-6 else 1e9
    # (a waxy tip polishes its own film instead: see BURNISH)
    dryness = 1.0 - min(wax / STICKS, 1.0)
    law[10] = FLATTEN * (1.0 - softness) ** 3 * binding * binding * dryness
    law[11] = FLAT_BITE * dryness
    law[12] = medium[5]  # how far it melts into the film it polishes
    return law, reach


@njit(cache=True, error_model="numpy")
def ends_of(xs, ys, square_start, square_end):
    """What :func:`within_ends` takes for a stroke through the points
    (``xs``, ``ys``), cut square at its start and at its end or not."""
    last = xs.shape[0] - 1
    ends = np.empty(10)
    ends[0], ends[1] = xs[0], ys[0]
    ends[2], ends[3] = heading(xs, ys, 0)
    ends[4] = square_start
    ends[5], ends[6] = xs[last], ys[last]
    ends[7], ends[8] = heading(xs, ys, last)
    ends[9] = square_end
    return ends


@njit(cache=True, error_model="numpy")
def ends_near(ends, near, after, before, reach):
    """Fill ``near`` with the square cuts of ``ends`` (see
    :func:`ends_of`) that apply to a tip ``after`` pixels along the stroke
    from its start and ``before`` pixels from its end: a cut only where the
    tip is within ``reach`` of it, so that a stroke that comes back across
    the line through one of its ends is not cut there."""
    for part in range(10):
        near[part] = ends[part]
    if after > reach:
        near[4] = 0.0
    if before > reach:
        near[9] = 0.0


@njit(cache=True, error_model="numpy")
def lay_stroke(
    tooth,
    press,
    dent,
    cover,
    held,
    under,
    loose,
    fresh,
    bonded,
    xs,
    ys,
    pressures,
    steps,
    shapes,
    tip,
    k,
    medium,
    paper,
    salt,
):
    """Draw one pass of a tip through the points (``xs``, ``ys``), bearing
    with ``pressures`` (0..1) and moving ``steps`` pixels at each.

    ``shapes`` holds for each point [the width of the tip's flat there,
    unworn, its tilt (0..1), the length of its side in pixels (what it marks
    across the stroke held flat)]. ``tip`` is [the full width of the lead
    (the flat never gets wider), how fast it wears (:data:`WEAR`, or 0 for a
    tip that keeps its width), the volume it has worn since it was
    sharpened, the spacing of the points, whether the stroke is cut square
    at its start (1) or ends there in the round of the tip (0), the same at
    its end, the spacing a tip this wide is sampled at (where it lands and
    lifts it goes at that pace, however short the stroke)]. A square cut
    applies only within the tip's reach of its end. Its mark is an ellipse,
    as long along the stroke as its flat is wide, and across it as wide as
    the flat, and the share ``tilt`` of what its side adds to that. Where it
    lands and where it lifts, all the paper under it gets a full pass
    (:func:`put_down`).

    ``medium`` is [scattering per unit volume, softness, binding, the share
    of wax in it, its sheen, how far it melts into a film of wax] (its
    absorption per band is ``k``) and ``paper`` is
    [depth, bite, give, hold]. The medium is laid where the tip touches the
    tooth.
    What the paper does not hold of it is dust, or paste if the medium is
    greasy (:func:`paste_of`), gathered in ``fresh`` (empty before and after)
    and laid at the end of the pass on top of what lay there (``loose``),
    which goes ``under`` it. With it goes the film of a waxy medium: the part
    of what would have been held that met not the paper but what lies over it
    (:func:`lying_share`). What is set or a film in the two layers
    (``bonded``, the volume of each kind in each) is not pressed in, and dust
    that is set takes less of the room in the tooth. Wax films over the
    tooth, which then takes less (:func:`resist_of`); pressed hard, the tip
    polishes it flat (``press``) and smears it along, and a hard tip pressed
    hard flattens the tooth itself. Where the paper is indented (``dent``,
    how far below the peaks of the tooth), the tip lays only what it reaches
    down to.

    Returns the volume the tip has lost: what it laid, less what it has
    dragged off the sheet.
    """
    lead, wear, worn, spacing = tip[0], tip[1], tip[2], tip[3]
    # (where it lands and lifts, the tip goes at its own pace, however short
    # the stroke: a stroke of no length is a dot)
    landing = spacing if spacing >= 0.5 * tip[6] else tip[6]
    law, reach = law_of(medium, paper)
    ends = ends_of(xs, ys, tip[4], tip[5])
    near = np.empty(10)  # (the cuts that apply where the tip is now)
    streaks = streaks_of(salt)
    row = np.empty((3, tooth.shape[1]))
    count = xs.shape[0]
    reached = 0.0  # the furthest the tip has been from its path
    # what a tip that polishes drags along
    drag = np.zeros(HELD)
    total = 0.0
    lying = 0.0  # what the pass leaves lying on the sheet, to settle
    # where the tip last bore, and how
    now = np.zeros(13)
    tip_at(now, shapes[0], shapes[0, 0], reach, 0, xs, ys, pressures)
    for i in range(count):
        step = steps[i]
        if pressures[i] <= 0.0 or step <= 0.0:
            continue
        flat = shapes[i, 0]
        if wear > 0.0:  # the flat grows with what the tip has laid
            flat = min((flat**3 + wear * max(worn + total, 0.0)) ** (1.0 / 3.0), lead)
        bound = tip_at(now, shapes[i], flat, reach, i, xs, ys, pressures)
        reached = max(reached, bound)
        ends_near(ends, near, i * spacing, (count - 1 - i) * spacing, bound + spacing)
        lands = i == 0 or pressures[i - 1] <= 0.0
        lifts = i == count - 1 or pressures[i + 1] <= 0.0
        # (the first and the last point take half a spacing, which is the
        # half of the tip ahead of them, behind them; what lands or lifts
        # there takes the other half too)
        if count > 1 and ((lands and i == 0) or (lifts and i == count - 1)):
            step += 0.5 * landing
        for way in (-1.0, 0.0, 1.0):
            if way == 0.0:
                dv, rest = bear(
                    tooth,
                    press,
                    dent,
                    cover,
                    held,
                    under,
                    loose,
                    fresh,
                    bonded,
                    drag,
                    k,
                    law,
                    now,
                    step,
                    near,
                    streaks,
                    row,
                    salt,
                )
            elif (way < 0.0 and lands) or (way > 0.0 and lifts):
                dv, rest = put_down(
                    tooth,
                    press,
                    dent,
                    cover,
                    held,
                    under,
                    loose,
                    fresh,
                    bonded,
                    drag,
                    k,
                    law,
                    now,
                    near,
                    streaks,
                    row,
                    salt,
                    landing,
                    way,
                )
            else:
                continue
            total += dv
            lying += rest
    if lying > 0.0:  # (else the paper holds it all: nothing to settle)
        pasty = paste_of(medium[1], medium[3])
        settle_dust(
            fresh,
            cover,
            bonded,
            under,
            loose,
            k,
            medium[0],
            medium[4],
            pasty,
            xs,
            ys,
            reached,
        )
    # what the tip still drags it leaves where it lifts off; only off the
    # sheet does it leave the paper with it
    return total - set_down(
        held, cover, drag, now[0], now[1], now[2], now[3], now[4], now[5], now[6]
    )


@_inlined
def spread_dust(loose, near, iy, ix, take, back):
    """Take the share ``take`` of the dust on top at a pixel onto the face of
    the tool (``near``, packed as dust is), and lay the share ``back`` of
    what is on its face down on the pixel. Returns the volumes laid and
    taken."""
    taken = loose[iy, ix, IV] * take
    given = (near[IV] + taken) * back
    for part in range(LOOSE):
        up = loose[iy, ix, part] * take
        down = (near[part] + up) * back
        near[part] += up - down
        loose[iy, ix, part] += down - up
    return given, taken


@_inlined
def slide_graphite(held, loose, slid, bonded, cover, iy, ix, rubbed, back):
    """Slide the graphite that the paper holds at a pixel under a tool that
    has ``rubbed`` it (how strongly it smears, times its share of a pass):
    part comes loose as dust on top, the rest goes onto the face of the tool
    (``slid``, packed as dust is), which lays the share ``back`` of what is
    on its face down again, held. Other pigments do not slide. Returns the
    volume laid down."""
    there = held[iy, ix, IV]
    slip = 0.0
    if rubbed > 0.0 and there > 0.0 and held[iy, ix, IX] > 0.0:
        graphite = min(held[iy, ix, IX] / there, 1.0)
        slip = min(SLIDE * rubbed * graphite**SLIDE_POWER, 1.0)
        barrier = hard_at(bonded, iy, ix)
        if barrier > 0.0:
            sealed = barrier / max(cover[iy, ix], 1e-6) / COVERS
            slip *= 1.0 - min(sealed, 1.0)
    if slip <= 0.0 and slid[IV] <= 0.0:
        return 0.0  # (no graphite here or on the tool: every other medium)
    for part in range(LOOSE):
        moved = held[iy, ix, part] * slip * SLIDE_LOOSE
        held[iy, ix, part] -= moved
        loose[iy, ix, part] += moved
    rest = slip * (1.0 - SLIDE_LOOSE) / max(1.0 - slip * SLIDE_LOOSE, 1e-9)
    given, _ = spread_dust(held, slid, iy, ix, min(rest, 1.0), back)
    return given


@_inlined
def lift_bonded(layer, bonded, load, iy, ix, which, up_dust, up_paste, up_held):
    """The share of a layer at a pixel that a tool takes when not all of it
    is loose dust: the shares ``up_dust`` of its loose dust, ``up_paste`` of
    its paste and ``up_held`` of what is set or a film in it. What it takes
    of each kind is taken off ``bonded``, and the paste is counted on the
    tool (``load``)."""
    there = layer[iy, ix, IV]
    if there <= 0.0:
        return 0.0
    paste = bonded[iy, ix, which, PASTE]
    hard = bonded[iy, ix, which, SET] + bonded[iy, ix, which, FILM]
    hard = min(hard / there, 1.0)
    pasty = min(paste / there, 1.0 - hard)
    bonded[iy, ix, which, SET] -= bonded[iy, ix, which, SET] * up_held
    bonded[iy, ix, which, FILM] -= bonded[iy, ix, which, FILM] * up_held
    bonded[iy, ix, which, PASTE] -= paste * up_paste
    load[IP] += paste * up_paste
    return up_dust * (1.0 - hard - pasty) + up_paste * pasty + up_held * hard


@_inlined
def wax_grip(held, cover, flat, iy, ix, scrape):
    """The share of its lift a tool keeps on what a pixel holds, for the
    wax there: all of it where there is none, less the thicker the wax is
    and the further it has been polished (``flat``), which a tool gets
    through as far as it scrapes (``scrape``, 0..1)."""
    covered = cover[iy, ix]
    if covered <= 0.0 or held[iy, ix, IW] <= 0.0:
        return 1.0
    waxy = min(held[iy, ix, IW] / covered / WAX_FAST, 1.0)
    fast = WAX_HOLDS + (POLISH_HOLDS - WAX_HOLDS) * flat[iy, ix]
    return 1.0 - (1.0 - scrape) * waxy * fast


@_inlined
def lift(held, under, loose, bonded, load, iy, ix, lifts, share):
    """Lift onto a tool (``load``) what it takes at a pixel in the ``share``
    of a pass at full pressure. ``lifts`` is [how strongly it lifts the
    dust, how strongly what is held, how strongly it smears, the share of
    its lift it keeps on what is held here (:func:`wax_grip`)]: it takes
    less of the dust underneath than of the dust on top; what is set or a
    film only as what the paper holds; and paste as dust if it smears, as
    what is held if it only lifts."""
    greasy = min(lifts[2], 1.0)
    up_held = min(lifts[1] * lifts[3] * share, 1.0)
    up_loose = min(lifts[0] * share, 1.0)
    up_under = min(UNDER_LIFT * lifts[0] * share, 1.0)
    kinds = bonded[iy, ix, 1, SET] + bonded[iy, ix, 1, FILM]
    if kinds + bonded[iy, ix, 1, PASTE] > 0.0:  # (else it is all loose dust)
        up_paste = up_held + max(up_loose - up_held, 0.0) * greasy
        up_loose = lift_bonded(
            loose, bonded, load, iy, ix, 1, up_loose, up_paste, up_held
        )
    kinds = bonded[iy, ix, 0, SET] + bonded[iy, ix, 0, FILM]
    if kinds + bonded[iy, ix, 0, PASTE] > 0.0:
        up_paste = up_held + max(up_under - up_held, 0.0) * greasy
        up_under = lift_bonded(
            under, bonded, load, iy, ix, 0, up_under, up_paste, up_held
        )
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


@_inlined
def spread_top(loose, bonded, near, iy, ix, rate, back):
    """Spread what lies on top at a pixel from grain to grain: take the share
    ``rate`` of what can be moved of it onto the face of the tool (``near``),
    and lay the share ``back`` of what is on its face down again; the paste
    in it stays paste. Returns the volumes laid and taken."""
    take = rate
    if bonded[iy, ix, 1, SET] + bonded[iy, ix, 1, FILM] > 0.0:
        take = rate * free_share(bonded, loose, iy, ix, 1)
    given, taken = spread_dust(loose, near, iy, ix, take, back)
    if bonded[iy, ix, 1, PASTE] > 0.0 or near[IP] > 0.0:
        up = bonded[iy, ix, 1, PASTE] * rate
        down = (near[IP] + up) * back
        near[IP] += up - down
        bonded[iy, ix, 1, PASTE] += down - up
    return given, taken


@_inlined
def stir_up(under, loose, bonded, iy, ix, stirred):
    """Stir the share ``stirred`` of what can be moved of the layer
    underneath at a pixel up into the layer on top. Returns the volume in
    the two layers that can be moved: the dust and the paste."""
    moving = stirred
    hard = bonded[iy, ix, 0, SET] + bonded[iy, ix, 0, FILM]
    if hard > 0.0:
        moving = stirred * free_share(bonded, under, iy, ix, 0)
    for part in range(LOOSE):
        moved = under[iy, ix, part] * moving
        under[iy, ix, part] -= moved
        loose[iy, ix, part] += moved
    if bonded[iy, ix, 0, PASTE] > 0.0:
        moved = bonded[iy, ix, 0, PASTE] * stirred
        bonded[iy, ix, 0, PASTE] -= moved
        bonded[iy, ix, 1, PASTE] += moved
    hard += bonded[iy, ix, 1, SET] + bonded[iy, ix, 1, FILM]
    return max(under[iy, ix, IV] + loose[iy, ix, IV] - hard, 0.0)


@njit(cache=True, error_model="numpy")
def rub_stroke(
    cover,
    flat,
    held,
    under,
    loose,
    bonded,
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
    scrape,
    carried,
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
    also spreads the dust on top over the pixels around, and slides the
    graphite that the paper holds (:func:`slide_graphite`).

    Not all that lies in the two layers is loose dust (``bonded``, the
    volume of each kind in each layer). What is set, by fixative or under a
    film, and a film itself are lifted only as held pigment is, and are
    neither pressed in nor stirred. Paste is moved as dust is by a tool that
    smears, and lifted only as held pigment is by one that does not; it is
    paste again where the tool lays it. Wax holds all that is lifted as held
    pigment is, the more the thicker it is and the further it has been
    polished (``flat``, the ``press`` of the sheet), but for the share
    ``scrape`` of the tool's lift, which cuts through it
    (:func:`wax_grip`).

    ``carried`` is what the tool carries, three rows packed as dust is with
    the volume of paste after ([K (N bands), S, volume, sheen, paste]): what
    it has lifted, the dust on its face, soon laid back, and the held
    graphite it is pushing along, likewise. The tool lands with it and it is
    updated in place, so a tool that is not cleaned between two passes gives
    back as it takes.
    """
    height, width = cover.shape
    bound = radius + 0.5
    rim = RIM * bound + 0.5
    per_pass = 1.0 / (2.0 * bound - rim)
    # of what it carries, the share the tool lays on a pixel it bears fully
    # on, per pixel moved
    per_pixel = lay / (2.0 * bound * math.pi * bound * bound)
    per_near = SPREAD / (2.0 * bound * math.pi * bound * bound)
    lifts = np.array([lift_loose, lift_held, smear, 1.0])
    streaks = streaks_of(salt)
    load = carried[0]
    near = carried[1]  # the dust on the face of the tool, soon laid back
    slid = carried[2]  # held graphite it is pushing along, likewise
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
            dy = iy + 0.5 - y
            xa, xb = row_span(x, dy, 0.0, 1.0, bound * bound, x0, x1)
            for ix in range(xa, xb):
                dx = ix + 0.5 - x
                d = math.sqrt(dx * dx + dy * dy)
                edge = min(bound - d, 1.0)
                if edge <= 0.0:
                    continue
                local = pressure * min((bound - d) / rim, 1.0) * edge
                local *= streak_at(dx, dy, tx, ty, streaks, salt)
                share = local * per_pass * step  # of one pass at full pressure
                lifts[3] = wax_grip(held, cover, flat, iy, ix, scrape)
                lift(held, under, loose, bonded, load, iy, ix, lifts, share)
                back = min(per_near * local * step, 1.0)
                pushed = slide_graphite(
                    held, loose, slid, bonded, cover, iy, ix, smear * share, back
                )
                # lay, as dust all over the pixel
                before = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
                down = min(per_pixel * local * step, 1.0)
                laid = load[IV] * down
                for part in range(LOOSE):
                    given = load[part] * down
                    load[part] -= given
                    loose[iy, ix, part] += given
                if load[IP] > 0.0:  # (the paste in it is paste again)
                    given = load[IP] * down
                    load[IP] -= given
                    bonded[iy, ix, 1, PASTE] += given
                # spread, from grain to grain
                rate = min(SPREADS * smear * share, 1.0)
                given, taken = spread_top(loose, bonded, near, iy, ix, rate, back)
                laid += given
                after = before + laid - taken
                if after <= 1e-9:
                    cover[iy, ix] = 0.0  # rubbed clean
                    continue
                bare = 1.0 - cover[iy, ix]
                cover[iy, ix] += bare * min((laid + pushed) / after, 1.0)
                # press in; stir up what is underneath; and push what is
                # loose into the low ground
                spent = min(press * share, 1.0)
                press_dust(held, under, loose, bonded, cover, iy, ix, spent)
                stirred = min(smear * share, 1.0)
                dusty = stir_up(under, loose, bonded, iy, ix, stirred) / after
                cover[iy, ix] += (1.0 - cover[iy, ix]) * stirred * dusty


@_inlined
def ball_grip(cover, held, under, loose, iy, ix):
    """The share of its ink a ball lays at a pixel: all of it on bare paper,
    less where it skids on a film of pigment or of wax."""
    covered = cover[iy, ix]
    if covered <= 0.0:
        return 1.0
    thick = (held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]) / covered
    skid = min(max((thick - SKID_FROM) / (SKID_FULL - SKID_FROM), 0.0), 1.0)
    waxy = min(held[iy, ix, IW] / covered / COVERS, 1.0)
    # (only the covered share of the pixel is slick)
    return (1.0 - skid * covered) * (1.0 - RESIST_DRY * waxy * covered)


@_inlined
def surge_at(travel, salt):
    """How much more or less ink than its share (1) the ball gives after
    rolling ``travel`` pixels."""
    along = travel / BALL_SURGE
    band = int(math.floor(along))
    blend = along - band
    surge = unit_noise(band, 1, salt + 2) * (1.0 - blend)
    surge += unit_noise(band + 1, 1, salt + 2) * blend
    return 1.0 + BALL_UNEVEN * (2.0 * surge - 1.0)


@_inlined
def blot(cover, held, under, loose, ink, inked, x, y, volume, radius, full, k, s):
    """Leave a drop of ink of this ``volume`` at (``x``, ``y``): a round dot
    of this ``radius``, thickest at its middle. It lies in the valleys of
    the tooth as on its peaks: the ink of a pixel is then on all of it if
    there is enough to cover it ``full`` thick, and on a share of it if
    not. It takes on a film of pigment or of wax as badly as the line does
    (:func:`ball_grip`). Returns the volume laid: less than the drop where
    it skids or lies off the sheet."""
    height, width = inked.shape
    x0, x1 = int(math.floor(x - radius)), int(math.ceil(x + radius))
    y0, y1 = int(math.floor(y - radius)), int(math.ceil(y + radius))
    weight = 0.0
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            dx, dy = ix + 0.5 - x, iy + 0.5 - y
            weight += max(1.0 - (dx * dx + dy * dy) / (radius * radius), 0.0)
    if weight <= 0.0:
        return 0.0
    laid = 0.0
    for iy in range(max(y0, 0), min(y1, height)):
        for ix in range(max(x0, 0), min(x1, width)):
            dx, dy = ix + 0.5 - x, iy + 0.5 - y
            share = max(1.0 - (dx * dx + dy * dy) / (radius * radius), 0.0)
            dv = volume * share / weight
            dv *= ball_grip(cover, held, under, loose, iy, ix)
            if dv <= 0.0:
                continue
            for band in range(k.shape[0]):
                ink[iy, ix, band] += dv * k[band]
            ink[iy, ix, IS] += dv * s
            ink[iy, ix, IV] += dv
            # (the ink there is on at least as much of the pixel as it
            # covers ``full`` thick)
            inked[iy, ix] = max(inked[iy, ix], min(ink[iy, ix, IV] / full, 1.0))
            laid += dv
    return laid


@_inlined
def turn_of(tx, ty, wx, wy):
    """The angle, in radians (0 to pi), between two unit headings."""
    return math.atan2(abs(tx * wy - ty * wx), tx * wx + ty * wy)


@njit(cache=True, error_model="numpy")
def roll_stroke(
    tooth,
    press,
    dent,
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

    The ball presses a groove into the paper (``dent``), the deeper the
    harder it bears, and goes down into one that is there: it is as small as
    what made it. And it gathers ink on its rim as it rolls, which it leaves
    as a drop (:func:`blot`) where it lands and where the line turns.

    Returns the volume of ink laid, drops included: what the pen has lost,
    but for what is still on its rim.
    """
    height, width = tooth.shape
    s, flow = pen[0], pen[1]
    depth, give = paper[0], paper[2]
    yields = GIVE_NONE + GIVE_FULL * give
    reach = BALL_REACH * yields
    reach = reach / depth if depth > 1e-6 else 1e9
    rise = BALL_ROUND / depth if depth > 1e-6 else 0.0
    bound = radius + 0.5
    per_pass = 1.0 / (2.0 * bound - 1.0)
    # the drop of ink on the rim of the ball: what it holds at most, and
    # what the pen lands with
    most = BALL_BLOB * BALL_INK * flow * bound * bound
    gathered = most * BALL_START * unit_noise(0, 2, salt + 3) ** BALL_SELDOM
    wx, wy = 1.0, 0.0  # the way it was last going
    down = False  # whether it was on the paper at the last point
    travel = 0.0
    total = 0.0
    for i in range(xs.shape[0]):
        pressure = pressures[i]
        step = steps[i]
        travel += step
        if pressure <= 0.0 or step <= 0.0:
            down = False
            continue
        x, y = xs[i], ys[i]
        tx, ty = heading(xs, ys, i)
        # where it lands it leaves all it had gathered; where the line
        # turns, as much of it as the turn brings round
        drop = gathered
        if down:
            drop *= min(turn_of(tx, ty, wx, wy) / BALL_TURN, 1.0)
        wx, wy, down = tx, ty, True
        if drop > 1e-9:
            left = blot(
                cover,
                held,
                under,
                loose,
                ink,
                inked,
                x,
                y,
                drop,
                BLOB_WIDE * bound,
                BALL_INK * flow,
                k,
                s,
            )
            gathered = max(gathered - left, 0.0)
            total += left
        rate = BALL_INK * flow * (BALL_LIGHT + (1.0 - BALL_LIGHT) * pressure)
        rate *= surge_at(travel, salt) * per_pass * step
        sink = BALL_DENT * yields * pressure
        x0 = max(int(math.floor(x - bound)), 0)
        x1 = min(int(math.ceil(x + bound)), width)
        y0 = max(int(math.floor(y - bound)), 0)
        y1 = min(int(math.ceil(y + bound)), height)
        rolled = 0.0
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dx, dy = ix + 0.5 - x, iy + 0.5 - y
                d = math.sqrt(dx * dx + dy * dy)
                edge = min(bound - d, 1.0)
                if edge <= 0.0:
                    continue
                across = (d / bound) ** 2
                dent[iy, ix] = max(dent[iy, ix], sink * (1.0 - across))
                level = 1.0 - reach * pressure**BALL_POWER + rise * across
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
                for band in range(k.shape[0]):
                    ink[iy, ix, band] += dv * k[band]
                ink[iy, ix, IS] += dv * s
                ink[iy, ix, IV] += dv
                rolled += dv
        total += rolled
        gathered = min(gathered + BALL_GATHER * rolled, most)
    return total


@njit(cache=True, error_model="numpy")
def indent_stroke(dent, xs, ys, pressures, steps, radius, paper):
    """Press a hard round point of ``radius`` pixels (a stylus) into the
    paper through the points (``xs``, ``ys``), bearing with ``pressures``
    (0..1). It lays nothing and moves nothing: it leaves a groove, as deep
    under its middle as it is pressed hard and as the paper gives (``paper``
    is [depth, bite, give, hold]), shallower towards its rim. ``dent`` holds
    the deepest the paper has been pushed down at each pixel, in depths of
    the tooth of a paper of depth 1."""
    height, width = dent.shape
    bound = radius + 0.5
    for i in range(xs.shape[0]):
        if pressures[i] <= 0.0 or steps[i] <= 0.0:
            continue
        x, y = xs[i], ys[i]
        sink = INDENT * (GIVE_NONE + GIVE_FULL * paper[2]) * pressures[i]
        x0 = max(int(math.floor(x - bound)), 0)
        x1 = min(int(math.ceil(x + bound)), width)
        y0 = max(int(math.floor(y - bound)), 0)
        y1 = min(int(math.ceil(y + bound)), height)
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dx, dy = ix + 0.5 - x, iy + 0.5 - y
                across = (dx * dx + dy * dy) / (bound * bound)
                if across < 1.0:
                    dent[iy, ix] = max(dent[iy, ix], sink * (1.0 - across))


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
