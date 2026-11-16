"""The compiled kernels that lay a medium on the paper, and their constants;
and the two a render gathers its layers with (:func:`thicken`,
:func:`show_through`).

Kernels are ``@njit(cache=True, error_model="numpy")``, without ``fastmath``
and without ``parallel``, so that a drawing repeats exactly.

What a kernel calls for every pixel is compiled into its caller
(``inline="always"``), and written so that numba counts no reference for it.
Numba counts a reference to an array when a helper is given it, and one back
when the helper is done with it, each an atomic operation: left in, they
were half the time of a stroke and two thirds of that of a rub. It drops
the pair only where it can see the two as one: where the helper is done
with the array at one place, whichever way it went. So, of what is called
for every pixel:

- a helper is given numbers where the kernel has them, and returns numbers
  (:func:`loose_ground`; a layer at a pixel is four of them:
  :func:`layer_at`);
- one that is given an array to read reads what it needs of it first, before
  its first ``if`` (:func:`wax_grip`), or in loops, with no ``if`` before
  the last of them (:func:`within_ends`);
- one that changes an array, or reads it late, has no early ``return`` and
  no ``if`` that holds its last use of it: what it does only sometimes is in
  a loop that runs once or not at all, ``for _ in once(...)``
  (:func:`press_dust`, :func:`lay_load`). What a loop uses is done with
  where the loop ends, whatever went on inside it. All that is done
  sometimes goes in one such loop, and no ``break`` leaves a loop: one that
  must stop early is a ``while`` that says when (:func:`sunk_into`);
- a kernel does not choose between two such helpers with ``if`` and ``else``
  (:func:`push_along` is one helper for both kinds of tool).

And a kernel that is called for every step of a stroke (:func:`bear`,
:func:`rub_under`, :func:`roll_over`) calls no other kernel and allocates
no array (its caller gives it room to work in), or its own arrays are
counted at every call. ``tools/refcounts.py`` tells what each kernel still
counts, and ``tests/test_refcounts.py`` holds it at nothing as it works;
``tools/benchmark.py`` measures what that is worth. None of it changes a
result.

Speed must not change the picture: ``tools/fingerprint.py`` tells. What is
exact here and must stay so: a table holds values worked out by the very
expression it replaces (:func:`streaks_of`); a loop skips only pixels that
it would have left as they were (:func:`row_span`, :func:`settle_dust`),
and a helper only work that would have left the pixel as it is, taking
nothing from it and adding nothing to it (:func:`press_dust`,
:func:`stir_up` and :func:`spread_dust` under an eraser, :func:`burnish`
under a dry tip that drags nothing); a helper that reads a pixel first reads
what it would have read later, of what nothing has changed since; and no
sum is taken in another order.
"""

from __future__ import annotations

import math

import numpy as np
from murex.spectral import N_BANDS
from numba import njit

#: What a kernel calls for every pixel (see above).
_inlined = njit(cache=True, error_model="numpy", inline="always")


@_inlined
def once(some):
    """What ``for _ in once(some):`` runs over: once if ``some``, and not at
    all if not. It is an ``if`` for a helper that is given arrays (see
    above): the helper is done with what the loop uses where the loop ends,
    and not here on one path and there on another."""
    return range(1 if some else 0)


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
#: bonded, of three kinds: dust that is set, by fixative or by a film of wax
#: laid over it; a film, the share of a medium that would have been held had
#: it met the paper and not what lay on it (all that a waxy medium lays over
#: dust; what a dry one lays over a bed of dust); and paste, what the paper
#: does not hold of a greasy medium. A tool lifts what is set and a film only as
#: it lifts what the paper holds, and neither stirs them nor presses them in;
#: paste it moves as far as it smears.
SET, FILM, PASTE = 0, 1, 2
KINDS = 3
#: What a tool carries is packed as dust is, with two more parts: the volume
#: of it that is paste, and the volume of wax that came with what it took.
IP, IC = N + 3, N + 4
CARRIED = N + 5
#: The ink of pens is packed as dust is, with three parts more: the colour of
#: the light it mirrors where it is heavy and bronzes (linear RGB, its
#: brightest part 1), times its volume.
IB = N + 3
INK = N + 6

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
#: That is the cone of lead a pencil's sharpening bares, which bears hardest
#: along its middle, so its band fades to its edges. A stick of charcoal,
#: pastel or crayon laid flat (:attr:`fusain.Medium.flat_side`) bears
#: evenly all along its length instead, and its band is even across and
#: ends where the stick does, as sharply as the rim of its point
#: (:func:`side_bearing`).
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
#: A dry medium lies on what it meets too. What it meets on its own ground
#: is its own crumbs, which do not keep it from the paper: its binding was
#: set with them there. Dust beyond those (:func:`bed_of`) by more than a
#: whisper, BED_FROM thick, is a bed between the tip and the paper: the tip
#: rides on it as far as it covers the paper (all of it once the bed is
#: COVERS thick), and its mark lies there on the bed, a film as a waxy
#: medium's is, but one that binds nothing. The bed is not in its way, so
#: it takes none of the room in the tooth, and where the tip rides on it
#: the paper under it is out of reach: what fills the tooth there is what
#: lies on the bed (:func:`riding`). But dust scrapes less off a tip than
#: the paper does, loose (it rolls) or set: riding on a bed, a tip lays
#: BED_BITE less in a pass.
BED_FROM = 0.02
BED_BITE = 0.2
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
#: Pressed harder than BURNISH_FROM, a tip polishes a waxy layer flat
#: (``polish``, the wax's own), and a pressed tooth is level, so the white
#: specks close up. It sets in by degrees: the wax (and, under a dry tip,
#: the paper) does not give way at one pressure everywhere, but at pressures
#: spread evenly over BURNISH_SOFT either side of BURNISH_FROM
#: (:func:`firmness`). So of the strokes of a shading near there, each a
#: little firmer or lighter than the next, none polishes all and none
#: nothing.
#:
#: A pressure polishes as far as it can, however often it is repeated: a
#: tip at full pressure has a load of BURNISH on a film as slick as it gets
#: (its firmness times that at a lighter pressure), and polishes the film
#: to the level of its load, at most 1 (:func:`settled`). So anything from
#: a pressure of 0.87 on polishes a slick film flat, 0.7 polishes it to
#: 0.42, 0.6 to 0.19 and 0.5 to 0.05, and no number of passes further. (The
#: load is what a pass at that pressure added to one ``press`` while every
#: pass added to it: twelve layers at 0.6 then polished as a firm one.)
BURNISH_FROM = 0.6
BURNISH_SOFT = 0.2
#: Where it sets in, BURNISH_FROM - BURNISH_SOFT, written out: in floating point
#: 0.6 - 0.2 is not 0.4, and a tip at 0.4 would have a firmness of 1e-33.
BURNISH_ONSET = 0.4
BURNISH = 1.5
#: A pass brings what it presses flat towards the level of its load, and
#: each part of the pass closes a share of the gap that is left: SETTLE
#: times the share of a pass that it is. The paper under the flat of a tip
#: has well under a pass's worth of the tip's full bearing (its rim bears
#: less, and not all of a pixel is touched), so a pass over a tooth left
#: alone ends at about three quarters of its level, which is about where
#: it ended while every pass added its load; a second closes most of what
#: is left, and a third changes next to nothing.
SETTLE = 2.0
#: Polishing also smears the waxy layer: of what the paper holds there (and
#: of a film of wax over the dust there), the tip drags along this share at
#: full pressure in one pass, and lays this many times what it drags back
#: down in the tooth over each of its own widths, valleys included: within
#: a pixel or two, so that it closes the white specks next to it without
#: carrying colour far.
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
#: A firm dry tip flattens the tooth itself (``crush``, the paper's own):
#: pressed harder than BURNISH_FROM (setting in by degrees, as above), the
#: hardest lead has a load of FLATTEN at full pressure, and crushes the
#: tooth to the level of its load, at most 1, as a film is polished: flat,
#: from far less than full pressure. A softer tip gives before the paper
#: does (its load goes as the cube of its hardness), and one that crumbles
#: to dust bears on the paper only with what holds together (as the square
#: of its binding): at full pressure a 4H crushes the tooth flat, an HB to
#: 0.38, a 2B to 0.18 and a 6B to 0.014, however long they are rubbed. So a
#: hard pencil pressed firmly levels the tooth, on which its graphite then
#: lies flat and shines evenly; a soft one hardly does, and charcoal and
#: pastel next to not at all. (A waxy tip polishes its film, as above.)
FLATTEN = 5.0
#: The tooth is what scrapes a dry medium off its stick, and a tooth pressed
#: flat bites less: a dry medium lays this much less on it in a pass. It
#: lays it evenly, though: all of a flattened tooth is touched. So with the
#: floor of a groove, where the point that pressed it took the peaks off.
FLAT_BITE = 0.7
#: A hard round point pressed into the paper leaves a groove. ``dent`` is
#: how far below the peaks of its tooth the paper has been pushed down, in
#: depths of the tooth of a paper of depth 1: nothing stands higher there,
#: so a tip wider than the groove, which rides on the paper either side,
#: skips it unless it reaches that far down; one narrower than the groove
#: goes down into it (:func:`sunk_into`). A stylus at full pressure
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
#: radius (both times how much the pen blots: :attr:`fusain.Pen.blots`);
#: what it would gather beyond that it does not, and that ink is nowhere
#: (where it would go is not modelled). A turn brings it round onto the
#: paper, all of it when the line turns by BALL_TURN radians, so a sharp
#: turn leaves a blob and a gentle curve nothing to see. The paper takes of
#: what comes round as much as the ball bears on it, the share of a firm
#: pass that the line lays there (from BALL_LIGHT at the lightest touch to
#: all of it at full pressure), and the rest stays on the rim: a light line
#: blots as lightly as it is drawn. A pen put to the paper has some on it
#: already. A pen in the hand lands with what it gathered before, and where
#: it lands at most the share BALL_START of that comes round, mostly next to
#: none (a number drawn between 0 and 1, to the power BALL_SELDOM, of it),
#: so that one line in four or so starts with a dot that shows; the rest
#: comes round at the next turn. A pen just picked up lands with that share
#: of the most the ball holds, all of which comes round.
BALL_GATHER = 0.05
BALL_BLOB = 4.0
BALL_TURN = 2.0
BALL_START = 0.8
BALL_SELDOM = 6.0
#: A drop too small to show stays on the rim, until the line has turned far
#: enough to bring round one that does: one of BALL_SHOWS firm passes over
#: the square of the ball's radius (what the ball holds at most is BALL_BLOB
#: of them), which lies at the middle of its dot about 2% as thick as the
#: line it is left on.
BALL_SHOWS = 0.05
#: A drop of ink is left as a round dot BLOB_WIDE times as wide as the ball
#: is where it meets the paper (where its surface comes down to the peaks of
#: the tooth: all of it pressed firmly, a narrower disc pressed lightly, as
#: the line is narrower), thickest at its middle: a little wider than the
#: line, and darker.
BLOB_WIDE = 1.2
#: The ball skids on a film of pigment: it lays less from the first of these
#: thicknesses (per covered area) and nothing from the second. On wax it
#: writes as badly as a dry medium does (RESIST_DRY, on a film COVERS thick).
SKID_FROM = 0.25
SKID_FULL = 1.0
#: A tool that rubs also spreads the dust under it from grain to grain: what
#: it takes onto its face that way (as strongly as it smears) it lays back
#: this many times over each of its own widths, so within a few pixels. That
#: is what evens a rubbed tone before the dust is worked into the paper, and
#: it is most of what a tool moves: the dust on its face goes a fraction of
#: its width (taken up and laid back several times, half a width or so), so
#: a join of two colours rubbed across is a passage a width or two wide.
#: Only what a tool lifts into its load (:attr:`fusain.Tool.lift_loose`,
#: little for a stump or a finger) is carried far, and it is given back to
#: bare tooth and to no other paper (:func:`hunger`).
SPREAD = 16.0
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
#: a medium without wax. Polished wax keeps the tooth level as far as it is
#: WAX_FAST thick, and as a tool takes it off the tooth comes back
#: (:func:`bare_tooth`).
WAX_FAST = 0.1
WAX_HOLDS = 0.6
POLISH_HOLDS = 0.9
#: Wax carries its pigment into the fibres of the paper, which it stains: of
#: what the paper holds of a waxy medium, this thickness (on the covered
#: share of the pixel) is out of reach of any eraser that does not cut the
#: paper, so a tint stays where a layer has been erased (:func:`stained`).
#: The fibres take only so much, so it is a thickness and not a share: the
#: stain of a light layer is most of what a heavy one leaves. A knife or a
#: sand eraser takes it as far as it scrapes.
STAIN = 0.008
#: An eraser slides on wax that holds against it, and drags some of it
#: along: this share of what it would have lifted, had the wax not held
#: (:func:`push_along`). It lays that back, held, DRAG_LAY of it over each
#: of its widths, on what is beside it and after it: a vinyl eraser smears
#: a burnished dark a width or so along its way.
WAX_DRAG = 0.05
DRAG_LAY = 1.0
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
    known = min(max(band, 0), STREAK_BANDS - 1)
    low, high = streaks[known], streaks[known + 1]
    if known != band:  # (a tip too wide for the table)
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
def side_bearing(dx, dy, tx, ty, wide, long, rounded):
    """As :func:`tip_bearing`, for the side of a stick laid flat: it bears
    alike all along its length, ``wide`` across the stroke (a half-width,
    to the tip of its end), and is ``long`` along it (half its thickness)
    with its ends rounded: it covers what lies within ``long`` of the line
    across the stroke through its middle, short of ``long`` from its ends.
    So the band it lays is even across, and ends within its thickness."""
    forward, sideways = dx * tx + dy * ty, dy * tx - dx * ty
    beyond = max(abs(sideways) - max(wide - long, 0.0), 0.0)
    inside = long - math.sqrt(forward * forward + beyond * beyond)
    if inside <= 0.0:
        return 0.0, 0.0
    return min(inside / (rounded * long + 0.5), 1.0), min(inside, 1.0)


@_inlined
def bearing_at(dx, dy, now):
    """How the tip at ``now`` (see :func:`bear`) bears on the pixel ``dx``,
    ``dy`` from its centre: :func:`side_bearing` for the side of a stick
    laid flat, else :func:`tip_bearing`."""
    tx, ty, wide, long, rounded = now[2], now[3], now[4], now[5], now[6]
    if now[13] > 0.0:
        return side_bearing(dx, dy, tx, ty, wide, long, rounded)
    return tip_bearing(dx, dy, tx, ty, wide, long, rounded)


@_inlined
def side_span(x, dy, tx, ty, wide, long, x0, x1):
    """As :func:`row_span`, for the side of a stick laid flat
    (:func:`side_bearing`): the pixels within ``long`` of the line across
    the stroke through its middle, on the row ``dy`` below it."""
    core = max(wide - long, 0.0)
    near, far = -core, core  # (the part of that line near enough the row)
    if abs(tx) > 1e-9:
        low, high = (dy - long - 1.0) / tx, (dy + long + 1.0) / tx
        near, far = max(min(low, high), -core), min(max(low, high), core)
    elif abs(dy) > long + 1.0:
        near, far = 0.0, -1.0
    if near > far:
        return x0, x0
    a, b = x - ty * near, x - ty * far
    return max(int(math.floor(min(a, b) - long - 1.0)), x0), min(
        int(math.ceil(max(a, b) + long + 1.0)), x1
    )


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
    dusty = loose[iy, ix, IV] > 0.0 or under[iy, ix, IV] > 0.0
    # (else no dust here, most pixels of a first pass; or a tool that
    # presses nothing in, an eraser)
    for _ in once(dusty and share > 0.0):
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
def firmness(bearing):
    """How hard a tip bearing with ``bearing`` (0..1) polishes or flattens
    what it bears on, 1 at full pressure (its load on what it presses flat
    is this times BURNISH or the flattening of its medium, and what it
    smears goes by it too): how far it bears harder than what
    gives way, as a share of how far full pressure does, where what gives
    way does so at pressures spread evenly over BURNISH_SOFT either side of
    BURNISH_FROM. So it is nothing up to BURNISH_SOFT below that, then
    grows as the square of the pressure past there, and from BURNISH_SOFT
    above it on is ``(bearing - BURNISH_FROM) / (1 - BURNISH_FROM)``, as
    it would be were there no spread: no corner where it sets in."""
    if bearing <= BURNISH_ONSET:
        return 0.0
    over = bearing - BURNISH_FROM
    if over < BURNISH_SOFT:
        over = (over + BURNISH_SOFT) ** 2 / (4.0 * BURNISH_SOFT)
    return over / (1.0 - BURNISH_FROM)


@_inlined
def settled(pressed, load, slick, passing):
    """How far a tooth is pressed flat (the crush of the paper, or the
    polish of its wax: ``pressed``, 0..1) after the share ``passing`` of a
    pass of a tip whose load is ``load`` (the flattening of its medium, or
    BURNISH, times its :func:`firmness`), on a film that is ``slick`` (1
    for the paper).

    A load presses to a level and no further: ``load``, at most 1, of how
    slick the film is. What is flatter than that already stays as it is: a
    lighter pass over a firmer one changes nothing. What is not goes
    towards it, by the share of the gap that is left that SETTLE times this
    part of the pass is (and as many times more as the load exceeds 1: the
    hardest leads are there at once). So the first of a pass over a tooth
    left alone raises it SETTLE times as fast as a pass did while every
    pass added its load to what was there, and the nearer the level, the
    less."""
    level = min(load, 1.0) * slick
    if level <= pressed:
        return pressed
    return pressed + (level - pressed) * min(SETTLE * max(load, 1.0) * passing, 1.0)


@_inlined
def press_flat(
    crush, polish, iy, ix, crushed, polished, bearing, flattens, slick, passing
):
    """Press a pixel flat as a tip that bears on it with ``bearing`` (its
    pressure there, the streaks of its lead aside) does in the share
    ``passing`` of a pass: a dry tip crushes the paper (``crush``; its load
    at full pressure is ``flattens``), and any tip polishes a film of wax
    as far as it is ``slick`` (``polish``), each towards the level of its
    load (:func:`settled`). ``crushed`` and ``polished`` are what the pixel
    holds of the two. Returns how level the tooth then is to a tip.

    (A tip that bears too lightly here, a waxy one on the paper and any
    tip where no wax is slick would leave what they do not press as it
    is: nothing is written of it.)"""
    steady = firmness(bearing)
    for _ in once(steady > 0.0):
        if flattens > 0.0:
            crush[iy, ix] = settled(crushed, flattens * steady, 1.0, passing)
            crushed = crush[iy, ix]
        if slick > 0.0:
            polish[iy, ix] = settled(polished, BURNISH * steady, slick, passing)
            polished = polish[iy, ix]
    return crushed + polished - crushed * polished


@_inlined
def drag_film(layer, bonded, drag, iy, ix, which, up):
    """Take the share ``up`` of the film in one of the two layers at a pixel
    (``which``: 0 underneath, 1 on top) onto a tip that polishes it
    (``drag``): its volume, and that share of the layer with it. (What lies
    in a layer is one mixture: a tool that lifts a film lifts it so too.)"""
    film = bonded[iy, ix, which, FILM]
    there = layer[iy, ix, IV]
    for _ in once(film > 0.0 and there > 0.0):
        share = up * min(film / there, 1.0)
        bonded[iy, ix, which, FILM] = film - film * up
        for part in range(LOOSE):
            taken = layer[iy, ix, part] * share
            layer[iy, ix, part] -= taken
            drag[part] += taken


@_inlined
def burnish(held, under, loose, bonded, drag, iy, ix, polish, lift, lay):
    """Smear a pixel that a tip polishes by ``polish`` (the share of one
    full pass on a film as slick as it gets): take ``lift`` times that share
    of what it holds onto the tip (``drag``, packed as the held layer is),
    and of a film that lies over the dust there (``bonded``), and lay the
    share ``lay`` of what the tip drags back down on it, in the tooth. (How
    far the wax is polished level is :func:`press_flat`'s to say.)

    Wax is wax wherever it lies, and the tip shears a film on dust as it
    shears one in the tooth: the wax of a film is counted with what the
    paper holds, and was dragged by that share already, its pigment not. A
    firm tip presses what it smears down into the valleys of the tooth,
    through the dust, which it presses in more readily once the film over
    it has gone (:func:`press_dust`)."""
    up = min(lift * polish, 1.0)
    down = min(lay, 1.0)
    drag_film(under, bonded, drag, iy, ix, 0, up)
    drag_film(loose, bonded, drag, iy, ix, 1, up)
    for part in range(HELD):
        taken = held[iy, ix, part] * up
        drag[part] += taken
        given = drag[part] * down
        drag[part] -= given
        held[iy, ix, part] += given - taken


@_inlined
def anything(packed):
    """Whether there is anything of what a tip drags or a tool carries
    (``packed``): any part of it at all."""
    some = False
    for part in range(packed.shape[0]):
        some = some or packed[part] != 0.0
    return some


@njit(cache=True, error_model="numpy")
def set_down(held, cover, drag, now):
    """Where the tip lifts off, at ``now`` (see :func:`bear`), lay what it
    still drags (``drag``, packed as the held layer is) on the marked paper
    under it. Returns the volume it could not lay: all of it, if no marked
    paper of the sheet is under the tip."""
    if drag[IV] <= 0.0:
        return 0.0
    height, width = cover.shape
    x, y = now[0], now[1]
    bound = max(now[4], now[5])
    x0 = max(int(math.floor(x - bound)), 0)
    x1 = min(int(math.ceil(x + bound)), width)
    y0 = max(int(math.floor(y - bound)), 0)
    y1 = min(int(math.ceil(y + bound)), height)
    under = 0.0
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            if cover[iy, ix] > 0.0:
                under += bearing_at(ix + 0.5 - x, iy + 0.5 - y, now)[1]
    if under <= 0.0:
        return drag[IV]
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            if cover[iy, ix] > 0.0:
                edge = bearing_at(ix + 0.5 - x, iy + 0.5 - y, now)[1]
                for part in range(HELD):
                    held[iy, ix, part] += drag[part] * edge / under
    return 0.0


@_inlined
def loose_in(bonded, layer, iy, ix, which):
    """The loose dust in one layer at a pixel (``which``: 0 underneath, 1 on
    top): what is neither set, nor a film, nor paste."""
    there = bonded[iy, ix, which, SET] + bonded[iy, ix, which, FILM]
    there += bonded[iy, ix, which, PASTE]
    return max(layer[iy, ix, IV] - there, 0.0)


@_inlined
def bond_dust(bonded, iy, ix, which, caught, film):
    """Bond the volume ``caught`` of the loose dust in one layer at a pixel:
    the share ``film`` of it is set, as under a film, and the rest turns to
    paste."""
    bonded[iy, ix, which, SET] += caught * film
    bonded[iy, ix, which, PASTE] += caught * (1.0 - film)


@_inlined
def settle_pixel(fresh, cover, bonded, under, loose, pasty, binds, iy, ix):
    """Make room at a pixel for what a pass has left to settle there
    (``fresh``: what the paper did not hold of it, and its film): what lay on
    top goes under, with what is bonded of it, and the film and the paste
    that come bond part of the loose dust (the share ``binds`` of it under
    a film or paste thick enough to cover it: nothing under the film of a
    dry medium). Returns the volumes of dust and paste, and of film, to lay
    on top; ``fresh`` is left empty."""
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
    # (the loose dust of each layer, before any of it is bonded)
    below = loose_in(bonded, under, iy, ix, 0)
    on_top = loose_in(bonded, loose, iy, ix, 1)
    if binder > 0.0 and binds > 0.0:
        share = binds * min(binder * thin, 1.0)
        bond_dust(bonded, iy, ix, 0, below * share, film / binder)
        bond_dust(bonded, iy, ix, 1, on_top * share, film / binder)
    bonded[iy, ix, 1, FILM] += film
    bonded[iy, ix, 1, PASTE] += paste
    return rest, film


@_inlined
def settle_row(
    fresh, cover, bonded, under, loose, k, s, sheen, pasty, binds, iy, x0, x1
):
    """Settle what a pass has left on the pixels ``x0`` to ``x1`` of a row
    (:func:`settle_dust`)."""
    for ix in range(x0, x1):
        if fresh[iy, ix, 0] <= 0.0 and fresh[iy, ix, 1] <= 0.0:
            continue  # (nothing, or settled already)
        rest, film = settle_pixel(
            fresh, cover, bonded, under, loose, pasty, binds, iy, ix
        )
        new = rest + film
        for band in range(k.shape[0]):
            loose[iy, ix, band] += new * k[band]
        loose[iy, ix, IS] += new * s
        loose[iy, ix, IV] += new
        loose[iy, ix, IX] += new * sheen


@njit(cache=True, error_model="numpy")
def settle_dust(
    fresh, cover, bonded, under, loose, k, s, sheen, pasty, binds, xs, ys, bound
):
    """At the end of a pass, put what it left to lie on the sheet (``fresh``,
    two volumes per pixel: what the paper does not hold of the medium, of
    which the share ``pasty`` is paste and the rest dust, and its film) on
    top of what lay there, which goes under it with what is bonded of it
    (``bonded``), and leave ``fresh`` empty again. A film of wax and paste
    bond part of the loose dust they come to lie on (``binds``: the share
    of it, under a film or paste thick enough to cover it; nothing under
    the film a dry medium leaves on a bed of dust). Only where the tip went:
    within ``bound`` of the points (``xs``, ``ys``)."""
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
                    binds,
                    iy,
                    x0,
                    min(x1, px0),
                )
                start = max(x0, px1)
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
                binds,
                iy,
                start,
                x1,
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
def bed_of(held, under, loose, bonded, covered, iy, ix, crumbs):
    """The bed of dust a dry tip meets at a pixel: how thick it is over the
    marked part of the pixel (``covered``), and the share of it that is
    loose, the rest being set. It is the dust, loose or set, beyond the
    crumbs the tip's own mark would have for what is fast there (``crumbs``
    times what the paper holds, a film and paste: how much dust the medium
    leaves for each part that is held), by more than BED_FROM. (0, 0) on a
    ground no dustier than the tip's own mark: bare paper, its own layers,
    or those of a medium that holds better."""
    lying = under[iy, ix, IV] + loose[iy, ix, IV]
    fixed = bonded[iy, ix, 0, SET] + bonded[iy, ix, 1, SET]
    stuck = bonded[iy, ix, 0, FILM] + bonded[iy, ix, 1, FILM]
    stuck += bonded[iy, ix, 0, PASTE] + bonded[iy, ix, 1, PASTE]
    fast = held[iy, ix, IV] + stuck
    if lying <= 0.0:
        return 0.0, 0.0  # (no dust here: most pixels of a first pass)
    dust = lying - stuck
    bed = (dust - crumbs * fast) / covered - BED_FROM
    if bed <= 0.0:
        return 0.0, 0.0
    return bed, min(max(dust - fixed, 0.0) / dust, 1.0)


@_inlined
def riding(held, under, loose, bonded, covered, iy, ix, crumbs, beds):
    """What a dry tip finds of a bed of dust at a pixel (:func:`bed_of`;
    ``beds`` is how far the tip is dry): the thickness that is not in its
    way in the tooth, and the share of the pixel on which it rides on the
    bed, where the mark the paper would have held is a film on the dust.
    The bed is not in its way, whether the tip ploughs through it or rides
    on it; and where it rides, the paper under the bed is out of its reach,
    so what the paper holds there is not either (with the crumbs counted
    for it): what fills the tooth for it there is what lies on the bed."""
    bed, afloat = bed_of(held, under, loose, bonded, covered, iy, ix, crumbs)
    paper = held[iy, ix, IV]
    if bed <= 0.0:
        return 0.0, 0.0
    rides = beds * min(bed / COVERS, 1.0)
    # (dust that is set took less of the room than loose dust)
    took = afloat + FIXED_ROOM * (1.0 - afloat)
    below = (1.0 + crumbs * took) * paper / covered
    return beds * bed * took + rides * below, rides


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
def span_of(now, dy, lean, half, top, x0, x1):
    """The pixels the tip at ``now`` (see :func:`bear`) can cover on the row
    ``dy`` below its centre, among those from ``x0`` to ``x1``: as
    :func:`row_span` (``lean``, ``half`` and ``top`` are :func:`tip_extent`),
    or :func:`side_span` for the side of a stick laid flat."""
    if now[13] > 0.0:
        return side_span(now[0], dy, now[2], now[3], now[4], now[5], x0, x1)
    return row_span(now[0], dy, lean, half, top, x0, x1)


@_inlined
def sunk_into(dent, now, x0, x1, y0, y1, lean, half, top):
    """How far down a groove the tip at ``now`` (see :func:`bear`) goes, as
    ``dent`` counts it: as far as the least indented of the paper it covers
    (``x0`` to ``x1``, ``y0`` to ``y1`` around it; the rest as for
    :func:`span_of`). It rests on the highest of what is under it: a tip
    wider than a groove rides on the paper either side and stays up; one
    narrower than the groove under it goes down into it."""
    height, width = dent.shape
    x, y = now[0], now[1]
    cx, cy = int(math.floor(x)), int(math.floor(y))
    # (the paper under its middle is not indented, most often: no row to see)
    level = 0 <= cy < height and 0 <= cx < width and dent[cy, cx] <= 0.0
    lowest = math.inf
    iy = y1 if level else y0
    # (as far as the first paper it covers that is not indented; loops that
    # end on that, where a ``return`` would be: see the top of the file)
    while iy < y1 and lowest > 0.0:
        dy = iy + 0.5 - y
        xa, xb = span_of(now, dy, lean, half, top, x0, x1)
        ix = xa
        while ix < xb and lowest > 0.0:
            if dent[iy, ix] < lowest and bearing_at(ix + 0.5 - x, dy, now)[1] > 0.0:
                lowest = dent[iy, ix]
            ix += 1
        iy += 1
    return lowest if 0.0 < lowest < math.inf else 0.0


@_inlined
def within_ends(px, py, ends):
    """How much of the pixel centred at (``px``, ``py``) a stroke cut square
    at its ends leaves: 1 but behind the line across its start or beyond the
    line across its end, where it is cut; and there too it is left where it
    lies in the band of the stroke itself, which a path that bends within
    the tip's reach of an end takes across that line.

    ``ends`` is [how many lines it holds for the start of the stroke, how
    many for its end, then for each line: x, y, the way the path goes there
    (x, y), how far either side of the path its band reaches] (see
    :func:`ends_near`). The first line of an end is the cut itself, right
    across the sheet; the others are across the path at the samples near
    that end, as far as the band of the stroke reaches. A pixel is left as
    far as it is on the stroke's side of any of them and in its band: ahead
    of it for the start, behind it for the end. (No ``if`` here but the
    last: what it reads of ``ends`` it reads in loops.)"""
    starts, stops = int(ends[0]), int(ends[1])
    kept = 1.0 - min(starts, 1)  # (all of it where no line is held)
    for line in range(starts):
        at = 2 + 5 * line
        dx, dy, tx, ty = px - ends[at], py - ends[at + 1], ends[at + 2], ends[at + 3]
        ahead = dx * tx + dy * ty
        beside = ends[at + 4] - abs(dy * tx - dx * ty)
        kept = max(kept, min(min(max(ahead + 0.5, 0.0), 1.0), max(beside, 0.0)))
    left = 1.0 - min(stops, 1)
    for line in range(starts, starts + stops):
        at = 2 + 5 * line
        dx, dy, tx, ty = ends[at] - px, ends[at + 1] - py, ends[at + 2], ends[at + 3]
        short = dx * tx + dy * ty
        beside = ends[at + 4] - abs(dy * tx - dx * ty)
        left = max(left, min(min(max(short + 0.5, 0.0), 1.0), max(beside, 0.0)))
    share = min(kept, left)
    return share if share > 1e-6 else 0.0  # (what rounding leaves past it)


@_inlined
def bear_on_row(row, now, iy, xa, count, ends, streaks, salt):
    """How a tip at ``now`` (see :func:`bear`) bears on ``count`` pixels of
    the row ``iy``, from ``xa`` on, which depends on nothing the sheet holds:
    for each, in the five rows of ``row`` (counted from ``xa``), how far inside
    the edge of the tip it is (0 if the tip does not mark it), how hard the
    tip bears on it, where it does mark it the level in the tooth it
    reaches there in this pass, how far that part of the tip slides
    over the paper for each pixel its middle goes (1, but where the stroke
    turns: ``now[14]``, the turn per pixel, which way the tip's left goes),
    and how hard the tip bears on it but for the streaks of its lead: its
    pressure, as that part of its face has it (what presses the tooth flat:
    :func:`settled`).

    It is worked out for the whole row before anything is laid, in loops
    that do one thing each: the processor then works on several pixels at
    once, where one loop that also laid the medium went a pixel at a time
    (each has its divisions, its square roots and a power to wait for)."""
    x, y, tx, ty = now[0], now[1], now[2], now[3]
    reach, pressure, at_x, at_y, turn = now[7], now[8], now[11], now[12], now[14]
    cuts = ends[0] + ends[1]  # (how many lines cut the stroke square here)
    dy = iy + 0.5 - y
    for at in range(count):
        ix = xa + at
        dx = ix + 0.5 - x
        share, edge = bearing_at(dx, dy, now)
        # (landing or lifting: only where it is put)
        put = edge > 0.0 and at_x == at_x
        if put and bearing_at(ix + 0.5 - at_x, iy + 0.5 - at_y, now)[1] <= 0.0:
            edge = 0.0
        row[0, at] = edge
        row[1, at] = pressure * share * streak_at(dx, dy, tx, ty, streaks, salt)
        row[4, at] = pressure * share
        # (where the stroke turns, the side of the tip on the outside of the
        # turn slides further than its middle, and the inside less)
        row[3, at] = abs(1.0 - turn * (dy * tx - dx * ty))
    # (cut square, in a loop of its own: only a tip near an end that is cut
    # square goes through it)
    for at in range(count if cuts > 0.0 else 0):
        if row[0, at] > 0.0:
            row[0, at] *= within_ends(xa + at + 0.5, iy + 0.5, ends)
    for at in range(count):
        ix = xa + at
        if row[0, at] > 0.0:
            level = 1.0 - reach * row[1, at] ** REACH_POWER
            level += JITTER * (unit_noise(ix, iy, salt) - 0.5)
            row[2, at] = level


@njit(cache=True, error_model="numpy")
def bear(
    tooth,
    crush,
    polish,
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
    this paper's tooth one of a paper of depth 1 is, its load on the tooth
    at full pressure, which crushes it (:func:`settled`), how much less of
    it a flattened tooth takes, how far it melts into a film of wax it polishes, the
    crumbs its mark has for each part of it that the paper holds, how far
    it meets a bed in dust beyond those (:func:`bed_of`)]. ``ends``
    is as for :func:`within_ends` and ``streaks`` as for :func:`streak_at`;
    ``row`` is room to work in, five rows as wide as the sheet
    (:func:`bear_on_row`). ``now`` goes on with [whether the tip is the side
    of a stick laid flat (:func:`side_bearing`), how fast the stroke turns
    (:func:`turning`)].

    The tooth it meets is level as far as the paper has been crushed
    (``crush``) and the wax on it polished (``polish``); pressed firmly, a
    dry tip raises the one and any tip on wax the other, each towards the
    level that its load reaches and no further (:func:`settled`).

    Returns the volume laid and what of it is left to settle on top."""
    height, width = tooth.shape
    x, y, tx, ty, wide, long = now[0], now[1], now[2], now[3], now[4], now[5]
    per_pass, dragged = now[9] * step, now[10] * step
    lay, full, dry, slides, filmy = law[0], law[1], law[2], law[3], law[4]
    binding, s, sheen, wax = law[5], law[6], law[7], law[8]
    sunk, flattens, smooth, melts = law[9], law[10], law[11], law[12]
    crumbs, beds = law[13], law[14]
    # (a tip that melts into the film stirs more of it, and lays it sooner)
    stir = BURNISH_LIFT + (MELT_LIFT - BURNISH_LIFT) * melts
    dragged *= 1.0 + (MELT_LAY / BURNISH_LAY - 1.0) * melts
    bound = max(wide, long)
    x0 = max(int(math.floor(x - bound)), 0)
    x1 = min(int(math.ceil(x + bound)), width)
    y0 = max(int(math.floor(y - bound)), 0)
    y1 = min(int(math.ceil(y + bound)), height)
    lean, half, top = tip_extent(tx, ty, wide, long)
    # (a tip narrower than a groove goes down into it)
    down = sunk_into(dent, now, x0, x1, y0, y1, lean, half, top) * sunk
    dragging = anything(drag)  # (most tips never drag anything)
    # (a tip that bears too lightly to press anything flat under its middle
    # presses nothing flat anywhere: most strokes of a light drawing)
    presses = firmness(min(now[8], 1.0)) > 0.0
    laid = 0.0
    lying = 0.0
    for iy in range(y0, y1):
        xa, xb = span_of(now, iy + 0.5 - y, lean, half, top, x0, x1)
        count = xb - xa
        if count <= 0:  # (none of this row is under the tip)
            continue
        # first how the tip bears on each pixel of the row, which is its
        # shape and nothing else; then what that does to the paper
        bear_on_row(row, now, iy, xa, count, ends, streaks, salt)
        for at in range(count):
            ix = xa + at
            edge = row[0, at]
            if edge <= 0.0:
                continue
            local, level, slide = row[1, at], row[2, at], row[3, at]
            # (to a tip riding on it a tooth pressed flat is level with its
            # peaks, whether the paper is crushed or its wax polished: the
            # wax levels its share of the relief the crush left; and nothing
            # in a groove stands higher than its floor)
            crushed, polished = crush[iy, ix], polish[iy, ix]
            flat = crushed + polished - crushed * polished
            raised = tooth[iy, ix] + (1.0 - tooth[iy, ix]) * flat
            floor = 1.0 - dent[iy, ix] * sunk
            above = min(raised, floor) + MICRO - level + down
            touch = min(max(above / (2.0 * MICRO), 0.0), 1.0) * edge
            if touch <= 0.0:
                continue
            covered = max(cover[iy, ix], touch)
            cover[iy, ix] = covered
            there = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
            thick = (there + fresh[iy, ix, 0] + fresh[iy, ix, 1]) / covered
            was_set = bonded[iy, ix, 0, SET] + bonded[iy, ix, 1, SET]
            thick -= (1.0 - FIXED_ROOM) * was_set / covered
            # (a bed of dust is not in a dry tip's way, nor is what the
            # paper holds under it where the tip rides on the bed)
            rides = 0.0
            if beds > 0.0:
                freed, rides = riding(
                    held, under, loose, bonded, covered, iy, ix, crumbs, beds
                )
                thick -= freed
            bearing = min(local, 1.0)
            passing = per_pass * slide * touch
            waxed = held[iy, ix, IW] / covered
            waxy = min(waxed / WAX_FULL, 1.0)
            # (a blender melts into the film, however thin it is)
            slick = waxy + (1.0 - waxy) * melts
            # what is pressed flat is pressed by the tip, not by the grain of
            # its lead: as firmly as the tip bears here, its streaks aside (a
            # hard tip crushes the tooth, whatever lies on it, and any tip
            # polishes a film of wax: each as far as its load can)
            if presses:
                flat = press_flat(
                    crush,
                    polish,
                    iy,
                    ix,
                    crushed,
                    polished,
                    min(row[4, at], 1.0),
                    flattens,
                    slick,
                    passing,
                )
            firm = firmness(bearing)
            if firm > 0.0:
                work, lay_back = firm * slick * passing, dragged * slide * touch
                # (a tip that smears nothing here and drags nothing would
                # leave the pixel as it is: a dry tip on paper without wax)
                if work > 0.0 or dragging:
                    dragging = True
                    burnish(
                        held, under, loose, bonded, drag, iy, ix, work, stir, lay_back
                    )
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
            # (and a flattened tooth bites it less, as does the floor of a
            # groove, which the point that pressed it left as flat)
            levelled = min(max((raised - floor) / MICRO, 0.0), 1.0)
            dv *= 1.0 - smooth * max(flat, levelled)
            # (and dust scrapes less off a tip that rides on it)
            dv *= 1.0 - BED_BITE * rides
            dv = min(dv * per_pass * slide, room) * touch
            # the share of what is held that lies on what it met and not in
            # the tooth, a film: all that a waxy tip lays over dust, and
            # what a dry one lays where it rides on a bed
            over = rides
            if filmy > 0.0:
                over += filmy * lying_share(under, loose, bonded, covered, iy, ix)
            lying += deposit(held, fresh, iy, ix, dv, binding, over, k, s, sheen, wax)
            laid += dv
    return laid, lying


@njit(cache=True, error_model="numpy")
def turning(xs, ys, i):
    """How fast the tip turns at sample ``i`` of the path through the points
    (``xs``, ``ys``): in radians per pixel, positive where it turns towards
    the side of the tip :func:`streak_at` counts as across it the positive
    way. Nothing at its ends. A tip is the same turned half round, so where
    the path turns by more than a quarter turn, the hand turns it the short
    way and draws it back: where a stroke doubles back on itself (to and
    fro), the tip does not turn at all."""
    if i <= 0 or i >= xs.shape[0] - 1:
        return 0.0
    ax, ay = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
    bx, by = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
    going = math.sqrt(ax * ax + ay * ay) + math.sqrt(bx * bx + by * by)
    if going <= 1e-9:
        return 0.0
    angle = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
    if angle > 0.5 * math.pi:
        angle -= math.pi
    elif angle < -0.5 * math.pi:
        angle += math.pi
    return 2.0 * angle / going


@_inlined
def in_parts(step, angle, wide, per_pass):
    """In how many parts a tip lays what it leaves as it slides ``step``
    pixels, turning by ``angle`` as the stroke turns: in one, all at once,
    where no part of the tip slides over a pixel more than once in the
    step, the most it can (a pass's worth: ``per_pass`` per pixel slid).
    Where the stroke turns so sharply that the far end of the tip (``wide``
    from its middle) would sweep further than that, in as many parts as
    that takes, the tip turned bit by bit across the angle it turns through
    in the step (:func:`turned`): what it sweeps is spread along the arc it
    sweeps, a pass's worth at most on each pixel at a time, and not piled
    on the pixels under it."""
    return max(int(math.ceil((step + wide * abs(angle)) * per_pass)), 1)


@_inlined
def turned(tx, ty, angle, part, parts):
    """The heading of a tip going the way (``tx``, ``ty``) for the ``part``
    of the ``parts`` in which it is laid as it turns by ``angle``
    (:func:`in_parts`): as it is, laid all at once."""
    if parts <= 1:
        return tx, ty
    turn = angle * ((part + 0.5) / parts - 0.5)
    cos, sin = math.cos(turn), math.sin(turn)
    return tx * cos - ty * sin, ty * cos + tx * sin


@njit(cache=True, error_model="numpy")
def tip_at(now, tilt, side, flat, reach, i, xs, ys, pressures, even):
    """Fill ``now`` (see :func:`bear`) for a tip ``flat`` wide at sample
    ``i``, held at ``tilt``, its side ``side`` long, whose side
    lies flat along its length (``even`` 1: a stick of charcoal, pastel or
    crayon) or is the cone of a sharpened lead (0). Returns how far from its
    centre it reaches."""
    wide = 0.5 * (flat + max(side - flat, 0.0) * tilt) + 0.5
    long = 0.5 * flat + 0.5
    # (a stick laid flat bears alike to its ends; a cone of lead bears
    # hardest along its middle, and its band is soft at the edges)
    laid_flat = even > 0.0 and tilt > 0.0
    rounded = RIM if laid_flat else RIM + (1.0 - RIM) * TILT_RIM * tilt
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
    now[13] = 1.0 if laid_flat else 0.0
    now[14] = turning(xs, ys, i)
    return max(wide, long)


@njit(cache=True, error_model="numpy")
def put_down(
    tooth,
    crush,
    polish,
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
    x, y, tx, ty, turn = now[0], now[1], now[2], now[3], now[14]
    count = int(math.ceil(2.0 * now[5] / spacing))
    now[11], now[12], now[14] = x, y, 0.0  # (it goes straight on)
    laid, lying = 0.0, 0.0
    for j in range(1, count + 1):
        now[0], now[1] = x + way * j * spacing * tx, y + way * j * spacing * ty
        dv, rest = bear(
            tooth,
            crush,
            polish,
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
    now[0], now[1], now[11], now[12], now[14] = x, y, math.nan, math.nan, turn
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
    law = np.empty(15)
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
    # the crumbs of its own mark, for each part of it the paper holds, and
    # how far it meets a bed in dust beyond those: as far as it is dry (a
    # medium that is all dust has nothing for the paper to hold either way)
    law[13], law[14] = 0.0, 0.0
    if binding > 1e-9:
        law[13], law[14] = (1.0 - binding) / binding, dryness
    return law, reach


@njit(cache=True, error_model="numpy")
def line_across(near, line, xs, ys, at, tx, ty, wide):
    """Put in ``near`` (see :func:`within_ends`), as its line ``line``, the
    line through the sample ``at`` of the path (``xs``, ``ys``) across the
    way (``tx``, ``ty``), for what lies within ``wide`` of the path."""
    slot = 2 + 5 * line
    near[slot], near[slot + 1] = xs[at], ys[at]
    near[slot + 2], near[slot + 3] = tx, ty
    near[slot + 4] = wide


@njit(cache=True, error_model="numpy")
def band_way(xs, ys, at, leg, end, ax, ay, far):
    """For the line through the sample ``at`` of the path (``xs``, ``ys``)
    across the way it goes from the sample ``leg`` to the next, which says
    where the band of the stroke is on the side of the cut at its end
    ``end`` (the first sample or the last, where the stroke goes the way
    (``ax``, ``ay``)): that way, as a unit vector, and whether the line can
    matter.

    It cannot where the two samples are one point, nor where, for all the
    paper within ``far`` of that end, it leaves nothing that the cut itself
    does not: where its sample is further behind the cut than the path
    turns between the two, across ``far``. In a straight stroke that is
    every one, and in one that does not bend near its end: such a stroke
    is cut by the one line, at the cost of one."""
    tx, ty = xs[leg + 1] - xs[leg], ys[leg + 1] - ys[leg]
    bx, by = xs[at] - xs[end], ys[at] - ys[end]
    going = math.sqrt(tx * tx + ty * ty)
    if going <= 1e-9:
        return 1.0, 0.0, False
    tx, ty = tx / going, ty / going
    # (how far past the cut the paper on the stroke's side of this line can
    # lie, at most: behind the cut, for the end; ahead of it, for the start)
    past = bx * tx + by * ty
    past = (past if end > 0 else -past) + far * math.hypot(tx - ax, ty - ay)
    return tx, ty, past > -1e-6


@njit(cache=True, error_model="numpy")
def ends_near(near, xs, ys, i, spacing, reach, wide, square_start, square_end):
    """Fill ``near`` with what :func:`within_ends` takes for a tip at
    sample ``i`` of a stroke through the points (``xs``, ``ys``),
    ``spacing`` apart, cut square at its start and at its end or not.

    A cut applies only where the tip is within ``reach`` of that end along
    the path, so that a stroke that comes back across the line through one
    of its ends is not cut there. It is the line through the end across the
    way the stroke goes there, and it takes what lies past it, but not the
    band of the stroke itself: what is within ``wide`` of the path at a
    sample within ``reach`` of that end, and on the stroke's side of the
    line across the path there (for the start, after the line across the
    way the path leaves that sample; for the end, before the line across
    the way it comes to it). So a flat end takes off what the tip lays
    beyond its path, and the stroke is what the width of the tip sweeps
    along the path, from one end to the other.

    A straight stroke has none of its band past the cut: of those lines,
    the one across its end is the furthest, and it is cut as by that line
    alone, to the bit; so is one that bends half a tip or more before its
    end, or curves no tighter than its tip. One that bends nearer its end
    (a hook, a curl) has the band it drew before the bend across the line
    of the cut, which took it away with the rest. Lines that cannot matter
    are left out (:func:`band_way`); ``near`` has room for a line at every
    sample for each end."""
    last = xs.shape[0] - 1
    # (no paper under a tip within reach of an end is further from that end:
    # the tip reaches less than ``reach`` from its middle)
    far = 2.0 * reach
    starts, lines = 0, 0
    if square_start > 0.0 and i * spacing <= reach:
        ax, ay = heading(xs, ys, 0)
        line_across(near, 0, xs, ys, 0, ax, ay, math.inf)
        lines = 1
        at = 1
        while at < last and at * spacing <= reach:
            tx, ty, matters = band_way(xs, ys, at, at, 0, ax, ay, far)
            # (written where the next will be, if it does not matter)
            line_across(near, lines, xs, ys, at, tx, ty, wide)
            lines += 1 if matters else 0
            at += 1
        starts = lines
    if square_end > 0.0 and (last - i) * spacing <= reach:
        ax, ay = heading(xs, ys, last)
        line_across(near, lines, xs, ys, last, ax, ay, math.inf)
        lines += 1
        at = last - 1
        while at > 0 and (last - at) * spacing <= reach:
            tx, ty, matters = band_way(xs, ys, at, at - 1, last, ax, ay, far)
            line_across(near, lines, xs, ys, at, tx, ty, wide)
            lines += 1 if matters else 0
            at -= 1
    near[0], near[1] = starts, lines - starts


@njit(cache=True, error_model="numpy")
def lay_stroke(
    tooth,
    crush,
    polish,
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
    applies only within the tip's reach of its end, and takes what lies
    past the stroke, not the band of the stroke itself where a hook brings
    it across the line of the cut (:func:`ends_near`). Its mark is an ellipse,
    as long along the stroke as its flat is wide, and across it as wide as
    the flat, and the share ``tilt`` of what its side adds to that; for a
    stick whose side lies flat, a band as wide, even across and rounded at
    its ends (:func:`side_bearing`). Where it lands and where it lifts, all
    the paper under it gets a full pass (:func:`put_down`); where the
    stroke turns, each part of the tip lays as far as it slides
    (:func:`turning`).

    ``medium`` is [scattering per unit volume, softness, binding, the share
    of wax in it, its sheen, how far it melts into a film of wax, whether
    its side lies flat along its length (1) or is the cone of a sharpened
    lead (0)] (its absorption per band is ``k``) and ``paper`` is
    [depth, bite, give, hold]. The medium is laid where the tip touches the
    tooth.
    What the paper does not hold of it is dust, or paste if the medium is
    greasy (:func:`paste_of`), gathered in ``fresh`` (empty before and after)
    and laid at the end of the pass on top of what lay there (``loose``),
    which goes ``under`` it. With it goes the film of the medium: the part
    of what would have been held that met not the paper but what lies over
    it, which for a waxy medium is any dust (:func:`lying_share`) and for a
    dry one a bed of it, dust beyond the crumbs of its own mark
    (:func:`bed_of`); such a bed takes none of the room in the tooth. What
    is set or a film in the two layers
    (``bonded``, the volume of each kind in each) is not pressed in, and dust
    that is set takes less of the room in the tooth. Wax films over the
    tooth, which then takes less (:func:`resist_of`); pressed hard, the tip
    polishes it flat (``polish``, the wax's) and smears it along, and a hard
    tip pressed hard flattens the tooth itself (``crush``, the paper's).
    Where the paper is indented (``dent``,
    how far below the peaks of the tooth), the tip lays only what it reaches
    down to; it goes down into a groove as far as all it covers is indented
    (:func:`sunk_into`), so a tip narrower than a groove follows it down.

    Returns the volume the tip has lost: what it laid, less what it has
    dragged off the sheet.
    """
    lead, wear, worn, spacing = tip[0], tip[1], tip[2], tip[3]
    # (where it lands and lifts, the tip goes at its own pace, however short
    # the stroke: a stroke of no length is a dot)
    landing = spacing if spacing >= 0.5 * tip[6] else tip[6]
    law, reach = law_of(medium, paper)
    count = xs.shape[0]
    # (the cuts that apply where the tip is now: room for a line across the
    # path at every sample, for each end that is cut square)
    near = np.zeros(2 + 10 * (count if tip[4] > 0.0 or tip[5] > 0.0 else 0))
    streaks = streaks_of(salt)
    row = np.empty((5, tooth.shape[1]))
    reached = 0.0  # the furthest the tip has been from its path
    # what a tip that polishes drags along
    drag = np.zeros(HELD)
    total = 0.0
    lying = 0.0  # what the pass leaves lying on the sheet, to settle
    # where the tip last bore, and how
    now = np.zeros(15)
    even = medium[6]  # (whether its side lies flat along its length)
    tip_at(
        now, shapes[0, 1], shapes[0, 2], shapes[0, 0], reach, 0, xs, ys, pressures, even
    )
    for i in range(count):
        step = steps[i]
        if pressures[i] <= 0.0 or step <= 0.0:
            continue
        flat = shapes[i, 0]
        if wear > 0.0:  # the flat grows with what the tip has laid
            flat = min((flat**3 + wear * max(worn + total, 0.0)) ** (1.0 / 3.0), lead)
        tilt, side = shapes[i, 1], shapes[i, 2]
        bound = tip_at(now, tilt, side, flat, reach, i, xs, ys, pressures, even)
        reached = max(reached, bound)
        # (a square cut applies within the tip's reach of an end, and leaves
        # the band of the stroke, which is as wide as the tip's flat: its
        # mark less the half pixel its rim is softened by)
        near_by, band = bound + spacing, now[4] - 0.5
        ends_near(near, xs, ys, i, spacing, near_by, band, tip[4], tip[5])
        lands = i == 0 or pressures[i - 1] <= 0.0
        lifts = i == count - 1 or pressures[i + 1] <= 0.0
        # (the first and the last point take half a spacing, which is the
        # half of the tip ahead of them, behind them; what lands or lifts
        # there takes the other half too)
        if count > 1 and ((lands and i == 0) or (lifts and i == count - 1)):
            step += 0.5 * landing
        for way in (-1.0, 0.0, 1.0):
            if way == 0.0:
                # (it turns as the stroke turns: where that is sharp, it is
                # laid in parts, turned bit by bit)
                tx, ty, angle = now[2], now[3], now[14] * step
                parts = in_parts(step, angle, now[4], now[9])
                dv, rest = 0.0, 0.0
                for part in range(parts):
                    now[2], now[3] = turned(tx, ty, angle, part, parts)
                    more, left = bear(
                        tooth,
                        crush,
                        polish,
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
                        step / parts,
                        near,
                        streaks,
                        row,
                        salt,
                    )
                    dv += more
                    rest += left
                now[2], now[3] = tx, ty
            elif (way < 0.0 and lands) or (way > 0.0 and lifts):
                dv, rest = put_down(
                    tooth,
                    crush,
                    polish,
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
        # (only wax binds the dust it comes to lie on)
        binds = BINDS * min(medium[3] / STICKS, 1.0)
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
            binds,
            xs,
            ys,
            reached,
        )
    # what the tip still drags it leaves where it lifts off; only off the
    # sheet does it leave the paper with it
    return total - set_down(held, cover, drag, now)


@_inlined
def spread_dust(loose, near, giving, iy, ix, take, back):
    """Take the share ``take`` of the dust on top at a pixel onto the face of
    the tool (``near``, packed as dust is), and lay the share ``back`` of
    what its face is giving in this step (``giving``: see
    :func:`face_gives`) down on the pixel. Returns the volumes laid and
    taken."""
    taken = loose[iy, ix, IV] * take
    given = giving[IV] * back
    # (over no part where nothing is taken and nothing laid: under a tool
    # that does not smear, with nothing on its face)
    for part in range(LOOSE if take > 0.0 or back > 0.0 else 0):
        up = loose[iy, ix, part] * take
        down = giving[part] * back
        near[part] += up - down
        loose[iy, ix, part] += down - up
    return given, taken


@_inlined
def face_gives(giving, near, rate):
    """What the face of a tool gives in one step of what is on it as the
    step begins (``near``): were it to give the share ``rate`` of it per
    step for as long as it had any, the pixels under it get between them
    what that leaves it without after one step, 1 - exp(-rate) of it, each
    by how the tool bears on it and whatever order they come in. ``giving``
    is filled with what a pixel gets for each unit of that share; what the
    face takes up in a step it gives from the next on. Returns how many
    such units there are to give: none, of a face with nothing on it."""
    each = 1.0
    if rate > 1e-9:
        each = (1.0 - math.exp(-rate)) / rate
    some = False
    for part in range(giving.shape[0]):
        giving[part] = max(near[part], 0.0) * each  # (but for rounding)
        some = some or giving[part] > 0.0
    return 1.0 / each if some else 0.0


@njit(cache=True, error_model="numpy")
def face_of(radius):
    """How a round tool of ``radius`` pixels bears on the paper under it,
    summed over its face: 1 for each pixel it bears fully on, less at its
    rim (as :func:`rub_under` works it out for a pixel, at a pressure of 1
    and without its streaks)."""
    bound = radius + 0.5
    rim = RIM * bound + 0.5
    rings = 256
    total = 0.0
    for ring in range(rings):
        inside = (ring + 0.5) * bound / rings  # how far inside its edge
        bears = min(inside / rim, 1.0) * min(inside, 1.0)
        total += 2.0 * math.pi * (bound - inside) * bears * bound / rings
    return total


@_inlined
def slip_of(rubbed, there, sheen, barrier, covered):
    """The share of what the paper holds at a pixel that slides under a tool
    that has ``rubbed`` it (how strongly it smears, times its share of a
    pass): of the volume ``there``, the graphite (``sheen``: other pigments
    do not slide), less the more of what is set or a film (``barrier``:
    :func:`hard_at`) lies over it on the covered share of the pixel."""
    if rubbed <= 0.0 or there <= 0.0 or sheen <= 0.0:
        return 0.0
    graphite = min(sheen / there, 1.0)
    slip = min(SLIDE * rubbed * graphite**SLIDE_POWER, 1.0)
    if barrier > 0.0:
        sealed = barrier / max(covered, 1e-6) / COVERS
        slip *= 1.0 - min(sealed, 1.0)
    return slip


@_inlined
def push_along(
    held, loose, slid, giving, bonded, cover, iy, ix, lifts, share, back, slides
):
    """What a tool pushes along of what the paper holds at a pixel, in the
    ``share`` of a pass at full pressure (``lifts`` as for :func:`lift`),
    onto its face (``slid``, packed as what a tool carries is), which lays
    some of what it gives in this step (``giving``) down again, held.
    Returns the volume laid down.

    A tool that smears slides the graphite (:func:`slip_of`): part of it
    comes loose as dust on top, the rest goes onto its face, which lays the
    share ``slides`` of what it gives (wax refuses it). Other pigments do
    not slide.

    One that does not, an eraser, slides on wax that holds against it, and
    drags some of what the paper holds along, with its wax: WAX_DRAG of
    what it would have lifted, had the wax not held. Its face lays the share
    ``back`` of what it gives, wax and all: wax sticks to paper and to wax
    alike."""
    smears, lift_held, grip, reached = lifts[2] > 0.0, lifts[1], lifts[3], lifts[6]
    there, sheen, waxed = held[iy, ix, IV], held[iy, ix, IX], held[iy, ix, IW]
    barrier, covered = hard_at(bonded, iy, ix), cover[iy, ix]
    slip = slip_of(lifts[2] * share, there, sheen, barrier, covered)
    drag = WAX_DRAG * min(lift_held * share, 1.0) * (1.0 - grip) * reached
    moves = drag > 0.0 and waxed > 0.0
    take, lays = drag, back
    if smears:
        rest = slip * (1.0 - SLIDE_LOOSE) / max(1.0 - slip * SLIDE_LOOSE, 1e-9)
        take, lays, moves = min(rest, 1.0), slides, slip > 0.0
    given = 0.0
    # (else no graphite, or no wax, here or on the tool: most pixels)
    for _ in once(moves or slid[IV] > 0.0):
        if smears:
            for part in range(LOOSE):
                moved = held[iy, ix, part] * slip * SLIDE_LOOSE
                held[iy, ix, part] -= moved
                loose[iy, ix, part] += moved
        elif moves:
            gone = min(there * drag / waxy_at(held, bonded, iy, ix), 1.0)
            wax = waxed * gone
            held[iy, ix, IW] -= wax
            slid[IC] += wax
        given, _ = spread_dust(held, slid, giving, iy, ix, take, lays)
        if not smears:
            wax = giving[IC] * back
            slid[IC] -= wax
            held[iy, ix, IW] += wax
    return given


@_inlined
def layer_at(layer, bonded, iy, ix, which):
    """What one of the two layers is at a pixel (``which``: 0 underneath, 1
    on top): its volume, and the volumes of it that are set, a film and
    paste."""
    return (
        layer[iy, ix, IV],
        bonded[iy, ix, which, SET],
        bonded[iy, ix, which, FILM],
        bonded[iy, ix, which, PASTE],
    )


@_inlined
def bound_in(layer):
    """Whether something in a layer (:func:`layer_at`) is not loose dust."""
    return layer[1] + layer[2] + layer[3] > 0.0


@_inlined
def loose_ground(waxed, below, on_top):
    """How far a film at a pixel is only as fast as the dust it lies on: the
    share of the dust of the two layers (``below`` and ``on_top``:
    :func:`layer_at`) that is loose, where the pixel has
    no wax (``waxed``, the wax there). A film of wax holds, and sets the
    dust under it; the film a dry medium leaves on a bed of dust has
    nothing to hold on to but the dust, and comes away with it: all of it
    on loose dust, none of it on dust that is set. (Worked out only where
    something in the two layers is not loose dust: it is not asked of every
    pixel a tool rubs.)"""
    if waxed > 0.0:
        return 0.0
    film = below[2] + on_top[2]
    if film <= 0.0:
        return 0.0
    dust = below[0] + on_top[0] - film
    dust -= below[3] + on_top[3]
    if dust <= 0.0:
        return 0.0
    fixed = below[1] + on_top[1]
    return min(max(dust - fixed, 0.0) / dust, 1.0)


@_inlined
def bonded_share(layer, up_dust, up_paste, up_held, adrift):
    """The share of a layer at a pixel (:func:`layer_at`) that a tool takes
    when not all of it is loose dust: the shares ``up_dust`` of its loose
    dust, ``up_paste`` of its paste and ``up_held`` of what is set or a
    film in it; but a film that is only as fast as the dust it lies on
    (``adrift``: :func:`loose_ground`) goes that far as the dust does."""
    there, fixed, film, paste = layer
    if there <= 0.0:
        return 0.0
    hard = min((fixed + film) / there, 1.0)
    pasty = min(paste / there, 1.0 - hard)
    taken = up_dust * (1.0 - hard - pasty) + up_paste * pasty + up_held * hard
    more = max(up_dust - up_held, 0.0) * adrift
    return taken + more * min(film / there, 1.0)


@_inlined
def take_bonded(bonded, load, iy, ix, which, layer, up_dust, up_paste, up_held, adrift):
    """Take off ``bonded`` what a tool takes of each kind in a layer at a
    pixel (``layer``: :func:`layer_at`, as it was), for the shares of
    :func:`bonded_share`, and count the paste on the tool (``load``).
    Nothing, of a layer that is all loose dust."""
    there, fixed, film, paste = layer
    more = max(up_dust - up_held, 0.0) * adrift  # (of the film, with the dust)
    for _ in once(there > 0.0 and bound_in(layer)):
        bonded[iy, ix, which, SET] -= fixed * up_held
        bonded[iy, ix, which, FILM] -= film * (up_held + more)
        bonded[iy, ix, which, PASTE] -= paste * up_paste
        load[IP] += paste * up_paste


@_inlined
def waxy_at(held, bonded, iy, ix):
    """The volume at a pixel that its wax is in: what the paper holds, the
    films and the paste. ``held[IW]`` counts the wax of a pixel wherever it
    lies, and it leaves with these, in proportion."""
    waxy = held[iy, ix, IV] + bonded[iy, ix, 0, FILM] + bonded[iy, ix, 1, FILM]
    return waxy + bonded[iy, ix, 0, PASTE] + bonded[iy, ix, 1, PASTE]


@_inlined
def wax_grip(held, cover, polish, iy, ix, scrape):
    """The share of its lift a tool keeps on what a pixel holds, for the
    wax there: all of it where there is none, less the thicker the wax is
    and the further it has been polished (``polish``: the wax's own, not
    what a dry tip crushed of the paper under it), which a tool gets
    through as far as it scrapes (``scrape``, 0..1)."""
    covered, wax, polished = cover[iy, ix], held[iy, ix, IW], polish[iy, ix]
    if covered <= 0.0 or wax <= 0.0:
        return 1.0
    waxy = min(wax / covered / WAX_FAST, 1.0)
    fast = WAX_HOLDS + (POLISH_HOLDS - WAX_HOLDS) * polished
    return 1.0 - (1.0 - scrape) * waxy * fast


@_inlined
def stained(held, cover, iy, ix, scrape):
    """The share of what the paper holds at a pixel that a tool reaches: all
    of it but the stain a waxy medium leaves in the fibres of the paper,
    STAIN thick on the share of the pixel that is covered (as far as what
    it holds is waxy), which a tool reaches only as far as it scrapes
    (``scrape``, 0..1), cutting the surface of the paper."""
    there, wax, covered = held[iy, ix, IV], held[iy, ix, IW], cover[iy, ix]
    if there <= 0.0 or wax <= 0.0:
        return 1.0
    waxy = min(wax / there / STICKS, 1.0)
    stain = STAIN * covered * waxy * (1.0 - scrape)
    return max(there - stain, 0.0) / there


@_inlined
def waxy_share(paper, below, on_top, up_paper, up_held, paste_top, paste_below):
    """The share of the wax of a pixel that goes with what a tool takes
    there: the shares ``up_paper`` of what the paper holds (``paper``, its
    volume), ``up_held`` of the films, ``paste_top`` of the paste in the
    layer on top and ``paste_below`` of the paste underneath (``below``
    and ``on_top``: :func:`layer_at`), of all that the wax is in
    (:func:`waxy_at`)."""
    waxy = paper + below[2] + on_top[2]
    waxy = waxy + below[3] + on_top[3]
    if waxy <= 0.0:
        return 0.0
    gone = paper * up_paper
    gone += (below[2] + on_top[2]) * up_held
    gone += on_top[3] * paste_top
    return min((gone + below[3] * paste_below) / waxy, 1.0)


@_inlined
def room_for(
    paper, waxed, below, on_top, greasy, room, reached, up_held, up_loose, up_under
):
    """The share of what :func:`lift` would take at a pixel with these
    shares (of what the paper holds, of the dust on top and underneath)
    that a tool with room for the volume ``room`` can take: 1, unless
    that is less than it would take. The pixel is as :func:`lift` reads
    it: the volume the paper holds (``paper``, of which the tool reaches
    the share ``reached``), its wax, and the two layers."""
    from_top, from_below = up_loose, up_under
    if bound_in(on_top):
        up_paste = up_held + max(up_loose - up_held, 0.0) * greasy
        adrift = loose_ground(waxed, below, on_top)
        from_top = bonded_share(on_top, up_loose, up_paste, up_held, adrift)
    if bound_in(below):
        up_paste = up_held + max(up_under - up_held, 0.0) * greasy
        adrift = loose_ground(waxed, below, on_top)
        from_below = bonded_share(below, up_under, up_paste, up_held, adrift)
    taken = on_top[0] * from_top + below[0] * from_below
    taken += paper * up_held * reached
    room = max(room, 0.0)
    if taken <= room:
        return 1.0
    return room / taken


@_inlined
def lift(held, under, loose, bonded, load, iy, ix, lifts, share):
    """Lift onto a tool (``load``) what it takes at a pixel in the ``share``
    of a pass at full pressure. ``lifts`` is [how strongly it lifts the
    dust, how strongly what is held, how strongly it smears, the share of
    its lift it keeps on what is held here (:func:`wax_grip`), how far it
    reaches the dust underneath (:func:`reach_under`), the volume it has
    room for, the share of what the paper holds it reaches (:func:`stained`)]:
    it takes less of the dust underneath than of the dust on top;
    what is set or a film only as what the paper holds, but a film without
    wax on loose dust with the dust; and paste as dust
    if it smears, as what is held if it only lifts. The wax of the pixel
    goes with what the paper holds, the films and the paste, as much of it
    as of them (:func:`waxy_at`), and is counted on the tool. It takes no
    more than it has room for: where that is less than it would take, it
    takes that share of each, pigment, paste and wax alike.

    It reads the pixel first, as it is before any of it goes, works out
    what it takes of each part from those numbers, and then takes it.

    Returns how far a film at the pixel is only as fast as the dust it lies
    on (:func:`loose_ground`), where it had to work that out: where
    something in the two layers is not loose dust. Elsewhere, less than
    nothing."""
    lift_loose, lift_held, greasy = lifts[0], lifts[1], min(lifts[2], 1.0)
    grip, reach, room, reached = lifts[3], lifts[4], lifts[5], lifts[6]
    paper, waxed = held[iy, ix, IV], held[iy, ix, IW]
    below = layer_at(under, bonded, iy, ix, 0)
    on_top = layer_at(loose, bonded, iy, ix, 1)
    up_held = min(lift_held * grip * share, 1.0)
    up_loose = min(lift_loose * share, 1.0)
    up_under = min(UNDER_LIFT * lift_loose * share, 1.0) * reach
    if room < math.inf:  # (a tool that holds only so much)
        fits = room_for(
            paper,
            waxed,
            below,
            on_top,
            greasy,
            room,
            reached,
            up_held,
            up_loose,
            up_under,
        )
        up_held, up_loose, up_under = fits * up_held, fits * up_loose, fits * up_under
    up_paper = up_held * reached  # (what the paper holds, but its stain)
    # (paste goes as dust if the tool smears, as what is held if it only lifts)
    paste_top = up_held + max(up_loose - up_held, 0.0) * greasy
    paste_below = up_held + max(up_under - up_held, 0.0) * greasy
    for _ in once(waxed > 0.0):  # (the wax goes with what it is in)
        wax = waxed * waxy_share(
            paper, below, on_top, up_paper, up_held, paste_top, paste_below
        )
        held[iy, ix, IW] -= wax
        load[IC] += wax
    adrift = -1.0  # (how far a film goes with its dust: not worked out yet)
    from_top, from_below = up_loose, up_under
    if bound_in(on_top):  # (else it is all loose dust)
        adrift = loose_ground(waxed, below, on_top)
        from_top = bonded_share(on_top, up_loose, paste_top, up_held, adrift)
    if bound_in(below):
        adrift = loose_ground(waxed, below, on_top)
        from_below = bonded_share(below, up_under, paste_below, up_held, adrift)
    take_bonded(bonded, load, iy, ix, 1, on_top, up_loose, paste_top, up_held, adrift)
    take_bonded(bonded, load, iy, ix, 0, below, up_under, paste_below, up_held, adrift)
    for part in range(LOOSE):
        taken = loose[iy, ix, part] * from_top
        loose[iy, ix, part] -= taken
        load[part] += taken
        taken = under[iy, ix, part] * from_below
        under[iy, ix, part] -= taken
        load[part] += taken
        taken = held[iy, ix, part] * up_paper
        held[iy, ix, part] -= taken
        load[part] += taken
    return adrift


@_inlined
def bare_tooth(crush, polish, held, cover, iy, ix, waxed, cut):
    """Give a pixel back the tooth that a tool has bared there, now that its
    wax has gone from ``waxed`` to what it holds. The polish of the wax
    (``polish``) goes with the wax: polished wax keeps the tooth level as
    far as it is thick enough to fill it, WAX_FAST thick, so the polish
    left is at most what the wax left is thick over that, and with all the
    wax gone, none. What a dry tip crushed of the paper itself (``crush``)
    stays, whatever is taken off it. But where a tool cuts the surface of
    the paper (``cut``, the share of it: a blade or grit), it takes the
    crush with it, and the polish of what lay there."""
    crushed, polished = crush[iy, ix], polish[iy, ix]
    for _ in once(crushed > 0.0 or polished > 0.0):
        if waxed > held[iy, ix, IW]:
            covered = max(cover[iy, ix], 1e-6)
            polished = min(polished, held[iy, ix, IW] / covered / WAX_FAST)
        crush[iy, ix] = crushed * (1.0 - cut)
        polish[iy, ix] = polished * (1.0 - cut)


@_inlined
def scrape_ink(ink, inked, load, iy, ix, cut, room):
    """Cut the share ``cut`` of the surface of the paper at a pixel off
    onto a tool (``load``), with the ink stained into it: that share of the
    ink, which the tool carries as it carries pigment, as far as it has
    room for it (``room``, the volume it can still take: where it is full,
    the ink stays on the paper). (The colour it bronzes is the ink's on the
    paper, and goes with what is taken.) ``ink`` has no rows if no pen has
    drawn on the sheet."""
    for _ in once(cut > 0.0 and ink.shape[0] > 0 and ink[iy, ix, IV] > 0.0):
        if ink[iy, ix, IV] * cut > room:  # (a tool that holds only so much)
            cut = max(room, 0.0) / ink[iy, ix, IV]
        for part in range(LOOSE):
            taken = ink[iy, ix, part] * cut
            ink[iy, ix, part] -= taken
            load[part] += taken
        for part in range(IB, INK):
            ink[iy, ix, part] -= ink[iy, ix, part] * cut
        if ink[iy, ix, IV] <= 0.0:  # (scraped clean)
            inked[iy, ix] = 0.0


@_inlined
def spread_top(held, loose, bonded, near, giving, iy, ix, rate, back, pasted, adrift):
    """Spread what lies on top at a pixel from grain to grain: take the share
    ``rate`` of what can be moved of it onto the face of the tool (``near``),
    and lay the share ``back`` of what its face is giving in this step
    (``giving``) down again; the paste in it stays paste, and is laid by the
    share ``pasted`` (more than the rest where wax refuses the dust). The
    wax of the paste goes with it (:func:`waxy_at`). A film that is only as
    fast as the dust it lies on (``adrift``: :func:`loose_ground`) is
    spread that far with the dust, and is dust where it is laid. Returns
    the volumes laid and taken."""
    take = rate
    if bonded[iy, ix, 1, SET] + bonded[iy, ix, 1, FILM] > 0.0:
        take = rate * free_share(bonded, loose, iy, ix, 1)
        film = bonded[iy, ix, 1, FILM]
        if adrift > 0.0 and film > 0.0:
            take += rate * adrift * min(film / loose[iy, ix, IV], 1.0)
            bonded[iy, ix, 1, FILM] = film - film * rate * adrift
    given, taken = spread_dust(loose, near, giving, iy, ix, take, back)
    for _ in once(bonded[iy, ix, 1, PASTE] > 0.0 or giving[IP] > 0.0):
        up = bonded[iy, ix, 1, PASTE] * rate
        wax = 0.0
        if up > 0.0 and held[iy, ix, IW] > 0.0:
            wax = held[iy, ix, IW] * min(up / waxy_at(held, bonded, iy, ix), 1.0)
        down = giving[IP] * pasted
        near[IP] += up - down
        bonded[iy, ix, 1, PASTE] += down - up
        near[IC] += wax - giving[IC] * pasted
        held[iy, ix, IW] += giving[IC] * pasted - wax
    return given, taken


@_inlined
def reach_under(bonded, cover, iy, ix, waxed):
    """How far a tool reaches the dust and the paste of the layer underneath
    at a pixel: all the way, but through what is set or a film of wax in
    the layer on top, which lies between them and covers it once it is
    ``COVERS`` thick. A tool slides on a film of wax laid over dust, and on
    dust that fixative has set, and does not stir what is under them. A
    film where the pixel has no wax (``waxed``, the wax there) is what a
    dry medium left on a bed of dust: it is not slick, and seals nothing;
    only what is set counts."""
    fixed, film, covered = (
        bonded[iy, ix, 1, SET],
        bonded[iy, ix, 1, FILM],
        cover[iy, ix],
    )
    sealed = fixed + film
    if sealed <= 0.0:
        return 1.0
    if waxed <= 0.0:
        sealed = fixed
    return 1.0 - min(sealed / max(covered, 1e-6) / COVERS, 1.0)


@_inlined
def stir_up(under, loose, bonded, iy, ix, stirred):
    """Stir the share ``stirred`` of what can be moved of the layer
    underneath at a pixel up into the layer on top. Returns the volume in
    the two layers that can be moved: the dust and the paste."""
    moving = stirred
    hard = bonded[iy, ix, 0, SET] + bonded[iy, ix, 0, FILM]
    if hard > 0.0:
        moving = stirred * free_share(bonded, under, iy, ix, 0)
    # (over no part where nothing is stirred: under an eraser)
    for part in range(LOOSE if moving > 0.0 else 0):
        moved = under[iy, ix, part] * moving
        under[iy, ix, part] -= moved
        loose[iy, ix, part] += moved
    if bonded[iy, ix, 0, PASTE] > 0.0:
        moved = bonded[iy, ix, 0, PASTE] * stirred
        bonded[iy, ix, 0, PASTE] -= moved
        bonded[iy, ix, 1, PASTE] += moved
    hard += bonded[iy, ix, 1, SET] + bonded[iy, ix, 1, FILM]
    return max(under[iy, ix, IV] + loose[iy, ix, IV] - hard, 0.0)


@_inlined
def wax_refuses(held, cover, iy, ix):
    """How much less of the dust that a tool brings a pixel takes for the wax
    there: what a dry medium laid by a stick loses to it (:data:`RESIST_DRY`,
    where the wax covers the tooth), on the share of the pixel that is
    covered. Wax is slick to dust whoever brings it."""
    covered, wax = cover[iy, ix], held[iy, ix, IW]
    if covered <= 0.0 or wax <= 0.0:
        return 0.0
    return RESIST_DRY * min(wax / covered / COVERS, 1.0) * covered


@_inlined
def dry_share(packed):
    """The share of what a tool carries (``packed``: its load, or what is on
    its face) that is dust and not paste, which sticks to wax."""
    there, paste = packed[IV], packed[IP]
    if there <= 0.0:
        return 1.0
    return 1.0 - min(paste / there, 1.0)


@_inlined
def hunger(cover, held, under, loose, iy, ix):
    """How bare the tooth at a pixel is, 0..1: 1 where it carries nothing,
    less where pigment lies on it, none where that covers it (``COVERS``
    thick on the share of the pixel that is covered). Bare tooth takes the
    dust off a loaded tool readily; paper whose tooth is covered, none."""
    covered = cover[iy, ix]
    there = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
    if covered <= 0.0:
        return 1.0
    thick = there / covered
    return 1.0 - covered * min(thick / COVERS, 1.0)


@_inlined
def lay_load(load, held, loose, bonded, iy, ix, down, sticks):
    """Lay the share ``down`` of the dust a tool has lifted (``load``) on a
    pixel, as dust all over it, and the share ``sticks`` of its paste, which
    is paste again and sticks, with the wax the tool took. Returns the
    volume laid."""
    laid = load[IV] * down
    # (else an eraser, or a tool on a tooth that is fed)
    for _ in once(laid > 0.0):
        for part in range(LOOSE):
            given = load[part] * down
            load[part] -= given
            loose[iy, ix, part] += given
        if load[IP] > 0.0:  # (its paste is paste again, and sticks)
            given = load[IP] * sticks
            load[IP] -= given
            bonded[iy, ix, 1, PASTE] += given
        if load[IC] > 0.0:  # (and the wax it took comes down with it)
            given = load[IC] * sticks
            load[IC] -= given
            held[iy, ix, IW] += given
    return laid


@njit(cache=True, error_model="numpy")
def rub_under(
    cover,
    crush,
    polish,
    held,
    under,
    loose,
    bonded,
    ink,
    inked,
    carried,
    giving,
    lifts,
    tool,
    now,
    step,
    streaks,
    salt,
):
    """What a round tool does to the paper under it as it slides ``step``
    pixels. ``tool`` is [its radius in pixels, how strongly it lifts the
    dust, and what the paper holds, the share of what it carries that it
    lays back over each of its own widths, how strongly it presses the dust
    in, how strongly it smears, how far it scrapes through wax, how its
    face bears, summed over it (:func:`face_of`), the volume it holds
    when it is full, and the share of what it pushes along that it lays
    back over each of its widths]; ``now`` is [x, y,
    heading (x, y), its pressure, and, for a tool landing or lifting, the x
    and y of where it does so (else NaN): it rubs then only the paper it
    covers there]. ``carried``, ``crush``, ``polish``, ``ink``, ``inked`` and
    ``streaks`` are as for :func:`rub_stroke`, and ``giving`` and ``lifts``
    are room to work in: two rows as long as those of ``carried``, and what
    :func:`lift` takes as ``lifts``.

    To what it has lifted (its load) the tool adds the less the fuller it
    is, and it gives the dust of it to bare tooth only (:func:`hunger`).
    What is on its face it shares out among the pixels under it in each
    step (:func:`face_gives`).

    Lift, push along, lay, spread, press in and stir each go over the parts
    of a pixel by themselves: two fifths of a rub's time, which going over
    them once does not win back (measured: fusain-czc). One pass for all of
    them, by shares worked out from the volumes first, comes out bit for
    bit if it rounds each part to float32 wherever a step stores it, so it
    saves loads and stores and no arithmetic; and it works every step out
    for every part, also the steps that a tool or a pixel skips. It is a
    twentieth to a tenth slower for a finger, no quicker for a stump and a
    third slower for an eraser. Spread, press and stir in one pass, and the
    two loops of :func:`push_along` in one, were a twentieth to a tenth
    quicker compiled for a machine with AVX-512 and a twentieth to a
    seventh slower compiled with ``NUMBA_CPU_NAME=haswell`` alone (which
    keeps the machine's AVX-512, and makes more loops 512 bits wide): for
    the machine, LLVM makes the loops that touch only the sheet (press,
    stir) 512 bits wide, and such a pass only happens to keep them at 256.
    Compiled without AVX-512, a stump rubs in a tenth less time, a finger
    in a twentieth less and an eraser in a twentieth more (fusain-lir; the
    README says how). Time a change compiled both ways
    (``tools/benchmark.py --cpu-against haswell``) before believing a gain
    here."""
    height, width = cover.shape
    x, y, tx, ty, pressure = now[0], now[1], now[2], now[3], now[4]
    at_x, at_y = now[5], now[6]
    put = at_x == at_x  # (not NaN)
    lift_loose, lift_held = tool[1], tool[2]
    press, smear, scrape = tool[4], tool[5], tool[6]
    per_full = 1.0 / tool[8]  # (nothing, for a tool that holds any amount)
    bound = tool[0] + 0.5
    rim = RIM * bound + 0.5
    per_pass = 1.0 / (2.0 * bound - rim)
    # of what it carries, the share the tool lays on a pixel it bears fully
    # on, per pixel moved
    per_pixel = tool[3] / (2.0 * bound * math.pi * bound * bound)
    per_near = SPREAD / (2.0 * bound * math.pi * bound * bound)
    per_slid = tool[9] / (2.0 * bound * math.pi * bound * bound)
    lifts[2] = smear  # (the rest of it is filled for each pixel)
    load = carried[0]
    near = carried[1]  # the dust on the face of the tool, soon laid back
    slid = carried[2]  # held graphite it is pushing along, likewise
    # (what its face gives in this step, and how much of it there is to give)
    gives, glide = giving[0], giving[1]
    room = face_gives(gives, near, per_near * step * pressure * tool[7])
    room_slid = face_gives(glide, slid, per_slid * step * pressure * tool[7])
    dry_face = dry_share(gives)  # (the share of that which is dust)
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
            if put:  # (landing or lifting: only the paper under it there)
                ax, ay = ix + 0.5 - at_x, iy + 0.5 - at_y
                if ax * ax + ay * ay >= bound * bound:
                    continue
            local = pressure * min((bound - d) / rim, 1.0) * edge
            local *= streak_at(dx, dy, tx, ty, streaks, salt)
            share = local * per_pass * step  # of one pass at full pressure
            sticks = min(per_pixel * local * step, 1.0)
            back = min(per_near * local * step, room)
            room -= back
            glides = min(per_slid * local * step, room_slid)
            room_slid -= glides
            # what is on its face, and the graphite it pushes, it lays
            # anywhere but on wax, where dust hardly clings; the dust of its
            # load on bare tooth, and no other; its paste sticks to anything
            takes, clings, slides = 1.0, back, glides
            if held[iy, ix, IW] > 0.0:
                takes = 1.0 - wax_refuses(held, cover, iy, ix)
                clings = back * (1.0 - (1.0 - takes) * dry_face)
                slides = glides * takes
            down = sticks
            if sticks > 0.0:  # (not an eraser, which lays nothing)
                takes *= hunger(cover, held, under, loose, iy, ix)
                down = sticks * (1.0 - (1.0 - takes) * dry_share(load))
            waxed = held[iy, ix, IW]
            reach = reach_under(bonded, cover, iy, ix, waxed)
            # (the fuller it is, the less it takes)
            room_left = max(1.0 - load[IV] * per_full, 0.0)
            lifts[0], lifts[1] = lift_loose * room_left, lift_held * room_left
            lifts[4] = reach
            lifts[3] = wax_grip(held, cover, polish, iy, ix, scrape)
            lifts[5] = tool[8] - load[IV]  # (the room it has left)
            lifts[6] = stained(held, cover, iy, ix, scrape)
            adrift = lift(held, under, loose, bonded, load, iy, ix, lifts, share)
            # (the tooth comes back as the wax that levelled it goes, and
            # what the paper itself was crushed by as far as a blade or grit
            # cuts it, with the ink stained into it)
            cut = scrape * min(lifts[1] * share, 1.0)
            bare_tooth(crush, polish, held, cover, iy, ix, waxed, cut)
            scrape_ink(ink, inked, load, iy, ix, cut, tool[8] - load[IV])
            pushed = push_along(
                held,
                loose,
                slid,
                glide,
                bonded,
                cover,
                iy,
                ix,
                lifts,
                share,
                glides,
                slides,
            )
            # lay, as dust all over the pixel
            before = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
            laid = lay_load(load, held, loose, bonded, iy, ix, down, sticks)
            # spread, from grain to grain
            rate = min(SPREADS * smear * share, 1.0)
            given, taken = spread_top(
                held, loose, bonded, near, gives, iy, ix, rate, clings, back, adrift
            )
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
            stirred = min(smear * share, 1.0) * reach
            dusty = stir_up(under, loose, bonded, iy, ix, stirred) / after
            cover[iy, ix] += (1.0 - cover[iy, ix]) * stirred * dusty


@njit(cache=True, error_model="numpy")
def put_tool(
    cover,
    crush,
    polish,
    held,
    under,
    loose,
    bonded,
    ink,
    inked,
    carried,
    giving,
    lifts,
    tool,
    now,
    streaks,
    salt,
    spacing,
    way,
):
    """Where a tool lands (``way`` -1) or lifts (``way`` 1), at ``now`` (see
    :func:`rub_under`): rub the paper it covers there as it would have been
    rubbed had the tool slid onto it from behind (off it ahead), so that all
    of that gets a full pass, as under the middle of a pass, and not only
    the paper ahead of the middle of the tool (behind it). It goes as it
    goes along a pass, ``spacing`` pixels at a time."""
    x, y, tx, ty = now[0], now[1], now[2], now[3]
    count = int(math.ceil(2.0 * (tool[0] + 0.5) / spacing))
    now[5], now[6] = x, y
    for j in range(1, count + 1):
        now[0], now[1] = x + way * j * spacing * tx, y + way * j * spacing * ty
        rub_under(
            cover,
            crush,
            polish,
            held,
            under,
            loose,
            bonded,
            ink,
            inked,
            carried,
            giving,
            lifts,
            tool,
            now,
            spacing,
            streaks,
            salt,
        )
    now[0], now[1], now[5], now[6] = x, y, math.nan, math.nan


@njit(cache=True, error_model="numpy")
def rub_stroke(
    cover,
    crush,
    polish,
    held,
    under,
    loose,
    bonded,
    ink,
    inked,
    xs,
    ys,
    pressures,
    steps,
    tool,
    carried,
    landing,
    salt,
):
    """Rub one pass of a round tool through the points (``xs``, ``ys``),
    bearing with ``pressures`` (0..1) and moving ``steps`` pixels at each.
    ``tool`` is [its radius in pixels, ``lift_loose``, ``lift_held``,
    ``lay``, ``press``, ``smear``, ``scrape``, ``holds``]. Where it bears
    fully, one
    pass lifts the shares ``lift_loose`` of the dust (less of the dust
    underneath) and ``lift_held`` of the held pigment onto the tool, presses
    the share ``press`` of the dust into the paper, and stirs the share
    ``smear`` of the dust underneath up into the dust on top and spreads it
    over the pixel; and over each of its own widths the tool lays the share
    ``lay`` of what it carries back down, as dust on top, where the tooth is
    bare; paper that carries pigment takes the less of it the more it
    carries, and none once that covers it (:func:`hunger`). The tool lifts
    the less the fuller it is, and no more once it carries ``holds`` for
    each pixel of its face. So a stump fills slowly as it is rubbed in a
    tone, leaves a tone as it is once it is full, whatever colour it
    carries, and tones bare paper. As strongly as it smears, it also
    spreads the dust on top over the pixels around
    (:data:`SPREAD`: this, the dust on its face, is most of what it
    moves, and it goes a fraction of the tool's width), and slides the
    graphite that the paper holds (:func:`push_along`).

    Not all that lies in the two layers is loose dust (``bonded``, the
    volume of each kind in each layer). What is set, by fixative or under a
    film, and a film itself are lifted only as held pigment is, and are
    neither pressed in nor stirred. Paste is moved as dust is by a tool that
    smears, and lifted only as held pigment is by one that does not; it is
    paste again where the tool lays it. Wax holds all that is lifted as held
    pigment is, the more the thicker it is and the further it has been
    polished (``polish``), but for the share
    ``scrape`` of the tool's lift, which cuts through it
    (:func:`wax_grip`). And wax is slick to the dust a tool brings, as it is
    to a dry stick: where it covers the tooth it takes next to none of the
    dust in the tool's load or on its face, nor of the graphite the tool
    pushes (:func:`wax_refuses`); that stays on the tool, which carries it
    on. Paste sticks to it, as to any paper, bare or not. The tool slides
    on a film of wax that lies over dust,
    and on dust that is set: what is loose in the layer underneath it
    reaches only as far as those do not cover it (:func:`reach_under`).
    The film a dry medium leaves on a bed of dust has no wax: it seals
    nothing, and it is lifted and spread with the loose dust it lies on, as
    far as that dust is loose (:func:`loose_ground`).

    As the wax goes, so does its polish, and the tooth it levelled is
    there again, but for what a dry tip crushed of the paper itself
    (``crush``), which no rubber eraser gives back. As far as a tool
    scrapes, it cuts the surface of the paper too (the
    share ``scrape`` of its lift of what the paper holds): the tooth it
    bares is the paper's own again (:func:`bare_tooth`), and the ink of pens
    stained into it (``ink``, packed as the ink of a sheet is, and
    ``inked``; arrays with no rows if no pen has drawn) comes off with it,
    onto the tool (:func:`scrape_ink`). No other tool touches the ink.

    ``carried`` is what the tool carries, three rows packed as dust is with
    the volume of paste and the volume of wax after ([K (N bands), S,
    volume, sheen, paste, wax]): what it has lifted, the dust on its face,
    soon laid back, and the held graphite it is pushing along, likewise.
    The wax of a pixel (``held[IW]``, wherever it lies there) goes with what
    it is in: a tool that takes what the paper holds, a film or paste takes
    as much of the wax, and lays it back where it lays its paste, so the
    smear of an oil pastel is as waxy as the pastel and the place it left
    less so. The tool lands with what it carries, which is updated in
    place, so a tool that is not cleaned between two passes takes no more
    than it gives.

    Where it lands and where it lifts, all the paper under the tool is
    rubbed as under the middle of a pass (:func:`put_tool`), at the pace
    ``landing`` (the spacing of the points, or that of a path long enough
    if they are closer: a pass of no length is a dab).
    """
    streaks = streaks_of(salt)
    count = xs.shape[0]
    gear = np.empty(10)  # (the tool, as rub_under takes it)
    gear[:7] = tool[:7]
    gear[7] = face_of(tool[0])
    gear[8] = tool[7] * math.pi * (tool[0] + 0.5) ** 2
    # (what it pushes along it lays back as it does the dust on its face,
    # if it smears; the wax it drags, if it does not, further on)
    gear[9] = SPREAD if tool[5] > 0.0 else DRAG_LAY
    now = np.empty(7)
    now[5], now[6] = math.nan, math.nan
    # what the face of the tool gives in a step, of its dust and of the
    # graphite it pushes
    giving = np.empty((2, carried.shape[1]))
    taking = np.empty(7)  # (what the tool lifts at a pixel: see lift)
    for i in range(count):
        step = steps[i]
        if pressures[i] <= 0.0 or step <= 0.0:
            continue
        now[0], now[1] = xs[i], ys[i]
        now[2], now[3] = heading(xs, ys, i)
        now[4] = pressures[i]
        lands = i == 0 or pressures[i - 1] <= 0.0
        lifts = i == count - 1 or pressures[i + 1] <= 0.0
        # (the first and the last point take half a spacing, which is the
        # half of the tool ahead of them, behind them; what lands or lifts
        # there takes the other half too)
        if count > 1 and ((lands and i == 0) or (lifts and i == count - 1)):
            step += 0.5 * landing
        for way in (-1.0, 0.0, 1.0):
            if way == 0.0:
                rub_under(
                    cover,
                    crush,
                    polish,
                    held,
                    under,
                    loose,
                    bonded,
                    ink,
                    inked,
                    carried,
                    giving,
                    taking,
                    gear,
                    now,
                    step,
                    streaks,
                    salt,
                )
            elif (way < 0.0 and lands) or (way > 0.0 and lifts):
                put_tool(
                    cover,
                    crush,
                    polish,
                    held,
                    under,
                    loose,
                    bonded,
                    ink,
                    inked,
                    carried,
                    giving,
                    taking,
                    gear,
                    now,
                    streaks,
                    salt,
                    landing,
                    way,
                )


@_inlined
def ball_grip(cover, held, under, loose, iy, ix):
    """The share of its ink a ball lays at a pixel: all of it on bare paper,
    less where it skids on a film of pigment or of wax."""
    covered, wax = cover[iy, ix], held[iy, ix, IW]
    there = held[iy, ix, IV] + under[iy, ix, IV] + loose[iy, ix, IV]
    if covered <= 0.0:
        return 1.0
    thick = there / covered
    skid = min(max((thick - SKID_FROM) / (SKID_FULL - SKID_FROM), 0.0), 1.0)
    waxy = min(wax / covered / COVERS, 1.0)
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
def stain(ink, iy, ix, dv, k, pen):
    """Add ``dv`` of the ink of ``pen`` (see :func:`roll_stroke`; its
    absorption is ``k``) to the pixel."""
    for band in range(k.shape[0]):
        ink[iy, ix, band] += dv * k[band]
    ink[iy, ix, IS] += dv * pen[0]
    ink[iy, ix, IV] += dv
    for part in range(3):
        ink[iy, ix, IB + part] += dv * pen[3 + part]


@_inlined
def blot(cover, held, under, loose, ink, inked, x, y, volume, radius, full, k, pen):
    """Leave a drop of ink of this ``volume`` at (``x``, ``y``): a round dot
    of this ``radius``, thickest at its middle. It lies in the valleys of
    the tooth as on its peaks, ``full`` thick at least: what of it a pixel
    gets is on as much of the pixel as it would cover so (all of it, if
    there is that much), wherever the line's ink is, so it falls on the
    pixel's bare paper in proportion. A drop inks as much more of a pixel
    as there is of it, and a tiny one next to nothing. It takes on a film
    of pigment or of wax as badly as the line does
    (:func:`ball_grip`). ``pen`` is as for :func:`roll_stroke`. Returns the
    volume laid: less than the drop where it skids or lies off the sheet."""
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
            stain(ink, iy, ix, dv, k, pen)
            # (the drop's share of the pixel, of which the share that is
            # bare is newly inked)
            inked[iy, ix] += (1.0 - inked[iy, ix]) * min(dv / full, 1.0)
            laid += dv
    return laid


@_inlined
def turn_of(tx, ty, wx, wy):
    """The angle, in radians (0 to pi), between two unit headings."""
    return math.atan2(abs(tx * wy - ty * wx), tx * wx + ty * wy)


@njit(cache=True, error_model="numpy")
def roll_over(
    tooth,
    crush,
    polish,
    dent,
    cover,
    held,
    under,
    loose,
    ink,
    inked,
    k,
    pen,
    ball,
    salt,
):
    """Lay the ink the ball leaves on the paper under it, at ``ball`` [x, y,
    how far from its centre it reaches, the level in the tooth its middle
    reaches down to, how far its surface rises from its middle to its rim,
    how deep it presses a groove under its middle, the ink it lays where it
    touches fully, and, for a ball landing or lifting, the x and y of where
    it does so (else NaN): it inks then only the paper it meets there (where
    its surface comes down to the peaks of the tooth: all of it pressed
    firmly, a narrower disc pressed lightly, as the line is narrower), and
    presses no groove]. ``pen`` is as for :func:`roll_stroke`. Returns the
    volume laid."""
    height, width = tooth.shape
    x, y, bound, low, rise = ball[0], ball[1], ball[2], ball[3], ball[4]
    sink, rate, at_x, at_y = ball[5], ball[6], ball[7], ball[8]
    put = at_x == at_x  # (landing or lifting)
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
            if put:  # (only what the ball meets where it is put)
                ax, ay = ix + 0.5 - at_x, iy + 0.5 - at_y
                there = (ax * ax + ay * ay) / (bound * bound)
                if there >= 1.0 or low + rise * there >= 1.0 + MICRO:
                    continue
            else:
                dent[iy, ix] = max(dent[iy, ix], sink * (1.0 - across))
            level = low + rise * across
            level += BALL_JITTER * JITTER * (unit_noise(ix, iy, salt) - 0.5)
            crushed, polished = crush[iy, ix], polish[iy, ix]
            flat = crushed + polished - crushed * polished
            ground = tooth[iy, ix] + (1.0 - tooth[iy, ix]) * flat
            above = ground + MICRO - level
            touch = min(max(above / (2.0 * MICRO), 0.0), 1.0) * edge
            if touch <= 0.0:
                continue
            dv = rate * touch * ball_grip(cover, held, under, loose, iy, ix)
            if dv <= 0.0:
                continue  # (it skids: no ink, and no more paper inked)
            inked[iy, ix] = max(inked[iy, ix], touch)
            stain(ink, iy, ix, dv, k, pen)
            rolled += dv
    return rolled


@njit(cache=True, error_model="numpy")
def roll_stroke(
    tooth,
    crush,
    polish,
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
    landing,
    radius,
    k,
    pen,
    paper,
    rim,
    salt,
):
    """Roll one pass of a ballpoint pen, its ball of ``radius`` pixels,
    through the points (``xs``, ``ys``), bearing with ``pressures`` (0..1)
    and moving ``steps`` pixels at each. Where it lands and where it lifts,
    all the paper it meets there gets a full pass, as under the middle of
    the line: what it would have had if the ball had rolled onto it from
    behind and off it ahead, ``landing`` pixels at a time (see
    :func:`put_down` and :func:`roll_over`).

    ``pen`` is [the scattering of its ink per unit volume, how freely it
    flows, how much it blots, the colour its ink bronzes (linear RGB, its
    brightest part 1)] (the ink's absorption per band is ``k``) and
    ``paper`` is [depth, bite, give, hold]. The ball lays ink at a steady
    rate wherever it touches the paper: how much does not depend on the
    tooth, as it does for a stick, and only a little on the pressure, which
    sets how much of the tooth the ball reaches. The ink is a stain in the
    paper (``ink``, packed as :data:`IB` says, on the share ``inked`` of
    each pixel), under whatever the paper carries; where that is thick or
    waxy, the ball skids and lays less.

    The ball presses a groove into the paper (``dent``), the deeper the
    harder it bears, and goes down into one that is there: it is as small as
    what made it. And it gathers ink on its rim as it rolls, which it leaves
    as a drop (:func:`blot`) where it lands and where the line turns, as
    much of it as it bears on the paper there, on a dot as much narrower as
    the paper it meets there is. ``rim`` holds the ink on the rim of the
    ball: what it lands with, of which a share drawn at random comes round
    where it lands (NaN for a pen just picked up, which lands with a drop
    drawn so and leaves it there), updated in place to what it lifts with.
    The rim holds at most the drop BALL_BLOB says: what the ball would
    gather beyond that is not gathered, and goes nowhere (it is neither on
    the paper nor on the rim). So the ink laid in drops and the ink on the
    rim are BALL_GATHER times (and ``blots`` times) the ink of the lines
    while the rim is below its most, and less once it has held its most.

    Returns the volume of ink laid, drops included: what the pen has lost,
    but for what it has gathered on its rim.
    """
    flow, blots = pen[1], pen[2]
    depth, give = paper[0], paper[2]
    yields = GIVE_NONE + GIVE_FULL * give
    reach = BALL_REACH * yields
    reach = reach / depth if depth > 1e-6 else 1e9
    rise = BALL_ROUND / depth if depth > 1e-6 else 0.0
    bound = radius + 0.5
    per_pass = 1.0 / (2.0 * bound - 1.0)
    # the drop of ink on the rim of the ball: what it holds at most, what
    # the pen lands with, and the share of that which comes round where it
    # lands (a pen just picked up lands with that much, all of which does)
    most = BALL_BLOB * BALL_INK * flow * blots * bound * bound
    seldom = unit_noise(0, 2, salt + 3) ** BALL_SELDOM
    gathered, coming = rim[0], BALL_START * seldom
    if gathered != gathered:
        gathered, coming = most * BALL_START * seldom, 1.0
    least = BALL_SHOWS * BALL_INK * flow * bound * bound  # (a drop that shows)
    # where the ball is, and how it bears (see :func:`roll_over`)
    ball = np.empty(9)
    ball[2], ball[4] = bound, rise
    put = int(math.ceil(2.0 * bound / landing))  # (passes where it is put)
    count = xs.shape[0]
    wx, wy = 1.0, 0.0  # the way it was last going
    down = False  # whether it was on the paper at the last point
    turned = 0.0  # how far the line has turned since the ball left a drop
    travel = 0.0
    total = 0.0
    for i in range(count):
        pressure = pressures[i]
        step = steps[i]
        travel += step
        if pressure <= 0.0 or step <= 0.0:
            down = False
            continue
        x, y = xs[i], ys[i]
        tx, ty = heading(xs, ys, i)
        # (the share of a firm pass it lays here)
        firm = BALL_LIGHT + (1.0 - BALL_LIGHT) * pressure
        lands = i == 0 or pressures[i - 1] <= 0.0
        lifts = i == count - 1 or pressures[i + 1] <= 0.0
        # where the line turns, as much of what it gathered comes round as
        # the turn brings round
        if down:
            turned += turn_of(tx, ty, wx, wy)
            comes = gathered * min(turned / BALL_TURN, 1.0)
        else:
            comes, turned = gathered * coming, 0.0
        wx, wy, down = tx, ty, True
        low = 1.0 - reach * pressure**BALL_POWER
        if comes >= least:
            # it leaves as much of it as it bears on the paper, on a dot as
            # much wider than the paper the ball meets as BLOB_WIDE says
            # (``meets`` is the square of the share of its radius that comes
            # down to the peaks of the tooth, as where it is put: see
            # :func:`roll_over`)
            drop = comes * firm
            meets = min((1.0 + MICRO - low) / rise, 1.0) if rise > 0.0 else 1.0
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
                BLOB_WIDE * bound * math.sqrt(meets),
                BALL_INK * flow,
                k,
                pen,
            )
            gathered = max(gathered - left, 0.0)
            total += left
            turned = 0.0
        # (the first and the last point take half a spacing, the half of
        # the ball ahead of them, behind them; where it lands or lifts there
        # they take the other half too)
        if count > 1 and ((lands and i == 0) or (lifts and i == count - 1)):
            step += 0.5 * landing
        rate = BALL_INK * flow * firm
        surge = surge_at(travel, salt) * per_pass
        ball[0], ball[1], ball[3] = x, y, low
        ball[5], ball[6] = BALL_DENT * yields * pressure, rate * (surge * step)
        ball[7], ball[8] = math.nan, math.nan
        rolled = roll_over(
            tooth,
            crush,
            polish,
            dent,
            cover,
            held,
            under,
            loose,
            ink,
            inked,
            k,
            pen,
            ball,
            salt,
        )
        # where it lands, the paper under it as if it had rolled on from
        # behind; where it lifts, as if it had rolled off ahead
        ball[6], ball[7], ball[8] = rate * (surge * landing), x, y
        for way in (-1.0, 1.0):
            if (way < 0.0 and not lands) or (way > 0.0 and not lifts):
                continue
            for j in range(1, put + 1):
                ball[0], ball[1] = (
                    x + way * j * landing * tx,
                    y + way * j * landing * ty,
                )
                rolled += roll_over(
                    tooth,
                    crush,
                    polish,
                    dent,
                    cover,
                    held,
                    under,
                    loose,
                    ink,
                    inked,
                    k,
                    pen,
                    ball,
                    salt,
                )
        total += rolled
        gathered = min(gathered + BALL_GATHER * blots * rolled, most)
    rim[0] = gathered
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


@njit(cache=True, error_model="numpy")
def thicken(layer, marked, share, kappa, sigma):
    """What a layer is to the light where it lies: for each marked pixel of
    its rows (``marked``, taken row after row), its absorption per band
    (a row of ``kappa``) and its scattering (``sigma``), over the share of
    the pixel that is covered (``share``, for each marked pixel): the
    pigment lies on that share, that much thicker."""
    at = 0
    for iy in range(marked.shape[0]):
        for ix in range(marked.shape[1]):
            if marked[iy, ix]:
                covered = share[at]
                for band in range(kappa.shape[1]):
                    kappa[at, band] = layer[iy, ix, band] / covered
                sigma[at] = layer[iy, ix, IS] / covered
                at += 1


@njit(cache=True, error_model="numpy")
def show_through(out, marked, share, seen):
    """Put what is seen where pigment lies (``seen``, a spectrum for each
    marked pixel, taken row after row) into the picture of these rows
    (``out``, which holds the paper, inked or not): on the covered share of
    each marked pixel (``share``), and what was there on the rest of it."""
    at = 0
    for iy in range(marked.shape[0]):
        for ix in range(marked.shape[1]):
            if marked[iy, ix]:
                covered = share[at]
                bare = 1.0 - covered
                for band in range(seen.shape[1]):
                    lit = covered * seen[at, band]
                    out[iy, ix, band] = lit + bare * out[iy, ix, band]
                at += 1


@_inlined
def film_graphite(volume, sheen, film, share):
    """The graphite of a dry film in a layer at a pixel, as its sheen: of
    the graphite in the layer (``sheen``), the share ``share`` that the
    paper holds of a pencil's mark, and never more than the film
    (``film``). And what lies with it in the layer (``volume``): the clay
    of its crumbs, the rest of the mark, and all that is matte and not the
    film's, its crumbs and any other dust."""
    graphite = min(share * max(sheen, 0.0), film)
    crumbs = (film - graphite) * (1.0 - share) / share
    beside = max(volume - film - (sheen - graphite), 0.0)
    return graphite, min(crumbs, beside), beside


@njit(cache=True, error_model="numpy")
def film_sheen(held, under, loose, bonded, cover, press, shine, how):
    """Add the sheen of a film of graphite on a bed of dust to that of what
    the paper holds (``shine``, (H, W), changed in place), at each pixel
    with a film and no wax. ``how`` holds, as for what the paper holds,
    how thick a film is that shines fully, how much thinner it may be on a
    tooth pressed flat, and how thick the matte that hides all but 1/e of
    it; and the share of a pencil's mark that is the film
    (:func:`film_graphite`). The film in the layer on top shines through
    all the matte in its layer: the dust a pass leaves on top with it
    cannot be told from dust laid over it later. The one underneath, which
    a later pass buried, lies on the older dust it was buried in, and
    shines through the clay of its crumbs and the matte of the layer on
    top. All of it together shines over no more than the covered share of
    the pixel. Pixels without a film are left as they are."""
    full, flat, hide, share = how[0], how[1], how[2], how[3]
    for iy in range(shine.shape[0]):
        for ix in range(shine.shape[1]):
            below, above = bonded[iy, ix, 0, FILM], bonded[iy, ix, 1, FILM]
            waxed = held[iy, ix, IW]
            if (below <= 0.0 and above <= 0.0) or waxed > 0.0:
                continue  # (no film here, or one taken for wax: most pixels)
            covered = cover[iy, ix]
            thick = max(covered, 1e-6)
            whole = full * (1.0 - flat * press[iy, ix]) * thick
            on_top, _, beside = film_graphite(
                loose[iy, ix, IV], loose[iy, ix, IX], above, share
            )
            sheen = 0.0
            if on_top > 0.0:
                sheen += min(on_top / whole, 1.0) * math.exp(-beside / thick / hide)
            over = max(loose[iy, ix, IV] - loose[iy, ix, IX], 0.0)
            beneath, crumbs, _ = film_graphite(
                under[iy, ix, IV], under[iy, ix, IX], below, share
            )
            if beneath > 0.0:
                seen = math.exp(-(over + crumbs) / thick / hide)
                sheen += min(beneath / whole, 1.0) * seen
            shine[iy, ix] = min(shine[iy, ix] + covered * sheen, covered)


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
