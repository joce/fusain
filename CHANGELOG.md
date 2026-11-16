# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `tools/benchmark.py` times processor time, not the wall clock, and shows
  the best of N runs (`-n`, 7 by default) with how far they spread. With
  `--against CHECKOUT` it runs another checkout of fusain alongside this
  one, each in a process of its own, takes each workload turn about with it,
  run by run, and shows the ratio of the best times, so that a busy machine
  falls on both alike; `--cpu` and `--cpu-against` set `NUMBA_CPU_NAME` for
  both and for the other alone (`--against . --cpu-against haswell`
  compares one checkout compiled two ways), and `--save FILE` writes the
  times for a later `--against FILE`. A new workload draws 2000 short
  strokes with flat ends (fusain-5sj).

- A mark can be given a seed of its own (fusain-ny8): `Sheet.stroke`,
  `shade`, `smudge`, `erase` and `rub` take an optional keyword-only
  `seed`, an integer from 0 to 2**31 - 1 (anything else, a `bool` included,
  raises ValueError before anything is drawn). A seeded mark takes all its
  random numbers from it and none from the sheet's generator: a stroke, a
  smudge or an erase gives its kernel the seed itself in place of the
  integer it would have drawn, and a `shade` or a `rub`, which also draw
  the places of their passes, the unsteadiness of the hand and the seeds of
  their strokes, draw them from `np.random.default_rng(seed)`. Its grain
  does not depend on what was drawn before it, whatever the sheet's
  generator has been through, and what is drawn after it does not depend on
  it, so that a drawing in which one part changes keeps the grain of every
  mark that does not. Without a seed everything is as it was, to the bit
  (every hash of `tools/fingerprint.py`, the thorough one too).

### Changed

- A stub of a shading wanders less (fusain-dpx): a stroke that a corner of
  the area cuts short is kept if it is half a tip long, and each of its
  knots wandered by a tenth of the tip's width however short it was, so
  that a stub 3 to 5 pixels long of a tip 8 wide was bent through up to a
  right angle (180 degrees at the worst) and ran across the hatching. No
  stroke now wanders by more than `hatching.WANDER_SHARE` (a twentieth) of
  its own length, so that a stub keeps within about 35 degrees of the
  hatching's direction; a stroke two tips long or more is as it was, and
  the generator is drawn from as often as before, so that what is drawn
  after a shading is not moved. The hashes of the fingerprint suite and
  the pictures of nine examples change (the drawings with stubs).

- The sheet keeps the paper's crush and the wax's polish apart
  (fusain-dq2): `Sheet.crush`, how far a firm dry tip has crushed the
  paper's own tooth, and `Sheet.polish`, how far the wax that lies there
  has been polished level. `Sheet.press` is no longer an array of the
  sheet but what a tip meets of the two, worked out when it is read
  (`crush + polish - crush * polish`: polished wax levels its share of the
  relief that the crush left); it is an array that cannot be written to
  (ValueError), so code that set `sheet.press` sets `sheet.crush` or
  `sheet.polish`. A sheet takes
  four bytes a pixel more (380 for 376). There was one `press` for both,
  and where a pixel held wax it was taken for the wax's polish. Measured
  on cartridge paper (`tests/test_burnish.py`):
  - A hard lead pressed under coloured pencil, the pencil then erased:
    three firm layers of 4H crushed the tooth by 0.908; three layers of a
    dark coloured pencil over them, rubbed out four times with a vinyl
    eraser, left 0.048 of it, and a light layer of HB shaded there
    touched 0.69 of the pixels. The crush (0.918 now) stays, to the bit,
    and the HB touches 0.94, as it does on the crushed tooth alone (fresh
    paper 0.68).
  - Wax laid lightly over a crushed tooth is not polished: six light
    layers of crayon (0.35) over those three of 4H had 0.64 of the gloss
    of polished wax and held against a vinyl eraser as a burnished layer
    does (a rub left 0.69 of them); they have no gloss now, and a rub
    leaves 0.26, as on fresh paper (0.30). The gloss of wax and how it
    holds against an eraser read the polish; the tooth a tip meets, how
    little it bites a dry medium, the relief and the sheen of graphite
    read both.
  - The polish of thin wax goes with the wax: a blender pressed hard over
    a very light layer polishes further than so little wax holds up, and
    after six rubs of a vinyl eraser a fifth of the pixels kept a
    hundredth or more of polish beyond what the wax left on them holds
    (its thickness over `WAX_FAST`); now none keeps 0.002. Where a tool
    takes wax, the polish left is at most what the wax left holds up.
  - A dry tip pressed firmly on wax raises both, a waxy one the polish
    only; a rubber eraser gives back the polish as it takes the wax and
    never the crush; a sand eraser or a knife takes both with the surface
    it cuts; fixative leaves both.

  The stroke, rub and pen kernels take the two arrays where they took one.
  On its own this leaves the five drawings of `tools/fingerprint.py`
  without wax as they were in all but the names of the two arrays, and
  changes the eight that have a dry tip over wax, wax over a crushed tooth
  or an eraser on polished wax; with the levels of the next entry, all
  thirteen change.
- A tip presses the tooth flat as far as its pressure can, and no further
  for being pressed again (fusain-2z5). The crush and the polish were each
  the sum of what every pass added, up to 1. Where the strokes of a
  shading overlap, a pixel has two or three passes for its neighbour's
  one, so a shading was polished in streaks; and any pressure above 0.4
  flattened the tooth in the end, given passes enough: six firm layers of
  HB crushed it by 0.99, as a 4H does, and twelve layers of a coloured
  pencil at 0.6 polished it by 0.82, as one firm layer does.

  Each now goes to a level set by the load (`fusain._engine.settled`). A
  tip that bears with firmness `f` (nothing up to a pressure of 0.4, 1 at
  full pressure, as before) polishes a film to `min(BURNISH * f, 1)` of
  how slick the film is, and crushes the paper to `min(crush * f, 1)`,
  `crush` being what its hardness makes of `FLATTEN`: the numbers that a
  pass's worth of sliding added before are where the pressure now stops. A
  slick film is polished flat from 0.87 on, to 0.42 at 0.7, 0.19 at 0.6
  and 0.05 at 0.5; at full pressure a 4H crushes the tooth flat, an HB to
  0.38, a 2B to 0.18, a 6B to 0.014. A pass closes the gap to its level
  `SETTLE` (2) times as fast as it added to the press before, so a pass
  over a tooth left alone ends about where it ended, a second closes most
  of what is left and a third changes next to nothing; a firmer pass over
  a lighter one raises the press to its own level, and a lighter one over
  a firmer one leaves it. And what presses flat is the tip, not the grain
  of its lead: the level goes by the tip's bearing (its pressure, less
  towards its rim) without the streaks that make the grain of a line.
  With them, each stroke left bands polished from a fifth to all the way,
  and going over it again went on polishing under the hardest bands of
  each pass. What a firm tip smears and lays is as it was.

  Measured on cartridge paper (`tests/test_burnish.py`, `test_surface.py`),
  before and now:
  - One pass (a tip 6 wide along each of eight lines, the press under its
    middle): the blender on six light layers of red 0.40 and 0.29 at 0.7,
    0.60 and 0.55 at 0.8, 0.88 and 0.92 at full pressure; an HB on bare
    paper 0.07 and 0.07, 0.13 and 0.13, 0.22 and 0.29; a 4H 0.17 and 0.16,
    0.31 and 0.35, 0.56 and 0.72. From 0.7 up a pass does what it did to
    within a third either way: a dry one crushes a third more at full
    pressure, where the weak bands of a lead crushed less than the tip
    does. Nearer 0.4 it does less (the blender two fifths less at 0.6, and
    a third of what it did at 0.5), since the streaks took the hardest
    bands of a lead past 0.4 where the tip's pressure is not yet there.
  - Going over it again: the blender at 0.7, one to three passes and six,
    0.40, 0.66, 0.87 and 0.96; now 0.29, 0.37, 0.40 and 0.41 (the level of
    0.7 is 0.42). An HB at 0.9: 0.17, 0.37, 0.55 and 0.97; now 0.21, 0.26,
    0.28 and 0.29. The blender at 0.7 over the blender at full pressure
    took 0.913 to 0.957, and leaves 0.928 at 0.929.
  - Layers: three and six firm layers crushed the tooth by 0.88 and 0.94
    under a 4H, 0.74 and 0.99 under an HB, 0.37 and 0.74 under a 2B, 0.03
    and 0.06 under a 6B; now 0.89 and 0.95, 0.34 and 0.37, 0.16 and 0.18,
    0.013 and 0.014. Twelve layers of a coloured pencil at 0.5, 0.6 and
    0.7 polished the wax by 0.29, 0.82 and 0.97, with 0.09, 0.54 and 0.73
    of the gloss of polished wax; now 0.05, 0.20 and 0.49, with 0.002,
    0.03 and 0.18. Coloured pencil is built in light layers, which fill
    the tooth with wax and keep it, and burnished at the end by pressing
    hard: one firm layer of the blender over those twelve polishes them by
    0.85 to 0.91, with 0.55 to 0.64 of the gloss.
  - How evenly one layer is polished (the spread of the polish smoothed
    over 3 pixels, as a share of its mean), at 0.7 and at full pressure.
    Under ruled strokes, all alike (`hand=0`): coloured pencil 0.35 and
    0.21, now 0.13 and 0.08; oil pastel 0.42 and 0.24, now 0.12 and 0.06;
    the blender over three light layers 0.28 and 0.07, now 0.06 and 0.01.
    With the hand as it is: coloured pencil 0.44 and 0.28, now 0.35 and
    0.20; wax crayon 0.60 and 0.39, now 0.49 and 0.29; oil pastel 0.65 and
    0.44, now 0.63 and 0.38; the blender 0.45 and 0.19, now 0.40 and 0.16.
    Two crossed layers at 0.6 (the measure of fusain-bre): coloured pencil
    0.58, now 0.41; oil pastel 0.73, now 0.57. The law no longer makes a
    shading uneven; the hand still does, and that is most of what there
    was. Its strokes are firm by 15% either way, which near where a tip
    begins to polish is the difference between a little and a lot (at 0.7
    their levels run from 0.18 to 0.77, a spread of 0.39 of their mean),
    they are lighter at their ends, and they stray, so that some paper is
    under the rim of a tip and never under its flat: an oil pastel at 0.7
    with every stroke as firm, 0.47; as firm at its ends too, 0.43; and in
    its place, 0.13. More layers even it out, each place keeping the level
    of the firmest pass over it: twelve layers at 0.7 are uneven by 0.10.

  What rested on the sum, and has moved:
  - A firm HB or 2B no longer flattens the tooth as a hard lead does. Three
    firm layers of HB shine 1.25 times as much as on a tooth left alone
    (1.4), six shine three fifths as unevenly (a third), and a light layer
    of 2B over them touches 0.72 of the pixels (0.90; fresh paper 0.73).
    Under a 4H all of it is as it was: 2.4 times the shine, and the 2B
    touches 0.92.
  - A film is polished no further than it is slick. Two layers of a
    coloured pencil at full pressure, whose film is thin yet, are polished
    by 0.55 with 0.18 of the gloss (0.77 and 0.37, their passes adding up
    past how slick the film was); of a crayon 0.83 and 0.72 (0.87 and
    0.81), of an oil pastel 0.85 and 0.75 (0.90 and 0.85).
  - Firm layers that are not burnished hold less fast against an eraser.
    Four layers of a dark coloured pencil at 0.8 polish the wax by 0.50
    (0.84), and a rub of a vinyl eraser leaves 0.47 of them and three rubs
    0.05, at 0.46 of the paper's tone (0.63 and 0.27, at 0.10: they held
    nearly as burnished ones). Burnished with the blender they are as they
    were: 0.68 and 0.32 (0.69 and 0.33), as dark as before, and the eraser
    takes 0.40 of what it takes of the firm layers alone (0.75).
  - The blender does what it did: one firm layer over three light ones of
    red leaves 0.16 of the white flecks (0.19) with 0.45 of the gloss
    (0.48), two leave 0.03 (0.04) with 0.76 (0.79); at 0.7 it closes two
    fifths of the flecks and at 0.5 a twentieth, as before. A white pencil
    leaves 0.19 of the flecks (0.23) with 0.24 of the gloss (0.29), and
    polishes 0.88 as far as the blender (0.94): it polishes a thin film
    as far as it melts into it. A burnished dark under the default lamp
    is at L* 9.8 (10.2; matte 6.6), and 17.9 with the lamp overhead
    (18.6). Burnished wax scraped by a knife or a sand eraser, and what
    is drawn there after, are as they were.
  - An HB at full pressure leaves 0.029 of the pixels white, where it left
    0.044: a pass crushes the tooth as far as its load does while it is
    over it, and the strokes of the same layer that overlap it touch more
    of a tooth already crushed.

  A stroke carries one more number for each pixel of its row (the bearing
  without the streaks), and takes 4 to 8% longer than before the two
  entries (processor time on a sheet of 1120 by 820, before and after turn
  and turn about: shading with an HB 1.73 s and 1.81 s, with a firm
  coloured pencil 2.11 s and 2.22 s): 2 to 3% for the second array, and
  as much for the level. Rubs and renders take what they took. Every
  drawing where a tip bears harder than 0.4
  changes, and so do all thirteen of `tools/fingerprint.py`, recorded
  again; one whose strokes stay under 0.35 or so is as it was but for a
  trace (the streaks no longer take a light stroke past 0.4: three layers
  of HB at 0.4 crushed the tooth by 0.004, and now by 0.0005).

### Fixed

- A partly polished tooth no longer glints pixel by pixel under the default
  lamp (fusain-419). In two layers of a dark crayon at 0.7 the peaks of the
  tooth were polished all the way (11% of the pixels past 0.8), and a peak
  whose facet leaned towards the lamp lifted the dark by up to 10.5 L*:
  1.4% of the pixels gained over 6. It is the rework of press that clears
  it (fusain-2z5): a tip polishes only as far as its pressure can, here
  0.66 at most, and the brightest pixel now adds 3.6 L*, the top
  thousandth 2.7, and none over 6. A lamp with a width, which was built to
  spread the gloss, took 0.5 L* off the brightest before that rework and
  0.08 after it, and is not in: the lamp stays a point.

- A tool rubbed over an area stays inside it (fusain-soe). `Sheet.rub` laid
  its passes with the hand of a shading: the ends of a pass strayed half the
  tool's width either way, past the edge as often as short of it, the
  outermost passes lay a quarter of a width in, so that the tool reached a
  quarter of its width past the sides, and they strayed and wandered from
  there. Since a loaded tool gives its dust to bare tooth (fusain-9zh), it
  dirtied the paper round every tone it rubbed: the clean finger that rubs
  the charcoal in the third row of `examples/crayons.png`, which loads as it
  rubs, darkened the bare paper beside it by up to 105 levels of 255 (its
  luminance from 0.87 to 0.25) on 2132 pixels, as far as 15 pixels past the
  ends of the passes and 7 past their sides; a stump or a finger over a
  shaded patch, as far as 7 and 14 pixels past the ends and 5 and 10 past
  the sides (a luminance down by up to 0.51 and 0.59), and an eraser took
  the tone the shading had laid past the edge. The hand that rubs now works
  the tool up to the edge and turns there (`hatching.passes`): the middle
  of the tool keeps half its width inside the outline, every pass runs from
  that line to that line, and the outermost passes run along it; in the
  middle of the area the passes stray and wander as before, but where the
  tool meets the edge they stray only as a point of a pass wanders (a tenth
  of a width, where a pencil's ends stray by half), and only inwards; the
  points of a pass and the curve through them are kept inside, of a polygon
  too and at any angle; and the tool bears as hard where it turns as in the
  middle, where it lightened as a pencil's stroke does where it lands and
  lifts. The same rubs leave the paper outside the area as it was: to the
  bit with passes square to its edges; at another angle, but for a few
  pixels changed by less than a thousandth of a firm layer; along a
  slanted edge, but for the pixels the edge cuts through. The ends of
  the passes are rubbed as their middles are: an eraser pressed lightly takes
  0.67, 0.94 and 1.00 as much as in the middle of the area at the last
  three pixels before their ends (was 0.45, 0.54 and 0.60), a stump smooths
  the grain there 0.47, 0.80 and 0.98 as much (was 0.47, 0.58, 0.76), a
  finger 0.32, 0.52 and 0.72 (was 0.43, 0.45, 0.52). Along the sides of the
  outermost passes only the rounded rim of the tool reaches the last pixels:
  the outermost one is rubbed 0.11 (eraser), 0.32 (stump) and 0.13 (finger)
  as much as the middle, where the old hand, reaching past the edge, rubbed
  it 0.71, 0.86 and 0.90, and the area is rubbed as its middle from about a
  third of the tool's width in; a second rub across the first works them up
  to the edge (a stump rubbed across then along: 0.55 and 0.66 at the
  outermost pixel, 0.86 and 0.89 at the next, was 0.77 and 0.89, 0.80 and
  0.93). A caller who wants the tool to go past the edge gives a larger
  area. A tool that no longer spends its load past the tone comes off it
  fuller: a stump rubbed over a patch of 6B 7% fuller (it then lays 0.87 of
  it in 28 widths of bare paper, toning it as before, where it laid 0.92 of
  less), a bare stump carries off 1.14 times what it holds from a rub of a
  dark tone (was 1.01); two thresholds of `tests/test_smudger.py` follow.
  Keeping the passes inside costs a rub about a tenth of a second over the
  sheet of `tools/benchmark.py` (a kneaded eraser over it took 1.5 s) and
  a hundredth or two over a patch 240 by 140. `Sheet.shade` is as it was,
  to the bit. Every drawing with a rub in
  it changes: of the thirteen of `tools/fingerprint.py`, the six that rub
  have new hashes and the seven that do not are as they were (and so, over
  `--thorough`, are the 163 of its 325 drawings that do not rub).

- A rub takes the same numbers from the sheet's generator whatever its
  pressure (fusain-2n6). `Sheet.rub` cut each pass into the stretches where
  the pressure is above nothing (a tool is lifted where it is none) and drew
  a seed from the sheet's generator for each, so a pressure that followed
  what was measured of the drawing, or that was changed a little, gave
  another number of draws, and every mark drawn after it had another grain.
  It now draws one seed for the rub, however many passes and stretches it
  falls into, and each stretch has a seed derived from it by integer
  arithmetic (the sample of the rub it starts at, counting from the first
  pass, times an odd step, below 2**31: the same on every platform, never
  the same for two stretches, and a stretch keeps its seed where the
  pressure is nothing elsewhere). One to the rub, not one to a pass: the
  passes of a rub go by the area, the tool and the hand, not by the
  pressure, but one seed to the rub holds however a rub comes to skip a
  pass. A stretch that starts where the rub does has the seed that was
  drawn. Two rubs over the same area whose pressure is nothing in different
  places leave the generator in the same state, and what is drawn after
  them lies with the same grain where they did not reach
  (`tests/test_rub.py`). Every drawing
  with a rub in it changes: of the thirteen of `tools/fingerprint.py`, the
  six that rub (`dust/charcoal paper/4`, `dust/rough/5`, `pastel/sanded
  card/6`, `wax/cartridge/7`, `wax/cold-pressed/8`, `drawing/toned grey/13`)
  have new hashes, the seven that do not are as they were.

- A shading takes one number from the sheet's generator, however many
  strokes it takes (fusain-92i). `Sheet.shade` drew the unsteadiness of the
  hand and a seed for each stroke from the sheet's generator, so how many
  numbers it drew followed how many strokes it laid, and with a `Stick`
  that follows its width, which follows its wear, which follows the
  pressure of all it drew before: a stick worn a little more or less gave
  every mark drawn after the shading another grain. It now draws one number
  and seeds a generator of the shading's own with it, as a `seed` given to
  it does (a shading without one is the shading seeded with the number it
  drew), from which the hand is drawn; each stroke has a seed derived from
  that number and its place in the shading, as the stretches of a rub do,
  so that a stroke's grain no longer depends on how many numbers the hand
  took before it. A worn stick still shades with fewer strokes; what is
  drawn after it has the grain it would have had (`tests/test_shade.py`).
  Every drawing with a shading in it changes, seeded or not: all thirteen
  of `tools/fingerprint.py` have new hashes, and the ten images of
  `examples/output/` drawn with a shading. Nine tests whose bounds held
  for the one sample of the hand they were measured on, and not for every
  sample (the old code failed them on other sheets too), have bounds that
  take in the spread over several sheets, with the values measured; the
  test of a rub's edge gives its rub a seed of its own, since an unsteady
  hand's passes can stray a width apart and leave a band between them
  rubbed a fifth as much as the middle; and two tests that stand in for
  `Sheet.stroke` pass on the seed a shading now gives it.

- An unsteady hand leaves no band between two passes of a rub. The passes
  of `hatching.passes` lie half a width apart, and each strayed across by
  0.15 of a width either way, so that two neighbours were now and then a
  width apart, and the band between them was reached only by the rims of
  the tool: a stump rubbed a row there 0.17 as much as the middle of the
  area, an eraser 0.10. Over 200 rubs of a vine charcoal tone (a stump, a
  finger and a light eraser, ten sheets, twenty seeds each), the least
  rubbed row across the passes was 0.07, 0.01 and 0.01 as rubbed as the
  middle at worst, and below a quarter one rub in twenty. A pass now keeps
  within `hatching.SPREAD`, 0.6 of a width, of the one before (or as far
  as the passes are meant to be apart, if that is further), and leaves
  those after it room to reach the last: 0.69, 0.66 and 0.50 at worst, where
  a steady hand leaves 0.81, 0.83 and 0.66. It draws the same numbers as
  before, so a rub moves only where a pass would have strayed too far.
  `tests/test_rub.py` checks the gaps over 200 samples of the hand, and the
  rows of a rub and its edge over five: the edge test took one sample, which
  had passed by luck (its bounds failed on other samples, before and after);
  its outermost pixels are now asked to be rubbed at all (the rim of a tool
  reaches them as little as a few thousandths as much as the middle), and
  a light eraser, which takes as each pass bears, half as much as the
  middle from the rim in. Five of the thirteen drawings of
  `tools/fingerprint.py` have new hashes (`dust/charcoal paper/4`,
  `dust/rough/5`, `wax/cartridge/7`, `wax/cold-pressed/8`, `drawing/toned
  grey/13`); `pastel/sanded card/6` rubs but no pass of it strayed too far.
  Of `examples/output/`, `charcoal.png`, `crayons.png` and `lake.png`
  change.

- A drop of ballpoint ink inks a pixel as far as it is ink (fusain-9dg).
  A drop inked a pixel as far as all the ink there would cover a firm pass
  thick, its own and the line's: any drop, however small, inked all of a
  pixel that held a firm pass, and closed the white of a heavy line. Along
  a line drawn three times at 0.8 between two rows of pixels, whose edge
  pixels are half bare (0.50), a drop of a billionth of a firm pass every
  four pixels left them a third bare (0.34) and darkened the line by a
  seventh (its luminance 0.355 to 0.303; on rough paper 0.50 to 0.38 bare,
  0.346 to 0.309). A drop now lies on as much of the pixel as it would
  cover a firm pass thick on its own, and on the pixel's bare paper in
  proportion, wherever the line's ink is (the share of the bare paper it
  inks is the share of the pixel it covers): it lies in the valleys of the
  tooth as on its peaks, so it does not go where the line's ink is any
  more than where it is not. The tiny drops change nothing to the third
  decimal (0.503 bare, 0.3546); a drop of 0.006, the least the rim of a
  pen three pixels wide lets go of, 0.503 to 0.499 and 0.3546 to 0.3532 (was to 0.329 and 0.300); one of 0.1, a
  large blob, to 0.437 and 0.331 (was 0.204 and 0.258). Four layers of firm
  black hatching from a pen in the hand: what the pen blots darkens them by
  0.4%, where it darkened them by 0.9%; one and two layers as before (0.1%
  and 0.2%). The ink is what it was: only how much of each pixel it lies
  on changes.
- A light ballpoint line leaves a narrower blob (fusain-js5). A blob was
  as wide at any pressure (`BLOB_WIDE` times the ball), while a light line
  is narrower than the ball. It is now `BLOB_WIDE` times as wide as the
  disc of the ball that meets the paper where it is left: the disc a ball
  meets the paper with where it lands and lifts, by the rule that makes a
  light line narrower (where its round surface comes down to the peaks of
  the tooth: all of the ball from about 0.4 on cartridge paper, a narrower
  disc below; on a paper that gives less, up to a firmer touch, 0.87 on
  printer paper). A pen six pixels wide on cartridge paper, where it
  lands with a drop: the blob's radius (from the spread of its ink) was
  4.18 at any pressure; now 3.12 at 0.05, 3.67 at 0.15, 3.99 at 0.3 and
  4.18 from 0.5 up, where the line is 3.57, 4.39, 4.66, 5.02 and 5.31 wide
  (from the spread of its ink): the blob at 0.05 is 0.75 of its width at
  0.9, the line 0.67. The same ink lies on the narrower dot, thicker: at
  its middle 0.011 at 0.05, where it was 0.006.

- A pencil drawn over pastel shines (fusain-q5k). Since fusain-wo4 a pencil
  over a bed of dust lays the graphite the paper would have held as a film
  on the dust, and `Sheet.shine("graphite")`, and with it the render under
  a lamp, counted only what the paper holds: on cartridge paper, along
  lines 4 wide drawn over two layers of a pale soft pastel, a 2B at full
  pressure shone 0.001 of its pixels where on bare paper it shines 0.431, a
  6B 0.003 (0.763), an HB 0.0005 (0.308), and under a lamp overhead the
  2B line added 0.0001 of the paper's white where on bare paper it adds
  0.025. The film now shines as what the paper holds does. A layer is one
  mixture (fusain-r56): what its film is made of, and what lies over what
  within it, are not kept. What can be told is that the graphite in a
  layer with a film came with a pencil's mark, the film and its crumbs, so
  the film is taken to hold the share of it that the paper holds of a
  pencil's mark (`FILM_GRAPHITE`, 0.85), never more than its own volume:
  exact for a pencil over pastel, in one pass or in many. In the layer on
  top the film shines through all the matte that lies with it, the clay of
  its crumbs as on paper and whatever dust is there: dust a light pass did
  not bury cannot be told from dust laid over the film later, and is taken
  to hide it, which errs on the matte side. A film a later pass buried lies
  on the older dust underneath and shines through the clay of its crumbs
  and the matte on top. Where a pixel has wax a film there is taken for one
  of wax, as a tool takes it. Loose graphite dust from elsewhere that lies
  with a film is taken for its crumbs, matte but for the film's share;
  where there is no film, or there is wax, the shine is as it was, to the
  bit. Now, of lines 0.88 to 0.90 of whose graphite is a film: a 2B at full
  pressure shines 0.381 (0.88 of the line on bare paper), a 6B 0.684
  (0.90), an HB 0.270 (0.88), and the same over the pastel fixed (0.381,
  0.684, 0.272); a 6B at 0.6 0.414 (0.87); a 2B at 0.6 0.175 (0.80) and at
  0.3 0.024 (0.33), a light line burying less of the pastel on top. Over
  one layer of the pastel a 2B at full pressure 0.323 (0.005 before).
  Overhead the lamp adds 0.022 to the 2B line and 0.040 to the 6B (0.025
  and 0.044 on bare paper). Six layers of 6B at full pressure shaded over
  the pastel shine 0.78, against 0.82 on bare paper (0.63 and 0.70 after
  the first). A stump or a finger drawn across a 6B line on loose pastel
  takes the film with the pastel and its shine with it; on fixed pastel
  0.83 of the film stays, and 0.87 and 0.81 of its shine; what they spread
  is dust and shines as it did. Soft pastel laid over the 6B lines at 0.5
  and 0.8 leaves 0.55 and 0.23 of their shine. Every hash of the sheets,
  the reflectances and the returned numbers of `tools/fingerprint.py` is
  unchanged on every paper (`--thorough`, 325 drawings); the shine, the
  render under the default lamp and the raking render change in the 184
  drawings that lay a pencil over a dustier medium's mark (the sticks,
  dust, surface, wax and drawing scenes). A render of a sheet with such
  films takes a kernel's pass more (`fusain._engine.film_sheen`: some
  0.04 s at 1120 by 820); one without, none.

- A pencil drawn over pastel shows on it (fusain-wo4). What the paper holds
  of a dry medium went in the tooth, under whatever dust lay there, and all
  that lay there was counted in the tip's way: over two layers of a pale
  soft pastel a 2B line laid a fiftieth of what it lays on bare paper and
  an HB a hundredth, and 0.85 of that little under the pastel, so the line
  was a hundredth as dark as on bare paper; over the same pastel fixed, a
  seventh, and a thirtieth as dark. A dry medium now lies on what it meets,
  as a waxy one does, but its own crumbs are not in its way: its binding
  is how much the paper holds of its mark on its own ground, crumbs and
  all. What keeps it from the paper is dust beyond those, loose or set:
  more than its own mark would have for what is fast at the pixel
  (`(1 - binding) / binding` of what the paper holds, of a film and of
  paste), by more than `BED_FROM`. That much is a bed between the tip and
  the paper. As far as the bed covers the paper, the tip rides on it and
  what the paper would have held of its mark is a film on the dust
  (`Sheet.film`), which binds none of it; the bed takes none of the room in
  the tooth, nor does what the paper holds under it, out of the tip's
  reach; and dust scrapes less off a tip than paper does, so riding on a
  bed a pass lays a fifth less (`BED_BITE`). On cartridge paper, means of
  39 lines 4 wide drawn at 0.6 (one line varies by a quarter from the
  next), as shares of what the line lays on bare paper and of how dark it
  is there against its ground, before and now: over two layers of a pale
  soft pastel laid at 0.6, a 2B 0.02 and 0.01, now 0.78 and 0.80; an HB
  0.01 and 0.00, now 0.77 and 0.72; a charcoal pencil 0.02 and 0.02, now
  0.77 and 0.91; a 6B 0.13 and 0.04, now 0.79 and 0.88; compressed
  charcoal 0.08 and 0.14, now 0.78 and 0.99. Over the same fixed: 2B 0.15
  and 0.03, now 0.79 and 0.81; HB 0.05 and 0.01, now 0.78 and 0.74;
  charcoal pencil 0.14 and 0.13, now 0.78 and 0.92. Over one layer a 2B
  0.19 and 0.07, now 0.77 and 0.78; over one laid lightly (0.3) 0.57 and
  0.31, now 0.80 and 0.79; over the dust of vine charcoal 0.69 and 0.66,
  now 0.82 and 0.67. A 2B shaded at 0.5 over the two layers 0.01 and 0.00,
  now 0.82 and 0.70 (fixed: 0.09 and 0.02, now 0.83 and 0.71). Of the
  line's graphite, 0.15 lay over the pastel (its crumbs); all of it does
  now, 0.8 of it over the light layer and 0.7 over the charcoal dust,
  through which the point reaches the paper. Twelve layers of a 2B at 0.6
  over the pastel take 0.94 of what they take on bare paper, each less
  than the one before, and the tone goes from 0.65 of the pastel's to
  0.46. A medium over its own layers, or on a ground no dustier than its
  own mark, meets no bed and draws as before, to the bit: every hash of
  the graphite and ballpoint drawings of `tools/fingerprint.py` is as it
  was, on every paper, and vine charcoal over pastel, dust on dust, lays
  what it laid (a tenth over the two layers). Over its own dust that a
  stump or a finger has moved a medium changes by next to nothing (a 2B
  shaded over a 6B rubbed with a stump lays 0.4% more). A firmer stick
  over a dustier one's mark changes, by the same rule: compressed charcoal
  over vine, a pencil over charcoal pencil. A film without wax is as fast
  as the dust it lies on: where a pixel has no wax, a tool lifts a film
  and spreads it with the dust, as far as the dust of the two layers is
  loose (`fusain._engine.loose_ground`), and reaches the dust under it,
  where it slides on a film of wax; a film of wax over dust is lifted and
  shields what is under it as it did, to the bit. Of a 2B line on loose
  pastel a kneaded eraser takes 0.79 in a pass, with the pastel under it
  (0.26 of a line on bare paper, 0.25 on fixed pastel), and a vinyl eraser
  0.93 (0.81 and 0.81); lifted as what the paper holds, the film was a
  lid, and the kneaded eraser took 0.36 of the line and left 0.80 of the
  pastel under it where it left 0.42 beside it (now 0.46). A finger drawn
  three times across the line carries off 0.20 of its graphite (0.07 on
  bare paper, 0.03 on fixed pastel) and leaves it 0.67 as dark (0.93,
  0.96). The film has no wax and leaves no stain: six passes of a vinyl
  eraser take all of the line on any of the three grounds, and the point
  of a knife along it leaves a hundredth of what lay in its way. Limits: pastel rubbed into the paper takes a
  pencil while dust is left on it (blended once with a finger, a 2B line
  0.77 as dark as on bare paper, from nothing; blended again crosswise,
  0.23), and a stick with more crumbs of its own finds such a tooth full
  (a charcoal pencil over the pastel blended once, 0.21); a film buried by
  later passes is one mixture with the dust of its layer, so a second
  layer of a 6B at full pressure leaves the patch a little lighter than
  the first (0.24 of the pastel's tone, then 0.29); and a film of graphite
  on dust did not shine (fixed since: fusain-q5k). No memory is added. Shading takes the time it
  took, but a soft pastel over charcoal, which now looks for a bed in the
  dust under it, about 8% more (2.42 to 2.62 s). Rubbing a tone that has
  such films in it (compressed charcoal over vine) takes about a seventh
  more, the paths for what is not loose dust being taken (with a finger
  5.90 to 6.71 s, with a kneaded eraser 5.52 to 6.34); rubbing graphite,
  which has none, stays within the noise (6.9 to 7.2 s). Thirteen of the
  sixteen example images are unchanged to the byte. In `broadway.png` the
  ironwork drawn in hard pastel over the soft pastel of the fronts shows
  (the post and the arm of the street lamp, the rails of the fire escapes:
  0.66% of the pixels, by up to 167 levels); in `deco.png` the darks that
  compressed charcoal and the charcoal pencil lay over the tones of vine
  are denser (a tenth of the pixels, most of them in the rubbed sky and by
  a level or two; on the tower by up to 56, 410 pixels by over 15);
  `lake.png` moves by a rounding (0.38%, up to 11).
- A white or a pale coloured pencil pressed hard burnishes nearly as the
  blender does (fusain-d6p). `Medium.melts` was 1 for the blender and 0 for
  every coloured pencil, so a white pencil, with which artists burnish,
  polished only as a red one does. A pale lead is mostly wax and white
  filler: `pencils.coloured` now sets `melts` by how pale the colour is
  (`pencils.melts_of`: 0.75 of the blender's for white, falling as the
  square of the colour's distance from white to nothing at 40 (CIE 1976):
  a cream 0.54, a pale blue 0.31, a pink or a mid grey 0.05, any strong
  colour 0; assumed, not measured). It lays its own colour as it does,
  which lightens what it burnishes. One firm layer over three layers of
  red at 0.5 on cartridge paper (a tenth of the pixels showing the paper):
  white left 0.044 of the pixels showing the paper and now 0.025 (the
  blender 0.021), with 0.17 of the gloss of polished wax, now 0.29 (0.48),
  and `press` 0.47, now 0.69 (0.74); the colour is a paler tint of the
  red, as light as it was (L* 66.9 before the white, 67.0 after it before,
  65.6 now; chroma 34, 37 and 40), where the blender deepens it to L* 57
  and chroma 59. A cream does nearly as well (0.027, 0.26 and 0.64); a
  lemon or the red itself as before.

- A vinyl eraser smears a dense or burnished layer of wax along its way
  (fusain-d6p). It lifted less of what wax held, and left the rest where it
  was: rubbed out of a burnished dark over bare paper, it left the paper
  as clean as it found it. A tool that does not smear (an eraser) now
  slides on wax that holds against it and drags some of it along
  (`WAX_DRAG`, 0.05 of what it would have lifted had the wax not held, so
  of a light layer next to nothing), and lays it back, held, wax and all,
  over a width or so (`DRAG_LAY`, 1 for each of its widths) on what is
  beside it and after it. What it drags is on it until it is laid, and the
  sheet and the eraser hold between them the pigment and the wax they
  held. A vinyl eraser rubbed to and fro (at 0.9, 10 wide) out of four
  firm layers of a near-black pencil, burnished, and on over the bare
  paper: the paper within a width of the dark is 15 from its own colour
  (CIE 1976), 12 at one to two widths, 8 at three to five and 2.6 at six
  to nine (nothing before); not burnished, 13, 9, 5 and 1.4; three layers
  of a red, burnished, 6, 4, 2.4 and 0.7; one light layer, 0.2. Within the
  dark it takes about as much as it did (0.726 of the burnished dark is
  left, from 0.732). The assumption, not a measurement: that the smear of
  a burnished black is plainly seen a width out and fades within a few.

- Coloured pencil, crayon and oil pastel leave a stain that no rubber
  eraser takes (fusain-d6p). In life wax carries its pigment into the
  fibres of the paper, and an erased layer always leaves a tint; in the
  model the last of a light layer came off cleanly. Of what the paper
  holds of a waxy medium, a thickness of `STAIN` (0.008, on the covered
  share of the pixel, as far as what it holds is waxy) is now out of reach
  of any tool but as far as it scrapes: a sand eraser or a knife cuts it
  away with the surface of the paper. It is a rule on what the sheet
  holds, not a quantity of its own (the fibres take only so much, so it is
  a thickness, of whatever mixture the paper holds there). A light layer
  of a near-black pencil at 0.3, rubbed with a vinyl eraser: on cartridge
  paper it went to 0.87, 0.99 and 1.00 of the paper's tone in one, two and
  three rubs, and now goes to 0.79, 0.86 and 0.87 and stays there (0.07 of
  its pigment); on a smooth paper (depth 0.4) to 0.66, 0.96 and 1.00, now
  0.48, 0.68 and 0.71 (0.05). Of a red one the tint is 2.5 from the paper
  on cartridge and 6 on the smooth paper (CIE 1976; nothing before). Two
  rubs of a sand eraser take most of it. Graphite leaves none.

- A knife or a sand eraser takes the ink of a ballpoint with the surface
  of the paper (fusain-d6p). The ink is a stain in the paper, and no tool
  touched it: a knife drawn through ink left it as it was. What cuts the
  surface of the paper (as far as a tool scrapes, by that share of its lift
  of what the paper holds) now takes the ink stained into it, onto the
  tool, which counts it with what it took (`Sheet.erase` returns it). Two
  layers of firm black hatching, at a fifth of the paper's tone: a knife
  at 0.9 scratches a line through them back to 0.98 of the paper's tone,
  leaving a two-hundredth of the ink on it, and one rub of a sand eraser
  takes nine tenths of the ink of a patch, which goes to 0.79 of the
  paper's tone. No rubber eraser and no stump touches the ink, as before.

- Where burnished wax is erased or scraped off, the tooth is there again
  for what is drawn next (fusain-6k1). Polishing wax raised the `press` of
  the sheet, as a hard tip that crushes the tooth does, and it stayed when
  the wax that levelled the tooth was taken off, so a tip touched all of
  the pixel there. Polished, it is the wax that is level: where a pixel
  holds wax, its `press` is now taken to be the wax's polish, which keeps
  the tooth level as far as the wax fills it (`WAX_FAST` thick); as a tool
  thins the wax below that, the polish goes, and with all of it the tooth
  is as the paper's own, but for what a dry tip crushed beyond what the
  wax held up. A sand eraser or a knife also cuts the surface of the paper
  as far as it scrapes, which takes even that (a tooth three firm layers
  of 4H crushed by 0.91 comes back to 0.15 under a sand eraser, and stays
  at 0.91 under a vinyl one). The sheet keeps one `press` for both: a waxy
  tip does not crush the tooth, and storing the crush apart would cost a
  float a pixel and an argument through every stroke kernel, for a crush
  under or over wax that is then taken off, which goes with it. Four firm
  layers of a dark coloured pencil burnished, then scraped back: a light
  layer of HB shaded over them touched 0.998 of the pixels where a knife
  had been and 0.96 where a sand eraser had, and now 0.64 and 0.73, where
  it touches 0.68 of fresh paper; its grain (the spread of what it laid
  from pixel to pixel, over its mean) was 0.35 and 0.44 and is 1.34 and
  1.24 (fresh paper: 1.27). The relief there was a twenty-eighth of the
  paper's and is as the paper's, `press` was 0.996 and is 0.000 and 0.097,
  and the gloss of the wax 0.000 and 0.082, now 0.000 and 0.019. A vinyl
  eraser rubbed three times over the burnished dark leaves most of its wax,
  and the tooth there as level as it was (`press` 0.996, now 0.991).

- A tool never takes more than it has room for. `Tool.holds` made a tool
  lift the less the fuller it was, but what it picked up off one pixel was
  not held to the room it had left, so a tool that holds less than one
  pixel's pickup ended up holding more than it holds and laid the excess
  back later: a stump 10 wide that holds 1e-6 for each pixel of its face
  (1e-4 in all) dragged 60 pixels through vine charcoal carried 17,
  170,000 times as much. What it takes off a pixel, of the dust, of what
  the paper holds, of a film and what is set, is now held to the room it
  has left, and the paste and the wax go with it in proportion: it carries
  no more than it holds, and the sheet and the tool hold between them what
  they held. So with the ink of a ballpoint that a sand eraser or a knife
  cuts out of the paper: it takes it as far as it has room left once it has
  taken the rest, and where it is full the ink stays on the paper (one
  that holds 1e-6 drawn through firm hatching took the ink of every pixel
  it cut, 167 to 351 times what it holds). No preset ever reaches the
  limit: every drawing of
  `tools/fingerprint.py --thorough` is as it was, to the bit.

- A smudge or an erase is as strong where it lands and lifts as in its
  middle. A tool rubbed only as it slid on, so the paper under it where it
  landed or lifted got from nothing (at its back) to a full pass (at its
  front), as a stroke did before its ends were made full. One pass of a
  kneaded eraser 14 wide at 0.7 over a 4B shading took 0.20 of the pigment
  at the ends of its path and 0.05 five pixels outside them, where it took
  0.31 further on (a vinyl eraser 0.73 and 0.23, against 0.91); a dab took
  0.07 (0.32), and a pass three pixels long 0.11 (0.46). A stump stirred up
  0.71 and 0.64 of the dust underneath at the two ends of its path, 0.24
  and 0.13 five pixels outside them and 0.37 under a dab, against 0.91
  further on. Where a tool lands and where it lifts, the paper it covers is
  now rubbed as though the tool had slid onto it from behind and off it
  ahead (`put_tool`, as `put_down` for a tip): an eraser takes 0.30 to 0.31
  (0.91) everywhere under its pass, out to the outline of the tool, and
  under a dab; a stump stirs 0.89 to 0.92, and 0.94 under a dab. A dab is
  one pass of the tool on the paper under it. Every picture that rubs or
  erases changes where the passes end; a short dab of an eraser takes four
  times what it took. `Sheet.rub` stops its passes half a width of the tool
  short of the edge of the area, as `Sheet.shade` does its round strokes,
  so that an area is rubbed up to its edge and no further (with a steady
  hand, a vinyl eraser takes 0.99 of what it takes in the middle just
  inside the edge and nothing outside it; run to the edge, full passes took
  as much from the half width outside the area as from inside, and the
  passes as they were before, two fifths, with a tenth less just inside).

- A flat end on a short hook no longer cuts into the stroke before the
  bend (fusain-hay). The cut of a flat end is the line through the end of
  the path across the way the stroke goes there, and it took whatever lay
  past that line of all that the tip laid within its reach of the end. A
  path that turns within that reach and ends (a hook shorter than half the
  tip) has the band it drew before the turn across that line, and lost it:
  a flat stroke 24 wide drawn 140 pixels and turned square for 2, 4 and 6
  pixels had, 4 to 12 pixels before the corner, 0.66, 0.53 and 0.87 as
  much on the inside half of the stroke as on the outside half, and of
  the 131, 128 and 122 pixel-passes that the stroke has past the line
  within half a tip before the corner with round ends, 42, 11 and 10 were
  left. Over tips 6 to 48 wide and hooks of an eighth of the tip to just
  under half of it, at the end of a stroke or at its start, turning either
  way (76 strokes), the band before the bend kept as little as 0.81 of
  what it has with round ends, its worst pixel nothing, and the inside
  half was as little as 0.34 of the outside half. The cut now takes what
  lies beyond the stroke and not the stroke: past the line, a pixel stays
  as far as it is in the band of the stroke itself, within half the tip's
  flat of the path at one of its samples within reach of that end and
  behind the line across the path there (ahead of it, where a stroke
  starts): `within_ends`, `ends_near`. A flat stroke is then what the
  width of its tip sweeps along its path, from one end to the other. The
  three strokes have 1.19, 1.29 and 1.34 times as much inside as outside
  (it is the inside of a bend, as before a bend with a long leg after
  it), and 109, 81 and 71 pixel-passes past the line before the corner:
  the band, up to the last sample before the corner, where it is cut
  square, as the tip goes round a corner from one sample to the next; of
  the 76, every one has in the band before the bend what it has with
  round ends, pixel for pixel and to the bit, and 1.03 to 1.42
  times as much inside as outside; nothing is left past the line beyond
  the corner or beside the band, before or now, and what leaves the tip
  is on the paper. A hook grows without a jump: with no hook the cut is
  across the long leg; over the first sample's spacing of hook (an eighth
  of the tip) it turns with the last two samples to lie across the hook,
  as it did; from there to half a tip it crosses less and less of the
  band before the bend, and at half a tip it is clear of it and the
  stroke is what it was. From no hook to one of 14 pixels, a quarter of a
  pixel at a time, a stroke 24 wide lays 2583 pixel-passes to 2885, no
  step more than 25 (a jump would be a quarter of the tip's disc, 113, or
  the half band the cut took, about 100). A stroke with no such hook is
  laid as it was, to the bit: a straight flat end, a last leg of half a
  tip or more, a curve no tighter than the tip, round ends. Every
  recorded hash of `tools/fingerprint.py` is as it was, and 318 of the
  325 drawings of `--thorough`; the seven are `drawing` on seven papers,
  where the unsteady hand bends the stroke of a flat-ended shading that
  the corner of its area leaves 3 to 5 pixels long, for a tip 8 wide: a
  hook as long as the stroke. No image of `examples/output/` changes:
  `strokes.png`, `deco.png` and `broadway.png`, whose scripts draw flat
  ends, are regenerated byte for byte, and the others draw none.

- A rubbed join of two colours is a passage from one to the other, a tool
  no longer carries dust far, and a stump that has been rubbed for long in
  a tone draws on bare paper for longer than it did. Where it bore, a stump
  or a finger lifted about two thirds of the loose dust into its load in a
  pass and laid the load back over several of its widths, so a finger
  rubbed to and fro across the join of an orange and a blue pastel left one
  even mix over all that was rubbed, and a finger rubbed over house fronts
  and a sky brought the fronts all over the sky. A real finger smears most
  of the dust within a fraction of its width. Four things changed:
  - What a stump or a finger lifts into its load in a pass is little
    (`tools.STUMP`: `lift_loose` 0.04, from 1.5; `tools.FINGER`: 0.03, from
    1.2). Most of what they move is the dust on their face, taken up as
    strongly as they smear and laid back within a few pixels (`SPREAD`, 16
    times over each width of the tool, from 4), so that it goes a fraction
    of the tool's width.
  - A tool holds only so much: `Tool.holds`, a volume for each pixel of its
    face (1.5 for a stump, 1 for a finger; by default any amount, as an
    eraser or a chamois does). It lifts the less the fuller it is and
    nothing once full, so it fills slowly: a stump takes a fifth of what it
    holds in one pass through 32 of its widths of vine charcoal, half in a
    rub of an area and over nine tenths in three; a heavy pastel fills it
    in one rub.
  - A tool gives the dust of its load to bare tooth only. `Tool.lay` (0.3
    for a stump, 0.25 for a finger) is the share of its load it lays over
    each of its widths on bare paper; paper that carries pigment takes
    less, and none once the pigment covers its tooth. So a full stump
    leaves a tone as it is however often it goes over it, a finger loaded
    with orange and rubbed over blue leaves it blue (a two-thousandth of
    orange on top), and still a loaded stump tones bare paper, and dirties
    what bare paper it strays onto at the edge of a tone. Paste sticks to
    any paper.
  - What is on the face of a tool is shared out, in each step, among the
    pixels under it by how the tool bears on each, whatever order they come
    in. (It was laid pixel by pixel from what was left, so the rows first
    visited got more: the trail of a stump was lopsided.)

  Measured, a finger 22 wide rubbed at 0.6 to and fro across the join, in
  18 passes five widths long: the share of the blue in what lies on top was
  0.42 one width of the finger left of the join and 0.57 one width right of
  it (0.38 and 0.61 at two widths), and is now 0.19 and 0.81 (0.08 and
  0.92); the passage from a quarter of the blue to three quarters was 5.5
  widths, all that was rubbed, and is 1.4. After 72 passes it was 5.7 widths
  and is 1.8; with oil pastels at 0.8 and 36 passes, 5.7 and 2.6. A stump
  20 wide dragged at 0.7 out of vine charcoal onto bare paper dragged out
  what 20 columns of the tone hold, two fifths of it within one width of the
  edge, and its trail three and a half widths out was a seventh of the tone
  (luminance 0.75 against the paper's); it drags out what 8 columns hold,
  two thirds of it within one width, and the trail there is a fiftieth of
  the tone (0.965). A stump rubbed all over a patch of 6B and then drawn
  over bare paper laid 0.59 of the paper's tone where it landed, 0.92 eight
  of its widths on, 0.98 after 16 and 0.99 after 24; it lays 0.72, 0.83,
  0.91 and 0.96 (loaded on vine charcoal: 0.38, 0.78, 0.94 and 0.98; now
  0.63, 0.80, 0.90 and 0.955). What fusain-6lr fixed still holds: eight rubs
  of a stump in the hand over 8B leave 0.993 of the graphite (0.996) and the
  tone as it was; where the passes of a rub land the pigment is within 2.2%
  of what it is further on; a finger evens vine charcoal laid with the side
  of the stick to 0.31 of its grain (0.43). A bare stump, which is clean
  at the start of every rub and takes away what it holds at the end, takes
  4.7% in those eight rubs (2.7%): one load each, and a stump now holds
  half as much again as it used to carry off such a tone.

- A stump or a finger lays its dust on paper and hardly on wax. A tool laid
  what it carried on any pixel, waxed or not, so charcoal smeared across a
  crayon line greyed it, where a stick of charcoal drawn over the same line
  skips it. Wax is now as slick to the dust a tool brings as to a dry
  stick: where it covers the tooth (a film `COVERS` thick, on the covered
  share of the pixel) it takes `RESIST_DRY` less of the dust in the tool's
  load and on its face, and of the graphite the tool pushes along; what it
  refuses stays on the tool, which carries it on. Paste sticks: what a
  finger carries of an oil pastel is laid on wax as on paper. A stump
  loaded with vine charcoal and dragged across one firm line of white
  crayon left on the line 0.97 of what it left on the paper either side,
  and the line came out as grey as that paper (0.71 of the paper's tone
  where it was crossed, 0.97 beside the stump's way); it now leaves 0.13,
  and the line is at 0.94 where the paper either side is at 0.76 to 0.79.
  A loaded finger rubbed across three firm layers of crayon left on them
  what it leaves at the same place on a bare sheet, whatever it carried;
  loaded with charcoal, soft pastel or 8B it leaves a tenth of that (0.08
  to 0.10: what the bare flecks between the crayon take), and loaded with
  oil pastel a half (0.47: its paste). Nothing is lost: the sheet and the
  tool hold between them what they held. A tool also slides on a film of wax that lies over dust, and
  on dust that is set: it stirs up and lifts the loose dust underneath only
  as far as those do not cover it. (It took the dust up through the film,
  and the wax no longer taking it back, a finger would have carried more
  off a patch of charcoal with crayon over it than off the charcoal alone:
  it shifts 0.37 as much within the patch, from 0.50, and of pastel 0.73,
  from 0.91.) Where the film is thin the wax is not yet slick, to a tool as
  to a stick: a white crayon line over rubbed charcoal still greys where a
  stump crosses it (0.60 of the paper's tone to 0.40, from 0.33).

- The wax of an oil pastel goes with the paste a finger smears. The sheet
  counts the wax of a pixel wherever it lies there, and a tool moved the
  paste without it: the smear had no wax and was not slick, the place it
  left kept all of it, and what an eraser took off with the held pigment
  was counted nowhere. The wax now goes with what it is in (what the paper
  holds, the films and the paste): a tool that takes a share of those takes
  that share of the wax, carries it (`Smudger.wax`) and lays it back with
  its paste. Three firm layers of oil pastel smeared out over bare paper
  with a finger: the smear, 0.9 thick, held no wax and holds 0.300 of its
  volume (the pastel's is 0.3); what was left where the pastel was laid
  held 0.46 of its volume and holds 0.298; the wax on the sheet and on the
  finger together was 15 short of the 5,695 laid and is all there. Charcoal
  shaded over the smear took a fifth of what bare paper takes (the tooth
  was full) and takes a fortieth (it is slick).

### Examples

- `examples/stilllife.py` plans its drawing in about an eighth less time
  (processor time 34 to 37 s before, 30 to 31 s after; three runs each, one
  after the other, on a machine busy with other work), and the picture is
  the same to the byte (`stilllife.png`). Two things, from a profile before
  (cProfile, 70 s) that put 12 s in the cloth's folds and 20 s in the
  bundles of strokes:
  - `cloth_part` worked out the folds over the table twice at the same
    points: `cloth_point` for the cloth's height, and again for the heading
    (the cloth's 218,400 points, in passes of 4096: 54 of the 163 passes of
    `folds_over`). The table is 84% of the cloth, and where the cloth lies
    on it a point is where the first call looked, to the bit.
    `cloth_surface` now returns the heading with the points (`cloth_point`
    is its points), and `cloth_part` uses it: 6.5 to 3.3 s of processor
    time for the cloth.
  - `Hatcher.bundle` made a call of Python round the compiled `_trace` for
    each way of each stroke (218,048 strokes, 436,096 calls, each with a
    zero-filled buffer, two copies and a `vstack`; only 9,544 strokes were
    laid, the rest were too short and thrown away). The strokes of a bundle
    are traced by one compiled `_trace_bundle`, from their starts (worked
    out by numpy in one go, with the single-precision products the loop
    had), into a buffer made for the bundle, where each stroke that is long
    enough is marked in `full`, as before, for the next to keep off. What
    stays in Python is what cannot be batched: the three numbers each
    stroke draws from the generator, in the order they were drawn, since a
    different order gives different numbers. The bundles without their
    strokes' drawing: 5.7 to 3.8 s of processor time (1.1 s of it the
    compiled kernel), and the drawing of the strokes `Hatcher.draw`, 3 s,
    is as it was.
  - Left as they are, the largest of what is left to plan: `Hatcher.chances`
    (4.5 s, 33,000 calls, each drawing the generator's numbers for the
    stretch of the ink map that holds the cells that want some) and the rest
    of `Hatcher.lay` (3 s of its own).

- `macaw.png` is drawn with other grain, and a change to one part of the
  model stays where it is (fusain-ny8). Each part of the drawing (the
  wing's feathers, the small feathers, the mottle, and each step) draws
  from a stream of its own, spawned from the one seed; a loop whose draws
  depend on what is seen has a stream for each item; the places of strokes
  and of small feathers are chosen pixel by pixel (a pixel that gains or
  loses weight gains or loses its own point, and no other moves); and every
  stroke, shade and erase has a seed (`Sheet`'s `seed`) from its part or its
  item. Hiding 144 pixels of the plumage beside a foot changed 82% of the
  drawn pixels before, and changes 2.6% now (three quarters of them within
  150 pixels of the patch). It is the same drawing: against the one before,
  the mean luminance of the bird goes from 0.2621 to 0.2613 (linear), the
  pixels drawn from 383,705 to 383,477, and in blocks of 100 pixels the
  luminance differs by 0.0042 at the median and 0.0248 at most. The number of strokes in a
  layer is now about what it was, not exactly.

- The four examples that call `Sheet.rub` (`charcoal.py`, `crayons.py`,
  `deco.py` and `lake.py`) draw another grain in every rub, and in all
  they draw after one: `charcoal.png` (the stump is loaded on a scrap of
  its own sheet, so only the stump it carries changes), `crayons.png`,
  `deco.png` and `lake.png`. The rest (`ballpoint`, `broadway`, `colour`,
  `farm`, `fixative`, `macaw`, `papers`, `pencils`, `shine`, `stilllife`,
  `strokes`, `tips`) rub with `smudge` or not at all, and are unchanged to
  the byte by this.
- `lake.png`: `LIGHTEST_RUB`, the least that `examples/lake.py` let a rub
  press anywhere in its mask so that a field that fell to nothing here and
  there did not cut each pass in stretches with a seed each, is gone: it
  was kept for the fault above and for nothing else. The rub is still made
  wherever the mask is, however little the field presses (so that the
  sheet's generator is drawn from just as often). With the new rub, the
  picture it makes with the floor and without it is the same to the pixel
  but for 0.6% of them, which differ by a level.
- `colour.png`, `crayons.png` and `macaw.png` change with the erasers and
  the knife above; every other image is unchanged to the byte. In
  `colour.png` the band the vinyl eraser rubs across the red keeps a pink
  stain on the half that was not burnished (its mean luminance goes from
  220 of 255 to 213) and is as it was on the burnished half. In `crayons.png` the
  lights the kneaded eraser lifts out of the oil pastel keep a little of
  the blue, and where it lifts off past the edge of the patch it leaves a
  faint smear of it. In `macaw.png` the lines the knife scratches through
  the burnished wax lose the gloss of the wax with the wax, by at most 4
  levels; its lights under the bill and in the eye stay.
- Every image is regenerated with the changes above. Seven are unchanged to
  the byte (`farm`, `papers`, `pencils`, `shine`, `stilllife`, `strokes`,
  `tips`: nothing in them is rubbed or erased), and `macaw.png` and
  `ballpoint.png` but for the ends of their eraser's and knife's passes.
- `colour.png`, `fixative.png` and `crayons.png`: the join of orange and
  blue rubbed with a finger is a passage from the one to the other, where
  it was one olive. As colour differences (CIE 1976, of which 2 can just
  be seen), two widths of the finger left of the join the patch of
  `fixative.png` was 65 from the orange and 78 from the blue and is 38 and
  89; two widths right of it, 72 and 72, now 87 and 46 (the two colours
  are 108 apart, and the middle of the passage, their mix, is about 70
  from both). `examples/colour.py` rubs each row of its join both ways, as
  the other two do: rubbed one way by turns, the rows showed as stripes,
  now that a pass carries a colour less than a width. In the lower half of
  `fixative.png`, the join that was fixed before it was rubbed stays a
  join, where the finger blurred it half-way.
- `charcoal.png`: the stump's trail out of each tone is shorter (vine
  charcoal: 0.50 of the paper's tone 16 pixels past the edge of the tone,
  now 0.84, and 0.69 at 50 pixels, now 0.95). `examples/charcoal.py` loads
  its stump on a scrap of the medium before it draws down the bare paper:
  a stump fills slowly, and one more pass through the tone no longer loads
  it. The lines it draws are about as dark as they were, and last longer
  (along them, against the paper's tone, compressed charcoal: 0.43, 0.81,
  0.79, 0.91, now 0.46, 0.77, 0.66, 0.81; vine: 0.78, 0.95, 0.95, 0.98, now
  0.85, 0.95, 0.92, 0.96; 4B: 0.91, 0.98, 0.99, 0.99, now 0.89, 0.96, 0.94,
  0.97).
- `crayons.png`, third row: where the stump crosses the crayon lines over
  rubbed charcoal it lays less dust on them, and they still grey (see
  Fixed).
- `deco.png`: `examples/deco.py` rubbed with a clean stump or finger at
  every stroke, which used to carry off much of the charcoal and now
  carries off little, and its finger went through the bank of cloud and on
  over the clear sky: under the new rubbing the picture came out darker
  (the clear sky 200 to 189 of 255, the faces turned from the sun 69 to 61)
  and the clear sky in bands (the means of 16 pixel blocks against the 80
  pixels around them spread 5.2, for 3.8). It now holds each tool through
  its passage, rubs the clear sky and the cloud with a finger each, lightly
  and in passes further apart, and lays a little less charcoal under what
  it rubs. Measured against the image as it was: clear sky, level 200.0 and
  now 200.1, the spread of 16 pixel blocks against their surroundings 3.8
  and 4.3, the grain from pixel to pixel 12.7 and 12.0; faces turned from
  the sun, level 68.7 and 68.8, grain 31.2 and 31.3; heavy cloud 38.3 and
  38.6; cast shadows 103 and 104; lit faces 174.5 and 174.1; the blocks
  round about 63 and 66; the whole sheet 120.8 and 119.5.
- `lake.png`: four fifths of its pixels change, a seventh of the sheet by
  over 8 levels (3.8 levels on average), and it is a little nearer the
  values it is drawn to: its mean distance from them goes from 9.5 levels
  to 9.1 (seen through a blur, from 6.6 to 6.3), and the share of the sheet
  over 20 levels too dark from 2.2% to 1.1% (too light: 5.9%, as before).
  The lights its erasers lift are full to their ends, and the crests of
  its ridges show a little more against bodies rubbed as evenly.
- `broadway.png`: a fifth of its pixels change, few by much (1.8% by over 8
  levels): the sky and the far end of the street, which are rubbed. Its
  finger is still pulled down from the sky over the far fronts and wiped
  before each pull: rubbed to and fro over both, it now takes the open sky
  a tenth of the way to the colour of the fronts where it took it four
  tenths, and the pulls three hundredths (seven).

### Added

- `Smudger.wax`: the volume of wax in what a tool in the hand carries.
- `Tool.holds`: how much pigment a tool holds when it is full, for each
  pixel of its face (see Fixed).

### Changed

- NumPy 1.x is no longer supported: fusain needs `numpy>=2.0`, and with it
  `numba>=0.60`, the first numba that runs on NumPy 2 (fusain-klv). The
  picture depended on NumPy's major version, and only NumPy 2 was ever
  tested: under NumPy 1.26.4 (with Python 3.12 and numba 0.68.1) the colour
  of every paper comes out as other numbers than under NumPy 2
  (`murex.spectral.srgb_to_reflectance`), so 44 of the 421 marks of
  `tools/fingerprint.py` differ, in all thirteen drawings, and
  `examples/stilllife.py` draws another picture. With the same Python and
  numba and NumPy 2.5.4 all 421 are as recorded, and so they are with the
  oldest pair now allowed, NumPy 2.0.2 and numba 0.60.0. Nothing changes
  for anyone on NumPy 2.

- A short stroke takes a little less time around its kernel (fusain-238),
  and no picture changes: the thorough fingerprint (325 drawings) and
  every image of `examples/output/` come out as before, to the byte. A
  pressure, a width or a tilt given as a number is made an array of that
  number straight away; murex was asked to evaluate it as a profile, and
  fusain then broadcast and checked what came back: about 4 microseconds
  each for a pressure and a width, 9 for a tilt. The shape of the tip is
  filled in column by column, where three arrays were stacked and copied
  (4 microseconds). Measured with the kernel replaced by a compiled
  function of the same arguments that does nothing (CPU time, best of 25,
  on a shared 4-core machine): each of the 2000 short strokes of
  `tools/benchmark.py` took 95 microseconds around its kernel and takes
  75; the strokes of a shading, whose pressure is a function, 167 and 155.
  With the kernel, the 2000 short strokes take 0.50 s of CPU, from 0.55 s
  (best of 20, three times over), a tenth less; 40 small patches of
  hatching, 1.64 to 1.72 s, from 1.71 to 1.75 s; shading the whole sheet,
  whose strokes are long, as before. What is left of the 75: murex's
  sampling of the path, about 33 (making the path, 5; its dense points,
  10; resampling them, 18, of which numpy spacing the samples takes 5 and
  interpolating them 5); fusain's own checks and arrays, about 14; handing
  the kernel its 19 arguments, 1.3 (one argument takes 0.15: packing them
  into fewer would save next to nothing).
- What a ballpoint's ball gathers beyond the most it holds is documented
  (fusain-45t; `HeldPen`, `fusain._engine.roll_stroke` and
  `docs/design.md`). The rim holds a drop at most (`BALL_BLOB`: 0.48 for a
  pen three pixels wide), and what the ball would gather beyond that is
  not gathered: it is on neither the paper nor the ball. So the ink of its
  drops and the ink on its ball are `BALL_GATHER` times that of its lines
  only while the rim has never been full; a straight line fills it, from a
  clean pen three pixels wide, in 130 pixels at full pressure, 240 at 0.5
  and 360 at 0.3. Nothing changes; a test pins it.
- Strokes take half the time, rubs with a finger or a stump four tenths of
  it and rubs with an eraser a fifth, and no picture changes: every array
  of a sheet after any sequence of calls, every number returned and every
  rendered image is what it was, to the last bit (the recorded fingerprint,
  the thorough one, 325 drawings, and every image of `examples/output/`).
  On a sheet of 1120 by 820 pixels (one core of a 2.3 GHz Xeon, best of 5):
  shading it with an HB of width 6 takes 1.4 s, from 2.9 s, and over a tone
  1.65 s, from 3.0 s; with a soft pastel of width 24, 1.0 s, from 2.0 s,
  and over charcoal 1.4 s, from 2.5 s; with the side of a stick of vine
  charcoal, 0.72 s, from 1.7 s; with a coloured pencil pressed hard,
  1.75 s, from 3.5 s; 2000 short strokes take 0.43 s, from 0.78 s; rubbing
  the sheet with a finger takes 2.4 s, from 6.6 s; with a stump over
  graphite, 3.2 s, from 7.1 s; with a kneaded eraser, 1.1 s, from 6.2 s.
  That is as fast as with the two options of numba's that its
  documentation does not list, or faster (on the code as it was, they gave
  1.46, 1.66, 1.08, 1.55, 0.76, 1.82 and 0.45 s; 2.34, 3.25 and 2.36 s),
  with documented options only (fusain-95m). Compiling the kernels the
  first time takes 19 s, from 18 s. What it took:
  - Numba counts a reference to an array each time a helper is given it,
    and one back when the helper is done with it, two atomic operations,
    and drops the pair only where it can see the two as one. With an early
    return in a helper, or an `if` that holds its last use of an array, it
    cannot, and these were counted for every pixel: 49 places in the kernel
    of a stroke, 40 in the one that settles its dust, 120 in that of a rub;
    half the time of a stroke and two thirds of that of a rub. The helpers
    are now written so that every pair is dropped. One that reads an array
    reads what it needs of it first, before its first `if`; what one does
    to the sheet only sometimes is in a loop that runs once or not at all
    (`for _ in once(...)`), where the `if` or the early return was; what a
    tool lifts is worked out from the numbers of the pixel, read once
    (`layer_at`), and then taken; one helper pushes along for both kinds of
    tool (`push_along`). The top of `fusain/_engine.py` says why and how.
  - The kernels called for every step of a stroke or a rub (`bear`,
    `rub_under`) call no other kernel and allocate nothing, so their own
    arrays are no longer counted at every call (30 places, and 48 in
    `slide_on`, a kernel that stood between `lay_stroke` and `bear`): what
    `sunk_into` works out is compiled into `bear`, `lay_stroke` turns the
    tip itself where a stroke turns sharply (`in_parts`, `turned`), and
    `rub_stroke` gives `rub_under` the room it works in.
  - `tools/refcounts.py` tells what each kernel still counts, from its
    LLVM, and `tests/test_refcounts.py` fails if one counts anything as it
    works (nothing, on Python 3.11 to 3.15).
  - A tip or a tool does no work that would leave a pixel as it is. An
    eraser presses nothing in, stirs nothing and spreads nothing, and went
    through all the parts of the pixel's dust for each, adding and taking
    nothing: half the time of its rub. A dry tip pressed firmly on paper
    without wax polishes nothing and drags nothing, and went through all
    that the paper holds there: 5 to 10% of a stroke of charcoal or pastel.
  - Rendering the sheet takes 0.6 s, from 1.0 s, and its reflectance
    0.45 s, from 0.8 s. Half of a render was numpy gathering the pixels
    that carry pigment out of each layer, dividing them by the share of the
    pixel that is covered, and putting what is seen back among the bare
    paper: two kernels do that (`thicken`, `show_through`), for the ink of
    pens too, and Kubelka-Munk (murex) works each layer out in place.
  For those who call `fusain._engine` themselves: `slide_on`,
  `slide_graphite`, `drag_wax` and `lift_bonded` are gone (`in_parts` and
  `turned`, `push_along` and `slip_of`, `take_bonded`); `tip_at` takes the
  tilt and the length of the side; `bear_on_row` a count of pixels;
  `rub_under` and `put_tool` room for what the tool lifts; `bond_dust` the
  volume it bonds (`loose_in`); and `loose_ground`, `bonded_share`,
  `waxy_share` and `room_for` take the numbers of a pixel, not the arrays.
- `rub_stroke` (in `fusain._engine`) takes the tool as one array ([radius,
  `lift_loose`, `lift_held`, `lay`, `press`, `smear`, `scrape`]) and the
  pace at which it lands and lifts; what the tool does as it slides one
  step is `rub_under`. What a tool carries has a sixth part, the wax that
  came with it. A rub takes a tenth to a sixth longer than it did (the
  landing and the lifting, mostly).
- `ends_near` (in `fusain._engine`) fills what `within_ends` takes from
  the path itself, for the tip at a sample: for each end that is cut
  square and within the tip's reach, the line of the cut and the lines
  across the path at the samples within that reach, each with the half
  width of the stroke's band ([how many for the start, how many for the
  end, then x, y, way (x, y) and half width for each]); `ends_of` is gone.
  A line that cannot matter is left out (`band_way`: one that leaves
  nothing the cut does not, for all the paper the tip can reach), so a
  stroke that does not bend near its end is cut by one line, as it was.
  `bear_on_row` cuts in a loop of its own, which only a tip near a square
  end goes through. A stroke takes what it took, with round ends or flat:
  over thirteen workloads (those of `tools/benchmark.py` that draw, and
  shadings, short strokes and tapering strokes with flat ends), the best
  of eleven runs of each, before and after turn and turn about on a busy
  machine, differ by 5% less to 4% more, as two runs of the same code do
  (HB shading 1.79 s and 1.75; 2000 short strokes 0.56 and 0.58, with flat
  ends 0.55 and 0.56).
- `tools.STUMP`, `tools.TORTILLON` and `tools.FINGER` lift less into their
  load, hold only so much and lay it on bare paper (see Fixed). Any `Tool`
  with a `lay` now gives the dust it carries to bare tooth only: one of
  one's own that lifts strongly takes from a tone what it lifts and gives
  none of it back there, as `tools.CHAMOIS` does.

### Fixed

- The side of a stick laid flat bears evenly across its length
  (fusain-bqe). Every tip held on its side marked an ellipse across the
  stroke, so the band of a stick of charcoal or pastel fell off from its
  middle to its edges. That is the side of a pencil, the cone of lead its
  sharpening bared, which bears hardest along its middle; a stick of
  charcoal, pastel or crayon laid flat bears alike all along its length.
  Which it is is the medium's (`Medium.flat_side`): a band laid by the side
  of a stick is now even across, rounded only within the stick's thickness
  at its ends, and cut straight across where it lands and lifts. A piece of
  vine charcoal (or compressed charcoal, or soft or hard pastel) 90 pixels
  long and 6 thick at 0.6: of the band, the share at 0.9 of its middle or
  more was 0.43, now 0.92 on a level paper (0.28 to 0.31, now 0.70, on
  cartridge paper, its streaks and its grain aside), and from there it
  falls to a tenth in 3.5 pixels, from 23.5. A pencil on its side is as
  it was. Where a stroke turns, each part of a tip now lays as far as it
  slides, so the outside of the turn lays as much as the inside: round a
  quarter circle of radius 150, a band 90 wide has 0.95 of its middle's
  tone on the inside and 0.96 on the outside, where it would have 1.10 and
  0.84 had each part laid as far as the middle went. (Every stroke that
  curves changes by that, a little at its rims.) No part of a tip lays more
  than a pass's worth on a pixel in a step: where a stroke turns so sharply
  that its far end would sweep further (under about a sixth of the
  half-width of the mark), the step is laid in parts turned bit by bit
  across the angle, and where it doubles back the tip is drawn back
  without turning round. A shading lays the bands
  of a stick laid flat side by side, overlapping only by their rounded
  ends, as its point's strokes overlap; they were half a band apart. Two
  crossed layers of the side of vine charcoal on Ingres paper at 0.8 are
  darker for it (0.47 of pigment per pixel, from 0.32), and leave 0.054 of
  the paper bare, from 0.10.
- A point narrower than a groove follows it down (fusain-d7v). Every tip
  rode on the paper either side of a groove, so a fine point drawn along
  an indented line skipped it. A tip now rests on the highest of the paper
  it covers: the sheet keeps no width of a groove, since the dent around
  the tip says it, for grooves of any width, crossed or pressed over each
  other. A tip wider than the groove still rides over it, and one
  narrower goes down into it. Along a groove indented at 0.7 by a stylus 3
  wide, on cartridge paper, an HB point a pixel wide lays on its floor
  0.75 of what it lays on the paper left alone (nothing before), a 2B
  point 3 wide 0.49 (nothing), and one 4 wide nothing, as before; in a
  groove 6 wide a 2B point 4 wide lays 0.41. A 2B 2 wide shaded across a
  groove lays in it 0.39 of what it lays beside it (nothing); broader
  tips still skip it. The floor of a groove is flat where the point that
  pressed it took the peaks off, as a flattened tooth is, and now bites a
  dry medium as little (`FLAT_BITE`): a fine point touches all of the
  floor and lays half as much on what it touches. So a groove fills less
  under a broad soft pencil that reaches down to it: indented at 0.3, a
  6B pressed hard lays 0.67 of what it lays beside it (0.96 before) and
  shaded lightly 0.24 (0.43), a 2B shaded lightly 0.13 (0.22); a 2B over
  a pen line laid at 0.2, 0.6 and 1.0, 0.89, 0.26 and 0.002 (0.91, 0.42
  and 0.005). Over three layers of light pen hatching, which pressed the
  paper down all over, an HB goes down with it and lays 1.10 of what it
  lays on paper left alone (0.98), and over firm hatching 0.65 (0.39).
- Burnishing drags a film that lies over dust (fusain-pv2). A firm pass
  polished and smeared only what the paper held; a film of wax laid over
  dust stayed where it was, though its wax, which is counted with what the
  paper holds, went with the tip. A firm tip now drags a film as it drags
  what the paper holds, by the same share (wax is wax wherever it lies,
  and it is the tip's polish that shears it), and lays it down in the
  tooth, through the dust, which it then presses in more readily. One
  firm layer of the blender over three layers of a dark coloured pencil
  on near-white soft pastel works 0.21 of the dark that lay on the dust
  into the tooth, from 0.01; drawn from such a patch out over bare paper,
  it smears 0.13 of the dark next to the edge, from 0.08, as it smears
  0.12 out of the tooth. It closes the white flecks as it did: over a
  blue pastel a quarter of them are left (0.071 to 0.018), over the paper
  a fifth (0.107 to 0.021). What the tip drags is the film's share of the
  layer it lies in, which is one mixture, as an eraser takes it. The
  medium and the wax are conserved.
- Burnishing and flattening set in by degrees (fusain-bre). A tip polished
  wax, or flattened the tooth, only where it bore harder than 0.6, and from
  nothing there: so of the strokes of a shading near that pressure, each a
  little firmer or lighter than the next, some polished and some did not,
  and the area was polished in streaks, which the gloss of wax showed (the
  navy ground of oil pastel in `examples/output/crayons.png`). The wax and
  the paper now give way at pressures spread evenly from 0.4 to 0.8
  (`BURNISH_SOFT`, 0.2 either side of `BURNISH_FROM`): a tip polishes as
  the square of how far it bears past 0.4, and from 0.8 on as it did
  (`fusain._engine.firmness`). How unevenly two crossed layers on cartridge
  paper are polished (the spread of `press` smoothed over the grain, as a
  share of its mean), at 0.55, 0.6 and 0.65: a dark coloured pencil 0.91,
  0.78 and 0.68 before, 0.63, 0.58 and 0.54 now; an oil pastel 1.34, 1.08
  and 0.87, now 0.81, 0.73 and 0.66; at full pressure as before (0.19 and
  0.17, now 0.18 and 0.15). The navy ground of `crayons.png`: 0.72, now
  0.62. A firm layer polishes a little more than it did where its lead
  bears least: the blender at 0.7 closes two fifths of the white flecks (a
  third before) and at 0.5 a twentieth (none); three firm layers of HB
  flatten the tooth by 0.74 (0.72). A light touch of 0.4 flattens a trace
  where the lead bears hardest (three layers of HB: 0.004, from 0).
- Under the default lamp a burnished dark stays deep (fusain-5je). The
  gloss of polished wax was in a lobe wider than graphite's
  (`WAX_SHININESS` 5), nine tenths of which the default lamp sent to the
  eye: four firm layers of a dark coloured pencil burnished with the
  blender went from L* 6.8 to 18.1 under it, and the firm layers alone
  from 10.3 to 17.9, following how far each stroke had polished them. A film of wax
  polished flat is smoother than one of graphite flakes, and its lobe is
  now narrower (`WAX_SHININESS` 80): its light falls to half about 15
  degrees from where the lamp is mirrored, and to a fifth at 23 degrees,
  where the default lamp is. That is assumed, not measured (a smooth
  dielectric whose facets lean about 0.16 on average, a semi-gloss
  finish): a burnished coloured pencil shines where it mirrors the lamp
  and is deep away from the glare. The burnished dark now goes to 10.3
  under the default lamp, and to 18.8 overhead (18.9 before); the firm
  layers to 12.6, and the gloss on them varies from place to place by 0.52
  L* (smoothed over the grain), from 1.51. Under a raking lamp polished
  wax no longer shines. Graphite and ink shine as they did.
  `examples/macaw.py` is rendered under the default lamp again, gloss
  included: on its burnished darks the gloss adds 1.3 L* (4.8 with the
  wide lobe), varying by 0.5 (1.7).
- A flattened patch is level in its relief, without pits (fusain-8au).
  `Sheet.surface_height` raised a flattened or polished tooth towards its
  peaks, as `press` does for a tip that comes after it, so a valley the tip
  never reached was left a pit as deep as the tooth beside a plateau: six
  firm layers of 4H on cartridge paper left 6% of the patch more than a
  quarter of the depth of the tooth below the rest, 0.85 deep on average,
  and under a raking lamp the patch looked as rough as the paper left
  alone (its lighting varied by 0.105, the paper's by 0.109). Now the peaks
  come down towards the valleys, by `press` of their height, so a tooth
  pressed flat is level with its valleys, a little below the paper around
  it: no pits, the relief varies a seventh as much as the paper's (0.029
  of the depth of the tooth, from 0.210), and a raking lamp shows it a
  seventh as much (0.015). Four firm layers of a dark coloured pencil,
  0.088, now 0.040; burnished, 0.021, now 0.003. What a tip touches of a
  flattened tooth, and so how a stroke lays over it, is as it was.
  With the narrower gloss of wax, fewer of the tooth's facets catch the
  lamp: on the firm dark the gloss varies from pixel to pixel by 1.1 L*
  under the default lamp, from 1.4, and at most adds 5.5, from 7.6.

### Examples

- `examples/macaw.py`: the light under the bill's cutting edge is
  scratched with the knife at 0.7, from 0.3. Part of that light was white
  flecks of the paper, which the blender now closes as it polishes at the
  rim of its tip; the knife makes it up. On the lower mandible within 12
  pixels of the cutting edge, the light above its dark summed 50.4 before
  (61 pixels lighter than L* 60), 43.4 with the knife at 0.3 (48), and is
  51.4 now (61).

### Changed

- Strokes and rubs are faster, by a quarter to a half, and no picture
  changes: every array of a sheet after any sequence of calls, every number
  returned and every rendered image is what it was, to the last bit. On a
  sheet of 1120 by 820 pixels (one core of a 2.1 GHz Xeon): shading it with
  an HB of width 6 takes 2.4 s, from 3.1 s; with a soft pastel of width 24,
  1.7 s, from 2.5 s; with the side of a stick of vine charcoal, 2.1 s, from
  2.9 s; rubbing it with a finger, 3.5 s, from 5.9 s; with a kneaded eraser,
  3.9 s, from 6.5 s; with a stump over graphite, 4.4 s, from 7.7 s; and
  hatching it with a ballpoint, 0.43 s, from 0.83 s. Rendering it takes
  1.3 s, from 3.2 s: most of it is Kubelka-Munk, which murex 0.2.0 works
  out in compiled code. What it took:
  - What a kernel calls for every pixel is compiled into its caller
    (`inline="always"`). Numba otherwise counts a reference to each array a
    helper is given, and back, at every call: a third of the time of a
    stroke and half of that of a rub. Two options of numba's that its
    documentation does not list (one switches that counting off, one
    compiles a helper into its caller later on) took all of that away and
    made strokes and rubs twice as fast; fusain does not rely on them. The rest is to be had by folding those helpers into their
    kernels by hand.
  - How a tip bears on the pixels of a row (its shape, the streaks of the
    pass, the level it reaches in the tooth) is worked out for the whole row
    before anything is laid on it; only the pixels of a row that the tip can
    cover are looked at (a stick on its side covers a small part of the
    square around it); the streaks of a pass are looked up in a table; the
    dust of a pass settles in one visit of each pixel; and the loops over
    the bands of a pigment are vectorised.
  - The dense points of a path are worked out once for a stroke, where
    they were worked out twice (to sample it and for its length): murex
    0.2.0 keeps them.
- A ballpoint no longer leaves a drop at every point of a gently curving
  line: what its ball brings round stays on the rim until it would show (a
  drop of `BALL_SHOWS`, 0.05, firm passes over the square of its radius, 2%
  as thick at its middle as the line), and the line has turned that far
  since the last drop. Nearly every step of a pen laid a drop nobody could
  see, at about 30% of its time. Hatching a sheet of 1120 by 820 pixels
  with a ballpoint takes 0.29 s, from 0.35 s, and over graphite 0.33 s,
  from 0.49 s. The picture changes a little: on a gentle arc the drops
  reach seven tenths of the pixels of the line, from all of them, and the
  ink of five arcs changes by a ten-thousandth; four layers of hatching are
  as dark to the third decimal. A drop also inks all of a pixel that holds
  a pass's worth of ink, so white specks in heavy hatching are no longer
  filled by drops nobody could see: in the pen drawings of
  `tools/fingerprint.py`, 0.4% of the pixels move by more than two levels
  (of 255), some by half their tone.
- fusain requires murex 0.2.0.
- `roll_stroke` (in `fusain._engine`) takes, in `pen`, how much the pen
  blots and the colour its ink bronzes, and the ink on the rim of its ball
  (`rim`, NaN for a pen just picked up); `blot` takes the pen.
- `lay_stroke` (in `fusain._engine`) takes in `medium` whether its side
  lies flat; the `now` of `bear` goes on with whether the tip is the side
  of a stick laid flat and how fast the stroke turns, its `row` has a
  fourth row, `set_down` takes `now`, `tip_at` whether the side lies flat,
  and `burnish` the two layers and `bonded`.
- `lay_stroke` and `rub_stroke` (in `fusain._engine`) take what they took;
  `bear`, `put_down` and `streak_at` take the streaks of the pass
  (`streaks_of`), and the first two the row they work in.

### Added

- `Medium.flat_side`: whether the side of a stick of the medium lies flat
  along its length, so that its band is even across (charcoal, pastels and
  crayons), or is the cone of lead a sharpening bared, whose band is soft
  at its edges (pencils, and the default).
- `HeldPen(pen)`: a ballpoint in the hand, as a `Stick` is a medium and a
  `Smudger` a tool in the hand. It keeps the ink its ball gathers from one
  line to the next until `wipe()` (which returns it); `load` is how much
  there is. It lands with what its last line gathered and did not leave:
  where it lands a share drawn at random comes round, as for a pen just
  picked up (mostly next to none), and the rest at the first sharp turn.
  A bare `Pen` is picked up afresh for every line, as before: it lands
  with a drop drawn at random, which it leaves there, and takes away what
  it gathers. `Sheet.stroke` and `Sheet.shade` take either, and return the
  ink laid; what a `HeldPen` gathered is on the paper or on its ball.
  Measured: a pen in the hand that drew a straight line of 6.53 of ink
  holds 0.327 (`BALL_GATHER`, 0.05, of it), and where the next line lands
  at 0.7 with a share of 0.52 coming round it leaves 0.1272 more than a pen
  that gathers nothing, 0.745 of 0.52 of 0.327, to four digits.
  `tools/fingerprint.py` draws one.
- `Pen.bronze`, and `pens.ballpoint(..., bronze=...)`: the colour heavy ink
  mirrors under a lamp. By default the light the ink absorbs most, as
  before; a black ink is a blend of dyes, which bronzes as the one that
  bronzes most (see Fixed). `Pen.mirrors` is that light in linear RGB.
- `Pen.blots`: how much ink a ballpoint's ball gathers, to leave as a blob
  where the line turns sharply and, now and then, where it lands. 1 for a
  usual pen (the presets), 0 for one that never blots, more for a leaky
  one. It scales the drop the ball can hold and what it gathers, and leaves
  the line as it is: `Pen.flow` was the only control, and it darkens the
  line with the blobs.
- `tools/fingerprint.py`: whether a change has altered the picture. It
  draws a suite of small drawings (every family of medium, sticks and tilt,
  width profiles and flat ends, shading, tools bare and in the hand, a
  knife, a blender, fixative, a stylus, ballpoint pens, on several papers)
  and hashes every array that the sheets, the sticks and the tools in the hand hold,
  the numbers the calls returned and the rendered images.
  `tools/fingerprint.json` holds the hashes of the code as it was before it
  was made faster, and `tests/test_fingerprint.py` checks them (they are
  skipped, with the reason, on a machine whose `pow` or numpy rounds
  differently from the one they were recorded on). `--thorough` draws the
  suite on every paper with three seeds, to save before a change and compare
  after; `--record` is for a change that is meant to alter the picture.
- `tools/benchmark.py`: the timings above.

### Added

- `tools.KNIFE` and `tools.SAND_ERASER`, and `Tool.scrape` (0 to 1): how
  far a tool cuts through wax, which holds against an eraser (see Fixed).
  The point of a knife scratches a line two pixels wide out of a dense or
  burnished layer of coloured pencil, crayon or oil pastel, back to the
  paper (sgraffito): one pass at 0.9 takes all but a hundredth of what lay
  on its line, wax included, so the line takes pencil again, and leaves
  what lies two pixels to either side alone. A sand eraser takes nine
  tenths of a burnished dark in one rub. What they take off is on the tool
  and returned by `Sheet.erase`, as for any eraser.
- `Medium.melts` (0 to 1): how far a medium melts into a film of wax it is
  pressed hard on, and stirs it. 1 for `pencils.BLENDER`, 0 for every other
  preset (see Fixed).

### Fixed

- A ballpoint line is as strong at its ends as in its middle. Where the
  ball landed and lifted, the paper under it had only the passes of the
  path's first and last points, so a line faded over the width of the
  ball, from half strength at the ends of its path, and a dot had a tenth
  of the ink of a line. The paper the ball meets where it lands and where
  it lifts (where its surface comes down to the peaks of the tooth: all of
  it pressed firmly, a narrower disc pressed lightly) now gets what it
  would have had if the ball had rolled onto it and off it, as a stick's
  does (`put_down`). Measured on a line of `pens.BLUE` that never blots,
  the ink of each column of pixels against the middle's: half a pixel
  behind where it lands, 0.28 at a pressure of 0.2 and 0.31 at 1.0, now
  1.04 and 1.03; half a pixel past where it lifts, 0.28 and 0.29, now
  1.03 and 0.98; a pixel and a half behind, 0.01 and 0.03, now 1.05 and
  1.04 (the round of the ball, as wide as the line). A dot is 0.91 to 1.0
  as dark as a line, from 0.09. Hatching a sheet with a ballpoint takes
  0.32 s, from 0.29 s.
- Black ballpoint ink bronzes a red between copper and violet, as a real
  one does, where it bronzed grey: its ink was one neutral colour, which
  absorbs all light alike, and so mirrors it alike. `pens.BLACK` is a blend
  of dyes whose violet bronzes. Eight layers of it hatched at 0.7, under a
  lamp overhead, add (0.0221, 0.0072, 0.0106) of light in linear RGB, from
  (0.0221, 0.0221, 0.0199): a chroma of 41 at a hue of 1 degree, from 6.
  The sheet keeps for each pixel the colour of its ink's bronze, mixed as
  the inks are (`Sheet.ink` has three more parts, after the volume: linear
  RGB, times the volume); blue and red ink bronze as they did.
- A light ballpoint line blots as lightly as it is drawn. The ball left all
  the ink it had gathered where it landed and where the line turned, so a
  faint line had blobs as big and as dark as a firm one's, and they showed
  on the light side of a hatching. The paper now takes of that drop as much
  as the ball bears on it, the share of a firm pass the line lays there
  (`BALL_LIGHT`, 0.15, at the lightest touch, all of it at full pressure);
  the rest stays on the rim, for the next turn. Measured on a pen that
  lands with a drop, the ink within five pixels of where the line lands,
  more than at its end: 3.5 times as much again at a pressure of 0.15, 1.4
  times at 0.5 and 0.86 at 0.9; now 0.98, 0.82 and 0.79. Over eighty lines,
  the darkest start is 1.87 times as dark as its line at 0.15 (2.60
  before), 1.49 at 0.5 (1.63) and 1.36 at 0.9 (1.38), and a fifth of the
  starts of the faintest lines are a third darker than their line (a third
  of them before; at 0.9 hardly any, as before). A square corner after 140
  pixels at 0.5 has 1.38 times the ink of the line beyond it, from 1.58; at
  full pressure blobs are as they were.
- The blender shows. `pencils.BLENDER` was a coloured pencil without
  colour, and pressed hard it made little difference: one firm layer of it
  over three layers of red at 0.5 on cartridge paper left two fifths of the
  white flecks, and in `examples/macaw.py`, on a paper with a shallow tooth,
  half of them. A blender is soft clear wax and nothing else, and it melts
  into the film it is pressed on: `Medium.melts` (0 to 1; 1 for the blender,
  0 for every other preset) says how far. A tip that melts polishes a film
  of wax however thin, takes up two and a half times as much of what is
  under it (`MELT_LIFT` 1.5, against `BURNISH_LIFT` 0.6) and lays it back a
  little sooner (`MELT_LAY` 6, against `BURNISH_LAY` 4). One firm layer now
  leaves a fifth of the flecks (2.0% of the pixels, from 10.7%) and two a
  twenty-fifth; the chroma of the patch goes from 34 to 59 and to 69
  (before, to 49 and to 60), its lightness from 67 to 57 and to 54, it is
  half as grainy, and it has 0.46 and 0.79 of the gloss of polished wax. On
  the macaw's plumage one layer takes the flecks from 7.3% to 1.1% (before,
  to 3.4%) and the chroma from 47 to 57. At a pressure of 0.7 the blender
  closes a third of the flecks; at 0.5 it does nothing. It stirs what it
  goes over: yellow and blue laid over each other become one green, and
  drawn from a dark out over bare paper it drags the dark along (a quarter
  of its pigment next to the edge, a thirteenth a tip's width out, under a
  hundredth three widths out). Nothing is lost: what the blender lays and
  what it drags are on the sheet. A coloured pencil or a crayon pressed
  hard burnishes as it did.
- Wax holds against an eraser. An eraser took a coloured pencil off as it
  takes graphite: one rub to and fro of a vinyl eraser took over nine tenths
  of a dense, burnished dark, and a kneaded one a third. (The dab reported in
  the issue lifted a third and showed no light only because a stroke
  less than a pixel long is a sixth of a pass.) Of what the paper holds
  where there is wax, an eraser now lifts less: `WAX_HOLDS` (0.6) less
  where the wax is `WAX_FAST` (0.1) thick or more, in proportion where it is
  thinner, and `POLISH_HOLDS` (0.9) less where it has been polished. One rub
  of a vinyl eraser at 0.9 takes 92% of one light layer of a dark coloured
  pencil, two thirds of three layers at 0.5, and a third of four firm
  layers, burnished or not; after three rubs three tenths of the firm layers
  are left, at a fifth of the paper's tone, and two fifths of the burnished
  ones, as dark as before. A kneaded eraser takes a quarter of the light
  layer in a rub and 7% of the burnished ones. Graphite, charcoal and
  pastel come off as they did. A vinyl eraser now takes half of an oil
  pastel in a rub, where it took over 0.7: a knife scrapes it off.

### Examples

- `examples/macaw.py`: the feathers, the bill and the brow are burnished
  with the blender, along the lie of the feathers and the arch of the horn
  (4,250 strokes of it after the 32,500 of the pencils), all but the end of
  the tail. The white flecks are gone from the plumage and its colour is
  deeper. No eraser takes a light out of that, so the lights are scratched
  with the knife: the edges of the flight feathers, the thin light under
  the bill's cutting edge, and the catchlight, which now sits on the edge of
  the pupil. The small dark touches under the feathers' tips go on before
  the burnishing; no feather is drawn across the eye; the image is rendered
  without gloss, as a scanner sees it (under a lamp the burnished wax greys
  the darks).
- `examples/colour.py`: the red square, half burnished, is crossed by three
  lights: a line indented before the red was laid, a vinyl eraser rubbed to
  and fro (it clears the half that kept its tooth and slides on the
  burnished one) and two lines scratched with the knife.
- Every image is regenerated. `colour.png` and `macaw.png` change as said;
  `shine.png` where its coloured pencil is burnished; `crayons.png` where a
  kneaded eraser and a finger go over oil pastel (by a few levels). The
  others are unchanged.

### Changed

- With the blender burnishing more, a red burnished with it has 0.78 of the
  gloss of polished wax (from 0.70), and goes from (222, 83, 56) to
  (225, 90, 67) under the default lamp.
- `tools.ERASER` and `tools.KNEADED` take less of a coloured pencil, a
  crayon or an oil pastel than they did (see Fixed).
- `rub_stroke` (in `fusain._engine`) takes the sheet's `press` after
  `cover` and the tool's `scrape` after `smear`; `lay_stroke` takes a
  medium of six numbers, the last how far it melts.

### Added

- `Sheet.indent(path, pressure=0.7, width=3.0)`: a stylus, a hard round
  point that lays nothing and presses a groove into the paper, and
  `Sheet.dent`, the depth of the groove at each pixel (below the peaks of
  the tooth, in depths of the tooth of a paper of depth 1). A pencil, a
  crayon or charcoal shaded over a groove rides on the paper either side and
  marks only what it reaches down to, so a line indented firmly stays the
  colour of the paper in a tone (whiskers, grass, fine highlights), and one
  indented lightly fills in under a soft pencil pressed hard: of what a 2B
  shaded at 0.5 lays on the paper beside it, it lays 64% in a groove
  indented at 0.3, 2% at 0.5 and nothing at 0.7; a 6B at full pressure 98%,
  68% and 12%. The groove is deeper on a paper that gives, shows as a line
  of shadow under a raking light, and no eraser takes it out.
- A ballpoint presses a groove too, the deeper the firmer the pen: a 2B
  shaded lightly over a pen line lays in it 97% of what it lays beside it
  if the pen was light (0.2), 58% at 0.6 and 3% at full pressure. The ball
  itself goes down into a groove, so pen lines cross as before.
- A ballpoint leaves blobs. The ball gathers a twentieth of the ink it lays
  on its rim, and a turn brings it round onto the paper: nearly all of it
  where the line turns back on itself, about two thirds at a square corner (after 140
  pixels of line, half as much ink again as on the line beyond the corner),
  a little at a time on a gentle curve, where it does not show. A pen put to
  the paper leaves what it had on it where it lands: mostly next to
  nothing, so that one line in four or so starts with a dot that shows and a
  few with a blob. A blob is a round dot a little wider than the line. It is
  ink the pen lost like the rest: `Sheet.stroke` returns it with the line.
- `Sheet.shine(of)` says how much of each pixel shines in each of three
  ways, `"graphite"`, `"wax"` or `"ink"` (all three together by default),
  and `Sheet.render` shows all three under the lamp; `gloss` scales them
  together.
- Heavy ballpoint ink bronzes under a lamp: from one and a half firm passes
  thick, fully at four and a half (eight layers of firm hatching bronze
  half-way, a single line not at all), in the colour the ink absorbs most:
  blue ink shines copper, red ink green, black ink grey. Its light is
  narrow: it shows with the lamp overhead, a tenth as much under the default
  lamp, and not under a raking one. What is drawn over the ink hides it.

### Fixed

- A firm hard pencil flattens the tooth. `Sheet.press` was raised only by
  burnishing wax. A dry tip pressed harder than 0.6 now flattens the tooth
  itself, the more the harder it is (as the cube of its hardness) and the
  better it holds together (as the square of its binding): three firm
  layers flatten it by 0.88 under a 4H, 0.72 under an HB, 0.36 under a 2B
  and 0.03 under a 6B; compressed charcoal and hard pastel by a few
  hundredths, vine charcoal and soft pastel next to not at all. A flattened
  tooth is touched all over, and bites a dry medium less (seven tenths less
  in a pass): a light layer of a soft pencil over a patch a hard one
  flattened lies on four fifths of it where it marks half of a tooth left
  alone, and half as unevenly, for about as much graphite. And graphite
  lies flat on it: a thinner film shines as much, so three firm layers of
  4H shine 2.5 times as much as the same graphite would on a tooth that had
  kept its shape, and six of HB a third as unevenly. Tones change little:
  one layer of HB at full pressure reflects 0.457 of the paper's light, from
  0.465, and of 4H 0.864, from 0.862.
- Polished wax shines. Only graphite had sheen. Wax now has a gloss, which
  is not the medium's but its polish's: how slick the film of wax is times
  the square of how far it has been pressed flat. A coloured pencil laid
  lightly has none; burnished with a blender it has seven tenths of the
  most; two layers of crayon go from none at a pressure of 0.3 to 0.02, 0.35
  and 0.75 at 0.6, 0.8 and full pressure. It counts wherever the wax lies (a film
  laid over dust shines; what the paper holds shines through whatever dust
  lies over it), and dust laid over it dulls it. Under the lamp it is white
  light, less than half as strong as graphite's and wider: a soft gloss,
  which a raking lamp still shows. Under the default lamp a burnished red
  goes from (223, 88, 60) to (225, 95, 72), and the black outlines of the
  crayon farm are about five levels lighter.

### Examples

- `examples/shine.py` shows all that shines, under its three lamps: a row is
  added with a 2H laid lightly and pressed hard, a coloured pencil half
  burnished, a crayon and an oil pastel pressed hard, and three ballpoints
  hatched eight times over. `examples/ballpoint.py` has a row for blobs (a
  zigzag and a scribble to and fro), for the groove of a pen under a soft
  pencil, and for loops written with a pen that has run dry and shaded over.
  `examples/pencils.py` has a row for the surface: indented lines under a
  light and a firm tone, blades of grass kept white, and a tone over a tooth
  a 2H flattened.
- Every image is regenerated but `broadway.png` and `deco.png`. The
  ballpoint still life has a dot here and there where a line starts, and
  the faint relief of its grooves; the crayon farm and the crayons of
  `crayons.png` and `colour.png` have the soft gloss of pressed wax; the
  lake in graphite changes where a hard lead was pressed firmly;
  `papers.png` is unchanged, and `charcoal.png`, `fixative.png`,
  `strokes.png`, `tips.png` and `macaw.png` nearly so.

### Changed

- `Sheet.shine()` is now what shines in any of the three ways; what it was
  is `Sheet.shine("graphite")`, which also counts how flat the tooth is
  pressed. A `Sheet` takes about 0.37 KB per pixel, from 0.36.
- `lay_stroke` and `roll_stroke` (in `fusain._engine`) take `dent` after
  `press`, and `roll_stroke` writes to it.

### Added

- The side of a stick is as long as the stick. A `Medium` says how many
  times its point the side of a stick of it marks held flat (`side`: 5 for
  pencils, as before, and for soft pastel; 6 for compressed charcoal and
  hard pastel; 8 for vine charcoal), and a `Stick` can give the length of
  its side in pixels (`Stick(charcoal.VINE, width=6, lead=6, side=90)` lays
  a band 90 pixels wide at `tilt=1`). By default a stick's side is its
  medium's `side` times its sharp point, and no longer grows as the point
  wears; part way over, a stick marks its tip and that share of what the
  side adds. `repr(Stick)` shows the side.
- A stroke's width and tilt can change along it, as its pressure can: one
  value per point of the path, or a function of u
  (`sheet.stroke(path, pencils.B2, width=lambda u: 8 - 6 * u)`), for a
  stroke that tapers or recedes in perspective, or a stick turned onto its
  side as it goes. What the tip lays on the paper under it does not change
  with its width.
- `Sheet.stroke(ends=...)` and `Sheet.shade(ends=...)`: `"round"` (the
  default), in the outline of the tip, or `"flat"`, cut square across at
  the ends of the path, so that a stroke is as long as its path; or a pair,
  one for each end. A cut applies only within the tip's reach of its end,
  so a path that comes back across the line through one of its ends is not
  cut there.

### Fixed

- A stroke is as strong at its ends as in its middle. The tip laid only as
  it slid on, so where it landed the paper under the back half of it got
  less, from nothing at its edge, and the same where it lifted: the ends of
  a stroke faded out over a whole width of the tip, the end of its path got
  half the tone of its middle, and a short stroke of a broad tip (a head, a
  hand) reached a third of it and a dot a fifth. Now the paper the tip
  covers where it lands and where it lifts is laid as though it had slid
  onto it and off it, and the ends of a stroke are the outline of its tip at
  the tone of the stroke. Where it lands and lifts the tip goes at the pace a tip
  of its width is sampled at, however short the stroke, so a stroke of
  almost no length costs and lays what a dot does. The strokes of `Sheet.shade` end half a tip short
  of the edge of the area when their ends are round, so that a shading stays
  inside it (it used to spill out by half a tip, faintly).

### Examples

- `examples/strokes.py` (`strokes.png`): round and flat ends, short dabs,
  strokes that recede and taper, and the side of a stick of charcoal and of
  pastel.
- Every image is regenerated. Shadings end crisply at the edge of their
  area. `broadway.py` cuts its fronts, awnings and the bars of the crossing
  square at their ends instead of insetting round ends by half a width
  (a car's pieces keep round ends inside them, so that it does not come out
  in blocks), narrows every stroke that runs into the distance as it goes,
  and draws its figures in one layer (two, laid opposite ways with a fix
  between, made up for the weak ends of short strokes). Its leaves are
  shaped by profiles: each dab of foliage swells from nothing where the
  stick lands to its middle and back (pressure and width), and a fallen
  leaf narrows toward its tips, so that the trees stay soft now that a
  stroke is full to its ends. `deco.py` stops its windows square, and a
  stroke that the edge of a face cuts short stops square at the edge, so
  that its round end no longer greys the lit face next to it.
  `stilllife.png` (ballpoint) and `papers.png` are unchanged.

### Changed

- `lay_stroke` (in `fusain._engine`) takes the width, tilt and side of the
  tip at each point, and the spacing of the points and how the stroke ends.
  Every picture changes where strokes end; see the examples.
- `Sheet.rub` carries what the tool picks up from one pass to the next, and
  over the stretches where the pressure is nothing and the tool is lifted.
  Each pass used to start clean and take its load away, which lifted a pale
  dot where every pass landed (dotted rows along the edges of a rubbed area)
  and wore a tone away: eight rubs of a stump over a firm layer of 8B took
  four tenths of the graphite off the sheet, and now take under three
  hundredths. A stump lifted over a gap now lands beyond it with the dust it
  carries. Pictures that rub an area change: `examples/lake.py` no longer
  presses its tones several times too hard to make up for what the rubbing
  took (it presses for the darkening wanted, by a measured table), and its
  ridges, water and reflections come out even; `examples/broadway.py` pulls
  its haze down from the sky with a finger wiped before each pull.
- `Sheet.smudge`, `Sheet.erase` and `Sheet.rub` return what the tool took
  off the sheet less what it laid back, which is less than nothing for a
  loaded `Smudger` that leaves more than it takes. For a bare tool that is
  what they returned before.

- What lies over the paper is no longer all dust. The two layers that lie
  over what the paper holds (`Sheet.under`, `Sheet.loose`) also take the film
  a waxy medium lays on dust and the paste of an oil pastel, and
  `Sheet.bonded` holds how much of each layer is not loose dust, of three
  kinds, each with a view of its own: `Sheet.fixed` (dust that is set, by
  fixative as before, or under a film), `Sheet.film` and `Sheet.paste`. A
  sheet takes about 0.36 KB per pixel, from 0.34. `Sheet.fix` sets paste as
  it sets dust. Pictures change little: graphite, charcoal, soft pastel and
  crayon on bare paper draw as before (ten of the fifteen example images are
  unchanged to the byte, `lake.png` but for a rounding in a few pixels); the
  layers of coloured pencil in `colour.png` and `macaw.png` shift by a few
  levels in places, a part of each layer now lying over the crumbs of the one
  before; and the oil pastels of `crayons.png` cover a little more evenly.
- A dry medium takes no better over a thin film of wax than over a thick
  one: to charcoal, pastel, graphite and a ballpoint, wax is as slick as it
  gets once it covers the tooth (a film 0.2 thick, from 0.35). A waxy medium
  over wax builds as before. The ballpoint lines across the patch of crayon
  in `ballpoint.png` skid a little more.

### Fixed

- A waxy medium laid over dust lies on it. What the paper holds of a medium
  was always rendered under the dust, so a coloured pencil or a crayon drawn
  over charcoal or pastel showed under it: a yellow crayon shaded over vine
  charcoal darkened it (luminance 0.48 to 0.38) where it now lightens it and
  turns it yellow (0.53), and over blue soft pastel the patch stayed blue
  (its red and green over its blue, in linear light: -0.25, now -0.01, the
  crayon taking badly over a full tooth). The film of a waxy medium now lies
  on the dust it meets, as far as the dust covers the paper, and binds part
  of it: a kneaded eraser rubbed over that crayon on charcoal lifts 1650
  units where it lifted 2260, and a finger carries off 18 where it carried
  30. A dry medium is held in the tooth, under the dust, as before.
- A kneaded eraser no longer lifts oil pastel as it lifts dust. What the
  paper does not hold of an oil pastel is paste: a finger, a stump or a
  chamois still smears it (and it is paste again where they lay it), but an
  eraser lifts it only as it lifts what the paper holds. A kneaded eraser
  rubbed over a layer of oil pastel took 53% of it and now takes 19%, as of
  a wax crayon, against 69% of vine charcoal; a vinyl eraser still takes
  80%. Dust that an oil pastel is laid over turns to paste with it.
- Charcoal shaded over one firm line of wax crayon skips it (wax resist).
  The line used to take four tenths of what the paper around it took and,
  being pressed level, to come out as dark as the shaded paper (luminance
  0.63 to 0.49, the paper around 0.50); it now takes under a tenth and keeps
  its tone (0.59). Three firm layers were needed before.
- Soft graphite smudges. A stump, a finger or a chamois hardly changed a
  pencil's tone, whatever the lead: only the little dust a pencil leaves
  moved. Graphite now slides under a tool that smears: what the paper holds
  is pushed from grain to grain and a little comes loose, the more graphite
  the lead has, so a rubbed 4B or 8B turns smooth and can be dragged out onto
  bare paper, an HB evens a little, and a hard lead hardly moves. Graphite
  under set fixative stays put, and no other medium slides. Pictures that rub
  a pencil change (`examples/output/charcoal.png`).
- Sanded card takes many more layers of pastel than paper, as it does in
  life: a `Paper` has a `hold` (how much its tooth holds for its depth), and
  `papers.SANDED` holds two and a half times what a paper of its depth does.
- Rubbing evens the dust out. A stump or a finger now spreads the dust under
  it from grain to grain before it is worked into the paper, so a rubbed
  tone is smooth: charcoal laid with the side of a stick no longer stays
  speckled, streaks between light strokes close, and a stump smooths
  compressed charcoal. It pressed the dust in where it lay before.
- A firm pencil no longer wears a waxed layer away. Pressed hard over wax, a
  tip drags part of the layer along; what it still carried when the stroke
  ended left the paper with it, so many short firm strokes stripped the
  layer. It is now laid where the tip lifts off; only what is dragged off the
  sheet is lost.

### Added

- A third row in `examples/crayons.py`, what lies over what: a sun and
  waves in one firm line of crayon each with charcoal shaded over them (wax
  resist), crayon lines over rubbed charcoal and over soft pastel with a
  stump dragged through, and a kneaded eraser pulled through soft pastel and
  through oil pastel.
- A pebbled paper: `Paper(texture="pebbled")`, and the preset
  `papers.CHARCOAL`, a cream charcoal paper. Its tooth is small irregular
  pits, each with its own depth and outline, in a net of rims: between a wove
  paper, on which a tone enlarged is fine flecks, and a rough one, which at a
  small grain is blotchy. A light tone of charcoal on it is a net of dark
  around pale pits; meant for a grain of 3 to 5 pixels. The other textures and
  papers draw as before.
- `Smudger(tool, width=None)`: a stump, a finger or a chamois in the hand,
  as a `Stick` is a medium in the hand, for `Sheet.smudge`, `Sheet.erase`
  and `Sheet.rub`. It keeps what it picks up from one pass to the next
  (`load`, `carried`) until `clean()`: once loaded it gives back about as
  much as it takes, so a tone can be rubbed over and over without lightening
  (200 drags of a finger through a dark tone left it as dark; wiped clean
  before each, they took it from 64 to 107 out of 255), and a stump loaded
  on a dark patch draws a pale tone on bare paper, which a bare tool, clean
  every time, could not. It has its own width, as a `Stick` has. The
  examples that rub to and fro hold their finger or stump this way
  (`charcoal.py`, which now also draws with the loaded stump, `colour.py`,
  `crayons.py`, `fixative.py`, `lake.py`).
- Four drawings as examples, each a script and the image it writes: an art
  deco tower in charcoal (`examples/deco.py`), Broadway in SoHo in pastel
  (`examples/broadway.py`), a great green macaw in coloured pencil
  (`examples/macaw.py`) and a still life in three ballpoints
  (`examples/stilllife.py`).
- Two more: morning mist on a mountain lake in graphite
  (`examples/lake.py`) and a naive farm in wax crayon (`examples/farm.py`).
- `Sheet.rub(area, tool, ...)`: rubs or erases a whole area, in passes side
  by side, to and fro. It is to `smudge` and `erase` what `shade` is to
  `stroke`: the pressure may be a function of (x, y), and where it is nothing
  the tool is lifted.
- Ballpoint pens, `fusain.pens` (`pens.BLUE`, `pens.BLACK`, `pens.RED`,
  `pens.ballpoint(colour, width)`, `Pen`), for `Sheet.stroke` and
  `Sheet.shade`. A line is even and continuous at any pressure, paler and
  narrower when light; the ink is clear, so crossings and layers of hatching
  darken. It is a stain in the paper (`Sheet.ink`, `Sheet.inked`), under
  everything else: no eraser takes it off, and a pencil draws over it. The
  ball skids on heavy graphite and on wax; where it lays nothing, it leaves
  the ink under it as it was.
- Fixative, `Sheet.fix(amount=1.0)`: sets the dust where it lies. A stump or
  a finger hardly moves it afterwards and a kneaded eraser lifts little; a
  hard eraser still takes it off. It shows as it lay, a little darker, and
  takes more layers, which are loose again: rubbing them leaves the fixed
  layer underneath alone, and a second coat darkens only what it newly sets.
  `Sheet.fixed` is the volume of dust set at each pixel, in each of the two
  layers of dust.
- Graphite shines. `Medium.sheen` is how much of a medium mirrors the light
  (graphite only, the more the softer the lead), `Sheet.shine()` is how much
  of each pixel does, and `Sheet.render` adds the lamp's reflection: heavy
  graphite turns silvery under a high lamp and stays dark under a raking one,
  where charcoal stays black. `render(gloss=0)` renders matte.

### Changed

- A drawing with graphite in it renders a little lighter in its darks under
  the default lamp, from the shine. `gloss=0` gives the earlier picture.
- The pigment layers of a `Sheet` have one more component, the sheen, before
  the wax of the held layer (`fusain.sheet.IX`).

- Wax crayons, `crayons.wax(colour)`: all held, translucent and waxy. The
  grain of the paper shows until one is pressed hard, layers mix, and a light
  crayon does not cover a dark one.
- Oil pastels, `pastels.oil(colour)`: greasy and thick. A finger smears
  them, less than soft pastel, and a light one half covers a dark one.
- What a film of wax takes depends on what is laid over it: a waxy pencil or
  crayon takes a little, as before; a greasy oil pastel sticks to it; and a
  dry medium (graphite, charcoal, soft pastel) hardly takes at all, where it
  took as much as a waxy one before.
- `Stick`: a medium in the hand, with a tip that wears. A point wears to a
  flat as it draws, so its line widens, the sooner the softer the lead;
  `sharpen()` gives the point back. `Sheet.stroke` and `Sheet.shade` take a
  stick in place of a medium.
- Coloured pencils, `pencils.coloured(colour)`, and `pencils.BLENDER`:
  translucent and waxy. Layers of two colours mix to the eye; the tooth turns
  slick as it fills; a firm pass polishes the wax and closes the white specks
  (burnishing). `Medium.wax` is the share of wax in a stick.
- Pastels, `pastels.soft(colour)` and `pastels.hard(colour)`: opaque, and
  nearly all dust. A light pastel covers a dark one, and a finger blends them.
- Dust lies in two layers, `Sheet.loose` over `Sheet.under`: a pass lays its
  dust on top of the dust that was there. Rubbing stirs the two together.
- Sanded card has an even grit and takes pastel all over.
- A burnished patch is level under the lamp, as it is under a tip.
- `tilt`, for `Sheet.stroke` and `Sheet.shade`: a stick held on its side marks
  a band up to five times wider than its point, lighter, grainier and soft at
  the edges. The tip is now an ellipse, as long along the stroke as the point
  is wide.
- Charcoal, `fusain.charcoal`: vine, compressed and charcoal pencil. Most of
  what it lays is dust.
- `Sheet.smudge` and `Sheet.erase`, with the tools of `fusain.tools`: stump,
  tortillon, finger and chamois drag the dust along, rub it in and grey the
  paper; kneaded and vinyl erasers take pigment off. `Tool` describes one.
- Going over dust with a tip presses some of it into the paper.
- Laid paper: softer laid lines, and chain lines that show pale, as furrows.
- `Sheet.shade`: shades a rectangle or a polygon with strokes side by side as
  a hand lays them, none quite straight or quite alike; the pressure can
  change across the area.
- Graphite pencils, `fusain.pencils`, from 9H to 9B, and `Sheet.stroke`: the
  tone depends on the grade, the pressure and the number of layers; a light
  touch marks only the peaks of the tooth, and the tooth fills up.
- A line has the grain of the paper inside it, ragged edges and fine streaks
  along it: the tip is flat across, bears hardest on the high ground, and only
  the softest leads pressed hard reach every valley.
- Wove papers have a fine, even tooth, about `grain` pixels from bump to
  bump, so that shading no longer comes out in blotches wider than the pencil.
- `Medium`: what a stick is made of (pigment, softness, binding).
- `Paper.give`: how much the surface yields under a tip.
- `Paper` and the presets of `fusain.papers`: wove, laid, rough and sanded
  textures, with the size, depth and bite of their tooth.
- `Sheet`: a sheet of paper holding its tooth and the pigment held on it or
  lying loose, rendered by Kubelka-Munk over the paper, flat or under a lamp.
- Project scaffolding: packaging, formatting, linting, type checking, tests,
  spelling, coverage, CI and Dependabot.

[Unreleased]: https://github.com/joce/fusain/commits/main
