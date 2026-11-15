# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- `lay_stroke` and `rub_stroke` (in `fusain._engine`) take what they took;
  `bear`, `put_down` and `streak_at` take the streaks of the pass
  (`streaks_of`), and the first two the row they work in.

### Added

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
