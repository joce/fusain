# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

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

- Four drawings as examples, each a script and the image it writes: an art
  deco tower in charcoal (`examples/deco.py`), Broadway in SoHo in pastel
  (`examples/broadway.py`), a great green macaw in coloured pencil
  (`examples/macaw.py`) and a still life in three ballpoints
  (`examples/stilllife.py`).
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
