# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
