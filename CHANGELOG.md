# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Graphite pencils, `fusain.pencils`, from 9H to 9B, and `Sheet.stroke`: the
  tone depends on the grade, the pressure and the number of layers; a light
  touch marks only the peaks of the tooth, and the tooth fills up.
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
