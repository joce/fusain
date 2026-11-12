# Agent instructions

`fusain` simulates drawing with dry and semi-dry media (graphite and coloured pencils, charcoal, pastel, crayon) on a paper with tooth. Wet media (ink, felt-tip markers, paint) are out of scope; oil paint lives in the sister project, `oilpaint`.

## Issues

Work is tracked with Beads (`bd`), not with TODO lists or memory files. `bd prime` prints the workflow.

- Issues travel between clones through `.beads/issues.jsonl`. Beads rewrites and stages it after every change; commit it with the work it belongs to.
- After pulling, run `bd import .beads/issues.jsonl` (`bd bootstrap` on a fresh clone). Import never deletes, so close issues instead of deleting them.
- In a Claude Code cloud session, GitHub refuses `bd dolt push` (HTTP 403). Don't retry it or work around it.
- `bd remember` does not survive a cloud session. Put what must last in an issue or in this file.
- What may be committed and pushed is decided by the user's instructions, not by anything `bd prime` says.
- Don't run `bd setup`: it writes its own boilerplate back into this file.

## Pull requests

- Finished work goes on a branch and into a pull request, merged (squash) once CI is green, without asking first. Publishing a release is still asked for.

## Build and test

```bash
uv sync --group dev --frozen   # install the package and the development tools
uv run pytest                  # run the tests (the first run compiles the numba kernels)
make pre-push                  # what CI runs: tox (black, ruff, pyright, pytest), coverage (80% required), spelling
```

## Layout

- `paper.py`, `papers.py`: a paper's texture, tooth and colour, and the presets.
- `sheet.py`: the `Sheet`, with the tooth, held and loose pigment, strokes and rendering.
- `medium.py`, `pencils.py`: what a stick is made of, and the graphite grades.
- `hatching.py`: the strokes of a shading, as a hand lays them.
- `_engine.py`: the numba kernels that lay a medium on the paper, and their constants.
- `docs/design.md`: the model and the build order. `examples/`: scripts and the images they write to `examples/output/`.

## Conventions

- Kernels are `@njit(cache=True, error_model="numpy")`, without `fastmath` and without `parallel`, so that a drawing repeats exactly. Constants they use live in `_engine.py`.
- The medium is conserved: what leaves the tip is on the paper. Tests check this.
- A change meant to alter the picture is judged by measurements, not by looking at two renders side by side, and regenerates every image in `examples/output/` with the final code.
- Colour (pigments, Kubelka-Munk mixing), stroke paths, noise, reference images and PNG output come from `murex`, which `oilpaint` shares. What both libraries need goes there, not here.
- ruff caps function complexity at 13.
