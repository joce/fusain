# Agent instructions

`fusain` simulates drawing with dry and semi-dry media (graphite and coloured pencils, charcoal, pastel, crayon, ballpoint pen) on a paper with tooth. Wet media (ink, felt-tip markers, paint) are out of scope; oil paint lives in the sister project, `oilpaint`.

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
uv run pytest -n auto          # the tests on every core
make pre-push                  # before a push: black, ruff, pyright, the tests once under coverage (80% required), spelling
make tox                       # every Python version and the built wheel, as CI runs them
uv run python tools/fingerprint.py   # whether the picture has changed: hashes of a suite of drawings
uv run python tools/benchmark.py     # how long strokes, rubs and renders take, in processor time; --against CHECKOUT compares with another checkout, run by run
uv run python tools/refcounts.py     # whether a kernel counts references as it works (it must not)
```

## Layout

- `paper.py`, `papers.py`: a paper's texture, tooth and colour, and the presets.
- `sheet.py`: the `Sheet`, with the tooth, how far the paper has been crushed (`crush`) and the wax on it polished (`polish`: `press` is what a tip meets of the two) and how far it has been indented (`dent`), the pigment it holds and the two layers that lie over it (dust, and what a binder holds there: `bonded`), strokes, a stylus, rubbing and erasing, fixative, what shines, and rendering.
- `medium.py`, `pencils.py`, `charcoal.py`, `pastels.py`, `crayons.py`: what a stick is made of; graphite and coloured pencils, charcoals, soft, hard and oil pastels, wax crayons.
- `stick.py`: a medium in the hand, with a tip that wears. `pens.py`: ballpoint pens, whose ink is a stain in the paper, and a pen in the hand (`HeldPen`), which keeps the ink its ball gathers.
- `tools.py`: stumps, fingers and erasers, which lay nothing of their own, and what scrapes wax off: a sand eraser and a knife. `smudger.py`: a tool in the hand, which keeps what it picks up.
- `hatching.py`: the strokes of a shading, as a hand lays them.
- `_engine.py`: the numba kernels that lay a medium on the paper, those a render gathers its layers with, and their constants.
- `docs/design.md`: the model and the build order. `examples/`: scripts and the images they write to `examples/output/`.

## Conventions

- Kernels are `@njit(cache=True, error_model="numpy")`, without `fastmath` and without `parallel`, so that a drawing repeats exactly. What they call for every pixel is `@_inlined` (the same, compiled into its caller: `inline="always"`). Only options that numba documents are used. Constants they use live in `_engine.py`.
- A kernel counts no reference to an array as it works: left in, that counting doubles the time of a stroke and nearly triples that of a rub. A per-pixel helper is given numbers where the kernel has them; given an array, it reads what it needs of it before its first `if`, and what it does to it only sometimes is in `for _ in once(...)`, not under an `if` or after an early `return`. A kernel called for every step calls no other kernel and allocates nothing. The top of `_engine.py` says why and how; `tools/refcounts.py` tells what is still counted, and `tests/test_refcounts.py` fails if anything is. Check it after any change to a kernel or a helper.
- A change that is not meant to alter the picture (an optimisation, a refactoring) leaves every hash of `tools/fingerprint.py` as it is (`tests/test_fingerprint.py` checks the recorded ones; for more, `--thorough --save` before and `--thorough --against` after) and regenerates every image byte for byte. One that is meant to alter it records the hashes again, `python tools/fingerprint.py --record`, with the images.
- The medium is conserved: what leaves the tip is on the paper, and what a stump or an eraser takes off is on the tool. Tests check this.
- A change meant to alter the picture is judged by measurements, not by looking at two renders side by side, and regenerates every image in `examples/output/` with the final code.
- Colour (pigments, Kubelka-Munk mixing), stroke paths, noise, reference images and PNG output come from `murex`, which `oilpaint` shares. What both libraries need goes there, not here.
- ruff caps function complexity at 13.
