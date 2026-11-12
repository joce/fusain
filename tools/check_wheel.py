"""Verify the installed wheel: it is imported from the environment (not the
source checkout), it knows its version, murex came with it, and its compiled
engine draws.

Run through ``uv run tox -e wheel``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import murex

import fusain

assert Path(fusain.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path.cwd().joinpath("pyproject.toml").exists()
assert fusain.__version__ != "0.0.0+unknown", fusain.__version__
for name in fusain.__all__:
    assert getattr(fusain, name) is not None, name
assert murex.Pigment("#3050c0").hex == "#3050c0"

# a pencil line across a small sheet: the numba engine compiles and draws
sheet = fusain.Sheet(120, 80, seed=1)
laid = sheet.stroke([(10, 40), (110, 40)], fusain.pencils.B2, pressure=0.8, width=4)
image = sheet.render(lighting=False)
assert laid > 0 and image.shape == (80, 120, 3), (laid, image.shape)
assert image[38:42, 20:100].mean() < image[:20].mean() - 0.1
print(f"fusain {fusain.__version__}: the installed wheel imports and draws.")
