"""Verify the installed wheel: it is imported from the environment (not the
source checkout) and it knows its version.

Run through ``uv run tox -e wheel``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fusain

assert Path(fusain.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path.cwd().joinpath("pyproject.toml").exists()
assert fusain.__version__ != "0.0.0+unknown", fusain.__version__
for name in fusain.__all__:
    assert getattr(fusain, name) is not None, name
print(f"fusain {fusain.__version__}: the installed wheel imports.")
