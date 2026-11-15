"""The kernels count no reference to an array as they work
(``tools/refcounts.py``): numba counts one wherever an array is handed to a
helper in a way it cannot see through, for every pixel, which costs more
than the work. The top of ``fusain/_engine.py`` says how a helper is written
so that it does not."""

import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).parent.parent / "tools" / "refcounts.py"


def test_no_kernel_counts_references_as_it_works():
    # (in a process of its own: the kernels are compiled afresh there, in a
    # cache of their own, which numba must be told of before it is imported)
    ran = subprocess.run(
        [sys.executable, str(TOOL)], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "bear" in ran.stdout and "rub_under" in ran.stdout
