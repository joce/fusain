"""How long the engine takes: shading, rubbing, hatching with a pen and
rendering a sheet of 1120 by 820 pixels, the best of several runs each::

    python tools/benchmark.py            # every workload, best of 5
    python tools/benchmark.py 3 rub pen  # those whose name has "rub" or "pen"

Run it before and after a change meant to make drawing faster (the first run
after a change to ``fusain/_engine.py`` compiles the kernels, which is not
timed), and ``tools/fingerprint.py`` to see that the picture has not changed.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import numpy as np

from fusain import Sheet, Smudger, charcoal, papers, pastels, pencils, pens, tools

W, H = 1120, 820
AREA = (0, 0, W, H)
RED, BLUE = "#c0392b", "#2456a6"


def _blank(paper=papers.CARTRIDGE) -> Sheet:
    return Sheet(W, H, paper=paper, seed=1)


def _charcoaled() -> Sheet:
    sheet = _blank(papers.CHARCOAL)
    sheet.shade(AREA, charcoal.VINE, 0.6, width=10)
    sheet.shade(AREA, charcoal.COMPRESSED, 0.4, width=8, angle=-40)
    return sheet


def _toned() -> Sheet:
    sheet = _blank()
    sheet.shade(AREA, pencils.B4, 0.6, width=8)
    return sheet


def _layered() -> Sheet:
    sheet = _toned()
    sheet.shade(AREA, pastels.soft(RED), 0.4, width=12, angle=-40)
    return sheet


def _short_strokes(sheet: Sheet) -> None:
    for y in np.linspace(10, H - 20, 40):
        for x in np.linspace(10, W - 50, 50):
            sheet.stroke([(x, y), (x + 30.0, y + 9.0)], pencils.B2, 0.6, width=3)


#: Each workload: the sheet it starts from (made once, copied for each run)
#: and what is timed.
WORKLOADS: dict[str, tuple[Callable[[], Sheet], Callable[[Sheet], object]]] = {
    "shade HB, width 6": (_blank, lambda s: s.shade(AREA, pencils.HB)),
    "shade HB over a tone": (_toned, lambda s: s.shade(AREA, pencils.HB, angle=-40)),
    "shade soft pastel, width 24": (
        lambda: _blank(papers.SANDED),
        lambda s: s.shade(AREA, pastels.soft(RED), 0.6, width=24),
    ),
    "shade soft pastel over charcoal": (
        _charcoaled,
        lambda s: s.shade(AREA, pastels.soft(RED), 0.6, width=24),
    ),
    "shade vine charcoal on its side": (
        lambda: _blank(papers.CHARCOAL),
        lambda s: s.shade(AREA, charcoal.VINE, 0.5, width=6, tilt=1.0),
    ),
    "shade coloured pencil, firm": (
        _blank,
        lambda s: s.shade(AREA, pencils.coloured(BLUE), 0.9, width=6),
    ),
    "rub charcoal with a finger": (
        _charcoaled,
        lambda s: s.rub(AREA, tools.FINGER, 0.6),
    ),
    "rub graphite with a stump": (
        _toned,
        lambda s: s.rub(AREA, Smudger(tools.STUMP), 0.6),
    ),
    "rub charcoal with a kneaded eraser": (
        _charcoaled,
        lambda s: s.rub(AREA, tools.KNEADED, 0.6),
    ),
    "hatch with a ballpoint pen": (_blank, lambda s: s.shade(AREA, pens.BLUE, 0.6)),
    "pen over graphite": (_toned, lambda s: s.shade(AREA, pens.BLUE, 0.6)),
    "2000 short strokes": (_blank, _short_strokes),
    "render": (_layered, lambda s: s.render()),
    "reflectance": (_layered, lambda s: s.reflectance()),
}


def _warm_up() -> None:
    """Compile every kernel, or load it from numba's cache."""
    sheet = Sheet(64, 64, seed=1)
    sheet.shade((4, 4, 60, 60), pencils.HB)
    sheet.rub((4, 4, 60, 60), tools.FINGER)
    sheet.shade((4, 4, 60, 60), pens.BLUE)
    sheet.indent([(4, 4), (60, 60)])
    sheet.render()


def main(arguments: list[str]) -> int:
    runs = int(arguments.pop(0)) if arguments and arguments[0].isdigit() else 5
    _warm_up()
    for name, (start, work) in WORKLOADS.items():
        if arguments and not any(word in name for word in arguments):
            continue
        base, times = start(), []
        for _ in range(runs):
            sheet = base.copy()
            began = time.perf_counter()
            work(sheet)
            times.append(time.perf_counter() - began)
        print(f"{name:36s} {min(times):7.3f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
