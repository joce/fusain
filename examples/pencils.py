"""Graphite pencils on cartridge paper: what grade, pressure and layering do.

Run from the repository root: ``python examples/pencils.py``. Writes
``examples/output/pencils.png``:

* top, one band per grade (4H, 2H, HB, 2B, 4B, 6B, 8B from the top), shaded
  with the pressure rising from left to right;
* bottom left, HB at a light pressure in 1, 2, 4, 8 and 16 layers;
* bottom right, lines: a 2B drawn lightly then firmly, and cross-hatching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fusain import Sheet, papers, pencils

GRADES = ("4H", "2H", "HB", "2B", "4B", "6B", "8B")


def shade(sheet, medium, pressure, x0, y0, x1, y1, width=3.0, gap=1.5) -> None:
    """One even layer over a rectangle: strokes side by side, overlapping."""
    for y in np.arange(y0, y1 + 0.5 * gap, gap):
        sheet.stroke([(x0, y), (x1, y)], medium, pressure=pressure, width=width)


def main() -> None:
    sheet = Sheet(900, 560, paper=papers.CARTRIDGE, seed=11)
    for row, name in enumerate(GRADES):
        top = 24 + row * 44
        shade(
            sheet,
            pencils.grade(name),
            lambda u: 0.05 + 0.95 * u,
            30,
            top,
            870,
            top + 32,
        )
    for i, layers in enumerate((1, 2, 4, 8, 16)):
        left = 30 + i * 86
        for _ in range(layers):
            shade(sheet, pencils.HB, 0.35, left, 350, left + 70, 530)
    rng = np.random.default_rng(5)
    for pressure, top in ((0.25, 360), (0.9, 400)):
        wave = [(480 + 26 * k, top + 14 * np.sin(k * 1.3)) for k in range(15)]
        sheet.stroke(wave, pencils.B2, pressure=pressure, width=2.5)
    for slant in (1, -1):
        for x in np.arange(500, 860, 7.0):
            x += rng.normal(0, 0.8)
            sheet.stroke(
                [(x, 450 + 40 * slant), (x + 40, 450 - 40 * slant)],
                pencils.HB,
                pressure=lambda u: 0.7 - 0.4 * u,
                width=2.0,
            )
    out = Path(__file__).parent / "output" / "pencils.png"
    sheet.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
