"""Graphite pencils on cartridge paper: what grade, pressure and layering do.

Run from the repository root: ``python examples/pencils.py``. Writes
``examples/output/pencils.png``:

* top, one band per grade (4H, 2H, HB, 2B, 4B, 6B, 8B from the top), shaded
  with the pressure rising from left to right;
* bottom left, HB at a light pressure in 1, 2, 4 and 8 layers, each layer at
  another angle;
* bottom right, 2B lines drawn lightly, firmly and hard, and cross-hatching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fusain import Sheet, papers, pencils

GRADES = ("4H", "2H", "HB", "2B", "4B", "6B", "8B")


def main() -> None:
    sheet = Sheet(900, 600, paper=papers.CARTRIDGE, seed=11)
    for row, name in enumerate(GRADES):
        top = 20 + row * 46
        box = (30, top, 870, top + 36)
        sheet.shade(box, pencils.grade(name), lambda x, y: 0.05 + 0.95 * (x - 30) / 840)
    for i, layers in enumerate((1, 2, 4, 8)):
        left = 30 + i * 112
        for layer in range(layers):
            sheet.shade(
                (left, 360, left + 96, 575), pencils.HB, 0.3, angle=30 + 47 * layer
            )
    for k, pressure in enumerate((0.2, 0.5, 0.95)):
        wave = [
            (500 + 28 * j, 380 + 34 * k + 11 * np.sin(j * 1.2 + k)) for j in range(14)
        ]
        sheet.stroke(wave, pencils.B2, pressure=pressure)
    sheet.shade((510, 480, 860, 570), pencils.HB, 0.6, width=4, angle=45, spacing=9)
    sheet.shade((510, 480, 860, 570), pencils.HB, 0.6, width=4, angle=-45, spacing=9)
    out = Path(__file__).parent / "output" / "pencils.png"
    sheet.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
