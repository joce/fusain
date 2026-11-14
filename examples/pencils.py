"""Graphite pencils on cartridge paper: what grade, pressure, layering and
the tilt of the pencil do, and what the pencil does to the paper.

Run from the repository root: ``python examples/pencils.py``. Writes
``examples/output/pencils.png``:

* top, one band per grade (4H, 2H, HB, 2B, 4B, 6B, 8B from the top), shaded
  with the pressure rising from left to right;
* bottom left, HB at a light pressure in 1, 2, 4 and 8 layers, each layer at
  another angle;
* bottom middle, a 4B held on its side: three broad strokes, light to firm,
  and a patch shaded that way, firmer towards the bottom;
* lower right, 2B lines drawn lightly, firmly and hard, and cross-hatching;
* bottom left, five lines indented with a stylus, from the lightest touch to
  the firmest, and a 2B shaded over them, lightly in the left half and hard
  in the right: the pencil skips a groove unless it reaches its floor;
* bottom middle, blades of grass indented first and a 4B shaded over them:
  they stay the white of the paper;
* bottom right, a 2H pressed hard over the left half, which flattens the
  tooth, and one light layer of 4B over all of it: grainy on the tooth that
  was left alone, even on the flattened one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fusain import Sheet, papers, pencils

GRADES = ("4H", "2H", "HB", "2B", "4B", "6B", "8B")


def surface(sheet: Sheet) -> None:
    """Grooves that a pencil skips, and a tooth pressed flat."""
    for k in range(5):
        y = 626 + 22 * k
        sheet.indent([(40, y), (280, y - 6)], 0.2 + 0.2 * k)
    sheet.shade((30, 610, 160, 730), pencils.B2, 0.35, angle=60)
    sheet.shade((160, 610, 290, 730), pencils.B2, 0.95, angle=60)
    rng = np.random.default_rng(5)
    for x in np.linspace(350, 590, 17):
        lean, tall = rng.uniform(-28, 28), rng.uniform(55, 105)
        blade = [(x, 728), (x + 0.3 * lean, 728 - 0.6 * tall), (x + lean, 728 - tall)]
        sheet.indent(blade, (0.9, 0.5), 2.5)
    for angle in (25, -40):
        sheet.shade((340, 610, 600, 730), pencils.B4, 0.5, angle=angle)
    for layer in range(5):
        sheet.shade((640, 610, 755, 730), pencils.H2, 1.0, angle=30 + 50 * layer)
    sheet.shade((640, 610, 870, 730), pencils.B4, 0.3, angle=-20)


def main() -> None:
    sheet = Sheet(900, 750, paper=papers.CARTRIDGE, seed=11)
    for row, name in enumerate(GRADES):
        top = 20 + row * 46
        box = (30, top, 870, top + 36)
        sheet.shade(box, pencils.grade(name), lambda x, y: 0.05 + 0.95 * (x - 30) / 840)
    for i, layers in enumerate((1, 2, 4, 8)):
        left = 30 + i * 88
        for layer in range(layers):
            sheet.shade(
                (left, 360, left + 76, 575), pencils.HB, 0.3, angle=30 + 47 * layer
            )
    for k, pressure in enumerate((0.3, 0.6, 1.0)):
        wave = [
            (400 + 30 * j, 380 + 34 * k + 8 * np.sin(j * 1.1 + k)) for j in range(7)
        ]
        sheet.stroke(wave, pencils.B4, pressure, width=5, tilt=1.0)
    side = (400, 480, 590, 575)
    sheet.shade(
        side, pencils.B4, lambda x, y: 0.3 + 0.7 * (y - 480) / 95, 5, 20, tilt=1.0
    )
    for k, pressure in enumerate((0.2, 0.5, 0.95)):
        wave = [
            (630 + 24 * j, 380 + 30 * k + 9 * np.sin(j * 1.2 + k)) for j in range(11)
        ]
        sheet.stroke(wave, pencils.B2, pressure=pressure)
    sheet.shade((630, 480, 870, 575), pencils.HB, 0.6, width=4, angle=45, spacing=9)
    sheet.shade((630, 480, 870, 575), pencils.HB, 0.6, width=4, angle=-45, spacing=9)
    surface(sheet)
    out = Path(__file__).parent / "output" / "pencils.png"
    sheet.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
