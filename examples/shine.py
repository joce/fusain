"""Graphite's shine: one drawing under three lamps.

Run from the repository root: ``python examples/shine.py``. Writes
``examples/output/shine.png``.

Left: an HB, a 4B and an 8B, each shaded lighter to firmer from left to
right. Right: three firm layers of 8B beside three of compressed charcoal.

Top, under a raking lamp: nothing mirrors it to the eye, and the graphite is
at its darkest. Middle, under the default lamp. Bottom, with the lamp straight
overhead: the heavy graphite mirrors it and turns silvery, where the charcoal
stays black.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from murex import write_png

from fusain import Sheet, charcoal, papers, pencils

LAMPS = ((-1.0, -1.0, 0.6), None, (0.0, 0.0, 1.0))


def drawing() -> Sheet:
    sheet = Sheet(900, 240, paper=papers.CARTRIDGE, seed=51)
    for row, pencil in enumerate((pencils.HB, pencils.B4, pencils.B8)):
        top = 20 + row * 70
        for layer in range(2):
            sheet.shade(
                (30, top, 520, top + 56),
                pencil,
                lambda x, y: 0.1 + 0.9 * (x - 30) / 490,
                angle=30 + 55 * layer,
            )
    for layer in range(3):
        angle = 30 + 50 * layer
        sheet.shade((570, 30, 710, 210), pencils.B8, 1.0, angle=angle)
        sheet.shade((730, 30, 870, 210), charcoal.COMPRESSED, 1.0, 8, angle)
    return sheet


def main() -> None:
    sheet = drawing()
    board = np.concatenate([sheet.render(light=lamp) for lamp in LAMPS])
    out = Path(__file__).parent / "output" / "shine.png"
    write_png(str(out), np.round(board * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
