"""What shines: one drawing under three lamps.

Run from the repository root: ``python examples/shine.py``. Writes
``examples/output/shine.png``.

Top row of the drawing. Left: an HB, a 4B and an 8B, each shaded lighter to
firmer from left to right. Right: three firm layers of 8B beside three of
compressed charcoal.

Bottom row, from the left: a 2H in six layers, laid lightly in its left half
and pressed hard in its right, where it flattens the tooth; a red coloured
pencil, its right half burnished with a colourless blender; a blue wax
crayon pressed hard; an orange oil pastel pressed hard; and blue, black and
red ballpoint, each hatched eight times over.

Top, under a raking lamp: nothing mirrors it to the eye, and the graphite is
at its darkest. Middle, under the default lamp: the heavy graphite is grey,
the 2H that flattened the tooth has a sheen where the same lead laid lightly
has none, and the polished wax has a faint gloss, which leaves its darks
deep. Bottom, with the lamp straight overhead: the heavy graphite mirrors it
and turns silvery, where the charcoal stays black; the polished wax mirrors
it too, with a soft white gloss; and the heavy ink bronzes, the blue towards
copper, the red towards green.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from murex import write_png

from fusain import Sheet, charcoal, crayons, papers, pastels, pencils, pens

LAMPS = ((-1.0, -1.0, 0.6), None, (0.0, 0.0, 1.0))


def graphite(sheet: Sheet) -> None:
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


def polished(sheet: Sheet) -> None:
    """A hard lead that flattens the tooth, and wax that is polished."""
    for layer in range(6):
        angle = 30 + 50 * layer
        sheet.shade((30, 250, 100, 350), pencils.H2, 0.5, angle=angle)
        sheet.shade((100, 250, 170, 350), pencils.H2, 1.0, angle=angle)
    red = pencils.coloured("#c0392b")
    for layer in range(3):
        sheet.shade((200, 250, 340, 350), red, 0.6, angle=30 + 55 * layer)
    for angle in (20, -50):
        sheet.shade((270, 245, 345, 355), pencils.BLENDER, 1.0, angle=angle)
    for layer in range(2):
        angle = 30 + 55 * layer
        sheet.shade((370, 250, 470, 350), crayons.wax("#2456a6"), 0.95, 9, angle)
        sheet.shade((500, 250, 600, 350), pastels.oil("#e67e22"), 0.9, 14, angle)


def heavy_ink(sheet: Sheet) -> None:
    for k, pen in enumerate((pens.BLUE, pens.BLACK, pens.RED)):
        left = 630 + 84 * k
        for layer in range(8):
            sheet.shade((left, 250, left + 72, 350), pen, 0.8, angle=35 + 47 * layer)


def drawing() -> Sheet:
    sheet = Sheet(900, 380, paper=papers.CARTRIDGE, seed=51)
    graphite(sheet)
    polished(sheet)
    heavy_ink(sheet)
    return sheet


def main() -> None:
    sheet = drawing()
    board = np.concatenate([sheet.render(light=lamp) for lamp in LAMPS])
    out = Path(__file__).parent / "output" / "shine.png"
    write_png(str(out), np.round(board * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
