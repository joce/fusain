"""Fixative: the same drawing rubbed twice, the second time after fixing.

Run from the repository root: ``python examples/fixative.py``. Writes
``examples/output/fixative.png``.

Each row is drawn alike and then rubbed alike; the bottom one is sprayed with
fixative in between. Left: vine charcoal, a stump dragged through it and out
onto the bare paper, and a kneaded eraser pulled through. Middle: orange and
blue soft pastel, the join rubbed with a finger. Right: a dark blue ground,
then (after the fixative) white pastel strokes over it, the lower two rubbed
with a finger: unfixed, the finger stirs the blue up into the white.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from murex import write_png

from fusain import Sheet, charcoal, papers, pastels, tools


def draw(sheet: Sheet) -> None:
    for angle in (30, -35):
        sheet.shade((30, 40, 250, 260), charcoal.VINE, 0.6, 12, angle)
    orange, blue = pastels.soft("#e67e22"), pastels.soft("#2e6fb7")
    for angle in (25, -40):
        sheet.shade((340, 40, 475, 260), orange, 0.7, 14, angle)
        sheet.shade((465, 40, 600, 260), blue, 0.7, 14, angle)
    for angle in (30, -35):
        sheet.shade((660, 40, 870, 260), pastels.soft("#1f2a44"), 0.7, 14, angle)


def rub(sheet: Sheet) -> None:
    for y in (90, 120, 150):
        sheet.smudge([(60, y), (310, y + 4)], tools.STUMP, 0.7)
    sheet.erase([(50, 215), (240, 210)], tools.KNEADED, 0.8)
    for k in range(18):  # a finger, to and fro across the join
        y = 100 + 6 * k
        ends = [(415, y), (525, y + 2)]
        sheet.smudge(ends, tools.FINGER, 0.6)
        sheet.smudge(ends[::-1], tools.FINGER, 0.6)
    white = pastels.soft("#f4f1ea")
    for y in (80, 125, 170, 215):
        sheet.stroke([(685, y), (765, y - 10), (845, y + 4)], white, 0.8, 14)
    for y in (170, 215):
        for ends in ([(680, y), (850, y)], [(850, y - 2), (680, y - 2)]):
            sheet.smudge(ends, tools.FINGER, 0.7)


def main() -> None:
    rows = []
    for fixed in (False, True):
        sheet = Sheet(900, 300, paper=papers.CARTRIDGE, seed=61)
        draw(sheet)
        if fixed:
            sheet.fix()
        rub(sheet)
        rows.append(sheet.render())
    out = Path(__file__).parent / "output" / "fixative.png"
    write_png(str(out), np.round(np.concatenate(rows) * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
