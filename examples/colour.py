"""Colour: coloured pencils on cartridge paper, and pastels on a dark paper.

Run from the repository root: ``python examples/colour.py``. Writes
``examples/output/colour.png``.

Top, coloured pencils. Left: red, yellow and blue, each lighter to firmer
from left to right. Middle: the same three laid over one another, which mix
to the eye: orange, green and violet where two overlap. Right: three layers
of red, its right half then burnished with a colourless blender.

Bottom, soft pastels on grey paper. Left: strokes of light colours over a
dark blue ground, which they cover. Middle: orange and blue side by side,
then the same with the join rubbed with a finger. Right: a white pastel over a
black one, and lines drawn with a hard pastel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from murex import write_png

from fusain import Sheet, papers, pastels, pencils, tools

RED, YELLOW, BLUE = "#c0392b", "#f2c40f", "#2456a6"


def coloured_pencils() -> Sheet:
    sheet = Sheet(900, 300, paper=papers.CARTRIDGE, seed=31)
    for row, colour in enumerate((RED, YELLOW, BLUE)):
        top = 24 + row * 88
        box = (30, top, 290, top + 72)
        sheet.shade(
            box, pencils.coloured(colour), lambda x, y: 0.15 + 0.85 * (x - 30) / 260
        )
    # three patches that overlap two by two
    for colour, box, angle in (
        (YELLOW, (340, 30, 520, 180), 35),
        (BLUE, (430, 110, 610, 270), -30),
        (RED, (340, 150, 480, 270), 70),
    ):
        for layer in range(2):
            sheet.shade(box, pencils.coloured(colour), 0.7, angle=angle + 50 * layer)
    for layer in range(3):
        sheet.shade(
            (660, 40, 870, 260), pencils.coloured(RED), 0.6, angle=30 + 55 * layer
        )
    for angle in (20, -50):
        sheet.shade((765, 30, 880, 270), pencils.BLENDER, 1.0, angle=angle)
    return sheet


def soft_pastels() -> Sheet:
    sheet = Sheet(900, 300, paper=papers.TONED_GREY.with_(texture="laid"), seed=32)
    night = pastels.soft("#1f2a44")
    for angle in (30, -35):
        sheet.shade((30, 30, 290, 270), night, 0.7, 14, angle)
    for colour, y in (
        ("#f4f1ea", 70),
        ("#f6d55c", 120),
        ("#8fd3f4", 170),
        ("#e2725b", 220),
    ):
        sheet.stroke(
            [(50, y), (160, y - 14), (270, y + 6)], pastels.soft(colour), 0.8, 14
        )
    orange, blue = pastels.soft("#e67e22"), pastels.soft("#2e6fb7")
    for top in (30, 160):
        for angle in (25, -40):
            sheet.shade((340, top, 475, top + 110), orange, 0.7, 14, angle)
            sheet.shade((465, top, 600, top + 110), blue, 0.7, 14, angle)
    for k in range(18):  # a finger, to and fro across the join
        y = 166 + 6 * k
        ends = [(415, y), (525, y + 2)]
        sheet.smudge(ends if k % 2 else ends[::-1], tools.FINGER, 0.6)
    for angle in (30, -35):
        sheet.shade((660, 30, 860, 150), pastels.soft("#161616"), 0.7, 14, angle)
    sheet.shade((700, 55, 820, 125), pastels.soft("#f4f1ea"), 0.8, 14, 60)
    chalk = pastels.hard("#f4f1ea")
    for k in range(4):
        wave = [
            (665 + 22 * j, 190 + 22 * k + 7 * np.sin(j * 1.3 + k)) for j in range(10)
        ]
        sheet.stroke(wave, chalk, 0.4 + 0.2 * k, 5)
    return sheet


def main() -> None:
    board = np.concatenate([coloured_pencils().render(), soft_pastels().render()])
    out = Path(__file__).parent / "output" / "colour.png"
    write_png(str(out), np.round(board * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
