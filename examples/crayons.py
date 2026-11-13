"""Wax crayons on cartridge paper, and oil pastels on a tan paper.

Run from the repository root: ``python examples/crayons.py``. Writes
``examples/output/crayons.png``.

Top, wax crayons. Left: red, yellow and blue, each lighter to firmer from
left to right. Middle: the same three laid over one another, which mix.
Right: a white crayon over a black one, which it does not cover, and lines
drawn lighter to firmer.

Bottom, oil pastels. Left: strokes of light colours over a dark blue ground,
which they half cover. Middle: orange and blue side by side, then the same
with the join rubbed with a finger. Right: a white oil pastel over a black
one, and lines drawn lighter to firmer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from murex import write_png

from fusain import Sheet, crayons, papers, pastels, tools

RED, YELLOW, BLUE = "#c0392b", "#f2c40f", "#2456a6"
WHITE, BLACK = "#f4f1ea", "#161616"


def wax_crayons() -> Sheet:
    sheet = Sheet(900, 300, paper=papers.CARTRIDGE, seed=41)
    for row, colour in enumerate((RED, YELLOW, BLUE)):
        top = 24 + row * 88
        box = (30, top, 290, top + 72)
        sheet.shade(
            box, crayons.wax(colour), lambda x, y: 0.15 + 0.85 * (x - 30) / 260, 9
        )
    for colour, box, angle in (
        (YELLOW, (340, 30, 520, 180), 35),
        (BLUE, (430, 110, 610, 270), -30),
        (RED, (340, 150, 480, 270), 70),
    ):
        for layer in range(2):
            sheet.shade(box, crayons.wax(colour), 0.7, 9, angle=angle + 50 * layer)
    for angle in (30, -35):
        sheet.shade((660, 30, 860, 150), crayons.wax(BLACK), 0.7, 9, angle)
    sheet.shade((700, 55, 820, 125), crayons.wax(WHITE), 0.8, 9, 60)
    red = crayons.wax(RED)
    for k in range(4):
        wave = [
            (665 + 22 * j, 190 + 22 * k + 7 * np.sin(j * 1.3 + k)) for j in range(10)
        ]
        sheet.stroke(wave, red, 0.25 + 0.25 * k, 7)
    return sheet


def oil_pastels() -> Sheet:
    sheet = Sheet(900, 300, paper=papers.TONED_TAN, seed=42)
    night = pastels.oil("#1f2a44")
    for angle in (30, -35):
        sheet.shade((30, 30, 290, 270), night, 0.7, 14, angle)
    for colour, y in (
        (WHITE, 70),
        ("#f6d55c", 120),
        ("#8fd3f4", 170),
        ("#e2725b", 220),
    ):
        sheet.stroke(
            [(50, y), (160, y - 14), (270, y + 6)], pastels.oil(colour), 0.8, 14
        )
    orange, blue = pastels.oil("#e67e22"), pastels.oil("#2e6fb7")
    for top in (30, 160):
        for angle in (25, -40):
            sheet.shade((340, top, 475, top + 110), orange, 0.7, 14, angle)
            sheet.shade((465, top, 600, top + 110), blue, 0.7, 14, angle)
    for k in range(18):  # a finger, to and fro across the join
        y = 166 + 6 * k
        ends = [(415, y), (525, y + 2)]
        sheet.smudge(ends, tools.FINGER, 0.8)
        sheet.smudge(ends[::-1], tools.FINGER, 0.8)
    for angle in (30, -35):
        sheet.shade((660, 30, 860, 150), pastels.oil(BLACK), 0.7, 14, angle)
    sheet.shade((700, 55, 820, 125), pastels.oil(WHITE), 0.8, 14, 60)
    red = pastels.oil(RED)
    for k in range(4):
        wave = [
            (665 + 22 * j, 190 + 22 * k + 7 * np.sin(j * 1.3 + k)) for j in range(10)
        ]
        sheet.stroke(wave, red, 0.25 + 0.25 * k, 9)
    return sheet


def main() -> None:
    board = np.concatenate([wax_crayons().render(), oil_pastels().render()])
    out = Path(__file__).parent / "output" / "crayons.png"
    write_png(str(out), np.round(board * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
