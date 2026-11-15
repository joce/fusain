"""Ballpoint pens, on cartridge paper.

Run from the repository root: ``python examples/ballpoint.py``. Writes
``examples/output/ballpoint.png``.

Left: lines in blue from the lightest touch to the firmest, then a line of
loops as a hand writes them, pressing harder on the way down. Middle: black
hatching in one to four layers, each at another angle, and under them one
layer pressed lighter to firmer from left to right. Right: a pencil sketch
inked over in black, its right half then rubbed with a vinyl eraser, which
takes the pencil and leaves the ink; and under it a red pen across a patch of
wax crayon, on which it skids.

Bottom. Left: a zigzag and a scribble to and fro: the ball gathers ink as it
rolls and leaves a blob where the line turns sharply, and now and then where
it starts. The scribble is written with a pen in the hand that has just drawn
the long line under it, so it lands with the ink it gathered there.
Middle: five lines pressed from lightest to firmest, and a soft pencil
shaded lightly over them: the ball presses a groove, which the pencil
skips where the pen was firm. Right: loops written with a pen that has run
dry (``Sheet.indent``) and shaded over: the grooves stay white.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
from murex import Polyline, write_png

from fusain import HeldPen, Sheet, crayons, papers, pencils, pens, tools


def lines(sheet: Sheet) -> None:
    for k in range(6):
        y = 40 + 22 * k
        wave = [(40 + 25 * j, y + 4 * math.sin(0.9 * j + k)) for j in range(10)]
        sheet.stroke(wave, pens.BLUE, 0.1 + 0.18 * k)
    # loops, as in handwriting: firmer going down than coming up
    turns = [0.11 * t for t in range(250)]
    loops = [(50 + 8 * a - 21 * math.sin(a), 232 + 21 * math.cos(a)) for a in turns]
    firm = [0.45 - 0.3 * math.sin(a) for a in turns]
    sheet.stroke(loops, pens.BLUE, firm)


def hatching(sheet: Sheet) -> None:
    for patch in range(4):
        left = 340 + 68 * patch
        for layer in range(patch + 1):
            sheet.shade(
                (left, 35, left + 56, 150), pens.BLACK, 0.5, angle=35 + 47 * layer
            )
    for angle in (30, -40):
        sheet.shade(
            (340, 180, 600, 265),
            pens.BLACK,
            lambda x, y: 0.05 + 0.95 * (x - 340) / 260,
            angle=angle,
        )


def sketch(sheet: Sheet) -> None:
    circle = [
        (765 + 62 * math.cos(0.1 * t), 95 + 55 * math.sin(0.1 * t)) for t in range(64)
    ]
    box = [(690, 45), (840, 40), (845, 150), (688, 146), (690, 45)]
    for path in (circle, box):
        for wobble in (-2.0, 1.5):  # searching pencil lines
            sheet.stroke([(x + wobble, y - wobble) for x, y in path], pencils.B2, 0.55)
    sheet.shade((700, 60, 830, 135), pencils.B2, 0.4, angle=40)
    sheet.stroke(circle, pens.BLACK, 0.6)
    for start, end in itertools.pairwise(box):
        sheet.stroke([start, end], pens.BLACK, 0.6)
    for k in range(9):  # the eraser, over the right half
        y = 35 + 14 * k
        sheet.erase([(770, y), (865, y)], tools.ERASER, 0.9, 18)
    for angle in (30, -40):
        sheet.shade((690, 190, 790, 265), crayons.wax("#f2c40f"), 0.9, 9, angle)
    for k in range(4):
        y = 200 + 18 * k
        sheet.stroke([(670, y), (870, y + 3)], pens.RED, 0.5)


def blobs(sheet: Sheet) -> None:
    """Sharp turns, where the ball leaves the ink it has gathered."""
    zigzag = [(40 + 25 * j, 372 if j % 2 else 322) for j in range(11)]
    sheet.stroke(Polyline(zigzag), pens.BLUE, 0.5)
    pen = HeldPen(pens.BLACK)  # (it lands with what the line under gathered)
    sheet.stroke([(40, 444), (290, 446)], pen, 0.45)
    scribble = [(40 + 250 * (j % 2), 396 + 5 * j) for j in range(8)]
    sheet.stroke(Polyline(scribble), pen, 0.45)


def grooves(sheet: Sheet) -> None:
    """The groove of a firm pen, and of a pen with no ink, under a pencil."""
    for k in range(5):
        y = 330 + 22 * k
        wave = [(350 + 24 * j, y + 4 * math.sin(0.9 * j + k)) for j in range(11)]
        sheet.stroke(wave, pens.BLUE, 0.2 + 0.2 * k)
    sheet.shade((340, 312, 600, 438), pencils.B4, 0.35, angle=60)
    turns = [0.11 * t for t in range(200)]
    for row in range(2):
        y = 345 + 52 * row
        loops = [(690 + 8 * a - 19 * math.sin(a), y + 19 * math.cos(a)) for a in turns]
        sheet.indent(loops, 0.75, 3.0)
    for angle in (55, -35):
        sheet.shade((670, 312, 870, 438), pencils.B4, 0.5, angle=angle)


def main() -> None:
    sheet = Sheet(900, 460, paper=papers.CARTRIDGE, seed=71)
    lines(sheet)
    hatching(sheet)
    sketch(sheet)
    blobs(sheet)
    grooves(sheet)
    out = Path(__file__).parent / "output" / "ballpoint.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
