"""Charcoal, a stump and erasers, with graphite beside them for comparison.

Run from the repository root: ``python examples/charcoal.py``. Writes
``examples/output/charcoal.png``. Three columns: vine charcoal, compressed
charcoal and a 4B pencil, each shaded lighter towards the bottom. Across all
three, from the top: untouched; a stump dragged through from left to right
and out onto the bare paper, where it leaves little, then loaded on a scrap
of the same medium and drawn down the bare paper beside it, which takes the
dust off it; a finger rubbed over; a kneaded eraser pulled through three
times; a vinyl eraser pulled through once.
"""

from __future__ import annotations

from pathlib import Path

from fusain import Sheet, Smudger, charcoal, papers, pencils, tools

COLUMNS = ((charcoal.VINE, 14.0), (charcoal.COMPRESSED, 14.0), (pencils.B4, 7.0))


def main() -> None:
    sheet = Sheet(900, 640, paper=papers.INGRES, seed=21)
    for i, (medium, width) in enumerate(COLUMNS):
        left = 60 + i * 280
        box = (left, 30, left + 200, 610)
        for angle in (35, -30):
            sheet.shade(
                box, medium, lambda x, y: 0.75 - 0.5 * (y - 30) / 580, width, angle
            )
        # a stump in the hand keeps what it picks up: dragged out of the
        # tone onto the bare paper
        stump = Smudger(tools.STUMP)
        for k in range(3):
            sheet.smudge(
                [(left + 20, 175 + 9 * k), (left + 262, 180 + 9 * k)], stump, 0.7
            )
        # a stump fills slowly: rubbed all over a scrap of the same medium,
        # it is loaded, and draws on the bare paper beside the tone
        scrap = Sheet(160, 120, paper=papers.INGRES, seed=22 + i)
        for angle in (35, -30):
            scrap.shade((10, 10, 150, 110), medium, 0.7, width, angle)
        scrap.rub((16, 16, 144, 104), stump, 0.7)
        sheet.smudge(
            [(left + 236, 222), (left + 246, 300), (left + 232, 380)], stump, 0.6
        )
        # a finger, rubbed to and fro
        finger = Smudger(tools.FINGER)
        for k in range(6):
            y = 290 + 9 * k
            ends = [(left + 15, y), (left + 185, y + 4)]
            sheet.smudge(ends if k % 2 else ends[::-1], finger, 0.6)
        for _ in range(3):
            sheet.erase([(left - 10, 425), (left + 210, 435)], tools.KNEADED, 0.7)
        sheet.erase([(left - 10, 530), (left + 210, 540)], tools.ERASER, 0.8)
    out = Path(__file__).parent / "output" / "charcoal.png"
    sheet.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
