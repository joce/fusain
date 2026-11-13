"""Tips: a point that wears as it draws.

Run from the repository root: ``python examples/tips.py``. Writes
``examples/output/tips.png``. One line drawn back and forth down the sheet
without lifting or sharpening, with a 2H pencil (left), a 2B (middle) and a
6B (right), each sharp at the top: the softer the lead, the sooner its
point is a flat and its line a band.
"""

from __future__ import annotations

from pathlib import Path

from murex import Polyline

from fusain import Sheet, Stick, papers, pencils

ROWS = 14


def main() -> None:
    sheet = Sheet(900, 360, paper=papers.CARTRIDGE, seed=41)
    for column, name in enumerate(("2H", "2B", "6B")):
        left = 40 + column * 290
        pencil = Stick(pencils.grade(name), width=2, lead=16)
        corners = []
        for row in range(ROWS):
            y = 30 + row * 22
            ends = [(left, y), (left + 240, y + 8)]
            corners += ends if row % 2 == 0 else ends[::-1]
        sheet.stroke(Polyline(corners), pencil, pressure=0.7)
    out = Path(__file__).parent / "output" / "tips.png"
    sheet.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
