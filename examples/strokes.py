"""The shape of a stroke: its ends, a width that changes along it, the side
of a stick.

Run from the repository root: ``python examples/strokes.py``. Writes
``examples/output/strokes.png``. On grey pastel paper, from the top:

- strokes of a hard pastel, round-ended and flat-ended, between the same two
  pencil ticks: a round end reaches half the tip past them, a flat one stops
  square at them; and short dabs of a broad soft pastel, as strong as the
  long strokes;
- a road and a fence that recede in perspective, each stroke narrowing as
  it goes away, and a brush-like stroke of a 6B that tapers to a point;
- a piece of vine charcoal laid flat (a band as wide as the stick is
  long), turned from its end onto its side as it goes, and a stick of soft
  pastel laid flat in flat-ended blocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fusain import Sheet, Stick, charcoal, papers, pastels, pencils

INK = "#2b2b3a"
WHITE = "#f1ede2"
RUST = "#a0482a"


def ends(sheet: Sheet) -> None:
    """Round ends against flat ends, between ticks; dabs."""
    for row, how in enumerate(("round", "flat")):
        y = 50 + 46 * row
        for x in (60.0, 330.0):
            sheet.stroke([(x, y - 22), (x, y + 22)], pencils.HB, 0.5, 1.5)
        sheet.stroke([(60, y), (330, y)], pastels.hard(INK), 0.8, 14.0, ends=how)
    for k in range(6):
        x = 420 + 70 * k
        dab = [(x, 80), (x + 14 + 4 * k, 66)]
        sheet.stroke(dab, pastels.soft(RUST), 0.75, 30.0)
    sheet.stroke([(420, 120), (800, 120)], pastels.soft(RUST), 0.75, 30.0)


def perspective(sheet: Sheet) -> None:
    """Strokes that narrow as they recede."""
    vanish = np.array([300.0, 210.0])
    for k, foot in enumerate(np.linspace(40.0, 560.0, 9)):
        start = np.array([foot, 380.0])
        end = start + 0.8 * (vanish - start)
        # (a width per point: as far as each is from the vanishing point)
        sheet.stroke(
            [start, end],
            pastels.soft(WHITE if k % 2 else INK),
            0.7,
            [16.0, 16.0 * 0.2],
            ends="flat",
        )
    tops = []
    for k in range(9):  # the posts of a fence going away
        near = 0.8**k  # (how big the k-th post looks against the first)
        x = 620 + 50 * (1 - near) / (1 - 0.8)
        foot = 380 - 45 * (1 - near)
        tops.append((x, foot - 90 * near))
        post = [(x, foot - 90 * near), (x, foot)]
        sheet.stroke(post, pastels.hard(INK), 0.8, 9 * near, ends="flat")
    # its rail, narrowing post by post
    widths = [5.0 * 0.8**k for k in range(9)]
    sheet.stroke(tops, pastels.hard(INK), 0.7, widths, ends="flat")
    stroke = [(640, 420), (720, 400), (820, 430), (880, 405)]
    sheet.stroke(
        stroke, pencils.B6, lambda u: 0.9 - 0.4 * u, lambda u: 14.0 * (1 - u) + 1.0
    )


def sides(sheet: Sheet) -> None:
    """The side of a stick."""
    vine = Stick(charcoal.VINE, width=6, lead=6, side=110)
    sheet.stroke([(60, 520), (420, 540)], vine, 0.55, tilt=1.0)
    sheet.stroke([(60, 640), (420, 640)], vine, 0.6, tilt=lambda u: u)
    block = Stick(pastels.soft(WHITE), width=10, lead=10, side=70)
    for k in range(4):
        x = 500 + 95 * k
        sheet.stroke(
            [(x, 500 + 12 * k), (x + 60, 500 + 12 * k)],
            block,
            0.7,
            tilt=1.0,
            ends="flat",
        )
    rust = Stick(pastels.soft(RUST), width=10, lead=10, side=70)
    sheet.stroke([(520, 620), (860, 640)], rust, 0.6, tilt=1.0, ends="flat")


def main() -> None:
    sheet = Sheet(900, 720, paper=papers.TONED_GREY, seed=7)
    ends(sheet)
    perspective(sheet)
    sides(sheet)
    out = Path(__file__).parent / "output" / "strokes.png"
    sheet.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
