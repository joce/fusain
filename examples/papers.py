"""The papers of :mod:`fusain.papers`, side by side under a raking light.

Run from the repository root: ``python examples/papers.py``. Writes
``examples/output/papers.png``, one swatch per paper in the order of
``papers.ALL``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from murex import write_png

from fusain import Sheet, papers

SWATCH = 220
GAP = 8
PER_ROW = 6
RAKING = (-0.6, -0.8, 0.45)


def main() -> None:
    rows = -(-len(papers.ALL) // PER_ROW)
    width = PER_ROW * SWATCH + (PER_ROW + 1) * GAP
    height = rows * SWATCH + (rows + 1) * GAP
    board = np.full((height, width, 3), 0.25, np.float32)
    for i, paper in enumerate(papers.ALL):
        sheet = Sheet(SWATCH, SWATCH, paper=paper, seed=7)
        y = GAP + (i // PER_ROW) * (SWATCH + GAP)
        x = GAP + (i % PER_ROW) * (SWATCH + GAP)
        board[y : y + SWATCH, x : x + SWATCH] = sheet.render(light=RAKING)
    out = Path(__file__).parent / "output" / "papers.png"
    write_png(str(out), np.round(board * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
