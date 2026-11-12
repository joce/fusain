"""Graphite pencils on cartridge paper: what grade, pressure and layering do.

Run from the repository root: ``python examples/pencils.py``. Writes
``examples/output/pencils.png``:

* top, one band per grade (4H, 2H, HB, 2B, 4B, 6B, 8B from the top), shaded
  with the pressure rising from left to right;
* bottom left, HB at a light pressure in 1, 2, 4 and 8 layers, each layer at
  another angle;
* bottom right, 2B lines drawn lightly, firmly and hard, and cross-hatching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fusain import Sheet, papers, pencils

GRADES = ("4H", "2H", "HB", "2B", "4B", "6B", "8B")
#: Width of the pencil's tip, in pixels: about two grains of the paper.
TIP = 6.0


def shade(sheet, medium, pressure, box, rng, angle=30.0, width=TIP) -> None:
    """Shade the rectangle ``box`` (x0, y0, x1, y1) as a hand does: strokes
    side by side at about ``angle`` degrees, none quite straight or quite
    alike, each lighter where the pencil lands and where it lifts.
    ``pressure`` is a number, or a function of x (0 at the left of the box,
    1 at its right)."""
    x0, y0, x1, y1 = box
    turn = np.radians(angle)
    along = np.array([np.cos(turn), -np.sin(turn)])
    across = np.array([np.sin(turn), np.cos(turn)])
    centre = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    reach = 0.5 * np.hypot(x1 - x0, y1 - y0)
    for offset in np.arange(-reach, reach, 0.55 * width):
        line = centre + (offset + rng.normal(0, 0.15 * width)) * across
        ends = [line + s * along for s in np.linspace(-reach, reach, 200)]
        inside = [p for p in ends if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
        if len(inside) < 3:
            continue
        start, end = inside[0], inside[-1]
        knots = max(int(np.hypot(*(end - start)) / 25), 2)
        path = [
            start + (end - start) * k / knots + rng.normal(0, 0.1 * width, 2)
            for k in range(knots + 1)
        ]
        firm = rng.uniform(0.85, 1.15)

        def bearing(u, path=path, firm=firm):
            x = path[0][0] + (path[-1][0] - path[0][0]) * u
            wanted = pressure((x - x0) / (x1 - x0)) if callable(pressure) else pressure
            landing = 0.55 + 0.45 * np.sin(np.pi * np.clip(u, 0.02, 0.98)) ** 0.5
            return np.clip(wanted * firm * landing, 0.0, 1.0)

        sheet.stroke(
            path, medium, pressure=bearing, width=width * rng.uniform(0.9, 1.1)
        )


def main() -> None:
    sheet = Sheet(900, 600, paper=papers.CARTRIDGE, seed=11)
    rng = np.random.default_rng(5)
    for row, name in enumerate(GRADES):
        top = 20 + row * 46
        box = (30, top, 870, top + 36)
        shade(sheet, pencils.grade(name), lambda x: 0.05 + 0.95 * x, box, rng)
    for i, layers in enumerate((1, 2, 4, 8)):
        left = 30 + i * 112
        for layer in range(layers):
            box = (left, 360, left + 96, 575)
            shade(sheet, pencils.HB, 0.3, box, rng, angle=30 + 47 * layer)
    for k, pressure in enumerate((0.2, 0.5, 0.95)):
        wave = [
            (500 + 28 * j, 380 + 34 * k + 11 * np.sin(j * 1.2 + k)) for j in range(14)
        ]
        sheet.stroke(wave, pencils.B2, pressure=pressure)
    for slant in (1, -1):
        for x in np.arange(510, 830, 9.0):
            x += rng.normal(0, 1.0)
            sheet.stroke(
                [(x, 525 + 45 * slant), (x + 45, 525 - 45 * slant)],
                pencils.HB,
                pressure=lambda u: 0.75 - 0.45 * u,
            )
    out = Path(__file__).parent / "output" / "pencils.png"
    sheet.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
