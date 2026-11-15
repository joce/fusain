"""A fingerprint of what the engine draws, to tell whether a change to the
code changes the picture.

A fixed suite of drawings (every family of medium, sticks and tilt, width
profiles and flat ends, shading, tools bare and in the hand, a knife, a
blender, fixative, a stylus, ballpoint pens bare and in the hand, on several
papers and seeds) is drawn on small sheets, and every array a sheet, a
smudger, a stick and a pen in the hand hold
is hashed, with the numbers the calls return and the rendered images. The
hashes are compared with those recorded in ``tools/fingerprint.json``::

    python tools/fingerprint.py              # compare with the recorded ones
    python tools/fingerprint.py --thorough --save before.json
    python tools/fingerprint.py --thorough --against before.json
    python tools/fingerprint.py --record     # the picture is meant to change

A change that is not meant to alter the picture (an optimisation, a
refactoring) must leave every hash as it is. ``--thorough`` draws the suite
on every paper with several seeds: save its hashes before such a change and
compare after. Record again only with a change that is meant to alter the
picture. ``tests/test_fingerprint.py`` runs the recorded suite.

Hashes are those of the machine they were recorded on: another platform's
``pow`` or numpy's ``exp`` may round a last bit differently. The file
therefore also records a hash of what that arithmetic gives
(:func:`arithmetic`); where it differs, the recorded hashes say nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import numba
import numpy as np
from numba import njit

from fusain import (
    HeldPen,
    Paper,
    Sheet,
    Smudger,
    Stick,
    charcoal,
    crayons,
    papers,
    pastels,
    pencils,
    pens,
    tools,
)

REFERENCE = Path(__file__).with_name("fingerprint.json")
#: The size of the sheets, in pixels: small, so that the suite is quick.
W, H = 176, 128
#: Renders that are hashed, as they are saved: under the default lamp, and
#: under a raking one with the relief and the gloss brought out.
RENDERS: tuple[tuple[str, dict], ...] = (
    ("image", {}),
    ("raking", {"light": (-0.6, -0.8, 0.5), "relief": 1.5, "gloss": 2.0}),
)
#: A lamp overhead, under which heavy ink bronzes.
OVERHEAD = ("overhead", {"light": (0.05, -0.05, 1.0)})
#: A paper with no tooth at all, which the kernels treat apart.
LEVEL = Paper("wove", grain=2.0, depth=0.0, bite=0.4, name="level")
RED, YELLOW, BLUE, GREEN = "#c0392b", "#f2c40f", "#2456a6", "#2e8b57"
DARK, PALE = "#141416", "#f4e9d0"


def digest(array: np.ndarray) -> str:
    """A hash of an array: its type, its shape and every bit of it."""
    array = np.ascontiguousarray(array)
    mark = hashlib.sha256(f"{array.dtype}{array.shape}".encode())
    mark.update(array.tobytes())
    return mark.hexdigest()[:16]


class Trace:
    """The marks of one drawing: a name and a hash for each thing looked at."""

    def __init__(self) -> None:
        self.marks: dict[str, str] = {}
        self._returned: list[float] = []

    def got(self, value: float | None) -> None:
        """Note a number that a call returned (they are hashed together)."""
        self._returned.append(math.nan if value is None else float(value))

    def state(self, name: str, thing: object) -> None:
        """Hash all that ``thing`` (a sheet, a smudger, a stick) holds: every
        array and every number among its attributes, whatever they are
        called, and the state of its random generator."""
        for key, value in sorted(vars(thing).items()):
            if isinstance(value, np.ndarray):
                self.marks[f"{name}.{key}"] = digest(value)
            elif isinstance(value, (int, float)):
                self.marks[f"{name}.{key}"] = digest(np.array(value, dtype=np.float64))
            elif isinstance(value, np.random.Generator):
                state = json.dumps(value.bit_generator.state, sort_keys=True)
                self.marks[f"{name}.{key}"] = hashlib.sha256(
                    state.encode()
                ).hexdigest()[:16]

    def look(self, name: str, sheet: Sheet, *more: tuple[str, dict]) -> None:
        """Hash a sheet: all it holds, and what it shows (rendered as
        :data:`RENDERS` say, and ``more``)."""
        self.state(name, sheet)
        self.marks[f"{name}:reflectance"] = digest(sheet.reflectance())
        self.marks[f"{name}:shine"] = digest(sheet.shine())
        for label, options in (*RENDERS, *more):
            image = np.round(sheet.render(**options) * 255).astype(np.uint8)
            self.marks[f"{name}:{label}"] = digest(image)

    def close(self) -> dict[str, str]:
        """The marks, with that of the numbers returned."""
        self.marks["returned"] = digest(np.array(self._returned, dtype=np.float64))
        return self.marks


def _wave(y: float, x0: float = 12.0, x1: float = W - 12.0, swing: float = 6.0):
    """A path across the sheet at height ``y`` that swings up and down."""
    xs = np.linspace(x0, x1, 6)
    return [(x, y + swing * math.sin(1.7 * k)) for k, x in enumerate(xs)]


def _zigzag(x0: float, y0: float, x1: float, y1: float, turns: int):
    """A path that turns sharply ``turns`` times between two heights."""
    xs = np.linspace(x0, x1, turns + 2)
    return [(x, y0 if k % 2 else y1) for k, x in enumerate(xs)]


def _fading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A pressure over the sheet: firm on the left, nothing on the right."""
    return np.clip(1.1 - 1.3 * x / W + 0.0 * y, 0.0, 1.0)


def graphite(trace: Trace, sheet: Sheet) -> None:
    """Pencil leads from hard to soft: lines, dots, profiles, shading."""
    for row, lead in enumerate((pencils.H4, pencils.HB, pencils.B2, pencils.B8)):
        y = 10.0 + 9.0 * row
        trace.got(sheet.stroke(_wave(y), lead, 0.3 + 0.2 * row, width=2.0 + row))
        trace.got(sheet.stroke([(8.0 + 5 * row, 50.0)], lead, 0.8, width=3.0 + row))
    # pressure and width that change along the stroke; a tip that lifts
    # in the middle of its path; one that leaves the sheet; a short one
    trace.got(sheet.stroke(_wave(48.0), pencils.B4, [0.1, 0.9, 0.2, 1.0, 0.5, 0.0]))
    trace.got(
        sheet.stroke(
            _wave(56.0),
            pencils.B2,
            lambda u: np.where(np.abs(u - 0.5) < 0.1, 0.0, 0.7),
            width=lambda u: 2.0 + 5.0 * u,
        )
    )
    trace.got(sheet.stroke([(150.0, 60.0), (W + 9.0, 40.0)], pencils.B6, 0.9, width=7))
    trace.got(sheet.stroke([(-6.0, 70.0), (30.0, -4.0)], pencils.HB, 0.6, width=5))
    trace.got(sheet.stroke([(60.0, 62.0), (60.6, 62.2)], pencils.B4, 0.7, width=6))
    trace.got(sheet.stroke(_wave(66.0), pencils.H4, 1.0, width=3, ends="flat"))
    # one that never touches the sheet, and one that never bears on it
    trace.got(sheet.stroke([(200.0, -30.0), (260.0, -20.0)], pencils.B2, 0.7, 5))
    trace.got(sheet.stroke(_wave(30.0), pencils.B2, 0.0))
    # the benchmark of the issue, small: an HB of width 6, two layers
    trace.got(sheet.shade((10, 74, 100, 122), pencils.HB, 0.5))
    trace.got(sheet.shade((10, 74, 100, 122), pencils.HB, 0.4, angle=-40))
    corners = [(108, 76), (168, 80), (160, 124), (112, 116)]
    trace.got(sheet.shade(corners, pencils.B6, _fading, width=5, angle=75, hand=0.0))
    trace.got(sheet.shade((104, 70, 172, 126), pencils.H2, 0.95, width=4, hand=2.0))
    trace.look("sheet", sheet)


def sticks(trace: Trace, sheet: Sheet) -> None:
    """Sticks that wear, held upright and on their side; ends cut square."""
    pencil = Stick(pencils.B2, width=2)
    for row in range(4):
        trace.got(sheet.stroke(_wave(8.0 + 5 * row), pencil, 0.8, tilt=row / 3))
    trace.state("pencil", pencil)
    trace.got(sheet.shade((8, 30, 80, 70), pencil, 0.7))
    pencil.sharpen(3.0)
    trace.got(sheet.shade((8, 30, 80, 70), pencil, 0.5, angle=110, tilt=0.6))
    twig = Stick(charcoal.VINE, width=6, lead=6, side=40)
    trace.got(sheet.stroke([(90, 50), (168, 46)], twig, 0.6, tilt=1.0, ends="flat"))
    trace.got(
        sheet.stroke(_wave(84.0), twig, 0.7, tilt=lambda u: u, ends=("flat", "round"))
    )
    chalk = Stick(pastels.hard(RED), width=5, lead=5)
    trace.got(sheet.stroke(_wave(100.0), chalk, 0.8, tilt=[0, 1, 0, 1, 0, 1]))
    trace.got(sheet.shade((90, 8, 170, 40), chalk, 0.6, tilt=1.0, ends="flat"))
    # bare media with a width and a tilt of their own along the stroke
    trace.got(
        sheet.stroke(
            _wave(114.0),
            charcoal.COMPRESSED,
            0.6,
            width=lambda u: 3.0 + 6.0 * (1.0 - u),
            tilt=lambda u: 0.5 * u,
            ends=("round", "flat"),
        )
    )
    trace.got(sheet.stroke([(20, 122), (150, 120)], pencils.B6, 0.5, 5, 1.0, "flat"))
    trace.got(sheet.shade((100, 60, 170, 80), pastels.soft(BLUE), 0.5, 8, tilt=0.8))
    for name, stick in (("pencil", pencil), ("twig", twig), ("chalk", chalk)):
        trace.state(f"{name}:end", stick)
    trace.look("sheet", sheet)


def dust(trace: Trace, sheet: Sheet) -> None:
    """Charcoal, and the tools that move it and take it off."""
    trace.got(sheet.shade((8, 8, 168, 60), charcoal.VINE, 0.6, width=8))
    trace.got(
        sheet.shade((8, 8, 168, 60), charcoal.COMPRESSED, 0.5, width=6, angle=-50)
    )
    trace.got(sheet.shade((8, 66, 168, 120), charcoal.PENCIL, 0.7, width=4))
    trace.got(sheet.shade((8, 66, 168, 120), pencils.B6, 0.8, width=6, angle=100))
    trace.got(sheet.smudge(_wave(20.0), tools.STUMP, 0.7))
    trace.got(sheet.smudge(_wave(34.0), tools.FINGER, [0.2, 0.9, 0.0, 0.6, 1.0, 0.3]))
    trace.got(sheet.smudge(_wave(48.0), tools.TORTILLON, 0.9, width=4))
    trace.got(sheet.smudge([(40.0, 90.0)], tools.FINGER, 0.8))
    trace.got(sheet.smudge([(150, 100), (W + 12, 90)], tools.CHAMOIS, 0.6))
    trace.got(sheet.erase(_wave(76.0), pressure=0.8))
    trace.got(sheet.erase(_wave(90.0), tools.ERASER, 0.9, width=7))
    trace.state("rubbed", sheet)
    finger, stump = Smudger(tools.FINGER), Smudger(tools.STUMP, 8)
    trace.got(sheet.rub((8, 8, 90, 60), finger, 0.6))
    trace.got(sheet.rub((60, 30, 168, 110), stump, _fading, angle=40, hand=0.5))
    trace.state("finger", finger)
    trace.got(sheet.smudge([(100, 124), (170, 122)], finger, 0.9))  # on bare paper
    trace.got(sheet.rub((8, 70, 60, 120), tools.CHAMOIS, 0.5, spacing=9))
    trace.got(sheet.rub((100, 8, 168, 40), tools.KNEADED, 0.7, width=9, angle=90))
    trace.got(sheet.shade((40, 40, 130, 100), charcoal.VINE, 0.5, width=7, angle=10))
    trace.got(finger.clean())
    trace.got(sheet.rub((40, 40, 130, 100), finger, 0.4))
    for name, smudger in (("finger", finger), ("stump", stump)):
        trace.state(f"{name}:end", smudger)
    trace.look("sheet", sheet)


def pastel(trace: Trace, sheet: Sheet) -> None:
    """Soft, hard and oil pastels in layers, fixative, and tools on paste."""
    trace.got(sheet.shade((8, 8, 84, 60), pastels.soft(DARK), 0.6, width=9))
    trace.got(sheet.shade((8, 8, 84, 60), pastels.soft(PALE), 0.7, width=9, angle=80))
    trace.got(sheet.shade((92, 8, 168, 60), pastels.hard(BLUE), 0.6, width=5))
    trace.got(sheet.fix(0.6))
    trace.state("fixed", sheet)
    trace.got(sheet.shade((40, 20, 130, 50), pastels.soft(RED), 0.6, width=8, angle=0))
    trace.got(sheet.smudge(_wave(30.0), tools.FINGER, 0.7))
    trace.got(sheet.erase(_wave(44.0), tools.KNEADED, 0.9))
    trace.got(sheet.fix())
    trace.got(sheet.shade((40, 20, 130, 50), pastels.soft(YELLOW), 0.5, width=8))
    # oil pastel: paste, which a light one ploughs into and a finger smears
    trace.got(sheet.shade((8, 68, 100, 120), pastels.oil(DARK), 0.7, width=8))
    trace.got(sheet.shade((8, 68, 100, 120), pastels.oil(YELLOW), 0.8, 8, angle=-30))
    trace.got(sheet.shade((70, 68, 168, 120), pastels.oil(RED), 0.95, width=10))
    trace.got(sheet.smudge(_wave(80.0), tools.FINGER, 0.8))
    trace.got(sheet.smudge(_wave(92.0), tools.STUMP, 0.8))
    trace.got(sheet.erase(_wave(104.0), tools.KNEADED, 0.9))
    trace.got(sheet.erase(_wave(114.0), tools.ERASER, 0.9))
    # dry media over the grease, and dust under it
    trace.got(sheet.shade((60, 50, 168, 90), pastels.soft(GREEN), 0.6, width=7))
    trace.got(sheet.shade((100, 80, 168, 124), charcoal.VINE, 0.6, width=7))
    trace.got(sheet.shade((100, 80, 168, 124), pastels.oil(PALE), 0.7, width=8))
    trace.got(sheet.rub((90, 60, 168, 124), Smudger(tools.FINGER, 16), 0.6))
    trace.look("sheet", sheet)


def wax(trace: Trace, sheet: Sheet) -> None:
    """Coloured pencils and crayons: films, resist, burnishing, a blender,
    erasers on wax, and a knife."""
    red, yellow = pencils.coloured(RED), pencils.coloured(YELLOW)
    blue, dark = pencils.coloured(BLUE), pencils.coloured(DARK)
    for layer in range(3):
        angle = 30.0 - 55.0 * layer
        trace.got(sheet.shade((8, 8, 110, 70), yellow, 0.4 + 0.1 * layer, 6, angle))
    trace.got(sheet.shade((40, 8, 168, 50), blue, 0.5, width=5))
    trace.got(sheet.shade((8, 8, 80, 40), red, 0.95, width=6, angle=60))  # burnished
    trace.got(sheet.shade((60, 30, 168, 70), dark, 0.7, width=5, angle=-20))
    trace.state("layered", sheet)
    trace.got(sheet.shade((8, 8, 120, 70), pencils.BLENDER, 0.9, width=7))
    trace.got(sheet.stroke(_wave(60.0), pencils.BLENDER, 1.0, width=9))
    # (what a blender drags it leaves where it lifts: here, off the sheet)
    trace.got(sheet.stroke([(120, 30), (W + 12, 34)], pencils.BLENDER, 1.0, 8))
    trace.got(sheet.stroke([(30.0, 30.0)], pencils.BLENDER, 1.0, width=8))
    trace.got(sheet.erase(_wave(16.0), tools.ERASER, 0.9))
    trace.got(sheet.erase(_wave(28.0), tools.KNEADED, 0.9))
    trace.got(sheet.erase(_wave(40.0), tools.SAND_ERASER, 0.9))
    trace.got(sheet.erase(_zigzag(20, 12, 150, 64, 5), tools.KNIFE, 0.9))
    trace.got(sheet.rub((120, 40, 168, 70), tools.SAND_ERASER, 0.8))
    # crayons: resist, a film over dust, charcoal over a line of wax
    trace.got(sheet.shade((8, 76, 90, 122), charcoal.VINE, 0.6, width=7))
    trace.got(sheet.shade((8, 76, 120, 122), crayons.wax(RED), 0.7, width=8))
    trace.got(sheet.shade((60, 76, 168, 122), crayons.wax(BLUE), 0.95, 8, angle=-45))
    trace.got(sheet.stroke(_wave(100.0), crayons.wax(PALE), 0.9, width=6))
    trace.got(sheet.shade((90, 84, 168, 122), charcoal.COMPRESSED, 0.6, width=7))
    trace.got(sheet.shade((90, 84, 168, 122), pencils.B4, 0.7, width=5, angle=120))
    trace.got(sheet.smudge(_wave(110.0), tools.FINGER, 0.8))
    trace.got(sheet.erase(_wave(118.0, swing=3.0), tools.ERASER, 0.9))
    trace.got(sheet.erase(_zigzag(16, 80, 160, 120, 3), tools.KNIFE, 1.0, width=3))
    trace.look("sheet", sheet)


def surface(trace: Trace, sheet: Sheet) -> None:
    """The paper itself: grooves, a tooth pressed flat, and what skips them."""
    for row, pressure in enumerate((0.2, 0.5, 0.8, 1.0)):
        sheet.indent(_wave(14.0 + 12.0 * row), pressure, width=2.0 + row)
    sheet.indent([(30.0, 100.0)], 0.9, width=6)
    sheet.indent(_zigzag(-8, 70, W + 8, 126, 4), lambda u: 0.9 * (1.0 - u), width=3)
    trace.got(sheet.shade((8, 8, 168, 62), pencils.HB, 0.5))
    trace.got(sheet.shade((8, 8, 90, 62), pencils.B8, 0.95, width=7, angle=-30))
    # a hard lead pressed firmly flattens the tooth; what comes after lies flat
    trace.got(sheet.shade((8, 68, 168, 124), pencils.H6, 1.0, width=5))
    trace.state("pressed", sheet)
    trace.got(sheet.shade((8, 68, 168, 124), pencils.B4, 0.6, width=6, angle=100))
    trace.got(sheet.shade((90, 30, 168, 100), pastels.soft(RED), 0.5, width=8))
    trace.got(sheet.shade((40, 40, 120, 110), crayons.wax(GREEN), 0.9, width=7))
    trace.got(sheet.smudge(_wave(50.0), tools.STUMP, 0.8))
    trace.got(sheet.erase(_wave(90.0), tools.ERASER, 0.9))
    sheet.indent(_wave(104.0), 0.7)  # through what is drawn
    trace.got(sheet.stroke(_wave(104.0), pencils.B6, 0.9, width=9))
    twin = sheet.copy()  # (drawing on one leaves the other alone)
    trace.got(twin.shade((8, 8, 168, 124), charcoal.COMPRESSED, 0.6, width=8))
    trace.got(sheet.shade((8, 8, 168, 124), charcoal.COMPRESSED, 0.6, width=8))
    trace.state("twin", twin)
    trace.look("sheet", sheet)


def ballpoint(trace: Trace, sheet: Sheet) -> None:
    """Ballpoint pens: lines, blobs at the turns, hatching, skidding, grooves;
    a pen that never blots, and one in the hand, which lands with what it
    gathered."""
    for row, pressure in enumerate((0.05, 0.3, 0.6, 1.0)):
        trace.got(sheet.stroke(_wave(10.0 + 7.0 * row), pens.BLUE, pressure))
    for row in range(6):  # (now and then a line starts with a blob)
        y = 40.0 + 3.0 * row
        trace.got(sheet.stroke([(10, y), (60, y)], pens.BLACK, 0.5 + 0.1 * row))
    trace.got(sheet.stroke(_zigzag(66, 36, 168, 60, 9), pens.RED, 0.8))
    trace.got(sheet.stroke([(120.0, 12.0)], pens.BLACK.with_(flow=2.5), 0.9))
    trace.got(sheet.stroke([(W + 1.0, 30.0)], pens.BLACK.with_(flow=2.5), 0.9))
    trace.got(sheet.stroke([(150, 20), (W + 10, 6)], pens.ballpoint(GREEN, 5), 0.7))
    trace.got(
        sheet.stroke(_wave(30.0), pens.BLUE, lambda u: np.where(u % 0.25 < 0.1, 0, 0.8))
    )
    trace.got(sheet.stroke(_wave(62.0, 66.0), pens.RED.with_(blots=0.0), 0.9))
    held = HeldPen(pens.BLACK.with_(blots=1.5))
    for row in range(3):  # (a long line, then lines and turns that it lands on)
        trace.got(sheet.stroke(_zigzag(70, 4 + 9 * row, 160, 10 + 9 * row, row), held))
    trace.got(sheet.shade((100, 66, 168, 90), held, 0.7, angle=80))
    trace.state("pen", held)
    trace.got(sheet.shade((8, 66, 90, 122), pens.BLUE, 0.6))
    trace.got(sheet.shade((8, 66, 90, 122), pens.BLUE, 0.9, angle=-40))
    for _ in range(4):  # (heavy ink bronzes)
        trace.got(sheet.shade((8, 100, 50, 122), pens.BLACK, 1.0, angle=70))
    trace.state("inked", sheet)
    # what the pen skids on, and what skips its grooves or covers its ink
    trace.got(sheet.shade((96, 66, 168, 122), pencils.B6, 0.8, width=6))
    trace.got(sheet.shade((96, 96, 168, 122), crayons.wax(YELLOW), 0.8, width=7))
    trace.got(sheet.shade((60, 70, 168, 118), pens.RED, 0.7, angle=10))
    trace.got(sheet.shade((40, 10, 120, 60), pencils.B2, 0.5, width=6, angle=120))
    trace.got(sheet.erase(_wave(52.0), tools.ERASER, 0.9))
    trace.got(sheet.smudge(_wave(84.0), tools.FINGER, 0.8))
    trace.look("sheet", sheet, OVERHEAD)


def drawing(trace: Trace, sheet: Sheet) -> None:
    """A little of everything on one sheet, as a drawing mixes them."""
    stump, pencil = Smudger(tools.STUMP), Stick(pencils.B4, width=3)
    sheet.indent(_zigzag(10, 10, 166, 40, 6), 0.8, width=2)
    trace.got(sheet.shade((0, 0, W, 50), charcoal.VINE, _fading, width=9, tilt=0.5))
    trace.got(sheet.rub((0, 0, W, 50), stump, 0.6))
    trace.got(sheet.fix(0.8))
    trace.got(sheet.shade((0, 30, W, 90), pencil, 0.6, angle=15))
    trace.got(sheet.shade((20, 40, 150, 120), pencils.coloured(BLUE), 0.6, width=5))
    trace.got(sheet.shade((20, 60, 150, 120), pastels.oil(YELLOW), 0.7, width=8))
    trace.got(sheet.smudge(_wave(70.0), stump, 0.8))
    trace.got(sheet.shade((60, 20, 140, 100), pencils.BLENDER, 0.9, width=7))
    trace.got(sheet.stroke(_zigzag(8, 60, 168, 124, 7), pens.BLACK, 0.8))
    trace.got(sheet.shade((90, 70, 170, 126), pens.BLUE, 0.7, angle=-60))
    trace.got(sheet.erase(_wave(100.0), tools.ERASER, 0.9))
    trace.got(sheet.erase(_zigzag(30, 50, 140, 110, 2), tools.KNIFE, 0.9))
    trace.got(sheet.rub((100, 0, W, 60), tools.KNEADED, _fading, angle=90))
    trace.got(sheet.shade((0, 90, 90, H), pastels.soft(RED), 0.5, 8, ends="flat"))
    trace.got(sheet.smudge(_wave(110.0), stump, 0.9))
    trace.got(sheet.stroke(_wave(20.0), pencil, 0.9, tilt=1.0))
    trace.state("stump", stump)
    trace.state("pencil", pencil)
    trace.look("sheet", sheet)


Scene = Callable[[Trace, Sheet], None]
#: The suite: each drawing, and the papers and seeds it is recorded on.
SUITE: tuple[tuple[Scene, tuple[tuple[Paper, int], ...]], ...] = (
    (graphite, ((papers.CARTRIDGE, 1), (LEVEL, 2))),
    (sticks, ((papers.INGRES, 3),)),
    (dust, ((papers.CHARCOAL, 4), (papers.ROUGH, 5))),
    (pastel, ((papers.SANDED, 6),)),
    (wax, ((papers.CARTRIDGE, 7), (papers.COLD_PRESSED, 8))),
    (surface, ((papers.BRISTOL, 9), (LEVEL, 10))),
    (ballpoint, ((papers.PRINTER, 11), (papers.NEWSPRINT, 12))),
    (drawing, ((papers.TONED_GREY, 13),)),
)
#: With ``--thorough``: every drawing on every paper, with these seeds.
THOROUGH_PAPERS = (*papers.ALL, LEVEL)
THOROUGH_SEEDS = (21, 22, 23)


def runs(thorough: bool = False) -> Iterator[tuple[str, Scene, Paper, int]]:
    """The drawings of the suite: a name, and what to draw on what."""
    for scene, recorded in SUITE:
        on = recorded
        if thorough:
            extra = [(p, s) for p in THOROUGH_PAPERS for s in THOROUGH_SEEDS]
            on = (*recorded, *extra)
        for paper, seed in on:
            yield f"{scene.__name__}/{paper.name}/{seed}", scene, paper, seed


def draw(scene: Scene, paper: Paper, seed: int) -> dict[str, str]:
    """Draw one drawing of the suite. Returns its marks."""
    trace = Trace()
    scene(trace, Sheet(W, H, paper=paper, seed=seed))
    return trace.close()


def fingerprint(thorough: bool = False) -> dict[str, dict[str, str]]:
    """Draw the suite. Returns the marks of each drawing, by name."""
    return {
        name: draw(scene, paper, seed) for name, scene, paper, seed in runs(thorough)
    }


@njit(error_model="numpy")
def _compiled(x, out):  # pragma: no cover (compiled)
    """What the kernels take from the platform's maths library."""
    for i in range(x.shape[0]):
        out[0, i] = x[i] ** 0.6
        out[1, i] = x[i] ** 0.3
        out[2, i] = x[i] ** 5.0
        out[3, i] = x[i] ** 6.0
        out[4, i] = x[i] ** 1.5
        out[5, i] = (x[i] * 300.0) ** (1.0 / 3.0)
        out[6, i] = math.sqrt(x[i])
        out[7, i] = math.atan2(x[i], 1.0 - 2.0 * x[i])


def arithmetic() -> str:
    """A hash of what this machine's arithmetic gives for the functions the
    drawing depends on, at fixed numbers: the compiled kernels' powers,
    roots and arc tangents, numpy's elementary functions in both precisions,
    and Python's powers. Two machines that give the same draw the same."""
    x = np.random.default_rng(0).random(20000)
    out = np.empty((8, x.size))
    _compiled(x, out)
    mark = hashlib.sha256(out.tobytes())
    for kind in (np.float64, np.float32):
        v = x.astype(kind)
        wide = (v * 12 - 6).astype(kind)
        for value in (
            np.exp(-wide),
            np.sin(wide),
            np.cos(wide),
            np.tanh(wide),
            np.sqrt(v),
            np.cbrt(v),
            v ** kind(1 / 2.4),
            v ** kind(2.4),
            v ** kind(12.0),
            np.arctan2(v, wide),
            np.hypot(v, wide),
        ):
            mark.update(np.ascontiguousarray(value).tobytes())
    powers = [float(t) ** (1.0 / 3.0) for t in x[:2000]]
    mark.update(np.array(powers).tobytes())
    return mark.hexdigest()[:16]


def where() -> str:
    """What the hashes were computed on, for the record."""
    return (
        f"{platform.system()} {platform.machine()}, Python "
        f"{platform.python_version()}, numpy {np.__version__}, "
        f"numba {numba.__version__}"
    )


def differences(
    marks: dict[str, dict[str, str]], recorded: dict[str, dict[str, str]]
) -> list[str]:
    """What differs between two fingerprints, one line for each mark."""
    lines = [f"{name}: not recorded" for name in marks if name not in recorded]
    lines += [f"{name}: not drawn" for name in recorded if name not in marks]
    for name in marks.keys() & recorded.keys():
        new, old = marks[name], recorded[name]
        lines += [
            f"{name}: {key} differs"
            for key in sorted(new.keys() | old.keys())
            if new.get(key) != old.get(key)
        ]
    return sorted(lines)


def load(path: Path = REFERENCE) -> dict:
    """A recorded fingerprint: {"arithmetic", "recorded on", "marks"}."""
    return json.loads(path.read_text(encoding="utf-8"))


def save(marks: dict[str, dict[str, str]], path: Path) -> None:
    """Record a fingerprint, with the arithmetic it was computed with."""
    record = {"arithmetic": arithmetic(), "recorded on": where(), "marks": marks}
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", "utf-8")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--thorough",
        action="store_true",
        help="draw the suite on every paper, with several seeds (not recorded)",
    )
    parser.add_argument("--save", type=Path, help="write the hashes to this file")
    parser.add_argument(
        "--against", type=Path, help="compare with this file, not the recorded one"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=f"record the hashes in {REFERENCE.name}: the picture is meant to change",
    )
    options = parser.parse_args(arguments)
    if options.record and options.thorough:
        parser.error("the thorough suite is not recorded: --save it to a file")
    marks = fingerprint(options.thorough)
    target = REFERENCE if options.record else options.save
    if target is not None:
        save(marks, target)
        print(f"{len(marks)} drawings recorded in {target}")
        return 0
    if options.thorough and options.against is None:
        parser.error("give --save FILE or --against FILE with --thorough")
    recorded = load(options.against or REFERENCE)
    if recorded["arithmetic"] != arithmetic():
        print(
            "This machine's arithmetic is not that of the recorded hashes "
            f"({recorded['recorded on']}): they say nothing here."
        )
    wanted = recorded["marks"]
    differing = differences(marks, wanted)
    for line in differing:
        print(line)
    if differing:
        print(f"{len(differing)} marks differ: the picture has changed.")
        return 1
    print(f"{len(marks)} drawings, {sum(map(len, marks.values()))} marks: identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
