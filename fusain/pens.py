"""Ballpoint pens: ``pens.BLUE``, ``pens.BLACK``, ``pens.RED``, or
``pens.ballpoint("#1c2f7a", width=3)``; and a pen in the hand,
:class:`HeldPen`, which keeps the ink its ball gathers.

A ballpoint is not a stick that the paper wears down: its ball rolls a
paste of dye onto the paper at a steady rate. Its line is even, about as
wide as the ball and nearly as dark at a light touch as at a firm one; the
ink is clear, so lines that cross and layers of hatching darken; and it
soaks into the paper, where no rubber eraser reaches it (a knife or a sand
eraser takes it with the surface of the paper). The ball gathers ink as it
rolls and leaves a blob where the line turns sharply and, now and then,
where it starts, the lighter and the narrower the lighter the line
(``blots=0`` for a pen that never blots). Pressed firmly it leaves a
groove, which a pencil shaded over it skips. And ink in heavy layers
bronzes under a lamp: it mirrors the light its dye absorbs most, so blue
ink shines copper; a black ink is a blend of dyes, and shines as its violet
does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from murex import ColorLike, Pigment
from murex import spectral as sp

from ._engine import BALL_INK

#: How strongly the ink hides what is under it: a dye, nearly clear.
OPACITY = 0.1


@dataclass(frozen=True)
class Pen:
    """A ballpoint pen.

    Parameters
    ----------
    ink:
        Its ink, as a :class:`murex.Pigment`.
    width:
        The width of its ball in pixels: about that of its line at a firm
        pressure. A light line is narrower.
    flow:
        How freely the ink comes: 1 for a good pen, less for one running
        dry, more for one that floods (its line is as much darker, and its
        blobs as much bigger).
    blots:
        How much ink its ball gathers, to leave as a blob where the line
        turns sharply and, now and then, where it lands: 1 for a usual pen,
        0 for one that never blots, more for a leaky one. Its line is the
        same either way.
    bronze:
        The colour heavy ink mirrors under a lamp, where it bronzes (see
        :meth:`fusain.Sheet.shine`). By default the light its ink absorbs
        most: copper for a blue ink, green for a red one, grey for a black
        made from one neutral colour. A real ink is a blend of dyes, which
        bronzes as the one that bronzes most: give its colour (only its hue
        and saturation count).
    name:
        Optional label.
    """

    ink: Pigment
    width: float = 3.0
    flow: float = 1.0
    blots: float = 1.0
    bronze: ColorLike | None = None
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.ink, Pigment):
            raise TypeError(f"ink must be a murex Pigment, got {self.ink!r}")
        for field in ("width", "flow"):
            value = getattr(self, field)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{field} must be a number above 0, got {value!r}")
        if not (math.isfinite(self.blots) and self.blots >= 0):
            raise ValueError(f"blots must be a number, 0 or above, got {self.blots!r}")
        if self.bronze is not None:
            sp.parse_color(self.bronze)  # (ValueError if it is no colour)

    @property
    def mirrors(self) -> np.ndarray:
        """The light its ink mirrors where it bronzes, in linear RGB: the
        square of its absorption, as a share of the most it absorbs (blue
        ink, which absorbs the reds, shines copper), or its ``bronze``,
        whose brightest part is 1."""
        if self.bronze is None:
            absorbed = self.ink.K / max(float(self.ink.K.max()), 1e-12)
            return np.maximum(sp.reflectance_to_linear_rgb(absorbed**2), 0.0)
        light = sp.srgb_to_linear(sp.parse_color(self.bronze))
        return light / max(float(light.max()), 1e-12)

    def with_(self, **changes) -> Pen:
        """A copy with some fields changed."""
        return replace(self, **changes)


def ballpoint(
    color: ColorLike,
    width: float = 3.0,
    name: str | None = None,
    bronze: ColorLike | None = None,
) -> Pen:
    """A ballpoint pen whose firm line on white paper is that colour (and
    whose heavy ink bronzes in the colour ``bronze``, if given: see
    :class:`Pen`)."""
    # (one firm pass leaves a film BALL_INK thick, far thinner than the
    # thickness 1 a colour is defined at)
    ink = Pigment(color, OPACITY, 1.0 / BALL_INK, name)
    return Pen(ink, width, bronze=bronze, name=name or f"{ink.hex} ballpoint")


BLUE = ballpoint("#22357e", name="blue ballpoint")
#: A black ballpoint's ink is a blend of dyes, chiefly a violet one, which is
#: what bronzes: copper with a violet cast.
BLACK = ballpoint("#26262a", name="black ballpoint", bronze="#a86478")
RED = ballpoint("#b3262b", name="red ballpoint")


class HeldPen:
    """A ballpoint pen as it is held: the pen, and the ink its ball has
    gathered on its rim, which it keeps from one stroke to the next until it
    is wiped. What one line gathers the next one lands with: now and then
    some of it comes off where it lands (as a pen just picked up leaves a
    dot), and the rest where the line first turns sharply::

        from fusain import HeldPen, Sheet, pens
        pen = HeldPen(pens.BLUE)
        sheet.stroke(path, pen)           # the ball gathers ink as it rolls
        sheet.stroke(other, pen)          # and lands with it
        sheet.shade(area, pen)            # each line with what the last left
        pen.wipe()

    Draw with a bare pen (``sheet.stroke(path, pens.BLUE)``) for one that is
    picked up afresh for every line: it lands with a drop drawn at random,
    mostly next to none, which it leaves where it lands, and takes away what
    it gathers.

    The rim of its ball holds a drop at most (``BALL_BLOB`` firm passes thick
    over the square of the ball's radius, half a pixel more than half its
    width, times the pen's ``flow`` and ``blots``: 0.48 for a pen 3 wide).
    What it would gather beyond that is not gathered, and is nowhere: not
    on the paper, not on the ball. So what its drops laid and what is on
    its ball are the share ``BALL_GATHER`` (times ``blots``) of the ink its
    lines laid only while the rim has never been full; long lines with no
    sharp turn (a hatching's) fill it, and keep it full.

    Parameters
    ----------
    pen:
        Which pen, e.g. ``pens.BLUE``.

    ``load`` is the volume of ink on the rim of its ball now: none when it is
    first taken up, never more than the drop it holds at most. It has its
    pen's width.
    """

    def __init__(self, pen: Pen):
        if not isinstance(pen, Pen):
            raise TypeError(f"pen must be a Pen, such as pens.BLUE, got {pen!r}")
        self.pen = pen
        # the ink on the rim of its ball, as the kernel keeps it
        self._rim = np.zeros(1)

    @property
    def width(self) -> float:
        """The width of its ball, in pixels."""
        return self.pen.width

    @property
    def load(self) -> float:
        """Volume of ink on the rim of its ball now: at most the drop it
        holds (see :class:`HeldPen`)."""
        return float(self._rim[0])

    def wipe(self) -> float:
        """Wipe its point on a rag: there is no ink on its rim afterwards.
        Returns the volume of ink that came off."""
        load = self.load
        self._rim[0] = 0.0
        return load

    def __repr__(self) -> str:
        name = self.pen.name or "pen"
        return f"HeldPen({name}, load={self.load:.3f})"
