"""Ballpoint pens: ``pens.BLUE``, ``pens.BLACK``, ``pens.RED``, or
``pens.ballpoint("#1c2f7a", width=3)``.

A ballpoint is not a stick that the paper wears down: its ball rolls a
paste of dye onto the paper at a steady rate. Its line is even, about as
wide as the ball and nearly as dark at a light touch as at a firm one; the
ink is clear, so lines that cross and layers of hatching darken; and it
soaks into the paper, where no eraser reaches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from murex import ColorLike, Pigment

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
        dry, more for one that blots.
    name:
        Optional label.
    """

    ink: Pigment
    width: float = 3.0
    flow: float = 1.0
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.ink, Pigment):
            raise TypeError(f"ink must be a murex Pigment, got {self.ink!r}")
        for field in ("width", "flow"):
            value = getattr(self, field)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{field} must be a number above 0, got {value!r}")

    def with_(self, **changes) -> Pen:
        """A copy with some fields changed."""
        return replace(self, **changes)


def ballpoint(color: ColorLike, width: float = 3.0, name: str | None = None) -> Pen:
    """A ballpoint pen whose firm line on white paper is that colour."""
    # (one firm pass leaves a film BALL_INK thick, far thinner than the
    # thickness 1 a colour is defined at)
    ink = Pigment(color, OPACITY, 1.0 / BALL_INK, name)
    return Pen(ink, width, name=name or f"{ink.hex} ballpoint")


BLUE = ballpoint("#22357e", name="blue ballpoint")
BLACK = ballpoint("#26262a", name="black ballpoint")
RED = ballpoint("#b3262b", name="red ballpoint")
