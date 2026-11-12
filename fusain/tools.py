"""Tools that lay nothing of their own: stumps and fingers that move what
is on the paper, and erasers that take it off. Used with
:meth:`fusain.Sheet.smudge` and :meth:`fusain.Sheet.erase`."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Tool:
    """What a tool does where it bears fully on the paper, in one pass.

    Parameters
    ----------
    lift_loose, lift_held:
        How strongly it lifts the dust, and the pigment the paper holds
        (0 = not at all; 1 takes about two thirds in a pass; more, nearly
        all).
    lay:
        The share of what it carries that it lays back down over each of its
        own widths (0 for an eraser, which keeps it).
    press:
        How strongly it presses the dust into the paper, where it is held.
    smear:
        How strongly it pushes the dust on the high ground into the low,
        which greys the white specks of the paper.
    width:
        Its usual width in pixels.
    name:
        Optional label.
    """

    lift_loose: float
    lift_held: float = 0.0
    lay: float = 0.0
    press: float = 0.0
    smear: float = 0.0
    width: float = 12.0
    name: str = ""

    def __post_init__(self):
        for field in ("lift_loose", "lift_held", "lay", "press", "smear"):
            value = getattr(self, field)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{field} must be 0 or more, got {value!r}")
        if self.lay > 1:
            raise ValueError(f"lay is a share, at most 1, got {self.lay!r}")
        if not (math.isfinite(self.width) and self.width > 0):
            raise ValueError(
                f"width must be a number of pixels above 0: {self.width!r}"
            )

    def with_(self, **changes) -> Tool:
        """A copy with some fields changed."""
        return replace(self, **changes)


#: A paper stump: firm and pointed. It drags the dust along and rubs it in.
STUMP = Tool(1.5, 0.02, lay=0.7, press=0.5, smear=2.0, width=10.0, name="stump")
#: A tortillon: a small, hard stump for details.
TORTILLON = STUMP.with_(width=5.0, name="tortillon")
#: A finger: broad and soft. It spreads the dust and presses it in.
FINGER = Tool(1.2, 0.01, lay=0.5, press=0.8, smear=2.5, width=22.0, name="finger")
#: A chamois: it wipes most of the dust away and spreads the rest thin.
CHAMOIS = Tool(1.6, 0.04, lay=0.12, press=0.1, smear=1.0, width=30.0, name="chamois")
#: A kneaded eraser: it lifts the dust, and a little of what is held.
KNEADED = Tool(2.5, 0.3, width=12.0, name="kneaded eraser")
#: A vinyl eraser: it rubs off nearly everything.
ERASER = Tool(4.0, 2.5, width=10.0, name="vinyl eraser")
