"""Wax crayons, of any colour: ``crayons.wax("#c0392b")``.

A crayon is pigment in hard wax. The paper holds all of what it lays; the
wax makes the tooth slick at once, so the grain of the paper shows through
until the crayon is pressed hard; and it is not opaque enough to cover: a
light crayon over a dark one mixes with it.
"""

from __future__ import annotations

from murex import ColorLike, Pigment

from .medium import Medium

#: How strongly a crayon hides what is under it, and how much stronger it is
#: than the thickness 1 a colour is defined at (a full tooth is thinner).
OPACITY = 0.65
STRENGTH = 2.5


def wax(color: ColorLike, name: str | None = None) -> Medium:
    """A wax crayon of that colour: how a full layer of it looks on white
    paper."""
    pigment = Pigment(color, OPACITY, STRENGTH, name)
    label = name or f"{pigment.hex} crayon"
    return Medium(pigment, softness=0.6, binding=1.0, wax=0.85, name=label)
