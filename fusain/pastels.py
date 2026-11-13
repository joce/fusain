"""Pastels, of any colour: ``pastels.soft("#c0392b")``, ``pastels.hard(...)``.

A pastel is pigment with barely enough gum to hold a stick together. It is
opaque, so a light one covers a dark one, and nearly all of what it lays is
dust, which a finger blends.
"""

from __future__ import annotations

from murex import ColorLike, Pigment

from .medium import Medium


def soft(color: ColorLike, name: str | None = None) -> Medium:
    """A soft pastel of that colour: crumbly, it fills the tooth fast and
    leaves almost nothing but dust."""
    pigment = Pigment(color, name=name)
    return Medium(
        pigment, softness=0.95, binding=0.15, name=name or f"{pigment.hex} pastel"
    )


def hard(color: ColorLike, name: str | None = None) -> Medium:
    """A hard pastel of that colour: more gum in it, for lines and first
    layers. It lays less, and more of it stays where it is put."""
    pigment = Pigment(color, name=name)
    label = name or f"{pigment.hex} hard pastel"
    return Medium(pigment, softness=0.6, binding=0.35, name=label)
