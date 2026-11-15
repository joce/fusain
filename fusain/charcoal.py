"""Charcoal: ``charcoal.VINE``, ``charcoal.COMPRESSED``, ``charcoal.PENCIL``.

Charcoal is burnt wood, with little or nothing to bind it: most of what it
lays stays on the paper as dust, which a finger or a stump moves and a
kneaded eraser lifts.
"""

from __future__ import annotations

from murex import Pigment

from .medium import Medium

#: Thickness 1 is about a tooth full of it, as for a pencil lead.
CARBON = Pigment("#1d1c1c", strength=0.12, name="charcoal")

#: Vine (willow) charcoal: a burnt twig. Soft and grey; nearly all of it is
#: dust, so it moves at a touch and lifts almost clean. A piece of twig laid
#: flat marks a band eight times as wide as its end, even all across.
VINE = Medium(
    Pigment(CARBON, strength=0.35, name="vine charcoal"),
    softness=0.6,
    binding=0.12,
    side=8.0,
    flat_side=True,
    name="vine charcoal",
)
#: Compressed charcoal: powder pressed with a little gum. Much blacker, and
#: more of it stays where it is put. A square stick, laid flat on a side six
#: times as long as it is thick.
COMPRESSED = Medium(
    Pigment(CARBON, strength=2.0, name="compressed charcoal"),
    softness=0.65,
    binding=0.45,
    side=6.0,
    flat_side=True,
    name="compressed charcoal",
)
#: A charcoal pencil: compressed charcoal with more binder, in wood. Harder,
#: for lines; its side is the cone its sharpening bared, as a pencil's is.
PENCIL = Medium(
    Pigment(CARBON, strength=1.5, name="charcoal pencil"),
    softness=0.55,
    binding=0.6,
    name="charcoal pencil",
)
