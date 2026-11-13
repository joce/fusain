"""Pencils. Graphite, by grade: ``pencils.HB``, ``pencils.B6`` (a 6B),
``pencils.H2`` (a 2H), or ``pencils.grade("6B")``. Coloured, of any colour:
``pencils.coloured("#c0392b")``; and ``pencils.BLENDER``, a colourless one.

A graphite lead is graphite fired with clay. More clay makes it harder: it
wears less, stays on the peaks of the tooth and leaves a paler grey. A
coloured lead is pigment in wax: translucent, so that layers of two colours
mix to the eye, and waxy, so that the tooth turns slick as it fills.
"""

from __future__ import annotations

from murex import ColorLike, Pigment

from .medium import Medium

#: The grades from hardest to softest.
GRADES = (
    *(f"{n}H" for n in range(9, 1, -1)),
    "H",
    "F",
    "HB",
    "B",
    *(f"{n}B" for n in range(2, 10)),
)

GRAPHITE = Pigment("#38383b", name="graphite")
#: The clay of a lead: pale, and nearly clear in so thin a film.
CLAY = Pigment("#cfcac0", opacity=0.35, name="clay")
#: A lead is far darker than a pigment made to cover at thickness 1: this
#: scales it so that thickness 1 is about a tooth full of the softest lead.
STRENGTH = 0.1
#: Share of graphite in the hardest and in the softest lead.
GRAPHITE_HARD, GRAPHITE_SOFT = 0.45, 0.92
#: Share of a graphite mark that the paper holds at once (the rest smudges).
BINDING = 0.85


def grade(name: str) -> Medium:
    """The pencil of that grade: ``"9H"`` ... ``"H"``, ``"F"``, ``"HB"``,
    ``"B"`` ... ``"9B"`` (in either case)."""
    key = str(name).upper()
    if key not in GRADES:
        raise ValueError(f"unknown pencil grade {name!r}; use one of {GRADES}")
    softness = GRADES.index(key) / (len(GRADES) - 1)
    graphite = GRAPHITE_HARD + (GRAPHITE_SOFT - GRAPHITE_HARD) * softness
    mixed = Pigment.mix((GRAPHITE, graphite), (CLAY, 1.0 - graphite))
    lead = Pigment(mixed, strength=STRENGTH, name=f"{key} lead")
    return Medium(lead, softness=softness, binding=BINDING, name=f"{key} pencil")


H9, H8, H7, H6, H5, H4, H3, H2 = (grade(f"{n}H") for n in range(9, 1, -1))
H, F, HB, B = grade("H"), grade("F"), grade("HB"), grade("B")
B2, B3, B4, B5, B6, B7, B8, B9 = (grade(f"{n}B") for n in range(2, 10))


#: A coloured lead: how strongly it hides what is under it, how soft it is,
#: and the shares of it that the paper holds and that are wax.
COLOURED_OPACITY = 0.4
#: A full layer is thinner than the thickness 1 a colour is defined at: the
#: lead is this much stronger, so that a full tooth shows its colour.
COLOURED_STRENGTH = 1.6
COLOURED_SOFTNESS = 0.55
COLOURED_BINDING = 0.92
COLOURED_WAX = 0.5


def coloured(color: ColorLike, name: str | None = None) -> Medium:
    """A coloured pencil of that colour: how a full layer of it looks on
    white paper."""
    lead = Pigment(color, COLOURED_OPACITY, COLOURED_STRENGTH, name)
    return Medium(
        lead,
        softness=COLOURED_SOFTNESS,
        binding=COLOURED_BINDING,
        wax=COLOURED_WAX,
        name=name or f"{lead.hex} pencil",
    )


#: A colourless blender: a pencil of wax with next to no pigment in it, to
#: burnish what has been drawn.
BLENDER = Medium(
    Pigment("#ffffff", opacity=0.02, name="blender"),
    softness=COLOURED_SOFTNESS,
    binding=1.0,
    wax=1.0,
    name="blender pencil",
)
