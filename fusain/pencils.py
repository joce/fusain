"""Graphite pencils, by grade: ``pencils.HB``, ``pencils.B6`` (a 6B),
``pencils.H2`` (a 2H), or ``pencils.grade("6B")``.

A lead is graphite fired with clay. More clay makes it harder: it wears
less, stays on the peaks of the tooth and leaves a paler grey.
"""

from __future__ import annotations

from murex import Pigment

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
