"""Pencils. Graphite, by grade: ``pencils.HB``, ``pencils.B6`` (a 6B),
``pencils.H2`` (a 2H), or ``pencils.grade("6B")``. Coloured, of any colour:
``pencils.coloured("#c0392b")``; and ``pencils.BLENDER``, a colourless one.

A graphite lead is graphite fired with clay. More clay makes it harder: it
wears less, stays on the peaks of the tooth and leaves a paler grey. Graphite
mirrors the light, so a heavy layer of a soft lead shines under a lamp. A
hard lead pressed firmly flattens the tooth instead: what is shaded over it
lies evenly, the paper bites it less, and the little graphite there has a
sheen. (A lead flattens it as far as its hardness and its pressure can, in
one layer or in many: a 4H flat, an HB by two fifths at most.) A
coloured lead is pigment in wax: translucent, so that layers of two colours
mix to the eye, and waxy, so that the tooth turns slick as it fills. Laid
lightly, ten or so layers go on building, and fill the tooth without
polishing it; a layer pressed hard fills and polishes the tooth, as far as
it is pressed, little takes over it, and it has a soft gloss under a
lamp. The colourless blender pressed hard burnishes what it goes over: the
white flecks of the paper close, and the colour is deeper and smoother; a
white or a pale pencil pressed hard does too, nearly as well the paler it
is, and lightens the colour as it does. Wax
holds against an eraser, and stains the paper, so that a tint stays where
a layer was erased: the lights of a coloured pencil drawing are kept, or
scratched out with ``tools.KNIFE``.
"""

from __future__ import annotations

import numpy as np
from murex import ColorLike, Pigment
from murex import spectral as sp

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
    return Medium(
        lead,
        softness=softness,
        binding=BINDING,
        sheen=graphite,  # (the graphite shines, the clay does not)
        name=f"{key} pencil",
    )


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
#: A pale lead is mostly wax and white filler, and little pigment: pressed
#: hard it melts into the film under it as a blender does, the more the
#: paler it is. A white lead melts PALE_MELTS as far as the blender, and
#: one PALE_RANGE or more from white (as a colour difference, CIE 1976)
#: not at all; in between, as the square of how far short of that it is,
#: so a cream or a pale tint melts nearly as a white, and a mid grey or a
#: pink hardly.
PALE_MELTS = 0.75
PALE_RANGE = 40.0


def melts_at(pale: float, softness: float) -> float:
    """How far a coloured lead melts (:attr:`fusain.Medium.melts`) that
    would melt ``pale`` at the softness of an ordinary coloured pencil, at
    that ``softness``: in proportion to it, so that a softer, waxier core
    burnishes more and a harder one less (assumed), up to 1, the blender."""
    return min(pale * softness / COLOURED_SOFTNESS, 1.0)


def melts_of(color: ColorLike) -> float:
    """How far a coloured lead of that colour melts into a film of wax it is
    pressed hard on (:attr:`fusain.Medium.melts`), at the softness of an
    ordinary coloured pencil: as far as it is pale. A
    white or a cream pencil burnishes nearly as the blender does, a pale
    blue or a pink one somewhat, a yellow or a red one not (assumed: a pale
    lead holds little pigment, and the wax and filler that make up the rest
    are what a blender is made of)."""
    srgb = np.asarray(Pigment(color).color, dtype=np.float64)
    lab = sp.srgb_to_lab(srgb)
    far = float(np.linalg.norm(lab - np.array([100.0, 0.0, 0.0])))
    return PALE_MELTS * max(1.0 - far / PALE_RANGE, 0.0) ** 2


def coloured(color: ColorLike, name: str | None = None) -> Medium:
    """A coloured pencil of that colour: how a full layer of it looks on
    white paper. A pale one pressed hard also burnishes what is under it,
    as the blender does, the more the paler it is (:func:`melts_of`), and
    lays its own pale colour as it does."""
    lead = Pigment(color, COLOURED_OPACITY, COLOURED_STRENGTH, name)
    pale = melts_of(color)
    return Medium(
        lead,
        softness=COLOURED_SOFTNESS,
        binding=COLOURED_BINDING,
        wax=COLOURED_WAX,
        name=name or f"{lead.hex} pencil",
        melts=pale,
        pale=pale,
    )


#: A colourless blender: a pencil of wax with next to no pigment in it, to
#: burnish what has been drawn. Pressed hard (0.8 or more), it melts into
#: the wax under it and presses it into the valleys of the tooth: one firm
#: layer closes four fifths of the white flecks.
BLENDER = Medium(
    Pigment("#ffffff", opacity=0.02, name="blender"),
    softness=COLOURED_SOFTNESS,
    binding=1.0,
    wax=1.0,
    name="blender pencil",
    melts=1.0,
)
