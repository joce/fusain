"""A medium: what a stick is made of and how it behaves on paper."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from murex import Pigment

from ._engine import SIDE


@dataclass(frozen=True)
class Medium:
    """What a pencil lead, a charcoal stick or a pastel is made of. Presets
    live in modules such as :mod:`fusain.pencils`.

    Parameters
    ----------
    pigment:
        What it leaves on the paper, as a :class:`murex.Pigment`.
    softness:
        0..1: how easily it wears and how far it gives to reach into the
        tooth. A hard lead (low) lays a thin, pale film on the peaks and
        fills up early; a soft one (high) lays a lot, reaches the valleys
        and goes on darkening.
    binding:
        0..1: the share of what it lays that is held by the paper at once,
        the rest lying on it as dust that can still be moved (or, for a
        medium that is waxy and soft, as paste, which only smears).
    wax:
        0..1: the share of it that is wax (or oil). Wax films over the tooth
        and makes it slick: later layers take less and less, and a firm pass
        polishes the film flat, which closes up the white specks of the
        paper (burnishing). Laid over dust, the film lies on the dust and
        binds part of it. Graphite, charcoal and soft pastel have none.
    sheen:
        0..1: how much of it mirrors the light once the paper holds it, as
        graphite does. A thick layer of it shines under a lamp, so that it
        never looks quite black. Charcoal, pastel and wax have none. (Wax
        has a gloss of another kind, which is not the medium's but its
        polish's: see :meth:`fusain.Sheet.shine`.)
    side:
        How many times as wide as its point the side of a stick of it
        marks, held flat (at full tilt): what length of it lies on the
        paper. The exposed lead of a sharpened pencil marks a band five
        times its point (the default); a piece of charcoal or a pastel laid
        flat marks a band as wide as the piece is long. A
        :class:`fusain.Stick` can give the length in pixels instead.
    name:
        Optional label.
    """

    pigment: Pigment
    softness: float
    binding: float = 0.85
    wax: float = 0.0
    sheen: float = 0.0
    side: float = SIDE
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.pigment, Pigment):
            raise TypeError(f"pigment must be a murex Pigment, got {self.pigment!r}")
        for field in ("softness", "binding", "wax", "sheen"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(
                    f"{field} must be between 0 and 1, got {getattr(self, field)!r}"
                )
        if not (math.isfinite(self.side) and self.side >= 1.0):
            raise ValueError(
                f"side must be at least 1 (as wide as the point), got {self.side!r}"
            )

    def with_(self, **changes) -> Medium:
        """A copy with some fields changed."""
        return replace(self, **changes)
