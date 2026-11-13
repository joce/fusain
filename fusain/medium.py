"""A medium: what a stick is made of and how it behaves on paper."""

from __future__ import annotations

from dataclasses import dataclass, replace

from murex import Pigment


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
        the rest lying on it as dust that can still be moved.
    wax:
        0..1: the share of it that is wax (or oil). Wax films over the tooth
        and makes it slick: later layers take less and less, and a firm pass
        polishes the film flat, which closes up the white specks of the
        paper (burnishing). Graphite, charcoal and soft pastel have none.
    sheen:
        0..1: how much of it mirrors the light once the paper holds it, as
        graphite does. A thick layer of it shines under a lamp, so that it
        never looks quite black. Charcoal, pastel and wax have none.
    name:
        Optional label.
    """

    pigment: Pigment
    softness: float
    binding: float = 0.85
    wax: float = 0.0
    sheen: float = 0.0
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.pigment, Pigment):
            raise TypeError(f"pigment must be a murex Pigment, got {self.pigment!r}")
        for field in ("softness", "binding", "wax", "sheen"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(
                    f"{field} must be between 0 and 1, got {getattr(self, field)!r}"
                )

    def with_(self, **changes) -> Medium:
        """A copy with some fields changed."""
        return replace(self, **changes)
