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
    name:
        Optional label.
    """

    pigment: Pigment
    softness: float
    binding: float = 0.85
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.pigment, Pigment):
            raise TypeError(f"pigment must be a murex Pigment, got {self.pigment!r}")
        for field in ("softness", "binding"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(
                    f"{field} must be between 0 and 1, got {getattr(self, field)!r}"
                )

    def with_(self, **changes) -> Medium:
        """A copy with some fields changed."""
        return replace(self, **changes)
