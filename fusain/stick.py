"""A stick in the hand: a medium, and the state of its tip."""

from __future__ import annotations

import math

from ._engine import WEAR
from .medium import Medium


class Stick:
    """A pencil, a stick of charcoal or a pastel, as it is held: its medium
    and its tip, which wears as it draws. A point wears to a flat, so its
    line widens until it is sharpened again::

        from fusain import Sheet, Stick, pencils
        pencil = Stick(pencils.B2, width=2)
        sheet.stroke(path, pencil)        # the line widens as it goes
        sheet.shade(area, pencil)         # and so do these strokes
        pencil.sharpen()

    Draw with a bare medium (``sheet.stroke(path, pencils.B2, width=4)``)
    for a tip that keeps its width.

    Parameters
    ----------
    medium:
        What it is made of, e.g. ``pencils.B2``.
    width:
        Width of its point in pixels when it is sharp.
    lead:
        Full width of the lead or stick in pixels: the tip never gets wider.
        By default six times the point. Give the same as ``width`` for a
        stick that is never sharpened, such as charcoal or pastel.
    side:
        Length in pixels of the side it lays on the paper when it is held
        flat (``tilt=1``): the width of the band it marks then. By default
        as many times its point as its medium says
        (:attr:`fusain.Medium.side`): five for a pencil, whose side is the
        lead its sharpening bared; for a piece of charcoal or pastel laid
        flat, which lays a band even across (:attr:`fusain.Medium.flat_side`),
        give its length::

            stick = Stick(charcoal.VINE, width=6, lead=6, side=90)
            sheet.stroke(path, stick, tilt=1)   # a band 90 pixels wide

    A soft medium lays more and so wears faster. ``width`` is the tip's width
    now and ``worn`` the volume it has laid since it was sharpened.
    """

    def __init__(
        self,
        medium: Medium,
        width: float = 2.0,
        lead: float | None = None,
        side: float | None = None,
    ):
        if not isinstance(medium, Medium):
            raise TypeError(f"medium must be a Medium, such as pencils.HB: {medium!r}")
        self.medium = medium
        self.point = _width("width", width)
        self.lead = 6.0 * self.point if lead is None else _width("lead", lead)
        if self.lead < self.point:
            raise ValueError(
                f"lead ({lead!r}) is the full width of the stick: at least its "
                f"point ({width!r})"
            )
        self._side = None if side is None else _width("side", side)
        if self._side is not None and self._side < self.point:
            raise ValueError(
                f"side ({side!r}) is the length of the stick: at least its "
                f"point ({width!r})"
            )
        self.worn = 0.0

    @property
    def side(self) -> float:
        """Length of its side in pixels: how wide a band it marks held flat
        (but never narrower than its tip is now)."""
        if self._side is None:
            return self.medium.side * self.point
        return self._side

    @property
    def width(self) -> float:
        """Width of the tip now, in pixels."""
        return min((self.point**3 + WEAR * self.worn) ** (1.0 / 3.0), self.lead)

    def sharpen(self, width: float | None = None) -> None:
        """Sharpen it: back to its point, or to a point ``width`` wide."""
        if width is not None:
            width = _width("width", width)
            if width > self.lead:
                raise ValueError(f"a point cannot be wider than the lead: {width!r}")
            if self._side is not None and width > self._side:
                raise ValueError(f"a point cannot be wider than the side: {width!r}")
            self.point = width
        self.worn = 0.0

    def __repr__(self) -> str:
        name = self.medium.name or "stick"
        return (
            f"Stick({name}, width={self.width:.2f}, lead={self.lead:g}, "
            f"side={self.side:g})"
        )


def _width(name: str, value) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a number of pixels above 0, got {value!r}")
    return value
