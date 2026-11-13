"""A stick: a medium with a tip that wears (behaviour 6 of docs/design.md)."""

from typing import Any

import numpy as np
import pytest

from fusain import Paper, Sheet, Stick, charcoal, pencils

LEVEL = Paper(depth=0)  # the width of a line shows exactly


def _rows(sheet, x) -> int:
    """How many rows the line marks around column ``x``: its width there."""
    return int(np.count_nonzero(sheet.pigment()[:, x : x + 30].sum(axis=1)))


def _line(medium, length=1500, pressure=0.6, **kw) -> tuple[Sheet, Stick]:
    sheet = Sheet(length + 100, 60, paper=LEVEL, seed=1)
    stick = Stick(medium, **kw)
    sheet.stroke([(50, 30), (50 + length, 30)], stick, pressure)
    return sheet, stick


def test_a_line_widens_as_its_point_wears():
    sheet, stick = _line(pencils.B2, width=3, lead=20)
    start, middle, end = _rows(sheet, 60), _rows(sheet, 780), _rows(sheet, 1500)
    assert start <= middle <= end and end >= start + 2
    assert 3.0 < stick.width < 20 and abs(end - 1 - stick.width) <= 1.5
    # a bare medium keeps its width all along
    plain = Sheet(1600, 60, paper=LEVEL, seed=1)
    plain.stroke([(50, 30), (1550, 30)], pencils.B2, 0.6, 3.0)
    assert _rows(plain, 60) == _rows(plain, 1500) == 4


def test_a_soft_lead_wears_faster_than_a_hard_one():
    widths = [
        _line(pencils.grade(g), width=3, lead=20)[1].width for g in ("2H", "HB", "6B")
    ]
    assert 3.0 < widths[0] < widths[1] < widths[2]
    # and pressing harder wears it faster
    light = _line(pencils.B2, pressure=0.3, width=3, lead=20)[1].width
    firm = _line(pencils.B2, pressure=0.9, width=3, lead=20)[1].width
    assert firm > light + 0.3


def test_what_it_wears_is_what_it_lays():
    sheet = Sheet(400, 200, seed=1)
    stick = Stick(pencils.B4, width=3)
    laid = sheet.stroke([(20, 30), (380, 30)], stick, 0.7)
    assert stick.worn == pytest.approx(laid) and laid > 0
    laid += sheet.shade((20, 60, 380, 180), stick, 0.5)
    assert stick.worn == pytest.approx(laid)
    np.testing.assert_allclose(sheet.pigment().sum(dtype=np.float64), laid, rtol=1e-4)
    # the width follows from it: the cube of the width grows with the volume
    grown = stick.width**3 - stick.point**3
    assert grown == pytest.approx(0.08 * stick.worn, rel=1e-6)


def test_the_next_stroke_starts_as_wide_as_the_last_ended():
    sheet = Sheet(1600, 120, paper=LEVEL, seed=1)
    stick = Stick(pencils.B6, width=3, lead=20)
    sheet.stroke([(50, 30), (1550, 30)], stick, 0.6)
    ended = _rows(sheet, 1500)
    sheet.stroke([(50, 90), (400, 90)], stick, 0.6)
    below = sheet.pigment()[60:]
    starts = int(np.count_nonzero(below[:, 60:90].sum(axis=1)))
    assert abs(starts - ended) <= 1


def test_sharpening_gives_the_point_back():
    _, stick = _line(pencils.B6, width=3, lead=20)
    assert stick.width > 5 and stick.worn > 0
    stick.sharpen()
    assert stick.width == 3 and stick.worn == 0
    stick.sharpen(2)  # to a finer point
    assert stick.width == stick.point == 2
    fresh = Sheet(300, 60, paper=LEVEL, seed=1)
    fresh.stroke([(50, 30), (200, 30)], stick, 0.6)
    assert _rows(fresh, 60) <= 4  # as fine as a point 2 wide draws
    with pytest.raises(ValueError, match="wider than the lead"):
        stick.sharpen(25)


def test_a_tip_never_gets_wider_than_the_lead():
    sheet, stick = _line(pencils.B8, length=3000, pressure=1.0, width=3, lead=5)
    assert stick.width == 5 and _rows(sheet, 3000) <= 6
    # a stick of charcoal is as wide as it is from the start
    sheet, stick = _line(charcoal.VINE, width=10, lead=10)
    assert stick.width == 10 and 10 <= _rows(sheet, 60) == _rows(sheet, 1500) <= 11
    assert Stick(pencils.HB, width=2).lead == 12  # six times the point


def test_shading_with_a_stick_wears_it_and_fills_the_area():
    sheet = Sheet(300, 200, seed=2)
    stick = Stick(pencils.B4, width=3)
    laid = sheet.shade((30, 30, 270, 170), stick, 0.6)
    assert laid > 0 and stick.width > 3.3
    assert (sheet.pigment()[60:140, 60:240] > 0).mean() > 0.6
    # tilted, its side marks a band five times its width now
    stick.sharpen()
    wide = Sheet(300, 100, paper=LEVEL, seed=2)
    wide.stroke([(30, 50), (60, 50)], stick, 0.6, tilt=1.0)
    assert 14 <= int(np.count_nonzero(wide.pigment()[:, 35:55].sum(axis=1))) <= 18


def test_a_stick_checks_what_it_is_given():
    with pytest.raises(TypeError, match="medium must be a Medium"):
        wrong: Any = "HB"
        Stick(wrong)
    with pytest.raises(ValueError, match="width must be"):
        Stick(pencils.HB, width=0)
    with pytest.raises(ValueError, match="lead must be"):
        Stick(pencils.HB, width=2, lead=float("nan"))
    with pytest.raises(ValueError, match="at least its point"):
        Stick(pencils.HB, width=4, lead=3)
    stick, sheet = Stick(pencils.HB, width=2), Sheet(100, 60, seed=1)
    for draw in (
        lambda: sheet.stroke([(10, 30), (90, 30)], stick, 0.5, 4.0),
        lambda: sheet.shade((10, 10, 90, 50), stick, 0.5, 4.0),
    ):
        with pytest.raises(TypeError, match="a Stick has its own width"):
            draw()
    assert not sheet.pigment().any() and stick.worn == 0
    assert repr(stick) == "Stick(HB pencil, width=2.00, lead=12)"
