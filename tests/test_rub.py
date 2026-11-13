"""Rubbing and erasing an area: Sheet.rub, measured."""

from typing import Any

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, charcoal, papers, pencils, tools

PATCH = (30, 30, 270, 170)
INSIDE = (slice(60, 140), slice(70, 230))


def _drawn(medium=charcoal.VINE, width=9.0) -> tuple[Sheet, float]:
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
    laid = sum(sheet.shade(PATCH, medium, 0.6, width, angle=a) for a in (30, -40))
    return sheet, laid


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


def _grain(sheet, where=INSIDE) -> float:
    spectrum = sheet.reflectance()[where].astype(np.float64)
    return float((sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE).std())


def test_rubbing_an_area_smooths_all_of_it_and_loses_nothing():
    sheet, laid = _drawn()
    grainy = _grain(sheet)
    taken = sheet.rub(PATCH, tools.FINGER, 0.6)
    assert taken > 0
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)
    assert _grain(sheet) < 0.5 * grainy
    for corner in ((slice(40, 70), slice(40, 90)), (slice(130, 160), slice(210, 260))):
        assert _grain(sheet, corner) < 0.6 * grainy


def test_only_the_area_is_rubbed():
    sheet, _ = _drawn()
    before = sheet.pigment().copy()
    sheet.rub((30, 30, 150, 170), tools.STUMP, 0.6)
    after = sheet.pigment()
    # (a stump is 10 wide: nothing changes more than its width past the area)
    assert np.array_equal(before[:, 165:], after[:, 165:])
    assert not np.array_equal(before[:, 40:140], after[:, 40:140])


def test_the_pressure_may_be_a_map_and_the_tool_is_lifted_where_it_is_nothing():
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
    laid = sheet.shade((30, 30, 200, 170), charcoal.VINE, 0.6, 9)
    before = sheet.pigment().copy()
    assert before[:, 230:].sum() == 0.0

    def left_and_far_right(x, y):
        return 0.7 * ((x < 100) | (x > 245))

    # rub the left of the patch and the bare paper far to its right, not between
    taken = sheet.rub((30, 60, 298, 140), tools.STUMP, left_and_far_right)
    after = sheet.pigment()
    assert not np.array_equal(before[:, 40:90], after[:, 40:90])
    assert np.array_equal(before[:, 115:230], after[:, 115:230])
    # lifted over the gap, the stump brings no dust to the bare paper beyond
    assert after[:, 230:].sum() == 0.0
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)


def test_an_eraser_clears_an_area():
    sheet, laid = _drawn(pencils.B4, 6.0)
    taken = sheet.rub((100, 60, 200, 140), tools.ERASER, 0.9)
    assert taken > 0.2 * laid
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)
    cleared = sheet.pigment()[75:125, 115:185].sum()
    kept = sheet.pigment()[75:125, 40:85].sum() * (70 / 45)
    assert cleared < 0.1 * kept


def test_the_passes_run_at_the_angle_asked_for_and_repeat_with_the_seed():
    def smeared(angle: float) -> np.ndarray:
        sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
        sheet.stroke([(150, 60), (150, 140)], charcoal.COMPRESSED, 0.8, 8)
        sheet.rub((100, 50, 200, 150), tools.STUMP, 0.7, angle=angle, hand=0.0)
        return sheet.pigment()

    across, upright = smeared(0.0), smeared(90.0)
    # rubbed across it, the line is dragged out sideways; along it, hardly
    assert across[70:130, 110:135].sum() > 3 * upright[70:130, 110:135].sum()
    assert np.array_equal(across, smeared(0.0))


@pytest.mark.parametrize(
    ("kw", "error", "message"),
    [
        ({"tool": "finger"}, TypeError, "tool must be a Tool"),
        ({"width": 0.0}, ValueError, "width must be"),
        ({"spacing": -1.0}, ValueError, "spacing must be"),
        ({"angle": float("nan")}, ValueError, "angle must be"),
        ({"hand": -0.5}, ValueError, "hand must be"),
        ({"area": (1, 2, 3)}, ValueError, "an area is"),
    ],
)
def test_rub_checks_its_arguments_before_it_rubs(kw, error, message):
    sheet, _ = _drawn()
    before = sheet.pigment().copy()
    args: dict[str, Any] = {"area": PATCH, "tool": tools.STUMP, **kw}
    with pytest.raises(error, match=message):
        sheet.rub(**args)
    assert np.array_equal(before, sheet.pigment())
