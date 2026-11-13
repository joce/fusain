"""Fixative: dust that is set stays where it is, shows as it lay and leaves
room for more, measured."""

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, charcoal, papers, pastels, pencils, tools
from fusain.sheet import IS, IV, N

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
RED, WHITE, BLACK = "#c0392b", "#f4f1ea", "#161616"


def _drawn(medium=None, width=12.0, layers=2) -> Sheet:
    sheet = Sheet(200, 150, paper=papers.CARTRIDGE, seed=1)
    medium = pastels.soft(RED) if medium is None else medium
    for layer in range(layers):
        sheet.shade(BOX, medium, 0.7, width, angle=30 + 50 * layer)
    return sheet


def _tone(sheet, where=INSIDE) -> float:
    spectrum = sheet.reflectance()[where].astype(np.float64).mean(axis=(0, 1))
    return float(sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE)


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


def _dust(sheet) -> np.ndarray:
    return sheet.under[..., IV] + sheet.loose[..., IV]


def _carried_out(sheet) -> float:
    """What a finger rubbed across the patch carries past its edge."""
    before = float(sheet.pigment()[:, 185:].sum())
    for k in range(3):
        sheet.smudge([(40, 60 + 8 * k), (198, 60 + 8 * k)], tools.FINGER, 0.7)
    return float(sheet.pigment()[:, 185:].sum()) - before


# --------------------------------------------------------------------------
# fixing


def test_fixing_sets_all_the_dust_once():
    sheet = _drawn()
    dust = float(_dust(sheet).sum(dtype=np.float64))
    assert not sheet.fixed.any()
    assert sheet.fix() == pytest.approx(dust, rel=1e-6)
    assert np.allclose(sheet.fixed, _dust(sheet), rtol=1e-5, atol=1e-7)
    tone = _tone(sheet)
    assert sheet.fix() == pytest.approx(0.0, abs=1e-3)
    assert _tone(sheet) == tone


def test_a_light_coat_sets_part_of_it():
    sheet = _drawn()
    dust = float(_dust(sheet).sum(dtype=np.float64))
    assert sheet.fix(0.5) == pytest.approx(0.5 * dust, rel=1e-5)
    assert sheet.fix(0.0) == 0.0
    assert sheet.fix() == pytest.approx(0.5 * dust, rel=1e-5)
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="amount"):
            sheet.fix(bad)


def test_a_blank_sheet_has_nothing_to_fix_and_graphite_little():
    blank = Sheet(60, 40, seed=1)
    before = blank.render()
    assert blank.fix() == 0.0
    assert np.array_equal(blank.render(), before)
    pencil = _drawn(pencils.B4, 6.0)
    assert pencil.fix() < 0.2 * _total(pencil)


def test_a_copy_keeps_its_own_fixative():
    sheet = _drawn()
    copy = sheet.copy()
    copy.fix()
    assert copy.fixed.any()
    assert not sheet.fixed.any()


# --------------------------------------------------------------------------
# how it looks


def test_fixed_pastel_is_a_little_darker_and_no_pigment_moves():
    sheet = _drawn()
    before, total = _tone(sheet), _total(sheet)
    layers = [layer[..., :N].copy() for layer in (sheet.held, sheet.under, sheet.loose)]
    scatter = float(sheet.loose[..., IS].sum())
    sheet.fix()
    assert 0.88 * before < _tone(sheet) < 0.99 * before
    assert _total(sheet) == total
    for was, layer in zip(layers, (sheet.held, sheet.under, sheet.loose), strict=True):
        assert np.array_equal(was, layer[..., :N])
    assert float(sheet.loose[..., IS].sum()) < scatter


def test_a_light_pastel_still_covers_a_dark_one_once_fixed():
    sheet = _drawn(pastels.soft(BLACK), layers=1)
    dark = _tone(sheet)
    sheet.shade(BOX, pastels.soft(WHITE), 0.8, 12, angle=-30)
    over = _tone(sheet)
    sheet.fix()
    assert 0.9 * over < _tone(sheet) < over
    assert _tone(sheet) > 10 * dark


# --------------------------------------------------------------------------
# what moves it


@pytest.mark.parametrize(
    "medium",
    [pastels.soft(RED), charcoal.VINE, charcoal.COMPRESSED],
    ids=["pastel", "vine", "compressed"],
)
def test_a_finger_hardly_moves_fixed_dust(medium):
    loose = _drawn(medium)
    half, fixed = loose.copy(), loose.copy()
    half.fix(0.5)
    fixed.fix()
    carried = [_carried_out(sheet) for sheet in (loose, half, fixed)]
    assert carried[2] < 0.05 * carried[0]
    assert 0.3 * carried[0] < carried[1] < 0.7 * carried[0]


def test_a_stump_and_a_kneaded_eraser_take_little_and_a_hard_eraser_most():
    loose = _drawn()
    fixed = loose.copy()
    fixed.fix()
    line = [(30, 75), (170, 75)]
    takes = {}
    for name, tool in (("stump", tools.STUMP), ("kneaded", tools.KNEADED)):
        a, b = loose.copy(), fixed.copy()
        rub = "smudge" if name == "stump" else "erase"
        takes[name] = getattr(b, rub)(line, tool, 0.8) / getattr(a, rub)(
            line, tool, 0.8
        )
    assert takes["stump"] < 0.05
    assert takes["kneaded"] < 0.35
    a, b = loose.copy(), fixed.copy()
    assert b.erase(line, tools.ERASER, 0.8) > 0.8 * a.erase(line, tools.ERASER, 0.8)


def test_rubbing_and_erasing_fixed_dust_conserves_it():
    sheet = _drawn()
    total = _total(sheet)
    sheet.fix()
    carried = 0.0
    for k in range(5):
        y = 40 + 18 * k
        carried += sheet.smudge([(30, y), (170, y)], tools.STUMP, 0.8)
        carried += sheet.erase([(30, y + 6), (170, y + 6)], tools.ERASER, 0.9)
    carried += sheet.erase([(30, 75), (170, 75)], tools.KNEADED, 0.8)
    assert _total(sheet) + carried == pytest.approx(total, rel=1e-5)
    assert sheet.fixed.min() >= 0.0
    assert (sheet.fixed <= _dust(sheet) + 1e-5).all()
    # where the hard eraser went, the fixative went with the dust
    assert sheet.fixed[44:48, 60:140].mean() < 0.3 * sheet.fixed[30:34, 60:140].mean()


# --------------------------------------------------------------------------
# drawing over it


def test_fixed_pastel_takes_more_and_the_new_layer_is_loose():
    loose = _drawn()
    fixed = loose.copy()
    fixed.fix()
    red = pastels.soft(RED)
    more = fixed.shade(BOX, red, 0.7, 12, angle=-20)
    assert more > 1.2 * loose.shade(BOX, red, 0.7, 12, angle=-20)
    # only what was laid since is free to move
    carried = _carried_out(fixed)
    assert 0.0 < carried < 0.5 * _carried_out(loose)
    assert carried > 5 * _carried_out(_fixed_only())


def _fixed_only() -> Sheet:
    sheet = _drawn()
    sheet.fix()
    return sheet


def test_a_pencil_does_not_press_fixed_dust_into_the_paper():
    loose = _drawn(charcoal.VINE)
    fixed = loose.copy()
    fixed.fix()
    pressed = []
    for sheet in (loose, fixed):
        dust = float(_dust(sheet).sum(dtype=np.float64))
        # (a blender leaves no dust of its own, and presses hard)
        sheet.shade(BOX, pencils.BLENDER, 1.0, angle=-20)
        pressed.append(dust - float(_dust(sheet).sum(dtype=np.float64)))
    assert pressed[0] > 100.0
    assert abs(pressed[1]) < 0.01 * pressed[0]
