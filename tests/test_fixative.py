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
    assert sheet.fixed.shape == (150, 200, 2)
    assert np.array_equal(sheet.fixed[..., 0], sheet.under[..., IV])
    assert np.array_equal(sheet.fixed[..., 1], sheet.loose[..., IV])
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
    assert 0.4 * carried[0] < carried[1] < 0.85 * carried[0]


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
    for which, layer in enumerate((sheet.under, sheet.loose)):
        assert (sheet.fixed[..., which] <= layer[..., IV] + 1e-5).all()
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
    assert 0.0 < carried < 0.7 * _carried_out(loose)
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


# --------------------------------------------------------------------------
# a fixed layer under a loose one


def _white_over_fixed_black(fix: bool) -> Sheet:
    sheet = _drawn(pastels.soft(BLACK), layers=1)
    if fix:
        sheet.fix()
    sheet.shade(BOX, pastels.soft(WHITE), 0.8, 12, angle=-30)
    return sheet


def _rubbed(sheet) -> Sheet:
    for k in range(6):
        y = 55 + 7 * k
        sheet.smudge([(40, y), (160, y)], tools.FINGER, 0.7)
        sheet.smudge([(160, y + 3), (40, y + 3)], tools.FINGER, 0.7)
    return sheet


def test_the_fixed_dust_goes_under_the_new_layer_and_stays_fixed():
    sheet = _drawn(pastels.soft(BLACK), layers=1)
    fixed = sheet.fix()
    sheet.shade(BOX, pastels.soft(WHITE), 0.8, 12, angle=-30)
    # (no fixative was added or lost; most of it went under with its dust)
    assert float(sheet.fixed.sum(dtype=np.float64)) == pytest.approx(fixed, rel=1e-5)
    assert sheet.fixed[INSIDE][..., 0].sum() > 5 * sheet.fixed[INSIDE][..., 1].sum()
    # the layer on top is loose
    top = float(sheet.loose[INSIDE][..., IV].sum())
    assert sheet.fixed[INSIDE][..., 1].sum() < 0.1 * top


def test_rubbing_a_new_layer_does_not_stir_up_the_fixed_one_under_it():
    band = (slice(55, 95), slice(60, 140))
    tones = {}
    for fix in (False, True):
        sheet = _white_over_fixed_black(fix)
        before = _tone(sheet, band)
        held = float(sheet.held[band][..., IV].sum())
        fixed = float(sheet.fixed[band][..., 0].sum())
        _rubbed(sheet)
        tones[fix] = _tone(sheet, band) / before
        pressed = float(sheet.held[band][..., IV].sum()) / held
        if fix:
            # the black stays where it was, set, and the finger presses next
            # to nothing into the paper through it
            assert float(sheet.fixed[band][..., 0].sum()) > 0.95 * fixed
            assert float(sheet.under[band][..., IV].sum()) > 0.95 * fixed
            assert pressed < 1.1
        else:
            assert pressed > 2.0
    # unfixed, the finger grinds the black and the white together into the
    # paper, a dark grey; fixed, a haze of white is left over the black
    assert tones[False] < 0.15
    assert tones[True] > 2 * tones[False]


def test_a_second_coat_darkens_only_what_it_newly_sets():
    sheet = _drawn(pastels.soft(BLACK), layers=1)
    sheet.fix()
    sheet.shade(BOX, pastels.soft(WHITE), 0.8, 12, angle=-30)
    was_set = sheet.fixed.copy()
    layers = (sheet.under, sheet.loose)
    scatter = [layer[..., IS].astype(np.float64) for layer in layers]
    assert sheet.fix() > 0
    for which, layer in enumerate(layers):
        volume = layer[..., IV].astype(np.float64)
        newly = np.clip(volume - was_set[..., which], 0.0, None)
        share = newly / np.maximum(volume, 1e-9)
        expected = scatter[which] * (1.0 - 0.15 * share)
        assert np.allclose(layer[..., IS], expected, rtol=1e-4, atol=1e-6)
    # where the first coat had set all of a layer, the second leaves it alone
    all_set = was_set[..., 0] >= 0.999 * sheet.under[..., IV]
    all_set &= sheet.under[..., IV] > 0
    assert all_set.sum() > 100
    assert np.allclose(sheet.under[..., IS][all_set], scatter[0][all_set], rtol=1e-3)
    # and the new layer on top, loose until now, scatters 15% less
    top = (slice(50, 100), slice(50, 150))
    ratio = float(sheet.loose[top][..., IS].sum()) / float(scatter[1][top].sum())
    assert 0.85 <= ratio < 0.86
    assert sheet.fix() == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------
# a full tooth, and why a pastellist fixes between stages


def test_the_tooth_is_full_of_pastel_after_five_or_six_layers():
    sheet = Sheet(200, 150, paper=papers.CARTRIDGE, seed=1)
    red = pastels.soft(RED)
    takes = [sheet.shade(BOX, red, 0.7, 12, angle=30 + 50 * k) for k in range(8)]
    shares = [take / takes[0] for take in takes]
    assert shares[1] > 0.6
    assert shares[3] > 0.25
    assert shares[6] < 0.2
    # sanded card grips layer after layer: it is far from full after twelve
    card = Sheet(200, 150, paper=papers.SANDED, seed=1)
    on_card = [card.shade(BOX, red, 0.7, 12, angle=30 + 50 * k) for k in range(12)]
    assert on_card[6] / on_card[0] > 0.4
    assert on_card[11] / on_card[0] > 0.25


def test_a_light_pastel_covers_a_loaded_tooth_poorly_unless_it_is_fixed_first():
    tones = {}
    for fix in (False, True):
        sheet = _drawn(pastels.soft(BLACK), layers=4)
        if fix:
            sheet.fix()
        sheet.shade(BOX, pastels.soft(WHITE), 0.8, 12, angle=-30)
        tones[fix] = _tone(sheet)
    one_layer = _drawn(pastels.soft(BLACK), layers=1)
    one_layer.shade(BOX, pastels.soft(WHITE), 0.8, 12, angle=-30)
    # over one layer of black the white covers well; over four, poorly
    assert tones[False] < 0.6 * _tone(one_layer)
    # fixed first, it covers nearly as well as over one layer
    assert tones[True] > 1.6 * tones[False]
    assert tones[True] > 0.8 * _tone(one_layer)
