"""Graphite's shine: what carries it, what hides it and how a lamp shows it,
measured."""

import itertools

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Medium, Sheet, charcoal, crayons, papers, pastels, pencils, tools
from fusain.sheet import IV, IX

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
OVERHEAD, RAKING = (0.0, 0.0, 1.0), (-1.0, -1.0, 0.5)


def _sheet() -> Sheet:
    return Sheet(200, 150, paper=papers.CARTRIDGE, seed=1)


def _shaded(medium, pressure=1.0, layers=1, width=None) -> Sheet:
    sheet = _sheet()
    for layer in range(layers):
        sheet.shade(BOX, medium, pressure, width, angle=30 + 50 * layer)
    return sheet


def _tone(image, where=INSIDE) -> float:
    linear = sp.srgb_to_linear(image[where].astype(np.float64))
    return float((linear @ sp.LUMINANCE).mean())


def _sheen(sheet) -> float:
    return float(
        sum(
            layer[..., IX].sum(dtype=np.float64)
            for layer in (sheet.held, sheet.under, sheet.loose)
        )
    )


# --------------------------------------------------------------------------
# the media


def test_only_graphite_has_sheen_and_a_soft_lead_has_more():
    sheens = [pencils.grade(name).sheen for name in pencils.GRADES]
    assert all(a < b for a, b in itertools.pairwise(sheens))
    assert 0.4 < sheens[0] < sheens[-1] < 1.0
    for matte in (
        charcoal.VINE,
        charcoal.COMPRESSED,
        charcoal.PENCIL,
        pastels.soft("#c0392b"),
        pastels.oil("#c0392b"),
        crayons.wax("#c0392b"),
        pencils.coloured("#c0392b"),
        pencils.BLENDER,
    ):
        assert matte.sheen == 0.0


def test_sheen_is_between_0_and_1():
    assert pencils.HB.with_(sheen=1.0).sheen == 1.0
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="sheen"):
            Medium(pencils.GRAPHITE, 0.5, sheen=bad)


# --------------------------------------------------------------------------
# on the sheet


def test_the_sheen_goes_where_the_graphite_goes():
    sheet = _sheet()
    laid = sheet.shade(BOX, pencils.B4, 0.7)
    assert _sheen(sheet) == pytest.approx(laid * pencils.B4.sheen, rel=1e-5)
    # (the dust has its share of it)
    assert sheet.loose[..., IX].sum() > 0
    carried = 0.0
    for k in range(6):
        carried += sheet.smudge([(30, 40 + 14 * k), (170, 40 + 14 * k)], tools.STUMP)
    carried += sheet.erase([(30, 75), (170, 75)], tools.ERASER, 0.9, 20)
    total = float(sheet.pigment().sum(dtype=np.float64))
    assert total + carried == pytest.approx(laid, rel=1e-5)
    assert _sheen(sheet) == pytest.approx(total * pencils.B4.sheen, rel=1e-4)
    for layer in (sheet.held, sheet.under, sheet.loose):
        assert layer[..., IX].min() >= 0.0
        assert (layer[..., IX] <= layer[..., IV] * 1.0001 + 1e-9).all()


def test_shine_grows_with_the_pressure_and_the_softness_of_the_lead():
    assert not _sheet().shine().any()
    light, firm = _shaded(pencils.HB, 0.3), _shaded(pencils.HB, 1.0)
    soft, heavy = _shaded(pencils.B6, 1.0), _shaded(pencils.B6, 1.0, 3)
    means = [float(s.shine()[INSIDE].mean()) for s in (light, firm, soft, heavy)]
    assert means[0] < 0.1 < means[1] < 0.4 < means[2] <= means[3] < 1.0
    assert means[2] > 0.6
    for sheet in (light, heavy):
        shine = sheet.shine()
        assert shine.shape == (150, 200)
        assert shine.min() >= 0.0 and shine.max() <= 1.0
    # nothing shines outside the patch
    assert not heavy.shine()[:10].any()


def test_charcoal_does_not_shine_and_its_dust_dulls_graphite():
    assert not _shaded(charcoal.COMPRESSED, 1.0, 2, 8).shine().any()
    sheet = _shaded(pencils.HB, 0.5)
    before = float(sheet.shine()[INSIDE].mean())
    sheet.shade(BOX, pastels.soft("#c0392b"), 0.5, 10, angle=-20)
    assert float(sheet.shine()[INSIDE].mean()) < 0.5 * before


def test_rubbing_graphite_in_with_a_stump_does_not_dull_it():
    sheet = _shaded(pencils.B6, 0.6)
    before = float(sheet.shine()[INSIDE].mean())
    for k in range(12):
        sheet.smudge([(30, 40 + 7 * k), (170, 40 + 7 * k)], tools.STUMP, 0.7)
    # (the stump carries a little of the graphite off with it)
    assert float(sheet.shine()[INSIDE].mean()) >= 0.97 * before


# --------------------------------------------------------------------------
# under the lamp


def test_heavy_graphite_mirrors_a_high_lamp_and_not_a_raking_one():
    sheet = _shaded(pencils.B6, 1.0, 3)
    white = _tone(_sheet().render())
    matte = _tone(sheet.render(gloss=0)) / white
    shiny = _tone(sheet.render()) / white
    assert matte < 0.07
    assert 0.02 < shiny - matte < 0.06
    gains = []
    for lamp in (RAKING, None, OVERHEAD):
        gain = _tone(sheet.render(light=lamp)) - _tone(
            sheet.render(light=lamp, gloss=0)
        )
        gains.append(gain)
    assert 0.0 <= gains[0] < 0.2 * gains[1]
    assert gains[1] < gains[2]
    # twice the gloss, twice the shine
    double = _tone(sheet.render(gloss=2)) / white
    assert double - matte == pytest.approx(2 * (shiny - matte), rel=0.02)


def test_a_light_layer_hardly_shines():
    sheet = _shaded(pencils.HB, 0.3)
    gain = _tone(sheet.render()) - _tone(sheet.render(gloss=0))
    assert 0.0 < gain < 0.005


def test_the_shine_has_no_colour():
    sheet = _shaded(pencils.B6, 1.0, 3)
    with_shine = sp.srgb_to_linear(sheet.render()[INSIDE].astype(np.float64))
    without = sp.srgb_to_linear(sheet.render(gloss=0)[INSIDE].astype(np.float64))
    added = (with_shine - without).mean(axis=(0, 1))
    assert added.min() > 0.02
    assert added.max() - added.min() < 0.002


def test_what_does_not_shine_renders_the_same_whatever_the_gloss():
    sheet = _shaded(charcoal.COMPRESSED, 0.8, 1, 8)
    sheet.shade(BOX, pencils.coloured("#2456a6"), 0.7, angle=-20)
    assert np.array_equal(sheet.render(), sheet.render(gloss=0))
    assert np.array_equal(sheet.render(), sheet.render(gloss=3))
    blank = _sheet()
    assert np.array_equal(blank.render(), blank.render(gloss=0))


def test_a_flat_render_has_no_shine():
    sheet = _shaded(pencils.B6, 1.0, 2)
    assert np.array_equal(
        sheet.render(lighting=False), sheet.render(lighting=False, gloss=5)
    )


def test_gloss_must_be_finite_and_not_negative():
    sheet = _sheet()
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gloss"):
            sheet.render(gloss=bad)
    with pytest.raises(ValueError, match="relief"):
        sheet.render(relief=-1.0)
