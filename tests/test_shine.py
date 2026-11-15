"""What shines: graphite, polished wax and heavy ballpoint ink. What carries
each, what hides it and how a lamp shows it, measured."""

import itertools

import numpy as np
import pytest
from murex import spectral as sp

from fusain import (
    Medium,
    Sheet,
    charcoal,
    crayons,
    papers,
    pastels,
    pencils,
    pens,
    tools,
)
from fusain._engine import BALL_INK, IW
from fusain.sheet import INK_FULL, IV, IX, SHINES

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


def test_only_graphite_has_a_sheen_of_its_own_and_a_soft_lead_has_more():
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
    for kind in SHINES:
        assert not _sheet().shine(kind).any()
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
    sheet = _sheet()
    # (charcoal has no sheen, wax laid lightly is not polished, and one line
    # of a pen running dry is not heavy ink)
    sheet.shade(BOX, charcoal.COMPRESSED, 0.4, 8, hand=0.0)
    sheet.shade(BOX, pencils.coloured("#2456a6"), 0.4, angle=-20, hand=0.0)
    sheet.stroke([(30, 75.5), (170, 75.5)], pens.BLUE.with_(flow=0.3), 0.5)
    assert sheet.held[..., IW].any() and sheet.ink is not None
    assert not sheet.shine().any()
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


# --------------------------------------------------------------------------
# polished wax

RED = "#c0392b"


def _gloss(sheet) -> float:
    return float(sheet.shine("wax")[INSIDE].mean())


def _added(sheet, lamp=None) -> np.ndarray:
    """The light the shine adds under a lamp, per channel (linear)."""
    lit = sp.srgb_to_linear(sheet.render(light=lamp)[INSIDE].astype(np.float64))
    matte = sp.srgb_to_linear(
        sheet.render(light=lamp, gloss=0)[INSIDE].astype(np.float64)
    )
    return (lit - matte).mean(axis=(0, 1))


def _burnished() -> Sheet:
    sheet = _shaded(pencils.coloured(RED), 0.5, 3)
    for angle in (20, -50):
        sheet.shade(BOX, pencils.BLENDER, 1.0, angle=angle)
    return sheet


def test_wax_laid_lightly_is_dull_and_burnished_it_is_glossy():
    light = _shaded(pencils.coloured(RED), 0.5, 3)
    assert _gloss(light) < 0.01
    firm = light.copy()
    firm.shade(BOX, pencils.coloured(RED), 1.0, angle=10)
    burnished = _burnished()
    assert 0.05 < _gloss(firm) < _gloss(burnished)
    assert _gloss(burnished) > 0.6
    gloss = burnished.shine("wax")
    assert gloss.min() >= 0.0 and gloss.max() <= 1.0
    assert not gloss[:10].any()
    # the wax has no sheen as graphite has: it is its polish that shines
    assert not burnished.shine("graphite").any()
    assert np.abs(burnished.shine() - gloss).max() < 1e-6


@pytest.mark.parametrize(
    "make, width", [(pencils.coloured, None), (crayons.wax, 9), (pastels.oil, 14)]
)
def test_the_gloss_of_wax_grows_with_how_hard_it_is_pressed(make, width):
    gloss, press = [], []
    for pressure in (0.3, 0.6, 0.8, 1.0):
        sheet = _shaded(make(RED), pressure, 2, width)
        gloss.append(_gloss(sheet))
        press.append(float(sheet.press[INSIDE].mean()))
    assert gloss[0] < 1e-3
    assert all(a < b for a, b in itertools.pairwise(gloss))
    assert all(a < b for a, b in itertools.pairwise(press))
    assert gloss[1] < 0.05 and gloss[3] > 0.25
    # it is the polish that shines: no more gloss than the tooth is pressed
    assert all(g <= p + 1e-6 for g, p in zip(gloss, press, strict=True))


def test_the_gloss_is_the_polish_of_the_wax_that_is_there():
    sheet = _burnished()
    gloss = _gloss(sheet)
    rough = sheet.copy()
    rough.press[:] = 0.0
    assert not rough.shine("wax").any()
    # (a film half polished still scatters most of the light)
    half = sheet.copy()
    half.press *= 0.5
    assert _gloss(half) == pytest.approx(0.25 * gloss, rel=1e-5)
    # a tooth pressed flat by a hard pencil has no wax on it to shine
    assert not _shaded(pencils.H4, 1.0, 4).shine("wax").any()


def test_wax_shines_where_it_lies_on_dust_as_on_the_paper():
    sheet = _sheet()
    for angle in (30, -40):
        sheet.shade(BOX, charcoal.VINE, 0.7, 12, angle)
    for layer in range(2):
        sheet.shade(BOX, crayons.wax(RED), 0.9, 9, angle=30 + 55 * layer)
    assert float(sheet.film[..., 1][INSIDE].mean()) > 0.05  # a film, on the dust
    gloss = _gloss(sheet)
    assert gloss > 0.1
    # were that film dust like the charcoal under it, it would hide the
    # gloss of the wax the paper holds
    buried = sheet.copy()
    buried.film[:] = 0.0
    assert _gloss(buried) < 0.5 * gloss


def test_dust_over_polished_wax_dulls_it():
    sheet = _shaded(crayons.wax(RED), 0.9, 2, 9)
    gloss = _gloss(sheet)
    assert gloss > 0.5
    dusty = sheet.copy()
    dusty.loose[..., IV] += 0.3 * dusty.cover  # (a layer of dust, as of pastel)
    assert _gloss(dusty) < 0.1 * gloss


def test_polished_wax_has_a_soft_white_gloss_under_the_lamp():
    sheet = _burnished()
    white = _tone(_sheet().render())
    added = _added(sheet)
    # white light, a few hundredths of the paper's
    assert added.min() / white > 0.012 and added.max() / white < 0.035
    assert added.max() - added.min() < 0.003
    # the colour is still the red's
    assert np.argmax(sheet.render()[INSIDE].mean(axis=(0, 1))) == 0
    # twice the gloss, twice as much; none, as a scanner sees it; flat, none
    double = sp.srgb_to_linear(sheet.render(gloss=2)[INSIDE].astype(np.float64))
    matte = sp.srgb_to_linear(sheet.render(gloss=0)[INSIDE].astype(np.float64))
    assert (double - matte).mean() == pytest.approx(2 * added.mean(), rel=0.03)
    assert np.array_equal(
        sheet.render(lighting=False), sheet.render(lighting=False, gloss=3)
    )


def test_the_gloss_of_wax_is_softer_and_wider_than_the_shine_of_graphite():
    wax, graphite = _burnished(), _shaded(pencils.B6, 1.0, 3)
    wide = []
    for sheet in (wax, graphite):
        raking, overhead = _added(sheet, RAKING).mean(), _added(sheet, OVERHEAD).mean()
        assert 0.0 < raking < overhead
        wide.append(raking / overhead)
    # under a raking lamp the wax keeps more of its gloss: it is wider
    assert wide[0] > 3 * wide[1]
    # and overhead it is fainter than graphite's, for as much of it
    per_shine = [
        _added(sheet, OVERHEAD).mean() / float(sheet.shine()[INSIDE].mean())
        for sheet in (wax, graphite)
    ]
    assert per_shine[0] < 0.7 * per_shine[1]


# --------------------------------------------------------------------------
# heavy ink


def _hatched(pen, layers, pressure=0.7) -> Sheet:
    sheet = _sheet()
    for layer in range(layers):
        sheet.shade(BOX, pen, pressure, angle=35 + 47 * layer)
    return sheet


def _bronze(sheet) -> float:
    return float(sheet.shine("ink")[INSIDE].mean())


def test_ink_bronzes_where_it_is_heavy():
    bronze = [_bronze(_hatched(pens.BLUE, layers)) for layers in (1, 2, 4, 6, 8)]
    assert bronze[0] < 0.002 and bronze[1] < 0.01  # (but for a blob or two)
    assert all(a < b for a, b in itertools.pairwise(bronze))
    assert bronze[2] > 0.03 and bronze[4] > 0.35
    sheet = _hatched(pens.BLUE, 8)
    shine = sheet.shine("ink")
    assert shine.min() >= 0.0 and shine.max() <= 1.0
    assert not shine[:10].any()
    # it is the ink that is thick enough, on the share of the paper it is on
    assert sheet.ink is not None and sheet.inked is not None
    thick = sheet.ink[..., IV] / np.maximum(sheet.inked, 1e-6)
    full = thick > INK_FULL * BALL_INK
    assert full.any() and np.allclose(shine[full], sheet.inked[full])
    assert not sheet.shine("wax").any() and not sheet.shine("graphite").any()


def test_what_is_drawn_over_heavy_ink_hides_its_bronze():
    sheet = _hatched(pens.BLUE, 8)
    bronze = _bronze(sheet)
    for layer in range(2):
        sheet.shade(BOX, pastels.soft("#f4f1ea"), 0.8, 12, angle=30 + 50 * layer)
    assert _bronze(sheet) < 0.1 * bronze


def test_blue_ink_bronzes_copper_red_ink_green_and_black_ink_violet():
    """Ink mirrors most the light it absorbs most; a black ink is a blend of
    dyes, and mirrors as its violet does, copper with a violet cast. (It
    bronzed grey: red, green and blue within a tenth of one another.)"""
    blue = _added(_hatched(pens.BLUE, 8), OVERHEAD)
    assert blue[0] > 1.3 * blue[1] > 3 * blue[2] > 0.0
    assert blue[0] > 0.01
    red = _added(_hatched(pens.RED, 8), OVERHEAD)
    assert red[1] > 0.01 and red[2] > 0.01 and red[0] < 0.1 * red[1]
    black = _added(_hatched(pens.BLACK, 8), OVERHEAD)
    assert black[0] > 1.5 * black[2] > 1.5 * 1.2 * black[1] > 0.0
    assert black[0] > 0.01


def test_the_bronze_shows_with_the_lamp_overhead_and_hardly_otherwise():
    sheet = _hatched(pens.BLUE, 8)
    overhead = _added(sheet, OVERHEAD).sum()
    usual, raking = _added(sheet).sum(), _added(sheet, RAKING).sum()
    assert 0.0 <= raking < 0.02 * overhead
    assert 0.0 < usual < 0.25 * overhead
    assert np.array_equal(
        sheet.render(lighting=False), sheet.render(lighting=False, gloss=3)
    )
    assert _added(_hatched(pens.BLUE, 1), OVERHEAD).max() < 1e-3


def test_the_three_shines_together():
    sheet = _sheet()
    sheet.shade((20, 20, 70, 130), pencils.B6, 1.0)
    for layer in range(2):
        sheet.shade((75, 20, 125, 130), crayons.wax(RED), 0.95, 9, 30 + 55 * layer)
    for layer in range(8):
        sheet.shade((130, 20, 180, 130), pens.BLACK, 0.8, angle=35 + 47 * layer)
    parts = [sheet.shine(kind) for kind in SHINES]
    for kind, part, cols in zip(SHINES, parts, (45, 100, 155), strict=True):
        where = (slice(50, 100), slice(cols - 10, cols + 10))
        assert part[where].mean() > 0.3, kind
        for other in parts:
            if other is not part:
                assert other[where].mean() < 0.01
    together = sheet.shine()
    assert np.all(together >= np.max(parts, axis=0) - 1e-6)
    assert together.max() <= 1.0
    with pytest.raises(ValueError, match="graphite"):
        sheet.shine("gold")
