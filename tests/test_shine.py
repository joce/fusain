"""What shines: graphite, polished wax and heavy ballpoint ink. What carries
each, what hides it and how a lamp shows it, measured."""

import itertools

import numpy as np
import pytest
from murex import spectral as sp
from murex.filters import box_blur

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
from fusain.sheet import (
    INK_FULL,
    IV,
    IX,
    SHEEN_FLAT,
    SHEEN_FULL,
    SHEEN_HIDE,
    SHINES,
)

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
# a film of graphite on a bed of dust (fusain-q5k)

PALE = "#f1e9d8"
#: Lines across the patch, 4 wide; the shine is taken along their middles.
LINES = [[(30, y), (170, y)] for y in range(36, 124, 14)]
ALONG = np.zeros((150, 200), bool)
for _path in LINES:
    ALONG[int(_path[0][1]) - 1 : int(_path[0][1]) + 1, 60:140] = True


def _bed(fixed=False, layers=2, pressure=0.6) -> Sheet:
    """Layers of a pale soft pastel, loose or fixed."""
    sheet = _sheet()
    for layer in range(layers):
        sheet.shade(BOX, pastels.soft(PALE), pressure, 14.0, angle=30 - 60 * layer)
    if fixed:
        sheet.fix()
    return sheet


def _lined(sheet, medium, pressure) -> Sheet:
    for path in LINES:
        sheet.stroke(path, medium, pressure, 4.0)
    return sheet


def _held_shine(sheet) -> np.ndarray:
    """The shine of what the paper holds alone, through the matte dust of
    both layers: all the shine of graphite there was before a film shone."""
    cover = np.maximum(sheet.cover, np.float32(1e-6))
    full = np.float32(SHEEN_FULL) * (1.0 - np.float32(SHEEN_FLAT) * sheet.press)
    film = np.minimum(sheet.held[..., IX] / cover / full, 1.0)
    dust = sheet.under[..., IV] + sheet.loose[..., IV]
    matte = np.maximum(dust - sheet.under[..., IX] - sheet.loose[..., IX], 0.0)
    return sheet.cover * film * np.exp(-matte / cover / np.float32(SHEEN_HIDE))


def _sum(array) -> float:
    return float(np.asarray(array).sum(dtype=np.float64))


def _filmed(sheet) -> np.ndarray:
    return (sheet.film > 0).any(axis=-1)


def _over_pastel(medium, pressure, fixed=False) -> tuple[Sheet, Sheet]:
    """The lines on bare paper, and over two layers of pastel."""
    return _lined(_sheet(), medium, pressure), _lined(_bed(fixed), medium, pressure)


def _shine_along(sheet) -> float:
    return float(sheet.shine("graphite")[ALONG].mean())


@pytest.mark.parametrize(
    ("medium", "pressure"),
    [(pencils.HB, 1.0), (pencils.B2, 1.0), (pencils.B6, 0.6), (pencils.B6, 1.0)],
)
@pytest.mark.parametrize("fixed", [False, True])
def test_a_pencil_line_drawn_firmly_over_pastel_shines_as_far_as_it_is_a_film(
    medium, pressure, fixed
):
    on_paper, over = _over_pastel(medium, pressure, fixed)
    shine, bare = _shine_along(over), _shine_along(on_paper)
    # all but a whisper of it is a film on the pastel, which did not shine
    # (a 2B at 1.0: 0.001, against 0.431 on bare paper)
    assert _sum(over.held[..., IX][ALONG]) < 0.02 * _sum(on_paper.held[..., IX][ALONG])
    # now it shines as the line on paper does, as far as there is as much
    # of it: 0.88 to 0.90 as much of it, and the shine 0.87 to 0.90 as much
    # (a 2B 0.381, a 6B 0.684), the clay of its crumbs over it as on paper
    share = _sum(over.film[ALONG]) / _sum(on_paper.held[..., IV][ALONG])
    assert share - 0.05 < shine / bare < share + 0.04
    # and the lamp shows it, in proportion
    glint, glint_bare = (
        _tone(s.render(light=OVERHEAD), ALONG)
        - _tone(s.render(light=OVERHEAD, gloss=0), ALONG)
        for s in (over, on_paper)
    )
    assert glint == pytest.approx(glint_bare * shine / bare, rel=0.1)


def test_a_light_line_over_pastel_shines_less_than_its_film_would():
    # a light pass buries little of the pastel on top, which stays in the
    # layer with its film: it cannot be told from pastel laid over the film,
    # and is taken to hide it (2B: 0.33 of the shine on bare paper at 0.3,
    # 0.80 at 0.6, 0.88 at 1.0; 0.87 to 0.90 as much of it is a film)
    ratios = []
    for pressure in (0.3, 0.6, 1.0):
        on_paper, over = _over_pastel(pencils.B2, pressure)
        share = _sum(over.film[ALONG]) / _sum(on_paper.held[..., IV][ALONG])
        ratios.append(_shine_along(over) / _shine_along(on_paper) / share)
    assert 0.2 < ratios[0] < ratios[1] < ratios[2] <= 1.0
    assert ratios[1] > 0.8 and ratios[2] > 0.95


def test_layers_of_a_pencil_over_pastel_shine_as_they_do_on_paper():
    on_paper, over = _sheet(), _bed()
    for layer in range(6):
        for sheet in (on_paper, over):
            sheet.shade((40, 40, 160, 110), pencils.B6, 1.0, 6.0, angle=40 + 50 * layer)
    # (layer on layer the films are buried in the pastel underneath, one
    # mixture with it: the graphite there is still the films' and their
    # crumbs'; 0.78 against 0.82)
    shine = float(over.shine("graphite")[INSIDE].mean())
    bare = float(on_paper.shine("graphite")[INSIDE].mean())
    assert 0.85 * bare < shine < bare


def test_a_line_on_bare_paper_shines_as_it_did():
    sheet = _lined(_sheet(), pencils.B4, 0.8)
    sheet.shade(BOX, pencils.HB, 0.5, angle=-30)
    for k in range(6):
        sheet.smudge([(30, 40 + 14 * k), (170, 40 + 14 * k)], tools.STUMP, 0.7)
    assert not sheet.film.any()
    assert np.array_equal(sheet.shine("graphite"), _held_shine(sheet))


def test_where_there_is_no_film_or_there_is_wax_the_shine_is_as_it_was():
    sheet = _lined(_bed(layers=1), pencils.B2, 0.8)
    sheet.shade((100, 20, 180, 130), crayons.wax("#c0392b"), 0.6, 8)
    sheet.shade((100, 20, 180, 130), pencils.B6, 0.8, 6, angle=-60)
    shine, held = sheet.shine("graphite"), _held_shine(sheet)
    as_it_was = ~_filmed(sheet) | (sheet.held[..., IW] > 0)
    assert as_it_was.any() and not as_it_was.all()
    assert np.array_equal(shine[as_it_was], held[as_it_was])
    # (a film where the pixel has wax is taken for one of wax)
    waxed = _filmed(sheet) & (sheet.held[..., IW] > 0)
    assert waxed.sum() > 500
    assert (shine[~as_it_was] > held[~as_it_was]).mean() > 0.9
    assert shine.max() <= 1.0 and (shine <= sheet.cover).all()


def test_loose_graphite_dust_moved_by_a_tool_does_not_shine():
    for fixed in (False, True):
        for tool in (tools.STUMP, tools.FINGER):
            sheet = _bed(fixed)
            sheet.stroke([(30, 78), (170, 78)], pencils.B6, 1.0, 4.0)
            line = (slice(76, 80), slice(60, 140))
            before = float(sheet.shine("graphite")[line].mean())
            film = _sum(sheet.film[line])
            for x in range(50, 150, 6):
                sheet.smudge([(x, 66), (x, 98)], tool, 0.8)
            # what it spreads is dust where it is laid, no film: what shines
            # there is what the paper holds of it, as it was
            beside = (slice(86, 100), slice(60, 140))
            assert _sum(sheet.under[..., IX][beside] + sheet.loose[..., IX][beside]) > 1
            assert not sheet.film[beside].any()
            assert np.array_equal(
                sheet.shine("graphite")[beside], _held_shine(sheet)[beside]
            )
            # on loose pastel the film comes away with the pastel, and its
            # shine with it; on fixed pastel it stays, and so does its shine,
            # in proportion (0.83 of the film is left, and 0.87 and 0.81 of
            # the shine: the stump took its crumbs off it, the finger pressed
            # its own dust in over it)
            after = float(sheet.shine("graphite")[line].mean())
            if fixed:
                left = _sum(sheet.film[line]) / film
                assert 0.85 * left * before < after < 1.1 * left * before
            else:
                assert after < 0.02 * before


def test_a_film_of_charcoal_does_not_shine_and_dust_over_a_film_of_graphite_dulls_it():
    assert not _lined(_bed(), charcoal.PENCIL, 0.8).shine().any()
    dulled = []
    for pressure in (0.5, 0.8):
        sheet = _lined(_bed(), pencils.B6, 1.0)
        before = _shine_along(sheet)
        sheet.shade(BOX, pastels.soft("#c0392b"), pressure, 10, angle=-20)
        dulled.append(_shine_along(sheet) / before)
    # (pastel over pastel lays less than over a line on bare paper, which
    # it dulls to 0.18 and 0.06: here 0.55 and 0.23)
    assert dulled[1] < 0.3 and dulled[1] < dulled[0] < 0.65


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
    # of a pen running dry is not heavy ink; neither is pressed hard enough
    # to flatten or polish anything, even where its tip bears hardest)
    sheet.shade(BOX, charcoal.COMPRESSED, 0.25, 8, hand=0.0)
    sheet.shade(BOX, pencils.coloured("#2456a6"), 0.25, angle=-20, hand=0.0)
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


def test_polished_wax_has_a_soft_white_gloss_where_it_mirrors_the_lamp():
    sheet = _burnished()
    white = _tone(_sheet().render())
    added = _added(sheet, OVERHEAD)
    # white light, a few hundredths of the paper's
    assert added.min() / white > 0.012 and added.max() / white < 0.035
    assert added.max() - added.min() < 0.003
    # the colour is still the red's
    assert np.argmax(sheet.render()[INSIDE].mean(axis=(0, 1))) == 0
    # twice the gloss, twice as much; none, as a scanner sees it; flat, none
    lit = sheet.render(light=OVERHEAD, gloss=2)[INSIDE].astype(np.float64)
    matte = sheet.render(light=OVERHEAD, gloss=0)[INSIDE].astype(np.float64)
    double = sp.srgb_to_linear(lit) - sp.srgb_to_linear(matte)
    assert double.mean() == pytest.approx(2 * added.mean(), rel=0.03)
    assert np.array_equal(
        sheet.render(lighting=False), sheet.render(lighting=False, gloss=3)
    )


def test_the_gloss_of_wax_is_fainter_and_narrower_than_the_shine_of_graphite():
    """Polished wax is satin; graphite is a broad metallic sheen."""
    wax, graphite = _burnished(), _shaded(pencils.B6, 1.0, 3)
    kept = []  # (of what it adds overhead, what it adds under the other lamps)
    for sheet in (wax, graphite):
        added = [_added(sheet, lamp).mean() for lamp in (RAKING, None, OVERHEAD)]
        assert 0.0 <= added[0] < added[1] < added[2]
        kept.append((added[0] / added[2], added[1] / added[2]))
    # under the default lamp the wax keeps a fifth of its gloss, graphite
    # most of its shine; under a raking one the wax none, graphite a little
    assert 0.15 < kept[0][1] < 0.3 and kept[1][1] > 0.7
    assert kept[0][0] < 1e-3 and kept[1][0] > 0.05
    # and overhead it is fainter than graphite's, for as much of it
    per_shine = [
        _added(sheet, OVERHEAD).mean() / float(sheet.shine()[INSIDE].mean())
        for sheet in (wax, graphite)
    ]
    assert per_shine[0] < 0.7 * per_shine[1]


def _lightness(sheet, lamp=None, gloss=1.0, where=INSIDE) -> np.ndarray:
    """L* of each pixel under a lamp."""
    image = sheet.render(light=lamp, gloss=gloss)[where].astype(np.float64)
    y = np.maximum(sp.srgb_to_linear(image) @ sp.LUMINANCE, 0.0)
    return np.where(y > 0.008856, 116.0 * np.cbrt(y) - 16.0, 903.3 * y)


def test_a_burnished_dark_stays_deep_under_the_default_lamp():
    # (fusain-5je) four firm layers of a dark coloured pencil, then burnished
    # with the blender: the gloss took it from L* 7 to 18 under the default
    # lamp, and the firm layers alone from 10 to 18, following how far each
    # stroke had polished them (by 1.5 from place to place)
    firm = _shaded(pencils.coloured("#141416"), 0.8, 4)
    burnished = firm.copy()
    for angle in (20, -50):
        burnished.shade(BOX, pencils.BLENDER, 1.0, angle=angle)
    added = {}
    for name, sheet in (("firm", firm), ("burnished", burnished)):
        for lamp, light in (("default", None), ("overhead", OVERHEAD)):
            change = _lightness(sheet, light) - _lightness(sheet, light, 0.0)
            added[name, lamp] = change
    assert _lightness(burnished, gloss=0.0).mean() < 8.0
    assert 0.0 < added["burnished", "default"].mean() < 4.0
    assert 0.0 < added["firm", "default"].mean() < 3.0
    # a soft, even gloss: smoothed over the grain, it varies by little
    smooth = box_blur(added["firm", "default"].astype(np.float32), 2)
    assert smooth.std() < 0.8
    # with the lamp where the wax mirrors it, it shines
    assert added["burnished", "overhead"].mean() > 2.5 * (
        added["burnished", "default"].mean()
    )
    assert added["burnished", "overhead"].mean() > 9.0


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
