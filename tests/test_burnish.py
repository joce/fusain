"""Coloured pencil worked over: the blender, the eraser on wax, the knife,
and a dark over a light (issue fusain-rz1), measured."""

import numpy as np
import pytest
from murex import spectral as sp
from murex.filters import box_blur

from fusain import Paper, Sheet, Smudger, crayons, papers, pastels, pencils, tools
from fusain._engine import (
    BURNISH_FROM,
    BURNISH_SOFT,
    IV,
    IW,
    POLISH_HOLDS,
    WAX_FAST,
    WAX_HOLDS,
    firmness,
    wax_grip,
)

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
RED, YELLOW, BLUE, DARK = "#c0392b", "#f2c40f", "#2456a6", "#141416"
#: A smooth drawing paper with a shallow tooth, as examples/macaw.py uses.
SMOOTH = Paper("wove", grain=2.0, depth=0.4, bite=0.5, give=0.4, color="#fbfaf6")


def _sheet(paper=papers.CARTRIDGE) -> Sheet:
    return Sheet(200, 150, paper=paper, seed=1)


def _layers(sheet, medium, count=2, pressure=0.7, width=6.0) -> float:
    return sum(
        sheet.shade(BOX, medium, pressure, width, angle=30 - 55 * layer)
        for layer in range(count)
    )


def _luminances(sheet, where=INSIDE) -> np.ndarray:
    spectrum = sheet.reflectance()[where].astype(np.float64)
    return sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE


def _paper(sheet) -> float:
    return float(sp.reflectance_to_linear_rgb(sheet.ground) @ sp.LUMINANCE)


def _flecks(sheet, where=INSIDE) -> float:
    """The share of the pixels that still show the paper: as light, or
    nearly, as the bare sheet."""
    return float((_luminances(sheet, where) > 0.9 * _paper(sheet)).mean())


def _lab(sheet, where=INSIDE) -> np.ndarray:
    """The colour of the area as a whole (L, a, b)."""
    spectrum = sheet.reflectance()[where].astype(np.float64).mean(axis=(0, 1))
    return sp.srgb_to_lab(sp.reflectance_to_srgb(spectrum))


def _chroma(sheet, where=INSIDE) -> float:
    return float(np.hypot(*_lab(sheet, where)[1:]))


def _gloss(sheet, where=INSIDE) -> float:
    return float(sheet.shine("wax")[where].mean())


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


def _left(sheet, before, where=INSIDE) -> float:
    """The share of the pigment of an area that is still there."""
    return float(sheet.pigment()[where].sum() / before.pigment()[where].sum())


# --------------------------------------------------------------------------
# the blender


def test_one_firm_layer_of_the_blender_closes_the_flecks_and_deepens_the_colour():
    sheet = _sheet()
    _layers(sheet, pencils.coloured(RED), 3, 0.5)
    flecks, chroma, light = _flecks(sheet), _chroma(sheet), _lab(sheet)[0]
    grain, pigment = _luminances(sheet).std(), _total(sheet)
    assert flecks > 0.08 and _gloss(sheet) < 0.01
    lost = sheet.shade(BOX, pencils.BLENDER, 1.0, angle=20)
    # a fifth of the white flecks are left; the red is far more saturated,
    # deeper, and half as grainy; and it has a gloss
    assert _flecks(sheet) < 0.25 * flecks
    assert _chroma(sheet) > 1.5 * chroma
    assert _lab(sheet)[0] < light - 5.0
    assert _luminances(sheet).std() < 0.6 * grain
    assert _gloss(sheet) > 0.35
    # it is the red still, and nothing was lost: the blender added its wax
    mean = sheet.reflectance()[INSIDE].mean((0, 1))
    assert np.argmax(sp.reflectance_to_srgb(mean)) == 0
    np.testing.assert_allclose(_total(sheet), pigment + lost, rtol=1e-5)
    # a second layer leaves next to none
    sheet.shade(BOX, pencils.BLENDER, 1.0, angle=-50)
    assert _flecks(sheet) < 0.05 * flecks
    assert _chroma(sheet) > 1.8 * chroma
    assert _gloss(sheet) > 0.7


def test_the_blender_works_as_hard_as_it_is_pressed():
    flecks, chroma, gloss = [], [], []
    for pressure in (0.5, 0.7, 0.85, 1.0):
        sheet = _sheet()
        _layers(sheet, pencils.coloured(RED), 3, 0.5)
        sheet.shade(BOX, pencils.BLENDER, pressure, angle=20)
        flecks.append(_flecks(sheet))
        chroma.append(_chroma(sheet))
        gloss.append(_gloss(sheet))
    assert flecks[0] > flecks[1] > flecks[2] > flecks[3]
    assert chroma[0] < chroma[1] < chroma[2] < chroma[3]
    # laid lightly it burnishes nothing; it has to be pressed
    assert gloss[0] < 0.01 and flecks[0] > 0.9 * 0.107
    assert flecks[1] > 2.5 * flecks[3] and chroma[3] > chroma[1] + 10.0


def test_the_blender_shows_at_drawing_sizes_on_a_smooth_paper():
    # (a fine grain and a shallow tooth, light layers: where the issue found
    # that it made next to no difference)
    sheet = _sheet(SMOOTH)
    _layers(sheet, pencils.coloured(RED), 2, 0.2)
    flecks, chroma = _flecks(sheet), _chroma(sheet)
    grain = _luminances(sheet).std()
    assert flecks > 0.03
    sheet.shade(BOX, pencils.BLENDER, 1.0, angle=20)
    assert _flecks(sheet) < 0.15 * flecks
    assert _chroma(sheet) > 1.25 * chroma
    assert _luminances(sheet).std() < 0.5 * grain
    assert _gloss(sheet) > 0.4


def test_the_blender_blends_two_colours_laid_over_each_other():
    sheet = _sheet()
    _layers(sheet, pencils.coloured(YELLOW), 2, 0.5)
    _layers(sheet, pencils.coloured(BLUE), 2, 0.5)

    def spread() -> float:
        """How far the pixels differ in colour: from yellow to blue."""
        spectrum = sheet.reflectance()[INSIDE].astype(np.float64)[::3, ::3]
        rgb = sp.reflectance_to_srgb(spectrum.reshape(-1, spectrum.shape[-1]))
        return float(np.std([sp.srgb_to_lab(c)[2] for c in rgb]))

    mottled, chroma = spread(), _chroma(sheet)
    sheet.shade(BOX, pencils.BLENDER, 1.0, angle=20)
    r, g, b = sp.reflectance_to_srgb(sheet.reflectance()[INSIDE].mean((0, 1)))
    assert g > r + 0.1 and g > b + 0.1  # one green
    assert spread() < 0.7 * mottled
    assert _chroma(sheet) > 1.5 * chroma


def test_the_blender_drags_colour_across_an_edge_but_not_far():
    sheet = _sheet()
    for angle in (30, -25, 80):
        sheet.shade((20, 20, 100, 130), pencils.coloured(DARK), 0.6, angle=angle)
    rows = slice(45, 105)

    def dark(columns) -> float:
        """Absorption per pixel there: the dark pigment, not the blender."""
        return float(sheet.held[rows, columns, :10].mean(dtype=np.float64))

    full = dark(slice(50, 85))
    assert dark(slice(108, 200)) == 0.0
    before, lost = _total(sheet), 0.0
    for y in range(40, 112, 3):  # from the dark out over the bare paper
        lost += sheet.stroke([(60, y), (160, y)], pencils.BLENDER, 1.0, 6)
    # a smear past the edge, strongest next to it: a tip's width out there
    # is a tenth of the dark, three widths out next to nothing
    assert 0.1 * full < dark(slice(104, 108)) < 0.5 * full
    assert 0.03 * full < dark(slice(108, 114)) < 0.2 * full
    assert dark(slice(122, 200)) < 0.01 * full
    # the dark itself is still there
    assert dark(slice(50, 85)) > 0.9 * full
    np.testing.assert_allclose(_total(sheet), before + lost, rtol=1e-5)


# --------------------------------------------------------------------------
# a film that lies over dust (fusain-pv2)

WHITE = "#f4f1ea"


def _over_dust(box=BOX) -> Sheet:
    """Three layers of a dark coloured pencil over a near-white soft pastel:
    most of the dark lies on the dust, a film."""
    sheet = _sheet()
    sheet.shade(box, pastels.soft(WHITE), 0.4, 9.0)
    for layer in range(3):
        sheet.shade(box, pencils.coloured(DARK), 0.5, 6.0, angle=30 - 55 * layer)
    return sheet


def _absorption(layer) -> float:
    return float(layer[..., :10].sum(dtype=np.float64))


def test_a_firm_pass_works_a_film_that_lies_over_dust_into_the_tooth():
    sheet = _over_dust()
    lying = _absorption(sheet.under) + _absorption(sheet.loose)
    held = _absorption(sheet.held)
    assert lying > 3.0 * held  # (the dark is on the dust)
    pigment, wax = _total(sheet), float(sheet.held[..., IW].sum(dtype=np.float64))
    lost = sheet.shade(BOX, pencils.BLENDER, 1.0, angle=20)
    # it drags the film along as it drags what the paper holds, and lays it
    # down in the tooth: a fifth of the dark that lay on the dust (before,
    # a hundredth: only the wax of the film, which is counted with what
    # the paper holds, went with it)
    after = _absorption(sheet.under) + _absorption(sheet.loose)
    assert after < 0.85 * lying
    assert _absorption(sheet.held) > held + 0.9 * (lying - after)
    # nothing is lost: the medium, and the wax, which the blender adds to
    np.testing.assert_allclose(_total(sheet), pigment + lost, rtol=1e-5)
    np.testing.assert_allclose(
        float(sheet.held[..., IW].sum(dtype=np.float64)),
        wax + lost * pencils.BLENDER.wax,
        rtol=1e-5,
    )
    assert sheet.bonded.min() >= 0.0


def test_the_blender_closes_the_flecks_of_a_film_over_dust_as_of_one_in_the_tooth():
    flecks = []
    for dust in (pastels.soft(BLUE), None):
        sheet = _sheet()
        if dust is not None:
            sheet.shade(BOX, dust, 0.3, 9.0)
        _layers(sheet, pencils.coloured(RED), 3, 0.5)
        before = _flecks(sheet)
        sheet.shade(BOX, pencils.BLENDER, 1.0, angle=20)
        flecks.append(_flecks(sheet) / before)
    # (a quarter of them left over the dust, a fifth over the paper)
    assert flecks[0] < 0.3 and flecks[1] < 0.25


def test_the_blender_drags_a_film_over_dust_across_an_edge_as_wax_in_the_tooth():
    def smeared(over_dust: bool) -> float:
        """What lies just past the edge after the blender is drawn out
        across it, as a share of the dark."""
        box = (20, 20, 100, 130)
        if over_dust:
            sheet = _over_dust(box)
        else:
            sheet = _sheet()
            for layer in range(3):
                sheet.shade(
                    box, pencils.coloured(DARK), 0.5, 6.0, angle=30 - 55 * layer
                )
        rows = slice(45, 105)

        def dark(columns) -> float:
            layers = (sheet.held, sheet.under, sheet.loose)
            return sum(
                float(a[rows, columns, :10].mean(dtype=np.float64)) for a in layers
            )

        full = dark(slice(50, 85))
        for y in range(40, 112, 3):
            sheet.stroke([(60, y), (160, y)], pencils.BLENDER, 1.0, 6)
        return dark(slice(104, 108)) / full

    # (over the dust, half of what it smeared out of the tooth)
    assert smeared(True) > 0.75 * smeared(False) > 0.0


def test_a_white_pencil_pressed_hard_burnishes_nearly_as_the_blender_and_lightens():
    # (fusain-d6p) a pale lead is mostly wax and white filler: pressed hard
    # it melts into the film under it as the blender does, the more the
    # paler it is, and lays its own pale colour as it does. (It burnished
    # only as a red pencil does: twice the flecks of the blender, a third of
    # its gloss, 0.47 of polish to the blender's 0.74)
    base = _sheet()
    _layers(base, pencils.coloured(RED), 3, 0.5)
    flecks = _flecks(base)
    white, blender, red = base.copy(), base.copy(), base.copy()
    white.shade(BOX, pencils.coloured("#ffffff"), 1.0, angle=20)
    blender.shade(BOX, pencils.BLENDER, 1.0, angle=20)
    red.shade(BOX, pencils.coloured(RED), 1.0, angle=20)
    assert _flecks(white) < 0.3 * flecks
    assert _flecks(white) < 1.3 * _flecks(blender)
    assert _gloss(white) > 0.5 * _gloss(blender) > 1.2 * _gloss(red)
    assert white.press[INSIDE].mean() > 0.9 * blender.press[INSIDE].mean()
    assert white.press[INSIDE].mean() > 1.3 * red.press[INSIDE].mean()
    # the blender deepens the colour; the white pencil leaves it a paler
    # tint of it, as light as it was
    lightness, chroma = _lab(white)[0], _chroma(white)
    assert lightness > _lab(blender)[0] + 5.0 and chroma < _chroma(blender) - 10.0
    assert abs(lightness - _lab(base)[0]) < 3.0
    # how far a pencil melts goes with how pale it is
    melts = [pencils.coloured(c).melts for c in ("#ffffff", "#f4f1ea", "#cfe8f5")]
    assert pencils.BLENDER.melts == 1.0 > melts[0] > melts[1] > melts[2] > 0.25
    for tint in ("#e8b4b8", "#a9a9ae"):  # (a pink, a mid grey)
        assert 0.0 < pencils.coloured(tint).melts < 0.1
    for strong in (RED, YELLOW, BLUE, DARK, "#f2e24a"):
        assert pencils.coloured(strong).melts == 0.0
    with pytest.raises(ValueError, match="melts must be between 0 and 1"):
        pencils.BLENDER.with_(melts=1.5)


# --------------------------------------------------------------------------
# where a tip begins to polish


def test_a_tip_begins_to_polish_by_degrees():
    pressures = np.linspace(0.0, 1.0, 401)
    firm = np.array([firmness(p) for p in pressures])
    low, high = BURNISH_FROM - BURNISH_SOFT, BURNISH_FROM + BURNISH_SOFT
    # nothing under a light touch; from there it grows, ever faster but
    # with no corner, to what a sharp onset at BURNISH_FROM gives from high
    assert not firm[pressures <= low].any()
    assert 0.0 < firmness(low + 0.05) < firmness(BURNISH_FROM) < firmness(high)
    slope = np.diff(firm) / np.diff(pressures)
    assert (slope >= 0.0).all() and (np.diff(slope) > -1e-6).all()
    assert np.abs(np.diff(slope)).max() < 0.05 * slope.max()
    sharp = (pressures - BURNISH_FROM) / (1.0 - BURNISH_FROM)
    beyond = pressures >= high
    assert np.allclose(firm[beyond], sharp[beyond])
    assert firmness(1.0) == pytest.approx(1.0)


def _patchy(sheet, where=INSIDE) -> float:
    """How unevenly an area is polished from stroke to stroke: the spread of
    its ``press``, smoothed over a few pixels (the grain of the paper), as a
    share of its mean."""
    press = box_blur(sheet.press[where], 3).astype(np.float64)
    return float(press.std() / press.mean())


@pytest.mark.parametrize(
    "make, width, most",
    [(pencils.coloured, 6.0, (0.7, 0.65, 0.6)), (pastels.oil, 14.0, (0.9, 0.8, 0.72))],
)
def test_strokes_near_where_a_tip_begins_to_polish_polish_alike(make, width, most):
    # (fusain-bre) the strokes of a shading bear each a little firmer or
    # lighter than the next; around where a tip begins to polish, some of
    # them did and some did not, and the area was polished in streaks
    # (two layers of a dark: 0.91, 0.78 and 0.68 for coloured pencil, 1.34,
    # 1.08 and 0.87 for oil pastel, when it began at 0.6 sharp)
    pressed = []
    for pressure, bound in zip((0.55, 0.6, 0.65), most, strict=True):
        sheet = _sheet()
        _layers(sheet, make(DARK), 2, pressure, width)
        assert _patchy(sheet) < bound, pressure
        pressed.append(float(sheet.press[INSIDE].mean()))
    # and it is polished the more, the harder it is pressed
    assert 0.0 < pressed[0] < pressed[1] < pressed[2]


# --------------------------------------------------------------------------
# erasing wax


def _dense(burnished: bool, paper=papers.CARTRIDGE) -> Sheet:
    sheet = _sheet(paper)
    _layers(sheet, pencils.coloured(DARK), 4, 0.8)
    if burnished:
        for angle in (20, -50):
            sheet.shade(BOX, pencils.BLENDER, 1.0, angle=angle)
    return sheet


AREA = (60, 60, 140, 92)
RUBBED = (slice(66, 86), slice(70, 130))


def test_a_light_layer_of_coloured_pencil_comes_off_a_vinyl_eraser():
    for paper in (papers.CARTRIDGE, SMOOTH):
        sheet = _sheet(paper)
        sheet.shade(BOX, pencils.coloured(DARK), 0.3)
        before = sheet.copy()
        sheet.rub(AREA, tools.ERASER, 0.9)
        assert _left(sheet, before, RUBBED) < 0.2
        # (0.55 before fusain-d6p: the stain it leaves is a tint, and of so
        # dark a pencil a grey)
        assert _luminances(sheet, RUBBED).mean() > 0.45 * _paper(sheet)
        sheet.rub(AREA, tools.ERASER, 0.9)
        # (fusain-d6p: what the wax stained the paper with stays; it took
        # all but 0.007 of it)
        assert 0.04 < _left(sheet, before, RUBBED) < 0.1


@pytest.mark.parametrize("paper", [papers.CARTRIDGE, SMOOTH])
def test_coloured_pencil_leaves_a_stain_that_only_what_cuts_the_paper_takes(paper):
    # (fusain-d6p) wax carries its pigment into the fibres of the paper: a
    # tint stays where a light layer has been erased, however long it is
    # rubbed (it came off to the last thousandth, and the paper was clean)
    bare = _sheet(paper)
    for colour in (DARK, RED):
        sheet = _sheet(paper)
        sheet.shade(BOX, pencils.coloured(colour), 0.3)
        before, scraped = sheet.copy(), sheet.copy()
        for _ in range(3):
            sheet.rub(AREA, tools.ERASER, 0.9)
        stain = _left(sheet, before, RUBBED)
        for _ in range(2):
            sheet.rub(AREA, tools.ERASER, 0.9)
        assert _left(sheet, before, RUBBED) > 0.97 * stain > 0.04
        tint = float(np.linalg.norm(_lab(sheet, RUBBED) - _lab(bare, RUBBED)))
        assert tint > 2.0
        # a vinyl eraser leaves it, and a kneaded one; a sand eraser cuts the
        # surface of the paper, and takes it
        for _ in range(2):
            scraped.rub(AREA, tools.SAND_ERASER, 0.9, 10)
        assert _left(scraped, before, RUBBED) < 0.3 * stain
    # graphite, which has no wax, leaves none
    sheet = _sheet(paper)
    sheet.shade(BOX, pencils.HB, 0.3)
    before = sheet.copy()
    for _ in range(3):
        sheet.rub(AREA, tools.ERASER, 0.9)
    assert _left(sheet, before, RUBBED) < 0.01


def test_a_dense_dark_hardly_lightens_under_an_eraser():
    # (three rubs: what is left of it, and its tone against the paper's)
    for burnished, left, tone in ((False, (0.2, 0.4), 0.3), (True, (0.3, 0.5), 0.05)):
        sheet = _dense(burnished)
        before = sheet.copy()
        dark = _luminances(sheet, RUBBED).mean()
        took = sheet.rub(AREA, tools.ERASER, 0.9)
        assert took > 0.0
        # one rub, to and fro, takes a third of it, and it is as dark as ever
        assert 0.55 < _left(sheet, before, RUBBED) < 0.85
        assert _luminances(sheet, RUBBED).mean() < dark + 0.02
        # three take over half: what kept its tooth begins to lighten, and
        # what was burnished is as dark as it was
        for _ in range(2):
            sheet.rub(AREA, tools.ERASER, 0.9)
        assert left[0] < _left(sheet, before, RUBBED) < left[1]
        assert _luminances(sheet, RUBBED).mean() < tone * _paper(sheet)


def test_burnished_wax_holds_faster_than_wax_that_kept_its_tooth():
    plain, burnished = _dense(False), _dense(True)
    took = [
        sheet.rub(AREA, tools.ERASER, 0.9) / sheet.pigment()[RUBBED].sum()
        for sheet in (plain.copy(), burnished.copy())
    ]
    assert took[1] < 0.9 * took[0]
    # (the issue's dab: a stroke less than a pixel long with the eraser is
    # one pass of it on the paper under it, which lifts a quarter of what
    # is there, and no light shows)
    before = burnished.copy()
    burnished.erase([(100, 75), (100.6, 75.4)], tools.ERASER, 1.0, 4.5)
    spot = (slice(74, 77), slice(99, 102))
    assert 0.7 < _left(burnished, before, spot) < 0.85
    assert _luminances(burnished, spot).mean() < 0.03


def _smeared(colour, layers, pressure, burnished):
    """A patch of coloured pencil, burnished or not, rubbed to and fro with
    a vinyl eraser in the hand from inside it out over the bare paper: how
    far from the paper's colour that is, at one, two, three to five and six
    to nine widths of the eraser out (CIE 1976), and the share of the
    pigment and of the wax that the sheet and the eraser hold between
    them."""
    sheet = Sheet(300, 150, paper=papers.CARTRIDGE, seed=1)
    for layer in range(layers):
        sheet.shade(
            (20, 20, 150, 130),
            pencils.coloured(colour),
            pressure,
            angle=30 - 55 * layer,
        )
    if burnished:
        for angle in (20, -50):
            sheet.shade((20, 20, 150, 130), pencils.BLENDER, 1.0, angle=angle)
    bare = Sheet(300, 150, paper=papers.CARTRIDGE, seed=1)
    total, wax = _total(sheet), float(sheet.held[..., IW].sum(dtype=np.float64))
    eraser = Smudger(tools.ERASER)
    sheet.rub((60, 45, 240, 105), eraser, 0.9)
    tints = []
    for x0, x1 in ((156, 166), (166, 176), (176, 196), (200, 236)):
        where = (slice(55, 95), slice(x0, x1))
        tints.append(float(np.linalg.norm(_lab(sheet, where) - _lab(bare, where))))
    kept = (_total(sheet) + eraser.load) / total
    kept_wax = (float(sheet.held[..., IW].sum(dtype=np.float64)) + eraser.wax) / wax
    return tints, (kept, kept_wax)


def test_a_vinyl_eraser_smears_a_dense_or_burnished_dark_along_its_way():
    # (fusain-d6p) wax that holds against an eraser is dragged along by it,
    # and laid on what is beside it and after it: a dirty halo a width or
    # so out of the dark, fading over several (there was none: the eraser
    # left the paper beside the dark as clean as it found it)
    for burnished, near in ((True, 12.0), (False, 9.0)):
        tints, kept = _smeared(DARK, 4, 0.8, burnished)
        assert tints[0] > near
        assert tints[0] > tints[1] > tints[2] > tints[3] > 0.5
        assert kept == pytest.approx((1.0, 1.0), abs=1e-5)
    # of a red, a tint; of a light layer, which the eraser takes, nothing
    tints, _ = _smeared(RED, 3, 0.7, True)
    assert 3.0 < tints[0] < 10.0
    tints, _ = _smeared(DARK, 1, 0.3, False)
    assert tints[0] < 0.5


def test_a_kneaded_eraser_barely_touches_coloured_pencil():
    light = _sheet()
    light.shade(BOX, pencils.coloured(DARK), 0.3)
    for sheet, least in ((light, 0.6), (_dense(False), 0.85), (_dense(True), 0.9)):
        before = sheet.copy()
        sheet.rub(AREA, tools.KNEADED, 0.9)
        assert least < _left(sheet, before, RUBBED) < 1.0


def test_wax_does_not_hold_what_has_no_wax():
    for medium in (pencils.B2, pencils.HB):
        sheet = _sheet()
        _layers(sheet, medium, 3, 0.8)
        before = sheet.copy()
        sheet.rub(AREA, tools.ERASER, 0.9)
        assert _left(sheet, before, RUBBED) < 0.2
    # and a crayon, which is waxier than a pencil, holds as a pencil does
    sheet = _sheet()
    _layers(sheet, crayons.wax(DARK), 2, 0.8, 9.0)
    before = sheet.copy()
    sheet.rub(AREA, tools.ERASER, 0.9)
    assert _left(sheet, before, RUBBED) > 0.5


def test_how_fast_wax_holds():
    held = np.zeros((1, 1, IW + 1), np.float32)
    cover, flat = np.ones((1, 1), np.float32), np.zeros((1, 1), np.float32)
    assert wax_grip(held, cover, flat, 0, 0, 0.0) == 1.0  # no wax
    held[0, 0, IW] = 0.5 * WAX_FAST
    assert wax_grip(held, cover, flat, 0, 0, 0.0) == pytest.approx(
        1.0 - 0.5 * WAX_HOLDS
    )
    held[0, 0, IW] = 3.0 * WAX_FAST
    assert wax_grip(held, cover, flat, 0, 0, 0.0) == pytest.approx(1.0 - WAX_HOLDS)
    flat[0, 0] = 1.0
    assert wax_grip(held, cover, flat, 0, 0, 0.0) == pytest.approx(1.0 - POLISH_HOLDS)
    # as far as a tool scrapes, the wax does not hold
    assert wax_grip(held, cover, flat, 0, 0, 1.0) == 1.0
    assert wax_grip(held, cover, flat, 0, 0, 0.5) == pytest.approx(
        1.0 - 0.5 * POLISH_HOLDS
    )
    cover[0, 0] = 0.0  # (bare paper)
    assert wax_grip(held, cover, flat, 0, 0, 0.0) == 1.0


# --------------------------------------------------------------------------
# the knife and the sand eraser


@pytest.mark.parametrize("paper", [papers.CARTRIDGE, SMOOTH])
def test_a_knife_scratches_a_fine_light_line_out_of_a_burnished_dark(paper):
    sheet = _dense(True, paper)
    before = sheet.copy()
    assert _luminances(sheet, RUBBED).mean() < 0.03
    took = sheet.erase([(60, 76), (140, 76)], tools.KNIFE, 0.9)
    line = (slice(75, 77), slice(70, 130))
    # on its line, back to near the paper
    assert _left(sheet, before, line) < 0.05
    assert _luminances(sheet, line).mean() > 0.65 * _paper(sheet)
    # a fine line: two pixels to either side nothing has moved
    for rows in (slice(60, 73), slice(80, 93)):
        beside = (rows, slice(70, 130))
        assert np.array_equal(sheet.pigment()[beside], before.pigment()[beside])
    # what it scraped off is counted
    assert took == pytest.approx(_total(before) - _total(sheet), rel=1e-5)
    # the wax went with the pigment, so the line can be drawn in again
    assert sheet.held[line][..., IW].sum() < 0.05 * before.held[line][..., IW].sum()
    laid = sheet.stroke([(60, 76), (140, 76)], pencils.coloured(RED), 0.7, 2)
    again = before.copy().stroke([(60, 76), (140, 76)], pencils.coloured(RED), 0.7, 2)
    assert laid > 3 * again


def test_a_knife_in_the_hand_keeps_what_it_scraped_off():
    sheet = _dense(True)
    before = _total(sheet)
    knife = Smudger(tools.KNIFE)
    for y in (70, 80):
        sheet.erase([(60, y), (140, y)], knife, 0.9)
    assert knife.load > 0.0
    assert knife.load == pytest.approx(before - _total(sheet), rel=1e-5)
    assert tools.KNIFE.scrape == 1.0 and tools.KNIFE.width == 2.0


def test_a_sand_eraser_takes_what_a_vinyl_one_slides_on():
    for burnished in (False, True):
        vinyl, sand = _dense(burnished), _dense(burnished)
        before = vinyl.copy()
        vinyl.rub(AREA, tools.ERASER, 0.9)
        took = sand.rub(AREA, tools.SAND_ERASER, 0.9, 10)
        assert _left(sand, before, RUBBED) < 0.15 < 0.5 < _left(vinyl, before, RUBBED)
        assert _luminances(sand, RUBBED).mean() > 0.3 * _paper(sand)
        assert took == pytest.approx(_total(before) - _total(sand), rel=1e-5)


def _over(sheet) -> np.ndarray:
    """A light layer of HB shaded over the sheet: what it laid at each pixel."""
    before = sheet.pigment().copy()
    sheet.shade(BOX, pencils.HB, 0.3, 6.0, angle=-20)
    return sheet.pigment() - before


def _touched(new: np.ndarray, where=RUBBED) -> float:
    """The share of the pixels of an area that a layer touched."""
    return float((new[where] > 0.02 * new[where].mean()).mean())


def _grain(new: np.ndarray, where=RUBBED) -> float:
    return float(new[where].std() / new[where].mean())


def _relief(sheet, where=RUBBED) -> float:
    return float(sheet.surface_height()[where].std())


@pytest.mark.parametrize(
    "tool, spacing", [(tools.KNIFE, 1.0), (tools.SAND_ERASER, None)]
)
def test_the_tooth_comes_back_where_burnished_wax_is_scraped_off(tool, spacing):
    # (fusain-6k1) the polish of wax is the wax's: it levels the tooth as
    # far as it fills it, and what is drawn where it has been scraped off
    # meets the tooth again, as on fresh paper. (It stayed level: a light
    # layer of HB touched 0.96 to 0.998 of the pixels there, where it
    # touches 0.68 of fresh paper, and lay three or four times as evenly)
    sheet, fresh = _dense(True), _sheet()
    sheet.rub(AREA, tool, 0.9, 10, spacing=spacing)
    assert sheet.press[RUBBED].mean() < 0.15  # (0.996)
    assert _gloss(sheet, RUBBED) < 0.025  # (none and 0.08)
    assert _relief(sheet) > 0.9 * _relief(fresh)  # (a twenty-eighth)
    new, bare = _over(sheet), _over(fresh)
    assert _touched(new) < _touched(bare) + 0.1
    assert _grain(new) > 0.85 * _grain(bare)


def test_where_burnished_wax_holds_against_an_eraser_it_stays_level():
    # a vinyl eraser leaves most of the wax, which keeps the tooth level
    sheet = _dense(True)
    pressed = sheet.press[RUBBED].mean()
    for _ in range(3):
        sheet.rub(AREA, tools.ERASER, 0.9)
    assert sheet.press[RUBBED].mean() > 0.95 * pressed
    assert _touched(_over(sheet)) > 0.9


def test_a_tooth_a_hard_lead_crushed_stays_crushed_unless_it_is_cut():
    # an eraser takes the graphite and leaves the tooth as the lead left it;
    # a sand eraser or a knife takes the surface of the paper with it
    sheet = _sheet()
    _layers(sheet, pencils.H4, 3, 1.0)
    crushed = sheet.press[RUBBED].mean()
    assert crushed > 0.5
    erased, sanded = sheet.copy(), sheet.copy()
    for _ in range(3):
        erased.rub(AREA, tools.ERASER, 0.9)
    assert erased.pigment()[RUBBED].mean() < 0.1 * sheet.pigment()[RUBBED].mean()
    assert erased.press[RUBBED].mean() == pytest.approx(crushed)
    sanded.rub(AREA, tools.SAND_ERASER, 0.9, 10)
    assert sanded.press[RUBBED].mean() < 0.2 * crushed


@pytest.mark.parametrize("scrape", [-0.1, 1.5, float("nan")])
def test_a_tool_scrapes_by_a_share(scrape):
    with pytest.raises(ValueError, match="scrape must be between 0 and 1"):
        tools.ERASER.with_(scrape=scrape)


# --------------------------------------------------------------------------
# a dark over a light


LEMON, TEAL = "#f2e24a", "#0f5a5e"


def _dark_over_light(paper, light_layers: int) -> tuple[float, float]:
    """Dark teal (two layers at 0.6) over layers of lemon (at 0.4): what the
    teal takes, as a share of what it takes on the bare paper, and how far
    the tone goes from the lemon's to that of the teal alone."""
    lemon, teal = pencils.coloured(LEMON), pencils.coloured(TEAL)
    alone, sheet = _sheet(paper), _sheet(paper)
    on_bare = sum(alone.shade(BOX, teal, 0.6, 4, angle=a) for a in (-40, 55))
    for angle in (30, -25, 80, 10)[:light_layers]:
        sheet.shade(BOX, lemon, 0.4, 4, angle=angle)
    light = _lab(sheet)[0]
    taken = sum(sheet.shade(BOX, teal, 0.6, 4, angle=a) for a in (-40, 55))
    way = (light - _lab(sheet)[0]) / (light - _lab(alone)[0])
    return taken / on_bare, float(way)


def test_a_dark_over_a_light_darkens_it_a_good_deal_while_the_tooth_has_room():
    # on cartridge paper four light layers leave room: the dark takes half
    # of what it takes on bare paper, and the tone goes most of the way
    taken, way = _dark_over_light(papers.CARTRIDGE, 4)
    assert 0.4 < taken < 0.65 and way > 0.6
    # on a shallow tooth one light layer leaves room too
    taken, way = _dark_over_light(SMOOTH, 1)
    assert taken > 0.45 and way > 0.55


def test_a_dark_takes_badly_over_a_shallow_tooth_that_light_layers_filled():
    # (what the issue saw on the paper of examples/macaw.py: there the darks
    # go first)
    shares = [_dark_over_light(SMOOTH, layers) for layers in (1, 2, 4)]
    assert shares[0][0] > shares[1][0] > shares[2][0]
    assert shares[0][1] > shares[1][1] > shares[2][1]
    taken, way = shares[2]
    assert 0.15 < taken < 0.3 and 0.25 < way < 0.45
    # the fourth light layer itself took a fifth of the first: the tooth
    # was full, as it is not on cartridge paper
    takes = {}
    for paper in (SMOOTH, papers.CARTRIDGE):
        sheet = _sheet(paper)
        lemon = pencils.coloured(LEMON)
        laid = [sheet.shade(BOX, lemon, 0.4, 4, angle=a) for a in (30, -25, 80, 10)]
        takes[paper.name] = laid[3] / laid[0]
    assert takes[SMOOTH.name] < 0.25 and takes[papers.CARTRIDGE.name] > 0.4


def test_what_the_volume_of_a_pixel_is_made_of():
    # (the blender's wax is pigment too, as far as the sheet counts it)
    sheet = _sheet()
    laid = sheet.shade(BOX, pencils.BLENDER, 1.0)
    np.testing.assert_allclose(sheet.held[..., IV].sum(), laid, rtol=1e-5)
    np.testing.assert_allclose(sheet.held[..., IW].sum(), laid, rtol=1e-3)
