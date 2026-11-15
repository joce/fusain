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


def test_a_white_pencil_pressed_hard_burnishes_too_but_less_than_the_blender():
    base = _sheet()
    _layers(base, pencils.coloured(RED), 3, 0.5)
    flecks = _flecks(base)
    white, blender = base.copy(), base.copy()
    white.shade(BOX, pencils.coloured("#f4f1ea"), 1.0, angle=20)
    blender.shade(BOX, pencils.BLENDER, 1.0, angle=20)
    # (the white pencil lays its own wax and colour in the flecks it
    # reaches; the blender melts into what is there, and stirs it)
    assert _flecks(white) < 0.5 * flecks and _gloss(white) > 0.1
    assert _flecks(blender) < 0.6 * _flecks(white)
    assert _gloss(blender) > 2.0 * _gloss(white)
    assert blender.press[INSIDE].mean() > 1.5 * white.press[INSIDE].mean()
    assert pencils.BLENDER.melts == 1.0 and pencils.coloured(RED).melts == 0.0
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
        assert _luminances(sheet, RUBBED).mean() > 0.55 * _paper(sheet)
        sheet.rub(AREA, tools.ERASER, 0.9)
        assert _left(sheet, before, RUBBED) < 0.03


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
    # (the issue's dab: a stroke less than a pixel long with the eraser
    # lifts a few hundredths of what is under it, and no light shows)
    before = burnished.copy()
    burnished.erase([(100, 75), (100.6, 75.4)], tools.ERASER, 1.0, 4.5)
    spot = (slice(74, 77), slice(99, 102))
    assert 0.9 < _left(burnished, before, spot) < 1.0
    assert _luminances(burnished, spot).mean() < 0.03


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
