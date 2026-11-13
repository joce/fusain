"""Colour: layers that mix or cover, and wax (behaviours 4 and 5 of
docs/design.md), measured."""

import itertools

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, papers, pastels, pencils, tools
from fusain._engine import IV, IW

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
RED, YELLOW, BLUE = "#c0392b", "#f2c40f", "#2456a6"
WHITE, BLACK = "#f4f1ea", "#161616"


def _sheet(paper=papers.CARTRIDGE) -> Sheet:
    return Sheet(200, 150, paper=paper, seed=1)


def _layers(sheet, medium, count=2, pressure=0.7, width=6.0) -> float:
    return sum(
        sheet.shade(BOX, medium, pressure, width, angle=30 - 55 * layer)
        for layer in range(count)
    )


def _colour(sheet, where=INSIDE) -> np.ndarray:
    """The colour of the area as a whole: sRGB of its mean reflectance."""
    spectrum = sheet.reflectance()[where].astype(np.float64).mean(axis=(0, 1))
    return sp.reflectance_to_srgb(spectrum)


def _luminance(sheet, where=INSIDE) -> float:
    return float(sp.srgb_to_linear(_colour(sheet, where)) @ sp.LUMINANCE)


def _grain(sheet, where=INSIDE) -> float:
    spectrum = sheet.reflectance()[where].astype(np.float64)
    return float((sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE).std())


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


# --------------------------------------------------------------------------
# coloured pencil: layers mix to the eye


def test_a_coloured_pencil_is_its_colour_paler_when_light():
    for colour in (RED, YELLOW, BLUE):
        want = sp.parse_color(colour)
        light, firm = _sheet(), _sheet()
        light.shade(BOX, pencils.coloured(colour), 0.3)
        _layers(firm, pencils.coloured(colour), 4)
        # the same hue: its strongest and weakest components are the colour's
        for sheet in (light, firm):
            got = _colour(sheet)
            assert np.argmax(got) == np.argmax(want), colour
            assert np.argmin(got) == np.argmin(want), colour
        assert _luminance(light) > _luminance(firm) + 0.1
        # four firm layers come close to the colour asked for (a little paper
        # still shows through)
        assert np.abs(_colour(firm) - want).max() < 0.25


def test_yellow_over_blue_is_green_and_so_is_blue_over_yellow():
    for first, second in ((BLUE, YELLOW), (YELLOW, BLUE)):
        sheet = _sheet()
        _layers(sheet, pencils.coloured(first))
        _layers(sheet, pencils.coloured(second))
        r, g, b = _colour(sheet)
        assert g > r + 0.1 and g > b + 0.1, (first, second)


def test_red_over_yellow_is_orange():
    sheet = _sheet()
    _layers(sheet, pencils.coloured(YELLOW))
    yellow = _colour(sheet)
    _layers(sheet, pencils.coloured(RED))
    r, g, b = _colour(sheet)
    assert r > g + 0.15 > b + 0.3
    assert g < yellow[1] - 0.1  # redder than the yellow it went over


def test_a_coloured_pencil_does_not_hide_what_is_under_it():
    dark = _sheet()
    _layers(dark, pencils.coloured(BLUE), 3)
    under = _luminance(dark)
    _layers(dark, pencils.coloured(YELLOW))
    alone = _sheet()
    _layers(alone, pencils.coloured(YELLOW))
    # yellow over dark blue stays dark: nowhere near yellow on white
    assert _luminance(dark) < under + 0.1 < _luminance(alone) - 0.3


# --------------------------------------------------------------------------
# pastel: the last layer covers


def _over(first, second, paper=papers.INGRES) -> Sheet:
    sheet = _sheet(paper)
    sheet.shade(BOX, pastels.soft(first), 0.7, 14.0)
    sheet.shade(BOX, pastels.soft(second), 0.7, 14.0, angle=-30)
    return sheet


def test_a_light_pastel_covers_a_dark_one_and_a_dark_one_a_light_one():
    black, white = _sheet(papers.INGRES), _sheet(papers.INGRES)
    black.shade(BOX, pastels.soft(BLACK), 0.7, 14.0)
    white.shade(BOX, pastels.soft(WHITE), 0.7, 14.0)
    assert _luminance(black) < 0.05 and _luminance(white) > 0.8
    light, dark = _luminance(_over(BLACK, WHITE)), _luminance(_over(WHITE, BLACK))
    assert light > 0.5 and dark < 0.08
    # the order is what makes the difference: the same two sticks
    assert light > 8 * dark


def test_a_pass_lays_its_dust_on_top_of_the_dust_that_was_there():
    sheet = _over(BLACK, WHITE)
    black, white = pastels.soft(BLACK).pigment, pastels.soft(WHITE).pigment

    def darkness(layer) -> float:
        """Absorption per unit volume, between the white's (0) and the black's (1)."""
        volume = layer[INSIDE][..., IV].sum(dtype=np.float64)
        k = layer[INSIDE][..., :10].sum(axis=(0, 1), dtype=np.float64) / volume
        return float(((k - white.K[:10]) / (black.K[:10] - white.K[:10])).mean())

    assert darkness(sheet.loose) < 0.15  # the white, with a little of the black
    # the black's dust has gone under (with the white of the first strokes,
    # which the next ones went over)
    assert darkness(sheet.under) > 0.6
    assert sheet.under[INSIDE][..., IV].sum() > 0.3 * sheet.loose[INSIDE][..., IV].sum()


def test_rubbing_blends_what_is_on_top_with_what_is_under():
    sheet = _over(BLACK, WHITE)
    before = _luminance(sheet)
    for k in range(8):
        sheet.smudge([(30, 55 + 6 * k), (170, 55 + 6 * k)], tools.FINGER, 0.7)
    after = _luminance(sheet)
    assert 0.03 < after < 0.5 * before  # a dark grey: neither stick's colour


def test_a_finger_blends_two_pastels_where_they_meet():
    sheet = _sheet(papers.INGRES)
    orange, blue = pastels.soft("#e67e22"), pastels.soft("#2e6fb7")
    sheet.shade((20, 20, 100, 130), orange, 0.7, 14.0)
    sheet.shade((100, 20, 180, 130), blue, 0.7, 14.0)
    left, right = (slice(50, 100), slice(60, 90)), (slice(50, 100), slice(110, 140))
    join = (slice(50, 100), slice(85, 115))
    apart = _colour(sheet, left), _colour(sheet, right)
    for k in range(10):
        ends = [(30, 52 + 5 * k), (170, 52 + 5 * k)]
        sheet.smudge(ends if k % 2 else ends[::-1], tools.FINGER, 0.7)
    r, _, b = _colour(sheet, join)
    # at the join, neither the orange (red far above blue) nor the blue
    assert abs(r - b) < 0.5 * min(apart[0][0] - apart[0][2], apart[1][2] - apart[1][0])
    # and the two sides have come closer to each other
    before = np.abs(apart[0] - apart[1]).max()
    assert np.abs(_colour(sheet, left) - _colour(sheet, right)).max() < 0.8 * before


def test_soft_pastel_lays_more_and_dustier_than_hard():
    def laid(medium) -> tuple[float, float]:
        sheet = _sheet(papers.INGRES)
        volume = sheet.shade(BOX, medium, 0.6, 10.0)
        dust = sheet.loose[..., IV].sum() + sheet.under[..., IV].sum()
        return volume, float(dust / sheet.pigment().sum())

    soft, hard = laid(pastels.soft(RED)), laid(pastels.hard(RED))
    assert soft[0] > 1.5 * hard[0] and soft[1] > hard[1] + 0.1
    assert pastels.soft(RED).name == "#c0392b pastel"
    assert pastels.hard(RED, name="sanguine").name == "sanguine"


def test_sanded_card_takes_pastel_all_over_and_holds_more_of_it():
    def filled(paper) -> tuple[float, float]:
        sheet = _sheet(paper)
        for layer in range(14):
            sheet.shade(BOX, pastels.soft(RED), 0.7, 14.0, angle=30 + 50 * layer)
        return float(sheet.cover[INSIDE].mean()), float(sheet.pigment()[INSIDE].mean())

    sanded, cartridge = filled(papers.SANDED), filled(papers.CARTRIDGE)
    assert sanded[0] > 0.9 and sanded[1] > 1.15 * cartridge[1]


# --------------------------------------------------------------------------
# wax


def test_only_a_waxy_medium_leaves_wax():
    for medium, waxy in (
        (pencils.coloured(RED), True),
        (pencils.B2, False),
        (pastels.soft(RED), False),
    ):
        sheet = _sheet()
        laid = _layers(sheet, medium)
        wax = float(sheet.held[..., IW].sum(dtype=np.float64))
        assert (wax > 0) is waxy, medium.name
        if waxy:
            np.testing.assert_allclose(wax, laid * medium.wax, rtol=1e-3)


def test_a_waxed_tooth_takes_badly():
    yellow = pencils.coloured(YELLOW)
    bare = _sheet().shade(BOX, yellow, 0.5)
    # over clear wax, laid lightly so that nothing is polished
    waxed = _sheet()
    _layers(waxed, pencils.BLENDER, 3, pressure=0.5)
    assert waxed.press.mean() < 0.05
    assert waxed.shade(BOX, yellow, 0.5) < 0.5 * bare
    # the more layers of a waxy pencil there are, the less the next one takes
    takes = []
    for count in (0, 2, 6):
        sheet = _sheet()
        _layers(sheet, pencils.coloured(BLUE), count)
        takes.append(sheet.shade(BOX, yellow, 0.5))
    assert takes[0] > 2 * takes[1] > 4 * takes[2]
    # and a waxy lead fills the tooth with less than the same lead without
    # its wax would
    lead = pencils.coloured(BLUE)
    waxy, dry = _sheet(), _sheet()
    assert _layers(waxy, lead, 5) < 0.95 * _layers(dry, lead.with_(wax=0.0), 5)


def test_a_firm_pass_polishes_wax_and_nothing_else():
    waxed = _sheet()
    _layers(waxed, pencils.coloured(RED), 3, pressure=0.5)
    assert waxed.press[INSIDE].mean() < 0.02  # a light touch does not polish
    waxed.shade(BOX, pencils.coloured(RED), 1.0)
    assert waxed.press[INSIDE].mean() > 0.3
    for medium in (pencils.B6, pastels.soft(RED)):
        sheet = _sheet()
        _layers(sheet, medium, 3, pressure=1.0)
        assert not sheet.press.any(), medium.name


def test_burnishing_closes_the_white_specks():
    sheet = _sheet()
    _layers(sheet, pencils.coloured(RED), 3, pressure=0.6)
    before, grain = _total(sheet), _grain(sheet)
    cover, tone = float(sheet.cover[INSIDE].mean()), _luminance(sheet)
    bare = float((sheet.pigment()[INSIDE] < 1e-3).mean())
    assert cover < 0.9 and bare > 0.01
    lost = sum(sheet.shade(BOX, pencils.BLENDER, 1.0, angle=a) for a in (20, -50))
    assert sheet.cover[INSIDE].mean() > 0.95
    assert (sheet.pigment()[INSIDE] < 1e-3).mean() < 0.2 * bare
    assert _grain(sheet) < 0.7 * grain
    assert _luminance(sheet) < tone  # richer: the paper no longer shows
    # the colour is the red's still: the blender has next to none of its own
    assert np.argmax(_colour(sheet)) == 0
    # what the blender lost is on the sheet, less what it dragged off it
    np.testing.assert_allclose(_total(sheet), before + lost, rtol=1e-5)


def test_a_polished_tooth_is_level_under_the_lamp_too():
    sheet = _sheet(papers.TONED_GREY)
    rough = sheet.surface_height()[INSIDE].std()
    _layers(sheet, pencils.BLENDER, 3, pressure=1.0)
    assert sheet.press[INSIDE].mean() > 0.9
    # nearly all of the patch is level with the peaks (a few pits the
    # blender never reached are left)
    height = sheet.surface_height()[INSIDE]
    assert (height > 0.9 * sheet.surface_height().max()).mean() > 0.85
    assert height.std() < 0.7 * rough
    # outside the patch the paper keeps its relief
    assert sheet.surface_height()[:15].std() > 0.9 * rough
    np.testing.assert_allclose(
        sheet.surface_height(2.0), 2 * sheet.surface_height(), rtol=1e-6
    )


def test_a_blender_lays_wax_and_next_to_no_colour():
    sheet = _sheet()
    _layers(sheet, pencils.BLENDER, 2, pressure=0.8)
    assert _luminance(sheet) > 0.95 * _luminance(_sheet())
    assert sheet.held[..., IW].sum() > 0
    assert pencils.BLENDER.wax == 1.0 and pencils.BLENDER.binding == 1.0


def test_coloured_pencils_and_media_by_name():
    red = pencils.coloured(RED)
    assert red.name == "#c0392b pencil" and red.wax > 0 and red.binding > 0.9
    assert pencils.coloured(RED, name="scarlet").name == "scarlet"
    assert pencils.coloured((0.2, 0.4, 0.6)).pigment.S > 0
    with pytest.raises(ValueError, match="wax must be"):
        red.with_(wax=1.5)
    with pytest.raises(ValueError, match="#rrggbb"):
        pencils.coloured("scarlet")


# --------------------------------------------------------------------------
# what a firm tip drags along


def _short_firm_strokes(sheet, medium, count=300, seed=5) -> list[float]:
    """Short strokes pressed hard, all over the patch, every way."""
    rng = np.random.default_rng(seed)
    laid = []
    for _ in range(count):
        x, y = rng.uniform(45, 155), rng.uniform(45, 105)
        turn = rng.uniform(0.0, np.pi)
        dx, dy = 8 * np.cos(turn), 8 * np.sin(turn)
        laid.append(
            sheet.stroke([(x - dx, y - dy), (x, y), (x + dx, y + dy)], medium, 0.95, 4)
        )
    return laid


def test_a_firm_pencil_does_not_wear_a_waxed_layer_away():
    sheet = _sheet()
    red = pencils.coloured(RED)
    _layers(sheet, red, 3, 0.6)
    before, tone = _total(sheet), _luminance(sheet)
    laid = _short_firm_strokes(sheet, pencils.BLENDER)
    # the blender drags the layer about, but where it lifts off it leaves
    # what it dragged: it never comes away with any of it
    assert min(laid) >= 0.0
    assert _total(sheet) == pytest.approx(before + sum(laid), rel=1e-6)
    assert _total(sheet) >= before
    assert _luminance(sheet) < 1.02 * tone


def test_what_a_tip_drags_off_the_sheet_is_counted():
    sheet = _sheet()
    red = pencils.coloured(RED)
    sheet.shade((0, 20, 200, 130), red, 0.6)
    sheet.shade((0, 20, 200, 130), red, 0.6, angle=-25)
    before = _total(sheet)
    laid = [
        sheet.stroke([(120, y), (240, y)], pencils.BLENDER, 1.0, 5)
        for y in range(30, 120, 6)
    ]
    # (what it dragged past the edge is gone with it, and is in the count)
    assert _total(sheet) == pytest.approx(before + sum(laid), rel=1e-6)


# --------------------------------------------------------------------------
# building up light layers, as a coloured pencil artist does


def test_light_layers_go_on_building_for_ten_or_so():
    sheet = _sheet()
    deep = pencils.coloured("#2c7a33")
    takes = [sheet.shade(BOX, deep, 0.3, angle=30 + 50 * k) for k in range(12)]
    shares = [take / takes[0] for take in takes]
    # each takes less than the one before, but slowly: the second most of
    # what the first did, the sixth still a third, the tenth about a fifth
    assert shares[1] > 0.7
    assert shares[5] > 0.3
    assert 0.12 < shares[9] < 0.3
    assert all(a > 0.85 * b for a, b in itertools.pairwise(shares))


def test_a_dark_pencil_shows_over_a_light_wash_unless_the_wash_was_pressed_hard():
    lime, deep = pencils.coloured("#b9d233"), pencils.coloured("#2c7a33")
    bare = _sheet()
    on_bare = bare.shade(BOX, deep, 0.6)
    alone = _luminance(bare)
    taken, tones = [], []
    for pressure in (0.3, 0.5, 0.75):
        sheet = _sheet()
        _layers(sheet, lime, 3, pressure)
        taken.append(sheet.shade(BOX, deep, 0.6, angle=-20) / on_bare)
        tones.append(_luminance(sheet))
    # over a light wash it takes most of what it takes on bare paper, and
    # the tone comes close to that of the dark pencil alone
    assert taken[0] > 0.55
    assert tones[0] < 1.2 * alone
    # the harder the wash was pressed, the less the dark pencil takes
    assert taken[0] > taken[1] > taken[2]
    assert taken[2] < 0.35
