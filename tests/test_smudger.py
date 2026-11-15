"""A tool in the hand: a stump or a finger that keeps what it carries from
pass to pass (behaviours 3 and 7 of docs/design.md), measured."""

from typing import Any

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, Smudger, charcoal, crayons, pastels, pencils, tools
from fusain._engine import IC, IP
from fusain.sheet import IS, IV, IW, N

TONE = (20, 20, 380, 180)  # the rectangle shaded
INNER = (120, 50, 300, 150)  # the part of it that is rubbed
ROWS = slice(62, 138)  # rows well inside the part rubbed


def _toned(medium=charcoal.VINE, width=9.0, area=TONE) -> tuple[Sheet, float]:
    sheet = Sheet(400, 200, seed=1)
    laid = sum(sheet.shade(area, medium, 0.6, width, angle=a) for a in (30, -40))
    return sheet, laid


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


def _luminance(sheet) -> np.ndarray:
    spectrum = sheet.reflectance().astype(np.float64)
    paper = sp.reflectance_to_linear_rgb(sheet.ground) @ sp.LUMINANCE
    return (sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE) / paper


def _loaded(medium=pencils.B6, tool=tools.STUMP) -> tuple[Sheet, Smudger, float]:
    """A sheet with a patch of ``medium`` on its left, a tool that has been
    rubbed all over the patch, and the volume the patch was laid with."""
    sheet = Sheet(500, 200, seed=1)
    patch = (20, 20, 120, 180)
    laid = sum(sheet.shade(patch, medium, 0.8, 7.0, angle=a) for a in (30, -40))
    held = Smudger(tool)
    sheet.rub((25, 25, 115, 175), held, 0.7)
    return sheet, held, laid


# --------------------------------------------------------------------------
# a loaded tool gives back as much as it takes


def test_rubbing_a_dark_tone_again_and_again_does_not_wear_it_away():
    # (the reproduction of fusain-6lr: each pass used to start clean, and
    # eight rubs took four tenths of the graphite with them)
    sheet = Sheet(300, 120, seed=1)
    area, inside = (10, 10, 290, 110), (slice(30, 90), slice(40, 260))
    laid = sheet.shade(area, pencils.B8, 0.9)
    stump = Smudger(tools.STUMP)
    sheet.rub(area, stump, 0.9)
    once = float(_luminance(sheet)[inside].mean())
    for _ in range(7):
        sheet.rub(area, stump, 0.9)
    assert _total(sheet) > 0.99 * laid
    assert _total(sheet) + stump.load == pytest.approx(laid, rel=1e-5)
    assert abs(float(_luminance(sheet)[inside].mean()) - once) < 0.01
    # a bare tool starts each rub clean, and still takes little: it carries
    # its load from one pass to the next of a rub, and each rub takes one
    # (a stump holds half as much again as it used to carry off such a tone)
    bare = Sheet(300, 120, seed=1)
    bare.shade(area, pencils.B8, 0.9)
    taken = sum(bare.rub(area, tools.STUMP, 0.9) for _ in range(8))
    assert 0.0 < taken < 0.06 * laid
    full = tools.STUMP.holds * np.pi * (0.5 * tools.STUMP.width + 0.5) ** 2
    # (a full load and what is on its face: 1.14 fulls a rub, where it was
    # 1.01 when its passes strayed past the tone and spent some on the bare
    # paper there)
    assert taken < 8 * 1.2 * full
    assert _total(bare) + taken == pytest.approx(laid, rel=1e-5)


def test_many_drags_of_a_finger_in_the_hand_leave_the_tone_as_dark():
    def dragged(finger) -> tuple[float, float]:
        sheet, _ = _toned(pencils.B8, 7.0, (20, 40, 380, 160))
        inside = (slice(70, 130), slice(60, 340))
        before = float(_luminance(sheet)[inside].mean())
        rng = np.random.default_rng(3)
        for _ in range(60):
            x, y = rng.uniform(40, 360), rng.uniform(50, 90)
            sheet.smudge([(x, y), (x, y + 60)], finger, 0.5)
        return before, float(_luminance(sheet)[inside].mean())

    # (smoothed, the tone is a little darker: its white specks have closed)
    before, kept = dragged(Smudger(tools.FINGER, 12))
    assert before - 0.035 < kept < before
    # a finger wiped clean before every drag leaves it lighter than that (it
    # used to lighten it steadily: it now takes far less into its load)
    before, wiped = dragged(tools.FINGER.with_(width=12.0))
    assert kept + 0.005 < wiped < before


@pytest.mark.parametrize(
    ("medium", "width", "tool"),
    [
        (charcoal.VINE, 9.0, tools.STUMP),
        (charcoal.VINE, 9.0, tools.FINGER),
        (pencils.B8, 7.0, tools.STUMP),
    ],
    ids=["charcoal and stump", "charcoal and finger", "8B and stump"],
)
def test_a_pass_leaves_no_pale_dot_where_it_lands(medium, width, tool):
    # the passes of a rub land by turns at either end of the area; each used
    # to lift a pale dot there, a fifth of the pigment under it
    sheet, laid = _toned(medium, width)
    before = sheet.pigment().astype(np.float64)
    taken = sheet.rub(INNER, tool, 0.7, hand=0.0)
    after = sheet.pigment().astype(np.float64)
    reach = int(tool.width)
    middle = slice(180, 240)
    kept = after[ROWS, middle].sum() / before[ROWS, middle].sum()
    for band in (slice(122, 122 + reach), slice(298 - reach, 298)):
        landed = after[ROWS, band].sum() / before[ROWS, band].sum()
        assert landed > 0.95 and abs(landed - kept) < 0.04
    assert 0.0 < taken < 0.03 * laid
    # and the rows of the passes do not show there as they did: the tone is
    # no more uneven from row to row where they land than in the middle
    tone = _luminance(sheet)
    rows = [float(tone[ROWS, x : x + reach].mean(axis=1).std()) for x in (122, 200)]
    assert rows[0] < 1.6 * rows[1]


# --------------------------------------------------------------------------
# how far it carries what it moves

ORANGE, BLUE = "#e67e22", "#2e6fb7"
JOIN, FINGER = 200, 22  # where the two colours meet, and the width of a finger


def _of_each(first, second, absorption: np.ndarray) -> np.ndarray:
    """How much there is of each of two media in what absorbs light so
    (K per band, in rows): the mix of the two that comes nearest to it."""
    both = np.stack([first.pigment.K, second.pigment.K], axis=1)
    return np.linalg.solve(both.T @ both, both.T @ absorption)


def _rubbed_join(make, pressure: float, passes: int) -> np.ndarray:
    """The share of the blue in what lies on top, column by column, once a
    finger has been rubbed to and fro across the join of an orange and a
    blue pastel, in passes five of its widths long."""
    sheet = Sheet(400, 200, seed=1)
    orange, blue = make(ORANGE), make(BLUE)
    for angle in (25, -40):
        sheet.shade((30, 30, JOIN + 5, 170), orange, 0.7, 14, angle)
        sheet.shade((JOIN - 5, 30, 370, 170), blue, 0.7, 14, angle)
    finger = Smudger(tools.FINGER)
    for k in range(passes):
        y = 46 + 108 * k / (passes - 1)
        ends = [(JOIN - 55, y), (JOIN + 55, y + 2)]
        sheet.smudge(ends if k % 2 else ends[::-1], finger, pressure)
    top = sheet.loose[70:130, :, :N].astype(np.float64).sum(axis=0)
    orange_there, blue_there = _of_each(orange, blue, top.T)
    share = blue_there / np.maximum(orange_there + blue_there, 1e-9)
    return np.convolve(share, np.ones(5) / 5, mode="same")


def _passage(blue: np.ndarray) -> float:
    """How wide the passage from one colour to the other is, in widths of
    the finger: from a quarter of the blue to three quarters."""
    return float(np.argmax(blue >= 0.75) - np.argmax(blue >= 0.25)) / FINGER


def test_a_finger_rubbed_across_a_join_leaves_a_passage_between_the_colours():
    # (the reproduction of fusain-9zh: the whole width rubbed came out one
    # even mix, 0.42 to 0.57 of blue from one width of the finger left of
    # the join to one width right of it, and 0.38 to 0.61 at two widths)
    blue = _rubbed_join(pastels.soft, 0.6, 18)
    assert blue[JOIN - FINGER] < 0.25 and blue[JOIN + FINGER] > 0.75
    assert blue[JOIN - 2 * FINGER] < 0.13 and blue[JOIN + 2 * FINGER] > 0.87
    assert 0.4 < blue[JOIN] < 0.6
    # a gradient a width or two wide (it was five and a half: all that was
    # rubbed), that rises all the way across
    assert 0.8 < _passage(blue) < 2.0
    across = blue[JOIN - 3 * FINGER : JOIN + 3 * FINGER : FINGER // 2]
    assert (np.diff(across) > 0).all()


def test_a_join_rubbed_for_long_is_still_a_passage():
    # four times the rubbing: wider, and still each colour on its own side
    blue = _rubbed_join(pastels.soft, 0.6, 72)
    assert blue[JOIN - 2 * FINGER] < 0.22 and blue[JOIN + 2 * FINGER] > 0.8
    assert 1.2 < _passage(blue) < 3.0


def test_a_finger_blends_a_join_of_oil_pastels_the_same_way():
    # paste smears as dust does (it was 0.47 to 0.55 at one width)
    blue = _rubbed_join(pastels.oil, 0.8, 36)
    assert blue[JOIN - FINGER] < 0.36 and blue[JOIN + FINGER] > 0.68
    assert blue[JOIN - 2 * FINGER] < 0.25 and blue[JOIN + 2 * FINGER] > 0.78
    assert _passage(blue) < 3.0


def test_a_loaded_finger_brings_little_onto_a_tone_of_another_colour():
    # a finger loaded on orange and rubbed over blue: paper that carries
    # pigment takes nothing off its load, so the blue stays blue
    sheet = Sheet(400, 200, seed=1)
    orange, blue = pastels.soft(ORANGE), pastels.soft(BLUE)
    for angle in (25, -40):
        sheet.shade((20, 20, 180, 180), orange, 0.7, 14, angle)
        sheet.shade((220, 20, 380, 180), blue, 0.7, 14, angle)
    finger = Smudger(tools.FINGER)
    sheet.rub((30, 30, 170, 170), finger, 0.6)
    assert finger.load > 0
    sheet.rub((230, 30, 370, 170), finger, 0.6)
    top = sheet.loose[60:140, 260:340, :N].astype(np.float64).sum(axis=(0, 1))
    orange_there, blue_there = _of_each(orange, blue, top)
    assert orange_there < 0.03 * blue_there


def test_a_stump_fills_slowly_and_takes_no_more_once_it_is_full():
    # a stump holds so much (``Tool.holds`` for each pixel of its face); it
    # takes less the fuller it is, and a full one takes nothing more
    full = tools.STUMP.holds * np.pi * (0.5 * tools.STUMP.width + 0.5) ** 2
    sheet, laid = _toned(charcoal.VINE, 9.0)
    stump = Smudger(tools.STUMP)
    # one pass through 32 widths of charcoal: a fifth of what it holds
    assert 0.1 * full < sheet.smudge([(40, 100), (360, 100)], stump, 0.7) < 0.4 * full
    # (rubbed well inside a tone: on the bare paper round it, it would leave
    # what it carries)
    lifted = []  # (what it has lifted, without the dust on its face)
    for _ in range(3):
        sheet.rub(INNER, stump, 0.7)
        lifted.append(float(stump._carried[0, IV]))
    assert 0.4 * full < lifted[0] < lifted[1] < lifted[2] < full
    assert lifted[1] - lifted[0] < 0.75 * lifted[0] and lifted[2] > 0.9 * full
    assert _total(sheet) + stump.load == pytest.approx(laid, rel=1e-5)
    # a heavy pastel fills it in one rub, and the next takes nothing
    sheet, laid = _toned(pastels.soft(BLUE), 9.0)
    stump = Smudger(tools.STUMP)
    sheet.rub(INNER, stump, 0.7)
    assert 0.9 * full < float(stump._carried[0, IV]) <= full * (1 + 1e-9)
    assert abs(sheet.rub(INNER, stump, 0.7)) < 0.05 * full
    # an eraser, a chamois or a tool of one's own holds whatever it takes
    eraser = Smudger(tools.KNEADED)
    assert sheet.rub(TONE, eraser, 0.9) > 20 * full


@pytest.mark.parametrize(
    "medium", [charcoal.VINE, pastels.oil(BLUE), pencils.coloured(BLUE)]
)
def test_a_tool_never_takes_more_than_it_has_room_for(medium):
    # a tool that holds less than it would pick up off one pixel takes what
    # it has room for there, and its load is never more than it holds (it
    # took a pixel's pickup whole, past what it held, and laid the excess
    # back later): pigment, paste and wax alike, and none of it lost
    for tool in (tools.STUMP, tools.ERASER):
        tiny = tool.with_(holds=1e-6)
        full = tiny.holds * np.pi * (0.5 * tiny.width + 0.5) ** 2
        sheet, _ = _toned(medium)
        total, wax = _total(sheet), _wax(sheet)
        held = Smudger(tiny)
        for y in (60, 100, 140):
            sheet.smudge([(40, y), (360, y)], held, 0.9)
            assert 0.0 < float(held._carried[0, IV]) <= full * (1 + 1e-6)
        assert _total(sheet) + held.load == pytest.approx(total, rel=1e-6)
        assert _wax(sheet) + held.wax == pytest.approx(wax, rel=1e-6, abs=1e-9)
        # paste and wax in proportion to the pigment: never more than all
        load = held._carried[0]
        assert load[IP] <= load[IV] * (1 + 1e-6) and load[IC] <= load[IV]


def test_bare_tooth_takes_the_dust_off_a_loaded_stump_and_covered_paper_does_not():
    # a stump loaded with charcoal, dragged over bare paper, on across a
    # patch of pastel and over bare paper again: it tones the paper on
    # either side, and leaves the pastel as it was
    _, stump, _ = _loaded(charcoal.VINE)
    sheet = Sheet(400, 200, seed=2)
    blue = pastels.soft(BLUE)
    for angle in (25, -40):
        sheet.shade((160, 20, 260, 180), blue, 0.7, 14, angle)
    drawn = sheet.copy()
    sheet.smudge([(40, 100), (380, 100)], stump, 0.6)
    brought = (sheet.pigment() - drawn.pigment()).astype(np.float64)
    way = slice(96, 104)
    before, after = brought[way, 90:140].mean(), brought[way, 290:340].mean()
    assert before > 0.03 and after > 0.4 * before
    top = sheet.loose[way, 180:240, :N].astype(np.float64).sum(axis=(0, 1))
    charcoal_there, blue_there = _of_each(charcoal.VINE, blue, top)
    assert charcoal_there < 0.03 * blue_there
    tone = _luminance(sheet)
    assert tone[way, 90:140].mean() < 0.85 and tone[way, 290:340].mean() < 0.93


# --------------------------------------------------------------------------
# on wax


def test_a_loaded_stump_leaves_a_crayon_line_clean():
    # a stump loaded with charcoal and dragged across one firm line of
    # white crayon on bare paper (fusain-ma9: the line used to take 0.97 of
    # what the paper either side of it took, and came out a tenth darker)
    _, stump, _ = _loaded(charcoal.VINE)
    sheet = Sheet(300, 200, seed=2)
    sheet.stroke([(40, 110), (260, 110)], crayons.wax("#f4f1ea"), 0.9, 7.0)
    drawn = sheet.copy()
    total = _total(sheet) + stump.load
    sheet.smudge([(150, 40), (150, 180)], stump, 0.7)
    brought = (sheet.pigment() - drawn.pigment()).astype(np.float64)
    way = slice(146, 154)
    before, after = brought[96:102, way].mean(), brought[118:124, way].mean()
    on_the_line = brought[108:112, way].mean()
    assert before > 0.02 and abs(on_the_line) < 0.2 * before
    # what the wax refuses stays on the stump, which carries it on
    assert after > 0.75 * before
    assert _total(sheet) + stump.load == pytest.approx(total, rel=1e-6)
    # the line keeps its tone where the stump crossed it; the paper greys
    tone = _luminance(sheet)
    crossed, beside = tone[108:112, way].mean(), tone[108:112, 110:130].mean()
    assert beside - 0.05 < crossed <= beside + 0.005
    assert tone[96:102, way].mean() < beside - 0.07


@pytest.mark.parametrize(
    ("medium", "least", "most"),
    [
        (charcoal.VINE, 0.0, 0.15),
        (pastels.soft("#2456a6"), 0.0, 0.15),
        (pencils.B8, 0.0, 0.15),
        (pastels.oil("#2456a6"), 0.35, 0.9),
    ],
    ids=["charcoal", "soft pastel", "8B", "oil pastel"],
)
def test_wax_refuses_the_dust_a_finger_brings_and_takes_its_paste(medium, least, most):
    # a finger loaded on a patch of the medium and rubbed across a sheet:
    # what it leaves on three firm layers of crayon, against what it leaves
    # at the same place on a sheet that is bare. Dust: a tenth (what the
    # bare flecks between the crayon take); paste sticks, and half of what
    # a finger carries of an oil pastel is paste
    left = []
    for waxed in (False, True):
        scrap = Sheet(500, 200, seed=1)
        for angle in (30, -40):
            scrap.shade((20, 20, 160, 180), medium, 0.8, 9.0, angle=angle)
        finger = Smudger(tools.FINGER)
        scrap.rub((25, 25, 155, 175), finger, 0.7)
        sheet = Sheet(400, 200, seed=2)
        for layer in range(3 if waxed else 0):
            sheet.shade(
                (120, 20, 380, 180), crayons.wax("#f4f1ea"), 0.8, 9.0, 30 + 50 * layer
            )
        drawn = sheet.copy()
        total = _total(sheet) + finger.load
        for y in (60, 100, 140):
            sheet.smudge([(40, y), (360, y)], finger, 0.7)
        brought = (sheet.pigment() - drawn.pigment()).astype(np.float64)
        left.append(float(brought[50:150, 160:340].mean()))
        assert _total(sheet) + finger.load == pytest.approx(total, rel=1e-6)
    on_paper, on_wax = left
    assert on_paper > 0.0008
    assert least * on_paper <= on_wax < most * on_paper


@pytest.mark.parametrize(("layers", "least"), [(1, 0.55), (3, 0.55)])
def test_a_finger_slides_on_a_film_of_wax_laid_over_dust(layers, least):
    # crayon shaded over charcoal lies on it as a film, which sets most of
    # the dust under it; of the rest, a finger rubbed all over used to stir
    # over nine tenths up through the film (it leaves two thirds: firm
    # layers of crayon drag part of their film along, and it is half as
    # thick as covers the dust)
    sheet = Sheet(200, 150, seed=1)
    box, inside = (20, 20, 180, 130), (slice(40, 110), slice(40, 160))
    sheet.shade(box, charcoal.VINE, 0.5, 9.0)
    for layer in range(layers):
        sheet.shade(box, crayons.wax("#f2c40f"), 0.8, 9.0, angle=50 * layer - 30)

    def loose_underneath() -> float:
        bonded = sheet.fixed[inside][..., 0] + sheet.film[inside][..., 0]
        return float((sheet.under[inside][..., IV] - bonded).sum(dtype=np.float64))

    before = loose_underneath()
    assert before > 100
    sheet.rub(box, tools.FINGER, 0.7)
    assert loose_underneath() > least * before


def _wax(sheet) -> float:
    return float(sheet.held[..., IW].sum(dtype=np.float64))


def test_the_wax_of_an_oil_pastel_goes_with_the_paste_a_finger_smears():
    # (fusain-9i8: the wax stayed where the pastel was laid; the smear had
    # none, and what was left behind had more than its share)
    oil = pastels.oil("#c0392b")
    sheet = Sheet(300, 150, seed=1)
    for layer in range(3):
        sheet.shade((20, 20, 150, 130), oil, 0.8, 9.0, angle=30 + 50 * layer)
    wax = _wax(sheet)
    assert wax == pytest.approx(oil.wax * _total(sheet), rel=1e-4)
    finger = Smudger(tools.FINGER)
    for _ in range(4):
        for k in range(6):
            sheet.smudge([(60, 50 + 10 * k), (200, 50 + 10 * k)], finger, 0.8)
    smear, left = (slice(55, 100), slice(160, 200)), (slice(55, 100), slice(40, 120))
    pigment = sheet.pigment().astype(np.float64)
    assert pigment[smear].mean() > 0.5
    for where in (smear, left):  # as waxy as the pastel, here and there
        there = float(sheet.held[where][..., IW].sum(dtype=np.float64))
        assert there == pytest.approx(oil.wax * pigment[where].sum(), rel=0.05)
    # the wax is on the sheet or on the finger, which gives it up when cleaned
    assert finger.wax > 0
    assert _wax(sheet) + finger.wax == pytest.approx(wax, rel=1e-6)
    finger.clean()
    assert finger.wax == 0.0
    # and the smear is slick: charcoal shaded over it hardly takes
    before = sheet.pigment().astype(np.float64)
    sheet.shade((155, 50, 205, 105), charcoal.VINE, 0.5, 9.0)
    bare = Sheet(300, 150, seed=1)
    bare.shade((155, 50, 205, 105), charcoal.VINE, 0.5, 9.0)
    took = (sheet.pigment() - before)[smear].sum()
    assert took < 0.1 * bare.pigment()[smear].sum()


@pytest.mark.parametrize(
    "tool",
    [
        tools.FINGER,
        tools.STUMP,
        tools.CHAMOIS,
        tools.KNEADED,
        tools.ERASER,
        tools.SAND_ERASER,
        tools.KNIFE,
    ],
    ids=lambda tool: tool.name,
)
def test_the_wax_a_tool_takes_off_is_on_the_tool(tool):
    sheet = Sheet(200, 150, seed=1)
    box = (20, 20, 180, 130)
    sheet.shade(box, charcoal.VINE, 0.5, 9.0)
    sheet.shade((20, 20, 100, 130), pastels.oil("#2456a6"), 0.7, 9.0, angle=-30)
    for layer in range(2):
        sheet.shade(
            (100, 20, 180, 130), pencils.coloured("#c0392b"), 0.8, angle=60 * layer
        )
    sheet.shade((60, 60, 140, 130), crayons.wax("#f2c40f"), 0.9, 9.0, angle=80)
    wax, total = _wax(sheet), _total(sheet)
    held = Smudger(tool)
    taken = sheet.rub((30, 30, 170, 100), held, 0.8)
    taken += sheet.smudge([(40, 110), (195, 116)], held, 0.9)
    assert held.wax >= 0.0 and _wax(sheet) < wax
    assert _wax(sheet) + held.wax == pytest.approx(wax, rel=1e-6)
    assert _total(sheet) + taken == pytest.approx(total, rel=1e-5)
    assert float(sheet.held[..., IW].min()) >= 0.0


# --------------------------------------------------------------------------
# what it carries


def test_a_loaded_stump_draws_a_pale_tone_on_bare_paper():
    sheet, stump, laid = _loaded()
    load = stump.load
    assert load > 0 and _total(sheet) + load == pytest.approx(laid, rel=1e-5)
    assert sheet.pigment()[:, 150:].max() == 0
    taken = sheet.smudge([(200, 60), (480, 60)], stump, 0.6)
    # it left most of what it carried: less than nothing taken (0.87 of it;
    # 0.92 when the rub that loaded it strayed past the patch and spent some
    # of its load on the bare paper there, so that it carried 7% less)
    assert taken == pytest.approx(stump.load - load) and taken < -0.85 * load
    trail = sheet.pigment()[:, 150:]
    assert float(trail.sum(dtype=np.float64)) == pytest.approx(-taken, rel=1e-4)
    assert trail[:50].max() == 0 and trail[70:].max() == 0  # only where it went
    # a tone paler than the patch it was loaded on, deepest where the stump
    # lands and fading as it goes: bare tooth takes the dust off a loaded
    # stump readily. It tones the paper for longer than it did: it used to
    # lay 0.59, 0.92, 0.98 and 0.99 of the paper's tone over these 28 widths
    tone = _luminance(sheet)
    along = [float(tone[56:64, x : x + 20].mean()) for x in (200, 280, 360, 440)]
    assert along == sorted(along)
    assert float(tone[40:160, 40:100].mean()) < 0.15 < 0.5 < along[0] < 0.76
    assert 0.76 < along[1] < 0.88 and 0.86 < along[2] < 0.95
    assert 0.93 < along[3] < 0.98
    assert _total(sheet) + stump.load == pytest.approx(laid, rel=1e-5)


def test_a_bare_tool_and_a_cleaned_one_bring_nothing_to_bare_paper():
    sheet, stump, laid = _loaded()
    on_the_sheet = _total(sheet)
    assert sheet.smudge([(200, 60), (480, 60)], tools.STUMP, 0.6) == 0.0
    load = stump.load
    assert stump.clean() == pytest.approx(load) and load > 0
    assert stump.load == 0.0 and not stump.carried.any() and stump.wax == 0.0
    assert sheet.smudge([(200, 100), (480, 100)], stump, 0.6) == 0.0
    assert sheet.pigment()[:, 150:].max() == 0
    # what was cleaned off is no longer anywhere
    assert _total(sheet) == on_the_sheet == pytest.approx(laid - load, rel=1e-5)


def test_a_tool_loaded_on_a_scrap_draws_on_another_sheet():
    _, stump, _ = _loaded(charcoal.VINE)
    load = stump.load
    sheet = Sheet(300, 100, seed=2)
    taken = sheet.rub((20, 20, 280, 80), stump, 0.6)
    assert taken < -0.5 * load
    assert _total(sheet) + stump.load == pytest.approx(load, rel=1e-5)
    assert float(_luminance(sheet)[12:28, 30:60].mean()) < 0.92


def test_what_it_carries_is_still_that_pigment():
    sheet, finger, laid = _loaded(charcoal.VINE, tools.FINGER)
    pigment, carried = charcoal.VINE.pigment, finger.carried
    assert carried[IV] == pytest.approx(finger.load) and finger.load > 0
    np.testing.assert_allclose(carried[:N], finger.load * pigment.K, rtol=1e-4)
    assert carried[IS] == pytest.approx(finger.load * pigment.S, rel=1e-4)
    # absorption and scattering are conserved as the volume is
    on_sheet = sum(
        layer[..., :N].sum(axis=(0, 1), dtype=np.float64)
        for layer in (sheet.held, sheet.under, sheet.loose)
    )
    np.testing.assert_allclose(on_sheet + carried[:N], laid * pigment.K, rtol=1e-4)


def test_it_mixes_what_it_picks_up():
    sheet = Sheet(300, 200, seed=1)
    red = pastels.soft("#c0392b")
    sheet.shade((20, 20, 140, 90), red, 0.6, 9.0)
    sheet.shade((20, 110, 140, 180), charcoal.VINE, 0.6, 9.0)
    finger = Smudger(tools.FINGER)

    def absorbs_red(packed) -> float:
        """The share of its absorption that is in the red bands."""
        return float(packed[N - 6 : N].sum() / packed[:N].sum())

    sheet.rub((25, 25, 135, 85), finger, 0.7)
    pastel = absorbs_red(finger.carried)
    assert pastel == pytest.approx(absorbs_red(red.pigment.K), rel=1e-4)
    sheet.rub((25, 115, 135, 175), finger, 0.7)
    # with charcoal on it too, it is greyer than the pastel and redder than
    # the charcoal
    assert pastel < absorbs_red(finger.carried) < absorbs_red(charcoal.VINE.pigment.K)


def test_an_eraser_in_the_hand_keeps_count_of_what_it_took():
    sheet, laid = _toned()
    eraser = Smudger(tools.KNEADED, 14)
    taken = sheet.erase([(40, 60), (360, 60)], eraser, 0.7)
    taken += sheet.erase([(40, 140), (360, 140)], eraser, 0.7)
    assert eraser.load == pytest.approx(taken) and taken > 0
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)
    # it lays nothing back, however much it carries
    bare = Sheet(200, 100, seed=1)
    assert bare.erase([(20, 50), (180, 50)], eraser, 0.9) == 0.0
    assert not bare.pigment().any() and eraser.load == pytest.approx(taken)


def test_a_drawing_rubbed_with_a_tool_in_the_hand_repeats_with_its_seed():
    def worked() -> tuple[Sheet, Smudger]:
        sheet, stump, _ = _loaded(charcoal.COMPRESSED)
        sheet.smudge([(60, 100), (400, 90)], stump, lambda u: 0.8 - 0.4 * u)
        return sheet, stump

    (a, first), (b, second) = worked(), worked()
    for name in ("held", "under", "loose", "cover"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(first.carried, second.carried)


# --------------------------------------------------------------------------
# what it is given


def test_a_smudger_has_its_tool_and_its_own_width():
    assert Smudger(tools.FINGER).width == tools.FINGER.width == 22
    stump = Smudger(tools.STUMP, 16)
    assert (stump.tool, stump.width, stump.load) == (tools.STUMP, 16, 0.0)
    assert repr(stump) == "Smudger(stump, width=16, load=0.00)"
    assert repr(Smudger(tools.Tool(1.0))) == "Smudger(tool, width=12, load=0.00)"
    # as wide as it is, it rubs as a bare tool of that width does
    one, other = _toned()[0], _toned()[0]
    path = [(40, 100), (360, 100)]
    assert one.smudge(path, stump, 0.7) == other.smudge(path, tools.STUMP, 0.7, 16)
    np.testing.assert_array_equal(one.loose, other.loose)


def test_a_smudger_checks_what_it_is_given():
    wrong: Any = "stump"
    with pytest.raises(TypeError, match="tool must be a Tool"):
        Smudger(wrong)
    with pytest.raises(TypeError, match="tool must be a Tool"):
        Smudger(Smudger(tools.STUMP))  # type: ignore[arg-type]
    for width in (0, -2.0, float("nan")):
        with pytest.raises(ValueError, match="width must be"):
            Smudger(tools.STUMP, width)
    sheet, _ = _toned()
    before = sheet.pigment().copy()
    stump = Smudger(tools.STUMP)
    path = [(40, 100), (360, 100)]
    for rub in (
        lambda: sheet.smudge(path, stump, 0.5, 8.0),
        lambda: sheet.erase(path, stump, 0.5, 8.0),
        lambda: sheet.rub(INNER, stump, 0.5, 8.0),
    ):
        with pytest.raises(TypeError, match="a Smudger has its own width"):
            rub()
    with pytest.raises(ValueError, match="pressure must be finite"):
        sheet.rub(INNER, stump, float("nan"))
    assert np.array_equal(before, sheet.pigment()) and stump.load == 0.0
