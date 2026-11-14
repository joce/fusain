"""A tool in the hand: a stump or a finger that keeps what it carries from
pass to pass (behaviours 3 and 7 of docs/design.md), measured."""

from typing import Any

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, Smudger, charcoal, pastels, pencils, tools
from fusain.sheet import IS, IV, N

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
    # its load from one pass to the next of a rub
    bare = Sheet(300, 120, seed=1)
    bare.shade(area, pencils.B8, 0.9)
    taken = sum(bare.rub(area, tools.STUMP, 0.9) for _ in range(8))
    assert 0.0 < taken < 0.04 * laid
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
    # a finger wiped clean before every drag lightens it steadily
    before, wiped = dragged(tools.FINGER.with_(width=12.0))
    assert wiped > before + 0.015 and wiped > kept + 0.03


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
# what it carries


def test_a_loaded_stump_draws_a_pale_tone_on_bare_paper():
    sheet, stump, laid = _loaded()
    load = stump.load
    assert load > 0 and _total(sheet) + load == pytest.approx(laid, rel=1e-5)
    assert sheet.pigment()[:, 150:].max() == 0
    taken = sheet.smudge([(200, 60), (480, 60)], stump, 0.6)
    # it left most of what it carried: less than nothing taken
    assert taken == pytest.approx(stump.load - load) and taken < -0.9 * load
    trail = sheet.pigment()[:, 150:]
    assert float(trail.sum(dtype=np.float64)) == pytest.approx(-taken, rel=1e-4)
    assert trail[:50].max() == 0 and trail[70:].max() == 0  # only where it went
    # a tone paler than the patch it was loaded on, that fades as it goes
    tone = _luminance(sheet)
    along = [float(tone[56:64, x : x + 20].mean()) for x in (200, 280, 360, 440)]
    assert along == sorted(along)
    assert float(tone[40:160, 40:100].mean()) < 0.15 < 0.4 < along[0] < 0.75
    assert 0.85 < along[1] < along[3] < 0.999
    assert _total(sheet) + stump.load == pytest.approx(laid, rel=1e-5)


def test_a_bare_tool_and_a_cleaned_one_bring_nothing_to_bare_paper():
    sheet, stump, laid = _loaded()
    on_the_sheet = _total(sheet)
    assert sheet.smudge([(200, 60), (480, 60)], tools.STUMP, 0.6) == 0.0
    load = stump.load
    assert stump.clean() == pytest.approx(load) and load > 0
    assert stump.load == 0.0 and not stump.carried.any()
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
    assert float(_luminance(sheet)[12:28, 30:60].mean()) < 0.9


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
