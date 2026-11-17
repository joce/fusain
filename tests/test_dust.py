"""Dust: what lies loose on the paper, and the tools that move it or take it
off (behaviour 3 of docs/design.md), measured."""

from typing import Any

import numpy as np
import pytest
from murex import filters
from murex import spectral as sp

from fusain import Sheet, Tool, charcoal, papers, pencils, tools
from fusain.sheet import IS, IV, N

PATCH = (20, 20, 180, 130)  # the rectangle shaded
BAND = (slice(62, 78), slice(40, 160))  # what a tool rubbed along y = 70 covers
ALONG = [(30, 70), (250, 70)]  # through the patch and out onto bare paper


def _drawn(medium, width=14.0, pressure=0.5, layers=2) -> Sheet:
    sheet = Sheet(260, 150, paper=papers.CARTRIDGE, seed=1)
    for layer in range(layers):
        sheet.shade(PATCH, medium, pressure, width, angle=30 - 65 * layer)
    return sheet


def _luminance(sheet, where=BAND) -> np.ndarray:
    spectrum = sheet.reflectance()[where].astype(np.float64)
    paper = sp.reflectance_to_linear_rgb(sheet.ground) @ sp.LUMINANCE
    return (sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE) / paper


def _loose_share(sheet) -> float:
    dust = sheet.loose[..., IV].sum() + sheet.under[..., IV].sum()
    return float(dust / sheet.pigment().sum())


def _layers(sheet):
    return sheet.held, sheet.under, sheet.loose


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


# --------------------------------------------------------------------------
# what is dust and what is held


def test_charcoal_is_mostly_dust_and_graphite_is_mostly_held():
    vine = _loose_share(_drawn(charcoal.VINE))
    compressed = _loose_share(_drawn(charcoal.COMPRESSED))
    pencil = _loose_share(_drawn(pencils.B4, width=7.0))
    assert vine > 0.7 > compressed > 0.3 > pencil > 0.05


def test_charcoal_runs_from_grey_to_black():
    def tone(medium, pressure) -> float:
        return float(_luminance(_drawn(medium, pressure=pressure, layers=1)).mean())

    assert tone(charcoal.VINE, 0.3) > 0.6 and tone(charcoal.VINE, 1.0) < 0.3
    assert tone(charcoal.COMPRESSED, 1.0) < 0.1
    for pressure in (0.3, 0.6, 1.0):
        assert tone(charcoal.COMPRESSED, pressure) < tone(charcoal.VINE, pressure) - 0.1


def test_going_over_dust_firmly_presses_it_into_the_paper():
    sheet = Sheet(260, 150, paper=papers.CARTRIDGE, seed=1)
    sheet.stroke([(30, 70), (230, 70)], charcoal.VINE, 0.5, 14.0)
    before = _loose_share(sheet)
    assert before == pytest.approx(1 - charcoal.VINE.binding, rel=1e-4)
    sheet.stroke([(30, 70), (230, 70)], charcoal.VINE, 1.0, 14.0)
    assert _loose_share(sheet) < before - 0.02


# --------------------------------------------------------------------------
# the stump


def test_a_stump_drags_dust_along_and_out_onto_bare_paper():
    sheet = _drawn(charcoal.VINE)
    before = _total(sheet)
    assert sheet.pigment()[:, 200:].max() == 0
    # (what lies in a column of the tone, over the width of the stump)
    tone = float(sheet.pigment()[60:80, 60:160].sum(dtype=np.float64)) / 100
    carried = sheet.smudge(ALONG, tools.STUMP, 0.7, 20.0)
    # what lies past the edge of the patch (at 180), column by column
    out = sheet.pigment()[60:80, 180:].astype(np.float64).sum(axis=0)
    # it drags out what eight columns of the tone hold (it was twenty, a
    # width of the stump), and most of that within one width of the edge
    # (it was two fifths): the dust is on its face, and soon laid back
    assert 5 * tone < out.sum() < 12 * tone
    assert out[:20].sum() > 0.6 * out.sum()
    # a small share goes further, fading as the stump runs out of dust
    assert out[20:45].sum() > 2.5 * out[55:80].sum()
    # to a faint trail (it was a seventh of the tone 70 pixels out), which
    # goes on to the end of the pass
    assert 0.01 * tone < out[62:78].mean() < 0.05 * tone
    # only where the stump went, and only forwards
    assert (
        sheet.pigment()[:55, 200:].max() == 0 and sheet.pigment()[85:, 200:].max() == 0
    )
    np.testing.assert_array_equal(
        sheet.pigment()[:, :18], _drawn(charcoal.VINE).pigment()[:, :18]
    )
    # nothing is lost: it is on the sheet or still on the stump
    assert carried > 0
    np.testing.assert_allclose(_total(sheet) + carried, before, rtol=1e-5)
    assert min(float(layer.min()) for layer in _layers(sheet)) >= 0


def test_a_stump_softens_the_grain_and_greys_the_paper():
    sheet = _drawn(charcoal.VINE)
    grain, cover = _luminance(sheet).std(), sheet.cover[BAND].mean()
    held = sheet.held[BAND][..., IV].sum()
    for _ in range(3):
        sheet.smudge(ALONG, tools.STUMP, 0.7, 20.0)
    assert _luminance(sheet).std() < 0.65 * grain
    assert sheet.cover[BAND].mean() > max(cover + 0.15, 0.95)
    assert sheet.held[BAND][..., IV].sum() > 1.2 * held  # rubbed in
    # the rest of the drawing is untouched
    other = _drawn(charcoal.VINE)
    np.testing.assert_array_equal(sheet.held[100:], other.held[100:])


def test_a_stump_leaves_streaks_along_its_way():
    sheet = _drawn(charcoal.COMPRESSED)
    sheet.smudge(ALONG, tools.STUMP, 0.7, 20.0)
    trail = sheet.pigment()[62:78, 200:225].astype(np.float64)
    trail /= trail.mean(axis=0)  # (without its fading, which is steep there)
    across = np.abs(np.diff(trail, axis=0)).mean()
    along = np.abs(np.diff(trail, axis=1)).mean()
    assert across > 2 * along


def test_what_the_paper_holds_is_hardly_dragged_out():
    def dragged_out(medium, width) -> float:
        sheet = _drawn(medium, width)
        before = float(sheet.pigment()[BAND].sum())
        sheet.smudge(ALONG, tools.STUMP, 0.7, 20.0)
        return float(sheet.pigment()[:, 200:].sum()) / before

    # (past one width of the stump from the edge, where little of either
    # goes: three times as much of the charcoal)
    assert dragged_out(charcoal.VINE, 14.0) > 2.5 * dragged_out(pencils.B4, 7.0)


def _grain_kept(medium, tool=tools.STUMP, fixed=False) -> float:
    """The share of a pencil shading's grain that is left after one pass of
    a tool along it."""
    sheet = _drawn(medium, 7.0)
    grain = _luminance(sheet).std()
    if fixed:
        sheet.fix()
    sheet.smudge(ALONG, tool, 0.7, 20.0)
    return float(_luminance(sheet).std() / grain)


def test_a_stump_evens_a_soft_pencil_and_hardly_a_hard_one():
    # graphite slides under a tool that smears, the more of it a lead has
    kept = [_grain_kept(pencils.grade(name)) for name in ("4H", "HB", "4B", "8B")]
    assert kept == sorted(kept, reverse=True)
    assert kept[0] > 0.75 and kept[2] < 0.65 and kept[3] < 0.45
    # no other pigment that the paper holds does: a coloured pencil keeps
    # its grain nearly as a hard lead does (a little of it was its crumbs,
    # which the stump wipes off the wax and does not lay back on it)
    assert _grain_kept(pencils.coloured("#c0392b")) > 0.7


def test_rubbing_a_pencil_tone_again_smooths_it_without_wiping_it_away():
    sheet = _drawn(pencils.B4, 7.0)
    tone, grain = _luminance(sheet).mean(), _luminance(sheet).std()
    for angle in (0.0, 90.0):
        sheet.rub(PATCH, tools.STUMP, 0.8, angle=angle)
    assert _luminance(sheet).std() < 0.4 * grain
    # (a little darker: the white specks have greyed over)
    assert tone - 0.1 < _luminance(sheet).mean() < tone


@pytest.mark.parametrize("tool", [tools.STUMP, tools.FINGER, tools.CHAMOIS])
def test_graphite_that_slides_is_all_somewhere(tool):
    sheet = _drawn(pencils.B8, 7.0)
    total = _total(sheet)
    carried = sheet.smudge(ALONG, tool, 0.7, 20.0)
    assert carried > 0
    np.testing.assert_allclose(_total(sheet) + carried, total, rtol=1e-5)


def test_an_eraser_does_not_slide_graphite_about():
    # (it lifts; it does not smear)
    sheet = _drawn(pencils.B8, 7.0)
    outside = float(sheet.pigment()[:, 200:].sum())
    sheet.erase(ALONG, tools.KNEADED, 0.7, 20.0)
    assert float(sheet.pigment()[:, 200:].sum()) == outside


def test_fixed_graphite_does_not_slide():
    assert _grain_kept(pencils.B4, fixed=True) > 0.75
    assert _grain_kept(pencils.B4, fixed=True) > _grain_kept(pencils.B4) + 0.15


# --------------------------------------------------------------------------
# erasers


def _left(medium, width, tool, passes) -> tuple[float, float]:
    """Share of the pigment left in the band after erasing, and its tone."""
    sheet = _drawn(medium, width)
    before, total = float(sheet.pigment()[66:74, 40:160].sum()), _total(sheet)
    taken = sum(
        sheet.erase([(10, 70), (190, 70)], tool, 0.7, 14.0) for _ in range(passes)
    )
    np.testing.assert_allclose(_total(sheet) + taken, total, rtol=1e-5)
    middle = (slice(66, 74), slice(40, 160))
    return float(sheet.pigment()[middle].sum()) / before, float(
        _luminance(sheet, middle).mean()
    )


def test_a_kneaded_eraser_lifts_dust_and_leaves_most_of_what_is_held():
    vine, tone = _left(charcoal.VINE, 14.0, tools.KNEADED, 3)
    assert vine < 0.2 and tone > 0.8
    pencil, _ = _left(pencils.B4, 7.0, tools.KNEADED, 3)
    assert pencil > 0.45
    # compressed charcoal is hard to get out
    compressed, _ = _left(charcoal.COMPRESSED, 14.0, tools.KNEADED, 3)
    assert vine < compressed < pencil


def test_a_vinyl_eraser_takes_nearly_all():
    pencil, tone = _left(pencils.B4, 7.0, tools.ERASER, 2)
    assert pencil < 0.1 and tone > 0.93
    vine, tone = _left(charcoal.VINE, 14.0, tools.ERASER, 2)
    assert vine < 0.05 and tone > 0.95


def test_an_eraser_clears_only_where_it_goes_and_lays_nothing():
    sheet = _drawn(charcoal.VINE)
    other = _drawn(charcoal.VINE)
    sheet.erase([(10, 70), (250, 70)], tools.ERASER, 0.8, 10.0)
    np.testing.assert_array_equal(sheet.held[:60], other.held[:60])
    np.testing.assert_array_equal(sheet.loose[82:], other.loose[82:])
    assert sheet.pigment()[:, 200:].max() == 0  # nothing dragged out
    assert (sheet.pigment() <= other.pigment()).all()
    assert min(float(layer.min()) for layer in _layers(sheet)) >= 0


def test_what_is_left_is_still_that_pigment():
    sheet = _drawn(charcoal.VINE)
    sheet.smudge(ALONG, tools.FINGER, 0.6)
    sheet.erase([(10, 100), (190, 100)], tools.KNEADED, 0.7)
    pigment = charcoal.VINE.pigment
    for layer in _layers(sheet):
        volume = layer[..., IV].sum(dtype=np.float64)
        np.testing.assert_allclose(
            layer[..., :N].sum(axis=(0, 1), dtype=np.float64),
            volume * pigment.K,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            layer[..., IS].sum(dtype=np.float64), volume * pigment.S, rtol=1e-4
        )


# --------------------------------------------------------------------------
# the ends of a pass


def _share_taken(tool, path, of="pigment") -> np.ndarray:
    """The share that one pass of ``tool`` along ``path`` takes of what lay
    under the middle of its way (along y = 70), per column: of the pigment,
    or of the layer of dust underneath, which a stump stirs up."""
    medium, width = (charcoal.VINE, 14.0) if of == "under" else (pencils.B4, 7.0)
    sheet = _drawn(medium, width)

    def there() -> np.ndarray:
        volume = sheet.under[..., IV] if of == "under" else sheet.pigment()
        return volume[68:72].astype(np.float64).sum(axis=0)

    before = there()
    rub = sheet.smudge if of == "under" else sheet.erase
    rub(path, tool, 0.7, 14.0)
    return np.where(before > 0, 1.0 - there() / np.maximum(before, 1e-9), 0.0)


@pytest.mark.parametrize(
    "tool, of",
    [(tools.ERASER, "pigment"), (tools.KNEADED, "pigment"), (tools.STUMP, "under")],
    ids=["vinyl eraser", "kneaded eraser", "stump"],
)
def test_a_pass_is_as_strong_where_it_lands_and_lifts_as_in_its_middle(tool, of):
    # a tool used to rub only as it slid on, so the paper under it where it
    # landed or lifted got from nothing (at its back) to a full pass (at its
    # front): at the ends of its path a kneaded eraser took two thirds of
    # what it took further on, and five pixels outside them a sixth
    taken = _share_taken(tool, [(60, 70), (140, 70)], of)
    middle = float(taken[90:110].mean())
    assert middle > 0.25
    for x in (55, 60, 67, 133, 140, 145):
        assert float(taken[x - 1 : x + 2].mean()) == pytest.approx(middle, rel=0.06)
    # round: out to the outline of the tool, which is 14 wide, and no further
    assert taken[50] == 0 and taken[149] == 0


@pytest.mark.parametrize(
    "tool, of",
    [(tools.ERASER, "pigment"), (tools.KNEADED, "pigment"), (tools.STUMP, "under")],
    ids=["vinyl eraser", "kneaded eraser", "stump"],
)
def test_a_dab_is_one_pass_of_the_tool_on_the_paper_under_it(tool, of):
    # (a dab of a kneaded eraser used to take a quarter of what a pass
    # takes, and a pass three pixels long a third)
    middle = float(_share_taken(tool, [(60, 70), (140, 70)], of)[90:110].mean())
    for path in ([(100, 70)], [(100, 70), (103, 70)]):
        taken = _share_taken(tool, path, of)
        assert float(taken[99:102].mean()) == pytest.approx(middle, rel=0.06)
        assert not taken[:91].any() and not taken[112:].any()


def test_what_a_tool_takes_where_it_lands_and_lifts_is_on_it():
    sheet = _drawn(charcoal.VINE)
    total = _total(sheet)
    taken = sheet.erase([(100, 70)], tools.KNEADED, 0.8, 20.0)
    taken += sheet.smudge([(60, 100), (64, 100)], tools.FINGER, 0.8)
    taken += sheet.smudge([(170, 40), (200, 44)], tools.STUMP, 0.8, 20.0)  # and off
    assert taken > 0
    np.testing.assert_allclose(_total(sheet) + taken, total, rtol=1e-5)
    assert min(float(layer.min()) for layer in _layers(sheet)) >= 0


# --------------------------------------------------------------------------
# tools


def test_the_dust_of_a_pass_has_settled_when_the_pass_is_done():
    sheet = _drawn(charcoal.VINE)
    sheet.stroke(
        [(-30, -30), (300, 200)], charcoal.COMPRESSED, 0.8, 9.0
    )  # off both ends
    sheet.stroke([(60, 75)], charcoal.VINE, 1.0, 12.0)  # a dot
    assert not sheet._fresh.any()
    copy = sheet.copy()
    copy.stroke([(30, 40), (200, 90)], charcoal.VINE, 0.6, 9.0)
    assert not copy._fresh.any() and copy._fresh is not sheet._fresh


def test_rubbing_bare_paper_does_nothing():
    sheet = Sheet(120, 80, seed=1)
    assert sheet.smudge([(10, 40), (110, 40)]) == 0.0
    assert sheet.erase([(10, 40), (110, 40)], tools.ERASER) == 0.0
    assert not sheet.pigment().any() and not sheet.cover.any()


def test_a_rubbed_drawing_repeats_with_its_seed():
    def worked() -> Sheet:
        sheet = _drawn(charcoal.COMPRESSED)
        sheet.smudge(ALONG, tools.STUMP, lambda u: 0.8 - 0.4 * u)
        sheet.erase([(10, 100), (190, 100)])
        return sheet

    a, b = worked(), worked()
    for name in ("held", "under", "loose", "cover"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_a_tool_has_its_own_width_unless_one_is_given():
    def reach(tool, width=None) -> int:
        sheet = _drawn(charcoal.VINE)
        other = _drawn(charcoal.VINE)
        sheet.smudge(ALONG, tool, 0.7, width)
        rows = np.nonzero((sheet.loose != other.loose).any(axis=(1, 2)))[0]
        return int(rows.max() - rows.min() + 1)

    assert reach(tools.FINGER) > 1.5 * reach(tools.STUMP)
    assert reach(tools.STUMP, 30.0) > 2 * reach(tools.STUMP)
    assert (
        tools.TORTILLON.width
        < tools.STUMP.width
        < tools.FINGER.width
        < tools.CHAMOIS.width
    )


def test_no_pressure_moves_nothing():
    sheet, other = _drawn(charcoal.VINE), _drawn(charcoal.VINE)
    assert sheet.smudge(ALONG, tools.STUMP, 0.0) == 0.0
    np.testing.assert_array_equal(sheet.loose, other.loose)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"lift_loose": -1}, "lift_loose must be"),
        ({"lift_held": float("nan")}, "lift_held must be"),
        ({"lay": 1.5}, "lay is a share"),
        ({"press": -0.1}, "press must be"),
        ({"smear": float("inf")}, "smear must be"),
        ({"width": 0}, "width must be"),
        ({"holds": 0.0}, "holds must be"),
        ({"holds": float("nan")}, "holds must be"),
    ],
)
def test_a_tool_checks_its_numbers(changes, message):
    with pytest.raises(ValueError, match=message):
        tools.STUMP.with_(**changes)
    mine = Tool(1.0, lay=0.3, name="rag")
    assert (mine.lift_held, mine.press, mine.width, mine.name) == (
        0.0,
        0.0,
        12.0,
        "rag",
    )
    # a tool holds any amount unless it says how much
    assert mine.holds == float("inf") == tools.KNEADED.holds == tools.CHAMOIS.holds
    assert tools.STUMP.holds > tools.FINGER.holds > 0


@pytest.mark.parametrize(
    "kw, error, message",
    [
        ({"tool": "stump"}, TypeError, "tool must be a Tool"),
        ({"tool": pencils.HB}, TypeError, "tool must be a Tool"),
        ({"width": -3}, ValueError, "width must be"),
        ({"pressure": float("nan")}, ValueError, "pressure must be finite"),
    ],
)
def test_rubbing_checks_its_arguments_before_it_touches_the_sheet(kw, error, message):
    sheet, other = _drawn(charcoal.VINE), _drawn(charcoal.VINE)
    arguments: dict[str, Any] = {"tool": tools.STUMP}
    arguments.update(kw)
    for rub in (sheet.smudge, sheet.erase):
        with pytest.raises(error, match=message):
            rub(ALONG, **arguments)
    np.testing.assert_array_equal(sheet.loose, other.loose)


# --------------------------------------------------------------------------
# rubbing evens the dust out


def _unevenness(sheet, where=(slice(60, 140), slice(70, 230))) -> tuple[float, float]:
    """How uneven a tone is: from pixel to pixel, and in streaks (seen
    through a blur nine pixels wide)."""
    spectrum = sheet.reflectance()[where].astype(np.float64)
    tone = sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE
    return float(tone.std()), float(filters.box_blur(tone, 4).std())


def _rub_all_over(sheet, tool, passes=2) -> None:
    for _ in range(passes):
        for k in range(0, 150, 9):
            ends = [(40, 28 + k), (260, 30 + k)]
            sheet.smudge(ends if (k // 9) % 2 else ends[::-1], tool, 0.6)


@pytest.mark.parametrize(
    "how",
    [
        {"pressure": 0.8, "width": 9, "tilt": 1},
        {"pressure": 0.4, "width": 7, "spacing": 4.2},
    ],
    ids=["the side of the stick", "light strokes apart"],
)
def test_a_finger_evens_out_charcoal_however_it_was_laid(how):
    sheet = Sheet(300, 200, paper=papers.INGRES, seed=1)
    for angle in (30, -40):
        sheet.shade((30, 30, 270, 170), charcoal.VINE, angle=angle, **how)
    grainy, streaky = _unevenness(sheet)
    # bare paper between: 0.036 to 0.071 of it over twenty sheets
    assert (sheet.pigment()[60:140, 70:230] == 0).mean() > 0.02
    _rub_all_over(sheet, tools.FINGER)
    smooth, even = _unevenness(sheet)
    assert smooth < 0.5 * grainy
    assert even < 0.6 * streaky
    assert (sheet.pigment()[60:140, 70:230] == 0).mean() == 0.0


def test_a_stump_smooths_compressed_charcoal_too():
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
    for angle in (30, -40):
        sheet.shade((30, 30, 270, 170), charcoal.COMPRESSED, 0.6, 9, angle=angle)
    grainy, _ = _unevenness(sheet)
    _rub_all_over(sheet, tools.STUMP)
    assert _unevenness(sheet)[0] < 0.2 * grainy


def test_spreading_the_dust_loses_none():
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
    laid = sheet.shade((30, 30, 270, 170), charcoal.VINE, 0.7, 9)
    carried = 0.0
    for k in range(0, 150, 9):
        carried += sheet.smudge([(20, 28 + k), (295, 30 + k)], tools.FINGER, 0.7)
    total = float(sheet.pigment().sum(dtype=np.float64))
    assert total + carried == pytest.approx(laid, rel=1e-5)
