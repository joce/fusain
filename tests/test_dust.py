"""Dust: what lies loose on the paper, and the tools that move it or take it
off (behaviour 3 of docs/design.md), measured."""

from typing import Any

import numpy as np
import pytest
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
    carried = sheet.smudge(ALONG, tools.STUMP, 0.7, 20.0)
    beyond = float(sheet.pigment()[60:80, 200:].sum())
    assert beyond > 0.01 * before
    # fading as the stump runs out of dust
    assert (
        sheet.pigment()[60:80, 200:225].sum()
        > 2 * sheet.pigment()[60:80, 235:260].sum()
    )
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
    across = np.abs(np.diff(trail, axis=0)).mean()
    along = np.abs(np.diff(trail, axis=1)).mean()
    assert across > 2 * along


def test_what_the_paper_holds_hardly_moves():
    def dragged_out(medium, width) -> float:
        sheet = _drawn(medium, width)
        before = float(sheet.pigment()[BAND].sum())
        sheet.smudge(ALONG, tools.STUMP, 0.7, 20.0)
        return float(sheet.pigment()[:, 200:].sum()) / before

    assert dragged_out(charcoal.VINE, 14.0) > 3 * dragged_out(pencils.B4, 7.0)
    # and the grain of a pencil shading survives the stump
    sheet = _drawn(pencils.B4, 7.0)
    grain = _luminance(sheet).std()
    sheet.smudge(ALONG, tools.STUMP, 0.7, 20.0)
    assert _luminance(sheet).std() > 0.85 * grain


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
# tools


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
