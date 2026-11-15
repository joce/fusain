"""Rubbing and erasing an area: Sheet.rub, measured."""

import itertools
from typing import Any

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, Smudger, charcoal, hatching, papers, pencils, tools
from fusain import sheet as sheet_module

PATCH = (30, 30, 270, 170)
INSIDE = (slice(60, 140), slice(70, 230))


def _drawn(medium=charcoal.VINE, width=9.0) -> tuple[Sheet, float]:
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
    laid = sum(sheet.shade(PATCH, medium, 0.6, width, angle=a) for a in (30, -40))
    return sheet, laid


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


def _grain(sheet, where=INSIDE) -> float:
    spectrum = sheet.reflectance()[where].astype(np.float64)
    return float((sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE).std())


def test_rubbing_an_area_smooths_all_of_it_and_loses_nothing():
    sheet, laid = _drawn()
    grainy = _grain(sheet)
    taken = sheet.rub(PATCH, tools.FINGER, 0.6)
    assert taken > 0
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)
    assert _grain(sheet) < 0.5 * grainy
    for corner in ((slice(40, 70), slice(40, 90)), (slice(130, 160), slice(210, 260))):
        assert _grain(sheet, corner) < 0.6 * grainy


def test_only_the_area_is_rubbed():
    sheet, _ = _drawn()
    before = sheet.pigment().copy()
    sheet.rub((30, 30, 150, 170), tools.STUMP, 0.6)
    after = sheet.pigment()
    # (a stump is 10 wide: nothing changes more than its width past the area)
    assert np.array_equal(before[:, 165:], after[:, 165:])
    assert not np.array_equal(before[:, 40:140], after[:, 40:140])


def test_an_area_is_rubbed_up_to_its_edge_and_no_further():
    # the passes of a rub are full to their ends, out to the outline of the
    # tool, and stop half a width short of the edge of the area. (Run to the
    # edge, they took as much from the half width of the eraser outside the
    # area as from inside it; and before they were full to their ends, two
    # fifths of that, and a tenth less just inside the edge than further in)
    sheet, _ = _drawn(pencils.B4, 6.0)
    before = sheet.pigment().astype(np.float64)
    sheet.rub((100, 60, 200, 140), tools.ERASER, 0.8, hand=0.0)
    after = sheet.pigment().astype(np.float64)
    there = np.maximum(before[80:120].sum(axis=0), 1e-9)
    taken = 1.0 - after[80:120].sum(axis=0) / there
    middle = float(taken[130:170].mean())
    assert middle > 0.9
    for inside in (slice(100, 104), slice(196, 200)):
        assert float(taken[inside].mean()) == pytest.approx(middle, rel=0.05)
    assert np.array_equal(before[80:120, :100], after[80:120, :100])
    assert np.array_equal(before[80:120, 200:], after[80:120, 200:])


def _luminance(sheet) -> np.ndarray:
    spectrum = sheet.reflectance().astype(np.float64)
    return sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE


@pytest.mark.parametrize("tool", [tools.STUMP, tools.FINGER, tools.ERASER])
@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_a_tool_rubbed_over_a_tone_leaves_nothing_past_its_edge(tool, angle):
    # a clean tool, loaded as it rubs, over the tone it rubs and nothing
    # else, with a natural hand. (It used to sweep past the edge as a
    # pencil's strokes overshoot, and to lay what it had picked up on the
    # bare paper there: the luminance of the paper fell by up to 0.5 within
    # a width of the tool of the edge, at the ends of the passes and along
    # their sides alike, and an eraser took the tone the shading had laid
    # past the edge)
    sheet, _ = _drawn()
    before, light = sheet.pigment().copy(), _luminance(sheet)
    sheet.rub(PATCH, Smudger(tool), 0.6, angle=angle)
    outside = np.ones(before.shape, bool)
    outside[30:170, 30:270] = False
    assert np.abs(sheet.pigment() - before)[outside].max() < 1e-3
    assert np.abs(_luminance(sheet) - light)[outside].max() < 1e-3


@pytest.mark.parametrize(
    ("tool", "pressure", "rim"),
    [(tools.STUMP, 0.6, 4), (tools.FINGER, 0.6, 7), (tools.ERASER, 0.15, 5)],
)
def test_a_rub_works_its_area_up_to_the_edge(tool, pressure, rim):
    # a tone wider than the area, so that it is even across the edge. The
    # outermost pixels of the area are rubbed, and from the rounded rim of
    # the tool in (about a third of its width, ``rim`` pixels) the area is
    # rubbed at the ends of the passes (left and right) and along their
    # sides (top and bottom) as at its middle. (A stump or a finger smooths
    # the grain; an eraser pressed lightly takes a share of the tone. Where
    # the ends of the passes strayed either way and lightened, a light
    # eraser took 0.4 to 0.75 as much as in the middle over a width in)
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
    for a in (30, -40):
        sheet.shade((10, 10, 290, 190), charcoal.VINE, 0.6, 9.0, angle=a)
    before, light = sheet.pigment().astype(np.float64), _luminance(sheet)
    sheet.rub((40, 40, 260, 160), Smudger(tool), pressure)
    after, dark = sheet.pigment().astype(np.float64), _luminance(sheet)

    def rubbed(rows, cols) -> float:
        if tool is tools.ERASER:
            return 1.0 - after[rows, cols].sum() / before[rows, cols].sum()
        return 1.0 - dark[rows, cols].std() / light[rows, cols].std()

    middle = rubbed(slice(70, 130), slice(90, 210))
    assert middle > 0.4
    for k in range(14):
        ends = (rubbed(slice(65, 135), 40 + k), rubbed(slice(65, 135), 259 - k))
        sides = (rubbed(40 + k, slice(80, 220)), rubbed(159 - k, slice(80, 220)))
        least = 0.04 if k < rim else 0.75
        assert min(ends) > least * middle, (k, ends)
        assert min(sides) > least * middle, (k, sides)


def test_the_passes_of_a_rub_keep_the_tool_inside_the_area():
    # the middle of the tool keeps half its width inside the outline, at
    # any angle, in a polygon too, and however unsteady the hand; the
    # outermost passes run along that line
    rectangle = hatching.as_outline((20, 30, 220, 130))
    triangle = hatching.as_outline([(10, 150), (120, 10), (240, 170)])
    for outline in (rectangle, triangle):
        for angle in (0.0, 30.0, 90.0, -45.0):
            for hand in (0.0, 1.0, 3.0):
                rng = np.random.default_rng(5)
                paths = list(hatching.passes(outline, angle, 12.0, 6.0, hand, rng))
                assert len(paths) > 10
                points = np.concatenate([path for path, _ in paths])
                depth = hatching._Sides(outline).nearest(points)[1]
                assert depth.min() >= 6.0 - 1e-6
    passes = list(hatching.passes(rectangle, 0.0, 12.0, 6.0, 1.0, rng))
    first, last = passes[0][0], passes[-1][0]
    assert first[:, 1].min() == pytest.approx(36.0) and first[:, 1].max() < 39.0
    assert last[:, 1].max() == pytest.approx(124.0) and last[:, 1].min() > 121.0
    for path, _ in passes:
        ends = np.sort(path[[0, -1], 0])
        assert 26.0 <= ends[0] < 29.0 and 211.0 < ends[1] <= 214.0


def test_the_pressure_may_be_a_map_and_the_tool_is_lifted_where_it_is_nothing():
    # (and it keeps what it carries while it is lifted)
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
    laid = sheet.shade((30, 30, 200, 170), charcoal.VINE, 0.6, 9)
    before = sheet.pigment().copy()
    assert before[:, 230:].sum() == 0.0

    def left_and_far_right(x, y):
        return 0.7 * ((x < 100) | (x > 245))

    # rub the left of the patch and the bare paper far to its right, not between
    taken = sheet.rub((30, 60, 298, 140), tools.STUMP, left_and_far_right)
    after = sheet.pigment()
    assert not np.array_equal(before[:, 40:90], after[:, 40:90])
    assert np.array_equal(before[:, 115:230], after[:, 115:230])
    # lifted over the gap, the stump lands on the bare paper beyond with the
    # dust it carries, and tones it (lightly: it has gone over little of the
    # tone, and a stump fills slowly; it used to bring a twentieth of what
    # was laid)
    assert after[:, 245:].sum() > 0.008 * laid
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)


def test_an_eraser_clears_an_area():
    sheet, laid = _drawn(pencils.B4, 6.0)
    taken = sheet.rub((100, 60, 200, 140), tools.ERASER, 0.9)
    assert taken > 0.2 * laid
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)
    cleared = sheet.pigment()[75:125, 115:185].sum()
    kept = sheet.pigment()[75:125, 40:85].sum() * (70 / 45)
    assert cleared < 0.11 * kept


def _rubbed_then_drawn(pressure) -> Sheet:
    """A sheet with its left part rubbed as ``pressure`` asks, and a stroke
    drawn after it on the right, where nothing was rubbed."""
    sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=3)
    sheet.shade((20, 20, 180, 180), charcoal.VINE, 0.6, 9)
    sheet.rub((20, 20, 180, 180), tools.STUMP, pressure)
    sheet.shade((220, 40, 290, 160), pencils.B4, 0.6, 5)
    sheet.stroke([(200, 20), (290, 180)], charcoal.COMPRESSED, 0.7, 8)
    return sheet


def test_what_a_rub_draws_from_the_generator_does_not_depend_on_its_pressure():
    def lifted_in(lo, hi, axis=0):
        return lambda x, y: 0.7 * ~(((x, y)[axis] > lo) & ((x, y)[axis] < hi))

    fields = {
        "even": 0.7,
        "gap": lifted_in(70, 110),  # (every pass falls in two runs)
        "gaps": lambda x, y: 0.7 * (np.sin(x / 9.0) > -0.3),  # (in five or six)
        "bare in the top": lifted_in(0, 80, axis=1),  # (some passes have none)
        "nowhere": 0.0,
    }
    sheets = {name: _rubbed_then_drawn(p) for name, p in fields.items()}
    even = sheets["even"]
    for name, sheet in sheets.items():
        # (the same state of the generator, and so the same grain for all that
        # is drawn after, where the rubs did not reach)
        assert sheet._rng.bit_generator.state == even._rng.bit_generator.state, name
        assert np.array_equal(sheet.pigment()[:, 190:], even.pigment()[:, 190:]), name
        assert np.array_equal(sheet.tooth, even.tooth), name
    # (and the rubs differ where they reached)
    assert not np.array_equal(sheets["gap"].pigment()[:, :180], even.pigment()[:, :180])
    assert not np.array_equal(
        sheets["bare in the top"].pigment()[:, :180], even.pigment()[:, :180]
    )


def test_the_seeds_of_the_runs_of_a_rub_repeat_and_are_all_different():
    seeds = [sheet_module._seed_of(123456789, step) for step in range(100_000)]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= seed < 1 << 31 for seed in seeds)
    # (the same on every platform: integer arithmetic, not Python's hash())
    assert seeds[:4] == [123456789, 630408902, 1137361015, 1644313128]
    assert seeds[1000] == 269428861
    # (and no two of them are neighbours, which a kernel's noise would show)
    assert min(abs(a - b) for a, b in itertools.combinations(seeds[:300], 2)) > 1000


def test_the_passes_run_at_the_angle_asked_for_and_repeat_with_the_seed():
    def smeared(angle: float) -> np.ndarray:
        sheet = Sheet(300, 200, paper=papers.CARTRIDGE, seed=1)
        sheet.stroke([(150, 60), (150, 140)], charcoal.COMPRESSED, 0.8, 8)
        sheet.rub((100, 50, 200, 150), tools.STUMP, 0.7, angle=angle, hand=0.0)
        return sheet.pigment()

    across, upright = smeared(0.0), smeared(90.0)
    # rubbed across it, the line is dragged out sideways; along it, hardly
    assert across[70:130, 110:135].sum() > 3 * upright[70:130, 110:135].sum()
    assert np.array_equal(across, smeared(0.0))


@pytest.mark.parametrize(
    ("kw", "error", "message"),
    [
        ({"tool": "finger"}, TypeError, "tool must be a Tool"),
        ({"width": 0.0}, ValueError, "width must be"),
        ({"spacing": -1.0}, ValueError, "spacing must be"),
        ({"angle": float("nan")}, ValueError, "angle must be"),
        ({"hand": -0.5}, ValueError, "hand must be"),
        ({"area": (1, 2, 3)}, ValueError, "an area is"),
        ({"pressure": float("nan")}, ValueError, "pressure must be finite"),
        (
            {"pressure": lambda x, y: np.where(x > 150, np.nan, 0.5)},
            ValueError,
            "pressure must be finite",
        ),
    ],
)
def test_rub_checks_its_arguments_before_it_rubs(kw, error, message):
    sheet, _ = _drawn()
    before = sheet.pigment().copy()
    args: dict[str, Any] = {"area": PATCH, "tool": tools.STUMP, **kw}
    with pytest.raises(error, match=message):
        sheet.rub(**args)
    assert np.array_equal(before, sheet.pigment())
