"""Shading an area with strokes side by side, as a hand does."""

from typing import Any

import numpy as np
import pytest

from fusain import Paper, Sheet, pencils

LEVEL = Paper(depth=0)  # nothing but the strokes themselves shows


def _sheet(paper=None, seed=2) -> Sheet:
    return Sheet(200, 150, paper=paper, seed=seed)


def test_a_shaded_rectangle_is_filled_and_nothing_else_is():
    sheet = _sheet()
    laid = sheet.shade((40, 30, 160, 110), pencils.B2, 0.5)
    pigment = sheet.pigment()
    np.testing.assert_allclose(pigment.sum(dtype=np.float64), laid, rtol=1e-4)
    assert (pigment[40:100, 50:150] > 0).mean() > 0.5
    # strokes overshoot their ends a little, as a hand's do, but stay near
    outside = pigment.copy()
    outside[20:120, 28:172] = 0
    assert outside.max() == 0
    assert pigment[:30, :].sum() + pigment[110:, :].sum() < 0.1 * laid


def test_a_polygon_is_shaded_inside_its_outline_only():
    sheet = _sheet(LEVEL)
    # a U: the notch between its arms stays bare
    outline = [
        (30, 20),
        (70, 20),
        (70, 90),
        (130, 90),
        (130, 20),
        (170, 20),
        (170, 130),
        (30, 130),
    ]
    sheet.shade(outline, pencils.B2, 0.6, hand=0)
    pigment = sheet.pigment()
    assert pigment[30:80, 80:120].max() == 0  # the notch
    for arm in (
        pigment[30:80, 38:62],
        pigment[30:80, 138:162],
        pigment[100:122, 40:160],
    ):
        assert (arm > 0).mean() > 0.9
    # a triangle
    triangle = _sheet(LEVEL)
    triangle.shade([(20, 130), (180, 130), (100, 20)], pencils.B2, 0.6, hand=0)
    marks = triangle.pigment() > 0
    assert marks[100:125, 80:120].mean() > 0.9
    assert not marks[:60, :40].any() and not marks[:60, 160:].any()


def test_strokes_run_at_the_angle_asked_for():
    def spread(angle) -> tuple[float, float]:
        sheet = _sheet(LEVEL)
        sheet.shade(
            (30, 30, 170, 120), pencils.B2, 0.6, 4.0, angle, spacing=9.0, hand=0
        )
        middle = sheet.pigment()[50:100, 50:150].astype(np.float64)
        # how much the tone changes going across the sheet, and going down it
        return float(np.abs(np.diff(middle, axis=1)).mean()), float(
            np.abs(np.diff(middle, axis=0)).mean()
        )

    across, down = spread(0.0)
    assert down > 20 * across  # strokes across the sheet: nothing changes along them
    across, down = spread(90.0)
    assert across > 20 * down
    across, down = spread(45.0)
    assert 0.5 < across / down < 2.0


def test_a_steady_hand_is_regular_and_a_natural_one_is_not():
    def measure(hand) -> tuple[float, float]:
        sheet = _sheet(LEVEL)
        sheet.shade(
            (20, 20, 180, 130), pencils.B2, 0.5, 4.0, 0.0, spacing=10.0, hand=hand
        )
        marks = sheet.pigment()[30:120, 40:160].astype(np.float64)
        rows = marks.mean(axis=1)
        # how alike one stroke and the next are: the tone down the sheet
        # against itself one spacing further
        regular = float(np.corrcoef(rows[:-10], rows[10:])[0, 1])
        on_a_stroke = rows > 0.5 * rows.max()
        uneven = float((marks.std(axis=1)[on_a_stroke] / rows[on_a_stroke]).mean())
        return regular, uneven

    regular, uneven = measure(0.0)
    assert regular > 0.95 and uneven < 0.005  # evenly spaced, the same all along
    regular, uneven = measure(1.0)
    assert regular < 0.9 and uneven > 0.05
    # and no two of a natural hand's strokes are alike
    a, b = _sheet(LEVEL, seed=1), _sheet(LEVEL, seed=1)
    a.shade((20, 20, 180, 130), pencils.B2, 0.5, hand=1.0)
    b.shade((20, 20, 180, 130), pencils.B2, 0.5, hand=0.0)
    assert not np.array_equal(a.held, b.held)


def test_the_tone_of_a_shading_is_that_of_its_pressure():
    def volume(pressure, hand=1.0) -> float:
        sheet = _sheet()
        sheet.shade((20, 20, 180, 130), pencils.B2, pressure, hand=hand)
        return float(sheet.pigment()[50:100, 50:150].mean())

    light, firm = volume(0.25), volume(0.8)
    assert firm > 2.5 * light > 0
    # the hand makes it uneven, not lighter or darker by much
    assert abs(volume(0.5) - volume(0.5, hand=0.0)) < 0.25 * volume(0.5, hand=0.0)


def test_pressure_may_change_across_the_area():
    sheet = _sheet()
    sheet.shade((20, 20, 180, 130), pencils.B2, lambda x, y: (x - 20) / 160)
    pigment = sheet.pigment()
    left, right = pigment[40:110, 30:70].mean(), pigment[40:110, 130:170].mean()
    assert right > 4 * left
    down = _sheet()
    down.shade((20, 20, 180, 130), pencils.B2, lambda x, y: (y - 20) / 110, angle=80)
    assert (
        down.pigment()[100:125, 40:160].mean()
        > 4 * down.pigment()[25:50, 40:160].mean()
    )


def test_shading_twice_builds_tone_and_repeats_with_the_seed():
    once, twice, again = _sheet(), _sheet(), _sheet()
    once.shade((20, 20, 180, 130), pencils.HB, 0.4)
    for sheet in (twice, again):
        sheet.shade((20, 20, 180, 130), pencils.HB, 0.4)
        sheet.shade((20, 20, 180, 130), pencils.HB, 0.4, angle=-40)
    assert twice.pigment().sum() > 1.4 * once.pigment().sum()
    np.testing.assert_array_equal(twice.held, again.held)
    assert not np.array_equal(_sheet(seed=3).tooth, once.tooth)


def test_a_very_unsteady_hand_still_draws():
    # (its strokes' widths were drawn from a range that went below zero)
    sheet = _sheet()
    assert sheet.shade((20, 20, 180, 130), pencils.B2, 0.5, hand=25.0) > 0


def test_spacing_sets_how_many_strokes_there_are():
    def laid(spacing) -> float:
        return _sheet(LEVEL).shade(
            (20, 20, 180, 130), pencils.B2, 0.5, 4.0, spacing=spacing
        )

    assert laid(4.0) > 1.7 * laid(8.0)


@pytest.mark.parametrize(
    "kw, error, message",
    [
        ({"area": (10, 10, 50)}, ValueError, "an area is a rectangle"),
        ({"area": [(10, 10), (50, 50)]}, ValueError, "an area is a rectangle"),
        ({"area": "all of it"}, TypeError, "an area is a rectangle"),
        ({"area": (10, 10, 50, float("nan"))}, ValueError, "finite numbers"),
        ({"medium": "HB"}, TypeError, "medium must be a Medium"),
        ({"width": 0}, ValueError, "width must be"),
        ({"spacing": -1}, ValueError, "spacing must be"),
        ({"angle": float("inf")}, ValueError, "angle must be"),
        ({"hand": -0.5}, ValueError, "hand must be"),
    ],
)
def test_shade_checks_its_arguments_before_it_draws(kw, error, message):
    sheet = _sheet()
    arguments: dict[str, Any] = {"area": (20, 20, 180, 130), "medium": pencils.HB}
    arguments.update(kw)
    with pytest.raises(error, match=message):
        sheet.shade(**arguments)
    assert not sheet.pigment().any()
