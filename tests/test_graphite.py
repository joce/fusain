"""Graphite on paper: grain, saturation and conservation (behaviours 1, 2 and
7 of docs/design.md), measured."""

from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from murex import Pigment, Polyline
from murex import spectral as sp

from fusain import Medium, Paper, Sheet, papers, pencils
from fusain._engine import unit_noise
from fusain.sheet import IS, IV, N

BOX = (10, 10, 130, 50)  # the rectangle shaded
INSIDE = (slice(15, 45), slice(20, 120))  # well inside it


def _sheet(paper=papers.CARTRIDGE, seed=1) -> Sheet:
    return Sheet(140, 60, paper=paper, seed=seed)


def _shade(sheet, medium, pressure, layers=1, width=3.0, gap=1.5) -> float:
    """Even layers over BOX: strokes side by side. Returns the volume laid."""
    x0, y0, x1, y1 = BOX
    laid = 0.0
    for _ in range(layers):
        for y in np.arange(y0, y1 + 0.5 * gap, gap):
            laid += sheet.stroke([(x0, y), (x1, y)], medium, pressure, width)
    return laid


def _tone(sheet) -> float:
    """Mean luminance well inside BOX, as a share of the bare paper's."""
    spectrum = sheet.reflectance()[INSIDE].astype(np.float64)
    paper = sp.reflectance_to_linear_rgb(sheet.ground) @ sp.LUMINANCE
    return float((sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE).mean() / paper)


def _shaded(medium, pressure, layers=1, paper=papers.CARTRIDGE) -> Sheet:
    sheet = _sheet(paper)
    _shade(sheet, medium, pressure, layers)
    return sheet


# --------------------------------------------------------------------------
# grain


def test_a_light_touch_marks_the_peaks_and_leaves_the_valleys_white():
    sheet = _shaded(pencils.HB, 0.3)
    tooth, pigment = sheet.tooth[INSIDE], sheet.pigment()[INSIDE]
    marked = sheet.cover[INSIDE] > 0.5  # at least half covered
    assert 0.25 < marked.mean() < 0.75
    assert tooth[marked].mean() > tooth[~marked].mean() + 0.2
    low, high = tooth < np.quantile(tooth, 0.25), tooth > np.quantile(tooth, 0.75)
    assert pigment[low].sum() < 0.02 * pigment.sum()
    assert pigment[high].sum() > 0.5 * pigment.sum()
    # the valleys of the picture are still the colour of the paper
    assert sheet.cover[INSIDE][low].mean() < 0.05


def test_pressing_harder_reaches_the_valleys_and_darkens():
    pressures = (0.15, 0.35, 0.6, 1.0)
    sheets = [_shaded(pencils.HB, p) for p in pressures]
    tones = [_tone(s) for s in sheets]
    covers = [float(s.cover[INSIDE].mean()) for s in sheets]
    assert all(a > b + 0.03 for a, b in pairwise(tones))
    assert all(a < b for a, b in pairwise(covers))
    # an HB pressed as hard as it goes still leaves white specks in the
    # valleys; only the softest leads reach them all
    assert covers[0] < 0.3 and 0.85 < covers[-1] < 0.97
    assert _shaded(pencils.B8, 1.0).cover[INSIDE].mean() > 0.99
    assert tones[0] > 0.93 and tones[-1] < 0.5


def test_softer_leads_are_darker_and_reach_further():
    names = ("4H", "2H", "HB", "2B", "4B", "6B", "8B")
    sheets = [_shaded(pencils.grade(name), 0.5) for name in names]
    tones = [_tone(s) for s in sheets]
    covers = [float(s.cover[INSIDE].mean()) for s in sheets]
    assert all(a > b + 0.02 for a, b in pairwise(tones))
    assert all(a < b for a, b in pairwise(covers))
    # pressed as hard as it goes, a hard lead stays pale and a soft one is dark
    assert _tone(_shaded(pencils.H4, 1.0)) > 0.65
    assert _tone(_shaded(pencils.B6, 1.0)) < 0.2


def test_a_smooth_paper_takes_an_even_tone_and_a_rough_one_a_broken_one():
    smooth = _shaded(pencils.B2, 0.35, paper=papers.BRISTOL)
    rough = _shaded(pencils.B2, 0.35, paper=papers.ROUGH)
    assert smooth.cover[INSIDE].mean() > 0.95
    assert rough.cover[INSIDE].mean() < 0.6
    level = _shaded(pencils.B2, 0.1, paper=Paper(depth=0))
    assert level.cover[INSIDE].min() == 1.0


def test_a_paper_that_gives_lets_the_tip_reach_further():
    firm = _shaded(pencils.HB, 0.4, paper=Paper(give=0.0))
    soft = _shaded(pencils.HB, 0.4, paper=Paper(give=1.0))
    assert soft.cover[INSIDE].mean() > firm.cover[INSIDE].mean() + 0.2
    assert _tone(soft) < _tone(firm) - 0.05


# --------------------------------------------------------------------------
# saturation


def test_each_layer_darkens_less_and_then_the_tooth_is_full():
    sheet, tones, counts = _sheet(), [1.0], (0, 1, 2, 4, 8, 16, 32, 64)
    for done, layers in pairwise(counts):
        _shade(sheet, pencils.HB, 0.35, layers - done)
        tones.append(_tone(sheet))
    per_layer = -np.diff(tones) / np.diff(counts)
    assert (per_layer > 0).all()  # it never lightens
    assert (np.diff(per_layer) < 0).all()  # each layer adds less than the last
    assert per_layer[0] > 0.1 and per_layer[-1] < 0.01 * per_layer[0]
    assert 0.4 < tones[-1] < 0.6  # light layers never make it dark


def test_a_hard_lead_stops_at_a_lighter_tone_than_a_soft_one():
    full = [_tone(_shaded(pencils.grade(g), 0.35, 48)) for g in ("2H", "HB", "4B")]
    assert full[0] > full[1] + 0.1 and full[1] > full[2] + 0.1
    # and no thickness anywhere is beyond what the lead can fill
    hard, soft = _shaded(pencils.H2, 1.0, 48), _shaded(pencils.B4, 1.0, 48)
    assert hard.pigment().max() < 0.5 * soft.pigment().max()


def test_a_toothier_paper_takes_more_before_it_is_full():
    def volume(paper) -> float:
        return float(_shaded(pencils.B2, 0.6, 48, paper=paper).pigment()[INSIDE].mean())

    assert volume(papers.BRISTOL) < volume(papers.CARTRIDGE) < volume(papers.ROUGH)

    def layers_to_fill(paper) -> int:
        sheet, last = _sheet(paper), 1.0
        for layer in range(1, 60):
            _shade(sheet, pencils.B2, 0.6)
            tone = _tone(sheet)
            if last - tone < 0.004:
                return layer
            last = tone
        return 60

    assert layers_to_fill(papers.BRISTOL) < layers_to_fill(papers.CARTRIDGE)


# --------------------------------------------------------------------------
# conservation


def test_what_leaves_the_tip_is_on_the_paper():
    sheet = _sheet()
    laid = _shade(sheet, pencils.B2, 0.6, layers=3)
    assert laid > 0
    np.testing.assert_allclose(sheet.pigment().sum(dtype=np.float64), laid, rtol=1e-4)
    # the paper holds at least the share the lead binds (going over it again
    # presses in some of the dust), and some dust is left
    held = sheet.held[..., IV].sum(dtype=np.float64)
    assert 0.999 * laid * pencils.B2.binding <= held < 0.99 * laid
    one_pass = _sheet()
    once = one_pass.stroke([(20, 30), (120, 30)], pencils.B2, 0.6, 4.0)
    held = one_pass.held[..., IV].sum(dtype=np.float64)
    assert held == pytest.approx(once * pencils.B2.binding, rel=1e-4)
    # and it is that lead: its absorption and scattering, in that volume
    pigment = pencils.B2.pigment
    for layer in (sheet.held, sheet.under, sheet.loose):
        volume = layer[..., IV].sum(dtype=np.float64)
        np.testing.assert_allclose(
            layer[..., :N].sum(axis=(0, 1), dtype=np.float64),
            volume * pigment.K,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            layer[..., IS].sum(dtype=np.float64), volume * pigment.S, rtol=1e-4
        )
    # nothing was laid outside the strokes
    assert sheet.pigment()[:6].max() == 0 and sheet.pigment()[55:].max() == 0


def test_off_the_sheet_nothing_is_laid_and_nothing_is_worn():
    sheet = _sheet()
    assert sheet.stroke([(-50, -20), (-10, -5)], pencils.B4, 0.8, 4.0) == 0.0
    assert not sheet.pigment().any() and not sheet.cover.any()
    # a stroke that runs off the edge wears only for its part on the paper
    part = sheet.stroke([(100, 30), (180, 30)], pencils.B4, 0.8, 4.0)
    whole = _sheet().stroke([(40, 30), (120, 30)], pencils.B4, 0.8, 4.0)
    assert 0.3 * whole < part < 0.7 * whole
    np.testing.assert_allclose(sheet.pigment().sum(dtype=np.float64), part, rtol=1e-4)


def test_no_pressure_lays_nothing():
    sheet = _sheet()
    assert sheet.stroke([(20, 30), (120, 30)], pencils.B4, pressure=0.0) == 0.0
    assert not sheet.pigment().any()


# --------------------------------------------------------------------------
# strokes


def test_a_drawing_repeats_with_its_seed_and_no_two_passes_are_alike():
    a, b = _shaded(pencils.HB, 0.4), _shaded(pencils.HB, 0.4)
    for name in ("held", "under", "loose", "cover"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    once, twice = _sheet(), _sheet()
    for sheet, passes in ((once, 1), (twice, 2)):
        for _ in range(passes):
            sheet.stroke([(20, 30), (120, 30)], pencils.HB, 0.35, 6.0)
    # the second pass touches pixels the first one missed
    assert (twice.cover > 0).sum() > 1.05 * (once.cover > 0).sum()
    # a copy goes on like the original
    copy = a.copy()
    for sheet in (a, copy):
        sheet.stroke([(20, 30), (120, 30)], pencils.B2, 0.5)
    np.testing.assert_array_equal(a.held, copy.held)


def test_the_tone_of_a_pass_does_not_depend_on_the_width_of_the_tip():
    def centre(width) -> float:
        sheet = _sheet()
        sheet.stroke([(10, 30), (130, 30)], pencils.B2, 0.6, width)
        return float(sheet.pigment()[29:31, 20:120].mean())

    narrow, wide = centre(3.0), centre(12.0)
    assert abs(narrow - wide) < 0.25 * wide
    # but a wider tip lays more in all
    assert _sheet().stroke([(10, 30), (130, 30)], pencils.B2, 0.6, 12.0) > 2.5 * (
        _sheet().stroke([(10, 30), (130, 30)], pencils.B2, 0.6, 3.0)
    )


def test_a_line_is_darkest_along_its_middle():
    sheet = _sheet(Paper(depth=0))
    sheet.stroke([(10, 30), (130, 30)], pencils.B2, 0.6, 9.0)
    across = sheet.pigment()[:, 40:100].mean(axis=1)
    assert across[29:31].min() > across[27] > across[26] > across[25] > 0
    assert across[:25].max() == 0 and across[35:].max() == 0


def test_a_line_has_streaks_along_it():
    # on a level paper nothing but the lead itself varies: it bears more
    # in some bands across the tip than in others, all along the stroke
    sheet = _sheet(Paper(depth=0))
    sheet.stroke(Polyline([(10, 30), (130, 30)]), pencils.B2, 0.6, 14.0)
    middle = sheet.pigment()[25:35, 30:110].astype(np.float64)
    assert middle.mean(axis=1).std() > 0.05 * middle.mean()
    assert middle.std(axis=1).max() < 0.01 * middle.mean()


def test_a_firm_line_keeps_the_grain_of_the_paper():
    sheet = _sheet()
    sheet.stroke(Polyline([(10, 30), (130, 30)]), pencils.HB, 0.8, 8.0)
    rows = (slice(28, 32), slice(20, 120))  # along the middle of the line
    tooth, pigment = sheet.tooth[rows].ravel(), sheet.pigment()[rows].ravel()
    # the high ground took more than the low, and some valleys took nothing
    assert np.corrcoef(tooth, pigment)[0, 1] > 0.5
    assert 0.02 < (sheet.cover[rows] < 0.1).mean() < 0.3
    assert pigment.std() > 0.3 * pigment.mean()


def _marked_rows(sheet) -> int:
    return int(np.count_nonzero(sheet.pigment()[:, 50:90].sum(axis=1)))


def test_a_pencil_on_its_side_marks_a_band_several_times_wider():
    marks = {}
    for tilt in (0.0, 0.5, 1.0):
        sheet = _sheet(Paper(depth=0))
        sheet.stroke(Polyline([(30, 30), (110, 30)]), pencils.B2, 0.6, 5.0, tilt)
        marks[tilt] = sheet.pigment()
        # broader across the stroke: up to 5 times, no more
        assert abs(_marked_rows(sheet) - 5.0 * (1 + 4 * tilt)) <= 2
        # but no longer: it starts and stops where the point would
        columns = np.nonzero(marks[tilt].sum(axis=0))[0]
        assert 26 <= columns.min() <= 28 and 111 <= columns.max() <= 113
    # its edges are soft: the tone falls away from the middle of the band,
    # where the point's is flat right up to its rim
    across = marks[1.0][:, 50:90].mean(axis=1)
    assert across[30 - 9 : 30 + 9].min() < 0.75 * across[29:31].mean()
    point = marks[0.0][:, 50:90].mean(axis=1)
    assert point[29:31].min() > 0.8 * point.max()


def test_the_side_of_a_lead_lays_a_lighter_grainier_tone_than_its_point():
    def middle(tilt, pressure) -> tuple[float, float]:
        sheet = _sheet()
        sheet.stroke([(10, 30), (130, 30)], pencils.B2, pressure, 5.0, tilt)
        rows = (slice(29, 31), slice(30, 110))
        return float(sheet.pigment()[rows].mean()), float(sheet.cover[rows].mean())

    for pressure in (0.3, 0.6, 1.0):
        point, side = middle(0.0, pressure), middle(1.0, pressure)
        assert side[0] < 0.75 * point[0]  # lighter
        assert side[1] < point[1] - 0.1  # and it skips more of the valleys

    # all the same it lays more in all, over its wider band
    def laid(tilt) -> float:
        return _sheet().stroke([(10, 30), (130, 30)], pencils.B2, 0.6, 5.0, tilt)

    assert laid(0.0) < laid(1.0) < 5 * laid(0.0)


def test_shading_with_the_side_takes_fewer_broader_strokes(monkeypatch):
    def shaded(tilt) -> tuple[float, float]:
        sheet = Sheet(200, 150, seed=2)
        sheet.shade((20, 20, 180, 130), pencils.B2, 0.6, 5.0, tilt=tilt)
        middle = (slice(50, 100), slice(50, 150))
        return float(sheet.cover[middle].mean()), float(sheet.pigment()[middle].mean())

    point, side = shaded(0.0), shaded(1.0)
    assert 0.3 < side[0] < point[0] - 0.1 and side[1] < point[1]

    # the default spacing follows the width of the mark: a fifth of the
    # strokes, each five times wider
    def strokes(tilt) -> list[float]:
        sheet, widths = Sheet(200, 150, seed=2), []
        draw = sheet.stroke

        def counting(path, medium, pressure, width, tilt):
            widths.append(width)
            return draw(path, medium, pressure, width, tilt)

        monkeypatch.setattr(sheet, "stroke", counting)
        sheet.shade((20, 20, 180, 130), pencils.B2, 0.6, 5.0, hand=0.0, tilt=tilt)
        return widths

    point, side = strokes(0.0), strokes(1.0)
    assert 4 * len(side) <= len(point) <= 6 * len(side)
    assert set(point) == set(side) == {5.0}  # the point's width, either way


@pytest.mark.parametrize("tilt", [-0.1, 1.5, float("nan")])
def test_tilt_is_between_point_and_side(tilt):
    sheet = _sheet()
    with pytest.raises(ValueError, match="tilt must be between 0"):
        sheet.stroke([(20, 30), (120, 30)], pencils.HB, 0.5, 4.0, tilt)
    with pytest.raises(ValueError, match="tilt must be between 0"):
        sheet.shade((20, 20, 100, 50), pencils.HB, tilt=tilt)
    assert not sheet.pigment().any()


def test_how_a_path_is_given_does_not_change_the_line():
    two = _sheet().stroke(Polyline([(20, 30), (120, 30)]), pencils.B2, 0.5)
    many = _sheet().stroke(
        Polyline([(x, 30) for x in range(20, 121, 5)]), pencils.B2, 0.5
    )
    np.testing.assert_allclose(many, two, rtol=0.02)


def test_pressure_may_vary_along_the_stroke():
    sheet = _sheet()
    sheet.stroke([(10, 30), (130, 30)], pencils.B2, lambda u: u, 6.0)
    pigment = sheet.pigment()
    assert pigment[:, 20:50].sum() < 0.3 * pigment[:, 90:120].sum()
    per_point = _sheet()
    per_point.stroke(
        Polyline([(10, 30), (70, 30), (130, 30)]), pencils.B2, [0, 1, 0], 6.0
    )
    middle = per_point.pigment()[:, 55:85].sum()
    assert middle > 3 * per_point.pigment()[:, 10:40].sum()
    # above 1 is as hard as it goes; below 0 is no pressure
    full = _sheet().stroke([(10, 30), (130, 30)], pencils.B2, 1.0)
    assert _sheet().stroke([(10, 30), (130, 30)], pencils.B2, 5.0) == full
    assert _sheet().stroke([(10, 30), (130, 30)], pencils.B2, -1.0) == 0.0


def test_a_single_point_is_a_dot():
    sheet = _sheet(Paper(depth=0))
    laid = sheet.stroke([(70, 30)], pencils.B4, 1.0, 8.0)
    assert laid > 0
    ys, xs = np.nonzero(sheet.pigment())
    assert abs(xs.mean() - 69.5) < 0.6 and abs(ys.mean() - 29.5) < 0.6
    assert xs.max() - xs.min() <= 9 and ys.max() - ys.min() <= 9


@pytest.mark.parametrize(
    "kw, error, message",
    [
        ({"width": 0}, ValueError, "width must be"),
        ({"width": float("nan")}, ValueError, "width must be"),
        ({"pressure": float("nan")}, ValueError, "pressure must be finite"),
        ({"pressure": [0.2, float("inf")]}, ValueError, "pressure must be finite"),
        ({"medium": "HB"}, TypeError, "medium must be a Medium"),
        ({"medium": Pigment("#333333")}, TypeError, "medium must be a Medium"),
    ],
)
def test_a_stroke_checks_its_arguments_before_it_draws(kw, error, message):
    sheet = _sheet()
    arguments: dict[str, Any] = {"medium": pencils.HB, "pressure": 0.5, "width": 2.0}
    arguments.update(kw)
    with pytest.raises(error, match=message):
        sheet.stroke([(20, 30), (120, 30)], **arguments)
    assert not sheet.pigment().any()


# --------------------------------------------------------------------------
# pencils and media


def test_the_grades_run_from_hard_and_pale_to_soft_and_dark():
    assert len(pencils.GRADES) == 20
    assert pencils.GRADES[0] == "9H" and pencils.GRADES[-1] == "9B"
    leads = [pencils.grade(name) for name in pencils.GRADES]
    softness = [lead.softness for lead in leads]
    assert softness[0] == 0.0 and softness[-1] == 1.0
    assert all(a < b for a, b in pairwise(softness))

    def lum(lead) -> float:
        return float(sp.srgb_to_linear(lead.pigment.masstone) @ sp.LUMINANCE)

    assert all(lum(a) > lum(b) for a, b in pairwise(leads))
    assert pencils.HB.name == "HB pencil" and pencils.HB.pigment.name == "HB lead"


def test_grades_by_name():
    assert pencils.grade("6b").softness == pencils.B6.softness
    assert pencils.grade("2H").softness == pencils.H2.softness
    assert pencils.H.softness < pencils.F.softness < pencils.HB.softness
    assert pencils.HB.softness < pencils.B.softness < pencils.B2.softness
    with pytest.raises(ValueError, match="unknown pencil grade"):
        pencils.grade("10B")


def test_a_medium_checks_its_numbers():
    lead = pencils.HB.pigment
    with pytest.raises(ValueError, match="softness must be"):
        Medium(lead, softness=1.5)
    with pytest.raises(ValueError, match="binding must be"):
        Medium(lead, softness=0.5, binding=-0.1)
    with pytest.raises(TypeError, match="murex Pigment"):
        bad: Any = "#333333"
        Medium(bad, softness=0.5)
    softer = pencils.HB.with_(softness=0.9, name="soft HB")
    assert (softer.softness, softer.name, softer.pigment) == (0.9, "soft HB", lead)
    assert pencils.HB.softness != 0.9


def test_the_noise_of_a_pass_is_even_and_always_the_same():
    values = np.array([[unit_noise(x, y, 7) for x in range(64)] for y in range(64)])
    assert values.min() >= 0 and values.max() < 1
    assert abs(values.mean() - 0.5) < 0.02 and abs(values.std() - 12**-0.5) < 0.02
    assert unit_noise(3, 5, 7) == values[5, 3]
    other = np.array([[unit_noise(x, y, 8) for x in range(64)] for y in range(64)])
    assert abs(np.corrcoef(values.ravel(), other.ravel())[0, 1]) < 0.05
    # neighbours are unrelated
    assert abs(np.corrcoef(values[:, 1:].ravel(), values[:, :-1].ravel())[0, 1]) < 0.05
