"""The geometry of a stroke: how strong it is at its ends, round or flat
ends, and a width and a tilt that change along it, measured."""

import time
from typing import Any

import numpy as np
import pytest
from murex import Polyline, spline

from fusain import Paper, Sheet, Stick, charcoal, papers, pastels, pencils, pens

LEVEL = Paper(depth=0)  # touched all over: the mark shows what the tip lays
PASTEL = pastels.soft("#203050")


def _sheet(paper=LEVEL, seed=1) -> Sheet:
    return Sheet(320, 160, paper=paper, seed=seed)


def _along(sheet, half=3) -> np.ndarray:
    """What lies along the middle of a stroke drawn along y = 80, per
    column."""
    return sheet.pigment()[80 - half : 80 + half].astype(np.float64).mean(axis=0)


def _laid_is_on_the_sheet(sheet, laid):
    assert laid > 0
    assert sheet.pigment().sum(dtype=np.float64) == pytest.approx(laid, rel=1e-5)


# --------------------------------------------------------------------------
# the ends of a stroke


@pytest.mark.parametrize(
    "medium, width", [(pencils.HB, 8.0), (PASTEL, 24.0), (charcoal.VINE, 14.0)]
)
def test_a_stroke_is_as_strong_at_its_ends_as_in_its_middle(medium, width):
    # the tip used to lay only as it slid on, so the paper under it where it
    # landed or lifted got from nothing (at its back) to a full pass (at
    # its front): half of one at the ends of the path, falling off over a
    # whole width of the tip
    sheet = _sheet()
    laid = sheet.stroke([(40, 80), (240, 80)], medium, 0.6, width)
    along = _along(sheet)
    middle = along[120:160].mean()
    for x in (40, 44, 236, 239):
        assert along[x] == pytest.approx(middle, rel=0.1)
    # round: out to the outline of the tip
    assert along[240 + int(width / 2) - 2] > 0.7 * middle
    assert along[240 + int(width / 2) + 2] == 0
    _laid_is_on_the_sheet(sheet, laid)


@pytest.mark.parametrize("paper", [LEVEL, papers.CARTRIDGE])
def test_a_short_stroke_is_as_strong_as_a_long_one(paper):
    # a short stroke of a broad tip (a head, a hand) used to reach a third of
    # a full pass, and a dot a fifth
    def peak(ends) -> float:
        total = np.zeros(320)
        for seed in range(4 if paper is LEVEL else 12):
            sheet = _sheet(paper, seed)
            sheet.stroke(ends, PASTEL, 0.6, 24.0)
            total += sheet.pigment()[74:86].mean(axis=0)
        return float(total.max())

    full = peak([(40, 80), (280, 80)])
    for ends in ([(100, 80)], [(100, 80), (106, 80)], [(100, 80), (112, 80)]):
        assert peak(ends) == pytest.approx(full, rel=0.08)


def test_a_dot_lays_a_pass_on_the_paper_under_the_tip():
    sheet = _sheet()
    laid = sheet.stroke([(100, 80)], pencils.B4, 0.6, 10.0)
    stroke = _sheet()
    stroke.stroke([(40, 80), (240, 80)], pencils.B4, 0.6, 10.0)
    assert sheet.pigment()[78:82, 98:102].mean() == pytest.approx(
        stroke.pigment()[78:82, 138:142].mean(), rel=0.1
    )
    ys, xs = np.nonzero(sheet.pigment())
    assert xs.max() - xs.min() <= 11 and ys.max() - ys.min() <= 11
    _laid_is_on_the_sheet(sheet, laid)


def test_a_flat_end_is_cut_square_at_the_end_of_the_path():
    sheet = _sheet()
    laid = sheet.stroke([(40, 80), (240, 80)], PASTEL, 0.6, 24.0, ends="flat")
    marked = sheet.pigment() > 0
    # as long as its path, all across
    cols = np.nonzero(marked.any(axis=0))[0]
    assert cols.min() == 40 and cols.max() == 239
    for row in range(70, 91):
        on = np.nonzero(marked[row])[0]
        assert on.min() == 40 and on.max() == 239
    along = _along(sheet, 10)
    middle = along[120:160].mean()
    assert along[40] == pytest.approx(middle, rel=0.1)
    assert along[239] == pytest.approx(middle, rel=0.1)
    _laid_is_on_the_sheet(sheet, laid)
    # flat at one end and round at the other
    sheet = _sheet()
    sheet.stroke([(40, 80), (240, 80)], PASTEL, 0.6, 24.0, ends=("flat", "round"))
    cols = np.nonzero((sheet.pigment() > 0).any(axis=0))[0]
    assert cols.min() == 40 and 250 <= cols.max() <= 253


def test_a_flat_end_follows_the_way_the_stroke_goes():
    # cut square across the stroke, whichever way it runs
    sheet = _sheet()
    sheet.stroke([(60, 40), (160, 140)], pencils.B4, 0.7, 16.0, ends="flat")
    ys, xs = np.nonzero(sheet.pigment())
    along = ((xs + 0.5 - 60) + (ys + 0.5 - 40)) / np.sqrt(2.0)
    assert along.min() > -0.75 and along.max() < 100 * np.sqrt(2.0) + 0.75


def test_a_flat_end_cuts_only_near_its_end():
    # a path that comes back across the line through its start: the cut
    # there is for the start, not for the stroke that comes back past it
    path = Polyline(np.array([(40, 40), (60, 40), (60, 80), (20, 80), (20, 100)]))
    flat, round_ = _sheet(), _sheet()
    laid = flat.stroke(path, PASTEL, 0.6, 6.0, ends="flat")
    round_.stroke(path, PASTEL, 0.6, 6.0)
    back = (slice(76, 97), slice(17, 39))  # (the stroke along y = 80 and down)
    assert flat.pigment()[back].sum() == pytest.approx(
        round_.pigment()[back].sum(), rel=0.05
    )
    # and the start is still cut square
    assert not flat.pigment()[34:47, :40].any()
    _laid_is_on_the_sheet(flat, laid)


def _hooked(leg: float, width: float = 24.0) -> tuple[np.ndarray, float]:
    """What a flat stroke along y = 80 leaves that turns square up at x = 200
    for ``leg`` pixels, and what a pass of it lays under its middle."""
    sheet = _sheet()
    path = Polyline(np.array([(60.0, 80.0), (200.0, 80.0), (200.0, 80.0 - leg)]))
    laid = sheet.stroke(path, PASTEL, 0.6, width, ends="flat")
    _laid_is_on_the_sheet(sheet, laid)
    drawn = sheet.pigment().astype(np.float64)
    return drawn, float(drawn[78:82, 100:140].mean())


@pytest.mark.parametrize("leg", [12.0, 16.0, 40.0])
def test_a_flat_end_after_a_hard_bend_is_cut_square_across_the_last_leg(leg):
    # the path turns a square corner and ends, half a tip or more further on
    drawn, full = _hooked(leg)
    end = int(80 - leg)
    assert not drawn[:end].any()  # nothing past the line across its end
    # and full strength up to it, over the middle half of the stroke
    assert drawn[end : end + 2, 194:206].mean() == pytest.approx(full, rel=0.1)


def test_a_flat_end_is_cut_square_where_the_path_curves_sharply_into_it():
    # a smooth path that bends through a right angle over its last tip's
    # length: the cut is across the way the stroke is going where it ends
    for width in (24.0, 12.0):
        sheet = _sheet()
        path = spline.as_path([(60, 80), (170, 80), (196, 72), (204, 48)])
        points = path.sample(0.5)[0]
        going = points[-1] - points[-2]
        going /= np.hypot(*going)
        assert going[1] < -0.9  # (upwards by then)
        laid = sheet.stroke(path, PASTEL, 0.6, width, ends="flat")
        _laid_is_on_the_sheet(sheet, laid)
        drawn = sheet.pigment().astype(np.float64)
        ys, xs = np.nonzero(drawn)
        off = np.stack([xs + 0.5 - points[-1][0], ys + 0.5 - points[-1][1]])
        ahead, aside = going @ off, np.array([-going[1], going[0]]) @ off
        assert ahead.max() < 0.75
        last = (ahead > -2.5) & (ahead < -0.5) & (np.abs(aside) < width / 4.0)
        full = float(drawn[78:82, 100:140].mean())
        assert drawn[ys[last], xs[last]].mean() == pytest.approx(full, rel=0.12)


@pytest.mark.xfail(
    reason="a flat end on a hook shorter than half the tip cuts into the stroke"
)
@pytest.mark.parametrize("leg", [2.0, 4.0, 6.0])
def test_a_flat_end_on_a_short_hook_leaves_the_stroke_before_it_whole(leg):
    # the path ends a few pixels after a square corner: the cut is taken
    # across those few pixels, and the tip is cut by it for a whole tip's
    # length before the end, so the stroke before the corner loses part of
    # the half of it that lies past that line (it should be as strong as
    # the other half, or stronger, being on the inside of the bend)
    drawn, _ = _hooked(leg)
    before = slice(188, 196)  # 4 to 12 pixels before the corner
    inside, outside = drawn[70:78, before].mean(), drawn[82:90, before].mean()
    assert inside > 0.95 * outside


@pytest.mark.parametrize("length", [1e-3, 1e-6])
def test_a_stroke_of_almost_no_length_is_a_dot(length):
    dot = _sheet()
    alone = dot.stroke([(100, 80)], PASTEL, 0.6, 24.0)
    sheet = _sheet()
    start = time.perf_counter()
    laid = sheet.stroke([(100, 80), (100 + length, 80)], PASTEL, 0.6, 24.0)
    # (it used to step as finely as its two points are apart where it
    # landed and lifted: millions of steps for a dab)
    assert time.perf_counter() - start < 0.5
    assert laid == pytest.approx(alone, rel=0.05)
    _laid_is_on_the_sheet(sheet, laid)


def test_flat_ends_shade_inside_the_area():
    def outside(ends) -> float:
        sheet = _sheet(papers.CARTRIDGE)
        sheet.shade((60, 40, 260, 120), PASTEL, 0.6, 12.0, 0.0, hand=0.0, ends=ends)
        drawn = sheet.pigment().astype(np.float64)
        return float(drawn[:, :60].sum() + drawn[:, 260:].sum()) / drawn.sum()

    # a round end reaches half the tip past the end of its stroke: the
    # strokes of a shading stop that much short of the edge
    assert outside("flat") == 0 and outside("round") < 0.003


# --------------------------------------------------------------------------
# a width and a tilt that change along the stroke


def test_a_stroke_can_taper():
    sheet = _sheet()
    laid = sheet.stroke([(40, 80), (280, 80)], PASTEL, 0.6, lambda u: 24.0 - 20.0 * u)
    marked = sheet.pigment() > 0
    for x in (60, 100, 140, 180, 220, 260):
        wide = 24.0 - 20.0 * (x - 40) / 240
        # (the rim of the tip reaches half a pixel past it either side)
        assert abs(int(np.count_nonzero(marked[:, x])) - (wide + 1.0)) <= 1.0
    # what it lays on the paper under it does not change with the width
    along = _along(sheet, 1)
    assert along[60:250].min() > 0.85 * along[60:250].max()
    _laid_is_on_the_sheet(sheet, laid)


def test_a_width_per_point_of_the_path():
    sheet = _sheet()
    sheet.stroke([(40, 80), (160, 80), (280, 80)], pencils.B4, 0.6, [4.0, 20.0, 4.0])
    rows = np.count_nonzero(sheet.pigment() > 0, axis=0)
    assert rows[160] >= 20 and rows[50] <= 6 and rows[270] <= 6
    # a width that does not change is the same as a number
    one, other = _sheet(papers.CARTRIDGE), _sheet(papers.CARTRIDGE)
    one.stroke([(40, 80), (160, 90), (280, 80)], pencils.B4, 0.6, 6.0)
    other.stroke([(40, 80), (160, 90), (280, 80)], pencils.B4, 0.6, [6.0] * 3)
    assert np.array_equal(one.pigment(), other.pigment())


def test_a_stick_turned_onto_its_side_as_it_goes():
    stick = Stick(charcoal.VINE, width=5, lead=5, side=60)
    sheet = _sheet()
    laid = sheet.stroke([(40, 80), (280, 80)], stick, 0.6, tilt=lambda u: u)
    rows = np.count_nonzero(sheet.pigment() > 0, axis=0)
    assert rows[50] <= 9 and 28 <= rows[160] <= 36 and rows[270] >= 55
    _laid_is_on_the_sheet(sheet, laid)
    assert stick.worn == pytest.approx(laid)


@pytest.mark.parametrize(
    "kw, error, message",
    [
        ({"width": lambda u: 4.0 - 8.0 * u}, ValueError, "width must be"),
        ({"width": [4.0, float("nan")]}, ValueError, "width must be"),
        ({"tilt": [0.0, 1.5]}, ValueError, "tilt must be between 0"),
        ({"tilt": lambda u: u - 1.0}, ValueError, "tilt must be between 0"),
        ({"ends": "square"}, ValueError, "ends must be"),
        ({"ends": ("flat",)}, ValueError, "ends must be"),
        ({"ends": ("flat", 1)}, ValueError, "ends must be"),
    ],
)
def test_a_stroke_checks_its_shape(kw, error, message):
    sheet = _sheet()
    arguments: dict[str, Any] = {"medium": pencils.HB, "pressure": 0.5, "width": 2.0}
    arguments.update(kw)
    with pytest.raises(error, match=message):
        sheet.stroke([(20, 30), (120, 30)], **arguments)
    with pytest.raises(ValueError, match="ends must be"):
        sheet.shade((10, 10, 100, 100), pencils.HB, ends="square")
    with pytest.raises(ValueError, match="ends round"):
        sheet.stroke([(20, 30), (120, 30)], pens.BLACK, ends="flat")
    with pytest.raises(ValueError, match="no side"):
        sheet.stroke([(20, 30), (120, 30)], pens.BLACK, tilt=[0.0, 1.0])
    assert not sheet.pigment().any()
