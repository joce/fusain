"""A stick: a medium with a tip that wears (behaviour 6 of docs/design.md)."""

import math
from typing import Any

import numpy as np
import pytest

from fusain import (
    Medium,
    Paper,
    Sheet,
    Stick,
    charcoal,
    crayons,
    papers,
    pastels,
    pencils,
)
from fusain import _engine as eng

LEVEL = Paper(depth=0)  # the width of a line shows exactly


def _rows(sheet, x) -> int:
    """How many rows the line marks around column ``x``: its width there."""
    return int(np.count_nonzero(sheet.pigment()[:, x : x + 30].sum(axis=1)))


def _line(medium, length=1500, pressure=0.6, **kw) -> tuple[Sheet, Stick]:
    sheet = Sheet(length + 100, 60, paper=LEVEL, seed=1)
    stick = Stick(medium, **kw)
    sheet.stroke([(50, 30), (50 + length, 30)], stick, pressure)
    return sheet, stick


def test_a_line_widens_as_its_point_wears():
    sheet, stick = _line(pencils.B2, width=3, lead=20)
    start, middle, end = _rows(sheet, 60), _rows(sheet, 780), _rows(sheet, 1500)
    assert start <= middle <= end and end >= start + 2
    assert 3.0 < stick.width < 20 and abs(end - 1 - stick.width) <= 1.5
    # a bare medium keeps its width all along
    plain = Sheet(1600, 60, paper=LEVEL, seed=1)
    plain.stroke([(50, 30), (1550, 30)], pencils.B2, 0.6, 3.0)
    assert _rows(plain, 60) == _rows(plain, 1500) == 4


def test_a_soft_lead_wears_faster_than_a_hard_one():
    widths = [
        _line(pencils.grade(g), width=3, lead=20)[1].width for g in ("2H", "HB", "6B")
    ]
    assert 3.0 < widths[0] < widths[1] < widths[2]
    # and pressing harder wears it faster
    light = _line(pencils.B2, pressure=0.3, width=3, lead=20)[1].width
    firm = _line(pencils.B2, pressure=0.9, width=3, lead=20)[1].width
    assert firm > light + 0.3


def test_what_it_wears_is_what_it_lays():
    sheet = Sheet(400, 200, seed=1)
    stick = Stick(pencils.B4, width=3)
    laid = sheet.stroke([(20, 30), (380, 30)], stick, 0.7)
    assert stick.worn == pytest.approx(laid) and laid > 0
    laid += sheet.shade((20, 60, 380, 180), stick, 0.5)
    assert stick.worn == pytest.approx(laid)
    np.testing.assert_allclose(sheet.pigment().sum(dtype=np.float64), laid, rtol=1e-4)
    # the width follows from it: the cube of the width grows with the volume
    grown = stick.width**3 - stick.point**3
    assert grown == pytest.approx(0.08 * stick.worn, rel=1e-6)


def test_the_next_stroke_starts_as_wide_as_the_last_ended():
    sheet = Sheet(1600, 120, paper=LEVEL, seed=1)
    stick = Stick(pencils.B6, width=3, lead=20)
    sheet.stroke([(50, 30), (1550, 30)], stick, 0.6)
    ended = _rows(sheet, 1500)
    sheet.stroke([(50, 90), (400, 90)], stick, 0.6)
    below = sheet.pigment()[60:]
    starts = int(np.count_nonzero(below[:, 60:90].sum(axis=1)))
    assert abs(starts - ended) <= 1


def test_sharpening_gives_the_point_back():
    _, stick = _line(pencils.B6, width=3, lead=20)
    assert stick.width > 5 and stick.worn > 0
    stick.sharpen()
    assert stick.width == 3 and stick.worn == 0
    stick.sharpen(2)  # to a finer point
    assert stick.width == stick.point == 2
    fresh = Sheet(300, 60, paper=LEVEL, seed=1)
    fresh.stroke([(50, 30), (200, 30)], stick, 0.6)
    assert _rows(fresh, 60) <= 4  # as fine as a point 2 wide draws
    with pytest.raises(ValueError, match="wider than the lead"):
        stick.sharpen(25)


def test_a_tip_never_gets_wider_than_the_lead():
    sheet, stick = _line(pencils.B8, length=3000, pressure=1.0, width=3, lead=5)
    assert stick.width == 5 and _rows(sheet, 3000) <= 6
    # a stick of charcoal is as wide as it is from the start
    sheet, stick = _line(charcoal.VINE, width=10, lead=10)
    assert stick.width == 10 and 10 <= _rows(sheet, 60) == _rows(sheet, 1500) <= 11
    assert Stick(pencils.HB, width=2).lead == 12  # six times the point


def test_shading_with_a_stick_wears_it_and_fills_the_area():
    sheet = Sheet(300, 200, seed=2)
    stick = Stick(pencils.B4, width=3)
    laid = sheet.shade((30, 30, 270, 170), stick, 0.6)
    assert laid > 0 and stick.width > 3.3
    assert (sheet.pigment()[60:140, 60:240] > 0).mean() > 0.6
    # tilted, its side marks a band five times its width now
    stick.sharpen()
    wide = Sheet(300, 100, paper=LEVEL, seed=2)
    wide.stroke([(30, 50), (60, 50)], stick, 0.6, tilt=1.0)
    assert 14 <= int(np.count_nonzero(wide.pigment()[:, 35:55].sum(axis=1))) <= 18


def _band(sheet: Sheet) -> int:
    """How many rows a stroke across the middle of the sheet marks."""
    middle = sheet.width // 2
    return int(np.count_nonzero(sheet.pigment()[:, middle - 10 : middle + 10].sum(1)))


def _side_stroke(medium, width=None, tilt=1.0) -> Sheet:
    sheet = Sheet(300, 200, paper=LEVEL, seed=4)
    sheet.stroke([(60, 100), (240, 100)], medium, 0.6, width, tilt)
    return sheet


def test_the_side_of_a_stick_is_as_long_as_its_medium_says():
    # a pencil's side is the lead its sharpening bared: five times its point
    assert pencils.HB.side == 5 and charcoal.VINE.side > pastels.soft("#c0392b").side
    assert 19 <= _band(_side_stroke(pencils.B2, 4.0)) <= 23
    # a piece of vine charcoal laid flat marks a band eight times its end
    assert 31 <= _band(_side_stroke(charcoal.VINE, 4.0)) <= 35
    # half way over, half of what the side adds
    assert 18 <= _band(_side_stroke(charcoal.VINE, 4.0, 0.5)) <= 21
    # a stick in the hand says how long its side is, in pixels
    stick = Stick(charcoal.VINE, width=6, lead=6, side=120)
    assert stick.side == 120 and "side=120" in repr(stick)
    sheet = Sheet(300, 200, paper=LEVEL, seed=4)
    laid = sheet.stroke([(60, 100), (240, 100)], stick, 0.6, tilt=1.0)
    assert 119 <= _band(sheet) <= 123
    assert laid == pytest.approx(sheet.pigment().sum(dtype=np.float64), rel=1e-4)
    # by default its medium's side times its point, whatever it has worn
    assert Stick(charcoal.VINE, width=5, lead=5).side == 40
    worn = Stick(pencils.B8, width=2, lead=40)
    worn.worn = 20000.0
    assert worn.width > worn.side == 10
    assert _band(_side_stroke(worn)) == _band(_side_stroke(worn, tilt=0.0))


def test_a_side_is_checked():
    with pytest.raises(ValueError, match="side must be at least 1"):
        Medium(pencils.HB.pigment, 0.5, side=0.5)
    with pytest.raises(ValueError, match="side must be at least 1"):
        pencils.HB.with_(side=float("nan"))
    with pytest.raises(ValueError, match="side must be"):
        Stick(charcoal.VINE, width=6, lead=6, side=-1)
    with pytest.raises(ValueError, match="at least its point"):
        Stick(charcoal.VINE, width=6, lead=6, side=4)
    stick = Stick(charcoal.VINE, width=4, lead=12, side=10)
    with pytest.raises(ValueError, match="wider than the side"):
        stick.sharpen(11)


def test_a_stick_checks_what_it_is_given():
    with pytest.raises(TypeError, match="medium must be a Medium"):
        wrong: Any = "HB"
        Stick(wrong)
    with pytest.raises(ValueError, match="width must be"):
        Stick(pencils.HB, width=0)
    with pytest.raises(ValueError, match="lead must be"):
        Stick(pencils.HB, width=2, lead=float("nan"))
    with pytest.raises(ValueError, match="at least its point"):
        Stick(pencils.HB, width=4, lead=3)
    stick, sheet = Stick(pencils.HB, width=2), Sheet(100, 60, seed=1)
    for draw in (
        lambda: sheet.stroke([(10, 30), (90, 30)], stick, 0.5, 4.0),
        lambda: sheet.shade((10, 10, 90, 50), stick, 0.5, 4.0),
    ):
        with pytest.raises(TypeError, match="a Stick has its own width"):
            draw()
    assert not sheet.pigment().any() and stick.worn == 0
    assert repr(stick) == "Stick(HB pencil, width=2.00, lead=12, side=10)"


# --------------------------------------------------------------------------
# the side of a stick laid flat (fusain-bqe)


def _across(sheet, columns=slice(120, 280)) -> np.ndarray:
    """What lies across a band drawn along y = 100, per row, smoothed over
    five rows (the streaks of a lead run along it), as a share of its
    middle."""
    raw = sheet.pigment()[:, columns].astype(np.float64).mean(axis=1)
    smooth = np.convolve(raw, np.ones(5) / 5, "same")
    return smooth / smooth[82:118].mean()


def _flat_and_edge(across: np.ndarray, side=90) -> tuple[float, float]:
    """The share of the band (``side`` rows about row 100) that has 0.9 of
    its middle or more, and over how many rows it falls from there to 0.1,
    either side on average."""
    rows = np.arange(across.size) - 100
    edges = []
    for way in (-1, 1):
        out = rows * way > 0
        full = (rows * way)[out & (across >= 0.9)].max()
        edges.append((rows * way)[out & (across >= 0.1)].max() - full)
    return float((across >= 0.9).sum() / side), float(np.mean(edges))


@pytest.mark.parametrize(
    "medium", [charcoal.VINE, charcoal.COMPRESSED, pastels.soft("#203050")]
)
def test_the_side_of_a_stick_laid_flat_lays_an_even_band(medium):
    # (its footprint was an ellipse: the band fell off from its middle, 0.43
    # of it at 0.9 of the middle or more, and over 23.5 rows to 0.1 at
    # either edge)
    assert medium.flat_side
    sheet = Sheet(400, 200, paper=LEVEL, seed=3)
    stick = Stick(medium, width=6, lead=6, side=90)
    laid = sheet.stroke([(60, 100), (340, 100)], stick, 0.6, tilt=1.0)
    flat, edge = _flat_and_edge(_across(sheet))
    assert flat > 0.85 and edge < 6.0
    assert laid == pytest.approx(sheet.pigment().sum(dtype=np.float64), rel=1e-5)
    # on a paper with a tooth too (0.28 to 0.31 before)
    rough = Sheet(400, 200, paper=papers.CARTRIDGE, seed=3)
    rough.stroke([(60, 100), (340, 100)], Stick(medium, 6, 6, 90), 0.6, tilt=1.0)
    assert _flat_and_edge(_across(rough))[0] > 0.6


def test_the_side_of_a_pencil_is_a_cone_and_its_band_fades():
    assert not pencils.B2.flat_side and not charcoal.PENCIL.flat_side
    assert pastels.hard("#203050").flat_side and crayons.wax("#c0392b").flat_side
    sheet = Sheet(400, 200, paper=LEVEL, seed=3)
    sheet.stroke([(60, 100), (340, 100)], pencils.B2, 0.6, 18.0, 1.0)
    flat, edge = _flat_and_edge(_across(sheet))
    assert flat < 0.5 and edge > 15.0
    # the same lead as a stick laid flat would lay an even band
    stick = pencils.B2.with_(flat_side=True)
    even = Sheet(400, 200, paper=LEVEL, seed=3)
    even.stroke([(60, 100), (340, 100)], stick, 0.6, 18.0, 1.0)
    assert _flat_and_edge(_across(even))[0] > 0.85
    with pytest.raises(TypeError, match="flat_side must be True or False"):
        pencils.B2.with_(flat_side=1)


def test_a_band_laid_flat_starts_and_stops_square():
    """The side of a stick lands and lifts all along its length at once."""
    sheet = Sheet(400, 200, paper=LEVEL, seed=3)
    sheet.stroke([(100, 100), (300, 100)], Stick(charcoal.VINE, 6, 6, 90), 0.6, tilt=1)
    marked = sheet.pigment() > 0
    starts = [int(np.argmax(marked[y])) for y in range(58, 142)]
    stops = [int(marked.shape[1] - np.argmax(marked[y, ::-1])) for y in range(58, 142)]
    # (straight across, within a pixel, but within its thickness of the ends
    # of the band, where it is rounded)
    assert max(starts) - min(starts) <= 1 and max(stops) - min(stops) <= 1
    assert min(starts) >= 95 and max(stops) <= 305


def test_a_band_laid_flat_is_as_even_where_the_stroke_turns():
    """Where the stroke turns, the side of the stick on the outside of the
    turn slides further than its middle, and lays as far as it slides."""
    sheet = Sheet(400, 400, paper=LEVEL, seed=3)
    stick = Stick(charcoal.VINE, width=6, lead=6, side=90)
    angles = np.linspace(0.0, 0.5 * np.pi, 40)
    path = [(100 + 150 * np.cos(a), 100 + 150 * np.sin(a)) for a in angles]
    laid = sheet.stroke(path, stick, 0.6, tilt=1.0)
    yy, xx = np.mgrid[0:400, 0:400]
    radius = np.hypot(xx + 0.5 - 100, yy + 0.5 - 100)
    angle = np.arctan2(yy + 0.5 - 100, xx + 0.5 - 100)
    away = (angle > 0.35) & (angle < 1.2)  # (from its ends)
    pigment = sheet.pigment().astype(np.float64)

    def within(low, high) -> float:
        return float(pigment[away & (radius >= low) & (radius < high)].mean())

    middle = within(140, 160)
    # (were each part to lay as far as the middle went, the inside would
    # have 1.10 of the middle and the outside 0.84; now 0.95 and 0.96)
    assert within(110, 130) == pytest.approx(middle, rel=0.1)
    assert within(170, 190) == pytest.approx(middle, rel=0.1)
    assert laid == pytest.approx(pigment.sum(), rel=1e-5)


#: Paths that turn as sharply as a path can: doubling back a little off the
#: line, exactly on it, with a needle-sharp hairpin, and in a zigzag.
SHARP = {
    "cusp": [(100, 150), (200, 150), (100, 150.5)],
    "reversal": [(100, 150), (200, 150), (100, 150)],
    "needle": [(100, 150), (200, 150), (200.2, 150.2), (100, 150.4)],
    "zigzag": [(60, 100), (110, 200), (160, 100), (210, 200), (260, 100), (310, 200)],
}


@pytest.mark.parametrize("film", [False, True], ids=["over dust", "over a film"])
@pytest.mark.parametrize("path", SHARP.values(), ids=SHARP.keys())
@pytest.mark.parametrize(
    "medium, width",
    [
        (charcoal.COMPRESSED, 6.0),
        (pencils.B2, 6.0),
        (Stick(pastels.soft("#203050"), 8, 8, 60), None),
    ],
    ids=["compressed charcoal", "a pencil", "a pastel"],
)
def test_the_side_of_a_tip_turned_sharply_leaves_nothing_below_nothing(
    medium, width, path, film
):
    # (a part of the tip was laid as far as it slid in a step, which at a
    # cusp is tens of times a pass: it pressed in more dust than there was,
    # and left the dust as low as -10)
    sheet = Sheet(400, 300, seed=1)
    sheet.shade((20, 20, 380, 280), pastels.soft("#e0c080"), 0.7, width=12)
    if film:
        sheet.shade((20, 20, 380, 280), crayons.wax("#2456a6"), 0.8, width=10)
    before = sheet.pigment().sum(dtype=np.float64)
    laid = sheet.stroke(path, medium, 1.0, width, tilt=1.0)
    for name in ("held", "under", "loose", "bonded", "cover", "crush", "polish"):
        assert getattr(sheet, name).min() > -1e-6, name
    assert sheet.pigment().sum(dtype=np.float64) == pytest.approx(before + laid)


def test_a_stick_drawn_to_and_fro_does_not_turn_round():
    """A tip is the same turned half round: where a stroke doubles back, the
    hand draws it back, and its band is even up to where it turns."""
    assert eng.turning(np.array([0.0, 5.0, 0.0]), np.zeros(3), 1) == 0.0
    sheet = Sheet(400, 300, paper=LEVEL, seed=1)
    stick = Stick(charcoal.VINE, 6, 6, 60)
    laid = sheet.stroke([(100, 150), (250, 150), (100, 150)], stick, 0.6, tilt=1.0)
    along = sheet.pigment()[125:175].astype(np.float64).mean(axis=0)
    # (it piled up where it turned: 0.71 at most, against 0.51 out and back)
    assert along[150:240].min() > 0.95 * along[150:240].max()
    assert sheet.pigment().max() < 1.2 * along[150:240].max()
    assert not along[254:].any()  # and it ends within its thickness
    assert laid == pytest.approx(sheet.pigment().sum(dtype=np.float64), rel=1e-5)


def test_a_shading_lays_the_bands_of_a_stick_laid_flat_side_by_side(monkeypatch):
    def gaps(medium) -> list[float]:
        sheet, centres = Sheet(300, 300, seed=2), []
        draw = sheet.stroke

        def recording(path, medium, pressure, width, tilt, ends, **mark):
            centres.append(float(np.mean([p[1] for p in path])))
            return draw(path, medium, pressure, width, tilt, ends, **mark)

        monkeypatch.setattr(sheet, "stroke", recording)
        sheet.shade((20, 20, 280, 280), medium, 0.6, 6.0, 0.0, hand=0.0, tilt=1.0)
        return list(np.diff(centres))

    # its bands abut, and only their rounded ends overlap, as a point's
    # strokes do: 30 - 6 + 0.55 * 6 apart; a pencil's soft bands half
    assert np.allclose(gaps(pastels.soft("#203050")), 27.3)
    assert np.allclose(gaps(pencils.B2), 0.55 * 30)


def test_a_hand_lays_each_band_of_a_flat_stick_beside_the_last(monkeypatch):
    def gaps(medium) -> np.ndarray:
        sheet, centres = Sheet(300, 300, seed=2), []
        draw = sheet.stroke

        def recording(path, medium, pressure, width, tilt, ends, **mark):
            centres.append(float(np.mean([p[1] for p in path])))
            return draw(path, medium, pressure, width, tilt, ends, **mark)

        monkeypatch.setattr(sheet, "stroke", recording)
        sheet.shade((20, 20, 280, 280), medium, 0.6, 6.0, 0.0, tilt=1.0)
        return np.diff(centres)

    # bands that meet by under half the stick's thickness stray by a share
    # of that thickness, not of their length, or they part and leave a
    # streak of bare paper between (a natural hand left lines of +10 to +20
    # in L* where soft pastel's bands met, +2.5 at most now); a pencil's
    # soft, overlapping strokes stray by a share of their mark
    flat, pencil = gaps(pastels.soft("#203050")), gaps(pencils.B2)
    assert flat.mean() == pytest.approx(27.3, abs=0.3)
    assert flat.std() < 0.4 * 6.0 < 4.0 * 6.0 * 0.15 < pencil.std()


@pytest.mark.parametrize("seed", range(3))
def test_the_span_of_a_row_misses_no_pixel_under_the_side_of_a_stick(seed):
    rng = np.random.default_rng(seed)
    now = np.zeros(15)
    now[13] = 1.0
    covered, spanned, square = 0, 0, 0
    for _ in range(100):
        long = rng.choice([0.6, 1.5, 3.5, 6.0]) * rng.uniform(0.9, 1.1)
        wide = long + rng.choice([0.0, 2.0, 20.0, 60.0]) * rng.uniform(0.9, 1.1)
        turn = rng.choice([0.0, 0.5 * math.pi, rng.uniform(0.0, 2.0 * math.pi)])
        x, y = rng.uniform(-3.0, 3.0, 2)
        now[:7] = [x, y, math.cos(turn), math.sin(turn), wide, long, 0.35]
        x0, x1 = math.floor(x - wide), math.ceil(x + wide)
        for iy in range(math.floor(y - wide), math.ceil(y + wide)):
            dy = iy + 0.5 - y
            xa, xb = eng.span_of(now, dy, 0.0, 1.0, 1.0, x0, x1)
            assert (x0 <= xa and xb <= x1) or xa == xb  # (or none)
            spanned, square = spanned + max(xb - xa, 0), square + x1 - x0
            for ix in range(x0, x1):
                if eng.bearing_at(ix + 0.5 - x, dy, now)[1] > 0.0:
                    assert xa <= ix < xb, (wide, long, turn, ix, iy)
                    covered += 1
    assert 5000 < covered <= spanned < 0.6 * square
