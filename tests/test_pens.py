"""Ballpoint pens: an even line at any pressure, ink that layers and that no
rubber eraser takes off (a knife or a sand eraser does, with the paper),
measured."""

import itertools

import numpy as np
import pytest
from murex import Pigment, Polyline
from murex import spectral as sp

from fusain import (
    HeldPen,
    Pen,
    Sheet,
    Smudger,
    crayons,
    papers,
    pastels,
    pencils,
    pens,
    tools,
)
from fusain import _engine as eng
from fusain.sheet import IV

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
LINE = [(20, 20), (280, 20)]
PRESSURES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def _luminance(sheet) -> np.ndarray:
    spectrum = sheet.reflectance().astype(np.float64)
    return sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE


def _tone(sheet, where=INSIDE) -> float:
    """Luminance of the area, as a share of the bare paper's."""
    white = float(_luminance(Sheet(8, 8, paper=sheet.paper, seed=1)).mean())
    return float(_luminance(sheet)[where].mean()) / white


def _line(pressure, paper=papers.CARTRIDGE, pen=pens.BLUE) -> Sheet:
    sheet = Sheet(300, 40, paper=paper, seed=1)
    sheet.stroke(LINE, pen, pressure)
    return sheet


def _ink(sheet) -> float:
    assert sheet.ink is not None
    return float(sheet.ink[..., IV].sum(dtype=np.float64))


def _across(sheet) -> np.ndarray:
    """Tone across the line, one value per row."""
    white = float(_luminance(Sheet(8, 8, paper=sheet.paper, seed=1)).mean())
    return _luminance(sheet)[:, 60:240].mean(axis=1) / white


# --------------------------------------------------------------------------
# the pen


def test_presets_and_names():
    assert pens.BLUE.name == "blue ballpoint"
    assert pens.ballpoint("#22357e").name == "#22357e ballpoint"
    assert pens.ballpoint("#22357e", width=5).width == 5
    assert pens.BLACK.with_(flow=0.5).flow == 0.5
    assert isinstance(pens.RED, Pen)


def test_a_pen_checks_what_it_is_given():
    with pytest.raises(TypeError, match="ink"):
        Pen("#22357e")  # type: ignore[arg-type]
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="width"):
            Pen(Pigment("#22357e"), width=bad)
        with pytest.raises(ValueError, match="flow"):
            Pen(Pigment("#22357e"), flow=bad)
    for bad in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="blots"):
            Pen(Pigment("#22357e"), blots=bad)
    assert Pen(Pigment("#22357e"), blots=0.0).blots == 0.0


def test_a_pen_has_its_own_width_and_no_side():
    sheet = Sheet(60, 40, seed=1)
    with pytest.raises(TypeError, match="own width"):
        sheet.stroke([(5, 5), (50, 30)], pens.BLUE, width=4)
    with pytest.raises(TypeError, match="own width"):
        sheet.shade((5, 5, 50, 30), pens.BLUE, width=4)
    with pytest.raises(ValueError, match="no side"):
        sheet.stroke([(5, 5), (50, 30)], pens.BLUE, tilt=0.5)
    assert sheet.ink is None
    assert not sheet.pigment().any()


# --------------------------------------------------------------------------
# a line


def test_the_ink_is_a_stain_and_what_the_pen_laid():
    sheet = Sheet(300, 40, seed=1)
    assert sheet.ink is None and sheet.inked is None
    laid = sheet.stroke(LINE, pens.BLUE, 0.5)
    assert laid > 0
    assert _ink(sheet) == pytest.approx(laid, rel=1e-5)
    assert sheet.inked is not None and 0.0 < sheet.inked.max() <= 1.0
    # nothing lies on the paper: no pigment, no cover
    assert not sheet.pigment().any()
    assert not sheet.cover.any()
    # and it is where the pen went
    assert not sheet.inked[:10].any() and not sheet.inked[30:].any()
    assert not sheet.inked[:, :10].any()


@pytest.mark.parametrize("paper", [papers.CARTRIDGE, papers.PRINTER, papers.BRISTOL])
def test_a_light_line_is_still_a_line(paper):
    """Not specks on the peaks of the tooth, as a light pencil leaves."""
    for pressure in PRESSURES:
        sheet = _line(pressure, paper)
        assert sheet.inked is not None
        core = sheet.inked[19:21, 60:240].max(axis=0)
        assert (core > 0.5).mean() > 0.98, pressure


def test_a_light_line_is_paler_and_narrower():
    darkest, taken = [], []
    for pressure in PRESSURES:
        across = _across(_line(pressure))
        darkest.append(float(across.min()))
        taken.append(float((1.0 - across).sum()))
    assert all(a > b for a, b in itertools.pairwise(darkest))
    assert all(a < b for a, b in itertools.pairwise(taken))
    assert darkest[0] > 0.55 and darkest[-1] < 0.08
    # at full pressure the line is nearly as wide as the ball (3 pixels)
    assert 2.4 < taken[-1] < 3.0
    assert taken[0] < 0.4 * taken[-1]


def test_a_firm_line_is_the_colour_of_the_pen():
    for pen, colour in ((pens.BLUE, "#22357e"), (pens.RED, "#b3262b")):
        sheet = _line(1.0, papers.PRINTER, pen)
        spectrum = sheet.reflectance()[20, 60:240].astype(np.float64).mean(axis=0)
        got = sp.reflectance_to_srgb(spectrum)
        assert np.abs(got - sp.parse_color(colour)).max() < 0.08


def test_the_flow_scales_the_ink():
    dry = Sheet(300, 40, seed=1)
    wet = Sheet(300, 40, seed=1)
    a = dry.stroke(LINE, pens.BLUE.with_(flow=0.5), 0.6)
    b = wet.stroke(LINE, pens.BLUE, 0.6)
    assert a == pytest.approx(0.5 * b, rel=1e-6)


def test_a_line_is_as_strong_at_its_ends_as_in_its_middle():
    """Where the ball lands and where it lifts, the paper it meets gets
    what it would have had if the ball had rolled onto it and off it: the
    ends of a line are as strong as its middle, out to the round of the
    ball, and a dot is as dark as a line. (The ends faded over the width
    of the ball: half a pixel behind where it landed, 0.3 of the middle's
    ink; a dot had a tenth of a line's.)"""
    dry = pens.BLUE.with_(blots=0.0)
    for pressure in (0.2, 1.0):
        columns, dots, lines = [], [], []
        for seed in range(1, 5):
            sheet = Sheet(200, 40, seed=seed)
            sheet.stroke([(20, 20.0), (180, 20.0)], dry, pressure)
            assert sheet.ink is not None
            columns.append(sheet.ink[..., IV].sum(axis=0, dtype=np.float64))
            lines.append(sheet.ink[18:22, 60:140, IV].mean())
            dot = Sheet(40, 40, seed=seed)
            dot.stroke([(20.0, 20.0)], dry, pressure)
            assert dot.ink is not None
            dots.append(dot.ink[18:22, 18:22, IV].mean())
        ink = np.mean(columns, axis=0)
        ink /= ink[60:140].mean()
        # (the ball reaches two pixels from where it is: the line runs from
        # the pixel 18 to the pixel 181, round at its ends)
        assert ink[19:22].min() > 0.85 and ink[19:22].max() < 1.15, pressure
        assert ink[178:181].min() > 0.85 and ink[178:181].max() < 1.15, pressure
        assert min(ink[18], ink[181]) > 0.6
        assert ink[17] == 0.0 and ink[182] == 0.0
        assert np.mean(dots) > 0.85 * np.mean(lines), pressure


def test_the_ink_does_not_come_quite_evenly():
    sheet = _line(0.6, papers.PRINTER)
    assert sheet.ink is not None
    along = sheet.ink[20, 60:240, IV]
    assert 0.03 < along.std() / along.mean() < 0.2


# --------------------------------------------------------------------------
# blobs: the ball gathers ink, and leaves it where it lands and where it turns

#: The most ink the ball of a pen 3 wide gathers.
MOST = eng.BALL_BLOB * eng.BALL_INK * (0.5 * 3 + 0.5) ** 2


def _firm(pressure: float) -> float:
    """The share of a firm pass a pen lays at this pressure: also the share
    of the ink that comes round off its rim that the paper takes."""
    return eng.BALL_LIGHT + (1.0 - eng.BALL_LIGHT) * pressure


def _near(sheet, x, y, radius=4.0) -> float:
    """The ink within ``radius`` of a point."""
    assert sheet.ink is not None
    ys, xs = np.mgrid[: sheet.height, : sheet.width]
    disc = (xs + 0.5 - x) ** 2 + (ys + 0.5 - y) ** 2 <= radius**2
    return float(sheet.ink[..., IV][disc].sum(dtype=np.float64))


def _lands_with(sheet) -> float:
    """The share of the most it can gather that the next pen put to this
    sheet lands with (see :func:`fusain._engine.roll_stroke`)."""
    salt = int(sheet.copy()._rng.integers(1 << 31))
    return eng.BALL_START * eng.unit_noise(0, 2, salt + 3) ** eng.BALL_SELDOM


def _clean(width, height, seed=1) -> Sheet:
    """A sheet on which the next pen lands with next to nothing gathered."""
    for salt in range(seed, seed + 100):
        sheet = Sheet(width, height, seed=salt)
        if _lands_with(sheet) < 1e-3:
            return sheet
    raise AssertionError("no pen lands clean")


def _loaded() -> int:
    """The seed of a sheet (200 by 40) on which the next pen lands with a
    blob."""
    for seed in range(100):
        if _lands_with(Sheet(200, 40, seed=seed)) > 0.5:
            return seed
    raise AssertionError("no pen of a hundred lands with a blob")


def _thickest(sheet) -> float:
    assert sheet.ink is not None
    return float(sheet.ink[..., IV].max())


def test_a_line_starts_with_a_dot_of_ink_now_and_then():
    gathered, starts, extra = [], [], []
    line = [(20, 20.5), (180, 20.5)]
    for seed in range(60):
        sheet, plain = Sheet(200, 40, seed=seed), Sheet(200, 40, seed=seed)
        gathered.append(_lands_with(sheet))
        laid = sheet.stroke(line, pens.BLUE, 0.5)
        assert _ink(sheet) == pytest.approx(laid, rel=1e-5)  # blob and all
        # the ink where the line starts, and where it ends (nothing but line)
        start, end = _near(sheet, 20, 20.5, 5), _near(sheet, 180, 20.5, 5)
        starts.append(start / end)
        # and more than a pen that never blots lays there
        plain.stroke(line, pens.BLUE.with_(blots=0.0), 0.5)
        extra.append((start - _near(plain, 20, 20.5, 5)) / MOST)
    share, ratio = np.array(gathered), np.array(starts)
    # mostly next to nothing; one line in four or so starts with a dot that
    # shows; never more than a pen has on it
    assert np.median(share) < 0.05
    assert 0.15 < (share > 0.1).mean() < 0.4
    assert 0.3 < share.max() <= eng.BALL_START
    # and as much of that as the pen bears on the paper is the ink on the
    # paper (but for a drop too small to show, which stays on the ball)
    assert np.abs(np.array(extra) - _firm(0.5) * share).max() < 0.01
    assert ratio[share < 0.01].min() > 0.7 and ratio[share < 0.01].max() < 1.3
    assert ratio[share > 0.3].min() > 1.3


def test_a_blob_is_a_dot_a_little_wider_than_the_line_and_thickest_in_the_middle():
    sheet = Sheet(200, 40, seed=_loaded())
    sheet.stroke([(20.5, 20.5), (180, 20.5)], pens.BLUE, 0.3)
    assert sheet.ink is not None and sheet.inked is not None
    ink, inked = sheet.ink[..., IV], sheet.inked
    # across the line, at its start and half way along
    start, run = ink[:, 20], ink[:, 100]
    assert start.max() > 1.5 * run.max()
    assert np.argmax(start) == 20
    assert np.count_nonzero(start) > np.count_nonzero(run)
    assert np.count_nonzero(start) <= 2 * eng.BLOB_WIDE * (0.5 * 3 + 0.5) + 1
    # all of the pixels under its middle are inked; beyond the line, where
    # the paper had no ink, it inks a share of a pixel, the more the nearer
    # its middle
    assert inked[20, 19:22].min() == 1.0 and inked[19:22, 20].min() == 1.0
    assert 0.0 < inked[18, 19] < inked[18, 20] < 1.0


def _corner(run: float, seed: int, turn=(160.5, 90.5)) -> tuple[float, float]:
    """A line that runs ``run`` pixels and turns sharply, from a pen that
    landed clean. Returns the ink at the corner, and in as much line
    further on."""
    sheet = _clean(200, 120, seed)
    path = Polyline([(160.5 - run, 30.5), (160.5, 30.5), turn])
    laid = sheet.stroke(path, pens.BLUE, 0.5)
    assert _ink(sheet) == pytest.approx(laid, rel=1e-5)
    further = (0.5 * (160.5 + turn[0]), 0.5 * (30.5 + turn[1]))
    return _near(sheet, 160.5, 30.5), _near(sheet, *further)


def test_a_sharp_turn_leaves_a_blob_the_bigger_the_longer_the_line_before_it():
    blobs, extra = [], []
    for run in (10, 40, 140):
        corners = [_corner(run, seed) for seed in (1, 20, 40, 60)]
        blobs.append(float(np.mean([at / beyond for at, beyond in corners])))
        extra.append(float(np.mean([at - beyond for at, beyond in corners])))
    assert blobs[0] < 1.25  # (nothing gathered yet: the line, no more)
    assert blobs[0] < blobs[1] < blobs[2]
    assert blobs[2] > 1.3
    # what gathers over a run is the share BALL_GATHER of what the run lays
    # (about 0.045 a pixel), and a square turn brings most of it round, of
    # which the paper takes as much as the pen bears on it
    gathered = eng.BALL_GATHER * 140 * 0.045 * _firm(0.5)
    assert 0.4 * gathered < extra[2] < 0.9 * gathered


def test_a_line_that_turns_back_leaves_all_it_gathered():
    square = [_corner(140, seed) for seed in (1, 20, 40, 60)]
    back = [_corner(140, seed, (20.5, 50.5)) for seed in (1, 20, 40, 60)]
    extra = [float(np.mean([at - beyond for at, beyond in c])) for c in (square, back)]
    assert extra[1] > 1.2 * extra[0]
    assert extra[1] < eng.BALL_GATHER * 140 * 0.045 * _firm(0.5) * 1.15


def test_a_gentle_curve_leaves_no_blob():
    """The ink comes round a little at a time, and the line is as even as
    a straight one."""
    arc = [(100 + 60 * np.cos(a), 75 - 60 * np.sin(a)) for a in np.linspace(0, 3, 40)]
    for seed in (1, 20, 40):
        curved, straight = _clean(200, 100, seed), _clean(200, 100, seed)
        bent = curved.stroke(arc, pens.BLUE, 0.5)
        ruled = straight.stroke([(10, 50.5), (190, 50.5)], pens.BLUE, 0.5)
        assert _thickest(curved) < 1.1 * _thickest(straight)
        # what gathered and came round is in the line: hardly more ink than
        # in a straight one (the rest is still on the rim)
        assert 0.99 < bent / ruled < 1.0 + eng.BALL_GATHER


def test_a_gentle_curve_leaves_its_ink_in_a_few_drops():
    """Not a drop too small to show at every point of the line, which took
    a third of the pen's time: what comes round stays on the rim until it
    would show. (Every pixel of the line had some of a drop.)"""
    arc = [(100 + 60 * np.cos(a), 75 - 60 * np.sin(a)) for a in np.linspace(0, 3, 40)]
    for seed in (1, 20, 40):
        inks = []
        for blots in (0.0, 1.0):
            sheet = _clean(200, 100, seed)
            sheet.stroke(arc, pens.BLUE.with_(blots=blots), 0.5)
            assert sheet.ink is not None
            inks.append(sheet.ink[..., IV])
        line = inks[0] > 0.0
        reached = (inks[1][line] != inks[0][line]).mean()
        assert 0.3 < reached < 0.8


def test_a_freer_pen_and_a_bigger_ball_gather_more():
    seed = _loaded()

    def blob(pen) -> float:
        sheet = Sheet(200, 40, seed=seed)
        sheet.stroke([(20, 20.5), (180, 20.5)], pen, 0.5)
        return _near(sheet, 20, 20.5, 10) - _near(sheet, 180, 20.5, 10)

    usual = blob(pens.BLUE)
    assert usual > 0.4 * MOST * _firm(0.5)
    assert blob(pens.BLUE.with_(flow=0.5)) == pytest.approx(0.5 * usual, rel=1e-4)
    # (as the square of its radius, half a pixel more than half its width)
    assert 2.0 * usual < blob(pens.BLUE.with_(width=6.0)) < 4.0 * usual


def _start_blob(pen, pressure: float, seed: int) -> tuple[float, float]:
    """The ink where a line lands, more than at its end, as a share of that
    at its end; and the volume laid."""
    sheet = Sheet(200, 40, seed=seed)
    laid = sheet.stroke([(20, 20.5), (180, 20.5)], pen, pressure)
    start, end = _near(sheet, 20, 20.5, 5), _near(sheet, 180, 20.5, 5)
    return start / end - 1.0, laid


def test_a_light_line_blots_as_lightly_as_it_is_drawn():
    """A blob is as much darker than its line at a light touch as at a firm
    one: the paper takes of the drop as much as the ball bears on it. (It
    took all of it, so a faint line had blobs as big and as dark as a firm
    line's: 3.5 times as much ink again as its line at 0.15, 0.86 at 0.9.)"""
    seed = _loaded()
    light, _ = _start_blob(pens.BLUE, 0.15, seed)
    firm, _ = _start_blob(pens.BLUE, 0.9, seed)
    assert firm > 0.5
    assert 0.8 * firm < light < 1.4 * firm


def test_a_pen_that_never_blots_and_a_leaky_one():
    seed = _loaded()
    usual, laid = _start_blob(pens.BLUE, 0.5, seed)
    never, dry = _start_blob(pens.BLUE.with_(blots=0.0), 0.5, seed)
    leaky, wet = _start_blob(pens.BLUE.with_(blots=2.0), 0.5, seed)
    assert usual > 0.5
    assert abs(never) < 0.05
    # the same line, and twice the blob
    assert wet - dry == pytest.approx(2.0 * (laid - dry), rel=1e-4)
    assert leaky > 1.8 * usual
    # and none where it turns either
    zigzag = Polyline([(20.5 + 20 * k, 20.5 + 40 * (k % 2)) for k in range(9)])
    sheets = []
    for blots in (0.0, 1.0):
        sheet = _clean(200, 80, 1)
        sheet.stroke(zigzag, pens.BLUE.with_(blots=blots), 0.6)
        assert sheet.ink is not None
        sheets.append(sheet.ink[..., IV])
    corner = (slice(15, 26), slice(95, 106))  # (a turn, at (100.5, 20.5))
    assert sheets[1][corner].sum() > 1.1 * sheets[0][corner].sum()


def _drops(sheet, volume, xs=range(80, 120, 4), y=20.0) -> float:
    """Leave a drop of this volume from a blue pen at each of ``xs``, as
    the ball does (:func:`fusain._engine.blot`). Returns the volume laid."""
    pen = pens.BLUE
    kept = np.array([pen.ink.S, pen.flow, pen.blots, *pen.mirrors], dtype=np.float64)
    k = np.ascontiguousarray(pen.ink.K, dtype=np.float64)
    laid = 0.0
    for x in xs:
        laid += eng.blot(
            sheet.cover,
            sheet.held,
            sheet.under,
            sheet.loose,
            sheet.ink,
            sheet.inked,
            x + 0.5,
            y,
            volume,
            eng.BLOB_WIDE * 2.0,
            eng.BALL_INK * pen.flow,
            k,
            kept,
        )
    return laid


@pytest.mark.parametrize("paper", [papers.CARTRIDGE, papers.ROUGH])
def test_a_drop_inks_a_pixel_as_far_as_it_is_ink(paper):
    """A drop inks as much of a pixel's bare paper as it would cover a firm
    pass thick, on its own: a tiny one next to nothing. (Any drop, however
    small, inked all of a pixel that held a pass of ink: a billionth of a
    firm pass, left every four pixels along a line drawn three times firmly
    between two rows, inked a third of the half of its edge pixels left
    bare, 0.50 to 0.34, and darkened the line by a seventh.)"""
    sheet = Sheet(200, 40, paper=paper, seed=3)
    for _ in range(3):
        sheet.stroke([(10, 20.0), (190, 20.0)], pens.BLUE.with_(blots=0.0), 0.8)
    assert sheet.ink is not None and sheet.inked is not None
    line = (slice(17, 24), slice(78, 122))
    edge = (sheet.inked[line] > 0.0) & (sheet.inked[line] < 1.0)
    assert edge.sum() > 60 and 0.4 < (1.0 - sheet.inked[line][edge]).mean() < 0.6
    tone = _luminance(sheet)[line].mean()
    for volume in (1e-9, 1e-4, 0.01):
        dropped = sheet.copy()
        assert dropped.ink is not None and dropped.inked is not None
        assert _drops(dropped, volume) == pytest.approx(10 * volume, rel=1e-6)
        assert _ink(dropped) == pytest.approx(_ink(sheet) + 10 * volume, rel=1e-6)
        # each pixel's bare paper is inked as far as the drop there covers
        # it a firm pass thick
        bare = 1.0 - sheet.inked[line].astype(np.float64)
        dv = dropped.ink[line][..., IV] - sheet.ink[line][..., IV]
        covers = np.minimum(dv / eng.BALL_INK, 1.0)
        inked = dropped.inked[line]
        assert inked == pytest.approx(1.0 - bare * (1.0 - covers), abs=1e-6)
        darker = 1.0 - _luminance(dropped)[line].mean() / tone
        if volume <= 1e-4:  # (next to nothing)
            assert (1.0 - inked[edge]).mean() > bare[edge].mean() - 0.001
            assert darker < 0.001
        else:
            assert 0.001 < darker < 0.05


def _blob_radius(pen, pressure: float) -> tuple[float, float]:
    """The radius of the blob a pen leaves where it lands (from the spread
    of its ink, as a dot thickest at its middle), and the width of its line
    further on (likewise)."""
    sheets = []
    for blots in (0.0, 1.0):
        sheet = Sheet(200, 40, seed=_loaded())
        sheet.stroke([(20.5, 20.5), (180, 20.5)], pen.with_(blots=blots), pressure)
        assert sheet.ink is not None
        sheets.append(sheet.ink[..., IV].astype(np.float64))
    ys, xs = np.mgrid[:40, :200]
    blob = sheets[1] - sheets[0]
    out = (xs + 0.5 - 20.5) ** 2 + (ys + 0.5 - 20.5) ** 2
    line = sheets[0][:, 60:140]
    across = (ys[:, 60:140] + 0.5 - 20.5) ** 2
    return (
        float(np.sqrt(3.0 * (blob * out).sum() / blob.sum())),
        float(2.0 * np.sqrt(3.0 * (line * across).sum() / line.sum())),
    )


def test_a_light_line_leaves_a_narrower_blob():
    """A blob is as much wider than the paper the ball meets where it is
    left as BLOB_WIDE says: a lighter line, narrower, leaves a narrower blob,
    still a little wider than the line. (It was as wide at any pressure: a
    pen six pixels wide left a blob of radius 4.2 at 0.05, where its line is
    3.6 wide.)"""
    pen = pens.ballpoint("#22357e", width=6)
    measured = [_blob_radius(pen, p) for p in (0.05, 0.15, 0.3, 0.9)]
    blobs, lines = [b for b, _ in measured], [w for _, w in measured]
    assert all(a < b for a, b in itertools.pairwise(blobs))
    assert all(a < b for a, b in itertools.pairwise(lines))
    # as the line narrows, so does the blob
    assert blobs[0] < 0.8 * blobs[-1]
    assert abs(blobs[0] / blobs[-1] - lines[0] / lines[-1]) < 0.12
    # and it is wider than its line (the ball, 6 wide, reaches 3.5 pixels)
    assert all(2.0 * b > 1.1 * w for b, w in zip(blobs, lines, strict=True))
    assert blobs[-1] == pytest.approx(eng.BLOB_WIDE * 3.5, rel=0.05)


def test_a_blob_off_the_edge_of_the_sheet_or_on_wax_is_not_counted_as_laid():
    for seed in range(40):
        sheet = Sheet(120, 40, seed=seed)
        laid = sheet.stroke([(-1.0, 20.5), (100, 20.5)], pens.BLUE, 0.5)
        assert _ink(sheet) == pytest.approx(laid, rel=1e-5)
    # on a patch of wax the ball skids, and its drop skids with it
    sheet, bare = Sheet(200, 150, seed=1), Sheet(200, 150, seed=1)
    for layer in range(3):
        sheet.shade(BOX, crayons.wax("#c0392b"), 0.9, 9, angle=30 + 50 * layer)
    on_wax = on_paper = 0.0
    for k in range(20):
        line = [(60, 40 + 3 * k), (140, 41 + 3 * k)]
        on_wax += sheet.stroke(line, pens.BLUE, 0.5)
        on_paper += bare.stroke(line, pens.BLUE, 0.5)
    assert _ink(sheet) == pytest.approx(on_wax, rel=1e-5)
    assert on_wax < 0.1 * on_paper


def test_a_wide_pen_and_a_rough_paper_and_a_level_one():
    wide = _line(0.6, pen=pens.ballpoint("#22357e", width=6))
    assert 1.6 * _ink(_line(0.6)) < _ink(wide) < 2.4 * _ink(_line(0.6))
    # on a rough paper the lightest line breaks up a little
    rough = _line(0.05, papers.ROUGH)
    assert rough.inked is not None
    assert 0.5 < (rough.inked[19:21, 60:240].max(axis=0) > 0.5).mean() < 0.98
    level = Sheet(300, 40, paper=papers.CARTRIDGE.with_(depth=0.0), seed=1)
    assert level.stroke(LINE, pens.BLUE, 0.3) > 0


def test_lines_that_cross_darken():
    sheet = Sheet(120, 120, paper=papers.PRINTER, seed=1)
    sheet.stroke([(10, 60), (110, 60)], pens.BLUE, 0.5)
    sheet.stroke([(60, 10), (60, 110)], pens.BLUE, 0.5)
    tone = _luminance(sheet)
    crossing = float(tone[59:61, 59:61].mean())
    single = float(tone[59:61, 20:40].mean())
    assert crossing < 0.6 * single


# --------------------------------------------------------------------------
# a pen in the hand: what one line gathers, the next lands with


def test_a_pen_in_the_hand_is_its_pen_and_the_ink_on_its_ball():
    pen = HeldPen(pens.BLUE)
    assert pen.pen is pens.BLUE and pen.width == 3.0 and pen.load == 0.0
    assert "blue ballpoint" in repr(pen)
    with pytest.raises(TypeError, match="Pen"):
        HeldPen(pencils.HB)  # type: ignore[arg-type]
    sheet = Sheet(120, 60, seed=1)
    with pytest.raises(TypeError, match="a HeldPen has its own width"):
        sheet.stroke([(5, 5), (50, 30)], pen, width=4)
    with pytest.raises(TypeError, match="a HeldPen has its own width"):
        sheet.shade((5, 5, 50, 30), pen, width=4)
    assert sheet.stroke([(10, 20.5), (110, 20.5)], pen, 0.5) > 0
    load = pen.load
    assert load > 0
    assert pen.wipe() == load and pen.load == 0.0


def _gathered_on_a_line(seed: int, share=lambda x: x > 0.5) -> tuple:
    """A pen in the hand that has drawn a straight line, on a sheet where
    the share of what it lands with that comes round where the next line
    lands is as asked. Returns the sheet, the pen and the ink it laid."""
    for salt in range(seed, seed + 200):
        sheet, pen = Sheet(200, 80, seed=salt), HeldPen(pens.BLUE)
        laid = sheet.stroke([(20, 20.5), (180, 20.5)], pen, 0.5)
        if share(_lands_with(sheet)):
            return sheet, pen, laid
    raise AssertionError("no such sheet")


def test_a_pen_in_the_hand_lands_with_what_it_gathered():
    """It lands with what the line before gathered. Where it lands, the
    share drawn for the line comes round (as for a pen just picked up: now
    and then much of it) and the paper takes as much of that as the pen
    bears on it; the rest stays on the ball, with what this line gathers."""
    sheet, pen, first = _gathered_on_a_line(1)
    # it is taken up clean, and a straight line leaves nothing of what it
    # gathers
    gathered = pen.load
    assert gathered == pytest.approx(eng.BALL_GATHER * first, rel=1e-5)
    # the next line, and the same with a pen that gathers nothing
    plain, dry = sheet.copy(), HeldPen(pens.BLUE.with_(blots=0.0))
    left = _firm(0.7) * _lands_with(sheet) * gathered
    line = [(20, 60.5), (120, 60.5)]
    second = sheet.stroke(line, pen, 0.7)
    bare = plain.stroke(line, dry, 0.7)
    assert second - bare == pytest.approx(left, rel=1e-4)
    assert _near(sheet, 20, 60.5, 5) - _near(plain, 20, 60.5, 5) == pytest.approx(
        left, rel=1e-4
    )
    assert pen.load == pytest.approx(gathered - left + eng.BALL_GATHER * bare, rel=1e-4)


def test_what_does_not_come_off_where_it_lands_comes_off_at_the_first_turn():
    """A line that turns sharply soon after it lands: a blob there from a
    pen in the hand that has drawn a long line before, none from a clean
    one."""
    corners = []
    for seed in (1, 20, 40):
        sheet, pen, _ = _gathered_on_a_line(seed, lambda x: x < 0.01)
        wiped = HeldPen(pens.BLUE)
        clean = sheet.copy()
        path = Polyline([(150.5, 40.5), (160.5, 40.5), (160.5, 75.5)])
        sheet.stroke(path, pen, 0.5)
        clean.stroke(path, wiped, 0.5)
        corners.append(_near(sheet, 160.5, 40.5) / _near(clean, 160.5, 40.5))
    assert min(corners) > 1.3


def test_what_a_pen_in_the_hand_gathers_is_on_the_paper_or_on_its_ball():
    """Over a scribble of sharp turns and a few lines, the ink the pen laid
    in drops and the ink still on its ball are what it gathered: the share
    BALL_GATHER of the ink of its lines."""
    zigzag = Polyline([(20.5 + 12 * k, 20.5 + 25 * (k % 2)) for k in range(14)])
    sheets, pens_, laid = [], [], []
    for blots in (0.0, 1.0):
        sheet, pen = _clean(220, 150, 2), HeldPen(pens.BLACK.with_(blots=blots))
        laid.append(sheet.stroke(zigzag, pen, 0.8))
        for k in range(3):  # (short, so that it never holds the most it can)
            y = 80.5 + 12 * k
            laid[-1] += sheet.stroke([(20, y), (50, y + 3)], pen, 0.8)
        sheets.append(sheet)
        pens_.append(pen)
    for sheet, total in zip(sheets, laid, strict=True):
        assert _ink(sheet) == pytest.approx(total, rel=1e-5)
    lines, drops = laid[0], laid[1] - laid[0]
    assert pens_[0].load == 0.0 and drops > 0.0
    assert drops + pens_[1].load == pytest.approx(eng.BALL_GATHER * lines, rel=1e-4)


def test_a_ball_holds_at_most_a_drop_and_gathers_no_more():
    """What the ball would gather beyond the most it holds is not gathered:
    it is neither on the paper nor on the ball. A long straight line fills
    the rim, and the next one keeps it full."""
    pen = HeldPen(pens.BLUE)
    sheet = Sheet(320, 60, seed=1)
    first = sheet.stroke([(10, 20.5), (310, 20.5)], pen, 0.8)
    assert eng.BALL_GATHER * first > 1.5 * MOST
    assert pen.load == pytest.approx(MOST, rel=1e-12)
    sheet.stroke([(10, 40.5), (310, 40.5)], pen, 0.8)
    assert pen.load == pytest.approx(MOST, rel=1e-12)


# --------------------------------------------------------------------------
# hatching


def test_layers_of_hatching_darken():
    sheet = Sheet(200, 150, seed=1)
    tones = []
    for layer in range(4):
        sheet.shade(BOX, pens.BLACK, 0.5, angle=30 + 47 * layer)
        tones.append(_tone(sheet))
    assert all(a > b for a, b in itertools.pairwise(tones))
    assert 0.4 < tones[0] < 0.65
    assert tones[-1] < 0.12


def test_hatching_is_lighter_under_a_lighter_hand():
    tones = []
    for pressure in (0.05, 0.2, 0.5, 1.0):
        sheet = Sheet(200, 150, seed=1)
        sheet.shade(BOX, pens.BLACK, pressure)
        tones.append(_tone(sheet))
    assert all(a > b for a, b in itertools.pairwise(tones))
    assert tones[0] > 0.8 and tones[-1] < 0.45


def test_the_lines_of_hatching_lie_side_by_side_unless_told():
    open_, close = Sheet(200, 150, seed=1), Sheet(200, 150, seed=1)
    a = open_.shade(BOX, pens.BLACK, 0.5, hand=0.0)
    b = close.shade(BOX, pens.BLACK, 0.5, spacing=2.4, hand=0.0)
    assert 1.8 * a < b < 2.2 * a


# --------------------------------------------------------------------------
# with the other media


def test_no_rubber_eraser_nor_a_finger_takes_the_ink_off():
    sheet = Sheet(200, 150, seed=1)
    sheet.shade(BOX, pens.BLUE, 0.6)
    ink, picture = _ink(sheet), sheet.render()
    assert sheet.erase([(30, 75), (170, 75)], tools.ERASER, 1.0, 20) == 0.0
    assert sheet.erase([(30, 60), (170, 60)], tools.KNEADED, 1.0) == 0.0
    assert sheet.smudge([(30, 90), (170, 90)], tools.FINGER, 1.0) == 0.0
    assert sheet.fix() == 0.0
    assert _ink(sheet) == ink
    assert np.array_equal(sheet.render(), picture)


def test_a_knife_or_a_sand_eraser_takes_the_ink_with_the_paper():
    # (the ink is a stain in the paper: what cuts the surface of the paper
    # takes it, and has it; a knife or a sand eraser touched none of it)
    sheet = Sheet(200, 150, seed=1)
    for layer in range(2):
        sheet.shade(BOX, pens.BLACK, 0.7, angle=30 + 60 * layer)
    before, ink = sheet.copy(), _ink(sheet)
    dark = _tone(sheet)
    assert sheet.ink is not None and before.ink is not None
    # a fine light line scratched out with the point of a knife
    took = sheet.erase([(40, 76), (160, 76)], tools.KNIFE, 0.9)
    line, beside = (slice(75, 77), slice(50, 150)), (slice(60, 72), slice(50, 150))
    assert sheet.ink[line][..., IV].sum() < 0.05 * before.ink[line][..., IV].sum()
    assert _tone(sheet, line) > 0.6 > 0.3 > dark
    assert np.array_equal(sheet.ink[beside], before.ink[beside])
    assert took == pytest.approx(ink - _ink(sheet), rel=1e-5)
    # a patch rubbed back with a sand eraser, most of the way
    ink = _ink(sheet)
    took = sheet.rub((60, 90, 140, 125), tools.SAND_ERASER, 0.9, 10)
    patch = (slice(96, 119), slice(66, 134))
    assert sheet.ink[patch][..., IV].sum() < 0.2 * before.ink[patch][..., IV].sum()
    assert _tone(sheet, patch) > 2.0 * dark
    assert took == pytest.approx(ink - _ink(sheet), rel=1e-5)
    assert float(sheet.ink[..., IV].min()) >= 0.0


def test_an_eraser_takes_the_pencil_and_leaves_the_ink():
    sheet = Sheet(200, 150, seed=1)
    sheet.shade(BOX, pencils.B2, 0.5)
    sheet.stroke([(30, 75), (170, 75)], pens.BLACK, 0.7)
    line = (slice(74, 76), slice(60, 140))
    beside = (slice(60, 66), slice(60, 140))
    ink = _ink(sheet)
    for k in range(8):
        y = 50 + 7 * k
        sheet.erase([(30, y), (170, y)], tools.ERASER, 1.0, 14)
    assert _ink(sheet) == ink
    assert _tone(sheet, beside) > 0.85
    assert _tone(sheet, line) < 0.25


@pytest.mark.parametrize("over", [False, True], ids=["ink", "ink over wax"])
def test_a_scraping_tool_takes_no_more_ink_than_it_has_room_for(over):
    # a sand eraser or a knife that holds next to nothing fills on the first
    # pixels it cuts, and leaves the ink of the rest on the paper (it took
    # the ink of every pixel it cut, whatever it held: 167 to 351 times as
    # much as it holds)
    for tool in (tools.SAND_ERASER, tools.KNIFE):
        tiny = tool.with_(holds=1e-6)
        full = tiny.holds * np.pi * (0.5 * tiny.width + 0.5) ** 2
        sheet = Sheet(200, 150, seed=1)
        if over:
            sheet.shade(BOX, pencils.coloured("#c0392b"), 0.6)
        for layer in range(2):
            sheet.shade(BOX, pens.BLACK, 0.7, angle=30 + 60 * layer)
        before, ink = sheet.copy(), _ink(sheet)
        pigment = float(sheet.pigment().sum(dtype=np.float64))
        held = Smudger(tiny)
        for y in (60, 76, 92):
            sheet.erase([(40, y), (160, y)], held, 0.9)
            # (what it has lifted: what it drags along on its face it lays
            # back, as a stump does)
            assert 0.0 < float(held._carried[0, IV]) <= full * (1 + 1e-6)
        # what the tool holds is what left the sheet, ink and pigment alike
        left = _ink(sheet) + float(sheet.pigment().sum(dtype=np.float64))
        assert left + held.load == pytest.approx(ink + pigment, rel=1e-6)
        assert _ink(sheet) <= ink
        # and the ink of the lines it cut once full is on the paper still
        assert sheet.ink is not None and before.ink is not None
        line = (slice(91, 93), slice(100, 150))
        assert sheet.ink[line][..., IV].sum() > 0.99 * before.ink[line][..., IV].sum()


def test_a_pencil_draws_over_light_ink_as_on_bare_paper():
    """The ink is in the paper, not on it. (A pen pressed firmly leaves
    grooves, which a pencil skips: see test_surface.py.)"""
    bare, inked = Sheet(200, 150, seed=1), Sheet(200, 150, seed=1)
    for layer in range(3):
        inked.shade(BOX, pens.BLUE, 0.25, angle=30 + 47 * layer)
    over = inked.shade(BOX, pencils.HB, 0.5)
    assert 0.9 < over / bare.shade(BOX, pencils.HB, 0.5) < 1.1
    firm = Sheet(200, 150, seed=1)
    for layer in range(3):
        firm.shade(BOX, pens.BLUE, 0.9, angle=30 + 47 * layer)
    assert firm.shade(BOX, pencils.HB, 0.5) < 0.6 * over


def _share_taken(under, layers, pressure, width=None) -> float:
    """What a layer of hatching lays over that, as a share of what it lays
    on bare paper."""
    drawn, bare = Sheet(200, 150, seed=1), Sheet(200, 150, seed=1)
    for layer in range(layers):
        drawn.shade(BOX, under, pressure, width, angle=30 + 50 * layer)
    over = drawn.shade(BOX, pens.BLUE, 0.5, angle=-20)
    return over / bare.shade(BOX, pens.BLUE, 0.5, angle=-20)


def test_the_ball_skids_on_heavy_graphite_and_on_wax_not_on_a_light_sketch():
    assert _share_taken(pencils.HB, 1, 0.3) > 0.9
    assert _share_taken(pencils.B6, 3, 1.0) < 0.05
    assert _share_taken(crayons.wax("#c0392b"), 3, 0.8, 9) < 0.25


def test_a_pass_that_skids_leaves_the_ink_under_it_as_it_was():
    sheet = Sheet(200, 150, seed=1)
    sheet.stroke([(30, 75), (170, 75)], pens.BLUE, 0.1)  # a light, narrow line
    assert sheet.ink is not None and sheet.inked is not None
    ink, inked = sheet.ink.copy(), sheet.inked.copy()
    for layer in range(4):
        sheet.shade(BOX, pencils.B8, 1.0, angle=30 + 50 * layer)
    # a firm, wider pass along the same line, on graphite too thick to write on
    wide = pens.ballpoint("#22357e", width=6)
    laid = sheet.stroke([(60, 75), (140, 75)], wide, 1.0)
    core = (slice(70, 81), slice(70, 130))
    assert laid < 0.02 * Sheet(200, 150, seed=1).stroke(
        [(60, 75), (140, 75)], wide, 1.0
    )
    skidded = sheet.ink[core][..., IV] == ink[core][..., IV]
    assert skidded.mean() > 0.9
    # where it laid nothing, no more of the pixel is inked than before
    assert np.array_equal(sheet.inked[core][skidded], inked[core][skidded])


def test_the_ink_is_under_what_is_drawn_over_it():
    sheet, plain = Sheet(200, 150, seed=1), Sheet(200, 150, seed=1)
    for layer in range(3):
        sheet.shade(BOX, pens.BLACK, 0.7, angle=30 + 47 * layer)
    assert _tone(sheet) < 0.15
    white = pastels.soft("#f4f1ea")
    for target in (sheet, plain):
        for layer in range(2):
            target.shade(BOX, white, 0.8, 12, angle=30 + 50 * layer)
    assert _tone(sheet) > 0.8 * _tone(plain)


def test_a_copy_has_its_own_ink_and_a_drawing_repeats():
    first, second = Sheet(200, 150, seed=3), Sheet(200, 150, seed=3)
    assert first.copy().ink is None
    for sheet in (first, second):
        sheet.shade(BOX, pens.BLUE, 0.5)
    assert first.ink is not None and second.ink is not None
    assert np.array_equal(first.ink, second.ink)
    copy = first.copy()
    copy.stroke([(30, 75), (170, 75)], pens.RED, 0.8)
    assert _ink(copy) > _ink(first)
    assert np.array_equal(first.ink, second.ink)
