"""Ballpoint pens: an even line at any pressure, ink that layers and that no
tool takes off, measured."""

import itertools

import numpy as np
import pytest
from murex import Pigment, Polyline
from murex import spectral as sp

from fusain import Pen, Sheet, crayons, papers, pastels, pencils, pens, tools
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


def test_the_ink_does_not_come_quite_evenly():
    sheet = _line(0.6, papers.PRINTER)
    assert sheet.ink is not None
    along = sheet.ink[20, 60:240, IV]
    assert 0.03 < along.std() / along.mean() < 0.2


# --------------------------------------------------------------------------
# blobs: the ball gathers ink, and leaves it where it lands and where it turns

#: The most ink the ball of a pen 3 wide gathers.
MOST = eng.BALL_BLOB * eng.BALL_INK * (0.5 * 3 + 0.5) ** 2


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
    for seed in range(60):
        sheet = Sheet(200, 40, seed=seed)
        gathered.append(_lands_with(sheet))
        laid = sheet.stroke([(20, 20.5), (180, 20.5)], pens.BLUE, 0.5)
        assert _ink(sheet) == pytest.approx(laid, rel=1e-5)  # blob and all
        # the ink where the line starts, and where it ends (nothing but line)
        start, end = _near(sheet, 20, 20.5, 5), _near(sheet, 180, 20.5, 5)
        starts.append(start / end)
        extra.append((start - end) / MOST)
    share, ratio = np.array(gathered), np.array(starts)
    # mostly next to nothing; one line in four or so starts with a dot that
    # shows; never more than a pen has on it
    assert np.median(share) < 0.05
    assert 0.15 < (share > 0.1).mean() < 0.4
    assert 0.3 < share.max() <= eng.BALL_START
    # and that is the ink on the paper (the line itself is not quite even)
    assert np.abs(np.array(extra) - share).max() < 0.15
    assert ratio[share < 0.01].min() > 0.7 and ratio[share < 0.01].max() < 1.3
    assert ratio[share > 0.3].min() > 1.6


def test_a_blob_is_a_dot_a_little_wider_than_the_line_and_thickest_in_the_middle():
    sheet = Sheet(200, 40, seed=_loaded())
    sheet.stroke([(20.5, 20.5), (180, 20.5)], pens.BLUE, 0.3)
    assert sheet.ink is not None and sheet.inked is not None
    ink, inked = sheet.ink[..., IV], sheet.inked
    # across the line, at its start and half way along
    start, run = ink[:, 20], ink[:, 100]
    assert start.max() > 2.5 * run.max()
    assert np.argmax(start) == 20
    assert np.count_nonzero(start) > np.count_nonzero(run)
    assert np.count_nonzero(start) <= 2 * eng.BLOB_WIDE * (0.5 * 3 + 0.5) + 1
    # it fills the tooth: all of the pixels under its middle are inked
    assert inked[20, 19:22].min() == 1.0 and inked[19:22, 20].min() == 1.0


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
    assert blobs[2] > 1.4
    # what gathers over a run is the share BALL_GATHER of what the run lays
    # (about 0.045 a pixel), and a square turn brings most of it round
    gathered = eng.BALL_GATHER * 140 * 0.045
    assert 0.4 * gathered < extra[2] < 0.9 * gathered


def test_a_line_that_turns_back_leaves_all_it_gathered():
    square = [_corner(140, seed) for seed in (1, 20, 40, 60)]
    back = [_corner(140, seed, (20.5, 50.5)) for seed in (1, 20, 40, 60)]
    extra = [float(np.mean([at - beyond for at, beyond in c])) for c in (square, back)]
    assert extra[1] > 1.2 * extra[0]
    assert extra[1] < eng.BALL_GATHER * 140 * 0.045 * 1.15


def test_a_gentle_curve_leaves_no_blob():
    """The ink comes round a little at a time, and the line is as even as
    a straight one."""
    arc = [(100 + 60 * np.cos(a), 75 - 60 * np.sin(a)) for a in np.linspace(0, 3, 40)]
    for seed in (1, 20, 40):
        curved, straight = _clean(200, 100, seed), _clean(200, 100, seed)
        bent = curved.stroke(arc, pens.BLUE, 0.5)
        ruled = straight.stroke([(10, 50.5), (190, 50.5)], pens.BLUE, 0.5)
        assert _thickest(curved) < 1.1 * _thickest(straight)
        # what gathered is in the line: a little more ink than a straight one
        assert 1.0 < bent / ruled < 1.0 + eng.BALL_GATHER


def test_a_freer_pen_and_a_bigger_ball_gather_more():
    seed = _loaded()

    def blob(pen) -> float:
        sheet = Sheet(200, 40, seed=seed)
        sheet.stroke([(20, 20.5), (180, 20.5)], pen, 0.5)
        return _near(sheet, 20, 20.5, 10) - _near(sheet, 180, 20.5, 10)

    usual = blob(pens.BLUE)
    assert usual > 0.4 * MOST
    assert blob(pens.BLUE.with_(flow=0.5)) == pytest.approx(0.5 * usual, rel=1e-4)
    # (as the square of its radius, half a pixel more than half its width)
    assert 2.0 * usual < blob(pens.BLUE.with_(width=6.0)) < 4.0 * usual


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


def test_no_tool_takes_the_ink_off():
    sheet = Sheet(200, 150, seed=1)
    sheet.shade(BOX, pens.BLUE, 0.6)
    ink, picture = _ink(sheet), sheet.render()
    assert sheet.erase([(30, 75), (170, 75)], tools.ERASER, 1.0, 20) == 0.0
    assert sheet.erase([(30, 60), (170, 60)], tools.KNEADED, 1.0) == 0.0
    assert sheet.smudge([(30, 90), (170, 90)], tools.FINGER, 1.0) == 0.0
    assert sheet.fix() == 0.0
    assert _ink(sheet) == ink
    assert np.array_equal(sheet.render(), picture)


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
