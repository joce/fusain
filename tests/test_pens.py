"""Ballpoint pens: an even line at any pressure, ink that layers and that no
tool takes off, measured."""

import itertools

import numpy as np
import pytest
from murex import Pigment
from murex import spectral as sp

from fusain import Pen, Sheet, crayons, papers, pastels, pencils, pens, tools
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


def test_a_pencil_draws_over_ink_as_on_bare_paper():
    bare, inked = Sheet(200, 150, seed=1), Sheet(200, 150, seed=1)
    for layer in range(3):
        inked.shade(BOX, pens.BLUE, 0.7, angle=30 + 47 * layer)
    over = inked.shade(BOX, pencils.HB, 0.5)
    assert 0.85 < over / bare.shade(BOX, pencils.HB, 0.5) < 1.15


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
