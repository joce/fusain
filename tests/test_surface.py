"""The surface of the paper: a firm hard tip flattens the tooth, and a
stylus or the ball of a pen indents a groove that a pencil skips,
measured."""

import itertools

import numpy as np
import pytest
from murex import spectral as sp

from fusain import (
    Paper,
    Sheet,
    charcoal,
    crayons,
    papers,
    pastels,
    pencils,
    pens,
    tools,
)
from fusain import _engine as eng
from fusain.sheet import TOOTH_HEIGHT

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
#: A line across the sheet, through the middle of a row of pixels; the two
#: rows under the middle of a stylus drawn along it, and the paper either side.
LINE = [(20, 75.5), (180, 75.5)]
GROOVE = (slice(75, 76), slice(50, 150))
AROUND = (slice(60, 68), slice(50, 150))
OVERHEAD, RAKING = (0.0, 0.0, 1.0), (-1.0, -1.0, 0.5)


def _sheet(paper=papers.CARTRIDGE) -> Sheet:
    return Sheet(200, 150, paper=paper, seed=1)


def _shaded(medium, pressure=1.0, layers=1, width=None) -> Sheet:
    sheet = _sheet()
    for layer in range(layers):
        sheet.shade(BOX, medium, pressure, width, angle=30 + 50 * layer)
    return sheet


def _pressed(sheet) -> float:
    return float(sheet.press[INSIDE].mean())


def _luminance(sheet, where) -> float:
    spectrum = sheet.reflectance()[where].astype(np.float64)
    white = sp.reflectance_to_linear_rgb(sheet.ground) @ sp.LUMINANCE
    return float((sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE).mean() / white)


def _light(image, where=INSIDE) -> np.ndarray:
    return sp.srgb_to_linear(image[where].astype(np.float64)) @ sp.LUMINANCE


def _shading(sheet, lamp, where=INSIDE) -> np.ndarray:
    """How a lamp lights each pixel of the relief, as a share of the light
    of a level sheet (what lies on the paper, and what shines, aside)."""
    lit = sheet.render(light=lamp, gloss=0)[where].astype(np.float64)
    flat = sheet.render(lighting=False)[where].astype(np.float64)
    share = sp.srgb_to_linear(lit) / np.maximum(sp.srgb_to_linear(flat), 1e-9)
    return share.mean(axis=2)


def _in_groove(sheet) -> float:
    """What lies in the groove, as a share of what lies on the paper beside."""
    pigment = sheet.pigment()
    return float(pigment[GROOVE].mean() / pigment[AROUND].mean())


# --------------------------------------------------------------------------
# a firm tip flattens the tooth


def test_a_firm_hard_pencil_flattens_the_tooth_and_a_soft_one_hardly():
    pressed = [
        _pressed(_shaded(pencils.grade(name), 1.0, 3))
        for name in ("4H", "HB", "2B", "6B")
    ]
    assert all(a > b for a, b in itertools.pairwise(pressed))
    assert pressed[0] > 0.8 and pressed[1] > 0.6
    assert 0.2 < pressed[2] < 0.5
    assert pressed[3] < 0.05


def test_the_firmer_and_the_more_often_the_flatter():
    # a light touch: next to none (where its lead bears hardest, a trace: it
    # sets in by degrees)
    assert _pressed(_shaded(pencils.HB, 0.4, 3)) < 0.005
    firm = [_pressed(_shaded(pencils.HB, p)) for p in (0.7, 0.85, 1.0)]
    assert 0.0 < firm[0] < firm[1] < firm[2] < 0.3
    often = [_pressed(_shaded(pencils.HB, 1.0, n)) for n in (1, 3, 6)]
    assert often[0] < often[1] < often[2]
    assert often[2] > 0.95
    assert _shaded(pencils.H4, 1.0, 8).press.max() <= 1.0


def test_charcoal_and_pastel_next_to_do_not_flatten_it():
    assert _pressed(_shaded(pastels.soft("#c0392b"), 1.0, 3, 10)) < 1e-3
    assert _pressed(_shaded(charcoal.VINE, 1.0, 3, 8)) < 0.02
    assert _pressed(_shaded(charcoal.COMPRESSED, 1.0, 3, 8)) < 0.1


def test_what_flattens_is_a_tip_that_is_hard_dry_and_all_of_a_piece():
    paper = np.array([1.0, 0.5, 0.5, 1.0])

    def crush(softness, binding=0.85, wax=0.0) -> float:
        medium = np.array([0.1, softness, binding, wax, 0.0, 0.0])
        return float(eng.law_of(medium, paper)[0][10])

    assert crush(0.0, 1.0) == pytest.approx(eng.FLATTEN)
    assert crush(0.0) > crush(0.3) > crush(0.6) > crush(0.9) > crush(1.0) == 0.0
    assert crush(0.3, 0.9) > crush(0.3, 0.4) > crush(0.3, 0.0) == 0.0
    # (a waxy tip polishes its own film instead)
    assert crush(0.3) > crush(0.3, wax=0.15) > crush(0.3, wax=0.3) == 0.0


def test_a_flattened_tooth_is_level_and_touched_all_over():
    sheet = _shaded(pencils.HB, 1.0, 3)
    rough = sheet.copy()
    rough.press[:] = 0.0  # the same graphite, on a tooth that kept its shape
    assert sheet.surface_height()[INSIDE].std() < 0.8 * (
        rough.surface_height()[INSIDE].std()
    )
    # (fusain-8au) pressed flat, its peaks come down level with its valleys,
    # and a valley the lead never reached is not left a pit: six firm layers
    # of 4H were level with the peaks, and 0.06 of the patch lay more than a
    # quarter of the depth of the tooth below that, 0.85 on average
    flat = _shaded(pencils.H4, 1.0, 6)
    kept = flat.copy()
    kept.press[:] = 0.0
    top = TOOTH_HEIGHT * flat.paper.grain * flat.paper.depth
    level = flat.surface_height()[INSIDE] / top
    untouched = kept.surface_height()[INSIDE] / top
    assert np.median(level) < 0.05 < 0.4 < np.median(untouched)
    assert not (np.median(level) - level > 0.1).any()
    assert (level - np.median(level) > 0.25).mean() < 0.01
    assert level.std() < 0.2 * untouched.std()
    # so it is smooth under the lamp: a raking light shows it a seventh as
    # much as the tooth left alone (it showed it as much)
    shading = [_shading(target, RAKING) for target in (flat, kept)]
    assert shading[0].std() < 0.25 * shading[1].std()
    # a light layer of a soft lead over it lies on all of it, evenly, where
    # on the rough tooth it marks the high ground; and it bites less, so no
    # more of it is laid for that
    added = []
    for target in (sheet, rough):
        before = target.pigment().copy()
        target.shade(BOX, pencils.B4, 0.3, angle=-20)
        added.append((target.pigment() - before)[INSIDE])
    flat, coarse = added
    assert (flat > 0.02).mean() > 1.3 * (coarse > 0.02).mean()
    assert flat.std() / flat.mean() < 0.7 * coarse.std() / coarse.mean()
    assert 0.8 < flat.mean() / coarse.mean() < 1.5


def test_a_flattened_tooth_bites_a_dry_medium_less():
    """One stroke along a strip pressed flat and along one left alone."""
    laid = {}
    for press in (0.0, 1.0):
        for medium in (pencils.B4, crayons.wax("#c0392b")):
            sheet = _sheet(Paper(depth=0.0))  # (level: all of it is touched)
            sheet.press[:] = press
            laid[press, medium.name] = sheet.stroke(LINE, medium, 0.5, 6)
    dry, waxy = pencils.B4.name, crayons.wax("#c0392b").name
    # (a little more than the share it bites: the tooth fills more slowly)
    assert 1 - eng.FLAT_BITE < laid[1.0, dry] / laid[0.0, dry] < 0.5
    assert laid[1.0, waxy] == pytest.approx(laid[0.0, waxy])


def test_graphite_shines_more_and_more_evenly_on_a_flattened_tooth():
    for name, more in (("4H", 2.0), ("HB", 1.2)):
        sheet = _shaded(pencils.grade(name), 1.0, 3)
        rough = sheet.copy()
        rough.press[:] = 0.0
        flat_shine = float(sheet.shine("graphite")[INSIDE].mean())
        assert flat_shine > more * float(rough.shine("graphite")[INSIDE].mean())
    # six firm layers of HB under the default lamp: the light it mirrors
    sheet = _shaded(pencils.HB, 1.0, 6)
    rough = sheet.copy()
    rough.press[:] = 0.0
    uneven = []
    for target in (sheet, rough):
        glint = _light(target.render()) - _light(target.render(gloss=0))
        uneven.append(float(glint.std() / glint.mean()))
    assert uneven[0] < 0.5 * uneven[1]
    # a soft lead shines by how much it lays, flattened or not
    soft = _shaded(pencils.B6, 1.0, 3)
    rough = soft.copy()
    rough.press[:] = 0.0
    assert float(soft.shine("graphite")[INSIDE].mean()) == pytest.approx(
        float(rough.shine("graphite")[INSIDE].mean()), rel=0.02
    )


# --------------------------------------------------------------------------
# a stylus indents a groove


def test_a_stylus_lays_nothing_and_leaves_a_groove():
    sheet = _sheet()
    assert not sheet.dent.any()
    assert sheet.indent(LINE, 0.7, 3.0) is None
    assert not sheet.pigment().any() and not sheet.cover.any()
    assert sheet.ink is None
    # as deep under its middle as it was pressed, on a paper of middling give
    assert float(sheet.dent.max()) == pytest.approx(0.7 * eng.INDENT, rel=1e-5)
    across = sheet.dent[72:79, 100]
    # round: deepest in the middle, nothing beyond the width of the point
    assert np.argmax(across) == 3
    assert across[0] == 0.0 and across[-1] == 0.0
    assert 0.0 < across[2] < across[3] and across[3] > across[4] > 0.0
    assert not sheet.dent[:70].any() and not sheet.dent[81:].any()


def test_a_groove_is_deeper_the_harder_the_stylus_and_the_softer_the_paper():
    deep = []
    for pressure in (0.2, 0.5, 1.0):
        sheet = _sheet()
        sheet.indent(LINE, pressure)
        deep.append(float(sheet.dent.max()))
    assert deep == pytest.approx([p * eng.INDENT for p in (0.2, 0.5, 1.0)], rel=1e-5)
    hard, soft = (_sheet(papers.CARTRIDGE.with_(give=give)) for give in (0.0, 1.0))
    for sheet in (hard, soft):
        sheet.indent(LINE, 0.5)
    assert hard.dent.max() < deep[1] < soft.dent.max()
    assert soft.dent.max() == pytest.approx(
        hard.dent.max() * (eng.GIVE_NONE + eng.GIVE_FULL) / eng.GIVE_NONE
    )


def test_the_deepest_pass_is_what_stays():
    sheet = _sheet()
    sheet.indent(LINE, 0.5)
    once = sheet.dent.copy()
    sheet.indent(LINE, 0.5)  # (the same again presses no deeper)
    assert np.array_equal(sheet.dent, once)
    sheet.indent(LINE, 0.2)
    assert np.array_equal(sheet.dent, once)
    sheet.indent(LINE, 0.9)
    assert sheet.dent.max() > once.max()
    # where the stylus is lifted (no pressure), no groove
    lifted = _sheet()
    lifted.indent(LINE, lambda u: np.where(np.abs(u - 0.5) < 0.1, 0.0, 0.6))
    assert not lifted.dent[:, 95:105].any()
    assert lifted.dent[:, 30:40].any() and lifted.dent[:, 160:170].any()


@pytest.mark.parametrize(
    "medium, width",
    [
        (pencils.HB, None),
        (pencils.B2, None),
        (pencils.B6, None),
        (charcoal.VINE, 8),
        (crayons.wax("#c0392b"), 8),
        (pastels.hard("#2b2b3a"), 8),
    ],
)
def test_a_tone_shaded_over_an_indented_line_skips_it(medium, width):
    sheet = _sheet()
    sheet.indent(LINE, 0.7)
    sheet.shade(BOX, medium, 0.5, width, angle=60)
    assert _in_groove(sheet) < 0.01
    # the line stays the colour of the paper, in a tone
    assert _luminance(sheet, GROOVE) > 0.97
    assert _luminance(sheet, AROUND) < 0.9


def test_a_light_groove_fills_under_a_firm_soft_pencil_and_a_deep_one_does_not():
    filled = {}
    for stylus in (0.3, 0.5, 0.7):
        for pressure in (0.3, 1.0):
            sheet = _sheet()
            sheet.indent(LINE, stylus)
            sheet.shade(BOX, pencils.B6, pressure, angle=60)
            filled[stylus, pressure] = _in_groove(sheet)
    # (fusain-d7v: the floor of a groove, pressed flat by the stylus, bites
    # a dry medium less, as a flattened tooth does; the 6B reaches it as it
    # did, and lays less there: 0.67 of what it lays beside where it laid
    # 0.96, 0.24 where 0.43 and 0.23 where 0.58)
    assert filled[0.3, 1.0] > 0.6
    assert 0.15 < filled[0.3, 0.3] < 0.4
    assert filled[0.5, 0.3] < 0.02 and filled[0.5, 1.0] > 0.15
    assert filled[0.7, 1.0] < 0.2
    # a harder lead reaches less far down
    hard, soft = _sheet(), _sheet()
    for sheet, pencil in ((hard, pencils.H2), (soft, pencils.B6)):
        sheet.indent(LINE, 0.4)
        sheet.shade(BOX, pencil, 0.8, angle=60)
    assert _in_groove(hard) < 0.5 * _in_groove(soft)


@pytest.mark.parametrize("paper", [papers.BRISTOL, papers.ROUGH, Paper(depth=0.0)])
def test_a_groove_is_skipped_on_a_smooth_paper_as_on_a_rough_one(paper):
    sheet = _sheet(paper)
    sheet.indent(LINE, 0.7)
    sheet.shade(BOX, pencils.B2, 0.5, angle=60)
    assert sheet.pigment()[AROUND].mean() > 0.0
    assert _in_groove(sheet) < 0.02


def _along_groove(stylus, groove, medium, width, pressure=0.5) -> tuple[Sheet, Sheet]:
    """A line drawn along a groove ``groove`` pixels wide, indented at
    ``stylus``, and the same line on the paper left alone."""
    grooved, plain = _sheet(), _sheet()
    grooved.indent(LINE, stylus, groove)
    for sheet in (grooved, plain):
        sheet.stroke(LINE, medium, pressure, width)
    return grooved, plain


def _on_floor(stylus, groove, medium, width) -> float:
    """What a line drawn along a groove lays on its floor, as a share of
    what it lays on the paper left alone."""
    grooved, plain = _along_groove(stylus, groove, medium, width)
    return float(grooved.pigment()[GROOVE].mean() / plain.pigment()[GROOVE].mean())


def test_a_point_narrower_than_a_groove_follows_it_down():
    # (fusain-d7v: every tip rode on the paper either side of a groove, and
    # a fine point drawn along one indented at 0.7 laid nothing on its
    # floor; a tip rests on the highest of the paper it covers)
    assert 0.5 < _on_floor(0.7, 3.0, pencils.HB, 1.0) < 0.95
    assert _on_floor(0.7, 3.0, pencils.B2, 3.0) > 0.3
    # one wider than the groove still rides over it
    assert _on_floor(0.7, 3.0, pencils.B2, 4.0) < 0.01
    # and what is narrower than a wider groove goes down into it
    assert _on_floor(0.7, 6.0, pencils.B2, 4.0) > 0.25
    assert _on_floor(0.7, 6.0, pencils.B2, 6.0) < 0.01
    # shaded across a groove, a fine point goes into it and a broad one
    # skips it
    fine, broad = _sheet(), _sheet()
    for sheet, width in ((fine, 2.0), (broad, 6.0)):
        sheet.indent(LINE, 0.7)
        sheet.shade(BOX, pencils.B2, 0.5, width, angle=60)
    assert _in_groove(fine) > 0.2 and _in_groove(broad) < 0.01


def test_the_floor_of_a_groove_is_level_and_bites_less():
    """The point that pressed a groove left its floor flat, as a hard tip
    pressed firmly leaves the tooth: a point drawn along it touches all of
    it, and the paper bites it less."""
    grooved, plain = _along_groove(0.7, 3.0, pencils.HB, 1.0)
    floor, paper = grooved.cover[GROOVE].mean(), plain.cover[GROOVE].mean()
    assert floor > 0.95 and paper < floor - 0.2
    per_floor = grooved.pigment()[GROOVE].mean() / floor
    assert per_floor < 0.6 * plain.pigment()[GROOVE].mean() / paper


def test_a_groove_shows_as_a_line_of_shadow_under_a_raking_light():
    sheet, blank = _sheet(), _sheet()
    sheet.indent(LINE, 0.8)
    # nothing is drawn: flat, the sheet is as it was
    assert np.array_equal(sheet.render(lighting=False), blank.render(lighting=False))
    lit, bare = (
        _light(sheet.render(light=RAKING), slice(None)),
        _light(blank.render(light=RAKING), slice(None)),
    )
    change = (lit - bare)[72:79, 50:150].mean(axis=1)
    # the side of the groove turned from the lamp (it shines from above
    # the top of the sheet) is in shadow, the side turned to it is lit
    assert change.min() < -0.1 and change.max() > 0.02
    assert np.argmin(change) < np.argmax(change)
    assert np.abs((lit - bare)[:60]).max() < 1e-6
    # under the default lamp, fairly high, it shows far less
    soft = _light(sheet.render(), slice(None)) - _light(blank.render(), slice(None))
    assert 0.0 < -soft[72:79, 50:150].mean(axis=1).min() < -0.5 * change.min()


def test_the_floor_of_a_groove_is_as_far_down_as_it_was_pressed():
    paper = papers.CARTRIDGE
    sheet = _sheet()
    top = TOOTH_HEIGHT * paper.grain * paper.depth
    assert float(sheet.surface_height().max()) == pytest.approx(top, rel=1e-3)
    sheet.indent(LINE, 0.25)  # (half a tooth deep: only the peaks are pushed down)
    depth = 0.25 * eng.INDENT
    height = sheet.surface_height()[75, 50:150]
    # (to within the ripple the round point leaves as it goes: its samples)
    assert height.max() == pytest.approx(top * (1.0 - depth / paper.depth), rel=0.02)
    low = sheet.tooth[75, 50:150] < 0.98 - depth
    assert low.any() and np.allclose(height[low], top * sheet.tooth[75, 50:150][low])
    sheet.indent(LINE, 1.0)  # below the valleys: a level floor
    floor = sheet.surface_height()[75, 50:150]
    assert np.allclose(floor, top * (1.0 - eng.INDENT), rtol=0.02)
    assert np.allclose(sheet.surface_height(2.0), 2.0 * sheet.surface_height())
    # a level paper has no relief but its grooves
    level = _sheet(Paper(depth=0.0))
    assert not level.surface_height().any()
    level.indent(LINE, 0.5)
    height = level.surface_height()
    assert height.max() == 0.0 and not height[:70].any()
    lowest = -0.5 * eng.INDENT * TOOTH_HEIGHT * level.paper.grain
    assert float(height.min()) == pytest.approx(lowest, rel=1e-5)


def test_what_was_drawn_goes_down_with_the_paper_and_no_eraser_takes_a_groove_out():
    sheet = _shaded(pencils.B2, 0.5)
    before, tone = sheet.pigment().copy(), sheet.reflectance().copy()
    sheet.indent(LINE, 0.8)
    assert np.array_equal(sheet.pigment(), before)
    assert np.array_equal(sheet.reflectance(), tone)
    groove = sheet.dent.copy()
    sheet.rub(BOX, tools.ERASER, 1.0)
    sheet.rub(BOX, tools.STUMP, 0.8)
    sheet.fix()
    assert np.array_equal(sheet.dent, groove)
    # shaded again, the line comes out light in the tone
    sheet.shade(BOX, pencils.B2, 0.5, angle=-30)
    assert _luminance(sheet, GROOVE) > 1.3 * _luminance(sheet, AROUND)


def test_a_copy_has_its_own_grooves_and_indenting_repeats():
    first, second = _sheet(), _sheet()
    for sheet in (first, second):
        sheet.indent([(30, 40), (100, 90), (170, 50)], (0.3, 0.9))
    assert np.array_equal(first.dent, second.dent)
    copy = first.copy()
    copy.indent(LINE, 0.9)
    assert np.array_equal(first.dent, second.dent)
    assert copy.dent.sum() > first.dent.sum()


def test_indent_checks_what_it_is_given():
    sheet = _sheet()
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="width"):
            sheet.indent(LINE, 0.5, bad)
    with pytest.raises(ValueError, match="pressure"):
        sheet.indent(LINE, float("nan"))
    assert not sheet.dent.any()
    # a groove off the edge of the sheet is no trouble, nor is a single point
    sheet.indent([(-20, 5), (30, -5)], 0.5)
    sheet.indent([(100.5, 100.5)], 2.0, 5.0)  # (the pressure is clipped to 1)
    assert float(sheet.dent[100, 100]) == pytest.approx(eng.INDENT)


# --------------------------------------------------------------------------
# the ball of a pen


def test_a_ballpoint_leaves_a_groove_the_deeper_the_firmer():
    deep = []
    for pressure in (0.2, 0.5, 1.0):
        sheet = _sheet()
        sheet.stroke(LINE, pens.BLUE, pressure)
        deep.append(float(sheet.dent.max()))
        assert not sheet.dent[:70].any() and not sheet.dent[81:].any()
    assert deep == pytest.approx([p * eng.BALL_DENT for p in (0.2, 0.5, 1.0)], rel=1e-5)
    assert eng.BALL_DENT < eng.INDENT  # (a stylus is made for it)


def test_a_pencil_shades_over_a_light_pen_line_and_skips_a_firm_one():
    filled = []
    for pressure in (0.2, 0.6, 1.0):
        sheet = _sheet()
        sheet.stroke(LINE, pens.BLUE, pressure)
        sheet.shade(BOX, pencils.B2, 0.3, angle=60)
        filled.append(_in_groove(sheet))
    # (the floor the ball pressed flat bites the pencil less, fusain-d7v:
    # 0.89 where it was 0.91, 0.26 where 0.42)
    assert filled[0] > 0.8
    assert 0.15 < filled[1] < 0.5
    assert filled[2] < 0.1


def test_the_ball_goes_down_into_a_groove():
    """It is as small as what made it: it writes in a groove as on the
    paper beside, and one pen line crosses another as if it were flat."""
    plain, grooved = _sheet(), _sheet()
    grooved.indent(LINE, 1.0, 6.0)
    for sheet in (plain, grooved):
        sheet.stroke(LINE, pens.BLUE, 0.4)
        sheet.stroke([(100, 20), (100, 130)], pens.BLUE, 0.4)
    assert plain.ink is not None and grooved.ink is not None
    assert np.array_equal(plain.ink, grooved.ink)
