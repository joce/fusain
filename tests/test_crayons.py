"""Wax crayons and oil pastels: what the paper holds of them, how they layer
and what takes over their wax, measured."""

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, charcoal, crayons, papers, pastels, pencils, tools
from fusain._engine import IV, RESIST, RESIST_DRY, resist_of

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
RED, BLUE = "#c0392b", "#2456a6"
WHITE, BLACK = "#f4f1ea", "#161616"
WIDTH = 9.0


def _sheet() -> Sheet:
    return Sheet(200, 150, paper=papers.CARTRIDGE, seed=1)


def _layers(sheet, medium, count=3, pressure=0.8) -> float:
    return sum(
        sheet.shade(BOX, medium, pressure, WIDTH, angle=30 + 50 * layer)
        for layer in range(count)
    )


def _luminance(sheet, where=INSIDE) -> float:
    spectrum = sheet.reflectance()[where].astype(np.float64).mean(axis=(0, 1))
    return float(sp.reflectance_to_linear_rgb(spectrum) @ sp.LUMINANCE)


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


def _dust(sheet) -> float:
    return float(sheet.loose[..., IV].sum() + sheet.under[..., IV].sum())


def _carried_out(sheet) -> float:
    """What a finger rubbed across the patch carries past its edge."""
    before = float(sheet.pigment()[:, 185:].sum())
    for k in range(3):
        sheet.smudge([(40, 60 + 8 * k), (198, 60 + 8 * k)], tools.FINGER, 0.7)
    return float(sheet.pigment()[:, 185:].sum()) - before


def _white_over_black(make) -> tuple[float, float, float]:
    """Luminance of a black patch, of a white layer over it, and of the white
    alone."""
    sheet, alone = _sheet(), _sheet()
    sheet.shade(BOX, make(BLACK), 0.7, WIDTH)
    black = _luminance(sheet)
    sheet.shade(BOX, make(WHITE), 0.8, WIDTH, angle=-30)
    alone.shade(BOX, make(WHITE), 0.8, WIDTH)
    return black, _luminance(sheet), _luminance(alone)


# --------------------------------------------------------------------------
# what they are


def test_names():
    assert crayons.wax(RED).name == "#c0392b crayon"
    assert crayons.wax(RED, "scarlet").name == "scarlet"
    assert pastels.oil(RED).name == "#c0392b oil pastel"
    assert pastels.oil(RED, "scarlet").pigment.name == "scarlet"


@pytest.mark.parametrize("make", [crayons.wax, pastels.oil])
def test_what_leaves_the_stick_is_on_the_paper(make):
    sheet = _sheet()
    laid = _layers(sheet, make(RED))
    laid += _layers(sheet, make(BLUE), 2, 1.0)
    assert _total(sheet) == pytest.approx(laid, rel=1e-6)


# --------------------------------------------------------------------------
# wax crayon


def test_the_paper_holds_all_of_a_crayon():
    sheet = _sheet()
    _layers(sheet, crayons.wax(RED), 1, 0.6)
    assert _dust(sheet) == 0.0
    total = _total(sheet)
    assert _carried_out(sheet) < 1e-3 * total


def test_a_crayon_shows_the_grain_until_it_is_pressed_hard():
    covers = []
    for pressure in (0.3, 0.6, 1.0):
        sheet = _sheet()
        sheet.shade(BOX, crayons.wax(RED), pressure, WIDTH)
        covers.append(float(sheet.cover[INSIDE].mean()))
    assert covers[0] < 0.5 < covers[1] < 0.8 < covers[2]


def test_a_crayon_is_waxy_layers_take_less_and_a_firm_one_polishes():
    sheet = _sheet()
    red = crayons.wax(RED)
    takes = [
        sheet.shade(BOX, red, 0.6, WIDTH, angle=30 + 50 * layer) for layer in range(3)
    ]
    assert takes[1] < 0.7 * takes[0]
    assert takes[2] < 0.7 * takes[1]
    polished = float(sheet.press[INSIDE].mean())
    sheet.shade(BOX, red, 1.0, WIDTH, angle=-20)
    assert float(sheet.press[INSIDE].mean()) > polished + 0.2
    assert float(sheet.cover[INSIDE].mean()) > 0.95


def test_a_white_crayon_does_not_cover_a_black_one():
    black, over, alone = _white_over_black(crayons.wax)
    assert alone > 0.8
    assert over < 1.2 * black
    assert over < 0.3 * alone


# --------------------------------------------------------------------------
# oil pastel


def test_a_white_oil_pastel_half_covers_a_black_one():
    black, over, alone = _white_over_black(pastels.oil)
    assert black < 0.1
    assert 4 * black < over < 0.6 * alone
    # a soft pastel covers better, a crayon worse
    assert _white_over_black(pastels.soft)[1] > 1.3 * over
    assert _white_over_black(crayons.wax)[1] < 0.5 * over


def test_a_finger_smears_oil_pastel_less_than_soft_pastel_more_than_crayon():
    carried = []
    for make in (crayons.wax, pastels.oil, pastels.soft):
        sheet = _sheet()
        sheet.shade(BOX, make(RED), 0.6, WIDTH)
        carried.append(_carried_out(sheet))
    assert 50 * carried[0] < carried[1] < 0.7 * carried[2]


def test_the_paper_holds_under_half_of_an_oil_pastel_at_once():
    sheet = _sheet()
    laid = sheet.shade(BOX, pastels.oil(RED), 0.5, WIDTH)
    assert 0.4 * laid < _dust(sheet) < 0.65 * laid


def test_oil_pastel_goes_on_thick():
    thick, thin = _sheet(), _sheet()
    assert _layers(thick, pastels.oil(RED)) > 2.5 * _layers(thin, crayons.wax(RED))
    assert float(thick.cover[INSIDE].mean()) > 0.99


# --------------------------------------------------------------------------
# what takes over wax


def test_resist_depends_on_what_is_laid():
    pencil, crayon, oil = pencils.coloured(RED), crayons.wax(RED), pastels.oil(RED)
    # a waxy medium that is not greasy: as before
    assert resist_of(pencil.softness, pencil.wax) == RESIST
    assert resist_of(crayon.softness, crayon.wax) == RESIST
    assert resist_of(pencils.BLENDER.softness, pencils.BLENDER.wax) == RESIST
    # a dry one hardly takes, however soft
    for dry in (pencils.B4, charcoal.VINE, pastels.soft(RED)):
        assert resist_of(dry.softness, dry.wax) == RESIST_DRY
    # a greasy one sticks
    assert resist_of(oil.softness, oil.wax) < 0.6 * RESIST
    assert resist_of(1.0, 1.0) == pytest.approx(RESIST * 0.2)


def _share_taken(base, medium) -> float:
    """What a layer of the medium takes over ``base``, as a share of what it
    takes on bare paper."""
    over, bare = base.copy(), _sheet()
    return over.shade(BOX, medium, 0.5, WIDTH, angle=-20) / bare.shade(
        BOX, medium, 0.5, WIDTH, angle=-20
    )


def test_dry_media_hardly_take_over_crayon_and_oil_pastel_does():
    base = _sheet()
    _layers(base, crayons.wax(RED))
    dry = [
        _share_taken(base, medium)
        for medium in (pencils.B4, charcoal.VINE, pastels.soft(BLUE))
    ]
    assert max(dry) < 0.2
    crayon = _share_taken(base, crayons.wax(BLUE))
    assert max(dry) < crayon < 0.3
    assert _share_taken(base, pastels.oil(BLUE)) > 0.6


def test_nearly_nothing_dry_takes_over_oil_pastel():
    base = _sheet()
    _layers(base, pastels.oil(RED))
    assert _share_taken(base, pencils.B4) < 0.05
    assert _share_taken(base, charcoal.VINE) < 0.05
