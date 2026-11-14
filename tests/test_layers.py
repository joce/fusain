"""What lies over what: a waxy medium laid over dust, the paste of an oil
pastel, and dry media over a line of wax, measured."""

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, Smudger, charcoal, crayons, papers, pastels, pencils, tools
from fusain._engine import IP, IV, PASTY, paste_of

BOX = (20, 20, 180, 130)
INSIDE = (slice(50, 100), slice(50, 150))
RED, YELLOW, BLUE = "#c0392b", "#f2c40f", "#2456a6"
WIDTH = 9.0


def _sheet() -> Sheet:
    return Sheet(200, 150, paper=papers.CARTRIDGE, seed=1)


def _drawn(medium, pressure=0.5) -> Sheet:
    sheet = _sheet()
    sheet.shade(BOX, medium, pressure, WIDTH)
    return sheet


def _rgb(sheet, where=INSIDE) -> np.ndarray:
    spectrum = sheet.reflectance()[where].astype(np.float64).mean(axis=(0, 1))
    return sp.reflectance_to_linear_rgb(spectrum)


def _luminance(sheet, where=INSIDE) -> float:
    return float(_rgb(sheet, where) @ sp.LUMINANCE)


def _yellow(sheet) -> float:
    """How yellow the patch is: its red and green over its blue."""
    r, g, b = _rgb(sheet)
    return float((r + g) / 2 - b)


def _total(sheet) -> float:
    return float(sheet.pigment().sum(dtype=np.float64))


def _lying(sheet) -> float:
    return float((sheet.under[..., IV] + sheet.loose[..., IV]).sum(dtype=np.float64))


def _sum(array) -> float:
    return float(array.sum(dtype=np.float64))


def _carried_out(sheet) -> float:
    """What a finger rubbed across the patch carries past its edge."""
    before = float(sheet.pigment()[:, 185:].sum())
    for k in range(3):
        sheet.smudge([(40, 60 + 8 * k), (198, 60 + 8 * k)], tools.FINGER, 0.7)
    return float(sheet.pigment()[:, 185:].sum()) - before


def _kneaded_share(medium, pressure=0.8) -> float:
    """The share of a layer of the medium on bare paper that a kneaded
    eraser rubbed over it lifts."""
    sheet = _sheet()
    laid = sheet.shade(BOX, medium, pressure, WIDTH, angle=-30)
    return sheet.rub(BOX, tools.KNEADED, 0.6) / laid


def _bonded_within_bounds(sheet) -> bool:
    """Nothing is bonded but what lies there, layer by layer."""
    if sheet.bonded.min() < 0.0:
        return False
    for which, layer in enumerate((sheet.under, sheet.loose)):
        bonded = sheet.bonded[:, :, which].sum(axis=-1)
        if (bonded > layer[..., IV] * (1 + 1e-4) + 1e-5).any():
            return False
    return True


WAXY = [crayons.wax(YELLOW), pencils.coloured(YELLOW), pastels.oil(YELLOW)]
DUSTS = [pastels.soft(BLUE), charcoal.VINE, charcoal.COMPRESSED]


# --------------------------------------------------------------------------
# a waxy medium over dust (fusain-ck0)


def test_on_bare_paper_the_paper_holds_a_crayon_and_nothing_is_bonded():
    sheet = _drawn(crayons.wax(YELLOW), 0.8)
    assert _lying(sheet) == 0.0
    assert not sheet.bonded.any()
    assert sheet.fixed.shape == sheet.film.shape == sheet.paste.shape == (150, 200, 2)


@pytest.mark.parametrize("dust", DUSTS)
def test_a_crayon_laid_over_dust_lies_on_it(dust):
    base = _drawn(dust)
    over = base.copy()
    laid = over.shade(BOX, crayons.wax(YELLOW), 0.8, WIDTH, angle=-30)
    film = _sum(over.film)
    # where there was dust it is a film over it, on top (but for what its
    # own later strokes buried); the paper holds the rest
    assert film > 0.4 * laid
    assert _sum(over.film[..., 1]) > 0.8 * film
    # and it shows: the patch turns yellow
    assert _yellow(over) > _yellow(base) + 0.25
    assert _bonded_within_bounds(over)


def test_a_light_crayon_over_charcoal_lightens_it():
    # (it used to lie under the charcoal, and only darkened it)
    for dust in (charcoal.VINE, charcoal.COMPRESSED):
        base = _drawn(dust)
        over = base.copy()
        over.shade(BOX, crayons.wax(YELLOW), 0.8, WIDTH, angle=-30)
        assert _luminance(over) > 1.05 * _luminance(base)


def test_a_coloured_pencil_over_pastel_shows_and_a_graphite_one_lays_no_film():
    base = _drawn(pastels.soft(BLUE))
    over = base.copy()
    over.shade(BOX, pencils.coloured(YELLOW), 0.8, WIDTH, angle=-30)
    assert _yellow(over) > _yellow(base) + 0.15
    assert _sum(over.film) > 0
    dry = base.copy()
    dry.shade(BOX, pencils.B4, 0.8, WIDTH, angle=-30)
    assert not dry.film.any() and not dry.paste.any() and not dry.fixed.any()


def test_the_thinner_the_dust_the_more_of_the_crayon_the_paper_holds():
    shares = []
    for pressure in (0.1, 0.25, 0.6):
        sheet = _drawn(charcoal.VINE, pressure)
        laid = sheet.shade(BOX, crayons.wax(YELLOW), 0.8, WIDTH, angle=-30)
        shares.append(_sum(sheet.film) / laid)
    assert shares[0] < shares[1] < shares[2]
    assert shares[0] < 0.5 < shares[2]


@pytest.mark.parametrize("dust", DUSTS)
def test_a_crayon_binds_part_of_the_dust_it_is_laid_over(dust):
    base = _drawn(dust)
    over = base.copy()
    laid = over.shade(BOX, crayons.wax(YELLOW), 0.8, WIDTH, angle=-30)
    # some of the dust is set, and not all of it
    assert 0.1 * _lying(base) < _sum(over.fixed) < 0.8 * _lying(base)
    # so an eraser lifts less than it would of the dust and of the crayon
    # apart, and a finger moves less, though there is more to move
    apart = base.copy().rub(BOX, tools.KNEADED, 0.6)
    apart += _kneaded_share(crayons.wax(YELLOW)) * laid
    assert over.copy().rub(BOX, tools.KNEADED, 0.6) < 0.9 * apart
    assert _carried_out(over.copy()) < 0.97 * _carried_out(base.copy())


def test_dust_laid_over_a_film_buries_it_with_what_it_set():
    sheet = _drawn(pastels.soft(BLUE), 0.3)
    sheet.shade(BOX, crayons.wax(YELLOW), 0.8, WIDTH, angle=-30)
    film, fixed = _sum(sheet.film), _sum(sheet.fixed)
    assert _sum(sheet.film[..., 1]) > 0.8 * film
    sheet.shade(BOX, pastels.oil(RED), 0.9, WIDTH, angle=80)
    assert _sum(sheet.film[..., 0]) > 0.7 * film
    assert _sum(sheet.fixed) >= fixed * (1 - 1e-5)
    assert _bonded_within_bounds(sheet)


@pytest.mark.parametrize("over", WAXY)
@pytest.mark.parametrize("dust", DUSTS)
def test_what_is_laid_over_dust_is_on_the_sheet_or_on_the_tool(over, dust):
    sheet = _sheet()
    laid = sheet.shade(BOX, dust, 0.5, WIDTH)
    laid += sheet.shade(BOX, over, 0.8, WIDTH, angle=-30)
    laid += sheet.shade(BOX, dust, 0.5, WIDTH, angle=70)
    assert _total(sheet) == pytest.approx(laid, rel=1e-5)
    taken = sheet.rub((30, 30, 170, 70), tools.FINGER, 0.7)
    taken += sheet.rub((30, 60, 170, 100), tools.KNEADED, 0.7)
    taken += sheet.erase([(30, 110), (170, 110)], tools.ERASER, 0.9)
    stump = Smudger(tools.STUMP)
    taken += sheet.rub((30, 30, 170, 120), stump, 0.6, angle=90)
    assert stump.load == pytest.approx(stump.clean())
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)
    assert _bonded_within_bounds(sheet)


# --------------------------------------------------------------------------
# the paste of an oil pastel (fusain-5cm)


def test_only_a_greasy_medium_leaves_paste():
    oil, crayon = pastels.oil(RED), crayons.wax(RED)
    assert paste_of(oil.softness, oil.wax) == 1.0
    assert paste_of(crayon.softness, crayon.wax) == 0.0
    pencil = pencils.coloured(RED)
    assert paste_of(pencil.softness, pencil.wax) == 0.0
    for dry in (pencils.B8, charcoal.VINE, pastels.soft(RED)):
        assert paste_of(dry.softness, dry.wax) == 0.0
    # (half as greasy as it gets is paste already)
    assert paste_of(0.6 + 0.4 * PASTY, 1.0) == pytest.approx(1.0)
    assert 0.0 < paste_of(0.7, 1.0) < 1.0


def test_what_the_paper_does_not_hold_of_an_oil_pastel_is_paste():
    sheet = _drawn(pastels.oil(RED), 0.6)
    lying = _lying(sheet)
    assert _sum(sheet.paste) + _sum(sheet.film) == pytest.approx(lying, rel=1e-4)
    assert _sum(sheet.paste) > 0.7 * lying
    assert _sum(sheet.fixed) < 1e-5 * lying
    # where a soft pastel leaves dust
    assert not _drawn(pastels.soft(RED), 0.6).bonded.any()


def test_a_kneaded_eraser_does_not_lift_oil_pastel_as_it_lifts_dust():
    shares = {}
    for name, medium in (
        ("oil", pastels.oil(RED)),
        ("crayon", crayons.wax(RED)),
        ("charcoal", charcoal.VINE),
        ("pastel", pastels.soft(RED)),
    ):
        sheet = _sheet()
        laid = sheet.shade(BOX, medium, 0.6, WIDTH)
        tone = _luminance(sheet)
        shares[name] = sheet.rub(BOX, tools.KNEADED, 0.6) / laid
        if name == "oil":  # it hardly lightens
            assert _luminance(sheet) < 1.1 * tone
    # (it used to lift over half of it)
    assert shares["oil"] < 0.25
    assert shares["oil"] < 0.4 * min(shares["charcoal"], shares["pastel"])
    # it comes off as a crayon does, which the paper holds
    assert shares["oil"] == pytest.approx(shares["crayon"], rel=0.15)


def test_a_vinyl_eraser_takes_half_of_an_oil_pastel_and_a_knife_scrapes_it_off():
    sheet = _sheet()
    laid = sheet.shade(BOX, pastels.oil(RED), 0.6, WIDTH)
    scraped = sheet.copy()
    # (the eraser slides on the wax in it; it used to take over 0.7)
    assert 0.4 * laid < sheet.rub(BOX, tools.ERASER, 0.6) < 0.65 * laid
    assert scraped.rub(BOX, tools.KNIFE, 0.6, 10) > 0.9 * laid


def test_what_a_finger_smears_of_an_oil_pastel_is_still_paste():
    sheet = _drawn(pastels.oil(RED), 0.6)
    beyond = (slice(None), slice(188, None))
    there = _sum(sheet.pigment()[beyond])
    assert there < 1.0 and _sum(sheet.paste[beyond]) <= there
    finger = Smudger(tools.FINGER)
    for k in range(3):
        sheet.smudge([(40, 60 + 8 * k), (198, 60 + 8 * k)], finger, 0.7)
    # the finger carries paste, and lays paste
    assert finger._carried[:, IP].sum() > 0.9 * finger.load > 0
    smeared = _sum(sheet.pigment()[beyond])
    assert smeared > 10.0 + there
    # (and presses part of it into the paper; none of it is dust)
    lying = _sum(sheet.under[beyond][..., IV] + sheet.loose[beyond][..., IV])
    assert lying > 0.5 * smeared
    assert _sum(sheet.paste[beyond]) > 0.95 * lying
    # so a kneaded eraser does not lift the smear either
    taken = sheet.rub((189, 40, 200, 100), tools.KNEADED, 0.7, width=6)
    assert taken < 0.3 * smeared
    assert _bonded_within_bounds(sheet)


def test_a_finger_stirs_the_paste_underneath_up_into_the_paste_on_top():
    sheet = _drawn(pastels.oil(BLUE), 0.7)
    sheet.shade(BOX, pastels.oil(YELLOW), 0.7, WIDTH, angle=-30)
    under = _sum(sheet.paste[..., 0])
    assert under > 0
    rubbed = sheet.copy()
    rubbed.rub(BOX, tools.FINGER, 0.7)
    assert _sum(rubbed.paste[..., 0]) < 0.5 * under
    assert _bonded_within_bounds(rubbed)


def test_an_oil_pastel_turns_the_dust_it_is_laid_over_to_paste():
    base = _drawn(charcoal.VINE)
    dust = _lying(base)
    over = base.copy()
    laid = over.shade(BOX, pastels.oil(YELLOW), 0.8, WIDTH, angle=-30)
    assert _lying(over) - _sum(over.bonded) < 0.5 * dust  # (the dust left)
    # so an eraser lifts far less than it would of the two apart
    apart = base.copy().rub(BOX, tools.KNEADED, 0.6)
    apart += _kneaded_share(pastels.oil(YELLOW)) * laid
    assert over.rub(BOX, tools.KNEADED, 0.6) < 0.75 * apart


def test_fixative_sets_paste_and_leaves_a_film_a_film():
    sheet = _drawn(charcoal.VINE)
    sheet.shade(BOX, crayons.wax(YELLOW), 0.8, WIDTH, angle=-30)
    sheet.shade((20, 20, 180, 70), pastels.oil(RED), 0.6, WIDTH, angle=80)
    film, lying = sheet.film.copy(), _lying(sheet)
    assert sheet.paste.any()
    smeared = _carried_out(sheet.copy())
    was_set = _sum(sheet.fixed)
    newly = sheet.fix()
    assert newly == pytest.approx(lying - _sum(film) - was_set, rel=1e-4)
    assert not sheet.paste.any()
    assert np.array_equal(sheet.film, film)
    assert _sum(sheet.bonded) == pytest.approx(lying, rel=1e-4)
    assert _carried_out(sheet) < 0.05 * smeared
    half = _drawn(pastels.oil(RED))
    paste = _sum(half.paste)
    half.fix(0.5)
    assert _sum(half.paste) == pytest.approx(0.5 * paste, rel=1e-5)
    assert _bonded_within_bounds(half)


# --------------------------------------------------------------------------
# dry media over one line of wax (fusain-8nq)

LINE = (slice(73, 77), slice(50, 150))
AROUND = (slice(40, 60), slice(50, 150))


def _line(pressure=0.8) -> Sheet:
    sheet = _sheet()
    sheet.stroke([(30, 75), (170, 75)], crayons.wax(YELLOW), pressure, 8)
    return sheet


@pytest.mark.parametrize(
    "medium, pressure",
    [(charcoal.VINE, 0.5), (charcoal.COMPRESSED, 0.4), (pencils.B4, 0.5)],
)
def test_a_dry_medium_shaded_over_one_firm_crayon_line_skips_it(medium, pressure):
    line = _line()
    tone = _luminance(line, LINE)
    over = line.copy()
    over.shade(BOX, medium, pressure, WIDTH, angle=60)
    on_line = float((over.pigment() - line.pigment())[LINE].mean())
    around = float(over.pigment()[AROUND].mean())
    # next to none of it takes on the line (four tenths of it used to)
    assert on_line < 0.12 * around
    # so the line keeps its tone, and stands out light against the shading
    # (it used to come out as dark as the paper around it)
    assert _luminance(over, LINE) > 0.85 * tone
    assert _luminance(over, LINE) > 1.15 * _luminance(over, AROUND)


def test_the_firmer_the_line_the_less_takes_over_it():
    taken = []
    for pressure in (0.3, 0.6, 1.0):
        line = _line(pressure)
        over = line.copy()
        over.shade(BOX, charcoal.VINE, 0.5, WIDTH, angle=60)
        taken.append(float((over.pigment() - line.pigment())[LINE].mean()))
    assert taken[0] > 2 * taken[1] > 4 * taken[2]


def test_a_crayon_still_takes_over_one_line_of_crayon():
    # (to a waxy medium the film is not yet as slick as it gets)
    line = _line()
    over, bare = line.copy(), _sheet()
    path = [(30, 75), (170, 75)]
    again = over.stroke(path, crayons.wax(BLUE), 0.8, 8)
    assert again > 0.3 * bare.stroke(path, crayons.wax(BLUE), 0.8, 8)
