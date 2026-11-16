"""What lies over what: a waxy medium laid over dust, the paste of an oil
pastel, dry media over a line of wax, and a dry medium over a bed of dust,
measured."""

import itertools

import numpy as np
import pytest
from murex import spectral as sp

from fusain import Sheet, Smudger, charcoal, crayons, papers, pastels, pencils, tools
from fusain._engine import IP, IV, IW, IX, PASTY, paste_of

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


def test_a_coloured_pencil_over_pastel_shows_and_a_graphite_one_binds_none_of_it():
    base = _drawn(pastels.soft(BLUE))
    over = base.copy()
    over.shade(BOX, pencils.coloured(YELLOW), 0.8, WIDTH, angle=-30)
    assert _yellow(over) > _yellow(base) + 0.15
    assert _sum(over.film) > 0 and _sum(over.fixed) > 0
    # a graphite pencil lies on the pastel too (fusain-wo4), but it has no
    # wax: it sets none of the dust, and leaves no paste
    dry = base.copy()
    dry.shade(BOX, pencils.B4, 0.8, WIDTH, angle=-30)
    assert dry.film.any() and not dry.paste.any() and not dry.fixed.any()


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

    def moved(sheet) -> float:
        """How much pigment a finger rubbed across the patch shifts in it."""
        before = sheet.pigment().astype(np.float64)
        _carried_out(sheet)
        return float(np.abs(sheet.pigment() - before)[:, :181].sum()) / 2

    # (what it does take up, the wax does not take back on the way, so
    # about as much as before reaches the bare paper past the edge)
    assert moved(over.copy()) < 0.8 * moved(base.copy())


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
    # (but for what it has rubbed off what the paper holds, which it keeps:
    # paper that carries pigment takes no dust back from it)
    assert finger._carried[:, IP].sum() > 0.85 * finger.load > 0
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
    # (it used to come out as dark as the paper around it; vine charcoal
    # leaves it from 1.13 to 1.20 times as light over six sheets)
    assert _luminance(over, LINE) > 0.85 * tone
    assert _luminance(over, LINE) > 1.1 * _luminance(over, AROUND)


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


# --------------------------------------------------------------------------
# a dry medium over a bed of dust (fusain-wo4)

PALE = "#f1e9d8"
#: Lines across the patch. (One line of a tip varies by a quarter from the
#: next, as its lead bears: several are drawn, and their mean is taken.)
LINES = [[(30, y), (170, y)] for y in range(36, 124, 14)]
ALONG = LINES[3]
#: Sticks that hold together, whose mark the paper holds for the most part.
FIRM = [pencils.B2, pencils.HB, charcoal.PENCIL]
DRY = [*FIRM, pencils.B6, charcoal.COMPRESSED, charcoal.VINE]
DRY += [pastels.soft(BLUE), pastels.hard(BLUE)]


def _bed(fixed=False, layers=2, pressure=0.6, dust=None) -> Sheet:
    """Layers of a pale soft pastel (or of ``dust``), loose or fixed."""
    sheet = _sheet()
    dust = pastels.soft(PALE) if dust is None else dust
    for layer in range(layers):
        sheet.shade(BOX, dust, pressure, 14.0, angle=30 - 60 * layer)
    if fixed:
        sheet.fix()
    return sheet


def _on(path) -> tuple[slice, slice]:
    """The middle of a line 4 wide along ``path``, away from its ends."""
    row = int(path[0][1])
    return (slice(row - 1, row + 1), slice(60, 140))


def _pencilled(sheet, medium, pressure=0.6) -> tuple[float, float]:
    """Draw the lines across the patch. Returns the volume they laid, and
    how dark they are against the ground they were drawn on: the share of
    its light they took, on average."""
    laid, dark = 0.0, 0.0
    for path in LINES:
        ground = _luminance(sheet, _on(path))
        laid += sheet.stroke(path, medium, pressure, 4.0)
        dark += (1.0 - _luminance(sheet, _on(path)) / ground) / len(LINES)
    return laid, dark


def _graphite(sheet, where=(slice(None), slice(None))) -> float:
    """The graphite at these pixels, wherever it lies (as its sheen)."""
    return sum(
        _sum(layer[where][..., IX]) for layer in (sheet.held, sheet.under, sheet.loose)
    )


def _nothing_negative(sheet) -> bool:
    arrays = (sheet.held, sheet.under, sheet.loose, sheet.bonded, sheet.cover)
    return all(float(array.min()) >= 0.0 for array in arrays)


@pytest.mark.parametrize("medium", FIRM)
@pytest.mark.parametrize("fixed", [False, True])
def test_a_pencil_line_drawn_over_pastel_shows_on_it(medium, fixed):
    on_paper, dark_on_paper = _pencilled(_sheet(), medium)
    base = _bed(fixed)
    over = base.copy()
    laid, dark = _pencilled(over, medium)
    # a tooth full of pastel dust took next to nothing of it (a fiftieth of
    # what bare paper takes of a 2B, a hundredth of an HB; a seventh and a
    # twentieth where the pastel was fixed), and that under the dust: the
    # line was a hundredth as dark as on bare paper
    assert laid > 0.55 * on_paper
    assert dark > 0.5 * dark_on_paper
    # what the paper would have held of it is a film on the pastel, on top
    film = _sum(over.film) - _sum(base.film)
    assert film == pytest.approx(medium.binding * laid, rel=0.02)
    assert _sum(over.film[..., 1]) == pytest.approx(film, rel=1e-4)
    # it has no wax: it sets none of the dust, and what was set stays so
    assert _sum(over.fixed) == pytest.approx(_sum(base.fixed), rel=1e-5)
    assert not over.paste.any()
    assert _total(over) == pytest.approx(_total(base) + laid, rel=1e-6)


@pytest.mark.parametrize("medium", FIRM)
def test_a_pencil_shaded_over_pastel_is_a_little_weaker_than_on_bare_paper(medium):
    def shaded(sheet) -> tuple[float, float]:
        ground = _luminance(sheet)
        laid = sheet.shade((40, 40, 160, 110), medium, 0.5, 6.0)
        return laid, 1.0 - _luminance(sheet) / ground

    on_paper, dark_on_paper = shaded(_sheet())
    for base in (_bed(layers=1, pressure=0.3), _bed(layers=1), _bed(), _bed(True)):
        laid, dark = shaded(base)
        # dust bites less than paper, loose or set, but it is no longer in
        # the way (over two layers a 2B laid a hundredth of this, and over
        # a light one half)
        assert 0.7 * on_paper < laid < 0.95 * on_paper
        assert 0.5 * dark_on_paper < dark < 0.95 * dark_on_paper


def _dust_under(sheet, path) -> float:
    """The pastel that lies along the middle of ``path``: what is in the two
    layers there and is not graphite."""
    row = int(path[0][1])
    where = (slice(row - 2, row + 2), slice(60, 140))
    lying = _sum(sheet.under[where][..., IV]) + _sum(sheet.loose[where][..., IV])
    lead = _sum(sheet.under[where][..., IX]) + _sum(sheet.loose[where][..., IX])
    return lying - lead / pencils.B2.sheen


def test_a_pencil_line_on_loose_pastel_comes_away_with_it_and_on_fixed_stays():
    left, smeared, kept, dust_left = {}, {}, {}, {}
    row = ALONG[0][1]
    below = (slice(row + 6, None), slice(None))
    beside = LINES[5]  # (a path with no line on it, clear of the eraser's)
    for name, sheet in (("paper", _sheet()), ("loose", _bed()), ("fixed", _bed(True))):
        sheet.stroke(ALONG, pencils.B2, 0.6, 4.0)
        there, on_line = _graphite(sheet), _graphite(sheet, _on(ALONG))
        erased = sheet.copy()
        for path in (ALONG, beside):
            erased.erase(path, tools.KNEADED, 0.6, 10.0)
        left[name] = _graphite(erased) / there
        if name != "paper":
            dust_left[name] = _dust_under(erased, beside) / _dust_under(sheet, beside)
            # the line's film does not shield the pastel under it (it left
            # twice as much pastel under the line as beside it)
            under = _dust_under(erased, ALONG) / _dust_under(sheet, ALONG)
            assert under < dust_left[name] + 0.15
        for x in (70, 100, 130):
            sheet.smudge([(x, row - 28), (x, row + 32)], tools.FINGER, 0.6)
        smeared[name] = _graphite(sheet, below) / there
        kept[name] = _graphite(sheet, _on(ALONG)) / on_line
    # a film with no wax is as fast as the dust it lies on: a kneaded eraser
    # lifts the line with the pastel under it, as it lifts that pastel (the
    # line lies on top and goes a little sooner), where it takes a quarter
    # of a line the paper holds; on fixed pastel the line is as fast as on
    # paper
    assert 0.05 < left["loose"] < left["paper"] - 0.3
    assert abs(left["loose"] - dust_left["loose"]) < 0.25
    assert left["fixed"] > left["paper"] - 0.05 and dust_left["fixed"] > 0.7
    # and a finger drawn across it smears it with the pastel
    assert smeared["loose"] > 1.4 * smeared["paper"] > 1.4 * smeared["fixed"]
    assert kept["loose"] < kept["paper"] - 0.1 < kept["fixed"]


@pytest.mark.parametrize("fixed", [False, True])
def test_a_pencil_line_on_pastel_leaves_no_stain_under_a_vinyl_eraser(fixed):
    # a stain is wax carried into the paper: the film of a pencil has none,
    # so a rubber eraser, in time, takes all of the line off the pastel as
    # it takes it off bare paper, and a knife along it cuts it out
    sheet = _bed(fixed)
    sheet.stroke(ALONG, pencils.B2, 0.6, 4.0)
    there, on_line = _graphite(sheet), _graphite(sheet, _on(ALONG))
    scratched = sheet.copy()
    for _ in range(6):
        sheet.erase(ALONG, tools.ERASER, 0.6, 10.0)
    assert _graphite(sheet) < 0.02 * there
    assert not sheet.held[..., IW].any()
    scratched.erase(ALONG, tools.KNIFE, 0.9, 2.0)
    assert _graphite(scratched, _on(ALONG)) < 0.05 * on_line
    assert _nothing_negative(sheet) and _nothing_negative(scratched)
    assert _bonded_within_bounds(sheet) and _bonded_within_bounds(scratched)


def test_a_finger_reaches_the_pastel_under_a_pencil_line_and_no_wax_comes_of_it():
    # the film of a dry medium is not slick, as a film of wax is: a finger
    # does not slide on it, and stirs up and lifts the pastel under the line
    # as it does the pastel beside it (it took up next to none of it)
    sheet = _bed()
    for path in LINES:
        sheet.stroke(path, pencils.B2, 0.7, 6.0)
    assert _sum(sheet.film[..., 1]) > 0.0
    before = sheet.under[..., IV].astype(np.float64)
    finger = Smudger(tools.FINGER)
    for x in (70, 100, 130):
        sheet.smudge([(x, 30), (x, 126)], finger, 0.6)
    gone = before - sheet.under[..., IV]
    on_lines, between = np.zeros(before.shape, bool), np.zeros(before.shape, bool)
    for path in LINES[:-1]:
        row = int(path[0][1])
        on_lines[row - 1 : row + 1, 62:138] = True
        between[row + 6 : row + 8, 62:138] = True
    taken_under = gone[on_lines].sum() / before[on_lines].sum()
    taken_beside = gone[between].sum() / before[between].sum()
    assert taken_under > 0.6 * taken_beside > 0.1
    assert not sheet.held[..., IW].any() and finger.wax == 0.0
    assert _nothing_negative(sheet) and _bonded_within_bounds(sheet)


def test_the_wax_of_a_sheet_is_on_it_or_on_the_tool_with_a_dry_film_among_it():
    sheet = _bed()
    for path in LINES[::2]:
        sheet.stroke(path, crayons.wax(BLUE), 0.9, 6.0)
    sheet.shade(BOX, pencils.B4, 0.7, 6.0, angle=75)  # (a dry film, across them)
    assert sheet.film.any()
    wax, pigment = _sum(sheet.held[..., IW]), _total(sheet)
    finger = Smudger(tools.FINGER)
    taken = sheet.rub((30, 30, 170, 80), finger, 0.7)
    taken += sheet.rub((30, 70, 170, 120), tools.KNEADED, 0.7)
    taken += sheet.erase(ALONG, tools.ERASER, 0.9)
    assert _total(sheet) + taken == pytest.approx(pigment, rel=1e-5)
    # (a bare tool takes its wax away with it: no more is left than there was)
    assert 0.0 < _sum(sheet.held[..., IW]) + finger.wax <= wax * (1 + 1e-6)
    assert _nothing_negative(sheet) and _bonded_within_bounds(sheet)


def test_the_thicker_the_dust_the_more_of_a_pencil_line_lies_on_it():
    shares = []
    for base in (
        _bed(dust=charcoal.VINE, layers=1, pressure=0.2),
        _bed(layers=1, pressure=0.3),
        _bed(layers=1),
        _bed(),
    ):
        over = base.copy()
        laid, _ = _pencilled(over, pencils.B2)
        shares.append(_sum(over.film) / (pencils.B2.binding * laid))
    # through thin dust the point reaches the paper, which holds its mark
    assert shares[0] < shares[1] < shares[2] <= shares[3] + 1e-4
    assert shares[0] < 0.6 and shares[1] < 0.9 and shares[2] > 0.95


@pytest.mark.parametrize("medium", DRY)
@pytest.mark.parametrize("paper", [papers.CARTRIDGE, papers.ROUGH])
def test_a_dry_medium_meets_no_bed_in_its_own_layers(medium, paper):
    # its own crumbs do not keep it from the paper: layer on layer, firm or
    # light, fixed or not, none of it is a film, and the paper holds the
    # share it binds (and what later passes press in)
    sheet = Sheet(200, 150, paper=paper, seed=1)
    laid = 0.0
    for layer, pressure in enumerate((0.3, 0.9, 0.5, 1.0)):
        laid += sheet.shade(BOX, medium, pressure, WIDTH, angle=30 - 50 * layer)
    assert not sheet.bonded.any()
    assert _sum(sheet.held[..., IV]) >= 0.999 * medium.binding * laid
    sheet.fix()
    before = _sum(sheet.held[..., IV])
    laid = sheet.shade(BOX, medium, 0.6, WIDTH, angle=80)
    assert not sheet.film.any() and not sheet.paste.any()
    assert _sum(sheet.held[..., IV]) - before >= 0.999 * medium.binding * laid


def test_a_firm_stick_marks_the_dust_of_a_crumbly_one_and_not_the_other_way():
    # compressed charcoal over a tooth full of soft pastel: it lies on it
    # (it took a twelfth of what bare paper takes, half of that under the
    # dust)
    on_paper, dark_on_paper = _pencilled(_sheet(), charcoal.COMPRESSED)
    over = _bed()
    laid, dark = _pencilled(over, charcoal.COMPRESSED)
    assert laid > 0.55 * on_paper and dark > 0.7 * dark_on_paper
    assert _sum(over.film) == pytest.approx(charcoal.COMPRESSED.binding * laid, 0.02)
    # vine charcoal is dust on dust: no bed, and little room in a full tooth
    on_paper, _ = _pencilled(_sheet(), charcoal.VINE)
    over = _bed()
    laid, _ = _pencilled(over, charcoal.VINE)
    assert not over.film.any() and laid < 0.25 * on_paper
    # and a soft pastel over a pencil's layers lies as it did: dust on top
    over = _drawn(pencils.B4, 0.7)
    over.shade(BOX, pastels.soft(PALE), 0.6, 14.0, angle=-30)
    assert not over.bonded.any()


def test_pastel_rubbed_into_the_paper_takes_a_pencil_while_dust_is_left_on_it():
    on_paper, dark_on_paper = _pencilled(_sheet(), pencils.B2)
    once, twice = _bed(), _bed()
    once.rub(BOX, tools.FINGER, 0.6)
    for angle in (0, 90):
        twice.rub(BOX, tools.FINGER, 0.6, angle=angle)
    # blended once, the pastel still has dust on it for the point to ride on
    # (it took a hundredth of what bare paper takes)
    laid, dark = _pencilled(once, pencils.B2)
    assert laid > 0.55 * on_paper and dark > 0.5 * dark_on_paper
    # rubbed in further, it fills the tooth: what the paper holds is in the
    # way of every medium alike
    laid, dark = _pencilled(twice, pencils.B2)
    assert laid < 0.5 * on_paper and dark < 0.4 * dark_on_paper


@pytest.mark.parametrize(
    "medium, pressure, fixed",
    [
        (pencils.B2, 0.6, False),
        (pencils.B6, 1.0, False),
        (pencils.B6, 1.0, True),
        (charcoal.PENCIL, 0.7, False),
    ],
)
def test_layers_of_a_pencil_over_pastel_build_and_level_off(medium, pressure, fixed):
    bare, sheet = _sheet(), _bed(fixed)
    ground = _luminance(sheet)
    laid, tones, on_paper = [], [], 0.0
    for layer in range(12):
        angle = 30 + 50 * layer
        on_paper += bare.shade(BOX, medium, pressure, 6.0, angle=angle)
        laid.append(sheet.shade(BOX, medium, pressure, 6.0, angle=angle))
        tones.append(_luminance(sheet) / ground)
    # each layer takes less than the one before it and the tooth fills, with
    # about as much as it holds on bare paper: what a pass leaves on the
    # pastel is in the tip's way as what the paper holds is (left loose
    # there, it piled up without end)
    # (a layer takes from 0.67 to 0.91 of the one before it, as its strokes
    # fall over those of the last)
    assert all(b < 0.95 * a for a, b in itertools.pairwise(laid[:7]))
    assert laid[-1] < 0.2 * laid[0]
    assert 0.8 * on_paper < sum(laid) < 1.1 * on_paper
    assert float(sheet.pigment().max()) < 5.0
    # and the tone does not fade under the layers that follow: the film
    # keeps the tip from pressing it, and the dust under it, into the paper
    assert tones[0] < 0.75 and min(tones) > 0.02
    if pressure < 1.0:
        assert tones[-1] < tones[0] - 0.05
    else:
        # (a full first layer is as dark as it gets: the layers after it
        # bury its film in the pastel's layer, one mixture with it, and
        # are thinner themselves; it was a third lighter again)
        assert tones[-1] < tones[0] + 0.12
    assert _nothing_negative(sheet) and _bonded_within_bounds(sheet)


@pytest.mark.parametrize("over", [pencils.B2, charcoal.PENCIL, charcoal.COMPRESSED])
@pytest.mark.parametrize("fixed", [False, True])
def test_what_a_dry_medium_lays_over_a_bed_is_on_the_sheet_or_on_the_tool(over, fixed):
    sheet = _bed(fixed)
    laid = _total(sheet)
    laid += sheet.shade(BOX, over, 0.7, 6.0, angle=-30)
    laid += sheet.stroke(ALONG, over, 0.9, 4.0)
    laid += sheet.shade(BOX, pastels.soft(BLUE), 0.5, 14.0, angle=70)
    laid += sheet.shade(BOX, over, 0.6, 6.0, angle=10)
    assert _total(sheet) == pytest.approx(laid, rel=1e-5)
    taken = sheet.rub((30, 30, 170, 70), tools.FINGER, 0.7)
    taken += sheet.rub((30, 60, 170, 100), tools.KNEADED, 0.7)
    taken += sheet.erase([(30, 110), (170, 110)], tools.ERASER, 0.9)
    stump = Smudger(tools.STUMP)
    taken += sheet.rub((30, 30, 170, 120), stump, 0.6, angle=90)
    assert stump.load == pytest.approx(stump.clean())
    assert _total(sheet) + taken == pytest.approx(laid, rel=1e-5)
    assert _nothing_negative(sheet) and _bonded_within_bounds(sheet)
