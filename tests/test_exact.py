"""What makes the engine fast changes no result: a table holds the values of
the expression it replaces, a loop skips only pixels it would have left as
they were, and a path worked out once is the path it was.
(``tests/test_fingerprint.py`` checks whole drawings.)"""

import math

import numpy as np
import pytest

from fusain import Sheet, charcoal, pastels, pencils
from fusain import _engine as eng
from fusain.sheet import _Dense, _sample


def test_the_streaks_of_a_pass_are_those_worked_out_one_by_one():
    salt = 123456789
    streaks = eng.streaks_of(salt)
    assert streaks.shape == (eng.STREAK_BANDS + 1,)
    for band in (0, 1, 63, 64, 127, 128):
        assert streaks[band] == eng.unit_noise(band, 0, salt + 1)

    def one_by_one(dx, dy, tx, ty):
        across = (dy * tx - dx * ty) / eng.STREAK_WIDTH + 64.0
        band = math.floor(across)
        blend = across - band
        streak = eng.unit_noise(band, 0, salt + 1) * (1.0 - blend)
        streak += eng.unit_noise(band + 1, 0, salt + 1) * blend
        return 1.0 + eng.STREAK * (2.0 * streak - 1.0)

    rng = np.random.default_rng(5)
    # (within the table, at its ends, and across a tip too wide for it)
    for reach in (3.0, 40.0, 83.0, 84.0, 400.0):
        for _ in range(200):
            dx, dy = rng.uniform(-reach, reach, 2)
            turn = rng.uniform(0.0, 2.0 * math.pi)
            tx, ty = math.cos(turn), math.sin(turn)
            got = eng.streak_at(dx, dy, tx, ty, streaks, salt)
            assert got == one_by_one(dx, dy, tx, ty)


@pytest.mark.parametrize("seed", range(4))
def test_the_span_of_a_row_misses_no_pixel_under_the_tip(seed):
    rng = np.random.default_rng(seed)
    covered, spanned, square = 0, 0, 0
    for _ in range(150):
        wide = rng.choice([0.6, 1.5, 3.5, 12.0, 45.5]) * rng.uniform(0.9, 1.1)
        long = rng.choice([0.6, 1.5, 3.5, 12.0]) * rng.uniform(0.9, 1.1)
        turn = rng.choice([0.0, 0.5 * math.pi, rng.uniform(0.0, 2.0 * math.pi)])
        tx, ty = math.cos(turn), math.sin(turn)
        x, y = rng.uniform(-3.0, 3.0, 2)
        bound = max(wide, long)
        x0, x1 = math.floor(x - bound), math.ceil(x + bound)
        lean, half, top = eng.tip_extent(tx, ty, wide, long)
        for iy in range(math.floor(y - bound), math.ceil(y + bound)):
            dy = iy + 0.5 - y
            xa, xb = eng.row_span(x, dy, lean, half, top, x0, x1)
            assert x0 <= xa and xb <= x1
            spanned, square = spanned + max(xb - xa, 0), square + x1 - x0
            for ix in range(x0, x1):
                edge = eng.tip_bearing(ix + 0.5 - x, dy, tx, ty, wide, long, 0.35)[1]
                if edge > 0.0:
                    assert xa <= ix < xb, (wide, long, turn, ix, iy)
                    covered += 1
    # (and the spans are most of what the tips cover, not the squares
    # around them)
    assert 10000 < covered <= spanned < 0.6 * square


def test_the_span_of_a_row_of_a_round_tool():
    """A round tool of radius r: every pixel within r of its centre."""
    rng = np.random.default_rng(9)
    for _ in range(200):
        r = rng.uniform(0.6, 30.0)
        x, y = rng.uniform(-2.0, 2.0, 2)
        x0, x1 = math.floor(x - r), math.ceil(x + r)
        for iy in range(math.floor(y - r), math.ceil(y + r)):
            dy = iy + 0.5 - y
            xa, xb = eng.row_span(x, dy, 0.0, 1.0, r * r, x0, x1)
            inside = [
                ix for ix in range(x0, x1) if math.hypot(ix + 0.5 - x, dy) < r + 1e-9
            ]
            assert all(xa <= ix < xb for ix in inside)
            # and it is a span, not the whole square: at most a pixel to
            # spare either side, and a rounding
            assert xb - xa <= len(inside) + 4


def test_dust_settles_once_wherever_the_tip_went():
    """A pass leaves nothing unsettled, however its path turns back on
    itself, leaves the sheet and comes back."""
    sheet = Sheet(90, 70, seed=3)
    path = [(-20, 30), (40, 35), (80, 10), (110, -15), (60, 60), (20, 20), (45, 35)]
    for medium in (charcoal.VINE, pastels.soft("#c0392b"), pastels.oil("#2456a6")):
        laid = sheet.stroke(path, medium, 0.7, width=9)
        assert laid > 0
        assert not sheet._fresh.any()
    sheet.shade((5, 5, 85, 65), pencils.B4, 0.6, tilt=0.8)
    assert not sheet._fresh.any()


def test_a_path_sampled_once_gives_what_it_gave():
    from murex import spline

    points = [(3.0, 4.0), (40.0, 9.0), (55.0, 40.0), (20.0, 60.0)]
    for path in (points, points[:2], points[:1], spline.as_path(points)):
        plain = spline.as_path(path)
        once = _Dense(spline.as_path(path))
        assert once.n_knots == plain.n_knots and once.length == plain.length
        for got, want in zip(once.sample(0.75), plain.sample(0.75), strict=True):
            assert np.array_equal(got, want)
        assert once._dense()[0] is once._dense()[0]  # (worked out once)
        _, sampled, u, _, spacing, pace = _sample(path, 6.0)
        want = plain.sample(0.75)
        assert np.array_equal(sampled, want[0]) and np.array_equal(u, want[2])
        assert pace == 0.75
        if len(sampled) > 1:
            assert spacing == plain.length / (len(sampled) - 1)
