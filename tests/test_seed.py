"""A mark with a seed of its own (issue fusain-ny8): its grain is the seed's,
it draws nothing from the sheet's generator, and what is drawn elsewhere
before it does not move it."""

import numpy as np
import pytest

from fusain import Sheet, Stick, charcoal, papers, pencils, pens, tools

#: What a sheet holds, which a mark changes.
STATE = (
    "tooth",
    "crush",
    "polish",
    "dent",
    "cover",
    "held",
    "under",
    "loose",
    "bonded",
    "ink",
)
#: Rows the marks below lie in (the stroke drawn elsewhere first lies below).
TOP = slice(0, 140)

MARKS = {
    "stroke": lambda s, seed: s.stroke(
        [(10, 90), (170, 100)], pencils.HB, 0.5, seed=seed
    ),
    "stroke of a stick": lambda s, seed: s.stroke(
        [(10, 90), (170, 100)], Stick(pencils.HB, width=3), 0.5, seed=seed
    ),
    "pen": lambda s, seed: s.stroke([(10, 90), (170, 100)], pens.BLUE, 0.5, seed=seed),
    "shade": lambda s, seed: s.shade((10, 70, 170, 120), pencils.HB, 0.5, seed=seed),
    "smudge": lambda s, seed: s.smudge(
        [(10, 30), (170, 40)], tools.STUMP, 0.6, seed=seed
    ),
    "erase": lambda s, seed: s.erase(
        [(10, 30), (170, 40)], tools.ERASER, 0.6, seed=seed
    ),
    "rub": lambda s, seed: s.rub((10, 15, 170, 55), tools.STUMP, 0.6, seed=seed),
}


def _sheet(before: int = 0) -> Sheet:
    """A sheet with a tone on it, and ``before`` strokes drawn elsewhere
    (without a seed, so that its generator has moved on)."""
    sheet = Sheet(200, 200, paper=papers.CARTRIDGE, seed=1)
    sheet.shade((10, 10, 170, 60), charcoal.VINE, 0.6, 9.0, seed=3)
    for k in range(before):
        sheet.stroke([(10, 170 + k), (180, 175 + k)], pencils.HB, 0.5)
    return sheet


def _rows(sheet: Sheet, name: str, rows: slice) -> np.ndarray:
    held = getattr(sheet, name)  # (the ink is none until a pen has drawn)
    return np.zeros(0) if held is None else held[rows].copy()


def _state(sheet: Sheet):
    return {name: _rows(sheet, name, TOP) for name in STATE}


def _same(a, b) -> bool:
    return all(np.array_equal(a[name], b[name]) for name in STATE)


@pytest.mark.parametrize("name", MARKS)
def test_a_seeded_mark_does_not_depend_on_what_was_drawn_before(name):
    mark = MARKS[name]
    a, b = _sheet(0), _sheet(3)
    assert a._rng.bit_generator.state != b._rng.bit_generator.state
    mark(a, 11)
    mark(b, 11)
    assert _same(_state(a), _state(b))


@pytest.mark.parametrize("name", MARKS)
def test_a_seeded_mark_draws_nothing_from_the_sheets_generator(name):
    sheet = _sheet(1)
    before = sheet._rng.bit_generator.state
    MARKS[name](sheet, 11)
    assert sheet._rng.bit_generator.state == before


@pytest.mark.parametrize("name", MARKS)
def test_a_seed_gives_the_grain_and_another_gives_another(name):
    states = []
    for seed in (11, 11, 12):
        sheet = _sheet()
        MARKS[name](sheet, seed)
        states.append(_state(sheet))
    assert _same(states[0], states[1])
    assert not _same(states[0], states[2])


@pytest.mark.parametrize("name", MARKS)
def test_what_is_drawn_after_a_seeded_mark_does_not_depend_on_it(name):
    """The strokes after it have the grain they would have had."""
    plain = _sheet()
    plain.stroke([(10, 150), (170, 160)], pencils.HB, 0.5)
    seeded = _sheet()
    MARKS[name](seeded, 11)
    seeded.stroke([(10, 150), (170, 160)], pencils.HB, 0.5)
    for name_ in STATE[:-1]:  # (no ink, which is none before a pen draws)
        rest = slice(140, None)
        a, b = _rows(plain, name_, rest), _rows(seeded, name_, rest)
        assert np.array_equal(a, b)


def test_a_mark_without_a_seed_draws_from_the_sheets_generator():
    sheet = _sheet()
    before = sheet._rng.bit_generator.state
    sheet.stroke([(10, 90), (170, 100)], pencils.HB, 0.5)
    assert sheet._rng.bit_generator.state != before


@pytest.mark.parametrize("name", MARKS)
@pytest.mark.parametrize("seed", [-1, 2**31, 1.5, "7", True, float("nan")])
def test_a_bad_seed_raises_before_anything_is_drawn(name, seed):
    sheet = _sheet()
    state, rng = _state(sheet), sheet._rng.bit_generator.state
    with pytest.raises(ValueError, match="seed must be"):
        MARKS[name](sheet, seed)
    assert _same(state, _state(sheet))
    assert sheet._rng.bit_generator.state == rng


@pytest.mark.parametrize("seed", [0, 2**31 - 1, np.int64(5)])
def test_the_ends_of_the_range_and_numpy_integers_are_seeds(seed):
    _sheet().stroke([(10, 90), (170, 100)], pencils.HB, 0.5, seed=seed)
