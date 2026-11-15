"""The macaw example draws each part from a stream of its own, and chooses
its places for strokes and feathers one pixel at a time: a change to one part
of the model stays where it is in the picture (issue fusain-ny8)."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from fusain import Sheet

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _example(name: str):
    """A module of examples/ (which is not a package)."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(EXAMPLES))


@pytest.fixture(scope="module")
def macaw():
    return _example("macaw")


@pytest.fixture(scope="module")
def model():
    return _example("macaw_model")


def _hand(macaw, part: str):
    hand = macaw.Hand(Sheet(8, 8, seed=1))
    hand.take(part)
    return hand


def test_each_part_has_a_stream_of_its_own(macaw):
    """What a part draws does not depend on what any other part drew."""
    seqs = macaw.streams()
    assert list(seqs) == list(macaw.PARTS)
    first = {p: np.random.default_rng(seqs[p]).random(4) for p in macaw.PARTS}
    assert len({tuple(v) for v in first.values()}) == len(macaw.PARTS)
    # (the same, whatever else was drawn from the others first)
    again = macaw.streams()
    for part in reversed(macaw.PARTS):
        rng = np.random.default_rng(again[part])
        assert np.array_equal(rng.random(4), first[part])
    # and a part added at the end moves none
    more = np.random.SeedSequence(macaw.SEED).spawn(len(macaw.PARTS) + 1)
    for part, seq in zip(macaw.PARTS, more, strict=False):
        assert np.array_equal(seq.generate_state(4), seqs[part].generate_state(4))


def test_a_keyed_stream_does_not_depend_on_the_others(macaw):
    hand = _hand(macaw, "ticks")
    want = hand.keyed(5, 7).random(3)
    hand.rng.random(100)
    hand.fork(3)
    hand.keyed(1).random(50)
    assert np.array_equal(hand.keyed(5, 7).random(3), want)
    assert not np.array_equal(hand.keyed(5, 8).random(3), want)
    # (and it is not a stream that was spawned)
    assert not any(
        np.array_equal(np.random.default_rng(c).random(3), want)
        for c in hand.seq.spawn(4)
    )


def test_a_pixel_losing_its_weight_loses_its_own_point(macaw):
    """Strokes are sown pixel by pixel: take weight from a patch and only the
    points in the patch go, the rest are where they were with the numbers
    they had."""
    h, w = 150, 220
    weight = np.tile(0.3 + 0.7 * np.sin(np.arange(w) / 30.0) ** 2, (h, 1))
    less = weight.copy()
    less[60:75, 100:115] = 0.0
    a = _hand(macaw, "plumage").sow(weight, 900 / weight.sum(), uniform=2, normal=1)
    b = _hand(macaw, "plumage").sow(less, 900 / weight.sum(), uniform=2, normal=1)
    pa, pb = a[0], b[0]
    assert abs(len(pa) - 900) < 4 * 30  # (about the count asked for)
    kept = {tuple(p) for p in pb.tolist()}
    gone = [p for p in pa.tolist() if tuple(p) not in kept]
    assert 0 < len(gone) < 12
    assert all(60 <= y < 75 and 100 <= x < 115 for x, y in gone)
    # the points that remain carry the same numbers
    where = {tuple(p): i for i, p in enumerate(pa.tolist())}
    for j, p in enumerate(pb.tolist()):
        i = where[tuple(p)]
        assert np.array_equal(a[1][:, i], b[1][:, j])
        assert np.array_equal(a[2][:, i], b[2][:, j])
    # and a pixel that gains weight gains its own point at most
    more = weight.copy()
    more[10:14, 10:14] = 5.0
    c = _hand(macaw, "plumage").sow(more, 900 / weight.sum())[0]
    kept = {tuple(p) for p in c.tolist()}
    assert all(
        tuple(p) in kept
        for p in pa.tolist()
        if not (10 <= p[1] < 14 and 10 <= p[0] < 14)
    )


def test_the_model_scatters_its_feathers_by_pixel(model):
    """Hide a few pixels of the plumage and the tiles go on as they were,
    but for the neighbourhood of the hidden ones."""
    h, w = 200, 260
    where = np.ones((h, w), dtype=bool)
    spacing = np.full((h, w), 6.0)
    hole = where.copy()
    hole[90:100, 120:130] = False
    a = model.scatter(where, spacing, np.random.default_rng(3))
    b = model.scatter(hole, spacing, np.random.default_rng(3))
    kept = {tuple(p) for p in b.tolist()}
    gone = [p for p in a.tolist() if tuple(p) not in kept]
    assert len(gone) < 0.02 * len(a)
    for x, y in gone:
        assert 90 - 4 * 6 <= y < 100 + 4 * 6 and 120 - 4 * 6 <= x < 130 + 4 * 6
