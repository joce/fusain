"""Papers: what is accepted, and what each texture's tooth is like."""

import numpy as np
import pytest

from fusain import Paper, papers
from fusain.paper import TEXTURES


def _tooth(paper, seed=3, h=120, w=160):
    return paper.tooth(h, w, np.random.default_rng(seed))


@pytest.mark.parametrize("texture", TEXTURES)
def test_tooth_is_a_height_from_0_to_1_that_repeats_with_its_seed(texture):
    paper = Paper(texture=texture)
    tooth = _tooth(paper)
    assert tooth.shape == (120, 160) and tooth.dtype == np.float32
    assert tooth.min() == 0.0 and tooth.max() == 1.0
    assert tooth.std() > 0.1  # it uses the range: there are peaks and valleys
    np.testing.assert_array_equal(tooth, _tooth(paper))
    assert not np.array_equal(tooth, _tooth(paper, seed=4))


def test_laid_paper_has_lines_across_the_sheet():
    tooth = _tooth(Paper(texture="laid", grain=4.0))
    # going down the sheet crosses the laid lines; going across follows them
    down = np.abs(np.diff(tooth, axis=0)).mean()
    across = np.abs(np.diff(tooth, axis=1)).mean()
    assert down > 2 * across
    # and the lines come back every `grain` pixels
    column = tooth[:, 40] - tooth[:, 40].mean()
    spectrum = np.abs(np.fft.rfft(column))
    assert np.argmax(spectrum[1:]) + 1 == round(len(column) / 4.0)


def test_grain_sets_the_size_of_the_tooth():
    for texture in TEXTURES:
        fine = _tooth(Paper(texture=texture, grain=2.0))
        coarse = _tooth(Paper(texture=texture, grain=8.0))
        step = [np.abs(np.diff(t, axis=0)).mean() for t in (fine, coarse)]
        assert step[0] > 1.5 * step[1], texture


def test_rough_paper_has_broader_bumps_than_wove_of_the_same_grain():
    wove = _tooth(Paper(texture="wove", grain=4.0))
    rough = _tooth(Paper(texture="rough", grain=4.0))
    assert np.abs(np.diff(wove, axis=1)).mean() > np.abs(np.diff(rough, axis=1)).mean()


def test_sanded_card_is_mostly_low_with_sharp_peaks():
    sanded = _tooth(Paper(texture="sanded"))
    wove = _tooth(Paper(texture="wove"))
    assert np.median(sanded) < np.median(wove) - 0.05


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"texture": "canvas"}, "unknown texture"),
        ({"grain": 0}, "grain must be"),
        ({"grain": float("nan")}, "grain must be"),
        ({"depth": -0.1}, "depth must be"),
        ({"depth": float("inf")}, "depth must be"),
        ({"bite": 1.5}, "bite must be"),
        ({"color": "paper white"}, "#rrggbb"),
    ],
)
def test_a_paper_checks_its_numbers(changes, message):
    with pytest.raises(ValueError, match=message):
        Paper(**changes)


def test_with_changes_a_copy():
    toned = papers.CARTRIDGE.with_(color="#b9a98c", name="toned")
    assert (toned.color, toned.name) == ("#b9a98c", "toned")
    assert toned.grain == papers.CARTRIDGE.grain
    assert papers.CARTRIDGE.color != "#b9a98c"
    with pytest.raises(ValueError, match="bite must be"):
        papers.CARTRIDGE.with_(bite=2)


def test_presets_are_named_and_distinct():
    names = [paper.name for paper in papers.ALL]
    assert len(set(names)) == len(names) == 10
    assert len(set(papers.ALL)) == len(papers.ALL)
    for paper in papers.ALL:
        assert paper.reflectance().shape == (24,)
