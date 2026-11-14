"""Papers: what is accepted, and what each texture's tooth is like."""

import numpy as np
import pytest

from fusain import Paper, Sheet, charcoal, papers
from fusain.paper import PIT_SPACING, TEXTURES


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


def test_wove_paper_is_even_at_the_scale_of_a_pencil_line():
    # (broad patches of high and low tooth made shading come out in blotches
    # wider than the pencil that drew it)
    tooth = _tooth(Paper(texture="wove", grain=3.0)).astype(np.float64)
    blocks = tooth[:120, :160].reshape(10, 12, 16, 10).mean(axis=(1, 3))
    assert blocks.std() < 0.02 and tooth.std() > 0.15
    # its bumps are about a grain across: the runs of high ground along a row
    high = np.pad(tooth > 0.5, ((0, 0), (1, 1)))
    starts = (~high[:, :-1] & high[:, 1:]).sum()
    assert 2.0 < high.sum() / starts < 4.5


def test_rough_paper_has_broader_bumps_than_wove_of_the_same_grain():
    wove = _tooth(Paper(texture="wove", grain=4.0))
    rough = _tooth(Paper(texture="rough", grain=4.0))
    assert np.abs(np.diff(wove, axis=1)).mean() > np.abs(np.diff(rough, axis=1)).mean()


def _share_by_scale(tooth, grain):
    """The share of the tooth's variation at wavelengths under 0.8 grain,
    from there to 4 grains, and above."""
    a = tooth.astype(np.float64) - tooth.mean()
    power = np.abs(np.fft.fft2(a)) ** 2
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    waves = grain * np.hypot(fy, fx)  # per grain
    fine = power[waves > 1 / 0.8].sum()
    broad = power[waves < 1 / 4.0].sum()
    return fine / power.sum(), 1 - (fine + broad) / power.sum(), broad / power.sum()


@pytest.mark.parametrize("grain", [3.0, 4.0, 5.0])
def test_pebbled_paper_varies_at_the_scale_of_its_pits_and_little_beside(grain):
    # (a wove paper has no more broad variation but is all fine flecks; a
    # rough one of this grain is blotchy)
    def shares(texture):
        tooth = _tooth(Paper(texture=texture, grain=grain), h=256, w=256)
        return _share_by_scale(tooth, grain)

    fine, medium, broad = shares("pebbled")
    assert medium > 0.75 and fine < 0.1 and broad < 0.15
    assert shares("rough")[2] > 3 * broad
    # no patches of high or low ground several pits wide
    tooth = _tooth(Paper(texture="pebbled", grain=3.0)).astype(np.float64)
    blocks = tooth[:120, :160].reshape(4, 30, 4, 40).mean(axis=(1, 3))
    assert blocks.std() < 0.02


def test_pebbled_paper_is_pits_in_a_net_of_rims():
    grain = 5.0
    tooth = _tooth(Paper(texture="pebbled", grain=grain), h=200, w=200)
    low = tooth < 0.35
    # the low ground is separate pits, most of the one to a square 1.3 grains
    # wide that there are (the shallowest stay above it, and some run together)
    pits = _count_patches(low)
    assert 0.45 < pits * (PIT_SPACING * grain) ** 2 / low.size < 1.1
    # and the high ground between them mostly hangs together
    high = tooth > 0.5
    assert _largest_patch(high) > 0.6 * high.sum()
    # there is no grain to it: across and down are alike
    down = np.abs(np.diff(tooth, axis=0)).mean()
    across = np.abs(np.diff(tooth, axis=1)).mean()
    assert 0.9 < down / across < 1.1


def test_the_pits_of_a_pebbled_paper_differ_in_depth():
    tooth = _tooth(Paper(texture="pebbled", grain=6.0), h=240, w=240)
    floors = [tooth[where].min() for where in _patches(tooth < 0.45) if where[0].size]
    assert len(floors) > 300
    assert np.percentile(floors, 90) - np.percentile(floors, 10) > 0.25


def test_a_charcoal_tone_on_pebbled_paper_is_neither_flecked_nor_blotchy():
    def tone(texture):
        paper = papers.CHARCOAL.with_(texture=texture)
        sheet = Sheet(260, 200, paper=paper, seed=2)
        for angle in (35, -30):
            sheet.shade((10, 10, 250, 190), charcoal.VINE, 0.3, 14.0, angle)
        grey = sheet.render(lighting=False)[30:170, 30:230].mean(axis=2)
        return grey - grey.mean()

    def patches(grey, size):
        """How much the tone differs between squares ``size`` wide."""
        h, w = grey.shape[0] // size, grey.shape[1] // size
        cut = grey[: h * size, : w * size]
        return cut.reshape(h, size, w, size).mean(axis=(1, 3)).std()

    wove, rough, pebbled = tone("wove"), tone("rough"), tone("pebbled")
    # the paper shows as much in each
    assert 0.8 < pebbled.std() / wove.std() < 1.2
    # squares five grains wide differ far less than on a rough paper...
    assert patches(pebbled, 20) < 0.6 * patches(rough, 20)
    # ...and neighbouring pixels are no less alike than on a wove one
    step = [np.abs(np.diff(t, axis=1)).mean() for t in (wove, pebbled)]
    assert step[1] < step[0]


def _patches(mask):
    """The patches of ``mask`` that hang together by their sides, as index
    arrays."""
    label = np.where(mask, np.arange(mask.size).reshape(mask.shape), -1)
    while True:  # every cell takes the highest label among its neighbours
        grown = label.copy()
        grown[1:] = np.maximum(grown[1:], label[:-1])
        grown[:-1] = np.maximum(grown[:-1], label[1:])
        grown[:, 1:] = np.maximum(grown[:, 1:], label[:, :-1])
        grown[:, :-1] = np.maximum(grown[:, :-1], label[:, 1:])
        grown[~mask] = -1
        if np.array_equal(grown, label):
            break
        label = grown
    return [np.nonzero(label == name) for name in np.unique(label[mask])]


def _count_patches(mask):
    return len(_patches(mask))


def _largest_patch(mask):
    return max(where[0].size for where in _patches(mask))


def test_sanded_card_is_an_even_field_of_sharp_grit():
    sanded = _tooth(Paper(texture="sanded", grain=3.0)).astype(np.float64)
    wove = _tooth(Paper(texture="wove", grain=3.0)).astype(np.float64)
    # as many grains high as low, and no broad patches of either
    assert abs(np.median(sanded) - 0.5) < 0.05
    blocks = sanded[:120, :160].reshape(10, 12, 16, 10).mean(axis=(1, 3))
    assert blocks.std() < 0.02
    # sharper than the bumps of a wove paper of the same grain
    assert (
        np.abs(np.diff(sanded, axis=1)).mean()
        > 1.1 * np.abs(np.diff(wove, axis=1)).mean()
    )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"texture": "canvas"}, "unknown texture"),
        ({"grain": 0}, "grain must be"),
        ({"grain": float("nan")}, "grain must be"),
        ({"depth": -0.1}, "depth must be"),
        ({"depth": float("inf")}, "depth must be"),
        ({"bite": 1.5}, "bite must be"),
        ({"give": -0.5}, "give must be"),
        ({"hold": 0.0}, "hold must be"),
        ({"hold": float("nan")}, "hold must be"),
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
    assert len(set(names)) == len(names) == 12
    assert len(set(papers.ALL)) == len(papers.ALL)
    for paper in papers.ALL:
        assert paper.reflectance().shape == (24,)
