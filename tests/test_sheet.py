"""The sheet: its state, and how it is rendered."""

from typing import Any

import numpy as np
import pytest
from murex import Pigment
from murex import spectral as sp

from fusain import Paper, Sheet, papers
from fusain.sheet import HELD, IS, IV, LOOSE, N

FLAT: dict[str, Any] = {"lighting": False}
RAKING: dict[str, Any] = {"light": (-0.6, -0.8, 0.45)}


def _lay(layer, where, pigment, thickness):
    """Put a layer of ``pigment`` straight into ``layer`` (held or loose)."""
    layer[*where, slice(0, N)] = pigment.K * thickness
    layer[*where, IS] = pigment.S * thickness
    layer[*where, IV] = thickness


def _lum(rgb):
    return sp.srgb_to_linear(rgb) @ sp.LUMINANCE


def test_a_new_sheet_is_bare_paper():
    sheet = Sheet(48, 32, paper=papers.INGRES, seed=1)
    assert (sheet.width, sheet.height) == (48, 32)
    assert sheet.tooth.shape == sheet.press.shape == (32, 48)
    assert sheet.held.shape == (32, 48, HELD) and sheet.loose.shape == (32, 48, LOOSE)
    for state in (sheet.tooth, sheet.press, sheet.held, sheet.loose):
        assert state.dtype == np.float32
    assert not sheet.pigment().any() and not sheet.press.any()
    np.testing.assert_allclose(
        sheet.reflectance(), np.broadcast_to(sheet.ground, (32, 48, N)), rtol=1e-6
    )
    # flat, it is the colour of the paper to within the rounding of a PNG
    flat = sheet.render(**FLAT)
    assert flat.shape == (32, 48, 3) and flat.dtype == np.float32
    assert np.abs(flat - sp.parse_color(papers.INGRES.color)).max() < 0.5 / 255


def test_the_default_paper_is_cartridge_like_and_the_seed_sets_the_tooth():
    assert Sheet(8, 8).paper == Paper()
    a, b, c = Sheet(40, 30, seed=5), Sheet(40, 30, seed=5), Sheet(40, 30, seed=6)
    np.testing.assert_array_equal(a.tooth, b.tooth)
    assert not np.array_equal(a.tooth, c.tooth)


def test_a_lamp_shows_the_tooth_and_a_raking_one_shows_it_more():
    sheet = Sheet(160, 120, paper=papers.COLD_PRESSED, seed=2)
    flat, lit, raking = (_lum(sheet.render(**kw)) for kw in (FLAT, {}, RAKING))
    assert flat.std() < 1e-6
    assert 0.002 < lit.std() < 0.05 and raking.std() > 3 * lit.std()
    # shading moves light around; it does not tint or darken the sheet much
    assert abs(lit.mean() - flat.mean()) < 0.01
    # light=None is the default lamp
    np.testing.assert_array_equal(sheet.render(light=None), sheet.render())
    # the slopes facing the lamp (upper left) are the bright ones
    gy, gx = np.gradient(sheet.tooth.astype(np.float64))
    assert np.corrcoef((gx * 0.6 + gy * 0.8).ravel(), raking.ravel())[0, 1] > 0.8


def test_relief_and_depth_scale_the_shading():
    sheet = Sheet(120, 90, seed=2)
    assert _lum(sheet.render(relief=0, **RAKING)).std() < 1e-6
    shallow, deep = (_lum(sheet.render(relief=r, **RAKING)).std() for r in (0.5, 2))
    assert deep > 2 * shallow
    level = Sheet(120, 90, paper=Paper(depth=0), seed=2)
    assert _lum(level.render(**RAKING)).std() < 1e-6
    np.testing.assert_allclose(
        sheet.surface_height(2.0), 2 * sheet.surface_height(), rtol=1e-6
    )
    assert level.surface_height().max() == 0


def test_held_pigment_looks_like_that_pigment_on_that_paper():
    paper = papers.TONED_TAN
    sheet = Sheet(40, 30, paper=paper, seed=1)
    blue = Pigment("#2a4fa8", opacity=0.4)
    _lay(sheet.held, (slice(5, 15), slice(10, 30)), blue, 0.7)
    flat = sheet.render(**FLAT)
    expected = blue.appearance(0.7, over=paper.color)
    assert np.abs(flat[5:15, 10:30] - expected).max() < 1e-3
    # elsewhere the paper is untouched
    assert np.abs(flat[20:] - sp.parse_color(paper.color)).max() < 0.5 / 255
    np.testing.assert_allclose(sheet.pigment()[5:15, 10:30], 0.7)
    assert sheet.pigment()[20:].max() == 0


def test_loose_pigment_lies_over_held_pigment():
    white, black = Pigment("#f4f4f0"), Pigment("#101010")
    sheet = Sheet(20, 10, seed=1)
    left, right = (slice(None), slice(0, 10)), (slice(None), slice(10, 20))
    _lay(sheet.held, left, black, 1.0)
    _lay(sheet.loose, left, white, 1.0)
    _lay(sheet.held, right, white, 1.0)
    _lay(sheet.loose, right, black, 1.0)
    flat = _lum(sheet.render(**FLAT))
    assert flat[:, :10].min() > 0.6 and flat[:, 10:].max() < 0.1
    # loose pigment alone is seen over the bare paper
    alone = Sheet(20, 10, seed=1)
    _lay(alone.loose, left, black, 0.005)
    grey = _lum(alone.render(**FLAT))
    assert 0.05 < grey[:, :10].mean() < 0.6 and grey[:, 10:].min() > 0.8


def test_a_copy_is_independent():
    sheet = Sheet(30, 20, seed=4)
    copy = sheet.copy()
    _lay(copy.held, (slice(0, 5), slice(0, 5)), Pigment("#202020"), 1.0)
    copy.press[:] = 0.5
    assert not sheet.pigment().any() and not sheet.press.any()
    assert copy.pigment().any() and copy.paper is sheet.paper
    np.testing.assert_array_equal(copy.tooth, sheet.tooth)
    # the two go on drawing the same random numbers
    assert sheet._rng.random() == copy._rng.random()


def test_save_writes_a_png(tmp_path):
    path = tmp_path / "sheet.png"
    Sheet(24, 16, seed=1).save(str(path), **RAKING)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert int.from_bytes(data[16:20], "big") == 24
    assert int.from_bytes(data[20:24], "big") == 16


def test_a_tall_sheet_is_rendered_in_several_passes_alike():
    sheet = Sheet(12, 150, seed=3)
    _lay(sheet.held, (slice(60, 70), slice(2, 8)), Pigment("#303030"), 0.5)
    whole = sheet.render()
    assert whole.shape == (150, 12, 3)
    assert whole[60:70, 2:8].max() < whole[0:50].min()


@pytest.mark.parametrize("size", [(1, 9), (9, 1), (1, 1)])
def test_a_sheet_one_pixel_wide_or_high_is_lit_too(size):
    sheet = Sheet(*size, seed=1)
    lit = sheet.render()
    assert lit.shape == (size[1], size[0], 3) and np.isfinite(lit).all()
    # the one pixel has no slope: it is the colour of the paper
    assert np.abs(Sheet(1, 1).render() - Sheet(1, 1).render(**FLAT)).max() < 1e-6


@pytest.mark.parametrize(
    "args, error, message",
    [
        ((0, 10), ValueError, "width must be"),
        ((10, -3), ValueError, "height must be"),
        ((10.5, 10), ValueError, "width must be"),
        (("wide", 10), TypeError, "width must be"),
        ((True, 10), TypeError, "width must be"),
    ],
)
def test_a_sheet_checks_its_size(args, error, message):
    with pytest.raises(error, match=message):
        Sheet(*args)
    whole: Any = 10.0  # a whole size given as a float is fine
    assert Sheet(whole, 8).width == 10


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"light": (0, 0, -1)}, "in front of the sheet"),
        ({"light": (0, 0, 0)}, "in front of the sheet"),
        ({"light": (1, 2)}, "three finite numbers"),
        ({"light": (1, float("nan"), 1)}, "three finite numbers"),
        ({"light": "above"}, "three numbers"),
        ({"relief": -1}, "relief must be"),
        ({"relief": float("inf")}, "relief must be"),
    ],
)
def test_render_checks_the_lamp_only_when_it_is_used(kw, message):
    sheet = Sheet(8, 8)
    with pytest.raises(ValueError, match=message):
        sheet.render(**kw)
    sheet.render(lighting=False, **kw)
