"""Paper: the surface a drawing is made on, and what its tooth is like."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from murex import ColorLike
from murex import noise as nz
from murex import spectral as sp

TEXTURES = ("wove", "laid", "rough", "sanded")

#: Laid paper: the chain lines are this many laid lines apart.
CHAIN_SPACING = 18.0


@dataclass(frozen=True)
class Paper:
    """A paper. Presets live in :mod:`fusain.papers`; change one with
    :meth:`with_` or make your own::

        from fusain import Paper, Sheet, papers
        sheet = Sheet(1200, 800, paper=papers.INGRES)
        toned = papers.CARTRIDGE.with_(color="#b9a98c")
        mine = Paper(texture="rough", grain=6, depth=1.5, bite=0.7)

    Parameters
    ----------
    texture:
        ``"wove"`` (felted fibres, the usual drawing paper), ``"laid"`` (fine
        parallel lines crossed by chain lines, as in Ingres paper),
        ``"rough"`` (the bumps of cold-pressed and rough watercolour paper)
        or ``"sanded"`` (the sharp grit of pastel card).
    grain:
        Size of the tooth in pixels: the width of a fibre bundle, the spacing
        of the laid lines, the size of a bump or of a grain of grit.
    depth:
        How deep the tooth is (0 = flat): how far a tip has to reach to touch
        the valleys, how many layers the paper takes, and how strongly the
        texture shows under a light.
    bite:
        0..1: how abrasive the surface is, how much it takes off a stick that
        touches it. Smooth hot-pressed paper bites little, sanded card a lot.
    give:
        0..1: how much the surface yields under a tip. A thin sheet on a hard
        desk, or a board, gives little; a thick soft paper, or a sheet on a
        pad of others, lets the tip sink in and reach further into the tooth.
    color:
        Colour of the paper.
    name:
        Optional label.
    """

    texture: str = "wove"
    grain: float = 3.0
    depth: float = 1.0
    bite: float = 0.5
    give: float = 0.5
    color: ColorLike = "#f3f0e6"
    name: str = "paper"

    def __post_init__(self):
        if self.texture not in TEXTURES:
            raise ValueError(f"unknown texture {self.texture!r}; use one of {TEXTURES}")
        if not (math.isfinite(self.grain) and self.grain > 0):
            raise ValueError(f"grain must be a size in pixels above 0: {self.grain!r}")
        if not (math.isfinite(self.depth) and self.depth >= 0):
            raise ValueError(f"depth must be 0 (flat) or more, got {self.depth!r}")
        for field in ("bite", "give"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(
                    f"{field} must be between 0 and 1, got {getattr(self, field)!r}"
                )
        sp.parse_color(self.color)  # refused here rather than at the first sheet

    def with_(self, **changes) -> Paper:
        """A copy with some fields changed."""
        return replace(self, **changes)

    def tooth(self, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
        """Height of the surface, (h, w) float32 from 0 (valley) to 1 (peak)."""
        make = {"wove": _wove, "laid": _laid, "rough": _rough, "sanded": _sanded}
        return _unit(make[self.texture](h, w, self.grain, rng)).astype(np.float32)

    def reflectance(self) -> np.ndarray:
        """Reflectance spectrum of the bare paper, (N_BANDS,)."""
        return sp.srgb_to_reflectance(self.color)


def _unit(a: np.ndarray) -> np.ndarray:
    """``a`` stretched to 0..1 (its rarest half percent at each end clipped)."""
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def _wove(h, w, grain, rng) -> np.ndarray:
    return nz.paper(h, w, grain, rng)


def _laid(h, w, grain, rng) -> np.ndarray:
    """Laid lines ``grain`` apart, running across, slightly wavy; chain lines
    down the sheet every :data:`CHAIN_SPACING` of them; fibres over both."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    wander = nz.fractal_noise_2d(h, w, grain * 12, rng, octaves=2) - 0.5
    lines = 0.5 + 0.5 * np.cos(2 * np.pi * (y + wander * grain * 0.8) / grain)
    chain = grain * CHAIN_SPACING
    across = (x + wander * grain) % chain
    chains = np.exp(-0.5 * (np.minimum(across, chain - across) / (0.6 * grain)) ** 2)
    fibres = nz.paper(h, w, grain * 0.8, rng)
    return 0.5 * lines + 0.2 * chains + 0.3 * fibres


def _rough(h, w, grain, rng) -> np.ndarray:
    bumps = nz.fractal_noise_2d(h, w, grain * 2.0, rng, octaves=3)
    fibres = nz.paper(h, w, grain * 0.5, rng)
    return 0.75 * _unit(bumps) + 0.25 * fibres


def _sanded(h, w, grain, rng) -> np.ndarray:
    grit = nz.value_noise_2d(h, w, max(grain * 0.5, 1.0), rng)
    card = nz.fractal_noise_2d(h, w, grain * 4.0, rng, octaves=2)
    # grit stands up sharply from the card: mostly low, with pointed peaks
    return 0.85 * _unit(grit) ** 1.6 + 0.15 * card
