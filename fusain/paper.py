"""Paper: the surface a drawing is made on, and what its tooth is like."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from murex import ColorLike
from murex import noise as nz
from murex import spectral as sp
from murex.filters import box_blur

TEXTURES = ("wove", "laid", "rough", "pebbled", "sanded")

#: Laid paper: the chain lines are this many laid lines apart.
CHAIN_SPACING = 18.0
#: Pebbled paper: its pits are this many ``grain`` apart, centre to centre
#: (a pit is about a grain across; the rest is the rim between two of them).
PIT_SPACING = 1.3
#: How far a pit may sit from its place in the grid, as a share of the
#: spacing: nearly all the way, so that no rows or columns of pits show.
PIT_SCATTER = 0.9
#: How far the walls of a pit reach in from its rim, as a share of the
#: spacing: its floor is what is further in than that.
PIT_WALL = 0.8
#: The shallowest pit, as a share of the deepest.
PIT_SHALLOWEST = 0.3
#: How far the outline of a pit strays from the straight sides its
#: neighbours would give it, as a share of the spacing.
PIT_STRAY = 0.3
#: Pebbled paper: how much of the height is in the lumps of the rims between
#: the pits, and how much in the fibres over everything, beside a depth of 1
#: for the deepest pit.
PEBBLE_LUMPS = 1.0
PEBBLE_FIBRES = 0.2


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
        ``"rough"`` (the bumps of cold-pressed and rough watercolour paper),
        ``"pebbled"`` (the small irregular pits of a charcoal or pastel
        paper, as if pressed with a felt) or ``"sanded"`` (the sharp grit of
        pastel card).
    grain:
        Size of the tooth in pixels: the distance between neighbouring bumps
        of a wove paper, the spacing of the laid lines, the size of a bump of
        a rough paper, of a pit of a pebbled one or of a grain of grit. This
        is what sets the scale of a drawing: on drawing paper a pencil line
        is two or three grains wide.
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
    hold:
        How much the tooth holds for its depth: 1 for a paper; more for a
        sanded card, whose grit grips layer after layer of pastel.
    """

    texture: str = "wove"
    grain: float = 3.0
    depth: float = 1.0
    bite: float = 0.5
    give: float = 0.5
    color: ColorLike = "#f3f0e6"
    name: str = "paper"
    hold: float = 1.0

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
        if not (math.isfinite(self.hold) and self.hold > 0):
            raise ValueError(f"hold must be a number above 0, got {self.hold!r}")
        sp.parse_color(self.color)  # refused here rather than at the first sheet

    def with_(self, **changes) -> Paper:
        """A copy with some fields changed."""
        return replace(self, **changes)

    def tooth(self, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
        """Height of the surface, (h, w) float32 from 0 (valley) to 1 (peak)."""
        make = {
            "wove": _wove,
            "laid": _laid,
            "rough": _rough,
            "pebbled": _pebbled,
            "sanded": _sanded,
        }
        return _unit(make[self.texture](h, w, self.grain, rng)).astype(np.float32)

    def reflectance(self) -> np.ndarray:
        """Reflectance spectrum of the bare paper, (N_BANDS,)."""
        return sp.srgb_to_reflectance(self.color)


def _unit(a: np.ndarray) -> np.ndarray:
    """``a`` stretched to 0..1 (its rarest half percent at each end clipped)."""
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def _wove(h, w, grain, rng) -> np.ndarray:
    """Felted fibres: small bumps about ``grain`` apart with short fibres
    running through them, and nothing broader: at the scale of a pencil line
    the sheet is even."""
    bumps = nz.value_noise_2d(h, w, max(0.5 * grain, 1.0), rng)
    fibres = sum(
        nz.aniso_noise(h, w, 3.0 * grain, max(0.5 * grain, 1.0), angle, rng, octaves=1)
        for angle in (20, 80, 140)
    )
    return _no_broad(0.6 * bumps + 0.4 * fibres / 3, grain)


def _no_broad(a: np.ndarray, grain: float) -> np.ndarray:
    """``a`` without what varies over more than a few ``grain``."""
    return a - box_blur(a, max(round(1.5 * grain), 1))


def _laid(h, w, grain, rng) -> np.ndarray:
    """Laid lines ``grain`` apart, running across, slightly wavy; the furrows
    of the chain lines down the sheet every :data:`CHAIN_SPACING` of them;
    fibres over both."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    wander = nz.fractal_noise_2d(h, w, grain * 12, rng, octaves=2) - 0.5
    lines = 0.5 + 0.5 * np.cos(2 * np.pi * (y + wander * grain * 0.8) / grain)
    chain = grain * CHAIN_SPACING
    across = (x + wander * grain) % chain
    chains = np.exp(-0.5 * (np.minimum(across, chain - across) / (0.6 * grain)) ** 2)
    fibres = nz.paper(h, w, grain * 0.8, rng)
    # (the chain wires leave shallow furrows, which a stick skips: on a
    # drawing they show as pale lines down the sheet)
    return 0.3 * lines - 0.1 * chains + 0.5 * fibres


def _rough(h, w, grain, rng) -> np.ndarray:
    bumps = nz.fractal_noise_2d(h, w, grain * 2.0, rng, octaves=3)
    fibres = nz.paper(h, w, grain * 0.5, rng)
    return 0.75 * _unit(bumps) + 0.25 * fibres


def _pebbled(h, w, grain, rng) -> np.ndarray:
    """Pits about ``grain`` across, :data:`PIT_SPACING` of it apart and
    scattered, each with its own depth and its own outline; lumpy rims
    between them; fibres over both. Nothing varies over more than a few pits
    and little over less than one: a tone laid lightly is a net of dark rims
    around pale pits, neither the fine flecks of a wove paper nor the blotches
    of a rough one."""
    cell = PIT_SPACING * grain
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    # (the pits are looked up a little beside where they are, differently
    # from place to place: their sides are not straight nor their corners
    # sharp, as those of the cells around scattered points would be)
    y += PIT_STRAY * cell * (2.0 * nz.fractal_noise_2d(h, w, 1.5 * cell, rng, 2) - 1.0)
    x += PIT_STRAY * cell * (2.0 * nz.fractal_noise_2d(h, w, 1.5 * cell, rng, 2) - 1.0)
    inside, deep = _pits(y, x, cell, rng)
    wall = np.clip(inside / (PIT_WALL * cell), 0.0, 1.0)
    pits = deep * (1.0 - (1.0 - wall) ** 2)
    lumps = nz.value_noise_2d(h, w, 0.75 * cell, rng)
    lumps -= box_blur(lumps, max(round(1.5 * cell), 1))
    fibres = nz.paper(h, w, max(0.5 * grain, 1.0), rng)
    return PEBBLE_LUMPS * lumps + PEBBLE_FIBRES * fibres - pits


def _pits(y, x, cell, rng) -> tuple[np.ndarray, np.ndarray]:
    """Pits scattered one to a square of side ``cell``, each owning the
    ground nearer to its centre than to any other. For the points (``y``,
    ``x``): how far inside its pit each is (0 on the rim, growing towards the
    middle; in pixels), and how deep that pit is (:data:`PIT_SHALLOWEST` to
    1)."""
    reach = 2  # squares away a point's two nearest centres can be
    pad = reach + 1  # (one more: the points stray off the sheet a little)
    ny = int(np.ceil(y.shape[0] / cell)) + 2 * pad
    nx = int(np.ceil(y.shape[1] / cell)) + 2 * pad
    centre_y = (np.arange(ny)[:, None] + 0.5 - pad) * cell
    centre_x = (np.arange(nx)[None, :] + 0.5 - pad) * cell
    centre_y = centre_y + PIT_SCATTER * cell * (rng.random((ny, nx)) - 0.5)
    centre_x = centre_x + PIT_SCATTER * cell * (rng.random((ny, nx)) - 0.5)
    depth = 1.0 - (1.0 - PIT_SHALLOWEST) * rng.random((ny, nx))
    row = np.clip(np.floor(y / cell).astype(np.intp) + pad, reach, ny - reach - 1)
    col = np.clip(np.floor(x / cell).astype(np.intp) + pad, reach, nx - reach - 1)
    nearest = np.full(y.shape, np.inf)
    second = np.full(y.shape, np.inf)
    deep = np.zeros(y.shape)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            j, i = row + dy, col + dx
            d = np.hypot(y - centre_y[j, i], x - centre_x[j, i])
            nearer = d < nearest
            second = np.where(nearer, nearest, np.minimum(second, d))
            deep = np.where(nearer, depth[j, i], deep)
            nearest = np.where(nearer, d, nearest)
    return second - nearest, deep


def _sanded(h, w, grain, rng) -> np.ndarray:
    """Grit glued to a card: sharp grains about ``grain`` apart, as many high
    as low, and nothing broader, so that a stick bites all over it."""
    grit = nz.value_noise_2d(h, w, max(0.5 * grain, 1.0), rng)
    finer = nz.value_noise_2d(h, w, max(0.25 * grain, 1.0), rng)
    return _no_broad(0.7 * grit + 0.3 * finer, grain)
