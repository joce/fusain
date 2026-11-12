"""A sheet of paper and what has been drawn on it."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
from murex import spectral as sp
from murex import write_png

from .paper import Paper

N = sp.N_BANDS
#: Pigment is packed per pixel as [K (N bands), S, volume], all extensive
#: (summed over the pigment present); the held layer has a fourth part, the
#: volume of wax in it.
IS, IV, IW = N, N + 1, N + 2
LOOSE, HELD = N + 2, N + 3

#: The lamp a sheet is rendered under by default (x, y, z; y points down the
#: image): from the upper left and fairly high, so that the tooth shows
#: faintly, as on a scan. A smaller z rakes across the surface.
DEFAULT_LIGHT = (-0.6, -0.8, 2.4)
#: Share of the light that comes from all around rather than from the lamp.
AMBIENT = 0.3
#: Height of the deepest tooth of a paper of depth 1, as a share of its grain.
TOOTH_HEIGHT = 0.3
#: Rows rendered at a time (bounds the float64 working memory).
ROWS = 64


class Sheet:
    """A sheet of paper to draw on.

    Parameters
    ----------
    width, height:
        Size in pixels.
    paper:
        A :class:`fusain.Paper`, e.g. ``papers.INGRES``. Default: cartridge
        paper.
    seed:
        Random seed for the paper's tooth and for the marks.

    Per pixel the sheet holds the height of the tooth (``tooth``, 0 valley to
    1 peak), how far it has been crushed or polished (``press``, 0 to 1), the
    pigment fixed to the paper (``held``) and the pigment lying on it as dust
    (``loose``), which is what a stump, an eraser or a later stroke can still
    move. Memory: about 0.22 KB per pixel.

    Raises ValueError or TypeError unless ``width`` and ``height`` are whole
    numbers above 0.
    """

    def __init__(
        self,
        width: int,
        height: int,
        paper: Paper | None = None,
        seed: int | None = None,
    ):
        self.width, self.height = _size("width", width), _size("height", height)
        self.paper = Paper() if paper is None else paper
        self._rng = np.random.default_rng(seed)
        H, W = self.height, self.width
        self.ground = self.paper.reflectance()
        self.tooth = self.paper.tooth(H, W, self._rng)
        self.press = np.zeros((H, W), np.float32)
        self.held = np.zeros((H, W, HELD), np.float32)
        self.loose = np.zeros((H, W, LOOSE), np.float32)

    def copy(self) -> Sheet:
        """An independent copy: drawing on one leaves the other alone."""
        new = object.__new__(Sheet)
        new.__dict__.update(self.__dict__)
        new._rng = np.random.default_rng()
        new._rng.bit_generator.state = self._rng.bit_generator.state
        for name in ("tooth", "press", "held", "loose"):
            setattr(new, name, getattr(self, name).copy())
        return new

    # ------------------------------------------------------------------
    def pigment(self) -> np.ndarray:
        """Volume of pigment at each pixel, held and loose, (H, W)."""
        return self.held[..., IV] + self.loose[..., IV]

    def reflectance(self) -> np.ndarray:
        """Reflectance spectrum of the drawing, (H, W, N_BANDS) float32: the
        loose pigment over the held pigment over the paper."""
        out = np.empty((self.height, self.width, N), np.float32)
        for rows in self._row_chunks():
            out[rows] = self._reflectance_rows(rows)
        return out

    def _row_chunks(self) -> Iterator[slice]:
        for y in range(0, self.height, ROWS):
            yield slice(y, min(y + ROWS, self.height))

    def _reflectance_rows(self, rows: slice) -> np.ndarray:
        held, loose = self.held[rows], self.loose[rows]
        out = np.empty((*held.shape[:2], N))
        out[:] = self.ground
        marked = (held[..., IV] > 0) | (loose[..., IV] > 0)
        if marked.any():
            h, lo = held[marked].astype(np.float64), loose[marked].astype(np.float64)
            under = sp.km_layer(h[:, :N], h[:, IS], self.ground)
            out[marked] = sp.km_layer(lo[:, :N], lo[:, IS], under)
        return out

    # ------------------------------------------------------------------
    def render(
        self,
        lighting: bool = True,
        light: Sequence[float] | None = DEFAULT_LIGHT,
        relief: float = 1.0,
    ) -> np.ndarray:
        """Render to an sRGB float image (H, W, 3) in 0..1.

        ``lighting`` shades the tooth of the paper as under a lamp in the
        direction ``light`` (x, y, z; y points down the image; a smaller z
        rakes across the surface and brings the tooth out). ``light=None`` is
        the default lamp, ``DEFAULT_LIGHT``: fairly high, so the tooth shows
        faintly, as on a scan. ``relief`` scales the apparent depth of the
        tooth. With ``lighting=False`` the sheet is rendered flat.

        When lit, raises ValueError unless ``light`` is three finite numbers
        with z > 0 and ``relief`` is finite and not negative.
        """
        shade = None
        if lighting:
            lamp = _check_light(DEFAULT_LIGHT if light is None else light)
            if not (math.isfinite(relief) and relief >= 0):
                raise ValueError(f"relief must be finite and >= 0, got {relief!r}")
            shade = self._shade(lamp, float(relief))
        out = np.empty((self.height, self.width, 3), np.float32)
        for rows in self._row_chunks():
            lin = sp.reflectance_to_linear_rgb(self._reflectance_rows(rows))
            if shade is not None:
                lin *= shade[rows][..., None]
            out[rows] = sp.linear_to_srgb(np.clip(lin, 0.0, 1.0))
        return out

    def surface_height(self, relief: float = 1.0) -> np.ndarray:
        """Height of the paper's surface in pixels, (H, W), as it is lit."""
        scale = TOOTH_HEIGHT * self.paper.grain * self.paper.depth * relief
        return self.tooth * np.float32(scale)

    def _shade(self, lamp: np.ndarray, relief: float) -> np.ndarray:
        """Brightness of each pixel under the lamp, 1 where the paper is level."""
        height = self.surface_height(relief).astype(np.float64)
        gy, gx = np.gradient(height)
        facing = (lamp[2] - gx * lamp[0] - gy * lamp[1]) / np.sqrt(1.0 + gx**2 + gy**2)
        lit = np.maximum(facing, 0.0) / lamp[2]
        return (AMBIENT + (1.0 - AMBIENT) * lit).astype(np.float32)

    def save(self, path: str, **render_options) -> None:
        """Render and save as PNG."""
        img = self.render(**render_options)
        write_png(path, np.round(img * 255).astype(np.uint8))


def _size(name: str, value) -> int:
    """``value`` as a whole number of pixels above 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a whole number of pixels, got {value!r}")
    if not (math.isfinite(value) and value == int(value) and value > 0):
        raise ValueError(f"{name} must be a whole number of pixels above 0: {value!r}")
    return int(value)


def _check_light(light) -> np.ndarray:
    """``light`` as a unit vector (float array of 3), or ValueError."""
    try:
        L = np.asarray(light, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"light must be three numbers (x, y, z), got {light!r}") from e
    if L.shape != (3,) or not np.isfinite(L).all():
        raise ValueError(f"light must be three finite numbers (x, y, z): {light!r}")
    if not L[2] > 0:
        raise ValueError(
            f"light must shine from in front of the sheet (z > 0), got {light!r}"
        )
    return L / np.linalg.norm(L)
