"""A sheet of paper and what has been drawn on it."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
from murex import spectral as sp
from murex import spline, write_png
from murex.spline import Profile

from . import _engine as eng
from . import hatching
from ._engine import HELD, IS, IV, LOOSE, N
from .medium import Medium
from .paper import Paper
from .tools import KNEADED, STUMP, Tool

__all__ = ["DEFAULT_LIGHT", "HELD", "IS", "IV", "LOOSE", "N", "Sheet"]

#: The lamp a sheet is rendered under by default (x, y, z; y points down the
#: image): from the upper left and fairly high, so that the tooth shows
#: faintly, as on a scan. A smaller z rakes across the surface.
DEFAULT_LIGHT = (-0.6, -0.8, 2.4)
#: Share of the light that comes from all around rather than from the lamp.
AMBIENT = 0.3
#: Height of the deepest tooth of a paper of depth 1, as a share of its grain.
TOOTH_HEIGHT = 0.12
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
    move, and the share of the pixel that carries pigment (``cover``): the
    rest shows bare paper. Memory: about 0.23 KB per pixel.

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
        self.cover = np.zeros((H, W), np.float32)
        self.held = np.zeros((H, W, HELD), np.float32)
        self.loose = np.zeros((H, W, LOOSE), np.float32)

    def copy(self) -> Sheet:
        """An independent copy: drawing on one leaves the other alone."""
        new = object.__new__(Sheet)
        new.__dict__.update(self.__dict__)
        new._rng = np.random.default_rng()
        new._rng.bit_generator.state = self._rng.bit_generator.state
        for name in ("tooth", "press", "cover", "held", "loose"):
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
            # the pigment lies on the covered share of the pixel, that much
            # thicker; the rest of the pixel is bare paper
            share = np.clip(self.cover[rows][marked].astype(np.float64), 1e-6, 1.0)
            h = held[marked].astype(np.float64) / share[:, None]
            lo = loose[marked].astype(np.float64) / share[:, None]
            under = sp.km_layer(h[:, :N], h[:, IS], self.ground)
            over = sp.km_layer(lo[:, :N], lo[:, IS], under)
            out[marked] = share[:, None] * over + (1.0 - share[:, None]) * self.ground
        return out

    # ------------------------------------------------------------------
    def stroke(
        self,
        path,
        medium: Medium,
        pressure: Profile = 0.5,
        width: float = 4.0,
        tilt: float = 0.0,
    ) -> float:
        """Draw one pass of ``medium`` along ``path``.

        ``path`` is a list of points (a smooth curve through them) or a
        :class:`murex.Path`; a single point is a dot. ``pressure`` (0 to 1,
        clipped) is how hard the tip bears on the paper, and what sets the
        tone: a light touch marks only the peaks of the tooth, a firm one
        reaches the valleys. A number, one value per point of the path, or a
        function of u (0 at the start of the stroke, 1 at its end).
        ``width`` is the width of the tip in pixels. A pencil point is a few
        times wider than the paper's grain; one narrower than the grain draws
        a dotted line. ``tilt`` (0 to 1) is how far over the stick is held: 0
        draws with its point, 1 with its side, which marks a band several
        times wider than the point, across the stroke: broader, lighter and
        grainier, with soft edges.

        Returns the volume of medium laid, which is what the tip lost.

        Raises ValueError, before anything is drawn, unless ``width`` is a
        finite number above 0, the pressures are finite and ``tilt`` is
        between 0 and 1.
        """
        if not isinstance(medium, Medium):
            raise TypeError(f"medium must be a Medium, such as pencils.HB: {medium!r}")
        if not 0.0 <= tilt <= 1.0:
            raise ValueError(f"tilt must be between 0 (point) and 1 (side): {tilt!r}")
        xs, ys, bearing, steps = self._sampled(path, pressure, width)
        paper, pigment = self.paper, medium.pigment
        return float(
            eng.lay_stroke(
                self.tooth,
                self.cover,
                self.held,
                self.loose,
                xs,
                ys,
                bearing,
                steps,
                0.5 * float(width) * _spread(tilt),
                0.5 * float(width),
                float(tilt),
                np.ascontiguousarray(pigment.K, dtype=np.float64),
                float(pigment.S),
                float(medium.softness),
                float(medium.binding),
                float(paper.depth),
                float(paper.bite),
                float(paper.give),
                int(self._rng.integers(1 << 31)),
            )
        )

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
        # (a sheet one pixel wide or high has no slope that way)
        gy = np.gradient(height, axis=0) if self.height > 1 else np.zeros_like(height)
        gx = np.gradient(height, axis=1) if self.width > 1 else np.zeros_like(height)
        facing = (lamp[2] - gx * lamp[0] - gy * lamp[1]) / np.sqrt(1.0 + gx**2 + gy**2)
        lit = np.maximum(facing, 0.0) / lamp[2]
        return (AMBIENT + (1.0 - AMBIENT) * lit).astype(np.float32)

    def smudge(
        self,
        path,
        tool: Tool = STUMP,
        pressure: Profile = 0.5,
        width: float | None = None,
    ) -> float:
        """Rub one pass of a stump, a finger or another :class:`fusain.Tool`
        along ``path``: it drags the dust along, presses some of it into the
        paper and pushes it into the valleys of the tooth, which softens and
        greys what was drawn. What the paper holds hardly moves.

        ``path`` and ``pressure`` are as for :meth:`stroke`; ``width`` is the
        width of the tool in pixels (by default the tool's own).

        Returns the volume of pigment left on the tool at the end.
        """
        return self._rub(path, tool, pressure, width)

    def erase(
        self,
        path,
        tool: Tool = KNEADED,
        pressure: Profile = 0.5,
        width: float | None = None,
    ) -> float:
        """Rub one pass of an eraser along ``path``. A kneaded eraser
        (``tools.KNEADED``, the default) lifts the dust and a little of what
        the paper holds; a vinyl one (``tools.ERASER``) takes nearly all.

        ``path`` and ``pressure`` are as for :meth:`stroke`; ``width`` is the
        width of the eraser in pixels (by default the tool's own).

        Returns the volume of pigment taken off the sheet.
        """
        return self._rub(path, tool, pressure, width)

    def _rub(self, path, tool: Tool, pressure: Profile, width: float | None) -> float:
        if not isinstance(tool, Tool):
            raise TypeError(f"tool must be a Tool, such as tools.STUMP, got {tool!r}")
        width = tool.width if width is None else width
        xs, ys, bearing, steps = self._sampled(path, pressure, width)
        load = eng.rub_stroke(
            self.cover,
            self.held,
            self.loose,
            xs,
            ys,
            bearing,
            steps,
            0.5 * float(width),
            float(tool.lift_loose),
            float(tool.lift_held),
            float(tool.lay),
            float(tool.press),
            float(tool.smear),
            int(self._rng.integers(1 << 31)),
        )
        return float(load[IV])

    def _sampled(self, path, pressure: Profile, width: float):
        """``path`` sampled for a tip of ``width``: what the kernels take (see
        :func:`fusain._engine.as_kernel_arrays`), or ValueError."""
        if not (math.isfinite(width) and width > 0):
            raise ValueError(f"width must be a number of pixels above 0, got {width!r}")
        line = spline.as_path(path)
        spacing = max(0.25, width / 8.0)
        points, _, u, knot = line.sample(spacing)
        bearing = spline.evaluate_profile(pressure, u, knot, line.n_knots)
        if not np.isfinite(bearing).all():
            raise ValueError("pressure must be finite all along the stroke")
        if len(points) > 1:
            spacing = line.length / (len(points) - 1)
        return eng.as_kernel_arrays(points, np.clip(bearing, 0.0, 1.0), spacing)

    def shade(
        self,
        area,
        medium: Medium,
        pressure: hatching.Pressure = 0.5,
        width: float = 6.0,
        angle: float = 30.0,
        spacing: float | None = None,
        hand: float = 1.0,
        tilt: float = 0.0,
    ) -> float:
        """Shade ``area`` with one layer of strokes side by side, as a hand
        does: none quite straight or quite alike, each lighter where the tip
        lands and where it lifts.

        ``area`` is a rectangle (x0, y0, x1, y1) or a polygon (three or more
        points). ``pressure`` is a number, or a function of where on the
        sheet (called with arrays of x and of y), for a tone that changes
        across the area. ``width`` is the width of the tip in pixels and
        ``angle`` the direction of the strokes in degrees (0 runs across the
        sheet, 90 up it). ``spacing`` is how far apart they are (by default
        a little over half a width, so that they overlap). ``hand`` is how
        unsteady the hand is: 1 for a natural one, 0 for ruler-straight
        strokes, evenly spaced and all alike. ``tilt`` is as for
        :meth:`stroke`: with the side of the stick the strokes are broader
        (and fewer), lighter and softer.

        Shade again at another angle to cross-hatch or to build tone.
        Returns the volume of medium laid.

        Raises ValueError or TypeError, before anything is drawn, unless
        ``width`` and ``spacing`` are finite numbers above 0, ``angle`` is
        finite and ``hand`` is finite and not negative.
        """
        if not isinstance(medium, Medium):
            raise TypeError(f"medium must be a Medium, such as pencils.HB: {medium!r}")
        if not 0.0 <= tilt <= 1.0:
            raise ValueError(f"tilt must be between 0 (point) and 1 (side): {tilt!r}")
        outline = hatching.as_outline(area)
        broad = width * _spread(tilt)  # how wide a stroke's mark is
        gap = hatching.SPACING * broad if spacing is None else spacing
        for name, value in (("width", width), ("spacing", gap)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be a number of pixels above 0: {value!r}"
                )
        if not math.isfinite(angle):
            raise ValueError(f"angle must be a finite number of degrees: {angle!r}")
        if not (math.isfinite(hand) and hand >= 0):
            raise ValueError(f"hand must be 0 (ruler-straight) or more, got {hand!r}")
        laid = 0.0
        for path, firm, tip in hatching.strokes(
            outline, angle, width, gap, hand, self._rng, broad
        ):
            along = hatching.bearing(pressure, path, firm, hand)
            laid += self.stroke(path, medium, along, tip, tilt)
        return laid

    def save(self, path: str, **render_options) -> None:
        """Render and save as PNG."""
        img = self.render(**render_options)
        write_png(path, np.round(img * 255).astype(np.uint8))


def _spread(tilt: float) -> float:
    """How many times wider than its point a stick marks at this tilt."""
    return 1.0 + (eng.SIDE - 1.0) * float(tilt)


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
