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
from ._engine import HELD, IS, IV, IX, LOOSE, N
from .medium import Medium
from .paper import Paper
from .pens import Pen
from .stick import Stick
from .tools import KNEADED, STUMP, Tool

__all__ = ["DEFAULT_LIGHT", "HELD", "IS", "IV", "IX", "LOOSE", "N", "Sheet"]

#: The lamp a sheet is rendered under by default (x, y, z; y points down the
#: image): from the upper left and fairly high, so that the tooth shows
#: faintly, as on a scan. A smaller z rakes across the surface.
DEFAULT_LIGHT = (-0.6, -0.8, 2.4)
#: Share of the light that comes from all around rather than from the lamp.
AMBIENT = 0.3
#: Height of the deepest tooth of a paper of depth 1, as a share of its grain.
TOOTH_HEIGHT = 0.12
#: Shine: what a film as shiny as it gets adds to the light a pixel sends
#: back (linear, the paper's white being 1) when it mirrors the lamp squarely.
SHEEN = 0.06
#: A held film of this much sheen per covered area is as shiny as it gets,
#: and matte dust this thick over it hides all but 1/e of its shine.
SHEEN_FULL = 0.6
SHEEN_HIDE = 0.1
#: How narrow the mirrored light is: graphite is satin, not a mirror.
SHININESS = 12.0
#: The lines of a pen's hatching are this many of its widths apart: side by
#: side, where a pencil's strokes overlap.
PEN_SPACING = 1.6
#: Fixative wets the dust it sets, which then scatters this much less light.
FIXATIVE_DARKENS = 0.15
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
    pigment fixed to the paper (``held``) and the pigment lying on it as dust,
    which is what a stump, an eraser or a later stroke can still move: in two
    layers, ``loose`` on top and ``under`` beneath it, so that what was laid
    last is seen over what was laid before. ``cover`` is the share of the
    pixel that carries pigment: the rest shows bare paper. Memory: about 0.34
    KB per pixel.

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
        self.under = np.zeros((H, W, LOOSE), np.float32)
        self.loose = np.zeros((H, W, LOOSE), np.float32)
        # the dust of the pass being drawn, until it settles (empty between
        # strokes)
        self._fresh = np.zeros((H, W), np.float32)
        # the volume of dust that fixative has set at each pixel, in the
        # dust underneath and in the dust on top
        self.fixed = np.zeros((H, W, 2), np.float32)
        # the ink of pens: a stain in the paper, packed as dust is, and the
        # share of each pixel it is on (None until a pen has drawn)
        self.ink: np.ndarray | None = None
        self.inked: np.ndarray | None = None

    def copy(self) -> Sheet:
        """An independent copy: drawing on one leaves the other alone."""
        new = object.__new__(Sheet)
        new.__dict__.update(self.__dict__)
        new._rng = np.random.default_rng()
        new._rng.bit_generator.state = self._rng.bit_generator.state
        names = ("tooth", "press", "cover", "held", "under", "loose", "_fresh", "fixed")
        for name in names:
            setattr(new, name, getattr(self, name).copy())
        if self.ink is not None and self.inked is not None:
            new.ink, new.inked = self.ink.copy(), self.inked.copy()
        return new

    # ------------------------------------------------------------------
    def pigment(self) -> np.ndarray:
        """Volume of pigment at each pixel, held and loose, (H, W)."""
        return self.held[..., IV] + self.under[..., IV] + self.loose[..., IV]

    def shine(self) -> np.ndarray:
        """How much of each pixel mirrors the light, 0..1, (H, W): where the
        paper holds a film of a medium with sheen (graphite), the more the
        thicker it is, and the less the more dust of another kind lies over
        it."""
        cover = np.maximum(self.cover, np.float32(1e-6))
        film = np.minimum(self.held[..., IX] / cover / np.float32(SHEEN_FULL), 1.0)
        dust = self.under[..., IV] + self.loose[..., IV]
        matte = np.maximum(dust - self.under[..., IX] - self.loose[..., IX], 0.0)
        return self.cover * film * np.exp(-matte / cover / np.float32(SHEEN_HIDE))

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
        layers = (self.held[rows], self.under[rows], self.loose[rows])
        out = np.empty((*layers[0].shape[:2], N))
        out[:] = self.ground
        stained = self._stain(out, rows)
        marked = layers[0][..., IV] > 0
        for layer in layers[1:]:
            marked |= layer[..., IV] > 0
        if marked.any():
            # the pigment lies on the covered share of the pixel, that much
            # thicker; the rest of the pixel is bare paper (inked or not)
            share = np.clip(self.cover[rows][marked].astype(np.float64), 1e-6, 1.0)
            bare = out[marked] if stained else self.ground
            seen = bare
            for layer in layers:  # from the paper up
                packed = layer[marked].astype(np.float64) / share[:, None]
                seen = sp.km_layer(packed[:, :N], packed[:, IS], seen)
            out[marked] = share[:, None] * seen + (1.0 - share[:, None]) * bare
        return out

    def _stain(self, out: np.ndarray, rows: slice) -> bool:
        """Stain the paper of these rows (``out``, its reflectance) with the
        ink in it. Returns whether there was any."""
        if self.ink is None or self.inked is None:
            return False
        inked = self.inked[rows] > 0
        if not inked.any():
            return False
        share = np.clip(self.inked[rows][inked].astype(np.float64), 1e-6, 1.0)
        packed = self.ink[rows][inked].astype(np.float64) / share[:, None]
        seen = sp.km_layer(packed[:, :N], packed[:, IS], self.ground)
        out[inked] = share[:, None] * seen + (1.0 - share[:, None]) * self.ground
        return True

    # ------------------------------------------------------------------
    def stroke(
        self,
        path,
        medium: Medium | Stick | Pen,
        pressure: Profile = 0.5,
        width: float | None = None,
        tilt: float = 0.0,
    ) -> float:
        """Draw one pass of ``medium`` along ``path``.

        ``path`` is a list of points (a smooth curve through them) or a
        :class:`murex.Path`; a single point is a dot. ``medium`` is what to
        draw with, such as ``pencils.HB``, or a :class:`fusain.Stick`, whose
        tip wears as it draws. ``pressure`` (0 to 1, clipped) is how hard the
        tip bears on the paper, and what sets the tone: a light touch marks
        only the peaks of the tooth, a firm one reaches the valleys. A
        number, one value per point of the path, or a function of u (0 at the
        start of the stroke, 1 at its end).

        ``width`` is the width of the tip in pixels (4 by default; a stick
        has its own, so it takes none). A pencil point is a few times wider
        than the paper's grain; one narrower than the grain draws a dotted
        line. ``tilt`` (0 to 1) is how far over the stick is held: 0 draws
        with its point, 1 with its side, which marks a band several times
        wider than the point, across the stroke: broader, lighter and
        grainier, with soft edges.

        ``medium`` can also be a :class:`fusain.Pen`, a ballpoint: it has its
        own width and no side (``tilt`` stays 0), its line is nearly as dark
        at a light touch as at a firm one, and its ink is a stain in the
        paper (:attr:`ink`), under everything else, which no tool takes off.

        Returns the volume of medium laid, which is what the tip lost.

        Raises ValueError or TypeError, before anything is drawn, unless
        ``width`` is a finite number above 0 (and not given for a stick or a
        pen), the pressures are finite and ``tilt`` is between 0 and 1.
        """
        if isinstance(medium, Pen):
            return self._roll(path, medium, pressure, width, tilt)
        stick = medium if isinstance(medium, Stick) else None
        if stick is not None:
            if width is not None:
                raise TypeError("a Stick has its own width: give none with it")
            medium, tip = stick.medium, (stick.point, stick.lead, eng.WEAR, stick.worn)
            width = stick.width
        elif isinstance(medium, Medium):
            width = 4.0 if width is None else width
            tip = (width, width, 0.0, 0.0)
        else:
            raise TypeError(
                "medium must be a Medium, such as pencils.HB, a Stick or a Pen: "
                f"{medium!r}"
            )
        if not 0.0 <= tilt <= 1.0:
            raise ValueError(f"tilt must be between 0 (point) and 1 (side): {tilt!r}")
        xs, ys, bearing, steps = self._sampled(path, pressure, width)
        paper, pigment = self.paper, medium.pigment
        laid = float(
            eng.lay_stroke(
                self.tooth,
                self.press,
                self.cover,
                self.held,
                self.under,
                self.loose,
                self._fresh,
                self.fixed,
                xs,
                ys,
                bearing,
                steps,
                np.array([*tip, _spread(tilt), tilt], dtype=np.float64),
                np.ascontiguousarray(pigment.K, dtype=np.float64),
                np.array(
                    [
                        pigment.S,
                        medium.softness,
                        medium.binding,
                        medium.wax,
                        medium.sheen,
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [paper.depth, paper.bite, paper.give, paper.hold], dtype=np.float64
                ),
                int(self._rng.integers(1 << 31)),
            )
        )
        if stick is not None:
            stick.worn = max(stick.worn + laid, 0.0)
        return laid

    def _roll(self, path, pen: Pen, pressure: Profile, width, tilt) -> float:
        """One pass of a ballpoint pen along ``path``; see :meth:`stroke`."""
        if width is not None:
            raise TypeError("a Pen has its own width: give none with it")
        if tilt != 0.0:
            raise ValueError(
                f"a pen has no side to draw with: tilt must be 0: {tilt!r}"
            )
        xs, ys, bearing, steps = self._sampled(path, pressure, pen.width)
        if self.ink is None or self.inked is None:
            self.ink = np.zeros((self.height, self.width, LOOSE), np.float32)
            self.inked = np.zeros((self.height, self.width), np.float32)
        paper = self.paper
        return float(
            eng.roll_stroke(
                self.tooth,
                self.press,
                self.cover,
                self.held,
                self.under,
                self.loose,
                self.ink,
                self.inked,
                xs,
                ys,
                bearing,
                steps,
                0.5 * float(pen.width),
                np.ascontiguousarray(pen.ink.K, dtype=np.float64),
                np.array([pen.ink.S, pen.flow], dtype=np.float64),
                np.array(
                    [paper.depth, paper.bite, paper.give, paper.hold], dtype=np.float64
                ),
                int(self._rng.integers(1 << 31)),
            )
        )

    # ------------------------------------------------------------------
    def render(
        self,
        lighting: bool = True,
        light: Sequence[float] | None = DEFAULT_LIGHT,
        relief: float = 1.0,
        gloss: float = 1.0,
    ) -> np.ndarray:
        """Render to an sRGB float image (H, W, 3) in 0..1.

        ``lighting`` shades the tooth of the paper as under a lamp in the
        direction ``light`` (x, y, z; y points down the image; a smaller z
        rakes across the surface and brings the tooth out). ``light=None`` is
        the default lamp, ``DEFAULT_LIGHT``: fairly high, so the tooth shows
        faintly, as on a scan. ``relief`` scales the apparent depth of the
        tooth. With ``lighting=False`` the sheet is rendered flat.

        The lamp also shows the shine of graphite: seen from straight above,
        a heavy layer mirrors a high lamp and looks silvery, and hardly
        mirrors a raking one. ``gloss`` scales that shine: 0 renders the
        drawing matte, as a scanner sees it.

        When lit, raises ValueError unless ``light`` is three finite numbers
        with z > 0 and ``relief`` and ``gloss`` are finite and not negative.
        """
        shade = glint = None
        if lighting:
            lamp = _check_light(DEFAULT_LIGHT if light is None else light)
            for name, value in (("relief", relief), ("gloss", gloss)):
                if not (math.isfinite(value) and value >= 0):
                    raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
            slopes = self._slopes(float(relief))
            shade = self._shade(lamp, slopes)
            if gloss > 0 and self.held[..., IX].any():
                glint = self._glint(lamp, slopes) * np.float32(SHEEN * gloss)
        out = np.empty((self.height, self.width, 3), np.float32)
        for rows in self._row_chunks():
            lin = sp.reflectance_to_linear_rgb(self._reflectance_rows(rows))
            if shade is not None:
                lin *= shade[rows][..., None]
            if glint is not None:
                lin += glint[rows][..., None]
            out[rows] = sp.linear_to_srgb(np.clip(lin, 0.0, 1.0))
        return out

    def surface_height(self, relief: float = 1.0) -> np.ndarray:
        """Height of the paper's surface in pixels, (H, W), as it is lit: its
        tooth, level with its peaks where it has been pressed flat."""
        scale = TOOTH_HEIGHT * self.paper.grain * self.paper.depth * relief
        ground = self.tooth + (1.0 - self.tooth) * self.press
        return ground * np.float32(scale)

    def _slopes(self, relief: float) -> tuple[np.ndarray, np.ndarray]:
        """The slope of the surface along x and along y, each (H, W)."""
        height = self.surface_height(relief).astype(np.float64)
        # (a sheet one pixel wide or high has no slope that way)
        gy = np.gradient(height, axis=0) if self.height > 1 else np.zeros_like(height)
        gx = np.gradient(height, axis=1) if self.width > 1 else np.zeros_like(height)
        return gx, gy

    @staticmethod
    def _facing(towards: np.ndarray, slopes: tuple[np.ndarray, np.ndarray]):
        """Cosine of the angle between the surface and a unit direction."""
        gx, gy = slopes
        facing = towards[2] - gx * towards[0] - gy * towards[1]
        return np.maximum(facing / np.sqrt(1.0 + gx**2 + gy**2), 0.0)

    def _shade(self, lamp: np.ndarray, slopes) -> np.ndarray:
        """Brightness of each pixel under the lamp, 1 where the paper is level."""
        lit = self._facing(lamp, slopes) / lamp[2]
        return (AMBIENT + (1.0 - AMBIENT) * lit).astype(np.float32)

    def _glint(self, lamp: np.ndarray, slopes) -> np.ndarray:
        """How much of the lamp each pixel mirrors to an eye straight above
        the sheet, 0..1: its shine, times how squarely its surface faces
        halfway between the lamp and the eye."""
        halfway = lamp + np.array([0.0, 0.0, 1.0])
        halfway /= np.linalg.norm(halfway)
        mirrored = self._facing(halfway, slopes) ** SHININESS
        return (self.shine() * mirrored).astype(np.float32)

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

    def rub(
        self,
        area,
        tool: Tool = STUMP,
        pressure: hatching.Pressure = 0.5,
        width: float | None = None,
        angle: float = 0.0,
        spacing: float | None = None,
        hand: float = 1.0,
    ) -> float:
        """Rub all of ``area`` with a stump, a finger or an eraser: passes
        side by side, to and fro, as a hand blends a sky or lifts a light
        out of a tone. It is to :meth:`smudge` and :meth:`erase` what
        :meth:`shade` is to :meth:`stroke`.

        ``area`` is a rectangle (x0, y0, x1, y1) or the corners of a
        polygon; ``pressure`` a number or a function of (x, y), as for
        :meth:`shade`; ``width`` the width of the tool (by default its own);
        ``angle`` the direction of the passes in degrees; ``spacing`` how far
        apart they are (by default half a width); ``hand`` how unsteady the
        hand is. Where the pressure is nothing for a stretch, the tool is
        lifted: it rubs on from clean, and what it carried is counted with
        what it took.

        Returns the volume of pigment the tool took off the sheet in all.

        Raises ValueError or TypeError, before anything is rubbed, as
        :meth:`shade` does.
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"tool must be a Tool, such as tools.STUMP, got {tool!r}")
        width = tool.width if width is None else width
        outline = hatching.as_outline(area)
        gap = 0.5 * width if spacing is None else spacing
        _check_hatching(width, gap, angle, hand)
        taken = 0.0
        passes = hatching.strokes(outline, angle, width, gap, hand, self._rng)
        for k, (path, firm, _) in enumerate(passes):
            path = path[::-1] if k % 2 else path  # to and fro
            along = hatching.bearing(pressure, path, firm, hand)
            points, _, u, _ = spline.as_path(path).sample(max(1.0, width / 8.0))
            touch = np.broadcast_to(np.asarray(along(u), dtype=np.float64), u.shape)
            marks = np.nonzero(touch > 0.0)[0]
            if marks.size < 2:
                continue
            for run in np.split(marks, np.nonzero(np.diff(marks) > 1)[0] + 1):
                if run.size >= 2:
                    taken += self._rub(points[run], tool, touch[run].tolist(), width)
        return taken

    def fix(self, amount: float = 1.0) -> float:
        """Spray fixative over the whole sheet: it sets the share ``amount``
        (0 to 1) of the dust that was still loose. Dust that is set stays
        where it is: a stump, a finger or a kneaded eraser hardly moves it,
        and only a hard eraser takes it off, as it does what the paper holds.
        It still shows as it lay, one layer over the other, a little darker:
        fixative wets the pigment, which then scatters less light. And it
        gives some tooth back, so that a layer that took no more takes some.

        What is drawn afterwards lies over it, as loose as ever.

        Returns the volume of dust newly set. Raises ValueError unless
        ``amount`` is between 0 and 1.
        """
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"amount must be between 0 and 1, got {amount!r}")
        total = 0.0
        for which, layer in enumerate((self.under, self.loose)):
            volume = layer[..., IV]
            newly = np.maximum(volume - self.fixed[..., which], 0.0)
            newly *= np.float32(amount)
            # (only what is newly set darkens)
            share = newly / np.maximum(volume, np.float32(1e-9))
            layer[..., IS] *= 1.0 - np.float32(FIXATIVE_DARKENS) * share
            self.fixed[..., which] += newly
            total += float(newly.sum(dtype=np.float64))
        return total

    def _rub(self, path, tool: Tool, pressure: Profile, width: float | None) -> float:
        if not isinstance(tool, Tool):
            raise TypeError(f"tool must be a Tool, such as tools.STUMP, got {tool!r}")
        width = tool.width if width is None else width
        xs, ys, bearing, steps = self._sampled(path, pressure, width)
        load = eng.rub_stroke(
            self.cover,
            self.held,
            self.under,
            self.loose,
            self.fixed,
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
        medium: Medium | Stick | Pen,
        pressure: hatching.Pressure = 0.5,
        width: float | None = None,
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
        across the area. ``width`` is the width of the tip in pixels (6 by
        default; a :class:`fusain.Stick` has its own, and wears as the
        shading goes) and ``angle`` the direction of the strokes in degrees
        (0 runs across the sheet, 90 up it). ``spacing`` is how far apart they
        are (by default a little over half a width, so that they overlap).
        ``hand`` is how
        unsteady the hand is: 1 for a natural one, 0 for ruler-straight
        strokes, evenly spaced and all alike. ``tilt`` is as for
        :meth:`stroke`: with the side of the stick the strokes are broader
        (and fewer), lighter and softer.

        Shade again at another angle to cross-hatch or to build tone.
        Returns the volume of medium laid.

        Raises ValueError or TypeError, before anything is drawn, unless
        ``width`` (not given for a stick) and ``spacing`` are finite numbers
        above 0, ``angle`` is finite, ``hand`` is finite and not negative and
        ``tilt`` is between 0 and 1.
        """
        own = isinstance(medium, Stick | Pen)  # (it has its own width)
        if own:
            if width is not None:
                kind = type(medium).__name__
                raise TypeError(f"a {kind} has its own width: give none with it")
            width = medium.width
        elif isinstance(medium, Medium):
            width = 6.0 if width is None else width
        else:
            raise TypeError(
                "medium must be a Medium, such as pencils.HB, a Stick or a Pen: "
                f"{medium!r}"
            )
        if not 0.0 <= tilt <= 1.0:
            raise ValueError(f"tilt must be between 0 (point) and 1 (side): {tilt!r}")
        outline = hatching.as_outline(area)
        broad = width * _spread(tilt)  # how wide a stroke's mark is
        apart = PEN_SPACING if isinstance(medium, Pen) else hatching.SPACING
        gap = apart * broad if spacing is None else spacing
        _check_hatching(width, gap, angle, hand)
        laid = 0.0
        for path, firm, tip in hatching.strokes(
            outline, angle, width, gap, hand, self._rng, broad
        ):
            along = hatching.bearing(pressure, path, firm, hand)
            # (a stick's tip is as wide as it has worn, whatever the hand)
            laid += self.stroke(path, medium, along, None if own else tip, tilt)
        return laid

    def save(self, path: str, **render_options) -> None:
        """Render and save as PNG."""
        img = self.render(**render_options)
        write_png(path, np.round(img * 255).astype(np.uint8))


def _check_hatching(width: float, gap: float, angle: float, hand: float) -> None:
    """ValueError unless strokes can be laid side by side with these."""
    for name, value in (("width", width), ("spacing", gap)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a number of pixels above 0: {value!r}")
    if not math.isfinite(angle):
        raise ValueError(f"angle must be a finite number of degrees: {angle!r}")
    if not (math.isfinite(hand) and hand >= 0):
        raise ValueError(f"hand must be 0 (ruler-straight) or more, got {hand!r}")


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
