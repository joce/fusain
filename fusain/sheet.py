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
from ._engine import FILM, HELD, IS, IV, IX, KINDS, LOOSE, PASTE, SET, N
from .medium import Medium
from .paper import Paper
from .pens import Pen
from .smudger import Smudger
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
    pigment fixed to the paper (``held``) and the pigment lying on it, in two
    layers, ``loose`` on top and ``under`` beneath it, so that what was laid
    last is seen over what was laid before. Most of that is dust, which is
    what a stump, an eraser or a later stroke can still move; ``bonded``
    holds how much of each layer is not: dust that is set (:attr:`fixed`), a
    film of wax laid over the dust (:attr:`film`) and greasy paste
    (:attr:`paste`). ``cover`` is the share of the pixel that carries
    pigment: the rest shows bare paper. Memory: about 0.36 KB per pixel.

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
        # what the pass being drawn leaves to lie on the sheet, until it
        # settles (empty between strokes): what the paper does not hold of
        # it, and its film
        self._fresh = np.zeros((H, W, 2), np.float32)
        # of what lies in the layer underneath (0) and in the layer on top
        # (1), the volume at each pixel that is not loose dust: set, film
        # and paste
        self.bonded = np.zeros((H, W, 2, KINDS), np.float32)
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
        names = ("tooth", "press", "cover", "held", "under", "loose", "_fresh")
        for name in (*names, "bonded"):
            setattr(new, name, getattr(self, name).copy())
        if self.ink is not None and self.inked is not None:
            new.ink, new.inked = self.ink.copy(), self.inked.copy()
        return new

    # ------------------------------------------------------------------
    @property
    def fixed(self) -> np.ndarray:
        """Volume of the dust that is set, (H, W, 2): in the layer underneath
        and in the layer on top. Fixative sets dust, and so does a film of
        wax laid over it. A view: writing to it changes the sheet."""
        return self.bonded[..., SET]

    @property
    def film(self) -> np.ndarray:
        """Volume of film in each of the two layers, (H, W, 2): what a waxy
        medium (a coloured pencil, a crayon, an oil pastel) laid not on the
        paper but on what lay over it. It stays on top of that, and no stump
        or kneaded eraser moves it. A view, as :attr:`fixed` is."""
        return self.bonded[..., FILM]

    @property
    def paste(self) -> np.ndarray:
        """Volume of paste in each of the two layers, (H, W, 2): what the
        paper does not hold of a greasy medium (an oil pastel), and the dust
        it was laid over and took up. A finger smears it; a kneaded eraser
        does not lift it. A view, as :attr:`fixed` is."""
        return self.bonded[..., PASTE]

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
        """Reflectance spectrum of the drawing, (H, W, N_BANDS) float32: what
        lies on top over what lies underneath over the held pigment over the
        paper."""
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
        width: Profile | None = None,
        tilt: Profile = 0.0,
        ends: Ends = "round",
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
        line. Like the pressure it can change along the stroke (one value per
        point of the path, or a function of u), for a stroke that tapers or
        that recedes in perspective: it is the width of the tip where the tip
        is, and what it lays per pixel does not change with it. ``tilt`` (0
        to 1, also a number or a profile) is how far over the stick is held:
        0 draws with its point, 1 with its side, which marks a band across
        the stroke as wide as the side is long (the medium's ``side`` times
        the width of the tip; a stick's own ``side``): broader, lighter and
        grainier, with soft edges. A stick turned over as it goes widens its
        mark from its point to its side.

        ``ends`` is how the stroke ends: ``"round"`` (the default), in the
        outline of the tip, which reaches half its length past the ends of
        the path; or ``"flat"``, cut square across at the ends of the path,
        as the end or the side of a stick drawn and lifted cleanly leaves
        it: a flat stroke is as long as its path. A pair, such as
        ``("flat", "round")``, says it for the start and for the end. Either
        way the tip lays as much where it lands and where it lifts as
        anywhere along the stroke, so a stroke is as strong at its ends as
        in its middle, and a short one as strong as a long one.

        ``medium`` can also be a :class:`fusain.Pen`, a ballpoint: it has its
        own width and no side (``tilt`` stays 0), and its ends are round;
        its line is nearly as dark
        at a light touch as at a firm one, and its ink is a stain in the
        paper (:attr:`ink`), under everything else, which no tool takes off.

        What the paper holds of the medium goes under the dust that lies
        there, in the tooth; but a waxy medium (a coloured pencil, a crayon,
        an oil pastel) is a film, which lies on the dust (:attr:`film`) and
        sets part of it (:attr:`fixed`).

        Returns the volume of medium laid, which is what the tip lost.

        Raises ValueError or TypeError, before anything is drawn, unless
        ``width`` is finite and above 0 all along (and not given for a stick
        or a pen), the pressures are finite, ``tilt`` is between 0 and 1 all
        along and ``ends`` is ``"round"``, ``"flat"`` or a pair of those.
        """
        if isinstance(medium, Pen):
            return self._roll(path, medium, pressure, width, tilt, ends)
        if isinstance(medium, Stick):
            if width is not None:
                raise TypeError("a Stick has its own width: give none with it")
            stick, tip = medium, (medium.lead, eng.WEAR, medium.worn)
            # (sampled for the tip as it is now; it wears from its point)
            width, point = medium.width, medium.point
        elif isinstance(medium, Medium):
            stick, tip = None, (math.inf, 0.0, 0.0)
            width = point = 4.0 if width is None else width
        else:
            raise TypeError(
                "medium must be a Medium, such as pencils.HB, a Stick or a Pen: "
                f"{medium!r}"
            )
        square = _square(ends)
        known = medium.medium if isinstance(medium, Stick) else medium
        line, points, u, knot, spacing, pace = _sample(path, width)
        bearing = _profile("pressure", pressure, line, u, knot)
        xs, ys, bearing, steps = eng.as_kernel_arrays(
            points, np.clip(bearing, 0.0, 1.0), spacing
        )
        widths = _profile("width", point, line, u, knot)
        if not (widths > 0.0).all():
            raise ValueError("width must be a number of pixels above 0 all along")
        tilts = spline.evaluate_profile(tilt, u, knot, line.n_knots)
        tilts = np.broadcast_to(np.asarray(tilts, dtype=np.float64), u.shape)
        if not ((tilts >= 0.0) & (tilts <= 1.0)).all():
            raise ValueError(f"tilt must be between 0 (point) and 1 (side): {tilt!r}")
        sides = known.side * widths if stick is None else np.full_like(u, stick.side)
        shapes = np.ascontiguousarray(np.stack([widths, tilts, sides], axis=1))
        if len(points) == 1:
            square = (0.0, 0.0)  # (a dot is the outline of the tip)
        paper, pigment = self.paper, known.pigment
        laid = float(
            eng.lay_stroke(
                self.tooth,
                self.press,
                self.cover,
                self.held,
                self.under,
                self.loose,
                self._fresh,
                self.bonded,
                xs,
                ys,
                bearing,
                steps,
                shapes,
                np.array([*tip, spacing, *square, pace], dtype=np.float64),
                np.ascontiguousarray(pigment.K, dtype=np.float64),
                np.array(
                    [
                        pigment.S,
                        known.softness,
                        known.binding,
                        known.wax,
                        known.sheen,
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

    def _roll(self, path, pen: Pen, pressure: Profile, width, tilt, ends) -> float:
        """One pass of a ballpoint pen along ``path``; see :meth:`stroke`."""
        if width is not None:
            raise TypeError("a Pen has its own width: give none with it")
        if not (isinstance(tilt, int | float) and tilt == 0.0):
            raise ValueError(
                f"a pen has no side to draw with: tilt must be 0: {tilt!r}"
            )
        if _square(ends) != (0.0, 0.0):
            raise ValueError(f"a ballpoint's line ends round: {ends!r}")
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
        tool: Tool | Smudger = STUMP,
        pressure: Profile = 0.5,
        width: float | None = None,
    ) -> float:
        """Rub one pass of a stump, a finger or another :class:`fusain.Tool`
        along ``path``: it drags the dust along, presses some of it into the
        paper and pushes it into the valleys of the tooth, which softens and
        greys what was drawn. What the paper holds hardly moves, but for
        graphite, which slides.

        ``path`` and ``pressure`` are as for :meth:`stroke`; ``width`` is the
        width of the tool in pixels (by default the tool's own).

        A bare tool is clean when it lands, so it leaves the paper lighter
        where it does, and it takes away what it still carries when it lifts.
        A :class:`fusain.Smudger` is a tool in the hand: it lands with what
        it picked up before and keeps what it carries, so that once loaded it
        gives back about as much as it takes, and tones bare paper. It has
        its own width, so it takes none.

        Returns the volume of pigment the tool took off the sheet, less what
        it laid back: for a bare tool, what was left on it at the end; for a
        ``Smudger`` that left more than it took, less than nothing.
        """
        return self._rub(path, _in_hand(tool, width), pressure)

    def erase(
        self,
        path,
        tool: Tool | Smudger = KNEADED,
        pressure: Profile = 0.5,
        width: float | None = None,
    ) -> float:
        """Rub one pass of an eraser along ``path``. A kneaded eraser
        (``tools.KNEADED``, the default) lifts the dust and a little of what
        the paper holds; a vinyl one (``tools.ERASER``) takes nearly all.
        The paste of an oil pastel, a film of wax laid over dust and the
        dust set under it come off as what the paper holds does.

        ``path`` and ``pressure`` are as for :meth:`stroke`; ``width`` is the
        width of the eraser in pixels (by default the tool's own; a
        :class:`fusain.Smudger` has its own and takes none).

        Returns the volume of pigment taken off the sheet.
        """
        return self._rub(path, _in_hand(tool, width), pressure)

    def rub(
        self,
        area,
        tool: Tool | Smudger = STUMP,
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
        hand is.

        The tool keeps what it carries from one pass to the next, and over
        the stretches where the pressure is nothing and it is lifted: it
        lands with it, so past the first few widths it gives back about as
        much as it takes, and the tone is blended, not worn away. A bare
        tool starts clean and takes away what it carries at the end; a
        :class:`fusain.Smudger` (which has its own width and takes none)
        starts with what it picked up before, and keeps it.

        Returns the volume of pigment the tool took off the sheet in all,
        less what it laid back (less than nothing for a ``Smudger`` that
        left more than it took).

        Raises ValueError or TypeError, before anything is rubbed, as
        :meth:`shade` does.
        """
        smudger = _in_hand(tool, width)
        width = smudger.width
        outline = hatching.as_outline(area)
        gap = 0.5 * width if spacing is None else spacing
        _check_hatching(width, gap, angle, hand)
        passes = []  # (what each pass presses, worked out before any rubs)
        strokes = hatching.strokes(outline, angle, width, gap, hand, self._rng)
        for k, (path, firm, _) in enumerate(strokes):
            path = path[::-1] if k % 2 else path  # to and fro
            along = hatching.bearing(pressure, path, firm, hand)
            points, _, u, _ = spline.as_path(path).sample(max(1.0, width / 8.0))
            touch = np.broadcast_to(np.asarray(along(u), dtype=np.float64), u.shape)
            if not np.isfinite(touch).all():
                raise ValueError("pressure must be finite all over the area")
            passes.append((points, touch))
        taken = 0.0
        for points, touch in passes:
            marks = np.nonzero(touch > 0.0)[0]
            if marks.size < 2:
                continue
            for run in np.split(marks, np.nonzero(np.diff(marks) > 1)[0] + 1):
                if run.size >= 2:
                    taken += self._rub(points[run], smudger, touch[run].tolist())
        return taken

    def fix(self, amount: float = 1.0) -> float:
        """Spray fixative over the whole sheet: it sets the share ``amount``
        (0 to 1) of the dust that was still loose, and of the paste of an
        oil pastel. Dust that is set stays
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
            # (a film is hard already: fixative sets the dust and the paste)
            hard = self.fixed[..., which] + self.film[..., which]
            newly = np.maximum(volume - hard, 0.0)
            newly *= np.float32(amount)
            # (only what is newly set darkens)
            share = newly / np.maximum(volume, np.float32(1e-9))
            layer[..., IS] *= 1.0 - np.float32(FIXATIVE_DARKENS) * share
            self.fixed[..., which] += newly
            self.paste[..., which] *= np.float32(1.0 - amount)
            total += float(newly.sum(dtype=np.float64))
        return total

    def _rub(self, path, smudger: Smudger, pressure: Profile) -> float:
        """One pass of a tool in the hand along ``path``. Returns the volume
        it took off the sheet, less what it laid back."""
        tool, width = smudger.tool, smudger.width
        xs, ys, bearing, steps = self._sampled(path, pressure, width)
        before = smudger.load
        eng.rub_stroke(
            self.cover,
            self.held,
            self.under,
            self.loose,
            self.bonded,
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
            smudger._carried,
            int(self._rng.integers(1 << 31)),
        )
        return smudger.load - before

    def _sampled(self, path, pressure: Profile, width: float):
        """``path`` sampled for a tip of ``width``: what the kernels take (see
        :func:`fusain._engine.as_kernel_arrays`), or ValueError."""
        line, points, u, knot, spacing, _ = _sample(path, width)
        bearing = _profile("pressure", pressure, line, u, knot)
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
        ends: Ends = "round",
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
        (and fewer), lighter and softer. ``ends`` is as for :meth:`stroke`:
        with ``"flat"`` every stroke stops square at the edge of the area.

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
        # (a round end reaches half the tip past the end of its stroke)
        flat_start, flat_end = _square(ends)
        inset = (0.5 * (1.0 - flat_start), 0.5 * (1.0 - flat_end))
        outline = hatching.as_outline(area)
        broad = _breadth(medium, width, tilt)  # how wide a stroke's mark is
        apart = PEN_SPACING if isinstance(medium, Pen) else hatching.SPACING
        gap = apart * broad if spacing is None else spacing
        _check_hatching(width, gap, angle, hand)
        laid = 0.0
        for path, firm, tip in hatching.strokes(
            outline, angle, width, gap, hand, self._rng, broad, inset
        ):
            along = hatching.bearing(pressure, path, firm, hand)
            # (a stick's tip is as wide as it has worn, whatever the hand)
            laid += self.stroke(path, medium, along, None if own else tip, tilt, ends)
        return laid

    def save(self, path: str, **render_options) -> None:
        """Render and save as PNG."""
        img = self.render(**render_options)
        write_png(path, np.round(img * 255).astype(np.uint8))


Ends = str | tuple[str, str]
#: How a stroke ends: in the outline of its tip, or cut square across.
ENDS = ("round", "flat")


def _square(ends: Ends) -> tuple[float, float]:
    """Whether a stroke is cut square at its start and at its end (1.0) or
    not (0.0), or ValueError."""
    pair = (ends, ends) if isinstance(ends, str) else tuple(ends)
    if len(pair) != 2 or not all(isinstance(e, str) and e in ENDS for e in pair):
        raise ValueError(
            f"ends must be 'round', 'flat' or a pair of those, got {ends!r}"
        )
    start, end = pair
    return float(start == "flat"), float(end == "flat")


def _sample(path, width: Profile):
    """``path`` sampled finely enough for a tip of ``width`` (a number, or a
    profile along it, for which the narrowest counts): the path, the points,
    where they are along it (u, and as knots), how far apart they are, and
    how far apart they would be on a path long enough (the pace of the tip
    where it lands and lifts). ValueError unless the width is a finite
    number of pixels above 0."""
    line = spline.as_path(path)
    if isinstance(width, int | float | np.number):
        narrowest = float(width)
    else:  # (look at it every pixel or so, to find the narrowest)
        _, _, u, knot = line.sample(1.0)
        narrowest = float(np.min(_profile("width", width, line, u, knot)))
    if not (math.isfinite(narrowest) and narrowest > 0):
        raise ValueError(f"width must be a number of pixels above 0, got {width!r}")
    pace = spacing = max(0.25, narrowest / 8.0)
    points, _, u, knot = line.sample(spacing)
    if len(points) > 1:
        spacing = line.length / (len(points) - 1)
    return line, points, u, knot, spacing, pace


def _profile(name: str, profile: Profile, line, u, knot) -> np.ndarray:
    """``profile`` evaluated at the points ``u`` (``knot``) of ``line``, or
    ValueError unless it is finite all along."""
    values = spline.evaluate_profile(profile, u, knot, line.n_knots)
    values = np.broadcast_to(np.asarray(values, dtype=np.float64), u.shape)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite all along the stroke")
    return values


def _check_hatching(width: float, gap: float, angle: float, hand: float) -> None:
    """ValueError unless strokes can be laid side by side with these."""
    for name, value in (("width", width), ("spacing", gap)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a number of pixels above 0: {value!r}")
    if not math.isfinite(angle):
        raise ValueError(f"angle must be a finite number of degrees: {angle!r}")
    if not (math.isfinite(hand) and hand >= 0):
        raise ValueError(f"hand must be 0 (ruler-straight) or more, got {hand!r}")


def _in_hand(tool: Tool | Smudger, width: float | None) -> Smudger:
    """``tool`` as it is held: itself if it is a :class:`fusain.Smudger`
    (which takes no ``width``), or a clean one of this bare tool."""
    if isinstance(tool, Smudger):
        if width is not None:
            raise TypeError("a Smudger has its own width: give none with it")
        return tool
    if not isinstance(tool, Tool):
        raise TypeError(
            f"tool must be a Tool, such as tools.STUMP, or a Smudger: {tool!r}"
        )
    return Smudger(tool, width)


def _breadth(medium: Medium | Stick | Pen, width: float, tilt: float) -> float:
    """How wide a tip ``width`` wide of ``medium`` marks at this tilt."""
    if isinstance(medium, Stick):
        side = medium.side
    elif isinstance(medium, Medium):
        side = medium.side * width
    else:
        return width  # (a pen has no side)
    return width + max(side - width, 0.0) * float(tilt)


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
