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
from ._engine import FILM, HELD, IS, IV, IW, IX, KINDS, LOOSE, PASTE, SET, N
from .medium import Medium
from .paper import Paper
from .pencils import BINDING as PENCIL_BINDING
from .pens import HeldPen, Pen
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
#: Three things shine under a lamp, each in its own way: graphite, polished
#: wax and heavy ballpoint ink.
SHINES = ("graphite", "wax", "ink")
#: Graphite's shine: what a film as shiny as it gets adds to the light a
#: pixel sends back (linear, the paper's white being 1) when it mirrors the
#: lamp squarely.
SHEEN = 0.06
#: A held film of this much sheen per covered area is as shiny as it gets,
#: and matte dust this thick over it hides all but 1/e of its shine.
SHEEN_FULL = 0.6
SHEEN_HIDE = 0.1
#: On a tooth pressed flat the graphite lies flat: a film this much thinner
#: is then as shiny as it gets. A hard lead pressed firmly, which lays
#: little and flattens the tooth, shines by this; a soft one by how much it
#: lays.
SHEEN_FLAT = 0.6
#: How narrow the mirrored light is: graphite is satin, not a mirror.
SHININESS = 12.0
#: A pencil drawn over a bed of dust lays a film on it (:attr:`Sheet.film`)
#: that shines as what the paper holds does. Of the graphite in a layer
#: with a film, the share that is the film's: what the paper holds of a
#: pencil's mark. The rest is its crumbs, loose and matte.
FILM_GRAPHITE = PENCIL_BINDING
#: The gloss of wax: what a film of wax as slick as it gets
#: (:data:`fusain._engine.WAX_FULL`), polished flat, adds to the light when
#: it mirrors the lamp squarely, and how narrow that light is. Less than
#: graphite, and narrower: a film of wax polished flat is smoother than one
#: of graphite flakes. Its light falls to half about 15 degrees from where
#: the lamp is mirrored and to a fifth at 23 degrees, where the default
#: lamp is mirrored, so that under it a burnished dark stays deep, as it
#: does in life away from the glare; with the lamp overhead it shines.
#: (Assumed, not measured: a smooth dielectric whose facets lean about
#: 0.16 on average, a semi-gloss finish.)
#: A film half polished still scatters most of the light: the gloss goes as
#: how far the surface is pressed flat, to the power WAX_POLISH.
WAX_SHEEN = 0.025
WAX_SHININESS = 80.0
WAX_POLISH = 2.0
#: The bronze of ink: what ink as heavy as it gets adds, in its own colour
#: (it mirrors most the light it absorbs most, so blue ink shines copper).
#: Ink as thick as INK_FROM firm passes of the ball begins to bronze, and
#: ink as thick as INK_FULL of them is as bronzed as it gets. A film of dry
#: ink is smooth, and the light it mirrors narrow: the bronze shows with
#: the lamp near overhead and hardly under the default one.
INK_SHEEN = 0.05
INK_FROM = 1.5
INK_FULL = 4.5
INK_SHININESS = 120.0
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
    1 peak), how far it has been pressed flat, in two parts, each 0 to 1:
    how far a firm dry tip has crushed the paper's own tooth (``crush``:
    its peaks come down towards its valleys, and stay there until the
    paper is cut) and how far the wax that lies on it has been polished
    level (``polish``, which goes with the wax); a tip that comes after
    touches all of what is level (:attr:`press`). And how far it has
    been indented (``dent``: the depth of a groove below the peaks of the
    tooth, in depths of the tooth of a paper of depth 1), the pigment fixed
    to the paper (``held``) and the pigment lying on it, in two layers,
    ``loose`` on top and ``under`` beneath it, so that what was laid last is
    seen over what was laid before. Most of that is dust, which is
    what a stump, an eraser or a later stroke can still move; ``bonded``
    holds how much of each layer is not: dust that is set (:attr:`fixed`), a
    film laid over the dust (:attr:`film`) and greasy paste
    (:attr:`paste`). ``cover`` is the share of the pixel that carries
    pigment: the rest shows bare paper. Memory: about 0.38 KB per pixel.

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
        self.crush = np.zeros((H, W), np.float32)
        self.polish = np.zeros((H, W), np.float32)
        self.dent = np.zeros((H, W), np.float32)
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
        # the ink of pens: a stain in the paper, packed as dust is with the
        # colour it bronzes after, and the share of each pixel it is on
        # (None until a pen has drawn)
        self.ink: np.ndarray | None = None
        self.inked: np.ndarray | None = None

    def copy(self) -> Sheet:
        """An independent copy: drawing on one leaves the other alone."""
        new = object.__new__(Sheet)
        new.__dict__.update(self.__dict__)
        new._rng = np.random.default_rng()
        new._rng.bit_generator.state = self._rng.bit_generator.state
        names = ("tooth", "crush", "polish", "dent", "cover", "held", "under")
        for name in (*names, "loose", "_fresh", "bonded"):
            setattr(new, name, getattr(self, name).copy())
        if self.ink is not None and self.inked is not None:
            new.ink, new.inked = self.ink.copy(), self.inked.copy()
        return new

    # ------------------------------------------------------------------
    @property
    def press(self) -> np.ndarray:
        """How far the tooth is level to a tip, 0..1, (H, W): crushed
        (``crush``), polished (``polish``), or both, the wax levelling its
        share of the relief that the crush left
        (``crush + polish - crush * polish``). A tip touches all of a
        tooth pressed flat, a dry medium is bitten less by it, graphite
        lies flat on it, and in the relief its peaks have come down by
        that share of their height. Worked out from the two, which are
        what the sheet holds: it cannot be written to (an array that
        refuses it, with ValueError: write to ``crush`` or ``polish``)."""
        pressed = self._pressed()
        pressed.flags.writeable = False
        return pressed

    def _pressed(self) -> np.ndarray:
        """:attr:`press`, as an array of its own that can be written to."""
        return self.crush + self.polish - self.crush * self.polish

    @property
    def fixed(self) -> np.ndarray:
        """Volume of the dust that is set, (H, W, 2): in the layer underneath
        and in the layer on top. Fixative sets dust, and so does a film of
        wax laid over it. A view: writing to it changes the sheet."""
        return self.bonded[..., SET]

    @property
    def film(self) -> np.ndarray:
        """Volume of film in each of the two layers, (H, W, 2): what a
        medium laid not on the paper but on what lay over it. A waxy medium
        (a coloured pencil, a crayon, an oil pastel) lays one wherever dust
        lies; a dry one (a pencil, a stick of charcoal) on a bed of dust,
        more of it than the crumbs of its own mark. It stays on top of the
        dust. A film of wax sets most of the dust under it, and a tool
        slides on it and lifts it only as it lifts what the paper holds: no
        stump or kneaded eraser moves it. The film of a dry medium is only
        as fast as the dust it lies on: a pencil line on loose pastel comes
        away with the pastel, and one on fixed pastel stays; its graphite
        shines as what the paper holds does (:meth:`shine`). A view, as
        :attr:`fixed` is."""
        return self.bonded[..., FILM]

    @property
    def paste(self) -> np.ndarray:
        """Volume of paste in each of the two layers, (H, W, 2): what the
        paper does not hold of a greasy medium (an oil pastel), and the dust
        it was laid over and took up. A finger smears it, with its wax; a
        kneaded eraser does not lift it. A view, as :attr:`fixed` is."""
        return self.bonded[..., PASTE]

    def pigment(self) -> np.ndarray:
        """Volume of pigment at each pixel, held and loose, (H, W)."""
        return self.held[..., IV] + self.under[..., IV] + self.loose[..., IV]

    def shine(self, of: str | None = None) -> np.ndarray:
        """How much of each pixel mirrors the light, 0..1, (H, W). Three
        things do, and ``of`` names one of them:

        ``"graphite"``: where the paper holds a film of a medium with sheen,
        or a pencil drawn over a bed of dust laid one on the dust
        (:attr:`film`), the more the thicker it is (a thinner film is
        enough on a tooth pressed flat, where it lies flat), and the less
        the more dust of another kind lies over it. Loose graphite dust is
        matte.

        ``"wax"``: where a film of wax has been polished: as far as the wax
        is thick enough to be slick and it has been polished
        (``polish``; half polished, it has a quarter of its gloss), wherever
        the wax lies, on the paper or as a film over dust, and the less the
        more dust lies over it. A coloured pencil burnished, a crayon or an
        oil pastel pressed hard.

        ``"ink"``: where the ink of a ballpoint is heavy, layer over layer,
        and nothing covers it: it bronzes.

        By default, all three together: the share of the pixel that mirrors
        the light in one way or another.

        Raises ValueError for another ``of``.
        """
        if of is None:
            dull = np.ones((self.height, self.width), np.float32)
            for kind in SHINES:
                dull *= 1.0 - self.shine(kind)
            return 1.0 - dull
        if of == "graphite":
            return self._graphite_sheen()
        if of == "wax":
            return self._wax_gloss()
        if of == "ink":
            return self._ink_bronze()
        raise ValueError(f"of must be one of {SHINES} or None, got {of!r}")

    def _graphite_sheen(self) -> np.ndarray:
        """The sheen of graphite; see :meth:`shine`. What the paper holds
        shines through the matte dust of both layers; a film of it on a bed
        of dust as well (:meth:`_film_sheen`), and the shine of all of it
        together is no more than the covered share of the pixel."""
        cover = np.maximum(self.cover, np.float32(1e-6))
        flat = np.float32(SHEEN_FLAT) * self._pressed()
        full = np.float32(SHEEN_FULL) * (1.0 - flat)
        film = np.minimum(self.held[..., IX] / cover / full, 1.0)
        dust = self.under[..., IV] + self.loose[..., IV]
        matte = np.maximum(dust - self.under[..., IX] - self.loose[..., IX], 0.0)
        shine = self.cover * film * np.exp(-matte / cover / np.float32(SHEEN_HIDE))
        if self._has_graphite_films():
            # (where there is no film it adds nothing, and the shine of what
            # the paper holds is as it was, to the bit)
            self._film_sheen(shine)
        return shine

    def _has_graphite_films(self) -> bool:
        """Whether any film lies in the layers where there is graphite."""
        if not self.film.any():
            return False
        return bool(self.under[..., IX].any() or self.loose[..., IX].any())

    def _film_sheen(self, shine: np.ndarray) -> None:
        """Add to ``shine`` the sheen of the films a pencil laid on a bed
        of dust (:func:`fusain._engine.film_sheen`).

        A layer is one mixture: its film is kept as a volume, and what it
        is made of, and what lies over what within the layer, are not kept
        apart from the dust. What can be told is that the graphite in a
        layer with a film came with a pencil's mark: the film, which the
        paper would have held, and its crumbs, which it would not. So the
        film is taken to hold the share of that graphite that the paper
        holds of a pencil's mark (:data:`FILM_GRAPHITE`), and never more
        than its own volume: exact for a pencil over pastel, in one pass or
        in many, and loose graphite dust from elsewhere that lies with a
        film is taken for its crumbs, matte but for the film's share. In
        the layer on top the film shines through all the matte that lies
        with it, the clay of its crumbs and any dust: what a light pass did
        not bury cannot be told from dust laid over the film later, and it
        is taken to hide it. That errs on the matte side: a light line over
        pastel shines less than its film would. A film a later pass buried
        lies on the older dust underneath, and shines through the clay of
        its crumbs, as a line on paper does, and the matte on top. Where a
        pixel has wax, a film there is taken for one of wax (as a tool
        takes it), and its gloss is the wax's."""
        how = np.array([SHEEN_FULL, SHEEN_FLAT, SHEEN_HIDE, FILM_GRAPHITE])
        eng.film_sheen(
            self.held,
            self.under,
            self.loose,
            self.bonded,
            self.cover,
            self._pressed(),
            shine,
            how,
        )

    def _wax_gloss(self) -> np.ndarray:
        """The gloss of polished wax; see :meth:`shine`. Seen from above,
        the topmost wax is a film in the layer on top, or else one in the
        layer underneath, under the dust on top, or else what the paper
        holds, under the dust of both."""
        cover = np.maximum(self.cover, np.float32(1e-6))
        slick = np.minimum(self.held[..., IW] / cover / np.float32(eng.WAX_FULL), 1.0)
        seen = np.ones_like(cover)  # (how much of the wax held shows)
        for which, layer in enumerate((self.under, self.loose)):
            film = self.film[..., which] + self.paste[..., which]
            dust = np.maximum(layer[..., IV] - film, 0.0)
            over = np.minimum(film / cover / np.float32(eng.COVERS), 1.0)
            seen *= np.exp(-dust / cover / np.float32(SHEEN_HIDE))
            seen = over + (1.0 - over) * seen
        return self.cover * slick * self.polish ** np.float32(WAX_POLISH) * seen

    def _ink_bronze(self) -> np.ndarray:
        """The bronze of heavy ink; see :meth:`shine`."""
        if self.ink is None or self.inked is None:
            return np.zeros((self.height, self.width), np.float32)
        inked = np.maximum(self.inked, np.float32(1e-6))
        passes = self.ink[..., IV] / inked / np.float32(eng.BALL_INK)
        heavy = np.clip((passes - INK_FROM) / (INK_FULL - INK_FROM), 0.0, 1.0)
        # (what the paper carries lies over the ink, on its share of the pixel)
        cover = np.maximum(self.cover, np.float32(1e-6))
        hidden = self.cover * (1.0 - np.exp(-self.pigment() / cover / SHEEN_HIDE))
        return (self.inked * heavy * (1.0 - hidden)).astype(np.float32)

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
        count = int(np.count_nonzero(marked))
        if count:
            # the pigment lies on the covered share of the pixel, that much
            # thicker; the rest of the pixel is bare paper (inked or not)
            share = np.clip(self.cover[rows][marked].astype(np.float64), 1e-6, 1.0)
            bare = out[marked] if stained else self.ground
            kappa, sigma = np.empty((count, N)), np.empty(count)
            seen = np.empty((count, N))
            for layer in layers:  # from the paper up
                eng.thicken(layer, marked, share, kappa, sigma)
                sp.km_layer(kappa, sigma, bare, out=seen)
                bare = seen  # (the next layer lies on what is seen so far)
            eng.show_through(out, marked, share, seen)
        return out

    def _stain(self, out: np.ndarray, rows: slice) -> bool:
        """Stain the paper of these rows (``out``, its reflectance) with the
        ink in it. Returns whether there was any."""
        if self.ink is None or self.inked is None:
            return False
        inked = self.inked[rows] > 0
        count = int(np.count_nonzero(inked))
        if not count:
            return False
        share = np.clip(self.inked[rows][inked].astype(np.float64), 1e-6, 1.0)
        kappa, sigma = np.empty((count, N)), np.empty(count)
        eng.thicken(self.ink[rows], inked, share, kappa, sigma)
        seen = sp.km_layer(kappa, sigma, self.ground)
        eng.show_through(out, inked, share, seen)
        return True

    # ------------------------------------------------------------------
    def stroke(
        self,
        path,
        medium: Medium | Stick | Pen | HeldPen,
        pressure: Profile = 0.5,
        width: Profile | None = None,
        tilt: Profile = 0.0,
        ends: Ends = "round",
        *,
        seed: int | None = None,
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
        grainier. The side of a pencil is the cone of lead its sharpening
        bared, and its band is soft at the edges; a stick of charcoal,
        pastel or crayon laid flat (its medium's ``flat_side``) lays a band
        even across, which ends where the stick does. A stick turned over as
        it goes widens its mark from its point to its side. Where the stroke
        turns, each part of the tip lays as far as it slides.

        ``ends`` is how the stroke ends: ``"round"`` (the default), in the
        outline of the tip, which reaches half its length past the ends of
        the path; or ``"flat"``, cut square across at the ends of the path,
        as the end or the side of a stick drawn and lifted cleanly leaves
        it: a flat stroke is as long as its path (the cut takes what lies
        beyond the stroke, so where the path turns just before it ends, the
        stroke before the turn stays whole). A pair, such as
        ``("flat", "round")``, says it for the start and for the end. Either
        way the tip lays as much where it lands and where it lifts as
        anywhere along the stroke, so a stroke is as strong at its ends as
        in its middle, and a short one as strong as a long one.

        ``medium`` can also be a :class:`fusain.Pen`, a ballpoint: it has
        its own width and no side (``tilt`` stays 0), and its ends are
        round; its line is nearly as dark at a light touch as at a firm one,
        and its ink is a stain in the paper (:attr:`ink`), under everything
        else, which no rubber eraser takes off (a knife or a sand eraser
        cuts it out with the paper). Its ball gathers ink as it rolls and
        leaves a blob where the line turns sharply and, now and then, where
        it starts, the lighter and the narrower the lighter the line (as
        much as the pen's ``blots`` says); and it presses a groove into the
        paper (``dent``), as :meth:`indent` does, the deeper the firmer the
        pen. A bare pen is picked up afresh for each line; a
        :class:`fusain.HeldPen`, a pen in the hand, lands with the ink its
        ball gathered on the line before (its ball holds a drop at most, and
        gathers no more).

        A dry tip pressed firmly (harder than about 0.6, setting in by
        degrees from 0.4) flattens the tooth (``crush``), the more the
        harder the tip: a hard pencil does, a soft one hardly, charcoal and
        pastel next to not at all. What is drawn there afterwards lies
        evenly, and the paper bites it less. Any tip pressed firmly on a
        film of wax polishes it (``polish``). Each goes as far as the
        pressure can, and no further for being gone over again: a firmer
        stroke over a lighter one presses the tooth flatter, a lighter one
        over a firmer one leaves it as it is, and light layers, however
        many, do not burnish.
        Where the paper is indented, a tip lays only what it reaches down
        to (:meth:`indent`): it rests on the highest of the paper it covers,
        so one narrower than a groove goes down into it.

        A medium lies on what it meets. What the paper holds of it is in
        the tooth, under such crumbs as its own; but what a waxy medium (a
        coloured pencil, a crayon, an oil pastel) lays where dust lies is a
        film on the dust (:attr:`film`), which sets part of it
        (:attr:`fixed`), and so is what a dry one (a pencil, a stick of
        charcoal) lays on a bed of dust, more of it than its own mark
        leaves: a pencil line shows over pastel, loose or fixed, a little
        weaker than on bare paper. Such a bed is not in a dry tip's way,
        nor is what lies under it; but pastel rubbed so far into the paper
        that no dust is left on it fills the tooth, and takes little.

        ``seed`` (an integer from 0 to 2**31 - 1) gives the stroke the grain
        of its own: it takes every random number it needs from it and none
        from the sheet's generator, so that its grain does not depend on what
        was drawn before it, and what is drawn after it does not depend on
        it. By default the grain is drawn from the sheet's generator.

        Returns the volume of medium laid, which is what the tip lost.

        Raises ValueError or TypeError, before anything is drawn, unless
        ``width`` is finite and above 0 all along (and not given for a stick
        or a pen), the pressures are finite, ``tilt`` is between 0 and 1 all
        along, ``ends`` is ``"round"``, ``"flat"`` or a pair of those and
        ``seed`` is none or an integer from 0 to 2**31 - 1.
        """
        _check_seed(seed)
        if isinstance(medium, Pen | HeldPen):
            return self._roll(path, medium, pressure, width, tilt, ends, seed)
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
                "medium must be a Medium, such as pencils.HB, a Stick, a Pen or "
                f"a HeldPen: {medium!r}"
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
        tilts = _along(tilt, line, u, knot)
        if not ((tilts >= 0.0) & (tilts <= 1.0)).all():
            raise ValueError(f"tilt must be between 0 (point) and 1 (side): {tilt!r}")
        shapes = np.empty((len(u), 3))
        shapes[:, 0], shapes[:, 1] = widths, tilts
        shapes[:, 2] = known.side * widths if stick is None else stick.side
        if len(points) == 1:
            square = (0.0, 0.0)  # (a dot is the outline of the tip)
        paper, pigment = self.paper, known.pigment
        laid = float(
            eng.lay_stroke(
                self.tooth,
                self.crush,
                self.polish,
                self.dent,
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
                        known.melts,
                        float(known.flat_side),
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [paper.depth, paper.bite, paper.give, paper.hold], dtype=np.float64
                ),
                self._seed(seed),
            )
        )
        if stick is not None:
            stick.worn = max(stick.worn + laid, 0.0)
        return laid

    def _roll(
        self, path, pen: Pen | HeldPen, pressure: Profile, width, tilt, ends, seed
    ) -> float:
        """One pass of a ballpoint pen along ``path``; see :meth:`stroke`."""
        if width is not None:
            kind = type(pen).__name__
            raise TypeError(f"a {kind} has its own width: give none with it")
        if not (isinstance(tilt, int | float) and tilt == 0.0):
            raise ValueError(
                f"a pen has no side to draw with: tilt must be 0: {tilt!r}"
            )
        if _square(ends) != (0.0, 0.0):
            raise ValueError(f"a ballpoint's line ends round: {ends!r}")
        line, points, u, knot, spacing, pace = _sample(path, pen.width)
        bearing = _profile("pressure", pressure, line, u, knot)
        xs, ys, bearing, steps = eng.as_kernel_arrays(
            points, np.clip(bearing, 0.0, 1.0), spacing
        )
        # (where it lands and lifts, the ball goes at its own pace, however
        # short the line: a line of no length is a dot)
        landing = spacing if spacing >= 0.5 * pace else pace
        if self.ink is None or self.inked is None:
            self.ink = np.zeros((self.height, self.width, eng.INK), np.float32)
            self.inked = np.zeros((self.height, self.width), np.float32)
        # (a bare pen is picked up afresh: the kernel draws what it lands with)
        rim = pen._rim if isinstance(pen, HeldPen) else np.full(1, np.nan)
        pen = pen.pen if isinstance(pen, HeldPen) else pen
        paper = self.paper
        return float(
            eng.roll_stroke(
                self.tooth,
                self.crush,
                self.polish,
                self.dent,
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
                landing,
                0.5 * float(pen.width),
                np.ascontiguousarray(pen.ink.K, dtype=np.float64),
                np.array(
                    [pen.ink.S, pen.flow, pen.blots, *pen.mirrors], dtype=np.float64
                ),
                np.array(
                    [paper.depth, paper.bite, paper.give, paper.hold], dtype=np.float64
                ),
                rim,
                self._seed(seed),
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

        The lamp also shows what shines (:meth:`shine`). Seen from straight
        above, a heavy layer of graphite mirrors a high lamp and looks
        silvery, and hardly mirrors a raking one; polished wax has a softer,
        narrower gloss, which shows with the lamp near overhead and hardly
        under the default lamp, so that a burnished dark stays deep; and
        heavy ballpoint ink bronzes, in the colour it absorbs most.
        ``gloss`` scales all of it: 0 renders the drawing matte, as a
        scanner sees it.

        When lit, raises ValueError unless ``light`` is three finite numbers
        with z > 0 and ``relief`` and ``gloss`` are finite and not negative.
        """
        shade = glint = bronze = None
        if lighting:
            lamp = _check_light(DEFAULT_LIGHT if light is None else light)
            for name, value in (("relief", relief), ("gloss", gloss)):
                if not (math.isfinite(value) and value >= 0):
                    raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
            slopes = self._slopes(float(relief))
            shade = self._shade(lamp, slopes)
            if gloss > 0:
                glint, bronze = self._glint(lamp, slopes, float(gloss))
        out = np.empty((self.height, self.width, 3), np.float32)
        for rows in self._row_chunks():
            lin = sp.reflectance_to_linear_rgb(self._reflectance_rows(rows))
            if shade is not None:
                lin *= shade[rows][..., None]
            if glint is not None:
                lin += glint[rows][..., None]
            if bronze is not None:
                self._add_bronze(lin, bronze[rows], rows)
            out[rows] = sp.linear_to_srgb(np.clip(lin, 0.0, 1.0))
        return out

    def surface_height(self, relief: float = 1.0) -> np.ndarray:
        """Height of the paper's surface in pixels, (H, W), as it is lit: its
        tooth, brought down where it has been crushed or its wax polished (by
        :attr:`press` of its height: pressed flat, it is level with the valleys,
        and a valley the tip never reached is not left a pit), and no
        higher than the floor of a groove where it has been indented. The
        peaks of an untouched tooth are at ``TOOTH_HEIGHT`` times the
        paper's grain and depth (times ``relief``), and its valleys at 0.

        (To a tip that comes after, what matters about a flattened tooth is
        that it is level: it touches all of it, as it would the peaks.)"""
        grain, depth = self.paper.grain, self.paper.depth
        if depth <= 0.0:  # (a level paper: nothing but its grooves)
            return -self.dent * np.float32(TOOTH_HEIGHT * grain * relief)
        # (the peaks come down towards the valleys)
        ground = self.tooth * (1.0 - self._pressed())
        ground = np.minimum(ground, 1.0 - self.dent / np.float32(depth))
        return ground * np.float32(TOOTH_HEIGHT * grain * depth * relief)

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

    def _glint(self, lamp: np.ndarray, slopes, gloss: float):
        """The light of the lamp each pixel mirrors to an eye straight above
        the sheet: what shines there, times how squarely its surface faces
        halfway between the lamp and the eye. Two parts, each (H, W) or
        None if there is none of it: the white light, off graphite and off
        polished wax, and the bronze of ink, which has the ink's colour
        (:meth:`_add_bronze`)."""
        halfway = lamp + np.array([0.0, 0.0, 1.0])
        halfway /= np.linalg.norm(halfway)
        facing = self._facing(halfway, slopes)
        white = None
        lobes = (
            ("graphite", IX, SHEEN, SHININESS),
            ("wax", IW, WAX_SHEEN, WAX_SHININESS),
        )
        for kind, part, sheen, shininess in lobes:
            if not self.held[..., part].any():
                # (graphite also shines as a film on a bed of dust)
                filmed = kind == "graphite" and self._has_graphite_films()
                if not filmed:
                    continue
            mirrored = self.shine(kind) * facing**shininess * (sheen * gloss)
            white = mirrored if white is None else white + mirrored
        if white is not None:
            white = white.astype(np.float32)
        bronze = None
        if self.ink is not None:
            heavy = self._ink_bronze()
            if heavy.any():
                bronze = heavy * facing**INK_SHININESS * (INK_SHEEN * gloss)
                bronze = bronze.astype(np.float32)
        return white, bronze

    def _add_bronze(self, lin: np.ndarray, bronze: np.ndarray, rows: slice) -> None:
        """Add the bronze of the ink of these rows to their light (``lin``,
        linear RGB), in the colour each ink mirrors (:attr:`fusain.Pen.mirrors`:
        blue ink, which absorbs the reds, shines copper), mixed as the inks
        are."""
        shining = bronze > 0
        if self.ink is None or not shining.any():
            return
        ink = self.ink[rows][shining].astype(np.float64)
        colour = ink[:, eng.IB : eng.IB + 3]
        colour /= np.maximum(ink[:, IV], 1e-12)[:, None]
        lin[shining] += bronze[shining][:, None] * colour

    def indent(self, path, pressure: Profile = 0.7, width: float = 3.0) -> None:
        """Press a groove into the paper along ``path`` with a stylus: a
        hard round point ``width`` pixels wide that lays nothing, such as
        an embossing tool, a knitting needle or a pen that has run dry.

        ``path`` and ``pressure`` are as for :meth:`stroke`. The groove is
        as deep as the stylus is pressed hard and as the paper gives
        (``dent``), and round across: deepest under the middle of the
        point. A pencil, a crayon or a stick of charcoal shaded over it
        afterwards rides on the paper either side and skips it, unless it
        is soft enough and pressed hard enough to reach its floor: a line
        indented firmly stays the colour of the paper under a tone, which
        is how whiskers and fine highlights are kept; one indented lightly
        fills in under a firm soft pencil. A point narrower than the groove
        follows it down and marks its floor, which the stylus left flat,
        and which bites it less, as a flattened tooth does. What was drawn
        there before goes down with the paper, and no eraser takes a groove
        out. Under a raking light it shows as a line of shadow.

        Raises ValueError, before anything is pressed, unless ``width`` is
        a finite number of pixels above 0 and the pressures are finite.
        """
        xs, ys, bearing, steps = self._sampled(path, pressure, width)
        paper = self.paper
        eng.indent_stroke(
            self.dent,
            xs,
            ys,
            bearing,
            steps,
            0.5 * float(width),
            np.array(
                [paper.depth, paper.bite, paper.give, paper.hold], dtype=np.float64
            ),
        )

    def smudge(
        self,
        path,
        tool: Tool | Smudger = STUMP,
        pressure: Profile = 0.5,
        width: float | None = None,
        *,
        seed: int | None = None,
    ) -> float:
        """Rub one pass of a stump, a finger or another :class:`fusain.Tool`
        along ``path``: it drags the dust along, presses some of it into the
        paper and pushes it into the valleys of the tooth, which softens and
        greys what was drawn. What the paper holds hardly moves, but for
        graphite, which slides. Most of the dust goes a fraction of the
        tool's width, so two colours rubbed across where they meet blend
        into a passage a width or two wide; the small share that the tool
        lifts into its load goes far.

        ``path`` and ``pressure`` are as for :meth:`stroke`; ``width`` is the
        width of the tool in pixels (by default the tool's own). The pass is
        as strong where the tool lands and where it lifts as in its middle,
        out to the outline of the tool, and a single point is a dab: one
        pass of the tool on the paper under it.

        A bare tool is clean when it lands, so it leaves the paper lighter
        where it does, and it takes away what it still carries when it lifts.
        A :class:`fusain.Smudger` is a tool in the hand: it lands with what
        it picked up before and keeps what it carries. It fills slowly and
        takes nothing more once it is full, so it leaves a tone as dark as
        it was; and bare paper takes the dust off it readily, so a loaded
        stump draws a tone of its own there (and leaves some wherever it
        strays off a tone). It has its own width, so it takes none.

        ``seed`` (an integer from 0 to 2**31 - 1) gives the pass the grain of
        its own, as for :meth:`stroke`: it draws nothing from the sheet's
        generator, and its grain does not depend on what was drawn before.

        Returns the volume of pigment the tool took off the sheet, less what
        it laid back: for a bare tool, what was left on it at the end; for a
        ``Smudger`` that left more than it took, less than nothing.
        """
        _check_seed(seed)
        return self._rub(path, _in_hand(tool, width), pressure, seed)

    def erase(
        self,
        path,
        tool: Tool | Smudger = KNEADED,
        pressure: Profile = 0.5,
        width: float | None = None,
        *,
        seed: int | None = None,
    ) -> float:
        """Rub one pass of an eraser along ``path``. A kneaded eraser
        (``tools.KNEADED``, the default) lifts the dust and a little of what
        the paper holds; a vinyl one (``tools.ERASER``) takes nearly all.
        The paste of an oil pastel, a film of wax laid over dust and the
        dust set under it come off as what the paper holds does; a pencil
        line drawn over loose pastel comes off with the pastel.

        Wax holds what it binds, and an eraser slides on it: a light layer
        of coloured pencil comes off a vinyl eraser fairly well, a dense
        one by degrees and a burnished one hardly; a kneaded eraser barely
        touches them. What holds against it, a vinyl eraser drags along
        and smears onto what is beside it and after it. And wax stains the
        paper: a tint always stays where they were, which only what cuts
        the paper takes. For a light in
        those, scratch it out with the point of a knife (``tools.KNIFE``, a
        fine line back to the paper), rub it out with a sand eraser
        (``tools.SAND_ERASER``), or keep it from the start by leaving the
        paper or by indenting it (:meth:`indent`).
        Burnished wax levels the tooth: where the wax is taken off, its
        polish goes with it and the tooth is there again for what is drawn
        next, but for what a hard tip crushed of the paper itself, under
        the wax or through it (``crush``), which only a sand eraser or a
        knife cuts away, with the surface of the paper. Those two take the
        ink of a ballpoint with it, which is a stain in the paper; no rubber
        eraser touches it.

        ``path`` and ``pressure`` are as for :meth:`stroke`; ``width`` is the
        width of the eraser in pixels (by default the tool's own; a
        :class:`fusain.Smudger` has its own and takes none). The eraser
        takes as much where it lands and where it lifts as in the middle of
        the pass, and a single point is a dab, which takes one pass's worth
        off the paper under it.

        ``seed`` (an integer from 0 to 2**31 - 1) gives the pass the grain of
        its own, as for :meth:`stroke`: it draws nothing from the sheet's
        generator, and its grain does not depend on what was drawn before.

        Returns the volume of pigment taken off the sheet.
        """
        _check_seed(seed)
        return self._rub(path, _in_hand(tool, width), pressure, seed)

    def rub(
        self,
        area,
        tool: Tool | Smudger = STUMP,
        pressure: hatching.Pressure = 0.5,
        width: float | None = None,
        angle: float = 0.0,
        spacing: float | None = None,
        hand: float = 1.0,
        *,
        seed: int | None = None,
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
        hand is. The area is rubbed up to its edge and no further, however
        unsteady the hand: it works the tool up to the edge and turns there,
        and does not sweep past it as a pencil's strokes overshoot (see
        :func:`fusain.hatching.passes`). The middle of the tool keeps half
        its width inside the edge, the ends of the passes stray only short
        of it, and the outermost passes run along it; the tool does not
        lighten where it turns, so the ends of the passes are rubbed as
        strongly as their middles. Along the sides of the outermost passes
        only the rounded rim of the tool reaches the edge, and the pixels
        there within about a third of the tool's width of it are rubbed
        less, the less the nearer: a second rub across the first works them
        up to the edge too. To rub past the edge, give a larger area.

        The tool keeps what it carries from one pass to the next, and over
        the stretches where the pressure is nothing and it is lifted: it
        lands with it, takes less the fuller it is and nothing once it is
        full, and the tone is blended, not worn away. A bare
        tool starts clean and takes away what it carries at the end; a
        :class:`fusain.Smudger` (which has its own width and takes none)
        starts with what it picked up before, and keeps it.

        However the pressure falls, a rub takes the same numbers from the
        sheet's generator (the passes' places and one seed for the grain of
        them all), so what is drawn after it has the grain it would have had.
        Given a ``seed`` (an integer from 0 to 2**31 - 1) it takes none from
        the sheet's generator: it has a generator of its own, from which it
        draws all it would have drawn from the sheet's, so that its grain
        does not depend on what was drawn before and what is drawn after does
        not depend on it.

        Returns the volume of pigment the tool took off the sheet in all,
        less what it laid back (less than nothing for a ``Smudger`` that
        left more than it took).

        Raises ValueError or TypeError, before anything is rubbed, as
        :meth:`shade` does (and for a ``seed`` that is not none or an integer
        from 0 to 2**31 - 1).
        """
        _check_seed(seed)
        rng = self._rng if seed is None else np.random.default_rng(seed)
        smudger = _in_hand(tool, width)
        width = smudger.width
        outline = hatching.as_outline(area)
        gap = 0.5 * width if spacing is None else spacing
        _check_hatching(width, gap, angle, hand)
        passes = []  # (what each pass presses, worked out before any rubs)
        # (a pass is as strong where the tool lands and lifts as in its
        # middle, out to the outline of the tool: the hand keeps the middle
        # of the tool half a width inside the edge of the area, and turns
        # there, so the tool comes to the edge and does not stray past it)
        strokes = hatching.passes(outline, angle, width, gap, hand, rng)
        for k, (path, firm) in enumerate(strokes):
            path = path[::-1] if k % 2 else path  # to and fro
            # (it turns at the ends, bearing as it does: it is not put down
            # and lifted there as a pencil's stroke is, lighter at either end)
            along = hatching.bearing(pressure, path, firm, 0.0)
            points, _, u, _ = spline.as_path(path).sample(max(1.0, width / 8.0))
            touch = np.broadcast_to(np.asarray(along(u), dtype=np.float64), u.shape)
            if not np.isfinite(touch).all():
                raise ValueError("pressure must be finite all over the area")
            passes.append((points, touch))
        # (one draw for the whole rub, whatever the pressure does: a run of a
        # pass has the seed of the sample it starts at, counted from the start
        # of the rub, so no two runs share one and none moves when the
        # pressure is nothing elsewhere)
        base = int(rng.integers(1 << 31))
        taken, first = 0.0, 0
        for points, touch in passes:
            marks = np.nonzero(touch > 0.0)[0]
            if marks.size >= 2:
                for run in np.split(marks, np.nonzero(np.diff(marks) > 1)[0] + 1):
                    if run.size >= 2:
                        seed = _seed_of(base, first + int(run[0]))
                        taken += self._rub(
                            points[run], smudger, touch[run].tolist(), seed
                        )
            first += len(points)
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

    def _rub(
        self, path, smudger: Smudger, pressure: Profile, seed: int | None = None
    ) -> float:
        """One pass of a tool in the hand along ``path``, with the grain of
        ``seed`` (by default one drawn from the sheet's generator). Returns
        the volume it took off the sheet, less what it laid back."""
        tool, width = smudger.tool, smudger.width
        line, points, u, knot, spacing, pace = _sample(path, width)
        bearing = _profile("pressure", pressure, line, u, knot)
        xs, ys, bearing, steps = eng.as_kernel_arrays(
            points, np.clip(bearing, 0.0, 1.0), spacing
        )
        before = smudger.load
        ink, inked = self.ink, self.inked
        if ink is None or inked is None:  # (no pen has drawn here)
            ink = np.zeros((0, 0, eng.INK), np.float32)
            inked = np.zeros((0, 0), np.float32)
        eng.rub_stroke(
            self.cover,
            self.crush,
            self.polish,
            self.held,
            self.under,
            self.loose,
            self.bonded,
            ink,
            inked,
            xs,
            ys,
            bearing,
            steps,
            np.array(
                [
                    0.5 * float(width),
                    tool.lift_loose,
                    tool.lift_held,
                    tool.lay,
                    tool.press,
                    tool.smear,
                    tool.scrape,
                    tool.holds,
                ],
                dtype=np.float64,
            ),
            smudger._carried,
            # (where it lands and lifts, the tool goes at its own pace,
            # however short the pass)
            spacing if spacing >= 0.5 * pace else pace,
            self._seed(seed),
        )
        return smudger.load - before

    def _seed(self, seed: int | None) -> int:
        """``seed``, or by default one drawn from the sheet's generator: what
        a kernel takes to salt the grain of a mark."""
        return int(self._rng.integers(1 << 31)) if seed is None else int(seed)

    def _sampled(self, path, pressure: Profile, width: float):
        """``path`` sampled for a tip of ``width``: what the kernels take (see
        :func:`fusain._engine.as_kernel_arrays`), or ValueError."""
        line, points, u, knot, spacing, _ = _sample(path, width)
        bearing = _profile("pressure", pressure, line, u, knot)
        return eng.as_kernel_arrays(points, np.clip(bearing, 0.0, 1.0), spacing)

    def shade(
        self,
        area,
        medium: Medium | Stick | Pen | HeldPen,
        pressure: hatching.Pressure = 0.5,
        width: float | None = None,
        angle: float = 30.0,
        spacing: float | None = None,
        hand: float = 1.0,
        tilt: float = 0.0,
        ends: Ends = "round",
        *,
        seed: int | None = None,
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
        are (by default a little over half a width, so that they overlap; the
        even bands of a stick laid flat side by side, overlapping only by
        their rounded ends).
        ``hand`` is how
        unsteady the hand is: 1 for a natural one, 0 for ruler-straight
        strokes, evenly spaced and all alike. ``tilt`` is as for
        :meth:`stroke`: with the side of the stick the strokes are broader
        (and fewer), lighter and softer. ``ends`` is as for :meth:`stroke`:
        with ``"flat"`` every stroke stops square at the edge of the area.

        Shade again at another angle to cross-hatch or to build tone.
        Given a ``seed`` (an integer from 0 to 2**31 - 1) the shading takes
        nothing from the sheet's generator: it has a generator of its own,
        from which it draws the unsteadiness of the hand and the grain of
        every stroke, so that its grain does not depend on what was drawn
        before and what is drawn after does not depend on it.
        Returns the volume of medium laid.

        Raises ValueError or TypeError, before anything is drawn, unless
        ``width`` (not given for a stick) and ``spacing`` are finite numbers
        above 0, ``angle`` is finite, ``hand`` is finite and not negative,
        ``tilt`` is between 0 and 1 and ``seed`` is none or an integer from 0
        to 2**31 - 1.
        """
        _check_seed(seed)
        own = isinstance(medium, Stick | Pen | HeldPen)  # (it has its own width)
        if own:
            if width is not None:
                kind = type(medium).__name__
                raise TypeError(f"a {kind} has its own width: give none with it")
            width = medium.width
        elif isinstance(medium, Medium):
            width = 6.0 if width is None else width
        else:
            raise TypeError(
                "medium must be a Medium, such as pencils.HB, a Stick, a Pen or "
                f"a HeldPen: {medium!r}"
            )
        if not 0.0 <= tilt <= 1.0:
            raise ValueError(f"tilt must be between 0 (point) and 1 (side): {tilt!r}")
        # (a round end reaches half the tip past the end of its stroke)
        flat_start, flat_end = _square(ends)
        inset = (0.5 * (1.0 - flat_start), 0.5 * (1.0 - flat_end))
        outline = hatching.as_outline(area)
        broad = _breadth(medium, width, tilt)  # how wide a stroke's mark is
        gap = _apart(medium, width, broad) if spacing is None else spacing
        _check_hatching(width, gap, angle, hand)
        laid = 0.0
        rng = self._rng if seed is None else np.random.default_rng(seed)
        for path, firm, tip in hatching.strokes(
            outline, angle, width, gap, hand, rng, broad, inset
        ):
            along = hatching.bearing(pressure, path, firm, hand)
            # (a stick's tip is as wide as it has worn, whatever the hand; the
            # grain of a stroke is drawn from the sheet's generator by default
            # and from the shade's own if it has one)
            mark = {} if seed is None else {"seed": int(rng.integers(1 << 31))}
            laid += self.stroke(
                path, medium, along, None if own else tip, tilt, ends, **mark
            )
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
    values = _along(profile, line, u, knot)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite all along the stroke")
    return values


def _along(profile: Profile, line, u, knot) -> np.ndarray:
    """``profile`` evaluated at the points ``u`` (``knot``) of ``line``: one
    float for each. A number is the same all along, as
    :func:`murex.spline.evaluate_profile` makes it, without the long way
    round (most strokes are given numbers, and a short one takes as long to
    evaluate as to draw)."""
    if isinstance(profile, float | int | np.floating | np.integer):
        return np.full(u.shape, float(profile))
    values = spline.evaluate_profile(profile, u, knot, line.n_knots)
    return np.broadcast_to(np.asarray(values, dtype=np.float64), u.shape)


def _check_seed(seed: int | None) -> None:
    """ValueError unless ``seed`` is none or an integer from 0 to 2**31 - 1
    (as a kernel takes it)."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise ValueError(f"seed must be an integer or none, got {seed!r}")
    if not 0 <= seed < 1 << 31:
        raise ValueError(f"seed must be from 0 to 2**31 - 1, got {seed!r}")


def _check_hatching(width: float, gap: float, angle: float, hand: float) -> None:
    """ValueError unless strokes can be laid side by side with these."""
    for name, value in (("width", width), ("spacing", gap)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a number of pixels above 0: {value!r}")
    if not math.isfinite(angle):
        raise ValueError(f"angle must be a finite number of degrees: {angle!r}")
    if not (math.isfinite(hand) and hand >= 0):
        raise ValueError(f"hand must be 0 (ruler-straight) or more, got {hand!r}")


#: What a seed steps by from one run of a rub to the next: odd, so that no two
#: of fewer than 2**31 steps give the same seed, and far from 1, so that two
#: seeds are not neighbours (a kernel's noise is salted with a seed and the
#: seeds next to it).
SEED_STEP = 0x1E3779B1


def _seed_of(base: int, step: int) -> int:
    """The seed ``step`` steps on from ``base``, as the kernels take it (below
    2**31): the same on every platform, and different for each ``step``
    (below 2**31)."""
    return (base + step * SEED_STEP) % (1 << 31)


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


def _breadth(
    medium: Medium | Stick | Pen | HeldPen, width: float, tilt: float
) -> float:
    """How wide a tip ``width`` wide of ``medium`` marks at this tilt."""
    if isinstance(medium, Stick):
        side = medium.side
    elif isinstance(medium, Medium):
        side = medium.side * width
    else:
        return width  # (a pen has no side)
    return width + max(side - width, 0.0) * float(tilt)


def _apart(medium: Medium | Stick | Pen | HeldPen, width: float, broad: float) -> float:
    """How far apart a hand lays the strokes of a shading whose tip is
    ``width`` wide and marks ``broad`` across: a little over half the mark,
    so that the strokes overlap and their soft edges make one tone; the
    lines of a pen side by side. The band of a stick laid flat is even
    across and ends within the stick's thickness: the bands are laid side
    by side, and only their ends overlap, as the strokes of its point do."""
    if isinstance(medium, Pen | HeldPen):
        return PEN_SPACING * broad
    known = medium.medium if isinstance(medium, Stick) else medium
    if known.flat_side:
        return broad - width + hatching.SPACING * width
    return hatching.SPACING * broad


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
