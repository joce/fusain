"""Tools that lay nothing of their own: stumps and fingers that move what
is on the paper, erasers that take it off, and a knife that scrapes it off.
Used with :meth:`fusain.Sheet.smudge`, :meth:`fusain.Sheet.erase` and
:meth:`fusain.Sheet.rub`: bare, a tool is clean each time it is put to the
paper; a :class:`fusain.Smudger` is one in the hand, which keeps what it
picks up.

A stump or a finger lays what it carries on paper, and hardly on wax, which
is as slick to the dust on a tool as to a dry stick: charcoal smeared across
a crayon line leaves it clean. (The paste of an oil pastel sticks.)

An eraser takes a dry medium off as far as it lifts. Wax holds: a light
layer of coloured pencil comes off a vinyl eraser fairly well, a dense one
badly, a burnished one hardly, and a kneaded eraser barely touches any of
them, and wax leaves a stain in the paper that no rubber eraser takes. A
light is scratched out of those with ``KNIFE`` or rubbed out with
``SAND_ERASER``, or kept from the start (:meth:`fusain.Sheet.indent`).
Those two cut the surface of the paper, and take the ink of a ballpoint
stained into it, which no rubber eraser touches."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Tool:
    """What a tool does where it bears fully on the paper, in one pass.

    Parameters
    ----------
    lift_loose, lift_held:
        How strongly it lifts the dust, and the pigment the paper holds,
        into its load (0 = not at all; 1 takes about two thirds in a pass;
        more, nearly all). Dust that is set, by fixative or under a film of
        wax, and the film itself come off as what the paper holds does. An
        eraser lifts strongly. A stump or a finger lifts little: most of
        what they move is the dust on their face (``smear``), which goes a
        fraction of their width; what is lifted is carried far, and fills
        the tool slowly (``holds``).
    lay:
        The share of its load that it lays back down over each of its own
        widths on bare paper (0 for an eraser, which keeps it). Bare tooth
        takes the dust off a loaded tool readily; paper that carries pigment
        takes less, and none once the pigment covers its tooth. So a loaded
        stump tones bare paper, and leaves a tone as it is. (The paste of an
        oil pastel sticks to any paper.)
    press:
        How strongly it presses the dust into the paper, where it is held.
    smear:
        How strongly it pushes the dust on the high ground into the low,
        which greys the white specks of the paper, and spreads it from grain
        to grain: it takes the dust onto its face and lays it back within a
        fraction of its width, which evens a tone and blends two colours
        where they meet. A tool that smears (1 or more) also moves the
        paste of an oil pastel as it moves dust; one that does not, an
        eraser, lifts paste only as it lifts what the paper holds.
    width:
        Its usual width in pixels.
    name:
        Optional label.
    scrape:
        0..1: how far it cuts through wax. Wax holds what it binds, and an
        eraser slides on it: of a coloured pencil, a crayon or an oil
        pastel it lifts the less the thicker the wax is, and hardly
        anything once the wax is polished. A tool that scrapes (grit, a
        blade) takes that share of what the wax would have kept from it,
        and cuts the surface of the paper by that share of what it lifts of
        what the paper holds: with it go the stain a waxy medium leaves in
        the paper, the ink of a ballpoint, and the flattening of the tooth,
        which is the paper's own again where it has been cut.
    holds:
        How much pigment it holds when it is full, as a volume for each
        pixel of its face (a firm layer of charcoal is about half a unit
        thick). It lifts the less the fuller it is, and nothing once full,
        so a stump or a finger fills slowly as it is rubbed in a tone and
        then leaves the tone alone. By default a tool holds whatever it
        takes, as an eraser does.
    """

    lift_loose: float
    lift_held: float = 0.0
    lay: float = 0.0
    press: float = 0.0
    smear: float = 0.0
    width: float = 12.0
    name: str = ""
    scrape: float = 0.0
    holds: float = math.inf

    def __post_init__(self):
        for field in ("lift_loose", "lift_held", "lay", "press", "smear"):
            value = getattr(self, field)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{field} must be 0 or more, got {value!r}")
        if self.lay > 1:
            raise ValueError(f"lay is a share, at most 1, got {self.lay!r}")
        if not (math.isfinite(self.scrape) and 0 <= self.scrape <= 1):
            raise ValueError(f"scrape must be between 0 and 1, got {self.scrape!r}")
        if not self.holds > 0:
            raise ValueError(f"holds must be above 0, got {self.holds!r}")
        if not (math.isfinite(self.width) and self.width > 0):
            raise ValueError(
                f"width must be a number of pixels above 0: {self.width!r}"
            )

    def with_(self, **changes) -> Tool:
        """A copy with some fields changed."""
        return replace(self, **changes)


#: A paper stump: firm and pointed. It smears the dust along and rubs it in.
#: A little of the dust stays in it at each pass: rubbed for long in a tone
#: it is full, and tones bare paper for thirty or so of its widths.
STUMP = Tool(
    0.04, 0.02, lay=0.3, press=0.5, smear=2.0, width=10.0, name="stump", holds=1.5
)
#: A tortillon: a small, hard stump for details.
TORTILLON = STUMP.with_(width=5.0, name="tortillon")
#: A finger: broad and soft. It spreads the dust and presses it in; it
#: fills more slowly than a stump, and holds less for its size.
FINGER = Tool(
    0.03, 0.01, lay=0.25, press=0.8, smear=2.5, width=22.0, name="finger", holds=1.0
)
#: A chamois: it wipes most of the dust away and spreads the rest thin.
CHAMOIS = Tool(1.6, 0.04, lay=0.12, press=0.1, smear=1.0, width=30.0, name="chamois")
#: A kneaded eraser: it lifts the dust, and a little of what is held; so
#: only a little of an oil pastel, which is paste, and next to nothing of
#: wax.
KNEADED = Tool(2.5, 0.3, width=12.0, name="kneaded eraser")
#: A vinyl eraser: it rubs off nearly everything but wax, on which it
#: slides: it lightens a dense layer of coloured pencil only by degrees, and
#: smears a dense or burnished one along its way.
ERASER = Tool(4.0, 2.5, width=10.0, name="vinyl eraser")
#: A sand eraser (an ink eraser, or the tip of an electric one): grit in
#: rubber, which abrades the wax a vinyl eraser slides on.
SAND_ERASER = Tool(4.0, 3.0, width=6.0, name="sand eraser", scrape=0.8)
#: The point of a knife, to scratch a fine light line out of a dense layer
#: (sgraffito): it scrapes off nearly all that lies in its way, wax or not.
KNIFE = Tool(6.0, 6.0, width=2.0, name="knife", scrape=1.0)
