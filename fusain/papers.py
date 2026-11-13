"""Papers to draw on: ``Sheet(1200, 800, paper=papers.INGRES)``."""

from __future__ import annotations

from .paper import Paper

#: The everyday drawing paper: a fine, even tooth.
CARTRIDGE = Paper(
    "wove", grain=3.0, depth=1.0, bite=0.5, color="#f3f0e6", name="cartridge"
)
#: Plain office paper: bright white, smooth and thin-toothed. It takes a
#: pencil line well but few layers.
PRINTER = Paper(
    "wove",
    grain=1.5,
    depth=0.4,
    bite=0.4,
    give=0.3,
    color="#fafbfb",
    name="printer paper",
)
#: Smooth board for fine pencil work: hardly any tooth, few layers.
BRISTOL = Paper(
    "wove", grain=2.0, depth=0.25, bite=0.3, give=0.2, color="#f8f7f2", name="bristol"
)
#: Thin, soft and greyish: quick charcoal studies.
NEWSPRINT = Paper(
    "wove", grain=2.5, depth=0.6, bite=0.45, give=0.6, color="#dcd6c6", name="newsprint"
)
#: Laid paper for charcoal and pastel: its lines show through a light tone.
INGRES = Paper("laid", grain=3.0, depth=1.0, bite=0.55, color="#efe9da", name="Ingres")
#: Watercolour paper, cold-pressed: a marked, bumpy tooth.
COLD_PRESSED = Paper(
    "rough",
    grain=5.0,
    depth=1.3,
    bite=0.6,
    give=0.7,
    color="#f2eee2",
    name="cold-pressed",
)
#: Rough watercolour paper: deep hollows that a light stroke skips.
ROUGH = Paper(
    "rough", grain=8.0, depth=1.8, bite=0.7, give=0.7, color="#f1ede0", name="rough"
)
#: Sanded pastel card: sharp grit that takes many layers.
SANDED = Paper(
    "sanded",
    grain=2.0,
    depth=1.2,
    bite=0.95,
    give=0.3,
    color="#d9cdb8",
    name="sanded card",
    hold=2.5,
)
#: Toned papers, to draw the lights as well as the darks.
TONED_GREY = CARTRIDGE.with_(color="#a9a8a4", name="toned grey")
TONED_TAN = CARTRIDGE.with_(color="#c3a680", name="toned tan")
#: Black paper, for pastel and white chalk.
BLACK = CARTRIDGE.with_(color="#1c1b1a", name="black")

ALL = (
    CARTRIDGE,
    PRINTER,
    BRISTOL,
    NEWSPRINT,
    INGRES,
    COLD_PRESSED,
    ROUGH,
    SANDED,
    TONED_GREY,
    TONED_TAN,
    BLACK,
)
