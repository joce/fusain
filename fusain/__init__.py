"""fusain: a drawing simulator for dry media (pencils, charcoal, pastel, crayon).

Quick start::

    from fusain import Sheet, papers

    sheet = Sheet(800, 600, paper=papers.INGRES, seed=1)
    sheet.save("paper.png", light=(-0.6, -0.8, 0.5))   # a raking light
"""

from . import charcoal, papers, pencils, tools
from ._about import __version__
from .medium import Medium
from .paper import Paper
from .sheet import Sheet
from .tools import Tool

__all__ = [
    "Medium",
    "Paper",
    "Sheet",
    "Tool",
    "__version__",
    "charcoal",
    "papers",
    "pencils",
    "tools",
]
