"""fusain: a drawing simulator for dry media (pencils, charcoal, pastel, crayon).

Quick start::

    from fusain import Sheet, papers

    sheet = Sheet(800, 600, paper=papers.INGRES, seed=1)
    sheet.save("paper.png", light=(-0.6, -0.8, 0.5))   # a raking light
"""

from . import papers
from ._about import __version__
from .paper import Paper
from .sheet import Sheet

__all__ = ["Paper", "Sheet", "__version__", "papers"]
