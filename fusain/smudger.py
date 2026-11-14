"""A tool in the hand: a stump, a finger or a chamois, and what it carries."""

from __future__ import annotations

import numpy as np

from ._engine import HELD, IV, LOOSE
from .stick import _width
from .tools import Tool


class Smudger:
    """A stump, a finger or a chamois as it is held: the tool, and the
    pigment it has picked up, which it keeps from one pass to the next until
    it is cleaned. A stump that is loaded gives back about as much as it
    takes, so it blends a tone without wearing it away, and it draws a pale
    tone of its own on bare paper::

        from fusain import Sheet, Smudger, pencils, tools
        stump = Smudger(tools.STUMP)
        sheet.rub(patch, stump)           # it loads as it rubs
        sheet.smudge(path, stump)         # and tones the bare paper
        stump.clean()

    Rub with a bare tool (``sheet.smudge(path, tools.STUMP)``) for one that
    is clean every time and takes away what it lifted.

    Parameters
    ----------
    tool:
        What it is, e.g. ``tools.STUMP``.
    width:
        Its width in pixels (by default the tool's own).

    ``load`` is the volume of pigment it carries now. An eraser can be held
    this way too: it lays nothing back, so its load is what it has taken off
    since it was cleaned.
    """

    def __init__(self, tool: Tool, width: float | None = None):
        if not isinstance(tool, Tool):
            raise TypeError(f"tool must be a Tool, such as tools.STUMP, got {tool!r}")
        self.tool = tool
        self.width = _width("width", tool.width if width is None else width)
        # what it has lifted, the dust on its face and the held graphite it
        # is pushing along, each packed as the dust of a sheet is, with the
        # volume of it that is paste after
        self._carried = np.zeros((3, HELD))

    @property
    def load(self) -> float:
        """Volume of pigment it carries now."""
        return float(self._carried[:, IV].sum())

    @property
    def carried(self) -> np.ndarray:
        """What it carries, packed as the dust of a sheet is: [K (N bands),
        S, volume, sheen]."""
        return self._carried[:, :LOOSE].sum(axis=0)

    def clean(self) -> float:
        """Clean it (a stump on sandpaper, a finger on a rag): it carries
        nothing afterwards. Returns the volume of pigment that came off."""
        load = self.load
        self._carried[:] = 0.0
        return load

    def __repr__(self) -> str:
        name = self.tool.name or "tool"
        return f"Smudger({name}, width={self.width:g}, load={self.load:.2f})"
