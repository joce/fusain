"""A sketch of an art deco tower, in charcoal on Ingres paper.

Run from the repository root: ``python examples/deco.py``. Writes
``examples/output/deco.png``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from murex import filters, write_png

from fusain import Sheet, charcoal, papers, tools

WIDTH, HEIGHT = 900, 1300

#: The camera: where the eye is (x and y on the ground, z up, in metres),
#: what it looks at, its focal length in pixels and where on the sheet the
#: point it looks at lands.
EYE = np.array([-262.0, -205.0, 2.0])
TARGET = np.array([0.0, 0.0, 170.0])
FOCAL = 1230.0
CENTRE = (470.0, 640.0)
#: Toward the sun: from the left, a little behind the eye and fairly high, so
#: that the faces turned to -x are lit and those turned to -y are not.
SUN = np.array([-0.62, 0.5, 0.6]) / math.sqrt(0.62**2 + 0.5**2 + 0.6**2)

#: The tower, tier by tier from the street: half its width, the height of its
#: top, and the bay that stands out of the middle of each face (half its
#: width, half the width of the frontispiece in its middle, how far it stands
#: out). A bay rises past the top of its tier, the frontispiece higher still.
TIERS = (
    (43.0, 24.0, 0.0, 0.0, 0.0),
    (36.0, 150.0, 18.0, 9.0, 4.0),
    (28.0, 214.0, 13.5, 6.5, 3.4),
    (21.5, 262.0, 10.5, 0.0, 2.8),
    (15.5, 290.0, 7.5, 0.0, 2.2),
    (11.0, 306.0, 0.0, 0.0, 0.0),
    (7.0, 318.0, 0.0, 0.0, 0.0),
    (4.0, 330.0, 0.0, 0.0, 0.0),
)
BAY_RISE, FRONT_RISE, FRONT_OUT = 0.4, 0.72, 1.5
#: The mast: its foot, its tip and half its width at the foot.
MAST = (330.0, 374.0, 1.5)
#: A ledge runs round the top of every tier and bay: how far it stands out
#: and how deep it is.
LEDGE_OUT, LEDGE_DEEP = 0.9, 1.6
#: A floor, a bay between two piers, the solid corner of a face and the
#: parapet over the top windows, in metres.
FLOOR, PITCH, CORNER, PARAPET = 4.0, 4.6, 1.4, 3.2

#: What is seen at a pixel that no face covers.
SKY, GROUND, NEEDLE = 0, -1, -2

Tone = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _basis() -> np.ndarray:
    """The camera's right, up and forward, as the rows of a matrix."""
    forward = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(right, forward), forward])


BASIS = _basis()


def project(points) -> np.ndarray:
    """Where on the sheet the camera sees ``points`` (..., 3): (..., 2)."""
    across, up, depth = np.moveaxis((np.asarray(points, float) - EYE) @ BASIS.T, -1, 0)
    return np.stack(
        [CENTRE[0] + FOCAL * across / depth, CENTRE[1] - FOCAL * up / depth], axis=-1
    )


@dataclass(frozen=True)
class Box:
    """A block of the building: its low and high corners (x, y, z), what it
    is, the tier it belongs to and, for its face turned to -x and for the one
    turned to -y, the stretches that hold windows: (from, to) along the face
    and the height the windows start at."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    part: str
    tier: int = 0
    bays: tuple[tuple, tuple] = ((), ())


@dataclass(frozen=True)
class Face:
    """An upright face of a box, turned to -x (``axis`` 0) or to -y (1): on
    the plane where that coordinate is ``at``, from ``lo`` to ``hi`` along
    the other and from ``z0`` to ``z1`` in height."""

    ident: int
    axis: int
    box: Box

    @property
    def at(self) -> float:
        return self.box.lo[self.axis]

    @property
    def lo(self) -> float:
        return self.box.lo[1 - self.axis]

    @property
    def hi(self) -> float:
        return self.box.hi[1 - self.axis]

    @property
    def z0(self) -> float:
        return self.box.lo[2]

    @property
    def z1(self) -> float:
        return self.box.hi[2]

    @property
    def lit(self) -> bool:
        return self.axis == 0

    def point(self, along, z) -> np.ndarray:
        """Where on the sheet the point ``along`` the face at height ``z`` is."""
        along, z = np.broadcast_arrays(np.asarray(along, float), np.asarray(z, float))
        at = np.full(along.shape, self.at)
        order = (at, along, z) if self.axis == 0 else (along, at, z)
        return project(np.stack(order, axis=-1))


def ident(index: int, axis: int) -> int:
    """What the map of what is seen holds for a face of the box ``index``."""
    return 3 * index + axis + 1


def cross(half: float, wide: float, z0: float, z1: float, part: str, tier: int, bays):
    """The two boxes of a cross in plan: arms ``wide`` either side of the
    axes, reaching ``half`` from the middle. ``bays`` are the windows of the
    front of an arm and of its side."""
    front, side = bays
    return [
        Box((-half, -wide, z0), (half, wide, z1), part, tier, (front, side)),
        Box((-wide, -half, z0), (wide, half, z1), part, tier, (side, front)),
    ]


def ledge(box: Box) -> Box:
    """The ledge round the top of a box."""
    x0, y0, _ = box.lo
    x1, y1, top = box.hi
    out = LEDGE_OUT
    return Box(
        (x0 - out, y0 - out, top - LEDGE_DEEP), (x1 + out, y1 + out, top), "ledge", box.tier
    )


def tier_boxes(k: int) -> list[Box]:
    """The boxes of one tier: its body, its bay and its frontispiece."""
    half, top, wide, inner, out = TIERS[k]
    below = TIERS[k - 1][1] if k else 0.0
    if not wide:
        span = ((-half + CORNER, half - CORNER, below),)
        return [Box((-half, -half, below), (half, half, top), "tier", k, (span, span))]
    wings = ((-half + CORNER, -wide, below), (wide, half - CORNER, below))
    boxes = [Box((-half, -half, below), (half, half, top), "tier", k, (wings, wings))]
    step = TIERS[k + 1][1] - top
    behind = TIERS[k + 1][0]
    side = ((-half - out + CORNER, -behind, top),)
    if inner:
        front = ((-wide + CORNER, -inner, below), (inner, wide - CORNER, below))
    else:
        front = ((-wide + CORNER, wide - CORNER, below),)
    rise = top + BAY_RISE * step
    boxes += cross(half + out, wide, below, rise, "bay", k, (front, side))
    if inner:
        front = ((-inner + CORNER, inner - CORNER, below),)
        rise = top + FRONT_RISE * step
        boxes += cross(half + out + FRONT_OUT, inner, below, rise, "bay", k, (front, ()))
    return boxes


def tower() -> list[Box]:
    """The boxes of the tower, with the ledges round their tops."""
    boxes = [box for k in range(len(TIERS)) for box in tier_boxes(k)]
    return boxes + [ledge(box) for box in boxes]


def neighbours() -> list[Box]:
    """The lower blocks around the foot of the tower."""
    blocks = (
        ((-150.0, -60.0), (40.0, 150.0), 46.0),
        ((-200.0, -150.0), (-90.0, -30.0), 22.0),
        ((60.0, 130.0), (-150.0, -85.0), 40.0),
        ((85.0, 140.0), (-185.0, -160.0), 20.0),
    )
    return [Box((x[0], y[0], 0.0), (x[1], y[1], top), "block") for x, y, top in blocks]


def hits(start: np.ndarray, rays: np.ndarray, box: Box):
    """Where rays meet a box: how far along each ray (infinite if it misses)
    and the axis of the face it enters by."""
    with np.errstate(divide="ignore", invalid="ignore"):
        one, other = (np.asarray(box.lo) - start) / rays, (np.asarray(box.hi) - start) / rays
    near = np.nan_to_num(np.minimum(one, other), nan=-np.inf)
    far = np.nan_to_num(np.maximum(one, other), nan=np.inf)
    enter, leave = near.max(axis=-1), far.min(axis=-1)
    return np.where((enter < leave) & (enter > 0), enter, np.inf), near.argmax(axis=-1)


def cast(boxes: list[Box]) -> tuple[np.ndarray, np.ndarray]:
    """What the eye sees at each pixel, (H, W): the ``ident`` of a face,
    ``SKY`` or ``GROUND``; and the point of the scene it sees there."""
    px, py = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    across, up = (px - CENTRE[0]) / FOCAL, (CENTRE[1] - py) / FOCAL
    rays = across[..., None] * BASIS[0] + up[..., None] * BASIS[1] + BASIS[2]
    down = rays[..., 2] < 0
    seen = np.where(down, GROUND, SKY).astype(np.int32)
    depth = np.where(down, -EYE[2] / np.minimum(rays[..., 2], -1e-9), np.inf)
    for index, box in enumerate(boxes):
        reach, axis = hits(EYE, rays, box)
        nearer = reach < depth
        depth[nearer] = reach[nearer]
        seen[nearer] = ident(index, 0) + axis[nearer]
    return seen, EYE + np.where(np.isfinite(depth), depth, 0.0)[..., None] * rays


def shadows(boxes: list[Box], seen: np.ndarray, where: np.ndarray):
    """Which pixels show a face turned from the sun, and which a face turned
    to it that a box stands in front of: two maps, (H, W)."""
    axis = (seen - 1) % 3
    normal = np.zeros(where.shape)
    for k in range(3):  # (the eye sees the faces turned to -x, -y and down)
        normal[..., k] = np.where((seen > 0) & (axis == k), -1.0, 0.0)
    normal[seen == GROUND] = (0.0, 0.0, 1.0)
    turned = (normal @ SUN <= 0) & (seen != SKY)
    start = where + 0.02 * normal
    rays = np.broadcast_to(SUN, where.shape)
    hidden = np.zeros(seen.shape, bool)
    for box in boxes:
        hidden |= np.isfinite(hits(start, rays, box)[0])
    return turned, hidden & ~turned & (seen != SKY)


def main() -> None:
    boxes = tower() + neighbours()
    seen, where = cast(boxes)
    turned, hidden = shadows(boxes, seen, where)
    img = np.full((HEIGHT, WIDTH), 0.55)
    img[seen == GROUND] = 0.4
    img[seen > 0] = 0.95
    img[hidden] = 0.5
    img[turned] = 0.22
    edge = np.zeros(seen.shape, bool)
    edge[1:] |= seen[1:] != seen[:-1]
    edge[:, 1:] |= seen[:, 1:] != seen[:, :-1]
    img[edge] *= 0.6
    out = Path("/tmp/claude-0/-home-user/3fc6ff15-148f-5405-9e2d-4b5533fe522f/scratchpad/deco/plan.png")
    write_png(str(out), np.round(np.repeat(img[..., None], 3, 2) * 255).astype(np.uint8))
    for z in (0, 24, 150, 330, 374):
        print(z, project((0, 0, z)).round())


if __name__ == "__main__":
    main()
    _ = (filters, Sheet, charcoal, papers, tools, Iterator)
