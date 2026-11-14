"""Morning mist on a mountain lake, in graphite pencil.

Run from the repository root: ``python examples/lake.py``. Writes
``examples/output/lake.png`` (about two minutes). With a folder as argument
(``python examples/lake.py studies/``) it also writes there the tonal study
the drawing is planned from, and its maps.

A mountain lake at dawn, seen from its near shore. The scene is built in
space before any pencil touches it: three ridges of mountains as height
fields, each further and higher than the one in front; forested hills and a
far shore of conifers; still water at the eye's feet that mirrors them all;
mist lying in bands on the water and in the valleys between the ridges; on
the near shore a group of spruces, each grown from its trunk, whorl by
whorl, and a dead snag among them; rocks; reeds; and a clinker-built rowing
boat drawn up on the shingle, its painter made fast to a stake. One light:
the glow of the sun still below the ridges, and the soft light of the sky,
so that near things stand dark against the light, their tops catching a
little of it. Seen through a camera, the scene gives the tone each part of
the sheet should reach, what is where, how veiled it is, and the direction
strokes should take; the hand then draws far to near, stepping back now and
then to see where tone is still wanting.

What each lead and tool does, far to near:

* Values: the palest tenth of the study is left as paper (the mist, the low
  sky, the snow, the far water under the mist), and the darkest is the
  full weight of 8B; the near ground, the bank and the rocks are given
  their planes before any mark is made.
* Sky: a 4B laid very lightly on its side, darker towards the top, in three
  crossed layers, each rubbed in with the finger until its grain is gone;
  the glow wiped lighter with the chamois, a few low clouds lifted with the
  kneaded eraser.
* Far ridges: a 2B (a 4B for the nearest) laid lightly on its side and
  rubbed with the finger to a haze, then rubbed level once more with the
  finger, easing in from its edges; 2H and HB down the fall lines of the
  nearer two; each crest a band of tone laid along it, firm at its upper
  edge and fading below (2H on the farthest, HB, then 2B on the nearest),
  pulled down into the mist with the stump. The snow lifted with the
  kneaded eraser.
* Forest and far shore: a 2B on its side in near-vertical strokes rubbed
  across with the stump; then every tree in short vertical 2B strokes of
  different heights, as hard as the tone still wants, hardest at
  mid-height and lifting off into the mist, a few tall ones standing clear
  as fine HB spires; the stump pulled down from their feet into the mist.
* Mist: reserved paper, long soft passes of the kneaded eraser wherever
  what lies behind it came out too dark, and at the end the vinyl eraser
  held flat and light where the kneaded one left grey.
* Water: a 2B on its side in crossing layers rubbed level with the finger;
  each reflection in its own lead, one grade softer than what it mirrors
  (2B for the forest, 6B for rocks, bank and boat, 8B for the spruces), on
  its side in two crossing layers, rubbed softly together and dragged
  gently down with the finger, a touch lighter than the thing and fading
  downward, then a last light layer (across and rubbed under near things,
  vertical under the far shore); the far water rubbed level; thin lights
  lifted across the reflections, two across the boat's, one across each
  rock's; the breath of wind
  lifted in overlapping passes of the kneaded eraser's edge, touched with
  the vinyl one near the far shore.
* Near shore: the point's bank one quiet dark tone, a 6B half on its side
  in two crossed layers and an 8B over them, each rubbed smooth with the
  stump and the finger; a narrow light along its top lifted with the
  kneaded eraser where the glow catches it; a few clumps of flicked 6B
  grass along that top and at the water's edge, nothing between but two or
  three 8B roots at each tree's foot; a dark 8B bank edge. The shingle
  bank a 2B and 4B rubbed, lighter on top and darker to the water, stones
  as small dabs, and near the eye pebbles, each a 4B arc beneath and a top
  lifted with the kneaded eraser.
* Rocks: kept clear of the bank's tone; a 4B half on its side rubbed with
  the tortillon; the sides in short 4B strokes round the form with a
  tortillon over them, the undersides and lower edges built dark in 8B,
  the lit tops lifted with the kneaded eraser; the waterline in 8B, one
  crack each with a sharp 6B.
* Spruces: 4B bundles along the boughs, each bough and trunk drawn along
  its length in 6B, then 8B at full pressure in layers into their cores,
  the ragged lit edges left; needles hung from the boughs with a 6B
  ``Stick`` sharpened as it goes.
* Boat: strokes laid along the planks, through the hull at many heights
  round its girth (HB, 2B, 4B outside; 4B, 6B, 8B inside), rubbed along with
  a tortillon, the inside then deepened in 8B; the thwarts and the gunwale
  left to catch the light; the sheers, the lands of the strakes, the stem
  and the painter in clean lines of a sharp 2B ``Stick``, sharpened after
  each; the stake in 8B; the reeds flicked up in 4B.

The paper is a smooth white drawing paper, shallower in the tooth than
cartridge. The sheet is rendered with ``gloss=0.2``: as a drawing looks
scanned or in soft light, the heaviest graphite keeping a little of its
silver while the darks stay deep.
"""

# cspell:words clinker thwart thwarts gunwale gunwales transom icosphere Catmull
# cspell:words keepdims interp geomspace lognormal zbuf unrubbed

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from murex import CatmullRom, filters, noise, write_png
from numba import njit

from fusain import Medium, Sheet, Stick, papers, pencils, tools

HERE = Path(__file__).parent
WIDTH, HEIGHT = 1500, 1000
#: The camera: focal length in pixels, where the eye's axis meets the sheet
#: (the horizon is the row CY), and the eye's height above the water in
#: metres. World axes: X to the right, Y up, Z away from the eye.
FOCAL = 1600.0
CX, CY = 750.0, 540.0
EYE = 2.2
SEED = 1907
#: Nothing nearer than this is drawn; rays end this far away (metres).
NEAR, FAR = 0.5, 40000.0

#: What each pixel shows.
SKY, CLOUD, SNOW, RIDGE3, RIDGE2, RIDGE1, HILLS, FOREST, WATER, GROUND = range(10)
SHINGLE, ROCK, REED, TRUNK, BOUGH, HULL, INSIDE, GUNWALE, THWART, ROPE = range(10, 20)
STAKE = 20
NAMES = [
    *("sky", "cloud", "snow", "ridge3", "ridge2", "ridge1", "hills", "forest"),
    *("water", "ground", "shingle", "rock", "reed", "trunk", "bough", "hull"),
    *("inside", "gunwale", "thwart", "rope", "stake"),
]
#: The depth layer of each: 0 the sky ... 7 the boat, the nearest thing.
LAYER = np.array([0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8])

#: Towards the glow of the sun, still below the ridges (from the scene).
SUN = np.array([0.11, 0.0, 1.0]) / math.hypot(0.11, 1.0)
#: The range of the pencil: the paper's grey and the darkest graphite, as
#: rendered (0..255).
PAPER, DARKEST = 241.0, 86.0


# ----------------------------------------------------------------------
# geometry


@dataclass
class Mesh:
    """Triangles in the world: vertices (n, 3), faces (m, 3), a normal per
    vertex, and per face what it shows from outside and from inside (ids and
    albedos) and the direction strokes would follow on it (a world vector)."""

    verts: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    ids: np.ndarray  # (m, 2) outside, inside
    albedo: np.ndarray  # (m, 2)
    grain: np.ndarray  # (m, 3)


@dataclass
class Stamps:
    """Discs in the world: centres (n, 3), radii (n,), id, albedo and the
    direction of the line they lie along (a world vector, (n, 3))."""

    points: np.ndarray
    radii: np.ndarray
    ids: np.ndarray
    albedo: np.ndarray
    grain: np.ndarray


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (verts[faces[:, k]] for k in range(3))
    fn = np.cross(b - a, c - a)
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def grid_faces(rows: int, cols: int) -> np.ndarray:
    """Two triangles per cell of a rows x cols grid of vertices."""
    i = np.arange((rows - 1) * (cols - 1))
    r, c = i // (cols - 1), i % (cols - 1)
    v = r * cols + c
    quads = np.stack([v, v + 1, v + cols + 1, v + cols], axis=1)
    return np.concatenate([quads[:, [0, 2, 1]], quads[:, [0, 3, 2]]])


def make_mesh(verts, faces, ident, albedo, grain=None, inside=(-1, 0.0), normals=None):
    m = len(faces)
    ids = np.empty((m, 2), np.int64)
    alb = np.empty((m, 2))
    ids[:, 0], ids[:, 1] = ident, inside[0]
    alb[:, 0], alb[:, 1] = albedo, inside[1]
    if grain is None:
        grain = np.zeros((m, 3))
    grain = np.broadcast_to(np.asarray(grain, float), (m, 3)).copy()
    if normals is None:
        normals = vertex_normals(verts, faces)
    return Mesh(verts, faces, normals, ids, alb, grain)


def merge_meshes(meshes: list[Mesh]) -> Mesh:
    offsets = np.cumsum([0] + [len(m.verts) for m in meshes[:-1]])
    return Mesh(
        np.concatenate([m.verts for m in meshes]),
        np.concatenate([m.faces + o for m, o in zip(meshes, offsets, strict=True)]),
        np.concatenate([m.normals for m in meshes]),
        np.concatenate([m.ids for m in meshes]),
        np.concatenate([m.albedo for m in meshes]),
        np.concatenate([m.grain for m in meshes]),
    )


def world_map(rng, shape, cell, octaves=3):
    """Fractal noise over a grid, 0..1 (murex)."""
    return noise.fractal_noise_2d(shape[0], shape[1], cell, rng, octaves=octaves)


def sample_grid(grid: np.ndarray, extent, X, Z):
    """Sample a grid laid over the world rectangle ``extent`` (x0, z0, x1,
    z1) at world points (X, Z)."""
    x0, z0, x1, z1 = extent
    gh, gw = grid.shape[:2]
    fx = np.clip((np.asarray(X) - x0) / (x1 - x0) * (gw - 1), 0, gw - 1.001)
    fz = np.clip((np.asarray(Z) - z0) / (z1 - z0) * (gh - 1), 0, gh - 1.001)
    return filters.bilinear(grid, fz, fx)


def frustum_grid(z_rows: np.ndarray, cols: int, reach: float = 1.12):
    """World X, Z of a grid whose columns fan out with the view: each row at
    one distance, spanning a little more than the sheet's width."""
    u = np.linspace(-reach, reach, cols)
    Z = np.repeat(z_rows[:, None], cols, axis=1)
    X = u[None, :] * Z * CX / FOCAL
    return X, Z


def height_mesh(X, Y, Z, ident, albedo) -> Mesh:
    """Ground as triangles, its normals up; strokes on it follow the fall
    line."""
    rows, cols = X.shape
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    faces = grid_faces(rows, cols)
    mesh = make_mesh(verts, faces, ident, albedo)
    flip = mesh.normals[:, 1] < 0  # (terrain faces up)
    if flip.mean() > 0.5:
        mesh.faces = mesh.faces[:, [0, 2, 1]]
        mesh.normals = -mesh.normals
    n = mesh.normals[mesh.faces].mean(axis=1)
    down = np.array([0.0, -1.0, 0.0]) - n * (-n[:, 1:2])
    flat = np.linalg.norm(down, axis=1) < 0.05
    down[flat] = (1.0, 0.0, 0.0)  # (level ground: across the view)
    mesh.grain = down
    return mesh


# ----------------------------------------------------------------------
# the mountains


@dataclass(frozen=True)
class Ridge:
    """A ridge: the distance of its crest, how far its slopes run in front
    and behind, its crest heights at points across the view (as a share of
    the sheet's half width, -1 left edge .. 1 right edge), how rough it is,
    the height above which snow lies, its albedo and id."""

    z: float
    front: float
    back: float
    crest: tuple[tuple[float, float], ...]
    rough: float
    snow: float
    albedo: float
    ident: int
    gullies: float


RIDGES = (
    Ridge(
        14500.0,
        5000.0,
        3000.0,
        (
            (-1.3, 1300),
            (-0.75, 1650),
            (-0.42, 2450),
            (-0.18, 1900),
            (0.1, 1350),
            (0.32, 1500),
            (0.62, 2050),
            (0.95, 1700),
            (1.3, 1400),
        ),
        0.035,
        2180.0,
        0.30,
        RIDGE3,
        0.06,
    ),
    Ridge(
        8200.0,
        3200.0,
        2200.0,
        (
            (-1.3, 1150),
            (-0.85, 1080),
            (-0.55, 900),
            (-0.2, 760),
            (0.1, 480),
            (0.3, 560),
            (0.58, 840),
            (0.85, 1000),
            (1.3, 1080),
        ),
        0.06,
        9e9,
        0.26,
        RIDGE2,
        0.14,
    ),
    Ridge(
        4300.0,
        1700.0,
        1300.0,
        (
            (-1.3, 520),
            (-0.95, 470),
            (-0.6, 380),
            (-0.3, 300),
            (0.0, 200),
            (0.15, 150),
            (0.34, 210),
            (0.6, 330),
            (0.9, 360),
            (1.3, 400),
        ),
        0.09,
        9e9,
        0.14,
        RIDGE1,
        0.22,
    ),
)


def crest_heights(ridge: Ridge, u: np.ndarray, rng) -> np.ndarray:
    """The ridge's crest height at view fractions ``u``: a smooth curve
    through its points, roughened with murex noise."""
    knots = np.array(ridge.crest, float)
    pts, _, _, _ = CatmullRom(knots).sample(0.002)
    order = np.argsort(pts[:, 0])
    base = np.interp(u, pts[order, 0], pts[order, 1])
    n = len(u)
    detail = (
        0.6 * (noise.value_noise_1d(n, n / 18, rng) - 0.5)
        + 0.3 * (noise.value_noise_1d(n, n / 55, rng) - 0.5)
        + 0.1 * (noise.value_noise_1d(n, n / 160, rng) - 0.5)
    )
    return base * (1.0 + 2.0 * ridge.rough * detail)


def ridge_mesh(ridge: Ridge, rng) -> Mesh:
    rows, cols = 120, 520
    z_rows = np.linspace(ridge.z - ridge.front, ridge.z + ridge.back, rows)
    X, Z = frustum_grid(z_rows, cols, reach=1.15)
    # the crest is a line across at the ridge's distance; each point of the
    # slopes takes the crest height of where it lies across
    u_crest = X * FOCAL / (ridge.z * CX)
    u_line = np.linspace(-1.5, 1.5, 1200)
    crest = np.interp(u_crest, u_line, crest_heights(ridge, u_line, rng))
    t = np.where(ridge.z > Z, (ridge.z - Z) / ridge.front, (Z - ridge.z) / ridge.back)
    slope = np.clip(1.0 - t, 0.0, 1.0) ** 1.35
    # spurs and gullies running down the fall line, broader low down
    gully = noise.aniso_noise(rows, cols, 14.0, 3.5, 90.0, rng, octaves=3)
    bumps = world_map(rng, (rows, cols), 22.0, 4)
    Y = crest * slope * (1.0 + ridge.gullies * (gully - 0.5) * (1.2 - slope))
    Y *= 1.0 + 0.12 * (bumps - 0.5)
    Y -= 40.0
    mesh = height_mesh(X, Y, Z, ridge.ident, ridge.albedo)
    snow_faces(mesh, ridge, bumps)
    return mesh


def snow_faces(mesh: Mesh, ridge: Ridge, patches: np.ndarray) -> None:
    """Snow lies on the highest ground, in the hollows more than on the
    ribs, and not where it is steep."""
    if ridge.snow > 1e8:
        return
    fy = mesh.verts[mesh.faces, 1].mean(axis=1)
    ny = mesh.normals[mesh.faces, 1].mean(axis=1)
    line = ridge.snow + 300.0 * (patches.ravel()[mesh.faces].mean(axis=1) - 0.5)
    snowy = (fy > line) & (ny > 0.45)
    mesh.ids[snowy, 0] = SNOW
    mesh.albedo[snowy, 0] = 0.95


# ----------------------------------------------------------------------
# the middle distance: hills and the far shore, under forest

SHORE_Z = 1150.0


def far_shore(X: np.ndarray, rng_line: np.ndarray) -> np.ndarray:
    """Distance of the far shore at X: an irregular line, with a bay."""
    u = X / 1400.0
    bay = -160.0 * np.exp(-(((u - 0.15) / 0.12) ** 2))
    return SHORE_Z + np.interp(u, np.linspace(-2, 2, len(rng_line)), rng_line) + bay


def hills_height(X, Z, line, bumps):
    rise = np.clip(Z - far_shore(X, line), -60.0, None)
    shore = np.where(rise < 0, rise * 0.2, np.minimum(rise * 0.12, 26.0))
    hill = 200.0 * bumps * filters.smoothstep(150.0, 1000.0, rise)
    return shore + hill - 2.0


def hills(rng) -> tuple[Mesh, Mesh, np.ndarray]:
    """The forested hills beyond the lake and the trees on them, as
    triangles and as (X, foot height, Z, height, lean) per tree."""
    line = 220.0 * (noise.value_noise_1d(400, 22, rng) - 0.5)
    rows, cols = 150, 520
    z_rows = np.geomspace(SHORE_Z - 300.0, 3300.0, rows)
    X, Z = frustum_grid(z_rows, cols, reach=1.2)
    extent = (-3500.0, 800.0, 3500.0, 3400.0)
    bumps = world_map(rng, (130, 350), 30.0, 3)
    B = sample_grid(bumps, extent, X, Z) ** 1.3
    Y = hills_height(X, Z, line, B)
    ground = height_mesh(X, Y, Z, HILLS, 0.13)
    trees, spec = forest(rng, line, bumps, extent)
    return ground, trees, spec


def forest(rng, line, bumps, extent) -> tuple[Mesh, np.ndarray]:
    """Conifers on the hills and along the far shore: each a spire of three
    or four tiers, a few metres apart, thinning to clearings."""
    pts = []
    for z in np.arange(SHORE_Z - 200.0, 3200.0, 7.0):
        spacing = 7.0 + z / 300.0
        half = z * 820.0 / FOCAL
        xs = np.arange(-half, half, spacing) + rng.uniform(0, spacing, 1)
        xs = xs + rng.uniform(-0.45, 0.45, len(xs)) * spacing
        zs = np.full(len(xs), z) + rng.uniform(-3.0, 3.0, len(xs))
        pts.append(np.stack([xs, zs], axis=1))
    P = np.concatenate(pts)
    clear = sample_grid(world_map(rng, (90, 240), 14.0, 3), extent, P[:, 0], P[:, 1])
    B = sample_grid(bumps, extent, P[:, 0], P[:, 1]) ** 1.3
    Yb = hills_height(P[:, 0], P[:, 1], line, B)
    clumps = sample_grid(world_map(rng, (90, 240), 6.0, 2), extent, P[:, 0], P[:, 1])
    keep = (Yb > 0.5) & (clear > 0.44) & (rng.random(len(P)) > 0.2)
    P, Yb, clumps = P[keep], Yb[keep], clumps[keep]
    heights = np.clip(rng.lognormal(np.log(18.0), 0.3, len(P)), 7.0, 36.0)
    heights *= 0.55 + 0.9 * clumps  # (taller stands and lower ones)
    lean = rng.normal(0.0, 0.03, len(P))
    spec = np.stack([P[:, 0], Yb, P[:, 1], heights, lean], axis=1)
    return spires(P, Yb, heights, lean, rng), spec


def spires(P, Yb, heights, lean, rng) -> Mesh:
    """Conifer silhouettes, three tiers each, as triangles facing the eye."""
    n = len(P)
    tiers = 3
    verts, faces = [], []
    for k in range(tiers):
        lo = 0.12 + 0.27 * k + rng.uniform(-0.04, 0.04, n)
        hi = np.minimum(lo + 0.5 + rng.uniform(0, 0.12, n), 1.0)
        if k == tiers - 1:
            hi = np.ones(n)
        half = 0.24 * heights * (1.0 - 0.32 * k) * rng.uniform(0.75, 1.15, n)
        base_y = Yb + lo * heights
        top_y = Yb + hi * heights
        cx = P[:, 0] + lean * heights * hi
        v = np.stack(
            [
                np.stack([P[:, 0] - half, base_y, P[:, 1]], 1),
                np.stack([P[:, 0] + half, base_y, P[:, 1]], 1),
                np.stack([cx, top_y, P[:, 1]], 1),
            ],
            axis=1,
        )
        faces.append(np.arange(3 * n).reshape(n, 3) + 3 * n * k)
        verts.append(v.reshape(-1, 3))
    verts = np.concatenate(verts)
    faces = np.concatenate(faces)
    normals = np.tile(
        np.array([0.0, 0.35, -1.0]) / math.hypot(0.35, 1.0), (len(verts), 1)
    )
    mesh = make_mesh(verts, faces, FOREST, 0.09, (0.0, 1.0, 0.0), normals=normals)
    mesh.albedo[:, 0] = np.tile(rng.uniform(0.06, 0.13, n), tiers)
    return mesh


# ----------------------------------------------------------------------
# the near shore

#: The near shores, as polygons in (X, Z): the point where the pines stand,
#: on the left, and the shingle bank the boat is drawn up on, on the right.
POINT_LAND = (
    (-60.0, 4.0),
    (-2.75, 4.0),
    (-2.9, 7.0),
    (-3.5, 10.0),
    (-4.5, 14.0),
    (-6.0, 20.0),
    (-8.1, 30.0),
    (-10.0, 42.0),
    (-11.8, 55.0),
    (-12.6, 65.0),
    (-12.1, 73.0),
    (-13.6, 81.0),
    (-18.0, 89.0),
    (-28.0, 95.0),
    (-60.0, 99.0),
)
BANK = (
    (-0.4, 4.0),
    (0.45, 7.6),
    (1.45, 9.4),
    (2.7, 10.9),
    (3.9, 12.15),
    (5.4, 13.25),
    (7.4, 14.1),
    (10.4, 14.9),
    (16.0, 15.3),
    (16.0, 4.0),
)


def signed_distance(x: np.ndarray, z: np.ndarray, polygon) -> np.ndarray:
    """Distance to the polygon's outline, positive inside."""
    P = np.asarray(polygon, float)
    Q = np.roll(P, -1, axis=0)
    best = np.full(x.shape, np.inf)
    inside = np.zeros(x.shape, bool)
    for (ax, az), (bx, bz) in zip(P, Q, strict=True):
        ex, ez = bx - ax, bz - az
        t = np.clip(((x - ax) * ex + (z - az) * ez) / (ex * ex + ez * ez), 0.0, 1.0)
        best = np.minimum(best, np.hypot(x - ax - t * ex, z - az - t * ez))
        crosses = (az > z) != (bz > z)
        at = ax + (z - az) * ex / np.where(ez == 0, 1e-9, ez)
        inside ^= crosses & (x < at)
    return np.where(inside, best, -best)


@functools.cache
def shore_noise() -> np.ndarray:
    """How the shores wander from their outlines (murex noise, its own
    seed so that every caller sees the same)."""
    return world_map(np.random.default_rng(SEED + 5), (420, 300), 9.0, 4)


def ground_height(X, Z, bumps):
    """Height of the near shores above the water, and how far inland each
    point is (negative out in the water)."""
    wander = sample_grid(shore_noise(), (-70.0, 0.0, 30.0, 140.0), X, Z) - 0.5
    d_point = signed_distance(X, Z, POINT_LAND) + 2.4 * wander
    d_bank = signed_distance(X, Z, BANK) + 1.0 * wander
    point = 0.22 * np.clip(d_point, -2.0, 2.5) + 0.05 * np.clip(
        d_point - 2.5, 0.0, 40.0
    )
    point += 2.0 * np.exp(-(((X + 23.0) / 9.0) ** 2) - ((Z - 70.0) / 14.0) ** 2)
    bank = 0.09 * np.clip(d_bank, -2.0, 8.0)
    d = np.maximum(d_point, d_bank)
    rough = 0.25 * (bumps - 0.5) * np.clip(d, 0.0, 1.5)
    return np.maximum(point, bank) + rough - 0.04, d


def ground(rng) -> Mesh:
    rows, cols = 220, 700
    z_rows = np.geomspace(2.5, 130.0, rows)
    X, Z = frustum_grid(z_rows, cols, reach=1.08)
    extent = (-70.0, 0.0, 30.0, 140.0)
    bumps = sample_grid(world_map(rng, (280, 200), 6.0, 4), extent, X, Z)
    Y, d = ground_height(X, Z, bumps)
    mesh = height_mesh(X, Y, Z, GROUND, 0.085)
    fd = d.ravel()[mesh.faces].mean(axis=1)
    fx = X.ravel()[mesh.faces].mean(axis=1)
    beach = np.where(fx > -1.0, 6.0, -9.0)  # (the bank is shingle; the point grass)
    near_water = fd < beach + 0.8 * bumps.ravel()[mesh.faces].mean(axis=1)
    mesh.ids[near_water, 0] = SHINGLE
    mesh.albedo[near_water, 0] = 0.15
    return mesh


# rocks: (X, Z, size, how flat, yaw)
ROCKS = (
    (-3.35, 9.9, 0.55, 0.6, 0.3),
    (-2.85, 8.6, 0.26, 0.55, 1.2),
    (-6.4, 21.0, 0.75, 0.6, 0.7),
    (-6.0, 22.6, 0.45, 0.5, 2.4),
    (-12.0, 56.0, 1.35, 0.55, 2.8),
    (-11.6, 52.5, 0.5, 0.6, 0.9),
    (0.95, 8.7, 0.33, 0.55, 0.4),
    (5.25, 12.85, 0.42, 0.6, 1.4),
    (8.6, 14.7, 0.8, 0.55, 2.2),
)


def icosphere(levels: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    v = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    f = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(levels):
        cache: dict[tuple[int, int], int] = {}
        new = []
        for a, b, c in faces:
            ab, bc, ca = (
                midpoint(verts, cache, i, j) for i, j in ((a, b), (b, c), (c, a))
            )
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts), np.array(faces)


def midpoint(verts: list, cache: dict, a: int, b: int) -> int:
    """The index of the vertex halfway between ``a`` and ``b`` on the unit
    sphere, made once and shared by both faces of the edge."""
    key = (min(a, b), max(a, b))
    if key not in cache:
        m = verts[a] + verts[b]
        verts.append(m / np.linalg.norm(m))
        cache[key] = len(verts) - 1
    return cache[key]


def rock(x, z, size, flat, yaw, rng) -> Mesh:
    """A weathered boulder: a sphere pushed out in a few lobes and cut by a
    few planes, so that it is rounded with a few flats, flattened, half
    sunk."""
    v, f = icosphere(3)
    r = np.ones(len(v))
    for _ in range(5):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        r += rng.uniform(0.1, 0.3) * np.maximum(v @ u, 0.0) ** 3
    v = v * r[:, None]
    for _ in range(6):  # facets: planes that shave it
        u = rng.normal(size=3)
        u[1] = abs(u[1]) * 0.6
        u /= np.linalg.norm(u)
        cut = rng.uniform(0.75, 0.95)
        over = v @ u - cut
        v -= np.maximum(over, 0.0)[:, None] * u[None, :]
    c, s = math.cos(yaw), math.sin(yaw)
    v = v * np.array([1.0, flat, 0.8 + 0.3 * rng.random()]) * size
    v = v @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    v += np.array([x, size * flat * 0.35, z])
    mesh = make_mesh(v, f, ROCK, 0.2)
    fn = mesh.normals[f].mean(axis=1)
    mesh.grain = np.cross(fn, np.array([0.0, 1.0, 0.0]))  # round the form
    return mesh


# ----------------------------------------------------------------------
# the pines


@dataclass(frozen=True)
class Pine:
    x: float
    z: float
    height: float
    lean: float
    seed: int
    spread: float
    dead: bool = False


#: The pines: where each stands, how tall, how it leans, and how far its
#: boughs reach for its height (a young tree is bushier).
PINES = (
    Pine(-33.5, 76.0, 22.0, -0.015, 5, 0.95),
    Pine(-23.7, 66.0, 27.0, -0.01, 1, 1.1),
    Pine(-24.1, 72.0, 18.0, 0.02, 3, 0.85),
    Pine(-17.4, 62.0, 16.5, 0.03, 2, 1.0),
    Pine(-15.6, 66.0, 10.5, -0.01, 4, 1.25),
    Pine(-14.0, 76.0, 13.0, 0.045, 7, 0.55, True),
)


def ground_at(x: float, z: float) -> float:
    y, _ = ground_height(np.array([x]), np.array([z]), np.array([0.5]))
    return float(y[0])


def curve(points: np.ndarray, radii: np.ndarray, ident, albedo, step=0.04):
    """Stamps along a world polyline, about ``step`` metres apart."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / step), 2)
    si = np.linspace(0.0, s[-1], n)
    P = np.stack([np.interp(si, s, points[:, k]) for k in range(3)], axis=1)
    R = np.interp(si, s, radii)
    G = np.gradient(P, axis=0)
    return Stamps(P, R, np.full(n, ident), np.full(n, albedo), G)


def merge_stamps(parts: list[Stamps]) -> Stamps:
    return Stamps(
        np.concatenate([p.points for p in parts]),
        np.concatenate([p.radii for p in parts]),
        np.concatenate([p.ids for p in parts]),
        np.concatenate([p.albedo for p in parts]),
        np.concatenate([p.grain for p in parts]),
    )


def bough(
    root, length, azimuth, droop, rng, bare=False
) -> tuple[np.ndarray, list[Stamps]]:
    """One bough: out from the trunk, sagging, its tip turned up a little;
    foliage along it, and twigs hanging from it in a fringe. A bare one
    (broken, or on a dead tree) is wood only."""
    s = np.linspace(0.0, 1.0, 14)
    out = np.array([math.cos(azimuth), 0.0, math.sin(azimuth)])
    sag = (
        -droop * length * s**1.5
        + 0.18 * droop * length * np.maximum(s - 0.75, 0) ** 2 * 16
    )
    wobble = rng.normal(0.0, 0.03 * length, (len(s), 3)) * s[:, None]
    P = root + out * length * s[:, None] + np.outer(sag, [0.0, 1.0, 0.0]) + wobble
    thick = (0.22 + 0.12 * rng.random()) * (
        0.45 + 0.55 * np.sin(np.pi * np.clip(s * 1.25, 0, 1))
    )
    thick *= 1.0 - 0.45 * s
    thick = np.maximum(thick, 0.06) * (0.6 + 0.4 * min(length / 3.0, 1.0))
    if bare:
        wood = np.linspace(0.05, 0.015, len(s)) * min(length / 1.5, 1.4)
        return P, [curve(P, wood, TRUNK, 0.12)]
    parts = [curve(P, thick, BOUGH, 0.05)]
    for sj in np.arange(0.18, 0.98, 0.11 + 0.05 * rng.random()):
        if rng.random() < 0.25:
            continue
        at = np.array([np.interp(sj, s, P[:, k]) for k in range(3)])
        drop = (0.25 + 0.45 * rng.random()) * (1.0 - 0.4 * sj) * min(length / 2.0, 1.2)
        tip = (
            at
            + out * 0.12 * drop
            + np.array([rng.normal(0, 0.05), -drop, rng.normal(0, 0.05)])
        )
        twig = np.array([at, 0.5 * (at + tip) + out * 0.04, tip])
        parts.append(curve(twig, np.array([0.11, 0.08, 0.05]), BOUGH, 0.05))
    return P, parts


def pine(tree: Pine) -> tuple[Stamps, list[np.ndarray]]:
    """A conifer grown from its trunk: whorls of boughs from the crown's base
    up, shorter and drooping less as they go up, some missing; a few dead
    stubs below the crown. Returns its stamps and the centre lines of its
    trunk and boughs."""
    rng = np.random.default_rng(SEED + 31 * tree.seed)
    axis, h = trunk_axis(tree, rng)
    H = tree.height
    radius = 0.03 + 0.24 * (1 - h / H) ** 1.2 * (H / 22)
    wood = 0.13 if tree.dead else 0.07
    parts = [curve(axis, radius, TRUNK, wood, 0.05)]
    lines = [axis]
    crown = rng.uniform(0.14, 0.26) * H
    hk = crown
    while hk < H - 0.4:
        up = (hk - crown) / (H - crown)
        reach = (
            0.19 * H * tree.spread * (1.0 - up) ** 0.95 * (0.3 + 0.7 * (1 - up) ** 0.3)
        )
        root = np.array([np.interp(hk, h, axis[:, k]) for k in range(3)])
        for azimuth in rng.uniform(0, 2 * np.pi, rng.integers(3, 7)):
            line, bits = whorl_bough(tree, root, reach, up, azimuth, rng)
            parts += bits
            if line is not None:
                lines.append(line)
        # (whorls come unevenly: now close, now a gap of a poor year)
        step = rng.uniform(0.3, 0.85) * (2.0 if rng.random() < 0.12 else 1.0)
        hk += step * (1.0 - 0.3 * up)
    parts.append(dead_stubs(axis, h, crown, rng))
    if not tree.dead:
        tip = axis[-1] + np.array([0.0, 0.9, 0.0])
        parts.append(
            curve(np.array([axis[-1], tip]), np.array([0.09, 0.03]), BOUGH, 0.05)
        )
    return merge_stamps(parts), lines


def trunk_axis(tree: Pine, rng) -> tuple[np.ndarray, np.ndarray]:
    """The trunk's centre line: leaning, and swaying as it grew, its crown
    bent over a little."""
    y0 = ground_at(tree.x, tree.z) - 0.3
    H = tree.height
    h = np.linspace(0.0, H, 60)
    sway = rng.uniform(0.15, 0.5) * np.sin(h / H * rng.uniform(2.0, 3.5) + tree.seed)
    bend = rng.normal(0.0, 0.6) * (h / H) ** 3
    x = tree.x + tree.lean * h + 0.4 * sway + bend
    return np.stack([x, y0 + h, tree.z + 0.3 * sway], 1), h


def whorl_bough(tree: Pine, root, reach, up, azimuth, rng):
    """One bough of a whorl, or none: now and then one is missing, one
    reaches far, one is broken off short. A dead tree has only broken
    ones."""
    luck = rng.random()
    if luck < 0.16:
        return None, []
    length = max(reach * rng.uniform(0.55, 1.15), 0.35)
    droop = (0.18 + 0.32 * (1 - up)) * rng.uniform(0.6, 1.3)
    broken = tree.dead or luck > 0.93
    if broken:
        length *= rng.uniform(0.15, 0.45)
    elif luck > 0.84:
        length *= 1.45
    return bough(root, length, azimuth, droop * (0.4 if broken else 1.0), rng, broken)


def dead_stubs(axis, h, crown, rng) -> Stamps:
    parts = []
    for hk in np.sort(rng.uniform(0.25 * crown, crown, 5)):
        root = np.array([np.interp(hk, h, axis[:, k]) for k in range(3)])
        a = rng.uniform(0, 2 * np.pi)
        out = np.array([math.cos(a), -0.25, math.sin(a)]) * rng.uniform(0.4, 1.0)
        parts.append(
            curve(np.array([root, root + out]), np.array([0.04, 0.015]), TRUNK, 0.08)
        )
    return merge_stamps(parts)


def pines() -> tuple[Stamps, list[np.ndarray]]:
    stamps, lines = [], []
    for tree in PINES:
        s, line = pine(tree)
        stamps.append(s)
        lines += line
    return merge_stamps(stamps), lines


def reeds(rng) -> tuple[Stamps, list[np.ndarray]]:
    """Clumps of reeds in the shallows: blades from a root, leaning and
    bending, some broken over."""
    clumps = (
        (-4.4, 13.0, 34),
        (-5.4, 16.8, 24),
        (-7.4, 26.0, 22),
        (-3.45, 10.5, 18),
        (5.9, 13.65, 30),
        (1.05, 8.95, 16),
    )
    parts = []
    for cx, cz, n in clumps:
        for _ in range(n):
            x, z = cx + rng.normal(0, 0.3), cz + rng.normal(0, 0.22)
            tall = rng.uniform(0.2, 0.75) * (1.3 if cz > 15 else 1.0)
            lean = rng.normal(0.0, 0.18, 2)
            s = np.linspace(0, 1, 8)[:, None]
            bend = s**2
            P = np.concatenate(
                [x + lean[0] * tall * bend, -0.1 + tall * s, z + lean[1] * tall * bend],
                1,
            )
            parts.append(curve(P, np.linspace(0.016, 0.004, 8), REED, 0.2, 0.01))
    return merge_stamps(parts), [p.points for p in parts]


# ----------------------------------------------------------------------
# the boat

#: Where the boat lies: its stern out on the water, its bow on the shingle.
STERN, BOW = np.array([0.85, 14.05]), np.array([3.95, 11.45])
#: It lies over on its side a little, towards the eye.
HEEL = math.radians(7.0)
LENGTH = float(np.linalg.norm(BOW - STERN))
STRAKES = 5


def half_beam(s):
    t = np.where(s < 0.42, (s - 0.42) / 0.66, (s - 0.42) / 0.58)
    return 0.66 * np.clip(1.0 - t * t, 0.0, 1.0)


def sheer(s):
    return 0.6 - 0.07 * s + 0.24 * s**3


def keel(s):
    return (
        0.12 * np.clip((0.22 - s) / 0.22, 0, 1) ** 2
        + 0.5 * np.clip((s - 0.72) / 0.28, 0, 1) ** 2
    )


def hull_point(s, v, side):
    """A point of the hull's outside, in the boat's frame (x forward from
    the transom, y to port, z up from the keel): ``s`` along it (0 transom,
    1 stem) and ``v`` round its girth (0 keel, 1 sheer). Each strake laps
    over the one below it, as a clinker hull's do."""
    theta = v * np.pi / 2
    lap = 0.014 * (1.0 - (v * STRAKES) % 1.0)
    y = (half_beam(s) * np.sin(theta) ** 0.62 + lap * np.sin(theta)) * side
    z = keel(s) + (sheer(s) - keel(s)) * (1.0 - np.cos(theta)) ** 0.9
    return np.stack(np.broadcast_arrays(s * LENGTH, y, z), axis=-1)


def width_at(s: float, z: float) -> float:
    """Half the hull's width inside at station ``s`` and height ``z``."""
    v = np.linspace(0.0, 1.0, 200)
    p = hull_point(np.full(200, s), v, 1.0)
    return float(np.interp(z, p[:, 2], p[:, 1]))


def boat_frame():
    """Rotation and offset from the boat's frame to the world: yawed so its
    bow points at BOW, pitched a little bow up where it rests on the shingle,
    sunk to its waterline."""
    d = (BOW - STERN) / LENGTH
    fwd = np.array([d[0], -0.03, d[1]])
    fwd /= np.linalg.norm(fwd)
    port = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    port /= np.linalg.norm(port)
    up = np.cross(fwd, port)
    port, up = (
        math.cos(HEEL) * port - math.sin(HEEL) * up,
        math.cos(HEEL) * up + math.sin(HEEL) * port,
    )
    R = np.stack([fwd, port, up], axis=0)  # rows: frame axes in the world
    origin = np.array([STERN[0], -0.03, STERN[1]])
    return R, origin


def to_world(p: np.ndarray) -> np.ndarray:
    R, origin = boat_frame()
    return p @ R + origin


def hull() -> Mesh:
    ns, nv = 70, 61
    s = np.linspace(0.0, 1.0, ns)[:, None]
    v = np.linspace(0.0, 1.0, nv)[None, :]
    meshes = []
    for side in (1.0, -1.0):
        P = hull_point(s, v, side).reshape(-1, 3)
        faces = grid_faces(ns, nv)
        n = vertex_normals(P, faces)
        if (n[:, 1] * side).mean() < 0:  # (outward: to port on the port side)
            faces = faces[:, [0, 2, 1]]
        m = make_mesh(to_world(P), faces, HULL, 0.42, inside=(INSIDE, 0.16))
        m.grain = np.tile(
            to_world(np.array([[1.0, 0, 0]])) - to_world(np.zeros((1, 3))),
            (len(faces), 1),
        )
        meshes.append(m)
    meshes.append(transom())
    return merge_meshes(meshes)


def transom() -> Mesh:
    v = np.linspace(0.0, 1.0, 40)
    rim = np.concatenate([hull_point(0.0, v[::-1], 1.0), hull_point(0.0, v[1:], -1.0)])
    centre = np.array([[0.0, 0.0, 0.5 * (sheer(0.0) + keel(0.0)) + 0.1]])
    verts = np.concatenate([centre, rim])
    faces = np.array([[0, k + 1, k + 2] for k in range(len(rim) - 1)])
    m = make_mesh(to_world(verts), faces, HULL, 0.36, inside=(INSIDE, 0.16))
    m.grain = np.tile(
        to_world(np.array([[0, 1.0, 0]])) - to_world(np.zeros((1, 3))), (len(faces), 1)
    )
    return m


def box(centre, axes, half, ident, albedo) -> Mesh:
    """A box in the boat's frame: its centre, its three axes (rows) and half
    sizes along them."""
    corners = np.array(
        [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float
    )
    P = centre + (corners * half) @ axes
    quads = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    faces = []
    for a, b, c, d in quads:
        faces += [(a, b, c), (a, c, d)]
    W = to_world(P)
    f = np.array(faces)
    # outward normals per face, flat
    centre_w = W.mean(axis=0)
    verts = W[f].reshape(-1, 3)
    tri = np.arange(len(verts)).reshape(-1, 3)
    a, b, c = (verts[tri[:, k]] for k in range(3))
    fn = np.cross(b - a, c - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    outward = np.sign(np.einsum("ij,ij->i", fn, (a + b + c) / 3 - centre_w))[:, None]
    fn *= outward
    m = make_mesh(verts, tri, ident, albedo, normals=np.repeat(fn, 3, axis=0))
    m.grain = np.tile(to_world(axes[:1]) - to_world(np.zeros((1, 3))), (len(tri), 1))
    return m


def fittings() -> Mesh:
    """Gunwales along the sheer, the stem, two thwarts and a stern seat."""
    parts = []
    for side in (1.0, -1.0):
        for s0 in np.linspace(0.0, 0.985, 30):
            s1 = s0 + 1.0 / 30
            a = hull_point(s0, 1.0, side)
            b = hull_point(min(s1, 1.0), 1.0, side)
            c = 0.5 * (a + b) + np.array([0.0, -0.022 * side, 0.025])
            along = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            across = np.cross(np.array([0, 0, 1.0]), along)
            up = np.cross(along, across)
            axes = np.stack([along, across, up])
            parts.append(
                box(
                    c,
                    axes,
                    np.array([np.linalg.norm(b - a) / 2 + 0.03, 0.03, 0.035]),
                    GUNWALE,
                    0.5,
                )
            )
    I3 = np.eye(3)
    for s, wide in ((0.36, 0.22), (0.66, 0.2), (0.07, 0.3)):
        z = sheer(s) - 0.2
        half = width_at(s, z) - 0.025
        parts.append(
            box(
                np.array([s * LENGTH, 0.0, z]),
                I3,
                np.array([wide / 2, half, 0.02]),
                THWART,
                0.45,
            )
        )
    stem = np.array([LENGTH + 0.02, 0.0, 0.5 * (keel(1.0) + sheer(1.0)) + 0.04])
    parts.append(
        box(
            stem,
            I3,
            np.array([0.03, 0.03, 0.5 * (sheer(1.0) - keel(1.0)) + 0.05]),
            GUNWALE,
            0.4,
        )
    )
    return merge_meshes(parts)


def rope_and_stake() -> tuple[Stamps, list[np.ndarray]]:
    """The painter: from the stem head, down over the shingle, to an iron
    stake driven in near the water."""
    head = to_world(np.array([[LENGTH + 0.02, 0.0, sheer(1.0) + 0.04]]))[0]
    stake = np.array([4.35, 0.0, 10.05])
    top = stake + np.array([0.0, 0.55, 0.0])
    lie = 0.5 * (head + top) + np.array([0.15, 0.0, -0.15])
    lie[1] = ground_at(lie[0], lie[2]) + 0.03
    hang = 0.5 * (head + lie)
    hang[1] = max(ground_at(hang[0], hang[2]) + 0.03, lie[1] - 0.05)
    rest = 0.5 * (lie + top)
    rest[1] = ground_at(rest[0], rest[2]) + 0.03
    t = np.linspace(0.0, 1.0, 60)[:, None]
    falls = (1 - t) ** 2 * head + 2 * t * (1 - t) * hang + t**2 * lie
    rises = (1 - t) ** 2 * lie + 2 * t * (1 - t) * rest + t**2 * top
    path = np.concatenate([falls, rises[1:]])
    rope = curve(path, np.full(len(path), 0.016), ROPE, 0.55, 0.008)
    post = np.array([[stake[0], -0.1, stake[2]], top + np.array([0, 0.06, 0])])
    stake_s = curve(post, np.array([0.03, 0.025]), STAKE, 0.12, 0.01)
    return merge_stamps([rope, stake_s]), [path, post]


def boat_lines() -> list[np.ndarray]:
    """The lines a draughtsman would draw the boat by, in the world: each
    sheer, the lands of the strakes, the stem and the transom's edge."""
    s = np.linspace(0.0, 1.0, 120)
    lines = []
    for side in (1.0, -1.0):
        lines.append(to_world(hull_point(s, np.ones_like(s), side) + [0, 0, 0.06]))
        lines += [
            to_world(hull_point(s, np.full_like(s, k / STRAKES), side))
            for k in range(1, STRAKES)
        ]
    v = np.linspace(0.0, 1.0, 40)
    rim = np.concatenate([hull_point(0.0, v[::-1], 1.0), hull_point(0.0, v[1:], -1.0)])
    lines.append(to_world(rim))
    stem = np.stack(
        [
            np.full(20, LENGTH + 0.03),
            np.zeros(20),
            np.linspace(keel(1.0), sheer(1.0) + 0.09, 20),
        ],
        1,
    )
    lines.append(to_world(stem))
    return lines


def boat() -> tuple[Mesh, Stamps, list[np.ndarray]]:
    rope, lines = rope_and_stake()
    return merge_meshes([hull(), fittings()]), rope, lines


# ----------------------------------------------------------------------
# seeing it: a z-buffer for each eye


@njit(cache=True, error_model="numpy")
def _raster(
    xy, iz, nrm, ids, alb, ang, cam, zbuf, ident, albedo, normal, angle
):  # pragma: no cover
    """Fill triangles (screen corners ``xy``, 1/Z at each ``iz``, a normal
    at each corner) into the buffers, nearest first. What lies under the
    water is not seen; the side of a face turned from the eye shows its
    inside (``ids``/``alb`` second column), or nothing if that id is < 0."""
    h, w = zbuf.shape
    for k in range(xy.shape[0]):
        x0, y0 = xy[k, 0, 0], xy[k, 0, 1]
        x1, y1 = xy[k, 1, 0], xy[k, 1, 1]
        x2, y2 = xy[k, 2, 0], xy[k, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        lo_x = max(int(math.floor(min(x0, x1, x2))), 0)
        hi_x = min(int(math.ceil(max(x0, x1, x2))), w - 1)
        lo_y = max(int(math.floor(min(y0, y1, y2))), 0)
        hi_y = min(int(math.ceil(max(y0, y1, y2))), h - 1)
        for py in range(lo_y, hi_y + 1):
            for px in range(lo_x, hi_x + 1):
                b0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
                b1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
                b2 = 1.0 - b0 - b1
                if b0 < 0.0 or b1 < 0.0 or b2 < 0.0:
                    continue
                inv = b0 * iz[k, 0] + b1 * iz[k, 1] + b2 * iz[k, 2]
                z = 1.0 / inv
                if z >= zbuf[py, px] or cam - (py - CY) * z / FOCAL < 0.0:
                    continue
                n = np.zeros(3)
                for c in range(3):
                    n[c] = (
                        b0 * nrm[k, 0, c] * iz[k, 0]
                        + b1 * nrm[k, 1, c] * iz[k, 1]
                        + b2 * nrm[k, 2, c] * iz[k, 2]
                    ) * z
                facing = n[0] * (px - CX) / FOCAL - n[1] * (py - CY) / FOCAL + n[2]
                side = 0
                if facing > 0.0:
                    side = 1
                    n = -n
                if ids[k, side] < 0:
                    continue
                zbuf[py, px] = z
                ident[py, px] = ids[k, side]
                albedo[py, px] = alb[k, side]
                norm = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) + 1e-12
                for c in range(3):
                    normal[py, px, c] = n[c] / norm
                angle[py, px] = ang[k]


@njit(cache=True, error_model="numpy")
def _stamp(
    px, py, pr, pz, ids, alb, ang, cam, zbuf, ident, albedo, normal, angle
):  # pragma: no cover
    """Fill discs into the buffers: each a slice of a round rod lying along
    the direction ``ang`` on the sheet, so lit as a rod."""
    h, w = zbuf.shape
    for k in range(px.shape[0]):
        r = max(pr[k], 0.6)
        ca, sa = math.cos(ang[k]), math.sin(ang[k])
        lo_x, hi_x = max(int(px[k] - r), 0), min(int(px[k] + r) + 1, w - 1)
        lo_y, hi_y = max(int(py[k] - r), 0), min(int(py[k] + r) + 1, h - 1)
        z = pz[k]
        for y in range(lo_y, hi_y + 1):
            if cam - (y - CY) * z / FOCAL < 0.0:
                continue
            for x in range(lo_x, hi_x + 1):
                dx, dy = (x - px[k]) / r, (y - py[k]) / r
                q = dx * dx + dy * dy
                if q > 1.0 or z >= zbuf[y, x]:
                    continue
                across = -dx * sa + dy * ca
                zbuf[y, x] = z
                ident[y, x] = ids[k]
                albedo[y, x] = alb[k]
                normal[y, x, 0] = -across * sa
                normal[y, x, 1] = -across * ca
                normal[y, x, 2] = -math.sqrt(max(1.0 - across * across, 0.0))
                angle[y, x] = ang[k]


@njit(cache=True, error_model="numpy")
def _band_depth(dx, dy, norm, cam, za, zb, band, tops, extent):  # pragma: no cover
    """Optical depth of one band of mist along a ray between za and zb."""
    z0, z1, dens, soft, fade = band[0], band[1], band[2], band[3], band[4]
    a, b = max(za, z0), min(zb, z1)
    if b <= a:
        return 0.0
    gh, gw = tops.shape[1], tops.shape[2]
    n = 24
    step = (b - a) / n
    tau = 0.0
    for i in range(n):
        z = a + (i + 0.5) * step
        fx = (dx * z - extent[0]) / (extent[2] - extent[0]) * (gw - 1)
        fz = (z - extent[1]) / (extent[3] - extent[1]) * (gh - 1)
        fx = min(max(fx, 0.0), gw - 1.001)
        fz = min(max(fz, 0.0), gh - 1.001)
        ix, iz = int(fx), int(fz)
        tx, tz = fx - ix, fz - iz
        top = (tops[0, iz, ix] * (1 - tx) + tops[0, iz, ix + 1] * tx) * (1 - tz) + (
            tops[0, iz + 1, ix] * (1 - tx) + tops[0, iz + 1, ix + 1] * tx
        ) * tz
        thick = (tops[1, iz, ix] * (1 - tx) + tops[1, iz, ix + 1] * tx) * (1 - tz) + (
            tops[1, iz + 1, ix] * (1 - tx) + tops[1, iz + 1, ix + 1] * tx
        ) * tz
        y = cam + dy * z
        t = min(max((top + soft - y) / (2.0 * soft), 0.0), 1.0)
        edge = min(max(min(z - z0, z1 - z) / fade, 0.0), 1.0)
        tau += dens * thick * t * t * (3.0 - 2.0 * t) * edge * edge * (3.0 - 2.0 * edge)
    return tau * step * norm


@njit(cache=True, error_model="numpy")
def _fog(zbuf, cam, mirrored, bands, tops, extent, air, out):  # pragma: no cover
    """Optical depth of the air and the mist between the eye and what each
    pixel shows. For the mirrored eye, from where the ray leaves the water."""
    h, w = zbuf.shape
    for py in range(h):
        dy = -(py - CY) / FOCAL
        for px in range(w):
            dx = (px - CX) / FOCAL
            norm = math.sqrt(dx * dx + dy * dy + 1.0)
            z0 = 0.0
            if mirrored:
                if dy <= 0.0:
                    out[py, px] = 0.0
                    continue
                z0 = -cam / dy
            z1 = min(zbuf[py, px], FAR)
            if z1 <= z0:
                out[py, px] = 0.0
                continue
            y0, y1 = cam + dy * z0, cam + dy * z1
            if abs(dy) < 1e-7:
                tau = air[0] * math.exp(-y0 / air[1]) * (z1 - z0) * norm
            else:
                tau = (
                    air[0]
                    * air[1]
                    / dy
                    * norm
                    * (math.exp(-y0 / air[1]) - math.exp(-y1 / air[1]))
                )
            for b in range(bands.shape[0]):
                tau += _band_depth(dx, dy, norm, cam, z0, z1, bands[b], tops[b], extent)
            out[py, px] = tau


@dataclass
class View:
    """What one eye sees: per pixel the distance along the axis (inf for
    nothing), the id, albedo and normal of the surface, and the direction
    on the sheet of the strokes that would follow it (radians)."""

    z: np.ndarray
    ident: np.ndarray
    albedo: np.ndarray
    normal: np.ndarray
    angle: np.ndarray


def screen_angle(points: np.ndarray, along: np.ndarray, cam: float) -> np.ndarray:
    """The direction on the sheet of a world vector at world points."""
    a = project(points, cam)
    b = project(
        points
        + 0.05
        * along
        / np.maximum(np.linalg.norm(along, axis=-1, keepdims=True), 1e-9),
        cam,
    )
    return np.arctan2(b[..., 1] - a[..., 1], b[..., 0] - a[..., 0])


def project(P: np.ndarray, cam: float) -> np.ndarray:
    """World points to the sheet: (x, y, Z) for an eye ``cam`` above the
    water (below it, for the mirrored eye)."""
    Z = np.maximum(P[..., 2], 1e-6)
    return np.stack(
        [CX + FOCAL * P[..., 0] / Z, CY - FOCAL * (P[..., 1] - cam) / Z, Z], axis=-1
    )


def raster_mesh(view: View, mesh: Mesh, cam: float) -> None:
    keep = (mesh.verts[mesh.faces, 2] > NEAR).all(axis=1)
    faces = mesh.faces[keep]
    S = project(mesh.verts, cam)[faces]
    centre = mesh.verts[faces].mean(axis=1)
    ang = screen_angle(centre, mesh.grain[keep], cam)
    _raster(
        np.ascontiguousarray(S[..., :2]),
        np.ascontiguousarray(1.0 / S[..., 2]),
        np.ascontiguousarray(mesh.normals[faces]),
        np.ascontiguousarray(mesh.ids[keep]),
        np.ascontiguousarray(mesh.albedo[keep]),
        ang,
        cam,
        view.z,
        view.ident,
        view.albedo,
        view.normal,
        view.angle,
    )


def stamp(view: View, stamps: Stamps, cam: float) -> None:
    S = project(stamps.points, cam)
    ang = screen_angle(stamps.points, stamps.grain, cam)
    _stamp(
        S[:, 0].copy(),
        S[:, 1].copy(),
        FOCAL * stamps.radii / S[:, 2],
        S[:, 2].copy(),
        stamps.ids.astype(np.int64),
        stamps.albedo.astype(np.float64),
        ang,
        cam,
        view.z,
        view.ident,
        view.albedo,
        view.normal,
        view.angle,
    )


@dataclass
class World:
    """Everything in the scene: meshes, stamps, and the centre lines of the
    pines' trunks and boughs and the lines of the boat (world polylines)."""

    meshes: list[Mesh]
    stamps: list[Stamps]
    pine_lines: list[np.ndarray]
    boat_lines: list[np.ndarray]
    reed_lines: list[np.ndarray]
    rope_lines: list[np.ndarray]
    trees: np.ndarray


def build_world() -> World:
    rng = np.random.default_rng(SEED)
    meshes = [ridge_mesh(r, rng) for r in RIDGES]
    hill_ground, forest_trees, tree_spec = hills(rng)
    meshes += [hill_ground, forest_trees]
    meshes.append(ground(rng))
    meshes += [rock(*spec, rng) for spec in ROCKS]
    hull_mesh, rope, rope_lines = boat()
    meshes.append(hull_mesh)
    trees, lines = pines()
    reed_stamps, reed_lines = reeds(rng)
    return World(
        meshes,
        [trees, reed_stamps, rope],
        lines,
        boat_lines(),
        reed_lines,
        rope_lines,
        tree_spec,
    )


def look(world: World, cam: float) -> View:
    view = View(
        np.full((HEIGHT, WIDTH), np.inf),
        np.full((HEIGHT, WIDTH), -1, np.int64),
        np.zeros((HEIGHT, WIDTH)),
        np.zeros((HEIGHT, WIDTH, 3)),
        np.zeros((HEIGHT, WIDTH)),
    )
    for mesh in world.meshes:
        raster_mesh(view, mesh, cam)
    for s in world.stamps:
        stamp(view, s, cam)
    return view


# ----------------------------------------------------------------------
# light, air and mist


def directions() -> np.ndarray:
    """The unit direction each pixel looks along, (H, W, 3)."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    d = np.stack([(x - CX) / FOCAL, -(y - CY) / FOCAL, np.ones_like(x)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def glow(d: np.ndarray) -> np.ndarray:
    """How near the sun's glow a direction looks, 0..1."""
    sun = np.array([SUN[0], -0.03, SUN[2]])
    sun /= np.linalg.norm(sun)
    near = np.arccos(np.clip(d @ sun, -1.0, 1.0))
    return np.exp(-near / 0.28)


def mist_light(d: np.ndarray) -> np.ndarray:
    """What the mist and the air send to the eye: nearly white, whitest
    towards the glow."""
    return 0.9 + 0.1 * glow(d)


def light(n: np.ndarray) -> np.ndarray:
    """The light a surface of normal ``n`` gets: the sky above it, and the
    glow behind the ridges, which only what faces away from the eye sees."""
    rim = np.clip((n @ SUN + 0.3) / 0.6, 0.0, 1.0) ** 2
    return 0.4 + 0.4 * n[..., 1] + 0.45 * rim


#: The thin clouds: they lie this high, over this part of the world.
CLOUD_HEIGHT = 2600.0
CLOUD_EXTENT = (-45000.0, 4000.0, 45000.0, 90000.0)


def cloud_map(rng) -> np.ndarray:
    streaks = noise.aniso_noise(320, 420, 30.0, 2.5, 3.0, rng, octaves=3)
    patch = world_map(rng, (320, 420), 25.0, 2)
    return filters.smoothstep(0.6, 0.7, streaks) * filters.smoothstep(0.5, 0.66, patch)


def sky_light(d: np.ndarray, cam: float, clouds: np.ndarray):
    """The sky: lightest low down and towards the glow, a few thin clouds
    lit from below. Returns its light, the clouds' cover, and the light of
    the sky alone."""
    elev = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    sky = 0.5 + 0.3 * np.exp(-np.maximum(elev, 0.0) / 0.11) + 0.12 * glow(d)
    up = np.maximum(d[..., 1], 1e-4)
    t = (CLOUD_HEIGHT - cam) / up
    X, Z = d[..., 0] * t, d[..., 2] * t
    cover = sample_grid(clouds, CLOUD_EXTENT, X, Z)
    cover *= filters.smoothstep(0.05, 0.08, d[..., 1]) * (
        1 - filters.smoothstep(0.14, 0.19, d[..., 1])
    )
    lit = sky * (1.0 - 0.75 * cover) + (0.98 + 0.02 * glow(d)) * 0.75 * cover
    return lit, cover, sky


#: The mist, in bands across the view: nearest and furthest distance, how
#: dense, how soft its top (metres), how its density fades at its two ends,
#: and the height of its top, low and high (it varies across the world).
MIST = (
    (30.0, 260.0, 0.004, 1.2, 25.0, 0.6, 3.5),
    (240.0, 1450.0, 0.0028, 5.0, 220.0, 3.0, 13.0),
    (2500.0, 4000.0, 0.0011, 40.0, 450.0, 50.0, 190.0),
    (4900.0, 7300.0, 0.0006, 80.0, 700.0, 150.0, 420.0),
    (9200.0, 13500.0, 0.0004, 150.0, 1200.0, 350.0, 900.0),
)
MIST_EXTENT = (-9000.0, 0.0, 9000.0, 16000.0)
#: The air: density at the water, and the height over which it thins.
AIR = (1.0 / 8500.0, 1200.0)


def mist_tops(rng) -> np.ndarray:
    """For each band, over the world: the height of its top, and how thick
    it lies there (it gathers in some places and thins in others)."""
    out = []
    for band, cell in zip(MIST, (1.5, 3.0, 9.0, 12.0, 14.0), strict=True):
        top = band[5] + (band[6] - band[5]) * world_map(rng, (160, 180), cell, 3)
        thick = 0.12 + 0.88 * filters.smoothstep(
            0.3, 0.68, world_map(rng, (160, 180), cell * 1.4, 3)
        )
        out.append(np.stack([top, thick]))
    return np.stack(out)


def veil(view: View, cam: float, mirrored: bool, tops: np.ndarray) -> np.ndarray:
    """Transmittance of the air and the mist to each pixel's surface."""
    tau = np.zeros((HEIGHT, WIDTH))
    bands = np.array([b[:5] for b in MIST])
    _fog(view.z, cam, mirrored, bands, tops, np.array(MIST_EXTENT), np.array(AIR), tau)
    return np.exp(-tau)


def surface_light(view: View) -> np.ndarray:
    L = view.albedo * light(view.normal)
    return np.where(view.ident == INSIDE, 0.45 * L, L)


# ----------------------------------------------------------------------
# the study: the scene as tones, and the maps the pencil works from


@dataclass
class Study:
    """The scene seen: its light (linear, the brightest mist about 1), the
    grey a pencil should reach at each pixel (0..255, from the paper's to
    the darkest graphite's), what each pixel shows (``ident``), what the
    water mirrors there (``mirrored``: the id of the thing reflected, else
    the same as ``ident``), how far it is (metres), how much air and mist
    veil it (0..1), how much the water is ruffled (0..1), the direction
    strokes should follow (radians on the sheet), how much cloud covers the
    sky, and the grey of the sky alone, without its clouds (0 where
    something stands in front of it), and how much each surface faces up
    (the y part of its normal)."""

    light: np.ndarray
    grey: np.ndarray
    ident: np.ndarray
    mirrored: np.ndarray
    depth: np.ndarray
    veil: np.ndarray
    ruffle: np.ndarray
    angle: np.ndarray
    cloud: np.ndarray
    sky: np.ndarray
    up: np.ndarray

    @property
    def tone(self) -> np.ndarray:
        """How dark the pencil should go: 0 the paper, 1 the darkest."""
        return np.clip((PAPER - self.grey) / (PAPER - DARKEST), 0.0, 1.0)

    @property
    def layer(self) -> np.ndarray:
        return LAYER[self.ident]

    @property
    def softness(self) -> np.ndarray:
        """How soft an edge there should be: lost in the air and the mist
        far off, found near the eye."""
        far = np.clip(np.log10(np.maximum(self.depth, 1.0) / 8.0) / 3.2, 0.0, 1.0)
        return np.clip(0.6 * self.veil + 0.6 * far, 0.0, 1.0)


def mark_sky(view: View, cam: float, clouds):
    """Fill what the eye sees of the sky; returns its light, the share of
    each pixel covered by cloud, and the light of the sky alone."""
    d = directions()
    sky = view.ident < 0
    L, cover, clear = sky_light(d, cam, clouds)
    view.ident[sky & (cover > 0.3)] = CLOUD
    view.ident[sky & (cover <= 0.3)] = SKY
    view.z[sky] = FAR
    view.angle[sky] = 0.0
    return np.where(sky, L, surface_light(view)), np.where(sky, cover, 0.0), clear


def mirror_image(world: World, clouds, tops) -> tuple[np.ndarray, np.ndarray]:
    """What the still water mirrors, as the eye mirrored under the water
    sees it, upside down; and the id of what it sees."""
    view = look(world, -EYE)
    rows = np.arange(HEIGHT)[:, None]
    view.ident[(view.ident < 0) & (rows >= CY)] = WATER  # (never used)
    L, _, _ = mark_sky(view, -EYE, clouds)
    T = veil(view, -EYE, True, tops)
    L = T * L + (1.0 - T) * mist_light(directions())
    # the faint unrest of the surface softens what it mirrors
    return filters.box_blur(L, 1, axis=0), view.ident


#: Where a breath of wind ruffles the water: a few streaks, each its
#: distance, where it runs across (X from, to) and how deep it is (metres).
RUFFLES = (
    (46.0, 3.0, 15.0, 1.3),
    (96.0, -14.0, 30.0, 2.2),
    (128.0, 18.0, 70.0, 3.0),
    (205.0, -40.0, 15.0, 5.0),
    (330.0, 30.0, 150.0, 7.0),
    (560.0, -160.0, -40.0, 12.0),
)


def ruffles(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """How ruffled the water is at (X, Z): a few streaks, soft at their
    ends."""
    out = np.zeros_like(Z)
    for z, x0, x1, deep in RUFFLES:
        across = np.exp(-(((Z - z) / deep) ** 2))
        mid, half = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
        along = 1.0 - filters.smoothstep(0.55, 1.0, np.abs(X - mid) / half)
        out = np.maximum(out, across * along)
    return out


def water_light(study_z, mirror, mirror_id, rng):
    """The water: what it mirrors (a little broken by ripples near the eye),
    weakened where the eye looks steeply into it, and the light of the
    ruffled streaks. Returns its light, the mirrored ids and the ruffle."""
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    below = np.maximum(rows - CY, 0.0)
    ripple = noise.aniso_noise(HEIGHT, WIDTH, 50.0, 1.6, 0.0, rng, octaves=2) - 0.5
    shift = (0.5 + 7.0 * (below / (HEIGHT - CY)) ** 1.6) * ripple
    src = np.clip(2.0 * CY - rows + shift, 0.0, CY - 0.001)
    seen = filters.bilinear(mirror, src, cols)
    seen_id = mirror_id[np.round(src).astype(int), cols.astype(int)]
    d = directions()
    cos_i = np.maximum(-d[..., 1], 0.0)
    fresnel = 0.02 + 0.98 * (1.0 - cos_i) ** 5
    reflect = np.maximum(fresnel, 0.42)  # (a little sky light from the mist too)
    L = reflect * seen + (1.0 - reflect) * 0.08
    X = d[..., 0] / np.maximum(d[..., 2], 1e-9) * study_z
    ruff = ruffles(study_z, X)
    return L * (1.0 - 0.8 * ruff) + 0.95 * 0.8 * ruff, seen_id, ruff


def to_grey(L: np.ndarray) -> np.ndarray:
    """Light to the grey of the pencil's range."""
    v = np.where(
        L <= 0.0031308, 12.92 * L, 1.055 * np.maximum(L, 0) ** (1 / 2.4) - 0.055
    )
    lo, hi = 0.21, 0.975
    return DARKEST + (PAPER - DARKEST) * np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def study(world: World) -> Study:
    rng = np.random.default_rng(SEED + 1)
    clouds, tops = cloud_map(rng), mist_tops(rng)
    view = look(world, EYE)
    rows = np.arange(HEIGHT)[:, None] * np.ones((1, WIDTH))
    water = (view.ident < 0) & (rows > CY)
    view.ident[water] = WATER
    view.z[water] = FOCAL * EYE / (rows[water] - CY)
    view.angle[water] = np.pi / 2
    L, cover, clear = mark_sky(view, EYE, clouds)
    mirror, mirror_id = mirror_image(world, clouds, tops)
    wL, seen_id, ruff = water_light(view.z, mirror, mirror_id, rng)
    L = np.where(water, wL, L)
    T = veil(view, EYE, False, tops)
    M = mist_light(directions())
    L = T * L + (1.0 - T) * M
    sky_alone = np.where(view.z >= FAR, to_grey(T * clear + (1.0 - T) * M), 0.0)
    return Study(
        L,
        to_grey(L),
        view.ident,
        np.where(water, seen_id, view.ident),
        view.z,
        1.0 - T,
        np.where(water, ruff, 0.0),
        view.angle,
        cover,
        sky_alone,
        view.normal[..., 1].copy(),
    )


def seen_runs(line: np.ndarray, st: Study, slack: float = 0.04) -> list[np.ndarray]:
    """The parts of a world polyline the eye sees, as polylines on the
    sheet: where nothing nearer stands in front of it."""
    P = project(line, EYE)
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, WIDTH - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, HEIGHT - 1)
    inside = (P[:, 0] >= 0) & (P[:, 0] < WIDTH) & (P[:, 1] >= 0) & (P[:, 1] < HEIGHT)
    seen = (
        inside & (P[:, 2] <= st.depth[yi, xi] * (1.0 + slack) + 0.05) & (line[:, 1] > 0)
    )
    runs = np.split(np.arange(len(P)), np.nonzero(np.diff(seen.astype(int)))[0] + 1)
    return [P[r, :2] for r in runs if seen[r[0]] and len(r) > 1]


def ridge_lines(st: Study) -> dict[int, np.ndarray]:
    """The crest of each ridge and the top of the forest, as polylines on
    the sheet: in each column, the highest pixel of it."""
    out = {}
    for ident in (RIDGE3, RIDGE2, RIDGE1, FOREST):
        mine = (st.ident == ident) | ((ident == RIDGE3) & (st.ident == SNOW))
        mine[int(CY) + 2 :] = False
        has = mine.any(axis=0)
        top = np.argmax(mine, axis=0)
        # (only where it stands against something further: not where a
        # nearer thing hides its top)
        above = st.ident[np.maximum(top - 1, 0), np.arange(WIDTH)]
        has &= (LAYER[above] <= LAYER[ident]) & (top > 0)
        xs = np.nonzero(has)[0]
        top = top.astype(float)
        out[ident] = np.stack([xs.astype(float), top[xs]], axis=1)
    return out


# ----------------------------------------------------------------------
# writing the study


PALETTE = np.array(
    [
        (235, 240, 250),
        (250, 250, 255),
        (255, 255, 255),
        (200, 205, 230),
        (170, 180, 215),
        (130, 145, 190),
        (120, 150, 120),
        (70, 110, 70),
        (150, 200, 230),
        (120, 100, 70),
        (190, 175, 150),
        (160, 160, 160),
        (180, 200, 90),
        (110, 70, 40),
        (30, 70, 40),
        (200, 120, 60),
        (90, 50, 30),
        (240, 200, 120),
        (220, 160, 90),
        (250, 80, 80),
        (60, 60, 60),
    ],
    np.uint8,
)


def write_contours(st: Study, world: World, path: Path) -> None:
    """The contours over a pale copy of the study, to check them."""
    img = np.repeat((180 + 0.3 * st.grey).astype(np.uint8)[..., None], 3, axis=2)
    lines = [(p, (40, 40, 200)) for p in ridge_lines(st).values()]
    for line in world.pine_lines:
        lines += [(r, (20, 120, 20)) for r in seen_runs(line, st)]
    for line in world.boat_lines:
        lines += [(r, (200, 40, 40)) for r in seen_runs(line, st, 0.01)]
    for pts, colour in lines:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        t = np.linspace(0, len(pts) - 1, max(int(seg * 2), 2))
        x = np.interp(t, np.arange(len(pts)), pts[:, 0]).round().astype(int)
        y = np.interp(t, np.arange(len(pts)), pts[:, 1]).round().astype(int)
        ok = (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
        img[y[ok], x[ok]] = colour
    write_png(str(path), img)


def write_study(st: Study, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    grey = np.round(st.grey).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    write_png(str(folder / "study_grey.png"), rgb)
    crops = {
        "crop_pines": (40, 0, 560, 640),
        "crop_boat": (780, 640, 1500, 1000),
        "crop_mist": (430, 330, 1180, 640),
    }
    for name, (x0, y0, x1, y1) in crops.items():
        c = rgb[y0:y1, x0:x1]
        write_png(str(folder / f"{name}.png"), np.repeat(np.repeat(c, 2, 0), 2, 1))
    small = rgb[: HEIGHT // 3 * 3, : WIDTH // 3 * 3].reshape(
        HEIGHT // 3, 3, WIDTH // 3, 3, 3
    )
    write_png(
        str(folder / "study_third.png"),
        np.round(small.mean(axis=(1, 3))).astype(np.uint8),
    )
    write_png(str(folder / "map_ids.png"), PALETTE[st.ident])
    write_png(str(folder / "map_mirrored.png"), PALETTE[st.mirrored])
    for name, field in (
        ("map_veil", st.veil),
        ("map_softness", st.softness),
        ("map_ruffle", st.ruffle),
    ):
        g = np.round(255 * (1.0 - np.clip(field, 0, 1))).astype(np.uint8)
        write_png(str(folder / f"{name}.png"), np.repeat(g[..., None], 3, axis=2))
    a = (st.angle % np.pi) / np.pi
    hue = np.stack([0.5 + 0.5 * np.cos(2 * np.pi * a + k) for k in (0, 2.1, 4.2)], -1)
    write_png(str(folder / "map_angle.png"), np.round(255 * hue * 0.9).astype(np.uint8))


# ----------------------------------------------------------------------
# the drawing: the hand, its marks and its tools

#: A smooth white drawing paper: shallower and finer in the tooth than
#: cartridge, so that rubbed tones go smooth, and deep enough to take a
#: real black.
SHEET_PAPER = papers.CARTRIDGE.with_(
    color="#f6f4ee", depth=0.55, grain=2.4, bite=0.42, give=0.4, name="drawing paper"
)
#: The drawing is shown as it looks scanned or in soft light: heavy
#: graphite keeps a little of its silver, and the darks stay deep.
GLOSS = 0.2
#: The greys the sheet renders flat (no lamp) for bare paper and for the
#: darkest 8B: the range the study's tones are laid out over.
WHITE, BLACK = 243.0, 68.0
#: How the study's tones are laid over that range: the palest tenth of the
#: study is left as paper (mist, the low sky, snow, far water), and above 1
#: keeps the lights generous while the darks stay deep.
LIGHTS = 1.15
PAPER_SHARE = 0.1


class Hand:
    """The draughtsman at work: the sheet, the study it is drawn from, the
    world's lines, the values and directions he plans to draw them in, and
    an eye on what the sheet looks like so far."""

    def __init__(self, st: Study, world: World):
        self.sheet = Sheet(WIDTH, HEIGHT, paper=SHEET_PAPER, seed=SEED)
        self.st, self.world = st, world
        self.rng = np.random.default_rng(SEED + 2)
        self.target = plan_values(st)
        self.angle = plan_angles(st)
        self.seen = np.full((HEIGHT, WIDTH), WHITE)
        # (what a passage still wants, kept while it is worked stroke by
        # stroke)
        self.gap = np.zeros((HEIGHT, WIDTH))

    def look(self) -> None:
        """Step back and look at the sheet (flat, without the lamp)."""
        self.seen = self.sheet.render(lighting=False).mean(axis=2) * 255.0

    def need(self, mask: np.ndarray, blur: int = 2) -> np.ndarray:
        """How much darker each pixel of ``mask`` still has to go, 0..1, as
        last seen."""
        gap = (self.seen - self.target) / (WHITE - BLACK)
        if blur:
            gap = filters.box_blur(gap, blur)
        return np.clip(gap, 0.0, 1.0) * mask

    def excess(self, mask: np.ndarray, blur: int = 3) -> np.ndarray:
        """How much too dark each pixel of ``mask`` is, 0..1."""
        gap = (self.target - self.seen) / (WHITE - BLACK)
        if blur:
            gap = filters.box_blur(gap, blur)
        return np.clip(gap, 0.0, 1.0) * mask

    def mask(self, *ids: int) -> np.ndarray:
        return np.isin(self.st.ident, ids)


def at(field: np.ndarray, x, y):
    """``field`` at sheet points, linearly between pixels."""
    y = np.clip(np.asarray(y, float), 0.0, HEIGHT - 1.001)
    x = np.clip(np.asarray(x, float), 0.0, WIDTH - 1.001)
    return filters.bilinear(field, y, x)


def lookup(field: np.ndarray):
    """A pressure for shading or rubbing that follows ``field``."""
    return lambda x, y: at(field, x, y)


def box_of(mask: np.ndarray, pad: int = 6) -> tuple[int, int, int, int]:
    """The rectangle round ``mask``; where it reaches the sheet's edge, it
    runs on past it, so that strokes do not lift off just inside it."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    over = 3 * pad
    return (
        x0 - (over if x0 == 0 else pad),
        y0 - (over if y0 == 0 else pad),
        x1 + (over if x1 == WIDTH - 1 else pad),
        y1 + (over if y1 == HEIGHT - 1 else pad),
    )


def shrink(mask: np.ndarray, by: int) -> np.ndarray:
    """``mask`` less a margin ``by`` pixels wide."""
    return filters.max_filter((~mask).astype(np.float32), by) < 0.5 if by > 0 else mask


def lay_tone(
    hand: Hand, mask, lead, pressure, angle, tilt=1.0, width=None, margin=0.15
):
    """One layer of strokes side by side over ``mask``, pressed as
    ``pressure`` (a field) asks: with the side of the lead by default.
    The strokes keep ``margin`` of their breadth inside the mask's edge."""
    breadth = (6.0 if width is None else width) * (1.0 + 4.0 * tilt)
    field = pressure * shrink(mask, int(margin * breadth))
    if field.max() > 0.0:
        hand.sheet.shade(box_of(mask), lead, lookup(field), width, angle, tilt=tilt)


def rub_area(hand: Hand, mask, tool, pressure, angle, width=None) -> None:
    """Rub over ``mask`` with a tool, passes to and fro, as hard as
    ``pressure`` (a number or a field) asks, the tool kept inside the
    mask's edge so that it does not lift what lies next to it."""
    reach = tool.width if width is None else width
    field = pressure * shrink(mask, int(0.2 * reach))
    if np.max(field) > 0.0:
        hand.sheet.rub(box_of(mask), tool, lookup(field), width, angle)


def fill_down(values: np.ndarray) -> np.ndarray:
    """Each NaN takes the nearest value above it in its column (or below,
    at the top)."""
    rows = np.arange(HEIGHT)[:, None] * np.ones((1, WIDTH), int)
    good = ~np.isnan(values)
    idx = np.maximum.accumulate(np.where(good, rows, 0), axis=0)
    out = np.take_along_axis(values, idx, axis=0)
    first = np.argmax(good, axis=0)
    top = values[first, np.arange(WIDTH)]
    return np.where(np.isnan(out), top[None, :], out)


@dataclass(frozen=True)
class Marks:
    """One kind of mark as a hand lays it: the lead (or a Stick), the width
    of its tip, how long the strokes are (pixels), how hard they press
    where nothing is needed and where all is, how many lie side by side in
    a bundle and how far apart (in widths), how the pressure runs along
    each ("taper": light at both ends, "flick": heavy where it lands,
    "swell": heavy where it ends), how far a bundle turns from the
    direction asked for (radians), how much each stroke bends, whether it
    follows the study's directions as it goes, and how far the lead is
    tilted onto its side."""

    lead: Medium | Stick
    width: float | None
    length: tuple[float, float]
    pressure: tuple[float, float]
    bundle: tuple[int, int] = (3, 6)
    spacing: float = 1.0
    profile: str = "taper"
    turn: float = 0.12
    bend: float = 0.04
    follow: bool = True
    tilt: float = 0.0

    def area(self) -> float:
        """About how much paper one bundle covers, in pixels."""
        w = (self.width or 3.0) * (1.0 + 4.0 * self.tilt)
        n = 0.5 * (self.bundle[0] + self.bundle[1])
        return n * w * max(self.spacing, 0.6) * 0.5 * sum(self.length)


def profile(kind: str, n: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n)
    if kind == "flick":
        return 1.0 - 0.85 * u**0.8
    if kind == "swell":
        return 0.25 + 0.75 * u**0.7
    return 0.45 + 0.55 * np.sin(np.pi * u) ** 0.5


def pick(rng, weight: np.ndarray, count: int) -> np.ndarray:
    """``count`` points of the sheet, more where ``weight`` is more."""
    w = weight.ravel()
    total = float(w.sum())
    if count <= 0 or total <= 0.0:
        return np.zeros((0, 2))
    idx = rng.choice(w.size, count, p=w / total)
    y, x = np.divmod(idx, WIDTH)
    return np.stack([x, y], axis=1) + rng.uniform(-0.5, 0.5, (count, 2))


def trace(field, x, y, heading, length, bend, rng, step=2.5) -> np.ndarray:
    """A stroke's path from (x, y): along ``heading``, or along the
    directions of ``field`` if one is given, curving a little."""
    n = max(int(length / step), 1)
    dx, dy = math.cos(heading), math.sin(heading)
    curl = rng.normal(0.0, bend)
    c, s = math.cos(curl), math.sin(curl)
    pts = [(x, y)]
    for _ in range(n):
        if field is not None:
            a = field[min(max(int(y), 0), HEIGHT - 1), min(max(int(x), 0), WIDTH - 1)]
            ex, ey = math.cos(a), math.sin(a)
            if ex * dx + ey * dy < 0.0:
                ex, ey = -ex, -ey
            dx, dy = 0.65 * dx + 0.35 * ex, 0.65 * dy + 0.35 * ey
        dx, dy = c * dx - s * dy, s * dx + c * dy
        norm = math.hypot(dx, dy) + 1e-12
        dx, dy = dx / norm, dy / norm
        x, y = x + step * dx, y + step * dy
        pts.append((x, y))
    return np.array(pts)


def within(pts: np.ndarray, keep: np.ndarray | None) -> np.ndarray:
    """The part of a path from its first point inside ``keep`` to where it
    first leaves it."""
    if keep is None:
        return pts
    xi = np.clip(pts[:, 0].astype(int), 0, WIDTH - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, HEIGHT - 1)
    ok = keep[yi, xi]
    if not ok.any():
        return pts[:0]
    start = int(np.argmax(ok))
    stop = start + int(np.argmin(ok[start:])) if not ok[start:].all() else len(pts)
    return pts[start:stop]


def draw_line(hand: Hand, pts: np.ndarray, lead, pressure, width=None, tilt=0.0):
    """One stroke along ``pts`` (pressure a number or one per point)."""
    if len(pts) < 2:
        return
    if isinstance(lead, Stick):
        hand.sheet.stroke(pts, lead, pressure, None, tilt)
    else:
        hand.sheet.stroke(pts, lead, pressure, width, tilt)


def lay_bundle(hand: Hand, marks: Marks, centre, heading, firm, field, keep) -> None:
    rng = hand.rng
    n = int(rng.integers(marks.bundle[0], marks.bundle[1] + 1))
    w = (marks.width or 3.0) * (1.0 + 4.0 * marks.tilt)
    along = np.array([math.cos(heading), math.sin(heading)])
    across = np.array([-along[1], along[0]])
    for j in range(n):
        off = (j - 0.5 * (n - 1)) * marks.spacing * w * rng.uniform(0.7, 1.3)
        length = rng.uniform(*marks.length)
        shift = rng.uniform(-0.25, 0.25) * length - 0.5 * length
        x, y = np.asarray(centre) + off * across + shift * along
        pts = within(trace(field, x, y, heading, length, marks.bend, rng), keep)
        if len(pts) < 2:
            continue
        p = firm * rng.uniform(0.85, 1.15) * profile(marks.profile, len(pts))
        draw_line(hand, pts, marks.lead, np.clip(p, 0.0, 1.0), marks.width, marks.tilt)


def lay_marks(hand: Hand, marks: Marks, need, cover=1.0, angle=None, keep=None) -> None:
    """Bundles of strokes where ``need`` asks, as many as cover that need
    ``cover`` times over: along the study's directions, or at ``angle``
    (radians on the sheet)."""
    count = int(cover * float(need.sum()) / marks.area())
    field = hand.angle if angle is None and marks.follow else None
    lo, hi = marks.pressure
    for cx, cy in pick(hand.rng, need, count):
        here = need[int(cy), int(cx)]
        heading = hand.angle[int(cy), int(cx)] if angle is None else angle
        heading += hand.rng.normal(0.0, marks.turn)
        lay_bundle(hand, marks, (cx, cy), heading, lo + (hi - lo) * here, field, keep)


def drag(hand: Hand, starts, lengths, tool, pressure, heading=np.pi / 2, width=None):
    """Pull a stump or a finger from each start, ``lengths`` pixels."""
    rng = hand.rng
    for (x, y), length in zip(starts, lengths, strict=True):
        a = heading + rng.normal(0.0, 0.05)
        end = (x + length * math.cos(a), y + length * math.sin(a))
        mid = (0.5 * (x + end[0]) + rng.normal(0, 1.0), 0.5 * (y + end[1]))
        hand.sheet.smudge([(x, y), mid, end], tool, pressure, width)


# ----------------------------------------------------------------------
# the drawing, far to near


def draw_sky(hand: Hand) -> None:
    """The sky: a 4B laid very lightly on its side, darker towards the top,
    each layer rubbed in with a finger until its grain is gone; the glow
    wiped lighter with the chamois; the clouds lifted with the kneaded
    eraser."""
    st = hand.st
    clear = fill_down(np.where(st.sky > 0, st.sky, np.nan))
    want = pencil_value(np.clip((PAPER - clear) / (PAPER - DARKEST), 0.0, 1.0))
    rows = np.arange(HEIGHT)[:, None] * np.ones((1, WIDTH))
    region = rows <= CY + 6
    press = (0.065 + 0.9 * want) * filters.smoothstep(0.004, 0.035, want)
    for k, angle in enumerate((15.0, 75.0, -40.0)):
        lay_tone(hand, region, pencils.B4, press * (1.0 - 0.1 * k), angle)
        if k:
            rub_area(hand, region, tools.FINGER, 0.9, 45.0 * k, width=34.0)
    for rub in (-45.0, 0.0):
        rub_area(hand, region, tools.FINGER, 0.9, rub, width=34.0)
    hand.look()
    sky = hand.mask(SKY, CLOUD)
    lift = hand.excess(region & (st.sky > 0), blur=6)
    rub_area(hand, region, tools.CHAMOIS, np.clip(lift * 6.0, 0.0, 0.8), 0.0, 44.0)
    lift_clouds(hand, sky)


def lift_clouds(hand: Hand, sky: np.ndarray) -> None:
    """Soft passes of the kneaded eraser along the clouds."""
    cover = hand.st.cloud * sky
    rng = hand.rng
    for cx, cy in pick(rng, cover, int(cover.sum() / 350.0)):
        length = rng.uniform(50.0, 140.0)
        a = rng.normal(0.0, 0.03)
        dx, dy = 0.5 * length * math.cos(a), 0.5 * length * math.sin(a)
        firm = 0.25 + 0.5 * cover[int(cy), int(cx)]
        bow = rng.normal(0.0, 2.0)
        path = [(cx - dx, cy - dy), (cx, cy + bow), (cx + dx, cy + dy)]
        hand.sheet.erase(path, tools.KNEADED, firm, rng.uniform(12.0, 20.0))


#: The ridges, far to near: the lead laid lightly on its side and rubbed to
#: a haze for the tone, the tool that rubs it, and the hard lead its edge
#: and its fall lines are drawn with (they do not smear).
RIDGE_HANDS = (
    (RIDGE3, pencils.B2, tools.FINGER, pencils.H4),
    (RIDGE2, pencils.B2, tools.FINGER, pencils.H2),
    (RIDGE1, pencils.B4, tools.FINGER, pencils.HB),
)


def ridge_mask(hand: Hand, ident: int) -> np.ndarray:
    return hand.mask(ident, SNOW) if ident == RIDGE3 else hand.mask(ident)


def draw_ridges(hand: Hand) -> None:
    """The far ridges: crossed light layers laid with the side of the lead
    and each rubbed to a haze; then each rubbed level with the finger, the
    pressure easing in from its edges, to one smooth pale tone fading down
    into the mist; each crest made crisp, the tone ending clean against
    the lighter sky; the snow lifted back to the paper."""
    for angles in ((20.0, -25.0), (60.0, 120.0)):
        hand.look()
        for ident, lead, tool, _ in RIDGE_HANDS:
            mask = ridge_mask(hand, ident)
            need = hand.need(mask)
            for k, angle in enumerate(angles):
                press = np.clip(0.12 + 6.0 * need, 0.0, 0.9) * (need > 0.006)
                lay_tone(hand, mask, lead, press * (1 - 0.3 * k), angle)
                rub_area(hand, mask, tool, 0.9, angle + 90.0, 24.0)
                rub_area(hand, mask, tool, 0.9, 0.0, 24.0)
    for ident, _, _, _ in RIDGE_HANDS:
        mask = ridge_mask(hand, ident)
        soft = filters.box_blur(shrink(mask, 4).astype(np.float64), 10)
        for angle in (90.0, 30.0, -30.0):
            rub_area(hand, mask, tools.FINGER, 0.8 * soft, angle, 30.0)
    hand.look()
    for ident, _, _, point in RIDGE_HANDS:
        mask = ridge_mask(hand, ident)
        if ident != RIDGE3:
            marks = Marks(point, 2.0, (10.0, 30.0), (0.12, 0.5), (2, 5), 1.3)
            lay_marks(hand, marks, hand.need(mask), 0.8, keep=mask)
        crest(hand, ident)
    snow = hand.mask(SNOW)
    rng = hand.rng
    for cx, cy in pick(rng, snow.astype(float), int(snow.sum() / 60.0)):
        dab = [(cx - 4, cy + rng.normal()), (cx + 4, cy + rng.normal())]
        hand.sheet.erase(dab, tools.KNEADED, 0.6, 8.0)


def draw_middle(hand: Hand) -> None:
    """The forested hills and the far shore: a 2B laid lightly on its side
    in near-vertical strokes and rubbed across with the stump; then the
    trees, short vertical strokes of varied height massed into a tone,
    darkest at mid-height and fading down into the mist, a few spires
    standing clear on the skyline; and the stump pulled down from their
    feet into the mist."""
    mask = hand.mask(HILLS, FOREST)
    hand.look()
    for angle in (85.0, 96.0):
        need = hand.need(mask)
        press = np.clip(0.1 + 3.5 * need, 0.0, 0.85) * (need > 0.01)
        lay_tone(hand, mask, pencils.B2, press, angle)
        rub_area(hand, mask, tools.STUMP, 0.85, angle - 85.0, 16.0)
        hand.look()
    draw_trees(hand)
    feather_down(hand, mask)


def draw_trees(hand: Hand) -> None:
    """Every tree of the far forest the eye sees, as strokes down its
    height; those whose tops stand against what is behind them, and are
    taller than their neighbours, as clear spires."""
    spec = hand.world.trees
    tip = np.stack(
        [spec[:, 0] + spec[:, 4] * spec[:, 3], spec[:, 1] + spec[:, 3], spec[:, 2]], 1
    )
    foot = np.stack([spec[:, 0], spec[:, 1] + 0.05 * spec[:, 3], spec[:, 2]], 1)
    T, Fp = project(tip, EYE), project(foot, EYE)
    xi = np.clip(T[:, 0].astype(int), 0, WIDTH - 1)
    yi = np.clip(T[:, 1].astype(int) + 1, 0, HEIGHT - 1)
    wood = (FOREST, HILLS)
    seen = (T[:, 0] >= 0) & (T[:, 0] < WIDTH) & np.isin(hand.st.ident[yi, xi], wood)
    seen &= T[:, 2] <= hand.st.depth[yi, xi] * 1.03
    skyline = ~np.isin(hand.st.ident[np.maximum(yi - 4, 0), xi], wood)
    tall = spec[:, 3] > np.percentile(spec[:, 3], 80)
    hand.look()
    hand.gap = hand.need(hand.mask(HILLS, FOREST), 3)
    for k in np.nonzero(seen)[0]:
        tree_strokes(hand, T[k, :2], Fp[k, :2], bool(skyline[k] and tall[k]))


def feather_down(hand: Hand, mask: np.ndarray) -> None:
    """Pull a stump from the foot of the forest down into the mist below
    it, so that the trees sink into it."""
    rows = np.arange(HEIGHT)[:, None]
    low = np.where(mask, rows, -1).max(axis=0)
    rng = hand.rng
    starts, lengths = [], []
    for x in range(0, WIDTH, 5):
        if low[x] < 0 or hand.st.veil[min(low[x] + 4, HEIGHT - 1), x] < 0.3:
            continue
        starts.append((x + rng.uniform(-2, 2), low[x] - rng.uniform(10.0, 26.0)))
        lengths.append(rng.uniform(20.0, 40.0))
    drag(hand, starts, lengths, tools.STUMP, 0.6, width=7.0)


def draw_mist(hand: Hand) -> None:
    """The mist: lifted with the kneaded eraser in long soft passes across
    the view, wherever what lies behind it came out too dark (the ridges
    are left to their own rubbing)."""
    hand.look()
    rows = np.arange(HEIGHT)[:, None] * np.ones((1, WIDTH))
    band = (hand.st.veil > 0.2) & (rows > CY - 260) & (rows < CY + 120)
    band &= ~hand.mask(TRUNK, BOUGH, ROCK, GROUND, SHINGLE, REED)
    band &= ~grow(hand.mask(RIDGE1, RIDGE2, RIDGE3, SNOW), 2)
    lift = hand.excess(band, blur=5)
    rng = hand.rng
    for cx, cy in pick(rng, lift, int(lift.sum() / 120.0)):
        length = rng.uniform(60.0, 200.0)
        firm = np.clip(0.2 + 1.8 * lift[int(cy), int(cx)], 0.0, 0.9)
        bow = rng.normal(0.0, 1.5)
        path = [(cx - length / 2, cy), (cx, cy + bow), (cx + length / 2, cy - bow)]
        hand.sheet.erase(path, tools.KNEADED, firm, rng.uniform(10.0, 22.0))


#: What the water mirrors near the eye (as against the far shore and sky).
#: (A reed's reflection is too thin to draw: the water's tone stands for
#: it.)
NEAR_THINGS = (GROUND, SHINGLE, ROCK, HULL, INSIDE, GUNWALE, THWART, ROPE, STAKE)


#: The water: for what it mirrors, the lead its reflection is drawn with
#: (one grade softer than the thing's own). The sky and the far ridges are
#: left to the water's own tone.
WATER_HANDS = (
    ((RIDGE1, HILLS, FOREST), pencils.B2),
    (NEAR_THINGS, pencils.B6),
    ((TRUNK, BOUGH), pencils.B8),
)


def block_in(hand: Hand) -> None:
    """Near the eye, round one: the big tones of the water and the bank
    laid in with the side of the lead and rubbed; the point's bank as one
    quiet dark tone; the rocks laid in; and the masses of the pines and
    the boat in bundles of strokes."""
    hand.look()
    water, shingle = hand.mask(WATER), hand.mask(SHINGLE)
    smooth_tone(hand, water, pencils.B2, (-20.0, 25.0), 2.2, 0.75)
    rub_area(hand, water, tools.FINGER, 0.9, 5.0)
    bank_tone(hand)
    smooth_tone(hand, shingle, pencils.B2, (10.0, -40.0), 2.4, 0.85)
    rock = hand.mask(ROCK)
    hand.look()
    need = hand.need(rock, 2)
    for angle in (30.0, -35.0):
        lay_tone(
            hand,
            rock,
            pencils.B4,
            np.clip(0.15 + 2.5 * need, 0.0, 0.9),
            angle,
            0.6,
            4.0,
            0.0,
        )
    rub_area(hand, rock, tools.TORTILLON, 0.6, 0.0)
    pine = hand.mask(TRUNK, BOUGH)
    mass = Marks(
        pencils.B4, 3.0, (8.0, 24.0), (0.5, 0.95), (3, 6), 0.8, "taper", 0.15, 0.05
    )
    lay_marks(hand, mass, hand.need(pine), 2.5, keep=pine)
    planks(hand, pencils.HB, pencils.B4)


def grow(mask: np.ndarray, by: int) -> np.ndarray:
    """``mask`` with a margin ``by`` pixels wide added round it."""
    return filters.max_filter(mask.astype(np.float32), by) > 0.5


def smooth_tone(
    hand: Hand, mask, lead, angles, gain, most, tool=tools.FINGER, tilt=1.0
):
    """Two layers laid crossing (with the side of the lead unless ``tilt``
    says otherwise), each rubbed in two directions: a smooth tone. It runs
    a little under what is next to it, as a hand lays a tone up to an edge
    and over it, so that no paper is left between."""
    mask = grow(mask, 6) & ~hand.mask(SKY, CLOUD, HILLS, FOREST, RIDGE1, RIDGE2, RIDGE3)
    need = hand.need(mask, 4)
    for k, angle in enumerate(angles):
        press = np.clip(0.1 + gain * need, 0.0, most) * (1.0 - 0.25 * k)
        lay_tone(hand, mask, lead, press, angle, tilt, None if tilt else 5.0, 0.08)
        rub_area(hand, mask, tool, 0.9, angle + 90.0)
        rub_area(hand, mask, tool, 0.9, angle + 45.0)


def build_up(hand: Hand) -> None:
    """Round two: the reflections laid into the water and dragged down;
    the glow on the top of the point lifted; stones on the bank; the
    rocks given their planes; the pines' boughs and trunks; the boat's
    strakes and its dark inside."""
    water = hand.mask(WATER)
    reflections(hand, water)
    top_light(hand)
    smooth_tone(
        hand, hand.mask(SHINGLE), pencils.B4, (25.0, -20.0), 2.5, 0.9, tools.STUMP
    )
    shingle = hand.mask(SHINGLE)
    stones = Marks(
        pencils.B4, 2.0, (4.0, 12.0), (0.3, 0.9), (1, 3), 1.6, "taper", 0.4, 0.1, False
    )
    lay_marks(hand, stones, hand.need(shingle), 2.0, 0.0, shingle)
    rock_form(hand)
    pine = hand.mask(TRUNK, BOUGH)
    marks = Marks(
        pencils.B6, 2.6, (6.0, 20.0), (0.6, 1.0), (3, 6), 0.8, "taper", 0.15, 0.05
    )
    lay_marks(hand, marks, np.sqrt(hand.need(pine)), 2.5, keep=pine)
    boughs(hand)
    planks(hand, pencils.B2, pencils.B6)
    along = -math.degrees(float(np.median(hand.st.angle[hand.mask(HULL)])))
    rub_area(hand, hand.mask(HULL, INSIDE), tools.TORTILLON, 0.7, along)


def finish(hand: Hand) -> None:
    """Round three: the reflections again where they want more; the
    deepest darks in 8B in the pines and the boat, built in the cores of
    the boughs and trunks; the rocks' waterlines and cracks; the point's
    grass clumps, roots and dark bank edge; then the sharp work: the
    boat's lines and its rope, reeds, pebbles."""
    water = hand.mask(WATER)
    reflections(hand, water, final=True)
    rock_details(hand)
    deepen(hand, hand.mask(TRUNK, BOUGH, INSIDE), 1)
    planks(hand, pencils.B4, pencils.B8)
    grass_clumps(hand)
    roots(hand)
    bank_edge(hand)
    boat_lines_drawn(hand)
    draw_reeds(hand)
    pebbles(hand, hand.mask(SHINGLE))


def last_look(hand: Hand) -> None:
    """A last look: the lights cleaned back to the paper (the mist, the low
    sky, the far water), first with the kneaded eraser and then, where it
    leaves grey, with the vinyl one; the breath of wind lifted across the
    water; the darkest darks once more; and the needles hung from the
    boughs."""
    clear_lights(hand)
    lift_ruffles(hand)
    pine = hand.mask(TRUNK, BOUGH)
    deepen(hand, pine, 1)
    needles(hand, pine)


def reflections(hand: Hand, water: np.ndarray, final: bool = False) -> None:
    """The reflections: for each kind of thing mirrored, its lead (one
    grade softer than the thing's own) laid on its side in two crossing
    layers as hard as the tone wants and rubbed softly together with the
    finger, and the finger dragged gently down through them, so that each
    is one soft mirror image. The finger takes some graphite away, so the
    last round lays a light layer where they still want it: across, and
    rubbed again, under near things; in vertical strokes under the far
    shore and the trees. Then the gaps of light across them, and the far
    water rubbed level."""
    hand.look()
    for ids, lead in WATER_HANDS:
        mask = water & np.isin(hand.st.mirrored, ids)
        near = ids == NEAR_THINGS
        need = hand.need(mask, 4 if lead is pencils.B2 else 8)
        if need.max() < 0.03:
            continue
        soft = filters.box_blur((need > 0.03).astype(np.float64), 10)
        if final:
            press = np.clip(0.1 + 2.5 * need, 0.0, 0.8) * (need > 0.03)
            lay_tone(hand, mask, lead, press, 0.0 if near else 90.0)
            if near:
                rub_area(hand, mask, tools.FINGER, 0.5 * soft, 0.0, 18.0)
            continue
        press = np.clip(0.12 + 3.0 * need, 0.0, 0.9) * (need > 0.03)
        for angle in (60.0, 120.0):
            lay_tone(hand, mask, lead, press, angle)
        rub_area(hand, mask, tools.FINGER, 0.6 * soft, 0.0, 18.0)
        pull_down(hand, need, 0.3 if near else 0.45)
    if final:
        gaps(hand, water)


def rock_form(hand: Hand) -> None:
    """The rocks in their planes: the side in 4B strokes round the form,
    the underside built dark in 8B, each where it wants more; a tortillon
    over the side to make one surface of it; the lit top lifted back with
    the kneaded eraser."""
    rock = hand.mask(ROCK)
    up = hand.st.up
    side = rock & (up > 0.05) & (up < 0.9)
    under = rock & (up <= 0.2)
    hand.look()
    form = Marks(
        pencils.B4, 2.2, (6.0, 16.0), (0.35, 0.95), (3, 5), 0.85, "taper", 0.12, 0.03
    )
    lay_marks(hand, form, hand.need(side), 3.0, keep=side)
    dark = Marks(
        pencils.B8, 2.2, (5.0, 14.0), (0.65, 1.0), (3, 5), 0.8, "taper", 0.12, 0.03
    )
    lay_marks(hand, dark, np.sqrt(hand.need(under)), 3.5, keep=under)
    rub_area(hand, side, tools.TORTILLON, 0.5, 30.0)
    top = rock & (up >= 0.86)
    rng = hand.rng
    for cx, cy in pick(rng, top.astype(float), int(top.sum() / 25.0)):
        a = rng.normal(0.0, 0.3)
        dx, dy = 5 * math.cos(a), 5 * math.sin(a)
        hand.sheet.erase(
            [(cx - dx, cy - dy), (cx + dx, cy + dy)], tools.KNEADED, 0.6, 9.0
        )


def rock_details(hand: Hand) -> None:
    """The dark line where each rock meets the water and one crack down
    each, from its top edge, with a sharp 6B."""
    rock = hand.mask(ROCK)
    below = np.roll(hand.mask(WATER), -2, axis=0) & rock
    line = Marks(
        pencils.B8, 2.2, (6.0, 18.0), (0.7, 0.95), (1, 2), 1.0, "taper", 0.05, 0.03
    )
    lay_marks(hand, line, below * 1.0, 3.0, angle=0.0)
    rng = hand.rng
    for x, z, size, flat, _ in ROCKS:
        P = project(np.array([[x, size * flat * 0.9, z]]), EYE)[0]
        reach = FOCAL * size * flat / P[2]
        x0 = P[0] + rng.uniform(-0.5, 0.5) * reach
        y0 = P[1] + rng.uniform(0.1, 0.4) * reach
        heading = np.pi / 2 + rng.normal(0.0, 0.5)
        length = rng.uniform(0.4, 0.9) * reach
        pts = within(trace(None, x0, y0, heading, length, 0.08, rng, 2.0), rock)
        draw_line(hand, pts, pencils.B6, 0.8 * profile("taper", len(pts)) ** 0.5, 1.5)


def planks(hand: Hand, outside, inside) -> None:
    """One layer on the boat, stroked along its planks: lines through the
    hull at many heights round its girth, in pieces, as hard as the tone
    still wants there; the near side's outside in ``outside``, the far
    side's inside, seen over the near gunwale, in ``inside``."""
    want = hand.need(hand.mask(HULL, INSIDE, GUNWALE, THWART), 1)
    rng = hand.rng
    s = np.linspace(0.0, 1.0, 160)
    for side, lead, width in ((1.0, outside, 2.2), (-1.0, inside, 2.6)):
        for v in np.linspace(0.02, 0.98, 34) + rng.uniform(-0.01, 0.01, 34):
            line = to_world(hull_point(s, np.full_like(s, v), side))
            for run in seen_runs(line, hand.st, 0.02):
                plank_strokes(hand, run, lead, width, want)


def plank_strokes(hand: Hand, run: np.ndarray, lead, width: float, want) -> None:
    rng = hand.rng
    k = 0
    while k < len(run) - 2:
        n = int(rng.integers(8, 30))
        piece = run[k : k + n + 1]
        k += n + int(rng.integers(0, 3))
        need = at(want, piece[:, 0], piece[:, 1])
        if need.max() < 0.02:
            continue
        firm = (
            np.clip(0.35 + 1.2 * need, 0.0, 1.0) * profile("taper", len(piece)) ** 0.5
        )
        draw_line(hand, piece, lead, firm, width)


def boat_lines_drawn(hand: Hand) -> None:
    """The boat's lines with a sharp 2B, sharpened after each: the sheers
    firm, the lands of the strakes lighter, the stem; the painter in one
    line; and the stake in 8B."""
    pencil = Stick(pencils.B2, width=2.0)
    for k, line in enumerate(hand.world.boat_lines):
        sheer = k % STRAKES == 0
        for run in seen_runs(line, hand.st, 0.01):
            firm = (0.8 if sheer else 0.5) * hand.rng.uniform(0.85, 1.1)
            draw_line(hand, run, pencil, firm * profile("taper", len(run)) ** 0.3)
        pencil.sharpen()
    rope, post = hand.world.rope_lines
    for run in seen_runs(rope, hand.st, 0.05):
        draw_line(hand, run, pencil, 0.8 * profile("taper", len(run)) ** 0.2)
        pencil.sharpen()
    for run in seen_runs(post, hand.st, 0.05):
        for dx in (-1.0, 0.0, 1.0):
            draw_line(hand, run + [dx, 0.0], pencils.B8, 0.95, 2.0)


def lift_mist(hand: Hand, where: np.ndarray, firm: float) -> None:
    """Long soft passes of the kneaded eraser across ``where``, as much as
    it came out too dark when last seen."""
    lift = hand.excess(where, blur=5)
    rng = hand.rng
    for cx, cy in pick(rng, lift, int(lift.sum() / 120.0)):
        length = rng.uniform(60.0, 200.0)
        p = float(np.clip(0.2 + 1.8 * firm * lift[int(cy), int(cx)], 0.0, 0.9))
        bow = rng.normal(0.0, 1.5)
        path = [(cx - length / 2, cy), (cx, cy + bow), (cx + length / 2, cy - bow)]
        hand.sheet.erase(path, tools.KNEADED, p, rng.uniform(10.0, 22.0))


def lift_ruffles(hand: Hand) -> None:
    """Where the wind touches the water: a few soft horizontal lights of
    different lengths, each lifted in two or three overlapping, wavering
    passes of the kneaded eraser's edge, the brightest near the far shore
    touched once more with the vinyl one along its middle."""
    rng = hand.rng
    water = grow(hand.mask(WATER), 2)
    for z, x0, x1, _ in RUFFLES:
        y = CY + FOCAL * EYE / z
        span = (x1 - x0) * rng.uniform(0.5, 0.9)
        start = x0 + rng.uniform(0.0, (x1 - x0) - span)
        left, right = CX + FOCAL * start / z, CX + FOCAL * (start + span) / z
        far = float(np.clip((z - 60.0) / 300.0, 0.0, 1.0))
        for _ in range(int(rng.integers(2, 4))):
            a = left + rng.uniform(0, 0.25) * (right - left)
            b = right - rng.uniform(0, 0.25) * (right - left)
            pts = np.stack([np.linspace(a, b, 9), y + rng.normal(0.0, 1.0, 9)], 1)
            pts = within(pts, water)
            if len(pts) > 1:
                firm = (0.45 + 0.35 * far) * profile("taper", len(pts)) ** 2
                hand.sheet.erase(
                    pts, tools.KNEADED, firm.tolist(), rng.uniform(5.0, 9.0)
                )
        if far > 0.3:
            mid = np.stack(
                [
                    np.linspace(
                        0.65 * left + 0.35 * right, 0.35 * left + 0.65 * right, 5
                    ),
                    np.full(5, y),
                ],
                1,
            )
            hand.sheet.erase(
                mid, tools.ERASER, (0.4 * profile("taper", 5) ** 2).tolist(), 3.5
            )


def roots(hand: Hand) -> None:
    """Two or three roots from each tree's foot, out and down into the
    bank."""
    rng = hand.rng
    for tree in PINES:
        foot = np.array([tree.x, ground_at(tree.x, tree.z) + 0.15, tree.z])
        for _ in range(int(rng.integers(2, 4))):
            a = rng.uniform(0, 2 * np.pi)
            reach = rng.uniform(0.6, 1.6) * tree.height / 20.0
            way = [math.cos(a) * reach, -0.35, math.sin(a) * reach]
            line = foot + np.outer(np.linspace(0, 1, 6), way)
            floor = ground_height(line[:, 0], line[:, 2], np.full(6, 0.5))[0]
            line[:, 1] = np.maximum(line[:, 1], floor - 0.05)
            pts = project(line, EYE)[:, :2]
            draw_line(hand, pts, pencils.B8, np.linspace(0.9, 0.3, 6), 2.4)


def pebbles(hand: Hand, shingle: np.ndarray) -> None:
    """The pebbles of the bank: near the eye each a small arc, darker
    beneath; further off fewer and smaller, till they are only tone."""
    rng = hand.rng
    near = shingle * np.clip((16.0 - hand.st.depth) / 6.0, 0.0, 1.0)
    for cx, cy in pick(rng, near, int(near.sum() / 500.0)):
        z = hand.st.depth[int(cy), int(cx)]
        r = FOCAL * rng.lognormal(math.log(0.035), 0.45) / z
        flat = rng.uniform(0.3, 0.65)
        spin = rng.normal(0.0, 0.25)
        a = np.linspace(rng.uniform(0.0, 0.6), np.pi - rng.uniform(0.0, 0.6), 7)
        pts = np.stack(
            [r * np.cos(a), flat * r * np.sin(a) * rng.uniform(0.8, 1.2, 7)], 1
        )
        c, s = math.cos(spin), math.sin(spin)
        pts = pts @ np.array([[c, s], [-s, c]])
        firm = rng.uniform(0.3, 0.75) * profile("taper", 7)
        draw_line(hand, pts + [cx, cy], pencils.B4, firm, 1.6 if r < 6 else 2.0)
        top = [(cx - 0.5 * r, cy - 0.15 * r), (cx + 0.5 * r, cy - 0.2 * r)]
        hand.sheet.erase(
            top, tools.KNEADED, rng.uniform(0.4, 0.7), max(1.2 * flat * r, 3.0)
        )


def draw_reeds(hand: Hand) -> None:
    """Each reed flicked up from its root with a 4B, firm where it lands,
    lifting off at its tip."""
    for line in hand.world.reed_lines:
        for run in seen_runs(line, hand.st, 0.08):
            z = float(line[:, 2].mean())
            wide = float(np.clip(FOCAL * 0.018 / z, 1.2, 2.6))
            firm = hand.rng.uniform(0.55, 0.9) * (
                1.0 - 0.8 * np.linspace(0, 1, len(run))
            )
            draw_line(hand, run, pencils.B4, firm, wide)


def boughs(hand: Hand) -> None:
    """Each bough and trunk the eye sees, along its length."""
    for line in hand.world.pine_lines:
        z = float(line[:, 2].mean())
        wide = float(np.clip(FOCAL * 0.12 / z, 1.8, 4.0))
        for run in seen_runs(line, hand.st, 0.06):
            firm = hand.rng.uniform(0.6, 0.95) * profile("taper", len(run))
            draw_line(hand, run, pencils.B6, firm, wide)


def needles(hand: Hand, pine: np.ndarray) -> None:
    """Short strokes hanging from the undersides of the boughs, with a 6B
    sharpened now and then."""
    fringe = pine & ~np.roll(pine, -3, axis=0) & hand.mask(BOUGH)
    pencil = Stick(pencils.B6, width=1.6)
    marks = Marks(
        pencil, None, (3.0, 9.0), (0.5, 0.9), (2, 4), 1.2, "flick", 0.35, 0.05
    )
    for _ in range(4):
        part = fringe * (hand.rng.random(fringe.shape) < 0.25)
        lay_marks(hand, marks, part * 1.0, marks.area() / 12.0, angle=np.pi / 2)
        pencil.sharpen()


def pencil_value(tone: np.ndarray) -> np.ndarray:
    """How dark the pencil goes for a tone of the study, 0 (paper) to 1."""
    return np.clip((tone - PAPER_SHARE) / (1.0 - PAPER_SHARE), 0.0, 1.0) ** LIGHTS


def plan_values(st: Study) -> np.ndarray:
    """The grey to reach at each pixel: the study's tones laid over the
    pencil's range, then the near ground, the bank and the rocks given
    their planes, and the reflections of near things kept a touch lighter
    than the things, fading downward, as a draughtsman simplifies what he
    sees."""
    target = WHITE - pencil_value(st.tone) * (WHITE - BLACK)
    ids = st.ident
    water = (ids == WATER).astype(np.float64)
    wet = filters.box_blur(water, 14)  # (how near the water)
    far = np.isin(ids, (SKY, CLOUD, RIDGE1, RIDGE2, RIDGE3, HILLS, FOREST))
    pine = np.isin(ids, (TRUNK, BOUGH)).astype(np.float64)
    ground = ids == GROUND
    # the top of the point catches the glow in a narrow band just under
    # the far light; it darkens towards the water and under the trees
    lit = filters.box_blur(np.roll(far, 5, axis=0).astype(np.float64), 5, axis=0)
    shade = filters.box_blur(np.roll(pine, 26, axis=0), 26, axis=0)
    shaped = target + 80.0 * lit - 22.0 * shade - 28.0 * np.clip(3.0 * wet, 0, 1)
    target = np.where(ground, shaped, target)
    # the shingle bank: a lighter top plane, darker to the water's edge
    bank = target + 26.0 - 50.0 * np.clip(3.0 * wet, 0.0, 1.0)
    target = np.where(ids == SHINGLE, bank, target)
    target = np.where(ids == ROCK, rock_planes(st), target)
    mirror = (ids == WATER) & np.isin(st.mirrored, NEAR_THINGS)
    down = rows_into(mirror)
    target = np.where(mirror, target + 16.0 + np.clip(0.9 * down, 0.0, 45.0), target)
    return np.clip(target, BLACK, WHITE)


def rock_planes(st: Study) -> np.ndarray:
    """The rocks in three planes: a lit top, nearly paper; a mid-toned
    side; a dark underside, darkest along the lower edge of each rock as
    seen."""
    up = st.up
    top = filters.smoothstep(0.86, 0.95, up)
    under = 1.0 - filters.smoothstep(0.05, 0.2, up)
    side = 1.0 - top - under
    planes = 200.0 * top + 122.0 * side + 78.0 * under
    rock = st.ident == ROCK
    down = rows_into(rock).astype(np.float64)
    up_from = rows_into(rock[::-1])[::-1].astype(np.float64)
    low = filters.smoothstep(0.45, 0.85, down / np.maximum(down + up_from, 1.0))
    return np.minimum(planes, 200.0 - 125.0 * low)


#: For each ridge, how its crest is drawn: the lead, how firm, and whether
#: a stump pulls the tone down from it.
CRESTS = {
    RIDGE3: (pencils.H2, 0.25, False),
    RIDGE2: (pencils.HB, 0.4, True),
    RIDGE1: (pencils.B2, 0.55, True),
}


def crest(hand: Hand, ident: int) -> None:
    """The crest of a ridge: a band of tone laid along it in long pieces,
    firm at its upper edge and fading below, so that the edge is the
    tone's own; where the mist covers it, it is lost."""
    lead, firm, pull = CRESTS[ident]
    line = ridge_lines(hand.st)[ident]
    rng = hand.rng
    k = 0
    while k < len(line) - 6:
        n = int(rng.integers(60, 180))
        piece = line[k : k + n : 2].astype(float)
        k += n + int(rng.integers(0, 12))
        clear = 1.0 - at(hand.st.veil, piece[:, 0], piece[:, 1])
        if len(piece) < 4 or clear.mean() < 0.05:
            continue
        p = firm * (0.4 + 0.6 * clear) * profile("taper", len(piece)) ** 0.4
        for drop, share in ((1.0, 1.0), (3.0, 0.55), (5.5, 0.3)):
            draw_line(hand, piece + [0.0, drop], lead, p * share, 2.2)
        if pull:
            starts = piece[:: int(rng.integers(3, 6))] + [0.0, 1.5]
            lengths = rng.uniform(8.0, 20.0, len(starts))
            drag(hand, starts, lengths, tools.STUMP, 0.5, width=5.0)


def tree_strokes(hand: Hand, tip, foot, clear_spire: bool) -> None:
    """One far tree: one to three vertical strokes of different lengths
    side by side, pressed as hard as the tone there still wants, hardest at
    mid-height and lifting off into the mist below; a spire standing clear
    is also drawn as a fine line from its very tip."""
    rng = hand.rng
    h = float(foot[1] - tip[1])
    if h < 3.0 or rng.random() < 0.25:
        return
    wide = rng.uniform(1.3, 1.9) if h < 22 else rng.uniform(1.6, 2.2)
    n = int(rng.integers(1, 4))
    ys = np.linspace(0.0, 1.0, 8)
    for j in range(n):
        dx = (j - 0.5 * (n - 1)) * h * rng.uniform(0.06, 0.12)
        top = rng.uniform(0.05, 0.4) if j != n // 2 else rng.uniform(0.0, 0.1)
        span = top + (1.0 - top) * ys * rng.uniform(0.6, 1.0)
        pts = np.stack([np.full(8, tip[0] + dx), tip[1] + h * span], 1)
        pts[:, 0] += rng.normal(0.0, 0.3, 8)
        need = at(hand.gap, pts[:, 0], pts[:, 1])
        mist = 1.0 - at(hand.st.veil, pts[:, 0], pts[:, 1])
        shape = 0.35 + 0.65 * np.sin(np.pi * np.clip(0.2 + 0.8 * ys, 0, 1)) ** 0.6
        p = (0.15 + 2.0 * need) * mist**1.6 * shape * rng.uniform(0.7, 1.15)
        draw_line(hand, pts, pencils.B2, np.clip(p, 0.0, 0.9), wide)
    if clear_spire:
        pts = np.array([tip + [0.0, -1.0], tip + [0.0, 0.45 * h]])
        clear = float(1.0 - at(hand.st.veil, tip[0], tip[1]))
        draw_line(hand, pts, pencils.HB, [0.15 * clear, 0.75 * clear], 1.4)


def deepen(hand: Hand, mask: np.ndarray, rounds: int) -> None:
    """8B laid in layers into the core of ``mask`` (a pixel in from its
    edge, so that the ragged lit edges stay), as long as it wants more."""
    core = shrink(mask, 1)
    marks = Marks(
        pencils.B8, 2.4, (5.0, 16.0), (1.0, 1.0), (3, 5), 0.6, "swell", 0.15, 0.05
    )
    for _ in range(rounds):
        hand.look()
        lay_marks(hand, marks, (hand.need(core, 1) > 0.02) * 1.0, 2.0, keep=core)


def clear_lights(hand: Hand) -> None:
    """Where the plan wants paper and the sheet came out grey (the mist,
    the far water under it): long soft passes of the kneaded eraser, and
    where it was greyer, of the vinyl one held flat and light, kept off
    the things that stand in the light and off the ridges."""
    solid = grow(
        hand.mask(
            TRUNK,
            BOUGH,
            ROCK,
            GROUND,
            SHINGLE,
            REED,
            HULL,
            INSIDE,
            GUNWALE,
            THWART,
            ROPE,
            STAKE,
        ),
        5,
    )
    solid |= grow(hand.mask(RIDGE1, RIDGE2, RIDGE3, SNOW), 2)
    where = (
        (hand.target > 226.0) & ~solid & ~hand.mask(SKY, CLOUD) & (hand.st.veil > 0.3)
    )
    hand.look()
    lift = hand.excess(where, blur=6)
    lift_mist(hand, where, 0.9)
    rng = hand.rng
    grey = np.clip(lift - 0.05, 0.0, 1.0)
    for cx, cy in pick(rng, grey, int(grey.sum() / 40.0)):
        length = rng.uniform(40.0, 150.0)
        bow = rng.normal(0.0, 1.2)
        path = [(cx - length / 2, cy - bow), (cx, cy + bow), (cx + length / 2, cy)]
        firm = float(np.clip(0.15 + 2.5 * grey[int(cy), int(cx)], 0.0, 0.5))
        hand.sheet.erase(
            path,
            tools.ERASER,
            (firm * profile("taper", 9)).tolist(),
            rng.uniform(10.0, 18.0),
        )


def pull_down(hand: Hand, need: np.ndarray, pressure: float) -> None:
    """Drag the finger down through a reflection from where it is darkest,
    further the nearer the water is to the eye."""
    rng = hand.rng
    starts = pick(rng, need, int(need.sum() / 60.0))
    if not len(starts):
        return
    below = np.maximum(starts[:, 1] - CY, 5.0)
    lengths = np.clip(below * rng.uniform(0.1, 0.25, len(starts)), 8.0, 70.0)
    drag(hand, starts, lengths, tools.FINGER, pressure, width=12.0)


def gaps(hand: Hand, water: np.ndarray) -> None:
    """Rub the far water level with the finger, so that the mist lies on a
    smooth mirror; then the horizontal lights across the near
    reflections, each a thin wavering pass of the kneaded eraser: two
    across the boat's, one across each rock's."""
    far = water & (hand.st.depth > 40.0)
    soft = filters.box_blur(far.astype(np.float64), 12)
    rub_area(hand, far, tools.FINGER, 0.8 * soft, 0.0, 24.0)
    boat = water & np.isin(hand.st.mirrored, (HULL, INSIDE, GUNWALE, THWART))
    if boat.any():
        ys = np.nonzero(boat.any(axis=1))[0]
        for share in (0.35, 0.68):
            gap_across(hand, boat, ys[0] + share * (ys[-1] - ys[0]))
    mirrored = water & (hand.st.mirrored == ROCK)
    for x, z, size, flat, _ in ROCKS:
        P = project(np.array([[x, 0.0, z]]), EYE)[0]
        reach = FOCAL * size * flat / z
        y = P[1] + 0.55 * reach
        x0, x1 = int(P[0] - FOCAL * size / z), int(P[0] + FOCAL * size / z)
        band = np.zeros_like(mirrored)
        band[:, max(x0, 0) : max(min(x1, WIDTH), 0)] = True
        if 0 <= y < HEIGHT and (mirrored & band)[int(y)].any():
            gap_across(hand, mirrored & band, y)


def grass_clumps(hand: Hand) -> None:
    """Grass on the point, in a few clumps of flicked 6B strokes of
    different lengths: along its lit top, standing dark against the mist,
    and at the water's edge; nothing between."""
    ground = hand.mask(GROUND)
    top = ground & (hand.target > 160.0)
    edge = ground & grow(hand.mask(WATER), 10) & ~grow(hand.mask(ROCK), 6)
    rng = hand.rng
    for weight, every in ((top, 260.0), (edge, 700.0)):
        for cx, cy in pick(rng, weight.astype(float), int(weight.sum() / every)):
            clump(hand, cx, cy)


def bank_edge(hand: Hand) -> None:
    """The bank of the point where it drops into the water: a dark,
    slightly undercut edge in 8B, broken here and there."""
    ground = hand.mask(GROUND)
    water = hand.mask(WATER)
    edge = ground & (
        filters.max_filter(np.roll(water, -3, axis=0).astype(np.float32), 1) > 0.5
    )
    marks = Marks(
        pencils.B8,
        2.4,
        (8.0, 26.0),
        (0.7, 1.0),
        (1, 3),
        1.0,
        "taper",
        0.08,
        0.03,
        False,
    )
    lay_marks(hand, marks, edge * 0.8, 2.0, angle=0.0, keep=grow(edge, 3))


def plan_angles(st: Study) -> np.ndarray:
    """The directions strokes follow: the study's, except on the point,
    where they run up the bank, away from the water's edge and back
    towards the trees, as the slope is felt."""
    water = (st.ident == WATER).astype(np.float64)
    near = filters.box_blur(filters.box_blur(water, 30), 30)
    gy, gx = np.gradient(near)
    norm = np.hypot(gx, gy) + 1e-9
    ux, uy = -gx / norm, -gy / norm - 0.6
    return np.where(st.ident == GROUND, np.arctan2(uy, ux), st.angle)


def rows_into(mask: np.ndarray) -> np.ndarray:
    """For each pixel of ``mask``, how many rows down it is from where its
    run of the mask in that column began."""
    rows = np.arange(HEIGHT)[:, None] * np.ones((1, WIDTH), int)
    start = np.maximum.accumulate(np.where(mask, 0, rows + 1), axis=0)
    return np.where(mask, rows - start + 1, 0)


def bank_tone(hand: Hand) -> None:
    """The point's bank as one quiet dark tone: a 6B half on its side in
    two crossed layers and an 8B over them, each rubbed smooth with the
    stump and the finger; kept off the rocks, so that they stand clear of
    it."""
    ground, rock = hand.mask(GROUND), hand.mask(ROCK)
    far = hand.mask(SKY, CLOUD, HILLS, FOREST, RIDGE1, RIDGE2, RIDGE3)
    bank = grow(ground, 5) & ~grow(rock, 1) & ~far
    hand.look()
    need = hand.need(bank, 4)
    press = np.clip(0.15 + 3.0 * need, 0.0, 0.95)
    for k, (lead, angle) in enumerate(
        ((pencils.B6, 60.0), (pencils.B6, -55.0), (pencils.B8, 15.0))
    ):
        lay_tone(hand, bank, lead, press * (1.0 - 0.1 * k), angle, 0.5, 5.0, 0.0)
        rub_area(hand, bank, tools.STUMP, 0.7, angle + 90.0)
        rub_area(hand, bank, tools.FINGER, 0.5, angle + 45.0)


def top_light(hand: Hand) -> None:
    """The glow on the top of the point: a narrow light lifted along it
    with the kneaded eraser, in short passes along the edge."""
    ground = hand.mask(GROUND)
    lit = ground * np.clip((hand.target - 150.0) / 40.0, 0.0, 1.0)
    rng = hand.rng
    for cx, cy in pick(rng, lit, int(lit.sum() / 25.0)):
        a = rng.normal(0.0, 0.15)
        half = rng.uniform(8.0, 18.0)
        dx, dy = half * math.cos(a), half * math.sin(a)
        firm = float(np.clip(0.35 + 0.5 * lit[int(cy), int(cx)], 0.0, 0.8))
        path = [(cx - dx, cy - dy), (cx, cy + rng.normal(0, 0.5)), (cx + dx, cy + dy)]
        hand.sheet.erase(
            path, tools.KNEADED, (firm * profile("taper", 7)).tolist(), 5.0
        )


def clump(hand: Hand, cx: float, cy: float) -> None:
    """One clump of grass: blades fanning up from a root."""
    rng = hand.rng
    z = max(float(hand.st.depth[int(cy), int(cx)]), 6.0)
    tall = FOCAL * 0.35 / z
    for _ in range(int(rng.integers(5, 12))):
        heading = -np.pi / 2 + rng.normal(0.0, 0.35)
        length = tall * rng.lognormal(0.0, 0.4)
        base = (cx + rng.normal(0.0, 0.2 * tall), cy + rng.normal(0.0, 0.05 * tall))
        pts = trace(None, *base, heading, length, 0.025, rng, 1.5)
        draw_line(
            hand,
            pts,
            pencils.B6,
            rng.uniform(0.55, 0.95) * profile("flick", len(pts)),
            1.6,
        )


def gap_across(hand: Hand, mask: np.ndarray, y: float) -> None:
    """A thin light lifted across ``mask`` at row ``y``."""
    rng = hand.rng
    xs = np.nonzero(mask[int(y)])[0]
    if len(xs) < 4:
        return
    a, b = xs[0] + rng.uniform(0, 4), xs[-1] - rng.uniform(0, 4)
    px = np.linspace(a, b, 9)
    pts = np.stack([px, y + rng.normal(0.0, 0.7, 9)], 1)
    hand.sheet.erase(
        pts, tools.KNEADED, (0.55 * profile("taper", 9) ** 0.5).tolist(), 3.0
    )


def drawing(st: Study, world: World) -> Hand:
    """Draw the study on a fresh sheet, far to near."""
    hand = Hand(st, world)
    for step in STEPS:
        step(hand)
    return hand


STEPS = (
    draw_sky,
    draw_ridges,
    draw_middle,
    draw_mist,
    block_in,
    build_up,
    finish,
    last_look,
)


def main(studies: Path | None = None) -> None:
    """Build the scene, see it, draw it and write ``examples/output/lake.png``;
    with a folder, also write the study and its maps there."""
    world = build_world()
    st = study(world)
    if studies is not None:
        write_study(st, studies)
        write_contours(st, world, studies / "map_contours.png")
        print(f"wrote the study to {studies}")
    hand = drawing(st, world)
    out = HERE / "output" / "lake.png"
    hand.sheet.save(str(out), gloss=GLOSS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
