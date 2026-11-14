"""Morning mist on a mountain lake, in graphite pencil.

Run from the repository root: ``python examples/lake.py``. With a folder as
argument (``python examples/lake.py studies/``) it writes there the tonal
study the drawing is planned from, and its maps.

A mountain lake at dawn, seen from its near shore. The scene is built in
space before any pencil touches it: three ridges of mountains as height
fields, each further and higher than the one in front; forested hills and a
far shore of conifers; still water at the eye's feet that mirrors them all;
mist lying in bands on the water and in the valleys between the ridges; on
the near shore a group of pines, each grown from its trunk, whorl by whorl;
rocks; reeds; and a clinker-built rowing boat drawn up on the shingle, its
bow rope made fast to a stake. One light: the glow of the sun still below
the ridges, and the soft light of the sky, so that near things stand dark
against the light, their tops catching a little of it.
"""

# cspell:words clinker thwart thwarts gunwale transom icosphere

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from murex import CatmullRom, filters, noise, write_png
from numba import njit

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
NAMES = (
    "sky cloud snow ridge3 ridge2 ridge1 hills forest water ground shingle rock "
    "reed trunk bough hull inside gunwale thwart rope stake"
).split()
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
        0.18,
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
    t = np.where(Z < ridge.z, (ridge.z - Z) / ridge.front, (Z - ridge.z) / ridge.back)
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


def hills(rng) -> tuple[Mesh, Mesh]:
    """The forested hills beyond the lake and the trees on them."""
    line = 220.0 * (noise.value_noise_1d(400, 22, rng) - 0.5)
    rows, cols = 150, 520
    z_rows = np.geomspace(SHORE_Z - 300.0, 3300.0, rows)
    X, Z = frustum_grid(z_rows, cols, reach=1.2)
    extent = (-3500.0, 800.0, 3500.0, 3400.0)
    bumps = world_map(rng, (130, 350), 30.0, 3)
    B = sample_grid(bumps, extent, X, Z) ** 1.3
    Y = hills_height(X, Z, line, B)
    ground = height_mesh(X, Y, Z, HILLS, 0.13)
    return ground, forest(rng, line, bumps, extent)


def forest(rng, line, bumps, extent) -> Mesh:
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
    keep = (Yb > 0.5) & (clear > 0.4) & (rng.random(len(P)) > 0.15)
    P, Yb = P[keep], Yb[keep]
    heights = np.clip(rng.lognormal(np.log(18.0), 0.3, len(P)), 7.0, 36.0)
    return spires(P, Yb, heights, rng)


def spires(P, Yb, heights, rng) -> Mesh:
    """Conifer silhouettes, three tiers each, as triangles facing the eye."""
    n = len(P)
    tiers = 3
    verts, faces = [], []
    lean = rng.normal(0.0, 0.03, n)
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


def signed_distance(X: np.ndarray, Z: np.ndarray, polygon) -> np.ndarray:
    """Distance to the polygon's outline, positive inside."""
    P = np.asarray(polygon, float)
    Q = np.roll(P, -1, axis=0)
    best = np.full(X.shape, np.inf)
    inside = np.zeros(X.shape, bool)
    for (ax, az), (bx, bz) in zip(P, Q, strict=True):
        ex, ez = bx - ax, bz - az
        t = np.clip(((X - ax) * ex + (Z - az) * ez) / (ex * ex + ez * ez), 0.0, 1.0)
        best = np.minimum(best, np.hypot(X - ax - t * ex, Z - az - t * ez))
        crosses = (az > Z) != (bz > Z)
        at = ax + (Z - az) * ex / np.where(ez == 0, 1e-9, ez)
        inside ^= crosses & (X < at)
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
    beach = np.where(fx > -1.0, 6.0, 1.2)  # (the bank is shingle well up)
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

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts), np.array(faces)


def rock(x, z, size, flat, yaw, rng) -> Mesh:
    """A boulder: a sphere pushed out in a few lobes and cut by a few
    planes, so that it has broad facets, flattened, half sunk."""
    v, f = icosphere(2)
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
    v, f = v[f].reshape(-1, 3), np.arange(3 * len(f)).reshape(-1, 3)
    mesh = make_mesh(v, f, ROCK, 0.2)
    a, b, cc = (v[f[:, k]] for k in range(3))
    fn = np.cross(b - a, cc - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    mesh.normals = np.repeat(fn, 3, axis=0)  # flat facets
    mesh.grain = np.cross(fn, np.array([0.0, 1.0, 0.0]))  # along the facet
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


#: The pines: where each stands, how tall, how it leans, and how far its
#: boughs reach for its height (a young tree is bushier).
PINES = (
    Pine(-33.5, 76.0, 24.0, -0.015, 5, 0.95),
    Pine(-23.7, 66.0, 23.5, -0.01, 1, 1.15),
    Pine(-24.1, 72.0, 19.0, 0.012, 3, 0.85),
    Pine(-17.4, 62.0, 17.5, 0.022, 2, 1.0),
    Pine(-14.3, 71.0, 11.5, 0.03, 4, 1.2),
    Pine(-14.1, 58.0, 5.5, -0.02, 6, 1.3),
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


def bough(root, length, azimuth, droop, rng) -> tuple[np.ndarray, list[Stamps]]:
    """One bough: out from the trunk, sagging, its tip turned up a little;
    foliage along it, and twigs hanging from it in a fringe."""
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
    y0 = ground_at(tree.x, tree.z) - 0.3
    H = tree.height
    h = np.linspace(0.0, H, 60)
    sway = 0.25 * np.sin(h / H * 2.6 + tree.seed)
    axis = np.stack(
        [tree.x + tree.lean * h + sway * 0.4, y0 + h, tree.z + sway * 0.3], 1
    )
    parts = [
        curve(axis, 0.03 + 0.24 * (1 - h / H) ** 1.2 * (H / 22), TRUNK, 0.07, 0.05)
    ]
    lines = [axis]
    crown = rng.uniform(0.14, 0.26) * H
    hk = crown
    while hk < H - 0.4:
        up = (hk - crown) / (H - crown)
        reach = (
            0.19 * H * tree.spread * (1.0 - up) ** 0.95 * (0.3 + 0.7 * (1 - up) ** 0.3)
        )
        for azimuth in rng.uniform(0, 2 * np.pi, rng.integers(3, 7)):
            if rng.random() < 0.18:
                continue
            length = max(reach * rng.uniform(0.55, 1.15), 0.35)
            droop = (0.18 + 0.32 * (1 - up)) * rng.uniform(0.6, 1.3)
            root = np.array([np.interp(hk, h, axis[:, k]) for k in range(3)])
            line, bits = bough(root, length, azimuth, droop, rng)
            parts += bits
            lines.append(line)
        hk += rng.uniform(0.38, 0.75) * (1.0 - 0.3 * up)
    parts.append(dead_stubs(axis, h, crown, rng))
    tip = axis[-1] + np.array([0.0, 0.9, 0.0])
    parts.append(curve(np.array([axis[-1], tip]), np.array([0.09, 0.03]), BOUGH, 0.05))
    return merge_stamps(parts), lines


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


def reeds(rng) -> Stamps:
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
    return merge_stamps(parts)


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


def rope_and_stake() -> Stamps:
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
    return merge_stamps([rope, stake_s])


def boat_lines() -> list[np.ndarray]:
    """The lines a draughtsman would draw the boat by, in the world: each
    sheer, the lands of the strakes, the stem and the transom's edge."""
    s = np.linspace(0.0, 1.0, 120)
    lines = []
    for side in (1.0, -1.0):
        lines.append(to_world(hull_point(s, np.ones_like(s), side) + [0, 0, 0.06]))
        for k in range(1, STRAKES):
            lines.append(to_world(hull_point(s, np.full_like(s, k / STRAKES), side)))
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


def boat() -> tuple[Mesh, Stamps]:
    return merge_meshes([hull(), fittings()]), rope_and_stake()


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


def build_world() -> World:
    rng = np.random.default_rng(SEED)
    meshes = [ridge_mesh(r, rng) for r in RIDGES]
    meshes += list(hills(rng))
    meshes.append(ground(rng))
    meshes += [rock(*spec, rng) for spec in ROCKS]
    hull_mesh, rope = boat()
    meshes.append(hull_mesh)
    trees, lines = pines()
    return World(meshes, [trees, reeds(rng), rope], lines, boat_lines())


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


def sky_light(d: np.ndarray, cam: float, clouds: np.ndarray) -> np.ndarray:
    """The sky: lightest low down and towards the glow, a few thin clouds
    lit from below."""
    elev = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    sky = 0.5 + 0.3 * np.exp(-np.maximum(elev, 0.0) / 0.11) + 0.12 * glow(d)
    up = np.maximum(d[..., 1], 1e-4)
    t = (CLOUD_HEIGHT - cam) / up
    X, Z = d[..., 0] * t, d[..., 2] * t
    cover = sample_grid(clouds, CLOUD_EXTENT, X, Z)
    cover *= filters.smoothstep(0.08, 0.13, d[..., 1]) * (
        1 - filters.smoothstep(0.26, 0.32, d[..., 1])
    )
    return sky * (1.0 - 0.75 * cover) + (0.98 + 0.02 * glow(d)) * 0.75 * cover, cover


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
    veil it (0..1), how much the water is ruffled (0..1) and the direction
    strokes should follow (radians on the sheet)."""

    light: np.ndarray
    grey: np.ndarray
    ident: np.ndarray
    mirrored: np.ndarray
    depth: np.ndarray
    veil: np.ndarray
    ruffle: np.ndarray
    angle: np.ndarray

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


def mark_sky(view: View, cam: float, clouds) -> tuple[np.ndarray, np.ndarray]:
    """Fill what the eye sees of the sky; returns its light and the share of
    each pixel covered by cloud."""
    d = directions()
    sky = view.ident < 0
    L, cover = sky_light(d, cam, clouds)
    view.ident[sky & (cover > 0.3)] = CLOUD
    view.ident[sky & (cover <= 0.3)] = SKY
    view.z[sky] = FAR
    view.angle[sky] = 0.0
    return np.where(sky, L, surface_light(view)), cover


def mirror_image(world: World, clouds, tops) -> tuple[np.ndarray, np.ndarray]:
    """What the still water mirrors, as the eye mirrored under the water
    sees it, upside down; and the id of what it sees."""
    view = look(world, -EYE)
    rows = np.arange(HEIGHT)[:, None]
    view.ident[(view.ident < 0) & (rows >= CY)] = WATER  # (never used)
    L, _ = mark_sky(view, -EYE, clouds)
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
    L, _ = mark_sky(view, EYE, clouds)
    mirror, mirror_id = mirror_image(world, clouds, tops)
    wL, seen_id, ruff = water_light(view.z, mirror, mirror_id, rng)
    L = np.where(water, wL, L)
    T = veil(view, EYE, False, tops)
    L = T * L + (1.0 - T) * mist_light(directions())
    return Study(
        L,
        to_grey(L),
        view.ident,
        np.where(water, seen_id, view.ident),
        view.z,
        1.0 - T,
        np.where(water, ruff, 0.0),
        view.angle,
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


def main(studies: Path | None = None) -> None:
    world = build_world()
    st = study(world)
    if studies is not None:
        write_study(st, studies)
        write_contours(st, world, studies / "map_contours.png")
        print(f"wrote the study to {studies}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
