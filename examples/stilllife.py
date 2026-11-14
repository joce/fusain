"""A still life, drawn in blue, black and red ballpoint.

Run from the repository root: ``python examples/stilllife.py FOLDER``.

Stage 1: the scene is modelled and lit, and written to FOLDER as a plain
study in grey and in colour, with the maps the pen will follow.

On a table covered with a striped cloth stand a blue-and-white jug, a dark
wine bottle, a white bowl of cherries, an apple, half an apple and a knife,
before a wall.  Every object is built exactly as a solid: the jug, bottle and
bowl turned from profiles, the fruit as deformed solids of revolution, the
cloth as a surface folded over the table and over its front edge.  One
camera in perspective sees them, lit by a soft key light from the upper left
and a weak fill, with cast shadows, highlights and the light the cloth
throws back into the shadows.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from murex import filters, write_png
from numba import njit

from fusain import Sheet, papers, pens

HERE = Path(__file__).parent
WIDTH, HEIGHT = 1400, 1100


# ------------------------------------------------------------- small tools


def unit_rows(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def unit(*v):
    a = np.array(v, float)
    return a / np.linalg.norm(a)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def value_noise(x, seed):
    table = np.random.default_rng(seed).random(4096)
    i = np.floor(x).astype(np.int64)
    t = x - i
    t = t * t * (3 - 2 * t)
    return table[i % 4096] * (1 - t) + table[(i + 1) % 4096] * t


def noise2(x, y, seed):
    """Smooth value noise in two dimensions (0 to 1)."""
    rng = np.random.default_rng(seed)
    table = rng.random((256, 256))
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    tx, ty = smoothstep(0, 1, x - xi), smoothstep(0, 1, y - yi)

    def at(i, j):
        return table[(yi + j) % 256, (xi + i) % 256]

    top = at(0, 0) * (1 - tx) + at(1, 0) * tx
    bottom = at(0, 1) * (1 - tx) + at(1, 1) * tx
    return top * (1 - ty) + bottom * ty


def catmull_rom(pts, n):
    """n points along a Catmull-Rom curve through ``pts`` (k, d), spaced
    evenly along its length."""
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    t = np.linspace(0, 1, 40)[:-1, None]
    seg = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        seg.append(
            0.5
            * (
                2 * p1
                + (p2 - p0) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (3 * p1 - p0 - 3 * p2 + p3) * t**3
            )
        )
    seg.append(pts[-1:])
    dense = np.vstack(seg)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, dense[:, k]) for k in range(dense.shape[1])], 1)


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return unit_rows(vn)


def grid_triangles(rows, cols, wrap=False):
    """Triangles of a grid of points (rows, cols), the columns closing
    round if ``wrap``."""
    i, j = np.meshgrid(
        np.arange(rows - 1), np.arange(cols if wrap else cols - 1), indexing="ij"
    )
    a = i * cols + j
    b = i * cols + (j + 1) % cols
    return np.concatenate(
        [
            np.stack([a, a + cols, b], -1).reshape(-1, 3),
            np.stack([b, a + cols, b + cols], -1).reshape(-1, 3),
        ]
    )


def shrink(img, k):
    h, w = img.shape[0] // k, img.shape[1] // k
    return img[: h * k, : w * k].reshape(h, k, w, k, *img.shape[2:]).mean(axis=(1, 3))


def to_linear(hex_colour):
    c = np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def as_png(lin):
    return np.round(to_srgb(lin) * 255).astype(np.uint8)


# ---------------------------------------------------------------- the parts

# What each part is made of: its material.
(
    M_WALL,
    M_TABLE,
    M_CLOTH,
    M_JUG,
    M_JUG_IN,
    M_BOTTLE,
    M_BOWL,
    M_CHERRY,
    M_STEM,
    M_APPLE,
    M_FLESH,
    M_BLADE,
    M_HANDLE,
    M_JUG_HANDLE,
) = range(14)


@dataclass
class Part:
    """A surface: its points, triangles, an object number and material,
    coordinates on its material (for patterns) and two tangents along which
    it would be hatched."""

    v: np.ndarray
    f: np.ndarray
    obj: int
    mat: int | np.ndarray
    uv: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    smooth: bool = True


def lathe(profile, n=180, deform=None, arc=None):
    """A surface turned from a profile of (radius, height) points about the
    y axis.  ``deform(theta, r, y)`` may return new (r, y).  Returns the
    points (rows along the profile, columns round), the material
    coordinates (angle, distance along the profile) and the two tangents
    (round, along the profile)."""
    if arc is None:
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    else:
        th = np.linspace(arc[0], arc[1], n)
    T, R = np.meshgrid(th, profile[:, 0], indexing="xy")
    _, Y = np.meshgrid(th, profile[:, 1], indexing="xy")
    if deform is not None:
        R, Y = deform(T, R, Y)
    v = np.stack([R * np.sin(T), Y, R * np.cos(T)], -1)
    s = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))]
    )
    uv = np.stack([T, np.broadcast_to(s[:, None], T.shape)], -1)
    t1 = unit_rows(np.stack([np.cos(T), 0 * T, -np.sin(T)], -1))
    t2 = unit_rows(np.gradient(v, axis=0))
    rows = len(profile)
    f = grid_triangles(rows, n, wrap=arc is None)
    return v.reshape(-1, 3), f, uv.reshape(-1, 2), t1.reshape(-1, 3), t2.reshape(-1, 3)


def place(part_v, at, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return part_v @ r.T + np.asarray(at, float)


def turn(t, yaw):
    return place(t, (0, 0, 0), yaw)


def turned(profile_pts, n_profile, obj, mat, at, yaw=0.0, n=180, deform=None):
    prof = catmull_rom(np.asarray(profile_pts, float), n_profile)
    v, f, uv, t1, t2 = lathe(prof, n, deform)
    return Part(place(v, at, yaw), f, obj, mat, uv, turn(t1, yaw), turn(t2, yaw))


def tube(path, radius, obj, mat, n=16, flat=1.0):
    """A tube of varying ``radius`` (one per point) along a 3D path, its
    section squashed by ``flat`` across the path's bend."""
    t = unit_rows(np.gradient(path, axis=0))
    ref = np.where(np.abs(t[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
    a = unit_rows(np.cross(t, ref))
    b = np.cross(t, a)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.asarray(radius, float)[:, None, None]
    ring = (
        np.cos(th)[None, :, None] * a[:, None]
        + flat * np.sin(th)[None, :, None] * b[:, None]
    )
    v = path[:, None] + r * ring
    rows = len(path)
    s = np.linspace(0, 1, rows)
    uv = np.stack(np.broadcast_arrays(th[None, :], s[:, None]), -1)
    t1 = unit_rows(np.cross(t[:, None], ring))
    t2 = np.broadcast_to(t[:, None], v.shape)
    return Part(
        v.reshape(-1, 3),
        grid_triangles(rows, n, wrap=True),
        obj,
        mat,
        uv.reshape(-1, 2),
        t1.reshape(-1, 3),
        t2.reshape(-1, 3),
    )


# ---------------------------------------------------------------- the scene

CLOTH_Y = 0.15  # the cloth's thickness: where things stand
TABLE_FRONT = 22.0  # z of the table's front edge
WALL_Z = -25.0
JUG_AT, JUG_YAW = (-14.0, -6.0), -0.35
BOTTLE_AT = (1.5, -16.0)
BOWL_AT = (-1.5, 6.5)
APPLE_AT = (13.5, 3.0)
HALF_AT, HALF_YAW = (9.0, 14.0), 0.5
KNIFE = ((27.0, 19.5), (12.5, 9.5))  # handle's end, blade's tip (x, z)
LOOSE_CHERRIES = [(-12.5, 12.5, 0.4), (-9.6, 15.2, 2.2)]  # x, z, yaw

# Objects, numbered.
O_WALL, O_TABLE, O_CLOTH, O_JUG, O_BOTTLE, O_BOWL, O_APPLE, O_HALF, O_KNIFE = range(9)
O_CHERRY = 20  # and up, one per cherry

JUG_PROFILE = [
    (0.0, 0.0),
    (4.0, 0.0),
    (4.5, 0.3),
    (4.4, 1.1),
    (5.2, 2.5),
    (7.0, 5.2),
    (7.9, 8.3),
    (7.6, 11.4),
    (6.3, 14.3),
    (4.8, 16.8),
    (4.5, 18.4),
    (4.9, 20.4),
    (5.5, 21.8),
    (5.45, 22.25),
    (5.1, 22.0),
    (4.6, 20.5),
    (4.15, 18.6),
    (4.3, 16.6),
    (5.0, 14.2),
    (0.0, 13.6),
]
JUG_RIM = 13  # index of the rim's top in the profile: inside after it
BOTTLE_PROFILE = [
    (0.0, 0.05),
    (3.7, 0.0),
    (4.1, 0.45),
    (4.15, 2.0),
    (4.15, 6.0),
    (4.15, 10.0),
    (4.15, 14.0),
    (4.15, 17.0),
    (4.1, 18.8),
    (3.9, 20.3),
    (3.45, 21.6),
    (2.8, 23.0),
    (2.0, 24.4),
    (1.45, 25.9),
    (1.35, 29.4),
    (1.42, 30.3),
    (1.72, 30.6),
    (1.72, 31.5),
    (1.45, 31.8),
    (0.0, 31.9),
]
BOWL_PROFILE = [
    (0.0, 0.0),
    (3.6, 0.0),
    (3.9, 0.3),
    (3.8, 0.9),
    (5.5, 1.6),
    (8.3, 3.2),
    (10.2, 4.8),
    (10.6, 5.4),
    (10.35, 5.65),
    (9.9, 5.05),
    (7.8, 3.45),
    (4.5, 1.95),
    (0.0, 1.55),
]
CHERRY_PROFILE = [
    (0.0, -1.1),
    (0.5, -1.05),
    (0.95, -0.8),
    (1.22, -0.25),
    (1.28, 0.25),
    (1.12, 0.75),
    (0.7, 1.05),
    (0.3, 0.97),
    (0.0, 0.85),
]
APPLE_PROFILE = [
    (0.0, 0.6),
    (0.8, 0.38),
    (2.2, 0.25),
    (3.4, 0.8),
    (4.2, 2.2),
    (4.45, 3.6),
    (4.3, 5.0),
    (3.8, 6.3),
    (2.6, 7.2),
    (1.2, 7.25),
    (0.5, 6.85),
    (0.0, 6.35),
]


def spout(T, R, Y, at=-np.pi / 2):
    """Pull the jug's rim out and up into a lip at angle ``at``."""
    d = np.angle(np.exp(1j * (T - at)))
    top = smoothstep(17.0, 22.0, Y) * (R > 4.9)
    w = np.exp(-((d / 0.42) ** 2))
    pinch = np.exp(-(((np.abs(d) - 0.75) / 0.25) ** 2))
    return R + top * (1.9 * w - 0.35 * pinch), Y + 0.7 * top * w**2


def jug_parts():
    at = (JUG_AT[0], CLOTH_Y, JUG_AT[1])
    body = turned(JUG_PROFILE, 320, O_JUG, M_JUG, at, JUG_YAW, 220, spout)
    rim_s = np.linalg.norm(
        np.diff(np.asarray(JUG_PROFILE[: JUG_RIM + 1]), axis=0), axis=1
    )
    inside = body.uv[:, 1] > rim_s.sum() + 0.05
    body.mat = np.where(inside[body.f].all(axis=1), M_JUG_IN, M_JUG)
    path = catmull_rom(
        np.array(
            [
                (4.2, 19.3, 0),
                (6.6, 19.7, 0),
                (8.8, 18.3, 0),
                (9.5, 15.4, 0),
                (8.8, 12.2, 0),
                (7.3, 9.4, 0),
            ]
        ),
        80,
    )
    handle = tube(path, np.full(80, 0.8), O_JUG, M_JUG_HANDLE, 20, 0.55)
    handle.v = place(handle.v, at, JUG_YAW)
    handle.t1, handle.t2 = turn(handle.t1, JUG_YAW), turn(handle.t2, JUG_YAW)
    return [body, handle]


def bottle_part():
    at = (BOTTLE_AT[0], CLOTH_Y, BOTTLE_AT[1])
    return turned(BOTTLE_PROFILE, 300, O_BOTTLE, M_BOTTLE, at, 0.0, 160)


def bowl_part():
    at = (BOWL_AT[0], CLOTH_Y, BOWL_AT[1])
    return turned(BOWL_PROFILE, 200, O_BOWL, M_BOWL, at, 0.0, 200)


def bowl_inner_height(r):
    """Height of the bowl's inner surface above its base at radius r."""
    inner = np.array(BOWL_PROFILE[9:])[::-1]
    return np.interp(r, inner[:, 0], inner[:, 1])


# ------------------------------------------------------------------ fruit

CHERRY_R = 1.25
#: Where the cherries are dropped into the bowl: (x, z) from its centre.
CHERRY_DROPS = [
    (0.0, 0.0),
    (2.4, 0.6),
    (-2.3, 0.9),
    (0.6, 2.5),
    (-0.9, -2.3),
    (1.6, -2.0),
    (3.9, -1.6),
    (-3.9, -1.0),
    (-1.6, 3.6),
    (2.4, 3.2),
    (-4.4, 2.6),
    (4.5, 1.6),
    (0.6, 0.9),
    (-1.0, 1.2),
    (1.5, 1.5),
    (-2.6, -2.8),
    (3.4, -3.6),
]


def drop_cherries():
    """Rest each cherry on the bowl or on those already in it."""
    placed = []
    for dx, dz in CHERRY_DROPS:
        r = math.hypot(dx, dz)
        y = bowl_inner_height(r + 0.6 * CHERRY_R) + CHERRY_R * 1.02
        for px, py, pz in placed:
            d = math.hypot(dx - px, dz - pz)
            if d < 2 * CHERRY_R:
                y = max(y, py + math.sqrt(4 * CHERRY_R**2 - d * d))
        placed.append((dx, y, dz))
    x0, z0 = BOWL_AT
    return [(x0 + dx, CLOTH_Y + y, z0 + dz) for dx, y, dz in placed]


def tilt(v, axis, angle):
    """Rotate points about a unit axis through the origin (Rodrigues)."""
    k = unit(*axis)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1 - c)


def cherry(k, at, rng, lying=False, yaw=0.0):
    """A cherry: a little heart-shaped, with a dimple and a curved stem."""
    body = turned(CHERRY_PROFILE, 60, O_CHERRY + k, M_CHERRY, (0, 0, 0), 0, 48)
    stem = np.array(
        [(0, 0.9, 0), (0.15, 1.9, 0.1), (0.55, 2.9, 0.25), (1.15, 3.6, 0.45)]
    )
    stem = stem * [1, rng.uniform(0.8, 1.15), 1]
    axis, angle = (rng.normal(), 0.0, rng.normal()), rng.uniform(0.3, 1.0)
    if lying:
        axis, angle = (0.0, 0.0, 1.0), 1.25
    parts = [
        body,
        tube(
            catmull_rom(stem, 40), np.linspace(0.09, 0.065, 40), O_CHERRY + k, M_STEM, 8
        ),
    ]
    for p in parts:
        spin = rng.uniform(0, 6.3) if not lying else 0.0
        for name in ("v", "t1", "t2"):
            q = place(getattr(p, name), (0, 0, 0), spin)
            q = tilt(q, axis, angle)
            q = place(q, (0, 0, 0), yaw)
            setattr(p, name, q + (np.asarray(at) if name == "v" else 0))
    return parts


def cherries():
    rng = np.random.default_rng(5)
    out = []
    for k, at in enumerate(drop_cherries()):
        out += cherry(k, at, rng)
    n = len(CHERRY_DROPS)
    for j, (x, z, yaw) in enumerate(LOOSE_CHERRIES):
        at = (x, CLOTH_Y + CHERRY_R * 0.97, z)
        out += cherry(n + j, at, rng, lying=True, yaw=yaw)
    return out


def apple_shape(T, R, Y):
    """Lobes and a little lopsidedness, more towards the top."""
    up = smoothstep(1.0, 6.0, Y)
    k = 1 + (0.035 * np.cos(5 * T + 0.5) * up + 0.04 * np.cos(T - 1.0)) * (R > 0.5)
    return R * k, Y


def apple_parts():
    at = (APPLE_AT[0], CLOTH_Y - 0.2, APPLE_AT[1])
    skin = turned(APPLE_PROFILE, 160, O_APPLE, M_APPLE, (0, 0, 0), 0, 160, apple_shape)
    stalk = tube(
        catmull_rom(np.array([(0, 6.3, 0), (0.1, 7.4, 0), (0.45, 8.5, 0.1)]), 20),
        np.linspace(0.2, 0.13, 20),
        O_APPLE,
        M_STEM,
        10,
    )
    for p in (skin, stalk):
        p.v = tilt(p.v, (0.3, 0, 1), -0.14) + at
        p.t1, p.t2 = tilt(p.t1, (0.3, 0, 1), -0.14), tilt(p.t2, (0.3, 0, 1), -0.14)
    return [skin, stalk]


def half_apple_parts():
    """Half an apple lying cut face up: the skin of one half, and the flat
    face of the cut (its material coordinates: across, along the core)."""
    prof = catmull_rom(np.array(APPLE_PROFILE), 160)
    v, f, uv, t1, t2 = lathe(prof, 90, apple_shape, arc=(np.pi, 2 * np.pi))
    skin = Part(v, f, O_HALF, M_APPLE, uv, t1, t2)
    rows, cols = len(prof), 41
    s = np.linspace(-1, 1, cols)
    r, y = prof[:, 0:1], prof[:, 1:2]
    face = np.stack(np.broadcast_arrays(0 * s, y, r * s), -1)
    fuv = np.stack(np.broadcast_arrays(s[None, :], y), -1)
    cut = Part(
        face.reshape(-1, 3),
        grid_triangles(rows, cols)[:, ::-1],
        O_HALF,
        M_FLESH,
        fuv.reshape(-1, 2),
        np.tile([0.0, 0.0, 1.0], (rows * cols, 1)),
        np.tile([0.0, 1.0, 0.0], (rows * cols, 1)),
    )
    parts = [skin, cut]
    lift = float(prof[:, 0].max()) * 1.02 + CLOTH_Y
    for p in parts:
        for name in ("v", "t1", "t2"):
            q = getattr(p, name) - ([0, 3.7, 0] if name == "v" else 0)
            q = q[:, [1, 0, 2]] * [-1, 1, 1]  # the cut face (x = 0) turns up
            q = tilt(q, (1, 0, 0), 0.16)  # and a little towards us
            q = place(q, (0, 0, 0), HALF_YAW)
            setattr(p, name, q + ((HALF_AT[0], lift, HALF_AT[1]) if name == "v" else 0))
    return parts


# ------------------------------------------------------------- the knife


def blade_outline(n=60):
    """Half-widths of the blade along it (0 at the bolster, 1 at the tip):
    a straight spine, the edge curving up to the point."""
    t = np.linspace(0, 1, n)
    edge = 1.9 * np.sqrt(np.clip(1 - t**2.2, 0, 1)) * (0.94 + 0.06 * (1 - t))
    return t, edge


def knife_parts():
    (hx, hz), (tx, tz) = KNIFE
    d = unit(tx - hx, 0.0, tz - hz)
    side = np.cross([0.0, 1.0, 0.0], d)
    handle_len, total = 10.0, math.hypot(tx - hx, tz - hz)
    start = np.array([hx, 0.0, hz])
    t, w = blade_outline()
    along = handle_len + t * (total - handle_len)
    y = CLOTH_Y + 0.62 - 0.45 * t  # resting on its handle and on its tip
    spine = start + np.outer(along, d) + np.outer(y, [0, 1, 0])
    across = np.linspace(0, 1, 9)
    top = spine[:, None] + (w[:, None] * across[None, :])[..., None] * side
    blade = Part(
        top.reshape(-1, 3),
        grid_triangles(len(t), 9),
        O_KNIFE,
        M_BLADE,
        np.stack(np.broadcast_arrays(across[None], t[:, None]), -1).reshape(-1, 2),
        np.tile(side, (top.size // 3, 1)),
        np.tile(d, (top.size // 3, 1)),
    )
    hp = start + np.outer(np.linspace(0.3, handle_len, 40), d) + [0, CLOTH_Y + 0.7, 0]
    radius = 0.75 * np.sqrt(np.clip(np.linspace(0, 1, 40) / 0.08, 0, 1)) * 0.98
    handle = tube(hp, radius, O_KNIFE, M_HANDLE, 20, 0.55)
    return [blade, handle]


# ------------------------------------------------------- cloth, table, wall

#: Folds of the cloth: (path of (x, z) points from back to front, height,
#: half-width), cm.
FOLDS = [
    ([(-34, -32), (-30, -12), (-27, 4), (-22, 20)], 1.54, 1.7),
    ([(-24, -33), (-22, -20), (-21, -12)], 1.26, 1.4),
    ([(-8, -34), (-7, -20), (-5, -8), (-6, -2)], 1.96, 1.6),
    ([(-10, -2), (-4, -3.5), (3, -4), (9, -2)], 3.36, 1.5),
    ([(-7, 0.5), (-2, -1), (4, -1.2)], 2.52, 1.1),
    ([(-9, -6), (-3, -8), (5, -8.5)], 2.24, 1.3),
    ([(6, -30), (8, -22), (9, -8), (8, -3)], 1.68, 1.5),
    ([(-6, 16), (-5, 19), (-6, 22.5)], 1.12, 1.2),
    ([(20, -30), (23, -12), (24, 0), (21, 12), (19, 22.5)], 1.54, 1.8),
    ([(33, -28), (34, -6), (38, 10), (37, 22.5)], 1.68, 1.9),
    ([(1, 17), (3, 20), (4, 22.5)], 0.84, 1.0),
    ([(-19, -9), (-25, -16), (-29, -24)], 0.9, 0.7),
    ([(-8, 1), (-11, 4), (-14, 8)], 0.7, 0.6),
    ([(4, 11), (6, 15), (5, 19)], 0.7, 0.6),
    ([(17, 5), (21, 3), (27, -1)], 0.8, 0.7),
    ([(-1, -20), (-6, -26), (-9, -32)], 0.9, 0.8),
    ([(13, -2), (15, -9), (14, -18)], 0.8, 0.7),
]
#: Where things stand on the cloth, which lies flat under them: (x, z, r).
FOOTPRINTS = [
    (*JUG_AT, 5.2),
    (*BOTTLE_AT, 4.8),
    (*BOWL_AT, 4.5),
    (*APPLE_AT, 3.5),
    (*HALF_AT, 3.0),
    (-12.5, 12.5, 1.3),
    (-9.6, 15.2, 1.3),
    (14.0, 10.5, 1.6),
    (17.5, 13.0, 1.6),
    (21.0, 15.5, 1.6),
    (24.5, 18.0, 1.6),
]
#: The cloth's free left edge: x at the back and at the front.
CLOTH_LEFT = (-75.0, -9.0)
CLOTH_RIGHT = 62.0
CLOTH_BACK = -23.0
HANG = 7.0  # how far it hangs over the front edge
EDGE_R = 0.9  # radius of the table's edge, under the cloth


def fold_lines(n=60):
    return [(catmull_rom(np.array(p, float), n), a, w) for p, a, w in FOLDS]


def segment_distance(q, a, b):
    ab = b - a
    t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(q - (a + t[:, None] * ab), axis=1), t


def folds_at(x, z):
    """Height of the folds over the table at (x, z), and the folds'
    direction there (unit (dx, dz)), weighted by how near each fold is."""
    q = np.stack([x.ravel(), z.ravel()], 1)
    height = np.zeros(len(q))
    direc = np.tile([0.0, 1.0], (len(q), 1)) * 0.2
    for line, amp, width in fold_lines():
        best = np.full(len(q), np.inf)
        tang = np.zeros((len(q), 2))
        along = np.zeros(len(q))
        for i in range(len(line) - 1):
            d, t = segment_distance(q, line[i], line[i + 1])
            closer = d < best
            best[closer] = d[closer]
            tang[closer] = unit_rows(line[i + 1] - line[i])
            along[closer] = ((i + t) / (len(line) - 1))[closer]
        taper = np.sin(np.pi * np.clip(along, 0, 1)) ** 0.5
        bump = amp * taper * np.exp(-0.5 * (best / width) ** 2)
        height = np.maximum(height, bump) + 0.25 * np.minimum(height, bump)
        direc += tang * np.exp(-0.5 * (best / (2 * width)) ** 2)[:, None]
    return height.reshape(x.shape), unit_rows(direc).reshape(*x.shape, 2)


def flat_under(x, z):
    out = np.ones_like(x)
    for fx, fz, r in FOOTPRINTS:
        d = np.hypot(x - fx, z - fz)
        out = np.minimum(out, smoothstep(r + 0.3, r + 2.8, d))
    return out


def cloth_point(u, v):
    """The cloth at (u across 0 to 1, v down its length in cm from the back):
    on the table, over the rounded edge, then hanging."""
    z_top = np.minimum(CLOTH_BACK + v, TABLE_FRONT)
    left = np.interp(z_top, [CLOTH_BACK, TABLE_FRONT], CLOTH_LEFT)
    x = left + u * (CLOTH_RIGHT - left)
    lift, _ = folds_at(x, z_top)
    noise = 0.3 * (noise2(x / 9, z_top / 9, 3) - 0.5)
    noise += 0.12 * (noise2(x / 2.5 + 40, z_top / 4.0, 6) - 0.5)
    y_top = CLOTH_Y + (lift + noise) * flat_under(x, z_top)
    past = np.clip(CLOTH_BACK + v - TABLE_FRONT, 0, None)
    arc = np.minimum(past, EDGE_R * np.pi / 2) / EDGE_R
    drop = np.clip(past - EDGE_R * np.pi / 2, 0, None)
    z = z_top + EDGE_R * np.sin(arc) + hanging_folds(x, drop)
    y = np.where(past > 0, CLOTH_Y - EDGE_R * (1 - np.cos(arc)) - drop, y_top)
    y = np.where(past > 0, y + (y_top - CLOTH_Y) * np.exp(-past / 1.5), y)
    return np.stack([x, y, z], -1)


def hanging_folds(x, drop):
    """The hanging part falls in a few soft tubes that grow downwards."""
    grow = np.clip(drop / 3.0, 0, 1) ** 0.8
    out = np.zeros_like(x)
    for cx, a, w in (
        (-6.0, 4.0, 1.7),
        (5.0, 3.0, 1.4),
        (17.0, 4.6, 1.9),
        (31.0, 3.4, 1.6),
    ):
        out += a * np.exp(-0.5 * ((x - cx - 0.15 * drop) / w) ** 2)
    return grow * out + 0.25 * grow * np.sin(x / 2.3)


def cloth_length():
    return TABLE_FRONT - CLOTH_BACK + EDGE_R * np.pi / 2 + HANG


def cloth_hem(U, V, length):
    """The hanging edge, uneven: how much shorter the cloth is there."""
    uneven = 1.6 * np.sin(U * 13.0 + 1.0) + 0.7 * np.sin(U * 37.0)
    return uneven * np.clip(V - length + 4.0, 0, None) / 4.0


def cloth_part(rows=420, cols=520):
    length = TABLE_FRONT - CLOTH_BACK + EDGE_R * np.pi / 2 + HANG
    v = np.linspace(0, length, rows)
    u = np.linspace(0, 1, cols)
    U, V = np.meshgrid(u, v)
    p = cloth_point(U, V - cloth_hem(U, V, length))
    t_across = unit_rows(np.gradient(p, axis=1))
    t_along = unit_rows(np.gradient(p, axis=0))
    top = p[..., 2] < TABLE_FRONT - 0.5
    _, d = folds_at(p[..., 0], p[..., 2])
    dirs = unit_rows(np.stack([d[..., 0], 0 * d[..., 0], d[..., 1]], -1))
    slope = np.sum(dirs * t_along, axis=-1, keepdims=True) * t_along
    slope += np.sum(dirs * t_across, axis=-1, keepdims=True) * t_across
    t_fold = np.where(top[..., None], unit_rows(slope), t_along)
    t_cross = unit_rows(np.cross(t_fold, np.cross(t_across, t_along)))
    width = CLOTH_RIGHT - np.interp(
        np.minimum(CLOTH_BACK + V, TABLE_FRONT), [CLOTH_BACK, TABLE_FRONT], CLOTH_LEFT
    )
    uv = np.stack([U * width, V], -1)
    return Part(
        p.reshape(-1, 3),
        grid_triangles(rows, cols),
        O_CLOTH,
        M_CLOTH,
        uv.reshape(-1, 2),
        t_fold.reshape(-1, 3),
        t_cross.reshape(-1, 3),
    )


def plane(origin, a, b, na, nb, obj, mat):
    """A flat rectangle from ``origin`` spanned by vectors a and b."""
    s, t = np.meshgrid(np.linspace(0, 1, na), np.linspace(0, 1, nb))
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = np.asarray(origin, float) + s[..., None] * a + t[..., None] * b
    uv = np.stack([s * np.linalg.norm(a), t * np.linalg.norm(b)], -1)
    n = v.reshape(-1, 3).shape[0]
    return Part(
        v.reshape(-1, 3),
        grid_triangles(nb, na),
        obj,
        mat,
        uv.reshape(-1, 2),
        np.tile(unit_rows(b), (n, 1)),
        np.tile(unit_rows(a), (n, 1)),
        False,
    )


def room_parts():
    top = plane(
        (-80, 0, WALL_Z),
        (160, 0, 0),
        (0, 0, TABLE_FRONT - WALL_Z),
        160,
        80,
        O_TABLE,
        M_TABLE,
    )
    front = plane(
        (-80, 0, TABLE_FRONT), (160, 0, 0), (0, -4.5, 0), 160, 6, O_TABLE, M_TABLE
    )
    wall = plane((-110, -10, WALL_Z), (220, 0, 0), (0, 90, 0), 110, 45, O_WALL, M_WALL)
    d1, d2 = unit(1, 1.3, 0), unit(1, -0.8, 0)
    wall.t1, wall.t2 = np.tile(d1, (len(wall.v), 1)), np.tile(d2, (len(wall.v), 1))
    return [top, front, wall]


def scene_parts():
    return (
        room_parts()
        + [cloth_part()]
        + jug_parts()
        + [bottle_part(), bowl_part()]
        + cherries()
        + apple_parts()
        + half_apple_parts()
        + knife_parts()
    )


# ------------------------------------------------------------- the camera


@dataclass
class Camera:
    """A pinhole camera at ``eye`` looking at ``target``; ``focal`` in
    pixels, the image ``size`` (height, width)."""

    eye: np.ndarray
    target: np.ndarray
    focal: float
    size: tuple
    centre: tuple = (0.0, 0.0)
    basis: np.ndarray = field(init=False)

    def __post_init__(self):
        f = unit(*(np.asarray(self.target) - self.eye))
        r = unit(*np.cross(f, [0.0, 1.0, 0.0]))
        self.basis = np.stack([r, np.cross(r, f), f])

    def project(self, p):
        """Pixel x, y and the distance along the view of points (n, 3)."""
        c = (np.asarray(p) - self.eye) @ self.basis.T
        d = np.maximum(c[..., 2], 1e-6)
        h, w = self.size
        x = w / 2 + self.centre[0] + self.focal * c[..., 0] / d
        y = h / 2 + self.centre[1] - self.focal * c[..., 1] / d
        return x, y, d

    def scaled(self, k):
        return Camera(
            self.eye,
            self.target,
            self.focal * k,
            (self.size[0] * k, self.size[1] * k),
            (self.centre[0] * k, self.centre[1] * k),
        )


CAMERA = Camera(
    np.array([2.0, 29.0, 96.0]),
    np.array([1.0, 8.0, -2.0]),
    1950.0,
    (HEIGHT, WIDTH),
    (-20.0, -5.0),
)


def _raster(px, py, pz, tris, depth, tid, bary):  # pragma: no cover
    h, w = depth.shape
    for t in range(tris.shape[0]):
        i, j, k = tris[t, 0], tris[t, 1], tris[t, 2]
        x0, x1, x2 = px[i], px[j], px[k]
        y0, y1, y2 = py[i], py[j], py[k]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xa = max(int(math.floor(min(x0, x1, x2))), 0)
        xb = min(int(math.ceil(max(x0, x1, x2))), w - 1)
        ya = max(int(math.floor(min(y0, y1, y2))), 0)
        yb = min(int(math.ceil(max(y0, y1, y2))), h - 1)
        for yy in range(ya, yb + 1):
            cy = yy + 0.5
            for xx in range(xa, xb + 1):
                cx = xx + 0.5
                l0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
                l1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
                l2 = 1.0 - l0 - l1
                if min(l0, l1, l2) < -1e-9:
                    continue
                z = l0 * pz[i] + l1 * pz[j] + l2 * pz[k]
                if z > depth[yy, xx]:
                    depth[yy, xx] = z
                    tid[yy, xx] = t
                    bary[yy, xx, 0] = l0
                    bary[yy, xx, 1] = l1
                    bary[yy, xx, 2] = l2


@dataclass
class Frame:
    """What one raster pass sees at each pixel."""

    depth: np.ndarray
    tid: np.ndarray
    bary: np.ndarray
    seen: np.ndarray = field(init=False)

    def __post_init__(self):
        self.seen = self.tid >= 0


def raster_xy(px, py, pz, f, size) -> Frame:
    """Rasterise triangles given pixel x, y and a depth that grows towards
    the viewer."""
    h, w = size
    depth = np.full((h, w), -np.inf)
    tid = np.full((h, w), -1, dtype=np.int64)
    bary = np.zeros((h, w, 3))
    _raster(px, py, np.ascontiguousarray(pz), f, depth, tid, bary)
    return Frame(depth, tid, bary)


def interpolate(attr, f, frame: Frame):
    """Per-pixel value of a per-vertex attribute (n, k)."""
    t = np.where(frame.seen, frame.tid, 0)
    corners = attr[f[t]]  # h, w, 3, k
    return np.einsum("hwc,hwck->hwk", frame.bary, corners)


@dataclass
class Scene:
    """All the parts merged: points, triangles, per-triangle object and
    material, per-point normals, material coordinates and tangents."""

    v: np.ndarray
    f: np.ndarray
    obj: np.ndarray
    mat: np.ndarray
    n: np.ndarray
    uv: np.ndarray
    t1: np.ndarray
    t2: np.ndarray


def merge(parts) -> Scene:
    vs, fs, objs, mats, ns, uvs, t1s, t2s = ([] for _ in range(8))
    count = 0
    for p in parts:
        vs.append(p.v)
        fs.append(p.f + count)
        objs.append(np.full(len(p.f), p.obj))
        mats.append(np.broadcast_to(p.mat, (len(p.f),)))
        n = vertex_normals(p.v, p.f)
        ns.append(n)
        uvs.append(p.uv)
        t1s.append(p.t1)
        t2s.append(p.t2)
        count += len(p.v)
    cat = np.concatenate
    return Scene(
        np.vstack(vs),
        np.vstack(fs),
        cat(objs),
        cat(mats),
        np.vstack(ns),
        np.vstack(uvs),
        np.vstack(t1s),
        np.vstack(t2s),
    )


@dataclass
class Buffers:
    """Per pixel: what is seen, where, which way it faces and the material
    coordinates and tangents there."""

    seen: np.ndarray
    obj: np.ndarray
    mat: np.ndarray
    dist: np.ndarray
    pos: np.ndarray
    normal: np.ndarray
    uv: np.ndarray
    t1: np.ndarray
    t2: np.ndarray


def see(sc: Scene, cam: Camera) -> Buffers:
    x, y, d = cam.project(sc.v)
    fr = raster_xy(x, y, 1.0 / d, sc.f, cam.size)
    t = np.maximum(fr.tid, 0)
    pos = interpolate(sc.v, sc.f, fr)
    n = unit_rows(interpolate(sc.n, sc.f, fr))
    to_eye = unit_rows(cam.eye - pos)
    n = n * np.where(np.sum(n * to_eye, axis=-1, keepdims=True) < 0, -1, 1)
    obj = np.where(fr.seen, sc.obj[t], -1)
    mat = np.where(fr.seen, sc.mat[t], -1)
    dist = np.where(fr.seen, 1.0 / np.maximum(fr.depth, 1e-9), np.inf)
    uv = interpolate(sc.uv, sc.f, fr)
    t1 = interpolate(sc.t1, sc.f, fr)
    t2 = interpolate(sc.t2, sc.f, fr)
    return Buffers(fr.seen, obj, mat, dist, pos, n, uv, t1, t2)


# --------------------------------------------------------------- the light

KEY = unit(-0.66, 0.5, 0.52)  # towards the window: upper left, in front
FILL = unit(0.8, 0.25, 0.55)
SHADOW_SCALE = 12.0  # pixels per cm of the shadow map


def light_basis(w):
    u = np.cross([0.0, 1.0, 0.0], w)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(w, u), w])


@njit(cache=False)
def _blocked(depth, x, y, z, bias):  # pragma: no cover
    h, w = depth.shape
    xi, yi = int(x), int(y)
    if 0 <= xi < w and 0 <= yi < h:
        return depth[yi, xi] - z - bias
    return -1.0


@njit(cache=False)
def _soft_shadow(sx, sy, sz, grad, depth, disc, search, angle, out):  # pragma: no cover
    """Percentage-closer soft shadows: find the blockers round each point,
    then filter over a penumbra that widens with their distance.  Each
    sample is compared with the plane of the receiving surface, ``grad``
    being its slope in depth per pixel."""
    k = disc.shape[0]
    for i in range(sx.shape[0]):
        bias = 0.04 + 0.8 * (abs(grad[i, 0]) + abs(grad[i, 1]))
        nb, acc = 0, 0.0
        for j in range(k):
            dx, dy = disc[j, 0] * search, disc[j, 1] * search
            z = sz[i] + grad[i, 0] * dx + grad[i, 1] * dy
            d = _blocked(depth, sx[i] + dx, sy[i] + dy, z, bias)
            if d > 0:
                nb += 1
                acc += d
        if nb == 0:
            out[i] = 1.0
            continue
        pen = max(acc / nb * angle, 1.0)
        lit = 0
        for j in range(k):
            dx, dy = disc[j, 0] * pen, disc[j, 1] * pen
            z = sz[i] + grad[i, 0] * dx + grad[i, 1] * dy
            if _blocked(depth, sx[i] + dx, sy[i] + dy, z, bias) <= 0:
                lit += 1
        out[i] = lit / k


def disc_samples(k=48):
    i = np.arange(k) + 0.5
    r = np.sqrt(i / k)
    a = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def shadows(sc: Scene, b: Buffers, softness=0.05):
    """How much of the key light reaches each pixel's point (0 to 1)."""
    basis = light_basis(KEY)
    casts = ~np.isin(sc.obj, (O_WALL, O_TABLE))
    lv = sc.v @ basis.T
    lo, hi = lv[:, :2].min(0) - 1, lv[:, :2].max(0) + 1
    size = (
        int((hi[1] - lo[1]) * SHADOW_SCALE) + 2,
        int((hi[0] - lo[0]) * SHADOW_SCALE) + 2,
    )
    px = (lv[:, 0] - lo[0]) * SHADOW_SCALE
    py = (hi[1] - lv[:, 1]) * SHADOW_SCALE
    depth = raster_xy(px, py, lv[:, 2], sc.f[casts], size).depth
    p = b.pos[b.seen] @ basis.T
    nl = b.normal[b.seen] @ basis.T
    nw = np.clip(nl[:, 2], 0.25, None)
    grad = np.stack([-nl[:, 0] / nw, nl[:, 1] / nw], axis=1) / SHADOW_SCALE
    out = np.ones(len(p))
    _soft_shadow(
        (p[:, 0] - lo[0]) * SHADOW_SCALE,
        (hi[1] - p[:, 1]) * SHADOW_SCALE,
        p[:, 2],
        grad,
        depth,
        disc_samples(),
        0.4 * SHADOW_SCALE,
        softness * SHADOW_SCALE,
        out,
    )
    full = np.ones(b.seen.shape)
    full[b.seen] = out
    return full


def occlusion(b: Buffers, cam: Camera):
    """Ambient occlusion from the depth buffer (in cm, towards the viewer):
    how much nearer surface there is round each point, over a few sizes."""
    z = np.where(b.seen, -b.dist, -1e3)
    occ = np.zeros_like(z)
    px_per_cm = cam.focal / 100.0
    for r_cm, wgt in ((0.25, 0.5), (0.8, 0.6), (2.0, 0.6), (5.0, 0.4)):
        bl = filters.box_blur(z, max(int(r_cm * px_per_cm), 1))
        d = np.clip(bl - z, 0, None)
        occ += (
            wgt
            * np.clip(d / r_cm, 0, 1)
            * np.exp(-np.clip(d - r_cm, 0, None) / (2 * r_cm))
        )
    return np.clip(1 - 0.4 * occ, 0.2, 1)


# ---------------------------------------------------- colours and patterns

# Regions of the picture, as stored in the region map.
REGIONS = [
    "paper", "wall", "table", "cloth", "stripe", "jug", "jug_blue", "jug_inside",
    "bottle", "label", "label_ink", "bowl", "cherry", "stem", "apple", "flesh",
    "core", "seed", "blade", "handle",
]  # fmt: skip
R = {name: k for k, name in enumerate(REGIONS)}
COLOURS = {
    "paper": "#fbfaf7",
    "wall": "#8b92aa",
    "table": "#5c2a26",
    "cloth": "#f2f2f4",
    "stripe": "#2f4a9a",
    "jug": "#f3f3f5",
    "jug_blue": "#23388a",
    "jug_inside": "#e9e9ed",
    "bottle": "#14171f",
    "label": "#f6f4ee",
    "label_ink": "#2a2a30",
    "bowl": "#f4f4f6",
    "cherry": "#7c0e1e",
    "stem": "#3c2216",
    "apple": "#b3192b",
    "flesh": "#f7f2e6",
    "core": "#ece2cc",
    "seed": "#2b1a12",
    "blade": "#b8bfcc",
    "handle": "#1a1a1e",
}
#: Gloss of each region: (strength, sharpness).
GLOSS = {
    "jug": (0.5, 120),
    "jug_blue": (0.5, 120),
    "jug_inside": (0.2, 60),
    "bowl": (0.45, 110),
    "bottle": (1.2, 500),
    "label": (0.03, 10),
    "cherry": (0.8, 180),
    "stem": (0.05, 20),
    "apple": (0.3, 60),
    "flesh": (0.06, 20),
    "core": (0.06, 20),
    "blade": (0.9, 90),
    "handle": (0.35, 70),
    "table": (0.04, 12),
}
#: Base materials' regions.
MAT_REGION = {
    M_WALL: "wall",
    M_TABLE: "table",
    M_CLOTH: "cloth",
    M_JUG: "jug",
    M_JUG_IN: "jug_inside",
    M_BOTTLE: "bottle",
    M_BOWL: "bowl",
    M_CHERRY: "cherry",
    M_STEM: "stem",
    M_APPLE: "apple",
    M_FLESH: "flesh",
    M_BLADE: "blade",
    M_HANDLE: "handle",
    M_JUG_HANDLE: "jug",
}


def profile_height(points, n, s):
    """Height on a turned profile at distance ``s`` along it."""
    prof = catmull_rom(np.asarray(points, float), n)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    return np.interp(s, d, prof[:, 1])


def band(y, lo, hi, soft=0.04):
    return smoothstep(lo - soft, lo + soft, y) * (
        1 - smoothstep(hi - soft, hi + soft, y)
    )


def jug_pattern(theta, y):
    """The painted blue: lines at foot, belly, neck and rim, and round the
    belly a vine with leaves, lying on the glaze."""
    lines = [
        (1.25, 1.55),
        (6.15, 6.45),
        (6.75, 6.9),
        (11.1, 11.25),
        (11.55, 11.85),
        (16.3, 16.5),
        (20.9, 21.3),
    ]
    out = np.zeros_like(y)
    for lo, hi in lines:
        out = np.maximum(out, band(y, lo, hi))
    vine = 9.0 + 1.0 * np.sin(7 * theta)
    out = np.maximum(out, smoothstep(0.17, 0.11, np.abs(y - vine)))
    k = np.round(theta * 14 / (2 * np.pi) - 0.25)
    tk = (k + 0.25) * 2 * np.pi / 14
    side = np.where(k % 2 == 0, 1.0, -1.0)
    leaf_y = 9.0 + 1.0 * np.sin(7 * tk) + side * 0.95
    dx, dy = (theta - tk) * 7.8 / 0.62, (y - leaf_y) / 0.36
    return np.maximum(out, smoothstep(1.1, 0.9, np.hypot(dx, dy)))


def label_pattern(theta, y):
    """The label: a paper band on the bottle, with a frame and lines of
    lettering; returns (label, ink)."""
    label = band(theta, -1.2, 0.95, 0.02) * band(y, 7.0, 14.5, 0.03)
    ink = band(y, 13.3, 13.5) + band(y, 7.3, 7.45)
    ink += band(y, 11.4, 12.4) * band(theta, -0.75, 0.45, 0.02)
    for lo, hi, a, b in (
        (9.9, 10.15, -0.6, 0.3),
        (9.0, 9.2, -0.45, 0.15),
        (8.2, 8.35, -0.3, 0.05),
    ):
        ink += band(y, lo, hi) * band(theta, a, b, 0.02)
    return label, np.clip(ink, 0, 1) * label


def stripes(u_cm, v_cm):
    """The cloth's blue stripes: a band of three across it every so often,
    and two lines along its free edge."""
    p = np.mod(v_cm - 15.0, 17.0)
    across = (
        band(p, 0.0, 1.3, 0.05) + band(p, 2.0, 2.35, 0.05) + band(p, 3.0, 3.35, 0.05)
    )
    along = band(u_cm, 2.0, 2.6, 0.05) + band(u_cm, 3.2, 3.45, 0.05)
    return np.clip(across + along, 0, 1)


def flesh_pattern(across_s, y):
    """The cut face: the skin's rim, the core and two seeds."""
    rim = smoothstep(0.93, 0.98, np.abs(across_s))
    core = smoothstep(1.05, 0.95, np.hypot((y - 3.9) / 2.3, across_s / 0.32))
    seeds = np.zeros_like(y)
    for sx in (-0.11, 0.11):
        seeds = np.maximum(
            seeds,
            smoothstep(1.1, 0.9, np.hypot((y - 3.6) / 0.55, (across_s - sx) / 0.06)),
        )
    return rim, core, seeds


def around(b: Buffers, at, yaw=0.0):
    """Angle round a vertical axis at (x, z), and height above the cloth."""
    th = np.arctan2(b.pos[..., 0] - at[0], b.pos[..., 2] - at[1]) - yaw
    return np.angle(np.exp(1j * th)), b.pos[..., 1] - CLOTH_Y


def regions(b: Buffers):
    """The region of each pixel, from the materials and their patterns."""
    names = np.array(["paper"] + [MAT_REGION[m] for m in range(len(MAT_REGION))])
    region = np.vectorize(R.get)(names[b.mat + 1]).astype(np.int16)
    th, y = around(b, JUG_AT, JUG_YAW)
    paint = (b.mat == M_JUG) & (jug_pattern(th, y) > 0.5)
    region[paint] = R["jug_blue"]
    th, y = around(b, BOTTLE_AT)
    label, ink = label_pattern(th, y)
    region[(b.mat == M_BOTTLE) & (label > 0.5)] = R["label"]
    region[(b.mat == M_BOTTLE) & (ink > 0.5)] = R["label_ink"]
    region[(b.mat == M_CLOTH) & (stripes(b.uv[..., 0], b.uv[..., 1]) > 0.5)] = R[
        "stripe"
    ]
    rim, core, seeds = flesh_pattern(b.uv[..., 0], b.uv[..., 1])
    flesh = b.mat == M_FLESH
    region[flesh & (core > 0.5)] = R["core"]
    region[flesh & (seeds > 0.5)] = R["seed"]
    region[flesh & (rim > 0.5)] = R["apple"]
    return region


def albedo(region, b: Buffers):
    out = np.zeros(region.shape + (3,))
    for name, hexc in COLOURS.items():
        out[region == R[name]] = to_linear(hexc)
    wood = 0.8 + 0.4 * noise2(b.pos[..., 0] / 0.4, b.pos[..., 2] / 6.0, 11)
    seam = 1 - 0.5 * smoothstep(
        0.25, 0.05, np.abs(np.mod(b.pos[..., 0] + 6.5, 13.0) - 6.5)
    )
    out[region == R["table"]] *= (wood * seam)[region == R["table"]][:, None]
    skin = region == R["apple"]
    streak = 0.75 + 0.45 * noise2(b.uv[..., 0] * 9, b.uv[..., 1] * 1.2, 4)
    out[skin] *= streak[skin][:, None]
    return out


@dataclass
class Light:
    diffuse: np.ndarray  # light falling on each point
    spec: np.ndarray  # light reflected off it
    shadow: np.ndarray
    ao: np.ndarray


def lighting(b: Buffers, region, shadow, ao, cam: Camera) -> Light:
    n = b.normal
    to_eye = unit_rows(cam.eye - b.pos)
    key = np.clip(n @ KEY, 0, 1) * shadow
    paper = region == R["label"]  # matt: an even light, a little less on the far side
    key[paper] = (0.55 + 0.3 * np.clip(n @ KEY, 0, 1) * shadow)[paper]
    fill = 0.16 * np.clip(n @ FILL, 0, 1) * ao
    amb = 0.1 * ao * (0.55 + 0.45 * n[..., 1])
    thing = ~np.isin(
        region, [R[k] for k in ("wall", "table", "cloth", "stripe", "paper")]
    )
    bounce = 0.16 * np.clip(0.35 - n[..., 1], 0, 1) * ao * thing
    half = unit_rows(KEY + to_eye)
    ndh = np.clip(np.sum(n * half, axis=-1), 0, 1)
    spec = np.zeros(ndh.shape)
    for name, (k, p) in GLOSS.items():
        m = region == R[name]
        spec[m] = k * ndh[m] ** p * shadow[m]
    glass = region == R["bottle"]
    rim = (1 - np.clip(np.sum(n * to_eye, axis=-1), 0, 1)) ** 3
    spec[glass] += 0.12 * rim[glass] + 0.35 * ndh_fill(n, to_eye)[glass] ** 200
    return Light(key + fill + amb + bounce, spec, shadow, ao)


def ndh_fill(n, to_eye):
    return np.clip(np.sum(n * unit_rows(FILL + to_eye), axis=-1), 0, 1)


def counterchange(b: Buffers, region, lum):
    """Darken the wall behind the lit sides of things and lighten it behind
    their shadow sides, so each stands out from its ground."""
    thing = ~np.isin(
        region, [R[k] for k in ("wall", "table", "cloth", "stripe", "paper")]
    )
    lit = filters.box_blur((thing & (lum > 0.35)).astype(float), 45)
    dark = filters.box_blur((thing & (lum < 0.12)).astype(float), 45)
    return np.clip(1 - 0.9 * lit + 0.7 * dark, 0.55, 1.4)


def blobs(shape, size, seed):
    """Smooth random blobs about ``size`` pixels across, 0 to 1."""
    rng = np.random.default_rng(seed)
    out = rng.random(shape)
    for _ in range(3):
        out = filters.box_blur(out, max(int(size / 2), 1))
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo) / (hi - lo), 0, 1)


def vignette(shape, seed=2):
    """1 inside the drawing, falling irregularly to 0 towards the edges."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - 0.5 * w) / (0.48 * w), (yy - 0.53 * h) / (0.54 * h))
    k = w / WIDTH
    wobble = 0.75 * blobs(shape, 160 * k, seed) + 0.25 * blobs(shape, 45 * k, seed + 1)
    return 1 - smoothstep(0.84, 1.0, d + 0.22 * (wobble - 0.5))


# --------------------------------------------------------------- the study


@dataclass
class Picture:
    b: Buffers
    region: np.ndarray
    light: Light
    colour: np.ndarray  # linear RGB
    raw: np.ndarray  # the same before the edges fade
    keep: np.ndarray  # where the drawing reaches: 1 inside, 0 beyond


def picture(sc: Scene, cam: Camera) -> Picture:
    b = see(sc, cam)
    region = regions(b)
    shadow = shadows(sc, b)
    wall = b.obj == O_WALL  # the wall far behind: its shadows broad and faint
    soft = filters.box_blur(np.where(wall, shadow, 1.0), int(cam.focal / 60))
    shadow = np.where(wall, 0.55 + 0.45 * soft, shadow)
    ao = occlusion(b, cam)
    lt = lighting(b, region, shadow, ao, cam)
    base = albedo(region, b)
    colour = base * lt.diffuse[..., None] + lt.spec[..., None]
    lum = colour @ np.array([0.2126, 0.7152, 0.0722])
    wall = (region == R["wall"])[..., None]
    xs = np.arange(region.shape[1])[None, :] / region.shape[1]
    grade = (1.15 - 0.45 * xs)[..., None] * counterchange(b, region, lum)[..., None]
    colour = np.where(wall, colour * grade, colour)
    keep = vignette(region.shape)[..., None]
    thing = ~np.isin(
        region, [R[k] for k in ("wall", "table", "cloth", "stripe", "paper")]
    )
    keep = np.maximum(keep, thing[..., None])
    paper = to_linear(COLOURS["paper"])
    raw = colour
    colour = np.where(b.seen[..., None], colour * keep + paper * (1 - keep), paper)
    return Picture(b, region, lt, colour, raw, keep[..., 0])


def grey(lin):
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    return np.repeat(y[..., None], 3, axis=-1)


def scaled_picture(sc: Scene, k=2) -> Picture:
    """The picture rendered k times larger, its colour shrunk back."""
    pic = picture(sc, CAMERA.scaled(k))
    pic.colour = shrink(pic.colour, k)
    return pic


CROPS = {
    "jug": (300, 250, 700, 750),
    "fruit": (560, 520, 1060, 900),
    "cloth": (150, 650, 750, 1050),
}


def write_studies(pic: Picture, out: Path, k=2):
    write_png(str(out / "study_colour.png"), as_png(pic.colour))
    write_png(str(out / "study_grey.png"), as_png(grey(pic.colour)))
    for name, (x0, y0, x1, y1) in CROPS.items():
        crop = pic.colour[y0:y1, x0:x1]
        big = np.repeat(np.repeat(crop, k, 0), k, 1)
        write_png(str(out / f"crop_{name}.png"), as_png(big))


# ------------------------------------------------- what the pen will follow


def picture_directions(b: Buffers, cam: Camera, t, eps=0.05):
    """A per-pixel tangent (h, w, 3) as a unit direction in the picture."""
    x0, y0, _ = cam.project(b.pos)
    x1, y1, _ = cam.project(b.pos + eps * t)
    d = np.stack([x1 - x0, y1 - y0], -1)
    return np.where(b.seen[..., None], unit_rows(d), 0.0)


RING = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


def trace(mask, start, limit=400000):
    """The boundary of the shape in ``mask`` through ``start`` (a pixel
    whose west neighbour is outside), followed clockwise (Moore)."""
    h, w = mask.shape
    path = [start]
    p, back = start, 0
    for _ in range(limit):
        for i in range(1, 9):
            k = (back + i) % 8
            x, y = p[0] + RING[k][0], p[1] + RING[k][1]
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                back = (k + 4) % 8
                p = (x, y)
                break
        else:
            return path
        if p == start:
            return path
        path.append(p)
    return path


def outlines(mask, min_len=40, most=12):
    """The boundaries of the larger shapes of a mask, as polylines (x, y)."""
    starts = mask & ~np.roll(mask, 1, axis=1)
    starts[:, 0] = mask[:, 0]
    done = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(starts)
    for x, y in zip(xs, ys, strict=True):
        if done[y, x]:
            continue
        path = np.array(trace(mask, (int(x), int(y))))
        done[path[:, 1], path[:, 0]] = True
        if len(path) >= min_len:
            out.append(path[::2].astype(float) + 0.5)
    out.sort(key=len, reverse=True)
    return out[:most]


def seen_line(cam: Camera, dist, pts, tol=0.4):
    """A 3D polyline in the picture: (k, 3) x, y and whether it is seen."""
    x, y, d = cam.project(pts)
    h, w = dist.shape
    xi, yi = np.clip(x.astype(int), 0, w - 1), np.clip(y.astype(int), 0, h - 1)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    seen = inside & (d <= dist[yi, xi] + tol)
    return np.stack([x, y, seen.astype(float)], 1)


def rims():
    """Rims of the turned things as 3D rings."""
    th = np.linspace(0, 2 * np.pi, 241)
    out = {}
    for name, (r, y), at, yaw in (
        ("rim_jug", (5.45, 22.25), JUG_AT, JUG_YAW),
        ("rim_bowl", (10.6, 5.4), BOWL_AT, 0.0),
        ("rim_bottle", (1.72, 31.5), BOTTLE_AT, 0.0),
        ("foot_jug", (4.5, 0.3), JUG_AT, JUG_YAW),
        ("foot_bowl", (3.9, 0.3), BOWL_AT, 0.0),
    ):
        rr, yy = r * np.ones_like(th), y * np.ones_like(th)
        if name == "rim_jug":
            rr, yy = spout(th, rr, yy)
        ring = np.stack([rr * np.sin(th), yy, rr * np.cos(th)], 1)
        out[name] = place(ring, (at[0], CLOTH_Y, at[1]), yaw)
    return out


def ridges():
    """The crests of the cloth's folds as 3D lines on it."""
    out = {}
    for k, (line, _, _) in enumerate(fold_lines(80)):
        x, z = line[:, 0], line[:, 1]
        lift, _ = folds_at(x, z)
        y = CLOTH_Y + lift * flat_under(x, z)
        out[f"fold_{k}"] = np.stack([x, y + 0.05, z], 1)
    return out


def contours(pic: Picture, cam: Camera, k):
    """Silhouettes of the objects, rims, folds and cast-shadow edges, as
    polylines in picture pixels: {name: (n, 3) x, y, seen}."""
    obj, region = pic.b.obj[::k, ::k], pic.region[::k, ::k]
    obj = np.where(obj >= O_CHERRY, O_CHERRY, obj)
    out = {}
    names = {
        O_JUG: "jug",
        O_BOTTLE: "bottle",
        O_BOWL: "bowl",
        O_APPLE: "apple",
        O_HALF: "half_apple",
        O_KNIFE: "knife",
        O_CHERRY: "cherries",
    }
    for o, name in names.items():
        for i, line in enumerate(outlines(obj == o, 20, 40)):
            out[f"outline_{name}_{i}"] = np.hstack([line, np.ones((len(line), 1))])
    dist = pic.b.dist[::k, ::k]
    small = Camera(
        cam.eye,
        cam.target,
        cam.focal / k,
        (cam.size[0] // k, cam.size[1] // k),
        (cam.centre[0] / k, cam.centre[1] / k),
    )
    for name, pts in (rims() | ridges()).items():
        out[name] = seen_line(small, dist, pts)
    ground = np.isin(region, [R[n] for n in ("cloth", "stripe", "table", "wall")])
    cast = ground & (pic.light.shadow[::k, ::k] < 0.5)
    for i, line in enumerate(outlines(cast, 30, 30)):
        out[f"shadow_{i}"] = np.hstack([line, np.ones((len(line), 1))])
    return out


def write_maps(pic: Picture, sc: Scene, out: Path, k=2):
    """Per-pixel maps at the picture's size, and the contours."""
    b, cam = pic.b, CAMERA.scaled(k)
    d1 = shrink(picture_directions(b, cam, b.t1), k)
    d2 = shrink(picture_directions(b, cam, b.t2), k)
    lum = pic.colour @ np.array([0.2126, 0.7152, 0.0722])
    np.savez_compressed(
        out / "maps.npz",
        obj=b.obj[::k, ::k].astype(np.int16),
        region=pic.region[::k, ::k].astype(np.int16),
        region_names=np.array(REGIONS),
        tone=lum.astype(np.float32),
        light=shrink(pic.light.diffuse, k).astype(np.float32),
        shadow=shrink(pic.light.shadow, k).astype(np.float32),
        highlight=shrink(pic.light.spec, k).astype(np.float32),
        normal=shrink(np.where(b.seen[..., None], b.normal, 0), k).astype(np.float32),
        dir1=unit_rows(d1).astype(np.float32),
        dir2=unit_rows(d2).astype(np.float32),
        depth=np.where(b.seen, b.dist, np.nan)[::k, ::k].astype(np.float32),
    )
    np.savez_compressed(out / "contours.npz", **contours(pic, cam, k))
    del sc


# ------------------------------------------------------------ the pen work

PAPER = papers.BRISTOL.with_(color="#fbfaf7")
RED = pens.ballpoint(pens.RED.ink.hex, width=2.0)
BLUE = pens.ballpoint(pens.BLUE.ink.hex, width=2.0)
BLACK = pens.ballpoint(pens.BLACK.ink.hex, width=2.0)
BOLD = pens.BLACK  # the 3 px ball, for the strongest lines
CELL = 4  # pixels per cell of the maps that count the ink laid
WHITES = ("jug", "jug_inside", "bowl", "cloth", "label", "flesh", "core")
FRUIT = ("cherry", "apple")


@dataclass
class PenMaps:
    """What the pen work reads off the study, per pixel of the picture."""

    obj: np.ndarray  # int: each object (each cherry its own)
    region: np.ndarray  # int: REGIONS
    shade: np.ndarray  # 0 in full light to 1 turned from it or in shadow
    dark: np.ndarray  # the study's darkness, 0 white to 1 black
    spec: np.ndarray  # the highlight
    dir1: np.ndarray  # (h, w, 2): round the form, along the folds or boards
    dir2: np.ndarray  # (h, w, 2): along the profile, across the folds
    keep: np.ndarray  # where the drawing reaches
    lines: dict  # contours: (n, 3) x, y, seen
    stems: list  # stems of the cherries and the apple, in pixels
    dist: np.ndarray  # distance from the eye of what is seen (cm)


def stem_lines(parts, cam: Camera, dist):
    """The centre lines of the stems, as seen polylines in the picture."""
    out = []
    for p in parts:
        if np.isscalar(p.mat) and p.mat == M_STEM:
            ring = 8 if p.obj != O_APPLE else 10
            centre = p.v.reshape(-1, ring, 3).mean(axis=1)
            out.append(seen_line(cam, dist, centre, 0.3))
    return out


def pen_maps(parts, sc: Scene) -> PenMaps:
    pic = picture(sc, CAMERA)
    b, cam = pic.b, CAMERA
    lum = pic.raw @ np.array([0.2126, 0.7152, 0.0722])
    shade = np.clip(1 - pic.light.diffuse / 0.95, 0, 1)
    return PenMaps(
        np.where(b.obj >= O_CHERRY, b.obj, b.obj).astype(np.int32),
        pic.region.astype(np.int32),
        shade.astype(np.float32),
        (1 - to_srgb(lum)).astype(np.float32),
        pic.light.spec.astype(np.float32),
        picture_directions(b, cam, b.t1).astype(np.float32),
        picture_directions(b, cam, b.t2).astype(np.float32),
        pic.keep.astype(np.float32),
        contours(pic, cam, 1),
        stem_lines(parts, cam, b.dist),
        b.dist.astype(np.float32),
    )


@njit(cache=False)
def _trace(
    x, y, dx, dy, field, part, want, target, tmin, turn, step, nmax, out, full
):  # pragma: no cover
    """Follow the direction field from (x, y), setting off along (dx, dy)
    and turned by ``turn`` (radians) from the field, while on the part
    ``want``, where the target is at least ``tmin`` and no other line of
    the layer is near (``full``).  Writes the points to ``out`` and returns
    how many."""
    h, w = part.shape
    c, s = math.cos(turn), math.sin(turn)
    crowd, n = 0, 0
    for _ in range(nmax):
        xi, yi = int(x), int(y)
        if xi < 0 or yi < 0 or xi >= w or yi >= h or part[yi, xi] != want:
            break
        if target[yi, xi] < tmin:
            break
        crowd = crowd + 1 if full[yi, xi] else 0
        if crowd > 6:
            break
        out[n, 0], out[n, 1] = x, y
        n += 1
        fx, fy = field[yi, xi, 0], field[yi, xi, 1]
        gx, gy = fx * c - fy * s, fx * s + fy * c
        if gx * dx + gy * dy < 0:
            gx, gy = -gx, -gy
        dx, dy = 0.75 * dx + 0.25 * gx, 0.75 * dy + 0.25 * gy
        norm = math.sqrt(dx * dx + dy * dy) + 1e-12
        dx, dy = dx / norm, dy / norm
        x += dx * step
        y += dy * step
    return n


@njit(cache=False)
def _stamp(occ, xs, ys, radius):  # pragma: no cover
    """Mark the pixels within ``radius`` of the points: the room a line
    keeps from the next one of its layer."""
    h, w = occ.shape
    r = int(math.ceil(radius))
    r2 = radius * radius
    for i in range(xs.shape[0]):
        cx, cy = xs[i], ys[i]
        for y in range(max(int(cy) - r, 0), min(int(cy) + r + 1, h)):
            for x in range(max(int(cx) - r, 0), min(int(cx) + r + 1, w)):
                if (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r2:
                    occ[y, x] = 1


def line_weight(p):
    """How much a line of the 2 px ball darkens, times its width (px), at
    pressure ``p``: measured on hatching."""
    return 1.25 * np.asarray(p) ** 0.55


@dataclass
class Pass:
    """One layer of hatching: an ink over the pixels where the target
    asks for it, at an angle to the form's first direction, up to ``cap``
    of coverage beyond ``floor``.  Strokes stay on the object (or, with
    ``by_region``, the region) where they start."""

    pen: pens.Pen
    angle: float  # degrees from dir1 towards dir2
    floor: float
    cap: float
    pressure: float
    length: tuple = (60, 200)
    count: tuple = (5, 12)
    spacing: tuple = (3.0, 13.0)
    shortest: int = 14  # points of 2 px
    by_region: bool = False


@dataclass
class Hatcher:
    """Lays hatching patch by patch on a sheet, keeping count of the ink."""

    sheet: Sheet
    maps: PenMaps
    rng: np.random.Generator
    strokes: int = 0
    full: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.uint8))

    def field(self, angle):
        a = math.radians(angle)
        d = math.cos(a) * self.maps.dir1 + math.sin(a) * self.maps.dir2
        return unit_rows(d).astype(np.float32)

    def lay(self, ps: Pass, target, limit=3000):
        """Hatch until what is laid comes near ``target`` (coverage) between
        the pass's floor and floor + cap."""
        goal = np.clip(target - ps.floor, 0, ps.cap).astype(np.float32)
        part = self.maps.region if ps.by_region else self.maps.obj
        field = self.field(ps.angle)
        h, w = goal.shape
        coarse = shrink(goal[: h // CELL * CELL, : w // CELL * CELL], CELL)
        laid = np.zeros_like(coarse)
        self.full = np.zeros(goal.shape, np.uint8)
        for _ in range(limit):
            deficit = coarse - laid
            pick = deficit * self.rng.random(deficit.shape) ** 0.35
            i = int(np.argmax(pick))
            cy, cx = divmod(i, coarse.shape[1])
            if deficit[cy, cx] < 0.3 * ps.cap:
                break
            centre = (np.array([cx, cy]) + self.rng.random(2)) * CELL
            if not self.bundle(ps, centre, goal, field, part, laid):
                laid[cy, cx] += ps.cap  # nothing could be laid here

    def bundle(self, ps: Pass, centre, goal, field, part, laid):
        """A bundle of near-parallel strokes across the patch at ``centre``."""
        rng = self.rng
        x, y = int(centre[0]), int(centre[1])
        if goal[y, x] <= 0:
            return False
        turn = math.radians(rng.normal(0, 6))
        fan = math.radians(rng.uniform(-2.0, 2.0))
        c, s = math.cos(turn), math.sin(turn)
        f = field[y, x]
        d = np.array([f[0] * c - f[1] * s, f[0] * s + f[1] * c])
        across = np.array([-d[1], d[0]])
        p = ps.pressure * rng.uniform(0.85, 1.15)
        gap = float(np.clip(line_weight(p) / max(goal[y, x], 1e-3), *ps.spacing))
        n = int(rng.integers(*ps.count))
        length = rng.uniform(*ps.length)
        want = part[y, x]
        laid_any = False
        for k in range(n):
            off = (k - (n - 1) / 2) * gap * rng.uniform(0.85, 1.15)
            start = centre + across * off + d * rng.normal(0, 0.15 * length)
            spread = turn + fan * (k - (n - 1) / 2)
            pts = self.trace(ps, start, d, spread, field, goal, part, want, length)
            if pts is not None:
                self.draw(ps, pts, p, goal, laid)
                _stamp(self.full, pts[:, 0], pts[:, 1], 0.62 * gap)
                laid_any = True
        return laid_any

    def trace(self, ps: Pass, start, d, turn, field, goal, part, want, length):
        half = [length * self.rng.uniform(0.35, 0.65)]
        half.append(length - half[0])
        out = np.zeros((600, 2))
        ways = []
        for sign, ln in zip((1, -1), half, strict=True):
            k = _trace(start[0], start[1], sign * d[0], sign * d[1], field, part,
                       want, goal, 0.12 * ps.cap, turn, 2.0, int(ln / 2), out,
                       self.full)  # fmt: skip
            ways.append(out[:k].copy())
        pts = np.vstack([ways[1][::-1], ways[0][1:]])
        return pts if len(pts) >= ps.shortest else None

    def draw(self, ps: Pass, pts, p, goal, laid):
        """One stroke, tapering at both ends and following the target."""
        u = np.linspace(0, 1, len(pts))
        taper = np.clip(np.sin(np.pi * u) * 1.6, 0, 1) ** 0.7
        xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
        here = goal[yi, xi] / max(ps.cap, 1e-6)
        press = p * taper * np.clip(0.55 + 0.6 * here, 0.5, 1.15)
        keep = slice(None, None, 2 if len(pts) < 20 else 3)
        path = pts[keep]
        if len(path) < 3:
            return
        self.sheet.stroke([tuple(q) for q in path], ps.pen, list(press[keep] + 0.01))
        self.strokes += 1
        cy = np.minimum(yi // CELL, laid.shape[0] - 1)
        cx = np.minimum(xi // CELL, laid.shape[1] - 1)
        np.add.at(laid, (cy, cx), line_weight(press) * 2.0 / CELL**2)


# --------------------------------------------------------------- tone targets


def steps(x, edges, heights):
    """A tone in a few separate steps: ``heights[k]`` added past ``edges[k]``."""
    out = np.zeros_like(x)
    for e, a in zip(edges, heights, strict=True):
        out = out + a * smoothstep(e - 0.05, e + 0.05, x)
    return out


def masks(m: PenMaps, names):
    return np.isin(m.region, [R[n] for n in names])


def near_fruit(m: PenMaps):
    """How near the red fruit each pixel is: where shadows turn violet."""
    fruit = masks(m, FRUIT).astype(float)
    return np.clip(filters.box_blur(fruit, 28) * 3.0, 0, 1)


def white_targets(m: PenMaps):
    """Shadows on white things: open blue in steps, a little black at the
    darkest, violet near the fruit; bare paper in the lights."""
    S = m.shade
    white = masks(m, WHITES + ("stripe", "jug_blue"))
    bare = 1 - smoothstep(0.04, 0.12, m.spec)
    blue = steps(S, (0.22, 0.45, 0.68), (0.04, 0.06, 0.07)) * bare
    red = 0.6 * near_fruit(m) * blue
    black = 0.05 * smoothstep(0.8, 0.95, S)
    k = np.where(white, m.keep, 0)
    return blue * k, red * k, black * k


def fruit_targets(m: PenMaps):
    """Red on the fruit, closer in shadow, with black and a little blue in
    the core of the shadow; the highlight bare paper."""
    S = m.shade
    fruit = masks(m, FRUIT)
    bare = 1 - smoothstep(0.08, 0.25, m.spec)
    red = (0.07 + steps(S, (0.3, 0.6), (0.08, 0.09))) * bare
    black = steps(S, (0.55, 0.8), (0.05, 0.07)) * bare
    blue = 0.04 * smoothstep(0.65, 0.85, S) * bare
    return red * fruit, black * fruit, blue * fruit


def dark_targets(m: PenMaps):
    """The bottle, the knife's handle and the table: black over blue (or
    red for the wood), the highlights left as paper."""
    D = m.dark
    bottle = masks(m, ("bottle", "handle"))
    bare = 1 - smoothstep(0.1, 0.3, m.spec)
    black = np.where(bottle, (0.12 + 0.35 * smoothstep(0.45, 0.9, D)) * bare, 0)
    blue = np.where(bottle, 0.1 + 0.12 * smoothstep(0.3, 0.7, D), 0) * bare
    table = masks(m, ("table",))
    black = black + np.where(table, 0.06 + 0.12 * smoothstep(0.6, 0.9, D), 0) * m.keep
    red = np.where(table, 0.1, 0) * m.keep
    return black, blue, red


def wall_targets(m: PenMaps):
    """The wall in open blue, black where it is darkest; its patches end
    raggedly where the drawing stops."""
    D = m.dark
    wall = masks(m, ("wall",))
    ragged = smoothstep(0.35, 0.6, m.keep)
    blue = np.where(wall, 0.04 + steps(D, (0.45, 0.6), (0.04, 0.05)), 0) * ragged
    black = np.where(wall, 0.05 * smoothstep(0.55, 0.7, D), 0) * ragged
    return blue, black


def small_targets(m: PenMaps):
    """Label lettering, seeds, the blade: black, and blue on the steel."""
    ink = masks(m, ("label_ink", "seed")) * 0.45
    blade = masks(m, ("blade",))
    steel = blade * (0.05 + 0.15 * smoothstep(0.3, 0.7, m.dark)) * (m.spec < 0.3)
    return ink.astype(np.float32), steel.astype(np.float32)


# ----------------------------------------------------- lines on the surfaces


def sample(a, pts):
    xi = np.clip(np.asarray(pts)[:, 0].astype(int), 0, a.shape[1] - 1)
    yi = np.clip(np.asarray(pts)[:, 1].astype(int), 0, a.shape[0] - 1)
    return a[yi, xi]


def seen_runs(line, min_len=3):
    """The runs of a polyline (k, 3: x, y, seen) that are seen."""
    seen = np.append(line[:, 2] > 0.5, False)
    out, start = [], None
    for i, ok in enumerate(seen):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                out.append(line[start:i, :2])
            start = None
    return out


def surface_line(h: Hatcher, pts3d, pen, lo, hi, tol=0.25, fade=True):
    """Draw a line lying on a surface: only where it is seen, its pressure
    from ``lo`` in the light to ``hi`` in shadow, tapering at its ends."""
    m = h.maps
    line = seen_line(CAMERA, m.dist, pts3d, tol)
    for run in seen_runs(line, 4):
        S = sample(m.shade, run)
        k = sample(m.keep, run) if fade else np.ones(len(run))
        u = np.linspace(0, 1, len(run))
        taper = np.clip(np.minimum(u, 1 - u) * len(run) / 6, 0, 1)
        press = (lo + (hi - lo) * S) * taper * smoothstep(0.25, 0.6, k)
        if press.max() < 0.03:
            continue
        step = 2 if len(run) < 40 else 3
        h.sheet.stroke([tuple(q) for q in run[::step]], pen, list(press[::step] + 0.005))


def jug_outside():
    prof = catmull_rom(np.asarray(JUG_PROFILE, float), 320)
    top = int(np.argmax(prof[:, 1]))
    return prof[: top + 1]


def jug_point(theta, y, lift=0.04):
    """A point on the jug's outer surface at angle ``theta`` and height y."""
    prof = jug_outside()
    r = np.interp(y, prof[:, 1], prof[:, 0]) + lift
    r, yy = spout(theta, r, y)
    local = np.stack([r * np.sin(theta), yy, r * np.cos(theta)], -1)
    return place(local, (JUG_AT[0], CLOTH_Y, JUG_AT[1]), JUG_YAW)


JUG_BANDS = [(1.25, 1.55), (6.15, 6.45), (6.75, 6.9), (11.1, 11.25), (11.55, 11.85),
             (16.3, 16.5), (20.9, 21.3)]  # fmt: skip


def draw_jug_pattern(h: Hatcher):
    """The painted blue: each band filled with rings close together, the
    vine as a swelling line, the leaves as small filled ovals."""
    th = np.linspace(-np.pi, np.pi, 720)
    for lo, hi in JUG_BANDS:
        for y in np.arange(lo + 0.03, hi, 0.07):
            surface_line(h, jug_point(th, np.full_like(th, y)), BLUE, 0.3, 0.55)
    for off in (-0.05, 0.0, 0.05):
        surface_line(h, jug_point(th, 9.0 + np.sin(7 * th) + off), BLUE, 0.35, 0.6)
    for k in range(14):
        tk = (k + 0.25) * 2 * np.pi / 14
        side = 1.0 if k % 2 == 0 else -1.0
        cy = 9.0 + np.sin(7 * tk) + side * 0.95
        a = np.linspace(0, 2 * np.pi, 40)
        for scale in (1.0, 0.75, 0.5, 0.25):
            t = tk + scale * 0.6 / 7.8 * np.cos(a)
            y = cy + scale * 0.34 * np.sin(a)
            surface_line(h, jug_point(t, y), BLUE, 0.35, 0.6)


def stripe_bands():
    """The stripes across the cloth: (from, to) along its length, cm."""
    out = []
    for j in range(5):
        base = 15.0 + 17.0 * (j - 1)
        out += [(base, base + 1.3), (base + 2.0, base + 2.35), (base + 3.0, base + 3.35)]
    return [(a, b) for a, b in out if 0 < a and b < cloth_length()]


def cloth_line(u, v):
    length = cloth_length()
    return cloth_point(u, v - cloth_hem(u, v, length)) + [0, 0.03, 0]


def draw_stripes(h: Hatcher):
    """The cloth's stripes as lines along them, close together, bending
    over every fold."""
    u = np.linspace(0, 1, 900)
    for lo, hi in stripe_bands():
        for v in np.arange(lo + 0.04, hi, 0.09):
            surface_line(h, cloth_line(u, np.full_like(u, v)), BLUE, 0.3, 0.55)
    v = np.linspace(0, cloth_length(), 700)
    z = np.minimum(CLOTH_BACK + v, TABLE_FRONT)
    width = CLOTH_RIGHT - np.interp(z, [CLOTH_BACK, TABLE_FRONT], CLOTH_LEFT)
    for lo, hi in ((2.0, 2.6), (3.2, 3.45)):
        for uc in np.arange(lo + 0.04, hi, 0.09):
            surface_line(h, cloth_line(uc / width, v), BLUE, 0.3, 0.55)


# ----------------------------------------------------------- the hatching

LONG = {"length": (90, 240), "count": (5, 11)}
SMALL = {"length": (14, 40), "count": (3, 6), "spacing": (2.5, 7.0), "shortest": 5}
FRUIT_KW = {"length": (40, 110), "count": (4, 9), "spacing": (2.5, 9.0), "shortest": 8}


def hatch_whites(h: Hatcher):
    blue, red, black = white_targets(h.maps)
    for ink, t, layers in (
        (BLUE, blue, ((0, 0.0, 0.04, 0.22), (62, 0.04, 0.06, 0.25), (-48, 0.1, 0.07, 0.28))),
        (RED, red, ((28, 0.0, 0.04, 0.2), (-75, 0.04, 0.05, 0.22))),
        (BLACK, black, ((80, 0.0, 0.05, 0.25),)),
    ):
        for angle, lo, cap, p in layers:
            h.lay(Pass(ink, angle, lo, cap, p, **LONG), t)


def hatch_fruit(h: Hatcher):
    red, black, blue = fruit_targets(h.maps)
    cherry = masks(h.maps, ("cherry",))
    for sel, kw in ((cherry, SMALL), (~cherry, FRUIT_KW)):
        for ink, t, layers in (
            (RED, red, ((0, 0.0, 0.07, 0.3), (58, 0.07, 0.08, 0.32), (-38, 0.15, 0.09, 0.35))),
            (BLACK, black, ((70, 0.0, 0.05, 0.3), (-20, 0.05, 0.07, 0.35))),
            (BLUE, blue, ((110, 0.0, 0.04, 0.28),)),
        ):
            for angle, lo, cap, p in layers:
                h.lay(Pass(ink, angle, lo, cap, p, **kw), np.where(sel, t, 0))


def hatch_darks(h: Hatcher):
    black, blue, red = dark_targets(h.maps)
    for ink, t, layers in (
        (BLUE, blue, ((0, 0.0, 0.1, 0.3), (75, 0.1, 0.12, 0.35))),
        (BLACK, black, ((90, 0.0, 0.12, 0.35), (15, 0.12, 0.14, 0.4),
                        (-50, 0.26, 0.2, 0.45))),
        (RED, red, ((35, 0.0, 0.1, 0.3),)),
    ):
        for angle, lo, cap, p in layers:
            h.lay(Pass(ink, angle, lo, cap, p, **LONG), t)


def hatch_wall(h: Hatcher):
    blue, black = wall_targets(h.maps)
    kw = {"length": (150, 340), "count": (6, 12), "spacing": (5.0, 16.0)}
    h.lay(Pass(BLUE, 0, 0.0, 0.05, 0.22, **kw), blue)
    h.lay(Pass(BLUE, 90, 0.05, 0.05, 0.24, **kw), blue)
    h.lay(Pass(BLACK, 30, 0.0, 0.05, 0.25, **kw), black)


def hatch_small(h: Hatcher):
    ink, steel = small_targets(h.maps)
    kw = {"length": (8, 30), "count": (3, 6), "spacing": (1.8, 4.0), "shortest": 3,
          "by_region": True}  # fmt: skip
    h.lay(Pass(BLACK, 0, 0.0, 0.45, 0.5, **kw), ink)
    h.lay(Pass(BLUE, 90, 0.0, 0.1, 0.3, **SMALL), steel)
    h.lay(Pass(BLACK, 0, 0.08, 0.1, 0.3, **SMALL), steel)


# ------------------------------------------------------------------- lines

#: Ink and weight of each object's outline: (pens, weight in shadow).
OUTLINES = {"jug": ((BLUE, BLACK), 0.45), "bowl": ((BLUE, BLACK), 0.4),
            "bottle": ((BLACK,), 0.6), "apple": ((BLACK, RED), 0.5),
            "half_apple": ((BLACK, RED), 0.45), "cherries": ((BLACK,), 0.5),
            "knife": ((BLACK,), 0.45)}  # fmt: skip


def resample(pts, step=2.0):
    pts = np.asarray(pts, float)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = max(int(d[-1] / step), 2)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], 1)


def inward(line, inside):
    """Unit normals of a polyline pointing into the shape ``inside``."""
    t = unit_rows(np.gradient(line, axis=0))
    n = np.stack([t[:, 1], -t[:, 0]], 1)
    flip = ~sample(inside, line + 3 * n)
    return np.where(flip[:, None], -n, n)


def weighted_runs(line, weight, low=0.08, min_len=6):
    on = np.append(weight > low, False)
    out, start = [], None
    for i, ok in enumerate(on):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                out.append((line[start:i], weight[start:i]))
            start = None
    return out


def draw_contour(h: Hatcher, line, weight, pens_):
    """A living line: its pressure follows ``weight`` along it, swelling and
    thinning, broken where the weight falls; restated where strong."""
    for run, w in weighted_runs(line, weight):
        n = len(run)
        u = np.linspace(0, 1, n)
        taper = np.clip(np.minimum(u, 1 - u) * n / 8, 0, 1) ** 0.7
        noise = 0.8 + 0.4 * value_noise(np.arange(n) / 9.0 + h.rng.uniform(0, 99), 4)
        press = w * taper * noise
        for k, pen in enumerate(pens_):
            if k and press.max() < 0.3:
                break
            off = h.rng.normal(0, 0.5, 2) if k else 0
            h.sheet.stroke([tuple(q) for q in run[::2] + off], pen,
                           list(press[::2] * (1 - 0.3 * k) + 0.005))  # fmt: skip


def draw_outlines(h: Hatcher):
    m = h.maps
    for name, line in m.lines.items():
        if not name.startswith("outline_"):
            continue
        what = name[len("outline_") : name.rindex("_")]
        pens_, strength = OUTLINES[what]
        pts = resample(line[:, :2], 2.0)
        inside = np.isin(m.obj, np.unique(sample(m.obj, pts + 0.0)))
        n = inward(pts, inside)
        S = sample(m.shade, pts + 3 * n)
        w = strength * smoothstep(0.15, 0.6, S) + 0.05
        draw_contour(h, pts, w, pens_)


def draw_rims_and_stems(h: Hatcher):
    m = h.maps
    for name, pen, lo, hi in (("rim_jug", BLUE, 0.25, 0.5), ("rim_bowl", BLUE, 0.2, 0.45),
                              ("rim_bottle", BLACK, 0.3, 0.5),
                              ("foot_jug", BLUE, 0.2, 0.45),
                              ("foot_bowl", BLUE, 0.2, 0.45)):  # fmt: skip
        for run in seen_runs(m.lines[name], 4):
            run = resample(run, 2.0)
            S = sample(m.shade, run)
            draw_contour(h, run, lo + (hi - lo) * S, (pen,))
    for stem in m.stems:
        for run in seen_runs(stem, 3):
            run = resample(run, 1.5)
            u = np.linspace(0, 1, len(run))
            w = 0.25 + 0.4 * np.sin(np.pi * u)
            h.sheet.stroke([tuple(q) for q in run], BLACK, list(w))
            h.sheet.stroke([tuple(q + [0.6, 0.3]) for q in run], RED, list(0.6 * w))


def drawing(m: PenMaps, seed=7) -> Sheet:
    sheet = Sheet(WIDTH, HEIGHT, paper=PAPER, seed=seed)
    h = Hatcher(sheet, m, np.random.default_rng(seed))
    hatch_whites(h)
    draw_jug_pattern(h)
    draw_stripes(h)
    hatch_fruit(h)
    hatch_darks(h)
    hatch_wall(h)
    hatch_small(h)
    draw_outlines(h)
    draw_rims_and_stems(h)
    return sheet


def main(studies: Path | None = None):
    """Draw the still life to ``examples/output/stilllife.png``; with a
    folder, also write the lit studies and the maps there."""
    parts = scene_parts()
    sc = merge(parts)
    if studies is not None:
        studies.mkdir(parents=True, exist_ok=True)
        pic = scaled_picture(sc)
        write_studies(pic, studies)
        write_maps(pic, sc, studies)
    sheet = drawing(pen_maps(parts, sc))
    out = HERE / "output" / "stilllife.png"
    write_png(str(out), np.round(sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
