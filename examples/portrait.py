"""Portrait study (work in progress)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from murex import CatmullRom, filters, write_png

from fusain import Sheet, papers, pens

WIDTH, HEIGHT = 1100, 1400
#: The model is computed on a grid this many times coarser than the sheet.
COARSE = 2
#: Smooth white paper, as a pen wants.
PAPER = papers.BRISTOL.with_(color="#fbfaf7")
#: Pixels of the sheet per unit of the model (about a head's height).
SCALE = 780.0
#: Where on the sheet the middle of the head (between the eyes) lands.
CENTRE = (520.0, 560.0)


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Turns head space (x to the sitter's left, y up, z out of the face)
    into camera space: turned by ``yaw``, nodded by ``pitch``, tilted by
    ``roll`` (degrees)."""
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ rx @ ry


HEAD = rotation(28.0, -5.0, -6.0)
HEAD_AT = np.array([0.0, 0.0, 0.0])
BODY = rotation(12.0, 0.0, 2.0)
BODY_AT = np.array([-0.06, 0.0, -0.05])


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def ellipsoid(p, c, r):
    q = (p - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sphere(p, c, r):
    return np.linalg.norm(p - np.asarray(c), axis=-1) - r


def capsule(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - h[:, None] * ba, axis=-1) - r


def softplus(x, k):
    return k * np.logaddexp(0.0, x / k)


def mirrored(p):
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    return q


def plane(p, normal, through):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    return (p - np.asarray(through)) @ n


def profile(knots):
    """A smooth function of y through ``knots`` ((y, value) pairs, y rising)."""
    ys, vs = CatmullRom(knots).sample(0.002)[0].T
    return lambda y: np.interp(y, ys, vs)


#: The face, as the cross-sections of a loft from the brow to the chin: at
#: each height, its half-width, the depth of its widest point and how far it
#: reaches out in front of that.
FACE_WIDE = profile(
    [(-0.44, 0.0), (-0.415, 0.075), (-0.38, 0.12), (-0.32, 0.18), (-0.25, 0.232),
     (-0.16, 0.27), (-0.05, 0.29), (0.08, 0.29), (0.2, 0.26)]
)
FACE_AT = profile(
    [(-0.45, 0.17), (-0.40, 0.15), (-0.32, 0.10), (-0.25, 0.04), (-0.16, 0.02),
     (-0.05, 0.02), (0.08, 0.02), (0.2, 0.0)]
)
FACE_FRONT = profile(
    [(-0.45, 0.03), (-0.425, 0.10), (-0.39, 0.14), (-0.34, 0.17), (-0.29, 0.22),
     (-0.24, 0.26), (-0.16, 0.28), (-0.05, 0.285), (0.08, 0.30), (0.2, 0.30)]
)


def face_sdf(p):
    """The loft of the face (see :data:`FACE_WIDE`)."""
    y = p[:, 1]
    a = np.maximum(FACE_WIDE(y), 0.004)
    b = np.maximum(FACE_FRONT(y), 0.004)
    c = FACE_AT(y)
    n = 2.3
    g = (np.abs(p[:, 0] / a) ** n + np.abs((p[:, 2] - c) / b) ** n) ** (1.0 / n)
    d = (g - 1.0) * np.minimum(a, b)
    d = smax(d, -0.445 - y, 0.02)
    return smax(d, y - 0.2, 0.08)


def head_sdf(p):
    """Distance to the head's skin, in head space."""
    q = mirrored(p)
    d = ellipsoid(p, (0.0, 0.11, -0.07), (0.30, 0.39, 0.41))
    d = smin(d, face_sdf(p), 0.06)
    # cheekbones and the chin
    d = smin(d, ellipsoid(q, (0.165, -0.05, 0.15), (0.09, 0.06, 0.10)), 0.08)
    d = smin(d, ellipsoid(p, (0.0, -0.375, 0.185), (0.06, 0.05, 0.07)), 0.06)
    # brow ridge, sockets, eyeballs and lids
    d = smin(d, ellipsoid(q, (0.11, 0.07, 0.25), (0.08, 0.03, 0.05)), 0.05)
    d = smax(d, -ellipsoid(q, (EYE[0], 0.0, 0.29), (0.072, 0.048, 0.05)), 0.03)
    d = smin(d, sphere(q, EYE, EYE_R), 0.01)
    d = smin(d, lids_sdf(q), 0.008)
    # nose
    nose = capsule(p, (0.0, 0.03, 0.285), (0.0, -0.145, 0.36), 0.013)
    nose = smin(nose, sphere(p, (0.0, -0.152, 0.345), 0.022), 0.02)
    nose = smin(nose, ellipsoid(q, (0.03, -0.165, 0.31), (0.02, 0.019, 0.024)), 0.016)
    d = smin(d, nose, 0.03)
    # the lips
    d = smin(d, ellipsoid(p, (0.0, -0.248, 0.285), (0.07, 0.021, 0.028)), 0.025)
    d = smin(d, ellipsoid(p, (0.0, -0.287, 0.28), (0.06, 0.026, 0.03)), 0.025)
    return d


#: The centre of the right eyeball (mirrored for the left) and its radius.
EYE = (0.122, 0.0, 0.212)
EYE_R = 0.07


#: The opening of the lids, as angles on the eyeball (radians) seen from the
#: front of the head: the inner and outer corners, and (as knots over the
#: angle across) the heights of the upper and lower lids.
CORNERS = (-1.0, 1.15)
UPPER = ((-1.0, -0.02), (-0.6, 0.24), (-0.1, 0.33), (0.45, 0.32), (0.9, 0.22), (1.15, 0.1))
LOWER = ((-1.0, -0.02), (-0.6, -0.26), (0.0, -0.36), (0.55, -0.3), (0.95, -0.12), (1.15, 0.1))
LID = 0.006


def eye_angles(q):
    """Angles across and up of points round the right eye (mirrored)."""
    v = q - np.asarray(EYE)
    across = np.arctan2(v[:, 0], v[:, 2])
    up = np.arctan2(v[:, 1], np.hypot(v[:, 0], v[:, 2]))
    return across, up


def lid_heights(across):
    up = np.interp(across, *np.array(UPPER).T)
    low = np.interp(across, *np.array(LOWER).T)
    return up, low


def lids_sdf(q):
    """The lids, a skin over the eyeball open between their edges."""
    across, up = eye_angles(q)
    top, bottom = lid_heights(across)
    inside = np.minimum(np.minimum(up - bottom, top - up), np.minimum(across - CORNERS[0], CORNERS[1] - across))
    shell = sphere(q, EYE, EYE_R + LID)
    return smax(shell, inside * EYE_R, 0.004)


#: The hair: where the fringe ends (its edge rises toward the temples), how
#: far the face shows between the locks either side of it and where the ends
#: of the hair hang.
FRINGE = (0.105, 0.025)
OPENING = profile([(-0.7, 0.26), (-0.4, 0.245), (-0.1, 0.235), (0.1, 0.235), (0.3, 0.2)])
HAIR_WIDE = profile([(-0.7, 0.29), (-0.6, 0.335), (-0.45, 0.35), (-0.2, 0.35), (0.1, 0.345), (0.3, 0.34)])


def hair_sdf(p):
    """Distance to the hair's outer surface, in head space."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    top = ellipsoid(p, (0.0, 0.11, -0.09), (0.345, 0.43, 0.44))
    a = HAIR_WIDE(y)
    g = np.sqrt((x / a) ** 2 + ((z + 0.09) / 0.45) ** 2)
    curtain = (g - 1.0) * np.minimum(a, 0.45)
    phi = np.arctan2(x, z + 0.08)
    ends = ENDS - 0.04 * z / 0.4 + 0.03 * (x / 0.4) ** 2 + 0.025 * np.sin(7 * phi + 1.0)
    ends += 0.015 * np.sin(17 * phi)
    curtain = smax(curtain, ends - y, 0.03)
    curtain = smax(curtain, y - 0.2, 0.1)
    d = smin(top, curtain, 0.08)
    edge = fringe_edge(x)
    face = smax(np.abs(x) - OPENING(y), -0.02 - z, 0.03)
    face = smin(face, 0.07 - z, 0.03)
    face = smax(face, y - edge, 0.012)
    return smax(d, -face, 0.015)


def body_sdf(p):
    """Distance to the neck, shoulders and chest, in body space."""
    q = mirrored(p)
    neck = capsule(p, (0.0, -0.10, -0.07), (0.0, -0.65, -0.05), 0.122)
    slope = capsule(q, (0.06, -0.52, -0.09), (0.52, -0.68, -0.09), 0.10)
    chest = ellipsoid(p, (0.0, -1.02, -0.03), (0.55, 0.40, 0.24))
    arms = capsule(q, (0.56, -0.78, -0.09), (0.64, -1.6, -0.09), 0.14)
    d = smin(slope, chest, 0.12)
    d = smin(d, arms, 0.08)
    return smin(d, neck, 0.10)


def to_head(c):
    return (c - HEAD_AT) @ HEAD


def to_body(c):
    return (c - BODY_AT) @ BODY


def bounded(sdf, c, local, bound):
    """``sdf`` of the points ``c`` (camera space; ``local`` takes them into
    the part's space), computed only for those within ``bound`` of it:
    farther, the distance to the bound."""
    d = bound.copy()
    near = np.nonzero(bound < 0.1)[0]
    if near.size:
        d[near] = sdf(local(c[near]))
    return d


def scene_sdf(c, hair: bool = True):
    """Distance in camera space, and which part is nearest (0 head, 1 body,
    2 hair)."""
    head_bound = np.linalg.norm(c - HEAD_AT - np.array([0.0, 0.0, 0.0]), axis=-1) - 0.56
    body_bound = np.maximum(c[:, 1] - (BODY_AT[1] - 0.05), np.abs(c[:, 0]) - 0.9)
    h = bounded(head_sdf, c, to_head, head_bound)
    f = bounded(hair_sdf, c, to_head, head_bound - 0.25)
    if not hair:
        f = np.full(len(c), 1.0)
    b = bounded(body_sdf, c, to_body, body_bound)
    d = np.minimum(np.minimum(h, b), f)
    part = np.where(f <= d, 2, np.where(b <= d, 1, 0)).astype(np.int8)
    return d, part


@dataclass
class Model:
    """What the model shows at each cell of the coarse grid."""

    hit: np.ndarray
    part: np.ndarray
    cam: np.ndarray
    normal: np.ndarray
    head: np.ndarray
    body: np.ndarray
    light: np.ndarray
    occlusion: np.ndarray
    shadow: np.ndarray
    under: np.ndarray
    behind: np.ndarray


def grid_coords():
    h, w = HEIGHT // COARSE, WIDTH // COARSE
    gy, gx = np.mgrid[0:h, 0:w]
    sx = (gx + 0.5) * COARSE
    sy = (gy + 0.5) * COARSE
    x = (sx - CENTRE[0]) / SCALE
    y = -(sy - CENTRE[1]) / SCALE
    return x, y


def skin_sdf(c):
    """Distance to the head alone, as if it had no hair."""
    return head_sdf(to_head(c)), None


def torso_sdf(c):
    """Distance to the body alone."""
    return body_sdf(to_body(c)), None


def march(x, y, sdf=scene_sdf):
    """Where rays from the viewer through (x, y) meet the surface ``sdf``
    measures, and whether they do."""
    z = np.full(x.shape, 0.7)
    active = np.ones(x.shape, bool)
    hit = np.zeros(x.shape, bool)
    for _ in range(200):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        c = np.stack([x[idx], y[idx], z[idx]], axis=-1)
        d, _ = sdf(c)
        z[idx] -= 0.6 * d
        done = d < 4e-4
        hit[idx[done]] = True
        gone = z[idx] < -1.0
        active[idx[done | gone]] = False
    return z, hit


def normals(c):
    e = 1.5e-3
    n = np.zeros_like(c)
    for i in range(3):
        o = np.zeros(3)
        o[i] = e
        n[:, i] = scene_sdf(c + o)[0] - scene_sdf(c - o)[0]
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def occlusion(c, n):
    ao = np.zeros(len(c))
    for i in range(1, 6):
        h = 0.018 * i
        ao += (h - scene_sdf(c + n * h)[0]) / (2.0 ** (i - 1))
    return np.clip(1.0 - 4.0 * ao, 0.0, 1.0)


LIGHT = np.array([-0.35, 0.55, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def soft_shadow(c, n, hair: bool = True, spread: float = 0.12):
    """How much of the light reaches each point: 0 in the shadow another
    part casts (the hair too, unless not ``hair``), with a penumbra that
    widens away from what casts it."""
    res = np.ones(len(c))
    t = np.full(len(c), 0.015)
    start = c + n * 0.015
    for _ in range(50):
        d = scene_sdf(start + LIGHT * t[:, None], hair)[0]
        res = np.where(t < 0.6, np.minimum(res, d / (0.004 + spread * t)), res)
        t += np.maximum(0.8 * d, 0.005)
    return filters.smoothstep(0.0, 1.0, np.clip(res, 0.0, 1.0))


def build_model() -> Model:
    x, y = grid_coords()
    shape = x.shape
    xf, yf = x.ravel(), y.ravel()
    # only march inside a generous box round the figure
    z, hit = march(xf, yf)
    c = np.stack([xf, yf, z], axis=-1)
    n = np.zeros_like(c)
    ao = np.ones(len(c))
    sh = np.ones(len(c))
    idx = np.nonzero(hit)[0]
    n[idx] = normals(c[idx])
    ao[idx] = occlusion(c[idx], n[idx])
    _, part = scene_sdf(c)
    head = idx[part[idx] != 1]
    sh[head] = soft_shadow(c[head], n[head])
    body = idx[part[idx] == 1]
    sh[body] = 0.4 + 0.6 * soft_shadow(c[body], n[body], hair=False, spread=0.6)
    lam = np.clip((n @ LIGHT + 0.25) / 1.25, 0.0, 1.0)
    light = lam * (0.35 + 0.65 * sh)
    # the skin under the hair, near where the hair shows
    under = np.full(c.shape, np.nan)
    idx = np.nonzero(hit & (part == 2))[0]
    zu, hu = march(xf[idx], yf[idx], skin_sdf)
    cu = np.stack([xf[idx], yf[idx], zu], axis=-1)
    under[idx[hu]] = to_head(cu[hu])
    behind = np.full(c.shape, np.nan)
    zb, hb = march(xf[idx], yf[idx], torso_sdf)
    cb = np.stack([xf[idx], yf[idx], zb], axis=-1)
    behind[idx[hb]] = to_body(cb[hb])
    return Model(
        hit.reshape(shape),
        part.reshape(shape),
        c.reshape(*shape, 3),
        n.reshape(*shape, 3),
        to_head(c).reshape(*shape, 3),
        to_body(c).reshape(*shape, 3),
        light.reshape(shape),
        ao.reshape(shape),
        sh.reshape(shape),
        under.reshape(*shape, 3),
        behind.reshape(*shape, 3),
    )


# ----------------------------------------------------------------------
# features, as curves on the head


def to_sheet(points):
    """Head-space points to sheet (x, y), and their depth toward the eye."""
    cam = np.asarray(points) @ HEAD.T + HEAD_AT
    xy = np.stack([CENTRE[0] + SCALE * cam[:, 0], CENTRE[1] - SCALE * cam[:, 1]], -1)
    return xy, cam[:, 2]


def on_head(xy, lift: float = 0.0):
    """The points of the skin in front of head-space (x, y), raised ``lift``
    off it."""
    xy = np.asarray(xy, float)
    p = np.stack([xy[:, 0], xy[:, 1], np.full(len(xy), 0.6)], -1)
    for _ in range(80):
        p[:, 2] -= 0.7 * head_sdf(p)
    p[:, 2] += lift
    return p


def eye_point(side: int, across, up, lift: float = LID):
    """Points on the eyeball of the right (``side`` 1) or left (-1) eye, at
    angles ``across`` (out toward the temple) and ``up``."""
    across, up = np.asarray(across, float), np.asarray(up, float)
    r = EYE_R + lift
    v = np.stack([np.sin(across) * np.cos(up), np.sin(up), np.cos(across) * np.cos(up)], -1)
    centre = np.array([side * EYE[0], EYE[1], EYE[2]])
    v[..., 0] *= side
    return centre + r * v


def lid_curve(side: int, upper: bool, n: int = 24):
    """The edge of a lid, in head space."""
    across = np.linspace(CORNERS[0], CORNERS[1], n)
    top, bottom = lid_heights(across)
    return eye_point(side, across, top if upper else bottom)


def gaze() -> np.ndarray:
    """Where the eyes look, in head space: at the viewer, a little aside."""
    g = 0.65 * np.array([0.0, 0.0, 1.0]) @ HEAD + np.array([0.0, 0.03, 0.35])
    return g / np.linalg.norm(g)


# ----------------------------------------------------------------------
# hatching: the lines along which a field crosses its levels

#: The corners of a cell, in the order of the bits of its case, and the
#: pairs of its edges each case joins (edges: 0 top, 1 right, 2 bottom,
#: 3 left).
SEGMENTS = {
    1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 5: ((3, 0), (1, 2)),
    6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),), 9: ((2, 0),), 10: ((0, 1), (2, 3)),
    11: ((2, 1),), 12: ((1, 3),), 13: ((1, 0),), 14: ((0, 3),),
}


def crossings(field: np.ndarray, jump: float = 1.5):
    """Every segment of every integer level line of ``field`` (NaN where it is
    not defined) through the cells of its grid, as (level, edge a, edge b,
    point a, point b); cells across which it jumps by ``jump`` or more are
    skipped (an edge that hides one surface behind another)."""
    h, w = field.shape
    f00, f01 = field[:-1, :-1], field[:-1, 1:]
    f10, f11 = field[1:, :-1], field[1:, 1:]
    corners = np.stack([f00, f01, f11, f10], -1)
    low, high = np.nanmin(corners, -1), np.nanmax(corners, -1)
    ok = np.isfinite(corners).all(-1) & (high - low < jump)
    first = np.floor(low) + 1
    count = np.where(ok, np.floor(high) - first + 1, 0).astype(int)
    count = np.maximum(count, 0)
    cells = np.repeat(np.arange(count.size), count.ravel())
    start = np.repeat(np.cumsum(count.ravel()) - count.ravel(), count.ravel())
    level = first.ravel()[cells] + (np.arange(cells.size) - start)
    ci, cj = np.divmod(cells, w - 1)
    c = corners.reshape(-1, 4)[cells]
    above = c > level[:, None]
    case = above[:, 0] * 1 + above[:, 1] * 2 + above[:, 2] * 4 + above[:, 3] * 8
    return ci, cj, c, level, case


def edge_point(ci, cj, c, level, edge):
    """Where the level crosses an edge of a cell, and that edge's id."""
    a = np.array([0, 1, 3, 0])[edge]
    b = np.array([1, 2, 2, 3])[edge]
    fa = np.take_along_axis(c, a[:, None], 1)[:, 0]
    fb = np.take_along_axis(c, b[:, None], 1)[:, 0]
    t = np.clip((level - fa) / np.where(fb == fa, 1e-9, fb - fa), 0.0, 1.0)
    ya = np.array([0, 0, 1, 0])[edge]
    xa = np.array([0, 1, 0, 0])[edge]
    horizontal = (edge == 0) | (edge == 2)
    x = cj + xa + np.where(horizontal, t, 0.0)
    y = ci + ya + np.where(horizontal, 0.0, t)
    eid = np.where(horizontal, (ci + ya) * 100003 + cj, 50000000 + ci * 100003 + cj + xa)
    return np.stack([x, y], -1), eid


def level_lines(field: np.ndarray, jump: float = 1.5):
    """The integer level lines of ``field``, each a polyline (N, 2) in grid
    (x, y), with its level."""
    ci, cj, c, level, case = crossings(field, jump)
    pairs = []
    for k, segs in SEGMENTS.items():
        sel = np.nonzero(case == k)[0]
        for ea, eb in segs:
            pairs.append((sel, np.full(sel.size, ea), np.full(sel.size, eb)))
    sel = np.concatenate([p[0] for p in pairs])
    ea = np.concatenate([p[1] for p in pairs])
    eb = np.concatenate([p[2] for p in pairs])
    pa, ia = edge_point(ci[sel], cj[sel], c[sel], level[sel], ea)
    pb, ib = edge_point(ci[sel], cj[sel], c[sel], level[sel], eb)
    lv = level[sel]
    return chain(lv, ia, ib, pa, pb)


def chain(lv, ia, ib, pa, pb):
    """Join segments that share an edge crossing into polylines."""
    ka = list(zip(lv.tolist(), ia.tolist()))
    kb = list(zip(lv.tolist(), ib.tolist()))
    touching: dict = {}
    for i in range(len(ka)):
        touching.setdefault(ka[i], []).append(i)
        touching.setdefault(kb[i], []).append(i)
    used = np.zeros(len(ka), bool)
    lines = []
    for i in range(len(ka)):
        if used[i]:
            continue
        used[i] = True
        ahead = walk(kb[i], touching, used, ka, kb, pa, pb)
        behind = walk(ka[i], touching, used, ka, kb, pa, pb)
        points = [*reversed(behind), pa[i], pb[i], *ahead]
        lines.append((float(lv[i]), np.array(points)))
    return lines


def walk(key, touching, used, ka, kb, pa, pb):
    """The points met going on from ``key`` through unused segments."""
    points = []
    while True:
        nxt = [j for j in touching[key] if not used[j]]
        if not nxt:
            return points
        j = nxt[0]
        used[j] = True
        if ka[j] == key:
            key, point = kb[j], pb[j]
        else:
            key, point = ka[j], pa[j]
        points.append(point)


def smooth_within(a: np.ndarray, mask: np.ndarray, r: int) -> np.ndarray:
    """``a`` (H, W, C) blurred over ``r`` cells, from within ``mask`` only."""
    m = mask.astype(float)
    num = filters.box_blur(a * m[..., None], r)
    den = filters.box_blur(m, r)
    return num / np.maximum(den, 1e-6)[..., None]


def sample(grid: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """``grid`` (a map of the coarse grid) at sheet points ``xy``."""
    h, w = grid.shape[:2]
    gx = np.clip(xy[:, 0] / COARSE - 0.5, 0, w - 1)
    gy = np.clip(xy[:, 1] / COARSE - 0.5, 0, h - 1)
    return filters.bilinear(grid, gy, gx)


def resample(points: np.ndarray, step: float) -> np.ndarray:
    """A polyline resampled every ``step`` along its length."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    at = np.concatenate([[0.0], np.cumsum(seg)])
    if at[-1] < step:
        return points[[0, -1]]
    s = np.arange(0.0, at[-1], step)
    return np.stack([np.interp(s, at, points[:, 0]), np.interp(s, at, points[:, 1])], -1)


def taper(n: int, ends: int) -> np.ndarray:
    """1 along a stroke of ``n`` points, easing to 0 over ``ends`` at each end."""
    i = np.arange(n, dtype=float)
    k = np.minimum(i + 0.5, n - 0.5 - i) / max(ends, 1)
    return np.sin(0.5 * np.pi * np.clip(k, 0.0, 1.0)) ** 1.2


#: The inks, with a fine ball for the hatching and the usual one for lines.
RED = pens.ballpoint("#b3262b", 2.0, name="red")
BLUE = pens.ballpoint("#22357e", 2.0, name="blue")
BLACK = pens.ballpoint("#26262a", 2.0, name="black")
RED_LINE = pens.ballpoint("#b3262b", 2.6, name="red")
BLUE_LINE = pens.ballpoint("#22357e", 2.6, name="blue")
BLACK_LINE = pens.ballpoint("#26262a", 2.6, name="black")


@dataclass(frozen=True)
class Layer:
    """One set of hatching in one ink: lines along the level lines of a
    family of planes ``gap`` pixels apart (one in ``every`` of them), laid
    where ``density`` exceeds ``threshold``, fully ``ramp`` above it."""

    pen: pens.Pen
    family: int
    every: int
    offset: int
    threshold: float
    ramp: float = 0.25
    firm: float = 0.3
    lift: float = 0.45


class Drawing:
    """The sheet and what is known of the figure on it."""

    def __init__(self, model: Model, seed: int):
        self.sheet = Sheet(WIDTH, HEIGHT, paper=PAPER, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.model = model
        self.lines: dict = {}

    def stroke(self, points, pen, pressure) -> None:
        self.sheet.stroke([tuple(p) for p in points], pen, list(pressure))

    def hatch(self, lines, layer: Layer, density: np.ndarray, reach: float = 110.0):
        """Lay ``layer`` along ``lines`` (level, polyline) in pieces about
        ``reach`` long."""
        for level, grid_pts in lines:
            if int(level) % layer.every != layer.offset or len(grid_pts) < 3:
                continue
            pts = resample((grid_pts + 0.5) * COARSE, 3.0)
            if len(pts) < 4:
                continue
            w = filters.smoothstep(layer.threshold, layer.threshold + layer.ramp, sample(density, pts))
            self.pieces(pts, w, layer, reach)

    def pieces(self, pts, w, layer: Layer, reach: float) -> None:
        rng = self.rng
        n = len(pts)
        i = -int(rng.uniform(0, reach / 3.0))
        while i < n:
            length = int(rng.uniform(0.6, 1.4) * reach / 3.0)
            a, b = max(i, 0), min(i + length, n)
            if b - a >= 4:
                self.piece(pts[a:b], w[a:b], layer)
            i += length - int(rng.integers(-2, 3))

    def piece(self, pts, w, layer: Layer) -> None:
        """One stroke, lifted where the tone is too light for it."""
        cut = self.rng.uniform(0.08, max(layer.lift, 0.09))
        on = w > cut
        if not on.any():
            return
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < 3:
                continue
            seg, ws = pts[a:b], w[a:b]
            normal = np.stack([-np.gradient(seg[:, 1]), np.gradient(seg[:, 0])], -1)
            normal /= np.maximum(np.hypot(*normal.T), 1e-9)[:, None]
            wobble = np.cumsum(self.rng.normal(0.0, 0.12, len(seg)))
            wobble -= np.linspace(wobble[0], wobble[-1], len(seg))
            seg = seg + normal * (wobble + self.rng.normal(0.0, 0.35))[:, None]
            firm = layer.firm * self.rng.uniform(0.85, 1.15)
            pressure = firm * (0.45 + 0.55 * ws) * taper(len(seg), 4)
            keep = slice(None, None, 2)
            self.stroke(seg[keep], layer.pen, pressure[keep])


#: The families of planes whose sections give the hatching on the head
#: (normals in head space), each crossing the others obliquely.
HEAD_FAMILIES = ((-0.35, 0.75, 0.55), (0.75, 0.5, 0.45), (0.8, -0.45, 0.4))


def family_lines(coords, mask, normal, gap: float, split: int):
    """The level lines of planes ``gap`` pixels apart, split ``split`` times
    finer, through ``coords`` where ``mask`` holds."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    field = np.where(mask, coords @ n * SCALE / gap * split, np.nan)
    return level_lines(field, jump=1.2 * split)


def skin_dark(model: Model) -> np.ndarray:
    """How dark the skin is wanted, 0 in the brightest light to 1."""
    lam = np.clip(model.normal @ LIGHT, -1.0, 1.0)
    lit = np.clip((lam + 0.35) / 1.35, 0.0, 1.0) * (0.3 + 0.7 * model.shadow)
    dark = 1.0 - lit * (0.55 + 0.45 * model.occlusion)
    return np.clip(dark, 0.0, 1.0)


#: The lips, in head space (x, y), from the sitter's right corner to her
#: left: the top of the upper lip, the parting and the bottom of the lower.
LIP_TOP = ((-0.086, -0.25), (-0.055, -0.247), (-0.02, -0.234), (0.0, -0.238),
           (0.02, -0.234), (0.055, -0.247), (0.086, -0.25))
LIP_PARTING = ((-0.086, -0.25), (-0.06, -0.262), (-0.025, -0.269), (0.0, -0.272),
               (0.025, -0.269), (0.06, -0.262), (0.086, -0.25))
LIP_BOTTOM = ((-0.086, -0.25), (-0.06, -0.282), (-0.03, -0.298), (0.0, -0.302),
              (0.03, -0.298), (0.06, -0.282), (0.086, -0.25))


def curve(knots, n: int = 40) -> np.ndarray:
    """A smooth curve through 2D ``knots``, ``n`` points along it."""
    path = CatmullRom(np.asarray(knots, float))
    pts = path.sample(path.length / (n - 1))[0]
    return pts


def inside(xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Which of the points ``xy`` (..., 2) lie inside ``polygon``."""
    x, y = xy[..., 0], xy[..., 1]
    res = np.zeros(x.shape, bool)
    px, py = polygon[:, 0], polygon[:, 1]
    for i in range(len(polygon)):
        x0, y0, x1, y1 = px[i - 1], py[i - 1], px[i], py[i]
        cross = (y0 > y) != (y1 > y)
        at = x0 + (y - y0) * (x1 - x0) / np.where(y1 == y0, 1e-12, y1 - y0)
        res ^= cross & (x < at)
    return res


def blob(p: np.ndarray, centre, radii) -> np.ndarray:
    """A soft round patch of the head, 1 at ``centre``."""
    q = (p - np.asarray(centre)) / np.asarray(radii)
    return np.exp(-np.sum(q * q, axis=-1))


@dataclass
class Maps:
    """Where each part of the figure shows, and how much of each ink it
    wants, on the coarse grid; ``skin`` is the head-space point of the skin
    seen (or, under the tips of the fringe, of the skin beneath them)."""

    face: np.ndarray
    eyes: np.ndarray
    lips: np.ndarray
    neck: np.ndarray
    hair: np.ndarray
    cloth: np.ndarray
    skin: np.ndarray
    body: np.ndarray
    dark: np.ndarray
    red: np.ndarray
    blue: np.ndarray
    black: np.ndarray


def eye_mask(model: Model) -> np.ndarray:
    """The opening of the lids, where the eyeball shows."""
    q = mirrored(model.head.reshape(-1, 3))
    across, up = eye_angles(q)
    top, bottom = lid_heights(across)
    near = np.linalg.norm(q - np.asarray(EYE), axis=-1) < EYE_R + 0.5 * LID
    opening = (up > bottom) & (up < top) & (across > CORNERS[0]) & (across < CORNERS[1])
    return (near & opening).reshape(model.hit.shape) & (model.part == 0) & model.hit


def lip_mask(model: Model) -> np.ndarray:
    outline = np.concatenate([curve(LIP_TOP), curve(LIP_BOTTOM)[::-1]])
    h = model.head
    return inside(h[..., :2], outline) & (h[..., 2] > 0.2) & (model.part == 0) & model.hit


def body_seen(model: Model):
    """The body-space point of the body at each cell, whether seen or behind
    the hair, and where there is one."""
    body = np.where((model.part == 2)[..., None], model.behind, model.body)
    there = model.hit & ((model.part == 1) | np.isfinite(model.behind[..., 0]))
    return np.nan_to_num(body), there


def cloth_mask(model: Model) -> np.ndarray:
    """The top she wears: below a wide, low neckline."""
    b, there = body_seen(model)
    neckline = -0.74 + 0.35 * (b[..., 0] / 0.3) ** 2
    return there & (b[..., 1] < neckline)


def head_extras(h: np.ndarray):
    """What the head's own colouring adds to the lighting: the flush of the
    cheeks, nose, chin and lids (red), the cool of the sockets and temples
    (blue), and where the face is kept warm and open (round the mouth)."""
    q = mirrored(h)
    flush = 0.6 * blob(q, (0.15, -0.13, 0.22), (0.085, 0.07, 0.2))
    flush += 0.3 * blob(h, (0.0, -0.16, 0.36), (0.035, 0.03, 0.1))
    flush += 0.25 * blob(h, (0.0, -0.40, 0.25), (0.05, 0.035, 0.1))
    sockets = blob(q, (EYE[0] - 0.01, 0.025, 0.28), (0.075, 0.045, 0.2))
    sockets += 0.6 * blob(q, (0.045, 0.0, 0.28), (0.025, 0.04, 0.2))
    temple = blob(q, (0.27, 0.1, 0.1), (0.06, 0.12, 0.2))
    mouth = blob(h, (0.0, -0.33, 0.28), (0.12, 0.09, 0.3))
    hollow = blob(h, (0.0, -0.322, 0.27), (0.04, 0.014, 0.1))
    return flush, 0.55 * sockets + 0.3 * temple + 0.35 * hollow, mouth


def veiled(model: Model) -> np.ndarray:
    """The forehead under the tips of the fringe, which shows between them."""
    u = model.under
    with np.errstate(invalid="ignore"):
        edge = fringe_edge(u[..., 0])
        near = (u[..., 1] < edge + 0.12) & (np.abs(u[..., 0]) < 0.32) & (u[..., 2] > 0.0)
    return model.hit & (model.part == 2) & np.isfinite(u[..., 0]) & near


def design(model: Model) -> Maps:
    """Where everything is, and the ink each place wants."""
    hit, part = model.hit, model.part
    eyes, lips, cloth, veil = eye_mask(model), lip_mask(model), cloth_mask(model), veiled(model)
    face = (hit & (part == 0) & ~eyes & ~lips) | veil
    skin = np.where(veil[..., None], model.under, model.head)
    body, there = body_seen(model)
    neck = there & ~cloth
    hair = hit & (part == 2)
    dark = skin_dark(model)
    below = skin[..., 1] - fringe_edge(np.clip(skin[..., 0], -0.3, 0.3))
    shade = filters.smoothstep(-0.05, 0.0, below) * (skin[..., 2] > 0.1)
    dark = np.where(face, np.maximum(dark, 0.6 * shade), dark)
    tuck = filters.box_blur(hair.astype(float), 8) * face * (np.abs(skin[..., 0]) > 0.15)
    dark = np.minimum(dark + 0.8 * tuck, 1.0)
    dark = np.where(veil, 0.5, np.where(hair & neck, 0.85, dark))
    flush, cool, mouth = head_extras(skin)
    on_head = face.astype(float)
    half = LIGHT + np.array([0.0, 0.0, 1.0])
    half /= np.linalg.norm(half)
    gleam = np.clip(model.normal @ half, 0.0, 1.0) ** 40 * model.shadow * on_head
    red = (0.24 + 0.76 * dark + flush * on_head) * (1.0 - 0.9 * gleam)
    blue = filters.smoothstep(0.42, 1.0, dark) + (cool + 0.4 * shade + tuck) * on_head
    blue *= 1.0 - 0.75 * mouth * on_head
    black = filters.smoothstep(0.8, 1.05, dark) * (1.0 - 0.9 * mouth * on_head)
    return Maps(face, eyes, lips, neck, hair, cloth, skin, body, dark, red, blue, black)


# ----------------------------------------------------------------------
# the features


def seen(drawing: Drawing, points3d: np.ndarray, slack: float = 0.012) -> np.ndarray:
    """Which head-space points are not hidden behind another part."""
    xy, depth = to_sheet(points3d)
    front = sample(drawing.model.cam[..., 2], xy)
    hit = sample(drawing.model.hit.astype(float), xy) > 0.5
    return ~hit | (depth > front - slack)


def runs_of(on: np.ndarray):
    """(start, end) of each run of True in ``on``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def line(drawing: Drawing, points3d, pen, pressure, slack: float = 0.012):
    """A stroke through head-space points, lifted where they are hidden;
    ``pressure`` is one value per point or a function of u along it."""
    pts3 = np.asarray(points3d, float)
    n = len(pts3)
    if callable(pressure):
        press = np.asarray(pressure(np.linspace(0.0, 1.0, n)), float)
    else:
        press = np.broadcast_to(np.asarray(pressure, float), (n,)).copy()
    xy, _ = to_sheet(pts3)
    for a, b in runs_of(seen(drawing, pts3, slack)):
        if b - a >= 2:
            drawing.stroke(xy[a:b], pen, press[a:b])


def swell(peak: float, at: float = 0.5, ends: float = 0.0):
    """A pressure that rises from ``ends`` to ``peak`` at ``at`` and falls."""

    def f(u):
        k = np.where(u < at, u / max(at, 1e-6), (1.0 - u) / max(1.0 - at, 1e-6))
        return ends + (peak - ends) * np.sin(0.5 * np.pi * np.clip(k, 0.0, 1.0)) ** 0.8

    return f


#: The iris and the pupil, as angles round the line of sight.
IRIS, PUPIL = 0.5, 0.17


def view_dir() -> np.ndarray:
    """Toward the viewer, in head space."""
    return np.array([0.0, 0.0, 1.0]) @ HEAD


def eye_frame(side: int):
    """The centre of an eyeball and the axes of its iris (sight, across, up)."""
    centre = np.array([side * EYE[0], EYE[1], EYE[2]])
    g = gaze()
    u = np.cross([0.0, 1.0, 0.0], g)
    u /= np.linalg.norm(u)
    return centre, g, u, np.cross(g, u)


def on_iris(side: int, r, a, lift: float = 0.0) -> np.ndarray:
    """Points of the eyeball ``r`` from the line of sight, round at ``a``."""
    centre, g, u, v = eye_frame(side)
    r, a = np.asarray(r, float)[..., None], np.asarray(a, float)[..., None]
    d = np.cos(r) * g + np.sin(r) * (np.cos(a) * u + np.sin(a) * v)
    return centre + (EYE_R + lift) * d


def in_opening(points: np.ndarray, margin: float = 0.0) -> np.ndarray:
    across, up = eye_angles(mirrored(points))
    top, bottom = lid_heights(across)
    return (up > bottom + margin) & (up < top - margin) & (across > CORNERS[0]) & (
        across < CORNERS[1]
    )


def glint(side: int) -> np.ndarray:
    """Where the light glints on the eye, as a direction from its centre: on
    the bulge of the cornea, toward where the light comes from."""
    h = LIGHT @ HEAD + view_dir()
    _, g, u, v = eye_frame(side)
    a = np.arctan2(h @ v, h @ u)
    p = on_iris(side, 0.5 * IRIS, a) - eye_frame(side)[0]
    return p / np.linalg.norm(p)


def disc_chords(side: int, radius: float, angle: float, gap: float, inner: float = 0.0):
    """Chords across the disc of the eyeball ``radius`` round the line of
    sight, at ``angle`` and ``gap`` apart (radians), outside ``inner``."""
    out = []
    for s in np.arange(-radius + 0.5 * gap, radius, gap):
        k = np.linspace(-1, 1, 9) * np.sqrt(max(radius * radius - s * s, 0.0))
        r = np.hypot(k, s)
        a = angle + np.arctan2(s, k)
        keep = r >= inner
        out.append((on_iris(side, r, a, 0.001), keep))
    return out


def iris_strokes(d: Drawing, side: int) -> None:
    """The iris, deep brown (black over red) and darker under the lid and at
    its rim, the pupil black, the glint left white and a warm fleck across
    from it."""
    centre, g, u, v = eye_frame(side)
    shine = glint(side)
    sets = (
        (RED, IRIS, 0.5, 0.035, 0.0),
        (RED, IRIS, 2.0, 0.04, 0.0),
        (BLACK, IRIS, 1.1, 0.05, 0.0),
        (BLACK, IRIS, 2.6, 0.06, 0.25),
        (BLACK, PUPIL, 0.3, 0.025, 0.0),
        (BLACK, PUPIL, 1.9, 0.025, 0.0),
    )
    for pen, radius, angle, gap, inner in sets:
        for pts, keep in disc_chords(side, radius, angle + d.rng.normal(0, 0.1), gap, inner):
            dirs = (pts - centre) / np.linalg.norm(pts - centre, axis=-1, keepdims=True)
            top = (dirs @ v) / max(np.sin(IRIS), 1e-6)
            firm = 0.4 + 0.25 * np.clip(top, 0, 1) if pen is BLACK else 0.45
            if radius == PUPIL:
                firm = 0.9
            ok = in_opening(pts) & (np.arccos(np.clip(dirs @ shine, -1, 1)) > 0.07) & keep
            press = firm * np.ones(len(pts))
            for a0, b0 in runs_of(ok):
                if b0 - a0 >= 2:
                    line(d, pts[a0:b0], pen, press[a0:b0], 0.03)
    ga = np.arctan2(shine @ v, shine @ u)
    for k in range(3):
        a = ga + np.pi + d.rng.normal(0, 0.2)
        pts = on_iris(side, np.linspace(0.22, 0.34, 4), np.full(4, a), 0.001)
        if in_opening(pts).all():
            line(d, pts, RED, 0.7, 0.03)


def lid_points(side: int, upper: bool, lo: float, hi: float, n: int = 20, rise=0.0, lift=LID):
    """Points along a lid's edge between angles ``lo`` and ``hi`` across,
    raised ``rise`` (radians) above (or below, for the lower lid) it."""
    across = np.linspace(lo, hi, n)
    top, bottom = lid_heights(across)
    up = top + rise if upper else bottom - rise
    return eye_point(side, across, up, lift)


def lids(d: Drawing, side: int) -> None:
    """The upper lid a thick dark line of several strokes, the crease above
    it, the lower lid light, lashes at the outer corner, and the shadow the
    upper lid casts on the eye."""
    rng = d.rng
    c0, c1 = CORNERS
    # the shadow of the lid on the eyeball, and pink at the corners
    for k in range(5):
        pts = lid_points(side, True, c0 + 0.25, c1 - 0.15, 16, -0.05 - 0.035 * k, 0.0)
        line(d, pts, BLUE, swell(0.32 - 0.04 * k, 0.5), 0.03)
    for k in range(4):
        pts = lid_points(side, True, c0 + 0.05, c0 + 0.35, 6, -0.08 - 0.07 * k, 0.0)
        line(d, pts, RED, swell(0.35, 0.3), 0.03)
        pts = lid_points(side, True, c1 - 0.3, c1 - 0.02, 6, -0.06 - 0.06 * k, 0.0)
        line(d, pts, RED, swell(0.3, 0.6), 0.03)
    # the upper lid, restated
    for k, (pen, firm) in enumerate(((BLACK_LINE, 0.95), (BLACK_LINE, 0.8), (BLUE_LINE, 0.7))):
        lo = c0 + 0.12 + 0.05 * k
        pts = lid_points(side, True, lo, c1 + 0.04, 22, 0.012 * k + rng.normal(0, 0.006))
        line(d, pts, pen, swell(firm, 0.62, 0.15), 0.03)
    # the crease
    for pen, firm in ((RED, 0.45), (BLUE, 0.32)):
        pts = lid_points(side, True, c0 + 0.35, c1 - 0.1, 18, 0.24, LID + 0.004)
        line(d, pts, pen, swell(firm, 0.55), 0.03)
    # the lower lid, light, and its rim
    pts = lid_points(side, False, c0 + 0.3, c1 - 0.02, 18, 0.03, LID)
    line(d, pts, RED, swell(0.42, 0.65), 0.03)
    pts = lid_points(side, False, c0 + 0.6, c1 - 0.05, 14, 0.11, LID + 0.003)
    line(d, pts, BLUE, swell(0.22, 0.6), 0.03)
    lashes(d, side)


def lashes(d: Drawing, side: int) -> None:
    """A few curved lashes flicking out at the outer corner."""
    corner, _ = to_sheet(eye_point(side, CORNERS[1], 0.0)[None])
    middle, _ = to_sheet(eye_point(side, 0.0, 0.0)[None])
    out = (corner - middle)[0]
    out /= np.linalg.norm(out)
    for k, a in enumerate(np.linspace(0.5, 1.15, 6)):
        top = lid_heights(np.array([a]))[0][0]
        root, _ = to_sheet(eye_point(side, a, top, LID)[None])
        lean = 0.25 + 0.22 * k
        way = np.array([0.0, -1.0]) * (1.0 - lean) + out * lean
        way /= np.linalg.norm(way)
        length = 8.0 + 8.0 * k / 5.0
        t = np.linspace(0, 1, 5)[:, None]
        path = root[0] + way * length * t + out * 4.0 * t**2
        d.stroke(path, BLACK, 0.7 * taper(5, 2) + 0.05)


#: The brow's spine (head-space x, y) from its head by the nose to its tail,
#: for the right (far) brow; the left one is its mirror.
BROW = ((0.035, 0.064), (0.07, 0.08), (0.12, 0.088), (0.165, 0.082), (0.205, 0.062))


def brow(d: Drawing, side: int) -> None:
    """A brow: hairs laid as they grow, thick by the nose and tapering out,
    over a violet under-tone."""
    rng = d.rng
    spine = curve(BROW, 60) * np.array([side, 1.0])
    tangent = np.gradient(spine, axis=0)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], -1) * side
    for k in range(3):
        pts = spine[5:52] + normal[5:52] * (0.003 * (k - 1))
        line(d, on_head(pts, 0.002), BLUE, swell(0.28, 0.3), 0.02)
    for _ in range(70):
        s = rng.beta(1.3, 2.2)
        i = int(s * 59)
        thick = 0.014 * (1.0 - s) ** 0.8 + 0.002
        start = spine[i] + normal[i] * thick * rng.uniform(-0.6, 0.5)
        rise = np.radians(55.0 * (1.0 - s / 0.45) if s < 0.45 else -14.0 * (s - 0.45) / 0.55)
        c, sn = np.cos(rise), np.sin(rise)
        way = tangent[i] * c + normal[i] * sn
        length = rng.uniform(0.018, 0.032) * (1.0 - 0.3 * s)
        t = np.linspace(0, 1, 5)[:, None]
        pts = start + way * length * t - normal[i] * 0.004 * t**2
        pen = BLACK if rng.random() < 0.7 else RED
        firm = rng.uniform(0.4, 0.62) * (1.0 - 0.35 * s)
        line(d, on_head(pts, 0.003), pen, firm * (0.15 + 0.85 * taper(5, 2)), 0.02)


def nose(d: Drawing) -> None:
    """The nose: a light line down the far side of the bridge and round the
    tip, the near wing, and the nostrils as small dark accents."""
    bridge = on_head(curve(((0.02, 0.02), (0.021, -0.06), (0.026, -0.12), (0.03, -0.15)), 16))
    line(d, bridge, BLUE, swell(0.3, 0.7), 0.01)
    line(d, bridge, RED, swell(0.3, 0.6), 0.01)
    wing = ((-0.054, -0.163), (-0.052, -0.174), (-0.044, -0.182), (-0.032, -0.186))
    for pen, firm in ((RED, 0.5), (BLUE, 0.35), (RED, 0.4)):
        line(d, on_head(curve(wing, 12), 0.002), pen, swell(firm, 0.4), 0.02)
    tip = ((0.032, -0.155), (0.02, -0.176), (0.0, -0.181), (-0.014, -0.179))
    line(d, on_head(curve(tip, 12), 0.002), RED, swell(0.4, 0.4), 0.02)
    near = ((-0.03, -0.182), (-0.018, -0.188), (-0.006, -0.184))
    far = ((0.008, -0.185), (0.02, -0.186), (0.03, -0.18))
    for knots, firm in ((near, 0.9), (far, 0.75)):
        pts = on_head(curve(knots, 8), 0.002)
        line(d, pts, BLACK_LINE, swell(firm, 0.45, 0.2), 0.03)
        line(d, pts, RED_LINE, swell(0.6, 0.5, 0.2), 0.03)


def lip_band(upper: bool, x: np.ndarray):
    """The heights of a lip's outer border and of the parting at ``x``."""
    border = curve(LIP_TOP if upper else LIP_BOTTOM, 80)
    parting = curve(LIP_PARTING, 80)
    return np.interp(x, *border.T), np.interp(x, *parting.T)


def lip_strokes(d: Drawing, upper: bool, pen, firm: float, slant: float, gap: float):
    """Strokes across a lip from its border to the parting, bowed round its
    fullness; on the lower lip a light patch is left on the fullest part."""
    rng = d.rng
    for x0 in np.arange(-0.085, 0.085, gap) + rng.uniform(0, gap):
        edge, part = lip_band(upper, np.array([x0]))
        span = abs(edge[0] - part[0])
        if span < 0.004:
            continue
        t = np.linspace(0.0, 1.0, 7)
        x = x0 + slant * (t - 0.5) * span + 0.25 * x0 * span * np.sin(np.pi * t) * 4
        edge, part = lip_band(upper, x)
        y = edge + (part - edge) * t
        press = firm * (0.35 + 0.65 * np.sin(np.pi * np.clip(t * 1.1, 0, 1)) ** 0.5)
        if not upper:
            shine = np.exp(-(((x + 0.012) / 0.028) ** 2 + ((t - 0.42) / 0.22) ** 2))
            press = press * (1.0 - 0.95 * shine)
        line(d, on_head(np.stack([x, y], -1), 0.002), pen, press, 0.02)


def lips(d: Drawing) -> None:
    """Full lips, the most saturated thing in the face: the upper darker (red
    crossed with blue and black), the lower bright red, the parting a dark
    lively line, thicker in the middle and at the tucked corners."""
    for slant in (-0.6, 0.6):
        lip_strokes(d, True, RED, 0.6, slant, 0.0055)
        lip_strokes(d, False, RED, 0.55, slant, 0.006)
    lip_strokes(d, True, RED, 0.5, 0.0, 0.006)
    lip_strokes(d, True, BLUE, 0.35, 0.0, 0.007)
    lip_strokes(d, False, RED, 0.5, 0.0, 0.0055)
    parting = on_head(curve(LIP_PARTING, 40), 0.001)

    def dip(u):
        return 0.45 + 0.4 * np.exp(-(((u - 0.5) / 0.15) ** 2)) + 0.4 * (
            np.exp(-((u / 0.07) ** 2)) + np.exp(-(((1 - u) / 0.07) ** 2))
        )

    line(d, parting, BLACK_LINE, dip, 0.02)
    line(d, parting + np.array([0.0, -0.0015, 0.0]), BLACK_LINE, lambda u: dip(u) * 0.8, 0.02)
    line(d, parting, RED_LINE, lambda u: dip(u) * 0.9, 0.02)
    for side in (-1, 1):
        corner = np.array([[0.08 * side, -0.253], [0.086 * side, -0.25], [0.089 * side, -0.247]])
        line(d, on_head(corner, 0.001), BLACK, swell(0.55, 0.4), 0.02)
    top = on_head(curve(LIP_TOP, 40), 0.001)
    line(d, top, RED, swell(0.35, 0.5, 0.1), 0.02)


# ----------------------------------------------------------------------
# the hair


#: The hair falls round a vertical axis through (x, z) = CROWN_AXIS, from
#: the crown at CROWN_TOP; where its ends hang beside and behind the face.
CROWN_AXIS = (0.02, -0.08)
CROWN_TOP = 0.56
ENDS = -0.62


def fringe_edge(x):
    """The height (head space) the fringe comes down to, across the brow."""
    wave = 0.012 * np.sin(55.0 * x + 0.5) + 0.008 * np.sin(97.0 * x)
    return FRINGE[0] + FRINGE[1] * (x / 0.24) ** 2 + 0.02 * x / 0.24 + wave


def hair_frame(h: np.ndarray):
    """For points of the hair: the angle round the axis, and how far along
    its fall each is (0 at the crown, 1 at the ends)."""
    x, y, z = h[..., 0], h[..., 1], h[..., 2]
    phi = np.arctan2(x - CROWN_AXIS[0], z - CROWN_AXIS[1])
    edge = fringe_edge(np.clip(x, -0.3, 0.3))
    front = filters.smoothstep(0.3, 0.2, np.abs(x)) * filters.smoothstep(-0.05, 0.1, z)
    front *= y > edge - 0.03
    end = front * edge + (1 - front) * ENDS
    t = np.clip((CROWN_TOP - y) / (CROWN_TOP - end), 0.0, 1.0)
    return phi, t


@dataclass
class Locks:
    """The locks the hair falls in: the angles that bound them round the
    axis, and how much darker or lighter each one is."""

    bounds: np.ndarray
    tone: np.ndarray


def make_locks(rng) -> Locks:
    widths = rng.uniform(0.1, 0.3, 80)
    bounds = -np.pi + np.concatenate([[0.0], np.cumsum(widths)])
    bounds = bounds[bounds < np.pi]
    bounds = np.append(bounds, np.pi)
    n = len(bounds) - 1
    return Locks(bounds, rng.uniform(-0.18, 0.15, n))


def strand_field(phi, t, locks: Locks, ends_high):
    """An angle-like field whose level lines are the strands: they sway
    gently together and, within each lock, draw together toward its end.
    Also: where across its lock each point is (-1 to 1), which lock, and how
    far out a strand there has been drawn in."""
    k = np.clip(np.searchsorted(locks.bounds, phi) - 1, 0, len(locks.tone) - 1)
    lo, hi = locks.bounds[k], locks.bounds[k + 1]
    centre = 0.5 * (lo + hi)
    tip = np.where(ends_high, 0.8, 0.9)
    gather = 1.0 + 3.0 * np.clip((t - tip) / (1.0 - tip), 0.0, 1.0) ** 1.4
    sway = 0.05 * np.sin(1.3 * np.pi * t + 2.0 * phi) + 0.025 * np.sin(2.9 * np.pi * t + 5.0 * phi)
    u = (phi - centre) / (0.5 * (hi - lo))
    field = centre + (phi - centre) * gather + sway
    return field, u, k, np.abs(u) * gather


def hair_tangent(h: np.ndarray) -> np.ndarray:
    """The way the strands run (down from the crown), in head space."""
    crown = np.array([CROWN_AXIS[0], CROWN_TOP + 0.05, CROWN_AXIS[1]])
    v = h - crown
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def hair_dark(model: Model, u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """How dark the hair is wanted: dark hair, darkest in the hollows between
    locks and away from the light, with a band of shine where the strands
    cross the light's reflection."""
    h = model.head
    n = model.normal @ HEAD
    tang = hair_tangent(h)
    tang = tang - np.sum(tang * n, -1, keepdims=True) * n
    tang /= np.maximum(np.linalg.norm(tang, axis=-1, keepdims=True), 1e-9)
    half = LIGHT @ HEAD + view_dir()
    half /= np.linalg.norm(half)
    across = np.sqrt(np.clip(1.0 - (tang @ half) ** 2, 0.0, 1.0))
    lit = np.clip((n @ (LIGHT @ HEAD) + 0.3) / 1.3, 0, 1) * (0.3 + 0.7 * model.shadow)
    shine = across**500 * lit * (1.0 - 0.6 * np.abs(u) ** 2)
    shine += np.exp(-(((n[..., 1] - 0.62) / 0.13) ** 2)) * lit * (1.0 - 0.5 * np.abs(u))
    h = model.head
    edge = fringe_edge(np.clip(h[..., 0], -0.3, 0.3))
    front = (np.abs(h[..., 0]) < 0.3) & (h[..., 2] > 0.0) & (h[..., 1] > -0.1)
    fine = filters.smoothstep(edge + 0.09, edge + 0.04, h[..., 1]) * front
    dark = 0.78 + 0.2 * (1.0 - lit) + 0.35 * np.abs(u) ** 2 + 0.4 * (1.0 - model.occlusion)
    return np.clip(dark - 0.9 * shine, 0.0, 1.2) * (1.0 - fine) - 0.5 * fine


#: The hair's strokes: (which of every six strands, ink, how dark the hair
#: must be for it).
HAIR_LAYERS = (
    (0, "red", -0.2),
    (3, "black", 0.1),
    (1, "black", 0.3),
    (4, "red", 0.4),
    (2, "black", 0.55),
    (5, "black", 0.7),
)


def hair_grid(d: Drawing, maps: Maps, locks: Locks, split: int, gap: float):
    """The field whose level lines are the strands, NaN where none is
    drawn, and how dark each place of the hair is wanted. The inside of the
    far curtain, seen behind the neck, hangs straight and is in shadow."""
    m = d.model
    phi, t = hair_frame(m.head)
    field, u, k, spread = strand_field(phi, t, locks, m.head[..., 1] > -0.1)
    dark = np.where(maps.hair, hair_dark(m, u, t) + locks.tone[k], 0.0)
    field = field * 0.35 * SCALE / gap * split
    n = m.normal @ HEAD
    out = m.head[..., [0, 2]] - np.array(CROWN_AXIS)
    inner = maps.hair & (np.sum(n[..., [0, 2]] * out, -1) < -0.02) & (m.head[..., 1] < 0.0)
    gx = np.arange(m.hit.shape[1])[None, :] * COARSE + 0.0 * m.head[..., 1]
    field = np.where(inner, 5000.0 + (gx + 6.0 * np.sin(m.head[..., 1] * 9.0)) / gap * split, field)
    dark = np.where(inner, 0.95 + 0.2 * np.abs(u), dark)
    seam = np.zeros(k.shape, bool)
    seam[:, 1:] |= k[:, 1:] != k[:, :-1]
    seam[1:, :] |= k[1:, :] != k[:-1, :]
    seam &= spread > 1.02
    border = near(inner, 1) * near(~inner, 1) > 0
    facing = n[..., 1] > -0.4
    shown = maps.hair & ~seam & ~border & ((spread < 1.0) | inner) & facing
    return np.where(shown, field, np.nan), dark


def draw_hair(d: Drawing, maps: Maps, locks: Locks) -> None:
    """The hair in long strokes down each lock, black among red."""
    split = 6
    grid, dark = hair_grid(d, maps, locks, split, 12.0)
    pens_ = {"red": RED, "black": BLACK, "blue": BLUE}
    lines = level_lines(grid, jump=1e9)
    for offset, ink, threshold in HAIR_LAYERS:
        firm = 0.5 if ink == "black" else 0.42
        layer = Layer(pens_[ink], 0, split, offset, threshold, 0.2, firm, 0.12)
        d.hatch(lines, layer, dark, reach=420.0)


def fringe_lock(d: Drawing, centre: float, half: float, dark: float) -> None:
    """One lock of the fringe: a bundle of strokes from under the hair above
    down over the forehead, drawing together into a few fine points at
    about brow height."""
    rng = d.rng
    edge = float(fringe_edge(np.array([centre]))[0])
    top = edge + 0.15
    count = max(int(2 * half / 0.003), 3)
    lean = rng.normal(0.0, 0.01)
    tips = centre + half * np.sort(rng.uniform(-0.55, 0.55, int(rng.integers(2, 4))))
    for k in range(count):
        s = (k + rng.uniform(0.2, 0.8)) / count * 2.0 - 1.0
        x0 = centre + half * s
        tip = tips[np.argmin(np.abs(tips - x0))]
        t = np.linspace(0.0, 1.0, 14)
        x = x0 + (tip - x0) * t**3 + lean * t**2
        end = edge - rng.uniform(-0.006, 0.014) + 0.012 * abs(s)
        y = top + (end - top) * t
        pts = on_head(np.stack([x, y], -1), 0.012 + 0.012 * (1.0 - t))
        pen = BLACK if rng.random() < 0.5 + 0.4 * dark else RED
        firm = rng.uniform(0.5, 0.7) * (0.75 + 0.4 * dark)
        press = firm * np.clip(t / 0.25, 0.0, 1.0) * np.clip((1.0 - t) / 0.3, 0.05, 1.0) ** 0.8
        line(d, pts, pen, press, 0.06)


def fringe(d: Drawing) -> None:
    """The fringe as separate locks across the brow, with slivers of
    forehead between them; darker on the side away from the light."""
    rng = d.rng
    x = -0.3
    while x < 0.3:
        half = rng.uniform(0.024, 0.04)
        centre = x + half
        dark = float(np.clip(0.4 + 1.2 * centre, 0.0, 1.0))
        fringe_lock(d, centre, half, dark)
        x = centre + half + rng.uniform(-0.012, 0.003)


def strays(d: Drawing, maps: Maps, count: int = 34) -> None:
    """A few loose hairs straying beyond the outline of the hair, in loops
    off its outermost strokes."""
    rng = d.rng
    ground = near(~d.model.hit, 2)
    field = filters.box_blur(maps.hair.astype(float), 1) + 0.5
    outline = [
        resample((p + 0.5) * COARSE, 3.0) for _, p in level_lines(field, jump=1e9) if len(p) > 40
    ]
    centre = np.array(CENTRE)
    for _ in range(count):
        pts = outline[int(rng.integers(len(outline)))]
        n = int(rng.integers(12, 40))
        if len(pts) <= n + 2:
            continue
        a = int(rng.integers(0, len(pts) - n))
        seg = pts[a : a + n]
        if sample(ground, seg).min() < 0.5 or seg[:, 1].mean() > CENTRE[1] - 60:
            continue
        out = seg - centre
        out /= np.linalg.norm(out, axis=-1, keepdims=True)
        lift = rng.uniform(2.0, 9.0) * np.sin(np.linspace(0, np.pi, n)) ** 0.7
        path = seg + out * (lift + rng.uniform(-1.0, 1.5))[:, None]
        pen = BLACK if rng.random() < 0.7 else RED
        d.stroke(path[::2], pen, (rng.uniform(0.25, 0.45) * taper(n, 6))[::2])


# ----------------------------------------------------------------------
# the skin and the cloth

#: The skin's hatching: (family, ink, which of every four lines, how much
#: of the ink a place must want for it).
FACE_LAYERS = (
    (0, "red", 0, 0.1),
    (1, "red", 0, 0.4),
    (2, "red", 0, 0.62),
    (0, "red", 2, 0.78),
    (1, "blue", 2, 0.05),
    (2, "blue", 2, 0.3),
    (0, "blue", 1, 0.55),
    (2, "black", 1, 0.1),
    (1, "black", 1, 0.5),
)
#: Families of planes for the neck and chest (normals in body space): round
#: the neck, down it, and across both.
BODY_FAMILIES = ((0.0, 0.85, 0.5), (0.9, -0.2, 0.4), (-0.75, -0.5, 0.45))
INKS = {"red": RED, "blue": BLUE, "black": BLACK}
#: How much the depth of the face is exaggerated for its hatching, so that
#: the lines bend more over its forms.
RELIEF = 1.5


def lay_layers(d: Drawing, families, layers, maps: Maps, reach: float) -> None:
    dens = {"red": maps.red, "blue": maps.blue, "black": maps.black}
    for family, ink, offset, threshold in layers:
        layer = Layer(INKS[ink], family, 4, offset, threshold)
        d.hatch(families[family], layer, dens[ink], reach)


def draw_skin(d: Drawing, maps: Maps) -> None:
    """The face and the neck, hatched along the sections of their forms."""
    skin = np.nan_to_num(maps.skin) * np.array([1.0, 1.0, RELIEF])
    coords = smooth_within(skin, maps.face, 6)
    fams = [family_lines(coords, maps.face, n, 8.5, 4) for n in HEAD_FAMILIES]
    lay_layers(d, fams, FACE_LAYERS, maps, 110.0)
    coords = smooth_within(maps.body, maps.neck, 6)
    fams = [family_lines(coords, maps.neck, n, 7.5, 4) for n in BODY_FAMILIES]
    lay_layers(d, fams, FACE_LAYERS, maps, 130.0)


def cloth_dark(d: Drawing, maps: Maps) -> np.ndarray:
    """How dark the top is wanted: lit from the left, with folds, and fading
    out toward the bottom and the sides of the sheet."""
    m = d.model
    b = maps.body
    lam = np.clip(m.normal @ LIGHT, -1.0, 1.0)
    lit = np.clip((lam + 0.3) / 1.3, 0.0, 1.0) * (0.35 + 0.65 * m.shadow)
    folds = 0.5 + 0.5 * np.sin(14.0 * (b[..., 0] * 0.8 - b[..., 1] * 0.6) + 3.0 * b[..., 0] ** 2)
    dark = 0.15 + 0.75 * (1.0 - lit) + 0.25 * folds**3
    dark = np.where(maps.hair, 0.75, dark)
    gy, gx = np.mgrid[0 : m.hit.shape[0], 0 : m.hit.shape[1]] * COARSE
    across = 50.0 * np.sin(gx * 0.011 + 1.0) + 25.0 * np.sin(gx * 0.029)
    down = 40.0 * np.sin(gy * 0.013) + 20.0 * np.sin(gy * 0.037 + 2.0)
    fade = filters.smoothstep(HEIGHT - 20, HEIGHT - 260, gy + across)
    fade *= filters.smoothstep(20, 180, gx + down) * filters.smoothstep(WIDTH - 20, WIDTH - 180, gx - down)
    return np.where(maps.cloth, dark * fade, 0.0)


#: The cloth's hatching: (family, which of every two lines, threshold).
CLOTH_LAYERS = ((0, 0, 0.08), (1, 0, 0.3), (0, 1, 0.55), (1, 1, 0.72))
#: The top's lines sweep out from the base of the neck (body space) down
#: the chest and round the shoulders, in two spirals crossing each other.
NECK_BASE = (0.0, -0.5, -0.08)
SWEEP = (0.5, 0.7)


def cloth_lines(body: np.ndarray, mask: np.ndarray, gap: float, split: int):
    v = body - np.asarray(NECK_BASE)
    r = np.linalg.norm(v, axis=-1)
    a = np.arctan2(v[..., 0], -v[..., 1])
    out = []
    for sign in (1.0, -1.0):
        field = (SWEEP[0] * a + sign * SWEEP[1] * r) * SCALE / gap * split
        out.append(level_lines(np.where(mask, field, np.nan), jump=1e9))
    return out


def draw_cloth(d: Drawing, maps: Maps) -> None:
    """The top, in blue: large sweeping arcs that follow the body under it,
    crossed into an open lattice and hatched again in the folds."""
    dark = cloth_dark(d, maps)
    fams = cloth_lines(maps.body, maps.cloth, 14.0, 2)
    for family, offset, threshold in CLOTH_LAYERS:
        layer = Layer(BLUE_LINE, family, 2, offset, threshold, 0.3, 0.33)
        d.hatch(fams[family], layer, dark, 240.0)


# ----------------------------------------------------------------------
# contours


def near(mask: np.ndarray, r: int = 3) -> np.ndarray:
    """1 within ``r`` cells of ``mask``, else 0."""
    return (filters.box_blur(mask.astype(float), r) > 0.01).astype(float)


def edge_strokes(d: Drawing, mask: np.ndarray, weight: np.ndarray, pens_, wobble=0.6):
    """The edge of ``mask`` drawn as a living line: as firm as ``weight``
    says (lifted where that is nothing), restated once per pen, each stroke
    a little off the last."""
    field = filters.box_blur(mask.astype(float), 1) + 0.5
    for _, grid_pts in level_lines(field, jump=1e9):
        if len(grid_pts) < 4:
            continue
        pts = resample((grid_pts + 0.5) * COARSE, 3.0)
        if len(pts) < 4:
            continue
        w = sample(weight, pts)
        for k, pen in enumerate(pens_):
            shift = d.rng.normal(0.0, wobble)
            ww = w * d.rng.uniform(0.8, 1.1) * (1.0 - 0.18 * k)
            for a, b in runs_of(ww > 0.08):
                if b - a < 4:
                    continue
                seg = pts[a:b] + shift
                d.stroke(seg[::2], pen, (ww[a:b] * taper(b - a, 5))[::2])


def draw_contours(d: Drawing, maps: Maps) -> None:
    """Firm lines where a form overlaps what is behind it (the jaw and chin
    over the neck, the far cheek against the hair, the neck against the
    top), light or none on the lit side."""
    m = d.model
    head = maps.face | maps.eyes | maps.lips
    neck, hair = near(maps.neck), near(maps.hair)
    ground = near(~m.hit)
    shade = filters.box_blur(maps.dark, 3)
    far = filters.smoothstep(0.0, 0.12, maps.skin[..., 0]) * (maps.skin[..., 1] < 0.02)
    w = 0.9 * neck + 0.6 * hair * np.nan_to_num(far) + 0.5 * ground
    w = np.minimum(w, 0.9) * (0.15 + 0.85 * filters.smoothstep(0.2, 0.7, shade))
    edge_strokes(d, head, w, (BLACK_LINE, BLUE_LINE, RED_LINE))
    w = (0.7 * near(maps.cloth) + 0.45 * ground) * (1.0 - hair) * (0.3 + 0.7 * shade)
    edge_strokes(d, maps.neck, np.minimum(w, 0.75), (BLUE_LINE, RED_LINE))
    w = 0.45 * ground * filters.box_blur(cloth_dark(d, maps), 4) * 2.0
    edge_strokes(d, maps.cloth, np.minimum(w, 0.55), (BLUE_LINE, BLUE_LINE), 1.2)


# ----------------------------------------------------------------------


def draw(model: Model, seed: int = 1911) -> Drawing:
    """The whole portrait."""
    d = Drawing(model, seed)
    maps = design(model)
    draw_cloth(d, maps)
    draw_skin(d, maps)
    for side in (1, -1):
        iris_strokes(d, side)
        lids(d, side)
        brow(d, side)
    nose(d)
    lips(d)
    draw_contours(d, maps)
    draw_hair(d, maps, make_locks(np.random.default_rng(seed)))
    fringe(d)
    strays(d, maps)
    return d


def main() -> None:
    model = build_model()
    drawing = draw(model)
    out = Path(__file__).parent / "output" / "portrait.png"
    write_png(str(out), np.round(drawing.sheet.render() * 255).astype(np.uint8))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
