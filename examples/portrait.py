"""Portrait study (work in progress)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from murex import filters

WIDTH, HEIGHT = 1100, 1400
#: The model is computed on a grid this many times coarser than the sheet.
COARSE = 2
#: Pixels of the sheet per unit of the model (about a head's height).
SCALE = 760.0
#: Where on the sheet the middle of the head (between the eyes) lands.
CENTRE = (560.0, 560.0)


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Turns head space (x to the sitter's left, y up, z out of the face)
    into camera space: turned by ``yaw``, nodded by ``pitch``, tilted by
    ``roll`` (degrees)."""
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ rx @ ry


HEAD = rotation(-26.0, -6.0, 5.0)
HEAD_AT = np.array([0.0, 0.0, 0.0])
BODY = rotation(-10.0, 0.0, 1.0)
BODY_AT = np.array([0.03, 0.0, -0.05])


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


def mirrored(p):
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    return q


def head_sdf(p):
    """Distance to the head's skin, in head space."""
    q = mirrored(p)
    # the face narrows toward the chin
    y = p[:, 1]
    narrow = 1.0 + 0.55 * np.clip(-(y + 0.05), 0.0, None)
    f = p.copy()
    f[:, 0] *= narrow
    d = ellipsoid(p, (0.0, 0.10, -0.07), (0.295, 0.40, 0.40))
    d = smin(d, ellipsoid(f, (0.0, -0.13, 0.04), (0.275, 0.36, 0.27)) * 0.8, 0.08)
    d = smin(d, ellipsoid(q, (0.165, -0.07, 0.17), (0.10, 0.075, 0.10)), 0.05)
    d = smin(d, sphere(p, (0.0, -0.385, 0.20), 0.07), 0.06)
    # brow ridge and sockets
    d = smin(d, ellipsoid(q, (0.10, 0.075, 0.245), (0.09, 0.035, 0.05)), 0.03)
    d = smax(d, -ellipsoid(q, (0.10, 0.0, 0.29), (0.068, 0.048, 0.05)), 0.035)
    d = smin(d, sphere(q, (0.10, 0.0, 0.212), 0.068), 0.008)
    # nose
    nose = capsule(p, (0.0, 0.05, 0.285), (0.0, -0.115, 0.375), 0.019)
    nose = smin(nose, sphere(p, (0.0, -0.13, 0.368), 0.03), 0.02)
    nose = smin(nose, sphere(q, (0.03, -0.148, 0.328), 0.025), 0.02)
    d = smin(d, nose, 0.025)
    # the barrel of the mouth and the lips
    d = smin(d, ellipsoid(p, (0.0, -0.25, 0.19), (0.13, 0.10, 0.12)), 0.05)
    d = smin(d, ellipsoid(p, (0.0, -0.232, 0.285), (0.062, 0.020, 0.03)), 0.02)
    d = smin(d, ellipsoid(p, (0.0, -0.272, 0.28), (0.055, 0.022, 0.03)), 0.02)
    return d


def body_sdf(p):
    """Distance to the neck, shoulders and chest, in body space."""
    neck = capsule(p, (0.0, -0.15, -0.06), (0.0, -0.80, -0.05), 0.112)
    yoke = capsule(p, (-0.40, -0.83, -0.10), (0.40, -0.83, -0.10), 0.13)
    chest = ellipsoid(p, (0.0, -1.18, -0.02), (0.50, 0.40, 0.24))
    arms = capsule(mirrored(p), (0.43, -0.86, -0.08), (0.52, -1.6, -0.08), 0.15)
    d = smin(yoke, chest, 0.12)
    d = smin(d, arms, 0.06)
    return smin(d, neck, 0.10)


def to_head(c):
    return (c - HEAD_AT) @ HEAD


def to_body(c):
    return (c - BODY_AT) @ BODY


def scene_sdf(c):
    """Distance in camera space, and which part is nearest (0 head, 1 body)."""
    h = head_sdf(to_head(c))
    b = body_sdf(to_body(c))
    return np.minimum(h, b), (b < h).astype(np.int8)


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


def grid_coords():
    h, w = HEIGHT // COARSE, WIDTH // COARSE
    gy, gx = np.mgrid[0:h, 0:w]
    sx = (gx + 0.5) * COARSE
    sy = (gy + 0.5) * COARSE
    x = (sx - CENTRE[0]) / SCALE
    y = -(sy - CENTRE[1]) / SCALE
    return x, y


def march(x, y):
    z = np.full(x.shape, 1.2)
    active = np.ones(x.shape, bool)
    hit = np.zeros(x.shape, bool)
    for _ in range(160):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        c = np.stack([x[idx], y[idx], z[idx]], axis=-1)
        d, _ = scene_sdf(c)
        z[idx] -= 0.7 * d
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


LIGHT = np.array([-0.55, 0.62, 0.56])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def soft_shadow(c, n):
    res = np.ones(len(c))
    t = np.full(len(c), 0.01)
    start = c + n * 0.004
    for _ in range(48):
        d = scene_sdf(start + LIGHT * t[:, None])[0]
        res = np.minimum(res, 10.0 * d / t)
        t += np.clip(d, 0.006, 0.06)
    return np.clip(res, 0.0, 1.0)


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
    sh[idx] = soft_shadow(c[idx], n[idx])
    _, part = scene_sdf(c)
    lam = np.clip((n @ LIGHT + 0.25) / 1.25, 0.0, 1.0)
    light = lam * (0.35 + 0.65 * sh)
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
    )
