"""Geometry helpers for flat pattern drafting (units: cm, y up)."""
import math
import numpy as np

import os
N = int(os.environ.get("PATTERN_N", "1200"))  # samples per curve segment


def P(x, y):
    return np.array([x, y], float)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def dirdeg(d):
    return P(math.cos(math.radians(d)), math.sin(math.radians(d)))


def angdeg(v):
    return math.degrees(math.atan2(v[1], v[0]))


def rot(p, c, deg):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(p, float) - c) @ R.T + c


def rotv(v, deg):
    return rot(v, P(0, 0), deg)


def plen(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _at(pts, s):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    i = int(np.clip(np.searchsorted(d, s) - 1, 0, len(pts) - 2))
    while d[i + 1] - d[i] < 1e-12 and i < len(pts) - 2:
        i += 1
    t = (s - d[i]) / (d[i + 1] - d[i])
    return pts[i] + t * (pts[i + 1] - pts[i])


def tan_start(pts, h=0.004):
    """2nd-order one-sided tangent on uniform arc-length points (robust to uneven sampling)."""
    p0, p1, p2 = pts[0], _at(pts, h), _at(pts, 2 * h)
    return unit(-3 * p0 + 4 * p1 - p2)


def tan_end(pts, h=0.004):
    return -tan_start(pts[::-1], h)


def cluster(n):
    """Parameter in [0,1], denser toward both ends (accurate end tangents / corner angles)."""
    u = np.linspace(0, 1, n)
    return 0.5 * u + 0.5 * (1 - np.cos(np.pi * u)) / 2


def line(p0, p1, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * p0 + t * p1


def hermite(p0, d0, p1, d1, k0=1.0, k1=None, n=N):
    """Cubic Hermite from p0 (direction d0) to p1 (arrival direction d1).
    Tangent magnitudes = k * chord."""
    if k1 is None:
        k1 = k0
    c = np.linalg.norm(p1 - p0)
    m0 = unit(d0) * k0 * c
    m1 = unit(d1) * k1 * c
    t = cluster(n)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def chain(pts, dirs, ks=None):
    """G1 Hermite chain through pts with given unit directions at each point."""
    out = []
    for i in range(len(pts) - 1):
        k = 1.0 if ks is None else ks[i]
        if isinstance(k, tuple):
            seg = hermite(pts[i], dirs[i], pts[i + 1], dirs[i + 1], k[0], k[1])
        else:
            seg = hermite(pts[i], dirs[i], pts[i + 1], dirs[i + 1], k)
        out.append(seg if i == 0 else seg[1:])
    return np.vstack(out)


def split_at_line(pts, L0, Ldir):
    n = P(-Ldir[1], Ldir[0])
    f = (pts - L0) @ n
    idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) <= 0)[0][0]
    t = f[idx] / (f[idx] - f[idx + 1])
    X = pts[idx] + t * (pts[idx + 1] - pts[idx])
    a = pts[: idx + 1] if t > 1e-9 else pts[:idx]
    b = pts[idx + 1:] if t < 1 - 1e-9 else pts[idx + 2:]
    return np.vstack([a, X]), np.vstack([X, b])


def point_at_len(pts, s):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    i = np.searchsorted(d, s) - 1
    i = max(0, min(i, len(pts) - 2))
    t = (s - d[i]) / (d[i + 1] - d[i])
    return pts[i] + t * (pts[i + 1] - pts[i]), unit(pts[i + 1] - pts[i])


def corner_angle(a_end_at_v, b_start_at_v):
    """Interior angle between curve a (ending at V) and curve b (starting at V)."""
    u = -tan_end(a_end_at_v)
    v = tan_start(b_start_at_v)
    return math.degrees(math.acos(max(-1, min(1, float(u @ v)))))


def angle_between_out(d1, d2):
    return math.degrees(math.acos(max(-1, min(1, float(unit(d1) @ unit(d2))))))


def max_turn(pts):
    """Largest direction change between consecutive samples (deg) -> kink detector."""
    d = np.diff(pts, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-9]
    a = np.arctan2(d[:, 1], d[:, 0])
    da = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.degrees(da.max())) if len(da) else 0.0


def rev(pts):
    return pts[::-1].copy()


def spline_through(pts, d0, d1, n=N):
    """C2 parametric cubic spline through pts (chord-length param), clamped end directions."""
    from scipy.interpolate import CubicSpline
    pts = np.asarray(pts, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    cs = CubicSpline(s, pts, bc_type=((1, unit(d0)), (1, unit(d1))), axis=0)
    t = cluster(n) * s[-1]
    return cs(t)


def inflections(pts, trim=0.02):
    """Number of curvature sign changes (ignores near-straight noise)."""
    d1 = np.gradient(pts, axis=0)
    d2 = np.gradient(d1, axis=0)
    k = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    m = int(len(k) * trim)
    k = k[m:len(k) - m]
    thr = np.abs(k).max() * 0.02
    sg = np.sign(k[np.abs(k) > thr])
    return int(np.sum(sg[1:] != sg[:-1]))
