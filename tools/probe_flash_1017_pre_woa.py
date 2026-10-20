"""Source-derived CPU oracle reused by Flash-1017 primitive regressions.

Transferred from 166a211fa98f86f223aa346b67a4c5a16f775696; the historical
campaign CLI is intentionally omitted. Not an executed official kernel oracle.
"""
import numpy as np


def bf16(x):
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def e4m3(x):
    # Enumerated positive finite E4M3FN codes; nearest, ties to even code.
    codes = np.arange(127, dtype=np.uint32)
    exp, frac = codes >> 3, codes & 7
    levels = np.where(exp == 0, frac * 2.**-9, (1 + frac / 8) * 2.**(exp.astype(int)-7)).astype(np.float32)
    a = np.minimum(np.abs(np.asarray(x, np.float32)), np.float32(448))
    hi = np.minimum(np.searchsorted(levels, a), 126)
    lo = np.maximum(hi - 1, 0)
    dl, dh = a - levels[lo], levels[hi] - a
    code = np.where((dh < dl) | ((dh == dl) & ((hi & 1) == 0)), hi, lo)
    return np.copysign(levels[code], x)


def reference(x):
    # Matches the fixed source's FP32 multiply by 1/448 and bitwise ceil-log2.
    x = bf16(x)
    g = x.reshape(*x.shape[:-1], -1, 32)
    a = np.maximum(np.max(np.abs(g), axis=-1, keepdims=True), np.float32(1e-4))
    bits = (a * np.float32(1/448)).view(np.uint32)
    exponent = ((bits >> 23) & 255).astype(np.int32) - 127 + ((bits & 0x7fffff) != 0)
    scale = ((exponent + 127).astype(np.uint32) << 23).view(np.float32)
    q = e4m3(np.clip(g / scale, -448, 448))
    return bf16(q * scale).reshape(x.shape), scale
