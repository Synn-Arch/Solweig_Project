#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the primitive-parity pin vectors (TASKS T12) as DATA.

Runs on a host with numba (the GPU host or any canonical-CPU environment):
computes the CPU canonical primitive results
(solweig_core/numba_cpu/math_compat — itself pinned against the frozen
oracle bits) over fixed lattices and writes them as raw uint32 hex to
``native/cuda/data/primitive_pins.json``. The CUDA tests compare device
outputs against these BITS — torch is never involved, so the pins are
torch-version-independent (the CPU canonical port is the frozen
reference, exactly like the T03/T07/T08 committed tables).

Lattices cover: full domain sweeps, subnormals, the UTCI physical domain,
signed zeros, infinities, multiple NaN payloads, and the exact boundaries
of every branch in the SLEEF ports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from solweig_core.numba_cpu import math_compat as mc  # noqa: E402

OUT = REPO_ROOT / "native" / "cuda" / "data" / "primitive_pins.json"

F32 = np.float32
U32 = np.uint32


def bits(x: float) -> int:
    return int(F32(x).view(U32))


def hexes(arr_f32: np.ndarray) -> list[str]:
    return [f"{int(b):08x}" for b in
            np.ascontiguousarray(arr_f32, dtype=np.float32).view(U32)]


def hexes_u32(arr: np.ndarray) -> list[str]:
    return [f"{int(b):08x}" for b in np.ascontiguousarray(arr, dtype=U32)]


SPECIAL_BITS = [
    0x00000000,  # +0
    0x80000000,  # -0
    0x00000001,  # smallest subnormal
    0x007fffff,  # largest subnormal
    0x00800000,  # FLT_MIN
    0x7f7fffff,  # FLT_MAX
    0x3f800000,  # 1.0
    0xbf800000,  # -1.0
    0x7f800000,  # +inf
    0xff800000,  # -inf
    0x7fc00000,  # qnan
    0xffc00000,  # -qnan
    0x7f800001,  # snan payload 1
    0x7fa00000,  # qnan payload 0x200000
]


def lattice_log(lo: float, hi: float, n: int, sign: float = 1.0) -> np.ndarray:
    vals = np.exp(np.linspace(np.log(lo), np.log(hi), n)).astype(np.float32)
    if sign < 0:
        vals = -vals
    return vals


def with_specials(arr: np.ndarray, extra_bits: list[int]) -> np.ndarray:
    sp = np.array(extra_bits, dtype=U32).view(np.float32)
    return np.concatenate([arr, sp])


def main() -> int:
    rng = np.random.default_rng(20240912)
    sets: dict[str, dict] = {}

    # --- exp -----------------------------------------------------------------
    x = np.concatenate([
        np.linspace(-110.0, 105.0, 2401).astype(np.float32),
        lattice_log(1e-38, 1e37, 400),
        (-lattice_log(1e-38, 1e37, 400)).astype(np.float32),
        np.linspace(-35.0, 8.0, 401).astype(np.float32),  # UTCI es domain
        rng.uniform(-3.0, 3.0, 500).astype(np.float32),
        np.array(SPECIAL_BITS, dtype=U32).view(np.float32),
    ])
    x = np.unique(np.ascontiguousarray(x, dtype=np.float32))
    sets["exp"] = {"x": hexes(x),
                   "y": hexes(np.array([mc.sleef_expf(F32(v)) for v in x],
                                       dtype=np.float32))}

    # --- log -----------------------------------------------------------------
    x = np.concatenate([
        np.exp(np.linspace(np.log(5e-39), np.log(3.0e38), 2401)).astype(np.float32),
        np.linspace(173.0, 333.0, 401).astype(np.float32),  # UTCI tk domain
        1.0 + np.array([F32(2.0) ** np.float32(e) for e in range(-8, 1)],
                       dtype=np.float32),
        1.0 - np.array([F32(2.0) ** np.float32(e) for e in range(-8, 1)],
                       dtype=np.float32),
        np.array([bits(v) for v in (0x00800000, 0x00800001, 0x007fffff,
                                     0x3f7fffff, 0x3f800001, 0x40490fdb)],
                 dtype=U32).view(np.float32),
        rng.uniform(0.01, 100.0, 500).astype(np.float32),
        np.array(SPECIAL_BITS, dtype=U32).view(np.float32),
    ])
    x = np.unique(np.ascontiguousarray(x, dtype=np.float32))
    sets["log"] = {"x": hexes(x),
                   "y": hexes(np.array([mc.sleef_logf_u1(F32(v)) for v in x],
                                       dtype=np.float32))}

    # --- powf (vector body): x lattice x y exponent set ----------------------
    xs = np.concatenate([
        np.linspace(0.25, 4.0, 301).astype(np.float32),
        lattice_log(1e-30, 1e30, 200),
        (-lattice_log(1e-8, 1e8, 100)).astype(np.float32),
        np.array([bits(0x3f7fffff), bits(0x3f800001), bits(0x3f800000)],  # ~1
                 dtype=U32).view(np.float32),
        np.array(SPECIAL_BITS, dtype=U32).view(np.float32),
    ])
    ys = np.array([4.0, 5.0, 6.0, 0.5, -0.5, 1.5, 2.5, 12.0, -3.0, 0.0,
                   float("inf"), float("-inf"), float("nan"),
                   16777216.5,  # > 2^24 non-integer (yisint via BIG24)
                   16777218.0],  # > 2^24 integer
                  dtype=np.float32)
    xx = np.repeat(xs, ys.size)
    yy = np.tile(ys, xs.size)
    sets["powf"] = {
        "x": hexes(xx), "e": hexes(yy),
        "y": hexes(np.array([mc.sleef_powf(F32(a), F32(b))
                             for a, b in zip(xx, yy)], dtype=np.float32)),
    }

    # --- opmath pow (scalar-tail lanes): x lattice x e in {4,5,6} -------------
    xs = np.concatenate([
        np.linspace(0.001, 30.0, 601).astype(np.float32),
        lattice_log(1e-38, 1e37, 400),
        (-lattice_log(1e-8, 1e8, 200)).astype(np.float32),
        np.exp(np.linspace(np.log(2.0), np.log(6e7), 200)).astype(np.float32),
        # x^6 overflow/underflow boundaries: x^6 > f32max -> x > 3538.8;
        # x^6 subnormal -> x < 1.5e-6
        np.linspace(3500.0, 3600.0, 101).astype(np.float32),
        np.exp(np.linspace(np.log(1.4e-6), np.log(1.6e-6), 101)).astype(np.float32),
        np.array(SPECIAL_BITS, dtype=U32).view(np.float32),
        rng.uniform(-5.0, 5.0, 500).astype(np.float32),
    ])
    for e in (4, 5, 6):
        sets[f"opmath_pow{e}"] = {
            "x": hexes(xs),
            "y": hexes(np.array(
                [mc._torch_pow_scalar(F32(v), F32(float(e)), True) for v in xs],
                dtype=np.float32)),
        }

    # --- plain IEEE arith + contraction witness --------------------------------
    a = np.concatenate([
        rng.uniform(-1e6, 1e6, 2000).astype(np.float32),
        lattice_log(1e-30, 1e30, 1000),
        (-lattice_log(1e-30, 1e30, 1000)).astype(np.float32),
    ])
    b = np.concatenate([
        rng.uniform(-1e6, 1e6, 2000).astype(np.float32),
        lattice_log(1e-30, 1e30, 1000),
        (-lattice_log(1e-30, 1e30, 1000)).astype(np.float32),
    ])
    # subnormal operands and results (FTZ witness)
    a[:250] = np.linspace(1e-45, 1e-38, 250).astype(np.float32)
    b[:250] = np.linspace(1e-45, 1e-38, 250).astype(np.float32)
    rng.shuffle(a)
    rng.shuffle(b)
    c = rng.uniform(-1e3, 1e3, a.size).astype(np.float32)
    # FMA-contraction-sensitive: a*b with |a*b| >> |c| and knife-edge rounding
    a[:500] = np.sqrt(rng.uniform(1e6, 1e12, 500)).astype(np.float32)
    b[:500] = np.sqrt(rng.uniform(1e6, 1e12, 500)).astype(np.float32)
    c[:500] = np.spacing(a[:500] * b[:500]).astype(np.float32) * rng.choice(
        np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float32), 500)
    sets["arith"] = {
        "a": hexes(a), "b": hexes(b), "c": hexes(c),
        "add": hexes(np.asarray(a + b, dtype=np.float32)),
        "sub": hexes(np.asarray(a - b, dtype=np.float32)),
        "mul": hexes(np.asarray(a * b, dtype=np.float32)),
        "div": hexes(np.asarray(a / b, dtype=np.float32)),
        "contract": hexes(np.asarray((a * b) + c, dtype=np.float32)),
    }

    # --- torch maximum + first-NaN wrappers ------------------------------------
    nan_q = np.array([0x7fc00000], dtype=U32).view(np.float32)[0]
    nan_q2 = np.array([0xffc00001], dtype=U32).view(np.float32)[0]

    def torch_maximum(p: float, q: float) -> float:  # march._torch_maximum
        return p if (p != p or p > q) else q

    am = np.concatenate([
        rng.uniform(-100.0, 100.0, 1000).astype(np.float32),
        np.full(50, 1.0, dtype=np.float32),        # ties
        np.full(10, 0.0, dtype=np.float32),
        np.full(10, -0.0, dtype=np.float32),
        np.full(5, nan_q, dtype=np.float32),
    ])
    bm = np.concatenate([
        rng.uniform(-100.0, 100.0, 1000).astype(np.float32),
        np.full(50, 1.0, dtype=np.float32),
        np.full(10, -0.0, dtype=np.float32),       # +0 vs -0 ties
        np.full(10, 0.0, dtype=np.float32),
        rng.uniform(-100.0, 100.0, 5).astype(np.float32),
    ])
    sets["maximum"] = {
        "a": hexes(am), "b": hexes(bm),
        "y": hexes(np.array([torch_maximum(F32(p), F32(q))
                             for p, q in zip(am, bm)], dtype=np.float32)),
    }
    an = np.concatenate([
        rng.uniform(-100.0, 100.0, 200).astype(np.float32),
        np.full(30, nan_q, dtype=np.float32),
        np.full(30, nan_q2, dtype=np.float32),
    ])
    bn = np.concatenate([
        rng.uniform(-100.0, 100.0, 200).astype(np.float32),
        rng.uniform(-100.0, 100.0, 60).astype(np.float32),
    ])
    sets["nadd"] = {"a": hexes(an), "b": hexes(bn),
                    "y": hexes(np.array([mc.nadd(F32(p), F32(q))
                                         for p, q in zip(an, bn)],
                                       dtype=np.float32))}
    sets["nmul"] = {"a": hexes(an), "b": hexes(bn),
                    "y": hexes(np.array([mc.nmul(F32(p), F32(q))
                                         for p, q in zip(an, bn)],
                                       dtype=np.float32))}

    # --- torch chunk-layout lane predicate (table check) ------------------------
    sizes = [1, 7, 8, 9, 15, 63, 64, 65, 100, 4095, 4096, 32767, 32768,
             32769, 65535, 65536, 100000, 262144, 263000, 500000]
    Ts = [1, 2, 8, 16]
    queries, expected = [], []
    for n in sizes:
        for T in Ts:
            if n <= 4096:
                idxs = list(range(n))
            else:
                wanted = set(range(64)) | {
                    n // 2 - 1, n // 2, n // 2 + 1, n - 9, n - 8, n - 7,
                    n - 1, 32767, 32768, 32769, 65535, 65536,
                }
                idxs = sorted(wanted & set(range(n)))
            for i in idxs:
                queries.append((i, n, T, mc.TORCH_PAR_GRAIN))
                expected.append(bool(mc._is_libm_lane(i, n, T,
                                                      mc.TORCH_PAR_GRAIN)))
    sets["is_libm_lane"] = {
        "i": [str(q[0]) for q in queries],
        "n": [str(q[1]) for q in queries],
        "T": [str(q[2]) for q in queries],
        "grain": [str(q[3]) for q in queries],
        "y": [1 if v else 0 for v in expected],
    }

    payload = {
        "schema": "sw-cuda-primitive-pins/1",
        "generator": "native/cuda/tools/gen_primitive_pins.py",
        "reference": "solweig_core/numba_cpu/math_compat.py (canonical CPU)",
        "numpy": np.__version__,
        "sets": sets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); "
          f"{len(sets)} sets, "
          + ", ".join(f"{k}:{len(v.get('y', v.get('i', [])))}" for k, v in sets.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
