# SPDX-License-Identifier: GPL-3.0-only
"""Canonical SVF fold over the packed bit state (TASKS T07, DESIGN 3.4/6.3).

Reproduces the ORIGINAL ``solweig_gpu.shadow.svf_calculator`` fold EXACTLY
(the oracle's veg-side outputs + SVFtotal), consuming the T07 packed
state (:mod:`solweig_core.bitplanes`) DIRECTLY — no unpack-to-cube, no
float cube materialisation:

* patch-major / annulus-minor recurrence in the original enumeration
  order: for patch p (ring-major, azimuth-minor), for annulus a:
  ``svfveg += w_iso * vegsh_p``; ``svfaveg += w_iso * vbsh_p``; then the
  E/S/W/N aniso pairs — same per-accumulator operation order as
  svf_calculator. No pre-summed annulus weights, no reordered patch
  reduction (both change float32 accumulation and are RED-designated).
* frozen constants (DESIGN 6.3): annulus weights and azimuths are the
  CAPTURED uint32 bit patterns of the original torch computation —
  never recomputed (torch-vs-numpy transcendental 1-ULP hazards). The
  tables are pinned against a live recompute of the original helpers by
  tests/ultrafast/test_svf_fold.py. ``last`` = 3.0459e-004 where
  ``vegdem2 == 0.0`` (-0.0 included, NaN excluded), ``trans`` = 0.03,
  ``SVFtotal = svf - (1 - svfveg) * (1 - trans)``.
* dtype discipline: every operand is float32 with explicit
  ``np.float32`` literals; accumulation is plain ``+`` in the original
  operand order; no fastmath, no reassociation, no FMA contraction, no
  float64 promotion, no blanket cast. Bit reads become exact 0.0/1.0
  float32 values before the first multiply.

Threading (T06 model, imported): rows are partitioned into an EXACT
contiguous chunk plan (:func:`plan_chunks`) re-verified before any
thread spawns (:func:`_verify_chunks`); workers are daemon
``threading.Thread``s calling ONE ``njit(cache=True, nogil=True,
fastmath=False)`` kernel — each thread's arithmetic IS serial
arithmetic, outputs are disjoint by row ownership, the packed inputs
are read-only. Numba auto-parallel compilation is never used (no
scheduler-owned execution semantics). Work below
:data:`SERIAL_MIN_CELLS` cells routes serial (zero thread spawn).

Affected-chunk-only output: ``cell_mask`` restricts which cells are
folded; with caller ``outputs`` base planes, every unmasked cell is
returned bit-identical to the base (nothing outside the affected set is
touched), and masked cells equal the full fold bit-for-bit.
"""
from __future__ import annotations

import hashlib
import threading
from typing import NamedTuple

import numpy as np
from numba import njit

from solweig_core import bitplanes as bp
from solweig_core.numba_cpu.sparse_march import _verify_chunks, plan_chunks

__all__ = [
    "FOLD_OUTPUT_NAMES",
    "FOLD_CONSTANTS_SHA256",
    "SERIAL_MIN_CELLS",
    "SvfFoldResult",
    "DIRECTION_E",
    "DIRECTION_N",
    "DIRECTION_S",
    "DIRECTION_W",
    "LAST_CONST",
    "N_ANNULUS",
    "N_PATCHES",
    "PATCH_AZIMUTH",
    "PATCH_RING",
    "TRANS",
    "W_ANISO",
    "W_ISO",
    "fold_svf",
    "frozen_tables_digest",
]

#: Sky patches for patch_option = 2 (frozen; pinned by tests).
N_PATCHES = 153
#: Sky rings.
N_RINGS = 8
#: Annulus count per ring (rings 0-6: 12, ring 7 (zenith): 6).
N_ANNULUS = np.array([12, 12, 12, 12, 12, 12, 12, 6], dtype=np.int32)

# ---------------------------------------------------------------------
# Frozen tables (DESIGN 6.3): CAPTURED uint32 bit patterns of the ORIGINAL
# torch computation — create_patches(patch_option=2) + annulus_weight + the
# iazimuth fill, run in solweig_gpu.shadow. Captured by
# artifacts/t07/capture_constants.py (read-only oracle capture); pinned
# against a LIVE recompute by tests/ultrafast/test_svf_fold.py. Weights are
# never recomputed here (torch-vs-numpy transcendental 1-ULP hazards); the
# padded slots past each ring's annulus count are exact +0.0 and unread.
# ---------------------------------------------------------------------
_W_ISO_U32 = np.array([
    0x3724d7c5, 0x37f729df, 0x384dcd3e, 0x388fe2c0, 0x38b8b201, 0x38e14793, 0x3904cb7b, 0x3918c9c1, 0x392c9868, 0x39403129, 0x39538e01, 0x3966a8d4,
    0x3980e65b, 0x398a77e9, 0x3993de4b, 0x399d168e, 0x39a61dcf, 0x39aef148, 0x39b78e28, 0x39bff1cd, 0x39c81994, 0x39d002f1, 0x39d7ab6e, 0x39df10a3,
    0x39f6a173, 0x39fdf66d, 0x3a027e1a, 0x3a05d849, 0x3a0908b9, 0x3a0c0e6d, 0x3a0ee870, 0x3a1195e3, 0x3a1415ec, 0x3a1667c5, 0x3a188ab5, 0x3a1a7e12,
    0x3a364c1e, 0x3a3821a2, 0x3a39bdb6, 0x3a3b1fdd, 0x3a3c47a7, 0x3a3d34b8, 0x3a3de6c4, 0x3a3e5d96, 0x3a3e9909, 0x3a3e9909, 0x3a3e5d96, 0x3a3de6c4,
    0x3a6eff38, 0x3a6dd3c5, 0x3a6c5e25, 0x3a6a9ecb, 0x3a689645, 0x3a664533, 0x3a63ac50, 0x3a60cc69, 0x3a5da666, 0x3a5a3b40, 0x3a568c0c, 0x3a5299ea,
    0x3a96d476, 0x3a93933e, 0x3a902401, 0x3a8c87cd, 0x3a88bfc7, 0x3a84cd17, 0x3a80b0fd, 0x3a78d97d, 0x3a700365, 0x3a66e270, 0x3a5d7978, 0x3a53cb6d,
    0x3abb703b, 0x3ab1fb62, 0x3aa84f08, 0x3a9e6e2d, 0x3a945bea, 0x3a8a1b60, 0x3a7f5f85, 0x3a6a38a6, 0x3a54c8b9, 0x3a3f166f, 0x3a29288a, 0x3a1305e3,
    0x3b5a3d69, 0x3b32ec74, 0x3b0b63b0, 0x3ac75ee4, 0x3a6f7076, 0x399fb0e8, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
], dtype=np.uint32).reshape(8, 12)
_W_ANISO_U32 = np.array([
    0x379fb107, 0x386f7091, 0x38c75ed5, 0x390b63ab, 0x3932ec72, 0x395a3d58, 0x3980a520, 0x39940374, 0x39a733a5, 0x39ba2fa1, 0x39ccf191, 0x39df738e,
    0x3a00e65b, 0x3a0a77e9, 0x3a13de4b, 0x3a1d168e, 0x3a261dcf, 0x3a2ef148, 0x3a378e28, 0x3a3ff1cd, 0x3a481994, 0x3a5002f1, 0x3a57ab6e, 0x3a5f10a3,
    0x3a76a173, 0x3a7df66d, 0x3a827e1a, 0x3a85d849, 0x3a8908b9, 0x3a8c0e6d, 0x3a8ee870, 0x3a9195e3, 0x3a9415ec, 0x3a9667c5, 0x3a988ab5, 0x3a9a7e12,
    0x3ab64c1e, 0x3ab821a2, 0x3ab9bdb6, 0x3abb1fdd, 0x3abc47a7, 0x3abd34b8, 0x3abde6c4, 0x3abe5d96, 0x3abe9909, 0x3abe9909, 0x3abe5d96, 0x3abde6c4,
    0x3ae30c0f, 0x3ae1ef96, 0x3ae08ca3, 0x3adee3a8, 0x3adcf528, 0x3adac1be, 0x3ad84a1a, 0x3ad58efe, 0x3ad29147, 0x3acf51e4, 0x3acbd1d8, 0x3ac81239,
    0x3b0c0e6d, 0x3b0908b9, 0x3b05d84a, 0x3b027e1a, 0x3afdf671, 0x3af6a173, 0x3aeeff8c, 0x3ae71319, 0x3adede95, 0x3ad6648c, 0x3acda7a6, 0x3ac4aa9b,
    0x3b240234, 0x3b1bbbf6, 0x3b134527, 0x3b0aa067, 0x3b01d06d, 0x3af1afe8, 0x3adf7395, 0x3accf191, 0x3aba2fa2, 0x3aa733a1, 0x3a940379, 0x3a80a527,
    0x3b5a3d69, 0x3b32ec74, 0x3b0b63b0, 0x3ac75ee4, 0x3a6f7076, 0x399fb0e8, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
], dtype=np.uint32).reshape(8, 12)
_AZIMUTH_U32 = np.array([
    0x00000000, 0x4139ce73, 0x41b9ce73, 0x420b5ad6, 0x4239ce73, 0x42684210, 0x428b5ad6, 0x42a294a5,
    0x42b9ce73, 0x42d10841, 0x42e84210, 0x42ff7bde, 0x430b5ad6, 0x4316f7bd, 0x432294a5, 0x432e318c,
    0x4339ce73, 0x43456b5a, 0x43510841, 0x435ca529, 0x43684210, 0x4373def7, 0x437f7bde, 0x43858c63,
    0x438b5ad6, 0x4391294a, 0x4396f7bd, 0x439cc631, 0x43a294a5, 0x43a86318, 0x43ae318c, 0x40800000,
    0x41800000, 0x41e00001, 0x42200001, 0x42500001, 0x42800000, 0x42980001, 0x42b00001, 0x42c80001,
    0x42e00001, 0x42f80001, 0x43080001, 0x43140001, 0x43200001, 0x432c0001, 0x43380001, 0x43440001,
    0x43500001, 0x435c0001, 0x43680001, 0x43740001, 0x43800000, 0x43860001, 0x438c0001, 0x43920001,
    0x43980001, 0x439e0001, 0x43a40001, 0x43aa0001, 0x40000000, 0x416db6dc, 0x41ddb6dc, 0x42224925,
    0x4255b6dc, 0x4284924a, 0x429e4925, 0x42b80000, 0x42d1b6dc, 0x42eb6db8, 0x4302924a, 0x430f6db7,
    0x431c4925, 0x43292493, 0x43360000, 0x4342db6e, 0x434fb6dc, 0x435c924a, 0x43696db8, 0x43764925,
    0x4381924a, 0x43880000, 0x438e6db7, 0x4394db6e, 0x439b4925, 0x43a1b6dc, 0x43a82493, 0x40a00000,
    0x41a00000, 0x420c0000, 0x42480000, 0x42820000, 0x42a00000, 0x42be0000, 0x42dc0000, 0x42fa0000,
    0x430c0000, 0x431b0000, 0x432a0000, 0x43390000, 0x43480000, 0x43570000, 0x43660000, 0x43750000,
    0x43820000, 0x43898000, 0x43910000, 0x43988000, 0x43a00000, 0x43a78000, 0x43af0000, 0x41000000,
    0x41d79436, 0x42379436, 0x4281af28, 0x42a79436, 0x42cd7944, 0x42f35e51, 0x430ca1af, 0x431f9436,
    0x433286bd, 0x43457944, 0x43586bca, 0x436b5e51, 0x437e50d8, 0x4388a1af, 0x43921af3, 0x439b9436,
    0x43a50d79, 0x43ae86bd, 0x00000000, 0x41dd89d9, 0x425d89d9, 0x42a62763, 0x42dd89d9, 0x430a7628,
    0x43262763, 0x4341d89e, 0x435d89d9, 0x43793b14, 0x438a7628, 0x43984ec5, 0x41200000, 0x4275b6dc,
    0x42e1b6dc, 0x43244925, 0x4357b6dc, 0x4385924a, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000,
], dtype=np.uint32)
PATCH_RING = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 6, 6, 6, 6, 6, 6, 6, 7,
], dtype=np.int32)
_LAST_U32 = np.uint32(0x399fb161)  # 3.0459e-004
_TRANS_U32 = np.uint32(0x3cf5c28f)  # 0.03

# float32 views of the frozen bit patterns (bit-identical by construction)
W_ISO = _W_ISO_U32.view(np.float32)
W_ANISO = _W_ANISO_U32.view(np.float32)
PATCH_AZIMUTH = _AZIMUTH_U32.view(np.float32)
LAST_CONST = np.array([_LAST_U32], dtype=np.uint32).view(np.float32)[0]
TRANS = np.array([_TRANS_U32], dtype=np.uint32).view(np.float32)[0]
FOLD_CONSTANTS_SHA256 = (
    "0a13e80a61b1cae69344c2517f45df6db6675bc2578b91cb909d2673c57519a8"
)

#: Directional membership from the FROZEN azimuth bits — comparisons only
#: (T01-clean ops; az is f32, the bounds are explicit f32 literals).
#: E: 0 <= az < 180, S: 90 <= az < 270, W: 180 <= az < 360,
#: N: az >= 270 or az < 90  (each patch belongs to exactly two).
DIRECTION_E = (PATCH_AZIMUTH >= np.float32(0.0)) & (PATCH_AZIMUTH < np.float32(180.0))
DIRECTION_S = (PATCH_AZIMUTH >= np.float32(90.0)) & (PATCH_AZIMUTH < np.float32(270.0))
DIRECTION_W = (PATCH_AZIMUTH >= np.float32(180.0)) & (PATCH_AZIMUTH < np.float32(360.0))
DIRECTION_N = (PATCH_AZIMUTH >= np.float32(270.0)) | (PATCH_AZIMUTH < np.float32(90.0))

#: Output order of :class:`SvfFoldResult` (matches svf_calculator names).
FOLD_OUTPUT_NAMES = (
    "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
    "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
    "svftotal",
)

#: Route serial (no thread spawn) below this many folded cells. A fold
#: cell costs ~153 patches x ~12 annuli of float32 work, so the floor is
#: far below T06's sparse-pair crossover on purpose.
SERIAL_MIN_CELLS = 4096


def frozen_tables_digest() -> str:
    """sha256 over the canonical serialization of the frozen tables.

    Content digest of: patch count, per-ring annulus counts, all
    annulus-weight and azimuth bit patterns, ring assignment, and the
    ``last``/``trans`` constants. Any table mutation changes it.
    """
    parts = [
        str(N_PATCHES),
        ",".join(str(int(x)) for x in N_ANNULUS),
        ",".join(f"{int(b):08x}" for b in _W_ISO_U32.reshape(-1)),
        ",".join(f"{int(b):08x}" for b in _W_ANISO_U32.reshape(-1)),
        ",".join(f"{int(b):08x}" for b in _AZIMUTH_U32.reshape(-1)),
        ",".join(str(int(x)) for x in PATCH_RING),
        f"{int(_LAST_U32):08x}",
        f"{int(_TRANS_U32):08x}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Kernel: the original recurrence, per cell, straight off the packed bytes
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True, fastmath=False)
def _fold_kernel(
    veg_bytes, vbsh_bytes, vegdem2, svf_building, mask, do_mask,
    w_iso, w_aniso, ring, na, dir_e, dir_s, dir_w, dir_n,
    last_const, one_minus_trans, row_lo, row_hi, out,
):
    """Fold rows [row_lo, row_hi) into ``out`` (rows, cols, 11).

    Per cell: patch-major/annulus-minor accumulation in the original
    order, the ``last`` correction, the 10 veg clamps, then SVFtotal.
    Every float is float32; bit reads become exact 0.0/1.0 first.
    """
    cols = vegdem2.shape[1]
    zero = np.float32(0.0)
    one = np.float32(1.0)
    for i in range(row_lo, row_hi):
        for j in range(cols):
            if do_mask:
                if not mask[i, j]:
                    continue
            svfveg = zero
            svfaveg = zero
            svfEveg = zero
            svfEaveg = zero
            svfSveg = zero
            svfSaveg = zero
            svfWveg = zero
            svfWaveg = zero
            svfNveg = zero
            svfNaveg = zero
            for p in range(153):
                r = ring[p]
                veg_byte = veg_bytes[i, j, p >> 3]
                v = np.float32((veg_byte >> (p & 7)) & 1)
                vbsh_byte = vbsh_bytes[i, j, p >> 3]
                b = np.float32((vbsh_byte >> (p & 7)) & 1)
                n_ann = na[r]
                de = dir_e[p]
                ds = dir_s[p]
                dw = dir_w[p]
                dn = dir_n[p]
                for a in range(n_ann):
                    w = w_iso[r, a]
                    wa = w_aniso[r, a]
                    svfveg = svfveg + w * v
                    svfaveg = svfaveg + w * b
                    if de:
                        svfEveg = svfEveg + wa * v
                        svfEaveg = svfEaveg + wa * b
                    if ds:
                        svfSveg = svfSveg + wa * v
                        svfSaveg = svfSaveg + wa * b
                    if dw:
                        svfWveg = svfWveg + wa * v
                        svfWaveg = svfWaveg + wa * b
                    if dn:
                        svfNveg = svfNveg + wa * v
                        svfNaveg = svfNaveg + wa * b
            # last correction: 3.0459e-004 where vegdem2 == 0.0
            # (-0.0 == 0.0 is True; NaN == 0.0 is False)
            if vegdem2[i, j] == zero:
                last_v = last_const
            else:
                last_v = zero
            svfSveg = svfSveg + last_v
            svfWveg = svfWveg + last_v
            svfSaveg = svfSaveg + last_v
            svfWaveg = svfWaveg + last_v
            # the 10 veg clamps (original order, [x > 1.] = 1.)
            if svfveg > one:
                svfveg = one
            if svfaveg > one:
                svfaveg = one
            if svfEveg > one:
                svfEveg = one
            if svfEaveg > one:
                svfEaveg = one
            if svfSveg > one:
                svfSveg = one
            if svfSaveg > one:
                svfSaveg = one
            if svfWveg > one:
                svfWveg = one
            if svfWaveg > one:
                svfWaveg = one
            if svfNveg > one:
                svfNveg = one
            if svfNaveg > one:
                svfNaveg = one
            svftotal = svf_building[i, j] - (one - svfveg) * one_minus_trans
            out[i, j, 0] = svfveg
            out[i, j, 1] = svfEveg
            out[i, j, 2] = svfSveg
            out[i, j, 3] = svfWveg
            out[i, j, 4] = svfNveg
            out[i, j, 5] = svfaveg
            out[i, j, 6] = svfEaveg
            out[i, j, 7] = svfSaveg
            out[i, j, 8] = svfWaveg
            out[i, j, 9] = svfNaveg
            out[i, j, 10] = svftotal


# ---------------------------------------------------------------------------
# Wrapper: validation, routing, exact-partition threading
# ---------------------------------------------------------------------------


class SvfFoldResult(NamedTuple):
    """The 11 fold outputs (float32 planes, svf_calculator names) + stats.

    ``svftotal`` is the SVFtotal of the original; everything else is the
    vegetated-side accumulator family. ``stats`` records the routing
    decision: cells folded, threads used, and the exact row chunk plan.
    """

    svfveg: np.ndarray
    svfEveg: np.ndarray
    svfSveg: np.ndarray
    svfWveg: np.ndarray
    svfNveg: np.ndarray
    svfaveg: np.ndarray
    svfEaveg: np.ndarray
    svfSaveg: np.ndarray
    svfWaveg: np.ndarray
    svfNaveg: np.ndarray
    svftotal: np.ndarray
    stats: dict


def _as_packed(x, name: str) -> bp.PackedBits:
    if not isinstance(x, bp.PackedBits):
        raise TypeError(
            f"{name} must be a bitplanes.PackedBits (pack through "
            f"bitplanes.pack_bits — the fold consumes packed input only), "
            f"got {type(x).__name__}"
        )
    if x.patch_count != N_PATCHES:
        raise ValueError(
            f"{name}.patch_count must be {N_PATCHES} (patch_option=2), "
            f"got {x.patch_count}"
        )
    return x


def _as_plane(x, name: str, shape) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2 or arr.dtype != np.dtype(np.float32):
        raise ValueError(
            f"{name} must be a 2-D float32 plane, got {arr.dtype.str} "
            f"{arr.shape}"
        )
    if arr.shape != tuple(shape):
        raise ValueError(
            f"{name} shape {arr.shape} does not match the packed state "
            f"{tuple(shape)}"
        )
    return np.ascontiguousarray(arr)


def fold_svf(
    vegsh_packed,
    vbsh_packed,
    vegdem2,
    svf_building,
    *,
    cell_mask=None,
    outputs=None,
    n_threads: int = 1,
    min_cells: int = SERIAL_MIN_CELLS,
) -> SvfFoldResult:
    """Fold the packed vegsh/vbsh state into the 11 svf_calculator outputs.

    ``vegdem2`` is the MarchInputs.vegdsm2 plane (the ``last`` predicate
    input); ``svf_building`` is the building-side svf plane (oracle
    output — the fold never recomputes the building march).

    ``cell_mask`` (bool plane, optional) folds ONLY masked cells; with
    ``outputs`` (a mapping of all 11 output names to base planes) every
    unmasked cell is returned bit-identical to its base. ``n_threads``
    partitions ROWS into an exact chunk plan (T06 model) — each thread
    owns disjoint rows, so no output cell is ever written twice and the
    packed inputs are read-only throughout.
    """
    veg = _as_packed(vegsh_packed, "vegsh_packed")
    vbsh = _as_packed(vbsh_packed, "vbsh_packed")
    rows, cols = veg.rows, veg.cols
    if (vbsh.rows, vbsh.cols) != (rows, cols):
        raise ValueError(
            f"packed shapes disagree: vegsh {veg.data.shape} vs vbsh "
            f"{vbsh.data.shape}"
        )
    vegdem2_c = _as_plane(vegdem2, "vegdem2", (rows, cols))
    svf_c = _as_plane(svf_building, "svf_building", (rows, cols))

    if cell_mask is None:
        mask = np.zeros((rows, cols), dtype=np.bool_)
        do_mask = False
        n_cells = rows * cols
    else:
        mask_arr = np.asarray(cell_mask)
        if mask_arr.ndim != 2 or mask_arr.dtype != np.dtype(np.bool_):
            raise ValueError(
                f"cell_mask must be a 2-D bool plane, got "
                f"{mask_arr.dtype.str} {mask_arr.shape}"
            )
        if mask_arr.shape != (rows, cols):
            raise ValueError(
                f"cell_mask shape {mask_arr.shape} does not match "
                f"{(rows, cols)}"
            )
        mask = np.ascontiguousarray(mask_arr)
        do_mask = True
        n_cells = int(mask.sum())

    if outputs is None:
        out = np.zeros((rows, cols, len(FOLD_OUTPUT_NAMES)), dtype=np.float32)
    else:
        missing = [name for name in FOLD_OUTPUT_NAMES if name not in outputs]
        if missing:
            raise ValueError(
                f"outputs must provide every fold plane, missing {missing}"
            )
        out = np.empty((rows, cols, len(FOLD_OUTPUT_NAMES)), dtype=np.float32)
        for k, name in enumerate(FOLD_OUTPUT_NAMES):
            out[:, :, k] = _as_plane(outputs[name], f"outputs[{name!r}]", (rows, cols))

    # exact row partition (T06 model): balanced by per-row folded-cell
    # count, re-verified before any thread spawns
    row_work = mask.sum(axis=1).astype(np.int64) if do_mask else np.full(
        rows, cols, dtype=np.int64
    )
    n_threads = int(n_threads)
    if n_threads <= 1 or rows < 2 or n_cells < int(min_cells):
        used = 1
        chunks = ((0, rows),)
    else:
        chunks = plan_chunks(row_work, n_threads)
        used = len(chunks)
    _verify_chunks(chunks, rows)

    one_minus_trans = np.float32(1.0) - TRANS

    def call(lo: int, hi: int) -> None:
        _fold_kernel(
            veg.data, vbsh.data, vegdem2_c, svf_c, mask, do_mask,
            W_ISO, W_ANISO, PATCH_RING, N_ANNULUS,
            DIRECTION_E.view(np.int8), DIRECTION_S.view(np.int8),
            DIRECTION_W.view(np.int8), DIRECTION_N.view(np.int8),
            LAST_CONST, one_minus_trans, lo, hi, out,
        )

    if used == 1:
        call(0, rows)
    else:
        call(0, 0)  # force compilation before threads hit the dispatch lock
        threads = [
            threading.Thread(target=call, args=(lo, hi), daemon=True)
            for lo, hi in chunks[1:]
        ]
        for t in threads:
            t.start()
        call(*chunks[0])  # the calling thread owns the first chunk
        for t in threads:
            t.join()

    planes = tuple(
        np.ascontiguousarray(out[:, :, k]) for k in range(len(FOLD_OUTPUT_NAMES))
    )
    stats = {
        "n_cells": int(rows * cols),
        "masked_cells": int(n_cells),
        "used_threads": int(used),
        "chunks": tuple((int(lo), int(hi)) for lo, hi in chunks),
        "route": "masked" if do_mask else "full",
    }
    return SvfFoldResult(*planes, stats)
