# SPDX-License-Identifier: GPL-3.0-only
"""UTCI compatibility kernel — torch-free bit-exact port of
``solweig_gpu/calculate_utci.py::utci_calculator`` (ultrafast T09).

Reproduces the ORIGINAL AST association and pow semantics of the oracle
verbatim — no Horner rewrites, no new FMA contraction, no float64
promotion, no blanket float32 cast, no pow->multiply where torch itself
does not, no NaN added or removed in masks. Each of those is a RED
witness mutation in tests/ultrafast/test_utci.py (M1-M7).

Two variants over the same kernel:

- dense  (``utci_calculator_dense``): consumes full (rows, cols) f32
  planes exactly where the oracle does; compacts the
  ``(~(Ta<=-999 | RH<=-999 | va<=-999 | Tmrt<=-999))`` lanes in
  row-major masked-select order, computes, scatters into a -999 plane.
- sparse (``utci_calculator_sparse``): consumes pre-compacted 1-D f32
  vectors (the same masked-select order the caller captured them in);
  bit-identical per valid element, including odd-length tails (the tail
  lanes are exactly where torch's opmath scalar pow remainder lives).

Bit-exactness against torch depends on the elementwise chunk layout of
the compacted extent n (see solweig_core.numba_cpu.math_compat): the
oracle's UTCI value depends on n, not just on the per-element inputs.
``torch_threads`` (default 8 — t08 capture manifest) and ``grain``
(32768) are parameters.

The saturation-pressure chain mirrors the oracle's scalar broadcast
contexts exactly: RH reaches ``es * RH / 100.`` as an f32 plane value
(torch f32-tensor broadcast, T03), the python-float literals 273.15 /
0.01 / 100. / 10.0 are single-rounded to f32 at their use site, and the
g coefficients are f32 tensor elements.

EXPRESSION_MANIFEST (below) is the machine-readable per-expression
record: primitive, dtype, context, and resolution decision for every
UTCI expression; hybrid status reads out of it (all clean, no fallbacks).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numba import njit, prange

from solweig_core.numba_cpu.math_compat import (
    DEFAULT_TORCH_THREADS,
    TORCH_PAR_GRAIN,
    _is_libm_lane,
    _torch_pow_scalar,
    nadd,
    ndiv,
    nmul,
    nsub,
    sleef_expf,
    sleef_logf_u1,
)

F32 = np.float32

# saturation-pressure coefficients: torch.tensor([...], dtype=float32)
# (calculate_utci.py utci_calculator, verbatim order)
_G = np.array([
    -2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2,
    1.6261698e-5, 7.0229056e-10, -1.8680009e-13, 2.7150305,
], dtype=np.float32)
G0, G1, G2, G3, G4, G5, G6, G7 = (_G[i] for i in range(8))

NEG999 = F32(-999.0)
TK_OFF = F32(273.15)      # Ta[valid] + 273.15 (f32-wrapped python float)
ES_SCALE = F32(0.01)      # torch.exp(es) * 0.01
RH_DIV = F32(100.0)       # es * RH / 100.
PA_DIV = F32(10.0)        # ehPa / 10.0


# --------------------------------------------------------------------------
# element kernel: one valid lane, k = its masked-select ordinal
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def _utci_element(ta, rh, tmrt, va, k, n, T, grain):
    # saturation vapour pressure chain (utci_calculator verbatim). Every
    # fp add/sub/mul/div goes through the first-NaN-operand wrappers
    # (math_compat) — torch's elementwise ops deterministically propagate
    # the FIRST array operand's NaN payload; a scalar chain in numba would
    # let LLVM's scheduler commute commutative sites per add and the
    # payload winner would be undefined. Finite results are bit-identical
    # to the plain ops.
    tk = nadd(ta, TK_OFF)
    use_libm = _is_libm_lane(k, n, T, grain)

    es = nmul(G7, sleef_logf_u1(tk))
    # for i in range(0, 7): es = es + g[i] * tk ** (i + 1 - 3.)
    es = nadd(es, nmul(G0, ndiv(F32(1.0), nmul(tk, tk))))    # tk ** -2.
    es = nadd(es, nmul(G1, ndiv(F32(1.0), tk)))              # tk ** -1.
    es = nadd(es, nmul(G2, F32(1.0)))                        # tk ** 0.
    es = nadd(es, nmul(G3, tk))                              # tk ** 1.
    es = nadd(es, nmul(G4, nmul(tk, tk)))                    # tk ** 2.
    es = nadd(es, nmul(G5, nmul(nmul(tk, tk), tk)))          # tk ** 3.
    es = nadd(es, nmul(G6, _torch_pow_scalar(tk, F32(4.0), use_libm)))
    es = nmul(sleef_expf(es), ES_SCALE)

    ehpa = ndiv(nmul(es, rh), RH_DIV)
    dtm = nsub(tmrt, ta)
    pa = ndiv(ehpa, PA_DIV)
    return _utci_poly_element(dtm, ta, va, pa, use_libm)


# --------------------------------------------------------------------------
# dense / sparse kernels
#
# prange parallelism is BIT-SAFE here by construction: every lane's value
# depends only on its own inputs and its lane ordinal (the layout model
# reads (k, n, T, grain), never the schedule); the dense count pass is an
# integer reduction and the fill/scatter passes write disjoint lanes.
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, parallel=True, error_model="numpy")
def _utci_dense_kernel(ta, rh, tmrt, va, out, T, grain):
    rows, cols = ta.shape
    n = 0
    offs = np.empty(rows + 1, dtype=np.int64)
    offs[0] = 0
    for r in range(rows):
        for c in range(cols):
            if not (ta[r, c] <= NEG999 or rh[r, c] <= NEG999
                    or va[r, c] <= NEG999 or tmrt[r, c] <= NEG999):
                n += 1
        offs[r + 1] = n
    for r in prange(rows):
        k = offs[r]
        for c in range(cols):
            if not (ta[r, c] <= NEG999 or rh[r, c] <= NEG999
                    or va[r, c] <= NEG999 or tmrt[r, c] <= NEG999):
                out[r, c] = _utci_element(ta[r, c], rh[r, c], tmrt[r, c],
                                          va[r, c], k, n, T, grain)
                k += 1


@njit(cache=True, fastmath=False, parallel=True, error_model="numpy")
def _utci_sparse_kernel(ta, rh, tmrt, va, out, T, grain):
    n = ta.size
    for k in prange(n):
        out[k] = _utci_element(ta[k], rh[k], tmrt[k], va[k], k, n, T, grain)


def _check_plane(x, name):
    a = np.asarray(x)
    if a.dtype != np.float32 or a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D float32 array, got {a.dtype} ndim={a.ndim}")
    return np.ascontiguousarray(a)


def _check_vec(x, name):
    a = np.asarray(x)
    if a.dtype != np.float32 or a.ndim != 1:
        raise ValueError(f"{name} must be a 1-D float32 array, got {a.dtype} ndim={a.ndim}")
    return np.ascontiguousarray(a)


def utci_calculator_dense(ta_plane, rh_plane, tmrt_plane, va_plane,
                          torch_threads=DEFAULT_TORCH_THREADS,
                          grain=TORCH_PAR_GRAIN):
    """Bit-exact port of ``utci_calculator`` on full planes.

    Returns the -999-filled (rows, cols) f32 plane (the oracle's
    ``UTCI_approx``); NaN lanes in the inputs stay valid lanes and
    propagate NaN — the oracle's ``<= -999`` masks never test NaN.
    """
    ta = _check_plane(ta_plane, "ta_plane")
    rh = _check_plane(rh_plane, "rh_plane")
    tmrt = _check_plane(tmrt_plane, "tmrt_plane")
    va = _check_plane(va_plane, "va_plane")
    if not (ta.shape == rh.shape == tmrt.shape == va.shape):
        raise ValueError(f"plane shape mismatch: {ta.shape} {rh.shape} {tmrt.shape} {va.shape}")
    out = np.full(ta.shape, F32(-999.0), dtype=np.float32)
    _utci_dense_kernel(ta, rh, tmrt, va, out, torch_threads, grain)
    return out


def utci_calculator_sparse(ta_v, rh_v, tmrt_v, va_v,
                           torch_threads=DEFAULT_TORCH_THREADS,
                           grain=TORCH_PAR_GRAIN):
    """Sparse/compacted variant: same kernel over the pre-compacted valid
    vectors (masked-select order). Bit-identical per valid element to the
    dense result on the same valid set, including odd-length tails."""
    ta = _check_vec(ta_v, "ta_v")
    rh = _check_vec(rh_v, "rh_v")
    tmrt = _check_vec(tmrt_v, "tmrt_v")
    va = _check_vec(va_v, "va_v")
    if not (ta.size == rh.size == tmrt.size == va.size):
        raise ValueError(f"vector length mismatch: {ta.size} {rh.size} {tmrt.size} {va.size}")
    out = np.empty(ta.size, dtype=np.float32)
    _utci_sparse_kernel(ta, rh, tmrt, va, out, torch_threads, grain)
    return out


def compact_valid(ta_plane, rh_plane, tmrt_plane, va_plane):
    """The oracle's boolean compaction in row-major masked-select order:
    ``~(Ta<=-999 | RH<=-999 | va10m<=-999 | Tmrt<=-999)``."""
    ta = _check_plane(ta_plane, "ta_plane")
    rh = _check_plane(rh_plane, "rh_plane")
    tmrt = _check_plane(tmrt_plane, "tmrt_plane")
    va = _check_plane(va_plane, "va_plane")
    valid = ~((ta <= NEG999) | (rh <= NEG999) | (va <= NEG999) | (tmrt <= NEG999))
    return (np.ascontiguousarray(ta[valid]), np.ascontiguousarray(rh[valid]),
            np.ascontiguousarray(tmrt[valid]), np.ascontiguousarray(va[valid]))


# --------------------------------------------------------------------------
# per-timestep met-step helper (mirrors utci_process/recompute_utci_steps)
# --------------------------------------------------------------------------
def met_step_planes(shape, ta_i, rh_i, ws_i, uhii_i=0.0):
    """Replicate the oracle's per-timestep plane construction bit-exactly:
    ``RH_mat = zeros + RH[i]``; ``Ta_mat = zeros + Ta[i] + uhii[i]``
    (left-assoc: f32(Ta[i]) plane, then + f32(uhii[i]));
    ``va10m = clamp(ones * Ws[i], min=0.15)`` (windcoeff == ones — the
    incremental/solver paths pass no wind-coefficient rasters).
    Returns (ta_plane, rh_plane, va_plane) f32."""
    ta = np.full(shape, F32(np.float32(ta_i) + F32(uhii_i)), dtype=np.float32)
    rh = np.full(shape, np.float32(rh_i), dtype=np.float32)
    va = np.full(shape, np.maximum(F32(0.15), np.float32(ws_i)), dtype=np.float32)
    return ta, rh, va


def buildings_from_dsm(building_dsm, dem):
    """utci_process scene prep: ``a - temp2``; <2 -> 1; >=2 -> 0."""
    a = np.asarray(building_dsm, dtype=np.float32)
    b = a - np.asarray(dem, dtype=np.float32)
    out = b.copy()
    out[b < 2.0] = 1.0
    out[b >= 2.0] = 0.0
    return out


def utci_met_step(tmrt_plane, ta_i, rh_i, ws_i, uhii_i=0.0,
                  torch_threads=DEFAULT_TORCH_THREADS,
                  grain=TORCH_PAR_GRAIN):
    """Per-timestep convenience: met scalars + Tmrt plane -> utci_calculator
    plane (the -999 variant; apply_buildings_mask produces the published
    NaN plane)."""
    tmrt = _check_plane(tmrt_plane, "tmrt_plane")
    ta_p, rh_p, va_p = met_step_planes(tmrt.shape, ta_i, rh_i, ws_i, uhii_i)
    return utci_calculator_dense(ta_p, rh_p, tmrt, va_p,
                                 torch_threads, grain)


def apply_buildings_mask(utci_mat, buildings):
    """utci_process:888-890 — ``UTCI = full(nan); UTCI[valid_mask] =
    UTCI_mat[valid_mask]`` with ``valid_mask = buildings == 1``."""
    out = np.full(utci_mat.shape, np.nan, dtype=np.float32)
    m = np.asarray(buildings) == 1
    out[m] = np.asarray(utci_mat, np.float32)[m]
    return out


# --------------------------------------------------------------------------
# typed expression manifest — machine-readable, read by tests/report
# --------------------------------------------------------------------------
EXPRESSION_MANIFEST = {
    "schema": "utci-expression-manifest/1",
    "oracle": "solweig_gpu/calculate_utci.py@ea3bd118eb15f9bc9320bc7becdc2a4a0f78a6f6",
    "expressions": {
        "utci.invalid_mask": {
            "source": "(Ta <= -999) | (RH <= -999) | (va10m <= -999) | (Tmrt <= -999)",
            "primitive": "cmp+or f32 planes (python-int -999 wrapped f32)",
            "context": "NaN lanes: NaN <= -999 is False -> NaN stays a VALID lane (never added/removed)",
            "resolution": "clean",
            "dtype": "bool mask over f32 planes",
        },
        "utci.tk": {
            "source": "Ta[valid_mask] + 273.15",
            "primitive": "add.f32 with f32-wrapped python float",
            "context": "f32 tensor + python float -> scalar single-rounded to f32 (T08 frozen context)",
            "resolution": "clean",
        },
        "utci.es.log": {
            "source": "g[7] * torch.log(tk)",
            "primitive": "torch.log -> SLEEF xlogf_u1 numba port",
            "context": "all lane classes (scalar remainders are SLEEF too); g[7] 0-dim f32 tensor element",
            "resolution": "clean",
            "evidence": "P5c torch==bundled-u10-symbol 0/36004; P5d port 0/38010 lattice, 0/21003 UTCI domain",
        },
        "utci.es.pow_m2": {
            "source": "tk ** -2.",
            "primitive": "pow scalar-exp exact form 1/(x*x)",
            "resolution": "clean", "evidence": "P5f e2/em2 exact forms 0 mismatch",
        },
        "utci.es.pow_m1": {
            "source": "tk ** -1.",
            "primitive": "pow scalar-exp exact form 1/x",
            "resolution": "clean", "evidence": "P5f",
        },
        "utci.es.pow_0": {
            "source": "tk ** 0.",
            "primitive": "pow scalar-exp exact form 1.0",
            "resolution": "clean", "evidence": "P5f (incl NaN, +-0 bases -> 1.0)",
        },
        "utci.es.pow_1": {
            "source": "tk ** 1.",
            "primitive": "pow scalar-exp exact form x",
            "resolution": "clean", "evidence": "P5f",
        },
        "utci.es.pow_2": {
            "source": "tk ** 2.",
            "primitive": "pow scalar-exp exact form x*x",
            "resolution": "clean", "evidence": "P5f",
        },
        "utci.es.pow_3": {
            "source": "tk ** 3.",
            "primitive": "pow scalar-exp exact form (x*x)*x",
            "resolution": "clean", "evidence": "P5f",
        },
        "utci.es.pow_4": {
            "source": "tk ** 4.",
            "primitive": "pow scalar-exp layout-routed: SLEEF xpowf body / opmath-f64 pow chunk-tail (P9)",
            "context": "extent n = valid-lane count; threads_eff = min(T, ceil(n/32768)); chunk = ceil(n/threads_eff); tail = per-chunk len%8 lanes",
            "resolution": "clean",
            "evidence": "P5e port 0-mismatch all tensor-exponent lattices; layout 15/15 sizes; P9 scalar-tail opmath f64 pow",
        },
        "utci.es.chain_add": {
            "source": "es = g[7]*log(tk); for i in 0..6: es = es + g[i] * tk**(i+1-3.)",
            "primitive": "left-assoc f32 add chain, term = g[i] * pow-result (mult after pow by precedence)",
            "resolution": "clean",
        },
        "utci.es.exp_scale": {
            "source": "es = torch.exp(es) * 0.01",
            "primitive": "SLEEF xexpf port; f32-wrapped 0.01",
            "resolution": "clean", "evidence": "P5/P5c 0 mismatch",
        },
        "utci.ehPa": {
            "source": "es * RH[valid_mask] / 100.",
            "primitive": "mul then div, f32",
            "context": "RH reaches here as an f32 plane VALUE (T03 python-float vs f32-tensor split: the oracle broadcast RH[i] into an f32 zeros plane; the multiplication sees f32(RH[i]), never the f64 python float)",
            "resolution": "clean",
        },
        "utci.D_Tmrt": {
            "source": "Tmrt[valid_mask] - Ta[valid_mask]",
            "primitive": "sub.f32", "resolution": "clean",
        },
        "utci.Pa": {
            "source": "ehPa / 10.0",
            "primitive": "div.f32 with f32-wrapped 10.0",
            "resolution": "clean",
        },
        "utci.poly.sum": {
            "source": "utci_polynomial 210-term left-assoc add chain starting Ta + (6.07562052E-01) + ...",
            "primitive": "left-assoc f32 adds; terms left-assoc mults ((coef * pow_form) * var ...)",
            "resolution": "clean",
            "note": ("kernel source generated from the oracle AST (t09 probes/gen_poly_kernel.py), "
                      "committed as the POLY_EXPRESSION_SOURCE statement block below; every fp op in "
                      "the chain goes through the first-NaN-operand wrappers (nadd/nmul — torch "
                      "elementwise NaN-payload semantics vs LLVM commutation); structural equality "
                      "re-asserted by tests/ultrafast/test_utci.py"),
        },
        "utci.poly.coefficients": {
            "source": "python float literals (e.g. 6.07562052E-01)",
            "primitive": "f64 literal single-rounded to f32 at use site",
            "resolution": "clean",
        },
        "utci.poly.pow_2_3": {
            "source": "Ta**2, va**3, D_Tmrt**2, Pa**3, ... (python-int exponents 2,3)",
            "primitive": "pow scalar-exp exact forms x*x, (x*x)*x (int==float Scalar dispatch, P5f)",
            "resolution": "clean",
        },
        "utci.poly.pow_4_5_6": {
            "source": "Ta**4..6, va**4..6, D_Tmrt**4..6, Pa**4..6 (python-int exponents)",
            "primitive": "pow scalar-exp layout-routed: SLEEF xpowf body / opmath-f64 pow chunk-tail (P9)",
            "resolution": "clean",
            "evidence": "P5f int==float dispatch; P5e body/tail; layout 15/15",
        },
        "utci.scatter": {
            "source": "UTCI_approx = full_like(Ta, -999); UTCI_approx[valid_mask] = poly(...)",
            "primitive": "row-major masked-select order scatter",
            "resolution": "clean",
        },
        "utci.outer_mask": {
            "source": "UTCI = full(nan); UTCI[valid_mask_read] = UTCI_mat[valid_mask_read] (buildings == 1)",
            "primitive": "caller-side NaN fill + scatter (utci_process; apply_buildings_mask mirrors it)",
            "resolution": "clean",
        },
    },
    "layout_model": {
        "threads_eff": "min(T, ceil(n/grain))",
        "chunk_lanes": "ceil(n/threads_eff) contiguous",
        "scalar_tail": ("per-chunk trailing len%8 lanes use torch's opmath scalar "
                         "pow (f64 pow single-rounded to f32 — P9); SLEEF for exp/log"),
        "small_n": "n <= grain -> single serial chunk (same tail rule)",
        "params": {
            "torch_threads": DEFAULT_TORCH_THREADS,
            "grain": TORCH_PAR_GRAIN,
            "provenance": "t08/capture/manifest.json torch_threads=8; grain 32768 empirical torch 2.14.0 CPU",
        },
        "exposure": "kernel arguments torch_threads/grain — bit-exactness is extent- and thread-count-dependent by construction",
    },
    "left_for_T10": {
        "R5_warm_start": (
            "thermal adapter will need the SAME layout discipline for any "
            "pow>=4 it introduces (Tstart/TgK chains — check T08 frozen "
            "contexts); EXPRESSION_MANIFEST schema is the template; "
            "math_compat primitives are shared"
        ),
    },
}


def get_expression_manifest():
    return dict(EXPRESSION_MANIFEST)


# --------------------------------------------------------------------------
# polynomial kernel — built from the oracle AST; SINGLE SOURCE OF TRUTH is
# POLY_EXPRESSION_SOURCE (test_utci.py re-parses the oracle and asserts
# structural equality). The kernel definition is the byte-exact
# materialization of this source into the REAL module file
# _utci_poly_gen.py (T14a): numba's disk-cache locators reject an exec'd
# definition (co_filename '<...>' — "no locator available"), so before
# T14a the kernel was exec'd here with cache=False and recompiled in
# every process. Statement text, order, wrappers and njit flags are
# unchanged; the import-time drift gate below pins the generated file to
# this source.
# --------------------------------------------------------------------------
POLY_EXPRESSION_SOURCE = """
acc = ta
acc = nadd(acc, F32(0.607562052))
acc = nadd(acc, nmul((-F32(0.0227712343)), ta))
acc = nadd(acc, nmul(F32(0.000806470249), nmul(ta, ta)))
acc = nadd(acc, nmul((-F32(0.000154271372)), nmul(nmul(ta, ta), ta)))
acc = nadd(acc, nmul((-F32(3.24651735e-06)), _torch_pow_scalar(ta, F32(4.0), use_libm)))
acc = nadd(acc, nmul(F32(7.32602852e-08), _torch_pow_scalar(ta, F32(5.0), use_libm)))
acc = nadd(acc, nmul(F32(1.35959073e-09), _torch_pow_scalar(ta, F32(6.0), use_libm)))
acc = nadd(acc, nmul((-F32(2.2583652)), va))
acc = nadd(acc, nmul(nmul(F32(0.0880326035), ta), va))
acc = nadd(acc, nmul(nmul(F32(0.00216844454), nmul(ta, ta)), va))
acc = nadd(acc, nmul(nmul((-F32(1.53347087e-05)), nmul(nmul(ta, ta), ta)), va))
acc = nadd(acc, nmul(nmul((-F32(5.72983704e-07)), _torch_pow_scalar(ta, F32(4.0), use_libm)), va))
acc = nadd(acc, nmul(nmul((-F32(2.55090145e-09)), _torch_pow_scalar(ta, F32(5.0), use_libm)), va))
acc = nadd(acc, nmul((-F32(0.751269505)), nmul(va, va)))
acc = nadd(acc, nmul(nmul((-F32(0.00408350271)), ta), nmul(va, va)))
acc = nadd(acc, nmul(nmul((-F32(5.21670675e-05)), nmul(ta, ta)), nmul(va, va)))
acc = nadd(acc, nmul(nmul(F32(1.94544667e-06), nmul(nmul(ta, ta), ta)), nmul(va, va)))
acc = nadd(acc, nmul(nmul(F32(1.14099531e-08), _torch_pow_scalar(ta, F32(4.0), use_libm)), nmul(va, va)))
acc = nadd(acc, nmul(F32(0.158137256), nmul(nmul(va, va), va)))
acc = nadd(acc, nmul(nmul((-F32(6.57263143e-05)), ta), nmul(nmul(va, va), va)))
acc = nadd(acc, nmul(nmul(F32(2.22697524e-07), nmul(ta, ta)), nmul(nmul(va, va), va)))
acc = nadd(acc, nmul(nmul((-F32(4.16117031e-08)), nmul(nmul(ta, ta), ta)), nmul(nmul(va, va), va)))
acc = nadd(acc, nmul((-F32(0.0127762753)), _torch_pow_scalar(va, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(9.66891875e-06), ta), _torch_pow_scalar(va, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(2.52785852e-09), nmul(ta, ta)), _torch_pow_scalar(va, F32(4.0), use_libm)))
acc = nadd(acc, nmul(F32(0.000456306672), _torch_pow_scalar(va, F32(5.0), use_libm)))
acc = nadd(acc, nmul(nmul((-F32(1.74202546e-07)), ta), _torch_pow_scalar(va, F32(5.0), use_libm)))
acc = nadd(acc, nmul((-F32(5.91491269e-06)), _torch_pow_scalar(va, F32(6.0), use_libm)))
acc = nadd(acc, nmul(F32(0.398374029), dtm))
acc = nadd(acc, nmul(nmul(F32(0.000183945314), ta), dtm))
acc = nadd(acc, nmul(nmul((-F32(0.00017375451)), nmul(ta, ta)), dtm))
acc = nadd(acc, nmul(nmul((-F32(7.60781159e-07)), nmul(nmul(ta, ta), ta)), dtm))
acc = nadd(acc, nmul(nmul(F32(3.77830287e-08), _torch_pow_scalar(ta, F32(4.0), use_libm)), dtm))
acc = nadd(acc, nmul(nmul(F32(5.43079673e-10), _torch_pow_scalar(ta, F32(5.0), use_libm)), dtm))
acc = nadd(acc, nmul(nmul((-F32(0.0200518269)), va), dtm))
acc = nadd(acc, nmul(nmul(nmul(F32(0.000892859837), ta), va), dtm))
acc = nadd(acc, nmul(nmul(nmul(F32(3.45433048e-06), nmul(ta, ta)), va), dtm))
acc = nadd(acc, nmul(nmul(nmul((-F32(3.77925774e-07)), nmul(nmul(ta, ta), ta)), va), dtm))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.69699377e-09)), _torch_pow_scalar(ta, F32(4.0), use_libm)), va), dtm))
acc = nadd(acc, nmul(nmul(F32(0.000169992415), nmul(va, va)), dtm))
acc = nadd(acc, nmul(nmul(nmul((-F32(4.99204314e-05)), ta), nmul(va, va)), dtm))
acc = nadd(acc, nmul(nmul(nmul(F32(2.47417178e-07), nmul(ta, ta)), nmul(va, va)), dtm))
acc = nadd(acc, nmul(nmul(nmul(F32(1.07596466e-08), nmul(nmul(ta, ta), ta)), nmul(va, va)), dtm))
acc = nadd(acc, nmul(nmul(F32(8.49242932e-05), nmul(nmul(va, va), va)), dtm))
acc = nadd(acc, nmul(nmul(nmul(F32(1.35191328e-06), ta), nmul(nmul(va, va), va)), dtm))
acc = nadd(acc, nmul(nmul(nmul((-F32(6.21531254e-09)), nmul(ta, ta)), nmul(nmul(va, va), va)), dtm))
acc = nadd(acc, nmul(nmul((-F32(4.99410301e-06)), _torch_pow_scalar(va, F32(4.0), use_libm)), dtm))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.89489258e-08)), ta), _torch_pow_scalar(va, F32(4.0), use_libm)), dtm))
acc = nadd(acc, nmul(nmul(F32(8.15300114e-08), _torch_pow_scalar(va, F32(5.0), use_libm)), dtm))
acc = nadd(acc, nmul(F32(0.00075504309), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul((-F32(5.65095215e-05)), ta), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul((-F32(4.52166564e-07)), nmul(ta, ta)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(F32(2.46688878e-08), nmul(nmul(ta, ta), ta)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(F32(2.42674348e-10), _torch_pow_scalar(ta, F32(4.0), use_libm)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(F32(0.00015454725), va), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(nmul(F32(5.2411097e-06), ta), va), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(8.75874982e-08)), nmul(ta, ta)), va), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.50743064e-09)), nmul(nmul(ta, ta), ta)), va), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul((-F32(1.56236307e-05)), nmul(va, va)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.33895614e-07)), ta), nmul(va, va)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(nmul(F32(2.49709824e-09), nmul(ta, ta)), nmul(va, va)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(F32(6.51711721e-07), nmul(nmul(va, va), va)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul(nmul(F32(1.94960053e-09), ta), nmul(nmul(va, va), va)), nmul(dtm, dtm)))
acc = nadd(acc, nmul(nmul((-F32(1.00361113e-08)), _torch_pow_scalar(va, F32(4.0), use_libm)), nmul(dtm, dtm)))
acc = nadd(acc, nmul((-F32(1.21206673e-05)), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul((-F32(2.1820366e-07)), ta), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul(F32(7.51269482e-09), nmul(ta, ta)), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul(F32(9.79063848e-11), nmul(nmul(ta, ta), ta)), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul(F32(1.25006734e-06), va), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.81584736e-09)), ta), va), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(3.52197671e-10)), nmul(ta, ta)), va), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul((-F32(3.3651463e-08)), nmul(va, va)), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul(nmul(F32(1.35908359e-10), ta), nmul(va, va)), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul(nmul(F32(4.1703262e-10), nmul(nmul(va, va), va)), nmul(nmul(dtm, dtm), dtm)))
acc = nadd(acc, nmul((-F32(1.30369025e-09)), _torch_pow_scalar(dtm, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(4.13908461e-10), ta), _torch_pow_scalar(dtm, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(9.22652254e-12), nmul(ta, ta)), _torch_pow_scalar(dtm, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul((-F32(5.08220384e-09)), va), _torch_pow_scalar(dtm, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(2.24730961e-11)), ta), va), _torch_pow_scalar(dtm, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(1.17139133e-10), nmul(va, va)), _torch_pow_scalar(dtm, F32(4.0), use_libm)))
acc = nadd(acc, nmul(F32(6.62154879e-10), _torch_pow_scalar(dtm, F32(5.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(4.0386326e-13), ta), _torch_pow_scalar(dtm, F32(5.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(1.95087203e-12), va), _torch_pow_scalar(dtm, F32(5.0), use_libm)))
acc = nadd(acc, nmul((-F32(4.73602469e-12)), _torch_pow_scalar(dtm, F32(6.0), use_libm)))
acc = nadd(acc, nmul(F32(5.12733497), pa))
acc = nadd(acc, nmul(nmul((-F32(0.312788561)), ta), pa))
acc = nadd(acc, nmul(nmul((-F32(0.0196701861)), nmul(ta, ta)), pa))
acc = nadd(acc, nmul(nmul(F32(0.00099969087), nmul(nmul(ta, ta), ta)), pa))
acc = nadd(acc, nmul(nmul(F32(9.51738512e-06), _torch_pow_scalar(ta, F32(4.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul((-F32(4.66426341e-07)), _torch_pow_scalar(ta, F32(5.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul(F32(0.548050612), va), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.00330552823)), ta), va), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.0016411944)), nmul(ta, ta)), va), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(5.16670694e-06)), nmul(nmul(ta, ta), ta)), va), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(9.52692432e-07), _torch_pow_scalar(ta, F32(4.0), use_libm)), va), pa))
acc = nadd(acc, nmul(nmul((-F32(0.0429223622)), nmul(va, va)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(0.00500845667), ta), nmul(va, va)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(1.00601257e-06), nmul(ta, ta)), nmul(va, va)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.81748644e-06)), nmul(nmul(ta, ta), ta)), nmul(va, va)), pa))
acc = nadd(acc, nmul(nmul((-F32(0.00125813502)), nmul(nmul(va, va), va)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.000179330391)), ta), nmul(nmul(va, va), va)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(2.34994441e-06), nmul(ta, ta)), nmul(nmul(va, va), va)), pa))
acc = nadd(acc, nmul(nmul(F32(0.000129735808), _torch_pow_scalar(va, F32(4.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(1.2906487e-06), ta), _torch_pow_scalar(va, F32(4.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul((-F32(2.28558686e-06)), _torch_pow_scalar(va, F32(5.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul((-F32(0.0369476348)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(0.00162325322), ta), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(3.1427968e-05)), nmul(ta, ta)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(2.59835559e-06), nmul(nmul(ta, ta), ta)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(4.77136523e-08)), _torch_pow_scalar(ta, F32(4.0), use_libm)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(0.0086420339), va), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(0.000687405181)), ta), va), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(9.13863872e-06)), nmul(ta, ta)), va), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul(F32(5.15916806e-07), nmul(nmul(ta, ta), ta)), va), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(3.59217476e-05)), nmul(va, va)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul(F32(3.28696511e-05), ta), nmul(va, va)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(7.10542454e-07)), nmul(ta, ta)), nmul(va, va)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.243823e-05)), nmul(nmul(va, va), va)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(7.385844e-09)), ta), nmul(nmul(va, va), va)), dtm), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(2.20609296e-07), _torch_pow_scalar(va, F32(4.0), use_libm)), dtm), pa))
acc = nadd(acc, nmul(nmul((-F32(0.00073246918)), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.87381964e-05)), ta), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(4.80925239e-06), nmul(ta, ta)), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(8.7549204e-08)), nmul(nmul(ta, ta), ta)), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(2.7786293e-05), va), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(5.06004592e-06)), ta), va), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul(F32(1.14325367e-07), nmul(ta, ta)), va), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(2.53016723e-06), nmul(va, va)), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(1.72857035e-08)), ta), nmul(va, va)), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(3.95079398e-08)), nmul(nmul(va, va), va)), nmul(dtm, dtm)), pa))
acc = nadd(acc, nmul(nmul((-F32(3.59413173e-07)), nmul(nmul(dtm, dtm), dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(7.04388046e-07), ta), nmul(nmul(dtm, dtm), dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.89309167e-08)), nmul(ta, ta)), nmul(nmul(dtm, dtm), dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(4.79768731e-07)), va), nmul(nmul(dtm, dtm), dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(nmul(F32(7.96079978e-09), ta), va), nmul(nmul(dtm, dtm), dtm)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(1.62897058e-09), nmul(va, va)), nmul(nmul(dtm, dtm), dtm)), pa))
acc = nadd(acc, nmul(nmul(F32(3.94367674e-08), _torch_pow_scalar(dtm, F32(4.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.18566247e-09)), ta), _torch_pow_scalar(dtm, F32(4.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul(nmul(F32(3.34678041e-10), va), _torch_pow_scalar(dtm, F32(4.0), use_libm)), pa))
acc = nadd(acc, nmul(nmul((-F32(1.15606447e-10)), _torch_pow_scalar(dtm, F32(5.0), use_libm)), pa))
acc = nadd(acc, nmul((-F32(2.80626406)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(F32(0.548712484), ta), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul((-F32(0.0039942841)), nmul(ta, ta)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul((-F32(0.000954009191)), nmul(nmul(ta, ta), ta)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(F32(1.93090978e-05), _torch_pow_scalar(ta, F32(4.0), use_libm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul((-F32(0.308806365)), va), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(0.0116952364), ta), va), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(0.000495271903), nmul(ta, ta)), va), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.90710882e-05)), nmul(nmul(ta, ta), ta)), va), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(F32(0.00210787756), nmul(va, va)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.000698445738)), ta), nmul(va, va)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(2.30109073e-05), nmul(ta, ta)), nmul(va, va)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(F32(0.00041785659), nmul(nmul(va, va), va)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(1.27043871e-05)), ta), nmul(nmul(va, va), va)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul((-F32(3.04620472e-06)), _torch_pow_scalar(va, F32(4.0), use_libm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(F32(0.0514507424), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.00432510997)), ta), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(8.99281156e-05), nmul(ta, ta)), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(7.14663943e-07)), nmul(nmul(ta, ta), ta)), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.000266016305)), va), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(nmul(F32(0.000263789586), ta), va), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(7.01199003e-06)), nmul(ta, ta)), va), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.000106823306)), nmul(va, va)), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(nmul(F32(3.61341136e-06), ta), nmul(va, va)), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(2.29748967e-07), nmul(nmul(va, va), va)), dtm), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(F32(0.000304788893), nmul(dtm, dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(6.42070836e-05)), ta), nmul(dtm, dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(1.16257971e-06), nmul(ta, ta)), nmul(dtm, dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(7.68023384e-06), va), nmul(dtm, dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(nmul((-F32(5.47446896e-07)), ta), va), nmul(dtm, dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(3.5993791e-08)), nmul(va, va)), nmul(dtm, dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul((-F32(4.36497725e-06)), nmul(nmul(dtm, dtm), dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(1.68737969e-07), ta), nmul(nmul(dtm, dtm), dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(2.67489271e-08), va), nmul(nmul(dtm, dtm), dtm)), nmul(pa, pa)))
acc = nadd(acc, nmul(nmul(F32(3.23926897e-09), _torch_pow_scalar(dtm, F32(4.0), use_libm)), nmul(pa, pa)))
acc = nadd(acc, nmul((-F32(0.0353874123)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul((-F32(0.22120119)), ta), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(F32(0.0155126038), nmul(ta, ta)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul((-F32(0.000263917279)), nmul(nmul(ta, ta), ta)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(F32(0.0453433455), va), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.00432943862)), ta), va), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(0.000145389826), nmul(ta, ta)), va), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(F32(0.00021750861), nmul(va, va)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(6.66724702e-05)), ta), nmul(va, va)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(F32(3.3321714e-05), nmul(nmul(va, va), va)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul((-F32(0.00226921615)), dtm), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(0.000380261982), ta), dtm), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(5.45314314e-09)), nmul(ta, ta)), dtm), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.000796355448)), va), dtm), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul(nmul(F32(2.53458034e-05), ta), va), dtm), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(6.31223658e-06)), nmul(va, va)), dtm), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(F32(0.000302122035), nmul(dtm, dtm)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul((-F32(4.77403547e-06)), ta), nmul(dtm, dtm)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul(nmul(F32(1.73825715e-06), va), nmul(dtm, dtm)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(nmul((-F32(4.09087898e-07)), nmul(nmul(dtm, dtm), dtm)), nmul(nmul(pa, pa), pa)))
acc = nadd(acc, nmul(F32(0.614155345), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul((-F32(0.0616755931)), ta), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(0.00133374846), nmul(ta, ta)), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(0.00355375387), va), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(0.000513027851)), ta), va), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(0.000102449757), nmul(va, va)), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul((-F32(0.00148526421)), dtm), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(4.11469183e-05)), ta), dtm), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul(nmul((-F32(6.80434415e-06)), va), dtm), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(nmul((-F32(9.77675906e-06)), nmul(dtm, dtm)), _torch_pow_scalar(pa, F32(4.0), use_libm)))
acc = nadd(acc, nmul(F32(0.0882773108), _torch_pow_scalar(pa, F32(5.0), use_libm)))
acc = nadd(acc, nmul(nmul((-F32(0.00301859306)), ta), _torch_pow_scalar(pa, F32(5.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(0.00104452989), va), _torch_pow_scalar(pa, F32(5.0), use_libm)))
acc = nadd(acc, nmul(nmul(F32(0.000247090539), dtm), _torch_pow_scalar(pa, F32(5.0), use_libm)))
acc = nadd(acc, nmul(F32(0.00148348065), _torch_pow_scalar(pa, F32(6.0), use_libm)))
"""

_POLY_DEF_SOURCE = (
    "def _utci_poly_element(dtm, ta, va, pa, use_libm):\n"
    + "".join("    " + line + "\n"
              for line in POLY_EXPRESSION_SOURCE.splitlines())
    + "    return acc\n"
)

# --------------------------------------------------------------------------
# T14a: the polynomial kernel definition lives in _utci_poly_gen.py, the
# byte-exact materialization of POLY_EXPRESSION_SOURCE (built below).
# --------------------------------------------------------------------------
_POLY_GEN_SHA256 = hashlib.sha256(
    POLY_EXPRESSION_SOURCE.encode("utf-8")).hexdigest()
_POLY_GEN_HEADER = f'''# SPDX-License-Identifier: GPL-3.0-only
"""GENERATED FILE — UTCI polynomial kernel (ultrafast T09/T14a).

Materialized byte-exactly from
``solweig_core/numba_cpu/utci.py::POLY_EXPRESSION_SOURCE`` (the single
source of truth; the oracle-AST structural-equality pin lives in
``tests/ultrafast/test_utci.py::test_ast_structural_equality``).

This file exists so the njit kernel is backed by a REAL source file and
numba's disk cache can locate it: a function exec'd from a string carries
``co_filename='<utci_poly_element>'`` and every numba cache locator
rejects a non-existent file
(RuntimeError: cannot cache function ... no locator available), which is
why the polynomial kernel was ``cache=False`` before T14a.

``utci.py`` verifies at import time that this file is byte-identical to
``utci._poly_gen_module_source()`` and refuses to import otherwise, so
this file and POLY_EXPRESSION_SOURCE cannot drift (drift gate — it fires
before any numba cache artifact, stale or not, can be served).

DO NOT EDIT BY HAND. After changing POLY_EXPRESSION_SOURCE, regenerate
(from the repo root):

    python -c "from solweig_core.numba_cpu.utci import _poly_gen_module_source as s; print(s(), end='')" > solweig_core/numba_cpu/_utci_poly_gen.py

POLY_EXPRESSION_SOURCE sha256: {_POLY_GEN_SHA256}
"""
from __future__ import annotations

import numpy as np
from numba import njit
from solweig_core.numba_cpu.math_compat import _torch_pow_scalar, nadd, nmul

F32 = np.float32


@njit(cache=True, fastmath=False, error_model="numpy")
'''


def _poly_gen_module_source():
    """Byte-exact regeneration source for ``_utci_poly_gen.py`` (T14a).

    Pure function of ``POLY_EXPRESSION_SOURCE``: header (with its sha256)
    + the ``_utci_poly_element`` definition built by the same statement
    builder that previously fed the exec. The generated module must
    always equal this string; the import-time drift gate enforces it.
    """
    return _POLY_GEN_HEADER + _POLY_DEF_SOURCE


# import-time drift gate (T14a): the generated module must be the
# byte-exact materialization of POLY_EXPRESSION_SOURCE. A drifted
# _utci_poly_gen.py could otherwise compile different arithmetic — or,
# worse, have a stale numba disk-cache artifact served for it (numba's
# staleness check is mtime+size, not content) — so refuse to import.
# The import sits at the bottom: it needs the builders above.
from solweig_core.numba_cpu import _utci_poly_gen  # noqa: E402

if _utci_poly_gen.__file__ is None:  # pragma: no cover - namespace pkg
    raise RuntimeError("_utci_poly_gen has no backing source file")
if Path(_utci_poly_gen.__file__).read_text() != _poly_gen_module_source():
    raise RuntimeError(
        "solweig_core/numba_cpu/_utci_poly_gen.py is not the byte-exact "
        "materialization of utci.POLY_EXPRESSION_SOURCE (T14a drift "
        "gate). Regenerate it with the command in its docstring.")
_utci_poly_element = _utci_poly_gen._utci_poly_element
