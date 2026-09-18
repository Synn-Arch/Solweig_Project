# SPDX-License-Identifier: GPL-3.0-only
"""T12 GATE 5 — CUDA UTCI stage bit-equality (TASKS T12, DESIGN 9.3).

The CUDA UTCI kernels (native/cuda/src/sw_utci.cu + the GENERATED
211-statement polynomial include/sw_utci_poly_generated.cuh) must
reproduce the CANONICAL CPU kernels (solweig_core/numba_cpu/utci.py)
BITS on:

* dense planes — the -999-filled output plane, raw uint32 equality,
  signed zero + NaN payload, no tolerance;
* sparse vectors — the pre-compacted valid lanes, including odd-length
  tails (the opmath scalar-pow remainder lanes of the torch layout);
* the torch chunk-layout lane model — a lane's value depends ONLY on
  its inputs and (k, n, T, grain): same values at ordinals straddling a
  chunk tail DIFFER (both CPU and CUDA), and the same (values, n) under
  different plane geometry is bit-identical (witness 7);
* layout-parameter agreement — the host's torch_threads/grain equal the
  CPU module's DEFAULT_TORCH_THREADS/TORCH_PAR_GRAIN.

RED witnesses at this stage: FMA contraction / reordered accumulation /
f64 promotion inside the 211-term chain (witness 5 — the raw pins kill
any re-association), approximate intrinsics in the es chain (witness 3),
CPU/GPU library discrepancy in exp/log/pow (witness 4), and schedule
dependence of the lane class (witness 7 — sw_is_libm_lane is a pure
function of ordinal+extent; any threadIdx/blockIdx leak mutates it).

The reference is the LIVE canonical CPU module (numba) — the GPU host
runs both sides in one process, so the differential needs no pin file
and still covers arbitrary extents.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from sw_cuda_harness import (  # noqa: E402
    compare_bits,
    load_host_module,
    require_canonical_runtime,
)

REPO_ROOT = ULTRA_DIR.parents[1]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"


@pytest.fixture(scope="module")
def rt():
    return require_canonical_runtime()


@pytest.fixture(scope="module")
def host():
    return load_host_module()


def _cpu():
    sys.path.insert(0, str(REPO_ROOT))
    from solweig_core.numba_cpu import utci as cpu_utci
    from solweig_core.numba_cpu import math_compat
    return cpu_utci, math_compat


# ---------------------------------------------------------------------------
# deterministic case builder — seeded, physical UTCI domains, sentinel and
# NaN lanes, duplicated value tuples (ordinal-dependence probes)
# ---------------------------------------------------------------------------

def make_planes(rng, rows, cols, invalid_frac=0.05, nan_frac=0.02,
                dup_rows=0):
    ta = rng.uniform(-10.0, 40.0, (rows, cols))
    rh = rng.uniform(5.0, 100.0, (rows, cols))
    tmrt = rng.uniform(10.0, 80.0, (rows, cols))
    va = rng.uniform(0.15, 15.0, (rows, cols))
    if invalid_frac > 0:
        m = rng.random((rows, cols)) < invalid_frac
        for arr in (ta, rh, tmrt, va):
            arr[m] = -999.0
    if nan_frac > 0:
        m = rng.random((rows, cols)) < nan_frac
        ta[m] = np.nan
        tmrt[m & (rng.random((rows, cols)) < 0.5)] = np.nan
    if dup_rows:
        for r in range(1, min(dup_rows, rows)):
            ta[r] = ta[0]
            rh[r] = rh[0]
            tmrt[r] = tmrt[0]
            va[r] = va[0]
    return tuple(np.ascontiguousarray(a, dtype=np.float32)
                 for a in (ta, rh, tmrt, va))


# ---------------------------------------------------------------------------
# generated-source discipline (no GPU / no numba needed)
# ---------------------------------------------------------------------------


class TestUtciGeneratedSource:
    def test_generated_header_matches_cpu_poly_source(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "sw_gen_utci", NATIVE_CUDA / "tools" / "gen_utci_cuda.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        committed = (NATIVE_CUDA / "include" /
                     "sw_utci_poly_generated.cuh").read_text()
        assert committed == mod.generate(), (
            "sw_utci_poly_generated.cuh drifted from utci.py "
            "POLY_EXPRESSION_SOURCE (regenerate: python "
            "native/cuda/tools/gen_utci_cuda.py)"
        )

    def test_layout_params_match_cpu_module(self, host):
        cpu_utci, math_compat = _cpu()
        assert host.UTCI_TORCH_THREADS == math_compat.DEFAULT_TORCH_THREADS
        assert host.UTCI_PAR_GRAIN == math_compat.TORCH_PAR_GRAIN


# ---------------------------------------------------------------------------
# dense raw-bit equality (live canonical CPU differential)
# ---------------------------------------------------------------------------


class TestUtciDenseRawBitEquality:
    @pytest.mark.parametrize("shape", [(37, 41), (64, 64), (181, 181),
                                       (200, 210), (256, 514)])
    def test_dense_plane_bits(self, rt, host, shape):
        cpu_utci, _ = _cpu()
        rng = np.random.default_rng(0x5eed_0000 + shape[0] * 1000 + shape[1])
        ta, rh, tmrt, va = make_planes(rng, *shape, dup_rows=3)
        want = cpu_utci.utci_calculator_dense(ta, rh, tmrt, va)
        got = host.run_utci_dense(rt, ta, rh, tmrt, va)
        msg, _ = compare_bits(want, got, f"dense{shape}")
        assert msg == "PASS", msg

    def test_dense_all_invalid_and_all_valid_edges(self, rt, host):
        cpu_utci, _ = _cpu()
        # all-invalid: the plane is exactly the -999 prefill
        z = np.full((13, 17), np.float32(-999.0))
        msg, _ = compare_bits(
            cpu_utci.utci_calculator_dense(z, z, z, z),
            host.run_utci_dense(rt, z, z, z, z), "all-invalid")
        assert msg == "PASS", msg
        # all-valid with NaN lanes (NaN stays a VALID lane, propagates)
        rng = np.random.default_rng(7)
        ta, rh, tmrt, va = make_planes(rng, 33, 29, invalid_frac=0.0)
        msg, _ = compare_bits(
            cpu_utci.utci_calculator_dense(ta, rh, tmrt, va),
            host.run_utci_dense(rt, ta, rh, tmrt, va), "all-valid+nan")
        assert msg == "PASS", msg

    def test_dense_deterministic_across_runs(self, rt, host):
        rng = np.random.default_rng(99)
        ta, rh, tmrt, va = make_planes(rng, 64, 64)
        a = host.run_utci_dense(rt, ta, rh, tmrt, va)
        b = host.run_utci_dense(rt, ta, rh, tmrt, va)
        msg, _ = compare_bits(a, b, "dense-rerun")
        assert msg == "PASS", msg


# ---------------------------------------------------------------------------
# sparse raw-bit equality + dense/sparse consistency
# ---------------------------------------------------------------------------


class TestUtciSparseRawBitEquality:
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 40005, 70000])
    def test_sparse_vector_bits(self, rt, host, n):
        cpu_utci, _ = _cpu()
        rng = np.random.default_rng(0x5_eed + n)
        ta = rng.uniform(-10.0, 40.0, n)
        rh = rng.uniform(5.0, 100.0, n)
        tmrt = rng.uniform(10.0, 80.0, n)
        va = rng.uniform(0.15, 15.0, n)
        if n > 100:
            ta[n // 2] = np.nan  # NaN payload lane
            ta[3] = ta[5]        # duplicate tuple at different ordinals
        vecs = tuple(np.ascontiguousarray(v, dtype=np.float32)
                     for v in (ta, rh, tmrt, va))
        want = cpu_utci.utci_calculator_sparse(*vecs)
        got = host.run_utci_sparse(rt, *vecs)
        msg, _ = compare_bits(want, got, f"sparse{n}")
        assert msg == "PASS", msg

    def test_dense_and_sparse_agree_on_the_same_valid_set(self, rt, host):
        cpu_utci, _ = _cpu()
        rng = np.random.default_rng(0xC0FFEE)
        ta, rh, tmrt, va = make_planes(rng, 200, 210)
        cv = cpu_utci.compact_valid(ta, rh, tmrt, va)
        cpu_dense = cpu_utci.utci_calculator_dense(ta, rh, tmrt, va)
        cpu_sparse = cpu_utci.utci_calculator_sparse(*cv)
        gpu_dense = host.run_utci_dense(rt, ta, rh, tmrt, va)
        gpu_sparse = host.run_utci_sparse(rt, *cv)
        # dense route's valid lanes == sparse route (both devices)
        for name, dense, sparse in (("cpu", cpu_dense, cpu_sparse),
                                    ("gpu", gpu_dense, gpu_sparse)):
            neg = np.float32(-999.0)
            valid = ~((ta <= neg) | (rh <= neg) | (va <= neg) | (tmrt <= neg))
            msg, _ = compare_bits(
                np.ascontiguousarray(sparse),
                np.ascontiguousarray(dense)[valid], f"{name}:dense==sparse")
            assert msg == "PASS", msg
        msg, _ = compare_bits(gpu_sparse, cpu_sparse, "gpu-sparse==cpu-sparse")
        assert msg == "PASS", msg


# ---------------------------------------------------------------------------
# torch chunk-layout lane model (witness 7: ordinal+extent only)
# ---------------------------------------------------------------------------


class TestUtciLaneLayout:
    def test_straddling_ordinals_differ_and_cuda_matches_both(self, rt, host):
        """n=40005, T=8, grain=32768: teff=2, C=20003; chunk-0 tail lanes
        20000..20002 are opmath scalar-pow lanes. The SAME value tuple at
        a vector-body ordinal and a chunk-tail ordinal must differ in the
        CPU reference (proving the layout is real) and the CUDA kernel
        must reproduce BOTH bits exactly.

        The 1-ulp pow split only survives the 211-op chain where a pow
        term has a large coefficient — the Pa polynomial terms (coef up
        to 0.61) — so candidate tuples sweep RH upward to amplify Pa and
        the test picks a pair the REFERENCE itself splits."""
        cpu_utci, _ = _cpu()
        n = 40005
        body, tail = 100, 20000

        # cheap prefilter: an n=9 vector has exactly ONE libm lane (8) —
        # the same tuple at lanes 0 and 8 splits iff the layout moves bits
        # for that tuple. The 1-ulp pow split only survives the 211-op
        # chain where a pow term carries a large coefficient (the Pa
        # terms) AND the derived Pa bits land on a splitting mantissa —
        # so the lattice sweeps tmrt (hence D_Tmrt AND Pa bits) and rh
        # together until the REFERENCE itself splits; that tuple is then
        # pinned in the 40005-lane vector.
        f = np.float32
        chosen = None
        for tmrt_i in (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0):
            for rh_i in (25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 85.0, 100.0):
                vals = (f(21.7), f(rh_i), f(tmrt_i), f(3.9))
                small = tuple(np.full(9, v, dtype=np.float32) for v in vals)
                ref9 = cpu_utci.utci_calculator_sparse(*small)
                if ref9[0].view(np.uint32) != ref9[8].view(np.uint32):
                    chosen = vals
                    break
            if chosen is not None:
                break
        assert chosen is not None, (
            "reference lost ordinal dependence: no (tmrt, rh) in the "
            "lattice splits lanes 0 vs 8 — the torch chunk-layout model "
            "is not being exercised"
        )

        rng = np.random.default_rng(0x57ADD1)
        ta = rng.uniform(-5.0, 38.0, n)
        rh = rng.uniform(20.0, 95.0, n)
        tmrt = rng.uniform(20.0, 75.0, n)
        va = rng.uniform(0.5, 10.0, n)
        ta[body] = ta[tail] = chosen[0]
        rh[body] = rh[tail] = chosen[1]
        tmrt[body] = tmrt[tail] = chosen[2]
        va[body] = va[tail] = chosen[3]
        vecs = tuple(np.ascontiguousarray(v, dtype=np.float32)
                     for v in (ta, rh, tmrt, va))
        want = cpu_utci.utci_calculator_sparse(*vecs)
        assert (want[body].view(np.uint32)
                != want[tail].view(np.uint32)), (
            "n=9 prefilter split but the n=40005 straddle pair did not — "
            "chunk-tail ordinals moved (layout model mismatch)"
        )
        got = host.run_utci_sparse(rt, *vecs)
        msg, _ = compare_bits(want, got, "straddle")
        assert msg == "PASS", msg

    def test_geometry_invariance_same_values_same_n(self, rt, host):
        """Same 400 valid values in row-major order under different plane
        geometry (400x1 / 20x200 / 40x10) are bit-identical: the lane
        value depends on (inputs, k, n) only, never (rows, cols)."""
        cpu_utci, _ = _cpu()
        rng = np.random.default_rng(0x6E0)
        n = 400
        ta = rng.uniform(-5.0, 38.0, n)
        rh = rng.uniform(20.0, 95.0, n)
        tmrt = rng.uniform(20.0, 75.0, n)
        va = rng.uniform(0.5, 10.0, n)
        vecs = tuple(np.ascontiguousarray(v, dtype=np.float32)
                     for v in (ta, rh, tmrt, va))
        ref = host.run_utci_sparse(rt, *vecs)
        for rows, cols in ((400, 1), (20, 20), (40, 10)):
            planes = tuple(v.reshape(rows, cols).copy() for v in vecs)
            got = host.run_utci_dense(rt, *planes)
            msg, _ = compare_bits(ref, got.reshape(-1), f"geom{rows}x{cols}")
            assert msg == "PASS", msg
        cpu_ref = cpu_utci.utci_calculator_dense(
            *[v.reshape(20, 20).copy() for v in vecs])
        msg, _ = compare_bits(cpu_ref.reshape(-1), ref, "cpu-geom==gpu-sparse")
        assert msg == "PASS", msg
