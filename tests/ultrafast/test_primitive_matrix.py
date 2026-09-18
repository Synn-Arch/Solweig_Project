"""Primitive characterization matrix tests (DESIGN 10.2, T01).

These are characterization tests over the frozen numeric profile of THIS
host/venv (torch 2.14 CPU + numpy 2.5, darwin/arm64): they pin which
primitive/context/shape tuples are bit-portable between torch-CPU and numpy
and which are not, and they prove the matrix machinery itself detects
injected bit differences.

If any of the "expected agreement" tests fails on new hardware or a new
torch/numpy build, the numeric profile has changed — that is a finding to
record, not a test to relax. The frozen artifacts live in
benchmarks/ultrafast/baseline/{primitive_matrix,numeric_profile}.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import primitive_matrix as pm  # noqa: E402

REPO_ROOT = HERE.parents[1]
MATRIX_JSON = REPO_ROOT / "benchmarks" / "ultrafast" / "baseline" / "primitive_matrix.json"
PROFILE_JSON = REPO_ROOT / "benchmarks" / "ultrafast" / "baseline" / "numeric_profile.json"

# Ops whose torch-CPU and numpy results agree bit-for-bit on the frozen
# profile (IEEE-exact arithmetic + round-to-nearest-even + integer pow).
PORTABLE_OPS = ("add", "sub", "mul", "div", "eq", "lt", "pow_int", "round", "sqrt")
# Ops known to diverge on the frozen profile (different libm lowering and/or
# NaN sign canonicalization).
DIVERGENT_OPS = ("sin", "cos", "tan", "exp", "log", "pow_float")


@pytest.fixture(scope="module")
def fast_matrix():
    return pm.run_matrix(shapes=pm.SHAPES_FAST)


def _records(matrix, *, pair, op=None, array_only=True):
    out = []
    for r in matrix.records:
        if r.get("pair") != pair:
            continue
        if op is not None and r.get("op") != op:
            continue
        if array_only and r.get("context") == "scalar_tensor":
            continue
        if "error" in r:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# coverage and machinery
# ---------------------------------------------------------------------------
class TestMatrixCoverage:
    OPS = ("add", "sub", "mul", "div", "pow_int", "pow_float", "maximum",
           "minimum", "eq", "lt", "round", "log", "exp", "sin", "cos", "tan", "sqrt")

    def test_every_op_every_context_recorded(self, fast_matrix):
        for op in self.OPS:
            scalar = [r for r in fast_matrix.records
                      if r["op"] == op and r["context"] == "scalar_tensor"]
            array_ctx = [r for r in fast_matrix.records
                         if r["op"] == op and r["context"] in ("dense", "broadcast_scalar")]
            assert scalar, f"{op}: no scalar_tensor records"
            assert array_ctx, f"{op}: no array-context records"

    def test_every_shape_boundary_covered(self, fast_matrix):
        shapes_seen = {tuple(r["shape"]) for r in fast_matrix.records
                       if r["context"] == "dense"}
        for shape in pm.SHAPES_FAST:
            assert shape in shapes_seen, f"shape {shape} missing from matrix"
        assert pm.E1_EXTENT_SHAPE in shapes_seen

    def test_special_value_pool_contains_contract_values(self):
        pool_bits = set(pm.special_values().view(pm.U32).tolist())
        must = {
            0x00000000, 0x80000000,                  # +-0
            0x7F800000, 0xFF800000,                  # +-inf
            0x7FC00000, 0xFFC00000,                  # +-quiet NaN
            int(np.float32(-999.0).view(pm.U32)),    # sentinel
            int(np.float32(np.pi).view(pm.U32)),     # trig boundary
        }
        assert must <= pool_bits, must - pool_bits
        # subnormals present
        sub = np.finfo(np.float32).smallest_subnormal
        assert int(sub.view(pm.U32)) in pool_bits

    def test_no_error_cells(self, fast_matrix):
        errors = [r for r in fast_matrix.records if "error" in r]
        assert not errors, errors[:3]

    def test_compacted_dispatch_records_present(self, fast_matrix):
        dc = [r for r in fast_matrix.records if "dense_vs_compacted" in r.get("pair", "")]
        assert dc, "no dense-vs-compacted records emitted"
        backends = {r["pair"].split("_")[0] for r in dc}
        assert backends == {"torch", "numpy"}


class TestMatrixMachineryBites:
    """RED witnesses for the comparison machinery itself (not source code)."""

    def test_injected_one_ulp_is_detected(self):
        a = pm.build_array((64,))
        b = a.copy()
        b[13] = np.nextafter(b[13], np.float32(np.inf))
        rec_bits = pm._bits(a)
        cand_bits = pm._bits(b)
        diff = rec_bits != cand_bits
        assert int(np.count_nonzero(diff)) == 1
        idx = int(np.flatnonzero(diff)[0])
        assert idx == 13

    def test_injected_nan_payload_swap_is_detected(self):
        a = pm.build_array((32,))
        b = a.copy()
        b[7] = np.array(0x7FC00001, dtype=pm.U32).view("<f4")
        assert int(np.count_nonzero(pm._bits(a) != pm._bits(b))) == 1

    def test_injected_signed_zero_flip_is_detected(self):
        a = np.zeros(8, dtype="<f4")
        b = a.copy()
        b[2] = -0.0
        assert int(np.count_nonzero(pm._bits(a) != pm._bits(b))) == 1

    def test_bool_results_compare_as_bits(self):
        cmp_bits = pm._bits(np.array([True, False])) != pm._bits(np.array([True, True]))
        assert int(np.count_nonzero(cmp_bits)) == 1


# ---------------------------------------------------------------------------
# frozen-profile characterization (host-pinned facts)
# ---------------------------------------------------------------------------
class TestFrozenProfileCharacterization:
    def test_portable_ops_agree_bitwise(self, fast_matrix):
        for op in PORTABLE_OPS:
            bad = [r for r in _records(fast_matrix, pair="torch_vs_numpy", op=op)
                   if r.get("mismatch_count")]
            assert not bad, f"{op} diverged: {bad[0]['first_mismatches'][:1]}"

    def test_divergent_ops_have_recorded_mismatches(self, fast_matrix):
        for op in DIVERGENT_OPS:
            bad = [r for r in _records(fast_matrix, pair="torch_vs_numpy", op=op)
                   if r.get("mismatch_count")]
            assert bad, (
                f"{op} expected divergent on the frozen profile but matched; "
                "numeric profile changed — re-record primitive_matrix.json")

    def test_torch_self_repeat_is_deterministic(self, fast_matrix):
        bad = [r for r in _records(fast_matrix, pair="torch_vs_torch_repeat")
               if r.get("mismatch_count")]
        assert not bad, bad[:2]

    def test_numpy_dense_vs_compacted_never_diverges(self, fast_matrix):
        bad = [r for r in _records(fast_matrix, pair="numpy_dense_vs_compacted")
               if r.get("mismatch_count")]
        assert not bad, "numpy ufunc dispatch diverged dense vs compacted"

    def test_torch_dense_vs_compacted_diverges_at_e1_extent(self, fast_matrix):
        """W2-class compaction-order sensitivity, reproduced deliberately.

        torch sin/cos on the masked-gather subset differ from the same
        positions of the dense result at the real E1 extent (248, 220) on
        the frozen profile; small shapes stay clean.
        """
        dc = [r for r in _records(fast_matrix, pair="torch_dense_vs_compacted")
              if r.get("mismatch_count")]
        assert dc, "expected torch dense-vs-compacted divergence at E1 extent"
        for r in dc:
            assert tuple(r["shape"]) == pm.E1_EXTENT_SHAPE
            assert r["op"] in ("sin", "cos")
        small_clean = [r for r in _records(fast_matrix, pair="torch_dense_vs_compacted")
                       if tuple(r["shape"]) != pm.E1_EXTENT_SHAPE]
        assert all(not r.get("mismatch_count") for r in small_clean)

    def test_scalar_context_nan_sign_divergence_recorded(self, fast_matrix):
        """torch canonicalizes NaN sign in log; numpy preserves it."""
        hits = [r for r in fast_matrix.records
                if r["op"] == "log" and r["context"] == "scalar_tensor"
                and r.get("mismatch_count")
                and any(m["classification"] == "nan_payload_or_sign"
                        for m in r["first_mismatches"])]
        assert hits
        m = hits[0]["first_mismatches"][0]
        assert m["reference_bits"] != m["candidate_bits"]


# ---------------------------------------------------------------------------
# numeric profile probes
# ---------------------------------------------------------------------------
class TestNumericProfile:
    def test_weak_scalar_promotion_stays_float32(self):
        probes = pm.promotion_probes()
        assert probes["np_f32_array_plus_python_float"] == "float32"
        assert probes["np_f32_array_plus_python_int"] == "float32"
        assert probes["torch_f32_tensor_plus_python_float"] == "torch.float32"
        assert probes["torch_f32_tensor_plus_python_int"] == "torch.float32"

    def test_strong_scalar_promotion_recorded(self):
        """numpy np.float64 scalars are strong (promote); torch 0-d is weak.

        This asymmetry is a scalar-cast trap for any port that computes a
        constant in float64 and lets it touch the tensor pipeline (DESIGN
        3.4): numpy would silently widen the whole array to float64 where
        torch keeps float32.
        """
        probes = pm.promotion_probes()
        assert probes["np_f32_array_plus_np_float64"] == "float64"
        # torch: 0-dim f64 tensor and np.float64 scalar are weak -> stays f32
        assert probes["torch_f32_tensor_plus_f64_scalar_tensor"] == "torch.float32"
        assert probes["torch_f32_tensor_plus_np_float64"] == "torch.float32"
        # torch: dimensioned f64 tensor is strong -> promotes
        assert probes["torch_f32_tensor_plus_f64_1d_tensor"] == "torch.float64"

    def test_round_is_round_half_to_even_both_backends(self):
        probes = pm.rounding_probes()
        assert probes["numpy_round"] == probes["torch_round"]
        assert probes["numpy_round"][:3] == [0.0, 2.0, 2.0]  # RTNE: 0.5->0, 1.5->2, 2.5->2

    def test_frozen_profile_json_exists_and_is_current_env(self):
        assert PROFILE_JSON.is_file(), "run: python tests/ultrafast/primitive_matrix.py --full ..."
        data = json.loads(PROFILE_JSON.read_text())
        env = data["environment"]
        import torch

        assert env["torch"] == torch.__version__
        assert env["numpy"] == np.__version__

    def test_frozen_matrix_json_coverage(self):
        assert MATRIX_JSON.is_file()
        data = json.loads(MATRIX_JSON.read_text())
        assert data["error_cell_count"] == 0
        shapes_seen = {tuple(r["shape"]) for r in data["records"]
                       if r["context"] == "dense"}
        for shape in pm.SHAPES_FULL:
            assert shape in shapes_seen, f"full sweep missing {shape}"


# ---------------------------------------------------------------------------
# full sweep (slow, opt-in)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestFullSweepOptIn:
    """Full shape sweep — run explicitly with SOLWEIG_ULTRA_PRIMITIVE_FULL=1."""

    def test_full_sweep(self):
        if os.environ.get("SOLWEIG_ULTRA_PRIMITIVE_FULL") != "1":
            pytest.skip("set SOLWEIG_ULTRA_PRIMITIVE_FULL=1 to run the full sweep")
        result = pm.run_matrix(shapes=pm.SHAPES_FULL)
        assert result.summary()["error_cell_count"] == 0
        shapes_seen = {tuple(r["shape"]) for r in result.records
                       if r["context"] == "dense"}
        for shape in pm.SHAPES_FULL:
            assert shape in shapes_seen
