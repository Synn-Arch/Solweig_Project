"""Repository bitwise-parity harness tests (T01).

Two-sided proof structure:
* GREEN cases prove the harness accepts genuinely equal planes (so it is not
  vacuously failing);
* RED witnesses prove each contract violation FAILs: signed zero, NaN
  payload/sign, 1-ULP, metadata (global time index / window origin), shape,
  dtype/byte order, empty-refusal, sentinel-vs-NaN mask swap, one-sided
  metadata, missing planes;
* self-parity proves the repository comparator and the design package's
  comparator agree on every verdict over a randomized corpus.

The witnesses ARE the tests: each injects a specific defect into candidate
data and asserts the harness reports FAIL with the right reason. No
production source is mutated to obtain the red results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bitwise_harness import (  # noqa: E402
    PlaneMetadata,
    assert_planes_equal,
    compare_plane_bits,
    compare_plane_bundles,
    cross_check_design_comparator,
)

U32 = np.dtype("<u4")


def special_plane(n=64):
    """A float32 plane containing every delicate value class, deterministically."""
    rng = np.random.default_rng(20090811)
    base = rng.standard_normal(n).astype(np.float32)
    specials = np.array([
        0.0, -0.0,                       # signed zeros
        np.float32(1.0), np.nextafter(np.float32(1.0), np.float32(2.0)),  # 1-ULP pair
        np.finfo(np.float32).max, np.finfo(np.float32).min,   # +-FLT_MAX
        np.finfo(np.float32).smallest_subnormal, -np.finfo(np.float32).smallest_subnormal,
        np.float32(-999.0),              # validity sentinel
        np.float32(np.pi), np.float32(-0.5),
        np.float32(55.7), np.float32(3.3e38),  # UTCI-ish and near-overflow
    ], dtype=np.float32)
    plane = np.concatenate([base, specials]).astype(np.float32)
    return plane


META_A = dict(global_time_index=(12,), window_origin=(0, 0))


def meta(**over):
    d = dict(plane="utci", global_time_index=(12,), window_origin=(126, 140))
    d.update(over)
    return PlaneMetadata(**d)


# ---------------------------------------------------------------------------
# GREEN: equal planes must PASS
# ---------------------------------------------------------------------------
class TestGreenEquality:
    def test_identical_arrays_pass(self):
        a = special_plane()
        rep = compare_plane_bits(a, a.copy())
        assert rep["status"] == "PASS"
        assert rep["mismatch_count"] == 0
        assert rep["reason"] == "equal"
        assert rep["reference_raw_data_sha256"] == rep["candidate_raw_data_sha256"]

    def test_same_nan_payload_everywhere_passes(self):
        bits = np.full(32, 0x7FC00000, dtype=U32)
        a = bits.view("<f4")
        b = np.full(32, 0x7FC00000, dtype=U32).view("<f4")
        rep = compare_plane_bits(a, b)
        assert rep["status"] == "PASS"
        assert rep["mask_comparison"]["nan_mask"]["reference_count"] == 32

    def test_noncontiguous_view_logically_identical_passes(self):
        base = np.arange(96, dtype=np.float32).reshape(6, 16)
        sliced = base[:, ::2]          # noncontiguous, strides differ
        direct = np.ascontiguousarray(sliced)
        assert not sliced.flags["C_CONTIGUOUS"]
        assert sliced.strides != direct.strides
        rep = compare_plane_bits(sliced, direct)
        assert rep["status"] == "PASS"

    def test_negative_stride_view_passes(self):
        a = special_plane(48)
        flipped_view = a[::-1]
        flipped_copy = np.ascontiguousarray(a[::-1].copy())
        rep = compare_plane_bits(flipped_view, flipped_copy)
        assert rep["status"] == "PASS"

    def test_multidimensional_c_order_sequence_passes(self):
        a = special_plane(287).reshape(6, 50)  # 287 base + 13 specials = 300
        rep = compare_plane_bits(a, a.copy(order="C"))
        assert rep["status"] == "PASS"

    def test_identical_metadata_passes(self):
        a = special_plane()
        rep = compare_plane_bits(a, a.copy(), ref_meta=meta(), cand_meta=meta())
        assert rep["status"] == "PASS"
        assert rep["metadata_comparison"]["checked"] is True

    def test_big_endian_pair_with_same_byteorder_passes(self):
        a = special_plane()
        rep = compare_plane_bits(a.astype(">f4"), a.astype(">f4"))
        assert rep["status"] == "PASS"

    def test_assert_planes_equal_no_raise(self):
        a = special_plane()
        assert_planes_equal(a, a.copy())  # must not raise


# ---------------------------------------------------------------------------
# RED witnesses: each defect must FAIL
# ---------------------------------------------------------------------------
class TestRedWitnesses:
    def test_signed_zero_flip_fails(self):
        ref = np.zeros(16, dtype="<f4")
        cand = np.zeros(16, dtype="<f4")
        cand[7] = -0.0  # single element sign flip
        rep = compare_plane_bits(ref, cand)
        assert rep["status"] == "FAIL"
        assert rep["mismatch_count"] == 1
        cls = rep["first_mismatches"][0]["classification"]
        assert cls == "signed_zero"
        assert rep["first_mismatches"][0]["reference_bits"] == "0x00000000"
        assert rep["first_mismatches"][0]["candidate_bits"] == "0x80000000"

    def test_distinct_nan_payloads_fail(self):
        ref = np.full(8, 0x7FC00000, dtype=U32).view("<f4")  # quiet NaN
        cand = np.full(8, 0x7FC00001, dtype=U32).view("<f4")  # payload differs
        rep = compare_plane_bits(ref, cand)
        assert rep["status"] == "FAIL"
        assert rep["mismatch_count"] == 8
        assert rep["first_mismatches"][0]["classification"] == "nan_payload"

    def test_nan_sign_flip_fails(self):
        ref = np.full(4, 0x7FC00000, dtype=U32).view("<f4")
        cand = np.full(4, 0xFFC00000, dtype=U32).view("<f4")  # sign bit set
        rep = compare_plane_bits(ref, cand)
        assert rep["status"] == "FAIL"
        assert rep["first_mismatches"][0]["classification"] == "nan_sign_or_payload"

    def test_one_ulp_delta_fails(self):
        ref = special_plane(63)
        cand = ref.copy()
        cand[31] = np.nextafter(cand[31], np.float32(np.inf))
        assert cand[31] != ref[31]
        rep = compare_plane_bits(ref, cand)
        assert rep["status"] == "FAIL"
        assert rep["mismatch_count"] == 1
        assert rep["first_mismatches"][0]["classification"] == "finite_1_ulp"

    def test_wrong_global_time_index_fails_despite_identical_bytes(self):
        a = special_plane()
        rep = compare_plane_bits(
            a, a.copy(),
            ref_meta=meta(global_time_index=(12,)),
            cand_meta=meta(global_time_index=(13,)),
        )
        assert rep["status"] == "FAIL"
        assert rep["mismatch_count"] == 0          # bytes identical
        assert rep["reason"] == "metadata_mismatch"
        diffs = rep["metadata_comparison"]["differences"]
        assert diffs["global_time_index"]["reference"] == [12]
        assert diffs["global_time_index"]["candidate"] == [13]

    def test_wrong_window_origin_fails_despite_identical_bytes(self):
        a = special_plane()
        rep = compare_plane_bits(
            a, a.copy(),
            ref_meta=meta(window_origin=(126, 140)),
            cand_meta=meta(window_origin=(0, 0)),
        )
        assert rep["status"] == "FAIL"
        assert rep["reason"] == "metadata_mismatch"
        assert "window_origin" in rep["metadata_comparison"]["differences"]

    def test_wrong_plane_name_fails(self):
        a = special_plane()
        rep = compare_plane_bits(
            a, a.copy(), ref_meta=meta(plane="utci"), cand_meta=meta(plane="tmrt"))
        assert rep["status"] == "FAIL"

    def test_metadata_only_on_one_side_fails(self):
        a = special_plane()
        rep = compare_plane_bits(a, a.copy(), ref_meta=meta(), cand_meta=None)
        assert rep["status"] == "FAIL"
        assert rep["metadata_comparison"]["reason"] == "metadata_missing_on_one_side"

    def test_shape_mismatch_fails(self):
        ref = np.zeros((24, 4, 5), dtype="<f4")
        cand = np.zeros((24, 5, 4), dtype="<f4")  # rows/cols transposed
        rep = compare_plane_bits(ref, cand)
        assert rep["status"] == "FAIL"
        assert rep["reason"] == "shape_mismatch"

    def test_dtype_byteorder_mismatch_fails(self):
        a = special_plane()
        rep = compare_plane_bits(a, a.astype(">f4"))
        assert rep["status"] == "FAIL"
        assert rep["reason"] == "dtype_or_byteorder_mismatch"

    def test_wrong_base_dtype_rejected(self):
        a = special_plane()
        with pytest.raises(TypeError, match="float32"):
            compare_plane_bits(a, a.astype(np.float64))

    def test_empty_comparison_refused_when_required(self):
        a = np.zeros((0,), dtype="<f4")
        rep = compare_plane_bits(a, a.copy(), require_nonempty=True)
        assert rep["status"] == "FAIL"
        assert rep["reason"] == "empty_comparison_refused"

    def test_empty_comparison_allowed_without_flag(self):
        a = np.zeros((0,), dtype="<f4")
        rep = compare_plane_bits(a, a.copy())
        assert rep["status"] == "PASS"

    def test_sentinel_swapped_for_nan_fails_and_mask_report_shows_it(self):
        ref = np.full(32, -999.0, dtype="<f4")
        cand = np.full(32, -999.0, dtype="<f4")
        cand[5] = np.float32("nan")  # invalid-mask value class swap
        rep = compare_plane_bits(ref, cand)
        assert rep["status"] == "FAIL"
        assert rep["mismatch_count"] == 1
        masks = rep["mask_comparison"]
        assert masks["sentinel_minus_999_mask"]["mask_bit_mismatch_count"] == 1
        assert masks["nan_mask"]["mask_bit_mismatch_count"] == 1
        assert rep["first_mismatches"][0]["classification"] == "nan_vs_non_nan"

    def test_sentinel_swapped_for_finite_fails(self):
        ref = np.full(16, -999.0, dtype="<f4")
        cand = np.full(16, -999.0, dtype="<f4")
        cand[3] = np.float32(0.0)
        rep = compare_plane_bits(ref, cand)
        assert rep["status"] == "FAIL"
        assert rep["first_mismatches"][0]["classification"] == "sentinel_minus_999"


# ---------------------------------------------------------------------------
# Bundle comparison over directories
# ---------------------------------------------------------------------------
def write_bundle(root: Path, planes: dict[str, np.ndarray], metadata: dict | None):
    root.mkdir(parents=True, exist_ok=True)
    for name, arr in planes.items():
        np.save(root / f"{name}.f32.npy", np.ascontiguousarray(arr, dtype="<f4"))
    if metadata is not None:
        (root / "metadata.json").write_text(json.dumps({"planes": metadata}))


class TestBundleComparison:
    def make_pair(self, tmp_path, mutate=None, with_meta=True):
        ref = {"utci": special_plane(64), "tmrt": special_plane(64) + 10}
        cand = {k: v.copy() for k, v in ref.items()}
        meta_a = {
            "utci": {"global_time_index": [12], "window_origin": [0, 0]},
            "tmrt": {"global_time_index": [12], "window_origin": [0, 0]},
        } if with_meta else None
        meta_b = json.loads(json.dumps(meta_a)) if with_meta else None
        if mutate:
            mutate(cand, meta_b)
        write_bundle(tmp_path / "ref", ref, meta_a)
        write_bundle(tmp_path / "cand", cand, meta_b)
        return tmp_path / "ref", tmp_path / "cand"

    def test_equal_bundles_pass(self, tmp_path):
        r, c = self.make_pair(tmp_path)
        rep = compare_plane_bundles(r, c)
        assert rep.all_equal is True
        assert set(rep.planes) == {"utci.f32.npy", "tmrt.f32.npy"}

    def test_one_bit_delta_fails_bundle(self, tmp_path):
        def mutate(cand, _meta):
            cand["utci"][10] = np.nextafter(cand["utci"][10], np.float32(np.inf))
        r, c = self.make_pair(tmp_path, mutate=mutate)
        rep = compare_plane_bundles(r, c)
        assert rep.all_equal is False
        assert rep.planes["utci.f32.npy"]["mismatch_count"] == 1

    def test_missing_plane_fails(self, tmp_path):
        def drop(cand, _meta):
            cand.pop("tmrt")
        r, c = self.make_pair(tmp_path, mutate=drop)
        rep = compare_plane_bundles(r, c)
        assert rep.all_equal is False
        assert rep.plane_set["reference_only"] == ["tmrt.f32.npy"]

    def test_wrong_time_index_in_sidecar_fails(self, tmp_path):
        def mutate(_cand, meta_b):
            meta_b["utci"]["global_time_index"] = [11]
        r, c = self.make_pair(tmp_path, mutate=mutate)
        rep = compare_plane_bundles(r, c)
        assert rep.all_equal is False
        assert rep.planes["utci.f32.npy"]["reason"] == "metadata_mismatch"

    def test_sidecar_on_one_side_only_fails(self, tmp_path):
        r, c = self.make_pair(tmp_path, with_meta=True)
        (c / "metadata.json").unlink()
        rep = compare_plane_bundles(r, c)
        assert rep.all_equal is False
        for plane_rep in rep.planes.values():
            assert plane_rep["reason"] == "metadata_missing_on_one_side"

    def test_empty_reference_dir_fails(self, tmp_path):
        r, c = self.make_pair(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        rep = compare_plane_bundles(empty, c)
        assert rep.all_equal is False


# ---------------------------------------------------------------------------
# Self-parity: repository comparator vs design package comparator
# ---------------------------------------------------------------------------
class TestSelfParity:
    CORPUS_SEED = 0xC0FFEE

    def test_verdicts_agree_on_random_corpus(self):
        rng = np.random.default_rng(self.CORPUS_SEED)
        for trial in range(60):
            n = int(rng.integers(1, 5000))
            a = rng.standard_normal(n).astype(np.float32)
            b = a.copy()
            # random corruption: sign flips, ulp nudges, NaN payloads, sentinels
            k = int(rng.integers(0, max(1, n // 10)))
            idx = rng.integers(0, n, size=k)
            mode = int(rng.integers(0, 4))
            if mode == 0:
                b[idx] = -b[idx]
            elif mode == 1:
                b[idx] = np.nextafter(b[idx], np.float32(np.inf))
            elif mode == 2:
                bits = np.full(k, 0x7FC00000, dtype=U32)
                bits += rng.integers(0, 16, size=k).astype(U32)
                b[idx] = bits.view("<f4")
            else:
                b[idx] = np.float32(-999.0)
            cross = cross_check_design_comparator(a, b)
            assert cross["agree"] is True, (trial, cross)
            assert cross["repository_mismatch_count"] == \
                cross["design_package_mismatch_count"]

    def test_verdicts_agree_on_special_planes(self):
        a = special_plane()
        with np.errstate(over="ignore"):
            cases = [a.copy(), -a, a * np.float32(1.0000001)]
        for b in cases:
            cross = cross_check_design_comparator(a, b.astype(np.float32))
            assert cross["agree"] is True

    def test_design_package_path_exists(self):
        from bitwise_harness import DESIGN_TOOLS
        assert (DESIGN_TOOLS / "bitwise_check.py").is_file()
