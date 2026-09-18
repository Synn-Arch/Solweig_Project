"""Self-tests for the standalone comparator, not SOLWEIG verification."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bitwise_check import assert_f32_bits_equal, compare_f32_arrays


def f32_bits(values):
    return np.asarray(values, dtype=np.uint32).view(np.float32)


class ExactFloat32ComparatorTests(unittest.TestCase):
    def test_identical_nan_payloads_pass(self):
        a = f32_bits([0x7FC00001, 0xFFC12345, 0x3F800000])
        result = compare_f32_arrays(a, a.copy())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["mismatch_count"], 0)

    def test_distinct_nan_payloads_fail(self):
        a, b = f32_bits([0x7FC00001]), f32_bits([0x7FC00002])
        self.assertTrue(np.array_equal(a, b, equal_nan=True))
        self.assertEqual(compare_f32_arrays(a, b)["mismatch_count"], 1)

    def test_nan_sign_and_signaling_payload_fail(self):
        a = f32_bits([0x7FC00001, 0x7F800001])
        b = f32_bits([0xFFC00001, 0x7FC00001])
        self.assertEqual(compare_f32_arrays(a, b)["mismatch_count"], 2)

    def test_signed_zero_fails(self):
        a, b = f32_bits([0x00000000]), f32_bits([0x80000000])
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(compare_f32_arrays(a, b)["status"], "FAIL")

    def test_one_ulp_fails(self):
        result = compare_f32_arrays(f32_bits([0x3F800000]), f32_bits([0x3F800001]))
        self.assertEqual(result["first_mismatches"][0]["candidate_bits"], "0x3f800001")

    def test_float64_refused(self):
        with self.assertRaises(TypeError):
            compare_f32_arrays(np.ones(3, dtype=np.float64), np.ones(3, dtype=np.float32))

    def test_shape_mismatch(self):
        a = np.arange(6, dtype=np.float32)
        self.assertEqual(compare_f32_arrays(a, a.reshape(2, 3))["reason"], "shape_mismatch")

    def test_strided_and_negative_stride_are_logical(self):
        a = np.arange(120, dtype=np.float32).reshape(10, 12)[::-2, ::3]
        self.assertFalse(a.flags.c_contiguous)
        self.assertEqual(compare_f32_arrays(a, a.copy(), chunk_elements=3)["status"], "PASS")

    def test_chunk_mismatch_coordinates_and_count(self):
        a = np.zeros((3, 7), dtype=np.float32)
        b = a.copy()
        b[0, 5] = 1
        b[2, 6] = 2
        r = compare_f32_arrays(a, b, chunk_elements=4, max_examples=1)
        self.assertEqual(r["mismatch_count"], 2)
        self.assertEqual(r["first_mismatches"][0]["logical_index"], [0, 5])

    def test_scalar(self):
        a = np.array(1.0, dtype=np.float32)
        b = np.array(2.0, dtype=np.float32)
        r = compare_f32_arrays(a, b)
        self.assertEqual(r["first_mismatches"][0]["logical_index"], [])

    def test_empty_policy(self):
        a = np.empty((0, 3), dtype=np.float32)
        self.assertEqual(compare_f32_arrays(a, a)["status"], "PASS")
        self.assertEqual(compare_f32_arrays(a, a, require_nonempty=True)["status"], "FAIL")

    def test_hash_matches_raw_c_order(self):
        a = f32_bits([0x7FC00001, 0x80000000, 0x3F800000])
        result = compare_f32_arrays(a, a, chunk_elements=1)
        self.assertEqual(result["reference_raw_data_sha256"], hashlib.sha256(a.tobytes()).hexdigest())

    def test_big_endian_same_and_different(self):
        a = np.array([1, 2], dtype=">f4")
        b = a.copy()
        self.assertEqual(compare_f32_arrays(a, b)["status"], "PASS")
        b[1] = 3
        self.assertEqual(compare_f32_arrays(a, b)["first_mismatches"][0]["reference_bits"], "0x40000000")
        self.assertEqual(compare_f32_arrays(a, a.astype("<f4"))["reason"], "dtype_or_byteorder_mismatch")

    def test_inputs_are_unchanged(self):
        a, b = f32_bits([0x7F800001, 0x80000000]), f32_bits([0x7FC00001, 0x00000000])
        old_a, old_b = a.tobytes(), b.tobytes()
        compare_f32_arrays(a, b)
        self.assertEqual(a.tobytes(), old_a)
        self.assertEqual(b.tobytes(), old_b)

    def test_assert_helper(self):
        with self.assertRaises(AssertionError):
            assert_f32_bits_equal(f32_bits([0]), f32_bits([0x80000000]))

    def test_bad_options(self):
        a = np.ones(1, dtype=np.float32)
        with self.assertRaises(ValueError):
            compare_f32_arrays(a, a, chunk_elements=0)
        with self.assertRaises(ValueError):
            compare_f32_arrays(a, a, max_examples=-1)

    def test_npy_cli_and_memmap(self):
        script = Path(__file__).with_name("bitwise_check.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "a.npy", f32_bits([0x7FC00001, 0x80000000]))
            np.save(root / "b.npy", f32_bits([0x7FC00002, 0x80000000]))
            p = subprocess.run(
                [sys.executable, str(script), str(root / "a.npy"), str(root / "b.npy"),
                 "--require-nonempty", "--report", str(root / "report.json")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(p.returncode, 1, p.stderr)
            self.assertEqual(json.loads((root / "report.json").read_text())["mismatch_count"], 1)

    def test_cli_cannot_overwrite_input(self):
        script = Path(__file__).with_name("bitwise_check.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.npy"
            np.save(a, np.ones(1, dtype=np.float32))
            original = a.read_bytes()
            p = subprocess.run(
                [sys.executable, str(script), str(a), str(a), "--report", str(a)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(p.returncode, 2)
            self.assertEqual(a.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
