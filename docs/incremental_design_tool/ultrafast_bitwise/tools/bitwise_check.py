"""Exact float32 .npy comparison, including signed zeros and NaN payloads.

This utility compares logical C-order element bytes without numerical casting.
It does not validate SOLWEIG physics, revisions, coverage, or backend profiles.
Those contracts must be checked by the repository integration harness.

Exit codes: 0 = equal, 1 = unequal, 2 = invalid input / execution error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _validate_array(value: np.ndarray, name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, not {type(value).__name__}")
    if value.dtype.kind != "f" or value.dtype.itemsize != 4:
        raise TypeError(f"{name} must have float32 dtype; got {value.dtype.str}")


def compare_f32_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_examples: int = 8,
    chunk_elements: int = 262_144,
    require_nonempty: bool = False,
) -> dict[str, Any]:
    """Compare exact float32 element bits; no equal_nan or tolerance semantics.

    Array strides may differ. Logical shape, dtype including byte order, and
    C-order element bytes must agree. Metadata such as scene/time/origin is
    deliberately left to the caller. Input arrays are not modified.

    .flat[start:stop] copies only the current chunk, preserving float32 bits
    while supporting negative strides and noncontiguous input. This is a
    correctness helper, not a microbenchmark implementation.
    """
    _validate_array(reference, "reference")
    _validate_array(candidate, "candidate")
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")

    base: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "float32_logical_c_order_raw_bits",
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": reference.dtype.str,
        "candidate_dtype": candidate.dtype.str,
        "reference_elements": int(reference.size),
        "candidate_elements": int(candidate.size),
        "signed_zero_bits_compared": True,
        "nan_payload_bits_compared": True,
        "coverage_and_profile_checked": False,
    }
    if reference.shape != candidate.shape:
        return {**base, "status": "FAIL", "reason": "shape_mismatch", "mismatch_count": None}
    if reference.dtype.str != candidate.dtype.str:
        return {**base, "status": "FAIL", "reason": "dtype_or_byteorder_mismatch", "mismatch_count": None}
    if require_nonempty and reference.size == 0:
        return {**base, "status": "FAIL", "reason": "empty_comparison_refused", "mismatch_count": None}

    ref_hash = hashlib.sha256()
    got_hash = hashlib.sha256()
    mismatches = 0
    examples: list[dict[str, Any]] = []
    # Viewing with matching endian gives meaningful hexadecimal bit patterns.
    uint_dtype = np.dtype(reference.dtype.str.replace("f", "u", 1))
    for start in range(0, int(reference.size), chunk_elements):
        stop = min(start + chunk_elements, int(reference.size))
        ref_chunk = reference.flat[start:stop]
        got_chunk = candidate.flat[start:stop]
        ref_hash.update(ref_chunk.tobytes(order="C"))
        got_hash.update(got_chunk.tobytes(order="C"))
        ref_bits = ref_chunk.view(uint_dtype)
        got_bits = got_chunk.view(uint_dtype)
        different = ref_bits != got_bits
        mismatches += int(np.count_nonzero(different))
        if len(examples) < max_examples and np.any(different):
            for local in np.flatnonzero(different)[: max_examples - len(examples)]:
                flat_index = start + int(local)
                coord = tuple(int(i) for i in np.unravel_index(flat_index, reference.shape))
                examples.append({
                    "flat_index": flat_index,
                    "logical_index": list(coord),
                    "reference_bits": f"0x{int(ref_bits[local]):08x}",
                    "candidate_bits": f"0x{int(got_bits[local]):08x}",
                })

    return {
        **base,
        "status": "PASS" if mismatches == 0 else "FAIL",
        "reason": "equal" if mismatches == 0 else "element_bit_mismatch",
        "mismatch_count": mismatches,
        "reference_raw_data_sha256": ref_hash.hexdigest(),
        "candidate_raw_data_sha256": got_hash.hexdigest(),
        "first_mismatches": examples,
        "nonempty": bool(reference.size),
    }


def assert_f32_bits_equal(reference: np.ndarray, candidate: np.ndarray, **kwargs: Any) -> None:
    report = compare_f32_arrays(reference, candidate, **kwargs)
    if report["status"] != "PASS":
        raise AssertionError(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _load_npy(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Only .npy input is supported, not {path.suffix!r}: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--require-nonempty", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.report is not None and args.report.resolve() in {
            args.reference.resolve(), args.candidate.resolve()
        }:
            raise ValueError("Report path must not overwrite either input")
        result = compare_f32_arrays(
            _load_npy(args.reference), _load_npy(args.candidate),
            max_examples=args.max_examples,
            require_nonempty=args.require_nonempty,
        )
        result["reference_file"] = str(args.reference)
        result["candidate_file"] = str(args.candidate)
        text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if result["status"] == "PASS" else 1
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
