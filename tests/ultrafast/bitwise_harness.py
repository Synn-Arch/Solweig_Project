"""Repository-grade raw-bit equality harness for SOLWEIG output planes.

This module extends the design package's ``tools/bitwise_check.py`` semantics
(DESIGN.ko.md 3.3) into the repository test suite. Equality is defined as:

* identical dtype (float32, byte order included),
* identical logical shape,
* identical C-order cell sequence,
* identical per-element 32-bit pattern — so +0.0 vs -0.0, distinct NaN
  payloads/signs, and 1-ULP deltas are all DIFFERENT.

Beyond the element loop, this harness additionally checks what the design
contract calls out explicitly and the standalone checker leaves to callers:

* validity / sentinel masks are compared as their own bit planes (an invalid
  ``-999`` fill must not silently trade places with a NaN or a finite value);
* sparse-patch metadata: global time index and window origin are compared as
  first-class fields — identical bytes with the wrong time index or origin
  still FAIL (this is a metadata comparison, not just bytes).

``np.allclose``, ``np.array_equal(equal_nan=True)`` and tolerance/ULP-distance
gates are never used as pass criteria. ULP distance is computed only to
classify/diagnose reported mismatches.

The module owns its own chunked uint32 comparison (independent code path from
the provided checker) and can cross-check both comparators against each other
(``cross_check_design_comparator``) — self-parity evidence that the repository
harness and the design package agree on every input pair.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DESIGN_TOOLS = (
    Path(__file__).resolve().parents[2]
    / "docs" / "incremental_design_tool" / "ultrafast_bitwise" / "tools"
)

F32 = np.dtype("<f4")
U32 = np.dtype("<u4")
SENTINEL = np.float32(-999.0)
_CHUNK = 1 << 18


# --------------------------------------------------------------------------
# Plane metadata (compared as data, not decoration)
# --------------------------------------------------------------------------
@dataclass
class PlaneMetadata:
    """Identity of a plane beyond its bytes (DESIGN 3.3 sparse-patch items)."""

    plane: str                      # e.g. "utci"
    global_time_index: tuple[int, ...]  # global (scene) time step of each frame
    window_origin: tuple[int, int]  # (row0, col0) of the window in the full grid
    shape: tuple[int, ...] = ()     # informational; arrays carry authoritative shape
    dtype_str: str = "<f4"

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane": self.plane,
            "global_time_index": list(self.global_time_index),
            "window_origin": list(self.window_origin),
            "shape": list(self.shape),
            "dtype_str": self.dtype_str,
        }


def _f32_report_base(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    return {
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
    }


def _ulp_distance_monotonic(bits_u32: np.ndarray) -> np.ndarray:
    """Map float32 bit patterns to a monotonic integer lattice (int64).

    Signed-magnitude float order -> total order: positive floats map to
    ``bits + 2**31`` and negative floats to ``-(int32(bits) + 1)`` so that
    ``-FLT_MAX < ... < -0.0 <= +0.0 < ... < +FLT_MAX`` is strictly
    increasing and adjacent lattice values are 1 ULP apart.

    Diagnostic only: used to label how far apart two finite mismatches are.
    Never a pass criterion.
    """
    as_int = bits_u32.view("<i4").astype(np.int64)
    return np.where(as_int < 0, -as_int - 1, as_int + (1 << 31))


def _classify_pair(ref_bits: int, cand_bits: int) -> str:
    """Best-effort category label for one mismatching element pair."""
    ref_f = np.uint32(ref_bits).view(F32)
    cand_f = np.uint32(cand_bits).view(F32)
    ref_nan, cand_nan = bool(np.isnan(ref_f)), bool(np.isnan(cand_f))
    if ref_nan and cand_nan:
        same_sign = (ref_bits >> 31) == (cand_bits >> 31)
        return "nan_payload" if same_sign else "nan_sign_or_payload"
    if ref_nan or cand_nan:
        return "nan_vs_non_nan"
    if ref_bits in (0x00000000, 0x80000000) and cand_bits in (0x00000000, 0x80000000):
        return "signed_zero"
    if np.isinf(ref_f) or np.isinf(cand_f):
        return "infinity"
    if ref_f == SENTINEL or cand_f == SENTINEL:
        return "sentinel_minus_999"
    dist = abs(int(_ulp_distance_monotonic(np.array([ref_bits], dtype=U32)[0]))
               - int(_ulp_distance_monotonic(np.array([cand_bits], dtype=U32)[0])))
    return f"finite_{dist}_ulp" if dist <= 4 else "finite_gt_4_ulp"


# --------------------------------------------------------------------------
# Core repository comparator
# --------------------------------------------------------------------------
def compare_plane_bits(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    ref_meta: PlaneMetadata | None = None,
    cand_meta: PlaneMetadata | None = None,
    require_nonempty: bool = False,
    max_examples: int = 8,
) -> dict[str, Any]:
    """Strict raw-bit comparison of two float32 planes plus their metadata.

    Returns a report dict; ``status`` is PASS only when dtype, logical shape,
    every element's uint32 pattern, the validity/sentinel mask planes, and
    (when metadata is supplied) the global time index and window origin all
    agree. Metadata is compared even when bytes are identical.
    """
    for name, arr in (("reference", reference), ("candidate", candidate)):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be numpy.ndarray, got {type(arr).__name__}")
        if arr.dtype.kind != "f" or arr.dtype.itemsize != 4:
            raise TypeError(f"{name} must be float32, got {arr.dtype.str}")

    report = _f32_report_base(reference, candidate)

    # --- structural --------------------------------------------------------
    if reference.shape != candidate.shape:
        return {**report, "status": "FAIL", "reason": "shape_mismatch",
                "mismatch_count": None}
    if reference.dtype.str != candidate.dtype.str:
        return {**report, "status": "FAIL", "reason": "dtype_or_byteorder_mismatch",
                "mismatch_count": None}
    if require_nonempty and reference.size == 0:
        return {**report, "status": "FAIL", "reason": "empty_comparison_refused",
                "mismatch_count": None}

    # --- element-wise raw bits (C-order logical sequence; strides ignored) --
    ref_hash = hashlib.sha256()
    cand_hash = hashlib.sha256()
    mismatches = 0
    examples: list[dict[str, Any]] = []
    for start in range(0, int(reference.size), _CHUNK):
        stop = min(start + _CHUNK, int(reference.size))
        ref_chunk = np.ascontiguousarray(reference.flat[start:stop])
        cand_chunk = np.ascontiguousarray(candidate.flat[start:stop])
        ref_hash.update(ref_chunk.tobytes())
        cand_hash.update(cand_chunk.tobytes())
        ref_bits = ref_chunk.view(U32)
        cand_bits = cand_chunk.view(U32)
        diff = ref_bits != cand_bits
        n_here = int(np.count_nonzero(diff))
        mismatches += n_here
        if n_here and len(examples) < max_examples:
            for local in np.flatnonzero(diff)[: max_examples - len(examples)]:
                flat_index = start + int(local)
                coord = tuple(int(i) for i in np.unravel_index(flat_index, reference.shape))
                rb, cb = int(ref_bits[local]), int(cand_bits[local])
                examples.append({
                    "flat_index": flat_index,
                    "logical_index": list(coord),
                    "reference_bits": f"0x{rb:08x}",
                    "candidate_bits": f"0x{cb:08x}",
                    "classification": _classify_pair(rb, cb),
                })
    report.update({
        "mismatch_count": mismatches,
        "reference_raw_data_sha256": ref_hash.hexdigest(),
        "candidate_raw_data_sha256": cand_hash.hexdigest(),
        "first_mismatches": examples,
        "nonempty": bool(reference.size),
    })

    # --- validity / sentinel mask planes (DESIGN 3.3: distinct categories) --
    mask_counts = {}
    mask_defs = {
        "nan_mask": np.isnan,
        "positive_inf_mask": lambda a: a == np.inf,
        "negative_inf_mask": lambda a: a == -np.inf,
        "sentinel_minus_999_mask": lambda a: a == SENTINEL,
    }
    for mask_name, pred in mask_defs.items():
        if reference.size == 0:
            continue
        ref_m = pred(reference)
        cand_m = pred(candidate)
        mask_counts[mask_name] = {
            "reference_count": int(np.count_nonzero(ref_m)),
            "candidate_count": int(np.count_nonzero(cand_m)),
            "mask_bit_mismatch_count": int(np.count_nonzero(ref_m != cand_m)),
        }
    report["mask_comparison"] = mask_counts
    mask_mismatches = sum(m["mask_bit_mismatch_count"] for m in mask_counts.values())
    report["mask_mismatch_count"] = mask_mismatches

    # --- sparse-patch metadata (time index + window origin) ----------------
    meta_result: dict[str, Any] = {"checked": False}
    if ref_meta is not None or cand_meta is not None:
        if ref_meta is None or cand_meta is None:
            meta_result = {"checked": True, "status": "FAIL",
                           "reason": "metadata_missing_on_one_side"}
            mismatches += 1  # metadata failure must fail the comparison
        else:
            diffs = {}
            if tuple(ref_meta.global_time_index) != tuple(cand_meta.global_time_index):
                diffs["global_time_index"] = {
                    "reference": list(ref_meta.global_time_index),
                    "candidate": list(cand_meta.global_time_index)}
            if tuple(ref_meta.window_origin) != tuple(cand_meta.window_origin):
                diffs["window_origin"] = {
                    "reference": list(ref_meta.window_origin),
                    "candidate": list(cand_meta.window_origin)}
            if ref_meta.plane != cand_meta.plane:
                diffs["plane"] = {"reference": ref_meta.plane,
                                  "candidate": cand_meta.plane}
            meta_result = {
                "checked": True,
                "status": "PASS" if not diffs else "FAIL",
                "differences": diffs,
            }
            if diffs:
                mismatches += 1
    report["metadata_comparison"] = meta_result

    ok = mismatches == 0
    report["status"] = "PASS" if ok else "FAIL"
    report["reason"] = (
        "equal" if ok
        else ("metadata_mismatch" if report["metadata_comparison"].get("status") == "FAIL"
              and report["mismatch_count"] == 0
              else "bit_or_mask_mismatch")
    )
    return report


def assert_planes_equal(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    ref_meta: PlaneMetadata | None = None,
    cand_meta: PlaneMetadata | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    report = compare_plane_bits(
        reference, candidate, ref_meta=ref_meta, cand_meta=cand_meta, **kwargs)
    if report["status"] != "PASS":
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    return report


# --------------------------------------------------------------------------
# Bundle comparison (directories of *.f32.npy + sidecar metadata)
# --------------------------------------------------------------------------
@dataclass
class BundleReport:
    all_equal: bool
    planes: dict[str, dict[str, Any]] = field(default_factory=dict)
    plane_set: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_equal": self.all_equal,
            "plane_set": self.plane_set,
            "planes": self.planes,
        }


def load_sidecar_metadata(sample_dir: Path) -> dict[str, dict[str, Any]]:
    """Read plane metadata recorded next to the planes (metadata.json)."""
    sidecar = sample_dir / "metadata.json"
    if not sidecar.is_file():
        return {}
    data = json.loads(sidecar.read_text())
    return {k: v for k, v in data.get("planes", {}).items()}


def _meta_from_sidecar(
    sidecar_entry: dict[str, Any], plane: str, arr: np.ndarray
) -> PlaneMetadata | None:
    if not sidecar_entry:
        return None
    return PlaneMetadata(
        plane=plane,
        global_time_index=tuple(sidecar_entry.get("global_time_index", ())),
        window_origin=tuple(sidecar_entry.get("window_origin", ())),
        shape=tuple(arr.shape),
        dtype_str=str(arr.dtype.str),
    )


def compare_plane_bundles(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    require_nonempty: bool = True,
    max_examples: int = 8,
) -> BundleReport:
    """Compare every ``*.f32.npy`` plane in two sample directories.

    Sidecar ``metadata.json`` files (global time index, window origin) are
    compared per plane when present on either side; a plane with metadata on
    only one side fails. Missing planes on either side fail.
    """
    ref_dir, cand_dir = Path(reference_dir), Path(candidate_dir)
    ref_names = {p.name for p in ref_dir.glob("*.f32.npy")}
    cand_names = {p.name for p in cand_dir.glob("*.f32.npy")}
    set_info = {
        "reference_only": sorted(ref_names - cand_names),
        "candidate_only": sorted(cand_names - ref_names),
    }
    ref_side = load_sidecar_metadata(ref_dir)
    cand_side = load_sidecar_metadata(cand_dir)

    planes: dict[str, dict[str, Any]] = {}
    all_equal = bool(ref_names) and not set_info["reference_only"] \
        and not set_info["candidate_only"]
    for name in sorted(ref_names & cand_names):
        plane = name[: -len(".f32.npy")]
        ref_arr = np.load(ref_dir / name, allow_pickle=False)
        cand_arr = np.load(cand_dir / name, allow_pickle=False)
        ref_meta = _meta_from_sidecar(ref_side.get(plane, {}), plane, ref_arr)
        cand_meta = _meta_from_sidecar(cand_side.get(plane, {}), plane, cand_arr)
        if (ref_meta is None) != (cand_meta is None):
            all_equal = False
            planes[name] = {
                "status": "FAIL",
                "reason": "metadata_missing_on_one_side",
                "mismatch_count": 1,
            }
            continue
        rep = compare_plane_bits(
            ref_arr, cand_arr, ref_meta=ref_meta, cand_meta=cand_meta,
            require_nonempty=require_nonempty, max_examples=max_examples)
        planes[name] = rep
        all_equal &= rep["status"] == "PASS"
    return BundleReport(all_equal=all_equal, planes=planes, plane_set=set_info)


# --------------------------------------------------------------------------
# Self-parity cross-check against the design package comparator
# --------------------------------------------------------------------------
def _load_design_comparator():
    if str(DESIGN_TOOLS) not in sys.path:
        sys.path.insert(0, str(DESIGN_TOOLS))
    from bitwise_check import compare_f32_arrays  # noqa: E402

    return compare_f32_arrays


def cross_check_design_comparator(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    """Run the design package comparator on the same pair.

    Both comparators must agree on the PASS/FAIL verdict; disagreement means
    one of them is broken (self-parity failure).
    """
    ours = compare_plane_bits(reference, candidate)
    theirs = _load_design_comparator()(reference, candidate)
    agree = (ours["status"] == "PASS") == (theirs["status"] == "PASS")
    return {
        "agree": agree,
        "repository_status": ours["status"],
        "design_package_status": theirs["status"],
        "repository_mismatch_count": ours.get("mismatch_count"),
        "design_package_mismatch_count": theirs.get("mismatch_count"),
    }
