"""T00 repeat self-comparison: raw-bit compare of two run output dirs.

Uses the design package's tools/bitwise_check.py compare_f32_arrays (float32
logical C-order raw bits; signed zeros and NaN payloads count as different).
Exit 0 iff every plane pair is equal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(
    "/Users/alansynn/Workspace/solweig/docs/incremental_design_tool/"
    "ultrafast_bitwise/tools/bitwise_check.py"
)
sys.path.insert(0, str(TOOLS.parent))
from bitwise_check import compare_f32_arrays  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    left, right = Path(args.left).resolve(), Path(args.right).resolve()
    left_names = {p.name for p in left.glob("*.f32.npy")}
    right_names = {p.name for p in right.glob("*.f32.npy")}
    results = {}
    all_equal = bool(left_names) and left_names == right_names
    if left_names != right_names:
        results["_plane_set_mismatch"] = {
            "left_only": sorted(left_names - right_names),
            "right_only": sorted(right_names - left_names),
        }
    for name in sorted(left_names & right_names):
        a = np.load(left / name)
        b = np.load(right / name)
        report = compare_f32_arrays(a, b)
        all_equal &= report.get("status") == "PASS"
        results[name] = report
    total_mismatches = sum(
        r.get("mismatch_count") or 0
        for r in results.values() if isinstance(r, dict)
    )
    payload = {
        "left": str(left),
        "right": str(right),
        "planes": sorted(left_names & right_names),
        "all_equal": all_equal,
        "total_mismatch_count": total_mismatches,
        "per_plane": results,
    }
    Path(args.out).resolve().write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "all_equal": all_equal,
        "total_mismatch_count": total_mismatches,
        "per_plane_status": {
            k: v.get("status") for k, v in results.items() if isinstance(v, dict)
        },
    }, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
