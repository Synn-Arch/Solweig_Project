# SPDX-License-Identifier: GPL-3.0-only
"""T10 mutation driver: apply one semantic mutation to thermal.py.

Usage: python tests/ultrafast/mutate_thermal.py <id>

The pristine backup + restore discipline is driven by the caller
(runs_mutations.sh): this script ONLY applies the named mutation to the
LIVE file, in place, by exact-string replacement (the same anchor
strings test_thermal.MUTATION_ANCHORS pins — a mutation that cannot
apply is an error, never a silent no-op).
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / (
    "solweig_core/numba_cpu/thermal.py"
)

MUTATIONS: dict[str, tuple[str, str, str]] = {
    # id: (old, new, one-line description)
    "M1": (
        "if fp.get(key) != geometry_fingerprint.get(key):",
        "if False and fp.get(key) != geometry_fingerprint.get(key):",
        "geometry-history gate disabled: previous scene's anchor accepted",
    ),
    "M2": (
        "wanted_met = met_prefix_digest(next_step)",
        "wanted_met = met_prefix_digest(r0)",
        "met digest spelled at r0 instead of the candidate's own next_step",
    ),
    "M3": (
        "start = int(state.next_step)",
        "start = int(state.next_step) + 1",
        "replay resumes one step PAST the anchor (skipped timestep)",
    ),
    "M4": (
        'twater = payload.get("Twater")',
        'twater = payload.get("Twater", 10.0)',
        "absent Twater silently defaulted to 10.0",
    ),
    "M5": (
        "outputs[int(t)] = out",
        "outputs[len(outputs)] = out",
        "t-scatter by position instead of global t",
    ),
    "M6": (
        "if actual != expected_plane_digests[name]:",
        "if False:",
        "torn-state digest fence disabled",
    ),
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in MUTATIONS:
        print(f"usage: {sys.argv[0]} <{'|'.join(MUTATIONS)}>", file=sys.stderr)
        return 2
    old, new, _ = MUTATIONS[sys.argv[1]]
    text = TARGET.read_text()
    count = text.count(old)
    if count != 1:
        print(
            f"anchor for {sys.argv[1]!r} matched {count} times (need 1)",
            file=sys.stderr,
        )
        return 3
    TARGET.write_text(text.replace(old, new))
    print(f"applied {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
