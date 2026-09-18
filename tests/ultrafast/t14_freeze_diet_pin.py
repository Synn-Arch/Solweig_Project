# SPDX-License-Identifier: GPL-3.0-only
"""Freeze the PRE-CHANGE full-solve digests over the compact capture.

Run once BEFORE the diffsh dense-cube diet (Change F) touches
``radiation.py``; ``test_diffsh_diet.py`` then demands every post-diet
solve reproduce these bit-exactly. Also cross-checks the dev-env digests
against the clean-env cold matrix (same code, same data, two venvs).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from solweig_core.numba_cpu import full_solve as fs  # noqa: E402

CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14/compact_capture")
COLD = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14/gates/clean_env_matrix_cold.json")
OUT = Path(__file__).resolve().parent / "diffsh_diet_pin.json"
PROFILE = "site_500-default"
LATITUDE = 30.312645


def digest(arr) -> str:
    a = np.ascontiguousarray(arr)
    if a.dtype == np.float32:
        a = a.view(np.uint32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def main() -> int:
    cap = fs.CaptureSet(cap_dir=str(CAP), profile=PROFILE, latitude=LATITUDE)
    res = fs.full_solve_capture(cap, profile=PROFILE)

    per_t = []
    for out in res.outputs:
        h = hashlib.sha256()
        for k in sorted(out):
            v = out[k]
            if hasattr(v, "tobytes"):
                h.update(k.encode())
                h.update(digest(v).encode())
        per_t.append(h.hexdigest())

    if COLD.is_file():  # same code, clean venv: must agree field-for-field
        w4 = json.loads(COLD.read_text())["workloads"]["W4_cold_full_solve"]
        bad = [i for i, (a, b) in enumerate(zip(per_t, w4["per_t"])) if a != b]
        assert not bad and len(per_t) == len(w4["per_t"]), f"env drift at {bad}"
        print("dev-env digests == clean-env cold digests: OK")

    pin = {
        "capture": str(CAP),
        "profile": PROFILE,
        "n_timesteps": int(res.n_timesteps),
        "per_t": per_t,
        "ci_series": [
            hashlib.sha256(repr(v).encode()).hexdigest()
            for v in res.ret_CI_series
        ],
        "anchor_fingerprint": res.anchor_fingerprint,
        "note": (
            "T14 Change F pre-change freeze: full_solve_capture over the "
            "compact capture BEFORE the diffsh dense-cube diet. Post-diet "
            "solves must reproduce every field bit-exactly."
        ),
    }
    OUT.write_text(json.dumps(pin, indent=1))
    print(f"pin written: {OUT} ({len(per_t)} timesteps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
