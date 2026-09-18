# SPDX-License-Identifier: GPL-3.0-only
"""T14a startup bench — process-to-ready of the torch-free CPU stack
before/after the UTCI-family disk-cache fix (R8).

Mirrors the t14 dependency-audit driver (``t14_prep/t14_jit_driver.py``)
so numbers are comparable: import every torch-free module, first-call
each kernel (marches, svf fold, utci dense/sparse/met, fused radiation
on the real t08 capture), and total process-to-ready as
``import + first-call JIT loads`` (p2r_jit_s) plus the same including
capture loads (p2r_full_s).

Two scenarios, each with an isolated ``NUMBA_CACHE_DIR``:

- cold: a fresh cache dir per run (first-compile path);
- warm: one prewarm run, then measured runs on the same dir (the
  shipped-cache / prebuilt-site deployment path).

Timing is n>=3 runs, medians reported; the JSON artifact lands in
``$ART/t14a/timing/startup_timing.json``. Wall numbers are host-load
dependent (audit host carried load 6-7); cold-vs-warm STRUCTURE is what
matters.

Run (repo root):  .venv/bin/python tests/ultrafast/bench_startup_t14a.py
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = Path(os.environ.get(
    "T14A_ART", "/Users/alansynn/Workspace/solweig_ultrafast_artifacts"))
TRACES = ART / "t03/traces"
CAPTURE = ART / "t08/capture"

# the audit driver's exact tables (same inputs -> comparable walls)
TRACE_SVF = "svf_shadow_az0x00000000_alt0x420c0000_s0x3f000000_40x80_A0x41a00000.json"
TRACE_W23 = "wallheight_23_az0x00000000_alt0x420c0000_s0x3f000000_40x80_A0x41a00000.json"

RUNNER = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["T14A_ROOT"])
import numpy as np

t = {}
def tick(key):
    t0 = time.perf_counter()
    t["_tick"] = t0
def tock(key):
    t[key] = time.perf_counter() - t["_tick"]

tick("import_stack")
from solweig_core import step_tables as st
from solweig_core import bitplanes as bp
from solweig_core.numba_cpu import march, sparse_march, svf_fold, utci
from solweig_core.numba_cpu import radiation as rad
tock("import_stack")

rng = np.random.default_rng(0)
R, C = 40, 80
a = rng.uniform(0, 20, (R, C)).astype(np.float32)
veg = (a * 0.3).astype(np.float32); veg2 = (a * 0.1).astype(np.float32)
bush = np.zeros((R, C), dtype=np.float32)

traces = os.environ.get("T14A_TRACES")
if traces:
    trace = json.load(open(os.path.join(traces, os.environ["T14A_TRACE_SVF"])))
    tab = st.build_table_from_trace(trace)
    tick("march_svf_shadow"); march.march_svf_shadow(tab, a, veg, veg2, bush); tock("march_svf_shadow")
    tr_w = json.load(open(os.path.join(traces, os.environ["T14A_TRACE_W23"])))
    tab_w = st.build_table_from_trace(tr_w)
    tick("march_wallheight23"); march.march_wallheight23(tab_w, a, veg, veg2, bush); tock("march_wallheight23")

rows = cols = 64
veg_p = bp.pack_bits(rng.integers(0, 2, (rows, cols, 153)).astype(np.bool_))
vbsh_p = bp.pack_bits(rng.integers(0, 2, (rows, cols, 153)).astype(np.bool_))
vegdem2 = rng.uniform(0, 5, (rows, cols)).astype(np.float32)
svfb = rng.uniform(0.5, 1.0, (rows, cols)).astype(np.float32)
tick("fold_svf"); svf_fold.fold_svf(veg_p, vbsh_p, vegdem2, svfb); tock("fold_svf")

ta = rng.uniform(-10, 40, (rows, cols)).astype(np.float32)
rh = rng.uniform(0, 100, (rows, cols)).astype(np.float32)
tmrt = rng.uniform(-10, 70, (rows, cols)).astype(np.float32)
va = rng.uniform(0, 17, (rows, cols)).astype(np.float32)
tick("utci_dense"); d = utci.utci_calculator_dense(ta, rh, tmrt, va); tock("utci_dense")
tick("utci_sparse"); s = utci.utci_calculator_sparse(ta.ravel()[:512], rh.ravel()[:512], tmrt.ravel()[:512], va.ravel()[:512]); tock("utci_sparse")
tick("utci_met_step"); m = utci.utci_met_step(tmrt, 21.3, 55.0, 3.1); tock("utci_met_step")

cap = os.environ.get("T14A_CAPTURE")
if cap:
    tick("load_static"); stat = rad.rad_static_from_capture(cap); tock("load_static")
    tick("load_bundle_t8"); bundle = rad.rad_bundle_from_capture(cap, 8); tock("load_bundle_t8")
    tick("load_state_t0"); state0 = rad.rad_state_from_capture(cap, 0, rows=stat.rows, cols=stat.cols); tock("load_state_t0")
    tick("fused_first"); rad.fused_radiation_timestep(stat, bundle, state0); tock("fused_first")

p2r_jit = t["import_stack"] + sum(
    v for k, v in t.items() if k in (
        "march_svf_shadow", "march_wallheight23", "fold_svf",
        "utci_dense", "utci_sparse", "utci_met_step", "fused_first"))
t.pop("_tick", None)
t["p2r_jit_s"] = p2r_jit
t["p2r_full_s"] = p2r_jit + sum(
    v for k, v in t.items() if k.startswith("load_"))
t["torch_in_modules"] = "torch" in sys.modules
print("@@BENCH@@" + json.dumps(t))
'''


def run_once(cache_dir: Path) -> dict:
    env = dict(os.environ)
    env["T14A_ROOT"] = os.environ.get("T14A_ROOT", str(REPO_ROOT))
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["T14A_TRACES"] = str(TRACES) if TRACES.is_dir() else ""
    env["T14A_CAPTURE"] = str(CAPTURE) if CAPTURE.is_dir() else ""
    env["T14A_TRACE_SVF"] = TRACE_SVF
    env["T14A_TRACE_W23"] = TRACE_W23
    proc = subprocess.run([sys.executable, "-c", RUNNER], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@BENCH@@")]
    assert line, proc.stdout + proc.stderr
    return json.loads(line[0][len("@@BENCH@@"):])


def main() -> None:
    import tempfile

    n = int(os.environ.get("T14A_RUNS", "3"))
    out_dir = ART / "t14a" / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "task": "T14a",
        "driver": "tests/ultrafast/bench_startup_t14a.py",
        "comparable_audit_driver": "t14_prep/t14_jit_driver.py",
        "repo": os.environ.get("T14A_ROOT", str(REPO_ROOT)),
        "repo_note": ("T14A_ROOT override = pristine pre-T14a tree copy for "
                      "the BEFORE arm" if os.environ.get("T14A_ROOT")
                      else "worktree (post-T14a)"),
        "runs_per_scenario": n,
        "env": {
            "python": sys.version.split()[0],
            "argv0": sys.executable,
            "loadavg_start": os.getloadavg(),
            "capture_used": CAPTURE.is_dir(),
            "traces_used": TRACES.is_dir(),
        },
    }
    try:
        import numba
        import llvmlite
        report["env"].update(numba=numba.__version__,
                             llvmlite=llvmlite.__version__,
                             numpy=__import__("numpy").__version__)
    except Exception:  # pragma: no cover
        pass

    scenarios: dict[str, list[dict]] = {"cold": [], "warm": []}
    with tempfile.TemporaryDirectory(prefix="t14a_bench_") as td:
        td = Path(td)
        for i in range(n):
            scenarios["cold"].append(run_once(td / f"cold{i}"))
        warm_dir = td / "warm"
        run_once(warm_dir)  # prewarm
        for i in range(n):
            scenarios["warm"].append(run_once(warm_dir))

    med: dict[str, dict[str, float]] = {}
    for name, rows_ in scenarios.items():
        med[name] = {}
        for key in rows_[0]:
            vals = [r[key] for r in rows_ if isinstance(r.get(key), float)]
            if vals:
                med[name][key] = statistics.median(vals)
        for r in rows_:
            assert not r["torch_in_modules"], "torch leaked into the stack"
    report["scenarios"] = scenarios
    report["medians_s"] = med

    out = out_dir / f"startup_timing{os.environ.get('T14A_TAG', '')}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"medians_s": med, "artifact": str(out)}, indent=2))


if __name__ == "__main__":
    main()
