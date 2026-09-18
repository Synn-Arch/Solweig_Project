# SPDX-License-Identifier: GPL-3.0-only
"""T14 clean-environment workload matrix driver (exit gate, card item 6).

Runs INSIDE the torch-free runtime environment (the clean venv with only
the ``solweig-core-cpu`` wheel + numpy/numba/scipy — no torch, no GDAL,
no repo checkout needed). Executable in the dev env too, where torch is
merely importable-but-never-imported (the trap direction: purity is
about the import graph, not about absence from disk).

Workload matrix (each records timings + digests, all raw-bit):

* W1 startup — import cost of the runtime facade.
* W2 unseen-angle step tables (R3) — eight angles including the 225 deg
  branch boundary; sha256 over the table payload.
* W3 routing — a met-edit decision resolves warm/cold; a geometry edit
  refuses typed.
* W4 cold full solve — 24 timesteps over the compact capture; per-plane
  sha256 + threaded-CI digest + phase timings (import/static/first-step)
  + peak RSS.
* W5 warm full solve — anchor warm start from W4's published anchor; bit
  equality of the overlapping outputs.
* W6 unseen-scale oracle leak — unseen-scale table generation stays
  torch-free; amplitude policy for an unseen scene refuses typed.
* W7 legacy backend — typed OracleBackendRefusal.

Output: one JSON line prefixed ``@@MATRIX@@`` (plus a report file when
``--out`` is given).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

_T0 = time.perf_counter()  # process wall clock (perf_counter is arbitrary-
# origin; deltas against _T0 are the process-relative measurements)


def _peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports BYTES (both documented behaviours)
    if platform.system() == "Darwin":
        return round(raw / (1024.0 * 1024.0), 1)
    return round(raw / 1024.0, 1)


def _digest(arr) -> str:
    import numpy as np

    a = np.ascontiguousarray(arr)
    if a.dtype == np.float32:
        a = a.view(np.uint32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _solve_digests(result) -> dict:
    per_t = []
    for out in result.outputs:
        keys = sorted(out)
        h = hashlib.sha256()
        for k in keys:
            v = out[k]
            if hasattr(v, "tobytes"):
                h.update(k.encode())
                h.update(_digest(v).encode())
        per_t.append(h.hexdigest())
    return {
        "n_timesteps": int(result.n_timesteps),
        "per_t": per_t,
        "ci_series": [
            hashlib.sha256(repr(v).encode()).hexdigest()
            for v in result.ret_CI_series
        ],
        "timings_ms": [round(x, 3) for x in result.timings_ms],
        "anchor_fingerprint": result.anchor_fingerprint,
        "anchor_refusal": (
            None
            if result.anchor_refusal is None
            else list(result.anchor_refusal)
        ),
    }


def run(capture_dir: str | None, *, quick: bool = False) -> dict:
    import numpy as np  # noqa: F401

    report: dict = {"workloads": {}, "env": {}}

    # -- environment --------------------------------------------------------
    t0 = time.perf_counter()
    from solweig_core import runtime as rt  # noqa: F401

    report["env"]["import_runtime_s"] = round(time.perf_counter() - t0, 4)
    report["env"]["torch_in_modules"] = "torch" in sys.modules
    report["env"]["solweig_gpu_in_modules"] = any(
        m.startswith("solweig_gpu") for m in sys.modules
    )
    report["env"]["python"] = sys.version.split()[0]
    report["env"]["third_party_tops"] = sorted(
        {
            m.split(".")[0]
            for m in sys.modules
            if m.split(".")[0] not in sys.stdlib_module_names
            and not m.startswith("solweig")
            and not m.startswith("__")
        }
    )

    def _tops_after() -> list[str]:
        return sorted(
            {
                m.split(".")[0]
                for m in sys.modules
                if m.split(".")[0] not in sys.stdlib_module_names
                and not m.startswith("solweig")
                and not m.startswith("__")
            }
        )

    # -- W2: unseen-angle step tables ----------------------------------------
    w2 = {}
    for az, alt in (
        (0.0, 35.0), (90.0, 6.0), (123.4, 45.0), (225.0, 35.0),
        (225.00001, 35.0), (270.0, 78.0), (315.0, 35.0), (359.999, 12.0),
    ):
        table = rt.step_table(
            "svf_shadow", az, alt, 0.5, 40, 80, 20.0
        )
        w2[f"az{az}"] = {
            "count": int(table.count),
            "payload_sha256": _digest(
                np.concatenate(
                    [
                        np.asarray(table.dx, np.int64),
                        np.asarray(table.dy, np.int64),
                        np.asarray(table.dz_bits, np.uint64),
                    ]
                )
            ),
        }
    report["workloads"]["W2_unseen_angle_tables"] = w2

    # -- W3: routing ----------------------------------------------------------
    runtime_obj = rt.Runtime.__new__(rt.Runtime)
    decision = runtime_obj.route(geometry_changed=False, r0=4, n_window=24)
    try:
        runtime_obj.route(geometry_changed=True, r0=0, n_window=24)
        regenerate = "NO-REFUSAL"
    except rt.CaptureRegenerationRefusal as refusal:
        regenerate = type(refusal).__name__
    report["workloads"]["W3_routing"] = {
        "met_edit_route": decision.route,
        "geometry_edit": regenerate,
    }

    # -- W6: unseen-scale + oracle leak ---------------------------------------
    unseen_scale_table = rt.step_table(
        "svf_shadow", 61.0, 30.0, 1.0 / 7.5, 32, 48, 12.0  # scale unseen anywhere
    )
    try:
        runtime_obj.amplitude_policy(a=None, vegdsm=None, vegdsm2=None)
        amp = "NO-REFUSAL"
    except rt.UnseenSceneRefusal as refusal:
        amp = type(refusal).__name__
    try:
        rt.require_runtime_backend("legacy-torch-cpu")
        legacy = "NO-REFUSAL"
    except rt.OracleBackendRefusal as refusal:
        legacy = type(refusal).__name__
    report["workloads"]["W6_unseen_scale"] = {
        "table_count": int(unseen_scale_table.count),
        "table_digest": _digest(
            np.asarray(unseen_scale_table.dz_bits, np.uint64)
        ),
        "amplitude_policy": amp,
        "W7_legacy_backend": legacy,
        "torch_still_absent": "torch" not in sys.modules,
    }

    # W8: process-to-ready proxy — the acceptance metric's shape
    # (process start -> first timestep output available): static load +
    # t00 bundle + one recompute + one fused kernel step, measured in
    # THIS process from _T0 (import cost included). Runs BEFORE W4 so the
    # process-to-first-output number is not polluted by a full solve that
    # happened earlier in the same interpreter.
    if capture_dir:
        from solweig_core.numba_cpu import met_recompute as mr  # noqa: E402
        from solweig_core.numba_cpu.radiation import (
            fused_radiation_timestep,
            rad_bundle_from_capture,
            rad_state_from_capture,
            rad_static_from_capture,
        )

        cap = Path(capture_dir)

        t_ready0 = time.perf_counter()
        st8 = rad_static_from_capture(cap)
        t_static = time.perf_counter()
        state8 = rad_state_from_capture(
            cap, 0, rows=st8.rows, cols=st8.cols
        )
        t_in8 = rad_bundle_from_capture(cap, 0)
        z8 = np.load(cap / "t00.npz")
        site8 = mr.site_params_from_capture(z8, 30.312645)
        tg8 = mr.time_geom_from_capture(z8)
        met8 = mr.met_from_capture(z8)
        res8 = mr.recompute_timestep(site8, met8, tg8, 1.0, None)
        out8, _state8 = fused_radiation_timestep(st8, t_in8, state8)
        t_ready = time.perf_counter()
        report["workloads"]["W8_process_to_ready"] = {
            "static_load_s": round(t_static - t_ready0, 4),
            "first_output_s": round(t_ready - t_static, 4),
            "process_to_first_output_s": round(t_ready - _T0, 4),
            "peak_rss_mb": _peak_rss_mb(),
            "first_output_digest": _digest(
                next(
                    v for v in out8.values()
                    if hasattr(v, "tobytes") and getattr(v, "ndim", 0) == 2
                )
            ),
        }
        # release W8's objects so W4 measures a SINGLE-scenario job peak
        # (two statics resident would double-count the runtime footprint)
        del st8, state8, t_in8, res8, out8, z8, site8, tg8, met8
        import gc

        gc.collect()

    # -- W4/W5: full solve over the compact capture ---------------------------
    if capture_dir and not quick:
        from solweig_core.numba_cpu import full_solve as fs

        cap = Path(capture_dir)
        t0 = time.perf_counter()
        cap_set = fs.CaptureSet(
            cap_dir=str(cap), profile="site_500-default",
            latitude=30.312645,
        )
        static_t = time.perf_counter()
        # mid-window anchor: the warm restart below continues from it
        cold = fs.full_solve_capture(
            cap_set, profile="site_500-default", publish_anchor_at=[11]
        )
        done_t = time.perf_counter()
        cold_digests = _solve_digests(cold)
        cold_digests["phases"] = {
            "capture_set_s": round(static_t - t0, 4),
            "full_solve_s": round(done_t - static_t, 4),
            "first_step_ms": round(cold.timings_ms[0], 3),
            "process_to_first_output_s": round(
                (time.perf_counter() - _T0), 4
            ),
        }
        cold_digests["peak_rss_mb"] = _peak_rss_mb()
        report["workloads"]["W4_cold_full_solve"] = cold_digests

        # W5: warm restart from the MID-WINDOW anchor — reruns steps 12..23;
        # the overlap must be raw-bit equal to the cold run's steps 12..23
        if cold.anchors:
            anchor_state, anchor_fp = cold.anchors[0]
            warm = fs.full_solve_capture(
                cap_set,
                profile="site_500-default",
                warm_state=anchor_state,
                warm_fingerprint=anchor_fp,
                r0=anchor_state.next_step,
            )
            warm_digests = _solve_digests(warm)
            # warm.outputs holds only the RESUMED suffix; n_timesteps is
            # the capture's window length on both results
            offset = len(cold.outputs) - len(warm.outputs)
            overlap_equal = (
                cold_digests["per_t"][offset:] == warm_digests["per_t"]
                and cold_digests["ci_series"][offset:]
                == warm_digests["ci_series"]
            )
            report["workloads"]["W5_warm_full_solve"] = {
                "n_timesteps": warm_digests["n_timesteps"],
                "resumed_from_step": int(anchor_state.next_step),
                "overlap_raw_bit_equal": bool(overlap_equal),
                "first_step_ms": round(warm_digests["timings_ms"][0], 3)
                if warm_digests["timings_ms"] else None,
                "total_s": round(
                    sum(warm_digests["timings_ms"]) / 1e3, 4
                ),
            }
        else:
            report["workloads"]["W5_warm_full_solve"] = {
                "skipped": "no mid-window anchor published"
            }

    report["env"]["third_party_tops_after_workloads"] = _tops_after()
    report["env"]["peak_rss_mb_after_workloads"] = _peak_rss_mb()

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        default="/Users/alansynn/Workspace/solweig_ultrafast_artifacts/"
        "t14/compact_capture",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the W4/W5 full solves (cross-env parity gates compare "
        "W2/W6/W8, which the quick matrix still produces)",
    )
    args = parser.parse_args(argv)
    report = run(
        args.capture if Path(args.capture).is_dir() else None,
        quick=args.quick,
    )
    payload = json.dumps(report, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print("@@MATRIX@@" + payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
