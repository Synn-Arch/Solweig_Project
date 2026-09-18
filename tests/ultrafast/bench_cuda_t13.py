# SPDX-License-Identifier: GPL-3.0-only
"""T13 benchmark — end-to-end residency/graphs/dispatch INCLUDING
transfer and synchronization (TASKS T13 exit criteria).

Every number below is WALL CLOCK over the full path (staging + H2D +
kernel + D2H + sync + host copy-out); kernel-only references from T12's
stage bench are cited for context. Legs:

  A. rad_day host-staged  — the T12 wrapper (202 MB H2D per call, the
     26 ms that dominated T12's 43.1 ms E2E).
  B. rad_day resident     — cold bind (static residency) then steady
     timesteps: full-download and 40-row-edit modes.
  C. rad_day graph-on     — first replay pays the capture+instantiate
     build; subsequent replays amortize (same scalar digest — the
     edit-recompute pattern; per-timestep scalar changes rebuild by
     design, see REPORT limitations).
  D. rad_night            — wrapper vs resident steady.
  E. utci sparse          — plain launch vs capacity-bucket graph.
  F. dispatch             — tiny edit (CPU must win on measurement) and
     full-site rad_day (CPU fused numba vs resident GPU E2E).
  G. leak/stress          — 200 no-download + 200 full-download cycles:
     allocation-count and device-memory drift.

Run on the GPU host:
  SW_REQUIRE_CUDA=1 CUDA_VISIBLE_DEVICES=<idle> SW_T12_T08_CAPTURE=... \
  ~/solweig_ultrafast/py tests/ultrafast/bench_cuda_t13.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))
from sw_cuda_harness import load_host_module, require_canonical_runtime  # noqa: E402

REPO_ROOT = ULTRA_DIR.parents[1]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"
sys.path.insert(0, str(REPO_ROOT))

_T13 = {}


def load_t13(name: str):
    if name in _T13:
        return _T13[name]
    host = load_host_module()
    sys.modules.setdefault("sw_cuda_host", host)
    reg = f"sw_cuda_{name}"
    mod = sys.modules.get(reg)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            reg, NATIVE_CUDA / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[reg] = mod
        spec.loader.exec_module(mod)
    _T13[name] = mod
    return mod


from test_cuda_residency import (  # noqa: E402
    RAD_DAY_NEXT,
    RAD_DAY_OUTPUTS,
    RAD_DAY_STATIC,
    RAD_DAY_VOLATILE,
    RAD_NIGHT_OUTPUTS,
    RAD_NIGHT_STATIC,
    RAD_NIGHT_VOLATILE,
)

DAY_OUT = RAD_DAY_OUTPUTS + RAD_DAY_NEXT


def _site(cap: Path, t: int):
    from solweig_core.numba_cpu.radiation import (
        rad_bundle_from_capture,
        rad_state_from_capture,
        rad_static_from_capture,
    )
    from sw_rad_bundle import build_rad_day_bundle, build_rad_night_bundle

    st = rad_static_from_capture(cap)
    t_in = rad_bundle_from_capture(cap, t)
    state = rad_state_from_capture(cap, t, rows=st.rows, cols=st.cols)
    if t_in.is_day:
        return st, t_in, state, build_rad_day_bundle(st, t_in, state)
    return st, t_in, state, build_rad_night_bundle(st, t_in)


def _wall(fn, n: int, warmup: int = 1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def _edit(bundle: dict, r0: int, nrows: int = 40) -> dict:
    edited = {}
    for k in ("shadow", "Tg_plane", "m_tg_in"):
        a = bundle[k].copy()
        a[r0:r0 + nrows] += np.float32(0.25)
        edited[k] = a
    return edited


def main() -> int:
    cap = Path(os.environ.get(
        "SW_T12_T08_CAPTURE",
        os.path.expanduser("~/solweig_ultrafast/work/t12/fixtures/"
                           "t08_capture")))
    assert cap.is_dir(), f"capture fixtures missing: {cap}"
    host = load_host_module()
    rt = require_canonical_runtime()
    residency = load_t13("residency")
    graphs = load_t13("graphs")
    dispatch_cost = load_t13("dispatch_cost")

    gpu_name = rt.device_name()
    load_avg = subprocess.run(["cat", "/proc/loadavg"], capture_output=True,
                              text=True).stdout.strip()
    res = {
        "host": subprocess.run(["hostname"], capture_output=True,
                               text=True).stdout.strip(),
        "gpu": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "loadavg": load_avg,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capture": str(cap),
    }

    st, t_in, state, day = _site(cap, 12)
    _, _, _, night = _site(cap, 5)
    rows, cols = st.rows, st.cols
    res["site"] = {"rows": int(rows), "cols": int(cols)}

    # -- A. host-staged wrapper (T12 baseline path) --------------------------
    res["A_rad_day_wrapper_e2e_ms"] = _wall(
        lambda: host.run_rad_day(rt, day), n=5, warmup=1) * 1e3

    # -- B. resident runner ---------------------------------------------------
    t0 = time.perf_counter()
    runner = residency.ResidentRadRunner(rt, chunk_rows=128)
    runner.bind({k: day[k] for k in RAD_DAY_STATIC}, static=True)
    runner.bind({k: day[k] for k in RAD_DAY_VOLATILE})
    res["B_cold_bind_s"] = time.perf_counter() - t0
    res["B_cold_bind_h2d_bytes"] = runner.h2d_bytes
    lease = runner.run_day(day, download=DAY_OUT)
    lease.wait()
    lease.publish()

    res["B_rad_day_resident_full_download_e2e_ms"] = _wall(
        lambda: (lambda l: (l.wait(), l.publish()))(
            runner.run_day(day, download=DAY_OUT)),
        n=10, warmup=1) * 1e3

    def edit_cycle(i: int, use_graph: bool):
        edited = _edit(day, (i * 37) % (rows - 40))
        runner.bind(edited)
        b2 = dict(day)
        b2.update(edited)
        lease = runner.run_day(b2, download=DAY_OUT, use_graph=use_graph)
        lease.wait()
        lease.publish()

    h2d_before = runner.h2d_bytes
    t0 = time.perf_counter()
    for i in range(10):
        edit_cycle(i, use_graph=False)
    res["B_rad_day_edit_mode_e2e_ms"] = (time.perf_counter() - t0) / 10 * 1e3
    res["B_edit_mode_h2d_bytes_per_iter"] = (runner.h2d_bytes -
                                             h2d_before) // 10

    # -- C. graph-on (same scalar digest: edit-recompute pattern) ------------
    runner2 = residency.ResidentRadRunner(rt, chunk_rows=128)
    runner2.bind({k: day[k] for k in RAD_DAY_STATIC}, static=True)
    runner2.bind({k: day[k] for k in RAD_DAY_VOLATILE})
    t0 = time.perf_counter()
    lease = runner2.run_day(day, download=DAY_OUT, use_graph=True)
    lease.wait()
    lease.publish()
    res["C_graph_build_plus_first_e2e_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for i in range(10):
        edited = _edit(day, (i * 37) % (rows - 40))
        runner2.bind(edited)
        b2 = dict(day)
        b2.update(edited)
        lease = runner2.run_day(b2, download=DAY_OUT, use_graph=True)
        lease.wait()
        lease.publish()
    res["C_graph_edit_mode_e2e_ms"] = (time.perf_counter() - t0) / 10 * 1e3
    res["C_graph_stats"] = runner2.graph_stats()
    runner2.close()

    # replay steady-state wall: 20 enqueued replays, one final sync —
    # per-replay stream-throughput incl. kernel (enqueue alone is µs)
    runner3 = residency.ResidentRadRunner(rt, chunk_rows=128)
    runner3.bind({k: day[k] for k in RAD_DAY_STATIC}, static=True)
    runner3.bind({k: day[k] for k in RAD_DAY_VOLATILE})
    lease = runner3.run_day(day, download=DAY_OUT, use_graph=True)
    lease.wait()
    lease.publish()
    t0 = time.perf_counter()
    for _ in range(20):
        runner3.run_day(day, download=(), use_graph=True)
    runner3.sync()
    res["C_graph_replay_wall_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    runner3.close()

    # -- D. rad_night ----------------------------------------------------------
    res["D_rad_night_wrapper_e2e_ms"] = _wall(
        lambda: host.run_rad_night(rt, night), n=10, warmup=1) * 1e3
    rn = residency.ResidentRadRunner(rt, chunk_rows=128)
    rn.bind({k: night[k] for k in RAD_NIGHT_STATIC}, static=True)
    rn.bind({k: night[k] for k in RAD_NIGHT_VOLATILE})
    lease = rn.run_night(night, download=RAD_NIGHT_OUTPUTS)
    lease.wait()
    lease.publish()
    res["D_rad_night_resident_e2e_ms"] = _wall(
        lambda: (lambda l: (l.wait(), l.publish()))(
            rn.run_night(night, download=RAD_NIGHT_OUTPUTS)),
        n=20, warmup=1) * 1e3
    rn.close()

    # -- E. utci sparse: plain vs bucket graph ---------------------------------
    rng = np.random.default_rng(0xBEEF)
    n = 250_000
    vecs = tuple(np.ascontiguousarray(
        rng.uniform(0.1, 40.0, n).astype(np.float32)) for _ in range(4))
    res["E_utci_plain_e2e_ms"] = _wall(
        lambda: host.run_utci_sparse(rt, *vecs), n=10, warmup=1) * 1e3
    t0 = time.perf_counter()
    pool = graphs.UtciGraphPool(rt)
    pool.run(*vecs)
    res["E_utci_graph_build_first_ms"] = (time.perf_counter() - t0) * 1e3
    res["E_utci_graph_e2e_ms"] = _wall(
        lambda: pool.run(*vecs), n=20, warmup=1) * 1e3
    res["E_utci_graph_stats"] = pool.stats()
    pool.destroy()

    # -- F. dispatch demo --------------------------------------------------------
    d = dispatch_cost.Dispatcher(rt)
    tiny = np.arange(16, dtype=np.float32)

    def cpu_tiny():
        return float(np.add(tiny, tiny).sum())

    def gpu_tiny():
        add, _s, _m, _dv = host.run_arith(rt, tiny, tiny)
        return float(np.asarray(add).sum())

    costs_tiny = d.measure("tiny-edit-16", cpu_tiny, gpu_tiny)
    res["F_tiny_edit"] = {
        "costs": costs_tiny,
        "selected": d.select("tiny-edit-16"),
        "cpu_cost_ms": costs_tiny["cpu"] * 1e3,
        "cuda_cost_ms": costs_tiny["cuda"] * 1e3,
    }

    d2 = dispatch_cost.Dispatcher(rt)

    def cpu_site():
        from solweig_core.numba_cpu.radiation import fused_radiation_timestep
        out, _next = fused_radiation_timestep(st, t_in, state)
        return out["Tmrt"].sum(dtype=np.float64)

    def gpu_site():
        lease = runner.run_day(day, download=("tmrt",))
        lease.wait()
        return float(lease.publish()["tmrt"].sum(dtype=np.float64))

    costs_site = d2.measure("rad-day-site", cpu_site, gpu_site)
    res["F_rad_day_site"] = {
        "costs": costs_site,
        "selected": d2.select("rad-day-site"),
        "cpu_cost_ms": costs_site["cpu"] * 1e3,
        "cuda_cost_ms": costs_site["cuda"] * 1e3,
    }

    # -- G. leak / stress ---------------------------------------------------------
    base_alloc = residency.device_alloc_count()
    base_free, _total = residency.mem_info()
    t0 = time.perf_counter()
    for _i in range(200):  # kernel-only cycles (no D2H)
        runner.run_day(day, download=())
    runner.sync()
    for _i in range(200):  # full-download cycles
        lease = runner.run_day(day, download=RAD_DAY_OUTPUTS)
        lease.wait()
        lease.publish()
    runner.sync()
    res["G_stress_400_cycles_s"] = time.perf_counter() - t0
    res["G_alloc_delta"] = residency.device_alloc_count() - base_alloc
    free2, _t2 = residency.mem_info()
    res["G_free_drift_bytes"] = int(base_free - free2)
    runner.close()

    out = Path(os.environ.get("SW_T13_BENCH_OUT",
                              "bench_cuda_t13.json"))
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
