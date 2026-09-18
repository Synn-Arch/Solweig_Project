# SPDX-License-Identifier: GPL-3.0-only
"""T12 — per-stage CUDA benchmarks: kernel-only (CUDA events), H2D/D2H,
and end-to-end at site_500 scale (TASKS T12; DESIGN 6.x/9.x).

Timing legs per stage (a SECONDARY metric — never traded for bits):

* ``kernel_only`` — CUDA-event time (sw_timer_begin/sw_timer_end) of the
  ABI launch over PRE-UPLOADED device buffers; reps, min + median
  reported. This is pure device execution: no memcpy, no prefix sum, no
  Python argument marshalling.
* ``h2d`` / ``d2h`` — wall-clock of the full input upload / output
  download set (blocking cudaMemcpy, exactly what the host wrappers do).
* ``e2e`` — wall-clock of the complete host wrapper call (H2D + launch
  + D2H), i.e. the caller-visible cost.

Stage inputs (all site_500 = 500x500):

* march_svf_shadow / march_wallheight23 — the frozen site march pins
  (real oracle-fed planes + real T03 step tables), first march timestep;
* fold — the frozen T07 tables + packed cubes / planes sized to the
  site (fold cost is data-independent: fixed 153-patch x annulus loop,
  bit values never change the op count);
* rad_day (capture t12) / rad_night (capture t05) — the t08 capture
  bundles via the same loader the gate-4 tests use (requires
  SW_T12_T08_CAPTURE);
* utci_dense / utci_sparse — physical-domain synthetic planes (~95%
  valid), matching the gate-5 differential domains.

Usage (GPU host):
    python tests/ultrafast/bench_cuda_stages.py --out bench.json \
        [--reps-k 50] [--reps-io 10]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ULTRA_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(ULTRA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from sw_cuda_harness import f32_from_hex, load_host_module  # noqa: E402

NATIVE_CUDA = REPO_ROOT / "native" / "cuda"


def stats(ms: list[float]) -> dict:
    return {"reps": len(ms), "min_ms": round(min(ms), 4),
            "median_ms": round(statistics.median(ms), 4),
            "mean_ms": round(statistics.fmean(ms), 4)}


def kernel_only(rt, launch, reps: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        launch()
    rt.sync()
    times = []
    for _ in range(reps):
        rt.timer_begin()
        launch()
        times.append(rt.timer_end())
    return stats(times)


def wall(rt, fn, reps: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
    rt.sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    rt.sync()
    return stats(times)


def h2d(rt, stage, reps: int) -> dict:
    """Wall-clock of the full input upload set (the exact memcpys the
    host wrappers issue — blocking cudaMemcpy each)."""
    def upload():
        for d in upload.keep:
            d.free()
        upload.keep = [rt.to_device(np.ascontiguousarray(a, dtype=dt))
                       for a, dt in zip(stage["arrays"], stage["dtypes"])]
    upload.keep = []
    res = wall(rt, upload, reps, warmup=1)
    for d in upload.keep:
        d.free()
    return res


# ---------------------------------------------------------------------------
# stage builders: each returns dict(name, shape, inputs(list[np]),
# outs(list[np shapes/dtypes]), launch(rt, devs, outs), e2e(rt))
# ---------------------------------------------------------------------------


def build_march(kernel: str):
    site = json.loads(
        (NATIVE_CUDA / "data" / "march_site_pins.json").read_text())["site"]
    shape = (site["rows"], site["cols"])
    t_first = sorted(site["per_t"], key=int)[0]
    rec = site["per_t"][t_first]
    planes = [f32_from_hex(site[k]).reshape(shape).copy()
              for k in ("a", "vegdsm", "vegdsm2")]
    cols = [np.array(rec["dx"], dtype=np.int32),
            np.array(rec["dy"], dtype=np.int32),
            f32_from_hex(rec["dz"])]
    if kernel == "wallheight23":
        cols.append(f32_from_hex(rec["dzprev"]))
        abi = "sw_run_march_wallheight23"
    else:
        abi = "sw_run_march_svf_shadow"
    rows, cols_n = shape

    def launch(rt, devs, outs):
        rc = rt.lib[abi](*[ctypes.c_void_p(d.ptr.value) for d in devs],
                         *[ctypes.c_void_p(o.ptr.value) for o in outs],
                         ctypes.c_int(rows), ctypes.c_int(cols_n),
                         ctypes.c_int(len(rec["dx"])))
        if rc != 0:
            raise RuntimeError(f"{abi} rc={rc}")

    def e2e(rt):
        if kernel == "wallheight23":
            return rt_host().run_march_wallheight23(
                rt, *planes, *cols)
        return rt_host().run_march_svf_shadow(rt, *planes, *cols)

    return {
        "name": f"march_{kernel}_site500",
        "shape": list(shape),
        "arrays": [*planes, *cols],
        "dtypes": [np.float32] * 3 + [np.int32, np.int32, np.float32]
        + ([np.float32] if kernel == "wallheight23" else []),
        "n_out": 3,
        "launch": launch,
        "e2e": e2e,
        "e2e_n_out": 3,
    }


def build_fold(rows: int = 500, cols: int = 500):
    pins = json.loads((NATIVE_CUDA / "data" / "fold_pins.json").read_text())
    t = pins["tables"]
    rng = np.random.default_rng(0x501DF01D)
    veg = rng.integers(0, 256, (rows, cols, 20)).astype(np.uint8)
    vbsh = rng.integers(0, 256, (rows, cols, 20)).astype(np.uint8)
    vegdem2 = rng.uniform(0.0, 18.0, (rows, cols)).astype(np.float32)
    svfb = rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
    tables = {
        "w_iso": f32_from_hex(t["w_iso"]).reshape(8, 12),
        "w_aniso": f32_from_hex(t["w_aniso"]).reshape(8, 12),
        "ring": np.array(t["ring"], dtype=np.int32),
        "na": np.array(t["na"], dtype=np.int32),
        "dir_e": np.array(t["dir_e"], dtype=np.int8),
        "dir_s": np.array(t["dir_s"], dtype=np.int8),
        "dir_w": np.array(t["dir_w"], dtype=np.int8),
        "dir_n": np.array(t["dir_n"], dtype=np.int8),
        "last_const": f32_from_hex([t["last_bits"]])[0],
        "one_minus_trans": f32_from_hex([t["one_minus_trans_bits"]])[0],
    }
    arrays = [veg, vbsh, vegdem2, svfb, tables["w_iso"], tables["w_aniso"],
              tables["ring"], tables["na"], tables["dir_e"], tables["dir_s"],
              tables["dir_w"], tables["dir_n"]]
    dtypes = [np.uint8, np.uint8, np.float32, np.float32, np.float32,
              np.float32, np.int32, np.int32, np.int8, np.int8, np.int8,
              np.int8]

    def launch(rt, devs, outs):
        rc = rt.lib.sw_run_fold(
            *[ctypes.c_void_p(d.ptr.value) for d in devs],
            ctypes.c_void_p(None),
            *[ctypes.c_void_p(o.ptr.value) for o in outs],
            ctypes.c_float(tables["last_const"]),
            ctypes.c_float(tables["one_minus_trans"]),
            ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(0))
        if rc != 0:
            raise RuntimeError(f"sw_run_fold rc={rc}")

    def e2e(rt):
        return rt_host().run_fold(rt, veg, vbsh, vegdem2, svfb, tables)

    return {"name": "fold_site500", "shape": [rows, cols],
            "arrays": arrays, "dtypes": dtypes, "n_out": 1,
            "launch": launch, "e2e": e2e, "e2e_n_out": 1,
            "out_shape": (rows, cols, 11)}


def build_rad(mode: str):
    cap = os.environ.get("SW_T12_T08_CAPTURE", "")
    assert cap, "SW_T12_T08_CAPTURE must point at the t08 capture fixtures"
    from solweig_core.numba_cpu.radiation import (
        rad_bundle_from_capture, rad_state_from_capture,
        rad_static_from_capture,
    )
    from sw_rad_bundle import build_rad_day_bundle, build_rad_night_bundle

    cap_path = Path(cap)
    st = rad_static_from_capture(cap_path)
    t_in = rad_bundle_from_capture(cap_path, 12 if mode == "day" else 5)
    if mode == "day":
        state = rad_state_from_capture(cap_path, 12, rows=st.rows,
                                       cols=st.cols)
        b = build_rad_day_bundle(st, t_in, state)
        input_spec = rt_host()._RAD_DAY_INPUTS  # noqa: SLF001
        scalar_spec = rt_host()._RAD_DAY_SCALAR_ARGS  # noqa: SLF001
        n_out = 25  # 19 returns + 6 next-state
        abi = "sw_run_rad_day"
    else:
        b = build_rad_night_bundle(st, t_in)
        input_spec = rt_host()._RAD_NIGHT_INPUTS  # noqa: SLF001
        scalar_spec = [("veg64", "d"), ("shd64", "d")]
        n_out = 7
        abi = "sw_run_rad_night"

    arrays = [np.ascontiguousarray(b[k], dtype=dt) for k, dt in input_spec]

    def scalars():
        args = []
        for key, kind in scalar_spec:
            if kind == "f":
                args.append(ctypes.c_float(np.float32(b[key])))
            elif kind == "d":
                args.append(ctypes.c_double(np.float64(b[key])))
            else:
                args.append(ctypes.c_int(int(b[key])))
        return args

    rows, cols = int(b["rows"]), int(b["cols"])

    def launch(rt, devs, outs):
        rc = rt.lib[abi](*[ctypes.c_void_p(d.ptr.value) for d in devs],
                         *[ctypes.c_void_p(o.ptr.value) for o in outs],
                         *scalars())
        if rc != 0:
            raise RuntimeError(f"{abi} rc={rc}")

    def e2e(rt):
        return (rt_host().run_rad_day(rt, b) if mode == "day"
                else rt_host().run_rad_night(rt, b))

    return {"name": f"rad_{mode}_site500", "shape": [rows, cols],
            "arrays": arrays, "dtypes": [dt for _, dt in input_spec],
            "n_out": n_out, "launch": launch, "e2e": e2e,
            "e2e_n_out": n_out, "out_shape": (rows, cols)}


def build_utci(rows: int = 500, cols: int = 500, valid_frac: float = 0.95):
    rng = np.random.default_rng(0xBEEF)
    ta = rng.uniform(-5.0, 38.0, (rows, cols))
    rh = rng.uniform(20.0, 95.0, (rows, cols))
    tmrt = rng.uniform(10.0, 75.0, (rows, cols))
    va = rng.uniform(0.3, 10.0, (rows, cols))
    m = rng.random((rows, cols)) > valid_frac
    for arr in (ta, rh, tmrt, va):
        arr[m] = -999.0
    planes = [np.ascontiguousarray(a, dtype=np.float32)
              for a in (ta, rh, tmrt, va)]
    n = int((~m).sum())
    neg = np.float32(-999.0)
    valid = np.logical_and.reduce(
        [p > neg for p in planes])
    counts_arr = valid.sum(axis=1).astype(np.int64)
    offs_arr = np.zeros(rows + 1, dtype=np.int64)
    offs_arr[1:] = np.cumsum(counts_arr)

    def prep_outs(rt, outs):
        counts, offs, _ = outs
        counts.from_numpy(counts_arr)
        offs.from_numpy(offs_arr)

    def dense_kernel_only(rt, devs, outs):
        counts, offs, out = outs
        rc = rt.lib.sw_utci_count(
            *[ctypes.c_void_p(d.ptr.value) for d in devs],
            ctypes.c_void_p(counts.ptr.value),
            ctypes.c_int(rows), ctypes.c_int(cols))
        if rc != 0:
            raise RuntimeError(f"sw_utci_count rc={rc}")
        rc = rt.lib.sw_utci_fill(
            *[ctypes.c_void_p(d.ptr.value) for d in devs],
            ctypes.c_void_p(offs.ptr.value),
            ctypes.c_void_p(out.ptr.value),
            ctypes.c_longlong(n), ctypes.c_int(8), ctypes.c_int(32768),
            ctypes.c_int(rows), ctypes.c_int(cols))
        if rc != 0:
            raise RuntimeError(f"sw_utci_fill rc={rc}")

    def dense_e2e(rt):
        return rt_host().run_utci_dense(rt, *planes)

    return {"name": "utci_dense_site500", "shape": [rows, cols],
            "arrays": planes, "dtypes": [np.float32] * 4,
            "n_out": 3, "launch": dense_kernel_only, "e2e": dense_e2e,
            "e2e_n_out": 1, "out_shape": (rows, cols),
            "extra_outs": [((rows,), np.int64), ((rows + 1,), np.int64),
                           ((rows, cols), np.float32)],
            "prep_outs": prep_outs,
            "meta": {"valid_lanes": n, "valid_frac": valid_frac}}


def build_utci_sparse(n: int = 237500):
    rng = np.random.default_rng(0x5BAE)
    vecs = [np.ascontiguousarray(
        rng.uniform(lo, hi, n), dtype=np.float32)
        for lo, hi in ((-5.0, 38.0), (20.0, 95.0), (10.0, 75.0),
                       (0.3, 10.0))]

    def launch(rt, devs, outs):
        rc = rt.lib.sw_utci_sparse(
            *[ctypes.c_void_p(d.ptr.value) for d in devs],
            ctypes.c_void_p(outs[0].ptr.value),
            ctypes.c_longlong(n), ctypes.c_int(8), ctypes.c_int(32768))
        if rc != 0:
            raise RuntimeError(f"sw_utci_sparse rc={rc}")

    def e2e(rt):
        return rt_host().run_utci_sparse(rt, *vecs)

    return {"name": f"utci_sparse_n{n}", "shape": [n],
            "arrays": vecs, "dtypes": [np.float32] * 4, "n_out": 1,
            "launch": launch, "e2e": e2e, "e2e_n_out": 1,
            "out_shape": (n,)}


_HOST = None


def rt_host():
    global _HOST
    if _HOST is None:
        _HOST = load_host_module()
    return _HOST


def bench_stage(rt, stage, reps_k: int, reps_io: int) -> dict:
    host = rt_host()
    # upload once for the kernel-only leg
    devs = [rt.to_device(np.ascontiguousarray(a, dtype=dt))
            for a, dt in zip(stage["arrays"], stage["dtypes"])]
    extra = stage.get("extra_outs")
    if extra:
        outs = [rt.alloc(s, dt) for s, dt in extra]
        prep = stage.get("prep_outs")
        if prep is not None:
            prep(rt, outs)
    else:
        oshape = stage.get("out_shape", tuple(stage["shape"]))
        odt = np.float32
        outs = [rt.alloc(oshape, odt) for _ in range(stage["n_out"])]
    kern = kernel_only(rt, lambda: stage["launch"](rt, devs, outs), reps_k)
    for d in devs + outs:
        d.free()

    # H2D: fresh uploads, wall clock
    up = h2d(rt, stage, reps_io)

    # D2H: the wrapper's output set, wall clock
    oshape = stage.get("out_shape", tuple(stage["shape"]))
    outs_h = [rt.alloc(oshape, np.float32)
              for _ in range(stage.get("e2e_n_out", stage["n_out"]))]
    down = wall(rt, lambda: [o.to_numpy() for o in outs_h], reps_io,
                warmup=1)
    for o in outs_h:
        o.free()

    # E2E: full wrapper (fresh H2D + launch + D2H inside)
    e2e = wall(rt, lambda: stage["e2e"](rt), reps_io, warmup=1)

    res = {"stage": stage["name"], "shape": stage["shape"],
           "kernel_only": kern, "h2d": up, "d2h": down, "e2e": e2e}
    if "meta" in stage:
        res["meta"] = stage["meta"]
    res["inputs_bytes"] = int(sum(a.nbytes for a in stage["arrays"]))
    res["outputs_bytes"] = int(np.prod(oshape) * 4
                               * stage.get("e2e_n_out", stage["n_out"]))
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="bench_cuda_stages.json")
    parser.add_argument("--reps-k", type=int, default=50)
    parser.add_argument("--reps-io", type=int, default=10)
    parser.add_argument("--stage", action="append", default=None)
    args = parser.parse_args()

    host = rt_host()
    from sw_cuda_harness import require_canonical_runtime
    rt = require_canonical_runtime()
    if rt is None:
        print("no CUDA runtime — bench must run on the GPU host",
              file=sys.stderr)
        return 2
    rt.assert_canonical_strict()

    stages = [build_march("svf_shadow"), build_march("wallheight23"),
              build_fold(), build_utci(), build_utci_sparse()]
    try:
        stages.append(build_rad("day"))
        stages.append(build_rad("night"))
    except AssertionError as exc:
        print(f"[skip] radiation stages: {exc}", file=sys.stderr)

    sel = stages
    if args.stage:
        want = set(args.stage)
        sel = [s for s in stages if s["name"] in want]
        missing = want - {s["name"] for s in sel}
        if missing:
            parser.error(f"unknown stages: {sorted(missing)}")

    results = {"device": rt.device_name(),
               "library": str(rt.lib_path),
               "build_flags": rt.build_flags,
               "gpu_ordinal": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
               "stages": []}
    for st in sel:
        print(f"bench {st['name']} …", file=sys.stderr)
        r = bench_stage(rt, st, args.reps_k, args.reps_io)
        results["stages"].append(r)
        print(json.dumps(r), file=sys.stderr)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
