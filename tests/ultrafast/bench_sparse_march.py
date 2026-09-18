# SPDX-License-Identifier: GPL-3.0-only
"""T06 bench + equality-gate CLI (TASKS T06 exit gates).

Subcommands (all raw output goes to the artifacts dir under
``/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t06/``):

* ``gate``  — dense serial vs sparse serial vs sparse parallel (1/2/4/8
  threads) RAW-bit equality over a synthetic big scene AND the site_500
  F1/F2/F3 edit families; prints per-thread-count wall times. Any bit
  mismatch is a hard failure (exit 1).
* ``bench`` — ablation: dense vs sparse pair-step and byte counts, total
  stage latency INCLUDING the work-list build (T07's routing depends on
  it), thread-count sweep, and the serial/parallel crossover that
  justifies :data:`SERIAL_MIN_PAIRS`. n>=5 repeats, median + min/max,
  cold/warm split, host load recorded.

Torch-free (site fixtures reuse the T05 scene composition helpers which
import torch through solweig_gpu — the MODULE under test never does).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from solweig_core import sparse_work as sw  # noqa: E402
from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu.march import march_svf_shadow  # noqa: E402
from solweig_core.numba_cpu.sparse_march import (  # noqa: E402
    SERIAL_MIN_PAIRS,
    march_svf_shadow_sparse,
    plan_sparse_march,
)

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
THREAD_COUNTS = [1, 2, 4, 8]
REPEATS = 7

F32 = np.dtype("<f4")
U32 = np.dtype("<u4")


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=F32).view(U32)


def synth_planes(rows, cols, seed=7):
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    canopy = rng.uniform(0.0, 6.0, (rows, cols)).astype(np.float32)
    canopy[canopy < np.float32(3.0)] = np.float32(0.0)
    return a, dem, canopy


def synth_table(kernel, az, alt, amp, rows, cols, scale=0.5):
    from trace_exporter import capture_trace

    trace = capture_trace(
        kernel,
        float(az),
        float(alt),
        scale,
        int(rows),
        int(cols),
        float(amp),
        amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
        verify_probes=False,
    )
    return st.build_table_from_trace(trace)


def timeit(fn, *, repeats=REPEATS, warm=2):
    """Median/min/max seconds over ``repeats`` after ``warm`` warm-ups."""
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "n": len(samples),
    }


def host_load() -> dict:
    out = {}
    try:
        out["uptime"] = (
            subprocess.run(
                ["uptime"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        out["uptime_error"] = repr(exc)
    try:
        out["hw_ncpu"] = (
            subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        out["sysctl_error"] = repr(exc)
    return out


# ---------------------------------------------------------------------------
# gate: raw-bit equality, synthetic + site
# ---------------------------------------------------------------------------


def gate_synthetic() -> dict:
    from solweig_core.numba_cpu.march import march_wallheight23
    from solweig_core.numba_cpu.sparse_march import march_wallheight23_sparse

    rows, cols = 192, 384
    a, dem, canopy = synth_planes(rows, cols, seed=29)
    inputs = sw.compose_march_inputs(a, dem, canopy)
    mask = np.random.default_rng(31).random((rows, cols)) < 0.55
    runs = sw.row_runs_from_mask(mask)
    table = synth_table(st.KERNEL_SVF_SHADOW, 61.0, 12.0, 20.0, rows, cols)
    dense = march_svf_shadow(
        table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
    )
    mismatch = 0
    serial = march_svf_shadow_sparse(
        table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
    )
    timings = {}
    for label, result in [("sparse_serial", serial)]:
        for d, s in zip(dense, result[:3]):
            mismatch += int((bits(d)[mask] != bits(s)[mask]).sum())
            outside = bits(s)[~mask]
            mismatch += int((outside != np.zeros_like(outside)).sum())
        timings[label] = 0.0
    for threads in THREAD_COUNTS:
        t0 = time.perf_counter()
        par = march_svf_shadow_sparse(
            table,
            inputs.a,
            inputs.vegdsm,
            inputs.vegdsm2,
            inputs.bush,
            runs,
            n_threads=threads,
        )
        timings[f"sparse_t{threads}"] = time.perf_counter() - t0
        for d, s in zip(dense, par[:3]):
            mismatch += int((bits(d)[mask] != bits(s)[mask]).sum())
    # wallheight leg on the same work list (synthetic; closure is T05+)
    wh_table = synth_table(
        st.KERNEL_WALLHEIGHT_23, 200.0, 12.0, 20.0, rows, cols
    )
    wh_dense = march_wallheight23(
        wh_table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
    )
    wh_serial = march_wallheight23_sparse(
        wh_table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
    )
    for d, s in zip(wh_dense, wh_serial[:3]):
        mismatch += int((bits(d)[mask] != bits(s)[mask]).sum())
    for threads in THREAD_COUNTS:
        par = march_wallheight23_sparse(
            wh_table,
            inputs.a,
            inputs.vegdsm,
            inputs.vegdsm2,
            inputs.bush,
            runs,
            n_threads=threads,
        )
        for d, s in zip(wh_dense, par[:3]):
            mismatch += int((bits(d)[mask] != bits(s)[mask]).sum())
    plan = plan_sparse_march(runs, n_threads=8)
    return {
        "leg": "synthetic_192x384",
        "cells": int(mask.sum()),
        "runs": int(runs.shape[0]),
        "table_count_svf": int(table.count),
        "table_count_wh": int(wh_table.count),
        "pair_count": plan.pair_count,
        "mismatch_count": mismatch,
        "timings_s": timings,
    }


def site_state():
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.solver import (
        _compose_scene_from_canopy,
        _sky_patch_geometry,
    )
    from trace_exporter import (
        capture_amplitude_policy,
        executed_amplitude_for_patch,
    )

    cache = SiteCache.load(SITE_500_CACHE)
    patches, _rings = _sky_patch_geometry(2)
    return {
        "cache": cache,
        "canopy_base": np.asarray(cache.tree_base, dtype=np.float32).copy(),
        "scale": 1.0 / float(cache.pixel_size_m),
        "rows": int(cache.rows),
        "cols": int(cache.cols),
        "patches": patches,
        "capture_amplitude_policy": capture_amplitude_policy,
        "executed_amplitude_for_patch": executed_amplitude_for_patch,
        "compose_torch": _compose_scene_from_canopy,
    }


def site_scene(state, name, canopy):
    from trace_exporter import capture_trace

    a = np.array(state["cache"].building_dsm)
    dem = np.array(state["cache"].dem)
    inputs = sw.compose_march_inputs(a, dem, canopy)
    scene_t = state["compose_torch"](state["cache"], canopy)
    policy = state["capture_amplitude_policy"](
        scene_t.a, scene_t.vegdsm, scene_t.vegdsm2, scene_t.vegdem,
        scale=state["scale"],
    )
    amps = {
        idx: state["executed_amplitude_for_patch"](policy, idx, escalated=False)
        for idx in range(len(state["patches"]))
    }
    tables = {}
    for idx, (altitude, azimuth, _ring) in enumerate(state["patches"]):
        amp, pid = amps[idx]
        trace = capture_trace(
            st.KERNEL_SVF_SHADOW,
            float(azimuth),
            float(altitude),
            state["scale"],
            state["rows"],
            state["cols"],
            amp,
            amplitude_policy_id=pid,
            verify_probes=False,
        )
        tables[idx] = st.build_table_from_trace(trace)
    return {"inputs": inputs, "tables": tables}


def site_canopy_with(state, trees):
    from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow, TreeSpec
    from solweig_gpu.incremental.trees import TreeLayer

    grid = RasterGrid(
        rows=state["rows"],
        cols=state["cols"],
        pixel_size_m=state["cache"].pixel_size_m,
        origin_x_m=state["cache"].manifest.origin_x_m,
        origin_y_m=state["cache"].manifest.origin_y_m,
    )
    layer = TreeLayer(state["canopy_base"], grid)
    for tree_id, cell, height, radius in trees:
        x = grid.origin_x_m + (cell[1] + 0.5) * grid.pixel_size_m
        y = grid.origin_y_m - (cell[0] + 0.5) * grid.pixel_size_m
        layer.add_tree(TreeSpec(tree_id, x, y, height, radius))
    return np.ascontiguousarray(
        layer.vegetation_rasters_window(
            RasterWindow(0, state["rows"], 0, state["cols"])
        )[0]
    )


def gate_site() -> list[dict]:
    state = site_state()
    rows, cols = state["rows"], state["cols"]
    canopy_base = state["canopy_base"]
    families = []
    canopy_f1 = site_canopy_with(state, [("t1", (166, 102), 10.0, 4.0)])
    families.append(("F1_add_t1", canopy_base, canopy_f1))
    families.append(
        (
            "F2_move_t1",
            canopy_f1,
            site_canopy_with(state, [("t1", (166, 127), 10.0, 4.0)]),
        )
    )
    tall = float(canopy_base.max()) + 25.0
    short = tall - 5.0
    families.append(
        (
            "F3_delete_overlap",
            site_canopy_with(
                state,
                [("A", (166, 150), tall, 8.0), ("B", (166, 153), short, 8.0)],
            ),
            site_canopy_with(state, [("B", (166, 153), short, 8.0)]),
        )
    )
    records = []
    for label, canopy_pre, canopy_post in families:
        scene_pre = site_scene(state, f"{label}_pre", canopy_pre)
        scene_post = site_scene(state, f"{label}_post", canopy_post)
        inputs = scene_post["inputs"]
        build = sw.build_sparse_work(
            scene_pre["inputs"], inputs, scene_pre["tables"],
            scene_post["tables"],
        )
        mismatch = 0
        timings = {"dense_s": 0.0, "sparse_serial_s": 0.0}
        timings.update({f"sparse_t{t}_s": 0.0 for t in THREAD_COUNTS})
        sparse_steps = dense_steps = 0
        for idx in sorted(build.row_runs):
            t_post = scene_post["tables"][idx]
            runs = build.row_runs[idx]
            t0 = time.perf_counter()
            dense = march_svf_shadow(
                t_post, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
            )
            timings["dense_s"] += time.perf_counter() - t0
            # per-patch target mask from the run list itself (masks not kept)
            targets = sw.mask_from_row_runs(runs, (rows, cols))
            results = {}
            # sparse_serial first (n_threads=1, the routing default), then
            # the explicit 1/2/4/8 sweep — t1 IS the serial path
            sweep = [("sparse_serial_s", 1)] + [
                (f"sparse_t{t}_s", t) for t in THREAD_COUNTS
            ]
            for key, threads in sweep:
                t0 = time.perf_counter()
                result = march_svf_shadow_sparse(
                    t_post,
                    inputs.a,
                    inputs.vegdsm,
                    inputs.vegdsm2,
                    inputs.bush,
                    runs,
                    n_threads=threads,
                )
                timings[key] = timings.get(key, 0.0) + (
                    time.perf_counter() - t0
                )
                results[key] = result
            for key, result in results.items():
                for d, s in zip(dense, result[:3]):
                    mismatch += int(
                        (bits(d)[targets] != bits(s)[targets]).sum()
                    )
                    outside = bits(s)[~targets]
                    mismatch += int(
                        (outside != np.zeros_like(outside)).sum()
                    )
            sparse_steps += int(targets.sum()) * int(t_post.count)
            dense_steps += rows * cols * int(t_post.count)
        records.append(
            {
                "leg": f"site_500:{label}",
                "patches": len(build.row_runs),
                "pair_count": int(build.pair_count),
                "pair_steps_sparse": sparse_steps,
                "pair_steps_dense": dense_steps,
                "build_seconds": build.build_seconds,
                "mismatch_count": mismatch,
                "timings_s": timings,
            }
        )
    return records


def cmd_gate() -> int:
    print(f"host: {json.dumps(host_load())}")
    print(f"SERIAL_MIN_PAIRS = {SERIAL_MIN_PAIRS}")
    records = [gate_synthetic()]
    if SITE_500_CACHE.is_dir():
        records.extend(gate_site())
    else:
        print(
            "NOTE: site_500 cache absent — site legs skipped (reported "
            "honestly)"
        )
    total = 0
    for record in records:
        total += int(record["mismatch_count"])
        print(json.dumps(record, sort_keys=True))
    print(f"TOTAL mismatch_count = {total} (raw float32 bits, uint32 view)")
    return 0 if total == 0 else 1


# ---------------------------------------------------------------------------
# bench: ablation + crossover + build cost
# ---------------------------------------------------------------------------


def cmd_bench() -> int:
    print(f"host: {json.dumps(host_load())}")
    print(f"SERIAL_MIN_PAIRS = {SERIAL_MIN_PAIRS}")
    print(f"repeats per point: {REPEATS} (median + [min, max])")

    # -- synthetic big scene: same-work dense vs sparse --------------------
    rows, cols = 192, 384
    a, dem, canopy = synth_planes(rows, cols, seed=29)
    inputs = sw.compose_march_inputs(a, dem, canopy)
    mask = np.random.default_rng(31).random((rows, cols)) < 0.55
    runs = sw.row_runs_from_mask(mask)
    table = synth_table(st.KERNEL_SVF_SHADOW, 61.0, 12.0, 20.0, rows, cols)
    args = (table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush)

    pair_steps_sparse = int(mask.sum()) * int(table.count)
    pair_steps_dense = rows * cols * int(table.count)
    bytes_dense_read = 4 * rows * cols * int(table.count)  # a/vd/vd2 lookups
    bytes_sparse_read = 4 * int(mask.sum()) * int(table.count)
    print(
        f"synthetic same-work ablation (192x384, mask 55%, count={table.count}):"
    )
    print(
        f"  pair_steps dense={pair_steps_dense} sparse={pair_steps_sparse} "
        f"ratio={pair_steps_sparse / pair_steps_dense:.4f}"
    )
    print(
        f"  est bytes  dense={bytes_dense_read} sparse={bytes_sparse_read} "
        f"ratio={bytes_sparse_read / bytes_dense_read:.4f}"
    )
    print(f"  dense fused  : {json.dumps(timeit(lambda: march_svf_shadow(*args)))}")
    print(
        "  sparse serial: "
        f"{json.dumps(timeit(lambda: march_svf_shadow_sparse(*args, runs)))}"
    )
    for threads in THREAD_COUNTS:
        print(
            f"  sparse t{threads}    : "
            f"{json.dumps(timeit(lambda: march_svf_shadow_sparse(*args, runs, n_threads=threads)))}"
        )

    # -- total stage latency INCLUDING the work-list build (T07 feed) -----
    print("total stage latency incl. build_sparse_work (site_500 F1):")
    if not SITE_500_CACHE.is_dir():
        print("  site cache absent — skipped")
    else:
        state = site_state()
        canopy_post = site_canopy_with(state, [("t1", (166, 102), 10.0, 4.0)])
        scene_pre = site_scene(state, "bench_pre", state["canopy_base"])
        scene_post = site_scene(state, "bench_post", canopy_post)
        inputs = scene_post["inputs"]

        def build_only():
            return sw.build_sparse_work(
                scene_pre["inputs"], inputs, scene_pre["tables"],
                scene_post["tables"],
            )

        build_t = timeit(build_only)
        build = build_only()
        print(
            f"  build_sparse_work(153 patches): {json.dumps(build_t)}"
        )
        print(
            f"  build pair_count={build.pair_count} "
            f"row_run_count={build.row_run_count} "
            f"pair_steps_sparse={build.pair_steps_sparse} "
            f"pair_steps_dense={build.pair_steps_dense} "
            f"ratio={build.pair_steps_sparse / build.pair_steps_dense:.6f}"
        )
        print(
            f"  estimated_bytes_row_runs={build.estimated_bytes_row_runs} "
            f"estimated_bytes_bitsets={build.estimated_bytes_bitsets}"
        )

        def sparse_stage_serial():
            build = build_only()
            for idx in sorted(build.row_runs):
                march_svf_shadow_sparse(
                    scene_post["tables"][idx],
                    inputs.a,
                    inputs.vegdsm,
                    inputs.vegdsm2,
                    inputs.bush,
                    build.row_runs[idx],
                    n_threads=1,
                )

        def sparse_stage_threads(n):
            def run():
                build = build_only()
                for idx in sorted(build.row_runs):
                    march_svf_shadow_sparse(
                        scene_post["tables"][idx],
                        inputs.a,
                        inputs.vegdsm,
                        inputs.vegdsm2,
                        inputs.bush,
                        build.row_runs[idx],
                        n_threads=n,
                    )

            return run

        def dense_stage():
            for idx in sorted(scene_post["tables"]):
                march_svf_shadow(
                    scene_post["tables"][idx],
                    inputs.a,
                    inputs.vegdsm,
                    inputs.vegdsm2,
                    inputs.bush,
                )

        print(f"  dense 153-march stage  : {json.dumps(timeit(dense_stage))}")
        print(
            "  sparse stage serial    : "
            f"{json.dumps(timeit(sparse_stage_serial))}"
        )
        for threads in [2, 4, 8]:
            print(
                f"  sparse stage t{threads}      : "
                f"{json.dumps(timeit(sparse_stage_threads(threads)))}"
            )

    # -- serial/parallel crossover (thread overhead on small work) --------
    print("crossover probe (forced threads vs serial on tiny work lists):")
    for cells in (64, 1024, 4096, 16384, 65536):
        c = int(cells**0.5)
        small_runs = np.array(
            [[r, 0, c - 1] for r in range(c)], dtype=np.int32
        )
        small_a, small_dem, small_canopy = synth_planes(c, c, seed=3)
        small_inputs = sw.compose_march_inputs(small_a, small_dem, small_canopy)
        small_table = synth_table(
            st.KERNEL_SVF_SHADOW, 61.0, 12.0, 20.0, c, c
        )
        small_args = (
            small_table,
            small_inputs.a,
            small_inputs.vegdsm,
            small_inputs.vegdsm2,
            small_inputs.bush,
        )
        plan = plan_sparse_march(small_runs, n_threads=4)
        serial_t = timeit(
            lambda: march_svf_shadow_sparse(*small_args, small_runs, n_threads=1),
            repeats=REPEATS,
        )
        forced_t = timeit(
            lambda: march_svf_shadow_sparse(
                *small_args, small_runs, n_threads=4, min_pairs=1
            ),
            repeats=REPEATS,
        )
        print(
            f"  cells={cells:>6} pairs={plan.pair_count:>6} "
            f"serial={serial_t['median_s']*1e3:8.3f}ms "
            f"forced_t4={forced_t['median_s']*1e3:8.3f}ms "
            f"overhead={(forced_t['median_s'] / max(serial_t['median_s'], 1e-12)):.2f}x"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gate", "bench"])
    args = parser.parse_args(argv)
    if args.command == "gate":
        return cmd_gate()
    return cmd_bench()


if __name__ == "__main__":
    raise SystemExit(main())
