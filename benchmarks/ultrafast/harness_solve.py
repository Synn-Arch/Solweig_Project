"""T00 baseline harness: full-domain and incremental exact solves, timed.

Import origin is pinned to the CURRENT WORKING DIRECTORY (sys.path.insert
before the first solweig_gpu import) so an oracle worktree run can never
silently pick up the editable-installed main tree. The resolved origin and
sha256 of the key scientific sources are recorded in every run record.

Run from the tree whose code should execute:
    cd <tree> && python <repo>/benchmarks/ultrafast/harness_solve.py \
        --cache-dir <tree>/site-cache/site_500 --site-dir <tree>/Input_subset/processed_inputs \
        --mode {full,edit} [--case E1|E2|E3|E4] [--warmup N --samples M] \
        --out-dir <artifacts>/<run_id> --record-out <artifacts>/raw_records.jsonl

Outputs (utci/tmrt/shadow) are saved as raw .f32.npy planes under --out-dir
for later bitwise comparison. All timings use time.perf_counter (monotonic).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_import_origin_to_cwd() -> None:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def import_origin() -> dict:
    import solweig_gpu

    pkg_dir = Path(solweig_gpu.__file__).resolve().parent
    key_files = {
        "shadow.py": pkg_dir / "shadow.py",
        "solweig.py": pkg_dir / "solweig.py",
        "utci_process.py": pkg_dir / "utci_process.py",
        "incremental/solver.py": pkg_dir / "incremental" / "solver.py",
        "incremental/veg_svf_state.py": pkg_dir / "incremental" / "veg_svf_state.py",
    }
    sources = {
        rel: sha256_file(p) for rel, p in key_files.items() if p.exists()
    }
    import numpy
    import torch

    return {
        "solweig_gpu_file": str(Path(solweig_gpu.__file__).resolve()),
        "cwd": cwd_resolved(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "source_sha256": sources,
    }


def cwd_resolved() -> str:
    return str(Path(os.getcwd()).resolve())


# --- edit cases (cell coords are row, col on the site_500 grid) -------------
S01_CELL = (166, 102)  # open cell per testing_validation.md
T1 = {"id": "t1", "cell": S01_CELL, "height_m": 10.0, "radius_m": 4.0}
MOVE_CELL = (166, 127)  # 50 m east of S01
T2_CELL = (166, 407)  # exactly 610 m east of S1 (305 pixels * 2 m)
DATE_STR = "2009-08-11"


def build_case(layer, grid, case: str) -> str:
    """Apply the case's edits to a fresh TreeLayer; return a spec string."""
    from solweig_gpu.incremental.geometry import TreeSpec

    def cell_xy(cell):
        return (
            grid.origin_x_m + (cell[1] + 0.5) * grid.pixel_size_m,
            grid.origin_y_m - (cell[0] + 0.5) * grid.pixel_size_m,
        )

    x1, y1 = cell_xy(T1["cell"])
    t1 = TreeSpec(T1["id"], x1, y1, T1["height_m"], T1["radius_m"])
    layer.add_tree(t1)
    if case == "E1":
        return "add t1 h10 r4 at (166,102)"
    if case == "E2":
        xb, yb = cell_xy(MOVE_CELL)
        layer.move_tree(T1["id"], x_m=xb, y_m=yb)
        return f"add t1 (166,102) then move to {MOVE_CELL} (50 m east)"
    if case == "E3":
        layer.update_tree(T1["id"], height_m=15.0)
        return "add t1 h10 r4 then height replace -> 15.0 m"
    if case == "E4":
        x2, y2 = cell_xy(T2_CELL)
        layer.add_tree(TreeSpec("t2", x2, y2, 10.0, 4.0))
        return "add t1 (166,102) + add t2 (166,407) 610 m east"
    raise ValueError(f"unknown case {case!r}")


def one_solve(cache, layer, grid, worker, write_window, run_out: Path):
    """Time admission-independent solve stages; save planes; return record."""
    import numpy as np
    from solweig_gpu.incremental.solver import (
        compose_full_scene_tensors,
        read_window_for_write_window,
        solve_window,
    )

    t = {}
    t0 = time.perf_counter()
    forcing = worker.forcing()
    t["forcing_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scene = compose_full_scene_tensors(cache, layer)
    t["compose_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    read = read_window_for_write_window(write_window, cache, scene, forcing)
    t["read_window_s"] = time.perf_counter() - t0

    stage_timings: dict[str, float] = {}
    t0 = time.perf_counter()
    outputs = solve_window(
        cache,
        layer,
        read_window=read,
        write_window=write_window,
        forcing=forcing,
        requested_variables=("utci", "tmrt", "shadow"),
        stage_timings=stage_timings,
        scene=scene,
    )
    t["solve_s"] = time.perf_counter() - t0
    t.update(stage_timings)

    run_out.mkdir(parents=True, exist_ok=True)
    plane_sha = {}
    for var, arr in outputs.items():
        path = run_out / f"{var}.f32.npy"
        np.save(path, np.ascontiguousarray(arr))
        plane_sha[var] = {
            "sha256": sha256_file(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": path.stat().st_size,
        }
    t["total_s"] = sum(
        t.get(k, 0.0) for k in ("forcing_s", "compose_s", "read_window_s", "solve_s")
    )
    return t, plane_sha, {"rows": read.height, "cols": read.width}, {
        "rows": write_window.height, "cols": write_window.width,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "edit"], required=True)
    ap.add_argument("--case", choices=["E1", "E2", "E3", "E4"], default="E1")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--site-dir", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--record-out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    pin_import_origin_to_cwd()
    origin = import_origin()

    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
    from solweig_gpu.incremental.trees import TreeLayer
    from solweig_gpu.incremental.worker import ExactWorker

    cache_dir = Path(args.cache_dir).resolve()
    cache = SiteCache.load(cache_dir)
    grid = RasterGrid(
        cache.rows, cache.cols, cache.pixel_size_m,
        cache.manifest.origin_x_m, cache.manifest.origin_y_m,
    )
    out_dir = Path(args.out_dir).resolve()

    records = []
    if args.mode == "full":
        layer = TreeLayer(cache.tree_base, grid)
        worker = ExactWorker(
            cache, layer, site_dir=Path(args.site_dir).resolve(),
            results_root=Path(args.results_root).resolve(),
            selected_date_str=DATE_STR,
        )
        write = RasterWindow(0, cache.rows, 0, cache.cols)
        timings, planes, read_ext, write_ext = one_solve(
            cache, layer, grid, worker, write, out_dir / "sample0",
        )
        records.append({
            "run_id": args.run_id, "mode": "full", "case": None,
            "phase": "cold_sample", "rep": 0,
            "timings_s": timings, "planes": planes,
            "read_window": read_ext, "write_window": write_ext,
        })
    else:
        reps = [("warmup", i) for i in range(args.warmup)] + [
            ("sample", i) for i in range(args.samples)
        ]
        for phase, i in reps:
            # fresh layer per rep: identical cold edit batch every time
            layer = TreeLayer(cache.tree_base, grid)
            spec = build_case(layer, grid, args.case)
            worker = ExactWorker(
                cache, layer, site_dir=Path(args.site_dir).resolve(),
                results_root=Path(args.results_root).resolve(),
                selected_date_str=DATE_STR,
            )
            t0 = time.perf_counter()
            dirty = worker.dirty_windows()
            admission_s = time.perf_counter() - t0
            write = dirty[0]
            for w in dirty[1:]:
                write = write.union(w)
            write = write.expand(worker.write_margin_pixels).clamp(
                rows=grid.rows, cols=grid.cols,
            )
            timings, planes, read_ext, write_ext = one_solve(
                cache, layer, grid, worker, write,
                out_dir / f"{phase}{i}",
            )
            timings["admission_s"] = admission_s
            timings["total_s"] += admission_s
            records.append({
                "run_id": args.run_id, "mode": "edit", "case": args.case,
                "edit_spec": spec, "phase": phase, "rep": i,
                "timings_s": timings, "planes": planes,
                "read_window": read_ext, "write_window": write_ext,
            })

    manifest_sha = sha256_file(cache_dir / "manifest.json")
    payload = {
        "origin": origin,
        "cache_dir": str(cache_dir),
        "cache_manifest_sha256": manifest_sha,
        "site_dir": str(Path(args.site_dir).resolve()),
        "date": DATE_STR,
        "grid": {
            "rows": cache.rows, "cols": cache.cols,
            "pixel_size_m": cache.pixel_size_m,
            "origin_x_m": cache.manifest.origin_x_m,
            "origin_y_m": cache.manifest.origin_y_m,
        },
        "records": records,
    }
    (out_dir / "run_record.json").write_text(json.dumps(payload, indent=2))
    rec_path = Path(args.record_out).resolve()
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rec_path, "a") as fh:
        for r in records:
            fh.write(json.dumps({
                "run_id": r["run_id"], "mode": r["mode"], "case": r.get("case"),
                "phase": r["phase"], "rep": r["rep"],
                "timings_s": r["timings_s"],
                "origin": origin["solweig_gpu_file"],
                "write_window": r.get("write_window"),
                "plane_sha256": {k: v["sha256"] for k, v in r["planes"].items()},
            }) + "\n")
    for r in records:
        t = r["timings_s"]
        print(
            f"{r['run_id']} {r['mode']}/{r.get('case') or ''} {r['phase']}{r['rep']}: "
            f"total={t.get('total_s', 0):.3f}s admission={t.get('admission_s', 0):.3f}s "
            f"compose={t.get('compose_s', 0):.3f}s svf={t.get('svf_seconds', 0):.3f}s "
            f"time_loop={t.get('time_loop_seconds', 0):.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
