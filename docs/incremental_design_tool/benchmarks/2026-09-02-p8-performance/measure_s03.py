"""P8.5 S03 anomaly: post-fix timing of the resize scenario on site_500.

Pre-merge T3 evidence (parts/S03.json): add t1 h6 r3 at (166,102) -> local
128.9 s (dirty 0.0635); update to h12 r5 -> local 917.4 s (dirty 0.1566),
a 7.1x jump for ~3.6x the write area. This script reruns the identical
two-job sequence through the P8 worker and records per-job wall time, mode,
windows, and the march-amplitude saving, to explain the superlinearity.

Hypotheses under test:
  H1 (march tail): pre-merge marches ran to the scene-wide amaxvalue
     (228.34 m ~ 1086 diagonal steps at the 6 deg floor); time-loop cost is
     per-step x window area, and job2's read window (0.85 site vs 0.57)
     amplified it. The window-relative effective_march_amplitude (P8.1)
     cuts the step count to the window's own relief.
  H2 (SVF replay): the Stage-7 vegetation SVF recompute (153 patches) is
     the dominant term and scales with window area; it was paid on both
     jobs but on a larger window in job2.
"""
import json
import sys
import time
from pathlib import Path

WORKTREE = Path("/Users/alansynn/Workspace/solweig/.claude/worktrees/p8-perf")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "tests"))

CACHE_DIR = Path("/tmp/p5sci/cache_alt3")
SITE_DIR = Path("/Users/alansynn/Workspace/solweig/Input_subset/processed_inputs")
DATE_STR = "2009-08-11"


def main() -> None:
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.geometry import RasterGrid, TreeSpec
    from solweig_gpu.incremental.solver import effective_march_amplitude
    from solweig_gpu.incremental.trees import TreeLayer
    from solweig_gpu.incremental.worker import ExactWorker

    report: dict = {"machine": "Apple M1 Pro 10-core/16GB, py3.14, torch 2.13 CPU",
                    "site": "site_500 500x500@2m", "date": DATE_STR}
    cache = SiteCache.load(CACHE_DIR)
    grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                      cache.manifest.origin_x_m, cache.manifest.origin_y_m)

    def cell_x(col):
        return grid.origin_x_m + (col + 0.5) * grid.pixel_size_m

    def cell_y(row):
        return grid.origin_y_m - (row + 0.5) * grid.pixel_size_m

    import shutil
    results_root = Path("/tmp/p8perf/s03_results")
    if results_root.exists():
        shutil.rmtree(results_root)
    layer = TreeLayer(cache.tree_base, grid)
    layer.add_tree(TreeSpec("t1", cell_x(102), cell_y(166), 6.0, 3.0))
    worker = ExactWorker(cache, layer, site_dir=SITE_DIR,
                         results_root=results_root, selected_date_str=DATE_STR)

    jobs = []
    for label, edit in (("add_h6_r3", None), ("resize_h12_r5", "update")):
        if edit == "update":
            layer.update_tree("t1", height_m=12.0, canopy_radius_m=5.0)
        t0 = time.perf_counter()
        outcome = worker.run(job_id=f"s03-{label}")
        wall = time.perf_counter() - t0
        assert outcome.status == "published", outcome
        w = outcome.write_windows
        if w is not None and not hasattr(w, "row_start"):
            w = w[0]  # tuple of windows
        jobs.append({
            "job": label,
            "mode": outcome.mode,
            "wall_seconds": round(wall, 1),
            "dirty_window": ([w.row_start, w.row_stop, w.col_start, w.col_stop]
                             if w is not None else None),
            "diagnostics": {k: v for k, v in (outcome.diagnostics or {}).items()
                            if isinstance(v, (int, float, str, bool))},
        })
        print(label, jobs[-1], flush=True)

    # March-amplitude context for H1: window relief vs scene amaxvalue.
    import numpy as np
    import torch

    a = torch.from_numpy(np.asarray(cache.building_dsm, dtype=np.float32))
    scene_amax = 228.34
    for job in jobs:
        r0, r1, c0, c1 = job["dirty_window"]
        window_a = a[r0:r1, c0:c1]
        amp = effective_march_amplitude(
            window_a, window_a.clone(), window_a.clone(),
            scene_amaxvalue=torch.tensor(scene_amax),
        )
        job["window_effective_amplitude_m"] = round(float(amp), 2)
        job["scene_amaxvalue_m"] = scene_amax
    report["jobs"] = jobs
    report["pre_merge_reference_seconds"] = [128.9, 917.4]
    print(json.dumps(report, indent=2))
    Path("/tmp/p8perf/s03_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
