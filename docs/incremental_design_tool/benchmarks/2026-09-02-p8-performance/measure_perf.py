"""P8.2 PERF-001/002: solve_window latency on the typical one-tree job.

Typical fixture per testing_validation.md: one 10 m tree (r=4) at the open
S01 cell (166,102) on site_500, 24 timesteps, exact local solve.
20 warmed reps; p50/p95 target < 20 s / < 60 s.
"""
import cProfile
import io
import pstats
import statistics
import sys
import time
from pathlib import Path

WORKTREE = Path("/Users/alansynn/Workspace/solweig/.claude/worktrees/p8-perf")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "tests"))

import numpy as np
import torch
from solweig_gpu.incremental.cache import SiteCache
from solweig_gpu.incremental.geometry import RasterGrid, TreeSpec
from solweig_gpu.incremental.solver import (
    compose_full_scene_tensors, read_window_for_write_window, solve_window,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import ExactWorker

CACHE_DIR = Path("/tmp/p5sci/cache_alt3")
SITE_DIR = Path("/Users/alansynn/Workspace/solweig/Input_subset/processed_inputs")
DATE_STR = "2009-08-11"
REPS = 20

cache = SiteCache.load(CACHE_DIR)
grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                  cache.manifest.origin_x_m, cache.manifest.origin_y_m)
x = grid.origin_x_m + 102.5 * 2.0
y = grid.origin_y_m - 166.5 * 2.0

layer = TreeLayer(cache.tree_base, grid)
layer.add_tree(TreeSpec("t1", x, y, 10.0, 4.0))
worker = ExactWorker(cache, layer, site_dir=SITE_DIR,
                     results_root=Path("/tmp/p8perf/results"),
                     selected_date_str=DATE_STR)
forcing = worker.forcing()
scene = compose_full_scene_tensors(cache, layer)
dirty = worker.dirty_windows()
write = dirty[0].expand(worker.write_margin_pixels).clamp(rows=grid.rows, cols=grid.cols)
read = read_window_for_write_window(write, cache, scene, forcing)
print(f"threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
print(f"write={write.area/grid.rows/grid.cols:.3f} site  read={read.area/grid.rows/grid.cols:.3f} site")

samples = []
for i in range(REPS + 3):  # 3 warmup
    t0 = time.perf_counter()
    out = solve_window(cache, layer, read_window=read, write_window=write,
                       forcing=forcing, requested_variables=("utci", "tmrt", "shadow"))
    dt = time.perf_counter() - t0
    if i >= 3:
        samples.append(dt)
    print(f"rep {i:2d}: {dt:7.2f}s" + ("  (warmup)" if i < 3 else ""))

samples.sort()
p50 = statistics.median(samples)
p95 = samples[max(0, int(round(0.95 * len(samples))) - 1)]
print(f"\nREPS={len(samples)}  min={samples[0]:.2f}  p50={p50:.2f}  p95={p95:.2f}  max={samples[-1]:.2f}")
print(f"gate PERF-001 p50<20s: {'PASS' if p50 < 20 else 'FAIL'}")
print(f"gate PERF-002 p95<60s: {'PASS' if p95 < 60 else 'FAIL'}")

# one profiled rep for the hot breakdown
prof = cProfile.Profile()
prof.enable()
solve_window(cache, layer, read_window=read, write_window=write,
             forcing=forcing, requested_variables=("utci", "tmrt", "shadow"))
prof.disable()
buf = io.StringIO()
stats = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
stats.print_stats(18)
print(buf.getvalue())
