"""P8.1 routing measurement: per-tree offsets + exact dirty fraction (site_500).

Uses the WORKTREE code (P8 branch) against the corrected real-site cache.
"""
import sys
from pathlib import Path

WORKTREE = Path("/Users/alansynn/Workspace/solweig/.claude/worktrees/p8-perf")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "tests"))

import numpy as np
from solweig_gpu.incremental.cache import SiteCache
from solweig_gpu.incremental.geometry import RasterGrid, TreeSpec, choose_recompute_mode
from solweig_gpu.incremental.solver import (
    compose_full_scene_tensors, load_site_forcing, read_window_for_write_window,
    required_halo_pixels,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import ExactWorker

CACHE_DIR = Path("/tmp/p5sci/cache_alt3")
SITE_DIR = Path("/Users/alansynn/Workspace/solweig/Input_subset/processed_inputs")
DATE_STR = "2009-08-11"

cache = SiteCache.load(CACHE_DIR)
grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                  cache.manifest.origin_x_m, cache.manifest.origin_y_m)
a = np.asarray(cache.building_dsm)
print(f"site: {cache.rows}x{cache.cols}@{cache.pixel_size_m}m  a in [{a.min():.2f}, {a.max():.2f}]")

def tree(row, col, h, r, tid="t1"):
    x = grid.origin_x_m + (col + 0.5) * grid.pixel_size_m
    y = grid.origin_y_m - (row + 0.5) * grid.pixel_size_m
    return TreeSpec(tid, x, y, h, r)

CASES = {
    "S01_add_h6_r3":   tree(166, 102, 6.0, 3.0),
    "typical_h10_r4":  tree(166, 102, 10.0, 4.0),
    "tall_h18_r6":     tree(166, 102, 18.0, 6.0),
}

layer = TreeLayer(cache.tree_base, grid)
worker = ExactWorker(cache, layer, site_dir=SITE_DIR,
                     results_root=Path("/tmp/p8perf/results"),
                     selected_date_str=DATE_STR)
config = worker._site_influence_config()
print(f"config: sun_floor={config.minimum_direct_sun_altitude_deg:.3f}deg "
      f"max_shadow={config.maximum_shadow_length_m:.1f}m offset={config.elevation_offset_m}")

forcing = worker.forcing()
scene = compose_full_scene_tensors(cache, layer)
halo = required_halo_pixels(cache, scene, forcing)
print(f"conservative halo: {halo}px (old read window = write+{halo})")
print(f"scene amaxvalue={float(scene.amaxvalue):.2f}")

for name, t in CASES.items():
    layer2 = TreeLayer(cache.tree_base, grid)
    layer2.add_tree(t)
    w = ExactWorker(cache, layer2, site_dir=SITE_DIR,
                    results_root=Path("/tmp/p8perf/results"),
                    selected_date_str=DATE_STR)
    cfg = w._site_influence_config()
    batch = layer2.coalesced_batch()
    offsets = w._batch_elevation_offsets(batch, cfg)
    windows = w.dirty_windows()
    area = sum(win.area for win in windows)
    frac = area / (grid.rows * grid.cols)
    mode = choose_recompute_mode(windows[0], grid) if windows else None
    # exact read window for the (first) write window
    write = windows[0]
    write_full = write.expand(w.write_margin_pixels).clamp(rows=grid.rows, cols=grid.cols) if windows else None
    read = read_window_for_write_window(write_full, cache, scene, forcing)
    print(f"\n[{name}] tree at cell {w._tree_cell(t)}")
    print(f"  per-tree offset: {[round(v,2) for v in offsets.values()]}")
    print(f"  dirty windows: {[(x.row_start,x.row_stop,x.col_start,x.col_stop) for x in windows]}")
    print(f"  dirty fraction: {frac:.4f} -> mode {mode.value if mode else None}")
    print(f"  write(=dirty+margin): {(write_full.row_start,write_full.row_stop,write_full.col_start,write_full.col_stop)}"
          f"  frac {write_full.area/(grid.rows*grid.cols):.4f}")
    print(f"  exact read: {(read.row_start,read.row_stop,read.col_start,read.col_stop)}"
          f"  frac {read.area/(grid.rows*grid.cols):.4f}")
    old = write_full.expand(halo).clamp(rows=grid.rows, cols=grid.cols)
    print(f"  old read (halo): frac {old.area/(grid.rows*grid.cols):.4f}")
