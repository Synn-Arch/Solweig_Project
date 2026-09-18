# SPDX-License-Identifier: GPL-3.0-only
"""Shared helpers for the R4 V0 pin and corridor brute force (r4-proof agent).

Everything here is READ-ONLY with respect to the repository; artifacts are
written only under /tmp/r4_proof/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Users/alansynn/Workspace/solweig")
sys.path.insert(0, str(REPO))

from solweig_gpu.incremental.cache import SiteCache  # noqa: E402
from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow  # noqa: E402
from solweig_gpu.incremental.solver import (  # noqa: E402
    FullSceneTensors,
    _recompute_veg_svf_window,
    _slice_cache_to_tensor,
    _sky_patch_geometry,
    compose_full_scene_tensors,
    effective_march_amplitude,
)
from solweig_gpu.incremental.trees import TreeLayer  # noqa: E402
from solweig_gpu.shadow import shadow as shadow_fn  # noqa: E402


def grid_for(cache: SiteCache) -> RasterGrid:
    return RasterGrid(
        rows=cache.rows,
        cols=cache.cols,
        pixel_size_m=cache.pixel_size_m,
        origin_x_m=cache.manifest.origin_x_m,
        origin_y_m=cache.manifest.origin_y_m,
    )


def baseline_scene(cache: SiteCache) -> tuple[FullSceneTensors, TreeLayer]:
    layer = TreeLayer(cache.tree_base, grid_for(cache), scenario_id="v0")
    return compose_full_scene_tensors(cache, layer), layer


def amplitude_report(cache: SiteCache, scene: FullSceneTensors) -> dict:
    a = scene.a
    vegdsm, vegdsm2 = scene.vegdsm, scene.vegdsm2
    bound = (
        torch.maximum(torch.maximum(a.max(), vegdsm.max()), vegdsm2.max())
        - torch.min(a)
    )
    eff = effective_march_amplitude(a, vegdsm, vegdsm2, scene_amaxvalue=scene.amaxvalue)
    return {
        "a_min": float(a.min()),
        "a_max": float(a.max()),
        "canopy_max": float(scene.canopy.max()),
        "dem_min": float(scene.dem.min()),
        "dem_max": float(scene.dem.max()),
        "vegdsm_max": float(vegdsm.max()),
        "vegdsm2_max": float(vegdsm2.max()),
        "vegdem_canopy_plus_dem_max": float(scene.vegdem.max()),
        "abs_amaxvalue": float(scene.amaxvalue),
        "relative_bound": float(bound),
        "effective_amplitude": float(eff),
        "clamp_active_bound_gt_abs": bool(bound > scene.amaxvalue),
        "bush_any_nonzero": bool(scene.bush.any()),
        "scale_px_per_m": 1.0 / float(cache.pixel_size_m),
    }


def sweep_absolute(cache: SiteCache, scene: FullSceneTensors):
    """Replicate the oracle svf_calculator march loop bit for bit.

    ``svf_calculator`` (shadow.py:545-586) calls
    ``shadow(amaxvalue_abs, a, vegdsm, vegdsm2, bush, azimuth, altitude, scale)``
    at FULL tile extent with the ABSOLUTE amaxvalue
    ``max(a.max(), (canopy + dem).max())`` -- exactly scene.amaxvalue as
    composed by compose_full_scene_tensors (solver.py:517).
    """
    rows, cols = cache.rows, cache.cols
    patches, _rings = _sky_patch_geometry(2)
    n = len(patches)
    vegshmat = np.zeros((rows, cols, n), dtype=np.float32)
    vbshmat = np.zeros((rows, cols, n), dtype=np.float32)
    amax = scene.amaxvalue
    a = scene.a
    vegdsm, vegdsm2, bush = scene.vegdsm, scene.vegdsm2, scene.bush
    scale = 1.0 / float(cache.pixel_size_m)
    for index in range(n):
        altitude, azimuth, _ring = patches[index]
        _sh, vegsh, vbshvegsh = shadow_fn(
            amax, a, vegdsm, vegdsm2, bush, azimuth, altitude, scale
        )
        vegshmat[:, :, index] = vegsh.numpy()
        vbshmat[:, :, index] = vbshvegsh.numpy()
    return vegshmat, vbshmat


def sweep_effective(cache: SiteCache, scene: FullSceneTensors):
    """The CURRENT solver replay path at full-tile extent (veg_changed=True).

    This is exactly what window_svf_bundle(cache, scene, full, veg_changed=True,
    cube_window=full) executes (solver.py:872-880) -- W2 march windows at
    full-tile accumulate extent are the full tile, and the march amplitude is
    effective_march_amplitude (window-relative, here global).
    """
    full = RasterWindow(0, cache.rows, 0, cache.cols)
    veg_scalars, vegshmat, vbshmat, svftotal = _recompute_veg_svf_window(
        cache,
        scene,
        full,
        _slice_cache_to_tensor(cache, "svf", full),
        accumulate_window=full,
    )
    return veg_scalars, vegshmat.numpy(), vbshmat.numpy(), svftotal


def compare_stack(replay: np.ndarray, cached: np.ndarray, label: str) -> dict:
    """Per-patch bit comparison of a replayed cube against the cache cube."""
    assert replay.shape == cached.shape, (replay.shape, cached.shape)
    assert replay.dtype == np.float32 and cached.dtype == np.float32
    patches, _rings = _sky_patch_geometry(2)
    per_patch = []
    for p in range(replay.shape[2]):
        diff = replay[:, :, p] != cached[:, :, p]
        count = int(diff.sum())
        entry = {
            "patch": p,
            "ring_altitude_deg": float(patches[p][0]),
            "azimuth_deg": float(patches[p][1]),
            "mismatch_cells": count,
        }
        if count:
            rows, cols = np.nonzero(diff)
            entry["bbox"] = [
                int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
            ]
            edge = np.minimum(
                np.minimum(rows, replay.shape[0] - 1 - rows),
                np.minimum(cols, replay.shape[1] - 1 - cols),
            )
            entry["edge_dist_min"] = int(edge.min())
            entry["edge_dist_max"] = int(edge.max())
            entry["edge_dist_median"] = float(np.median(edge))
            # sign of the disagreement: (replay value, cache value) counts
            vals = {}
            for rv, cv in zip(replay[diff, p], cached[diff, p]):
                vals[f"replay{rv:g}_cache{cv:g}"] = vals.get(f"replay{rv:g}_cache{cv:g}", 0) + 1
            entry["value_pairs"] = vals
            entry["sample_cells"] = [
                (int(r), int(c)) for r, c in list(zip(rows, cols))[:20]
            ]
        per_patch.append(entry)
    total = sum(e["mismatch_cells"] for e in per_patch)
    mismatching = [e for e in per_patch if e["mismatch_cells"]]
    print(
        f"[{label}] total mismatching (cell,patch) pairs: {total} "
        f"across {len(mismatching)}/{replay.shape[2]} patches"
    )
    return {
        "label": label,
        "total_mismatch_pairs": total,
        "patches_with_mismatch": len(mismatching),
        "bitwise_identical": total == 0,
        "per_patch_mismatching": mismatching,
    }


def edge_distance_stats(mask: np.ndarray) -> dict:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return {"count": 0}
    r_max, c_max = mask.shape[0] - 1, mask.shape[1] - 1
    edge = np.minimum(
        np.minimum(rows, r_max - rows), np.minimum(cols, c_max - cols)
    )
    return {
        "count": int(len(rows)),
        "edge_dist_min": int(edge.min()),
        "edge_dist_median": float(np.median(edge)),
        "edge_dist_max": int(edge.max()),
    }
