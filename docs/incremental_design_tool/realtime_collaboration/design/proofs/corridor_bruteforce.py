# SPDX-License-Identifier: GPL-3.0-only
"""R4 deliverable B -- corridor-sufficiency brute force on synthetic tiles.

Claim under test (design r4-veg-svf-occlusion-state.md section 3): after a
vegetation edit with changed-cell set C, the veg-SVF bits at (target, patch)
can change only if the patch's march ray from the target intersects C.

This script:

1. builds a 60x60 synthetic tile (flat DEM, two buildings, scattered baseline
   trees) and runs ~20 sequential random edits (add / move / resize / delete,
   incl. one taller-than-site-max add and the delete of that tree, and corner
   placements);
2. for every edit compares FULL-TILE replay bits before vs after the edit and
   checks the flip set against the corridor closure
       closure_p = {t : exists executed-or-reachable step s with
                    t + (dx_s, dy_s) in C}  U  C
   computed with shadow()'s EXACT per-step offsets (float32 while-stop over
   the union of the pre/post march step counts);
3. phase-A march-window check: re-marching only the closure cells on the W2
   reach-expanded window must reproduce the full replay bitwise, while the
   RAW closure bbox (no reach expansion) is shown to diverge;
4. a dedicated clamped-amplitude corner probe (tall tree ON a tall building,
   delete the dominant tree) probing the boundary of the amplitude argument.

Evidence: /tmp/r4_proof/corridor_bruteforce.json (+ stdout log).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/r4_proof")
from common import REPO  # noqa: E402

from solweig_gpu.incremental.geometry import (  # noqa: E402
    RasterGrid,
    RasterWindow,
    TreeSpec,
)
from solweig_gpu.incremental.solver import (  # noqa: E402
    FullSceneTensors,
    _patch_march_reach_pixels,
    _patch_march_window,
    _quadrant_read_direction,
    _sky_patch_geometry,
    compose_full_scene_tensors,
    effective_march_amplitude,
)
from solweig_gpu.incremental.trees import (  # noqa: E402
    TreeLayer,
    rasterize_tree_patch,
)
from solweig_gpu.shadow import shadow as shadow_fn  # noqa: E402

PATCHES, RINGS = _sky_patch_geometry(2)
N_PATCHES = len(PATCHES)


# ---------------------------------------------------------------------------
# Synthetic site plumbing (compose_full_scene_tensors called on a stub cache)
# ---------------------------------------------------------------------------


@dataclass
class StubCache:
    rows: int
    cols: int
    pixel_size_m: float
    building_dsm: np.ndarray
    dem: np.ndarray
    tree_base: np.ndarray
    manifest: object = None

    def __post_init__(self):
        self.manifest = type(
            "M",
            (),
            {"origin_x_m": 0.0, "origin_y_m": float(self.rows * self.pixel_size_m)},
        )()


def make_site(rows, cols, pixel, dem, building, canopy_base) -> tuple[StubCache, RasterGrid, TreeLayer]:
    cache = StubCache(
        rows=rows,
        cols=cols,
        pixel_size_m=pixel,
        building_dsm=building.astype(np.float32),
        dem=dem.astype(np.float32),
        tree_base=canopy_base.astype(np.float32),
    )
    grid = RasterGrid(rows, cols, pixel, 0.0, rows * pixel)
    layer = TreeLayer(cache.tree_base, grid, scenario_id="bf")
    return cache, grid, layer


def scene_of(cache, layer) -> FullSceneTensors:
    return compose_full_scene_tensors(cache, layer)


def bits_of(cache, scene):
    """Full-tile replay bits via the CURRENT solver path (veg_changed=True)."""
    from solweig_gpu.incremental.solver import _recompute_veg_svf_window

    full = RasterWindow(0, cache.rows, 0, cache.cols)
    _scalars, vegsh, vbsh, _svftotal = _recompute_veg_svf_window(
        cache,
        scene,
        full,
        torch.zeros((cache.rows, cache.cols)),
        accumulate_window=full,
    )
    return vegsh.numpy() != 0, vbsh.numpy() != 0


def amplitude_of(cache, scene) -> float:
    full = RasterWindow(0, cache.rows, 0, cache.cols)
    from solweig_gpu.incremental.solver import _crop

    a = _crop(scene.a, full)
    return float(
        effective_march_amplitude(
            a,
            _crop(scene.vegdsm, full),
            _crop(scene.vegdsm2, full),
            scene_amaxvalue=scene.amaxvalue,
        )
    )


# ---------------------------------------------------------------------------
# Exact march offsets (replicates shadow()'s scalar float32 loop)
# ---------------------------------------------------------------------------


def march_offsets(azimuth, altitude, amaxvalue: float, scale: float, sizex: int, sizey: int):
    """Per-step (dx, dy) replicating shadow.py lines 194-255 bit for bit.

    ``azimuth``/``altitude`` are 0-dim float32 tensors (the values the replay
    passes); ``amaxvalue`` the float amplitude of the while-stop. Returns the
    list of (dx, dy) integer shifts for every EXECUTED step.
    """
    degrees = torch.pi / 180.0
    az = azimuth
    if float(az) == 0.0:
        az = az * 0.0 + 1e-12
    az = az * degrees
    alt = altitude * degrees
    dx = torch.tensor(0.0, dtype=torch.float32)
    dy = torch.tensor(0.0, dtype=torch.float32)
    dz = torch.tensor(0.0, dtype=torch.float32)
    amax = torch.tensor(float(amaxvalue), dtype=torch.float32)

    pibyfour = torch.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = torch.sin(az)
    cosazimuth = torch.cos(az)
    tanazimuth = torch.tan(az)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1.0 / sinazimuth))
    dscos = torch.abs((1.0 / cosazimuth))
    tanaltitudebyscale = torch.tan(alt) / scale

    index = 1.0
    offsets = []
    while bool(amax >= dz) and bool(torch.abs(dx) < sizex) and bool(torch.abs(dy) < sizey):
        if bool(pibyfour <= az < threetimespibyfour) or bool(
            fivetimespibyfour <= az < seventimespibyfour
        ):
            dy = signsinazimuth * index
            dx = -1.0 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos
        dz = ds * index * tanaltitudebyscale
        offsets.append((int(dx), int(dy)))
        index += 1.0
    return offsets


def corridor_mask(C: np.ndarray, offsets) -> np.ndarray:
    """{t : t + (dx_s, dy_s) in C for some executed step s}, out-of-bounds reads dropped."""
    shape = C.shape
    out = np.zeros(shape, dtype=bool)
    for dx, dy in offsets:
        r0, r1 = max(0, -dx), shape[0] - max(0, dx)
        c0, c1 = max(0, -dy), shape[1] - max(0, dy)
        if r0 >= r1 or c0 >= c1:
            continue
        out[r0:r1, c0:c1] |= C[r0 + dx : r1 + dx, c0 + dy : c1 + dy]
    return out


def footprint_mask(grid, window, *trees) -> np.ndarray:
    mask = np.zeros((window.height, window.width), dtype=bool)
    for tree in trees:
        if tree is None:
            continue
        canopy, _trunk = rasterize_tree_patch(tree, grid, window)
        mask |= canopy != 0
    return mask


# ---------------------------------------------------------------------------
# The sufficiency check for one edit
# ---------------------------------------------------------------------------


@dataclass
class EditCheck:
    name: str
    amax_before: float = 0.0
    amax_after: float = 0.0
    abs_before: float = 0.0
    abs_after: float = 0.0
    bound_before: float = 0.0
    bound_after: float = 0.0
    clamp_before: bool = False
    clamp_after: bool = False
    c_cells_exact: int = 0
    c_cells_footprint: int = 0
    vegsh_flips: int = 0
    vbsh_flips: int = 0
    flips_in_C_not_raycorridor: int = 0
    violations: list = field(default_factory=list)  # off-closure flips (the claim breaker)


def _bound_abs(scene) -> tuple[float, float, bool]:
    bound = float(
        torch.maximum(
            torch.maximum(scene.a.max(), scene.vegdsm.max()), scene.vegdsm2.max()
        )
        - torch.min(scene.a)
    )
    absa = float(scene.amaxvalue)
    return bound, absa, bound > absa


def check_edit(cache, grid, scene_before, scene_after, C_exact, C_foot, name) -> EditCheck:
    rows, cols = cache.rows, cache.cols
    scale = 1.0 / cache.pixel_size_m
    a_before = scene_before
    vb, vbb = bits_of(cache, scene_before)
    va, vab = bits_of(cache, scene_after)
    amp_b = amplitude_of(cache, scene_before)
    amp_a = amplitude_of(cache, scene_after)

    res = EditCheck(
        name=name,
        amax_before=amp_b,
        amax_after=amp_a,
        abs_before=float(scene_before.amaxvalue),
        abs_after=float(scene_after.amaxvalue),
    )
    res.bound_before, _b, res.clamp_before = _bound_abs(scene_before)
    res.bound_after, _a2, res.clamp_after = _bound_abs(scene_after)
    res.c_cells_exact = int(C_exact.sum())
    res.c_cells_footprint = int(C_foot.sum())

    amp_union = max(amp_b, amp_a)
    for index in range(N_PATCHES):
        altitude, azimuth, _ring = PATCHES[index]
        offsets = march_offsets(
            azimuth, altitude, amp_union, scale, rows, cols
        )
        corridor = corridor_mask(C_exact, offsets)
        closure = corridor | C_exact
        for cube_before, cube_after, label in (
            (vb, va, "vegsh"),
            (vbb, vab, "vbsh"),
        ):
            diff = cube_before[:, :, index] != cube_after[:, :, index]
            n = int(diff.sum())
            if label == "vegsh":
                res.vegsh_flips += n
            else:
                res.vbsh_flips += n
            if n == 0:
                continue
            off = diff & ~closure
            if off.any():
                r, c = np.nonzero(off)
                res.violations.append(
                    {
                        "patch": index,
                        "ring_altitude_deg": float(altitude),
                        "azimuth_deg": float(azimuth),
                        "stack": label,
                        "cells": [(int(x), int(y)) for x, y in zip(r[:10], c[:10])],
                        "count": int(off.sum()),
                    }
                )
            # evidence that C itself must be part of the closure
            in_c_only = diff & C_exact & ~corridor
            if label == "vegsh":
                res.flips_in_C_not_raycorridor += int(in_c_only.sum())
    return res


# ---------------------------------------------------------------------------
# Tile 1: 60x60, buildings are the tallest structures (no-clamp regime)
# ---------------------------------------------------------------------------


def build_main_tile(*, trees_off_se_block: bool = False):
    """60x60 tile. ``trees_off_se_block`` keeps random trees OFF the 4 m SE
    block as well, which keeps the scene in the NO-CLAMP regime (the clamp
    activates exactly when some vegdsm cell exceeds the absolute amplitude,
    i.e. a tree standing on a building)."""
    rows = cols = 60
    pixel = 2.0
    dem = np.zeros((rows, cols), dtype=np.float32)
    building = dem.copy()
    building[4:12, 4:12] = 12.0  # tallest structure (north-west block)
    building[40:50, 38:48] = 4.0
    grid = RasterGrid(rows, cols, pixel, 0.0, rows * pixel)
    canopy = np.zeros((rows, cols), dtype=np.float32)
    for tree in (
        TreeSpec("base1", 30.5 * pixel, 20.0 * pixel, 6.0, 3.0),
        TreeSpec("base2", 45.0 * pixel, 12.0 * pixel, 8.0, 3.5),
        TreeSpec("base3", 14.0 * pixel, 44.0 * pixel, 5.0, 2.5),
        TreeSpec("base4", 50.0 * pixel, 50.0 * pixel, 7.0, 2.0),
    ):
        c, _t = rasterize_tree_patch(tree, grid, grid.full_window)
        np.maximum(canopy, c, out=canopy)
    return make_site(rows, cols, pixel, dem, building, canopy)


def run_main_sequence(*, noclamp: bool = False) -> list[EditCheck]:
    cache, grid, layer = build_main_tile(trees_off_se_block=noclamp)
    rng = np.random.default_rng(20260904)

    def pos(row, col):
        return (col + 0.5) * cache.pixel_size_m, (cache.rows - row - 0.5) * cache.pixel_size_m

    def on_block(r, c):
        nw = 4 <= r < 12 and 4 <= c < 12
        se = 40 <= r < 50 and 38 <= c < 48
        return nw or (se and noclamp)

    def open_tree():
        """Draw (row, col, height, radius). In the no-clamp variant the whole
        crown disc must clear both building blocks: a crown overhanging a
        building raises vegdsm above the absolute amplitude and activates the
        effective-amplitude clamp (measured: that alone breaks corridor
        sufficiency under amplitude changes)."""
        while True:
            r, c = int(rng.integers(2, 56)), int(rng.integers(2, 56))
            h = float(rng.uniform(3, 11))
            rad = float(rng.uniform(1.5, 4))
            if on_block(r, c):
                continue
            if noclamp:
                radc = rad / cache.pixel_size_m
                for r0, r1, c0, c1 in ((4, 12, 4, 12), (40, 50, 38, 48)):
                    if (
                        r0 - radc <= r <= r1 - 1 + radc
                        and c0 - radc <= c <= c1 - 1 + radc
                    ):
                        break
                else:
                    return r, c, h, rad
                continue
            return r, c, h, rad

    results: list[EditCheck] = []
    state_scene = scene_of(cache, layer)

    def apply(name, mutator, *trees):
        nonlocal state_scene
        canopy_before = layer.vegetation_rasters_window(grid.full_window)[0].copy()
        scene_before = state_scene
        mutator()
        canopy_after = layer.vegetation_rasters_window(grid.full_window)[0]
        scene_after = scene_of(cache, layer)
        C_exact = canopy_before != canopy_after
        full = grid.full_window
        C_foot = footprint_mask(grid, full, *trees)
        t0 = time.time()
        res = check_edit(cache, grid, scene_before, scene_after, C_exact, C_foot, name)
        res.seconds = round(time.time() - t0, 1)  # type: ignore[attr-defined]
        state_scene = scene_after
        print(
            f"[edit] {name}: C_exact={res.c_cells_exact} C_foot={res.c_cells_footprint} "
            f"amp {res.amax_before:g}->{res.amax_after:g} abs {res.abs_before:g}->{res.abs_after:g} "
            f"clamp {res.clamp_before}/{res.clamp_after} "
            f"vegsh_flips={res.vegsh_flips} vbsh_flips={res.vbsh_flips} "
            f"violations={len(res.violations)} ({res.seconds}s)"  # type: ignore[attr-defined]
        )
        results.append(res)
        return res

    # adds
    for i in range(4):
        r, c, h, rad = open_tree()
        tree = TreeSpec(f"t{i}", *pos(r, c), h, rad)
        apply(f"add{i}", lambda tree=tree: layer.add_tree(tree), tree, None)
    # boundary: corner adds
    corner1 = TreeSpec("corner_ne", *pos(2, 57), 9.0, 3.0)
    apply("add_corner_ne", lambda: layer.add_tree(corner1), corner1, None)
    corner2 = TreeSpec("corner_sw", *pos(57, 2), 6.0, 2.5)
    apply("add_corner_sw", lambda: layer.add_tree(corner2), corner2, None)
    # overlap with an existing baseline tree
    overlap = TreeSpec("overlap", 30.5 * cache.pixel_size_m, 20.0 * cache.pixel_size_m, 4.0, 3.0)
    apply("add_under_base1", lambda: layer.add_tree(overlap), overlap, None)
    # amaxvalue RAISER: taller than the 12 m site max
    tall = TreeSpec("tall", *pos(30, 30), 18.0, 3.0)
    apply("add_tall_raiser", lambda: layer.add_tree(tall), tall, None)
    # raise it further
    apply(
        "resize_tall_20m",
        lambda: layer.update_tree("tall", height_m=20.0),
        TreeSpec("tall", *pos(30, 30), 20.0, 3.0),
        TreeSpec("tall", *pos(30, 30), 18.0, 3.0),
    )
    # amaxvalue DROPPER: delete the tall tree
    apply(
        "delete_tall_dropper",
        lambda: layer.delete_tree("tall"),
        None,
        TreeSpec("tall", *pos(30, 30), 20.0, 3.0),
    )
    # moves (incl. corner to corner)
    apply(
        "move_corner_ne",
        lambda: layer.move_tree("corner_ne", x_m=pos(56, 3)[0], y_m=pos(56, 3)[1]),
        TreeSpec("corner_ne", *pos(56, 3), 9.0, 3.0),
        TreeSpec("corner_ne", *pos(2, 57), 9.0, 3.0),
    )
    r, c = 20, 50  # fixed clear move target (both blocks avoided)
    apply(
        "move_t0",
        lambda r=r, c=c: layer.move_tree("t0", x_m=pos(r, c)[0], y_m=pos(r, c)[1]),
        TreeSpec("t0", *pos(r, c), layer._live["t0"].height_m, layer._live["t0"].canopy_radius_m),
        TreeSpec("t0", 0, 0, 1.0, 1),  # radius/pos overridden below
    )
    # resize to overlap a neighbour
    apply(
        "resize_t1_wide",
        lambda: layer.update_tree("t1", canopy_radius_m=6.0),
        TreeSpec("t1", 0, 0, 1.0, 6.0),
        TreeSpec("t1", 0, 0, 1.0, 1.0),
    )
    # deletes of ordinary trees
    apply("delete_corner_sw", lambda: layer.delete_tree("corner_sw"), None, corner2)
    apply("delete_overlap", lambda: layer.delete_tree("overlap"), None, overlap)
    # final move+resize of a big tree
    apply(
        "move_resize_base2friend",
        lambda: layer.move_tree("t2", x_m=pos(10, 50)[0], y_m=pos(10, 50)[1]),
        TreeSpec("t2", *pos(10, 50), layer._live["t2"].height_m, layer._live["t2"].canopy_radius_m),
        TreeSpec("t2", 0, 0, 1.0, 1.0),
    )
    return results


# ---------------------------------------------------------------------------
# Phase-A march-window demonstration (one representative edit)
# ---------------------------------------------------------------------------


def phase_a_demo() -> dict:
    cache, grid, layer = build_main_tile(trees_off_se_block=True)
    tree = TreeSpec("demo", 30.5 * cache.pixel_size_m, 30.0 * cache.pixel_size_m, 10.0, 3.0)
    canopy_before = layer.vegetation_rasters_window(grid.full_window)[0].copy()
    scene_before = scene_of(cache, layer)
    layer.add_tree(tree)
    canopy_after = layer.vegetation_rasters_window(grid.full_window)[0]
    scene_after = scene_of(cache, layer)
    C = canopy_before != canopy_after

    va, vab = bits_of(cache, scene_after)
    rows, cols = cache.rows, cache.cols
    scale = 1.0 / cache.pixel_size_m
    amp = amplitude_of(cache, scene_after)
    full = RasterWindow(0, rows, 0, cols)

    expanded_bad = {"patches_checked": 0, "patches_diverging": 0, "worst_cells": 0}
    raw_bad = {"patches_checked": 0, "patches_diverging": 0, "worst_cells": 0}

    for index in list(range(0, 31)) + list(range(61, 89)):  # 6 deg + 30 deg rings
        altitude, azimuth, _ring = PATCHES[index]
        reach = _patch_march_reach_pixels(amp, scale, float(altitude))
        offsets = march_offsets(azimuth, altitude, max(amp, amplitude_of(cache, scene_before)), scale, rows, cols)
        closure = corridor_mask(C, offsets) | C
        if not closure.any():
            continue
        rs, cs = np.nonzero(closure)
        cw = RasterWindow(
            int(rs.min()), int(rs.max()) + 1, int(cs.min()), int(cs.max()) + 1
        )
        # W2-style: expand by reach in the patch's READ directions, clamp to tile
        mw = _patch_march_window(cw, full, azimuth_deg=float(azimuth), reach_pixels=reach)

        def crop(t, w):
            return t[w.row_start : w.row_stop, w.col_start : w.col_stop]

        for label, w in (("expanded", mw), ("raw", cw)):
            a_w = crop(scene_after.a, w)
            eff_w = effective_march_amplitude(
                a_w, crop(scene_after.vegdsm, w), crop(scene_after.vegdsm2, w),
                scene_amaxvalue=scene_after.amaxvalue,
            )
            _sh, vegsh_w, vbsh_w = shadow_fn(
                eff_w, a_w, crop(scene_after.vegdsm, w),
                crop(scene_after.vegdsm2, w), crop(scene_after.bush, w),
                azimuth, altitude, scale,
            )
            vegsh_w = vegsh_w.numpy() != 0
            vbsh_w = vbsh_w.numpy() != 0
            sl = (
                slice(cw.row_start - w.row_start, cw.row_stop - w.row_start),
                slice(cw.col_start - w.col_start, cw.col_stop - w.col_start),
            )
            ref_vegsh = va[cw.row_start : cw.row_stop, cw.col_start : cw.col_stop, index]
            ref_vbsh = vab[cw.row_start : cw.row_stop, cw.col_start : cw.col_stop, index]
            bad = [
                int((vegsh_w[sl] != ref_vegsh).sum()),
                int((vbsh_w[sl] != ref_vbsh).sum()),
            ]
            target = expanded_bad if label == "expanded" else raw_bad
            target["patches_checked"] += 1
            if bad[0] or bad[1]:
                target["patches_diverging"] += 1
                target["worst_cells"] = max(target["worst_cells"], max(bad))
    return {"reach_expanded_march": expanded_bad, "raw_bbox_march": raw_bad}


# ---------------------------------------------------------------------------
# Tile 2: the clamped-amplitude corner probe (tree ON a tall building)
# ---------------------------------------------------------------------------


def corner_probe() -> dict:
    rows = cols = 120
    pixel = 2.0
    dem = np.zeros((rows, cols), dtype=np.float32)
    building = dem.copy()
    building[10:20, 10:20] = 30.0  # tall building block
    grid = RasterGrid(rows, cols, pixel, 0.0, rows * pixel)
    canopy = np.zeros((rows, cols), dtype=np.float32)
    # tall tree ON the building: vegdsm = 30 + 25 = 55 > abs amplitude
    t_on_bldg = TreeSpec("on_bldg", 15.5 * pixel, (rows - 15.5) * pixel, 25.0, 3.0)
    c, _t = rasterize_tree_patch(t_on_bldg, grid, grid.full_window)
    np.maximum(canopy, c, out=canopy)
    cache, grid, layer = make_site(rows, cols, pixel, dem, building, canopy)

    # dominant tree on open ground: defines the absolute amplitude (45 m)
    dom = TreeSpec("dom", 80.0 * pixel, (rows - 80.0) * pixel, 45.0, 4.0)
    layer.add_tree(dom)
    scene_before = scene_of(cache, layer)
    layer.delete_tree("dom")
    scene_after = scene_of(cache, layer)

    amp_b = amplitude_of(cache, scene_before)
    amp_a = amplitude_of(cache, scene_after)
    canopy_before = scene_before.canopy.numpy()
    canopy_after = scene_after.canopy.numpy()
    C = canopy_before != canopy_after
    print(
        f"[corner] amp {amp_b:g} -> {amp_a:g}; abs "
        f"{float(scene_before.amaxvalue):g} -> {float(scene_after.amaxvalue):g}; "
        f"vegdsm.max {float(scene_before.vegdsm.max()):g}; C cells {int(C.sum())}"
    )

    res = check_edit(cache, grid, scene_before, scene_after, C, C, "corner_clamped_dropper")
    print(
        f"[corner] vegsh_flips={res.vegsh_flips} vbsh_flips={res.vbsh_flips} "
        f"violations={len(res.violations)}"
    )
    return {
        "amax_before": amp_b,
        "amax_after": amp_a,
        "abs_before": float(scene_before.amaxvalue),
        "abs_after": float(scene_after.amaxvalue),
        "vegdsm_max_before": float(scene_before.vegdsm.max()),
        "bound_gt_abs_clamped": True,
        "vegsh_flips": res.vegsh_flips,
        "vbsh_flips": res.vbsh_flips,
        "violations": res.violations[:20],
        "violation_count": len(res.violations),
    }


if __name__ == "__main__":
    out = {}

    def seq_json(results):
        return [
            {
                "name": r.name,
                "amax_before": r.amax_before,
                "amax_after": r.amax_after,
                "abs_before": r.abs_before,
                "abs_after": r.abs_after,
                "bound_before": r.bound_before,
                "bound_after": r.bound_after,
                "clamp_before": r.clamp_before,
                "clamp_after": r.clamp_after,
                "c_cells_exact": r.c_cells_exact,
                "c_cells_footprint": r.c_cells_footprint,
                "vegsh_flips": r.vegsh_flips,
                "vbsh_flips": r.vbsh_flips,
                "flips_in_C_not_raycorridor": r.flips_in_C_not_raycorridor,
                "violations": r.violations[:10],
                "violation_count": len(r.violations),
            }
            for r in results
        ]

    t0 = time.time()
    print("=== NO-CLAMP 60x60 edit sequence (trees kept off all buildings) ===")
    out["main_sequence_noclamp"] = seq_json(run_main_sequence(noclamp=True))
    print(f"no-clamp sequence done in {time.time() - t0:.0f}s")

    t0 = time.time()
    print("=== CLAMPED 60x60 edit sequence (random trees may land on the 4 m block) ===")
    out["main_sequence_clamped"] = seq_json(run_main_sequence(noclamp=False))
    print(f"clamped sequence done in {time.time() - t0:.0f}s")

    t0 = time.time()
    print("=== phase-A march-window demo ===")
    out["phase_a"] = phase_a_demo()
    print(f"phase-A demo done in {time.time() - t0:.0f}s: {out['phase_a']}")

    t0 = time.time()
    print("=== clamped-amplitude corner probe ===")
    out["corner_probe"] = corner_probe()
    print(f"corner probe done in {time.time() - t0:.0f}s")

    json.dump(out, open("/tmp/r4_proof/corridor_bruteforce.json", "w"), indent=1)
    print("wrote /tmp/r4_proof/corridor_bruteforce.json")
