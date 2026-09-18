# SPDX-License-Identifier: GPL-3.0-only
"""Focused repro: inspect the off-closure flips of the amaxvalue-raiser edit."""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/r4_proof")
from corridor_bruteforce import (  # noqa: E402
    N_PATCHES,
    PATCHES,
    amplitude_of,
    bits_of,
    build_main_tile,
    corridor_mask,
    march_offsets,
    scene_of,
)
from solweig_gpu.incremental.geometry import TreeSpec  # noqa: E402
from solweig_gpu.shadow import shadow as shadow_fn  # noqa: E402

cache, grid, layer = build_main_tile()
for i in range(4):
    pass  # the raiser edit alone reproduces (independent edits)

# reproduce edit sequence up to the raiser with the same RNG stream
rng = np.random.default_rng(20260904)


def pos(row, col):
    return (col + 0.5) * cache.pixel_size_m, (cache.rows - row - 0.5) * cache.pixel_size_m


def open_pos():
    while True:
        r, c = int(rng.integers(2, 56)), int(rng.integers(2, 56))
        if not (4 <= r < 12 and 4 <= c < 12) and not (40 <= r < 50 and 38 <= c < 48):
            return r, c


for i in range(4):
    r, c = open_pos()
    layer.add_tree(TreeSpec(f"t{i}", *pos(r, c), float(rng.uniform(3, 11)), float(rng.uniform(1.5, 4))))
layer.add_tree(TreeSpec("corner_ne", *pos(2, 57), 9.0, 3.0))
layer.add_tree(TreeSpec("corner_sw", *pos(57, 2), 6.0, 2.5))
layer.add_tree(TreeSpec("overlap", 30.5 * cache.pixel_size_m, 20.0 * cache.pixel_size_m, 4.0, 3.0))

canopy_before = layer.vegetation_rasters_window(grid.full_window)[0].copy()
scene_before = scene_of(cache, layer)
tall = TreeSpec("tall", *pos(30, 30), 18.0, 3.0)
layer.add_tree(tall)
canopy_after = layer.vegetation_rasters_window(grid.full_window)[0]
scene_after = scene_of(cache, layer)
C = canopy_before != canopy_after
print("C cells:", np.argwhere(C).tolist())

vb, vbb = bits_of(cache, scene_before)
va, vab = bits_of(cache, scene_after)
amp_b = amplitude_of(cache, scene_before)
amp_a = amplitude_of(cache, scene_after)
rows, cols = cache.rows, cache.cols
scale = 1.0 / cache.pixel_size_m
print(f"amp {amp_b} -> {amp_a}")

# replicate _recompute's march amplitude as tensors, to be identical
from solweig_gpu.incremental.solver import effective_march_amplitude  # noqa: E402

amp_union = max(amp_b, amp_a)
for index in range(N_PATCHES):
    altitude, azimuth, _ring = PATCHES[index]
    offsets = march_offsets(azimuth, altitude, amp_union, scale, rows, cols)
    corridor = corridor_mask(C, offsets)
    closure = corridor | C
    diff = vb[:, :, index] != va[:, :, index]
    off = diff & ~closure
    if not off.any():
        continue
    r, c = np.nonzero(off)
    print(
        f"VIOLATION patch {index} alt={float(altitude)} az={float(azimuth)} "
        f"steps={len(offsets)} cells={list(zip(r.tolist(), c.tolist()))[:8]} n={len(r)}"
    )
    # manual ray walk for the first violating cell
    t = (int(r[0]), int(c[0]))
    hits = []
    for dx, dy in offsets:
        rr, cc = t[0] + dx, t[1] + dy
        if 0 <= rr < rows and 0 <= cc < cols and C[rr, cc]:
            hits.append((dx, dy))
    print(f"  cell {t} ray hits C at steps: {hits}")
    # does a LONGER march (oracle absolute amplitude) hit C from this cell?
    extra = march_offsets(azimuth, altitude, float(scene_after.amaxvalue), scale, rows, cols)
    beyond = [o for o in extra if o not in offsets]
    hits2 = []
    for dx, dy in beyond:
        rr, cc = t[0] + dx, t[1] + dy
        if 0 <= rr < rows and 0 <= cc < cols and C[rr, cc]:
            hits2.append((dx, dy))
    print(f"  union steps={len(offsets)} abs-amp steps={len(extra)} beyond-union hits: {hits2}")
    # was the flip caused by step-count difference? compare trajectories with
    # fixed inputs: march the AFTER scene at the BEFORE amplitude and vice versa
    for label, scene, amp in (
        ("after-scene@amp_before", scene_after, amp_b),
        ("after-scene@amp_after", scene_after, amp_a),
    ):
        _sh, vegsh, _v = shadow_fn(
            torch.tensor(amp), scene.a, scene.vegdsm, scene.vegdsm2, scene.bush,
            azimuth, altitude, scale,
        )
        v = vegsh.numpy() != 0
        print(f"  {label}: cell bit = {v[t]}")
    print(f"  replay-before bit = {vb[t[0], t[1], index]}, replay-after bit = {va[t[0], t[1], index]}")
    break
