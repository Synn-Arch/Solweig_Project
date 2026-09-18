# SPDX-License-Identifier: GPL-3.0-only
"""T26 ablation runner: §474 whole-cell bundle reuse (DESIGN §8.3).

Mode:
  hitrate — structured-synthetic scenes through edit generations:
            per-generation hits/misses/unique classes, the C2
            verify-inclusive cost accounting (raw fold save vs
            fp+lookup+insert+assemble overhead), measured cache size.
            Every generation is asserted bit-equal to a fresh direct
            fold (correctness rides every row).

WALLS are PROVISIONAL (no measurement window): correctness-class runs
under shared-host load, loadavg published per row. Formal paired timing
queues behind a lead OPEN window.

Scenes are structured-synthetic by design: the t08 capture carries only
the 2-D ``march_vegsh`` plane, not the 153-patch packed march state, so
real-scene class counts need a march run (t19b lane) — recorded as a
T29 caveat. The bracket [uniform-open ... fully-random] bounds the
class-diversity axis the hit rate depends on.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solweig_core import bitplanes as bp                      # noqa: E402
from solweig_core.numba_cpu import svf_fold as sf             # noqa: E402
from tests.ultrafast import t26_bundle_cache as t26           # noqa: E402

ART = t26.ART
N_PATCHES = sf.N_PATCHES


def env_row() -> dict:
    a1, a5, a15 = os.getloadavg()
    return {"load1": round(a1, 2), "load5": round(a5, 2),
            "load15": round(a15, 2), "python": sys.version.split()[0],
            "platform": sys.platform, "ts": time.time()}


def rss_kib() -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def planes_of(res):
    return {n: np.ascontiguousarray(getattr(res, n))
            for n in sf.FOLD_OUTPUT_NAMES}


def planes_equal(a: dict, b: dict) -> bool:
    return all(np.array_equal(a[n].view(np.uint32), b[n].view(np.uint32))
               for n in sf.FOLD_OUTPUT_NAMES)


# ---------------------------------------------------------------------------
# scenes (structured-synthetic)
# ---------------------------------------------------------------------------


def scene_uniform_open(rows: int, cols: int):
    """Every cell: vegsh all-open, vbsh all-zero, vegdem2 non-zero,
    DISTINCT svf_building per cell — the zero-diversity extreme for
    variant B (one class for the whole tile; svf_building is NOT in the
    key because svftotal is recomputed per cell)."""
    veg = np.ones((rows, cols, N_PATCHES), np.uint8)
    vbsh = np.zeros((rows, cols, N_PATCHES), np.uint8)
    v2 = np.full((rows, cols), 2.5, np.float32)
    svfb = ((np.arange(rows * cols).reshape(rows, cols) % 997).astype(
        np.float32) / 997.0 * 0.8 + 0.1)
    return bp.pack_bits(veg), bp.pack_bits(vbsh), v2, svfb


def scene_structured(rows: int, cols: int, seed: int = 7):
    """Block/corridor structure: buildings (veg zero), open sky (veg
    one, vbsh zero), vegetation corridors (shared vbsh patterns), plus a
    noisy 5% fringe — few dominant classes + long tail, the axis real
    march bits live on."""
    rng = np.random.default_rng(seed)
    veg = np.ones((rows, cols, N_PATCHES), np.uint8)
    vbsh = np.zeros((rows, cols, N_PATCHES), np.uint8)
    n_blk = max(4, rows * cols // 5000)
    for _ in range(n_blk):
        r0 = int(rng.integers(0, rows - 25))
        c0 = int(rng.integers(0, cols - 25))
        h = int(rng.integers(8, 25))
        w = int(rng.integers(8, 25))
        veg[r0:r0 + h, c0:c0 + w] = 0
    for _ in range(n_blk):
        r0 = int(rng.integers(0, rows - 12))
        c0 = int(rng.integers(0, cols - 42))
        h = int(rng.integers(4, 12))
        w = int(rng.integers(6, 30))
        pattern = (rng.random((1, N_PATCHES)) < 0.5).astype(np.uint8)
        vbsh[r0:r0 + h, c0:c0 + w] = pattern
    fringe = rng.random((rows, cols)) < 0.05
    n_f = int(fringe.sum())
    veg[fringe] = (rng.random((n_f, N_PATCHES)) < 0.3).astype(np.uint8)
    vbsh[fringe] = (rng.random((n_f, N_PATCHES)) < 0.5).astype(np.uint8)
    v2 = (rng.random((rows, cols)) * 10).astype(np.float32)
    v2[rng.random((rows, cols)) < 0.15] = np.float32(0.0)
    svfb = (rng.random((rows, cols)) * 0.8 + 0.1).astype(np.float32)
    return bp.pack_bits(veg), bp.pack_bits(vbsh), v2, svfb


def scene_random(rows: int, cols: int, seed: int = 11):
    """Fully-random bits — the zero-hit extreme (every fingerprint
    unique)."""
    rng = np.random.default_rng(seed)
    veg = (rng.random((rows, cols, N_PATCHES)) < 0.3).astype(np.uint8)
    vbsh = (rng.random((rows, cols, N_PATCHES)) < 0.5).astype(np.uint8)
    v2 = (rng.random((rows, cols)) * 10).astype(np.float32)
    v2[rng.random((rows, cols)) < 0.2] = np.float32(0.0)
    svfb = (rng.random((rows, cols)) * 0.8 + 0.1).astype(np.float32)
    return bp.pack_bits(veg), bp.pack_bits(vbsh), v2, svfb


def block_copies(rng, shapes, n_blocks: int = 3):
    """Random same-scene block copy specs (source -> target), the edit
    model the incremental path sees: geometry edits move existing
    patterns around."""
    rows, cols = shapes
    specs = []
    for _ in range(n_blocks):
        h = int(rng.integers(4, 16))
        w = int(rng.integers(4, 16))
        tr = int(rng.integers(0, rows - h))
        tc = int(rng.integers(0, cols - w))
        sr = int(rng.integers(0, rows - h))
        sc = int(rng.integers(0, cols - w))
        specs.append((slice(tr, tr + h), slice(tc, tc + w),
                      slice(sr, sr + h), slice(sc, sc + w)))
    return specs


def apply_edit(cur, specs):
    """Apply block copies to (veg_data, vbsh_data, v2) — vegdem2 moves
    with the block so copied cells keep their fingerprint identity."""
    veg_d, vbsh_d, v2 = cur
    veg_n, vbsh_n, v2_n = veg_d.copy(), vbsh_d.copy(), v2.copy()
    for tr, tc, sr, sc in specs:
        veg_n[tr, tc] = veg_d[sr, sc]
        vbsh_n[tr, tc] = vbsh_d[sr, sc]
        v2_n[tr, tc] = v2[sr, sc]
    return veg_n, vbsh_n, v2_n


# ---------------------------------------------------------------------------
# hitrate mode
# ---------------------------------------------------------------------------


def run_hitrate(rounds: int = 3) -> int:
    jf = ART / "hitrate.jsonl"
    scenes = {
        "uniform_open_500": scene_uniform_open(500, 500),
        "structured_500": scene_structured(500, 500),
        "structured_200": scene_structured(200, 200, seed=8),
        "random_300": scene_random(300, 300),
    }
    for name, (veg, vbsh, v2, svfb) in scenes.items():
        rows, cols = veg.rows, veg.cols
        rng = np.random.default_rng(abs(hash(name)) % (2**31))
        cache = t26.BundleCache(variant="B")
        cur = (veg.data, vbsh.data, v2)
        prev_data = cur  # the caller's carried packed state
        prev_planes = None
        per_cell_us = None
        for gen in range(rounds + 1):
            if gen == 0:
                # gen-0: full tile through the cache (all miss), asserted
                # bit-equal; this also pins per-cell fold cost
                mask = None
            else:
                chg3 = (cur[0] != prev_data[0]) | (cur[1] != prev_data[1])
                chg2 = cur[2].view(np.uint32) != prev_data[2].view(np.uint32)
                mask = chg3.any(-1) | chg2
                if not mask.any():
                    continue
            veg_c = bp.PackedBits(data=np.ascontiguousarray(cur[0]),
                                  patch_count=N_PATCHES)
            vbsh_c = bp.PackedBits(data=np.ascontiguousarray(cur[1]),
                                   patch_count=N_PATCHES)
            v2_c = cur[2]
            out, tim = t26.cached_fold(veg_c, vbsh_c, v2_c, svfb, cache,
                                       cell_mask=mask,
                                       outputs=prev_planes)
            # correctness rides every row: cached fold == fresh masked
            # fold of the same cells
            ref = sf.fold_svf(veg_c, vbsh_c, v2_c, svfb, cell_mask=mask,
                              outputs=prev_planes)
            ok = planes_equal(planes_of(out), planes_of(ref))
            assert ok, f"{name} gen{gen}: cached fold != direct masked fold"
            # C2 baseline: the SAME cells folded fresh with NO cache
            t0 = time.perf_counter()
            sf.fold_svf(veg_c, vbsh_c, v2_c, svfb, cell_mask=mask,
                        outputs=prev_planes)
            baseline_ms = (time.perf_counter() - t0) * 1e3
            cached_total_ms = (tim["fp_build_ms"] + tim["hit_lookup_ms"]
                               + tim["miss_fold_ms"] + tim["assemble_ms"]
                               + tim["insert_ms"])
            if tim["n_misses"]:
                per_cell_us = tim["miss_fold_ms"] / tim["n_misses"] * 1e3
            raw_save_ms = (tim["n_hits"] * per_cell_us / 1e3
                           if per_cell_us else None)
            row = {
                "kind": "hitrate_gen", "scene": name, "gen": gen,
                "cells": rows * cols,
                "dirty": int(mask.sum()) if mask is not None else rows * cols,
                "ok": ok,
                "n_hits": tim["n_hits"], "n_misses": tim["n_misses"],
                "n_classes": len(cache),
                "cache_mib": round(cache.approx_bytes() / 2**20, 2),
                "fp_build_ms": round(tim["fp_build_ms"], 3),
                "hit_lookup_ms": round(tim["hit_lookup_ms"], 3),
                "miss_fold_ms": round(tim["miss_fold_ms"], 2),
                "assemble_ms": round(tim["assemble_ms"], 3),
                "insert_ms": round(tim["insert_ms"], 3),
                "cached_total_ms": round(cached_total_ms, 2),
                "baseline_masked_fold_ms": round(baseline_ms, 2),
                "speedup_vs_masked_baseline": round(
                    baseline_ms / cached_total_ms, 3) if cached_total_ms else None,
                "per_cell_fold_us": (round(per_cell_us, 3)
                                     if per_cell_us else None),
                "raw_hit_save_ms": (round(raw_save_ms, 2) if raw_save_ms
                                    is not None else None),
                "walls_provisional": True,
                **env_row(),
            }
            jsonl(jf, row)
            print(json.dumps({k: row[k] for k in (
                "scene", "gen", "ok", "dirty", "n_hits", "n_misses",
                "n_classes", "cached_total_ms",
                "baseline_masked_fold_ms",
                "speedup_vs_masked_baseline") if k in row}))
            prev_planes = planes_of(out)
            if gen < rounds:
                # the caller's carried state is THIS gen's scene; the next
                # gen's dirty mask = diff(edit(S_g), S_g)
                prev_data = (cur[0].copy(), cur[1].copy(), cur[2].copy())
                specs = block_copies(rng, (rows, cols))
                cur = apply_edit(cur, specs)
    print(f"\nHITRATE rows -> {jf}")
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "hitrate"
    if mode == "hitrate":
        return run_hitrate()
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
