# SPDX-License-Identifier: GPL-3.0-only
"""T07 fold bench: full-site and affected-chunks-only fold timings.

Measures (secondary metric — never traded for bits):

* the ORIGINAL svf_calculator call once (context: what the fold replaces
  on the veg side) — torch is imported for this leg ONLY (bench file,
  T06 bench_sparse_march.py precedent; the runner modules stay
  torch-free);
* pack of the two producer cubes;
* full-site fold: serial and 2/4/8 threads, n >= 5 each;
* affected-chunks-only fold (10% random mask + caller base planes):
  serial and 4 threads, n >= 5;
* cold/warm JIT split via a FRESH numba cache dir in a subprocess.

Usage:
    .venv/bin/python tests/ultrafast/bench_svf_fold.py \
        [--artifacts DIR] [--runs N] [--cold-jit]

Raw JSON lands in the artifacts dir (never /tmp-only).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

import numpy as np

from solweig_core import bitplanes as bp
from solweig_core import sparse_work as sw
from solweig_core.numba_cpu import svf_fold

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
DEFAULT_ARTIFACTS = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t07"
)


def site_scene():
    from solweig_gpu.incremental.cache import SiteCache

    cache = SiteCache.load(SITE_500_CACHE)
    canopy = np.ascontiguousarray(np.asarray(cache.tree_base, dtype=np.float32))
    a = np.array(cache.building_dsm, dtype=np.float32)
    dem = np.array(cache.dem, dtype=np.float32)
    inputs = sw.compose_march_inputs(a, dem, canopy)
    return cache, inputs


def oracle_outputs(inputs, scale):
    import torch
    from solweig_gpu.shadow import svf_calculator

    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    t0 = time.perf_counter()
    ret = svf_calculator(
        patch_option=2,
        amaxvalue=torch.tensor(np.float32(inputs.amaxvalue)),
        a=t(inputs.a),
        vegdem=t(inputs.vegdsm),
        vegdem2=t(inputs.vegdsm2),
        bush=t(inputs.bush),
        scale=float(scale),
    )
    elapsed = time.perf_counter() - t0
    names = (
        "svf", "svfaveg", "svfE", "svfEaveg", "svfEveg", "svfN",
        "svfNaveg", "svfNveg", "svfS", "svfSaveg", "svfSveg", "svfveg",
        "svfW", "svfWaveg", "svfWveg", "vegshmat", "vbshvegshmat",
        "shmat", "SVFtotal",
    )
    return dict(zip(names, [x.numpy() for x in ret])), elapsed


def timed(fn, n):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "n": len(samples),
        "median_s": round(statistics.median(samples), 6),
        "min_s": round(min(samples), 6),
        "max_s": round(max(samples), 6),
        "spread_s": round(max(samples) - min(samples), 6),
        "samples_s": [round(x, 6) for x in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--cold-jit", action="store_true")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    if args.cold_jit:
        # fresh numba cache dir -> first fold = cold compile+run
        cold_dir = args.artifacts / "numba_cold_cache"
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = str(cold_dir)
        code = (
            "import time, json, numpy as np, sys; "
            "sys.path.insert(0, {repo!r}); "
            "from solweig_core import bitplanes as bp; "
            "from solweig_core.numba_cpu import svf_fold; "
            "rng = np.random.default_rng(1); "
            "veg = (rng.random((64, 64, 153)) < 0.3).astype(np.float32); "
            "vb = (rng.random((64, 64, 153)) < 0.3).astype(np.float32); "
            "v2 = np.zeros((64, 64), dtype=np.float32); "
            "svf = np.full((64, 64), 0.5, dtype=np.float32); "
            "t0 = time.perf_counter(); "
            "svf_fold.fold_svf(bp.pack_bits(veg), bp.pack_bits(vb), v2, svf); "
            "cold = time.perf_counter() - t0; "
            "t0 = time.perf_counter(); "
            "svf_fold.fold_svf(bp.pack_bits(veg), bp.pack_bits(vb), v2, svf); "
            "warm = time.perf_counter() - t0; "
            "print(json.dumps({{'cold_s': cold, 'warm_s': warm}}))"
        ).format(repo=str(REPO_ROOT))
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
        )
        print(out.stdout.strip())
        if out.returncode != 0:
            print(out.stderr, file=sys.stderr)
            raise SystemExit(1)
        (args.artifacts / "fold_cold_jit.json").write_text(
            out.stdout.strip() + "\n"
        )
        return

    if not SITE_500_CACHE.is_dir():
        raise SystemExit(f"site cache {SITE_500_CACHE} not present")

    load_before = os.getloadavg()
    cache, inputs = site_scene()
    oracle, oracle_s = oracle_outputs(inputs, 1.0 / float(cache.pixel_size_m))

    t0 = time.perf_counter()
    packed_veg = bp.pack_bits(oracle["vegshmat"])
    packed_vbsh = bp.pack_bits(oracle["vbshvegshmat"])
    pack_s = time.perf_counter() - t0

    shape = oracle["svf"].shape
    rows, cols = shape
    record = {
        "scene": "site_500",
        "shape": [int(rows), int(cols)],
        "host_load_before": [round(x, 2) for x in load_before],
        "oracle_svf_calculator_s": round(oracle_s, 3),
        "pack_both_cubes_s": round(pack_s, 6),
        "threads": {},
    }

    for threads in (1, 2, 4, 8):
        record["threads"][threads] = timed(
            lambda: svf_fold.fold_svf(
                packed_veg, packed_vbsh, inputs.vegdsm2, oracle["svf"],
                n_threads=threads,
            ),
            args.runs,
        )

    rng = np.random.default_rng(71)
    mask = rng.random(shape) < 0.10
    base = {
        name: np.full(shape, np.float32(-7.0), dtype=np.float32)
        for name in svf_fold.FOLD_OUTPUT_NAMES
    }
    record["affected_10pct"] = {
        "masked_cells": int(mask.sum()),
        "serial": timed(
            lambda: svf_fold.fold_svf(
                packed_veg, packed_vbsh, inputs.vegdsm2, oracle["svf"],
                cell_mask=mask, outputs=base,
            ),
            args.runs,
        ),
        "threads4": timed(
            lambda: svf_fold.fold_svf(
                packed_veg, packed_vbsh, inputs.vegdsm2, oracle["svf"],
                cell_mask=mask, outputs=base, n_threads=4,
            ),
            args.runs,
        ),
    }
    record["host_load_after"] = [round(x, 2) for x in os.getloadavg()]

    out_path = args.artifacts / "fold_perf.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
