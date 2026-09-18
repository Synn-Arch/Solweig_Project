#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Benchmark packed visibility primitives on a representative 500×500 tile."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from solweig_gpu.incremental.bitmask import pack_visibility, weighted_sum_from_packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--cols", type=int, default=500)
    parser.add_argument("--patches", type=int, default=153)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dense = rng.integers(0, 2, size=(args.rows, args.cols, args.patches), dtype=np.uint8)
    weights = rng.random(args.patches, dtype=np.float32)

    started = time.perf_counter()
    packed = pack_visibility(dense)
    pack_seconds = time.perf_counter() - started

    started = time.perf_counter()
    packed_sum = weighted_sum_from_packed(packed, weights)
    reduction_seconds = time.perf_counter() - started

    started = time.perf_counter()
    dense_sum = np.sum(dense.astype(np.float32) * weights, axis=-1)
    dense_seconds = time.perf_counter() - started

    report = {
        "shape": [args.rows, args.cols, args.patches],
        "dense_uint8_bytes": int(dense.nbytes),
        "dense_float32_equivalent_bytes": int(dense.size * 4),
        "packed_bytes": int(packed.nbytes),
        "float32_to_packed_ratio": float((dense.size * 4) / packed.nbytes),
        "pack_seconds": pack_seconds,
        "packed_weighted_sum_seconds": reduction_seconds,
        "dense_weighted_sum_seconds": dense_seconds,
        "maximum_absolute_error": float(np.max(np.abs(packed_sum - dense_sum))),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
