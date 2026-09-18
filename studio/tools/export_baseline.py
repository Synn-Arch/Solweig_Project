#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Export a completed baseline scenario result into a site cache.

Populates ``<cache_dir>/baseline_results/`` (``utci``/``tmrt`` float32
``.npy`` planes plus ``metadata.json``) from a scenario's published result.
With stored baseline outputs the server materializes every new scenario's
baseline instantly (``materialize_baseline_result``) instead of paying a
full-tile solve per scenario at connect time.

Usage:
    python tools/export_baseline.py \
        --result-dir /state/scenarios/<scenario_id>/results/0 \
        --cache-dir /path/to/site-cache/<site_id> \
        [--variables utci tmrt]

The result directory is the one holding ``manifest.json`` + ``payload.bin``
(as served under ``/state/scenarios/<id>/results/<version>/``). Works on a
directory copied out of the container (``docker cp``) — the site cache is
usually mounted read-only in the deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from solweig_gpu.server.patch_codec import decode_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="directory holding manifest.json + payload.bin of the baseline result",
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="site cache directory (<site_id>)"
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["utci", "tmrt"],
        help="variables to store (default: utci tmrt)",
    )
    args = parser.parse_args()

    manifest_path = args.result_dir / "manifest.json"
    payload_path = args.result_dir / "payload.bin"
    manifest = json.loads(manifest_path.read_text())
    payload = payload_path.read_bytes()

    arrays = decode_payload(manifest, payload)

    available = [name for name in args.variables if name in arrays]
    if not available:
        print(
            f"result carries none of {args.variables} "
            f"(has {sorted(arrays)})",
            file=sys.stderr,
        )
        return 2

    baseline_dir = args.cache_dir / "baseline_results"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    time_steps = int(arrays[available[0]].shape[0])
    for name in available:
        array = np.ascontiguousarray(arrays[name], dtype=np.float32)
        np.save(baseline_dir / f"{name}.f32.npy", array)
        print(f"wrote {baseline_dir / (name + '.f32.npy')} shape={array.shape}")
    (baseline_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variables": available,
                "time_steps": time_steps,
                "source": f"scenario export {args.result_dir}",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {baseline_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
