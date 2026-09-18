"""T00 synthetic fixture generators (code only; traces generated in T01).

Writes small synthetic scenes as raw .npy planes + a spec JSON into the
approved artifact root (never the repository tree). These are inputs for
later primitive/march trace characterization, not scientific outputs.

    python benchmarks/ultrafast/make_synth_fixtures.py \
        --out-root /Users/alansynn/Workspace/solweig_ultrafast_artifacts/synth \
        --shape 40 80 [--pixel-size 4.0 --one-step]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def build_scene(rows: int, cols: int, pixel_size_m: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260907)  # frozen seed: fixtures must be reproducible
    dem = np.full((rows, cols), 0.5, dtype=np.float32)
    dem += rng.uniform(0.0, 0.2, (rows, cols)).astype(np.float32)
    dsm = dem.copy()
    # one off-center building block (non-square placement on purpose)
    br0, br1 = max(0, rows // 5), max(2, rows // 5 + 3)
    bc0, bc1 = max(0, cols // 3), max(4, cols // 3 + 5)
    dsm[br0:br1, bc0:bc1] += 8.0
    # one tree: canopy above ground + trunk ratio layout like Trees.tif pair
    veg = np.zeros((rows, cols), dtype=np.float32)
    tr0, tc0 = (2 * rows) // 3, (3 * cols) // 4
    r_m = 4.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    mask = ((yy - tr0) ** 2 + (xx - tc0) ** 2) * pixel_size_m**2 <= r_m**2
    veg[mask] = 10.0
    trunk = np.zeros((rows, cols), dtype=np.float32)
    trunk[tr0, tc0] = 1.0
    return {"dem": dem, "building_dsm": dsm - dem, "veg_canopy": veg, "veg_trunk": trunk}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--shape", nargs=2, type=int, default=[40, 80], metavar=("ROWS", "COLS"))
    ap.add_argument("--pixel-size", type=float, default=2.0)
    ap.add_argument("--one-step", action="store_true",
                    help="emit a single-timestep forcing stub (4 m one-step witness)")
    args = ap.parse_args()
    rows, cols = args.shape
    out = Path(args.out_root).resolve() / f"synth_{rows}x{cols}_px{args.pixel_size:g}m"
    out.mkdir(parents=True, exist_ok=True)
    planes = build_scene(rows, cols, args.pixel_size)
    written = {}
    for name, arr in planes.items():
        p = out / f"{name}.f32.npy"
        np.save(p, arr)
        written[name] = {"shape": list(arr.shape), "nbytes": p.stat().st_size}
    if args.one_step:
        forcing = np.array([[20.0, 50.0, 2.0, 800.0, 0.0]], dtype=np.float32)
        p = out / "forcing_one_step.f32.npy"
        np.save(p, forcing)
        written["forcing_one_step"] = {"shape": [1, 5], "nbytes": p.stat().st_size}
    (out / "spec.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 20260907,
        "rows": rows, "cols": cols,
        "pixel_size_m": args.pixel_size,
        "one_step": bool(args.one_step),
        "files": written,
    }, indent=2))
    print(f"wrote {len(written)} planes + spec.json under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
