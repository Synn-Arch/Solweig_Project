#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the CUDA SVF fold parity pins (TASKS T12) as DATA.

Runs on the GPU host (numba present; torch NOT needed). Computes the
CANONICAL CPU fold (solweig_core/numba_cpu/svf_fold.py — the frozen T07
reference, bit-equal to the original svf_calculator fold) over
deterministic synthetic packed states:

* random bit cubes at several densities (the fold consumes PACKED bits —
  any pattern is valid input; uniform random stresses every accumulator
  path harder than the natural correlated cubes);
* degenerate cubes (all-zero veg bits, all-one both);
* special-value planes (vegdem2 with +0.0/-0.0/NaN blocks for the ``last``
  predicate; svf_building with NaN/±0/sub-unit/super-unit lanes);
* a masked case pinning the affected-chunk-only contract (masked cells
  equal the full fold, unmasked cells bit-identical to the caller's base).

The frozen constant tables are serialized FROM THE MODULE and the pin file
records BOTH the module digest and an INDEPENDENT digest recomputed from
the serialized arrays — any single-bit table mutation breaks the recorded
equality (tests re-verify both).

Outputs raw uint32 hex / packed-byte hex to native/cuda/data/fold_pins.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from solweig_core import bitplanes as bp  # noqa: E402
from solweig_core.numba_cpu import svf_fold as sf  # noqa: E402

OUT = REPO_ROOT / "native" / "cuda" / "data"

F32 = np.float32
U32 = np.uint32
P = 153


def hexes(arr_f32: np.ndarray) -> list[str]:
    return [f"{int(b):08x}" for b in
            np.ascontiguousarray(arr_f32, dtype=np.float32).view(U32).ravel()]


def bytes_hex(arr_u8: np.ndarray) -> list[str]:
    return [f"{int(b):02x}" for b in
            np.ascontiguousarray(arr_u8, dtype=np.uint8).ravel()]


def tables_payload() -> dict:
    """Serialize the frozen tables FROM THE MODULE + digest cross-check."""
    tables = {
        "w_iso": sf.W_ISO.copy(),
        "w_aniso": sf.W_ANISO.copy(),
        "ring": sf.PATCH_RING.copy(),
        "na": sf.N_ANNULUS.copy(),
        "azimuth": sf.PATCH_AZIMUTH.copy(),
        "dir_e": sf.DIRECTION_E.view(np.int8).copy(),
        "dir_s": sf.DIRECTION_S.view(np.int8).copy(),
        "dir_w": sf.DIRECTION_W.view(np.int8).copy(),
        "dir_n": sf.DIRECTION_N.view(np.int8).copy(),
        "last_const": np.float32(sf.LAST_CONST),
        "one_minus_trans": np.float32(1.0) - np.float32(sf.TRANS),
    }
    # independent digest over the SERIALIZED arrays (exact serialization of
    # frozen_tables_digest): proves the JSON tables are the module's bits.
    parts = [
        str(P),
        ",".join(str(int(x)) for x in tables["na"]),
        ",".join(f"{int(b):08x}" for b in tables["w_iso"].view(U32).ravel()),
        ",".join(f"{int(b):08x}" for b in tables["w_aniso"].view(U32).ravel()),
        ",".join(f"{int(b):08x}" for b in
                 tables["azimuth"].view(U32).ravel()),
        ",".join(str(int(x)) for x in tables["ring"]),
        f"{int(np.array([tables['last_const']], dtype=F32).view(U32)[0]):08x}",
        f"{int(np.array([sf.TRANS], dtype=F32).view(U32)[0]):08x}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    assert digest == sf.FOLD_CONSTANTS_SHA256, (
        f"serialized tables digest {digest} != module "
        f"{sf.FOLD_CONSTANTS_SHA256}"
    )
    return {
        "n_patches": P,
        "bytes_per_patch": int(bp.bytes_per_patch(P)),
        "ring_slots": 12,
        "w_iso": hexes(tables["w_iso"]),
        "w_aniso": hexes(tables["w_aniso"]),
        "ring": [int(x) for x in tables["ring"]],
        "na": [int(x) for x in tables["na"]],
        "azimuth": hexes(tables["azimuth"]),
        "dir_e": [int(x) for x in tables["dir_e"]],
        "dir_s": [int(x) for x in tables["dir_s"]],
        "dir_w": [int(x) for x in tables["dir_w"]],
        "dir_n": [int(x) for x in tables["dir_n"]],
        "last_bits": f"{int(np.array([tables['last_const']], dtype=F32).view(U32)[0]):08x}",
        "trans_bits": f"{int(np.array([sf.TRANS], dtype=F32).view(U32)[0]):08x}",
        "one_minus_trans_bits": f"{int(np.array([tables['one_minus_trans']], dtype=F32).view(U32)[0]):08x}",
        "fold_constants_sha256_module": sf.FOLD_CONSTANTS_SHA256,
        "fold_constants_sha256_serialized": digest,
    }


def rand_bits(rows: int, cols: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols, P)) < p


def special_vegdem2(rows: int, cols: int, seed: int) -> np.ndarray:
    """+0.0 block, -0.0 cells, a NaN cell, normal lanes (last predicate)."""
    rng = np.random.default_rng(seed)
    plane = rng.uniform(0.5, 4.0, (rows, cols)).astype(F32)
    plane[0:3, 0:5] = F32(0.0)
    plane[2, 3] = F32(-0.0)
    plane[min(9, rows - 1), min(70, cols - 1)] = F32(-0.0)
    plane[1, 1] = np.float32("nan")
    return plane


def special_svf(rows: int, cols: int, seed: int) -> np.ndarray:
    """NaN/±0/sub-unit/super-unit lanes (clamps + svftotal edge paths)."""
    rng = np.random.default_rng(seed)
    plane = rng.uniform(0.0, 1.3, (rows, cols)).astype(F32)
    plane[0, 0] = np.float32("nan")
    plane[0, 1] = F32(-0.0)
    plane[1, 0] = F32(5.0)
    plane[1, 1] = F32(-2.0)
    return plane


def run_reference(veg_bits, vbsh_bits, vegdem2, svf_building, *,
                  cell_mask=None, outputs=None):
    packed_veg = bp.pack_bits(np.ascontiguousarray(veg_bits))
    packed_vbsh = bp.pack_bits(np.ascontiguousarray(vbsh_bits))
    result = sf.fold_svf(packed_veg, packed_vbsh, vegdem2, svf_building,
                         cell_mask=cell_mask, outputs=outputs)
    return result, (packed_veg.data.copy(), packed_vbsh.data.copy())


def case_entry(name: str, veg_bits, vbsh_bits, vegdem2, svf_building, *,
               cell_mask=None, out_base_stack=None):
    kwargs = {}
    if cell_mask is not None:
        kwargs["cell_mask"] = cell_mask
        outputs = {n: out_base_stack[:, :, k]
                   for k, n in enumerate(sf.FOLD_OUTPUT_NAMES)}
        kwargs["outputs"] = outputs
    result, (veg_bytes, vbsh_bytes) = run_reference(
        veg_bits, vbsh_bits, vegdem2, svf_building, **kwargs)
    rows, cols = vegdem2.shape
    entry = {
        "name": name, "rows": rows, "cols": cols,
        "veg_bytes": bytes_hex(veg_bytes),
        "vbsh_bytes": bytes_hex(vbsh_bytes),
        "vegdem2": hexes(vegdem2), "svf_building": hexes(svf_building),
    }
    if cell_mask is not None:
        entry["mask"] = [int(b) for b in
                         np.ascontiguousarray(cell_mask, dtype=np.uint8).ravel()]
        entry["out_base"] = hexes(out_base_stack)
    for k, n in enumerate(sf.FOLD_OUTPUT_NAMES):
        entry[n] = hexes(result[k])
    return entry


def gen_cases() -> list[dict]:
    cases = []
    # density sweep (uniform random bits — every accumulator path live)
    for name, p, seed in (("rand_p10", 0.10, 1), ("rand_p30", 0.30, 2),
                          ("rand_p50", 0.50, 3), ("rand_p70", 0.70, 4),
                          ("rand_p90", 0.90, 5)):
        rows, cols = 33, 61
        veg = rand_bits(rows, cols, p, seed)
        vbsh = rand_bits(rows, cols, 0.5, seed + 100)
        vegdem2 = np.random.default_rng(seed + 200).uniform(
            0.0, 3.0, (rows, cols)).astype(F32)
        vegdem2[5:9, 10:20] = F32(0.0)  # last-correction lanes
        svf = np.random.default_rng(seed + 300).uniform(
            0.0, 1.0, (rows, cols)).astype(F32)
        cases.append(case_entry(name, veg, vbsh, vegdem2, svf))
    # degenerate cubes
    rows, cols = 24, 40
    veg = rand_bits(rows, cols, 0.5, 6)
    vbsh = rand_bits(rows, cols, 0.5, 7)
    vegdem2 = np.full((rows, cols), F32(1.5))
    svf = np.full((rows, cols), F32(0.7))
    cases.append(case_entry("zeros_veg", np.zeros((rows, cols, P), dtype=bool),
                            vbsh, vegdem2, svf))
    cases.append(case_entry("ones_both", np.ones((rows, cols, P), dtype=bool),
                            np.ones((rows, cols, P), dtype=bool),
                            special_vegdem2(rows, cols, 8),
                            special_svf(rows, cols, 9)))
    # special-value planes
    rows, cols = 33, 61
    cases.append(case_entry(
        "special_planes", rand_bits(rows, cols, 0.5, 10),
        rand_bits(rows, cols, 0.4, 11),
        special_vegdem2(rows, cols, 12), special_svf(rows, cols, 13)))
    # wider plane
    rows, cols = 64, 96
    cases.append(case_entry(
        "wide_64x96", rand_bits(rows, cols, 0.5, 14),
        rand_bits(rows, cols, 0.5, 15),
        special_vegdem2(rows, cols, 16), special_svf(rows, cols, 17)))
    # masked case: 35% of cells folded over a random (NaN-bearing) base
    rows, cols = 40, 80
    rng = np.random.default_rng(18)
    base = rng.uniform(-1.0, 2.0, (rows, cols, 11)).astype(F32)
    base[3, 4, 2] = np.float32("nan")
    base[5, 6, 0] = F32(-0.0)
    mask = np.random.default_rng(19).random((rows, cols)) < 0.35
    cases.append(case_entry(
        "masked_affine", rand_bits(rows, cols, 0.6, 20),
        rand_bits(rows, cols, 0.5, 21),
        special_vegdem2(rows, cols, 22), special_svf(rows, cols, 23),
        cell_mask=mask, out_base_stack=base))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tables = tables_payload()
    cases = gen_cases()
    path = args.out or (OUT / "fold_pins.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "sw-cuda-fold-pins/1",
        "generator": "native/cuda/tools/gen_fold_pins.py",
        "reference": "solweig_core/numba_cpu/svf_fold.py (canonical CPU)",
        "numpy": np.__version__,
        "tables": tables,
        "cases": cases,
    }, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
