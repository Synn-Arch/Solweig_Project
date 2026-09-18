# SPDX-License-Identifier: GPL-3.0-only
"""Convert dense SVF shadow stacks to bit-packed ``.npy`` caches.

The SVF stage writes three binary patch stacks (``shadowmat``,
``vegshadowmat``, ``vbshmat``) as float32 to ``shadowmats_<tile>.npz``
(~146 MiB each at 500x500x153). This command converts them to packed
uint8 arrays (~5 MiB each, 20 bytes/pixel) plus an authoritative JSON
manifest recording shapes, patch count, bit order, polarity, and source
hashes so a warm worker can validate before use.

Usage::

    python -m solweig_gpu.incremental.pack_svf \
        --npz processed_inputs/SVF/shadowmats_0_0.npz \
        --out-dir processed_inputs/SVF/packed_0_0 \
        [--validate-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .bitmask import (
    PACKED_STACK_NAMES,
    SHADOW_STACK_POLARITY,
    PackedVisibility,
    bytes_per_pixel,
    pack_visibility,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_binary(array: np.ndarray, name: str) -> np.ndarray:
    unique = np.unique(array)
    if not set(unique.tolist()) <= {0.0, 1.0}:
        raise ValueError(
            f"stack {name!r} is not binary {{0, 1}} (found values "
            f"{unique[:5].tolist()}{'...' if unique.size > 5 else ''}); "
            "refusing to pack lossy data"
        )
    return array


def convert_npz(npz_path: Path, out_dir: Path, *, validate_only: bool = False) -> dict:
    """Convert one ``shadowmats_<tile>.npz`` to packed arrays + manifest.

    Returns the manifest dict. With ``validate_only`` the stacks are checked
    (binary, finite, shared shape/patch axis) and nothing is written.
    """
    with np.load(npz_path) as archive:
        missing = [key for key in PACKED_STACK_NAMES if key not in archive.files]
        if missing:
            raise ValueError(f"{npz_path} is missing stacks: {missing}")
        stacks = {key: _check_binary(archive[key], key) for key in PACKED_STACK_NAMES}

    shapes = {tuple(value.shape) for value in stacks.values()}
    if len(shapes) != 1:
        raise ValueError(f"stacks disagree on shape: {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) != 3:
        raise ValueError(f"expected (rows, cols, patches) stacks, got shape {shape}")
    rows, cols, patch_count = shape
    if patch_count <= 0:
        raise ValueError("patch axis must be positive")

    manifest = {
        "schema_version": 1,
        "source_npz": npz_path.name,
        "source_sha256": sha256_file(npz_path),
        "rows": int(rows),
        "cols": int(cols),
        "patch_count": int(patch_count),
        "bytes_per_pixel": bytes_per_pixel(patch_count),
        "bitorder": "little",
        "polarity": SHADOW_STACK_POLARITY,
        "stacks": {},
    }

    if validate_only:
        return manifest

    out_dir.mkdir(parents=True, exist_ok=True)
    for source_key, stem in PACKED_STACK_NAMES.items():
        packed: PackedVisibility = pack_visibility(stacks[source_key])
        out_path = out_dir / f"{stem}.npy"
        np.save(out_path, packed.data)
        manifest["stacks"][source_key] = {
            "file": out_path.name,
            "sha256": sha256_file(out_path),
            "nbytes": int(packed.nbytes),
            "dense_nbytes": int(stacks[source_key].nbytes),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_packed_dir(out_dir: Path) -> dict[str, PackedVisibility]:
    """Load a packed directory written by :func:`convert_npz`.

    Validates every packed file against the manifest checksum before
    returning memory-mapped :class:`PackedVisibility` stacks.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text())
    patch_count = manifest["patch_count"]
    bitorder = manifest["bitorder"]
    stacks: dict[str, PackedVisibility] = {}
    for source_key, entry in manifest["stacks"].items():
        path = out_dir / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {path}")
        data = np.load(path, mmap_mode="r")
        stacks[source_key] = PackedVisibility(
            data=data, patch_count=patch_count, bitorder=bitorder
        )
    return stacks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npz", type=Path, required=True,
                        help="path to shadowmats_<tile>.npz")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output directory for packed .npy files + manifest.json")
    parser.add_argument("--validate-only", action="store_true",
                        help="check the stacks without writing anything")
    args = parser.parse_args(argv)

    manifest = convert_npz(args.npz, args.out_dir, validate_only=args.validate_only)
    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
