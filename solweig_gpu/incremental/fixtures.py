# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic scientific fixtures for the incremental design tool.

The repository does not redistribute site rasters. This module therefore
generates fixture *states* deterministically from a locally supplied source
site (``Input_subset`` by default) and records a manifest with content hashes
so every state is reproducible and verifiable.

Fixture states cover the four editable tree operations:

* ``baseline``      - the supplied vegetation raster, unmodified;
* ``add``           - baseline plus one documented tree;
* ``move``          - the same tree displaced to a second open location;
* ``resize``        - the same tree with larger crown and height;
* ``delete``        - baseline again (restoration target).

Tree rasterization follows the full-pipeline convention: the vegetation
raster stores canopy-top height above the ground surface (metres), crown
footprints are discs, and overlapping canopies combine with ``maximum``.
The trunk zone is derived downstream (``vegdem2 = veg * 0.25 + dem``), so no
separate trunk raster is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio

from .geometry import RasterGrid, TreeSpec

FIXTURE_SCHEMA_VERSION = 1

# Deterministic editable tree used across fixture states. Values stay inside
# the documented validation ranges (height [3, 40] m, diameter [1, 30] m).
FIXTURE_TREE_BASE = TreeSpec(
    tree_id="fixture_tree_A",
    x_m=0.0,  # filled in relative to the source site at generation time
    y_m=0.0,
    height_m=10.0,
    canopy_radius_m=5.5,
    trunk_ratio=0.25,
    transmissivity=0.03,
)

FIXTURE_TREE_RESIZED = TreeSpec(
    tree_id="fixture_tree_A",
    x_m=0.0,
    y_m=0.0,
    height_m=18.0,
    canopy_radius_m=8.0,
    trunk_ratio=0.25,
    transmissivity=0.03,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def rasterize_tree(
    vegetation: np.ndarray,
    tree: TreeSpec,
    grid: RasterGrid,
    *,
    in_place: bool = False,
) -> np.ndarray:
    """Paint one tree crown into a vegetation-above-ground raster.

    Returns a copy unless ``in_place`` is true. Crown cells take the canopy
    top height; overlaps with existing vegetation combine with ``maximum``,
    matching the full-pipeline input-construction rule.
    """
    if vegetation.shape != (grid.rows, grid.cols):
        raise ValueError(
            f"vegetation shape {vegetation.shape} does not match grid "
            f"({grid.rows}, {grid.cols})"
        )
    out = vegetation if in_place else vegetation.copy()
    radius = tree.canopy_radius_m
    min_x = tree.x_m - radius
    max_x = tree.x_m + radius
    min_y = tree.y_m - radius
    max_y = tree.y_m + radius
    window = grid.world_bounds_to_window(
        min_x_m=min_x, min_y_m=min_y, max_x_m=max_x, max_y_m=max_y
    )
    if window.is_empty:
        return out
    for row in range(window.row_start, window.row_stop):
        for col in range(window.col_start, window.col_stop):
            centre_x = grid.origin_x_m + (col + 0.5) * grid.pixel_size_m
            centre_y = grid.origin_y_m - (row + 0.5) * grid.pixel_size_m
            if (centre_x - tree.x_m) ** 2 + (centre_y - tree.y_m) ** 2 <= radius**2:
                if tree.height_m > out[row, col]:
                    out[row, col] = tree.height_m
    return out


def find_open_cell(
    vegetation: np.ndarray,
    building_dsm: np.ndarray,
    dem: np.ndarray,
    grid: RasterGrid,
    *,
    clearance_m: float = 40.0,
    preference: Callable[[int, int], float] | None = None,
) -> tuple[float, float]:
    """Deterministically find an open-terrain cell near the site centre.

    A cell qualifies when no building (<= 2 m above ground) and no existing
    vegetation lies within the clearance radius. Candidates are ranked by
    distance from the site centre (smallest first) so the result is stable.
    """
    clearance_px = int(np.ceil(clearance_m / grid.pixel_size_m))
    rows, cols = grid.rows, grid.cols
    centre_row = rows / 2.0
    centre_col = cols / 2.0
    best: tuple[float, int, int] | None = None
    for row in range(clearance_px, rows - clearance_px):
        for col in range(clearance_px, cols - clearance_px):
            region_veg = vegetation[
                row - clearance_px : row + clearance_px + 1,
                col - clearance_px : col + clearance_px + 1,
            ]
            if region_veg.max() > 0:
                continue
            region_bld = building_dsm[
                row - clearance_px : row + clearance_px + 1,
                col - clearance_px : col + clearance_px + 1,
            ]
            region_dem = dem[
                row - clearance_px : row + clearance_px + 1,
                col - clearance_px : col + clearance_px + 1,
            ]
            if np.any(region_bld - region_dem > 2.0):
                continue
            distance = float((row - centre_row) ** 2 + (col - centre_col) ** 2)
            rank = preference(row, col) if preference is not None else 0.0
            key = (distance + rank, row, col)
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError("no open-terrain cell found; increase clearance radius")
    _, row, col = best
    x_m = grid.origin_x_m + (col + 0.5) * grid.pixel_size_m
    y_m = grid.origin_y_m - (row + 0.5) * grid.pixel_size_m
    return x_m, y_m


def _write_state(
    target_dir: Path,
    name: str,
    vegetation: np.ndarray,
    profile: dict,
) -> dict:
    path = target_dir / f"trees_{name}.tif"
    profile = dict(profile)
    profile.update(count=1, dtype="float32", nodata=0.0)
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(vegetation.astype("float32"), 1)
    return {"file": path.name, "sha256": _sha256(path)}


def build_fixture_states(
    source_dir: Path,
    target_dir: Path,
    *,
    tree: TreeSpec | None = None,
    move_offset_m: tuple[float, float] = (200.0, 0.0),
) -> dict:
    """Generate all fixture tree states and a verified manifest.

    Parameters
    ----------
    source_dir:
        Directory containing ``Trees.tif``, ``Building_DSM.tif``, ``DEM.tif``.
    target_dir:
        Output directory; created if missing.
    tree:
        Override the default fixture tree (position is replaced by a
        deterministic open-terrain cell).
    move_offset_m:
        Displacement applied to derive the ``move`` state's new position.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_dir / "Trees.tif") as dataset:
        vegetation = dataset.read(1).astype("float32")
        profile = dataset.profile
        transform = dataset.transform
    with rasterio.open(source_dir / "Building_DSM.tif") as dataset:
        building_dsm = dataset.read(1)
    with rasterio.open(source_dir / "DEM.tif") as dataset:
        dem = dataset.read(1)

    rows, cols = vegetation.shape
    grid = RasterGrid(
        rows=rows,
        cols=cols,
        pixel_size_m=float(abs(transform.a)),
        origin_x_m=float(transform.c),
        origin_y_m=float(transform.f),
    )

    base_tree = tree if tree is not None else FIXTURE_TREE_BASE
    tree_x, tree_y = find_open_cell(vegetation, building_dsm, dem, grid)
    # Shrink the documented 200 m displacement on small synthetic sites so the
    # move target stays inside the raster while remaining far from the origin.
    extent_m = min(rows, cols) * grid.pixel_size_m
    offset_east = min(move_offset_m[0], 0.35 * extent_m)
    offset_north = min(move_offset_m[1], 0.35 * extent_m)
    add_tree = TreeSpec(
        tree_id=base_tree.tree_id,
        x_m=tree_x,
        y_m=tree_y,
        height_m=base_tree.height_m,
        canopy_radius_m=base_tree.canopy_radius_m,
        trunk_ratio=base_tree.trunk_ratio,
        transmissivity=base_tree.transmissivity,
    )
    moved_tree = TreeSpec(
        tree_id=add_tree.tree_id,
        x_m=add_tree.x_m + offset_east,
        y_m=add_tree.y_m + offset_north,
        height_m=add_tree.height_m,
        canopy_radius_m=add_tree.canopy_radius_m,
        trunk_ratio=add_tree.trunk_ratio,
        transmissivity=add_tree.transmissivity,
    )
    resized_tree = TreeSpec(
        tree_id=FIXTURE_TREE_RESIZED.tree_id,
        x_m=add_tree.x_m,
        y_m=add_tree.y_m,
        height_m=FIXTURE_TREE_RESIZED.height_m,
        canopy_radius_m=FIXTURE_TREE_RESIZED.canopy_radius_m,
        trunk_ratio=add_tree.trunk_ratio,
        transmissivity=add_tree.transmissivity,
    )

    add_state = rasterize_tree(vegetation, add_tree, grid)
    move_state = rasterize_tree(vegetation, moved_tree, grid)
    resize_state = rasterize_tree(vegetation, resized_tree, grid)

    manifest: dict = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "grid": {
            "rows": rows,
            "cols": cols,
            "pixel_size_m": grid.pixel_size_m,
            "origin_x_m": grid.origin_x_m,
            "origin_y_m": grid.origin_y_m,
        },
        "source": {
            "trees": _sha256(source_dir / "Trees.tif"),
            "building_dsm": _sha256(source_dir / "Building_DSM.tif"),
            "dem": _sha256(source_dir / "DEM.tif"),
        },
        "trees": {
            "add": asdict(add_tree),
            "move_from": asdict(add_tree),
            "move_to": asdict(moved_tree),
            "resize": asdict(resized_tree),
        },
        "states": {},
    }

    states = {
        "baseline": vegetation,
        "add": add_state,
        "move": move_state,
        "resize": resize_state,
        "delete": vegetation,
    }
    for name, state in states.items():
        manifest["states"][name] = _write_state(target_dir, name, state, profile)

    manifest["invariants"] = {
        "delete_equals_baseline": manifest["states"]["delete"]["sha256"]
        == manifest["states"]["baseline"]["sha256"],
        "add_changes_cells": bool(np.any(add_state != vegetation)),
        "move_changes_cells": bool(np.any(move_state != vegetation)),
        "resize_changes_cells": bool(np.any(resize_state != vegetation)),
    }
    if not all(manifest["invariants"].values()):
        raise ValueError(f"fixture invariants failed: {manifest['invariants']}")

    partial = target_dir / "manifest.partial.json"
    final = target_dir / "manifest.json"
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    partial.replace(final)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Input_subset"),
        help="Directory containing the source site rasters",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("fixtures/site_500"),
        help="Output directory for fixture states and manifest",
    )
    args = parser.parse_args()
    manifest = build_fixture_states(args.source, args.target)
    print(json.dumps(manifest["invariants"], indent=2))
    print(f"manifest written to {(args.target / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
