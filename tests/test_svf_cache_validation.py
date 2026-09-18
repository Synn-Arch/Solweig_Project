# SPDX-License-Identifier: GPL-3.0-only
"""SVF cache extent validation.

The SVF cache is keyed by tile number only. A cache produced for a
different window of the source mosaic must never be silently reused:
``_svf_cache_exists`` must report False (so the wrapper recomputes) and
``load_cached_svf_outputs`` must refuse it. Regression test for the
stale-foreign-cache incident recorded in the incremental WORKLOG.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from solweig_gpu.utci_process import (
    _svf_cache_exists,
    load_cached_svf_outputs,
)

ROWS, COLS = 4, 6
GT = (621734.71, 2.0, 0.0, 3354614.35, 0.0, -2.0)


def _write_tif(path: Path, gt, data: np.ndarray | None = None) -> None:
    if data is None:
        data = np.full((ROWS, COLS), 0.5, dtype=np.float32)
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        transform=Affine.from_gdal(*gt),
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def _build_cache(
    site_dir: Path,
    *,
    svf_gt=GT,
    svf_size=(ROWS, COLS),
) -> Path:
    """Create processed_inputs layout: Building_DSM/ + SVF/ cache files."""
    dsm_dir = site_dir / "Building_DSM"
    dsm_dir.mkdir(parents=True)
    _write_tif(dsm_dir / "Building_DSM_0_0.tif", GT)

    svf_dir = site_dir / "SVF"
    svf_dir.mkdir()
    rows, cols = svf_size
    _write_tif(svf_dir / "SkyViewFactor_0_0.tif", svf_gt, np.full((rows, cols), 0.7, np.float32))

    names = [
        "svf", "svfE", "svfS", "svfW", "svfN",
        "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
        "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
    ]
    with zipfile.ZipFile(svf_dir / "svfs_0_0.zip", "w") as zf:
        for name in names:
            data = np.full((rows, cols), 0.4, np.float32)
            with MemoryFile() as mem:
                with rasterio.open(
                    mem,
                    "w",
                    driver="GTiff",
                    width=cols,
                    height=rows,
                    count=1,
                    dtype="float32",
                    transform=Affine.from_gdal(*svf_gt),
                ) as dst:
                    dst.write(data, 1)
                zf.writestr(f"{name}.tif", mem.read())

    np.savez_compressed(
        svf_dir / "shadowmats_0_0.npz",
        **{key: np.ones((rows, cols, 3), np.float32)
           for key in ("shadowmat", "vegshadowmat", "vbshmat")},
    )
    return dsm_dir / "Building_DSM_0_0.tif"


class TestSvfCacheExtentValidation:
    def test_matching_cache_is_usable(self, tmp_path: Path) -> None:
        dsm = _build_cache(tmp_path)
        assert _svf_cache_exists(str(dsm), "0_0")
        bundle = load_cached_svf_outputs(str(dsm), "0_0")
        assert len(bundle) == 19

    def test_shifted_origin_rejected(self, tmp_path: Path) -> None:
        shifted = (GT[0] + 662.0,) + GT[1:]  # the real incident's offset
        dsm = _build_cache(tmp_path, svf_gt=shifted)
        assert not _svf_cache_exists(str(dsm), "0_0")
        with pytest.raises(ValueError, match="stale/foreign SVF cache"):
            load_cached_svf_outputs(str(dsm), "0_0")

    def test_shifted_north_rejected(self, tmp_path: Path) -> None:
        shifted = (GT[0], GT[1], GT[2], GT[3] + 2250.0, GT[4], GT[5])
        dsm = _build_cache(tmp_path, svf_gt=shifted)
        assert not _svf_cache_exists(str(dsm), "0_0")

    def test_wrong_raster_size_rejected(self, tmp_path: Path) -> None:
        dsm = _build_cache(tmp_path, svf_size=(ROWS + 2, COLS))
        assert not _svf_cache_exists(str(dsm), "0_0")

    def test_missing_files_rejected(self, tmp_path: Path) -> None:
        dsm = _build_cache(tmp_path)
        (tmp_path / "SVF" / "shadowmats_0_0.npz").unlink()
        assert not _svf_cache_exists(str(dsm), "0_0")

    def test_tiny_geotransform_drift_tolerated(self, tmp_path: Path) -> None:
        drifted = (GT[0] + 1e-9,) + GT[1:]
        dsm = _build_cache(tmp_path, svf_gt=drifted)
        assert _svf_cache_exists(str(dsm), "0_0")

    def test_garbage_zip_falls_back_to_recompute(self, tmp_path: Path) -> None:
        # Corrupt-but-present cache must report "unusable" (soft fallback to
        # recompute), never raise out of _svf_cache_exists under
        # gdal.UseExceptions.
        dsm = _build_cache(tmp_path)
        (tmp_path / "SVF" / f"svfs_0_0.zip").write_bytes(b"not a zip archive")
        assert not _svf_cache_exists(str(dsm), "0_0")

    def test_truncated_svftotal_falls_back_to_recompute(self, tmp_path: Path) -> None:
        dsm = _build_cache(tmp_path)
        svf_tif = tmp_path / "SVF" / "SkyViewFactor_0_0.tif"
        svf_tif.write_bytes(svf_tif.read_bytes()[:64])  # truncated header
        assert not _svf_cache_exists(str(dsm), "0_0")

    def test_zip_missing_svf_member_falls_back_to_recompute(self, tmp_path: Path) -> None:
        dsm = _build_cache(tmp_path)
        zip_path = tmp_path / "SVF" / "svfs_0_0.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = {name: zf.read(name) for name in zf.namelist()}
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, data in members.items():
                if name != "svf.tif":
                    zf.writestr(name, data)
        assert not _svf_cache_exists(str(dsm), "0_0")
