# SPDX-License-Identifier: GPL-3.0-only
"""Read-only, memory-mapped baseline site cache for the incremental worker.

A :class:`SiteCache` maps every fixed SOLWEIG input array from uncompressed
``.npy`` files using ``numpy.load(..., mmap_mode="r")``. The warm worker path
therefore performs zero GDAL or rasterio raster reads; rasters are converted
exactly once by the cache builder (``cache_builder.py``).

Semantics of the mapped SVF patch cubes follow the existing standalone SVF
cache produced by ``solweig_gpu``:

- ``shadowmat``: building-only sky-patch visibility;
- ``vegshadowmat``: vegetation-only sky-patch visibility;
- ``vbshmat``: combined building-and-vegetation sky-patch visibility.

All spatial windows are half-open: rows ``[row_start, row_stop)`` and columns
``[col_start, col_stop)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .geometry import RasterWindow
from .manifest import (
    MANIFEST_FILENAME,
    SVF_PATCH_ARRAYS,
    SVF_SCALAR_ARRAYS,
    SiteManifest,
    load_manifest,
    read_npy_header,
    sha256_file,
)

#: Arrays at or below this size always have their full checksum verified by
#: ``SiteCache.self_test``. Larger arrays are verified on a fixed stride so the
#: startup cost stays bounded while tampering is still detected.
SELF_TEST_MAX_FULL_BYTES = 64 * 1024 * 1024

#: Fraction of large arrays whose checksums are verified by ``self_test``.
SELF_TEST_LARGE_STRIDE = 4


class CacheValidationError(ValueError):
    """Raised when cache array files disagree with their manifest entries."""

    def __init__(self, message: str, *, array: str | None = None) -> None:
        self.array = array
        super().__init__(message)


class SiteCache:
    """Read-only memory-mapped view of a built site cache directory.

    Use :meth:`load` to open a cache. Returned arrays are ``mode='r'`` memory
    maps: NumPy rejects writes to them structurally, so the shared baseline
    cannot be mutated through this API.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        manifest: SiteManifest,
        arrays: Mapping[str, np.memmap],
    ) -> None:
        """Assemble a cache from an already-mapped array set.

        Prefer :meth:`load`, which reads the manifest, maps the arrays, and
        runs the startup self-test.
        """
        self._cache_dir = Path(cache_dir)
        self.manifest = manifest
        self._arrays: dict[str, np.memmap] = dict(arrays)

    @classmethod
    def load(
        cls,
        cache_dir: str | Path,
        *,
        run_self_test: bool = True,
    ) -> "SiteCache":
        """Map a built cache directory without any raster reads.

        Raises :class:`~solweig_gpu.incremental.manifest.ManifestError` when
        the manifest is invalid and :class:`CacheValidationError` when array
        files are missing, unreadable, or fail the self-test.
        """
        root = Path(cache_dir)
        manifest = load_manifest(root / MANIFEST_FILENAME)
        arrays: dict[str, np.memmap] = {}
        for name in sorted(manifest.arrays):
            entry = manifest.arrays[name]
            path = root / entry.path
            if not path.is_file():
                raise CacheValidationError(
                    f"cache array '{name}' file '{entry.path}' is missing", array=name
                )
            arrays[name] = np.load(path, mmap_mode="r")
        cache = cls(root, manifest, arrays)
        if run_self_test:
            cache.self_test()
        return cache

    # ------------------------------------------------------------------
    # Array access
    # ------------------------------------------------------------------

    def array(self, name: str) -> np.memmap:
        """Return the read-only memory map for a cache array name."""
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(
                f"unknown cache array '{name}'; available arrays: "
                f"{sorted(self._arrays)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._arrays

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._arrays))

    def array_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._arrays))

    def window(self, name: str, window: RasterWindow) -> np.ndarray:
        """Return a half-open spatial slice of a cached array.

        Rows ``[row_start, row_stop)`` and columns ``[col_start, col_stop)``.
        Trailing dimensions such as the sky-patch axis are preserved. The
        returned view inherits the read-only mapping of its parent array.
        """
        array = self.array(name)
        if array.ndim < 2:
            raise ValueError(f"array '{name}' has no spatial axes to window")
        if (
            window.row_start < 0
            or window.row_stop > array.shape[0]
            or window.col_start < 0
            or window.col_stop > array.shape[1]
        ):
            raise ValueError(
                f"window ({window.row_start}, {window.row_stop}, "
                f"{window.col_start}, {window.col_stop}) exceeds array "
                f"'{name}' shape {array.shape}"
            )
        return array[
            window.row_start : window.row_stop, window.col_start : window.col_stop
        ]

    @property
    def dem(self) -> np.memmap:
        return self.array("dem")

    @property
    def building_dsm(self) -> np.memmap:
        return self.array("building_dsm")

    @property
    def tree_base(self) -> np.memmap:
        return self.array("tree_base")

    @property
    def walls(self) -> np.memmap:
        return self.array("walls")

    @property
    def wall_aspect(self) -> np.memmap:
        return self.array("wall_aspect")

    @property
    def landcover(self) -> np.memmap:
        """Optional base land cover; absent on sites without a landcover raster."""
        return self.array("landcover")

    @property
    def met(self) -> np.memmap:
        """Meteorological forcing table shaped ``[time_steps, columns]``."""
        return self.array("met")

    @property
    def solar(self) -> np.memmap:
        """Solar-geometry series shaped ``[time_steps, len(columns)]``."""
        return self.array("solar")

    @property
    def svf(self) -> Mapping[str, np.memmap]:
        """Scalar sky-view-factor rasters keyed like the SVF zip members."""
        return {name: self._arrays[name] for name in SVF_SCALAR_ARRAYS if name in self._arrays}

    @property
    def svf_patches(self) -> Mapping[str, np.memmap]:
        """Patch cubes shaped ``[rows, cols, patch]`` from the shadowmats NPZ."""
        return {name: self._arrays[name] for name in SVF_PATCH_ARRAYS if name in self._arrays}

    # ------------------------------------------------------------------
    # Site geometry
    # ------------------------------------------------------------------

    @property
    def rows(self) -> int:
        return self.manifest.rows

    @property
    def cols(self) -> int:
        return self.manifest.cols

    @property
    def pixel_size_m(self) -> float:
        return self.manifest.pixel_size_m

    @property
    def time_steps(self) -> int:
        return self.manifest.time_steps

    @property
    def patch_count(self) -> int:
        return self.manifest.patch_count

    @property
    def site_id(self) -> str:
        return self.manifest.site_id

    @property
    def tile_key(self) -> str:
        return self.manifest.tile_key

    @property
    def model_version(self) -> str:
        return self.manifest.model_version

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def layout(self) -> Mapping[str, str]:
        """Cache array name to path relative to the cache directory."""
        return {
            name: self.manifest.arrays[name].path
            for name in sorted(self.manifest.arrays)
        }

    # ------------------------------------------------------------------
    # Validation and metadata
    # ------------------------------------------------------------------

    def _checksum_sample(self) -> tuple[str, ...]:
        """Deterministic checksum sample: all small arrays plus a fixed stride."""
        names = sorted(self.manifest.arrays)
        entries = self.manifest.arrays
        small = [name for name in names if entries[name].nbytes <= SELF_TEST_MAX_FULL_BYTES]
        large = [name for name in names if entries[name].nbytes > SELF_TEST_MAX_FULL_BYTES]
        if not large:
            return tuple(names)
        stride = max(1, len(large) // SELF_TEST_LARGE_STRIDE)
        sampled = set(large[::stride])
        sampled.update((large[0], large[-1]))
        return tuple(sorted(small + list(sampled)))

    def self_test(self) -> None:
        """Verify manifest headers and checksums of a deterministic sample.

        Every array's ``.npy`` header shape and dtype is checked. Checksums
        are checked for all small arrays and a fixed stride of large ones, so
        startup stays bounded on big sites while tampering is still detected.
        Called automatically by :meth:`load` unless disabled.
        """
        problems: list[str] = []
        for name in sorted(self.manifest.arrays):
            entry = self.manifest.arrays[name]
            field = f"arrays.{name}"
            path = self._cache_dir / entry.path
            if not path.is_file():
                problems.append(f"{field}.path: file '{entry.path}' is missing")
                continue
            try:
                shape, dtype, _ = read_npy_header(path)
            except ValueError as error:
                problems.append(f"{field}: unreadable .npy header ({error})")
                continue
            if shape != tuple(entry.shape):
                problems.append(
                    f"{field}.shape: manifest {list(entry.shape)} does not match "
                    f"file {list(shape)}"
                )
            if dtype != np.dtype(entry.dtype):
                problems.append(
                    f"{field}.dtype: manifest {entry.dtype} does not match file {dtype}"
                )
        for name in self._checksum_sample():
            entry = self.manifest.arrays[name]
            digest = sha256_file(self._cache_dir / entry.path)
            if digest != entry.sha256:
                problems.append(
                    f"arrays.{name}.sha256: manifest {entry.sha256} does not match "
                    f"file {digest}"
                )
        if problems:
            raise CacheValidationError(
                "cache self-test failed: " + "; ".join(problems),
                array=problems[0].split(".")[1] if problems[0].startswith("arrays.") else None,
            )

    def metadata(self) -> dict[str, Any]:
        """Result metadata describing the cache a published result came from.

        Includes the cache layout and model version so results can be
        invalidated when the cache or model changes.
        """
        return {
            "site_id": self.manifest.site_id,
            "tile_key": self.manifest.tile_key,
            "cache_schema_version": self.manifest.cache_schema_version,
            "model_version": self.manifest.model_version,
            "manifest_sha256": sha256_file(self._cache_dir / MANIFEST_FILENAME),
            "rows": self.manifest.rows,
            "cols": self.manifest.cols,
            "pixel_size_m": self.manifest.pixel_size_m,
            "time_steps": self.manifest.time_steps,
            "patch_count": self.manifest.patch_count,
            "layout": dict(self.layout),
        }
