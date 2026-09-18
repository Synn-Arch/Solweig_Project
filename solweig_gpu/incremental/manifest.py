# SPDX-License-Identifier: GPL-3.0-only
"""Site manifest schema for the incremental SOLWEIG baseline cache.

The manifest is the single source of truth for a built site cache. It records
the source rasters of a prepared site directory, the memory-mapped ``.npy``
arrays produced from them, and the model and schema stamps that every result
derived from the cache must carry. The layout follows
``docs/incremental_design_tool/data_model.md``.

All paths stored in the manifest are relative: source paths are relative to
the prepared site directory and array paths are relative to the cache
directory. This keeps manifest bytes deterministic for identical inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Bump when the cache file layout, dtypes, axis order, or semantics change.
CACHE_SCHEMA_VERSION = 1

#: Model stamp written into the manifest and exposed through result metadata.
CACHE_MODEL_VERSION = "solweig-gpu-2.0.0+incremental.cache.1"

MANIFEST_FILENAME = "manifest.json"

#: Columns persisted in the solar-geometry series, in storage order.
SOLAR_GEOMETRY_COLUMNS = (
    "altitude_deg",
    "azimuth_deg",
    "zenith_rad",
    "jday",
    "dectime",
    "altmax_deg",
    "leafon",
)

#: Cache array name -> file path relative to the cache directory.
CACHE_ARRAY_LAYOUT: Mapping[str, str] = {
    "building_dsm": "static/building_dsm.f32.npy",
    "dem": "static/dem.f32.npy",
    "tree_base": "static/tree_base.f32.npy",
    "walls": "static/walls.f32.npy",
    "wall_aspect": "static/wall_aspect.f32.npy",
    "landcover": "static/landcover.u8.npy",
    "svf": "svf/svf.f32.npy",
    "svfE": "svf/svfE.f32.npy",
    "svfS": "svf/svfS.f32.npy",
    "svfW": "svf/svfW.f32.npy",
    "svfN": "svf/svfN.f32.npy",
    "svfveg": "svf/svfveg.f32.npy",
    "svfEveg": "svf/svfEveg.f32.npy",
    "svfSveg": "svf/svfSveg.f32.npy",
    "svfWveg": "svf/svfWveg.f32.npy",
    "svfNveg": "svf/svfNveg.f32.npy",
    "svfaveg": "svf/svfaveg.f32.npy",
    "svfEaveg": "svf/svfEaveg.f32.npy",
    "svfSaveg": "svf/svfSaveg.f32.npy",
    "svfWaveg": "svf/svfWaveg.f32.npy",
    "svfNaveg": "svf/svfNaveg.f32.npy",
    "svftotal": "svf/svftotal.f32.npy",
    "shadowmat": "svf/shadowmat.f32.npy",
    "vegshadowmat": "svf/vegshadowmat.f32.npy",
    "vbshmat": "svf/vbshmat.f32.npy",
    "met": "forcing/meteorology.f32.npy",
    "solar": "forcing/solar_geometry.f32.npy",
}

#: SVF rasters that share the site shape, loaded from the SVF zip and TIFF.
SVF_SCALAR_ARRAYS = (
    "svf",
    "svfE",
    "svfS",
    "svfW",
    "svfN",
    "svfveg",
    "svfEveg",
    "svfSveg",
    "svfWveg",
    "svfNveg",
    "svfaveg",
    "svfEaveg",
    "svfSaveg",
    "svfWaveg",
    "svfNaveg",
    "svftotal",
)

#: SVF patch cubes shaped ``[rows, cols, patch]`` from the shadowmats NPZ.
SVF_PATCH_ARRAYS = ("shadowmat", "vegshadowmat", "vbshmat")

#: Arrays every valid cache must contain.
REQUIRED_CACHE_ARRAYS = (
    "building_dsm",
    "dem",
    "tree_base",
    "walls",
    "wall_aspect",
    *SVF_SCALAR_ARRAYS,
    *SVF_PATCH_ARRAYS,
    "met",
    "solar",
)

#: Arrays that a site may legitimately lack.
OPTIONAL_CACHE_ARRAYS = ("landcover",)

#: Source raster keys every manifest must describe.
REQUIRED_RASTER_KEYS = ("building_dsm", "dem", "trees", "walls", "aspect", "svftotal")

#: All source raster keys a manifest may describe.
KNOWN_RASTER_KEYS = (*REQUIRED_RASTER_KEYS, "landcover")

#: Non-raster source keys every manifest must describe.
REQUIRED_SOURCE_KEYS = ("svf_zip", "svf_npz", "met")

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Base class for manifest schema and validation failures."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


class MissingManifestKeyError(ManifestError):
    """A required manifest key is absent, null, or empty."""


class ManifestMismatchError(ManifestError):
    """A manifest value does not match the file or schema it describes."""


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(text: str) -> str:
    """Return a prefixed ``sha256:`` digest of a string such as CRS WKT."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_npy_header(path: str | Path) -> tuple[tuple[int, ...], np.dtype, bool]:
    """Read only the ``.npy`` header, returning ``(shape, dtype, fortran_order)``."""
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported .npy format version {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype), bool(fortran_order)


@dataclass(frozen=True, slots=True)
class RasterSourceEntry:
    """A source GeoTIFF of the prepared site directory."""

    path: str
    sha256: str
    shape: tuple[int, int]
    dtype: str
    nodata: float | None
    geotransform: tuple[float, float, float, float, float, float]
    crs_wkt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "shape": [int(value) for value in self.shape],
            "dtype": self.dtype,
            "nodata": None if self.nodata is None else float(self.nodata),
            "geotransform": [float(value) for value in self.geotransform],
            "crs_wkt_hash": self.crs_wkt_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, field: str) -> "RasterSourceEntry":
        path = _required_str(data, "path", field=field)
        shape = _required_shape(data, "shape", field=field, length=2)
        geotransform = _required_geotransform(data, field=field)
        nodata: float | None = None
        if data.get("nodata") is not None:
            nodata = _to_float(data["nodata"], field=f"{field}.nodata")
        return cls(
            path=path,
            sha256=_required_sha256(data, field=field),
            shape=shape,
            dtype=_required_str(data, "dtype", field=field),
            nodata=nodata,
            geotransform=geotransform,
            crs_wkt_hash=_required_hash_value(data, "crs_wkt_hash", field=field),
        )


@dataclass(frozen=True, slots=True)
class AuxSourceEntry:
    """A non-raster source file (SVF zip, shadowmats NPZ, or met table)."""

    path: str
    sha256: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": self.path, "sha256": self.sha256}
        if self.shape is not None:
            payload["shape"] = [int(value) for value in self.shape]
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, field: str) -> "AuxSourceEntry":
        shape: tuple[int, ...] | None = None
        if data.get("shape") is not None:
            shape = _required_shape(data, "shape", field=field, length=None)
        dtype: str | None = None
        if data.get("dtype") is not None:
            dtype = _required_str(data, "dtype", field=field)
        return cls(
            path=_required_str(data, "path", field=field),
            sha256=_required_sha256(data, field=field),
            shape=shape,
            dtype=dtype,
        )


@dataclass(frozen=True, slots=True)
class ArrayEntry:
    """A cache ``.npy`` array produced by the cache builder."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": [int(value) for value in self.shape],
            "dtype": self.dtype,
            "nbytes": int(self.nbytes),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, field: str) -> "ArrayEntry":
        shape = _required_shape(data, "shape", field=field, length=None)
        if "nbytes" not in data:
            raise MissingManifestKeyError(
                f"manifest is missing required key '{field}.nbytes'",
                field=f"{field}.nbytes",
            )
        nbytes = data["nbytes"]
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ManifestError(
                f"{field}.nbytes must be a non-negative integer", field=f"{field}.nbytes"
            )
        return cls(
            path=_required_str(data, "path", field=field),
            shape=shape,
            dtype=_required_str(data, "dtype", field=field),
            nbytes=nbytes,
            sha256=_required_sha256(data, field=field),
        )


@dataclass(frozen=True, slots=True)
class SolarGeometrySpec:
    """Location parameters used to persist the solar-geometry series once."""

    latitude: float
    longitude: float
    altitude_m: float
    utc_offset_hours: float
    columns: tuple[str, ...] = SOLAR_GEOMETRY_COLUMNS

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "altitude_m": float(self.altitude_m),
            "utc_offset_hours": float(self.utc_offset_hours),
            "columns": list(self.columns),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, field: str) -> "SolarGeometrySpec":
        columns = data.get("columns")
        if not isinstance(columns, list) or not all(
            isinstance(name, str) and name for name in columns
        ):
            raise ManifestError(
                f"{field}.columns must be a list of column names", field=f"{field}.columns"
            )
        return cls(
            latitude=_to_float(data["latitude"], field=f"{field}.latitude"),
            longitude=_to_float(data["longitude"], field=f"{field}.longitude"),
            altitude_m=_to_float(data["altitude_m"], field=f"{field}.altitude_m"),
            utc_offset_hours=_to_float(
                data["utc_offset_hours"], field=f"{field}.utc_offset_hours"
            ),
            columns=tuple(columns),
        )


@dataclass(frozen=True, slots=True)
class SiteManifest:
    """Validated contents of one site cache ``manifest.json``."""

    site_id: str
    tile_key: str
    model_version: str
    cache_schema_version: int
    rows: int
    cols: int
    pixel_size_m: float
    origin_x_m: float
    origin_y_m: float
    crs_wkt_hash: str
    time_steps: int
    patch_count: int
    rasters: Mapping[str, RasterSourceEntry]
    sources: Mapping[str, AuxSourceEntry]
    solar_geometry: SolarGeometrySpec
    arrays: Mapping[str, ArrayEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_schema_version": int(self.cache_schema_version),
            "model_version": self.model_version,
            "site_id": self.site_id,
            "tile_key": self.tile_key,
            "crs_wkt_hash": self.crs_wkt_hash,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "pixel_size_m": float(self.pixel_size_m),
            "origin_x_m": float(self.origin_x_m),
            "origin_y_m": float(self.origin_y_m),
            "time_steps": int(self.time_steps),
            "patch_count": int(self.patch_count),
            "solar_geometry": self.solar_geometry.to_dict(),
            "rasters": {
                name: self.rasters[name].to_dict() for name in sorted(self.rasters)
            },
            "sources": {
                name: self.sources[name].to_dict() for name in sorted(self.sources)
            },
            "arrays": {
                name: self.arrays[name].to_dict() for name in sorted(self.arrays)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SiteManifest":
        """Parse and schema-check manifest JSON, naming the offending field."""
        for key in (
            "cache_schema_version",
            "model_version",
            "site_id",
            "tile_key",
            "crs_wkt_hash",
            "rows",
            "cols",
            "pixel_size_m",
            "origin_x_m",
            "origin_y_m",
            "time_steps",
            "patch_count",
            "rasters",
            "sources",
            "solar_geometry",
            "arrays",
        ):
            if key not in data or data[key] is None:
                raise MissingManifestKeyError(
                    f"manifest is missing required key '{key}'", field=key
                )

        version = data["cache_schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ManifestError(
                "cache_schema_version must be an integer", field="cache_schema_version"
            )
        if version != CACHE_SCHEMA_VERSION:
            raise ManifestMismatchError(
                f"cache_schema_version {version} does not match supported version "
                f"{CACHE_SCHEMA_VERSION}",
                field="cache_schema_version",
            )

        rows = _to_positive_int(data["rows"], field="rows")
        cols = _to_positive_int(data["cols"], field="cols")
        pixel_size = _to_positive_float(data["pixel_size_m"], field="pixel_size_m")
        rasters_section = _to_mapping(data["rasters"], field="rasters")
        for name in REQUIRED_RASTER_KEYS:
            if name not in rasters_section:
                raise MissingManifestKeyError(
                    f"manifest is missing required key 'rasters.{name}'",
                    field=f"rasters.{name}",
                )
        unknown_rasters = sorted(set(rasters_section) - set(KNOWN_RASTER_KEYS))
        if unknown_rasters:
            raise ManifestError(
                f"rasters contains unknown entries: {unknown_rasters}", field="rasters"
            )

        sources_section = _to_mapping(data["sources"], field="sources")
        for name in REQUIRED_SOURCE_KEYS:
            if name not in sources_section:
                raise MissingManifestKeyError(
                    f"manifest is missing required key 'sources.{name}'",
                    field=f"sources.{name}",
                )

        solar = SolarGeometrySpec.from_dict(
            _to_mapping(data["solar_geometry"], field="solar_geometry"),
            field="solar_geometry",
        )

        arrays_section = _to_mapping(data["arrays"], field="arrays")
        for name in REQUIRED_CACHE_ARRAYS:
            if name not in arrays_section:
                raise MissingManifestKeyError(
                    f"manifest is missing required key 'arrays.{name}'",
                    field=f"arrays.{name}",
                )
        for name in sorted(arrays_section):
            if name not in CACHE_ARRAY_LAYOUT:
                raise ManifestError(
                    f"arrays contains unknown entry '{name}'", field=f"arrays.{name}"
                )
            if arrays_section[name].get("path") != CACHE_ARRAY_LAYOUT[name]:
                raise ManifestMismatchError(
                    f"arrays.{name}.path must be '{CACHE_ARRAY_LAYOUT[name]}' "
                    f"for this schema version",
                    field=f"arrays.{name}.path",
                )

        return cls(
            site_id=_required_str(data, "site_id", field=""),
            tile_key=_required_str(data, "tile_key", field=""),
            model_version=_required_str(data, "model_version", field=""),
            cache_schema_version=version,
            rows=rows,
            cols=cols,
            pixel_size_m=pixel_size,
            origin_x_m=_to_float(data["origin_x_m"], field="origin_x_m"),
            origin_y_m=_to_float(data["origin_y_m"], field="origin_y_m"),
            crs_wkt_hash=_required_hash_value(data, "crs_wkt_hash", field=""),
            time_steps=_to_positive_int(data["time_steps"], field="time_steps"),
            patch_count=_to_positive_int(data["patch_count"], field="patch_count"),
            rasters={
                name: RasterSourceEntry.from_dict(
                    rasters_section[name], field=f"rasters.{name}"
                )
                for name in sorted(rasters_section)
            },
            sources={
                name: AuxSourceEntry.from_dict(
                    sources_section[name], field=f"sources.{name}"
                )
                for name in sorted(sources_section)
            },
            solar_geometry=solar,
            arrays={
                name: ArrayEntry.from_dict(arrays_section[name], field=f"arrays.{name}")
                for name in sorted(arrays_section)
            },
        )

    def write(self, path: str | Path, *, indent: int = 2) -> None:
        """Write the manifest as stable JSON with sorted keys and a newline."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), sort_keys=True, indent=indent) + "\n"
        path.write_text(text, encoding="utf-8")

    def validate(self, cache_dir: str | Path) -> None:
        """Fully validate cache arrays against this manifest.

        Checks file existence, ``.npy`` header shape and dtype, and the SHA-256
        digest of every array file. Raises :class:`ManifestMismatchError`
        naming each offending ``arrays.<name>.<field>``.
        """
        problems: list[str] = []
        root = Path(cache_dir)
        for name in sorted(self.arrays):
            entry = self.arrays[name]
            field = f"arrays.{name}"
            path = root / entry.path
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
            digest = sha256_file(path)
            if digest != entry.sha256:
                problems.append(
                    f"{field}.sha256: manifest {entry.sha256} does not match "
                    f"file {digest}"
                )
        if problems:
            raise ManifestMismatchError("; ".join(problems), field=problems[0].split(":")[0])

    def verify_sources(self, site_dir: str | Path) -> None:
        """Re-hash the prepared site sources against this manifest."""
        problems: list[str] = []
        root = Path(site_dir)
        sections = (("rasters", self.rasters), ("sources", self.sources))
        for section_name, section in sections:
            for name in sorted(section):
                entry = section[name]
                field = f"{section_name}.{name}"
                path = root / entry.path
                if not path.is_file():
                    problems.append(f"{field}.path: file '{entry.path}' is missing")
                    continue
                digest = sha256_file(path)
                if digest != entry.sha256:
                    problems.append(
                        f"{field}.sha256: manifest {entry.sha256} does not match "
                        f"file {digest}"
                    )
        if problems:
            raise ManifestMismatchError("; ".join(problems), field=problems[0].split(":")[0])


def load_manifest(path: str | Path) -> SiteManifest:
    """Load and schema-validate a site manifest from ``path``."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(
            f"manifest file not found: {manifest_path}", field="manifest"
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(
            f"manifest is not valid JSON: {error}", field="manifest"
        ) from error
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object", field="manifest")
    return SiteManifest.from_dict(data)


def _required_str(data: Mapping[str, Any], key: str, *, field: str) -> str:
    prefix = f"{field}." if field else ""
    if key not in data:
        raise MissingManifestKeyError(
            f"manifest is missing required key '{prefix}{key}'", field=f"{prefix}{key}"
        )
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ManifestError(
            f"{prefix}{key} must be a non-empty string", field=f"{prefix}{key}"
        )
    return value


def _required_sha256(data: Mapping[str, Any], *, field: str) -> str:
    value = _required_str(data, "sha256", field=field)
    if not _HEX64_RE.fullmatch(value):
        raise ManifestError(
            f"{field}.sha256 must be a 64-character hex digest",
            field=f"{field}.sha256",
        )
    return value


def _required_hash_value(data: Mapping[str, Any], key: str, *, field: str) -> str:
    prefix = f"{field}." if field else ""
    value = _required_str(data, key, field=field)
    if not value.startswith("sha256:") or not _HEX64_RE.fullmatch(value[len("sha256:"):]):
        raise ManifestError(
            f"{prefix}{key} must be a 'sha256:<hex>' digest", field=f"{prefix}{key}"
        )
    return value


def _required_shape(
    data: Mapping[str, Any], key: str, *, field: str, length: int | None
) -> tuple[int, ...]:
    prefix = f"{field}." if field else ""
    if key not in data:
        raise MissingManifestKeyError(
            f"manifest is missing required key '{prefix}{key}'", field=f"{prefix}{key}"
        )
    value = data[key]
    if not isinstance(value, list) or not all(
        isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
        for dimension in value
    ):
        raise ManifestError(
            f"{prefix}{key} must be a list of positive integers", field=f"{prefix}{key}"
        )
    if length is not None and len(value) != length:
        raise ManifestError(
            f"{prefix}{key} must contain {length} dimensions", field=f"{prefix}{key}"
        )
    return tuple(int(dimension) for dimension in value)


def _required_geotransform(
    data: Mapping[str, Any], *, field: str
) -> tuple[float, float, float, float, float, float]:
    if "geotransform" not in data:
        raise MissingManifestKeyError(
            f"manifest is missing required key '{field}.geotransform'",
            field=f"{field}.geotransform",
        )
    value = data["geotransform"]
    if (
        not isinstance(value, list)
        or len(value) != 6
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        raise ManifestError(
            f"{field}.geotransform must be six numbers", field=f"{field}.geotransform"
        )
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _to_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field} must be a number", field=field)
    number = float(value)
    if not isfinite(number):
        raise ManifestError(f"{field} must be finite", field=field)
    return number


def _to_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{field} must be a positive integer", field=field)
    return value


def _to_positive_float(value: Any, *, field: str) -> float:
    number = _to_float(value, field=field)
    if number <= 0:
        raise ManifestError(f"{field} must be positive", field=field)
    return number


def _to_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be a JSON object", field=field)
    return value
