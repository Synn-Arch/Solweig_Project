# SPDX-License-Identifier: GPL-3.0-only
"""Versioned result patches for the exact incremental worker (Stage 10).

A :class:`ResultPatch` is the atomic unit of publication produced by the
Phase-5 worker. It carries:

- ``schema_version`` so future readers can reject/upgrade old patches;
- ``job_id`` / ``scene_revision`` provenance (Stage 10: the patch is only
  valid against the scene revision it was computed from);
- the half-open ``write_window`` (never the read window: halo cells are
  inputs, not results) plus the ``read_window`` for provenance;
- one float32 array per published variable with shape
  ``(time_steps, window_height, window_width)``;
- a SHA-256 checksum per variable and for the JSON metadata document.

Publication is two-phase (Stage 10 atomicity): :func:`stage_patch` writes
everything into a temporary directory, ``publish_staged_patch`` performs the
final ``os.rename`` (atomic within a filesystem) only after the caller has
re-checked scene supersession. A partially written or tampered patch never
loads silently: :func:`load_patch` verifies every checksum before returning.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from solweig_gpu.incremental.geometry import RasterWindow

PATCH_SCHEMA_VERSION = 1
PATCH_METADATA_FILE = "patch.json"
PATCH_VARIABLES_DIR = "variables"

#: Variables the exact worker is allowed to publish (single source of truth
#: shared with the solver; lower-case names match run_utci_window outputs).
SUPPORTED_VARIABLES = ("utci", "tmrt", "shadow")


class PatchError(RuntimeError):
    """Raised when a patch cannot be staged, verified, or published."""


@dataclass(frozen=True, slots=True)
class PatchMetadata:
    """Metadata-only view of a published patch (u-c1b L8).

    Everything :meth:`load_patch` verifies before touching array bytes:
    provenance, windows, the variable list, and the RECORDED checksums
    (presented unverified — no array file is hashed or opened). Consumers
    that index patches without reading their arrays (the executor's
    ``(node, time)`` store recording) use
    :func:`load_patch_metadata`; consumers that read result content must
    still go through :func:`load_patch`, where the full SHA-256
    verification stays.
    """

    job_id: str
    scene_revision: int
    mode: str
    write_window: RasterWindow
    read_window: RasterWindow
    site_id: str
    tile_key: str
    cache_manifest_sha256: str
    model_version: str
    variables: tuple[str, ...]
    time_start: int
    time_stop: int | None
    schema_version: int
    created_utc: str
    checksums: dict[str, str]
    time_indices: tuple[int, ...] | None = None

    @property
    def n_time_steps(self) -> int:
        """Timesteps the patch covers.

        Requires a recorded ``time_stop``: when the patch left it unset the
        length lives only in the array file, which this metadata view
        deliberately never opens — read the patch through :func:`load_patch`
        instead of guessing.
        """
        if self.time_indices is not None:
            return len(self.time_indices)
        if self.time_stop is None:
            raise PatchError(
                "patch metadata carries no time_stop; the step count lives "
                "in the arrays — use load_patch to read them"
            )
        return int(self.time_stop - self.time_start)


class PatchChecksumError(PatchError):
    """A stored patch failed checksum verification (tampering or truncation)."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_to_dict(window: RasterWindow) -> dict:
    return {
        "row_start": int(window.row_start),
        "row_stop": int(window.row_stop),
        "col_start": int(window.col_start),
        "col_stop": int(window.col_stop),
    }


def _window_from_dict(data: dict) -> RasterWindow:
    return RasterWindow(
        row_start=int(data["row_start"]),
        row_stop=int(data["row_stop"]),
        col_start=int(data["col_start"]),
        col_stop=int(data["col_stop"]),
    )


@dataclass
class ResultPatch:
    """A versioned, checksummed, window-scoped set of result arrays."""

    job_id: str
    scene_revision: int
    mode: str  # "local" or "full"
    write_window: RasterWindow
    read_window: RasterWindow
    site_id: str
    tile_key: str
    cache_manifest_sha256: str
    model_version: str
    variables: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    time_start: int = 0
    time_stop: int | None = None
    time_indices: tuple[int, ...] | None = None
    schema_version: int = PATCH_SCHEMA_VERSION
    created_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    checksums: dict[str, str] = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    def __post_init__(self) -> None:
        self.variables = tuple(self.variables)
        if self.mode not in ("local", "full"):
            raise PatchError(f"unsupported patch mode {self.mode!r}")
        unknown = [name for name in self.variables if name not in SUPPORTED_VARIABLES]
        if unknown:
            raise PatchError(f"unsupported patch variables: {unknown}")
        missing = [name for name in self.variables if name not in self.arrays]
        if missing:
            raise PatchError(f"missing arrays for variables: {missing}")
        if self.time_indices is not None:
            indices = tuple(self.time_indices)
            if not indices:
                raise PatchError(
                    "time_indices must be non-empty when present (a sparse "
                    "patch carries at least one timestep)"
                )
            if any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0
                for i in indices
            ):
                raise PatchError(
                    "time_indices must be non-negative integers, got "
                    f"{indices}"
                )
            if any(b <= a for a, b in zip(indices, indices[1:])):
                raise PatchError(
                    f"time_indices must be strictly increasing, got {indices}"
                )
            if self.time_start is not None and indices[0] != self.time_start:
                raise PatchError(
                    "time_indices[0] must equal time_start "
                    f"({indices[0]} != {self.time_start})"
                )
            if (
                self.time_stop is not None
                and indices[-1] != self.time_stop - 1
            ):
                raise PatchError(
                    "time_indices[-1] must equal time_stop - 1 "
                    f"({indices[-1]} != {self.time_stop - 1})"
                )
            self.time_indices = indices
        height = self.write_window.height
        width = self.write_window.width
        for name, array in self.arrays.items():
            if array.dtype != np.float32:
                raise PatchError(f"variable {name!r} must be float32, got {array.dtype}")
            expected = (self.n_time_steps, height, width)
            if array.shape != expected:
                raise PatchError(
                    f"variable {name!r} shape {array.shape} does not match "
                    f"(time, window) {expected}"
                )
            if not array.flags["C_CONTIGUOUS"]:
                # keep on-disk layout deterministic; copies are cheap relative to solve
                self.arrays[name] = np.ascontiguousarray(array)

    @property
    def n_time_steps(self) -> int:
        if self.time_indices is not None:
            return len(self.time_indices)
        stop = self.time_stop if self.time_stop is not None else self.arrays[self.variables[0]].shape[0]
        return int(stop - self.time_start)

    def variable(self, name: str) -> np.ndarray:
        if name not in self.arrays:
            raise PatchError(f"patch does not contain variable {name!r}")
        return self.arrays[name]

    def apply_into(self, target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Write every variable into ``target`` inside the write window.

        Never touches cells outside the window (SCI-002). ``target`` arrays
        are full-site ``(time, rows, cols)`` float32 arrays mutated in place.

        Sparse patches (``time_indices`` set) scatter row ``k`` into timestep
        ``time_indices[k]`` — the ABSENT timesteps are never written, so a
        consumer must never read them as values produced by this patch.
        """
        w = self.write_window
        indices = self.time_indices
        for name, array in self.arrays.items():
            dst = target[name]
            if dst.ndim != 3:
                raise PatchError(f"target[{name!r}] must be 3-D (time, rows, cols)")
            if indices is not None and max(indices) >= dst.shape[0]:
                raise PatchError(
                    f"target[{name!r}] has {dst.shape[0]} timesteps; sparse "
                    f"patch writes up to t={max(indices)}"
                )
            for k in range(array.shape[0]):
                time_index = indices[k] if indices is not None else self.time_start + k
                region = dst[time_index,
                             w.row_start : w.row_stop, w.col_start : w.col_stop]
                if region.shape != array[k].shape:
                    raise PatchError(
                        f"target[{name!r}] window region {region.shape} does "
                        f"not match patch row {array[k].shape}"
                    )
                region[...] = array[k]
        return target

    # -- serialisation ----------------------------------------------------
    def metadata(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "scene_revision": int(self.scene_revision),
            "mode": self.mode,
            "write_window": _window_to_dict(self.write_window),
            "read_window": _window_to_dict(self.read_window),
            "site_id": self.site_id,
            "tile_key": self.tile_key,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "model_version": self.model_version,
            "variables": list(self.variables),
            "time_start": int(self.time_start),
            "time_stop": None if self.time_stop is None else int(self.time_stop),
            "time_indices": (
                None if self.time_indices is None else list(self.time_indices)
            ),
            "created_utc": self.created_utc,
            "checksums": dict(self.checksums),
        }

    def compute_checksums(self, directory: Path) -> dict[str, str]:
        checksums = {}
        for name in self.variables:
            checksums[name] = _sha256_file(
                directory / PATCH_VARIABLES_DIR / f"{name}.f32.npy"
            )
        return checksums


def new_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


def stage_patch(patch: ResultPatch, staging_root: Path) -> Path:
    """Write patch contents into ``staging_root/<job_id>.tmp`` (Phase 1).

    The staging directory name embeds the job id so concurrent jobs never
    collide; it ends with ``.tmp`` so it can never be mistaken for a
    published patch.
    """
    staging_dir = staging_root / f"{patch.job_id}.tmp"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    variables_dir = staging_dir / PATCH_VARIABLES_DIR
    variables_dir.mkdir(parents=True)
    for name in patch.variables:
        np.save(variables_dir / f"{name}.f32.npy", patch.arrays[name])
    checksums = patch.compute_checksums(staging_dir)
    patch.checksums = checksums
    (staging_dir / PATCH_METADATA_FILE).write_text(
        json.dumps({**patch.metadata(), "checksums": checksums}, indent=2, sort_keys=True)
        + "\n"
    )
    return staging_dir


def publish_staged_patch(staging_dir: Path, results_root: Path, scene_revision: int) -> Path:
    """Atomically move a staged patch into the results store (Phase 2).

    The final directory name encodes the scene revision. ``os.rename`` on the
    same filesystem is atomic: readers either see no patch at all or the
    complete patch directory.
    """
    final_dir = results_root / f"rev-{int(scene_revision):06d}-{staging_dir.name.removesuffix('.tmp')}"
    if final_dir.exists():
        raise PatchError(f"patch directory already exists: {final_dir}")
    os.rename(staging_dir, final_dir)
    return final_dir


def _read_patch_metadata(patch_dir: Path) -> dict:
    """Parse and schema-check ``patch.json`` (shared by load/load-metadata)."""
    metadata_path = patch_dir / PATCH_METADATA_FILE
    if not metadata_path.is_file():
        raise PatchError(f"no {PATCH_METADATA_FILE} under {patch_dir}")
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("schema_version", -1)) != PATCH_SCHEMA_VERSION:
        raise PatchError(
            f"patch schema version {metadata.get('schema_version')!r} is not "
            f"supported (expected {PATCH_SCHEMA_VERSION})"
        )
    return metadata


def load_patch(patch_dir: Path) -> ResultPatch:
    """Load and verify a published patch (checksums + schema version)."""
    metadata = _read_patch_metadata(patch_dir)
    variables = tuple(metadata["variables"])
    arrays: dict[str, np.ndarray] = {}
    checksums = metadata.get("checksums", {})
    for name in variables:
        array_path = patch_dir / PATCH_VARIABLES_DIR / f"{name}.f32.npy"
        if not array_path.is_file():
            raise PatchError(f"patch variable file missing: {array_path}")
        if name not in checksums:
            # Metadata corruption that DROPS a checksum entry must not
            # silently bypass verification.
            raise PatchError(
                f"patch metadata has no checksum for variable {name!r}"
            )
        actual = _sha256_file(array_path)
        if actual != checksums[name]:
            raise PatchChecksumError(
                f"checksum mismatch for variable {name!r}: stored "
                f"{checksums[name][:16]}..., actual {actual[:16]}..."
            )
        arrays[name] = np.load(array_path, mmap_mode=None)
    time_stop = metadata.get("time_stop")
    time_indices = metadata.get("time_indices")
    patch = ResultPatch(
        job_id=metadata["job_id"],
        scene_revision=int(metadata["scene_revision"]),
        mode=metadata["mode"],
        write_window=_window_from_dict(metadata["write_window"]),
        read_window=_window_from_dict(metadata["read_window"]),
        site_id=metadata["site_id"],
        tile_key=metadata["tile_key"],
        cache_manifest_sha256=metadata["cache_manifest_sha256"],
        model_version=metadata["model_version"],
        variables=variables,
        arrays=arrays,
        time_start=int(metadata.get("time_start", 0)),
        time_stop=None if time_stop is None else int(time_stop),
        time_indices=(
            None
            if time_indices is None
            else tuple(int(i) for i in time_indices)
        ),
        schema_version=int(metadata["schema_version"]),
        created_utc=metadata.get("created_utc", ""),
        checksums=checksums,
    )
    # shape consistency between metadata window and stored arrays
    expected = (patch.n_time_steps, patch.write_window.height, patch.write_window.width)
    for name, array in arrays.items():
        if array.shape != expected:
            raise PatchError(
                f"stored array {name!r} shape {array.shape} inconsistent with "
                f"metadata window {expected}"
            )
    return patch


def load_patch_metadata(patch_dir: Path) -> PatchMetadata:
    """Load a published patch's metadata WITHOUT verifying or reading arrays.

    Perf note (u-c1b L8): the executor indexes every published patch into
    the ``(node, time)`` store right after publication — it needs the
    variable list, the write window, the mode, and the time bounds, not
    the arrays. Re-reading and SHA-256-hashing every ``.npy`` just to
    index metadata costs one full array pass per patch on the hot publish
    path, so this variant parses ``patch.json`` only (schema version and
    structure still validated; checksums returned as RECORDED, never
    verified here). Full verification stays in :func:`load_patch`, at the
    point where result content is actually read.
    """
    metadata = _read_patch_metadata(patch_dir)
    return PatchMetadata(
        job_id=metadata["job_id"],
        scene_revision=int(metadata["scene_revision"]),
        mode=metadata["mode"],
        write_window=_window_from_dict(metadata["write_window"]),
        read_window=_window_from_dict(metadata["read_window"]),
        site_id=metadata["site_id"],
        tile_key=metadata["tile_key"],
        cache_manifest_sha256=metadata["cache_manifest_sha256"],
        model_version=metadata["model_version"],
        variables=tuple(metadata["variables"]),
        time_start=int(metadata.get("time_start", 0)),
        time_stop=(
            None
            if metadata.get("time_stop") is None
            else int(metadata["time_stop"])
        ),
        schema_version=int(metadata["schema_version"]),
        created_utc=metadata.get("created_utc", ""),
        checksums=dict(metadata.get("checksums", {})),
        time_indices=(
            None
            if metadata.get("time_indices") is None
            else tuple(int(i) for i in metadata["time_indices"])
        ),
    )


def discard_staging(staging_root: Path) -> None:
    """Remove every ``*.tmp`` staging directory (aborted jobs publish nothing)."""
    if not staging_root.exists():
        return
    for entry in staging_root.iterdir():
        if entry.name.endswith(".tmp") and entry.is_dir():
            shutil.rmtree(entry)
