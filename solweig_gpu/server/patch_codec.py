# SPDX-License-Identifier: GPL-3.0-only
"""Binary result-patch codec for the SOLWEIG design-tool HTTP API.

The API serves one rectangular float32 patch per result version. The
uncompressed byte order follows ``docs/incremental_design_tool/data_model.md``:

.. code-block:: text

    C-order, little-endian
    variable-major
    then time
    then row
    then column

The payload is zstd-compressed (level 3) and checksummed with SHA-256 over the
compressed bytes actually served, so the manifest ``checksum`` field equals the
``ETag`` a client can verify after download.

Payload *encoding* is negotiated per request via the ``Accept`` header: clients
that cannot decompress zstd in the browser (no JS DecompressionStream for zstd)
request ``application/vnd.solweig.patch+identity`` and receive the same
uncompressed byte order. Each response carries its own SHA-256 — the zstd
checksum (also the durable manifest ``checksum``) or the identity checksum —
computed over exactly the bytes on the wire, so the checksum/ETag contract
holds for both encodings.

Decoding validates the checksum, schema version, per-variable dtype, shape and
total byte length, and rejects mismatches with specific exceptions naming the
offending variable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zstandard

from solweig_gpu.incremental.geometry import RasterWindow

#: Binary patch layout version (``data_model.md`` versioning rules).
PATCH_SCHEMA_VERSION = 1

#: Media type of the compressed payload served by the API.
PATCH_MEDIA_TYPE = "application/vnd.solweig.patch+zstd"

#: Media type of the uncompressed (identity) payload encoding.
IDENTITY_MEDIA_TYPE = "application/vnd.solweig.patch+identity"

#: zstd compression level used for every served payload.
ZSTD_COMPRESSION_LEVEL = 3

#: Variables the codec accepts (single source of truth shared with the worker).
SUPPORTED_VARIABLES = ("utci", "tmrt", "shadow")

#: Float dtype of every patch variable, little-endian on the wire.
PATCH_DTYPE = "<f4"

#: Nodata marker used in manifests.
NODATA = "nan"

_CHECKSUM_PREFIX = "sha256:"


class PatchCodecError(ValueError):
    """Base class for patch encoding/decoding failures."""


class PayloadCompressionError(PatchCodecError):
    """The payload is not valid zstd data or the compression field is wrong."""


class PayloadChecksumError(PatchCodecError):
    """The payload does not match the manifest checksum (tampering/truncation)."""


class PayloadSchemaError(PatchCodecError):
    """The manifest schema version is not supported."""


class PayloadDtypeError(PatchCodecError):
    """A manifest variable declares an unsupported dtype."""


class PayloadShapeError(PatchCodecError):
    """Manifest shapes are inconsistent with the payload byte length."""


class PayloadLengthError(PatchCodecError):
    """The decompressed payload length does not match the manifest."""


class PartialTimeScatterError(PatchCodecError):
    """A partial-time payload cannot be scattered without its global times.

    A payload covering a SELECTED subset of the site's time axis must name
    its global ``time_indices`` in the manifest, and those indices must
    match the payload's time depth. Guessing dense offsets (``t = 0..n``)
    would land a changed-``t`` plane at the wrong global time while the
    actually-changed time kept its stale value — the DESIGN §18.4
    ``sparse t index -> dense offset scatter`` mutation class. Raised by
    the composition scatter (:func:`solweig_gpu.server.jobs.
    _apply_result_into_state`) instead of silently mis-scattering; the
    dense-prefix fallback stays legal exactly when the payload covers the
    site's FULL time axis.
    """

    def __init__(self, variable: str, payload_times: int, site_times: int) -> None:
        self.variable = str(variable)
        self.payload_times = int(payload_times)
        self.site_times = int(site_times)
        super().__init__(
            f"variable {self.variable!r} payload covers {self.payload_times} of "
            f"{self.site_times} site time steps without matching manifest "
            "time_indices; refusing to scatter at dense offsets (the covered "
            "planes would land at the wrong global times)"
        )


@dataclass(frozen=True)
class VariableMeta:
    """One variable entry of a result manifest."""

    name: str
    shape: tuple[int, int, int]  # (time, rows, cols)
    dtype: str = "float32"
    nodata: str = NODATA

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": [int(v) for v in self.shape],
            "nodata": self.nodata,
        }

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(PATCH_DTYPE).itemsize


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def variable_meta(name: str, shape: Sequence[int]) -> VariableMeta:
    """Build a manifest variable entry from an array shape ``(time, rows, cols)``."""
    shape_tuple = tuple(int(v) for v in shape)
    if len(shape_tuple) != 3:
        raise PatchShapeError(
            f"variable {name!r} shape must be (time, rows, cols), got {shape_tuple}"
        )
    return VariableMeta(name=name, shape=shape_tuple)


def uncompressed_bytes(arrays: Mapping[str, np.ndarray], variables: Iterable[str]) -> bytes:
    """Serialize arrays in variable-major, time, row, column C-order LE order."""
    chunks: list[bytes] = []
    for name in variables:
        array = np.asarray(arrays[name])
        if array.ndim != 3:
            raise PatchShapeError(
                f"variable {name!r} must be 3-D (time, rows, cols), got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise PayloadDtypeError(
                f"variable {name!r} must be floating point, got {array.dtype}"
            )
        contiguous = np.ascontiguousarray(array, dtype=PATCH_DTYPE)
        chunks.append(contiguous.view(np.uint8).tobytes())
    return b"".join(chunks)


def compress_payload(raw: bytes, *, level: int = ZSTD_COMPRESSION_LEVEL) -> bytes:
    """Compress the uncompressed patch bytes with zstd."""
    return zstandard.ZstdCompressor(level=level).compress(raw)


def checksum_payload(payload: bytes) -> str:
    """SHA-256 over the payload bytes passed in, prefixed for the manifest.

    Called with the bytes actually served — the zstd payload (whose checksum
    is the durable manifest ``checksum``) or the identity bytes (whose
    checksum is returned in the response headers only).
    """
    return _CHECKSUM_PREFIX + hashlib.sha256(payload).hexdigest()


def select_payload_media_type(accept_header: str | None) -> str | None:
    """Pick the payload media type for an ``Accept`` header (negotiation).

    Returns :data:`PATCH_MEDIA_TYPE` (zstd, the contract default) or
    :data:`IDENTITY_MEDIA_TYPE`, or ``None`` when the client accepts neither.
    zstd wins when both are acceptable: identity exists for clients that
    *cannot* decompress zstd, not as a cheaper-to-verify alternative.
    Wildcards (``*/*``, ``application/*``) and a missing header mean "no
    preference" and keep the zstd default.
    """
    zstd_ok, identity_ok = True, False
    if accept_header:
        ranges = []
        for part in accept_header.split(","):
            media = part.split(";", 1)[0].strip().lower()
            if media:
                ranges.append(media)
        if ranges:
            def acceptable(media_type: str) -> bool:
                return (
                    media_type in ranges
                    or "*/*" in ranges
                    or (
                        media_type.startswith("application/")
                        and "application/*" in ranges
                    )
                )

            zstd_ok = acceptable(PATCH_MEDIA_TYPE)
            identity_ok = acceptable(IDENTITY_MEDIA_TYPE)
    if zstd_ok:
        return PATCH_MEDIA_TYPE
    if identity_ok:
        return IDENTITY_MEDIA_TYPE
    return None


def encode_payload(
    arrays: Mapping[str, np.ndarray],
    variables: Sequence[str],
    *,
    level: int = ZSTD_COMPRESSION_LEVEL,
) -> tuple[list[VariableMeta], bytes, str]:
    """Encode ``arrays`` into ``(variable metadata, payload, checksum)``.

    ``variables`` fixes the variable-major order of the uncompressed bytes and
    therefore of the manifest list.
    """
    unknown = [name for name in variables if name not in SUPPORTED_VARIABLES]
    if unknown:
        raise PatchCodecError(f"unsupported patch variables: {unknown}")
    missing = [name for name in variables if name not in arrays]
    if missing:
        raise PatchCodecError(f"missing arrays for variables: {missing}")
    meta = [variable_meta(name, np.asarray(arrays[name]).shape) for name in variables]
    payload = compress_payload(uncompressed_bytes(arrays, variables), level=level)
    return meta, payload, checksum_payload(payload)


def build_manifest(
    *,
    scenario_id: str,
    scene_version: int,
    window: RasterWindow,
    time_indices: Sequence[int],
    variables: Sequence[VariableMeta],
    payload_url: str,
    checksum: str,
    model_version: str,
    site_cache_version: str,
    site_id: str | None = None,
    exact: bool = True,
    metrics: Mapping[str, Any] | None = None,
    limitations: Sequence[str] = (),
    statistics: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Assemble a result manifest following the API contract field order."""
    manifest: dict[str, Any] = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scene_version": int(scene_version),
        "exact": bool(exact),
        "model_version": model_version,
        "site_cache_version": site_cache_version,
        "window": {
            "row_start": int(window.row_start),
            "row_stop": int(window.row_stop),
            "col_start": int(window.col_start),
            "col_stop": int(window.col_stop),
        },
        "time_indices": [int(t) for t in time_indices],
        "variables": [entry.to_dict() for entry in variables],
        "payload_url": payload_url,
        "compression": "zstd",
        "checksum": checksum,
    }
    if site_id is not None:
        manifest["site_id"] = site_id
    if statistics:
        manifest["statistics"] = {
            str(name): [dict(entry) for entry in per_time]
            for name, per_time in statistics.items()
        }
    manifest["metrics"] = dict(metrics or {})
    manifest["limitations"] = list(limitations)
    return manifest


def plane_statistics(array: np.ndarray) -> list[dict[str, Any]]:
    """Per-time-step finite statistics for one result variable.

    Served verbatim inside manifests (``statistics``) so view-only legend
    requests can be answered from metadata alone — no payload decode at
    all. Only finite values count; an all-nodata plane reports ``count: 0``
    with no min/max/mean, matching the legend contract.
    """
    array = np.asarray(array)
    if array.ndim != 3:
        raise PatchShapeError(
            f"variable must be 3-D (time, rows, cols) for statistics, got {array.shape}"
        )
    entries: list[dict[str, Any]] = []
    for index in range(int(array.shape[0])):
        finite = array[index][np.isfinite(array[index])]
        entry: dict[str, Any] = {"time_index": index, "count": int(finite.size)}
        if finite.size:
            entry["min"] = float(finite.min())
            entry["max"] = float(finite.max())
            entry["mean"] = float(finite.mean())
        entries.append(entry)
    return entries


def result_statistics(arrays: Mapping[str, np.ndarray]) -> dict[str, list[dict[str, Any]]]:
    """Statistics document for every variable of a result being published."""
    return {
        str(name): plane_statistics(np.asarray(arrays[name])) for name in arrays
    }


def write_result_files(
    directory: Path,
    manifest: Mapping[str, Any],
    payload: bytes,
) -> tuple[Path, Path]:
    """Atomically write ``manifest.json`` + ``payload.bin`` into ``directory``.

    Both files are written to temporary names in the same directory and renamed
    into place, so a reader never observes a partially written result.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest_tmp = directory / "manifest.json.tmp"
    payload_tmp = directory / "payload.bin.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    payload_tmp.write_bytes(payload)
    manifest_path = directory / "manifest.json"
    payload_path = directory / "payload.bin"
    manifest_tmp.replace(manifest_path)
    payload_tmp.replace(payload_path)
    return manifest_path, payload_path


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def expected_nbytes(variables: Sequence[Mapping[str, Any]]) -> int:
    """Total uncompressed size implied by the manifest variable entries."""
    total = 0
    for entry in variables:
        shape = tuple(int(v) for v in entry["shape"])
        dtype = str(entry.get("dtype", "float32"))
        if dtype != "float32":
            raise PayloadDtypeError(
                f"variable {entry.get('name')!r} dtype must be 'float32', got {dtype!r}"
            )
        if len(shape) != 3:
            raise PayloadShapeError(
                f"variable {entry.get('name')!r} shape must be (time, rows, cols), "
                f"got {shape}"
            )
        if any(v < 0 for v in shape):
            raise PayloadShapeError(
                f"variable {entry.get('name')!r} shape must be non-negative, got {shape}"
            )
        total += int(np.prod(shape)) * np.dtype(PATCH_DTYPE).itemsize
    return total


def decompress_payload(payload: bytes, manifest: Mapping[str, Any]) -> bytes:
    compression = str(manifest.get("compression", "zstd"))
    if compression != "zstd":
        raise PayloadCompressionError(
            f"unsupported compression {compression!r}; expected 'zstd'"
        )
    try:
        return zstandard.ZstdDecompressor().decompressobj().decompress(payload)
    except zstandard.ZstdError as error:
        raise PayloadCompressionError(f"payload is not valid zstd data: {error}") from error


def verify_payload_checksum(payload: bytes, checksum: str) -> None:
    actual = checksum_payload(payload)
    if actual != checksum:
        raise PayloadChecksumError(
            f"payload checksum mismatch: manifest {checksum}, actual {actual}"
        )


def decode_payload(
    manifest: Mapping[str, Any],
    payload: bytes,
    *,
    verify_checksum: bool = True,
    expected_variables: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Decode a zstd payload against its manifest, validating everything.

    Returns one float32 array per variable, shaped ``(time, rows, cols)`` in
    manifest (variable-major) order.
    """
    if verify_checksum:
        checksum = manifest.get("checksum")
        if not isinstance(checksum, str) or not checksum:
            raise PayloadChecksumError("manifest is missing a checksum")
        verify_payload_checksum(payload, checksum)
    raw = decompress_payload(payload, manifest)
    return _decode_raw(manifest, raw, expected_variables)


def decode_identity_payload(
    manifest: Mapping[str, Any],
    payload: bytes,
    *,
    expected_variables: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Decode an identity (uncompressed) payload served on request.

    The manifest ``checksum`` describes the durable zstd payload, so it
    cannot verify identity bytes; clients verify the response's
    ``X-SOLWEIG-Checksum``/``ETag`` headers (computed over the served bytes)
    instead. Everything else — schema version, variables, byte length,
    window cross-check — is validated exactly as for the zstd encoding.
    """
    return _decode_raw(manifest, payload, expected_variables)


def _decode_raw(
    manifest: Mapping[str, Any],
    raw: bytes,
    expected_variables: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Validate manifest structure and unpack uncompressed patch bytes."""
    if int(manifest.get("schema_version", -1)) != PATCH_SCHEMA_VERSION:
        raise PayloadSchemaError(
            f"patch schema version {manifest.get('schema_version')!r} is not "
            f"supported (expected {PATCH_SCHEMA_VERSION})"
        )
    variables = list(manifest.get("variables", []))
    if not variables:
        raise PayloadShapeError("manifest declares no variables")
    names = [str(entry["name"]) for entry in variables]
    if expected_variables is not None and tuple(names) != tuple(expected_variables):
        raise PayloadShapeError(
            f"manifest variables {names} do not match expected "
            f"{list(expected_variables)}"
        )
    total = expected_nbytes(variables)
    if len(raw) != total:
        raise PayloadLengthError(
            f"payload is {len(raw)} bytes but the manifest implies "
            f"{total} bytes"
        )
    itemsize = np.dtype(PATCH_DTYPE).itemsize
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for entry in variables:
        shape = tuple(int(v) for v in entry["shape"])
        count = int(np.prod(shape))
        block = raw[offset : offset + count * itemsize]
        array = np.frombuffer(block, dtype=PATCH_DTYPE).reshape(shape)
        arrays[str(entry["name"])] = np.array(array, dtype=np.float32, copy=True)
        offset += count * itemsize
    # Cross-check the manifest window against variable shapes (rows, cols).
    window = manifest.get("window")
    if isinstance(window, Mapping):
        height = int(window["row_stop"]) - int(window["row_start"])
        width = int(window["col_stop"]) - int(window["col_start"])
        for name, array in arrays.items():
            if array.shape[1] != height or array.shape[2] != width:
                raise PayloadShapeError(
                    f"variable {name!r} shape {array.shape} does not match the "
                    f"manifest window {height}x{width}"
                )
    return arrays


def payload_etag(checksum: str) -> str:
    """HTTP ETag header value for a checksum (``sha256:<hex>`` -> quoted)."""
    hex_digest = checksum.removeprefix(_CHECKSUM_PREFIX)
    return f'"sha256-{hex_digest}"'


# ---------------------------------------------------------------------------
# Single-plane extraction
# ---------------------------------------------------------------------------

#: Compressed-chunk size used when streaming a plane out of a zstd payload.
_DECOMPRESS_CHUNK = 64 * 1024


def _plane_span(manifest: Mapping[str, Any], variable: str) -> tuple[int, int, tuple[int, int, int]]:
    """Byte span ``(start, stop)`` of one variable inside the raw payload."""
    if int(manifest.get("schema_version", -1)) != PATCH_SCHEMA_VERSION:
        raise PayloadSchemaError(
            f"patch schema version {manifest.get('schema_version')!r} is not "
            f"supported (expected {PATCH_SCHEMA_VERSION})"
        )
    itemsize = np.dtype(PATCH_DTYPE).itemsize
    offset = 0
    for entry in manifest.get("variables", []):
        name = str(entry.get("name"))
        shape = tuple(int(v) for v in entry.get("shape", ()))
        count = int(np.prod(shape)) * itemsize
        if name == variable:
            if len(shape) != 3:
                raise PayloadShapeError(
                    f"variable {name!r} shape must be (time, rows, cols), got {shape}"
                )
            return offset, offset + count, shape
        offset += count
    names = [str(entry.get("name")) for entry in manifest.get("variables", [])]
    raise PayloadShapeError(f"variable {variable!r} is not in the manifest ({names})")


def decode_plane(
    manifest: Mapping[str, Any],
    payload: bytes,
    variable: str,
    time_index: int,
) -> np.ndarray:
    """Extract exactly one ``(rows, cols)`` plane of one variable.

    This is the decode-free legend fallback for manifests published before
    per-time-step ``statistics`` existed. The compressed checksum is verified
    first (it covers the zstd bytes, independent of decompression), then the
    payload is streamed through a zstd ``decompressobj`` and decompression
    STOPS as soon as the requested plane's bytes are complete — variables
    after it are never decompressed at all, and a plane early in the stream
    never materializes the tail. Memory is bounded by the plane's end offset
    in the *uncompressed* stream (zstd frames are not seekable, so the
    prefix before the plane is unavoidably decompressed).
    """
    checksum = manifest.get("checksum")
    if isinstance(checksum, str) and checksum:
        verify_payload_checksum(payload, checksum)
    start, stop, shape = _plane_span(manifest, variable)
    rows, cols = shape[1], shape[2]
    if not 0 <= int(time_index) < shape[0]:
        raise PayloadShapeError(
            f"time index {time_index} is outside variable {variable!r}'s "
            f"{shape[0]} time steps"
        )
    itemsize = np.dtype(PATCH_DTYPE).itemsize
    plane_start = start + int(time_index) * rows * cols * itemsize
    plane_stop = plane_start + rows * cols * itemsize
    dobj = zstandard.ZstdDecompressor().decompressobj()
    buffer = bytearray()
    for offset in range(0, len(payload), _DECOMPRESS_CHUNK):
        try:
            buffer += dobj.decompress(payload[offset : offset + _DECOMPRESS_CHUNK])
        except zstandard.ZstdError as error:
            raise PayloadCompressionError(
                f"payload is not valid zstd data: {error}"
            ) from error
        if len(buffer) >= plane_stop:
            break
    if len(buffer) < plane_stop:
        raise PayloadLengthError(
            f"payload decompressed to {len(buffer)} bytes but the requested "
            f"plane of {variable!r} ends at byte {plane_stop}"
        )
    plane = np.frombuffer(
        bytes(buffer[plane_start:plane_stop]), dtype=PATCH_DTYPE
    ).reshape(rows, cols)
    return np.array(plane, dtype=np.float32, copy=True)
