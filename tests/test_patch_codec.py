# SPDX-License-Identifier: GPL-3.0-only
"""Patch codec tests: byte order, zstd round trip, and rejection of tampered
or inconsistent payloads (API gate API-003)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zstandard

from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.server import patch_codec
from solweig_gpu.server.patch_codec import (
    PayloadChecksumError,
    PayloadCompressionError,
    PayloadDtypeError,
    PayloadLengthError,
    PayloadSchemaError,
    PayloadShapeError,
    build_manifest,
    checksum_payload,
    decode_payload,
    encode_payload,
    payload_etag,
    uncompressed_bytes,
    write_result_files,
)


def _sample_arrays(t: int = 2, h: int = 3, w: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260901)
    return {
        "utci": rng.uniform(10, 40, size=(t, h, w)).astype(np.float32),
        "tmrt": rng.uniform(20, 70, size=(t, h, w)).astype(np.float32),
    }


class TestByteOrder:
    def test_variable_major_time_row_column_c_order(self) -> None:
        arrays = _sample_arrays()
        raw = uncompressed_bytes(arrays, ["utci", "tmrt"])
        expected = (
            np.ascontiguousarray(arrays["utci"], dtype="<f4").tobytes()
            + np.ascontiguousarray(arrays["tmrt"], dtype="<f4").tobytes()
        )
        assert raw == expected
        # First four bytes are the very first variable's first time step,
        # first row, first column (little-endian float32).
        first = arrays["utci"][0, 0, 0]
        assert raw[:4] == np.float32(first).astype("<f4").tobytes()

    def test_round_trip_through_zstd(self) -> None:
        arrays = _sample_arrays()
        meta, payload, checksum = encode_payload(arrays, ["utci", "tmrt"])
        manifest = build_manifest(
            scenario_id="scn_x",
            scene_version=3,
            window=RasterWindow(0, 3, 0, 4),
            time_indices=[0, 1],
            variables=meta,
            payload_url="/payload",
            checksum=checksum,
            model_version="test",
            site_cache_version="site:cache-1",
        )
        assert manifest["compression"] == "zstd"
        assert zstandard.ZstdDecompressor().decompressobj().decompress(payload)
        decoded = decode_payload(manifest, payload)
        assert set(decoded) == {"utci", "tmrt"}
        for name, array in arrays.items():
            np.testing.assert_array_equal(decoded[name], array)
            assert decoded[name].dtype == np.float32

    def test_manifest_contract_fields(self) -> None:
        arrays = _sample_arrays()
        meta, payload, checksum = encode_payload(arrays, ["utci"])
        manifest = build_manifest(
            scenario_id="scn_x",
            scene_version=18,
            window=RasterWindow(96, 288, 160, 352),
            time_indices=[12],
            variables=meta,
            payload_url="/api/v1/scenarios/scn_x/results/18/payload",
            checksum=checksum,
            model_version="solweig-gpu-2.0.0+incremental.cache.1",
            site_cache_version="campus-1km-v1:cache-1",
            site_id="campus-1km-v1",
            metrics={"mean_utci_delta_c": -0.84},
            limitations=["limit-a"],
        )
        assert manifest["schema_version"] == 1
        assert manifest["scene_version"] == 18
        assert manifest["exact"] is True
        assert manifest["site_id"] == "campus-1km-v1"
        assert manifest["site_cache_version"] == "campus-1km-v1:cache-1"
        assert manifest["model_version"] == "solweig-gpu-2.0.0+incremental.cache.1"
        assert manifest["window"] == {
            "row_start": 96, "row_stop": 288, "col_start": 160, "col_stop": 352,
        }
        assert manifest["variables"] == [
            {"name": "utci", "dtype": "float32", "shape": [2, 3, 4], "nodata": "nan"}
        ]
        assert manifest["time_indices"] == [12]
        assert manifest["compression"] == "zstd"
        assert manifest["checksum"].startswith("sha256:")
        assert manifest["metrics"] == {"mean_utci_delta_c": -0.84}
        assert manifest["limitations"] == ["limit-a"]
        assert manifest["payload_url"].endswith("/18/payload")


class TestRejections:
    def _manifest_for(self, arrays, **overrides) -> tuple[dict, bytes]:
        meta, payload, checksum = encode_payload(arrays, ["utci", "tmrt"])
        manifest = build_manifest(
            scenario_id="scn_x",
            scene_version=1,
            window=RasterWindow(0, arrays["utci"].shape[1], 0, arrays["utci"].shape[2]),
            time_indices=list(range(arrays["utci"].shape[0])),
            variables=meta,
            payload_url="/payload",
            checksum=checksum,
            model_version="test",
            site_cache_version="site:cache-1",
        )
        manifest.update(overrides)
        return manifest, payload

    def test_tampered_checksum_rejected(self) -> None:
        manifest, payload = self._manifest_for(_sample_arrays())
        corrupted = payload[:-1] + bytes([payload[-1] ^ 0xFF])
        with pytest.raises(PayloadChecksumError, match="checksum mismatch"):
            decode_payload(manifest, corrupted)

    def test_truncated_payload_rejected(self) -> None:
        """A half-frame that decompresses short fails the length check."""
        manifest, payload = self._manifest_for(_sample_arrays())
        truncated = payload[: len(payload) // 2]
        manifest["checksum"] = checksum_payload(truncated)
        with pytest.raises(PayloadLengthError):
            decode_payload(manifest, truncated)

    def test_payload_shorter_than_manifest_rejected(self) -> None:
        """Bytes removed from the end: checksum catches it first."""
        manifest, payload = self._manifest_for(_sample_arrays())
        with pytest.raises(PayloadChecksumError):
            decode_payload(manifest, payload[:-8])

    def test_shape_mismatch_rejected(self) -> None:
        arrays = _sample_arrays()
        manifest, payload = self._manifest_for(arrays)
        manifest["variables"][0]["shape"] = [9, 9, 9]
        with pytest.raises(PayloadLengthError):
            decode_payload(manifest, payload)

    def test_window_shape_mismatch_rejected(self) -> None:
        arrays = _sample_arrays()
        manifest, payload = self._manifest_for(arrays)
        manifest["window"] = {"row_start": 0, "row_stop": 7, "col_start": 0, "col_stop": 4}
        with pytest.raises(PayloadShapeError, match="utci"):
            decode_payload(manifest, payload)

    def test_dtype_mismatch_rejected(self) -> None:
        arrays = _sample_arrays()
        manifest, payload = self._manifest_for(arrays)
        manifest["variables"][0]["dtype"] = "float64"
        with pytest.raises(PayloadDtypeError, match="utci"):
            decode_payload(manifest, payload)

    def test_non_zstd_payload_rejected(self) -> None:
        arrays = _sample_arrays()
        manifest, _ = self._manifest_for(arrays)
        garbage = b"this is definitely not zstd" * 4
        manifest["checksum"] = checksum_payload(garbage)
        with pytest.raises(PayloadCompressionError):
            decode_payload(manifest, garbage)

    def test_wrong_compression_field_rejected(self) -> None:
        manifest, payload = self._manifest_for(_sample_arrays())
        manifest["compression"] = "gzip"
        with pytest.raises(PayloadCompressionError, match="gzip"):
            decode_payload(manifest, payload)

    def test_unsupported_schema_version_rejected(self) -> None:
        manifest, payload = self._manifest_for(_sample_arrays())
        manifest["schema_version"] = 99
        with pytest.raises(PayloadSchemaError):
            decode_payload(manifest, payload)

    def test_expected_variables_mismatch_rejected(self) -> None:
        manifest, payload = self._manifest_for(_sample_arrays())
        with pytest.raises(PayloadShapeError, match="do not match expected"):
            decode_payload(manifest, payload, expected_variables=["tmrt", "utci"])


class TestFilesAndHelpers:
    def test_write_result_files_atomic_layout(self, tmp_path: Path) -> None:
        arrays = _sample_arrays()
        meta, payload, checksum = encode_payload(arrays, ["utci", "tmrt"])
        manifest = build_manifest(
            scenario_id="scn_x",
            scene_version=4,
            window=RasterWindow(0, 3, 0, 4),
            time_indices=[0, 1],
            variables=meta,
            payload_url="/payload",
            checksum=checksum,
            model_version="test",
            site_cache_version="site:cache-1",
        )
        directory = tmp_path / "results" / "4"
        manifest_path, payload_path = write_result_files(directory, manifest, payload)
        assert manifest_path.name == "manifest.json"
        assert payload_path.name == "payload.bin"
        assert not list(directory.glob("*.tmp"))
        stored_manifest = json.loads(manifest_path.read_text())
        decoded = decode_payload(stored_manifest, payload_path.read_bytes())
        np.testing.assert_array_equal(decoded["utci"], arrays["utci"])

    def test_etag_matches_checksum_format(self) -> None:
        checksum = checksum_payload(b"payload-bytes")
        assert payload_etag(checksum) == f'"sha256-{checksum.removeprefix("sha256:")}"'

    def test_unsupported_variable_rejected(self) -> None:
        arrays = _sample_arrays()
        with pytest.raises(patch_codec.PatchCodecError, match="unsupported"):
            encode_payload(arrays, ["utci", "wind_speed"])


class TestStatistics:
    """Per-time-step manifest statistics (decode-free legends, u-d4)."""

    def test_statistics_match_full_decode_values(self) -> None:
        arrays = _sample_arrays()
        arrays["utci"][1, 0, 0] = np.nan  # nodata cells must not count
        stats = patch_codec.result_statistics(arrays)
        utci = stats["utci"]
        assert [entry["time_index"] for entry in utci] == [0, 1]
        for index, entry in enumerate(utci):
            finite = arrays["utci"][index][np.isfinite(arrays["utci"][index])]
            assert entry["count"] == int(finite.size)
            assert entry["min"] == float(finite.min())
            assert entry["max"] == float(finite.max())
            assert entry["mean"] == pytest.approx(float(finite.mean()))

    def test_all_nodata_plane_reports_zero_count_without_min_max_mean(self) -> None:
        arrays = _sample_arrays()
        arrays["utci"][1, :, :] = np.nan
        entry = patch_codec.result_statistics(arrays)["utci"][1]
        assert entry == {"time_index": 1, "count": 0}

    def test_build_manifest_carries_statistics_when_given(self) -> None:
        arrays = _sample_arrays()
        meta, payload, checksum = encode_payload(arrays, ["utci"])
        manifest = build_manifest(
            scenario_id="scn_x",
            scene_version=1,
            window=RasterWindow(0, 3, 0, 4),
            time_indices=[0, 1],
            variables=meta,
            payload_url="/payload",
            checksum=checksum,
            model_version="test",
            site_cache_version="site:cache-1",
            statistics=patch_codec.result_statistics(arrays),
        )
        assert manifest["statistics"]["utci"][0]["time_index"] == 0
        assert "min" in manifest["statistics"]["utci"][0]


class TestPlaneExtraction:
    """decode_plane: bounded streaming fallback for statistics-less manifests."""

    def _stored(self) -> tuple[dict, bytes, dict[str, np.ndarray]]:
        arrays = _sample_arrays()
        meta, payload, checksum = encode_payload(arrays, ["utci", "tmrt"])
        manifest = build_manifest(
            scenario_id="scn_x",
            scene_version=1,
            window=RasterWindow(0, arrays["utci"].shape[1], 0, arrays["utci"].shape[2]),
            time_indices=list(range(arrays["utci"].shape[0])),
            variables=meta,
            payload_url="/payload",
            checksum=checksum,
            model_version="test",
            site_cache_version="site:cache-1",
        )
        return manifest, payload, arrays

    def test_plane_matches_full_decode_for_every_variable_and_time(self) -> None:
        manifest, payload, arrays = self._stored()
        for name in ("utci", "tmrt"):
            for index in range(arrays[name].shape[0]):
                np.testing.assert_array_equal(
                    patch_codec.decode_plane(manifest, payload, name, index),
                    arrays[name][index],
                )

    def test_plane_verifies_checksum_before_decompressing(self) -> None:
        manifest, payload, _arrays = self._stored()
        corrupted = payload[:-1] + bytes([payload[-1] ^ 0xFF])
        with pytest.raises(PayloadChecksumError, match="checksum mismatch"):
            patch_codec.decode_plane(manifest, corrupted, "utci", 0)

    def test_plane_rejects_time_index_outside_shape(self) -> None:
        manifest, payload, _arrays = self._stored()
        with pytest.raises(PayloadShapeError, match="outside variable"):
            patch_codec.decode_plane(manifest, payload, "utci", 9)

    def test_plane_rejects_unknown_variable(self) -> None:
        manifest, payload, _arrays = self._stored()
        with pytest.raises(PayloadShapeError, match="not in the manifest"):
            patch_codec.decode_plane(manifest, payload, "shadow", 0)

    def test_stream_prefix_yields_plane_as_soon_as_complete(self) -> None:
        """The early-stop contract: a growing compressed prefix produces the
        target plane before the whole stream is consumed."""
        manifest, payload, arrays = self._stored()
        dobj = zstandard.ZstdDecompressor().decompressobj()
        needed = int(arrays["utci"][0].nbytes)
        produced = bytearray()
        consumed = 0
        step = 8
        while len(produced) < needed and consumed < len(payload):
            produced += dobj.decompress(payload[consumed : consumed + step])
            consumed += step
        assert len(produced) >= needed
        plane = np.frombuffer(
            bytes(produced[:needed]), dtype=patch_codec.PATCH_DTYPE
        ).reshape(arrays["utci"].shape[1], arrays["utci"].shape[2])
        np.testing.assert_array_equal(plane, arrays["utci"][0])
