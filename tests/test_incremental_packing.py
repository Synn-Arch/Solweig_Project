# SPDX-License-Identifier: GPL-3.0-only
"""P3 tests: dense-to-packed SVF stack conversion, iterators, weighted reductions.

PACK-001 (20 bytes/pixel at 153 patches), PACK-002 (weighted outputs match
dense reference), plus the conversion-command contract: binary validation,
manifest checksums, deterministic conversion, and warm-load verification.
"""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest

from solweig_gpu.incremental.bitmask import (
    PACKED_STACK_NAMES,
    SHADOW_STACK_POLARITY,
    bytes_per_pixel,
    iter_patch_blocks,
    pack_visibility,
    packed_nbytes,
    spatial_window_view,
    unpack_patch,
    unpack_visibility,
    weighted_sum_from_packed,
)
from solweig_gpu.incremental.pack_svf import (
    convert_npz,
    load_packed_dir,
    sha256_file,
)

ROWS, COLS, PATCHES = 37, 23, 153


def _binary_stack(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(ROWS, COLS, PATCHES)).astype(np.float32)


class TestPackingBudget:
    def test_pack_001_bytes_per_pixel(self) -> None:
        """PACK-001: 153 patches pack to exactly 20 bytes per pixel."""
        assert bytes_per_pixel(153) == 20
        assert packed_nbytes((500, 500), 153) == 500 * 500 * 20
        dense = 500 * 500 * 153 * 4
        ratio = dense / packed_nbytes((500, 500), 153)
        assert ratio == pytest.approx(30.6, rel=0.01)

    def test_polarity_documented_for_every_stack(self) -> None:
        assert set(SHADOW_STACK_POLARITY) == set(PACKED_STACK_NAMES)
        for text in SHADOW_STACK_POLARITY.values():
            assert "1 = sky visible" in text


class TestIterPatchBlocks:
    def test_blocks_cover_all_patches_exactly_once(self) -> None:
        blocks = list(iter_patch_blocks(153, block_size=8))
        assert blocks[0] == (0, 8)
        assert sum(stop - start for start, stop in blocks) == 153
        flat = [p for start, stop in blocks for p in range(start, stop)]
        assert flat == list(range(153))

    def test_blocks_byte_aligned_except_tail(self) -> None:
        for start, stop in iter_patch_blocks(153, block_size=8):
            if stop != 153:
                assert stop % 8 == 0

    def test_single_patch(self) -> None:
        assert list(iter_patch_blocks(1)) == [(0, 1)]


class TestSpatialWindowView:
    def test_view_equals_dense_slice(self) -> None:
        rng = np.random.default_rng(7)
        dense = _binary_stack(rng)
        packed = pack_visibility(dense)
        view = spatial_window_view(packed, slice(5, 30), slice(2, 21))
        expected = dense[5:30, 2:21, :]
        np.testing.assert_array_equal(unpack_visibility(view), expected.astype(bool))

    def test_view_is_zero_copy(self) -> None:
        dense = np.zeros((4, 4, 8), dtype=np.float32)
        packed = pack_visibility(dense)
        view = spatial_window_view(packed, slice(0, 2), slice(0, 2))
        assert view.data.base is packed.data or view.data.base is not None


class TestWeightedReductions:
    def test_pack_002_random_weights_match_dense(self) -> None:
        rng = np.random.default_rng(11)
        dense = _binary_stack(rng)
        weights = rng.random(PATCHES).astype(np.float64)
        packed = pack_visibility(dense)
        got = weighted_sum_from_packed(packed, weights, output_dtype=np.float64)
        expected = dense.astype(np.float64) @ weights
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_svf_style_reduction_matches_dense(self) -> None:
        """SVF is a positive-weight reduction of the shadow stack."""
        rng = np.random.default_rng(13)
        dense = _binary_stack(rng)
        weights = np.full(PATCHES, 1.0 / PATCHES)
        packed = pack_visibility(dense)
        got = weighted_sum_from_packed(packed, weights, output_dtype=np.float32)
        expected = (dense * (1.0 / PATCHES)).sum(axis=-1)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)

    def test_directional_window_reduction(self) -> None:
        """Directional SVF = weighted reduction restricted to a spatial window."""
        rng = np.random.default_rng(17)
        dense = _binary_stack(rng)
        weights = rng.random(PATCHES)
        packed = spatial_window_view(pack_visibility(dense), slice(3, 20), slice(4, 15))
        got = weighted_sum_from_packed(packed, weights, output_dtype=np.float64)
        expected = dense[3:20, 4:15, :].astype(np.float64) @ weights
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


class TestRandomPatchMutation:
    def test_mutations_preserve_other_patches(self) -> None:
        rng = np.random.default_rng(19)
        dense = _binary_stack(rng)
        packed = pack_visibility(dense)
        data = packed.data.copy()

        for _ in range(25):
            patch = int(rng.integers(0, PATCHES))
            rows = slice(*(sorted(rng.integers(0, ROWS, size=2).tolist())))
            cols = slice(*(sorted(rng.integers(0, COLS, size=2).tolist())))
            values = rng.integers(0, 2, size=(
                rows.stop - rows.start, cols.stop - cols.start)).astype(np.float32)
            from solweig_gpu.incremental.bitmask import set_patch_window
            set_patch_window(packed, patch_index=patch, row_slice=rows,
                             col_slice=cols, values=values)
            dense[rows, cols, patch] = values

        np.testing.assert_array_equal(unpack_visibility(packed), dense.astype(bool))
        assert (data != packed.data).any()


class TestConversionCommand:
    @pytest.fixture()
    def npz_path(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(23)
        path = tmp_path / "shadowmats_0_0.npz"
        np.savez_compressed(
            path,
            **{key: _binary_stack(rng) for key in PACKED_STACK_NAMES},
        )
        return path

    def test_round_trip_and_manifest(self, npz_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "packed"
        manifest = convert_npz(npz_path, out_dir)
        assert (out_dir / "manifest.json").is_file()
        assert manifest["patch_count"] == PATCHES
        assert manifest["bytes_per_pixel"] == 20
        for entry in manifest["stacks"].values():
            assert entry["nbytes"] == ROWS * COLS * 20

        with np.load(npz_path) as archive:
            stacks = load_packed_dir(out_dir)
            for key, packed in stacks.items():
                np.testing.assert_array_equal(
                    unpack_visibility(packed), archive[key].astype(bool)
                )

    def test_deterministic_conversion(self, npz_path: Path, tmp_path: Path) -> None:
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        convert_npz(npz_path, out_a)
        convert_npz(npz_path, out_b)
        for stem in [f"{name}.npy" for name in PACKED_STACK_NAMES.values()] + ["manifest.json"]:
            assert sha256_file(out_a / stem) == sha256_file(out_b / stem), stem

    def test_nonbinary_rejected(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(29)
        bad = rng.integers(0, 2, size=(8, 8, PATCHES)).astype(np.float32)
        bad[0, 0, 0] = 0.5
        path = tmp_path / "bad.npz"
        np.savez_compressed(path, **{key: bad for key in PACKED_STACK_NAMES})
        with pytest.raises(ValueError, match="not binary"):
            convert_npz(path, tmp_path / "out")

    def test_missing_stack_rejected(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(31)
        path = tmp_path / "partial.npz"
        np.savez_compressed(path, shadowmat=_binary_stack(rng))
        with pytest.raises(ValueError, match="missing stacks"):
            convert_npz(path, tmp_path / "out")

    def test_validate_only_writes_nothing(self, npz_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "packed"
        manifest = convert_npz(npz_path, out_dir, validate_only=True)
        assert manifest["patch_count"] == PATCHES
        assert not out_dir.exists()

    def test_checksum_tamper_detected(self, npz_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "packed"
        convert_npz(npz_path, out_dir)
        target = out_dir / "shadowmat_packed.npy"
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 0xFF
        target.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_packed_dir(out_dir)

    def test_single_patch_plane_extraction(self, npz_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "packed"
        convert_npz(npz_path, out_dir)
        stacks = load_packed_dir(out_dir)
        with np.load(npz_path) as archive:
            dense = archive["shadowmat"]
        for patch in (0, 1, 76, 152):
            np.testing.assert_array_equal(
                unpack_patch(stacks["shadowmat"], patch),
                dense[:, :, patch].astype(bool),
            )
