from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.bitmask import (
    bytes_per_pixel,
    create_packed_memmap,
    pack_visibility,
    packed_nbytes,
    set_patch_window,
    unpack_patch,
    unpack_visibility,
    weighted_sum_from_packed,
)


def test_pack_round_trip_non_byte_aligned_patch_count() -> None:
    rng = np.random.default_rng(4)
    dense = rng.integers(0, 2, size=(7, 11, 153), dtype=np.uint8)
    packed = pack_visibility(dense)
    assert packed.data.shape == (7, 11, 20)
    assert np.array_equal(unpack_visibility(packed), dense.astype(bool))


def test_unpack_single_patch_matches_dense() -> None:
    dense = np.zeros((4, 5, 10), dtype=np.uint8)
    dense[1:3, 2:4, 7] = 1
    packed = pack_visibility(dense)
    assert np.array_equal(unpack_patch(packed, 7), dense[..., 7].astype(bool))


def test_set_patch_window_updates_only_requested_window() -> None:
    packed = pack_visibility(np.zeros((6, 6, 9), dtype=np.uint8))
    set_patch_window(
        packed,
        patch_index=8,
        row_slice=slice(2, 5),
        col_slice=slice(1, 4),
        values=np.ones((3, 3), dtype=bool),
    )
    plane = unpack_patch(packed, 8)
    assert plane.sum() == 9
    assert plane[2:5, 1:4].all()


def test_weighted_sum_matches_dense_reference() -> None:
    rng = np.random.default_rng(7)
    dense = rng.integers(0, 2, size=(12, 8, 17), dtype=np.uint8)
    weights = rng.normal(size=17).astype(np.float32)
    packed = pack_visibility(dense)
    expected = np.sum(dense.astype(np.float32) * weights, axis=-1)
    actual = weighted_sum_from_packed(packed, weights)
    assert np.allclose(actual, expected, atol=1e-6)


def test_memory_size_is_approximately_one_bit_per_patch() -> None:
    assert bytes_per_pixel(153) == 20
    assert packed_nbytes((500, 500), 153) == 5_000_000
    dense_float32_bytes = 500 * 500 * 153 * 4
    assert packed_nbytes((500, 500), 153) < dense_float32_bytes / 30


def test_create_packed_memmap(tmp_path: Path) -> None:
    path = tmp_path / "visibility.npy"
    packed = create_packed_memmap(path, spatial_shape=(3, 4), patch_count=9)
    packed.data[...] = 0
    packed.data.flush()
    reopened = create_packed_memmap(path, spatial_shape=(3, 4), patch_count=9, mode="r+")
    assert reopened.data.shape == (3, 4, 2)


def test_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="binary"):
        pack_visibility(np.array([[[0, 2]]], dtype=np.uint8))


def test_big_endian_bit_order_round_trip() -> None:
    rng = np.random.default_rng(11)
    dense = rng.integers(0, 2, size=(3, 4, 19), dtype=np.uint8)
    packed = pack_visibility(dense, bitorder="big")
    assert np.array_equal(unpack_visibility(packed), dense.astype(bool))
    for patch_index in range(dense.shape[-1]):
        assert np.array_equal(
            unpack_patch(packed, patch_index),
            dense[..., patch_index].astype(bool),
        )


def test_set_patch_window_can_clear_bits_without_touching_other_patches() -> None:
    dense = np.ones((5, 5, 10), dtype=np.uint8)
    packed = pack_visibility(dense, bitorder="big")
    set_patch_window(
        packed,
        patch_index=7,
        row_slice=slice(1, 4),
        col_slice=slice(2, 5),
        values=np.zeros((3, 3), dtype=bool),
    )
    unpacked = unpack_visibility(packed)
    assert not unpacked[1:4, 2:5, 7].any()
    assert unpacked[..., 6].all()
    assert unpacked[..., 8].all()


def test_unpack_visibility_patch_range() -> None:
    dense = np.arange(2 * 3 * 13, dtype=np.uint8).reshape(2, 3, 13) % 2
    packed = pack_visibility(dense)
    assert np.array_equal(
        unpack_visibility(packed, patch_start=3, patch_stop=9),
        dense[..., 3:9].astype(bool),
    )
