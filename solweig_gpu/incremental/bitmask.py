# SPDX-License-Identifier: GPL-3.0-only
"""Bit-packed visibility storage for incremental SOLWEIG calculations.

The existing SVF cache stores binary patch visibility as float32. Packing the
patch axis reduces storage by up to 32x and allows an incremental worker to read
only the spatial window and patch bits needed for the current update.

Polarity convention (verified against ``svf_calculator`` in
``solweig_gpu/shadow.py``: ``svf = sum(weight * sh)``): every packed stack uses
**1 = sky visible / unshadowed, 0 = occluded**. This holds for all three dense
stacks written by ``save_svf_zip_npz_outputs`` (``shadowmat``, ``vegshadowmat``,
``vbshmat`` — each verified binary {0, 1} on the 500x500x153 fixture cache).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

#: Human-readable polarity of each dense source stack written by the SVF stage.
SHADOW_STACK_POLARITY: dict[str, str] = {
    "shadowmat": "1 = sky visible (building/ground shadowing only)",
    "vegshadowmat": "1 = sky visible through/above vegetation",
    "vbshmat": "1 = sky visible considering vegetation and buildings",
}

#: Dense npz key -> packed output file stem used by the conversion command.
PACKED_STACK_NAMES: dict[str, str] = {
    "shadowmat": "shadowmat_packed",
    "vegshadowmat": "vegshadowmat_packed",
    "vbshmat": "vbshmat_packed",
}


@dataclass(frozen=True, slots=True)
class PackedVisibility:
    """Packed binary visibility with the patch axis stored in the final dimension."""

    data: np.ndarray
    patch_count: int
    bitorder: str = "little"

    def __post_init__(self) -> None:
        if self.patch_count <= 0:
            raise ValueError("patch_count must be positive")
        if self.bitorder not in {"little", "big"}:
            raise ValueError("bitorder must be 'little' or 'big'")
        if self.data.dtype != np.uint8:
            raise TypeError("packed data must use uint8")
        expected_bytes = bytes_per_pixel(self.patch_count)
        if self.data.ndim < 1 or self.data.shape[-1] != expected_bytes:
            raise ValueError(
                f"packed final dimension must be {expected_bytes} bytes for "
                f"{self.patch_count} patches"
            )

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.data.shape[:-1]

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)



def bytes_per_pixel(patch_count: int) -> int:
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")
    return ceil(patch_count / 8)


def packed_nbytes(spatial_shape: Iterable[int], patch_count: int) -> int:
    shape = tuple(int(value) for value in spatial_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("spatial_shape must contain positive dimensions")
    return int(np.prod(shape, dtype=np.int64)) * bytes_per_pixel(patch_count)


def _as_binary_bool(visibility: np.ndarray) -> np.ndarray:
    array = np.asarray(visibility)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError("visibility must have a non-empty patch axis")
    if array.dtype == np.bool_:
        return array
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("visibility values must be binary (0 or 1)")
    return array.astype(np.bool_, copy=False)


def pack_visibility(visibility: np.ndarray, *, bitorder: str = "little") -> PackedVisibility:
    """Pack a dense binary array along its final patch axis."""
    if bitorder not in {"little", "big"}:
        raise ValueError("bitorder must be 'little' or 'big'")
    binary = _as_binary_bool(visibility)
    packed = np.packbits(binary, axis=-1, bitorder=bitorder)
    return PackedVisibility(packed, patch_count=binary.shape[-1], bitorder=bitorder)


def unpack_visibility(
    packed: PackedVisibility,
    *,
    patch_start: int = 0,
    patch_stop: int | None = None,
) -> np.ndarray:
    """Unpack a contiguous patch range without exposing padded bits."""
    stop = packed.patch_count if patch_stop is None else patch_stop
    if not 0 <= patch_start <= stop <= packed.patch_count:
        raise ValueError("invalid patch range")
    dense = np.unpackbits(
        packed.data,
        axis=-1,
        count=packed.patch_count,
        bitorder=packed.bitorder,
    )
    return dense[..., patch_start:stop].astype(np.bool_, copy=False)


def unpack_patch(packed: PackedVisibility, patch_index: int) -> np.ndarray:
    """Return one boolean patch plane without unpacking all patch planes."""
    if not 0 <= patch_index < packed.patch_count:
        raise IndexError("patch_index out of range")
    byte_index, bit_index = divmod(patch_index, 8)
    if packed.bitorder == "little":
        mask = np.uint8(1 << bit_index)
    else:
        mask = np.uint8(1 << (7 - bit_index))
    return (packed.data[..., byte_index] & mask) != 0


def set_patch_window(
    packed: PackedVisibility,
    *,
    patch_index: int,
    row_slice: slice,
    col_slice: slice,
    values: np.ndarray,
) -> None:
    """Update one patch plane in a 2-D spatial window in place."""
    if packed.data.ndim != 3:
        raise ValueError("set_patch_window requires [rows, cols, bytes] packed data")
    if not 0 <= patch_index < packed.patch_count:
        raise IndexError("patch_index out of range")

    target_shape = packed.data[row_slice, col_slice, 0].shape
    binary_values = _as_binary_bool(np.asarray(values)[..., np.newaxis])[..., 0]
    if binary_values.shape != target_shape:
        raise ValueError(f"values shape {binary_values.shape} does not match {target_shape}")

    byte_index, bit_index = divmod(patch_index, 8)
    if packed.bitorder == "little":
        bit = np.uint8(1 << bit_index)
    else:
        bit = np.uint8(1 << (7 - bit_index))

    target = packed.data[row_slice, col_slice, byte_index]
    target[...] = np.where(binary_values, target | bit, target & np.uint8(~bit))


def weighted_sum_from_packed(
    packed: PackedVisibility,
    weights: np.ndarray,
    *,
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Accumulate weighted visibility without materializing a dense 3-D array."""
    weight_array = np.asarray(weights)
    if weight_array.shape != (packed.patch_count,):
        raise ValueError(
            f"weights must have shape ({packed.patch_count},), got {weight_array.shape}"
        )
    result = np.zeros(packed.spatial_shape, dtype=output_dtype)
    for patch_index, weight in enumerate(weight_array):
        if weight != 0:
            result += unpack_patch(packed, patch_index).astype(output_dtype) * weight
    return result


def create_packed_memmap(
    path: str | Path,
    *,
    spatial_shape: tuple[int, ...],
    patch_count: int,
    mode: str = "w+",
) -> PackedVisibility:
    """Create or open a ``.npy`` memory map for packed visibility data."""
    if not spatial_shape or any(dimension <= 0 for dimension in spatial_shape):
        raise ValueError("spatial_shape must contain positive dimensions")
    final_shape = (*spatial_shape, bytes_per_pixel(patch_count))
    data = np.lib.format.open_memmap(path, mode=mode, dtype=np.uint8, shape=final_shape)
    return PackedVisibility(data=data, patch_count=patch_count)


def iter_patch_blocks(
    patch_count: int,
    *,
    block_size: int = 8,
) -> Iterator[tuple[int, int]]:
    """Yield half-open ``[start, stop)`` patch blocks aligned to byte boundaries.

    Streaming consumers process a handful of patches at a time instead of
    materialising all patch planes; aligning blocks to byte boundaries keeps
    every block independently packable/unpackable.
    """
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    start = 0
    while start < patch_count:
        # Round the stop up to a byte boundary (or the total patch count).
        stop = min(start + block_size, patch_count)
        stop = min(ceil(stop / 8) * 8, patch_count) if stop < patch_count else stop
        if stop <= start:  # pragma: no cover - defensive
            stop = min(start + 8, patch_count)
        yield start, stop
        start = stop


# ---------------------------------------------------------------------------
# T22: 2-bit extended vbsh storage (DESIGN.ko.md 8.1/§432, TASKS T22)
#
# The march's vbsh output is integer-valued f32 in {0, 1, 2}: the final
# graph ``1 - (thr(vbsh_acc) - vegsh_final)`` is exact small-integer float
# arithmetic, and vbsh == 2 arises ONLY in the one-step regime (the only
# accumulated step is s == 0, whose contribution the first-step exception
# DISCARDS via ``vbshvegsh.zero_()`` — leaving ``1 - (0 - 1)`` exactly 2.0;
# multi-step marches provably never emit 2). The 1-bit codec above refuses
# the 2.0-carrying regime loudly; this codec stores the full {0, 1, 2}
# domain for the T22 ablation (engagement/cost verdict), with the SAME
# typed-refusal discipline: values outside {0, 1, 2} are never truncated.
#
# Layout: 4 patches per byte, patch p at bit shift 2*(p & 3) of byte p >> 2,
# value stored as its 2-bit little-endian integer code (00/01/10). 153
# patches -> 39 bytes/pixel (vs 20 for 1-bit; ceil(2*153/8) = 39).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackedVisibility2Bit:
    """Packed ternary {0, 1, 2} visibility, patch axis in the final dimension."""

    data: np.ndarray
    patch_count: int

    def __post_init__(self) -> None:
        if self.patch_count <= 0:
            raise ValueError("patch_count must be positive")
        if self.data.dtype != np.uint8:
            raise TypeError("packed data must use uint8")
        expected_bytes = bytes_per_pixel_2bit(self.patch_count)
        if self.data.ndim < 1 or self.data.shape[-1] != expected_bytes:
            raise ValueError(
                f"packed final dimension must be {expected_bytes} bytes for "
                f"{self.patch_count} patches (2-bit packing)"
            )

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.data.shape[:-1]

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


def bytes_per_pixel_2bit(patch_count: int) -> int:
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")
    return ceil(2 * patch_count / 8)


def _as_ternary_u8(visibility: np.ndarray) -> np.ndarray:
    array = np.asarray(visibility)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError("visibility must have a non-empty patch axis")
    if not np.all((array == 0) | (array == 1) | (array == 2)):
        raise ValueError(
            "2-bit visibility values must be ternary (0, 1 or 2); the march "
            "domain is exactly {0, 1, 2} — refusing to truncate any other value"
        )
    return array.astype(np.uint8, copy=False)


def pack_visibility_2bit(visibility: np.ndarray) -> PackedVisibility2Bit:
    """Pack a dense {0, 1, 2} array along its final patch axis (2 bits each)."""
    values = _as_ternary_u8(visibility)
    patch_count = values.shape[-1]
    padded = patch_count + (-patch_count % 4)  # pad to whole bytes
    if padded != patch_count:
        pad = np.zeros(values.shape[:-1] + (padded - patch_count,), dtype=np.uint8)
        values = np.concatenate([values, pad], axis=-1)
    grouped = values.reshape(values.shape[:-1] + (padded // 4, 4))
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    packed = (grouped << shifts).sum(axis=-1, dtype=np.uint8)
    return PackedVisibility2Bit(packed, patch_count=patch_count)


def unpack_visibility_2bit(packed: PackedVisibility2Bit) -> np.ndarray:
    """Unpack to uint8 {0, 1, 2} values (all patches, padded bits never read)."""
    grouped = packed.data[..., :, np.newaxis] >> np.array([0, 2, 4, 6], dtype=np.uint8)
    values = (grouped & np.uint8(3)).reshape(packed.data.shape[:-1] + (-1,))
    return values[..., : packed.patch_count]


def unpack_patch_2bit(packed: PackedVisibility2Bit, patch_index: int) -> np.ndarray:
    """Return one uint8 {0, 1, 2} patch plane without unpacking all patches."""
    if not 0 <= patch_index < packed.patch_count:
        raise IndexError("patch_index out of range")
    byte_index, bit_shift = patch_index >> 2, (patch_index & 3) * 2
    return (packed.data[..., byte_index] >> np.uint8(bit_shift)) & np.uint8(3)


def set_patch_window_2bit(
    packed: PackedVisibility2Bit,
    *,
    patch_index: int,
    row_slice: slice,
    col_slice: slice,
    values: np.ndarray,
) -> None:
    """Update one 2-bit patch plane in a 2-D spatial window in place."""
    if packed.data.ndim != 3:
        raise ValueError(
            "set_patch_window_2bit requires [rows, cols, bytes] packed data"
        )
    if not 0 <= patch_index < packed.patch_count:
        raise IndexError("patch_index out of range")

    target_shape = packed.data[row_slice, col_slice, 0].shape
    ternary = _as_ternary_u8(np.asarray(values)[..., np.newaxis])[..., 0]
    if ternary.shape != target_shape:
        raise ValueError(
            f"values shape {ternary.shape} does not match {target_shape}"
        )

    byte_index, bit_shift = patch_index >> 2, (patch_index & 3) * 2
    mask = np.uint8(3 << bit_shift)
    target = packed.data[row_slice, col_slice, byte_index]
    target[...] = (target & np.uint8(~mask)) | (ternary << np.uint8(bit_shift))


def weighted_sum_from_packed_2bit(
    packed: PackedVisibility2Bit,
    weights: np.ndarray,
    *,
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Accumulate weighted ternary visibility without a dense 3-D array."""
    weight_array = np.asarray(weights)
    if weight_array.shape != (packed.patch_count,):
        raise ValueError(
            f"weights must have shape ({packed.patch_count},), got {weight_array.shape}"
        )
    result = np.zeros(packed.spatial_shape, dtype=output_dtype)
    for patch_index, weight in enumerate(weight_array):
        if weight != 0:
            result += (
                unpack_patch_2bit(packed, patch_index).astype(output_dtype) * weight
            )
    return result


def spatial_window_view_2bit(
    packed: PackedVisibility2Bit,
    row_slice: slice,
    col_slice: slice,
) -> PackedVisibility2Bit:
    """Zero-copy 2-bit packed view of a 2-D spatial window (half-open slices)."""
    if packed.data.ndim != 3:
        raise ValueError(
            "spatial_window_view_2bit requires [rows, cols, bytes] packed data"
        )
    view = packed.data[row_slice, col_slice, :]
    return PackedVisibility2Bit(data=view, patch_count=packed.patch_count)


def spatial_window_view(
    packed: PackedVisibility,
    row_slice: slice,
    col_slice: slice,
) -> PackedVisibility:
    """Zero-copy packed view of a 2-D spatial window (half-open slices).

    The byte axis is untouched, so the view remains a valid
    :class:`PackedVisibility` and all patch data for the window is retained.
    """
    if packed.data.ndim != 3:
        raise ValueError("spatial_window_view requires [rows, cols, bytes] packed data")
    view = packed.data[row_slice, col_slice, :]
    return PackedVisibility(data=view, patch_count=packed.patch_count, bitorder=packed.bitorder)
