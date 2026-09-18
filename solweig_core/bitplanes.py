# SPDX-License-Identifier: GPL-3.0-only
"""Packed binary per-patch state for the SVF fold (TASKS T07, DESIGN 3.4).

Layout (mirrors the R4 ``PackedVisibility`` of
``solweig_gpu/incremental/bitmask.py``, which is torch-importing oracle
code — this module is the torch-free standalone form):

* one ``uint8`` array of shape ``(rows, cols, bytes_per_patch(P))`` where
  ``P`` is the patch count (153 for ``patch_option=2``);
* the patch axis is the LAST axis so a cell's whole patch state is one
  contiguous byte group;
* little bit order: patch ``p`` lives in byte ``p // 8`` of the group,
  bit ``p & 7`` (``np.packbits(..., bitorder="little")`` order);
* tail bits ``[P, 8 * bytes)`` are ALWAYS zero — padding is deterministic,
  so two packs of the same logical values are byte-identical regardless
  of the source array's memory layout (C/Fortran/strided view).

Binary fence (typed refusal, never a bool collapse): the producer state
is exactly ``{0.0, 1.0}`` by VALUE membership (``x == 0.0 or x == 1.0``,
like the R4 fence). ``-0.0`` is ``== 0.0`` and packs as zero; any other
value — including the one-step regime's ``vbsh == 2.0``, NaN, and ±inf —
is refused with :class:`ValueError`. The one-step regime therefore cannot
enter the packed representation silently; its fallback lives in the
solver (out of scope here).

Unique-writer updates (:func:`set_patch_window`) are guarded at CHUNK
granularity (:data:`CHUNK_CELLS` = 64): claim sets are checked for
overlap BEFORE any write (:func:`verify_chunk_ownership`), and exact
tile partitions are checked by :func:`verify_chunk_partition`. Two
threads never read-modify-write the same packed byte: a byte group
belongs to exactly one cell, a cell belongs to exactly one chunk, and a
chunk is owned by exactly one writer.

Roundtrip contract: ``unpack_bits(pack_bits(x)) == (x != 0)`` and
``pack_bits(unpack_bits(p)).data == p.data`` — bit-exact, including the
deterministic tail padding (gates in tests/ultrafast/test_bitplanes.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "CHUNK_CELLS",
    "ChunkOwnershipError",
    "PackedBits",
    "bytes_per_patch",
    "cell_chunk",
    "chunk_grid_index",
    "chunk_slices",
    "chunk_window",
    "chunks_overlapping_mask",
    "pack_bits",
    "unpack_bits",
    "unpack_patch",
    "set_patch_window",
    "verify_chunk_ownership",
    "verify_chunk_partition",
]

#: Chunk side in cells. Ownership of packed updates is granted at chunk
#: granularity (64 x 64 cells); mirrors the R4 incremental state chunking.
CHUNK_CELLS = 64


class ChunkOwnershipError(RuntimeError):
    """Two writers claimed the same packed chunk (shared-byte race)."""


@dataclass(frozen=True)
class PackedBits:
    """A packed binary patch state: ``data`` is ``uint8 (rows, cols, B)``,
    patch axis last, little bit order, deterministic zero tail."""

    data: np.ndarray
    patch_count: int

    def __post_init__(self) -> None:
        arr = self.data
        if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(np.uint8):
            raise ValueError(
                f"PackedBits.data must be uint8, got {arr if not isinstance(arr, np.ndarray) else arr.dtype}"
            )
        if arr.ndim != 3:
            raise ValueError(
                f"PackedBits.data must be (rows, cols, bytes), got {arr.shape}"
            )
        if not int(self.patch_count) > 0:
            raise ValueError(f"patch_count must be positive, got {self.patch_count}")
        if arr.shape[2] != bytes_per_patch(int(self.patch_count)):
            raise ValueError(
                f"byte axis {arr.shape[2]} does not fit patch_count "
                f"{self.patch_count} (need {bytes_per_patch(int(self.patch_count))})"
            )

    @property
    def rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def cols(self) -> int:
        return int(self.data.shape[1])


def bytes_per_patch(patch_count: int) -> int:
    """Number of bytes per cell holding ``patch_count`` little-order bits."""
    patch_count = int(patch_count)
    if patch_count <= 0:
        raise ValueError(f"patch_count must be positive, got {patch_count}")
    return (patch_count + 7) // 8


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def _as_binary_bits(dense, ndim: int = 3) -> np.ndarray:
    """Value-membership binary fence -> bool array, layout independent.

    Accepts bool directly; floats/integers must be exactly 0.0/1.0 by
    value (-0.0 == 0.0 passes and packs as zero; NaN never equals either
    and is refused). Anything else — including 2.0, the one-step regime
    vbsh — is a typed refusal, never a silent ``!= 0`` bool collapse.
    """
    arr = np.asarray(dense)
    if arr.ndim != ndim:
        shape_name = {2: "(rows, cols)", 3: "(rows, cols, patches)"}[ndim]
        raise ValueError(
            f"dense patch state must be {shape_name}, got {arr.shape}"
        )
    if arr.dtype == np.bool_:
        return arr
    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        raise ValueError(
            f"dense patch state must be bool or numeric, got {arr.dtype}"
        )
    if arr.dtype not in (np.float32, np.float64):
        # integers and exotic float sizes still go through the value fence
        arr = arr.astype(np.float64, copy=False)
    is_zero = arr == 0.0
    is_one = arr == 1.0
    bad = ~(is_zero | is_one)
    if bool(bad.any()):
        first = np.argwhere(bad)[0]
        raise ValueError(
            "dense patch state is not binary: values must be exactly "
            f"0.0 or 1.0 (first offender at {tuple(int(x) for x in first)} "
            f"= {arr[tuple(int(x) for x in first)]!r}; the one-step regime "
            "vbsh == 2.0 is not packable by design)"
        )
    return is_one


def pack_bits(dense) -> PackedBits:
    """Pack a binary patch cube into :class:`PackedBits`.

    Deterministic: the same logical values pack to identical bytes from
    any source layout, and tail bits are always zero.
    """
    bits = _as_binary_bits(dense)
    patch_count = int(bits.shape[2])
    data = np.packbits(
        np.asanyarray(bits, dtype=np.uint8), axis=2, bitorder="little"
    )
    return PackedBits(data=np.ascontiguousarray(data), patch_count=patch_count)


def unpack_bits(packed: PackedBits) -> np.ndarray:
    """Unpack to a bool ``(rows, cols, patch_count)`` cube."""
    _check_packed(packed)
    return np.unpackbits(
        packed.data, axis=2, bitorder="little", count=packed.patch_count
    ).view(np.bool_)


def _check_packed(packed) -> PackedBits:
    if not isinstance(packed, PackedBits):
        raise TypeError(
            f"expected PackedBits (pack through pack_bits), got "
            f"{type(packed).__name__}"
        )
    return packed


def unpack_patch(packed: PackedBits, index: int) -> np.ndarray:
    """One patch plane as bool ``(rows, cols)`` (little bit order)."""
    _check_packed(packed)
    index = int(index)
    if index < 0 or index >= packed.patch_count:
        raise IndexError(
            f"patch index {index} outside [0, {packed.patch_count})"
        )
    byte = packed.data[:, :, index // 8]
    return ((byte >> np.uint8(index % 8)) & np.uint8(1)).view(np.bool_)


# ---------------------------------------------------------------------------
# Unique-writer window updates
# ---------------------------------------------------------------------------


def _as_slices(rows: int, cols: int, rows_slice, cols_slice):
    def _norm(s, n, name):
        if isinstance(s, slice):
            start, stop, step = s.indices(n)
            if step != 1:
                raise ValueError(f"{name} slice step must be 1, got {step}")
        else:
            start, stop = int(s), int(s) + 1
            if start < 0:
                raise IndexError(f"{name} index {start} out of range")
        if start < 0 or stop > n or start >= stop:
            raise IndexError(
                f"{name} window [{start}, {stop}) outside [0, {n}) or empty"
            )
        return start, stop

    return (
        _norm(rows_slice, rows, "row"),
        _norm(cols_slice, cols, "col"),
    )


def set_patch_window(
    packed: PackedBits,
    patch_index: int,
    rows_slice,
    cols_slice,
    values,
) -> None:
    """Set one patch's bits over a cell window, in place.

    Single-writer primitive: the caller must own every chunk the window
    touches (see :func:`verify_chunk_ownership`). Only the target
    patch's bit is modified — every other patch bit in the same bytes is
    preserved bit-for-bit.
    """
    _check_packed(packed)
    patch_index = int(patch_index)
    if patch_index < 0 or patch_index >= packed.patch_count:
        raise IndexError(
            f"patch index {patch_index} outside [0, {packed.patch_count})"
        )
    (r0, r1), (c0, c1) = _as_slices(
        packed.rows, packed.cols, rows_slice, cols_slice
    )
    window = np.asarray(values)
    if window.shape != (r1 - r0, c1 - c0):
        raise ValueError(
            f"values shape {window.shape} does not match window "
            f"({r1 - r0}, {c1 - c0})"
        )
    bits = _as_binary_bits(window, ndim=2)

    byte_index = patch_index // 8
    bit = np.uint8(1 << (patch_index % 8))
    sub = packed.data[r0:r1, c0:c1, byte_index]
    # clear the target bit, then set it where the values say so
    sub &= np.uint8(bit ^ np.uint8(0xFF))
    sub |= bit * bits


# ---------------------------------------------------------------------------
# Chunk grid
# ---------------------------------------------------------------------------


def chunk_grid_index(rows: int, cols: int) -> tuple[int, int]:
    """Number of chunks along each axis (ceil division)."""
    return (
        (int(rows) + CHUNK_CELLS - 1) // CHUNK_CELLS,
        (int(cols) + CHUNK_CELLS - 1) // CHUNK_CELLS,
    )


def chunk_window(rows: int, cols: int, chunk_r: int, chunk_c: int):
    """The cell window ``(rows, cols)`` slices owned by one chunk."""
    r0 = chunk_r * CHUNK_CELLS
    c0 = chunk_c * CHUNK_CELLS
    return (
        (r0, min(r0 + CHUNK_CELLS, int(rows))),
        (c0, min(c0 + CHUNK_CELLS, int(cols))),
    )


def chunk_slices(rows: int, cols: int) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """All chunk windows in row-major chunk order."""
    n_r, n_c = chunk_grid_index(rows, cols)
    for cr in range(n_r):
        for cc in range(n_c):
            yield chunk_window(rows, cols, cr, cc)


def cell_chunk(rows: int, cols: int, r: int, c: int) -> tuple[int, int]:
    """The chunk owning one cell (cells never straddle chunks)."""
    return (int(r) // CHUNK_CELLS, int(c) // CHUNK_CELLS)


def chunks_overlapping_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Chunks containing at least one masked cell, in row-major order."""
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.dtype != np.bool_:
        raise ValueError(f"mask must be a 2-D bool array, got {arr.dtype} {arr.shape}")
    rows, cols = arr.shape
    out: list[tuple[int, int]] = []
    for (r0, r1), (c0, c1) in chunk_slices(rows, cols):
        if bool(arr[r0:r1, c0:c1].any()):
            out.append((r0 // CHUNK_CELLS, c0 // CHUNK_CELLS))
    return out


# ---------------------------------------------------------------------------
# Ownership guards (refuse BEFORE any write)
# ---------------------------------------------------------------------------


def verify_chunk_ownership(claims: Mapping[str, set]) -> None:
    """Refuse overlapping chunk claims across writers.

    ``claims`` maps writer name -> set of ``(chunk_r, chunk_c)`` tiles.
    Raises :class:`ChunkOwnershipError` (a ``RuntimeError``) naming the
    first contested chunk; two writers owning the same chunk would
    read-modify-write the same packed bytes.
    """
    seen: dict[tuple[int, int], str] = {}
    for writer in sorted(claims):
        for chunk in sorted(claims[writer]):
            if chunk in seen:
                raise ChunkOwnershipError(
                    f"chunk {chunk} claimed by both {seen[chunk]!r} and "
                    f"{writer!r} — packed bytes would be shared; refusing "
                    "before any write"
                )
            seen[chunk] = writer


def verify_chunk_partition(
    ranges: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    n_chunk_cols: int,
) -> None:
    """Refuse a tile-range list that is not an EXACT partition.

    Each range is ``((r_lo, r_hi), (c_lo, c_hi))`` covering chunk rows
    ``[r_lo, r_hi)`` and chunk cols ``[c_lo, c_hi)``; the union of all
    covered tiles must be exactly the full grid, every tile exactly
    once (no gap, no duplicate). The number of chunk rows is inferred
    from the ranges; ``n_chunk_cols`` is given.
    """
    n_chunk_cols = int(n_chunk_cols)
    if n_chunk_cols <= 0:
        raise ValueError(f"n_chunk_cols must be positive, got {n_chunk_cols}")
    seen: set[tuple[int, int]] = set()
    n_rows = 0
    for (r_lo, r_hi), (c_lo, c_hi) in ranges:
        r_lo, r_hi, c_lo, c_hi = int(r_lo), int(r_hi), int(c_lo), int(c_hi)
        if not (0 <= r_lo < r_hi and 0 <= c_lo < c_hi <= n_chunk_cols):
            raise ChunkOwnershipError(
                f"invalid chunk range rows [{r_lo}, {r_hi}) cols "
                f"[{c_lo}, {c_hi}) against n_chunk_cols {n_chunk_cols}"
            )
        n_rows = max(n_rows, r_hi)
        for cr in range(r_lo, r_hi):
            for cc in range(c_lo, c_hi):
                tile = (cr, cc)
                if tile in seen:
                    raise ChunkOwnershipError(
                        f"chunk {tile} owned twice in partition {ranges}"
                    )
                seen.add(tile)
    full = {(cr, cc) for cr in range(n_rows) for cc in range(n_chunk_cols)}
    missing = sorted(full - seen)
    if missing:
        raise ChunkOwnershipError(
            f"chunk partition misses {missing[:4]}"
            f"{'...' if len(missing) > 4 else ''} — ranges {list(ranges)}"
        )
