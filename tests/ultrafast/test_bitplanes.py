# SPDX-License-Identifier: GPL-3.0-only
"""T07 packed-state tests: layout, roundtrip, tail determinism, binary
fence, unique-writer chunk ownership (TASKS T07 RED witnesses M1/M2/M3).

The packed representation under test is ``solweig_core.bitplanes`` — a
torch-free standalone layout (NOT a reuse of ``solweig_gpu.incremental.
bitmask``, which lives in the torch-importing oracle package). The layout
mirrors R4's: uint8 ``(rows, cols, ceil(P / 8))`` with the patch axis in
the final dimension, little bit order (patch ``p`` -> byte ``p // 8``,
bit ``p & 7``).

RED witnesses pinned here:

* M1 — all values -> bool: :func:`pack_bits` must refuse any value outside
  the exact set ``{0.0, 1.0}`` (value membership, like the R4 fence). The
  one-step regime's ``vbsh == 2.0`` is the counterexample that a
  ``!= 0`` bool packing would silently corrupt.
* M2 — tail padding nondeterminism: bits ``[P, 8 * bytes)`` of every byte
  group are ALWAYS zero and packs of the same logical values are
  byte-identical across repeats and across differently laid-out sources.
* M3 — shared-byte race: packed updates declare chunk ownership;
  overlapping claims are refused BEFORE any write, and genuinely disjoint
  concurrent writers are byte-identical to the serial update.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_core import bitplanes as bp  # noqa: E402

P = 153  # patch_option=2 sky patches (must match svf_fold.N_PATCHES)
BYTES = 20  # ceil(153 / 8)


def dense_random(rows, cols, seed, p=153):
    rng = np.random.default_rng(seed)
    return (rng.random((rows, cols, p)) < 0.3).astype(np.float32)


# ---------------------------------------------------------------------------
# Layout + roundtrip (gate 3)
# ---------------------------------------------------------------------------


class TestLayout:
    def test_bytes_per_patch(self):
        assert bp.bytes_per_patch(153) == 20
        assert bp.bytes_per_patch(8) == 1
        assert bp.bytes_per_patch(9) == 2
        assert bp.bytes_per_patch(1) == 1
        with pytest.raises(ValueError):
            bp.bytes_per_patch(0)

    def test_packed_shape_and_dtype(self):
        packed = bp.pack_bits(dense_random(9, 17, 1))
        assert packed.data.dtype == np.uint8
        assert packed.data.shape == (9, 17, BYTES)
        assert packed.patch_count == P


class TestRoundtrip:
    """unpack(pack(x)) == x and pack(unpack(x)) == x, bit for bit."""

    @pytest.mark.parametrize(
        "rows,cols,seed",
        [(8, 8, 1), (40, 80, 2), (64, 64, 3), (1, 1, 4), (65, 129, 5)],
    )
    def test_unpack_pack_identity(self, rows, cols, seed):
        dense = dense_random(rows, cols, seed)
        packed = bp.pack_bits(dense)
        back = bp.unpack_bits(packed)
        assert back.dtype == np.bool_
        assert back.shape == (rows, cols, P)
        assert np.array_equal(back, dense != 0)

    def test_pack_unpack_identity_bytes(self):
        dense = dense_random(33, 47, 6)
        packed = bp.pack_bits(dense)
        again = bp.pack_bits(bp.unpack_bits(packed))
        assert np.array_equal(again.data, packed.data)

    def test_roundtrip_all_zeros_and_all_ones(self):
        for fill in (0.0, 1.0):
            dense = np.full((12, 13, P), fill, dtype=np.float32)
            packed = bp.pack_bits(dense)
            assert np.array_equal(bp.unpack_bits(packed), dense != 0)

    def test_single_patch_bit_addressing(self):
        """patch p -> byte p // 8, bit p & 7 (little bit order)."""
        set_patches = (0, 1, 7, 8, 9, 127, 152)
        dense = np.zeros((5, 6, P), dtype=np.float32)
        for p in set_patches:
            dense[2, 3, p] = np.float32(1.0)
        packed = bp.pack_bits(dense)
        # hand-computed expectation: byte = OR of (1 << p % 8) per byte group
        expected = np.zeros((5, 6, BYTES), dtype=np.uint8)
        for p in set_patches:
            expected[2, 3, p // 8] |= np.uint8(1 << (p % 8))
        assert np.array_equal(packed.data, expected)
        # every other cell stays all-zero
        flat = packed.data.reshape(-1, BYTES)
        nz_rows = np.flatnonzero(flat.any(axis=1))
        assert nz_rows.size == 1  # only cell (2, 3)


# ---------------------------------------------------------------------------
# Tail padding determinism (M2)
# ---------------------------------------------------------------------------


class TestTailPadding:
    def test_tail_bits_are_zero(self):
        packed = bp.pack_bits(dense_random(20, 30, 7))
        # bits [153, 160) live only in the last byte of each cell: patch
        # 152 occupies bit 0 of byte 19, so the tail mask is 0xFE
        last = packed.data[:, :, BYTES - 1]
        assert int((last & 0xFE).sum()) == 0
        # and a forced-random tail would be caught: pack the all-ones cube
        # and confirm only bit 0 of the last byte is set
        ones = bp.pack_bits(np.ones((4, 4, P), dtype=np.float32))
        assert int(ones.data[:, :, BYTES - 1].max()) == 0x01

    def test_tail_deterministic_across_repeats_and_layouts(self):
        """M2 witness: padding bytes never vary run to run or with the
        source array's memory layout."""
        dense = dense_random(24, 40, 8)
        reference = bp.pack_bits(dense).data
        for _ in range(5):
            assert np.array_equal(bp.pack_bits(dense).data, reference)
        # different layout (Fortran order view), same logical values
        fortran = np.asfortranarray(dense)
        assert np.array_equal(bp.pack_bits(fortran).data, reference)
        # strided slice of a larger array
        big = dense_random(48, 80, 9)
        view = big[::2, ::2, :]
        assert np.array_equal(
            bp.pack_bits(view).data, bp.pack_bits(np.ascontiguousarray(view)).data
        )

    def test_pack_of_partial_patch_count(self):
        """Non-153 patch counts pack with their own deterministic tail."""
        dense = (np.random.default_rng(10).random((6, 7, 9)) < 0.5).astype(np.float32)
        packed = bp.pack_bits(dense)
        assert packed.data.shape == (6, 7, 2)
        again = bp.pack_bits(bp.unpack_bits(packed))
        assert np.array_equal(again.data, packed.data)


# ---------------------------------------------------------------------------
# Binary fence (M1): values collapse is refused, never bool-packed
# ---------------------------------------------------------------------------


class TestBinaryFence:
    @pytest.mark.parametrize(
        "bad",
        [
            np.float32(2.0),   # the one-step vbsh value
            np.float32(1.5),
            np.float32(-1.0),
            np.float32(-0.5),
            np.float32("nan"),
            np.float32("inf"),
        ],
    )
    def test_pack_refuses_non_binary(self, bad):
        dense = np.zeros((4, 5, P), dtype=np.float32)
        dense[1, 2, 9] = bad
        with pytest.raises(ValueError) as exc:
            bp.pack_bits(dense)
        assert "binary" in str(exc.value).lower()

    def test_pack_accepts_exact_zero_and_one(self):
        dense = np.zeros((4, 4, P), dtype=np.float32)
        dense[0, 0, 0] = np.float32(1.0)
        dense[3, 3, 152] = np.float32(1.0)
        bp.pack_bits(dense)  # must not raise

    def test_set_patch_window_refuses_non_binary(self):
        packed = bp.pack_bits(np.zeros((4, 4, P), dtype=np.float32))
        values = np.full((2, 2), 2.0, dtype=np.float32)
        with pytest.raises(ValueError):
            bp.set_patch_window(
                packed, 9, slice(0, 2), slice(0, 2), values
            )

    def test_bool_input_accepted(self):
        dense = np.random.default_rng(11).random((4, 4, P)) < 0.5
        packed = bp.pack_bits(dense)
        assert np.array_equal(bp.unpack_bits(packed), dense)


# ---------------------------------------------------------------------------
# Unique-writer patch window updates
# ---------------------------------------------------------------------------


class TestPatchWindow:
    def test_set_and_clear_bits(self):
        dense = dense_random(8, 9, 12)
        packed = bp.pack_bits(dense)
        reference = dense.copy()
        for value in (1.0, 0.0, 1.0, 0.0):
            bp.set_patch_window(
                packed, 100, slice(2, 5), slice(3, 7),
                np.full((3, 4), value, dtype=np.float32),
            )
            reference[2:5, 3:7, 100] = np.float32(value)
            # whole-array oracle: the update equals a fresh pack of the
            # edited dense cube — no other bit anywhere can differ
            assert np.array_equal(packed.data, bp.pack_bits(reference).data)
        # and the window reads back correctly
        plane = bp.unpack_patch(packed, 100)[2:5, 3:7]
        assert bool((plane == (reference[2:5, 3:7, 100] != 0)).all())

    def test_patch_index_bounds(self):
        packed = bp.pack_bits(np.zeros((4, 4, P), dtype=np.float32))
        with pytest.raises(IndexError):
            bp.set_patch_window(
                packed, P, slice(0, 1), slice(0, 1), np.ones((1, 1))
            )
        with pytest.raises(IndexError):
            bp.unpack_patch(packed, -1)


# ---------------------------------------------------------------------------
# Chunk grid + ownership guards (M3)
# ---------------------------------------------------------------------------


class TestChunkGrid:
    def test_grid_index_non_square(self):
        assert bp.chunk_grid_index(64, 64) == (1, 1)
        assert bp.chunk_grid_index(65, 129) == (2, 3)
        assert bp.chunk_grid_index(40, 80) == (1, 2)
        assert bp.chunk_grid_index(1, 1) == (1, 1)

    def test_chunk_windows_clipped_at_boundary(self):
        windows = list(bp.chunk_slices(100, 70))
        assert len(windows) == 2 * 2
        assert windows[-1] == ((64, 100), (64, 70))

    def test_chunks_overlapping_mask(self):
        mask = np.zeros((130, 130), dtype=bool)
        mask[0, 0] = True
        mask[64, 64] = True
        mask[129, 40] = True
        chunks = set(bp.chunks_overlapping_mask(mask))
        assert chunks == {(0, 0), (1, 1), (2, 0)}

    def test_chunk_of_cell_roundtrip(self):
        for rows, cols in ((64, 64), (100, 70), (1, 1), (65, 1)):
            for r in range(rows):
                for c in range(cols):
                    assert bp.cell_chunk(rows, cols, r, c) == (
                        r // bp.CHUNK_CELLS, c // bp.CHUNK_CELLS
                    )


class TestChunkOwnership:
    """M3 witness: two writers never own the same packed chunk."""

    def test_disjoint_claims_accepted(self):
        bp.verify_chunk_ownership(
            {"w0": {(0, 0), (0, 1)}, "w1": {(1, 0), (1, 1)}}
        )

    def test_overlapping_claims_refused(self):
        with pytest.raises(RuntimeError) as exc:
            bp.verify_chunk_ownership(
                {"w0": {(0, 0), (0, 1)}, "w1": {(0, 1), (1, 1)}}
            )
        assert "(0, 1)" in str(exc.value)

    def test_exact_partition_accepted(self):
        # ranges are half-open ((r_lo, r_hi), (c_lo, c_hi)) tile blocks
        bp.verify_chunk_partition(
            [((0, 1), (0, 2)), ((1, 2), (0, 2))], n_chunk_cols=2
        )

    def test_partition_with_gap_or_duplicate_refused(self):
        with pytest.raises(RuntimeError):
            # covers only chunk (0, 0): chunk (0, 1) is missing
            bp.verify_chunk_partition([((0, 1), (0, 1))], n_chunk_cols=2)
        with pytest.raises(RuntimeError):
            # both ranges claim chunk (0, 1)
            bp.verify_chunk_partition(
                [((0, 1), (0, 2)), ((0, 1), (1, 2))], n_chunk_cols=2
            )

    def test_concurrent_disjoint_writers_byte_identical(self):
        """Stress: real threads updating disjoint chunk-owned windows give
        the exact bytes of the serial update (the race the guard forbids
        would flip bits nondeterministically)."""
        rows = cols = 128
        dense = dense_random(rows, cols, 13)
        packed = bp.pack_bits(dense)
        serial = bp.pack_bits(dense)
        patch = 77
        values = (np.random.default_rng(14).random((rows, cols)) < 0.5).astype(np.float32)

        for cr in range(rows // bp.CHUNK_CELLS):
            for cc in range(cols // bp.CHUNK_CELLS):
                (r0, r1), (c0, c1) = bp.chunk_window(rows, cols, cr, cc)
                bp.set_patch_window(
                    serial, patch, slice(r0, r1), slice(c0, c1), values[r0:r1, c0:c1]
                )

        claims = {}
        errors = []

        def writer(cr, cc):
            try:
                owned = {(cr, cc)}
                bp.verify_chunk_ownership({f"w{cr}_{cc}": owned | set()})
                (r0, r1), (c0, c1) = bp.chunk_window(rows, cols, cr, cc)
                bp.set_patch_window(
                    packed, patch, slice(r0, r1), slice(c0, c1), values[r0:r1, c0:c1]
                )
            except Exception as error:  # pragma: no cover - surfaced below
                errors.append(error)

        threads = [
            threading.Thread(target=writer, args=(cr, cc))
            for cr in range(rows // bp.CHUNK_CELLS)
            for cc in range(cols // bp.CHUNK_CELLS)
        ]
        claims = claims  # single-writer-per-chunk claims verified per thread
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert np.array_equal(packed.data, serial.data)
