# SPDX-License-Identifier: GPL-3.0-only
"""T06 gates: sparse Numba march runner + safe CPU parallelism
(DESIGN.ko.md 3.5/7.5/7.6/7.7, TASKS T06).

Gate structure (TASKS.ko.md T06 + lead contract):

* **RED first** — this file was written and run BEFORE
  ``solweig_core/numba_cpu/sparse_march.py`` existed; the recorded first
  failure is the collection ``ModuleNotFoundError``
  (artifacts/t06/red_first_run.txt).
* **Same state machine** — every sparse target executes the T04 dense
  kernel's per-target primitive graph VERBATIM (the runner replicates the
  dense loop body because march.py is read-only; the body is copied, never
  re-derived): OOB exact zero, first-step reset, target-local trunk gate,
  suppression order, final threshold graph, float32 discipline.
* **Equality gate** — dense serial vs sparse serial vs sparse parallel at
  1/2/4/8 threads compare through raw uint32 bit views (signed zero and NaN
  payload included). Outside the work list the sparse planes must be
  bit-exactly the base/init value (no stray writes).
* **Routing** — work below :data:`SERIAL_MIN_PAIRS` routes serial with NO
  thread spawn (a measured decision; the crossover is benchmarked in
  artifacts/t06/bench_ablation.txt); T05's ``ROUTE_DENSE_PAIR_FRACTION``
  per-patch dense routing is honoured by the build-level runner.
* **Parallel ownership** — row-run chunk ownership is an EXACT partition
  (each run index in exactly one chunk, verified at plan time and re-checked
  before threads spawn); threads are plain ``threading.Thread``s calling one
  ``njit(nogil=True, fastmath=False)`` kernel, so each thread's arithmetic
  IS serial arithmetic — no numba ``parallel=True``/``prange``, no
  reduction, no reassociation.
* **Cancellation seam** — a generation counter is re-read before EVERY
  row-run; a cancelled (stale) generation writes nothing. After
  ``SparseMarchLaunch.cancel()`` returns (it joins), no further output
  write can occur, so the buffers are safe to recycle.
* **Domain guard** — target addressing is int32 row-run coordinates; the
  runner refuses shapes at the int32 boundary (an explicit guard: such
  domains are unallocatable on this host, the guard is the contract).
* **RED witnesses** (card list, each killed by a designated test via the
  recorded mutations M1-M5 in artifacts/t06/mutations/): duplicate
  pointer/target ownership across chunks, thread-overhead regression on
  small work, target id overflow, recycled buffer write after cancellation,
  physical crop changing logical stop.
* **Purity** — no torch import, no fastmath/parallel/prange in the runner;
  imports and runs torch-free in a subprocess.

Site gates are skipped when the site cache is absent. The T05 closure
builder only proves ``svf_shadow`` closures; the ``wallheight_23`` sparse
runner is gated on synthetic work lists (documented limitation).
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from trace_exporter import capture_trace  # noqa: E402
from solweig_core import sparse_work as sw  # noqa: E402
from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu.march import (  # noqa: E402
    march_svf_shadow,
    march_wallheight23,
)
from solweig_core.numba_cpu.sparse_march import (  # noqa: E402
    INT32_MAX,
    SERIAL_MIN_PAIRS,
    SparseMarchLaunch,
    check_work_domain,
    march_build,
    march_svf_shadow_sparse,
    march_wallheight23_sparse,
    plan_chunks,
    plan_sparse_march,
)
from solweig_core.numba_cpu.sparse_march import (  # noqa: E402
    _svf_shadow_runs_kernel as svf_runs_kernel,
)

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
ARTIFACT_DIR = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t06"
)
F32 = np.dtype("<f4")
U32 = np.dtype("<u4")
ZERO_BITS = np.array(np.float32(0.0).view(np.uint32), dtype=np.uint32)

#: Small march grid (closure-style fixtures; non-square on purpose).
SYN_ROWS, SYN_COLS = 40, 80
#: Larger grid that actually crosses SERIAL_MIN_PAIRS so the parallel path
#: is genuinely exercised (73,728 cells; a ~55% mask ~ 40k pairs).
BIG_ROWS, BIG_COLS = 192, 384

SVF_ANGLES = [
    (37.0, 35.0),
    (225.0, 35.0),
    (90.0, 6.0),
    (0.0, 78.0),
    (45.0, 90.0),
]
WH_ANGLES = [
    (37.0, 35.0),
    (225.0, 42.0),
    (200.0, 6.0),
    (0.0, 78.0),
    (270.0, 90.0),
]
THREAD_COUNTS = [1, 2, 4, 8]


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=F32).view(U32)


def synth_planes(rows, cols, seed=7, negative_dem=False):
    """Deterministic (a, dem, canopy) float32 numpy planes (T05 family)."""
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    if negative_dem:
        dem -= np.float32(12.0)
    a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    a[3 : min(9, rows), 4 : min(12, cols)] += np.float32(18.0)
    canopy = rng.uniform(0.0, 6.0, (rows, cols)).astype(np.float32)
    canopy[canopy < np.float32(3.0)] = np.float32(0.0)
    return a, dem, canopy


def synth_inputs(a, dem, canopy):
    return sw.compose_march_inputs(a, dem, canopy)


def synth_table(kernel, az, alt, amp, rows, cols, scale=0.5):
    trace = capture_trace(
        kernel,
        float(az),
        float(alt),
        scale,
        int(rows),
        int(cols),
        float(amp),
        amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
        verify_probes=False,
    )
    return st.build_table_from_trace(trace)


def random_mask(rows, cols, density, seed):
    return np.random.default_rng(seed).random((rows, cols)) < density


def corridor_style_mask(table, rows, cols, seed):
    """A realistic closure-shaped work list: corridor of a changed disc."""
    rng = np.random.default_rng(seed)
    C = np.zeros((rows, cols), dtype=bool)
    r0, c0 = rows // 2, cols // 2
    rr, cc = np.ogrid[:rows, :cols]
    C[(rr - r0) ** 2 + (cc - c0) ** 2 <= 36] = True
    C |= rng.random((rows, cols)) < 0.01
    offsets = sorted({(int(x), int(y)) for x, y in zip(table.dx, table.dy)})
    return sw.closure_mask(C, offsets, offsets)


def assert_sparse_matches_dense(
    dense_out, sparse_out, targets, label, *, names=("sh", "vegsh", "vbsh")
):
    """Raw-bit equality at the targets AND bit-exact base outside them.

    The sparse contract: every target cell's three outputs are bit-equal to
    the dense kernel's, and no other cell is ever written.
    """
    assert targets.any(), f"{label}: fixture has an empty work list"
    for name, d, s in zip(names, dense_out, sparse_out):
        db, sb = bits(d), bits(s)
        got_t = sb[targets]
        want_t = db[targets]
        bad = np.nonzero(got_t != want_t)[0]
        assert bad.size == 0, (
            f"{label} plane {name}: {bad.size} target cells differ in raw "
            f"bits, first at flat {bad[0]} "
            f"(got 0x{got_t[bad[0]]:08x}, want 0x{want_t[bad[0]]:08x})"
        )
        outside = sb[~targets]
        assert np.array_equal(outside, np.zeros_like(outside)), (
            f"{label} plane {name}: sparse runner wrote outside the work list"
        )


# ---------------------------------------------------------------------------
# Routing plan: serial threshold, exact chunk partition (M1/M2 witnesses)
# ---------------------------------------------------------------------------


class TestPlanRouting:
    def test_small_work_routes_serial_no_threads(self):
        """Below SERIAL_MIN_PAIRS the plan must be a single serial chunk —
        no thread spawn for tiny edits (thread-overhead witness M2)."""
        runs = np.array([[0, 0, 63], [1, 10, 73], [5, 0, 0]], dtype=np.int32)
        plan = plan_sparse_march(runs, n_threads=8)
        assert plan.pair_count == 64 + 64 + 1 < SERIAL_MIN_PAIRS
        assert plan.used_threads == 1
        assert plan.chunks == ((0, 3),)
        assert plan.route == "sparse"

    def test_n_threads_one_is_always_serial(self):
        runs = np.array([[0, 0, 999_999]], dtype=np.int32)
        plan = plan_sparse_march(runs, n_threads=1)
        assert plan.pair_count >= SERIAL_MIN_PAIRS
        assert plan.used_threads == 1
        assert plan.chunks == ((0, 1),)

    def test_large_work_routes_parallel(self):
        runs = np.array([[0, 0, 999_999], [1, 0, 999_999]], dtype=np.int32)
        plan = plan_sparse_march(runs, n_threads=4)
        assert plan.pair_count >= SERIAL_MIN_PAIRS
        # only 2 runs: worker count is capped by the run count
        assert plan.used_threads == 2
        assert len(plan.chunks) == 2

    def test_empty_runs_plan(self):
        runs = np.zeros((0, 3), dtype=np.int32)
        plan = plan_sparse_march(runs, n_threads=8)
        assert plan.pair_count == 0
        assert plan.used_threads == 1
        assert plan.chunks == ()

    def test_chunk_partition_is_exact(self):
        """M1 witness: every run index is owned by EXACTLY one chunk —
        ranges are sorted, contiguous, non-empty and cover [0, n_runs).
        Feeds both uniform and lumpy (one huge run) length profiles."""
        profiles = []
        for n_runs, threads in (
            (1, 1), (1, 8), (2, 8), (7, 4), (16, 4), (17, 4), (100, 8),
            (8193, 8), (1000, 3),
        ):
            profiles.append((np.ones(n_runs, dtype=np.int64), threads))
        lumpy = np.ones(200, dtype=np.int64)
        lumpy[7] = 10_000  # one run dwarfs the rest — boundaries skip
        profiles.append((lumpy, 8))
        for lengths, threads in profiles:
            n_runs = int(lengths.shape[0])
            chunks = plan_chunks(lengths, threads)
            assert chunks, (n_runs, threads)
            expected = 0
            seen: list[int] = []
            for lo, hi in chunks:
                assert 0 <= lo < hi <= n_runs, (n_runs, threads, chunks)
                assert lo == expected, (n_runs, threads, chunks)
                expected = hi
                seen.extend(range(lo, hi))
            assert expected == n_runs, (n_runs, threads, chunks)
            assert seen == list(range(n_runs)), (
                "duplicate or missing run ownership"
            )
            assert len(chunks) <= threads

    def test_chunk_partition_deterministic(self):
        for n_runs, threads in ((97, 4), (33, 8)):
            lengths = np.ones(n_runs, dtype=np.int64)
            assert plan_chunks(lengths, threads) == plan_chunks(
                lengths, threads
            )


# ---------------------------------------------------------------------------
# Serial sparse equality: svf_shadow (synthetic)
# ---------------------------------------------------------------------------


class TestSerialSparseSvf:
    def _scene(self, rows=SYN_ROWS, cols=SYN_COLS, seed=7, negative_dem=False):
        a, dem, canopy = synth_planes(rows, cols, seed, negative_dem)
        inputs = synth_inputs(a, dem, canopy)
        return inputs

    def test_serial_sparse_matches_dense_angle_sweep(self):
        inputs = self._scene()
        for az, alt in SVF_ANGLES:
            table = synth_table(
                st.KERNEL_SVF_SHADOW, az, alt, 20.0, SYN_ROWS, SYN_COLS
            )
            targets = corridor_style_mask(table, SYN_ROWS, SYN_COLS, seed=3)
            runs = sw.row_runs_from_mask(targets)
            dense = march_svf_shadow(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
            )
            sparse = march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                runs,
            )
            assert sparse.plan.used_threads == 1
            assert_sparse_matches_dense(
                dense, sparse, targets, f"svf({az},{alt})"
            )

    def test_serial_sparse_random_masks_and_negative_dem(self):
        inputs = self._scene(seed=11, negative_dem=True)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 133.0, 42.0, 25.0, SYN_ROWS, SYN_COLS
        )
        for density, seed in ((0.05, 1), (0.3, 2), (0.8, 3)):
            targets = random_mask(SYN_ROWS, SYN_COLS, density, seed)
            runs = sw.row_runs_from_mask(targets)
            dense = march_svf_shadow(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
            )
            sparse = march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                runs,
            )
            assert_sparse_matches_dense(
                dense, sparse, targets, f"density={density}"
            )

    def test_full_tile_work_list_equals_dense_everywhere(self):
        """The strongest serial check: a full-tile work list reproduces the
        dense kernel bit-for-bit on EVERY cell (both variants)."""
        inputs = self._scene(seed=13)
        full = np.ones((SYN_ROWS, SYN_COLS), dtype=bool)
        runs = sw.row_runs_from_mask(full)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 61.0, 30.0, 15.0, SYN_ROWS, SYN_COLS
        )
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        sparse = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        for name, d, s in zip(("sh", "vegsh", "vbsh"), dense, sparse):
            assert np.array_equal(bits(d), bits(s)), name

    def test_one_step_witness_inside_work_list(self):
        """One-step regime cells (vbsh == 2.0) inside a sparse work list
        reproduce bit-exactly — the runner has no packed-state fence."""
        inputs = self._scene(seed=13)
        canopy = inputs.canopy.copy()
        canopy[10:20, 30:60] = np.float32(5.5)
        inputs = synth_inputs(inputs.a, inputs.dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 0.0, 78.0, 4.0, SYN_ROWS, SYN_COLS,
            scale=0.25,
        )
        assert table.count == 1, "fixture is not one-step"
        targets = random_mask(SYN_ROWS, SYN_COLS, 0.5, 7)
        runs = sw.row_runs_from_mask(targets)
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        sparse = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        assert_sparse_matches_dense(dense, sparse, targets, "one-step")
        bits2 = np.uint32(0x40000000)
        in_mask_2 = (bits(dense[2])[targets] == bits2).any()
        assert in_mask_2, "fixture has no vbsh == 2.0 cell in the work list"

    def test_zero_step_table(self):
        """count == 0: targets resolve from init state only."""
        inputs = self._scene(seed=5)
        table = st.StepTable(
            key=st.StepTableKey(
                semantics_profile="canonical_cpu_v1",
                kernel_variant=st.KERNEL_SVF_SHADOW,
                source_geometry_hash="0" * 64,
                angle_input_bits=(st.f32_bits_hex(37.0), st.f32_bits_hex(35.0)),
                scale_bits=st.f32_bits_hex(0.5),
                logical_rows=SYN_ROWS,
                logical_cols=SYN_COLS,
                amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
                executed_amplitude_bits=st.f32_bits_hex(-1.0),
                boundary_policy_id=st.BOUNDARY_SHIFT_WINDOW_V1,
            ),
            count=0,
            dx=np.zeros(0, dtype=np.int32),
            dy=np.zeros(0, dtype=np.int32),
            dz_bits=np.zeros(0, dtype=np.uint32),
            stop_reason=st.STOP_AMPLITUDE,
            branch_id=np.zeros(0, dtype=np.uint8),
            previous_dz_bits=np.zeros(0, dtype=np.uint32),
        )
        targets = random_mask(SYN_ROWS, SYN_COLS, 0.3, 9)
        runs = sw.row_runs_from_mask(targets)
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        sparse = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        assert_sparse_matches_dense(dense, sparse, targets, "zero-step")

    def test_extreme_aspect(self):
        rows, cols = 5, 1000
        a, dem, canopy = synth_planes(rows, cols, seed=17)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 100.0, 35.0, 1e6, rows, cols
        )
        targets = random_mask(rows, cols, 0.4, 11)
        runs = sw.row_runs_from_mask(targets)
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        sparse = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        assert_sparse_matches_dense(dense, sparse, targets, "5x1000")

    def test_empty_runs_returns_zeros(self):
        inputs = self._scene()
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, SYN_ROWS, SYN_COLS
        )
        runs = np.zeros((0, 3), dtype=np.int32)
        result = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        for plane in result[:3]:
            assert plane.shape == (SYN_ROWS, SYN_COLS)
            assert np.array_equal(bits(plane), np.zeros_like(bits(plane)))

    def test_base_planes_preserved_outside_targets(self):
        """A provided base is copied and only the targets are overwritten."""
        inputs = self._scene(seed=19)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, SYN_ROWS, SYN_COLS
        )
        targets = random_mask(SYN_ROWS, SYN_COLS, 0.3, 13)
        runs = sw.row_runs_from_mask(targets)
        rng = np.random.default_rng(23)
        base = tuple(
            rng.uniform(-5.0, 5.0, (SYN_ROWS, SYN_COLS)).astype(np.float32)
            for _ in range(3)
        )
        result = march_svf_shadow_sparse(
            table,
            inputs.a,
            inputs.vegdsm,
            inputs.vegdsm2,
            inputs.bush,
            runs,
            base=base,
        )
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        for b, s, d in zip(base, result, dense):
            assert np.array_equal(bits(s)[~targets], bits(b)[~targets])
            assert np.array_equal(bits(s)[targets], bits(d)[targets])

    def test_strided_input_bit_equal(self):
        a, dem, canopy = synth_planes(80, 160, seed=19)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, SYN_ROWS, SYN_COLS
        )
        targets = random_mask(SYN_ROWS, SYN_COLS, 0.3, 17)
        runs = sw.row_runs_from_mask(targets)
        strided = tuple(
            getattr(inputs, name)[:SYN_ROWS, :SYN_COLS]
            for name in ("a", "vegdsm", "vegdsm2", "bush")
        )
        assert not strided[0].flags["C_CONTIGUOUS"]
        got_strided = march_svf_shadow_sparse(table, *strided, runs)
        got_dense = march_svf_shadow_sparse(
            table,
            np.ascontiguousarray(strided[0]),
            np.ascontiguousarray(strided[1]),
            np.ascontiguousarray(strided[2]),
            np.ascontiguousarray(strided[3]),
            runs,
        )
        for s, d in zip(got_strided[:3], got_dense[:3]):
            assert np.array_equal(bits(s), bits(d))


# ---------------------------------------------------------------------------
# Serial sparse equality: wallheight_23 (synthetic work lists)
# ---------------------------------------------------------------------------


class TestSerialSparseWallheight:
    def test_serial_sparse_matches_dense_angle_sweep(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        for az, alt in WH_ANGLES:
            table = synth_table(
                st.KERNEL_WALLHEIGHT_23, az, alt, 20.0, SYN_ROWS, SYN_COLS
            )
            targets = corridor_style_mask(table, SYN_ROWS, SYN_COLS, seed=5)
            runs = sw.row_runs_from_mask(targets)
            dense = march_wallheight23(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
            )
            sparse = march_wallheight23_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                runs,
            )
            assert_sparse_matches_dense(
                dense, sparse, targets, f"wh({az},{alt})"
            )

    def test_full_tile_work_list_equals_dense_everywhere(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS, seed=13)
        inputs = synth_inputs(a, dem, canopy)
        full = np.ones((SYN_ROWS, SYN_COLS), dtype=bool)
        runs = sw.row_runs_from_mask(full)
        table = synth_table(
            st.KERNEL_WALLHEIGHT_23, 200.0, 30.0, 15.0, SYN_ROWS, SYN_COLS
        )
        dense = march_wallheight23(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        sparse = march_wallheight23_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        for name, d, s in zip(("vegsh", "sh", "vbsh"), dense, sparse):
            assert np.array_equal(bits(d), bits(s)), name


# ---------------------------------------------------------------------------
# Parallel equality: 1/2/4/8 threads vs serial, both variants
# ---------------------------------------------------------------------------


class TestParallelEquality:
    @pytest.fixture(scope="class")
    def big_scene(self):
        a, dem, canopy = synth_planes(BIG_ROWS, BIG_COLS, seed=29)
        return synth_inputs(a, dem, canopy)

    @pytest.fixture(scope="class")
    def big_targets(self):
        mask = random_mask(BIG_ROWS, BIG_COLS, 0.55, 31)
        assert mask.sum() > SERIAL_MIN_PAIRS, "fixture must cross the threshold"
        return mask

    def test_parallel_matches_serial_svf(self, big_scene, big_targets):
        """THE exit gate (synthetic leg): dense serial vs sparse serial vs
        sparse parallel at 1/2/4/8 threads, raw uint32 bits."""
        inputs = big_scene
        runs = sw.row_runs_from_mask(big_targets)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 61.0, 12.0, 20.0, BIG_ROWS, BIG_COLS
        )
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        serial = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        assert serial.plan.used_threads == 1
        assert_sparse_matches_dense(dense, serial, big_targets, "par-svf")
        n_runs = runs.shape[0]
        for threads in THREAD_COUNTS:
            par = march_svf_shadow_sparse(
                table,
                inputs.a,
                inputs.vegdsm,
                inputs.vegdsm2,
                inputs.bush,
                runs,
                n_threads=threads,
            )
            assert par.plan.used_threads == min(threads, n_runs)
            for name, s, p in zip(("sh", "vegsh", "vbsh"), serial, par):
                assert np.array_equal(bits(s), bits(p)), (
                    f"svf {threads} threads: plane {name} diverged"
                )

    def test_parallel_matches_serial_wallheight(self, big_scene, big_targets):
        inputs = big_scene
        runs = sw.row_runs_from_mask(big_targets)
        table = synth_table(
            st.KERNEL_WALLHEIGHT_23, 200.0, 12.0, 20.0, BIG_ROWS, BIG_COLS
        )
        dense = march_wallheight23(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        serial = march_wallheight23_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        assert_sparse_matches_dense(dense, serial, big_targets, "par-wh")
        for threads in THREAD_COUNTS:
            par = march_wallheight23_sparse(
                table,
                inputs.a,
                inputs.vegdsm,
                inputs.vegdsm2,
                inputs.bush,
                runs,
                n_threads=threads,
            )
            for name, s, p in zip(("vegsh", "sh", "vbsh"), serial, par):
                assert np.array_equal(bits(s), bits(p)), (
                    f"wh {threads} threads: plane {name} diverged"
                )

    def test_parallel_no_stray_writes(self, big_scene, big_targets):
        inputs = big_scene
        runs = sw.row_runs_from_mask(big_targets)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 61.0, 12.0, 20.0, BIG_ROWS, BIG_COLS
        )
        par = march_svf_shadow_sparse(
            table,
            inputs.a,
            inputs.vegdsm,
            inputs.vegdsm2,
            inputs.bush,
            runs,
            n_threads=4,
        )
        for name, p in zip(("sh", "vegsh", "vbsh"), par):
            outside = bits(p)[~big_targets]
            assert np.array_equal(outside, np.zeros_like(outside)), name

    def test_parallel_deterministic_repeat(self, big_scene, big_targets):
        inputs = big_scene
        runs = sw.row_runs_from_mask(big_targets)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 61.0, 12.0, 20.0, BIG_ROWS, BIG_COLS
        )
        args = (table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush)
        r1 = march_svf_shadow_sparse(*args, runs, n_threads=4)
        r2 = march_svf_shadow_sparse(*args, runs, n_threads=4)
        for p1, p2 in zip(r1[:3], r2[:3]):
            assert np.array_equal(bits(p1), bits(p2))

    def test_small_work_executes_serial_in_runner(self):
        """Routing integration: a tiny work list through the full runner
        must execute with used_threads == 1 (no thread spawn)."""
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, SYN_ROWS, SYN_COLS
        )
        targets = np.zeros((SYN_ROWS, SYN_COLS), dtype=bool)
        targets[10, 10:14] = True
        runs = sw.row_runs_from_mask(targets)
        result = march_svf_shadow_sparse(
            table,
            inputs.a,
            inputs.vegdsm,
            inputs.vegdsm2,
            inputs.bush,
            runs,
            n_threads=8,
        )
        assert result.plan.pair_count < SERIAL_MIN_PAIRS
        assert result.plan.used_threads == 1


# ---------------------------------------------------------------------------
# T05 build integration: closure -> runner, per-patch dense routing
# ---------------------------------------------------------------------------


class TestBuildIntegration:
    def _edit_fixture(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS, seed=7)
        rr, cc = np.ogrid[:SYN_ROWS, :SYN_COLS]
        canopy_edit = canopy.copy()
        canopy_edit[(rr - 20) ** 2 + (cc - 50) ** 2 <= 36] = np.float32(12.0)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, canopy_edit)
        amp_pre = float(sw.effective_amplitude(pre))
        amp_post = float(sw.effective_amplitude(post))
        angles = [(37.0, 35.0), (90.0, 6.0), (0.0, 78.0)]
        tables_pre = {
            i: synth_table(
                st.KERNEL_SVF_SHADOW, az, alt, amp_pre, SYN_ROWS, SYN_COLS
            )
            for i, (az, alt) in enumerate(angles)
        }
        tables_post = {
            i: synth_table(
                st.KERNEL_SVF_SHADOW, az, alt, amp_post, SYN_ROWS, SYN_COLS
            )
            for i, (az, alt) in enumerate(angles)
        }
        return pre, post, tables_pre, tables_post

    def test_closure_work_list_matches_dense_post(self):
        pre, post, tables_pre, tables_post = self._edit_fixture()
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        masks = build.masks()
        assert build.pair_count > 0
        for threads in (1, 4):
            for idx in sorted(build.row_runs):
                dense_post = march_svf_shadow(
                    tables_post[idx],
                    post.a,
                    post.vegdsm,
                    post.vegdsm2,
                    post.bush,
                )
                sparse = march_svf_shadow_sparse(
                    tables_post[idx],
                    post.a,
                    post.vegdsm,
                    post.vegdsm2,
                    post.bush,
                    build.row_runs[idx],
                    n_threads=threads,
                )
                assert_sparse_matches_dense(
                    dense_post, sparse, masks[idx], f"build patch {idx}"
                )

    def test_march_build_routes_and_splices(self):
        """The build-level runner: sparse patches match the dense post
        march at the closure; a full-tile (route-dense) patch produces the
        complete dense plane and is flagged route='dense'."""
        pre, post, tables_pre, tables_post = self._edit_fixture()
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        masks = build.masks()
        results = march_build(
            build, tables_post, post.a, post.vegdsm, post.vegdsm2, post.bush,
            n_threads=4,
        )
        assert set(results) == set(build.row_runs)
        for idx in sorted(results):
            dense_post = march_svf_shadow(
                tables_post[idx],
                post.a,
                post.vegdsm,
                post.vegdsm2,
                post.bush,
            )
            result = results[idx]
            if idx in build.route_dense_patches:
                assert result.plan.route == "dense"
                for d, s in zip(dense_post, result):
                    assert np.array_equal(bits(d), bits(s))
            else:
                assert result.plan.route == "sparse"
                assert_sparse_matches_dense(
                    dense_post, result, masks[idx], f"march_build {idx}"
                )

    def test_march_build_full_tile_closure_routes_dense(self):
        """Empty C + changed tables -> honest full-tile work list, which the
        build runner must route through the dense kernel (T05 ceiling)."""
        a, dem, canopy = synth_planes(24, 32, seed=13)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, canopy.copy())
        tables_pre = {
            0: synth_table(
                st.KERNEL_SVF_SHADOW, 37.0, 35.0, 10.0, 24, 32
            )
        }
        tables_post = {
            0: synth_table(
                st.KERNEL_SVF_SHADOW, 37.0, 35.0, 25.0, 24, 32
            )
        }
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        assert build.table_changed_with_empty_C
        assert build.route_dense
        results = march_build(
            build, tables_post, post.a, post.vegdsm, post.vegdsm2, post.bush
        )
        assert results[0].plan.route == "dense"
        dense = march_svf_shadow(
            tables_post[0], post.a, post.vegdsm, post.vegdsm2, post.bush
        )
        for d, s in zip(dense, results[0]):
            assert np.array_equal(bits(d), bits(s))


# ---------------------------------------------------------------------------
# Cancellation seam (M4 witness)
# ---------------------------------------------------------------------------


class TestCancellation:
    @pytest.fixture(scope="class")
    def cancel_fixture(self):
        a, dem, canopy = synth_planes(BIG_ROWS, BIG_COLS, seed=29)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 61.0, 12.0, 20.0, BIG_ROWS, BIG_COLS
        )
        targets = random_mask(BIG_ROWS, BIG_COLS, 0.55, 31)
        runs = sw.row_runs_from_mask(targets)
        return inputs, table, runs, targets

    def test_kernel_stale_generation_writes_nothing(self, cancel_fixture):
        """A worker whose expected generation is stale must write NOTHING
        (deterministic, no timing): outputs stay at their base bits."""
        inputs, table, runs, _targets = cancel_fixture
        from solweig_core.numba_cpu.march import _table_columns

        a = np.ascontiguousarray(inputs.a, dtype=np.float32)
        vegdem = np.ascontiguousarray(inputs.vegdsm, dtype=np.float32)
        vegdem2 = np.ascontiguousarray(inputs.vegdsm2, dtype=np.float32)
        dxs, dys, dzs = _table_columns(table, with_dzprev=False)
        shape = a.shape
        sh = np.zeros(shape, dtype=np.float32)
        vegsh = np.zeros(shape, dtype=np.float32)
        vbsh = np.zeros(shape, dtype=np.float32)
        gen = np.array([7], dtype=np.int64)  # bumped: no longer 0
        svf_runs_kernel(
            a, vegdem, vegdem2, dxs, dys, dzs, runs, 0, runs.shape[0],
            gen, 0, sh, vegsh, vbsh, True,
        )
        for plane in (sh, vegsh, vbsh):
            assert np.array_equal(bits(plane), np.zeros_like(bits(plane))), (
                "stale-generation worker wrote outputs"
            )

    def test_kernel_live_generation_writes_targets(self, cancel_fixture):
        inputs, table, runs, targets = cancel_fixture
        from solweig_core.numba_cpu.march import _table_columns

        a = np.ascontiguousarray(inputs.a, dtype=np.float32)
        vegdem = np.ascontiguousarray(inputs.vegdsm, dtype=np.float32)
        vegdem2 = np.ascontiguousarray(inputs.vegdsm2, dtype=np.float32)
        dxs, dys, dzs = _table_columns(table, with_dzprev=False)
        sh = np.zeros(a.shape, dtype=np.float32)
        vegsh = np.zeros(a.shape, dtype=np.float32)
        vbsh = np.zeros(a.shape, dtype=np.float32)
        gen = np.zeros(1, dtype=np.int64)
        svf_runs_kernel(
            a, vegdem, vegdem2, dxs, dys, dzs, runs, 0, runs.shape[0],
            gen, 0, sh, vegsh, vbsh, True,
        )
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        for d, s in zip(dense, (sh, vegsh, vbsh)):
            assert np.array_equal(bits(d)[targets], bits(s)[targets])

    def test_launch_cancel_before_start_no_writes(self, cancel_fixture):
        """cancel() then start(): every worker sees the stale generation
        immediately and the result planes stay at their init bits — and
        stay bit-stable afterwards (no late writes after cancel joined)."""
        inputs, table, runs, _targets = cancel_fixture
        launch = SparseMarchLaunch.svf_shadow(
            table,
            inputs.a,
            inputs.vegdsm,
            inputs.vegdsm2,
            inputs.bush,
            runs,
            n_threads=4,
        )
        assert launch.plan.used_threads == 4
        launch.cancel()
        assert launch.cancelled
        launch.start()
        result = launch.result()
        for plane in result[:3]:
            assert np.array_equal(bits(plane), np.zeros_like(bits(plane))), (
                "cancelled launch wrote outputs"
            )
        # recycled-buffer safety: no write may land after cancel() returned
        snapshot = [p.tobytes() for p in result[:3]]
        time.sleep(0.05)
        assert [p.tobytes() for p in result[:3]] == snapshot

    def test_launch_normal_result_matches_function_api(self, cancel_fixture):
        inputs, table, runs, targets = cancel_fixture
        launch = SparseMarchLaunch.svf_shadow(
            table,
            inputs.a,
            inputs.vegdsm,
            inputs.vegdsm2,
            inputs.bush,
            runs,
            n_threads=4,
        )
        launch.start()
        got = launch.result()
        assert not launch.cancelled
        want = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        for g, w in zip(got[:3], want[:3]):
            assert np.array_equal(bits(g), bits(w))
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        assert_sparse_matches_dense(dense, got, targets, "launch-normal")

    def test_recycled_buffer_not_written_by_stale_worker(self):
        """The card's recycled-buffer RED witness, deterministic form: the
        cancelled build's buffers are handed to a NEW launch (recycled);
        the old build's stale-generation worker must not touch them."""
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, SYN_ROWS, SYN_COLS
        )
        targets = random_mask(SYN_ROWS, SYN_COLS, 0.3, 3)
        runs = sw.row_runs_from_mask(targets)
        old = SparseMarchLaunch.svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs,
            n_threads=4,
        )
        old.cancel()  # buffers would now be recycled by the caller
        recycled = old.result()  # planes stay zero-filled
        before = [p.tobytes() for p in recycled[:3]]
        # a late stale worker runs against the recycled buffers
        old.run_stale_worker_against(recycled[:3])
        assert [p.tobytes() for p in recycled[:3]] == before, (
            "late stale worker wrote into recycled buffers"
        )


# ---------------------------------------------------------------------------
# Domain + input guards (M3 witness)
# ---------------------------------------------------------------------------


class TestGuards:
    def test_domain_guard_refuses_int32_boundary(self):
        """Target addressing is int32 row-run coordinates (T05 layout);
        shapes at the int32 boundary are refused explicitly. Such domains
        are unallocatable on this host — the guard IS the contract."""
        assert check_work_domain(500, 500) is None
        assert check_work_domain(1, 1) is None
        with pytest.raises(ValueError, match="int32"):
            check_work_domain(INT32_MAX, 8)
        with pytest.raises(ValueError, match="int32"):
            check_work_domain(8, INT32_MAX)
        with pytest.raises(ValueError):
            check_work_domain(0, 8)
        with pytest.raises(ValueError):
            check_work_domain(8, 0)
        with pytest.raises(ValueError):
            check_work_domain(-3, 8)

    def test_runner_refuses_oversize_logical_domain(self):
        """The guard is wired into the runner through the table's logical
        domain (not a dead helper): an oversize logical shape refuses even
        when the physical planes are small (the table IS the domain claim
        the march's stop context was built for)."""
        a, dem, canopy = synth_planes(16, 32, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        oversize = st.StepTable(
            key=st.StepTableKey(
                semantics_profile="canonical_cpu_v1",
                kernel_variant=st.KERNEL_SVF_SHADOW,
                source_geometry_hash="0" * 64,
                angle_input_bits=(st.f32_bits_hex(37.0), st.f32_bits_hex(35.0)),
                scale_bits=st.f32_bits_hex(0.5),
                logical_rows=INT32_MAX,
                logical_cols=4,
                amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
                executed_amplitude_bits=st.f32_bits_hex(20.0),
                boundary_policy_id=st.BOUNDARY_SHIFT_WINDOW_V1,
            ),
            count=2,
            dx=np.array([1, 2], dtype=np.int32),
            dy=np.array([0, 0], dtype=np.int32),
            dz_bits=np.array([0x3C23D70A, 0x3CA3D70A], dtype=np.uint32),
            stop_reason=st.STOP_AMPLITUDE,
            branch_id=np.zeros(2, dtype=np.uint8),
            previous_dz_bits=np.zeros(2, dtype=np.uint32),
        )
        runs = sw.row_runs_from_mask(random_mask(16, 32, 0.2, 5))
        with pytest.raises(ValueError):
            march_svf_shadow_sparse(
                oversize,
                inputs.a,
                inputs.vegdsm,
                inputs.vegdsm2,
                inputs.bush,
                runs,
            )

    def test_runs_outside_plane_refused(self):
        a, dem, canopy = synth_planes(16, 32, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, 16, 32
        )
        bad_row = np.array([[16, 0, 3]], dtype=np.int32)
        with pytest.raises(ValueError, match="row"):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                bad_row,
            )
        bad_col = np.array([[0, 0, 32]], dtype=np.int32)
        with pytest.raises(ValueError, match="col"):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                bad_col,
            )
        inverted = np.array([[0, 5, 4]], dtype=np.int32)
        with pytest.raises(ValueError, match="c0"):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                inverted,
            )
        negative = np.array([[-1, 0, 3]], dtype=np.int32)
        with pytest.raises(ValueError):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                negative,
            )

    def test_runs_dtype_and_shape_refused(self):
        a, dem, canopy = synth_planes(16, 32, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, 16, 32
        )
        with pytest.raises(ValueError, match="float"):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                np.array([[0.0, 0, 3]]),
            )
        with pytest.raises(ValueError, match="shape"):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                np.zeros((2, 2), dtype=np.int32),
            )

    def test_variant_mismatch_refused(self):
        a, dem, canopy = synth_planes(16, 32, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        wh_table = synth_table(
            st.KERNEL_WALLHEIGHT_23, 37.0, 35.0, 20.0, 16, 32
        )
        runs = sw.row_runs_from_mask(random_mask(16, 32, 0.3, 7))
        with pytest.raises(ValueError, match="variant"):
            march_svf_shadow_sparse(
                wh_table, inputs.a, inputs.vegdsm, inputs.vegdsm2,
                inputs.bush, runs,
            )
        svf_table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, 16, 32
        )
        with pytest.raises(ValueError, match="variant"):
            march_wallheight23_sparse(
                svf_table, inputs.a, inputs.vegdsm, inputs.vegdsm2,
                inputs.bush, runs,
            )

    def test_table_shape_mismatch_refused(self):
        a, dem, canopy = synth_planes(16, 32, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, 8, 32
        )
        runs = sw.row_runs_from_mask(random_mask(16, 32, 0.3, 7))
        with pytest.raises(ValueError, match="shape"):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush,
                runs,
            )

    def test_bush_refused(self):
        a, dem, canopy = synth_planes(16, 32, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        bushy = inputs.bush.copy()
        bushy[2, 3] = np.float32(2.0)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, 16, 32
        )
        runs = sw.row_runs_from_mask(random_mask(16, 32, 0.3, 7))
        with pytest.raises(ValueError, match="bush"):
            march_svf_shadow_sparse(
                table, inputs.a, inputs.vegdsm, inputs.vegdsm2, bushy, runs
            )

    def test_wrong_plane_dtype_refused(self):
        a, dem, canopy = synth_planes(16, 32, seed=7)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 20.0, 16, 32
        )
        runs = sw.row_runs_from_mask(random_mask(16, 32, 0.3, 7))
        with pytest.raises(ValueError, match="float32"):
            march_svf_shadow_sparse(
                table,
                inputs.a.astype(np.float64),
                inputs.vegdsm,
                inputs.vegdsm2,
                inputs.bush,
                runs,
            )


# ---------------------------------------------------------------------------
# Physical crop must not change logical semantics (M5 designated fixture)
# ---------------------------------------------------------------------------


class TestNoPhysicalCrop:
    def test_reads_reach_beyond_closure_bbox(self):
        """A crop-based implementation would zero reads outside the work
        list's bounding box; the runner must read the FULL logical domain.
        Fixture: a low-altitude (long-march) table whose corridor exits the
        bbox on one side, with tall vegetation just outside the bbox so the
        read values actually change target outputs."""
        rows, cols = 64, 96
        rng = np.random.default_rng(43)
        dem = np.zeros((rows, cols), dtype=np.float32)
        a = dem + rng.uniform(0.0, 2.0, (rows, cols)).astype(np.float32)
        canopy = np.zeros((rows, cols), dtype=np.float32)
        # target blob at the domain centre
        rr, cc = np.ogrid[:rows, :cols]
        targets = (rr - 32) ** 2 + (cc - 48) ** 2 <= 25
        # tall canopy ring well OUTSIDE the target bbox, inside the domain
        outside_ring = (rr - 32) ** 2 + (cc - 48) ** 2 <= 900
        outside_ring &= ~((rr - 32) ** 2 + (cc - 48) ** 2 <= 400)
        canopy[outside_ring] = np.float32(30.0)
        inputs = synth_inputs(a, dem, canopy)
        table = synth_table(
            st.KERNEL_SVF_SHADOW, 90.0, 6.0, 60.0, rows, cols
        )
        runs = sw.row_runs_from_mask(targets)
        # fixture honesty: the bbox must NOT contain the ring, and the march
        # must be long enough to read it
        tr, tc = np.nonzero(targets)
        assert tr.max() + 1 < rows and tr.min() > 0
        assert not (
            outside_ring[tr.min() : tr.max() + 1, tc.min() : tc.max() + 1]
        ).any(), "fixture broken: ring inside the target bbox"
        assert table.count > 30, "fixture march too short to leave the bbox"
        dense = march_svf_shadow(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
        )
        sparse = march_svf_shadow_sparse(
            table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush, runs
        )
        assert_sparse_matches_dense(dense, sparse, targets, "no-crop")
        # the read actually mattered: some target sees the tall canopy
        assert (bits(dense[1])[targets] == np.uint32(0)).any(), (
            "fixture too weak: outside-bbox canopy never shadowed a target"
        )


# ---------------------------------------------------------------------------
# site_500 gate: dense serial vs sparse serial vs sparse parallel
# ---------------------------------------------------------------------------

SITE_SUBSET_ALTITUDES = {6.0, 42.0, 78.0, 90.0}


@pytest.fixture(scope="module")
def site():
    if not SITE_500_CACHE.is_dir():
        pytest.skip(f"site cache {SITE_500_CACHE} not present")
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.solver import (
        _compose_scene_from_canopy,
        _sky_patch_geometry,
    )
    from trace_exporter import capture_amplitude_policy, executed_amplitude_for_patch

    cache = SiteCache.load(SITE_500_CACHE)
    canopy_base = np.asarray(cache.tree_base, dtype=np.float32).copy()
    patches, _rings = _sky_patch_geometry(2)
    return {
        "cache": cache,
        "canopy_base": canopy_base,
        "scale": 1.0 / float(cache.pixel_size_m),
        "rows": int(cache.rows),
        "cols": int(cache.cols),
        "patches": patches,
        "capture_amplitude_policy": capture_amplitude_policy,
        "executed_amplitude_for_patch": executed_amplitude_for_patch,
        "compose_torch": _compose_scene_from_canopy,
    }


_SCENE_CACHE: dict[str, dict] = {}


def site_scene(state, name, canopy):
    cached = _SCENE_CACHE.get(name)
    if cached is not None:
        return cached
    a = np.array(state["cache"].building_dsm)
    dem = np.array(state["cache"].dem)
    inputs = sw.compose_march_inputs(a, dem, canopy)
    scene_t = state["compose_torch"](state["cache"], canopy)
    policy = state["capture_amplitude_policy"](
        scene_t.a, scene_t.vegdsm, scene_t.vegdsm2, scene_t.vegdem,
        scale=state["scale"],
    )
    amps = {
        idx: state["executed_amplitude_for_patch"](policy, idx, escalated=False)
        for idx in range(len(state["patches"]))
    }
    cached = {"inputs": inputs, "amps": amps}
    _SCENE_CACHE[name] = cached
    return cached


def site_table(state, scene, idx):
    altitude, azimuth, _ring = state["patches"][idx]
    amp, pid = scene["amps"][idx]
    trace = capture_trace(
        st.KERNEL_SVF_SHADOW,
        float(azimuth),
        float(altitude),
        state["scale"],
        state["rows"],
        state["cols"],
        amp,
        amplitude_policy_id=pid,
        verify_probes=False,
    )
    return st.build_table_from_trace(trace)


def site_canopy_with(state, trees):
    from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow, TreeSpec
    from solweig_gpu.incremental.trees import TreeLayer

    grid = RasterGrid(
        rows=state["rows"],
        cols=state["cols"],
        pixel_size_m=state["cache"].pixel_size_m,
        origin_x_m=state["cache"].manifest.origin_x_m,
        origin_y_m=state["cache"].manifest.origin_y_m,
    )
    layer = TreeLayer(state["canopy_base"], grid)
    for tree_id, cell, height, radius in trees:
        x = grid.origin_x_m + (cell[1] + 0.5) * grid.pixel_size_m
        y = grid.origin_y_m - (cell[0] + 0.5) * grid.pixel_size_m
        layer.add_tree(TreeSpec(tree_id, x, y, height, radius))
    canopy = np.ascontiguousarray(
        layer.vegetation_rasters_window(
            RasterWindow(0, state["rows"], 0, state["cols"])
        )[0]
    )
    return canopy


def subset_patch_indices(state):
    return [
        i
        for i, patch in enumerate(state["patches"])
        if float(patch[0]) in SITE_SUBSET_ALTITUDES
    ]


def run_site_gate(state, label, canopy_pre, canopy_post, patch_indices):
    """Dense serial vs sparse serial vs sparse parallel 1/2/4/8 on one
    site edit family; returns the measurement record (0 mismatches)."""
    scene_pre = site_scene(state, f"{label}_pre", canopy_pre)
    scene_post = site_scene(state, f"{label}_post", canopy_post)
    inputs_post = scene_post["inputs"]
    tables_pre = {
        idx: site_table(state, scene_pre, idx) for idx in patch_indices
    }
    tables_post = {
        idx: site_table(state, scene_post, idx) for idx in patch_indices
    }
    build = sw.build_sparse_work(
        scene_pre["inputs"], inputs_post, tables_pre, tables_post,
        keep_masks=True,
    )
    masks = build.masks()
    timings = {"dense_s": 0.0, "sparse_serial_s": 0.0}
    timings.update({f"sparse_t{t}_s": 0.0 for t in THREAD_COUNTS})
    mismatch_count = 0
    targets_total = 0
    for idx in sorted(tables_post):
        t_post = tables_post[idx]
        runs = build.row_runs.get(idx)
        targets = masks.get(
            idx, np.zeros((state["rows"], state["cols"]), dtype=bool)
        )
        t0 = time.perf_counter()
        dense = march_svf_shadow(
            t_post, inputs_post.a, inputs_post.vegdsm, inputs_post.vegdsm2,
            inputs_post.bush,
        )
        timings["dense_s"] += time.perf_counter() - t0
        variants = {"serial": None}
        for name, threads in [("serial", 1)] + [
            (f"t{t}", t) for t in THREAD_COUNTS
        ]:
            if runs is None or runs.shape[0] == 0:
                continue
            t0 = time.perf_counter()
            result = march_svf_shadow_sparse(
                t_post, inputs_post.a, inputs_post.vegdsm, inputs_post.vegdsm2,
                inputs_post.bush, runs, n_threads=threads,
            )
            timings[
                "sparse_serial_s" if name == "serial" else f"sparse_{name}_s"
            ] += time.perf_counter() - t0
            variants[name] = result
        targets_total += int(targets.sum())
        for name, result in variants.items():
            if result is None:
                continue
            for d, s in zip(dense, result):
                mismatch_count += int((bits(d)[targets] != bits(s)[targets]).sum())
                outside = bits(s)[~targets]
                mismatch_count += int(
                    (outside != np.zeros_like(outside)).sum()
                )
    assert mismatch_count == 0, f"{label}: {mismatch_count} bit mismatches"
    return {
        "family": label,
        "patches": len(tables_post),
        "pair_count": int(build.pair_count),
        "row_run_count": int(build.row_run_count),
        "targets_total": targets_total,
        "pair_steps_sparse": int(build.pair_steps_sparse),
        "pair_steps_dense": int(build.pair_steps_dense),
        "build_seconds": build.build_seconds,
        "timings": timings,
        "mismatch_count": mismatch_count,
    }


class TestSiteGate:
    def test_site_f1_f2_f3_gate(self, site):
        canopy_base = site["canopy_base"]
        canopy_f1 = site_canopy_with(site, [("t1", (166, 102), 10.0, 4.0)])
        canopy_f2_pre = canopy_f1
        canopy_f2_post = site_canopy_with(site, [("t1", (166, 127), 10.0, 4.0)])
        tall = float(canopy_base.max()) + 25.0
        short = tall - 5.0
        canopy_f3_pre = site_canopy_with(
            site,
            [("A", (166, 150), tall, 8.0), ("B", (166, 153), short, 8.0)],
        )
        canopy_f3_post = site_canopy_with(
            site, [("B", (166, 153), short, 8.0)]
        )
        subset = subset_patch_indices(site)
        records = [
            run_site_gate(site, "F1_add_t1", canopy_base, canopy_f1, subset),
            run_site_gate(
                site, "F2_move_t1", canopy_f2_pre, canopy_f2_post, subset
            ),
            run_site_gate(
                site,
                "F3_delete_overlap",
                canopy_f3_pre,
                canopy_f3_post,
                list(range(len(site["patches"]))),
            ),
        ]
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "site_gate_measurement.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n"
        )
        for record in records:
            assert record["pair_count"] > 0, record["family"]


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestPurity:
    def test_source_torch_free_no_fastmath_no_prange(self):
        source = (
            REPO_ROOT / "solweig_core" / "numba_cpu" / "sparse_march.py"
        ).read_text()
        assert "import torch" not in source
        assert "from torch" not in source
        assert "fastmath=True" not in source
        assert "parallel=True" not in source
        assert "prange" not in source

    def test_module_torch_free_subprocess(self):
        code = (
            "import sys; "
            "import numpy as np; "
            "from solweig_core import step_tables as st; "
            "from solweig_core.numba_cpu.sparse_march import ("
            "march_svf_shadow_sparse, plan_sparse_march); "
            "assert 'torch' not in sys.modules; "
            "table = st.build_table_from_trace({"
            "'schema_version': 1, "
            "'key': {'semantics_profile': 'canonical_cpu_v1', "
            "'kernel_variant': 'svf_shadow', "
            "'source_geometry_hash': 'x'*64, "
            "'angle_input_bits': ['0x42140000','0x420c0000'], "
            "'scale_bits': '0x3f000000', "
            "'logical_rows': 4, 'logical_cols': 8, "
            "'amplitude_policy_id': 'synthetic_probe', "
            "'executed_amplitude_bits': '0x41a00000', "
            "'boundary_policy_id': 'shadow_shift_window_v1'}, "
            "'count': 0, 'dx': [], 'dy': [], 'dz_bits': [], "
            "'stop_reason': 'amplitude', "
            "'azimuth_zero_substituted': False, 'trace_id': 'purity'}); "
            "a = np.zeros((4, 8), dtype=np.float32); "
            "runs = np.array([[0, 0, 7]], dtype=np.int32); "
            "out = march_svf_shadow_sparse("
            "table, a, a.copy(), a.copy(), a.copy(), runs); "
            "assert out[0].shape == (4, 8) and out[0].dtype == np.float32; "
            "plan = plan_sparse_march(runs, n_threads=4); "
            "assert plan.used_threads == 1"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
