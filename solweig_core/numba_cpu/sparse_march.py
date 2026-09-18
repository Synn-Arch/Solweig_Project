# SPDX-License-Identifier: GPL-3.0-only
"""Sparse Numba march runner + safe CPU parallelism (DESIGN.ko.md
3.5/7.5/7.6/7.7, TASKS T06).

The T05 closure builder (:func:`solweig_core.sparse_work.build_sparse_work`)
says WHICH ``(cell, patch)`` state machines must re-run; this module RUNS
them. Every sparse target executes the T04 dense kernel's per-target
primitive graph VERBATIM — the loop bodies below are copied from
``solweig_core/numba_cpu/march.py`` (read-only for this task), with only
the cell-loop head replaced by row-run iteration. Nothing is re-derived:
OOB reads stay the exact zero temporary, the first-step reset, the
target-local trunk gate, the suppression order and the final threshold
graph are byte-for-byte the dense code, so the sparse path executes, for
each target cell, the SAME per-step sequence as the dense path.

Parallelism model (DESIGN 3.5 — "independent cells may run in parallel"):

* Work list representation is the T05 sorted row-run array ``(row, c0,
  c1_inclusive)`` int32. Targets are addressed by (row, col) coordinates
  — a LINEAR cell index is never formed, so no ``row*cols + col`` int32
  overflow exists by construction; :func:`check_work_domain` refuses
  logical domains at the int32 addressing boundary as an explicit guard.
* Ownership is a static EXACT partition of run indices into contiguous
  ranges, balanced by cumulative cell count (:func:`plan_chunks`). Each
  run belongs to exactly ONE range; ranges are re-verified before any
  thread spawns (:func:`_verify_chunks`). Duplicate ownership is refused,
  never "optimised" into a benign race.
* Threads are plain ``threading.Thread``s (daemon) calling ONE
  ``njit(cache=True, nogil=True, fastmath=False)`` kernel with disjoint
  run ranges. Every thread's arithmetic IS serial arithmetic — the same
  compiled code, the same per-target order — so bit equality across
  thread counts is structural, and the gate proves it. Numba
  auto-parallel compilation is never used: no reduction, no
  reassociation, no scheduler-owned execution semantics to argue about.
* Work below :data:`SERIAL_MIN_PAIRS` pairs routes serial (one kernel
  call on the calling thread, zero thread spawn) — a measured decision;
  the crossover is benchmarked in the T06 artifacts. T05's
  ``ROUTE_DENSE_PAIR_FRACTION`` per-patch ceiling is honoured by the
  build-level runner :func:`march_build` (dense-routed patches go through
  the T04 dense kernel unchanged).

Cancellation seam (narrow, explicit):

* Every kernel re-reads a shared generation counter before EACH row-run;
  a worker whose expected generation is stale writes nothing and returns.
* :class:`SparseMarchLaunch` exposes start/cancel/result. ``cancel()``
  bumps the generation and JOINS the workers (a worker notices at its
  next run boundary — the staleness window is one row-run), so once
  ``cancel()`` returns no further output write can occur and the buffers
  are safe to recycle. A RED witness pins the seam: a stale-generation
  worker must not write.

The runner never crops scene buffers to the closure bbox: physical reads
always address the FULL logical domain (a crop would flip in-bounds reads
to OOB zeros at the bbox edge and change target outputs — the card's
"physical crop changing logical stop" witness). The table's logical shape
IS the march's stop context and is checked against the physical planes.

Supported domain: ``bush.max() <= 0`` (re-checked through the T04 guard).
The T05 closure builder proves closures for the ``svf_shadow`` variant
only; the ``wallheight_23`` sparse runner is provided for explicit work
lists (synthetic gates) — its closure builder is future work.

T19 suffix exit (T27 verdict C6): both run kernels carry the same
post-onset-and-suppressed fixed-point break as the dense kernels
(:mod:`solweig_core.numba_cpu.march` — the exit lines below stay textually
parallel with the dense module's, per this file's verbatim-copy
discipline). The ``allow_exit`` flag is decided ONCE per march call at the
Python boundary from the a-plane NaN guard (:func:`march._suffix_exit_allowed`),
so every chunk/thread of one launch behaves identically; the exit is
output-invariant, so thread-count bit equality (the T06 gate) is
structural exactly as before.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import numpy as np
from numba import njit

from solweig_core import step_tables as st
from solweig_core.numba_cpu.march import (
    _as_kernel_buffer,
    _check_bush,
    _check_table,
    _suffix_exit_allowed,
    _table_columns,
    _torch_maximum,
    march_svf_shadow,
)

__all__ = [
    "INT32_MAX",
    "SERIAL_MIN_PAIRS",
    "SparseMarchPlan",
    "SvfSparseResult",
    "WallheightSparseResult",
    "SparseMarchLaunch",
    "check_work_domain",
    "plan_chunks",
    "plan_sparse_march",
    "march_svf_shadow_sparse",
    "march_wallheight23_sparse",
    "march_build",
]

#: int32 boundary for target addressing. Row-run coordinates and the
#: kernel's column iteration are int32; ``limit`` leaves headroom for the
#: ``c1 + 1`` and shifted-index arithmetic to stay in range.
INT32_MAX = np.iinfo(np.int32).max
_INT32_ADDRESS_LIMIT = INT32_MAX - 1

#: Route serial (no thread spawn) below this many target cells. Thread
#: spawn+join costs tens of microseconds; a sparse pair-step costs a few
#: nanoseconds, so tiny edits must never pay for threads. Initial value
#: from the representation-cost model; the measured crossover lives in
#: the T06 bench artifacts (bench_ablation.txt).
SERIAL_MIN_PAIRS = 16384


# ---------------------------------------------------------------------------
# Kernels: the dense per-target state machine over a row-run work list
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True, fastmath=False)
def _svf_shadow_runs_kernel(
    a, vegdem, vegdem2, dxs, dys, dzs, runs, run_lo, run_hi, gen, expect_gen,
    sh_out, vegsh_out, vbsh_out, allow_exit,
):
    """``march._svf_shadow_kernel``'s cell body, driven by row runs.

    The body between ``for j ...`` and the output writes is copied
    VERBATIM from the dense kernel (do not "clean up" the two kernels
    separately — they must stay textually parallel so any semantics fix
    lands in both). The generation check at each run head is the
    cancellation seam: a stale generation writes nothing.
    """
    rows = a.shape[0]
    cols = a.shape[1]
    count = dxs.shape[0]
    zero = np.float32(0.0)
    one = np.float32(1.0)
    big = np.float32(1000.0)
    for k in range(run_lo, run_hi):
        if gen[0] != expect_gen:
            return
        i = runs[k, 0]
        c0 = runs[k, 1]
        c1 = runs[k, 2]
        for j in range(c0, c1 + 1):
            a_t = a[i, j]
            f = a_t                       # f = a.clone()
            sh = zero
            vegsh = zero                  # vegsh = zeros + bushplant (0 here)
            vbsh = zero
            for s in range(count):
                dz = dzs[s]
                si = i + dxs[s]
                sj = j + dys[s]
                if si >= 0 and si < rows and sj >= 0 and sj < cols:
                    tv = vegdem[si, sj] - dz
                    tv2 = vegdem2[si, sj] - dz
                    ta = a[si, sj] - dz
                else:
                    tv = zero             # temporaries keep their zero init
                    tv2 = zero
                    ta = zero
                f = _torch_maximum(f, ta)
                if f > a_t:
                    sh = one
                else:
                    sh = zero
                fab = one if tv > a_t else zero
                gab = one if tv2 > a_t else zero
                vegsh2 = fab - gab
                vegsh = _torch_maximum(vegsh, vegsh2)
                if vegsh * sh > zero:      # building suppression
                    vegsh = zero
                vbsh = vegsh + vbsh        # accumulator (source operand order)
                if s == 0:                 # index == 1.: first-step exception
                    fv = tv - ta
                    if fv <= zero:
                        fv = big
                    if fv < dz:
                        vegsh = one
                    if vegdem2[i, j] > a_t:    # target-local trunk gate
                        vegsh = vegsh * one
                    else:
                        vegsh = vegsh * zero
                    vbsh = zero           # vbshvegsh.zero_()
                if allow_exit and sh == one and vegsh == zero:
                    # T19/T27-C6: post-onset-and-suppressed fixed point —
                    # the remaining steps are exactly 0-contribution. The
                    # vegsh == zero conjunct is load-bearing (the s == 0
                    # exception can leave vegsh == 1 AFTER suppression).
                    break
            # final graph, source order (shadow.py:303-316, bush tail skipped)
            sh = one - sh
            if vbsh > zero:
                vbsh = one
            vbsh = vbsh - vegsh
            if vegsh > zero:
                vegsh = one
            vegsh = one - vegsh
            vbsh = one - vbsh
            sh_out[i, j] = sh
            vegsh_out[i, j] = vegsh
            vbsh_out[i, j] = vbsh


@njit(cache=True, nogil=True, fastmath=False)
def _wallheight23_runs_kernel(
    a, vegdem, vegdem2, dxs, dys, dzs, dzprevs, runs, run_lo, run_hi, gen,
    expect_gen, sh_out, vegsh_out, vbsh_out, allow_exit,
):
    """``march._wallheight23_kernel``'s cell body, driven by row runs."""
    rows = a.shape[0]
    cols = a.shape[1]
    count = dxs.shape[0]
    zero = np.float32(0.0)
    one = np.float32(1.0)
    four = np.float32(4.0)
    for k in range(run_lo, run_hi):
        if gen[0] != expect_gen:
            return
        i = runs[k, 0]
        c0 = runs[k, 1]
        c1 = runs[k, 2]
        for j in range(c0, c1 + 1):
            a_t = a[i, j]
            f = a_t                        # f = a
            sh = zero
            vegsh = zero                   # zeros + bushplant (0 under guard)
            vbsh = zero
            for s in range(count):
                dz = dzs[s]
                dzp = dzprevs[s]
                si = i + dxs[s]
                sj = j + dys[s]
                if si >= 0 and si < rows and sj >= 0 and sj < cols:
                    tv = vegdem[si, sj] - dz
                    tv2 = vegdem2[si, sj] - dz
                    ta = a[si, sj] - dz
                    tlf = vegdem[si, sj] - dzp
                    tlg = vegdem2[si, sj] - dzp
                else:
                    tv = zero
                    tv2 = zero
                    ta = zero
                    tlf = zero
                    tlg = zero
                f = _torch_maximum(f, ta)
                if f > a_t:
                    sh = one
                else:
                    sh = zero
                fa = one if tv > a_t else zero
                ga = one if tv2 > a_t else zero
                lfa = one if tlf > a_t else zero
                lga = one if tlg > a_t else zero
                v2 = fa + ga               # fabovea + gabovea
                v2 = v2 + lfa              # + lastfabovea.float()
                v2 = v2 + lga              # + lastgabovea.float()
                if v2 == four:
                    v2 = zero
                if v2 > zero:
                    v2 = one
                vegsh = _torch_maximum(vegsh, v2)
                if vegsh * sh > zero:       # building suppression
                    vegsh = zero
                vbsh = vbsh + vegsh         # accumulator (source operand order)
                if allow_exit and sh == one and vegsh == zero:
                    # T19/T27-C6, wallheight_23 form: same fixed point. The
                    # dzprev coupling (tlf/tlg -> lfa/lga -> v2) only feeds
                    # v2 INSIDE the step, and the post-max suppression
                    # re-zeroes vegsh before the accumulator, so
                    # (sh=1, vegsh=0, vbsh) is stationary for every later
                    # step. This variant has NO first-step exception (its
                    # step 0 is the (0, 0) self comparison), so no extra
                    # conjunct is needed beyond the svf form.
                    break
            # final graph, source order (solweig.py:1198-1205)
            sh = one - sh
            if vbsh > zero:
                vbsh = one
            vbsh = vbsh - vegsh
            if vegsh > zero:
                vegsh = one
            vegsh = one - vegsh
            vbsh = one - vbsh
            sh_out[i, j] = sh
            vegsh_out[i, j] = vegsh
            vbsh_out[i, j] = vbsh


# ---------------------------------------------------------------------------
# Domain + work-list validation
# ---------------------------------------------------------------------------


def check_work_domain(rows, cols) -> None:
    """Refuse logical domains at the int32 target-addressing boundary.

    Targets are addressed as int32 (row, col) coordinates from the T05
    row-run layout and the kernels iterate columns in int32; a domain at
    the boundary would wrap ``c1 + 1`` and the shifted-index arithmetic.
    Such domains are unallocatable on this host — the guard is the
    contract, checked against the TABLE's logical shape (the stop
    context) and the physical planes.
    """
    rows = int(rows)
    cols = int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(
            f"logical domain {rows}x{cols} must be positive"
        )
    if rows > _INT32_ADDRESS_LIMIT or cols > _INT32_ADDRESS_LIMIT:
        raise ValueError(
            f"logical domain {rows}x{cols} exceeds the int32 target-"
            f"addressing boundary {_INT32_ADDRESS_LIMIT} (row-run "
            "coordinates and column iteration are int32) — refusing "
            "rather than wrapping"
        )


def _as_runs(runs) -> np.ndarray:
    """Structural validation: (n, 3) integer array, int32-representable,
    non-negative coordinates, ``c0 <= c1``. Returns contiguous int32."""
    arr = np.asarray(runs)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"runs must be an (n, 3) row-run array, got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"runs must be an integer array, got dtype {arr.dtype}"
        )
    if int(arr.min()) < 0 or int(arr.max()) > INT32_MAX:
        raise ValueError(
            "run coordinates must be in [0, int32max], got "
            f"[{int(arr.min())}, {int(arr.max())}]"
        )
    arr32 = np.ascontiguousarray(arr, dtype=np.int32)
    rows_bad = arr32[:, 0] < 0
    if rows_bad.any():
        raise ValueError(
            f"row-run row must be non-negative, first bad row "
            f"{int(arr32[rows_bad][0, 0])}"
        )
    cols_bad = (arr32[:, 1] < 0) | (arr32[:, 2] < 0)
    if cols_bad.any():
        first = arr32[cols_bad][0]
        raise ValueError(
            f"row-run col endpoints must be non-negative, got c0={int(first[1])} "
            f"c1={int(first[2])}"
        )
    inverted = arr32[:, 1] > arr32[:, 2]
    if inverted.any():
        first = arr32[inverted][0]
        raise ValueError(
            f"row-run c0 must be <= c1, got c0={int(first[1])} > "
            f"c1={int(first[2])}"
        )
    return arr32


def _validate_runs_against_plane(runs32: np.ndarray, rows: int, cols: int):
    if runs32.shape[0] == 0:
        return runs32
    row_bad = runs32[:, 0] >= rows
    if row_bad.any():
        raise ValueError(
            f"row-run row {int(runs32[row_bad][0, 0])} outside the plane "
            f"rows [0, {rows})"
        )
    col_bad = runs32[:, 2] >= cols
    if col_bad.any():
        raise ValueError(
            f"row-run col {int(runs32[col_bad][0, 2])} outside the plane "
            f"cols [0, {cols})"
        )
    return runs32


# ---------------------------------------------------------------------------
# Planning: exact chunk ownership + serial/parallel routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseMarchPlan:
    """One march execution's routing decision (pure data, deterministic)."""

    pair_count: int
    n_runs: int
    used_threads: int
    chunks: tuple[tuple[int, int], ...]
    route: str = "sparse"


def plan_chunks(run_lengths, n_threads: int) -> tuple[tuple[int, int], ...]:
    """EXACT contiguous partition of run indices into <= n_threads ranges.

    Boundaries are placed at cumulative CELL-COUNT targets (load balance:
    run widths are wildly heterogeneous in corridor closures), computed
    deterministically. The result is sorted, contiguous, non-empty and
    covers ``[0, n_runs)`` with every run in exactly one range — the
    ownership invariant; :func:`_verify_chunks` re-checks it before any
    thread spawns. ``run_lengths`` may be any 1-D non-negative integer
    sequence.
    """
    lengths = np.asarray(run_lengths, dtype=np.int64).reshape(-1)
    if lengths.size and int(lengths.min()) < 0:
        raise ValueError("run lengths must be non-negative")
    n_runs = int(lengths.shape[0])
    if n_runs == 0 or n_threads <= 0:
        return ()
    total = int(lengths.sum())
    if total <= 0:
        # degenerate all-empty work: fall back to a count partition so the
        # ranges still tile [0, n_runs)
        total = n_runs
        lengths = np.ones(n_runs, dtype=np.int64)
    n_workers = min(int(n_threads), n_runs)
    cum = np.zeros(n_runs + 1, dtype=np.int64)
    np.cumsum(lengths, out=cum[1:])
    chunks: list[tuple[int, int]] = []
    prev_hi = 0
    for k in range(n_workers):
        target = (k + 1) * total // n_workers if k + 1 < n_workers else total
        hi = int(np.searchsorted(cum, target, side="left"))
        hi = max(hi, prev_hi)
        hi = min(hi, n_runs)
        if k + 1 == n_workers:
            hi = n_runs
        if hi > prev_hi:
            chunks.append((prev_hi, hi))
        prev_hi = hi
    return tuple(chunks)


def _verify_chunks(chunks, n_runs: int) -> None:
    """Runtime ownership guard: exact partition or refuse to spawn."""
    expected = 0
    for lo, hi in chunks:
        if not (lo == expected and hi > lo and hi <= n_runs):
            raise RuntimeError(
                f"chunk plan is not an exact partition of [0, {n_runs}): "
                f"{chunks} — duplicate or missing run ownership refused"
            )
        expected = hi
    if expected != n_runs:
        raise RuntimeError(
            f"chunk plan does not cover [0, {n_runs}): {chunks}"
        )


def _run_lengths(runs32: np.ndarray) -> np.ndarray:
    if runs32.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return (
        runs32[:, 2].astype(np.int64) - runs32[:, 1].astype(np.int64) + 1
    )


def plan_sparse_march(
    runs, n_threads: int = 1, *, min_pairs: int = SERIAL_MIN_PAIRS
) -> SparseMarchPlan:
    """Route one work list: serial below ``min_pairs`` pairs, else a
    static exact-partition parallel plan over row runs."""
    runs32 = _as_runs(runs)
    n_runs = int(runs32.shape[0])
    lengths = _run_lengths(runs32)
    pair_count = int(lengths.sum())
    if n_threads <= 1 or n_runs == 0 or pair_count < int(min_pairs):
        used = 1
        chunks = ((0, n_runs),) if n_runs else ()
    else:
        chunks = plan_chunks(lengths, n_threads)
        used = len(chunks)
    return SparseMarchPlan(
        pair_count=pair_count,
        n_runs=n_runs,
        used_threads=used,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute_chunks(call: Callable[[int, int], None], plan: SparseMarchPlan):
    """Run the plan's chunks: one chunk inline (serial, zero thread spawn),
    several chunks on daemon threads with the calling thread taking the
    first chunk. Every thread runs the same njit kernel with ``nogil`` —
    true parallelism with serial arithmetic per thread.
    """
    chunks = plan.chunks
    if not chunks:
        return
    _verify_chunks(chunks, plan.n_runs)
    if len(chunks) == 1:
        lo, hi = chunks[0]
        call(lo, hi)
        return
    call(0, 0)  # force compilation before threads hit the dispatch lock
    threads = [
        threading.Thread(target=call, args=(lo, hi), daemon=True)
        for lo, hi in chunks[1:]
    ]
    for t in threads:
        t.start()
    call(*chunks[0])  # the calling thread owns the first chunk
    for t in threads:
        t.join()


class SvfSparseResult(NamedTuple):
    sh: np.ndarray
    vegsh: np.ndarray
    vbshvegsh: np.ndarray
    plan: SparseMarchPlan


class WallheightSparseResult(NamedTuple):
    vegsh: np.ndarray
    sh: np.ndarray
    vbshvegsh: np.ndarray
    plan: SparseMarchPlan


def _alloc_outputs(base, shape) -> list[np.ndarray]:
    """Three fresh output planes: copies of ``base`` or zeros."""
    if base is None:
        return [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    planes = list(base)
    if len(planes) != 3:
        raise ValueError(f"base must be three planes, got {len(planes)}")
    out = []
    for plane in planes:
        arr = np.asarray(plane)
        if arr.shape != tuple(shape) or arr.dtype != np.dtype(np.float32):
            raise ValueError(
                f"base plane must be float32 of shape {tuple(shape)}, got "
                f"{arr.dtype.str} {arr.shape}"
            )
        out.append(np.ascontiguousarray(arr).copy())
    return out


def _prepare(
    variant: str,
    table: st.StepTable,
    a,
    vegdem,
    vegdem2,
    bush,
    runs,
    n_threads,
    base,
    min_pairs,
):
    """Shared validation + buffer prep for both variants."""
    check_work_domain(table.key.logical_rows, table.key.logical_cols)
    buffers = [
        _as_kernel_buffer(x, name, f"sparse_{variant}:{table.trace_id}")
        for x, name in ((a, "a"), (vegdem, "vegdem"), (vegdem2, "vegdem2"), (bush, "bush"))
    ]
    shape = buffers[0].shape
    _check_table(table, variant, shape)
    check_work_domain(shape[0], shape[1])
    _check_bush(buffers[3])
    runs32 = _validate_runs_against_plane(
        _as_runs(runs), int(shape[0]), int(shape[1])
    )
    plan = plan_sparse_march(runs32, n_threads, min_pairs=min_pairs)
    outs = _alloc_outputs(base, shape)
    return buffers, runs32, plan, outs


def march_svf_shadow_sparse(
    table: st.StepTable,
    a,
    vegdem,
    vegdem2,
    bush,
    runs,
    *,
    n_threads: int = 1,
    base=None,
    min_pairs: int = SERIAL_MIN_PAIRS,
    suffix_exit: bool = True,
) -> SvfSparseResult:
    """Sparse replay of :func:`solweig_core.numba_cpu.march.march_svf_shadow`
    over a row-run work list.

    Returns ``(sh, vegsh, vbshvegsh)`` planes where every work-list cell
    is bit-equal to the dense kernel and every other cell keeps its base
    (zeros when no base is given) bit-for-bit. ``suffix_exit`` toggles the
    T19 post-onset suffix exit (bit-invariant; the a-plane NaN guard
    decides per call — see :func:`march._suffix_exit_allowed`).
    """
    buffers, runs32, plan, outs = _prepare(
        st.KERNEL_SVF_SHADOW,
        table, a, vegdem, vegdem2, bush, runs, n_threads, base, min_pairs,
    )
    dxs, dys, dzs = _table_columns(table, with_dzprev=False)
    allow_exit = _suffix_exit_allowed(buffers[0], suffix_exit)
    gen = np.zeros(1, dtype=np.int64)

    def call(lo: int, hi: int) -> None:
        _svf_shadow_runs_kernel(
            buffers[0], buffers[1], buffers[2], dxs, dys, dzs, runs32, lo,
            hi, gen, 0, outs[0], outs[1], outs[2], allow_exit,
        )

    _execute_chunks(call, plan)
    return SvfSparseResult(outs[0], outs[1], outs[2], plan)


def march_wallheight23_sparse(
    table: st.StepTable,
    a,
    vegdem,
    vegdem2,
    bush,
    runs,
    *,
    n_threads: int = 1,
    base=None,
    min_pairs: int = SERIAL_MIN_PAIRS,
    suffix_exit: bool = True,
) -> WallheightSparseResult:
    """Sparse replay of the ``shadowingfunction_wallheight_23`` march over
    an explicit row-run work list (the T05 closure builder proves
    ``svf_shadow`` closures only — wallheight work lists come from
    callers that computed them independently). ``suffix_exit`` toggles the
    T19 post-onset suffix exit (bit-invariant)."""
    buffers, runs32, plan, outs = _prepare(
        st.KERNEL_WALLHEIGHT_23,
        table, a, vegdem, vegdem2, bush, runs, n_threads, base, min_pairs,
    )
    dxs, dys, dzs, dzprevs = _table_columns(table, with_dzprev=True)
    allow_exit = _suffix_exit_allowed(buffers[0], suffix_exit)
    gen = np.zeros(1, dtype=np.int64)

    def call(lo: int, hi: int) -> None:
        _wallheight23_runs_kernel(
            buffers[0], buffers[1], buffers[2], dxs, dys, dzs, dzprevs,
            runs32, lo, hi, gen, 0, outs[0], outs[1], outs[2], allow_exit,
        )

    _execute_chunks(call, plan)
    return WallheightSparseResult(outs[1], outs[0], outs[2], plan)


class SparseMarchLaunch:
    """Explicit start/cancel lifecycle around one parallel sparse march.

    Unlike the synchronous :func:`march_svf_shadow_sparse` /
    :func:`march_wallheight23_sparse`, the workers are ALL spawned threads
    (the launching thread stays free to call :meth:`cancel`).

    Cancellation contract: ``cancel()`` bumps the shared generation
    counter and JOINS the workers; a worker notices the stale generation
    at its next row-run boundary and writes nothing further, so after
    ``cancel()`` returns no output write can occur and the planes are
    safe to recycle. The staleness window (work already in flight when
    ``cancel()`` lands) is bounded by one row-run.
    """

    def __init__(
        self,
        call: Callable[[int, int, Sequence[np.ndarray], int], None],
        plan: SparseMarchPlan,
        outs: list[np.ndarray],
        gen: np.ndarray,
        result_type,
    ):
        self._call = call
        self._plan = plan
        self._outs = outs
        self._gen = gen
        self._result_type = result_type
        self._expect = int(gen[0])
        self._threads: list[threading.Thread] = []
        self._started = False
        self._cancelled = False
        self._lock = threading.Lock()

    # -- construction -----------------------------------------------------

    @classmethod
    def svf_shadow(cls, table, a, vegdem, vegdem2, bush, runs, *, n_threads=2,
                   suffix_exit=True):
        buffers, runs32, plan, outs = _prepare(
            st.KERNEL_SVF_SHADOW,
            table, a, vegdem, vegdem2, bush, runs, n_threads, None,
            SERIAL_MIN_PAIRS,
        )
        dxs, dys, dzs = _table_columns(table, with_dzprev=False)
        allow_exit = _suffix_exit_allowed(buffers[0], suffix_exit)
        gen = np.zeros(1, dtype=np.int64)

        def call(lo, hi, outs, expect):
            _svf_shadow_runs_kernel(
                buffers[0], buffers[1], buffers[2], dxs, dys, dzs, runs32,
                lo, hi, gen, expect, outs[0], outs[1], outs[2], allow_exit,
            )

        return cls(call, plan, outs, gen, SvfSparseResult)

    @classmethod
    def wallheight23(cls, table, a, vegdem, vegdem2, bush, runs, *, n_threads=2,
                     suffix_exit=True):
        buffers, runs32, plan, outs = _prepare(
            st.KERNEL_WALLHEIGHT_23,
            table, a, vegdem, vegdem2, bush, runs, n_threads, None,
            SERIAL_MIN_PAIRS,
        )
        dxs, dys, dzs, dzprevs = _table_columns(table, with_dzprev=True)
        allow_exit = _suffix_exit_allowed(buffers[0], suffix_exit)
        gen = np.zeros(1, dtype=np.int64)

        def call(lo, hi, outs, expect):
            _wallheight23_runs_kernel(
                buffers[0], buffers[1], buffers[2], dxs, dys, dzs, dzprevs,
                runs32, lo, hi, gen, expect, outs[0], outs[1], outs[2],
                allow_exit,
            )

        return cls(call, plan, outs, gen, WallheightSparseResult)

    # -- lifecycle ---------------------------------------------------------

    @property
    def plan(self) -> SparseMarchPlan:
        return self._plan

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("launch already started")
            self._started = True
            chunks = self._plan.chunks
            if not chunks:
                return
            _verify_chunks(chunks, self._plan.n_runs)
            self._call(0, 0, self._outs, self._expect)  # compile warm-up
            for lo, hi in chunks:
                thread = threading.Thread(
                    target=self._worker, args=(lo, hi), daemon=True
                )
                self._threads.append(thread)
                thread.start()

    def _worker(self, lo: int, hi: int) -> None:
        self._call(lo, hi, self._outs, self._expect)

    def cancel(self) -> None:
        """Bump the generation and join the workers (bounded staleness)."""
        with self._lock:
            self._cancelled = True
            self._gen[0] += 1
        for thread in self._threads:
            thread.join()

    def result(self):
        """Join the workers and return the result tuple."""
        for thread in self._threads:
            thread.join()
        return self._result_type(self._outs[0], self._outs[1], self._outs[2], self._plan)

    def run_stale_worker_against(self, planes) -> None:
        """Audit hook: run ONE stale-generation worker call (this launch's
        original expected generation) against caller-provided planes —
        exactly what a late thread from a cancelled build would do. Must
        write nothing; only meaningful after :meth:`cancel`.
        """
        if not self._cancelled:
            raise RuntimeError(
                "stale-worker hook is only meaningful after cancel()"
            )
        self._call(0, self._plan.n_runs, list(planes), self._expect)


# ---------------------------------------------------------------------------
# Build-level runner (T05 work list -> per-patch routing)
# ---------------------------------------------------------------------------


def march_build(
    build,
    tables,
    a,
    vegdem,
    vegdem2,
    bush,
    *,
    n_threads: int = 1,
    min_pairs: int = SERIAL_MIN_PAIRS,
    suffix_exit: bool = True,
) -> dict[int, SvfSparseResult]:
    """Run one :class:`solweig_core.sparse_work.SparseWorkBuild`.

    Per patch: sparse march over the closure's row runs, EXCEPT patches
    the T05 builder flagged over ``ROUTE_DENSE_PAIR_FRACTION`` — those
    route through the T04 dense kernel unchanged (a sparse iteration over
    more than half the tile pays indirection per cell and loses to the
    dense sweep), and the result is flagged ``route='dense'`` with full
    planes. Empty-work builds return ``{}``. ``suffix_exit`` toggles the
    T19 post-onset suffix exit on every routed path (bit-invariant).
    """
    dense_patches = set(build.route_dense_patches)
    results: dict[int, SvfSparseResult] = {}
    for idx in sorted(build.row_runs):
        table = tables[idx]
        if idx in dense_patches:
            dense = march_svf_shadow(
                table, a, vegdem, vegdem2, bush, suffix_exit=suffix_exit
            )
            plan = SparseMarchPlan(
                pair_count=int(build.rows) * int(build.cols),
                n_runs=0,
                used_threads=1,
                chunks=(),
                route="dense",
            )
            results[idx] = SvfSparseResult(dense[0], dense[1], dense[2], plan)
        else:
            results[idx] = march_svf_shadow_sparse(
                table,
                a,
                vegdem,
                vegdem2,
                bush,
                build.row_runs[idx],
                n_threads=n_threads,
                min_pairs=min_pairs,
                suffix_exit=suffix_exit,
            )
    return results
