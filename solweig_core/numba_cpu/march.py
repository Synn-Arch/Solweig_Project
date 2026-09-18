# SPDX-License-Identifier: GPL-3.0-only
"""Dense serial Numba march kernel (DESIGN.ko.md 7.6 / 12.1, TASKS T04).

This module is the exact state-machine reference the GPU/C++ backends will
share: for every target cell it replays, IN EXECUTION ORDER, the primitive
graph of the original torch kernels — driven by the frozen T03
:class:`solweig_core.step_tables.StepTable` (whose executed step set already
includes the final overshoot step; termination is never re-derived here).

Kernel variants are NEVER conflated (DESIGN 7.1/7.2):

* ``svf_shadow`` — :func:`solweig_gpu.shadow.shadow`. Per step: running max
  ``f`` update, then ``sh`` update; vegetation max -> building suppression ->
  vb accumulator; first step (index 1) carries the vegetation condition
  (``firstvegdem``), the TARGET-LOCAL trunk gate (``vegdem2[t] > a[t]``) and
  the ``vbshvegsh.zero_()`` reset.
* ``wallheight_23`` — :func:`solweig_gpu.solweig.shadowingfunction_wallheight_23`.
  Step 0 is the (0, 0) self comparison; every step additionally consumes the
  PREVIOUS step's dz (``dzprev``, carried by the table's
  ``previous_dz_bits``) through the ``lastfabovea``/``lastgabovea`` planes.
  The wall/aspect tail of the original is march-independent and out of T04
  scope; only the three march outputs are produced.

Preserved semantics (every row of DESIGN 7.1's table):

* out-of-bounds source reads see the EXACT ZERO temporary value — never
  -inf, never ``0 - dz``;
* ``max`` follows torch elementwise semantics — NaN propagates from either
  operand, otherwise the larger, with the second operand on ties;
* the vb accumulator stays a float32 plane end-to-end: the one-step regime
  produces ``vbsh == 2.0`` outputs (``1 - (0 - 1)``) that no bool packing
  can represent, so no representation change is made (DESIGN 7.1 last row);
* final output applies the accumulator threshold, the veg subtraction and
  the inversions in the source's exact order with the source's casts.

Numeric discipline: the kernels are ``njit(cache=True, nogil=True,
fastmath=False)`` and SERIAL — the cell loop never dispatches in parallel;
threading is added only in T06 after independent cell ownership is
verified. All data-path arithmetic is
float32: values loaded from float32 arrays stay float32 and every literal
that touches data is an explicit ``np.float32`` (a bare Python float literal
would promote the expression to float64 and can flip knife-edge
comparisons). ``source - dz`` computed in float64 is exact, so a promoted
comparison CAN disagree with the float32-rounded one — this is why the
promotion discipline is part of the contract, not style.

T19 suffix exit (T27 verdict C6 — "f-floor post-onset suffix early exit",
PROVABLE): at the END of every step iteration (after the full step,
including the ``s == 0`` exception block) the kernels may break out of the
march when ``sh == 1 AND vegsh == 0`` — the post-onset-and-suppressed
FIXED POINT of the state machine. From that point on every remaining step
is exactly 0-contribution to the three outputs: ``f`` is
comparison-monotone (``sh`` stays 1), the post-max building suppression
re-zeroes ``vegsh`` every step, and ``vbsh = 0 + vbsh`` is the bit-exact
identity for the non-negative float32 integers vbsh accumulates. The
``vegsh == 0`` conjunct is LOAD-BEARING: the ``s == 0`` exception block
sets ``vegsh = 1`` AFTER suppression, so a naive ``sh == 1``-only escape
flips ``vbsh`` 1 -> 2 whenever onset lands on step 0 with the vegetation
condition true and ``count >= 2`` (the designated RED witness,
tests/ultrafast/test_t19_suffix_exit.py). The exit is OUTPUT-INVARIANT by
the fixed-point theorem, so ``suffix_exit=True`` and ``suffix_exit=False``
produce identical raw bits on every NaN-free scene; scenes whose a-plane
carries NaN disable the exit per batch (:func:`_suffix_exit_allowed`)
because ``_torch_maximum`` ABSORBS NaN and ``sh`` can fall back to 0 after
onset (the theorem's only exception).

Supported domain (T04 scope): ``bush.max() <= 0`` — the original's bush
blocks engage global (marched-extent) reduction predicates that break
per-target independence; the wrapper refuses such scenes instead of
silently approximating them. Sparse work lists, parallelism, SIMD and the
canonical SVF fold are NOT here (T05-T07).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from solweig_core import step_tables as st
from solweig_core.abi import ArrayView

__all__ = ["march_svf_shadow", "march_wallheight23"]


# ---------------------------------------------------------------------------
# torch elementwise max semantics (NaN propagates; second operand on tie)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True, fastmath=False)
def _torch_maximum(x, y):
    """torch.max(a, b) elementwise: (isnan(x) or x > y) ? x : y."""
    if x != x or x > y:
        return x
    return y


# ---------------------------------------------------------------------------
# svf_shadow: solweig_gpu/shadow.py:166-322, bush-free domain
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True, fastmath=False)
def _svf_shadow_kernel(
    a, vegdem, vegdem2, dxs, dys, dzs, sh_out, vegsh_out, vbsh_out, allow_exit
):
    rows = a.shape[0]
    cols = a.shape[1]
    count = dxs.shape[0]
    zero = np.float32(0.0)
    one = np.float32(1.0)
    big = np.float32(1000.0)
    for i in range(rows):
        for j in range(cols):
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


# ---------------------------------------------------------------------------
# wallheight_23: solweig_gpu/solweig.py:1079-1205 march section
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True, fastmath=False)
def _wallheight23_kernel(
    a, vegdem, vegdem2, dxs, dys, dzs, dzprevs, sh_out, vegsh_out, vbsh_out,
    allow_exit,
):
    rows = a.shape[0]
    cols = a.shape[1]
    count = dxs.shape[0]
    zero = np.float32(0.0)
    one = np.float32(1.0)
    four = np.float32(4.0)
    for i in range(rows):
        for j in range(cols):
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
# Python boundary: validation, guards, typed-buffer preparation
# ---------------------------------------------------------------------------


def _as_kernel_buffer(array, name: str, domain: str) -> np.ndarray:
    """Validate one scene plane at the ABI seam and return its buffer.

    Goes through :class:`solweig_core.abi.ArrayView` so dtype/shape/strides
    are carried as types; the kernel needs C-contiguous float32, and the
    single sanctioned boundary copy (:meth:`ArrayView.to_contiguous`) is used
    for strided input rather than silently indexing it as contiguous.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"{name} must be numpy.ndarray, got {type(array).__name__}"
        )
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise ValueError(
            f"{name} must be float32, got {array.dtype.str}"
        )
    view = ArrayView.from_numpy(array, logical_domain_id=domain)
    return view.to_contiguous().to_numpy()


def _check_table(table: st.StepTable, variant: str, shape: tuple[int, int]) -> None:
    if table.key.kernel_variant != variant:
        raise ValueError(
            f"table kernel_variant {table.key.kernel_variant!r} does not match "
            f"the {variant!r} march"
        )
    if (table.key.logical_rows, table.key.logical_cols) != tuple(shape):
        raise ValueError(
            "scene shape does not match the table's logical domain "
            f"{table.key.logical_rows}x{table.key.logical_cols} (the executed "
            "step set is only valid for that shape)"
        )


def _check_bush(bush: np.ndarray) -> None:
    if float(bush.max()) > 0.0:
        raise ValueError(
            "bush.max() > 0 is outside the T04 domain: the original bush "
            "blocks use marched-extent reduction predicates that break "
            "per-target independence (supported from the sparse tasks on)"
        )


def _table_columns(table: st.StepTable, *, with_dzprev: bool) -> list[np.ndarray]:
    """dx/dy as int32 and dz (plus dzprev) as exact-bit float32 arrays."""
    columns = [
        np.ascontiguousarray(table.dx, dtype=np.int32),
        np.ascontiguousarray(table.dy, dtype=np.int32),
        np.ascontiguousarray(table.dz_bits, dtype=np.uint32).view(np.float32),
    ]
    if with_dzprev:
        columns.append(
            np.ascontiguousarray(table.previous_dz_bits, dtype=np.uint32).view(
                np.float32
            )
        )
    return columns


def _suffix_exit_allowed(a_buffer: np.ndarray, suffix_exit: bool) -> bool:
    """T19/T27-C6 NaN guard: may the post-onset suffix exit engage?

    The fixed-point theorem holds on finite ``ta`` only: torch-max ABSORBS
    NaN (``x != x`` keeps returning x), so a NaN arriving through
    ``ta = a[si, sj] - dz`` makes ``f`` NaN from then on and ``sh`` falls
    back to 0 AFTER onset — the one way the suffix is not 0-contribution.
    NaN in vegdem/vegdem2 is already absorbed by the original semantics
    (the fab/gab comparisons are false, i.e. 0), and a NaN target ``a_t``
    never triggers onset at all (conservatively safe). One vectorized
    isnan pass over the a-plane per march batch (~1 MiB at 500x500 f32);
    any NaN disables the exit for the whole batch — the full march then
    runs and the bits are the original ones by construction.
    """
    if not suffix_exit:
        return False
    return not bool(np.isnan(a_buffer).any())


def march_svf_shadow(table: st.StepTable, a, vegdem, vegdem2, bush, *,
                     suffix_exit: bool = True):
    """Dense serial Numba replay of :func:`solweig_gpu.shadow.shadow`.

    Returns ``(sh, vegsh, vbshvegsh)`` as fresh float32 planes, bit-equal to
    the original kernel on the bush-free domain. ``suffix_exit`` toggles the
    T19 post-onset suffix exit; both settings produce identical raw bits
    (the exit only skips provably 0-contribution steps, and NaN-carrying
    a-planes disable it automatically — see :func:`_suffix_exit_allowed`).
    """
    buffers = [
        _as_kernel_buffer(x, name, f"march_svf_shadow:{table.trace_id}")
        for x, name in ((a, "a"), (vegdem, "vegdem"), (vegdem2, "vegdem2"), (bush, "bush"))
    ]
    _check_table(table, st.KERNEL_SVF_SHADOW, buffers[0].shape)
    _check_bush(buffers[3])
    shape = buffers[0].shape
    dxs, dys, dzs = _table_columns(table, with_dzprev=False)
    allow_exit = _suffix_exit_allowed(buffers[0], suffix_exit)
    sh_out = np.zeros(shape, dtype=np.float32)
    vegsh_out = np.zeros(shape, dtype=np.float32)
    vbsh_out = np.zeros(shape, dtype=np.float32)
    _svf_shadow_kernel(
        buffers[0], buffers[1], buffers[2], dxs, dys, dzs, sh_out, vegsh_out,
        vbsh_out, allow_exit,
    )
    return sh_out, vegsh_out, vbsh_out


def march_wallheight23(table: st.StepTable, a, vegdem, vegdem2, bush, *,
                       suffix_exit: bool = True):
    """Dense serial Numba replay of the ``shadowingfunction_wallheight_23``
    march.

    Returns ``(vegsh, sh, vbshvegsh)`` (the source's march-output order);
    the wall/aspect tail is march-independent and out of T04 scope.
    ``suffix_exit`` toggles the T19 post-onset suffix exit (bit-invariant;
    see :func:`march_svf_shadow`).
    """
    buffers = [
        _as_kernel_buffer(x, name, f"march_wallheight23:{table.trace_id}")
        for x, name in ((a, "a"), (vegdem, "vegdem"), (vegdem2, "vegdem2"), (bush, "bush"))
    ]
    _check_table(table, st.KERNEL_WALLHEIGHT_23, buffers[0].shape)
    _check_bush(buffers[3])
    shape = buffers[0].shape
    dxs, dys, dzs, dzprevs = _table_columns(table, with_dzprev=True)
    allow_exit = _suffix_exit_allowed(buffers[0], suffix_exit)
    sh_out = np.zeros(shape, dtype=np.float32)
    vegsh_out = np.zeros(shape, dtype=np.float32)
    vbsh_out = np.zeros(shape, dtype=np.float32)
    _wallheight23_kernel(
        buffers[0],
        buffers[1],
        buffers[2],
        dxs,
        dys,
        dzs,
        dzprevs,
        sh_out,
        vegsh_out,
        vbsh_out,
        allow_exit,
    )
    return vegsh_out, sh_out, vbsh_out
