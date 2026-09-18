# SPDX-License-Identifier: GPL-3.0-only
"""T05 gates: sparse closure + row-run/chunked-bitset work builder
(DESIGN.ko.md 7.3/7.4/7.5/7.7, TASKS T05).

Gate structure (TASKS.ko.md T05 + lead contract):

* **RED first** — this file was written and run BEFORE
  ``solweig_core/sparse_work.py`` existed; the recorded first failure is the
  collection ``ModuleNotFoundError`` (artifacts/t05/red_first_run.txt).
* **RED witnesses** (card list, each with a designated test): ``U C``
  removed from the closure (target-local trunk gate), old footprint missed,
  overlap-hidden canopy deletion, smaller-clamped amplitude transition,
  regime narrowing, OOB/corner cells, empty changed raster but changed
  global stop context (must NOT shortcut).
* **Closure completeness (site)** — site_500, >=3 real E1-style edit
  families (add near canopy, move, delete-overlap with amplitude change):
  every ``(cell, patch)`` whose full-domain T04 dense-march output bits
  differ between full-pre and full-post is inside the built closure.
* **Untouched-state equality (site)** — closure-only recompute (windowed
  T04 march on the R4 reach-expanded window) spliced into the pre state
  equals the full post recompute, raw bits, 0 mismatches.
* **Guard regressions** — the preserved R4 guards (bush, clamp-regime
  amplitude with the strict ``>`` and the amplitude-change conjunct,
  one-step regime, state identity), BOTH directions, pinned against the
  original torch-expressed predicates.
* **Representations** — sorted row runs AND 64x64 chunked bitsets with
  deterministic compaction, byte-identical repeated builds, memory
  estimates and the documented dense-routing ceiling.
* **Purity** — no torch import, no fastmath/parallel in sparse_work.py.
* **Mutation gate** — >=3 source-level mutations of sparse_work.py, each
  killed by a designated test; applied by script and recorded in
  artifacts/t05/mutations/, NOT in this file.

Site gates are skipped when the site cache is absent. The full-153-patch
family is F3 (delete-overlap); F1/F2 use the altitude subset {6, 42, 78,
90} (runtime-bounded, documented in the report).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from trace_exporter import (  # noqa: E402
    capture_amplitude_policy,
    capture_trace,
    executed_amplitude_for_patch,
)
from solweig_core import sparse_work as sw  # noqa: E402
from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu.march import march_svf_shadow  # noqa: E402

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
ARTIFACT_DIR = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t05"
)
F32 = np.dtype("<f4")
U32 = np.dtype("<u4")

#: Synthetic march grid shared by the closure witnesses (non-square on
#: purpose: the T03/T04 row/col-swap lesson).
SYN_ROWS, SYN_COLS = 40, 80
#: (azimuth, altitude) sweep for synthetic closures: mid sun, diagonal
#: branch boundary, low sun (long march), one-step witness angle, zenith.
SYN_ANGLES = [
    (37.0, 35.0),
    (225.0, 35.0),
    (90.0, 6.0),
    (0.0, 78.0),
    (45.0, 90.0),
]


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=F32).view(U32)


def synth_planes(rows, cols, seed=7, negative_dem=False):
    """Deterministic (a, dem, canopy) float32 numpy planes."""
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    if negative_dem:
        dem -= np.float32(12.0)
    a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    a[3:9, 4:12] += np.float32(18.0)
    canopy = rng.uniform(0.0, 6.0, (rows, cols)).astype(np.float32)
    canopy[canopy < np.float32(3.0)] = np.float32(0.0)
    return a, dem, canopy


def paint_disc(canopy, r0, c0, radius, height):
    """New canopy with a constant-height disc max-combined in (the
    TreeLayer rasterization combine rule, trees.py:342 ``np.maximum``)."""
    rows, cols = canopy.shape
    out = canopy.copy()
    rr, cc = np.ogrid[:rows, :cols]
    disc = (rr - r0) ** 2 + (cc - c0) ** 2 <= radius * radius
    out[disc] = np.maximum(out[disc], np.float32(height))
    return out


def synth_table(az, alt, amp, rows=SYN_ROWS, cols=SYN_COLS, scale=0.5):
    trace = capture_trace(
        st.KERNEL_SVF_SHADOW,
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


def synth_tables(angles, amp, **kw):
    return {
        i: synth_table(az, alt, amp, **kw)
        for i, (az, alt) in enumerate(angles)
    }


def synth_inputs(a, dem, canopy):
    return sw.compose_march_inputs(a, dem, canopy)


def full_march(table, inputs):
    """T04 dense kernel over the full grid (the verified reference)."""
    return march_svf_shadow(
        table, inputs.a, inputs.vegdsm, inputs.vegdsm2, inputs.bush
    )


def changed_cells(plane_pre, plane_post):
    return bits(plane_pre) != bits(plane_post)


def assert_closure_complete(pre, post, tables_pre, tables_post, masks, label):
    """Gate-2 core: every changed (cell, patch) is inside the closure.

    Returns per-patch change counts for sensitivity reporting.
    """
    change_counts = {}
    for idx in sorted(tables_pre):
        out_pre = full_march(tables_pre[idx], pre)
        out_post = full_march(tables_post[idx], post)
        D = masks.get(idx)
        if D is None:
            D = np.zeros((pre.a.shape[0], pre.a.shape[1]), dtype=bool)
        count = 0
        for name, p0, p1 in zip(("sh", "vegsh", "vbsh"), out_pre, out_post):
            changed = changed_cells(p0, p1)
            count += int(changed.sum())
            missed = changed & ~D
            assert not missed.any(), (
                f"{label} patch {idx}: {int(missed.sum())} changed cells "
                f"outside the closure on plane {name}, first at "
                f"{tuple(int(v) for v in np.argwhere(missed)[0])}"
            )
        change_counts[idx] = count
    return change_counts


def slow_corridor_reference(C, offsets):
    """Independent per-cell corridor oracle (explicit bounds checks)."""
    rows, cols = C.shape
    out = np.zeros((rows, cols), dtype=bool)
    c_cells = set(map(tuple, np.argwhere(C)))
    for i in range(rows):
        for j in range(cols):
            for dx, dy in offsets:
                si, sj = i + int(dx), j + int(dy)
                if 0 <= si < rows and 0 <= sj < cols and (si, sj) in c_cells:
                    out[i, j] = True
                    break
    return out


# ---------------------------------------------------------------------------
# Composed march inputs: torch-free mirror of the proven compose
# ---------------------------------------------------------------------------


class TestComposeMirror:
    def test_synthetic_compose_bit_equal_original(self):
        from solweig_gpu.incremental.solver import _compose_scene_from_canopy

        for seed, neg in ((7, False), (11, True)):
            a, dem, canopy = synth_planes(40, 80, seed=seed, negative_dem=neg)
            mine = sw.compose_march_inputs(a, dem, canopy)
            theirs = _compose_scene_from_canopy(
                _TorchSiteStub(a, dem), canopy
            )
            for name in ("a", "vegdem", "vegdem2", "bush", "vegdsm", "vegdsm2"):
                assert np.array_equal(
                    bits(getattr(mine, name)),
                    bits(getattr(theirs, name).numpy()),
                ), f"compose plane {name} diverged (seed={seed})"
            assert np.float32(mine.amaxvalue).view(U32) == np.float32(
                float(theirs.amaxvalue)
            ).view(U32)

    def test_compose_signed_zero_and_nan_policy(self):
        """-0.0 canopy survives the ``< 0`` clamp and NaN propagates — the
        mirror must reproduce the original's bits exactly on both."""
        from solweig_gpu.incremental.solver import _compose_scene_from_canopy

        a, dem, canopy = synth_planes(24, 32, seed=3)
        canopy = canopy.copy()
        canopy[5, 5] = np.float32(-0.0)
        canopy[6, 6] = np.float32(np.nan)
        mine = sw.compose_march_inputs(a, dem, canopy)
        theirs = _compose_scene_from_canopy(_TorchSiteStub(a, dem), canopy)
        for name in ("vegdem", "vegdem2", "bush", "vegdsm", "vegdsm2"):
            assert np.array_equal(
                bits(getattr(mine, name)), bits(getattr(theirs, name).numpy())
            ), name
        # the documented degenerate facts themselves (not just equality):
        # IEEE (-0.0) + (+0.0 dem) rounds to +0.0 (the zero sign is lost by
        # addition — exactly why value-equality policy on C is safe), and a
        # NaN canopy propagates into vegdsm.
        assert not np.signbit(mine.vegdem[5, 5])
        assert mine.vegdem[5, 5] == 0.0
        assert np.isnan(mine.vegdsm[6, 6])

    def test_compose_rejects_bad_shapes_and_dtypes(self):
        a, dem, canopy = synth_planes(16, 16)
        with pytest.raises(ValueError, match="shape"):
            sw.compose_march_inputs(a, dem, canopy[:8])
        with pytest.raises(ValueError, match="float32"):
            sw.compose_march_inputs(
                a.astype(np.float64), dem, canopy
            )


class _TorchSiteStub:
    """Minimal stand-in with the two cache members compose reads."""

    def __init__(self, a, dem):
        self.building_dsm = a
        self.dem = dem


# ---------------------------------------------------------------------------
# C: composed-surface source diff (DESIGN 7.3)
# ---------------------------------------------------------------------------


class TestComposedDiff:
    def _scene(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS)
        return a, dem, canopy

    def test_delete_footprint_exact(self):
        """OLD footprint witness: deleting a canopy blob changes exactly the
        blob's cells (where the blob was the composed winner)."""
        a, dem, canopy = self._scene()
        canopy_edit = paint_disc(canopy, 20, 50, 6, 12.0)
        pre = synth_inputs(a, dem, canopy_edit)
        post = synth_inputs(a, dem, canopy)  # deletion
        C = sw.composed_change_mask(pre, post)
        disc = np.zeros((SYN_ROWS, SYN_COLS), dtype=bool)
        rr, cc = np.ogrid[:SYN_ROWS, :SYN_COLS]
        disc = (rr - 20) ** 2 + (cc - 50) ** 2 <= 36
        winner = canopy_edit > canopy  # cells the deleted blob dominated
        expected = disc & winner
        assert np.array_equal(C, expected)
        assert C.any()

    def test_overlap_hidden_reveal_exact(self):
        """Overlap-hidden canopy deletion witness: deleting the SHORTER
        disc A changes only cells A dominated; cells where the surviving
        TALLER canopy B wins are composition-unchanged and MUST NOT be in
        C (the deletion is hidden by the overlap)."""
        a, dem, canopy = self._scene()
        canopy_base = paint_disc(canopy, 20, 40, 6, 12.0)  # tall B stays
        pre_canopy = paint_disc(canopy_base, 20, 46, 6, 5.0)  # short A
        post_canopy = canopy_base  # A deleted, B remains
        pre = synth_inputs(a, dem, pre_canopy)
        post = synth_inputs(a, dem, post_canopy)
        C = sw.composed_change_mask(pre, post)
        rr, cc = np.ogrid[:SYN_ROWS, :SYN_COLS]
        disc_a = (rr - 20) ** 2 + (cc - 46) ** 2 <= 36
        dominated = pre_canopy > post_canopy  # A strictly dominated here
        expected = disc_a & dominated
        assert np.array_equal(C, expected), (
            "overlap-hidden deletion: C must equal the A-dominated cells"
        )
        hidden = disc_a & ~dominated
        assert hidden.any(), "fixture has no overlap-hidden cells"
        assert not (C & hidden).any()

    def test_equality_policy_signed_zero_nan(self):
        """Stated equality: IEEE float32 VALUE equality. A +0.0 -> -0.0
        canopy flip is UNCHANGED (addition collapses both to +0.0 in every
        composed plane anyway); a NaN canopy cell is CHANGED (conservative
        superset)."""
        a, dem, canopy = synth_planes(24, 32, seed=5)
        canopy = canopy.copy()
        canopy[4, 4] = np.float32(0.0)
        pre = synth_inputs(a, dem, canopy)
        canopy_neg = canopy.copy()
        canopy_neg[4, 4] = np.float32(-0.0)
        post = synth_inputs(a, dem, canopy_neg)
        assert not sw.composed_change_mask(pre, post).any()
        canopy_nan = canopy.copy()
        canopy_nan[8, 8] = np.float32(np.nan)
        post_nan = synth_inputs(a, dem, canopy_nan)
        C = sw.composed_change_mask(pre, post_nan)
        assert C[8, 8]
        assert int(C.sum()) >= 1

    def test_vegdsm_only_change_witness(self):
        """Old-footprint defect witness: a canopy change visible ONLY in
        vegdsm. At a = 256 (f32 ulp 2^-15), h = 3e-5 m sits ABOVE the
        half-ulp (256 + h rounds up, so vegdsm changes) while h/4 sits
        BELOW it (256 + h/4 rounds back to a, so vegdsm2 is zeroed on BOTH
        sides) — dropping the vegdsm term from the diff misses the cell."""
        rows, cols = 12, 16
        a = np.full((rows, cols), np.float32(256.0), dtype=np.float32)
        dem = np.zeros((rows, cols), dtype=np.float32)
        canopy_pre = np.zeros((rows, cols), dtype=np.float32)
        canopy_post = canopy_pre.copy()
        h = np.float32(3e-5)
        t = (6, 8)
        canopy_post[t] = h
        pre = synth_inputs(a, dem, canopy_pre)
        post = synth_inputs(a, dem, canopy_post)
        # fixture facts: vegdsm changed, vegdsm2/bush/a unchanged at t
        assert bits(pre.vegdsm)[t] != bits(post.vegdsm)[t]
        assert bits(pre.vegdsm2)[t] == bits(post.vegdsm2)[t]
        C = sw.composed_change_mask(pre, post)
        assert int(C.sum()) == 1 and C[t]

    def test_candidate_restrict_and_verification(self):
        """``restrict`` narrows the diff (the C_candidate contract) and the
        debug recompute refuses a candidate that misses real changes."""
        a, dem, canopy = self._scene()
        canopy_post = paint_disc(canopy, 20, 50, 6, 12.0)
        canopy_post = paint_disc(canopy_post, 30, 10, 5, 9.0)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, canopy_post)
        full = sw.composed_change_mask(pre, post)
        near = np.zeros((SYN_ROWS, SYN_COLS), dtype=bool)
        near[10:40, 30:80] = True  # covers the first disc only
        restricted = sw.composed_change_mask(pre, post, restrict=near)
        assert np.array_equal(restricted, full & near)
        # a candidate that misses the second disc must be refused loudly
        with pytest.raises(ValueError, match="candidate"):
            sw.composed_change_mask(
                pre, post, restrict=near, verify_restrict=True
            )


# ---------------------------------------------------------------------------
# Corridor: OOB/corner exactness
# ---------------------------------------------------------------------------


class TestCorridor:
    def test_matches_independent_reference(self):
        """OOB/corner witness: clipped shifts only; no negative-index
        wraparound, offsets larger than the grid, corners on tiny masks."""
        rng = np.random.default_rng(23)
        for trial in range(12):
            rows = int(rng.integers(3, 14))
            cols = int(rng.integers(3, 17))
            C = rng.random((rows, cols)) < 0.25
            if trial % 3 == 0 and rows and cols:
                C[0, 0] = True
                C[-1, -1] = True
            offsets = {
                (
                    int(rng.integers(-cols - 2, cols + 3)),
                    int(rng.integers(-rows - 2, rows + 3)),
                )
                for _ in range(7)
            } | {(0, 1), (-1, 0), (rows, cols), (-rows, -cols)}
            got = sw.corridor_mask(C, sorted(offsets))
            want = slow_corridor_reference(C, offsets)
            assert np.array_equal(got, want), f"trial {trial}"

    def test_matches_r4_corridor_mask(self):
        """Pin against the proven R4 implementation (same indexing)."""
        from solweig_gpu.incremental.veg_svf_state import (
            corridor_mask as r4_corridor,
        )

        rng = np.random.default_rng(31)
        for _ in range(6):
            rows, cols = 20, 33
            C = rng.random((rows, cols)) < 0.2
            offsets = [
                (int(v), int(w))
                for v, w in rng.integers(-25, 26, size=(9, 2))
            ]
            mine = sw.corridor_mask(C, offsets)
            theirs = r4_corridor(C, offsets)
            assert np.array_equal(mine, theirs)

    def test_corner_cells_no_wraparound(self):
        C = np.zeros((6, 9), dtype=bool)
        C[0, 0] = True
        got = sw.corridor_mask(C, [(1, 0), (0, 1), (-1, 0), (0, -1)])
        want = np.zeros((6, 9), dtype=bool)
        want[0, 1] = True  # reads (0,0) at (0,0)+(0,1)... t+d == (0,0)
        want[1, 0] = True
        assert np.array_equal(got, want)
        assert not got[-1, -1]  # no wraparound read


# ---------------------------------------------------------------------------
# Synthetic closure completeness via the T04 dense kernel
# ---------------------------------------------------------------------------


class TestClosureSynthetic:
    def _tables(self, pre, post):
        amp_pre = float(sw.effective_amplitude(pre))
        amp_post = float(sw.effective_amplitude(post))
        tables_pre = synth_tables(SYN_ANGLES, amp_pre)
        tables_post = synth_tables(SYN_ANGLES, amp_post)
        return tables_pre, tables_post, amp_pre, amp_post

    def test_trunk_gate_target_local_witness(self):
        """ ``U C`` witness: a one-cell trunk-zone edit changes the edited
        cell's own output through the TARGET-LOCAL trunk gate; the cell is
        in NO corridor (offsets never include (0,0)) — only ``C U corridor``
        covers it. Scene is deliberately UNCLAMPED (elevated flat DEM) so
        the clamp guard stays silent and the closure itself is load-bearing."""
        rng = np.random.default_rng(21)
        dem = np.full((SYN_ROWS, SYN_COLS), np.float32(50.0), dtype=np.float32)
        a = dem + rng.uniform(0.0, 1.0, (SYN_ROWS, SYN_COLS)).astype(np.float32)
        canopy = np.zeros((SYN_ROWS, SYN_COLS), dtype=np.float32)
        canopy = paint_disc(canopy, 5, 5, 3, 15.0)  # pre amplitude multi-step
        t = (10, 30)
        canopy_post = canopy.copy()
        canopy_post[t] = np.float32(40.0)  # trunk gate flips at t only
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, canopy_post)
        tables_pre, tables_post, amp_pre, amp_post = self._tables(pre, post)
        assert amp_pre > 10.0 and amp_post > amp_pre, (amp_pre, amp_post)
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        C = build.C
        assert C.sum() == 1 and C[t]
        for idx, table in tables_pre.items():
            corridor = sw.corridor_mask(
                C, sorted({(int(x), int(y)) for x, y in zip(table.dx, table.dy)})
            )
            assert not corridor[t], (
                "witness fixture broken: t landed in a corridor"
            )
        masks = build.masks()
        counts = assert_closure_complete(
            pre, post, tables_pre, tables_post, masks, "trunk-gate"
        )
        assert any(v > 0 for v in counts.values()), (
            "witness fixture too weak: no output changed at all"
        )
        assert any(masks[i][t] for i in masks), "closure misses the C cell"

    def test_delete_blob_closure_complete(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS)
        canopy_pre = paint_disc(canopy, 20, 50, 6, 12.0)
        pre = synth_inputs(a, dem, canopy_pre)
        post = synth_inputs(a, dem, canopy)
        tables_pre, tables_post, _ap, _aq = self._tables(pre, post)
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        counts = assert_closure_complete(
            pre, post, tables_pre, tables_post, build.masks(), "delete-blob"
        )
        assert sum(counts.values()) > 0

    def test_overlap_hidden_deletion_closure_complete(self):
        """Shorter A deleted over taller surviving B: A-only cells change,
        the overlap is composition-hidden (B already wins there)."""
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS)
        base = paint_disc(canopy, 20, 40, 6, 12.0)
        pre_canopy = paint_disc(base, 20, 46, 6, 5.0)
        pre = synth_inputs(a, dem, pre_canopy)
        post = synth_inputs(a, dem, base)
        tables_pre, tables_post, _ap, _aq = self._tables(pre, post)
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        counts = assert_closure_complete(
            pre, post, tables_pre, tables_post, build.masks(), "overlap-del"
        )
        assert sum(counts.values()) > 0

    def test_amplitude_growth_union_offsets(self):
        """Pre/post tables differ (effective amplitude grew): the closure
        must union BOTH offset sets — the longer post march reads C at
        offsets the pre march never executed. Unclamped growth keeps the
        clamp guard silent, so the closure itself is load-bearing."""
        rows, cols = 48, 64
        rng = np.random.default_rng(9)
        dem = np.full((rows, cols), np.float32(50.0), dtype=np.float32)
        dem[:, cols // 2 :] += np.float32(30.0)  # high ground east
        a = dem + rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
        canopy = np.zeros((rows, cols), dtype=np.float32)
        canopy = paint_disc(canopy, 24, 16, 5, 20.0)  # on LOW ground
        canopy_pre = canopy
        canopy_post = paint_disc(canopy_pre, 24, 16, 5, 40.0)  # taller
        pre = synth_inputs(a, dem, canopy_pre)
        post = synth_inputs(a, dem, canopy_post)
        amp_pre = float(sw.effective_amplitude(pre))
        amp_post = float(sw.effective_amplitude(post))
        assert amp_post > amp_pre, "fixture does not grow the amplitude"
        tables_pre = synth_tables(SYN_ANGLES, amp_pre, rows=rows, cols=cols)
        tables_post = synth_tables(SYN_ANGLES, amp_post, rows=rows, cols=cols)
        assert any(
            tables_pre[i].count != tables_post[i].count for i in tables_pre
        ), "fixture does not change the executed step sets"
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        counts = assert_closure_complete(
            pre, post, tables_pre, tables_post, build.masks(), "amp-growth"
        )
        assert sum(counts.values()) > 0

    def test_amplitude_shrink_pre_only_offsets(self):
        """Amplitude SHRINK witness for the pre-offsets corridor term.

        The executed step sets nest in the amplitude (a shorter march is a
        prefix), so when the amplitude shrinks across an edit the PRE table
        has offsets the POST table lacks: a target whose only C read goes
        through a pre-only step is covered by NEITHER C nor the post
        corridor. The builder-level set algebra is asserted directly
        (physics-independent, deterministic): the pre-only coverage set

            T = {t = c - d_pre : c in C, d_pre in pre-only, in bounds}
                \\ C \\ {t : t + d_post in C for some post offset}

        must be non-empty for some patch, and every t in T must be in that
        patch's closure (and in both representations). Dropping the
        ``corridor_mask(C, offsets_pre)`` term (lead mutation M5) removes
        exactly these cells. The growth fixture cannot cover this: there
        post offsets are a superset, making the pre term redundant; and the
        site F3 shrink family is value-inert for this term (no changed
        output lands in the pre-only region), so the algebra witness is the
        designated killer. The full-march completeness check also runs —
        under the mutation it may or may not trip depending on where the
        scene's changed outputs land; membership in T is what must fail.
        """
        rows, cols = 48, 64
        rng = np.random.default_rng(29)
        dem = np.full((rows, cols), np.float32(50.0), dtype=np.float32)
        dem[:, cols // 2 :] += np.float32(30.0)  # high ground east
        a = dem + rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
        canopy_pre = paint_disc(
            np.zeros((rows, cols), dtype=np.float32), 24, 16, 5, 40.0
        )
        canopy_post = paint_disc(
            np.zeros((rows, cols), dtype=np.float32), 24, 16, 5, 20.0
        )
        pre = synth_inputs(a, dem, canopy_pre)
        post = synth_inputs(a, dem, canopy_post)
        amp_pre = float(sw.effective_amplitude(pre))
        amp_post = float(sw.effective_amplitude(post))
        assert amp_post < amp_pre, "fixture does not shrink the amplitude"
        tables_pre = synth_tables(SYN_ANGLES, amp_pre, rows=rows, cols=cols)
        tables_post = synth_tables(SYN_ANGLES, amp_post, rows=rows, cols=cols)
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        C_cells = set(map(tuple, np.argwhere(build.C)))
        assert C_cells, "fixture changed no composed cell"
        chosen = None
        for idx in sorted(tables_pre):
            offs_pre = set(sw.table_offsets(tables_pre[idx]))
            offs_post = set(sw.table_offsets(tables_post[idx]))
            pre_only = offs_pre - offs_post
            if not pre_only:
                continue
            assert tables_pre[idx].count > tables_post[idx].count
            candidates = {
                (int(r - dx), int(c - dy))
                for r, c in C_cells
                for dx, dy in pre_only
                if 0 <= r - dx < rows and 0 <= c - dy < cols
            }
            covered_post = set(
                map(
                    tuple,
                    np.argwhere(sw.corridor_mask(build.C, sorted(offs_post))),
                )
            )
            witness = candidates - covered_post - C_cells
            if witness:
                chosen = (idx, witness)
                break
        assert chosen is not None, (
            "fixture too weak: no patch has a pre-only corridor cell "
            "uncovered by C or the post corridor"
        )
        idx, witness = chosen
        D = build.masks()[idx]
        for t in witness:
            assert D[t], (
                f"patch {idx}: pre-only corridor cell {t} missing from the "
                "closure (pre-offsets term dropped?)"
            )
        assert np.array_equal(
            sw.mask_from_row_runs(build.row_runs[idx], (rows, cols)), D
        )
        assert np.array_equal(sw.mask_from_chunk_bitset(build.bitsets[idx]), D)
        # physics layer (bonus; value-inertness documented above)
        assert_closure_complete(
            pre, post, tables_pre, tables_post, build.masks(), "amp-shrink"
        )

    def test_closure_always_contains_C(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS)
        canopy_post = paint_disc(canopy, 20, 50, 5, 12.0)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, canopy_post)
        tables = synth_tables(SYN_ANGLES, 20.0)
        build = sw.build_sparse_work(
            pre, post, dict(tables), dict(tables), keep_masks=True
        )
        masks = build.masks()
        for idx, D in masks.items():
            assert (D & build.C).sum() == build.C.sum(), (
                f"patch {idx}: closure does not contain all of C"
            )


# ---------------------------------------------------------------------------
# Empty-work shortcut (DESIGN 7.4: only with identical stop context)
# ---------------------------------------------------------------------------


class TestEmptyWorkShortcut:
    def _scenes(self):
        a, dem, canopy = synth_planes(24, 32, seed=13)
        inputs = synth_inputs(a, dem, canopy)
        return inputs, synth_inputs(a, dem, canopy.copy())

    def test_identical_scenes_and_tables_shortcut(self):
        pre, post = self._scenes()
        tables = synth_tables(SYN_ANGLES, 20.0, rows=24, cols=32)
        build = sw.build_sparse_work(
            pre, post, dict(tables), dict(tables), keep_masks=True
        )
        assert build.empty_work
        assert build.pair_count == 0
        assert not build.masks()

    def test_empty_raster_but_stop_context_changed(self):
        """RED witness: changed raster is EMPTY but the executed tables
        differ (global stop context changed) — must NOT shortcut; the
        honest work list is the full tile, flagged."""
        pre, post = self._scenes()
        tables_pre = synth_tables(SYN_ANGLES, 10.0, rows=24, cols=32)
        tables_post = synth_tables(SYN_ANGLES, 25.0, rows=24, cols=32)
        assert any(
            tables_pre[i].count != tables_post[i].count for i in tables_pre
        )
        build = sw.build_sparse_work(
            pre, post, tables_pre, tables_post, keep_masks=True
        )
        assert not build.empty_work
        assert build.table_changed_with_empty_C
        masks = build.masks()
        for idx, D in masks.items():
            assert D.all(), f"patch {idx} must be the full tile"


# ---------------------------------------------------------------------------
# Preserved guards (DESIGN 7.4, both directions, pinned to R4)
# ---------------------------------------------------------------------------


def _guard_scene(a_max, vegdsm_max, vegdem_max, a_min=0.0, shape=(8, 8)):
    """Direct MarchInputs for the clamp guard (raw fields, like the guard
    reads them); vegdsm2/bush/canopy are inert for this predicate."""
    zeros = lambda: np.zeros(shape, dtype=np.float32)  # noqa: E731
    a = zeros()
    a[0, 0] = np.float32(a_max)
    a[1, 1] = np.float32(a_min)
    vegdsm = zeros()
    vegdsm[2, 2] = np.float32(vegdsm_max)
    vegdem = zeros()
    vegdem[3, 3] = np.float32(vegdem_max)
    dem = zeros()
    canopy = zeros()
    return sw.MarchInputs(
        a=a,
        canopy=canopy,
        dem=dem,
        vegdem=vegdem,
        vegdem2=zeros(),
        bush=zeros(),
        vegdsm=vegdsm,
        vegdsm2=zeros(),
        amaxvalue=np.float32(max(a_max, vegdem_max)),
    )


def _torch_scene_of(inputs):
    from solweig_gpu.incremental.solver import FullSceneTensors

    t = lambda x: torch.from_numpy(  # noqa: E731
        np.ascontiguousarray(x, dtype=np.float32).copy()
    )
    return FullSceneTensors(
        a=t(inputs.a),
        canopy=t(inputs.canopy),
        dem=t(inputs.dem),
        vegdem=t(inputs.vegdem),
        vegdem2=t(inputs.vegdem2),
        bush=t(inputs.bush),
        vegdsm=t(inputs.vegdsm),
        vegdsm2=t(inputs.vegdsm2),
        amaxvalue=torch.tensor(np.float32(inputs.amaxvalue)),
    )


class TestGuards:
    def test_bush_nonzero_refuses_pre_and_post(self):
        pre = _guard_scene(2.0, 5.0, 4.0)
        post = _guard_scene(2.0, 6.0, 4.0)
        assert sw.bush_guard(pre, post) is None
        for which in ("pre", "post"):
            bushy = post if which == "post" else pre
            bushy = sw.MarchInputs(
                **{
                    **vars(bushy),
                    "bush": bushy.bush + np.float32(2.0),
                }
            )
            scene_pre = bushy if which == "pre" else pre
            scene_post = bushy if which == "post" else post
            reason = sw.bush_guard(scene_pre, scene_post)
            assert reason is not None and "bush" in reason, which

    def test_clamp_guard_case_c_refuses(self):
        """Smaller-clamped amplitude transition (Case C): refuse."""
        small = _guard_scene(0.0, 10.0, 6.0)   # bound 10 > amax 6 (clamped)
        large = _guard_scene(0.0, 20.0, 20.0)  # eff 20
        reason = sw.clamped_amplitude_change_guard(small, large)
        assert reason is not None and "clamp" in reason
        reason = sw.clamped_amplitude_change_guard(large, small)
        assert reason is not None  # symmetric: smaller scene is `small`

    def test_clamp_guard_steady_clamp_does_not_refuse(self):
        """Amplitude-constant transition on a clamped scene: the
        amplitude-change conjunct keeps it on the sparse path."""
        scene = _guard_scene(0.0, 10.0, 6.0)
        assert sw.clamped_amplitude_change_guard(scene, scene) is None

    def test_clamp_guard_unclamped_change_does_not_refuse(self):
        """Case B: amplitude changed, smaller scene unclamped -> None."""
        s1 = _guard_scene(0.0, 5.0, 30.0)   # bound 5 <= amax 30
        s2 = _guard_scene(0.0, 10.0, 30.0)  # bound 10 <= amax 30
        assert sw.clamped_amplitude_change_guard(s1, s2) is None
        assert sw.clamped_amplitude_change_guard(s2, s1) is None

    def test_clamp_guard_strict_bound_equality_boundary(self):
        """bound == scene_amaxvalue on the smaller scene: strict ``>`` keeps
        it sparse; one nextafter up must refuse."""
        exact = _guard_scene(0.0, 6.0, 6.0)      # bound 6 == amax 6
        larger = _guard_scene(0.0, 9.0, 9.0)     # eff 9 > 6
        assert sw.clamped_amplitude_change_guard(exact, larger) is None
        assert sw.clamped_amplitude_change_guard(larger, exact) is None
        nudged = _guard_scene(0.0, float(np.nextafter(np.float32(6.0), np.float32(np.inf))), 6.0)
        assert sw.clamped_amplitude_change_guard(nudged, larger) is not None

    def test_clamp_guard_pinned_to_original(self):
        """Decision + amplitude/bound bits equal the torch-expressed R4
        predicate on mirrored scenes."""
        from solweig_gpu.incremental.veg_svf_state import (
            clamped_amplitude_change_reason,
            _scene_amplitude,
        )

        pairs = [
            (
                _guard_scene(0.0, 10.0, 6.0),
                _guard_scene(0.0, 20.0, 20.0),
            ),
            (
                _guard_scene(0.0, 5.0, 30.0),
                _guard_scene(0.0, 10.0, 30.0),
            ),
            (
                _guard_scene(1.5, 7.25, 3.5),
                _guard_scene(1.5, 7.25, 3.5),
            ),
        ]
        for pre, post in pairs:
            mine = sw.clamped_amplitude_change_guard(pre, post)
            theirs = clamped_amplitude_change_reason(
                _torch_scene_of(pre), _torch_scene_of(post)
            )
            assert (mine is None) == (theirs is None)
            for inputs in (pre, post):
                bits_mine = np.float32(
                    sw.effective_amplitude(inputs)
                ).view(U32)
                bits_theirs = np.float32(
                    _scene_amplitude(_torch_scene_of(inputs))
                ).view(U32)
                assert bits_mine == bits_theirs

    def test_regime_narrowing_refuses(self):
        """Multi-step -> one-step refuses (real tables: 78 deg at scale 0.25
        is one-step iff dz_1 > amplitude)."""
        tables_pre = synth_tables([(0.0, 78.0)], 20.0, rows=16, cols=16)
        tables_post = synth_tables([(0.0, 78.0)], 4.0, rows=16, cols=16)
        assert tables_pre[0].count > 1
        assert tables_post[0].count == 1
        reason = sw.one_step_regime_guard(tables_pre, tables_post)
        assert reason is not None and "one-step" in reason

    def test_regime_narrowing_pinned_to_original(self):
        from solweig_gpu.incremental.veg_svf_state import (
            _regime_transition_reason,
        )

        tables_pre = synth_tables([(0.0, 78.0)], 20.0, rows=16, cols=16)
        tables_post = synth_tables([(0.0, 78.0)], 4.0, rows=16, cols=16)
        amp_pre = float(
            st.u32_bits_to_f32(
                st.bits_hex_to_u32(tables_pre[0].key.executed_amplitude_bits)
            )
        )
        amp_post = float(
            st.u32_bits_to_f32(
                st.bits_hex_to_u32(tables_post[0].key.executed_amplitude_bits)
            )
        )
        assert (
            _regime_transition_reason(amp_pre, amp_post, 0.25) is not None
        ) == (
            sw.one_step_regime_guard(tables_pre, tables_post) is not None
        )

    def test_regime_growth_and_steady_do_not_refuse(self):
        tables = synth_tables([(0.0, 78.0)], 20.0, rows=16, cols=16)
        assert sw.one_step_regime_guard(dict(tables), dict(tables)) is None
        wider = synth_tables([(37.0, 35.0)], 40.0, rows=16, cols=16)
        assert wider[0].count > tables[0].count
        assert sw.one_step_regime_guard(tables, wider) is None

    def test_regime_any_one_step_state_refuses(self):
        """Representability: a one-step patch on EITHER scene refuses (the
        packed-bit state cannot hold vbsh == 2.0). Includes one->multi
        growth, where the R4 narrowing predicate alone stays silent — the
        baseline fence is what refuses it there."""
        one = synth_tables([(0.0, 78.0)], 4.0, rows=16, cols=16)
        multi = synth_tables([(0.0, 78.0)], 20.0, rows=16, cols=16)
        assert one[0].count == 1 and multi[0].count > 1
        assert sw.one_step_regime_guard(one, multi) is not None
        assert sw.one_step_regime_guard(multi, one) is not None
        assert sw.one_step_regime_guard(dict(one), dict(one)) is not None

    def test_state_identity_guard(self):
        good = {
            "schema_version": 1,
            "site_id": "s",
            "tile_key": "t",
            "cache_manifest_sha256": "x" * 64,
            "patch_geometry_id": "p" * 64,
            "rows": 10,
            "cols": 20,
            "chunk_checksums_sha256": "c" * 64,
        }
        assert sw.state_identity_guard(good, dict(good)) is None
        for field in ("site_id", "tile_key", "cache_manifest_sha256",
                      "patch_geometry_id", "chunk_checksums_sha256"):
            bad = dict(good)
            bad[field] = "mismatch"
            reason = sw.state_identity_guard(good, bad)
            assert reason is not None and field in reason
        mismatched_shape = dict(good, rows=11)
        assert sw.state_identity_guard(good, mismatched_shape) is not None
        unknown = dict(good, extra_dependency="?")
        assert sw.state_identity_guard(good, unknown) is not None
        missing = {k: v for k, v in good.items() if k != "rows"}
        assert sw.state_identity_guard(good, missing) is not None

    def test_build_applies_guards_in_order(self):
        a, dem, canopy = synth_planes(24, 32, seed=17)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, paint_disc(canopy, 10, 10, 3, 12.0))
        tables = synth_tables(SYN_ANGLES, 20.0, rows=24, cols=32)
        good_identity = {
            "schema_version": 1,
            "site_id": "s",
            "tile_key": "t",
            "cache_manifest_sha256": "x",
            "patch_geometry_id": "p",
            "rows": 24,
            "cols": 32,
            "chunk_checksums_sha256": "c",
        }
        # clean build passes
        sw.build_sparse_work(
            pre,
            post,
            dict(tables),
            dict(tables),
            identity_pre=good_identity,
            identity_post=dict(good_identity),
        )
        # identity mismatch refuses
        with pytest.raises(sw.SparseWorkFallback, match="identity"):
            sw.build_sparse_work(
                pre,
                post,
                dict(tables),
                dict(tables),
                identity_pre=good_identity,
                identity_post=dict(good_identity, site_id="other"),
            )
        # nonzero bush refuses
        bushy = sw.MarchInputs(
            **{**vars(post), "bush": post.bush + np.float32(1.0)}
        )
        with pytest.raises(sw.SparseWorkFallback, match="bush"):
            sw.build_sparse_work(pre, bushy, dict(tables), dict(tables))
        # one-step tables refuse
        one = synth_tables([(0.0, 78.0)], 4.0, rows=24, cols=32)
        with pytest.raises(sw.SparseWorkFallback, match="one-step"):
            sw.build_sparse_work(pre, post, dict(one), dict(one))
        # clamp Case C refuses
        small = _guard_scene(0.0, 10.0, 6.0, shape=(24, 32))
        large = _guard_scene(0.0, 20.0, 20.0, shape=(24, 32))
        with pytest.raises(sw.SparseWorkFallback, match="clamp"):
            sw.build_sparse_work(small, large, dict(tables), dict(tables))

    def test_build_validates_tables(self):
        a, dem, canopy = synth_planes(24, 32, seed=19)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, paint_disc(canopy, 8, 8, 3, 12.0))
        tables = synth_tables(SYN_ANGLES, 20.0, rows=24, cols=32)
        with pytest.raises(ValueError, match="keys"):
            sw.build_sparse_work(pre, post, dict(tables), {0: tables[0]})
        wrong_shape = synth_tables(SYN_ANGLES, 20.0, rows=16, cols=16)
        with pytest.raises(ValueError, match="shape"):
            sw.build_sparse_work(pre, post, wrong_shape, wrong_shape)


# ---------------------------------------------------------------------------
# Sparse representations, metrics, determinism
# ---------------------------------------------------------------------------


class TestRepresentations:
    def _random_masks(self):
        rng = np.random.default_rng(41)
        shapes = [(1, 1), (7, 13), (40, 80), (64, 64), (65, 129)]
        for rows, cols in shapes:
            for density in (0.02, 0.3, 0.9):
                yield rng.random((rows, cols)) < density
        yield np.zeros((5, 5), dtype=bool)
        yield np.ones((5, 5), dtype=bool)

    def test_row_run_roundtrip_and_sorted_order(self):
        for mask in self._random_masks():
            runs = sw.row_runs_from_mask(mask)
            rebuilt = sw.mask_from_row_runs(runs, mask.shape)
            assert np.array_equal(rebuilt, mask)
            if len(runs):
                order = list(map(tuple, runs.tolist()))
                assert order == sorted(order), "runs not row-major sorted"
                # runs are maximal: no two runs in a row touch or overlap
                by_row = {}
                for r, c0, c1 in map(tuple, runs.tolist()):
                    by_row.setdefault(r, []).append((c0, c1))
                for r, spans in by_row.items():
                    spans_sorted = sorted(spans)
                    for (a0, a1), (b0, _b1) in zip(
                        spans_sorted, spans_sorted[1:]
                    ):
                        assert b0 > a1 + 1, f"row {r} runs not maximal"

    def test_chunked_bitset_roundtrip(self):
        for mask in self._random_masks():
            bitset = sw.chunk_bitset_from_mask(mask)
            rebuilt = sw.mask_from_chunk_bitset(bitset)
            assert np.array_equal(rebuilt, mask)
            keys = list(bitset.chunk_keys)
            assert keys == sorted(keys)

    def test_build_emits_both_representations_consistently(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, paint_disc(canopy, 20, 50, 5, 12.0))
        tables = synth_tables(SYN_ANGLES, 20.0)
        build = sw.build_sparse_work(
            pre, post, dict(tables), dict(tables), keep_masks=True
        )
        masks = build.masks()
        assert build.pair_count > 0
        assert set(build.row_runs) == set(build.bitsets) == set(masks)
        total = 0
        for idx, runs in build.row_runs.items():
            total += int(sw.mask_from_row_runs(runs, (SYN_ROWS, SYN_COLS)).sum())
            bitset = build.bitsets[idx]
            assert np.array_equal(
                sw.mask_from_chunk_bitset(bitset), masks[idx]
            )
        assert total == build.pair_count
        assert build.row_run_count == sum(
            len(r) for r in build.row_runs.values()
        )

    def test_metrics_arithmetic_and_dense_ceiling(self):
        a, dem, canopy = synth_planes(24, 32)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(a, dem, paint_disc(canopy, 10, 16, 2, 12.0))
        tables = synth_tables(SYN_ANGLES, 20.0, rows=24, cols=32)
        build = sw.build_sparse_work(pre, post, dict(tables), dict(tables))
        n_runs = build.row_run_count
        n_patches = len(build.row_runs)
        assert (
            build.estimated_bytes_row_runs
            == sw.ROW_RUN_BYTES_PER_RUN * n_runs + 4 * n_patches
        )
        chunks = sum(len(b.chunk_keys) for b in build.bitsets.values())
        assert (
            build.estimated_bytes_bitsets
            == sw.CHUNK_BYTES * chunks + 4 * n_patches
        )
        assert build.dense_equivalent_bytes == sum(
            (24 * 32 + 7) // 8 for _ in build.row_runs
        )
        assert not build.route_dense
        # a full-domain closure (C empty but tables changed) must flag dense
        tables_hi = synth_tables(SYN_ANGLES, 60.0, rows=24, cols=32)
        full = sw.build_sparse_work(pre, pre, tables, tables_hi)
        assert full.route_dense and full.route_dense_patches

    def test_deterministic_rebuild_byte_identical(self):
        a, dem, canopy = synth_planes(SYN_ROWS, SYN_COLS)
        pre = synth_inputs(a, dem, canopy)
        post = synth_inputs(
            a, dem, paint_disc(paint_disc(canopy, 20, 50, 5, 12.0), 12, 20, 4, 9.0)
        )
        tables = synth_tables(SYN_ANGLES, 20.0)
        b1 = sw.build_sparse_work(pre, post, dict(tables), dict(tables))
        b2 = sw.build_sparse_work(pre, post, dict(tables), dict(tables))
        assert list(b1.row_runs) == list(b2.row_runs)
        for idx in b1.row_runs:
            assert b1.row_runs[idx].tobytes() == b2.row_runs[idx].tobytes()
            assert (
                b1.bitsets[idx].words.tobytes()
                == b2.bitsets[idx].words.tobytes()
            )
            assert b1.bitsets[idx].chunk_keys == b2.bitsets[idx].chunk_keys
        assert b1.to_dict() == b2.to_dict()


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestPurity:
    def test_source_is_torch_free_no_fastmath(self):
        source = (REPO_ROOT / "solweig_core" / "sparse_work.py").read_text()
        assert "import torch" not in source
        assert "from torch" not in source
        assert "fastmath=True" not in source
        assert "parallel=True" not in source
        assert "prange" not in source

    def test_module_torch_free_subprocess(self):
        code = (
            "import sys; import numpy as np; "
            "from solweig_core import sparse_work as sw; "
            "assert 'torch' not in sys.modules; "
            "a = np.zeros((6, 8), dtype=np.float32); "
            "pre = sw.compose_march_inputs(a, a.copy(), a.copy()); "
            "post = sw.compose_march_inputs(a, a.copy(), a.copy()); "
            "post.vegdsm[2, 3] = np.float32(1.0); "
            "C = sw.composed_change_mask(pre, post); "
            "D = sw.corridor_mask(C, [(1, 0), (0, 1)]); "
            "assert D.sum() > 0 and (D & ~C).any()"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# site_500 gates: closure completeness + untouched-state splice + measurement
# ---------------------------------------------------------------------------


SITE_SUBSET_ALTITUDES = {6.0, 42.0, 78.0, 90.0}


@pytest.fixture(scope="module")
def site():
    if not SITE_500_CACHE.is_dir():
        pytest.skip(f"site cache {SITE_500_CACHE} not present")
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.geometry import RasterGrid
    from solweig_gpu.incremental.solver import (
        _compose_scene_from_canopy,
        _sky_patch_geometry,
    )

    cache = SiteCache.load(SITE_500_CACHE)
    grid = RasterGrid(
        rows=cache.rows,
        cols=cache.cols,
        pixel_size_m=cache.pixel_size_m,
        origin_x_m=cache.manifest.origin_x_m,
        origin_y_m=cache.manifest.origin_y_m,
    )
    canopy_base = np.asarray(cache.tree_base, dtype=np.float32).copy()
    patches, _rings = _sky_patch_geometry(2)
    return {
        "cache": cache,
        "grid": grid,
        "canopy_base": canopy_base,
        "scale": 1.0 / float(cache.pixel_size_m),
        "rows": int(cache.rows),
        "cols": int(cache.cols),
        "patches": patches,
        "compose_torch": _compose_scene_from_canopy,
    }


_SCENE_CACHE: dict[str, dict] = {}


def site_scene(state, name, canopy):
    """(MarchInputs, torch scene, amplitude policy, executed amps)."""
    cached = _SCENE_CACHE.get(name)
    if cached is not None:
        return cached
    a = np.array(state["cache"].building_dsm)
    dem = np.array(state["cache"].dem)
    inputs = sw.compose_march_inputs(a, dem, canopy)
    scene_t = state["compose_torch"](state["cache"], canopy)
    policy = capture_amplitude_policy(
        scene_t.a,
        scene_t.vegdsm,
        scene_t.vegdsm2,
        scene_t.vegdem,
        scale=state["scale"],
    )
    amps = {}
    for idx in range(len(state["patches"])):
        amps[idx] = executed_amplitude_for_patch(
            policy, idx, escalated=False
        )
    cached = {
        "inputs": inputs,
        "scene_t": scene_t,
        "policy": policy,
        "amps": amps,
    }
    _SCENE_CACHE[name] = cached
    return cached


def site_tables(state, scene_name, canopy, patch_indices):
    scene = site_scene(state, scene_name, canopy)
    tables = {}
    for idx in patch_indices:
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
        tables[idx] = st.build_table_from_trace(trace)
    return tables, scene


def cell_xy(state, cell):
    grid = state["grid"]
    return (
        grid.origin_x_m + (cell[1] + 0.5) * grid.pixel_size_m,
        grid.origin_y_m - (cell[0] + 0.5) * grid.pixel_size_m,
    )


def site_canopy_with(state, trees):
    """Rasterize base + trees through the REAL TreeLayer (max combine)."""
    from solweig_gpu.incremental.geometry import TreeSpec
    from solweig_gpu.incremental.trees import TreeLayer

    layer = TreeLayer(state["canopy_base"], state["grid"])
    for tree_id, cell, height, radius in trees:
        x, y = cell_xy(state, cell)
        layer.add_tree(TreeSpec(tree_id, x, y, height, radius))
    canopy = np.ascontiguousarray(
        layer.vegetation_rasters_window(
            _raster_window(state, 0, state["rows"], 0, state["cols"])
        )[0]
    )
    return canopy, layer


def _raster_window(state, r0, r1, c0, c1):
    from solweig_gpu.incremental.geometry import RasterWindow

    return RasterWindow(int(r0), int(r1), int(c0), int(c1))


def subset_patch_indices(state):
    return [
        i
        for i, patch in enumerate(state["patches"])
        if float(patch[0]) in SITE_SUBSET_ALTITUDES
    ]


def all_patch_indices(state):
    return list(range(len(state["patches"])))


def run_site_family(
    state,
    label,
    canopy_pre,
    canopy_post,
    tables_pre,
    tables_post,
    scene_pre,
    scene_post,
    *,
    gate3=True,
):
    """Build the closure; verify completeness (gate 2) and the
    closure-only splice (gate 3); return the measurement record."""
    from solweig_gpu.incremental.solver import (
        _patch_march_reach_pixels,
        _patch_march_window,
    )

    rows, cols = state["rows"], state["cols"]
    scale = state["scale"]
    build = sw.build_sparse_work(
        scene_pre["inputs"],
        scene_post["inputs"],
        tables_pre,
        tables_post,
        keep_masks=True,
    )
    masks = build.masks()
    eff_pre = float(sw.effective_amplitude(scene_pre["inputs"]))
    eff_post = float(sw.effective_amplitude(scene_post["inputs"]))
    reach_amp = max(eff_pre, eff_post)
    full_window = _raster_window(state, 0, rows, 0, cols)

    total_changed = 0
    misses_outside = 0
    splice_mismatch_outside = 0
    splice_mismatch_inside = 0
    checked = 0
    pair_steps_sparse = 0
    pair_steps_dense = 0
    for idx in sorted(tables_pre):
        table_pre = tables_pre[idx]
        table_post = tables_post[idx]
        out_pre = full_march(table_pre, scene_pre["inputs"])
        out_post = full_march(table_post, scene_post["inputs"])
        D = masks.get(idx, np.zeros((rows, cols), dtype=bool))
        splice = [p.copy() for p in out_pre]
        if D.any() and gate3:
            rs, cs = np.nonzero(D)
            altitude, azimuth, _ring = state["patches"][idx]
            bbox = _raster_window(
                state, rs.min(), rs.max() + 1, cs.min(), cs.max() + 1
            )
            reach = _patch_march_reach_pixels(
                reach_amp, scale, float(altitude)
            )
            window = _patch_march_window(
                bbox, full_window, azimuth_deg=float(azimuth), reach_pixels=reach
            )
            amp_post, pid = scene_post["amps"][idx]
            trace = capture_trace(
                st.KERNEL_SVF_SHADOW,
                float(azimuth),
                float(altitude),
                scale,
                window.row_stop - window.row_start,
                window.col_stop - window.col_start,
                amp_post,
                amplitude_policy_id=pid,
                verify_probes=False,
            )
            win_table = st.build_table_from_trace(trace)
            rs_sl = slice(window.row_start, window.row_stop)
            cs_sl = slice(window.col_start, window.col_stop)
            win_out = march_svf_shadow(
                win_table,
                scene_post["inputs"].a[rs_sl, cs_sl],
                scene_post["inputs"].vegdsm[rs_sl, cs_sl],
                scene_post["inputs"].vegdsm2[rs_sl, cs_sl],
                scene_post["inputs"].bush[rs_sl, cs_sl],
            )
            local = D[rs_sl, cs_sl]
            for k in range(3):
                splice[k][D] = win_out[k][local]
        checked += 1
        pair_steps_sparse += int(D.sum()) * int(table_post.count)
        if D.any():
            pair_steps_dense += rows * cols * int(table_post.count)
        for name, p0, p1, sp in zip(
            ("sh", "vegsh", "vbsh"), out_pre, out_post, splice
        ):
            changed = changed_cells(p0, p1)
            total_changed += int(changed.sum())
            miss = changed & ~D
            misses_outside += int(miss.sum())
            if gate3:
                splice_diff = changed_cells(sp, p1)
                splice_mismatch_outside += int(
                    (splice_diff & ~D).sum()
                )
                splice_mismatch_inside += int((splice_diff & D).sum())
    record = {
        "family": label,
        "patches_checked": checked,
        "C_cells": int(build.C.sum()),
        "pair_count": int(build.pair_count),
        "row_run_count": int(build.row_run_count),
        "patches_with_work": len(build.row_runs),
        "pair_steps_sparse": pair_steps_sparse,
        "pair_steps_dense": pair_steps_dense,
        "full_cells": rows * cols,
        "total_changed_cells": total_changed,
        "misses_outside_closure": misses_outside,
        "splice_mismatch_outside_closure": splice_mismatch_outside,
        "splice_mismatch_inside_closure": splice_mismatch_inside,
        "estimated_bytes_row_runs": int(build.estimated_bytes_row_runs),
        "estimated_bytes_bitsets": int(build.estimated_bytes_bitsets),
        "dense_equivalent_bytes": int(build.dense_equivalent_bytes),
        "route_dense": bool(build.route_dense),
        "route_dense_patches": list(build.route_dense_patches),
        "build_seconds": build.build_seconds,
        "eff_amplitude_pre": eff_pre,
        "eff_amplitude_post": eff_post,
        "tables_differ": any(
            tables_pre[i].content_digest() != tables_post[i].content_digest()
            for i in tables_pre
        ),
    }
    assert misses_outside == 0, (
        f"{label}: {misses_outside} changed (cell,patch) cells outside the "
        "closure"
    )
    if gate3:
        assert splice_mismatch_outside == 0, (
            f"{label}: {splice_mismatch_outside} untouched cells diverged "
            "after the closure-only splice"
        )
        assert splice_mismatch_inside == 0, (
            f"{label}: {splice_mismatch_inside} closure cells diverged "
            "between the windowed recompute and the full post march"
        )
    assert total_changed > 0, f"{label}: fixture changed no output (vacuous)"
    return build, record


def _write_measurement(records):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "site_measurement.json"
    existing = []
    if path.is_file():
        existing = json.loads(path.read_text())
    by_family = {r["family"]: r for r in existing}
    for record in records:
        by_family[record["family"]] = record
    path.write_text(
        json.dumps(list(by_family.values()), indent=2, sort_keys=True) + "\n"
    )
    return path


class TestSiteClosure:
    def test_site_compose_mirror_bits(self, site):
        scene = site_scene(site, "base", site["canopy_base"])
        theirs = scene["scene_t"]
        mine = scene["inputs"]
        for name in ("a", "vegdem", "vegdem2", "bush", "vegdsm", "vegdsm2"):
            assert np.array_equal(
                bits(getattr(mine, name)), bits(getattr(theirs, name).numpy())
            ), name
        assert np.float32(mine.amaxvalue).view(U32) == np.float32(
            float(theirs.amaxvalue)
        ).view(U32)

    def test_f1_add_tree_near_canopy(self, site, measurements):
        """E1 exactly: add t1 h10 r4 at (166,102); subset altitudes."""
        canopy_post, layer = site_canopy_with(
            site, [("t1", (166, 102), 10.0, 4.0)]
        )
        canopy_pre = site["canopy_base"]
        patch_indices = subset_patch_indices(site)
        tables_pre, scene_pre = site_tables(
            site, "base", canopy_pre, patch_indices
        )
        tables_post, scene_post = site_tables(
            site, "f1_post", canopy_post, patch_indices
        )
        # the E1 dirty window must contain every changed composed cell
        dirty = layer.affected_window()
        window_mask = np.zeros((site["rows"], site["cols"]), dtype=bool)
        window_mask[
            dirty.row_start : dirty.row_stop, dirty.col_start : dirty.col_stop
        ] = True
        build = sw.build_sparse_work(
            scene_pre["inputs"],
            scene_post["inputs"],
            tables_pre,
            tables_post,
            candidate_mask=window_mask,
            verify_candidate=True,
            keep_masks=True,
        )
        _b, record = run_site_family(
            site,
            "F1_add_t1",
            canopy_pre,
            canopy_post,
            tables_pre,
            tables_post,
            scene_pre,
            scene_post,
        )
        measurements.append(record)

    def test_f2_move_tree(self, site, measurements):
        """E2-style move: pre = base + t1 at (166,102), post = base + t1 at
        (166,127) — old AND new footprints in the diff."""
        canopy_pre, _layer = site_canopy_with(
            site, [("t1", (166, 102), 10.0, 4.0)]
        )
        canopy_post, _l2 = site_canopy_with(
            site, [("t1", (166, 127), 10.0, 4.0)]
        )
        patch_indices = subset_patch_indices(site)
        tables_pre, scene_pre = site_tables(
            site, "f1_post", canopy_pre, patch_indices
        )
        tables_post, scene_post = site_tables(
            site, "f2_post", canopy_post, patch_indices
        )
        _b, record = run_site_family(
            site,
            "F2_move_t1",
            canopy_pre,
            canopy_post,
            tables_pre,
            tables_post,
            scene_pre,
            scene_post,
        )
        measurements.append(record)

    def test_f3_delete_overlap_all_patches(self, site, measurements):
        """Delete-overlap with amplitude change, ALL 153 patches: pre =
        base + tall A + shorter overlapping B; post = base + B (A deleted —
        the overlap-hidden reveal)."""
        tall = float(site["canopy_base"].max()) + 25.0  # raises vegdsm.max
        short = tall - 5.0
        canopy_pre, _layer = site_canopy_with(
            site,
            [
                ("A", (166, 150), tall, 8.0),
                ("B", (166, 153), short, 8.0),
            ],
        )
        canopy_post, _l2 = site_canopy_with(
            site, [("B", (166, 153), short, 8.0)]
        )
        patch_indices = all_patch_indices(site)
        tables_pre, scene_pre = site_tables(
            site, "f3_pre", canopy_pre, patch_indices
        )
        tables_post, scene_post = site_tables(
            site, "f3_post", canopy_post, patch_indices
        )
        assert record_amp_change(site, scene_pre, scene_post), (
            "fixture does not change the effective amplitude"
        )
        build, record = run_site_family(
            site,
            "F3_delete_overlap",
            canopy_pre,
            canopy_post,
            tables_pre,
            tables_post,
            scene_pre,
            scene_post,
        )
        assert record["tables_differ"], (
            "fixture did not change the executed step tables"
        )
        # determinism on the real site (gate 7)
        again = sw.build_sparse_work(
            scene_pre["inputs"],
            scene_post["inputs"],
            tables_pre,
            tables_post,
        )
        for idx in build.row_runs:
            assert (
                build.row_runs[idx].tobytes()
                == again.row_runs[idx].tobytes()
            )
        measurements.append(record)

    def test_measurement_artifact_written(self, measurements):
        assert measurements, "no family ran"
        path = _write_measurement(measurements)
        data = json.loads(path.read_text())
        assert {r["family"] for r in data} >= {
            r["family"] for r in measurements
        }
        for record in measurements:
            fraction = record["pair_steps_sparse"] / max(
                1, record["pair_steps_dense"]
            )
            print(
                f"[t05] {record['family']}: pairs {record['pair_count']} "
                f"({record['pair_count'] / record['full_cells']:.4f} of one "
                f"patch-plane), pair-step ratio {fraction:.4f}, "
                f"C cells {record['C_cells']}, build "
                f"{record['build_seconds'] * 1000:.1f} ms"
            )


def record_amp_change(state, scene_pre, scene_post):
    return float(sw.effective_amplitude(scene_pre["inputs"])) != float(
        sw.effective_amplitude(scene_post["inputs"])
    )


@pytest.fixture(scope="module")
def measurements():
    return []
