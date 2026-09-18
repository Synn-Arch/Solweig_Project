# SPDX-License-Identifier: GPL-3.0-only
"""G2.1 item 4: the first warm-start consumer — met sparse replay.

Design: ``docs/incremental_design_tool/realtime_collaboration/design/
r5-temporal-checkpoints-and-fast-serve.md`` section G2.1 (binding).

A RADIATION-AFFECTING meteorology edit at row ``r0`` today falls back to a
full ``0..T`` replay although rows ``< r0`` are provably unchanged (the
thermal state chain is causal — state@k depends only on inputs ``0..k-1``).
The consumer here serves ``t < r0`` from the result store under the SAME
strict FULL-coverage gate the r3a fast path uses, solves the suffix warm
from a fingerprint-matching checkpoint, and publishes ONE sparse patch
(``time_start = r0``). Fingerprint mismatch / torn checkpoint / absent
checkpoint is NEVER an error surfaced: the prefix replays cold from step 0
(the split solve) and the reason rides the outcome as fallback telemetry.

Hard fences: the executor E2E served arrays (composed from the store over
ALL timesteps) are bitwise today's cold full replay; the anchor committed
at publication is the state@r0 the suffix actually consumed; a checkpoint
whose met prefix covers an edited row can never warm-serve (the digest is
the fence, not the revision).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solweig_gpu.incremental.checkpoints import (
    list_checkpoints,
    load_checkpoint,
)
from solweig_gpu.incremental.result import load_patch
from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile

from tests.test_incremental_executor import _executor, _met_edit
from tests.test_incremental_worker import (
    DATE_STR,
    TINY_COLS,
    TINY_EPSG,
    TINY_ORIGIN,
    TINY_PIXEL,
    TINY_ROWS,
    TINY_TREE,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
)

MET_HOURS = tuple(range(0, 24))  # T = 24 timesteps (the honest ~24 baseline)


def _warm_executor(tmp_path: Path):
    """A real-physics executor: prepared site + REAL baseline SVF.

    ``_make_tiny_site``'s fake 16-patch SVF cubes cannot feed
    ``solve_window``'s cache-slice path (the physics iterates 153 patches),
    so the warm consumer's solves need the prepared-site fixture the
    warm==cold parity tests use.
    """
    grid, site = _make_prepared_site(
        tmp_path / "site",
        rows=TINY_ROWS,
        cols=TINY_COLS,
        pixel=TINY_PIXEL,
        origin=TINY_ORIGIN,
        epsg=TINY_EPSG,
        base_trees=(TINY_TREE,),
        met_hours=MET_HOURS,
    )
    _compute_baseline_svf(site)
    # _executor builds the cache from the site; the real SVF outputs the
    # baseline pass wrote are picked up by build_site_cache.
    return _executor(tmp_path, grid=grid, site=site)


def _rad_edit(executor, *, row: int, revision: int, value: float, edit_id: str):
    """A validated RADIATION-AFFECTING met edit (air_temperature) at ``row``."""
    return _met_edit(
        executor,
        revision=revision,
        times=(row,),
        time_index=row,
        value=value,
        edit_id=edit_id,
    )


def _anchor(executor, revision: int):
    root = executor.results_root / executor._scenario_id / "checkpoints"
    return load_checkpoint(root / f"rev-{revision:06d}")


class TestMetWarmSparsePath:
    def test_cold_split_then_warm_resume_with_anchor_chain(self, tmp_path):
        executor = _warm_executor(tmp_path)
        total = len(MET_HOURS)

        # Seed: a fresh store cannot serve any prefix row, so the first
        # radiation edit refuses the warm path and publishes today's cold
        # FULL replay (which captures the state@T anchor, item 3).
        first = executor.execute([_rad_edit(executor, row=1, revision=0,
                                            value=21.0, edit_id="m1")])
        assert first.published and first.mode == "full"
        assert "met_warm_path_refusal" in first.diagnostics
        assert "coverage" in first.diagnostics["met_warm_path_refusal"]
        seed_patch = load_patch(first.patch_paths[0])
        assert seed_patch.time_start == 0

        # Second edit at row 20: the only anchor is state@24 whose met
        # prefix covers the edited row — it can never warm-serve. The
        # consumer still publishes the SPARSE suffix (cold split 0..20 +
        # warm 20..24) and commits a fresh anchor@20 at this revision.
        second = executor.execute([_rad_edit(executor, row=20, revision=1,
                                             value=29.0, edit_id="m2")])
        assert second.published and second.mode == "full"
        assert second.diagnostics["met_warm_path"] is True
        assert second.diagnostics["met_warm_r0"] == 20
        assert second.diagnostics["met_warm_resume_step"] == 0
        assert second.diagnostics["met_warm_refusal"] is not None
        patch = load_patch(second.patch_paths[0])
        assert patch.time_start == 20
        assert patch.time_stop == total
        assert patch.variable("utci").shape[0] == total - 20
        anchor = _anchor(executor, 2)
        assert anchor.next_step == 20
        assert set(anchor.thermal_tensors) == {
            "Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N", "TgOut1",
        }

        # Third edit at row 21: the anchor@20's met prefix (rows 0..19) is
        # unchanged, so the suffix solves WARM from step 20 — the honest
        # ~24 -> ~4 full-tile steps win — and re-anchors at 21.
        third = executor.execute([_rad_edit(executor, row=21, revision=2,
                                            value=30.0, edit_id="m3")])
        assert third.published
        assert third.diagnostics["met_warm_resume_step"] == 20
        assert third.diagnostics["met_warm_checkpoint_revision"] == 2
        assert third.diagnostics["met_warm_refusal"] is None
        assert third.diagnostics["met_warm_steps_solved"] == total - 20
        warm_patch = load_patch(third.patch_paths[0])
        assert warm_patch.time_start == 21
        assert warm_patch.variable("utci").shape[0] == total - 21
        assert _anchor(executor, 3).next_step == 21

    def test_e2e_served_arrays_equal_cold_replay_bitwise(self, tmp_path):
        executor = _warm_executor(tmp_path)
        total = len(MET_HOURS)

        executor.execute([_rad_edit(executor, row=1, revision=0,
                                    value=21.0, edit_id="m1")])
        executor.execute([_rad_edit(executor, row=20, revision=1,
                                    value=29.0, edit_id="m2")])
        executor.execute([_rad_edit(executor, row=21, revision=2,
                                    value=30.0, edit_id="m3")])

        # Compose the served utci planes exactly like the read path does:
        # per (t), paint the FULL-coverage entries in ascending revision
        # order (G1.1 per-cell latest validity).
        full = executor.grid.full_window
        served = np.empty(
            (total, full.height, full.width), dtype=np.float32
        )
        for t in range(total):
            coverage = executor.store.window_coverage("utci", t, window=full)
            assert coverage.status.value == "full"
            patches: dict[Path, object] = {}
            for entry in sorted(
                coverage.entries, key=lambda item: item.scene_revision
            ):
                if entry.patch_path not in patches:
                    patches[entry.patch_path] = load_patch(entry.patch_path)
                patch = patches[entry.patch_path]
                row = (
                    patch.time_indices.index(t)
                    if patch.time_indices is not None
                    else t - patch.time_start
                )
                w = entry.write_window
                pw = patch.write_window
                served[
                    t,
                    w.row_start : w.row_stop,
                    w.col_start : w.col_stop,
                ] = patch.variable("utci")[
                    row,
                    w.row_start - pw.row_start : w.row_stop - pw.row_start,
                    w.col_start - pw.col_start : w.col_stop - pw.col_start,
                ]

        # Today's cold replay: a full-tile run under the committed overlay.
        forcing = load_site_forcing(
            executor.cache,
            site_dir=executor.site_dir,
            selected_date_str=executor.selected_date_str,
            overlay=executor._applied_forcing_overlay,
        )
        cold = run_full_tile(
            executor.cache,
            executor.layer,
            forcing=forcing,
            site_dir=executor.site_dir,
            scratch_dir=tmp_path / "scratch_cold",
            requested_variables=("utci",),
        )
        oracle = cold["utci"]
        assert oracle.shape == served.shape
        # Bitwise, not ==: NaN payloads must match byte-exactly.
        assert served.tobytes() == oracle.tobytes()

    def test_fingerprint_mismatch_inside_prefix_replays_cold_split(
        self, tmp_path
    ):
        executor = _warm_executor(tmp_path)

        executor.execute([_rad_edit(executor, row=1, revision=0,
                                    value=21.0, edit_id="m1")])
        executor.execute([_rad_edit(executor, row=20, revision=1,
                                    value=29.0, edit_id="m2")])
        # The anchor@20's met prefix covers rows 0..19; an edit at row 5
        # invalidates it (the state HEARD the old row 5) — cold split.
        edit = executor.execute([_rad_edit(executor, row=5, revision=2,
                                           value=27.0, edit_id="m3")])
        assert edit.published
        assert edit.diagnostics["met_warm_resume_step"] == 0
        # The refusal names the boundary property either way: the anchor's
        # met prefix covers the edited row (the structural next_step guard
        # fires first; the digest check agrees on the same boundary).
        refusal = edit.diagnostics["met_warm_refusal"]
        assert "met_prefix" in refusal or "beyond the first changed row" in refusal
        patch = load_patch(edit.patch_paths[0])
        assert patch.time_start == 5
        assert _anchor(executor, 3).next_step == 5

    def test_torn_anchor_falls_back_cold_split(self, tmp_path):
        executor = _warm_executor(tmp_path)

        executor.execute([_rad_edit(executor, row=1, revision=0,
                                    value=21.0, edit_id="m1")])
        executor.execute([_rad_edit(executor, row=20, revision=1,
                                    value=29.0, edit_id="m2")])
        # Tear the anchor@20 the way a kill mid-persist would: truncated
        # tensor bytes. The scan must skip it (typed rejection consumed,
        # never surfaced) and replay the prefix cold.
        root = executor.results_root / executor._scenario_id / "checkpoints"
        tensor = root / "rev-000002" / "tensors" / "Tgmap1.npy"
        raw = tensor.read_bytes()
        tensor.write_bytes(raw[: len(raw) // 2])
        assert list_checkpoints(root) == (1, 2)

        edit = executor.execute([_rad_edit(executor, row=21, revision=2,
                                           value=30.0, edit_id="m3")])
        assert edit.published
        assert edit.diagnostics["met_warm_resume_step"] == 0
        refusal = edit.diagnostics["met_warm_refusal"]
        assert "torn/corrupt" in refusal or "corrupt" in refusal
        patch = load_patch(edit.patch_paths[0])
        assert patch.time_start == 21

    def test_sparse_patch_replaces_only_the_suffix_rows(self, tmp_path):
        executor = _warm_executor(tmp_path)

        executor.execute([_rad_edit(executor, row=1, revision=0,
                                    value=21.0, edit_id="m1")])
        executor.execute([_rad_edit(executor, row=20, revision=1,
                                    value=29.0, edit_id="m2")])
        # Row 19 (inside the unchanged prefix) keeps its revision-1 entry;
        # row 20 is owned by the revision-2 sparse patch.
        before = executor.store.window_entries("utci", 19)
        at_r0 = executor.store.window_entries("utci", 20)
        assert {e.scene_revision for e in before} == {1}
        assert any(e.scene_revision == 2 for e in at_r0)
        assert all(e.scene_revision >= 1 for e in at_r0)
