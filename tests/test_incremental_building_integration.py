# SPDX-License-Identifier: GPL-3.0-only
"""U-C6 tests: the building executor wiring (u-c5 chain -> publication).

Fast suite (chain stubbed, no GDAL/torch physics): the dispatch contract —
a committed building batch routes the WHOLE batch through
``PlanExecutor._regenerate_building_chain``, mixed batches ride the chain
(tree layer in, forcing overlay in, paint overlay in), a mid-chain failure
rolls back under the u-c1b discipline AND wipes the executor's scenario
root, a post-publish failure removes the orphan patch (u-c1b M4), and the
worker adopts the publication so nothing is left pending.

Scientific suite (real physics, one 48x48 real-SVF site): a building batch
through the executor publishes FULL-tile results BITWISE-equal to an
independently-run ``regenerate_building_batch`` oracle at the same
committed state — with the scenario's accumulated overlays riding the
chain (met edit published first, then the building batch: the results
reflect BOTH) — plus store coverage at the publication revision, patch
provenance from the rebuilt scenario cache, baseline immutability
(byte-hash), rollback on an injected mid-chain failure, and the
walllimit/raster-edge disclosures surfacing in the executed plan's
diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solweig_gpu.incremental.edit_types import EditCommand
from solweig_gpu.incremental.executor import ExecutorError, PlanExecutor
from solweig_gpu.incremental.regenerate import (
    BuildingRegenerationResult,
    RegenerationError,
)
from solweig_gpu.incremental.result import load_patch
from solweig_gpu.incremental.trees import TreeLayer

from tests.test_incremental_buildings import (
    DATE_STR,
    NEW_BLOCK_FOOT,
    SITE_BLOCK_FOOT,
    TINY_ORIGIN,
    TINY_PIXEL,
    _chain_edits,
    _hash_tree,
    baseline,
)
from tests.test_incremental_lc_integration import PAINT_WINDOW, _lc_overlay, _paint_command
from tests.test_incremental_met_integration import _executor_met_edit
from tests.test_incremental_remediation_uc1b import _stub_local_every_window
from tests.test_incremental_worker import LOCAL_ALWAYS

#: The committed scenario addition used across the suite: a NEW 15 m block.
NEW_BLOCK_STATE = {
    "building_id": "new-block",
    "footprint_m": [list(vertex) for vertex in NEW_BLOCK_FOOT],
    "height_m": 15.0,
}

#: A SECOND scenario addition (u-d4c building->building chain): a 10 m
#: block in the south-east quadrant, disjoint from NEW_BLOCK_FOOT's
#: footprint (rows/cols distinct) and from the baseline site block.
BLOCK_B_FOOT = (
    (1053.0, 1923.0),
    (1067.0, 1923.0),
    (1067.0, 1909.0),
    (1053.0, 1909.0),
)
NEW_BLOCK_B_STATE = {
    "building_id": "new-block-b",
    "footprint_m": [list(vertex) for vertex in BLOCK_B_FOOT],
    "height_m": 10.0,
}

#: OVERLAPPING session-add footprints (u-d4c remediation F1): two blocks
#: in the site's south-east interior — a region free of BOTH baseline
#: blocks (rows 3:9 x cols 2:9 and rows 29:36 x cols 28:36 on the 48x48
#: grid) — shifted 8 m against each other so FA∩FB is a non-empty 4x8
#: cell band. Unlike NEW_BLOCK_FOOT/BLOCK_B_FOOT (explicitly disjoint),
#: these pin the fold's overlap-order semantics.
_OVERLAP_RECT = (
    (1072.0, 1968.0),
    (1088.0, 1968.0),
    (1088.0, 1952.0),
    (1072.0, 1952.0),
)
OVERLAP_A_FOOT = _OVERLAP_RECT
#: B is A shifted 8 m WEST: cols 32:40 vs A's cols 36:44, rows 16:24 both.
OVERLAP_B_FOOT = (
    (1064.0, 1968.0),
    (1080.0, 1968.0),
    (1080.0, 1952.0),
    (1064.0, 1952.0),
)

#: The baseline block, raised 12 m -> 18 m (the update leg of _chain_edits).
RAISED_SITE_BLOCK_STATE = {
    "building_id": "site-block",
    "footprint_m": [list(vertex) for vertex in SITE_BLOCK_FOOT],
    "height_m": 18.0,
}

#: The baseline block as it sits in the prepared site (12 m).
BASE_SITE_BLOCK_STATE = {
    "building_id": "site-block",
    "footprint_m": [list(vertex) for vertex in SITE_BLOCK_FOOT],
    "height_m": 12.0,
}


def _building_command(
    *,
    revision: int = 0,
    state: dict = NEW_BLOCK_STATE,
    old_state: dict | None = None,
    edit_id: str = "bld-1",
    outputs: tuple[str, ...] = ("utci", "tmrt", "shadow"),
) -> EditCommand:
    return EditCommand(
        edit_id=edit_id,
        scenario_id="default",
        base_scene_revision=revision,
        adapter_id="building_geometry",
        operation="add" if old_state is None else "update",
        old_state=old_state,
        new_state=state,
        requested_outputs=outputs,
        requested_times=(1,),
    )


def _add_edits(*states: dict):
    """Committed add-only massing edits for the given block states (u-d4c)."""
    from solweig_gpu.incremental.adapters.building import massing_edits_from_deltas
    from solweig_gpu.incremental.edit_types import (
        BuildingMassingDelta,
        ObjectStateChange,
    )

    delta = BuildingMassingDelta(
        source_node_id="building_dsm",
        adapter_id="building_geometry",
        objects=tuple(
            ObjectStateChange(
                object_id=state["building_id"], before=None, after=dict(state)
            )
            for state in states
        ),
        windows=(),
    )
    return massing_edits_from_deltas((delta,))


def _add_only_edits():
    """One committed add-only massing edit (the scientific add leg)."""
    return _add_edits(NEW_BLOCK_STATE)


def _massing_edit(state: dict, *, deleted: bool = False):
    """One adapter-validated massing edit for ``state`` (delete: the
    state's spec in as ``before``, nothing after) — the session-order
    oracle leg for the overlap-order tests (u-d4c remediation F1)."""
    from solweig_gpu.incremental.adapters.building import massing_edits_from_deltas
    from solweig_gpu.incremental.edit_types import (
        BuildingMassingDelta,
        ObjectStateChange,
    )

    delta = BuildingMassingDelta(
        source_node_id="building_dsm",
        adapter_id="building_geometry",
        objects=(
            ObjectStateChange(
                object_id=state["building_id"],
                before=dict(state) if deleted else None,
                after=None if deleted else dict(state),
            ),
        ),
        windows=(),
    )
    return massing_edits_from_deltas((delta,))[0]


def _delete_command(
    *,
    revision: int = 0,
    state: dict = NEW_BLOCK_STATE,
    edit_id: str = "bld-del",
    outputs: tuple[str, ...] = ("utci", "tmrt", "shadow"),
) -> EditCommand:
    """A committed delete of ``state``'s building (old_state in, None out)."""
    return EditCommand(
        edit_id=edit_id,
        scenario_id="default",
        base_scene_revision=revision,
        adapter_id="building_geometry",
        operation="delete",
        old_state=state,
        new_state=None,
        requested_outputs=outputs,
        requested_times=(1,),
    )


def _veg_command(executor: PlanExecutor, *, edit_id: str = "veg-1") -> EditCommand:
    return EditCommand(
        edit_id=edit_id,
        scenario_id="default",
        base_scene_revision=executor.scene_revision,
        adapter_id="vegetation_geometry",
        operation="add",
        old_state=None,
        new_state={
            "tree_id": "t1",
            "x_m": TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
            "y_m": TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
            "height_m": 4.0,
            "canopy_radius_m": 2.0,
        },
        requested_outputs=("utci",),
        requested_times=(1,),
    )


def _scenario_root(tmp_path: Path) -> Path:
    return tmp_path / "results" / "default" / ".building"


def _bitwise(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Bitwise equality with NaN==NaN (utci saturates NaN at identical
    out-of-domain cells; ``np.array_equal`` without ``equal_nan`` would
    fail two IDENTICAL arrays that carry NaNs)."""
    return bool(np.array_equal(actual, expected, equal_nan=True))


# ---------------------------------------------------------------------------
# Fast suite: dispatch, mixed batches, rollback, watermark adoption
# ---------------------------------------------------------------------------


class TestBuildingDispatchFast:
    @pytest.fixture()
    def ex(self, tmp_path: Path, monkeypatch):
        from solweig_gpu.incremental import executor as executor_mod
        from solweig_gpu.incremental.buildings import BuildingRasterResult
        from tests.test_incremental_lc_integration import _tiny_executor

        executor = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_chain(**kwargs):
            captured.update(kwargs)
            cache = executor.cache
            variables = tuple(kwargs["requested_variables"])
            zeros = {
                name: np.zeros(
                    (cache.time_steps, cache.rows, cache.cols), dtype=np.float32
                )
                for name in variables
            }
            return BuildingRegenerationResult(
                scenario_site_dir=(
                    executor.results_root / "default" / ".building" / "site"
                ),
                cache_dir=executor.results_root / "default" / ".building" / "cache",
                cache=cache,
                raster=np.zeros((cache.rows, cache.cols), dtype=np.float32),
                records=BuildingRasterResult(
                    raster=np.zeros((cache.rows, cache.cols), dtype=np.float32),
                    records=(),
                ),
                outputs=zeros,
                stages=("stage_scenario_site", "run_scenario_full_tile"),
            )

        monkeypatch.setattr(executor_mod, "regenerate_building_batch", fake_chain)
        return SimpleNamespace(executor=executor, captured=captured, tmp_path=tmp_path)

    def test_building_add_publishes_full_and_adopts_watermarks(self, ex) -> None:
        executed = ex.executor.execute(
            [ex.executor.validate(_building_command())]
        )
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 1
        assert ex.executor.scene_revision == 1
        assert executed.diagnostics["building_batch"] is True
        assert executed.diagnostics["regeneration_stages"] == [
            "stage_scenario_site",
            "run_scenario_full_tile",
        ]
        # Store coverage: every timestep at the publication revision, full.
        assert ex.executor.store.revision_at("utci", 0) == 1
        assert ex.executor.store.lookup("utci", 0).mode == "full"
        # The worker adopted the out-of-band publication: nothing pending,
        # staged overlays synced to the published state.
        assert ex.executor._worker.pending_batch() is None
        assert ex.executor._worker.publication_watermarks.landcover_overlay is None

    def test_mixed_building_and_met_batch_carries_forcing_overlay(self, ex) -> None:
        met = _executor_met_edit(ex.executor, time_index=1, value=26.0)
        building = ex.executor.validate(_building_command())
        executed = ex.executor.execute([met, building])
        assert executed.published
        assert executed.mode == "full"
        overlay = ex.captured["forcing_overlay"]
        assert overlay is not None
        change = overlay.changes[0]
        assert change.variable == "air_temperature"
        assert change.time_index == 1
        assert change.after_value == 26.0
        # committed on publication
        assert ex.executor._applied_forcing_overlay == overlay

    def test_mixed_building_and_paint_batch_carries_landcover_overlay(self, ex) -> None:
        paint = ex.executor.validate(_paint_command())
        building = ex.executor.validate(_building_command(edit_id="bld-2"))
        executed = ex.executor.execute([paint, building])
        assert executed.published
        overlay = ex.captured["landcover_overlay"]
        assert overlay is not None
        baseline_grid = np.asarray(ex.executor.cache.landcover)
        assert np.all(overlay.resolve(baseline_grid)[40:48, 40:48] == 1)
        assert ex.executor._applied_landcover_overlay == overlay

    def test_mixed_building_and_veg_batch_rides_tree_layer(self, ex) -> None:
        veg = ex.executor.validate(_veg_command(ex.executor))
        building = ex.executor.validate(_building_command(edit_id="bld-2"))
        executed = ex.executor.execute([veg, building])
        assert executed.published
        # The chain received the executor's LIVE layer, already carrying
        # the replayed tree (the mixed batch's vegetation rides the chain).
        assert ex.captured["tree_layer"] is ex.executor.layer
        assert [
            tree.tree_id for tree in ex.captured["tree_layer"].current_trees()
        ] == ["t1"]
        # The replayed edits are consumed: nothing left pending on the
        # worker after the out-of-band publication.
        assert ex.executor._worker.pending_batch() is None

    def test_mid_chain_failure_rolls_back_and_wipes_scenario_root(
        self, ex, monkeypatch
    ) -> None:
        from solweig_gpu.incremental import executor as executor_mod

        def exploding(**kwargs):
            raise RegenerationError(
                "regeneration stage 'regenerate_svf' failed after "
                "('stage_scenario_site', 'write_scenario_building_dsm'): boom"
            )

        monkeypatch.setattr(executor_mod, "regenerate_building_batch", exploding)
        # Leave a marker in the scenario root: the rollback must wipe it.
        marker = _scenario_root(ex.tmp_path) / "partial" / "site"
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "stale.txt").write_text("partial chain")

        watermarks_before = ex.executor._worker.publication_watermarks
        with pytest.raises(ExecutorError, match="regeneration chain failed"):
            ex.executor.execute([ex.executor.validate(_building_command())])
        # u-c1b discipline: state, worker, store, results all rewound, and
        # the partial scenario tree wiped (the documented durability rule).
        assert ex.executor.scene_revision == 0
        assert ex.executor._worker.publication_watermarks == watermarks_before
        assert ex.executor.store.max_revision() == 0
        assert not list(ex.executor.results_root.rglob("rev-*"))
        assert not _scenario_root(ex.tmp_path).exists()

    def test_veg_batch_after_building_routes_chain_with_clean_worker(self, ex) -> None:
        # u-d4c routing contract: after a building publication ANY later
        # batch — vegetation included — rides the regeneration chain with
        # the FULL accumulated building fold, so the published building
        # survives it (the pre-u-d4c engine routed this batch through the
        # baseline-bound worker and silently reverted the massing). The
        # u-c6 watermark adoption contract still holds: the chain-routed
        # batch's replayed tree edits and staged overlays are CONSUMED,
        # leaving the worker with no phantom pending state.
        building = ex.executor.validate(_building_command())
        assert ex.executor.execute([building]).published
        assert ex.executor._worker.pending_batch() is None

        executed = ex.executor.execute(
            [ex.executor.validate(_veg_command(ex.executor, edit_id="veg-2"))]
        )
        assert executed.published
        assert executed.mode == "full", (
            "a vegetation batch after a published building edit must ride "
            "the regeneration chain so the building survives it (u-d4c)"
        )
        assert executed.diagnostics["building_batch"] is True
        assert executed.diagnostics["chain_carries_new_edits"] is False
        assert executed.diagnostics["building_fold_size"] == 1
        # No phantom pending state survived the chain publication: the
        # batch's replayed tree edit was consumed by the adoption.
        assert ex.executor._worker.pending_batch() is None

    def test_post_publish_failure_rolls_back_published_patch(self, ex, monkeypatch):
        # A failure AFTER the chain published its patch (at the store
        # publish) must remove the orphan rev directory (u-c1b M4) and
        # rewind everything, including the scenario root.
        from tests.test_incremental_remediation_uc1b import _fail_store_publish

        _fail_store_publish(monkeypatch, ex.executor)
        with pytest.raises(RuntimeError, match="store rejected"):
            ex.executor.execute([ex.executor.validate(_building_command())])
        assert ex.executor.scene_revision == 0
        assert not list(ex.executor.results_root.rglob("rev-*"))
        assert not _scenario_root(ex.tmp_path).exists()


class TestBuildingFoldSemantics:
    """u-d4c Evidence B: the executor-level fold's accumulation rules.

    The chain is stubbed (the ``ex`` harness) so these tests assert WHAT
    the executor hands the chain — the full accumulated fold — without
    physics. The scientific accumulation chains (fold carried into real
    results, bitwise) live in TestBuildingAccumulationDifferential.
    """

    @pytest.fixture()
    def ex(self, tmp_path: Path, monkeypatch):
        from solweig_gpu.incremental import executor as executor_mod
        from solweig_gpu.incremental.buildings import BuildingRasterResult
        from tests.test_incremental_lc_integration import _tiny_executor

        executor = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_chain(**kwargs):
            captured.update(kwargs)
            cache = executor.cache
            variables = tuple(kwargs["requested_variables"])
            zeros = {
                name: np.zeros(
                    (cache.time_steps, cache.rows, cache.cols), dtype=np.float32
                )
                for name in variables
            }
            return BuildingRegenerationResult(
                scenario_site_dir=(
                    executor.results_root / "default" / ".building" / "site"
                ),
                cache_dir=executor.results_root / "default" / ".building" / "cache",
                cache=cache,
                raster=np.zeros((cache.rows, cache.cols), dtype=np.float32),
                records=BuildingRasterResult(
                    raster=np.zeros((cache.rows, cache.cols), dtype=np.float32),
                    records=(),
                ),
                outputs=zeros,
                stages=("stage_scenario_site", "run_scenario_full_tile"),
            )

        monkeypatch.setattr(executor_mod, "regenerate_building_batch", fake_chain)
        return SimpleNamespace(executor=executor, captured=captured, tmp_path=tmp_path)

    def _run(self, executor: PlanExecutor, command: EditCommand):
        executed = executor.execute([executor.validate(command)])
        assert executed.published, "every fold-semantics batch must publish"
        return executed

    def _add(self, executor: PlanExecutor, state: dict, edit_id: str):
        return self._run(
            executor,
            _building_command(
                revision=executor.scene_revision, state=state, edit_id=edit_id
            ),
        )

    def test_fold_accumulates_across_two_building_batches(self, ex) -> None:
        """add A (published) -> add B (published): B's chain carries BOTH."""
        self._add(ex.executor, NEW_BLOCK_STATE, "bld-a")
        second = self._add(ex.executor, NEW_BLOCK_B_STATE, "bld-b")
        edits = ex.captured["edits"]
        assert [edit.building_id for edit in edits] == [
            "new-block",
            "new-block-b",
        ]
        assert all(edit.after is not None for edit in edits)
        assert second.diagnostics["building_fold_size"] == 2
        assert second.diagnostics["chain_carries_new_edits"] is True
        assert list(ex.executor._applied_massing_edits) == [
            "new-block",
            "new-block-b",
        ]

    def test_last_write_wins_update_after_add(self, ex) -> None:
        """add B(h=10) -> update B(h=22): one entry, after=h22, before=None."""
        self._add(ex.executor, NEW_BLOCK_B_STATE, "bld-b")
        raised = dict(NEW_BLOCK_B_STATE)
        raised["height_m"] = 22.0
        self._run(
            ex.executor,
            _building_command(
                revision=ex.executor.scene_revision,
                state=raised,
                old_state=NEW_BLOCK_B_STATE,
                edit_id="bld-b2",
            ),
        )
        edits = ex.captured["edits"]
        assert len(edits) == 1
        assert edits[0].building_id == "new-block-b"
        assert edits[0].before is None  # baseline anchor: never in baseline
        assert edits[0].after.height_m == 22.0

    def test_delete_session_added_keeps_tombstone(self, ex) -> None:
        """add A -> delete A: the fold KEEPS a tombstone (deletes stay
        deletes; an emptied fold would route the worker into its
        nothing-pending no-op gate and raise)."""
        self._add(ex.executor, NEW_BLOCK_STATE, "bld-a")
        deleted = self._run(
            ex.executor,
            _delete_command(
                revision=ex.executor.scene_revision,
                state=NEW_BLOCK_STATE,
                edit_id="bld-del",
            ),
        )
        assert deleted.mode == "full"  # the chain routed, not the worker
        fold = ex.executor._applied_massing_edits
        assert list(fold) == ["new-block"]
        tombstone = fold["new-block"]
        assert tombstone.before is not None
        assert tombstone.after is None
        chain_edits = ex.captured["edits"]
        assert len(chain_edits) == 1
        assert chain_edits[0].after is None

    def test_delete_baseline_id_stays_delete(self, ex) -> None:
        """delete of a BASELINE building anchors at the baseline spec."""
        self._run(
            ex.executor,
            _delete_command(
                revision=ex.executor.scene_revision,
                state=BASE_SITE_BLOCK_STATE,
                edit_id="bld-del-base",
            ),
        )
        fold = ex.executor._applied_massing_edits
        assert list(fold) == ["site-block"]
        assert fold["site-block"].before.height_m == 12.0
        assert fold["site-block"].after is None
        # A second, other-family batch still erases the baseline block on
        # re-stage: the delete rides every later chain run.
        met = _executor_met_edit(
            ex.executor,
            time_index=1,
            value=26.0,
            outputs=("utci", "tmrt", "shadow"),
            edit_id="met-2",
        )
        assert ex.executor.execute([met]).published
        edits = ex.captured["edits"]
        assert [edit.building_id for edit in edits] == ["site-block"]
        assert edits[0].after is None

    def test_readd_after_delete_folds_to_the_new_spec(self, ex) -> None:
        """add A -> delete A -> re-add A(h=18): net vs baseline is the
        re-added spec (the tombstone's footprint resets to DEM ground the
        baseline already carries, then the new spec paints)."""
        self._add(ex.executor, NEW_BLOCK_STATE, "bld-a")
        self._run(
            ex.executor,
            _delete_command(
                revision=ex.executor.scene_revision,
                state=NEW_BLOCK_STATE,
                edit_id="bld-del",
            ),
        )
        reborn = dict(NEW_BLOCK_STATE)
        reborn["height_m"] = 18.0
        self._add(ex.executor, reborn, "bld-a2")
        fold = ex.executor._applied_massing_edits
        assert len(fold) == 1
        assert fold["new-block"].after.height_m == 18.0

    def test_delete_a_after_add_a_and_b_folds_to_b_only_effect(
        self, ex
    ) -> None:
        """The lead's accumulation repro: add A, add B, delete A — the
        chain carries B's add and A's tombstone; nothing silently drops."""
        self._add(ex.executor, NEW_BLOCK_STATE, "bld-a")
        self._add(ex.executor, NEW_BLOCK_B_STATE, "bld-b")
        self._run(
            ex.executor,
            _delete_command(
                revision=ex.executor.scene_revision,
                state=NEW_BLOCK_STATE,
                edit_id="bld-del-a",
            ),
        )
        fold = ex.executor._applied_massing_edits
        assert fold["new-block"].after is None  # tombstone
        assert fold["new-block-b"].after is not None
        assert fold["new-block-b"].after.height_m == 10.0

    def test_overlapping_delete_diverges_from_session_order_at_overlap(
        self, ex, monkeypatch
    ) -> None:
        """u-d4c remediation F1 — the fold's OVERLAP-ORDER divergence,
        pinned with OVERLAPPING footprints (the suite's other fold tests
        use disjoint ones).

        The fold applies in dict FIRST-INSERTION order, so add A, add B
        overlapping A, delete A routes ``[A_tombstone(reset FA), B(paint
        FB)]``: the reset lands BEFORE B's paint, so B stays INTACT at
        FA∩FB. A session-order sequential oracle (``[A_add, B_add,
        A_delete]``) truncates B there instead — its reset lands LAST and
        the rasterizer's before-footprint reset is unconditional. The
        divergence is bounded to the overlap cells of the two distinct
        ids; deleting the LATER-inserted id instead (mirror) truncates A
        at the overlap in BOTH orderings, so the divergence is
        asymmetric. See the ORDER-AND-OVERLAP DISCLOSURE in
        ``_staged_massing_fold``."""
        from solweig_gpu.incremental import executor as executor_mod
        from solweig_gpu.incremental.buildings import (
            BuildingLayer,
            BuildingRasterResult,
            polygon_cell_mask,
        )
        from solweig_gpu.incremental.regenerate import _read_tile
        from tests.test_incremental_lc_integration import _tiny_executor

        a_state = {
            "building_id": "ov-a",
            "footprint_m": [list(vertex) for vertex in OVERLAP_A_FOOT],
            "height_m": 15.0,
        }
        b_state = {
            "building_id": "ov-b",
            "footprint_m": [list(vertex) for vertex in OVERLAP_B_FOOT],
            "height_m": 10.0,
        }
        self._add(ex.executor, a_state, "ov-a")
        self._add(ex.executor, b_state, "ov-b")
        self._run(
            ex.executor,
            _delete_command(
                revision=ex.executor.scene_revision,
                state=a_state,
                edit_id="ov-del-a",
            ),
        )

        # The chain received the fold in FIRST-INSERTION order: A's
        # tombstone FIRST (its dict slot was created by the add), then
        # B's add.
        edits = ex.captured["edits"]
        assert [edit.building_id for edit in edits] == ["ov-a", "ov-b"]
        assert edits[0].before is not None and edits[0].after is None
        assert edits[1].before is None and edits[1].after.height_m == 10.0

        # Rasterize BOTH orderings against the baseline tiles exactly as
        # the chain's stage 2 (write_scenario_building_dsm) does.
        executor = ex.executor
        base, _gt, _wkt, _nd, _ = _read_tile(
            executor.site_dir, "Building_DSM", executor.cache.tile_key
        )
        dem, *_ = _read_tile(
            executor.site_dir, "DEM", executor.cache.tile_key
        )
        grid = executor.grid

        def rasterize(edits_tuple):
            layer = BuildingLayer(base, dem, grid)
            layer.apply_edits(edits_tuple)
            return layer.rasterize()

        fold_raster = rasterize(tuple(edits))
        session_raster = rasterize(
            (
                _massing_edit(a_state),
                _massing_edit(b_state),
                _massing_edit(a_state, deleted=True),
            )
        )

        mask_a, _ = polygon_cell_mask(OVERLAP_A_FOOT, grid, grid.full_window)
        mask_b, _ = polygon_cell_mask(OVERLAP_B_FOOT, grid, grid.full_window)
        overlap = mask_a & mask_b
        assert overlap.any(), "fixture error: the footprints must overlap"
        ground = np.float32(0.0)  # the tiny site's flat DEM

        # FOLD semantics: B intact at the overlap (its paint lands after
        # the tombstone's reset).
        assert np.all(fold_raster[overlap] == ground + np.float32(10.0))
        # SESSION-order sequential oracle: B truncated at the overlap
        # (the delete's unconditional reset lands last).
        assert np.all(session_raster[overlap] == ground)
        # Outside the overlap the two orderings agree — the divergence
        # is bounded to the overlap cells of the distinct ids.
        assert np.array_equal(
            fold_raster[~overlap], session_raster[~overlap]
        ), "the fold/session divergence must be bounded to overlap cells"

        # Mirror: delete the LATER-inserted id — fold [A, B_tombstone]
        # truncates A at the overlap too (the reset lands AFTER A's
        # paint), so both orderings agree everywhere here; the divergence
        # above is the asymmetric one.
        mirror_executor = _tiny_executor(ex.tmp_path / "mirror")
        captured_mirror: dict = {}

        def mirror_stub(**kwargs):
            captured_mirror.update(kwargs)
            cache = mirror_executor.cache
            variables = tuple(kwargs["requested_variables"])
            zeros = {
                name: np.zeros(
                    (cache.time_steps, cache.rows, cache.cols), dtype=np.float32
                )
                for name in variables
            }
            return BuildingRegenerationResult(
                scenario_site_dir=(
                    mirror_executor.results_root / "default" / ".building" / "site"
                ),
                cache_dir=(
                    mirror_executor.results_root / "default" / ".building" / "cache"
                ),
                cache=cache,
                raster=np.zeros((cache.rows, cache.cols), dtype=np.float32),
                records=BuildingRasterResult(
                    raster=np.zeros((cache.rows, cache.cols), dtype=np.float32),
                    records=(),
                ),
                outputs=zeros,
                stages=("stage_scenario_site", "run_scenario_full_tile"),
            )

        assert executor_mod.regenerate_building_batch is not None
        monkeypatch.setattr(executor_mod, "regenerate_building_batch", mirror_stub)
        for state, edit_id in ((a_state, "ov-a"), (b_state, "ov-b")):
            executed = mirror_executor.execute(
                [
                    mirror_executor.validate(
                        _building_command(
                            revision=mirror_executor.scene_revision,
                            state=state,
                            edit_id=edit_id,
                        )
                    )
                ]
            )
            assert executed.published
        executed = mirror_executor.execute(
            [
                mirror_executor.validate(
                    _delete_command(
                        revision=mirror_executor.scene_revision,
                        state=b_state,
                        edit_id="ov-del-b",
                    )
                )
            ]
        )
        assert executed.published

        mirror_edits = captured_mirror["edits"]
        assert [edit.building_id for edit in mirror_edits] == ["ov-a", "ov-b"]
        assert mirror_edits[1].after is None  # B's tombstone is LAST
        mirror_fold_raster = rasterize(tuple(mirror_edits))
        mirror_session_raster = rasterize(
            (
                _massing_edit(a_state),
                _massing_edit(b_state),
                _massing_edit(b_state, deleted=True),
            )
        )
        assert np.all(mirror_fold_raster[overlap] == ground)
        assert np.array_equal(
            mirror_fold_raster[~overlap], mirror_session_raster[~overlap]
        )
        assert np.array_equal(mirror_fold_raster, mirror_session_raster), (
            "deleting the later-inserted id must agree with session order"
        )

    def test_failed_chain_routed_batch_leaves_fold_untouched(self, ex, monkeypatch) -> None:
        """A met batch after a published building edit fails mid-chain:
        the published fold, revision, and overlays all stay put."""
        from solweig_gpu.incremental import executor as executor_mod

        self._add(ex.executor, NEW_BLOCK_STATE, "bld-a")
        fold_before = dict(ex.executor._applied_massing_edits)
        revision_before = ex.executor.scene_revision
        forcing_before = ex.executor._applied_forcing_overlay

        def exploding(**kwargs):
            raise RegenerationError("regeneration stage 'regenerate_svf' failed: boom")

        monkeypatch.setattr(executor_mod, "regenerate_building_batch", exploding)
        with pytest.raises(ExecutorError, match="regeneration chain failed"):
            ex.executor.execute(
                [
                    _executor_met_edit(
                        ex.executor,
                        time_index=1,
                        value=26.0,
                        outputs=("utci", "tmrt", "shadow"),
                        edit_id="met-2",
                    )
                ]
            )
        assert ex.executor._applied_massing_edits == fold_before
        assert ex.executor.scene_revision == revision_before
        assert ex.executor._applied_forcing_overlay == forcing_before
        assert not _scenario_root(ex.tmp_path).exists()

    def test_value_noop_batch_with_fold_stays_noop(self, ex) -> None:
        """A batch folding VALUE-equal to the published state is still
        intercepted whole-batch — the fold does not re-arm recomputes."""
        self._add(ex.executor, NEW_BLOCK_STATE, "bld-a")
        met = _executor_met_edit(
            ex.executor,
            time_index=1,
            value=26.0,
            outputs=("utci", "tmrt", "shadow"),
            edit_id="met-2",
        )
        assert ex.executor.execute([met]).published
        revision = ex.executor.scene_revision
        # Same met edit again: value-equal to the published forcing — a
        # no-op, revision does not advance, chain is not invoked.
        ex.captured.clear()
        repeat = ex.executor.execute(
            [
                _executor_met_edit(
                    ex.executor,
                    time_index=1,
                    value=26.0,
                    outputs=("utci", "tmrt", "shadow"),
                    edit_id="met-3",
                )
            ]
        )
        assert repeat.status == "no-op"
        assert ex.executor.scene_revision == revision
        assert ex.captured == {}

    def test_fresh_scenario_never_calls_the_chain(self, ex, monkeypatch) -> None:
        """Routing identity: with NO published building edits the fold is
        empty and a vegetation batch routes the worker byte-identically to
        pre-u-d4c (the chain is never invoked)."""
        from solweig_gpu.incremental import executor as executor_mod
        from tests.test_incremental_met_integration import _stub_patch

        def forbidden(**kwargs):
            raise AssertionError("the chain must not run for a building-free scenario")

        monkeypatch.setattr(executor_mod, "regenerate_building_batch", forbidden)
        executor = ex.executor

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            return [
                _stub_patch(("utci",), job_id, revision, window, mode="local")
                for window in write_windows
            ]

        def fake_full(job_id, revision, forcing):
            return [
                _stub_patch(
                    ("utci",), job_id, revision, executor.grid.full_window, mode="full"
                )
            ]

        executor._worker._solve_local = fake_local
        executor._worker._solve_full = fake_full
        executed = executor.execute(
            [executor.validate(_veg_command(executor, edit_id="veg-1"))]
        )
        assert executed.published
        # LOCAL_ALWAYS policy keeps a tree's influence window below the
        # full-recompute threshold: the WINDOWED worker path, not a chain.
        assert executed.mode == "local"
        assert "building_batch" not in executed.diagnostics
        assert executor._applied_massing_edits == {}


# ---------------------------------------------------------------------------
# Scientific suite: real physics, executor vs oracle differential
# ---------------------------------------------------------------------------


@pytest.fixture()
def executor(baseline, tmp_path: Path) -> PlanExecutor:
    return PlanExecutor(
        cache=baseline.cache,
        layer=TreeLayer(baseline.cache.tree_base, baseline.grid),
        site_dir=baseline.site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )


@pytest.fixture(scope="module")
def plain_oracle(baseline, tmp_path_factory) -> BuildingRegenerationResult:
    """Independent no-overlay oracle: raise the baseline block + add one."""
    from solweig_gpu.incremental.regenerate import regenerate_building_batch

    return regenerate_building_batch(
        edits=_chain_edits(),
        baseline_site_dir=baseline.site,
        baseline_cache=baseline.cache,
        scenario_root=tmp_path_factory.mktemp("uc6_oracle_plain") / "scenario",
        selected_date_str=DATE_STR,
    )


@pytest.fixture(scope="module")
def add_only_oracle(baseline, tmp_path_factory) -> BuildingRegenerationResult:
    """Independent no-overlay oracle over the single-add edit set."""
    from solweig_gpu.incremental.regenerate import regenerate_building_batch

    return regenerate_building_batch(
        edits=_add_only_edits(),
        baseline_site_dir=baseline.site,
        baseline_cache=baseline.cache,
        scenario_root=tmp_path_factory.mktemp("uc6_oracle_add") / "scenario",
        selected_date_str=DATE_STR,
    )


class TestBuildingExecutorScientific:
    def test_building_batch_matches_oracle_bitwise(
        self, baseline, executor, plain_oracle
    ) -> None:
        raise_block = executor.validate(
            _building_command(
                state=RAISED_SITE_BLOCK_STATE,
                old_state=BASE_SITE_BLOCK_STATE,
                edit_id="bld-raise",
            )
        )
        add_block = executor.validate(_building_command(edit_id="bld-add"))
        executed = executor.execute([raise_block, add_block])
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 1

        # BITWISE differential vs the independently-run oracle (same
        # committed massing edits, no overlays, baseline vegetation).
        patch = load_patch(executed.patch_paths[0])
        for name in ("utci", "tmrt", "shadow"):
            assert _bitwise(patch.arrays[name], plain_oracle.outputs[name]), (
                f"{name}: executor building results diverged from the "
                "regenerate_building_batch oracle"
            )

        # Store coverage at the publication revision, full mode, all times.
        assert executor.store.available_times("utci") == (0, 1, 2)
        for time_index in range(3):
            assert executor.store.revision_at("utci", time_index) == 1
            assert executor.store.lookup("utci", time_index).mode == "full"

        # Patch provenance: the SCENARIO cache the chain rebuilt, not the
        # baseline cache.
        assert patch.mode == "full"
        assert patch.write_window == executor.grid.full_window
        assert patch.cache_manifest_sha256 != baseline.cache.metadata()[
            "manifest_sha256"
        ]
        assert set(patch.variables) == {"utci", "tmrt", "shadow"}

        # Baseline immutability (hash-verified, both site and cache).
        assert _hash_tree(baseline.site) == baseline.site_hashes
        assert _hash_tree(baseline.cache_dir) == baseline.cache_hashes

    def test_met_then_building_reflects_both_edits(
        self, baseline, executor, tmp_path, add_only_oracle
    ) -> None:
        from solweig_gpu.incremental.regenerate import regenerate_building_batch

        # 1) Publish a met edit first (real worker, full tile).
        met = _executor_met_edit(executor, time_index=1, value=26.0)
        met_executed = executor.execute([met])
        assert met_executed.published
        applied = executor._applied_forcing_overlay
        assert applied is not None
        assert applied.changes[0].after_value == 26.0

        # 2) The building batch carries the ACCUMULATED overlay: its
        #    results reflect BOTH the met edit and the massing edit.
        building = executor.validate(
            _building_command(revision=executor.scene_revision)
        )
        executed = executor.execute([building])
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 2

        oracle_with_met = regenerate_building_batch(
            edits=_add_only_edits(),
            baseline_site_dir=baseline.site,
            baseline_cache=baseline.cache,
            scenario_root=tmp_path / "oracle_met" / "scenario",
            selected_date_str=DATE_STR,
            forcing_overlay=applied,
        )
        patch = load_patch(executed.patch_paths[0])
        for name in ("utci", "tmrt", "shadow"):
            assert _bitwise(patch.arrays[name], oracle_with_met.outputs[name]), (
                f"{name}: building batch did not carry the published met "
                "overlay bitwise"
            )
        # Differential evidence the overlay MATTERED: the same building
        # edit WITHOUT the met overlay produces different products.
        assert not _bitwise(
            patch.arrays["utci"], add_only_oracle.outputs["utci"]
        ), (
            "the met overlay changed nothing on the building batch's "
            "results; the differential is vacuous"
        )

    def test_mid_chain_failure_rolls_back_everything(
        self, baseline, executor, monkeypatch
    ) -> None:
        import solweig_gpu.solweig_gpu as pipeline

        watermarks_before = executor._worker.publication_watermarks

        def boom(*args, **kwargs):
            raise RuntimeError("SVF exploded")

        monkeypatch.setattr(pipeline, "calculate_svf", boom)
        with pytest.raises(ExecutorError, match="regenerate_svf") as info:
            executor.execute([executor.validate(_building_command())])
        assert "regenerate_walls_aspect" in str(info.value)
        # u-c1b wholesale rollback + the u-c6 scenario-root wipe.
        assert executor.scene_revision == 0
        assert executor._worker.publication_watermarks == watermarks_before
        assert executor.store.max_revision() == 0
        assert not list(executor.results_root.rglob("rev-*"))
        assert not (executor.results_root / "default" / ".building").exists()
        assert _hash_tree(baseline.site) == baseline.site_hashes
        assert _hash_tree(baseline.cache_dir) == baseline.cache_hashes

        # The retry on a healthy chain publishes.
        monkeypatch.undo()
        retry = executor.execute(
            [executor.validate(_building_command(edit_id="bld-retry"))]
        )
        assert retry.published
        assert retry.mode == "full"
        assert retry.scene_revision == 1

    def test_walllimit_and_edge_disclosures_surface(self, baseline, executor) -> None:
        # A block BELOW the wall threshold (3 m) and one TOUCHING the
        # raster edge: both disclosures must reach the executed plan's
        # diagnostics, never be silently dropped.
        low = executor.validate(
            _building_command(
                state={
                    "building_id": "low-block",
                    "footprint_m": [list(vertex) for vertex in NEW_BLOCK_FOOT],
                    "height_m": 2.5,  # < WALL_LIMIT_M (3.0)
                },
                edit_id="bld-low",
            )
        )
        edge = executor.validate(
            _building_command(
                state={
                    "building_id": "edge-block",
                    "footprint_m": [
                        [TINY_ORIGIN[0], TINY_ORIGIN[1]],
                        [TINY_ORIGIN[0] + 40.0, TINY_ORIGIN[1]],
                        [TINY_ORIGIN[0] + 40.0, TINY_ORIGIN[1] - 40.0],
                        [TINY_ORIGIN[0], TINY_ORIGIN[1] - 40.0],
                    ],
                    "height_m": 10.0,
                },
                edit_id="bld-edge",
            )
        )
        executed = executor.execute([low, edge])
        assert executed.published
        disclosures = executed.diagnostics["building_disclosures"]
        assert any("wall threshold" in note for note in disclosures), disclosures
        assert any("raster edge" in note for note in disclosures), disclosures

    def test_worker_state_clean_after_building_publication(
        self, baseline, executor
    ) -> None:
        # After a building publication the worker must carry NO phantom
        # pending state: the out-of-band publication is adopted, so the
        # staged overlays equal the published watermarks and nothing is
        # pending (a stale "pending" overlay would force every later job
        # full-tile). On this 48x48 site a tree's influence windows can
        # legitimately cover the whole tile, so ROUTING is asserted on the
        # fast suite; here the watermark mechanism is asserted directly.
        building = executor.validate(_building_command())
        assert executor.execute([building]).published
        worker = executor._worker
        assert worker.pending_batch() is None
        assert worker._forcing_overlay == worker._published_forcing_overlay
        assert worker._landcover_overlay == worker._published_landcover_overlay

        _stub_local_every_window(executor)
        executed = executor.execute([executor.validate(_veg_command(executor))])
        assert executed.published
        assert executed.diagnostics["forcing_overlay_pending"] is False
        assert executed.diagnostics["landcover_overlay_pending"] is False


# ---------------------------------------------------------------------------
# u-d4c scientific suite: building edits ACCUMULATE across batches
# ---------------------------------------------------------------------------


class TestBuildingAccumulationDifferential:
    """u-d4c (lead-confirmed defect): the published building state must
    survive EVERY later batch, in every family.

    Pre-u-d4c engine defect (pre-existing since the u-c5/u-c6 chain seam,
    exposed user-facing by the u-d4 universal transport): ONLY batches
    carrying new building deltas routed through
    ``PlanExecutor._regenerate_building_chain``, and the chain staged the
    scenario site from BASELINE with just that batch's edits. Any solve
    AFTER a published building edit ran against the BASELINE arrays and
    silently reverted the building — including a SECOND building batch
    (add A published, then add B: B's chain re-staged from baseline and
    published B WITHOUT A).

    Each chain below publishes a building edit first, then a second-family
    edit; the second batch's published products must match a twin
    full-domain oracle (an independently-run ``regenerate_building_batch``
    carrying BOTH the building and the second family's edit) BITWISE —
    never a relaxed tolerance — and must demonstrably differ from the
    same second family applied to the building-free baseline (the
    differential is not vacuous).
    """

    def _publish_building(
        self, executor: PlanExecutor, *, state: dict, edit_id: str
    ):
        executed = executor.execute(
            [
                executor.validate(
                    _building_command(
                        revision=executor.scene_revision,
                        state=state,
                        edit_id=edit_id,
                    )
                )
            ]
        )
        assert executed.published, "the building batch itself must publish"
        assert executed.mode == "full"
        return executed

    def _second_patch(self, executor: PlanExecutor, executed):
        """The second batch must publish FULL tile from the chain."""
        assert executed.published, "the second-family batch must publish"
        assert executed.mode == "full", (
            "a batch after a published building edit must ride the "
            "regeneration chain (full tile), never the baseline-bound "
            "worker path"
        )
        return load_patch(executed.patch_paths[0])

    def _twin(
        self,
        baseline,
        tmp_path: Path,
        *,
        edits,
        forcing_overlay=None,
        landcover_overlay=None,
        model_parameters=None,
        tree_layer=None,
    ):
        from solweig_gpu.incremental.regenerate import regenerate_building_batch

        return regenerate_building_batch(
            edits=edits,
            baseline_site_dir=baseline.site,
            baseline_cache=baseline.cache,
            scenario_root=tmp_path / "twin" / "scenario",
            selected_date_str=DATE_STR,
            forcing_overlay=forcing_overlay,
            landcover_overlay=landcover_overlay,
            model_parameters=model_parameters,
            tree_layer=tree_layer,
        )

    def _baseline_full_run(
        self,
        baseline,
        tmp_path: Path,
        *,
        overlay=None,
        landcover_overlay=None,
        model_parameters=None,
        layer=None,
    ):
        """The building-free twin: the standard full-domain oracle seams."""
        from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile

        return run_full_tile(
            baseline.cache,
            layer if layer is not None else TreeLayer(
                baseline.cache.tree_base, baseline.grid
            ),
            forcing=load_site_forcing(
                baseline.cache,
                site_dir=baseline.site,
                selected_date_str=DATE_STR,
                overlay=overlay,
            ),
            site_dir=baseline.site,
            scratch_dir=tmp_path / "baseline_scratch",
            requested_variables=("utci", "tmrt", "shadow"),
            landcover_overlay=landcover_overlay,
            model_parameters=model_parameters,
        )

    def test_building_then_building_keeps_both(
        self, baseline, executor, tmp_path, add_only_oracle
    ) -> None:
        """add A (published) -> add B (published): B's results carry BOTH."""
        self._publish_building(
            executor, state=NEW_BLOCK_STATE, edit_id="bld-a"
        )
        second = executor.execute(
            [
                executor.validate(
                    _building_command(
                        revision=executor.scene_revision,
                        state=NEW_BLOCK_B_STATE,
                        edit_id="bld-b",
                    )
                )
            ]
        )
        patch = self._second_patch(executor, second)
        twin = self._twin(
            baseline,
            tmp_path,
            edits=_add_edits(NEW_BLOCK_STATE, NEW_BLOCK_B_STATE),
        )
        for name in ("utci", "tmrt", "shadow"):
            assert _bitwise(patch.arrays[name], twin.outputs[name]), (
                f"{name}: the second building batch did not carry the "
                "first published building bitwise"
            )
        # Non-vacuous: B changes the twin vs A alone.
        assert not _bitwise(
            twin.outputs["utci"], add_only_oracle.outputs["utci"]
        ), "block B changed nothing; the differential is vacuous"
        # Baseline immutability holds across BOTH chains.
        assert _hash_tree(baseline.site) == baseline.site_hashes
        assert _hash_tree(baseline.cache_dir) == baseline.cache_hashes

    def test_building_then_met_keeps_building(
        self, baseline, executor, tmp_path
    ) -> None:
        from solweig_gpu.incremental.adapters.met_time import overlay_from_deltas

        self._publish_building(
            executor, state=NEW_BLOCK_STATE, edit_id="bld-a"
        )
        met = _executor_met_edit(
            executor,
            time_index=1,
            value=26.0,
            outputs=("utci", "tmrt", "shadow"),
            edit_id="met-2",
        )
        executed = executor.execute([met])
        patch = self._second_patch(executor, executed)
        overlay = overlay_from_deltas([met.delta])
        twin = self._twin(
            baseline, tmp_path, edits=_add_only_edits(), forcing_overlay=overlay
        )
        for name in ("utci", "tmrt", "shadow"):
            assert _bitwise(patch.arrays[name], twin.outputs[name]), (
                f"{name}: the met batch after a building edit dropped the "
                "building from the published scene"
            )
        # Non-vacuous BOTH ways: the met edit matters, and the building
        # matters (the met-only baseline run differs from the published
        # building+met products).
        met_only = self._baseline_full_run(baseline, tmp_path, overlay=overlay)
        assert not _bitwise(patch.arrays["utci"], met_only["utci"]), (
            "the building changed nothing on the met batch's results; the "
            "differential is vacuous"
        )

    def test_building_then_params_keeps_building(
        self, baseline, executor, tmp_path
    ) -> None:
        self._publish_building(
            executor, state=NEW_BLOCK_STATE, edit_id="bld-a"
        )
        params = executor.validate(
            EditCommand(
                edit_id="params-2",
                scenario_id=executor._scenario_id,
                base_scene_revision=executor.scene_revision,
                adapter_id="model_receptor_parameters",
                operation="update",
                old_state=None,
                new_state={"albedo_b": 0.35},
                requested_outputs=("utci", "tmrt", "shadow"),
                requested_times=(1,),
            )
        )
        executed = executor.execute([params])
        patch = self._second_patch(executor, executed)
        twin = self._twin(
            baseline,
            tmp_path,
            edits=_add_only_edits(),
            model_parameters={"albedo_b": 0.35},
        )
        for name in ("utci", "tmrt", "shadow"):
            assert _bitwise(patch.arrays[name], twin.outputs[name]), (
                f"{name}: the params batch after a building edit dropped "
                "the building from the published scene"
            )
        params_only = self._baseline_full_run(
            baseline, tmp_path, model_parameters={"albedo_b": 0.35}
        )
        assert not _bitwise(patch.arrays["utci"], params_only["utci"]), (
            "the building changed nothing on the params batch's results; "
            "the differential is vacuous"
        )

    def test_building_then_veg_keeps_building(
        self, baseline, executor, tmp_path
    ) -> None:
        self._publish_building(
            executor, state=NEW_BLOCK_STATE, edit_id="bld-a"
        )
        veg = executor.validate(_veg_command(executor, edit_id="veg-2"))
        executed = executor.execute([veg])
        patch = self._second_patch(executor, executed)
        # The twin's tree layer carries the same tree the batch replayed
        # (the tree spec is client-declared scenario INPUT state, taken
        # from the live layer — never executor routing state).
        twin_layer = TreeLayer(baseline.cache.tree_base, baseline.grid)
        twin_layer.add_tree(executor.layer.current_trees()[0])
        twin = self._twin(
            baseline,
            tmp_path,
            edits=_add_only_edits(),
            tree_layer=twin_layer,
        )
        assert patch.variables == ("utci",)
        assert _bitwise(patch.arrays["utci"], twin.outputs["utci"]), (
            "utci: the vegetation batch after a building edit dropped the "
            "building from the published scene"
        )
        veg_only = self._baseline_full_run(baseline, tmp_path, layer=twin_layer)
        assert not _bitwise(patch.arrays["utci"], veg_only["utci"]), (
            "the building changed nothing on the vegetation batch's "
            "results; the differential is vacuous"
        )

    def test_building_then_landcover_keeps_building(
        self, baseline, executor, tmp_path
    ) -> None:
        self._publish_building(
            executor, state=NEW_BLOCK_STATE, edit_id="bld-a"
        )
        # The baseline class grid is uniformly 1, so class 5 is a real
        # value change (a same-class repaint would be intercepted as a
        # whole-batch no-op).
        paint = executor.validate(
            _paint_command(
                revision=executor.scene_revision,
                window=PAINT_WINDOW,
                classes=5,
                outputs=("utci", "tmrt", "shadow"),
                edit_id="lc-2",
            )
        )
        executed = executor.execute([paint])
        patch = self._second_patch(executor, executed)
        overlay = _lc_overlay(baseline.cache, window=PAINT_WINDOW, classes=5)
        twin = self._twin(
            baseline,
            tmp_path,
            edits=_add_only_edits(),
            landcover_overlay=overlay,
        )
        for name in ("utci", "tmrt", "shadow"):
            assert _bitwise(patch.arrays[name], twin.outputs[name]), (
                f"{name}: the land-cover batch after a building edit "
                "dropped the building from the published scene"
            )
        lc_only = self._baseline_full_run(
            baseline, tmp_path, landcover_overlay=overlay
        )
        assert not _bitwise(patch.arrays["utci"], lc_only["utci"]), (
            "the building changed nothing on the land-cover batch's "
            "results; the differential is vacuous"
        )
