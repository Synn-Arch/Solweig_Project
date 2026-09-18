# SPDX-License-Identifier: GPL-3.0-only
"""T25 gates: spatial/COW checkpoint warm start (DESIGN.ko.md 11/§571-§579).

The whole-tile fingerprint refuses warm start on ANY geometry edit. This
file proves the chunk-granular margin the design hypothesizes:

* capture — per-chunk composed-scene/landcover digests + march globals
  (schema 3), with v2 checkpoints loading unchanged (behavior-preserving
  migration, the T10 deferral's obligation);
* plan — the clean-chunk predicate with reach dilation, the amplitude
  ANY-crossing global fence (the T22 regime-widening template), the
  landcover no-dilation rule, and the scalar-side hard refusal (a met or
  parameter edit invalidates the carried scalars themselves — they cannot
  be chunk-mixed);
* compose — checkpoint bytes on clean chunks, cold sub-window replay
  bytes on dirty chunks, bit-identical to a fresh full rebuild (the
  elementwise-chain argument), with typed refusals when the replay does
  not cover the plan;
* oracle — the theorem itself, on the REAL physics loop
  (``solve_window`` over prepared sites): a canopy edit leaves thermal
  planes bit-identical outside its reach-dilated neighborhood, so the
  COW splice equals a fresh rebuild AND the warm continuation from the
  spliced state equals the cold suffix — while an understated reach
  provably serves stale bytes (the RED witness that makes the dilation
  fence load-bearing, not decorative).

Hard contract pinned here: applicability is DIGEST-ONLY. Scene revisions
never enter the planner — two records that differ only in revision plan
identically — and the svf store's own chunk state is not an input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from solweig_gpu.incremental import checkpoints as cp
from solweig_gpu.incremental.checkpoints import (
    CHUNK_CELLS,
    LOADABLE_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    CheckpointError,
    CheckpointRecord,
    capture_spatial_cow,
    chunk_scene_digests,
    compose_chunk_cow_state,
    load_checkpoint,
    plan_chunk_cow_warm_start,
    thermal_checkpoint,
    thermal_fingerprint,
    write_checkpoint,
)
from solweig_gpu.incremental.geometry import RasterWindow, TreeSpec
from solweig_gpu.incremental.veg_svf_state import _CHUNK_CELLS as VEG_CHUNK_CELLS

# ---------------------------------------------------------------------------
# Synthetic fixtures (fast classes)
# ---------------------------------------------------------------------------

ROWS, COLS = 200, 130  # 4x3 chunk grid at 64 cells, ragged edge chunks
AMPL = 12.0  # effective amplitude (amaxvalue - min(a)), fixture constant
AMAX = 12.0  # global stop: the 12 m block dominates every tree
SCALE = 0.5  # cells per meter (pixel 2.0 m)
REACH_D1 = 58  # 58 // 64 + 1 = 1
REACH_D3 = 130  # 130 // 64 + 1 = 3


def _scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    dem = np.zeros((ROWS, COLS), dtype=np.float32)
    building = dem.copy()
    building[16:40, 8:26] = 12.0  # the tallest structure, like the worker site
    canopy = (rng.random((ROWS, COLS)) > 0.98).astype(np.float32) * 3.0
    landcover = np.ones((ROWS, COLS), dtype=np.uint8)
    landcover[60:80, 40:52] = 3  # a water class block
    return building, canopy, dem, landcover


def _met(rows: int = 3) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random((rows, 6)).astype(np.float64)


def _fingerprint(scene, met, params=None) -> dict[str, str]:
    building, canopy, dem, landcover = scene
    return thermal_fingerprint(
        building_dsm=building,
        canopy=canopy,
        dem=dem,
        resolved_landcover=landcover,
        model_parameters=params,
        met_prefix=met,
    )


def _cow(scene, *, chunk_cells: int = CHUNK_CELLS) -> dict:
    building, canopy, dem, landcover = scene
    return capture_spatial_cow(
        building_dsm=building,
        canopy=canopy,
        dem=dem,
        resolved_landcover=landcover,
        march_amplitude=AMPL,
        amaxvalue=AMAX,
        scale=SCALE,
        chunk_cells=chunk_cells,
    )


def _state(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.standard_normal((ROWS, COLS)).astype(np.float32)
        for name in cp.THERMAL_TENSOR_NAMES
    }


def _write_checkpoint(
    tmp_path: Path,
    scene,
    *,
    chunk_cells: int = CHUNK_CELLS,
    revision: int = 3,
    fingerprint: dict[str, str] | None = None,
):
    met = _met()
    record = CheckpointRecord(
        scene_revision=revision,
        coverage_manifest=(),
        thermal_tensors=_state(),
        thermal_scalars={"CI": 1.0, "firstdaytime": 0.0, "timeadd": 2.0},
        next_step=2,
        input_fingerprint=_fingerprint(scene, met)
        if fingerprint is None
        else fingerprint,
        spatial_cow=_cow(scene, chunk_cells=chunk_cells),
    )
    root = tmp_path / "checkpoints"
    return write_checkpoint(record, root), met


def _plan(record, scene, *, reach_pixels: int = REACH_D1,
          chunk_cells: int = CHUNK_CELLS, met: np.ndarray | None = None,
          params=None, amplitude: float = AMPL, amaxvalue: float = AMAX,
          scale: float = SCALE):
    building, canopy, dem, landcover = scene
    return plan_chunk_cow_warm_start(
        record,
        building_dsm=building,
        canopy=canopy,
        dem=dem,
        resolved_landcover=landcover,
        model_parameters=params,
        met_prefix=_met() if met is None else met,
        march_amplitude=amplitude,
        amaxvalue=amaxvalue,
        scale=scale,
        reach_pixels=reach_pixels,
        chunk_cells=chunk_cells,
    )


def _edit_chunk(scene, cr: int, cc: int, *, landcover: bool = False):
    building, canopy, dem, lc = scene
    scene = (
        building,
        canopy,
        dem,
        lc.copy() if landcover else lc,
    )
    if landcover:
        scene[3][cr * 64 : (cr + 1) * 64, cc * 64 : (cc + 1) * 64] ^= 1
    else:
        canopy = scene[1].copy()
        canopy[cr * 64 : (cr + 1) * 64, cc * 64 : (cc + 1) * 64] += 1.0
        scene = (building, canopy, dem, scene[3])
    return scene


class TestADigestSpelling:
    def test_deterministic_same_inputs_same_digests(self) -> None:
        scene = _scene()
        first = chunk_scene_digests(
            building_dsm=scene[0], canopy=scene[1], dem=scene[2],
            resolved_landcover=scene[3],
        )
        second = chunk_scene_digests(
            building_dsm=scene[0], canopy=scene[1], dem=scene[2],
            resolved_landcover=scene[3],
        )
        assert first == second

    def test_single_chunk_scene_edit_flips_exactly_that_chunk(self) -> None:
        scene = _scene()
        before = _cow(scene)
        after = _cow(_edit_chunk(scene, 2, 1))
        changed = [
            key
            for key in before["scene"]
            if before["scene"][key] != after["scene"][key]
        ]
        assert changed == ["2,1"]
        # Landcover untouched by a canopy edit.
        assert before["landcover"] == after["landcover"]

    def test_landcover_edit_flips_only_its_own_chunk_no_dilation_input(
        self,
    ) -> None:
        scene = _scene()
        before = _cow(scene)
        after = _cow(_edit_chunk(scene, 3, 2, landcover=True))
        assert before["scene"] == after["scene"]
        changed = [
            key
            for key in before["landcover"]
            if before["landcover"][key] != after["landcover"][key]
        ]
        assert changed == ["3,2"]

    def test_ragged_grid_counts_every_chunk(self) -> None:
        cow = _cow(_scene())
        # 200x130 at 64 cells -> ceil(200/64) x ceil(130/64) = 4x3.
        assert (cow["chunk_rows"], cow["chunk_cols"]) == (4, 3)
        assert len(cow["scene"]) == 12 == len(cow["landcover"])

    def test_absent_landcover_is_a_stable_spelling(self, tmp_path: Path) -> None:
        building, canopy, dem, _ = scene = _scene()
        record = CheckpointRecord(
            scene_revision=1,
            coverage_manifest=(),
            thermal_tensors=_state(),
            thermal_scalars={"CI": 1.0},
            next_step=1,
            input_fingerprint=_fingerprint(scene, _met()),
            spatial_cow=capture_spatial_cow(
                building_dsm=building,
                canopy=canopy,
                dem=dem,
                resolved_landcover=None,
                march_amplitude=AMPL,
                amaxvalue=AMAX,
                scale=SCALE,
            ),
        )
        loaded = load_checkpoint(write_checkpoint(record, tmp_path / "cp"))
        assert loaded.spatial_cow is not None
        assert loaded.spatial_cow["landcover"] is None

    def test_chunk_cells_matches_the_svf_store_grid(self) -> None:
        # Drift fence: the checkpoint chunk grid and the SVF store's chunk
        # grid are the SAME grid by design — a silent divergence would make
        # the two COW layers talk past each other.
        assert CHUNK_CELLS == VEG_CHUNK_CELLS == 64

    def test_validator_refuses_wrong_digest_count(self, tmp_path: Path) -> None:
        scene = _scene()
        cow = _cow(scene)
        del cow["scene"]["0,0"]
        record = CheckpointRecord(
            scene_revision=1,
            coverage_manifest=(),
            thermal_tensors=_state(),
            thermal_scalars={"CI": 1.0},
            next_step=2,
            input_fingerprint=_fingerprint(scene, _met()),
            spatial_cow=cow,
        )
        with pytest.raises(CheckpointError, match="11 digest"):
            write_checkpoint(record, tmp_path / "cp2")


class TestBSchemaMigration:
    def test_v3_roundtrip_carries_spatial_block_bitwise(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, _ = _write_checkpoint(tmp_path, scene)
        loaded = load_checkpoint(revision_dir)
        assert loaded.schema_version == 3
        assert loaded.spatial_cow is not None
        expected = _cow(scene)
        assert loaded.spatial_cow["scene"] == expected["scene"]
        assert loaded.spatial_cow["landcover"] == expected["landcover"]
        assert loaded.spatial_cow["march_amplitude_bits"] == expected["march_amplitude_bits"]
        assert loaded.spatial_cow["amaxvalue_bits"] == expected["amaxvalue_bits"]
        assert loaded.spatial_cow["scale_bits"] == expected["scale_bits"]

    def test_v2_checkpoint_loads_behavior_preserving(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, _ = _write_checkpoint(tmp_path, scene)
        meta_path = revision_dir / "checkpoint.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = 2
        meta["payload"].pop("spatial_cow")
        meta["payload_sha256"] = cp._payload_digest(meta["payload"])
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        fresh = load_checkpoint(revision_dir)
        assert fresh.schema_version == 2
        assert fresh.spatial_cow is None
        # The thermal payload itself is untouched: byte-equal planes.
        v3 = load_checkpoint(
            _write_checkpoint(tmp_path / "again", _scene())[0]
        )
        for name in cp.THERMAL_TENSOR_NAMES:
            assert fresh.thermal_tensors[name].tobytes() == v3.thermal_tensors[name].tobytes()
        # Whole-tile warm start policy survives: select_replay still matches.
        record, reason = cp.select_replay_checkpoint(
            tmp_path / "checkpoints", _fingerprint(scene, _met())
        )
        assert record is not None and reason is None

    def test_v2_refuses_chunk_cow_with_typed_reason(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, _ = _write_checkpoint(tmp_path, scene)
        meta_path = revision_dir / "checkpoint.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = 2
        meta["payload"].pop("spatial_cow")
        meta["payload_sha256"] = cp._payload_digest(meta["payload"])
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        loaded = load_checkpoint(revision_dir)
        plan = _plan(loaded, scene)
        assert not plan.applicable
        assert "no spatial digests" in plan.refusal_reason
        assert plan.chunks_cow_reused == 0

    def test_loadable_schemas_are_exactly_two_and_three(
        self, tmp_path: Path
    ) -> None:
        assert LOADABLE_SCHEMA_VERSIONS == (2, 3)
        assert SCHEMA_VERSION == 3
        scene = _scene()
        for bad in (1, 4, SCHEMA_VERSION + 1):
            revision_dir, _ = _write_checkpoint(tmp_path / f"r{bad}", scene)
            meta_path = revision_dir / "checkpoint.json"
            meta = json.loads(meta_path.read_text())
            meta["schema_version"] = bad
            meta_path.write_text(json.dumps(meta))
            with pytest.raises(CheckpointError, match="schema"):
                load_checkpoint(revision_dir)

    def test_tampered_spatial_block_rejected_by_payload_digest(
        self, tmp_path: Path
    ) -> None:
        scene = _scene()
        revision_dir, _ = _write_checkpoint(tmp_path, scene)
        meta_path = revision_dir / "checkpoint.json"
        meta = json.loads(meta_path.read_text())
        meta["payload"]["spatial_cow"]["scene"]["0,0"] = "f" * 64
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        with pytest.raises(CheckpointError, match="payload digest"):
            load_checkpoint(revision_dir)


class TestCPlannerFences:
    def test_identical_inputs_all_clean(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        plan = _plan(load_checkpoint(revision_dir), scene, met=met)
        assert plan.applicable
        assert plan.refusal_reason is None
        assert plan.chunks_cow_reused == plan.chunks_total == 12
        assert plan.dirty_chunks == ()
        assert plan.dirty_window is None
        assert plan.cells_replayed_fraction == 0.0

    def test_one_chunk_edit_dirties_the_d1_neighborhood(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        plan = _plan(
            load_checkpoint(revision_dir),
            _edit_chunk(scene, 1, 1),
            met=met,
            reach_pixels=REACH_D1,
        )
        assert plan.applicable
        # D = 58 // 64 + 1 = 1 around (1,1): rows 0-2, cols 0-2 clipped.
        assert set(plan.dirty_chunks) == {
            (r, c) for r in (0, 1, 2) for c in (0, 1, 2)
        }
        assert plan.dirty_window == RasterWindow(0, 192, 0, 130)
        assert plan.dirty_causes == {"scene_digest": 1}

    def test_larger_reach_widens_the_dilation_monotonically(
        self, tmp_path: Path
    ) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        plan = _plan(
            load_checkpoint(revision_dir),
            _edit_chunk(scene, 1, 1),
            met=met,
            reach_pixels=REACH_D3,  # D = 3 covers the whole 4x3 grid
        )
        assert plan.chunks_replayed == plan.chunks_total
        assert plan.cells_replayed_fraction == 1.0

    def test_landcover_edit_needs_no_dilation(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        plan = _plan(
            load_checkpoint(revision_dir),
            _edit_chunk(scene, 3, 2, landcover=True),
            met=met,
            reach_pixels=REACH_D1,
        )
        assert plan.applicable
        assert set(plan.dirty_chunks) == {(3, 2)}
        assert plan.dirty_causes == {"landcover_digest": 1}

    def test_landcover_absence_transition_is_an_edit(self, tmp_path: Path) -> None:
        scene = _scene()
        building, canopy, dem, _ = scene
        record = CheckpointRecord(
            scene_revision=1,
            coverage_manifest=(),
            thermal_tensors=_state(),
            thermal_scalars={"CI": 1.0},
            next_step=2,
            input_fingerprint=_fingerprint(scene, _met()),
            spatial_cow=capture_spatial_cow(
                building_dsm=building, canopy=canopy, dem=dem,
                resolved_landcover=None,
                march_amplitude=AMPL, amaxvalue=AMAX, scale=SCALE,
            ),
        )
        plan = _plan(record, scene)
        # Every chunk's landcover moved absent -> present: all dirty, but
        # the plan stays applicable (no fence fired — this is per-chunk).
        assert plan.applicable
        assert plan.chunks_replayed == plan.chunks_total
        assert plan.dirty_causes == {"landcover_digest": 12}

    def test_met_prefix_edit_refuses_every_chunk(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        met[0, 0] += 0.5
        plan = _plan(load_checkpoint(revision_dir), scene, met=met)
        assert not plan.applicable
        assert "met_prefix" in plan.refusal_reason
        assert "scalars" in plan.refusal_reason
        assert plan.chunks_cow_reused == 0

    def test_model_parameter_edit_refuses_every_chunk(
        self, tmp_path: Path
    ) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        plan = _plan(
            load_checkpoint(revision_dir), scene, met=met,
            params={"albedo": 0.3},
        )
        assert not plan.applicable
        assert "model_parameters" in plan.refusal_reason

    def test_shape_scale_and_amaxvalue_fences(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        building, canopy, dem, lc = scene
        shrunk = (building[:100, :].copy(), canopy[:100, :].copy(),
                  dem[:100, :].copy(), lc[:100, :].copy())
        plan_shape = _plan(record, shrunk, met=met)
        assert not plan_shape.applicable
        assert "grid shape" in plan_shape.refusal_reason

        plan_scale = _plan(record, scene, met=met, scale=0.25)
        assert not plan_scale.applicable
        assert "scale bits" in plan_scale.refusal_reason

        plan_amax = _plan(record, scene, met=met, amaxvalue=13.0)
        assert not plan_amax.applicable
        assert "amaxvalue bits" in plan_amax.refusal_reason

        plan_amp = _plan(record, scene, met=met, amplitude=11.0)
        assert not plan_amp.applicable
        assert "march amplitude bits" in plan_amp.refusal_reason

    def test_effective_amplitude_fence_is_the_t22_widening_template(
        self, tmp_path: Path
    ) -> None:
        # Raster bytes all unchanged, ONLY the reported march globals move:
        # the fence must still fire — a step-policy change flips march
        # outputs where every digest matches (T22 widening analog).
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        plan = _plan(record, scene, met=met, amplitude=AMPL * (1 + 1e-7))
        assert not plan.applicable
        assert "march amplitude bits" in plan.refusal_reason

    def test_revision_blindness_hard_contract(self, tmp_path: Path) -> None:
        # Applicability is DIGEST-ONLY: the same digests, planes and
        # fingerprint under a different scene_revision must plan
        # identically. Scene revision and SVF-store state never enter.
        scene = _scene()
        edited = _edit_chunk(scene, 1, 1)
        rev_a, met = _write_checkpoint(tmp_path, scene, revision=3)
        rev_b, _ = _write_checkpoint(tmp_path / "b", scene, revision=99)
        plan_a = _plan(load_checkpoint(rev_a), edited, met=met)
        plan_b = _plan(load_checkpoint(rev_b), edited, met=met)
        assert plan_a.applicable == plan_b.applicable
        assert plan_a.dirty_chunks == plan_b.dirty_chunks
        assert plan_a.cells_replayed_fraction == plan_b.cells_replayed_fraction
        assert plan_a.dirty_causes == plan_b.dirty_causes

    def test_reach_underestimation_marks_clean_chunks_stale(
        self, tmp_path: Path
    ) -> None:
        # The RED witness at digest level: the same edit, two reach
        # claims. The understated dilation marks chunks clean that the
        # correct one knows are within the edit's march reach.
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        edited = _edit_chunk(scene, 1, 1)
        correct = _plan(record, edited, met=met, reach_pixels=REACH_D3)
        understated = _plan(record, edited, met=met, reach_pixels=REACH_D1)
        assert set(understated.dirty_chunks) < set(correct.dirty_chunks)
        stale = set(correct.dirty_chunks) - set(understated.dirty_chunks)
        assert stale  # non-empty: these cells would be served stale bytes


class TestDComposeFences:
    def test_dirty_chunks_require_replay_tensors(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        plan = _plan(
            load_checkpoint(revision_dir), _edit_chunk(scene, 1, 1), met=met
        )
        with pytest.raises(CheckpointError, match="REPLAY"):
            compose_chunk_cow_state(load_checkpoint(revision_dir), plan)

    def test_inapplicable_plan_refuses_compose(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        met[0, 0] += 0.5
        plan = _plan(record, scene, met=met)
        with pytest.raises(CheckpointError, match="not applicable"):
            compose_chunk_cow_state(record, plan)

    def test_all_clean_compose_is_pure_checkpoint_bytes(
        self, tmp_path: Path
    ) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        plan = _plan(record, scene, met=met)
        planes, scalars = compose_chunk_cow_state(record, plan)
        assert scalars == record.thermal_scalars
        for name in cp.THERMAL_TENSOR_NAMES:
            assert planes[name].tobytes() == record.thermal_tensors[name].tobytes()

    def test_replay_shape_and_name_fences(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        plan = _plan(
            record, _edit_chunk(scene, 1, 1), met=met
        )
        window = plan.dirty_window
        good = {
            name: np.zeros(
                (window.row_stop - window.row_start,
                 window.col_stop - window.col_start),
                dtype=np.float32,
            )
            for name in cp.THERMAL_TENSOR_NAMES
        }
        missing = {k: v for k, v in good.items() if k != "TgOut1"}
        with pytest.raises(CheckpointError, match="missing the thermal plane"):
            compose_chunk_cow_state(record, plan, missing)
        wrong_shape = dict(good)
        wrong_shape["TgOut1"] = good["TgOut1"][:, :-1].copy()
        with pytest.raises(CheckpointError, match="dirty window extent"):
            compose_chunk_cow_state(record, plan, wrong_shape)

    def test_replay_scalar_divergence_is_a_soundness_bug_refused(
        self, tmp_path: Path
    ) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        plan = _plan(record, _edit_chunk(scene, 1, 1), met=met)
        window = plan.dirty_window
        replay = {
            name: np.zeros(
                (window.row_stop - window.row_start,
                 window.col_stop - window.col_start),
                dtype=np.float32,
            )
            for name in cp.THERMAL_TENSOR_NAMES
        }
        diverged = dict(record.thermal_scalars)
        diverged["CI"] = diverged["CI"] + 1e-6
        with pytest.raises(CheckpointError, match="diverged"):
            compose_chunk_cow_state(
                record, plan, replay, dirty_replay_scalars=diverged
            )

    def test_compose_splice_partitions_the_grid(self, tmp_path: Path) -> None:
        # Every cell written exactly once: clean chunks carry checkpoint
        # bytes, dirty chunks carry replay bytes — verify against an
        # explicit expected splice built chunk by chunk.
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        edited = _edit_chunk(scene, 1, 1)
        plan = _plan(record, edited, met=met)
        window = plan.dirty_window
        rng = np.random.default_rng(11)
        replay = {
            name: rng.standard_normal(
                (window.row_stop - window.row_start,
                 window.col_stop - window.col_start)
            ).astype(np.float32)
            for name in cp.THERMAL_TENSOR_NAMES
        }
        planes, _ = compose_chunk_cow_state(record, plan, replay)
        for name in cp.THERMAL_TENSOR_NAMES:
            expected = np.empty((ROWS, COLS), dtype=np.float32)
            for cr in range(4):
                for cc in range(3):
                    rs = slice(cr * 64, min((cr + 1) * 64, ROWS))
                    cs = slice(cc * 64, min((cc + 1) * 64, COLS))
                    if plan.clean[cr, cc]:
                        expected[rs, cs] = record.thermal_tensors[name][rs, cs]
                    else:
                        local_rs = slice(
                            rs.start - window.row_start, rs.stop - window.row_start
                        )
                        local_cs = slice(
                            cs.start - window.col_start, cs.stop - window.col_start
                        )
                        expected[rs, cs] = replay[name][local_rs, local_cs]
            assert planes[name].tobytes() == expected.tobytes()


# ---------------------------------------------------------------------------
# Real-physics oracle (prepared sites + solve_window)
# ---------------------------------------------------------------------------

ORACLE_ROWS = ORACLE_COLS = 192
TREE_A = TreeSpec(
    "a", 1000.0 + 10.5 * 2.0, 2000.0 - 10.5 * 2.0, 3.0, 2.0,
)
# The B-scene edit: a FLAT 11.9 m block (20x20 cells) far from tree A.
# Below the 12 m block, so adding it cannot move the site-wide march
# globals (the fixture's baseline-amaxvalue invariant). A dome-tapered
# tree cannot serve as the understated-reach witness — its flank height
# tapers to zero, so its influence never out-reaches the extra chunk ring
# ANY dilation >= 1 buys — but a flat-top block occludes the full
# 11.9 m / 0.21 m-per-px = ~56 px beyond its edge, far past the 16 px
# ring boundary at 16-cell chunks: understating the dilation then
# provably serves stale bytes.
BLOCK_ROWS = slice(96, 116)
BLOCK_COLS = slice(128, 148)
BLOCK_HEIGHT = 11.9


def _oracle_worker(tmp_path: Path, base_trees, *, extra_block: bool = False):
    from osgeo import gdal

    from solweig_gpu.incremental.solver import compose_full_scene_tensors
    from solweig_gpu.incremental.trees import TreeLayer
    from solweig_gpu.incremental.worker import ExactWorker
    from tests.test_incremental_worker import (
        DATE_STR,
        LOCAL_ALWAYS,
        TINY_EPSG,
        TINY_ORIGIN,
        _build_cache,
        _compute_baseline_svf,
        _make_prepared_site,
        _write_tif,
    )

    grid, site = _make_prepared_site(
        tmp_path / "site",
        rows=ORACLE_ROWS,
        cols=ORACLE_COLS,
        pixel=2.0,
        origin=TINY_ORIGIN,
        epsg=TINY_EPSG,
        base_trees=base_trees,
        met_hours=range(10, 13),
    )
    if extra_block:
        path = site / "Building_DSM" / "Building_DSM_0_0.tif"
        dataset = gdal.Open(str(path))
        building = dataset.ReadAsArray().astype(np.float32)
        dataset = None
        building[BLOCK_ROWS, BLOCK_COLS] = BLOCK_HEIGHT
        _write_tif(path, building, origin=TINY_ORIGIN, pixel=2.0, epsg=TINY_EPSG)
    _compute_baseline_svf(site)
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="t25",
    )
    layer = TreeLayer(cache.tree_base, grid)
    worker = ExactWorker(
        cache,
        layer,
        site_dir=site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )
    scene = compose_full_scene_tensors(cache, layer)
    return worker, grid, scene


def _oracle_solve(worker, grid, *, time_stop=None, initial_state=None,
                  write_window=None, return_final_state=False):
    from solweig_gpu.incremental.solver import solve_window

    write = write_window if write_window is not None else grid.full_window
    return solve_window(
        worker.cache,
        worker.layer,
        read_window=grid.full_window,
        write_window=write,
        forcing=worker.forcing(),
        requested_variables=("utci", "tmrt", "shadow"),
        time_start=0,
        time_stop=time_stop,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def _oracle_reach(scene, worker) -> int:
    from solweig_gpu.incremental.solver import (
        _patch_march_reach_pixels,
        _sky_patch_geometry,
    )

    amplitude = float(scene.amaxvalue.item()) - float(
        torch.min(scene.a).item()
    )
    scale = 1.0 / float(worker.cache.pixel_size_m)
    return max(
        _patch_march_reach_pixels(amplitude, scale, float(alt))
        for alt, _az, _ring in _sky_patch_geometry(2)[0]
    )


def _bytes(plane) -> bytes:
    if isinstance(plane, torch.Tensor):
        plane = plane.detach().cpu().numpy()
    return np.ascontiguousarray(plane, dtype=np.float32).tobytes()


class TestEOracleRealPhysics:
    """The theorem on the real loop: COW == fresh rebuild, bitwise."""

    @pytest.fixture()
    def ab(self, tmp_path: Path):
        worker_a, grid_a, scene_a = _oracle_worker(tmp_path / "a", (TREE_A,))
        worker_b, grid_b, scene_b = _oracle_worker(
            tmp_path / "b", (TREE_A,), extra_block=True
        )
        return worker_a, worker_b, grid_a, scene_a, scene_b

    def test_march_globals_unchanged_by_the_added_block(self, ab) -> None:
        _wa, _wb, _ga, scene_a, scene_b = ab
        # The B edit is amplitude-invariant by construction; pin it — the
        # rest of the oracle depends on the global fences NOT firing.
        assert float(scene_a.amaxvalue.item()) == float(scene_b.amaxvalue.item())
        # Canopy untouched; only the composed building surface moved.
        assert _bytes(scene_a.canopy) == _bytes(scene_b.canopy)
        assert _bytes(scene_a.a) != _bytes(scene_b.a)

    def test_theorem_edit_outside_dilation_leaves_planes_bit_identical(
        self, ab, tmp_path: Path
    ) -> None:
        worker_a, worker_b, grid, scene_a, scene_b = ab
        k = 2
        _out_a, state_a = _oracle_solve(
            worker_a, grid, time_stop=k, return_final_state=True
        )
        _out_b, state_b = _oracle_solve(
            worker_b, grid, time_stop=k, return_final_state=True
        )
        reach = _oracle_reach(scene_b, worker_b)
        # The dirty window per the planner's own math.
        record = self._checkpoint(tmp_path, worker_a, scene_a, state_a, k)
        plan = self._plan(record, worker_b, scene_b, reach)
        assert plan.applicable and plan.dirty_window is not None
        window = plan.dirty_window
        inside = np.zeros((grid.rows, grid.cols), dtype=bool)
        inside[
            window.row_start : window.row_stop,
            window.col_start : window.col_stop,
        ] = True
        # Non-vacuous: the edit changed SOME cell inside the dirty window.
        some_inside_diff = any(
            bool(
                (
                    np.frombuffer(_bytes(state_a[name]), np.float32).reshape(
                        grid.rows, grid.cols
                    )[inside]
                    != np.frombuffer(
                        _bytes(state_b[name]), np.float32
                    ).reshape(grid.rows, grid.cols)[inside]
                ).any()
            )
            for name in cp.THERMAL_TENSOR_NAMES
        )
        assert some_inside_diff, "edit had no effect inside its window?"
        # The theorem: outside the dilated window, every plane is
        # bit-identical between the two scenes' cold runs.
        for name in cp.THERMAL_TENSOR_NAMES:
            a = np.frombuffer(_bytes(state_a[name]), np.float32).reshape(
                grid.rows, grid.cols
            )
            b = np.frombuffer(_bytes(state_b[name]), np.float32).reshape(
                grid.rows, grid.cols
            )
            assert np.array_equal(a[~inside], b[~inside]), name

    def test_cow_compose_equals_fresh_rebuild_bitwise(self, ab, tmp_path: Path) -> None:
        worker_a, worker_b, grid, scene_a, scene_b = ab
        k = 2
        _out_a, state_a = _oracle_solve(
            worker_a, grid, time_stop=k, return_final_state=True
        )
        _out_b, state_b = _oracle_solve(
            worker_b, grid, time_stop=k, return_final_state=True
        )
        record = self._checkpoint(tmp_path, worker_a, scene_a, state_a, k)
        reach = _oracle_reach(scene_b, worker_b)
        plan = self._plan(record, worker_b, scene_b, reach)
        # Dirty chunks replay COLD over their bounding window.
        _replay_out, replay_state = _oracle_solve(
            worker_b,
            grid,
            time_stop=k,
            write_window=plan.dirty_window,
            return_final_state=True,
        )
        planes, scalars = compose_chunk_cow_state(
            record,
            plan,
            {
                name: replay_state[name]
                for name in cp.THERMAL_TENSOR_NAMES
            },
            dirty_replay_scalars={
                name: replay_state[name]
                for name in ("CI", "firstdaytime", "timeadd", "Twater")
                if name in replay_state
            },
        )
        for name in cp.THERMAL_TENSOR_NAMES:
            assert planes[name].tobytes() == _bytes(state_b[name]), name
        for name in ("CI", "firstdaytime", "timeadd"):
            assert scalars[name] == float(state_b[name]), name

    def test_warm_continuation_from_cow_state_equals_cold_suffix(
        self, ab, tmp_path: Path
    ) -> None:
        worker_a, worker_b, grid, scene_a, scene_b = ab
        k = 2
        _out_a, state_a = _oracle_solve(
            worker_a, grid, time_stop=k, return_final_state=True
        )
        cold_b = _oracle_solve(worker_b, grid)
        record = self._checkpoint(tmp_path, worker_a, scene_a, state_a, k)
        reach = _oracle_reach(scene_b, worker_b)
        plan = self._plan(record, worker_b, scene_b, reach)
        _replay_out, replay_state = _oracle_solve(
            worker_b,
            grid,
            time_stop=k,
            write_window=plan.dirty_window,
            return_final_state=True,
        )
        planes, scalars = compose_chunk_cow_state(
            record,
            plan,
            {name: replay_state[name] for name in cp.THERMAL_TENSOR_NAMES},
        )
        initial_state = {
            **{name: torch.from_numpy(planes[name].copy())
               for name in cp.THERMAL_TENSOR_NAMES},
            "CI": scalars.get("CI", 1.0),
            "firstdaytime": scalars.get("firstdaytime", 1.0),
            "timeadd": scalars.get("timeadd", 0.0),
            "Twater": replay_state.get("Twater"),
            "next_step": k,
        }
        from solweig_gpu.incremental.solver import solve_window

        warm = solve_window(
            worker_b.cache,
            worker_b.layer,
            read_window=grid.full_window,
            write_window=grid.full_window,
            forcing=worker_b.forcing(),
            requested_variables=("utci", "tmrt", "shadow"),
            time_start=k,
            time_stop=None,
            initial_state=initial_state,
        )
        for name in cold_b:
            assert warm[name].tobytes() == cold_b[name][k:].tobytes(), name

    def test_understated_reach_serves_stale_bytes_red_witness(
        self, ab, tmp_path: Path
    ) -> None:
        # Same pipeline with the reach deliberately understated so the
        # dilation misses part of the block's true influence: the compose
        # must now DIFFER from the fresh rebuild — the witness that the
        # dilation fence is load-bearing, not decorative. The block's
        # measured thermal influence at k=2 reaches ~14 px beyond its edge
        # (the solar-shadow footprint); at 8-cell chunks the D=1 ring an
        # understated reach of 7 buys covers only 8 px, so clean chunks
        # receive bytes the edit already falsified. D_correct = 59//8+1 = 8.
        worker_a, worker_b, grid, scene_a, scene_b = ab
        k = 2
        _out_a, state_a = _oracle_solve(
            worker_a, grid, time_stop=k, return_final_state=True
        )
        _out_b, state_b = _oracle_solve(
            worker_b, grid, time_stop=k, return_final_state=True
        )
        cells = 8
        record = self._checkpoint(
            tmp_path, worker_a, scene_a, state_a, k, chunk_cells=cells
        )
        reach = _oracle_reach(scene_b, worker_b)
        correct = self._plan(record, worker_b, scene_b, reach, chunk_cells=cells)
        understated = self._plan(
            record, worker_b, scene_b, 7, chunk_cells=cells
        )
        assert understated.dirty_chunks and correct.dirty_window is not None
        assert set(understated.dirty_chunks) < set(correct.dirty_chunks)
        window = understated.dirty_window
        _ro, replay_state = _oracle_solve(
            worker_b,
            grid,
            time_stop=k,
            write_window=correct.dirty_window,
            return_final_state=True,
        )
        planes, _scalars = compose_chunk_cow_state(
            record,
            understated,
            {
                name: replay_state[name][
                    window.row_start - correct.dirty_window.row_start : window.row_stop - correct.dirty_window.row_start,
                    window.col_start - correct.dirty_window.col_start : window.col_stop - correct.dirty_window.col_start,
                ]
                for name in cp.THERMAL_TENSOR_NAMES
            },
        )
        stale = 0
        for name in cp.THERMAL_TENSOR_NAMES:
            fresh = np.frombuffer(_bytes(state_b[name]), np.float32)
            stale += int(np.count_nonzero(fresh != planes[name].reshape(-1)))
        assert stale > 0, (
            "understated dilation served identical bytes — witness vacuous"
        )

    # -- helpers ------------------------------------------------------------

    def _checkpoint(self, tmp_path, worker, scene, state, k,
                    chunk_cells: int = CHUNK_CELLS):
        met = worker.forcing().met_table[:k]
        fingerprint = thermal_fingerprint(
            building_dsm=scene.a.numpy(),
            canopy=scene.canopy.numpy(),
            dem=scene.dem.numpy(),
            resolved_landcover=(
                np.asarray(worker.cache.landcover)
                if "landcover" in worker.cache
                else None
            ),
            model_parameters=None,
            met_prefix=met,
        )
        record = thermal_checkpoint(
            scene_revision=1,
            store=cp.TemporalResultStore(),
            final_state=state,
            input_fingerprint=fingerprint,
        )
        record.spatial_cow = capture_spatial_cow(
            building_dsm=scene.a.numpy(),
            canopy=scene.canopy.numpy(),
            dem=scene.dem.numpy(),
            resolved_landcover=(
                np.asarray(worker.cache.landcover)
                if "landcover" in worker.cache
                else None
            ),
            march_amplitude=float(scene.amaxvalue.item())
            - float(torch.min(scene.a).item()),
            amaxvalue=float(scene.amaxvalue.item()),
            scale=1.0 / float(worker.cache.pixel_size_m),
            chunk_cells=chunk_cells,
        )
        revision_dir = write_checkpoint(
            record, tmp_path / f"oracle_cp_{chunk_cells}"
        )
        return load_checkpoint(revision_dir)

    def _plan(self, record, worker, scene, reach_pixels,
              chunk_cells: int = CHUNK_CELLS):
        return plan_chunk_cow_warm_start(
            record,
            building_dsm=scene.a.numpy(),
            canopy=scene.canopy.numpy(),
            dem=scene.dem.numpy(),
            resolved_landcover=(
                np.asarray(worker.cache.landcover)
                if "landcover" in worker.cache
                else None
            ),
            model_parameters=None,
            met_prefix=worker.forcing().met_table[: int(record.next_step)],
            march_amplitude=float(scene.amaxvalue.item())
            - float(torch.min(scene.a).item()),
            amaxvalue=float(scene.amaxvalue.item()),
            scale=1.0 / float(worker.cache.pixel_size_m),
            reach_pixels=reach_pixels,
            chunk_cells=chunk_cells,
        )


class TestFAblationTelemetry:
    """§579 ledger: access width + page-table cost, arithmetic gates."""

    def test_counter_identities_on_synthetic_site(self, tmp_path: Path) -> None:
        scene = _scene()
        revision_dir, met = _write_checkpoint(tmp_path, scene)
        record = load_checkpoint(revision_dir)
        plan = _plan(record, _edit_chunk(scene, 1, 1), met=met)
        assert plan.chunks_total == 12
        assert plan.chunks_cow_reused + plan.chunks_replayed == plan.chunks_total
        assert plan.cells_total == ROWS * COLS
        reused_cells = plan.cells_total - plan.cells_replayed
        assert plan.cells_replayed_fraction == pytest.approx(
            plan.cells_replayed / plan.cells_total
        )
        assert plan.plane_bytes_cow_reused == reused_cells * 4 * len(
            record.thermal_tensors
        )

    def test_500_site_ledger_printed(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(5)
        rows = cols = 500
        dem = np.zeros((rows, cols), np.float32)
        building = dem.copy()
        building[40:100, 20:65] = 12.0
        canopy = (rng.random((rows, cols)) > 0.99).astype(np.float32) * 3.0
        landcover = np.ones((rows, cols), np.uint8)
        scene = (building, canopy, dem, landcover)
        edited = (
            building,
            canopy.copy(),
            dem,
            landcover,
        )
        edited[1][256:320, 256:320] += 1.0  # the center chunk (4, 4)
        met = _met()
        record = CheckpointRecord(
            scene_revision=1,
            coverage_manifest=(),
            thermal_tensors={
                name: np.zeros((rows, cols), np.float32)
                for name in cp.THERMAL_TENSOR_NAMES
            },
            thermal_scalars={"CI": 1.0},
            next_step=2,
            input_fingerprint=_fingerprint(scene, met),
            spatial_cow=_cow(scene),
        )
        plan = plan_chunk_cow_warm_start(
            record,
            building_dsm=edited[0],
            canopy=edited[1],
            dem=edited[2],
            resolved_landcover=edited[3],
            model_parameters=None,
            met_prefix=met,
            march_amplitude=AMPL,
            amaxvalue=AMAX,
            scale=SCALE,
            reach_pixels=58,  # D=1 at 64-cell chunks
        )
        assert plan.applicable
        print(
            f"\nT25 ledger @500x500, one 64x64 chunk edited, D=1: "
            f"chunks {plan.chunks_replayed}/{plan.chunks_total} replayed, "
            f"cells_replayed_fraction={plan.cells_replayed_fraction:.4f}, "
            f"plane_bytes_cow_reused={plan.plane_bytes_cow_reused}, "
            f"page_table_bytes={plan.page_table_bytes}"
        )
        # 9 dirty chunks of 64 at the center: 9*64*64 / 250000.
        assert plan.cells_replayed == 9 * 64 * 64
        assert plan.page_table_bytes == len(
            json.dumps(dict(record.spatial_cow), sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
        )
