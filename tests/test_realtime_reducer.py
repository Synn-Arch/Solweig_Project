# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic epoch reducer tests (R1 reducer wave).

Every test drives ``DeterministicEpochReducer`` as a pure function:
same inputs -> equal ``EpochReduction`` values (dataclass equality),
conflicts recorded as data, every accepted operation audited even when
it cancels out (collaborative_state.md, "Epoch reduction").
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import time

import pytest

from solweig_gpu.server.realtime.reducer import CHUNK_SIZE_PX, DeterministicEpochReducer
from solweig_gpu.server.realtime.types import (
    CanonicalState,
    ConflictKind,
    EpochReducer,
    Operation,
)

WS = "ws-reducer-test"


def make_op(
    seq: int,
    family: str,
    verb: str,
    entity: str | None = None,
    actor: str = "alice",
    payload: dict | None = None,
    epoch: int = 1,
    ws: str = WS,
    op_id: str | None = None,
) -> Operation:
    return Operation(
        workspace_id=ws,
        operation_id=op_id or f"op-{seq:04d}",
        actor_id=actor,
        client_sequence=seq,
        base_revision=0,
        source_family=family,
        entity_id=entity,
        verb=verb,
        payload=payload if payload is not None else {},
        received_at="2026-09-04T00:00:00Z",
        accepted_at="2026-09-04T00:00:00Z",
        server_sequence=seq,
        epoch_id=epoch,
    )


def kinds(records) -> list[str]:
    return [record.kind for record in records]


@pytest.fixture()
def reducer() -> EpochReducer:
    return DeterministicEpochReducer()


@pytest.fixture()
def base(reducer: EpochReducer) -> CanonicalState:
    return reducer.initial_state(WS)


def conflicts_of(reduced, kind: str, family: str | None = None):
    return [
        record
        for record in reduced.conflicts
        if record.kind == kind and (family is None or record.source_family == family)
    ]


# ---------------------------------------------------------------------------
# initial_state family shapes
# ---------------------------------------------------------------------------


def test_initial_state_has_all_family_shapes(reducer: EpochReducer) -> None:
    state = reducer.initial_state(WS)
    assert state.workspace_id == WS
    assert state.workspace_revision == 0
    families = state.families
    for family in ("vegetation_geometry", "building_geometry"):
        assert families[family] == {"objects": {}, "tombstones": {}}
    assert families["landcover_surface"] == {"chunks": {}}
    for family in ("meteorological_forcing", "model_receptor_parameters"):
        assert families[family] == {"segments": {}}
    assert families["output_view"] == {"view": {}, "per_user": {}}


def test_reduce_empty_operations_is_identity(reducer: EpochReducer, base) -> None:
    result = reducer.reduce_epoch(base, [])
    assert result.reduced.workspace_revision == 0
    assert result.reduced.epoch_id == 0
    assert result.reduced.family_deltas == {}
    assert result.reduced.audit_operations == ()
    assert result.reduced.conflicts == ()
    assert result.reduced.is_noop is True
    assert result.state.families == base.families


# ---------------------------------------------------------------------------
# Object geometry: generations, tombstones, LWW
# ---------------------------------------------------------------------------


def test_add_creates_generation_zero_and_add_command(reducer, base) -> None:
    ops = [
        make_op(1, "vegetation_geometry", "add", entity="tree-1",
                payload={"values": {"x_m": 12.0, "y_m": 8.0, "height_m": 9.0}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    tree = result.state.families["vegetation_geometry"]["objects"]["tree-1"]
    assert tree["generation"] == 0
    assert tree["x_m"] == 12.0 and tree["height_m"] == 9.0
    delta = result.reduced.family_deltas["vegetation_geometry"]
    assert delta.is_noop is False
    assert len(delta.commands) == 1
    command = delta.commands[0]
    assert command["operation"] == "add"
    assert command["target"] == "tree-1"
    assert command["values"]["height_m"] == 9.0
    assert delta.summary == {
        "objects_added": 1, "objects_moved": 0,
        "objects_replaced": 0, "objects_deleted": 0,
    }


def test_add_over_existing_bumps_generation_records_recreation(reducer, base) -> None:
    ops = [
        make_op(1, "building_geometry", "add", entity="b-1",
                payload={"values": {"height_m": 20.0, "footprint_m": [[0, 0], [4, 0], [4, 4]]}}),
        make_op(2, "building_geometry", "add", entity="b-1", actor="bob",
                payload={"values": {"building_id": "b-1", "height_m": 30.0}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    building = result.state.families["building_geometry"]["objects"]["b-1"]
    assert building["generation"] == 1
    assert building["height_m"] == 30.0
    assert "building_id" not in building  # identity key never stored as a field
    records = conflicts_of(result.reduced, ConflictKind.GENERATION_RECREATED)
    assert len(records) == 1
    assert records[0].winner_operation_id == "op-0002"
    assert records[0].loser_operation_ids == ("op-0001",)
    assert records[0].detail["from_generation"] == 0
    assert records[0].detail["to_generation"] == 1


def test_add_then_delete_new_entity_is_noop_with_full_audit(reducer, base) -> None:
    ops = [
        make_op(1, "vegetation_geometry", "add", entity="tree-9",
                payload={"values": {"x_m": 1.0, "y_m": 1.0, "height_m": 5.0}}),
        make_op(2, "vegetation_geometry", "delete", entity="tree-9"),
    ]
    result = reducer.reduce_epoch(base, ops)
    delta = result.reduced.family_deltas["vegetation_geometry"]
    assert delta.is_noop is True
    assert delta.commands == ()
    assert delta.summary["objects_added"] == 0
    assert len(result.reduced.audit_operations) == 2  # never dropped
    assert [op.operation_id for op in result.reduced.audit_operations] == [
        "op-0001", "op-0002",
    ]
    state = result.state.families["vegetation_geometry"]
    assert state["objects"] == {}
    assert state["tombstones"] == {"tree-9": 0}
    assert result.reduced.is_noop is True
    # the epoch still records a canonical revision
    assert result.reduced.workspace_revision == 1


def test_delete_unknown_entity_recorded_without_state_change(reducer, base) -> None:
    ops = [
        make_op(1, "vegetation_geometry", "delete", entity="ghost"),
        make_op(2, "vegetation_geometry", "add", entity="tree-1",
                payload={"values": {"height_m": 5.0}}),
        make_op(3, "vegetation_geometry", "delete", entity="tree-1"),
        make_op(4, "vegetation_geometry", "delete", entity="tree-1"),  # stale
    ]
    result = reducer.reduce_epoch(base, ops)
    state = result.state.families["vegetation_geometry"]
    assert state["objects"] == {}
    assert state["tombstones"] == {"tree-1": 0}
    assert result.reduced.conflicts == ()  # recorded in audit, not conflicts
    assert len(result.reduced.audit_operations) == 4
    delta = result.reduced.family_deltas["vegetation_geometry"]
    assert delta.is_noop is True  # nothing survived versus the empty baseline


def test_update_on_tombstoned_entity_rejected(reducer, base) -> None:
    ops = [
        make_op(1, "building_geometry", "add", entity="b-1",
                payload={"values": {"height_m": 20.0}}),
        make_op(2, "building_geometry", "delete", entity="b-1"),
        make_op(3, "building_geometry", "replace", entity="b-1", actor="bob",
                payload={"values": {"height_m": 99.0}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    state = result.state.families["building_geometry"]
    assert "b-1" not in state["objects"]  # no accidental resurrection
    records = conflicts_of(result.reduced, ConflictKind.TOMBSTONE_REJECTED)
    assert len(records) == 1
    assert records[0].winner_operation_id == "op-0002"  # the delete wins
    assert records[0].loser_operation_ids == ("op-0003",)
    assert records[0].detail["targeting_generation"] == 0
    # delete-then-add (a NEW generation) is the legitimate resurrect path
    ops2 = ops + [
        make_op(4, "building_geometry", "add", entity="b-1", actor="bob",
                payload={"values": {"height_m": 21.0}}),
    ]
    result2 = reducer.reduce_epoch(base, ops2)
    resurrected = result2.state.families["building_geometry"]["objects"]["b-1"]
    assert resurrected["generation"] == 1
    assert resurrected["height_m"] == 21.0
    assert kinds(result2.reduced.conflicts).count(ConflictKind.GENERATION_RECREATED) == 1


def test_tombstone_rejection_across_epoch_boundary(reducer, base) -> None:
    epoch1 = reducer.reduce_epoch(base, [
        make_op(1, "vegetation_geometry", "add", entity="tree-2", epoch=1,
                payload={"values": {"height_m": 6.0}}),
        make_op(2, "vegetation_geometry", "delete", entity="tree-2", epoch=1),
    ])
    # epoch 2: an old client updates the entity tombstoned in epoch 1
    epoch2 = reducer.reduce_epoch(epoch1.state, [
        make_op(3, "vegetation_geometry", "replace", entity="tree-2", actor="bob",
                epoch=2, payload={"values": {"height_m": 7.0}}),
    ])
    records = conflicts_of(epoch2.reduced, ConflictKind.TOMBSTONE_REJECTED)
    assert len(records) == 1
    # the delete lives in a previous epoch; the record documents the stand-off
    assert records[0].winner_operation_id == "op-0003"
    assert records[0].detail["winner_is_epoch_delete"] is False
    assert "tree-2" not in epoch2.state.families["vegetation_geometry"]["objects"]
    assert epoch2.reduced.family_deltas["vegetation_geometry"].is_noop is True


def test_multi_actor_same_entity_field_lww(reducer, base) -> None:
    ops = [
        make_op(1, "building_geometry", "add", entity="b-1",
                payload={"values": {"height_m": 10.0, "footprint_m": [[0, 0], [2, 0], [2, 2]]}}),
        make_op(2, "building_geometry", "replace", entity="b-1", actor="bob",
                payload={"values": {"height_m": 20.0}}),
        make_op(3, "building_geometry", "replace", entity="b-1", actor="alice",
                payload={"values": {"height_m": 15.0}}),
        # different field, different actor: merges in, never a field conflict
        make_op(4, "building_geometry", "replace", entity="b-1", actor="carol",
                payload={"values": {"footprint_m": [[1, 1], [3, 1], [3, 3]]}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    building = result.state.families["building_geometry"]["objects"]["b-1"]
    assert building["height_m"] == 15.0  # last write by server_sequence
    assert building["footprint_m"] == [[1, 1], [3, 1], [3, 3]]
    records = conflicts_of(result.reduced, ConflictKind.LAST_WRITE_WINS)
    # the add's own writes count: height_m raced by all three actors,
    # footprint_m raced by the adder and carol
    by_field = {record.detail["field"]: record for record in records}
    assert set(by_field) == {"height_m", "footprint_m"}
    assert by_field["height_m"].winner_operation_id == "op-0003"
    assert by_field["height_m"].loser_operation_ids == ("op-0001", "op-0002")
    assert by_field["footprint_m"].winner_operation_id == "op-0004"
    assert by_field["footprint_m"].loser_operation_ids == ("op-0001",)
    delta = result.reduced.family_deltas["building_geometry"]
    # add + replaces coalesce into a single add command for the new entity
    assert [c["operation"] for c in delta.commands] == ["add"]


def test_move_replaces_position_fields_only(reducer, base) -> None:
    ops = [
        make_op(1, "vegetation_geometry", "add", entity="tree-1",
                payload={"values": {"x_m": 0.0, "y_m": 0.0, "height_m": 8.0}}),
        make_op(2, "vegetation_geometry", "move", entity="tree-1", actor="bob",
                payload={"values": {"x_m": 3.0, "y_m": 4.0, "height_m": 99.0}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    tree = result.state.families["vegetation_geometry"]["objects"]["tree-1"]
    assert (tree["x_m"], tree["y_m"]) == (3.0, 4.0)
    assert tree["height_m"] == 8.0  # move never touches non-position fields
    delta = result.reduced.family_deltas["vegetation_geometry"]
    assert delta.summary["objects_moved"] == 0  # baseline lacks the entity: net add
    assert [c["operation"] for c in delta.commands] == ["add"]

    # versus a baseline that already carries the entity: net move
    epoch2 = reducer.reduce_epoch(result.state, [
        make_op(3, "vegetation_geometry", "move", entity="tree-1", epoch=2,
                payload={"values": {"x_m": 9.0, "y_m": 9.0}}),
    ])
    delta2 = epoch2.reduced.family_deltas["vegetation_geometry"]
    assert delta2.summary["objects_moved"] == 1
    assert delta2.commands[0]["operation"] == "move"
    assert delta2.commands[0]["old_values"]["x_m"] == 3.0


# ---------------------------------------------------------------------------
# Raster paint: chunked LWW
# ---------------------------------------------------------------------------


def test_landcover_chunk_lww_and_overwrite_conflict(reducer, base) -> None:
    ops = [
        make_op(1, "landcover_surface", "paint",
                payload={"window": {"row_start": 0, "row_stop": 3,
                                    "col_start": 0, "col_stop": 3}, "class": 2}),
        make_op(2, "landcover_surface", "paint", actor="bob",
                payload={"window": {"row_start": 2, "row_stop": 5,
                                    "col_start": 2, "col_stop": 5}, "class": 3}),
    ]
    result = reducer.reduce_epoch(base, ops)
    chunk = result.state.families["landcover_surface"]["chunks"]["0:0"]
    assert chunk["2:2"] == 3  # later op's cells win in the overlap
    assert chunk["0:0"] == 2
    assert chunk["4:4"] == 3
    records = conflicts_of(result.reduced, ConflictKind.RASTER_CHUNK_OVERWRITTEN)
    assert len(records) == 1
    assert records[0].entity_id == "0:0"
    assert records[0].winner_operation_id == "op-0002"
    assert records[0].loser_operation_ids == ("op-0001",)
    assert records[0].detail["cells_conflicted"] == 1
    delta = result.reduced.family_deltas["landcover_surface"]
    # 9 + 9 painted cells minus 1 overlap = 17 net changed cells
    assert delta.summary["cells_changed"] == 17
    assert delta.summary["chunks_changed"] == 1
    painted = {
        (cell[0], cell[1])
        for command in delta.commands
        for cell in command["target"]["cells"]
    }
    assert len(painted) == 17
    assert delta.is_noop is False


def test_landcover_cell_list_paint_and_disjoint_chunks(reducer, base) -> None:
    ops = [
        make_op(1, "landcover_surface", "paint",
                payload={"cells": [[0, 0], [1, 1]], "class": 5}),
        make_op(2, "landcover_surface", "paint", actor="bob",
                payload={"cells": [[CHUNK_SIZE_PX, CHUNK_SIZE_PX]], "class": 6}),
    ]
    result = reducer.reduce_epoch(base, ops)
    assert result.reduced.conflicts == ()  # different chunks: no conflict
    chunks = result.state.families["landcover_surface"]["chunks"]
    assert chunks["0:0"] == {"0:0": 5, "1:1": 5}
    assert chunks["1:1"] == {f"{CHUNK_SIZE_PX}:{CHUNK_SIZE_PX}": 6}
    delta = result.reduced.family_deltas["landcover_surface"]
    assert delta.summary == {"cells_changed": 3, "chunks_changed": 2}


def test_landcover_repaint_of_baseline_value_is_noop(reducer, base) -> None:
    epoch1 = reducer.reduce_epoch(base, [
        make_op(1, "landcover_surface", "paint",
                payload={"window": {"row_start": 0, "row_stop": 2,
                                    "col_start": 0, "col_stop": 2}, "class": 4}),
    ])
    epoch2 = reducer.reduce_epoch(epoch1.state, [
        make_op(2, "landcover_surface", "paint", actor="bob", epoch=2,
                payload={"window": {"row_start": 0, "row_stop": 2,
                                    "col_start": 0, "col_stop": 2}, "class": 4}),
    ])
    delta = epoch2.reduced.family_deltas["landcover_surface"]
    assert delta.is_noop is True
    assert delta.commands == ()


# ---------------------------------------------------------------------------
# Scalar/range: segment normalization
# ---------------------------------------------------------------------------


def test_scalar_lww_per_timestep(reducer, base) -> None:
    ops = [
        make_op(1, "meteorological_forcing", "set",
                payload={"time_index": 5, "values": {"air_temperature": 22.0}}),
        make_op(2, "meteorological_forcing", "set", actor="bob",
                payload={"time_index": 5, "values": {"air_temperature": 25.0}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    segments = result.state.families["meteorological_forcing"]["segments"]
    assert segments["air_temperature"] == [(5, 5, 25.0)]
    records = conflicts_of(result.reduced, ConflictKind.LAST_WRITE_WINS,
                           "meteorological_forcing")
    assert len(records) == 1
    assert records[0].winner_operation_id == "op-0002"
    assert records[0].loser_operation_ids == ("op-0001",)
    delta = result.reduced.family_deltas["meteorological_forcing"]
    assert delta.commands[0]["values"]["segments"] == [[5, 5, 25.0]]


def test_overlapping_ranges_from_two_actors_segment(reducer, base) -> None:
    ops = [
        make_op(1, "meteorological_forcing", "set",
                payload={"time_start": 0, "time_stop": 10,
                         "values": {"air_temperature": 22.0}}),
        make_op(2, "meteorological_forcing", "set", actor="bob",
                payload={"time_start": 4, "time_stop": 6,
                         "values": {"air_temperature": 30.0}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    segments = result.state.families["meteorological_forcing"]["segments"]
    assert segments["air_temperature"] == [
        (0, 3, 22.0), (4, 6, 30.0), (7, 10, 22.0),
    ]
    records = conflicts_of(result.reduced, ConflictKind.RANGE_SEGMENTED)
    assert len(records) == 1
    assert records[0].winner_operation_id == "op-0002"
    assert records[0].loser_operation_ids == ("op-0001",)
    assert records[0].detail["field"] == "air_temperature"
    delta = result.reduced.family_deltas["meteorological_forcing"]
    assert delta.summary == {"fields_changed": 1, "segments_changed": 3}
    assert delta.commands[0]["target"] == "air_temperature"
    assert delta.commands[0]["values"]["segments"] == [
        [0, 3, 22.0], [4, 6, 30.0], [7, 10, 22.0],
    ]


def test_adjacent_equal_value_segments_merge_canonically(reducer, base) -> None:
    ops = [
        make_op(1, "meteorological_forcing", "set",
                payload={"time_start": 0, "time_stop": 5,
                         "values": {"air_temperature": 22.0}}),
        make_op(2, "meteorological_forcing", "set", actor="bob",
                payload={"time_start": 6, "time_stop": 10,
                         "values": {"air_temperature": 22.0}}),
        # same actor re-painting a covered subrange with the same value:
        # overlap normalizes away, state stays canonical, no conflict
        make_op(3, "meteorological_forcing", "set",
                payload={"time_start": 2, "time_stop": 3,
                         "values": {"air_temperature": 22.0}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    segments = result.state.families["meteorological_forcing"]["segments"]
    assert segments["air_temperature"] == [(0, 10, 22.0)]
    assert result.reduced.conflicts == ()  # no overlap ever happened


def test_unbounded_whole_series_write_wins_to_end(reducer, base) -> None:
    ops = [
        make_op(1, "meteorological_forcing", "set",
                payload={"time_start": 0, "time_stop": 10,
                         "values": {"relative_humidity": 60}}),
        make_op(2, "meteorological_forcing", "set", actor="bob",
                payload={"values": {"relative_humidity": 55}}),  # whole series
    ]
    result = reducer.reduce_epoch(base, ops)
    segments = result.state.families["meteorological_forcing"]["segments"]
    assert segments["relative_humidity"] == [(0, None, 55)]


def test_model_parameters_and_selected_date_time_use_segments(reducer, base) -> None:
    ops = [
        make_op(1, "model_receptor_parameters", "set",
                payload={"values": {"albedo_asphalt": 0.12}}),
        make_op(2, "selected_date_time", "select",
                payload={"values": {"time_index": 14}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    params = result.state.families["model_receptor_parameters"]["segments"]
    assert params["albedo_asphalt"] == [(0, None, 0.12)]
    when = result.state.families["selected_date_time"]["segments"]
    assert when["time_index"] == [(0, None, 14)]
    assert set(result.reduced.family_deltas) == {
        "model_receptor_parameters", "selected_date_time",
    }


# ---------------------------------------------------------------------------
# View state
# ---------------------------------------------------------------------------


def test_view_select_lww_and_per_user_exclusion(reducer, base) -> None:
    ops = [
        make_op(1, "output_view", "select",
                payload={"values": {"layer": "utci", "time_index": 3}}),
        make_op(2, "output_view", "select", actor="bob",  # per-user only
                payload={"scope": "user", "values": {"layer": "tmrt"}}),
        make_op(3, "output_view", "select", actor="carol",
                payload={"values": {"layer": "tmrt", "time_index": 7}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    state = result.state.families["output_view"]
    assert state["view"] == {"layer": "tmrt", "time_index": 7}
    assert state["per_user"] == {"bob": {"layer": "tmrt"}}
    records = conflicts_of(result.reduced, ConflictKind.LAST_WRITE_WINS,
                           "output_view")
    assert len(records) == 1
    assert records[0].winner_operation_id == "op-0003"
    assert records[0].loser_operation_ids == ("op-0001",)
    delta = result.reduced.family_deltas["output_view"]
    assert delta.commands[0]["values"] == {"layer": "tmrt", "time_index": 7}
    assert "per_user" not in str(delta.commands)  # never in the delta
    assert delta.summary == {"workspace_view_changed": 1, "per_user_updates": 1}


def test_per_user_only_epoch_is_view_noop(reducer, base) -> None:
    ops = [
        make_op(1, "output_view", "select",
                payload={"scope": "user", "values": {"layer": "utci"}}),
        make_op(2, "output_view", "select", actor="bob",
                payload={"scope": "user", "values": {"layer": "tmrt"}}),
    ]
    result = reducer.reduce_epoch(base, ops)
    delta = result.reduced.family_deltas["output_view"]
    assert delta.is_noop is True
    assert delta.commands == ()
    assert delta.summary["per_user_updates"] == 2
    assert result.state.families["output_view"]["per_user"] == {
        "alice": {"layer": "utci"}, "bob": {"layer": "tmrt"},
    }
    assert result.reduced.conflicts == ()  # per-user races are not conflicts


# ---------------------------------------------------------------------------
# Determinism discipline
# ---------------------------------------------------------------------------


def test_duplicate_server_sequence_raises_value_error(reducer, base) -> None:
    ops = [
        make_op(1, "vegetation_geometry", "add", entity="tree-1",
                payload={"values": {"height_m": 5.0}}),
        make_op(1, "vegetation_geometry", "add", entity="tree-2",
                payload={"values": {"height_m": 6.0}}),
    ]
    with pytest.raises(ValueError, match="server_sequence"):
        reducer.reduce_epoch(base, ops)


def test_reducer_never_mutates_baseline(reducer, base) -> None:
    ops = [
        make_op(1, "vegetation_geometry", "add", entity="tree-1",
                payload={"values": {"height_m": 5.0}}),
        make_op(2, "landcover_surface", "paint",
                payload={"cells": [[7, 8]], "class": 3}),
        make_op(3, "output_view", "select", payload={"values": {"layer": "utci"}}),
    ]
    snapshot = copy.deepcopy(base)
    reducer.reduce_epoch(base, ops)
    assert base == snapshot


def test_replay_determinism_reduce_twice_equal(reducer, base) -> None:
    ops = [
        make_op(1, "vegetation_geometry", "add", entity="tree-1",
                payload={"values": {"x_m": 1.0, "y_m": 2.0}}),
        make_op(2, "vegetation_geometry", "replace", entity="tree-1", actor="bob",
                payload={"values": {"height_m": 7.0}}),
        make_op(3, "landcover_surface", "paint", actor="bob",
                payload={"window": {"row_start": 0, "row_stop": 70,
                                    "col_start": 0, "col_stop": 70}, "class": 1}),
        make_op(4, "meteorological_forcing", "set", actor="carol",
                payload={"time_start": 0, "time_stop": 23,
                         "values": {"air_temperature": 21.5}}),
        make_op(5, "output_view", "select", actor="bob",
                payload={"values": {"layer": "tmrt"}}),
    ]
    first = reducer.reduce_epoch(base, ops)
    second = reducer.reduce_epoch(base, list(reversed(ops)))  # input order is irrelevant
    assert first.reduced == second.reduced
    assert first.state == second.state


def test_permutation_invariance_within_commutative_subsets(reducer, base) -> None:
    """Permuted orders of commutative ops (server_sequence reassigned to the
    permutation) must yield identical family deltas and final state."""
    templates = [
        ("vegetation_geometry", "add", "tree-1", "alice", {"values": {"height_m": 5.0}}),
        ("vegetation_geometry", "add", "tree-2", "bob", {"values": {"height_m": 6.0}}),
        ("landcover_surface", "paint", None, "alice",
         {"cells": [[0, 0], [0, 1]], "class": 2}),
        ("landcover_surface", "paint", None, "bob",
         {"cells": [[CHUNK_SIZE_PX + 5, 3]], "class": 4}),
    ]

    def build(order: tuple[int, ...]) -> list[Operation]:
        ops: list[Operation] = []
        for seq, idx in enumerate(order, start=1):
            family, verb, entity, actor, payload = templates[idx]
            op = make_op(seq, family, verb, entity=entity, actor=actor,
                         payload=payload,
                         op_id=f"{family}-{entity or idx}")
            # client_sequence is the actor's own stable order; only the
            # server total order follows the permutation
            ops.append(dataclasses.replace(op, client_sequence=idx + 1))
        return ops

    def audit_ids(result) -> list[str]:
        return sorted(op.operation_id for op in result.reduced.audit_operations)

    reference = reducer.reduce_epoch(base, build((0, 1, 2, 3)))
    for order in itertools.permutations(range(4)):
        if order == (0, 1, 2, 3):
            continue
        candidate = reducer.reduce_epoch(base, build(order))
        assert candidate.reduced.family_deltas == reference.reduced.family_deltas
        assert candidate.state.families == reference.state.families
        # same accepted operation set; sequence fields follow the permutation
        assert audit_ids(candidate) == audit_ids(reference)
    assert reference.reduced.conflicts == ()


def test_chained_epochs_match_single_shot_fold(reducer, base) -> None:
    ops1 = [
        make_op(1, "vegetation_geometry", "add", entity="tree-1", epoch=1,
                payload={"values": {"height_m": 5.0}}),
        make_op(2, "vegetation_geometry", "delete", entity="tree-1", epoch=1),
        make_op(3, "building_geometry", "add", entity="b-1", epoch=1,
                payload={"values": {"height_m": 20.0}}),
    ]
    ops2 = [
        make_op(4, "building_geometry", "replace", entity="b-1", actor="bob", epoch=2,
                payload={"values": {"height_m": 25.0}}),
        make_op(5, "meteorological_forcing", "set", actor="carol", epoch=2,
                payload={"time_index": 9, "values": {"air_temperature": 19.0}}),
    ]
    epoch1 = reducer.reduce_epoch(base, ops1)
    epoch2 = reducer.reduce_epoch(epoch1.state, ops2)
    single = reducer.reduce_epoch(base, ops1 + ops2)

    # identical final state regardless of where the epoch boundary falls
    assert epoch2.state.families == single.state.families
    # the hidden intermediate revision is the only difference
    assert epoch2.state.workspace_revision == single.state.workspace_revision + 1

    # epoch 2's delta is measured against the chained baseline (b-1 exists
    # there with height 20), not against the empty pre-epoch-1 baseline
    delta2 = epoch2.reduced.family_deltas["building_geometry"]
    assert delta2.commands[0]["operation"] == "replace"
    assert delta2.commands[0]["old_values"]["height_m"] == 20.0
    assert delta2.commands[0]["values"]["height_m"] == 25.0
    # single-shot sees one coalesced add at the final height
    single_delta = single.reduced.family_deltas["building_geometry"]
    assert single_delta.commands[0]["operation"] == "add"
    assert single_delta.commands[0]["values"]["height_m"] == 25.0

    # with a purely no-op first epoch, epoch 2's deltas equal the deltas of
    # one reduce over the concatenated op stream (the noop epoch contributed
    # only its audited vegetation entry, which epoch 2's families exclude)
    noop_epoch = reducer.reduce_epoch(base, ops1[:2])  # add then delete only
    tail = reducer.reduce_epoch(noop_epoch.state, ops2)
    straight = reducer.reduce_epoch(base, ops1[:2] + ops2)
    assert set(tail.reduced.family_deltas) <= set(straight.reduced.family_deltas)
    assert tail.reduced.family_deltas == {
        family: straight.reduced.family_deltas[family]
        for family in tail.reduced.family_deltas
    }
    assert straight.reduced.family_deltas["vegetation_geometry"].is_noop is True


# ---------------------------------------------------------------------------
# Performance guard
# ---------------------------------------------------------------------------


def test_reduce_ten_thousand_operations_under_two_seconds(reducer, base) -> None:
    ops: list[Operation] = []
    seq = 0
    for i in range(4000):  # object traffic across 400 entities, 4 actors
        entity = f"tree-{i % 400}"
        actor = ("alice", "bob", "carol", "dave")[i % 4]
        verb = ("add", "replace", "move", "delete")[i % 4]
        seq += 1
        ops.append(make_op(seq, "vegetation_geometry", verb, entity=entity,
                           actor=actor, epoch=1,
                           payload={"values": {"x_m": float(i % 90),
                                               "y_m": float(i % 90),
                                               "height_m": 3.0 + (i % 20)}}))
    for i in range(3000):  # paint traffic spread over many chunks
        seq += 1
        ops.append(make_op(seq, "landcover_surface", "paint",
                           actor=("alice", "bob")[i % 2], epoch=1,
                           payload={"cells": [[i % 480, (i * 7) % 480],
                                              [i % 480, (i * 7 + 1) % 480]],
                                    "class": i % 6}))
    for i in range(2000):  # scalar/range traffic over 24 fields
        seq += 1
        payload = (
            {"time_index": i % 24,
             "values": {f"var_{i % 24}": float(i % 30)}}
            if i % 2 == 0
            else {"time_start": i % 12, "time_stop": i % 12 + 4,
                  "values": {f"var_{i % 24}": float(i % 30)}}
        )
        ops.append(make_op(seq, "meteorological_forcing", "set",
                           actor=("alice", "carol")[i % 2], epoch=1,
                           payload=payload))
    for i in range(800):  # parameter traffic
        seq += 1
        ops.append(make_op(seq, "model_receptor_parameters", "set",
                           actor="dave", epoch=1,
                           payload={"values": {f"param_{i % 40}": 0.1 + (i % 50) / 100}}))
    for i in range(200):  # view traffic
        seq += 1
        ops.append(make_op(seq, "output_view", "select",
                           actor=("alice", "bob")[i % 2], epoch=1,
                           payload={"values": {"layer": ("utci", "tmrt")[i % 2],
                                               "time_index": i % 24}}))
    assert len(ops) == 10000
    assert len({op.server_sequence for op in ops}) == 10000

    started = time.perf_counter()
    result = reducer.reduce_epoch(base, ops)
    elapsed = time.perf_counter() - started
    assert len(result.reduced.audit_operations) == 10000
    assert result.reduced.workspace_revision == 1
    assert elapsed < 2.0, f"10k-operation reduce took {elapsed:.3f}s (bound 2s)"
