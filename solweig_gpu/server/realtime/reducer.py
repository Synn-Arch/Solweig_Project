# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic source-specific epoch reduction (collaborative_state.md).

Pure function, no I/O: same ``(baseline, operations)`` produce equal
:class:`~solweig_gpu.server.realtime.types.EpochReduction` values on every
call (replay determinism — no clock, no RNG, no dict-ordering dependence;
every iteration at a serialization boundary walks sorted keys).

Family semantics (conflicts are data, never exceptions):

* **Object geometry** (``vegetation_geometry`` / ``building_geometry``):
  ``add`` creates generation 0 on a fresh entity id; an add over an
  existing (live or tombstoned) id bumps the generation and records
  ``GENERATION_RECREATED`` (an add is the only resurrect path — spec:
  "delete followed by update requires a new generation or is rejected").
  ``delete`` tombstones the current generation (delete of an unknown or
  already-tombstoned entity is audited with no state change).
  ``replace``/``update`` fold field-level last-write-wins by
  ``server_sequence``; ``move`` is the same fold restricted to the
  family's position/footprint fields. A replace/move targeting a
  tombstoned entity is recorded as ``TOMBSTONE_REJECTED`` and never
  resurrects. Concurrent writes to one field by two or more distinct
  actors record ``LAST_WRITE_WINS`` (winner = highest sequence).
* **Raster paint** (``landcover_surface``): per-cell last-write-wins by
  ``server_sequence`` inside fixed 64 px chunk buckets. Two or more
  distinct actors painting into the same chunk record
  ``RASTER_CHUNK_OVERWRITTEN`` (winner = the chunk's last op).
* **Scalar/range** (``meteorological_forcing``,
  ``model_receptor_parameters``, ``selected_date_time``): per-field
  segment maps normalized disjoint and *canonical* — inserting a range
  carves overlapping coverage (later value wins per timestep) and
  adjacent segments with equal values merge, so any fold order reaching
  the same (field, timestep -> value) function yields the same state.
  Cross-actor overlap of multi-timestep ranges records
  ``RANGE_SEGMENTED``; cross-actor overwrite of a single timestep
  records ``LAST_WRITE_WINS``. A write with no time bounds covers the
  unbounded segment ``(0, None)`` (whole-series reset/preset).
* **View** (``output_view``): workspace-wide ``select`` is last-write on
  the whole view document (cross-actor races record ``LAST_WRITE_WINS``);
  ``scope: "user"`` selects land in ``per_user`` substate which NEVER
  enters the family delta (per-user view is not scientific state).

Operation payload grammar (mirrors the universal edit item,
``solweig_gpu/server/universal.py``): field values live under
``payload["values"]`` (a flat payload is accepted as the field mapping),
a land-cover paint carries ``window`` (half-open
``{row_start, row_stop, col_start, col_stop}``) or ``cells`` plus
``class`` under ``payload`` or ``payload["values"]``, and scalar writes
carry ``time_index`` or inclusive ``time_start``/``time_stop`` at the
payload top level. Identity keys (``tree_id``/``building_id``) are
stripped from stored fields — the entity id is the state key. Payloads
the reducers does not understand (wrong verb for the family, missing
window/class, non-integer bounds) are audited with no state change:
domain validation belongs to the adapter registry, which accepted the
operation before it reached the log. The one hard error is a duplicated
``server_sequence`` (the store owns dedup; the reducer asserts the total
order it folds is strict) — ``ValueError``.

Canonical state v1 shapes (``CanonicalState.families``, opaque to every
other module):

.. code-block:: python

    building_geometry | vegetation_geometry:
        {"objects": {entity_id: {**fields, "generation": int}},
         "tombstones": {entity_id: generation}}
    landcover_surface:
        {"chunks": {f"{row//64}:{col//64}": {f"{row}:{col}": class_index}}}
    meteorological_forcing | model_receptor_parameters | selected_date_time:
        {"segments": {field: [(start_t, end_t, value), ...]}}  # disjoint, sorted, canonical
    output_view:
        {"view": {...last workspace selection...}, "per_user": {actor_id: {...}}}

``FamilyDelta.commands`` are universal-edit-item-shaped mappings
(``adapter``/``operation``/``target``/``values``/``old_values``) holding
the epoch-final net change versus the baseline
(:func:`solweig_gpu.server.universal.command_from_item` maps them onto
:class:`~solweig_gpu.incremental.edit_types.EditCommand` records);
``summary`` carries per-family counts. Every accepted operation appears
in ``audit_operations`` in ``server_sequence`` order regardless of
cancellation.
"""

from __future__ import annotations

from typing import Any, Mapping

from solweig_gpu.server.realtime.types import (
    CanonicalState,
    ConflictKind,
    ConflictRecord,
    EpochReduction,
    EpochReducer,
    FamilyDelta,
    Operation,
    ReducedEpoch,
)

__all__ = ["CHUNK_SIZE_PX", "DeterministicEpochReducer"]

#: Fixed raster chunk edge (pixels) for land-cover paint state/conflicts.
CHUNK_SIZE_PX = 64

_OBJECT_FAMILIES = ("building_geometry", "vegetation_geometry")
_SEGMENT_FAMILIES = (
    "meteorological_forcing",
    "model_receptor_parameters",
    "selected_date_time",
)
#: Identity keys never stored inside object fields (entity id is the key).
_IDENTITY_KEYS = {
    "building_geometry": "building_id",
    "vegetation_geometry": "tree_id",
}
#: Move = replacement of position/footprint fields only (spec).
_POSITION_FIELDS = {
    "vegetation_geometry": ("u", "v", "x_m", "y_m"),
    "building_geometry": ("footprint_m",),
}
#: Keys never treated as object fields when a payload arrives flat.
_FLAT_PAYLOAD_SKIP = frozenset(
    ("values", "target", "old_values", "time_index", "time_start",
     "time_stop", "scope", "window", "cells", "class", "value")
)

_ALL_FAMILIES = _OBJECT_FAMILIES + (
    "landcover_surface",
    *_SEGMENT_FAMILIES,
    "output_view",
)


# ---------------------------------------------------------------------------
# Family state shapes
# ---------------------------------------------------------------------------


def _empty_family_state(family: str) -> dict[str, Any]:
    if family in _OBJECT_FAMILIES:
        return {"objects": {}, "tombstones": {}}
    if family == "landcover_surface":
        return {"chunks": {}}
    if family in _SEGMENT_FAMILIES:
        return {"segments": {}}
    if family == "output_view":
        return {"view": {}, "per_user": {}}
    return {}


def _copy_families(families: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Structural copy of a canonical ``families`` mapping (never shared).

    Key iteration is sorted so the rebuilt mapping's insertion order is a
    pure function of content, not of the baseline's construction history.
    """
    work: dict[str, Any] = {}
    for family in sorted(families):
        state = families[family] or {}
        if family in _OBJECT_FAMILIES:
            work[family] = {
                "objects": {
                    eid: dict(fields)
                    for eid, fields in sorted(state.get("objects", {}).items())
                },
                "tombstones": dict(state.get("tombstones", {})),
            }
        elif family == "landcover_surface":
            work[family] = {
                "chunks": {
                    chunk: dict(cells)
                    for chunk, cells in sorted(state.get("chunks", {}).items())
                }
            }
        elif family in _SEGMENT_FAMILIES:
            work[family] = {
                "segments": {
                    field: [tuple(segment) for segment in segments]
                    for field, segments in sorted(state.get("segments", {}).items())
                }
            }
        elif family == "output_view":
            work[family] = {
                "view": dict(state.get("view", {})),
                "per_user": {
                    actor: dict(selection)
                    for actor, selection in sorted(
                        state.get("per_user", {}).items()
                    )
                },
            }
        else:
            work[family] = dict(state)
    for family in _ALL_FAMILIES:
        work.setdefault(family, _empty_family_state(family))
    return work


# ---------------------------------------------------------------------------
# Payload accessors (universal-item grammar; validation stays upstream)
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _item_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Field mapping of an operation: ``values`` nesting or flat payload."""
    nested = payload.get("values")
    if isinstance(nested, Mapping):
        return dict(nested)
    return {k: v for k, v in payload.items() if k not in _FLAT_PAYLOAD_SKIP}


def _object_fields(family: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = _item_values(payload)
    identity = _IDENTITY_KEYS.get(family)
    if identity is not None and identity in fields:
        fields = {k: v for k, v in fields.items() if k != identity}
    return fields


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _paint_class(payload: Mapping[str, Any]) -> int | None:
    for candidate in (
        payload.get("class"),
        payload.get("value"),
        _mapping(payload.get("values")).get("class"),
    ):
        if _is_index(candidate):
            return int(candidate)
    return None


def _cell_key(row: int, col: int) -> tuple[str, str]:
    return (f"{row // CHUNK_SIZE_PX}:{col // CHUNK_SIZE_PX}", f"{row}:{col}")


def _chunk_sort_key(chunk_key: str) -> tuple[int, int]:
    row_block, _, col_block = chunk_key.partition(":")
    return (int(row_block), int(col_block))


def _cell_sort_key(cell_key: str) -> tuple[int, int]:
    row, _, col = cell_key.partition(":")
    return (int(row), int(col))


def _paint_cells(payload: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    """Cells of one paint: a half-open brush window or an explicit list."""
    window = payload.get("window")
    if not isinstance(window, Mapping):
        target = payload.get("target")
        if isinstance(target, Mapping) and "row_start" in target:
            window = target
    cells: set[tuple[int, int]] = set()
    if isinstance(window, Mapping) and all(
        _is_index(window.get(k)) for k in ("row_start", "row_stop", "col_start", "col_stop")
    ):
        rows = range(int(window["row_start"]), int(window["row_stop"]))
        cols = range(int(window["col_start"]), int(window["col_stop"]))
        cells.update((row, col) for row in rows for col in cols)
    raw_cells = payload.get("cells")
    if raw_cells is None:
        raw_cells = _mapping(payload.get("target")).get("cells")
    if isinstance(raw_cells, (list, tuple)):
        for item in raw_cells:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(_is_index(part) for part in item)
            ):
                cells.add((int(item[0]), int(item[1])))
    return tuple(sorted(cells))


def _time_bounds(payload: Mapping[str, Any]) -> tuple[int, int | None] | None:
    """Inclusive ``(start, stop)`` of a scalar/range write (None = unbounded).

    ``time_index`` edits one timestep ``(t, t)``; ``time_start``/``time_stop``
    edit the closed range ``[start, stop]`` (the met adapter's inclusive
    convention); neither covers the whole series ``(0, None)``.
    """
    values = _mapping(payload.get("values"))
    start = payload.get("time_start", values.get("time_start"))
    stop = payload.get("time_stop", values.get("time_stop"))
    index = payload.get("time_index", values.get("time_index"))
    if start is not None or stop is not None:
        if not _is_index(start) or not _is_index(stop) or stop < start:
            return None
        return (int(start), int(stop))
    if index is not None:
        if not _is_index(index):
            return None
        return (int(index), int(index))
    return (0, None)


# ---------------------------------------------------------------------------
# Segment algebra (disjoint, canonical, closed intervals; None stop = open)
# ---------------------------------------------------------------------------


def _seg_stop(stop: int | None) -> float:
    return float("inf") if stop is None else float(stop)


def _overlaps(
    start_a: int, stop_a: int | None, start_b: int, stop_b: int | None
) -> bool:
    return start_a <= _seg_stop(stop_b) and start_b <= _seg_stop(stop_a)


def _insert_segment(
    family: str,
    segments_map: dict[str, list[tuple[int, int | None, Any]]],
    owners: dict[str, list[Any]],
    field: str,
    start: int,
    stop: int | None,
    value: Any,
    op: Operation,
    conflicts: list[ConflictRecord],
) -> None:
    """Carve ``[start, stop]`` out of ``field`` coverage, then insert.

    Later writes win per timestep (the carve removes overlapping coverage
    before the insert), surviving baseline pieces keep their value/owner,
    and adjacent equal-value segments merge so the segment list is a
    canonical function of the (timestep -> value) mapping.
    """
    segments = segments_map.get(field, [])
    field_owners = owners.setdefault(field, [None] * len(segments))
    kept: list[tuple[int, int | None, Any]] = []
    kept_owners: list[Any] = []
    for segment, owner in zip(segments, field_owners):
        seg_start, seg_stop, seg_value = segment
        if not _overlaps(start, stop, seg_start, seg_stop):
            kept.append(segment)
            kept_owners.append(owner)
            continue
        if owner is not None and owner.actor_id != op.actor_id:
            ranged = start < stop if stop is not None else True
            seg_ranged = seg_start < seg_stop if seg_stop is not None else True
            kind = (
                ConflictKind.RANGE_SEGMENTED
                if ranged or seg_ranged
                else ConflictKind.LAST_WRITE_WINS
            )
            conflicts.append(
                ConflictRecord(
                    kind=kind,
                    source_family=family,
                    entity_id=field,
                    winner_operation_id=op.operation_id,
                    loser_operation_ids=(owner.operation_id,),
                    detail={
                        "field": field,
                        "inserted": [start, stop],
                        "overlapped": [seg_start, seg_stop],
                    },
                )
            )
        if seg_start < start:
            kept.append((seg_start, start - 1, seg_value))
            kept_owners.append(owner)
        # An unbounded insert (stop is None) covers [start, inf): no right
        # remainder survives. Otherwise a segment ending past the insert
        # keeps [stop + 1, seg_stop] (seg_stop None = still unbounded).
        if stop is not None and (seg_stop is None or seg_stop > stop):
            kept.append((stop + 1, seg_stop, seg_value))
            kept_owners.append(owner)
    kept.append((start, stop, value))
    kept_owners.append(op)
    order = sorted(range(len(kept)), key=lambda i: kept[i][0])
    kept = [kept[i] for i in order]
    kept_owners = [kept_owners[i] for i in order]
    merged: list[tuple[int, int | None, Any]] = []
    merged_owners: list[Any] = []
    for segment, owner in zip(kept, kept_owners):
        if (
            merged
            and merged[-1][2] == segment[2]
            and merged[-1][1] is not None
            and merged[-1][1] + 1 == segment[0]
        ):
            prev_owner = merged_owners[-1]
            merged[-1] = (merged[-1][0], segment[1], segment[2])
            merged_owners[-1] = owner if prev_owner is None else prev_owner
            continue
        merged.append(segment)
        merged_owners.append(owner)
    segments_map[field] = merged
    owners[field] = merged_owners


# ---------------------------------------------------------------------------
# Epoch fold bookkeeping
# ---------------------------------------------------------------------------


class _Fold:
    """Epoch-internal tracking that never leaks into canonical state."""

    def __init__(self) -> None:
        self.conflicts: list[ConflictRecord] = []
        # object families
        self.add_ops: dict[str, list[Operation]] = {}
        self.delete_ops: dict[str, Operation] = {}
        self.field_writers: dict[tuple[str, str, str], list[Operation]] = {}
        # landcover
        self.chunk_ops: dict[str, list[Operation]] = {}
        self.chunk_cell_actors: dict[tuple[str, str], set[str]] = {}
        # segments
        self.segment_owners: dict[str, list[Any]] = {}
        # view
        self.view_writers: list[Operation] = []
        self.per_user_updates = 0


def _fold_object(
    family: str, state: dict[str, Any], op: Operation, fold: _Fold
) -> None:
    entity_id = op.entity_id
    if not entity_id:
        return  # malformed upstream; audited only
    objects: dict[str, dict[str, Any]] = state["objects"]
    tombstones: dict[str, int] = state["tombstones"]
    fields = _object_fields(family, _mapping(op.payload))
    verb = op.verb
    if verb == "add":
        was_live = entity_id in objects
        was_tombstoned = entity_id in tombstones
        if was_live:
            previous = int(objects[entity_id]["generation"])
            generation = previous + 1
        elif was_tombstoned:
            previous = int(tombstones[entity_id])
            generation = previous + 1
            del tombstones[entity_id]
        else:
            previous = None
            generation = 0
        if previous is not None:
            prior_adds = fold.add_ops.get(entity_id, ())
            fold.conflicts.append(
                ConflictRecord(
                    kind=ConflictKind.GENERATION_RECREATED,
                    source_family=family,
                    entity_id=entity_id,
                    winner_operation_id=op.operation_id,
                    loser_operation_ids=(
                        (prior_adds[-1].operation_id,) if prior_adds else ()
                    ),
                    detail={
                        "from_generation": previous,
                        "to_generation": generation,
                        "resurrected": was_tombstoned,
                    },
                )
            )
        fold.add_ops.setdefault(entity_id, []).append(op)
        objects[entity_id] = {**fields, "generation": generation}
        # the add's fields are writes too: another actor replacing them
        # later in the epoch is a cross-actor last-write-wins race
        for name in sorted(fields):
            fold.field_writers.setdefault((family, entity_id, name), []).append(op)
        return
    if verb in ("replace", "update", "move"):
        live = objects.get(entity_id)
        if live is None:
            if entity_id in tombstones:
                delete_op = fold.delete_ops.get(entity_id)
                fold.conflicts.append(
                    ConflictRecord(
                        kind=ConflictKind.TOMBSTONE_REJECTED,
                        source_family=family,
                        entity_id=entity_id,
                        winner_operation_id=(
                            delete_op.operation_id
                            if delete_op is not None
                            else op.operation_id
                        ),
                        loser_operation_ids=(op.operation_id,),
                        detail={
                            "targeting_generation": int(tombstones[entity_id]),
                            "winner_is_epoch_delete": delete_op is not None,
                        },
                    )
                )
            return  # unknown or tombstoned: audited, no state change
        if verb == "move":
            allowed = _POSITION_FIELDS.get(family, ())
            fields = {k: v for k, v in fields.items() if k in allowed}
        for name in sorted(fields):
            fold.field_writers.setdefault((family, entity_id, name), []).append(op)
            live[name] = fields[name]
        return
    if verb == "delete":
        live = objects.pop(entity_id, None)
        if live is not None:
            tombstones[entity_id] = int(live["generation"])
            fold.delete_ops[entity_id] = op
        # delete of an unknown/tombstoned entity: audited, no state change


def _fold_paint(state: dict[str, Any], op: Operation, fold: _Fold) -> None:
    payload = _mapping(op.payload)
    code = _paint_class(payload)
    if code is None:
        return
    chunks: dict[str, dict[str, int]] = state["chunks"]
    for row, col in _paint_cells(payload):
        chunk_key, cell_key = _cell_key(row, col)
        bucket = chunks.get(chunk_key)
        if bucket is None:
            bucket = chunks[chunk_key] = {}
        bucket[cell_key] = code  # later server_sequence wins per cell
        fold.chunk_ops.setdefault(chunk_key, []).append(op)
        fold.chunk_cell_actors.setdefault((chunk_key, cell_key), set()).add(
            op.actor_id
        )


def _fold_segments(
    family: str, state: dict[str, Any], op: Operation, fold: _Fold
) -> None:
    payload = _mapping(op.payload)
    bounds: tuple[int, int | None]
    if family == "selected_date_time":
        # the selection IS the value: a selected timestep is content, not a
        # temporal bound, so every write covers the unbounded segment
        bounds = (0, None)
    else:
        bounds = _time_bounds(payload)
        if bounds is None:
            return
    start, stop = bounds
    values = _item_values(payload)
    segments_map: dict[str, list[tuple[int, int | None, Any]]] = state["segments"]
    for field in sorted(values):
        _insert_segment(
            family,
            segments_map,
            fold.segment_owners,
            field,
            start,
            stop,
            values[field],
            op,
            fold.conflicts,
        )


def _fold_view(state: dict[str, Any], op: Operation, fold: _Fold) -> None:
    payload = _mapping(op.payload)
    nested = payload.get("values")
    if isinstance(nested, Mapping):
        selection = dict(nested)
    else:
        selection = {k: v for k, v in payload.items() if k != "scope"}
    ordered = {k: selection[k] for k in sorted(selection)}
    if payload.get("scope") == "user":
        state["per_user"][op.actor_id] = ordered
        fold.per_user_updates += 1
    else:
        state["view"] = ordered
        fold.view_writers.append(op)


# ---------------------------------------------------------------------------
# Family deltas (net change versus baseline, universal-item-shaped commands)
# ---------------------------------------------------------------------------


def _object_delta(
    family: str,
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    fold: _Fold,
) -> FamilyDelta:
    base_objects: Mapping[str, dict[str, Any]] = baseline.get("objects", {})
    final_objects: Mapping[str, dict[str, Any]] = final.get("objects", {})
    position = _POSITION_FIELDS.get(family, ())
    commands: list[dict[str, Any]] = []
    counts = {"objects_added": 0, "objects_moved": 0, "objects_replaced": 0,
              "objects_deleted": 0}
    for entity_id in sorted(set(base_objects) | set(final_objects)):
        base_entry = base_objects.get(entity_id)
        final_entry = final_objects.get(entity_id)
        base_fields = (
            {k: v for k, v in sorted(base_entry.items()) if k != "generation"}
            if base_entry is not None
            else None
        )
        final_fields = (
            {k: v for k, v in sorted(final_entry.items()) if k != "generation"}
            if final_entry is not None
            else None
        )
        if final_fields is not None and base_fields is None:
            operation = "add"
            counts["objects_added"] += 1
        elif final_fields is None and base_fields is not None:
            operation = "delete"
            counts["objects_deleted"] += 1
        elif final_fields == base_fields:
            continue  # physically identical (generation-only change)
        else:
            changed = {
                name
                for name in set(base_fields) | set(final_fields)
                if base_fields.get(name) != final_fields.get(name)
            }
            operation = "move" if changed <= set(position) else "replace"
            counts["objects_moved" if operation == "move" else "objects_replaced"] += 1
        commands.append(
            {
                "adapter": family,
                "operation": operation,
                "target": entity_id,
                "values": final_fields if final_fields is not None else {},
                "old_values": base_fields,
                "time_index": None,
            }
        )
    return FamilyDelta(
        family=family,
        is_noop=not commands,
        commands=tuple(commands),
        summary=dict(sorted(counts.items())),
    )


def _landcover_delta(
    family: str,
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    fold: _Fold,
) -> FamilyDelta:
    base_chunks: Mapping[str, dict[str, int]] = baseline.get("chunks", {})
    final_chunks: Mapping[str, dict[str, int]] = final.get("chunks", {})
    commands: list[dict[str, Any]] = []
    cells_changed = 0
    chunks_changed = 0
    for chunk_key in sorted(final_chunks, key=_chunk_sort_key):
        final_cells = final_chunks[chunk_key]
        base_cells = base_chunks.get(chunk_key, {})
        diff = {
            cell: code for cell, code in final_cells.items()
            if base_cells.get(cell) != code
        }
        if not diff:
            continue
        chunks_changed += 1
        cells_changed += len(diff)
        by_class: dict[int, list[tuple[int, int]]] = {}
        for cell in sorted(diff, key=_cell_sort_key):
            row, _, col = cell.partition(":")
            by_class.setdefault(diff[cell], []).append((int(row), int(col)))
        for code in sorted(by_class):
            commands.append(
                {
                    "adapter": family,
                    "operation": "paint",
                    "target": {
                        "chunk": chunk_key,
                        "cells": [[row, col] for row, col in by_class[code]],
                    },
                    "values": {"class": code},
                    "old_values": None,
                    "time_index": None,
                }
            )
    return FamilyDelta(
        family=family,
        is_noop=not commands,
        commands=tuple(commands),
        summary={
            "cells_changed": cells_changed,
            "chunks_changed": chunks_changed,
        },
    )


def _segment_delta(
    family: str,
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    fold: _Fold,
) -> FamilyDelta:
    base_segments: Mapping[str, list] = baseline.get("segments", {})
    final_segments: Mapping[str, list] = final.get("segments", {})
    commands: list[dict[str, Any]] = []
    fields_changed = 0
    segments_changed = 0
    for field in sorted(final_segments):
        segments = [tuple(s) for s in final_segments[field]]
        if segments == [tuple(s) for s in base_segments.get(field, [])]:
            continue
        fields_changed += 1
        segments_changed += len(segments)
        commands.append(
            {
                "adapter": family,
                "operation": "set",
                "target": field,
                "values": {
                    "segments": [
                        [start, stop, value] for start, stop, value in segments
                    ]
                },
                "old_values": {
                    "segments": [
                        [start, stop, value]
                        for start, stop, value in (
                            tuple(s) for s in base_segments.get(field, [])
                        )
                    ]
                },
                "time_index": None,
            }
        )
    return FamilyDelta(
        family=family,
        is_noop=not commands,
        commands=tuple(commands),
        summary={
            "fields_changed": fields_changed,
            "segments_changed": segments_changed,
        },
    )


def _view_delta(
    family: str,
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    fold: _Fold,
) -> FamilyDelta:
    base_view = baseline.get("view", {})
    final_view = final.get("view", {})
    changed = final_view != base_view
    commands: tuple[dict[str, Any], ...] = (
        (
            {
                "adapter": family,
                "operation": "select",
                "target": None,
                "values": dict(final_view),
                "old_values": dict(base_view) if base_view else None,
                "time_index": None,
            },
        )
        if changed
        else ()
    )
    return FamilyDelta(
        family=family,
        is_noop=not changed,  # per-user substate never enters the delta
        commands=commands,
        summary={
            "workspace_view_changed": 1 if changed else 0,
            "per_user_updates": fold.per_user_updates,
        },
    )


def _aggregate_conflicts(fold: _Fold) -> list[ConflictRecord]:
    """Conflicts computed after the fold, in sorted-key order."""
    records: list[ConflictRecord] = []
    for (family, entity_id, field), writers in sorted(fold.field_writers.items()):
        if len({op.actor_id for op in writers}) < 2:
            continue
        ordered = list(writers)  # fold runs in server_sequence order already
        records.append(
            ConflictRecord(
                kind=ConflictKind.LAST_WRITE_WINS,
                source_family=family,
                entity_id=entity_id,
                winner_operation_id=ordered[-1].operation_id,
                loser_operation_ids=tuple(
                    op.operation_id for op in ordered[:-1]
                ),
                detail={"field": field},
            )
        )
    for chunk_key in sorted(fold.chunk_ops, key=_chunk_sort_key):
        # one op appends once per painted cell; dedup by operation id in
        # first-touch order (== server_sequence order: the fold is ordered)
        ops_by_id: dict[str, Operation] = {}
        for op in fold.chunk_ops[chunk_key]:
            ops_by_id.setdefault(op.operation_id, op)
        ops = list(ops_by_id.values())
        if len({op.actor_id for op in ops}) < 2:
            continue
        cells_conflicted = sum(
            1
            for (chunk, _cell), actors in fold.chunk_cell_actors.items()
            if chunk == chunk_key and len(actors) >= 2
        )
        records.append(
            ConflictRecord(
                kind=ConflictKind.RASTER_CHUNK_OVERWRITTEN,
                source_family="landcover_surface",
                entity_id=chunk_key,
                winner_operation_id=ops[-1].operation_id,
                loser_operation_ids=tuple(op.operation_id for op in ops[:-1]),
                detail={
                    "chunk": chunk_key,
                    "cells_conflicted": cells_conflicted,
                },
            )
        )
    if len({op.actor_id for op in fold.view_writers}) >= 2:
        records.append(
            ConflictRecord(
                kind=ConflictKind.LAST_WRITE_WINS,
                source_family="output_view",
                entity_id=None,
                winner_operation_id=fold.view_writers[-1].operation_id,
                loser_operation_ids=tuple(
                    op.operation_id for op in fold.view_writers[:-1]
                ),
                detail={"scope": "workspace"},
            )
        )
    return records


# ---------------------------------------------------------------------------
# The reducer
# ---------------------------------------------------------------------------


class DeterministicEpochReducer(EpochReducer):
    """Pure per-family fold of one epoch's operations (see module docs)."""

    def initial_state(self, workspace_id: str) -> CanonicalState:
        return CanonicalState(
            workspace_id=workspace_id,
            workspace_revision=0,
            families={family: _empty_family_state(family)
                      for family in _ALL_FAMILIES},
        )

    def reduce_epoch(
        self, baseline: CanonicalState, operations
    ) -> EpochReduction:
        ops = list(operations)
        seen: set[int] = set()
        for op in ops:
            if op.server_sequence in seen:
                raise ValueError(
                    "server_sequence values must be unique within an epoch; "
                    f"duplicate {op.server_sequence} (deduplication is the "
                    "operation store's job, the reducer only folds the "
                    "total order)"
                )
            seen.add(op.server_sequence)
        ordered = sorted(ops, key=lambda op: op.server_sequence)

        work = _copy_families(baseline.families)
        fold = _Fold()
        present: list[str] = []
        for op in ordered:
            family = op.source_family
            if family not in present:
                present.append(family)
            if family in _OBJECT_FAMILIES:
                _fold_object(family, work[family], op, fold)
            elif family == "landcover_surface":
                if op.verb == "paint":
                    _fold_paint(work[family], op, fold)
            elif family in _SEGMENT_FAMILIES:
                if op.verb in ("set", "select", "update"):
                    _fold_segments(family, work[family], op, fold)
            elif family == "output_view":
                if op.verb == "select":
                    _fold_view(work[family], op, fold)
            # any other family/verb combination: audited only

        builders = {
            **{family: _object_delta for family in _OBJECT_FAMILIES},
            "landcover_surface": _landcover_delta,
            **{family: _segment_delta for family in _SEGMENT_FAMILIES},
            "output_view": _view_delta,
        }
        family_deltas = {
            family: builders[family](
                family, baseline.families.get(family) or {}, work[family], fold
            )
            for family in sorted(present)
        }
        revision = baseline.workspace_revision + (1 if ordered else 0)
        reduced = ReducedEpoch(
            workspace_id=baseline.workspace_id,
            epoch_id=ordered[-1].epoch_id if ordered else 0,
            workspace_revision=revision,
            family_deltas=family_deltas,
            audit_operations=tuple(ordered),
            conflicts=tuple(fold.conflicts + _aggregate_conflicts(fold)),
        )
        state = CanonicalState(
            workspace_id=baseline.workspace_id,
            workspace_revision=revision,
            families=work,
        )
        return EpochReduction(reduced=reduced, state=state)
