# SPDX-License-Identifier: GPL-3.0-only
"""Frozen contracts for the collaborative realtime operation plane.

Authored once by the lead (R1 interface freeze). Waves implement against
these types; semantic changes require a lead decision recorded in the
worklog. Names follow ``docs/incremental_design_tool/realtime_collaboration/
collaborative_state.md`` and ``realtime_contract.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

#: Edit families the universal plane can carry. Mirrors the integrated +
#: view-only adapters from edit_registry.yaml; ``dem`` and
#: ``dynamic_wind_from_geometry`` stay typed refusals at the transport layer
#: and therefore do not appear here.
SourceFamily = Literal[
    "building_geometry",
    "vegetation_geometry",
    "landcover_surface",
    "meteorological_forcing",
    "model_receptor_parameters",
    "selected_date_time",
    "output_view",
]

#: Operation verbs (collaborative_state.md). ``set`` covers scalar/range
#: value writes, ``select`` covers view/time selection.
OperationVerb = Literal[
    "add", "replace", "move", "delete", "paint", "set", "select"
]

#: Deadline-bound fast result classes (realtime_contract.yaml). The exact
#: lane publishes its own class independently of these.
FastResultClass = Literal[
    "fast_exact", "fast_qualified", "visual_pending"
]

#: Epoch lifecycle (epoch_scheduler.md).
EpochStatus = Literal[
    "open",
    "closed",
    "reducing",
    "fast_planned",
    "fast_published",
    "exact_targeted",
]


@dataclass(frozen=True)
class Operation:
    """One durably accepted collaborative operation.

    ``payload`` carries the family-specific typed data validated by the
    existing adapter registry (universal edit items / tree edit objects).
    The server assigns ``server_sequence`` (total order) and ``epoch_id``;
    client clocks never determine final ordering.
    """

    workspace_id: str
    operation_id: str
    actor_id: str
    client_sequence: int | None
    base_revision: int | None
    source_family: SourceFamily
    entity_id: str | None
    verb: OperationVerb
    payload: Mapping[str, Any]
    received_at: str
    accepted_at: str
    server_sequence: int
    epoch_id: int


class ConflictKind:
    """Conflict taxonomy reported by the reducer (never fatal — the reducer
    resolves deterministically and records what happened)."""

    LAST_WRITE_WINS = "last_write_wins"
    TOMBSTONE_REJECTED = "tombstone_rejected_update"
    GENERATION_RECREATED = "generation_recreated"
    RASTER_CHUNK_OVERWRITTEN = "raster_chunk_overwritten"
    RANGE_SEGMENTED = "range_segmented"
    COMMUTATIVE_MERGED = "commutative_merged"


@dataclass(frozen=True)
class ConflictRecord:
    kind: str
    source_family: SourceFamily
    entity_id: str | None
    winner_operation_id: str
    loser_operation_ids: tuple[str, ...]
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyDelta:
    """Epoch-final compact delta for one source family.

    ``commands`` holds zero or more already-validated family edit commands
    (the existing ``edit_types.EditCommand`` payload shapes) coalesced to the
    epoch-final state; an empty tuple with ``is_noop`` True means the
    epoch's operations cancelled out (e.g. add then delete of the same
    entity). Operations are never dropped from the audit to produce a noop.
    """

    family: SourceFamily
    is_noop: bool
    commands: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReducedEpoch:
    """Deterministic reduction of every operation assigned to one epoch."""

    workspace_id: str
    epoch_id: int
    workspace_revision: int
    family_deltas: Mapping[str, FamilyDelta]
    audit_operations: tuple[Operation, ...]
    conflicts: tuple[ConflictRecord, ...]

    @property
    def is_noop(self) -> bool:
        return all(delta.is_noop for delta in self.family_deltas.values())


@dataclass(frozen=True)
class EpochState:
    """Durable epoch record (realtime_epochs table)."""

    workspace_id: str
    epoch_id: int
    status: EpochStatus
    first_sequence: int | None
    last_sequence: int | None
    workspace_revision: int | None
    opened_at: str
    closed_at: str | None


@dataclass(frozen=True)
class CanonicalState:
    """Family state snapshot the reducer folds operations onto.

    v1 shape: per-family opaque mappings the reducer owns. Vegetation and
    building carry object tables (entity_id -> generation payload +
    tombstones); landcover carries chunked paint state; meteorology and
    model parameters carry segmented (field, timestep-range) values;
    output_view carries the workspace view document. The reducer never
    touches rasters — spatial work happens in the plan/solve stages.
    """

    workspace_id: str
    workspace_revision: int
    families: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


#: Re-exported alias: an epoch reduction result plus the canonical state it
#: produced (EpochReduction = ReducedEpoch + resulting CanonicalState).
@dataclass(frozen=True)
class EpochReduction:
    reduced: ReducedEpoch
    state: CanonicalState


class EpochReducer:
    """Pure deterministic reducer contract (implemented in ``reducer.py``).

    ``reduce_epoch`` MUST:

    - consume ``operations`` sorted by ``server_sequence``;
    - include EVERY operation in ``audit_operations`` even when cancelled
      out (add-then-delete yields a noop FamilyDelta, both ops audited);
    - resolve conflicts per family semantics (object generations/tombstones,
      raster chunk last-write-wins, scalar/range segmentation, view select);
    - produce the same output for the same input on every call (replay
      determinism; no clock, no RNG, no dict-ordering dependence);
    - raise nothing for semantically valid input — ordering conflicts are
      data, not errors.
    """

    def initial_state(self, workspace_id: str) -> CanonicalState:
        raise NotImplementedError

    def reduce_epoch(
        self, baseline: CanonicalState, operations: Sequence[Operation]
    ) -> EpochReduction:
        raise NotImplementedError


__all__ = [
    "CanonicalState",
    "ConflictKind",
    "ConflictRecord",
    "EpochReduction",
    "EpochReducer",
    "EpochState",
    "EpochStatus",
    "FastResultClass",
    "FamilyDelta",
    "Operation",
    "OperationVerb",
    "ReducedEpoch",
    "SourceFamily",
]
