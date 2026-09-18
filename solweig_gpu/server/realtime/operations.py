# SPDX-License-Identifier: GPL-3.0-only
"""Durable idempotent operation log + epoch assignment (R1 operation plane).

This module owns the TRANSPORT layer of the realtime operation plane
(``docs/incremental_design_tool/realtime_collaboration/collaborative_state.md``
+ ``epoch_scheduler.md``):

* the frozen vocabularies re-derived from :mod:`.types` (families, verbs,
  epoch statuses) so schema and interface cannot drift;
* transport-level shape validation for submitted operations (family/verb
  membership, non-empty payload). Family PAYLOAD validation is NOT
  restated here — it stays with the executor bridge's adapter registry,
  exactly like the universal edit transport;
* response rendering for the operation/epoch routes.

Durability lives in :meth:`solweig_gpu.server.store.Store.append_operations`
(one ``BEGIN IMMEDIATE`` transaction per batch under the store's single
connection + RLock; per-workspace contiguous ``server_sequence``; globally
unique ``operation_id`` as the idempotency key; epoch assignment to the
workspace's open epoch). R2's scheduler builds closure on top; this module
only exposes the status-transition vocabulary it needs.
"""

from __future__ import annotations

from typing import Any, Mapping, get_args

from solweig_gpu.server.models import ApiError
from solweig_gpu.server.realtime.types import (
    EpochStatus,
    EpochState,
    OperationVerb,
    SourceFamily,
)
from solweig_gpu.server.store import EpochRecord, OperationRecord

__all__ = [
    "EPOCH_STATUSES",
    "MAX_OPERATIONS_PER_BATCH",
    "OPERATION_VERBS",
    "SOURCE_FAMILIES",
    "TERMINAL_EPOCH_STATUSES",
    "epoch_body",
    "operation_body",
    "validate_operation_shape",
]

#: Admission bound for one operation batch (the request model enforces it;
#: re-stated here so the plane's contract is readable from one place).
MAX_OPERATIONS_PER_BATCH = 128

#: The 7 integrated/view families the plane can carry (types.SourceFamily).
#: ``dem`` and ``dynamic_wind_from_geometry`` are typed refusals at this
#: layer by design — they do not appear in the vocabulary at all.
SOURCE_FAMILIES: frozenset[str] = frozenset(get_args(SourceFamily))

#: Operation verbs (types.OperationVerb).
OPERATION_VERBS: frozenset[str] = frozenset(get_args(OperationVerb))

#: Epoch lifecycle vocabulary (types.EpochStatus).
EPOCH_STATUSES: frozenset[str] = frozenset(get_args(EpochStatus))

#: Terminal epoch statuses: no further transitions, no re-entry. Every
#: other status may be mid-flight at a crash, which is why restart
#: recovery (R2) treats non-terminal epochs as recoverable and this store
#: never auto-closes anything on open.
TERMINAL_EPOCH_STATUSES: frozenset[str] = frozenset(
    {"fast_published", "exact_targeted"}
)


def validate_operation_shape(item: Mapping[str, Any], *, index: int = 0) -> None:
    """Transport-level shape checks for one submitted operation.

    Raises a typed :class:`ApiError` (4xx envelope, the ``_conflict``
    convention) for: unknown ``source_family``, unknown ``verb``, or an
    empty ``payload``. Payload JSON-serializability is guaranteed by the
    request model (``allow_inf_nan=False``); payload byte size is bounded
    by the existing request-body cap enforced in the app middleware — this
    layer deliberately adds no second size restatement.
    """
    field = f"operations[{index}]"
    family = str(item.get("source_family", ""))
    if family not in SOURCE_FAMILIES:
        raise ApiError(
            "unknown_source_family",
            f"unknown source_family {family!r}; the realtime plane carries "
            f"the integrated/view families {sorted(SOURCE_FAMILIES)} only "
            "(dem and dynamic wind are typed refusals at this layer)",
            field=f"{field}.source_family",
        )
    verb = str(item.get("verb", ""))
    if verb not in OPERATION_VERBS:
        raise ApiError(
            "unknown_operation_verb",
            f"unknown operation verb {verb!r}; allowed: {sorted(OPERATION_VERBS)}",
            field=f"{field}.verb",
        )
    payload = item.get("payload")
    if not isinstance(payload, Mapping) or not dict(payload):
        raise ApiError(
            "invalid_operation_payload",
            "an operation payload must be a non-empty JSON object",
            field=f"{field}.payload",
        )


def operation_body(record: OperationRecord) -> dict[str, Any]:
    """Render one durable operation for the API (GET lookup / acks)."""
    return {
        "workspace_id": record.workspace_id,
        "operation_id": record.operation_id,
        "server_sequence": record.server_sequence,
        "epoch_id": record.epoch_id,
        "actor_id": record.actor_id,
        "client_sequence": record.client_sequence,
        "base_revision": record.base_revision,
        "source_family": record.source_family,
        "entity_id": record.entity_id,
        "verb": record.verb,
        "payload": dict(record.payload),
        "received_at": record.received_at,
        "accepted_at": record.accepted_at,
    }


def epoch_body(record: EpochRecord | EpochState) -> dict[str, Any]:
    """Render one epoch record for the API (``GET .../epochs``)."""
    return {
        "workspace_id": record.workspace_id,
        "epoch_id": record.epoch_id,
        "status": record.status,
        "first_sequence": record.first_sequence,
        "last_sequence": record.last_sequence,
        "workspace_revision": record.workspace_revision,
        "opened_at": record.opened_at,
        "closed_at": record.closed_at,
        "terminal": record.status in TERMINAL_EPOCH_STATUSES,
    }
