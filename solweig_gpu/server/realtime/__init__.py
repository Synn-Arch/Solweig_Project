# SPDX-License-Identifier: GPL-3.0-only
"""Collaborative realtime operation plane (R1+).

Package layout (one writer per file, see docs/incremental_design_tool/
realtime_collaboration/* and the R0 gap register):

- ``types``      — frozen operation/epoch/revision contracts (THIS file's
                   exports; authored by the lead, shared by every wave).
- ``operations`` — durable idempotent operation log + epoch assignment
                   (store migration v7).
- ``reducer``    — deterministic source-specific epoch reduction (pure).
- ``epochs``     — 100 ms per-workspace epoch state machine.
- ``broadcast``  — SSE canonical-revision / result-class fan-out.
- ``telemetry``  — solweig_rt_* latency-stage histograms (outside the store
                   RLock).

The revision model (collaborative_state.md):

    workspace_revision  canonical merged edit state (scenario scene_version)
    fast_revision       newest revision with deadline-bound analysis
    exact_revision      newest revision with an exact result
    exact_base_revision exact state a compensated fast result is anchored to

Invariant: ``exact_revision <= fast_revision <= workspace_revision``.
"""

from solweig_gpu.server.realtime.types import (
    CanonicalState,
    ConflictKind,
    ConflictRecord,
    EpochReduction,
    EpochReducer,
    EpochState,
    EpochStatus,
    FastResultClass,
    FamilyDelta,
    Operation,
    OperationVerb,
    ReducedEpoch,
    SourceFamily,
)

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
