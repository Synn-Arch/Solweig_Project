# Collaborative state and epoch reducer

## Authoritative operation stream

Every client submits idempotent operations, not complete unordered scene files. Each operation carries:

```yaml
workspace_id: opaque identifier
operation_id: globally unique idempotency key
actor_id: session/user identifier
client_sequence: monotonic within actor session
base_revision: last server revision observed
hlc: optional hybrid logical clock supplied by client
source_family: building_geometry|vegetation_geometry|landcover_surface|meteorological_forcing|model_receptor_parameters|selected_date_time|output_view
entity_id: object, raster layer, parameter, or view id
operation: add|replace|move|delete|paint|set|select
payload: validated typed data
received_at: server timestamp
```

The server assigns a total `server_sequence` and epoch. Client clocks never determine final safety-critical ordering alone.

## Conflict semantics

Conflict behavior is defined by source type.

### Object geometry

Building and vegetation objects use stable entity IDs.

- Adds create an entity generation.
- Deletes create a tombstone for that generation.
- Property replacements use server-sequence last-write-wins per field unless a domain-specific merge is explicitly registered.
- Move is a replacement of position/footprint fields, not a separate untracked side effect.
- Delete followed by update requires a new generation or is rejected; an old client cannot resurrect a tombstoned generation accidentally.

### Raster paint

Land-cover and raster brush operations are split into fixed chunks. Within each cell/chunk, ordering follows `server_sequence`; later writes win unless the adapter declares a commutative merge. The reducer retains the original stroke audit but materializes one epoch-final sparse mask/value update.

### Scalar and temporal inputs

Meteorological, date/time, and model parameter values use last validated write per `(field, timestep/range)` with explicit units. Overlapping ranges are normalized into disjoint segments before planning.

### View state

Shared view operations may be workspace-wide or per-user. Per-user view changes never invalidate scientific state. Workspace-wide requested-output changes may schedule a missing scientific product but do not modify physical sources.

## Epoch reduction

At every epoch boundary:

1. Read all durably accepted operations assigned to the epoch.
2. Sort by `server_sequence`.
3. Reduce operations into the epoch's final canonical workspace state.
4. Produce one compact delta per source family.
5. Preserve old state required for invalidation, including deleted/moved geometry and overwritten raster chunks.
6. Combine heterogeneous deltas into one dependency-graph plan.
7. Increment `workspace_revision` once for the epoch, not once per source family.
8. Publish canonical state and collaboration diff.

All accepted operations are represented even when several cancel out. For example, add then delete may yield no final source delta, but both operations remain in the audit and the canonical revision records the epoch.

## Revision model

Track at least:

```text
workspace_revision: canonical merged edit state
fast_revision: newest revision with deadline-bound analysis representation
exact_revision: newest revision with exact SOLWEIG result
exact_base_revision: exact state used by a compensated fast result
```

Invariant:

```text
exact_revision <= fast_revision <= workspace_revision
```

A client can render current objects from `workspace_revision`, thermal data from `fast_revision`, and disclose whether `fast_revision` is exact, qualified, or pending.

## Deterministic replay

Given the same baseline, operation stream, registry versions, and model/cache hashes, replay must produce the same canonical state and dependency plan. Add property-based and permutation tests for operations declared commutative. For non-commutative operations, tests verify server-sequence determinism.
