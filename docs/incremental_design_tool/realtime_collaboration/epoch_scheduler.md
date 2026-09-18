# Epoch scheduler and queue semantics

## Fixed micro-epochs

Use a 100 ms scheduler tick rather than collecting edits for a full second. A one-second collection window leaves no compute budget. Ten micro-epochs allow new edits to enter quickly while preserving up to roughly 900 ms for fast planning and computation.

Each workspace has an epoch state machine:

```text
OPEN -> CLOSED/REDUCING -> FAST_PLANNED -> FAST_PUBLISHED
                                  \-> EXACT_TARGETED
```

Epochs close on schedule even if no edits arrive. Empty epochs do not create computation.

## Queue separation

Use four logical queues:

1. `ingest`: idempotent durable operations awaiting epoch assignment.
2. `reduce`: closed workspace epochs awaiting deterministic reduction.
3. `fast`: deadline-ordered fast plans; bounded and admission controlled.
4. `exact`: latest-state reconciliation targets; revision chasing and supersession aware.

A single generic FIFO scientific queue is insufficient because a 38-second exact job would block many one-second updates.

## Fast queue

- Schedule by absolute deadline, then workspace fairness.
- Plan from epoch-final canonical state.
- Merge compatible dirty windows and dependency nodes once.
- Use cost prediction before admission.
- Reserve CPU capacity for state reduction and publication; exact work cannot starve the fast lane.
- Enforce a hard compute budget per fast epoch.
- Publish a lower analysis class before the deadline if the preferred fast kernel cannot finish.

The fast lane may use a dedicated process or CPU set. It should never wait behind the exact solver's full-tile job.

## Exact queue

The exact queue contains at most one running and one latest pending target per workspace.

```text
new canonical revision arrives
  -> replace pending exact target with newest revision
  -> retain operations in durable audit
  -> running job checks supersession at safe stage boundaries
  -> stale completion is discarded
  -> next job evaluates latest canonical snapshot
```

This is state supersession, not edit loss. The latest snapshot contains all accepted operations after deterministic reduction.

### Revision numbering (r1-epochs-review L3)

`workspace_revision` (the `scenarios.scene_version` column) is assigned both by epoch closes and by the legacy direct-edit paths (`/edits`, `/universal-edits`, reset), which interleave on the same counter. Under mixed direct-edit plus realtime use, observed revisions may SKIP values — contiguity of revisions is NOT a contract. What is contractual: monotonicity, and every durably accepted realtime operation landing in exactly one epoch's canonical revision (zero loss).

## Cross-workspace fairness

Use deficit round-robin or weighted fair scheduling. Cost units are derived from dependency class, changed cells, read-window estimate, timesteps, outputs, and cache availability. An instructor/demo workspace may receive a documented weight, but no workspace should permanently starve.

## Deadline budgeting

For a 1,000 ms target, an initial budget can be apportioned as:

| Stage | Budget |
|---|---:|
| Network, authentication, durable append | 100 ms |
| Wait to epoch close | <= 100 ms |
| Reduce and dependency plan | 100 ms |
| Fast computation | 550 ms |
| Encode, publish, broadcast | 150 ms |

These are target budgets, not measured facts. Telemetry must report each stage and the scheduler must adapt thresholds from observed p95/p99.

## Backpressure

Admission uses predicted fast cost and current reserved capacity. If a proposed operation would violate the guaranteed envelope:

- ask the client to reduce brush size/rate;
- temporarily coarsen preview only if it remains within a validated fast class;
- or reject with retry metadata before durable acceptance.

Once accepted, an operation is not dropped to save the deadline.

## Recovery

Epoch assignment, operation sequence, canonical revision, and publication state are durable. On restart:

1. recover incomplete closed epochs;
2. replay their operations deterministically;
3. republish idempotently if needed;
4. compact exact targets to the latest revision;
5. never reuse a partially staged fast/exact payload without checksum and revision verification.
