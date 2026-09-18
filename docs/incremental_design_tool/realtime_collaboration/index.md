# Collaborative real-time SOLWEIG editing

## Purpose

This specification extends the universal interactive editor into a collaborative, multi-session workspace with a strict one-second fast-update objective. It is not limited to trees. An update epoch may contain heterogeneous edits to any enabled SOLWEIG source family, including building massing, vegetation, land cover, selected date and time, meteorological forcing, audited model or receptor parameters, and view or output selection.

The product requirement is:

> Every operation accepted during one update interval must be present in the shared canonical workspace state and in the next deadline-bound fast analysis revision. No accepted operation may be silently omitted, even when many users and source families edit concurrently.

The exact SOLWEIG result is tracked separately. Current measurements show that a representative geometry edit still requires roughly 38 seconds on the measured native CPU path, with approximately 20 seconds in vegetation SVF replay and 17 seconds in the truncated time loop. Therefore, claiming arbitrary exact geometry recomputation within one second would be unsupported. The architecture uses a deadline-bound fast lane plus an asynchronous exact reconciliation lane, and it makes the result class visible to users.

## Result classes

| Class | Deadline | Meaning |
|---|---:|---|
| `canonical_state` | p99 <= 250 ms | All accepted edits are merged deterministically and broadcast to collaborators. |
| `fast_exact` | p99 <= 1,000 ms | Exact result from cache or a bounded exact kernel. |
| `fast_qualified` | p99 <= 1,000 ms | Compensated result with validated error envelope and provenance. |
| `visual_pending` | p99 <= 1,000 ms | Geometry/state reflects every edit, while the last exact field remains visibly stale; used only outside a qualified fast model's domain. |
| `exact_reconciled` | edit-family SLO | Full scientific result for a specific workspace revision. |

The one-second contract concerns the accepted edit's **visible and analyzable fast revision**, not an unqualified claim that every possible SOLWEIG recomputation is exact in one second. Exactness is explicit and revisioned.

## Core architecture

```text
multi-session operation stream
  -> durable idempotent append
  -> deterministic workspace reducer
  -> 100 ms micro-epoch close
  -> heterogeneous operation coalescing
  -> one dependency plan for the final epoch state
  -> deadline/cost planner
       -> fast exact kernel, or
       -> validated compensated kernel, or
       -> explicit visual_pending fallback
  -> atomic fast revision publication by <= 1 s
  -> exact revision-chasing queue
  -> exact SOLWEIG reconciliation
  -> correction patch and error telemetry
```

The exact queue is not one FIFO job per pointer event. It is a per-workspace latest-state queue. Intermediate revisions may be superseded, but their operations are not lost because the next job evaluates the authoritative snapshot containing all accepted operations.

## Documents

- [Service-level and admission contract](service_level_contract.md)
- [Collaborative state and epoch reducer](collaborative_state.md)
- [Epoch scheduler and queue semantics](epoch_scheduler.md)
- [Computation architecture](compute_architecture.md)
- [Compensation, uncertainty, and exactness](compensation_and_exactness.md)
- [Optimization roadmap](optimization_roadmap.md)
- [Validation and load plan](validation_plan.md)
- [Machine-readable contract](realtime_contract.yaml)
- [ADR 0004](../adr/0004-collaborative-dual-rate-realtime.md)

## Agent entry points

- `docs/incremental_design_tool/agent/goal_condition_realtime_collab.txt`
- `docs/incremental_design_tool/agent/mission_realtime_collab.yaml`
- `docs/incremental_design_tool/agent/master_prompt_realtime_collab.md`
- `docs/incremental_design_tool/agent/multi_agent_realtime_collab.md`

## Non-negotiable scientific rules

- The current full-domain SOLWEIG path remains the exact oracle.
- Fast compensation is never labeled exact.
- Error bounds are qualified by edit family, magnitude, spatial context, selected output, time range, and model/cache revision.
- Read-window physics, the 6-degree sky-patch floor, accumulation ordering, temporal dependencies, units, nodata, and visibility polarity are not altered merely to reach latency.
- If a fast model is outside its validated domain, the system publishes `visual_pending` rather than an invented field.
- An older exact result never overwrites a newer fast or exact workspace revision.
