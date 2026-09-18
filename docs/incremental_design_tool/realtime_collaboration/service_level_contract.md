# Service-level and admission contract

## Why admission is part of the guarantee

No finite CPU system can guarantee a fixed deadline for unbounded users, payloads, changed cells, or operation rates. The one-second target is therefore guaranteed only inside a measured and published **admission envelope**. An operation counts as accepted only after the server has durably recorded it and admitted its workspace/epoch cost. Work outside the envelope is flow-controlled or rejected before acceptance; it is never accepted and then silently dropped.

## Timing definitions

All measurements use a monotonic server clock.

- `t_receive`: final request byte received.
- `t_accept`: idempotency, authorization, schema, revision, and admission checks complete; durable event append committed.
- `t_epoch_close`: first 100 ms epoch boundary after acceptance.
- `t_canonical_publish`: merged workspace revision published to collaborators.
- `t_fast_publish`: deadline-bound analysis revision published.
- `t_exact_publish`: exact SOLWEIG reconciliation published.

The primary latency is `t_fast_publish - t_accept`.

## Required initial SLOs

| SLO | Target inside admitted envelope |
|---|---:|
| Operation acknowledgement | p95 <= 50 ms, p99 <= 150 ms |
| Remote collaborator state visibility | tick-anchored: p50 within [W, 2W); p99 <= 2W + tick-lateness budget (see below) |
| Epoch duration (W) | 100 ms nominal; configurable 50-200 ms |
| Tick-wake lateness (`solweig_rt_epoch_tick_lateness_ms`) | p99 <= 350 ms |
| Canonical workspace revision | tick-anchored: p99 <= 2W + tick-lateness budget from `t_accept`; close pipeline p99 <= 5 ms after epoch close |
| Fast analysis revision | p99 <= 1,000 ms from `t_accept` |
| Accepted-operation inclusion | 100%; zero omitted operations |
| Duplicate idempotency effect | exactly one canonical effect |
| Stale publication | zero accepted stale overwrites |
| Availability during one-hour class | >= 99.9% successful admitted fast updates |

These are commissioning targets until reproduced on the deployment node. Reports must not present them as achieved before load evidence exists.

## Tick-anchored visibility and canonical SLOs (2026-09-06 re-spec)

The original absolute figures (visibility p95 <= 150 ms / p99 <= 300 ms;
canonical p99 <= 250 ms) are unsatisfiable by design at epoch tick W
whenever a single tick wakes late: an accepted operation waits for the
next epoch close (observed as the [W, 2W) aliasing floor in every run),
close-pipeline compute is sub-millisecond, so the tail IS tick-wake
lateness. Four measurement runs — R8 calibration, the one-hour soak,
the R8b re-measure, and the quiet-host re-commission at the lowest host
load achievable on the reference machine (launch 1-min load 2.77; in-run
gauge mean 4.56, min 2.38) — all land visibility p99 in 466-493 ms with
the tail FLAT in host load; the host-load hypothesis is refuted
everywhere it was measurable on that machine.

| Quantity | Re-specified target (W = configured epoch tick) | Measured (4 runs) |
|---|---|---|
| Visibility p50 | within [W, 2W) | 141-155 ms |
| Visibility p99 | <= 2W + 350 ms tick-lateness budget (550 ms at W=100) | 466-493 ms |
| Tick-wake lateness p99 | <= 350 ms (its own SLO, below) | 316-323 ms |
| Canonical close pipeline p99 | <= 5 ms after epoch close | 0.1-0.6 ms |

The tick-lateness budget is itself an SLO: a lateness p99 above 350 ms
means the deployment is out of contract for a host/scheduling reason
(priority inversion, CPU starvation), not an application reason, and
commissioning must record it as such. Lowering W shrinks the [W, 2W)
floor but NOT the lateness term. The absolute 300 ms figure could only
be certified on a host whose scheduler sustains near-zero tick
lateness for a full calibration run; no such window exists on the
shared reference machine. Decision recorded 2026-09-06 (user-approved
after the quiet-host re-commission delivered a confirmed miss).

## Exact-result SLOs are class-specific

A single exact deadline for every edit family is scientifically and computationally misleading. Publish measured exact SLOs by dependency class:

| Dependency class | Initial exact target |
|---|---:|
| View/cached output selection | <= 100 ms |
| Receptor-only or UTCI-only from cached Tmrt | stretch <= 250 ms |
| Selected-hour forcing edit with reusable geometry | stretch <= 1 s |
| Local land-cover edit | stretch <= 2 s |
| Vegetation/building geometry | initially asynchronous; improve from measured baseline |
| DEM or full regeneration | full-only, asynchronous |

The targets are not considered passed until they are benchmarked with the actual site and workload. Exact geometry work begins from the measured approximately 38-second native reference, not from an assumed one-second baseline.

## Admission envelope

Each deployment publishes:

```yaml
max_connected_sessions_per_workspace: measured
max_accepted_operations_per_second: measured
max_operations_per_epoch: measured
max_changed_cells_per_epoch: measured
max_geometry_objects_touched_per_epoch: measured
max_payload_bytes_per_operation: configured
max_active_workspaces: measured
fast_compute_budget_ms_per_epoch: measured
```

The commissioning load matrix should include 1, 4, 16, and 32 sessions and progressively increase operations/second and changed cells/epoch until an SLO fails. The largest configuration with safety headroom becomes the production envelope.

## Degradation ladder

The system never degrades canonical shared-state correctness. It may degrade the analysis class in this order:

1. `fast_exact` from existing cache.
2. `fast_exact` from bounded specialized kernel.
3. `fast_qualified` from a validated response operator.
4. Lower-resolution or reduced-product `fast_qualified` with explicit error/coverage metadata.
5. `visual_pending`: exact shared geometry/state plus the last exact field marked stale.
6. Backpressure or rejection before acceptance if even state/preview SLO cannot be met.

The client displays the class, source revision, exact base revision, uncertainty, and expected reconciliation status.

## Overload behavior

- Do not create one durable scientific job per mouse move.
- Coalesce operations into epoch-final source deltas.
- Keep an append-only operation audit, but schedule computation from the canonical snapshot.
- If the exact lane is behind, it jumps to the latest eligible workspace revision after the current safe cancellation boundary.
- Do not discard accepted edits to reduce the dirty region.
- Do not let exact backlog grow without bound; compact superseded exact targets while retaining operation history.
