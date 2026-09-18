# R7 design: un-reserving `fast_qualified` — anchored vegetation compensation v1

Status: implemented (feat `631dcf9`, red tests `4648a59`). Scope: the third
result class goes from reserved (r2b downgraded it unconditionally) to
publishable under a structural qualifier contract, filled by a compensated
fast kernel for THE gap family — vegetation geometry.

## Scout: why vegetation is the gap family

Serving-stack audit (scheduler.py, jobs.py, executor_bridge.py, types.py,
broadcast.py, admission.py, service_level_contract.md, epoch_scheduler.md,
operations.py, reducer.py):

- **View-only epochs** are structurally `fast_exact`: `ViewCacheKernel`
  re-classifies pure view selection without physics.
- **Meteorology epochs** are served by the exact lane's own bounded fast
  paths — the r3a met fast path and G2.1 warm path live in
  `utci_process.py` / `incremental/store.py` / `executor.py` / `jobs.py`.
  They are exact-lane serving, not fast-lane kernels; the fast lane has
  nothing to add for them.
- **Vegetation geometry** has no bounded exact path anywhere: windowed
  exact solves cost seconds, a full-tile solve ~130 s measured on the
  site_500 cache. That is the gap `fast_qualified` exists to fill — as the
  LAST resort (priority discipline: view → exact paths → compensation).

## The qualifier contract (structural, kernel-agnostic)

A kernel may publish `fast_qualified` ONLY when the payload is a complete
anchored qualifier (`realtime/qualification.py` validates; the scheduler
enforces):

```
payload = {
  "qualifier": {
    "model_version": str,          # non-empty
    "form": str,                   # how to read the delta
    "domain": {"families": [...], "variables": [...], ...},
    "error_evidence": {
      "metrics": {var: {...}},     # MEASURED holdout error
      "n_holdout": int >= 1,
      "evidence": str,             # pointer to the calibration run
      "calibration_run_id": str,
    },
    "reconciliation": str,         # how exact catch-up supersedes
  },
  "exact_base_revision": int >= 0, # the ANCHOR
  "base_planes": {...},            # WHAT base was used, never implicit
}
```

ANY defect (bounded vocabulary `QUALIFIER_DEFECTS`, 12 ids) downgrades the
result to `visual_pending` loudly: the defect ids ride the payload
(`downgrade_reasons`), `logger.error` fires, and the typed counter
`solweig_rt_fast_qualified_downgraded_total{reason=...}` increments.
"Never reduce precision silently" is enforced by structure, not convention.
The r2b reserved-class fence narrows to this qualifier-completeness fence:
a bare `fast_qualified` still downgrades, and the r2 test pinning that
(test_realtime_r2_fast_lane.py:797) passes unchanged — no test weakened.

## The supersession fence

A qualified frame must never survive as authoritative after the exact lane
publishes a result covering its revision: when
`exact_result_version >= qualified_publication.workspace_revision`, the next
`run_once` pass re-publishes the frame once as `fast_exact` with
`superseded_qualified: true` (idempotent via publication-history
replacement; no fast-revision churn). Exact-below-revision frames are NOT
superseded — they are still the newest view.

## Model v1: geometric overlay, not fitted

Per edited tree, per sun-up timestep, the predicted vegetation-shadow
region is the swept "capsule" between the crown-disc projection at canopy
top and at canopy bottom — both displaced with the REAL
`geometry.shadow_vector_m` machinery (same 5-degree sun floor and 300 m cap
as the incremental invalidation layer) — optionally unioned with the crown
disc itself. The predicted change set is the XOR of the base-capsule union
and the current-capsule union. Flat-ground projection, no wall interaction,
shadow plane only, disc crown: every limitation rides the payload.

Why not a fitted model: the mission rule — fitted coefficients must come
from offline calibration on REAL exact solves with error measured on
HELD-OUT solves, and the geometric overlay is PREFERRED when a fitted model
cannot clear honest holdout reporting. v1 ships the overlay; nothing is
regressed onto the exact solves. The holdout numbers below are the honest
bound the qualifier declares.

The one a-priori discrete modeling choice — whether the crown disc itself
belongs to the predicted shadow set — was arbitrated by the seeded
calibration split, not by looking at holdout: `cast_corridor_only`
(exclude crown) won on calibration (mean IoU 0.518 vs 0.469) and the choice
generalized on holdout (0.569 vs 0.503). The kernel predicts with exactly
the variant whose holdout error its qualifier declares
(`DEFAULT_QUALIFIER_EVIDENCE["include_crown"] = False`).

## Calibration + holdout harness (real exact solves)

`proofs/r7_veg_compensation_holdout.py` (pattern of `g22_crash_restart.py`
/ `g1_superset_diff.py`): 8 deterministic vegetation edit cases (add /
move / resize, single- and two-tree, interior and near-edge) driven through
the REAL exact path (`run_full_tile` → `compute_utci`, 125–137 s per
solve) on a `/tmp` copy of the site_500 cache (canonical cache READ-ONLY).
Ground truth per timestep is the bitwise diff of the edited run's shadow
planes against the baseline run's. Prediction comes from the shipping
kernel code (`compensated.shadow_delta_masks`) — no reimplementation drift.
Vacuity guard (a case that changes no cells aborts) and drift guard (>5%
of cells flipped aborts — solver nondeterminism or wrong baseline).

Seeded 4/4 calibration/holdout split (seed 20260905), run id
`r7-veg-holdout-site500-20260905`, host COD-MBP16-LOAN2, uptime 1030.2 s
(2026-09-05). Evidence: `proofs/r7_veg_compensation_holdout.json`
(curated copy of `/tmp/r7_proof/holdout.json`; per-timestep confusion
counts stripped for size, aggregates verbatim).

### Holdout numbers (reported on the held-out half ONLY, n=4)

| metric            | p50    | p95    | max    | min    |
|-------------------|--------|--------|--------|--------|
| shadow_iou        | 0.668  | 0.783  | 0.787  | 0.152  |
| shadow_precision  | 0.802  | 0.863  | 0.869  | 0.157  |
| shadow_recall     | 0.856  | 0.893  | 0.893  | 0.699  |

Truth-weighted micro recall 0.870. Per case (holdout):
`add_fixture` IoU 0.758, `add_tall` 0.787, `add_small` 0.578,
`move_200m_east` 0.152.

Honest shape: recall is consistently high — the overlay finds the changed
region. Precision is lower — the overlay over-predicts the change area.
Worst case is the far move: the model predicts flips along BOTH the old
and new cast corridors, but in exact truth cells already shadowed by
buildings do not flip when tree shadow is added on top (the disclosed
no-wall-interaction limitation). Resizes under-predict (the real shadow of
a grown canopy widens more than the disc-capsule at low sun). These bounds
are exactly what the qualifier declares; a consumer that cannot tolerate
them treats `fast_qualified` as `visual_pending` (the class ladder makes
that a local decision).

### Latency (the deadline gate's honesty)

Kernel compute on the heaviest case (2 trees, 13 sun-up timesteps,
500×500): p50 1.80 ms / p95 1.96 ms / p99 2.25 ms (40 reps). Shipped
affine prediction `1.5 ms + 6.0 ms/tree` = 13.5 ms covers 1.25× p99 with
margin; the harness FAILS the run if the prediction ever stops covering
measured p99. `/tmp/r7_proof/latency.json`.

## Wiring

`app._default_fast_lane` builds `VegetationCompensationKernel` only when a
site registry exists AND `DEFAULT_QUALIFIER_EVIDENCE` is non-None; with no
measured evidence the default lane stays `ViewCacheKernel` — compensation
is never wired on unmeasured numbers. Injected `fast_lane_factory` keeps
the 2-arg `(store, hub)` protocol.

## Non-coverage (honest)

- Building / landcover families are NOT compensated (ineligible →
  `visual_pending`); only pure vegetation epochs qualify.
- tmrt/utci deltas are not modelled in v1 (shadow plane only).
- Single-site measurement (site_500, 2009-08-11 forcing): the error bound
  is site-scoped, not cross-site.
- Anchor staleness below-revision (exact published between anchor and
  workspace revision) is allowed by design — the supersession fence covers
  the moment exact catches the qualified revision; drift in between is
  bounded by the declared holdout error, not re-measured per frame.
- Flat-ground / no-wall-interaction capsule geometry (disclosed in every
  payload's `limitations`).
