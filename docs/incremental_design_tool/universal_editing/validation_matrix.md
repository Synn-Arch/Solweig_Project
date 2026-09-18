# Universal editing validation matrix

## Adapter qualification rule

An edit type is not “supported” merely because the UI can modify it. It becomes supported only when its adapter passes source-delta, dependency, spatial/temporal invalidation, full-oracle, state/version, and performance validation.

## Required tests for every scientific adapter

1. Schema, units, finite values, ranges, coordinate and nodata validation.
2. Add/update/delete or equivalent operation semantics.
3. Deterministic source-delta application.
4. Old-state cleanup and new-state application.
5. Expected invalidated dependency nodes.
6. Expected reusable upstream nodes.
7. Local/full scope decision and fallback reasons.
8. Temporal replay start and stop.
9. Incremental versus fresh full-domain oracle.
10. Boundary-ring comparison for local writes.
11. Outside-write-window invariance.
12. Rapid edit coalescing and stale result rejection.
13. Restart/replay from durable edit events.
14. CPU runtime, peak RSS, cache reads, and result size.
15. Preview/exact UI distinction.

## Edit-family matrix

| Family | Nominal spatial scope | Temporal scope | Oracle fixtures |
|---|---|---|---|
| Vegetation geometry | local directional or full fallback | replay where stateful | isolated, overlap, low sun, boundary, delete |
| Building geometry | local/full, initially conservative | replay | height, footprint, near vegetation, long shadow, wall/aspect |
| DEM | full initially | replay | grading, relative-height effects, boundary |
| Land cover | local | replay for thermal state | class replacements, mixed mask, reset |
| Meteorology | full downstream | changed and dependent times | Ta, RH, radiation, wind speed/direction |
| Time/date | full dynamic stages | selected/replay | morning/noon/low sun/date change |
| Model/receptor parameter | full downstream stage-specific | affected times | lower/upper bound and baseline restore |
| View-only | none | none | verify zero solver jobs |

## Dependency tests

For each adapter, assert both positive and negative dependencies. Example:

- land-cover edit invalidates surface/radiation/comfort but does not invalidate wall/aspect;
- meteorology edit invalidates dynamic outputs but does not rebuild SVF;
- building edit invalidates walls/aspect and visibility;
- view-only layer switch produces no scientific job;
- geometry edit does not claim a new wind field unless the wind-extension adapter is explicitly executed.

## Cross-family scenarios

Required combined fixtures:

1. building mass + adjacent tree;
2. tree + land-cover paint below canopy;
3. geometry edit followed by time change before exact result;
4. forcing change while a local geometry job runs;
5. building delete plus surface reset;
6. multiple local windows from different adapter families;
7. local source edit combined with global forcing edit;
8. rollback to baseline across all changed source overlays.

The expected plan may contain local upstream stages and global downstream stages. Verify execution order and stale publication behavior.

## Scientific metrics

For UTCI, Tmrt, and continuous intermediate outputs record max absolute error, MAE, RMSE, p50/p95/p99 absolute error, count/fraction above tolerance, boundary-ring metrics, and outside-window difference. For masks/visibility/shadow record mismatch count and fraction.

Tolerance is established from deterministic repeatability and full-window adapter noise. It is not tuned per edit type to hide an invalid dependency or insufficient halo.

## Performance evidence

Record per adapter and combined scenario:

- cold/warm cache;
- changed source and operation;
- invalidated stages;
- read/write pixel counts;
- local/full decisions and reasons;
- temporal steps replayed;
- wall, CPU, and stage times;
- peak RSS;
- bytes read/written and patch size;
- target machine and thread configuration.

A global forcing edit may be accepted as interactive even if it touches every cell, provided upstream cache reuse meets the latency target.
