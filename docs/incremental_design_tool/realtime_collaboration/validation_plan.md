# Validation and load plan

## Validation dimensions

The real-time system must pass four independent dimensions:

1. collaboration correctness;
2. deadline behavior under an admitted envelope;
3. scientific exactness or qualified compensation;
4. exact reconciliation and failure recovery.

## Collaboration correctness tests

- deterministic replay of a recorded multi-user operation stream;
- duplicate delivery and retry idempotency;
- out-of-order client arrival resolved by server sequence;
- conflicting object move/update/delete;
- overlapping raster strokes;
- scalar/range update overlap;
- heterogeneous source-family epoch;
- add then delete in one epoch;
- all accepted operation IDs traceable into the published workspace revision;
- no operation assigned to two epochs or none;
- reconnect from an older revision receives a complete deterministic catch-up.

## Deadline tests

Run a matrix over sessions, operations/second, changed cells, source families, and exact-lane load. Measure p50/p95/p99/max for:

- acknowledgement;
- epoch wait;
- reduce/plan;
- fast compute;
- publication/broadcast;
- end-to-end accepted-to-fast latency.

A gate passes only when:

```text
accepted operations omitted = 0
fast revision p99 <= 1,000 ms
stale overwrites = 0
server errors within envelope = 0 or declared availability budget
```

Run at least a one-hour soak and a burst at twice the admitted rate to verify backpressure/degradation.

## Scientific exact tests

For each enabled exact fast path and exact reconciliation path, compare with fresh full-domain oracle results using identical model/cache/input versions. Include each adapter family and cross-family batches. Record maximum error, MAE, RMSE, quantiles, threshold exceedance, boundary ring, untouched-region invariance, and time-state checkpoints.

Geometry fixtures include additions, removals, movement, resize, overlap, building adjacency, low sun, high objects, raster boundary, and repeated edits based on a previous exact revision.

## Qualified fast-model tests

Split calibration and holdout fixtures by scene region and edit combination. Validate:

- error by edit magnitude and density;
- unseen combinations inside declared domain;
- cross-family nonlinear interactions;
- accumulated drift over many epochs;
- out-of-domain detector recall;
- conservative error-band coverage;
- correction magnitude when exact result arrives.

A fast model cannot be enabled merely because average error is low. Tail and spatially localized errors must meet its declared contract.

## Queue and scheduler tests

- exact job running while ten fast epochs publish;
- latest exact target compacts superseded revisions without edit loss;
- fairness across many workspaces;
- instructor weight without starvation;
- exact process crash/restart;
- fast worker crash before/after stage write;
- admission refusal before acceptance;
- accepted operation survives server restart;
- checksum/revision rejection of partial or stale payload.

## Independent verification

Each computation optimization has:

1. author agent;
2. independent scientific differential agent;
3. independent performance reproduction agent.

The release red team receives only the contract and repository, not the author's interpretation of expected results.

## Evidence package

Every benchmark report includes commit SHA, dirty status, machine/OS, host load, CPU allocation, thread environment, cache state, fixture IDs, exact/fast class, revisions, operation rate, changed area, timings, RSS, output hashes, numerical metrics, and failures. Raw samples accompany summaries.
