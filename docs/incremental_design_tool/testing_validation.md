# Testing and scientific validation

## Test strategy

The incremental feature requires four independent forms of evidence:

1. Unit correctness of geometry, packing, and patch application.
2. Scientific equivalence against full-domain SOLWEIG.
3. End-to-end API and browser behavior.
4. Performance and memory measurements on the deployment node.

Passing only unit tests is not sufficient for scientific release.

## Existing scaffold tests

### Python

```bash
pytest -q \
  tests/test_incremental_geometry.py \
  tests/test_incremental_bitmask.py \
  tests/test_incremental_edits.py \
  tests/test_incremental_design_docs.py
```

These tests cover:

- shadow direction and length caps;
- old-plus-new move invalidation;
- block alignment and clipping;
- local/full mode selection;
- transitive window merging;
- 153-patch pack/unpack exactness;
- single-patch extraction and mutation;
- packed weighted sum equivalence;
- edit coalescing semantics;
- required documentation and frontend assets.

### Frontend

```bash
cd studio
node --test tests/*.test.mjs
node --check app.mjs renderer.mjs model.mjs solver_kernel.mjs solver_worker.mjs
```

These tests cover homography round trips, shadow direction, dirty-window inclusion, grid alignment, preview patch shape, outside-window immutability, and delete-to-baseline behavior.

## Scientific equivalence harness

Implement a test helper with two paths:

```python
full = run_full_solver(site, complete_tree_state, forcing)
incremental = baseline.copy()
for patch in run_incremental_solver(site, edit_sequence, forcing):
    apply_patch(incremental, patch)
compare(full, incremental)
```

The full solver must start from the same tree state, model version, forcing, cache generation settings, and output options.

## Required scientific fixtures

| Fixture ID | Setup | Risk exercised |
|---|---|---|
| SCI-001 | Add one 10 m tree in open terrain at noon | Basic local shade |
| SCI-002 | Move a tree 200 m | Old-region cleanup and two windows |
| SCI-003 | Delete a tree | Restoration to baseline |
| SCI-004 | Resize height and crown | Changed SVF radius and rasterization |
| SCI-005 | Two overlapping crowns | Nonlinear combination |
| SCI-006 | Tree next to a building | Building and vegetation visibility interaction |
| SCI-007 | Tree at site boundary | Clipping and halo |
| SCI-008 | Low solar altitude | Long shadow and fallback |
| SCI-009 | Ten trees in one cluster | Stress and saturation |
| SCI-010 | Full-day replay | Time-dependent state |
| SCI-011 | Rapid add-move-delete sequence | Edit coalescing |
| SCI-012 | Dirty area above threshold | Full recompute fallback |

## Numeric comparisons

Record differences only on valid pedestrian cells.

Recommended initial thresholds, subject to baseline numerical behavior:

| Metric | Initial threshold |
|---|---|
| UTCI mean absolute error | ≤ 0.02°C |
| UTCI 99th percentile absolute error | ≤ 0.10°C |
| UTCI maximum absolute error | ≤ 0.25°C, investigated individually |
| Tmrt mean absolute error | ≤ 0.05°C |
| Binary direct-shadow disagreement | ≤ 0.1% of compared cells |
| Outside-write-window change | exactly zero in stored result |

If the local solver shares identical operations and only crops arrays, tighter machine-precision agreement may be possible. Do not loosen thresholds to hide boundary or temporal-state errors.

## Boundary diagnostics

For every fixture, create an error map and summarize error by distance from the write-window edge. A spike near the edge indicates insufficient halo or missing external occluders.

```text
0 to 2 cells from edge
3 to 8 cells
9 to 16 cells
interior
```

Increase the read halo until the edge distribution is indistinguishable from the interior, or document a mathematically sufficient bound.

## Property-based tests

Generate randomized valid trees and assert:

- dirty windows are inside the grid after clipping;
- a move window contains the old and new single-tree windows;
- increasing height does not reduce the conservative SVF radius;
- pack then unpack returns the exact binary cube;
- applying a patch changes no element outside its window;
- add then delete coalesces to no-op before execution;
- scene versions increase monotonically;
- stale results never replace a newer exact result.

Use a fixed seed and save any failing case as a regression fixture.

## Cache conversion validation

When converting existing NPZ visibility arrays to packed files:

1. Validate values are binary.
2. Hash the dense source.
3. Pack and write metadata.
4. Reopen through memory mapping.
5. Unpack every patch in a test conversion.
6. Compare exact equality.
7. Compare weighted SVF reductions.
8. Record source and destination checksums.

## API contract tests

Required cases:

- scenario creation is idempotent;
- edit request increments scene version once;
- repeated idempotency key returns the same response;
- stale `If-Match` returns conflict;
- invalid tree properties return a field-specific error;
- job status transitions are legal;
- superseded job cannot publish current result;
- result manifest shape matches payload byte length;
- binary patch applies correctly to a known target array;
- reset returns to baseline without unnecessary computation;
- scenario isolation prevents cross-user patches.

## Browser end-to-end tests

Use Playwright in a dedicated frontend CI job once the production API mock exists.

Required journeys:

1. Load baseline and verify exact state.
2. Drag a tree from component library onto the site.
3. Verify preview appears immediately.
4. Verify one edit request is emitted on drop.
5. Return a delayed exact patch and verify it applies.
6. Move the tree while the first job runs.
7. Return the older result and verify it is ignored.
8. Return the latest result and verify exact state.
9. Resize a crown with many input events and verify one debounced request.
10. Delete the tree and verify old influence is cleared.
11. Toggle baseline comparison.
12. Simulate worker failure and verify design state remains.

Capture screenshots for exact, preview, computing, and error states.

## Performance benchmark protocol

Each result must record:

```text
commit SHA
site cache version
machine type and CPU model
available RAM
thread environment variables
warm or cold cache
window dimensions and fraction
tree count
time-step count
requested variables
stage timings
peak RSS
output bytes
```

Run at least 20 warmed repetitions for typical fixtures and five for worst cases. Report median, p90, and p95.

## Queue load test

Simulate a class session:

- 20 scenarios;
- one edit per scenario every 120 s on average;
- bursts of five edits within 10 s;
- one exact worker;
- 30-minute run.

Measure:

- queue wait p50 and p95;
- jobs eliminated by coalescing;
- superseded results;
- worker utilization;
- maximum queue length;
- scenario time to latest exact state.

If the queue exceeds the teaching tolerance, reduce exact frequency, prioritize the active instructor scenario, or add a second worker. Do not silently drop committed edits.

## Memory tests

Use a subprocess so peak RSS is attributable and recoverable.

Required assertions:

- idle warm worker fits steady-state budget;
- one typical local job fits peak budget;
- one threshold-sized local job fits peak budget;
- full fallback fits budget or streams safely;
- repeated jobs do not leak more than a small defined amount;
- worker restart recovers after an injected allocation failure.

## CI organization

Recommended jobs:

```text
python-unit
frontend-unit
frontend-static-smoke
scientific-small-fixture
scientific-nightly-full-fixture
performance-manual-or-scheduled
```

Do not run the largest raster fixture on every small documentation change. Keep one compact deterministic scientific fixture in ordinary CI and run the complete supplied tile nightly or before release.

## Release gate

A release candidate must include:

- test report with fixture IDs;
- full-versus-incremental error summary;
- error maps for boundary-sensitive fixtures;
- benchmark table on target CPU node;
- peak memory table;
- current limitations;
- model and cache version hashes;
- signed-off acceptance criteria from [Agent execution plan](agent_execution_plan.md).
