# Rigorous validation pipeline

## Principle

Incremental computation is accepted only as an optimization of the existing full-domain model. The burden of proof is therefore differential: for every supported edit class and difficult geometric case, the incremental path must reproduce the oracle inside the allowed numerical tolerance and must not alter cells it does not own.

## Validation tiers

### T0: static and structural

Run on every relevant change:

```bash
python -m compileall solweig_gpu tests
python -m json.tool studio/assets/baseline.json >/dev/null
node --check studio/app.mjs
node --check studio/model.mjs
node --check studio/renderer.mjs
node --check studio/solver_kernel.mjs
node --check studio/solver_worker.mjs
pytest -q tests/test_incremental_design_docs.py
```

For YAML mission files, parse them with the project's available YAML parser in CI. If no YAML package is guaranteed, keep a stdlib-compatible structural test or validate in the documentation environment.

### T1: unit

Required classes:

- coordinate transforms and half-open windows;
- solar-shadow vector direction and clipping;
- dirty-region union for old/new tree states;
- block alignment;
- local/full fallback predicate;
- bit-pack/unpack and bit polarity;
- packed weighted reductions;
- edit coalescing and ordering;
- tree preset validation;
- local rasterization primitives;
- scene revision and stale-response checks.

Commands currently available:

```bash
pytest -q tests/test_incremental_geometry.py \
          tests/test_incremental_bitmask.py \
          tests/test_incremental_edits.py
node --test studio/tests/*.test.mjs
```

### T2: integration

Add and maintain tests for:

1. cache builder -> manifest -> warm memory-map loader;
2. original dense visibility -> packed cache -> identical selected planes;
3. edit -> dirty window -> local tree raster -> local worker -> result patch;
4. add/move/delete restoration semantics;
5. rapid edit supersession;
6. restart-safe job state once persistence is implemented;
7. patch encoding/decoding and frontend texture update.

Use temporary directories. Do not depend on a developer's absolute data paths.

### T3: scientific differential

#### Required deterministic fixtures

At minimum, include:

| ID | Edit | Geometry | Reason |
|---|---|---|---|
| S01 | add | isolated medium tree, interior | nominal case |
| S02 | move | old and new influence regions disjoint | validates union invalidation |
| S03 | resize | canopy and height increase | validates expanded influence |
| S04 | delete | isolated editable tree | validates restoration |
| S05 | add | near raster boundary | clipping and halo |
| S06 | add | near building | building/vegetation interaction |
| S07 | add | overlapping existing vegetation | nonlinearity |
| S08 | add two | overlapping shadows | combination rule |
| S09 | move | across block boundary | cache/window alignment |
| S10 | rapid edits | add -> move -> resize | coalescing and stale jobs |
| S11 | tall tree | low solar altitude | large influence/worst shadow |
| S12 | many trees | dirty fraction near threshold | local/full fallback boundary |

For every fixture, generate two branches of computation from the same baseline:

```text
A: full-domain oracle with final edited vegetation state
B: baseline + incremental edit pipeline
```

Compare the same output timestep(s) and state initialization.

#### Required metrics

For each continuous output such as Tmrt and UTCI, record:

- number of compared valid pixels;
- max absolute error;
- mean absolute error;
- RMSE;
- p50, p95, p99 absolute error;
- count and fraction above tolerance;
- metrics inside write window;
- metrics on a configurable boundary ring;
- metrics outside write window.

For categorical/binary shadow or visibility outputs:

- mismatch count;
- mismatch fraction;
- mismatches on boundary ring;
- mismatches outside write window.

#### Tolerance policy

Do not encode a permissive tolerance simply to make the suite green. Establish tolerance in this order:

1. Run full-domain oracle twice to quantify deterministic repeatability.
2. Run the same algorithm through full-window and window-adapter paths to measure refactoring noise.
3. Use the smallest tolerance above demonstrated numerical noise, with explicit rationale in `testing_validation.md`.
4. Preserve stricter exact equality for integer masks, indices, patch layout, and untouched output regions whenever feasible.

Any tolerance change requires a documented before/after result and reviewer-visible justification.

#### Temporal correctness

Because SOLWEIG propagates state through hourly timesteps, validation must not compare a locally recomputed target hour initialized from inconsistent state. For an edit assumed to exist throughout the modeled day:

- start local temporal replay from a documented safe predecessor state;
- replay each required timestep sequentially within the read window;
- compare target-hour and optionally intermediate states against full-domain oracle;
- include at least one fixture where earlier shade changes later ground-temperature-related state.

### T4: performance

Record both cold and warm paths when meaningful.

Required fields per run:

```text
commit_sha
fixture_id
machine_cpu
logical_cpu_count
memory_total
python_version
torch_version
numpy_version
gdal_version
thread_env
cache_state: cold|warm
edit_type
dirty_pixels
dirty_fraction
read_window_pixels
write_window_pixels
fallback: true|false
wall_seconds
cpu_seconds
peak_rss_mb
cache_load_seconds
compute_seconds
serialize_seconds
result_bytes
```

Minimum benchmark scenarios:

- no-op/stale edit;
- nominal one-tree edit;
- move with two separated influence regions;
- low-sun large ROI;
- threshold-near local case;
- forced full-tile fallback;
- full-domain CPU oracle.

Run enough repetitions to report median and p95 or the raw sample set. Avoid claiming p95 from fewer than a meaningful number of repeated runs.

### T5: full repository

Final release validation must execute the repository's supported test matrix or the closest reproducible local equivalent. Existing CI uses conda with GDAL/PyTorch and tests Python 3.10, 3.11, and 3.12 on Linux and macOS. At minimum, the implementation branch must pass the primary Linux environment before merge.

Also build Sphinx documentation with warnings treated seriously. Resolve broken internal links and malformed code blocks introduced by the incremental documentation.

### T6: deployment and interaction smoke

A human or automated browser should exercise:

1. load baseline scene;
2. drag tree preview without exact server request on every pointer move;
3. drop tree and receive a job ID;
4. observe progress;
5. perform a second edit before first result returns;
6. verify stale first result is discarded;
7. receive exact second result;
8. verify only returned patch changes;
9. delete tree and verify restoration;
10. refresh/restart once persistence exists and verify scenario consistency.

The UI must expose whether data shown are preview or exact.

## Gate evidence format

Every mandatory gate in the worklog should contain evidence in this shape:

```text
Gate: SCI-001
Status: PASS
Commit: <sha>
Command: pytest ...
Fixture(s): S01-S04
Result: 4 passed; UTCI max abs ...; Tmrt max abs ...
Artifacts: docs/incremental_design_tool/validation/<report>
Notes: ...
```

`PASS` without a command/result is not valid evidence.

## Regression policy

Every scientific bug found after a phase is completed must gain a fixture or test that fails before the fix and passes afterward. Do not rely on the original report alone.
