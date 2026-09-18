# AI-agent execution plan

## Purpose

This page is the operational handoff for an implementation agent. It defines the required read order, file boundaries, task sequence, tests, and stop conditions. The agent should update this page as implementation status changes.

## Nonnegotiable rules

1. Preserve the existing full-domain solver as the scientific oracle.
2. Do not introduce an approximate algorithm into the exact worker without a separate mode, validation report, and ADR.
3. Do not mutate shared baseline caches.
4. Do not publish a result for a stale scene version.
5. Do not optimize before recording a benchmark for the same fixture.
6. Do not claim wind-field response to user-added trees in the first release.
7. Keep all raster windows half-open.
8. Add tests before changing visibility polarity or patch-axis layout.
9. Run the full existing repository test suite before each merge.
10. Use small, reviewable commits aligned with the phases below.

## Read first

Read, in order:

```text
docs/incremental_design_tool/architecture.md
docs/incremental_design_tool/data_model.md
docs/incremental_design_tool/incremental_algorithm.md
docs/incremental_design_tool/cpu_optimization.md
docs/incremental_design_tool/testing_validation.md
```

Then inspect:

```text
solweig_gpu/shadow.py
solweig_gpu/solweig.py
solweig_gpu/utci_process.py
solweig_gpu/incremental/
studio/
```

## Phase 0: establish reproducible baseline

### Tasks

- Add a compact, redistributable scientific fixture or fixture-generation script.
- Record full-domain CPU runtime, peak RSS, output hashes, array shapes, and cache load time.
- Record the exact commands and environment.
- Add a benchmark result template under `docs/incremental_design_tool/benchmarks/`.
- Verify existing tests pass before scientific refactoring.

### Exit criteria

- `BASE-001`: Full-domain reference output is reproducible.
- `BASE-002`: Fixture includes add, move, resize, and delete tree states.
- `BASE-003`: Baseline hashes and model revision are recorded.

### Suggested commit

```text
test(incremental): add reproducible full-domain oracle fixture
```

## Phase 1: input and output window interfaces

### Files likely to change

```text
solweig_gpu/utci_process.py
solweig_gpu/solweig.py
solweig_gpu/shadow.py
solweig_gpu/incremental/geometry.py
new solweig_gpu/incremental/io.py
```

### Tasks

- Introduce a typed `RasterWindow` parameter at the orchestration boundary.
- Allow preloaded arrays and metadata instead of file paths.
- Add read and write window separation.
- Add output-variable selection.
- Replace output lists with preallocated or streaming writers.
- Preserve the full-window behavior as the default.

### Tests

- Full window returns the same arrays as the previous code.
- A crop result matches slicing the full result for a fixture without external influence.
- Unrequested outputs are not allocated or written.
- Output patches use correct half-open shape.

### Exit criteria

- `WIN-001`: Existing public API remains functional.
- `WIN-002`: Full-window numerical output is unchanged within tolerance.
- `WIN-003`: Peak memory decreases when only UTCI is requested.

### Suggested commit

```text
refactor(incremental): add windowed solver inputs and streamed outputs
```

## Phase 2: baseline cache object

### New modules

```text
solweig_gpu/incremental/cache.py
solweig_gpu/incremental/manifest.py
solweig_gpu/incremental/cache_builder.py
```

### Tasks

- Define and validate site manifest schema.
- Convert fixed arrays to `.npy` memory maps.
- Separate building-only and vegetation visibility semantics.
- Add cache checksum validation.
- Load cache once at worker startup.
- Add a CLI command to build or validate the incremental cache.

### Tests

- Manifest mismatch produces an explicit error.
- Window slices match original arrays.
- Read-only arrays cannot be modified through the cache API.
- Cache builder is deterministic for the fixture.

### Exit criteria

- `CACHE-001`: Warm worker performs no GDAL raster reads per ordinary job.
- `CACHE-002`: Cache self-test runs at startup.
- `CACHE-003`: Cache layout and model version appear in result metadata.

### Suggested commit

```text
feat(incremental): add validated memory-mapped site cache
```

## Phase 3: visibility bit packing

### Existing scaffold

```text
solweig_gpu/incremental/bitmask.py
```

### Tasks

- Add a conversion command for current dense shadow matrices.
- Name and test the polarity of each source array.
- Replace dense serving caches with packed arrays.
- Add window and patch-block iterators.
- Implement weighted reductions without full unpacking.
- Remove full-size `diffsh` allocation.

### Tests

- Exact dense-versus-packed round trip.
- Dense-versus-packed SVF and directional-SVF reduction.
- Randomized patch mutation tests.
- Peak memory benchmark for the supplied 500 × 500 × 153 fixture.

### Exit criteria

- `PACK-001`: Packed cache uses approximately 20 bytes per pixel for 153 patches.
- `PACK-002`: Weighted outputs match dense reference.
- `PACK-003`: No exact worker stage materializes all patch planes as Float32.

### Suggested commit

```text
perf(incremental): stream bit-packed visibility patches
```

## Phase 4: tree rasterization and spatial query

### New modules

```text
solweig_gpu/incremental/trees.py
solweig_gpu/incremental/spatial_index.py
```

### Tasks

- Define server-side tree preset validation.
- Implement deterministic window-local crown and trunk rasterization.
- Ensure local coordinates translate exactly to full raster coordinates.
- Query all trees affecting the read window.
- Support add, move, update, and delete.

### Tests

- Local rasterization equals full rasterization sliced to the same window.
- Overlapping trees follow the full-pipeline combination rule.
- Boundary clipping is exact.
- Random translation tests preserve raster results.

### Exit criteria

- `TREE-001`: Every editable tree property is represented in rasterization.
- `TREE-002`: Base vegetation remains immutable.
- `TREE-003`: Move and delete restore old regions correctly.

### Suggested commit

```text
feat(incremental): rasterize editable trees in local windows
```

## Phase 5: exact local scientific worker

### New modules

```text
solweig_gpu/incremental/worker.py
solweig_gpu/incremental/solver.py
solweig_gpu/incremental/result.py
```

### Tasks

- Connect edit coalescing and dirty-window computation.
- Build read and write windows with halo.
- Update local vegetation visibility and SVF.
- Replay temporal state.
- Run exact Tmrt and UTCI for requested variables.
- Stage and publish versioned patches.
- Add cooperative supersession checks between stages.

### Tests

Run all scientific fixtures in [Testing and validation](testing_validation.md).

### Exit criteria

- `SCI-001`: Add, move, update, and delete match full recomputation.
- `SCI-002`: Outside-window stored results remain unchanged.
- `SCI-003`: Boundary error thresholds pass.
- `SCI-004`: Fallback activates for unsafe local jobs.

### Suggested commit

```text
feat(incremental): add exact CPU window worker
```

## Phase 6: API and persistence

### Suggested package

```text
solweig_gpu/server/
├── app.py
├── models.py
├── store.py
├── jobs.py
├── routes_scenarios.py
├── routes_jobs.py
└── patch_codec.py
```

A separate example package is also acceptable if the core project should not depend on a web framework by default. Use an optional dependency group.

### Tasks

- Implement API contract and OpenAPI schema.
- Add SQLite migrations.
- Add durable events and job states.
- Add idempotency and optimistic version checks.
- Add binary patch codec.
- Add health and metrics endpoints.

### Tests

- API contract tests from the testing document.
- Process restart and job recovery.
- Stale publication race.
- Scenario isolation.

### Exit criteria

- `API-001`: All mutation endpoints are idempotent.
- `API-002`: Scene conflicts are explicit.
- `API-003`: Binary patch metadata and payload validate.
- `API-004`: Restart recovery is tested.

### Suggested commit

```text
feat(server): expose versioned incremental analysis API
```

## Phase 7: connect production frontend

### Tasks

- Replace browser preview worker with API adapter.
- Preserve preview rendering and local design state.
- Implement status polling or server events.
- Decode and apply binary patches with `texSubImage2D`.
- Implement stale-result rejection.
- Add exactness and error states.
- Add keyboard placement and ARIA live status.

### Tests

Run all browser journeys in the testing document.

### Exit criteria

- `UI-001`: One server edit is sent per committed interaction.
- `UI-002`: Slider changes are debounced.
- `UI-003`: Older exact results are ignored.
- `UI-004`: Failure preserves design state.
- `UI-005`: Model scope is visible.

### Suggested commit

```text
feat(demo): connect design workspace to exact CPU worker
```

## Phase 8: profile and optimize measured hotspots

### Tasks

- Run fixture matrix on target node.
- Record stage timings and RSS.
- Select the top two bottlenecks.
- Port only those kernels to Numba or compiled code.
- Re-run scientific equivalence after each change.
- Tune thread counts and full-mode threshold.

### Exit criteria

- Performance and memory budgets in `cpu_optimization.md` pass.
- No correctness threshold regresses.
- Benchmark report includes cold and warm cases.

### Suggested commits

```text
perf(incremental): compile local shadow kernel with numba
perf(incremental): reuse worker scratch buffers
```

## Phase 9: deployment hardening

### Tasks

- Add process supervision and resource limits.
- Add cache startup validation and readiness.
- Add structured logs and metrics.
- Add TTL cleanup.
- Run classroom queue load test.
- Write demo-day runbook.

### Exit criteria

- Worker recovers from crash.
- Interrupted jobs recover.
- Load test meets queue tolerance.
- Public deployment has HTTPS and request limits.

## Commands to run before every merge

```bash
pytest --cov=solweig_gpu --cov-report=term-missing

cd studio
node --test tests/*.test.mjs
node --check app.mjs renderer.mjs model.mjs solver_kernel.mjs solver_worker.mjs

# When documentation dependencies are available
sphinx-build -W -b html docs docs/_build/html
```

When scientific files change, also run the full-versus-incremental fixture harness and attach its report.

## Agent progress template

Update this block in a working issue or PR:

```text
Current phase:
Current commit:
Fixture and cache version:
Files changed:
Tests run:
Scientific comparison:
Performance before:
Performance after:
Peak RSS before:
Peak RSS after:
Known limitations:
Next exact step:
```

## Stop conditions

Stop and request human review when:

- visibility polarity is unclear;
- the full reference itself is nondeterministic beyond tolerance;
- local boundary errors persist after a conservative halo;
- wind-field recomputation becomes a requirement;
- a proposed optimization changes scientific equations;
- cache or result licensing prevents repository fixtures;
- direct commits to protected `main` are disallowed;
- target latency requires an approximation not covered by current ADRs.

## Autonomous operation companion documents

For uninterrupted agentic implementation, also read and follow:

- [Master agent prompt](agent/master_prompt.md)
- [Machine-readable mission and gates](agent/mission.yaml)
- [Autonomous runbook](agent/runbook.md)
- [Rigorous validation pipeline](agent/validation_pipeline.md)
- [Recovery and continuity](agent/recovery_and_continuity.md)
- [End-to-end pipeline](agent/pipeline.md)
- [Worklog template](agent/worklog_template.md)
- [Handoff template](agent/handoff_template.md)

The master prompt defines hard stop conditions. Routine implementation failures should be resolved inside the autonomous loop rather than escalated.
