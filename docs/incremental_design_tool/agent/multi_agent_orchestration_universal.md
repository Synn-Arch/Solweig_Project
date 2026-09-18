# Multi-agent orchestration for universal editing

## Parallel workstreams

The lead agent should run as many independent workstreams as the platform safely supports, normally four to eight:

1. repository and scientific dependency cartography;
2. dependency-graph and planner implementation;
3. building/DEM geometry adapters;
4. vegetation adapter;
5. land-cover/surface adapter;
6. forcing/time/model-parameter adapters;
7. scientific oracle and differential harness;
8. cache/bitpack/window optimization;
9. frontend capability-registry and exact-patch integration;
10. API, persistence, supersession and deployment;
11. independent scientific, concurrency, performance and release red teams.

Not all should write simultaneously. Use isolated branches/worktrees and disjoint ownership. Shared core files have one writer while reviewer and verifier agents operate read-only.

## Required independence

- Adapter author does not approve its dependency map alone.
- Scientific-core author does not run the only differential validation.
- Performance author does not provide the only benchmark.
- Frontend author does not provide the only stale-result/E2E test.
- Release auditor has no authorship in the final implementation wave.

## Task packet

Every subagent receives task/gate IDs, owned files, read-only files, exact goal/non-goals, source nodes, required dependency effects, fixtures, commands, performance evidence, hard stops and handoff schema.

## Handoff

```yaml
task_id:
status: PASS|FAIL|BLOCKED
branch:
commit_sha:
source_nodes_changed: []
dependency_nodes_expected: []
owned_files_changed: []
commands_run: []
tests: {passed: [], failed: []}
scientific_metrics: {}
performance_metrics: {}
assumptions: []
risks: []
next_step:
```

## Integration waves

- Wave A: audits and baseline measurements.
- Wave B: graph/planner, registry, validation harness and independent adapters in parallel.
- Wave C: exact scientific execution integration plus non-author review and differential verification.
- Wave D: API/persistence, frontend, performance and deployment.
- Wave E: fresh adversarial red teams.

After each wave, inspect clean branches, integrate in dependency order, run T0/T1, then T2/T3/T4 as affected, record evidence in WORKLOG, and only then start the next wave.

A stalled subagent does not stop unrelated work. Preserve its branch and handoff, reassign the task, and continue. Two divergent failed attempts should trigger a diagnosis-only agent before another implementation attempt.
