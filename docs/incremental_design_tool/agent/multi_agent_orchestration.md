# Multi-agent orchestration protocol

The lead agent should maximize **useful** parallelism while preserving scientific traceability and avoiding concurrent edits to the same code.

## Lead responsibilities

The lead is scheduler, integrator, and final verifier. It owns `mission.yaml`, `WORKLOG.md`, the subagent ledger, file ownership, merge order, cross-component integration, final T5/T6 validation, and publish/fallback decisions. Delegate independently testable work rather than implementing everything serially.

## Concurrency policy

Use the maximum subagent concurrency the platform permits when tasks do not share writable files or mutable fixtures. A normal wave should use 4-8 subagents if available; read-only investigation may use more.

Isolation order:
1. platform-native isolated subagent workspace;
2. one Git worktree/branch per write agent;
3. explicit disjoint file ownership.

Suggested branch: `agent/<phase>/<task-id>-<role>`.

Never permit two write agents to modify the same owned file concurrently. `shadow.py`, `solweig.py`, and `utci_process.py` have one active writer at a time. Other agents may review them read-only. Subagents never force-push, rewrite shared history, merge to main, or bypass protection.

## Wave A: reconnaissance, parallel

Launch immediately:
- **repo-cartographer**: call graph, shapes, mutable state, I/O, tests, CI, ownership risks.
- **oracle-auditor**: full-domain semantics, temporal dependencies, shadow/SVF conventions, deterministic fixture needs.
- **baseline-benchmarker**: reproducible runtime/RSS/cache/output baseline and machine metadata.
- **frontend-auditor**: interaction, scene revision/job lifecycle, rendering and patch interfaces.
- **validation-engineer**: T0-T6 matrix and test gaps.

These are read-only unless given separate test/doc ownership.

## Wave B: foundation, parallel disjoint writers

- **window/invalidation implementer**: ROI, read/write windows, halo, old/new union, fallback predicates.
- **cache/bitpack implementer**: immutable cache, packed visibility, mmap, round-trip tests.
- **tree-raster implementer**: deterministic dynamic vegetation rasterization and edit coalescing.
- **validation-harness implementer**: oracle fixtures, metrics, boundary ring and outside-window checks.
- **frontend implementer**: exact-job interface, preview/exact distinction, stale rejection, texture patches.

## Wave C: scientific-core integration

- **local-solver implementer**: exact ROI solver and temporal replay.
- **scientific reviewer**: independent read-only audit of equations, array semantics, halo safety, oracle comparison.
- **numerical verifier**: independently execute T3 and investigate discrepancies.
- **performance profiler**: profile only after correctness is green.

Reviewer/verifier must not author the implementation they validate.

## Wave D: service and operations

Parallelize:
- **API/persistence implementer**
- **CPU optimizer**, only profile-backed changes
- **deployment/CI implementer**
- **frontend E2E verifier**

## Wave E: independent red team

Before release use fresh non-author agents:
- **scientific red-team**: low sun, tall trees, overlaps, building adjacency, boundaries, temporal state.
- **state/concurrency red-team**: rapid edits, cancellation, stale jobs, restart, patch ordering.
- **performance red-team**: reproduce latency/RSS on recorded target CPU.
- **release auditor**: APIs, docs, limitations, generated files, licenses, branch/reproducibility.

Red-team failure reopens the owning phase. Never waive by increasing tolerance without scientific justification.

## Task packet for every subagent

Provide:
- task ID and phase/gates;
- exact goal and non-goals;
- owned writable and read-only files;
- authoritative docs;
- fixtures and baseline commands;
- required tests/metrics;
- dependencies;
- hard stops;
- required handoff format.

A subagent must not edit unowned files without explicit ownership transfer.

## Handoff contract

Every subagent returns:

```yaml
task_id:
status: PASS|FAIL|BLOCKED
branch:
commit_sha:
owned_files_changed: []
commands_run: []
tests:
  passed:
  failed:
metrics: {}
scientific_assumptions: []
risks: []
known_failures: []
recommended_merge_order: []
next_step:
```

Read-only tasks may use `commit_sha: null`. Record all handoffs in `WORKLOG.md`.

## Integration protocol

After each wave:
1. require clean subagent branches;
2. inspect handoffs/diffs;
3. reject unrelated/generated changes;
4. integrate in dependency order;
5. run T0/T1;
6. run T2 for interface/persistence changes;
7. run T3 after scientific/window/cache/raster/temporal changes;
8. resolve conflicts only on integration branch;
9. use a fresh reviewer for non-trivial scientific conflict resolution;
10. checkpoint WORKLOG before next wave.

## Failure and reassignment

A failed subagent must not block independent work. Preserve its branch/commit/log, summarize failure, and reassign implementation-local failures to a fresh subagent. If two independent attempts fail differently, spawn a diagnosis-only agent before a third implementation attempt. Hard stops block only dependent paths while independent work finishes.

## Continuity

At least hourly and after every merge wave record active agents, branches/worktrees, file ownership, last green tier, pending cherry-picks, failures, benchmark environment, and exact next command. On context reset, reconstruct from WORKLOG, branches/worktrees, and handoffs. Do not recreate the plan from memory.

## Minimum independent verification

- Scientific core: author + scientific reviewer + independent T3 runner.
- Performance optimization: author + independent benchmark reproduction.
- API concurrency: author + independent rapid-edit/stale-state tester.
- Frontend exact/preview behavior: author + independent E2E verifier.
- Release: fresh release auditor after T5/T6.

Use more subagents whenever work is genuinely independent.
