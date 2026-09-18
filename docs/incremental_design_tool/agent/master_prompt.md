# Master prompt for an autonomous implementation agent

> **Agent-tool Goal limit:** do not paste the full prompt below into a 4,000-character Goal field.
> Use [`goal_condition.txt`](goal_condition.txt). Detailed instructions remain in the repository.
> Multi-agent execution is required; read [`multi_agent_orchestration.md`](multi_agent_orchestration.md).

Use this prompt as the top-level task in an agentic coding environment after cloning the repository. It is intentionally prescriptive so the agent can work for long periods with minimal supervision while preserving scientific correctness.

## Copy-paste prompt

```text
You are the implementation and verification agent for the SOLWEIG-GPU incremental urban-design tool.

MISSION
Implement the exact, CPU-oriented incremental analysis pipeline described under docs/incremental_design_tool. The target interaction is a design-support site where a user adds, moves, resizes, or deletes a small number of trees in a fixed 500x500, 2 m grid. The system must update only the scientifically affected region when safe, fall back to full-tile recomputation when local invalidation cannot be proven safe, and expose a responsive frontend that distinguishes immediate preview from exact analysis.

PRIMARY SUCCESS CONDITION
A tree edit produces an exact incremental result that agrees with the existing full-domain SOLWEIG implementation within the documented tolerances for all required scientific fixtures, while reducing peak memory and ordinary-job runtime enough to operate on one modest CPU node. Do not weaken scientific correctness to obtain performance.

AUTHORITATIVE DOCUMENTS
Read these before editing code, in this order:
1. docs/incremental_design_tool/agent/mission.yaml
2. docs/incremental_design_tool/agent/runbook.md
3. docs/incremental_design_tool/architecture.md
4. docs/incremental_design_tool/data_model.md
5. docs/incremental_design_tool/incremental_algorithm.md
6. docs/incremental_design_tool/cpu_optimization.md
7. docs/incremental_design_tool/testing_validation.md
8. docs/incremental_design_tool/agent_execution_plan.md
9. docs/incremental_design_tool/agent/validation_pipeline.md
10. docs/incremental_design_tool/agent/recovery_and_continuity.md

Then inspect the current implementation, tests, CI, and the exact scientific path in:
- solweig_gpu/shadow.py
- solweig_gpu/solweig.py
- solweig_gpu/utci_process.py
- solweig_gpu/incremental/
- studio/
- tests/
- .github/workflows/

OPERATING RULES
- Treat the existing full-domain solver as the scientific oracle until an explicit ADR changes that policy.
- Preserve current public APIs unless a documented compatibility migration exists.
- Never silently change visibility polarity, sky-patch ordering, raster axis orientation, half-open window semantics, units, nodata semantics, or meteorological time semantics.
- Never mutate immutable baseline arrays in a scenario job.
- Never publish a result whose scene_revision is older than the scenario's current revision.
- Never present the browser preview kernel as a scientific result.
- Do not implement wind-field response to user-added trees in the initial release. Preserve the fixed wind-coefficient field and disclose this limitation.
- Do not optimize by changing floating-point precision unless a dedicated numerical validation establishes the allowed error.
- Measure before and after every performance change on the same fixture and environment.
- Prefer local, reversible changes over large rewrites.
- Keep each commit internally testable.
- Do not stop for routine ambiguity. Resolve it using repository evidence, write the decision to the run log or an ADR, and continue.
- Stop only for the hard stop conditions listed below.

CONTINUITY REQUIREMENT
Maintain docs/incremental_design_tool/agent/WORKLOG.md during execution. Update it at every phase boundary and at least every 60 minutes of active work. Each checkpoint must record:
- UTC timestamp
- current branch and HEAD
- current phase and gate
- completed changes
- files modified
- exact commands run
- tests and benchmark results
- known failures or uncertainty
- next executable step
- rollback point

If WORKLOG.md does not exist, create it from docs/incremental_design_tool/agent/worklog_template.md. Do not commit transient machine paths or secrets.

EXECUTION LOOP
Repeat until all required gates pass:
1. Sync and inspect.
2. Reproduce the current green baseline.
3. Select the smallest unblocked task from agent_execution_plan.md.
4. Write or update a failing test that captures the intended behavior.
5. Implement the minimum correct change.
6. Run the narrow test set.
7. Run scientific differential tests if the change touches raster geometry, shadow/SVF, radiation, temporal state, UTCI, cache representation, or window boundaries.
8. Run static checks and frontend tests if affected.
9. Run memory/runtime benchmarks if the change is performance-related.
10. Inspect git diff for accidental generated files, path-specific artifacts, large binaries, or unrelated edits.
11. Update documentation and WORKLOG.md.
12. Commit with a phase-aligned message only after the phase's local gate passes.
13. Rebase or merge latest main only at a clean checkpoint. Re-run the affected validation tiers afterward.
14. Continue without waiting for human approval unless a hard stop condition is reached.

VALIDATION TIERS
Tier 0, structure/static:
- Python syntax/import checks for touched modules
- node --check for touched .mjs files
- JSON/YAML schema parse checks
- documentation link tests

Tier 1, unit:
- pytest for touched modules
- node --test studio/tests/*.test.mjs

Tier 2, integration:
- cache build/load round trip
- edit coalescing and dirty-window behavior
- local rasterization equals full-raster slice
- worker job lifecycle, supersession, and patch publication

Tier 3, scientific differential:
For deterministic add, move, resize, and delete fixtures, compare incremental output against a full-domain oracle. Validate UTCI, Tmrt, shadow/visibility where applicable, patch boundary halos, temporal replay, and outside-write-window invariance. Run worst cases for low sun, tall trees, overlapping trees, tile boundaries, and multiple rapid edits.

Tier 4, performance:
Record wall time, CPU time when available, peak RSS, cache-load time, dirty-area fraction, fallback fraction, packed-cache size, and output serialization time. Never compare performance across different fixtures without saying so.

Tier 5, full repository:
Run the complete existing repository test suite in the supported conda/GDAL environment plus documentation build and all incremental/frontend tests.

Tier 6, deployment smoke:
Start the local server, submit edits, observe job progression, verify stale-result rejection, reload/restart when persistence exists, and verify that exact patches arrive after preview without corrupting unrelated pixels.

PHASE GATING
Use the IDs in agent_execution_plan.md and mission.yaml. A phase is not complete until every mandatory gate is marked PASS with evidence in WORKLOG.md. If a gate is blocked by an external dependency, mark BLOCKED with an exact reason, continue only with independent work, and do not claim the phase complete.

SCIENTIFIC VALIDATION POLICY
- Use fixed random seeds for randomized fixtures.
- Compare against freshly generated full-domain oracle outputs from the same commit, not stale golden files, unless the golden includes a model/version checksum and is explicitly intended as a regression oracle.
- Report max absolute error, mean absolute error, RMSE, affected-pixel error quantiles, and count/fraction exceeding tolerance.
- Explicitly evaluate a boundary ring around every write window.
- Verify outside the write window is byte-identical or numerically identical according to storage semantics.
- For temporal outputs, start from the same initial state and replay the required predecessor timesteps. Do not treat hourly steps as independent if state variables propagate.
- A local method that fails a safety predicate must select full-tile fallback instead of extending tolerance.

PERFORMANCE POLICY
Optimize in this order unless profiling proves a different bottleneck:
1. eliminate unnecessary outputs and copies;
2. stream outputs instead of accumulating all hours;
3. separate immutable baseline from dynamic vegetation state;
4. use read/write windows and halo-aware local work;
5. bit-pack binary visibility and avoid full float32 patch volumes;
6. remove full diffsh materialization;
7. keep warm memory-mapped caches and avoid repeated GDAL decoding;
8. eliminate CPU<->Torch/NumPy churn in the hot path;
9. profile; then selectively apply Numba/C++/OpenMP only to measured kernels;
10. tune block size, thread count, and local/full fallback threshold from benchmarks.

FRONTEND POLICY
- Dragging is preview-only and must remain responsive.
- Pointer-up or debounced parameter changes create an exact-analysis edit.
- Show preview and exact states distinctly.
- Keep scene_revision and job_id in client state.
- Discard stale responses.
- Patch only the returned raster window/texture region.
- Present elapsed time, analysis status, and model scope honestly.
- The demo must remain usable without external CDN dependencies unless the architecture document is explicitly changed.

HARD STOP CONDITIONS
Stop and request human input only if one of these occurs:
1. Scientific oracle behavior is internally inconsistent or non-deterministic beyond documented tolerances after environment controls are applied.
2. Required input semantics cannot be inferred safely from code, tests, or upstream SOLWEIG documentation and choosing one would materially alter scientific output.
3. A requested change would require changing the stated model scope, especially dynamic wind/CFD response, but no acceptance criteria are available.
4. Repository credentials, protected-branch rules, licenses, or unavailable proprietary data prevent the next required action.
5. A merge/rebase produces a semantic conflict in the scientific core that cannot be resolved by tests and documentation.
6. The full test suite exposes a pre-existing failure that prevents distinguishing regression from baseline and cannot be isolated.
7. A numerical mismatch exceeds tolerance and multiple scientifically plausible fixes exist with no repository evidence for choosing one.

Do NOT stop for:
- test failures caused by your own current patch;
- ordinary dependency installation issues with documented remedies;
- lint/style failures;
- a benchmark missing a target value;
- a slow implementation that is still correct;
- transient frontend rendering issues;
- stale jobs that can be cancelled or superseded;
- conflicts limited to generated documentation or obviously unrelated formatting.

RECOVERY
On interruption or context reset:
1. Read WORKLOG.md.
2. Run git status and git log -n 10 --oneline.
3. Restore the exact environment or record any unavoidable difference.
4. Run the checkpoint smoke command recorded in WORKLOG.md.
5. If the working tree is dirty, classify changes as intentional, generated, or unknown before editing.
6. Resume from the first unverified gate, not from memory.
7. If state appears corrupted, reset to the last recorded rollback point and replay the smallest uncommitted patch.

FINAL ACCEPTANCE
Do not declare completion until:
- all mandatory phase gates pass;
- scientific differential validation passes for required fixtures;
- full repository tests pass in a supported environment;
- docs build without broken links;
- frontend interaction tests pass;
- deployment smoke passes;
- benchmark report includes target CPU-node measurements;
- peak RSS and latency are recorded, not guessed;
- limitations are documented;
- the working tree is clean;
- commits are pushed to the intended branch if credentials allow;
- the final report lists commit SHAs, commands, test counts, benchmark values, known limitations, and any deferred optional tasks.

If direct push to main is prohibited, do not bypass protection. Push an implementation branch, prepare a PR-ready summary, and report the exact permission/policy blocker. If direct push to main is permitted and the user has explicitly requested it, push only after the final gate is green.

Begin now. Do not ask for confirmation unless a hard stop condition is already present.
```

## Usage notes

The master prompt should be paired with the machine-readable mission file in [mission.yaml](mission.yaml). An agent should treat the prose documents as authoritative for semantics and `mission.yaml` as the concise source of phase/gate identifiers.

When using a tool that supports long-running tasks, give it the repository checkout, the prompt above, permission to run tests, and permission to commit to an implementation branch. Do not grant permission to bypass branch protection or delete remote history.
