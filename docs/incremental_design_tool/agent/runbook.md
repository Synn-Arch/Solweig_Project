# Autonomous agent runbook

## Objective

This runbook turns the design specification into a continuous implementation loop that can survive long tasks, context-window resets, machine restarts, failed experiments, and upstream changes without repeatedly asking a human what to do next.

## 1. Bootstrap

At the start of a fresh agent session:

```bash
git status --short
git branch --show-current
git log -n 10 --oneline
python --version
node --version || true
```

Then inspect environment-specific package instructions in the repository. Use the project's supported conda/GDAL setup for the complete suite. Do not substitute a different GDAL major version for final scientific validation without recording it.

Create a working branch unless the user has explicitly authorized direct work on an unprotected `main`:

```bash
git fetch origin
git switch main
git pull --ff-only
git switch -c feat/incremental-design-worker
```

If a branch already exists, continue it instead of creating a duplicate.

## 2. Baseline before editing

Run the smallest checks that establish the checkout is sane, then the supported full suite before scientific refactoring. Record every command and result in `agent/WORKLOG.md`.

At minimum:

```bash
python -m compileall solweig_gpu tests
pytest -q tests/test_incremental_geometry.py \
          tests/test_incremental_bitmask.py \
          tests/test_incremental_edits.py \
          tests/test_incremental_design_docs.py
node --test studio/tests/*.test.mjs
```

The final baseline must also execute the repository-wide test command in the supported environment.

## 3. Task-selection algorithm

Select work using this priority:

1. A failing mandatory gate in the current phase.
2. A missing test required to make that gate measurable.
3. A correctness bug exposed by that test.
4. A required integration boundary for the next gate.
5. Performance work only after correctness gates for that stage are green.
6. Documentation and observability needed to make the implementation recoverable.
7. Optional polish.

Do not jump to later-phase performance work because it appears easy if an earlier scientific correctness gate remains unresolved.

## 4. Inner development loop

For each task:

```text
REPRODUCE -> TEST -> IMPLEMENT -> NARROW VERIFY -> DIFFERENTIAL VERIFY
-> PERF VERIFY IF RELEVANT -> DIFF REVIEW -> DOC/WORKLOG -> COMMIT
```

### Reproduce

Write down the observed behavior and exact command. For a bug, create the smallest deterministic reproducer before changing code.

### Test first when feasible

For observable behavior, add a failing test first. Refactors that are strictly internal may use characterization tests instead.

### Implement minimally

Prefer adding an adapter/window abstraction around the existing scientific implementation before rewriting formulas. Preserve the oracle path.

### Narrow verify

Run the closest unit and module integration tests immediately. Do not wait for a large suite to discover syntax errors.

### Differential verify

Any change involving the scientific path requires oracle comparison before commit. See [validation_pipeline.md](validation_pipeline.md).

### Performance verify

Performance claims require before/after numbers from the same fixture and environment. Include warm and cold measurements when cache behavior changes.

### Diff review

Use:

```bash
git status --short
git diff --stat
git diff --check
git diff
```

Reject accidental binaries, generated cache artifacts, absolute paths, credentials, `.DS_Store`, runtime result rasters, and unrelated formatting.

### Checkpoint and commit

Update `WORKLOG.md`, then create a focused commit. The commit itself is the rollback unit.

## 5. Long-running task policy

A task that takes minutes or hours must be made restartable whenever practical.

For fixture generation and benchmarks:

- write outputs into a unique temporary/run directory;
- write a manifest only after all expected outputs exist;
- include code commit, environment fingerprint, fixture ID, and input hashes;
- never overwrite the last known-good benchmark result in place;
- use atomic rename from `.partial` to final artifacts where supported;
- on restart, verify hashes and reuse completed immutable stages.

For scientific jobs:

- use `job_id` and `scene_revision`;
- stage result patches before publishing;
- commit publication metadata atomically;
- allow a superseded job to terminate after the current safe stage;
- never partially overwrite the scenario's visible exact result.

## 6. Context-reset procedure

If the agent loses conversational context, repository state is the source of truth. Read, in order:

1. `docs/incremental_design_tool/agent/WORKLOG.md`
2. `git status`
3. recent commits
4. current phase in `agent_execution_plan.md`
5. the current failing test or benchmark referenced by the worklog

Do not infer unfinished work from memory.

## 7. Dependency failure policy

If an install or CI dependency fails:

1. determine whether it is local/transient or project-wide;
2. use the documented environment manager rather than ad-hoc binary replacement;
3. record versions and the exact error;
4. continue independent documentation/unit work if scientific validation is blocked;
5. mark affected gates `BLOCKED`, not `PASS`;
6. retry at the next checkpoint or on a clean environment.

A temporary inability to run one validation tier does not justify skipping it at final acceptance.

## 8. Upstream synchronization

Only synchronize upstream from a clean checkpoint.

```bash
git status --porcelain  # must be empty
git fetch origin
git rebase origin/main
```

After rebase, run at least the validation tiers affected by upstream changes. If upstream touches `shadow.py`, `solweig.py`, `utci_process.py`, cache schemas, raster I/O, or the frontend state protocol, run the scientific differential or relevant integration tiers before continuing.

## 9. Failure triage hierarchy

When a test fails:

1. Confirm the failure is deterministic.
2. Determine whether baseline `main` also fails in the same environment.
3. Reduce to the smallest fixture.
4. Classify as geometry, cache representation, scientific calculation, temporal state, serialization, concurrency/staleness, frontend state, or environment.
5. Fix the root cause, not the tolerance, unless the documented tolerance itself is demonstrably wrong.
6. Add a regression test.
7. Re-run the parent validation tier.

## 10. End-of-session checkpoint

Before voluntarily ending a session, leave a recoverable repository:

- commit all verified intended changes, or clearly record intentionally uncommitted files;
- no untracked scientific result data;
- update `WORKLOG.md` with the next exact command;
- state the last green validation tier;
- state the rollback commit;
- record any running external job IDs and whether they are safe to ignore/restart.

A future agent must be able to resume without asking what happened.
