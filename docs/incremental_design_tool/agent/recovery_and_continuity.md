# Recovery, checkpointing, and uninterrupted execution

## Goal

The implementation should continue through routine failures with minimal human intervention. Continuity is achieved by making repository state explicit, keeping scientific jobs idempotent, and making validation restartable.

## Checkpoint layers

### Layer A: Git checkpoint

A green focused commit is the primary rollback point. Avoid keeping hours of verified work only in the working tree.

### Layer B: Worklog checkpoint

`WORKLOG.md` records the semantic state that Git cannot express: current gate, environment, exact validation command, benchmark state, known blocker, and next step.

### Layer C: generated-run manifest

Long scientific or benchmark runs should write a run manifest containing input hashes and commit SHA. Completed immutable stages may be reused after interruption only if the manifest matches.

### Layer D: service job state

The production design uses explicit job state and scene revision. Published exact results change atomically. In-progress results are disposable and can be rerun.

## Worklog protocol

Create `WORKLOG.md` from [worklog_template.md](worklog_template.md). It is an execution artifact, not a substitute for permanent design docs. If committing it creates excessive churn, keep a concise committed phase-status section and write detailed transient logs under an ignored local run directory. The important requirement is that the next agent can recover the current state.

Update on:

- start of session;
- before and after a phase gate;
- before a risky scientific refactor;
- after upstream rebase;
- after a benchmark set;
- after discovering a blocker;
- before ending a session.

## Atomic artifact pattern

For expensive deterministic generation:

```text
run/<run-id>/
  manifest.partial.json
  outputs.partial/
```

After all files and hashes verify:

```text
manifest.json
outputs/
DONE
```

Never treat the presence of one output file as evidence that the stage completed.

## Idempotent agent actions

Prefer commands that can be safely repeated. Examples:

- cache build to a versioned destination followed by atomic activation;
- fixture generation keyed by fixture/version hash;
- tests in temporary directories;
- migrations that first read schema version;
- result patch publication conditional on current `scene_revision`.

Avoid scripts that append uncontrolled duplicate records or overwrite the only copy of a baseline artifact.

## Crash recovery decision tree

```text
Agent/session interrupted
        |
        v
Read WORKLOG + git status
        |
        +-- clean tree? -- yes --> run checkpoint smoke --> resume first unverified gate
        |
        no
        |
classify dirty files
        |
        +-- known intended + tests recorded? --> reconstruct/rerun validation, then commit
        +-- generated only? --> remove/regenerate from manifest
        +-- unknown mixed changes? --> save patch, reset to rollback commit, replay selectively
```

Before discarding unknown changes, save them locally:

```bash
git diff > /tmp/incremental-recovery.patch
git status --short > /tmp/incremental-recovery-status.txt
```

Do not commit corrupted or unexplained state just to make the tree clean.

## Handling stale and superseded work

A rapid editing UI naturally creates obsolete jobs. Obsolescence is normal, not an exception.

The worker should check supersession:

- before expensive cache/window preparation;
- after visibility/SVF stage;
- between temporal blocks where safe;
- immediately before publication.

A superseded job may exit as `SUPERSEDED`, leaving no visible partial patch. It should not be retried automatically.

## Retry policy

Classify failures:

| Class | Examples | Retry |
|---|---|---|
| deterministic input | invalid tree, outside site | no |
| stale/superseded | newer scene revision | no |
| transient resource | temporary I/O, worker restart | bounded retry |
| memory pressure | local ROI unexpectedly large | retry once as full fallback only if safer; otherwise fail explicitly |
| scientific mismatch | oracle validation failure | no automatic tolerance change; investigate |
| cache corruption | checksum mismatch | rebuild immutable cache, then retry |

Retries must be bounded and observable. Avoid infinite polling or infinite job retries.

## Branch and merge continuity

The preferred workflow is implementation branch -> green gates -> PR -> main. If direct `main` commit is explicitly requested and permitted, still commit at phase boundaries and never force-push rewritten history.

If branch protection prevents direct push, that is not a reason to bypass it. Push the branch and leave a PR-ready handoff.

## External access failure

If GitHub write access or another external service is unavailable:

1. continue all work that can be verified locally;
2. maintain clean commits in the local repository;
3. generate a `git format-patch` bundle from the appropriate base;
4. record the expected base commit;
5. verify the patch by applying it to a clean copy and rerunning at least structural/unit validation;
6. report the exact denied operation and status code;
7. do not claim the remote branch was updated.

## Minimizing human interruptions

The agent should make a decision and continue when the decision is reversible and non-scientific. Examples include file organization, test helper naming, internal data classes, logging structure, and local benchmark harness shape.

The agent should use repository evidence and document the choice when the decision affects behavior but has one clearly safer option. Examples include using full fallback when ROI safety is uncertain, rejecting invalid tree values, and refusing stale result publication.

Only the hard-stop conditions in the master prompt require human input.
