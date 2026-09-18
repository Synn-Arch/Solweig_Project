# R5 design: temporal checkpoints, sparse replay, and per-cell fast serve

Status: brief (lead-authored after r5b-design agent died; reviewed before
implementation packets dispatch). Scope covers the two R5 levers in execution
order: **G1** completes R5a (cell-remainder retention @ 357cdbe) by letting the
met fast path actually *serve* mixed-provenance coverage; **G2** is the
original R5 body (temporal checkpoints + sparse replay over the time axis).

## Where R5a stopped (and why G1 exists)

R5a fixed the *coverage* half of the fast-path freshness gate. After a
windowed revision-N+1 batch supersedes a revision-N full-tile entry, the
TemporalResultStore now retains the rev-N cell remainders, so
`window_coverage("tmrt", t, full)` reports FULL again
(`solweig_gpu/incremental/store.py`, `_subtract_windows` supersession branch).

The *serve* half is still refused. `executor.py` `_run_met_utci_fast_path`
requires, at executor.py:1585-1593:

1. `revisions = {entry.scene_revision ...}; len(revisions) != 1 → refuse`
   ("mixed provenance"), and
2. `revision != store.revision_at("tmrt", t) → refuse` ("not the latest").

Post-retention coverage is `{N, N+1}` → check 1 refuses. Even with one
revision, a *newer* published revision for that `(kind, t)` trips check 2.
So retention restored FULL coverage that the gate still declines to serve —
the lever's user-visible value (utci fast path surviving a geometry batch)
is zero until G1.

## G1 — per-cell-latest-validity relaxation

### The contract to prove

Composed serve over retained entries equals the revision-`R` full recompute
**iff**, for every cell `c`:

> the newest entry covering `c` has `scene_revision == R` **and** its value
> equals the value the full recompute at `R` produces at `c`.

The second clause decomposes into the **window-planner superset property**:
for every revision transition `r → r+1` that published windows `W(r+1)`,
every cell whose recomputed value changed between scene `r` and scene `r+1`
lies inside `W(r+1)`. Cells outside the write windows then *provably hold
their rev-r values at rev r+1*, and painting entries in ascending-revision
order reconstructs the recompute exactly.

This property is **not yet pinned**. R5a's re-pinned tests assert the store
semantics (composed serve == new-patch values inside batch windows, rev-N
values in the retained remainder, no phantom gaps) — the remainder values
are trivially rev-N because the retained entry *is* rev-N. Nothing today
asserts those rev-N remainder values equal an *independent full recompute
at N+1*. That differential is the missing proof, and it is exactly the
parity fence class W2 established for the geometry side.

### Steps (red-first, one commit each, non-author review + differential)

**G1.0 — superset differential harness (the proof).** Seeded edits (tree
add/move/delete/resizse, multi-op batches; reuse the V4 site families) drive
a windowed rev-N+1 batch over a rev-N full-tile publication. For each case:
run the full-tile recompute at N+1 AND the composed store serve; assert
bitwise equality **tile-wide** (not just inside the batch windows). Any
mismatch is a planner-window undershoot — a correctness defect to be fixed
in the *window derivation*, never absorbed by widening the serve gate.
Record: sites × edits × outcome, EXIT code, host load, full log.

**G1.1 — gate relaxation.** In `executor.py`, replace checks 1-2 with:
build the composed plane by painting `coverage.entries` in ascending
`scene_revision` order (later revisions overwrite); simultaneously build a
max-revision raster from the same paint order; serve iff
`(max_revision_raster == store.revision_at("tmrt", t)).all()`. A cell whose
newest covering revision lags the published head = a window undershoot
somewhere in history = refuse (stale fence preserved, same refusal class
text). PARTIAL coverage still refuses (unchanged). No new policy switch:
the relaxation is admissible *only* because G1.0 pins the superset
property; if G1.0 finds an undershoot family, G1.1 does not ship until the
planner fixes it.

**MANDATORY same-change fix (review evidence, not optional):** the
`seen_patches` dedup at executor.py:1600-1604 assumes ≤1 store entry per
patch path per key. R5a broke that invariant (a superseded full-tile patch
legitimately owns up to 4 narrowed remainder entries). r5a-review2's probe
D replicated the failure: with the mixed-provenance gate relaxed but the
dedup kept, only the first remainder's slice lands and 384/576 cells stay
silently NaN. The dedup is unreachable today ONLY because the gate
refuses first — exactly the gate G1.1 relaxes. G1.1 MUST dedup only the
`load_patch` call (load each patch file once), never the per-entry slice;
every entry paints its own window.

**G1.2 — heterogeneous coalescing + re-serve tests.** Red-first: met
fast-path serve after geometry batches (mixed {N, N+1}) now succeeds with
bitwise-correct planes; simulated stale cell (hand-trimmed write window
leaving an uncovered-impacted cell) still refuses; fast-path result
classification (`fast_exact`) unchanged. Pin the SLO story: utci fast p99
stays under the 1000 ms fast-analysis budget with a windowed batch in
history.

Non-goals / fences: no change to `window_coverage` semantics, retention
math, or the PARTIAL refusal; no relaxation for non-tmrt kinds (the met
fast path is the only consumer today); never widen by trusting planner
windows without the G1.0 differential.

## G2 — temporal checkpoints + sparse replay

Problem: time-axis edits (selected_date_time, meteorology ranges) and crash
recovery currently recompute or replay whole series. The durable state
already pins per-(kind, t) coverage; what is missing is (a) *checkpoints*
— periodically captured full solver state (thermal ground state at k, veg
SVF occluder store already has its own) that bound replay length — and
(b) *sparse replay* — replay only the times whose inputs changed, using
checkpoints as entry points.

Two independently-produced agent design briefs (r5b-design, r5b-design2,
2026-09-05, both file:line-verified) converged on the same packet shape;
their code-proven findings are folded in below and are binding on the
implementation packets:

- **Checkpoint payload (exact):** 6 float32 full-tile planes —
  Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, TgOut1 (the surface thermal
  chain, utci_process.py:614-619; mutated only in TsWaveDelay_2015a,
  solweig.py:451-488) — plus scalars CI, firstdaytime, timeadd, Twater,
  plus next_step. Twater/CI are recomputed at midnight rows
  (utci_process.py:790-805): omitting them breaks warm starts mid-day.
- **Fingerprint:** key checkpoints by digests of the inputs that determine
  the state — composed scene digest + resolved landcover digest + canonical
  model-params digest + met-prefix digest (rows 0..next_step-1) — NOT by
  (workspace, revision). Revisions can skip under mixed direct-edit +
  realtime traffic, and `_check_baseline_amplitude` (solver.py:1580-1608)
  interacts with march regime: a weak key is silently wrong science. On
  fingerprint mismatch: typed refusal → cold replay + fallback_reason
  telemetry; never an error surfaced, never a guess.
- **Extent:** solver state is allocated at WRITE-window extent
  (utci_process.py:599, :614) but the thermal chain is elementwise (no
  neighbor reads) ⇒ a windowed trajectory is a crop of the full-tile
  trajectory. Persist FULL-TILE states only; windowed consumers slice.
- **Warm start helps only ACROSS jobs/revisions, never within one job:**
  every write window must run its full t-series to emit its own planes
  (outputs differ per window), so two windows of one job cannot share
  temporal state. The genuine intra-job duplication is the per-t march at
  overlapping read extents — the `precomputed_shadows` seam
  (utci_process.py:806-824) is the separate adjacent lever (defer; own
  packet).
- **Default-None additive plumbing:** `initial_state` /
  `return_final_state` parameters on `run_utci_window` (and threading
  through `solve_window`) must default to bit-identical behavior; the
  warm==cold fence below is the proof.
- **Expected win (honest):** a radiation-affecting met edit at row r0
  stops paying the unchanged prefix — site_500 row-20 edit drops the
  loop from ~24 to ~4 full-tile steps (~40 s → ~8 s tloop under load).
  No win claimed for veg/paint/params families (chain invalid from the
  first daytime step — honest boundary).

### Steps

**G2.0 — checkpoint store module (in flight, g2-checkpoints agent).**
`solweig_gpu/incremental/checkpoints.py` (or `temporal_state.py`):
CheckpointRecord with the exact payload above, input-digest fingerprint,
atomic write (staged dir + rename, the VegOcclusionStore idiom — clone
from veg_svf_state.py:822-1007 / worker.py `_commit_veg_state`
:1397-1421), checksum validation on load with typed refusal → None (caller
falls back to cold), rotation keep-last-K (default 3) + base. Bitwise
roundtrip is a hard gate; torn/corrupt checkpoint MUST be rejected.

**G2.1 — state-capture plumbing + first consumer.** Additive
default-None `initial_state` / `return_final_state` / state-extent
parameters on `run_utci_window` (utci_process.py), threaded through
`solve_window` (solver.py, next to the existing `time_stop` fence
:1679-1687); worker captures the full-tile final state after `_solve_full`
and commits at publication (never before — a superseded job persists
nothing). First consumer: the REFUSED branch of the r3a met fast path
(executor.py:1536-1541) — a radiation-affecting met edit at row r0 today
falls back to a full 0..T replay although rows < r0 are provably unchanged;
serve t < r0 from the store under the existing strict gate, solve r0..k
warm from the checkpoint, publish one sparse patch (time_start = r0;
ResultPatch.time_indices already exists, executor.py:1648-1650). Parity
fence: warm(k..N, state@k) == cold(0..N)[k..N] bitwise, and the executor
E2E result == today's cold replay bitwise.

**G2.2 — crash-restart validation.** Kill/restart harness at seeded points
(mid-batch, post-accept/pre-publish, post-checkpoint); assert zero accepted
ops lost, revision monotonicity, and no torn checkpoints (checksum reject
→ fall back to previous). Feeds R8's chaos suite.

Fences: checkpoints are REPRODUCE aids, never a source of truth ahead of
the op log; replay order is accumulation order (never reordered while
claiming bitwise parity); checkpoint tensors are float32-exact (no dtype
narrowing silently).

## Order and gates

G1.0 → G1.1 → G1.2 → G2.0 → G2.1 → G2.2. Each step: red test first, one
commit, full-log canonical with EXIT + uptime, non-author review, ledger
row. G1.0's differential is the gate for G1.1 — no differential, no
relaxation.
