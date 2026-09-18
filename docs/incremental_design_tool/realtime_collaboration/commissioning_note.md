# Commissioning note: measured envelope, chaos evidence, and operational guidance (R8)

Status: R8 final-gate evidence (load, chaos, soak, commissioning), collected
2026-09-05 on the development host. Every number below is measured; evidence
files and log paths are listed in the inventory. Nothing here is a claim about
the fly demo node — the fly section gives an honest expected-envelope ratio
and explicitly does not present remote numbers.

## Host and workload

| Item | Value |
|---|---|
| Host | Apple M1 Pro, 10 cores, macOS (see `host` block in each evidence JSON) |
| Host load during runs | load average ~5.0 (recorded `uptime` in every evidence file) |
| Server | REAL `create_app` production wiring in a uvicorn subprocess: 100 ms epoch scheduler, deadline fast lane with the R7 compensated vegetation kernel, `FastAdmission` at documented defaults, real universal-dispatch exact solver |
| Site | real prepared 128x128 site with genuine SVF artifacts and a real full-run baseline (built once under `/tmp/r8_proof`, read-only afterwards) |
| Workload | 4 concurrent clients across 2 workspaces, ~1 op/s/client, op mix view ~70% / vegetation ~15% / meteorology ~10% / landcover ~5%, with deliberate duplicate resubmissions (1 in 15) and stale-base ops (1 in 15) |
| Disclosed knob | the legacy per-scenario edit budget (20/min) and per-IP budget (30/min) are DISABLED for the envelope runs — they are deployment defence-in-depth, not the realtime admission contract. The realtime admission envelope ran at its documented defaults (128 ops/epoch, 128 ops/s/workspace, 100k cells/epoch, 64 KiB/op, 256 workspaces, 550 ms compute budget). |

## Measured envelope (10-minute calibration run)

Evidence: `/tmp/r8_proof/load.json` (driver log `/tmp/r8_proof/calibrate_driver.log`).

| Metric | n | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Op accept -> canonical SSE visibility (ms) | 2486 | 142.3 | 364.1 | 480.7 | 537.4 |
| Fast revision, class `fast_exact` (ms, server) | 2244 | 155 | 389 | 490 | 536 |
| Fast revision, class `visual_pending` (ms, server) | 1376 | 152 | 380 | 497 | 540 |
| Fast publish -> client SSE arrival (ms) | 2244 | 1.0 | 1.5 | 4.4 | 14.6 |
| Exact job queued -> finished (ms, durable rows) | 555 | 2409 | 2614 | 2751 | 5080 |
| Exact catch-up, canonical -> exact coverage (ms, 500 ms poll-limited) | 7 | 385 | — | 4529 | 4529 |

Counts: 2486 ops submitted (all acked, zero client errors), 150 duplicate
replays flagged, 0 fingerprint conflicts, 164 deliberate stale-base ops
(173 acked with `base_divergent`, advisory as designed), 0 SSE drops, 0
reconnects, 0 admission rejections at this load (the load sits far inside the
envelope). Result classes observed: `fast_exact` 2244, `visual_pending` 1376,
`fast_qualified` 0 (workload property: the R7 compensated kernel publishes
`fast_exact` for vegetation edits on this site; met/landcover edits publish
`visual_pending`; no op in this mix produced a qualified response, so
qualified supersession count is 0). Queue depth never exceeded 2 queued /
1 running; final exact lag 0 on both workspaces after quiesce.

### Verdict vs the service-level contract SLOs

| SLO (service_level_contract.md) | Target | Measured | Verdict |
|---|---:|---:|---|
| Remote collaborator visibility p95 / p99 | <= 150 / 300 ms | 364 / 481 | MISS (p50 142 within) |
| Canonical workspace revision p99 from accept | <= 250 ms | 481 | MISS |
| Fast analysis revision p99 from accept | <= 1000 ms | 490 (`fast_exact`), 497 (`visual_pending`) | PASS |
| Epoch duration | 100 ms nominal | 100 ms fixed tick | PASS (see wiring note) |
| Accepted-op inclusion | 100% | 100% (zero-loss verified against the durable log every run) | PASS |
| Duplicate idempotency | exactly one effect | duplicates flagged, zero extra durable rows, byte-identical replay fold | PASS |
| Op acknowledgement p95 / p99 | <= 50 / 150 ms | not instrumented | NOT MEASURED (disclosed) |
| Availability (1-hour class) | >= 99.9% | soak: 0 failed fast updates, 0 SSE drops | PASS (see soak) |

The visibility/revision SLO misses are real and attributed below; they are a
tail-latency effect of the 100 ms ticker under host load, not a queueing or
correctness problem. p50 sits at the structural aliasing floor.

## Latency decomposition and attribution

Server-side telemetry (per-phase histograms, `/metrics` realtime registry,
in the calibration evidence) splits visibility completely:

| Phase (server side) | p50 | p99 |
|---|---:|---:|
| epoch wait (accept -> close) | 150 | 480 |
| epoch reduce | 0.05 | 0.13 |
| canonical publish | 0.10 | 0.48 |
| fast compute | 0.008 | 0.13 |

So visibility == epoch wait, one-for-one (client-side SSE transport adds
p99 4.4 ms — exonerated). Attribution ladder for the wait tail:

1. **Structural floor confirmed.** The scheduler closes an epoch on the first
   tick where its age >= the window, so a wait is uniform in [W, 2W) =
   [100, 200) ms: p50 = 150 (measured exactly), p99 ~ 199 by floor alone.
   The SLO numbers are satisfiable by this design *if ticks fire on time*.
2. **Close work exonerated** — reduce/publish/compute are sub-millisecond.
3. **Exact-lane physics exonerated** — a diagnostic run with the exact
   solver stubbed to a no-op (`R8_DIAG_STUB_SOLVER=1`,
   `/tmp/r8_proof/diag_stub.json`) still measured visibility p99 465.8 ms.
4. **Window aliasing exonerated as the tail cause** — a 50 ms-window probe
   (`/tmp/r8_proof/probe50.json`) moved p99 only 481 -> 451, not to ~150.
5. **Remaining cause: ticker wake overshoot under scheduler/GIL contention.**
   12 all-thread stack dumps across the soak (faulthandler, 60 s cadence)
   show the epoch-scheduler thread healthy — parked in the tick `wait()` at
   nearly every sample, never deadlocked, never inside long work. The
   measured tail (~280 ms above the structural floor at p99) is consistent
   with the 100 ms `Event.wait(interval)` loop waking late on a host at load
   ~5 with ~13 Python threads in the server process plus the load driver.
   This is bounded by observation, not fully attributed: a tick-duration
   histogram does not exist yet (see follow-ups).

**Wiring surprise (commissioning-relevant):** `create_app(coalescing_window_ms=...)`
reaches ONLY the JobRunner legacy-lane coalescing. The epoch scheduler's tick
is a fixed 100 ms default (`EpochScheduler` default `tick_ms`); it is tunable
only by injecting `epoch_scheduler_factory`. The probe in item 4 therefore
changed the JobRunner knob, not the ticker — which is why it moved nothing.
Any deployment that wants a 50-200 ms epoch (contract line "configurable
50-200 ms") must inject a custom scheduler factory.

## Admission and backpressure (chaos case `accept_queue_overflow`)

6 flood workers x 60 batches x 8 ops against one workspace: 16 batches
accepted, 344 refused 429 `fast_lane_overloaded`, 0 transport errors, server
alive throughout. Durable-log cross-check: exactly the accepted ids present,
zero phantom rows from refused batches — reject happens BEFORE the durable
append (fenced in CI by `tests/test_realtime_r8_chaos.py`). Localhost
sequential accept throughput measured ~2.8 ms/op (41 ops inside one 117 ms
epoch window), so the 128 ops/s/workspace envelope binds long before the
host does.

## Overload soaklet (2x calibration load, 5 minutes)

Evidence: `/tmp/r8_proof/overload.json`. 8 clients, 2505 ops, verdict PASS.
Visibility p99 458 ms (statistically unchanged from 1x), queue depth bounded
(2 queued / 1 running), RSS bounded (peak 439 MB), zero failed jobs, exact
lag drained to 0. Honest finding: at 2x nominal load the admission envelope
never engages (the workload reaches ~4 ops/s/workspace vs the 128 ops/s
bound) — backpressure engagement is evidenced by the flood case above, not
by this soaklet. No OOM, no unbounded queue growth, full recovery.

## Chaos matrix (8/8 PASS)

Evidence: `/tmp/r8_proof/chaos.json`, per-case artifacts under
`/tmp/r8_proof/runs/<case>/evidence.json`.

| Case | Injected fault | Verdict |
|---|---|---|
| `accept_queue_overflow` | 6-worker flood past the envelope | PASS (344x 429 pre-acceptance, zero phantoms, server live) |
| `client_disconnect_mid_epoch` | SSE drop at t+20s | PASS (1 drop, 1 reconnect, zero-loss via catch-up, visibility unaffected) |
| `crash_epoch_mid_close` | `os._exit` after 3rd closed-epoch mark, before reduction | PASS (restart re-drives the stranded closed epoch; zero loss; revisions 1..N) |
| `crash_exact_mid_solve` | `os._exit` inside the physics series (4th step) | PASS (job re-queued post-restart; no torn survivors) |
| `crash_fast_publish_pre_commit` | `os._exit` on entry of 2nd fast-revision advance | PASS (fast scan self-heal) |
| `crash_fast_publish_post_commit` | `os._exit` after 2nd fast-revision commit | PASS (durable revision is truth; broadcast resumed) |
| `crash_accept_mid_txn` | `os._exit` on entry of 25th op append | PASS (idempotent client redelivery; no double-apply) |
| `duplicate_stale_determinism` | 41-op identical sequences into 2 workspaces + duplicates + stale bases + mutated-id conflict | PASS (byte-identical canonical state; duplicate replays flagged; mutated id 409 `operation_id_reused`) |

Two harness-model findings from the first pass, both CORRECT server behavior
that the client must respect (the fixed harness re-run passed 8/8): a lost
POST is indistinguishable from a lost ACK, so redelivery must reuse the same
operation ids (fingerprint dedupe makes this safe); and a client must forget
optimistically-added entities whose adds never landed, or later edits trip
the exact lane's unfoldable-native-op fence (typed `edit_rejected`, span
settles, lane advances — validated as the designed loss model).

## One-hour soak

Evidence: `/tmp/r8_proof/soak.json`; driver log `/tmp/r8_proof/soak_driver.log`;
server log with 60 s all-thread stack dumps `/tmp/r8_proof/logs/soak.log`.
Single uninterrupted run (4 clients, 2 workspaces, 3600 s + 300 s quiesce,
uptime 3651.6 s, host load ~5.0 recorded via `uptime`), launched only after
calibration passed. Verdict PASS, zero defects.

| Metric | n | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Op accept -> canonical SSE visibility (ms) | 14752 | 155.0 | 372.4 | 491.5 | 660.3 |
| Fast revision, class `fast_exact` (ms, server) | 13320 | 181 | 417 | 535 | 728 |
| Fast revision, class `visual_pending` (ms, server) | 8018 | 185 | 430 | 544 | 866 |
| Exact job queued -> finished (ms) | 2814 | 2747 | 3781 | 4040 | 6749 |

Every percentile n is far above the 1000-sample stability threshold — the
p99s are distributional. Sustained-abuse counts across the hour: 929
duplicate replays (all flagged, zero extra durable rows), 923 deliberate
stale-base ops (984 acked `base_divergent`, advisory as designed), 0
fingerprint conflicts, 0 client errors, 0 SSE drops, 0 reconnects, 0 missed
events — availability during the one-hour class is 100% of admitted fast
updates (>= 99.9% SLO: PASS). Zero-loss held for the full run: 7357 + 7395
acked == durable rows, revisions exactly 1..N on both workspaces, exact lag
drained to 0 by quiesce, queue depth never exceeded 2 queued / 1 running,
0 failed jobs.

**RSS verdict: no leak.** 680 samples over 3605.6 s: first 402.9 MB, peak
442.9 MB, last 320.8 MB, min 253.4 MB; second-half/first-half mean ratio
0.915 and lsq slope -50.4 MB/h (NEGATIVE). The series steps UP ~135 MB in
the first ~5 minutes (first exact jobs materialize numpy scratch) and steps
DOWN ~80 MB around t+2470 s (allocator returns pages) — bounded working-set
behavior, not growth. Latency was stable end-to-end: soak p99s sit within
2-11% of the 10-minute calibration's, so no degradation over the hour.

## Expected envelope on the fly demo (shared-cpu-1x, 2 GB) — honest ratio

No remote load numbers exist; the demo node has not been load-tested (the
demo currently serves the site_500 cache, a different and larger site than
the R8 harness's 128x128). What transfers and what does not:

* **Correctness invariants transfer fully** — they are CPU-independent
  (zero-loss, reject-before-acceptance, monotonic triple, idempotency).
* **Latency floors transfer, tails get worse.** The [100, 200) ms aliasing
  floor is tick-driven, not CPU-driven. The tail is scheduler-contention
  driven: one shared vCPU (vs 10 local cores) with the same thread count
  means worse ticker wake overshoot than the local p99 481 ms. Expect the
  visibility SLO misses to widen, not close, on the demo node.
* **Exact lane scales with single-core physics speed.** Measured 2.4 s/job
  (p50) on 128x128 on the M1 Pro; the demo's site_500 full-run reference is
  ~38-40 s locally and was commissioned expecting ~60-90 s on the shared
  vCPU (~1.6-2.2x single-thread ratio). The exact lane stays asynchronous
  and lag-bounded by design; plan UX around `visual_pending`, not exact.
* **Memory fits with headroom at this workload.** Local peak RSS 443 MB
  (calibration) against the 2 GB volume-backed machine; one workspace,
  bounded epochs, and the exact queue at depth <= 2 keeps the working set
  near the fast-lane arrays and sqlite page cache. Do not extrapolate this
  to many concurrent workspaces on the demo node.
* **Admission envelope is CPU-independent** (rate/cell/payload/workspace
  bounds are configured, not measured against CPU), so it engages at the
  same op rates on fly — it is the primary protection for the shared vCPU.

## Operational guidance

* **Admission thresholds**: keep the documented defaults (128 ops/epoch,
  128 ops/s/workspace) for the demo; they bound accept work far below host
  capacity (measured ~2.8 ms/op accept). Tighten `max_accepted_per_second`
  first if the ticker tail worsens on the demo node — accept-path GIL churn
  competes with the ticker.
* **Queue bounds**: the exact lane is serial by design (depth never exceeded
  2 queued / 1 running at 2x load). Alert if `jobs.queued` stays > 4 for
  minutes — that is exact demand outrunning single-core physics, the
  designed response is `visual_pending` + drained lag, not queue growth.
* **Restart procedure**: `os._exit` at any of the five armed seams recovered
  fully with a plain process restart on the same state root — no manual
  intervention, no torn epochs, stranded closed epochs re-drive themselves
  on boot (`EpochScheduler.start()` recovery). For the fly demo: one
  `flyctl machine restart` is the entire crash procedure.
* **Epoch window**: `coalescing_window_ms` does NOT change the epoch tick —
  that is `create_app(epoch_tick_ms=...)` (50-200 ms domain, validated;
  wired in R8b). Do not "tune" the wrong knob in production.
* **Client contract**: redeliver lost posts with the SAME operation ids;
  forget optimistic entities on definitive refusal; treat
  `edit_rejected` typed failures as client-model bugs (server settles and
  advances either way).

## Follow-ups (measured, not fixed, in R8)

1. ~~Tick-duration histogram so tail attribution stops being inference~~ —
   DONE in R8b: `solweig_rt_epoch_tick_lateness_ms` (see the R8b appendix).
2. Op-acknowledgement RTT instrumentation (the 50/150 ms ack SLO is
   currently unmeasured).
3. The contract's full commissioning matrix (1/16/32 sessions, increasing
   ops/s and cells/epoch to an SLO failure) — R8 measured 4 and 8 sessions;
   the rest is future work on a dedicated load host.
4. `fast_qualified` class exercised by a workload that produces qualified
   responses (none in this mix; supersession count 0 is a workload property,
   not an absence of the mechanism).

## Honest non-coverage

* The 1/16/32-session points and the SLO-failure search of the envelope.
* Any remote/fly measurement (no fake remote numbers in this note).
* Op-ack RTT percentiles.
* `fast_qualified`/supersession under load.
* ~~The visibility tail's root cause bounded by observation, not direct
   measurement~~ — resolved in R8b: the tick-lateness histogram measures it
   per tick (below).

## R8b appendix: quiet-host attempt, tick-lateness evidence, epoch-tick wiring

### The epoch tick is now actually configurable

`create_app(epoch_tick_ms=...)` (R8 review N3): validated against the
contract's own 50-200 ms domain — out-of-domain values raise `ValueError`
at construction; `None` keeps the 100 ms scheduler byte-identical. The
wiring is cadence-proven, not plumbing-proven: closed-epoch rows' mean
`closed_at - opened_at` is ~75 ms at tick 50 vs ~150 ms at tick 100 (the
[W, 2W) floor measured on the real scheduler thread). The R8 "wiring
surprise" paragraph above is now historical: the knob exists, and
`coalescing_window_ms`'s docstring says out loud that it never reaches
the scheduler.

### Tick-lateness histogram: the tail is now direct evidence

New standard metric `solweig_rt_epoch_tick_lateness_ms` — per-tick wake
lateness beyond one interval, observed every scheduler loop iteration on
the monotonic clock. Measured in the R8b run (see below, n=5328 ticks):

| Histogram | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| tick wake lateness (ms) | 10.0 | 35.1 | 316.6 | 388.1 |
| epoch wait (ms) | 155 | 358 | 466 | 544 |
| visibility, client-measured (ms) | 147 | 355 | 467 | 546 |

Composition: an epoch's wait is [W, 2W) floor (p99 ~199 at W=100) plus
the lateness of the late ticks it sat through. One p99-late tick
(316.6 ms) on top of a mid-floor wait composes the measured wait p99
(466) — the attribution ladder's conclusion ("ticker wakes late under
load") is now a measured identity, not an inference from refutations.

### Quiet-host re-measurement: DEFERRED per protocol — here is why

Protocol: record `uptime` before/during/after; if load stays > 3,
defer/retry once rather than ship a noisy number. What happened on this
host (samples in `/tmp/r8_proof/quiet_attempt_uptime.log` and the driver
log):

| When | 1-min load | Dominant consumer |
|---|---:|---|
| 17:12 (attempt 1) | 2.84 -> 3.51 | concurrent `pytest tests/` in the main checkout + agents |
| 17:29 (retry window) | 4.20 | active agent session at ~86% CPU |
| 17:40 (launch decision) | 5.31-6.57 | same |

The load is not this harness's own (all R8/R8b processes were verified
dead); it is concurrent work on the shared development host. Two windows
stayed above 3, so the CLEAN quiet-host number is deferred, not faked.

What WAS run (honestly labeled, at-load): the identical calibration
workload at recorded load 4.2-6.6 (`host_load` gauge mean 4.8),
evidence `/tmp/r8_proof/r8b_visibility.json`, driver log
`/tmp/r8_proof/r8b_visibility_driver.log`, verdict PASS (zero defects,
n=2488 ops):

* visibility p99 466.8 ms at load ~4.8 — statistically unchanged from
  the R8 calibration's 480.7 at load ~5.0 and the soak's 491.5 at
  ~5.0. In the measured load band (4-6.6) the p99 is flat; the quiet
  host (< 3) remains the open case.
* tick lateness p99 316.6 ms (above) — the load-band tail's direct
  mechanism.

### Where this leaves the visibility SLO

The 300 ms p99 SLO is unmet in every measurement to date (loads 4-6.6),
and the miss is now directly attributed to scheduler tick wake lateness
under host load — not to the close pipeline (sub-ms), not to the exact
solver (stub-refuted), not to SSE transport (p99 4.4 ms). Whether a
genuinely quiet host or a dedicated core closes the gap is UNVERIFIED;
that is the commissioning condition the clean number would have tested.
This is a goal-level decision for the user, not ours to make: either
(a) re-commission on a quiet/dedicated host before judging the SLO, or
(b) re-specify the SLO in tick multiples / epoch-close-anchored terms
(the [W, 2W) floor plus bounded tick lateness), consistent with the
reviewer's adjudication that the miss is SLO mis-specification
compounded by host-load lateness, not structural.

## QUIET-HOST RE-COMMISSION (2026-09-06): window found, not held — p99 492.8 ms, verdict unchanged

The follow-up measurement the R8b appendix deferred. Binding protocol
(quiet-host): poll `uptime`, launch only if 1-min load < 3.0, up to
60 min of window-hunting, record every sample, never start above 3.0,
never ship a noisy number.

### The window hunt (38 minutes, 18 samples, all of them)

| time | 1-min load | | time | 1-min load |
|---|---:|---|---|---:|
| 08:44 | 8.30 | | 09:03 | 4.72 |
| 08:48 | 6.14 | | 09:05 | 3.62 |
| 08:50 | 6.63 | | 09:07 | 3.12 |
| 08:52 | 4.52 | | 09:09 | 4.15 |
| 08:54 | 6.59 | | 09:11 | 4.14 |
| 08:56 | 7.25 | | 09:13 | 9.92 (burst) |
| 08:59 | 3.82 | | 09:16 | 5.45 |
| 09:01 | 4.97 | | 09:18 | 4.22 |
| | | | 09:20 | 5.12 |
| | | | **09:22** | **2.77 -> LAUNCH** |

The floor is structural on this shared host: a 5-day-old agent session
at ~56-64% CPU, an endpoint-security daemon at ~26-35%, WindowServer
~24-30%, Chrome renderers, plus unrelated long-lived servers (ollama, a
2-day-old `solweig_gpu.server`). At 09:22 the 1-min average caught a
trough below 3 (5/15-min averages were still 4.72/5.41); the gate was
met, so the calibration run launched per protocol.

### What actually happened during the run (the honest part)

The window did NOT hold. The host's own bursts plus the harness's
footprint (driver + uvicorn server + 4 clients, by design co-located)
pushed the in-run load back up immediately, then the host calmed late in
the body. Every reading taken:

| when | 1-min load |
|---|---:|
| 09:22 pre-launch (gate) | 2.77 |
| harness host@start (post server boot) | 4.39 |
| t+63 s | 4.13 |
| t+123 s | 5.59 |
| t+183 s | 9.07 (host burst) |
| t+244 s | 6.40 |
| t+304 s | 4.79 |
| t+364 s | 4.11 |
| t+424 s | 3.15 |
| t+485 s | 2.63 |
| t+545 s | 3.83 |
| 09:33 end | 3.77 |
| 09:35 / 09:38 post-run (harness dead) | 5.84 / 5.35 |

`solweig_rt_host_load` gauge over the whole run (0.5 s sampling,
n=1116): **min 2.377 / mean 4.563 / max 9.074**. Post-run floor ~5.3
with zero harness processes attributes most in-run load to the HOST,
not the workload. Conclusion: on this machine a sustained < 3.0
calibration-length window does not exist to be found; 2.77 was a
transient trough, and ~160 s of the 600 s body ran at load 3.2 or below.

### Measured numbers (whole-run — percentiles cannot be windowed post-hoc)

Identical workload to R8b (calibrate: 600 s, 4 clients x 1 op/s,
2 workspaces, view 70% / vegetation 15% / meteorology 10% / landcover
5%, plus 1-in-15 stale and 1-in-15 duplicate submissions; 2,477 ops,
verdict PASS, zero defects, zero SSE drops/reconnects). Evidence:
`/tmp/r8_proof/r8b_quiet_visibility.json` +
`r8b_quiet_visibility_driver.log` (orphan check: zero harness processes
alive after exit).

| metric | p50 | p95 | p99 | max | n |
|---|---:|---:|---:|---:|---:|
| visibility (accept -> SSE canonical arrival) | 141.2 | 372.0 | **492.8** | 554.6 | 2477 |
| epoch wait (accept -> close) | 151 | 377 | 487 | 553 | 1806 |
| tick wake lateness | 10.0 | 39.0 | 322.7 | 437.8 | 5304 (4096 retained) |
| publish / reduce (close pipeline) | 0.11 / 0.05 | 0.28 / 0.10 | 0.60 / 0.13 | 121 / 0.9 | 3612 / 1806 |

Composition identity holds again: wait p99 487 ~= [W, 2W) floor top
(200 at W=100) + tick-lateness p99 (322.7). Per workspace: p99 477.3 /
508.5. Fast classes: fast_exact p99 506 ms (n=2236), visual_pending
p99 512 ms (n=1376).

### The four-run picture

| run | 1-min load @ start (gauge mean) | vis p99 | wait p99 | tick p99 |
|---|---|---:|---:|---:|
| R8 calibration | 3.27 | 480.7 | 480 | (no histogram) |
| R8 soak (1 h) | 5.21 | 491.5 | 509 | (no histogram) |
| R8b re-measure | 5.45 (4.8) | 466.8 | 466 | 316.6 |
| this run (quietst launch) | 2.77 -> 4.39 (4.56, min 2.38) | 492.8 | 487 | 322.7 |

Down to the lowest load yet observed at a run start (R8 calibration's
3.27) and through this run's genuine sub-3.2 late-body dip, the p99
never left the 466-493 band. Within every sampled condition the tail is
flat in load; the host-load-induced-tail hypothesis has now failed to
show ANY load sensitivity anywhere it was measurable, while the
structural floor (p50 ~= 141-155 ms ~= the [W, 2W) floor) and the
in-process mechanism (tick lateness with GIL/OS contention from the
workload's own accept + exact-solver threads — present even at the
quietest observed load) reproduce identically.

### Verdict vs the 300 ms SLO — and the recommendation

* p99 = 492.8 ms > 300 ms: **SLO still missed** in the quietest
  conditions achievable on this host (pre-launch 2.77; in-run gauge
  mean 4.56, min 2.38). Stating a commissioning condition "host load <
  X" is NOT possible from this evidence: no X < 3 held for ten minutes
  here, and the whole-run percentiles cannot certify what the ~160 s
  sub-3.2 stretch alone would have shown.
* Explicit recommendation (the R8b option (b)): **re-specify the
  visibility SLO in tick multiples / epoch-close-anchored terms** —
  p50 within [W, 2W) plus bounded tick lateness (e.g. p99 <= 2W +
  tick-lateness budget), which every run to date satisfies with the
  mechanism fully measured. The absolute 300 ms figure is unsatisfiable
  by design at W=100 ms whenever a single tick lands ~320 ms late,
  which happened at every host load yet sampled.
* The one condition that could still falsify "the floor is structural,
  not host-load" — a dedicated host sustaining < 3.0 (or an idle core /
  raised scheduler-thread priority) for a full calibration run —
  remains unmeasured and, on this shared machine, unmeasurable. If the
  re-spec is not acceptable, that measurement is the next and only
  step; do not re-run on this host expecting a different band.

## SLO re-spec decision (2026-09-06, user-approved)

Outcome (b) of the R8b decision frame: the quiet-host re-commission
confirmed the miss, and the user approved the tick-multiple /
epoch-close-anchored re-spec (the agent recommendation and the review
adjudication's primary option).

**New contract** (W = configured epoch tick, 100 ms default):

| Quantity | Target | Measured (4 runs: R8 calib, 1 h soak, R8b, quiet-host) |
|---|---|---|
| Visibility p50 | within [W, 2W) | 141-155 ms |
| Visibility p99 | <= 2W + 350 ms (550 ms at W=100) | 466-493 ms |
| Tick-wake lateness p99 | <= 350 ms (own SLO, `solweig_rt_epoch_tick_lateness_ms`) | 316-323 ms |
| Canonical close pipeline p99 | <= 5 ms after epoch close | 0.1-0.6 ms |
| Fast analysis revision p99 | <= 1,000 ms (unchanged, absolute) | 506-544 ms PASS |

Under the re-spec every measurement run to date is INSIDE contract,
with the mechanism fully measured (composition identity: wait p99 ~=
2W floor top + tick-lateness p99). The absolute 300 ms figure is
retired, not waived: it remains achievable only on a host whose
scheduler sustains near-zero tick lateness through a calibration run —
no such window exists on the shared reference machine (four runs, tail
flat in load 2.77-6.6). A future deployment on a dedicated host SHOULD
re-measure and may re-adopt the absolute figure; the lateness SLO is
the instrument that would justify it.

Artifacts updated in the same decision: `service_level_contract.md`
(SLO table + re-spec section), `realtime_contract.yaml`
(`slo_targets`), `goal_condition_realtime_collab.txt` (mission text,
in-place amendment with provenance).
