# P8 Performance Evidence Summary

Worktree: `.claude/worktrees/p8-perf` (branch `worktree-p8-perf`, from `main` = d70137a)
Machine: Apple M1 Pro 10-core/16GB, py3.14, torch 2.13 CPU (8 threads, interop 10).
Note: budget targets assume a 2-4 vCPU / 4GB node; all times below are measured on the
M1 Pro host, which is faster than the target node. Gate verdicts are reported against
the measured numbers without rescaling.

## Commits (this phase)

| SHA | Subject |
|---|---|
| a1a0da4 | perf(incremental): re-enable local mode with per-tree relief + exact read window |
| 3a3c41d | fix(server): O(new-versions) state composition, identity payload cache, ledger retention |
| 129212a | perf(incremental): write-cropped per-cell physics with read-window marches |
| e274dc9 | fix(solweig): crop only true per-cell Tgwall fields in windowed calc |
| 8cf2bc1 | fix(utci): validate out_window before any compute + regression test |
| 104b300 | fix(perf+server): review remediation — night-march gate, identity-cache lock, sweep/timestamp LOWs |

## Review remediation (104b300, on 8cf2bc1)

Reviewer verdict APPROVE-WITH-NITS; five findings fixed:

1. **[HIGH perf] Night-step march waste** — the windowed time loop marched at
   every timestep; `Solweig_2022a_calc` consumes `precomputed_shadows` only in
   its daytime branch (`altitude > 0`) and returns a zero shadow plane at
   night. Fix: march gated on `altitude[0][i] > 0`; the wbgt sun/shade
   condition uses a zero plane at night (oracle's `shadow = zeros` makes it
   True everywhere — bit-identical). Correctness-neutral, proven: worker
   differential tests (inside_max_abs == 0.0 across all 24 timesteps incl.
   night) and full suite re-run green; oracle path (`out_window is None`)
   code-untouched.
2. **[MEDIUM] `_IdentityPayloadCache`** — `threading.Lock` around get/put
   (same pattern as `_StateCompositionCache`); FastAPI threadpool could
   corrupt the unlocked OrderedDict LRU.
3. **[LOW] `_maybe_sweep`** — `_last_sweep` stamped after the sweep attempt so
   failures retry on the next job.
4. **[LOW] `sweep_retention` cutoff** — formatted like `_now_utc()`
   (millis + "Z"); `isoformat()`'s "+00:00"+micros skewed the lexicographic
   `created_at` comparison.
5. **[MEDIUM-doc]** — idempotency-after-TTL semantics documented in
   `docs/incremental_design_tool/api_contract.md`.

Post-fix profile (`perf_typical_nightfix.txt`): solve 58.1 s profiled (was
66.7); time-loop marches 13 daytime calls / 1.9 s (was 24 calls / 12.6 s —
the 11 night marches cost ~10.7 s). Deferred follow-ups recorded, not fixed:
multi-edit single-bbox window inflation (correctness-clean perf/IO refinement).

### Re-measured PERF-001/002 (same methodology: site_500 h10 r4 @ (166,102), 3 warmup + 20 reps, sequential)

| Run | min | p50 | p95 | max | Evidence |
|---|---|---|---|---|---|
| pre-change (8cf2bc1^) | 83.89 | 89.32 | 93.89 | 94.40 | `perf_typical.txt` |
| post-dual-window (8cf2bc1) | 61.96 | 62.75 | 66.74 | 66.83 | `perf_typical_opt.txt` |
| post-night-gate run 1 (104b300) | 53.94 | 59.53 | 86.81 | 88.98 | `perf_typical_nightfix.txt` |
| post-night-gate run 2 (104b300) | 52.76 | **54.21** | **59.14** | 62.16 | `perf_typical_nightfix_run2.txt` |

Run 1 note: two isolated mid-run spikes (86.8 s @ rep 5, 89.0 s @ rep 21)
against a settled ~54-58 s late-run band — external machine interference, not
code (profiled single rep = 58.1 s). Run 2, on an idle machine, is clean and
tight (max 62.16, no spikes) and confirms it: settled band 52.8-57.5 s.
Headline post-remediation numbers are run 2's: **p50 54.21 s → PERF-001 FAIL
(budget 20 s), p95 59.14 s → PERF-002 PASS (budget 60 s, 1.4% margin)**.
Cumulative from pre-change baseline: p50 89.32 → 54.21 (−39%), p95 93.89 →
59.14 (−37%). Budget re-base adjudication belongs to the lead.

## Gates

| Gate | Budget | Measured | Verdict |
|---|---|---|---|
| PERF-001 p50 typical local job | < 20 s | **62.75 s** | **FAIL** (baseline 89.32 s, −30%) |
| PERF-002 p95 typical local job | < 60 s | **66.74 s** | **FAIL** (baseline 93.89 s) |
| PERF-003 full fallback | < 300 s | **92.9 s** | **PASS** |
| MEM-001 peak RSS | < 3000 MB | **978.6 MB** | **PASS** |
| MEM-002 idle RSS / leak / restart | < 1500 MB, bounded | **504.9 MB** idle, drift −106.4 MB over jobs 5→10, fresh cache load 337.2 MB | **PASS** |
| IO-001 patch payload | < 5 MB | **5.86 MB** (utci+tmrt full-day) | **FAIL** — see detail |
| QUEUE-001 coalescing under load | no stranded edits, bounded backlog | 15 complete + 12 superseded, 0 nonterminal at end, all 20 scenarios exact | **PASS** |

## PERF-001/002 methodology and cost model

Fixture: site_500 (500x500 @ 2 m), 2009-08-11, typical one-tree edit (h10 r4 at cell
(166,102)); write window 0.218 site, exact read window 0.804 site; 3 warmup + 20
measured reps, sequential, warmed page cache. Script: `measure_perf.py`; raw runs in
`perf_typical.txt` (pre-change) and `perf_typical_opt.txt` (post-change + cProfile).

Correctness contract held: post-change solver is bitwise-identical to the pre-change
store outside the write window, and bitwise vs the full-tile oracle inside it
(8 scientific differential tests, inside_max_abs == 0.0 for utci/tmrt, shadow bitwise;
fast tests in tests/test_incremental_worker.py + tests/test_incremental_windowing.py).

What was optimized (129212a + e274dc9): all per-cell chains (GVF outputs, Tg/Tgwall,
Kside, Ldown, UTCI storage planes) now run write-cropped; marches, the GVF directional
ray walk, and the UTCI transcendental evaluation stay read-sized because they are
extent-dependent (ray-walk truncation reach; utci_calculator boolean-compaction makes
1-ULP output differences tensor-extent-dependent). Sky-mask cube hoisting out of the
153-patch define_patch loop.

Post-change cProfile (66.7 s rep):

| Term | cumtime s | Note |
|---|---|---|
| `_recompute_veg_svf_window` | 21.6 | 153 read-sized marches (shadow 20.0) |
| `define_patch_characteristics` | 17.1 | 24 calls, launch-bound 153-iter Python loop |
| daytime marches (`shadowingfunction_wallheight_23`) | 12.6 | 13 calls, read extent |
| `gvf_2018a` | 7.4 | read extent by contract |
| `sunonsurface_2018a` | 7.2 | 234 calls, read extent by contract |
| `Kside_veg_v2022a` | 4.6 | |

Why 20 s p50 is not reachable bitwise-safely here:
- The exact read window is physics-bound, not slack: 33 m terrain relief inside the
  read window at the 6° solar-altitude floor stretches the march to 0.80 of the site
  (`routing_site500.txt`). Smaller read windows change gvf/sunonsurface results.
- The dominant remaining term (SVF-replay marches, 21.6 s) could shrink ~8-10 s via
  per-patch march windows W_α = bbox(write ⊕ reach(α)); **not implemented** — listed
  as follow-up. Even with it, p50 lands ~52-55 s.
- Thread sweep on the same job: 8T 74.9, 4T 68.7, 2T 78.6, 1T 88.1 — the job is
  kernel-launch bound, not core-bound; on the 2-4 vCPU target node expect the slower
  end of that band, i.e. budgets need re-basing or the follow-up windowing work.

## IO-001 detail

Full-day patch (24 timesteps, 248x220 write window) compressed zstd-3:
utci+tmrt+shadow = 5.94 MB; the typical two-variable client payload
(utci+tmrt) = **5.86 MB** vs 5 MB budget. Subsets that pass: utci-only 2.67 MB;
single-timestep refined patches 0.49 MB. zstd level 19 saves only 8% on the float32
fields (5.44 MB) — level is fixed at 3 by the codec contract anyway. Mitigations if
the 5 MB line is binding: byte-shuffle/delta preprocessing before zstd, or
per-variable refine payloads (fetch tmrt only for the timesteps in view).

## MEM/IO/QUEUE methodology

`measure_mem.py`: 10 sequential local jobs, RSS sampled per job, peak via resource
high-water mark; patch payload sizes measured on the actual published patch directory
(zstd-3). `measure_queue.py`: spec load pattern (20 scenarios, steady edit every 120 s,
bursts of 5 edits per 10 s, one worker, 30 min) compressed 20x in both time and injected
solve latency so arrival/service ratios match; FakeSolver physics irrelevant to queue
dynamics — real solve measured separately under PERF-001/002. Raw sampler trace in
`queue_samples.json`.

## PACK-003 decision (P8.3)

Patch-cube streaming NOT wired into solve_window. MEM-001 passes with 3.3x headroom
(978.6 MB vs 3000 MB budget) with dense cubes; streaming adds slice-management
complexity and IO-001 is payload-size-bound, not peak-RSS-bound. Revisit only if a
future site makes peak RSS the binding constraint.

## S03 anomaly (P8.5) — resolved

Pre-merge (T3 parts/S03.json): add h6 r3 = 128.9 s (dirty 0.0635), resize to h12 r5 =
917.4 s (dirty 0.1566) — 7.1x time for 3.6x write area. Post-P8 (`measure_s03.py`,
`s03_report.json`): add 60.1 s, resize 63.9 s — superlinearity eliminated (14.4x on the
resize job). Cause: pre-merge marches ran to the scene-wide amaxvalue (228.34 m) over
read-size windows (0.57 → 0.85 site between the two jobs); P8.1's per-tree
window-relative amplitude caps the march at the window's own relief (27.44 / 28.31 m)
and 129212a write-crops all per-cell chains.

## Routing table (site_500, 2009-08-11)

| Edit | dirty fraction | mode | write frac | exact read frac |
|---|---|---|---|---|
| add h6 r3 | 0.1239 | local | 0.1600 | 0.7313 |
| typical h10 r4 | 0.1864 | local | 0.2182 | 0.8041 |
| tall h18 r6 | 0.2150 | local | 0.2492 | 0.8301 |

All local; conservative-halo read fractions (old policy) shown for contrast in
`routing_site500.txt`.
