# SOLWEIG-GPU incremental design tool — performance optimization handoff

> Written 2026-09-04 at `main` @ `b75ca6d` (== `feat/incremental-design-worker`, pushed to
> `origin/main`). This document is the single complete handoff for any future session that
> wants to continue, reproduce, or extend the CPU performance work. It contains: every
> measured benchmark with its conditions, the full phase-level breakdown of where job time
> goes, what was changed and why, what was verified and how, the levers that were REFUSED
> (with reasons), and the levers that REMAIN (ranked, with estimated impact, risk, and why
> they were not taken). If you only read one document before touching the solver, read this
> one plus `docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md` rows W1/W2/W4-verify.

---

## 1. Executive summary

Interactive single-tree-edit job on the real `site_500` tile (500×500, 2 m pixels, 24 time
steps), measured end-to-end through the HTTP API:

| Stage | Job duration | Conditions |
|---|---|---|
| Container (docker compose), pre-optimization | **~301 s** | 4 vCPU container tax on top; anatomy: 183.4 s time loop (24 steps) + 113.75 s veg-SVF replay |
| Native macOS, pre-wave baseline | **~71 s** | host processes, no container (W3 migration) |
| After W1 (time-prefix truncation etc.) | **41.5 / 42.2 s** | idle-ish host |
| After W2 (per-patch march windows) | **38.9 / 38.0 s** | idle-ish host |
| W4 independent reproduction | **37.5 / 37.8 s** | idle-ish host |
| After W2, under concurrent load | **47.0 / 47.7 / 47.0 s** | host load-avg ~14 (another `claude` process at 90 % CPU) |
| Same, full-day request (truncation off) | **74.5 s** | `time_indices` = all 24 → shows the truncation lever's value |

Cumulative: **301 s → ~38 s idle (7.9×)**, of which container removal (W3) is 4.2× and
in-solver work (W1+W2) is 1.88×. Scenario creation is **0.27–0.44 s** (instant baseline
materialization; was a full ~7-minute solve before the `baseline_results` cache).

All W1+W2 changes are **bitwise-identical** to pre-wave output (verified independently —
see §7). No tolerance was ever relaxed.

## 2. Environment and deployment topology (as of this document)

### Host

- Apple M1 Pro, 10 cores (8 performance + 2 efficiency), 16 GB RAM, macOS 26.5.2.
- Repo: `/Users/alansynn/Workspace/solweig`, venv `.venv` (python 3.14.7, GDAL 3.13.3 via
  brew + pip `GDAL==3.13.3` for `osgeo`).
- Solver runs CPU-only (torch CPU, numpy). Threads: default (W2's in-worktree 8-thread
  measurement is thread-count-specific — see §5.4).

### Live stack (host-native, NOT docker — containers are legacy/fallback)

```
API      : .venv python -m solweig_gpu.server   127.0.0.1:8001
Frontend : examples/incremental_design_tool/serve.py  127.0.0.1:8765 --api http://127.0.0.1:8001
Tunnel   : cloudflared quick tunnel → 8765 (URL changes on every restart; free/no-account)
```

API env (exact, for restart):

```
SOLWEIG_STATE_ROOT=$PWD/state
SOLWEIG_SITE_ID=site_500
SOLWEIG_CACHE_ROOT=$PWD/site-cache
SOLWEIG_SITE_DIR_SITE_500=$PWD/Input_subset/processed_inputs
SOLWEIG_SELECTED_DATE_STR_SITE_500=2009-08-11
SOLWEIG_REQUESTS_PER_MINUTE=120
SOLWEIG_HOST=127.0.0.1  SOLWEIG_PORT=8001
```

Launch pattern that survives session task-kills (background tasks were mass-killed twice
on 2026-09-04 — always use this, never `run_in_background`, for long-lived servers):

```bash
cd /Users/alansynn/Workspace/solweig
nohup env SOLWEIG_STATE_ROOT="$PWD/state" SOLWEIG_SITE_ID=site_500 \
  SOLWEIG_CACHE_ROOT="$PWD/site-cache" \
  SOLWEIG_SITE_DIR_SITE_500="$PWD/Input_subset/processed_inputs" \
  SOLWEIG_SELECTED_DATE_STR_SITE_500=2009-08-11 SOLWEIG_REQUESTS_PER_MINUTE=120 \
  SOLWEIG_HOST=127.0.0.1 SOLWEIG_PORT=8001 \
  .venv/bin/python -m solweig_gpu.server > /tmp/solweig-api.log 2>&1 & disown
```

Logs: `/tmp/solweig-{api,fe,tunnel}.log`. A session watchdog cron checks
live/fe/tunnel/browser every 23 min.

### Data policy (inviolable)

`Input_rasters/`, `Input_subset/` are Dropbox-backed and gitignored. Never commit site
data, never bake into images. `site-cache/` and `state/` are likewise gitignored local
deployment artifacts.

## 3. Where the seconds go — job anatomy

A windowed tree-edit job's `metrics` (W1 made these honest; see §6.1):

| Metric | Meaning | Typical value (single-step request, idle) |
|---|---|---|
| `duration_ms` | total job wall inside worker | ~37,500–39,000 ms |
| `svf_seconds` | vegetation SVF replay phase (patch march loop) | ~19.8–20.3 s |
| `time_loop_seconds` | UTCI/Tmrt time loop, truncated to requested prefix | ~17.1–17.5 s (13 steps for `time_indices [12]`; full 24 steps ≈ 48 s) |
| remainder | compose/patch encode/IO | ~1 s |
| `read_window_fraction` | read (occluder-reach) window ÷ full tile | 0.98802 (247,005 / 250,000 px) |
| `window_fraction` | write (dirty) window ÷ full tile | 0.29568 (73,920 / 250,000 px) |

Both fractions were independently rederived from pixel counts by the W4 verifier — the
telemetry is not double-reporting the write fraction.

**Why `read_window_fraction` is 0.988 and cannot shrink (parity fence):** the read window
is the occluder-mask bbox from `read_window_for_write_window`
(`solweig_gpu/incremental/solver.py:666-752`): `max(aDSM, vegDSM, vegDSM2) − target_floor
> cheb_m·tan(6°)` with `LOWEST_SKY_PATCH_ALTITUDE_DEG = 6.0` (`solver.py:129`). For
site_500: occluder max 241.83 m vs target floor 185.42 m → reach 537 px > 500 px tile →
read fraction 0.998 of the write-extent expansion ends up ~99 % of the tile for a 30 %
write window. **Shrinking the reach or the 6° floor changes the physics and is forbidden.**

Time-loop linearity: ~1.7–2.0 s per time step (500×500 window solve per hour). The
truncation lever (W1) exploits `requested_result.time_indices` being an encode-time slice:
computing only the causal prefix `0..max(idx)` is bitwise-safe because every step is
independent in the emitted planes.

### Container tax (why native)

Measured job-level tax 4–5.5× (301 s container vs 71 s native / 61.3 s in-job). Docker on
this host was also the source of the worst operational incidents (disk-full wedges,
healthcheck starvation). Docker containers still exist but are LEGACY — never
`docker compose up`.

## 4. Benchmark inventory — every measured run

### 4.1 Container era (2026-09-03, docker compose, site_500)

- Full scenario baseline solve: **~7 min** (per new scenario, before baseline cache).
- Single tree-edit job: **~301 s**. In-container phase anatomy: time loop 183.4 s (24
  steps — no truncation then), veg-SVF replay 113.75 s (153 patches; the 6° band is ~65 %
  of the phase).

### 4.2 Native pre-wave (2026-09-04 early, W3 migration)

- Single tree-edit: **71 s** (4.2× vs container).
- Scenario creation with `baseline_results` present: **0.44 s** (status `exact`, no job).

### 4.3 After W1 merge `ce76d79` (native, idle-ish)

Two consecutive jobs: `duration_ms` **41,514 / 42,246**; `svf_seconds` 24.5 / 24.5;
`time_loop_seconds` 17.1 / 17.2; `read_window_fraction` 0.98802; `window_fraction` 0.29568.

### 4.4 W2 in-worktree micro-benchmark (threads=8, real site cache copy)

- `svf_seconds`: **22.62 → 16.90 s** (1.34×).
- March cell-steps: **2.83e9 → 1.41e9** (halved).
- Default-thread in-process run at head measured svf **18.2 s** (W4 verifier) — the 16.90
  figure is thread-count-specific; direction and magnitude reproduce, exact number does not.

### 4.5 After W2 merge `838e420` (native, idle-ish)

Two jobs: `duration_ms` **38,854 / 37,991**; `svf_seconds` 20.3 / 20.0; `time_loop_seconds`
17.6 / 17.5.

### 4.6 W4 independent reproduction (same code, own protocol)

`duration_ms` **37,528 / 37,794**; `svf_seconds` 19.85 / 19.80; `time_loop_seconds` 17.12 /
17.46; wall 38.5 / 40.5 s.

### 4.7 Under concurrent host load (2026-09-04 late, this document's fresh run)

Host load-avg ~14 (an unrelated `claude` process at 90.8 % CPU, plus agent-desktop
processes). Same code as 4.5/4.6:

| Run | duration_ms | svf_s | tloop_s |
|---|---|---|---|
| single-step rep1 | 47,470 | 24.54 | 22.35 |
| single-step rep2 | 47,660 | 24.88 | 22.21 |
| single-step rep3 | 46,992 | 23.99 | 22.44 |
| full-day (24 idx) | 74,547 | 25.99 | 48.02 |

Reading: concurrency inflates everything ~25 %; the truncation lever is directly visible
(22 s @ 13 steps vs 48 s @ 24 steps). **Quote the idle numbers for capability claims and
always record host load with any new number.**

### 4.8 Scenario creation (baseline fast path)

`POST /scenarios` → **0.26–0.44 s**, `status: "exact"`, no job. Mechanism:
`<cache>/baseline_results/{utci,tmrt}.f32.npy` + `metadata.json` (schema_version 1,
variables, time_steps, source), produced by
`examples/incremental_design_tool/tools/export_baseline.py` from a completed scenario
result. Without it: one full-tile solve per scenario (~7 min native era, ~5+ min today).

### 4.9 Suite runtimes (for planning verification work)

- Full canonical pytest: **~17 min** (1072 passed / 5 skipped / 2 failed env — §7.2).
- Node frontend suite: 127/127, <1 s.
- Perf wave files alone: wave1+wave2 = 40 tests ≈ 47 s.

## 5. What was done, wave by wave

### 5.1 W1 — telemetry honesty, heartbeat, causal time-prefix truncation, trivial dedups

Merge `ce76d79` (commits `a2853eb` perf, `5901d3d` test, `c529eeb` docs, `c123402` scope
fix). Files: `solweig_gpu/server/{jobs.py, executor_bridge.py}`,
`solweig_gpu/incremental/{solver.py, worker.py, executor.py}`, `solweig_gpu/utci_process.py`,
`tests/test_incremental_perf_wave1.py` (22 tests), `docs/.../api_contract.md`.

- **Honest metrics** (additive; `window_fraction` unchanged): `read_window_fraction`,
  `svf_seconds`, `time_loop_seconds` on worker+bootstrap+bridge paths. Sites:
  `resolve_time_stop`/`union_window_area_fraction`/`stage_timing_metrics` in `jobs.py`;
  solver `solve_window(..., stage_timings=...)`; `solver.py:1532/1601`; bridge
  `executor_bridge.py:815-833`; write fraction `geometry.py:516`.
- **Heartbeat ticker** (`server/jobs.py:612`, daemon thread, 5 s): heartbeats flow during
  gated solves. Before: `/health/ready` starved to 503 during any 300 s+ solve (container
  `failingStreak=39` incidents). After: mid-job probes return 200 (W4 probed 6×).
- **Causal time-prefix truncation**: `resolve_time_stop` (`jobs.py:352`, called at
  `jobs.py:1174` and `executor_bridge.py:591`) truncates the time loop to
  `0..max(requested time_indices)+1`. Plumbed as `solve_window(time_stop=...)` /
  `run(time_stop=...)`; patches self-describe truncated coverage. Deliberately NOT
  truncated (documented deviations): legacy `compute_utci` full-tile path (no time seam;
  superset patches are composition-safe), forced-full/fallback branches, building
  regeneration chain, and PlanExecutor dispatch (`_run_worker`) — the node store records
  full-series coverage per publication and an existing contract test asserts exactly
  that; partial coverage would need read-side reconciliation the seam lacks.
- **Trivial dedups**: compose/read-window derived exactly once per solve; pre-sleep
  `_supersede_scan` (supersedes stale queued jobs before sleeping); `cube_window`
  pre-cropped SVF bundle (`window_svf_bundle(cube_window=)` in solver;
  `svf_bundle_cubes_cropped` in `utci_process.py`, refuses without `out_window`) — saves
  the per-solve shadowmat crop (~153 MB slice per solve at write extent).

### 5.2 W2 — per-patch march windows + hoisted sky-patch geometry

Merge `838e420` (commits `4304c3a` perf, `166e0b6` test). **Only
`solweig_gpu/incremental/solver.py`** (+339/−82) + new test file (18 tests).

- `_patch_march_reach_pixels`: per-ring step bound `ceil(amp·scale/tan(alt))+1`
  (zenith→1). `_quadrant_read_direction`: march read axes row=−sign(cosθ),
  col=+sign(sinθ) (1e-12 trig tolerance) — note the axis semantics: in `shadow()`, dx
  (from −sign(cos)) is the ROW shift, dy the COL shift.
- `_patch_march_window`: per-patch march window = accumulate window expanded by the ring
  reach in READ directions only, clamped to the read window. Accumulators/cubes allocated
  at accumulate extent; per-patch march runs on an `a[grid]` sub-window (cached per
  (ring, quadrant) — one reach per ring). Bush-gate fallback to full-window marches.
- Veg scalars zero-padded back to read extent so the 19-tuple bundle contract is unchanged
  (`run_utci_window` crops to write at entry — padding never consumed).
- `_sky_patch_geometry` + process-level `_SKY_PATCH_GEOMETRY` memo: `create_patches(2)`,
  iazimuth fill, `aziintervalaniso`, per-ring (iso, aniso) annulus-weight pairs — computed
  once per process instead of per solve (annulus-weight computations dropped 3660 → 180
  per solve).

### 5.3 W3 — native migration (ops lever, not code)

State copied out of the container (`state/`, 671 MB: `scenarios/` + `store.sqlite3`),
API+serve.py+tunnel as host processes. Removed the 4–5.5× container tax and the container
healthcheck/starvation failure mode entirely.

### 5.4 Baseline instant materialization (product-level lever)

`tools/export_baseline.py` + `baseline_results/` in the site cache (§4.8). The
`state/site_500` cache also carries `metfiles/` (cloned from
`Input_subset/processed_inputs/metfiles`) — the server needs them co-located or via
`SOLWEIG_SITE_DIR_<ID>`.

## 6. The 8-agent adjudication that scoped W1/W2 (why these levers)

A multi-agent workflow (5 analysts + 2 measurers + 1 adjudicator, schema'd findings;
journal in the session `subagents/workflows` dir, run `w78l8fw6q`) produced the roadmap:

- **Step 1 (trivial)** — honest metrics + heartbeat ticker. → done in W1.
- **Step 2 (small, bit-identical)** — time-prefix truncation (~84 s container-scale), dedups,
  shadowmat slice, supersede-before-sleep. → done in W1.
- **Step 3 (medium, bitwise-gated)** — per-patch march windows (~17–23 s) + invariant hoist
  (~25–35 s container-scale). → done in W2.
- **Ops** — go native (4–5.5×). → done in W3.
- **Endgame (not taken)** — delta-SVF reuse (~115–135 s container-scale). See §9.2.
- **Rejected outright** — numba/JIT/import-time/thread-env levers (≤5 s each, not worth
  dependency/determinism risk); read-window shrink (**parity-breaking, forbidden**).

## 7. Verification evidence (W4, independent non-author)

Full report in `SUBAGENT_LEDGER.md` row `W4-verify`. Essentials:

### 7.1 Bitwise parity (the inviolable gate)

Cross-wave differential, own harness, real `site-cache/site_500` read-only, identical edit
on both sides, direct solver level, throwaway worktrees at base `3bd8aba` (pre-wave) vs
head `838e420`:

- Full-solve `run_full_tile` (24,500,500): utci `7d002c29f84e173c`, tmrt
  `65ef43ae86a0f7a5`, shadow `6c62bcea66d8c625` — base==head, `np.array_equal(equal_nan=True)`
  TRUE for all three.
- Windowed full-series (24,264,280): tmrt `b31ce077a8230df8`, utci `1c7a386b0173ebdf`,
  shadow `38c0874934017928` — base==head TRUE.
- `time_stop=13` prefix bitwise == full-series steps 0..12 (truncation-only change).
- HTTP-served patch of live jobs decodes byte-identical to direct solver step-12 planes;
  rep-to-rep served hashes equal. (Server, worker, direct solver = one arithmetic.)

### 7.2 Suites

- Full canonical at `838e420`: **1072 passed / 5 skipped / 2 failed**; the 2 =
  `tests/test_cli.py::test_cli_help` + `test_cli_missing_required_args`,
  `FileNotFoundError: 'thermal_comfort'` — environmental (CLI entry point not on PATH when
  suite runs without `.venv/bin` on PATH; passes 2/2 with
  `PATH="$PWD/.venv/bin:$PATH"`; identical failures at base `3bd8aba` → pre-existing).
- Node: 127/127. Perf wave files: 22 + 18 tests, both RED-first (W1: 14F/3P pre-impl;
  W2: 18F pre-impl).

### 7.3 Findings

Zero above INFO. INFO items: W2's ledger phrase "served-plane hashes" = solver-level
outputs (clarified); default patch carries utci+tmrt only (shadow must be requested);
svf 16.90 s figure is threads=8-specific (default-thread 18.2 s).

## 8. Parity fences — what is NOT optimizable without breaking science

1. **Read-window geometry** (`read_window_for_write_window`, `solver.py:666-752`) and
   **`LOWEST_SKY_PATCH_ALTITUDE_DEG = 6.0`** (`solver.py:129`). Physics reach. Forbidden.
2. **Float accumulation order.** Per-cell march accumulation must stay in the same order —
   any re-batching across patches or reordered sums breaks bitwise identity. This is why
   cross-patch march batching was refused.
3. **Full-series coverage contracts.** PlanExecutor node store and full-tile/legacy paths
   publish full-series coverage; truncating them needs read-side reconciliation that does
   not exist. Do not "finish" this without designing that seam.
4. **`shadow.py` march arithmetic** is the scientific core inherited from reference
   SOLWEIG; W2 touched only call extents around it, never its math.
5. **In-scope refusal list (documented in W2 report):** per-window tighter `amaxvalue`
   (out of adjudicated two-lever scope), cross-patch batching (float determinism),
   occluder-bbox windows (skirts the read-window fence), numba/JIT, thread-env vars
   (≤5 s each, adjudicated not-worth).

## 9. Remaining optimization opportunities (ranked, NOT taken)

### 9.1 `shadow()` per-step kernel-launch overhead — the biggest in-process lever

~40 % of `svf_seconds` (≈8–10 s at idle) is per-step call overhead: 153 patches × up to
274 march steps each = up to ~42 k small numpy call batches. Only refactoring `shadow()`
internals (fusing per-step numpy ops, hoisting slices, batching steps with identical
shapes) can cut it. **Risk: HIGH** — it is the scientific core; every refactor must
re-prove bitwise identity per cell. This was explicitly out of W2's adjudicated scope.
Suggested protocol if taken: W2-style throwaway-worktree A/B + corner-quadrant oracles +
real-site hash comparison (§7.1 pattern).

Secondary observation: on this workload the (+,+) quadrant's march windows still reach
~186 k of 201 k cells (write + 274-cell reach ≥ read extent in both axes), and the
dominant 6° ring's 274-step reach absorbs most window margins — i.e. **per-patch windowing
is near its ceiling for site_500**; denser-vegetation sites would benefit more.

### 9.2 Delta-SVF reuse for repeat edits ("endgame", adjudicated but not scheduled)

Second edit in the SAME scenario with unchanged vegetation outside the new influence
region could reuse the previous solve's veg-SVF cubes and recompute only the delta region.
Estimated container-scale saving was ~115–135 s (≈ most of `svf_seconds` on repeat edits).
**Risk: cache-invalidation semantics + bitwise gate** (reused cubes must be provably
bitwise-identical to what a fresh replay would produce for untouched cells). Big design
task; needs its own wave + non-author verification.

### 9.3 GPU port of the march (strategic)

The repo is SOLWEIG-**GPU** — the GPU pipeline exists for the batch path. Porting the
windowed march to torch CUDA would attack both `svf_seconds` and `time_loop_seconds`.
Float determinism on GPU differs from CPU → the bitwise gate vs CPU output cannot hold by
construction; would need a new tolerance-based gate (a policy change, not an engineering
one — the current gates are inviolable by design).

### 9.4 Transfer/UX (not solver, but user-perceived latency)

- Browser `DecompressionStream("zstd")` is 0 % global (Node 26.8.1 throws; Chrome ≤154
  no; Firefox flag-gated) → patches travel identity-encoded. Server already negotiates
  correctly. When browsers ship zstd, payload bytes drop ~2× with zero code change.
- Quick-tunnel RTT ~0.5–1 s; first-connect badge can lag ~90 s on payload download while
  the server-side connect sequence is already complete (log-verified). Not a server issue.

### 9.5 Explicitly dead ends (do not revisit without new evidence)

- Thread-count/OMP env tuning, import-time work, numba JIT: each ≤5 s container-scale.
- Read-window shrink / 6° floor raise: parity-breaking, forbidden.
- Container deployment on this host: 4–5.5× tax + operational wedges; native is strictly
  better here.

## 10. Reproduction protocols

### 10.1 Live HTTP benchmark (all numbers in §4.3–4.7 this way)

```python
# .venv/bin/python; API on 127.0.0.1:8001
import json, time, urllib.request, uuid
B = "http://127.0.0.1:8001/api/v1"
def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(B+path, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if body is not None: r.add_header("Idempotency-Key", str(uuid.uuid4()))
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read() or b"{}")
scn = req("POST", "/scenarios", {"site_id": "site_500"})     # ~0.3 s, status exact
edit = req("POST", f"/scenarios/{scn['scenario_id']}/edits", {
    "base_scene_version": scn["scene_version"],
    "edits": [{"operation": "add", "tree": {"tree_id": "bm-1", "component_type": "broad_canopy",
        "u": 0.575, "v": 0.345, "height_m": 18.0, "canopy_diameter_m": 11.0,
        "trunk_ratio": 0.25, "transmissivity": 0.03, "phenology": "deciduous"}}],
    "requested_result": {"time_indices": [12]}})              # job_id in response
# poll GET /api/v1/jobs/{job_id} every 3 s until status "complete" (NOT "succeeded")
# metrics: duration_ms / svf_seconds / time_loop_seconds / read_window_fraction / window_fraction
```

Full-day variant: `"time_indices": list(range(24))`.

Gotchas: job status string is `"complete"`; edits require `base_scene_version` and the
nested `tree` object (flat `{kind, op}` shapes 400); rate limit is 120 req/min
(`429 Too Many Requests` — include polls in your budget, sleep ~60 s when tripped);
ALWAYS record host load-avg with the result.

### 10.2 Cross-HEAD bitwise differential (W4 pattern)

Throwaway worktrees at base and head; direct solver level against the real cache
(READ-ONLY); identical edit both sides; `np.array_equal(a, b, equal_nan=True)` on
utci/tmrt/shadow for full-solve + windowed paths; additionally `time_stop` prefix vs
full-series prefix equality. Harness reference: W4's `/tmp/w4_differential.py` (ephemeral —
recreate from §7.1; in-suite equivalents: `test_incremental_perf_wave2.py` cross-HEAD
oracle, `test_incremental_worker.py` differentials).

### 10.3 Suites

```bash
cd /Users/alansynn/Workspace/solweig
.venv/bin/python -m pytest tests/ -q            # ~17 min; expect 1072P/5S/2F-env
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest tests/test_cli.py -q   # 2/2 pass
cd examples/incremental_design_tool && node --test tests/*.test.mjs           # 127/127
```

## 11. Artifact map

| Artifact | Where |
|---|---|
| Commits (this effort) | `a2853eb..c123402` (W1), `4304c3a`,`166e0b6` (W2), merges `ce76d79`,`838e420`, ledger `b381675`,`008912e`,`b75ca6d` — all on `origin/main` |
| Perf wave tests | `tests/test_incremental_perf_wave1.py` (22), `tests/test_incremental_perf_wave2.py` (18) |
| Subagent ledger rows | `docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md`: W1, W2, W4-verify (appendix: U-DEP, DEPLOY-HOTFIX rows for the deployment work) |
| Worklog | `docs/incremental_design_tool/agent/WORKLOG.md`: W1/W2/W4 entries (2026-09-04) |
| 8-agent adjudication journal | session dir `subagents/workflows/` run `w78l8fw6q` (ephemeral session storage; roadmap reproduced in §6) |
| W2 bench artifacts | `/tmp/w2_bench/{before,after}` (ephemeral) |
| Baseline export tool | `examples/incremental_design_tool/tools/export_baseline.py` |
| Live logs | `/tmp/solweig-{api,fe,tunnel}.log` |
| Site cache / state | `site-cache/site_500/` (incl. `baseline_results/`), `state/` — gitignored, local only |

## 12. Operational gotchas learned the hard way (2026-09-04)

1. **Background task mass-kills**: harness background tasks were killed twice (all three
   servers died with them). Long-lived servers → `nohup ... & disown` in a FOREGROUND
   Bash call (§2). Watchdog cron carries the full nohup recovery recipe.
2. **Quick tunnel URL churn**: every cloudflared restart = new URL; old URLs 530. Any
   automation holding a URL must be recreated on tunnel replacement (watchdog cron does
   self-recreate).
3. **API rate limit** `SOLWEIG_REQUESTS_PER_MINUTE=120` includes polls → benchmark
   scripts trip 429 easily; poll at ≥3 s and back off 60 s.
4. **`test_cli` env failures** look like regressions but are PATH artifacts (§7.2).
5. **Orca snapshot without `--page`** can hit a stale/empty tab — pin
   `--page <browserPageId>` from `orca tab list`.
6. **Host load skews benchmarks ~25 %** — always capture `uptime` alongside numbers.
7. `.venv` was destroyed once by external disk cleanup mid-session; rebuild recipe in
   WORKLOG (pip install -e ".[test,server]" + `GDAL==3.13.3` matching brew).
