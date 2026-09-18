# R0 — realtime-collaboration wave 0: reproduction, parity, cartography

Date: 2026-09-04. HEAD at measurement: `f3ad111` (main). Five parallel non-author agents
(perf reproduction, suite parity, queue/state cartography, universal adapter cartography,
frontend collaboration audit). This report is the R0 gate record; raw artifacts sit beside
it in this directory and in the SUBAGENT_LEDGER R0 row.

## 1. Performance reproduction (agent r0-perf, protocol = handoff §10.1 verbatim)

Fresh server per handoff §2 nohup recipe; `/health/live` + `/health/ready` 200 first probe.
All samples tagged **loaded**: host load1 8.55 → 11.49 during runs (unrelated processes,
left untouched per protocol).

| Run | duration_ms | svf_s | tloop_s |
|---|---:|---:|---:|
| single-step [12] rep1 | 39,987.4 | 20.85 | 18.61 |
| single-step [12] rep2 | 39,462.0 | 21.12 | 17.85 |
| full-day range(24) | 59,586.4 | 20.26 | 38.82 |

- `read_window_fraction` 0.98802 and `window_fraction` 0.29568: bit-exact on all runs;
  dirty window rows 36-300, cols 148-428 every time.
- Scenario creation: 0.354 / 0.303 / 0.297 s, status `exact` (n=3).
- Phase remainder duration−(svf+tloop) = 0.49-0.52 s (handoff said ~1 s).

### Verdicts vs the 2026-09-04 perf handoff

| Handoff claim | Verdict |
|---|---|
| §4.5/§4.6 idle 37.5-38.9 s, svf 19.8-20.3, tloop 17.1-17.6 | **CONFIRMED load-adjusted** — reproduction is +2.9-6.5% at load ~9-11, exactly interpolating the handoff's own idle→load-14 model |
| §4.7 loaded 47.0-47.7 s @ load ~14 | CONSISTENT (not directly comparable; lower load here) |
| §4.7 full-day 74.5 s @ load 14 | CONSISTENT — 59.6 s @ load ~10-11; truncation lever visible in both (tloop full/single 2.08 here vs 2.15 handoff) |
| §4.8 scenario creation 0.26-0.44 s | CONFIRMED in band |
| §4.3 (W1-only 41.5 s) | superseded by W2 at HEAD by design; svf correctly ~21 not 24.5 |

No value needed correction; nothing failed to reproduce. Zero errors/fallbacks/429s in
`/tmp/solweig-api.log`; all jobs mode `local`.

**Planning values adopted for the mission** (idle reference, load-adjusted band):
single-step geometry edit ≈ 38-40 s; svf phase ≈ 20-21 s; tloop ≈ 17-19 s;
full-day ≈ 60-75 s; scenario creation ≈ 0.3-0.44 s.

## 2. Suite and parity (agent r0-parity)

- `git diff 838e420..HEAD --stat`: **zero changes under `solweig_gpu/`** — compute source
  untouched since the bitwise-parity point; deltas are docs + CI removal only.
  Deviation noted for the ledger: commit 6045035 (labeled docs(realtime)) also added
  `tests/test_realtime_collaboration_spec.py` (6 additive YAML/doc-contract tests, all
  passing). Benign; flagged so the parity fence record stays honest.
- Full canonical suite: **2 failed / 1078 passed / 5 skipped** (1085.7 s). 1078 = 1072
  W4 baseline + 6 new spec tests. The 2 failures are exactly the sanctioned
  `test_cli` PATH artifact; with `PATH="$PWD/.venv/bin:$PATH"` they pass 2/2.
- Node frontend suite: **127/127** (196 ms).
- Oracle spot-check (perf wave1 + wave2 + worker): **95 passed / 0 failed** — all bitwise
  cross-HEAD oracles green.
- Versions: python 3.14.7, torch 2.14.0 CPU, numba 0.67.0, numpy 2.5.2, pyyaml 6.0.3.

## 3. Cartography conclusions (agents r0-queue, r0-universal, r0-frontend)

Full registers live in the ledger R0 row summaries; decisions they forced:

1. **R1 is one store-centric wave.** The `scene_version` equality assumption is
   load-bearing at store.py:1012/1107/1513, routes_scenarios.py:202-221,
   jobs.py:766/882-901, executor_bridge.py:424-432 — multi-writer acceptance plus the
   workspace/fast/exact revision triple must re-key every guard in the same wave.
2. **R2 lanes**: scheduler thread (ticker pattern, jobs.py:617-635); exact lane moves to
   a ProcessPoolExecutor(max_workers=1) (store-mediated guards are process-agnostic);
   fast lane in-process dedicated thread. SQLite: per-thread WAL read connections,
   single-writer epoch batching, retention off the hot path.
3. **Reducer seed** = executor_bridge.py:597-628 ledger replay (already deterministic,
   sequence-ordered).
4. **Broadcast** = extend the existing SSE `/events` surface with canonical-revision and
   result-class events; frontend gains a subscription module (it has zero EventSource
   usage today; its transport is 300 ms polling).
5. **R3 ranking** (r0-universal): (a) met-variables-at-timestep UTCI-only fast path,
   (b) receptor-only model params (Fside/Fup/Fcyl/cyl/height), (c) `selected_date_time`
   adapter (currently a typed NOT_INTEGRATED refusal; needs solar-geometry unpinning),
   (d) radiation view-layer caching. Fences kept: wbgt NEVER_PLANNED, landcover water
   class 7, wind-coefficient sites, dynamic_wind.
6. **Frontend blast radius** for per-operation epochs: ~70% of exact_session tests
   (If-Match/scene-version accounting, one-POST-per-gesture, 409 recovery) — rewrite is
   failing-first, not incidental.

## 4. R0 exit criterion

"A fresh report confirms or corrects every value used for planning" — met: every handoff
value either confirmed or consistent-with-load; suite green at HEAD; the four design-input
maps delivered with file:line evidence. R0 CLOSED.
