# Interaction simulations — cognitive-first UX specification

Status: paper prototype, pre-implementation. This document specifies EVERY user-facing
interaction of the redesigned SOLWEIG Studio realtime collaborative design studio before any
redesign code exists. It is the cognitive and interaction contract that the visual design
(light theme, color, typography — separate deliverable) and the implementation waves must
satisfy. Where behavior already exists in the current build, the simulation cites the real
DOM ids, event names, and modules so an implementer can diff live behavior against this
spec. Where behavior is new (interactive submission through the realtime operation plane),
the simulation is the definition of record.

Two bars govern every decision below:

- **Undergraduate bar.** A student who has taken one environmental-science course must be
  able to use every screen and explain every state in their own words.
- **Pro bar.** A thermal-comfort researcher must find the full disclosure surface — revision
  anchors, result provenance, qualifier evidence, per-stage compute classification — without
  the undergraduate ever being forced to look at it. Never dumbed down; layered instead.

Sources this spec is built on (identifiers quoted verbatim):

- `examples/incremental_design_tool/index.html` — current DOM (ids cited as `#id`)
- `examples/incremental_design_tool/app.mjs` — interaction layer (symbols: `commitDesignEdit`,
  `scheduleAnalysis`, `handleSessionStatus`, `startCollabTelemetry`, …)
- `examples/incremental_design_tool/realtime_client.mjs` — `RealtimeClient` public API:
  `submitOperations`, `subscribeWorkspace`, `revisionsFor`, `classFor`, `subscriptionState`,
  `RealtimeAdmissionError`, `onStall`
- `examples/incremental_design_tool/realtime_badge.mjs` — `realtimeBadgeState`,
  `CLASS_LABELS`, `collabSubscriptionAlive`
- `examples/incremental_design_tool/README.md` — capability-driven UI, acceptance gates
  UI-001…UI-005, fast-lane consumption contract
- `solweig_gpu/server/routes_realtime.py` — operation plane API and error envelopes
- `docs/incremental_design_tool/realtime_collaboration/service_level_contract.md` — SLOs,
  admission envelope, degradation ladder
- `docs/incremental_design_tool/realtime_collaboration/collaborative_state.md` — operation
  grammar, conflict semantics, revision model

Adoption stance. The current build submits edits through the exact-session plane
(`POST /api/v1/scenarios/{id}/edits`) and uses the realtime plane display-only
(`app.mjs` "Collaborative-plane telemetry (r2c, read-only)"). This spec makes the realtime
operation plane the PRIMARY submission path for the redesigned studio. Simulations are
tagged **[LIVE]** (exists today; re-specified only where cognition demands a change) or
**[NEW]** (does not exist; this document is its definition). The existing acceptance gates
carry over to the new plane unchanged in intent: exactly one committed operation per gesture
(UI-001), debounced sliders (UI-002), stale results ignored (UI-003), failures preserve
design state (UI-004), model scope disclosed (UI-005).

---

## 1. Cognitive model

The studio is teachable if the user holds ten concepts. Every screen and simulation below
is built to introduce, reinforce, or verify exactly these — nothing else needs to be taught.

### 1.1 The ten concepts

| # | Concept | One-sentence plain definition | What the user SEES that teaches it | Misconception it prevents |
|---|---------|-------------------------------|------------------------------------|---------------------------|
| 1 | **Scene** | The design being studied: trees, land cover, buildings, weather inputs, and analysis settings, all together. | The map itself (`#glCanvas` + `#overlayCanvas`) with the scene context chip (`#sceneTimeLabel`). | That "the map" and "my design" are different things; the map IS the scene. |
| 2 | **Edit** | One committed change to one kind of thing in the scene — add a tree, paint ground, set an air temperature. | Every edit gesture ends with one row in the Activity ledger and one toast line (`#analysisToast`). | That the app saves continuously and vaguely; no, discrete edits are the unit of record. |
| 3 | **Edit family** | The KIND of input an edit touches (vegetation, land cover, meteorology, time, model parameters, view); the family decides how expensive the analysis will be. | The family cards in "Universal editors" (`#editFamiliesRoot`) and the family tag on every ledger row. | That all edits cost the same; a view selection is free, a geometry edit is expensive. |
| 4 | **Revision triple** | Three counters of how current the shared scene is: `workspace_revision` (edits), `fast_revision` (quick analysis), `exact_revision` (verified physics). They only ever satisfy exact ≤ fast ≤ workspace. | The compact `W·F·E` readout and the `R` number in the result-class badge (`#realtimePill` today; promoted in this redesign). | That there is one version number; there are three, and the gaps between them ARE the pending work. |
| 5 | **Result class** | How much to trust the picture on screen right now: `fast_exact` (real physics, fast path), `fast_qualified` (physics with a stated error bar), `visual_pending` (placeholder, exact owed), `exact_reconciled` (verified). | The result-class badge on the scene header + the preview-marker treatment on the map. | That the thermal map is always equally true; the badge is the truth label of the pixels. |
| 6 | **Epoch** | The server collects everyone's edits for ~100 ms, orders them, and publishes one merged step; nobody ever waits for a lock. | The `R` number ticking up once per batch in the badge; ledger rows sharing one `R` when edits landed in the same epoch. | That collaboration needs turn-taking or that simultaneous edits conflict dangerously. |
| 7 | **Operation** | The permanent, numbered receipt for one edit (`operation_id`, `server_sequence`); the same edit re-sent is recognized, not duplicated. | Ledger rows carry seq numbers; the outbox shows `sending → acked seq 41 → applied R15`. | "Did my edit land twice?" — no; idempotency is visible, not hidden. |
| 8 | **Admission** | The server publishes how much work it can do per second; if you exceed it, it says "not yet" BEFORE accepting, so nothing accepted is ever silently dropped. | The held-edit state in the outbox and the admission notice with a countdown (`retry_after_ms`). | That 429/503 mean "broken" or "lost"; they mean "held, and here is when". |
| 9 | **Dirty region** | The rectangle of the map that actually needs recomputing after an edit — the rest is reused as-is. | The dashed outline on the map (`#roiLayerToggle`) and the window caption (`#roiCaption`: "35,480 / 250,000 cells"). | That the whole map recomputes every time, or that results outside the outline are wrong. |
| 10 | **Result basis (supersession)** | Every picture belongs to ONE revision; when the scene moves past that revision, the picture is marked stale until replaced — it is never quietly wrong. | The `(stale)` badge state and the superseded note in the tooltip ("fast result for R15 superseded by R16"). | That an old result silently overwrites a newer one, or that "stale" means "corrupt". |

### 1.2 The five-stage pipeline — the spine of the whole UI

Every committed edit travels the same five stages. The UI names them with the same five
words everywhere (badge, toast, ledger, worker-steps panel). This is the single most
important teaching device in the redesign: one vocabulary from gesture to verified physics.

| Stage | Word | What happens | Typical time (inside SLO) | Where the user sees it |
|-------|------|--------------|---------------------------|------------------------|
| 0 | **Gesture** | Drag, paint, slide, type — the user acts. | instant | Map / editor cards |
| 1 | **Sketch** | The browser's local preview kernel repaints the affected region — non-scientific, immediate. | ~0.1–0.6 s | Preview delta layer at reduced opacity + preview marker (`state.previewStale`, `renderer.updatePreviewHeatmap`) |
| 2 | **Stamp** | The edit is submitted as an operation; the server stamps it with `server_sequence` and `epoch_id`. | p95 ≤ 50 ms, p99 ≤ 150 ms | Outbox row flips `sending → acked seq N` |
| 3 | **Share** | The epoch closes; the server merges, publishes `canonical_revision`; everyone's `R` ticks. | ~100–550 ms (one to two epochs, tick-anchored) | Badge `R` increments; ledger row `applied · R15` |
| 4 | **Paint** | The fast lane publishes a deadline-bound result (`result_class`), replacing the sketch. | p99 ≤ 1,000 ms from acceptance | Badge class changes; preview layer gives way to the fast plane |
| 5 | **Verify** | The exact SOLWEIG lane reconciles the fast result (`exact_revision` catches up). | By family: ≤ 100 ms (views) … asynchronous (geometry) | Badge settles to `exact`; map region patched by the exact plane |

The job-steps list (`#jobSteps`) is re-labeled to these five stages (today it shows four
exact-session steps: coalesce → dirty window → recompute → patch). The old four steps
remain as the sub-display of the Verify stage for the exact lane.

### 1.3 Undergrad glossary — the words the UI copy MUST reuse

UI copy never invents synonyms. One concept, one word, everywhere.

| UI word | Undergrad sentence (usable verbatim in tooltips) | Pro addendum (disclosure surfaces) |
|---------|--------------------------------------------------|------------------------------------|
| **Scene** | "Everything you are designing: trees, ground, weather, and settings, in one map." | Scene = canonical workspace state at `workspace_revision`. |
| **Edit** | "One change you committed — not a save, not a draft." | One operation (or one idempotent batch of 1–128). |
| **Family** | "The kind of input you changed. Cheap families answer in under a second; geometry families are verified later." | `source_family`; drives the dependency-class exact SLO. |
| **Epoch** | "The server merges everyone's edits into shared steps about ten times a second." | W = 100 ms nominal (configurable 50–200 ms); one `workspace_revision` increment per epoch. |
| **Revision / R** | "How many shared steps the scene has taken. Higher = newer." | `workspace_revision`; invariant exact ≤ fast ≤ workspace. |
| **Sketch** | "The instant browser-only repaint. A sketch is never a scientific result." | Local preview kernel; drawn at reduced opacity with a marker while pending. |
| **Fast exact** | "Real physics, answered in about a second, from the incremental fast lane." | `result_class: fast_exact`; authoritative for its `targetRevision` until superseded. |
| **Qualified estimate** | "Physics with a stated margin of error, shown while the exact answer is still computing. Open the badge for the error evidence." | `result_class: fast_qualified` + qualifier disclosure: error evidence, coverage, limitations, `exact_base_revision`. |
| **Preview (owed)** | "A placeholder on top of the last verified picture. The exact answer is still owed." | `result_class: visual_pending`; last exact field marked stale; exact lane may be asynchronous. |
| **Exact** | "Verified SOLWEIG physics for the revision you are looking at." | `exact_reconciled`: `exact_revision ≥ workspace_revision`. |
| **Stale** | "This picture describes an older scene. It is still honest — it says which revision it describes." | Superseded: `lastCanonicalRevision > targetRevision`; never rendered as authoritative. |
| **Held** | "Your edit is safe locally but the server has not accepted it yet — it will go when capacity returns." | Admission rejection (429/503) BEFORE durable append; `retry_after_ms` honored. |
| **Receipt / seq** | "The server's numbered receipt for your edit. Quote it and anyone can find that exact change." | `operation_id` + `server_sequence` + `epoch_id`. |
| **Dirty region** | "The rectangle being refreshed. Outside it, the previous result is reused — still valid, not forgotten." | ROI + halo; window fraction in `#roiCaption`. |

Words the UI must NOT use: "sync" (ambiguous between share/paint/verify), "save" (implies
drafts), "conflict" as a user-facing alarm (the server orders edits; divergence is advisory),
"error" for anything expected (held, stale, catching up are states, not errors).

---

## 2. Screen map

### 2.1 Screen list

| Screen | Job (one line) | Dominant elements |
|--------|----------------|-------------------|
| S0 Boot / loading | Load the site fixture and capability document; promise that interactivity is seconds away. | brand, spinner, `#capabilityBadge` = loading… |
| S1 Offline preview studio | Full single-user design experience with the local sketch kernel; clearly labeled non-scientific. | whole shell; `#connectionPill` state `local`; 3 starter trees |
| S2 Connecting | Show the retry ladder honestly; keep the workspace usable while connecting. | `#connectionPill` `connecting`, retry counter, map stays editable |
| S3 Connected · baseline | First verified picture; establish trust vocabulary before any editing. | `exactBadge` settles; `#realtimePill` appears; zero trees |
| S4 Designing (steady state) | The default working screen; five-stage pipeline visible at a glance. | badge `R·class`, sketch layer, dirty outline, Activity drawer |
| S5 Exact pending | Sustained `visual_pending` while the exact lane works; the user must never wonder whether the app forgot. | badge "Preview (owed)" + age counter; verify-stage sub-steps |
| S6 Degraded / backpressure | Ladder rungs and held edits, stated as capacity facts with countdowns, never as failure. | badge "Qualified estimate" + qualifier panel; `#heldCount` chip; admission notice |
| S7 Transport recovery | Stream is reconnecting or catching up; local editing continues. | `#transportPill` `reconnecting`/`catching_up`; drawer gap marker |
| S8 Failure | Something genuinely failed; state what survived, offer the one recovery action. | `#failureBanner`, `#siteMismatchBanner`, `#realtimePill` `failed` |
| S9 Collaboration | Other people are present and their edits arrive as ordered, attributed receipts. | presence bar, remote pulses, ledger rows with actor tags |
| S10 Inspect (Layers tab) | Read published layers without queuing any solver work. | `#viewsTab` controls, `#viewResult`, `#viewZeroBadge` |

### 2.2 Wireframes

**W-1 — The shell (S1/S3/S4/S5/S9 share this frame; only the marked cells change state).**
Existing ids in parentheses; NEW ids marked ➕.

```
+------------------------------------------------------------------------------------------------+
| (brand)            SCENARIO [Campus shade study v]   [W15 F15 E9 | Fast exact]  [A][M]+2  Reset  |
|  SOLWEIG Studio                                      ^ result badge        ^ presence   Export   |
|                                                     |  ➕#resultClassBadge |  ➕#presenceBar        |
|  (connectionPill) exact server · (workerPill) sketch |  ➕#transportPill Live|  ➕#heldCount (hidden)  |
+-------------------+----------------------------------------------------------------------------+
| DESIGN | LAYERS   |  [select][pan]        [Compare baseline (#compareButton)]                   |
| (tabs) |         | +--------------------------------------------------------------------+   |
|-------------------| |                                                                    |   |
| CANOPY LIBRARY    | |                        MAP  (#glCanvas)                             |   |
|  (#componentList) | |              sketch / fast / exact plane  +  overlay                 |   |
|  drag a card …    | |                        (#overlayCanvas)                            |   |
|                   | |   [12:00 · Summer design day  #sceneTimeLabel]  (north) (scale)    |   |
| UNIVERSAL EDITORS | |   [! failureBanner]  [! siteMismatchBanner]                        |   |
|  (#editFamiliesRoot) | |   [▶ analysisToast: Paint · fast exact · 0.6 s]                  |   |
|  (#capabilityBadge)| |                                                                    |   |
|                   | +--------------------------------------------------------------------+   |
| VISIBLE LAYERS    |  TIMELINE (#timeSlider 08…18)   ➕#timelineFreshness o o ● o o          |
|  heat/shadow/ROI  |  [Comfortable ▓▓▓▓▓▓▓ Extreme heat]  legend 30–44°C UTCI                |
|  (3 checkboxes)   |  [Activity ▾ #activityToggle …………………… ➕#activityList (drawer)]        |
+-------------------+----------------------------------------------------------------------------+
| RIGHT PANEL: LIVE ANALYSIS                                                     (collapses on |
|  metrics (#meanDeltaMetric #peakDeltaMetric #areaMetric #roiMetric) v(#sceneVersion) | small     |
|  SELECTION (#selectedTreeName #selectionProperties #deleteTreeButton)                 | viewports)|
|  PIPELINE (#jobSteps → Sketch Stamp Share Paint Verify)  (#jobStatusDetail)           |           |
|  DEPENDENCY IMPACT (#impactPanel changed/recomputed/reused)                           |           |
|  MODEL SCOPE (#exactBadge #modelScopeList)  [Run again (#rerunButton)]                |           |
+------------------------------------------------------------------------------------------------+
```

Region jobs:

- **Topbar** — identity (who/what scenario), trust (result badge), company (presence), and
  the two global verbs (Reset, Export). Transport health is a separate small pill
  (`➕#transportPill`) deliberately distinct from the result badge (see decision D3).
- **Left panel** — the editing vocabulary: what can be added (library), what can be edited
  (family cards, generated from `GET /api/v1/capabilities`), what is drawn (layers).
- **Map** — the scene and its truth label. Every trust signal has a home ON or BESIDE the
  pixels it describes, never buried in a panel.
- **Timeline** — the time lens plus per-timestep freshness (➕, see I-05).
- **Activity drawer** — the ledger: one row per operation, yours and others'. Collapsed by
  default to a single-line summary; one click expands (see §6).
- **Right panel** — numbers (metrics), the selected object, the pipeline stages, impact
  classification, and the scientific scope disclosure (UI-005).

**W-2 — S0 Boot / loading.**

```
+--------------------------------------------------------------+
|  (brand) SOLWEIG Studio          …loading site fixture       |
|                                                                |
|              [ spinner ]  Loading campus-1km-v1                |
|                                                                |
|   · Site imagery …………………… done                                 |
|   · Capability document …… done  (5 editors, 3 layers)         |
|   · Sketch kernel ……………… warming up                            |
|                                                                |
|   Offline preview mode — no ?api= URL parameter.               |
|   Connect an exact server to enable verified physics.          |
+--------------------------------------------------------------+
```

Job: make the wait legible (named steps, not a bare spinner) and teach the mode split
(offline vs connected) before a single pixel of analysis appears.

**W-3 — S2 Connecting (retry ladder visible, workspace stays usable).**

```
+--------------------------------------------------------------+
| (connectionPill: connecting) Exact server unreachable          |
|                                 · retry 2 of 3 (4 s backoff)   |
|  MAP IS FULLY EDITABLE — edits are HELD and replay in order    |
|  ➕#heldCount: 2 held edits (sketches shown)                    |
+--------------------------------------------------------------+
```

Job: convert "it's broken" into "it's retrying with a plan"; held edits are the proof that
nothing is lost (existing behavior: held backlog replays on the attempt that lands).

**W-4 — S5/S6 detail: badge + held states on the scene header.**

```
 S5 exact pending                      S6 degraded (qualified)
+--------------------------------+    +------------------------------------+
| W15 F14 E9 | Preview (owed) 42s |    | W18 F16 E9 | Qualified estimate ~   |
| exact owed since R9 · verify:   |    | ±0.6°C · coverage 87% · open ▸      |
| [geometry: 3/7 steps]           |    | exact owed since R9 · 2 edits held  |
+--------------------------------+    +------------------------------------+
```

Job: the badge states the class, the revision gap, and the age; the qualifier chip (`~`)
opens the disclosure panel (§5). Age counter starts quiet (no number under 10 s) so normal
fast-lane latency never reads as delay.

**W-5 — Activity drawer expanded (ledger row anatomy).**

```
| ACTIVITY ………………………………………… live · R18 · 3 actors |
|--------------------------------------------------|
| you    seq 41  vegetation · add    Broad canopy 03  [Sketch ✓ Stamp ✓ Share ✓ Paint ✓ Verify …] |
| Maya   seq 42  landcover · paint   210×140 cells     [Sketch ✓ Stamp ✓ Share ✓ Paint ✓ Verify …] |
| you    seq 43  meteorology · set   air_temp 14:00    [Sketch ✓ Stamp ✓ Share ✓ Paint ✓ Verify …] |
| you    —     admission · held      landcover · paint  [Held — retry in 1.2 s]                    |
|--------------------------------------------------|
|  every row: actor · seq · family · verb · entity · five-stage dots |
```

Job: one row per operation, each with the five-stage progress dots; held rows show the
countdown. Clicking a row opens the receipt (operation_id, epoch, timestamps, payload
summary). This single surface teaches concepts 2, 6, 7, and 8 simultaneously.

**W-6 — S7 transport recovery.**

```
+--------------------------------------------------------------+
| ➕#transportPill: Reconnecting · attempt 2 of 3                |
|  Live updates paused. Your map shows the last shared state     |
|  (R18) and your edits are queued. Nothing is lost.            |
|  [gap marker in ledger: ⋯ catching up ⋯]                       |
+--------------------------------------------------------------+
```

**W-7 — S8 failure.**

```
+--------------------------------------------------------------+
| (!) Exact analysis failed (kernel_panic): worker exited.       |
|     Your design is preserved — 12 trees, 3 held nothing.       |
|     [Retry exact analysis (#retryButton)]                     |
+--------------------------------------------------------------+
```

Job: failure copy states WHAT failed, WHAT survived, and the ONE action (existing pattern
from UI-004; copy rules in §4).

---

## 3. Interaction inventory + simulations

### 3.0 Inventory

| Id | Interaction | Status |
|----|-------------|--------|
| I-01 | Open the studio offline (boot) | [LIVE] |
| I-02 | Connect to the server (`?api=`) full handshake | [LIVE] + [NEW] subscribe step |
| I-03 | Read the UTCI map, legend, and metrics | [LIVE], metric inheritance [NEW] |
| I-04 | Scrub time offline | [LIVE] |
| I-05 | Scrub time connected (personal lens + refinement + freshness) | [LIVE] + freshness [NEW] |
| I-06 | Toggle layers (heat / shadows / dirty region) | [LIVE] |
| I-07 | Compare against baseline | [LIVE] |
| I-08 | Inspect a published layer (Layers tab) | [LIVE] |
| I-09 | Add a tree by drag & drop | [LIVE], submit [NEW] |
| I-10 | Add a tree by button / keyboard | [LIVE] |
| I-11 | Move a tree by direct manipulation (incl. cancel) | [LIVE], cancel semantics [NEW] |
| I-12 | Nudge the selected tree with arrow keys | [LIVE] |
| I-13 | Delete a tree | [LIVE] |
| I-14 | Edit object properties (debounce + inline validation) | [LIVE] |
| I-15 | Paint land cover (window addressing) | [LIVE] |
| I-16 | Edit meteorology (time-indexed scalar) | [LIVE] |
| I-17 | Edit model parameters | [LIVE] |
| I-18 | Commit a family with no transport (typed refusal) | [LIVE] |
| I-19 | Submit an edit through the realtime operation plane (five-stage walkthrough) | [NEW] — the core new capability |
| I-20 | Watch the result class step down the degradation ladder | [NEW] |
| I-21 | Exact supersession flip (stale → verify) | [NEW] |
| I-22 | Duplicate submission / idempotent replay / reused id | [NEW] |
| I-23 | Stale `base_revision` advisory (`base_divergent`) | [NEW] |
| I-24 | Admission rejection 429 (workspace burst) | [NEW] |
| I-25 | Admission rejection 503 (server saturated) | [NEW] |
| I-26 | Ack stall (10 s) + receipt lookup | [NEW] |
| I-27 | SSE drop → reconnect → catch-up | [LIVE client] / [NEW surface] |
| I-28 | Heartbeat gap (`missed_events`) recovery | [LIVE client] / [NEW surface] |
| I-29 | Offline ↔ connected transitions | [LIVE] |
| I-30 | Presence: collaborator join / leave | [NEW, derived] |
| I-31 | A collaborator's operation arrives | [NEW surface] |
| I-32 | Two people edit the same entity (conflict intuition) | [NEW] |
| I-33 | Exact analysis failure + retry | [LIVE] |
| I-34 | Site identity mismatch | [LIVE] |
| I-35 | Typed refusals: `realtime_disabled`, `rate_limited`, `operation_not_found` | [LIVE server] / [NEW surface] |

### 3.1 Viewing and inspection

---

#### I-01 Open the studio offline [LIVE]

**(a) Trigger.** Navigate to the studio URL without `?api=`.

**(b) Simulated sequence.**

1. Browser loads `index.html`; `app.mjs` fetches `./assets/baseline.json` and
   `./assets/site_base.webp`. Screen shows W-2 (S0): named loading steps.
2. Capability UI builds from the bundled snapshot `assets/capabilities_snapshot.json` —
   `#editFamiliesRoot` fills with the family cards; `#capabilityBadge` shows the document
   version. (Same code path as connected mode: the snapshot is a verbatim dump.)
3. The sketch kernel worker signals `ready`; `scheduleAnalysis` runs the full-scene first
   pass; `#analysisToast` settles to "Analysis synchronized".
4. `#connectionPill` shows `data-state="local"`, text "Local preview mode". `#exactBadge`
   shows "Preview". `#realtimePill` stays `hidden`.
5. Offline mode keeps the three starter trees; the map shows the sketch UTCI overlay, tree
   glyphs, and shadows at the default hour 12.

**(c) Edges.** Fixture fetch fails → full-screen typed failure with the failing step named;
no half-loaded shell. Snapshot unusable → no editors are shown and the studio says so
(the same rule as connected mode: never fall back to a second document).

**(d) Cognitive payoff.** The user learns the mode split (offline = sketch only), sees the
word "Preview" attached to the badge from second zero, and meets the family cards as the
editing vocabulary before touching anything.

---

#### I-02 Connect to the exact + realtime server [LIVE + NEW]

**(a) Trigger.** Open `?api=/api&site=campus-1km-v1` (same-origin proxy) or click a
"Connect" affordance added in the redesign.

**(b) Simulated sequence.**

1. `resolveApiBase` de-doubles the prefix; `#connectionPill` → `connecting`, badge text
   "Connecting to exact server". The workspace remains editable; committed gestures while
   connecting are HELD (existing held-backlog behavior) and the `➕#heldCount` chip appears.
2. `ExactSession.connect` runs: `GET /api/v1/capabilities` (falling back to `/capabilities`
   alias), scenario resolution, baseline result. If the site cache has
   `baseline_results/`, the baseline is instant (`status: "exact"`, no job); otherwise the
   first full solve runs and the boot screen names it ("Verify · full-tile baseline solve,
   this can take minutes" — honest by SLO: exact geometry is asynchronous).
3. Baseline lands → `handleBaselineResult`: `state.exact` cube loaded; `refreshExactTexture`
   uploads the displayed-hour plane; `#exactBadge` settles from "Preview" to "Exact" state;
   `updateExactMetrics` fills the metric cards; connected mode starts with ZERO trees (the
   server scene is authoritative) — the empty map with a "Drag a tree from the library"
   hint is intentional and labeled as such.
4. **[NEW]** On session `ready`, `startCollabTelemetry` subscribes the realtime plane:
   `RealtimeClient.subscribeWorkspace(scenarioId)` opens
   `GET /api/v1/workspaces/{id}/events`. First SSE frame arrives: event
   `canonical_revision` with `{snapshot: true, operations: [], workspace_revision: R0,
   fast_revision: F0, exact_revision: E0}` — the badge renders `R0 ·` + class; before any
   class-bearing frame the pill shows the awaiting state ("Collab · live"), per
   `realtimeBadgeState`.
5. `➕#transportPill` shows "Live"; `➕#resultClassBadge` (scene header, promoted from the
   topbar pill) shows the triple `W·F·E` and the class word.

**(c) Edges.** Transient connect failure → retry 1 s/2 s/4 s (three attempts) with the
counter in the pill (W-3); held edits replay in order on the attempt that lands. Permanent
4xx → surfaced immediately, held backlog released with each edit marked in the ledger.
Capability document unavailable → typed `capability_document_unavailable`: NO editors are
shown, stated as such (never fall back to the offline snapshot). Subscribe TOCTOU is
server-corrected by a second `snapshot: true` frame; the badge just ticks up, no flicker.

**(d) Cognitive payoff.** Connection is a five-step visible handshake, not a spinner; the
user sees the triple appear and understands "server scene wins, zero trees" instead of
wondering where their demo trees went.

---

#### I-03 Read the UTCI map, legend, and metrics [LIVE + NEW]

**(a) Trigger.** Looking at the map at rest; hovering the legend or metric cards.

**(b) Simulated sequence.**

1. The map shows the active plane at the current class (sketch overlay / fast plane /
   exact plane). The legend (30°C → 44°C UTCI, "Comfortable" → "Extreme heat") is the
   reading key; hovering a legend band highlights cells in that band on the map.
2. Hovering the map (crosshair) shows a data probe: UTCI value, cell coordinates, active
   layer, and the badge's class + revision copied into the probe ("41.2°C UTCI · Qualified
   estimate · R18") — the probe NEVER shows a bare number without its truth label.
3. Metric cards (`#meanDeltaMetric`, `#peakDeltaMetric`, `#areaMetric`, `#roiMetric`)
   inherit the current class marker (a small echo of the badge chip, not a second full
   badge — one trust home, echoes elsewhere). `#meanDeltaCaption` states the window
   ("window rows 210–352, cols 118–306") so no number is read without its scope.
4. `#sceneVersion` shows `v` = workspace revision, keeping the old card meaningful as the
   `R` number.

**(c) Edges.** While a sketch is showing, the metric cards dim to their last verified
values with an "at R15" caption instead of showing sketch-derived numbers (sketches are
never quoted as data — they are visual only). Compare mode (I-07) swaps the map to baseline
and blanks live metrics with "baseline shown".

**(d) Cognitive payoff.** "Every number wears its truth label and its scope." This is the
core anti-misreading rule the whole result-class design (§5) rests on.

---

#### I-04 Scrub time offline [LIVE]

**(a) Trigger.** Drag `#timeSlider` (8–18) in offline mode.

**(b) Simulated sequence.**

1. `input` fires continuously; `#timeValue` and `#sceneTimeLabel` update instantly;
   shadows re-render from the local kernel each frame (`render()` with `state.hour`).
2. After 350 ms of quiet, `scheduleAnalysis` recomputes the sketch for the settled hour;
   `#analysisToast` shows "Updating thermal analysis · Solar time changed to 15:00".

**(c) Edges.** Rapid scrubbing coalesces to the final hour (debounce); no per-tick jobs
(overload rule: never one job per mouse move).

**(d) Cognitive payoff.** Time is a lens you aim, and the analysis follows the STOP, not
the drag.

---

#### I-05 Scrub time connected — personal lens + refinement + freshness [LIVE + NEW]

**(a) Trigger.** Drag `#timeSlider` while connected.

**(b) Simulated sequence.**

1. Instant: the displayed exact plane swaps locally from the published cube
   (`refreshExactTexture`) — no server round-trip, because the baseline cube carries every
   cached timestep. Shadows and tree glyphs update per frame.
2. **[NEW] `➕#timelineFreshness`:** each timestep tick under the slider carries a
   freshness dot derived from the revision triple:
   - filled = exact plane for the current workspace revision,
   - half = cached exact plane from an older revision (stale-but-honest),
   - hollow = never computed.
   After an edit at 14:00, the 14:00 dot is half-filled: the map will show the OLD scene's
   physics for that hour until refinement lands — the user is told BEFORE scrubbing there.
3. After 350 ms of quiet, the refinement request rides the view family (per
   `collaborative_state.md` §View state: workspace-wide requested-output changes may
   schedule a missing scientific product but do not modify physical sources): a
   `view.select` / requested-output operation for `time_indices: [hour]`. `state.previewStale`
   flips true; the sketch layer covers the affected window; `#jobSteps` shows Share →
   Paint → Verify progress.
4. The fast frame for the refinement publishes; the timestep dot fills; the sketch layer
   clears.

**(c) Edges.** Scrubbing during exact-pending does not cancel the owed refinement — the
newest requested timestep wins (exact lane jumps to the latest eligible revision).
Scrubbing is PERSONAL: other collaborators' displayed hours are never changed by your
scrub (per-user view state), and this is stated in the timeline tooltip ("your time lens —
shared with no one").

**(d) Cognitive payoff.** "Looking at another hour is free and personal; refining a hour
for the design is a queued product request." The freshness dots convert an invisible
staleness hazard into a two-second glance.

---

#### I-06 Toggle layers [LIVE]

**(a) Trigger.** `#heatLayerToggle`, `#shadowLayerToggle`, `#roiLayerToggle`.

**(b) Simulated sequence.** Each `change` flips its flag and re-renders; pure view state,
no operations, no jobs, nothing broadcast. The dirty-region row's caption ("CPU recompute
window") is re-worded in the redesign to "Refresh window — the area being recomputed".

**(c) Edges.** Hiding the dirty region never hides the badge or sketch marker — trust
signals are not dismissible by layer toggles.

**(d) Cognitive payoff.** Layers are a personal lens (concept 10 reinforcement); the truth
label outlives every cosmetic choice.

---

#### I-07 Compare against baseline [LIVE]

**(a) Trigger.** Click `#compareButton` (aria-pressed toggles).

**(b) Simulated sequence.**

1. `state.compareBaseline = true`; render swaps the map to the baseline plane at reduced
   heat opacity (0.48 vs 0.64), hides trees/selection/ghost/sketch, and the button stays
   pressed. Copy near the button: "Baseline — the scene before any edit in this scenario".
2. Click again → live plane returns; the badge resumes the current class.

**(c) Edges.** Edits during compare are refused with a one-line reason ("Leave compare mode
to edit") rather than silently queued. (Pointer input is already blocked in compare mode.)

**(d) Cognitive payoff.** Comparison is a deliberate before/after reading mode, clearly
bounded, never confused with the live proposal.

---

#### I-08 Inspect a published layer (Layers tab) [LIVE]

**(a) Trigger.** Layers tab → choose layer + operation (+ optional compare layer, time
index) → `#viewRunButton`.

**(b) Simulated sequence.**

1. `POST /api/v1/scenarios/{id}/views` with the composed request.
2. `#viewZeroBadge` quotes the SERVER's flags (`zero_scientific_jobs` / `job_enqueued`)
   verbatim — "0 jobs" only when the server says so.
3. `#viewResult` renders metadata (dtype, shape, nodata), legend statistics, or cached
   time-step chips. The copy line `#viewBasisCopy` states the rule: selecting among cached
   layers never queues a solver job.

**(c) Edges.** Typed refusal `view_not_available` keeps its `published_layers` list in the
result card ("Available layers: …"). Cached-time mismatch lists `cached_time_indices`.

**(d) Cognitive payoff.** The studio clearly separates READING published science (free)
from WRITING the scene (costed) — concepts 2/3 reinforced.

---

### 3.2 Designing (local edits)

All designing simulations share the five-stage spine (§1.2). Stages 1–2 are identical in
shape for every family; what differs is the Verify timing by dependency class (view ≤ 100 ms;
receptor/UTCI-only stretch 250 ms; selected-hour forcing stretch 1 s; local land cover
stretch 2 s; vegetation/building geometry asynchronous; DEM full-only asynchronous).

**Debounce contract (unchanged, UI-002):** property inputs re-sketch after 150 ms of quiet
and COMMIT after a 480 ms trailing window (spec window 400–600 ms); the final value wins;
the pre-debounce tree state is kept so the old influence region is never lost; a drag that
lands during a pending debounce rebases it (`propertyEditor.rebase`) so the late commit
cannot snap the tree back.

---

#### I-09 Add a tree by drag & drop [LIVE + NEW submit]

**(a) Trigger.** `dragstart` on a `#componentList` card (`data-preset` = shade/broad/evergreen),
drag over the map, `drop`.

**(b) Simulated sequence.**

1. `dragstart`: `state.dragPreset` set; `#sceneCard` gains `.is-dragging`; the card's
   aria-label already teaches the keyboard alternative ("Press Enter to add at the scene
   center, or drag onto the map").
2. `dragover` on `#sceneCard`: a ghost tree previews under the cursor
   (`state.ghostTree`); outside the site bounds (`isUvInsideSite`) the ghost disappears —
   the drop is simply impossible there, no error needed. The `#dropHint` overlay ("Drop
   tree on site") shows during the drag via CSS (`.scene-card.is-dragging .drop-hint`); in
   the redesign it appears only for the first drag of a session (teach-once) and thereafter
   only when the cursor is outside the droppable site bounds.
3. `drop` inside the site: `addTree(preset, u, v)` → tree pushed, selected,
   `syncSelectionPanel` fills `#selectionProperties`.
4. Sketch: `state.previewStale = true`; `scheduleAnalysis(rect, "Broad canopy 03 added")`
   — the local kernel repaints the crown's influence region in ~0.6 s at reduced opacity
   with the preview marker; the dirty-region outline shows the window.
5. **[NEW] Stamp:** `submitOperations(workspaceId, [{sourceFamily: "vegetation",
   entityId: tree.id, verb: "add", payload: {height_m: 18, trunk_zone_ratio: …, position …}}])`
   → one ledger row appears in state `sending`, flips to `acked seq 41` when the POST
   returns (p95 ≤ 50 ms).
6. **[NEW] Share:** next epoch close → SSE `canonical_revision {workspace_revision: 15,
   operations: [op 41 …]}` → the row flips to `applied · R15`; the submit promise resolves
   `{operationId, serverSequence: 41, epochId, duplicate: false}`.
7. **[NEW] Paint/Verify:** geometry is an asynchronous Verify class — the fast frame
   typically carries `result_class: "visual_pending"` (badge "Preview (owed)", sketch layer
   stays); the exact lane later publishes `exact_revision 15` → region patch applied
   (`renderer.updateHeatRegion`), sketch cleared, badge "Exact".

**(c) Edges.** Drop outside the site → nothing happens (ghost already said so). Drop during
compare mode → refused (I-07). Held (I-24/I-25) → row stays `Held` with countdown, sketch
remains. Duplicate retry (network) → same `operation_id` replays; ack carries
`duplicate: true`; ledger row does not duplicate.

**(d) Cognitive payoff.** One gesture = one operation = one receipt, and the five stages
are WATCHABLE end to end. The sketch answers "did it work?" instantly; the receipt answers
"is it permanent?"; the badge answers "is it physics yet?"

---

#### I-10 Add a tree by button / keyboard [LIVE]

**(a) Trigger.** Click an `Add` button (`[data-add-preset]`), or focus a component card
and press Enter/Space.

**(b) Simulated sequence.** Same as I-09 steps 3–7 with placement at the scene focus with
a small ordinal jitter (`addTreeAtSceneCenter`). Successive keyboard adds spread out so
trees never stack invisibly.

**(c) Edges.** None new — the accessibility path is a first-class citizen, not a fallback;
its commits are indistinguishable from drag commits downstream.

**(d) Cognitive payoff.** "The map is reachable without a mouse" — and the pipeline
vocabulary is identical, so nothing new must be learned.

---

#### I-11 Move a tree by direct manipulation [LIVE + NEW cancel semantics]

**(a) Trigger.** `pointerdown` on a tree glyph in `#overlayCanvas`, drag, `pointerup`.

**(b) Simulated sequence.**

1. `pointerdown`: hit test (`findTreeAtCanvasPoint`, 33 px tolerance) selects the tree
   (`#selectedTreeName` updates), stores `oldTree`, captures the pointer.
2. `pointermove`: tree follows the cursor live (clamped inside the site with a 2 % margin);
   shadows re-render per frame — pure local sketch, zero network.
3. `pointerup` with a changed position: `commitDesignEdit(old, new, "Broad canopy 03
   moved")` → operation `verb: "move"` (a replacement of position fields, not a side
   effect), then the five stages as in I-09. `propertyEditor.rebase` keeps any pending
   debounce on the new position.

**(c) Edges — redesign decision D7 (cancel semantics).** Today `pointercancel` runs the
same commit path (`finishPointerDrag` commits if the position changed). The redesign
specifies: `pointercancel` and pressing Escape during a drag RESTORE `oldTree` and commit
NOTHING — cancel must mean cancel. (The window into this rule: OS interruptions like
notifications must not spend the user's edit budget.)

**(d) Cognitive payoff.** Direct manipulation feels instant because stages 1 is local; the
commit is one clean operation at the END of the gesture (UI-001), and cancel is honest.

---

#### I-12 Nudge the selected tree with arrow keys [LIVE]

**(a) Trigger.** Focus `#overlayCanvas` with a selection; press arrows (Shift = 5× step).

**(b) Simulated sequence.** Each press moves the tree 0.004 UV (0.02 with Shift), renders,
and feeds `propertyEditor.input` — so nudges DEBOUNCE EXACTLY like slider edits: many
nudges, one commit after 480 ms of quiet. The pipeline chip on the selection header shows
"sketching… commit in 0.5 s" while the window is open.

**(c) Edges.** Held-nudge-then-drag rebases as in I-11.

**(d) Cognitive payoff.** Fine positioning is cheap and coalesced; the user sees the commit
countdown and stops fearing "one job per keypress".

---

#### I-13 Delete a tree [LIVE]

**(a) Trigger.** `#deleteTreeButton` (or Delete key with a selection).

**(b) Simulated sequence.**

1. Tree removed locally; selection clears; sketch recomputes the union of the old
   influence region (the region it USED to cool must be restored to un-cooled physics —
   `dirtyRectForEdit(oldTree, null, …)`).
2. Operation `verb: "delete"` → stages as usual. The ledger row for the ADD of that tree
   stays visible (operations are permanent receipts); the delete row references the same
   `entity_id`.

**(c) Edges.** Delete of an already-deleted entity (remote raced you, I-32): the server's
tombstone rule (an old generation cannot be resurrected) means the reducer no-ops the
second delete but the epoch still records it; the ledger row resolves `applied · no source
change` — a fact, not an error.

**(d) Cognitive payoff.** Deleting is an edit like any other — receipt, epoch, recompute of
the FREED region — not a file-system deletion. Sets up "delete is undo for add" (§3.3).

---

#### I-14 Edit object properties (debounce + inline validation) [LIVE]

**(a) Trigger.** Drag a generated slider / type into a number field in
`#selectionProperties` (controls generated from the capability document with its bounds).

**(b) Simulated sequence.**

1. 150 ms quiet → sketch recomputes (tree glyph and thermal sketch move together).
2. 480 ms trailing quiet → ONE commit (`verb: "update"`), carrying only touched
   properties (`touchedProperties`), with `old_values` for move/update/delete validation.
3. Server 422 `invalid_tree_geometry` with a `field` → `handleValidation` flashes
   `.is-invalid` on the matching generated control (aliasing `canopy_diameter_m` →
   `canopy_radius_m`) for 2.2 s and the failure line quotes the server's field message;
   design state untouched.

**(c) Edges.** Out-of-bounds values are refused client-side by the schema bounds before
any send (the control clamps). A rejected edit leaves the sketch showing the REJECTED
value for at most 2.2 s, then snaps back to the last accepted value — the snap-back is the
confirmation the server is authoritative.

**(d) Cognitive payoff.** "Type freely; the commit is the pause; the server is the
referee." Field-level rejection teaches WHERE the rule lives without a modal in sight.

---

#### I-15 Paint land cover (window addressing) [LIVE]

**(a) Trigger.** `landcover_surface` family card → operation `paint` → fill the four
window bounds (row/col start/stop) and class value → "Validate & commit".

**(b) Simulated sequence.**

1. Client validation: all four bounds must be finite (`"paint window: fill all four
   row/col bounds"`), polygon JSON parses, `composeFamilyEdit` schema-checks the draft
   (`DraftValidationError` shows inline in the card's `errorLine` — nothing is sent).
2. Commit → one operation `{source_family: "landcover", verb: "paint", target:
   {row_start, row_stop, col_start, col_stop}, payload: {class…}}`.
3. The sketch previews the painted rectangle immediately (family-local, stretch ≤ 2 s
   exact class); the dirty region outlines the paint window; the ledger row summarizes
   "210×140 cells".
4. Redesign addition: a map-native paint affordance (brush on `#overlayCanvas` with the
   family card armed) composes EXACTLY this window operation on pointerup — the brush is a
   gesture, the window is the grammar.

**(c) Edges.** Overlapping strokes from two actors: chunked, ordered by `server_sequence`,
later writes win per cell unless the adapter declares a commutative merge — surfaced
quietly as two receipts and one merged canonical result (I-32). Strokes coalesce into one
epoch-final sparse mask (the reducer materializes one update; the audit keeps the strokes).

**(d) Cognitive payoff.** Raster edits are windowed, chunked, and merge by rule — the
undergrad sentence: "the last brush on a cell wins; everything else merges."

---

#### I-16 Edit meteorology (time-indexed scalar) [LIVE]

**(a) Trigger.** `meteorological_forcing` family card → operation `set` → field (e.g.
air_temperature), value, `time_index` → "Validate & commit".

**(b) Simulated sequence.**

1. Draft validated (units shown from the schema); one operation `{verb: "set", payload:
   {field, value, units}, time_index}`.
2. Selected-hour forcing with reusable geometry is a stretch-≤1 s exact class: the badge
   typically walks Preview → Fast exact within a second — the FASTEST full-class change
   the studio can show, and the demo moment for "the second-second physics" claim.
3. **[NEW]** The scene context chip (`31.6°C · RH 47% · wind 2.4 m/s`, static text in
   today's `index.html`) becomes live: it renders the canonical meteorology for the
   displayed timestep, annotated with the revision it adopted.

**(c) Edges.** Overlapping time ranges are normalized server-side into disjoint segments
before planning; the ledger row for a ranged set shows the normalized segments in its
receipt view. A `set` that disagrees with a collaborator's simultaneous `set` on the same
(field, timestep) resolves last-validated-write-wins (I-32) — quietly.

**(d) Cognitive payoff.** Weather is a table of per-hour values with per-(field, time)
write rules — precise enough for a pro, explainable as "last saved value per cell of the
table".

---

#### I-17 Edit model parameters [LIVE]

**(a) Trigger.** `model_receptor_parameters` family card → change a parameter → "Validate
& commit".

**(b) Simulated sequence.** Same as I-16 (scalar `set`, no time index). Receptor-only or
UTCI-only changes from cached Tmrt are a stretch-≤250 ms class — the badge may go Sketch →
Fast exact faster than the toast can animate; the toast therefore never animates class
changes under 300 ms (no flicker for speed).

**(c) Edges.** Rejected properties (e.g. `transmissivity` if fenced) render with the
schema's rejection reason INSTEAD of an editor (`#rejectedPropertiesNote`) — the document's
own text, never paraphrased (UEDIT-010 rule).

**(d) Cognitive payoff.** Parameters are just another family — one grammar, no special
cases to learn.

---

#### I-18 Commit a family with no transport (typed refusal) [LIVE]

**(a) Trigger.** "Validate & commit" on an integrated-but-unwired adapter.

**(b) Simulated sequence.** `composeFamilyEdit` throws `FamilyTransportError`; the card
shows the transport note and the EXACT payload JSON that will be sent once a transport
lands ("nothing was submitted" stated verbatim). No POST fires; no ledger row appears.

**(c) Edges.** The disabled-card pattern (non-integrated adapters) keeps the document's
status/disclosure text verbatim.

**(d) Cognitive payoff.** The studio never fakes success; absence of a capability is a
displayed fact with the payload ready — trust through visible refusal.

---

### 3.3 Undo, cancel, and withdraw semantics (definitional box)

There is no global undo in this redesign wave. The semantics that DO exist are stated so
users never hunt for Ctrl+Z:

| Gesture | Cancel/undo affordance | Boundary |
|---------|------------------------|----------|
| Slider / nudge debounce | Keep typing; only the trailing value commits | Cancelled by continuation |
| Map drag | Escape or `pointercancel` restores `oldTree`, commits nothing [NEW D7] | Before pointerup only |
| Held edit (admission) | **Withdraw** button on the held ledger row [NEW] | Local-only, BEFORE acceptance; after `acked` there is no withdraw |
| Add you regret | Delete it (the compensating operation; both receipts remain) | Permanent audit by design |
| Exact job in flight | No user cancel; supersession IS the system's cancellation (stale results are dropped by UI-003) | Server-owned |

Rationale taught in the UI footnote: "Every edit is a shared, numbered fact. You cannot
un-happen a fact to other people's screens — you add a compensating fact." (Tombstone rule
makes add→delete clean; resurrecting needs a NEW generation.)

---

### 3.4 The realtime operation plane — the new capability

---

#### I-19 Submit an edit through the realtime operation plane [NEW] — the core walkthrough

This is the same pipeline as I-09 steps 5–7, specified once in full as the definition of
record for interactive submission.

**(a) Trigger.** Any committed design gesture (I-09…I-18) in connected mode.

**(b) Simulated sequence — the five stages with wire detail.**

1. **Stamp — submit.** The client builds the batch: `POST /api/v1/workspaces/{id}/operations`
   with header `Idempotency-Key: {actor_id}:batch-{n}` and body
   `{actor_id, operations: [{operation_id: "{actor_id}:op-{client_sequence}",
   client_sequence, base_revision, source_family, entity_id, verb, payload}]}` (1–128
   operations; the UI submits ONE per gesture, UI-001).
   `base_revision` = last observed `workspace_revision` — advisory only; the client never
   gates on it.
2. **Stamp — ack.** 200 in p95 ≤ 50 ms: `{accepted: [{operation_id, server_sequence: 41,
   epoch_id: "e12", duplicate: false}], workspace_revision: 14, fast_revision: 14,
   exact_revision: 9, exact_base_revision: 9}`. Ledger row → `acked seq 41`. If
   `base_revision` ≠ `workspace_revision`, the ack carries `base_divergent: true` → I-23.
3. **Share — epoch close.** Within one to two epochs (~100–550 ms) SSE
   `canonical_revision {workspace_revision: 15, epoch_id: "e12", operations: [op 41],
   fast_revision: 14, exact_revision: 9}`. The client applies the operation exactly once
   (`appliedOperationIds` dedupe), resolves the submit promise
   `{operationId, serverSequence: 41, epochId: "e12", duplicate: false}`, badge `R` → 15,
   ledger row → `applied · R15`.
4. **Paint — fast lane.** ≤ 1 s from acceptance, SSE `fast_revision {fast_revision: 15,
   workspace_revision: 15, result_class, exact_base_revision, provenance}`. The fast
   plane replaces the sketch (sketch opacity 0.42 layer clears); the badge adopts the
   class word; the toast shows "Paint · {class} · {t}s".
5. **Verify — exact lane.** SSE `exact_revision {exact_revision: 15,
   workspace_revision: 15}` when the exact result reconciles: region patch applied via
   `renderer.updateHeatRegion` (sub-rectangle upload only), `exactBadge` settles, badge
   class → Exact, `#jobSteps`' Verify stage completes with the family's realized metrics
   (window fraction, mode, duration).

**(c) Edges.** Network failure mid-POST → identical idempotent body re-sent (1 s/2 s/4 s,
three retries) → I-22. Admission rejection → I-24/I-25 (never retried blind). SSE dead at
submit time → the ack fence still resolves via the catch-up GET (the promise does not
depend on the stream). Ack slower than 10 s → `onStall` → I-26.

**(d) Cognitive payoff.** The whole distributed-systems story — idempotency, ordering,
revision gaps, provisional results — is ONE five-step progress bar the user can watch,
with receipts at every step.

---

#### I-20 Watch the result class step down the degradation ladder [NEW]

**(a) Trigger.** The same cheap edit type (e.g. meteorology `set`) submitted while server
load rises from idle to saturated — or the demo button "Load: replay ladder" in a pro
debug drawer.

**(b) Simulated sequence.** Successive fast frames carry different `result_class` for the
same KIND of edit; the badge walks the ladder rungs in order:

1. Rung 1–2 `fast_exact` (from cache / bounded kernel): badge "Fast exact", tooltip
   "computed exactly for R18 by the incremental fast lane".
2. Rung 3 `fast_qualified` (validated response operator): badge "Qualified estimate ~"
   with the qualifier chip; opening the panel shows error evidence, coverage, limitations,
   `exact_base_revision: 14` ("built on the exact result for R14"). Metrics inherit the
   `~` echo and read e.g. "−1.1°C ± 0.6".
3. Rung 4 reduced-resolution `fast_qualified`: panel states "half resolution · coverage
   61% · redrawn cells outlined".
4. Rung 5 `visual_pending`: badge "Preview (owed)"; the sketch layer persists over the
   last exact field, explicitly marked stale; tooltip "exact owed since R14".
5. Rung 6 backpressure: no result class at all — the edit itself is HELD before acceptance
   (I-24/I-25). The ladder is presented as capacity facts with countdowns, never errors.

**(c) Edges.** Class can only step down while load persists and steps back up as capacity
returns — the badge animates upward transitions but not downward ones (downgrade is
logged in the ledger row's receipt; upgrading is the reassuring one to celebrate). An
unknown future class displays conservatively as Preview (`effectiveClass` rule) — never a
blank badge.

**(d) Cognitive payoff.** "The picture always tells me HOW it was made." The ladder
becomes a legible speed–trust dial rather than a hidden degradation, and the pro gets the
full rung provenance in the panel.

---

#### I-21 Exact supersession flip (stale → verify) [NEW]

**(a) Trigger.** A fast result for R15 is displayed; a newer canonical revision R16
publishes (the user's own next edit, or a collaborator's — I-31) before the fast lane
re-paints.

**(b) Simulated sequence.**

1. `canonical_revision {workspace_revision: 16}` arrives → the supersession fence trips
   (`lastCanonicalRevision 16 > targetRevision 15`) → `classFor().superseded = true` →
   badge appends "(stale)"; the map KEEPS the R15 fast plane (it is still the best
   picture) with the stale marker; nothing is blanked.
2. The next `fast_revision` frame for R16 (or the `exact_revision` frame, whichever
   first) replaces it; `(stale)` clears; if the exact frame covered R16 the class settles
   to Exact directly.
3. During the stale window the toast reads "Scene moved to R16 — refreshing the picture".

**(c) Edges.** Reconnect can MISS fast frames entirely (fast payloads are SSE-only, not
durable): a Preview with no follow-up fast frame is a legitimate terminal state until the
next epoch — the badge simply says "Preview (owed)" with the age counter; the client never
waits for a fast frame that may never come (contract rule).

**(d) Cognitive payoff.** "Stale" is a truth label with an expiry reason, not an error —
the picture names the exact revision it describes (concept 10) and the user never catches
the UI lying by omission.

---

#### I-22 Duplicate submission / idempotent replay / reused id [NEW]

**(a) Trigger.** (i) The POST times out client-side but the server processed it; the retry
re-sends the identical body. (ii) A user double-clicks a commit button. (iii) A bug or a
restored tab re-submits the same `operation_id` with DIFFERENT content.

**(b) Simulated sequence.**

1. (i) Retry lands: `append_operations` recognizes the fingerprint → ack
   `{duplicate: true, server_sequence: <original>}`; the ledger row does NOT duplicate
   (one `operation_id`, one row); the row's receipt notes "duplicate ack — original seq 41".
   The submit promise had already resolved (or resolves now) through the canonical fence;
   the canonical effect count is exactly one.
2. (ii) The second click within the same gesture is swallowed client-side (committing
   affordances disable during `sending`); even if it raced, (i) covers it.
3. (iii) Server: `409 operation_id_reused` (details carry the `operation_id`) → ledger row
   flips to `Refused — id reused`; copy per §4: the ORIGINAL operation is unchanged;
   nothing was double-applied. The client generates a fresh `operation_id` for the next
   gesture automatically.

**(c) Edges.** A 429 drawn by an idempotent RETRY of an already-durable batch reconciles
through the catch-up GET before any rejection is surfaced (r2b F7): if the operation
appears, the submit resolves through the canonical fence and the user sees only `applied ·
R15` — the near-miss is logged in the receipt, never alarmed.

**(d) Cognitive payoff.** "Re-sending is safe; changing your mind needs a new receipt."
Idempotency stops being folklore because the ledger shows duplicates resolving to one
effect.

---

#### I-23 Stale base_revision advisory (`base_divergent`) [NEW]

**(a) Trigger.** The user commits while others have advanced the workspace — their
`base_revision` (what they saw, R13) is older than the server's current
`workspace_revision` (R14) at accept time.

**(b) Simulated sequence.**

1. The ack body carries `base_divergent: true` (logged server-side; NEVER a rejection —
   the server's reducer owns ordering).
2. The ledger row shows a small advisory chip: "based on R13 — server merged it into R14".
   Clicking it: one sentence — "You edited an older view of the scene. Your edit still
   applied; the server ordered it after everyone else's. Nothing was overwritten."
3. If the merged result differs from what the user's sketch predicted (e.g. someone had
   moved the same tree), the canonical frame's reconciliation pulse (I-31) marks the
   entity; the selection panel shows the adopted canonical values.

**(c) Edges.** Advisory is NEVER escalated: no dialog, no blocking, no "resolve conflicts"
flow. The chip is the entire surface. (Contrast with the legacy exact plane's
`scene_version_conflict`, which reloads authoritative state — different plane, different
rule; the ledger labels which plane a row rode.)

**(d) Cognitive payoff.** The lock-free mental model in one chip: "no locks, no blocking —
the server sequences everyone." Fear of "clobbering" is replaced by a curiosity click.

---

#### I-24 Admission rejection — 429 workspace burst [NEW]

**(a) Trigger.** The workspace exceeds its published admission envelope (e.g. burst of
paint operations; `max_accepted_operations_per_second`).

**(b) Simulated sequence.**

1. `POST /operations` returns 429 BEFORE any durable append — nothing was accepted, so
   nothing can be lost. Body details: `retry_after_ms`, `advice[]`.
2. The gesture's ledger row flips to `Held — burst limit`; `➕#heldCount` appears in the
   topbar ("2 held"). The sketch ALREADY showed the user's intent (stage 1 precedes
   admission — local preview never waits for the server).
3. A countdown ticks from `retry_after_ms` (typically ~1–2 s); the row auto-resubmits the
   SAME `operation_id` when it reaches zero (idempotent, so even a slightly-early retry is
   safe). On ack, the row continues the normal five stages.
4. Copy (active, actionable): "Held: this workspace is editing faster than the analysis
   budget. Resubmitting in 1.2 s. Your edits are safe — keep working."

**(c) Edges.** Sustained bursting (counter keeps resetting): the notice grows a
recommendation line built from the server's `advice` (e.g. "paint larger strokes less
often — the envelope allows N ops/s"), and the commit affordances enter a soft cooldown
(shown as a thin progress bar on the button, never a disabled mystery). Withdraw stays
available (§3.3). A 429 on a RETRY first reconciles via catch-up (I-22 edge).

**(d) Cognitive payoff.** Flow control you can SEE: held ≠ lost, the countdown is a
promise, and the envelope is a published number the UI quotes rather than a wall it
bounces off silently.

---

#### I-25 Admission rejection — 503 server saturated [NEW]

**(a) Trigger.** `503 fast_lane_unavailable` — the server-wide fast lane cannot admit new
work (capacity saturated for everyone).

**(b) Simulated sequence.**

1. Ledger row → `Held — server at capacity`; `➕#heldCount` badge turns to the saturated
   tone. Copy: "Server busy: new analysis work is paused fleet-wide. Held edits resubmit
   when capacity returns (retry hint 30 s). Local sketches keep working."
2. Held rows accumulate in ORDER; they resubmit as one batch per gesture when the retry
   hint expires, preserving the user's sequence (each keeps its own `operation_id`).
3. While saturated the studio may also show ladder rung 5 (Preview owed) for already
   accepted work — the two surfaces are visually distinct: held = NOT accepted (topbar +
   row), Preview = accepted, result owed (badge).

**(c) Edges.** Saturation does NOT disable viewing, scrubbing cached timesteps, or the
Layers tab (view-only work stays available). If saturation outlasts a generous window
(> 5 min), the drawer offers "Keep editing offline-style; held edits will follow" — never
a forced logout or data drop.

**(d) Cognitive payoff.** "The server protects everyone's second; my work queues politely."
Global versus workspace-scoped causes are named differently — the first diagnostic
distinction a pro user actually needs.

---

#### I-26 Ack stall (10 s) + receipt lookup [NEW]

**(a) Trigger.** A submit's ack does not appear in any `canonical_revision` within
`REALTIME_STALL_TIMEOUT_MS` = 10 s (`onStall`).

**(b) Simulated sequence.**

1. The ledger row gains a quiet warning state: "No receipt yet — checking". The client
   performs the idempotent retry lookup `GET /api/v1/workspaces/{id}/operations/{operation_id}`.
2. Found → the row resolves with the receipt (seq, epoch) exactly as if the frame had
   arrived; the near-miss is logged.
3. `404 operation_not_found` → the original POST never reached durability; the client
   resubmits the identical `operation_id` (still idempotent) with visible state
   "resubmitting". Copy: "The server has no record of this edit yet — resending the same
   receipt id."

**(c) Edges.** The stall surface NEVER offers blind resubmission with a NEW id (that is
the one path to duplicate effects); idempotency makes the honest retry free. If the
subscription itself is reconnecting, stall checks queue behind catch-up (the fence may
resolve them first).

**(d) Cognitive payoff.** Even the pathological case has a named procedure: "check the
receipt, resend the same receipt." Trust is procedural, not hopeful.

---

### 3.5 Connectivity and collaboration

---

#### I-27 SSE drop → reconnect → catch-up [LIVE client / NEW surface]

**(a) Trigger.** The events stream errors (network drop, proxy idle timeout).

**(b) Simulated sequence.**

1. `#transportPill` → "Reconnecting · attempt 1 of 3" with backoff (1 s/2 s/4 s, three
   retries). W-6 copy: live updates paused, map shows last shared state (R18), edits queue.
2. The client opens the replacement stream FIRST (buffering), then
   `GET /api/v1/workspaces/{id}/operations?since_server_sequence={lastServerSequence}`
   replays every operation not yet APPLIED, sorted by `server_sequence`; operation_id
   dedupe makes re-carried operations harmless. Buffered frames then flush in order.
3. Pill → "Live"; the ledger shows a gap marker ("⋯ caught up 6 operations") and the badge
   `R` jumps to the present (R23). Remote operations that landed during the gap apply with
   their pulses compressed into one "3 edits from Maya, 2 from you" summary line.

**(c) Edges.** Catch-up GET itself fails transiently → back off and retry within the
budget; a NON-transient failure fails the subscription (`failed`); budget exhausted →
terminal "failed" state with a Reconnect button — never an infinite spinner. The first
frame of every fresh subscription is a `snapshot: true` `canonical_revision` (empty
operations — history belongs to the catch-up GET), and a TOCTOU-close epoch produces a
corrective snapshot: the badge only ever ticks UP.

**(d) Cognitive payoff.** "Reconnect is a catch-up with a receipt trail, not a reload."
Nothing re-downloads wholesale; nothing applies twice; the user can keep editing straight
through.

---

#### I-28 Heartbeat gap (`missed_events`) recovery [LIVE client / NEW surface]

**(a) Trigger.** The SSE hub drops a lagging subscriber's OLDEST frames and flags
`missed_events: N` on the next `heartbeat` — the stream itself stays open.

**(b) Simulated sequence.**

1. The pill flickers to "Catching up" (sub-second, no full reconnect): the gap fence
   buffers live frames, the same catch-up GET replays the dropped operations, buffered
   frames flush in order, pill returns to "Live".
2. The ledger gains a one-line gap marker (count of recovered operations). If the queue
   overflows AGAIN mid-recovery, the next heartbeat starts the next cycle — the user sees
   at most a flicker; correctness is never at stake.

**(c) Edges.** A failed recovery GET hands the whole recovery to the reconnect cycle
(I-27). Heartbeats with no `missed_events` are invisible except in the pro tooltip ("last
heartbeat 3 s ago") — quiescence must look like peace, not like death.

**(d) Cognitive payoff.** Silent self-healing with a paper trail: the system polices its
own liveness, and the proof is a ledger line, not user anxiety.

---

#### I-29 Offline ↔ connected transitions [LIVE]

**(a) Trigger.** (i) Start offline, then add `?api=` / connect. (ii) Start connected, lose
the network entirely (transport AND submit both failing).

**(b) Simulated sequence.**

1. (i) Offline work is a LOCAL sketch state; on connect, the server scene is
   authoritative — the studio states exactly what happens to local-only work in a
   transition card: local trees are offered as a PENDING BATCH of add-operations to
   submit (each its own receipt), or discardable. Nothing silently vanishes and nothing
   silently uploads.
2. (ii) Submits exhaust retries → rows flip to `Held — offline`; `➕#transportPill`
   "Offline · will reconnect"; the studio keeps full sketch editing (the I-01 experience)
   with the badge pinned at the last shared state "R18 · stale". On reconnect, held rows
   resubmit in order with their original ids.

**(c) Edges.** The badge must never claim a class it cannot support offline: with no
stream, the badge shows the last known class WITH the stale marker and an "offline" tone
— mirroring the existing rule that offline preview never claims "Exact" (L-6).

**(d) Cognitive payoff.** The mode split from I-01 pays off: offline is a first-class
mode with a defined merge path, not a degraded accident.

---

#### I-30 Presence — collaborator join / leave [NEW, derived]

**(a) Trigger.** Another actor's operations appear on the stream (join/become active);
an actor goes quiet (leave).

**(b) Simulated sequence.**

1. `➕#presenceBar` shows actor chips (initial + assigned hue consistent per actor_id per
   session). Presence is DERIVED from observed operations — the tooltip says so:
   "Shown: people whose edits this session has observed. Not a guaranteed roster."
2. New actor's first operation → chip pops in with their first-edit attribution in the
   ledger ("Maya · first edit"). Quiet actors dim after 2 minutes and drop after 10.
3. Your own chip is always first and labeled "you". Hover any chip → their recent
   operations filtered in the drawer.

**(c) Edges.** Server-side presence/roster is out of scope this wave (no presence channel
exists) — the UI claims nothing the plane does not provide, and says so in one tooltip
line. Self-chips never count toward "N others".

**(d) Cognitive payoff.** Company is visible without ceremony, and the honesty note keeps
the pro from over-trusting the roster.

---

#### I-31 A collaborator's operation arrives [NEW surface]

**(a) Trigger.** SSE `canonical_revision` carrying another actor's operation while you
work.

**(b) Simulated sequence.**

1. The badge `R` ticks; the drawer's newest row renders with their chip and receipt
   (seq, family, verb, entity).
2. The SCENE reconciles: the edited entity pulses once (`.pulse-remote`), the dirty-region
   outline now frames THEIR window, and the fast frame for the new revision repaints both
   your and their influence regions (you see the merged world, not their camera).
3. If they edited the tree you have SELECTED: your selection persists; the selection panel
   values update to canonical with a one-line reason "updated by Maya's edit (seq 44)" —
   no focus steal, no jump.

**(c) Edges.** Operations apply exactly once per `operation_id` regardless of how often
they are re-carried (event + catch-up + snapshot paths all dedupe). Their per-user VIEW
(timestep lens) never reaches you (per-user view state rule).

**(d) Cognitive payoff.** "Others' work arrives as attributed, ordered facts on MY terms" —
the collaboration model (§6) demonstrated in one pulse.

---

#### I-32 Two people edit the same entity (conflict intuition) [NEW]

**(a) Trigger.** You and Maya both move tree `T-07`, nearly simultaneously.

**(b) Simulated sequence.**

1. Both submits are accepted (admission is about RATE, not overlap); stamps seq 41 (you),
   seq 42 (Maya) in the same epoch `e12`.
2. The reducer sorts by `server_sequence`: your `move` writes T-07's position fields, then
   Maya's overwrites the same fields — last write wins PER FIELD (her position; your
   height edit from seq 40 survives because it is a different field).
3. Canonical R15 publishes; both screens converge on Maya's position. Your ledger row
   resolves `applied · R15`; a soft note (not an error) appears if your fields were
   superseded: "position superseded by seq 42 (Maya)". If your base was older you also saw
   the I-23 chip.
4. Your local sketch briefly disagreed with the canonical result; the reconciliation pulse
   (I-31) shows T-07 settling to canonical — visibly, without drama.

**(c) Edges.** Same-field races on raster cells (I-15) resolve per cell/chunk; scalar
races per (field, timestep) (I-16). Delete-vs-update: the tombstone rule rejects updates
to a deleted generation; the updater sees "T-07 was deleted by seq 43 — start a new tree"
as an inline advisory. NO dialog, NO lock, NO "resolve conflicts" screen exists anywhere.

**(d) Cognitive payoff.** The three-sentence conflict model (§6.4) becomes lived
experience: no locks; the server stamps order; later stamps win the overlapping field and
everything else merges.

---

#### I-33 Exact analysis failure + retry [LIVE]

**(a) Trigger.** The exact job fails (`failed` status / network drop mid-poll or mid-verify).

**(b) Simulated sequence.**

1. `#failureBanner` appears with the typed code and message; `#workerPill` shows the
   failure; the badge falls back to the last good class marked stale.
2. The banner copy preserves state explicitly (UI-004): "Your design is preserved — N
   trees, all receipts intact. Retry exact analysis." `#retryButton` re-sends the
   unacknowledged attempt with the SAME idempotency key (or requests a recompute).
3. Sketches and shared state are untouched — the failure is scoped to the Verify stage;
   the pipeline chip shows Sketch ✓ Stamp ✓ Share ✓ Paint ✓ Verify ✗.

**(c) Edges.** Repeated failure keeps the banner (never auto-dismisses an unresolved
failure); the drawer keeps each attempt as a receipt row so support conversations have
seq numbers.

**(d) Cognitive payoff.** Failure of a LANE is not failure of the WORK: the five-stage
model localizes the damage and the retry is the same receipt id — composure through
structure.

---

#### I-34 Site identity mismatch [LIVE]

**(a) Trigger.** `409 site_identity_mismatch` — the scenario's pinned site identity no
longer matches the deployment's site cache.

**(b) Simulated sequence.**

1. `#siteMismatchBanner` renders the pinned vs current identity field-by-field
   (`#siteMismatchIdentities`), explains that a swapped site cache may not ride a live
   workspace, and offers exactly one action: `#reconnectButton` "Start a fresh scenario".
2. Commits resolve as LOCAL no-ops (no zombie retries); `reset()` is refused locally (a
   reset cannot repin) with a one-line reason; view-only operations (Layers tab) STAY
   available because they never mutate the pinned scene.

**(c) Edges.** The banner is modal in meaning but not in layout — reading and inspection
remain possible; the design (trees) stays visible and exportable.

**(d) Cognitive payoff.** Even the fatal-ish case names the invariant (site identity),
shows the diff, and preserves everything it can — the pro bar: no dead ends.

---

#### I-35 Typed refusals: `realtime_disabled`, `rate_limited`, `operation_not_found` [LIVE server / NEW surface]

**(a) Trigger.** (i) `503 realtime_disabled` — no broadcast hub on this deployment.
(ii) `rate_limited` — the per-scenario mutation budget (legacy edits plane). (iii)
`404 operation_not_found` — receipt lookup miss (see I-26).

**(b) Simulated sequence.**

1. (i) The studio detects `realtime_disabled` once, does NOT enter a reconnect loop, and
   downshifts cleanly: `➕#transportPill` hidden, ledger hidden, badge pinned to the exact
   session's own states; banner-lite notice: "Realtime collaboration is not enabled on
   this server. Single-user exact mode continues." Editing through the legacy plane stays
   fully functional — capability-driven UI: absent capabilities are absent UI.
2. (ii) The mutation-budget limit surfaces on the committing affordance with the server's
   wait hint; held semantics as in I-24 but labeled "scenario rate limit" (a different
   cause than the workspace burst).
3. (iii) Handled in I-26: lookup miss → resubmit same id; the ledger row narrates it.

**(c) Edges.** All three must degrade WITHOUT touching design state (UI-004 applies to
every typed refusal) and without retry storms (`realtime_disabled` retried on a
reconnect loop is the classic bug — explicitly forbidden in §4's must-not column).

**(d) Cognitive payoff.** "The studio knows what it can and cannot do HERE, and says
which is which" — capability honesty at the transport level, matching the family-card
honesty at the editing level.

---

## 4. Error + edge state matrix

Copy rules (binding for every message this UI emits):

1. Name what happened, in the active voice, with the typed code available on expand.
2. Name what survived. 3. Name the single next action. 4. No apologies ("sorry"), no
vagueness ("something went wrong"), no anthropomorphism ("the server doesn't like…").
5. Never blame the user; never say "error" for an expected state (held, stale, catching
up, qualified).

| Wire code / condition | HTTP | When it happens | UI surface + message copy (verbatim-ready) | Recovery path | MUST NOT |
|---|---|---|---|---|---|
| `operation_id_reused` | 409 | Same `operation_id` re-sent with different content (fingerprint conflict) | Ledger row `Refused` + toast: "Receipt id already used for a different edit. The original edit is unchanged. The next edit gets a fresh id." | None needed; client auto-mints a new id for the next gesture | Resubmit the same id; imply the original was damaged; generic "failed" |
| Admission — workspace burst | 429 | Batch exceeds the workspace admission envelope; rejected BEFORE durable append | Row `Held — burst limit` + countdown: "Held: this workspace is editing faster than the analysis budget. Resubmitting in {retry_after_ms}. Your edits are safe — keep working." | Auto-resubmit same `operation_id` at countdown zero; advice lines surface on repeat | Drop the edit; blind-retry immediately; call it an error |
| `fast_lane_unavailable` (server_saturated) | 503 | Fast lane cannot admit fleet-wide | Row `Held — server at capacity` + topbar count: "Server busy: new analysis work is paused fleet-wide. Held edits resubmit when capacity returns (hint {retry_after_ms}). Local sketches keep working." | Ordered held-batch resubmit on hint expiry; offline-style continuation | Disable viewing/layers; drop held edits; distinguish poorly from 429 (copy names the scope) |
| `realtime_disabled` | 503 | No broadcast hub on deployment | One-time notice: "Realtime collaboration is not enabled on this server. Single-user exact mode continues." | None; UI hides collab surfaces (capability-driven) | Reconnect loop; greyed-out collab widgets that look broken |
| `operation_not_found` | 404 | Receipt lookup for an id the server never accepted | Row state: "The server has no record of this edit yet — resending the same receipt id." | Idempotent resubmit with the SAME id | Mint a new id (duplicate-effect risk); mark as user error |
| `rate_limited` (mutation budget, legacy edits plane) | 429 | Per-scenario mutation rate exceeded on `/edits` | Commit affordance cooldown: "Scenario rate limit reached — next edit in {wait} s." | Wait; queue held locally as on the operation plane | Conflate with workspace burst admission; storm retries |
| `invalid_tree_geometry` (and any 422 validation) | 422 | Draft fails server field validation | Inline: ".is-invalid" flash on the exact control + "The server rejected {field}: {server message}. Adjust the highlighted control." | Fix the value; state never touched | Modal dialogs; wiping the input; paraphrasing the server message |
| `DraftValidationError` (client, pre-send) | — | Draft fails schema composition client-side | Card `errorLine`: exact field + expected shape (e.g. "paint window: fill all four row/col bounds") | Fix in place; nothing was sendable | Send anyway; hide that validation happened |
| `FamilyTransportError` (no transport) | — | Integrated adapter without a wired transport | Card note + payload JSON: "…The validated payload below is exactly the universal item this frontend will send once a transport lands; nothing was submitted." | None (build-time) | Fake success; omit the payload |
| `scene_version_conflict` (legacy plane) | 409 | Edit posted against an older `scene_version` | Quiet recovery per README policy: adopt authoritative scene, reapply ONCE or discard with announce: "The scenario changed on the server; edit reapplied/discarded ({reason})." | Automatic; one reapply max | Silent divergence; infinite reapply; alarming dialog |
| `site_identity_mismatch` | 409 | Site cache swapped under the scenario | `#siteMismatchBanner` with field-by-field pinned-vs-current diff + "Start a fresh scenario" button; commits are local no-ops; reset refused with reason; views stay available | Fresh scenario via `#reconnectButton` | Zombie retries; pretending a reset can fix identity; hiding the diff |
| `capability_document_unavailable` | — | Connected mode cannot fetch a usable capability document | Editors area states the fact; NO editors render; offline snapshot is never used in connected mode | Fix server; reload | Fall back to the snapshot; show empty mysterious panel |
| `network_error` / submit retries exhausted | 0 | POST fails through 1 s/2 s/4 s ×3 | Row `Held — offline` + pill "Offline · will reconnect": "Connection lost. Edits are held with their receipt ids and resubmit in order when it returns." | Auto on reconnect (same ids) | Drop; new ids; modal per failure |
| Stream `reconnect_failed` (budget exhausted) | — | 3 reconnect attempts failed | Pill `failed` + Reconnect button: "Live updates paused after 3 attempts. Your map and edits are intact. Reconnect when ready." | Manual reconnect button | Infinite spinner; auto-refresh losing state |
| `catch_up_failed` → handed to reconnect | — | Catch-up GET failed during gap recovery | Same surface as reconnecting (one flicker); no separate alarm | Reconnect cycle (corrective snapshot frame) | Double-apply; stranding on stale revisions |
| Ack stall (`onStall`, 10 s) | — | No canonical appearance of an acked op | Row "No receipt yet — checking" → GET receipt (I-26) | Idempotent resubmit SAME id if 404 | Offer new-id resubmission; unhandled rejection |
| `missed_events` heartbeat | — | SSE queue overflow dropped oldest frames | Sub-second "Catching up" flicker + ledger gap line (I-28) | Automatic | Treat as failure; full reconnect page |
| `unsupported_payload_compression` | — | zstd payload to a client without zstd decode | Failure banner (UI-004-safe) naming identity encoding | Negotiate identity on retry | Silent wrong render — the cardinal sin |
| `view_not_available` (views) | 4xx | Requested layer not published | Result card lists `published_layers`; cached-time mismatch lists `cached_time_indices` | Choose a listed layer | Hide the available set |

Degradation-ladder overlay (§ of I-20): rungs 1–6 map to badge classes Preview→Held and
are never labeled errors; the ladder text always includes the rung's cause ("from cache",
"bounded kernel", "response operator", "reduced resolution", "placeholder", "rejected
before acceptance").

---

## 5. Result-class communication design

The result class is the studio's most important sentence. Design decisions:

**D1 — One trust home.** The class lives in exactly ONE primary badge
(`➕#resultClassBadge`) on the scene header, attached to the pixels it describes. Every
other surface (metric cards, data probe, ledger rows, timeline freshness, toasts) shows an
ECHO (small chip / marker), never a competing full badge. One question, one answer.

**D2 — Words + symbol, color-independent.** Every class renders with its word AND a
symbol distinguishable without color (grayscale-safe, per accessibility):

| Wire class | Badge word + symbol | Map treatment | Metrics treatment | Tooltip sentence (hover) |
|---|---|---|---|---|
| `fast_exact` | **Fast exact** ● | Full-opacity fast plane | Plain numbers + `R` anchor | "Real physics for R{n}, computed in about a second by the incremental fast lane." |
| `fast_qualified` | **Qualified estimate** ◐ + `~` | Fast plane + qualifier chip on badge; reduced-res rung outlines affected cells | Numbers with stated ± and coverage echo ("−1.1°C ± 0.6") | "Physics with a stated margin of error, shown while the exact answer computes. Open for the evidence." |
| `visual_pending` | **Preview (owed)** ◌ pulsing | Sketch layer at reduced opacity (0.42) + preview marker; last exact field beneath, marked stale | Dimmed to last verified values with "at R{n}" caption | "A placeholder over the last verified picture. The exact answer for R{n} is still owed." |
| `exact_reconciled` | **Exact** ■ | Full exact plane | Plain numbers, `E = R` shown in triple | "Verified SOLWEIG physics for R{n}." |
| superseded | any + **(stale)** | Kept picture + stale marker (I-21) | Captioned with the revision they describe | "This picture describes R{n−k}; the scene is at R{n}. Refreshing." |

**D3 — Transport ≠ trust.** The stream-health pill (`➕#transportPill`: Live /
Reconnecting / Catching up / Offline / failed) is a SEPARATE control from the class badge.
Today's `#realtimePill` conflates them (phase states override class text). The redesign
splits them: connection health answers "am I hearing the server?", the class answers "can
I trust the pixels?" — different questions, different widgets, never one label doing both.

**D4 — Age without anxiety.** The owed-age counter appears only after 10 s of pending
("exact owed · 42 s"), so normal sub-second fast-lane latency never reads as delay. Pro
tooltip always exposes the full triple (`W15 F14 E9`) and `ageMs`.

**D5 — Qualified never passes for exact.** The `~` qualifier is sticky: it appears on the
badge AND on every metric echo AND in the data probe, in the same gesture cluster. The
qualification panel is one click from ALL of them (they all open the same panel). An
undergraduate cannot quote a qualified number without its `±` in the same string.

**Qualification panel outline** (opens from any `~`; content model):

1. Header — class chip + anchors: "Qualified estimate for workspace R14 · fast R14 ·
   exact base R9".
2. Plain meaning — the glossary sentence (§1.3), verbatim.
3. Error evidence — method ("validated response operator"), stated uncertainty
   (±X°C), how the bound was produced.
4. Coverage — region and timesteps qualified; reduced-resolution rung states
   "half resolution" and outlines affected cells; uncovered cells listed by window.
5. Limitations — the server's limitation strings VERBATIM (UI-005 rule: never
   paraphrase), e.g. "Tree-induced local wind-field changes are not recomputed."
6. Expected reconciliation — "exact lane: R9 → R14 in progress; vegetation geometry
   verifies asynchronously" + the family's exact SLO class (§1.2 table).
7. Anchor line — `exact_base_revision` explained: "built on the verified exact result
   for R9; the difference R9→R14 is what the operator is estimating."
8. Link — Model scope card (`#modelScopeList`, `#modelVersionValue`,
   `#siteCacheVersionValue`).

**Unknown future classes** display conservatively as Preview (`effectiveClass` rule) with
the raw wire class visible in the pro tooltip — forward compatibility without
over-claiming.

---

## 6. Collaboration model

**6.1 The whole model in five sentences (usable as an onboarding card):**

1. You and your collaborators edit the same scene; the server merges everyone's edits into
   shared steps about ten times a second.
2. Every edit is a numbered receipt; re-sending a receipt never duplicates the edit.
3. Nobody locks anything — the server stamps the order; when two edits touch the same
   field, the later stamp wins and everything else merges.
4. If you edited an older view of the scene, the server still merges yours — you see a
   small "based on R13" chip, never a block.
5. Other people's edits arrive on your screen as attributed facts; your time lens and
   layer choices stay yours.

**6.2 Surfaces.** Presence bar (I-30, derived roster with honesty tooltip) · Activity
drawer/ledger (W-5: one row per operation — actor chip, seq, family · verb, entity,
five-stage dots, receipt on click) · remote-edit pulses on the scene (I-31) · revision
ticks in the badge (one `R` increment per epoch, rows sharing an `R` reveal batching).

**6.3 What is shared vs personal.**

| Shared (workspace-wide) | Personal (per-user, never broadcast) |
|---|---|
| All scene edits (8 families) | Displayed timestep lens (I-05) |
| Requested outputs / published products | Layer toggles, compare mode |
| Revision triple, result class | Selection, camera, drawer expansion |

**6.4 Conflict intuition by source type** (from `collaborative_state.md`, phrased for the
ledger's supersession notes):

| Family | Rule when two edits overlap | Ledger phrasing |
|---|---|---|
| Object geometry (building, vegetation) | Last write wins PER FIELD by `server_sequence`; delete tombstones a generation | "position superseded by seq 42 (Maya)" |
| Raster paint (land cover) | Per cell/chunk, later `server_sequence` wins unless the adapter declares a commutative merge; one epoch-final mask | "merged with seq 43 — later stroke wins per cell" |
| Scalars (meteorology, time, parameters) | Last validated write per (field, timestep/range); ranges normalized to disjoint segments | "air_temp 14:00 keeps seq 41 (newer)" |
| View state | Per-user changes never invalidate science; workspace-wide output requests may schedule products | — |

**6.5 No-lock etiquette affordances.** The studio deliberately provides no "claim tree"
or "user is editing" lock UI: locks would contradict the epoch model and teach fear.
The pulse + supersession notes are the entire arbitration surface. (An OPTIONAL
"highlight what Maya last touched" filter lives in the drawer; following/telepointing is
out of scope this wave.)

---

## 7. Cognitive risk register

Top 10 ways this UI could confuse, each traced to its countermeasure above.

| # | Risk | Why it would confuse | Countermeasure | Trace |
|---|------|----------------------|----------------|-------|
| 1 | A qualified or pending picture is read as verified physics | The map looks identical in every class | D1/D2/D5: one badge, word+symbol, sticky `~` on every echo and probe; panel evidence | §5, I-03, I-20 |
| 2 | Three revision numbers feel like noise, so users ignore them | "v15? R18? which is current?" | The triple is collapsed to one `R` headline with `W·F·E` for pros; the badge tells the gap story ("exact owed since R9") | §1.1-4, §5 D4, I-02 |
| 3 | 429/503 read as "broken" or "my edit is lost" | Rejection feels like failure | Held rows + countdown + "your edits are safe" copy; held vs owed visually distinct | I-24, I-25, §4 |
| 4 | `base_divergent` read as rejection or "someone overwrote me" | "Divergence" sounds alarming | Advisory chip with the one-sentence merge story; no dialog ever | I-23, §6.1-4 |
| 5 | The dirty-region outline read as a selection or as "results only valid inside" | An outline is ambiguous | Renamed "Refresh window"; caption quantifies ("35,480 / 250,000 cells"); glossary states outside is reused, still valid | §1.1-9, I-06 |
| 6 | Time scrubbing: per-tick jobs feared, or stale hours read as fresh | Cached planes look authoritative | Debounce + freshness dots + "personal lens" tooltip; refinement is an explicit product request | I-04, I-05 |
| 7 | Retry storms / fear of double-submit during retries | Idempotency is invisible | Receipts in the ledger; duplicate acks resolve visibly to one effect; stall path resends the SAME id only | I-22, I-26, §3.3 |
| 8 | Reconnect/catch-up read as data loss or a hung app | Silence during buffering | W-6 copy names what is preserved; gap markers; badge only ticks up; editing continues | I-27, I-28, I-29 |
| 9 | Debounced commits feel ignored → users re-commit and burst | 480 ms is perceptible | Pipeline chip shows the commit countdown; nudge/slider coalescing is displayed, not hidden | I-12, I-14, §3.2 |
| 10 | Ledger floods during multi-actor bursts (epochs batch many ops) | One row per operation × 10 Hz | Rows group by epoch (`R18 · 3 edits`), summary lines on catch-up, drawer collapses to one headline | W-5, I-27, I-31 |

Bonus guard (not counted): offline/connected mode confusion — countered by the mode split
taught at boot (I-01) and the held-resubmit merge path (I-29).

---

## Appendix A — Design decisions introduced by this spec (implementer index)

| Id | Decision | Where |
|----|----------|-------|
| D1 | One trust home (`#resultClassBadge`); echoes elsewhere, never duplicate badges | §5 |
| D2 | Class words + grayscale-safe symbols; unknown classes degrade to Preview | §5 |
| D3 | Split transport health (`#transportPill`) from result trust (class badge) | §5 |
| D4 | Owed-age counter only after 10 s | §5 |
| D5 | Sticky `~` qualifier across badge, metrics, probe; one shared qualification panel | §5 |
| D6 | Five-stage pipeline vocabulary (Sketch/Stamp/Share/Paint/Verify) replaces the four exact-session step labels | §1.2 |
| D7 | `pointercancel`/Escape during a drag restores `oldTree` and commits nothing (behavior change from today's `finishPointerDrag` commit-on-cancel) | I-11 |
| D8 | Timeline freshness dots per timestep; scrubbing is a personal lens; refinement rides the view family | I-05 |
| D9 | Activity ledger = one row per operation (yours and others'), epoch-grouped, receipts on click; absorbs the outbox | W-5, §6 |
| D10 | Presence is derived from observed operations, labeled as such; no server roster claims | I-30 |
| D11 | Held vs owed are distinct surfaces: held = not accepted (row + topbar count); owed = accepted, result pending (badge) | I-24/I-25 vs I-20 |
| D12 | Withdraw exists only for pre-acceptance held edits; no global undo; delete compensates add | §3.3 |

All proposed NEW ids: `#resultClassBadge`, `#transportPill`, `#presenceBar`, `#heldCount`,
`#activityToggle`, `#activityList`, `#timelineFreshness`, `.pulse-remote`, `.qual-trigger`
(qualification panel opener), and the qualification panel itself. Everything else keeps
its existing id from `index.html`.
