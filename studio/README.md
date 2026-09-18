# Incremental design-tool frontend prototype

This directory contains a dependency-free browser prototype for the CPU incremental design-tool architecture described in `docs/incremental_design_tool/`.

The prototype is deliberately separated into two layers:

- `model.mjs`, `solver_kernel.mjs`, and `solver_worker.mjs` implement deterministic edit geometry and a lightweight browser preview kernel.
- `app.mjs` and `renderer.mjs` implement the interaction and visualization. The exact path talks to the HTTP job contract in `docs/incremental_design_tool/api_contract.md` through `api_client.mjs` and `exact_session.mjs`; the preview kernel remains client-side and non-scientific.
- `capabilities.mjs` and `family_edits.mjs` implement the **capability-driven UI** (U-D2): the editor list, every property bound, the class vocabulary, the view-layer list, and the dependency-stage classification are GENERATED from the server's capability document. No physics vocabulary is hardcoded in this frontend.

The preview kernel is **not** a scientific replacement for SOLWEIG. It exists to exercise the required interaction states, dirty-window protocol, patch application, cancellation semantics, and rendering behavior while the exact server result is pending.

## Run

```bash
cd studio
python serve.py
# open http://127.0.0.1:8765/
```

No package installation or build step is required. Without `?api=` the studio runs fully offline (local preview kernel only).

## Connect to the exact CPU worker

Start the SOLWEIG-GPU API server, then serve this page with the same-origin reverse proxy so the browser avoids CORS (the API sends no CORS headers). The API boots from environment variables (`python -m solweig_gpu.server`):

```bash
# terminal 1 — API + in-process worker (listens on 0.0.0.0:8000)
# SOLWEIG_STATE_ROOT defaults to ./state in the CURRENT directory — set it
# explicitly so the SQLite store and results land where you expect.
SOLWEIG_STATE_ROOT=/path/to/state \
SOLWEIG_SITE_ID=campus-1km-v1 \
SOLWEIG_CACHE_ROOT=/path/to/site-cache \
python -m solweig_gpu.server

# terminal 2 — frontend + same-origin /api proxy
cd studio
python serve.py --api http://127.0.0.1:8000
# open http://127.0.0.1:8765/
```

`SOLWEIG_SITE_ID` is required (comma-separated for multiple sites); each site's cache resolves to `SOLWEIG_CACHE_ROOT/<site_id>/`. The full variable table lives in `docs/incremental_design_tool/deployment_operations.md` (§Configuration); `Dockerfile`/`docker-compose.yml`/`fly.toml` at the repo root give the containerized shape.

- When served with `--api`, the page itself injects the connected defaults (`SOLWEIG_API_BASE="/api"` and, from the server's `SOLWEIG_SITE_ID`, `SOLWEIG_SITE_ID`), so a bare `http://127.0.0.1:8765/` boots connected with no query parameters.
- `?api=<base-url>` activates connected mode. Any relative base (recommended, e.g. `/api`) goes through the proxy above; an absolute URL also works if the API is CORS-enabled. An EMPTY `?api=` overrides the injected default and forces offline mode.
- `?site=<site-id>` selects the site (overrides the injected default; offline default `campus-1km-v1`).
- Connected mode starts with **zero trees**: the server's baseline scene is authoritative. Offline mode keeps the three starter trees.

The proxy forwards `Idempotency-Key`, `If-Match`, and `Accept` headers, response ETags, and the custom `X-SOLWEIG-*` payload headers, and streams binary patch bodies unchanged (`tools/smoke_proxy.mjs` verifies this end to end).

## Capability-driven UI (U-D2)

At session start the studio fetches the engine's capability document (`GET /api/v1/capabilities`, falling back to the `/capabilities` alias when the versioned path is absent) and **builds the edit UI from it**:

- **Editor list** — adapters with `integrated: true` and at least one operation beyond the `view_only` vocabulary become editors; `output_view` stays a view adapter; the rest render as disabled cards carrying their status and disclosure text verbatim (UEDIT-010: the dynamic-wind parity note is the document's own text, never paraphrased).
- **Property editors** — sliders/number inputs with the schema's `minimum`/`maximum`/`exclusive_*`/`default`, class pickers as `valid_classes − fenced_classes`, variable lists from each adapter's `property_schema`, rejected properties (`transmissivity`) shown with the schema's rejection reason instead of an editor. The old hardcoded height/canopy/transmissivity sliders are gone; the same panel is generated from the document.
- **Offline demo** — `assets/capabilities_snapshot.json` is a verbatim dump of the live document (see below), so the offline mode runs the *same* UI code path. Connected mode **never** falls back to the snapshot: if the server does not serve a usable document the studio says so and shows no editors (typed error `capability_document_unavailable`).

### Family edits and their transports

`family_edits.mjs` converts a document-generated draft into contract edit items for the transport that can carry it. Both integrated transports are live:

- `tree-edits-v1` — `POST /api/v1/scenarios/{id}/edits` for `vegetation_geometry` (the schema's `canopy_radius_m` maps to the contract's `canopy_diameter_m` ×2).
- `universal-edits-v1` — `POST /api/v1/scenarios/{id}/edits/universal` for `meteorological_forcing`, `landcover_surface`, `building_geometry`, and `model_receptor_parameters`, sending the universal item grammar `{adapter, operation, values, target?, time_index?, old_values?}` the server validates through its adapter registry.

An adapter that the document marks integrated before this build wires a transport table entry for it still gets a **typed refusal, never a silent drop**: its editor composes and schema-validates the draft, then `composeFamilyEdit` throws a `FamilyTransportError` carrying the exact item it is ready to send — no fake success, no wasted POSTs, and one transport-table row away from sendable.

### Dependency impact panel (UEDIT-009)

After each committed edit the impact panel classifies every dependency node for the edited family's closure:

- **changed** — the edit family's source node(s),
- **recomputed** — the rest of the family's `dirty_nodes` closure,
- **reused** — every other node in the document's vocabulary,

plus a scope line from the job's realized `metrics` (`window_fraction`, `mode`, `fallback_reason`; the `no-op` mode reads "no recompute — the published result was re-served").

Server gap (reported, not patched): **no endpoint publishes the executed plan**, so the stage classification is derived from the document's declared `edit_families` closure rather than the per-stage execution record; only the metrics above describe what actually ran. Job bodies carry `mode`/`window_fraction`/`fallback_reason`/`dirty_windows` — the plan with per-node changed/reused/recomputed staging lives only inside `PlanExecutor`.

### View-only panel (UEDIT-007)

The Layers tab exposes the document's `view_only` operations (`select_layer`, `compare`, `legend`, `cached_time`) over the published layers (`producible_layers`), posting `POST /api/v1/scenarios/{id}/views`. Results render layer metadata (dtype/shape/nodata), legend statistics, or the cached time-step chips; the zero-jobs badge quotes the **server's own** `zero_scientific_jobs`/`job_enqueued` flags rather than asserting freeness client-side. Refusals keep their typed details (`view_not_available` with `published_layers`, cached-time mismatches with `cached_time_indices`).

### Site-identity mismatch

A `409 site_identity_mismatch` (the scenario's pinned site identity no longer matches the live deployment geometry) fails visibly: a banner shows the pinned vs current identity field-by-field, commits resolve as local no-ops (no zombie retries), `reset()` is refused locally (a reset cannot repin a scenario — recovery is a fresh scenario via the Reconnect button), and view-only operations stay available because they never mutate the pinned scene.

## Exact-session behavior (acceptance gates)

| Gate | Behavior |
| --- | --- |
| UI-001 | Every **committed** interaction sends exactly one `POST /edits` with `base_scene_version` = last known `scene_version`, a generated `Idempotency-Key`, and `If-Match: "scene-version-N"`. Mutations queue in-order, so consecutive edits chain base versions without self-inflicted 409s. |
| UI-002 | Property sliders debounce: preview recompute after 150 ms of quiet, commit after a 480 ms trailing window (spec: 400–600 ms; the final value wins). The tree state at the **start** of the debounce interval is kept so the old influence region is not lost. |
| UI-003 | Results whose `scene_version` is older than the awaited version are dropped silently; out-of-order poll completions are ignored via a poll-generation guard; payloads are validated (schema version, SHA-256 checksum over the served bytes, ETag cross-check, window match) before `texSubImage2D` writes the sub-rectangle. |
| UI-004 | Failures (job `failed`, network drop) leave the design state untouched: trees, edits, and preview remain, a failure banner appears, and the Retry button re-sends the unacknowledged attempt with the **same** idempotency key or requests a recompute. |
| UI-005 | The model-scope card shows the manifest `limitations` (e.g. "Tree-induced local wind-field changes are not recomputed."), `model_version`, and `site_cache_version`, and marks the browser overlay as a non-scientific approximation. |

Conflict policy (spec leaves reapply-or-discard open): on `409 scene_version_conflict` the client reloads `GET` the scenario, adopts the authoritative scene, and reapplies the edit **once** — an `add` only if the tree id is absent, a `move`/`update`/`delete` only if the tree id still exists. Otherwise the edit is discarded and announced. The recovery POST uses a fresh idempotency key.

Time-slider changes cannot be a bare `POST /edits` (the contract requires `edits` `min_length=1`), so they ride a no-op `update` edit for the currently selected tree carrying `requested_result: {time_indices: [hour]}`. The displayed hour switches to the baseline plane immediately; the exact patch refines it when the job completes.

Invalid geometry (`422 invalid_tree_geometry`) surfaces the server's `field` on the matching slider without discarding any state.

Known limitation (L-7): `connect()` infers the site grid from the baseline manifest as `rows = window.row_stop, cols = window.col_stop`, i.e. it assumes the baseline window is the full grid anchored at the origin. Sites whose stored baseline does not cover the full raster from (0,0) would report a wrong grid shape; no such site exists in the current server.

Edits committed while the session is still connecting are held and replayed in order once the baseline loads (never posted against a half-connected session). A transient connect failure (network drop, 5xx, 429) reconnects with bounded exponential backoff — 1 s, 2 s, 4 s, three retries — and the held backlog replays on the attempt that lands; permanent errors (4xx) surface immediately and release the backlog. Reset during the connecting window drops the held backlog (nothing was ever POSTed, so the server scene is already the baseline) so stale edits never replay onto the post-reset baseline. Transient `GET /jobs` failures (network, 5xx, 429) retry with exponential backoff instead of killing the poll loop.

## Patch payload compression: zstd vs identity

The API contract's patch media type is `application/vnd.solweig.patch+zstd`. The client **feature-detects and negotiates**:

- `zstdStreamSupported()` probes `new DecompressionStream("zstd")`.
- Unsupported (the common case today) → `Accept: application/vnd.solweig.patch+identity` with no zstd offer.
- Supported → `Accept: application/vnd.solweig.patch+zstd, application/vnd.solweig.patch+identity;q=0.5`.
- If zstd arrives anyway without support, decoding raises `unsupported_payload_compression` (message names identity), producing a UI-004-safe Failed state rather than a silent wrong render.

Evidence for the default (2026-09-01):

- Node v26.8.1: `DecompressionStream` accepts only `gzip` and `deflate` — `new DecompressionStream("zstd")` throws. Asserted live in `tests/api_client.test.mjs` ("runtime lacks zstd DecompressionStream in this node").
- caniuse "DecompressionStream zstd": 0% global support. Chrome ≤154 does not ship it (Chrome 129+ added zstd only for HTTP `Content-Encoding`, not the JS API); Firefox 138+ has it behind a disabled-by-default flag; Safari does not ship it.

Because of that, the round-trip test path is identity (fully exercised in node); the zstd branch is covered by an injected stub DecompressionStream, and the unsupported-zstd error path is asserted directly. The server honors the negotiation: a client that offers only `application/vnd.solweig.patch+identity` receives raw planes with `Content-Type: ...+identity` and an `X-SOLWEIG-Checksum` covering **the served bytes** (the manifest's own checksum still covers the zstd encoding, so the client verifies against the header whenever it is present). The served content type — not the manifest's `compression` field — decides whether `decodePatch` decompresses.

Checksums: `manifest.checksum` (and the `ETag`) are SHA-256 over the **compressed/served** bytes, so both are verified before decompression.

## Test

```bash
node --test tests/*.test.mjs          # 174 tests: adapter, session gates, debounce/mappings, kernel, capabilities, families, views, site mismatch, realtime
node tools/smoke_proxy.mjs            # serve.py --api reverse-proxy smoke test
node --check app.mjs renderer.mjs model.mjs solver_kernel.mjs solver_worker.mjs api_client.mjs exact_session.mjs capabilities.mjs family_edits.mjs
node --input-type=module -e "await import('./app.mjs')"   # DOM-free smoke import
```

- `tests/api_client.test.mjs` — adapter contract against a fake `fetch`: request shapes, error envelopes, Accept negotiation, checksum/ETag validation, little-endian variable-major decode, zstd branch (stub) and identity round trip.
- `tests/exact_session.test.mjs` — the session gates against an in-memory contract server with a manual clock: exactly-one-POST, full poll cycle, stale rejection, out-of-order polls, superseded dropped, failed-preserves-state + retry, idempotent replay, 409 recovery/discard, 422 validation, no-op recompute edits, reset, connect retry with backlog replay, reset-while-connecting backlog drop.
- `tests/capabilities.test.mjs` — the capability-document fixture (the checked-in snapshot) drives every generated behavior: editor lists, control bounds and defaults, class picker = valid − fenced, rejected/blocked properties, stage classification (partition property asserted), scope lines from metrics, view vocabulary, zero-job flag passthrough, unknown-status passthrough.
- `tests/family_edits.test.mjs` — draft → contract composition: radius↔diameter conversion, schema defaults, delete items, bounds rejection, and the universal-payload disclosure for families without a transport.
- `tests/views.test.mjs` — view request casing, typed refusal envelopes (matching the server's flattened `error` object), session gate, exact-version default, mutation-queue bypass.
- `tests/site_mismatch.test.mjs` — 409 handling: pinned/current identities recorded, blocked commits are local no-ops, reset refused locally, views stay available, a fresh session clears the flag.
- `tests/design_gates.test.mjs` — debounce coalescing and the 400–600 ms window, contract tree mapping, time-index/window math, `applyExactPatch` region writes, colorization.
- `tests/model.test.mjs`, `tests/solver_kernel.test.mjs` — pre-existing geometry and kernel tests (unchanged).
- `tests/contract_harness.mjs` — shared `FakeContractServer` + manual clock (not itself a test file); it also serves the capability snapshot because `connect()` requires the document.

A browser smoke test can be run with Chromium:

```bash
python serve.py --port 8765 &
chromium --headless --no-sandbox --disable-gpu \
  --window-size=1600,1000 \
  --screenshot=/tmp/solweig-studio.png \
  http://127.0.0.1:8765/
```

## Rebuild the visual fixture

`tools/build_assets.py` accepts a 500×500 SOLWEIG test tile and creates a non-georeferenced presentation image plus a 128×128 UTCI fixture. The raw rasters are intentionally not committed.

Rebuilding assets requires the optional developer packages `rasterio`, `opencv-python`, `Pillow`, and `scipy`. The checked-in demo itself has no runtime package dependency.

```bash
python tools/build_assets.py \
  --building /path/to/Building_DSM.tif \
  --dem /path/to/DEM.tif \
  --trees /path/to/Trees.tif \
  --landcover /path/to/Landcover.tif \
  --utci /path/to/UTCI_0_0.tif \
  --shadow /path/to/Shadow_0_0.tif \
  --hour-band 12 \
  --output assets
```

## Refresh the offline capability snapshot

`assets/capabilities_snapshot.json` is a verbatim dump of the live capability document (from `solweig_gpu.incremental.capabilities.capability_document()`), so the offline demo builds the same generated UI the connected studio does. Regenerate it whenever the document's vocabulary changes:

```bash
python -c "
import json
from solweig_gpu.incremental.capabilities import capability_document
print(json.dumps(capability_document(), indent=2, sort_keys=True))
" > assets/capabilities_snapshot.json
node --test tests/capabilities.test.mjs   # the fixture is the test document
```

Connected mode never reads this file — it always fetches the server's document and fails visibly (no editors) if the server does not serve one.

## Export a baseline result into the site cache

Without `<cache_dir>/baseline_results/` every new scenario pays one full-tile
solve at connect time (minutes on CPU). With it, scenario creation
materializes the baseline instantly (`status: "exact"`, no job). After the
first full solve, export it once:

```bash
docker cp <api-container>:/state/scenarios/<scenario_id>/results/0 /tmp/scn0
python tools/export_baseline.py --result-dir /tmp/scn0 --cache-dir site-cache/<site_id>
```

## Realtime client (in adoption)

`realtime_client.mjs` is a standalone, DOM-free client for the collaborative
operation plane (`docs/incremental_design_tool/realtime_collaboration/`). It
owns per-actor operation identity (`actorId` + monotonic `client_sequence`),
advisory-`base_revision` submits with idempotent 1s/2s/4s retries, and an
EventSource subscription that tracks the `workspace/fast/exact` revision
triple, the result-class state machine (`classFor`), and zero-loss reconnect
(`GET .../operations?since_server_sequence=N` closes the event gap; operations
apply exactly once per `operation_id`). `fetch`, the EventSource factory,
timers, and the clock are injectable, so it runs under `node --test` without a
server. The studio UI uses it read-only today (the collab badge below); edit
submission through the operation plane lands in a later wave:

```bash
node --test tests/realtime_client.test.mjs tests/realtime_fast_lane.test.mjs \
  tests/realtime_badge.test.mjs
```

### Fast-lane consumption contract (r2c)

What the client guarantees while consuming the R2 fast lane, and what a
consumer of `classFor`/`revisionsFor` may rely on:

**Revision triple.** Every workspace satisfies
`exact_revision <= fast_revision <= workspace_revision`. All three arrive on
`canonical_revision` frames (the epoch scheduler, including the subscribe-time
and corrective `snapshot: true` frames, which carry empty `operations` —
history belongs to the catch-up GET) and are tracked monotonically in
`revisionsFor`; no frame, however late or duplicated, can regress them.

**Wire order.** For any revision R, the `canonical_revision` frame reaches
subscribers BEFORE the matching `fast_revision` frame (r2b one-scan grace) —
the client relies on this for the initial render, but stays robust to
reconnects, where fast frames may be missed entirely.

**Result classes.** `fast_revision` frames carry `result_class`:
`fast_exact` (the fast result is authoritative for its revision) or
`visual_pending` (a visual placeholder — the exact result is still owed).
`fast_qualified` is reserved server-side and never on the wire this wave.
`classFor` reports the raw class (`class`) plus a conservative display class
(`effectiveClass`): anything that is not a current `fast_exact` or
`exact_reconciled` — reserved classes and unknown future classes included —
displays as `visual_pending`.

**Supersession fence.** A fast result approximates one canonical revision
(`targetRevision`, from the frame's `workspace_revision`). Once a newer
canonical revision is rendered (canonical frames, exact frames, and the
catch-up GET body advance that watermark; fast frames never do), the fast
result is `superseded` and `authoritative` is false — the client never renders
a fast result behind the last rendered canonical revision.

**Fast frames are lossy by design.** Fast payloads are in-memory + SSE only;
they are not in the durable operation log. After a reconnect, the catch-up GET
recovers operations but NO fast payloads — a `visual_pending` result with no
follow-up fast frame is a legitimate terminal state until the next epoch. The
client never waits for one.

**missed_events (broadcast.py L2).** A subscriber whose SSE queue overflowed
has its OLDEST frames dropped and receives `missed_events: N` on the next
heartbeat. The client answers by closing the gap exactly like a reconnect:
the gap fence goes up (later frames buffer), the catch-up GET replays the
dropped operations, then the buffered frames flush in order — everything
applies exactly once. A failed recovery GET hands over to the full reconnect
cycle, whose fresh subscription receives a corrective snapshot.

**Admission rejections.** The server rejects BEFORE durable acceptance:
429 (workspace burst) and 503 `fast_lane_unavailable` (server-wide) carry
`details.retry_after_ms` (and usually `advice`). `submitOperations` surfaces
them as a typed `RealtimeAdmissionError`
(`kind: "burst" | "server_saturated"`, `retryAfterMs`, `advice`) instead of
retrying blind — the UI decides when to resubmit. Because an idempotent RETRY
of an ALREADY-DURABLE batch can also draw a 429 (r2b F7), the client
reconciles through the catch-up GET before surfacing a rejection on a retry
attempt: if the batch turns out to be durable, the submit resolves through the
normal canonical fence rather than rejecting.

### Collab badge (demo)

`realtime_badge.mjs` renders the badge state (pure `realtimeBadgeState` +
DOM glue `applyRealtimeBadge`); the studio topbar gains a third pill
(`#realtimePill`, hidden until a scenario connects) showing the revision
triple and the effective result class — `visual preview` (pulsing, exact
owed), `fast exact`, `exact`, `(stale)` once superseded, plus
catching-up/reconnecting/failed states. `app.mjs` subscribes it read-only to
the connected scenario's workspace on session `ready`; it never gates edits.

