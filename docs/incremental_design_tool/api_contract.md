# API contract

## Design principles

- Edits are durable before acknowledgement.
- Every mutation is versioned and idempotent.
- Exact analysis is asynchronous.
- Results are rectangular binary patches with explicit metadata.
- The client rejects stale results.
- The API exposes model limitations and result provenance.

The examples use JSON over HTTP. A later implementation may use generated OpenAPI models, but field names and semantics should remain stable.

## Create scenario

### Request

```text
POST /api/v1/scenarios
Content-Type: application/json
Idempotency-Key: create-class-demo-01
```

```json
{
  "site_id": "campus-1km-v1",
  "name": "Section A team 4",
  "initial_state": "baseline"
}
```

### Response

```text
201 Created
```

```json
{
  "scenario_id": "scn_01J7B4...",
  "site_id": "campus-1km-v1",
  "scene_version": 0,
  "exact_result_version": 0,
  "status": "exact",
  "trees": [],
  "result_manifest_url": "/api/v1/scenarios/scn_01J7B4/results/0"
}
```

## Read scenario

```text
GET /api/v1/scenarios/{scenario_id}
```

Returns the authoritative tree list, versions, active job, and model-scope metadata.

## Capabilities document

```text
GET /api/v1/capabilities
```

The engine's self-description and the single source of truth for edit
families: one `adapters[]` entry per family (id, status, `integrated` flag,
operations, source nodes, dirty-node closure, and a `property_schema` whose
bounds, defaults, class vocabularies, and fences are the adapter's own),
plus `edit_families`, document-level `fences`, and the `view_only` section
(operations, producible layers, adapter ids, `zero_scientific_jobs` flag).
Clients generate their edit UI from this document; they never hardcode a
physics vocabulary (see [Frontend spec](frontend_spec.md)). A short alias
`GET /capabilities` exists for bootstrap.

The adapter's validators are authoritative for every edit submitted on the
universal transport (below): an in-range-looking value a client dreamed up
is still refused by the server's registry ruling.

## Commit family edits (universal transport)

The tree endpoint carries vegetation items only. The four other integrated
families — `meteorological_forcing`, `landcover_surface`,
`building_geometry`, `model_receptor_parameters` — travel as universal edit
items on their own endpoint, with the same mutation discipline as tree
edits: `Idempotency-Key` + `If-Match` preconditions, one scene-version
bump, one job on the shared queue, coalescing, and typed refusals.

### Request

```text
POST /api/v1/scenarios/{scenario_id}/edits/universal
Content-Type: application/json
Idempotency-Key: browser-7-famedit-3
If-Match: "scene-version-17"
```

```json
{
  "base_scene_version": 17,
  "edits": [
    {
      "adapter": "meteorological_forcing",
      "operation": "update_time_row",
      "values": {
        "air_temperature": 31.5,
        "humidity": 14,
        "radiation": 800,
        "wind_speed": 2.4,
        "pressure": 1013,
        "uhii": 0
      },
      "target": null,
      "time_index": 12,
      "old_values": null
    },
    {
      "adapter": "landcover_surface",
      "operation": "paint",
      "values": { "class": 5 },
      "target": { "row_start": 201, "row_stop": 245, "col_start": 228, "col_stop": 272 },
      "time_index": null,
      "old_values": null
    }
  ],
  "requested_result": { "time_indices": [12], "variables": ["utci", "tmrt"] }
}
```

Item grammar:

- `adapter` — a family id the capability document lists as integrated.
- `operation` — one of that adapter's registered operations.
- `values` — the adapter's property vocabulary (schema-validated
  server-side by the registry; the document's bounds and fences rule).
- `target` — the address the operation needs: a raster window object
  (`{row_start, row_stop, col_start, col_stop}`, half-open) for land-cover
  paints, the entity id string for building operations, `null` otherwise.
- `time_index` — for temporally addressed operations (e.g. met rows).
- `old_values` — the entity's declared current state, required by
  operations that validate a transition (building move/update).

A batch may mix families (one version bump, one job). Vegetation stays on
the tree transport BY DESIGN: sending a `vegetation_geometry` item here is
`400 unsupported_transport` naming `tree-edits-v1` (tree authority — ids,
quotas, reset semantics — lives in the tree contract).

### Response

Same shape as the tree endpoint's `202`, plus `"transport":
"universal-edits-v1"`. Job status, result manifests, payloads, and SSE are
shared with tree edits.

### Typed refusals

- `400 unsupported_transport` — adapter id is tree-only or not integrated
  (the message names `tree-edits-v1` or points at `GET /api/v1/capabilities`).
- `400 invalid_family_batch` — an `output_view` item mixed with family
  items (view-only items must travel alone).
- `422 invalid_edit_state` — the registry or adapter refused the payload.
  The `message` is the validator's own verbatim ruling — fence names,
  domains, and grammar come from the adapter, never restated by the
  transport — and the envelope carries `adapter` and `operation` fields.
  Example: painting fenced water class 7 is refused with the adapter's
  "… class 7 (water) is fenced …" message.
- `400 invalid_request` — transport-shape problems (a paint without a
  window target, a building edit without an id target, `If-Match` and
  `base_scene_version` disagreeing).

Refusals enqueue nothing, bump no version, and consume no idempotency key.

### View-only guarantee on the universal transport

A batch whose single item is `output_view` is answered inline from the
published result — `200` with `job_enqueued: false` and
`zero_scientific_jobs: true`, exactly like `POST /views` (below): no event,
no store mutation, no submit anywhere in that branch.

## View-only operations

```text
POST /api/v1/scenarios/{scenario_id}/views
```

```json
{ "operation": "select_layer", "layer": "utci", "time_index": 0 }
```

Operations: `select_layer`, `compare`, `legend`, `cached_time`. Every
answer is composed from ALREADY-PUBLISHED state only (capability document,
scenario record, stored manifest/payload): **zero scientific jobs** — the
response states `job_enqueued: false` and `zero_scientific_jobs: true`, and
the server enforces it by construction (no store mutation, no submit in any
branch). `legend` is decode-free when the manifest carries per-plane
`statistics` (min/max/mean/count served from metadata; a streaming
single-plane decode covers older manifests). Refusals are typed:
`view_not_available` (409, with `published_layers`), `result_not_ready`
(404), `invalid_request` (400, off-vocabulary layer/operation).

## Commit one or more edits

### Request

```text
POST /api/v1/scenarios/{scenario_id}/edits
Content-Type: application/json
Idempotency-Key: browser-7-edit-42
If-Match: "scene-version-17"
```

```json
{
  "base_scene_version": 17,
  "edits": [
    {
      "operation": "add",
      "tree": {
        "tree_id": "tree_01J7B4N8P0KQ",
        "component_type": "broad_canopy",
        "u": 0.575,
        "v": 0.345,
        "height_m": 18.0,
        "canopy_diameter_m": 11.0,
        "trunk_ratio": 0.25,
        "transmissivity": 0.03,
        "phenology": "deciduous"
      }
    }
  ],
  "requested_result": {
    "time_indices": [12],
    "variables": ["utci", "tmrt"],
    "refine_full_day": true
  }
}
```

For move or update, send the complete replacement tree object. For delete, send `tree_id`.

### Response

```text
202 Accepted
ETag: "scene-version-18"
```

```json
{
  "scenario_id": "scn_01J7B4...",
  "scene_version": 18,
  "exact_result_version": 17,
  "status": "refining",
  "job_id": "job_01J7B6...",
  "coalescing_window_ms": 500,
  "status_url": "/api/v1/jobs/job_01J7B6..."
}
```

## Version conflict

When `base_scene_version` or `If-Match` is stale:

```text
409 Conflict
```

```json
{
  "error": {
    "code": "scene_version_conflict",
    "message": "The scenario changed after the client's base version.",
    "current_scene_version": 19,
    "scenario_url": "/api/v1/scenarios/scn_01J7B4..."
  }
}
```

The client reloads authoritative state and decides whether to reapply its local edit.

### Site identity mismatch (409)

A scenario pins the site identity (site id, cache/tile fingerprint) it was
created with. If the deployment's live site no longer matches — cache
rebuilt, tile swapped — every scenario mutation is refused:

```json
{
  "error": {
    "code": "site_identity_mismatch",
    "message": "the scenario is pinned to a different version of site 'campus-1km-v1'",
    "pinned_identity": {"site_id": "campus-1km-v1", "tile_key": "..."},
    "current_identity": {"site_id": "campus-1km-v1", "tile_key": "..."}
  }
}
```

This is permanent for the scenario: clients fail fast on further mutations
(no retry loop) and recover by creating a fresh scenario against the current
site. Administrative re-pointing of a results root to a new site cache is an
explicit operation (`rebase_results_root`), never an implicit reinterpretation.

## Job status

```text
GET /api/v1/jobs/{job_id}
```

Queued response:

```json
{
  "job_id": "job_01J7B6...",
  "scenario_id": "scn_01J7B4...",
  "target_scene_version": 18,
  "status": "queued",
  "queue_position": 1,
  "progress": null
}
```

Running response:

```json
{
  "job_id": "job_01J7B6...",
  "target_scene_version": 18,
  "status": "running",
  "stage": "time_loop",
  "progress": {
    "completed_time_steps": 9,
    "total_time_steps": 24
  },
  "mode": "local",
  "window": {
    "row_start": 96,
    "row_stop": 288,
    "col_start": 160,
    "col_stop": 352
  }
}
```

Complete response:

```json
{
  "job_id": "job_01J7B6...",
  "target_scene_version": 18,
  "status": "complete",
  "result_manifest_url": "/api/v1/scenarios/scn_01J7B4/results/18",
  "metrics": {
    "duration_ms": 12840,
    "peak_rss_bytes": 812646400,
    "mode": "local",
    "window_fraction": 0.147,
    "read_window_fraction": 0.312,
    "svf_seconds": 41.8,
    "time_loop_seconds": 96.4,
    "mean_utci_delta_c": -0.84,
    "peak_utci_delta_c": -3.42,
    "improved_area_m2": 8940
  },
  "impact_plan": {
    "schema_version": 1,
    "status": "executed",
    "mode": "local",
    "scene_revision": 18,
    "job_id": "job_01J7B6...",
    "nodes": [
      {
        "node": "meteorology",
        "stage": "changed",
        "why": "edited family input"
      },
      {
        "node": "radiation",
        "stage": "recomputed",
        "why": "downstream of meteorology",
        "spatial_scope": "full"
      },
      {
        "node": "tmrt",
        "stage": "reused",
        "why": "patch cache hit"
      }
    ],
    "routing": {
      "mode": "windowed",
      "write_windows": 1,
      "transport_window": "local",
      "fallback_reason": null,
      "fallback_reasons": []
    },
    "estimates": { "work_units": 96, "memory_bytes": 812646400 },
    "realized": {
      "patch_count": 1,
      "published_variables": ["utci", "tmrt"],
      "unpublished_variables": [],
      "diagnostics": []
    }
  }
}
```

Terminal statuses are `complete`, `superseded`, `failed`, and `cancelled`.

Solve telemetry in `metrics` (all additive; older fields keep their meaning):
`window_fraction` is the write-window union as a site-area fraction;
`read_window_fraction` (perf wave 1) is the same fraction for the read union —
the influence halo the solve actually touched, so `read_window_fraction >=
window_fraction` always holds and the ratio itself quantifies halo overhead.
`svf_seconds` / `time_loop_seconds` (perf wave 1) are the two dominant solve
stages measured separately; they are omitted when unavailable (the legacy
full-tile bootstrap has no per-stage seams). An explicit `time_indices`
request truncates the solve to its causal time prefix `t = 0..max(index)` —
bit-identical per step to the full-series run, since temporal state only
carries forward.

### Executed impact plan

Executor-path jobs (any family edit) serve the plan the planner actually
executed, on any terminal job body (`impact_plan`, plan schema 1). Per node:
`stage` is `changed` (an edit wrote it), `recomputed` (regenerated), or
`reused` (served from cache), with the planner's own `why` string and, when
applicable, `spatial_scope` (`windows` or `full`) plus temporal scope.
`routing` reports the realized routing (mode, write-window count, transport
window, fallback reasons); `estimates` vs `realized` gives planned work
against what actually published. Tree-only jobs carry no `impact_plan`;
clients derive their stage classification from the capability document's
dirty-node closures as a fallback. Nodes the plan does not list were reused
by construction.

## Result manifest

```text
GET /api/v1/scenarios/{scenario_id}/results/{scene_version}
```

```json
{
  "schema_version": 1,
  "scenario_id": "scn_01J7B4...",
  "scene_version": 18,
  "exact": true,
  "model_version": "solweig-gpu-2.0.0+incremental.1",
  "site_cache_version": "campus-1km-v1:cache-1",
  "window": {
    "row_start": 96,
    "row_stop": 288,
    "col_start": 160,
    "col_stop": 352
  },
  "time_indices": [12],
  "variables": [
    {
      "name": "utci",
      "dtype": "float32",
      "shape": [1, 192, 192],
      "nodata": "nan"
    },
    {
      "name": "tmrt",
      "dtype": "float32",
      "shape": [1, 192, 192],
      "nodata": "nan"
    }
  ],
  "payload_url": "/api/v1/scenarios/scn_01J7B4/results/18/payload",
  "compression": "zstd",
  "checksum": "sha256:...",
  "metrics": {
    "read_window_fraction": 0.312,
    "svf_seconds": 41.8,
    "time_loop_seconds": 96.4,
    "mean_utci_delta_c": -0.84,
    "peak_utci_delta_c": -3.42,
    "improved_area_m2": 8940
  },
  "statistics": {
    "utci": [
      { "time_index": 12, "count": 36864, "min": 18.2, "max": 41.5, "mean": 27.9 }
    ]
  },
  "limitations": [
    "Tree-induced local wind-field changes are not recomputed."
  ]
}
```

`statistics` carries per-variable, per-time-step min/max/mean/count over the
published window. It exists so legends and color scales are served from
metadata alone — a client never decodes the payload just to draw a legend.

## Binary payload

```text
GET /api/v1/scenarios/{scenario_id}/results/{scene_version}/payload
Accept: application/vnd.solweig.patch+zstd
```

Response headers:

```text
Content-Type: application/vnd.solweig.patch+zstd
Content-Length: ...
ETag: "sha256-..."
X-SOLWEIG-Scene-Version: 18
X-SOLWEIG-Schema-Version: 1
```

The uncompressed bytes follow the order documented in [Data model](data_model.md).

### Content negotiation (added post-review)

No shipping browser exposes a zstd `DecompressionStream` (caniuse 2026-09-01:
0%; Node 26.8.1 supports gzip/deflate only), so the server also serves an
identity encoding:

- `Accept: application/vnd.solweig.patch+zstd` (also `*/*`, `application/*`,
  or no header) → `Content-Type: application/vnd.solweig.patch+zstd`.
- `Accept: application/vnd.solweig.patch+identity` with zstd not acceptable →
  `Content-Type: application/vnd.solweig.patch+identity`; the body is the raw
  uncompressed byte stream (same layout as above).
- Neither acceptable → `406 not_acceptable`.

`ETag` and `X-SOLWEIG-Checksum` are computed over the SERVED bytes for the
selected encoding, so a validator obtained for one encoding must not be
replayed against the other (the server emits distinct validators per
encoding). The result manifest's `checksum` field always records the zstd
bytes.

## Server events

Polling is sufficient for the first release. Optional Server-Sent Events:

```text
GET /api/v1/scenarios/{scenario_id}/events
Accept: text/event-stream
```

Example events:

```text
event: job-progress
data: {"job_id":"job_...","scene_version":18,"stage":"time_loop","completed":9,"total":24}

event: result-ready
data: {"scene_version":18,"result_manifest_url":"/api/v1/scenarios/.../results/18"}
```

The browser still validates versions after receiving an event.

## Reset scenario

```text
POST /api/v1/scenarios/{scenario_id}/reset
If-Match: "scene-version-18"
Idempotency-Key: reset-18
```

Reset is another versioned mutation. It removes editable trees and creates an exact baseline result without running the worker when no other dynamic state exists.

## Export

```text
POST /api/v1/scenarios/{scenario_id}/exports
```

```json
{
  "scene_version": 18,
  "format": "cog",
  "variables": ["utci", "tmrt"],
  "time_indices": [12]
}
```

Exports can be asynchronous. Do not force the browser visualization patch format to serve scientific GIS export requirements.

## Error envelope

```json
{
  "error": {
    "code": "invalid_tree_geometry",
    "message": "canopy_diameter_m must be between 1 and 30",
    "field": "edits[0].tree.canopy_diameter_m",
    "request_id": "req_01J7..."
  }
}
```

Required codes include:

- `invalid_request`;
- `scenario_not_found`;
- `scene_version_conflict`;
- `invalid_tree_geometry`;
- `unsupported_transport` (universal transport: tree-only or not-integrated adapter);
- `invalid_edit_state` (422: registry/adapter refusal, message verbatim);
- `invalid_family_batch` (a view-only item mixed into a family batch);
- `site_identity_mismatch` (409: scenario pinned to a different site identity);
- `idempotency_key_reused` (409: same key, different content);
- `view_not_available` (409: layer not in the published result);
- `site_limit_exceeded`;
- `job_not_found`;
- `job_failed`;
- `result_not_ready`;
- `cache_version_mismatch`;
- `rate_limited`.

## Idempotency

The server stores `(scenario_id, idempotency_key)` with the resulting response. Repeating the same request returns the same mutation result. Reusing a key with different content returns `409 idempotency_key_reused`.

Stored idempotency records are garbage-collected after a retention window (24 hours by default; `sweep_retention`). A client that retries a mutation with the same key after its record has been swept does not get the stored response back — the request re-executes and publishes a **new** scene version. This is the Stripe-style semantics: idempotency protects against duplicated in-flight/short-term retries (network timeouts, double clicks, client restarts within the retention window), not a guarantee of permanent result storage. Long-running or resumed clients should treat a new scene version after a retry as the signal that the original record expired, and re-read scenario state rather than assuming the first attempt was lost.

## OpenAPI implementation guidance

Use typed request and response models. Generate the client types or validate them in shared tests. The API layer should convert normalized UV coordinates to projected world coordinates through the site manifest, not through frontend constants.
