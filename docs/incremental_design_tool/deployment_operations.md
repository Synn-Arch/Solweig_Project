# Deployment and operations

## Initial deployment target

Use one Linux CPU node with:

- 2 to 4 vCPU;
- 4 GB RAM preferred;
- local SSD or persistent disk;
- one public HTTPS endpoint;
- one API process and one scientific worker process;
- no GPU requirement;
- no distributed queue requirement.

A 1 GB free instance is a later optimization target, not the first reliable classroom deployment.

## Process model

```text
reverse proxy (optional)
└── API service
    ├── static frontend
    ├── REST endpoints
    ├── SQLite connection
    └── worker supervisor / IPC
        └── scientific worker
```

The worker can communicate through a multiprocessing queue plus durable job rows. The durable store is the source of truth. In-memory messages are only wake-up signals.

## Suggested filesystem

```text
/opt/solweig-app/
├── application/
├── cache/<site_id>/
├── state/app.sqlite3
├── results/<scenario_id>/<scene_version>/
├── exports/
├── logs/
└── tmp/
```

Mount `cache` read-only in the worker when possible. Write staged patches under the same filesystem as final results so atomic rename is available.

## Configuration

The server boots from environment variables via `python -m solweig_gpu.server`
(`solweig_gpu/server/__main__.py`), which maps them onto `create_app(...)`
keyword arguments. This section describes implemented behavior only: unset
optional variables keep the `create_app` defaults, an invalid value refuses
to boot with an error naming the variable and value (exit code 2), and an
unknown `SOLWEIG_*` variable prints a stderr warning and is ignored.

### Implemented environment variables

| Variable | Default | Maps to |
| --- | --- | --- |
| `SOLWEIG_STATE_ROOT` | `./state` | `create_app(state_root=...)`; the SQLite store lives at `<state_root>/store.sqlite3` and per-scenario results under `<state_root>/scenarios/` |
| `SOLWEIG_SITE_ID` | *(required)* | keys of `create_app(sites=...)`; comma-separated list for multiple sites |
| `SOLWEIG_CACHE_ROOT` | *(required with `SITE_ID`)* | each site's `cache_dir`, resolved as `<CACHE_ROOT>/<site_id>/` |
| `SOLWEIG_SITE_DIR_<ID>` | *(unset)* | `sites[<id>]["site_dir"]` |
| `SOLWEIG_SELECTED_DATE_STR_<ID>` | *(unset; `SiteConfig` default `2024-06-20`)* | `sites[<id>]["selected_date_str"]` |
| `SOLWEIG_HOST` | `0.0.0.0` | uvicorn bind host |
| `SOLWEIG_PORT` | `8000` | uvicorn bind port |
| `SOLWEIG_MAX_TREES_PER_SCENARIO` | `200` (create_app) | `max_trees_per_scenario` |
| `SOLWEIG_COALESCE_MS` | `500.0` (create_app) | `coalescing_window_ms` |
| `SOLWEIG_EDITS_PER_MINUTE` | `20` (create_app) | `edits_per_minute` (per-scenario mutation budget) |
| `SOLWEIG_REQUESTS_PER_MINUTE` | `30` (create_app) | `requests_per_minute_per_ip` (in-process; enforce at the proxy for multi-process deployments) |

`<ID>` is the site id upper-cased with every non-alphanumeric character
replaced by an underscore (`campus-1km-v1` → `CAMPUS_1KM_V1`). There is no
keyword form for disabling a limit: empty/unset means the default and only
plain integers are accepted. `0` passes through as the integer 0, which
`create_app` itself treats as disabled for the rate-limit knobs.

### Process-env-only variables (honored by C libraries)

These are never read by the Python boot code but are respected by the
compiled libraries for the process lifetime; set them in the container or
systemd environment:

| Variable | Effect |
| --- | --- |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | OpenMP / BLAS thread pools |
| `NUMBA_NUM_THREADS` | Numba JIT thread pool |
| `GDAL_CACHEMAX` | GDAL raster block cache (bytes, or MiB with an `M` suffix) |

Thread values must be tuned to the VM. Prevent multiple libraries from each
using all cores.

### Not implemented (formerly documented — do not set)

The original configuration draft listed variables no code reads; setting
them today only triggers the unknown-variable warning. Use the implemented
variables above instead:

- `SOLWEIG_STATE_DB` / `SOLWEIG_RESULT_ROOT` — the state layout derives
  from `SOLWEIG_STATE_ROOT` alone;
- `SOLWEIG_JOB_COALESCE_MS` — superseded by `SOLWEIG_COALESCE_MS`;
- `SOLWEIG_WORKER_COUNT` — the deployment is one API process with one
  in-process worker;
- `SOLWEIG_SCIENTIFIC_THREADS` — set `NUMBA_NUM_THREADS` instead;
- `SOLWEIG_FULL_RECOMPUTE_FRACTION` — full-recompute fallback is decided
  inside the executor, not configured;
- `SOLWEIG_SCENARIO_TTL_HOURS` / `SOLWEIG_RESULT_TTL_HOURS` — retention is
  store-internal (`Store.sweep_retention`); there is no operator-configured
  scenario/result TTL today.

### Container deployment

`Dockerfile`, `docker-compose.yml`, and `fly.toml` at the repo root hold a
working single-node CPU shape that also ports to any container host:

- API image: `python:3.14-slim`, CPU-only torch wheels, the `osgeo` GDAL
  bindings compiled in a builder stage against Debian's `libgdal` (no Linux
  wheels exist, so the runtime stage carries `libgdal36`), non-root runtime
  user, `EXPOSE 8000`, `CMD ["python", "-m", "solweig_gpu.server"]`.
- Volumes: state **read-write** (`/state` with `SOLWEIG_STATE_ROOT=/state`),
  site caches **read-only** (`/site-cache` with
  `SOLWEIG_CACHE_ROOT=/site-cache`, one `/site-cache/<site_id>/` per site).
  Site data never ships inside the image — `.dockerignore` excludes every
  site-data location in the repository.
- Frontend: a second process running
  `studio/serve.py --api http://api:8000` for the
  same-origin `/api` proxy (the compose file includes that service).
- Numba kernels are compiled without `cache=True`, so expect a few seconds
  of JIT on the first edit after each process start.

## Startup sequence

1. Validate environment and writable directories.
2. Open and migrate the scenario database.
3. Load the site manifest.
4. Validate all cache checksums and array shapes.
5. Memory-map baseline arrays.
6. Run a small cache self-test.
7. Start the scientific worker.
8. Recover queued or running jobs from the previous process.
9. Start accepting HTTP traffic.
10. Report readiness only after baseline cache validation succeeds.

## Health endpoints

### Liveness

```text
GET /health/live
```

Returns success when the API process event loop is responsive.

### Readiness

```text
GET /health/ready
```

Returns success only when:

- database is accessible;
- site cache is validated;
- worker heartbeat is recent;
- result filesystem is writable;
- no unrecoverable schema mismatch exists.

### Worker diagnostics

```text
GET /health/worker
```

Restricted in production. Include worker PID, revision, cache version, idle/running state, current job, RSS, and last successful job time.

## Logging

Use structured JSON logs with:

```text
timestamp
level
request_id
scenario_id
scene_version
job_id
stage
window
mode
duration_ms
rss_bytes
message
```

Do not log entire tree arrays or binary patches. Tree positions may be logged at debug level only if privacy policy permits.

## Metrics

Minimum metrics:

- HTTP request count, latency, and errors;
- scenario count;
- queued, running, complete, failed, and superseded jobs;
- queue wait time;
- scientific duration by stage;
- local versus full mode count;
- dirty-window fraction;
- worker RSS;
- output patch bytes;
- cache warm-up duration;
- worker restarts.

A lightweight `/metrics` endpoint or periodic log aggregation is sufficient initially.

## Job scheduling policy

Recommended classroom policy:

1. Coalesce updates for the same scenario for 500 ms.
2. Keep only the latest pending target version per scenario.
3. Use round-robin fairness across scenarios.
4. Give an optional instructor/demo scenario higher priority.
5. Do not preempt a native kernel mid-stage.
6. Check supersession between rasterization, visibility, time-loop blocks, and serialization.

## Persistence and cleanup

- Store tree events and scenario metadata in SQLite.
- Store patches as versioned files with checksums.
- Keep baseline cache immutable.
- Expire inactive anonymous scenarios after the configured TTL.
- Retain the latest exact result and a small number of prior versions for debugging.
- Remove abandoned temporary directories at startup.
- Vacuum SQLite during low-use periods, not during a live class.

### Schema upgrades and family routing (v5/v6)

Schema v5 adds `scenarios.carries_family_edits`, the durable routing signal
that survives retention pruning of family edit events (u-e1 F1). On upgrade
the migration backfills the flag from any family events that still survive
in the ledger.

Schema v6 closes the residual upgrade window (u-e3c attack C): a pre-v5
database whose family events were **already pruned** past the retention
tail upgrades with the flag unset and no surviving family event — no
backfill can see state that no longer exists — yet such a scenario's
executor-state directory (`results_root/<scenario>/executor-state/`)
proves it ran family jobs. The routing reader now also arms on that
directory when its coverage postdates the scenario's
`last_reset_sequence` (added by v6, backfilled from surviving reset
events, written by every reset going forward): the rescue lifts the
post-reset state exactly, and never resurrects a pre-reset snapshot,
whose family overlays the reset voided.

Operational caveat: a scenario whose RESET event was pruned by retention
before a v6 upgrade keeps watermark 0, so its executor state cannot be
distinguished from a never-reset scenario and the rescue may route its
next tree edit onto the executor, which would restore the pre-reset
snapshot. Reaching that corner requires the retention tail's worth of
published edits between the reset and the upgrade. If an operator
suspects it, one reset of the affected scenario after upgrading re-arms
the watermark and re-publishes the clean baseline.

One further compound corner of the same class: the routing reader trusts
the executor-state coverage documents slightly more broadly than the
bridge does (it reads a staged pending's watermark without the producing
job's completion gate, and a sidecar whose named generation directory was
lost). Losing a generation directory while the family ledger is fully
pruned on an attack-C row can publish baseline+trees without family state
— bit-identical to the pre-fix legacy misroute, i.e. no regression, but
the operator remedy is the same: reset the affected scenario to re-arm
clean state.

## Backup and recovery

For a teaching demo, baseline cache can be recreated from versioned inputs. Back up:

- site manifest and cache-generation script;
- scenario database if student work must persist;
- exported results;
- deployment configuration excluding secrets.

Recovery sequence:

1. Restore application and baseline cache.
2. Restore database.
3. Validate result checksums.
4. Mark interrupted jobs queued.
5. Restart worker and API.

## Deployment modes

### Local instructor laptop

Suitable for a single-user demonstration. Serve API and worker on localhost. Use the same job contract so cloud deployment does not require a frontend rewrite.

### Small cloud VM

Recommended for a class site. Use HTTPS, persistent disk, process supervision, resource limits, and scenario TTL.

### Static frontend plus remote worker

Possible later. The API contract remains unchanged. Ensure CORS, authentication, and binary patch caching are configured.

## Resource limits

Set process memory and CPU limits slightly below node capacity. On a 4 GB node:

- reserve at least 0.7 to 1.0 GB for OS, API, filesystem cache, and supervision;
- target worker peak below 3.0 GB;
- reject or full-stream jobs whose estimated memory exceeds the limit;
- restart worker after an out-of-memory exit.

## Security controls

- HTTPS in public deployment;
- opaque scenario IDs;
- request body limit;
- per-IP and per-scenario rate limits;
- numeric validation and finite checks;
- maximum tree count and property bounds — `create_app` defaults to
  `max_trees_per_scenario=200` with per-deployment overrides; set it
  explicitly for classroom sizing rather than relying on the default
  (p9-rel evidence note, 2026-09-02);
- no user-controlled filesystem paths;
- restrictive export filenames;
- content security policy for the frontend;
- dependency scanning and pinned deployment environment.

## Demo-day operating procedure

Before class or presentation:

1. Restart the service.
2. Verify readiness and worker heartbeat.
3. Run one representative tree edit.
4. Confirm exact result time and memory.
5. Reset the demo scenario.
6. Preload the frontend and baseline textures.
7. Keep a baseline screenshot and exported result as fallback.
8. Monitor queue, worker RSS, and errors during the session.

## Capacity reasoning

One worker is sufficient when edits are infrequent relative to job duration. For example, 20 students editing once every two minutes produces an average arrival rate of one job every six seconds before coalescing. If a typical exact job consumes 10 seconds of CPU, one worker cannot keep every scenario current. Options are:

- coalesce more aggressively;
- provide immediate preview and slower exact refinement;
- prioritize the active scenario;
- use two workers on a larger node;
- constrain the class to shared scenarios or staged exercises.

Measure actual interaction rates before choosing concurrency.
