# System architecture

## Architectural decision

The system uses one always-on CPU service and one in-process or colocated background worker. Static geometry is precomputed and shared. Each scenario stores only editable tree objects and sparse output patches. The worker serializes exact scientific jobs, coalesces rapid edits, computes a conservative dirty region, recomputes that region using the current complete tree state, and patches the scenario result.

This architecture is appropriate because the study area is fixed and small, user edits are sparse, and update latency of seconds to roughly one minute is acceptable.

## Context diagram

*(Mermaid diagram; shown here as source text.)*

```text
flowchart LR
    U[Student or designer] -->|drag, move, resize, delete| F[Browser design tool]
    F -->|POST committed edit| A[CPU API service]
    A -->|append scene event| D[(SQLite or embedded store)]
    A -->|enqueue latest scene version| Q[In-memory durable-aware queue]
    Q --> W[Single exact worker]
    W --> C[(Shared baseline cache)]
    W -->|window patch| S[(Scenario patch store)]
    A -->|status or server event| F
    F -->|GET binary patch| A
```

## Runtime topology

The first deployment should be one process tree on one node:

```text
systemd or container runtime
└── application supervisor
    ├── HTTP API process
    │   ├── request validation
    │   ├── scenario/event persistence
    │   ├── job status endpoints
    │   └── static frontend and result delivery
    └── scientific worker process
        ├── immutable baseline arrays
        ├── packed visibility cache
        ├── one job execution loop
        └── result patch writer
```

A separate worker process is preferred over a background thread because native numerical libraries can hold the Python GIL unpredictably, a process can be restarted after an out-of-memory condition, and memory usage can be measured independently. Shared immutable data can later move to read-only memory maps.

## Component responsibilities

### Browser application

The browser owns:

- pointer-level interaction and drag previews;
- editable tree controls;
- the current client-side scene version;
- approximate direct-shadow rendering during interaction;
- rendering of baseline and exact result textures;
- stale-result rejection;
- status polling or server-event subscription;
- patch application to a texture or typed array.

The browser does not own scientific truth. A local preview must be visually labeled as preview or updating.

### API service

The API owns:

- authentication or anonymous classroom session identity;
- scenario creation and lookup;
- optimistic scene-version checks;
- validation of tree properties and site bounds;
- edit persistence before acknowledgement;
- job creation and coalescing signals;
- result metadata and binary patch delivery;
- stable idempotency behavior.

The API must acknowledge a committed edit only after the edit event is durable.

### Exact worker

The worker owns:

- baseline cache loading and validation;
- edit coalescing;
- dirty-region computation;
- local-versus-full fallback selection;
- dynamic vegetation rasterization;
- vegetation visibility and SVF updates;
- time-sequential local SOLWEIG evaluation;
- result patch serialization;
- scientific metrics and timing measurements;
- atomic publication of a result for one scene version.

### Baseline cache

The baseline cache is immutable for a deployed site and model version. It contains:

- DEM and building DSM;
- baseline vegetation raster;
- wall height and wall aspect;
- land-cover inputs;
- building-only visibility or shadow information;
- baseline vegetation visibility, if the scenario begins with existing trees;
- baseline SVF and directional SVF fields;
- meteorological forcing and precomputed solar positions;
- baseline hourly outputs;
- cache schema and model-version metadata.

### Scenario state

Scenario state is sparse and append-oriented:

- tree object map keyed by stable tree ID;
- monotonic `scene_version`;
- ordered edit events;
- current exact result version;
- pending or running job metadata;
- output chunks that differ from baseline;
- optional snapshots for fast classroom reset.

Do not copy the entire baseline raster for every student scenario. Baseline arrays are shared read-only. Scenario differences are stored as tree objects and chunk patches.

## End-to-end edit sequence

*(Mermaid diagram; shown here as source text.)*

```text
sequenceDiagram
    participant B as Browser
    participant A as API
    participant DB as Scenario store
    participant W as CPU worker
    participant C as Baseline cache

    B->>B: Render drag preview locally
    B->>A: POST edit with base_scene_version
    A->>DB: Validate and append event
    DB-->>A: scene_version = N+1
    A-->>B: 202 Accepted + job_id + scene_version
    A->>W: Signal latest pending version
    W->>DB: Read unprocessed edits
    W->>W: Coalesce edits by tree ID
    W->>W: Build old ∪ new dirty region
    W->>C: Map required static cache windows
    W->>W: Recompute dynamic vegetation and outputs
    W->>DB: Atomically publish patch for version N+1
    W-->>A: Mark job complete
    A-->>B: Result-ready event or polling response
    B->>A: GET patch for scene_version N+1
    A-->>B: Binary patch + metrics
    B->>B: Apply only if N+1 is still current
```

## Concurrency model

### Scientific jobs

Start with exactly one scientific worker and one active exact job. This avoids memory multiplication and produces stable classroom behavior. Multiple HTTP requests may run concurrently, but exact calculations are serialized.

### Edit coalescing

Edits submitted while a job is pending should be coalesced before execution. Edits submitted while a job is running should create a newer target version. The running job can finish, but its result is discarded unless it still matches the latest target version.

Do not attempt unsafe cancellation inside a PyTorch, NumPy, or Numba kernel. Use cooperative cancellation between stages and stale-result rejection at publication.

### Scene version invariant

For every scenario:

```text
baseline_version <= exact_result_version <= scene_version
```

The browser may show an exact result for an older version while a newer preview is displayed, but it must indicate that refinement is pending.

## Data ownership and mutability

| Data | Owner | Mutability | Sharing |
|---|---|---|---|
| Baseline raster inputs | Baseline cache | Immutable | Shared by all scenarios |
| Building visibility | Baseline cache | Immutable | Shared |
| Base vegetation visibility | Baseline cache | Immutable | Shared |
| Editable tree objects | Scenario store | Mutable through events | Per scenario |
| Dynamic vegetation raster | Worker scratch or chunk store | Rebuilt per dirty window | Per job/scenario |
| Exact result patches | Scenario patch store | Append by version | Per scenario |
| Browser preview | Browser | Ephemeral | Per client |

## Failure handling

### Worker crash

A job remains `queued` or becomes `retryable`. The worker process restarts, reloads read-only caches, and resumes from durable scene events. No partially written patch may be visible.

### Out-of-memory condition

The worker process must be terminated and restarted. The next attempt should either use a smaller chunk size or force a full-tile streaming path. Record peak RSS and the selected mode in the job error.

### API restart

Durable scenario events and job states remain in SQLite or another embedded store. In-memory queue state is rebuilt from jobs with nonterminal status.

### Stale result

A result whose `scene_version` is lower than the scenario's latest version can be retained for debugging but must not become the current exact result.

### Invalid edit

Reject edits that place a tree outside the site, use nonpositive dimensions, exceed configured height or crown limits, or reference a tree ID that does not exist for move, update, or delete.

## Security and classroom isolation

Even for an anonymous teaching site:

- generate opaque scenario IDs;
- limit edit payload size;
- cap tree count per scenario;
- cap active jobs per client or session;
- validate all numeric values for finiteness;
- prevent filesystem paths from entering API payloads;
- store exports outside the static application directory;
- expire inactive scenarios and patches;
- never execute user-provided Python or expressions.

## Evolution path

The architecture can later scale without changing the client contract:

1. Move the worker to a separate node.
2. Replace the embedded queue with Redis or a managed queue.
3. Store patches in object storage.
4. Add several workers partitioned by scenario.
5. Add optional GPU workers for full-domain or batch jobs.

The first implementation should not introduce these distributed components prematurely.
