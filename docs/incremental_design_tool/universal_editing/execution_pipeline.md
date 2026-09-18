# Universal incremental execution pipeline

## Runtime sequence

```text
user edit
 -> frontend preview via adapter metadata
 -> versioned EditCommand
 -> server validation and durable event
 -> coalesce compatible edits
 -> adapter source deltas
 -> dependency-graph resolution
 -> per-node spatial/temporal impact
 -> local/full safety and memory planner
 -> execute invalidated stages in topological order
 -> stage result patches
 -> verify current scene revision
 -> atomic publication
 -> frontend exact patch and state transition
```

## Heterogeneous edit batching

A single user action or debounce interval may include multiple source families, for example:

- add a canopy and change the pavement below it;
- move a building mass and adjust selected time;
- change weather while several geometry edits are pending.

The planner combines source deltas first, then resolves dependencies once. It must not run one complete pipeline per object.

## Coalescing rules

- repeated replacement of the same object retains first old and final new state;
- paint strokes of the same source class may union masks or retain ordered last-write-wins semantics;
- forcing edits to the same variable/time keep the last validated value;
- add then delete before exact execution is a no-op;
- edits to different source nodes remain separate deltas but share one dependency plan;
- a running result becomes stale when a newer scene revision is committed.

## Planner output

The planner produces a topologically ordered stage plan such as:

```yaml
scene_revision: 28
source_deltas:
  - adapter: vegetation_object
    windows: [[96, 224, 140, 292]]
  - adapter: landcover_brush
    windows: [[160, 212, 180, 260]]
stages:
  - node: vegetation_visibility
    spatial: windows
    read_windows: [...]
  - node: svf
    spatial: windows
  - node: surface_thermal_state
    spatial: windows
    temporal: replay
    time_start: 0
  - node: tmrt
    spatial: windows
    temporal: replay
  - node: utci
    spatial: windows
    temporal: requested
reused:
  - walls
  - wall_aspect
  - building_visibility
  - forcing
```

If selected time or meteorology also changes, downstream spatial scope may become full while upstream local geometry stages remain windowed. The planner must support mixed scopes in one job.

## Execution engine

The engine receives source overlays and immutable baseline cache views. It executes only invalidated nodes in graph order. A stage API should accept:

- read and write windows;
- temporal range or replay checkpoint;
- source and upstream derived views;
- requested outputs;
- scratch allocator;
- supersession token;
- metrics collector.

## Publication

Each output patch includes:

- scenario and scene revision;
- model, site-cache, dependency-graph, and adapter versions;
- source edit IDs;
- variable and timestep;
- half-open write window;
- dtype, nodata, compression, and checksum;
- exact/preview flag;
- limitations, including fixed wind field when applicable.

The browser applies a patch only when its revision matches current scenario state.

## Collaborative deadline behavior

The original asynchronous pipeline remains the exact reconciliation path. Multi-session real-time behavior is defined by the [collaborative real-time extension](../realtime_collaboration/index.md):

1. durably accepted operations enter 100 ms per-workspace epochs;
2. heterogeneous edits reduce into one canonical revision and dependency plan;
3. a CPU-reserved fast lane publishes `fast_exact`, `fast_qualified`, or `visual_pending` within the admitted one-second SLO;
4. a separate exact queue revision-chases the latest canonical workspace snapshot;
5. supersession removes obsolete compute targets, never accepted operations;
6. exact correction patches publish only when revision fences pass.

Polling or server events report job/revision state; polling never triggers solver execution. The fast lane must remain available while an exact geometry job takes tens of seconds.
