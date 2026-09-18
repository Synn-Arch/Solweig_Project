# End-to-end implementation and runtime pipelines

## A. Development pipeline

```text
clone/sync
  -> environment fingerprint
  -> reproduce green baseline
  -> phase task selection
  -> characterization/failing test
  -> implementation
  -> T0 static
  -> T1 unit
  -> T2 integration
  -> T3 scientific differential when scientific path touched
  -> T4 benchmark when performance touched
  -> diff review
  -> docs/worklog checkpoint
  -> focused commit
  -> next task
  -> ...
  -> T5 full repository
  -> T6 deployment smoke
  -> final benchmark report
  -> push/PR/main according to permissions
```

A failure routes back to the smallest responsible stage. It does not skip forward.

## B. Runtime edit pipeline

```text
browser drag
  -> local visual preview only
pointer-up / debounced parameter edit
  -> POST edit with base scene_revision
API validates edit
  -> increments scene_revision
  -> coalesces queued edits
  -> computes conservative dirty region
worker checks local-safety predicate
  -> local read/write windows OR full-tile fallback
load warm immutable baseline views
  -> rasterize current editable trees affecting read window
  -> recompute dynamic vegetation visibility/SVF
  -> replay required temporal state
  -> compute exact Tmrt/UTCI patch
  -> encode staged patch
publication transaction checks scene_revision
  -> stale? discard as SUPERSEDED
  -> current? atomically publish patch + metadata
client receives result
  -> verifies job_id + revision
  -> patches only returned texture/raster region
  -> marks state EXACT
```

## C. Cache-build pipeline

```text
fixed site inputs
  -> validate projection/grid/shape
  -> generate or ingest wall/aspect
  -> full scientific preprocessing
  -> separate immutable building-only state
  -> convert binary visibility to packed layout
  -> write memory-mapped arrays
  -> hash every artifact
  -> write manifest.partial
  -> cache self-test
  -> atomic activate manifest
  -> warm worker opens read-only mappings
```

## D. Scientific validation pipeline

```text
baseline site + deterministic edit fixture
      |                         |
      v                         v
full-domain oracle         incremental pipeline
      |                         |
      +-----------+-------------+
                  v
            aligned comparison
                  |
        +---------+----------+
        |         |          |
      interior  boundary   outside write window
        |         |          |
        +---------+----------+
                  v
        metrics + mismatch map
                  |
          pass tolerance?
             |          |
            yes         no
             |          |
      record evidence   minimize fixture -> diagnose -> regression test
```

## E. Release pipeline

A release candidate must have:

1. all mission.yaml mandatory gates passing;
2. no unresolved scientific mismatch;
3. clean full repository tests in the supported environment;
4. documentation build;
5. frontend tests and deployment smoke;
6. target CPU benchmark with RSS and latency;
7. limitation disclosure;
8. reproducible commit SHA and cache/model schema versions.
