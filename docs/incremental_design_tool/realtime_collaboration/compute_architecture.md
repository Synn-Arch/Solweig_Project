# Computation architecture

## Two computational lanes

### Deadline-bound fast lane

The fast lane must complete inside the admitted one-second budget. It chooses the fastest qualified path per invalidated dependency graph:

```text
cache hit
  -> exact bounded kernel
  -> exact incremental response
  -> validated compensated response
  -> visual_pending fallback
```

### Exact reconciliation lane

The exact lane runs the existing scientifically equivalent equations, incorporates all operations in the canonical target revision, and corrects the fast result. It is optimized aggressively but is not allowed to block the fast lane.

## Family-specific fast paths

### View and cached output

No scientific recomputation. Switch texture, timestep, legend, or cached variable. This should be exact in milliseconds.

### Receptor-only and UTCI-only parameters

Cache Tmrt and atmospheric fields. Evaluate the UTCI polynomial in a fused vectorized/compiled kernel on the requested window or full tile. Avoid SVF and radiation stages.

### Meteorology and selected time

Reuse geometry, walls, visibility, and SVF. Precompute solar geometry and static coefficients. For selected-hour edits, evaluate only the causally required dynamic stages. Maintain exact state checkpoints where temporal dependence requires predecessor replay.

### Land cover and surfaces

Use chunk-local source overlays, dependency-aware thermal-state checkpoints, and sparse dirty propagation. A fast compensated response can use class-transition response tables calibrated against exact replay; exact reconciliation replays the required temporal state.

### Vegetation and building geometry

This is the principal hard path. Current geometry edits nearly fill the read window and spend most time in 153-patch visibility/SVF replay and the time loop. The exact one-second strategy cannot be another full patch march. It requires persistent incremental visibility data structures.

## Exact geometry optimization target

### Persistent per-cell directional occlusion state

Maintain versioned structures that permit geometry removal as well as addition:

- packed visibility or horizon state by cell and sky patch/direction;
- top-k or ordered occluder candidates per directional ray, or equivalent contribution/refcount structure;
- object-to-affected-ray reverse index;
- chunk version and baseline checksum;
- separate building and vegetation contribution semantics.

For an edit batch:

1. Find rays/directions touched by old and new geometry.
2. Remove old object contributions using the reverse index.
3. Insert new contributions.
4. Re-evaluate only cells whose winning occluder or transmissivity composition changed.
5. Update SVF accumulators by subtracting old patch contribution and adding new contribution in the same scientific order where exact parity requires it.
6. Mark downstream radiation/thermal cells and time ranges dirty.

A simple cached previous SVF cube is insufficient for deletion/overlap correctness unless it can prove which object determined each direction.

### Direct selected-time shadow path

For the visible hour, direct shadow depends on one solar direction rather than all 153 sky patches. Compute/publish this exact layer first when possible. It can improve the fast Tmrt estimate while the broader diffuse/longwave effects remain qualified.

### Delta-SVF reuse

The existing handoff identifies delta-SVF reuse as the highest-value repeat-edit design lever. Implement it as a versioned dependency cache, not an ad hoc reuse flag. Cache keys include site, workspace exact base revision, geometry chunk versions, patch geometry version, transmissivity/phenology, and read-window provenance.

## Time-loop optimization

- Persist validated checkpoints for time-dependent surface state.
- Recompute from the earliest dirty timestep, not always zero, when causality proves it safe.
- Keep separate checkpoints per exact workspace revision or use copy-on-write chunks.
- Fuse repeated elementwise kernels and reuse preallocated scratch buffers.
- Remove CPU/Torch/NumPy copies in the hot path.
- Process only requested products while retaining dependencies required for correctness.
- Profile before Numba/C++/OpenMP changes; require independent numerical comparison.

## Response operator for the fast lane

For edit classes where exact one-second computation is not yet reached, use a locally calibrated operator anchored to the latest exact state:

```text
fast_field(revision n)
  = exact_field(exact_base)
  + validated_delta(all source changes since exact_base)
```

Possible implementations include exact lookup tables, sparse local response kernels, local Jacobians, reduced-resolution exact solves with calibrated correction, or a trained surrogate. Selection is an engineering decision gated by validation, not an assumption that machine learning is necessary.

The operator consumes the **combined epoch delta**, not independent per-operation deltas when interactions are nonlinear. Cross-family and overlap fixtures determine where composition is allowed.

## Publication and correction

A fast payload records:

- workspace, fast, and exact-base revisions;
- edit IDs/epoch;
- result class;
- dependency graph and cache/model hashes;
- spatial/time coverage;
- uncertainty/error qualification;
- exact reconciliation job ID.

When exact results arrive, publish a correction patch only if the target revision is still relevant. The frontend may blend visually, but stored numerical values switch atomically.
