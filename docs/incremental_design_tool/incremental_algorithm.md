# Incremental scientific algorithm

## Goal

Given an immutable baseline site and an edit to one or more user-controlled trees, compute the same scientific output as a full-domain run while evaluating only a conservative spatial window whenever possible.

The algorithm must prioritize correctness over locality. A local update is an optimization, not a different model. Any condition that invalidates the local assumptions triggers full-tile recomputation.

## Core invariants

1. Moves and deletes invalidate the old influence region.
2. Adds and updates invalidate the new influence region.
3. The exact calculation inside the dirty window uses the complete current tree set that can affect that window, not only the edited tree.
4. The computation window includes a halo. Only the interior write window is published when boundary conditions require a larger read window.
5. Time-dependent surface states are replayed from the required warm-up point.
6. A local patch is compared against the same baseline, cache version, and forcing as the full solver.
7. Results outside the declared write window are unchanged.

## Stage 0: baseline preparation

Before interactive serving, compute and validate the static site package:

- DEM and building DSM;
- base vegetation DSM;
- wall height and wall aspect;
- land cover;
- building-only visibility or shadow patch masks;
- base vegetation visibility masks;
- SVF and directional SVF;
- meteorological forcing and solar position for every supported timestep;
- baseline Tmrt and UTCI;
- valid pedestrian-cell mask.

The static package is created with the existing GPU or CPU pipeline offline. Interactive requests must not rebuild wall or building geometry.

## Stage 1: coalesce edits

Rapid edits are reduced by tree ID.

Examples:

```text
add A -> move A -> resize A       becomes add A(final)
move A -> move A                  becomes move A(first_old, final_new)
add A -> delete A before analysis becomes no-op
update A -> delete A              becomes delete A(first_old)
```

The implemented `coalesce_tree_edits` function provides this semantic reduction. It does not decide job cancellation or persistence.

## Stage 2: compute conservative influence regions

### Direct shadow length

For tree height `H` and solar altitude `alpha`:

```{math}
L = H / tan(alpha)
```

The direct shadow extends opposite the solar azimuth. Use the model's azimuth convention consistently: azimuth is clockwise from north and points toward the sun.

Low sun can produce extremely long theoretical shadows. Configure:

- `minimum_direct_sun_altitude_deg`, initially 5 degrees;
- `maximum_shadow_length_m`, initially 300 m.

The cap is a deployment policy and must be validated against the site's extent and supported times. If the application includes lower sun angles where the cap would truncate meaningful influence, force full recomputation or increase the cap.

### Sky-view influence radius

For the lowest sky patch altitude `beta`:

```{math}
R_svf = H / tan(beta)
```

With `beta = 6 degrees`, a 10 m tree has a geometric radius of approximately 95 m and an 18 m tree approximately 171 m. This radius is conservative because canopy geometry and terrain can reduce visibility, but the invalidation stage must not assume that reduction.

### Tree influence bound

For each old and new tree:

1. Start with a circle centered on the trunk with radius `max(canopy_radius, R_svf)`.
2. For each relevant timestep, project the direct-shadow endpoint.
3. Add a corridor around the trunk-to-endpoint segment with radius equal to canopy radius plus safety margin.
4. Add a configurable safety margin.
5. Convert the world bound to a half-open raster window.
6. Expand to cache block boundaries.
7. Clip to the site.

The current conservative implementation returns an axis-aligned bounding window. A later optimization may use several smaller windows or a polygon mask, but the rectangular path is easier to validate and stream.

### Edit dirty window

```text
dirty(edit) = influence(old_tree) ∪ influence(new_tree)
```

For a batch:

```text
dirty(batch) = merge(dirty(edit_i), merge_gap)
```

Nearby windows should be merged when the saved repeated setup cost exceeds the additional cells.

## Stage 3: choose local or full mode

Use local mode only when all conditions hold:

- dirty area fraction is below the configured threshold, initially 30 percent;
- all required cache arrays support windowed access;
- the selected time range and warm-up state can be reconstructed locally;
- the dynamic tree count within the influence query is below a configured cap;
- the dirty window and halo fit memory budget;
- no cache schema mismatch exists;
- no unsupported physics option is enabled.

Otherwise, use full-tile mode.

The area threshold must be benchmarked. A 30 percent rectangle can cost more than 30 percent of a full run if setup and cache access dominate.

## Stage 4: derive read and write windows

Keep two windows:

```text
write_window = dirty window whose result is published
read_window  = write_window expanded by numerical and geometric halo
```

The halo must cover:

- raster shifts used by ray casting;
- interpolation or rotation kernels;
- neighboring cells used by wall and surface terms;
- shadow sources outside the write window that cast into it;
- tree crowns overlapping the boundary;
- chunk alignment.

The scientific solver receives the read window. After computation, discard halo cells and publish only the write window.

A simple first implementation may set `write_window == read_window` with a conservative halo included in both. This is correct but sends a larger patch. Split them once validation fixtures exist.

## Stage 5: query affecting trees

Maintain a spatial index over base and user trees. Query all tree crowns and trunks whose influence geometry intersects the read window. Do not rasterize only the edited trees.

For the first implementation, the scenario tree count may be small enough for a linear scan. Preserve the query interface so an R-tree or uniform grid can replace it later.

## Stage 6: rasterize dynamic vegetation

Reconstruct the vegetation inputs for the read window:

```text
current vegetation = immutable base vegetation + current scenario tree objects
```

A tree component should rasterize at least:

- canopy height or top DSM;
- trunk-zone height, using `trunk_ratio`;
- canopy footprint;
- transmissivity or vegetation class metadata.

When crowns overlap, use the same combination rule as the full input construction. Commonly the top surface uses maximum height. Never add heights.

Rasterization must be deterministic across full and local paths. Share one implementation and test it with translated windows.

## Stage 7: update visibility and SVF

### Static and dynamic split

Building-only visibility is immutable. Tree edits affect vegetation and combined visibility. The local worker should:

1. map the building visibility bits for the read window;
2. reconstruct vegetation visibility for current trees;
3. combine building and vegetation according to existing SOLWEIG semantics;
4. update vegetation-adjusted SVF terms through weighted reduction;
5. avoid materializing `[rows, cols, patches]` float arrays.

### Patch streaming

For each sky patch or a small patch block:

```python
visibility = unpack_patch(packed_building, patch_index)
veg_visibility = compute_or_unpack_dynamic_vegetation_patch(...)
combined = combine(visibility, veg_visibility)
accumulate_svf(combined, patch_weight)
release_patch_temporaries()
```

The exact combination must be taken from the current `shadow.py` and `solweig.py` semantics. The design document intentionally does not rename polarity. Add tests before refactoring because `1` can mean visible, sunlit, or unshadowed depending on the array.

## Stage 8: replay temporal state

The existing model carries state such as ground-temperature maps between timesteps. Therefore a visible-hour update cannot generally evaluate only one independent timestep.

Define a warm-up policy:

- For a 24-hour teaching day, replay all timesteps from the first forcing row through the latest requested hour inside the read window.
- For full-day metrics, replay all 24 timesteps.
- If future forcing spans several days, begin from a documented state checkpoint or the start of the required spin-up period.

Static forcing values can be preloaded. Spatial state arrays are allocated only for the read window.

## Stage 9: local SOLWEIG evaluation

The target worker interface should resemble:

```python
result = run_incremental_window(
    site_cache=site_cache,
    scenario_trees=current_trees,
    read_window=read_window,
    write_window=write_window,
    time_start=0,
    time_stop=24,
    requested_variables=("utci", "tmrt"),
    cancellation_token=token,
)
```

The implementation should reuse scientific equations from the current solver. It should not maintain a separate approximate Python formula for production.

### Required refactoring boundaries

- Input arrays must be accepted as window views rather than reopened file paths.
- Output selection must prevent allocation of unrequested variables.
- Time-step results must stream to a patch writer or preallocated output, not Python lists.
- Building and base cache arrays must remain read-only.
- CPU backend selection and thread count must be explicit.

## Stage 10: publish patch atomically

A job writes to a temporary location:

```text
scenario/<id>/results/.tmp/<job_id>/...
```

After checksum and metadata validation:

1. begin a store transaction;
2. verify target scene version is still publishable;
3. rename or move the patch to the versioned result location;
4. update `exact_result_version`;
5. mark the job complete;
6. commit the transaction.

The client applies the patch only when its `scene_version` matches the client's current scene version. Older exact patches may be cached but not displayed as current.

## Reference pseudocode

```python
def execute_latest_job(scenario_id: str) -> None:
    scenario = store.load_scenario(scenario_id)
    events = store.unprocessed_events(scenario_id)
    batch = coalesce_tree_edits(events)

    if not batch.edits:
        store.mark_events_processed(events)
        return

    windows = dirty_windows_for_batch(
        batch,
        grid=site.grid,
        sun_positions=site.sun_positions,
        config=site.influence_config,
    )

    for write_window in windows:
        mode = choose_recompute_mode(write_window, site.grid)
        if mode == FULL:
            write_window = site.grid.full_window

        read_window = build_read_window(write_window, site)
        trees = spatial_index.query(read_window, scenario.current_trees)

        dynamic_vegetation = rasterize_trees(
            base_vegetation=site.base_vegetation.window(read_window),
            trees=trees,
            window=read_window,
        )

        visibility = update_local_visibility(
            building_bits=site.building_visibility.window(read_window),
            base_vegetation_bits=site.base_vegetation_visibility.window(read_window),
            dynamic_vegetation=dynamic_vegetation,
        )

        patch = run_incremental_window(
            site_cache=site,
            visibility=visibility,
            vegetation=dynamic_vegetation,
            read_window=read_window,
            write_window=write_window,
            time_start=0,
            time_stop=site.time_steps,
            requested_variables=("utci", "tmrt"),
        )

        validate_patch(patch, write_window)
        result_store.stage(patch, scenario.scene_version)

    result_store.publish_if_current(
        scenario_id=scenario_id,
        target_scene_version=scenario.scene_version,
    )
```

## Edge cases

### Tree near site boundary

Clip rasterization and output to the site, but retain all in-site cells that can be affected. A tree center outside the site is rejected in the first release.

### Tall tree or low sun

If the influence bound covers most of the site, use full mode. Do not silently truncate influence to preserve latency.

### Overlapping trees

Recompute from baseline plus all current trees within the query. Do not add per-tree UTCI deltas.

### Move across a long distance

The dirty region is the union of old and new influence regions. It may produce two separated windows. Preserve separate windows when they are far apart rather than creating one large bounding rectangle.

### Delete while a previous job runs

Create a newer scene version. The previous result cannot publish as current.

### Time-slider change without geometry change

Direct-shadow direction changes. The frontend can update its preview immediately. The exact worker may reuse visibility and rerun only radiation/time evaluation if cached inputs support it.

### Several users

Each scenario shares immutable baseline caches but has separate tree state, versioning, queue state, and result patches. One worker serializes exact jobs fairly, using latest-version coalescing within each scenario.

## Correctness oracle

The oracle is full-domain recomputation from the same complete tree state. Every incremental test fixture must compare the patched result with this oracle, not with the previous incremental result.
