# Editable scope and edit families

## Capability model

Every candidate editor capability must be classified before implementation:

| Status | Meaning |
|---|---|
| `native_input` | SOLWEIG already accepts the source directly; an interactive adapter is still required. |
| `derived_input` | The value is produced from another source and should normally not be edited directly. |
| `adapter_required` | Scientifically supported in principle, but local invalidation, persistence, or UI mapping must be implemented. |
| `full_only_initially` | Exact interactive changes are allowed, but the first safe implementation recomputes the full tile or all downstream cells. |
| `scientific_extension` | Requires new model science or parameterization and must not be implied by the first release. |
| `view_only` | Changes visualization or output selection and does not alter scientific state. |

## Edit families

### Building geometry or urban massing

Examples:

- add or remove a building volume;
- move a conceptual massing block;
- change footprint or height;
- edit a small Building DSM raster region.

Primary source: `building_dsm`.

Invalidates, depending on implementation:

```text
building_dsm
 -> walls / wall_aspect
 -> building_visibility / shadow matrices / SVF
 -> shortwave and longwave radiation
 -> surface thermal state
 -> Tmrt
 -> UTCI and WBGT
```

This is often spatially bounded, but low solar altitude and sky-view effects can produce long influence distances. The adapter must calculate an old-plus-new influence region and select full-tile fallback when the safe window is too large or wall/visibility locality is not validated.

### Vegetation geometry

Examples:

- trees;
- hedges;
- canopy masses;
- vegetation DSM brush edits;
- vegetation height, crown, trunk-zone, or transmissivity changes.

Primary source: `vegetation_dsm` plus any adapter metadata needed to reconstruct trunk and canopy behavior.

Invalidates vegetation visibility/SVF, direct shade, radiation, temporal surface state, Tmrt, UTCI, and WBGT. Tree editing is the reference implementation of this family, not the architecture's full scope.

Dynamic airflow response to edited vegetation is a separate scientific extension. The initial exact result may continue using precomputed wind coefficients, with explicit disclosure.

### Terrain or DEM

Examples:

- terrain sculpting;
- grading;
- plaza elevation changes.

Primary source: `dem`.

DEM edits affect the relative height interpretation of buildings and vegetation and can alter walls, horizons, and shadows. The safe initial policy is `full_only_initially` unless a scientific validation demonstrates a bounded local adapter with sufficient halo. The UI may still provide immediate geometry preview.

### Land cover and surface assignment

Examples:

- change asphalt to grass;
- assign water, bare soil, or another supported land-cover class;
- paint a local surface polygon or raster cells.

Primary source: `landcover`.

Potential downstream effects include ground albedo/emissivity/thermal behavior, radiation exchange, temporal surface state, Tmrt, UTCI, and WBGT. The adapter must use the actual land-cover class table and current SOLWEIG equations rather than inventing browser-only physical properties.

A local material paint can use a local window, but temporal state may require replay from an earlier timestep.

### Meteorological forcing and solar time

Examples:

- selected date or hour;
- air temperature;
- relative humidity;
- wind speed and direction;
- global, direct, and diffuse shortwave radiation where supplied;
- longwave or cloud-related forcing represented by the active met format;
- UHI diagnostic input where supported.

Primary sources: `meteorology`, `selected_date_time`, and directional wind-coefficient selection.

These edits usually have global spatial scope but invalidate only downstream dynamic stages. They should reuse DEM, building, wall, SVF, and visibility caches. A global downstream recompute can still be fast enough for interactive use on the fixed 500×500 tile.

Changing wind direction may select a different precomputed directional coefficient raster. Editing geometry and claiming a newly solved wind field is not supported unless the wind model is rerun and validated.

### Model and receptor parameters

Examples present in the current implementation include global building/ground albedo and emissivity, shortwave/longwave absorption coefficients, cylindrical or standing-person view-factor settings, vegetation transmissivity or phenology rules, and output options.

These parameters must be audited in code before exposure. Classification depends on where each parameter enters the equations:

- radiation or surface parameter change: global downstream radiation/Tmrt/comfort invalidation;
- receptor-only parameter: Tmrt or comfort stage only;
- output selection: no scientific recomputation if the output is already cached, otherwise recompute only the missing product.

Do not expose a parameter merely because it is a module-level constant. It needs unit, range, provenance, user-facing meaning, and validation.

### View and analysis selection

Examples:

- switch between UTCI, Tmrt, shadow, SVF, or radiation layers;
- change legend range;
- compare scenario and baseline;
- select a cached timestep.

These are `view_only` unless the requested scientific product has not been computed. They must not generate unnecessary solver jobs.

## Initial adapter priority

Recommended sequence:

1. vegetation geometry reference adapter;
2. meteorology/time adapter that reuses geometry caches;
3. land-cover surface-paint adapter;
4. building massing adapter;
5. audited global model/receptor parameters;
6. DEM editing after full-only correctness is established;
7. optional scientific extensions such as dynamic wind response.

The agent should verify this order against measured user value, scientific risk, and bottlenecks rather than treating it as immutable.
