# Frontend specification for universal editing

## Product model

The frontend is a registry-driven environmental design workspace. It must not be organized around one permanent “tree library.” The tool palette is generated from server capability metadata and can contain geometry, surfaces, environmental controls, model parameters, and view-only analysis controls.

## Tool groups

### Geometry

Potential tools:

- building/massing block;
- tree, hedge, or canopy component;
- vegetation brush;
- terrain/grading tool when enabled.

Geometry objects support selection, move, dimensions, height, duplication, delete, and typed properties defined by the adapter schema.

### Surface

Potential tools:

- land-cover brush;
- polygon assignment;
- supported material/class presets;
- reset to baseline.

The UI shows cell resolution and prevents false sub-cell precision.

### Environment

Potential controls:

- date and time;
- air temperature and relative humidity;
- wind speed and direction;
- audited radiation inputs;
- scenario forcing presets.

These controls may cause global downstream analysis but should display which expensive geometry caches remain reused.

### Model and receptor

Expose only parameters audited and documented with scientific units, valid range, and effect. Advanced parameters belong behind a disclosure panel.

### Analysis and view

- UTCI, Tmrt, WBGT, shadow, SVF, and supported radiation layers;
- baseline/scenario comparison;
- time-series chart;
- exactness and model/version metadata;
- export.

Switching a cached layer is view-only and should not enqueue computation.

## Common interaction state machine

Every scientific editor uses the same state model:

```text
IDLE_EXACT
 -> EDITING_PREVIEW
 -> COMMITTED_STALE
 -> QUEUED
 -> COMPUTING
 -> EXACT
```

Failure leaves the edited design visible with the last exact result explicitly marked stale. It must not silently revert the user's design.

## Adapter-provided preview

Each adapter supplies a preview descriptor:

- building: realistic massing and approximate cast shadow;
- vegetation: canopy/trunk geometry and approximate shade;
- land cover: material color/texture and estimated affected cells;
- meteorology: immediate visual/timeline change and “global downstream update” badge;
- model parameter: textual scope and pending exact analysis.

Preview is interaction feedback, not scientific output.

## Realistic rendering requirements

- perspective or map-accurate site context;
- physically plausible sun direction for preview shadows;
- distinct building, vegetation, and surface materials;
- heat layer blended without obscuring geometry;
- dirty-region or full-domain scope visualization;
- stage progress tied to dependency nodes, not generic spinner only;
- clear labels for Preview, Exact, Stale, and Full-tile fallback;
- no external CDN requirement for the teaching demo unless explicitly approved.

## Dependency-aware status panel

Show what changed and what is being recomputed:

```text
Changed: Building DSM + Land cover
Reused: DEM, forcing
Recomputing: Walls -> Visibility -> SVF -> Radiation -> Tmrt -> UTCI
Scope: 22% local; full temporal replay
```

For weather changes:

```text
Changed: Air temperature, relative humidity
Reused: Geometry, walls, visibility, SVF
Recomputing: Tmrt/UTCI over full tile
```

This makes the tool look and behave like genuine design-analysis software rather than hiding all work behind a single loading state.

## Capability API

The browser should obtain a registry resembling:

```json
{
  "adapters": [
    {
      "id": "building_massing",
      "group": "geometry",
      "operations": ["add", "move", "update", "delete"],
      "property_schema": {},
      "preview": "massing_shadow",
      "support_status": "full_only_initially"
    }
  ]
}
```

The server remains authoritative for validation and impact planning.
