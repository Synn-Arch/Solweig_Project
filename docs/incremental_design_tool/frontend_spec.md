# Frontend specification

## Experience goal

The application should look and behave like a professional environmental design tool. It should not resemble a form that launches a batch script. Users manipulate spatial components directly, receive immediate visual feedback, and see exact analysis refine progressively.

The repository prototype is in `studio/`. It is dependency-free and uses the supplied test tile to create a non-georeferenced visual fixture.

## Visual hierarchy

### Persistent regions

1. Top bar: project, scenario, worker state, reset, export.
2. Left panel: tree component library and analysis layers.
3. Main scene: site model, thermal overlay, trees, shadows, dirty region.
4. Time control: solar hour and UTCI legend.
5. Right panel: impact metrics, selected-tree properties, incremental job stages, scope disclosure.

The prototype screenshot in [the overview](index.md) is the reference composition, not a pixel-perfect requirement.

## Interaction states

| State | Visual behavior | Scientific status |
|---|---|---|
| Exact | Solid result texture and ready indicator | `exact_result_version == scene_version` |
| Dragging | Ghost tree and direct-shadow preview | No mutation committed |
| Preview | New tree rendered, local approximate overlay optional | Mutation committed, exact result stale |
| Queued | Dirty region visible, queue status shown | Job has not started |
| Computing | Stage progress and animated indicator | Exact worker active |
| Exact updated | Patch blends or replaces preview | Result matches current version |
| Superseded | No older patch is applied | Newer edit exists |
| Failed | Preview remains with clear warning | Last exact result remains available |

Do not remove the user's tree when scientific refinement fails. Preserve the design state and explain that the analysis is stale.

## Drag and drop flow

1. `dragstart` stores a component preset ID.
2. Pointer movement over the scene converts canvas coordinates to normalized site UV.
3. A ghost tree and approximate shadow render at the candidate point.
4. Drop outside the site is rejected visually.
5. Drop inside the site creates a stable tree ID and commits one edit request.
6. The tree renders immediately in preview state.
7. The dirty region is shown after server acknowledgement.
8. The exact patch replaces the affected texture when ready.

Only `pointerup` or `drop` commits an edit. Pointer movement must not create server jobs.

## Moving an existing tree

On pointer down:

- select the nearest tree within a screen-space hit radius;
- save an immutable copy of the old tree;
- capture the pointer.

During move:

- update only the local visual tree position;
- show a ghost or preview shadow;
- do not send requests.

On pointer up:

- send the final tree object once;
- include the last acknowledged scene version;
- compute preview invalidation as `old influence ∪ new influence`;
- retain the old exact result until the new patch arrives.

## Property editing

Height, canopy diameter, and transmissivity controls use debounce:

```text
input event -> update visual tree immediately
no input for 400 to 600 ms -> commit one replacement tree
```

The client stores the tree state at the beginning of the debounce interval so the old influence region is not lost.

## Time control

Changing solar time immediately updates the approximate shadow direction. Exact handling depends on the requested product:

- Selected-hour view: request that hour, plus any required temporal warm-up.
- Full-day metrics: keep existing exact metrics marked stale until the full replay completes.
- Rapid slider movement: debounce server work and retain only the latest hour.

The browser prototype changes only tree-driven preview effects against a fixed baseline fixture. Production must use the server's exact time result.

## Preview rendering

The preview may include:

- a projected direct-shadow shape;
- a bounded cooling tint near the crown and shadow;
- an approximate dirty-region outline;
- immediate tree geometry and selection handles.

It must not claim scientific exactness. Recommended labels are `Preview`, `Updating analysis`, and `Exact`.

## Exact patch application

The browser keeps:

```text
baseline Float32 texture or array
current exact texture or array
current scene version
current exact result version
preview design objects
```

When a patch arrives:

1. Validate schema version.
2. Validate scenario ID and site ID.
3. Validate patch dimensions and byte length.
4. Verify checksum if supplied.
5. Reject if `patch.scene_version != client.scene_version`.
6. Apply data to the half-open window.
7. Upload only the changed texture subregion when WebGL supports it.
8. Update metrics and exact result version.
9. Clear the corresponding dirty region.

For WebGL, use `texSubImage2D` rather than replacing the full texture.

## Rendering layers

Recommended order:

1. Dark application background.
2. Site context or base map.
3. Exact UTCI or Tmrt texture.
4. Preview delta texture, only while stale.
5. Existing scene trees.
6. Proposed trees and cast shadows.
7. Dirty-region outline.
8. Selection affordances and labels.
9. Status, legend, scale, and north arrow.

## Color and perception

- Use a perceptually ordered thermal scale with a clearly labeled range.
- Preserve sufficient base-map contrast below the heat layer.
- Do not use red and green alone to encode state.
- Use line style and text for dirty region and exactness.
- Keep building cells transparent or visibly masked.
- Avoid implying sub-cell precision when the model grid is 2 m.

The scientific export retains numeric Float32 values. The rendered color texture is a display derivative.

## Coordinate mapping

The prototype maps site UV coordinates through a homography for the perspective scene. Production map implementations may use a GIS view, but the data flow remains:

```text
pointer canvas px -> site UV -> server site transform -> projected meters -> raster row/column
```

The server remains authoritative for UV-to-world conversion.

## Component presets

A preset defines defaults, not a separate scientific model:

```json
{
  "component_type": "broad_canopy",
  "label": "Broad canopy",
  "height_m": 18,
  "canopy_diameter_m": 11,
  "trunk_ratio": 0.25,
  "transmissivity": 0.03,
  "phenology": "deciduous"
}
```

The server validates the preset and allowed edits.

## Analysis metrics

The right panel may display:

- mean UTCI difference in the updated region;
- peak cooling among valid pedestrian cells;
- area improved by at least a configured threshold;
- dirty-window fraction;
- exact computation duration;
- result version and model revision.

Metrics must state their spatial and temporal aggregation. `Mean UTCI change` without a region and time is ambiguous.

## Responsive behavior

The primary teaching experience targets laptop and desktop widths. Minimum functional width is approximately 1180 px. At smaller widths:

- collapse the component library to a drawer;
- collapse the analysis panel to a bottom sheet;
- keep the scene and exactness status visible;
- preserve keyboard access to all tree properties.

## Accessibility

- Every draggable component also supports keyboard placement through an `Add` command.
- All property controls have labels and output values.
- Status changes use an ARIA live region in production.
- Color encodings have text alternatives.
- Focus order follows top bar, component library, scene controls, analysis panel.
- The canvas has a textual scene summary and selected-tree controls outside the canvas.

## Frontend file map

| File | Responsibility |
|---|---|
| `index.html` | Application structure and accessible controls |
| `styles.css` | Visual system and responsive layout |
| `model.mjs` | Coordinate transforms, dirty rectangles, tree presets |
| `solver_kernel.mjs` | Deterministic preview kernel and patch logic |
| `solver_worker.mjs` | Off-main-thread preview job protocol |
| `renderer.mjs` | WebGL heat texture and Canvas tree overlay |
| `app.mjs` | State, interactions, stale-job handling, metrics |
| `assets/baseline.json` | Downsampled non-scientific browser fixture |
| `assets/site_base.webp` | Non-georeferenced presentation scene |

## Replacement path for production

Replace the Web Worker call in `app.mjs` with the asynchronous API contract while preserving the state machine:

```text
preview edit
-> POST /edits
-> store returned scene_version and job_id
-> poll or subscribe
-> GET result manifest
-> GET binary patch
-> validate version
-> apply patch
```

Keep `model.mjs` only for immediate preview and visual dirty-region estimation. The server's window is authoritative.
