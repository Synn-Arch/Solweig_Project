# Data model and coordinate conventions

## Coordinate systems

All code paths must distinguish four coordinate spaces explicitly.

| Space | Units | Origin and axes | Use |
|---|---|---|---|
| Geographic/projected world | meters | CRS-defined east/north | Raster metadata and scientific geometry |
| Raster index | integer cells | row increases south, column increases east | Window reads and writes |
| Site normalized UV | unitless | `(0,0)` top-left, `u` east, `v` south | Frontend and portable API placement |
| Canvas | pixels | `(0,0)` top-left | Rendering and pointer interaction |

Do not pass ambiguous `x` and `y` fields without a suffix or schema definition. Use names such as `x_m`, `row_start`, `u`, and `canvas_x_px`.

## Half-open raster windows

Every raster window uses half-open bounds:

```text
rows    [row_start, row_stop)
columns [col_start, col_stop)
```

This convention is mandatory for Python slices, API patches, cache chunks, and frontend application. A window's dimensions are:

```text
height = row_stop - row_start
width  = col_stop - col_start
```

The implemented `RasterWindow` type in `solweig_gpu/incremental/geometry.py` follows this convention.

## Site manifest

Each deployable site has one immutable manifest.

```json
{
  "site_id": "campus-1km-v1",
  "cache_schema_version": 1,
  "model_version": "solweig-gpu-2.0.0+incremental.1",
  "crs_wkt_hash": "sha256:...",
  "rows": 500,
  "cols": 500,
  "pixel_size_m": 2.0,
  "origin_x_m": 621734.7066,
  "origin_y_m": 3354614.3479,
  "time_steps": 24,
  "patch_count": 153,
  "lowest_sky_patch_altitude_deg": 6.0,
  "baseline_result_version": "2026-08-31-fixture-1"
}
```

The worker must validate shape, dtype, transform hash, time-step count, and patch count before accepting jobs.

## Tree object

A tree object is the authoritative design component.

```json
{
  "tree_id": "tree_01J7B4N8P0KQ",
  "component_type": "broad_canopy",
  "u": 0.575,
  "v": 0.345,
  "height_m": 18.0,
  "canopy_diameter_m": 11.0,
  "trunk_ratio": 0.25,
  "transmissivity": 0.03,
  "phenology": "deciduous",
  "metadata": {
    "label": "Broad canopy 02"
  }
}
```

### Required validation

- `tree_id` is stable and unique within a scenario.
- `u` and `v` are finite and inside `[0,1]`.
- `height_m` is finite and inside a configured range, initially `[3, 40]`.
- `canopy_diameter_m` is finite and inside `[1, 30]`.
- `trunk_ratio` is in `[0,1)` (a trunk ratio of exactly 1 would leave no
  canopy; `solweig_gpu/server/models.py` rejects it — p9-rel evidence note,
  2026-09-02). `transmissivity` is in `[0,1]`.
- The rasterized canopy is clipped to the site boundary.
- Tree presets are server-side validated. The client cannot invent unrestricted component types.

## Edit event

Edits are append-only and capture both old and new state.

```json
{
  "scenario_id": "scn_01J7B4...",
  "event_id": "evt_01J7B5...",
  "sequence": 42,
  "base_scene_version": 17,
  "operation": "move",
  "tree_id": "tree_01J7B4N8P0KQ",
  "old_tree": { "...": "complete previous tree object" },
  "new_tree": { "...": "complete new tree object" },
  "submitted_at": "2026-08-31T19:43:42.384Z",
  "idempotency_key": "browser-session-7:edit-42"
}
```

The server reconstructs `old_tree` from authoritative scenario state and should not trust the client copy for correctness. The client copy is useful only for optimistic UI and conflict diagnostics.

## Scenario snapshot

```json
{
  "scenario_id": "scn_01J7B4...",
  "site_id": "campus-1km-v1",
  "scene_version": 18,
  "exact_result_version": 17,
  "status": "refining",
  "trees": ["...tree objects..."],
  "active_job_id": "job_01J7B6...",
  "created_at": "2026-08-31T19:40:00Z",
  "updated_at": "2026-08-31T19:44:12Z"
}
```

## Job record

```json
{
  "job_id": "job_01J7B6...",
  "scenario_id": "scn_01J7B4...",
  "target_scene_version": 18,
  "status": "running",
  "mode": "local",
  "window": {
    "row_start": 96,
    "row_stop": 288,
    "col_start": 160,
    "col_stop": 352
  },
  "time_start_index": 0,
  "time_stop_index": 24,
  "queued_at": "2026-08-31T19:44:12Z",
  "started_at": "2026-08-31T19:44:13Z",
  "finished_at": null,
  "worker_revision": "git:abc1234",
  "metrics": null,
  "error": null
}
```

## Result patch

A patch contains a rectangular window and one or more variables. The first release should return UTCI and optionally Tmrt for the selected time or full time series.

### Metadata envelope

```json
{
  "schema_version": 1,
  "scenario_id": "scn_01J7B4...",
  "scene_version": 18,
  "model_version": "solweig-gpu-2.0.0+incremental.1",
  "window": {
    "row_start": 96,
    "row_stop": 288,
    "col_start": 160,
    "col_stop": 352
  },
  "variables": [
    {
      "name": "utci",
      "dtype": "float32",
      "shape": [24, 192, 192],
      "nodata": "nan",
      "byte_offset": 0,
      "byte_length": 3538944
    }
  ],
  "compression": "zstd",
  "checksum": "sha256:..."
}
```

### Binary order

Unless changed by a versioned schema:

```text
C-order, little-endian
variable-major
then time
then row
then column
```

The frontend must not infer shape from byte length alone. It validates the metadata envelope and scene version before applying the patch.

## Baseline cache layout

Recommended initial filesystem layout:

```text
cache/<site_id>/
├── manifest.json
├── static/
│   ├── dem.f32.npy
│   ├── building_dsm.f32.npy
│   ├── tree_base.f32.npy
│   ├── walls.f32.npy
│   ├── wall_aspect.f32.npy
│   ├── landcover.u8.npy
│   └── valid_pedestrian_mask.u8.npy
├── visibility/
│   ├── building.packed.npy
│   ├── vegetation_base.packed.npy
│   ├── combined_base.packed.npy
│   └── metadata.json
├── radiation/
│   ├── svf.f32.npy
│   ├── directional_svf.f32.npy
│   └── static_coefficients.npz
├── forcing/
│   ├── meteorology.f32.npy
│   ├── solar_positions.f32.npy
│   └── timestamps.json
└── baseline_results/
    ├── utci.f32.npy
    ├── tmrt.f32.npy
    └── metadata.json
```

Use `.npy` memory maps for fixed-shape arrays in the first implementation. Avoid compressed NPZ for arrays that must be window-read repeatedly because NPZ decompression materializes whole members.

## Packed visibility metadata

```json
{
  "schema_version": 1,
  "rows": 500,
  "cols": 500,
  "patch_count": 153,
  "bytes_per_pixel": 20,
  "bitorder": "little",
  "axis_order": ["row", "column", "packed_patch_byte"],
  "semantic": "1 means visible or sunlit according to source array",
  "source_dtype": "float32-binary",
  "checksum": "sha256:..."
}
```

The bit convention must be documented because some current arrays use visibility while other outputs use shadow or sunlit semantics. Do not pack an array until its semantic polarity is named and tested.

## Versioning rules

- Increment `cache_schema_version` when file layout, dtype, bit order, axis order, or semantics change.
- Increment `result schema_version` when binary patch layout changes.
- Include the exact model revision in every published result.
- Invalidate scenario patches when the site cache or scientific model version changes.
- Do not serve patches produced by a different cache manifest, even if shapes happen to match.
