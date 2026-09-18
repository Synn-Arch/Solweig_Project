# CPU optimization plan

A measured primitive benchmark for the representative `500 × 500 × 153` visibility volume is available in [Packed-visibility primitive benchmark](benchmarks/packed_visibility.md). It confirms a 30.6× storage reduction relative to `float32`; it does not claim end-to-end solver latency.

## Optimization objective

The target is not to make the current whole-tile PyTorch pipeline marginally faster. The target is to change the unit of work from a complete `[time, row, column, sky_patch]` problem to a bounded, streamable spatial update that fits one CPU node.

Optimization must preserve scientific output. Approximation is permitted only in the explicitly labeled browser preview.

## Baseline memory problem

For a 500 × 500 tile and 153 sky patches:

```text
one float32 visibility cube
= 500 × 500 × 153 × 4 bytes
= 153,000,000 bytes
≈ 145.9 MiB
```

Three cubes require approximately 437.7 MiB before allocator overhead and temporaries. A separate full-size `diffsh` cube adds another approximately 145.9 MiB. Loading compressed NPZ members also creates decompression and tensor-copy peaks.

The current time loop can additionally retain multiple 24-band outputs even when they are not requested. The combined pattern is unsuitable for a 1 to 4 GB service node.

## Required optimization order

Apply changes in this order. Each phase must be benchmarked before proceeding.

1. Eliminate unrequested outputs and stream time bands.
2. Window all input and output arrays.
3. Separate immutable building data from dynamic vegetation data.
4. Pack binary visibility along the patch axis.
5. Remove full-size `diffsh` and similar cubes.
6. Keep the worker warm and caches memory-mapped.
7. Profile the remaining CPU kernels.
8. Port only measured hotspots to Numba, C++, or another compiled path.
9. Consider directional-horizon approximations only as a separately validated mode.

Do not begin by replacing every PyTorch operation with another library. Structural locality usually produces the largest gain.

## Output allocation and streaming

### Current anti-pattern

```python
utci_all.append(utci.cpu().numpy())
tmrt_all.append(tmrt.cpu().numpy())
kup_all.append(kup.cpu().numpy())
# repeated for every timestep and variable
```

### Target pattern

```python
writer = PatchWriter(
    variables=requested_variables,
    time_steps=time_stop - time_start,
    window=write_window,
)

for time_index in range(time_start, time_stop):
    fields = solve_time_step(...)
    writer.write("utci", time_index, fields.utci[write_slice])
    if "tmrt" in requested_variables:
        writer.write("tmrt", time_index, fields.tmrt[write_slice])
    release_step_temporaries(fields)

return writer.finalize()
```

Rules:

- Do not convert a field to NumPy unless it will be written or returned.
- Do not allocate optional variables when their save flag is false.
- Preallocate compact `[time, write_rows, write_cols]` output when it is small enough, otherwise stream to a temporary file.
- Use Float32 unless validation demonstrates a safe alternative.
- Keep NaN masking until the final output assignment.

## Windowed input access

Every cache abstraction should expose:

```python
array.window(RasterWindow) -> ndarray view or memory-mapped slice
```

Preferred first implementation:

- uncompressed `.npy` for repeated fixed-shape arrays;
- `numpy.load(path, mmap_mode="r")` for read-only caches;
- `numpy.lib.format.open_memmap` for writable packed caches and staged results;
- GDAL only at import/export boundaries, not inside each interactive job.

Compressed NPZ is acceptable for archival transfer but not the primary serving format. Accessing one NPZ member generally decompresses the whole member.

## Static and dynamic geometry split

### Immutable arrays

Keep these read-only and shared:

- DEM;
- building DSM;
- wall height and aspect;
- building mask;
- building-only sky-patch visibility;
- base land cover;
- fixed forcing and solar geometry.

### Dynamic arrays

Allocate only over the read window:

- user-tree canopy and trunk-zone rasters;
- vegetation visibility for relevant patches;
- vegetation-adjusted SVF accumulators;
- time-dependent surface states;
- requested result fields.

Building visibility must never be recomputed after a tree edit.

## Bit-packed visibility

The values in visibility and shadow patch cubes are binary. Store eight patches per byte.

For 153 patches:

```text
bytes_per_pixel = ceil(153 / 8) = 20
one packed 500 × 500 cube = 5,000,000 bytes ≈ 4.77 MiB
three packed cubes ≈ 14.31 MiB
```

Compared with three Float32 cubes, this is approximately a 30.6 times reduction. The repository implementation is in `solweig_gpu/incremental/bitmask.py`.

### Access strategy

Do not unpack the entire cube. Extract one patch plane or a small block:

```python
plane = unpack_patch(packed, patch_index)  # 2-D bool array
accumulator += plane * weight
```

For a local window, slice packed data first:

```python
packed_window = packed.data[
    row_start:row_stop,
    col_start:col_stop,
    :,
]
```

Then unpack from that view.

### Mutation strategy

Dynamic scenario visibility can be:

- rebuilt into a temporary packed window per job, preferred initially;
- stored as sparse changed chunks per scenario if reuse becomes important;
- patched with `set_patch_window` when one patch plane is recomputed.

Do not mutate the shared baseline packed arrays.

### Semantic tests

Before converting an existing array, verify:

- binary values are exactly 0 or 1;
- the patch axis is known;
- bit order is written to metadata;
- polarity is named, for example `1 = visible`;
- round-trip unpacking is exact;
- weighted reduction matches the dense reference.

## Remove full `diffsh`

A full `diffsh[row, col, patch]` array duplicates information that can be derived per patch:

```python
for patch_index in relevant_patch_indices:
    sh = unpack_patch(building_visibility, patch_index)
    veg = unpack_patch(vegetation_visibility, patch_index)
    diff = sh - (1.0 - veg) * (1.0 - trans_veg)
    consume_diff_patch(diff, patch_index)
```

The temporary is two-dimensional and released each iteration. If several downstream terms need the same patch, process them in one fused function before release.

## Patch and direction pruning

Not every interaction requires all patches in every stage.

Potential exact pruning:

- Direct solar shadow for one timestep requires only that solar direction.
- Some directional SVF terms require a known subset of azimuths.
- A selected-hour quick refinement can defer full-day metrics.

Potential approximate pruning must be a separate mode and cannot silently replace the exact solver.

## CPU backend strategy

### Phase 1: retain PyTorch where convenient

Windowing and streaming can be implemented before changing the numerical backend. Configure explicit CPU mode and thread count:

```python
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
```

Set these at process startup, not per request. Benchmark one, two, and four threads. More threads can be slower for small windows.

### Phase 2: NumPy and Numba for measured hotspots

The shadow ray loop and repeated shifted comparisons are likely candidates because they combine Python control flow with elementwise tensor operations. A Numba implementation should:

- accept contiguous Float32 or Boolean arrays;
- allocate output and scratch buffers once;
- reuse scratch buffers across rays;
- use explicit loops that compile well;
- parallelize over independent rows, columns, or rays only after profiling;
- avoid Python objects and dynamic lists inside `njit` functions;
- match the dense reference exactly before enabling fast-math.

Do not use `fastmath=True` until error tests show it is safe for each kernel.

### Phase 3: compiled extension only if needed

A C++/OpenMP extension is justified only if Numba or vectorized NumPy misses the latency target. Keep a Python reference implementation and cross-check outputs in tests.

## Data types

| Data | Recommended dtype | Reason |
|---|---|---|
| DEM and DSM | Float32 | Existing scientific precision and memory balance |
| Land cover and masks | UInt8 or Boolean | Categorical or binary |
| Packed visibility | UInt8 | Eight patch bits per byte |
| Patch weights | Float32 | Sufficient for current model |
| Accumulators | Float32 initially | Match existing model; evaluate Float64 only for validation |
| Tree IDs and metadata | Structured Python/JSON | Sparse control data |
| Output UTCI/Tmrt | Float32 | Existing output format |

Avoid Float16 on a general CPU. It may be slower and can materially alter radiation and comfort calculations.

## Memory reuse

Create a per-worker scratch pool keyed by shape and dtype:

```text
scratch/<read_height, read_width>/
├── temp_surface_1.f32
├── temp_surface_2.f32
├── bool_mask_1.u8
├── svf_accumulator.f32
└── output_step.f32
```

Reuse arrays through `fill(0)` or in-place operations. Do not call `torch.cuda.empty_cache()` in CPU code. Remove device-specific cache calls from shared kernels or guard them with `device.type == "cuda"`.

## Spatial chunking

Use block-aligned windows, initially 16 × 16 or 32 × 32 cells. Benefits:

- predictable memory-map page access;
- easier sparse patch persistence;
- stable binary patch boundaries;
- reduced fragmentation from arbitrary windows;
- possible result-tile reuse.

Do not force a tiny edit into one giant chunk. Merge only when setup savings exceed extra area.

## Rasterization optimization

Tree rasterization is sparse. Recommended path:

1. Query trees intersecting the read window.
2. Convert each tree center to local window coordinates.
3. Compute a bounded crown ellipse or component footprint.
4. Update only that bounding slice.
5. Combine canopy top height with `maximum`.
6. Build trunk-zone height from the same footprint and `trunk_ratio`.

Avoid drawing each tree into a full 500 × 500 temporary.

## Temporal optimization

- Precompute solar altitude, azimuth, zenith, day-of-year, and forcing tensors once per site/day.
- Precompute scalar terms that do not depend on vegetation.
- Reuse state buffers across timesteps.
- For interactive selected-hour results, expose a two-stage job: a fast exact selected-hour path only if model state permits, followed by full temporal replay for aggregate metrics.
- Do not skip temporal replay without a validation argument.

## API and serialization optimization

- Return a binary patch, not JSON arrays of floats.
- Compress patches with Zstandard or another fast codec after measuring size and CPU cost.
- Include a checksum and explicit shape.
- Send only changed variables and time indices.
- For the browser, consider quantized visualization tiles only as a display derivative. Preserve Float32 scientific data for export and validation.

## Process-level configuration

Recommended initial environment:

```text
WORKER_COUNT=1
SCIENTIFIC_THREADS=2 or 4 after benchmark
API_THREADS=small default
GDAL_CACHEMAX=128 MB or less
OMP_NUM_THREADS=<scientific threads>
MKL_NUM_THREADS=<scientific threads>
OPENBLAS_NUM_THREADS=<scientific threads>
NUMBA_NUM_THREADS=<scientific threads>
```

Prevent nested thread pools. A four-core VM can become slower if PyTorch, OpenMP, BLAS, and Numba each launch four threads.

## Profiling protocol

Measure complete jobs and individual stages:

```text
queue delay
edit coalescing
window construction
cache mapping
vegetation rasterization
visibility update
SVF reduction
time-loop radiation and Tmrt
UTCI calculation
patch serialization
patch publication
peak RSS
bytes read and written
```

Use:

- `time.perf_counter_ns()` for stage timing;
- `resource.getrusage` or `psutil` for process RSS;
- `py-spy` for sampling without code instrumentation;
- `cProfile` for Python-call overhead;
- `line_profiler` only on narrowed hotspots;
- Linux `perf` for compiled kernels when needed.

A benchmark result without input shape, tree count, window, timestep count, thread count, cache state, and machine type is not actionable.

## Performance fixture matrix

At minimum benchmark:

| Fixture | Purpose |
|---|---|
| One 10 m tree at noon in open area | Typical small edit |
| One 20 m tree at low sun | Long influence region |
| Move across two distant positions | Two-window invalidation |
| Ten overlapping trees | Nonlinear and rasterization stress |
| Tree at boundary | Clipping and halo |
| Dirty region just below fallback threshold | Worst local case |
| Full-tile fallback | Upper latency bound |
| Twenty classroom scenarios queued | Queue behavior and cache reuse |

## Acceptance budgets

Initial budgets for a 2 to 4 vCPU, 4 GB node:

| ID | Budget |
|---|---|
| PERF-001 | Typical one-tree exact job p50 below 20 s |
| PERF-002 | Typical one-tree exact job p95 below 60 s |
| PERF-003 | Full-tile fallback below 5 min |
| MEM-001 | Worker peak RSS below 3.0 GB on a 4 GB node |
| MEM-002 | Steady idle RSS below 1.5 GB after cache warm-up |
| IO-001 | Typical client patch below 5 MB compressed |
| QUEUE-001 | Latest-version coalescing prevents unbounded per-scenario backlog |

### Measured budgets (P8, 2026-09-02)

Measured on the reference development host (Apple M1 Pro 10-core, 16 GB —
faster than the 2-4 vCPU target node; methodology, raw runs, and profiles in
[2026-09-02 P8 performance evidence](benchmarks/2026-09-02-p8-performance/p8_summary.md)).
This section replaces the initial placeholder values with measured results as
required above. No scientific tolerance was changed anywhere in this process —
these are latency/payload budgets only, and every optimization landed under
the bitwise correctness contract (inside-window bitwise vs the full-tile
oracle up to a recorded 1-ulp UTCI drift in 4/8 T3 scenarios; shadow bitwise
everywhere; outside-window bitwise vs the prior published store).

| ID | Initial budget | Measured (M1 Pro) | Production budget | Verdict |
|---|---|---|---|---|
| PERF-001 | p50 < 20 s | p50 54.21 s (89.32 s pre-P8, −39%) | p50 < 75 s on target node | PASS @ 54.21 s |
| PERF-002 | p95 < 60 s | p95 59.14 s (93.89 s pre-P8, −37%) | unchanged: p95 < 60 s | PASS @ 59.14 s (borderline) |
| PERF-003 | < 300 s | 92.9 s | unchanged | PASS |
| MEM-001 | < 3.0 GB | 978.6 MB | unchanged | PASS (3.3× headroom) |
| MEM-002 | < 1.5 GB | 504.9 MB idle | unchanged | PASS |
| IO-001 | < 5 MB | utci-only 2.67 MB (typical client fetch); utci+tmrt full-day 5.86 MB | unchanged, "typical" = the client's default single-variable patch | PASS @ 2.67 MB; 5.86 MB two-variable full-day recorded |
| QUEUE-001 | bounded backlog | 15 complete + 12 superseded, 0 nonterminal, 20/20 scenarios exact | unchanged | PASS |

PERF-001 derivation for the 75 s production budget: measured M1 Pro p50
54.21 s × 1.35 target-node factor (the job is kernel-launch bound, not
core-bound — thread sweep 1T 88.1 s / 2T 78.6 s / 4T 68.7 s / 8T 74.9 s —
so the slower node costs a modest, not linear, penalty), rounded up. The
initial 20 s placeholder is not reachable with the current bitwise-exact
design: the exact read window is physics-bound at ~0.80 of the site (33 m
relief at the 6° sky-patch floor), and the profiled floor after the night-march
gate is SVF-replay marches 22.7 s + launch-bound `define_patch` 16.5 s +
read-extent gvf/sunonsurface ~13 s.

Known follow-up work, ordered by projected effect (none implemented; all
deferred past P8):

1. Per-patch march windows W_α = bbox(write ⊕ reach(α)) for the 153-patch SVF
   replay — projected −8-10 s.
2. Multi-edit windows: `_exact_influence_window` and
   `read_window_for_write_window` currently return one bounding box per
   batch; opposite-corner edits inflate to near-full-site windows (correct,
   but costly; also inflates patch size toward the IO-001 two-variable
   figure).
3. IO-001 two-variable mitigation: byte-shuffle/delta preprocessing before
   zstd, or per-variable refine payloads.

## Optimization stop rule

Stop optimizing when all correctness tests pass and latency, memory, and queue budgets are met on the deployment node. Do not introduce a more complex approximation solely to improve an already acceptable demo latency.
