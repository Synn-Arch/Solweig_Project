# Full-domain oracle baseline (P0)

Deterministic reference measurement of the unmodified full-domain SOLWEIG-GPU
CPU path on the 500 x 500, 2 m design-tool site. This record anchors
`BASE-001`, `BASE-003`, and all later T4 comparisons.

## Fixture

| Field | Value |
|---|---|
| Fixture ID | `site_500` (see `solweig_gpu/incremental/fixtures.py`, manifest `fixtures/site_500/manifest.json`) |
| Grid | 500 x 500 cells, 2.0 m, EPSG:32614 |
| Origin (upper-left) | 621734.7066 E, 3354614.3479 N |
| Site rasters | `Input_subset/{Building_DSM,DEM,Trees}.tif` (local, gitignored; source hashes in fixture manifest) |
| Meteorological forcing | `ownmet_Forcing_data.txt`, DOY 223 2009, hourly; run date 2020-08-13 |
| Time steps | 24 |
| SVF cache | warm (precomputed `Input_subset/processed_inputs/SVF`) |
| Edit states | baseline, add, move, resize, delete (fixture generator invariants all true) |

## Environment

| Field | Value |
|---|---|
| Machine | Apple M1 Pro, 10 logical cores, 16 GB RAM |
| OS | macOS (Darwin 25.5.0) |
| Python | 3.14 (repo `.venv`) |
| numpy | 2.5.2 |
| torch | 2.13.0 (CPU execution; MPS present but solver pins CPU tensors) |
| GDAL | 3.13.3 |
| rasterio | 1.5.1 |
| numba | 0.67.0 (installed during P0; required by `calculate_wbgt` import chain) |
| Thread env | defaults (no OMP/MKL pinning in this run) |
| Command | `/usr/bin/time -l .venv/bin/python run_test.py` (warm SVF cache) |

## Results (run 1, warm SVF cache)

| Metric | Value |
|---|---|
| Tile compute wall time | 80.73 s |
| Process wall time | 100.64 s |
| Peak RSS | 1,412,087,808 B (1347 MiB / 1.41 GB) |
| Outputs | UTCI/TMRT/Shadow multi-band GeoTIFF, 24 bands each |

Run 2 (determinism check): see `2026-09-01-full-domain-baseline.json` for the
machine-readable record and bitwise output comparison.

## Output hashes (run 1)

```text
925ee0f70a6442eb34cd48ab8febe134db5d030d091d6d95fd44814a26194970  Shadow_0_0.tif
04d91548504e428c8724987d7f74af97027e2c251902c39d2f40101ff5aa5f71  TMRT_0_0.tif
351e18ade151d67ca29380193cda08c8df1a5376e76a1516c18f23416a6db1d1  UTCI_0_0.tif
```

GeoTIFF container bytes include timestamps only in band metadata; bitwise
equality across runs is expected and required for `BASE-001`.

## Model revision

Oracle = unmodified full-domain solver at commit recording this benchmark
(see `commit_sha` in the JSON record).
