# Scaffolding validation report

## Scope

This report records validation of the design documents, reusable incremental primitives, browser prototype, and generated presentation assets. It does not validate a scientific ROI-cropped SOLWEIG worker because that worker is intentionally a later implementation phase.

## Environment

| Field | Value |
|---|---|
| Date | 2026-08-31 |
| Platform | Linux 6.18.35 x86_64 |
| CPU | AMD EPYC 9V74, 5 vCPU exposed |
| Memory | 5.8 GiB |
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| Node.js | 22.16.0 |

## Python validation

Command:

```bash
PYTHONPATH=. python -m pytest -q \
  --cov=solweig_gpu.incremental \
  --cov-report=term-missing \
  tests/test_incremental_*.py
```

Result:

```text
34 passed
Incremental package statement coverage: 84%
```

The isolated workspace used a minimal `solweig_gpu/__init__.py` test stub because the complete repository dependency environment, including GDAL, was not installed in the validation container. The stub was removed after the test run. CI in the complete repository imports the normal package and runs the existing full suite.

Validated behaviors include:

- old and new influence-region invalidation;
- north-up world-to-raster conversion;
- block alignment and full-recompute thresholds;
- transitive dirty-window merging;
- non-finite and fractional geometry rejection;
- edit-chain coalescing and conflict detection;
- packed visibility round trips for little- and big-bit order;
- single-patch decode and in-place set/clear;
- packed weighted reduction against a dense reference;
- memory-map shape handling;
- required documentation, frontend files, fixture provenance, selectors, and internal document links.

## Frontend unit validation

Command:

```bash
cd examples/incremental_design_tool
node --experimental-test-coverage --test tests/*.test.mjs
```

Result:

```text
11 passed
Combined line coverage for model.mjs and solver_kernel.mjs: 89.18%
Combined branch coverage: 77.78%
Combined function coverage: 84.91%
```

Validated behaviors include:

- homography round trips;
- invalid homography rejection;
- solar-shadow direction convention;
- move invalidation across old and new positions;
- bounded, aligned grid windows;
- stale worker result rejection by job ID and scene revision;
- accumulation of unapplied dirty regions across rapid edits;
- fixture decoding and building-mask handling;
- patch-only computation;
- no mutation outside the patch window;
- restoration of baseline values after tree deletion.

All JavaScript modules also passed `node --check`.

## Static-server smoke test

The repository-root static server returned the frontend HTML, module source, WebP scene asset, and JSON fixture with successful HTTP responses. The returned fixture passed `python -m json.tool` validation.

A full automated Chromium interaction test is a release-phase requirement. The validation container could not initialize Chromium graphics services, so visual inspection used the deterministic render harness and the checked-in screenshot. This environment limitation does not replace the Playwright tests specified in [Testing and validation](../testing_validation.md).

## Asset reproducibility

Command:

```bash
python examples/incremental_design_tool/tools/build_assets.py \
  --building /path/to/Building_DSM.tif \
  --dem /path/to/DEM.tif \
  --trees /path/to/Trees.tif \
  --landcover /path/to/Landcover.tif \
  --utci /path/to/UTCI_0_0.tif \
  --shadow /path/to/Shadow_0_0.tif \
  --hour-band 12 \
  --output /tmp/rebuilt_solweig_assets
```

The rebuild matched the checked-in files byte for byte:

| File | SHA-256 |
|---|---|
| `assets/site_base.webp` | `0d4f245904bea7ca37be3e76566bf366c34cab8cca7339c001e2edab3b9db5d3` |
| `assets/baseline.json` | `0e7c5e71124c724d3bff3eda2b95001918f7464c96e635bbd68a76936fae1044` |

The source rasters are not committed. The fixture metadata explicitly marks the browser analysis as non-scientific.

## Optimization primitive benchmark

The representative `500 × 500 × 153` packed-visibility benchmark completed with a 30.6× storage reduction relative to `float32`. Full results and caveats are in [Packed-visibility primitive benchmark](../benchmarks/packed_visibility.md).

## Outstanding release gates

The following remain intentionally open:

1. Implement the exact ROI-cropped SOLWEIG worker.
2. Compare local results with full-domain recomputation on add, move, resize, and delete fixtures.
3. Measure target-VM p50, p95, and peak RSS.
4. Add production API, persistence, recovery, and scenario isolation.
5. Add Playwright interaction and visual-regression tests in a graphics-capable CI job.
6. Run the full existing repository test suite in the normal conda/GDAL environment after integration.

The prototype must not be presented as scientifically validated until gates 1 and 2 pass.
