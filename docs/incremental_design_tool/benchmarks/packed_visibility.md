# Packed-visibility primitive benchmark

## Purpose

This benchmark validates the memory argument behind the packed sky-patch representation. It is not an end-to-end SOLWEIG latency result. The benchmark generates one binary visibility volume with the supplied study-tile dimensions, packs the patch axis, evaluates a weighted patch reduction directly from the packed data, and compares it with a dense NumPy reference.

## Reproduction

```bash
PYTHONPATH=. python examples/incremental_design_tool/tools/benchmark_primitives.py
```

The default shape is `500 × 500 × 153`. The benchmark script has no GDAL or PyTorch dependency.

## Recorded environment

| Field | Value |
|---|---|
| Date | 2026-08-31 |
| CPU | AMD EPYC 9V74 |
| Available vCPU | 5 |
| Memory | 5.8 GiB |
| Python | 3.13.5 |
| NumPy | 2.3.5 |

The environment is an ephemeral container, not the target deployment VM. Use these measurements as a reproducibility check and as evidence for representation choice, not as a service-level latency guarantee.

## Result

| Metric | Measured value |
|---|---:|
| Dense `uint8` visibility | 38,250,000 bytes |
| Dense `float32` equivalent | 153,000,000 bytes |
| Packed visibility | 5,000,000 bytes |
| `float32` to packed reduction | 30.6× |
| Packing time | 0.200 s |
| Packed weighted reduction | 0.029 s |
| Dense weighted reduction | 0.581 s |
| Maximum absolute difference | 2.67 × 10⁻⁵ |
| Whole-process wall time | 1.69 s |
| Maximum process RSS | 437,168 KiB |

The dense reference allocates a temporary `float32` volume for multiplication. The packed implementation processes one bit plane at a time, so this particular benchmark is also faster than the dense expression. This speed comparison is implementation-specific. The stable conclusion is the 30.6× storage reduction relative to the current `float32` representation.

The small numerical difference is caused by floating-point accumulation order. The visibility bits round-trip exactly. Scientific validation must compare final Tmrt and UTCI, not infer final equivalence from this primitive benchmark.

## Interpretation for the worker

Three current `500 × 500 × 153` `float32` visibility volumes require approximately 459 MB before temporary arrays. Three packed volumes require approximately 15 MB. The incremental worker should therefore:

1. store immutable building and baseline-vegetation visibility as packed, memory-mapped arrays;
2. decode only the requested spatial window and patch plane;
3. avoid constructing `rows × cols × patches` floating-point intermediates;
4. reuse one or a small number of 2-D scratch buffers;
5. rerun this benchmark after any bit-order, patch-order, or dtype change.

Raw measurements are stored in [`2026-08-31-packed-visibility.json`](2026-08-31-packed-visibility.json).
