// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_utci.cu — UTCI stage, bit-exact CUDA port of the CPU canonical
// ``solweig_core/numba_cpu/utci.py`` (TASKS T12; DESIGN 9.3).
//
// Structure mirrors the CPU two-variant model over ONE element kernel:
//
//   * dense  — count kernel (per-row valid counts) + host prefix sum +
//     fill kernel (one thread per cell, in-row ordinal scan for the
//     masked-select lane k). The extent n and the ordinal k are EXACTLY
//     the CPU's: row-major masked-select order of
//     ``~(Ta<=-999 | RH<=-999 | va<=-999 | Tmrt<=-999)``.
//   * sparse — one thread per pre-compacted valid element.
//
// The 211-statement polynomial lives in the GENERATED header
// (include/sw_utci_poly_generated.cuh, produced from the CPU module's
// POLY_EXPRESSION_SOURCE — tests re-generate and assert byte equality).
// The saturation-pressure chain below is the verbatim CPU association;
// every literal is an exact hex float (no decimal parsing disagreement).
//
// Schedule independence (RED witness 7): a lane's value depends ONLY on
// its inputs and (k, n, T, grain) — never on threadIdx/blockIdx. The
// count pass is an integer reduction; fill/sparse lanes write disjoint
// outputs (no atomics anywhere).
// ---------------------------------------------------------------------------
#include "../include/sw_strict_math.cuh"
#include "../include/sw_kernels_decl.cuh"
#include "../include/sw_utci_poly_generated.cuh"

// saturation-pressure chain constants (utci.py _G / NEG999 / offsets,
// exact float32 bit patterns)
#define SW_UTCI_G0 (-0x1.6292620000000p+11f)
#define SW_UTCI_G1 (-0x1.78c13a0000000p+12f)
#define SW_UTCI_G2 (0x1.38aea40000000p+4f)
#define SW_UTCI_G3 (-0x1.c090ec0000000p-6f)
#define SW_UTCI_G4 (0x1.10d3760000000p-16f)
#define SW_UTCI_G5 (0x1.82169c0000000p-31f)
#define SW_UTCI_G6 (-0x1.a4a2ec0000000p-43f)
#define SW_UTCI_G7 (0x1.5b861e0000000p+1f)
#define SW_UTCI_TK_OFF  (0x1.1126660000000p+8f)
#define SW_UTCI_ES_SCALE (0x1.47ae140000000p-7f)
#define SW_UTCI_RH_DIV (0x1.9000000000000p+6f)
#define SW_UTCI_PA_DIV (0x1.4000000000000p+3f)
#define SW_UTCI_NEG999 (-0x1.f380000000000p+9f)

// the oracle's validity predicate: NaN lanes are VALID (NaN <= -999 is
// false) and stay valid lanes that propagate NaN through the element.
__device__ __forceinline__ bool sw_utci_valid(float ta, float rh,
                                              float tmrt, float va) {
    return !(ta <= SW_UTCI_NEG999 || rh <= SW_UTCI_NEG999 ||
             va <= SW_UTCI_NEG999 || tmrt <= SW_UTCI_NEG999);
}

// one valid lane, k = its masked-select ordinal, n = valid extent
// (utci.py _utci_element, verbatim association)
__device__ __forceinline__ float sw_utci_element(float ta, float rh,
                                                 float tmrt, float va,
                                                 long long k, long long n,
                                                 int T, int grain) {
    float tk = sw_nadd(ta, SW_UTCI_TK_OFF);
    bool use_libm = sw_is_libm_lane(k, n, T, grain);

    float es = sw_nmul(SW_UTCI_G7, sw_sleef_logf_u1(tk));
    // for i in range(0, 7): es = es + g[i] * tk ** (i + 1 - 3.)
    es = sw_nadd(es, sw_nmul(SW_UTCI_G0,
                             sw_ndiv(1.0f, sw_nmul(tk, tk))));   // tk ** -2.
    es = sw_nadd(es, sw_nmul(SW_UTCI_G1, sw_ndiv(1.0f, tk)));    // tk ** -1.
    es = sw_nadd(es, sw_nmul(SW_UTCI_G2, 1.0f));                 // tk ** 0.
    es = sw_nadd(es, sw_nmul(SW_UTCI_G3, tk));                   // tk ** 1.
    es = sw_nadd(es, sw_nmul(SW_UTCI_G4, sw_nmul(tk, tk)));      // tk ** 2.
    es = sw_nadd(es, sw_nmul(SW_UTCI_G5,
                             sw_nmul(sw_nmul(tk, tk), tk)));     // tk ** 3.
    es = sw_nadd(es, sw_nmul(SW_UTCI_G6,
             sw_torch_pow_scalar(tk, 0x1.0000000000000p+2f, use_libm)));
    es = sw_nmul(sw_sleef_expf(es), SW_UTCI_ES_SCALE);

    float ehpa = sw_ndiv(sw_nmul(es, rh), SW_UTCI_RH_DIV);
    float dtm = sw_nsub(tmrt, ta);
    float pa = sw_ndiv(ehpa, SW_UTCI_PA_DIV);
    return sw_utci_poly_element(dtm, ta, va, pa, use_libm);
}

// --- dense pass 1: per-row valid counts --------------------------------
__global__ void sw_utci_count_kernel(const float* __restrict__ ta,
                                     const float* __restrict__ rh,
                                     const float* __restrict__ tmrt,
                                     const float* __restrict__ va,
                                     long long* __restrict__ counts,
                                     int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    long long c = 0;
    const float* rt_ = ta + (long long)r * cols;
    const float* rr_ = rh + (long long)r * cols;
    const float* rm_ = tmrt + (long long)r * cols;
    const float* rv_ = va + (long long)r * cols;
    for (int j = 0; j < cols; ++j) {
        if (sw_utci_valid(rt_[j], rr_[j], rm_[j], rv_[j])) c += 1;
    }
    counts[r] = c;
}

// --- dense pass 2: fill valid lanes (thread per cell) -------------------
// The lane ordinal k = offs[r] + (valid cells before c in row r) is
// recomputed by a bounded in-row scan — pure integer work, no schedule
// dependence, and orders of magnitude cheaper than the 211-op element.
__global__ void sw_utci_fill_kernel(const float* __restrict__ ta,
                                    const float* __restrict__ rh,
                                    const float* __restrict__ tmrt,
                                    const float* __restrict__ va,
                                    const long long* __restrict__ offs,
                                    float* __restrict__ out,
                                    long long n, int T, int grain,
                                    int rows, int cols) {
    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (cell >= total) return;
    int r = (int)(cell / cols);
    int c = (int)(cell - (long long)r * cols);
    float v_ta = ta[cell], v_rh = rh[cell], v_tmrt = tmrt[cell],
          v_va = va[cell];
    if (!sw_utci_valid(v_ta, v_rh, v_tmrt, v_va)) return;  // out stays -999
    long long k = offs[r];
    const float* rt_ = ta + (long long)r * cols;
    const float* rr_ = rh + (long long)r * cols;
    const float* rm_ = tmrt + (long long)r * cols;
    const float* rv_ = va + (long long)r * cols;
    for (int j = 0; j < c; ++j) {
        if (sw_utci_valid(rt_[j], rr_[j], rm_[j], rv_[j])) k += 1;
    }
    out[cell] = sw_utci_element(v_ta, v_rh, v_tmrt, v_va, k, n, T, grain);
}

// --- sparse: one thread per pre-compacted valid element ------------------
__global__ void sw_utci_sparse_kernel(const float* __restrict__ ta,
                                      const float* __restrict__ rh,
                                      const float* __restrict__ tmrt,
                                      const float* __restrict__ va,
                                      float* __restrict__ out,
                                      long long n, int T, int grain) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    out[k] = sw_utci_element(ta[k], rh[k], tmrt[k], va[k], k, n, T, grain);
}

// --- T13 graph-bucket variant ----------------------------------------------
// Same lane math, but the logical extent n lives in DEVICE memory
// (n_dev[0]) so an instantiated CUDA graph with FIXED launch geometry
// serves every n inside its capacity bucket (a memcpy node inside the
// graph refreshes the word from pinned host memory per replay). Lanes
// k >= n are bucket padding: the guard returns BEFORE any input read or
// output write — padding targets must never leak into results (T13 RED
// witness 2, tests/ultrafast/test_cuda_graphs.py).
extern "C" __global__ void sw_utci_sparse_bucket_kernel(
    const float* __restrict__ ta, const float* __restrict__ rh,
    const float* __restrict__ tmrt, const float* __restrict__ va,
    float* __restrict__ out, const long long* __restrict__ n_dev,
    int T, int grain) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = n_dev[0];
    if (k >= n) return;
    out[k] = sw_utci_element(ta[k], rh[k], tmrt[k], va[k], k, n, T, grain);
}

