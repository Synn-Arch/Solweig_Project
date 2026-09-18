// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_fold.cu — CUDA SVF fold kernel (ultrafast bitwise plan, TASKS T12;
// DESIGN 3.4/6.3). One thread per target cell replays, IN EXECUTION ORDER,
// the exact float32 recurrence of solweig_core/numba_cpu/svf_fold.py
// (the frozen T07 reference, itself bit-equal to the original
// svf_calculator fold), consuming the T07 packed bit state and the FROZEN
// constant tables as DATA (annulus weights and direction predicates are
// captured bit patterns — never recomputed on device).
//
// Preserved semantics (every row of the T07 module contract):
//   * patch-major / annulus-minor recurrence in the ORIGINAL enumeration
//     order: for patch p, for annulus a: svfveg += w_iso * v,
//     svfaveg += w_iso * b, then the E/S/W/N aniso pairs — same
//     per-accumulator operation order. No pre-summed annulus weights, no
//     reordered patch reduction (both change float32 accumulation and are
//     RED-designated — the raw-bit pins are the designated killer).
//   * every multiply and add is a separate IEEE float32 op (the strict
//     build's --fmad=false makes a + b * c compile to mul-then-add; no
//     contraction, no reassociation);
//   * bit reads become exact 0.0/1.0 float32 values before the first
//     multiply;
//   * ``last`` = 3.0459e-004 where vegdem2 == 0.0 (-0.0 included, NaN
//     excluded) is added to svfSveg/svfWveg/svfSaveg/svfWaveg only;
//   * the 10 veg clamps run in the source order ([x > 1.] = 1. — NaN-safe);
//   * SVFtotal = svf_building - (1 - svfveg) * (1 - trans).
//
// No reductions, no cross-thread communication: a lane's outputs depend
// only on (target cell, packed bytes, tables) — never the schedule. The
// masked route leaves every unmasked cell's 11 outputs untouched (the
// host pre-fills the output buffer with the caller's base planes), which
// is the affected-chunk-only contract of the CPU wrapper.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include "../include/sw_kernels_decl.cuh"
#include "../include/sw_strict_math.cuh"

// bytes_per_patch(153) — bitplanes little-order packing, 20 bytes/cell.
#define SW_FOLD_PATCHES 153
#define SW_FOLD_BYTES ((SW_FOLD_PATCHES + 7) / 8)
#define SW_FOLD_RING_SLOTS 12
#define SW_FOLD_OUT 11

extern "C" __global__ void sw_fold_kernel(
    const unsigned char *__restrict__ veg_bytes,
    const unsigned char *__restrict__ vbsh_bytes,
    const float *__restrict__ vegdem2, const float *__restrict__ svf_building,
    const float *__restrict__ w_iso, const float *__restrict__ w_aniso,
    const int *__restrict__ ring, const int *__restrict__ na,
    const signed char *__restrict__ dir_e, const signed char *__restrict__ dir_s,
    const signed char *__restrict__ dir_w, const signed char *__restrict__ dir_n,
    const unsigned char *__restrict__ mask, float *__restrict__ out,
    float last_const, float one_minus_trans, int rows, int cols,
    int do_mask) {

    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (cell >= total) {
        return;
    }
    if (do_mask && !mask[cell]) {
        return;  // affected-chunk-only: keep the caller's base bits
    }
    const float zero = 0.0f;
    const float one = 1.0f;
    const unsigned char *veg_cell = veg_bytes + cell * SW_FOLD_BYTES;
    const unsigned char *vbsh_cell = vbsh_bytes + cell * SW_FOLD_BYTES;

    float svfveg = zero;
    float svfaveg = zero;
    float svfEveg = zero;
    float svfEaveg = zero;
    float svfSveg = zero;
    float svfSaveg = zero;
    float svfWveg = zero;
    float svfWaveg = zero;
    float svfNveg = zero;
    float svfNaveg = zero;

    for (int p = 0; p < SW_FOLD_PATCHES; ++p) {
        int r = ring[p];
        float v = (float)((veg_cell[p >> 3] >> (p & 7)) & 1);
        float b = (float)((vbsh_cell[p >> 3] >> (p & 7)) & 1);
        int n_ann = na[r];
        signed char de = dir_e[p];
        signed char ds = dir_s[p];
        signed char dw = dir_w[p];
        signed char dn = dir_n[p];
        for (int a = 0; a < n_ann; ++a) {
            float w = w_iso[r * SW_FOLD_RING_SLOTS + a];
            float wa = w_aniso[r * SW_FOLD_RING_SLOTS + a];
            svfveg = svfveg + w * v;
            svfaveg = svfaveg + w * b;
            if (de) {
                svfEveg = svfEveg + wa * v;
                svfEaveg = svfEaveg + wa * b;
            }
            if (ds) {
                svfSveg = svfSveg + wa * v;
                svfSaveg = svfSaveg + wa * b;
            }
            if (dw) {
                svfWveg = svfWveg + wa * v;
                svfWaveg = svfWaveg + wa * b;
            }
            if (dn) {
                svfNveg = svfNveg + wa * v;
                svfNaveg = svfNaveg + wa * b;
            }
        }
    }
    // last correction: 3.0459e-004 where vegdem2 == 0.0
    // (-0.0 == 0.0 is true; NaN == 0.0 is false)
    float last_v = (vegdem2[cell] == zero) ? last_const : zero;
    svfSveg = svfSveg + last_v;
    svfWveg = svfWveg + last_v;
    svfSaveg = svfSaveg + last_v;
    svfWaveg = svfWaveg + last_v;
    // the 10 veg clamps (original order, [x > 1.] = 1.)
    if (svfveg > one) {
        svfveg = one;
    }
    if (svfaveg > one) {
        svfaveg = one;
    }
    if (svfEveg > one) {
        svfEveg = one;
    }
    if (svfEaveg > one) {
        svfEaveg = one;
    }
    if (svfSveg > one) {
        svfSveg = one;
    }
    if (svfSaveg > one) {
        svfSaveg = one;
    }
    if (svfWveg > one) {
        svfWveg = one;
    }
    if (svfWaveg > one) {
        svfWaveg = one;
    }
    if (svfNveg > one) {
        svfNveg = one;
    }
    if (svfNaveg > one) {
        svfNaveg = one;
    }
    // svftotal: the only NaN source is svf_building itself (the weights are
    // finite captured bits and v/b are exact 0.0/1.0). CPU SSE sub returns
    // the quieted INPUT NaN (payload + sign preserved); CUDA canonicalizes
    // NaN arithmetic to 0x7fffffff — quiet the input explicitly instead.
    float svb = svf_building[cell];
    float svftotal;
    if (svb != svb) {
        svftotal = sw_i2f(sw_f2i(svb) | 0x00400000);
    } else {
        svftotal = svb - (one - svfveg) * one_minus_trans;
    }
    float *out_cell = out + cell * SW_FOLD_OUT;
    out_cell[0] = svfveg;
    out_cell[1] = svfEveg;
    out_cell[2] = svfSveg;
    out_cell[3] = svfWveg;
    out_cell[4] = svfNveg;
    out_cell[5] = svfaveg;
    out_cell[6] = svfEaveg;
    out_cell[7] = svfSaveg;
    out_cell[8] = svfWaveg;
    out_cell[9] = svfNaveg;
    out_cell[10] = svftotal;
}
