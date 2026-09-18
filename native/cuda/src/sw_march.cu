// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_march.cu — CUDA march kernels (ultrafast bitwise plan, TASKS T12;
// DESIGN.ko.md 7.1/7.6). One thread per TARGET cell replays, IN EXECUTION
// ORDER, the exact primitive graph of solweig_core/numba_cpu/march.py
// (the frozen T04 reference, itself bit-equal to the original torch
// kernels), driven by the frozen T03 step tables consumed as DATA
// (dx/dy int32, dz/dzprev raw float32 bits — never recomputed).
//
// Preserved semantics (every row of DESIGN 7.1):
//   * out-of-bounds source reads see the EXACT ZERO temporary value —
//     never -inf, never 0 - dz;
//   * max follows torch elementwise semantics (sw_torch_maximum);
//   * the vb accumulator stays float32 end-to-end (vbsh == 2.0 regime);
//   * the final graph applies threshold / subtraction / inversions in the
//     source's exact order with the source's f32 casts.
//
// No reductions, no cross-thread communication: a lane's outputs depend
// only on (target cell, table, planes) — never the schedule (RED witness
// 7) and no packed-byte writes exist at this stage (RED witness 6 comes
// with the bitplane packer).
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include "../include/sw_kernels_decl.cuh"
#include "../include/sw_strict_math.cuh"

// svf_shadow: solweig_gpu/shadow.py:166-322 march (bush-free domain).
// First step (s == 0) carries the vegetation condition, the TARGET-LOCAL
// trunk gate (vegdem2[t] > a[t]) and the vbshvegsh.zero_() reset.
extern "C" __global__ void sw_march_svf_shadow(
    const float *__restrict__ a, const float *__restrict__ vegdem,
    const float *__restrict__ vegdem2, const int *__restrict__ dxs,
    const int *__restrict__ dys, const float *__restrict__ dzs,
    int rows, int cols, int count,
    float *__restrict__ sh_out, float *__restrict__ vegsh_out,
    float *__restrict__ vbsh_out) {

    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (cell >= total) {
        return;
    }
    int i = (int)(cell / cols);
    int j = (int)(cell % cols);
    const float zero = 0.0f;
    const float one = 1.0f;
    const float big = 1000.0f;

    float a_t = a[cell];
    float f = a_t;                  // f = a.clone()
    float sh = zero;
    float vegsh = zero;             // vegsh = zeros + bushplant (0 here)
    float vbsh = zero;

    for (int s = 0; s < count; ++s) {
        float dz = dzs[s];
        int si = i + dxs[s];
        int sj = j + dys[s];
        float tv, tv2, ta;
        if (si >= 0 && si < rows && sj >= 0 && sj < cols) {
            long long src = (long long)si * cols + sj;
            tv = vegdem[src] - dz;
            tv2 = vegdem2[src] - dz;
            ta = a[src] - dz;
        } else {
            tv = zero;              // temporaries keep their zero init
            tv2 = zero;
            ta = zero;
        }
        f = sw_torch_maximum(f, ta);
        sh = (f > a_t) ? one : zero;
        float fab = (tv > a_t) ? one : zero;
        float gab = (tv2 > a_t) ? one : zero;
        float vegsh2 = fab - gab;
        vegsh = sw_torch_maximum(vegsh, vegsh2);
        if (vegsh * sh > zero) {    // building suppression
            vegsh = zero;
        }
        vbsh = vegsh + vbsh;        // accumulator (source operand order)
        if (s == 0) {               // index == 1.: first-step exception
            float fv = tv - ta;
            if (fv <= zero) {
                fv = big;
            }
            if (fv < dz) {
                vegsh = one;
            }
            if (vegdem2[cell] > a_t) {   // target-local trunk gate
                vegsh = vegsh * one;
            } else {
                vegsh = vegsh * zero;
            }
            vbsh = zero;            // vbshvegsh.zero_()
        }
    }
    // final graph, source order (shadow.py:303-316, bush tail skipped)
    sh = one - sh;
    if (vbsh > zero) {
        vbsh = one;
    }
    vbsh = vbsh - vegsh;
    if (vegsh > zero) {
        vegsh = one;
    }
    vegsh = one - vegsh;
    vbsh = one - vbsh;
    sh_out[cell] = sh;
    vegsh_out[cell] = vegsh;
    vbsh_out[cell] = vbsh;
}

// wallheight_23: solweig_gpu/solweig.py:1079-1205 march section. Step 0 is
// the (0, 0) self comparison; every step additionally consumes the PREVIOUS
// step's dz (dzprev, carried by the frozen table) through the
// lastfabovea/lastgabovea planes.
extern "C" __global__ void sw_march_wallheight23(
    const float *__restrict__ a, const float *__restrict__ vegdem,
    const float *__restrict__ vegdem2, const int *__restrict__ dxs,
    const int *__restrict__ dys, const float *__restrict__ dzs,
    const float *__restrict__ dzprevs, int rows, int cols, int count,
    float *__restrict__ sh_out, float *__restrict__ vegsh_out,
    float *__restrict__ vbsh_out) {

    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (cell >= total) {
        return;
    }
    int i = (int)(cell / cols);
    int j = (int)(cell % cols);
    const float zero = 0.0f;
    const float one = 1.0f;
    const float four = 4.0f;

    float a_t = a[cell];
    float f = a_t;                  // f = a
    float sh = zero;
    float vegsh = zero;             // zeros + bushplant (0 under guard)
    float vbsh = zero;

    for (int s = 0; s < count; ++s) {
        float dz = dzs[s];
        float dzp = dzprevs[s];
        int si = i + dxs[s];
        int sj = j + dys[s];
        float tv, tv2, ta, tlf, tlg;
        if (si >= 0 && si < rows && sj >= 0 && sj < cols) {
            long long src = (long long)si * cols + sj;
            tv = vegdem[src] - dz;
            tv2 = vegdem2[src] - dz;
            ta = a[src] - dz;
            tlf = vegdem[src] - dzp;
            tlg = vegdem2[src] - dzp;
        } else {
            tv = zero;
            tv2 = zero;
            ta = zero;
            tlf = zero;
            tlg = zero;
        }
        f = sw_torch_maximum(f, ta);
        sh = (f > a_t) ? one : zero;
        float fa = (tv > a_t) ? one : zero;
        float ga = (tv2 > a_t) ? one : zero;
        float lfa = (tlf > a_t) ? one : zero;
        float lga = (tlg > a_t) ? one : zero;
        float v2 = fa + ga;         // fabovea + gabovea
        v2 = v2 + lfa;              // + lastfabovea.float()
        v2 = v2 + lga;              // + lastgabovea.float()
        if (v2 == four) {
            v2 = zero;
        }
        if (v2 > zero) {
            v2 = one;
        }
        vegsh = sw_torch_maximum(vegsh, v2);
        if (vegsh * sh > zero) {    // building suppression
            vegsh = zero;
        }
        vbsh = vbsh + vegsh;        // accumulator (source operand order)
    }
    // final graph, source order (solweig.py:1198-1205)
    sh = one - sh;
    if (vbsh > zero) {
        vbsh = one;
    }
    vbsh = vbsh - vegsh;
    if (vegsh > zero) {
        vegsh = one;
    }
    vegsh = one - vegsh;
    vbsh = one - vbsh;
    sh_out[cell] = sh;
    vegsh_out[cell] = vegsh;
    vbsh_out[cell] = vbsh;
}
