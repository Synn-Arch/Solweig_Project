// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_primitives.cu — device-side PRIMITIVE PARITY PROBES (TASKS T12).
//
// Each kernel consumes committed pin vectors (f32 bit patterns) and writes
// raw results; equality against the CPU canonical primitive bits is judged
// HOST-side on uint32 views (tests/ultrafast/test_cuda_primitives.py).
// These probes are the primary gate of the whole task: the strict math
// layer must be bit-equal to solweig_core/numba_cpu/math_compat BEFORE
// any march/fold/radiation/UTCI stage is trusted.
//
// Mutation switches (build.py variant flags — the RED witnesses):
//   SW_MUT_APPROX_EXPF -> __expf in place of the SLEEF xexpf port
//   SW_MUT_LIBM_LOGF   -> CUDA libm logf in place of the SLEEF port
// (FMA contraction / FTZ / fast-math are build-flag mutations, no source
//  switch needed: the arith/contract probes below flip bits under them.)
// ---------------------------------------------------------------------------
#include "../include/sw_strict_math.cuh"
#include "../include/sw_kernels_decl.cuh"

#ifdef SW_MUT_APPROX_EXPF
#define SW_EXPF_PROBE(x) __expf(x)
#else
#define SW_EXPF_PROBE(x) sw_sleef_expf(x)
#endif

#ifdef SW_MUT_LIBM_LOGF
#define SW_LOGF_PROBE(x) logf(x)
#else
#define SW_LOGF_PROBE(x) sw_sleef_logf_u1(x)
#endif

extern "C" __global__ void sw_probe_expf(const float* __restrict__ x,
                                         float* __restrict__ y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = SW_EXPF_PROBE(x[i]);
}

extern "C" __global__ void sw_probe_logf(const float* __restrict__ x,
                                         float* __restrict__ y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = SW_LOGF_PROBE(x[i]);
}

extern "C" __global__ void sw_probe_powf(const float* __restrict__ x,
                                         const float* __restrict__ e,
                                         float* __restrict__ y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sw_sleef_powf(x[i], e[i]);
}

extern "C" __global__ void sw_probe_opmath_powf(const float* __restrict__ x,
                                                float* __restrict__ y,
                                                long long n, int e) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sw_opmath_powf(x[i], e);
}

// plain IEEE ops — witnesses FTZ (subnormal operands/results) and any
// div/sqrt approximation drift
extern "C" __global__ void sw_probe_arith(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ add,
                                          float* __restrict__ sub,
                                          float* __restrict__ mul,
                                          float* __restrict__ dv,
                                          long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        add[i] = a[i] + b[i];
        sub[i] = a[i] - b[i];
        mul[i] = a[i] * b[i];
        dv[i] = a[i] / b[i];
    }
}

// contraction witness: a*b + c written as two separate IEEE ops —
// --fmad=false keeps it un-fused, --fmad=true contracts it
extern "C" __global__ void sw_probe_contract(const float* __restrict__ a,
                                             const float* __restrict__ b,
                                             const float* __restrict__ c,
                                             float* __restrict__ y,
                                             long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = a[i] * b[i];
        y[i] = t + c[i];
    }
}

extern "C" __global__ void sw_probe_maximum(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ y,
                                            long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sw_torch_maximum(a[i], b[i]);
}

// first-NaN-operand wrappers (torch elementwise NaN-payload semantics)
extern "C" __global__ void sw_probe_narith(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           float* __restrict__ add,
                                           float* __restrict__ sub,
                                           float* __restrict__ mul,
                                           float* __restrict__ dv,
                                           long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        add[i] = sw_nadd(a[i], b[i]);
        sub[i] = sw_nsub(a[i], b[i]);
        mul[i] = sw_nmul(a[i], b[i]);
        dv[i] = sw_ndiv(a[i], b[i]);
    }
}

// lane-class probe: the torch chunk layout predicate, checked for pure
// ordinal+extent dependence (RED witness 7 — mutations of sw_is_libm_lane
// that read the schedule cannot pass the (i, n, T, grain) table)
extern "C" __global__ void sw_probe_is_libm_lane(const long long* __restrict__ i_,
                                                 const long long* __restrict__ n_,
                                                 const int* __restrict__ T_,
                                                 const int* __restrict__ grain_,
                                                 int* __restrict__ y,
                                                 long long cnt) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q < cnt)
        y[q] = sw_is_libm_lane(i_[q], n_[q], T_[q], grain_[q]) ? 1 : 0;
}
