// SPDX-License-Identifier: GPL-3.0-only
// ---------------------------------------------------------------------------
// sw_strict_math.cuh — STRICT float32 arithmetic + SLEEF device ports
// (ultrafast bitwise plan, TASKS T12; DESIGN.ko.md 6.2/13.5).
//
// The canonical CUDA profile reproduces the canonical CPU primitive BITS
// (solweig_core/numba_cpu/math_compat.py, the committed T09 reference):
//
//   * add/sub/mul/div/sqrt are plain IEEE round-to-nearest ops. The build
//     pins --fmad=false --ftz=false --prec-div=true --prec-sqrt=true so no
//     contraction, no flush-to-zero, no approximate div/sqrt is ever
//     inserted. Enabling FMA contraction / FTZ / __expf-class intrinsics
//     anywhere in this layer is a RED witness (tests/ultrafast mutations).
//   * exp  -> SLEEF xexpf scalar port          (math_compat.sleef_expf)
//   * log  -> SLEEF xlogf_u1 scalar port       (math_compat.sleef_logf_u1)
//   * powf -> SLEEF xpowf scalar port          (math_compat.sleef_powf)
//   * torch opmath pow (Scalar exponent, scalar-tail lanes, f64 pow
//     single-rounded to f32) -> sw_opmath_powf: EXACT integer-mantissa
//     x^e for integer e in [4,6] (the only exponents the UTCI layout
//     model routes to the opmath path), rounded once to double (nearest,
//     ties-to-even) then once to float — bit-equal to
//     float32(pow(float64(x), float64(e))) for exactly representable
//     products, pinned against the CPU port on the committed lattices.
//
// Every decimal constant is written as an EXACT hex float literal (bit
// patterns generated from the CPU reference module) — nvcc decimal parsing
// never gets the chance to disagree with numpy.
//
// SLEEF ports follow sleefsimdsp.c (SLEEF 3.8, Boost Software License 1.0,
// Naoki Shibata and contributors) at the exact association of the CPU
// reference; the ONLY fused multiply-adds are the explicit __fmaf_rn calls
// below (SLEEF's own vfma sites).
// ---------------------------------------------------------------------------
#ifndef SW_STRICT_MATH_CUH_
#define SW_STRICT_MATH_CUH_

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// strict helpers (bit discipline)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int   sw_f2i(float x) { return __float_as_int(x); }
__device__ __forceinline__ float sw_i2f(int i)   { return __int_as_float(i); }
// llvm.rint.f32 mirror (round-to-nearest-even under default RN mode)
__device__ __forceinline__ float sw_rintf(float x) { return rintf(x); }

// torch elementwise max(a, b): NaN propagates from either operand,
// otherwise the larger, with the SECOND operand on ties (a!=a -> a is
// checked first, then a > b; math_compat._torch_maximum / march.py).
__device__ __forceinline__ float sw_torch_maximum(float x, float y) {
    return (x != x || x > y) ? x : y;
}

// first-NaN-operand arithmetic (torch elementwise NaN-payload semantics;
// math_compat.nadd/nsub/nmul/ndiv). Finite results are the plain IEEE op.
__device__ __forceinline__ float sw_nadd(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return a + b;
}
__device__ __forceinline__ float sw_nsub(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return a - b;
}
__device__ __forceinline__ float sw_nmul(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return a * b;
}
__device__ __forceinline__ float sw_ndiv(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return a / b;
}

// ---------------------------------------------------------------------------
// SLEEF xexpf (sleefsimdsp.c:1314) — torch.exp, every lane class
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sw_sleef_expf(float d) {
    const float R_LN2F = 0x1.7154760000000p+0f;
    const float L2UF   = 0x1.62e4000000000p-1f;
    const float L2LF   = 0x1.7f7d1c0000000p-20f;
    int q = (int)sw_rintf(d * R_LN2F);
    float s = __fmaf_rn((float)q, -L2UF, d);
    s = __fmaf_rn((float)q, -L2LF, s);

    float u = 0x1.a057b40000000p-13f;
    u = __fmaf_rn(u, s, 0x1.6d2d920000000p-10f);
    u = __fmaf_rn(u, s, 0x1.11114c0000000p-7f);
    u = __fmaf_rn(u, s, 0x1.5554f40000000p-5f);
    u = __fmaf_rn(u, s, 0x1.5555560000000p-3f);
    u = __fmaf_rn(u, s, 0x1.000000p-1f);

    u = 1.0f + __fmaf_rn(s * s, u, s);

    // vldexp2_vf_vf_vi2(u, q): u * 2^(q>>1) * 2^(q-(q>>1))
    int h = q >> 1;
    u = u * sw_i2f((int)(((unsigned)(h + 127)) << 23))
          * sw_i2f((int)(((unsigned)(q - h + 127)) << 23));

    if (d < -104.0f) u = 0.0f;
    if (100.0f < d)  u = __int_as_float(0x7F800000);
    // NaN input: the CPU port (LLVM fma/mul on x86) propagates the INPUT
    // payload (quieted, sign preserved) through the whole chain; CUDA
    // device arithmetic canonicalizes NaN results to 0x7fffffff. Mirror
    // the CPU rule explicitly (RED witness 4 territory).
    if (d != d) u = sw_i2f(sw_f2i(d) | 0x00400000);
    return u;
}

// ---------------------------------------------------------------------------
// SLEEF xlogf_u1 (sleefsimdsp.c:1528) — torch.log
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sw_sleef_logf_u1(float d0) {
    const float FLT_MIN = 0x1.000000p-126f;
    const float TWO64   = 0x1.000000p+64f;
    const float LN2HI   = 0x1.62e4300000000p-1f;
    const float LN2LO   = -0x1.05c6100000000p-29f;
    const float INF     = __int_as_float(0x7F800000);
    const float NAN_Q   = __int_as_float(0x7FC00000);

    float d = d0;
    bool o = d < FLT_MIN;
    if (o) d = d * TWO64;

    int e = (int)(((unsigned)(sw_f2i(d * 0x1.5555560000000p+0f) >> 23) & 0xFFu)) - 127;
    float m = sw_i2f(sw_f2i(d) + ((-e) << 23));
    if (o) e = e - 64;

    float s_x = LN2HI * (float)e;
    float s_y = __fmaf_rn(LN2LO, (float)e, __fmaf_rn(LN2HI, (float)e, -s_x));

    float n_x = m - 1.0f;
    float n_v = n_x + 1.0f;
    float n_y = (-1.0f - (n_x - n_v)) + (m - n_v);

    float dn_x = 1.0f + m;
    float dn_v = dn_x - 1.0f;
    float dn_y = (1.0f - (dn_x - dn_v)) + (m - dn_v);

    float t_rcp = 1.0f / dn_x;
    float x_x = n_x * t_rcp;
    float u_ = __fmaf_rn(t_rcp, n_x, -x_x);
    float v_ = __fmaf_rn(-dn_x, t_rcp, 1.0f);
    v_ = __fmaf_rn(-dn_y, t_rcp, v_);
    float x_y = __fmaf_rn(x_x, v_, __fmaf_rn(n_y, t_rcp, u_));

    float x2 = x_x * x_x;

    float t = 0x1.35feb80000000p-2f;
    t = __fmaf_rn(t, x2, 0x1.9933940000000p-2f);
    t = __fmaf_rn(t, x2, 0x1.5555b40000000p-1f);

    float sc_x = x_x * 2.0f;
    float sc_y = x_y * 2.0f;
    float s2_x = s_x + sc_x;
    float s2_y = ((s_x - s2_x) + sc_x) + s_y + sc_y;

    float term = (x2 * x_x) * t;
    float s3_x = s2_x + term;
    float s3_y = ((s2_x - s3_x) + term) + s2_y + 0.0f;

    float r = s3_x + s3_y;

    if (d0 == INF) r = INF;
    if (d0 < 0.0f || d0 != d0) r = NAN_Q;
    if (d0 == 0.0f) r = -INF;
    return r;
}

// ---------------------------------------------------------------------------
// SLEEF logkf (sleefsimdsp.c:1538) -> (s_x, s_y)   [xpowf internals]
// ---------------------------------------------------------------------------
__device__ __forceinline__ void sw_logkf(float d0, float *s3_x, float *s3_y) {
    const float FLT_MIN = 0x1.000000p-126f;
    const float TWO64   = 0x1.000000p+64f;
    const float LN2HI   = 0x1.62e4300000000p-1f;
    const float LN2LO   = -0x1.05c6100000000p-29f;

    float d = d0;
    bool o = d < FLT_MIN;
    if (o) d = d * TWO64;
    int e = (int)(((unsigned)(sw_f2i(d * 0x1.5555560000000p+0f) >> 23) & 0xFFu)) - 127;
    float m = sw_i2f(sw_f2i(d) + ((-e) << 23));
    if (o) e = e - 64;

    float s_x = LN2HI * (float)e;
    float s_y = __fmaf_rn(LN2LO, (float)e, __fmaf_rn(LN2HI, (float)e, -s_x));

    float n_x = m - 1.0f;
    float n_v = n_x + 1.0f;
    float n_y = (-1.0f - (n_x - n_v)) + (m - n_v);
    float dn_x = 1.0f + m;
    float dn_v = dn_x - 1.0f;
    float dn_y = (1.0f - (dn_x - dn_v)) + (m - dn_v);

    float t_rcp = 1.0f / dn_x;
    float x_x = n_x * t_rcp;
    float u_ = __fmaf_rn(t_rcp, n_x, -x_x);
    float v_ = __fmaf_rn(-dn_x, t_rcp, 1.0f);
    v_ = __fmaf_rn(-dn_y, t_rcp, v_);
    float x_y = __fmaf_rn(x_x, v_, __fmaf_rn(n_y, t_rcp, u_));

    float x2_x = x_x * x_x;
    float x2_y = __fmaf_rn(x_x + x_x, x_y, __fmaf_rn(x_x, x_x, -x2_x));

    float t = 0x1.ec2d140000000p-3f;
    t = __fmaf_rn(t, x2_x, 0x1.23f4940000000p-2f);
    t = __fmaf_rn(t, x2_x, 0x1.999bb20000000p-2f);
    float c_x = 0x1.5555540000000p-1f;
    float c_y = 0x1.fb67060000000p-29f;

    float sc_x = x_x * 2.0f;
    float sc_y = x_y * 2.0f;
    float s2_x = s_x + sc_x;
    float s2_y = ((s_x - s2_x) + sc_x) + s_y + sc_y;

    float a_x = x2_x * x_x;
    float a_y = __fmaf_rn(x2_x, x_y, __fmaf_rn(x2_y, x_x, __fmaf_rn(x2_x, x_x, -a_x)));

    float b_x = x2_x * t;
    float b_y = __fmaf_rn(x2_y, t, __fmaf_rn(x2_x, t, -b_x));

    float cc_x = b_x + c_x;
    float cc_v = cc_x - b_x;
    float cc_t = (b_x - (cc_x - cc_v)) + (c_x - cc_v);
    float cc_y = cc_t + (b_y + c_y);

    float d_x = a_x * cc_x;
    float d_y = __fmaf_rn(a_x, cc_y, __fmaf_rn(a_y, cc_x, __fmaf_rn(a_x, cc_x, -d_x)));

    float r_x = s2_x + d_x;
    float r_y = ((s2_x - r_x) + d_x) + s2_y + d_y;
    *s3_x = r_x;
    *s3_y = r_y;
}

// vldexp_vf_vf_vi2 (sleefsimdsp.c:316) — 4-mul split with clamp
__device__ __forceinline__ float sw_vldexp(float x, int q) {
    int m = q >> 31;
    m = (((m + q) >> 6) - m) << 4;
    q = q - (m << 2);
    m = m + 127;
    if (!(m > 0)) m = 0;
    if (m > 255) m = 255;
    float u = sw_i2f((int)(((unsigned)m) << 23));
    x = (((x * u) * u) * u) * u;
    u = sw_i2f((int)(((unsigned)(q + 127)) << 23));
    return x * u;
}

// SLEEF expkf (sleefsimdsp.c:1569)
__device__ __forceinline__ float sw_expkf(float d_x, float d_y) {
    const float R_LN2F = 0x1.7154760000000p+0f;
    const float L2UF   = 0x1.62e4000000000p-1f;
    const float L2LF   = 0x1.7f7d1c0000000p-20f;

    float u = (d_x + d_y) * R_LN2F;
    int q = (int)sw_rintf(u);

    float s_x = d_x + (float)q * -L2UF;
    float v_ = s_x - d_x;
    float s_y = ((d_x - (s_x - v_)) + ((float)q * -L2UF - v_)) + d_y;
    float s2_x = s_x + (float)q * -L2LF;
    v_ = s2_x - s_x;
    float s2_y = ((s_x - (s2_x - v_)) + ((float)q * -L2LF - v_)) + s_y;

    float s3_x = s2_x + s2_y;
    float s3_y = (s2_x - s3_x) + s2_y;

    u = 0x1.655dec0000000p-10f;
    u = __fmaf_rn(u, s3_x, 0x1.1222d60000000p-7f);
    u = __fmaf_rn(u, s3_x, 0x1.555e980000000p-5f);
    u = __fmaf_rn(u, s3_x, 0x1.5554bc0000000p-3f);
    u = __fmaf_rn(u, s3_x, 0x1.fffff60000000p-2f);

    float sq_x = s3_x * s3_x;
    float sq_y = __fmaf_rn(s3_x + s3_x, s3_y, __fmaf_rn(s3_x, s3_x, -sq_x));
    float mu_x = sq_x * u;
    float mu_y = __fmaf_rn(sq_y, u, __fmaf_rn(sq_x, u, -mu_x));
    float t_x = s3_x + mu_x;
    float t_y = ((s3_x - t_x) + mu_x) + s3_y + mu_y;

    float t2_x = 1.0f + t_x;
    float t2_y = (1.0f - t2_x) + t_x + t_y;

    u = t2_x + t2_y;
    u = sw_vldexp(u, q);

    if (d_x < -104.0f) u = 0.0f;
    return u;
}

// ---------------------------------------------------------------------------
// SLEEF xpowf (sleefsimdsp.c:1618) — torch.pow vector-body lanes
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sw_sleef_powf(float x, float y) {
    const float BIG24 = 16777216.0f;
    const float INF   = __int_as_float(0x7F800000);
    const float NAN_Q = __int_as_float(0x7FC00000);

    bool yisint = (truncf(y) == y) || (fabsf(y) > BIG24);
    bool yisodd = (((int)y & 1) == 1) && yisint && (fabsf(y) < BIG24);

    float l_x, l_y;
    sw_logkf(fabsf(x), &l_x, &l_y);
    float p_x = l_x * y;
    float p_y = __fmaf_rn(l_y, y, __fmaf_rn(l_x, y, -p_x));
    float result = sw_expkf(p_x, p_y);

    if (result != result) result = INF;

    float sg;
    if (x > 0.0f) {
        sg = 1.0f;
    } else if (yisint) {
        sg = yisodd ? -1.0f : 1.0f;
    } else {
        sg = NAN_Q;
    }
    // x < 0, non-integer y: the CPU port multiplies by the NAN_Q literal
    // and x86 keeps ITS payload (0x7fc00000) for finite/inf/zero results;
    // CUDA canonicalizes any NaN arithmetic to 0x7fffffff. Propagate the
    // literal payload explicitly.
    if (sg != sg) {
        result = NAN_Q;
    } else {
        result = result * sg;   // +-1.0: exact sign, no rounding
    }

    float efx = sw_i2f(sw_f2i(fabsf(x) - 1.0f) ^ (sw_f2i(y) & (-2147483647 - 1)));
    if (y == INF || y == -INF) {
        float r2 = (efx == 0.0f) ? 1.0f : INF;
        if (efx < 0.0f) r2 = 0.0f;
        result = r2;
    }

    if (x == INF || x == -INF || x == 0.0f) {
        bool xor_ = ((sw_f2i(y) & (-2147483647 - 1)) != 0) != (x == 0.0f);
        float v = xor_ ? 0.0f : INF;
        float sgn = yisodd ? x : 1.0f;
        v = sw_i2f(sw_f2i(v) ^ (sw_f2i(sgn) & (-2147483647 - 1)));
        result = v;
    }

    if (x != x || y != y) {
        result = sw_i2f(sw_f2i(result) | (-1));
    }

    if (y == 0.0f || x == 1.0f) {
        result = 1.0f;
    }
    return result;
}

// ---------------------------------------------------------------------------
// torch chunk-layout lane class (math_compat._is_libm_lane, T09 P5e/P9):
// True iff torch evaluates lane ``i`` of an n-lane elementwise op with the
// scalar remainder (opmath f64 pow) instead of the 4-lane SLEEF vector
// body — a PURE function of the lane ordinal and the extent, never the
// schedule (RED witness 7: any dependence on threadIdx/blockIdx here).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool sw_is_libm_lane(long long i, long long n,
                                                int T, int grain) {
    if (T <= 1 || n <= grain) {
        return i >= n - (n % 8);
    }
    long long teff = (n + (long long)grain - 1) / (long long)grain;
    if (teff > (long long)T) teff = (long long)T;
    long long C = (n + teff - 1) / teff;
    long long k = i / C;
    long long e = (k + 1) * C;
    if (e > n) e = n;
    long long L = e - k * C;
    return i >= e - (L % 8);
}

// ---------------------------------------------------------------------------
// torch opmath pow (Scalar exponent): pow(f64(x), f64(e)) single-rounded
// to f32 — the scalar-tail lanes of the torch chunk layout (T09 P9).
//
// Implemented as an EXACT integer-mantissa power for integer e in [4, 6]
// (the only exponents UTCI routes here): x^e for float32 x has at most
// 24*6 = 144 significand bits, carried exactly in a three-limb (144-bit)
// product — a plain unsigned __int128 WRAPS for e = 6 (m^6 up to 2^144),
// which is exactly the class of silent corruption the pins catch —
// rounded once to double (nearest, ties-to-even), then once to float.
// That is the correctly-rounded value of x^e, which is what the host
// reference (numba -> llvm.pow.f64 -> libm pow, correctly rounded on
// exactly representable products) produces. Pinned against the CPU port
// on the committed lattices (tests/ultrafast/test_cuda_primitives.py).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sw_opmath_powf(float x, int e) {
    // e in {4,5,6} guaranteed by the host wrapper (typed refusal otherwise)
    unsigned int bits = (unsigned int)sw_f2i(x);
    unsigned int frac = bits & 0x7FFFFFu;
    unsigned int be   = (bits >> 23) & 0xFFu;
    bool neg = (bits >> 31) != 0u;

    if (be == 0xFFu) {
        if (frac != 0u) {
            // glibc pow(NaN, y): quiet the sNaN, keep the payload, and
            // multiply by -1 when signbit(x) && y odd — the negation
            // FLIPS the NaN sign bit (pinned: e=4/6 ffc00000 -> ffc00000,
            // e=5 ffc00000 -> 7fc00000, 7f800001 -> 7fc000001-style quiet)
            unsigned int q = bits | 0x00400000u;
            if ((e & 1) && neg) q &= 0x7FFFFFFFu;
            return sw_i2f((int)q);
        }
        // +-inf: even e -> +inf, odd e -> sign preserved
        return (e & 1) ? x : __int_as_float(0x7F800000);
    }
    unsigned long long m;   // 24-bit significand integer
    int e2;                 // |x| == m * 2^e2
    if (be == 0u) {
        if (frac == 0u) {
            // pow(+-0, e>0): -0 only for odd e on a negative zero
            return ((e & 1) && neg) ? __int_as_float(0x80000000) : 0.0f;
        }
        int shift = __clz(frac) - 8;              // MSB -> bit 23; frac < 2^23
        m = (unsigned long long)(frac << shift);
        e2 = -149 - shift;
    } else {
        m = (unsigned long long)(frac | 0x800000u);
        e2 = (int)be - 150;
    }

    // P = m^e exactly, as three limbs: P = top*2^128 + hi*2^64 + lo.
    // m <= 2^24 and e <= 6, so top < 2^41 and no limb ever overflows.
    unsigned long long top = 0ull, hi = 0ull, lo = 1ull;
    for (int i = 0; i < e; ++i) {
        unsigned __int128 t = (unsigned __int128)lo * m;
        lo = (unsigned long long)t;
        unsigned __int128 t2 = (unsigned __int128)hi * m
                             + (unsigned long long)(t >> 64);
        hi = (unsigned long long)t2;
        top = top * m + (unsigned long long)(t2 >> 64);
    }

    // MSB position lp of P (P > 0 here; limbs below the MSB may be zero)
    int lp;
    if (top != 0ull)       lp = 128 + (63 - __clzll(top));
    else if (hi != 0ull)   lp = 64 + (63 - __clzll(hi));
    else                   lp = 63 - __clzll(lo);

    // round to 53 significand bits (nearest, ties-to-even):
    // keep bits [lp, lp-52]; guard at lp-53; sticky strictly below that
    int keep_lo = lp - 52;             // lowest kept bit position
    unsigned long long mant;
    int guard = 0, sticky = 0;
    if (keep_lo > 0) {
        // low 128 bits of P; bits at/above keep_lo < 128 always live here
        // (keep_lo <= lp-52 <= 143-52 = 91)
        unsigned __int128 low = ((unsigned __int128)hi << 64) | lo;
        unsigned __int128 shifted =
            (keep_lo >= 16)
                ? ((low >> keep_lo)
                   | (((unsigned __int128)top) << (128 - keep_lo)))
                : (low >> keep_lo);   // keep_lo < 16 => P < 2^68 => top == 0
        mant = (unsigned long long)shifted;
        guard = (int)((low >> (keep_lo - 1)) & 1);
        if (keep_lo >= 2) {
            sticky = ((low & ((((unsigned __int128)1) << (keep_lo - 1)) - 1))
                      != 0);
        }
    } else {
        // keep_lo <= 0: P fits 53 bits, no rounding (shift left is exact)
        mant = lo << (-keep_lo);      // top == hi == 0 in this regime
    }
    int exp2 = keep_lo + e * e2;       // value = mant * 2^exp2

    if (guard && (sticky || (mant & 1ull))) {
        mant += 1ull;
        if (mant == (1ull << 53)) {    // carry: 1.111...2 -> 10.000...0
            mant >>= 1;
            exp2 += 1;
        }
    }
    // double: value = (1 + frac) * 2^(exp2+52); always normal in this domain
    // (biased exponent in [129, ~1791] for e in {4,5,6} over all float32 x)
    unsigned long long dbits =
        ((unsigned long long)(exp2 + 52 + 1023) << 52)
        | (mant & 0xFFFFFFFFFFFFFull);
    float r = (float)__longlong_as_double((long long)dbits);  // RN f64 -> f32
    return ((e & 1) && neg) ? -r : r;  // integer-e sign (exact negation)
}

// torch.pow(t, Scalar-e) one lane (math_compat._torch_pow_scalar): SLEEF
// xpowf on vector-body lanes; scalar-tail lanes the opmath f64 pow
// single-rounded to f32 (the exact integer-mantissa form above).
// ``e`` arrives as 4.0f/5.0f/6.0f from the generated UTCI code — the
// generator never emits other exponents for _torch_pow_scalar calls.
__device__ __forceinline__ float sw_torch_pow_scalar(float x, float e,
                                                     bool use_libm) {
    if (use_libm) return sw_opmath_powf(x, (int)e);
    return sw_sleef_powf(x, e);
}

#endif  // SW_STRICT_MATH_CUH_
