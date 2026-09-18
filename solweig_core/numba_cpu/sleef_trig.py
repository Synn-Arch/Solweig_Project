"""Bit-exact numba ports of the SLEEF float32 u1 (= u10) trig kernels that
torch CPU f32 trigonometry lowers to on this host (arm64 macOS).

BUILD IDENTIFICATION (verified empirically, probe P3g):
torch 2.14.0 arm64 is built with -DAT_BUILD_ARM_VEC256_WITH_SLEEF, so
Vectorized<float> sin/cos/asin/acos map (aten/src/ATen/cpu/vec/vec128/
vec128_float_neon.h) to Sleef_{sin,cos,asin,acos}f4_u10.  torch pins
third_party/sleef @ 5a1d179df9cf652951b59010a2d2075372d67f68 and builds the
AdvSIMD config (ENABLE_ADVSIMD -> CONFIG 1 -> #define ENABLE_FMA_SP,
FULL_FP_ROUNDING, ACCURATE_SQRT).  A pinned-source native compile of exactly
that config matches torch bit-for-bit (0/800000 across sin/cos/asin/acos),
and so does this port (same probe).  The AdvSIMDNOFMA (CONFIG 2) build does
NOT match torch (93-172/200000 per function) — it is not what torch runs.

Ported verbatim from the pinned sources (src/libm/sleefsimdsp.c +
src/common/df.h, ENABLE_FMA_SP branch):

- xsinf_u1  (torch_sleefsimdsp.c:975)   -> sleef_sinf_u1
- xcosf_u1  (torch_sleefsimdsp.c:1073)  -> sleef_cosf_u1
- xasinf_u1 (torch_sleefsimdsp.c:1934)  -> sleef_asinf_u1
- xacosf_u1 (torch_sleefsimdsp.c:1954)  -> sleef_acosf_u1

Only the fast path is ported: |d| < TRIGRANGEMAX2f (=125.0) for sin/cos.
The SOLWEIG argument domains never approach that bound; the vector wrappers
at the bottom of this module enforce it (raising, never silently degrading).
The rempif slow path (table gather) is NOT ported — that is the honest
boundary of this module.

FMA discipline: exactly where SLEEF wrote vfma/vmla/vfmapn/vfmanp the port
calls math_compat.fmaf32 (llvm.fma.f32) with the SLEEF operand semantics:
  vfma(a,b,c)  = c + a*b   -> fmaf32(a, b, c)
  vmla(a,b,c)  = c + a*b   -> fmaf32(a, b, c)
  vfmanp(a,b,c)= c - a*b   -> fmaf32(-a, b, c)
  vfmapn(a,b,c)= a*b - c   -> fmaf32(a, b, -c)
numba fastmath is OFF everywhere so no contraction can appear anywhere
else.  f2i (bitcast) comes from math_compat.  vrint is np.rint
(round-to-nearest-even; vrndnq_f32).  vsqrt is np.sqrt on float32 (hardware
vsqrtq; ACCURATE_SQRT).  vrec is plain f32 division (vdivq).

The df ("double-float") helpers are transcriptions of the df.h FMA_SP
branch; the original macro name is in each comment.
"""

import numpy as np
from numba import njit

from .math_compat import f2i, fmaf32

F32 = np.float32

# misc.h constants (float literals; each is the exact f32 of the C double)
_PI_A2F = F32(3.1414794921875)
_PI_B2F = F32(0.00011315941810607910156)
_PI_C2F = F32(1.9841872589410058936e-09)
_TRIGRANGEMAX2F = F32(125.0)
_M_1_PI_F = F32(0.3183098861837906715377675267450287)  # (float)M_1_PI

# halved reductions (cosf_u1): -PI_A2f*0.5f etc — exact in f32
_PI_A2F_H = F32(-1.57073974609375)
_PI_B2F_H = F32(-5.657970905303955078e-05)
_PI_C2F_H = F32(-9.920936294705029468e-10)

# df π constants (vcast_vf2_f_f pairs)
_PI_DF_HI = F32(3.1415927410125732422)
_PI_DF_LO = F32(-8.7422776573475857731e-08)
_PI_O4_DF_HI = F32(F32(3.1415927410125732422) / F32(4.0))
_PI_O4_DF_LO = F32(F32(-8.7422776573475857731e-08) / F32(4.0))
_PI_O2_DF_HI = F32(F32(3.1415927410125732422) / F32(2.0))
_PI_O2_DF_LO = F32(F32(-8.7422776573475857731e-08) / F32(2.0))

# sin/cos u1 polynomial (float literals)
_S0 = F32(2.6083159809786593541503e-06)
_S1 = F32(-0.0001981069071916863322258)
_S2 = F32(0.00833307858556509017944336)
_S3 = F32(-0.166666597127914428710938)

# asin/acos u1 polynomial (float literals)
_A0 = F32(+0.4197454825e-1)
_A1 = F32(+0.2424046025e-1)
_A2 = F32(+0.4547423869e-1)
_A3 = F32(+0.7495029271e-1)
_A4 = F32(+0.1666677296e+0)

_NEG_ZERO_BITS = np.int32(-2147483648)  # 0x80000000 — sign bit only

_NJD = dict(cache=True, fastmath=False, error_model="numpy", nogil=True)


# ---------------------------------------------------------------------------
# df.h helpers — ENABLE_FMA_SP branch, scalar transcription
# ---------------------------------------------------------------------------

@njit(**_NJD)
def _dfnormalize(t):  # dfnormalize_vf2_vf2
    s = t[0] + t[1]
    return (s, (t[0] - s) + t[1])


@njit(**_NJD)
def _dfadd_f_f(x, y):  # dfadd_vf2_vf_vf
    s = x + y
    return (s, (x - s) + y)


@njit(**_NJD)
def _dfadd2_f_f(x, y):  # dfadd2_vf2_vf_vf
    s = x + y
    v = s - x
    return (s, (x - (s - v)) + (y - v))


@njit(**_NJD)
def _dfadd_df_f(x, y):  # dfadd_vf2_vf2_vf
    s = x[0] + y
    return (s, ((x[0] - s) + y) + x[1])


@njit(**_NJD)
def _dfadd_f_df(x, y):  # dfadd_vf2_vf_vf2
    s = x + y[0]
    return (s, ((x - s) + y[0]) + y[1])


@njit(**_NJD)
def _dfadd2_df_f(x, y):  # dfadd2_vf2_vf2_vf
    s = x[0] + y
    v = s - x[0]
    t = (x[0] - (s - v)) + (y - v)
    return (s, t + x[1])


@njit(**_NJD)
def _dfadd2_f_df(x, y):  # dfadd2_vf2_vf_vf2
    s = x + y[0]
    v = s - x
    t = (x - (s - v)) + (y[0] - v)
    return (s, t + y[1])


@njit(**_NJD)
def _dfadd1_df_df(x, y):  # dfadd_vf2_vf2_vf2  (|x| >= |y|)
    s = x[0] + y[0]
    return (s, ((x[0] - s) + y[0]) + x[1] + y[1])


@njit(**_NJD)
def _dfadd2_df_df(x, y):  # dfadd2_vf2_vf2_vf2
    s = x[0] + y[0]
    v = s - x[0]
    t = (x[0] - (s - v)) + (y[0] - v)
    return (s, t + (x[1] + y[1]))


@njit(**_NJD)
def _dfsub_df_df(x, y):  # dfsub_vf2_vf2_vf2
    s = x[0] - y[0]
    t = (x[0] - s) - y[0]
    t = t + x[1]
    return (s, t - y[1])


@njit(**_NJD)
def _dfsub_df_f(x, y):  # dfsub_vf2_vf2_vf
    s = x[0] - y
    return (s, ((x[0] - s) - y) + x[1])


@njit(**_NJD)
def _dfmul_f_f(x, y):  # dfmul_vf2_vf_vf
    s = x * y
    # vfmapn(x, y, s) = x*y - s   (NOT s - x*y)
    return (s, fmaf32(x, y, -s))


@njit(**_NJD)
def _dfsqu_df(x):  # dfsqu_vf2_vf2
    s = x[0] * x[0]
    # t = vfma(x.x+x.x, x.y, vfmapn(x.x, x.x, s));  vfmapn(a,b,c) = a*b - c
    t = fmaf32(x[0] + x[0], x[1], fmaf32(x[0], x[0], -s))
    return (s, t)


@njit(**_NJD)
def _dfmul_df_df(x, y):  # dfmul_vf2_vf2_vf2
    s = x[0] * y[0]
    # innermost: vfmapn(x.x, y.x, s) = x.x*y.x - s
    t = fmaf32(x[0], y[1], fmaf32(x[1], y[0], fmaf32(x[0], y[0], -s)))
    return (s, t)


@njit(**_NJD)
def _dfmul_vf(t, x):  # dfmul_vf_vf2_vf2 — returns vfloat
    return fmaf32(t[0], x[0], fmaf32(t[1], x[0], t[0] * x[1]))


@njit(**_NJD)
def _dfrec_f(d):  # dfrec_vf2_vf
    s = F32(1.0) / d
    return (s, s * fmaf32(-d, s, F32(1.0)))


@njit(**_NJD)
def _dfscale(x, s):  # dfscale_vf2_vf2_vf
    return (x[0] * s, x[1] * s)


@njit(**_NJD)
def _dfsqrt_f(d):  # dfsqrt_vf2_vf (non-ENABLE_RECSQRT_SP branch)
    t = np.sqrt(d)
    r = _dfmul_df_df(_dfadd2_df_df((d, F32(0.0)), _dfmul_f_f(t, t)),
                     _dfrec_f(t))
    return _dfscale(r, F32(0.5))


# ---------------------------------------------------------------------------
# xsinf_u1 / xcosf_u1 — fast path only (|d| < TRIGRANGEMAX2f)
# ---------------------------------------------------------------------------

@njit(**_NJD)
def sleef_sinf_u1(d):
    """xsinf_u1 fast path. d: float32, |d| < 125."""
    u = np.rint(d * _M_1_PI_F)
    q = np.int32(u)
    v = fmaf32(u, -_PI_A2F, d)
    s = _dfadd2_f_f(v, u * -_PI_B2F)
    s = _dfadd_df_f(s, u * -_PI_C2F)
    # |d| < TRIGRANGEMAX2f: rempif branch not taken
    t = s
    s2 = _dfsqu_df(s)

    p = fmaf32(_S0, s2[0], _S1)
    p = fmaf32(p, s2[0], _S2)
    x = _dfadd_f_df(F32(1.0), _dfmul_df_df(_dfadd_f_f(_S3, p * s2[0]), s2))
    r = _dfmul_vf(t, x)

    if (q & 1) == 1:
        r = -r
    if f2i(d) == _NEG_ZERO_BITS:  # -0.0 passthrough
        return d
    return r


@njit(**_NJD)
def sleef_cosf_u1(d):
    """xcosf_u1 fast path. d: float32, |d| < 125."""
    dq = fmaf32(np.rint(fmaf32(d, _M_1_PI_F, F32(-0.5))), F32(2.0), F32(1.0))
    q = np.int32(dq)
    s = _dfadd2_f_f(d, dq * _PI_A2F_H)
    s = _dfadd2_df_f(s, dq * _PI_B2F_H)
    s = _dfadd2_df_f(s, dq * _PI_C2F_H)
    # |d| < TRIGRANGEMAX2f: rempif branch not taken
    t = s
    s2 = _dfsqu_df(s)

    p = fmaf32(_S0, s2[0], _S1)
    p = fmaf32(p, s2[0], _S2)
    x = _dfadd_f_df(F32(1.0), _dfmul_df_df(_dfadd_f_f(_S3, p * s2[0]), s2))
    r = _dfmul_vf(t, x)

    if (q & 2) == 0:
        r = -r
    return r


# ---------------------------------------------------------------------------
# xasinf_u1 / xacosf_u1
# ---------------------------------------------------------------------------

@njit(**_NJD)
def _asinf_u1_core(d):  # d >= 0
    small = d < F32(0.5)
    if small:
        x2 = d * d
        x0 = d
        x1 = F32(0.0)
    else:
        x2 = (F32(1.0) - d) * F32(0.5)
        xs = _dfsqrt_f(x2)
        x0 = xs[0]
        x1 = xs[1]
        if d == F32(1.0):
            x0 = F32(0.0)
            x1 = F32(0.0)

    u = _A0
    u = fmaf32(u, x2, _A1)
    u = fmaf32(u, x2, _A2)
    u = fmaf32(u, x2, _A3)
    u = fmaf32(u, x2, _A4)
    u = u * (x2 * x0)

    if small:
        return u + x0
    # y = ((π/4 df) - x) - u ; r = (y.x + y.y) * 2   (x keeps BOTH parts)
    y = _dfsub_df_f(
        _dfsub_df_df((_PI_O4_DF_HI, _PI_O4_DF_LO), (x0, x1)), u)
    return (y[0] + y[1]) * F32(2.0)


@njit(**_NJD)
def sleef_asinf_u1(d):
    """xasinf_u1. d: float32, |d| <= 1."""
    r = _asinf_u1_core(np.abs(d))
    return np.copysign(r, d)  # vmulsign_vf_vf_vf(r, d)


@njit(**_NJD)
def sleef_acosf_u1(d):
    """xacosf_u1. d: float32, |d| <= 1."""
    small = np.abs(d) < F32(0.5)
    ad = np.abs(d)
    if small:
        x2 = d * d
        x0 = ad
        x1 = F32(0.0)
    else:
        x2 = (F32(1.0) - ad) * F32(0.5)
        if ad == F32(1.0):
            x0 = F32(0.0)
            x1 = F32(0.0)
        else:
            xs = _dfsqrt_f(x2)
            x0 = xs[0]
            x1 = xs[1]

    u = _A0
    u = fmaf32(u, x2, _A1)
    u = fmaf32(u, x2, _A2)
    u = fmaf32(u, x2, _A3)
    u = fmaf32(u, x2, _A4)
    u = u * (x2 * x0)

    if small:
        # y = (π/2 df) - (mulsign(x0,d) + mulsign(u,d))
        y = _dfsub_df_f(
            _dfsub_df_df((_PI_O2_DF_HI, _PI_O2_DF_LO),
                         (np.copysign(x0, d), F32(0.0))),
            np.copysign(u, d))
    else:
        # x = (x0,x1) + u ; y = dfscale(x, 2)
        y = _dfadd1_df_df((x0, x1), (u, F32(0.0)))
        y = _dfscale(y, F32(2.0))
        if d < F32(0.0):
            y = _dfsub_df_df((_PI_DF_HI, _PI_DF_LO), y)
    return y[0] + y[1]


# ---------------------------------------------------------------------------
# vector wrappers (host side) — range-checked, never silently degrading
# ---------------------------------------------------------------------------

def sleef_sin_v1(x):
    """float32 array sin (xsinf_u1). Raises outside the ported fast path."""
    import numpy as _np
    a = _np.ascontiguousarray(x, dtype=_np.float32)
    if a.size and (float(_np.abs(a).max()) >= float(_TRIGRANGEMAX2F)):
        raise ValueError(
            "sleef_sin_v1: |d| >= TRIGRANGEMAX2f (125.0) — SLEEF rempif "
            "slow path is not ported; refusing rather than degrading")
    return _sleef_sin_loop(a)


def sleef_cos_v1(x):
    """float32 array cos (xcosf_u1). Raises outside the ported fast path."""
    import numpy as _np
    a = _np.ascontiguousarray(x, dtype=_np.float32)
    if a.size and (float(_np.abs(a).max()) >= float(_TRIGRANGEMAX2F)):
        raise ValueError(
            "sleef_cos_v1: |d| >= TRIGRANGEMAX2f (125.0) — SLEEF rempif "
            "slow path is not ported; refusing rather than degrading")
    return _sleef_cos_loop(a)


def sleef_asin_v1(x):
    import numpy as _np
    a = _np.ascontiguousarray(x, dtype=_np.float32)
    if a.size and (float(_np.abs(a).max()) > 1.0):
        raise ValueError("sleef_asin_v1: |d| > 1 — domain error")
    return _sleef_asin_loop(a)


def sleef_acos_v1(x):
    import numpy as _np
    a = _np.ascontiguousarray(x, dtype=_np.float32)
    if a.size and (float(_np.abs(a).max()) > 1.0):
        raise ValueError("sleef_acos_v1: |d| > 1 — domain error")
    return _sleef_acos_loop(a)


@njit(**_NJD)
def _sleef_sin_loop(a):
    out = np.empty(a.shape, dtype=np.float32)
    for i in range(a.size):
        out.flat[i] = sleef_sinf_u1(a.flat[i])
    return out


@njit(**_NJD)
def _sleef_cos_loop(a):
    out = np.empty(a.shape, dtype=np.float32)
    for i in range(a.size):
        out.flat[i] = sleef_cosf_u1(a.flat[i])
    return out


@njit(**_NJD)
def _sleef_asin_loop(a):
    out = np.empty(a.shape, dtype=np.float32)
    for i in range(a.size):
        out.flat[i] = sleef_asinf_u1(a.flat[i])
    return out


@njit(**_NJD)
def _sleef_acos_loop(a):
    out = np.empty(a.shape, dtype=np.float32)
    for i in range(a.size):
        out.flat[i] = sleef_acosf_u1(a.flat[i])
    return out
