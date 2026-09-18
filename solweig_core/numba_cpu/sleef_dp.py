# SPDX-License-Identifier: GPL-3.0-only
"""Torch-exact float64 math primitives for the torch-free CPU kernels
(ultrafast T11).

Bit-for-bit reproduction (uint64 views, signed zeros included) of the
float64 results ``torch`` produces on this host for the f64
transcendentals the met-recompute chains use:

======================  ====================================================
torch f64 primitive     resolution (SLEEF pin 5a1d179, AdvSIMD CONFIG=1,
                        ENABLE_FMA_DP; torch binds the *u10* symbols)
======================  ====================================================
exp                     SLEEF xexp, FMA_DP POLY10 branch (sleefsimddp.c:2130)
log                     SLEEF xlog_u1 (sleefsimddp.c:2255, non-AVX512 branch)
sin                     SLEEF xsin_u1 (sleefsimddp.c:505, |d|<15 fast path)
cos                     SLEEF xcos_u1 (sleefsimddp.c:771, |d|<15 fast path)
pow (x>0, finite e)     SLEEF xpow main path (sleefsimddp.c:2343) via
                        logk (:2208) + expk (:2307)
======================  ====================================================

Evidence chain (t11 probes):
- probe_p4_prim.py P1: torch f64 0-dim exp/sin/cos/log != libm (math
  module) on ~5% of random inputs -> torch does NOT use the std:: scalar
  remainder for 0-dim f64 elementwise; torch f64 pow differs from python
  ``**`` on 3/2000 -> SLEEF xpow, not std::pow (note PowKernel.cpp only
  special-cases exp 2/-2/3 for pow_tensor_scalar; **4 falls to Vec::pow).
- probe_p5_0d.py: torch f64 0-dim == 8-element vector lane 0 (0/64 for
  exp and sin) -> the Vectorized<double> body is the semantics for every
  lane class, exactly like the f32 case T09 established.
- pytorch v2.2 aten/src/ATen/cpu/vec/vec256/vec256_double.h binds
  Sleef_{sin,cos}d4_u10 -> xsin_u1/xcos_u1 (u1 bodies, NOT the xsin/xcos
  fast paths), Sleef_expd4_u10 -> xexp, Sleef_logd4_u10 -> xlog_u1,
  Sleef_powd4_u10 -> xpow.

Ports inline dd.h ENABLE_FMA_DP helpers (dd.h:190-235) at their exact
association; llvm.fma.f64 is the ONLY fused op (SLEEF builds with
-ffp-contract=off, fastmath=False mirrors that). Estrin macros per
estrin.h with MLA(x,y,z) = fma(x, y, z).

Range discipline (typed refusals, mirroring sleef_trig.py):
- sin/cos raise for |d| >= TRIGRANGEMAX2 (15.0): every met-chain
  argument lives in [0, pi]; the TRIGRANGEMAX/rempi branches are
  unreachable and unported.
- pow raises for x <= 0 or non-finite x/y: met chains only ever raise
  positive bases (absolute temperatures) to finite powers; xpow's
  sign/inf/NaN masks are unreachable there.
- log/exp carry the SLEEF_DBL_MIN / +-overflow fixups the C bodies have.

POW ROUTING DISCOVERY (t11 probe P6, decisive for torch-free met edits):
torch pow(Tensor, Scalar) on f32/f64 dispatches with exp as DOUBLE to
pow_tensor_scalar_optimized_kernel (PowKernel.cpp v2.2): exp 2/-2/3 exact
mults, 0.5/-0.5/-1 sqrt family, everything else cpu_kernel_vec with
vector body = Vec::pow (SLEEF xpow) and scalar remainder = std::pow
(opmath f64 for f32). For pow — unlike the unary ops — the 0-dim
(n=1) result comes from the SCALAR remainder: torch f64 0-dim `t ** y`
== python `x ** y` (libm std::pow) on 0/40000 samples across y in
{3.7, 4} and x in [0.1, 350]; torch f32 0-dim == fl32(math.pow(f64,
f64)) 0/20000. So met chains' 0-dim `**4` (Lwall, ta273_pow4, dp
surfaces, Lsky tables) reproduce with plain python `**`; f32 PLANE
`**4` (day Lup_pre, gvflup_extra) uses math_compat.torch_pow_scalar_vector
(T09 lane-routed SLEEF xpowf). sleef_powd_u10 here is the f64
vector-body twin, kept for completeness and gated against torch
vector lanes (probe p6b).

SLEEF is Boost Software License 1.0 (Naoki Shibata and contributors).
"""
from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import extending, njit, types

F64 = np.float64

# --------------------------------------------------------------------------
# intrinsics
# --------------------------------------------------------------------------


@extending.intrinsic(typing=True)
def fmaf64(typingctx, a, b, c):
    """llvm.fma.f64 — the ONLY fused multiply-add; used exactly where
    SLEEF wrote vfma/vmla/vfmapn/vfmanp."""
    if not (a == types.float64 and b == types.float64 and c == types.float64):
        return None
    sig = types.float64(a, b, c)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()] * 3)
        fn = builder.module.globals.get("llvm.fma.f64")
        if fn is None:
            fn = ir.Function(builder.module, fnty, name="llvm.fma.f64")
        return builder.call(fn, args)

    return sig, codegen


@extending.intrinsic(typing=True)
def f2i64(typingctx, x):
    """Bit reinterpret double -> int64 (for ilogb/exponent bit tricks)."""
    if x != types.float64:
        return None
    sig = types.int64(x)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.IntType(64))

    return sig, codegen


@extending.intrinsic(typing=True)
def i2f64(typingctx, i):
    """Bit reinterpret int64 -> double (for pow2i / ldexp3)."""
    if i != types.int64:
        return None
    sig = types.float64(i)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.DoubleType())

    return sig, codegen


@extending.intrinsic(typing=True)
def rintd(typingctx, x):
    # vrndnq_f64 under the default round-to-nearest-even mode
    if x != types.float64:
        return None
    sig = types.float64(x)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        fn = builder.module.globals.get("llvm.rint.f64")
        if fn is None:
            fn = ir.Function(builder.module, fnty, name="llvm.rint.f64")
        return builder.call(fn, args)

    return sig, codegen


# --------------------------------------------------------------------------
# constants (misc.h; exact C double literals)
# --------------------------------------------------------------------------
L2U = F64(0.69314718055966295651160180568695068359375)
L2L = F64(0.28235290563031577122588448175013436025525412068e-12)
R_LN2 = F64(1.442695040888963407359924681001892137426645954152985934135449406931)
LOG_DBL_MAX = F64(709.782712893384)  # 0x1.62e42fefa39efp+9
TRIGRANGEMAX2 = F64(15.0)
PI_A2 = F64(3.141592653589793116)
PI_B2 = F64(1.2246467991473532072e-16)
PI_A2_H = F64(3.141592653589793116 * 0.5)  # C constant-folds; exact halving
PI_B2_H = F64(1.2246467991473532072e-16 * 0.5)
M_1_PI = F64(0.318309886183790671537767526745028724)
SLEEF_DBL_MIN = F64(2.2250738585072014e-308)
TWO64 = F64(18446744073709551616.0)
INF64 = F64(np.inf)
NAN64 = F64(np.nan)

LN2_HI = F64(0.693147180559945286226764)
LN2_LO = F64(2.319046813846299558417771e-17)
LOGK_C0 = F64(0.666666666666666629659233)
LOGK_C1 = F64(3.80554962542412056336616e-17)


# --------------------------------------------------------------------------
# exponent helpers (commonfuncs.h:300/326/349/353)
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def _vilogb2k(d):
    # (bits >> 52) & 0x7ff - 0x3ff   (int64 form of the AdvSIMD low lane)
    return np.int32(((f2i64(d) >> np.int64(52)) & np.int64(0x7FF)) - np.int64(0x3FF))


@njit(cache=True, fastmath=False, error_model="numpy")
def _pow2i(q):
    # i2f64((q + 1023) << 52)   (AdvSIMD 32-bit (q+0x3ff)<<20 form)
    return i2f64((np.int64(q) + np.int64(1023)) << np.int64(52))


@njit(cache=True, fastmath=False, error_model="numpy")
def _ldexp2(d, q):
    # d * 2^(q>>1) * 2^(q - (q>>1))
    h = q >> np.int32(1)
    return d * _pow2i(h) * _pow2i(q - h)


@njit(cache=True, fastmath=False, error_model="numpy")
def _ldexp3(d, q):
    # bits + (q << 52)
    return i2f64(f2i64(d) + (np.int64(q) << np.int64(52)))


# --------------------------------------------------------------------------
# dd.h helpers, ENABLE_FMA_DP variants (dd.h:190-235), exact association.
# vdouble2 = (x, y) python tuple.
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def _ddnormalize(t0, t1):
    s = t0 + t1
    return s, (t0 - s) + t1


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddscale(d0, d1, s):
    return d0 * s, d1 * s


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddadd_xx(x, y):
    # ddadd_vd2_vd_vd
    s = x + y
    return s, (x - s) + y


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddadd2_xx(x, y):
    # ddadd2_vd2_vd_vd
    s = x + y
    v = s - x
    return s, (x - (s - v)) + (y - v)


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddadd_xy(x0, x1, y):
    # ddadd_vd2_vd2_vd : vadd_vd_3vd((x-s), y, x.y)
    s = x0 + y
    return s, ((x0 - s) + y) + x1


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddadd2_xy(x0, x1, y):
    # ddadd2_vd2_vd2_vd
    s = x0 + y
    v = s - x0
    w = (x0 - (s - v)) + (y - v)
    return s, w + x1


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddadd_yx(x, y0, y1):
    # ddadd_vd2_vd_vd2 : vadd_vd_3vd((x-s), y.x, y.y)
    s = x + y0
    return s, ((x - s) + y0) + y1


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddadd_xxyy(x0, x1, y0, y1):
    # ddadd_vd2_vd2_vd2 : vadd_vd_4vd((x-s), y.x, x.y, y.y)
    s = x0 + y0
    return s, (((x0 - s) + y0) + x1) + y1


@njit(cache=True, fastmath=False, error_model="numpy")
def _dddiv(n0, n1, d0, d1):
    # dddiv_vd2_vd2_vd2 (FMA_DP)
    t = F64(1.0) / d0
    s = n0 * t
    u = fmaf64(t, n0, -s)
    v = fmaf64(-d1, t, fmaf64(-d0, t, F64(1.0)))
    return s, fmaf64(s, v, fmaf64(n1, t, u))


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddmul_xy(x0, x1, y):
    # ddmul_vd2_vd2_vd
    s = x0 * y
    return s, fmaf64(x1, y, fmaf64(x0, y, -s))


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddmul_xxyy(x0, x1, y0, y1):
    # ddmul_vd2_vd2_vd2
    s = x0 * y0
    return s, fmaf64(x0, y1, fmaf64(x1, y0, fmaf64(x0, y0, -s)))


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddmul_v(x0, x1, y0, y1):
    # ddmul_vd_vd2_vd2 (single result)
    return fmaf64(x0, y0, fmaf64(x1, y0, x0 * y1))


@njit(cache=True, fastmath=False, error_model="numpy")
def _ddsqu(x0, x1):
    # ddsqu_vd2_vd2 (FMA_DP)
    s = x0 * x0
    return s, fmaf64(x0 + x0, x1, fmaf64(x0, x0, -s))


# --------------------------------------------------------------------------
# xexp (sleefsimddp.c:2130, ENABLE_FMA_DP branch) — torch f64 exp
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_expd(d):
    u = rintd(d * R_LN2)
    q = np.int32(u)  # dql already integral -> conversion exact

    s = fmaf64(u, -L2U, d)
    s = fmaf64(u, -L2L, s)

    s2 = s * s
    s4 = s2 * s2
    s8 = s4 * s4
    # POLY10 = MLA(s8, POLY2(s, c9, c8), POLY8(s, s2, s4, c7..c0))
    # POLY8  = MLA(s4, POLY4(s, s2, c7..c4), POLY4(s, s2, c3..c0))
    # POLY4(a, a2, c3..c0) = MLA(a2, MLA(a, c3, c2), MLA(a, c1, c0))
    p4hi = fmaf64(s2, fmaf64(s, F64(0.2755762628169491192e-6), F64(0.2755723402025388239e-5)),
                  fmaf64(s, F64(0.2480158687479686264e-4), F64(0.1984126989855865850e-3)))
    p4lo = fmaf64(s2, fmaf64(s, F64(0.1388888888914497797e-2), F64(0.8333333333314938210e-2)),
                  fmaf64(s, F64(0.4166666666666602598e-1), F64(0.1666666666666669072e0)))
    u2 = fmaf64(s8,
                fmaf64(s, F64(0.2081276378237164457e-8), F64(0.2511210703042288022e-7)),
                fmaf64(s4, p4hi, p4lo))
    u2 = fmaf64(u2, s, F64(0.5))
    u2 = fmaf64(u2, s, F64(1.0))
    u2 = fmaf64(u2, s, F64(1.0))

    u2 = _ldexp2(u2, q)

    if d > LOG_DBL_MAX:
        u2 = INF64
    if d < F64(-1000.0):
        u2 = F64(0.0)
    return u2


# --------------------------------------------------------------------------
# xlog_u1 (sleefsimddp.c:2255, non-AVX512 branch) — torch f64 log
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_logd_u1(d0):
    if d0 == INF64:
        return INF64
    if d0 < F64(0.0) or d0 != d0:
        return NAN64
    if d0 == F64(0.0):
        return -INF64

    d = d0
    o = d < SLEEF_DBL_MIN
    if o:
        d = d * TWO64

    e = _vilogb2k(d * F64(1.0 / 0.75))
    m = _ldexp3(d, -e)
    if o:
        e = np.int32(e - 64)

    n0, n1 = _ddadd2_xx(F64(-1.0), m)
    dn0, dn1 = _ddadd2_xx(F64(1.0), m)
    x0, x1 = _dddiv(n0, n1, dn0, dn1)
    x2 = x0 * x0  # plain double product (NOT ddsqu here)

    x4 = x2 * x2
    x8 = x4 * x4
    # POLY7(x2, x4, x8, c6..c0) = MLA(x8, POLY3(x2, x4, c6, c5, c4),
    #                                 POLY4(x2, x4, c3, c2, c1, c0))
    p3 = fmaf64(x4, F64(0.1532076988502701353e0),
                fmaf64(x2, F64(0.1525629051003428716e0), F64(0.1818605932937785996e0)))
    p4 = fmaf64(x4, fmaf64(x2, F64(0.2222214519839380009e0), F64(0.2857142932794299317e0)),
                fmaf64(x2, F64(0.3999999999635251990e0), F64(0.6666666666667333541e0)))
    t = fmaf64(x8, p3, p4)

    s0, s1 = _ddmul_xy(LN2_HI, LN2_LO, F64(np.float64(e)))
    sc0, sc1 = _ddscale(x0, x1, F64(2.0))
    s0, s1 = _ddadd_xxyy(s0, s1, sc0, sc1)
    s0, s1 = _ddadd_xy(s0, s1, (x2 * x0) * t)

    return s0 + s1


# --------------------------------------------------------------------------
# xsin_u1 / xcos_u1 (sleefsimddp.c:505 / :771, TRIGRANGEMAX2 fast paths)
# --------------------------------------------------------------------------
_SIN_C5 = F64(2.72052416138529567917983e-15)
_SIN_C4 = F64(-7.6429259411395447190023e-13)
_SIN_C3 = F64(1.60589370117277896211623e-10)
_SIN_C2 = F64(-2.5052106814843123359368e-08)
_SIN_C1 = F64(2.75573192104428224777379e-06)
_SIN_C0 = F64(-0.000198412698412046454654947)
_SIN_P1 = F64(0.00833333333333318056201922)
_SIN_Q1 = F64(-0.166666666666666657414808)


@njit(cache=True, fastmath=False, error_model="numpy")
def _sin_tail(t0, t1, sq0, sq1):
    # shared xsin_u1/xcos_u1 tail: s = ddsqu(t); s2 = s.x^2; s4 = s2^2
    s2 = sq0 * sq0
    s4 = s2 * s2
    # POLY6 = MLA(s4, POLY2(sq0, c5, c4), POLY4(sq0, s2, c3..c0))
    p4 = fmaf64(s2, fmaf64(sq0, _SIN_C3, _SIN_C2),
                fmaf64(sq0, _SIN_C1, _SIN_C0))
    u2 = fmaf64(s4, fmaf64(sq0, _SIN_C5, _SIN_C4), p4)
    u2 = fmaf64(u2, sq0, _SIN_P1)

    m0, m1 = _ddadd_xx(_SIN_Q1, u2 * sq0)
    mm0, mm1 = _ddmul_xxyy(m0, m1, sq0, sq1)
    x0, x1 = _ddadd_yx(F64(1.0), mm0, mm1)
    return _ddmul_v(t0, t1, x0, x1)


@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_sind_u10(d):
    if not (abs(d) < TRIGRANGEMAX2):
        raise ValueError("sleef_sind_u10: |d| >= TRIGRANGEMAX2 (15) unported")

    dql = rintd(d * M_1_PI)
    ql = np.int32(dql)
    u = fmaf64(dql, -PI_A2, d)
    s0, s1 = _ddadd_xx(u, dql * -PI_B2)

    sq0, sq1 = _ddsqu(s0, s1)
    r = _sin_tail(s0, s1, sq0, sq1)

    if (ql & np.int32(1)) == np.int32(1):
        r = -r
    if d == F64(0.0):
        return d
    return r


@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_cosd_u10(d):
    if not (abs(d) < TRIGRANGEMAX2):
        raise ValueError("sleef_cosd_u10: |d| >= TRIGRANGEMAX2 (15) unported")

    dql = rintd(fmaf64(d, M_1_PI, F64(-0.5)))
    dql = fmaf64(F64(2.0), dql, F64(1.0))
    ql = np.int32(dql)
    s0, s1 = _ddadd2_xx(d, dql * -PI_A2_H)
    s0, s1 = _ddadd_xy(s0, s1, dql * -PI_B2_H)

    sq0, sq1 = _ddsqu(s0, s1)
    r = _sin_tail(s0, s1, sq0, sq1)

    if (ql & np.int32(2)) == np.int32(0):
        r = -r
    return r


# --------------------------------------------------------------------------
# logk (:2208) / expk (:2307) / xpow (:2343) — torch f64 pow.
# Guarded: x > 0, finite y (all xpow sign/inf/NaN masks unreachable).
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def _logk(d0):
    d = d0
    o = d < SLEEF_DBL_MIN
    if o:
        d = d * TWO64

    e = _vilogb2k(d * F64(1.0 / 0.75))
    m = _ldexp3(d, -e)
    if o:
        e = np.int32(e - 64)

    n0, n1 = _ddadd2_xx(F64(-1.0), m)
    dn0, dn1 = _ddadd2_xx(F64(1.0), m)
    x0, x1 = _dddiv(n0, n1, dn0, dn1)
    xq0, xq1 = _ddsqu(x0, x1)  # dd square here (unlike xlog_u1)

    x4 = xq0 * xq0
    x8 = x4 * x4
    x16 = x8 * x8
    # POLY9(xq0, x4, x8, x16, c8..c0) = MLA(x16, c8, POLY8)
    # POLY8 = MLA(x8, POLY4(xq0, x4, c7..c4), POLY4(xq0, x4, c3..c0))
    p4hi = fmaf64(x4, fmaf64(xq0, F64(0.103239680901072952701192), F64(0.117754809412463995466069)),
                  fmaf64(xq0, F64(0.13332981086846273921509), F64(0.153846227114512262845736)))
    p4lo = fmaf64(x4, fmaf64(xq0, F64(0.181818180850050775676507), F64(0.222222222230083560345903)),
                  fmaf64(xq0, F64(0.285714285714249172087875), F64(0.400000000000000077715612)))
    t = fmaf64(x16, F64(0.116255524079935043668677), fmaf64(x8, p4hi, p4lo))

    s0, s1 = _ddmul_xy(LN2_HI, LN2_LO, F64(np.float64(e)))
    sc0, sc1 = _ddscale(x0, x1, F64(2.0))
    s0, s1 = _ddadd_xxyy(s0, s1, sc0, sc1)
    xa0, xa1 = _ddmul_xxyy(xq0, xq1, x0, x1)
    s0, s1 = _ddadd_xxyy(s0, s1, *_ddmul_xxyy(xa0, xa1, LOGK_C0, LOGK_C1))
    xb0, xb1 = _ddmul_xxyy(xq0, xq1, xa0, xa1)
    s0, s1 = _ddadd_xxyy(s0, s1, *_ddmul_xy(xb0, xb1, t))
    return s0, s1


@njit(cache=True, fastmath=False, error_model="numpy")
def _expk(d0, d1):
    u = (d0 + d1) * R_LN2
    dq = rintd(u)
    q = np.int32(dq)

    s0, s1 = _ddadd2_xy(d0, d1, dq * -L2U)
    s0, s1 = _ddadd2_xy(s0, s1, dq * -L2L)
    s0, s1 = _ddnormalize(s0, s1)

    s2 = s0 * s0
    s4 = s2 * s2
    s8 = s4 * s4
    # POLY10 (expk coeff set)
    p4hi = fmaf64(s2, fmaf64(s0, F64(2.75572496725023574143864e-06), F64(2.48014973989819794114153e-05)),
                  fmaf64(s0, F64(0.000198412698809069797676111), F64(0.0013888888939977128960529)))
    p4lo = fmaf64(s2, fmaf64(s0, F64(0.00833333333332371417601081), F64(0.0416666666665409524128449)),
                  fmaf64(s0, F64(0.166666666666666740681535), F64(0.500000000000000999200722)))
    u2 = fmaf64(s8, fmaf64(s0, F64(2.51069683420950419527139e-08), F64(2.76286166770270649116855e-07)),
                fmaf64(s4, p4hi, p4lo))

    t0, t1 = _ddadd_yx(F64(1.0), s0, s1)
    t0, t1 = _ddadd_xxyy(t0, t1, *_ddmul_xy(*_ddsqu(s0, s1), u2))

    u3 = t0 + t1
    u3 = _ldexp2(u3, q)

    if d0 < F64(-1000.0):
        u3 = F64(0.0)
    return u3


@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_powd_u10(x, y):
    if not (x > F64(0.0)) or not (x < INF64) or not (abs(y) < INF64):
        raise ValueError("sleef_powd_u10: requires 0 < x < inf, |y| < inf")
    l0, l1 = _logk(x)
    d0, d1 = _ddmul_xy(l0, l1, y)
    r = _expk(d0, d1)
    if d0 > LOG_DBL_MAX:
        r = INF64
    return r
