# SPDX-License-Identifier: GPL-3.0-only
"""Torch-exact float32 primitives beyond the T09 set (ultrafast T11).

Adds the two f32 primitives the daylen / clearnessindex met chains need
that math_compat.py (T09) and sleef_trig.py do not carry:

======================  ====================================================
torch f32 primitive     resolution
======================  ====================================================
tan                     SLEEF xtanf_u1 (sleefsimdsp.c:1641) — torch binds
                        Sleef_tanf8_u10; rename.h: `#define xtanf_u1
                        Sleef_tanf_u10`. |d| < TRIGRANGEMAX2f fast path.
pow(Scalar, Tensor)     libm powf — pow_tensor_tensor's SCALAR remainder
                        is std::pow(float, float) (PowKernel.cpp v2.2
                        cpu_kernel_vec); 0-dim n=1 results come from that
                        remainder. probe p7b: torch `0.935 ** f32_0d` ==
                        libm powf on 0/40000 (probe p7 also ruled out
                        fl32(math.pow(f64, f64)) at 523/4000 and
                        fl32(pow(f32,f32) via f64) at 24/4000).
======================  ====================================================

Consumers (met_recompute.py):
- daylen: SOC = tan(RAD*DEC) * tan(RAD*XLAT)  (f32 0-dims)
- clearnessindex_2013b: Tar = 0.935 ** m      (f32 0-dim, day route)
- Perez_v3 intClearness==0: m_c exponent base pow (Tensor, Tensor)

SLEEF is Boost Software License 1.0 (Naoki Shibata and contributors).
"""
from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import extending, njit, types

from .math_compat import f2i, fmaf32
from .sleef_trig import (
    _dfadd2_f_f,
    _dfadd_df_f,
    _dfadd_f_df,
    _dfadd_f_f,
    _dfmul_df_df,
    _dfsqu_df,
    _NEG_ZERO_BITS,
    _NJD,
    _TRIGRANGEMAX2F,
)

F32 = np.float32


# --------------------------------------------------------------------------
# libm powf binding (external function; resolves from libm at JIT load,
# the same symbol torch's scalar remainder calls)
# --------------------------------------------------------------------------
@extending.intrinsic(typing=True)
def _powf32_intrinsic(typingctx, x, y):
    """C libm powf(float, float) — torch pow(Scalar, Tensor) 0-dim bits."""
    if not (x == types.float32 and y == types.float32):
        return None
    sig = types.float32(x, y)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.FloatType(), [ir.FloatType()] * 2)
        fn = builder.module.globals.get("powf")
        if fn is None:
            fn = ir.Function(builder.module, fnty, name="powf")
        return builder.call(fn, args)

    return sig, codegen


@njit(**_NJD)
def powf32(x, y):
    """C libm powf(float, float) — torch pow(Scalar, Tensor) 0-dim bits."""
    return _powf32_intrinsic(x, y)


# --------------------------------------------------------------------------
# dd reciprocal of a dd pair (dfrec_vf2_vf2, ENABLE_FMA_SP, df.h:224)
# --------------------------------------------------------------------------
@njit(**_NJD)
def _dfrec_df_df(d0, d1):
    s = F32(1.0) / d0
    # s * fmanp(d.y, s, fmanp(d.x, s, 1)) ; vfmanp(a,b,c) = c - a*b
    y = fmaf32(-d1, s, fmaf32(-d0, s, F32(1.0)))
    return s, s * y


# --------------------------------------------------------------------------
# xtanf_u1 (sleefsimdsp.c:1641, non-DETERMINISTIC branch) — torch f32 tan.
# Fast path only: |d| < TRIGRANGEMAX2f (125.0); the rempif slow path is
# unported and refused (daylen arguments live in [-pi/2, pi/2]).
# --------------------------------------------------------------------------
_TAN_2_M1PI = F32(2 * 0.318309886183790671537767526745028724)  # C: 2*M_1_PI as double, cast
_TAN_PA_H = F32(3.1414794921875 * 0.5)     # -PI_A2f*0.5f (exact halving)
_TAN_PB_H = F32(0.00011315941810607910156 * 0.5)   # -PI_B2f*0.5f (exact)
_TAN_PC_H = F32(1.9841872589410058936e-09 * 0.5)   # -PI_C2f*0.5f (exact)


@njit(**_NJD)
def sleef_tanf_u10(d):
    if not (abs(d) < _TRIGRANGEMAX2F):
        raise ValueError("sleef_tanf_u10: |d| >= TRIGRANGEMAX2f (125) unported")

    u = np.rint(d * _TAN_2_M1PI)
    q = np.int32(u)
    v = fmaf32(u, -_TAN_PA_H, d)
    s0, s1 = _dfadd2_f_f(v, u * -_TAN_PB_H)
    s0, s1 = _dfadd_df_f((s0, s1), u * -_TAN_PC_H)

    o = (q & np.int32(1)) == np.int32(1)
    if o:  # xor sign mask on both parts
        s0 = -s0
        s1 = -s1

    t0, t1 = s0, s1
    s0, s1 = _dfsqu_df((s0, s1))
    # dfnormalize_vf2_vf2
    ns = s0 + s1
    s0, s1 = ns, (s0 - ns) + s1

    u2 = F32(0.00446636462584137916564941)
    u2 = fmaf32(u2, s0, F32(-8.3920182078145444393158e-05))
    u2 = fmaf32(u2, s0, F32(0.0109639242291450500488281))
    u2 = fmaf32(u2, s0, F32(0.0212360303848981857299805))
    u2 = fmaf32(u2, s0, F32(0.0540687143802642822265625))

    x0, x1 = _dfadd_f_f(F32(0.133325666189193725585938), u2 * s0)
    # dfadd_vf2_vf_vf2(1, dfmul(s, dfadd_vf2_vf_vf2(0.333…, dfmul(s, x))))
    x0, x1 = _dfadd_f_df(
        F32(1.0),
        _dfmul_df_df(_dfadd_f_df(F32(0.33333361148834228515625),
                                 _dfmul_df_df((s0, s1), (x0, x1))),
                     (s0, s1)))
    x0, x1 = _dfmul_df_df((t0, t1), (x0, x1))

    if o:
        x0, x1 = _dfrec_df_df(x0, x1)

    r = x0 + x1
    if f2i(d) == _NEG_ZERO_BITS:  # -0.0 passthrough
        return d
    return r
