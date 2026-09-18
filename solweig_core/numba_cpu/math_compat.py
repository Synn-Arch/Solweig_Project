# SPDX-License-Identifier: GPL-3.0-only
"""Torch-exact float32 math primitives for the torch-free CPU kernels
(DESIGN.ko.md 5.2/12.1, ultrafast T09).

This module reproduces, bit-for-bit (uint32 views, signed zeros and NaN
payloads included), the float32 results ``torch`` 2.14.0 produces on this
host for every primitive the UTCI expression uses:

======================  ====================================================
primitive               resolution
======================  ====================================================
add/sub/mul/div f32     IEEE elementwise (path-independent) — T01
exp                     SLEEF xexpf scalar port (all lanes incl. scalar
                        remainders) — T09 P5/P5c
log                     SLEEF xlogf_u1 scalar port (torch's SLEEF pin
                        5a1d17 maps the u10 symbol to the u1 body) —
                        T09 P5c/P5d
pow, scalar exp -2..3   torch exact forms: 1, x, x*x, (x*x)*x, 1/x,
                        1/(x*x) — T09 P5f
pow, scalar exp 4..6    SLEEF xpowf port on vector-body lanes; the
                        per-chunk scalar-tail lanes use torch's opmath
                        scalar pow = pow(f64(x), f64(e)) single-rounded
                        to f32, under the empirically locked torch
                        elementwise chunk layout — T09 P5e/P5f/P9
======================  ====================================================

All primitives are CLEAN: there are no torch fallbacks, so
``HYBRID_STATUS["fallback_primitives"]`` is empty and the module never
imports torch (a subprocess test in tests/ultrafast/test_utci.py asserts
this).

Ports follow SLEEF 3.8 ``src/libm/sleefsimdsp.c`` AdvSIMD CONFIG=1 +
ENABLE_FMA_SP semantics with df.h helpers inlined at their exact
association (SLEEF is Boost Software License 1.0, Naoki Shibata and
contributors).

Torch chunk layout (P5e/P5f, 15/15 fresh sizes validated by exact
position-set equality): for an n-lane elementwise op under ``T`` OpenMP
threads and grain ``g`` (torch 2.14.0 CPU: 32768), torch runs
``threads_eff = min(T, ceil(n/g))`` contiguous chunks of
``ceil(n/threads_eff)`` lanes; inside each chunk lanes ``[0, len - len%8)``
take the 4-lane SLEEF vector body and the trailing ``len%8`` lanes take the
scalar remainder (torch opmath: f64 pow, single-rounded to f32 — P9).
``n <= grain`` runs one serial chunk. This is
observable ONLY through pow exps >= 4 (SLEEF vs the f64 scalar pow differ
on ~2% of inputs); add/mul/div and exp/log are bit-identical on both lane
classes (exact IEEE / path-independent SLEEF).
"""
from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import extending, njit, types

F32 = np.float32

# --------------------------------------------------------------------------
# torch elementwise chunk layout (module constants — see module docstring)
# --------------------------------------------------------------------------
TORCH_PAR_GRAIN = 32768
DEFAULT_TORCH_THREADS = 8  # t08/capture/manifest.json: torch_threads


@extending.intrinsic(typing=True)
def fmaf32(typingctx, a, b, c):
    """llvm.fma.f32 — the ONLY fused multiply-add in the ports, exactly
    where SLEEF wrote vfma/vmla (new FMA contraction anywhere else is a
    T09 RED witness)."""
    if not (a == types.float32 and b == types.float32 and c == types.float32):
        return None
    sig = types.float32(a, b, c)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.FloatType(), [ir.FloatType()] * 3)
        fn = builder.module.globals.get("llvm.fma.f32")
        if fn is None:
            fn = ir.Function(builder.module, fnty, name="llvm.fma.f32")
        return builder.call(fn, args)

    return sig, codegen


@extending.intrinsic(typing=True)
def f2i(typingctx, x):
    if x != types.float32:
        return None
    sig = types.int32(x)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.IntType(32))

    return sig, codegen


@extending.intrinsic(typing=True)
def i2f(typingctx, i):
    if i != types.int32:
        return None
    sig = types.float32(i)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.FloatType())

    return sig, codegen


@extending.intrinsic(typing=True)
def rintf32(typingctx, x):
    # vrndnq_f32 under default round-to-nearest-even
    if x != types.float32:
        return None
    sig = types.float32(x)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
        fn = builder.module.globals.get("llvm.rint.f32")
        if fn is None:
            fn = ir.Function(builder.module, fnty, name="llvm.rint.f32")
        return builder.call(fn, args)

    return sig, codegen


INF = F32(np.inf)
NAN_Q = F32(np.nan)
FLT_MIN = F32(1.1754943508222875079687365372222456772406151193726e-38)
TWO64 = F32(18446744073709551616.0)
LN2HI = F32(0.69314718246459960938)      # 0x3f317218
LN2LO = F32(-1.904654323148236017e-09)   # 0xb102e308
R_LN2F = F32(1.4426950216293335)
L2UF = F32(0.693145751953125)
L2LF = F32(1.428606765330187045e-06)
BIG24 = F32(16777216.0)  # 1 << 24


# --------------------------------------------------------------------------
# SLEEF xexpf (sleefsimdsp.c:1314) — torch.exp on every lane class
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_expf(d):
    q = np.int32(rintf32(d * R_LN2F))
    s = fmaf32(np.float32(q), -L2UF, d)
    s = fmaf32(np.float32(q), -L2LF, s)

    u = F32(0.000198527617612853646278381)
    u = fmaf32(u, s, F32(0.00139304355252534151077271))
    u = fmaf32(u, s, F32(0.00833336077630519866943359))
    u = fmaf32(u, s, F32(0.0416664853692054748535156))
    u = fmaf32(u, s, F32(0.166666671633720397949219))
    u = fmaf32(u, s, F32(0.5))

    u = F32(1.0) + fmaf32(s * s, u, s)

    # vldexp2_vf_vf_vi2(u, q): u * 2^(q>>1) * 2^(q-(q>>1))
    h = np.int32(q >> 1)
    u = u * i2f(np.int32((h + 127) << 23)) * i2f(np.int32((q - h + 127) << 23))

    if d < F32(-104.0):
        u = F32(0.0)
    if F32(100.0) < d:
        u = INF
    return u


# --------------------------------------------------------------------------
# SLEEF xlogf_u1 (sleefsimdsp.c:1528 == pytorch pin 5a1d17:2274) — torch.log
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_logf_u1(d0):
    d = d0
    o = d < FLT_MIN
    if o:
        d = d * TWO64

    # vilogb2k(d * 4/3); vldexp3(d, -e)
    e = np.int32(((f2i(d * F32(4.0 / 3.0)) >> 23) & 0xFF) - 127)
    m = i2f(np.int32(f2i(d) + np.int32(np.int32(-e) << 23)))
    if o:
        e = np.int32(e - 64)

    # s = dfmul_vf2_vf2_vf((LN2HI, LN2LO), e)   [df.h:214 FMA_SP]
    s_x = LN2HI * np.float32(e)
    s_y = fmaf32(LN2LO, np.float32(e), fmaf32(LN2HI, np.float32(e), -s_x))

    # n = dfadd2(-1, m); dn = dfadd2(1, m)      [df.h:116]
    n_x = m - F32(1.0)
    n_v = n_x + F32(1.0)
    n_y = (F32(-1.0) - (n_x - n_v)) + (m - n_v)

    dn_x = F32(1.0) + m
    dn_v = dn_x - F32(1.0)
    dn_y = (F32(1.0) - (dn_x - dn_v)) + (m - dn_v)

    # x = dfdiv(n, dn)                          [df.h:183 FMA_SP]
    t_rcp = F32(1.0) / dn_x  # vrec = vdiv(1, d.x)
    x_x = n_x * t_rcp
    u_ = fmaf32(t_rcp, n_x, -x_x)
    v_ = fmaf32(-dn_x, t_rcp, F32(1.0))
    v_ = fmaf32(-dn_y, t_rcp, v_)
    x_y = fmaf32(x_x, v_, fmaf32(n_y, t_rcp, u_))

    x2 = x_x * x_x

    t = F32(0.3027294874)
    t = fmaf32(t, x2, F32(0.3996108174))
    t = fmaf32(t, x2, F32(0.6666694880))

    # s = dfadd(s, dfscale(x, 2))               [df.h:151, :107]
    sc_x = x_x * F32(2.0)
    sc_y = x_y * F32(2.0)
    s2_x = s_x + sc_x
    s2_y = ((s_x - s2_x) + sc_x) + s_y + sc_y

    # s = dfadd(s, (x2 * x.x) * t)              [df.h:151, y.y = 0 kept]
    term = (x2 * x_x) * t
    s3_x = s2_x + term
    s3_y = ((s2_x - s3_x) + term) + s2_y + F32(0.0)

    r = s3_x + s3_y

    # fixups (select order preserved)
    if d0 == INF:
        r = INF
    if d0 < F32(0.0) or d0 != d0:
        r = NAN_Q
    if d0 == F32(0.0):
        r = -INF
    return r


# --------------------------------------------------------------------------
# SLEEF xpowf internals (logkf, vldexp, expkf) and xpowf — torch.pow on
# vector-body lanes
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def _logkf(d0):
    """SLEEF logkf (sleefsimdsp.c:1538) -> (s_x, s_y). FMA_SP df.h forms."""
    d = d0
    o = d < FLT_MIN
    if o:
        d = d * TWO64
    e = np.int32(((f2i(d * F32(4.0 / 3.0)) >> 23) & 0xFF) - 127)
    m = i2f(np.int32(f2i(d) + np.int32(np.int32(-e) << 23)))
    if o:
        e = np.int32(e - 64)

    s_x = LN2HI * np.float32(e)
    s_y = fmaf32(LN2LO, np.float32(e), fmaf32(LN2HI, np.float32(e), -s_x))

    n_x = m - F32(1.0)
    n_v = n_x + F32(1.0)
    n_y = (F32(-1.0) - (n_x - n_v)) + (m - n_v)
    dn_x = F32(1.0) + m
    dn_v = dn_x - F32(1.0)
    dn_y = (F32(1.0) - (dn_x - dn_v)) + (m - dn_v)

    t_rcp = F32(1.0) / dn_x
    x_x = n_x * t_rcp
    u_ = fmaf32(t_rcp, n_x, -x_x)
    v_ = fmaf32(-dn_x, t_rcp, F32(1.0))
    v_ = fmaf32(-dn_y, t_rcp, v_)
    x_y = fmaf32(x_x, v_, fmaf32(n_y, t_rcp, u_))

    # x2 = dfsqu(x)                             [df.h:196]
    x2_x = x_x * x_x
    x2_y = fmaf32(x_x + x_x, x_y, fmaf32(x_x, x_x, -x2_x))

    t = F32(0.240320354700088500976562)
    t = fmaf32(t, x2_x, F32(0.285112679004669189453125))
    t = fmaf32(t, x2_x, F32(0.400007992982864379882812))
    c_x = F32(0.66666662693023681640625)
    c_y = F32(3.69183861259614332084311e-09)

    # s = dfadd(s, dfscale(x, 2))               [df.h:151, :107]
    sc_x = x_x * F32(2.0)
    sc_y = x_y * F32(2.0)
    s2_x = s_x + sc_x
    s2_y = ((s_x - s2_x) + sc_x) + s_y + sc_y

    # A = dfmul(x2, x)                          [df.h:205]
    a_x = x2_x * x_x
    a_y = fmaf32(x2_x, x_y, fmaf32(x2_y, x_x, fmaf32(x2_x, x_x, -a_x)))

    # B = dfmul(x2, t)                          [df.h:214]
    b_x = x2_x * t
    b_y = fmaf32(x2_y, t, fmaf32(x2_x, t, -b_x))

    # C = dfadd2(B, c)                          [df.h:158]
    cc_x = b_x + c_x
    cc_v = cc_x - b_x
    cc_t = (b_x - (cc_x - cc_v)) + (c_x - cc_v)
    cc_y = cc_t + (b_y + c_y)

    # D = dfmul(A, C)                           [df.h:205]
    d_x = a_x * cc_x
    d_y = fmaf32(a_x, cc_y, fmaf32(a_y, cc_x, fmaf32(a_x, cc_x, -d_x)))

    # s = dfadd(s, D)                           [df.h:151]
    s3_x = s2_x + d_x
    s3_y = ((s2_x - s3_x) + d_x) + s2_y + d_y
    return s3_x, s3_y


@njit(cache=True, fastmath=False, error_model="numpy")
def _vldexp(x, q):
    """vldexp_vf_vf_vi2 (sleefsimdsp.c:316) — 4-mul split with clamp."""
    m = np.int32(q >> 31)
    m = np.int32((((m + q) >> 6) - m) << 4)
    q = np.int32(q - np.int32(m << 2))
    m = np.int32(m + 127)
    if not m > 0:
        m = np.int32(0)
    if m > 255:
        m = np.int32(255)
    u = i2f(np.int32(m << 23))
    x = (((x * u) * u) * u) * u
    u = i2f(np.int32((q + 127) << 23))
    return x * u


@njit(cache=True, fastmath=False, error_model="numpy")
def _expkf(d_x, d_y):
    """SLEEF expkf (sleefsimdsp.c:1569) -> vfloat."""
    u = (d_x + d_y) * R_LN2F
    q = np.int32(rintf32(u))

    # s = dfadd2(d, q*-L2U); dfadd2(s, q*-L2L)  [df.h:122]
    s_x = d_x + np.float32(q) * -L2UF
    v_ = s_x - d_x
    s_y = ((d_x - (s_x - v_)) + (np.float32(q) * -L2UF - v_)) + d_y
    s2_x = s_x + np.float32(q) * -L2LF
    v_ = s2_x - s_x
    s2_y = ((s_x - (s2_x - v_)) + (np.float32(q) * -L2LF - v_)) + s_y

    # dfnormalize                                [df.h:102]
    s3_x = s2_x + s2_y
    s3_y = (s2_x - s3_x) + s2_y

    u = F32(0.00136324646882712841033936)
    u = fmaf32(u, s3_x, F32(0.00836596917361021041870117))
    u = fmaf32(u, s3_x, F32(0.0416710823774337768554688))
    u = fmaf32(u, s3_x, F32(0.166665524244308471679688))
    u = fmaf32(u, s3_x, F32(0.499999850988388061523438))

    # t = dfadd(s, dfmul(dfsqu(s), u))
    sq_x = s3_x * s3_x
    sq_y = fmaf32(s3_x + s3_x, s3_y, fmaf32(s3_x, s3_x, -sq_x))
    mu_x = sq_x * u
    mu_y = fmaf32(sq_y, u, fmaf32(sq_x, u, -mu_x))
    t_x = s3_x + mu_x
    t_y = ((s3_x - t_x) + mu_x) + s3_y + mu_y

    # t = dfadd2(1, t)                           [df.h:146]
    t2_x = F32(1.0) + t_x
    t2_y = (F32(1.0) - t2_x) + t_x + t_y

    u = t2_x + t2_y
    u = _vldexp(u, q)

    if d_x < F32(-104.0):
        u = F32(0.0)
    return u


@njit(cache=True, fastmath=False, error_model="numpy")
def sleef_powf(x, y):
    """SLEEF xpowf (sleefsimdsp.c:1618) scalar port."""
    yisint = (np.trunc(y) == y) or (abs(y) > BIG24)
    yisodd = ((np.int32(y) & 1) == 1) and yisint and (abs(y) < BIG24)

    l_x, l_y = _logkf(abs(x))
    # p = dfmul(logkf_result, y)                 [df.h:214]
    p_x = l_x * y
    p_y = fmaf32(l_y, y, fmaf32(l_x, y, -p_x))
    result = _expkf(p_x, p_y)

    if result != result:
        result = INF

    if x > F32(0.0):
        sg = F32(1.0)
    elif yisint:
        sg = F32(-1.0) if yisodd else F32(1.0)
    else:
        sg = NAN_Q
    result = result * sg

    efx = i2f(np.int32(f2i(abs(x) - F32(1.0)) ^ np.int32(f2i(y) & np.int32(-2147483648))))
    if y == INF or y == -INF:
        r2 = F32(1.0) if efx == F32(0.0) else INF
        if efx < F32(0.0):
            r2 = F32(0.0)
        result = r2

    if x == INF or x == -INF or x == F32(0.0):
        xor_ = ((f2i(y) & np.int32(-2147483648)) != 0) != (x == F32(0.0))
        v = F32(0.0) if xor_ else INF
        sgn = x if yisodd else F32(1.0)
        v = i2f(np.int32(f2i(v) ^ np.int32(f2i(sgn) & np.int32(-2147483648))))
        result = v

    if x != x or y != y:
        result = i2f(np.int32(f2i(result) | np.int32(-1)))

    if y == F32(0.0) or x == F32(1.0):
        result = F32(1.0)
    return result


# --------------------------------------------------------------------------
# torch pow(Tensor, Scalar) — layout-routed primitive
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False, error_model="numpy")
def _is_libm_lane(i, n, T, grain):
    """True iff torch evaluates lane ``i`` of an n-lane elementwise op with
    the scalar remainder (torch opmath f64 pow) instead of the 4-lane SLEEF vector body
    (see module docstring for the chunk model)."""
    if T <= 1 or n <= grain:
        return i >= n - (n % 8)
    teff = (n + grain - 1) // grain
    if teff > T:
        teff = T
    C = (n + teff - 1) // teff
    k = i // C
    e = (k + 1) * C
    if e > n:
        e = n
    L = e - k * C
    return i >= e - (L % 8)


@njit(cache=True, fastmath=False, error_model="numpy")
def _torch_pow_scalar(x, e, use_libm):
    """torch.pow(t, e) for one lane: SLEEF xpowf body lanes; scalar-tail
    lanes use torch's opmath scalar pow — pow(float64(x), float64(e))
    single-rounded to f32, NOT the host f32 powf (T09 P9: torch n=1 and
    uniform-array tail lanes give 0x40d7da59 for (0x3fbb8123, 5.0) ==
    the f64 pow, while libm powf gives 0x40d7da58). ``e`` is a runtime
    f32 so numba cannot contract it to a multiply chain."""
    if use_libm:
        return F32(np.float64(x) ** np.float64(e))
    return sleef_powf(x, e)


@njit(cache=True, fastmath=False, error_model="numpy")
def torch_pow_scalar_exact(x, e):
    """torch pow(Tensor, Scalar) exact forms for e in {-2,-1,0,1,2,3} —
    position-independent (P5f: 0 mismatch on 21003-value UTCI lattice)."""
    if e == F32(-2.0):
        return F32(1.0) / (x * x)
    if e == F32(-1.0):
        return F32(1.0) / x
    if e == F32(0.0):
        return F32(1.0)
    if e == F32(1.0):
        return x
    if e == F32(2.0):
        return x * x
    return (x * x) * x  # e == 3


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def torch_pow_scalar_vector(xs, e, T, grain):
    """Vector form of torch.pow(xs, e) for any Scalar exponent: exact forms
    for |e|<=3, layout routing for e in {4,5,6} (and any other |e|>3 —
    verified only for 4/5/6; see HYBRID_STATUS notes)."""
    n = xs.size
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        ae = abs(e)
        if ae <= F32(3.0):
            out[i] = torch_pow_scalar_exact(xs[i], e)
        else:
            out[i] = _torch_pow_scalar(
                xs[i], e, _is_libm_lane(i, n, T, grain))
    return out


# --------------------------------------------------------------------------
# first-NaN-operand arithmetic — torch elementwise NaN-payload semantics
# --------------------------------------------------------------------------
# torch's elementwise add/sub/mul/div (plain vector loops, no SLEEF)
# deterministically propagate the FIRST array operand's NaN payload:
# nan_a + x -> payload(a), x - nan_b -> payload(b), etc. (verified on
# scalar and vector probes, T09). A scalar accumulation chain in numba
# cannot rely on that: LLVM's machine scheduler may commute commutative
# fp instructions per add site (empirically the winning payload OSCILLATES
# across the 210-term chain), so every fp add/sub/mul/div in the UTCI
# expression goes through these wrappers, which select the first NaN
# operand's bits explicitly. For finite operands the result is the plain
# IEEE op — bit-identical to the unwrapped form.
@njit(cache=True, fastmath=False, error_model="numpy")
def nadd(a, b):
    if a != a:
        return a
    if b != b:
        return b
    return a + b


@njit(cache=True, fastmath=False, error_model="numpy")
def nsub(a, b):
    if a != a:
        return a
    if b != b:
        return b
    return a - b


@njit(cache=True, fastmath=False, error_model="numpy")
def nmul(a, b):
    if a != a:
        return a
    if b != b:
        return b
    return a * b


@njit(cache=True, fastmath=False, error_model="numpy")
def ndiv(a, b):
    if a != a:
        return a
    if b != b:
        return b
    return a / b


# --------------------------------------------------------------------------
# primitive resolution table — read out by tests and the report
# --------------------------------------------------------------------------
HYBRID_STATUS = {
    "torch_free": True,
    "fallback_primitives": {},
    "notes": (
        "All UTCI primitives resolved CLEAN torch-free. torch.log mechanism: "
        "pytorch's bundled SLEEF pin 5a1d17 funcproto {\"log\",10,1,0,0} "
        "maps the u10 symbol to the xlogf_u1 body (disassembly constants "
        "0x3e9aff5c/0x3ecc99ca/0x3f2aaada decode to xlogf_u1 coefficients; "
        "P5d port 0-mismatch). pow scalar-exponent >=4: SLEEF xpowf vector "
        "body + opmath-f64 scalar-tail lanes (pow(f64, f64) rounded to f32, "
        "T09 P9) under the locked torch "
        "chunk layout (T=8 on site_500 per t08 capture manifest, grain "
        "32768) — extent/threads dependent by construction, parameters "
        "exposed as kernel arguments."
    ),
}

PRIMITIVE_RESOLUTION = {
    "add.f32": "clean: IEEE elementwise (T01 matrix)",
    "sub.f32": "clean: IEEE elementwise (T01 matrix)",
    "mul.f32": "clean: IEEE elementwise (T01 matrix)",
    "div.f32": "clean: IEEE elementwise (T01 matrix)",
    "exp.f32": "clean: SLEEF xexpf port, all lane classes (T09 P5/P5c)",
    "log.f32": "clean: SLEEF xlogf_u1 port, all lane classes (T09 P5c/P5d)",
    "pow_scalar.-2..3": "clean: torch exact forms (T09 P5f)",
    "pow_scalar.4,5,6": (
        "clean: SLEEF xpowf body + opmath-f64 chunk-tail lanes (P9) under the locked "
        "layout model (T09 P5e/P5f; 15/15 size validation)"
    ),
}


def get_hybrid_status():
    """Public readout — the compatibility layer's torch-fallback state.
    Never hidden: any future fallback MUST register itself here."""
    return dict(HYBRID_STATUS), dict(PRIMITIVE_RESOLUTION)
