# SPDX-License-Identifier: GPL-3.0-only
"""Torch-free regeneration of every met-dependent bundle field (T11).

Given the frozen-geometry capture of one timestep (t08 capture npz) plus an
EDITED met series (Ta / RH / radG / radD / radI / P / Twater), this module
regenerates bit-exactly the fields the fused radiation kernels consume that
depend on meteorology:

    day: Ta / Ta_f64 / radI / radD / ta273_pow4 / ks_sunlit / ks_shaded /
         lv / lumChi / Lsky_down2 / Lsky_side2 / Tg_pre / Tg / Tgwall /
         Lup_pre / Lwall / gvflup_extra / dp_*_surface / CI_out / ea / esky
    night: ea / esky / CI_out / ta273_pow4 / Lsky_down2 / Lsky_side2 /
         night_Lup / night_water_override / dp_*_surface

Dtype routing is the one measured in the t11 probes (p3..p9) against the
torch oracle (solweig.py @ c11ac4d9); every transcendental goes through the
SLEEF numba ports (sleef_dp.py f64, sleef_trig.py / math_compat.py f32,
sleef_f32_extra.py tan + libm powf). pow routing: torch 0-dim pow == libm
(python ``**`` for f64, fl32(math.pow(f64,f64)) for Tensor**Scalar-f32,
powf for Scalar**Tensor-f32 / Tensor**Tensor-0dim); torch f32 PLANE
``**4`` == math_compat.torch_pow_scalar_vector (T09).

Key dtype pins (oracle verbatim):
  * main-body zen/altitude/azimuth/dectime/altmax/Twater are pyfloat ARGS
    recorded as __f64 but tensorized INSIDE Solweig_2022a_calc
    (torch.tensor(pyfloat) -> f32), so corr / sinarg-numerator / Tgamp all
    start from fl32 casts of those values;
  * main-body Ta/RH/radG/radD/radI/P stay f64 0-dim tensors;
  * clearnessindex day call: zen/jday/RH(already /100)/radG/P as
    torch.tensor(pyfloat) f32 0-dims, Ta weak pyfloat (2nd call); the
    midnight f64 call keeps torch.tensor(G+1.) and a2/b2 f32;
  * diffusefraction ensure_tensors Ta/RH to f32 0-dims and divides RH/100
    INSIDE; alfa = altitude*(pi/180) computed in the CALLER's dtype
    (f32 for the body call, f64->fl32 for the I0 call).

Provenance fences (typed refusals, never silent no-ops):
  - geometry edits (this module regenerates MET only; geometry comes from
    the capture of the matching building profile),
  - landcover == 1 (live water bodies),
  - onlyglobal != 1 / anisotropic_sky != 1 / patch_option != 2 /
    elvis not in {0, 1} — unported model branches,
  - Perez altitude < 0 (complex-path AirMass unported).

Identity contract: with UNEDITED met (the capture's own arg_ values) the
regenerated fields equal the capture bits for every timestep (raw uint32 /
uint64 views). tests/ultrafast/test_met_recompute.py gates all 24.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .math_compat import sleef_expf, sleef_logf_u1, torch_pow_scalar_vector
from .sleef_dp import (
    sleef_cosd_u10,
    sleef_expd,
    sleef_logd_u1,
    sleef_sind_u10,
)
from .sleef_f32_extra import powf32, sleef_tanf_u10
from .sleef_trig import (
    sleef_acos_v1,
    sleef_asinf_u1,
    sleef_cos_v1,
    sleef_cosf_u1,
    sleef_sinf_u1,
    sleef_sin_v1,
)

F32 = np.float32

_PI = math.pi
_FL32_PI = F32(_PI)
_FL32_D2R = F32(_PI / 180.0)
_FL32_R2D = F32(180.0 / _PI)
_SBC32 = F32(5.67051e-8)

#: torch parallel grain/threads used by cpu_kernel_vec pow (T09 measured)
_TORCH_THREADS = 8
_TORCH_GRAIN = 32768


class MetEditRefusal(Exception):
    """Typed refusal: an edit family this module explicitly does not serve."""


# --------------------------------------------------------------------------
# f32 scalar helpers (torch promotion semantics)
# --------------------------------------------------------------------------
def _sqrt32(x):
    return np.sqrt(F32(x))  # IEEE sqrtf (torch sqrt kernel)


def _rsqrt32(x):
    # torch `**-0.5` on f32 -> rsqrt kernel -> fl32(1 / fl32(sqrt)) (p7)
    return F32(F32(1.0) / _sqrt32(x))


def _pow_scalar_f32(x, e):
    # torch pow(Tensor f32 0-dim, Scalar) / pow(Tensor, Tensor 0-dim):
    # scalar remainder std::pow(opmath double) rounded once (p6c/p7b)
    return F32(math.pow(float(F32(x)), float(e)))


def _min1(x: float) -> float:
    """python min(tensor_value, 1.0): 1.0 wins only when strictly smaller."""
    return x if not (1.0 < x) else 1.0


# --------------------------------------------------------------------------
# frozen patch tables (create_patches(2), shadow.py:352 — deterministic)
# --------------------------------------------------------------------------
def create_patch_tables_2():
    """(skyvaultalt_deg, skyvaultazi_deg) f32 arrays, create_patches(2)."""
    skyvaultaltint = [6, 18, 30, 42, 54, 66, 78, 90]
    azistart = [0, 4, 2, 5, 8, 0, 10, 0]
    patches_in_band = [31, 30, 28, 24, 19, 13, 7, 1]
    alts = []
    azis = []
    for j in range(8):
        c = patches_in_band[j]
        # torch `360 / int64 0-dim` reciprocal-multiply (patch_steradian
        # pin; 360/30 is 12.000001f, not the exact 12.0f of true division)
        aziint = F32(F32(360.0) * F32(1.0 / c))
        for k in range(patches_in_band[j]):
            alts.append(F32(skyvaultaltint[j]))
            azis.append(F32(F32(k) * aziint + F32(azistart[j])))
    return (np.array(alts, dtype=np.float32),
            np.array(azis, dtype=np.float32))


# --------------------------------------------------------------------------
# steradian table (Lcyl_v2022a:1709 / Kside lumChi replay, identical loop)
# --------------------------------------------------------------------------
def patch_steradian(patch_alt_deg32):
    """steradian[i] loop verbatim from the patch-altitude column."""
    pa = np.asarray(patch_alt_deg32, dtype=np.float32)
    n = pa.shape[0]
    skyalt, skyalt_c = np.unique(pa, return_counts=True)
    counts = {float(a): int(c) for a, c in zip(skyalt, skyalt_c)}
    pa0 = pa[0]
    ster = np.zeros(n, dtype=np.float32)
    for i in range(n):
        c = counts[float(pa[i])]
        # torch `360 / int64 0-dim` is a RECIPROCAL-MULTIPLY in f32
        # (scalar-numerator div rewrites to tensor * fl32(1/360-path);
        # pinned by probe P11g over all 8 Tregenza bands: 360/30 gives
        # 12.000001f, NOT the exact 12.0f of a true division)
        q32 = F32(F32(360.0) * F32(1.0 / c))
        base = q32 * _FL32_D2R  # q * deg2rad_t (f32 0-dim x f32 0-dim)
        if c > 1:
            ster[i] = base * (sleef_sinf_u1((pa[i] + pa0) * _FL32_D2R)
                              - sleef_sinf_u1((pa[i] - pa0) * _FL32_D2R))
        else:
            ster[i] = base * (sleef_sinf_u1(pa[i] * _FL32_D2R)
                              - sleef_sinf_u1((pa[i - 1] + pa0) * _FL32_D2R))
    return ster


# --------------------------------------------------------------------------
# daylen (solweig.py:47) — f32 0-dim chain; SNUP
# --------------------------------------------------------------------------
def daylen_snup_f32(doy_f: float, xlat_f: float):
    d = F32(doy_f)
    lat = F32(xlat_f)
    dec = F32(-23.45) * sleef_cosf_u1(
        F32(2.0 * _PI) * (d + F32(10.0)) / F32(365.0))
    soc = sleef_tanf_u10(F32(_PI / 180.0) * dec) * sleef_tanf_u10(
        F32(_PI / 180.0) * lat)
    soc = F32(min(max(float(soc), -1.0), 1.0))  # torch.clamp
    dayl = F32(12.0) + F32(24.0) * sleef_asinf_u1(soc) / F32(_PI)
    return F32(12.0) - dayl / F32(2.0)


# --------------------------------------------------------------------------
# sun_distance (solweig.py:835)
# --------------------------------------------------------------------------
def _sun_distance32(jday32):
    b = F32(2.0 * _PI) * jday32 / F32(365.0)
    inner = (F32(1.00011)
             + F32(0.034221) * sleef_cosf_u1(b)
             + F32(0.001280) * sleef_sinf_u1(b)
             + F32(0.000719) * sleef_cosf_u1(F32(2.0) * b)
             + F32(0.000077) * sleef_sinf_u1(F32(2.0) * b))
    return _sqrt32(inner)


def _sun_distance64(jday64):
    b = 2.0 * _PI * jday64 / 365.0
    inner = (1.00011
             + 0.034221 * sleef_cosd_u10(b)
             + 0.001280 * sleef_sind_u10(b)
             + 0.000719 * sleef_cosd_u10(2.0 * b)
             + 0.000077 * sleef_sind_u10(2.0 * b))
    return math.sqrt(inner)


# --------------------------------------------------------------------------
# clearnessindex_2013b (solweig.py:851)
# --------------------------------------------------------------------------
_CI_LAT_TABLE = [
    (10.0, [3.37, 2.85, 2.80, 2.64]),
    (20.0, [2.99, 3.02, 2.70, 2.93]),
    (30.0, [3.60, 3.00, 2.98, 2.93]),
    (40.0, [3.04, 3.11, 2.92, 2.94]),
    (50.0, [2.70, 2.95, 2.77, 2.71]),
    (60.0, [2.52, 3.07, 2.67, 2.93]),
    (70.0, [1.76, 2.69, 2.61, 2.61]),
    (80.0, [1.60, 1.67, 2.24, 2.63]),
    (90.0, [1.11, 1.44, 1.94, 2.02]),
]


def _ci_G(latitude: float, jday) -> float:
    g = _CI_LAT_TABLE[-1][1]
    for bound, table in _CI_LAT_TABLE:
        if latitude < bound:
            g = table
            break
    jf = float(jday)
    if jf > 335 or jf <= 60:
        return g[0]
    if jf <= 152:
        return g[1]
    if jf <= 244:
        return g[2]
    return g[3]


def clearnessindex_f32(zen32, jday32, Ta_py, RH100_32, radG32, P32,
                       latitude: float):
    """Day route: zen/jday/RH(/100 done)/radG/P f32 0-dims; Ta weak pyfloat.

    f32-semantics note: python-float arithmetic here is f64, so every torch
    'weak scalar' product must round through F32 at each step; numpy f32
    scalars keep each op in f32.
    """
    if P32 == F32(-999.0):
        p = F32(1013.0)
    else:
        p = P32 * F32(10.0)
    d = _sun_distance32(jday32)
    cosz = sleef_cosf_u1(zen32)
    m = F32(35.0) * cosz * _rsqrt32(F32(1224.0) * (cosz * cosz) + F32(1.0))
    trpg = (F32(1.021)
            - F32(0.084) * _sqrt32(m * (F32(0.000949) * p + F32(0.051))))
    g = _ci_G(latitude, jday32)
    ta32 = F32(Ta_py)
    a2 = F32(17.27)
    b2 = F32(237.7)
    x = (a2 * ta32) / (b2 + ta32) + sleef_logf_u1(RH100_32)
    td = (b2 * x) / (a2 - x)
    td = td * F32(1.8) + F32(32.0)
    u = sleef_expf(F32(0.1133) - sleef_logf_u1(F32(g + 1.0))
                   + F32(0.0393) * td)
    tw = F32(1.0) - F32(0.077) * _pow_scalar_f32(u * m, 0.3)
    tar = powf32(F32(0.935), m)
    i0 = ((((F32(1370.0) * cosz) * trpg) * tw) * d) * tar
    if abs(zen32) > F32(_PI / 2.0) or i0 != i0:
        i0 = F32(0.0)
    corr = (F32(0.1473)
            * sleef_logf_u1(F32(90.0) - (zen32 / _FL32_PI) * F32(180.0))
            + F32(0.3454))
    ci_uncorr = radG32 / i0
    ci = ci_uncorr + (F32(1.0) - corr)
    i0et = F32(1370.0) * cosz * d
    kt = radG32 / i0et
    return i0, ci, kt, i0et, ci_uncorr


def clearnessindex_f64(zen64, jday64, Ta64, RH100_64, radG64, P64,
                       latitude: float):
    """Midnight route: all f64 0-dims (RH already /100) except the f32
    torch.tensor(G+1.) and f32 a2/b2 constants inside the shared body."""
    if P64 == -999.0:
        p = 1013.0
    else:
        p = P64 * 10.0
    d = _sun_distance64(jday64)
    cosz = sleef_cosd_u10(zen64)
    m = 35.0 * cosz * (1.0 / math.sqrt(1224.0 * (cosz * cosz) + 1.0))
    trpg = 1.021 - 0.084 * math.sqrt(m * (0.000949 * p + 0.051))
    g = _ci_G(latitude, jday64)
    a2 = float(F32(17.27))
    b2 = float(F32(237.7))
    x = (a2 * Ta64) / (b2 + Ta64) + sleef_logd_u1(RH100_64)
    td = (b2 * x) / (a2 - x)
    td = td * 1.8 + 32.0
    # 0.1133 - torch.log(torch.tensor(G+1.)) is an f32 0-dim; the + 0.0393*Td
    # add promotes to f64 (fl64 of the f32 sum + f64 term)
    u = sleef_expd(float(F32(0.1133) - sleef_logf_u1(F32(g + 1.0)))
                   + 0.0393 * td)
    tw = 1.0 - 0.077 * ((u * m) ** 0.3)
    tar = 0.935 ** m
    i0 = ((((1370.0 * cosz) * trpg) * tw) * d) * tar
    if abs(zen64) > _PI / 2 or i0 != i0:
        i0 = 0.0
    corr = 0.1473 * sleef_logd_u1(90.0 - (zen64 / _PI) * 180.0) + 0.3454
    ci_uncorr = radG64 / i0
    ci = ci_uncorr + (1.0 - corr)
    i0et = 1370.0 * cosz * d
    kt = radG64 / i0et
    return i0, ci, kt, i0et, ci_uncorr


def midnight_ci_f64(zen64, jday64, Ta64, RH64, radG64, P64, latitude: float):
    """Driver midnight recompute: f64 clearnessindex + the (CI>1)|(CI==inf)
    clamp, verbatim from the time loop."""
    _, ci, _, _, _ = clearnessindex_f64(zen64, jday64, Ta64, RH64 / 100.0,
                                        radG64, P64, latitude)
    if (ci > 1.0) or math.isinf(ci):
        ci = 1.0
    return ci


# --------------------------------------------------------------------------
# diffusefraction (solweig.py:928)
# --------------------------------------------------------------------------
def _diffusefraction(radG32, alfa32, altitude_cmp, Kt, Ta32, RH32):
    """radI/radD f32 0-dims.

    altitude_cmp: the altitude ARG in its caller domain — an F32 scalar for
    the body call (torch.tensor(altitude.item())), or a python float for
    the I0 call (weak; compare `altitude < 1` then happens in f64, which is
    bit-equivalent because 1.0 is exact). Kt: F32 or python 1.0 (I0 call).
    """
    kt = F32(Kt)
    if (Ta32 <= F32(-999.00) or RH32 <= F32(-999.00)
            or Ta32 != Ta32 or RH32 != RH32):
        if kt <= F32(0.3):
            rad_d = radG32 * (F32(1.020) - F32(0.248) * kt)
        elif F32(0.3) < kt < F32(0.78):
            rad_d = radG32 * (F32(1.45) - F32(1.67) * kt)
        else:
            rad_d = radG32 * F32(0.147)
    else:
        rh100 = RH32 / F32(100.0)
        salfa = sleef_sinf_u1(alfa32)
        if kt <= F32(0.3):
            rad_d = radG32 * (F32(1.0) - F32(0.232) * kt
                              + F32(0.0239) * salfa
                              - F32(0.000682) * Ta32
                              + F32(0.0195) * rh100)
        elif F32(0.3) < kt < F32(0.78):
            rad_d = radG32 * (F32(1.329) - F32(1.716) * kt
                              + F32(0.267) * salfa
                              - F32(0.00357) * Ta32
                              + F32(0.106) * rh100)
        else:
            rad_d = radG32 * (F32(0.426) * kt
                              - F32(0.256) * salfa
                              + F32(0.00349) * Ta32
                              + F32(0.0734) * rh100)
    rad_i = (radG32 - rad_d) / sleef_sinf_u1(alfa32)
    if rad_i < F32(0.0):
        rad_i = F32(0.0)
    if altitude_cmp < 1 and rad_i > radG32:
        rad_i = radG32
    if rad_d > radG32:
        rad_d = radG32
    return rad_i, rad_d


# --------------------------------------------------------------------------
# Perez_v3 (solweig.py:1227) — f32, patch_option 2 tables
# --------------------------------------------------------------------------
_PZ_A1 = [1.3525, -1.2219, -1.1000, -0.5484, -0.6000, -1.0156, -1.0000, -1.0500]
_PZ_A2 = [-0.2576, -0.7730, -0.2515, -0.6654, -0.3566, -0.3670, 0.0211, 0.0289]
_PZ_A3 = [-0.2690, 1.4148, 0.8952, -0.2672, -2.5000, 1.0078, 0.5025, 0.4260]
_PZ_A4 = [-1.4366, 1.1016, 0.0156, 0.7117, 2.3250, 1.4051, -0.5119, 0.3590]
_PZ_B1 = [-0.7670, -0.2054, 0.2782, 0.7234, 0.2937, 0.2875, -0.3000, -0.3250]
_PZ_B2 = [0.0007, 0.0367, -0.1812, -0.6219, 0.0496, -0.5328, 0.1922, 0.1156]
_PZ_B3 = [1.2734, -3.9128, -4.5000, -5.6812, -5.6812, -3.8500, 0.7023, 0.7781]
_PZ_B4 = [-0.1233, 0.9156, 1.1766, 2.6297, 1.8415, 3.3750, -1.6317, 0.0025]
_PZ_C1 = [2.8000, 6.9750, 24.7219, 33.3389, 21.0000, 14.0000, 19.0000, 31.0625]
_PZ_C2 = [0.6004, 0.1774, -13.0812, -18.3000, -4.7656, -0.9999, -5.0000, -14.5000]
_PZ_C3 = [1.2375, 6.4477, -37.7000, -62.2500, -21.5906, -7.1406, 1.2438, -46.1148]
_PZ_C4 = [1.0000, -0.1239, 34.8438, 52.0781, 7.2492, 7.5469, -1.9094, 55.3750]
_PZ_D1 = [1.8734, -1.5798, -5.0000, -3.5000, -3.5000, -3.4000, -4.0000, -7.2312]
_PZ_D2 = [0.6297, -0.5081, 1.5218, 0.0016, -0.1554, -0.1078, 0.0250, 0.4050]
_PZ_D3 = [0.9738, -1.7812, 3.9229, 1.1477, 1.4062, -1.0750, 0.3844, 13.3500]
_PZ_D4 = [0.2809, 0.1080, -2.6204, 0.1062, 0.3988, 1.5702, 0.2656, 0.6234]
_PZ_E1 = [0.0356, 0.2624, -0.0156, 0.4659, 0.0032, -0.0672, 1.0468, 1.5000]
_PZ_E2 = [-0.1246, 0.0672, 0.1597, -0.3296, 0.0766, 0.4016, -0.3788, -0.6426]
_PZ_E3 = [-0.5718, -0.2190, 0.4199, -0.0876, -0.0656, 0.3017, -2.4517, 1.8564]
_PZ_E4 = [0.9938, -0.4285, -0.5562, -0.0329, -0.1294, -0.4844, 1.4656, 0.5636]


def perez_v3(zen_deg_f: float, azi_deg_f: float, radD32, radI32, jday_f,
             alt_deg32_arr, azi_deg32_arr):
    """lv (n_patches, 3) f32 — cols alt/azi degrees roundtrip + weight.

    zen_deg_f / azi_deg_f / jday_f arrive as .item() pyfloats of the f32
    0-dims the caller made; the F32() casts below reproduce the
    torch.tensor(pyfloat) f32 wraps inside Perez_v3.
    """
    zen32 = F32(zen_deg_f) * _FL32_D2R
    azi32 = F32(azi_deg_f) * _FL32_D2R
    alt32 = F32(90.0 - zen_deg_f) * _FL32_D2R
    if 90.0 - zen_deg_f < 0.0:
        raise MetEditRefusal(
            "perez_v3: altitude < 0 — complex-path AirMass unported")

    z3 = (zen32 * zen32) * zen32  # torch.pow(zen, 3)
    pc = (((radD32 + radI32) / (radD32 + F32(1.041) * z3))
          / (F32(1.0) + F32(1.041) * z3))
    day_angle = jday_f * 2.0 * _PI / 365.0  # pyfloat f64 chain
    da32 = F32(day_angle)
    two_da32 = da32 * F32(2.0)  # 2 * torch.tensor(day_angle)
    i0 = F32(1367.0) * (F32(1.00011)
                        + F32(0.034221) * sleef_cosf_u1(da32)
                        + F32(0.00128) * sleef_sinf_u1(da32)
                        + F32(0.000719) * sleef_cosf_u1(two_da32)
                        + F32(0.000077) * sleef_sinf_u1(two_da32))
    if alt32 >= F32(10.0) * _FL32_D2R:
        air_mass = F32(1.0) / sleef_sinf_u1(alt32)
    else:
        air_mass = (F32(1.0) / sleef_sinf_u1(alt32)
                    + F32(0.50572) * _pow_scalar_f32(
                        (F32(180.0) * alt32) / _FL32_PI + F32(6.07995),
                        -1.6364))
    pb = (air_mass * radD32) / i0
    if radD32 <= F32(10.0):
        pb = F32(0.0)

    if pc < F32(1.065):
        ic = 0
    elif pc < F32(1.230):
        ic = 1
    elif pc < F32(1.500):
        ic = 2
    elif pc < F32(1.950):
        ic = 3
    elif pc < F32(2.800):
        ic = 4
    elif pc < F32(4.500):
        ic = 5
    elif pc < F32(6.200):
        ic = 6
    else:
        ic = 7

    def _m4(c1, c2, c3, c4):
        c1v = F32(c1[ic])
        c2v = F32(c2[ic])
        c3v = F32(c3[ic])
        c4v = F32(c4[ic])
        return c1v + c2v * zen32 + pb * (c3v + c4v * zen32)

    m_a = _m4(_PZ_A1, _PZ_A2, _PZ_A3, _PZ_A4)
    m_b = _m4(_PZ_B1, _PZ_B2, _PZ_B3, _PZ_B4)
    m_e = _m4(_PZ_E1, _PZ_E2, _PZ_E3, _PZ_E4)
    if ic > 0:
        m_c = _m4(_PZ_C1, _PZ_C2, _PZ_C3, _PZ_C4)
        m_d = _m4(_PZ_D1, _PZ_D2, _PZ_D3, _PZ_D4)
    else:
        c0 = F32(_PZ_C1[0])
        c1v = F32(_PZ_C2[0])
        c2v = F32(_PZ_C3[0])
        d0 = F32(_PZ_D1[0])
        d1v = F32(_PZ_D2[0])
        d2v = F32(_PZ_D3[0])
        d3v = F32(_PZ_D4[0])
        m_c = sleef_expf(F32(powf32(pb * (c0 + c1v * zen32), c2v))) - F32(1.0)
        m_d = (-sleef_expf(F32(pb * (d0 + d1v * zen32))) + d2v
               + pb * d3v * pb)

    svalt_r = alt_deg32_arr * _FL32_D2R          # f32 vector
    sven_r = (F32(90.0) - alt_deg32_arr) * _FL32_D2R
    svazi_r = azi_deg32_arr * _FL32_D2R

    css = (sleef_sin_v1(svalt_r) * float(sleef_sinf_u1(alt32))
           + float(sleef_cosf_u1(alt32)) * sleef_cos_v1(svalt_r)
           * sleef_cos_v1(np.abs(svazi_r - azi32)))
    # oracle :1351-1352 — left-to-right association, m_e*css*css is
    # ((m_e x css) x css), and f2 adds chain left-to-right too
    lv = ((F32(1.0) + m_a * _expf_v(m_b / sleef_cos_v1(sven_r)))
          * (F32(1.0) + m_c * _expf_v(m_d * sleef_acos_v1(css))
             + (m_e * css) * css))
    lv = lv / _torch_sum_f32(lv)
    out = np.empty((alt_deg32_arr.shape[0], 3), dtype=np.float32)
    out[:, 0] = svalt_r * _FL32_R2D
    out[:, 1] = svazi_r * _FL32_R2D
    out[:, 2] = lv
    return out


def _expf_v(a):
    """f32 vector exp via the scalar SLEEF port (153 lanes; python loop)."""
    out = np.empty(a.shape, dtype=np.float32)
    flat = np.asarray(a, dtype=np.float32).ravel()
    of = out.ravel()
    for i in range(flat.size):
        of[i] = sleef_expf(flat[i])
    return out


def _ceil_log2(v: int) -> int:
    return 0 if v <= 1 else (v - 1).bit_length()


#: Vectorized<float>::size() on this arm64 host (NEON); probes P10i-P10k.
_VEC_F32 = 4
#: at::parallel_reduce grain — above this, equal chunks combined in order.
_SUM_GRAIN = 32768


def _cascade_chunk(x):
    """vectorized_inner_sum on one contiguous chunk (SumKernel.cpp v2.2.0).

    row_sum< vacc_t >(row, vec_stride=4*4B, vec_size=n/4) with ilp_factor 4
    -> multi_row_sum<4 levels, 4 rows> over the VECTOR array, then the
    row_sum tail into partial[0], the ilp fold, the SCALAR tail into
    final_acc, then the lane partials left-folded. Bitwise-pinned by the
    P10i probe battery (57k trials, n=1..40000).
    """
    n = x.size
    V = _VEC_F32
    if n < V:
        r = F32(0.0)
        for i in range(n):
            r = F32(r + x[i])
        return r
    vec_size = n // V
    size_ilp = vec_size // 4
    level_power = max(4, _ceil_log2(max(size_ilp, 1)) // 4)
    level_step = 1 << level_power
    acc = np.zeros((4, 4, V), dtype=np.float32)

    def vec(e):
        return x[e * V:e * V + V]

    i = 0
    while i + level_step <= size_ilp:
        for j in range(level_step):
            base = (i + j) * 4
            for k in range(4):
                acc[0, k] = np.add(acc[0, k], vec(base + k))
        i += level_step
        for lvl in range(1, 4):
            for k in range(4):
                acc[lvl, k] = np.add(acc[lvl, k], acc[lvl - 1, k])
                acc[lvl - 1, k] = F32(0.0)
            mask = (level_step - 1) << (lvl * level_power)
            if (i & mask) != 0:
                break
    while i < size_ilp:
        base = i * 4
        for k in range(4):
            acc[0, k] = np.add(acc[0, k], vec(base + k))
        i += 1
    for lvl in range(1, 4):
        for k in range(4):
            acc[0, k] = np.add(acc[0, k], acc[lvl, k])
    p = acc[0].copy()
    for e in range(size_ilp * 4, vec_size):
        p[0] = np.add(p[0], vec(e))
    for k in range(1, 4):
        p[0] = np.add(p[0], p[k])
    final = F32(0.0)
    for e in range(vec_size * V, n):
        final = F32(final + x[e])
    for lane in range(V):
        final = F32(final + p[0, lane])
    return final


def _torch_sum_f32(a):
    """torch.sum of a contiguous f32 1-d — the cascade_sum port.

    aten/src/ATen/native/cpu/SumKernel.cpp (v2.2.0; unchanged through the
    2.14 capture build per the probe batteries): output pre-zeroed +
    CastStoreAccumulate, so the value is the sequential combination of the
    parallel_reduce chunks (equal split when n > grain, combined in index
    order — P10k). Verified bitwise against torch on this host: P10i
    (0 mismatches, n <= 32768) and P10k (chunked n > 32768).
    """
    x = np.ascontiguousarray(a, dtype=np.float32).ravel()
    n = x.size
    if n == 0:
        return F32(0.0)
    if n <= _SUM_GRAIN:
        return _cascade_chunk(x)
    nchunks = -(-n // _SUM_GRAIN)
    base = n // nchunks
    r = F32(0.0)
    s = 0
    for c in range(nchunks):
        sz = base if c < nchunks - 1 else n - base * (nchunks - 1)
        r = F32(r + _cascade_chunk(x[s:s + sz]))
        s += sz
    return r


# --------------------------------------------------------------------------
# model2 (Martin & Berhalh 1984 band emissivity) + Lsky tables
# --------------------------------------------------------------------------
def model2_esky_band(esky64: float, skyalt_deg32):
    """esky_band per unique altitude — f32 chain (Lcyl/model2 verbatim).

    torch.exp returns a dimensioned f32 tensor; (1 - esky) is an f64 0-dim
    tensor, and the dimensioned f32 operand wins the promotion, so the
    band math is fl32: the (1-esky) 0-dim is cast to f32 BEFORE the vector
    multiply (TensorIterator common-dtype cast), 1 - ... is an f32 sub.
    """
    skyalt = np.unique(np.asarray(skyalt_deg32, dtype=np.float32))
    bands = np.empty(skyalt.shape[0], dtype=np.float32)
    one_minus32 = F32(1.0 - esky64)  # f64 sub, then the pre-mult f32 cast
    for i, a32 in enumerate(skyalt):
        skyzen32 = F32(90.0) - a32
        cz = sleef_cosf_u1(skyzen32 * _FL32_D2R)
        t32 = sleef_expf(F32(0.308) * (F32(1.7) - F32(1.0) / cz))
        bands[i] = F32(1.0 - F32(one_minus32 * t32))
    return skyalt, bands


def lsky_tables(esky64: float, Ta64: float, patch_alt_deg32):
    """(Lsky_down2, Lsky_side2, steradian) col-2 vectors, Lcyl verbatim.

    Per band, ALL f32 (P11h torch-replay 0-differ): te·SBC is an f32
    1-vector; ×(Ta+273.15)**4 (f64 0-dim) keeps the dimensioned f32 dtype
    (ta273 rounds fl32 pre-mult); / torch.tensor(np.pi) is a cpu-scalar
    denominator -> ATen reciprocal-multiply in f32; then × steradian ×
    sin/cos, one f32 rounding per op, left-to-right.
    """
    pa = np.asarray(patch_alt_deg32, dtype=np.float32)
    ster = patch_steradian(pa)
    skyalt, bands32 = model2_esky_band(esky64, pa)
    ta273_64 = (Ta64 + 273.15) ** 4  # f64 0-dim pow == std::pow == python **
    ta273_32 = F32(ta273_64)
    down = np.empty(pa.shape[0], dtype=np.float32)
    side = np.empty(pa.shape[0], dtype=np.float32)
    # torch.tensor(np.pi) is an f32 0-dim; A / it is a TRUE f32 division
    # (P11k: fl32(A/fl32(pi)) matches torch on every band value where the
    # reciprocal-multiply flavor is 1 ulp off — no ATen scalar rewrite on
    # this operand shape, unlike 360/int64-0dim in patch_steradian)
    for i in range(pa.shape[0]):
        band32 = bands32[int(np.searchsorted(skyalt, pa[i]))]
        # te (f32 1-vector) * SBC (f32 0-dim) * ta273 (f64 0-dim): the
        # dimensioned f32 operand wins, so ta273 rounds fl32 BEFORE the
        # multiply — every step f32, one rounding each (P11i: the f64
        # chain loses 1 bit on band 0)
        a32 = F32(F32(band32 * _SBC32) * ta273_32)
        q32 = F32(a32 / _FL32_PI)
        down[i] = F32(F32(q32 * ster[i])
                      * sleef_sinf_u1(pa[i] * _FL32_D2R))
        side[i] = F32(F32(q32 * ster[i])
                      * sleef_cosf_u1(pa[i] * _FL32_D2R))
    return down, side, ster


def lumChi_table(lv, radD_k32):
    """frozen_lumChi: radTot loop + (lv[:,2]*radD)/radTot."""
    lv_t = np.asarray(lv, dtype=np.float32)
    pa = lv_t[:, 0]
    lum = lv_t[:, 2]
    ster = patch_steradian(pa)
    rad_tot = F32(0.0)
    for i in range(pa.shape[0]):
        rad_tot = F32(rad_tot + lum[i] * ster[i]
                      * sleef_sinf_u1(pa[i] * _FL32_D2R))
    return (lum * F32(radD_k32)) / rad_tot


# --------------------------------------------------------------------------
# inputs / site context
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetInputs:
    Ta: float
    RH: float
    radG: float
    radD: float
    radI: float
    P: float
    Twater: float = 15.0
    #: capture sentinel: Twater was the initial empty python LIST (non-water
    #: site, never established by a midnight row) — the oracle then
    #: broadcasts an empty tensor into Lup[lc_grid == 3] and the recorded
    #: night water override bit is 0.0 (radiation_oracle.py:386-392).
    Twater_is_list: bool = False


@dataclass(frozen=True)
class TimeGeom:
    zen: float
    altitude: float
    azimuth: float
    jday: float
    dectime: float
    altmax: float


@dataclass(frozen=True)
class SiteParams:
    latitude: float
    onlyglobal: int
    anisotropic_sky: int
    patch_option: int
    landcover: int
    elvis: int
    albedo_b: float
    ewall: float
    TmaxLST: float
    TmaxLST_wall: float
    TgK_wall: float
    Tstart_wall: float
    TgK: np.ndarray          # f32 plane
    Tstart: np.ndarray       # f32 plane
    emis_grid: np.ndarray    # f32 plane
    buildings: np.ndarray    # f32 plane


def check_site_support(site: SiteParams) -> None:
    if site.landcover == 1:
        raise MetEditRefusal(
            "landcover == 1 (live water bodies): Twater mutation + lc_grid "
            "water surfaces are not served by met_recompute")
    if site.onlyglobal != 1:
        raise MetEditRefusal(
            f"onlyglobal == {site.onlyglobal} != 1: measured radI/radD path "
            "unported (diffusefraction split not taken)")
    if site.anisotropic_sky != 1:
        raise MetEditRefusal(
            f"anisotropic_sky == {site.anisotropic_sky} != 1: isotropic "
            "Ldown branch unported")
    if site.patch_option != 2:
        raise MetEditRefusal(
            f"patch_option == {site.patch_option} != 2: only the 153-patch "
            "Tregenza table is ported")
    if site.elvis not in (0, 1):
        raise MetEditRefusal(f"elvis == {site.elvis} not in {{0, 1}}")


def site_params_from_capture(z, latitude: float) -> SiteParams:
    """Site context from a t08 capture. latitude is a REQUIRED parameter:
    the capture records no latitude key (verified against the manifest),
    so the caller supplies it with provenance (site config / capture
    driver constants)."""
    site = SiteParams(
        latitude=float(latitude),
        onlyglobal=int(z["arg_onlyglobal__int"]),
        anisotropic_sky=int(z["arg_anisotropic_sky__int"]),
        patch_option=int(z["arg_patch_option__int"]),
        landcover=int(z["arg_landcover__int"]),
        elvis=int(z["arg_elvis__int"]),
        albedo_b=float(z["arg_albedo_b__f64"]),
        ewall=float(z["arg_ewall__f64"]),
        TmaxLST=float(z["arg_TmaxLST__f64"]),
        TmaxLST_wall=float(z["arg_TmaxLST_wall__f64"]),
        TgK_wall=float(z["arg_TgK_wall__f64"]),
        Tstart_wall=float(z["arg_Tstart_wall__f64"]),
        TgK=np.asarray(z["arg_TgK"], dtype=np.float32),
        Tstart=np.asarray(z["arg_Tstart"], dtype=np.float32),
        emis_grid=np.asarray(z["arg_emis_grid"], dtype=np.float32),
        buildings=np.asarray(z["arg_buildings"], dtype=np.float32),
    )
    check_site_support(site)
    return site


def time_geom_from_capture(z) -> TimeGeom:
    return TimeGeom(
        zen=float(z["arg_zen__f64"]),
        altitude=float(z["arg_altitude__f64"]),
        azimuth=float(z["arg_azimuth__f64"]),
        jday=float(z["arg_jday__f64"]),
        dectime=float(z["arg_dectime__f64"]),
        altmax=float(z["arg_altmax__f64"]),
    )


def met_from_capture(z) -> MetInputs:
    # frozen_Twater_used_f64 / frozen_Twater_is_list are only serialized on
    # timesteps whose frozen derivations consumed them; Twater is inert
    # unless landcover == 1 (refused), so default 15.0 where absent.
    is_list = bool(int(z["frozen_Twater_is_list"])) \
        if "frozen_Twater_is_list" in z.files else False
    if "frozen_Twater_used_f64" in z.files:
        tw = float(z["frozen_Twater_used_f64"])
    else:
        tw = 0.0 if is_list else 15.0
    return MetInputs(
        Ta=float(z["arg_Ta__0d"]),
        RH=float(z["arg_RH__0d"]),
        radG=float(z["arg_radG__0d"]),
        radD=float(z["arg_radD__0d"]),
        radI=float(z["arg_radI__0d"]),
        P=float(z["arg_P__0d"]),
        Twater=tw,
        Twater_is_list=is_list,
    )


# --------------------------------------------------------------------------
# timestep recompute
# --------------------------------------------------------------------------
def recompute_timestep(site: SiteParams, met: MetInputs, tg: TimeGeom,
                        CI_thread: float, shadow_plane,
                        patch_alt_deg32=None, patch_azi_deg32=None,
                        CI_thread_is_tensor: bool = False):
    """Regenerate every met-dependent bundle field for one timestep.

    shadow_plane: the day shadow plane (sunon_in_shadow bits) — geometry
    frozen from the capture of the SAME building profile (provenance fence:
    a geometry edit must regenerate the capture, not reuse this one).
    CI_thread / CI_thread_is_tensor: the driver-threaded clearness index
    (utci_process.py:967 assigns the RETURNED CI back into the loop var).
    Night steps consume it; the flavor matters — a day return of
    min(CI, 1.0) is an f32 TENSOR when raw CI <= 1.0 (its esky correction
    runs CI*esky with the f32 value promoted, and (1-CI)*1. rounds fl32
    BEFORE the f64 add) and the pyfloat 1.0 otherwise (no correction).
    Initial state and midnight-clamped values are pyfloat 1.0.
    Returns a dict of numpy scalars / arrays keyed by bundle field name.
    """
    check_site_support(site)
    if patch_alt_deg32 is None or patch_azi_deg32 is None:
        patch_alt_deg32, patch_azi_deg32 = create_patch_tables_2()

    Ta64 = float(met.Ta)
    RH64 = float(met.RH)
    radG64 = float(met.radG)
    ta273_64 = (Ta64 + 273.15) ** 4  # torch f64 pow == libm == python **

    # Vapor pressure / Prata emissivity (all f64)
    ea = (6.107 * 10.0 ** ((7.5 * Ta64) / (237.3 + Ta64))
          * (RH64 / 100.0))
    msteg = 46.5 * (ea / (Ta64 + 273.15))
    esky = (1 - (1 + msteg) * sleef_expd(
        -((1.2 + 3.0 * msteg) ** 0.5))) + site.elvis

    if tg.altitude > 0:
        return _recompute_day(site, met, tg, Ta64, RH64, radG64, ta273_64,
                              ea, esky, shadow_plane, patch_alt_deg32,
                              patch_azi_deg32)
    return _recompute_night(site, met, tg, Ta64, ta273_64, ea, esky,
                            CI_thread, patch_alt_deg32,
                            CI_thread_is_tensor)


def _recompute_day(site, met, tg, Ta64, RH64, radG64, ta273_64, ea, esky,
                   shadow_plane, patch_alt, patch_azi):
    zen32 = F32(tg.zen)
    jday32 = F32(tg.jday)
    alt32 = F32(tg.altitude)
    dec32 = F32(tg.dectime)
    altmax32 = F32(tg.altmax)
    ta32 = F32(Ta64)
    rh32 = F32(RH64)
    radg32 = F32(radG64)

    # clearnessindex — 2nd-call semantics (Ta weak == fl32; identical bits
    # to the discarded 1st call)
    i0, ci_fresh, kt, _, _ = clearnessindex_f32(
        zen32, jday32, met.Ta, rh32 / F32(100.0), radg32, F32(met.P),
        site.latitude)
    ret_ci = _min1(float(ci_fresh))  # python min(CI, 1.0)

    # radI / radD (body call: everything f32 0-dim)
    alfa_body32 = alt32 * F32(_PI / 180.0)
    rad_i, rad_d = _diffusefraction(radg32, alfa_body32, alt32, kt,
                                    ta32, rh32)

    # Tg ramp — MIXED-precision chain (oracle :2103-2107, pinned by the
    # P11d probes): dectime / altmax are F64 0-dim tensors in the body
    # (sun_position_sp returns numpy f64 -> torch.tensor keeps f64), so
    #   num = (dectime - floor) - SNUP/24    -> f64 minus an f32 0-dim
    #                                           (SNUP from f32 daylen)
    #   den = TmaxLST/24 - SNUP/24            -> F32 (weak pyfloat - f32)
    #   arg = (num / den) * np.pi/2           -> f64
    #   torch.sin(arg)                        -> f64 (== sleef_sind_u10)
    # Tgamp = TgK(f32 PLANE) * altmax(f64 0-dim) -> dimensioned f32 wins;
    # Tgampwall = TgK_wall(pyfloat) * altmax(f64) + Tstart_wall -> f64.
    snup32 = daylen_snup_f32(tg.jday, site.latitude)
    snup24_32 = snup32 / F32(24.0)
    num64 = (tg.dectime - float(math.floor(tg.dectime))) - float(snup24_32)
    den32 = F32(site.TmaxLST / 24.0) - snup24_32
    denw32 = F32(site.TmaxLST_wall / 24.0) - snup24_32
    sin64 = sleef_sind_u10(np.float64(
        (num64 / float(den32)) * (_PI / 2.0)))
    sinw64 = sleef_sind_u10(np.float64(
        (num64 / float(denw32)) * (_PI / 2.0)))
    tgamp = site.TgK * altmax32 + site.Tstart    # f32 plane, fl32(altmax)
    tg_pre_sin = tgamp * F32(sin64)              # plane * fl32(f64 sin)
    ampw64 = site.TgK_wall * tg.altmax + site.Tstart_wall            # f64
    tgwall64 = float(ampw64 * sinw64)
    if tgwall64 < 0.0:
        tgwall64 = 0.0

    # radI0 call (Kt = 1., weak altitude -> f64 alfa -> fl32; the
    # `altitude < 1` compare runs in f64 on the weak pyfloat — exact either
    # way since 1.0 is exact)
    rad_i0, rad_d0 = _diffusefraction(
        i0, F32(tg.altitude * (_PI / 180.0)), tg.altitude, 1.0, ta32, rh32)
    # corr / radG0 run on the F64 body zen/altitude tensors (oracle
    # :2109-2117): corr is an f64 0-dim (logd port), radG0 promotes f64
    # (radI0 f32 0-dim x sin64 f64 0-dim -> 0-dim x 0-dim -> f64).
    corr64 = (0.1473 * sleef_logd_u1(
        90.0 - (tg.zen / _PI) * 180.0) + 0.3454)
    # CI_Tg/CI_TgG raw are f64 0-dims (radG f64 0-dim / promoted f64);
    # min(x, 1.0) keeps the TENSOR when raw <= 1.0 and takes the pyfloat
    # 1.0 when raw > 1.0. The flavor decides the Tg/Tgwall multiply dtype:
    #   Tg (f32 plane) x f64 0-dim tensor -> dimensioned f32 wins -> fl32
    #   Tgwall (f64 0-dim) x f64 0-dim tensor -> f64 mult;  x pyfloat 1.0
    #   -> the f64 value unchanged
    ci_tg_raw = radG64 / float(rad_i0) + (1.0 - corr64)
    radg0_64 = (float(rad_i0) * float(sleef_sind_u10(
        np.float64(tg.altitude * (_PI / 180.0)))) + float(rad_d0))
    ci_tgg_raw = radG64 / radg0_64 + (1.0 - corr64)
    ci_tgg_is_tensor = not (1.0 < ci_tgg_raw)
    ci_tgg = ci_tgg_raw if ci_tgg_is_tensor else 1.0
    ci_tg = ci_tg_raw if not (1.0 < ci_tg_raw) else 1.0
    tg_plane = tg_pre_sin * F32(ci_tgg)
    if ci_tgg_is_tensor:
        tgwall64 = tgwall64 * ci_tgg_raw

    # ks sunlit / shaded (weak albedo_b)
    cosalt32 = sleef_cosf_u1(alt32 * F32(_PI / 180.0))
    ks_sun = ((F32(site.albedo_b) * (rad_i * cosalt32)
               + rad_d * F32(0.5)) / _FL32_PI)
    ks_shd = (((F32(site.albedo_b) * rad_d) * F32(0.5)) / _FL32_PI)

    # esky correction with the FRESH day CI — after the day/night branch,
    # before Lcyl. CI is an f32 0-dim tensor (min kept the tensor side):
    # CI*esky promotes f64 (CI cast up), (1-CI)*1. stays f32, the final
    # 0-dim + 0-dim add promotes f64 — so the second term rounds to fl32
    # BEFORE the f64 add.
    if float(ci_fresh) < float(F32(0.95)):
        ci32 = F32(ci_fresh)
        esky = float(ci32) * esky + float(F32(1.0 - ci32))

    # Perez lv + lumChi + Lsky. zenDeg is an F64 chain in the body
    # (zen f64 0-dim * (180 / np.pi)); the F32 cast happens INSIDE
    # Perez_v3 (torch.tensor(pyfloat) -> f32), so keep the f64 product
    zen_deg_f = tg.zen * (180.0 / _PI)
    azi_deg_f = float(F32(tg.azimuth))
    lv = perez_v3(zen_deg_f, azi_deg_f, rad_d, rad_i, float(jday32),
                  patch_alt, patch_azi)
    lumchi = lumChi_table(lv, rad_d)
    lsky_down, lsky_side, ster = lsky_tables(esky, Ta64, lv[:, 0])

    # sunonsurface Lup / Lwall (radiation_oracle derive_lup_lwall verbatim)
    emis32 = site.emis_grid
    se = _SBC32 * emis32                       # f32 plane
    core = (tg_plane * np.asarray(shadow_plane, dtype=np.float32)
            + F32(Ta64)) + F32(273.15)         # f32 plane
    p4 = torch_pow_scalar_vector(core.ravel(), F32(4.0),
                                 _TORCH_THREADS, _TORCH_GRAIN).reshape(
                                     core.shape)
    lup_pre = se * p4 - se * F32(ta273_64)
    c32 = _SBC32 * F32(site.ewall)             # f32 0-dim
    lwall64 = (float(c32) * ((tgwall64 + Ta64 + 273.15) ** 4)
               - float(c32) * ta273_64)
    lwall32 = F32(lwall64)

    # gvflup extra (post-mutation Tg == Tg for landcover != 1)
    buildings32 = np.asarray(site.buildings, dtype=np.float32)
    extra = lup_pre * (buildings32 * F32(-1.0) + F32(1.0))

    # dp surfaces (derive_dp_surfaces verbatim; sunlit order Ta + Tgwall)
    dp_veg = (float(c32) * ta273_64) / float(_FL32_PI)
    dp_sun = (float(c32) * ((Ta64 + tgwall64 + 273.15) ** 4)) / float(
        _FL32_PI)

    return {
        "ea": np.float64(ea),
        "esky": np.float64(esky),
        "ret_CI": np.float64(float(F32(ret_ci))),
        "CI_out": F32(ret_ci),
        "CI_flavor_is_tensor": bool(float(ci_fresh) <= 1.0),
        "CI_Tg": np.float64(ci_tg),
        "CI_TgG": np.float64(ci_tgg),
        "radI": rad_i,
        "radD": rad_d,
        "I0": i0,
        "Kt": kt,
        "Ta_f64": np.float64(Ta64),
        "Ta32": ta32,
        "ta273_pow4_f64": np.float64(ta273_64),
        "ks_sunlit": ks_sun,
        "ks_shaded": ks_shd,
        "lv": lv,
        "lumChi": lumchi,
        "steradian": ster,
        "Lsky_down2": lsky_down,
        "Lsky_side2": lsky_side,
        "Tg_pre": tg_plane,
        "Tg": tg_plane,
        "Tgwall_f64": np.float64(tgwall64),
        "Tgwall32": F32(tgwall64),
        "Lup_pre": lup_pre,
        "Lwall32": lwall32,
        "gvflup_extra": extra,
        "dp_veg_surface_f64": np.float64(dp_veg),
        "dp_shaded_surface_f64": np.float64(dp_veg),
        "dp_sunlit_surface_f64": np.float64(dp_sun),
        "Twater32": F32(met.Twater),
    }


def _recompute_night(site, met, tg, Ta64, ta273_64, ea, esky, CI_thread,
                     patch_alt, ci_thread_is_tensor):
    # esky correction with the THREADED CI (oracle :2253-2256). Tensor
    # flavor (f32 0-dim): CI*esky promotes f64 from the f32 value and
    # (1-CI)*1. rounds fl32 BEFORE the f64 add. Pyfloat flavor is always
    # 1.0 -> the `CI < 0.95` guard is False -> no correction.
    ci64 = float(CI_thread)
    if ci_thread_is_tensor and ci64 < 0.95:
        ci32 = F32(ci64)
        esky = float(ci32) * esky + float(F32(1.0 - ci32))

    lsky_down, lsky_side, ster = lsky_tables(esky, Ta64, patch_alt)

    # night Lup: zero planes + fl32(Ta) + 273.15 -> uniform f32 plane pow4
    emis32 = np.asarray(site.emis_grid, dtype=np.float32)
    u32 = F32(F32(Ta64) + F32(273.15))
    rows, cols = emis32.shape
    core = np.full((rows, cols), u32, dtype=np.float32)
    p4 = torch_pow_scalar_vector(core.ravel(), F32(4.0),
                                 _TORCH_THREADS, _TORCH_GRAIN).reshape(
                                     core.shape)
    night_lup = (_SBC32 * emis32) * p4

    # nocturnal water override (0-dim f32 pow -> scalar remainder). The
    # empty-list Twater sentinel records the bit 0.0 — the oracle's
    # Lup[lc_grid == 3] = (empty broadcast) never writes on a water-free
    # grid, and the harness freezes that honestly (radiation_oracle.py).
    if met.Twater_is_list:
        water_override = F32(0.0)
    else:
        tw32 = F32(met.Twater)
        water_override = F32((_SBC32 * F32(0.98))
                             * _pow_scalar_f32(tw32 + F32(273.15), 4.0))

    # dp surfaces with Tgwall = 0 (int64 night zero)
    c32 = _SBC32 * F32(site.ewall)
    dp_veg = (float(c32) * ta273_64) / float(_FL32_PI)
    dp_sun = (float(c32) * ((Ta64 + 0.0 + 273.15) ** 4)) / float(_FL32_PI)

    return {
        "ea": np.float64(ea),
        "esky": np.float64(esky),
        "ret_CI": np.float64(ci64),
        "CI_out": F32(ci64),
        "CI_flavor_is_tensor": bool(ci_thread_is_tensor),
        "CI_Tg": np.float64(ci64),
        "CI_TgG": np.float64(ci64),
        "Ta_f64": np.float64(Ta64),
        "ta273_pow4_f64": np.float64(ta273_64),
        "Lsky_down2": lsky_down,
        "Lsky_side2": lsky_side,
        "steradian": ster,
        "night_Lup": night_lup,
        "night_water_override": water_override,
        "dp_veg_surface_f64": np.float64(dp_veg),
        "dp_shaded_surface_f64": np.float64(dp_veg),
        "dp_sunlit_surface_f64": np.float64(dp_sun),
        "Twater32": F32(met.Twater),
    }


# --------------------------------------------------------------------------
# Twater midnight advance — torch-free (utci_process.py:921-923 verbatim:
# Twater = np.mean(Ta_[jday[0] == np.floor(dectime[i])]) at midnight rows
# or i == 0, landcover == 1 sites; Ta_ is the f64 numpy met series, so the
# reduction is numpy's blocked-pairwise f64 mean — identical when replayed
# on numpy here; met edits thread through the Ta rows directly)
# --------------------------------------------------------------------------
def twater_advance(Ta_f64_rows):
    """Twater = np.mean over the f64 Ta rows of the current day.

    Ta_f64_rows: Ta[jday[0] == floor(dectime[i])] as a contiguous f64
    array (the full-solve driver builds the mask from jday / dectime).
    Returns the f64 mean the oracle stores into Twater.
    """
    rows = np.ascontiguousarray(Ta_f64_rows, dtype=np.float64)
    if rows.size == 0:
        raise MetEditRefusal(
            "twater_advance: no Ta rows matched the day — jday/dectime "
            "mask inconsistent with the met series")
    return float(np.mean(rows))
