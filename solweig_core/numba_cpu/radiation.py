# SPDX-License-Identifier: GPL-3.0-only
"""T08 radiation stage: hoisted mirror + ordered fusion (torch-free).

Bit-exact re-implementation of the solweig_gpu radiation chain for the
site_500 configuration (cyl=1, anisotropic sky=1, landcover=1,
usevegdem=1, onlyglobal=1, patch_option=2, 153 patches), consuming:

  * static scene planes + packed shadow cubes (solweig_core.bitplanes),
  * a per-timestep FROZEN bundle of transcendental results captured from
    the oracle (uint32 bit patterns; producer = tests/ultrafast/
    radiation_oracle.py). Frozen because torch pow/sin/cos/exp match no
    torch-free recompute on this host (T01 matrix + t08 probes):
      - plane ** 4 / 0-dim ** 4 : torch SLEEF powf 1-ULP off libm (~2%)
        and ~50% off multiply chains -- Lup/Lwall/sunlit_surface/night
        Lup planes and the (Ta+273.15)**4 scalar are frozen bits.
      - sin/cos/tan/exp scalars (sinalt, cosalt, weight1, esky chain,
        radI/radD/radI0, corr, CI_Tg/CI_TgG, steradian, lumChi, Lsky
        tables, lv) frozen bits.
      - shaded_or_sunlit per-(t,patch) bool planes frozen + packed.
  * per-timestep march outputs (vegsh, sh, wallsun planes).

Everything else is CLEAN float32 plane arithmetic (numpy == torch == numba
bit-equal on the used domains; t08 probe evidence) mirrored in the ORIGINAL
expression order: no reassociation, no FMA, no fastmath, masked fills keep
their exact semantics, torch maximum's -0.0-tie/NaN-canonical form is
mirrored explicitly.

Ordered fusion (DESIGN 9.3): define_patch's reflected_on_surfaces term
(line ~1614) depends only on the SAME cell's completed first-pass
Ldown_sky; the two passes fuse per cell with a cell-local barrier. The
fused Numba kernel below performs the whole daytime radiation stage
per cell in original loop order (18-azimuth walk -> TsWaveDelay -> Kup/
Kside -> define_patch two-pass -> Lside_veg -> Sstr -> Tmrt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

__all__ = [
    "RadiationStatic", "RadTimestepInputs", "RadLoopState",
    "mirror_gvf", "mirror_sunonsurface", "mirror_kup", "mirror_kside",
    "mirror_tsw_delay", "mirror_define_patch", "mirror_lside_veg",
    "mirror_radiation_timestep", "TORCH_MAXIMUM_NAN_U32",
    "rad_static_from_capture", "rad_bundle_from_capture",
    "fused_radiation_timestep", "dense_diffsh_from_packed",
]

SBC32 = np.float32(5.67051e-8)
PI32 = np.float32(np.pi)
#: diffsh mirror constant (static): fl32(1.0) - fl32(1.0 - 0.03) in the
#: source expression ``sh - (1 - veg) * fl32(1.0 - 0.03)``. Module-level
#: so the T14 Change F mutation witness can pin/verify it; the kernels
#: bake it at compile time, the numpy mirrors read it at call time —
#: both MUST stay this exact value (fl32(0.97) is value-identical,
#: 0x3f7851ec; anything else changes diffsh bits).
DIFFSH_C97 = np.float32(1.0 - 0.03)
#: torch NaN canonical payload observed for max/min/log (T01)
TORCH_MAXIMUM_NAN_U32 = np.uint32(0x7FC00000)


def torch_maximum_scalar(x: np.ndarray, v: np.float32) -> np.ndarray:
    """torch.maximum(plane, 0-dim) mirror: NaN-canonical, self on tie.

    Probe-3 evidence: torch keeps -0.0 when self=-0.0/other=+0.0 (tie ->
    self) and canonicalizes NaN to 0x7FC00000; numpy maximum returns +0.0
    on that tie and propagates NaN payloads.
    """
    out = np.where(x >= v, x, v).astype(np.float32, copy=False)
    nan_mask = np.isnan(x)
    if nan_mask.any():
        out = out.copy()
        out[nan_mask] = np.float32(np.nan)
    return out


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------

@dataclass
class RadiationStatic:
    """Timestep-invariant scene (plain numpy; planes are f32 C-contig)."""
    rows: int
    cols: int
    buildings: np.ndarray
    walls: np.ndarray
    aspect_rad: np.ndarray      # (dirwalls * pi / 180) torch-order product
    wallbol: np.ndarray
    lc_grid: np.ndarray = None   # unused: water mutation frozen (Tg planes)
    alb_grid: np.ndarray = None
    emis_grid: np.ndarray = None
    TgK: np.ndarray = None
    Tstart: np.ndarray = None
    TgK_wall: np.ndarray = None
    Tstart_wall: np.ndarray = None
    TmaxLST: np.ndarray = None
    TmaxLST_wall: np.ndarray = None
    svfbuveg: np.ndarray = None
    shmat_packed: np.ndarray = None    # (rows, cols, ceil(P/8)) uint8
    vegshmat_packed: np.ndarray = None
    vbshvegshmat_packed: np.ndarray = None
    #: real patch count P (the packed cubes carry ceil(P/8)*8 bit slots;
    #: slots >= P are padding). Kept so dense materialization and the
    #: kernels agree with the frozen patch tables.
    n_patches: int = None
    #: dense diffsh cube — RETIRED from the runtime path (T14 Change F
    #: RSS diet: 146 MiB resident + 3x146 MiB load transients at
    #: site_500). None unless a consumer explicitly asks for
    #: ``rad_static_from_capture(..., dense_diffsh=True)`` (CUDA host
    #: bundle); kernels/mirrors recompute per cell/per patch, bit-equal.
    diffsh: np.ndarray = field(default=None)


@dataclass
class RadTimestepInputs:
    """Per-timestep frozen bundle + march outputs."""
    is_day: bool
    # march outputs (read-extent planes)
    vegsh: np.ndarray = None
    sh: np.ndarray = None
    wallsun: np.ndarray = None
    shadow: np.ndarray = None          # sh - (1-vegsh)*(1-psi), CLEAN
    # scalars (frozen bits; f32 unless noted)
    psi: np.float32 = np.float32(0.0)
    Ta_f64: float = 0.0                # python-float mirror of Ta[i]
    Ta: np.float32 = np.float32(0.0)
    Twater_f64: float = 0.0            # python float (np.mean mirror)
    radI: np.float32 = np.float32(0.0)
    radD: np.float32 = np.float32(0.0)
    radG: np.float32 = np.float32(0.0)
    altitude: np.float32 = np.float32(0.0)
    azimuth: np.float32 = np.float32(0.0)
    sinalt: np.float32 = np.float32(0.0)
    cosalt: np.float32 = np.float32(0.0)
    ta273_pow4: np.float32 = np.float32(0.0)      # (Ta + 273.15) ** 4 bit
    ks_sunlit: np.float32 = np.float32(0.0)       # Kside sunlit surface
    ks_shaded: np.float32 = np.float32(0.0)
    timestepdec_f64: float = 0.0
    CI_out: object = None               # recorded ret_CI (state threading)
    # frozen planes
    Tg_pre: np.ndarray = None           # PRE water-mutation ground temp
    Tg: np.ndarray = None               # post-mutation (= Solweig return)
    Tgwall: np.ndarray = None
    Lup_pre: np.ndarray = None          # sunonsurface entry Lup (j=0)
    gvflup_extra: np.ndarray = None     # line-277 term, post-mutation Tg
    Lwall: np.ndarray = None
    F_sh: np.ndarray = None             # post NaN->0.5
    azimuth_f64: float = 0.0            # dp guard scalar (f64 0-dim arg)
    altitude_f64: float = 0.0
    dp_veg_surface_f64: float = 0.0     # f64 0-dim chains (frozen bits)
    dp_shaded_surface_f64: float = 0.0
    dp_sunlit_surface_f64: float = 0.0
    night_Lup: np.ndarray = None
    night_water_override: np.float32 = np.float32(0.0)
    # frozen tables (P = 153)
    lv: np.ndarray = None               # (P, 3) f32
    patch_sin: np.ndarray = None        # sin(patch_altitude) (P,)
    patch_cos: np.ndarray = None        # cos(patch_altitude) (P,)
    steradian: np.ndarray = None        # (P,)
    lumChi: np.ndarray = None           # (P,)
    Lsky_down2: np.ndarray = None       # Lsky_down[:, 2] (P,)
    Lsky_side2: np.ndarray = None       # Lsky_side[:, 2] (P,)
    card_e: np.ndarray = None           # bool (P,) East membership
    card_s: np.ndarray = None
    card_w: np.ndarray = None
    card_n: np.ndarray = None
    card_cos_e: np.ndarray = None       # cos((90 - patch_azi) * d2r) (P,)
    card_cos_s: np.ndarray = None
    card_cos_w: np.ndarray = None
    card_cos_n: np.ndarray = None
    sunlit_packed: np.ndarray = None    # (rows, cols, ceil(P/8)) uint8
    shaded_packed: np.ndarray = None
    dp_guard_idx: np.ndarray = None     # guard-true patch indices (dp sos set)
    dp_sunlit_packed: np.ndarray = None  # same cube; dp bits at kside_n+k
    dp_shaded_packed: np.ndarray = None
    kside_n_patches: int = 0            # Kside's unguarded sos prefix length
    # walk tables (static but carried here for the kernel)
    walk_dy: np.ndarray = None          # (18, 11) int
    walk_dx: np.ndarray = None
    az_statics: dict = None             # per-j azimuth/azilow/azihigh/branch
    # TsWaveDelay frozen weights, one per call (6 daytime calls)
    tsw_weight1: list = field(default_factory=list)


@dataclass
class RadLoopState:
    """State threaded through the time loop (utci_process wiring)."""
    firstdaytime: int
    timeadd: float                      # python f64
    CI: np.float32
    Tgmap1: np.ndarray
    Tgmap1E: np.ndarray
    Tgmap1S: np.ndarray
    Tgmap1W: np.ndarray
    Tgmap1N: np.ndarray
    TgOut1: np.ndarray


# ---------------------------------------------------------------------------
# Hoisted mirrors (numpy, verbatim expression order)
# ---------------------------------------------------------------------------

def mirror_tsw_delay(Tgmap0, w1: np.float32, firstdaytime: int,
                     timeadd: float, timestepdec: float, Tgmap1):
    """TsWaveDelay_2015a mirror. w1 = frozen weight bit for THIS call."""
    if firstdaytime == 1:
        Tgmap1 = Tgmap0
    one_minus_w = np.float32(1.0) - w1
    if timeadd >= (59 / 1440):
        new_map = Tgmap0 * one_minus_w + Tgmap1 * w1
        Lup = new_map
        Tgmap1_out = new_map
        timeadd_out = timestepdec if timestepdec > (59 / 1440) else 0
    else:
        timeadd_out = timeadd + timestepdec
        Lup = Tgmap0 * one_minus_w + Tgmap1 * w1
        Tgmap1_out = Tgmap1
    return Lup, timeadd_out, Tgmap1_out


def mirror_sunonsurface(st: RadiationStatic, t_in: RadTimestepInputs,
                        j: int, sunwall: np.ndarray, shadow: np.ndarray,
                        Lup_pre: np.ndarray, Lwall: np.ndarray,
                        first_val: int = 1, second_val: float = 11.0):
    """One azimuth's sunonsurface_2018a walk (verbatim slice semantics).

    Border behavior: the shifted-slice writes leave a border of each temp
    plane holding the PREVIOUS step's values (zeros initially); the walk
    consumes those stale values -- replicated exactly by allocating the
    temp planes once per call and slice-assigning per step, as the source
    does.
    """
    rows, cols = st.rows, st.cols
    buildings, walls = st.buildings, st.walls
    alb_grid, emis_grid = st.alb_grid, st.emis_grid
    albshadow = alb_grid * shadow
    alb = alb_grid
    aspect = st.aspect_rad
    wallbol = st.wallbol
    az = t_in.az_statics

    sizex, sizey = walls.shape
    tempsh = np.zeros((sizex, sizey), dtype=np.float32)
    tempbu = np.zeros((sizex, sizey), dtype=np.float32)
    tempbub = np.zeros((sizex, sizey), dtype=np.float32)
    tempbubwall = np.zeros((sizex, sizey), dtype=np.float32)
    tempwallsun = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumsh = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumwall = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumLupsh = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumLwall = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumalbsh = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumalbwall = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumalbnosh = np.zeros((sizex, sizey), dtype=np.float32)
    weightsumalbwallnosh = np.zeros((sizex, sizey), dtype=np.float32)
    tempLupsh = np.zeros((sizex, sizey), dtype=np.float32)
    tempalbsh = np.zeros((sizex, sizey), dtype=np.float32)
    tempalbnosh = np.zeros((sizex, sizey), dtype=np.float32)

    f = buildings
    albedo_b32 = np.float32(0.2)
    second_i = int(round(second_val))

    for n in range(second_i):
        dy = t_in.walk_dy[j, n]
        dx = t_in.walk_dx[j, n]
        absdx = abs(dx)
        absdy = abs(dy)
        xc1 = int((dx + absdx) / 2)
        xc2 = int(sizex + (dx - absdx) / 2)
        yc1 = int((dy + absdy) / 2)
        yc2 = int(sizey + (dy - absdy) / 2)
        xp1 = int(-((dx - absdx) / 2))
        xp2 = int(sizex - (dx + absdx) / 2)
        yp1 = int(-((dy - absdy) / 2))
        yp2 = int(sizey - (dy + absdy) / 2)

        tempbu[xp1:xp2, yp1:yp2] = buildings[xc1:xc2, yc1:yc2]
        tempsh[xp1:xp2, yp1:yp2] = shadow[xc1:xc2, yc1:yc2]
        tempLupsh[xp1:xp2, yp1:yp2] = Lup_pre[xc1:xc2, yc1:yc2]
        tempalbsh[xp1:xp2, yp1:yp2] = albshadow[xc1:xc2, yc1:yc2]
        tempalbnosh[xp1:xp2, yp1:yp2] = alb[xc1:xc2, yc1:yc2]
        f = np.minimum(f, tempbu)

        shadow2 = tempsh * f
        weightsumsh += shadow2

        Lupsh = tempLupsh * f
        weightsumLupsh += Lupsh

        albsh = tempalbsh * f
        weightsumalbsh += albsh

        albnosh = tempalbnosh * f
        weightsumalbnosh += albnosh

        tempwallsun[xp1:xp2, yp1:yp2] = sunwall[xc1:xc2, yc1:yc2]
        tempb = tempwallsun * f
        tempbwall = f * np.float32(-1.0) + np.float32(1.0)
        tempbub = ((tempb + tempbub) > 0).astype(np.float32)
        tempbubwall = ((tempbwall + tempbubwall) > 0).astype(np.float32)
        weightsumLwall += tempbub * Lwall
        weightsumalbwall += tempbub * albedo_b32
        weightsumwall += tempbub
        weightsumalbwallnosh += tempbubwall * albedo_b32

        ind = 1
        if (n + 1) <= first_val:
            weightsumwall_first = weightsumwall / np.float32(ind)
            weightsumsh_first = weightsumsh / np.float32(ind)
            wallsuninfluence_first = weightsumwall_first > 0
            weightsumLwall_first = weightsumLwall / np.float32(ind)
            weightsumLupsh_first = weightsumLupsh / np.float32(ind)
            weightsumalbwall_first = weightsumalbwall / np.float32(ind)
            weightsumalbsh_first = weightsumalbsh / np.float32(ind)
            weightsumalbwallnosh_first = weightsumalbwallnosh / np.float32(ind)
            weightsumalbnosh_first = weightsumalbnosh / np.float32(ind)
            wallinfluence_first = weightsumalbwallnosh_first > 0
            ind += 1

    wallsuninfluence_second = weightsumwall > 0
    wallinfluence_second = weightsumalbwallnosh > 0

    azimuth = az["azimuth"][j]
    azilow = az["azilow"][j]
    azihigh = az["azihigh"][j]
    branch = int(az["branch"][j])
    if branch == 1:
        facesh = (
            ((aspect < azilow) | (aspect >= azihigh)).astype(np.float32)
            - wallbol + np.float32(1.0)
        )
    else:
        facesh = (
            ((aspect > azilow) | (aspect <= azihigh)).astype(np.float32)
            * np.float32(-1.0) + np.float32(1.0)
        )

    second32 = np.float32(second_val)
    keep = (weightsumwall == second32).astype(np.float32) - facesh
    keep[keep == np.float32(-1.0)] = np.float32(0.0)

    first_p1 = np.float32(first_val + 1)
    first32 = np.float32(first_val)
    second_p1 = np.float32(second_val + 1)

    wif = wallsuninfluence_first.astype(np.float32)
    wis = wallsuninfluence_second.astype(np.float32)
    lif = wallinfluence_first.astype(np.float32)
    lis_ = wallinfluence_second.astype(np.float32)

    gvf1 = ((weightsumwall_first + weightsumsh_first) / first_p1) * wif + \
        (weightsumsh_first) / first32 * (wif * np.float32(-1.0) + np.float32(1.0))
    weightsumwall = weightsumwall.copy()
    weightsumwall[keep == np.float32(1.0)] = np.float32(0.0)
    gvf2 = ((weightsumwall + weightsumsh) / second_p1) * wis + \
        (weightsumsh) / second32 * (wis * np.float32(-1.0) + np.float32(1.0))
    gvf2 = np.minimum(gvf2, np.float32(1.0))
    gvf2[gvf2 > np.float32(1.0)] = np.float32(1.0)

    gvfLup1 = ((weightsumLwall_first + weightsumLupsh_first) / first_p1) * wif + \
        (weightsumLupsh_first) / first32 * (wif * np.float32(-1.0) + np.float32(1.0))
    weightsumLwall = weightsumLwall.copy()
    weightsumLwall[keep == np.float32(1.0)] = np.float32(0.0)
    gvfLup2 = ((weightsumLwall + weightsumLupsh) / second_p1) * wis + \
        (weightsumLupsh) / second32 * (wis * np.float32(-1.0) + np.float32(1.0))

    gvfalb1 = ((weightsumalbwall_first + weightsumalbsh_first) / first_p1) * wif + \
        (weightsumalbsh_first) / first32 * (wif * np.float32(-1.0) + np.float32(1.0))
    weightsumalbwall = weightsumalbwall.copy()
    weightsumalbwall[keep == np.float32(1.0)] = np.float32(0.0)
    gvfalb2 = ((weightsumalbwall + weightsumalbsh) / second_p1) * wis + \
        (weightsumalbsh) / second32 * (wis * np.float32(-1.0) + np.float32(1.0))

    gvfalbnosh1 = ((weightsumalbwallnosh_first + weightsumalbnosh_first) / first_p1) * lif + \
        (weightsumalbnosh_first) / first32 * (lif * np.float32(-1.0) + np.float32(1.0))
    # NOTE: gvfalbnosh2 divides by `second` (NOT second+1) -- source line
    # kept verbatim; this asymmetry is load-bearing.
    gvfalbnosh2 = ((weightsumalbwallnosh + weightsumalbnosh) / second32) * lis_ + \
        (weightsumalbnosh) / second32 * (lis_ * np.float32(-1.0) + np.float32(1.0))

    gvf = (gvf1 * np.float32(0.5) + gvf2 * np.float32(0.4)) / np.float32(0.9)
    gvfLup = (gvfLup1 * np.float32(0.5) + gvfLup2 * np.float32(0.4)) / np.float32(0.9)
    gvfLup = gvfLup + t_in.gvflup_extra
    gvfalb = (gvfalb1 * np.float32(0.5) + gvfalb2 * np.float32(0.4)) / np.float32(0.9)
    gvfalb = gvfalb + alb_grid * (buildings * np.float32(-1.0) + np.float32(1.0)) * shadow
    gvfalbnosh = (gvfalbnosh1 * np.float32(0.5) + gvfalbnosh2 * np.float32(0.4)) / np.float32(0.9)
    gvfalbnosh = gvfalbnosh * buildings + \
        alb_grid * (buildings * np.float32(-1.0) + np.float32(1.0))

    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2


def mirror_gvf(st: RadiationStatic, t_in: RadTimestepInputs):
    """gvf_2018a mirror: 18 azimuths + cardinal sums (verbatim order)."""
    walls = st.walls
    buildings = st.buildings
    wallsun = t_in.wallsun
    shadow = t_in.shadow
    rows, cols = st.rows, st.cols

    sunwall = ((wallsun / walls * buildings) == 1).astype(np.float32)

    emis_grid = st.emis_grid
    acc = {k: np.zeros((rows, cols), dtype=np.float32) for k in
           ("gvfLup", "gvfalb", "gvfalbnosh", "gvfSum")}
    card_acc = {k: {c: np.zeros((rows, cols), dtype=np.float32)
                    for c in ("Lup", "alb", "albnosh")}
                for k in ("E", "S", "W", "N")}
    azimuthA_deg = np.arange(5, 359, 20, dtype=np.float32)

    for j in range(18):
        _, gvfLupi, gvfalbi, gvfalbnoshi, gvf2 = mirror_sunonsurface(
            st, t_in, j, sunwall, shadow, t_in.Lup_pre, t_in.Lwall
        )
        acc["gvfLup"] += gvfLupi
        acc["gvfalb"] += gvfalbi
        acc["gvfalbnosh"] += gvfalbnoshi
        acc["gvfSum"] += gvf2

        az = float(azimuthA_deg[j])
        if 0 <= az < 180:
            card_acc["E"]["Lup"] += gvfLupi
            card_acc["E"]["alb"] += gvfalbi
            card_acc["E"]["albnosh"] += gvfalbnoshi
        if 90 <= az < 270:
            card_acc["S"]["Lup"] += gvfLupi
            card_acc["S"]["alb"] += gvfalbi
            card_acc["S"]["albnosh"] += gvfalbnoshi
        if 180 <= az < 360:
            card_acc["W"]["Lup"] += gvfLupi
            card_acc["W"]["alb"] += gvfalbi
            card_acc["W"]["albnosh"] += gvfalbnoshi
        if az >= 270 or az < 90:
            card_acc["N"]["Lup"] += gvfLupi
            card_acc["N"]["alb"] += gvfalbi
            card_acc["N"]["albnosh"] += gvfalbnoshi

    ta_pow4_term = (SBC32 * emis_grid) * t_in.ta273_pow4
    n18 = np.float32(18.0)
    n9 = np.float32(9.0)

    gvfLup = acc["gvfLup"] / n18 + ta_pow4_term
    gvfalb = acc["gvfalb"] / n18
    gvfalbnosh = acc["gvfalbnosh"] / n18
    out = {"gvfLup": gvfLup, "gvfalb": gvfalb, "gvfalbnosh": gvfalbnosh}
    for c in ("E", "S", "W", "N"):
        out[f"gvfLup{c}"] = card_acc[c]["Lup"] / n9 + ta_pow4_term
        out[f"gvfalb{c}"] = card_acc[c]["alb"] / n9
        out[f"gvfalbnosh{c}"] = card_acc[c]["albnosh"] / n9
    gvfNorm = acc["gvfSum"] / n18
    gvfNorm = gvfNorm.copy()
    gvfNorm[buildings == 0] = np.float32(1.0)
    out["gvfSum"] = acc["gvfSum"]
    out["gvfNorm"] = gvfNorm
    return out


def mirror_kup(st: RadiationStatic, t_in: RadTimestepInputs, gvf):
    """Kup_veg_2015a mirror (verbatim order)."""
    sinalt = t_in.sinalt
    radI = t_in.radI
    radD = t_in.radD
    radG = t_in.radG
    svfbuveg = st.svfbuveg
    albedo_b32 = np.float32(0.2)
    F_sh = t_in.F_sh
    one = np.float32(1.0)
    inner = (radD * svfbuveg
             + albedo_b32 * (one - svfbuveg)
             * (radG * (one - F_sh) + radD * F_sh))
    out = {}
    for name, albn, nosh in (
        ("Kup", gvf["gvfalb"], gvf["gvfalbnosh"]),
        ("KupE", gvf["gvfalbE"], gvf["gvfalbnoshE"]),
        ("KupS", gvf["gvfalbS"], gvf["gvfalbnoshS"]),
        ("KupW", gvf["gvfalbW"], gvf["gvfalbnoshW"]),
        ("KupN", gvf["gvfalbN"], gvf["gvfalbnoshN"]),
    ):
        out[name] = (albn * radI * sinalt) + inner * nosh
    return out


def _bit(packed: np.ndarray, i: int) -> np.ndarray:
    """Patch bit i of a packed cube -> bool plane."""
    byte = packed[:, :, i >> 3]
    bit = (byte >> np.uint8(i & 7)) & np.uint8(1)
    return bit.astype(bool)


def _diffsh_plane(st: "RadiationStatic", idx: int) -> np.ndarray:
    """diffsh patch plane computed from the PACKED cubes (T14 Change F).

    Bit-identical to slicing the retired dense cube: the source built
    ``sh.astype(f32) - (fl32(1.0) - veg.astype(f32)) * fl32(1.0-0.03)``
    once; per-patch operands are exactly 0.0/1.0 so the same op order on
    the same constant gives the same bits.
    """
    one = np.float32(1.0)
    sh_f = _bit(st.shmat_packed, idx).astype(np.float32)
    veg_f = _bit(st.vegshmat_packed, idx).astype(np.float32)
    return sh_f - (one - veg_f) * DIFFSH_C97


def dense_diffsh_from_packed(st: "RadiationStatic") -> np.ndarray:
    """Materialize the dense diffsh cube from the packed cubes.

    Runtime code NEVER calls this (it was the 146 MiB resident cube the
    RSS diet removed); it exists for consumers of the old dense layout —
    the CUDA host bundle — and is bit-equal to the pre-diet cube.
    """
    n_patches = st.n_patches
    if n_patches is None:
        n_patches = st.shmat_packed.shape[2] * 8  # unpadded capture
    out = np.empty((st.rows, st.cols, n_patches), dtype=np.float32)
    for idx in range(n_patches):
        out[:, :, idx] = _diffsh_plane(st, idx)
    return out


def mirror_kside(st: RadiationStatic, t_in: RadTimestepInputs, gvf, kup):
    """Kside_veg_v2022a cyl+anisotropic mirror (verbatim patch order)."""
    rows, cols = st.rows, st.cols
    shadow = t_in.shadow
    KsideI = (shadow * t_in.radI) * t_in.cosalt
    KsideD = np.zeros((rows, cols), dtype=np.float32)
    Kref_sun = np.zeros((rows, cols), dtype=np.float32)
    Kref_sh = np.zeros((rows, cols), dtype=np.float32)
    Kref_veg = np.zeros((rows, cols), dtype=np.float32)
    sunlit_surface = t_in.ks_sunlit
    shaded_surface = t_in.ks_shaded
    lumChi = t_in.lumChi
    steradian = t_in.steradian
    patch_cos = t_in.patch_cos
    veg_packed = st.vegshmat_packed
    vbsh_packed = st.vbshvegshmat_packed
    sh_packed = st.shmat_packed
    sunlit_packed = t_in.sunlit_packed
    shaded_packed = t_in.shaded_packed
    n_patches = lumChi.shape[0]

    for idx in range(n_patches):
        anglIncC = patch_cos[idx]  # cos(pa) * cos(0.) ; cos(0.) == 1 exact
        ds = _diffsh_plane(st, idx)
        KsideD += ((ds * lumChi[idx]) * anglIncC) * steradian[idx]

        temp_vegsh = (~_bit(veg_packed, idx)) | (~_bit(vbsh_packed, idx))
        Kref_veg += ((shaded_surface * temp_vegsh) * steradian[idx]) * patch_cos[idx]

        temp_sh = (~_bit(sh_packed, idx)) & _bit(vbsh_packed, idx)
        # mask_sun == ((1 - shmat) * vbshvegshmat) == 1  (static cubes)
        sunlit_patches = _bit(sunlit_packed, idx).astype(np.float32)
        shaded_patches = _bit(shaded_packed, idx).astype(np.float32)
        temp_sh_f = temp_sh.astype(np.float32)
        Kref_sun += ((((sunlit_surface * sunlit_patches) * temp_sh_f)
                      * steradian[idx]) * patch_cos[idx])
        Kref_sh += ((((shaded_surface * shaded_patches) * temp_sh_f)
                     * steradian[idx]) * patch_cos[idx])

    Kside = ((((KsideI + KsideD) + Kref_sun) + Kref_sh) + Kref_veg)
    half = np.float32(0.5)
    return {
        "Keast": kup["KupE"] * half,
        "Ksouth": kup["KupS"] * half,
        "Kwest": kup["KupW"] * half,
        "Knorth": kup["KupN"] * half,
        "KsideI": KsideI,
        "KsideD": KsideD,
        "Kside": Kside,
    }


def _f64_acc(acc32: np.ndarray, P: float, mask: np.ndarray) -> np.ndarray:
    """torch `f32 += f64-plane` mirror: sum in f64, round once per add.

    Probe evidence (t08/promotion): torch promotes the f32 accumulator
    and the f64 RHS to f64, adds, then stores fl32 -- NOT a cast of the
    RHS first. `P * mask` is exact in f64 (mask in {0,1}).
    """
    out = acc32.astype(np.float64)
    out += P * mask
    return out.astype(np.float32)


def mirror_define_patch(st: RadiationStatic, t_in: RadTimestepInputs,
                        Lup_tsw: np.ndarray, component: str = None):
    """define_patch_characteristics mirror (two-pass, verbatim order).

    Dtype contract (dp_in_*__dtype pins + t08 promotion probes):
      * solar_altitude/solar_azimuth arrive as f64 0-DIM tensors ->
        `azimuth_difference = abs(solar_azimuth - patch_azimuth[idx])`
        is an F64 0-dim value; the guard compares in f64.
      * vegetation/shaded/sunlit surfaces are F64 0-dim chains
        (ewall*SBC f32 0-dim, then 0-dim x 0-dim -> f64 pow/div).
        Every veg/sun/sh term multiplies them against f32 0-dims (full
        f64 promotion) and a BOOL mask plane (bool promotes the f64
        0-dim chain to an f64 PLANE); the f32 accumulators then sum in
        f64 and round once per add (_f64_acc). Cardinal factors ccos
        multiply the f64 plane BEFORE the accumulate.
      * sky and reflected terms stay all-f32 (f32 plane chains).

    Reflection barrier: reflected_on_surfaces needs this cell's completed
    first-pass Ldown_sky only -- the passes are kept in source order
    (sky/veg/sun+shaded cardinals interleaved per patch) and the barrier
    sits between the loops exactly as in the source.

    component: sky_masks-variant replay (oracle dp_component_calls): one
    of "sky" | "veg" | "sun" | "refl" keeps only that mask family active
    (all others zero -- bit-safe, accumulators are +0.0-only and every
    added term is nonnegative).
    """
    rows, cols = st.rows, st.cols
    zeros = lambda: np.zeros((rows, cols), dtype=np.float32)
    Ldown_sky, Ldown_veg, Ldown_sun, Ldown_sh, Ldown_ref = (
        zeros(), zeros(), zeros(), zeros(), zeros())
    Lside_sky, Lside_veg, Lside_sun, Lside_sh, Lside_ref = (
        zeros(), zeros(), zeros(), zeros(), zeros())
    Least, Lwest, Lnorth, Lsouth = zeros(), zeros(), zeros(), zeros()

    veg_f64 = float(t_in.dp_veg_surface_f64)
    shaded_f64 = float(t_in.dp_shaded_surface_f64)
    sunlit_f64 = float(t_in.dp_sunlit_surface_f64)

    steradian = t_in.steradian
    patch_sin = t_in.patch_sin
    patch_cos = t_in.patch_cos
    Lsky_down2 = t_in.Lsky_down2
    Lsky_side2 = t_in.Lsky_side2
    # cardinal accumulators behind a mutable dict: _f64_acc RETURNS a new
    # array, so rebinding a tuple slot would drop the update
    accs = {"e": Least, "s": Lsouth, "w": Lwest, "n": Lnorth}
    card = (("e", t_in.card_e, t_in.card_cos_e),
            ("s", t_in.card_s, t_in.card_cos_s),
            ("w", t_in.card_w, t_in.card_cos_w),
            ("n", t_in.card_n, t_in.card_cos_n))

    sh_packed = st.shmat_packed
    veg_packed = st.vegshmat_packed
    vbsh_packed = st.vbshvegshmat_packed
    n_patches = steradian.shape[0]

    # guard outcome per patch: f64 0-dim comparisons, verbatim
    # `abs(solar_azimuth - patch_azimuth[idx]) > 90 and < 270 and
    #  solar_altitude > 0` (f64 - f32 0-dim -> f64)
    patch_azimuth = t_in.lv[:, 1]
    azdiff = np.abs(np.float64(t_in.azimuth_f64)
                    - patch_azimuth.astype(np.float64))
    guard = (azdiff > 90.0) & (azdiff < 270.0) & (t_in.altitude_f64 > 0.0)

    no_mask = np.zeros((rows, cols), dtype=bool)
    # NOTE: the oracle's F-mask sub-calls keep the real guard; variants
    # other than "sun" zero the guarded terms because their temp_sh mask
    # is F, not because the guard changes.

    for idx in range(n_patches):
        sh_b = _bit(sh_packed, idx)
        veg_b = _bit(veg_packed, idx)
        vbsh_b = _bit(vbsh_packed, idx)
        mask_sky_b = sh_b & veg_b
        mask_veg_b = (~veg_b) | (~vbsh_b)
        mask_sun_b = (~sh_b) & vbsh_b
        mask_refl_b = (~sh_b) | (~veg_b) | (~vbsh_b)
        if component == "sky":
            temp_sky = mask_sky_b
            temp_vegsh = no_mask
            temp_sh = no_mask
            refl_active = False
        elif component == "veg":
            temp_sky = no_mask
            temp_vegsh = mask_veg_b
            temp_sh = no_mask
            refl_active = False
        elif component == "sun":
            temp_sky = no_mask
            temp_vegsh = no_mask
            temp_sh = mask_sun_b
            refl_active = False
        elif component == "refl":
            temp_sky = no_mask
            temp_vegsh = no_mask
            temp_sh = no_mask
            refl_active = True
        else:
            temp_sky = mask_sky_b
            temp_vegsh = mask_veg_b
            temp_sh = mask_sun_b
            refl_active = True
        temp_sky_f = temp_sky.astype(np.float32)
        Ldown_sky += temp_sky_f * Lsky_down2[idx]
        Lside_sky += temp_sky_f * Lsky_side2[idx]

        # f64 0-dim chains: (surface * steradian[idx]) * trig[idx]
        st64 = np.float64(steradian[idx])
        pcos64 = np.float64(patch_cos[idx])
        psin64 = np.float64(patch_sin[idx])
        P_veg_c = (veg_f64 * st64) * pcos64
        P_veg_s = (veg_f64 * st64) * psin64
        Lside_veg = _f64_acc(Lside_veg, P_veg_c, temp_vegsh)
        Ldown_veg = _f64_acc(Ldown_veg, P_veg_s, temp_vegsh)

        for tag, memb, ccos in card:
            if memb[idx]:
                cc64 = np.float64(ccos[idx])
                accs[tag] += (temp_sky_f * Lsky_side2[idx]) * ccos[idx]
                accs[tag] = _f64_acc(accs[tag], P_veg_c * cc64, temp_vegsh)

        temp_sh = (~sh_b) & vbsh_b          # == mask_sun (static cubes)
        P_sh_c = (shaded_f64 * st64) * pcos64
        P_sh_s = (shaded_f64 * st64) * psin64
        P_sun_c = (sunlit_f64 * st64) * pcos64
        P_sun_s = (sunlit_f64 * st64) * psin64
        if guard[idx]:
            # dp's own shaded_or_sunlit set: call-order offset = Kside's
            # unguarded prefix + this patch's rank among guard-true idx
            k = int(t_in.kside_n_patches
                    + np.searchsorted(t_in.dp_guard_idx, idx))
            sunlit_patches = _bit(t_in.dp_sunlit_packed, k)
            shaded_patches = _bit(t_in.dp_shaded_packed, k)
            gate = sunlit_patches & temp_sh
            Lside_sun = _f64_acc(Lside_sun, P_sun_c, gate)
            Ldown_sun = _f64_acc(Ldown_sun, P_sun_s, gate)
            gate_sh = shaded_patches & temp_sh
            Lside_sh = _f64_acc(Lside_sh, P_sh_c, gate_sh)
            Ldown_sh = _f64_acc(Ldown_sh, P_sh_s, gate_sh)
            for tag, memb, ccos in card:
                if memb[idx]:
                    cc64 = np.float64(ccos[idx])
                    accs[tag] = _f64_acc(accs[tag], P_sun_c * cc64, gate)
                    accs[tag] = _f64_acc(accs[tag], P_sh_c * cc64, gate_sh)
        else:
            Lside_sh = _f64_acc(Lside_sh, P_sh_c, temp_sh)
            Ldown_sh = _f64_acc(Ldown_sh, P_sh_s, temp_sh)
            for tag, memb, ccos in card:
                if memb[idx]:
                    cc64 = np.float64(ccos[idx])
                    accs[tag] = _f64_acc(accs[tag], P_sh_c * cc64, temp_sh)

    # REFLECTION BARRIER (source: after loop 1, before loop 2)
    # ewall is RE-TENSORIZED at function top (torch.tensor(ewall)), so
    # (1 - ewall) is an f32 tensor subtract: fl32(1)-fl32(0.9) = 0x3DCCCCCD
    one_minus_ewall = np.float32(1.0) - np.float32(0.9)
    reflected = (((Ldown_sky + Lup_tsw) * one_minus_ewall)
                 * np.float32(0.5)) / PI32

    for idx in range(n_patches):
        sh_b = _bit(sh_packed, idx)
        veg_b = _bit(veg_packed, idx)
        vbsh_b = _bit(vbsh_packed, idx)
        if component == "refl":
            temp_sh = (~sh_b) | (~veg_b) | (~vbsh_b)
        elif component is not None:
            temp_sh = no_mask
        else:
            temp_sh = (~sh_b) | (~veg_b) | (~vbsh_b)
        temp_sh_f = temp_sh.astype(np.float32)
        Lside_ref += (((reflected * steradian[idx])
                       * patch_cos[idx]) * temp_sh_f)
        Ldown_ref += (((reflected * steradian[idx])
                       * patch_sin[idx]) * temp_sh_f)
        for tag, memb, ccos in card:
            if memb[idx]:
                accs[tag] += ((((reflected * steradian[idx])
                                * patch_cos[idx]) * temp_sh_f) * ccos[idx])

    Least, Lsouth, Lwest, Lnorth = (
        accs["e"], accs["s"], accs["w"], accs["n"])

    Lside = ((((Lside_sky + Lside_veg) + Lside_sh) + Lside_sun) + Lside_ref)
    Ldown = ((((Ldown_sky + Ldown_veg) + Ldown_sh) + Ldown_sun) + Ldown_ref)
    return {
        "Ldown": Ldown, "Lside": Lside,
        "Lside_sky": Lside_sky, "Lside_veg": Lside_veg,
        "Lside_sh": Lside_sh, "Lside_sun": Lside_sun,
        "Lside_ref": Lside_ref,
        "Least": Least, "Lwest": Lwest, "Lnorth": Lnorth, "Lsouth": Lsouth,
    }


def mirror_lside_veg(lup_family):
    """Lside_veg_v2022a anisotropic branch (trivial; verbatim 0.5)."""
    half = np.float32(0.5)
    return (lup_family["LupE"] * half, lup_family["LupS"] * half,
            lup_family["LupW"] * half, lup_family["LupN"] * half)


def mirror_radiation_timestep(st: RadiationStatic, t_in: RadTimestepInputs,
                              state: RadLoopState):
    """One Solweig_2022a_calc radiation step (day or night), verbatim order.

    Returns (outputs dict keyed by the Solweig return names, next state).
    """
    rows, cols = st.rows, st.cols
    out: dict = {}
    fd_in = state.firstdaytime
    ta_in = state.timeadd
    tsd = t_in.timestepdec_f64

    if t_in.is_day:
        # dRad (aniLum * radD)
        aniLum = np.zeros((rows, cols), dtype=np.float32)
        lv2 = t_in.lv[:, 2]
        for idx in range(lv2.shape[0]):
            aniLum += _diffsh_plane(st, idx) * lv2[idx]
        dRad = aniLum * t_in.radD

        gvf = mirror_gvf(st, t_in)

        # TsWaveDelay x5 (entry timeadd; returned timeadd discarded)
        tsw_out = {}
        for key, src in (("Lup", gvf["gvfLup"]), ("E", gvf["gvfLupE"]),
                         ("S", gvf["gvfLupS"]), ("W", gvf["gvfLupW"]),
                         ("N", gvf["gvfLupN"])):
            map_in = getattr(state, "Tgmap1" + ("" if key == "Lup" else key))
            w1 = t_in.tsw_weight1[{"Lup": 0, "E": 1, "S": 2, "W": 3,
                                   "N": 4}[key]]
            LupX, _ta_unused, map_out = mirror_tsw_delay(
                src, w1, fd_in, ta_in, tsd, map_in)
            tsw_out[key] = LupX
            tsw_out["map_" + key] = map_out

        # TgOut (timeadd threaded HERE only)
        TgTemp = t_in.Tg * t_in.shadow + t_in.Ta
        TgOut, ta_out, TgOut1_new = mirror_tsw_delay(
            TgTemp, t_in.tsw_weight1[5], fd_in, ta_in, tsd, state.TgOut1)

        kup = mirror_kup(st, t_in, gvf)
        kside = mirror_kside(st, t_in, gvf, kup)

        # Kdown
        one = np.float32(1.0)
        albedo_b32 = np.float32(0.2)
        Kdown = (((t_in.radI * t_in.shadow) * t_in.sinalt) + dRad
                 + (albedo_b32 * (one - st.svfbuveg))
                 * (t_in.radG * (one - t_in.F_sh) + t_in.radD * t_in.F_sh))

        # Ldown section
        dp = mirror_define_patch(st, t_in, tsw_out["Lup"])
        Least, Lsouth, Lwest, Lnorth = mirror_lside_veg({
            "LupE": tsw_out["E"], "LupS": tsw_out["S"],
            "LupW": tsw_out["W"], "LupN": tsw_out["N"],
        })

        # Sstr / Tmrt (cyl == 1, anisotropic == 1)
        absK, absL = 0.7, 0.95
        Fcyl, Fup, Fside = 0.28, 0.06, 0.22
        short = (kside["Kside"] * np.float32(Fcyl)
                 + (Kdown + kup["Kup"]) * np.float32(Fup)
                 + (((kside["Knorth"] + kside["Keast"])
                     + kside["Ksouth"]) + kside["Kwest"])
                 * np.float32(Fside))
        long_ = (((dp["Ldown"] + tsw_out["Lup"]) * np.float32(Fup)
                  + dp["Lside"] * np.float32(Fcyl)
                  + (((Lnorth + Least) + Lsouth) + Lwest)
                  * np.float32(Fside)))
        Sstr = short * np.float32(absK) + long_ * np.float32(absL)
        Tmrt = np.sqrt(np.sqrt(Sstr / (np.float32(absL) * SBC32))) \
            - np.float32(273.2)

        # POI cardinals += Least_ etc (cyl & aniso)
        Least = Least + dp["Least"]
        Lsouth = Lsouth + dp["Lsouth"]
        Lwest = Lwest + dp["Lwest"]
        Lnorth = Lnorth + dp["Lnorth"]

        out.update({
            "Tmrt": Tmrt, "Kdown": Kdown, "Kup": kup["Kup"],
            "Ldown": dp["Ldown"], "Lup": tsw_out["Lup"], "Tg": t_in.Tg,
            "shadow": t_in.shadow, "Keast": kside["Keast"],
            "Ksouth": kside["Ksouth"], "Kwest": kside["Kwest"],
            "Knorth": kside["Knorth"], "Least": Least, "Lsouth": Lsouth,
            "Lwest": Lwest, "Lnorth": Lnorth,
            "KsideI": kside["KsideI"], "TgOut": TgOut,
            "Lside": dp["Lside"], "CI_Tg": None, "CI_TgG": None,
            "KsideD": kside["KsideD"], "dRad": dRad, "Kside": kside["Kside"],
        })
        next_state = RadLoopState(
            firstdaytime=0, timeadd=ta_out,
            CI=(state.CI if t_in.CI_out is None
                else np.float32(t_in.CI_out)),
            Tgmap1=tsw_out["map_Lup"], Tgmap1E=tsw_out["map_E"],
            Tgmap1S=tsw_out["map_S"], Tgmap1W=tsw_out["map_W"],
            Tgmap1N=tsw_out["map_N"], TgOut1=TgOut1_new,
        )
        return out, next_state

    # ---------------- nighttime ----------------
    Lup = t_in.night_Lup
    TgOut = t_in.Ta + np.zeros((rows, cols), dtype=np.float32)
    dp = mirror_define_patch(st, t_in, Lup)
    Least, Lsouth, Lwest, Lnorth = mirror_lside_veg(
        {"LupE": Lup, "LupS": Lup, "LupW": Lup, "LupN": Lup})
    absK, absL = 0.7, 0.95
    Fcyl, Fup, Fside = 0.28, 0.06, 0.22
    zero = np.zeros((rows, cols), dtype=np.float32)
    Kdown = zero.copy()
    kupK = zero.copy()
    KsideI = zero.copy()
    KsideD = zero.copy()
    Kside_v = zero.copy()
    Knorth, Keast, Ksouth, Kwest = zero.copy(), zero.copy(), zero.copy(), zero.copy()
    short = ((Kside_v * np.float32(Fcyl)
              + (Kdown + kupK) * np.float32(Fup)
              + (((Knorth + Keast) + Ksouth) + Kwest) * np.float32(Fside)))
    long_ = (((dp["Ldown"] + Lup) * np.float32(Fup)
              + dp["Lside"] * np.float32(Fcyl)
              + (((Lnorth + Least) + Lsouth) + Lwest) * np.float32(Fside)))
    Sstr = short * np.float32(absK) + long_ * np.float32(absL)
    Tmrt = np.sqrt(np.sqrt(Sstr / (np.float32(absL) * SBC32))) \
        - np.float32(273.2)
    Least = Least + dp["Least"]
    Lsouth = Lsouth + dp["Lsouth"]
    Lwest = Lwest + dp["Lwest"]
    Lnorth = Lnorth + dp["Lnorth"]
    out.update({
        "Tmrt": Tmrt, "Kdown": Kdown, "Kup": kupK, "Ldown": dp["Ldown"],
        "Lup": Lup, "Tg": zero, "shadow": zero.copy(),
        "Keast": Keast, "Ksouth": Ksouth, "Kwest": Kwest, "Knorth": Knorth,
        "Least": Least, "Lsouth": Lsouth, "Lwest": Lwest, "Lnorth": Lnorth,
        "KsideI": KsideI, "TgOut": TgOut, "Lside": dp["Lside"],
        "CI_Tg": state.CI, "CI_TgG": state.CI, "KsideD": KsideD,
        "dRad": zero.copy(), "Kside": Kside_v,
    })
    # Night leaves the Tgmap1 family and TgOut1 UNCHANGED (source never
    # touches them in the nocturnal branch); only firstdaytime/timeadd
    # reset (both to 1 / 0).
    next_state = RadLoopState(
        firstdaytime=1, timeadd=0.0, CI=state.CI,
        Tgmap1=state.Tgmap1, Tgmap1E=state.Tgmap1E,
        Tgmap1S=state.Tgmap1S, Tgmap1W=state.Tgmap1W,
        Tgmap1N=state.Tgmap1N, TgOut1=state.TgOut1,
    )
    return out, next_state


# ---------------------------------------------------------------------------
# Capture loaders (tests / bench; plain npz reads, no torch)
# ---------------------------------------------------------------------------

def _f32(x) -> np.float32:
    return np.float32(x)


def _unpack_bits(data: np.ndarray) -> np.ndarray:
    """(rows, cols, nbytes) uint8 -> bool cube, little bit order."""
    return np.unpackbits(data, axis=2, bitorder="little").view(np.bool_)


def rad_static_from_capture(cap_dir: Path, *,
                            dense_diffsh: bool = False) -> RadiationStatic:
    """Load the runtime static planes.

    Default (T14 Change F RSS diet): packed cubes only, ``diffsh=None``;
    kernels/mirrors recompute diffsh per cell from the bits — bit-equal,
    ~590 MiB cheaper at site_500. ``dense_diffsh=True`` materializes the
    legacy dense cube for consumers of the old layout (CUDA host bundle).
    """
    z = np.load(Path(cap_dir) / "static.npz")
    n_patches = int(z["n_patches"]) if "n_patches" in z else 153
    shmat_packed = z["shmat_packed"]
    rows, cols = shmat_packed.shape[0], shmat_packed.shape[1]
    diffsh = None
    if dense_diffsh:
        # legacy layout: unpack once, subtract in the source op order
        sh = _unpack_bits(shmat_packed)[:, :, :n_patches]
        veg = _unpack_bits(z["vegshmat_packed"])[:, :, :n_patches]
        one = np.float32(1.0)
        dense = sh.astype(np.float32) - (one - veg.astype(np.float32)) \
            * DIFFSH_C97
        diffsh = np.ascontiguousarray(dense, dtype=np.float32)
        del sh, veg, dense
    return RadiationStatic(
        rows=rows, cols=cols,
        buildings=z["buildings"], walls=z["walls"],
        aspect_rad=z["aspect_rad"], wallbol=z["wallbol"],
        lc_grid=z["lc_grid"] if "lc_grid" in z else None,
        alb_grid=z["alb_grid"],
        emis_grid=z["emis_grid"], TgK=z["TgK"], Tstart=z["Tstart"],
        TgK_wall=z["TgK_wall"], Tstart_wall=z["Tstart_wall"],
        TmaxLST=z["TmaxLST"], TmaxLST_wall=z["TmaxLST_wall"],
        svfbuveg=z["svfbuveg"],
        shmat_packed=shmat_packed,
        vegshmat_packed=z["vegshmat_packed"],
        vbshvegshmat_packed=z["vbshvegshmat_packed"],
        n_patches=n_patches,
        diffsh=diffsh,
    )


def rad_bundle_from_capture(cap_dir: Path, t: int) -> RadTimestepInputs:
    cap_dir = Path(cap_dir)
    z = np.load(cap_dir / f"t{t:02d}.npz")
    walk = np.load(cap_dir / "walk_offsets.npz")
    azs = np.load(cap_dir / "azimuth_statics.npz")

    def scalar(name, default=None):
        if f"{name}__0d" in z:
            return np.float32(z[f"{name}__0d"])
        if f"{name}__f64" in z:
            return np.float64(z[f"{name}__f64"])
        if f"{name}__int" in z:
            return int(z[f"{name}__int"])
        if f"{name}__bool" in z:
            return int(z[f"{name}__bool"])
        if name in z:
            return z[name]
        return default

    alt = scalar("arg_altitude")
    alt_f = float(np.float32(alt))
    is_day = alt_f > 0.0

    lv = z["lv"].astype(np.float32) if "lv" in z else None
    patch_alt = lv[:, 0] if lv is not None else None

    # cardinal membership + cos tables: frozen capture (needs lv azimuths);
    # derived tables are stored by the producer when present, else derived
    # here with CLEAN comparisons (bool only; cos tables must be frozen).
    t_in = RadTimestepInputs(is_day=is_day)
    t_in.psi = _f32(scalar("arg_psi"))
    t_in.Ta = _f32(scalar("arg_Ta"))
    t_in.Ta_f64 = float(scalar("arg_Ta"))
    if "frozen_Twater_used_f64" in z:
        t_in.Twater_f64 = float(z["frozen_Twater_used_f64"])
    t_in.radI = _f32(scalar("arg_radI", np.float32(0)))
    t_in.radD = _f32(scalar("arg_radD", np.float32(0)))
    t_in.radG = _f32(scalar("arg_radG"))
    t_in.altitude = _f32(scalar("arg_altitude"))
    t_in.azimuth = _f32(scalar("arg_azimuth"))
    t_in.sinalt = (_f32(z["frozen_sinalt_f64"]) if "frozen_sinalt_f64" in z
                   else _f32(z["frozen_sinalt"]))
    t_in.cosalt = _f32(z["frozen_cosalt"])
    t_in.ta273_pow4 = (
        _f32(z["frozen_Ta273_pow4_f64"]) if "frozen_Ta273_pow4_f64" in z
        else _f32(z["frozen_Ta273_pow4"]))
    # dp guard scalars: solar_azimuth/solar_altitude arrive as f64 0-dim
    # tensors; the guard compares in f64
    if "dp_in_solar_azimuth__0d" in z:
        t_in.azimuth_f64 = float(z["dp_in_solar_azimuth__0d"])
        t_in.altitude_f64 = float(z["dp_in_solar_altitude__0d"])
    else:
        t_in.azimuth_f64 = float(t_in.azimuth)
        t_in.altitude_f64 = float(t_in.altitude)
    t_in.ks_sunlit = _f32(z["frozen_ks_sunlit"]) if "frozen_ks_sunlit" in z \
        else np.float32(0.0)
    t_in.ks_shaded = _f32(z["frozen_ks_shaded"]) if "frozen_ks_shaded" in z \
        else np.float32(0.0)
    t_in.timestepdec_f64 = float(scalar("arg_timestepdec", 0.0))
    if "ret_CI__0d" in z:
        t_in.CI_out = np.float32(z["ret_CI__0d"])
    elif "ret_CI__f64" in z:
        t_in.CI_out = np.float32(z["ret_CI__f64"])
    if is_day and "kup_in_radI__0d" in z:
        # the body RECOMPUTES radI/radD via diffusefraction (arg_radI/
        # arg_radD stay the -999 onlyglobal sentinels at day); Kup/
        # Kdown/Kside consume the f32 0-dim body values
        t_in.radI = _f32(z["kup_in_radI__0d"])
        t_in.radD = _f32(z["kup_in_radD__0d"])

    if is_day:
        t_in.vegsh = z["march_vegsh"]
        t_in.sh = z["march_sh"]
        t_in.wallsun = z["march_wallsun"]
        one = np.float32(1.0)
        t_in.shadow = t_in.sh - (one - t_in.vegsh) * (one - t_in.psi)
        t_in.Tg_pre = z["sunon_in_Tg"]
        t_in.Tg = z["ret_Tg"]
        # Tgwall arrives as a 0-DIM tensor at this site (scalar wall-temp
        # model); plane form kept as fallback
        if "sunon_in_Tgwall" in z:
            t_in.Tgwall = z["sunon_in_Tgwall"]
        else:
            t_in.Tgwall = np.float32(z["sunon_in_Tgwall__0d"][()])
        t_in.Lup_pre = z["frozen_Lup_pre"]
        t_in.Lwall = z["frozen_Lwall"]
        t_in.gvflup_extra = z["frozen_gvflup_extra"]
        if "F_sh_fixed" in z:
            t_in.F_sh = z["F_sh_fixed"]
        else:
            # fallback: the Kup ARG snapshot (post NaN->0.5 plane)
            fsh = np.array(z["kup_in_F_sh"], dtype=np.float32).copy()
            fsh[np.isnan(fsh)] = np.float32(0.5)
            t_in.F_sh = fsh
        t_in.sunlit_packed = z["sunlit_packed"]
        t_in.shaded_packed = z["shaded_packed"]
        # The recorded sos cube holds ALL shaded_or_sunlit calls of the
        # timestep in call order: Kside's 153 unguarded calls (altitude
        # re-tensorized f32, line 636) THEN define_patch's guard-true
        # calls (solar_altitude f64 0-dim -- a DIFFERENT tan chain, and
        # the bits differ at knife-edge cells; t16 evidence). Each
        # consumer must read its own set: Kside mirror -> bits 0..152,
        # dp mirror -> bits 153..153+n_guard-1 keyed by the guard-true
        # patch indices (ascending, as the dp loop emits them).
        pz_all = z["dp_in_patch_azimuth"].astype(np.float32)
        azd = np.abs(np.float64(t_in.azimuth_f64)
                     - pz_all.astype(np.float64))
        g_mask = (azd > 90.0) & (azd < 270.0) & (t_in.altitude_f64 > 0.0)
        t_in.dp_guard_idx = np.nonzero(g_mask)[0].astype(np.int64)
        t_in.dp_sunlit_packed = t_in.sunlit_packed
        t_in.dp_shaded_packed = t_in.shaded_packed
        t_in.kside_n_patches = pz_all.shape[0]
        t_in.lv = lv
        # patch trig + tables from define_patch args (recorded planes)
        pa = z["dp_in_patch_altitude"].astype(np.float32)
        pz = z["dp_in_patch_azimuth"].astype(np.float32)
        ster = z["dp_in_steradian"].astype(np.float32)
        t_in.patch_sin = np.sin(pa * (np.pi / 180.0)).astype(np.float32) \
            if "frozen_patch_sin" not in z else z["frozen_patch_sin"]
        t_in.patch_cos = np.cos(pa * (np.pi / 180.0)).astype(np.float32) \
            if "frozen_patch_cos" not in z else z["frozen_patch_cos"]
        t_in.steradian = ster
        t_in.Lsky_down2 = z["dp_in_Lsky_down"][:, 2].astype(np.float32)
        t_in.Lsky_side2 = z["dp_in_Lsky_side"][:, 2].astype(np.float32)
        # lumChi from kside args? frozen in producer as frozen_lumChi
        t_in.lumChi = (z["frozen_lumChi"].astype(np.float32)
                       if "frozen_lumChi" in z else None)
        # cardinal membership: CLEAN mirror of the source comparisons on
        # the recorded patch azimuth bits (python-float vs f32 tensor
        # compares identically for these table values)
        t_in.card_e = (pz > 360) | (pz < 180)
        t_in.card_s = (pz > 90) & (pz < 270)
        t_in.card_w = (pz > 180) & (pz < 360)
        t_in.card_n = (pz > 270) | (pz < 90)
        d2r = np.float32(np.pi / 180.0)
        t_in.card_cos_e = np.cos((np.float32(90.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_e" not in z else z["frozen_card_cos_e"]
        t_in.card_cos_s = np.cos((np.float32(180.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_s" not in z else z["frozen_card_cos_s"]
        t_in.card_cos_w = np.cos((np.float32(270.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_w" not in z else z["frozen_card_cos_w"]
        t_in.card_cos_n = np.cos((np.float32(0.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_n" not in z else z["frozen_card_cos_n"]
        # dp surfaces are F64 0-dim chains (shaded == vegetation
        # bit-identically; sunlit always scalar: Tgwall is 0-dim here)
        t_in.dp_veg_surface_f64 = float(z["frozen_dp_veg_surface_f64"])
        t_in.dp_shaded_surface_f64 = float(z["frozen_dp_shaded_surface_f64"])
        t_in.dp_sunlit_surface_f64 = float(z["frozen_dp_sunlit_surface_f64"])
        t_in.tsw_weight1 = [
            _f32(z[f"tsw{i}_frozen_weight1"]) for i in range(6)
        ]
    else:
        t_in.night_Lup = z["frozen_night_Lup"]
        t_in.night_water_override = _f32(z["frozen_night_water_override"])
        t_in.lv = lv if lv is not None else _night_lv_from_dp(z)
        # night define_patch inputs
        pa = z["dp_in_patch_altitude"].astype(np.float32)
        pz = z["dp_in_patch_azimuth"].astype(np.float32)
        t_in.steradian = z["dp_in_steradian"].astype(np.float32)
        t_in.Lsky_down2 = z["dp_in_Lsky_down"][:, 2].astype(np.float32)
        t_in.Lsky_side2 = z["dp_in_Lsky_side"][:, 2].astype(np.float32)
        t_in.patch_sin = np.sin(pa * (np.pi / 180.0)).astype(np.float32) \
            if "frozen_patch_sin" not in z else z["frozen_patch_sin"]
        t_in.patch_cos = np.cos(pa * (np.pi / 180.0)).astype(np.float32) \
            if "frozen_patch_cos" not in z else z["frozen_patch_cos"]
        t_in.card_e = (pz > 360) | (pz < 180)
        t_in.card_s = (pz > 90) & (pz < 270)
        t_in.card_w = (pz > 180) & (pz < 360)
        t_in.card_n = (pz > 270) | (pz < 90)
        d2r = np.float32(np.pi / 180.0)
        t_in.card_cos_e = np.cos((np.float32(90.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_e" not in z else z["frozen_card_cos_e"]
        t_in.card_cos_s = np.cos((np.float32(180.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_s" not in z else z["frozen_card_cos_s"]
        t_in.card_cos_w = np.cos((np.float32(270.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_w" not in z else z["frozen_card_cos_w"]
        t_in.card_cos_n = np.cos((np.float32(0.0) - pz) * d2r).astype(np.float32) \
            if "frozen_card_cos_n" not in z else z["frozen_card_cos_n"]
        t_in.dp_veg_surface_f64 = float(z["frozen_dp_veg_surface_f64"])
        t_in.dp_shaded_surface_f64 = float(z["frozen_dp_shaded_surface_f64"])
        t_in.dp_sunlit_surface_f64 = float(z["frozen_dp_sunlit_surface_f64"])
        t_in.sunlit_packed = None
        t_in.shaded_packed = None

    t_in.walk_dy = walk["dy"]
    t_in.walk_dx = walk["dx"]
    t_in.az_statics = {k: azs[k] for k in
                       ("azimuth", "azilow", "azihigh", "branch")}
    return t_in


def _night_lv_from_dp(z) -> np.ndarray:
    """Night L_patches (altitude, azimuth, 0) from define_patch args."""
    pa = z["dp_in_patch_altitude"].astype(np.float32)
    pz = z["dp_in_patch_azimuth"].astype(np.float32)
    out = np.zeros((pa.shape[0], 3), dtype=np.float32)
    out[:, 0] = pa
    out[:, 1] = pz
    return out


def rad_state_from_capture(cap_dir: Path, t: int, prev_state=None,
                           rows: int = 0, cols: int = 0) -> RadLoopState:
    """Initial state for timestep t (entry state recorded in args)."""
    z = np.load(Path(cap_dir) / f"t{t:02d}.npz")
    if prev_state is None:
        if "arg_firstdaytime__int" in z:
            fd = int(z["arg_firstdaytime__int"][()])
        else:
            fd = int(z["arg_firstdaytime__f64"][()])
        ta = float(z["arg_timeadd__f64"][()]) if "arg_timeadd__f64" in z \
            else float(z["arg_timeadd__int"][()])
        if "arg_CI__0d" in z:
            CI = np.float32(z["arg_CI__0d"][()])
        elif "arg_CI__f64" in z:
            CI = np.float32(z["arg_CI__f64"][()])
        else:
            CI = np.float32(1.0)
        maps = {}
        for k in ("Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N"):
            key = f"arg_{k}"
            maps[k] = z[key] if key in z else np.zeros((rows, cols), np.float32)
        tgout1 = z["arg_TgOut1"] if "arg_TgOut1" in z else np.zeros((rows, cols), np.float32)
        return RadLoopState(
            firstdaytime=fd, timeadd=ta, CI=CI,
            Tgmap1=maps["Tgmap1"], Tgmap1E=maps["Tgmap1E"],
            Tgmap1S=maps["Tgmap1S"], Tgmap1W=maps["Tgmap1W"],
            Tgmap1N=maps["Tgmap1N"], TgOut1=tgout1,
        )
    return prev_state


# ---------------------------------------------------------------------------
# Fused cell-owned kernel (phase 2 -- DESIGN 9.3 ordered fusion)
# ---------------------------------------------------------------------------

def fused_radiation_timestep(st: RadiationStatic, t_in: RadTimestepInputs,
                             state: RadLoopState):
    """Cell-owned fused kernel: the whole radiation stage per cell, one
    Numba call, bit-identical to the mirror above (same stage order).

    The plane-slice walk becomes a per-cell recursion: at step n a temp
    plane holds src[r+dx, c+dy] inside the write window and the PREVIOUS
    step's value outside (zeros before step 0), exactly the stale-border
    semantics of the slice writes. define_patch's two passes fuse per
    cell with a cell-local reflection barrier (the reflected term needs
    only this cell's completed first-pass Ldown_sky). Every expression
    keeps the mirror's dtype context: f32 locals for plane chains,
    f64 locals + single rounding for the torch `f32 += f64-plane`
    accumulators, explicit np.float32 literals throughout.
    """
    rows, cols = st.rows, st.cols
    if not t_in.is_day:
        return _fused_night_host(st, t_in, state)
    return _fused_day_host(st, t_in, state)


# -- numba helpers (bit discipline: f32 locals, np.float32 literals) ------

@njit(cache=True, nogil=True, fastmath=False)
def _pbit(pb, r, c, i):
    # patch bit i of packed cube (rows, cols, nbytes) uint8, little order
    return ((pb[r, c, i >> 3] >> (i & 7)) & 1) != 0


@njit(cache=True, nogil=True, fastmath=False)
def _tsw_blend(src, w1, m_in):
    # TsWaveDelay blend: fl32(src*(1-w1)) + fl32(m_in*w1), rounded once
    omw = np.float32(1.0) - w1
    return src * omw + m_in * w1


@njit(cache=True, nogil=True, fastmath=False)
def _fused_day_kernel(
    rows, cols, n_patches, kside_n,
    buildings, aspect, wallbol, alb_grid, emis_grid, svfbuveg,
    sh_pb, veg_pb, vbsh_pb, sun_pb, shd_pb,
    dp_rank, guard_true,
    shadow, sunwall, albshadow, alb, Lup_pre, gvflup_extra,
    lv2, ster, psin, pcos, lumChi, lsky_d2, lsky_s2,
    card_e, card_s, card_w, card_n,
    ccos_e, ccos_s, ccos_w, ccos_n,
    ks_sun, ks_shd, radI, radD, radG, sinalt, cosalt,
    pv_c, pv_s, psh_c, psh_s, psun_c, psun_s,
    pvc_e, pvc_s, pvc_w, pvc_n,
    pshc_e, pshc_s, pshc_w, pshc_n,
    psunc_e, psunc_s, psunc_w, psunc_n,
    ta273, Lwall32,
    F_sh, Tg_plane, Ta32,
    walk_dy, walk_dx, wxp1, wxp2, wyp1, wyp2,
    az_branch, az_low, az_high,
    jE, jS, jW, jN,
    w1_0, w1_1, w1_2, w1_3, w1_4, w1_5,
    fd_eq_1, branch2,
    m_lup_in, m_e_in, m_s_in, m_w_in, m_n_in, m_tg_in,
    o_tmrt, o_kdown, o_kup, o_ldown, o_lup,
    o_ke, o_ks, o_kw, o_kn,
    o_le, o_ls, o_lw, o_ln, o_ksidei, o_tgout, o_lside,
    o_ksided, o_drad, o_kside,
    n_lup, n_e, n_s, n_w, n_n, n_tg,
):
    one = np.float32(1.0)
    half = np.float32(0.5)
    m_one = np.float32(-1.0)
    alb_b = np.float32(0.2)
    second32 = np.float32(11.0)
    first32 = np.float32(1.0)
    first_p1 = np.float32(2.0)
    second_p1 = np.float32(12.0)
    c05 = np.float32(0.5)
    c04 = np.float32(0.4)
    c09 = np.float32(0.9)
    f_cyl = np.float32(0.28)
    f_up = np.float32(0.06)
    f_side = np.float32(0.22)
    abs_k = np.float32(0.7)
    abs_l = np.float32(0.95)
    sbc = np.float32(5.67051e-8)
    pi32 = np.float32(np.pi)
    ome = one - np.float32(0.9)   # fl32(1.0) - fl32(0.9), torch subtract
    c97 = DIFFSH_C97             # diffsh diet: per-cell recompute (T14 F)
    n18 = np.float32(18.0)
    n9 = np.float32(9.0)
    r2732 = np.float32(273.2)

    for r in range(rows):
        for c in range(cols):
            # ---- dRad (aniLum over patches, ascending idx) ----
            ani = np.float32(0.0)
            for idx in range(n_patches):
                shf = one if _pbit(sh_pb, r, c, idx) else np.float32(0.0)
                vegf = one if _pbit(veg_pb, r, c, idx) else np.float32(0.0)
                ds = shf - (one - vegf) * c97
                ani += ds * lv2[idx]
            dRad = ani * radD
            o_drad[r, c] = dRad

            # ---- 18-azimuth sunonsurface walk (per-cell recursion) ----
            acc_lup = np.float32(0.0)
            acc_alb = np.float32(0.0)
            acc_nosh = np.float32(0.0)
            ce_l = np.float32(0.0); ce_a = np.float32(0.0); ce_n = np.float32(0.0)
            cs_l = np.float32(0.0); cs_a = np.float32(0.0); cs_n = np.float32(0.0)
            cw_l = np.float32(0.0); cw_a = np.float32(0.0); cw_n = np.float32(0.0)
            cn_l = np.float32(0.0); cn_a = np.float32(0.0); cn_n = np.float32(0.0)
            b_row = buildings[r, c]
            a_row = alb_grid[r, c]
            sh_rc = shadow[r, c]
            no_b_term = a_row * (b_row * m_one + one)

            for j in range(18):
                f = b_row
                tbu = np.float32(0.0)
                tsh_p = np.float32(0.0)
                tlsh = np.float32(0.0)
                talsh = np.float32(0.0)
                tanosh = np.float32(0.0)
                tws = np.float32(0.0)
                wsh = np.float32(0.0)
                wwall = np.float32(0.0)
                wlupsh = np.float32(0.0)
                wlwall = np.float32(0.0)
                walbsh = np.float32(0.0)
                walbwall = np.float32(0.0)
                walbnosh = np.float32(0.0)
                walbwnosh = np.float32(0.0)
                tbub = np.float32(0.0)
                tbubw = np.float32(0.0)
                # _first snapshots (n+1 <= first == 1 -> step 0 only)
                wsh_f = np.float32(0.0)
                wwall_f = np.float32(0.0)
                wlup_f = np.float32(0.0)
                wlwall_f = np.float32(0.0)
                walb_f = np.float32(0.0)
                walbw_f = np.float32(0.0)
                walbn_f = np.float32(0.0)
                walbwn_f = np.float32(0.0)

                for n in range(11):
                    if f == np.float32(0.0):
                        # T18 K2 dead-scan elision — f is the running
                        # buildings floor; once +0.0 it can never rise
                        # (tbu ∈ {0,1} >= 0). The skipped terms are each
                        # ``plane_value * 0.0`` = ±0.0 (finiteness fenced
                        # per t by _assert_walk_planes_finite), a bit-level
                        # no-op: the accumulators start at +0.0 and IEEE
                        # RN addition never yields -0.0 unless BOTH
                        # operands are -0.0, so x + (±0.0) preserves x.
                        # Latches: tempb = tws*f = +0.0 (tws ∈ {0,1}) so
                        # (tempb+tbub)>0 keeps tbub; tempbwall =
                        # f*m_one+one = +1.0 so (1.0+tbubw)>0 forces
                        # tbubw := one. tbu < f is false for tbu ∈ {0,1}.
                        tbubw = one
                    else:
                        dy = walk_dy[j, n]
                        dx = walk_dx[j, n]
                        if (wxp1[j, n] <= r) and (r < wxp2[j, n]) \
                                and (wyp1[j, n] <= c) and (c < wyp2[j, n]):
                            # T18 K3: (j,n)-invariant window bounds from
                            # the precomputed tables (identical integer
                            # expressions, hoisted out of the cell loop)
                            rr = r + dx
                            cc = c + dy
                            tbu = buildings[rr, cc]
                            tsh_p = shadow[rr, cc]
                            tlsh = Lup_pre[rr, cc]
                            talsh = albshadow[rr, cc]
                            tanosh = alb[rr, cc]
                            tws = sunwall[rr, cc]
                        if tbu < f:
                            f = tbu

                        wsh += tsh_p * f
                        wlupsh += tlsh * f
                        walbsh += talsh * f
                        walbnosh += tanosh * f
                        tempb = tws * f
                        tempbwall = f * m_one + one
                        tbub = one if (tempb + tbub) > np.float32(0.0) else np.float32(0.0)
                        tbubw = one if (tempbwall + tbubw) > np.float32(0.0) else np.float32(0.0)
                    wlwall += tbub * Lwall32
                    walbwall += tbub * alb_b
                    wwall += tbub
                    walbwnosh += tbubw * alb_b
                    if n == 0:
                        # T18 K4: division by exactly fl32(1.0) is a
                        # bit-identity on finite operands; the step-0
                        # values are provably finite (one accumulated term
                        # of finite_plane_value × f ∈ {0,1}, or tbub ×
                        # finite scalar) — snapshot taken directly.
                        wsh_f = wsh
                        wwall_f = wwall
                        wlup_f = wlupsh
                        wlwall_f = wlwall
                        walb_f = walbsh
                        walbw_f = walbwall
                        walbn_f = walbnosh
                        walbwn_f = walbwnosh

                wif = one if wwall_f > np.float32(0.0) else np.float32(0.0)
                wis = one if wwall > np.float32(0.0) else np.float32(0.0)
                lif = one if walbwn_f > np.float32(0.0) else np.float32(0.0)
                lis_ = one if walbwnosh > np.float32(0.0) else np.float32(0.0)
                nif_ = wif * m_one + one
                nis_ = wis * m_one + one
                nlf_ = lif * m_one + one
                nls_ = lis_ * m_one + one

                br = az_branch[j]
                lo = az_low[j]
                hi = az_high[j]
                asp = aspect[r, c]
                if br == 1:
                    b2 = one if (asp < lo) or (asp >= hi) else np.float32(0.0)
                    facesh = b2 - wallbol[r, c] + one
                else:
                    b2 = one if (asp > lo) or (asp <= hi) else np.float32(0.0)
                    facesh = b2 * m_one + one

                keep = (one if wwall == second32 else np.float32(0.0)) - facesh
                if keep == m_one:
                    keep = np.float32(0.0)
                if keep == one:
                    wwall = np.float32(0.0)
                    wlwall = np.float32(0.0)
                    walbwall = np.float32(0.0)

                gvf2 = ((wwall + wsh) / second_p1) * wis + (wsh / second32) * nis_
                if gvf2 > one:
                    gvf2 = one
                gl1 = ((wlwall_f + wlup_f) / first_p1) * wif + (wlup_f / first32) * nif_
                gl2 = ((wlwall + wlupsh) / second_p1) * wis + (wlupsh / second32) * nis_
                ga1 = ((walbw_f + walb_f) / first_p1) * wif + (walb_f / first32) * nif_
                ga2 = ((walbwall + walbsh) / second_p1) * wis + (walbsh / second32) * nis_
                gn1 = ((walbwn_f + walbn_f) / first_p1) * lif + (walbn_f / first32) * nlf_
                # gvfalbnosh2 divides by `second` (NOT second+1) -- source
                # asymmetry kept verbatim (see mirror note)
                gn2 = ((walbwnosh + walbnosh) / second32) * lis_ + (walbnosh / second32) * nls_

                g_lup = (gl1 * c05 + gl2 * c04) / c09 + gvflup_extra[r, c]
                g_alb = (ga1 * c05 + ga2 * c04) / c09 + no_b_term * sh_rc
                g_nosh = (gn1 * c05 + gn2 * c04) / c09 * b_row + no_b_term

                acc_lup += g_lup
                acc_alb += g_alb
                acc_nosh += g_nosh
                if jE[j]:
                    ce_l += g_lup; ce_a += g_alb; ce_n += g_nosh
                if jS[j]:
                    cs_l += g_lup; cs_a += g_alb; cs_n += g_nosh
                if jW[j]:
                    cw_l += g_lup; cw_a += g_alb; cw_n += g_nosh
                if jN[j]:
                    cn_l += g_lup; cn_a += g_alb; cn_n += g_nosh

            ta_pow4 = (sbc * emis_grid[r, c]) * ta273
            gvfLup = acc_lup / n18 + ta_pow4
            gvfLupE = ce_l / n9 + ta_pow4
            gvfLupS = cs_l / n9 + ta_pow4
            gvfLupW = cw_l / n9 + ta_pow4
            gvfLupN = cn_l / n9 + ta_pow4
            gvfalb = acc_alb / n18
            gvfalbE = ce_a / n9
            gvfalbS = cs_a / n9
            gvfalbW = cw_a / n9
            gvfalbN = cn_a / n9
            gvfNosh = acc_nosh / n18
            gvfNoshE = ce_n / n9
            gvfNoshS = cs_n / n9
            gvfNoshW = cw_n / n9
            gvfNoshN = cn_n / n9

            # ---- TsWaveDelay x6 (entry fd/timeadd uniform per call) ----
            m_in = gvfLup if fd_eq_1 else m_lup_in[r, c]
            lup_out = _tsw_blend(gvfLup, w1_0, m_in)
            o_lup[r, c] = lup_out
            n_lup[r, c] = lup_out if branch2 else m_in
            m_in = gvfLupE if fd_eq_1 else m_e_in[r, c]
            lup_e = _tsw_blend(gvfLupE, w1_1, m_in)
            n_e[r, c] = lup_e if branch2 else m_in
            m_in = gvfLupS if fd_eq_1 else m_s_in[r, c]
            lup_s = _tsw_blend(gvfLupS, w1_2, m_in)
            n_s[r, c] = lup_s if branch2 else m_in
            m_in = gvfLupW if fd_eq_1 else m_w_in[r, c]
            lup_w = _tsw_blend(gvfLupW, w1_3, m_in)
            n_w[r, c] = lup_w if branch2 else m_in
            m_in = gvfLupN if fd_eq_1 else m_n_in[r, c]
            lup_n = _tsw_blend(gvfLupN, w1_4, m_in)
            n_n[r, c] = lup_n if branch2 else m_in
            TgTemp = Tg_plane[r, c] * sh_rc + Ta32
            m_in = TgTemp if fd_eq_1 else m_tg_in[r, c]
            tg_blend = _tsw_blend(TgTemp, w1_5, m_in)
            o_tgout[r, c] = tg_blend
            n_tg[r, c] = tg_blend if branch2 else m_in

            # ---- Kup family ----
            svf = svfbuveg[r, c]
            fsh = F_sh[r, c]
            inner = radD * svf + alb_b * (one - svf) * (radG * (one - fsh) + radD * fsh)
            Kup = (gvfalb * radI * sinalt) + inner * gvfNosh
            KupE = (gvfalbE * radI * sinalt) + inner * gvfNoshE
            KupS = (gvfalbS * radI * sinalt) + inner * gvfNoshS
            KupW = (gvfalbW * radI * sinalt) + inner * gvfNoshW
            KupN = (gvfalbN * radI * sinalt) + inner * gvfNoshN
            o_kup[r, c] = Kup
            o_ke[r, c] = KupE * half
            o_ks[r, c] = KupS * half
            o_kw[r, c] = KupW * half
            o_kn[r, c] = KupN * half

            # ---- Kside (Kside's own sos bits 0..P-1) ----
            KsideI = (sh_rc * radI) * cosalt
            o_ksidei[r, c] = KsideI
            KsideD = np.float32(0.0)
            Krs = np.float32(0.0)
            Krsh = np.float32(0.0)
            Krv = np.float32(0.0)
            for idx in range(n_patches):
                vegb = _pbit(veg_pb, r, c, idx)
                shb = _pbit(sh_pb, r, c, idx)
                vegf = one if vegb else np.float32(0.0)
                shf = one if shb else np.float32(0.0)
                ds = shf - (one - vegf) * c97
                KsideD += ((ds * lumChi[idx]) * pcos[idx]) * ster[idx]
                vbshb = _pbit(vbsh_pb, r, c, idx)
                tvb = one if (not vegb) or (not vbshb) else np.float32(0.0)
                Krv += ((ks_shd * tvb) * ster[idx]) * pcos[idx]
                tsh_ = one if (not shb) and vbshb else np.float32(0.0)
                slb = _pbit(sun_pb, r, c, idx)
                sdb = _pbit(shd_pb, r, c, idx)
                slf = one if slb else np.float32(0.0)
                sdf = one if sdb else np.float32(0.0)
                Krs += ((((ks_sun * slf) * tsh_) * ster[idx]) * pcos[idx])
                Krsh += ((((ks_shd * sdf) * tsh_) * ster[idx]) * pcos[idx])
            Kside = ((((KsideI + KsideD) + Krs) + Krsh) + Krv)
            o_ksided[r, c] = KsideD
            o_kside[r, c] = Kside

            # ---- Kdown ----
            Kdown = ((radI * sh_rc) * sinalt) + dRad \
                + (alb_b * (one - svf)) * (radG * (one - fsh) + radD * fsh)
            o_kdown[r, c] = Kdown

            # ---- define_patch two-pass, cell-local reflection barrier ----
            Ls_sky = np.float32(0.0)
            Ld_sky = np.float32(0.0)
            Ls_veg = np.float32(0.0)
            Ld_veg = np.float32(0.0)
            Ls_sun = np.float32(0.0)
            Ld_sun = np.float32(0.0)
            Ls_sh = np.float32(0.0)
            Ld_sh = np.float32(0.0)
            ae = np.float32(0.0)
            aso = np.float32(0.0)
            aw = np.float32(0.0)
            an = np.float32(0.0)
            for idx in range(n_patches):
                shb = _pbit(sh_pb, r, c, idx)
                vegb = _pbit(veg_pb, r, c, idx)
                vbshb = _pbit(vbsh_pb, r, c, idx)
                sky_m = one if shb and vegb else np.float32(0.0)
                veg_m = np.float64(1.0) if (not vegb) or (not vbshb) else np.float64(0.0)
                sun_m = one if (not shb) and vbshb else np.float32(0.0)
                Ld_sky += sky_m * lsky_d2[idx]
                Ls_sky += sky_m * lsky_s2[idx]
                # T18 K1: Pv/Psh/Psun products are patch-index-invariant —
                # precomputed per t by the host (see _dp_precalc_day); the
                # card variants fold the cardinal-cos multiply with the
                # ORIGINAL association preserved bit-for-bit.
                Ls_veg = np.float32(np.float64(Ls_veg) + pv_c[idx] * veg_m)
                Ld_veg = np.float32(np.float64(Ld_veg) + pv_s[idx] * veg_m)
                ce_b = card_e[idx]
                cs_b = card_s[idx]
                cw_b = card_w[idx]
                cn_b = card_n[idx]
                if ce_b:
                    ae += (sky_m * lsky_s2[idx]) * ccos_e[idx]
                    ae = np.float32(np.float64(ae) + pvc_e[idx] * veg_m)
                if cs_b:
                    aso += (sky_m * lsky_s2[idx]) * ccos_s[idx]
                    aso = np.float32(np.float64(aso) + pvc_s[idx] * veg_m)
                if cw_b:
                    aw += (sky_m * lsky_s2[idx]) * ccos_w[idx]
                    aw = np.float32(np.float64(aw) + pvc_w[idx] * veg_m)
                if cn_b:
                    an += (sky_m * lsky_s2[idx]) * ccos_n[idx]
                    an = np.float32(np.float64(an) + pvc_n[idx] * veg_m)

                if guard_true[idx]:
                    k = kside_n + dp_rank[idx]
                    sl = _pbit(sun_pb, r, c, k)
                    sd = _pbit(shd_pb, r, c, k)
                    g1 = np.float64(1.0) if sl and (sun_m == one) else np.float64(0.0)
                    g2 = np.float64(1.0) if sd and (sun_m == one) else np.float64(0.0)
                    Ls_sun = np.float32(np.float64(Ls_sun) + psun_c[idx] * g1)
                    Ld_sun = np.float32(np.float64(Ld_sun) + psun_s[idx] * g1)
                    Ls_sh = np.float32(np.float64(Ls_sh) + psh_c[idx] * g2)
                    Ld_sh = np.float32(np.float64(Ld_sh) + psh_s[idx] * g2)
                    if ce_b:
                        ae = np.float32(np.float64(ae) + psunc_e[idx] * g1)
                        ae = np.float32(np.float64(ae) + pshc_e[idx] * g2)
                    if cs_b:
                        aso = np.float32(np.float64(aso) + psunc_s[idx] * g1)
                        aso = np.float32(np.float64(aso) + pshc_s[idx] * g2)
                    if cw_b:
                        aw = np.float32(np.float64(aw) + psunc_w[idx] * g1)
                        aw = np.float32(np.float64(aw) + pshc_w[idx] * g2)
                    if cn_b:
                        an = np.float32(np.float64(an) + psunc_n[idx] * g1)
                        an = np.float32(np.float64(an) + pshc_n[idx] * g2)
                else:
                    Ls_sh = np.float32(np.float64(Ls_sh) + psh_c[idx] * np.float64(sun_m))
                    Ld_sh = np.float32(np.float64(Ld_sh) + psh_s[idx] * np.float64(sun_m))
                    if ce_b:
                        ae = np.float32(np.float64(ae) + pshc_e[idx] * np.float64(sun_m))
                    if cs_b:
                        aso = np.float32(np.float64(aso) + pshc_s[idx] * np.float64(sun_m))
                    if cw_b:
                        aw = np.float32(np.float64(aw) + pshc_w[idx] * np.float64(sun_m))
                    if cn_b:
                        an = np.float32(np.float64(an) + pshc_n[idx] * np.float64(sun_m))

            # barrier: reflected term needs only THIS cell's Ldown_sky
            refl = (((Ld_sky + lup_out) * ome) * half) / pi32
            Ls_ref = np.float32(0.0)
            Ld_ref = np.float32(0.0)
            for idx in range(n_patches):
                shb = _pbit(sh_pb, r, c, idx)
                vegb = _pbit(veg_pb, r, c, idx)
                vbshb = _pbit(vbsh_pb, r, c, idx)
                rm = one if (not shb) or (not vegb) or (not vbshb) else np.float32(0.0)
                Ls_ref += (((refl * ster[idx]) * pcos[idx]) * rm)
                Ld_ref += (((refl * ster[idx]) * psin[idx]) * rm)
                if card_e[idx]:
                    ae += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_e[idx])
                if card_s[idx]:
                    aso += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_s[idx])
                if card_w[idx]:
                    aw += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_w[idx])
                if card_n[idx]:
                    an += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_n[idx])

            Lside = ((((Ls_sky + Ls_veg) + Ls_sh) + Ls_sun) + Ls_ref)
            Ldown = ((((Ld_sky + Ld_veg) + Ld_sh) + Ld_sun) + Ld_ref)
            o_lside[r, c] = Lside
            o_ldown[r, c] = Ldown

            # ---- Lside_veg (tsw cardinals * 0.5) ----
            le_lv = lup_e * half
            ls_lv = lup_s * half
            lw_lv = lup_w * half
            ln_lv = lup_n * half

            # ---- Sstr / Tmrt (cyl + anisotropic) ----
            ke_s = KupE * half
            ks_s = KupS * half
            kw_s = KupW * half
            kn_s = KupN * half
            short = Kside * f_cyl + (Kdown + Kup) * f_up \
                + (((kn_s + ke_s) + ks_s) + kw_s) * f_side
            long_ = (Ldown + lup_out) * f_up + Lside * f_cyl \
                + (((ln_lv + le_lv) + ls_lv) + lw_lv) * f_side
            Sstr = short * abs_k + long_ * abs_l
            o_tmrt[r, c] = np.sqrt(np.sqrt(Sstr / (abs_l * sbc))) - r2732

            # ---- POI cardinals (lsideveg + dp) ----
            o_le[r, c] = le_lv + ae
            o_ls[r, c] = ls_lv + aso
            o_lw[r, c] = lw_lv + aw
            o_ln[r, c] = ln_lv + an


@njit(cache=True, nogil=True, fastmath=False)
def _fused_night_kernel(
    rows, cols, n_patches,
    sh_pb, veg_pb, vbsh_pb,
    night_Lup,
    ster, psin, pcos, lsky_d2, lsky_s2,
    card_e, card_s, card_w, card_n,
    ccos_e, ccos_s, ccos_w, ccos_n,
    pv_c, pv_s, psh_c, psh_s,
    pvc_e, pvc_s, pvc_w, pvc_n,
    pshc_e, pshc_s, pshc_w, pshc_n,
    o_tmrt, o_ldown, o_lside, o_le, o_ls, o_lw, o_ln,
):
    one = np.float32(1.0)
    half = np.float32(0.5)
    f_cyl = np.float32(0.28)
    f_up = np.float32(0.06)
    f_side = np.float32(0.22)
    abs_k = np.float32(0.7)
    abs_l = np.float32(0.95)
    sbc = np.float32(5.67051e-8)
    r2732 = np.float32(273.2)

    for r in range(rows):
        for c in range(cols):
            Lup = night_Lup[r, c]
            Ls_sky = np.float32(0.0)
            Ld_sky = np.float32(0.0)
            Ls_veg = np.float32(0.0)
            Ld_veg = np.float32(0.0)
            Ls_sh = np.float32(0.0)
            Ld_sh = np.float32(0.0)
            ae = np.float32(0.0)
            aso = np.float32(0.0)
            aw = np.float32(0.0)
            an = np.float32(0.0)
            for idx in range(n_patches):
                shb = _pbit(sh_pb, r, c, idx)
                vegb = _pbit(veg_pb, r, c, idx)
                vbshb = _pbit(vbsh_pb, r, c, idx)
                sky_m = one if shb and vegb else np.float32(0.0)
                veg_m = np.float64(1.0) if (not vegb) or (not vbshb) else np.float64(0.0)
                sun_m = one if (not shb) and vbshb else np.float32(0.0)
                Ld_sky += sky_m * lsky_d2[idx]
                Ls_sky += sky_m * lsky_s2[idx]
                # T18 K1: patch-index-invariant products precomputed per t
                # (see _dp_precalc_night; association preserved bit-exact).
                Ls_veg = np.float32(np.float64(Ls_veg) + pv_c[idx] * veg_m)
                Ld_veg = np.float32(np.float64(Ld_veg) + pv_s[idx] * veg_m)
                if card_e[idx]:
                    ae += (sky_m * lsky_s2[idx]) * ccos_e[idx]
                    ae = np.float32(np.float64(ae) + pvc_e[idx] * veg_m)
                if card_s[idx]:
                    aso += (sky_m * lsky_s2[idx]) * ccos_s[idx]
                    aso = np.float32(np.float64(aso) + pvc_s[idx] * veg_m)
                if card_w[idx]:
                    aw += (sky_m * lsky_s2[idx]) * ccos_w[idx]
                    aw = np.float32(np.float64(aw) + pvc_w[idx] * veg_m)
                if card_n[idx]:
                    an += (sky_m * lsky_s2[idx]) * ccos_n[idx]
                    an = np.float32(np.float64(an) + pvc_n[idx] * veg_m)
                # night: solar_altitude <= 0 -> guard False everywhere ->
                # else branch (shaded term on temp_sh)
                Ls_sh = np.float32(np.float64(Ls_sh) + psh_c[idx] * np.float64(sun_m))
                Ld_sh = np.float32(np.float64(Ld_sh) + psh_s[idx] * np.float64(sun_m))
                if card_e[idx]:
                    ae = np.float32(np.float64(ae) + pshc_e[idx] * np.float64(sun_m))
                if card_s[idx]:
                    aso = np.float32(np.float64(aso) + pshc_s[idx] * np.float64(sun_m))
                if card_w[idx]:
                    aw = np.float32(np.float64(aw) + pshc_w[idx] * np.float64(sun_m))
                if card_n[idx]:
                    an = np.float32(np.float64(an) + pshc_n[idx] * np.float64(sun_m))

            ome = one - np.float32(0.9)
            refl = (((Ld_sky + Lup) * ome) * half) / np.float32(np.pi)
            Ls_ref = np.float32(0.0)
            Ld_ref = np.float32(0.0)
            for idx in range(n_patches):
                shb = _pbit(sh_pb, r, c, idx)
                vegb = _pbit(veg_pb, r, c, idx)
                vbshb = _pbit(vbsh_pb, r, c, idx)
                rm = one if (not shb) or (not vegb) or (not vbshb) else np.float32(0.0)
                Ls_ref += (((refl * ster[idx]) * pcos[idx]) * rm)
                Ld_ref += (((refl * ster[idx]) * psin[idx]) * rm)
                if card_e[idx]:
                    ae += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_e[idx])
                if card_s[idx]:
                    aso += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_s[idx])
                if card_w[idx]:
                    aw += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_w[idx])
                if card_n[idx]:
                    an += ((((refl * ster[idx]) * pcos[idx]) * rm) * ccos_n[idx])

            Lside = ((((Ls_sky + Ls_veg) + Ls_sh) + np.float32(0.0)) + Ls_ref)
            Ldown = ((((Ld_sky + Ld_veg) + Ld_sh) + np.float32(0.0)) + Ld_ref)
            o_lside[r, c] = Lside
            o_ldown[r, c] = Ldown

            le_lv = Lup * half
            ls_lv = Lup * half
            lw_lv = Lup * half
            ln_lv = Lup * half
            short = np.float32(0.0) * f_cyl + (np.float32(0.0) + np.float32(0.0)) * f_up \
                + (((np.float32(0.0) + np.float32(0.0)) + np.float32(0.0)) + np.float32(0.0)) * f_side
            long_ = (Ldown + Lup) * f_up + Lside * f_cyl \
                + (((ln_lv + le_lv) + ls_lv) + lw_lv) * f_side
            Sstr = short * abs_k + long_ * abs_l
            o_tmrt[r, c] = np.sqrt(np.sqrt(Sstr / (abs_l * sbc))) - r2732
            o_le[r, c] = le_lv + ae
            o_ls[r, c] = ls_lv + aso
            o_lw[r, c] = lw_lv + aw
            o_ln[r, c] = ln_lv + an


def _plane(rows, cols):
    return np.empty((rows, cols), dtype=np.float32)


def _dp_precalc_day(t_in: RadTimestepInputs):
    """Per-t precompute of the define_patch pass-1 patch-index-invariant
    f64 products (T18 K1 structural CSE — bit-exact: each array is the
    SAME expression the kernel evaluated per cell, evaluated once; f64
    numpy elementwise == the kernel's scalar IEEE ops on identical inputs).

    Base: ``Pv_c = (veg64 * st64) * pc64`` and the s/c/sun variants —
    every factor is per-t (scalar surface × per-patch trig), so the
    per-cell recomputation 250k× per timestep was pure structural waste.
    Card variants additionally fold the cardinal cos multiply, PRESERVING
    the original association ``(Pv_c * ccos_x) * mask`` as
    ``pvc_x[idx] * mask`` with ``pvc_x[idx] == Pv_c * ccos_x[idx]`` exact.
    """
    ster64 = t_in.steradian.astype(np.float64)
    pcos64 = t_in.patch_cos.astype(np.float64)
    psin64 = t_in.patch_sin.astype(np.float64)
    veg64 = np.float64(t_in.dp_veg_surface_f64)
    shd64 = np.float64(t_in.dp_shaded_surface_f64)
    sun64 = np.float64(t_in.dp_sunlit_surface_f64)
    pv_c = (veg64 * ster64) * pcos64
    pv_s = (veg64 * ster64) * psin64
    psh_c = (shd64 * ster64) * pcos64
    psh_s = (shd64 * ster64) * psin64
    psun_c = (sun64 * ster64) * pcos64
    psun_s = (sun64 * ster64) * psin64
    cards = {}
    for d, cc in (("e", t_in.card_cos_e), ("s", t_in.card_cos_s),
                  ("w", t_in.card_cos_w), ("n", t_in.card_cos_n)):
        cc64 = cc.astype(np.float64)
        cards[f"pvc_{d}"] = pv_c * cc64
        cards[f"pshc_{d}"] = psh_c * cc64
        cards[f"psunc_{d}"] = psun_c * cc64
    return (pv_c, pv_s, psh_c, psh_s, psun_c, psun_s,
            cards["pvc_e"], cards["pvc_s"], cards["pvc_w"], cards["pvc_n"],
            cards["pshc_e"], cards["pshc_s"], cards["pshc_w"],
            cards["pshc_n"],
            cards["psunc_e"], cards["psunc_s"], cards["psunc_w"],
            cards["psunc_n"])


def _walk_window_tables(t_in: RadTimestepInputs, rows: int, cols: int):
    """Per-(j,n) write-window bounds (T18 K3): the kernel's integer
    expressions ``xp1..yp2`` depend only on the walk offsets, not on the
    cell — precomputed once per t with the IDENTICAL integer arithmetic
    (pure integer refactor, no FP values touched)."""
    dx = t_in.walk_dx.astype(np.int64)
    dy = t_in.walk_dy.astype(np.int64)
    adx = np.abs(dx)
    ady = np.abs(dy)
    wxp1 = -((dx - adx) // 2)
    wxp2 = rows - (dx + adx) // 2
    wyp1 = -((dy - ady) // 2)
    wyp2 = cols - (dy + ady) // 2
    return wxp1, wxp2, wyp1, wyp2


def _assert_walk_planes_finite(st: RadiationStatic, t_in: RadTimestepInputs,
                               sunwall: np.ndarray,
                               albshadow: np.ndarray):
    """T18 K2 premise fence (loud, never silent): the f-floor dead-scan
    elision skips ``acc += plane_value * f`` terms once ``f == +0.0`` —
    exact IFF every multiplied plane value is finite (a NaN/Inf would
    otherwise propagate as NaN*0.0 = NaN). The construction argument
    (march/met outputs are finite chains over finite rasters) is
    enforced live per day-timestep for the planes the walk reads.

    Lead addition (T18 landing review): ``buildings`` and ``sunwall``
    get domain checks beyond finiteness. The original only clamps
    ``buildings[buildings >= 2.] = 0.`` (utci_process :679/:1169), so a
    NEGATIVE finite value would let the original running min drop below
    +0.0 while the elision keeps f == +0.0 — a silent divergence the
    isfinite gate cannot see (fractional positives are proven safe: at
    f == +0.0 ``tbu < f`` is False and every skipped term is still a
    ×±0.0 no-op). ``sunwall`` is bool-cast (⊆ {0.0, 1.0}) by
    construction in the caller; the check keeps this fence
    self-contained instead of relying on that remote construction."""
    for name, v in (("shadow", t_in.shadow), ("Lup_pre", t_in.Lup_pre),
                    ("albshadow", albshadow), ("alb", st.alb_grid),
                    ("buildings", st.buildings)):
        if not np.isfinite(v).all():
            raise RuntimeError(
                f"walk plane {name} contains non-finite values — the "
                "f-floor dead-scan elision premise (finite multiplicands) "
                "is violated; refusing to solve silently wrong (T18 K2 "
                "fence)")
    for name, v in (("buildings", st.buildings), ("sunwall", sunwall)):
        if not bool(((v == 0.0) | (v == 1.0)).all()):
            raise RuntimeError(
                f"{name} plane outside {{0.0, 1.0}} — the f-floor "
                "dead-scan elision premise (running min of buildings can "
                "never drop below +0.0; tws ∈ {0,1} keeps the latch "
                "no-ops exact) is violated; refusing to solve silently "
                "wrong (T18 K2 fence)")
    if not np.isfinite(t_in.Lwall):
        raise RuntimeError("Lwall scalar non-finite (T18 K2 fence)")


def _dp_precalc_night(t_in: RadTimestepInputs):
    """Night-kernel subset of :func:`_dp_precalc_day` (no sun terms)."""
    ster64 = t_in.steradian.astype(np.float64)
    pcos64 = t_in.patch_cos.astype(np.float64)
    psin64 = t_in.patch_sin.astype(np.float64)
    veg64 = np.float64(t_in.dp_veg_surface_f64)
    shd64 = np.float64(t_in.dp_shaded_surface_f64)
    pv_c = (veg64 * ster64) * pcos64
    pv_s = (veg64 * ster64) * psin64
    psh_c = (shd64 * ster64) * pcos64
    psh_s = (shd64 * ster64) * psin64
    cards = {}
    for d, cc in (("e", t_in.card_cos_e), ("s", t_in.card_cos_s),
                  ("w", t_in.card_cos_w), ("n", t_in.card_cos_n)):
        cc64 = cc.astype(np.float64)
        cards[f"pvc_{d}"] = pv_c * cc64
        cards[f"pshc_{d}"] = psh_c * cc64
    return (pv_c, pv_s, psh_c, psh_s,
            cards["pvc_e"], cards["pvc_s"], cards["pvc_w"], cards["pvc_n"],
            cards["pshc_e"], cards["pshc_s"], cards["pshc_w"],
            cards["pshc_n"])


def _fused_day_host(st: RadiationStatic, t_in: RadTimestepInputs,
                    state: RadLoopState):
    rows, cols = st.rows, st.cols
    n_p = int(t_in.steradian.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        # walls==0 cells -> inf/nan -> (== 1) False -> 0.0, source semantics
        sunwall = ((t_in.wallsun / st.walls * st.buildings) == 1).astype(np.float32)
    albshadow = st.alb_grid * t_in.shadow
    dp_pre = _dp_precalc_day(t_in)
    wxp1, wxp2, wyp1, wyp2 = _walk_window_tables(t_in, rows, cols)
    _assert_walk_planes_finite(st, t_in, sunwall, albshadow)

    guard_true = np.zeros(n_p, dtype=np.bool_)
    dp_rank = np.full(n_p, -1, dtype=np.int64)
    gi = t_in.dp_guard_idx
    guard_true[gi] = True
    dp_rank[gi] = np.arange(gi.shape[0], dtype=np.int64)

    az_deg = [5 + 20 * j for j in range(18)]
    jE = np.array([0 <= a < 180 for a in az_deg], dtype=np.bool_)
    jS = np.array([90 <= a < 270 for a in az_deg], dtype=np.bool_)
    jW = np.array([180 <= a < 360 for a in az_deg], dtype=np.bool_)
    jN = np.array([(a >= 270 or a < 90) for a in az_deg], dtype=np.bool_)

    ta_in = state.timeadd
    tsd = t_in.timestepdec_f64
    thr = 59 / 1440
    if ta_in >= thr:
        branch2 = True
        ta_out = tsd if tsd > thr else 0.0
    else:
        branch2 = False
        ta_out = ta_in + tsd
    fd_eq_1 = state.firstdaytime == 1

    w1 = t_in.tsw_weight1
    o_tmrt = _plane(rows, cols); o_kdown = _plane(rows, cols)
    o_kup = _plane(rows, cols); o_ldown = _plane(rows, cols)
    o_lup = _plane(rows, cols); o_ke = _plane(rows, cols)
    o_ks = _plane(rows, cols); o_kw = _plane(rows, cols)
    o_kn = _plane(rows, cols); o_le = _plane(rows, cols)
    o_ls = _plane(rows, cols); o_lw = _plane(rows, cols)
    o_ln = _plane(rows, cols); o_ksidei = _plane(rows, cols)
    o_tgout = _plane(rows, cols); o_lside = _plane(rows, cols)
    o_ksided = _plane(rows, cols); o_drad = _plane(rows, cols)
    o_kside = _plane(rows, cols)
    n_lup = _plane(rows, cols); n_e = _plane(rows, cols)
    n_s = _plane(rows, cols); n_w = _plane(rows, cols)
    n_n = _plane(rows, cols); n_tg = _plane(rows, cols)

    _fused_day_kernel(
        rows, cols, n_p, int(t_in.kside_n_patches),
        st.buildings, st.aspect_rad, st.wallbol, st.alb_grid,
        st.emis_grid, st.svfbuveg,
        st.shmat_packed, st.vegshmat_packed, st.vbshvegshmat_packed,
        t_in.sunlit_packed, t_in.shaded_packed,
        dp_rank, guard_true,
        t_in.shadow, sunwall, albshadow, st.alb_grid, t_in.Lup_pre,
        t_in.gvflup_extra,
        t_in.lv[:, 2].copy(), t_in.steradian, t_in.patch_sin,
        t_in.patch_cos, t_in.lumChi, t_in.Lsky_down2, t_in.Lsky_side2,
        t_in.card_e, t_in.card_s, t_in.card_w, t_in.card_n,
        t_in.card_cos_e, t_in.card_cos_s, t_in.card_cos_w,
        t_in.card_cos_n,
        t_in.ks_sunlit, t_in.ks_shaded, t_in.radI, t_in.radD, t_in.radG,
        t_in.sinalt, t_in.cosalt,
        *dp_pre,
        t_in.ta273_pow4, np.float32(t_in.Lwall),
        t_in.F_sh, t_in.Tg, t_in.Ta,
        t_in.walk_dy, t_in.walk_dx, wxp1, wxp2, wyp1, wyp2,
        t_in.az_statics["branch"], t_in.az_statics["azilow"],
        t_in.az_statics["azihigh"],
        jE, jS, jW, jN,
        w1[0], w1[1], w1[2], w1[3], w1[4], w1[5],
        fd_eq_1, branch2,
        state.Tgmap1, state.Tgmap1E, state.Tgmap1S, state.Tgmap1W,
        state.Tgmap1N, state.TgOut1,
        o_tmrt, o_kdown, o_kup, o_ldown, o_lup,
        o_ke, o_ks, o_kw, o_kn,
        o_le, o_ls, o_lw, o_ln, o_ksidei, o_tgout, o_lside,
        o_ksided, o_drad, o_kside,
        n_lup, n_e, n_s, n_w, n_n, n_tg,
    )
    out = {
        "Tmrt": o_tmrt, "Kdown": o_kdown, "Kup": o_kup, "Ldown": o_ldown,
        "Lup": o_lup, "Tg": t_in.Tg, "shadow": t_in.shadow,
        "Keast": o_ke, "Ksouth": o_ks, "Kwest": o_kw, "Knorth": o_kn,
        "Least": o_le, "Lsouth": o_ls, "Lwest": o_lw, "Lnorth": o_ln,
        "KsideI": o_ksidei, "TgOut": o_tgout, "Lside": o_lside,
        "CI_Tg": None, "CI_TgG": None,
        "KsideD": o_ksided, "dRad": o_drad, "Kside": o_kside,
    }
    next_state = RadLoopState(
        firstdaytime=0, timeadd=ta_out,
        CI=(state.CI if t_in.CI_out is None else np.float32(t_in.CI_out)),
        Tgmap1=n_lup, Tgmap1E=n_e, Tgmap1S=n_s, Tgmap1W=n_w,
        Tgmap1N=n_n, TgOut1=n_tg,
    )
    return out, next_state


def _fused_night_host(st: RadiationStatic, t_in: RadTimestepInputs,
                      state: RadLoopState):
    rows, cols = st.rows, st.cols
    n_p = int(t_in.steradian.shape[0])
    o_tmrt = _plane(rows, cols); o_ldown = _plane(rows, cols)
    o_lside = _plane(rows, cols)
    o_le = _plane(rows, cols); o_ls = _plane(rows, cols)
    o_lw = _plane(rows, cols); o_ln = _plane(rows, cols)
    _fused_night_kernel(
        rows, cols, n_p,
        st.shmat_packed, st.vegshmat_packed, st.vbshvegshmat_packed,
        t_in.night_Lup,
        t_in.steradian, t_in.patch_sin, t_in.patch_cos,
        t_in.Lsky_down2, t_in.Lsky_side2,
        t_in.card_e, t_in.card_s, t_in.card_w, t_in.card_n,
        t_in.card_cos_e, t_in.card_cos_s, t_in.card_cos_w,
        t_in.card_cos_n,
        *_dp_precalc_night(t_in),
        o_tmrt, o_ldown, o_lside, o_le, o_ls, o_lw, o_ln,
    )
    zero = np.zeros((rows, cols), dtype=np.float32)
    tgout_plane = np.full((rows, cols), np.float32(t_in.Ta + np.float32(0.0)),
                          dtype=np.float32)
    out = {
        "Tmrt": o_tmrt, "Kdown": zero.copy(), "Kup": zero.copy(),
        "Ldown": o_ldown, "Lup": t_in.night_Lup, "Tg": zero.copy(),
        "shadow": zero.copy(),
        "Keast": zero.copy(), "Ksouth": zero.copy(), "Kwest": zero.copy(),
        "Knorth": zero.copy(),
        "Least": o_le, "Lsouth": o_ls, "Lwest": o_lw, "Lnorth": o_ln,
        "KsideI": zero.copy(), "TgOut": tgout_plane, "Lside": o_lside,
        "CI_Tg": state.CI, "CI_TgG": state.CI, "KsideD": zero.copy(),
        "dRad": zero.copy(), "Kside": zero.copy(),
    }
    next_state = RadLoopState(
        firstdaytime=1, timeadd=0.0, CI=state.CI,
        Tgmap1=state.Tgmap1, Tgmap1E=state.Tgmap1E,
        Tgmap1S=state.Tgmap1S, Tgmap1W=state.Tgmap1W,
        Tgmap1N=state.Tgmap1N, TgOut1=state.TgOut1,
    )
    return out, next_state
