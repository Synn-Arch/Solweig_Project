# SPDX-License-Identifier: GPL-3.0-only
"""Shared bundle builder for the T12 CUDA radiation gates.

``build_rad_day_bundle`` / ``build_rad_night_bundle`` assemble the FLAT
kernel-input dict consumed by ``native/cuda/host.py:run_rad_day/night``
from (RadiationStatic, RadTimestepInputs, RadLoopState) objects — the
same host-side assembly solweig_core/numba_cpu/radiation.py performs for
the fused kernels (sunwall, albshadow, guard/rank tables, jE..jN, the
f64 timeadd branch). numpy-only, torch-free: the CUDA runtime never sees
a dataclass, only arrays and scalars.
"""
from __future__ import annotations

import numpy as np

__all__ = ["build_rad_day_bundle", "build_rad_night_bundle"]


def _j_flags():
    az_deg = [5 + 20 * j for j in range(18)]
    jE = np.array([0 <= a < 180 for a in az_deg], dtype=np.bool_)
    jS = np.array([90 <= a < 270 for a in az_deg], dtype=np.bool_)
    jW = np.array([180 <= a < 360 for a in az_deg], dtype=np.bool_)
    jN = np.array([(a >= 270 or a < 90) for a in az_deg], dtype=np.bool_)
    return jE, jS, jW, jN


def build_rad_day_bundle(st, t_in, state) -> dict:
    """Day bundle (mirrors _fused_day_host's assembly exactly)."""
    rows, cols = st.rows, st.cols
    n_p = int(t_in.steradian.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        # walls==0 cells -> inf/nan -> (== 1) False -> 0.0, source semantics
        sunwall = ((t_in.wallsun / st.walls * st.buildings) == 1).astype(np.float32)
    albshadow = st.alb_grid * t_in.shadow

    guard_true = np.zeros(n_p, dtype=np.bool_)
    dp_rank = np.full(n_p, -1, dtype=np.int64)
    gi = t_in.dp_guard_idx
    guard_true[gi] = True
    dp_rank[gi] = np.arange(gi.shape[0], dtype=np.int64)

    jE, jS, jW, jN = _j_flags()

    ta_in = state.timeadd
    tsd = t_in.timestepdec_f64
    thr = 59 / 1440
    if ta_in >= thr:
        branch2 = True
    else:
        branch2 = False
    fd_eq_1 = state.firstdaytime == 1

    w1 = t_in.tsw_weight1
    az = t_in.az_statics
    return {
        "buildings": st.buildings, "aspect": st.aspect_rad,
        "wallbol": st.wallbol, "alb_grid": st.alb_grid, "alb": st.alb_grid,
        "emis_grid": st.emis_grid, "svfbuveg": st.svfbuveg,
        "diffsh": (
            st.diffsh if st.diffsh is not None
            else __import__(
                "solweig_core.numba_cpu.radiation", fromlist=["x"]
            ).dense_diffsh_from_packed(st)
        ),
        "sh_pb": st.shmat_packed, "veg_pb": st.vegshmat_packed,
        "vbsh_pb": st.vbshvegshmat_packed,
        "sun_pb": t_in.sunlit_packed, "shd_pb": t_in.shaded_packed,
        "dp_rank": dp_rank, "guard_true": guard_true.astype(np.int8),
        "shadow": t_in.shadow, "sunwall": sunwall,
        "albshadow": albshadow, "Lup_pre": t_in.Lup_pre,
        "gvflup_extra": t_in.gvflup_extra,
        "lv2": t_in.lv[:, 2].copy(), "ster": t_in.steradian,
        "psin": t_in.patch_sin, "pcos": t_in.patch_cos,
        "lumChi": t_in.lumChi, "lsky_d2": t_in.Lsky_down2,
        "lsky_s2": t_in.Lsky_side2,
        "card_e": t_in.card_e.astype(np.int8),
        "card_s": t_in.card_s.astype(np.int8),
        "card_w": t_in.card_w.astype(np.int8),
        "card_n": t_in.card_n.astype(np.int8),
        "ccos_e": t_in.card_cos_e, "ccos_s": t_in.card_cos_s,
        "ccos_w": t_in.card_cos_w, "ccos_n": t_in.card_cos_n,
        "walk_az_low": np.asarray(az["azilow"], dtype=np.float32),
        "walk_az_high": np.asarray(az["azihigh"], dtype=np.float32),
        "walk_az_branch": np.asarray(az["branch"], dtype=np.int32),
        "walk_dy": np.asarray(t_in.walk_dy, dtype=np.int32),
        "walk_dx": np.asarray(t_in.walk_dx, dtype=np.int32),
        "jE": jE.astype(np.int8), "jS": jS.astype(np.int8),
        "jW": jW.astype(np.int8), "jN": jN.astype(np.int8),
        "F_sh": t_in.F_sh, "Tg_plane": t_in.Tg,
        "m_lup_in": state.Tgmap1, "m_e_in": state.Tgmap1E,
        "m_s_in": state.Tgmap1S, "m_w_in": state.Tgmap1W,
        "m_n_in": state.Tgmap1N, "m_tg_in": state.TgOut1,
        "ks_sun": t_in.ks_sunlit, "ks_shd": t_in.ks_shaded,
        "radI": t_in.radI, "radD": t_in.radD, "radG": t_in.radG,
        "sinalt": t_in.sinalt, "cosalt": t_in.cosalt,
        "veg64": float(t_in.dp_veg_surface_f64),
        "shd64": float(t_in.dp_shaded_surface_f64),
        "sun64": float(t_in.dp_sunlit_surface_f64),
        "ta273": t_in.ta273_pow4, "Lwall32": np.float32(t_in.Lwall),
        "Ta32": t_in.Ta,
        "w1_0": w1[0], "w1_1": w1[1], "w1_2": w1[2],
        "w1_3": w1[3], "w1_4": w1[4], "w1_5": w1[5],
        "rows": rows, "cols": cols, "n_patches": n_p,
        "kside_n": int(t_in.kside_n_patches),
        "fd_eq_1": 1 if fd_eq_1 else 0, "branch2": 1 if branch2 else 0,
        "sun_stride": int(t_in.sunlit_packed.shape[2]),
    }


def build_rad_night_bundle(st, t_in) -> dict:
    """Night bundle (mirrors _fused_night_host's assembly exactly)."""
    return {
        "sh_pb": st.shmat_packed, "veg_pb": st.vegshmat_packed,
        "vbsh_pb": st.vbshvegshmat_packed, "night_Lup": t_in.night_Lup,
        "ster": t_in.steradian, "psin": t_in.patch_sin,
        "pcos": t_in.patch_cos,
        "lsky_d2": t_in.Lsky_down2, "lsky_s2": t_in.Lsky_side2,
        "card_e": t_in.card_e.astype(np.int8),
        "card_s": t_in.card_s.astype(np.int8),
        "card_w": t_in.card_w.astype(np.int8),
        "card_n": t_in.card_n.astype(np.int8),
        "ccos_e": t_in.card_cos_e, "ccos_s": t_in.card_cos_s,
        "ccos_w": t_in.card_cos_w, "ccos_n": t_in.card_cos_n,
        "veg64": float(t_in.dp_veg_surface_f64),
        "shd64": float(t_in.dp_shaded_surface_f64),
        "rows": st.rows, "cols": st.cols,
        "n_patches": int(t_in.steradian.shape[0]),
    }
