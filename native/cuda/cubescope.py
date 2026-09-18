# SPDX-License-Identifier: GPL-3.0-only
"""T13 follow-up — the cube-scope solver-loop driver on CUDA.

Runs the ORACLE'S 24-timestep solver order (march -> Solweig state
in/out -> utci -> publish) with the T12 CUDA kernels in place of the
fused CPU stage, threading the cross-timestep state exactly as
``solweig_core/numba_cpu/full_solve.py:full_solve_capture`` does — the
driver whose IDENTITY gate (T11) proved the host-threaded carry
reproduces the capture's per-t ``ret_*`` bits bitwise:

* per t: ``recompute_timestep`` regenerates the met-dependent bundle
  fields (CI threaded via the driver scalar ``ci_thread`` with the
  tensor-flavor flag), ``_apply_day/night_overrides`` writes them onto
  the capture bundle, and the RADIATION kernel consumes (statics,
  bundle, STATE) and emits the published planes plus the six NEXT-state
  planes (``n_lup``/``n_e``/``n_s``/``n_w``/``n_n``/``n_tg`` ==
  Tgmap1/Tgmap1E/S/W/N/TgOut1 — ``radiation._fused_day_host``'s
  next-state construction mirrored scalar-for-scalar);
* day: ``firstdaytime -> 0``, ``timeadd`` advanced by the 59/1440
  branch (``_fused_day_host`` lines 1766-1774); night: planes carried
  unchanged, ``firstdaytime -> 1``, ``timeadd -> 0``;
* utci: the oracle's per-t construction (utci_process.py:977-1049) —
  ``Ta_mat = zeros + Ta[i] + uhii[i]``, ``RH_mat = zeros + RH[i]``,
  ``va = clamp(ones * Ws[i], 0.15)`` — evaluated by the CUDA dense
  UTCI kernel and scattered into a NaN plane over ``buildings == 1``;
* published: shadow = the bundle's frozen march composition at day,
  zeros at night; tmrt = the kernel's Tmrt; utci = the NaN-scattered
  plane. Cube digests follow the oracle convention (sha256 over the
  contiguous (n_t, rows, cols) f32 bytes).

STATE-CARRY DECISION (documented per the task contract): host-carried
state with per-t kernel launches. The operation order matches the
legacy oracle's solver loop because it IS ``full_solve_capture``'s
order — met recompute, bundle assembly, kernel, CI re-thread — with the
kernel swapped; T11's identity gate pins that order to the capture
bits, and T12/T13 pin kernel == fused-CPU bits (canonical) and the
resident runner == wrapper bits. No march is launched here: the
capture's frozen march planes are the provenance-true inputs the T12
kernel gates consume (the oracle's wall-leg tail divergence is
documented in the digest lattice; wall planes are not published
outputs).

Torch-free (numpy + the ctypes CUDA host); profile discipline is the
caller's: pass ONE ``CudaRuntime`` (canonical for the control gate,
legacy_cuda_v1 for the characterization lane — never mixed).

T17 host-side loop optimization (bit-parity preserved — kernels receive
byte-identical inputs and produce byte-identical outputs; the cube pins
and the committed loop's per-t digests are the fence):

* the dense diffsh cube is materialized ONCE per call
  (``dense_diffsh_from_packed`` — the bundle builder's own fallback
  function, so the bundle's diffsh bits are identical by construction;
  the pre-T17 shape recomputed this ~146 MiB cube inside
  ``build_rad_day_bundle`` at EVERY day timestep);
* the scene-invariant static bind is hoisted out of the per-timestep
  loop: day statics bind at the FIRST day timestep, night statics
  (a name/content subset of the day set — both are pure ``st``
  references in the pinned builder) at the FIRST night timestep, and
  neither is rebound afterwards. The committed per-t rebind scanned
  identical bytes every timestep and always transferred zero chunks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]

__all__ = ["CubeScopeResult", "run_cube_scope"]

#: static/volatile split (test_cuda_residency's RAD_DAY_* spelling).
RAD_DAY_STATIC = ("buildings", "aspect", "wallbol", "alb_grid", "emis_grid",
                  "svfbuveg", "diffsh", "sh_pb", "veg_pb", "vbsh_pb", "alb")
RAD_DAY_VOLATILE = ("sun_pb", "shd_pb", "dp_rank", "guard_true", "shadow",
                    "sunwall", "albshadow", "Lup_pre", "gvflup_extra",
                    "lv2", "ster", "psin", "pcos", "lumChi", "lsky_d2",
                    "lsky_s2", "card_e", "card_s", "card_w", "card_n",
                    "ccos_e", "ccos_s", "ccos_w", "ccos_n",
                    "walk_az_low", "walk_az_high", "walk_az_branch",
                    "walk_dy", "walk_dx", "jE", "jS", "jW", "jN",
                    "F_sh", "Tg_plane", "m_lup_in", "m_e_in", "m_s_in",
                    "m_w_in", "m_n_in", "m_tg_in")
RAD_DAY_NEXT = ("n_lup", "n_e", "n_s", "n_w", "n_n", "n_tg")
RAD_NIGHT_STATIC = ("sh_pb", "veg_pb", "vbsh_pb")
RAD_NIGHT_VOLATILE = ("night_Lup", "ster", "psin", "pcos", "lsky_d2",
                      "lsky_s2", "card_e", "card_s", "card_w", "card_n",
                      "ccos_e", "ccos_s", "ccos_w", "ccos_n")


def _load_sibling(name: str):
    """Load a native/cuda sibling ONCE under 'sw_cuda_<name>', sharing the
    single 'sw_cuda_host' identity (DeviceArray/runtime caches never cross
    module copies; residency's contract)."""
    key = f"sw_cuda_{name}"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, PKG_DIR / f"{name}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


def _load_rad_bundle_builder():
    """The T12-pinned bundle assembly (tests/ultrafast/sw_rad_bundle.py) —
    single-sourced from the file the T12/T13 gates pin, never duplicated
    here (a copy could drift from the pinned assembly)."""
    key = "sw_cuda_rad_bundle"
    mod = sys.modules.get(key)
    if mod is None:
        path = REPO_ROOT / "tests" / "ultrafast" / "sw_rad_bundle.py"
        spec = importlib.util.spec_from_file_location(key, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


@dataclass
class CubeScopeResult:
    """One profile's cube-scope run: the three published cubes, per-t
    digests (tmrt = kernel plane; utci_mat = the PRE-scatter -999 plane
    the oracle's per-t utci digest covers; shadow = published plane),
    the final carried state and CI series, and transfer/timing facts."""

    profile_id: str
    cubes: dict[str, np.ndarray]
    per_t_digests: dict[str, list[str]]
    ret_CI_series: list[float]
    final_firstdaytime: int
    final_timeadd: float
    n_timesteps: int
    h2d_bytes: int
    timings_ms: list[float] = field(default_factory=list)


def run_cube_scope(rt, cap_dir, met_path, *, latitude: float,
                   chunk_rows: int = 128) -> CubeScopeResult:
    """Drive the 24-t solver loop on ONE CUDA profile.

    ``cap_dir``: the t08 capture bundle (frozen geometry + per-t inputs).
    ``met_path``: the site met table (np.loadtxt semantics, 1 header
    row; cols 9/10/11 = Ws/RH/Ta, col 24 = uhii when present) — the
    capture records no wind column, and the utci stage consumes the SAME
    table the oracle read.
    """
    cap_dir = Path(cap_dir)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from solweig_core.numba_cpu.met_recompute import (
        met_from_capture, recompute_timestep, site_params_from_capture,
        time_geom_from_capture,
    )
    from solweig_core.numba_cpu.radiation import (
        RadLoopState, dense_diffsh_from_packed, rad_bundle_from_capture,
        rad_state_from_capture, rad_static_from_capture,
    )
    # the exact override assembly full_solve_capture applies (imported,
    # never reimplemented — drift here would desync from the T11 identity)
    from solweig_core.numba_cpu.full_solve import (
        _apply_day_overrides, _apply_night_overrides,
    )

    # _load_sibling("host") registers sys.modules["sw_cuda_host"] when
    # absent — residency's single-identity lookup contract
    host = _load_sibling("host")
    residency = _load_sibling("residency")
    rad_bundle = _load_rad_bundle_builder()

    man = json.loads((cap_dir / "manifest.json").read_text())
    n = int(man["n_timesteps"])
    rows = int(man["grid"]["rows"])
    cols = int(man["grid"]["cols"])
    st = rad_static_from_capture(cap_dir)
    assert (st.rows, st.cols) == (rows, cols)
    if st.diffsh is None:
        # T17: materialize the dense diffsh cube ONCE — with the pinned
        # builder's own fallback function, so every later bundle's
        # "diffsh" entry is the identical array (the pre-T17 shape
        # recomputed it per day timestep inside build_rad_day_bundle).
        st.diffsh = dense_diffsh_from_packed(st)

    site = site_params_from_capture(np.load(cap_dir / "t00.npz"), latitude)
    series = [met_from_capture(np.load(cap_dir / f"t{t:02d}.npz"))
              for t in range(n)]
    met = np.loadtxt(met_path, skiprows=1, dtype=np.float64)
    if met.shape[0] < n:
        raise ValueError(
            f"met table has {met.shape[0]} rows < capture {n} timesteps")
    uhii = (met[:, 24] if met.shape[1] > 24 else np.zeros(n))

    f32 = np.float32
    met_planes = [
        (np.full((rows, cols), f32(f32(met[t, 11]) + f32(uhii[t])),
                 dtype=np.float32),
         np.full((rows, cols), f32(met[t, 10]), dtype=np.float32),
         np.full((rows, cols), np.maximum(f32(0.15), f32(met[t, 9])),
                 dtype=np.float32))
        for t in range(n)
    ]

    valid = st.buildings == 1
    state: RadLoopState = rad_state_from_capture(cap_dir, 0, rows=rows,
                                                 cols=cols)
    ci_thread = float(state.CI)
    ci_flavor_is_tensor = False

    cubes = {name: np.empty((n, rows, cols), dtype=np.float32)
             for name in ("shadow", "tmrt", "utci")}
    per_t = {name: [] for name in ("shadow", "tmrt", "utci_mat")}
    ret_ci: list[float] = []
    timings: list[float] = []

    runner = residency.ResidentRadRunner(rt, chunk_rows=chunk_rows)
    # T17: statics bind once per phase-kind (the bundles' static entries
    # are pure ``st`` references in the pinned builder — scene-invariant
    # across timesteps), never rebound per t afterwards.
    day_statics_bound = False
    night_statics_bound = False
    try:
        for t in range(n):
            t0 = time.perf_counter()
            z = np.load(cap_dir / f"t{t:02d}.npz")
            tg = time_geom_from_capture(z)
            t_in = rad_bundle_from_capture(cap_dir, t)
            shadow_arg = (z["sunon_in_shadow"]
                          if t_in.is_day and "sunon_in_shadow" in z.files
                          else None)
            res = recompute_timestep(site, series[t], tg, ci_thread,
                                     shadow_arg,
                                     CI_thread_is_tensor=ci_flavor_is_tensor)
            if t_in.is_day:
                _apply_day_overrides(t_in, res, series[t])
            else:
                _apply_night_overrides(t_in, res, series[t])

            if t_in.is_day:
                b = rad_bundle.build_rad_day_bundle(st, t_in, state)
                if not day_statics_bound:
                    runner.bind({k: b[k] for k in RAD_DAY_STATIC},
                                static=True)
                    day_statics_bound = True
                runner.bind({k: b[k] for k in RAD_DAY_VOLATILE})
                lease = runner.run_day(b, download=("tmrt",) + RAD_DAY_NEXT)
                lease.wait()
                pl = lease.publish()
                # _fused_day_host's scalar carry, mirrored exactly
                ta_in = state.timeadd
                tsd = t_in.timestepdec_f64
                thr = 59 / 1440
                if ta_in >= thr:
                    ta_out = tsd if tsd > thr else 0.0
                else:
                    ta_out = ta_in + tsd
                state = RadLoopState(
                    firstdaytime=0, timeadd=ta_out,
                    CI=(state.CI if t_in.CI_out is None
                        else np.float32(t_in.CI_out)),
                    Tgmap1=pl["n_lup"], Tgmap1E=pl["n_e"],
                    Tgmap1S=pl["n_s"], Tgmap1W=pl["n_w"],
                    Tgmap1N=pl["n_n"], TgOut1=pl["n_tg"],
                )
                shadow_plane = t_in.shadow
            else:
                b = rad_bundle.build_rad_night_bundle(st, t_in)
                if not night_statics_bound:
                    runner.bind({k: b[k] for k in RAD_NIGHT_STATIC},
                                static=True)
                    night_statics_bound = True
                runner.bind({k: b[k] for k in RAD_NIGHT_VOLATILE})
                lease = runner.run_night(b, download=("tmrt",))
                lease.wait()
                pl = lease.publish()
                state = RadLoopState(
                    firstdaytime=1, timeadd=0.0, CI=state.CI,
                    Tgmap1=state.Tgmap1, Tgmap1E=state.Tgmap1E,
                    Tgmap1S=state.Tgmap1S, Tgmap1W=state.Tgmap1W,
                    Tgmap1N=state.Tgmap1N, TgOut1=state.TgOut1,
                )
                shadow_plane = np.zeros((rows, cols), dtype=np.float32)

            ci_thread = float(res["ret_CI"])
            ci_flavor_is_tensor = bool(res["CI_flavor_is_tensor"])
            ret_ci.append(ci_thread)

            tmrt_plane = np.ascontiguousarray(pl["tmrt"], dtype=np.float32)
            ta_p, rh_p, va_p = met_planes[t]
            utci_mat = host.run_utci_dense(rt, ta_p, rh_p, tmrt_plane, va_p)
            utci_plane = np.full((rows, cols), np.nan, dtype=np.float32)
            utci_plane[valid] = utci_mat[valid]

            cubes["shadow"][t] = shadow_plane
            cubes["tmrt"][t] = tmrt_plane
            cubes["utci"][t] = utci_plane
            per_t["shadow"].append(_digest(shadow_plane))
            per_t["tmrt"].append(_digest(tmrt_plane))
            per_t["utci_mat"].append(_digest(utci_mat))
            timings.append((time.perf_counter() - t0) * 1e3)
    finally:
        h2d = runner.h2d_bytes
        runner.close()

    return CubeScopeResult(
        profile_id=rt.profile_id,
        cubes=cubes,
        per_t_digests=per_t,
        ret_CI_series=ret_ci,
        final_firstdaytime=int(state.firstdaytime),
        final_timeadd=float(state.timeadd),
        n_timesteps=n,
        h2d_bytes=h2d,
        timings_ms=timings,
    )


def _digest(plane: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(plane, dtype=np.float32).tobytes()).hexdigest()
