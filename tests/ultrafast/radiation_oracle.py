# SPDX-License-Identifier: GPL-3.0-only
"""T08 radiation capture producer (torch allowed HERE; runtime stays clean).

Records every radiation-chain intermediate of a LIVE oracle solve (pinned
worktree) and serializes, per timestep, the frozen bundle + stage records
the torch-free runtime consumes and the gates compare against.

Freeze-set rationale (artifacts/t08/probe_numeric_context.py,
probe_pow4_source.py, probe_pow4_0dim.py):
  * CLEAN (numpy == torch == numba on used domains): plane add/sub/mul/div,
    weak-scalar casts (python float / 0-dim f32 tensor -> fl32), min/max
    (except torch maximum -0.0 tie + NaN canonicalization -- mirrored),
    eq/lt bools, sqrt, round.
  * HAZARD (frozen as captured bits): plane ** 4 AND 0-dim ** 4 (torch
    SLEEF powf matches NO torch-free recompute: libm powf 1-ULP off on
    ~2%, multichain ~50%), sin/cos/tan/exp/log (T01), 10 ** x (ea).
  * Frozen per t: Tg, Tgwall, sunonsurface Lup (pre/post water mutation),
    Lwall, define_patch sunlit_surface plane, F_sh (post NaN->0.5),
    sunlit/shaded bool cubes (packed), lv, Lsky tables, steradian, lumChi,
    cardinal tables, scalar chain bits (ea, esky, I0, CI, radI, radD,
    radI0, corr, CI_Tg, CI_TgG, SNUP, weight1, (Ta+273.15)**4, sin/cos
    altitude, night water-override bit).
  * Stage returns recorded per t for the raw-equality gates; deep timesteps
    additionally record per-j sunonsurface returns and 4 sky_masks-variant
    define_patch component sub-calls (sky-only / veg-only / refl-only /
    mask_sun-only -- bit-safe: accumulators are +0.0-only and all added
    values are positive).

Usage:
    .venv/bin/python tests/ultrafast/radiation_oracle.py \
        --artifacts .../t08 --deep 5,12,23 capture
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

ORACLE_TREE = Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc")
DEFAULT_ARTIFACTS = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08")
DATE_STR = "2009-08-11"

# Args whose values are timestep-INVARIANT big cubes; record at t==0 only
# (bit-identical at every t by construction -- static scene tensors).
HEAVY_STATIC_ARGS = {"diffsh", "shmat", "vegshmat", "vbshvegshmat", "asvf",
                     "svfalfa", "svfbuveg", "dsm", "vegdem", "vegdem2",
                     "svf", "svfveg", "svfaveg"}

SOLWEIG_RETURNS = [
    "Tmrt", "Kdown", "Kup", "Ldown", "Lup", "Tg", "ea", "esky", "I0",
    "CI", "shadow", "firstdaytime", "timestepdec", "timeadd", "Tgmap1",
    "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N", "Keast", "Ksouth",
    "Kwest", "Knorth", "Least", "Lsouth", "Lwest", "Lnorth", "KsideI",
    "TgOut1", "TgOut", "radI", "radD", "Lside", "L_patches", "CI_Tg",
    "CI_TgG", "KsideD", "dRad", "Kside",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_oracle_imports() -> dict:
    """Import solweig_gpu from the pinned oracle worktree (T00 model)."""
    tree = str(ORACLE_TREE.resolve())
    if tree not in sys.path:
        sys.path.insert(0, tree)
    import solweig_gpu

    origin = Path(solweig_gpu.__file__).resolve().parent
    if str(origin) != str(ORACLE_TREE.resolve() / "solweig_gpu"):
        raise SystemExit(
            f"solweig_gpu resolved to {origin}, expected {ORACLE_TREE / 'solweig_gpu'}"
        )
    import torch

    return {
        "solweig_gpu_file": str(origin),
        "oracle_tree": tree,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "source_sha256": {
            rel: sha256_file(origin / rel)
            for rel in ("shadow.py", "solweig.py", "utci_process.py")
        },
    }


def pack_bool_cube(cube_u8: np.ndarray) -> np.ndarray:
    """(rows, cols, P) {0,1} uint8 -> bitplanes-packed uint8 cube."""
    from solweig_core import bitplanes

    return bitplanes.pack_bits(cube_u8).data


# --------------------------------------------------------------------------
# Recorder with incremental per-t serialization + pruning
# --------------------------------------------------------------------------

class RadiationRecorder:
    """Wraps oracle functions; records args/returns tagged by timestep.

    The Solweig_2022a_calc wrapper is the timestep counter; right after each
    call returns, `on_timestep` fires, which serializes that timestep's
    payload to disk and prunes older records (memory stays bounded to ~2
    timesteps of planes).
    """

    def __init__(self, deep_ts: set[int], cap_dir: Path) -> None:
        self.t_index = -1
        self.records: dict[str, list] = {}
        self._orig: dict[tuple, object] = {}
        self.deep_ts = deep_ts
        self.cap_dir = cap_dir
        self.on_timestep = None  # set by driver: fn(t) -> None

    # -- recording ---------------------------------------------------------
    @staticmethod
    def _snap_args(orig, args, kwargs, t, entry) -> dict:
        import torch

        bound = inspect.signature(orig).bind(*args, **kwargs)
        out = {}
        for k, v in bound.arguments.items():
            if k in ("location", "sky_masks", "out_slice"):
                continue
            if k == "precomputed_shadows":
                # march outputs per t (vegsh, sh, wallsh, wallsun,
                # wallshve, facesun) -- runtime inputs, keep every t
                if isinstance(v, tuple):
                    entry["precomputed"] = [
                        RadiationRecorder._snap(x) for x in v
                    ]
                continue
            if t > 0 and k in HEAVY_STATIC_ARGS:
                continue
            out[k] = RadiationRecorder._snap(v)
        return out

    @staticmethod
    def _snap(x):
        import torch

        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return ("0d", x.item(), str(x.dtype).split(".")[-1])
            return np.ascontiguousarray(x.detach().cpu().numpy()).copy()
        if isinstance(x, (bool, int, float)):
            return ("py", x)
        return ("other", type(x).__name__)

    def wrap(self, module, name: str, counter: bool = False,
             always_returns: bool = True) -> None:
        import functools

        orig = getattr(module, name)
        rec = self
        self._orig[(module, name)] = orig

        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            if counter:
                rec.t_index += 1
            t = rec.t_index
            entry = {"t": t}
            entry["args"] = rec._snap_args(orig, args, kwargs, t, entry)
            result = orig(*args, **kwargs)
            keep = always_returns or t in rec.deep_ts
            if keep:
                if isinstance(result, tuple):
                    entry["returns"] = [rec._snap(r) for r in result]
                else:
                    entry["returns"] = [rec._snap(result)]
            rec.records.setdefault(name, []).append(entry)
            if counter and rec.on_timestep is not None:
                rec.on_timestep(t)
            return result

        setattr(module, name, wrapper)

    def entries(self, name: str, t: int) -> list[dict]:
        return [r for r in self.records.get(name, []) if r["t"] == t]

    def prune_before(self, t: int) -> None:
        for name in self.records:
            self.records[name] = [r for r in self.records[name]
                                  if r["t"] >= t - 1]

    def restore(self) -> None:
        for (module, name), orig in self._orig.items():
            setattr(module, name, orig)
        self._orig.clear()


# --------------------------------------------------------------------------
# Payload assembly
# --------------------------------------------------------------------------

def _enc(payload: dict, key: str, val) -> None:
    """Encode a snapped value into npz-compatible entries."""
    if isinstance(val, np.ndarray):
        payload[key] = val
        return
    tag = val[0]
    if tag == "0d":
        payload[f"{key}__0d"] = np.float32(val[1]) if val[2] == "float32" else np.float64(val[1])
        payload[f"{key}__dtype"] = val[2]
    elif tag == "py":
        v = val[1]
        if isinstance(v, bool):
            payload[f"{key}__bool"] = np.int8(v)
        elif isinstance(v, int):
            payload[f"{key}__int"] = np.int64(v)
        elif isinstance(v, float):
            payload[f"{key}__f64"] = np.float64(v)
        else:
            payload[f"{key}__other"] = str(v)


def _tt(torch, v):
    """Snapped recorder value -> torch tensor (0-dim for scalars)."""
    if isinstance(v, tuple):
        return torch.tensor(float(v[1]))
    return torch.from_numpy(np.array(v, dtype=np.float32))


def derive_lup_lwall(torch, args0):
    """sunonsurface entry Lup/Lwall, VERBATIM source expressions, from the
    recorded j=0 args (pre water-mutation Tg).

    Dtype contexts (sunon_in_*__dtype pins):
      * Ta arrives as an f64 0-DIM TENSOR, Tgwall as f64 (day) / int64
        (night) 0-dim. 0-dim x 0-dim promotes fully to f64, so
        (Ta + 273.15) ** 4 and the whole Lwall chain are F64 (torch f64
        pow == libm pow, probe_powf64), rounded once where an f32 PLANE
        consumes them (torch casts the f64 0-dim operand to f32 FIRST
        when the other side is an f32 plane -- probe_promotion).
      * the (Tg * shadow + Ta + 273.15) ** 4 term stays an f32 PLANE
        (dimensioned f32 wins, Ta cast-first) -- torch SLEEF plane
        powf, replayed through torch here.
      * ewall is RE-TENSORIZED at sunonsurface top (line 114):
        torch.tensor(ewall) -> f32 0-dim, so SBC * ewall is an f32
        0-dim product before the f64 chain multiplies in.
    Lup -> f32 plane; Lwall -> f64 0-dim consumed as tempbub (f32
    plane) * Lwall with cast-first rounding, i.e. fl32(Lwall).
    """
    SBC = torch.tensor(5.67051e-8)
    emis_grid = _tt(torch, args0["emis_grid"])
    Tg = _tt(torch, args0["Tg"])
    shadow = _tt(torch, args0["shadow"])
    Ta_v = args0["Ta"]
    Ta_f = float(Ta_v[1]) if isinstance(Ta_v, tuple) else float(np.array(Ta_v))
    Ta64 = torch.tensor(Ta_f, dtype=torch.float64)
    ta273_64 = (Ta64 + 273.15) ** 4
    ewall = torch.tensor(float(args0["ewall"][1]))  # re-tensorized f32 0-dim
    Tgw = args0["Tgwall"]
    Tgw_f = float(Tgw[1]) if isinstance(Tgw, tuple) else float(np.array(Tgw))
    # int64 night / f64 day both promote to f64 against Ta64 (exact 0.0)
    Tgwall64 = torch.tensor(Tgw_f, dtype=torch.float64)
    Lup = SBC * emis_grid * (Tg * shadow + Ta64 + 273.15) ** 4 \
        - SBC * emis_grid * ta273_64
    Lwall64 = SBC * ewall * (Tgwall64 + Ta64 + 273.15) ** 4 \
        - SBC * ewall * ta273_64
    return Lup.numpy(), np.float32(Lwall64.item())


def derive_gvflup_extra(torch, sunon_args_j0, post_Tg):
    """Line-277 gvfLup extra term with POST-mutation Tg (verbatim)."""
    SBC = torch.tensor(5.67051e-8)
    emis_grid = _tt(torch, sunon_args_j0["emis_grid"])
    shadow = _tt(torch, sunon_args_j0["shadow"])
    Ta_v = sunon_args_j0["Ta"]
    Ta_f = float(Ta_v[1]) if isinstance(Ta_v, tuple) else float(np.array(Ta_v))
    Ta64 = torch.tensor(Ta_f, dtype=torch.float64)
    ta273_64 = (Ta64 + 273.15) ** 4
    buildings = _tt(torch, sunon_args_j0["buildings"])
    Tg = torch.from_numpy(np.asarray(post_Tg, dtype=np.float32))
    extra = ((SBC * emis_grid * (Tg * shadow + Ta64 + 273.15) ** 4)
             - SBC * emis_grid * ta273_64) * (buildings * -1 + 1)
    return extra.numpy()


def derive_dp_surfaces(torch, dp_args):
    """define_patch vegetation/shaded/sunlit surfaces (verbatim).

    Dtype contexts (dp_in_*__dtype pins): Ta f64 0-dim, Tgwall f64 (day)
    / int64 (night) 0-dim, ewall RE-TENSORIZED f32 0-dim (line 1505).
    0-dim x 0-dim promotes fully to f64:
      (ewall * SBC) -> f32 0-dim c = fl32(fl32(0.9) * fl32(SBC));
      ((Ta + 273.15) ** 4) -> f64 0-dim (torch f64 pow == libm pow);
      (c * pow4) -> f64 0-dim; (/ torch.tensor(np.pi)) -> f64 0-dim
      (0-dim / 0-dim promotes; pi f32 promoted exactly).
    vegetation_surface == shaded_surface bit-identically (same expr).
    The dp accumulators consume these f64 0-dims against BOOL mask
    planes -> f64 planes -> `f32 += f64 plane` sums in f64 and rounds
    once per add (probe_inplace), so the F64 values are authoritative.
    """
    SBC = torch.tensor(5.67051e-8)
    Ta = dp_args["Ta"]
    Ta_f = float(Ta[1]) if isinstance(Ta, tuple) else float(np.array(Ta))
    Ta64 = torch.tensor(Ta_f, dtype=torch.float64)
    ta273_64 = (Ta64 + 273.15) ** 4
    ewall = torch.tensor(float(dp_args["ewall"][1]))  # f32 0-dim
    vegetation_surface = ((ewall * SBC * ta273_64) / torch.tensor(np.pi))
    Tgwall = dp_args["Tgwall"]
    Tgw_f = float(Tgwall[1]) if isinstance(Tgwall, tuple) else float(np.array(Tgwall))
    Tgwall64 = torch.tensor(Tgw_f, dtype=torch.float64)
    sunlit = ((ewall * SBC * ((Ta64 + Tgwall64 + 273.15) ** 4))
              / torch.tensor(np.pi))
    return float(vegetation_surface.item()), float(sunlit.item())


def frozen_tsw_weight1(torch, timeadd_f: float, timestepdec_f: float) -> float:
    """weight1 bits for one TsWaveDelay call (verbatim branch order).

    Source order: firstdaytime reset does NOT touch timeadd/weight1; the
    >= branch computes w on the ENTRY timeadd; the < branch advances
    timeadd by timestepdec FIRST and computes w on the NEW timeadd.
    """
    if timeadd_f >= (59 / 1440):
        w = torch.exp(-33.27 * torch.tensor(timeadd_f))
    else:
        w = torch.exp(-33.27 * torch.tensor(timeadd_f + timestepdec_f))
    return w.item()


def build_timestep_payload(rec: RadiationRecorder, t: int, torch,
                           orig_define) -> tuple[dict, dict]:
    sol = rec.entries("Solweig_2022a_calc", t)[0]
    sa, sr = sol["args"], sol["returns"]
    payload: dict = {}
    summary: dict = {}

    for name, val in zip(SOLWEIG_RETURNS, sr):
        _enc(payload, f"ret_{name}", val)

    for key in sa:
        _enc(payload, f"arg_{key}", sa[key])

    # frozen scalar bits (torch replay, verbatim source expressions)
    alt_v = sa["altitude"]
    alt_f = float(alt_v[1]) if isinstance(alt_v, tuple) else float(np.array(alt_v))
    # Solweig-body `altitude` is an f64 0-DIM TENSOR by the radiation
    # section (kup_in_altitude__dtype = float64); Kup AND Kdown evaluate
    # torch.sin(altitude * (np.pi / 180)) as an F64 chain (0-dim x weak
    # python f64 pi) -> f64 sin. torch f64 sin != numpy f64 sin on this
    # host (probe_sinf64), so the f64 SIN bit itself is frozen; the f32
    # consumers (gvfalb * radI plane, cast-first) see fl32(sin_f64).
    alt64 = torch.tensor(alt_f, dtype=torch.float64)
    payload["frozen_sinalt_f64"] = np.float64(
        torch.sin(alt64 * (np.pi / 180.)).item())
    # cosalt context: Kside receives `altitude.item()` (python float,
    # kside_in_altitude__f64 = "py") and RE-TENSORIZES at line 636
    # `altitude = torch.tensor(altitude)` -> f32 0-dim; deg2rad = f32
    # TENSOR fl32(pi/180). f32 SLEEF cosf chain.
    payload["frozen_cosalt"] = np.float32(
        torch.cos(torch.tensor(alt_f) * torch.tensor(np.pi / 180.)).item())
    # (Ta + 273.15) ** 4: Ta is an f64 0-DIM TENSOR inside every stage
    # (dp/sunon/lcyl_in_Ta__dtype = float64) -> F64 pow (torch f64 pow
    # == libm pow exactly, probe_powf64); f32-plane consumers see
    # fl32(pow4_f64) cast-first (gvf tail, Lup/Lwall second terms).
    Ta_v = sa["Ta"]
    Ta_f = float(Ta_v[1]) if isinstance(Ta_v, tuple) else float(np.array(Ta_v))
    Ta64 = torch.tensor(Ta_f, dtype=torch.float64)
    payload["frozen_Ta273_pow4_f64"] = np.float64(((Ta64 + 273.15) ** 4).item())
    # night: final Lup plane (recorded return, post water-override) +
    # override bit replay. Twater may still be the initial empty python
    # LIST (never established by a midnight row); the oracle then
    # broadcasts an EMPTY tensor into `Lup[lc_grid == 3]`, which only
    # evaluates on a water-free grid -- record the flag honestly.
    if alt_f <= 0 and isinstance(sr[4], np.ndarray):
        payload["frozen_night_Lup"] = sr[4]
        Tw_v = sa["Twater"]
        if isinstance(Tw_v, tuple):
            payload["frozen_Twater_is_list"] = np.int8(1)
            payload["frozen_Twater_used_f64"] = np.float64(0.0)
            payload["frozen_night_water_override"] = np.float32(0.0)
        else:
            payload["frozen_Twater_is_list"] = np.int8(0)
            Tw = float(np.array(Tw_v))
            payload["frozen_Twater_used_f64"] = np.float64(Tw)
            Twater_t = torch.tensor(Tw)
            SBC = torch.tensor(5.67051e-8)
            override = (SBC * 0.98 * (Twater_t + 273.15) ** 4).float()
            payload["frozen_night_water_override"] = np.float32(override.item())

    # stage args/returns
    for fname, prefix in (
        ("gvf_2018a", "gvf"), ("Kup_veg_2015a", "kup"),
        ("Kside_veg_v2022a", "kside"), ("Lcyl_v2022a", "lcyl"),
        ("define_patch_characteristics", "dp"),
        ("Lside_veg_v2022a", "lsideveg"),
    ):
        entries = rec.entries(fname, t)
        if not entries:
            continue
        e = entries[0]
        for k, v in e["args"].items():
            _enc(payload, f"{prefix}_in_{k}", v)
        if "returns" in e:
            for i, rv in enumerate(e["returns"]):
                _enc(payload, f"{prefix}_ret{i}", rv)

    # Kside sunlit/shaded surfaces (verbatim; albedo_b python 0.2 weak).
    # radI/radD here are the KSIDE ARGS (f32 0-dim, body-recomputed via
    # diffusefraction -- the Solweig-level arg_radI/radD stay the -999
    # onlyglobal sentinels at day). Kside re-tensorizes altitude to f32
    # 0-dim at line 636; cos(altitude * deg2rad) is an f32 SLEEF chain
    # and (radI * cos) an f32 0-dim product -- all-f32 expression.
    if "kside_in_radI__0d" in payload:
        radI_f = float(payload["kside_in_radI__0d"])
        radD_f = float(payload["kside_in_radD__0d"])
        albedo_b = float(sa["albedo_b"][1])
        alt32 = torch.tensor(alt_f)
        ks_sun = ((albedo_b * (torch.tensor(radI_f)
                               * torch.cos(alt32 * torch.tensor(np.pi / 180.)))
                   + (torch.tensor(radD_f) * 0.5)) / torch.pi)
        ks_shd = ((albedo_b * torch.tensor(radD_f) * 0.5) / torch.pi)
        payload["frozen_ks_sunlit"] = np.float32(ks_sun.item())
        payload["frozen_ks_shaded"] = np.float32(ks_shd.item())

    # frozen patch trig tables (per-idx verbatim replay of the oracle's
    # 0-dim expressions: torch.sin(pa[idx] * deg2rad) with deg2rad an f32
    # TENSOR -- sin/cos are HAZARD ops, so these bits are authoritative)
    if "dp_in_patch_altitude" in payload:
        pa_t = torch.from_numpy(payload["dp_in_patch_altitude"].astype(np.float32))
        pz_t = torch.from_numpy(payload["dp_in_patch_azimuth"].astype(np.float32))
        deg2rad_t = torch.tensor(np.pi / 180)
        n_p = pa_t.shape[0]
        psin = np.zeros(n_p, dtype=np.float32)
        pcos = np.zeros(n_p, dtype=np.float32)
        cc = {c: np.zeros(n_p, dtype=np.float32) for c in "eswn"}
        base = {"e": 90, "s": 180, "w": 270, "n": 0}
        for i in range(n_p):
            psin[i] = torch.sin(pa_t[i] * deg2rad_t).item()
            pcos[i] = torch.cos(pa_t[i] * deg2rad_t).item()
            for c, b in base.items():
                cc[c][i] = torch.cos((b - pz_t[i]) * deg2rad_t).item()
        payload["frozen_patch_sin"] = psin
        payload["frozen_patch_cos"] = pcos
        for c in "eswn":
            payload[f"frozen_card_cos_{c}"] = cc[c]

    # frozen lumChi lands after the Perez lv block below (needs `lv`).

    # TsWaveDelay calls (6 per daytime t) + frozen weight1 bits
    for i, e in enumerate(rec.entries("TsWaveDelay_2015a", t)):
        # TsWaveDelay's first parameter is named gvfLup (aliased to Tgmap0
        # inside the function); keep the source name in the payload keys
        _enc(payload, f"tsw{i}_in_gvfLup", e["args"]["gvfLup"])
        _enc(payload, f"tsw{i}_in_firstdaytime", e["args"]["firstdaytime"])
        _enc(payload, f"tsw{i}_in_timeadd", e["args"]["timeadd"])
        _enc(payload, f"tsw{i}_in_timestepdec", e["args"]["timestepdec"])
        _enc(payload, f"tsw{i}_in_Tgmap1", e["args"]["Tgmap1"])
        if "returns" in e:
            _enc(payload, f"tsw{i}_ret_Lup", e["returns"][0])
            _enc(payload, f"tsw{i}_ret_timeadd", e["returns"][1])
            _enc(payload, f"tsw{i}_ret_Tgmap1", e["returns"][2])
        ta = float(e["args"]["timeadd"][1])
        tsd = float(e["args"]["timestepdec"][1])
        payload[f"tsw{i}_frozen_weight1"] = np.float32(
            frozen_tsw_weight1(torch, ta, tsd)
        )

    # march outputs this t (from precomputed_shadows kwarg)
    if "precomputed" in sol:
        names = ("vegsh", "sh", "wallsh", "wallsun", "wallshve", "facesun")
        for name, val in zip(names, sol["precomputed"]):
            _enc(payload, f"march_{name}", val)

    # cylindric_wedge (F_sh raw) + fixed plane. The module-attribute
    # wrapper does not intercept every solver call site, so the
    # authoritative F_sh is the Kup ARG snapshot (the oracle's post
    # NaN->0.5 plane, the same object Kdown consumes); cylindric_wedge
    # returns stay best-effort.
    cw = rec.entries("cylindric_wedge", t)
    if cw and "returns" in cw[0]:
        _enc(payload, "F_sh_raw", cw[0]["returns"][0])
    if "kup_in_F_sh" in payload:
        fsh = np.array(payload["kup_in_F_sh"], dtype=np.float32).copy()
        fsh[np.isnan(fsh)] = np.float32(0.5)  # no-op when already fixed
        payload["F_sh_fixed"] = fsh

    # shaded_or_sunlit -> packed cubes
    sos = rec.entries("shaded_or_sunlit", t)
    if sos and "returns" in sos[0]:
        sunlit = np.stack([e["returns"][0] for e in sos]).astype(np.uint8)
        shaded = np.stack([e["returns"][1] for e in sos]).astype(np.uint8)
        payload["sunlit_packed"] = pack_bool_cube(np.transpose(sunlit, (1, 2, 0)))
        payload["shaded_packed"] = pack_bool_cube(np.transpose(shaded, (1, 2, 0)))
        summary["n_sos"] = len(sos)

    # Perez lv
    pv = rec.entries("Perez_v3", t)
    if pv:
        _enc(payload, "lv", pv[0]["returns"][0])

    # frozen lumChi (Kside anisotropic internal: radTot loop + weak radD,
    # verbatim replay from the recorded lv bits; radD = the KSIDE ARG
    # f32 0-dim, i.e. the body-recomputed value, NOT the -999 sentinel)
    if "lv" in payload and alt_f > 0 and "kside_in_radD__0d" in payload:
        radD_k = torch.tensor(float(payload["kside_in_radD__0d"]))
        lv_t = torch.from_numpy(payload["lv"].astype(np.float32))
        patch_altitude = lv_t[:, 0]
        patch_luminance = lv_t[:, 2]
        deg2rad_t = torch.tensor(torch.pi / 180.0)
        skyalt, skyalt_c = torch.unique(patch_altitude, return_counts=True)
        radTot = torch.zeros(1)
        steradian = torch.zeros(patch_altitude.shape[0])
        for i in range(patch_altitude.shape[0]):
            if skyalt_c[skyalt == patch_altitude[i]] > 1:
                steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad_t) * (
                    torch.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad_t)
                    - torch.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad_t))
            else:
                steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad_t) * (
                    torch.sin((patch_altitude[i]) * deg2rad_t)
                    - torch.sin((patch_altitude[i - 1] + patch_altitude[0]) * deg2rad_t))
            radTot += (patch_luminance[i] * steradian[i]
                       * torch.sin(patch_altitude[i] * deg2rad_t))
        lumChi = (patch_luminance * radD_k) / radTot
        payload["frozen_lumChi"] = lumChi.numpy().astype(np.float32)

    # sunonsurface j=0 args + frozen Lup/Lwall replays + per-j returns (deep)
    sun = rec.entries("sunonsurface_2018a", t)
    if sun:
        e0 = sun[0]["args"]
        for k in ("Tg", "Tgwall", "shadow", "sunwall", "buildings", "walls",
                  "aspect", "emis_grid", "Ta", "ewall"):
            _enc(payload, f"sunon_in_{k}", e0[k])
        Lup_pre, Lwall = derive_lup_lwall(torch, e0)
        payload["frozen_Lup_pre"] = Lup_pre
        payload["frozen_Lwall"] = Lwall
        post_Tg = None
        for name, val in zip(SOLWEIG_RETURNS, sr):
            if name == "Tg" and isinstance(val, np.ndarray):
                post_Tg = val
        if post_Tg is not None:
            payload["frozen_gvflup_extra"] = derive_gvflup_extra(torch, e0, post_Tg)
        if t in rec.deep_ts:
            for j, e in enumerate(sun):
                if "returns" not in e:
                    continue
                for i, rv in enumerate(e["returns"]):
                    _enc(payload, f"sunon_j{j}_ret{i}", rv)

    # define_patch surfaces (frozen F64 values; the dp accumulators sum
    # them in f64 against bool masks and round once per add -- probe)
    dpe = rec.entries("define_patch_characteristics", t)
    if dpe:
        dargs = dpe[0]["args"]
        veg_s, sunlit_s = derive_dp_surfaces(torch, dargs)
        payload["frozen_dp_veg_surface_f64"] = np.float64(veg_s)
        payload["frozen_dp_shaded_surface_f64"] = np.float64(veg_s)
        payload["frozen_dp_sunlit_surface_f64"] = np.float64(sunlit_s)
    return payload, summary


def dp_component_calls(torch, orig_define, dp_args_t, shmat, vegshmat,
                       vbshvegshmat, deep_payload: dict) -> None:
    """sky_masks-variant define_patch sub-calls (component ground truth).

    Zero-adds are bit-safe: every accumulator starts at +0.0 and only
    receives adds of nonnegative-masked positive values.
    """
    mask_sky = ((shmat == 1) & (vegshmat == 1))
    mask_veg = ((vegshmat == 0) | (vbshvegshmat == 0))
    mask_sun = (((1 - shmat) * vbshvegshmat) == 1)
    mask_refl = ((shmat == 0) | (vegshmat == 0) | (vbshvegshmat == 0))
    F = torch.zeros_like(mask_sky[0])
    variants = {
        "sky": (mask_sky, F, F, F),
        "veg": (F, mask_veg, F, F),
        "sun": (F, F, mask_sun, F),
        "refl": (F, F, F, mask_refl),
    }
    for comp, masks in variants.items():
        outs = orig_define(
            dp_args_t["solar_altitude"], dp_args_t["solar_azimuth"],
            dp_args_t["patch_altitude"], dp_args_t["patch_azimuth"],
            dp_args_t["steradian"], dp_args_t["asvf"],
            shmat, vegshmat, vbshvegshmat,
            dp_args_t["Lsky_down"], dp_args_t["Lsky_side"],
            dp_args_t["Lsky"], dp_args_t["Lup"],
            dp_args_t["Ta"], dp_args_t["Tgwall"], dp_args_t["ewall"],
            dp_args_t["rows"], dp_args_t["cols"],
            sky_masks=masks,
        )
        for i, o in enumerate(outs):
            deep_payload[f"dpcomp_{comp}_ret{i}"] = np.ascontiguousarray(
                o.detach().cpu().numpy()).copy()


# --------------------------------------------------------------------------
# Capture driver
# --------------------------------------------------------------------------

def run_capture(artifacts: Path, deep_ts: list[int]) -> dict:
    origin = pin_oracle_imports()
    import torch

    import solweig_gpu.solweig as sw
    import solweig_gpu.utci_process as up
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
    from solweig_gpu.incremental.solver import (
        compose_full_scene_tensors, read_window_for_write_window, solve_window,
    )
    from solweig_gpu.incremental.trees import TreeLayer
    from solweig_gpu.incremental.worker import ExactWorker

    cap_dir = artifacts / "capture"
    cap_dir.mkdir(parents=True, exist_ok=True)

    rec = RadiationRecorder(set(deep_ts), cap_dir)
    for name in ("gvf_2018a", "Kup_veg_2015a", "Kside_veg_v2022a",
                 "Lcyl_v2022a", "define_patch_characteristics",
                 "Lside_veg_v2022a", "TsWaveDelay_2015a", "cylindric_wedge",
                 "shaded_or_sunlit", "Perez_v3", "daylen"):
        rec.wrap(sw, name)
    # sunonsurface per-j returns only at deep timesteps (memory)
    rec.wrap(sw, "sunonsurface_2018a", always_returns=False)
    rec.wrap(up, "Solweig_2022a_calc", counter=True)
    orig_define = rec._orig[(sw, "define_patch_characteristics")]

    # static cubes captured at t==0 from the Solweig args
    static_state: dict = {}

    def on_timestep(t: int) -> None:
        if t == 0:
            # everything static is captured from the t==0 records BEFORE
            # any pruning can drop them
            sol0 = rec.entries("Solweig_2022a_calc", 0)[0]["args"]
            static_state["shmat"] = torch.from_numpy(
                np.array(sol0["shmat"], dtype=np.float32))
            static_state["vegshmat"] = torch.from_numpy(
                np.array(sol0["vegshmat"], dtype=np.float32))
            static_state["vbshvegshmat"] = torch.from_numpy(
                np.array(sol0["vbshvegshmat"], dtype=np.float32))
            if isinstance(sol0.get("asvf"), np.ndarray):
                static_state["asvf_t"] = torch.from_numpy(
                    np.array(sol0["asvf"], dtype=np.float32))
            if isinstance(sol0.get("svfbuveg"), np.ndarray):
                static_state["svfbuveg"] = sol0["svfbuveg"].astype(np.float32)
            if isinstance(sol0.get("diffsh"), np.ndarray):
                static_state["diffsh_ref"] = sol0["diffsh"]
            # TgK/Tstart planes + scalar wall/LST coefficients are Solweig
            # args (recorded at t0 only; heavy-static pruning drops later t)
            for key in ("TgK", "Tstart"):
                if isinstance(sol0.get(key), np.ndarray):
                    static_state[key] = sol0[key].astype(np.float32)
            for key in ("TgK_wall", "Tstart_wall", "TmaxLST", "TmaxLST_wall"):
                v = sol0.get(key)
                if isinstance(v, tuple) and v[0] in ("py", "0d"):
                    static_state[key] = np.float32(float(v[1]))
        if "buildings" not in static_state:
            # stash gvf statics at the FIRST gvf call (t0 may be night, so
            # keying on t==0 misses them)
            gvf_here = rec.entries("gvf_2018a", t)
            if gvf_here:
                for key in ("buildings", "walls", "dirwalls", "lc_grid",
                            "alb_grid", "emis_grid", "TgK", "Tstart",
                            "TgK_wall", "Tstart_wall", "TmaxLST",
                            "TmaxLST_wall"):
                    v = gvf_here[0]["args"].get(key)
                    if isinstance(v, np.ndarray):
                        static_state[key] = v.astype(np.float32).copy()
        payload, summary = build_timestep_payload(rec, t, torch, orig_define)
        # deep component sub-calls need torch cube views of the statics
        if t in rec.deep_ts and static_state and "dp_in_solar_altitude" in payload:
            dargs = rec.entries("define_patch_characteristics", t)[0]["args"]
            import torch as _t
            def tt(name):
                v = dargs[name]
                if isinstance(v, tuple):
                    return _t.tensor(float(v[1]))
                return _t.from_numpy(np.array(v, dtype=np.float32))
            dp_t = {
                "solar_altitude": tt("solar_altitude"),
                "solar_azimuth": tt("solar_azimuth"),
                "patch_altitude": tt("patch_altitude"),
                "patch_azimuth": tt("patch_azimuth"),
                "steradian": tt("steradian"),
                "asvf": static_state.get("asvf_t"),
                "Lsky_down": tt("Lsky_down"),
                "Lsky_side": tt("Lsky_side"),
                "Lsky": tt("Lsky"),
                "Lup": tt("Lup"),
                "Ta": tt("Ta"),
                "Tgwall": tt("Tgwall"),
                "ewall": tt("ewall"),
                "rows": dargs["rows"], "cols": dargs["cols"],
            }
            if dp_t["asvf"] is not None:
                dp_component_calls(torch, orig_define, dp_t,
                                   static_state["shmat"],
                                   static_state["vegshmat"],
                                   static_state["vbshvegshmat"], payload)
        np.savez(cap_dir / f"t{t:02d}.npz", **payload)
        print(f"  serialized t{t:02d} ({len(payload)} arrays)", flush=True)
        rec.prune_before(t)

    rec.on_timestep = on_timestep

    cache = SiteCache.load(ORACLE_TREE / "site-cache" / "site_500")
    grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                      cache.manifest.origin_x_m, cache.manifest.origin_y_m)
    layer = TreeLayer(cache.tree_base, grid)
    worker = ExactWorker(cache, layer,
                         site_dir=ORACLE_TREE / "Input_subset" / "processed_inputs",
                         results_root=artifacts / "scratch_results",
                         selected_date_str=DATE_STR)
    forcing = worker.forcing()
    scene = compose_full_scene_tensors(cache, layer)
    write = RasterWindow(0, cache.rows, 0, cache.cols)
    read = read_window_for_write_window(write, cache, scene, forcing)

    t0 = time.perf_counter()
    outputs = solve_window(cache, layer, read_window=read, write_window=write,
                           forcing=forcing,
                           requested_variables=("utci", "tmrt", "shadow"),
                           stage_timings={}, scene=scene)
    solve_s = time.perf_counter() - t0
    rec.restore()

    # ---- static scene (collected at t==0, pre-prune) -----------------------
    static = {
        "shmat_packed": pack_bool_cube(static_state["shmat"].numpy().astype(np.uint8)),
        "vegshmat_packed": pack_bool_cube(static_state["vegshmat"].numpy().astype(np.uint8)),
        "vbshvegshmat_packed": pack_bool_cube(static_state["vbshvegshmat"].numpy().astype(np.uint8)),
    }
    for key in ("buildings", "walls", "dirwalls", "lc_grid", "alb_grid",
                "emis_grid", "TgK", "Tstart", "TgK_wall", "Tstart_wall",
                "TmaxLST", "TmaxLST_wall", "svfbuveg"):
        if key in static_state:
            static[key] = static_state[key]
    if "dirwalls" in static:
        static["aspect_rad"] = (
            torch.from_numpy(static["dirwalls"]) * torch.pi / 180
        ).numpy()
    if "walls" in static:
        static["wallbol"] = (torch.from_numpy(static["walls"]) > 0).float().numpy()
    if "diffsh_ref" in static_state:
        np.save(cap_dir / "diffsh_ref.npy", static_state["diffsh_ref"])
    # patch count (np.unpackbits pads to byte multiples; slice consumers)
    if "diffsh_ref" in static_state:
        static["n_patches"] = np.int64(static_state["diffsh_ref"].shape[2])
    np.savez(cap_dir / "static.npz", **static)

    # walk offsets + azimuth statics (torch replay of source expressions)
    azimuthA = torch.arange(5, 359, 20, dtype=torch.float32)
    dy_tab = np.zeros((18, 11), dtype=np.int64)
    dx_tab = np.zeros((18, 11), dtype=np.int64)
    for j in range(18):
        azimuth = azimuthA[j] * (torch.pi / 180)
        pibyfour = torch.pi / 4
        threetimespibyfour = 3 * pibyfour
        fivetimespibyfour = 5 * pibyfour
        seventimespibyfour = 7 * pibyfour
        sinazimuth = torch.sin(azimuth)
        cosazimuth = torch.cos(azimuth)
        tanazimuth = torch.tan(azimuth)
        signsinazimuth = torch.sign(sinazimuth)
        signcosazimuth = torch.sign(cosazimuth)
        index = 0
        for n in range(11):
            if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (
                fivetimespibyfour <= azimuth and azimuth < seventimespibyfour
            ):
                dy = signsinazimuth * index
                dx = -1 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            else:
                dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
                dx = -1 * signcosazimuth * index
            dy_tab[j, n] = int(dy.item())
            dx_tab[j, n] = int(dx.item())
            index += 1
    np.savez(cap_dir / "walk_offsets.npz", dy=dy_tab, dx=dx_tab)

    az, lo, hi, br = [], [], [], []
    for j in range(18):
        azimuth = azimuthA[j] * (torch.pi / 180)
        azilow = azimuth - torch.pi / 2
        azihigh = azimuth + torch.pi / 2
        c1 = bool(azilow >= 0 and azihigh < 2 * torch.pi)
        c2 = bool(azilow < 0 and azihigh <= 2 * torch.pi)
        c3 = bool(azilow > 0 and azihigh >= 2 * torch.pi)
        if c2:
            azilow = azilow + 2 * torch.pi
        if c3:
            azihigh = azihigh - 2 * torch.pi
        az.append(np.float32(azimuth))
        lo.append(np.float32(azilow))
        hi.append(np.float32(azihigh))
        br.append(1 if c1 else (2 if c2 else 3))
    np.savez(cap_dir / "azimuth_statics.npz",
             azimuth=np.array(az, dtype=np.float32),
             azilow=np.array(lo, dtype=np.float32),
             azihigh=np.array(hi, dtype=np.float32),
             branch=np.array(br, dtype=np.int64))

    # per-t manifest with digests
    files = {}
    for path in sorted(cap_dir.glob("*.npz")):
        files[path.name] = sha256_file(path)
    manifest = {
        "mode": "capture",
        "origin": origin,
        "date": DATE_STR,
        "solve_wall_s": round(solve_s, 3),
        "n_timesteps": rec.t_index + 1,
        "grid": {"rows": cache.rows, "cols": cache.cols},
        "deep_ts": deep_ts,
        "outputs": {
            k: hashlib.sha256(np.ascontiguousarray(v).view(np.uint8).tobytes()).hexdigest()
            for k, v in outputs.items()
        },
        "files": files,
    }
    (cap_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--deep", type=str, default="5,12,23")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.cmd == "capture":
        deep = [int(x) for x in args.deep.split(",") if x != ""]
        manifest = run_capture(args.artifacts, deep)
        print(json.dumps({k: manifest[k] for k in
                          ("solve_wall_s", "n_timesteps", "grid", "deep_ts")},
                         indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
