# SPDX-License-Identifier: GPL-3.0-only
"""T11 met_recompute gates: identity (all 24 t, raw bits), torch-free
purity, and the typed-refusal fences for unsupported edit families.

The identity contract: with UNEDITED met (the capture's own arg_ values)
recompute_timestep regenerates every met-dependent bundle field
bit-exactly (raw uint32/uint64 views — signed zero and NaN payload
distinct, no allclose/1-ULP) for every timestep, with the driver's CI
value+flavor threaded exactly as utci_process.py:967 does.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from solweig_core.numba_cpu import met_recompute as mr

WT = Path(__file__).resolve().parents[2]
CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")
LATITUDE = 30.312645  # site_500 manifest solar_geometry.latitude

DAY_MAP = [
    ("ret_ea__0d", "ea"),
    ("ret_esky__0d", "esky"),
    ("ret_CI__f64", "ret_CI"),
    ("ret_CI_Tg__f64", "CI_Tg"),
    ("ret_CI_TgG__f64", "CI_TgG"),
    ("ret_radI__0d", "radI"),
    ("ret_radD__0d", "radD"),
    ("ret_I0__0d", "I0"),
    ("frozen_ks_sunlit", "ks_sunlit"),
    ("frozen_ks_shaded", "ks_shaded"),
    ("frozen_Ta273_pow4_f64", "ta273_pow4_f64"),
    ("sunon_in_Tgwall__0d", "Tgwall_f64"),
    ("frozen_Lwall", "Lwall32"),
    ("frozen_dp_veg_surface_f64", "dp_veg_surface_f64"),
    ("frozen_dp_shaded_surface_f64", "dp_shaded_surface_f64"),
    ("frozen_dp_sunlit_surface_f64", "dp_sunlit_surface_f64"),
]
DAY_ARR_MAP = [
    ("kside_in_lv", "lv"),
    ("frozen_lumChi", "lumChi"),
    ("dp_in_steradian", "steradian"),
    ("sunon_in_Tg", "Tg"),
    ("frozen_Lup_pre", "Lup_pre"),
    ("frozen_gvflup_extra", "gvflup_extra"),
]
DAY_COL2_MAP = [
    ("dp_in_Lsky_down", "Lsky_down2"),
    ("dp_in_Lsky_side", "Lsky_side2"),
]
NIGHT_MAP = [
    ("ret_ea__0d", "ea"),
    ("ret_esky__0d", "esky"),
    ("ret_CI__f64", "ret_CI"),
    ("frozen_Ta273_pow4_f64", "ta273_pow4_f64"),
    ("frozen_dp_veg_surface_f64", "dp_veg_surface_f64"),
    ("frozen_dp_shaded_surface_f64", "dp_shaded_surface_f64"),
    ("frozen_dp_sunlit_surface_f64", "dp_sunlit_surface_f64"),
    ("frozen_night_water_override", "night_water_override"),
]
NIGHT_ARR_MAP = [
    ("dp_in_steradian", "steradian"),
    ("frozen_night_Lup", "night_Lup"),
]
NIGHT_COL2_MAP = [
    ("dp_in_Lsky_down", "Lsky_down2"),
    ("dp_in_Lsky_side", "Lsky_side2"),
]

#: Mutation anchors for the lead's independent mutation run (t11/mutations
#: README points here): each entry names a seam this file kills.
MUTATION_ANCHORS = {
    "met_recompute.patch_steradian.reciprocal_mult":
        "test_identity_all_timesteps_raw_bits",
    "met_recompute.lsky_tables.f32_chain_true_pi_div":
        "test_identity_all_timesteps_raw_bits",
    "met_recompute.create_patch_tables_2.aziint_reciprocal_mult":
        "test_identity_all_timesteps_raw_bits",
    "met_recompute.perez_v3.f64_zen_deg_left_assoc":
        "test_identity_all_timesteps_raw_bits",
    "met_recompute.check_site_support.refusals":
        "test_refusal_landcover_water_bodies",
    "met_recompute.met_from_capture.Twater_sentinel":
        "test_night_water_override_sentinel",
}


def capture_present() -> bool:
    return (CAP / "t00.npz").exists()


def _bits(a):
    a = np.asarray(a)
    if a.dtype == np.float64:
        return a.view(np.uint64)
    if a.dtype == np.float32:
        return a.view(np.uint32)
    return a


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_identity_all_timesteps_raw_bits() -> None:
    """UNEDITED met regenerates every met-dependent field bit-exactly for
    ALL 24 timesteps (day+night), CI value+flavor threaded driver-faithful.
    """
    z_static = np.load(CAP / "t00.npz", allow_pickle=True)
    site = mr.site_params_from_capture(z_static, LATITUDE)
    ci_thread = 1.0
    ci_thread_is_tensor = False
    mism: list[str] = []
    n_t = json.loads((CAP / "manifest.json").read_text())["n_timesteps"]
    for t in range(n_t):
        z = z_static if t == 0 else np.load(CAP / f"t{t:02d}.npz",
                                            allow_pickle=True)
        met = mr.met_from_capture(z)
        tg = mr.time_geom_from_capture(z)
        shadow = (z["sunon_in_shadow"]
                  if "sunon_in_shadow" in z.files else None)
        res = mr.recompute_timestep(
            site, met, tg, ci_thread, shadow,
            CI_thread_is_tensor=ci_thread_is_tensor)
        if "arg_CI__f64" in z.files and t > 0 \
                and float(z["arg_CI__f64"]) != ci_thread:
            mism.append(f"t{t:02d}/arg_CI__f64 threaded {ci_thread!r} != "
                        f"capture {float(z['arg_CI__f64'])!r}")
        ci_thread = float(res["ret_CI"])
        ci_thread_is_tensor = bool(res["CI_flavor_is_tensor"])
        maps = ((DAY_MAP, DAY_ARR_MAP, DAY_COL2_MAP) if tg.altitude > 0
                else (NIGHT_MAP, NIGHT_ARR_MAP, NIGHT_COL2_MAP))
        for cap_key, res_key in maps[0]:
            if cap_key not in z.files:
                continue
            want, got = z[cap_key], np.asarray(res[res_key])
            if want.shape != got.shape or want.dtype != got.dtype:
                mism.append(f"t{t:02d}/{cap_key} shape/dtype drift")
            elif not np.array_equal(_bits(want), _bits(got)):
                mism.append(f"t{t:02d}/{cap_key} bits differ")
        for cap_key, res_key in maps[1]:
            if cap_key not in z.files:
                continue
            want, got = z[cap_key], np.asarray(res[res_key])
            if not np.array_equal(_bits(want), _bits(got)):
                mism.append(f"t{t:02d}/{cap_key} bits differ")
        for cap_key, res_key in maps[2]:
            if cap_key not in z.files:
                continue
            want, got = z[cap_key][:, 2], np.asarray(res[res_key])
            if not np.array_equal(_bits(want), _bits(got)):
                mism.append(f"t{t:02d}/{cap_key}[:,2] bits differ")
    assert not mism, "; ".join(mism)


# ---------------------------------------------------------------------------
# purity: the runtime path never imports torch (AST + subprocess)
# ---------------------------------------------------------------------------

RUNTIME_MODULES = [
    "solweig_core/numba_cpu/met_recompute.py",
    "solweig_core/numba_cpu/full_solve.py",
    "solweig_core/numba_cpu/fallback_router.py",
]


def test_runtime_modules_never_name_torch() -> None:
    for rel in RUNTIME_MODULES:
        src = (WT / rel).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] != "torch", f"{rel}: torch"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root != "torch", f"{rel}: torch"


def test_runtime_import_torch_free_subprocess() -> None:
    code = (
        "import sys;"
        "import solweig_core.numba_cpu.full_solve,"
        "solweig_core.numba_cpu.fallback_router;"
        "assert 'torch' not in sys.modules, 'torch leaked';"
        "print('PURE')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(WT))
    assert out.returncode == 0, out.stderr
    assert "PURE" in out.stdout


# ---------------------------------------------------------------------------
# typed refusals: unsupported edit families are never silent no-ops
# ---------------------------------------------------------------------------

def _site(**overrides) -> mr.SiteParams:
    z = np.load(CAP / "t00.npz", allow_pickle=True)
    site = mr.site_params_from_capture(z, LATITUDE)
    if overrides:
        site = mr.SiteParams(**{**vars(site), **overrides})
    return site


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_refusal_landcover_water_bodies() -> None:
    """RED witness: a live-water site (landcover==1) must raise, not
    silently no-op the edit."""
    with pytest.raises(mr.MetEditRefusal, match="landcover == 1"):
        mr.check_site_support(_site(landcover=1))


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
@pytest.mark.parametrize("field,value,needle", [
    ("onlyglobal", 0, "onlyglobal"),
    ("anisotropic_sky", 0, "anisotropic_sky"),
    ("patch_option", 1, "patch_option"),
    ("elvis", 2, "elvis"),
])
def test_refusal_unported_model_branches(field, value, needle) -> None:
    with pytest.raises(mr.MetEditRefusal, match=needle):
        mr.check_site_support(_site(**{field: value}))


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_refusal_perez_negative_altitude() -> None:
    """Perez altitude < 0 (complex-path AirMass) is a typed refusal."""
    pa, pz = mr.create_patch_tables_2()
    with pytest.raises(mr.MetEditRefusal, match="altitude < 0"):
        mr.perez_v3(zen_deg_f=95.0, azi_deg_f=180.0,
                    radD32=np.float32(100.0), radI32=np.float32(300.0),
                    jday_f=172.0, alt_deg32_arr=pa, azi_deg32_arr=pz)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_refusal_recompute_timestep_surfaces_site_checks() -> None:
    z = np.load(CAP / "t08.npz", allow_pickle=True)
    site = _site(landcover=1)
    with pytest.raises(mr.MetEditRefusal):
        mr.recompute_timestep(site, mr.met_from_capture(z),
                              mr.time_geom_from_capture(z), 1.0, None)


# ---------------------------------------------------------------------------
# frozen-bit sentinels
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_night_water_override_sentinel() -> None:
    """The empty-list Twater sentinel records the 0.0 bit — never a
    guessed water temperature."""
    z = np.load(CAP / "t00.npz", allow_pickle=True)
    met = mr.met_from_capture(z)
    assert met.Twater_is_list is True
    res = mr.recompute_timestep(
        _site(), met, mr.time_geom_from_capture(z), 1.0, None)
    assert np.asarray(res["night_water_override"]).view(np.uint32) \
        == np.uint32(0)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_twater_advance_is_numpy_mean_bits() -> None:
    """Twater midnight advance == numpy blocked-pairwise f64 mean of the
    day's Ta rows (torch-free; the f64 reduction is bit-identical to the
    oracle's numpy)."""
    rng = np.random.default_rng(11)
    rows = rng.standard_normal(24)
    assert mr.twater_advance(rows) == float(np.mean(rows))
    with pytest.raises(mr.MetEditRefusal):
        mr.twater_advance(np.zeros(0, dtype=np.float64))
