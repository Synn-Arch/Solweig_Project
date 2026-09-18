# SPDX-License-Identifier: GPL-3.0-only
"""T08 radiation hoist + ordered fusion tests (RED first).

Raw-bit parity vs the ORIGINAL solweig_gpu radiation path, per DESIGN.ko.md
3.3/6.3/9.2/9.3: uint32 bit comparison of every captured intermediate,
signed zero + NaN payload included, no allclose/equal_nan acceptance.

torch is allowed in THIS file (live-oracle recompute for the frozen-table
pins and the differential fixtures, T07 bench-file precedent); the runtime
module under test (solweig_core.numba_cpu.radiation) must stay torch-free
(pinned by test_radiation_purity_torch_free below).

Site/capture-dependent tests skip when the t08 capture bundle or the
pinned oracle tree is absent; the committed frozen-pin JSON
(radiation_frozen_pin.json) keeps the frozen-input digests self-standing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.ultrafast.bitwise_harness import assert_planes_equal  # noqa: E402

# RED witness: the runtime module did not exist at task start (T08 process
# step 3 — failing-first, recorded under artifacts/t08/red_first_run.txt).
from solweig_core.numba_cpu import radiation  # noqa: E402

ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08")
CAP = ART / "capture"
ORACLE_TREE = Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc")
PIN = Path(__file__).parent / "radiation_frozen_pin.json"

#: timesteps exercising every code path: night, first-daytime (fd==1),
#: deep midday, the Kside/dp sos knife-edge step.
SPOT_TS = (0, 8, 12, 16)

#: RED-witness anchors (mutations/mutations table); each must stay present
#: EXACTLY once so the M1-M5 mutation cycles stay reproducible.
MUTATION_ANCHORS = {
    "M1_reflection_barrier": "refl = (((Ld_sky + lup_out) * ome) * half) / pi32",
    "M2_direction_boundary": "gn2 = ((walbwnosh + walbnosh) / second32) * lis_",
    # T18 K1: Psun_c became the per-t precomputed psun_c[idx] (same
    # scalar-context accumulate, bit-exact CSE) — anchor tracks the spot.
    "M3_scalar_context": "np.float32(np.float64(Ls_sun) + psun_c[idx] * g1)",
    "M4_mask_call_set": "k = kside_n + dp_rank[idx]",
    "M5_halo_walk": "tbu = buildings[rr, cc]",
}


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def capture_present() -> bool:
    return (CAP / "manifest.json").is_file()


def test_radiation_module_imports() -> None:
    """RED-first: the T08 runtime module exists and is importable."""
    assert radiation is not None


def test_radiation_purity_torch_free() -> None:
    """The runtime module must not import torch (AST + subprocess pin).

    AST walk over import statements only -- docstrings legitimately
    mention torch when documenting WHY values are frozen.
    """
    import ast

    tree = ast.parse(Path(radiation.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "torch" not in name, f"torch import: {name}"
    code = (
        "import sys; sys.path.insert(0, {root!r}); "
        "from solweig_core.numba_cpu import radiation; "
        "assert 'torch' not in sys.modules, 'torch leaked into runtime'"
    ).format(root=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_mutation_anchors_unique() -> None:
    """Each M1-M5 anchor appears exactly once in the runtime module."""
    src = Path(radiation.__file__).read_text()
    for name, anchor in MUTATION_ANCHORS.items():
        assert src.count(anchor) == 1, f"{name}: anchor count {src.count(anchor)}"


# ---------------------------------------------------------------------------
# Frozen-input pins (committed digests; self-standing)
# ---------------------------------------------------------------------------

def _digest(arr) -> tuple:
    a = np.ascontiguousarray(arr)
    if a.dtype == np.bool_:
        a = a.view(np.uint8)
    return [hashlib.sha256(a.tobytes()).hexdigest()[:16],
            str(a.dtype), list(a.shape)]


@pytest.mark.skipif(not capture_present() or not PIN.is_file(),
                    reason="t08 capture bundle or frozen pin absent")
def test_frozen_pin_digests() -> None:
    """Loader-consumed frozen inputs match the committed digests."""
    pin = json.loads(PIN.read_text())
    z = np.load(CAP / "static.npz")
    for k, want in pin["static"].items():
        assert _digest(z[k]) == want, f"static/{k}"
    for name, section in (("walk_offsets.npz", "walk"),
                          ("azimuth_statics.npz", "azimuth")):
        zz = np.load(CAP / name)
        for k, want in pin[section].items():
            assert _digest(zz[k]) == want, f"{section}/{k}"
    for t in range(24):
        z = np.load(CAP / f"t{t:02d}.npz")
        tp = pin["timesteps"][f"t{t:02d}"]
        for k, want in tp.items():
            assert _digest(z[k]) == want, f"t{t:02d}/{k}"


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_frozen_live_recompute_pins() -> None:
    """Frozen transcendental bits == live torch replay of the source
    expressions (T07 pattern: digest AND live recompute, no numpy trust).

    torch f64 sin != numpy f64 sin on this host (probe evidence), so the
    pin is a torch replay, not a numpy recomputation.
    """
    pytest.importorskip("torch")
    import torch

    for t in (8, 12, 16):
        z = np.load(CAP / f"t{t:02d}.npz")
        alt_f = float(z["arg_altitude__f64"])
        alt64 = torch.tensor(alt_f, dtype=torch.float64)
        got = torch.sin(alt64 * (np.pi / 180.)).item()
        assert np.float64(got).tobytes() == np.float64(
            z["frozen_sinalt_f64"]).tobytes(), f"t{t} sinalt"
        got = torch.cos(torch.tensor(alt_f)
                        * torch.tensor(np.pi / 180.)).item()
        assert np.float32(got).tobytes() == np.float32(
            z["frozen_cosalt"]).tobytes(), f"t{t} cosalt"
        Ta_f = float(z["arg_Ta__0d"])
        Ta64 = torch.tensor(Ta_f, dtype=torch.float64)
        got = ((Ta64 + 273.15) ** 4).item()
        assert np.float64(got).tobytes() == np.float64(
            z["frozen_Ta273_pow4_f64"]).tobytes(), f"t{t} ta273_pow4"
        # TsWaveDelay weight1: verbatim branch order on entry timeadd
        # (timeadd recorded as int at day steps, f64 at t0)
        ta = float(z["tsw0_in_timeadd__int"]) \
            if "tsw0_in_timeadd__int" in z \
            else float(z["tsw0_in_timeadd__f64"])
        tsd = float(z["tsw0_in_timestepdec__f64"])
        w = (torch.exp(-33.27 * torch.tensor(ta)).item()
             if ta >= (59 / 1440)
             else torch.exp(-33.27 * torch.tensor(ta + tsd)).item())
        assert np.float32(w).tobytes() == np.float32(
            z["tsw0_frozen_weight1"]).tobytes(), f"t{t} weight1"


# ---------------------------------------------------------------------------
# Capture parity (uint32 raw compare, signed zero + NaN payload distinct)
# ---------------------------------------------------------------------------

PLANE_RETS = ["Tmrt", "Kdown", "Kup", "Ldown", "Lup", "Tg", "shadow",
              "Keast", "Ksouth", "Kwest", "Knorth", "Least", "Lsouth",
              "Lwest", "Lnorth", "KsideI", "TgOut", "Lside",
              "KsideD", "dRad", "Kside"]


@pytest.fixture(scope="module")
def rad_static():
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    return radiation.rad_static_from_capture(CAP)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
@pytest.mark.parametrize("t", SPOT_TS)
def test_mirror_full_timestep_raw_parity(rad_static, t) -> None:
    """Hoisted mirror vs oracle capture: every return plane uint32-equal."""
    t_in = radiation.rad_bundle_from_capture(CAP, t)
    state = radiation.rad_state_from_capture(
        CAP, t, rows=rad_static.rows, cols=rad_static.cols)
    z = np.load(CAP / f"t{t:02d}.npz")
    out, _ = radiation.mirror_radiation_timestep(rad_static, t_in, state)
    for name in PLANE_RETS:
        assert_planes_equal(z[f"ret_{name}"], out[name])


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_fused_full_loop_raw_parity(rad_static) -> None:
    """FUSED kernel vs oracle capture: ALL 24 timesteps, state threaded
    t0->t23 (Tmrt + shadow + every return + next-state maps)."""
    manifest = json.loads((CAP / "manifest.json").read_text())
    n_t = manifest["n_timesteps"]
    state = radiation.rad_state_from_capture(
        CAP, 0, rows=rad_static.rows, cols=rad_static.cols)
    for t in range(n_t):
        t_in = radiation.rad_bundle_from_capture(CAP, t)
        z = np.load(CAP / f"t{t:02d}.npz")
        out, state = radiation.fused_radiation_timestep(
            rad_static, t_in, state)
        for name in PLANE_RETS:
            assert_planes_equal(z[f"ret_{name}"], out[name])
        for name in ("Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W",
                     "Tgmap1N", "TgOut1"):
            key = f"ret_{name}"
            if key in z:
                assert_planes_equal(z[key], getattr(state, name))


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_fused_equals_mirror_bitwise(rad_static) -> None:
    """Fusion invariant: fused kernel bits == hoisted mirror bits on the
    knife-edge timestep (t16, the Kside/dp sos divergence step)."""
    for t in (0, 8, 16):
        t_in = radiation.rad_bundle_from_capture(CAP, t)
        state = radiation.rad_state_from_capture(
            CAP, t, rows=rad_static.rows, cols=rad_static.cols)
        m_out, m_state = radiation.mirror_radiation_timestep(
            rad_static, t_in, state)
        f_out, f_state = radiation.fused_radiation_timestep(
            rad_static, t_in, state)
        for name in PLANE_RETS:
            assert_planes_equal(m_out[name], f_out[name])
        for name in ("Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W",
                     "Tgmap1N", "TgOut1"):
            assert_planes_equal(getattr(m_state, name),
                                getattr(f_state, name))


# ---------------------------------------------------------------------------
# Synthetic differential vs the live torch oracle (sunonsurface walk)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ORACLE_TREE.is_dir(), reason="oracle tree absent")
def test_synthetic_sunonsurface_walk_differential() -> None:
    """18-azimuth walk on a synthetic scene: mirror == live oracle torch,
    raw uint32 (stale-border walk semantics + first/second asymmetry).

    Call contract mirrors the site capture: first/second arrive as f32
    0-dim (1.0 / 22.0) with scale 0.5 -> effective round(1*0.5)=1 (floor
    to 1) and round(22*0.5)=11; Lup/Lwall/gvflup_extra come from the
    producer's torch replay of the source expressions.
    """
    torch = pytest.importorskip("torch")
    tree = str(ORACLE_TREE.resolve())
    if tree not in sys.path:
        sys.path.insert(0, tree)
    import solweig_gpu.solweig as sw

    from tests.ultrafast import radiation_oracle as ro

    rows, cols = 40, 80
    rng = np.random.default_rng(20090811)
    buildings = (rng.random((rows, cols)) < 0.3).astype(np.float32)
    walls = (rng.random((rows, cols)) < 0.2).astype(np.float32)
    dirwalls = (rng.integers(0, 360, (rows, cols))).astype(np.float32)
    shadow = rng.random((rows, cols)).astype(np.float32)
    Tg = (rng.random((rows, cols)) * 30 + 5).astype(np.float32)
    alb_grid = (rng.random((rows, cols)) * 0.25 + 0.05).astype(np.float32)
    emis_grid = np.full((rows, cols), np.float32(0.95))
    wallsun = (rng.random((rows, cols)) < 0.5).astype(np.float32)
    lc_grid = np.zeros((rows, cols), dtype=np.float32)
    Ta_f, Tgw_f, Twater_f = 20.0, 15.0, 10.0

    # torch views of the synthetic scene
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    buildings_t, walls_t, shadow_t = t(buildings), t(walls), t(shadow)
    Tg_t, alb_t, emis_t = t(Tg), t(alb_grid), t(emis_grid)
    dirwalls_t, lc_t = t(dirwalls), t(lc_grid)
    wallsun_t = t(wallsun)
    aspect_t = dirwalls_t * torch.pi / 180     # call-site expression
    Ta64 = torch.tensor(Ta_f, dtype=torch.float64)
    Tgw64 = torch.tensor(Tgw_f, dtype=torch.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sunwall_np = ((wallsun / walls * buildings) == 1).astype(np.float32)
    sunwall_t = t(sunwall_np)

    # frozen-family inputs via the producer's verbatim replays
    args0 = {
        "emis_grid": emis_grid, "Tg": Tg, "shadow": shadow,
        "buildings": buildings,
        "Ta": ("0d", Ta_f, "float64"), "Tgwall": ("0d", Tgw_f, "float64"),
        "ewall": ("py", 0.9),
    }
    Lup_pre, Lwall = ro.derive_lup_lwall(torch, args0)
    gvflup_extra = ro.derive_gvflup_extra(torch, args0, Tg)

    azimuthA = torch.arange(5, 359, 20, dtype=torch.float32)
    # walk offsets + azimuth branch statics (producer replay, SLEEF trig)
    dy_tab = np.zeros((18, 11), dtype=np.int64)
    dx_tab = np.zeros((18, 11), dtype=np.int64)
    az_list, lo_list, hi_list, br_list = [], [], [], []
    for j in range(18):
        azimuth = azimuthA[j] * (torch.pi / 180)
        pibyfour = torch.pi / 4
        t3, f5, s7 = 3 * pibyfour, 5 * pibyfour, 7 * pibyfour
        sa, ca, ta = torch.sin(azimuth), torch.cos(azimuth), torch.tan(azimuth)
        ssa, sca = torch.sign(sa), torch.sign(ca)
        index = 0
        for n in range(11):
            if (pibyfour <= azimuth and azimuth < t3) or \
                    (f5 <= azimuth and azimuth < s7):
                dy = ssa * index
                dx = -1 * sca * torch.abs(torch.round(index / ta))
            else:
                dy = ssa * torch.abs(torch.round(index * ta))
                dx = -1 * sca * index
            dy_tab[j, n] = int(dy.item())
            dx_tab[j, n] = int(dx.item())
            index += 1
        azilow = azimuth - torch.pi / 2
        azihigh = azimuth + torch.pi / 2
        c1 = bool(azilow >= 0 and azihigh < 2 * torch.pi)
        c2 = bool(azilow < 0 and azihigh <= 2 * torch.pi)
        c3 = bool(azilow > 0 and azihigh >= 2 * torch.pi)
        if c2:
            azilow = azilow + 2 * torch.pi
        if c3:
            azihigh = azihigh - 2 * torch.pi
        az_list.append(np.float32(azimuth))
        lo_list.append(np.float32(azilow))
        hi_list.append(np.float32(azihigh))
        br_list.append(1 if c1 else (2 if c2 else 3))

    st = radiation.RadiationStatic(
        rows=rows, cols=cols, buildings=buildings, walls=walls,
        aspect_rad=aspect_t.numpy(), wallbol=(walls > 0).astype(np.float32),
        alb_grid=alb_grid, emis_grid=emis_grid)
    t_in = radiation.RadTimestepInputs(is_day=True)
    t_in.az_statics = {
        "azimuth": np.array(az_list, np.float32),
        "azilow": np.array(lo_list, np.float32),
        "azihigh": np.array(hi_list, np.float32),
        "branch": np.array(br_list, np.int64),
    }
    t_in.walk_dy = dy_tab
    t_in.walk_dx = dx_tab
    t_in.shadow = shadow
    t_in.gvflup_extra = gvflup_extra

    for j in range(18):
        rets = sw.sunonsurface_2018a(
            azimuthA[j], 0.5, buildings_t, shadow_t, sunwall_t.clone(),
            torch.tensor(1.0), torch.tensor(22.0), aspect_t, walls_t,
            Tg_t.clone(), Tgw64, Ta64, emis_t, 0.9, alb_t, 5.67051e-8,
            0.2, Twater_f, lc_t, 0,
        )
        m = radiation.mirror_sunonsurface(
            st, t_in, j, sunwall_np, shadow, Lup_pre, Lwall)
        for i in range(5):
            assert_planes_equal(
                np.asarray(rets[i].detach().numpy(), dtype=np.float32),
                m[i])
