# SPDX-License-Identifier: GPL-3.0-only
"""T13 follow-up — cube-scope legacy digest gates (TASKS T12/T13 deferral).

The full 24-timestep solver cubes (shadow / tmrt / utci) are the last
digest tier the CUDA lanes had not pinned: they require the solver's
cross-timestep state carry (the six Tg planes + CI/firstdaytime/timeadd)
threaded through the radiation kernels in the ORACLE'S SOLVER ORDER
(march -> Solweig state in/out -> utci -> publish). The loop driver
under test is ``native/cuda/cubescope.py`` — ``full_solve_capture``'s
driver with the fused CPU kernel swapped for the CUDA radiation kernels
(T13 ResidentRadRunner) and the CUDA UTCI stage.

Gates, in evidence order:

* CONTROL (canonical profile, no cura needed) — the loop must reproduce
  the LOCAL t08 capture's published cube digests (manifest ``outputs``:
  the torch-2.14-macOS reference this repo's kernels are pinned to).
  This validates the loop driver + state threading on CUDA against a
  host-local absolute reference; it is the designated MUTATION KILLER
  for state-carry drops (a reset of the Tg planes between timesteps
  must break it).
* OBJECTIVE (legacy_cuda_v1 profile) — the shadow cube must match the
  UNIVERSAL pin 6c62bcea… (bit-identical across devices/hosts), and the
  tmrt/utci cubes are compared against the original cura GPU oracle
  pins (bf0cc3c7… / 32d4cbb0…, legacy_cuda_v1_index.json
  gpu_harness_PRIMARY). Cross-env bit equality for tmrt/utci is NOT
  expected (the lead's digest lattice: 4 distinct bit-patterns; our
  kernels consume the t08 capture's CPU-flavored frozen bits while the
  cura oracle computed per-t radiation internals on device) — the
  mismatch is CHARACTERIZED (per-t digest agreement table, lane counts,
  max |Δ| vs the gpu_outcheck spot planes), never relaxed away.
* PURITY — the driver is torch-free and never lets the legacy profile
  claim canonical certification.

Site/capture-dependent tests skip when the t08 capture bundle or the
met table is absent (SW_T12_T08_CAPTURE / SW_CUBESCOPE_MET); under
SW_REQUIRE_CUDA=1 unavailability is a hard failure (sw_cuda_harness
contract).

MUTATION KILLERS (designated; run via the external mutation cycle,
tests/ultrafast pattern — see REPORT):

* ``cubescope.state_carry.day_planes``   -> control gate
  (test_canonical_cubes_match_local_capture_manifest)
* ``cubescope.state_carry.scalars``      -> control gate
  (firstdaytime/timeadd reset between timesteps)
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from sw_cuda_harness import (  # noqa: E402
    compare_bits, load_host_module, require_canonical_runtime,
    require_runtime,
)

REPO_ROOT = ULTRA_DIR.parents[1]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"

T08_CAPTURE = os.environ.get(
    "SW_T12_T08_CAPTURE",
    str(Path.home() / "solweig_ultrafast" / "work" / "t12" / "fixtures" /
        "t08_capture"),
)
#: site met table (24 forcing rows; cols 9/10/11 = Ws/RH/Ta) — cura's
#: staged site data, or the local oracle tree mirror.
MET_DEFAULTS = (
    Path.home() / "solweig_ultrafast" / "data" / "Input_subset" /
    "processed_inputs" / "metfiles" / "metfile_0_0.txt",
    Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc") /
    "site-cache" / "site_500" / "metfiles" / "metfile_0_0.txt",
)
MET_PATH = os.environ.get("SW_CUBESCOPE_MET", None) or next(
    (str(p) for p in MET_DEFAULTS if p.is_file()), None)
ORACLE_BASELINE = os.environ.get(
    "SW_CUBESCOPE_ORACLE_BASELINE",
    str(Path.home() / "solweig_ultrafast" / "work" / "oracle_baseline"),
)
LATITUDE = 30.312645  # site_500 manifest solar_geometry.latitude

#: HARD pins (legacy_cuda_v1_index.json reference_labels, local mirror
#: ~/Workspace/solweig_ultrafast_artifacts/cura_oracle_baseline/).
PIN_SHADOW_UNIVERSAL = \
    "6c62bcea66d8c6256cf90647705dd4e713a7ca44dd6d76caadaf93f6de1c8c09"
PIN_CURA_TMRT = \
    "bf0cc3c7f3fa5ea697d01c01e10f801f606c6edd24fb9ca3a40a68072218a183"
PIN_CURA_UTCI = \
    "32d4cbb0412b991d1d72c29b0bb958f375bdb0df040a40a467ac071769385a6e"
#: local t08 capture manifest outputs (the CONTROL reference).
PIN_LOCAL_TMRT = \
    "65ef43ae86a0f7a5948d002a828e7dd005992c149f620d36432914ea677ad524"
PIN_LOCAL_UTCI = \
    "7d002c29f84e173c70de6960dc7cc8fae44e6cae4a76836a0b8fdda472d9f864"

#: the published utci scatter's valid-lane count (t09 gate; arg_buildings).
N_VALID_EXPECTED = 209422


# ---------------------------------------------------------------------------
# module loading (single host-module identity, residency contract)
# ---------------------------------------------------------------------------

_modules: dict[str, object] = {}


def _load(name: str):
    """Load a native/cuda module ONCE sharing the harness host module."""
    if name in _modules:
        return _modules[name]
    host = load_host_module()
    sys.modules.setdefault("sw_cuda_host", host)
    key = f"sw_cuda_{name}"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            key, NATIVE_CUDA / f"{name}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    _modules[name] = mod
    return mod


def capture_present() -> bool:
    return (Path(T08_CAPTURE) / "manifest.json").is_file()


def met_present() -> bool:
    return MET_PATH is not None and Path(MET_PATH).is_file()


def cube_digest(cube: np.ndarray) -> str:
    """The oracle's digest convention over a (n_t, rows, cols) f32 cube."""
    return hashlib.sha256(
        np.ascontiguousarray(cube, dtype=np.float32).tobytes()).hexdigest()


@functools.lru_cache(maxsize=4)
def _run_loop(profile_id: str):
    """Run the 24-t cube-scope loop ONCE per profile (module lifetime)."""
    host = load_host_module()
    rt = (require_canonical_runtime()
          if profile_id == host.CANONICAL_CUDA_V1
          else require_runtime(profile_id))
    if rt is None:
        pytest.skip(f"{profile_id} CUDA runtime unavailable")
    cubescope = _load("cubescope")
    return cubescope.run_cube_scope(
        rt, Path(T08_CAPTURE), MET_PATH, latitude=LATITUDE)


@pytest.fixture(scope="module")
def host():
    return load_host_module()


@pytest.fixture(scope="module")
def rt_legacy(host):
    rt = require_runtime(host.LEGACY_CUDA_V1)
    if rt is None:
        pytest.skip("legacy CUDA runtime unavailable")
    return rt


@pytest.fixture(scope="module")
def legacy_result(host, rt_legacy):
    return _run_loop(host.LEGACY_CUDA_V1)


@pytest.fixture(scope="module")
def canonical_result(host):
    rt = require_canonical_runtime()
    if rt is None:
        pytest.skip("canonical CUDA runtime unavailable")
    return _run_loop(host.CANONICAL_CUDA_V1)


@pytest.fixture(scope="module")
def capture_manifest():
    return json.loads((Path(T08_CAPTURE) / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# purity + input coherence
# ---------------------------------------------------------------------------


class TestPurityAndInputs:
    def test_driver_module_torch_free(self):
        p = NATIVE_CUDA / "cubescope.py"
        if not p.is_file():
            pytest.fail(f"missing loop driver {p}")
        src = p.read_text()
        assert "import torch" not in src, "cube-scope driver must stay torch-free"

    @pytest.mark.skipif(not capture_present() or not met_present(),
                        reason="t08 capture / met table absent")
    def test_capture_met_coherence(self, capture_manifest):
        """The met table IS the capture's forcing: Ta/RH (f64) agree with
        the recorded args at every timestep, and the buildings scatter
        mask carries the t09 valid-lane count."""
        sys.path.insert(0, str(REPO_ROOT))
        from solweig_core.numba_cpu.radiation import rad_static_from_capture
        met = np.loadtxt(MET_PATH, skiprows=1, dtype=np.float64)
        n = int(capture_manifest["n_timesteps"])
        assert met.shape[0] == n, f"met rows {met.shape[0]} != {n}"
        cap = Path(T08_CAPTURE)
        for t in range(n):
            z = np.load(cap / f"t{t:02d}.npz")
            assert float(z["arg_Ta__0d"]) == float(met[t, 11]), f"t{t} Ta"
            assert float(z["arg_RH__0d"]) == float(met[t, 10]), f"t{t} RH"
        st = rad_static_from_capture(cap)
        assert int(np.count_nonzero(st.buildings == 1)) == N_VALID_EXPECTED


# ---------------------------------------------------------------------------
# CONTROL gate — canonical loop vs the LOCAL capture cube digests
# ---------------------------------------------------------------------------


class TestCanonicalControlGate:
    @pytest.mark.skipif(not capture_present() or not met_present(),
                        reason="t08 capture / met table absent")
    def test_canonical_cubes_match_local_capture_manifest(self,
                                                          canonical_result,
                                                          capture_manifest):
        """CONTROL (designated state-carry mutation killer): the canonical
        CUDA loop reproduces the local t08 capture's published cube
        digests — shadow/tmrt/utci — proving the solver-order state
        threading + CUDA stages equal the reference solve bit-for-bit."""
        want = capture_manifest["outputs"]
        # fence fixture drift: the manifest must carry the pinned digests
        assert want["shadow"] == PIN_SHADOW_UNIVERSAL
        assert want["tmrt"] == PIN_LOCAL_TMRT
        assert want["utci"] == PIN_LOCAL_UTCI
        cubes = canonical_result.cubes
        for name, pin in (("shadow", PIN_SHADOW_UNIVERSAL),
                          ("tmrt", PIN_LOCAL_TMRT),
                          ("utci", PIN_LOCAL_UTCI)):
            d = cube_digest(cubes[name])
            assert d == pin, (
                f"canonical {name} cube {d} != local capture pin {pin}"
            )


# ---------------------------------------------------------------------------
# OBJECTIVE gates — legacy profile vs the cura GPU oracle pins
# ---------------------------------------------------------------------------


class TestLegacyObjectiveGates:
    @pytest.mark.skipif(not capture_present() or not met_present(),
                        reason="t08 capture / met table absent")
    def test_legacy_shadow_cube_matches_universal_pin(self, legacy_result):
        """The shadow lineage stays universal under the legacy profile:
        frozen march bits + the clean shadow composition."""
        d = cube_digest(legacy_result.cubes["shadow"])
        assert d == PIN_SHADOW_UNIVERSAL, (
            f"legacy shadow cube {d} != universal pin {PIN_SHADOW_UNIVERSAL}"
        )

    @pytest.mark.skipif(not capture_present() or not met_present(),
                        reason="t08 capture / met table absent")
    @pytest.mark.xfail(
        strict=True,
        reason="cube-scope bit identity vs the cura oracle is NOT "
               "achievable under the frozen-capture-inputs constraint: "
               "our lanes consume the t08 capture's CPU-flavored frozen "
               "bits (macOS torch 2.14 lineage) while the oracle computed "
               "per-t radiation internals on device (torch 2.5.1+cu121) — "
               "measured 0/24 per-t digest agreement for tmrt AND utci, "
               "spot-plane magnitudes exactly the documented cross-device "
               "lattice (tmrt day max 0.345 C, utci day max 0.0852 C, "
               "night max ~6e-05 C). Pins stay hardcoded as the target; "
               "if this ever XPASSes the input regime changed — "
               "re-examine, do not relax.",
    )
    def test_legacy_tmrt_utci_cubes_vs_cura_pins(self, legacy_result):
        """OBJECTIVE (fenced): legacy-profile tmrt/utci cube bit identity
        vs the original cura GPU oracle pins."""
        d_t = cube_digest(legacy_result.cubes["tmrt"])
        d_u = cube_digest(legacy_result.cubes["utci"])
        assert d_t == PIN_CURA_TMRT, (
            f"legacy tmrt cube {d_t} != cura pin {PIN_CURA_TMRT}"
        )
        assert d_u == PIN_CURA_UTCI, (
            f"legacy utci cube {d_u} != cura pin {PIN_CURA_UTCI}"
        )

    @pytest.mark.skipif(not capture_present() or not met_present(),
                        reason="t08 capture / met table absent")
    def test_legacy_loop_deterministic(self, host, rt_legacy,
                                       legacy_result):
        """A second fresh legacy run reproduces every cube digest
        bit-for-bit (the measured divergence vs cura is a stable fact,
        not run-to-run noise)."""
        cubescope = _load("cubescope")
        again = cubescope.run_cube_scope(
            rt_legacy, Path(T08_CAPTURE), MET_PATH, latitude=LATITUDE)
        for name in ("shadow", "tmrt", "utci"):
            assert cube_digest(again.cubes[name]) == \
                cube_digest(legacy_result.cubes[name]), name


# ---------------------------------------------------------------------------
# CHARACTERIZATION — deviation magnitudes + per-t digest agreement
# ---------------------------------------------------------------------------


def _uint32(plane: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(plane, dtype=np.float32).view(np.uint32)


def deviation_stats(want: np.ndarray, got: np.ndarray) -> dict:
    w, g = _uint32(want).reshape(-1), _uint32(got).reshape(-1)
    bad = w != g
    n = int(bad.sum())
    stats: dict = {"n_lanes": int(w.size), "n_differ": n,
                   "frac_differ": round(n / max(w.size, 1), 8)}
    if n:
        fw = np.asarray(want, np.float32).reshape(-1)[bad].astype(np.float64)
        fg = np.asarray(got, np.float32).reshape(-1)[bad].astype(np.float64)
        both = np.isfinite(fw) & np.isfinite(fg)
        if both.any():
            d = np.abs(fw[both] - fg[both])
            stats["max_abs"] = float(d.max())
            stats["mean_abs"] = float(d.mean())
        stats["n_nan_mismatch"] = int((np.isnan(fw) != np.isnan(fg)).sum())
    return stats


def _record(name: str, payload: dict) -> None:
    root = os.environ.get("SW_CUBESCOPE_ARTIFACTS")
    if not root:
        return
    out = Path(root) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


class TestLegacyCharacterization:
    @pytest.mark.skipif(not capture_present() or not met_present(),
                        reason="t08 capture / met table absent")
    def test_deviations_recorded_and_fenced(self, legacy_result,
                                            canonical_result):
        """Legacy vs canonical (full cubes, both devices-identical inputs)
        and legacy vs the cura oracle's published spot planes
        (gpu_outcheck t00/t05/t12/t23): differing-lane counts + max |Δ|
        recorded to SW_CUBESCOPE_ARTIFACTS, and FENCED — the divergence
        must stay inside the characterized envelope (it moves if the
        profile or constants drift, which is exactly what this gate is
        for; nothing here may be relaxed to chase a green)."""
        report: dict = {
            "legacy_vs_canonical": {}, "legacy_vs_cura_spot": {},
            "per_t_digest_agreement": {}, "cube_digests": {},
        }

        report["cube_digests"]["canonical"] = {
            name: cube_digest(canonical_result.cubes[name])
            for name in ("shadow", "tmrt", "utci")}
        report["cube_digests"]["legacy"] = {
            name: cube_digest(legacy_result.cubes[name])
            for name in ("shadow", "tmrt", "utci")}
        report["cube_digests"]["pins"] = {
            "shadow_universal": PIN_SHADOW_UNIVERSAL,
            "cura_tmrt": PIN_CURA_TMRT, "cura_utci": PIN_CURA_UTCI,
            "local_tmrt": PIN_LOCAL_TMRT, "local_utci": PIN_LOCAL_UTCI,
        }

        for name in ("shadow", "tmrt", "utci"):
            report["legacy_vs_canonical"][name] = deviation_stats(
                canonical_result.cubes[name], legacy_result.cubes[name])
        # shadow must be bit-identical legacy vs canonical (frozen bits +
        # clean composition; FMA never touches it)
        assert report["legacy_vs_canonical"]["shadow"]["n_differ"] == 0

        spot_root = (Path(ORACLE_BASELINE) / "gpu_outcheck")
        checked_spots = 0
        if spot_root.is_dir():
            for t in (0, 5, 12, 23):
                p = spot_root / f"spot_out_t{t:02d}.npz"
                if not p.is_file():
                    continue
                z = np.load(p)
                for name in ("shadow", "tmrt", "utci"):
                    want = np.ascontiguousarray(z[f"out_{name}"],
                                                dtype=np.float32)
                    got = legacy_result.cubes[name][t]
                    report["legacy_vs_cura_spot"][f"t{t:02d}:{name}"] = \
                        deviation_stats(want, got)
                checked_spots += 1
        report["n_spot_timesteps"] = checked_spots

        # per-t digest agreement vs the oracle's intermediates (kernel
        # Tmrt return + the PRE-scatter utci_calculator plane) — counted
        # for BOTH profiles: canonical agreeing nowhere either is the
        # evidence that the divergence is the input-flavor regime (the
        # capture's CPU-lineage frozen bits vs the oracle's on-device
        # internals), not the legacy FMA contraction
        inter_p = Path(ORACLE_BASELINE) / "gpu_harness" / "intermediates.json"
        agree: dict = {}
        if inter_p.is_file():
            inter = json.loads(inter_p.read_text())
            oracle_tmrt = {int(r["t"]): r["returns"]["Tmrt"]["sha256"]
                           for r in inter["solweig"]}
            oracle_utci = {int(r["t"]): r["return"]["sha256"]
                           for r in inter["utci"]}
            for label, result in (("legacy", legacy_result),
                                  ("canonical", canonical_result)):
                counts = {"tmrt": [0, 0], "utci": [0, 0]}
                for t in range(result.n_timesteps):
                    hit_t = (result.per_t_digests["tmrt"][t]
                             == oracle_tmrt.get(t))
                    counts["tmrt"][0 if hit_t else 1] += 1
                    hit_u = (result.per_t_digests["utci_mat"][t]
                             == oracle_utci.get(t))
                    counts["utci"][0 if hit_u else 1] += 1
                agree[label] = counts
        report["per_t_digest_agreement"] = agree
        _record("cubescope_legacy_characterization.json", report)

        # FENCE (characterized envelope; evidence-backed, not aspirational):
        # (a) legacy vs canonical (identical inputs, only --fmad differs):
        #     shadow 0 lanes; tmrt/utci diverge on a single-digit-ulp scale
        #     (measured: tmrt 6.8% lanes max 6.1e-05, utci 5.0% lanes max
        #     6.9e-05 — FMA contraction, never a physics-scale change);
        # (b) legacy vs the cura oracle spot planes: magnitudes within the
        #     documented cross-device lattice (tmrt day max 0.345 C, utci
        #     day max 0.0852 C, night ~6e-05 C) — the input-flavor regime;
        # (c) per-t agreement vs the oracle stays below full coverage for
        #     BOTH profiles (bit identity is not achievable in this regime
        #     — the xfail fence above records the target).
        lv = report["legacy_vs_canonical"]
        assert lv["shadow"]["n_differ"] == 0
        for name in ("tmrt", "utci"):
            assert 0 < lv[name]["n_differ"] <= int(0.20 * lv[name]["n_lanes"]), \
                f"{name}: {lv[name]}"
            assert lv[name].get("max_abs", 0.0) <= 5e-4, f"{name}: {lv[name]}"
        for key, st in report["legacy_vs_cura_spot"].items():
            if ":tmrt" in key:
                assert st.get("max_abs", 0.0) <= 0.5, f"{key}: {st}"
            if ":utci" in key:
                assert st.get("max_abs", 0.0) <= 0.2, f"{key}: {st}"
        for label in ("legacy", "canonical"):
            counts = agree.get(label)
            if counts is None:
                continue
            for name in ("tmrt", "utci"):
                assert counts[name][1] > 0, (
                    f"{label}/{name}: full per-t agreement with the cura "
                    "oracle contradicts the xfail fence — re-examine the "
                    "input regime")
