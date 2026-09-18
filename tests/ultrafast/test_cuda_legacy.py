# SPDX-License-Identifier: GPL-3.0-only
"""T12 — legacy_cuda_v1 profile vs the ORIGINAL torch-CUDA oracle
(TASKS T12; DESIGN 3.1 profile separation; lead relay contract).

The legacy profile (aten-default-like ``--fmad=true`` characterization
build, UNCERTIFIED) is tied to the original oracle run recorded under
``~/solweig_ultrafast/work/oracle_baseline/`` (read-only; torch
2.5.1+cu121, repo dc61d85, RTX A6000):

HARD BIT GATES (env-independent — the shadow lineage):

* per-t march: the legacy build's wallheight23 sh/vegsh planes over the
  site_500 march pins digest-match the oracle gpu_harness
  ``intermediates.json`` per-t march digests for every march timestep
  (the same inputs whose canonical digests the site pins carry);
* legacy == canonical on the same march inputs (``--fmad=true`` moves
  no march bit — the march recurrence is compare/select arithmetic).

DIAGNOSTICS ONLY (never PASS gates — cross-env bit equality is
unachievable by the lead's digest lattice: 4 distinct tmrt/utci
bit-patterns across devices/hosts):

* legacy vs canonical deviation magnitudes on the radiation day bundle
  (capture t12), night bundle (capture t05) and a dense UTCI plane:
  differing-uint32 counts, max |Δ| and max ulp per output, recorded to
  the artifacts root when ``SW_T12_ARTIFACTS`` is set.

The full 24-t published-cube digests (shadow 6c62bcea…, tmrt
bf0cc3c7…, utci 32d4cbb0…) require the oracle solver's cross-timestep
state carry and live in T13 integration; the per-t march anchors here
pin the shadow lineage at kernel level.
"""
from __future__ import annotations

import hashlib
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
    compare_bits, f32_from_hex, load_host_module,
    require_canonical_runtime, require_runtime,
)

REPO_ROOT = ULTRA_DIR.parents[1]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"

ORACLE_BASELINE = os.environ.get(
    "SW_T12_ORACLE_BASELINE",
    str(Path.home() / "solweig_ultrafast" / "work" / "oracle_baseline"),
)
T08_CAPTURE = os.environ.get(
    "SW_T12_T08_CAPTURE",
    str(Path.home() / "solweig_ultrafast" / "work" / "t12" / "fixtures" /
        "t08_capture"),
)


def plane_digest(arr_f32: np.ndarray) -> str:
    """The oracle's digest_value convention for a float32 plane."""
    return hashlib.sha256(
        np.ascontiguousarray(arr_f32, dtype=np.float32).tobytes()
    ).hexdigest()


def oracle_march_digests() -> dict[int, dict[str, str]] | None:
    p = Path(ORACLE_BASELINE) / "gpu_harness" / "intermediates.json"
    if not p.is_file():
        return None
    inter = json.loads(p.read_text())
    out: dict[int, dict[str, str]] = {}
    for rec in inter["march"]:
        out[int(rec["t"])] = {
            name: r["sha256"] for name, r in rec["returns"].items()
            if isinstance(r, dict) and "sha256" in r
        }
    return out


def site_march_inputs():
    path = NATIVE_CUDA / "data" / "march_site_pins.json"
    if not path.is_file():
        pytest.skip(f"site march pins not generated ({path})")
    site = json.loads(path.read_text())["site"]
    shape = (site["rows"], site["cols"])
    return site, shape, path


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
def rt_canonical():
    return require_canonical_runtime()


# ---------------------------------------------------------------------------
# HARD gates: the shadow lineage, legacy build vs the original oracle
# ---------------------------------------------------------------------------


class TestLegacyMarchOracleAnchor:
    def test_legacy_march_matches_oracle_per_t_digests(self, rt_legacy,
                                                       host):
        """The legacy build reproduces the ORIGINAL torch-CUDA oracle's
        per-t march sh/vegsh bits for every march timestep (direct
        kernel-level tie to gpu_harness intermediates.json)."""
        oracle = oracle_march_digests()
        if oracle is None:
            pytest.skip(
                f"oracle baseline not available ({ORACLE_BASELINE})")
        site, shape, _ = site_march_inputs()
        a = f32_from_hex(site["a"]).reshape(shape)
        vegdsm = f32_from_hex(site["vegdsm"]).reshape(shape)
        vegdsm2 = f32_from_hex(site["vegdsm2"]).reshape(shape)
        failures = []
        checked = 0
        # per_t keys are FORCING indices; the oracle march records the
        # solver timestep t = forcing_index - 1 (site pins' own
        # oracle_cross_check carries the same mapping: t=7 ↔ forcing 8).
        for t, rec in sorted(site["per_t"].items(), key=lambda kv: int(kv[0])):
            want = oracle.get(int(t) - 1)
            if want is None or "sh" not in want or "vegsh" not in want:
                continue
            sh, vegsh, _ = host.run_march_wallheight23(
                rt_legacy, a, vegdsm, vegdsm2,
                np.array(rec["dx"], dtype=np.int32),
                np.array(rec["dy"], dtype=np.int32),
                f32_from_hex(rec["dz"]), f32_from_hex(rec["dzprev"]))
            checked += 1
            for name, got, w in (("sh", sh, want["sh"]),
                                 ("vegsh", vegsh, want["vegsh"])):
                d = plane_digest(got)
                if d != w:
                    failures.append(f"t={t} {name}: {d} != oracle {w}")
        assert checked >= 10, (
            f"only {checked} march timesteps cross-checked — oracle "
            "intermediates and site pins disagree on coverage"
        )
        assert not failures, "\n".join(failures[:6])

    def test_legacy_march_bits_equal_canonical(self, rt_legacy, rt_canonical,
                                               host):
        """--fmad=true moves no march bit: legacy and canonical builds are
        bit-identical on the site march (compare/select recurrence)."""
        if rt_canonical is None:
            pytest.skip("canonical CUDA runtime unavailable")
        site, shape, _ = site_march_inputs()
        a = f32_from_hex(site["a"]).reshape(shape)
        vegdsm = f32_from_hex(site["vegdsm"]).reshape(shape)
        vegdsm2 = f32_from_hex(site["vegdsm2"]).reshape(shape)
        for t, rec in sorted(site["per_t"].items(), key=lambda kv: int(kv[0]))[:4]:
            args = (a, vegdsm, vegdsm2, np.array(rec["dx"], dtype=np.int32),
                    np.array(rec["dy"], dtype=np.int32),
                    f32_from_hex(rec["dz"]), f32_from_hex(rec["dzprev"]))
            leg = host.run_march_wallheight23(rt_legacy, *args)
            can = host.run_march_wallheight23(rt_canonical, *args)
            for i, name in enumerate(("sh", "vegsh", "vbshvegsh")):
                msg, _ = compare_bits(can[i], leg[i], f"t={t} {name}")
                assert msg == "PASS", msg

    def test_legacy_refuses_canonical_certification(self, rt_legacy, host):
        """Reaffirmed on this profile's gates: the legacy build can never
        claim canonical parity (it must raise, not return)."""
        with pytest.raises(host.RuntimeMismatch, match="not the canonical"):
            rt_legacy.assert_canonical_strict()


# ---------------------------------------------------------------------------
# DIAGNOSTICS: legacy vs canonical deviation magnitudes (characterization
# only — differences are EXPECTED; nothing here can fail on inequality)
# ---------------------------------------------------------------------------


def _uint32(plane: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(plane, dtype=np.float32).view(np.uint32)


def deviation_stats(want: np.ndarray, got: np.ndarray) -> dict:
    w, g = _uint32(want).reshape(-1), _uint32(got).reshape(-1)
    diff_mask = w != g
    n = int(diff_mask.sum())
    stats: dict = {"n_lanes": int(w.size), "n_differ": n,
                   "frac_differ": round(n / max(w.size, 1), 8)}
    if n:
        fw = want.reshape(-1)[diff_mask].astype(np.float64)
        fg = got.reshape(-1)[diff_mask].astype(np.float64)
        both_finite = np.isfinite(fw) & np.isfinite(fg)
        if both_finite.any():
            absd = np.abs(fw[both_finite] - fg[both_finite])
            stats["max_abs"] = float(absd.max())
            stats["mean_abs"] = float(absd.mean())
        stats["n_nan_mismatch"] = int(
            (np.isnan(fw) != np.isnan(fg)).sum())
    return stats


class TestLegacyDeviationDiagnostics:
    def test_rad_day_t12_deviations_recorded(self, rt_legacy, rt_canonical,
                                             host):
        if rt_canonical is None or not Path(T08_CAPTURE).is_dir():
            pytest.skip("canonical runtime or t08 capture unavailable")
        from solweig_core.numba_cpu.radiation import (
            rad_bundle_from_capture, rad_state_from_capture,
            rad_static_from_capture,
        )
        from sw_rad_bundle import build_rad_day_bundle

        cap = Path(T08_CAPTURE)
        st = rad_static_from_capture(cap)
        t_in = rad_bundle_from_capture(cap, 12)
        state = rad_state_from_capture(cap, 12, rows=st.rows, cols=st.cols)
        b = build_rad_day_bundle(st, t_in, state)
        can = host.run_rad_day(rt_canonical, b)
        leg = host.run_rad_day(rt_legacy, b)
        report = {name: deviation_stats(can[name], leg[name])
                  for name in sorted(can)}
        # control: canonical rerun must stay bit-identical
        can2 = host.run_rad_day(rt_canonical, b)
        for name in sorted(can):
            msg, _ = compare_bits(can[name], can2[name],
                                  f"canonical-rerun {name}")
            assert msg == "PASS", msg
        _record("legacy_vs_canonical_rad_day_t12.json", report)
        assert all("n_differ" in v for v in report.values())

    def test_rad_night_t05_deviations_recorded(self, rt_legacy, rt_canonical,
                                               host):
        if rt_canonical is None or not Path(T08_CAPTURE).is_dir():
            pytest.skip("canonical runtime or t08 capture unavailable")
        from solweig_core.numba_cpu.radiation import (
            rad_bundle_from_capture, rad_static_from_capture,
        )
        from sw_rad_bundle import build_rad_night_bundle

        cap = Path(T08_CAPTURE)
        st = rad_static_from_capture(cap)
        t_in = rad_bundle_from_capture(cap, 5)
        b = build_rad_night_bundle(st, t_in)
        can = host.run_rad_night(rt_canonical, b)
        leg = host.run_rad_night(rt_legacy, b)
        report = {name: deviation_stats(can[name], leg[name])
                  for name in sorted(can)}
        _record("legacy_vs_canonical_rad_night_t05.json", report)
        assert all("n_differ" in v for v in report.values())

    def test_utci_dense_deviations_recorded(self, rt_legacy, rt_canonical,
                                            host):
        if rt_canonical is None:
            pytest.skip("canonical CUDA runtime unavailable")
        rng = np.random.default_rng(0x1E6A)
        planes = tuple(
            np.ascontiguousarray(a, dtype=np.float32) for a in (
                rng.uniform(-5.0, 38.0, (200, 210)),
                rng.uniform(20.0, 95.0, (200, 210)),
                rng.uniform(10.0, 75.0, (200, 210)),
                rng.uniform(0.3, 10.0, (200, 210))))
        can = host.run_utci_dense(rt_canonical, *planes)
        leg = host.run_utci_dense(rt_legacy, *planes)
        report = {"utci": deviation_stats(can, leg)}
        _record("legacy_vs_canonical_utci_dense.json", report)
        assert "n_differ" in report["utci"]


def _record(name: str, payload: dict) -> None:
    root = os.environ.get("SW_T12_ARTIFACTS")
    if not root:
        return
    out = Path(root) / "legacy_diagnostics" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
