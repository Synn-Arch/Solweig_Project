# SPDX-License-Identifier: GPL-3.0-only
"""T12 GATE 2 — CUDA march kernels bit-equality (TASKS T12, DESIGN 7.1).

The CUDA march (native/cuda/src/sw_march.cu, one thread per target cell
replaying the frozen T03 step tables as DATA) must reproduce the CANONICAL
CPU march BITS (solweig_core/numba_cpu/march.py, the frozen T04 reference)
on:

* every frozen T03 trace scene (74) plus the variant grid — full-plane
  raw uint32 equality (signed zero + NaN payload, no tolerance);
* the site_500 anchor: the exact planes/amax/solar bits the oracle GPU
  harness fed the original kernel — per-timestep sha256 digests over the
  exact float32 bytes (raw-bit equality by digest), cross-anchored to the
  oracle baseline intermediates (march sh/vegsh are the UNIVERSAL anchor:
  bit-identical across GPU, CPU and every composition).

RED witnesses: schedule independence (#7 — outputs depend only on target
cell + table + planes; no cross-thread state exists at this stage) and the
prerequisite primitive discipline (#1-#4) carried over from GATE 1.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from sw_cuda_harness import (  # noqa: E402
    compare_bits,
    f32_from_hex,
    load_host_module,
    load_pins,
    require_canonical_runtime,
)

NATIVE_CUDA = ULTRA_DIR.parents[1] / "native" / "cuda"


@pytest.fixture(scope="module")
def rt():
    return require_canonical_runtime()


@pytest.fixture(scope="module")
def host():
    return load_host_module()


@pytest.fixture(scope="module")
def march_pins():
    path = NATIVE_CUDA / "data" / "march_pins.json"
    if not path.is_file():
        pytest.skip(f"march pins not generated ({path})")
    return json.loads(path.read_text())["sets"]


def plane_digest(arr_f32: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(arr_f32, dtype=np.float32).tobytes()
    ).hexdigest()


def _run_entry(host, rt, entry):
    shape = (entry["rows"], entry["cols"])
    a = f32_from_hex(entry["a"]).reshape(shape)
    vegdem = f32_from_hex(entry["vegdem"]).reshape(shape)
    vegdem2 = f32_from_hex(entry["vegdem2"]).reshape(shape)
    dx = np.array(entry["dx"], dtype=np.int32)
    dy = np.array(entry["dy"], dtype=np.int32)
    dz = f32_from_hex(entry["dz"])
    if entry["variant"] == "svf_shadow":
        return host.run_march_svf_shadow(rt, a, vegdem, vegdem2, dx, dy, dz)
    dzprev = f32_from_hex(entry["dzprev"])
    return host.run_march_wallheight23(rt, a, vegdem, vegdem2, dx, dy, dz,
                                       dzprev)


# ---------------------------------------------------------------------------
# frozen T03 traces + variant grid (full-plane raw-bit equality)
# ---------------------------------------------------------------------------


class TestMarchFrozenTraces:
    def test_frozen74_raw_bit_equal(self, rt, host, march_pins):
        entries = march_pins.get("frozen74")
        if not entries:
            pytest.skip("frozen74 set not generated")
        failures = []
        for entry in entries:
            sh, vegsh, vbsh = _run_entry(host, rt, entry)
            for name, got, want in (("sh", sh, entry["sh"]),
                                    ("vegsh", vegsh, entry["vegsh"]),
                                    ("vbsh", vbsh, entry["vbsh"])):
                msg, _ = compare_bits(f32_from_hex(want).reshape(got.shape),
                                      got, f"{entry['trace_id']}:{name}")
                if msg != "PASS":
                    failures.append(msg)
        assert not failures, "\n".join(failures[:6])
        n_svf = sum(1 for e in entries if e["variant"] == "svf_shadow")
        assert (n_svf, len(entries) - n_svf) == (37, 37)

    def test_variants_raw_bit_equal(self, rt, host, march_pins):
        entries = march_pins.get("variants")
        if not entries:
            pytest.skip("variant set not generated (needs torch)")
        failures = []
        for entry in entries:
            sh, vegsh, vbsh = _run_entry(host, rt, entry)
            for name, got, want in (("sh", sh, entry["sh"]),
                                    ("vegsh", vegsh, entry["vegsh"]),
                                    ("vbsh", vbsh, entry["vbsh"])):
                msg, _ = compare_bits(f32_from_hex(want).reshape(got.shape),
                                      got, f"{entry['trace_id']}:{name}")
                if msg != "PASS":
                    failures.append(msg)
        assert not failures, "\n".join(failures[:6])

    def test_one_step_vbsh_witness(self, rt, host, march_pins):
        """The one-step regime produces vbsh == 2.0 (1 - (0 - 1)) that no
        bool packing can represent — the f32 accumulator must survive."""
        entries = (march_pins.get("frozen74") or []) + \
            (march_pins.get("variants") or [])
        saw_two = False
        for entry in entries:
            _sh, _vegsh, vbsh = _run_entry(host, rt, entry)
            want = f32_from_hex(entry["vbsh"])
            saw_two = saw_two or bool(np.any(want == np.float32(2.0)))
        assert saw_two, "no pin exercises the vbsh == 2.0 witness"


# ---------------------------------------------------------------------------
# site_500 anchor (digest equality + oracle cross-anchor)
# ---------------------------------------------------------------------------


class TestMarchSiteAnchor:
    def test_site_500_per_t_digests(self, rt, host):
        path = NATIVE_CUDA / "data" / "march_site_pins.json"
        if not path.is_file():
            pytest.skip(f"site march pins not generated ({path})")
        site = json.loads(path.read_text())["site"]
        shape = (site["rows"], site["cols"])
        a = f32_from_hex(site["a"]).reshape(shape)
        vegdsm = f32_from_hex(site["vegdsm"]).reshape(shape)
        vegdsm2 = f32_from_hex(site["vegdsm2"]).reshape(shape)
        failures = []
        for t, rec in sorted(site["per_t"].items(), key=lambda kv: int(kv[0])):
            dx = np.array(rec["dx"], dtype=np.int32)
            dy = np.array(rec["dy"], dtype=np.int32)
            dz = f32_from_hex(rec["dz"])
            dzprev = f32_from_hex(rec["dzprev"])
            sh, vegsh, vbsh = host.run_march_wallheight23(
                rt, a, vegdsm, vegdsm2, dx, dy, dz, dzprev)
            for name, got, want in (
                    ("sh", sh, rec["sh_sha256"]),
                    ("vegsh", vegsh, rec["vegsh_sha256"]),
                    ("vbsh", vbsh, rec["vbsh_sha256"])):
                d = plane_digest(got)
                if d != want:
                    failures.append(f"t={t} {name}: digest {d} != {want}")
        assert not failures, "\n".join(failures[:6])

    def test_site_pins_cross_anchored_to_oracle(self):
        """The pin digests themselves equal the oracle GPU baseline per-t
        march digests (universal anchor: sh/vegsh bit-identical across
        GPU/CPU/compositions) — proves the CPU reference generation fed
        the same planes/tables/amax the original consumed."""
        path = NATIVE_CUDA / "data" / "march_site_pins.json"
        if not path.is_file():
            pytest.skip(f"site march pins not generated ({path})")
        site = json.loads(path.read_text())["site"]
        checks = site.get("oracle_cross_check")
        if not checks:
            pytest.skip("site pins generated without oracle cross-check")
        bad = [c for c in checks
               if not (c.get("sh") and c.get("vegsh"))]
        assert not bad, f"CPU-reference vs oracle mismatch at t={[c['t'] for c in bad]}"
