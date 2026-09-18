# SPDX-License-Identifier: GPL-3.0-only
"""T12 GATE 4 — CUDA radiation stage bit-equality (TASKS T12, DESIGN 9.3).

The CUDA radiation kernels (native/cuda/src/sw_radiation.cu, one thread
per target cell replaying the fused T08 reference's per-cell graph —
18-azimuth walk, TsWaveDelay x6, Kup/Kside/Kdown, define_patch two-pass
with the cell-local reflection barrier, Sstr/Tmrt) must reproduce the
CANONICAL CPU fused kernel BITS (solweig_core/numba_cpu/radiation.py)
on:

* the crop pin set — every kernel input AND output of a fixed 48x48
  window of the t08 captures, raw uint32 equality on all outputs (day:
  19 planes + 6 next-state planes; night: 7 planes), signed zero + NaN
  payload, no tolerance;
* the site_500 anchor — full-plane sha256 digests over the exact float32
  bytes (raw-bit equality by digest) for every captured timestep, when
  the capture fixtures are available (SW_T12_T08_CAPTURE) and their
  sha256s match the pins.

RED witnesses at this stage: every float32 plane chain keeps the source
association (reordered accumulation / FMA contraction / f64-promotion
loss are killed by the raw pins — FP reduction reordering, witness 5);
the packed cubes are read-only per lane and outputs are disjoint by
cell, so no packed-byte write exists here (witness 6 belongs to the
bitplane packer); lane outputs depend only on the cell's read extent
(schedule independence, witness 7).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
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
def pins():
    path = NATIVE_CUDA / "data" / "rad_pins.json"
    if not path.is_file():
        pytest.skip(f"radiation pins not generated ({path})")
    return json.loads(path.read_text())


def u8_from_hex(hex_list) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_list], dtype=np.uint8)


def plane_digest(arr_f32: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(arr_f32, dtype=np.float32).tobytes()).hexdigest()


def deserialize(rec: dict, host) -> dict:
    """Pinned inputs -> flat bundle dict (kernel indexing is linear; the
    dtype dispatch mirrors the generator's serialization exactly)."""
    dtmap = dict(host._RAD_DAY_INPUTS) if rec["is_day"] \
        else dict(host._RAD_NIGHT_INPUTS)
    f32s = host._RAD_DAY_F32_SCALARS if rec["is_day"] else ()
    f64s = host._RAD_DAY_F64_SCALARS if rec["is_day"] \
        else host._RAD_NIGHT_F64_SCALARS
    ints = host._RAD_DAY_INT_SCALARS if rec["is_day"] \
        else host._RAD_NIGHT_INT_SCALARS
    b = {}
    for k, v in rec["inputs"].items():
        if k in dtmap:
            dt = dtmap[k]
            if dt == np.float32:
                b[k] = f32_from_hex(v)
            elif dt == np.uint8:
                b[k] = u8_from_hex(v)
            else:
                b[k] = np.array(v, dtype=dt)
        elif k in f64s:
            b[k] = struct.unpack(">d", bytes.fromhex(v))[0]
        elif k in f32s:
            b[k] = f32_from_hex([v])[0]
        else:
            b[k] = int(v)
    return b


def run_pinned(host, rt, rec) -> dict:
    b = deserialize(rec, host)
    if rec["is_day"]:
        return host.run_rad_day(rt, b)
    return host.run_rad_night(rt, b)


# ---------------------------------------------------------------------------
# crop pin set — full-plane raw-bit equality on every output
# ---------------------------------------------------------------------------


class TestRadCropRawBitEquality:
    def test_all_timesteps_all_outputs(self, rt, host, pins):
        failures = []
        n_day = n_night = 0
        for rec in pins["timesteps"]:
            got = run_pinned(host, rt, rec)
            for name, want_hex in rec["outputs"].items():
                g = got[name]
                msg, _ = compare_bits(
                    f32_from_hex(want_hex).reshape(g.shape), g,
                    f"t{rec['t']}:{name}")
                if msg != "PASS":
                    failures.append(msg)
            for name, want_hex in rec.get("next", {}).items():
                g = got[name]
                msg, _ = compare_bits(
                    f32_from_hex(want_hex).reshape(g.shape), g,
                    f"t{rec['t']}:{name}")
                if msg != "PASS":
                    failures.append(msg)
            if rec["is_day"]:
                n_day += 1
            else:
                n_night += 1
        assert not failures, "\n".join(failures[:6])
        assert n_day >= 1 and n_night >= 1, (
            "pin set must cover both the day and the night kernel"
        )

    def test_day_pin_set_exercises_the_walk_borders(self, pins):
        """The crop is 48x48 — walk offsets reach outside the window, so
        the stale-border recursion (temp locals keep the previous step's
        value) is pinned, not just the interior fast path."""
        day = [r for r in pins["timesteps"] if r["is_day"]]
        assert day, "no day timestep pinned"
        crop = day[0]["crop"]
        assert crop["r1"] - crop["r0"] <= 64 and crop["c1"] - crop["c0"] <= 64


# ---------------------------------------------------------------------------
# site_500 anchor — digest equality over the full capture planes
# ---------------------------------------------------------------------------


class TestRadSiteDigests:
    def test_full_site_digests(self, rt, host, pins):
        cap = os.environ.get("SW_T12_T08_CAPTURE", "")
        if not cap or not Path(cap).is_dir():
            pytest.skip(
                f"t08 capture fixtures not available (SW_T12_T08_CAPTURE)"
            )
        # fixture identity: every npz sha256 must match the pins
        for name, want in pins["capture"]["files_sha256"].items():
            p = Path(cap) / name
            assert p.is_file(), f"capture file missing: {p}"
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            assert got == want, f"capture fixture drifted: {name}"

        sys.path.insert(0, str(ULTRA_DIR.parents[1]))
        from solweig_core.numba_cpu.radiation import (
            rad_bundle_from_capture,
            rad_state_from_capture,
            rad_static_from_capture,
        )
        from sw_rad_bundle import (  # noqa: E402
            build_rad_day_bundle,
            build_rad_night_bundle,
        )

        failures = []
        for rec in pins["timesteps"]:
            t = rec["t"]
            st = rad_static_from_capture(Path(cap))
            t_in = rad_bundle_from_capture(Path(cap), t)
            state = rad_state_from_capture(Path(cap), t, rows=st.rows,
                                           cols=st.cols)
            if rec["is_day"]:
                b = build_rad_day_bundle(st, t_in, state)
                got = host.run_rad_day(rt, b)
            else:
                b = build_rad_night_bundle(st, t_in)
                got = host.run_rad_night(rt, b)
            for name, want in rec["site_digests"].items():
                d = plane_digest(got[name])
                if d != want:
                    failures.append(
                        f"t{t}:{name}: digest {d} != {want}")
            assert not failures, "\n".join(failures[:6])
        assert not failures
