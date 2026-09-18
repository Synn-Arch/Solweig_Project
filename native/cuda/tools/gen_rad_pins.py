#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the CUDA radiation parity pins (TASKS T12) as DATA.

Runs on the GPU host (numba present; torch NOT needed — the T08 capture
npz files carry every frozen bundle bit). For each captured timestep
(defaults t=1,5,12,23):

* runs the CANONICAL CPU fused kernel (solweig_core/numba_cpu/
  radiation.py fused_radiation_timestep) on the FULL site planes and
  records per-plane sha256 digests (raw-bit equality by digest);
* crops a fixed window, runs the same canonical kernel on the crop, and
  pins the crop's full input/output set as raw hex (first-mismatch
  debugging at raw-bit level);
* records the capture files' sha256 so tests can verify fixture
  identity before trusting the full-site digest gate.

The bundle assembly (sunwall, albshadow, guard/rank tables, jE..jN,
timeadd branch) is the SHARED module tests/ultrafast/sw_rad_bundle.py —
the generator and the tests build byte-identical kernel inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "ultrafast"))
sys.path.insert(0, str(REPO_ROOT / "native" / "cuda"))

from solweig_core.numba_cpu.radiation import (  # noqa: E402
    fused_radiation_timestep,
    rad_bundle_from_capture,
    rad_state_from_capture,
    rad_static_from_capture,
)
from sw_rad_bundle import build_rad_day_bundle, build_rad_night_bundle  # noqa: E402
import host as cuda_host  # noqa: E402  (spec tables only; torch-free)

OUT = REPO_ROOT / "native" / "cuda" / "data"

U32 = np.uint32
F32 = np.float32

DAY_PIN_MAP = {
    "tmrt": "Tmrt", "kdown": "Kdown", "kup": "Kup", "ldown": "Ldown",
    "lup": "Lup", "ke": "Keast", "ks": "Ksouth", "kw": "Kwest",
    "kn": "Knorth",
    "le": "Least", "ls": "Lsouth", "lw": "Lwest", "ln": "Lnorth",
    "ksidei": "KsideI", "tgout": "TgOut", "lside": "Lside",
    "ksided": "KsideD", "drad": "dRad", "kside": "Kside",
}
NEXT_PIN_MAP = {
    "n_lup": "Tgmap1", "n_e": "Tgmap1E", "n_s": "Tgmap1S",
    "n_w": "Tgmap1W", "n_n": "Tgmap1N", "n_tg": "TgOut1",
}
NIGHT_PIN_MAP = {
    "tmrt": "Tmrt", "ldown": "Ldown", "lside": "Lside",
    "le": "Least", "ls": "Lsouth", "lw": "Lwest", "ln": "Lnorth",
}


def hexes(arr_f32: np.ndarray) -> list[str]:
    return [f"{int(b):08x}" for b in
            np.ascontiguousarray(arr_f32, dtype=F32).view(U32).ravel()]


def bytes_hex(arr_u8: np.ndarray) -> list[str]:
    return [f"{int(b):02x}" for b in
            np.ascontiguousarray(arr_u8, dtype=np.uint8).ravel()]


def ints_hex(arr: np.ndarray) -> list[int]:
    return [int(b) for b in np.ascontiguousarray(arr).ravel()]


def f32_bits(x) -> str:
    return f"{int(np.array([F32(x)], dtype=F32).view(U32)[0]):08x}"


def f64_bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def plane_digest(arr_f32: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(arr_f32, dtype=F32).tobytes()).hexdigest()


def serialize_bundle(b: dict, inputs, f32_scalars, f64_scalars,
                     int_scalars) -> dict:
    out = {}
    for key, dt in inputs:
        arr = np.ascontiguousarray(b[key], dtype=dt)
        if dt == np.float32:
            out[key] = hexes(arr)
        elif dt == np.uint8:
            out[key] = bytes_hex(arr)
        else:
            out[key] = ints_hex(arr)
    for key in f32_scalars:
        out[key] = f32_bits(b[key])
    for key in f64_scalars:
        out[key] = f64_bits(b[key])
    for key in int_scalars:
        out[key] = int(b[key])
    return out


def crop_st(st, sl):
    s = lambda a: np.ascontiguousarray(a[sl])
    return replace(
        st, rows=sl[0].stop - sl[0].start, cols=sl[1].stop - sl[1].start,
        buildings=s(st.buildings), walls=s(st.walls),
        aspect_rad=s(st.aspect_rad), wallbol=s(st.wallbol),
        alb_grid=s(st.alb_grid), emis_grid=s(st.emis_grid),
        svfbuveg=s(st.svfbuveg), shmat_packed=s(st.shmat_packed),
        vegshmat_packed=s(st.vegshmat_packed),
        vbshvegshmat_packed=s(st.vbshvegshmat_packed), diffsh=s(st.diffsh),
    )


def crop_t_in(t_in, sl):
    s = lambda a: np.ascontiguousarray(a[sl])
    updates = {}
    for k in ("vegsh", "sh", "wallsun", "shadow", "Tg_pre", "Tg",
              "Lup_pre", "gvflup_extra", "F_sh", "sunlit_packed",
              "shaded_packed", "night_Lup"):
        v = getattr(t_in, k, None)
        if v is not None:
            updates[k] = s(v)
    return replace(t_in, **updates)


def crop_state(state, sl):
    s = lambda a: np.ascontiguousarray(a[sl])
    return replace(
        state, Tgmap1=s(state.Tgmap1), Tgmap1E=s(state.Tgmap1E),
        Tgmap1S=s(state.Tgmap1S), Tgmap1W=s(state.Tgmap1W),
        Tgmap1N=s(state.Tgmap1N), TgOut1=s(state.TgOut1),
    )


def gen_timestep(cap_dir: Path, t: int, sl) -> dict:
    st = rad_static_from_capture(cap_dir)
    t_in = rad_bundle_from_capture(cap_dir, t)
    state = rad_state_from_capture(cap_dir, t, rows=st.rows, cols=st.cols)

    # full-site canonical CPU run -> digests (raw-bit equality by digest)
    full_out, full_next = fused_radiation_timestep(st, t_in, state)
    if t_in.is_day:
        site_digests = {k: plane_digest(full_out[v])
                        for k, v in DAY_PIN_MAP.items()}
        site_next = {k: plane_digest(getattr(full_next, v))
                     for k, v in NEXT_PIN_MAP.items()}
    else:
        site_digests = {k: plane_digest(full_out[v])
                        for k, v in NIGHT_PIN_MAP.items()}
        site_next = {}

    # crop canonical CPU run -> raw hex pins
    st_c = crop_st(st, sl)
    t_c = crop_t_in(t_in, sl)
    state_c = crop_state(state, sl)
    out_c, next_c = fused_radiation_timestep(st_c, t_c, state_c)

    rec = {"t": t, "is_day": bool(t_in.is_day),
           "crop": {"r0": sl[0].start, "r1": sl[0].stop,
                    "c0": sl[1].start, "c1": sl[1].stop},
           "site_digests": site_digests, "site_next_digests": site_next}
    if t_in.is_day:
        b = build_rad_day_bundle(st_c, t_c, state_c)
        rec["inputs"] = serialize_bundle(
            b, cuda_host._RAD_DAY_INPUTS, cuda_host._RAD_DAY_F32_SCALARS,
            cuda_host._RAD_DAY_F64_SCALARS, cuda_host._RAD_DAY_INT_SCALARS)
        rec["outputs"] = {k: hexes(out_c[v])
                          for k, v in DAY_PIN_MAP.items()}
        rec["next"] = {k: hexes(getattr(next_c, v))
                       for k, v in NEXT_PIN_MAP.items()}
    else:
        b = build_rad_night_bundle(st_c, t_c)
        rec["inputs"] = serialize_bundle(
            b, cuda_host._RAD_NIGHT_INPUTS, (),
            cuda_host._RAD_NIGHT_F64_SCALARS,
            cuda_host._RAD_NIGHT_INT_SCALARS)
        rec["outputs"] = {k: hexes(out_c[v])
                          for k, v in NIGHT_PIN_MAP.items()}
        rec["next"] = {}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True,
                    help="t08 capture dir (static.npz, tXX.npz, ...)")
    ap.add_argument("--ts", default="1,5,12,23")
    ap.add_argument("--crop", default="120:168,200:248",
                    help="r0:r1,c0:c1 crop window for the hex pin set")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rs, cs = args.crop.split(",")
    r0, r1 = (int(x) for x in rs.split(":"))
    c0, c1 = (int(x) for x in cs.split(":"))
    sl = (slice(r0, r1), slice(c0, c1))

    files = {}
    for p in sorted(args.capture.iterdir()):
        if p.suffix == ".npz":
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    steps = [gen_timestep(args.capture, int(t), sl)
             for t in (int(x) for x in args.ts.split(","))]

    path = args.out or (OUT / "rad_pins.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "sw-cuda-rad-pins/1",
        "generator": "native/cuda/tools/gen_rad_pins.py",
        "reference": "solweig_core/numba_cpu/radiation.py (canonical CPU)",
        "numpy": np.__version__,
        "capture": {"dir_recorded": str(args.capture), "files_sha256": files},
        "crop": {"r0": r0, "r1": r1, "c0": c0, "c1": c1},
        "timesteps": steps,
    }, indent=1))
    n_day = sum(1 for s in steps if s["is_day"])
    print(f"wrote {path} ({path.stat().st_size} bytes, "
          f"{len(steps)} timesteps: {n_day} day / {len(steps) - n_day} night)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
