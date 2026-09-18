#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the CUDA march parity pins (TASKS T12) as DATA.

Runs on the GPU host (numba present; torch only for the site/variant
modes). Computes the CANONICAL CPU march outputs
(solweig_core/numba_cpu/march.py — the frozen T04 reference) over:

* ``--frozen74 DIR`` — every frozen T03 trace (tables rebuilt from the
  JSON records, scenes from the deterministic synthetic generator that
  the T04 bit gate uses). No torch needed.
* ``--variants`` — the 7-case variant grid completing the frozen 74
  (needs torch: tables come from trace_exporter.capture_trace).
* ``--site`` — the site_500 anchor: the exact planes/amax/solar bits the
  oracle GPU harness fed ``shadowingfunction_wallheight_23`` (a, vegdsm,
  vegdsm2, effective amax, forcing altitude/azimuth per day-timestep),
  with per-t output digests cross-checked against the oracle baseline
  intermediates (universal anchor 6c62bcea... class).

Outputs raw uint32 hex to ``native/cuda/data/march_pins.json`` (small
full-plane sets) and ``march_site_pins.json`` (scene planes + per-t
columns + output sha256 digests — the 500x500 planes are compared by
digest, which is raw-bit equality over the exact float32 bytes).

torch is used ONLY here (reference/table generation); the CUDA runtime
and its tests stay torch-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu.march import (  # noqa: E402
    march_svf_shadow,
    march_wallheight23,
)

OUT = REPO_ROOT / "native" / "cuda" / "data"

F32 = np.float32
U32 = np.uint32


def hexes(arr_f32: np.ndarray) -> list[str]:
    return [f"{int(b):08x}" for b in
            np.ascontiguousarray(arr_f32, dtype=np.float32).view(U32).ravel()]


def hexes_i32(arr: np.ndarray) -> list[str]:
    return [str(int(b)) for b in np.ascontiguousarray(arr, dtype=np.int32)]


def plane_digest(arr_f32: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(arr_f32, dtype=np.float32).tobytes()
    ).hexdigest()


def synthetic_scene(rows: int, cols: int, *, seed: int = 7,
                    negative_dem: bool = False, no_vegetation: bool = False,
                    no_building: bool = False, overlap: bool = False):
    """Deterministic synthetic scene (same generator family as the T04 bit
    gate) — numpy float32 planes, torch-free."""
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    if negative_dem:
        dem -= np.float32(12.0)
    if no_building:
        a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    else:
        a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
        a[3: min(9, rows), 4: min(12, cols)] += np.float32(18.0)
    if no_vegetation:
        canopy = np.zeros((rows, cols), dtype=np.float32)
    else:
        canopy = rng.uniform(0.0, 6.0, (rows, cols)).astype(np.float32)
        canopy[canopy < np.float32(3.0)] = np.float32(0.0)
        canopy[min(10, rows): min(20, rows),
               min(30, cols): min(60, cols)] = np.float32(5.5)
        if overlap:
            canopy[min(2, rows): min(11, rows),
                   min(6, cols): min(16, cols)] = np.float32(25.0)
    vegdem = canopy + dem
    vegdem2 = canopy * np.float32(0.25) + dem
    bush = np.zeros((rows, cols), dtype=np.float32)
    return (np.ascontiguousarray(a, dtype=np.float32),
            np.ascontiguousarray(vegdem, dtype=np.float32),
            np.ascontiguousarray(vegdem2, dtype=np.float32),
            bush)


def _bits_f32(hexbits: str) -> float:
    return float(st.u32_bits_to_f32(st.bits_hex_to_u32(hexbits)))


def _table_columns(table: st.StepTable, *, with_dzprev: bool):
    cols = [
        np.ascontiguousarray(table.dx, dtype=np.int32),
        np.ascontiguousarray(table.dy, dtype=np.int32),
        np.ascontiguousarray(table.dz_bits, dtype=U32).view(np.float32),
    ]
    if with_dzprev:
        cols.append(
            np.ascontiguousarray(table.previous_dz_bits, dtype=U32).view(
                np.float32))
    return cols


def _run_case(kernel_variant: str, table, a, vegdem, vegdem2, bush):
    if kernel_variant == st.KERNEL_SVF_SHADOW:
        outs = march_svf_shadow(table, a, vegdem, vegdem2, bush)
        # returns (sh, vegsh, vbsh) — canonical storage order
    else:
        vegsh, sh, vbsh = march_wallheight23(table, a, vegdem, vegdem2, bush)
        outs = (sh, vegsh, vbsh)  # source returns (vegsh, sh, vbsh)
    return dict(zip(("sh", "vegsh", "vbsh"), outs))


def gen_frozen74(traces_dir: Path) -> list[dict]:
    sets = []
    for path in sorted(traces_dir.glob("*.json")):
        if path.name.startswith("trace_manifest"):
            continue
        trace = json.loads(path.read_text())
        table = st.build_table_from_trace(trace)
        key = table.key
        rows, cols = key.logical_rows, key.logical_cols
        a, vegdem, vegdem2, bush = synthetic_scene(rows, cols, seed=7)
        outs = _run_case(key.kernel_variant, table, a, vegdem, vegdem2, bush)
        with_prev = key.kernel_variant == st.KERNEL_WALLHEIGHT_23
        cols_arr = _table_columns(table, with_dzprev=with_prev)
        entry = {
            "trace_id": table.trace_id,
            "variant": key.kernel_variant,
            "rows": rows,
            "cols": cols,
            "a": hexes(a), "vegdem": hexes(vegdem),
            "vegdem2": hexes(vegdem2),
            "dx": hexes_i32(cols_arr[0]), "dy": hexes_i32(cols_arr[1]),
            "dz": hexes(cols_arr[2]),
            "count": int(len(cols_arr[0])),
            "sh": hexes(outs["sh"]), "vegsh": hexes(outs["vegsh"]),
            "vbsh": hexes(outs["vbsh"]),
        }
        if with_prev:
            entry["dzprev"] = hexes(cols_arr[3])
        sets.append(entry)
    return sets


VARIANT_CASES = [
    # (azimuth, altitude, scale, amplitude, scene kwargs) — the grid from
    # the T04 gate that the frozen 74 does not cover.
    (0.0, 78.0, 2.0, 20.0, {}),
    (37.0, 6.0, 2.0, 15.0, {}),
    (225.0, 42.0, 0.5, 20.0, {"no_vegetation": True}),
    (181.0, 42.0, 0.5, 20.0, {"no_building": True}),
    (133.0, 78.0, 1.0, 25.0, {"overlap": True}),
    (312.0, 6.0, 0.5, 30.0, {"negative_dem": True}),
    (90.0, 90.0, 0.5, 60.0, {"negative_dem": True}),
]


def _boot_optional_env() -> None:
    """cura (GPU host): make the ABI-compatible system osgeo importable and
    stub ONLY gdal_array (numpy-1.x binary that aborts under numpy 2.5) —
    the same bootstrap the oracle baseline uses. No-op elsewhere."""
    import types

    dp = "/usr/lib/python3/dist-packages"
    if Path(dp).is_dir() and dp not in sys.path:
        sys.path.append(dp)
    try:
        import osgeo  # noqa: F401
        import osgeo.gdal  # noqa: F401
    except Exception:
        return
    if "osgeo.gdal_array" not in sys.modules:
        ga = types.ModuleType("osgeo.gdal_array")
        ga._UseExceptions = lambda *a, **k: None
        sys.modules["osgeo.gdal_array"] = ga
        osgeo.gdal_array = ga


def gen_variants() -> list[dict]:
    _boot_optional_env()
    sys.path.insert(0, str(REPO_ROOT / "tests" / "ultrafast"))
    from trace_exporter import capture_trace  # noqa: E402 (torch path)

    sets = []
    for az, alt, scale, amp, kw in VARIANT_CASES:
        for kernel in (st.KERNEL_SVF_SHADOW, st.KERNEL_WALLHEIGHT_23):
            a, vegdem, vegdem2, bush = synthetic_scene(40, 80, seed=11, **kw)
            trace = capture_trace(
                kernel, np.float32(az), np.float32(alt),
                float(np.float32(scale)), 40, 80, np.float32(amp),
                amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
                verify_probes=False,
            )
            table = st.build_table_from_trace(trace)
            outs = _run_case(kernel, table, a, vegdem, vegdem2, bush)
            with_prev = kernel == st.KERNEL_WALLHEIGHT_23
            cols_arr = _table_columns(table, with_dzprev=with_prev)
            entry = {
                "trace_id": f"variant_az{az}_alt{alt}_s{scale}_{kernel}",
                "variant": kernel, "rows": 40, "cols": 80,
                "a": hexes(a), "vegdem": hexes(vegdem),
                "vegdem2": hexes(vegdem2),
                "dx": hexes_i32(cols_arr[0]), "dy": hexes_i32(cols_arr[1]),
                "dz": hexes(cols_arr[2]), "count": int(len(cols_arr[0])),
                "sh": hexes(outs["sh"]), "vegsh": hexes(outs["vegsh"]),
                "vbsh": hexes(outs["vbsh"]),
            }
            if with_prev:
                entry["dzprev"] = hexes(cols_arr[3])
            sets.append(entry)
    return sets


def gen_site(data_dir: Path, oracle_intermediates: Path | None) -> dict:
    """site_500 anchor: mirror the oracle harness composition exactly."""
    _boot_optional_env()
    import torch  # noqa: E402 (reference generation only)

    from solweig_gpu.incremental.cache import SiteCache  # noqa: E402
    from solweig_gpu.incremental.geometry import RasterGrid  # noqa: E402
    from solweig_gpu.incremental.solver import (  # noqa: E402
        compose_full_scene_tensors,
        effective_march_amplitude,
    )
    from solweig_gpu.incremental.trees import TreeLayer  # noqa: E402
    from solweig_gpu.incremental.worker import ExactWorker  # noqa: E402
    sys.path.insert(0, str(REPO_ROOT / "tests" / "ultrafast"))
    from trace_exporter import capture_trace  # noqa: E402

    cache_dir = data_dir / "Input_subset" / "processed_inputs" / \
        "incremental_cache_0_0"
    site_dir = data_dir / "Input_subset" / "processed_inputs"
    cache = SiteCache.load(cache_dir)
    grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                      cache.manifest.origin_x_m, cache.manifest.origin_y_m)
    layer = TreeLayer(cache.tree_base, grid)
    scratch = OUT.parent / "build" / "site_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    worker = ExactWorker(cache, layer, site_dir=str(site_dir),
                         results_root=str(scratch),
                         selected_date_str="2009-08-11")
    forcing = worker.forcing()
    scene = compose_full_scene_tensors(cache, layer)

    # oracle harness (gpu_harness PRIMARY) amax: effective windowed amplitude
    canopy_pos = torch.clamp(scene.canopy, min=0.0)
    amax = effective_march_amplitude(
        scene.a, canopy_pos + scene.a, canopy_pos * 0.25 + scene.a,
        scene_amaxvalue=scene.amaxvalue)

    # run_utci_window internal composition (utci_process.py:668-676),
    # float32 elementwise — bit-identical CPU or GPU
    temp1 = scene.canopy.clone()
    temp1[temp1 < 0.0] = 0.0
    temp2 = scene.dem
    a = scene.a
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    # utci_process.py:670: bush = ~(vegdem2 * vegdem) * vegdem with
    # vegdem = temp1 + temp2, vegdem2 = temp1*0.25 + temp2
    vegdem_full = temp1 + temp2
    vegdem2_full = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2_full * vegdem_full) * vegdem_full

    scale = 1.0 / float(cache.pixel_size_m)
    rows, cols = int(a.shape[0]), int(a.shape[1])

    a_np = np.ascontiguousarray(a.numpy(), dtype=np.float32)
    vegdsm_np = np.ascontiguousarray(vegdsm.numpy(), dtype=np.float32)
    vegdsm2_np = np.ascontiguousarray(vegdsm2.numpy(), dtype=np.float32)
    bush_np = np.ascontiguousarray(bush.numpy(), dtype=np.float32)

    altitudes = np.asarray(forcing.altitude)
    azimuths = np.asarray(forcing.azimuth)
    n_t = altitudes.shape[1]

    per_t = {}
    n_march = 0
    for t in range(n_t):
        alt = float(altitudes[0][t])
        azi = float(azimuths[0][t])
        if alt <= 0.0:
            continue  # night: the march is skipped entirely
        n_march += 1
        trace = capture_trace(
            st.KERNEL_WALLHEIGHT_23, azi, alt, scale, rows, cols,
            float(amax),
            amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
            verify_probes=False,
        )
        table = st.build_table_from_trace(trace)
        outs = march_wallheight23(table, a_np, vegdsm_np, vegdsm2_np, bush_np)
        cols_arr = _table_columns(table, with_dzprev=True)
        per_t[str(t)] = {
            "az_bits": st.f32_bits_hex(azi),
            "alt_bits": st.f32_bits_hex(alt),
            "amax_bits": st.f32_bits_hex(float(amax)),
            "count": int(len(cols_arr[0])),
            "dx": hexes_i32(cols_arr[0]), "dy": hexes_i32(cols_arr[1]),
            "dz": hexes(cols_arr[2]), "dzprev": hexes(cols_arr[3]),
            "sh_sha256": plane_digest(outs[1]),
            "vegsh_sha256": plane_digest(outs[0]),
            "vbsh_sha256": plane_digest(outs[2]),
        }

    site = {
        "rows": rows, "cols": cols,
        "amax_bits": st.f32_bits_hex(float(amax)),
        "scale_repr": repr(scale),
        "n_timesteps": int(n_t), "n_march_timesteps": n_march,
        "a": hexes(a_np), "vegdsm": hexes(vegdsm_np),
        "vegdsm2": hexes(vegdsm2_np),
        "per_t": per_t,
    }

    # cross-check against the oracle baseline intermediates when supplied:
    # march sh/vegsh are bit-identical GPU/CPU (lead contract), so the CPU
    # canonical digests MUST equal the oracle per-t digests. The oracle
    # recorder's t label is the loop index + 1 (verified: oracle t=7 digest
    # == forcing-index-8 digest, oracle t=12 == index 13, exact match) —
    # align with that offset.
    if oracle_intermediates is not None and oracle_intermediates.is_file():
        inter = json.loads(oracle_intermediates.read_text())
        checks = []
        for entry in inter["march"]:
            label = int(entry["t"])
            src_i = label + 1
            key = str(src_i)
            rec_mine = per_t.get(key)
            if rec_mine is None:
                checks.append({"t": label, "status": "MISSING_CPU_MARCH"})
                continue
            want_sh = entry["returns"]["sh"]["sha256"]
            want_vegsh = entry["returns"]["vegsh"]["sha256"]
            ok_sh = rec_mine["sh_sha256"] == want_sh
            ok_vegsh = rec_mine["vegsh_sha256"] == want_vegsh
            checks.append({"t": label, "forcing_index": src_i,
                           "sh": ok_sh, "vegsh": ok_vegsh,
                           "want_sh": want_sh, "want_vegsh": want_vegsh})
            rec_mine["oracle_t_label"] = label
        site["oracle_cross_check"] = checks
        n_ok = sum(1 for c in checks if c.get("sh") and c.get("vegsh"))
        print(f"site oracle cross-check: {n_ok}/{len(checks)} timesteps "
              f"bit-equal (sh+vegsh)")
    return site


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen74", type=Path, default=None,
                    help="dir with the frozen T03 trace JSONs")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--site", type=Path, default=None,
                    help="oracle data dir (Input_subset parent)")
    ap.add_argument("--oracle-intermediates", type=Path, default=None,
                    help="gpu_harness intermediates.json for cross-check")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sets: dict[str, object] = {}
    if args.frozen74:
        frozen = gen_frozen74(args.frozen74)
        sets["frozen74"] = frozen
        print(f"frozen74: {len(frozen)} traces")
    if args.variants:
        var = gen_variants()
        sets["variants"] = var
        print(f"variants: {len(var)} cases")
    if args.site:
        site = gen_site(args.site, args.oracle_intermediates)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "march_site_pins.json").write_text(json.dumps({
            "schema": "sw-cuda-march-site-pins/1",
            "generator": "native/cuda/tools/gen_march_pins.py",
            "reference": "solweig_core/numba_cpu/march.py (canonical CPU)",
            "site": site,
        }, indent=1))
        print(f"wrote {OUT / 'march_site_pins.json'}")
    if args.out or sets:
        path = args.out or (OUT / "march_pins.json")
        payload = {
            "schema": "sw-cuda-march-pins/1",
            "generator": "native/cuda/tools/gen_march_pins.py",
            "reference": "solweig_core/numba_cpu/march.py (canonical CPU)",
            "numpy": np.__version__,
            "sets": sets,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1))
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
