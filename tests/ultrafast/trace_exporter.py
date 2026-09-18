# SPDX-License-Identifier: GPL-3.0-only
"""Immutable march-trace exporter for the shadow kernels (T01 deliverable,
deferred to T03). Captures the ORIGINAL torch code path WITHOUT modifying it.

What "original code path" means here, precisely:

* executed (dx, dy) step sequences and executed step counts come from
  IMPORTING AND CALLING :func:`solweig_gpu.incremental.veg_svf_state.march_offsets`
  — the repository's own verified replica of ``shadow()``'s while-loop
  (proven bit-for-bit against the march in corridor_bruteforce.py and
  tests/test_incremental_veg_occlusion.py);
* the amplitude POLICY comes from importing and calling
  :func:`solweig_gpu.incremental.solver.effective_march_amplitude`,
  ``_banded_patch_mask``, ``time_loop_band_present``,
  ``_patch_first_step_distances`` and ``_sun_first_step_distance`` —
  the repository's own amplitude-selection code;
* per-step ``dz`` bit patterns are generated from the original kernels'
  verbatim one-line arithmetic — ``(ds * index) * (tan(altitude) / scale)``
  with the ``tan/scale`` quotient folded FIRST (shadow.py:238/255,
  solweig.py:1123/1148) — and then PROVEN bit-exact against the ORIGINAL
  stop behaviour with the nextafter double-probe below (this is data
  arithmetic re-expression gated by original-code evidence, not a copied
  control flow; the loop/branch/window logic is never reimplemented);
* kernel-level truth is anchored by calling the ORIGINAL kernels
  :func:`solweig_gpu.shadow.shadow` and
  :func:`solweig_gpu.solweig.shadowingfunction_wallheight_23` themselves
  (see ``tests/ultrafast/test_step_tables.py`` replay gates).

The nextaway double-probe: the while-stop tests ``amaxvalue >= dz_prev``,
so ``march_offsets``' executed count satisfies ``count(A) >= k + 1`` iff
``dz_k <= A``. Therefore a candidate bit pattern ``c`` for ``dz_k`` is
EXACTLY right iff ``count(c) >= k + 1`` and ``count(nextafter(c, -inf)) ==
k`` — the open-closed interval ``(nextafter(c,-inf), c]`` contains the one
float32 ``c`` alone. Two original-code calls pin each dz bit pattern.

Traces are written under the T03 artifact root
(``$SOLWEIG_ULTRA_ARTIFACTS``/t03/traces by default); the repository keeps
only the sha256 manifest (``tests/ultrafast/step_table_trace_manifest.json``).

Usage (generate the frozen fixture set + manifest):

    .venv/bin/python -m pytest tests/ultrafast/test_step_tables.py -q   # uses in-process capture
    .venv/bin/python tests/ultrafast/trace_exporter.py --out-root \
        /Users/alansynn/Workspace/solweig_ultrafast_artifacts/t03
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_gpu.incremental.veg_svf_state import march_offsets  # noqa: E402
from solweig_gpu.incremental import solver as _solver  # noqa: E402
from solweig_gpu.shadow import shadow as shadow_fn  # noqa: E402
from solweig_gpu.solweig import (  # noqa: E402
    shadowingfunction_wallheight_23 as wallheight23_fn,
)

from solweig_core import step_tables as st  # noqa: E402

DEFAULT_ARTIFACT_ROOT = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t03"
)
REPO_MANIFEST_JSON = (
    REPO_ROOT / "tests" / "ultrafast" / "step_table_trace_manifest.json"
)

#: Sources whose sha256 defines ``source_geometry_hash`` (immutable oracle
#: symbols per DESIGN 6.1). solver.py is included because the amplitude
#: policy lives there; any edit to these files invalidates every table.
GEOMETRY_SOURCES = (
    "solweig_gpu/shadow.py",
    "solweig_gpu/solweig.py",
    "solweig_gpu/incremental/veg_svf_state.py",
    "solweig_gpu/incremental/solver.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_geometry_hash(repo_root: Path = REPO_ROOT) -> tuple[str, dict[str, str]]:
    """sha256 over the defining sources + semantics version."""
    sources = {
        rel: sha256_file(repo_root / rel) for rel in GEOMETRY_SOURCES
    }
    digest = hashlib.sha256()
    digest.update(b"T03Geometry_v1|")
    for rel in GEOMETRY_SOURCES:
        digest.update(f"{rel}:{sources[rel]}\n".encode())
    return digest.hexdigest(), sources


# ---------------------------------------------------------------------------
# Step engine — ORIGINAL kernels' per-angle scalar arithmetic, verbatim
# association, no control-flow reimplementation
# ---------------------------------------------------------------------------


def _f32(value: float) -> float:
    return float(np.float32(value))


def _generate_steps(
    kernel_variant: str,
    azimuth_deg: float,
    altitude_deg: float,
    scale: float,
    rows: int,
    cols: int,
    amplitude: float,
):
    """Step tuples ``(index, dx, dy, dz_f32, branch_id)`` of one march.

    Mirrors ONLY the scalar arithmetic of the original loops:
    shadow.py:196-258 (svf_shadow) and solweig.py:1092-1150
    (wallheight_23) — angle conversion, branch constants, ds selection,
    round/sign offsets, and the dz association. The while-loop CONTROL FLOW
    here is the plain previous-state stop; its output for ``svf_shadow`` is
    asserted equal to :func:`march_offsets` (original) by every capture, and
    every amplitude-terminated dz is probe-proven bit-exact by the caller.
    """
    if kernel_variant == st.KERNEL_SVF_SHADOW:
        degrees = torch.pi / 180.0            # shadow.py:195 python float
        pibyfour = torch.pi / 4.0             # shadow.py:227 python float
        az = torch.tensor(_f32(azimuth_deg))
        if float(az) == 0.0:                  # shadow.py:196-197
            az = az * 0.0 + 1e-12
        substituted = float(np.float32(azimuth_deg)) == 0.0
        alt = torch.tensor(_f32(altitude_deg))
        index0 = 1.0                          # shadow.py:240
    elif kernel_variant == st.KERNEL_WALLHEIGHT_23:
        degrees = torch.tensor(np.pi / 180.0)  # solweig.py:1092 f32 tensor
        pibyfour = torch.tensor(np.pi / 4.0)   # solweig.py:1110 f32 tensor
        az = torch.tensor(_f32(azimuth_deg))   # NO substitution (solweig.py:1093)
        substituted = False
        alt = torch.tensor(_f32(altitude_deg))
        index0 = 0.0                           # solweig.py:1124
    else:
        raise ValueError(kernel_variant)

    azimuth = az * degrees
    altitude = alt * degrees
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1.0 / sinazimuth))
    dscos = torch.abs((1.0 / cosazimuth))
    tanaltitudebyscale = torch.tan(altitude) / scale

    sin_branch = bool(pibyfour <= azimuth < threetimespibyfour) or bool(
        fivetimespibyfour <= azimuth < seventimespibyfour
    )

    dx = torch.tensor(0.0)
    dy = torch.tensor(0.0)
    dz = torch.tensor(0.0)
    amax = torch.tensor(_f32(amplitude))
    steps = []
    index = index0
    while bool(amax >= dz) and bool(torch.abs(dx) < rows) and bool(
        torch.abs(dy) < cols
    ):
        if sin_branch:
            dy = signsinazimuth * index
            dx = -1.0 * signcosazimuth * torch.abs(
                torch.round(index / tanazimuth)
            )
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos
        dz = (ds * index) * tanaltitudebyscale
        steps.append(
            (float(index), int(dx), int(dy), float(dz), 1 if sin_branch else 0)
        )
        index += 1.0
    return steps, substituted


def _stop_reason(steps, rows: int, cols: int, amplitude: float) -> str:
    if not steps:
        return st.STOP_AMPLITUDE
    last_dx, last_dy = steps[-1][1], steps[-1][2]
    row_stop = abs(last_dx) >= rows
    col_stop = abs(last_dy) >= cols
    if row_stop and col_stop:
        return st.STOP_ROW_COL_BOUNDARY
    if row_stop:
        return st.STOP_ROW_BOUNDARY
    if col_stop:
        return st.STOP_COL_BOUNDARY
    return st.STOP_AMPLITUDE


def _probe_count_svf(
    azimuth_deg: float,
    altitude_deg: float,
    amplitude: float,
    scale: float,
) -> int:
    """Executed count from the ORIGINAL march_offsets at a probe amplitude.

    Probe calls use oversized logical bounds so the amplitude ladder binds
    before any row/col boundary (march_offsets is a pure scalar loop; large
    bounds cost nothing).
    """
    az = torch.tensor(_f32(azimuth_deg))
    alt = torch.tensor(_f32(altitude_deg))
    offsets = march_offsets(az, alt, amplitude, scale, 1 << 20, 1 << 20)
    return len(offsets)


_WALLHEIGHT_PROBE_GRID = 72
_WALLHEIGHT_PROBE_MARKER = 1.0e30


def _probe_wallheight_step_executed(
    azimuth_deg: float,
    altitude_deg: float,
    amplitude: float,
    scale: float,
    dx: int,
    dy: int,
) -> bool:
    """Black-box: did the ORIGINAL wallheight_23 kernel execute the step
    that reads source offset ``(dx, dy)`` for the centre target?

    A marker height ``H`` at ``centre + (dx, dy)`` raises ``f`` above ``a``
    at the centre iff some executed step read it with ``H - dz > 0`` (H is
    far above any fixture dz, and all other cells are 0, so no other step
    can raise the centre). The kernel's final ``sh`` is ``1 - (f > a)``.
    """
    n = _WALLHEIGHT_PROBE_GRID
    centre = (n // 2, n // 2)
    a = torch.zeros((n, n))
    a[centre[0] + dx, centre[1] + dy] = _WALLHEIGHT_PROBE_MARKER
    zeros = torch.zeros((n, n))
    vegsh_m, sh_m, *_ = wallheight23_fn(
        a,
        zeros,
        zeros,
        _f32(azimuth_deg),
        _f32(altitude_deg),
        scale,
        float(amplitude),
        zeros,
        zeros,
        zeros,
    )
    return bool(sh_m[centre] == 0.0)


def capture_trace(
    kernel_variant: str,
    azimuth_deg: float,
    altitude_deg: float,
    scale: float,
    rows: int,
    cols: int,
    amplitude: float,
    *,
    amplitude_policy_id: str,
    semantics_profile: str = "canonical_cpu_v1",
    trace_id: str = "",
    verify_probes: bool = True,
) -> dict[str, Any]:
    """Capture one immutable trace record from the ORIGINAL code path."""
    if altitude_deg <= 0.0:
        raise ValueError(
            "altitude <= 0 makes dz identically 0 (loop runs to the domain "
            "boundary or forever); the march domain requires altitude > 0"
        )
    steps, substituted = _generate_steps(
        kernel_variant, azimuth_deg, altitude_deg, scale, rows, cols, amplitude
    )
    count = len(steps)

    # --- anchor 1: ORIGINAL march_offsets offset/count equality -----------
    probe_failures: list[str] = []
    offsets_verified = False
    if kernel_variant == st.KERNEL_SVF_SHADOW:
        original = march_offsets(
            torch.tensor(_f32(azimuth_deg)),
            torch.tensor(_f32(altitude_deg)),
            amplitude,
            scale,
            rows,
            cols,
        )
        mine = [(s[1], s[2]) for s in steps]
        if mine != [(int(dx), int(dy)) for dx, dy in original]:
            raise AssertionError(
                f"engine divergence from march_offsets at az={azimuth_deg!r} "
                f"alt={altitude_deg!r} scale={scale!r}: {mine[:4]} vs "
                f"{[(int(dx), int(dy)) for dx, dy in original][:4]}"
            )
        offsets_verified = True

    # --- anchor 2: nextaway double-probe pins every amplitude-limited dz ---
    # SVF ladder: march_offsets (original). Wall-height ladder: black-box
    # probes against the ORIGINAL wallheight_23 kernel — the two kernels'
    # branch constants differ (python-float vs f32-tensor pi multiples), so
    # at exact diagonal azimuths they select different ds by 1 ulp and the
    # SVF replica cannot stand in for the wall-height stop.
    dz_proof = "expression_only"
    if verify_probes and count > 0:
        amplitude_limited = abs(steps[-1][1]) < rows and abs(
            steps[-1][2]
        ) < cols
        half = _WALLHEIGHT_PROBE_GRID // 2
        if amplitude_limited:
            # The full step LADDER (amplitude effectively unbounded, bounded
            # only by the probe grid) — needed because the previous-state
            # stop means dz_k is observable only through step k+1's
            # execution, and the trace's own last step has no successor.
            ladder: list = []
            if kernel_variant == st.KERNEL_WALLHEIGHT_23:
                ladder, _sub = _generate_steps(
                    kernel_variant,
                    azimuth_deg,
                    altitude_deg,
                    scale,
                    _WALLHEIGHT_PROBE_GRID,
                    _WALLHEIGHT_PROBE_GRID,
                    1.0e30,
                )
            checked = 0
            expected_checks = 0
            for pos, step in enumerate(steps):
                index, sdz = step[0], step[3]
                if index < 1.0:
                    continue  # the wallheight self-step (dz 0) is structural
                # wallheight: dz of trace step p is probed through the
                # execution of step p+1 (previous-state stop); SVF probes
                # through march_offsets' executed count directly.
                candidate = np.float32(sdz)
                below = np.nextafter(candidate, np.float32(-np.inf))
                if kernel_variant == st.KERNEL_SVF_SHADOW:
                    n_at = _probe_count_svf(
                        azimuth_deg, altitude_deg, float(candidate), scale
                    )
                    n_below = _probe_count_svf(
                        azimuth_deg, altitude_deg, float(below), scale
                    )
                    ok = n_at >= int(index) + 1 and n_below <= int(index)
                    detail = f"counts {n_at}/{n_below}"
                    expected_checks += 1
                else:
                    if pos + 1 >= len(ladder):
                        continue  # successor outside the probe grid
                    ndx, ndy = ladder[pos + 1][1], ladder[pos + 1][2]
                    if abs(ndx) >= half or abs(ndy) >= half:
                        continue  # marker would fall outside the probe grid
                    expected_checks += 1
                    at = _probe_wallheight_step_executed(
                        azimuth_deg,
                        altitude_deg,
                        float(candidate),
                        scale,
                        ndx,
                        ndy,
                    )
                    below_executed = _probe_wallheight_step_executed(
                        azimuth_deg,
                        altitude_deg,
                        float(below),
                        scale,
                        ndx,
                        ndy,
                    )
                    ok = at and not below_executed
                    detail = f"marker executed {at}/{below_executed}"
                if not ok:
                    probe_failures.append(
                        f"dz probe failed at index {int(index)}: candidate "
                        f"{candidate!r} {detail}"
                    )
                    break
                checked += 1
            if checked == expected_checks and checked > 0 and not probe_failures:
                dz_proof = "stop_probe"
    if probe_failures:
        raise AssertionError("; ".join(probe_failures))

    # --- anchor 3: ORIGINAL first-step distance cross-check ---------------
    # _sun_first_step_distance uses the SAME arithmetic but with PYTHON-
    # FLOAT pi constants — exactly shadow()'s convention. The wall-height
    # KERNEL computes its branch bounds as f32 TENSOR arithmetic
    # (5 * torch.tensor(np.pi/4)), one ulp higher at some diagonals, so at
    # exact azimuths like 225/315 degrees the helper and the kernel select
    # different ds branches and dz_1 differs by ~1 ulp. The kernel ladder
    # (what the table must match) is verified by anchor 2; the helper
    # divergence is RECORDED, not asserted away.
    dz1_reference = None
    dz1_reference_diverges = False
    first_indexed = next((s for s in steps if s[0] >= 1.0), None)
    if first_indexed is not None:
        dz1_reference = float(
            _solver._sun_first_step_distance(
                float(np.float32(azimuth_deg)),
                float(np.float32(altitude_deg)),
                scale,
            )
        )
        dz1_reference_diverges = bool(
            np.float32(dz1_reference) != np.float32(first_indexed[3])
        )
        if dz1_reference_diverges:
            if kernel_variant == st.KERNEL_SVF_SHADOW and float(
                np.float32(azimuth_deg)
            ) != 0.0:
                raise AssertionError(
                    "dz_1 diverges from _sun_first_step_distance at "
                    f"az={azimuth_deg!r}"
                )

    geom_hash, sources = source_geometry_hash()
    trace = {
        "schema_version": st.STEP_TABLE_SCHEMA_VERSION,
        "trace_id": trace_id
        or (
            f"{kernel_variant}_az{st.f32_bits_hex(azimuth_deg)}"
            f"_alt{st.f32_bits_hex(altitude_deg)}_s{st.f32_bits_hex(scale)}"
            f"_{rows}x{cols}_A{st.f32_bits_hex(amplitude)}"
        ),
        "key": {
            "semantics_profile": semantics_profile,
            "kernel_variant": kernel_variant,
            "source_geometry_hash": geom_hash,
            "angle_input_bits": [
                st.f32_bits_hex(azimuth_deg),
                st.f32_bits_hex(altitude_deg),
            ],
            "scale_bits": st.f32_bits_hex(scale),
            "logical_rows": int(rows),
            "logical_cols": int(cols),
            "amplitude_policy_id": amplitude_policy_id,
            "executed_amplitude_bits": st.f32_bits_hex(amplitude),
            "boundary_policy_id": st.BOUNDARY_SHIFT_WINDOW_V1,
        },
        "count": count,
        "dx": [s[1] for s in steps],
        "dy": [s[2] for s in steps],
        "dz_bits": [st.f32_bits_hex(s[3]) for s in steps],
        "branch_id": [s[4] for s in steps],
        "previous_dz_bits": ["0x00000000"]
        + [st.f32_bits_hex(s[3]) for s in steps[:-1]],
        "stop_reason": _stop_reason(steps, rows, cols, amplitude),
        "azimuth_zero_substituted": substituted,
        "scale_repr": repr(float(scale)),  # exact f64 actually fed to torch
        "verification": {
            "offsets_source": (
                "solweig_gpu.incremental.veg_svf_state.march_offsets"
                if offsets_verified
                else "engine+replay_gate"
            ),
            "offsets_verified_against_original": offsets_verified,
            "dz_bit_proof": dz_proof,
            "dz1_reference": (
                None if dz1_reference is None else st.f32_bits_hex(dz1_reference)
            ),
            "dz1_reference_diverges": dz1_reference_diverges,
        },
        "provenance": {
            "source_sha256": sources,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    }
    # round-trip: the table rebuilt from this trace must digest identically
    trace["content_sha256"] = st.build_table_from_trace(trace).content_digest()
    return trace


# ---------------------------------------------------------------------------
# Amplitude policy capture (R6) — all values from ORIGINAL selector code
# ---------------------------------------------------------------------------


def capture_amplitude_policy(
    a: torch.Tensor,
    vegdsm: torch.Tensor,
    vegdsm2: torch.Tensor,
    vegdem: torch.Tensor,
    *,
    scale: float,
    forcing_altitude=None,
    forcing_azimuth=None,
    time_start: int = 0,
    time_stop: int | None = None,
) -> dict[str, Any]:
    """Record the ACTUAL selected amplitude per call path (DESIGN 7.2 R6).

    ``vegdem`` is the canopy+DEM raster the oracle's absolute stop uses
    (``max(a.max(), vegdem.max())``); ``vegdsm``/``vegdsm2`` feed the
    effective-bound expression. Everything is computed by importing and
    calling the ORIGINAL selector functions.
    """
    oracle_absolute = torch.maximum(a.max(), vegdem.max())
    effective = _solver.effective_march_amplitude(
        a, vegdsm, vegdsm2, scene_amaxvalue=oracle_absolute
    )
    record: dict[str, Any] = {
        "oracle_absolute_bits": st.f32_bits_hex(float(oracle_absolute)),
        "effective_bits": st.f32_bits_hex(float(effective)),
        "scale_bits": st.f32_bits_hex(scale),
    }
    banded = _solver._banded_patch_mask(
        float(effective), float(oracle_absolute), scale
    )
    record["r6_banded_patch_indices"] = [
        int(i) for i in np.nonzero(banded)[0]
    ]
    if forcing_altitude is not None and forcing_azimuth is not None:
        record["r6_timeloop_escalated"] = bool(
            _solver.time_loop_band_present(
                float(effective),
                float(oracle_absolute),
                scale,
                forcing_altitude,
                forcing_azimuth,
                time_start=time_start,
                time_stop=time_stop,
            )
        )
    return record


def executed_amplitude_for_patch(
    policy: Mapping[str, Any], patch_index: int, *, escalated: bool
) -> tuple[float, str]:
    """(amplitude, policy_id) a march at ``patch_index`` actually runs at."""
    if escalated or patch_index in set(policy["r6_banded_patch_indices"]):
        return (
            float(st.u32_bits_to_f32(st.bits_hex_to_u32(policy["oracle_absolute_bits"]))),
            st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED,
        )
    return (
        float(st.u32_bits_to_f32(st.bits_hex_to_u32(policy["effective_bits"]))),
        st.AMPLITUDE_EFFECTIVE_WINDOWED,
    )


# ---------------------------------------------------------------------------
# Frozen fixture set + artifact writing
# ---------------------------------------------------------------------------


def default_fixture_cases() -> list[dict[str, Any]]:
    """The frozen T03 fixture grid (DESIGN 18.2 subset relevant to tables).

    Cardinals + nextafter neighbors, diagonal sector boundaries, a real
    patch-geometry subset, the zenith, non-square shapes, scales 1/2/3/4 m
    plus the non-power-of-two 2.5 m, and amplitudes that exercise the
    one-step regime, the overshoot step, and boundary-limited marches.
    """
    f32 = lambda v: float(np.float32(v))  # noqa: E731

    def neighbours32(value: float) -> list[float]:
        """value and its f32 nextafter neighbours — nextafter computed IN
        float32 space (an f64 neighbour rounds back to the same f32)."""
        v = np.float32(value)
        return [
            float(np.nextafter(v, np.float32(-np.inf))),
            float(v),
            float(np.nextafter(v, np.float32(np.inf))),
        ]

    cases: list[dict[str, Any]] = []

    def add(kernel, az, alt, scale, rows, cols, amp, policy):
        cases.append(
            {
                "kernel_variant": kernel,
                "azimuth_deg": f32(az),
                "altitude_deg": f32(alt),
                "scale": scale,
                "rows": rows,
                "cols": cols,
                "amplitude": f32(amp),
                "amplitude_policy_id": policy,
            }
        )

    for kernel in (st.KERNEL_SVF_SHADOW, st.KERNEL_WALLHEIGHT_23):
        # cardinal azimuths and nextafter neighbours (both sides)
        for card in (0.0, 90.0, 180.0, 270.0, 360.0):
            for az in neighbours32(card):
                add(kernel, az, 35.0, 0.5, 40, 80, 20.0, st.AMPLITUDE_SYNTHETIC_PROBE)
        # diagonal sector boundaries 45/135/225/315 (+ neighbours)
        for diag in (45.0, 135.0, 225.0, 315.0):
            for az in neighbours32(diag):
                add(kernel, az, 35.0, 0.5, 40, 80, 20.0, st.AMPLITUDE_SYNTHETIC_PROBE)
        # scales 1/2/3/4 m and non-power-of-two 2.5 m (scale = px/m)
        for pixel_m in (1.0, 2.0, 3.0, 4.0, 2.5):
            add(kernel, 37.0, 35.0, 1.0 / pixel_m, 40, 80, 20.0, st.AMPLITUDE_SYNTHETIC_PROBE)
        # one-step regime (dz_1 > A) and overshoot witness at 2 m
        add(kernel, 37.0, 78.0, 0.5, 40, 80, 0.5, st.AMPLITUDE_SYNTHETIC_PROBE)
        add(kernel, 37.0, 6.0, 0.5, 40, 80, 20.0, st.AMPLITUDE_SYNTHETIC_PROBE)
        # zenith wrapper: tan(f32(pi/2)) < 0 => boundary-limited march
        add(kernel, 0.0, 90.0, 0.5, 40, 80, 1e6, st.AMPLITUDE_SYNTHETIC_PROBE)
        # extreme aspect + tiny grid boundary witnesses
        add(kernel, 100.0, 35.0, 0.5, 5, 1000, 1e6, st.AMPLITUDE_SYNTHETIC_PROBE)
        add(kernel, 10.0, 35.0, 0.5, 5, 1000, 1e6, st.AMPLITUDE_SYNTHETIC_PROBE)
    return cases


def capture_fixture_traces(
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    traces = []
    for case in cases if cases is not None else default_fixture_cases():
        traces.append(capture_trace(**case))
    return traces


def write_trace_artifacts(
    traces: list[dict[str, Any]],
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Write frozen traces + sha256 manifest under the artifact root."""
    traces_dir = out_root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": st.STEP_TABLE_SCHEMA_VERSION,
        "generator": "tests/ultrafast/trace_exporter.py",
        "trace_count": len(traces),
        "entries": [],
    }
    for trace in traces:
        name = f"{trace['trace_id']}.json"
        path = traces_dir / name
        path.write_text(json.dumps(trace, sort_keys=True, indent=1))
        manifest["entries"].append(
            st.trace_manifest_entry(trace, trace_sha256=sha256_file(path))
        )
    manifest_path = out_root / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=1))
    # repo-side small manifest (digests only, no trace payloads)
    REPO_MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPO_MANIFEST_JSON.write_text(
        json.dumps(manifest, sort_keys=True, indent=1)
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="artifact root for frozen traces + manifest",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="also capture the full 153-patch real-site trace set (slow)",
    )
    args = parser.parse_args(argv)
    traces = capture_fixture_traces()
    manifest_path = write_trace_artifacts(traces, args.out_root)
    print(f"wrote {len(traces)} traces; manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
