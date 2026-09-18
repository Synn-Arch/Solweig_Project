# SPDX-License-Identifier: GPL-3.0-only
"""GENERAL torch-free march step-table producer (T14 Change A, seam R3).

T03 shipped ``solweig_core.step_tables.build_table_from_trace`` — a
producer that derives tables from immutable traces captured by the torch
dev exporter, and explicitly did NOT claim arbitrary inputs. This module
is the T14 general producer: it executes the march scalar arithmetic for
ANY (azimuth, altitude, scale, rows, cols, amplitude) with NO torch, and
is gated bit-for-bit against the 74 frozen T03 traces plus a live
differential against the original torch code path
(``tests/ultrafast/test_step_table_general.py``).

Bit-exactness contract (why every line below is shaped the way it is):

The original kernels run their scalar arithmetic on 0-dim float32
tensors. Probe-verified torch scalar semantics this mirrors (T14
micro-probes, dev env):

* ``f32_tensor <python float>`` casts the SCALAR to float32 first (NOT an
  f64 compare) — the svf branch bounds are python-float ``pi`` multiples
  in shadow.py, so the producer folds them as f64 products then one f32
  cast (``F32(5*pi/4) == 0x407B53D1``); the wallheight_23 kernel uses
  f32 TENSOR bounds (``5 * torch.tensor(np.pi/4) == 0x407B53D2``) — one
  ulp higher at the 225/315 diagonals, which FLIPS the branch there. The
  two flavors are never conflated (T03 author report finding).
* ``f32_tensor * <python float>`` multiplies in float32 after casting the
  scalar (numpy NEP50 weak promotion is bit-identical).
* torch 0-dim f32 ``sin``/``cos``/``tan`` lower to SLEEF on arm64 —
  ``sleef_sinf_u1``/``sleef_cosf_u1``/``sleef_tanf_u10`` return the exact
  torch bits (0/36 mismatches probed; the frozen-trace byte gate pins
  them at the zenith and cardinal edges).
* ``torch.round`` == ``numpy.rint`` (round-half-even); ``torch.sign`` ==
  ``numpy.sign``.
* dz association is ``(ds * index) * (tan(altitude)/scale)`` with the
  ``tan/scale`` quotient folded FIRST (shadow.py:238/255,
  solweig.py:1123/1148; the naive ``ds*index*tan/sscale`` is 1 ulp off at
  non-power-of-two scales — T03 review NEW-LOW).
* the while-stop is PREVIOUS-state (``amax >= dz_prev``), so the final
  OVERSHOOT step (its own dz already above the amplitude) is executed
  and recorded.
* ``svf_shadow`` starts ``index`` at 1 and substitutes azimuth ``0.0``
  with ``1e-12`` (shadow.py:196-197, 240); ``wallheight_23`` starts at
  0 (the ``(0, 0)`` self comparison) and never substitutes
  (solweig.py:1092-1124).

The module is host-numpy scalar code (no numba kernel needed: table
builds are cached by key, not hot) — only the transcendental ports are
``@njit``. It imports nothing beyond the torch-free core stack.
"""
from __future__ import annotations

import math

import numpy as np

from solweig_core import step_tables as st
from solweig_core.numba_cpu.sleef_trig import sleef_cosf_u1, sleef_sinf_u1
from solweig_core.numba_cpu.sleef_f32_extra import sleef_tanf_u10

F32 = np.float32

#: sha256 over the oracle sources that define the march geometry
#: (trace_exporter.GEOMETRY_SOURCES, tag ``T03Geometry_v1``). Editing any
#: of them invalidates every table key by design; in the DEV env the gate
#: ``test_default_hash_is_pinned_constant`` re-derives this live and
#: forces a re-pin. In the RUNTIME (no torch, no solweig_gpu) the pinned
#: constant is the only form available — tables built for new angles
#: still carry oracle-lineage identity without importing the oracle.
SOURCE_GEOMETRY_HASH = (
    "0fea60b03d1953638ced0bf59d0d3a2af6518f47ffabc74be418e2c15dfc2ee0"
)

KERNEL_SVF_SHADOW = st.KERNEL_SVF_SHADOW
KERNEL_WALLHEIGHT_23 = st.KERNEL_WALLHEIGHT_23

# ---------------------------------------------------------------------------
# Branch-bound constants, one pair per kernel flavor (T03 author report:
# the 1-ulp divergence at 5*pi/4 flips the branch at az = 225/315 deg).
# Mutables are module-level ON PURPOSE: the mutation tests patch them to
# prove the gates bite (they are read at call time, never at import time).
# ---------------------------------------------------------------------------

#: svf_shadow (shadow.py:195, 227-230): python-float pi multiples, cast
#: to f32 at the comparison boundary.
SVF_PIBYFOUR = F32(math.pi / 4.0)                 # 0x3F490FDB
SVF_THREEPIBYFOUR = F32(3.0 * (math.pi / 4.0))    # 0x4016CBE4
SVF_FIVEPIBYFOUR = F32(5.0 * (math.pi / 4.0))     # 0x407B53D1
SVF_SEVENPIBYFOUR = F32(7.0 * (math.pi / 4.0))    # 0x40AFEDDF

#: wallheight_23 (solweig.py:1092, 1110-1113): f32 tensor arithmetic on
#: torch.tensor(np.pi/4) — single f32 rounding per multiple.
WH_PIBYFOUR = F32(math.pi / 4.0)                              # 0x3F490FDB
WH_THREEPIBYFOUR = F32(F32(math.pi / 4.0) * F32(3.0))         # 0x4016CBE4
WH_FIVEPIBYFOUR = F32(F32(math.pi / 4.0) * F32(5.0))          # 0x407B53D2
WH_SEVENPIBYFOUR = F32(F32(math.pi / 4.0) * F32(7.0))         # 0x40AFEDE0

#: degrees conversion (both flavors hold the same f32 bits:
#: torch.pi/180.0 as python float == torch.tensor(np.pi/180.0)).
DEGREES = F32(math.pi / 180.0)                     # 0x3C8EFA35

#: svf azimuth-zero substitution (shadow.py:197).
AZ_SUBSTITUTE = 1e-12
#: wallheight index origin (solweig.py:1124); svf starts at 1.
WALLHEIGHT_INDEX0 = 0.0
#: dz association switch: "quotient_first" (original) vs "scale_last"
#: (the re-associated form the byte gate must catch).
DZ_ASSOCIATION = "quotient_first"
#: |round(...)| switch (the abs the branch bodies apply).
ABS_ON_ROUND = True
#: degree->radian conversion switch: "f32" (original: f32 az * f32
#: pi/180) vs "f64_once" (compute in f64, round once — a 1-ulp trap the
#: differential gate must catch on ~8% of unseen angles).
ANGLE_CONVERSION = "f32"

_MAX_STEPS = 1 << 22  # structural sanity bound; a real march stops far earlier


def _steps_general(
    kernel_variant: str,
    azimuth_deg: float,
    altitude_deg: float,
    scale: float,
    rows: int,
    cols: int,
    amplitude: float,
) -> tuple[list[tuple[float, int, int, np.float32, int]], bool]:
    """Executed march steps as ``(index, dx, dy, dz_f32, branch_id)``.

    Host-numpy mirror of the kernels' scalar arithmetic (see module
    docstring for every probe-verified semantic). Returns the step list
    and whether the azimuth-zero substitution fired.
    """
    if kernel_variant == KERNEL_SVF_SHADOW:
        index = 1.0
        az = F32(azimuth_deg)
        substituted = bool(az == F32(0.0))
        if substituted:
            az = F32(az * F32(0.0) + F32(AZ_SUBSTITUTE))
        pibyfour = SVF_PIBYFOUR
        three = SVF_THREEPIBYFOUR
        five = SVF_FIVEPIBYFOUR
        seven = SVF_SEVENPIBYFOUR
    elif kernel_variant == KERNEL_WALLHEIGHT_23:
        index = WALLHEIGHT_INDEX0
        az = F32(azimuth_deg)
        substituted = False
        pibyfour = WH_PIBYFOUR
        three = WH_THREEPIBYFOUR
        five = WH_FIVEPIBYFOUR
        seven = WH_SEVENPIBYFOUR
    else:
        raise ValueError(f"unknown kernel_variant {kernel_variant!r}")

    if ANGLE_CONVERSION == "f32":
        azimuth = F32(az * DEGREES)
        altitude = F32(F32(altitude_deg) * DEGREES)
    else:  # mutation path: f64 product rounded once
        azimuth = F32(float(az) * (math.pi / 180.0))
        altitude = F32(float(F32(altitude_deg)) * (math.pi / 180.0))

    sinaz = F32(sleef_sinf_u1(azimuth))
    cosaz = F32(sleef_cosf_u1(azimuth))
    tanaz = F32(sleef_tanf_u10(azimuth))
    tanalt = F32(sleef_tanf_u10(altitude))
    signsin = F32(np.sign(sinaz))
    signcos = F32(np.sign(cosaz))
    # Both kernels compute BOTH ds values even though the branch uses one;
    # at sin(az)==0 (wallheight never substitutes) 1/0 = +inf exactly as in
    # torch — silencing only the numpy scalar-divide notice, not the value.
    with np.errstate(divide="ignore", invalid="ignore"):
        dssin = F32(np.abs(F32(1.0) / sinaz))
        dscos = F32(np.abs(F32(1.0) / cosaz))
    if DZ_ASSOCIATION == "quotient_first":
        tbs = F32(tanalt / F32(scale))      # tan(altitude)/scale folded first
    else:  # mutation path: the naive re-associated form
        tbs = None

    sin_branch = bool(
        (pibyfour <= azimuth < three) or (five <= azimuth < seven)
    )

    amax = F32(amplitude)
    dx = F32(0.0)
    dy = F32(0.0)
    dz = F32(0.0)
    rows_f = F32(rows)
    cols_f = F32(cols)
    steps: list[tuple[float, int, int, np.float32, int]] = []
    while bool(amax >= dz) and bool(abs(dx) < rows_f) and bool(
        abs(dy) < cols_f
    ):
        idx = F32(index)
        if sin_branch:
            dy = F32(signsin * idx)
            rounded = np.rint(F32(idx / tanaz))
            if ABS_ON_ROUND:
                rounded = np.abs(rounded)
            dx = F32(F32(F32(-1.0) * signcos) * F32(rounded))
            ds = dssin
        else:
            rounded = np.rint(F32(idx * tanaz))
            if ABS_ON_ROUND:
                rounded = np.abs(rounded)
            dy = F32(signsin * F32(rounded))
            dx = F32(F32(F32(-1.0) * signcos) * idx)
            ds = dscos
        if DZ_ASSOCIATION == "quotient_first":
            dz = F32(F32(ds * idx) * tbs)
        else:
            dz = F32(F32(F32(ds * idx) * tanalt) / F32(scale))
        steps.append((float(index), int(dx), int(dy), dz, 1 if sin_branch else 0))
        index += 1.0
        if len(steps) > _MAX_STEPS:
            raise RuntimeError(
                "march exceeded the structural step bound — inputs "
                f"({kernel_variant}, az={azimuth_deg!r}, alt="
                f"{altitude_deg!r}, scale={scale!r}) make the stop "
                "unreachable; refusing to loop forever"
            )
    return steps, substituted


def _stop_reason_of(
    steps, rows: int, cols: int
) -> str:
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


def build_table_general(
    kernel_variant: str,
    azimuth_deg: float,
    altitude_deg: float,
    scale: float,
    rows: int,
    cols: int,
    amplitude: float,
    *,
    amplitude_policy_id: str = st.AMPLITUDE_SYNTHETIC_PROBE,
    semantics_profile: str = "canonical_cpu_v1",
    boundary_policy_id: str = st.BOUNDARY_SHIFT_WINDOW_V1,
    source_geometry_hash: str | None = None,
) -> st.StepTable:
    """Build the executed step table of one march WITHOUT torch (R3).

    Parameters are the raw scalar inputs the kernels receive: angles in
    degrees (rounded to float32 exactly once, mirroring the 0-dim f32
    tensors the kernels get), ``scale`` in pixels per metre as the exact
    float64 the caller feeds the kernel, integer logical bounds, and the
    float amplitude of the while-stop (rounded to float32 once — the
    kernels' ``amaxvalue`` tensor). The key carries the raw f32 bits of
    every float input, per the T03 table contract.
    """
    if altitude_deg <= 0.0:
        raise ValueError(
            "altitude <= 0 makes dz identically 0 (loop runs to the domain "
            "boundary or forever); the march domain requires altitude > 0"
        )
    rows = int(rows)
    cols = int(cols)
    if rows < 1 or cols < 1:
        raise ValueError("rows/cols must be positive")
    if amplitude_policy_id not in st.AMPLITUDE_POLICIES:
        raise ValueError(
            f"unknown amplitude_policy_id {amplitude_policy_id!r}"
        )
    steps, substituted = _steps_general(
        kernel_variant, azimuth_deg, altitude_deg, scale, rows, cols, amplitude
    )
    count = len(steps)
    dz_bits = np.asarray(
        [F32(s[3]).view(np.uint32) for s in steps], dtype=np.uint32
    )
    table = st.StepTable(
        key=st.StepTableKey(
            semantics_profile=semantics_profile,
            kernel_variant=kernel_variant,
            source_geometry_hash=(
                SOURCE_GEOMETRY_HASH
                if source_geometry_hash is None
                else str(source_geometry_hash)
            ),
            angle_input_bits=(
                st.f32_bits_hex(azimuth_deg),
                st.f32_bits_hex(altitude_deg),
            ),
            scale_bits=st.f32_bits_hex(scale),
            logical_rows=rows,
            logical_cols=cols,
            amplitude_policy_id=amplitude_policy_id,
            executed_amplitude_bits=st.f32_bits_hex(amplitude),
            boundary_policy_id=boundary_policy_id,
        ),
        count=count,
        dx=np.asarray([s[1] for s in steps], dtype=np.int32),
        dy=np.asarray([s[2] for s in steps], dtype=np.int32),
        dz_bits=dz_bits,
        stop_reason=_stop_reason_of(steps, rows, cols),
        branch_id=np.asarray([s[4] for s in steps], dtype=np.uint8),
        previous_dz_bits=np.asarray(
            [np.uint32(0)] + list(dz_bits[:-1]) if count else [],
            dtype=np.uint32,
        ),
        azimuth_zero_substituted=substituted,
        trace_id=(
            f"{kernel_variant}_az{st.f32_bits_hex(azimuth_deg)}"
            f"_alt{st.f32_bits_hex(altitude_deg)}_s{st.f32_bits_hex(scale)}"
            f"_{rows}x{cols}_A{st.f32_bits_hex(amplitude)}"
        ),
    )
    return table
