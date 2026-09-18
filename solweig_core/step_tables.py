# SPDX-License-Identifier: GPL-3.0-only
"""Exact march step tables for the shadow kernels (DESIGN.ko.md 7.2, T03).

A :class:`StepTable` is the frozen, replayable form of one march's EXECUTED
step sequence: the per-step ``(dx, dy)`` integer shifts, the raw float32 bit
pattern of every step's ``dz``, and the executed step count — including the
final OVERSHOOT step the original while-loop performs when its PREVIOUS-state
stop (``amaxvalue >= dz_prev``) was still satisfied at loop entry.

Two kernel variants exist and are NEVER conflated (DESIGN 7.1/7.2):

* ``svf_shadow`` — :func:`solweig_gpu.shadow.shadow` (SVF patch march).
  ``index`` starts at 1; azimuth ``0.0`` is substituted with ``1e-12``; the
  first step carries the ``index == 1`` accumulator-reset exception.
* ``wallheight_23`` — :func:`solweig_gpu.solweig.shadowingfunction_wallheight_23`
  (time-loop march). ``index`` starts at 0 (step 0 is the ``(0, 0)`` self
  comparison); NO azimuth-zero substitution; ``dz`` association is the
  explicitly parenthesised ``(ds * index) * (tan(altitude) / scale)`` with
  the ``tan/scale`` quotient folded FIRST.

T03 scope: this module is deliberately TORCH-FREE (solweig_core contract,
T02). It owns the table types, the amplitude-policy registry, canonical
digests, and the FIRST producer — :func:`build_table_from_trace` — which
derives tables from immutable traces captured from the original code path
by ``tests/ultrafast/trace_exporter.py``. A Torch-free general producer for
arbitrary inputs is T14, explicitly NOT claimed here.

All float scalars are stored as their exact float32 bit patterns (uint32,
hex-encoded in JSON). No reinterpretation of ``dz`` as an integer value ever
happens; comparisons are on raw bits only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: Table/traces schema version (bump on any layout or semantics change).
STEP_TABLE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Kernel variants (DESIGN 7.2: SVF shadow and wall-height march are distinct)
# ---------------------------------------------------------------------------

KERNEL_SVF_SHADOW = "svf_shadow"
KERNEL_WALLHEIGHT_23 = "wallheight_23"

KERNEL_VARIANTS = (KERNEL_SVF_SHADOW, KERNEL_WALLHEIGHT_23)

# ---------------------------------------------------------------------------
# Amplitude policy registry (R6 preservation, DESIGN 7.2)
#
# The EXECUTED amplitude of a march is not always the same function of the
# scene. T03 records, per table, which policy selected the amplitude and the
# exact float32 bits that were used. Never collapse these to "A_eff".
# ---------------------------------------------------------------------------

AMPLITUDE_ORACLE_ABSOLUTE = "oracle_absolute"
AMPLITUDE_EFFECTIVE_WINDOWED = "effective_windowed"
AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED = "effective_r6_banded_escalated"
AMPLITUDE_EFFECTIVE_R6_TIMELOOP_ESCALATED = "effective_r6_timeloop_escalated"
AMPLITUDE_SYNTHETIC_PROBE = "synthetic_probe"

AMPLITUDE_POLICIES: dict[str, str] = {
    AMPLITUDE_ORACLE_ABSOLUTE: (
        "amaxvalue = max(a.max(), vegdem.max()) as passed by the full-domain "
        "oracle (svf_calculator / full time loop) — the absolute scene stop"
    ),
    AMPLITUDE_EFFECTIVE_WINDOWED: (
        "effective_march_amplitude(a, vegdsm, vegdsm2, scene_amaxvalue) — "
        "relative-bound early-exit amplitude, clamped to the scene value; "
        "used for non-banded patches/loops in the incremental exact path"
    ),
    AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED: (
        "R6 per-patch escalation: a sky patch whose first-step dz lands in "
        "(A_eff, A_abs] marches at the ABSOLUTE scene.amaxvalue (identical "
        "executed step set to the oracle by construction)"
    ),
    AMPLITUDE_EFFECTIVE_R6_TIMELOOP_ESCALATED: (
        "R6 time-loop escalation: when any timestep's sun first-step "
        "distance lands in (window amplitude, absolute] the whole wall-"
        "height loop marches at the ABSOLUTE scene.amaxvalue"
    ),
    AMPLITUDE_SYNTHETIC_PROBE: (
        "synthetic fixture amplitude chosen directly by the trace exporter "
        "(probe / regime witness construction)"
    ),
}

#: Boundary policy: shadow()'s int() shift-window arithmetic
#: (xc1/xp1 ... xp2/yp2 of shadow.py:262-269 / solweig.py:1128-1136), where
#: out-of-bounds reads are dropped (temporaries stay at their zero init).
BOUNDARY_SHIFT_WINDOW_V1 = "shadow_shift_window_v1"

#: Stop reasons recorded on a table.
STOP_AMPLITUDE = "amplitude"
STOP_ROW_BOUNDARY = "row_boundary"
STOP_COL_BOUNDARY = "col_boundary"
STOP_ROW_COL_BOUNDARY = "row_col_boundary"

STOP_REASONS = (
    STOP_AMPLITUDE,
    STOP_ROW_BOUNDARY,
    STOP_COL_BOUNDARY,
    STOP_ROW_COL_BOUNDARY,
)


# ---------------------------------------------------------------------------
# Bit helpers (numpy-only; exact float32 <-> uint32 roundtrip)
# ---------------------------------------------------------------------------


def f32_bits_hex(value: float) -> str:
    """Exact float32 bit pattern of ``value`` as 0x-prefixed hex.

    The value is first rounded to float32 exactly as a torch scalar cast
    would (round-to-nearest, ties-to-even via numpy's float32 conversion).
    """
    return f"0x{np.uint32(np.float32(value).view(np.uint32)):08x}"


def bits_hex_to_u32(hexbits: str) -> int:
    return int(hexbits, 16)


def u32_to_bits_hex(bits: int) -> str:
    if not 0 <= int(bits) <= 0xFFFFFFFF:
        raise ValueError(f"bits {bits!r} outside uint32 range")
    return f"0x{int(bits):08x}"


def u32_bits_to_f32(bits: int) -> np.float32:
    return np.uint32(bits).view(np.float32)


# ---------------------------------------------------------------------------
# Key + table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepTableKey:
    """Identity of one march step table (DESIGN 7.2 key fields verbatim)."""

    semantics_profile: str
    kernel_variant: str
    source_geometry_hash: str
    angle_input_bits: tuple[str, str]      # (azimuth_bits, altitude_bits), degrees f32
    scale_bits: str                        # pixels-per-metre, f32 bits
    logical_rows: int
    logical_cols: int
    amplitude_policy_id: str
    executed_amplitude_bits: str           # f32 bits of the EXECUTED stop value
    boundary_policy_id: str

    def __post_init__(self) -> None:
        if self.kernel_variant not in KERNEL_VARIANTS:
            raise ValueError(f"unknown kernel_variant {self.kernel_variant!r}")
        if self.amplitude_policy_id not in AMPLITUDE_POLICIES:
            raise ValueError(
                f"unknown amplitude_policy_id {self.amplitude_policy_id!r}"
            )
        if len(self.angle_input_bits) != 2:
            raise ValueError("angle_input_bits must be (azimuth, altitude)")
        for name in ("angle_input_bits",):
            for hexbits in getattr(self, name):
                bits_hex_to_u32(hexbits)
        bits_hex_to_u32(self.scale_bits)
        bits_hex_to_u32(self.executed_amplitude_bits)
        if self.logical_rows < 1 or self.logical_cols < 1:
            raise ValueError("logical_rows/logical_cols must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics_profile": self.semantics_profile,
            "kernel_variant": self.kernel_variant,
            "source_geometry_hash": self.source_geometry_hash,
            "angle_input_bits": list(self.angle_input_bits),
            "scale_bits": self.scale_bits,
            "logical_rows": int(self.logical_rows),
            "logical_cols": int(self.logical_cols),
            "amplitude_policy_id": self.amplitude_policy_id,
            "executed_amplitude_bits": self.executed_amplitude_bits,
            "boundary_policy_id": self.boundary_policy_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepTableKey":
        return cls(
            semantics_profile=str(data["semantics_profile"]),
            kernel_variant=str(data["kernel_variant"]),
            source_geometry_hash=str(data["source_geometry_hash"]),
            angle_input_bits=(
                str(data["angle_input_bits"][0]),
                str(data["angle_input_bits"][1]),
            ),
            scale_bits=str(data["scale_bits"]),
            logical_rows=int(data["logical_rows"]),
            logical_cols=int(data["logical_cols"]),
            amplitude_policy_id=str(data["amplitude_policy_id"]),
            executed_amplitude_bits=str(data["executed_amplitude_bits"]),
            boundary_policy_id=str(data["boundary_policy_id"]),
        )


@dataclass
class StepTable:
    """The EXECUTED step sequence of one march (DESIGN 7.2 table fields).

    ``dx``/``dy`` are the integer row/col shifts of every step the original
    while-loop BODY ran — including the final overshoot step whose ``dz``
    already exceeds the amplitude (the loop tests the PREVIOUS ``dz``). The
    order is execution order.

    Debug columns (DESIGN "optional_debug"):

    * ``branch_id`` — 1 = sin-dominant branch (``[pi/4, 3pi/4) U [5pi/4,
      7pi/4)``), 0 = cos-dominant (else) branch, per step.
    * ``previous_dz_bits`` — the ``dz`` the while-condition saw at entry to
      each step (the previous-state stop evidence; for kernel
      ``wallheight_23`` this is also the ``dzprev`` consumed by the
      ``lastfabovea``/``lastgabovea`` planes).
    """

    key: StepTableKey
    count: int
    dx: np.ndarray            # int32[count]
    dy: np.ndarray            # int32[count]
    dz_bits: np.ndarray       # uint32[count]
    stop_reason: str
    branch_id: np.ndarray     # uint8[count]  (optional debug)
    previous_dz_bits: np.ndarray  # uint32[count] (optional debug)
    azimuth_zero_substituted: bool = False
    trace_id: str = ""

    def __post_init__(self) -> None:
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(f"unknown stop_reason {self.stop_reason!r}")
        arrays = {
            "dx": (self.dx, np.dtype("int32")),
            "dy": (self.dy, np.dtype("int32")),
            "dz_bits": (self.dz_bits, np.dtype("uint32")),
            "branch_id": (self.branch_id, np.dtype("uint8")),
            "previous_dz_bits": (self.previous_dz_bits, np.dtype("uint32")),
        }
        for name, (array, dtype) in arrays.items():
            array = np.asarray(array, dtype=dtype)
            if array.ndim != 1 or array.shape[0] != self.count:
                raise ValueError(
                    f"{name} must be 1-D of length count={self.count}, "
                    f"got shape {array.shape}"
                )
            setattr(self, name, array)
        if self.key.kernel_variant == KERNEL_WALLHEIGHT_23 and self.count > 0:
            # The wall-height march always begins with the index-0 self step.
            if int(self.dx[0]) != 0 or int(self.dy[0]) != 0:
                raise ValueError(
                    "wallheight_23 tables must start with the (0, 0) self step"
                )

    # -- digest -----------------------------------------------------------

    def content_digest(self) -> str:
        """sha256 over the canonical payload: count, dx, dy, dz_bits."""
        digest = hashlib.sha256()
        digest.update(b"T03StepTable_v1|")
        digest.update(f"count={int(self.count)}|".encode())
        digest.update(
            np.ascontiguousarray(self.dx, dtype="<i4").tobytes()
        )
        digest.update(b"|")
        digest.update(
            np.ascontiguousarray(self.dy, dtype="<i4").tobytes()
        )
        digest.update(b"|")
        digest.update(
            np.ascontiguousarray(self.dz_bits, dtype="<u4").tobytes()
        )
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STEP_TABLE_SCHEMA_VERSION,
            "key": self.key.to_dict(),
            "count": int(self.count),
            "dx": [int(v) for v in self.dx],
            "dy": [int(v) for v in self.dy],
            "dz_bits": [u32_to_bits_hex(int(b)) for b in self.dz_bits],
            "branch_id": [int(b) for b in self.branch_id],
            "previous_dz_bits": [
                u32_to_bits_hex(int(b)) for b in self.previous_dz_bits
            ],
            "stop_reason": self.stop_reason,
            "azimuth_zero_substituted": bool(self.azimuth_zero_substituted),
            "trace_id": self.trace_id,
            "content_sha256": self.content_digest(),
        }


# ---------------------------------------------------------------------------
# Byte comparison (the T03 exit gate: table == original trace)
# ---------------------------------------------------------------------------


def compare_step_tables(
    reference: StepTable, candidate: StepTable
) -> dict[str, Any]:
    """Raw comparison report between two tables (never NaN-aware, never
    value-tolerant: ``dz`` compares as uint32 bit patterns)."""
    report: dict[str, Any] = {
        "keys_equal": reference.key == candidate.key,
        "reference_key": reference.key.to_dict(),
        "candidate_key": candidate.key.to_dict(),
        "reference_count": int(reference.count),
        "candidate_count": int(candidate.count),
        "counts_equal": int(reference.count) == int(candidate.count),
        "mismatch_count": 0,
        "first_mismatches": [],
    }
    if not report["counts_equal"]:
        report["mismatch_count"] = -1  # structural: count differs
        report["all_equal"] = False
        report["first_mismatches"].append(
            {
                "kind": "count",
                "reference": int(reference.count),
                "candidate": int(candidate.count),
            }
        )
        return report
    mismatches = 0
    for column, ref, cand in (
        ("dx", reference.dx, candidate.dx),
        ("dy", reference.dy, candidate.dy),
        ("dz_bits", reference.dz_bits, candidate.dz_bits),
    ):
        diff = np.nonzero(
            np.asarray(ref, dtype=np.int64) != np.asarray(cand, dtype=np.int64)
        )[0]
        mismatches += int(diff.size)
        for i in diff[:8]:
            report["first_mismatches"].append(
                {
                    "kind": column,
                    "step": int(i),
                    "reference": int(ref[i]),
                    "candidate": int(cand[i]),
                }
            )
    report["mismatch_count"] = mismatches
    report["all_equal"] = (
        report["keys_equal"] and mismatches == 0 and report["counts_equal"]
    )
    return report


# ---------------------------------------------------------------------------
# First producer (T03): build a table from an immutable original trace
# ---------------------------------------------------------------------------


def _hex_list_to_u32_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [bits_hex_to_u32(str(v)) for v in values], dtype=np.uint32
    )


def build_table_from_trace(trace: Mapping[str, Any]) -> StepTable:
    """Derive a :class:`StepTable` from a captured original-code trace.

    This is the trace-derived producer the T03 card explicitly allows. It
    performs NO march arithmetic of its own: every value comes from the
    trace, which ``tests/ultrafast/trace_exporter.py`` captured by importing
    and executing the ORIGINAL torch code path (and probe-verified against
    the original stop behaviour). Fences:

    * schema version must match exactly;
    * ``count`` must equal every column length;
    * columns are materialised at their exact dtypes (int32/int32/uint32);
    * ``dz`` is never converted to float or int values here — bits only.
    """
    version = int(trace["schema_version"])
    if version != STEP_TABLE_SCHEMA_VERSION:
        raise ValueError(
            f"trace schema_version {version} != {STEP_TABLE_SCHEMA_VERSION}"
        )
    count = int(trace["count"])
    dx = np.asarray(trace["dx"], dtype=np.int32)
    dy = np.asarray(trace["dy"], dtype=np.int32)
    dz_bits = _hex_list_to_u32_array(trace["dz_bits"])
    branch_id = np.asarray(trace.get("branch_id", [0] * count), dtype=np.uint8)
    previous_dz_bits = _hex_list_to_u32_array(
        trace.get("previous_dz_bits", ["0x00000000"] * count)
    )
    for name, column in (
        ("dx", dx),
        ("dy", dy),
        ("dz_bits", dz_bits),
        ("branch_id", branch_id),
        ("previous_dz_bits", previous_dz_bits),
    ):
        if column.ndim != 1 or column.shape[0] != count:
            raise ValueError(
                f"trace column {name!r} length {column.shape[0]} != count "
                f"{count}"
            )
    table = StepTable(
        key=StepTableKey.from_dict(trace["key"]),
        count=count,
        dx=dx,
        dy=dy,
        dz_bits=dz_bits,
        stop_reason=str(trace["stop_reason"]),
        branch_id=branch_id,
        previous_dz_bits=previous_dz_bits,
        azimuth_zero_substituted=bool(
            trace.get("azimuth_zero_substituted", False)
        ),
        trace_id=str(trace.get("trace_id", "")),
    )
    expected = trace.get("content_sha256")
    if expected is not None and expected != table.content_digest():
        raise ValueError(
            "trace content_sha256 does not match the rebuilt table — the "
            "trace payload was corrupted or truncated"
        )
    return table


# ---------------------------------------------------------------------------
# Manifest helpers (repo-side digests of the frozen artifact traces)
# ---------------------------------------------------------------------------


def trace_manifest_entry(trace: Mapping[str, Any], *, trace_sha256: str) -> dict[str, Any]:
    """Small JSON-safe manifest record for one frozen trace file."""
    table = build_table_from_trace(trace)
    return {
        "trace_id": str(trace.get("trace_id", "")),
        "trace_sha256": trace_sha256,
        "content_sha256": table.content_digest(),
        "count": int(table.count),
        "stop_reason": table.stop_reason,
        "key": table.key.to_dict(),
    }


def load_manifest(path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
