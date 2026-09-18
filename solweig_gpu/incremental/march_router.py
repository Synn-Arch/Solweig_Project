# SPDX-License-Identifier: GPL-3.0-only
"""T19b production SVF-march lane router (GOAL_PROMPT_T18_CPU v3, T19b).

The selected-exact production lane marched its per-patch SVF shadows with
the torch kernel (:func:`solweig_gpu.shadow.shadow`) even after T04 landed
the bit-pinned Numba twin
(:func:`solweig_core.numba_cpu.march.march_svf_shadow`) — the twin had no
production caller, so its measured contribution was structurally zero (T19
HONEST attribution note). This module is the routing seam: the SAME call
shape as ``shadow()``, answered by the registered Numba lane when the scene
is inside its domain, and by the torch kernel otherwise.

Registration discipline (mirrors :mod:`solweig_core.dispatch` / T11's
``numba-full-solve``): lanes are an EXPLICIT vocabulary resolved from a
declared preference — never a capability probe, never a silent re-target.
An unknown lane name is refused LOUDLY (:class:`MarchLaneError`), while a
scene outside the Numba march's domain (bush > 0, non-float32 input,
degenerate shape — the wrapper's typed ``ValueError`` surface) is a typed
REFUSAL that falls back to the torch kernel automatically and is recorded
in the lane stats and telemetry: never a silent approximation.

Bit-exactness contract: on the routed lane the executed step set comes from
the general torch-free step-table producer
(:func:`solweig_core.numba_cpu.step_table_gen.build_table_general`, R3 —
gated bit-for-bit against the 74 frozen T03 traces plus a live torch
differential), and the march body is the T04-pinned kernel — so the
returned planes are raw-bit equal to ``shadow()``'s on every scene the
wrapper accepts. Any divergence would be a gate failure, not a routing
decision. Inputs cross as read-only contiguous copies only when strided
(bit-preserving data movement); outputs are fresh float32 planes wrapped
zero-copy back into torch tensors, matching ``shadow()``'s CPU outputs.

Purity note: the torch kernel is imported lazily and only as the fallback
callable — the module itself needs just numpy/torch for the boundary
conversions, and ``solweig_core`` loads lazily so importing this router
never pays the numba stack cost.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch

__all__ = [
    "LANE_TORCH_SHADOW",
    "LANE_NUMBA_MARCH",
    "MARCH_LANES",
    "SVF_MARCH_LANE_ENV_FLAG",
    "MarchLaneError",
    "resolve_march_lane",
    "shadow_march",
    "lane_stats",
    "reset_lane_stats",
]

#: The legacy torch kernel lane (``solweig_gpu.shadow.shadow``).
LANE_TORCH_SHADOW = "legacy-torch-shadow"
#: The exact Numba march lane (T04 kernel + R3 general step tables).
LANE_NUMBA_MARCH = "numba-svf-march"
#: The registered lane vocabulary — resolution never leaves this tuple.
MARCH_LANES = (LANE_TORCH_SHADOW, LANE_NUMBA_MARCH)

#: Declared-lane environment flag (T15/T24 ``SOLWEIG_RT_*`` convention).
#: Unset/empty resolves to the Numba lane (the routing IS the deliverable);
#: ``torch`` forces the legacy kernel byte-identically (A/B base arm,
#: kill-switch). Read at call time, like T15/T24 flags.
SVF_MARCH_LANE_ENV_FLAG = "SOLWEIG_RT_SVF_MARCH_LANE"

#: Short lane aliases accepted from the flag (the registered names above
#: are always accepted too).
_LANE_ALIASES = {
    "": LANE_NUMBA_MARCH,
    "numba": LANE_NUMBA_MARCH,
    LANE_NUMBA_MARCH: LANE_NUMBA_MARCH,
    "torch": LANE_TORCH_SHADOW,
    LANE_TORCH_SHADOW: LANE_TORCH_SHADOW,
}

#: Amplitude-policy ids this router records on built tables (mapped to the
#: R6 registry ids in :mod:`solweig_core.step_tables` lazily). The policy is
#: provenance metadata on the table key — it never changes the steps.
_AMPLITUDE_POLICIES = ("effective_windowed", "r6_band_escalated", "oracle_absolute")


class MarchLaneError(ValueError):
    """The declared march lane is outside the registered vocabulary.

    Loud by design (dispatch discipline): a typo'd flag must never be
    guessed into a lane. Scene-domain refusals are NOT this error — they
    fall back to the torch lane and are recorded in the lane stats.
    """


# ---------------------------------------------------------------------------
# Lane stats (process-level telemetry; per-solve deltas are taken by the
# call sites, so concurrent solves never misattribute each other's counts)
# ---------------------------------------------------------------------------

_STATS_LOCK = threading.Lock()
_STATS: dict[str, Any] = {
    "numba_routed": 0,
    "numba_refused": 0,
    "torch_forced": 0,
    "last_refusal": None,
}


def lane_stats() -> dict[str, Any]:
    """Snapshot of the process-level lane counters (copy; never mutated)."""
    with _STATS_LOCK:
        return dict(_STATS)


def reset_lane_stats() -> None:
    """Zero the counters (tests and bench harnesses only)."""
    with _STATS_LOCK:
        _STATS.update(
            {
                "numba_routed": 0,
                "numba_refused": 0,
                "torch_forced": 0,
                "last_refusal": None,
            }
        )


def _record(key: str, value: Any = None) -> None:
    with _STATS_LOCK:
        if key == "refusal":
            _STATS["numba_refused"] += 1
            _STATS["last_refusal"] = value
        else:
            _STATS[key] += 1


# ---------------------------------------------------------------------------
# Step-table cache: one build per unique (angle bits, scale bits, shape,
# amplitude bits, policy) — the producer is deterministic, so the memoized
# table is the table the first caller got.
# ---------------------------------------------------------------------------

_TABLE_CACHE: dict[tuple, Any] = {}


def _policy_id(policy: str) -> str:
    from solweig_core import step_tables as st

    mapping = {
        "effective_windowed": st.AMPLITUDE_EFFECTIVE_WINDOWED,
        "r6_band_escalated": st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED,
        "oracle_absolute": st.AMPLITUDE_ORACLE_ABSOLUTE,
    }
    try:
        return mapping[policy]
    except KeyError:
        raise MarchLaneError(
            f"amplitude policy {policy!r} is outside the router vocabulary "
            f"{_AMPLITUDE_POLICIES}; refusing to guess a table provenance"
        ) from None


def _table_for(
    azimuth_deg: float,
    altitude_deg: float,
    scale: float,
    rows: int,
    cols: int,
    amplitude: float,
    policy: str,
):
    """Memoized general step table for one march geometry.

    The torch kernel's executed step set is a pure function of exactly
    these inputs (its while-stop compares the f32 amplitude against the
    f32 dz ladder and the f32 shifts against the marched extent's shape),
    so the cache key carries every input's f32 BITS — two f64 spellings
    that round to the same f32 march input share one table, exactly as
    the kernel itself would.
    """
    from solweig_core import step_tables as st
    from solweig_core.numba_cpu.step_table_gen import build_table_general

    key = (
        st.KERNEL_SVF_SHADOW,
        st.f32_bits_hex(azimuth_deg),
        st.f32_bits_hex(altitude_deg),
        st.f32_bits_hex(scale),
        int(rows),
        int(cols),
        st.f32_bits_hex(amplitude),
        policy,
    )
    table = _TABLE_CACHE.get(key)
    if table is None:
        table = build_table_general(
            st.KERNEL_SVF_SHADOW,
            azimuth_deg,
            altitude_deg,
            scale,
            int(rows),
            int(cols),
            amplitude,
            amplitude_policy_id=_policy_id(policy),
        )
        _TABLE_CACHE[key] = table
    return table


def _as_numpy(array: Any, name: str) -> np.ndarray:
    """Boundary view of one scene plane (no copy when contiguous)."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    raise ValueError(
        f"{name} must be a torch tensor or numpy array, got "
        f"{type(array).__name__}"
    )


def _scalar_float(value: Any) -> float:
    """Exact float of one march scalar (0-dim f32 tensors round-trip)."""
    if isinstance(value, torch.Tensor):
        return float(value.reshape(()).item())
    return float(value)


# ---------------------------------------------------------------------------
# The routing seam
# ---------------------------------------------------------------------------


def resolve_march_lane(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the declared SVF-march lane (never a probe, never a guess).

    Unset/empty -> the Numba lane (the routing is this card's deliverable).
    ``torch`` -> the legacy kernel, byte-identical to the pre-routing
    production path (the A/B base arm and the kill-switch). Anything else
    is refused loudly — an unknown lane must never be guessed.
    """
    source = os.environ if environ is None else environ
    value = str(source.get(SVF_MARCH_LANE_ENV_FLAG, "")).strip().lower()
    try:
        return _LANE_ALIASES[value]
    except KeyError:
        raise MarchLaneError(
            f"{SVF_MARCH_LANE_ENV_FLAG}={value!r} is outside the registered "
            f"march lanes {sorted(set(_LANE_ALIASES))}; refusing to guess a "
            "march backend (set 'numba' or 'torch', or unset the flag)"
        ) from None


def shadow_march(
    amaxvalue,
    a,
    vegdem,
    vegdem2,
    bush,
    azimuth,
    altitude,
    scale,
    *,
    amplitude_policy: str = "effective_windowed",
    torch_shadow=None,
    telemetry: MutableMapping[str, Any] | None = None,
):
    """``shadow()``'s contract, routed to the registered march lane.

    Same argument order as :func:`solweig_gpu.shadow.shadow` — the
    production call sites hand their torch tensors straight through. On
    the Numba lane the inputs cross as (contiguous-only-when-strided)
    boundary views, the executed step set comes from the R3 general
    producer keyed on the marched extent's OWN shape, and the T04 kernel
    returns planes raw-bit equal to the torch kernel's; on refusal
    (outside the wrapper's domain) or on the declared torch lane the
    original kernel runs byte-identically.

    ``amplitude_policy`` records which R6 policy selected the executed
    amplitude (table-key provenance only — never changes the steps).
    ``torch_shadow`` lets the caller hand in its module-level ``shadow``
    binding so test spies on that name keep witnessing the fallback lane.
    ``telemetry`` (optional) receives ``{"lane", "numba_routed",
    "numba_refused", "refusal"}`` for the caller's diagnostics.
    """
    if torch_shadow is None:
        from solweig_gpu.shadow import shadow as torch_shadow

    lane = resolve_march_lane()
    if lane == LANE_TORCH_SHADOW:
        _record("torch_forced")
        if telemetry is not None:
            telemetry["lane"] = LANE_TORCH_SHADOW
        return torch_shadow(
            amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale
        )

    try:
        a_np = _as_numpy(a, "a")
        vegdem_np = _as_numpy(vegdem, "vegdem")
        vegdem2_np = _as_numpy(vegdem2, "vegdem2")
        bush_np = _as_numpy(bush, "bush")
        table = _table_for(
            _scalar_float(azimuth),
            _scalar_float(altitude),
            float(scale),
            int(a_np.shape[0]),
            int(a_np.shape[1]),
            _scalar_float(amaxvalue),
            amplitude_policy,
        )
        from solweig_core.numba_cpu.march import march_svf_shadow

        sh_np, vegsh_np, vbsh_np = march_svf_shadow(
            table, a_np, vegdem_np, vegdem2_np, bush_np
        )
        _record("numba_routed")
        if telemetry is not None:
            telemetry["lane"] = LANE_NUMBA_MARCH
            telemetry["numba_routed"] = (
                telemetry.get("numba_routed", 0) + 1
            )
        return (
            torch.from_numpy(sh_np),
            torch.from_numpy(vegsh_np),
            torch.from_numpy(vbsh_np),
        )
    except MarchLaneError:
        raise
    except (ValueError, RuntimeError, TypeError) as refusal:
        # Typed scene-domain refusal (bush > 0, dtype/shape, degenerate
        # extent, unreachable stop): record and fall back to the torch
        # kernel with the ORIGINAL arguments — never approximate.
        _record("refusal", f"{type(refusal).__name__}: {refusal}")
        if telemetry is not None:
            telemetry["lane"] = LANE_TORCH_SHADOW
            telemetry["numba_refused"] = (
                telemetry.get("numba_refused", 0) + 1
            )
            telemetry["refusal"] = str(refusal)
        return torch_shadow(
            amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale
        )

