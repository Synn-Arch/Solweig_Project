# SPDX-License-Identifier: GPL-3.0-only
"""T11 fallback ROUTER: dense vs sparse vs full-domain, decided by cost
and shape — measured, never assumed.

The incremental engine's fallback lanes and their gate conditions, in
DECISION ORDER (the table is the contract; tests pin each row):

=== ============================= =======================================
 #  condition                     route
=== ============================= =======================================
 1  geometry edit                 REGENERATE + cold full solve
                                   (warm start across a geometry edit is
                                   FORBIDDEN even when an anchor exists;
                                   the capture is per-profile)
 2  met edit + applicable anchor  WARM SPARSE REPLAY from anchor
    AND warm is measured cheaper   next_step (T10 replay lane)
 3  met edit, no anchor — OR      COLD FULL SOLVE (full_solve_capture)
    warm measured more expensive
 4  any request whose collected    batched collect (split/spill under a
    outputs exceed the memory      memory budget — never an OOM crash)
    budget
=== ============================= =======================================

Cost model: per-step milliseconds of the two engines (cold full solve vs
sparse replay) plus per-collected-timestep output bytes, all MEASURED on
this host (bench_full_solve.py writes the JSON; defaults below are the
recorded medians). ``set_measured_costs`` replaces them at runtime; a
route is never chosen on an unmeasured guess — a missing measurement
falls back to the correctness-gated default (rows 1/3), never row 2.

Purity: numpy-only. No torch, no probing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import thermal

__all__ = [
    "ROUTE_REGENERATE",
    "ROUTE_WARM_SPARSE",
    "ROUTE_COLD_FULL",
    "RouterError",
    "MeasuredCosts",
    "RouteDecision",
    "route_fallback",
    "set_measured_costs",
    "load_measured_costs",
]

ROUTE_REGENERATE = "regenerate-cold-full"
ROUTE_WARM_SPARSE = "warm-sparse-replay"
ROUTE_COLD_FULL = "cold-full-solve"

#: bench_full_solve.py JSON (medians on this host; site_500 capture).
#: Location via SOLWEIG_ULTRAFAST_COSTS (absolute path); unset = no default
#: file (callers pass path explicitly or rely on the recorded medians).
_DEFAULT_COSTS_PATH = (
    Path(os.environ["SOLWEIG_ULTRAFAST_COSTS"])
    if os.environ.get("SOLWEIG_ULTRAFAST_COSTS") else None
)


class RouterError(RuntimeError):
    """The fallback request itself is malformed (typed, never silent)."""


@dataclass(frozen=True)
class MeasuredCosts:
    """Measured per-step engine costs + per-collected-t output bytes.

    full_ms_per_step: cold full solve (met_recompute + fused kernel),
    median ms/step on this host.
    sparse_ms_per_step: warm sparse replay (T10 lane), median ms/step.
    output_bytes_per_t: resident bytes of one timestep's collected
    output planes (PLANE_RETS at rows x cols f32).
    """

    full_ms_per_step: float
    sparse_ms_per_step: float
    output_bytes_per_t: int

    def describe(self) -> dict[str, Any]:
        return {
            "full_ms_per_step": self.full_ms_per_step,
            "sparse_ms_per_step": self.sparse_ms_per_step,
            "output_bytes_per_t": self.output_bytes_per_t,
        }


#: Recorded medians (bench_full_solve.py on the dev host, site_500 capture
#: 500x500, 24 steps, host load 35-65 recorded in the JSON). Replaced at
#: runtime by set_measured_costs or by load_measured_costs when the JSON
#: exists.
_MEASURED = MeasuredCosts(
    full_ms_per_step=996.976,
    sparse_ms_per_step=695.849,
    output_bytes_per_t=21_000_000,
)


def set_measured_costs(costs: MeasuredCosts) -> None:
    global _MEASURED
    if costs.full_ms_per_step <= 0 or costs.sparse_ms_per_step <= 0:
        raise RouterError(
            "measured costs must be positive — refusing to route on an "
            "unmeasured (zero/negative) cost"
        )
    _MEASURED = costs


def load_measured_costs(path: Path | None = _DEFAULT_COSTS_PATH) -> bool:
    """Load the bench JSON if present. Returns whether it was loaded."""
    global _MEASURED
    if path is None:
        return False
    p = Path(path)
    if not p.exists():
        return False
    data = json.loads(p.read_text())
    set_measured_costs(MeasuredCosts(
        full_ms_per_step=float(data["full_ms_per_step"]),
        sparse_ms_per_step=float(data["sparse_ms_per_step"]),
        output_bytes_per_t=int(data["output_bytes_per_t"]),
    ))
    return True


def measured_costs() -> MeasuredCosts:
    return _MEASURED


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    resume_step: int
    steps_to_advance: int
    est_cost_ms: float
    collect_batches: tuple[tuple[int, ...], ...]
    decision_table: dict = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reason": self.reason,
            "resume_step": self.resume_step,
            "steps_to_advance": self.steps_to_advance,
            "est_cost_ms": self.est_cost_ms,
            "collect_batches": [list(b) for b in self.collect_batches],
            "decision_table": self.decision_table,
        }


def _batch_collect_ts(collect_ts: Sequence[int], costs: MeasuredCosts,
                      memory_budget_bytes: int) -> tuple[tuple[int, ...], ...]:
    """Split the collect list under the memory budget (spill/split, not
    crash). Each batch's collected outputs must fit the budget; a single
    timestep larger than the budget forms its own batch (recorded, never
    silently dropped)."""
    if memory_budget_bytes <= 0:
        raise RouterError("memory_budget_bytes must be positive")
    per_t = max(int(costs.output_bytes_per_t), 1)
    max_per_batch = max(memory_budget_bytes // per_t, 1)
    batches: list[tuple[int, ...]] = []
    cur: list[int] = []
    for ts in collect_ts:
        cur.append(int(ts))
        if len(cur) >= max_per_batch:
            batches.append(tuple(cur))
            cur = []
    if cur:
        batches.append(tuple(cur))
    return tuple(batches)


def route_fallback(
    *,
    geometry_changed: bool,
    r0: int,
    n_window: int,
    anchor_decision: thermal.AnchorDecision | None = None,
    collect_ts: Sequence[int] | None = None,
    memory_budget_bytes: int = 2 * 1024 ** 3,
    costs: MeasuredCosts | None = None,
) -> RouteDecision:
    """Choose the fallback lane. Correctness gates beat cost; cost only
    breaks ties the gates leave open.

    ``geometry_changed``: the edit moved geometry (building edit family).
    ``r0``: first changed met row (``n_window`` when nothing changed).
    ``anchor_decision``: thermal.select_warm_anchor's verdict over the
    CURRENT candidates (already policy-gated; the router adds the
    geometry-edit veto on top — a geometry edit invalidates even an
    anchor that classify_anchor accepted for a matching fingerprint,
    because the capture regeneration replaces the frozen planes).
    ``collect_ts``: requested output timesteps (drives the memory bound).
    """
    if n_window <= 0:
        raise RouterError(f"n_window must be positive, got {n_window}")
    if not 0 <= r0 <= n_window:
        raise RouterError(
            f"r0 {r0} outside the window [0, {n_window}]")
    c = costs if costs is not None else _MEASURED
    table: dict[str, Any] = {
        "geometry_changed": bool(geometry_changed),
        "r0": int(r0),
        "n_window": int(n_window),
        "anchor_selected": None if anchor_decision is None
        else (anchor_decision.selected is not None),
        "anchor_resume_step": None if anchor_decision is None
        else int(anchor_decision.resume_step),
        "costs": c.describe(),
    }

    if collect_ts is None:
        collect_ts = list(range(n_window))
    collect = sorted({int(x) for x in collect_ts})
    for ts in collect:
        if not 0 <= ts < n_window:
            raise RouterError(
                f"collect ts {ts} outside the window [0, {n_window})")
    batches = _batch_collect_ts(collect, c, memory_budget_bytes)
    table["collect_batches"] = [list(b) for b in batches]
    table["memory_budget_bytes"] = int(memory_budget_bytes)

    # Row 1 — geometry edit: regenerate the capture, cold full solve.
    # The warm anchor is FORBIDDEN across a geometry edit even when
    # classify_anchor accepted it (fingerprint of the regenerated scene
    # is not yet known at routing time; never risk stale planes).
    if geometry_changed:
        dec = RouteDecision(
            route=ROUTE_REGENERATE,
            reason=(
                "geometry edit: capture regeneration + cold full solve "
                "(warm start across a geometry edit is forbidden; anchor "
                f"vetoed={table['anchor_selected']})"
            ),
            resume_step=0,
            steps_to_advance=n_window,
            est_cost_ms=float(n_window * c.full_ms_per_step),
            collect_batches=batches,
            decision_table=table,
        )
        return dec

    # Rows 2/3 — met edit: warm sparse replay vs cold full solve, decided
    # by the MEASURED per-step costs over the steps each lane advances.
    resume = 0
    if anchor_decision is not None and anchor_decision.selected is not None:
        resume = int(anchor_decision.resume_step)
    warm_steps = n_window - resume
    warm_cost = float(warm_steps * c.sparse_ms_per_step)
    cold_cost = float(n_window * c.full_ms_per_step)
    table["warm_cost_ms"] = warm_cost
    table["cold_cost_ms"] = cold_cost

    if anchor_decision is not None and anchor_decision.selected is not None:
        if warm_steps <= 0:
            # anchor covers the whole window: nothing to advance; the
            # requested outputs are served from the store (collected
            # batches above still bound the output memory)
            return RouteDecision(
                route=ROUTE_WARM_SPARSE,
                reason="anchor at window end: no steps to advance",
                resume_step=resume,
                steps_to_advance=0,
                est_cost_ms=0.0,
                collect_batches=batches,
                decision_table=table,
            )
        if warm_cost < cold_cost:
            return RouteDecision(
                route=ROUTE_WARM_SPARSE,
                reason=(
                    f"measured cheaper: warm replay {warm_steps} steps @"
                    f"{c.sparse_ms_per_step:.3f}ms = {warm_cost:.1f}ms < "
                    f"cold full {n_window} steps @"
                    f"{c.full_ms_per_step:.3f}ms = {cold_cost:.1f}ms"
                ),
                resume_step=resume,
                steps_to_advance=warm_steps,
                est_cost_ms=warm_cost,
                collect_batches=batches,
                decision_table=table,
            )
        table["warm_veto"] = (
            f"measured cost vetoes warm: {warm_cost:.1f}ms >= cold "
            f"{cold_cost:.1f}ms — the lower-density lane is NOT forced"
        )

    reason = "no applicable warm anchor"
    if anchor_decision is not None:
        reason = (f"no applicable warm anchor "
                  f"({anchor_decision.invalidation}: "
                  f"{anchor_decision.reason})")
    return RouteDecision(
        route=ROUTE_COLD_FULL,
        reason=reason,
        resume_step=0,
        steps_to_advance=n_window,
        est_cost_ms=cold_cost,
        collect_batches=batches,
        decision_table=table,
    )
