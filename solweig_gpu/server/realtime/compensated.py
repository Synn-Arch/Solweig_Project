# SPDX-License-Identifier: GPL-3.0-only
"""R7 compensated fast kernel: anchored vegetation shadow-delta overlay (v1).

THE GAP FAMILY. Scout-verified across the serving stack: view-only epochs
are structurally ``fast_exact`` (``ViewCacheKernel``); meteorology epochs
are served by the exact lane's own bounded fast paths (r3a met fast path +
G2.1 warm path live in ``utci_process.py`` / ``incremental/store.py`` /
``executor.py`` / ``jobs.py`` — exact-lane serving, not fast-lane kernels).
Vegetation geometry has NO bounded exact path (windowed exact solves are
seconds, full-tile ~130 s measured on site_500), which is exactly the gap
this kernel fills — as the LAST resort (priority discipline).

THE MODEL (``geometric_overlay`` v1, no fitted thermal numbers). Per edited
tree, per sun-up timestep: the canopy occupies the volume
``[trunk_ratio * height, height]`` above the crown disc. The cast-shadow
region is the swept "capsule" between the crown-disc projection at the
canopy top and at the canopy bottom (both displaced with the REAL
``geometry.shadow_vector_m`` machinery — same 5-degree sun floor and 300 m
cap the incremental invalidation layer uses), unioned with the crown disc
itself (ground under the canopy is vegetation shadow). The predicted change
set per timestep is the XOR of the capsule unions of the BASE tree set
(canonical vegetation at the anchor revision) and the CURRENT tree set
(epoch-final canonical state). Flat-ground projection, no wall interaction,
shadow plane only — every limitation is disclosed in the payload.

WHY NOT A FITTED MODEL. The mission rule: fitted thermal coefficients must
come from offline calibration on REAL exact solves with error measured on
HELD-OUT solves, and a purely geometric overlay is PREFERRED when a fitted
model cannot clear honest holdout reporting. v1 ships the geometric overlay
— its holdout error (measured by
``docs/incremental_design_tool/realtime_collaboration/design/proofs/
r7_veg_compensation_holdout.py`` on real exact solves of the site_500
cache copy) is the qualifier's ``error_evidence``.

ANCHORING. The kernel NEVER emits standalone planes. The payload is a
delta descriptor against the latest EXACT result carrying shadow planes
(``Store.latest_result``), carrying ``exact_base_revision`` plus
``base_planes`` (which result, which variables). Without an exact base the
kernel refuses (``visual_pending`` / ``no_exact_base``) — an unanchored
"compensation" would be a silent precision reduction.

DEADLINE. ``predicted_cost_ms`` is an affine model over the edited-tree
count with constants measured by the R7 latency benchmark
(``/tmp/r7_proof/latency.json``); the lane's deadline gate refuses the
kernel BEFORE running it when the prediction cannot fit the remaining
budget (the degradation ladder stays intact).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    SunPosition,
    shadow_vector_m,
)
from solweig_gpu.server.realtime.scheduler import FastKernelResult, FastPlan

__all__ = [
    "MODEL_FORM",
    "MODEL_VERSION",
    "RECONCILIATION_NOTE",
    "DEFAULT_QUALIFIER_EVIDENCE",
    "PREDICTED_OVERHEAD_MS",
    "PREDICTED_PER_TREE_MS",
    "TreeShape",
    "VegetationCompensationKernel",
    "predict_shadow_delta",
    "shadow_delta_masks",
]

#: Compensation model identity (rides every qualified payload).
MODEL_VERSION = "veg-shadow-comp-v1"
MODEL_FORM = "geometric_overlay"
RECONCILIATION_NOTE = (
    "exact lane targets this revision; delta overlay is discarded on arrival "
    "(supersession fence re-publishes fast_exact)"
)

#: Disclosed model limitations (part of the qualification contract).
MODEL_LIMITATIONS = (
    "flat-ground projection (no terrain amplitude in the capsule sweep)",
    "no wall/building interaction (capsule may overlap building shadows)",
    "shadow plane only; tmrt/utci deltas are not modelled in v1",
    "crown modelled as a hard disc at canopy top/bottom (SOLWEIG trunk-zone "
    "rasterization approximation)",
)

#: Measured holdout evidence behind the shipped qualifier (verbatim from the
#: R7 calibration harness run — ``r7_veg_compensation_holdout.py --profile
#: full``, run id ``r7-veg-holdout-site500-20260905``, 2026-09-05, host
#: COD-MBP16-LOAN2, uptime 1030.2 s; committed copy: ``design/proofs/
#: r7_veg_compensation_holdout.json``). 8 real vegetation edits through the
#: REAL exact path (``run_full_tile`` → ``compute_utci``, ~125–137 s per
#: solve) on a /tmp copy of the site_500 cache; seeded 4/4 calibration/
#: holdout split; the ONE a-priori variant choice (cast-corridor-only vs
#: with-crown) frozen on the calibration half — every number below is from
#: the HELD-OUT half only. Micro cell-level change-detection metrics of the
#: predicted shadow-delta masks vs the bitwise diff of exact shadow planes.
#: Honest shape: recall is high (the overlay finds the changed region),
#: precision is lower (it over-predicts the change area, worst for moves
#: where old and new corridors both predict flips but overlapping building
#: shadows do not flip in exact truth).
DEFAULT_QUALIFIER_EVIDENCE: Mapping[str, Any] | None = {
    # Variant selected on the calibration split (mean IoU 0.518 vs 0.469);
    # the choice generalized on holdout (0.569 vs 0.503).
    "include_crown": False,  # cast_corridor_only
    "site_scope": (
        "site_500 (500x500 @ 2 m, forcing 2009-08-11), 8 edit cases, "
        "4 held out — single-site measurement, not a cross-site bound"
    ),
    "metrics": {
        "shadow_iou": {
            "n": 4, "p50": 0.6680711767668289, "p95": 0.7826922576312131,
            "p99": 0.786211544568992, "max": 0.7870913663034367,
            "min": 0.1518987341772152,
        },
        "shadow_precision": {
            "n": 4, "p50": 0.8017485783582701, "p95": 0.8634010734176362,
            "p99": 0.8675923330924078, "max": 0.8686401480111008,
            "min": 0.1571753986332574,
        },
        "shadow_recall": {
            "n": 4, "p50": 0.8558370123760585, "p95": 0.8933220394049801,
            "p99": 0.8934122670627277, "max": 0.8934348239771646,
            "min": 0.6993464052287581,
        },
        "shadow_recall_weighted": {"n": 4, "value": 0.8695652173913043},
    },
    "n_holdout": 4,
    "evidence": (
        "docs/incremental_design_tool/realtime_collaboration/design/proofs/"
        "r7_veg_compensation_holdout.json (curated copy of /tmp/r7_proof/"
        "holdout.json; /tmp/r7_proof/latency.json for the cost model)"
    ),
    "calibration_run_id": "r7-veg-holdout-site500-20260905",
}

#: Conservative influence policy (matches the incremental layer's defaults:
#: 5-degree direct-sun floor, 300 m shadow cap — the HARD RULE sky floor
#: applies to sky patches, untouched here).
_CONFIG = InfluenceConfig()

#: Affine cost model constants (ms), measured by the R7 latency benchmark:
#: fixed planning overhead + per-edited-tree capsule rasterization over the
#: sun-up timesteps. See /tmp/r7_proof/latency.json provenance.
PREDICTED_OVERHEAD_MS = 1.5
PREDICTED_PER_TREE_MS = 6.0


@dataclass(frozen=True)
class TreeShape:
    """The physical fields the capsule model needs (defaults per adapter)."""

    x_m: float
    y_m: float
    height_m: float
    canopy_radius_m: float = 5.0
    trunk_ratio: float = 0.25


class _SiteSource(Protocol):
    """Site geometry + deterministic sun positions (the plan-time inputs).

    ``geometry`` returns the manifest grid mapping (rows/cols/pixel/origin);
    ``sun_positions`` returns ``(altitude_deg, azimuth_deg)`` per timestep
    for the scenario's selected date — deterministic, from the site cache.
    """

    def geometry(self, site_id: str) -> Mapping[str, Any]: ...

    def sun_positions(self, site_id: str) -> Sequence[Sequence[float]]: ...


class SiteRegistrySunSource:
    """Adapt ``jobs.SiteRegistry`` to :class:`_SiteSource` (app wiring)."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def geometry(self, site_id: str) -> Mapping[str, Any]:
        return self._registry.geometry(site_id)

    def sun_positions(self, site_id: str) -> list[tuple[float, float]]:
        solar = np.asarray(self._registry.cache(site_id).solar)
        return [(float(row[0]), float(row[1])) for row in solar]


# ---------------------------------------------------------------------------
# Canonical-state tree extraction (reducer object-table shape)
# ---------------------------------------------------------------------------


def _tree_shapes(state: Mapping[str, Any] | None) -> dict[str, TreeShape]:
    """TreeSpec-like shapes from a canonical vegetation object table.

    Malformed objects (missing position/height) are skipped — domain
    validation belongs upstream; compensation never guesses dimensions.
    """
    objects = (state or {}).get("objects") or {}
    shapes: dict[str, TreeShape] = {}
    for entity_id, fields in objects.items():
        if not isinstance(fields, Mapping):
            continue
        try:
            shapes[str(entity_id)] = TreeShape(
                x_m=float(fields["x_m"]),
                y_m=float(fields["y_m"]),
                height_m=float(fields["height_m"]),
                canopy_radius_m=float(fields.get("canopy_radius_m", 5.0)),
                trunk_ratio=float(fields.get("trunk_ratio", 0.25)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return shapes


def _tree_edits(
    base_state: Mapping[str, Any] | None, current_state: Mapping[str, Any] | None
) -> dict[str, tuple[TreeShape | None, TreeShape | None]]:
    """Physical (generation-ignoring) tree diff: id -> (old, new)."""
    base = _tree_shapes(base_state)
    current = _tree_shapes(current_state)
    edits: dict[str, tuple[TreeShape | None, TreeShape | None]] = {}
    for entity_id in sorted(set(base) | set(current)):
        old = base.get(entity_id)
        new = current.get(entity_id)
        if old == new:
            continue  # physically identical (generation-only churn)
        edits[entity_id] = (old, new)
    return edits


# ---------------------------------------------------------------------------
# The geometric shadow model (single source of truth for kernel + harness)
# ---------------------------------------------------------------------------


def _disc(
    acc: np.ndarray,
    *,
    center_row: float,
    center_col: float,
    radius_cells: float,
) -> None:
    """Paint one disc into ``acc`` (clipped to the grid)."""
    if not math.isfinite(radius_cells) or radius_cells <= 0:
        return
    rows, cols = acc.shape
    r_min = max(0, int(math.floor(center_row - radius_cells)))
    r_max = min(rows - 1, int(math.ceil(center_row + radius_cells)))
    c_min = max(0, int(math.floor(center_col - radius_cells)))
    c_max = min(cols - 1, int(math.ceil(center_col + radius_cells)))
    if r_min > r_max or c_min > c_max:
        return
    row_idx = np.arange(r_min, r_max + 1)[:, None]
    col_idx = np.arange(c_min, c_max + 1)[None, :]
    mask = (row_idx - center_row) ** 2 + (col_idx - center_col) ** 2 <= (
        radius_cells**2 + 1e-9
    )
    acc[r_min : r_max + 1, c_min : c_max + 1] |= mask


def _capsule(
    acc: np.ndarray,
    tree: TreeShape,
    sun: SunPosition,
    *,
    pixel_size_m: float,
    origin_x_m: float,
    origin_y_m: float,
    include_crown: bool = True,
    maximum_shadow_length_m: float = _CONFIG.maximum_shadow_length_m,
) -> None:
    """Union the tree's shadow capsule into ``acc`` (north-up grid).

    Grid mapping (RasterGrid convention): ``col = (x - origin_x) / pixel``,
    ``row = (origin_y - y) / pixel`` — rows increase southward, columns
    eastward, so a northward displacement decreases the row index.
    """
    center_row = (origin_y_m - tree.y_m) / pixel_size_m
    center_col = (tree.x_m - origin_x_m) / pixel_size_m
    radius = tree.canopy_radius_m / pixel_size_m
    # The crown disc itself: ground under the canopy is vegetation shadow.
    # (``include_crown=False`` is the cast-corridor-only variant — the one
    # a-priori modeling choice the R7 calibration split arbitrates.)
    if include_crown:
        _disc(
            acc,
            center_row=center_row,
            center_col=center_col,
            radius_cells=radius,
        )
    if sun.altitude_deg <= 0.0:
        return  # sun below the horizon: no cast shadow in the exact model
    # Canopy-top and canopy-bottom disc displacements (clamped identically
    # to the incremental layer: 5-degree floor, 300 m cap).
    top_e, top_n = shadow_vector_m(
        tree.height_m,
        sun,
        minimum_altitude_deg=_CONFIG.minimum_direct_sun_altitude_deg,
        maximum_length_m=maximum_shadow_length_m,
    )
    bottom_e, bottom_n = shadow_vector_m(
        max(tree.trunk_ratio * tree.height_m, 1e-6),
        sun,
        minimum_altitude_deg=_CONFIG.minimum_direct_sun_altitude_deg,
        maximum_length_m=maximum_shadow_length_m,
    )
    # (row offset) = -north / pixel ; (col offset) = east / pixel
    top_row_off, top_col_off = -top_n / pixel_size_m, top_e / pixel_size_m
    bot_row_off, bot_col_off = -bottom_n / pixel_size_m, bottom_e / pixel_size_m
    _disc(
        acc,
        center_row=center_row + top_row_off,
        center_col=center_col + top_col_off,
        radius_cells=radius,
    )
    _disc(
        acc,
        center_row=center_row + bot_row_off,
        center_col=center_col + bot_col_off,
        radius_cells=radius,
    )
    # Sweep the segment between the bottom and top disc centers (the slanted
    # canopy volume's projection), one sample per pixel of travel.
    span = math.hypot(top_row_off - bot_row_off, top_col_off - bot_col_off)
    steps = min(int(math.ceil(span)) + 1, 256)
    for step in range(1, max(steps - 1, 0)):
        frac = step / max(steps - 1, 1)
        _disc(
            acc,
            center_row=center_row + bot_row_off + frac * (top_row_off - bot_row_off),
            center_col=center_col + bot_col_off + frac * (top_col_off - bot_col_off),
            radius_cells=radius,
        )


def shadow_delta_masks(
    edits: Mapping[str, tuple[TreeShape | None, TreeShape | None]],
    *,
    rows: int,
    cols: int,
    pixel_size_m: float,
    origin_x_m: float,
    origin_y_m: float,
    sun_positions: Sequence[Sequence[float]],
    include_crown: bool = True,
) -> list[tuple[int, np.ndarray]]:
    """Per-sun-up-timestep boolean CHANGE masks of a tree edit set.

    Each entry is ``(t, mask)`` where ``mask`` is the XOR of the
    base-capsule union and the current-capsule union (the cells whose
    vegetation-shadow state the model predicts to flip). Sun-down timesteps
    contribute nothing (the exact model casts no tree shadow then). This is
    the single source of truth shared by the kernel and the calibration
    harness — the harness measures THIS function against real exact solves.
    """
    masks: list[tuple[int, np.ndarray]] = []
    if not edits:
        return masks
    for t, row in enumerate(sun_positions):
        try:
            sun = SunPosition(
                altitude_deg=float(row[0]), azimuth_deg=float(row[1])
            )
        except (TypeError, ValueError, IndexError):
            continue
        if sun.altitude_deg <= 0.0:
            continue
        current_union = np.zeros((rows, cols), dtype=bool)
        base_union = np.zeros((rows, cols), dtype=bool)
        grid = dict(
            pixel_size_m=pixel_size_m,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            include_crown=include_crown,
        )
        for old, new in edits.values():
            if new is not None:
                _capsule(current_union, new, sun, **grid)
            if old is not None:
                _capsule(base_union, old, sun, **grid)
        changed = current_union ^ base_union
        if changed.any():
            masks.append((t, changed))
    return masks


def predict_shadow_delta(
    edits: Mapping[str, tuple[TreeShape | None, TreeShape | None]],
    *,
    rows: int,
    cols: int,
    pixel_size_m: float,
    origin_x_m: float,
    origin_y_m: float,
    sun_positions: Sequence[Sequence[float]],
    include_crown: bool = True,
) -> list[dict[str, Any]]:
    """Wire-sized per-timestep coverage of :func:`shadow_delta_masks`.

    One entry per timestep with a non-empty mask: ``{"t", "window"
    (half-open), "changed_cells"}``.
    """
    coverage: list[dict[str, Any]] = []
    for t, changed in shadow_delta_masks(
        edits,
        rows=rows,
        cols=cols,
        pixel_size_m=pixel_size_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        sun_positions=sun_positions,
        include_crown=include_crown,
    ):
        rows_changed, cols_changed = np.nonzero(changed)
        coverage.append(
            {
                "t": t,
                "window": {
                    "row_start": int(rows_changed.min()),
                    "row_stop": int(rows_changed.max()) + 1,
                    "col_start": int(cols_changed.min()),
                    "col_stop": int(cols_changed.max()) + 1,
                },
                "changed_cells": int(rows_changed.size),
            }
        )
    return coverage


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


class VegetationCompensationKernel:
    """Compensated vegetation fast kernel: anchored geometric overlay.

    Priority discipline (compensation LAST): view-only epochs keep the
    structural ``fast_exact`` classification; only PURE vegetation epochs
    are compensation candidates; met/building/landcover/mixed epochs are
    served by the exact lane and get an honest ``visual_pending``.
    """

    name = "veg_shadow_comp_v1"

    def __init__(
        self,
        *,
        store: Any,
        sites: _SiteSource,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self._store = store
        self._sites = sites
        self._evidence = evidence if evidence is not None else DEFAULT_QUALIFIER_EVIDENCE

    # -- protocol ---------------------------------------------------------------

    def predicted_cost_ms(self, plan: FastPlan) -> float:
        # Each operation touches at most one tree; the affine model is
        # conservative (edited trees <= operations). View classification is
        # effectively free.
        if plan.families and all(f == "output_view" for f in plan.families):
            return 1.0
        return PREDICTED_OVERHEAD_MS + PREDICTED_PER_TREE_MS * max(
            plan.operation_count, 0
        )

    def eligible(self, plan: FastPlan) -> bool:
        """Compensation candidacy: PURE vegetation epochs only."""
        return bool(plan.families) and all(
            family == "vegetation_geometry" for family in plan.families
        )

    def run(self, plan: FastPlan) -> FastKernelResult:
        # Structural exact path first (priority discipline).
        if plan.families and all(f == "output_view" for f in plan.families):
            return FastKernelResult(
                result_class="fast_exact",
                payload={"kernel": self.name, "view_only": True},
            )
        if not self.eligible(plan):
            return FastKernelResult(
                result_class="visual_pending",
                payload={
                    "kernel": self.name,
                    "reason": "ineligible_families",
                    "families": list(plan.families),
                },
            )

        scenario = self._store.get_scenario(plan.workspace_id)
        if scenario is None:
            return self._pending("no_scenario")
        base = self._store.latest_result(plan.workspace_id)
        if base is None or not _manifest_has_shadow(base.manifest):
            return self._pending("no_exact_base")

        anchor = int(base.scene_version)
        base_state = self._store.canonical_state_at(plan.workspace_id, anchor)
        base_family = (
            base_state.families.get("vegetation_geometry") if base_state else None
        )
        current_state = plan.canonical_state.get("vegetation_geometry")
        edits = _tree_edits(base_family, current_state)
        if not edits:
            return self._pending("no_vegetation_delta")

        try:
            geometry = dict(self._sites.geometry(scenario.site_id))
            suns = list(self._sites.sun_positions(scenario.site_id))
        except Exception:  # noqa: BLE001 - site data problems are refusals
            return self._pending("no_site_geometry")

        if self._evidence is None:
            # Anchored and computable, but no MEASURED holdout evidence is
            # wired: refuse to claim qualification (no guessing — the
            # mission's hard rule).
            return FastKernelResult(
                result_class="visual_pending",
                payload={
                    "kernel": self.name,
                    "reason": "no_calibration_evidence",
                },
            )

        evidence = dict(self._evidence)
        # The variant the calibration split selected (``cast_corridor_only``
        # means the crown disc itself is NOT part of the predicted shadow
        # set). Defaults to with-crown for hand-built evidence dicts, but the
        # SHIPPED evidence carries the measured selection — the kernel must
        # predict with exactly the variant whose holdout error it declares.
        include_crown = bool(evidence.get("include_crown", True))
        variant = "with_crown" if include_crown else "cast_corridor_only"
        coverage = predict_shadow_delta(
            edits,
            rows=int(geometry["rows"]),
            cols=int(geometry["cols"]),
            pixel_size_m=float(geometry["pixel_size_m"]),
            origin_x_m=float(geometry["origin_x_m"]),
            origin_y_m=float(geometry["origin_y_m"]),
            sun_positions=suns,
            include_crown=include_crown,
        )

        payload: dict[str, Any] = {
            "kernel": self.name,
            "qualifier": {
                "model_version": MODEL_VERSION,
                "form": MODEL_FORM,
                "domain": {
                    "families": ["vegetation_geometry"],
                    "variables": ["shadow"],
                    "site_scope": str(evidence.get("site_scope", "measured")),
                    "variant": variant,
                },
                "error_evidence": {
                    "metrics": dict(evidence.get("metrics", {})),
                    "n_holdout": int(evidence.get("n_holdout", 0)),
                    "evidence": str(evidence.get("evidence", "")),
                    "calibration_run_id": str(evidence.get("calibration_run_id", "")),
                },
                "reconciliation": RECONCILIATION_NOTE,
            },
            "exact_base_revision": anchor,
            "base_planes": {
                "result_version": anchor,
                "variables": ["shadow"],
                "source": "exact lane result",
            },
            "delta": {
                "trees": len(edits),
                "time_steps": len(suns),
                "model": MODEL_FORM,
                "edits": {
                    entity_id: {
                        "kind": (
                            "added"
                            if old is None
                            else "removed" if new is None else "changed"
                        )
                    }
                    for entity_id, (old, new) in sorted(edits.items())
                },
            },
            "coverage": coverage,
            "limitations": list(MODEL_LIMITATIONS),
        }
        return FastKernelResult(result_class="fast_qualified", payload=payload)

    # -- helpers ----------------------------------------------------------------

    def _pending(self, reason: str) -> FastKernelResult:
        return FastKernelResult(
            result_class="visual_pending",
            payload={"kernel": self.name, "reason": reason},
        )


def _manifest_has_shadow(manifest: Mapping[str, Any]) -> bool:
    """Whether an exact result manifest carries shadow planes."""
    variables = manifest.get("variables")
    if not isinstance(variables, (list, tuple)):
        return False
    return any(
        isinstance(entry, Mapping) and entry.get("name") == "shadow"
        for entry in variables
    )
