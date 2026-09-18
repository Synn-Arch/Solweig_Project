# SPDX-License-Identifier: GPL-3.0-only
"""R7 -- the compensated-vegetation holdout harness (calibration + honest
held-out error measurement).

What this proves (and how):

* **The model is measured against REAL exact solves.** Every case drives a
  genuine vegetation edit sequence through the REAL full-tile exact path
  (``run_full_tile`` -> ``compute_utci``) on a COPY of the site_500 cache
  under ``/tmp`` (the canonical site cache is READ-ONLY; the harness
  clones it first with an APFS clone so the original is never touched).
  Ground truth per timestep is the bitwise diff of the edited run's shadow
  planes against the baseline run's shadow planes -- same solver, same
  inputs, so equal bits everywhere the trees do not matter.
* **Prediction comes from the shipping kernel code.** The harness imports
  ``solweig_gpu.server.realtime.compensated.shadow_delta_masks`` -- the
  exact function the fast lane's kernel runs. No reimplementation drift.
* **Calibration/holdout discipline.** The model has exactly ONE a-priori
  discrete modeling choice: whether the crown disc itself (ground under
  the canopy) belongs to the predicted shadow set, or only the cast
  corridor. The seeded shuffle splits the cases; the choice is frozen on
  the CALIBRATION half only, and every reported error number comes from
  the HELD-OUT half. Calibration-half diagnostics are kept in the
  evidence under a clearly separate key for transparency, but the
  qualifier's ``error_evidence`` uses holdout numbers only.
* **No fitted thermal numbers.** The v1 model is the purely geometric
  overlay (capsule union sweep with the real ``shadow_vector_m`` machinery
  -- 5-degree sun floor, 300 m cap). Nothing is regressed onto the exact
  solves; if this holdout error were unacceptable, the honest move is to
  NOT ship the qualifier, not to fit coefficients quietly.
* **Deadline honesty.** The latency stage measures the kernel's compute
  p50/p95/p99 on the heaviest case and FAILS the run if the kernel's
  affine ``predicted_cost_ms`` constants do not cover measured p99 with
  margin -- the scheduler's deadline gate is only as honest as the
  prediction.

Output: ``/tmp/r7_proof/holdout.json`` (full evidence) and
``/tmp/r7_proof/latency.json``. Baseline planes:
``/tmp/r7_proof/base_shadow.npy`` (computed once, ~130 s, reused).

Usage::

    python r7_veg_compensation_holdout.py [--profile quick|full]

``quick`` is a smoke profile (2 cases, not valid evidence -- the JSON is
marked ``not_for_evidence``). ``full`` is the evidence profile (8 cases,
5 calibration-free holdout reporting).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_gpu.incremental.cache import SiteCache  # noqa: E402
from solweig_gpu.incremental.geometry import RasterGrid, TreeSpec  # noqa: E402
from solweig_gpu.incremental.manifest import load_manifest  # noqa: E402
from solweig_gpu.incremental.solver import (  # noqa: E402
    SiteForcing,
    load_site_forcing,
    run_full_tile,
)
from solweig_gpu.incremental.trees import TreeLayer  # noqa: E402
from solweig_gpu.server.realtime.compensated import (  # noqa: E402
    MODEL_FORM,
    MODEL_VERSION,
    PREDICTED_OVERHEAD_MS,
    PREDICTED_PER_TREE_MS,
    TreeShape,
    shadow_delta_masks,
)

PROOF_ROOT = Path("/tmp/r7_proof")
EVIDENCE_JSON = PROOF_ROOT / "holdout.json"
LATENCY_JSON = PROOF_ROOT / "latency.json"
BASELINE_NPY = PROOF_ROOT / "base_shadow.npy"

#: The canonical site cache is READ-ONLY repository data; the harness
#: works on an APFS clone under /tmp (clone is cheap; original untouched).
SOURCE_CACHE = Path("/Users/alansynn/Workspace/solweig/site-cache/site_500")
CACHE_DIR = PROOF_ROOT / "site_500"
SITE_DIR = Path("/Users/alansynn/Workspace/solweig/Input_subset/processed_inputs")
SELECTED_DATE = "2009-08-11"  # the site_500 deployment date (benchmarks doc)

CALIBRATION_RUN_ID = "r7-veg-holdout-site500-20260905"
_SPLIT_SEED = 20260905

#: fixture_tree_A (fixtures/site_500/manifest.json) -- the canonical tree.
TREE_A = dict(
    tree_id="fixture_tree_A",
    x_m=622249.7066,
    y_m=3354163.3479,
    height_m=10.0,
    canopy_radius_m=5.5,
    trunk_ratio=0.25,
)


def _tree(tree_id: str, *, x_m: float, y_m: float, height_m: float,
          canopy_radius_m: float, trunk_ratio: float = 0.25) -> TreeSpec:
    return TreeSpec(
        tree_id=tree_id,
        x_m=x_m,
        y_m=y_m,
        height_m=height_m,
        canopy_radius_m=canopy_radius_m,
        trunk_ratio=trunk_ratio,
    )


# ---------------------------------------------------------------------------
# Cases: deterministic edit sequences over the treeless site_500 baseline
# ---------------------------------------------------------------------------


def _cases() -> list[dict[str, Any]]:
    """Seeded-free (fully deterministic) edit sequences: add/move/resize,
    single- and multi-tree, interior and near-edge placements."""
    a = TREE_A

    def add(tree: TreeSpec) -> Callable[[TreeLayer], Any]:
        return lambda layer: layer.add_tree(tree)

    def move(tree_id: str, *, x_m: float, y_m: float) -> Callable[[TreeLayer], Any]:
        return lambda layer: layer.move_tree(tree_id, x_m=x_m, y_m=y_m)

    def update(tree_id: str, **changes: float) -> Callable[[TreeLayer], Any]:
        return lambda layer: layer.update_tree(tree_id, **changes)

    tall = _tree(
        "case_tall", x_m=a["x_m"] + 30.0, y_m=a["y_m"] + 15.0,
        height_m=18.0, canopy_radius_m=8.0,
    )
    small = _tree(
        "case_small", x_m=a["x_m"] - 40.0, y_m=a["y_m"] - 25.0,
        height_m=6.0, canopy_radius_m=3.0,
    )
    west = _tree(
        "case_west", x_m=a["x_m"] - 180.0, y_m=a["y_m"] + 5.0,
        height_m=12.0, canopy_radius_m=6.0,
    )
    north = _tree(
        "case_north", x_m=a["x_m"] + 8.0, y_m=a["y_m"] + 190.0,
        height_m=9.0, canopy_radius_m=5.0,
    )
    pair_b = _tree(
        "case_pair_b", x_m=a["x_m"] - 80.0, y_m=a["y_m"] - 60.0,
        height_m=12.0, canopy_radius_m=6.0,
    )
    fixture = _tree(
        "fixture_tree_A", x_m=a["x_m"], y_m=a["y_m"],
        height_m=a["height_m"], canopy_radius_m=a["canopy_radius_m"],
        trunk_ratio=a["trunk_ratio"],
    )
    return [
        {"name": "add_fixture", "ops": [add(fixture)]},
        {"name": "add_tall", "ops": [add(tall)]},
        {"name": "add_small", "ops": [add(small)]},
        {"name": "add_west_offset", "ops": [add(west)]},
        {"name": "add_north_offset", "ops": [add(north)]},
        {
            "name": "add_pair",
            "ops": [add(fixture), add(pair_b)],
        },
        {
            "name": "move_200m_east",
            "ops": [
                add(fixture),
                move("fixture_tree_A", x_m=a["x_m"] + 200.0, y_m=a["y_m"]),
            ],
        },
        {
            "name": "resize_h18_r8",
            "ops": [
                add(fixture),
                update("fixture_tree_A", height_m=18.0, canopy_radius_m=8.0),
            ],
        },
    ]


def _layer_edits(layer: TreeLayer) -> dict[str, tuple[TreeShape | None, TreeShape | None]]:
    """Collapse the layer's edit log to per-tree NET (first old, last new)."""

    def shape(spec: TreeSpec | None) -> TreeShape | None:
        if spec is None:
            return None
        return TreeShape(
            x_m=spec.x_m, y_m=spec.y_m, height_m=spec.height_m,
            canopy_radius_m=spec.canopy_radius_m, trunk_ratio=spec.trunk_ratio,
        )

    net: dict[str, dict[str, TreeShape | None]] = {}
    for edit in layer.edits:
        entry = net.setdefault(edit.tree_id, {"old": None, "new": None})
        if entry["old"] is None and edit.old_tree is not None:
            entry["old"] = shape(edit.old_tree)
        entry["new"] = shape(edit.new_tree)
    return {
        tree_id: (entry["old"], entry["new"]) for tree_id, entry in net.items()
    }


# ---------------------------------------------------------------------------
# Real exact solves
# ---------------------------------------------------------------------------


def _ensure_cache_copy() -> None:
    if CACHE_DIR.exists():
        return
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[r7] cloning site cache (APFS clone) {SOURCE_CACHE} -> {CACHE_DIR}")
    shutil.copytree(SOURCE_CACHE, CACHE_DIR, copy_function=shutil.copy2)


def _baseline_shadow(
    cache: SiteCache, forcing: SiteForcing, manifest: Any
) -> np.ndarray:
    if BASELINE_NPY.exists():
        planes = np.load(BASELINE_NPY)
        _check_planes(planes, manifest, "cached baseline")
        return planes
    print("[r7] computing baseline exact solve (~130 s)...")
    started = time.monotonic()
    layer = TreeLayer(
        np.asarray(cache.tree_base), _grid(manifest), scenario_id="r7_baseline"
    )
    planes = _solve_shadow(cache, layer, forcing, PROOF_ROOT / "scratch_base")
    np.save(BASELINE_NPY, planes)
    print(f"[r7] baseline done in {time.monotonic() - started:.1f}s")
    return planes


def _grid(manifest: Any) -> RasterGrid:
    return RasterGrid(
        rows=manifest.rows,
        cols=manifest.cols,
        pixel_size_m=manifest.pixel_size_m,
        origin_x_m=manifest.origin_x_m,
        origin_y_m=manifest.origin_y_m,
    )


def _check_planes(planes: np.ndarray, manifest: Any, label: str) -> None:
    if planes.shape != (manifest.time_steps, manifest.rows, manifest.cols):
        raise SystemExit(
            f"[r7] {label} planes shape {planes.shape} does not match the "
            f"manifest {(manifest.time_steps, manifest.rows, manifest.cols)}"
        )


def _solve_shadow(
    cache: SiteCache, layer: TreeLayer, forcing: SiteForcing, scratch: Path
) -> np.ndarray:
    """One REAL full-tile exact solve; returns the shadow planes [t, r, c]."""
    result = run_full_tile(
        cache,
        layer,
        forcing=forcing,
        site_dir=SITE_DIR,
        scratch_dir=scratch,
        requested_variables=("shadow",),
    )
    planes = np.asarray(result["shadow"], dtype=np.float32)
    return planes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _case_metrics(
    truth: np.ndarray,
    pred_masks: list[tuple[int, np.ndarray]],
) -> dict[str, Any]:
    """Micro precision/recall/IoU of predicted change cells vs truth cells."""
    tp = fp = fn = 0
    truth_cells = 0
    pred_cells = 0
    per_t: list[dict[str, int]] = []
    pred_by_t = dict(pred_masks)
    for t in range(truth.shape[0]):
        g = truth[t]
        p = pred_by_t.get(t, np.zeros_like(g, dtype=bool))
        g_cells = int(g.sum())
        p_cells = int(p.sum())
        truth_cells += g_cells
        pred_cells += p_cells
        step_tp = int((p & g).sum())
        tp += step_tp
        fp += p_cells - step_tp
        fn += g_cells - step_tp
        per_t.append({"t": t, "truth": g_cells, "pred": p_cells, "tp": step_tp})
    denom_iou = tp + fp + fn
    return {
        "iou_micro": (tp / denom_iou) if denom_iou else 1.0,
        "precision_micro": (tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall_micro": (tp / (tp + fn)) if (tp + fn) else 1.0,
        "truth_cells": truth_cells,
        "pred_cells": pred_cells,
        "per_timestep": per_t,
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": float(np.percentile(ordered, 50)),
        "p95": float(np.percentile(ordered, 95)),
        "p99": float(np.percentile(ordered, 99)),
        "max": float(ordered[-1]),
        "min": float(ordered[0]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


PROFILES: dict[str, dict[str, Any]] = {
    "quick": {"cases": 2, "latency_reps": 10, "not_for_evidence": True},
    "full": {"cases": 8, "latency_reps": 40},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=os_env_profile(),
        choices=sorted(PROFILES),
    )
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]

    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()

    _ensure_cache_copy()
    manifest = load_manifest(CACHE_DIR / "manifest.json")
    cache = SiteCache.load(CACHE_DIR)
    grid = _grid(manifest)
    forcing = load_site_forcing(
        cache, site_dir=SITE_DIR, selected_date_str=SELECTED_DATE
    )
    suns = [(float(row[0]), float(row[1])) for row in np.asarray(cache.solar)]
    geometry = dict(
        rows=manifest.rows,
        cols=manifest.cols,
        pixel_size_m=manifest.pixel_size_m,
        origin_x_m=manifest.origin_x_m,
        origin_y_m=manifest.origin_y_m,
        sun_positions=suns,
    )

    base_planes = _baseline_shadow(cache, forcing, manifest)

    cases = _cases()[: profile["cases"]]
    print(f"[r7] profile={args.profile} cases={[c['name'] for c in cases]}")

    # -- real exact solves + prediction per variant -------------------------
    # Solves are deterministic; the truth diff is cached per case under
    # /tmp/r7_proof so a late-stage crash (or a metrics-schema fix) never
    # costs the ~130 s per case again. Cached truth is re-guarded below.
    variants = {"with_crown": True, "cast_corridor_only": False}
    rows: list[dict[str, Any]] = []
    for case in cases:
        layer = TreeLayer(np.asarray(cache.tree_base), grid, scenario_id=case["name"])
        for op in case["ops"]:
            op(layer)
        edits = _layer_edits(layer)
        truth_path = PROOF_ROOT / f"truth_{case['name']}.npy"
        if truth_path.exists():
            truth = np.load(truth_path)
            solve_s = -1.0  # cached: no fresh solve this run
            cached = True
        else:
            scratch = PROOF_ROOT / f"scratch_{case['name']}"
            t0 = time.monotonic()
            planes = _solve_shadow(cache, layer, forcing, scratch)
            solve_s = time.monotonic() - t0
            _check_planes(planes, manifest, case["name"])
            shutil.rmtree(scratch, ignore_errors=True)  # planes are in memory
            truth = planes != base_planes
            np.save(truth_path, truth)
            cached = False
        truth_cells = int(truth.sum())
        # Vacuity + drift guards: an edit case MUST change cells, and must
        # NOT flip a global-scale fraction (trees are local features).
        if truth_cells == 0:
            raise SystemExit(f"[r7] VACUOUS: case {case['name']} changed no cells")
        total = truth.size
        if truth_cells > 0.05 * total:
            raise SystemExit(
                f"[r7] DRIFT: case {case['name']} flipped {truth_cells}/{total} "
                "cells (>5%) -- solver nondeterminism or wrong baseline"
            )
        variant_metrics = {
            name: _case_metrics(truth, shadow_delta_masks(
                edits, **geometry, include_crown=include
            ))
            for name, include in variants.items()
        }
        row = {
            "case": case["name"],
            "edits": edits,
            "edit_kinds": [
                ("add" if old is None else "remove" if new is None else "change")
                for old, new in edits.values()
            ],
            "solve_seconds": None if cached else round(solve_s, 1),
            "solve_cached": cached,
            "truth_cells": truth_cells,
            "variants": variant_metrics,
        }
        rows.append(row)
        print(
            f"[r7] {case['name']}: truth={truth_cells} cells, "
            + ("solve=cached, " if cached else f"solve={solve_s:.0f}s, ")
            + ", ".join(
                f"{name}: iou={m['iou_micro']:.3f} p={m['precision_micro']:.3f} "
                f"r={m['recall_micro']:.3f}"
                for name, m in variant_metrics.items()
            )
        )

    # -- calibration/holdout split (seeded; freeze variant on calibration) --
    rng = np.random.default_rng(_SPLIT_SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_calibration = len(rows) // 2
    calibration_idx = sorted(order[:n_calibration])
    holdout_idx = sorted(order[n_calibration:])
    if not holdout_idx:
        raise SystemExit("[r7] VACUOUS: empty holdout split (need >= 2 cases)")

    def mean_iou(indices: list[int], variant: str) -> float:
        return float(
            np.mean([rows[i]["variants"][variant]["iou_micro"] for i in indices])
        )

    selection = {
        variant: {
            "calibration_mean_iou": mean_iou(calibration_idx, variant),
            "holdout_mean_iou": mean_iou(holdout_idx, variant),
        }
        for variant in variants
    }
    chosen = max(selection, key=lambda v: selection[v]["calibration_mean_iou"])
    print(
        f"[r7] split calibration={calibration_idx} holdout={holdout_idx}; "
        f"variant selected on calibration: {chosen} "
        f"({selection[chosen]['calibration_mean_iou']:.3f})"
    )

    holdout_rows = [rows[i] for i in holdout_idx]
    holdout_iou = [r["variants"][chosen]["iou_micro"] for r in holdout_rows]
    holdout_precision = [
        r["variants"][chosen]["precision_micro"] for r in holdout_rows
    ]
    holdout_recall = [r["variants"][chosen]["recall_micro"] for r in holdout_rows]
    micro_tp = sum(
        r["variants"][chosen]["truth_cells"] * r["variants"][chosen]["recall_micro"]
        for r in holdout_rows
    )

    evidence: dict[str, Any] = {
        "schema": "r7-veg-compensation-holdout/1",
        "not_for_evidence": profile.get("not_for_evidence", False),
        "site": {
            "cache": "site_500",
            "model_version_cache": getattr(manifest, "model_version", None),
            "selected_date": SELECTED_DATE,
            "grid": {
                "rows": manifest.rows,
                "cols": manifest.cols,
                "pixel_size_m": manifest.pixel_size_m,
            },
            "time_steps": manifest.time_steps,
        },
        "model": {"model_version": MODEL_VERSION, "form": MODEL_FORM},
        "calibration_run_id": CALIBRATION_RUN_ID,
        "split": {
            "seed": _SPLIT_SEED,
            "calibration_cases": [rows[i]["case"] for i in calibration_idx],
            "holdout_cases": [rows[i]["case"] for i in holdout_idx],
        },
        "variant_selection": {"options": selection, "chosen": chosen},
        "holdout": {
            "n": len(holdout_rows),
            "metrics": {
                "shadow_iou": _percentiles(holdout_iou),
                "shadow_precision": _percentiles(holdout_precision),
                "shadow_recall": _percentiles(holdout_recall),
            },
            "micro_recall_weighted": float(
                micro_tp / max(sum(r["truth_cells"] for r in holdout_rows), 1)
            ),
            "per_case": [
                {
                    "case": r["case"],
                    "iou": r["variants"][chosen]["iou_micro"],
                    "precision": r["variants"][chosen]["precision_micro"],
                    "recall": r["variants"][chosen]["recall_micro"],
                    "truth_cells": r["truth_cells"],
                    "pred_cells": r["variants"][chosen]["pred_cells"],
                }
                for r in holdout_rows
            ],
        },
        "calibration_diagnostics_only": {
            "note": "NOT part of the qualifier error_evidence (holdout only)",
            "per_case": [
                {
                    "case": r["case"],
                    "iou": r["variants"][chosen]["iou_micro"],
                }
                for r in (rows[i] for i in calibration_idx)
            ],
        },
        "cases_raw": [
            {k: v for k, v in row.items() if k != "edits"} for row in rows
        ],
    }

    # -- latency: the kernel's own compute, measured, prediction-gated ------
    heaviest = max(rows, key=lambda r: len(r["edits"]))
    latency_edits = heaviest["edits"]
    reps = int(profile["latency_reps"])
    samples_ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        shadow_delta_masks(latency_edits, **geometry, include_crown=variants[chosen])
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    measured = _percentiles(samples_ms)
    predicted_ms = PREDICTED_OVERHEAD_MS + PREDICTED_PER_TREE_MS * len(latency_edits)
    latency = {
        "schema": "r7-veg-compensation-latency/1",
        "case": heaviest["case"],
        "trees": len(latency_edits),
        "reps": reps,
        "measured_ms": measured,
        "predicted_cost_ms": predicted_ms,
        "predicted_constants": {
            "overhead_ms": PREDICTED_OVERHEAD_MS,
            "per_tree_ms": PREDICTED_PER_TREE_MS,
        },
        "prediction_covers_measured_p99": bool(
            predicted_ms >= 1.25 * float(measured["p99"])
        ),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    if not latency["prediction_covers_measured_p99"]:
        print(
            f"[r7] LATENCY DISHONESTY: predicted {predicted_ms:.1f} ms does not "
            f"cover 1.25x measured p99 {measured['p99']:.1f} ms -- raise the "
            "PREDICTED_* constants before shipping the kernel"
        )
        LATENCY_JSON.write_text(json.dumps(latency, indent=2, sort_keys=True))
        EVIDENCE_JSON.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        return 2

    uptime_s = time.monotonic() - started
    evidence["latency"] = latency
    evidence["uptime"] = {
        "seconds": round(uptime_s, 1),
        "started_utc": started_wall.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    evidence["host"] = platform.node()
    evidence["python"] = platform.python_version()

    PROOF_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    LATENCY_JSON.write_text(json.dumps(latency, indent=2, sort_keys=True))

    holdout_metrics = evidence["holdout"]["metrics"]
    print(f"[r7] evidence: {EVIDENCE_JSON}")
    print(f"[r7] latency: {LATENCY_JSON}")
    print(
        "[r7] HOLDOUT (reported): "
        + "; ".join(
            f"{name}: p50={m['p50']:.3f} p95={m['p95']:.3f} max={m['max']:.3f}"
            for name, m in holdout_metrics.items()
        )
    )
    print(f"[r7] uptime: {uptime_s:.1f}s")
    return 0


def os_env_profile() -> str:
    return os.environ.get("R7_PROFILE", "full")


if __name__ == "__main__":
    raise SystemExit(main())
