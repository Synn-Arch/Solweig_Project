# SPDX-License-Identifier: GPL-3.0-only
"""T10 G2.1 current-state measurement: does the LIVE warm route hit?

A probe + pinning fixture pair (ultrafast bitwise plan, task T10 card:
"measure the CURRENT G2.1 warm-start path FIRST"). This module drives the
REAL incremental executor (prepared-site fixture, real SVF, real physics —
the same recipe ``tests/test_incremental_met_warm_path.py`` uses) through
a canonical edit sequence and records, per edit:

  * which route served it (worker full / met warm sparse path),
  * the warm diagnostics (``met_warm_r0`` / ``met_warm_resume_step`` /
    ``met_warm_steps_solved`` / ``met_warm_refusal``),
  * the anchor inventory under ``<results>/<scenario>/checkpoints`` after
    each publication (revision, thermal vs coverage-only, next_step),
  * wall seconds per batch (rough signal only — the exit metric is
    steps_solved, never wall time).

This is a MEASUREMENT harness, not a runtime module: it may import the
torch-side executor freely. The answers it produces are pinned by
``tests/ultrafast/test_g21_warm_measurement.py`` and recorded under
``~/Workspace/solweig_ultrafast_artifacts/t10/``.

Honest-scoring rule (DESIGN 11.1): a checkpoint whose only content is a
coverage manifest is NOT a thermal hit. A warm hit requires
``met_warm_resume_step > 0`` on a THERMAL checkpoint AND a real
``met_warm_steps_solved`` reduction versus the fresh-full T steps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t10"
)

MET_HOURS = tuple(range(0, 24))  # T = 24 timesteps (honest ~24 baseline)


def build_warm_executor(tmp_path: Path):
    """The real-physics executor: prepared site + REAL baseline SVF.

    Same recipe as ``tests/test_incremental_met_warm_path.py`` — the fake
    16-patch SVF cubes of the plain tiny-site fixture cannot feed
    ``solve_window`` (the physics iterates 153 patches).
    """
    from tests.test_incremental_executor import _executor
    from tests.test_incremental_worker import (
        TINY_COLS,
        TINY_EPSG,
        TINY_ORIGIN,
        TINY_PIXEL,
        TINY_ROWS,
        TINY_TREE,
        _build_cache,
        _compute_baseline_svf,
        _make_prepared_site,
    )

    grid, site = _make_prepared_site(
        tmp_path / "site",
        rows=TINY_ROWS,
        cols=TINY_COLS,
        pixel=TINY_PIXEL,
        origin=TINY_ORIGIN,
        epsg=TINY_EPSG,
        base_trees=(TINY_TREE,),
        met_hours=MET_HOURS,
    )
    _compute_baseline_svf(site)
    return _executor(tmp_path, grid=grid, site=site)


def rad_met_edit(executor, *, row: int, revision: int, value: float,
                 edit_id: str):
    """A validated RADIATION-AFFECTING met edit (air_temperature) at ``row``."""
    from tests.test_incremental_executor import _met_edit

    return _met_edit(
        executor,
        revision=revision,
        times=(row,),
        time_index=row,
        value=value,
        edit_id=edit_id,
    )


def veg_edit(executor, *, revision: int, edit_id: str):
    """A validated vegetation-geometry edit (tree height change)."""
    from tests.test_incremental_executor import _veg_command

    command = _veg_command(revision=revision, edit_id=edit_id, times=(1,))
    return executor.validate(command)


def anchor_inventory(executor) -> list[dict[str, Any]]:
    """Every checkpoint on disk: revision, thermal-ness, next_step."""
    from solweig_gpu.incremental.checkpoints import (
        list_checkpoints,
        load_checkpoint,
    )

    root = executor.results_root / executor._scenario_id / "checkpoints"
    out: list[dict[str, Any]] = []
    for revision in list_checkpoints(root):
        record = load_checkpoint(root / f"rev-{revision:06d}")
        out.append(
            {
                "revision": revision,
                "thermal": bool(record.thermal_tensors),
                "next_step": record.next_step,
                "fingerprinted": record.input_fingerprint is not None,
                "scalars": sorted(record.thermal_scalars),
            }
        )
    return out


def run_edit(executor, edit) -> dict[str, Any]:
    started = time.monotonic()
    executed = executor.execute([edit])
    elapsed = time.monotonic() - started
    diagnostics = dict(executed.diagnostics)
    return {
        "edit_id": edit.command.edit_id,
        "adapter": edit.adapter_id,
        "published": bool(executed.published),
        "mode": executed.mode,
        "wall_s": round(elapsed, 3),
        "met_warm_path": diagnostics.get("met_warm_path"),
        "met_warm_r0": diagnostics.get("met_warm_r0"),
        "met_warm_resume_step": diagnostics.get("met_warm_resume_step"),
        "met_warm_steps_solved": diagnostics.get("met_warm_steps_solved"),
        "met_warm_checkpoint_revision": diagnostics.get(
            "met_warm_checkpoint_revision"
        ),
        "met_warm_refusal": diagnostics.get("met_warm_refusal"),
        "met_warm_path_refusal": diagnostics.get("met_warm_path_refusal"),
        "met_fast_path_refusal": diagnostics.get("met_fast_path_refusal"),
        "anchors_after": anchor_inventory(executor),
        # Honest score: a REAL warm hit is resume_step > 0 on the warm
        # route (a thermal checkpoint actually advanced the solve).
        "warm_hit": bool(
            diagnostics.get("met_warm_path")
            and diagnostics.get("met_warm_resume_step", 0) > 0
        ),
    }


def measure_sequence(tmp_path: Path) -> dict[str, Any]:
    """The canonical edit sequence; returns the full measurement record."""
    executor = build_warm_executor(tmp_path)
    total = len(MET_HOURS)
    edits: list[dict[str, Any]] = []

    # m1 — seed: fresh store cannot serve any prefix row; the warm path
    # refuses (coverage) and the worker's full solve publishes 0..T cold
    # (capturing the state@T anchor).
    edits.append(
        run_edit(executor, rad_met_edit(executor, row=1, revision=0,
                                        value=21.0, edit_id="m1"))
    )
    # m2 — first edit INSIDE the day: the only anchor is state@T whose met
    # prefix covers the edited row; the warm path still serves the SPARSE
    # suffix with a cold split and re-anchors at r0=20.
    edits.append(
        run_edit(executor, rad_met_edit(executor, row=20, revision=1,
                                        value=29.0, edit_id="m2"))
    )
    # m3 — late edit after the r0 anchor: the anchor@20's met prefix
    # (rows 0..19) is unchanged, so the suffix solves WARM from k=20.
    edits.append(
        run_edit(executor, rad_met_edit(executor, row=21, revision=2,
                                        value=30.0, edit_id="m3"))
    )
    # m4 — early edit that invalidates every existing anchor's prefix.
    edits.append(
        run_edit(executor, rad_met_edit(executor, row=5, revision=3,
                                        value=27.0, edit_id="m4"))
    )
    # v1 — GEOMETRY edit after all met anchors: thermal reuse across a
    # scene-geometry change is forbidden (DESIGN 11.2); measure which
    # route actually serves it and that no warm route engages.
    edits.append(
        run_edit(executor, veg_edit(executor, revision=4, edit_id="v1"))
    )

    return {
        "total_timesteps": total,
        "fresh_full_steps": total,
        "edits": edits,
        "summary": {
            "warm_hits": sum(1 for e in edits if e["warm_hit"]),
            "warm_hit_edit_ids": [
                e["edit_id"] for e in edits if e["warm_hit"]
            ],
            "best_steps_solved": min(
                (
                    e["met_warm_steps_solved"]
                    for e in edits
                    if e["met_warm_steps_solved"] is not None
                ),
                default=None,
            ),
            "thermal_anchors_final": [
                a for a in anchor_inventory(executor) if a["thermal"]
            ],
        },
    }


def measure_geometry_sequence(tmp_path: Path) -> dict[str, Any]:
    """The live R1 witness sequence: a met edit after a GEOMETRY edit
    must NOT warm from the previous scene's anchor even when its clock
    position and met prefix both fit.

    m1 (row 1) seeds the store; m2 (row 20) stages anchor@20 under
    geometry A; the vegetation edit moves the scene to geometry B (its
    own full-solve anchor@T is structurally unusable); m5 (row 21) then
    finds anchor@20 — met prefix rows 0..19 UNCHANGED (only the scene
    lineage differs). A met-only applicability check would resume at 20
    and warm-serve the PREVIOUS scene's thermal state; the
    geometry-history gate must refuse it (resume 0, cold split).
    """
    executor = build_warm_executor(tmp_path)
    total = len(MET_HOURS)
    edits: list[dict[str, Any]] = []
    edits.append(
        run_edit(executor, rad_met_edit(executor, row=1, revision=0,
                                        value=21.0, edit_id="g1"))
    )
    edits.append(
        run_edit(executor, rad_met_edit(executor, row=20, revision=1,
                                        value=29.0, edit_id="g2"))
    )
    edits.append(
        run_edit(executor, veg_edit(executor, revision=2, edit_id="g3"))
    )
    edits.append(
        run_edit(executor, rad_met_edit(executor, row=21, revision=3,
                                        value=30.0, edit_id="g4"))
    )
    anchors_before = edits[1]["anchors_after"]  # after m2: anchor@20 live
    anchors_final = edits[-1]["anchors_after"]
    anchor20_revisions = [
        a["revision"] for a in anchors_before
        if a["next_step"] == 20 and a["thermal"]
    ]
    return {
        "total_timesteps": total,
        "edits": edits,
        "summary": {
            "anchor20_present_before_geometry_edit": bool(anchor20_revisions),
            "anchor20_revision": (
                anchor20_revisions[0] if anchor20_revisions else None
            ),
            # rotation keeps the last 3 revisions: the anchor@20 must
            # still be ON DISK when the post-geometry met edit scans, or
            # the refusal would be trivial (absence, not the gate).
            "anchor20_survives_rotation": any(
                a["revision"] == anchor20_revisions[0]
                for a in anchors_final
            ) if anchor20_revisions else False,
            "final_met_edit": {
                "r0": edits[-1]["met_warm_r0"],
                "resume_step": edits[-1]["met_warm_resume_step"],
                "warm_hit": edits[-1]["warm_hit"],
                "refusal": edits[-1]["met_warm_refusal"],
            },
        },
    }


def write_measurement(record: dict[str, Any], path: Path | None = None):
    path = path or (ARTIFACT_DIR / "g21_warm_measurement.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # Distinct results roots: the sequences must not scan each
        # other's checkpoints (measurement contamination).
        record = measure_sequence(Path(td) / "main")
        geometry_record = measure_geometry_sequence(Path(td) / "geometry")
    out = write_measurement(record)
    geometry_out = write_measurement(
        geometry_record, ARTIFACT_DIR / "g21_geometry_witness.json"
    )
    print(json.dumps(record["summary"], indent=2))
    print(json.dumps(geometry_record["summary"], indent=2))
    print(f"written: {out}")
    print(f"written: {geometry_out}")
