# SPDX-License-Identifier: GPL-3.0-only
"""T10 pinning test: the measured G2.1 current-state answer.

DESIGN 11.1: "first verify with metrics/tests whether the current caller
actually hits the warm route — the existence of a coverage-only checkpoint
must not be interpreted as completion of thermal reuse." This file pins
the measurement produced by ``tests/ultrafast/g21_measurement.py`` (real
executor, prepared site, real SVF, canonical edit sequence):

  * WARM HIT REAL: a late met edit (row 21) after an r0 anchor (state@20)
    resumes at k=20 and solves 4 of 24 steps — a REAL steps_solved
    reduction through a THERMAL checkpoint (not a coverage-only marker).
  * WARM HITS ARE NARROW: the seed edit refuses on prefix coverage; the
    first in-day edit finds only the state@T anchor, which can NEVER
    warm-serve (its met prefix covers any edited row, r0 < T); an early
    edit invalidates every anchor prefix. Warm hits exist ONLY through
    the warm path's own r0 re-anchoring chain.
  * GEOMETRY EDITS NEVER WARM: a vegetation-geometry edit engages no warm
    route at all (structural eligibility), and the full-solve anchor it
    publishes belongs to a different geometry history (fingerprint-separated
    in T10's classifier; here pinned as: no met_warm diagnostics).

This test may import the torch-side executor (measurement harness, not a
runtime module; T08/T09 test-file precedent for torch in tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.ultrafast.g21_measurement import (
    ARTIFACT_DIR,
    MET_HOURS,
    measure_geometry_sequence,
    measure_sequence,
    write_measurement,
)

pytestmark = pytest.mark.integration


def test_g21_warm_route_actually_hits_and_reduces_steps(tmp_path):
    record = measure_sequence(tmp_path)
    total = record["total_timesteps"]
    edits = {e["edit_id"]: e for e in record["edits"]}

    # The pinned current-state answer (any change to the warm route's
    # reachability must change this test DELIBERATELY, not silently).
    hits = record["summary"]["warm_hit_edit_ids"]
    assert hits == ["m3"], (
        f"warm-hit set changed: {hits} — the G2.1 measurement answer "
        "moved; re-measure and re-pin"
    )
    m3 = edits["m3"]
    assert m3["met_warm_path"] is True
    assert m3["met_warm_resume_step"] == 20
    assert m3["met_warm_steps_solved"] == total - 20
    assert m3["met_warm_checkpoint_revision"] == 2
    assert m3["met_warm_refusal"] is None

    # The anchor chain: m2's cold split is what created the state@20
    # anchor m3 warmed from (a THERMAL checkpoint, fingerprinted).
    m2 = edits["m2"]
    assert m2["met_warm_path"] is True
    assert m2["met_warm_resume_step"] == 0
    assert m2["met_warm_steps_solved"] == total  # cold split = full count
    anchors_m2 = {a["revision"]: a for a in m2["anchors_after"]}
    assert anchors_m2[2]["thermal"] and anchors_m2[2]["next_step"] == 20
    assert anchors_m2[2]["fingerprinted"]
    assert set(anchors_m2[2]["scalars"]) >= {
        "CI", "firstdaytime", "timeadd", "Twater"
    }

    # The seed edit cannot warm: a fresh store serves no prefix row.
    m1 = edits["m1"]
    assert not m1["warm_hit"]
    assert "coverage" in (m1["met_warm_path_refusal"] or "")

    # An early edit invalidates every anchor's met prefix: cold split.
    m4 = edits["m4"]
    assert m4["met_warm_path"] is True
    assert m4["met_warm_resume_step"] == 0

    # A geometry edit engages NO warm route (no met_warm diagnostics at
    # all) — geometry-history invalidation is structural today.
    v1 = edits["v1"]
    assert v1["adapter"] == "vegetation_geometry"
    assert v1["met_warm_path"] is None
    assert not v1["warm_hit"]

    # steps_solved reduction is real: 24 -> 4 on the warm hit.
    assert record["summary"]["best_steps_solved"] == 4
    assert total == 24 == len(MET_HOURS)

    # Record the measurement under the artifacts root when it exists.
    if ARTIFACT_DIR.is_dir():
        path = write_measurement(record)
        data = json.loads(path.read_text())
        assert data["summary"] == record["summary"]
        assert Path(path).is_file()


def test_g21_geometry_edit_never_reuses_previous_scene_anchor(tmp_path):
    """Live R1 witness: a met edit after a GEOMETRY edit refuses the
    previous scene's anchor even though its clock position and met
    prefix both fit (DESIGN 11.2: a noon checkpoint of the previous
    scene is invalid because "the time matches" says nothing)."""
    record = measure_geometry_sequence(tmp_path)
    summary = record["summary"]

    # The trap is armed: anchor@20 (geometry A) exists and survives the
    # geometry edit's rotation; its met prefix (rows 0..19) is unchanged
    # by the row-21 edit — a met-only applicability check WOULD resume
    # at 20 and warm-serve the previous scene's thermal state.
    assert summary["anchor20_present_before_geometry_edit"] is True
    assert summary["anchor20_revision"] == 2
    assert summary["anchor20_survives_rotation"] is True

    # ...and is refused: the final met edit replays the prefix cold.
    final = summary["final_met_edit"]
    assert final["r0"] == 21
    assert final["resume_step"] == 0
    assert final["warm_hit"] is False
    assert final["refusal"] is not None

    if ARTIFACT_DIR.is_dir():
        path = write_measurement(
            record, ARTIFACT_DIR / "g21_geometry_witness.json"
        )
        assert json.loads(path.read_text())["summary"] == summary
