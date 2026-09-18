# SPDX-License-Identifier: GPL-3.0-only
"""G2.2 gate: the crash-restart proof harness (R5 design brief, final step).

The harness (docs/incremental_design_tool/realtime_collaboration/design/
proofs/g22_crash_restart.py) kills a REAL executor subprocess at every
durable seam of the commit path (mid-solve, pre-publish, post-publish/
pre-checkpoint, mid-checkpoint-write -- staged ``rev-*.tmp`` pre-rename
and same-revision rmtree pre-rename -- and post-checkpoint), then
performs the restart exactly as a fresh process would (fresh executor +
``restore_scenario_state`` from the newest snapshot + replay of the op
log; the op log is truth, checkpoints are reproduce aids) and proves:

1. zero accepted ops lost (final revision + per-op routing reproduce the
   uninterrupted oracle exactly);
2. revision monotonicity (every snapshot's claimed revision has its
   published patch bytes on disk, checksum-loadable);
3. no torn checkpoints survive (every listed record loads; a kill in the
   write window leaves old/new/neither, never a hybrid; byte tampering is
   rejected with the typed CheckpointError and falls back to the previous
   valid record);
4. restart determinism (the restarted scenario's store-composed served
   planes are BITWISE the uninterrupted run's, float32, NaN-aware).

Sweep profiles: smoke/suite/thorough (pytest default smoke; the
``G22_SWEEP_SCALE`` environment variable overrides, mirroring the V4/G1
gates). The smoke profile is the eight mandated kill points plus the
corruption case.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_G22_HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs" / "incremental_design_tool" / "realtime_collaboration"
    / "design" / "proofs" / "g22_crash_restart.py"
)

#: The mandated smoke kill points (R5 design brief: mid-batch,
#: post-accept/pre-publish, post-publish/pre-checkpoint, mid-checkpoint
#: write, post-checkpoint, inside the warm consumer's split, plus the
#: two-crash chain for the same-revision rmtree window).
MANDATED_KILL_CASES = {
    "op0_solve_mid_series",
    "op0_pre_publish_cold_restart",
    "op2_warm_split_suffix_entry",
    "op2_pre_publish",
    "op2_post_publish_pre_checkpoint",
    "op2_checkpoint_staged_pre_rename",
    "op2_checkpoint_final_removed_pre_rename",
    "op4_post_checkpoint",
}


def _load_g22_harness():
    """Load the proof harness by path (it is a CLI tool, not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "g22_crash_restart", _G22_HARNESS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses resolve cls.__module__ through sys.modules
    # BEFORE exec_module finishes -- register first or import crashes.
    sys.modules["g22_crash_restart"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def g22():
    return _load_g22_harness()


@pytest.fixture(scope="module")
def g22_sweep(g22):
    """One sweep per test module run (the profile's cases, shared)."""
    profile = os.environ.get("G22_SWEEP_SCALE", "smoke")
    return g22.run_sweep(profile)


class TestG22CrashRestart:
    def test_harness_loads_with_pinned_scope(self, g22) -> None:
        """The harness is a first-class module: its sweep surface (profiles,
        the mandated kill points, the kill markers) is pinned so a rename
        or an accidental scope shrink fails HERE, not silently."""
        assert set(g22.PROFILES) == {"smoke", "suite", "thorough"}
        smoke_names = {case["name"] for case in g22.PROFILES["smoke"]["cases"]}
        assert MANDATED_KILL_CASES <= smoke_names, (
            "the smoke profile no longer covers every mandated kill point"
        )
        assert g22.KILL_EXIT == 70 and g22.NOOP_EXIT == 71

    def test_every_kill_point_restarts_bitwise(self, g22, g22_sweep) -> None:
        """THE property. Every kill case: the kill FIRED at its seam, the
        disk it left behind satisfies the no-torn-checkpoint and revision
        monotonicity inspections, and the restart replayed the op log to
        the oracle's exact routing with BITWISE-equal served planes. Any
        defect (torn survivor, lost op, routing divergence, serve
        mismatch) fails the sweep."""
        totals = g22_sweep["totals"]
        assert totals["cases"] > 0, "sweep ran zero cases"
        assert totals["failed"] == 0, (
            f"crash-restart property violated: {totals['defects'][:3]}"
        )
        assert totals["defects"] == []
        assert not totals["vacuous"], totals["vacuity_reasons"]

        cases = {case["name"]: case for case in g22_sweep["cases"]}
        for name in MANDATED_KILL_CASES:
            case = cases[name]
            assert case["verdict"] == "PASS", (name, case.get("defect"))
            assert case["kill_fired"] is True, (name, "the kill never fired")
            restart = case["restart"]
            assert restart["ops_replayed"] >= 1, name
            assert restart["final_revision"] == 5, (name, restart)
            assert restart["cells_bitwise_compared"] > 0, name
        # The staged-pre-rename kill leaves the staging dir invisible to
        # listings, and the rmtree-window chain leaves NEITHER old nor new
        # rev-000003 -- both pinned per-kill-point, not just on average.
        staged = cases["op2_checkpoint_staged_pre_rename"]["disk"]
        assert any(name.endswith(".tmp") for name in staged["staging_tmp"]), staged
        removed = cases["op2_checkpoint_final_removed_pre_rename"]["disk"]
        assert 3 not in [c["revision"] for c in removed["checkpoints"]], (
            "the rmtree-window kill must leave rev-000003 absent (neither "
            "old nor new), never a hybrid"
        )
        assert cases["op2_checkpoint_final_removed_pre_rename"]["kill2_fired"] is True

    def test_sweep_non_vacuity(self, g22, g22_sweep) -> None:
        """The property is only PROVEN if the sweep actually exercised the
        regime it guards: at least one restart WARMED from a checkpoint
        that only exists on disk, at least one mid-solve kill confirmed
        inside the physics series, the staged ``.tmp`` window observed at
        a checkpoint kill, a staged patch directory observed at a
        pre-publish kill, and both torn-checkpoint rejections recorded."""
        totals = g22_sweep["totals"]
        assert totals["warm_after_restart_cases"] >= 1, (
            "no restart ever consumed a disk checkpoint warm: the warm "
            "resume was never exercised across a process death"
        )
        assert totals["torn_rejections"] >= 2
        assert totals["staged_tmp_observed"] >= 1
        assert totals["dot_staging_observed"] >= 1
        assert totals["mid_series_confirmed"] >= 1
        assert totals["cells_bitwise_compared"] > 0

    def test_corruption_rejects_and_falls_back(self, g22, g22_sweep) -> None:
        """Property 3's checksum half: a flipped tensor byte and a truncated
        ``checkpoint.json`` are both rejected with the typed
        ``CheckpointError``; ``load_latest_checkpoint`` falls back to the
        previous valid record; and the restart after the tear still
        reproduces the oracle's planes BITWISE (on the cold split -- the
        design-mandated fallback -- which the G2.1 parity fence already
        proved bitwise-equal to the warm path)."""
        cases = {case["name"]: case for case in g22_sweep["cases"]}
        assert "corruption_torn_newest" in cases, (
            "corruption case missing from the sweep evidence"
        )
        corruption = cases["corruption_torn_newest"]
        assert corruption["verdict"] == "PASS"
        assert corruption["tensor_reject"], "flipped tensor byte was not rejected"
        assert corruption["metadata_reject"], "truncated metadata was not rejected"
        restart = corruption["restart"]
        assert restart["ops_replayed"] == 3, restart
        assert restart["cells_bitwise_compared"] > 0
        assert restart["final_revision"] == 5, restart
