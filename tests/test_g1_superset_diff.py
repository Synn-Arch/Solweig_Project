# SPDX-License-Identifier: GPL-3.0-only
"""G1.0 gate: the window-planner superset differential (R5 design, G1).

The harness (docs/incremental_design_tool/realtime_collaboration/design/
proofs/g1_superset_diff.py) drives, per seeded case, a REAL PlanExecutor
over a real prepared site through:

1. a seeding batch of tall vegetation edits whose planned influence union
   routes FULL — a full-tile publication at scene revision N (the store
   records tile-wide entries for ``utci``/``tmrt``/``time_shadow``);
2. k probe batches of small vegetation edits (seeded add / move / resize /
   delete / multi-op) that route WINDOWED — a revision-N+1 publication
   whose write windows supersede exactly the cells they repaint, R5a
   retention keeping the rev-N cell remainders;
3. the differential: the COMPOSED STORE SERVE (newest-covering-entry per
   cell, sliced the way the executor's own plane assembly slices, payloads
   checksum-verified through ``load_patch``) vs an INDEPENDENT full-tile
   recompute (``run_full_tile`` on the post-batch scene, fresh forcing and
   scratch) — BITWISE (``np.array_equal(..., equal_nan=True)``, float32)
   and TILE-WIDE, not just inside the batch windows.

That is the window-planner superset property G1.1's gate relaxation is
admissible ONLY under: every cell whose recomputed value changed between
scene N and scene N+1 lies inside the revision-N+1 write windows, so cells
outside them provably hold their rev-N values and painting entries in
ascending-revision order reconstructs the recompute exactly. Any mismatch
OUTSIDE the batch windows is a planner undershoot — a correctness defect
to fix in the window derivation, never by widening the serve gate.

Sweep profiles: smoke/suite/thorough (pytest default smoke; the
``G1_SWEEP_SCALE`` environment variable overrides, mirroring the V4
gate's ``R4B_SWEEP_SCALE``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_G1_HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs" / "incremental_design_tool" / "realtime_collaboration"
    / "design" / "proofs" / "g1_superset_diff.py"
)


def _load_g1_harness():
    """Load the proof harness by path (it is a CLI tool, not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "g1_superset_diff", _G1_HARNESS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses resolve cls.__module__ through sys.modules
    # BEFORE exec_module finishes -- register first or import crashes.
    sys.modules["g1_superset_diff"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def g1():
    return _load_g1_harness()


@pytest.fixture(scope="module")
def g1_sweep(g1):
    """One sweep per test module run (the profile's cases, shared)."""
    profile = os.environ.get("G1_SWEEP_SCALE", "smoke")
    return g1.run_sweep(profile)


class TestG1SupersetDiff:
    def test_harness_loads_with_pinned_scope(self, g1) -> None:
        """The harness is a first-class module: its public sweep surface
        (profiles, sites, defect classification) is pinned so a rename or
        an accidental scope shrink fails HERE, not silently downstream."""
        assert set(g1.PROFILES) == {"smoke", "suite", "thorough"}
        assert len(g1.SITES) >= 3, "the sweep must cover at least 3 site families"
        assert g1.UNDERSHOOT == "undershoot"

    def test_composed_serve_bitwise_full_recompute_tile_wide(
        self, g1, g1_sweep
    ) -> None:
        """THE property. Every case's every step: composed serve == the
        independent full-tile recompute, bitwise, tile-wide, float32,
        NaN-aware. Zero defects of any class (an undershoot, an in-window
        mismatch, or a calibration failure all fail the sweep)."""
        totals = g1_sweep["totals"]
        assert totals["cases"] > 0, "sweep ran zero cases"
        assert totals["defects"] == [], (
            f"superset property violated: {totals['defects'][:3]}"
        )

    def test_sweep_non_vacuity(self, g1, g1_sweep) -> None:
        """The property is only PROVEN if the sweep actually exercised the
        regime it guards: at least one WINDOWED probe batch (strict
        sub-windows of the tile), FULL tile-wide coverage after retention
        on every compared (node, t), and at least one composed serve whose
        provenance is genuinely MIXED (rev-N remainders surviving beside
        rev-N+1 windows). A sweep where every probe routed full-tile
        proves nothing about retention."""
        totals = g1_sweep["totals"]
        assert totals["windowed_probes"] > 0, (
            "no probe batch routed WINDOWED: the superset differential "
            "never left the trivial full-recompute regime"
        )
        assert totals["mixed_serves"] > 0, (
            "no composed serve carried mixed provenance {N, N+1}: R5a "
            "retention was never exercised"
        )
        assert totals["full_coverage_serves"] == totals["compared_serves"], (
            "a compared (node, t) was not FULL after retention"
        )
        assert totals["cells_bitwise_compared"] > 0
