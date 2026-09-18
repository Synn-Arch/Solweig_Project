from __future__ import annotations

from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "incremental_design_tool"
REALTIME = DOC / "realtime_collaboration"
AGENT = DOC / "agent"


def test_realtime_document_set_exists() -> None:
    required = {
        "index.md",
        "service_level_contract.md",
        "collaborative_state.md",
        "epoch_scheduler.md",
        "compute_architecture.md",
        "compensation_and_exactness.md",
        "optimization_roadmap.md",
        "validation_plan.md",
        "realtime_contract.yaml",
    }
    assert {path.name for path in REALTIME.iterdir() if path.is_file()} >= required
    assert (DOC / "adr" / "0004-collaborative-dual-rate-realtime.md").is_file()


def test_goal_condition_fits_agent_limit_and_is_universal() -> None:
    goal = (AGENT / "goal_condition_realtime_collab.txt").read_text(encoding="utf-8")
    assert len(goal) <= 4000
    for phrase in (
        "all enabled building, vegetation, land-cover",
        "100 ms micro-epochs",
        "p99 <=1,000 ms",
        "fast_qualified",
        "exact latest-state revision-chasing queue",
        "persistent object-to-ray",
        "Never omit an accepted edit",
    ):
        assert phrase in goal


def test_machine_contract_has_lossless_dual_rate_invariants() -> None:
    contract = yaml.safe_load((REALTIME / "realtime_contract.yaml").read_text())
    assert contract["scheduler"]["micro_epoch_ms"] == 100
    assert contract["slo_targets"]["fast_revision_p99_ms"] == 1000
    assert contract["slo_targets"]["accepted_operation_inclusion_percent"] == 100
    assert contract["scheduler"]["one_fifo_job_per_raw_edit"] == "forbidden"
    assert "fast_qualified" in contract["result_classes"]
    assert "visual_pending" in contract["result_classes"]
    assert len(contract["required_optimization_waves"]) == 9


def test_agent_mission_requires_independent_validation_and_admission() -> None:
    mission = yaml.safe_load((AGENT / "mission_realtime_collab.yaml").read_text())
    assert mission["multi_agent"]["required"] is True
    assert mission["multi_agent"]["scientific_change_requires_non_author_review_and_differential"] is True
    assert mission["multi_agent"]["performance_claim_requires_independent_reproduction"] is True
    assert mission["runtime"]["accepted_operations_may_be_dropped"] is False
    assert mission["runtime"]["exact_lane_may_block_fast_lane"] is False
    assert mission["slo_targets"]["qualified_only_inside_measured_admission_envelope"] is True


def test_docs_do_not_claim_current_exact_one_second_geometry() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in REALTIME.glob("*.md"))
    assert "claiming arbitrary exact geometry recomputation within one second would be unsupported" in combined
    assert "approximately 37.5-39 seconds" in combined
    assert "visual_pending" in combined


def test_universal_index_links_realtime_extension() -> None:
    index = (DOC / "universal_editing" / "index.md").read_text(encoding="utf-8")
    pipeline = (DOC / "universal_editing" / "execution_pipeline.md").read_text(encoding="utf-8")
    assert "../realtime_collaboration/index.md" in index
    assert "100 ms per-workspace epochs" in pipeline
    assert "supersession removes obsolete compute targets, never accepted operations" in pipeline
