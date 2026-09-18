import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs" / "incremental_design_tool"
DEMO_ROOT = ROOT / "studio"


def test_required_design_documents_exist() -> None:
    required = {
        "index.md",
        "architecture.md",
        "data_model.md",
        "incremental_algorithm.md",
        "cpu_optimization.md",
        "api_contract.md",
        "frontend_spec.md",
        "testing_validation.md",
        "benchmarks/packed_visibility.md",
        "benchmarks/2026-08-31-packed-visibility.json",
        "validation/scaffolding_validation.md",
        "deployment_operations.md",
        "agent_execution_plan.md",
        "agent/master_prompt.md",
        "agent/mission.yaml",
        "agent/runbook.md",
        "agent/validation_pipeline.md",
        "agent/recovery_and_continuity.md",
        "agent/pipeline.md",
        "agent/worklog_template.md",
        "agent/handoff_template.md",
        "adr/0001-incremental-cpu-worker.md",
        "adr/0002-preview-and-exact-results.md",
        "adr/0003-fixed-wind-scope.md",
    }
    missing = sorted(path for path in required if not (DOC_ROOT / path).is_file())
    assert not missing, f"missing design documents: {missing}"


def test_sphinx_root_links_incremental_design_tool() -> None:
    index = (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    assert "incremental_design_tool/index" in index


def test_agent_plan_contains_phase_gates_and_commands() -> None:
    plan = (DOC_ROOT / "agent_execution_plan.md").read_text(encoding="utf-8")
    for marker in [
        "Phase 0",
        "Phase 5",
        "Phase 9",
        "Exit criteria",
        "Commands to run before every merge",
        "Stop conditions",
    ]:
        assert marker in plan


def test_frontend_is_dependency_free_and_complete() -> None:
    required = {
        "index.html",
        "styles.css",
        "app.mjs",
        "model.mjs",
        "renderer.mjs",
        "solver_kernel.mjs",
        "solver_worker.mjs",
        "serve.py",
        "assets/site_base.webp",
        "assets/baseline.json",
        "tests/model.test.mjs",
        "tests/solver_kernel.test.mjs",
    }
    missing = sorted(path for path in required if not (DEMO_ROOT / path).is_file())
    assert not missing, f"missing frontend files: {missing}"

    html = (DEMO_ROOT / "index.html").read_text(encoding="utf-8")
    assert "<script type=\"module\" src=\"./app.mjs\"></script>" in html
    assert "https://" not in html
    assert "http://" not in html


def test_every_app_query_selector_id_exists_in_html() -> None:
    html = (DEMO_ROOT / "index.html").read_text(encoding="utf-8")
    app = (DEMO_ROOT / "app.mjs").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([A-Za-z][A-Za-z0-9_-]*)"', html))
    selected = set(re.findall(r'document\.querySelector\("#([A-Za-z][A-Za-z0-9_-]*)"\)', app))
    assert selected <= ids, f"selectors without matching HTML IDs: {sorted(selected - ids)}"


def test_browser_fixture_shape_and_provenance() -> None:
    fixture = json.loads((DEMO_ROOT / "assets" / "baseline.json").read_text(encoding="utf-8"))
    analysis = fixture["analysis"]
    expected = analysis["gridWidth"] * analysis["gridHeight"]
    assert len(analysis["utci"]) == expected
    assert len(analysis["buildingMask"]) == expected
    assert fixture["provenance"]["scientificUse"] is False
    assert fixture["scene"]["widthMeters"] == 1000
    assert fixture["scene"]["heightMeters"] == 1000


def test_documentation_preview_asset_is_nonempty() -> None:
    preview = ROOT / "docs" / "_static" / "incremental-design-tool-preview.webp"
    assert preview.is_file()
    assert preview.stat().st_size > 50_000


def test_incremental_document_internal_links_resolve() -> None:
    markdown_link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    broken: list[tuple[str, str]] = []
    for document in DOC_ROOT.rglob("*.md"):
        for target in markdown_link.findall(document.read_text(encoding="utf-8")):
            path_target = target.split("#", 1)[0]
            if not path_target or "://" in path_target or path_target.startswith("mailto:"):
                continue
            if not (document.parent / path_target).resolve().exists():
                broken.append((str(document.relative_to(DOC_ROOT)), target))
    assert not broken, f"broken internal document links: {broken}"


def test_frontend_html_has_unique_ids_and_resolvable_local_assets() -> None:
    html_path = DEMO_ROOT / "index.html"
    html = html_path.read_text(encoding="utf-8")
    ids = re.findall(r'id="([A-Za-z][A-Za-z0-9_-]*)"', html)
    assert len(ids) == len(set(ids)), "frontend HTML contains duplicate IDs"

    local_targets = re.findall(r'(?:src|href)="(\./[^"?#]+|\.\./\.\./[^"?#]+)"', html)
    missing = [target for target in local_targets if not (html_path.parent / target).resolve().exists()]
    assert not missing, f"frontend references missing local assets: {missing}"


def test_agent_master_prompt_defines_autonomous_loop_and_hard_stops() -> None:
    prompt = (DOC_ROOT / "agent" / "master_prompt.md").read_text(encoding="utf-8")
    for marker in [
        "EXECUTION LOOP",
        "VALIDATION TIERS",
        "HARD STOP CONDITIONS",
        "RECOVERY",
        "FINAL ACCEPTANCE",
        "WORKLOG.md",
        "scene_revision",
    ]:
        assert marker in prompt


def test_agent_mission_declares_all_phases_and_validation_tiers() -> None:
    mission = (DOC_ROOT / "agent" / "mission.yaml").read_text(encoding="utf-8")
    for phase in [f"id: P{i}" for i in range(10)]:
        assert phase in mission
    for tier in [f"T{i}:" for i in range(7)]:
        assert tier in mission
    assert "existing_full_domain_solweig" in mission
    assert "single_cpu_node" in mission


def test_recovery_doc_has_restart_and_patch_fallback_procedure() -> None:
    recovery = (DOC_ROOT / "agent" / "recovery_and_continuity.md").read_text(encoding="utf-8")
    for marker in [
        "Crash recovery decision tree",
        "git format-patch",
        "scene revision",
        "Atomic artifact pattern",
        "Retry policy",
    ]:
        assert marker in recovery


def test_validation_pipeline_covers_scientific_and_performance_evidence() -> None:
    validation = (DOC_ROOT / "agent" / "validation_pipeline.md").read_text(encoding="utf-8")
    for marker in [
        "scientific differential",
        "max absolute error",
        "boundary ring",
        "outside write window",
        "peak_rss_mb",
        "full repository",
        "deployment and interaction smoke",
    ]:
        assert marker.lower() in validation.lower()
