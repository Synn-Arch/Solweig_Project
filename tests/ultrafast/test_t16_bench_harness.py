"""Honesty-scaffolding tests for the T16 CPU-lane bench harness.

The designated MUTATION DEMONSTRATIONS (each mutates a plausible harness
bug and must be caught by an assertion in ``benchmarks/ultrafast/
t16_cpu_lane.py`` / ``run.py``):

* **double-counted stage** — a stage table whose entries sum past the
  measured E2E (the classic overlap/double-count) must raise
  :class:`StageSumMismatch`, never publish an unexplainable number;
* **p95 over warmup-only rows** — summarizing before any sample rows
  exist must raise :class:`SamplePhaseError`, never emit a quiet 0.0;
* **truncated run** — fewer sample rows than the protocol requested must
  raise :class:`SampleCountError` rather than a confident-looking p95.

Plus the verdict vocabulary (PASS / TARGET_MISSED only for numeric rows,
BLOCKED authoritative verdicts, NOT_RUN rows keep their reason) and the
report-row assembly over synthetic artifacts. No benchmark executes here.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks" / "ultrafast"
for p in (str(BENCH_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

lane = importlib.import_module("t16_cpu_lane")
run_mod = importlib.import_module("run")


# ---------------------------------------------------------------------------
# designated mutation demonstrations
# ---------------------------------------------------------------------------
def test_mutation_double_counted_stage_is_caught():
    """MUTATION: 'solve' double-counted (sum of stages exceeds E2E).

    A harness that timed overlapping intervals (e.g. counting both the
    dispatch wrapper and the inner solve) produces a stage table that no
    longer explains the wall — the run must abort.
    """
    honest = {"accept_s": 0.05, "queue_wait_s": 0.02, "solve_s": 1.0,
              "publish_s": 0.01, "fetch_decode_apply_s": 0.02}
    e2e = sum(honest.values())
    lane.assert_exclusive_stage_sum(honest, e2e)  # honest table passes

    # exact-explanation regime: no tolerance to hide a mis-timed stage in
    exact = {"tol_rel": 0.0, "tol_abs_s": 0.0}
    double_counted = {**honest, "solve_s": honest["solve_s"] * 2.0}
    with pytest.raises(lane.StageSumMismatch):
        lane.assert_exclusive_stage_sum(double_counted, e2e, **exact)

    # omitted work (a stage silently dropped from the table) is equally
    # unexplainable — and aborts even at the DEFAULT tolerance here,
    # because the gap (0.5 s) exceeds max(2%, 50 ms)
    omitted = {**honest, "solve_s": 0.5}
    with pytest.raises(lane.StageSumMismatch):
        lane.assert_exclusive_stage_sum(omitted, e2e)


def test_mutation_p95_over_warmup_only_rows_is_caught():
    """MUTATION: percentile computed while only warmup rows exist.

    summarize_samples must refuse (a) zero sample rows outright and
    (b) percentile_checked over an empty list — the failure mode is a
    p95 of 0.0 'passing' a target, which must be impossible.
    """
    warmup_only = [{"phase": "warmup", "e2e_s": 0.5},
                   {"phase": "warmup", "e2e_s": 0.4}]
    with pytest.raises(lane.SamplePhaseError):
        lane.summarize_samples(warmup_only, "e2e_s", runs_requested=2)
    with pytest.raises(lane.SamplePhaseError):
        lane.percentile_checked([], 95)

    # warmup rows coexisting with samples must not contaminate the p95
    rows = warmup_only + [{"phase": "sample", "e2e_s": 10.0},
                          {"phase": "sample", "e2e_s": 20.0}]
    summary = lane.summarize_samples(rows, "e2e_s", runs_requested=2)
    assert summary["n"] == 2
    assert summary["min"] == 10.0  # the 0.4/0.5 warmup rows are excluded


def test_mutation_truncated_run_is_caught():
    """MUTATION: the run died at 5 of the requested 30 samples.

    A truncated run must not yield a summary at all (SampleCountError),
    so a partial p95 can never masquerade as the protocol p95.
    """
    rows = [{"phase": "sample", "e2e_s": float(i)} for i in range(5)]
    with pytest.raises(lane.SampleCountError):
        lane.summarize_samples(rows, "e2e_s", runs_requested=30)

    full = rows + [{"phase": "sample", "e2e_s": 5.0}] * 25
    summary = lane.summarize_samples(full, "e2e_s", runs_requested=30)
    assert summary["n"] == 30


def test_percentile_checked_known_values():
    vals = list(range(1, 101))  # 1..100
    # linear interpolation: pos = 99 * 0.95 = 94.05 -> 95.05
    assert lane.percentile_checked(vals, 95) == pytest.approx(95.05)
    assert lane.percentile_checked(vals, 50) == pytest.approx(50.5)
    assert lane.percentile_checked(vals, 0) == pytest.approx(1.0)
    assert lane.percentile_checked(vals, 100) == pytest.approx(100.0)
    assert lane.percentile_checked([7.0], 95) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# verdict vocabulary (run.py report rows)
# ---------------------------------------------------------------------------
def test_verdict_vocabulary_numeric_rows():
    # final is the release bar; inside-milestone-outside-final is still a miss
    assert run_mod._verdict_vs_targets(50.0, 1000.0, 100.0) == "PASS"
    assert run_mod._verdict_vs_targets(500.0, 1000.0, 100.0) == "TARGET_MISSED"
    assert run_mod._verdict_vs_targets(5000.0, 1000.0, 100.0) == "TARGET_MISSED"


def test_known_variants_cover_t16_cpu_lane():
    assert "baseline-legacy" in run_mod.KNOWN_VARIANTS
    for v in ("cpu-numba-selected-exact", "cpu-numba-full-day",
              "cpu-numba-full-recompute-warm"):
        assert v in run_mod.KNOWN_VARIANTS


def _write_t16_artifacts(root: Path) -> None:
    """Synthetic T16 artifact set shaped like the real outputs (values
    chosen so latency rows MISS and memory/startup rows PASS, exercising
    both branches of the verdict mapping)."""
    (root / "bench").mkdir(parents=True, exist_ok=True)
    (root / "bench" / "bench_cpu-numba-selected-exact.json").write_text(json.dumps({
        "e2e_s": {"n": 30, "p50": 35.0, "p95": 36.0, "min": 30.0, "max": 40.0},
        "stage_s": {"accept_s": {"p95": 0.5}, "queue_wait_s": {"p95": 0.5},
                    "solve_s": {"p95": 34.9}, "publish_s": {"p95": 0.01},
                    "fetch_decode_apply_s": {"p95": 0.01}},
        "sampling": {"repeats": 30, "warmup": 5, "pairs": 10},
        "sampling_compliance": {"repeats_requested": 30},
        "source_sha256": {"solweig_core/runtime.py": "a" * 64},
    }))
    (root / "bench" / "bench_cpu-numba-full-day.json").write_text(json.dumps({
        "warm": {"n": 100, "p50": 14.0, "p95": 15.0, "min": 13.0, "max": 16.0},
        "cold": {"n": 100, "p50": 15.0, "p95": 16.0, "min": 14.0, "max": 17.0},
        "stage_ablation_s": {"stages_s": {"static_load_s": 0.42,
                                          "bundle_load_s": 0.25,
                                          "met_recompute_s": 0.24,
                                          "fused_kernel_s": 14.4},
                             "pass_wall_s": 15.1},
        "source_sha256": {"solweig_core/runtime.py": "a" * 64,
                          "solweig_core/numba_cpu/full_solve.py": "b" * 64},
    }))
    (root / "bench" / "bench_cpu-numba-full-recompute-warm.json").write_text(json.dumps({
        "full_recompute_warm": {"n": 100, "p50": 15.0, "p95": 15.5,
                                "min": 14.0, "max": 16.0},
        "auxiliary_anchor_resume_suffix": {"n": 5, "p50": 8.0, "p95": 8.5,
                                           "min": 7.5, "max": 9.0},
        "source_sha256": {"solweig_core/numba_cpu/full_solve.py": "b" * 64},
    }))
    (root / "t16_memory.json").write_text(json.dumps({
        "steady": {"steady_rss_mib": 210.0, "verdict": "PASS"},
        "representative_job_peak": {"peak_rss_mib": 600.0, "verdict": "PASS"},
        "twenty_warm_scenarios": {"peak_rss_mb": 900.0, "verdict": "PASS"},
        "source_sha256": {"solweig_core/runtime.py": "a" * 64},
    }))
    (root / "t16_startup.json").write_text(json.dumps({
        "process_to_ready": {
            "warm_p2r_full_s": {"n": 6, "p95": 2.0},
            "cold_p2r_full_s": {"n": 6, "p95": 9.0},
            "verdict_warm_prebuilt_vs_3000ms": "PASS"},
        "unseen_signature": {"first_table_ms_p50": 300.0},
        "cold_first_edit": {"cold_first_edit_e2e_s": 90.0},
        "site_cache_build": {"capture_regeneration_s": 62.0},
        "source_sha256": {"solweig_core/runtime.py": "a" * 64},
    }))


def test_t16_rows_verdicts_and_disclosures(tmp_path):
    _write_t16_artifacts(tmp_path)
    rows, bottlenecks, unverified = run_mod._t16_rows(tmp_path)

    by_id = {r["row_id"]: r for r in rows}

    # measured latency rows MISS (dev-host) and carry the BLOCKED
    # authoritative verdict — the value itself is never massaged
    sel = by_id["cpu_selected_exact"]
    assert sel["status"] == "TARGET_MISSED"
    assert sel["measured_p95_ms"] == 36000.0
    assert sel["authoritative_four_core_verdict"] == "BLOCKED"
    assert "four-physical-core" in sel["blocked_reason"]
    fday = by_id["cpu_full_day_local"]
    assert fday["status"] == "TARGET_MISSED"
    assert fday["measured_p95_ms"] == 16000.0  # max(cold 16s, warm 15s)
    assert by_id["cpu_full_recompute_warm"]["status"] == "TARGET_MISSED"
    assert by_id["cpu_full_recompute_warm"]["measured_p95_ms"] == 15500.0

    # memory/startup PASS on their own verdicts but stay BLOCKED
    # authoritative (four-core host unavailable)
    for rid in ("cpu_aot_single_site_single_scenario_steady_rss_max",
                "cpu_aot_representative_job_peak_rss_max",
                "cpu_aot_twenty_warm_scenarios_process_peak_rss_max",
                "cpu_aot_prebuilt_site_process_to_ready_p95_max"):
        assert by_id[rid]["status"] == "PASS", rid
        assert by_id[rid]["authoritative_four_core_verdict"] == "BLOCKED"

    # NOT_RUN rows keep their reasons
    assert by_id["ux_local_geometry_feedback"]["status"] == "NOT_RUN"
    assert "browser" in by_id["ux_local_geometry_feedback"]["reason"]
    for rid in ("cuda_selected_exact", "cuda_full_day_local",
                "cuda_full_recompute_warm"):
        assert by_id[rid]["status"] == "NOT_RUN", rid

    # parity contracts: cpu row cites gate evidence, cuda rows NOT_RUN
    assert by_id["parity:cpu_candidate_vs_pinned_cpu_full_oracle"]["status"] == "PASS"
    assert "test_full_solve.py" in str(
        by_id["parity:cpu_candidate_vs_pinned_cpu_full_oracle"]["evidence"])
    assert by_id["parity:cuda_canonical_vs_cpu_canonical"]["status"] == "NOT_RUN"
    assert by_id["parity:cuda_legacy_vs_pinned_cuda_full_oracle"]["status"] == "NOT_RUN"

    # every TARGET_MISSED latency row has a bottleneck entry with
    # exclusive stages and a next-candidate from T17
    miss_targets = {b["for_target"] for b in bottlenecks}
    assert {"cpu_selected_exact", "cpu_full_day_local",
            "cpu_full_recompute_warm"} <= miss_targets
    for b in bottlenecks:
        assert b["dominant_stage"]
        assert all("T17" in c for c in b["next_candidates"])
        stages = b.get("exclusive_stage_p95_s") or b.get("exclusive_stage_s")
        assert stages and isinstance(stages, dict)

    # the sample-count caveat for the paired_expensive regime surfaces
    assert any("paired_expensive" in u for u in unverified)
    assert any("four-core" in u for u in unverified)

    allowed = {"PASS", "FAIL", "BLOCKED", "TARGET_MISSED", "NOT_RUN"}
    for r in rows:
        assert r.get("status") in allowed, r


def test_t16_rows_with_missing_artifacts_mark_not_run(tmp_path):
    rows, _, _ = run_mod._t16_rows(tmp_path)
    by_id = {r["row_id"]: r for r in rows}
    for rid in ("cpu_selected_exact", "cpu_full_day_local",
                "cpu_full_recompute_warm"):
        assert by_id[rid]["status"] == "NOT_RUN", rid


def test_report_writes_t16_section(tmp_path, capsys):
    _write_t16_artifacts(tmp_path)
    rc = run_mod.main(["report", "--artifacts", str(tmp_path)])
    assert rc == 0
    payload = json.loads((tmp_path / "report.json").read_text())
    md = (tmp_path / "report.md").read_text()

    section = payload["t16_cpu_lane"]
    assert section["rows"] and section["bottlenecks_for_target_missed"]
    assert section["certified_source_sha256"]["solweig_core/runtime.py"] == "a" * 64
    assert section["certified_source_sha256"][
        "solweig_core/numba_cpu/full_solve.py"] == "b" * 64
    assert "hash_conflicts" not in section  # artifacts agree per file

    for needle in ("T16 CPU lane", "ux_local_geometry_feedback",
                   "TARGET_MISSED", "Unverified cases", "BLOCKED"):
        assert needle in md, needle
    # markdown table row for the selected-exact miss carries the number
    assert "cpu_selected_exact" in md and "36000" in md


def test_report_t16_hash_conflict_is_flagged(tmp_path):
    _write_t16_artifacts(tmp_path)
    bench = tmp_path / "bench" / "bench_cpu-numba-full-day.json"
    data = json.loads(bench.read_text())
    data["source_sha256"]["solweig_core/runtime.py"] = "c" * 64  # diverged
    bench.write_text(json.dumps(data))
    meta = run_mod._t16_sources_and_compliance(tmp_path)
    assert meta["hash_conflicts"]["solweig_core/runtime.py"] == [
        "a" * 64, "c" * 64]
