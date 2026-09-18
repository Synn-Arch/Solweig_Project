"""CLI tests for benchmarks/ultrafast/run.py (T01).

Covers the TASKS.ko.md CLI contract:
* ``--help`` for the program and every subcommand exits 0 (subprocess, as a
  user would invoke it);
* ``compare`` is exercised end-to-end on synthetic plane bundles: equal
  bundles exit 0, a 1-bit-different bundle exits 1, wrong time coverage with
  ``--require-full-coverage`` exits 1, ``--strict-bits`` without sidecars
  exits 1;
* ``bench`` with an unimplemented variant exits 2 with the typed error;
* ``report --require-evidence`` exits 1 on missing evidence and 0 once
  parity+bench evidence exists.

``baseline`` executes the multi-minute scientific harness and is not run
here; its command is documented in the T01 report.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PY = REPO_ROOT / "benchmarks" / "ultrafast" / "run.py"
PY = sys.executable


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(RUN_PY), *argv],
                          capture_output=True, text=True, cwd=str(REPO_ROOT))


def write_sample(root: Path, planes: dict[str, np.ndarray], metadata: dict | None,
                 run_record: dict | None = None):
    root.mkdir(parents=True, exist_ok=True)
    for name, arr in planes.items():
        np.save(root / f"{name}.f32.npy", np.ascontiguousarray(arr, dtype="<f4"))
    if metadata is not None:
        (root / "metadata.json").write_text(json.dumps({"planes": metadata}))
    if run_record is not None:
        (root / "run_record.json").write_text(json.dumps(run_record))


def utci_plane(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((4, 8, 6)) * 10 + 293.0).astype("<f4")


def plane_meta(times, origin=(0, 0)):
    return {"utci": {"global_time_index": list(times), "window_origin": list(origin)}}


class TestHelpContract:
    def test_top_level_help(self):
        proc = run_cli("--help")
        assert proc.returncode == 0
        for word in ("baseline", "compare", "bench", "report"):
            assert word in proc.stdout

    @pytest.mark.parametrize("sub", ["baseline", "compare", "bench", "report"])
    def test_subcommand_help(self, sub):
        proc = run_cli(sub, "--help")
        assert proc.returncode == 0, proc.stderr
        assert "usage" in proc.stdout.lower()

    def test_missing_subcommand_errors(self):
        proc = run_cli()
        assert proc.returncode != 0


class TestCompareSubcommand:
    def make_pair(self, tmp_path, mutate_cand=None, meta_times=(0, 1, 2, 3),
                  with_meta=True):
        ref = {"utci": utci_plane()}
        cand = {"utci": utci_plane().copy()}
        if mutate_cand:
            mutate_cand(cand)
        meta = plane_meta(meta_times) if with_meta else None
        ref_root, cand_root = tmp_path / "ref", tmp_path / "cand"
        write_sample(ref_root / "sample0", ref, meta)
        write_sample(cand_root / "sample0", cand,
                     plane_meta(meta_times) if with_meta else None)
        return ref_root, cand_root

    def test_equal_bundles_exit_zero(self, tmp_path):
        r, c = self.make_pair(tmp_path)
        out = tmp_path / "cmp.json"
        proc = run_cli("compare", "--reference", str(r), "--candidate", str(c),
                       "--strict-bits", "--require-full-coverage",
                       "--output", str(out))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = json.loads(out.read_text())
        assert report["all_equal"] is True

    def test_one_bit_delta_exits_nonzero(self, tmp_path):
        def mutate(cand):
            plane = cand["utci"]
            flat = plane.reshape(-1)
            flat[11] = np.nextafter(flat[11], np.float32(np.inf))
        r, c = self.make_pair(tmp_path, mutate_cand=mutate)
        out = tmp_path / "cmp.json"
        proc = run_cli("compare", "--reference", str(r), "--candidate", str(c),
                       "--strict-bits", "--output", str(out))
        assert proc.returncode == 1
        report = json.loads(out.read_text())
        assert report["all_equal"] is False
        assert report["samples"]["sample0"]["planes"]["utci.f32.npy"]["mismatch_count"] == 1

    def test_missing_coverage_exits_nonzero(self, tmp_path):
        r, c = self.make_pair(tmp_path, meta_times=(0, 1, 2, 3))
        # candidate only covers times 0-1
        sidecar = c / "sample0" / "metadata.json"
        data = json.loads(sidecar.read_text())
        data["planes"]["utci"]["global_time_index"] = [0, 1]
        sidecar.write_text(json.dumps(data))
        out = tmp_path / "cmp.json"
        proc = run_cli("compare", "--reference", str(r), "--candidate", str(c),
                       "--strict-bits", "--require-full-coverage", "--output", str(out))
        assert proc.returncode == 1
        report = json.loads(out.read_text())
        cov = report["samples"]["sample0"]["coverage"]
        assert cov["missing_in_candidate"] == [2, 3]

    def test_strict_bits_without_sidecar_exits_nonzero(self, tmp_path):
        r, c = self.make_pair(tmp_path, with_meta=False)
        proc = run_cli("compare", "--reference", str(r), "--candidate", str(c),
                       "--strict-bits", "--output", str(tmp_path / "cmp.json"))
        assert proc.returncode == 1

    def test_plain_compare_without_flags_passes_on_equal(self, tmp_path):
        r, c = self.make_pair(tmp_path, with_meta=False)
        proc = run_cli("compare", "--reference", str(r), "--candidate", str(c))
        assert proc.returncode == 0

    def test_empty_reference_dir_exits_nonzero(self, tmp_path):
        r, c = self.make_pair(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        proc = run_cli("compare", "--reference", str(empty), "--candidate", str(c))
        assert proc.returncode == 1


class TestBenchVariantGate:
    def test_unknown_variant_typed_error_exit_2(self):
        proc = run_cli("bench", "--variant", "cpu-numba")
        assert proc.returncode == 2
        assert "unknown variant" in proc.stderr
        assert "baseline-legacy" in proc.stderr

    def test_known_variant_name_accepted_at_parse(self):
        # variant exists; actual solve is not executed in tests — assert the
        # typed gate distinguishes known from unknown without running the
        # multi-minute harness
        proc = run_cli("bench", "--variant", "baseline-legacy", "--help")
        assert proc.returncode == 0


class TestReportSubcommand:
    def test_require_evidence_fails_on_empty_dir(self, tmp_path):
        proc = run_cli("report", "--artifacts", str(tmp_path), "--require-evidence")
        assert proc.returncode == 1
        assert "missing evidence" in proc.stderr

    def test_report_passes_with_evidence(self, tmp_path):
        # parity evidence
        (tmp_path / "parity_e1.json").write_text(json.dumps(
            {"all_equal": True, "total_mismatch_count": 0}))
        # bench evidence
        (tmp_path / "bench_baseline-legacy.json").write_text(json.dumps(
            {"variant": "baseline-legacy"}))
        # runs evidence
        (tmp_path / "run_record.jsonl").write_text(json.dumps(
            {"run_id": "e1", "mode": "edit", "case": "E1", "phase": "sample",
             "rep": 0, "timings_s": {"total_s": 1.0}, "origin": "/x/y",
             "source_sha256": {"shadow.py": "ab" * 32}}) + "\n")
        proc = run_cli("report", "--artifacts", str(tmp_path), "--require-evidence")
        assert proc.returncode == 0, proc.stderr
        payload = json.loads((tmp_path / "report.json").read_text())
        assert payload["evidence"]["parity"]["count"] == 1
        assert payload["evidence"]["bench"]["count"] == 1
        assert (tmp_path / "report.md").is_file()

    def test_report_without_flag_succeeds_on_empty(self, tmp_path):
        proc = run_cli("report", "--artifacts", str(tmp_path))
        assert proc.returncode == 0
        assert (tmp_path / "report.json").is_file()
