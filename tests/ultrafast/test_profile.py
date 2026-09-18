"""Tests for solweig_core.profile — numerical profile identity + certification.

RED witnesses included (each must FAIL certification):
* a manifest captured from one tree certified against a DIFFERENT expected
  root (the oracle cannot be silently swapped, even by a byte-identical twin),
* a source file edited after the manifest was taken,
* a tree missing mandatory scientific sources,
* a manifest pinned to frozen hashes that no longer match.

GREEN guards in the same file prove certification is not vacuously failing:
the main tree self-certifies PASS, and the oracle worktree certifies against
its frozen pin (benchmarks/ultrafast/baseline/oracle_pin.json).

Post-T02 contract: the main tree advances (solver.py seam landed in 6977086)
while the oracle stays pinned at e0d19fc, so main-vs-oracle hash identity is
NOT asserted anywhere — the oracle pin is self-standing.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_core import profile as sp  # noqa: E402
from solweig_core.profile import (
    CANONICAL_CPU_V1,
    LEGACY_CUDA_V1,
    ProfileManifest,
    certify,
    snapshot_tree,
)

ORACLE_TREE = Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc")
ORACLE_PIN_JSON = REPO_ROOT / "benchmarks" / "ultrafast" / "baseline" / "oracle_pin.json"
ORACLE_COMMIT = "e0d19fc98220a333ea312efe4604780803bbdd17"


@pytest.fixture(scope="module")
def imported_manifest():
    """Capture the manifest of the solweig_gpu this test process imports."""
    return sp.capture_imported_origin(CANONICAL_CPU_V1)


# ---------------------------------------------------------------------------
# profile registry
# ---------------------------------------------------------------------------
class TestProfileRegistry:
    def test_known_ids_and_policy_fields(self):
        assert CANONICAL_CPU_V1 == "canonical_cpu_v1"
        assert LEGACY_CUDA_V1 == "legacy_cuda_v1"
        for pid in (CANONICAL_CPU_V1, LEGACY_CUDA_V1):
            policy = sp.PROFILE_REGISTRY[pid]
            for field in ("rounding_mode", "ftz_daz", "fma_contraction",
                          "reduction_order", "weak_scalar_promotion"):
                assert field in policy, (pid, field)

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError, match="unknown profile_id"):
            snapshot_tree(REPO_ROOT, "cuda_loose_v9")

    def test_manifest_roundtrip(self, imported_manifest):
        clone = ProfileManifest.from_json(imported_manifest.to_json())
        assert clone == imported_manifest


# ---------------------------------------------------------------------------
# GREEN: self-certification passes
# ---------------------------------------------------------------------------
class TestCertifyGreen:
    def test_self_certify_no_expected_root(self, imported_manifest):
        report = certify(imported_manifest)
        assert report["status"] == "PASS", report

    def test_self_certify_against_main_tree(self, imported_manifest):
        report = certify(imported_manifest, expected_root=REPO_ROOT)
        assert report["status"] == "PASS", report

    def test_recorded_hashes_match_disk(self, imported_manifest):
        # independent recomputation (not via solweig_core) on one file
        origin = Path(imported_manifest.solweig_gpu_file)
        assert origin.is_file()
        direct = hashlib.sha256((origin.parent / "shadow.py").read_bytes()).hexdigest()
        assert imported_manifest.source_sha256["shadow.py"] == direct

    def test_snapshot_tree_matches_imported_origin(self, imported_manifest):
        snap = snapshot_tree(REPO_ROOT, CANONICAL_CPU_V1)
        assert snap.source_sha256 == imported_manifest.source_sha256
        assert Path(snap.solweig_gpu_file) == Path(imported_manifest.solweig_gpu_file)

    def test_certify_with_matching_frozen_pin(self, imported_manifest):
        report = certify(
            imported_manifest,
            expected_root=REPO_ROOT,
            expected_source_sha256=dict(imported_manifest.source_sha256),
        )
        assert report["status"] == "PASS", report


# ---------------------------------------------------------------------------
# RED witnesses: certification must FAIL
# ---------------------------------------------------------------------------
class TestCertifyRedWitnesses:
    def test_wrong_origin_root_fails_even_with_identical_hashes(self):
        """The core anti-swap witness (reworked post-T02).

        Built entirely from the oracle side: a manifest captured from the
        oracle worktree certifies PASS against the oracle root, and the SAME
        manifest — same origin file, same content hashes — FAILS when the
        declared expected root is flipped to the main tree. The witness
        property is exactly "identical origin content + wrong declared root
        = FAIL"; main-tree hashes are never assumed to equal oracle hashes
        (they legitimately diverge after the T02 solver seam, 6977086).
        """
        if not (ORACLE_TREE / "solweig_gpu" / "__init__.py").is_file():
            pytest.skip("pinned oracle worktree not present on this host")
        oracle_manifest = snapshot_tree(ORACLE_TREE, LEGACY_CUDA_V1)
        good = certify(oracle_manifest, expected_root=ORACLE_TREE)
        assert good["status"] == "PASS", good
        bad = certify(oracle_manifest, expected_root=REPO_ROOT)
        assert bad["status"] == "FAIL"
        failed = {c["check"] for c in bad["checks"] if not c["ok"]}
        assert failed == {"origin_inside_expected_root"}, failed

    def test_fake_origin_path_fails(self, imported_manifest, tmp_path):
        fake = imported_manifest.__class__.from_dict(
            {**imported_manifest.__dict__,
             "solweig_gpu_file": str(tmp_path / "solweig_gpu" / "__init__.py")})
        report = certify(fake)
        assert report["status"] == "FAIL"
        failed = {c["check"] for c in report["checks"] if not c["ok"]}
        assert "origin_file_exists" in failed

    def test_source_edited_after_capture_fails(self, tmp_path):
        """Copy the real key sources, snapshot, flip one byte in shadow.py."""
        pkg = tmp_path / "solweig_gpu"
        (pkg / "incremental").mkdir(parents=True)
        src_pkg = REPO_ROOT / "solweig_gpu"
        for rel in sp.KEY_SOURCES:
            shutil.copy2(src_pkg / rel, pkg / rel)
        shutil.copy2(src_pkg / "__init__.py", pkg / "__init__.py")
        snap = snapshot_tree(tmp_path, CANONICAL_CPU_V1)
        assert certify(snap)["status"] == "PASS"
        # mutate: append a byte to the shadow march source
        with open(pkg / "shadow.py", "ab") as fh:
            fh.write(b"\n# silent edit\n")
        report = certify(snap)
        assert report["status"] == "FAIL"
        failed = {c["check"] for c in report["checks"] if not c["ok"]}
        assert "sources_unchanged_on_disk" in failed

    def test_missing_mandatory_source_fails(self, tmp_path):
        pkg = tmp_path / "solweig_gpu"
        (pkg / "incremental").mkdir(parents=True)
        src_pkg = REPO_ROOT / "solweig_gpu"
        for rel in sp.KEY_SOURCES:
            if rel == "incremental/veg_svf_state.py":
                continue  # the exact file the SVF fold state lives in
            shutil.copy2(src_pkg / rel, pkg / rel)
        shutil.copy2(src_pkg / "__init__.py", pkg / "__init__.py")
        snap = snapshot_tree(tmp_path, CANONICAL_CPU_V1)
        report = certify(snap)
        assert report["status"] == "FAIL"
        failed = {c["check"] for c in report["checks"] if not c["ok"]}
        assert "mandatory_sources_present" in failed

    def test_frozen_pin_mismatch_fails(self, imported_manifest):
        tampered = dict(imported_manifest.source_sha256)
        tampered["shadow.py"] = "0" * 64
        report = certify(imported_manifest, expected_source_sha256=tampered)
        assert report["status"] == "FAIL"
        failed = {c["check"] for c in report["checks"] if not c["ok"]}
        assert "manifest_matches_frozen_pin" in failed

    def test_assert_certified_raises_on_failure(self, imported_manifest):
        with pytest.raises(AssertionError, match="origin_inside_expected_root"):
            sp.assert_certified(imported_manifest, expected_root="/nonexistent-root")


# ---------------------------------------------------------------------------
# oracle worktree pin (self-standing drift detection; main is NOT compared)
# ---------------------------------------------------------------------------
class TestOracleWorktree:
    def test_oracle_worktree_matches_frozen_pin(self):
        """The oracle worktree TODAY still matches the committed pin.

        Post-T02 the main tree advances independently, so this check no
        longer compares main to oracle. It certifies the live oracle
        worktree against benchmarks/ultrafast/baseline/oracle_pin.json:
        same origin root, every key-source sha256 equal, and the pin's
        commit id is the pinned e0d19fc. Oracle-side drift (an edited
        oracle file, a moved worktree) FAILS here.
        """
        pin = json.loads(ORACLE_PIN_JSON.read_text())
        assert pin["oracle_commit"] == ORACLE_COMMIT
        root = Path(pin["origin_root"])
        if not (root / "solweig_gpu" / "__init__.py").is_file():
            pytest.skip("pinned oracle worktree not present on this host")
        snap = snapshot_tree(root, LEGACY_CUDA_V1)
        report = certify(snap, expected_root=root,
                         expected_source_sha256=pin["source_sha256"])
        assert report["status"] == "PASS", report

    def test_pin_covers_exactly_the_key_sources(self):
        """The pin cannot silently go stale relative to solweig_core.KEY_SOURCES."""
        pin = json.loads(ORACLE_PIN_JSON.read_text())
        assert set(pin["source_sha256"]) == set(sp.KEY_SOURCES)
