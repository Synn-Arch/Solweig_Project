"""Numerical profile ids and import-origin/source-hash certification.

DESIGN.ko.md 3.4: the reference execution context is itself an input. Every
acceptance artifact must carry (a) the ``solweig_gpu.__file__`` the producing
process actually imported and (b) per-run sha256 of the key scientific
sources, so a silently swapped oracle tree cannot pass certification even
when its plane bytes happen to match.

Two capture modes:

* :func:`capture_imported_origin` — imports ``solweig_gpu`` in *this* process
  and records what was imported (runtime versions included). Call it from the
  tree that should execute (cwd pinning is the caller's job, as in
  ``benchmarks/ultrafast/harness_solve.py``).
* :func:`snapshot_tree` — hashes a tree on disk without importing it. Used to
  certify reference artifacts after the fact and to inspect the pinned oracle
  worktree from the main tree.

:func:`certify` re-hashes the key sources from disk at certification time and
checks the recorded origin path, so it detects both a wrong tree and a file
edited after the manifest was taken.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Superset of the T00 harness key files; the three shadow/solver/veg sources
# are mandatory for oracle certification, solweig/utci_process are tracked as
# the remaining scientific entry points T00 already recorded.
KEY_SOURCES: tuple[str, ...] = (
    "shadow.py",
    "solweig.py",
    "utci_process.py",
    "incremental/solver.py",
    "incremental/veg_svf_state.py",
)
# Files whose absence makes origin certification structurally impossible
# (they define the shadow march and the incremental exact solve).
MANDATORY_SOURCES: frozenset[str] = frozenset(
    {"shadow.py", "incremental/solver.py", "incremental/veg_svf_state.py"}
)

CANONICAL_CPU_V1 = "canonical_cpu_v1"
LEGACY_CUDA_V1 = "legacy_cuda_v1"

# Rounding-policy fields are *declared contracts* for each profile (what a
# kernel must preserve), not measurements; observed behavior comes from the
# primitive characterization matrix (benchmarks/ultrafast/run.py + tests).
PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
    CANONICAL_CPU_V1: {
        "device_class": "cpu",
        "rounding_mode": "round_to_nearest_even",
        "ftz_daz": "denormals_honored_ftz_off_daz_off",
        "fma_contraction": "forbidden_outside_primitive_implementation",
        "reduction_order": "original_sequential_or_declared_tree_only",
        "weak_scalar_promotion": "float32_tensor_plus_python_float_stays_float32",
        "compaction": "must_preserve_original_valid_ordinal_and_lane_context",
    },
    LEGACY_CUDA_V1: {
        "device_class": "cuda",
        "rounding_mode": "round_to_nearest_even",
        "ftz_daz": "device_default_reported_per_run",
        "fma_contraction": "aten_default_allowed_inside_primitive",
        "reduction_order": "aten_default_recorded_per_run",
        "weak_scalar_promotion": "float32_tensor_plus_python_float_stays_float32",
        "compaction": "legacy_boolean_compaction_extent_dependent_w2_known",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OracleEnvironmentError(RuntimeError):
    """Typed refusal (T14 / R1): an oracle-side operation was attempted in
    an environment that cannot provide the oracle.

    ``capture_imported_origin`` imports ``solweig_gpu`` (whose module
    graph includes torch) IN THIS PROCESS. In the torch-free runtime that
    import fails; a bare ``ModuleNotFoundError`` would hide WHICH seam
    broke. This error names the seam and the torch-free alternative
    (:func:`snapshot_tree`), so certification contexts can hash an oracle
    tree on disk instead of importing it.
    """

    def __init__(self, operation: str, cause: ModuleNotFoundError) -> None:
        missing = getattr(cause, "name", None) or "an oracle dependency"
        super().__init__(
            f"{operation} requires the ORACLE environment (the dev env with "
            f"torch + solweig_gpu importable); this process cannot import it "
            f"({cause.__class__.__name__}: missing {missing!r}). The "
            "torch-free alternative is snapshot_tree(tree_root, profile_id), "
            "which hashes the oracle sources on disk WITHOUT importing them. "
            "The runtime never silently falls back to another oracle."
        )
        self.operation = operation
        self.missing_module = missing
        self.cause = cause


@dataclass
class ProfileManifest:
    """Everything needed to pin the execution context of one run/artifact."""

    profile_id: str
    solweig_gpu_file: str  # resolved __file__ of the imported package
    source_sha256: dict[str, str] = field(default_factory=dict)
    captured_at_utc: str = ""
    capture_mode: str = "imported"  # "imported" | "tree_snapshot"
    cwd: str = ""
    python: str = ""
    numpy: str = ""
    torch: str = ""
    torch_file: str = ""
    torch_threads: int | None = None
    torch_interop_threads: int | None = None
    machine: str = ""
    rounding_policy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.captured_at_utc:
            self.captured_at_utc = _utc_now()
        if not self.machine:
            import platform

            self.machine = f"{platform.system()}/{platform.machine()}"
        if self.profile_id in PROFILE_REGISTRY and not self.rounding_policy:
            self.rounding_policy = dict(PROFILE_REGISTRY[self.profile_id])

    # --- serialization -----------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProfileManifest":
        data = json.loads(text)
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileManifest":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


def _package_dir_from_origin(origin_file: Path) -> Path:
    # .../solweig_gpu/__init__.py -> .../solweig_gpu
    return origin_file.resolve().parent


def _hash_key_sources(pkg_dir: Path) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for rel in KEY_SOURCES:
        p = pkg_dir / rel
        if p.is_file():
            hashes[rel] = sha256_file(p)
        else:
            missing.append(rel)
    return hashes, missing


def capture_imported_origin(profile_id: str) -> ProfileManifest:
    """Record the solweig_gpu package THIS process actually imports.

    The caller is responsible for cwd/path pinning before the first
    ``solweig_gpu`` import (see harness_solve.pin_import_origin_to_cwd); if
    the module is already in sys.modules we report where it came from.
    """
    if profile_id not in PROFILE_REGISTRY:
        raise ValueError(
            f"unknown profile_id {profile_id!r}; known: {sorted(PROFILE_REGISTRY)}"
        )
    try:
        import solweig_gpu  # noqa: F401  (origin is the point of the import)
    except ModuleNotFoundError as cause:
        # T14/R1: torch-free runtime must refuse loudly and typed, never
        # leak a bare ModuleNotFoundError from an implicit import chain.
        raise OracleEnvironmentError(
            "profile.capture_imported_origin", cause
        ) from cause

    origin = Path(solweig_gpu.__file__).resolve()
    pkg_dir = origin.parent
    hashes, _missing = _hash_key_sources(pkg_dir)

    import numpy

    manifest = ProfileManifest(
        profile_id=profile_id,
        solweig_gpu_file=str(origin),
        source_sha256=hashes,
        capture_mode="imported",
        cwd=str(Path.cwd().resolve()),
        python=sys.version.split()[0],
        numpy=numpy.__version__,
    )
    try:
        import torch

        manifest.torch = torch.__version__
        manifest.torch_file = str(Path(torch.__file__).resolve())
        manifest.torch_threads = int(torch.get_num_threads())
        manifest.torch_interop_threads = int(torch.get_num_interop_threads())
    except ModuleNotFoundError:
        pass  # torch-free certification context (T14): origin is still pinned
    return manifest


def snapshot_tree(tree_root: str | Path, profile_id: str) -> ProfileManifest:
    """Hash a tree's solweig_gpu sources on disk without importing them."""
    if profile_id not in PROFILE_REGISTRY:
        raise ValueError(
            f"unknown profile_id {profile_id!r}; known: {sorted(PROFILE_REGISTRY)}"
        )
    root = Path(tree_root).resolve()
    origin = root / "solweig_gpu" / "__init__.py"
    if not origin.is_file():
        raise FileNotFoundError(f"not a solweig tree (no solweig_gpu package): {root}")
    hashes, _missing = _hash_key_sources(origin.parent)
    return ProfileManifest(
        profile_id=profile_id,
        solweig_gpu_file=str(origin),
        source_sha256=hashes,
        capture_mode="tree_snapshot",
        cwd=str(Path.cwd().resolve()),
        python=sys.version.split()[0],
    )


def certify(
    manifest: ProfileManifest,
    *,
    expected_root: str | Path | None = None,
    expected_source_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Re-verify a manifest against the current disk state.

    Checks, in order:
      1. profile_id is a known profile.
      2. the recorded origin file exists on disk now.
      3. mandatory key sources are present in the manifest (an origin that
         could not hash shadow/solver/veg_svf_state is not certifiable).
      4. optional ``expected_root``: the origin must live inside that tree —
         identical file hashes in a different tree are still a FAIL (the
         oracle cannot be silently swapped for a byte-identical twin).
      5. optional ``expected_source_sha256``: manifest hashes must equal it
         (frozen-pin check, independent of current disk state).
      6. current disk hashes must equal the manifest hashes (detects edits
         after capture).

    Returns a report dict with ``status`` PASS/FAIL and per-check results.
    Never raises on certification failure; invalid arguments raise.
    """
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    ok = record(
        "profile_id_known",
        manifest.profile_id in PROFILE_REGISTRY,
        manifest.profile_id,
    )

    origin = Path(manifest.solweig_gpu_file)
    origin_ok = origin.is_file() and origin.name == "__init__.py"
    ok &= record("origin_file_exists", origin_ok, str(origin))

    if expected_root is not None:
        root_resolved = Path(expected_root).resolve()
        try:
            inside = str(origin.resolve()).startswith(str(root_resolved) + os.sep)
        except OSError:
            inside = False
        ok &= record(
            "origin_inside_expected_root",
            inside,
            f"origin={origin} expected_root={root_resolved}",
        )

    missing_mandatory = sorted(MANDATORY_SOURCES - set(manifest.source_sha256))
    ok &= record(
        "mandatory_sources_present",
        not missing_mandatory,
        f"missing={missing_mandatory}",
    )

    if expected_source_sha256 is not None:
        diffs = {
            rel: {"manifest": manifest.source_sha256.get(rel),
                  "expected": expected_source_sha256.get(rel)}
            for rel in set(expected_source_sha256) | set(manifest.source_sha256)
            if expected_source_sha256.get(rel) != manifest.source_sha256.get(rel)
        }
        ok &= record(
            "manifest_matches_frozen_pin",
            not diffs,
            json.dumps(diffs, sort_keys=True)[:400],
        )

    if origin_ok:
        current, missing_now = _hash_key_sources(_package_dir_from_origin(origin))
        edited = {
            rel: {"manifest": manifest.source_sha256.get(rel), "disk": current.get(rel)}
            for rel in set(manifest.source_sha256) | set(current)
            if manifest.source_sha256.get(rel) != current.get(rel)
        }
        ok &= record(
            "sources_unchanged_on_disk",
            not missing_now and not edited,
            f"missing_now={missing_now} edited={json.dumps(edited, sort_keys=True)[:400]}",
        )

    return {
        "schema_version": 1,
        "status": "PASS" if ok else "FAIL",
        "profile_id": manifest.profile_id,
        "origin": manifest.solweig_gpu_file,
        "checks": checks,
    }


def assert_certified(manifest: ProfileManifest, **kwargs: Any) -> dict[str, Any]:
    report = certify(manifest, **kwargs)
    if report["status"] != "PASS":
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    return report
