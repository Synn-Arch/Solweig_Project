# SPDX-License-Identifier: GPL-3.0-only
"""T14 Change D gates: the runtime DISTRIBUTION and its witnesses.

RED witnesses demanded by the task card:

* **packaging-extra-missing** — the runtime distribution
  (``packaging/runtime-cpu``) declares exactly numpy/numba/scipy; the
  oracle stack (torch), preprocess stack (GDAL/rasterio), plotting,
  notebook IO and any compiler toolchain are absent from the wheel's
  dependency graph AND its contents (no ``solweig_gpu``, no tests, no
  CUDA lane).
* **torch-free execution in a genuinely clean environment** — the
  clean venv (wheel + numpy/numba/scipy ONLY) imports the runtime,
  builds step tables, refuses unseen scenes typed, and torch never
  enters ``sys.modules``. The dev-env twin proves the harder direction:
  torch importable on disk yet still never imported.
* **ISA mismatch** — a numba cache directory warm under one CPU name
  does NOT serve a process with a different ``NUMBA_CPU_NAME``: the
  artifacts are rewritten (fresh compile) while the computed bits stay
  identical.
* **cross-env parity** — the quick workload matrix (W2 unseen-angle
  tables, W6 unseen-scale, W8 process-to-ready) produces IDENTICAL
  digests in the dev repo env and the clean wheel env.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ULTRA_DIR = Path(__file__).resolve().parent
PYPROJECT = REPO_ROOT / "packaging" / "runtime-cpu" / "pyproject.toml"
WHEEL_DIR = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14/dist-runtime"
)
CLEAN_PY = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14/"
    "venv-runtime/bin/python"
)
CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14/compact_capture")

FORBIDDEN_DEPS = (
    "torch", "gdal", "GDAL", "rasterio", "matplotlib", "netCDF4",
    "pandas", "jupyter", "ipykernel", "notebook", "cupy", "nvidia",
)
ALLOWED_DEPS = {"numpy", "numba", "scipy"}


def _clean_python() -> Path | None:
    return CLEAN_PY if CLEAN_PY.is_file() else None


def _wheel() -> Path | None:
    if not WHEEL_DIR.is_dir():
        return None
    wheels = sorted(WHEEL_DIR.glob("solweig_core_cpu-*.whl"))
    return wheels[-1] if wheels else None


def _run_env(py: str | Path, script: str, *, pythonpath: str | None = None,
             env_extra: dict | None = None) -> str:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [str(py), "-c", script], capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    return proc.stdout


class TestPackagingWitness:
    def test_pyproject_dependencies_are_minimal(self):
        with open(PYPROJECT, "rb") as fh:
            spec = tomllib.load(fh)
        deps = spec["project"]["dependencies"]
        names = {d.split(">")[0].split("<")[0].split("=")[0].strip()
                 for d in deps}
        assert names <= ALLOWED_DEPS, (
            f"runtime grew dependencies beyond numpy/numba/scipy: {names}"
        )
        for dep in deps:
            for bad in FORBIDDEN_DEPS:
                assert bad not in dep, f"forbidden dependency {bad!r} in {deps}"

    def test_wheel_metadata_and_contents_clean(self):
        wheel = _wheel()
        if wheel is None:
            pytest.skip("runtime wheel not built yet")
        with zipfile.ZipFile(wheel) as z:
            names = z.namelist()
            meta = z.read(
                next(n for n in names if n.endswith(".dist-info/METADATA"))
            ).decode()
        requires = [
            line.split(":", 1)[1].strip()
            for line in meta.splitlines()
            if line.startswith("Requires-Dist:")
        ]
        for req in requires:
            for bad in FORBIDDEN_DEPS:
                assert bad not in req.lower(), f"{bad!r} required: {req}"
        # contents: ONLY the CPU runtime packages + dist-info
        allowed_prefixes = ("solweig_core/", "solweig_core_cpu-")
        for name in names:
            assert name.startswith(allowed_prefixes), (
                f"wheel ships foreign content: {name}"
            )
        # the oracle/CUDA/test trees never ship
        forbidden_paths = ("solweig_gpu", "native", "tests/", "docs/",
                           "setup.py", "packaging/")
        for name in names:
            for bad in forbidden_paths:
                assert bad not in name, f"wheel ships {bad}: {name}"

    def test_installed_footprint_small(self):
        if _clean_python() is None:
            pytest.skip("clean venv absent")
        site = json.loads(_run_env(
            _clean_python(),
            "import json, solweig_core, pathlib;"
            "print(json.dumps(str(pathlib.Path(solweig_core.__file__).parent)))",
        ))
        pkg_bytes = sum(
            p.stat().st_size for p in Path(site).rglob("*") if p.is_file()
        )
        wheel = _wheel()
        wheel_bytes = wheel.stat().st_size if wheel else 0
        assert pkg_bytes < 10 * 2**20, (
            f"installed solweig_core is {pkg_bytes / 2**20:.1f} MiB (>10; "
            "plain .py + __pycache__ of the CPU core only)"
        )
        assert wheel_bytes < 2**20, (
            f"wheel is {wheel_bytes / 2**20:.2f} MiB (>1)"
        )


class TestCleanVenvRuntime:
    def test_import_report_and_step_table_without_torch(self):
        py = _clean_python()
        if py is None:
            pytest.skip("clean venv absent")
        script = r"""
import hashlib, json, sys
import numpy as np
import solweig_core.runtime as rt
assert "torch" not in sys.modules
assert not any(m.startswith("solweig_gpu") for m in sys.modules)
rep = rt.runtime_dependency_report()
t = rt.step_table("svf_shadow", 123.4, 45.0, 0.5, 40, 80, 20.0)
payload = np.concatenate([
    np.asarray(t.dx, np.int64),
    np.asarray(t.dy, np.int64),
    np.asarray(t.dz_bits, np.uint64),
])
try:
    rt.Runtime.__new__(rt.Runtime).amplitude_policy(a=None, vegdsm=None,
                                                     vegdsm2=None)
    amp = "NO-REFUSAL"
except rt.UnseenSceneRefusal:
    amp = "UnseenSceneRefusal"
print("@@CLEAN@@", json.dumps({
    "torch": "torch" in sys.modules,
    "third_party": sorted(rep["third_party"]),
    "count": int(t.count),
    "digest": hashlib.sha256(payload.tobytes()).hexdigest(),
    "amplitude_policy": amp,
}))
"""
        out = _run_env(py, script)
        payload = json.loads(out.split("@@CLEAN@@")[1])
        assert payload["torch"] is False
        assert payload["amplitude_policy"] == "UnseenSceneRefusal"
        tops = {m.split("-")[0].split(">")[0] for m in payload["third_party"]}
        assert tops <= {"numpy", "numba", "llvmlite", "scipy"}, tops
        # same table from the DEV env (repo tree): bit-identical
        dev = json.loads(_run_env(
            sys.executable, script, pythonpath=str(REPO_ROOT),
        ).split("@@CLEAN@@")[1])
        assert dev["digest"] == payload["digest"], (
            "clean-env step table diverged from the dev env"
        )

    def test_unseen_scale_never_imports_torch_in_dev_env(self):
        """The trap direction: torch IS importable in the dev env, an
        unseen-scale request must still solve torch-free or refuse
        typed — never silently import the oracle."""
        script = r"""
import sys
import solweig_core.runtime as rt
t = rt.step_table("svf_shadow", 61.0, 30.0, 1.0 / 7.5, 32, 48, 12.0)
try:
    rt.require_runtime_backend("legacy-torch-cpu")
    legacy = "NO-REFUSAL"
except rt.OracleBackendRefusal:
    legacy = "OracleBackendRefusal"
print("@@LEAK@@", sys.modules.get("torch") is None, int(t.count), legacy)
"""
        clean, dev = None, None
        if _clean_python() is not None:
            clean = _run_env(_clean_python(), script)
        dev = _run_env(sys.executable, script, pythonpath=str(REPO_ROOT))
        assert dev.split("@@LEAK@@")[1].split()[0] == "True", (
            "torch entered sys.modules during an unseen-scale solve (dev)"
        )
        assert "OracleBackendRefusal" in dev
        if clean is not None:
            assert clean.split("@@LEAK@@")[1].split()[0] == "True"


class TestISAMismatchCache:
    def test_foreign_cpu_name_misses_cache_preserves_bits(self, tmp_path):
        cache = tmp_path / "nbcache"
        cache.mkdir()
        probe = ULTRA_DIR / "t14_isa_probe.py"

        def run(cpu_name: str | None) -> tuple[str, frozenset]:
            env = {"NUMBA_CACHE_DIR": str(cache)}
            if cpu_name is not None:
                env["NUMBA_CPU_NAME"] = cpu_name
            proc = subprocess.run(
                [sys.executable, str(probe), str(cache)],
                capture_output=True, text=True,
                env={**{k: v for k, v in os.environ.items()
                        if k not in ("NUMBA_CPU_NAME", "NUMBA_CACHE_DIR")},
                     **env},
            )
            assert proc.returncode == 0, proc.stderr[-2000:]
            lines = [l for l in proc.stdout.splitlines() if l.strip()]
            result = next(l for l in lines if l.startswith("@@RESULT@@"))
            artifacts = frozenset(l for l in lines if not l.startswith("@@"))
            return result, artifacts

        cold = run(None)          # compiles, populates the cache
        warm = run(None)          # pure hit: artifacts byte-identical
        assert warm == cold, (
            "warm same-CPU run rewrote cache artifacts (unexpected)"
        )
        foreign = run("cyclone")  # different ISA key
        assert foreign[0] == cold[0], (
            "bits changed under a foreign CPU name"
        )
        assert foreign[1] != warm[1], (
            "foreign-ISA run was served by the same cache artifacts — "
            "the ISA mismatch did not force a recompile"
        )


class TestCrossEnvParity:
    def test_quick_matrix_digests_match_across_envs(self, tmp_path):
        py = _clean_python()
        if py is None:
            pytest.skip("clean venv absent")
        if not (CAP / "t00.npz").is_file():
            pytest.skip("compact capture absent")
        driver = ULTRA_DIR / "t14_clean_env_driver.py"
        cache = tmp_path / "nbcache"
        cache.mkdir()

        def matrix(py_exec: str, pythonpath: str | None) -> dict:
            env = {
                k: v for k, v in os.environ.items()
                if k not in ("NUMBA_CPU_NAME", "NUMBA_CACHE_DIR", "PYTHONPATH")
            }
            env["NUMBA_CACHE_DIR"] = str(cache)
            if pythonpath is not None:
                env["PYTHONPATH"] = pythonpath
            proc = subprocess.run(
                [py_exec, str(driver), "--quick"],
                capture_output=True, text=True, env=env,
            )
            assert proc.returncode == 0, proc.stderr[-2000:]
            return json.loads(
                proc.stdout.split("@@MATRIX@@")[1]
            )

        clean = matrix(str(py), None)          # installed wheel
        dev = matrix(sys.executable, str(REPO_ROOT))  # repo tree
        for name, matrix_ in (("clean", clean), ("dev", dev)):
            assert matrix_["env"]["torch_in_modules"] is False, name
        assert not any(
            m == "torch" for m in
            clean["env"]["third_party_tops_after_workloads"]
        )
        assert clean["workloads"]["W2_unseen_angle_tables"] == \
            dev["workloads"]["W2_unseen_angle_tables"], \
            "unseen-angle tables diverged across environments"
        assert clean["workloads"]["W6_unseen_scale"]["table_digest"] == \
            dev["workloads"]["W6_unseen_scale"]["table_digest"]
        assert clean["workloads"]["W8_process_to_ready"][
            "first_output_digest"
        ] == dev["workloads"]["W8_process_to_ready"]["first_output_digest"], \
            "first radiation output diverged across environments"
