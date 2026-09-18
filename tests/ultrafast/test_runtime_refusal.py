# SPDX-License-Identifier: GPL-3.0-only
"""T14 Change B gates: the torch-free runtime facade + typed refusals.

``solweig_core.runtime`` is the TRUE torch-free runtime entry (T14 card
item 3): it exposes the cached-capture solve path and the general
step-table producer, and refuses — LOUDLY and TYPED, never via a bare
``ModuleNotFoundError`` or a silent import — everything that belongs to
the oracle environment:

* the ``legacy-torch-cpu`` backend (:class:`OracleBackendRefusal`);
* capture REGENERATION (geometry edits / new scenes — the frozen planes
  come from the oracle; the runtime replays, it never regenerates)
  (:class:`CaptureRegenerationRefusal`);
* unseen-scene amplitude-policy computation (scene statistics the oracle
  computes with torch) (:class:`UnseenSceneRefusal`);
* oracle-origin profile capture in a torch-free process
  (:class:`solweig_core.profile.OracleEnvironmentError`, R1) — with
  :func:`solweig_core.profile.snapshot_tree` as the working torch-free
  alternative.

RED witnesses (T14 card item 5, subset — the remaining packaging/ISA
witnesses live in test_runtime_packaging.py):

* transitively imported torch in a CLEAN interpreter (simulated via a
  meta-path blocker so the dev env can run it);
* hidden ``.to()`` adapters and torch type annotations in the runtime
  module surface (AST-level).
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_core import runtime as rt  # noqa: E402

#: the modules the torch-free runtime actually imports (directly or
#: lazily). The AST purity witnesses scan exactly this surface.
RUNTIME_MODULE_SURFACE = [
    REPO_ROOT / "solweig_core" / "runtime.py",
    REPO_ROOT / "solweig_core" / "profile.py",
    REPO_ROOT / "solweig_core" / "step_tables.py",
    REPO_ROOT / "solweig_core" / "dispatch.py",
]
RUNTIME_MODULE_SURFACE += sorted(
    (REPO_ROOT / "solweig_core" / "numba_cpu").glob("*.py")
)

#: blocks torch AND solweig_gpu at import time inside the subprocess —
#: a faithful torch-free-environment simulation runnable from the dev env
#: (the real clean venv runs are the T14 exit matrix, Change D).
_BLOCKER = """
import sys


class _Blocker:
    blocked = ("torch", "solweig_gpu")

    def find_module(self, name, path=None):
        if name.split(".")[0] in self.blocked:
            return self
        return None

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.blocked:
            raise ModuleNotFoundError(
                f"import of {name!r} blocked (torch-free simulation)",
                name=name,
            )
        return None

    def load_module(self, name):
        raise ModuleNotFoundError(name=name)


sys.meta_path.insert(0, _Blocker())
"""


def _run_blocked(code: str):
    """Run ``code`` in a subprocess where torch/solweig_gpu CANNOT import."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKER + "\n" + code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result


# ---------------------------------------------------------------------------
# Purity: the facade never pulls torch, even with torch installed
# ---------------------------------------------------------------------------


class TestRuntimePurity:
    def test_import_without_torch(self):
        code = (
            "import sys\n"
            "from solweig_core import runtime\n"
            "assert 'torch' not in sys.modules, 'torch imported transitively'\n"
            "assert not any(m.startswith('solweig_gpu') for m in sys.modules)\n"
            "print('PURE', runtime.RUNTIME_BACKEND)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("PURE"), result.stdout

    def test_import_torch_free_simulated(self):
        """The blocked-import subprocess: importing the facade and
        RESOLVING a backend works; torch is absent and unimportable."""
        code = (
            "from solweig_core import runtime\n"
            "assert runtime.require_runtime_backend() == 'numba-full-solve'\n"
            "import sys\n"
            "assert 'torch' not in sys.modules\n"
            "print('OK')\n"
        )
        result = _run_blocked(code)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_ast_no_hidden_to_adapter(self):
        """RED witness: no hidden torch ``.to(device/dtype)`` adapter call
        anywhere in the runtime module surface (an AST scan, not text —
        string mentions in comments do not trip it)."""
        offenders = []
        for path in RUNTIME_MODULE_SURFACE:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "to"
                    ):
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"hidden .to() adapter calls: {offenders}"

    def test_ast_no_torch_annotations(self):
        """RED witness: no torch type annotations (torch.Tensor / nn.* /
        device) in function signatures or variable annotations across the
        runtime module surface — an annotation import would pull torch."""
        offenders = []
        for path in RUNTIME_MODULE_SURFACE:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.AnnAssign):
                    names.append(node.annotation)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in (
                        list(node.args.args)
                        + list(node.args.kwonlyargs)
                        + ([node.args.vararg] if node.args.vararg else [])
                        + ([node.args.kwarg] if node.args.kwarg else [])
                    ):
                        if arg.annotation is not None:
                            names.append(arg.annotation)
                    if node.returns is not None:
                        names.append(node.returns)
                for ann in names:
                    for sub in ast.walk(ann):
                        if isinstance(sub, ast.Name) and sub.id == "torch":
                            offenders.append(
                                f"{path.name}:{getattr(node, 'lineno', '?')}"
                            )
                        if (
                            isinstance(sub, ast.Attribute)
                            and isinstance(sub.value, ast.Name)
                            and sub.value.id == "torch"
                        ):
                            offenders.append(
                                f"{path.name}:{getattr(node, 'lineno', '?')}"
                            )
        assert not offenders, f"torch annotations in runtime surface: {offenders}"

    def test_no_torch_import_at_module_level(self):
        """No torch/solweig_gpu import executes at MODULE import time in
        the runtime surface. Function-local imports (guarded or not — they
        run only when the oracle-path function is actually called) are the
        allowed pattern; the AST scan follows module-level statements but
        never descends into function/class bodies."""
        offenders = []

        def scan(node: ast.AST, path: Path) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue  # not module scope
                if isinstance(child, ast.Import):
                    if any(
                        a.name.split(".")[0] in ("torch", "solweig_gpu")
                        for a in child.names
                    ):
                        offenders.append(f"{path.name}:{child.lineno}")
                elif isinstance(child, ast.ImportFrom):
                    mod = (child.module or "").split(".")[0]
                    if mod in ("torch", "solweig_gpu"):
                        offenders.append(f"{path.name}:{child.lineno}")
                else:
                    scan(child, path)

        for path in RUNTIME_MODULE_SURFACE:
            scan(ast.parse(path.read_text()), path)
        assert not offenders, (
            f"module-level torch/solweig_gpu imports in runtime surface: "
            f"{offenders}"
        )


# ---------------------------------------------------------------------------
# Backend resolution + typed refusals
# ---------------------------------------------------------------------------


class TestBackendRefusals:
    def test_legacy_backend_refused_typed(self):
        with pytest.raises(rt.OracleBackendRefusal) as info:
            rt.require_runtime_backend("legacy-torch-cpu")
        message = str(info.value)
        assert "legacy-torch-cpu" in message
        assert "oracle" in message.lower()
        assert info.value.backend == "legacy-torch-cpu"

    def test_runtime_backend_resolves(self):
        assert rt.require_runtime_backend() == "numba-full-solve"
        assert (
            rt.require_runtime_backend("numba-full-solve")
            == "numba-full-solve"
        )

    def test_unknown_backend_is_valueerror(self):
        with pytest.raises(ValueError):
            rt.require_runtime_backend("does-not-exist")

    def test_dispatch_registration_consistent(self):
        from solweig_core import dispatch

        assert rt.RUNTIME_BACKEND in dispatch._BACKENDS["cpu"]
        for backend in rt.ORACLE_BACKENDS:
            assert backend in dispatch._BACKENDS["cpu"], (
                f"oracle backend {backend} drifted out of dispatch"
            )

    def test_capture_construction_requires_latitude(self):
        """Runtime must carry the site latitude into CaptureSet (T16 lead
        finding: cap_set() constructed CaptureSet WITHOUT the required
        ``latitude`` field, so the facade's own solve entry was
        unconstructible — no caller exercised it).

        Latitude is keyword-required: a Runtime that never pinned its
        site's latitude must not be constructible at all; a nonexistent
        capture WITH a latitude still refuses typed.
        """
        with pytest.raises(TypeError):
            rt.Runtime("/definitely/not/a/capture")
        with pytest.raises(rt.RuntimeRefusal):
            rt.Runtime("/definitely/not/a/capture", latitude=30.0)

    def test_cap_set_passes_latitude_through(self):
        """With a real capture dir, cap_set() builds a CaptureSet whose
        latitude is EXACTLY the one given (float-rounded once)."""
        cap = Path(
            "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/"
            "t14/compact_capture"
        )
        if not (cap / "t00.npz").is_file():
            pytest.skip("compact capture absent (dev-env artifact)")
        runtime = rt.Runtime(cap, latitude=30.312645)
        cs = runtime.cap_set()
        assert type(cs).__name__ == "CaptureSet"
        assert cs.latitude == 30.312645
        assert cs.profile == "canonical_cpu_v1"


class TestTypedRefusals:
    def test_regeneration_refused_typed(self):
        """A geometry edit routes to REGENERATE; the runtime converts the
        router decision into a typed refusal naming the oracle env — the
        frozen planes cannot be rebuilt without it."""
        runtime = rt.Runtime.__new__(rt.Runtime)  # no capture needed for route
        with pytest.raises(rt.CaptureRegenerationRefusal) as info:
            runtime.route(geometry_changed=True, r0=0, n_window=24)
        assert "regenerat" in str(info.value).lower()

    def test_met_edit_routes_without_refusal(self):
        runtime = rt.Runtime.__new__(rt.Runtime)
        decision = runtime.route(geometry_changed=False, r0=4, n_window=24)
        assert decision.route in (
            "warm-sparse-replay",
            "cold-full-solve",
        ), decision.describe()

    def test_unseen_scene_amplitude_refused_typed(self):
        runtime = rt.Runtime.__new__(rt.Runtime)
        with pytest.raises(rt.UnseenSceneRefusal) as info:
            runtime.amplitude_policy(a=None, vegdsm=None, vegdsm2=None)
        assert "oracle" in str(info.value).lower()

    def test_step_table_generation_is_runtime_supported(self):
        """Unseen-angle table generation IS a runtime capability (R3) —
        no refusal, no torch."""
        table = rt.step_table(
            "svf_shadow", 123.4, 45.0, 0.5, 32, 48, 12.0
        )
        assert table.count > 0
        assert table.key.kernel_variant == "svf_shadow"


# ---------------------------------------------------------------------------
# R1: oracle-origin capture refuses typed in a torch-free process
# ---------------------------------------------------------------------------


class TestOracleOriginGuard:
    def test_capture_imported_origin_refuses_typed(self):
        """R1: in a process where torch/solweig_gpu cannot import,
        capture_imported_origin raises OracleEnvironmentError — NOT a bare
        ModuleNotFoundError — and names snapshot_tree as the alternative."""
        code = (
            "from solweig_core import profile\n"
            "try:\n"
            "    profile.capture_imported_origin('canonical_cpu_v1')\n"
            "except profile.OracleEnvironmentError as err:\n"
            "    assert 'snapshot_tree' in str(err), err\n"
            "    assert err.missing_module in ('torch', 'solweig_gpu')\n"
            "    print('TYPED-REFUSAL')\n"
            "else:\n"
            "    raise SystemExit('no refusal raised')\n"
        )
        result = _run_blocked(code)
        assert result.returncode == 0, result.stderr
        assert "TYPED-REFUSAL" in result.stdout

    def test_bare_modulenotfound_never_leaks(self):
        code = (
            "from solweig_core import profile\n"
            "try:\n"
            "    profile.capture_imported_origin('canonical_cpu_v1')\n"
            "except ModuleNotFoundError:\n"
            "    raise SystemExit('BARE ModuleNotFoundError leaked')\n"
            "except profile.OracleEnvironmentError:\n"
            "    print('OK')\n"
        )
        result = _run_blocked(code)
        assert result.returncode == 0, result.stderr
        assert "BARE" not in result.stdout

    def test_snapshot_tree_works_torch_free(self):
        """The documented torch-free alternative hashes the oracle tree on
        disk WITHOUT importing anything from it."""
        code = (
            "from pathlib import Path\n"
            "from solweig_core import profile\n"
            "manifest = profile.snapshot_tree(\n"
            "    Path('solweig_gpu').resolve().parent, 'canonical_cpu_v1')\n"
            "assert manifest.capture_mode == 'tree_snapshot'\n"
            "assert 'shadow.py' in manifest.source_sha256\n"
            "assert manifest.source_sha256['shadow.py']\n"
            "import sys\n"
            "assert 'torch' not in sys.modules\n"
            "print('SNAPSHOT-OK')\n"
        )
        result = _run_blocked(code)
        assert result.returncode == 0, result.stderr
        assert "SNAPSHOT-OK" in result.stdout
