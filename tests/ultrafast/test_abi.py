"""Tests for solweig_core.abi — the torch-neutral buffer ABI (DESIGN 5.3).

RED witnesses encoded here (each must FAIL a naive implementation):

* a buffer claiming CUDA residency fails ABI validation on this host —
  solweig_core can host ``'cpu'`` only and never allocates or borrows CUDA
  memory (it cannot even see torch);
* a stale borrowed buffer: the view keeps its owner ALIVE (weakref
  witness), and a FROZEN view detects post-creation mutation of the owner
  (:class:`StaleBufferError`);
* noncontiguous input (negative-stride, sliced, transposed) is never
  confused with contiguous: ``is_contiguous`` is computed from the real
  strides, ``require_contiguous`` refuses strided views with a typed
  error, ``to_contiguous`` copies at the boundary preserving dtype and
  LOGICAL order, and a transpose-view roundtrip preserves values by
  logical index;
* declared metadata that lies about the owner (shape/strides/dtype/rank,
  a writable borrow of a read-only owner, zero-stride broadcast memory) is
  rejected at validation.

Plus the torch-freedom gate for the whole ``solweig_core`` package: a
fresh interpreter imports it without torch appearing in ``sys.modules``,
even though torch IS importable in this environment, and a fake
``torch.cuda.is_available() -> True`` module planted in ``sys.modules``
before the import changes nothing.
"""
from __future__ import annotations

import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_core import abi as core_abi  # noqa: E402
from solweig_core.abi import ArrayView  # noqa: E402
from solweig_core.status import (  # noqa: E402
    AbiError,
    AbiValidationError,
    NonContiguousError,
    RefusalReason,
    StaleBufferError,
)

DOMAIN = "site:test:8x6"


def make_grid(rows=4, cols=6, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, (rows, cols)).astype(dtype)


# ---------------------------------------------------------------------------
# torch-freedom of the whole core package
# ---------------------------------------------------------------------------
class TestNoTorch:
    def test_importing_solweig_core_never_imports_torch(self):
        # torch IS installed in this venv: the point is that importing the
        # core does not pull it in (a fresh interpreter, cwd at repo root).
        code = (
            "import sys;"
            "import solweig_core, solweig_core.abi, solweig_core.request, "
            "solweig_core.dispatch, solweig_core.status;"
            "assert 'torch' not in sys.modules, 'solweig_core imported torch';"
            "print('ok')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_fake_cuda_visible_torch_is_never_consulted(self):
        # RED witness for "CPU-target request on a CUDA-visible environment
        # must not allocate CUDA": plant a booby-trapped fake torch whose
        # cuda.is_available() claims True and whose every attribute access
        # fails the test if touched. solweig_core must import and answer
        # dispatch questions without ever consulting it.
        fake = (
            "import sys\n"
            "class _Boom:\n"
            "    def __getattr__(self, name):\n"
            "        raise AssertionError(f'fake torch attribute touched: {name}')\n"
            "    def __call__(self, *a, **k):\n"
            "        raise AssertionError('fake torch called')\n"
            "cuda = _Boom()\n"
            "def is_available():\n"
            "    return True\n"
            "sys.modules['torch'] = _Boom()\n"
        )
        code = (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});\n"
            f"{fake}\n"
            "from solweig_core import dispatch, request;\n"
            "req = request.SolveRequestView(\n"
            "    logical_domain_id='d', rows=8, cols=6, origin_x_m=0.0, "
            "origin_y_m=0.0, pixel_size_m=1.0,\n"
            "    read_window=request.PhysicalWindow(0, 8, 0, 6),\n"
            "    write_window=request.PhysicalWindow(1, 3, 1, 3),\n"
            "    time=request.TimeCoverage(0, None, 4),\n"
            "    profile_id='canonical_cpu_v1', device='cpu');\n"
            "result = dispatch.resolve(req);\n"
            "assert result.ready and result.plan.device == 'cpu';\n"
            "cuda_req = request.SolveRequestView(\n"
            "    logical_domain_id='d', rows=8, cols=6, origin_x_m=0.0, "
            "origin_y_m=0.0, pixel_size_m=1.0,\n"
            "    read_window=request.PhysicalWindow(0, 8, 0, 6),\n"
            "    write_window=request.PhysicalWindow(1, 3, 1, 3),\n"
            "    time=request.TimeCoverage(0, None, 4),\n"
            "    profile_id='canonical_cpu_v1', device='cuda');\n"
            "blocked = dispatch.resolve(cuda_req);\n"
            "assert blocked.status.value == 'blocked';\n"
            "assert blocked.refusal.reason.value == 'cuda_unavailable';\n"
            "print('ok')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_core_sources_contain_no_torch_import(self):
        # AST-level check: docstrings may MENTION torch (explaining why it
        # is banned); actual import statements may not exist. The four T02
        # modules are scanned strictly. profile.py (T01-owned) has one
        # DELIBERATE guarded ``import torch`` inside try/except — it
        # certifies torch contexts, it never computes with them — pinned
        # separately below.
        import ast

        for module in ("abi.py", "request.py", "status.py", "dispatch.py"):
            tree = ast.parse((REPO_ROOT / "solweig_core" / module).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    assert name.split(".")[0] != "torch", (
                        f"{module} imports torch ({name})"
                    )

    def test_profile_torch_import_stays_guarded_and_optional(self):
        # T01's certification module observes torch to RECORD it; the import
        # must remain inside a try/except so a torch-free runtime (T14
        # packaging) still certifies origin. If this ever changes, the
        # runtime subprocess witnesses above will fail too.
        import ast

        tree = ast.parse((REPO_ROOT / "solweig_core" / "profile.py").read_text())
        torch_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "torch" for alias in node.names)
        ]
        assert len(torch_imports) == 1, "profile.py torch import count changed"
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                # child-id -> owning parent, first-wins (BFS order records a
                # statement's owning block before deeper relationships)
                parents.setdefault(id(child), parent)
        for node in torch_imports:
            parent = parents.get(id(node))
            assert isinstance(parent, ast.Try), (
                "profile.py's torch import must stay inside a try/except block"
            )
            assert parent.handlers, "the guarding except clause was removed"


# ---------------------------------------------------------------------------
# construction + basic contract
# ---------------------------------------------------------------------------
class TestFromNumpy:
    def test_view_records_owner_dtype_shape_strides_verbatim(self):
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        assert view.owner is array
        assert view.dtype == np.dtype("<f4")
        assert view.shape == (4, 6)
        assert view.strides == array.strides  # bytes, verbatim
        assert view.global_origin == (0, 0)
        assert view.residency == "cpu"
        assert view.is_contiguous()

    def test_writable_owner_defaults_to_writable_view(self):
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        assert view.read_only is False

    def test_read_only_owner_forces_read_only_view(self):
        array = make_grid()
        array.setflags(write=False)
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        assert view.read_only is True

    def test_non_ndarray_owner_refused(self):
        with pytest.raises(AbiValidationError) as exc:
            ArrayView.from_numpy([1, 2, 3], logical_domain_id=DOMAIN)
        assert exc.value.refusal.reason is RefusalReason.ABI_INVALID

    def test_empty_domain_id_refused(self):
        with pytest.raises(AbiValidationError):
            ArrayView.from_numpy(make_grid(), logical_domain_id="")

    def test_to_numpy_borrows_no_copy(self):
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        assert view.to_numpy() is array


# ---------------------------------------------------------------------------
# RED witness: wrong residency fails ABI validation
# ---------------------------------------------------------------------------
class TestResidency:
    def test_cuda_residency_buffer_fails_validation(self):
        # __post_init__ validates: the lying construction itself refuses.
        array = make_grid()
        with pytest.raises(AbiValidationError) as exc:
            ArrayView(
                owner=array,
                dtype=array.dtype,
                shape=tuple(array.shape),
                strides=tuple(array.strides),
                global_origin=(0, 0),
                logical_domain_id=DOMAIN,
                residency="cuda",
            )
        assert "residency" in str(exc.value)
        assert exc.value.refusal.reason is RefusalReason.ABI_INVALID

    def test_unknown_residency_fails_validation(self):
        array = make_grid()
        with pytest.raises(AbiValidationError):
            ArrayView(
                owner=array,
                dtype=array.dtype,
                shape=tuple(array.shape),
                strides=tuple(array.strides),
                global_origin=(0, 0),
                logical_domain_id=DOMAIN,
                residency="tpu",
            )

    def test_residency_is_always_cpu_from_the_public_constructor(self):
        view = ArrayView.from_numpy(make_grid(), logical_domain_id=DOMAIN)
        assert view.residency == core_abi.RESIDENCY_CPU


# ---------------------------------------------------------------------------
# RED witness: stale borrowed buffer
# ---------------------------------------------------------------------------
class TestStaleBorrow:
    def test_view_keeps_owner_alive(self):
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        ref = weakref.ref(array)
        del array
        assert ref() is not None  # the view's owner reference pins it
        assert view.to_numpy().shape == (4, 6)

    def test_frozen_view_detects_post_creation_mutation(self):
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN, freeze=True)
        array[0, 0] = np.float32(999.0)
        with pytest.raises(StaleBufferError) as exc:
            view.assert_not_mutated()
        assert exc.value.refusal.reason is RefusalReason.STALE_BUFFER
        assert "frozen_sha256" in (exc.value.refusal.context or {})

    def test_frozen_view_passes_when_owner_untouched(self):
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN, freeze=True)
        view.assert_not_mutated()  # no raise

    def test_mutation_through_a_different_view_of_the_same_owner_detected(self):
        array = make_grid()
        frozen = ArrayView.from_numpy(array, logical_domain_id=DOMAIN, freeze=True)
        other = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        other.to_numpy()[1, 1] = np.float32(-1.0)
        with pytest.raises(StaleBufferError):
            frozen.assert_not_mutated()

    def test_unfrozen_assert_refuses_loudly(self):
        view = ArrayView.from_numpy(make_grid(), logical_domain_id=DOMAIN)
        with pytest.raises(AbiError, match="never frozen"):
            view.assert_not_mutated()

    def test_documented_allowed_mutation_for_unfrozen_views(self):
        # The seam's deliberate policy: caller-owned MUTABLE planes (the
        # canopy) may be mutated through the boundary by the legacy path
        # (torch.from_numpy shares memory; the compose clamp writes through
        # for negative canopy). No staleness machinery engages unless a
        # view was explicitly frozen.
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        array[:] = np.float32(7.0)
        assert view.to_numpy()[0, 0] == np.float32(7.0)  # visible, allowed


# ---------------------------------------------------------------------------
# RED witness: noncontiguous input never confused as contiguous
# ---------------------------------------------------------------------------
class TestNonContiguity:
    def test_negative_stride_view_is_not_contiguous(self):
        array = make_grid()
        view = ArrayView.from_numpy(array[::-1, ::-1], logical_domain_id=DOMAIN)
        assert view.strides == array[::-1, ::-1].strides
        assert not view.is_contiguous()

    def test_row_sliced_view_is_not_contiguous(self):
        array = make_grid()
        view = ArrayView.from_numpy(array[::2], logical_domain_id=DOMAIN)
        assert view.shape == (2, 6)
        assert not view.is_contiguous()

    def test_column_sliced_view_is_not_contiguous(self):
        array = make_grid()
        view = ArrayView.from_numpy(array[:, ::2], logical_domain_id=DOMAIN)
        assert view.shape == (4, 3)
        assert not view.is_contiguous()

    def test_require_contiguous_refuses_strided_with_typed_error(self):
        array = make_grid()
        view = ArrayView.from_numpy(array[::-1], logical_domain_id=DOMAIN)
        with pytest.raises(NonContiguousError) as exc:
            view.require_contiguous()
        assert exc.value.refusal.reason is RefusalReason.NON_CONTIGUOUS
        assert "to_contiguous" in str(exc.value)  # the sanctioned escape hatch

    def test_require_contiguous_passes_contiguous_view(self):
        view = ArrayView.from_numpy(make_grid(), logical_domain_id=DOMAIN)
        assert view.require_contiguous() is view

    def test_boundary_copy_preserves_dtype_and_logical_order(self):
        array = make_grid(seed=3)
        view = ArrayView.from_numpy(array[::-1, ::-1], logical_domain_id=DOMAIN)
        copied = view.to_contiguous()
        assert copied is not view
        assert copied.is_contiguous()
        assert copied.dtype == view.dtype == np.dtype("<f4")
        assert copied.same_logical_values(view)
        np.testing.assert_array_equal(copied.to_numpy(), array[::-1, ::-1])

    def test_transpose_roundtrip_preserves_values_by_logical_index(self):
        array = make_grid(seed=4)
        transposed = ArrayView.from_numpy(array.T, logical_domain_id=DOMAIN)
        assert not transposed.is_contiguous()
        assert transposed.shape == (6, 4)
        back = ArrayView.from_numpy(
            np.ascontiguousarray(transposed.to_numpy().T), logical_domain_id=DOMAIN
        )
        assert back.same_logical_values(ArrayView.from_numpy(array, logical_domain_id=DOMAIN))
        for r in range(4):
            for c in range(6):
                assert transposed.to_numpy()[c, r] == array[r, c]

    def test_same_logical_values_ignores_strides_not_values(self):
        array = make_grid(seed=5)
        strided = ArrayView.from_numpy(array[::-1], logical_domain_id=DOMAIN)
        direct = ArrayView.from_numpy(np.ascontiguousarray(array[::-1]), logical_domain_id=DOMAIN)
        assert strided.same_logical_values(direct)
        perturbed = np.ascontiguousarray(array[::-1])
        perturbed[0, 0] = np.nextafter(perturbed[0, 0], np.float32(np.inf))
        assert not strided.same_logical_values(
            ArrayView.from_numpy(perturbed, logical_domain_id=DOMAIN)
        )

    def test_zero_stride_broadcast_memory_refused(self):
        array = make_grid(rows=1)
        broadcast = np.broadcast_to(array, (4, 6))
        with pytest.raises(AbiValidationError, match="zero byte stride"):
            ArrayView.from_numpy(broadcast, logical_domain_id=DOMAIN)

    def test_size_one_axes_tolerate_any_stride(self):
        array = make_grid(rows=1)  # axis 0 has size 1
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN)
        view.validate()  # no raise: degenerate axes carry no overlap risk


# ---------------------------------------------------------------------------
# RED witness: declared metadata that lies about the owner
# ---------------------------------------------------------------------------
class TestLyingMetadata:
    def test_declared_shape_mismatch_rejected(self):
        array = make_grid()
        with pytest.raises(AbiValidationError, match="declared shape"):
            ArrayView(
                owner=array,
                dtype=array.dtype,
                shape=(5, 6),
                strides=tuple(array.strides),
                global_origin=(0, 0),
                logical_domain_id=DOMAIN,
            )

    def test_declared_strides_mismatch_rejected(self):
        array = make_grid()
        with pytest.raises(AbiValidationError, match="strides"):
            ArrayView(
                owner=array,
                dtype=array.dtype,
                shape=(4, 6),
                strides=(99, 4),
                global_origin=(0, 0),
                logical_domain_id=DOMAIN,
            )

    def test_declared_dtype_mismatch_rejected(self):
        array = make_grid()
        with pytest.raises(AbiValidationError, match="dtype"):
            ArrayView(
                owner=array,
                dtype=np.dtype("<f8"),
                shape=(4, 6),
                strides=tuple(array.strides),
                global_origin=(0, 0),
                logical_domain_id=DOMAIN,
            )

    def test_origin_rank_mismatch_rejected(self):
        array = make_grid()
        with pytest.raises(AbiValidationError, match="global_origin rank"):
            ArrayView(
                owner=array,
                dtype=array.dtype,
                shape=(4, 6),
                strides=tuple(array.strides),
                global_origin=(0,),
                logical_domain_id=DOMAIN,
            )

    def test_negative_origin_rejected(self):
        array = make_grid()
        with pytest.raises(AbiValidationError, match="negative global origin"):
            ArrayView(
                owner=array,
                dtype=array.dtype,
                shape=(4, 6),
                strides=tuple(array.strides),
                global_origin=(-1, 0),
                logical_domain_id=DOMAIN,
            )

    def test_writable_borrow_of_read_only_owner_rejected(self):
        array = make_grid()
        array.setflags(write=False)
        with pytest.raises(AbiValidationError, match="read-only owner"):
            ArrayView(
                owner=array,
                dtype=array.dtype,
                shape=(4, 6),
                strides=tuple(array.strides),
                global_origin=(0, 0),
                logical_domain_id=DOMAIN,
                read_only=False,
            )

    def test_stricter_read_only_over_writable_owner_allowed(self):
        array = make_grid()
        view = ArrayView.from_numpy(array, logical_domain_id=DOMAIN, read_only=True)
        assert view.read_only is True
        view.validate()


# ---------------------------------------------------------------------------
# describe (run-record surface)
# ---------------------------------------------------------------------------
class TestDescribe:
    def test_describe_quotes_the_abi_fields(self):
        view = ArrayView.from_numpy(
            make_grid()[::2], global_origin=(10, 20), logical_domain_id=DOMAIN
        )
        described = view.describe()
        assert described["global_origin"] == [10, 20]
        assert described["logical_domain_id"] == DOMAIN
        assert described["residency"] == "cpu"
        assert described["read_only"] is False
        assert described["contiguous"] is False
        assert described["dtype"] == "<f4"
