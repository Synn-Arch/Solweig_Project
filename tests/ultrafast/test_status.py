"""Tests for solweig_core.status — typed refusal/fallback vocabulary.

T02 contract (TASKS.ko.md): no implicit device choice, no silent coercion.
Every refusal carries a reason code; the enums' string values are frozen
(run records and comparison reports quote them verbatim), and the typed
error hierarchy lets callers branch on the refusal object instead of
parsing message text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_core.status import (  # noqa: E402
    AbiError,
    AbiValidationError,
    DispatchBlockedError,
    DispatchStatus,
    FallbackReason,
    NonContiguousError,
    RefusalReason,
    RequestValidationError,
    SolweigCoreError,
    StaleBufferError,
    TypedRefusal,
)


class TestEnums:
    def test_refusal_reason_values_are_frozen_strings(self):
        assert RefusalReason.CUDA_UNAVAILABLE.value == "cuda_unavailable"
        assert RefusalReason.UNSUPPORTED_DEVICE.value == "unsupported_device"
        assert RefusalReason.ABI_INVALID.value == "abi_invalid"
        assert RefusalReason.REQUEST_INVALID.value == "request_invalid"
        assert RefusalReason.STALE_BUFFER.value == "stale_buffer"
        assert RefusalReason.NON_CONTIGUOUS.value == "non_contiguous"

    def test_dispatch_status_values(self):
        assert DispatchStatus.READY.value == "ready"
        assert DispatchStatus.BLOCKED.value == "blocked"

    def test_fallback_reason_records_legacy_roundtrip(self):
        # The T02 seam routes through the ORIGINAL math behind the ABI
        # boundary; that routing is recorded, never silent.
        assert FallbackReason.LEGACY_ROUNDTRIP.value == "legacy_roundtrip"

    def test_enum_roundtrip_through_json(self):
        import json

        assert json.loads(json.dumps(RefusalReason.STALE_BUFFER.value)) == "stale_buffer"
        assert RefusalReason("stale_buffer") is RefusalReason.STALE_BUFFER


class TestTypedRefusal:
    def test_to_dict_is_json_stable(self):
        refusal = TypedRefusal(
            RefusalReason.CUDA_UNAVAILABLE,
            "no CUDA backend on this host",
            context={"device": "cuda"},
        )
        data = refusal.to_dict()
        assert data["reason"] == "cuda_unavailable"
        assert data["context"] == {"device": "cuda"}

    def test_str_carries_reason_code(self):
        refusal = TypedRefusal(RefusalReason.ABI_INVALID, "detail here")
        assert "abi_invalid" in str(refusal)
        assert "detail here" in str(refusal)


class TestErrorHierarchy:
    def test_typed_errors_carry_the_refusal(self):
        for cls, reason in (
            (AbiValidationError, RefusalReason.ABI_INVALID),
            (StaleBufferError, RefusalReason.STALE_BUFFER),
            (NonContiguousError, RefusalReason.NON_CONTIGUOUS),
            (RequestValidationError, RefusalReason.REQUEST_INVALID),
            (DispatchBlockedError, RefusalReason.CUDA_UNAVAILABLE),
        ):
            refusal = TypedRefusal(reason, "x")
            error = cls(refusal)
            assert isinstance(error, SolweigCoreError)
            assert error.refusal is refusal
            assert error.refusal.reason is reason

    def test_abi_error_subtree(self):
        assert issubclass(AbiValidationError, AbiError)
        assert issubclass(StaleBufferError, AbiError)
        assert issubclass(NonContiguousError, AbiError)

    def test_status_module_never_imports_torch(self):
        import subprocess

        code = (
            "import sys;"
            "from solweig_core import status;"
            "assert 'torch' not in sys.modules, 'status imported torch';"
            "print('ok')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout
