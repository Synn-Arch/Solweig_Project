"""Tests for solweig_core.dispatch — explicit device/backend resolution.

RED witnesses (TASKS T02):

* ``device='cuda'`` on this host returns a typed BLOCKED result — plan is
  ``None``, status is BLOCKED, the refusal reason is
  ``cuda_unavailable`` — and consuming it raises
  :class:`DispatchBlockedError`. It is NEVER answered with a CPU plan
  marked success: a silent CPU-as-success substitute would publish CPU-bit
  results under a CUDA request.
* availability never changes the answer (the module cannot even see
  torch; see the fake-torch witness in test_abi.py).
* dispatch never auto-selects: a READY plan's device is always the
  REQUEST's device, and the legacy roundtrip backend is recorded as a
  fallback reason, not hidden.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_core import dispatch as core_dispatch  # noqa: E402
from solweig_core.request import PhysicalWindow, SolveRequestView, TimeCoverage  # noqa: E402
from solweig_core.status import (  # noqa: E402
    DispatchBlockedError,
    DispatchStatus,
    FallbackReason,
    RefusalReason,
)


def make_request(device="cpu", **overrides):
    kwargs = dict(
        logical_domain_id="site:test:100x80",
        rows=100,
        cols=80,
        origin_x_m=1000.0,
        origin_y_m=2000.0,
        pixel_size_m=2.0,
        read_window=PhysicalWindow(0, 100, 0, 80),
        write_window=PhysicalWindow(10, 20, 10, 20),
        time=TimeCoverage(0, None, 24),
        profile_id="canonical_cpu_v1",
        device=device,
    )
    kwargs.update(overrides)
    return SolveRequestView(**kwargs)


class TestCpuResolution:
    def test_cpu_request_is_ready_with_legacy_backend(self):
        result = core_dispatch.resolve(make_request())
        assert result.status is DispatchStatus.READY
        assert result.ready
        assert result.plan is not None
        assert result.plan.device == "cpu"
        assert result.plan.backend == "legacy-torch-cpu"

    def test_legacy_roundtrip_is_recorded_not_hidden(self):
        result = core_dispatch.resolve(make_request())
        assert result.plan.fallback_reason is FallbackReason.LEGACY_ROUNDTRIP

    def test_plan_echoes_profile_and_domain(self):
        result = core_dispatch.resolve(make_request())
        assert result.plan.profile_id == "canonical_cpu_v1"
        assert result.plan.logical_domain_id == "site:test:100x80"

    def test_raise_if_blocked_returns_the_plan_when_ready(self):
        result = core_dispatch.resolve(make_request())
        assert result.raise_if_blocked() is result.plan

    def test_supported_devices_is_a_static_vocabulary(self):
        assert core_dispatch.supported_devices() == ("cpu", "cuda")


class TestCudaBlocked:
    def test_cuda_request_is_blocked_with_typed_refusal(self):
        result = core_dispatch.resolve(make_request(device="cuda"))
        assert result.status is DispatchStatus.BLOCKED
        assert not result.ready
        assert result.plan is None  # never a CPU plan in disguise
        assert result.refusal.reason is RefusalReason.CUDA_UNAVAILABLE
        assert "never re-targets" in result.refusal.detail

    def test_consuming_a_blocked_result_raises_typed_error(self):
        result = core_dispatch.resolve(make_request(device="cuda"))
        with pytest.raises(DispatchBlockedError) as exc:
            result.raise_if_blocked()
        assert exc.value.refusal.reason is RefusalReason.CUDA_UNAVAILABLE

    def test_blocked_result_describes_itself(self):
        described = core_dispatch.resolve(make_request(device="cuda")).describe()
        assert described["status"] == "blocked"
        assert described["plan"] is None
        assert described["refusal"]["reason"] == "cuda_unavailable"

    def test_cuda_never_falls_back_to_cpu_as_success(self):
        # The precise witness: for a CUDA request there exists NO ready
        # answer whose device is cpu. Scanning every observable field of
        # the result, nothing claims success.
        result = core_dispatch.resolve(make_request(device="cuda"))
        observables = (result.status, result.ready, result.plan)
        assert observables == (DispatchStatus.BLOCKED, False, None)


class TestNoAutoSelection:
    def test_unknown_device_refused_at_the_request_gate_first(self):
        # The public path: the request vocabulary refuses 'hip' before
        # dispatch ever sees it (layered defense).
        with pytest.raises(Exception, match="vocabulary"):
            core_dispatch.resolve(make_request(device="hip"))

    def test_dispatch_level_unknown_device_guard_is_defense_in_depth(self):
        # If a device ever passes request validation but the backend
        # registry does not know it, dispatch refuses — never redirects.
        import solweig_core.dispatch as dispatch_module

        original = dispatch_module._BACKENDS
        dispatch_module._BACKENDS = {"cpu": ("legacy-torch-cpu",)}
        try:
            result = dispatch_module.resolve(make_request(device="cuda"))
        finally:
            dispatch_module._BACKENDS = original
        assert result.status is DispatchStatus.BLOCKED
        assert result.refusal.reason is RefusalReason.UNSUPPORTED_DEVICE
        assert "never guesses" in result.refusal.detail

    def test_ready_plan_device_always_equals_the_requested_device(self):
        # Auto-selection would manifest as plan.device != request.device.
        for device in ("cpu",):
            request = make_request(device=device)
            result = core_dispatch.resolve(request)
            assert result.ready
            assert result.plan.device == request.device

    def test_invalid_request_is_refused_before_device_resolution(self):
        request = make_request(
            read_window=PhysicalWindow(0, 50, 0, 50),
            write_window=PhysicalWindow(90, 95, 70, 75),  # escapes the read window
        )
        with pytest.raises(Exception, match="not contained"):
            core_dispatch.resolve(request)

    def test_dispatch_module_has_no_availability_probe(self):
        # Static witness on CODE ONLY (docstrings may explain why probing
        # is banned): no availability querying of any kind — device choice
        # is declarative.
        import io
        import tokenize

        source = (REPO_ROOT / "solweig_core" / "dispatch.py").read_text()
        code_only = io.StringIO()
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                code_only.write(token.string + " ")
        for banned in ("is_available", "device_count", "get_device"):
            assert banned not in code_only.getvalue(), (
                f"dispatch.py code contains {banned!r}"
            )
