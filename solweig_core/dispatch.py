"""Explicit device/backend resolution (DESIGN.ko.md 5.2, TASKS T02).

The request DECLARES its device; :func:`resolve` answers with a
:class:`DispatchPlan` (READY) or a :class:`TypedRefusal` (BLOCKED). There
is no availability-driven selection anywhere in this module: solweig_core
never imports torch and therefore never asks ``torch.cuda.is_available()``
— a CPU-target request uses CPU buffers only, even on a host where a GPU
is visible to the legacy tree, and a CUDA request is BLOCKED with a typed
refusal on this host rather than silently re-targeted to CPU (a silent
CPU-as-success substitute would publish CPU-bit results under a CUDA
request and defeat the profile contract).

The only backend today is the legacy roundtrip: the T02 adapter converts
buffers at the ABI boundary and runs the ORIGINAL numerical path
(``solweig_gpu.incremental.solver.solve_window``) unchanged. New backends
(numba CPU lanes, native) register in :data:`_BACKENDS` as they land
(T04+), still keyed by declared device — never by probe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solweig_core.request import SolveRequestView
from solweig_core.status import (
    DispatchStatus,
    FallbackReason,
    RefusalReason,
    TypedRefusal,
)

__all__ = [
    "DispatchPlan",
    "DispatchResult",
    "supported_devices",
    "resolve",
]

#: Backends per DECLARED device. The T02 backend routes through the legacy
#: torch-CPU math behind the ABI boundary (recorded as
#: ``FallbackReason.LEGACY_ROUNDTRIP``); the T11 backend is the torch-free
#: numba CPU full-solve lane (``solweig_core.numba_cpu.full_solve``, first
#: backend of the fallback router). 'cuda' has no entry on this host and
#: that absence IS the refusal.
_BACKENDS: dict[str, tuple[str, ...]] = {
    "cpu": ("legacy-torch-cpu", "numba-full-solve"),
    "cuda": (),  # no CUDA backend on this host; never silently CPU
}


@dataclass(frozen=True)
class DispatchPlan:
    """A resolved execution plan for one request."""

    device: str
    backend: str
    profile_id: str
    logical_domain_id: str
    #: Why this plan routes where it does (recorded, never silent).
    fallback_reason: FallbackReason | None = None
    status: DispatchStatus = DispatchStatus.READY

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "backend": self.backend,
            "profile_id": self.profile_id,
            "logical_domain_id": self.logical_domain_id,
            "fallback_reason": (
                self.fallback_reason.value if self.fallback_reason else None
            ),
            "status": self.status.value,
        }


@dataclass(frozen=True)
class DispatchResult:
    """READY with a plan, or BLOCKED with a typed refusal — never both.

    A caller that ignores ``status`` and reads ``plan`` on a BLOCKED result
    gets ``None``; consuming a BLOCKED result as success requires an
    explicit (buggy) act. The adapter seam raises
    :class:`~solweig_core.status.DispatchBlockedError` on BLOCKED.
    """

    status: DispatchStatus
    plan: DispatchPlan | None = None
    refusal: TypedRefusal | None = None

    @property
    def ready(self) -> bool:
        return self.status is DispatchStatus.READY

    def raise_if_blocked(self) -> DispatchPlan:
        if self.status is not DispatchStatus.READY or self.plan is None:
            from solweig_core.status import DispatchBlockedError

            raise DispatchBlockedError(
                self.refusal
                or TypedRefusal(RefusalReason.UNKNOWN_BACKEND, "dispatch blocked")
            )
        return self.plan

    def describe(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "plan": self.plan.describe() if self.plan else None,
            "refusal": self.refusal.to_dict() if self.refusal else None,
        }


def supported_devices() -> tuple[str, ...]:
    """Devices solweig_core can even consider — a static vocabulary."""
    return tuple(sorted(_BACKENDS))


def resolve(request: SolveRequestView,
            prefer_backend: str | None = None) -> DispatchResult:
    """Resolve the DECLARED device of a validated request.

    Rules, in order:

    * the request must already have passed ``validate()`` (re-validated
      here defensively);
    * ``'cpu'`` resolves to the first CPU backend — the legacy roundtrip
      by default, recorded as such. ``prefer_backend`` (T11 fallback
      router seam) EXPLICITLY names a registered backend of the declared
      device — never a probe, never a silent re-target (an unknown
      preference is BLOCKED with :attr:`RefusalReason.UNKNOWN_BACKEND`);
    * ``'cuda'`` is BLOCKED on this host with reason
      :attr:`RefusalReason.CUDA_UNAVAILABLE`. It is NEVER answered with a
      CPU plan: availability probing and silent re-targeting are both
      absent by construction (this module cannot even see torch).
    """
    request.validate()

    if request.device not in _BACKENDS:
        return DispatchResult(
            DispatchStatus.BLOCKED,
            refusal=TypedRefusal(
                RefusalReason.UNSUPPORTED_DEVICE,
                f"device {request.device!r} is outside the dispatch vocabulary "
                f"{sorted(_BACKENDS)}; dispatch never guesses",
                context={"device": request.device},
            ),
        )

    backends = _BACKENDS[request.device]
    if not backends:
        return DispatchResult(
            DispatchStatus.BLOCKED,
            refusal=TypedRefusal(
                RefusalReason.CUDA_UNAVAILABLE,
                "device 'cuda' was declared but this host provides no CUDA "
                "backend to solweig_core; refusing loudly — solweig_core never "
                "probes CUDA availability and never re-targets a CUDA request "
                "to CPU",
                context={"device": request.device, "logical_domain_id": request.logical_domain_id},
            ),
        )

    backend = backends[0]
    if prefer_backend is not None:
        if prefer_backend not in backends:
            return DispatchResult(
                DispatchStatus.BLOCKED,
                refusal=TypedRefusal(
                    RefusalReason.UNKNOWN_BACKEND,
                    f"preferred backend {prefer_backend!r} is not registered "
                    f"for device {request.device!r} (registered: "
                    f"{list(backends)}); dispatch never guesses",
                    context={"device": request.device,
                             "prefer_backend": prefer_backend},
                ),
            )
        backend = prefer_backend
    fallback = (
        FallbackReason.LEGACY_ROUNDTRIP
        if backend.startswith("legacy")
        else None
    )
    return DispatchResult(
        DispatchStatus.READY,
        plan=DispatchPlan(
            device=request.device,
            backend=backend,
            profile_id=request.profile_id,
            logical_domain_id=request.logical_domain_id,
            fallback_reason=fallback,
        ),
    )
