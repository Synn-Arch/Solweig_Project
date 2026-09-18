"""Typed refusal / fallback vocabulary (DESIGN.ko.md 5.2/5.3, TASKS T02).

Every refusal in the new core carries a machine-readable reason code, and
every fallback records WHY the stronger path was not taken: no implicit
device choice, no silent coercion. The T02 seam routes a CPU request
through the legacy numerical path — that routing is a *recorded* fallback
(``FallbackReason.LEGACY_ROUNDTRIP``), never a hidden substitution.

This module must stay torch-free (``solweig_core`` never imports torch;
enforced by ``tests/ultrafast/test_abi.py``). It is the leaf of the core
import graph: ``abi``/``request``/``dispatch`` import from here, never the
other way round.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RefusalReason(str, Enum):
    """Machine-readable reason a view/request/dispatch was refused.

    String values are JSON-stable: run records and comparison reports quote
    them verbatim.
    """

    #: The request declared ``device='cuda'`` on a host where solweig_core
    #: has no CUDA device. Refused loudly; NEVER silently re-targeted to CPU
    #: (a silent CPU substitute would publish CPU-bit results under a CUDA
    #: request and defeat the profile contract).
    CUDA_UNAVAILABLE = "cuda_unavailable"
    #: The request declared a device outside the supported vocabulary.
    UNSUPPORTED_DEVICE = "unsupported_device"
    #: A buffer view failed ABI validation (shape/stride/residency/owner).
    ABI_INVALID = "abi_invalid"
    #: A solve request failed structural validation (windows, time coverage,
    #: dtype plan, profile).
    REQUEST_INVALID = "request_invalid"
    #: A frozen (freeze-on-lend) view's owner was mutated after view
    #: creation.
    STALE_BUFFER = "stale_buffer"
    #: The request named a backend solweig_core does not know.
    UNKNOWN_BACKEND = "unknown_backend"
    #: A view required C-contiguity from a strided buffer without a boundary
    #: copy (kernels must handle strides, copy at the boundary, or refuse —
    #: never assume contiguity).
    NON_CONTIGUOUS = "non_contiguous"


class FallbackReason(str, Enum):
    """Why execution used a weaker/older path. Recorded, never silent."""

    #: T02 seam: the core adapter routes the solve through the ORIGINAL
    #: legacy numerical path (torch CPU) behind the ABI boundary. This is
    #: the sanctioned initial backend, not a degradation.
    LEGACY_ROUNDTRIP = "legacy_roundtrip"
    #: Worker routing: a local job was judged unsafe (halo/window guards)
    #: and fell back to the full-tile path.
    FULL_TILE_UNSAFE_LOCAL = "full_tile_unsafe_local"
    #: A future kernel refused (typed) and the batch re-routed to an exact
    #: but slower path. Reserved for T03+ consumers.
    KERNEL_REFUSED = "kernel_refused"


class DispatchStatus(str, Enum):
    """Outcome of an explicit device/backend resolution."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TypedRefusal:
    """A refusal with its reason code and human detail.

    ``context`` carries the identifiers the refusal is about (device name,
    plane name, window) so a log line alone is enough to reproduce it.
    """

    reason: RefusalReason
    detail: str = ""
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "detail": self.detail,
            "context": dict(self.context) if self.context else None,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial __repr__ helper
        ctx = f" context={self.context}" if self.context else ""
        return f"TypedRefusal({self.reason.value}: {self.detail}{ctx})"


# ---------------------------------------------------------------------------
# Typed errors (raised at seams; the enums above ride inside them)
# ---------------------------------------------------------------------------
class SolweigCoreError(RuntimeError):
    """Base class for every typed error raised by solweig_core."""


class AbiError(SolweigCoreError):
    """A buffer/view violated the torch-neutral buffer ABI."""


class AbiValidationError(AbiError):
    """Structural ABI validation failed (owner/shape/strides/residency).

    Carries the :class:`TypedRefusal` so callers can log the reason code
    instead of parsing message text.
    """

    def __init__(self, refusal: TypedRefusal) -> None:
        super().__init__(str(refusal))
        self.refusal = refusal


class StaleBufferError(AbiError):
    """A frozen view detected mutation of its owner after view creation."""

    def __init__(self, refusal: TypedRefusal) -> None:
        super().__init__(str(refusal))
        self.refusal = refusal


class NonContiguousError(AbiError):
    """A strided buffer was handed to a caller that cannot honor strides.

    The caller must either handle the strides, take a boundary copy
    (``ArrayView.to_contiguous``), or refuse — treating a strided view as
    contiguous is exactly the bug this error exists to prevent.
    """

    def __init__(self, refusal: TypedRefusal) -> None:
        super().__init__(str(refusal))
        self.refusal = refusal


class RequestValidationError(SolweigCoreError):
    """A solve request failed structural validation."""

    def __init__(self, refusal: TypedRefusal) -> None:
        super().__init__(str(refusal))
        self.refusal = refusal


class DispatchBlockedError(SolweigCoreError):
    """Raised when an adapter consumed a BLOCKED dispatch result.

    The refusal is attached; message text never replaces the reason code.
    """

    def __init__(self, refusal: TypedRefusal) -> None:
        super().__init__(str(refusal))
        self.refusal = refusal


__all__ = [
    "RefusalReason",
    "FallbackReason",
    "DispatchStatus",
    "TypedRefusal",
    "SolweigCoreError",
    "AbiError",
    "AbiValidationError",
    "StaleBufferError",
    "NonContiguousError",
    "RequestValidationError",
    "DispatchBlockedError",
]
