# SPDX-License-Identifier: GPL-3.0-only
"""The TRUE torch-free runtime facade (T14, DESIGN.ko.md 5.4).

What the RUNTIME is: replay of cached known-site constants plus the
torch-free recompute lanes — the numba full solve over a frozen capture
(:mod:`solweig_core.numba_cpu.full_solve`), met-edit recomputation, the
fallback router, and the general step-table producer (R3,
:mod:`solweig_core.numba_cpu.step_table_gen`). Everything imports lazily:
importing this module costs stdlib-only; the numba stack loads on first
use.

What the RUNTIME is NOT: an oracle. These operations belong to the oracle
environment (the dev env with torch + solweig_gpu) and are REFUSED here,
typed and loud — never a silent import, never a bare
``ModuleNotFoundError``:

* the ``legacy-torch-cpu`` backend — torch kernels
  (:class:`OracleBackendRefusal`);
* capture REGENERATION — geometry edits, new sites/scales: the frozen
  planes are oracle products; the runtime replays captures, it never
  rebuilds them (:class:`CaptureRegenerationRefusal`);
* unseen-scene amplitude-policy computation — scene statistics the oracle
  selects with torch (:class:`UnseenSceneRefusal`);
* imported-origin profile capture — :func:`solweig_core.profile.
  capture_imported_origin` raises
  :class:`solweig_core.profile.OracleEnvironmentError` (R1); the
  torch-free alternative is :func:`solweig_core.profile.snapshot_tree`.

Unseen ANGLES are explicitly NOT refused: the general step-table producer
is torch-free and runtime-supported (:func:`step_table`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "RUNTIME_BACKEND",
    "ORACLE_BACKENDS",
    "RuntimeRefusal",
    "OracleBackendRefusal",
    "CaptureRegenerationRefusal",
    "UnseenSceneRefusal",
    "Runtime",
    "require_runtime_backend",
    "step_table",
    "runtime_dependency_report",
]

#: the one backend the runtime executes.
RUNTIME_BACKEND = "numba-full-solve"

#: dispatch-registered CPU backends that need the oracle environment.
ORACLE_BACKENDS = ("legacy-torch-cpu",)


class RuntimeRefusal(RuntimeError):
    """Base typed refusal: this operation is outside the torch-free
    runtime's contract. Never caught here — the caller surfaces it."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class OracleBackendRefusal(RuntimeRefusal):
    """A torch-backend execution was requested in the torch-free runtime."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(
            f"backend {backend!r} requires the ORACLE environment (torch "
            "kernels); the torch-free runtime executes only "
            f"{RUNTIME_BACKEND!r}. Install/run the oracle extras in the "
            "dev environment, or keep the runtime on the native backend — "
            "it never falls back to torch silently."
        )


class CaptureRegenerationRefusal(RuntimeRefusal):
    """A capture regeneration (geometry edit / new scene or scale) was
    routed into the torch-free runtime."""

    def __init__(self, decision: Any = None) -> None:
        self.decision = decision
        detail = (
            ""
            if decision is None
            else f" Router decision: {getattr(decision, 'reason', '')}"
        )
        super().__init__(
            "capture regeneration is an ORACLE-environment operation: the "
            "frozen planes (march bits, packed cubes, frozen "
            "transcendentals) are products of the torch full-domain "
            "solve. The torch-free runtime replays cached captures and "
            "refuses to rebuild them. Regenerate the capture in the dev "
            "environment, then bring the new capture directory here."
            + detail
        )


class UnseenSceneRefusal(RuntimeRefusal):
    """Scene statistics (amplitude policy) were requested for a scene the
    runtime has no capture-side record for."""

    def __init__(self, what: str) -> None:
        super().__init__(
            f"{what} for an unseen scene is an ORACLE-environment "
            "operation (the selectors consume torch scene tensors); the "
            "runtime consumes amplitude policies RECORDED in captures. "
            "Compute the policy in the dev environment and ship it with "
            "the capture."
        )


def require_runtime_backend(preferred: str | None = None) -> str:
    """Resolve the backend the runtime will execute.

    ``None``/``RUNTIME_BACKEND`` -> the native backend. Anything in
    :data:`ORACLE_BACKENDS` raises :class:`OracleBackendRefusal`;
    unknown names raise ``ValueError`` (never a guess).
    """
    if preferred is None or preferred == RUNTIME_BACKEND:
        return RUNTIME_BACKEND
    if preferred in ORACLE_BACKENDS:
        raise OracleBackendRefusal(preferred)
    raise ValueError(
        f"unknown backend {preferred!r}; runtime backends: "
        f"{[RUNTIME_BACKEND]}; oracle-only backends: {list(ORACLE_BACKENDS)}"
    )


def step_table(
    kernel_variant: str,
    azimuth_deg: float,
    altitude_deg: float,
    scale: float,
    rows: int,
    cols: int,
    amplitude: float,
    **kwargs: Any,
):
    """General torch-free step-table producer (R3) — runtime-supported.

    See :func:`solweig_core.numba_cpu.step_table_gen.build_table_general`.
    Unseen angles are a runtime capability; the AMPLITUDE policy for an
    unseen scene is not (use :class:`UnseenSceneRefusal` path instead).
    """
    from solweig_core.numba_cpu.step_table_gen import build_table_general

    return build_table_general(
        kernel_variant,
        azimuth_deg,
        altitude_deg,
        scale,
        rows,
        cols,
        amplitude,
        **kwargs,
    )


def runtime_dependency_report() -> dict[str, Any]:
    """The runtime's closed third-party dependency set (audit-derived).

    numpy + numba (+ llvmlite, and scipy for sparse-work helpers) with
    numba's own transitives (coverage, yaml). torch, GDAL/rasterio,
    matplotlib, netCDF4 and the compiler toolchain are NOT runtime
    dependencies — they are dev/oracle/preprocess extras.
    """
    return {
        "runtime_backend": RUNTIME_BACKEND,
        "third_party": ["numpy", "numba", "llvmlite", "scipy"],
        "numba_transitives": ["coverage", "yaml"],
        "excluded_from_runtime": [
            "torch",
            "GDAL",
            "rasterio",
            "matplotlib",
            "netCDF4",
            "compiler toolchain (AOT builders)",
        ],
        "oracle_only_operations": [
            "legacy-torch-cpu backend",
            "capture regeneration (geometry edits, new sites/scales)",
            "unseen-scene amplitude policy",
            "imported-origin profile capture (use snapshot_tree)",
        ],
    }


class Runtime:
    """One cached site's torch-free runtime session.

    Construction is deliberately cheap (path checks only): the capture
    set and the numba stack load lazily so ``Runtime(...)`` import cost
    never dominates startup. The heavy solve entry is :meth:`solve`
    (:func:`solweig_core.numba_cpu.full_solve.full_solve_capture`).
    """

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        profile: str = "canonical_cpu_v1",
        latitude: float,
    ) -> None:
        self.capture_dir = Path(capture_dir)
        self.profile = profile
        self.latitude = float(latitude)
        if not (self.capture_dir / "t00.npz").is_file():
            raise RuntimeRefusal(
                f"capture dir {self.capture_dir} has no t00.npz — not a "
                "capture set; build one in the oracle environment first"
            )
        self._cap_set: Any = None

    # -- lazy capture access ------------------------------------------------

    def cap_set(self):
        from solweig_core.numba_cpu import full_solve

        if self._cap_set is None:
            self._cap_set = full_solve.CaptureSet(
                cap_dir=str(self.capture_dir), profile=self.profile,
                latitude=self.latitude,
            )
        return self._cap_set

    # -- routing ------------------------------------------------------------

    def route(
        self,
        *,
        geometry_changed: bool,
        r0: int,
        n_window: int,
        anchor_decision: Any = None,
        collect_ts: Sequence[int] | None = None,
        memory_budget_bytes: int = 2 * 1024 ** 2,
        costs: Any = None,
    ):
        """Route a fallback request; REGENERATE becomes a typed refusal.

        The router's decision table is correctness-first (geometry veto,
        anchor policy, then measured cost); only its REGENERATE lane is
        outside the runtime's contract — that lane rebuilds frozen planes
        and is refused here instead of silently mis-executing.
        """
        from solweig_core.numba_cpu import fallback_router

        decision = fallback_router.route_fallback(
            geometry_changed=geometry_changed,
            r0=r0,
            n_window=n_window,
            anchor_decision=anchor_decision,
            collect_ts=collect_ts,
            memory_budget_bytes=memory_budget_bytes,
            costs=costs,
        )
        if decision.route == fallback_router.ROUTE_REGENERATE:
            raise CaptureRegenerationRefusal(decision)
        return decision

    # -- execution ----------------------------------------------------------

    def solve(
        self,
        met_series: Sequence[Any] | None = None,
        *,
        warm_state: Any = None,
        warm_fingerprint: Mapping[str, str] | None = None,
        r0: int = 0,
        publish_anchor: bool = True,
        publish_anchor_at: Sequence[int] | None = None,
    ):
        """Cold or warm full solve over the cached capture (torch-free).

        Arguments pass through to
        :func:`solweig_core.numba_cpu.full_solve.full_solve_capture`;
        geometry edits must NOT come through here (they route to
        :class:`CaptureRegenerationRefusal` via :meth:`route`).
        """
        from solweig_core.numba_cpu import full_solve

        return full_solve.full_solve_capture(
            self.cap_set(),
            met_series,
            profile=self.profile,
            warm_state=warm_state,
            warm_fingerprint=warm_fingerprint,
            r0=r0,
            publish_anchor=publish_anchor,
            publish_anchor_at=publish_anchor_at,
        )

    # -- refused oracle operations ------------------------------------------

    def amplitude_policy(self, *args: Any, **kwargs: Any) -> None:
        """Refused: unseen-scene amplitude selection is oracle-side."""
        raise UnseenSceneRefusal(
            "amplitude-policy computation"
        )

    def regenerate(self, *args: Any, **kwargs: Any) -> None:
        """Refused: capture regeneration is oracle-side."""
        raise CaptureRegenerationRefusal(None)
