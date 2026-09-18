# SPDX-License-Identifier: GPL-3.0-only
"""T10 exact sparse temporal replay: thermal state + replay driver +
checkpoint applicability policy (torch-free).

Consumes the T08 fused radiation kernel (import read-only) and mirrors
the CURRENT consumer of the temporal checkpoints —
``PlanExecutor._run_met_warm_sparse_path`` +
``solweig_gpu/incremental/checkpoints.py`` — exactly:

* STATE: the six float32 planes ``Tgmap1``, ``Tgmap1E``, ``Tgmap1S``,
  ``Tgmap1W``, ``Tgmap1N``, ``TgOut1`` plus the carried scalars ``CI``,
  ``firstdaytime``, ``timeadd``, ``Twater``. ``Twater`` has an explicit
  not-established spelling (``None`` — the pre-midnight ``[]``): absent
  on restore means not-established, NEVER a guessed water temperature.
  T08 froze the water mutation into the frozen ``Tg`` planes, so this
  kernel carries ``Twater`` for checkpoint-contract fidelity without
  advancing it; the live midnight recompute belongs to the torch-side
  solver (T11 fallback owns torch-free parity there).
* ``next_step``: the state is AFTER completing timesteps ``0..next_step-1``
  and a replay resumes AT ``next_step`` (checkpoints.py's binding
  spelling). Restores validate it; replays start exactly there.
* SPLIT SEMANTICS: a met edit at the first changed row ``r0`` advances
  ``k..r0`` DISCARDING outputs (the prefix is served from the store) and
  solves ``r0..T`` collecting — ``split_replay_thermal``.
* APPLICABILITY: geometry-history invalidation (the scene lineage: the
  ``composed_scene`` / ``resolved_landcover`` / ``model_parameters``
  digests) is SEPARATE from met-prefix invalidation (the ``met_prefix``
  digest). A checkpoint of a previous scene can never warm-start the
  current geometry however matching its clock position is (DESIGN 11.2);
  a met mismatch is a time-axis boundary property reported at the
  candidate's OWN ``next_step`` — the digest of the CURRENT forcing rows
  ``0..next_step-1``, never the request's ``r0`` (DESIGN 11.2/11.1).
  This module does POLICY on caller-supplied digests only: the ONE
  digest spelling stays ``checkpoints.thermal_fingerprint`` (torch side)
  / the test-side mirrors — no duplicated digest arithmetic here.

On-disk contract: UNCHANGED (SCHEMA_VERSION 2). Everything here is an
in-memory extension: payloads are plain dicts in the checkpoint record's
``thermal_tensors``/``thermal_scalars``/``next_step`` vocabulary, and the
optional per-plane sha256 fence mirrors the on-disk checksum layer for
torn/mixed same-shape state.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .radiation import RadLoopState, RadiationStatic, fused_radiation_timestep

__all__ = [
    "CARRIED_SCALAR_NAMES",
    "GEOMETRY_FINGERPRINT_KEYS",
    "MET_FINGERPRINT_KEYS",
    "THERMAL_PLANE_NAMES",
    "AnchorDecision",
    "CheckpointCandidate",
    "ReplayOutcome",
    "ThermalState",
    "ThermalStateError",
    "capture_thermal_state",
    "classify_anchor",
    "cold_thermal_state",
    "payload_plane_digests",
    "replay_thermal",
    "restore_loop_state",
    "select_warm_anchor",
    "split_replay_thermal",
    "thermal_state_from_payload",
]

#: The six cross-timestep planes (checkpoints.THERMAL_TENSOR_NAMES
#: spelling, verified against utci_process.py:614-619).
THERMAL_PLANE_NAMES = (
    "Tgmap1",
    "Tgmap1E",
    "Tgmap1S",
    "Tgmap1W",
    "Tgmap1N",
    "TgOut1",
)

#: The carried scalars (checkpoints.THERMAL_SCALAR_NAMES spelling).
#: ``Twater`` is optional-with-None semantics (not-established).
CARRIED_SCALAR_NAMES = ("CI", "firstdaytime", "timeadd", "Twater")

#: The fingerprint keys that describe the SCENE LINEAGE (next_step
#: independent). A mismatch here is geometry-history contamination: the
#: checkpoint belongs to a previous scene and can never serve.
GEOMETRY_FINGERPRINT_KEYS = (
    "composed_scene",
    "resolved_landcover",
    "model_parameters",
)

#: The fingerprint key that describes the TIME AXIS: the digest of the
#: forcing rows ``0..next_step-1`` — exactly the rows the captured state
#: consumed.
MET_FINGERPRINT_KEYS = ("met_prefix",)


class ThermalStateError(RuntimeError):
    """A thermal state was refused: structurally invalid, torn/mixed, or
    a replay request that would duplicate/skip a timestep.

    Restore side: missing planes/scalars, dtype drift, shape
    incoherence, grid mismatch, per-plane digest mismatch, coverage-only
    payloads (``next_step=None``). Replay side: collect/stop bounds that
    disagree with the anchor's ``next_step``. Never a silent default —
    the caller falls back to a cold replay.
    """


# ---------------------------------------------------------------------------
# The in-memory thermal state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalState:
    """A validated, snapshot-isolated cross-timestep state.

    ``next_step``: the state is after completing timesteps
    ``0..next_step-1``; a replay resumes AT ``next_step``. ``Twater`` is
    ``None`` for the not-established spelling. Planes are float32
    (rows, cols) C-contiguous copies — the frozen dataclass fields plus
    copy-on-boundary keep snapshots independent of kernel aliasing (the
    T08 night path returns the INPUT plane objects as next-state).
    """

    next_step: int
    firstdaytime: int
    timeadd: float
    CI: float
    Twater: float | None
    Tgmap1: np.ndarray
    Tgmap1E: np.ndarray
    Tgmap1S: np.ndarray
    Tgmap1W: np.ndarray
    Tgmap1N: np.ndarray
    TgOut1: np.ndarray

    def plane(self, name: str) -> np.ndarray:
        if name not in THERMAL_PLANE_NAMES:
            raise ThermalStateError(f"unknown thermal plane {name!r}")
        return getattr(self, name)

    def plane_digests(self) -> dict[str, str]:
        return {
            name: hashlib.sha256(
                np.ascontiguousarray(getattr(self, name))
            ).hexdigest()
            for name in THERMAL_PLANE_NAMES
        }


def _copy_plane(name: str, value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ThermalStateError(
            f"thermal plane {name!r} must be a float32 numpy array, got "
            f"{type(value).__name__}"
        )
    if value.dtype != np.float32:
        raise ThermalStateError(
            f"thermal plane {name!r} must be float32, got {value.dtype!r} "
            "— thermal planes are float32-exact; narrowing/widening "
            "silently is refused"
        )
    if value.ndim != 2:
        raise ThermalStateError(
            f"thermal plane {name!r} must be a (rows, cols) map, got "
            f"ndim={value.ndim}"
        )
    return np.ascontiguousarray(value).copy()


def cold_thermal_state(*, rows: int, cols: int) -> ThermalState:
    """The cold-start state ``run_utci_window`` allocates at time_start=0:
    zero planes, ``CI=1.0``, ``firstdaytime=1``, ``timeadd=0.`` and the
    not-established ``Twater`` (the pre-midnight ``[]``)."""
    planes = {
        name: np.zeros((rows, cols), dtype=np.float32)
        for name in THERMAL_PLANE_NAMES
    }
    return ThermalState(
        next_step=0,
        firstdaytime=1,
        timeadd=0.0,
        CI=1.0,
        Twater=None,
        **planes,
    )


def capture_thermal_state(
    *, loop_state: RadLoopState, next_step: int, twater: float | None
) -> ThermalState:
    """Snapshot a live ``RadLoopState`` (+ ``Twater``) into an immutable
    ``ThermalState`` at ``next_step`` — planes are copied, so later
    kernel steps (which may alias the input planes at night) can never
    mutate a captured checkpoint."""
    if isinstance(next_step, bool) or not isinstance(next_step, (int, np.integer)):
        raise ThermalStateError(
            f"next_step must be a non-negative int, got {next_step!r}"
        )
    if int(next_step) < 0:
        raise ThermalStateError(
            f"next_step must be a non-negative int, got {next_step!r}"
        )
    return ThermalState(
        next_step=int(next_step),
        firstdaytime=int(loop_state.firstdaytime),
        timeadd=float(loop_state.timeadd),
        CI=float(loop_state.CI),
        Twater=None if twater is None else float(twater),
        **{
            name: np.ascontiguousarray(
                getattr(loop_state, name)
            ).copy()
            for name in THERMAL_PLANE_NAMES
        },
    )


def payload_plane_digests(payload: Mapping[str, Any]) -> dict[str, str]:
    """sha256 per required plane over contiguous bytes — the in-memory
    mirror of the on-disk per-tensor checksum (spelled identically to
    ``checkpoints._tensor_checksum``)."""
    out: dict[str, str] = {}
    for name in THERMAL_PLANE_NAMES:
        if name not in payload:
            raise ThermalStateError(
                f"payload is missing the thermal plane {name!r}"
            )
        out[name] = hashlib.sha256(
            np.ascontiguousarray(payload[name])
        ).hexdigest()
    return out


def _scalar_number(payload: Mapping[str, Any], name: str) -> float:
    if name not in payload:
        raise ThermalStateError(
            f"thermal state is missing the carried scalar {name!r}"
        )
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ThermalStateError(
            f"carried scalar {name!r} must be a number, got "
            f"{type(value).__name__}"
        )
    return float(value)


def _scalar_int(payload: Mapping[str, Any], name: str) -> int:
    if name not in payload:
        raise ThermalStateError(
            f"thermal state is missing the carried scalar {name!r}"
        )
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ThermalStateError(
            f"carried scalar {name!r} must be an integer, got "
            f"{type(value).__name__}"
        )
    number = float(value)
    if not number.is_integer():
        raise ThermalStateError(
            f"carried scalar {name!r} must be an integer, got {value!r}"
        )
    return int(number)


def thermal_state_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_plane_digests: Mapping[str, str] | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> ThermalState:
    """Validate a checkpoint-vocabulary payload ATOMICALLY, then build.

    The payload spells the checkpoint record's thermal vocabulary:
    the six plane arrays + ``CI``/``firstdaytime``/``timeadd`` (+ optional
    ``Twater``: absent or ``None`` = the not-established pre-midnight
    state — carried as ``None``, NEVER silently defaulted to a number)
    and ``next_step``. Every check runs BEFORE any construction: a
    refusal leaves nothing half-built and never touches caller state.

    Torn/mixed same-shape state: ``expected_plane_digests`` (see
    :func:`payload_plane_digests`) re-raises the on-disk checksum fence
    in memory — a splice of two same-revision captures is refused here,
    not silently served. ``grid_shape`` additionally pins the plane
    extent to the live solver grid.
    """
    if not isinstance(payload, Mapping):
        raise ThermalStateError(
            f"thermal payload must be a mapping, got {type(payload).__name__}"
        )

    # next_step FIRST: a coverage-only payload (next_step=None, the
    # executor write hook's marker) has no warm state at all.
    raw_next = payload.get("next_step")
    if raw_next is None:
        raise ThermalStateError(
            "payload is coverage-only: next_step is None (a publication "
            "boundary marker with no warm state) — warm start refused, "
            "replay cold"
        )
    if isinstance(raw_next, bool) or not isinstance(
        raw_next, (int, np.integer)
    ):
        raise ThermalStateError(
            f"next_step must be a non-negative int, got {raw_next!r}"
        )
    if int(raw_next) < 0:
        raise ThermalStateError(
            f"next_step must be a non-negative int, got {raw_next!r}"
        )

    planes: dict[str, np.ndarray] = {}
    for name in THERMAL_PLANE_NAMES:
        if name not in payload:
            raise ThermalStateError(
                f"thermal state is missing the plane {name!r}"
            )
        planes[name] = _copy_plane(name, payload[name])
    shapes = {plane.shape for plane in planes.values()}
    if len(shapes) != 1:
        raise ThermalStateError(
            f"thermal planes must all share one shape, got {sorted(shapes)} "
            "— a mixed/torn state is refused atomically"
        )
    if grid_shape is not None:
        shape = next(iter(shapes))
        if shape != (int(grid_shape[0]), int(grid_shape[1])):
            raise ThermalStateError(
                f"thermal plane shape {shape} does not match the solver "
                f"grid {grid_shape} — slice the full-tile state to the "
                "working extent before the warm start"
            )

    if expected_plane_digests is not None:
        for name in THERMAL_PLANE_NAMES:
            if name not in expected_plane_digests:
                raise ThermalStateError(
                    f"expected_plane_digests is missing the plane {name!r}"
                )
            actual = hashlib.sha256(
                np.ascontiguousarray(planes[name])
            ).hexdigest()
            if actual != expected_plane_digests[name]:
                raise ThermalStateError(
                    f"thermal plane {name!r} digest mismatch: expected "
                    f"{expected_plane_digests[name]!r}, recomputed "
                    f"{actual!r} — torn or mixed state, refusing"
                )

    ci = _scalar_number(payload, "CI")
    firstdaytime = _scalar_int(payload, "firstdaytime")
    timeadd = _scalar_number(payload, "timeadd")
    # Twater: absent/None = not-established (the pre-midnight [] the
    # checkpoint drops) — never a default water temperature.
    twater = payload.get("Twater")
    if twater is not None and (
        isinstance(twater, bool)
        or not isinstance(twater, (int, float, np.integer, np.floating))
    ):
        raise ThermalStateError(
            f"carried scalar 'Twater' must be a number or None, got "
            f"{type(twater).__name__}"
        )

    return ThermalState(
        next_step=int(raw_next),
        firstdaytime=firstdaytime,
        timeadd=timeadd,
        CI=ci,
        Twater=None if twater is None else float(twater),
        **planes,
    )


def restore_loop_state(
    state: ThermalState, *, grid_shape: tuple[int, int] | None = None
) -> tuple[RadLoopState, float | None]:
    """Bridge a ``ThermalState`` into the kernel's ``RadLoopState``.

    Validate-then-build (atomic): builds NEW plane copies so the replay
    can never mutate the caller's checkpoint state, and never writes
    into an existing live loop state — a refusal leaves every caller
    object bitwise untouched. Returns ``(loop_state, twater)``.
    """
    if not isinstance(state, ThermalState):
        raise ThermalStateError(
            f"expected a ThermalState, got {type(state).__name__}"
        )
    if grid_shape is not None:
        shape = state.Tgmap1.shape
        if shape != (int(grid_shape[0]), int(grid_shape[1])):
            raise ThermalStateError(
                f"thermal plane shape {shape} does not match the solver "
                f"grid {grid_shape} — refusing the warm start"
            )
    loop_state = RadLoopState(
        firstdaytime=int(state.firstdaytime),
        timeadd=float(state.timeadd),
        CI=np.float32(state.CI),
        **{
            name: np.ascontiguousarray(state.plane(name)).copy()
            for name in THERMAL_PLANE_NAMES
        },
    )
    return loop_state, state.Twater


# ---------------------------------------------------------------------------
# Applicability policy: geometry-history vs met-prefix (separated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointCandidate:
    """The checkpoint-record facts the policy needs (a narrow projection
    of ``checkpoints.CheckpointRecord``): ``thermal=False`` or
    ``next_step=None`` marks a coverage-only record."""

    scene_revision: int
    next_step: int | None
    thermal: bool
    input_fingerprint: Mapping[str, str] | None


@dataclass(frozen=True)
class AnchorDecision:
    """The outcome of an anchor scan.

    ``selected``/``resume_step``: the warm entry point (``resume_step``
    is the selected checkpoint's own ``next_step``; 0 = cold). ``reason``
    mirrors the newest candidate's outcome for fallback telemetry
    (executor ``met_warm_refusal`` spelling). ``invalidation`` is the
    SEPARATED rejection family:

    * ``"warm"`` — selected;
    * ``"geometry-history"`` — the scene lineage is stale (a previous
      scene's checkpoint: never servable under the current geometry,
      whatever its clock position or met prefix);
    * ``"met-prefix"`` — a time-axis boundary mismatch (the anchor heard
      different forcing in ``0..next_step-1``);
    * ``"future-prefix"`` — the anchor's met prefix covers the edited
      row (``next_step > r0``);
    * ``"coverage-only"`` — no warm state at all (NOT a thermal hit);
    * ``"none"`` — no candidates.
    """

    selected: CheckpointCandidate | None
    resume_step: int
    invalidation: str
    reason: str | None


def classify_anchor(
    candidate: CheckpointCandidate,
    *,
    geometry_fingerprint: Mapping[str, str],
    met_prefix_digest: Callable[[int], str],
    r0: int,
) -> tuple[str | None, str | None]:
    """Why ``candidate`` may NOT warm-start, as ``(reason, invalidation)``
    — ``(None, None)`` means applicable (resume at its own next_step).

    ``geometry_fingerprint``: the CURRENT digests of the scene-lineage
    keys (:data:`GEOMETRY_FINGERPRINT_KEYS`) — next_step independent.
    ``met_prefix_digest``: callable ``next_step -> digest`` over the
    CURRENT forcing rows ``0..next_step-1``; it is invoked AT THE
    CANDIDATE'S OWN ``next_step`` (never the request's ``r0`` — the
    applicability key describes what the captured state consumed).
    """
    revision = int(candidate.scene_revision)
    if not candidate.thermal or candidate.next_step is None:
        return (
            f"checkpoint rev-{revision:06d} is coverage-only (no warm "
            "state)",
            "coverage-only",
        )
    next_step = int(candidate.next_step)
    if next_step > int(r0):
        return (
            f"checkpoint rev-{revision:06d} next_step {next_step} is "
            f"beyond the first changed row {r0}: its met prefix covers "
            "the edited row",
            "future-prefix",
        )
    fp = candidate.input_fingerprint
    if fp is None:
        return (
            f"checkpoint rev-{revision:06d} carries no input fingerprint "
            "(coverage-only); warm start refused",
            "coverage-only",
        )
    # Geometry-history FIRST: a stale scene lineage is the unresolvable
    # contamination — reporting a met reason for it would understate the
    # refusal (and a met-only check would silently ACCEPT a previous
    # scene's matching-clock checkpoint: the R1 witness).
    for key in GEOMETRY_FINGERPRINT_KEYS:
        if fp.get(key) != geometry_fingerprint.get(key):
            return (
                f"geometry-history mismatch at {key!r}: checkpoint "
                f"rev-{revision:06d} recorded {fp.get(key)!r}, current "
                f"scene lineage digest {geometry_fingerprint.get(key)!r} "
                "— a checkpoint of a previous scene can never warm-start "
                "the current geometry",
                "geometry-history",
            )
    wanted_met = met_prefix_digest(next_step)
    actual_met = fp.get(MET_FINGERPRINT_KEYS[0])
    if actual_met != wanted_met:
        return (
            f"input fingerprint mismatch at 'met_prefix': checkpoint "
            f"rev-{revision:06d} recorded {actual_met!r}, current met "
            f"prefix digest {wanted_met!r} (computed at the candidate's "
            f"own next_step={next_step})",
            "met-prefix",
        )
    return None, None


def select_warm_anchor(
    candidates: Sequence[CheckpointCandidate],
    *,
    geometry_fingerprint: Mapping[str, str],
    met_prefix_digest: Callable[[int], str],
    r0: int,
) -> AnchorDecision:
    """Pick the warm-start entry point, newest-first (executor scan
    order): the first candidate whose applicability holds — an older
    checkpoint under matching inputs beats a newer one under stale
    inputs. The reported ``reason``/``invalidation`` mirror the NEWEST
    candidate's outcome (fallback telemetry). Never raises on mismatch,
    never guesses: no match means cold."""
    ordered = sorted(
        candidates, key=lambda c: int(c.scene_revision), reverse=True
    )
    newest_reason: str | None = None
    newest_invalidation: str | None = None
    for candidate in ordered:
        reason, invalidation = classify_anchor(
            candidate,
            geometry_fingerprint=geometry_fingerprint,
            met_prefix_digest=met_prefix_digest,
            r0=r0,
        )
        if reason is None:
            return AnchorDecision(
                selected=candidate,
                resume_step=int(candidate.next_step or 0),
                invalidation="warm",
                reason=None,
            )
        if newest_reason is None:
            newest_reason = reason
            newest_invalidation = invalidation
    if newest_reason is None:
        newest_reason = "no checkpoint available — cold replay from step 0"
        newest_invalidation = "none"
    return AnchorDecision(
        selected=None,
        resume_step=0,
        invalidation=newest_invalidation or "none",
        reason=newest_reason,
    )


# ---------------------------------------------------------------------------
# The replay driver (exact sparse temporal replay)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayOutcome:
    """One replay's result.

    ``outputs``: the collected timestep outputs keyed by GLOBAL ``t``
    (never a positional index — the t-scatter fence). ``final_state``:
    the carried state at ``next_step == t_stop`` (checkpointable).
    ``steps_solved``: timesteps actually advanced (the honest cost —
    a coverage-only marker produces no reduction here). ``advanced``:
    the global ts the loop consumed (the off-by-one ledger).
    """

    outputs: dict[int, dict[str, Any]]
    final_state: ThermalState
    steps_solved: int
    advanced: tuple[int, ...]


def replay_thermal(
    static: RadiationStatic,
    bundle_fn: Callable[[int], Any],
    state: ThermalState,
    *,
    collect_ts: Sequence[int],
    t_stop: int | None = None,
) -> ReplayOutcome:
    """Advance the thermal state through the T08 fused kernel.

    Resumes EXACTLY at ``state.next_step`` (never ``k-1``: duplicate;
    never ``k+1``: skip), consumes ``bundle_fn(t)`` per timestep, and
    collects outputs only at the requested GLOBAL ts (selected/sparse/
    full coverage: a ``[12]`` request advances the causal prefix but
    emits only ``t=12``). ``t_stop`` defaults to ``max(collect_ts)+1``;
    the returned ``final_state`` carries ``next_step == t_stop``.
    """
    if not isinstance(state, ThermalState):
        raise ThermalStateError(
            f"expected a ThermalState, got {type(state).__name__}"
        )
    start = int(state.next_step)
    ts = sorted({int(t) for t in collect_ts})
    if ts and ts[0] < start:
        raise ThermalStateError(
            f"collect t={ts[0]} precedes the anchor's next_step={start}: "
            "the anchor already consumed that timestep — refusing rather "
            "than replaying it twice (off-by-one fence)"
        )
    if t_stop is None:
        t_stop = (ts[-1] + 1) if ts else start
    t_stop = int(t_stop)
    if ts and ts[-1] >= t_stop:
        raise ThermalStateError(
            f"collect t={ts[-1]} is not below t_stop={t_stop}: the "
            "requested band would be silently dropped — refusing"
        )
    if t_stop < start:
        raise ThermalStateError(
            f"t_stop={t_stop} precedes the anchor's next_step={start}"
        )

    loop_state, twater = restore_loop_state(
        state, grid_shape=(int(static.rows), int(static.cols))
    )
    collect = set(ts)
    outputs: dict[int, dict[str, Any]] = {}
    for t in range(start, t_stop):
        t_in = bundle_fn(t)
        out, loop_state = fused_radiation_timestep(static, t_in, loop_state)
        if t in collect:
            outputs[int(t)] = out
    final_state = capture_thermal_state(
        loop_state=loop_state, next_step=t_stop, twater=twater
    )
    return ReplayOutcome(
        outputs=outputs,
        final_state=final_state,
        steps_solved=t_stop - start,
        advanced=tuple(range(start, t_stop)),
    )


def split_replay_thermal(
    static: RadiationStatic,
    bundle_fn: Callable[[int], Any],
    state: ThermalState,
    *,
    r0: int,
    t_stop: int,
) -> ReplayOutcome:
    """The executor's two-phase split solve (G2.1 item 4 semantics).

    Phase 1 advances ``state.next_step..r0`` DISCARDING outputs (the
    unchanged prefix stays served from the result store); phase 2 solves
    ``r0..t_stop`` collecting. The anchor must not lie beyond the split
    point — a checkpoint whose met prefix covers the edited row is a
    typed refusal here (the scan in :func:`select_warm_anchor` skips it;
    calling this directly with one is a caller bug, not a cold fallback).
    """
    r0 = int(r0)
    if int(state.next_step) > r0:
        raise ThermalStateError(
            f"anchor next_step={state.next_step} is beyond the split "
            f"point r0={r0}: its met prefix covers the edited row — "
            "refusing rather than replaying the edited prefix from a "
            "state that already heard it"
        )
    phase1 = replay_thermal(
        static, bundle_fn, state, collect_ts=(), t_stop=r0
    )
    phase2 = replay_thermal(
        static,
        bundle_fn,
        phase1.final_state,
        collect_ts=range(r0, int(t_stop)),
        t_stop=int(t_stop),
    )
    return ReplayOutcome(
        outputs=phase2.outputs,
        final_state=phase2.final_state,
        steps_solved=phase1.steps_solved + phase2.steps_solved,
        advanced=phase1.advanced + phase2.advanced,
    )
