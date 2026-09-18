# SPDX-License-Identifier: GPL-3.0-only
"""T11 torch-free full-domain fallback solve (exact, all edit families).

The incremental engine's last-resort lane: when no sparse/warm shortcut is
applicable, re-solve the WHOLE time window from the frozen-geometry
capture of ONE building profile, with the caller's (possibly edited) met
series threaded through :mod:`met_recompute` and the T08 fused kernel.

Provenance fences (typed errors, never silent wrongness):
  * ``ProfileFenceError`` — the capture set is BOUND to the profile label
    it was regenerated for. A solve request naming a different profile
    refuses: geometry-edit caches are per-profile and a geometry edit must
    REGENERATE the capture (never reuse another profile's planes).
  * ``StaleCheckpointError`` — a thermal checkpoint handed in for warm
    start is refused unless its geometry fingerprint matches the CURRENT
    capture's scene digests AND its met prefix matches the CURRENT series
    (:func:`thermal.classify_anchor` governs; unchanged on-disk schema).

Exactness contract: with the UNEDITED capture met, the per-t output dicts
equal the capture ``ret_*`` planes bit-for-bit (raw uint32) for every
timestep — the identity gate of met_recompute (all regenerated bundle
fields) composed with the T08 fused-kernel raw parity. An edited series
changes ONLY the met-dependent fields; geometry/march/walk tables stay
frozen bits of the capture.

Purity: runtime path imports numpy + this package only (the closed set
{numpy, numba, llvmlite, scipy} — torch appears nowhere, including
transitively: met_recompute is SLEEF numba ports).
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .met_recompute import (
    MetInputs,
    met_from_capture,
    recompute_timestep,
    site_params_from_capture,
    time_geom_from_capture,
    twater_advance,
)
from .radiation import (
    RadLoopState,
    RadiationStatic,
    fused_radiation_timestep,
    rad_bundle_from_capture,
    rad_state_from_capture,
    rad_static_from_capture,
)
from . import thermal

__all__ = [
    "CaptureSet",
    "ConvergenceBoundary",
    "ConvergenceReference",
    "FullSolveError",
    "FullSolveResult",
    "ProfileFenceError",
    "StaleCheckpointError",
    "full_solve_capture",
    "geometry_fingerprint_of_capture",
    "met_prefix_digest_of_series",
]


class FullSolveError(RuntimeError):
    """A full-solve request was refused (typed, never a silent no-op)."""


class ProfileFenceError(FullSolveError):
    """Capture/profile provenance mismatch — wrong-profile fallback."""


class StaleCheckpointError(FullSolveError):
    """A supplied thermal checkpoint cannot warm-start this solve."""


@dataclass(frozen=True)
class CaptureSet:
    """A frozen-geometry capture directory bound to ONE building profile.

    ``profile``: provenance label supplied at capture-regeneration time
    (the geometry-edit identity). Solves are served ONLY for the profile
    they request — ``full_solve_capture`` enforces the equality.
    """

    cap_dir: Path
    profile: str
    latitude: float

    @property
    def manifest(self) -> dict:
        with open(Path(self.cap_dir) / "manifest.json") as fh:
            return json.load(fh)

    @property
    def n_timesteps(self) -> int:
        return int(self.manifest["n_timesteps"])


def geometry_fingerprint_of_capture(cap_set: CaptureSet) -> dict[str, str]:
    """Scene-lineage digests in ``checkpoints.thermal_fingerprint``
    vocabulary (the :data:`thermal.GEOMETRY_FINGERPRINT_KEYS` spelling so
    ``classify_anchor`` compares like-for-like). The manifest's recorded
    digests win when present; otherwise a digest is computed over the
    STATIC geometry planes the kernel consumes (shape + every static
    plane's bytes) and stamped into every missing key.
    """
    man = cap_set.manifest
    fp: dict[str, str] = {}
    recorded = man.get("geometry_fingerprint")
    if isinstance(recorded, Mapping):
        fp.update({k: str(v) for k, v in recorded.items()})
    missing = [k for k in thermal.GEOMETRY_FINGERPRINT_KEYS if k not in fp]
    if missing:
        st = rad_static_from_capture(Path(cap_set.cap_dir))
        h = hashlib.sha256()
        h.update(f"{st.rows}:{st.cols}".encode())
        for name in sorted(vars(st)):
            v = getattr(st, name)
            if isinstance(v, np.ndarray):
                h.update(name.encode())
                h.update(np.ascontiguousarray(v).tobytes())
        digest = h.hexdigest()
        for k in missing:
            fp[k] = digest
    return fp


def met_prefix_digest_of_series(met_series: Sequence[MetInputs]) -> str:
    """Digest of forcing rows ``0..n-1`` (the ``met_prefix`` spelling):
    sha256 over the f64 text of the six fields per row, in row order.
    """
    h = hashlib.sha256()
    for m in met_series:
        h.update(
            f"{float(m.Ta)!r}|{float(m.RH)!r}|{float(m.radG)!r}|"
            f"{float(m.radD)!r}|{float(m.radI)!r}|{float(m.P)!r}".encode()
        )
    return h.hexdigest()


def _apply_day_overrides(t_in, res: Mapping[str, Any], met: MetInputs):
    """Write the regenerated met-dependent fields onto the day bundle."""
    t_in.Ta = np.float32(res["Ta32"])
    t_in.Ta_f64 = float(res["Ta_f64"])
    t_in.ta273_pow4 = np.float32(res["ta273_pow4_f64"])
    t_in.radI = np.float32(res["radI"])
    t_in.radD = np.float32(res["radD"])
    t_in.radG = np.float32(met.radG)
    t_in.ks_sunlit = np.float32(res["ks_sunlit"])
    t_in.ks_shaded = np.float32(res["ks_shaded"])
    t_in.lv = res["lv"]
    t_in.steradian = res["steradian"]
    t_in.Lsky_down2 = res["Lsky_down2"]
    t_in.Lsky_side2 = res["Lsky_side2"]
    t_in.lumChi = res["lumChi"]
    t_in.Tg_pre = res["Tg_pre"]
    t_in.Tg = res["Tg"]
    t_in.Tgwall = np.float32(res["Tgwall32"])
    t_in.Lup_pre = res["Lup_pre"]
    t_in.Lwall = res["Lwall32"]
    t_in.gvflup_extra = res["gvflup_extra"]
    t_in.dp_veg_surface_f64 = float(res["dp_veg_surface_f64"])
    t_in.dp_shaded_surface_f64 = float(res["dp_shaded_surface_f64"])
    t_in.dp_sunlit_surface_f64 = float(res["dp_sunlit_surface_f64"])
    t_in.CI_out = np.float32(res["CI_out"])
    t_in.Twater_f64 = float(met.Twater)


def _apply_night_overrides(t_in, res: Mapping[str, Any], met: MetInputs):
    """Night bundle: the met-dependent subset the night kernel consumes."""
    t_in.Ta = np.float32(met.Ta)
    t_in.Ta_f64 = float(res["Ta_f64"])
    t_in.ta273_pow4 = np.float32(res["ta273_pow4_f64"])
    t_in.Lsky_down2 = res["Lsky_down2"]
    t_in.Lsky_side2 = res["Lsky_side2"]
    t_in.steradian = res["steradian"]
    t_in.night_Lup = res["night_Lup"]
    t_in.night_water_override = np.float32(res["night_water_override"])
    t_in.dp_veg_surface_f64 = float(res["dp_veg_surface_f64"])
    t_in.dp_shaded_surface_f64 = float(res["dp_shaded_surface_f64"])
    t_in.dp_sunlit_surface_f64 = float(res["dp_sunlit_surface_f64"])
    t_in.CI_out = np.float32(res["CI_out"])
    t_in.Twater_f64 = float(met.Twater)


# ---------------------------------------------------------------------------
# T20 / DESIGN §589: equality-triggered temporal convergence
# ---------------------------------------------------------------------------
#
# A dirty recurrence may reuse a committed run's suffix ONLY when, at some
# boundary, (a) the reference future inputs are bit-identical AND (b) the
# carried state is bit-identical — six planes + firstdaytime/timeadd/CI +
# the threaded f64 CI + its tensor/pyfloat flavor (the flavor decides the
# night esky branch, so it is load-bearing). Partial equality or
# small-temperature early exit is FORBIDDEN (§589 verbatim). The reuse is
# then exact by induction: recompute_timestep and the fused kernel are
# deterministic pure functions of (state, CI thread, flavor, met row,
# capture bytes), so bit-identical carried state + bit-identical future
# inputs produce bit-identical outputs for the whole suffix.

_LOOP_PLANE_NAMES = ("Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W",
                     "Tgmap1N", "TgOut1")


def _f64_bits(x: float) -> bytes:
    # exact f64 bit pattern: signed zero and NaN payload distinct
    return struct.pack(">d", float(x))


def _f32_bits(x) -> bytes:
    # exact f32 bit pattern (widening a float32 to f64 is exact, so the
    # pack below never loses payload or sign-of-zero information)
    return struct.pack(">f", float(np.float32(x)))


def _row_bits(m) -> bytes:
    """Bit key of one forcing row: every MetInputs field, NaN payloads and
    signed zeros included (stricter than ``met_prefix_digest``, which
    hashes only the six scalar fields)."""
    return b"".join([
        _f64_bits(m.Ta), _f64_bits(m.RH), _f64_bits(m.radG),
        _f64_bits(m.radD), _f64_bits(m.radI), _f64_bits(m.P),
        _f64_bits(m.Twater),
        b"\x01" if m.Twater_is_list else b"\x00",
    ])


def _future_inputs_equal(series, ref_series, start: int) -> bool:
    for t in range(start, len(series)):
        if _row_bits(series[t]) != _row_bits(ref_series[t]):
            return False
    return True


def _state_bits_equal(a: RadLoopState, b: RadLoopState) -> bool:
    """All-bits carried-state equality: six planes as raw uint32 (NaN
    payload / signed zero distinct), plus every carried scalar."""
    for name in _LOOP_PLANE_NAMES:
        pa = np.asarray(getattr(a, name))
        pb = np.asarray(getattr(b, name))
        if pa.shape != pb.shape or pa.dtype != pb.dtype:
            return False
        if not np.array_equal(np.ascontiguousarray(pa).view(np.uint32),
                              np.ascontiguousarray(pb).view(np.uint32)):
            return False
    if int(a.firstdaytime) != int(b.firstdaytime):
        return False
    if _f64_bits(a.timeadd) != _f64_bits(b.timeadd):
        return False
    return _f32_bits(a.CI) == _f32_bits(b.CI)


def _copy_output_dict(out: Mapping) -> dict:
    """Defensive copy of one timestep's output dict (planes copied; the
    caller may mutate its result without corrupting the reference)."""
    return {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
            for k, v in out.items()}


def _copy_loop_state(state: RadLoopState) -> RadLoopState:
    return RadLoopState(
        firstdaytime=int(state.firstdaytime),
        timeadd=float(state.timeadd),
        CI=np.float32(state.CI),
        **{name: np.ascontiguousarray(getattr(state, name)).copy()
           for name in _LOOP_PLANE_NAMES},
    )


@dataclass(frozen=True)
class ConvergenceBoundary:
    """The reference run's carried facts at one boundary (after completing
    ``step`` timesteps): loop state + threaded CI + its flavor."""

    state: RadLoopState
    ci_thread: float
    ci_flavor: bool


@dataclass(frozen=True)
class ConvergenceReference:
    """A committed execution of THIS capture a dirty recurrence may
    converge back to (DESIGN §589).

    ``outputs``/``ret_ci_series``/``ret_ci_flavors`` are indexed by
    ABSOLUTE timestep (``start_step`` is the reference run's own first
    executed step — the index base). ``boundaries`` maps an absolute
    boundary ``s`` to the reference's carried state AFTER ``s`` steps;
    a dirty run may fire there only when no requested mid-anchor lies at
    or beyond ``s``. ``geometry_fingerprint`` must equal the CURRENT
    capture's scene digests — a regenerated capture refuses the reference
    (stale-scene reuse is a forbidden shortcut, never a fallback).

    Arrays handed to a converged solve are defensively copied; the
    reference stays valid for later requests.
    """

    met_series: tuple
    outputs: Sequence
    ret_ci_series: Sequence
    ret_ci_flavors: Sequence
    boundaries: Mapping[int, ConvergenceBoundary]
    start_step: int
    final_state: RadLoopState
    geometry_fingerprint: Mapping[str, str]
    profile: str
    n_timesteps: int

    @classmethod
    def from_result(
        cls,
        result: FullSolveResult,
        *,
        met_series: Sequence,
        start_step: int,
        geometry_fingerprint: Mapping[str, str],
        boundaries: Mapping[int, ConvergenceBoundary],
    ) -> ConvergenceReference:
        """Bind a committed :class:`FullSolveResult` (plus the boundary
        snapshots its caller retained) into a convergence reference. The
        result's ``profile``/``n_timesteps``/``outputs``/CI series are
        authoritative; ``start_step`` is the base the outputs are
        indexed from (the result does not retain it)."""
        return cls(
            met_series=tuple(met_series),
            outputs=tuple(result.outputs),
            ret_ci_series=tuple(result.ret_CI_series),
            ret_ci_flavors=tuple(getattr(result, "ret_CI_flavors", ())),
            boundaries={int(k): v for k, v in boundaries.items()},
            start_step=int(start_step),
            final_state=result.final_state,
            geometry_fingerprint=dict(geometry_fingerprint),
            profile=result.profile,
            n_timesteps=int(result.n_timesteps),
        )


@dataclass
class FullSolveResult:
    outputs: list                      # per-t kernel output dicts
    final_state: RadLoopState
    ret_CI_series: list                # f64 driver-threaded CI values
    anchor_published: thermal.ThermalState | None
    anchor_fingerprint: dict | None    # the published anchor's fp dict
    anchor_refusal: tuple | None       # (reason, invalidation) if refused
    profile: str
    n_timesteps: int
    #: per-t CI thread flavor (the tensor/pyfloat spelling the step heard;
    #: load-bearing for the night esky branch — retained for §589 keys)
    ret_CI_flavors: list = field(default_factory=list)
    timings_ms: list = field(default_factory=list)
    #: mid-window anchors (publish_anchor_at): (state, fingerprint) at
    #: each requested step — the incremental engine's checkpoint cadence
    anchors: list = field(default_factory=list)
    #: §589 convergence record when a suffix was reused from a reference:
    #: {"fired_at", "reused_steps", "boundaries_probed", "check_cost_ms",
    #:  "copied_output_bytes", "suppressed_at", "suppressed_reason"}
    convergence: dict | None = None


def full_solve_capture(
    cap_set: CaptureSet,
    met_series: Sequence[MetInputs] | None = None,
    *,
    profile: str,
    rows: int = 500,
    cols: int = 500,
    warm_state: thermal.ThermalState | None = None,
    warm_fingerprint: Mapping[str, str] | None = None,
    r0: int = 0,
    publish_anchor: bool = True,
    publish_anchor_at: Sequence[int] | None = None,
    convergence_reference: ConvergenceReference | None = None,
) -> FullSolveResult:
    """Run the whole window on the capture, met-edits threaded exactly.

    ``met_series``: full-length MetInputs sequence (edited family). None
    uses the capture's own frozen per-t met (the identity configuration).
    ``warm_state`` + ``warm_fingerprint``: optional checkpoint warm start
    — REFUSED unless the fingerprint's geometry keys match the capture's
    CURRENT scene digests and its ``met_prefix`` matches the CURRENT
    series at the checkpoint's own next_step (classify_anchor policy).
    ``r0``: the request's first changed met row (0 = treat the whole
    series as changed; the router supplies the edit position).
    ``publish_anchor``: capture the final thermal state as a warm anchor
    (self-classified via classify_anchor; nothing is written to disk —
    publication is the returned in-memory state + fingerprint dict).
    ``publish_anchor_at``: additional mid-window anchor steps (state is
    captured AFTER completing each listed timestep; next_step = step+1).
    ``convergence_reference``: optional §589 committed run of THIS
    capture. At every loop boundary with a reference snapshot, the
    carried state is compared ALL-BITS; only when the state AND every
    future forcing row are bit-identical does the solve stop and serve
    the suffix from the reference (defensively copied). Partial
    equality is never accepted; a mismatched reference (other profile,
    window, or scene lineage) is a typed refusal, never a silent
    ignore. Default None keeps the driver byte-identical to its
    no-reference behavior.
    """
    cap_dir = Path(cap_set.cap_dir)
    n = cap_set.n_timesteps
    if profile != cap_set.profile:
        raise ProfileFenceError(
            f"capture set at {cap_dir} was regenerated for profile "
            f"{cap_set.profile!r}; a {profile!r} solve would consume "
            "another profile's frozen geometry — regenerate the capture "
            "for this profile instead"
        )
    if met_series is not None and len(met_series) != n:
        raise FullSolveError(
            f"met series length {len(met_series)} != capture "
            f"n_timesteps {n}"
        )

    z0 = np.load(cap_dir / "t00.npz")
    site = site_params_from_capture(z0, cap_set.latitude)
    st: RadiationStatic = rad_static_from_capture(cap_dir)
    geom_fp = geometry_fingerprint_of_capture(cap_set)

    # §589 reference fences: a foreign reference (other profile, window,
    # scene lineage, or series length) is a caller bug — the wrong-scene
    # suffix it could serve is a forbidden shortcut. Typed refusal, never
    # a silent ignore and never a wrong reuse.
    if convergence_reference is not None:
        ref = convergence_reference
        if ref.profile != profile:
            raise FullSolveError(
                f"convergence reference is for profile {ref.profile!r}, "
                f"not {profile!r} — wrong-profile suffix reuse refused"
            )
        if ref.n_timesteps != n:
            raise FullSolveError(
                f"convergence reference window {ref.n_timesteps} != "
                f"capture window {n}"
            )
        if dict(ref.geometry_fingerprint) != geom_fp:
            raise FullSolveError(
                "convergence reference scene lineage differs from the "
                "CURRENT capture digests — a regenerated capture never "
                "reuses a previous scene's suffix"
            )
        if len(ref.met_series) != n:
            raise FullSolveError(
                f"convergence reference series length {len(ref.met_series)} "
                f"!= capture n_timesteps {n}"
            )
    else:
        ref = None

    # series resolved up-front: the met-prefix fence and the warm start
    # both need the CURRENT forcing rows
    if met_series is None:
        series = [met_from_capture(np.load(cap_dir / f"t{t:02d}.npz"))
                  for t in range(n)]
    else:
        series = list(met_series)

    start_step = 0
    if warm_state is not None:
        if warm_fingerprint is None:
            raise StaleCheckpointError(
                "warm state carries no input fingerprint — cannot verify "
                "geometry lineage or met prefix; refusing (cold replay "
                "is the safe fallback)"
            )
        candidate = thermal.CheckpointCandidate(
            scene_revision=1,
            next_step=int(warm_state.next_step),
            thermal=True,
            input_fingerprint=dict(warm_fingerprint),
        )
        reason, invalidation = thermal.classify_anchor(
            candidate,
            geometry_fingerprint=geom_fp,
            met_prefix_digest=lambda k: met_prefix_digest_of_series(
                series[:k]),
            r0=int(r0),
        )
        if reason is not None:
            raise StaleCheckpointError(
                f"warm state refused ({invalidation}): {reason}")
        start_step = int(warm_state.next_step)
        if start_step > n:
            raise StaleCheckpointError(
                f"warm state next_step {start_step} beyond the capture "
                f"window {n}"
            )

    state: RadLoopState = rad_state_from_capture(cap_dir, 0, rows=rows,
                                                 cols=cols)
    if warm_state is not None:
        state, _twater = thermal.restore_loop_state(
            warm_state, grid_shape=(st.rows, st.cols))

    ci_thread = float(state.CI)
    # capture t00 entry CI is the pyfloat 1.0 flavor; a warm entry's CI
    # flavor is unknown → treated as pyfloat (classify_anchor already
    # fenced the series identity, so the correction guard is exact)
    ci_flavor_is_tensor = False
    outputs: list = []
    ret_ci_series: list = []
    ret_ci_flavors: list = []
    timings_ms: list = []
    mid_anchors: list = []
    anchor_steps = (set(int(a) for a in publish_anchor_at)
                    if publish_anchor_at else set())
    for a in anchor_steps:
        if not 0 <= a < n:
            raise FullSolveError(
                f"publish_anchor_at step {a} outside [0, {n})")

    # §589 convergence bookkeeping
    conv: dict | None = None
    conv_suppressed: dict | None = None
    conv_probed = 0
    conv_cost = 0.0

    for t in range(start_step, n):
        if ref is not None:
            _chk0 = time.perf_counter()
            snap = ref.boundaries.get(int(t))
            if snap is not None and t >= ref.start_step:
                conv_probed += 1
                # a requested mid-anchor at/after t could not be produced
                # if the suffix [t, n) were reused — the boundary is
                # forfeited (conservative; anchors outrank convergence)
                _anchor_block = any(a >= t for a in anchor_steps)
                if not _anchor_block:
                    # cheap predicates first: forcing-row bytes, then the
                    # CI thread scalars, the plane scan last
                    future_equal = _future_inputs_equal(
                        series, ref.met_series, t)
                    ci_equal = (_f64_bits(ci_thread)
                                == _f64_bits(snap.ci_thread))
                    flavor_equal = ci_flavor_is_tensor == snap.ci_flavor
                    state_equal = _state_bits_equal(state, snap.state)
                    if (state_equal and ci_equal and flavor_equal
                            and future_equal):
                        conv_cost += (time.perf_counter() - _chk0) * 1e3
                        copied = 0
                        for tt in range(t, n):
                            outputs.append(_copy_output_dict(ref.outputs[tt]))
                            ret_ci_series.append(ref.ret_ci_series[tt])
                            ret_ci_flavors.append(ref.ret_ci_flavors[tt])
                            for v in ref.outputs[tt].values():
                                if isinstance(v, np.ndarray):
                                    copied += v.nbytes
                        # the replay's terminal state IS the reference's
                        # final state (same induction that proves the
                        # suffix); copied so the caller can never mutate
                        # the reference through the result
                        state = _copy_loop_state(ref.final_state)
                        conv = {
                            "fired_at": int(t),
                            "reused_steps": int(n - t),
                            "boundaries_probed": int(conv_probed),
                            "check_cost_ms": conv_cost,
                            "copied_output_bytes": int(copied),
                            "suppressed_at": (None if conv_suppressed is None
                                              else conv_suppressed["at"]),
                            "suppressed_reason": (
                                None if conv_suppressed is None
                                else conv_suppressed["reason"]),
                        }
                        break
                elif conv_suppressed is None:
                    conv_suppressed = {
                        "at": int(t),
                        "reason": (
                            "requested mid-anchor inside the reusable "
                            "region (anchors outrank convergence)"
                        ),
                    }
            conv_cost += (time.perf_counter() - _chk0) * 1e3

        t0 = time.perf_counter()
        z = np.load(cap_dir / f"t{t:02d}.npz")
        tg = time_geom_from_capture(z)
        met = series[t]
        t_in = rad_bundle_from_capture(cap_dir, t)
        # sunonsurface's shadow ARG as the capture recorded it (identical
        # bits to t_in.shadow's clean recomputation — T08 raw parity — but
        # the recorded arg is the provenance-true input)
        shadow = (z["sunon_in_shadow"]
                  if t_in.is_day and "sunon_in_shadow" in z.files else None)
        res = recompute_timestep(
            site, met, tg, ci_thread, shadow,
            CI_thread_is_tensor=ci_flavor_is_tensor)
        if t_in.is_day:
            _apply_day_overrides(t_in, res, met)
        else:
            _apply_night_overrides(t_in, res, met)
        out, state = fused_radiation_timestep(st, t_in, state)
        ci_thread = float(res["ret_CI"])
        ci_flavor_is_tensor = bool(res["CI_flavor_is_tensor"])
        ret_ci_series.append(ci_thread)
        ret_ci_flavors.append(ci_flavor_is_tensor)
        outputs.append(out)
        timings_ms.append((time.perf_counter() - t0) * 1e3)
        if t in anchor_steps:
            mid_anchors.append((
                thermal.capture_thermal_state(
                    loop_state=state, next_step=t + 1, twater=None),
                {**geom_fp,
                 thermal.MET_FINGERPRINT_KEYS[0]:
                     met_prefix_digest_of_series(series[:t + 1])},
            ))

    if conv is None and conv_suppressed is not None:
        # a boundary would have been checked but was forfeited: record it
        conv = {
            "fired_at": None,
            "reused_steps": 0,
            "boundaries_probed": int(conv_probed),
            "check_cost_ms": conv_cost,
            "copied_output_bytes": 0,
            "suppressed_at": conv_suppressed["at"],
            "suppressed_reason": conv_suppressed["reason"],
        }

    anchor_published = None
    anchor_fingerprint = None
    anchor_refusal = None
    if publish_anchor:
        ts = thermal.capture_thermal_state(
            loop_state=state, next_step=n, twater=None)
        anchor_fingerprint = {
            **geom_fp,
            thermal.MET_FINGERPRINT_KEYS[0]:
                met_prefix_digest_of_series(series),
        }
        # self-classify: the published anchor must be servable for a next
        # request whose first changed row is at/after its next_step
        cand = thermal.CheckpointCandidate(
            scene_revision=1, next_step=n, thermal=True,
            input_fingerprint=dict(anchor_fingerprint))
        reason, invalidation = thermal.classify_anchor(
            cand,
            geometry_fingerprint=geom_fp,
            met_prefix_digest=lambda k: met_prefix_digest_of_series(
                series[:k]),
            r0=n,
        )
        if reason is None:
            anchor_published = ts
        else:
            anchor_published = None
            anchor_refusal = (reason, invalidation)
            anchor_fingerprint = None

    return FullSolveResult(
        outputs=outputs,
        final_state=state,
        ret_CI_series=ret_ci_series,
        ret_CI_flavors=ret_ci_flavors,
        anchor_published=anchor_published,
        anchor_fingerprint=anchor_fingerprint,
        anchor_refusal=anchor_refusal,
        profile=cap_set.profile,
        n_timesteps=n,
        timings_ms=timings_ms,
        anchors=mid_anchors,
        convergence=conv,
    )


def twater_midnight_schedule(met_series: Sequence[MetInputs],
                             dectime: Sequence[float]) -> list[float]:
    """Twater advance schedule (torch-free; utci_process.py:921-923):
    at i == 0 and every midnight row,
    ``Twater = np.mean(Ta[floor(dectime) == floor(dectime[i])])`` — the
    numpy blocked-pairwise f64 mean replayed on numpy (``twater_advance``).
    Landcover==1 sites only; refused sites never call this.
    """
    ta_rows = np.asarray([float(m.Ta) for m in met_series],
                         dtype=np.float64)
    floors = np.floor(np.asarray(
        [float(d) for d in dectime], dtype=np.float64))
    out: list[float] = []
    for i in range(len(met_series)):
        if i == 0 or floors[i] == floors[max(i - 1, 0)] + 1.0:
            out.append(twater_advance(ta_rows[floors == floors[i]]))
        else:
            out.append(out[-1])
    return out
