# SPDX-License-Identifier: GPL-3.0-only
"""T21 prefix-accumulator suffix fold ablation harness (DESIGN §807).

The candidate: store the EXACT accumulator prefix of the temporal
canonical fold (:mod:`solweig_core.numba_cpu.full_solve`'s day loop) at
timestep boundaries of a committed run; when a dirty recurrence arrives
(first changed forcing row ``r0``), restore the latest boundary
``a <= r0`` and re-run the suffix fold ``[a, n)`` instead of folding the
whole day from 0. This is NOT the forbidden subtract-old/add-new scheme
(acceptance ``forbidden_shortcuts``): the identical prefix input bits and
the identical accumulator bits let the ORIGINAL recurrence continue
mid-fold, so the suffix outputs are the fold's own bits by induction.

Carried-state enumeration at a boundary (after completing step ``a-1``)
— complete by §589's own vocabulary (six planes + four scalars + the
load-bearing CI flavor; ``full_solve.py`` module comment spells the same
eleven items):

======================  ==============  ================================
item                    kind            consumed by
======================  ==============  ================================
Tgmap1                  f32 plane       TsWaveDelay blend (Lup)
Tgmap1E/S/W/N           f32 planes      TsWaveDelay blends (cardinals)
TgOut1                  f32 plane       TsWaveDelay blend (TgOut)
firstdaytime            int scalar      TsWaveDelay first-day reset
timeadd                 f64 scalar      TsWaveDelay 59/1440 branch
CI                      f32 scalar      night CI_Tg/CI_TgG outputs
ci_thread               f64 scalar      recompute_timestep CI thread
ci_flavor_is_tensor     bool scalar     recompute_timestep esky branch
next_step               int (metadata)  boundary identity
met_prefix digest       bytes (fence)   classify_anchor-style validity
======================  ==============  ================================

Nothing else is carried: ``Twater`` is a per-row forcing input (the
driver passes ``twater=None``), per-timestep bundles and the static
scene are inputs, and the day kernel allocates fresh output planes
(state planes are read-only; the night branch re-publishes the INPUT
plane objects unchanged — restored copies are therefore never mutated).

A missed accumulator is silent divergence, so the mutations in
:mod:`tests.ultrafast.test_t21_suffix_fold` drop each suspicious item
and require the comparator to notice.

Discipline: harness-only module — production code is imported, never
modified. The :func:`run_window` loop mirrors
``full_solve.full_solve_capture``'s body (no convergence/anchor
bookkeeping) and is pinned to it bit-for-bit by
``test_t21_suffix_fold.test_harness_loop_matches_production_full_solve``.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solweig_core.numba_cpu import full_solve as fs
from solweig_core.numba_cpu.met_recompute import (
    met_from_capture,
    recompute_timestep,
    site_params_from_capture,
    time_geom_from_capture,
)
from solweig_core.numba_cpu.radiation import (
    RadLoopState,
    fused_radiation_timestep,
    rad_bundle_from_capture,
    rad_state_from_capture,
    rad_static_from_capture,
)

CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")
LATITUDE = 30.312645
PROFILE = "site_500-default"
ART = Path.home() / "Workspace" / "solweig_ultrafast_artifacts" / "t21"

#: The six carried f32 planes (full_solve._LOOP_PLANE_NAMES spelling).
LOOP_PLANES = fs._LOOP_PLANE_NAMES

#: Payload bytes of one boundary checkpoint at site_500 (500, 500):
#: 6 planes x 1,000,000 B + scalars (int fd 8 + f64 timeadd 8 + f32 CI 4
#: + f64 ci_thread 8 + bool flavor 1 + int next_step 8 + sha256 32).
PLANE_BYTES = 500 * 500 * 4
BOUNDARY_PAYLOAD_BYTES = (
    6 * PLANE_BYTES + 8 + 8 + 4 + 8 + 1 + 8 + 32
)


def capture_present() -> bool:
    return (CAP / "t00.npz").exists()


def f64_bits(x: float) -> bytes:
    return struct.pack(">d", float(x))


def f32_bits(x) -> bytes:
    return struct.pack(">f", float(np.float32(x)))


# ---------------------------------------------------------------------------
# Boundary checkpoint: the EXACT accumulator prefix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Boundary:
    """Bit-exact prefix checkpoint of the temporal fold.

    Every float is stored at its native width — ``ci_thread`` keeps its
    f64 bits (the production warm path collapses it through the f32
    ``state.CI``; this harness does not, and mutation M1 measures whether
    that collapse is even observable on this capture).
    """

    next_step: int
    firstdaytime: int
    timeadd: float            # exact f64
    CI: np.float32            # exact f32
    ci_thread: float          # exact f64
    ci_flavor: bool
    met_prefix: str           # digest of forcing rows [0, next_step)
    planes: dict              # name -> f32 (rows, cols) C-contig copy

    @property
    def payload_bytes(self) -> int:
        return BOUNDARY_PAYLOAD_BYTES


def met_prefix_digest(series, k: int) -> str:
    return fs.met_prefix_digest_of_series(series[:k])


def capture_boundary(state: RadLoopState, ci_thread: float, ci_flavor: bool,
                     next_step: int, series) -> Boundary:
    """Deep-copy the complete carried state at boundary ``next_step``."""
    return Boundary(
        next_step=int(next_step),
        firstdaytime=int(state.firstdaytime),
        timeadd=float(state.timeadd),
        CI=np.float32(state.CI),
        ci_thread=float(ci_thread),
        ci_flavor=bool(ci_flavor),
        met_prefix=met_prefix_digest(series, int(next_step)),
        planes={name: np.ascontiguousarray(
            getattr(state, name)).copy() for name in LOOP_PLANES},
    )


def restore_boundary(b: Boundary, *, mutate: str | None = None
                     ) -> tuple[RadLoopState, float, bool]:
    """Rebuild (state, ci_thread, ci_flavor) from a checkpoint.

    ``mutate`` names a deliberate defect (M1..M5) for the
    counterexample search; None is the faithful restore.
    """
    planes = {name: b.planes[name].copy() for name in LOOP_PLANES}
    ci_thread = float(b.ci_thread)
    ci_flavor = bool(b.ci_flavor)
    timeadd = float(b.timeadd)
    firstdaytime = int(b.firstdaytime)
    if mutate == "M1_ci_thread_f32_collapse":
        # the production warm path's spelling: float(state.CI)
        ci_thread = float(np.float32(b.CI))
    elif mutate == "M2_flavor_flip":
        ci_flavor = not ci_flavor
    elif mutate == "M3_timeadd_drop":
        timeadd = 0.0
    elif mutate == "M4_firstdaytime_drop":
        firstdaytime = 1
    elif mutate == "M5_plane_nan":
        # an unabsorbable corruption: NaN propagates through the blend
        # chain into the same step's outputs (comparator teeth, floor end)
        planes["Tgmap1"][0, 0] = np.float32(np.nan)
    elif mutate == "M5b_plane_1ulp_normal":
        # the 1-ULP floor probe on a NORMAL-valued cell: measures whether
        # f32 rounding absorbs a 1-ulp state corruption (propagation
        # diagnostic, NOT a gate — absorption is arithmetic, not a
        # comparator failure)
        vals = planes["Tgmap1"].view(np.uint32).reshape(-1)
        f = planes["Tgmap1"].reshape(-1)
        idx = int(np.argmax(np.abs(f) >= np.float32(1e-3)))
        vals[idx] += np.uint32(1)
    elif mutate is not None:
        raise ValueError(f"unknown mutation {mutate!r}")
    state = RadLoopState(
        firstdaytime=firstdaytime,
        timeadd=timeadd,
        CI=np.float32(b.CI),
        **planes,
    )
    return state, ci_thread, ci_flavor


def mutated_boundary(b: Boundary, mutate: str) -> Boundary:
    """A checkpoint with one deliberate defect (fence fields preserved —
    the mutation must reach the FOLD, not trip the restore fence)."""
    state, ci_thread, ci_flavor = restore_boundary(b, mutate=mutate)
    return Boundary(
        next_step=b.next_step,
        firstdaytime=int(state.firstdaytime),
        timeadd=float(state.timeadd),
        CI=np.float32(state.CI),
        ci_thread=float(ci_thread),
        ci_flavor=bool(ci_flavor),
        met_prefix=b.met_prefix,
        planes={name: getattr(state, name) for name in LOOP_PLANES},
    )


def mutation_first_divergence(cap_set, es, b_mut: Boundary, dirty_outputs):
    """Fold from a mutated checkpoint, stopping at the first output
    divergence against the faithful dirty run.

    Returns None only when the suffix COMPLETED (all ``n - next_step``
    steps) — a partial window is a detected divergence: (t, plane) of
    the first step whose bits left the faithful run.
    """
    suf = run_window(cap_set, es, boundary=b_mut,
                     expect_outputs=dirty_outputs)
    expected = int(cap_set.n_timesteps) - int(b_mut.next_step)
    if len(suf.steps) == expected:
        return None
    t = suf.steps[-1]
    return (t, first_output_divergence(suf.outputs[-1],
                                       dirty_outputs[t], t))


def boundary_bits_equal(a: Boundary, b: Boundary) -> bool:
    """All-bits boundary equality: planes as raw uint32, f64/f32 scalars
    by bit pattern (signed zero and NaN payload distinct)."""
    for name in LOOP_PLANES:
        pa = np.ascontiguousarray(a.planes[name]).view(np.uint32)
        pb = np.ascontiguousarray(b.planes[name]).view(np.uint32)
        if not np.array_equal(pa, pb):
            return False
    return (int(a.firstdaytime) == int(b.firstdaytime)
            and f64_bits(a.timeadd) == f64_bits(b.timeadd)
            and f32_bits(a.CI) == f32_bits(b.CI)
            and f64_bits(a.ci_thread) == f64_bits(b.ci_thread)
            and bool(a.ci_flavor) == bool(b.ci_flavor))


def state_bits_equal(a: RadLoopState, b: Boundary) -> bool:
    """A live RadLoopState against a checkpoint (raw bits)."""
    for name in LOOP_PLANES:
        pa = np.ascontiguousarray(getattr(a, name)).view(np.uint32)
        pb = np.ascontiguousarray(b.planes[name]).view(np.uint32)
        if not np.array_equal(pa, pb):
            return False
    return (int(a.firstdaytime) == int(b.firstdaytime)
            and f64_bits(a.timeadd) == f64_bits(b.timeadd)
            and f32_bits(a.CI) == f32_bits(b.CI))


# ---------------------------------------------------------------------------
# The harness fold loop (mirror of full_solve.full_solve_capture's body)
# ---------------------------------------------------------------------------


@dataclass
class WindowResult:
    outputs: list                 # per-t output dicts, absolute t index
    final_state: RadLoopState
    ci_thread: float
    ci_flavor: bool
    ret_ci_series: list
    ret_ci_flavors: list
    timings_ms: list              # per executed step (fs timings_ms span)
    steps: list                   # executed absolute step indices
    states: list = None           # per-t post-step carried state (opt-in)


def run_window(cap_set: "fs.CaptureSet", series, *,
               boundary: Boundary | None = None,
               start_step: int | None = None,
               expect_outputs: list | None = None,
               collect_states: bool = False):
    """Fold ``[start, n)`` of the capture, met-edits threaded exactly.

    ``boundary``: restore a prefix checkpoint (state, ci_thread, flavor)
    instead of the capture's cold entry — §807's scheme. ``start_step``
    defaults to the boundary's ``next_step`` (or 0).

    ``expect_outputs`` (mutation diagnostics): compare each completed
    step's planes against the given reference list as soon as it is
    produced and stop at the FIRST diverging step (returns that partial
    result; cheap counterexample probes).

    ``collect_states``: retain the post-step carried state per executed
    step (plane references are safe to keep: the kernels never write
    into state planes — day steps replace them, night steps re-publish
    them unchanged) for the per-step oracle next-state comparison.

    ``WindowResult.states`` is indexed by ABSOLUTE step (state AFTER
    completing that step); None-filled when not collected.
    """
    cap_dir = Path(cap_set.cap_dir)
    n = cap_set.n_timesteps
    if boundary is not None:
        start = int(boundary.next_step) if start_step is None else int(start_step)
        state, ci_thread, ci_flavor = restore_boundary(boundary)
        if start > 0:
            # the fence §807 inherits from classify_anchor: the boundary
            # is only applicable when the rows it consumed are unchanged
            if met_prefix_digest(series, start) != boundary.met_prefix:
                raise fs.StaleCheckpointError(
                    f"boundary next_step={start} met_prefix mismatch "
                    "(restored prefix would silently fabricate history)")
    else:
        start = 0 if start_step is None else int(start_step)
        state = rad_state_from_capture(cap_dir, 0, rows=500, cols=500)
        ci_thread = float(state.CI)
        ci_flavor = False

    site = site_params_from_capture(np.load(cap_dir / "t00.npz"),
                                    cap_set.latitude)
    st: RadiationStatic = rad_static_from_capture(cap_dir)

    outputs: list = []
    ret_ci_series: list = []
    ret_ci_flavors: list = []
    timings_ms: list = []
    steps: list = []
    states: list = [] if collect_states else None
    for t in range(start, n):
        t0 = time.perf_counter()
        z = np.load(cap_dir / f"t{t:02d}.npz")
        tg = time_geom_from_capture(z)
        met = series[t]
        t_in = rad_bundle_from_capture(cap_dir, t)
        shadow = (z["sunon_in_shadow"]
                  if t_in.is_day and "sunon_in_shadow" in z.files else None)
        res = recompute_timestep(
            site, met, tg, ci_thread, shadow,
            CI_thread_is_tensor=ci_flavor_is_tensor_flag(ci_flavor))
        if t_in.is_day:
            fs._apply_day_overrides(t_in, res, met)
        else:
            fs._apply_night_overrides(t_in, res, met)
        out, state = fused_radiation_timestep(st, t_in, state)
        ci_thread = float(res["ret_CI"])
        ci_flavor = bool(res["CI_flavor_is_tensor"])
        ret_ci_series.append(ci_thread)
        ret_ci_flavors.append(ci_flavor)
        outputs.append(out)
        timings_ms.append((time.perf_counter() - t0) * 1e3)
        steps.append(t)
        if collect_states:
            states.append((state, ci_thread, ci_flavor))
        if expect_outputs is not None:
            ref = expect_outputs[t]
            if first_output_divergence(out, ref, t) is not None:
                return WindowResult(outputs, state, ci_thread, ci_flavor,
                                    ret_ci_series, ret_ci_flavors,
                                    timings_ms, steps, states)
    return WindowResult(outputs, state, ci_thread, ci_flavor,
                        ret_ci_series, ret_ci_flavors, timings_ms, steps,
                        states)


def ci_flavor_is_tensor_flag(flavor: bool) -> bool:
    # spelled out so the mirror reads exactly like the production call
    return bool(flavor)


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

#: The 21 kernel output planes (test_full_solve.py PLANES spelling).
PLANES = ["Tmrt", "Kdown", "Kup", "Ldown", "Lup", "Tg", "shadow",
          "Keast", "Ksouth", "Kwest", "Knorth", "Least", "Lsouth",
          "Lwest", "Lnorth", "KsideI", "TgOut", "Lside",
          "KsideD", "dRad", "Kside"]


def bits(a):
    a = np.asarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 else a.view(np.uint32)


def outputs_bit_equal(a: dict, b: dict, t: int) -> bool:
    for name in PLANES:
        if not np.array_equal(bits(a[name]), bits(b[name])):
            return False
    return True


def first_output_divergence(got: dict, ref: dict, t: int):
    """First diverging plane name at step ``t``, or None."""
    for name in PLANES:
        if not np.array_equal(bits(got[name]), bits(ref[name])):
            return name
    return None


# ---------------------------------------------------------------------------
# Validity fence (the classify_anchor rule §807 inherits)
# ---------------------------------------------------------------------------


def select_anchor(boundaries: dict, series, r0: int) -> Boundary | None:
    """max{a in boundaries : a <= r0 and met_prefix matches} (None -> no
    applicable checkpoint; a full recompute from 0 is the fallback)."""
    best = None
    for a in sorted(boundaries):
        b = boundaries[a]
        if a <= r0 and met_prefix_digest(series, a) == b.met_prefix:
            best = b
    return best


# ---------------------------------------------------------------------------
# Memory ledger + JSONL
# ---------------------------------------------------------------------------


def rss_kib() -> int:
    """Instantaneous RSS in KiB (ps; macOS/Linux)."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def loadavg_row() -> dict:
    a1, a5, a15 = os.getloadavg()
    return {"load1": round(a1, 2), "load5": round(a5, 2),
            "load15": round(a15, 2)}


def jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cap_set() -> "fs.CaptureSet":
    return fs.CaptureSet(cap_dir=CAP, profile=PROFILE, latitude=LATITUDE)


def unedited_series() -> list:
    n = json.loads((CAP / "manifest.json").read_text())["n_timesteps"]
    return [met_from_capture(np.load(CAP / f"t{t:02d}.npz"))
            for t in range(n)]
