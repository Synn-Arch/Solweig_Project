# SPDX-License-Identifier: GPL-3.0-only
"""T20 equality-triggered temporal convergence gates (DESIGN §589).

The dirty recurrence may reuse a committed run's suffix ONLY at a
boundary where (a) every future forcing row is bit-identical AND (b) the
carried state is bit-identical — six planes + firstdaytime/timeadd/CI +
the threaded f64 CI + its flavor. Partial equality or
small-temperature early exit is FORBIDDEN; the gates below try to sneak
a partial equality past the driver and refuse to pass if it ever fires.

Gates:
* IDENTITY RESUME — warm start + identical series fires at the entry
  boundary and the copied suffix is raw-bit identical to both the
  unassisted replay and the reference run (the §589 induction, witnessed
  end to end).
* EDITED-ROW RECONVERGENCE — a single-row diffuse-radiation edit whose
  perturbation the f32 recurrence absorbs (probe-measured: hit at
  boundary 13 on this capture) fires and reproduces the UNASSISTED
  edited replay bit-for-bit — the reused suffix is what physics would
  have recomputed, not the pre-edit bytes.
* PARTIAL EQUALITY REFUSAL — 1-ULP-corrupting EVERY boundary plane (one
  mantissa/sign bit per finite cell) must prevent the fire; the result
  stays bit-identical to the unassisted replay, and the poisoned
  reference bytes are never served. This is the designated killer for
  M-T20-1 (state comparison weakened to Tgmap1 only) and M-T20-1b
  (tolerance compare: a 1-ULP / ±0.0 cell passes allclose, fails bits).
* FUTURE-INPUT IDENTITY — a single edited row anywhere in [s, n) must
  prevent the fire (killer for M-T20-2, current-row-only check).
* FOREIGN REFERENCE — wrong profile / window / scene lineage / series
  length is a typed refusal, never a silent ignore, never a wrong reuse.
* ANCHOR PRECEDENCE — a requested mid-anchor forfeits ITS OWN boundary
  (anchors outrank convergence): the fire defers to the next clean
  boundary, or is fully forfeited when an anchor blocks them all;
  both are honestly recorded.

torch-free like the module under test; capture-dependent tests skip
when the compact capture bundle is absent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from solweig_core.numba_cpu import full_solve as fs
from solweig_core.numba_cpu import thermal
from solweig_core.numba_cpu.met_recompute import met_from_capture
from solweig_core.numba_cpu.radiation import RadLoopState

REPO_ROOT = Path(__file__).resolve().parents[2]
CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14"
           "/compact_capture")
PROFILE = "site_500-default"
LATITUDE = 30.312645

#: probe-measured convergence shape on THIS capture
#: (~/Workspace/solweig_ultrafast_artifacts/t20/probe_boundaries.jsonl):
#: a radD*0.5 edit at day row 12 is absorbed by the f32 recurrence and
#: the boundary-13 carried state returns to bit-equality.
EXPECT_EDIT12_FIRE_AT = 13

#: designated mutations and the gates that must fail under them
MUTATION_ANCHORS = {
    "M-T20-1": "test_partial_state_equality_is_refused_all_planes",
    "M-T20-1b": "test_partial_state_equality_is_refused_all_planes",
    "M-T20-2": "test_single_future_row_edit_prevents_fire",
}

PLANES = ["Tmrt", "Kdown", "Kup", "Ldown", "Lup", "Tg", "shadow",
          "Keast", "Ksouth", "Kwest", "Knorth", "Least", "Lsouth",
          "Lwest", "Lnorth", "KsideI", "TgOut", "Lside",
          "KsideD", "dRad", "Kside"]


def capture_present() -> bool:
    return (CAP / "t00.npz").is_file()


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 \
        else a.view(np.uint32)


def _cap_set() -> fs.CaptureSet:
    return fs.CaptureSet(cap_dir=str(CAP), profile=PROFILE,
                         latitude=LATITUDE)


def _capture_series() -> list:
    n = json.loads((CAP / "manifest.json").read_text())["n_timesteps"]
    return [met_from_capture(np.load(CAP / f"t{t:02d}.npz"))
            for t in range(n)]


def _boundary_from_anchor(anchor_state, result: fs.FullSolveResult,
                          step: int, rows: int, cols: int
                          ) -> fs.ConvergenceBoundary:
    """The reference's carried facts at absolute boundary ``step`` from
    its published anchor + CI thread bookkeeping (the driver restores
    warm states through the same ``restore_loop_state`` path)."""
    loop_state, _tw = thermal.restore_loop_state(
        anchor_state, grid_shape=(rows, cols))
    return fs.ConvergenceBoundary(
        state=loop_state,
        ci_thread=result.ret_CI_series[step - 1],
        ci_flavor=result.ret_CI_flavors[step - 1],
    )


def _assert_outputs_equal(got: fs.FullSolveResult,
                          want: fs.FullSolveResult,
                          *, abs_steps: range, got_base: int,
                          want_base: int, msg: str) -> None:
    """Raw-bit equality over ABSOLUTE timesteps; ``*_base`` is each
    result's own first executed step (outputs are run-relative lists)."""
    for t in abs_steps:
        g, w = t - got_base, t - want_base
        for name in PLANES:
            assert np.array_equal(_bits(got.outputs[g][name]),
                                  _bits(want.outputs[w][name])), \
                f"{msg}: t{t:02d}/{name} bits differ"
        assert got.ret_CI_series[g] == want.ret_CI_series[w], \
            f"{msg}: t{t:02d} threaded CI differs"


def _digest_outputs(result: fs.FullSolveResult) -> str:
    h = hashlib.sha256()
    for out in result.outputs:
        for k in sorted(out):
            v = out[k]
            if isinstance(v, np.ndarray):
                h.update(k.encode())
                h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def _poisoned_outputs_11up(ref: fs.ConvergenceReference
                           ) -> tuple:
    """Fresh copies of the reference's served outputs with a sentinel
    added to Tmrt[0, 0] from step 11 on: ANY fire that serves the
    reference surfaces these bytes in the run's digest."""
    out = []
    for t, d in enumerate(ref.outputs):
        if t < 11:
            out.append(d)
            continue
        p = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray)
                 else v) for k, v in d.items()}
        p["Tmrt"][0, 0] = np.float32(float(p["Tmrt"][0, 0]) + 99.0)
        out.append(p)
    return tuple(out)


def _ulp_poison_planes(planes: dict, clean: str | None = None) -> dict:
    """Flip the last mantissa bit of every FINITE cell of every plane
    except ``clean`` (zeros flip their SIGN bit: the ±0.0 case a
    tolerance compare accepts and a bits compare refuses). Non-finite
    cells are left alone."""
    out = {}
    for name, p in planes.items():
        if name == clean:
            out[name] = np.asarray(p)
            continue
        src = np.ascontiguousarray(p, dtype=np.float32)
        u = src.view(np.uint32).copy()  # copy: XOR must not touch src
        u[np.isfinite(src)] ^= np.uint32(1)
        bad = u.view(np.float32)
        assert not np.array_equal(bad, src), name
        # numerically adjacent everywhere (the tolerance-compare trap:
        # a default-tolerance allclose accepts every poisoned cell)
        assert np.allclose(bad, src, equal_nan=True), name
        out[name] = bad
    return out


@pytest.fixture(scope="module")
def rows_cols():
    man = json.loads((CAP / "manifest.json").read_text())
    grid = man.get("grid", {})
    return int(grid.get("rows", 500)), int(grid.get("cols", 500))


@pytest.fixture(scope="module")
def ref_cold(rows_cols):
    """The committed run: cold identity solve publishing the boundary
    anchors its convergence references are built from."""
    if not capture_present():
        pytest.skip("compact capture absent")
    rows, cols = rows_cols
    return fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        # published AFTER completing step s-1 → anchor next_step == s:
        # boundary-s states for s in {1, 11, 12, 13}
        publish_anchor_at=[0, 10, 11, 12])


@pytest.fixture(scope="module")
def ref_anchor(ref_cold):
    return {a_state.next_step: (a_state, fp)
            for a_state, fp in ref_cold.anchors}


@pytest.fixture(scope="module")
def ref_reference(ref_cold, ref_anchor, rows_cols):
    rows, cols = rows_cols
    boundaries = {
        s: _boundary_from_anchor(ref_anchor[s][0], ref_cold, s, rows, cols)
        for s in (11, 12, 13)
    }
    return fs.ConvergenceReference.from_result(
        ref_cold, met_series=_capture_series(), start_step=0,
        geometry_fingerprint=fs.geometry_fingerprint_of_capture(_cap_set()),
        boundaries=boundaries)


# ---------------------------------------------------------------------------
# identity resume (the §589 entry case — deterministic by construction)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="compact capture absent")
def test_identity_warm_resume_fires_at_entry(ref_cold, ref_anchor,
                                             ref_reference, rows_cols):
    """Warm start from the reference's own anchor + identical series:
    fires at the entry boundary; the copied suffix is raw-bit equal to
    BOTH the unassisted warm replay and the reference run."""
    rows, cols = rows_cols
    anchor_state, anchor_fp = ref_anchor[11]
    digest_before = _digest_outputs(ref_cold)

    conv_run = fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        warm_state=anchor_state, warm_fingerprint=anchor_fp, r0=11,
        convergence_reference=ref_reference)
    plain_run = fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        warm_state=anchor_state, warm_fingerprint=anchor_fp, r0=11)

    conv = conv_run.convergence
    assert conv is not None and conv["fired_at"] == 11, \
        f"identity resume must fire at boundary 11, got {conv}"
    assert conv["reused_steps"] == 24 - 11
    assert conv["copied_output_bytes"] > 0
    assert len(conv_run.outputs) == 13
    assert len(conv_run.timings_ms) == 0  # no step was executed

    # the copied suffix is the unassisted replay's bits...
    _assert_outputs_equal(conv_run, plain_run, abs_steps=range(11, 24),
                          got_base=11, want_base=11,
                          msg="converged vs unassisted warm replay")
    # ...and the reference run's own bits (the §589 induction)
    _assert_outputs_equal(conv_run, ref_cold, abs_steps=range(11, 24),
                          got_base=11, want_base=0,
                          msg="converged vs reference run")
    # terminal state: bit-identical to both
    assert fs._state_bits_equal(conv_run.final_state, plain_run.final_state)
    assert fs._state_bits_equal(conv_run.final_state, ref_cold.final_state)

    # the reference was not corrupted by serving the suffix
    assert _digest_outputs(ref_cold) == digest_before


@pytest.mark.skipif(not capture_present(), reason="compact capture absent")
def test_edited_row_reconvergence_replays_not_recycles(ref_cold,
                                                       ref_reference,
                                                       rows_cols):
    """radD*0.5 at day row 12 (probe: f32-absorbed by boundary 13): the
    fire must reproduce the UNASSISTED EDITED replay — the reused
    suffix is recomputed physics, not pre-edit bytes."""
    rows, cols = rows_cols
    series = [replace(m, radD=m.radD * 0.5) if t == 12 else m
              for t, m in enumerate(_capture_series())]
    unassisted = fs.full_solve_capture(
        _cap_set(), series, profile=PROFILE, rows=rows, cols=cols,
        publish_anchor=False)
    conv_run = fs.full_solve_capture(
        _cap_set(), series, profile=PROFILE, rows=rows, cols=cols,
        publish_anchor=False, convergence_reference=ref_reference)

    conv = conv_run.convergence
    assert conv is not None and conv["fired_at"] == EXPECT_EDIT12_FIRE_AT, \
        (f"expected the probe-measured fire at boundary "
         f"{EXPECT_EDIT12_FIRE_AT}, got {conv} — the f32 re-convergence "
         "shape drifted; re-run probe_eqconv.py before re-pinning")
    # the pre-boundary prefix (0..12) was genuinely executed...
    assert len(conv_run.timings_ms) == EXPECT_EDIT12_FIRE_AT
    # ...and the whole window equals the unassisted edited replay
    _assert_outputs_equal(conv_run, unassisted, abs_steps=range(0, 24),
                          got_base=0, want_base=0,
                          msg="converged vs unassisted edited replay")
    assert fs._state_bits_equal(conv_run.final_state,
                                unassisted.final_state)


# ---------------------------------------------------------------------------
# partial equality must NEVER fire (M-T20-1 / M-T20-1b killer)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="compact capture absent")
@pytest.mark.parametrize("clean_plane", list(fs._LOOP_PLANE_NAMES))
def test_partial_state_equality_is_refused_all_planes(ref_reference,
                                                      rows_cols,
                                                      clean_plane):
    """Five of six boundary planes are 1-ULP-corrupted (``clean_plane``
    stays exact; zeros elsewhere sign-flip): the true all-bits key
    refuses; the run stays bit-identical to the unassisted replay and
    the poisoned reference bytes are NEVER served.

    Kills M-T20-1 (state comparison restricted to Tgmap1): in the
    clean_plane=Tgmap1 case the weakened key's one inspected plane is
    EXACT, so it fires and serves poison — this gate fails. Kills
    M-T20-1b (tolerance compare): every poisoned cell is allclose-equal
    to the live value, so a tolerance key fires in EVERY case."""
    rows, cols = rows_cols
    plain = fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        publish_anchor=False)
    plain_bits = _digest_outputs(plain)

    b11 = ref_reference.boundaries[11]
    planes = {name: np.asarray(getattr(b11.state, name))
              for name in fs._LOOP_PLANE_NAMES}
    bad_planes = _ulp_poison_planes(planes, clean=clean_plane)
    bad_b11 = fs.ConvergenceBoundary(
        state=RadLoopState(firstdaytime=b11.state.firstdaytime,
                           timeadd=b11.state.timeadd,
                           CI=np.float32(b11.state.CI),
                           **bad_planes),
        ci_thread=b11.ci_thread, ci_flavor=b11.ci_flavor)
    # ONLY boundary 11 is offered: the poisoned state must be the sole
    # reason the fire is refused (later genuine boundaries could fire
    # legitimately and are deliberately not part of this gate)
    poisoned = replace(ref_reference,
                       boundaries={11: bad_b11},
                       outputs=_poisoned_outputs_11up(ref_reference))

    run = fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        publish_anchor=False, convergence_reference=poisoned)
    assert run.convergence is None, \
        f"partial equality fired: {run.convergence}"
    assert _digest_outputs(run) == plain_bits, \
        "a partial-equality fire surfaced poisoned reference bytes"


# ---------------------------------------------------------------------------
# future-input identity is required (M-T20-2 killer)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="compact capture absent")
def test_single_future_row_edit_prevents_fire(ref_cold, ref_anchor,
                                              rows_cols):
    """One edited night row anywhere in the future (t=22) must prevent
    the fire even though the boundary-1 state is bit-identical (night
    steps carry no plane perturbation). Copying the reference's row-22
    bytes would serve pre-edit physics — the gate refuses to pass if
    the driver ever does.

    Kills M-T20-2 (future-input identity reduced to row t: at boundary
    1 row 1 is unedited so the weakened key fires at 1 and serves the
    reference's stale suffix)."""
    rows, cols = rows_cols
    anchor_state, anchor_fp = ref_anchor[1]
    ref = fs.ConvergenceReference.from_result(
        ref_cold, met_series=_capture_series(), start_step=0,
        geometry_fingerprint=fs.geometry_fingerprint_of_capture(_cap_set()),
        boundaries={1: _boundary_from_anchor(anchor_state, ref_cold, 1,
                                             rows, cols)})
    series = [replace(m, Ta=m.Ta + 1.0) if t == 22 else m
              for t, m in enumerate(_capture_series())]
    unassisted = fs.full_solve_capture(
        _cap_set(), series, profile=PROFILE, rows=rows, cols=cols,
        publish_anchor=False)
    run = fs.full_solve_capture(
        _cap_set(), series, profile=PROFILE, rows=rows, cols=cols,
        publish_anchor=False, convergence_reference=ref)
    assert run.convergence is None, \
        f"fire with an edited future row: {run.convergence}"
    _assert_outputs_equal(run, unassisted, abs_steps=range(0, 24),
                          got_base=0, want_base=0,
                          msg="edited-row run vs unassisted replay")
    # and the stale reference bytes were NOT served at the edited row
    assert not np.array_equal(_bits(run.outputs[22]["Tmrt"]),
                              _bits(ref_cold.outputs[22]["Tmrt"])), \
        "row 22 output equals the pre-edit reference — stale physics"


# ---------------------------------------------------------------------------
# foreign reference: typed refusal (never silent, never wrong reuse)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="compact capture absent")
def test_foreign_reference_is_typed_refusal(ref_reference, rows_cols):
    rows, cols = rows_cols
    wrong_profile = replace(ref_reference, profile="site_500-other")
    with pytest.raises(fs.FullSolveError, match="profile"):
        fs.full_solve_capture(_cap_set(), None, profile=PROFILE,
                              rows=rows, cols=cols,
                              convergence_reference=wrong_profile)
    wrong_window = replace(ref_reference, n_timesteps=23)
    with pytest.raises(fs.FullSolveError, match="window"):
        fs.full_solve_capture(_cap_set(), None, profile=PROFILE,
                              rows=rows, cols=cols,
                              convergence_reference=wrong_window)
    bad_fp = dict(ref_reference.geometry_fingerprint)
    bad_fp["composed_scene"] = "0" * 64
    wrong_scene = replace(ref_reference, geometry_fingerprint=bad_fp)
    with pytest.raises(fs.FullSolveError, match="scene lineage"):
        fs.full_solve_capture(_cap_set(), None, profile=PROFILE,
                              rows=rows, cols=cols,
                              convergence_reference=wrong_scene)
    short_series = replace(ref_reference,
                           met_series=ref_reference.met_series[:-1])
    with pytest.raises(fs.FullSolveError, match="series length"):
        fs.full_solve_capture(_cap_set(), None, profile=PROFILE,
                              rows=rows, cols=cols,
                              convergence_reference=short_series)


# ---------------------------------------------------------------------------
# anchors outrank convergence
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="compact capture absent")
def test_requested_mid_anchor_defers_then_blocks_the_fire(
        ref_cold, ref_anchor, ref_reference, rows_cols):
    """Anchor precedence is PER BOUNDARY: a requested anchor at 11
    forfeits boundary 11 only — the fire defers to boundary 12 (no
    anchor lies at/after 12, and the executed step 11 keeps the carried
    state on the reference trajectory), and the record says both. With
    an anchor at 23, EVERY boundary is blocked and nothing is reused.
    Either way every requested anchor is produced and the outputs stay
    bit-identical to the unassisted run."""
    rows, cols = rows_cols
    anchor_state, anchor_fp = ref_anchor[11]

    def run_with(publish_at):
        return fs.full_solve_capture(
            _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
            warm_state=anchor_state, warm_fingerprint=anchor_fp, r0=11,
            publish_anchor_at=publish_at,
            convergence_reference=ref_reference)

    defer = run_with([11])
    plain = fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        warm_state=anchor_state, warm_fingerprint=anchor_fp, r0=11,
        publish_anchor_at=[11])
    conv = defer.convergence
    assert conv is not None and conv["fired_at"] == 12 \
        and conv["suppressed_at"] == 11, \
        f"expected suppression at 11 with a deferred fire at 12, got {conv}"
    assert conv["reused_steps"] == 12
    _assert_outputs_equal(defer, plain, abs_steps=range(11, 24),
                          got_base=11, want_base=11,
                          msg="deferred run vs unassisted replay")
    assert len(defer.anchors) == 1 and defer.anchors[0][0].next_step == 12

    block = run_with([11, 23])
    conv_b = block.convergence
    assert conv_b is not None and conv_b["fired_at"] is None \
        and conv_b["suppressed_at"] == 11, \
        f"expected a fully forfeited fire, got {conv_b}"
    assert len(block.timings_ms) == 13  # every step executed
    _assert_outputs_equal(block, plain, abs_steps=range(11, 24),
                          got_base=11, want_base=11,
                          msg="blocked run vs unassisted replay")
    assert [a[0].next_step for a in block.anchors] == [12, 24]


# ---------------------------------------------------------------------------
# default behavior: no reference -> unchanged, honestly recorded
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="compact capture absent")
def test_no_reference_default_unchanged(rows_cols):
    rows, cols = rows_cols
    run = fs.full_solve_capture(_cap_set(), None, profile=PROFILE,
                                rows=rows, cols=cols,
                                publish_anchor_at=[11])
    assert run.convergence is None
    assert len(run.ret_CI_flavors) == run.n_timesteps
    assert all(isinstance(f, bool) for f in run.ret_CI_flavors)


# ---------------------------------------------------------------------------
# predicate pins (fast, capture-free): the equality key is all-bits
# ---------------------------------------------------------------------------

def _mk_state(planes, fdt=1, ta=0.0625, ci=np.float32(0.9)):
    return RadLoopState(firstdaytime=fdt, timeadd=ta, CI=np.float32(ci),
                        **planes)


def _rng_planes():
    rng = np.random.default_rng(3)
    return {n: (rng.standard_normal((3, 2)) * 300).astype(np.float32)
            for n in fs._LOOP_PLANE_NAMES}


def test_predicate_plane_bits_nan_payload_and_signed_zero() -> None:
    p = _rng_planes()
    p["TgOut1"][0, 0] = np.float32(0.0)
    a = _mk_state(p)
    b = _mk_state({k: v.copy() for k, v in p.items()})
    assert fs._state_bits_equal(a, b)
    z = _mk_state({k: v.copy() for k, v in p.items()})
    z.TgOut1[0, 0] = np.float32(-0.0)
    assert float(z.TgOut1[0, 0]) == 0.0
    assert not fs._state_bits_equal(a, z), "plane signed zero distinguished"
    from struct import pack, unpack
    payload = unpack(">I", pack(">f", float("nan")))[0] | 0x00400000
    nan2 = np.frombuffer(pack(">I", payload), dtype=np.float32)[0]
    n1 = _mk_state({k: v.copy() for k, v in p.items()})
    n2 = _mk_state({k: v.copy() for k, v in p.items()})
    n1.Tgmap1[2, 1] = np.float32("nan")
    n2.Tgmap1[2, 1] = nan2
    assert not fs._state_bits_equal(n1, n2), "NaN payload distinguished"
    s = _mk_state({k: v.copy() for k, v in p.items()})
    s.Tgmap1N[1, 0] = np.float32(12345.0)
    assert not fs._state_bits_equal(a, s), "one-cell plane drift refuses"


def test_predicate_scalars_all_bits() -> None:
    p = _rng_planes()
    a = _mk_state(p)
    assert not fs._state_bits_equal(a, _mk_state(p, fdt=0)), "firstdaytime"
    assert not fs._state_bits_equal(a, _mk_state(p, ta=-0.0)), \
        "timeadd signed zero"
    assert not fs._state_bits_equal(a, _mk_state(p, ci=np.float32(-0.0))), \
        "CI signed zero"


def test_row_bits_are_full_field_raw() -> None:
    # synthetic: capture night rows may already carry the list sentinel
    from solweig_core.numba_cpu.met_recompute import MetInputs
    m = MetInputs(Ta=1.0, RH=2.0, radG=3.0, radD=4.0, radI=5.0, P=6.0)
    assert fs._row_bits(replace(m)) == fs._row_bits(m)
    assert fs._row_bits(replace(m, Twater=-0.0)) != \
        fs._row_bits(replace(m, Twater=0.0)), "Twater signed zero"
    assert fs._row_bits(replace(m, Twater_is_list=True)) != \
        fs._row_bits(m), "Twater list sentinel distinguished"
    assert fs._future_inputs_equal([m, replace(m)], [m, replace(m)], 0)
    assert not fs._future_inputs_equal(
        [m, replace(m, Ta=m.Ta + 1.0)], [m, replace(m)], 0), \
        "an edited future row is caught"
    assert fs._future_inputs_equal(
        [m, replace(m, Ta=m.Ta + 1.0)], [m, replace(m, Ta=m.Ta + 1.0)], 0)
