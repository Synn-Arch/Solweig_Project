# SPDX-License-Identifier: GPL-3.0-only
"""T11 full-domain fallback solve gates.

* IDENTITY: unedited met -> every output plane + the threaded CI series
  equal the capture ``ret_*`` bits for ALL 24 timesteps (raw uint32).
* PROVENANCE: wrong-profile solves, and warm starts whose geometry
  fingerprint or met prefix disagree with the CURRENT inputs, are typed
  errors — never silent wrongness.
* EDITED-MET DIFFERENTIAL (the T10 handoff proof available torch-free):
  a genuinely perturbed suffix (Ta + 1.0 from t=12) drives a warm resume
  from a mid-window anchor and a fresh full solve of the SAME series;
  the two agree bitwise on every resumed output and the threaded CI.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from solweig_core.numba_cpu import full_solve as fs
from solweig_core.numba_cpu.met_recompute import MetInputs, met_from_capture

WT = Path(__file__).resolve().parents[2]
CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")
LATITUDE = 30.312645
PROFILE = "site_500-default"
EDIT_FROM = 12

PLANES = ["Tmrt", "Kdown", "Kup", "Ldown", "Lup", "Tg", "shadow",
          "Keast", "Ksouth", "Kwest", "Knorth", "Least", "Lsouth",
          "Lwest", "Lnorth", "KsideI", "TgOut", "Lside",
          "KsideD", "dRad", "Kside"]

MUTATION_ANCHORS = {
    "full_solve._apply_day_overrides.field_set":
        "test_full_solve_identity_all_timesteps",
    "full_solve.profile_fence":
        "test_profile_fence_wrong_profile",
    "full_solve.warm_fence.met_prefix":
        "test_warm_fence_refuses_changed_prefix",
    "full_solve.warm_fence.geometry_history":
        "test_warm_fence_refuses_geometry_mismatch",
    "full_solve.ci_threading.flavor":
        "test_full_solve_identity_all_timesteps",
    "full_solve.anchor_publication.self_classify":
        "test_anchor_publication_self_classifies",
}


def capture_present() -> bool:
    return (CAP / "t00.npz").exists()


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 else a.view(np.uint32)


def _cap_set() -> fs.CaptureSet:
    return fs.CaptureSet(cap_dir=CAP, profile=PROFILE, latitude=LATITUDE)


def _unedited_series() -> list[MetInputs]:
    n = json.loads((CAP / "manifest.json").read_text())["n_timesteps"]
    return [met_from_capture(np.load(CAP / f"t{t:02d}.npz"))
            for t in range(n)]


def _edited_series() -> list[MetInputs]:
    """Genuinely perturbed suffix: Ta + 1.0 from EDIT_FROM on."""
    return [replace(m, Ta=m.Ta + 1.0) if t >= EDIT_FROM else m
            for t, m in enumerate(_unedited_series())]


@pytest.fixture(scope="module")
def rows_cols():
    man = json.loads((CAP / "manifest.json").read_text())
    return int(man.get("rows", 500)), int(man.get("cols", 500))


@pytest.fixture(scope="module")
def solve_unedited(rows_cols):
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    rows, cols = rows_cols
    return fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        publish_anchor_at=[EDIT_FROM - 1])


@pytest.fixture(scope="module")
def solve_edited(rows_cols):
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    rows, cols = rows_cols
    return fs.full_solve_capture(
        _cap_set(), _edited_series(), profile=PROFILE, rows=rows, cols=cols,
        publish_anchor_at=[EDIT_FROM - 1])


# ---------------------------------------------------------------------------
# identity (the binding contract)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_full_solve_identity_all_timesteps(solve_unedited) -> None:
    """Unedited met: every ret plane + threaded CI bits, ALL 24 t."""
    mism: list[str] = []
    for t in range(solve_unedited.n_timesteps):
        z = np.load(CAP / f"t{t:02d}.npz")
        out = solve_unedited.outputs[t]
        for name in PLANES:
            key = f"ret_{name}"
            if key not in z.files:
                continue
            want, got = z[key], np.asarray(out[name])
            if want.shape != got.shape or want.dtype != got.dtype:
                mism.append(f"t{t:02d}/{key} shape/dtype drift")
            elif not np.array_equal(_bits(want), _bits(got)):
                mism.append(f"t{t:02d}/{key} bits differ")
        if "ret_CI__f64" in z.files:
            if float(z["ret_CI__f64"]) != solve_unedited.ret_CI_series[t]:
                mism.append(f"t{t:02d}/ret_CI threading drift")
    assert not mism, "; ".join(mism)


# ---------------------------------------------------------------------------
# provenance fences (RED witnesses: never silent, never wrong-profile)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_profile_fence_wrong_profile(rows_cols) -> None:
    """A solve naming a different profile than the capture's own is a
    typed error — the fallback result with the WRONG profile must not
    exist."""
    rows, cols = rows_cols
    with pytest.raises(fs.ProfileFenceError, match="profile"):
        fs.full_solve_capture(_cap_set(), None, profile="site_500-painted",
                              rows=rows, cols=cols)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_series_length_fence(rows_cols) -> None:
    rows, cols = rows_cols
    with pytest.raises(fs.FullSolveError, match="n_timesteps"):
        fs.full_solve_capture(_cap_set(), _unedited_series()[:-1],
                              profile=PROFILE, rows=rows, cols=cols)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_warm_fence_refuses_future_prefix(solve_unedited, rows_cols) -> None:
    """Anchor whose met prefix covers the edited row (r0=0 default: the
    whole series is treated as changed) -> future-prefix refusal."""
    rows, cols = rows_cols
    anchor_state, anchor_fp = solve_unedited.anchors[0]
    assert anchor_state.next_step == EDIT_FROM
    with pytest.raises(fs.StaleCheckpointError, match="future-prefix"):
        fs.full_solve_capture(
            _cap_set(), _edited_series(), profile=PROFILE,
            rows=rows, cols=cols,
            warm_state=anchor_state, warm_fingerprint=anchor_fp)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_warm_fence_refuses_changed_prefix(solve_unedited, rows_cols) -> None:
    """Anchor heard a DIFFERENT prefix than the current series at its own
    next_step (edit at t=5, request r0=15: clock position allows it, the
    digest does not) -> met-prefix refusal."""
    rows, cols = rows_cols
    anchor_state, anchor_fp = solve_unedited.anchors[0]
    series = [replace(m, Ta=m.Ta + 1.0) if t == 5 else m
              for t, m in enumerate(_unedited_series())]
    with pytest.raises(fs.StaleCheckpointError, match="met-prefix"):
        fs.full_solve_capture(
            _cap_set(), series, profile=PROFILE,
            rows=rows, cols=cols, r0=15,
            warm_state=anchor_state, warm_fingerprint=anchor_fp)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_warm_fence_refuses_geometry_mismatch(solve_unedited, rows_cols) -> None:
    """Stale scene lineage (a previous geometry) can never warm-start the
    current capture, however matching the clock/met prefix is."""
    rows, cols = rows_cols
    anchor_state, anchor_fp = solve_unedited.anchors[0]
    bad_fp = dict(anchor_fp)
    bad_fp["composed_scene"] = "0" * 64
    with pytest.raises(fs.StaleCheckpointError, match="geometry-history"):
        fs.full_solve_capture(
            _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
            r0=EDIT_FROM,
            warm_state=anchor_state, warm_fingerprint=bad_fp)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_warm_fence_refuses_fingerprintless(solve_unedited, rows_cols) -> None:
    rows, cols = rows_cols
    anchor_state, _ = solve_unedited.anchors[0]
    with pytest.raises(fs.StaleCheckpointError, match="fingerprint"):
        fs.full_solve_capture(
            _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
            warm_state=anchor_state, warm_fingerprint=None)


# ---------------------------------------------------------------------------
# edited-met behavior + the warm-vs-fresh internal differential
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_edited_suffix_changes_met_outputs_not_geometry(
        solve_unedited, solve_edited) -> None:
    """The perturbed suffix changes the met-dependent outputs from its
    first row, and leaves the frozen GEOMETRY planes untouched (shadow
    bits identical for every t — an edit never repaints geometry)."""
    for t in range(solve_edited.n_timesteps):
        sh_old = np.asarray(solve_unedited.outputs[t]["shadow"])
        sh_new = np.asarray(solve_edited.outputs[t]["shadow"])
        assert np.array_equal(_bits(sh_old), _bits(sh_new)), \
            f"t{t:02d} shadow bits drifted under a MET edit"
    t = EDIT_FROM
    old = np.asarray(solve_unedited.outputs[t]["Tmrt"])
    new = np.asarray(solve_edited.outputs[t]["Tmrt"])
    assert not np.array_equal(_bits(old), _bits(new)), \
        "perturbed Ta suffix left Tmrt untouched at its first edited row"
    # prefix rows are served from the same inputs -> identical bits
    t = EDIT_FROM - 1
    assert np.array_equal(
        _bits(np.asarray(solve_unedited.outputs[t]["Tmrt"])),
        _bits(np.asarray(solve_edited.outputs[t]["Tmrt"]))), \
        "unedited prefix row drifted"


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_warm_vs_fresh_differential_edited_met(solve_edited, rows_cols) -> None:
    """T10 handoff proof (torch-free): warm resume from a mid-window
    anchor of the SAME edited series reproduces the fresh full solve
    bitwise — outputs, and the driver-threaded CI series."""
    rows, cols = rows_cols
    anchor_state, anchor_fp = solve_edited.anchors[0]
    warm = fs.full_solve_capture(
        _cap_set(), _edited_series(), profile=PROFILE, rows=rows, cols=cols,
        warm_state=anchor_state, warm_fingerprint=anchor_fp, r0=EDIT_FROM)
    assert warm.outputs[len(warm.outputs) - 1] is not None
    for i, t in enumerate(range(EDIT_FROM, solve_edited.n_timesteps)):
        for name in PLANES:
            fresh = np.asarray(solve_edited.outputs[t][name])
            got = np.asarray(warm.outputs[i][name])
            assert np.array_equal(_bits(fresh), _bits(got)), \
                f"t{t:02d}/{name} warm != fresh (edited-met differential)"
        assert warm.ret_CI_series[i] == solve_edited.ret_CI_series[t], \
            f"t{t:02d} CI warm != fresh"
    assert warm.final_state.CI == solve_edited.final_state.CI


# ---------------------------------------------------------------------------
# warm-anchor publication
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_anchor_publication_self_classifies(solve_unedited) -> None:
    """The published anchor carries its fingerprint and self-classifies
    as servable (classify_anchor governs; nothing touches the disk)."""
    assert solve_unedited.anchor_published is not None
    assert solve_unedited.anchor_refusal is None
    fp = solve_unedited.anchor_fingerprint
    from solweig_core.numba_cpu import thermal
    assert set(thermal.GEOMETRY_FINGERPRINT_KEYS) <= set(fp)
    assert "met_prefix" in fp
    # a solve warm-starting from it advances zero steps (window covered)
    rows, cols = 500, 500
    covered = fs.full_solve_capture(
        _cap_set(), None, profile=PROFILE, rows=rows, cols=cols,
        warm_state=solve_unedited.anchor_published,
        warm_fingerprint=solve_unedited.anchor_fingerprint,
        r0=solve_unedited.n_timesteps)
    assert covered.outputs == []
    assert covered.final_state.CI == solve_unedited.final_state.CI


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_twater_midnight_schedule_bits() -> None:
    """Schedule advance: Twater refreshes only at i==0 and day
    boundaries, from the current day's Ta rows (numpy f64 mean)."""
    series = _unedited_series()
    n = len(series)
    dectime = [float(np.load(CAP / f"t{t:02d}.npz")["arg_dectime__f64"])
               for t in range(n)]
    sched = fs.twater_midnight_schedule(series, dectime)
    assert len(sched) == n
    ta = np.asarray([m.Ta for m in series], dtype=np.float64)
    floors = np.floor(np.asarray(dectime))
    for i, v in enumerate(sched):
        if i == 0 or floors[i] != floors[i - 1]:
            assert v == float(np.mean(ta[floors == floors[i]]))
        else:
            assert v == sched[i - 1]
