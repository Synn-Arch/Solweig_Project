# SPDX-License-Identifier: GPL-3.0-only
"""T21 gates: prefix-accumulator suffix fold ablation (DESIGN §807).

The §807 candidate stores the EXACT accumulator prefix of the temporal
canonical fold at committed boundaries and re-runs the suffix fold from
there (never subtract-old/add-new). Gates:

* IDENTITY      — the harness loop IS the production fold: bit-equal to
                  ``full_solve_capture`` on all 21 planes raw uint32, the
                  f64 CI series, the per-t flavors, and the final carried
                  state (including the f64 CI thread bits).
* FIDELITY      — checkpoint at boundary ``a``, dirty edit first changed
                  row ``r0 >= a``: restore + suffix ``[a, 24)`` equals the
                  full dirty recompute bitwise (outputs, CI series bits,
                  flavors, final state).
* ORACLE        — a restored suffix on the UNEDITED series equals the
                  capture's oracle-recorded ``ret_*`` bits (independent
                  of any harness run).
* MUTATIONS     — dropping each carried accumulator (timeadd,
                  firstdaytime, a plane bit) is DETECTED by the same
                  comparators; the two CI-thread mutations pin their
                  measured behaviour (M1 collapse, M2 flavor).
* FENCE         — a boundary whose met prefix covers the edit row is
                  never selected (classify_anchor's future-prefix rule).
* LEDGER        — the checkpoint memory accounting cited by the report
                  (6,000,069 B per boundary at site_500; k-sweep vs the
                  256 MiB steady target).

Heavy folds are module fixtures (one reference pass + one dirty pass
serve every test). Skips cleanly when the t08 capture bundle is absent.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from solweig_core.numba_cpu import full_solve as fs
from solweig_core.numba_cpu import thermal

from tests.ultrafast import t21_suffix_fold as t21

CAP = t21.CAP
ANCHOR = 12
EDIT_FROM = 12


def capture_present() -> bool:
    return t21.capture_present()


def _edited_series(series, r0: int):
    return [replace(m, Ta=m.Ta + 1.0) if t >= r0 else m
            for t, m in enumerate(series)]


@pytest.fixture(scope="module")
def series():
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    return t21.unedited_series()


@pytest.fixture(scope="module")
def production(series):
    """The production driver's own full pass (identity reference)."""
    return fs.full_solve_capture(t21.cap_set(), None, profile=t21.PROFILE,
                                 publish_anchor=False)


@pytest.fixture(scope="module")
def reference(series):
    """Harness full pass with a Boundary captured after EVERY step."""
    from tests.ultrafast.t21_ablation_run import run_window_with_boundaries
    win, boundaries = run_window_with_boundaries(t21.cap_set(), series)
    return win, boundaries


@pytest.fixture(scope="module")
def dirty12(series):
    """Full dirty recompute with Ta + 1.0 from row 12 (the comparison
    target any restore-based suffix must reproduce)."""
    from tests.ultrafast.t21_ablation_run import run_window_with_boundaries
    es = _edited_series(series, EDIT_FROM)
    win, _ = run_window_with_boundaries(t21.cap_set(), es)
    final = t21.capture_boundary(win.final_state, win.ci_thread,
                                 win.ci_flavor, t21.cap_set().n_timesteps, es)
    return es, win, final


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_harness_loop_matches_production_full_solve(production, reference) -> None:
    """The harness fold loop is the production fold: all 21 planes of all
    24 timesteps raw-bit equal, plus the f64 CI series bits, per-t
    flavors, and the final carried state (f64 thread bits included)."""
    win, boundaries = reference
    n = production.n_timesteps
    for t in range(n):
        assert t21.first_output_divergence(
            win.outputs[t], production.outputs[t], t) is None, \
            f"t{t:02d}: harness loop != production fold"
    assert all(t21.f64_bits(x) == t21.f64_bits(y)
               for x, y in zip(win.ret_ci_series, production.ret_CI_series))
    assert win.ret_ci_flavors == list(production.ret_CI_flavors)
    prod_final = t21.capture_boundary(
        production.final_state, production.ret_CI_series[-1],
        production.ret_CI_flavors[-1], n, t21.unedited_series())
    assert t21.boundary_bits_equal(boundaries[n], prod_final)


# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_restore_fidelity_anchor_eq_edit(series, dirty12, reference) -> None:
    """Edit lands at r0 == 12; the boundary at 12 restores exactly."""
    _, boundaries = reference
    es, dwin, dfinal = dirty12
    suf = t21.run_window(t21.cap_set(), es, boundary=boundaries[ANCHOR])
    for i, t in enumerate(range(ANCHOR, 24)):
        assert t21.first_output_divergence(
            suf.outputs[i], dwin.outputs[t], t) is None, \
            f"t{t:02d}: restored suffix != full dirty recompute"
    assert all(t21.f64_bits(x) == t21.f64_bits(y)
               for x, y in zip(suf.ret_ci_series, dwin.ret_ci_series[ANCHOR:]))
    assert suf.ret_ci_flavors == dwin.ret_ci_flavors[ANCHOR:]
    suf_final = t21.capture_boundary(suf.final_state, suf.ci_thread,
                                     suf.ci_flavor, 24, es)
    assert t21.boundary_bits_equal(suf_final, dfinal)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_restore_fidelity_anchor_older_than_edit(series, dirty12,
                                                 reference) -> None:
    """Boundary at 8 with the edit at r0 == 12: rows [8, 12) re-fold with
    UNCHANGED inputs, so the restore stays exact across them."""
    _, boundaries = reference
    a = 8
    es, dwin, dfinal = dirty12
    suf = t21.run_window(t21.cap_set(), es, boundary=boundaries[a])
    for i, t in enumerate(range(a, 24)):
        assert t21.first_output_divergence(
            suf.outputs[i], dwin.outputs[t], t) is None, \
            f"t{t:02d}: mid-anchor restore != full dirty recompute"
    assert all(t21.f64_bits(x) == t21.f64_bits(y)
               for x, y in zip(suf.ret_ci_series, dwin.ret_ci_series[a:]))
    suf_final = t21.capture_boundary(suf.final_state, suf.ci_thread,
                                     suf.ci_flavor, 24, es)
    assert t21.boundary_bits_equal(suf_final, dfinal)


# ---------------------------------------------------------------------------
# independent oracle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_restored_suffix_matches_oracle_capture(series, reference) -> None:
    """Per-step, the restored suffix equals the ORACLE's own recorded
    outputs AND next-state (ret_Tgmap1 family, ret_timeadd,
    ret_firstdaytime, ret_CI) — independent of every harness run."""
    _, boundaries = reference
    suf = t21.run_window(t21.cap_set(), series,
                         boundary=boundaries[ANCHOR], collect_states=True)
    for i, t in enumerate(range(ANCHOR, 24)):
        z = np.load(CAP / f"t{t:02d}.npz")
        for name in t21.PLANES:
            key = f"ret_{name}"
            if key not in z.files:
                continue
            assert np.array_equal(t21.bits(suf.outputs[i][name]),
                                  t21.bits(z[key])), \
                f"t{t:02d}/{name}: restored suffix != oracle capture bits"
        if "ret_CI__f64" in z.files:
            assert t21.f64_bits(suf.ret_ci_series[i]) == \
                t21.f64_bits(float(z["ret_CI__f64"])), f"t{t:02d} CI thread"
        else:
            assert suf.ret_ci_series[i] is not None
        # carried state AFTER this step vs the oracle's recorded next-state
        st, ci_thread, _flav = suf.states[i]
        for name in t21.LOOP_PLANES:
            key = f"ret_{name}"
            assert key in z.files and np.array_equal(
                t21.bits(getattr(st, name)), t21.bits(z[key])), \
                f"t{t:02d}/{name}: carried state != oracle next-state"
        if "ret_timeadd__f64" in z.files:
            assert t21.f64_bits(st.timeadd) == \
                t21.f64_bits(float(z["ret_timeadd__f64"])), \
                f"t{t:02d} timeadd"
        assert int(st.firstdaytime) == \
            int(z["ret_firstdaytime__int"][()]), f"t{t:02d} firstdaytime"
        if "ret_CI__f64" in z.files:
            assert t21.f32_bits(st.CI) == \
                t21.f32_bits(float(z["ret_CI__f64"])), f"t{t:02d} state CI"


# ---------------------------------------------------------------------------
# mutations: the comparators must notice a dropped accumulator
# ---------------------------------------------------------------------------


def _mutated_first_divergence(reference, dirty12, mutate):
    _, boundaries = reference
    es, dwin, _ = dirty12
    b = t21.mutated_boundary(boundaries[ANCHOR], mutate)
    return t21.mutation_first_divergence(t21.cap_set(), es, b, dwin.outputs)


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_mutation_timeadd_drop_detected(dirty12, reference) -> None:
    """timeadd threads the TsWaveDelay 59/1440 branch — dropping it must
    diverge (first via the Tgmap1 channel the branch rewrites)."""
    div = _mutated_first_divergence(reference, dirty12, "M3_timeadd_drop")
    assert div is not None, "timeadd drop went undetected"


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_mutation_firstdaytime_drop_detected(dirty12, reference) -> None:
    """firstdaytime gates the TsWaveDelay first-day reset — forcing it
    back to 1 must diverge at the first re-folded day step."""
    div = _mutated_first_divergence(reference, dirty12,
                                    "M4_firstdaytime_drop")
    assert div is not None, "firstdaytime drop went undetected"


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_mutation_plane_nan_detected(dirty12, reference) -> None:
    """An unabsorbable plane corruption (NaN in one cell of a restored
    Tgmap1) must reach the same step's outputs and be detected."""
    div = _mutated_first_divergence(reference, dirty12, "M5_plane_nan")
    assert div is not None, "NaN plane corruption went undetected"


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_mutation_1ulp_behaviour_pinned(dirty12, reference) -> None:
    """M5b: a 1-ULP poke on a normal-valued restored cell — propagation
    through the f32 blend chain is an ARITHMETIC property (rounding may
    absorb it); whatever this capture does is pinned so a semantic
    change in the chain surfaces here."""
    div = _mutated_first_divergence(reference, dirty12,
                                    "M5b_plane_1ulp_normal")
    pin = Path(t21.ART / "mutation_pin.json")
    if pin.exists():
        want = json.loads(pin.read_text())
        assert (div is None) == want["M5b_absorbed"], \
            f"M5b absorption changed: {div}"
    else:
        pytest.skip(f"mutation pin not yet written ({pin})")


@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_mutation_ci_thread_behaviour_pinned(dirty12, reference) -> None:
    """M1 (collapse the f64 CI thread through the f32 state.CI — the
    production warm path's spelling) and M2 (flip the tensor/pyfloat
    flavor) — the measured behaviour on this capture is pinned, whichever
    way it falls. (Filled from the parity artifacts; a change in capture
    or recompute chain that alters this must surface here.)"""
    m1 = _mutated_first_divergence(reference, dirty12,
                                   "M1_ci_thread_f32_collapse")
    m2 = _mutated_first_divergence(reference, dirty12, "M2_flavor_flip")
    pin = Path(t21.ART / "mutation_pin.json")
    if pin.exists():
        want = json.loads(pin.read_text())
        assert _div_key(m1) == want["M1_first_divergence"], \
            f"M1 behaviour changed: {m1} != {want['M1_first_divergence']}"
        assert _div_key(m2) == want["M2_first_divergence"], \
            f"M2 behaviour changed: {m2} != {want['M2_first_divergence']}"
    else:
        pytest.skip(f"mutation pin not yet written ({pin}); parity run "
                    "records it")


def _div_key(div):
    return None if div is None else [div[0], div[1]]


# ---------------------------------------------------------------------------
# fence + ledger (no folds)
# ---------------------------------------------------------------------------


def test_anchor_selection_never_covers_the_edit_row(series) -> None:
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    from tests.ultrafast.t21_ablation_run import run_window_with_boundaries
    # boundaries are only needed for steps <= 12, but the harness pass is
    # the cheapest correct source; use a tiny synthetic map instead —
    # select_anchor is pure arithmetic over digests.
    es = _edited_series(series, EDIT_FROM)
    ref_digest = {a: t21.met_prefix_digest(series, a) for a in (4, 8, 12, 16)}
    bnds = {}
    for a, d in ref_digest.items():
        bnds[a] = t21.Boundary(
            next_step=a, firstdaytime=0, timeadd=0.0, CI=np.float32(1.0),
            ci_thread=1.0, ci_flavor=False, met_prefix=d,
            planes={n: np.zeros((1, 1), np.float32)
                    for n in t21.LOOP_PLANES})
    picked = t21.select_anchor(bnds, es, EDIT_FROM)
    assert picked is not None and picked.next_step == 12, \
        "a boundary whose prefix covers the edit row must never be picked"
    assert t21.select_anchor(bnds, es, 3) is None, \
        "no applicable boundary -> full recompute is the fallback"


def test_classify_anchor_future_prefix_refusal(series) -> None:
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    es = _edited_series(series, EDIT_FROM)
    cand = thermal.CheckpointCandidate(
        scene_revision=1, next_step=16, thermal=True,
        input_fingerprint={"met_prefix": t21.met_prefix_digest(series, 16)})
    reason, invalidation = thermal.classify_anchor(
        cand, geometry_fingerprint={"scene": "x"},
        met_prefix_digest=lambda k: t21.met_prefix_digest(es, k), r0=12)
    assert invalidation == "future-prefix" and reason is not None


def test_checkpoint_payload_ledger_site500() -> None:
    """The memory accounting the T21 report cites (payload math)."""
    assert t21.PLANE_BYTES == 500 * 500 * 4
    assert t21.BOUNDARY_PAYLOAD_BYTES == 6_000_069
    per_t = t21.BOUNDARY_PAYLOAD_BYTES / 2**20
    assert round(per_t, 3) == 5.722
    k24 = 24 * t21.BOUNDARY_PAYLOAD_BYTES / 2**20
    assert round(k24, 1) == 137.3
    # campaign steady RSS pre-T21 (T16 measurement) + k=24 is far over
    assert 286.9 + k24 > 256.0
    # even the minimal useful store breaches at the current baseline
    assert 286.9 + 4 * per_t > 256.0
