# SPDX-License-Identifier: GPL-3.0-only
"""T10 exact sparse temporal replay tests (RED first).

Raw-bit equality of warm / cold / fresh-full temporal replay against the
T08 oracle capture, plus the policy fences the task card binds:

  R1  a PREVIOUS scene's noon checkpoint after a geometry edit is
      refused (geometry-history contamination) — geometry-history
      invalidation is SEPARATE from met-prefix invalidation;
  R2  a state restored without CI/Twater is refused or carried as
      explicitly-not-established, never silently defaulted;
  R3  next_step off-by-one (duplicate or skipped timestep) breaks the
      oracle-anchored warm equality and the bundle-request ledger;
  R4  a late met edit replays ONLY the invalidated met prefix (resume
      at the anchor's own next_step; digest spelled at the CANDIDATE's
      next_step, never the request's r0);
  R5  timestep outputs scatter to GLOBAL t (selected/sparse/full time
      coverage correctness);
  R6  a same-revision torn/mixed state is refused ATOMICALLY (per-plane
      digest fence; a live destination state stays bitwise untouched).

torch is allowed in THIS file (oracle-side comparisons); the runtime
module under test (solweig_core.numba_cpu.thermal) must stay torch-free
(pinned below, T08 precedent). Site/capture-dependent tests skip when
the t08 capture bundle is absent.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.ultrafast.bitwise_harness import assert_planes_equal  # noqa: E402

# RED witness: the T10 runtime module does not exist at task start
# (failing-first run recorded under artifacts/t10/red_first_run.txt).
from solweig_core.numba_cpu import radiation, thermal  # noqa: E402
from solweig_core.numba_cpu.thermal import (  # noqa: E402
    CARRIED_SCALAR_NAMES,
    GEOMETRY_FINGERPRINT_KEYS,
    THERMAL_PLANE_NAMES,
    ThermalStateError,
)

ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t10")
CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")

#: RED-witness anchors (mutations/mutations table); each must stay present
#: EXACTLY once so the mutation cycles stay reproducible.
MUTATION_ANCHORS = {
    "M1_geometry_gate": "if fp.get(key) != geometry_fingerprint.get(key):",
    "M2_met_digest_own_next_step": "wanted_met = met_prefix_digest(next_step)",
    "M3_replay_start": "start = int(state.next_step)",
    "M4_twater_not_defaulted": "twater = payload.get(\"Twater\")",
    "M5_t_scatter": "outputs[int(t)] = out",
    "M6_torn_fence": "if actual != expected_plane_digests[name]:",
}

PLANE_RETS = (
    "Tmrt", "Kdown", "Kup", "Ldown", "Lup", "Keast", "Ksouth", "Kwest",
    "Knorth", "Least", "Lsouth", "Lwest", "Lnorth", "KsideI", "TgOut",
    "Lside", "KsideD", "dRad", "Kside",
)


def capture_present() -> bool:
    return (CAP / "manifest.json").is_file()


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()


# ---------------------------------------------------------------------------
# Purity + structure pins
# ---------------------------------------------------------------------------

def test_thermal_module_imports() -> None:
    """RED-first: the T10 runtime module exists and is importable."""
    assert thermal is not None


def test_thermal_purity_torch_free() -> None:
    """The runtime module must not import torch (AST + subprocess pin)."""
    import ast

    tree = ast.parse(Path(thermal.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "torch" not in name, f"torch import: {name}"
    code = (
        "import sys; sys.path.insert(0, {root!r}); "
        "from solweig_core.numba_cpu import thermal; "
        "assert 'torch' not in sys.modules, 'torch leaked into runtime'"
    ).format(root=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_mutation_anchors_unique() -> None:
    """Each mutation anchor appears exactly once in the runtime module."""
    src = Path(thermal.__file__).read_text()
    for name, anchor in MUTATION_ANCHORS.items():
        assert src.count(anchor) == 1, (
            f"{name}: anchor count {src.count(anchor)}"
        )


def test_state_vocabulary_matches_checkpoint_contract() -> None:
    """Six planes / four scalars, spelled exactly like checkpoints.py."""
    assert THERMAL_PLANE_NAMES == (
        "Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N", "TgOut1",
    )
    assert CARRIED_SCALAR_NAMES == ("CI", "firstdaytime", "timeadd", "Twater")
    assert GEOMETRY_FINGERPRINT_KEYS == (
        "composed_scene", "resolved_landcover", "model_parameters",
    )


# ---------------------------------------------------------------------------
# Synthetic state + payload fixtures (fast; no capture needed)
# ---------------------------------------------------------------------------

def _planes(rows=6, cols=5, seed=7):
    rng = np.random.default_rng(seed)
    return {
        name: rng.standard_normal((rows, cols)).astype(np.float32)
        for name in THERMAL_PLANE_NAMES
    }


def _payload(next_step=12, twater=9.5, rows=6, cols=5, seed=7):
    p = _planes(rows, cols, seed)
    p.update(
        {
            "CI": 0.834,
            "firstdaytime": 0,
            "timeadd": 0.321,
            "next_step": (
                next_step if next_step is None else int(next_step)
            ),
        }
    )
    if twater is not _OMIT:
        p["Twater"] = twater
    return p


_OMIT = object()


def _loop_state():
    planes = _planes()
    return radiation.RadLoopState(
        firstdaytime=1,
        timeadd=0.0,
        CI=np.float32(1.0),
        **{k: planes[k] for k in THERMAL_PLANE_NAMES},
    )


# ---------------------------------------------------------------------------
# R2 — missing Twater/CI: refused or explicitly not-established, never a
# silent default
# ---------------------------------------------------------------------------

class TestR2CarriedScalars:
    def test_missing_ci_refused(self):
        payload = _payload()
        del payload["CI"]
        with pytest.raises(ThermalStateError, match="CI"):
            thermal.thermal_state_from_payload(payload)

    def test_missing_firstdaytime_and_timeadd_refused(self):
        for name in ("firstdaytime", "timeadd"):
            payload = _payload()
            del payload[name]
            with pytest.raises(ThermalStateError, match=name):
                thermal.thermal_state_from_payload(payload)

    def test_missing_plane_refused(self):
        payload = _payload()
        del payload["Tgmap1W"]
        with pytest.raises(ThermalStateError, match="Tgmap1W"):
            thermal.thermal_state_from_payload(payload)

    def test_twater_absent_carried_as_not_established_never_defaulted(self):
        # Absent Twater IS legitimate (the pre-midnight []: checkpoints
        # drop it when None) — but it must surface as None, never as a
        # guessed water temperature.
        state = thermal.thermal_state_from_payload(_payload(twater=_OMIT))
        assert state.Twater is None

    def test_twater_none_carried_as_none(self):
        state = thermal.thermal_state_from_payload(_payload(twater=None))
        assert state.Twater is None

    def test_twater_value_roundtrips(self):
        state = thermal.thermal_state_from_payload(_payload(twater=9.5))
        assert state.Twater == 9.5

    def test_coverage_only_payload_refused(self):
        # next_step=None = a coverage-only checkpoint: no warm state, and
        # restoring one must be a typed refusal, not a cold-looking state.
        payload = _payload(next_step=None)
        with pytest.raises(ThermalStateError, match="next_step"):
            thermal.thermal_state_from_payload(payload)

    def test_float64_plane_refused(self):
        payload = _payload()
        payload["TgOut1"] = payload["TgOut1"].astype(np.float64)
        with pytest.raises(ThermalStateError, match="float32"):
            thermal.thermal_state_from_payload(payload)


# ---------------------------------------------------------------------------
# R6 — same-revision torn state refused atomically
# ---------------------------------------------------------------------------

class TestR6TornState:
    def test_mixed_same_shape_planes_refused_by_digest_fence(self):
        # Two captures at the SAME revision (state@12 and state@13); a
        # torn load splices TgOut1 of the second into the first. Shape
        # coherence alone cannot see it — the per-plane digest fence must.
        good = _payload(next_step=12, seed=7)
        other = _payload(next_step=13, seed=8)
        torn = dict(good)
        torn["TgOut1"] = other["TgOut1"]
        digests = thermal.payload_plane_digests(good)
        with pytest.raises(ThermalStateError, match="TgOut1"):
            thermal.thermal_state_from_payload(
                torn, expected_plane_digests=digests
            )

    def test_digest_fence_rejects_bit_flips(self):
        payload = _payload()
        digests = thermal.payload_plane_digests(payload)
        payload["Tgmap1"] = payload["Tgmap1"].copy()
        payload["Tgmap1"][0, 0] = np.float32(
            payload["Tgmap1"][0, 0] + np.float32(1e-3)
        )
        with pytest.raises(ThermalStateError, match="Tgmap1"):
            thermal.thermal_state_from_payload(
                payload, expected_plane_digests=digests
            )

    def test_shape_incoherence_refused_even_without_digests(self):
        payload = _payload()
        payload["Tgmap1N"] = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ThermalStateError, match="shape"):
            thermal.thermal_state_from_payload(payload)

    def test_refusal_leaves_live_state_untouched(self):
        # Atomicity: a refused restore must not mutate a live loop state
        # the caller holds. Restore builds NEW objects only.
        live = _loop_state()
        live_before = {
            name: getattr(live, name).copy() for name in THERMAL_PLANE_NAMES
        }
        payload = _payload(rows=4, cols=4)  # wrong grid shape
        with pytest.raises(ThermalStateError):
            state = thermal.thermal_state_from_payload(
                payload, grid_shape=(6, 5)
            )
            thermal.restore_loop_state(state, grid_shape=(6, 5))
        for name in THERMAL_PLANE_NAMES:
            assert getattr(live, name).tobytes() == live_before[name].tobytes()


# ---------------------------------------------------------------------------
# R1 / R4 — applicability policy: geometry-history vs met-prefix
# ---------------------------------------------------------------------------

def _candidate(revision, next_step, *, geom="gA", met="m", thermal_state=True):
    return thermal.CheckpointCandidate(
        scene_revision=revision,
        next_step=next_step,
        thermal=thermal_state,
        input_fingerprint={
            "composed_scene": geom,
            "resolved_landcover": "lc",
            "model_parameters": "mp",
            "met_prefix": met,
        },
    )


class TestR1GeometryHistorySeparation:
    def test_previous_scene_noon_checkpoint_refused_after_geometry_edit(self):
        # R1: a noon (next_step=12) thermal checkpoint of the PREVIOUS
        # scene, met prefix unchanged, next_step <= r0 — a met-only check
        # would ACCEPT it. The geometry-history gate must refuse.
        cand = _candidate(revision=4, next_step=12, geom="gOLD")
        decision = thermal.select_warm_anchor(
            [cand],
            geometry_fingerprint={
                "composed_scene": "gNEW",
                "resolved_landcover": "lc",
                "model_parameters": "mp",
            },
            met_prefix_digest=lambda next_step: "m",
            r0=20,
        )
        assert decision.selected is None
        assert decision.resume_step == 0
        assert decision.invalidation == "geometry-history"
        assert "geometry-history" in decision.reason
        assert "composed_scene" in decision.reason

    def test_geometry_reason_dominates_when_met_also_differs(self):
        # The two invalidation families must stay distinguishable: a
        # checkpoint stale on BOTH axes reports GEOMETRY (the spatial
        # lineage is the unresolvable one), never a bare met reason.
        cand = _candidate(revision=4, next_step=12, geom="gOLD", met="mOLD")
        decision = thermal.select_warm_anchor(
            [cand],
            geometry_fingerprint={
                "composed_scene": "gNEW",
                "resolved_landcover": "lc",
                "model_parameters": "mp",
            },
            met_prefix_digest=lambda next_step: "mNEW",
            r0=20,
        )
        assert decision.invalidation == "geometry-history"

    def test_matching_geometry_and_met_selects_the_anchor(self):
        cand = _candidate(revision=4, next_step=12)
        decision = thermal.select_warm_anchor(
            [cand],
            geometry_fingerprint={
                "composed_scene": "gA",
                "resolved_landcover": "lc",
                "model_parameters": "mp",
            },
            met_prefix_digest=lambda next_step: "m",
            r0=20,
        )
        assert decision.selected is cand
        assert decision.resume_step == 12
        assert decision.invalidation == "warm"
        assert decision.reason is None

    def test_newest_matching_geometry_beats_older(self):
        # An older checkpoint under the CURRENT geometry wins over a
        # newer one under a stale scene (fingerprint decides, not age).
        stale_new = _candidate(revision=9, next_step=20, geom="gOLD")
        fresh_old = _candidate(revision=5, next_step=12, geom="gA")
        decision = thermal.select_warm_anchor(
            [stale_new, fresh_old],
            geometry_fingerprint={
                "composed_scene": "gA",
                "resolved_landcover": "lc",
                "model_parameters": "mp",
            },
            met_prefix_digest=lambda next_step: "m",
            r0=21,
        )
        assert decision.selected is fresh_old
        assert decision.invalidation == "warm"

    def test_newest_applicable_selected_among_matching(self):
        # Lead pin (T10 verification): with MULTIPLE applicable
        # candidates the NEWEST is the anchor — the executor scans
        # newest-first and breaks, so the thermal-side policy must agree
        # (an oldest-first scan would resume from a colder next_step and
        # inflate steps_solved for every met edit). Order of the input
        # list must not matter.
        older = _candidate(revision=5, next_step=12)
        newer = _candidate(revision=9, next_step=20)
        geometry = {
            "composed_scene": "gA",
            "resolved_landcover": "lc",
            "model_parameters": "mp",
        }
        for candidates in ([older, newer], [newer, older]):
            decision = thermal.select_warm_anchor(
                candidates,
                geometry_fingerprint=geometry,
                met_prefix_digest=lambda next_step: "m",
                r0=21,
            )
            assert decision.selected is newer, candidates
            assert decision.resume_step == 20
            assert decision.invalidation == "warm"

    def test_coverage_only_never_selects(self):
        cand = _candidate(revision=4, next_step=None, thermal_state=False)
        decision = thermal.select_warm_anchor(
            [cand],
            geometry_fingerprint={
                "composed_scene": "gA",
                "resolved_landcover": "lc",
                "model_parameters": "mp",
            },
            met_prefix_digest=lambda next_step: "m",
            r0=20,
        )
        assert decision.selected is None
        assert decision.resume_step == 0
        # Coverage-only markers are NOT thermal hits (DESIGN 11.1).
        assert "coverage-only" in decision.reason


class TestR4MetPrefixInvalidation:
    def test_future_prefix_checkpoint_refused(self):
        # anchor@21 for an edit at r0=20: its met prefix heard row 20.
        cand = _candidate(revision=4, next_step=21)
        decision = thermal.select_warm_anchor(
            [cand],
            geometry_fingerprint={
                "composed_scene": "gA",
                "resolved_landcover": "lc",
                "model_parameters": "mp",
            },
            met_prefix_digest=lambda next_step: "m",
            r0=20,
        )
        assert decision.selected is None
        assert decision.invalidation == "future-prefix"
        assert "beyond the first changed row" in decision.reason

    def test_digest_spelled_at_candidate_own_next_step(self):
        # R4/card: the met digest is computed over rows [0:next_step) of
        # the CANDIDATE — rows in [next_step, r0) are irrelevant. Spelling
        # it at r0 instead produces BOTH failure modes; this test pins
        # hit-when-rows-beyond-the-anchor-differ and
        # miss-when-rows-under-the-anchor-differ.
        def digest_for(rows):
            def _d(next_step):
                # distinct digest per prefix content
                return "".join(rows[:next_step])
            return _d

        current_rows = list("mmmmmmmmmmmm") + list("ZZZZZZZZ")  # 12 + 8
        # checkpoint@12 recorded digest over rows[0:12] == "mmmm..."
        cand = _candidate(revision=4, next_step=12, met="m" * 12)
        geom = {
            "composed_scene": "gA",
            "resolved_landcover": "lc",
            "model_parameters": "mp",
        }
        decision = thermal.select_warm_anchor(
            [cand],
            geometry_fingerprint=geom,
            met_prefix_digest=digest_for(current_rows),
            r0=20,
        )
        # rows [12:20) differ from the capture — must STILL hit (the
        # state never read them).
        assert decision.selected is cand
        # ...unless the first next_step rows themselves differ.
        edited_early = list("mmmmmmmmmmXm") + list("ZZZZZZZZ")
        decision2 = thermal.select_warm_anchor(
            [_candidate(revision=4, next_step=12, met="m" * 12)],
            geometry_fingerprint=geom,
            met_prefix_digest=digest_for(edited_early),
            r0=20,
        )
        assert decision2.selected is None
        assert decision2.invalidation == "met-prefix"
        assert "met_prefix" in decision2.reason

    def test_resume_step_is_the_anchor_next_step(self):
        cand = _candidate(revision=7, next_step=20, met="m" * 20)
        geom = {
            "composed_scene": "gA",
            "resolved_landcover": "lc",
            "model_parameters": "mp",
        }
        decision = thermal.select_warm_anchor(
            [cand],
            geometry_fingerprint=geom,
            met_prefix_digest=lambda next_step: "m" * next_step,
            r0=21,
        )
        assert decision.resume_step == 20


# ---------------------------------------------------------------------------
# Capture-driven fixtures (real site_500 radiation states)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rad_static():
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    return radiation.rad_static_from_capture(CAP)


@pytest.fixture(scope="module")
def oracle():
    """The capture's own per-t return planes + entry states (ground truth)."""
    if not capture_present():
        pytest.skip("t08 capture bundle absent")
    manifest = json.loads((CAP / "manifest.json").read_text())
    n_t = manifest["n_timesteps"]
    rets = {}
    for t in range(n_t):
        z = np.load(CAP / f"t{t:02d}.npz")
        rets[t] = {name: z[f"ret_{name}"] for name in
                   PLANE_RETS + ("Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W",
                                 "Tgmap1N", "TgOut1")}
    return {"n_t": n_t, "rets": rets}


def oracle_anchor_state(t_entry: int) -> "thermal.ThermalState":
    """A checkpoint whose planes/scalars are the ORACLE's recorded entry
    state of timestep ``t_entry`` (next_step == t_entry)."""
    z = np.load(CAP / f"t{t_entry:02d}.npz")
    payload = {name: z[f"arg_{name}"] for name in THERMAL_PLANE_NAMES}
    if "arg_CI__0d" in z:
        ci = np.float32(z["arg_CI__0d"][()])
    else:
        ci = np.float32(z["arg_CI__f64"][()])
    payload["CI"] = float(ci)
    payload["firstdaytime"] = int(z["arg_firstdaytime__int"][()])
    if "arg_timeadd__f64" in z:
        payload["timeadd"] = float(z["arg_timeadd__f64"][()])
    else:
        payload["timeadd"] = float(z["arg_timeadd__int"][()])
    payload["Twater"] = None  # not-established spelling (pre-midnight)
    payload["next_step"] = int(t_entry)
    return thermal.thermal_state_from_payload(payload)


def plain_bundle(t: int) -> radiation.RadTimestepInputs:
    return radiation.rad_bundle_from_capture(CAP, t)


def edited_bundle(t: int, r0: int, delta: float = 2.0):
    """A met-EDITED bundle for t >= r0 (air-temperature/radiation forcing
    perturbed in the kernel's direct f32 consumption lanes).

    Honest scope note: the frozen transcendental bits derived from met
    (Lup_pre/Lwall/Tg/ta273_pow4, ...) stay at their captured values —
    torch-free recompute of EDITED frozen bits is impossible on this host
    (T01 matrix) and belongs to the T11 torch fallback. The differential
    this feeds proves warm==fresh state threading under genuinely changed
    per-timestep inputs, not edited-met physics.
    """
    b = radiation.rad_bundle_from_capture(CAP, t)
    if t >= r0:
        b.Ta = np.float32(b.Ta + np.float32(delta))
        b.Ta_f64 = float(b.Ta)
        b.radG = np.float32(b.radG + np.float32(30.0))
        if not b.is_day:
            pass  # night TgOut plane = Ta + 0.0 responds to Ta directly
    return b


# ---------------------------------------------------------------------------
# Equality gates: fresh-full / cold-split / warm vs the oracle capture
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
class TestWarmColdFreshEquality:
    """GATE: warm/cold/fresh-full raw float32-bit equality.

    No allclose/array_equal/1-ULP tolerance anywhere in this path: every
    comparison is uint32-view (NaN payloads and signed zeros included).
    """

    def test_fresh_full_matches_oracle_capture(self, rad_static, oracle):
        n_t = oracle["n_t"]
        out = thermal.replay_thermal(
            rad_static,
            plain_bundle,
            thermal.cold_thermal_state(
                rows=rad_static.rows, cols=rad_static.cols
            ),
            collect_ts=range(n_t),
            t_stop=n_t,
        )
        assert out.steps_solved == n_t
        for t in range(n_t):
            for name in PLANE_RETS:
                assert_planes_equal(oracle["rets"][t][name],
                                    out.outputs[t][name])

    def test_warm_from_oracle_anchor_matches_oracle(self, rad_static,
                                                    oracle):
        # WARM: restore the ORACLE's own entry state@12 (an independent
        # ground-truth checkpoint) and replay 12..24 — every output plane
        # must equal the capture's bits, proving the restored state is
        # bit-compatible with the fresh loop's carried state.
        n_t = oracle["n_t"]
        anchor = oracle_anchor_state(12)
        out = thermal.replay_thermal(
            rad_static, plain_bundle, anchor,
            collect_ts=range(12, n_t), t_stop=n_t,
        )
        assert out.steps_solved == n_t - 12  # R4: only the suffix solved
        for t in range(12, n_t):
            for name in PLANE_RETS:
                assert_planes_equal(oracle["rets"][t][name],
                                    out.outputs[t][name])
        # ...and the carried state matches the oracle's post-t23 state.
        for name in THERMAL_PLANE_NAMES:
            assert_planes_equal(oracle["rets"][n_t - 1][name],
                                getattr(out.final_state, name))
        assert out.final_state.next_step == n_t

    def test_cold_split_equals_fresh_full(self, rad_static, oracle):
        # COLD: no usable anchor (k=0) — the executor's split solve. The
        # discarded prefix must thread the state exactly: suffix outputs
        # bitwise == the fresh-full outputs.
        n_t = oracle["n_t"]
        r0 = 20
        fresh = thermal.replay_thermal(
            rad_static, plain_bundle,
            thermal.cold_thermal_state(rows=rad_static.rows,
                                       cols=rad_static.cols),
            collect_ts=range(n_t), t_stop=n_t,
        )
        split = thermal.split_replay_thermal(
            rad_static, plain_bundle,
            thermal.cold_thermal_state(rows=rad_static.rows,
                                       cols=rad_static.cols),
            r0=r0, t_stop=n_t,
        )
        assert split.steps_solved == n_t
        assert sorted(split.outputs) == list(range(r0, n_t))
        for t in range(r0, n_t):
            for name in PLANE_RETS:
                assert (bits(fresh.outputs[t][name])
                        == bits(split.outputs[t][name])).all()
        for name in THERMAL_PLANE_NAMES:
            assert (bits(getattr(fresh.final_state, name))
                    == bits(getattr(split.final_state, name))).all()

    def test_warm_equals_fresh_full_under_edited_suffix(self, rad_static,
                                                        oracle):
        # REAL EDIT SEQUENCE (internal differential): a met edit at
        # r0=20 changes the per-timestep inputs from r0 on. Warm (from a
        # checkpoint@12 captured on the UNEDITED run) and fresh-full both
        # consume the same edited bundles; causality + exact state
        # threading must give bitwise-identical suffix outputs and final
        # state. This kills restore/duplicate/skip defects that an
        # unedited sequence cannot expose.
        n_t = oracle["n_t"]
        r0 = 20
        anchor = oracle_anchor_state(12)
        edited = lambda t: edited_bundle(t, r0)
        fresh = thermal.replay_thermal(
            rad_static, edited,
            thermal.cold_thermal_state(rows=rad_static.rows,
                                       cols=rad_static.cols),
            collect_ts=range(n_t), t_stop=n_t,
        )
        warm = thermal.replay_thermal(
            rad_static, edited, anchor,
            collect_ts=range(anchor.next_step, n_t), t_stop=n_t,
        )
        # the edit actually changed the physics (not a vacuous test)
        z = np.load(CAP / "t23.npz")
        assert not (bits(fresh.outputs[23]["Tmrt"]
                         == bits(z["ret_Tmrt"])).all())
        for t in range(anchor.next_step, n_t):
            for name in PLANE_RETS:
                assert (bits(fresh.outputs[t][name])
                        == bits(warm.outputs[t][name])).all()
        for name in THERMAL_PLANE_NAMES:
            assert (bits(getattr(fresh.final_state, name))
                    == bits(getattr(warm.final_state, name))).all()
        # R4 economics: the warm route solved 12 steps, not 24.
        assert warm.steps_solved == 12 and fresh.steps_solved == 24


# ---------------------------------------------------------------------------
# R3 / R5 — resume ledger + global t scatter + coverage
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
class TestResumeAndScatter:
    def _counting_bundles(self):
        ledger = []

        def fn(t):
            ledger.append(int(t))
            return plain_bundle(t)

        return fn, ledger

    def test_replay_resumes_exactly_at_next_step(self, rad_static):
        # R3: restoring state@k must replay FROM k — the first bundle
        # requested is t=k (never k-1: duplicate; never k+1: skip).
        fn, ledger = self._counting_bundles()
        anchor = oracle_anchor_state(12)
        out = thermal.replay_thermal(
            rad_static, fn, anchor, collect_ts=(), t_stop=24
        )
        assert ledger == list(range(12, 24))
        assert out.steps_solved == 12
        assert out.final_state.next_step == 24

    def test_replay_before_anchor_refused(self, rad_static):
        # A collect t < next_step asks for output the anchor already
        # consumed — a silent recompute would be an off-by-one science
        # bug; it is a typed refusal instead.
        anchor = oracle_anchor_state(12)
        with pytest.raises(ThermalStateError, match="collect"):
            thermal.replay_thermal(
                rad_static, plain_bundle, anchor,
                collect_ts=(11, 12), t_stop=24,
            )

    def test_selected_time_outputs_scatter_to_global_t(self, rad_static,
                                                       oracle):
        # R5 + DESIGN 11.4: a selected-time request [12] advances the
        # causal prefix but EMITS only t=12 — at its GLOBAL index, never
        # parked at positional 0.
        out = thermal.replay_thermal(
            rad_static, plain_bundle,
            thermal.cold_thermal_state(rows=rad_static.rows,
                                       cols=rad_static.cols),
            collect_ts=(12,), t_stop=None,
        )
        assert sorted(out.outputs) == [12]
        for name in PLANE_RETS:
            assert_planes_equal(oracle["rets"][12][name],
                                out.outputs[12][name])
        assert out.steps_solved == 13  # causal prefix 0..12 inclusive

    def test_multi_band_selected_times(self, rad_static, oracle):
        out = thermal.replay_thermal(
            rad_static, plain_bundle,
            thermal.cold_thermal_state(rows=rad_static.rows,
                                       cols=rad_static.cols),
            collect_ts=(5, 17), t_stop=None,
        )
        assert sorted(out.outputs) == [5, 17]
        for t in (5, 17):
            for name in PLANE_RETS:
                assert_planes_equal(oracle["rets"][t][name],
                                    out.outputs[t][name])

    def test_sparse_suffix_band_exactly_r0_to_t(self, rad_static, oracle):
        n_t = oracle["n_t"]
        out = thermal.split_replay_thermal(
            rad_static, plain_bundle,
            thermal.cold_thermal_state(rows=rad_static.rows,
                                       cols=rad_static.cols),
            r0=20, t_stop=n_t,
        )
        assert sorted(out.outputs) == list(range(20, n_t))
        for t in range(20, n_t):
            for name in PLANE_RETS:
                assert_planes_equal(oracle["rets"][t][name],
                                    out.outputs[t][name])

    def test_collect_beyond_stop_refused(self, rad_static):
        with pytest.raises(ThermalStateError, match="t_stop"):
            thermal.replay_thermal(
                rad_static, plain_bundle,
                thermal.cold_thermal_state(rows=rad_static.rows,
                                           cols=rad_static.cols),
                collect_ts=(23,), t_stop=12,
            )


# ---------------------------------------------------------------------------
# steps_solved reduction (exit criterion) — artifact recorded
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture bundle absent")
def test_steps_solved_reduction_recorded(rad_static, oracle):
    """Exit criterion: warm/cold/fresh steps accounting. A coverage-only
    checkpoint (resume 0) is NOT a thermal hit; the anchor@20's warm
    suffix solves 4 of 24."""
    n_t = oracle["n_t"]
    cold_state = thermal.cold_thermal_state(rows=rad_static.rows,
                                            cols=rad_static.cols)
    fresh = thermal.replay_thermal(
        rad_static, plain_bundle, cold_state,
        collect_ts=range(n_t), t_stop=n_t,
    )
    cold_split = thermal.split_replay_thermal(
        rad_static, plain_bundle, cold_state, r0=20, t_stop=n_t,
    )
    anchor20 = oracle_anchor_state(20)
    warm = thermal.split_replay_thermal(
        rad_static, plain_bundle, anchor20, r0=21, t_stop=n_t,
    )
    assert (fresh.steps_solved, cold_split.steps_solved,
            warm.steps_solved) == (24, 24, 4)
    record = {
        "fresh_full_steps": fresh.steps_solved,
        "cold_split_steps": cold_split.steps_solved,
        "warm_hit_steps": warm.steps_solved,
        "warm_resume_step": anchor20.next_step,
        "coverage_only_checkpoint_steps": fresh.steps_solved,
    }
    if ART.is_dir():
        path = ART / "steps_solved_reduction.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
