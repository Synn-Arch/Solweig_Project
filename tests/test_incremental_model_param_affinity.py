# SPDX-License-Identifier: GPL-3.0-only
"""R3b: model-parameter affinity audit — the durable artifact of the
cheap-exact reachability analysis (r3b Step 0, 2026-09-04).

R3b's mission was the NEXT cheap exact fast path after r3a's utci-only met
path. The reachability audit across the remaining families and planes:

* **VIEW/RECEPTOR edits** — no lever. No receptor-point concept exists
  anywhere (no receptor node in edit_graph.GRAPH_NODES, no receptor family
  in edit_registry.BUILTIN_ADAPTER_METADATA, the store keys are
  (node, time) only); "receptor parameters" are site-global scalars. R0's
  ranking (b) — "receptor-only params subset (Fside/Fup/Fcyl/cyl/height)"
  — is REFUTED by the physics: every one of them enters Sstr, the human-body
  radiant load (solweig.py:2287-2296, Fside/Fup/Fcyl/cyl via branch
  selection; absK/absL as its coefficients), and the published tmrt raster
  derives from it (Tmrt = (Sstr/(absL*SBC))^0.25 - 273.2, solweig.py:2299).
  ``height`` also moves the wall-height shadow walk reach
  (utci_process.py:729-735), ``elvis`` the sky emissivity (solweig.py:2046),
  ``anisotropic_sky`` the diffuse loops (solweig.py:684-688),
  ``transVeg``/``firstdayleaf``/``lastdayleaf`` the vegetation
  transmissivity/leaf cycle (utci_process.py:662/:713-724/:726/:743/:747),
  ``albedo_b`` the wall-reflection weighting (solweig.py:213-215),
  ``ewall`` the wall long-wave term (solweig.py:132). A parameter batch can
  therefore never narrow below the radiation closure: tmrt changes, so no
  comfort-only recompute (the r3a shape) is admissible.
* **CACHE/VIEW plane reuse for shadow-free timesteps** — no lever. Every
  edit family that dirties time_shadow also dirties svf
  (edit_graph: building_visibility→{svf,time_shadow},
  vegetation_visibility→{svf,time_shadow}), and svf is time-invariant, so a
  geometry/paint edit moves radiation products at NIGHT timesteps too
  (Ldown through sky view, Lup/Tg through surface state); there is no
  provably-untouched timestep set to serve from cache. For meteorology the
  radiation_affecting variables reach night steps through esky/Ldown
  (solweig.py:2042-2059) and the nocturnal ground-heat lowering
  (:2131-2133).
* **Forcing beyond r3a's utci_only set** — no lever.
  MET_VARIABLE_AFFINITY already classifies the ENTIRE safe vocabulary
  (edit_registry.METEOROLOGY_VARIABLES_SAFE); the radiation_affecting
  members carry solweig.py consumption evidence, and whole-series /
  update_range met edits already expand to per-timestep ForcingChange
  records (adapters/met_time.py _expand_changes), so they already ride the
  r3a fast path when utci_only.
* **One real lever found, OUT OF R3b's file grant** — met fast-path
  DURABILITY. The store's R2 supersession drops a recorded entry WHOLE when
  a newer windowed batch intersects it (store.py publish commit step, the
  R2 retention comment), and a full-tile entry intersects every window, so
  after the FIRST windowed vegetation/landcover batch the executor's tmrt@t
  coverage is PARTIAL and every later utci_only met edit refuses to the
  full solve (r3a's freshness gate; pinned by
  test_incremental_met_fast_path.test_windowed_tmrt_publication_breaks_freshness).
  Repairing that needs per-cell supersession/retention semantics in
  store.py — another packet's file — so R3b reports it instead of editing.

What IS shippable inside the grant is this audit's durable artifact,
mirroring the r3a idiom (evidence lives in code with a drift guard, so a
later packet cannot re-derive the refuted ranking):
:data:`solweig_gpu.incremental.edit_registry.MODEL_PARAMETERS_AFFINITY` —
every safe model parameter classified ``radiation_affecting`` with
file:line consumer evidence and an import-time drift guard.
"""

from __future__ import annotations

import pytest

from solweig_gpu.incremental.edit_registry import (
    MODEL_PARAMETERS_AFFINITY,
    MODEL_PARAMETERS_AFFINITY_EVIDENCE,
    MODEL_PARAMETERS_SAFE,
    AdapterRegistryError,
    _validate_model_parameter_affinity,
)
from solweig_gpu.incremental.edit_types import (
    EditCommand,
    ModelParameterChange,
    ModelParameterDelta,
    SpatialScope,
    ValidatedEdit,
)
from solweig_gpu.incremental.edit_graph import SceneGraphState, default_edit_graph
from solweig_gpu.incremental.planner import EditPlanner
from solweig_gpu.incremental.geometry import RasterGrid

PARAM_ADAPTER_ID = "model_receptor_parameters"
PARAM_SOURCE_NODE = "model_parameters"

GRID = RasterGrid(64, 64, 2.0)


# ---------------------------------------------------------------------------
# The affinity table is pinned (and proven from code)
# ---------------------------------------------------------------------------


class TestModelParameterAffinity:
    def test_every_safe_parameter_is_radiation_affecting(self) -> None:
        # R0's ranking (b) hoped a "receptor-only" subset (Fside/Fup/Fcyl/
        # cyl/height/...) could recompute comfort without radiation. The
        # code refutes it: Sstr consumes ALL of them (solweig.py:2287-2296)
        # and Tmrt derives from Sstr (:2299), so tmrt moves and the r3a
        # comfort-only shape is inadmissible for every parameter batch.
        assert set(MODEL_PARAMETERS_AFFINITY) == set(MODEL_PARAMETERS_SAFE)
        assert set(MODEL_PARAMETERS_AFFINITY.values()) == {"radiation_affecting"}

    def test_every_affinity_carries_fileline_evidence(self) -> None:
        assert set(MODEL_PARAMETERS_AFFINITY_EVIDENCE) == set(
            MODEL_PARAMETERS_AFFINITY
        )
        for parameter, evidence in MODEL_PARAMETERS_AFFINITY_EVIDENCE.items():
            assert evidence, parameter
            # Radiation affinity is proven by physics consumption sites.
            assert ("solweig.py:" in evidence) or (
                "utci_process.py:" in evidence
            ), parameter

    def test_key_drift_is_refused(self) -> None:
        # A parameter added to the safe set without a classified affinity
        # (or an invented entry) is a hard import-time error: an
        # unclassified parameter would silently inherit no bucket and a
        # future packet could mistake it for a comfort-only lever.
        evidence = dict(MODEL_PARAMETERS_AFFINITY_EVIDENCE)
        affinity = dict(MODEL_PARAMETERS_AFFINITY)
        del affinity["height"]
        with pytest.raises(AdapterRegistryError, match="drifted"):
            _validate_model_parameter_affinity(affinity, evidence)
        affinity = dict(MODEL_PARAMETERS_AFFINITY)
        affinity["phantom_parameter"] = "radiation_affecting"
        with pytest.raises(AdapterRegistryError, match="drifted"):
            _validate_model_parameter_affinity(affinity, evidence)

    def test_value_typo_is_refused(self) -> None:
        # L1-style (r3a review): a value typo ("radiation-only") would
        # import clean and classify the parameter into NO bucket.
        evidence = dict(MODEL_PARAMETERS_AFFINITY_EVIDENCE)
        for typo in ("radiation-only", "comfort", "", "RADIATION_AFFECTING"):
            bad = dict(MODEL_PARAMETERS_AFFINITY)
            bad["Fcyl"] = typo
            with pytest.raises(AdapterRegistryError, match="affinity"):
                _validate_model_parameter_affinity(bad, evidence)
        # The shipped table itself validates clean.
        _validate_model_parameter_affinity(
            MODEL_PARAMETERS_AFFINITY, MODEL_PARAMETERS_AFFINITY_EVIDENCE
        )

    def test_registry_adapters_cover_the_same_safe_set(self) -> None:
        # The coupling that makes the table load-bearing: the builtin
        # model_receptor_parameters adapter accepts exactly the safe set,
        # so every parameter a user CAN edit is one the table classifies.
        from solweig_gpu.incremental.edit_registry import builtin_registry

        registry = builtin_registry()
        metadata = registry.get_metadata(PARAM_ADAPTER_ID)
        assert metadata.parameters_safe == MODEL_PARAMETERS_SAFE


# ---------------------------------------------------------------------------
# Planner consequence: a parameter batch keeps the FULL radiation closure
# ---------------------------------------------------------------------------


def _parameter_edit(parameter: str, after: float, edit_id: str = "param-1"):
    """A validated-shape model-parameter edit (engine-level build, mirroring
    the met fast-path tests' delta construction)."""
    return ValidatedEdit(
        command=EditCommand(
            edit_id=edit_id,
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id=PARAM_ADAPTER_ID,
            operation="update",
            old_state={parameter: 0.28},
            new_state={parameter: after},
            requested_outputs=("utci",),
            requested_times=(1,),
        ),
        adapter_id=PARAM_ADAPTER_ID,
        schema_version=1,
        source_node_id=PARAM_SOURCE_NODE,
        delta=ModelParameterDelta(
            PARAM_SOURCE_NODE,
            PARAM_ADAPTER_ID,
            (ModelParameterChange(parameter, 0.28, after),),
        ),
    )


class TestModelParameterClosure:
    def test_parameter_batch_keeps_full_radiation_closure(self) -> None:
        planner = EditPlanner(grid=GRID)
        plan = planner.plan(
            [_parameter_edit("Fcyl", 0.4)], SceneGraphState.initial(default_edit_graph())
        )
        scopes = {impact.node_id: impact.spatial_scope for impact in plan.node_impacts}
        # The graph edge model_parameters -> radiation (edit_graph.py:144)
        # makes the closure radiation -> surface_thermal_state -> tmrt ->
        # utci; every hop plans FULL.
        for node_id in (
            "radiation",
            "surface_thermal_state",
            "tmrt",
            "utci",
        ):
            assert scopes[node_id] is SpatialScope.FULL, node_id
        assert "tmrt" not in plan.reusable_nodes
        assert "radiation" not in plan.reusable_nodes

    def test_person_geometry_parameters_are_no_exception(self) -> None:
        # The refuted R0 ranking, pinned per parameter: none of the
        # person-geometry parameters admits a comfort-only closure.
        for parameter in ("Fside", "Fup", "Fcyl", "cyl", "height"):
            planner = EditPlanner(grid=GRID)
            plan = planner.plan(
                [_parameter_edit(parameter, 0.5, edit_id=f"param-{parameter}")],
                SceneGraphState.initial(default_edit_graph()),
            )
            scopes = {
                impact.node_id: impact.spatial_scope for impact in plan.node_impacts
            }
            assert scopes["tmrt"] is SpatialScope.FULL, parameter
            assert scopes["utci"] is SpatialScope.FULL, parameter
