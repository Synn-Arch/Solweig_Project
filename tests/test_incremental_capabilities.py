# SPDX-License-Identifier: GPL-3.0-only
"""Capability document tests (U-D1, UEDIT-007 + frontend contract).

The document must be DERIVED, never duplicated: every assertion here
compares against the owning source (builtin registry metadata, the
adapter spec tables, the dependency graph, the executor's executable
family set), so a drift between the document and any source fails here
instead of misleading a frontend.
"""

from __future__ import annotations

import dataclasses

import pytest

from solweig_gpu.incremental.adapters.building import (
    HEIGHT_MAX_M,
    HEIGHT_MIN_M,
    MIN_FOOTPRINT_VERTICES,
)
from solweig_gpu.incremental.adapters.met_time import VARIABLE_SPECS
from solweig_gpu.incremental.adapters.model_parameters import PARAMETER_SPECS
from solweig_gpu.incremental.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    capability_document,
    full_registry,
    locality_modes_for_scope,
    producible_view_layers,
)
from solweig_gpu.incremental.edit_graph import default_edit_graph
from solweig_gpu.incremental.edit_registry import (
    LANDCOVER_FENCED_CLASSES,
    LANDCOVER_VALID_CLASSES,
    METEOROLOGY_VARIABLES_BLOCKED,
    METEOROLOGY_VARIABLES_SAFE,
    MODEL_PARAMETERS_BLOCKED,
    MODEL_PARAMETERS_SAFE,
    AdapterStatus,
    builtin_adapter_metadata,
    builtin_registry,
)
from solweig_gpu.incremental.executor import EXECUTABLE_SOURCE_NODES
from solweig_gpu.incremental.geometry import TreeSpec
from solweig_gpu.incremental.trees import (
    TREE_CANOPY_DIAMETER_RANGE_M,
    TREE_HEIGHT_RANGE_M,
    TRUNK_RATIO_RANGE,
)
from solweig_gpu.incremental.trees import validate_tree_spec


def test_document_mirrors_builtin_registry_exactly() -> None:
    """Every builtin adapter appears once, with metadata verbatim."""
    document = capability_document()
    expected = {metadata.id: metadata for metadata in builtin_adapter_metadata()}
    entries = {entry["id"]: entry for entry in document["adapters"]}
    assert set(entries) == set(expected)
    for adapter_id, metadata in expected.items():
        entry = entries[adapter_id]
        assert entry["group"] == metadata.group
        assert entry["status"] == metadata.status.value
        assert entry["source_nodes"] == list(metadata.source_nodes)
        assert entry["operations"] == list(metadata.operations)
        assert entry["nominal_spatial_scope"] == metadata.nominal_spatial_scope
        assert entry["temporal_scope"] == metadata.temporal_scope
        assert entry["preview"] == metadata.preview
        assert entry["validation_fixtures"] == list(metadata.validation_fixtures)


def test_integrated_flags_match_executable_source_nodes() -> None:
    """The ``integrated`` flags cannot drift from the executor's own set.

    The executor executes exactly the five single-source scientific
    families in ``EXECUTABLE_SOURCE_NODES``; an adapter is "integrated"
    exactly when ``register_default_adapters`` binds an implementation
    for it, which is the same fact. If one side widens without the other,
    this fails.
    """
    document = capability_document()
    integrated_sources = sorted(
        entry["source_nodes"][0]
        for entry in document["adapters"]
        if entry["integrated"] and len(entry["source_nodes"]) == 1
    )
    assert integrated_sources == sorted(EXECUTABLE_SOURCE_NODES)
    # And the flags come from a registry with implementations bound:
    # metadata-only registries answer False without touching the document.
    registry = builtin_registry()
    metadata_only = capability_document(registry)
    assert not any(
        entry["integrated"] for entry in metadata_only["adapters"]
    )
    assert full_registry().get_adapter("vegetation_geometry") is not None


def test_dirty_nodes_are_downstream_closures() -> None:
    graph = default_edit_graph()
    document = capability_document()
    entries = {entry["id"]: entry for entry in document["adapters"]}
    for adapter_id, metadata in {
        metadata.id: metadata for metadata in builtin_adapter_metadata()
    }.items():
        assert entries[adapter_id]["dirty_nodes"] == sorted(
            graph.downstream(metadata.source_nodes)
        )
    # The per-family table answers the same question keyed by source node.
    for node_id, family in document["edit_families"].items():
        assert family["dirty_nodes"] == sorted(graph.downstream((node_id,)))


def test_fences_section_mirrors_registry_rulings() -> None:
    fences = capability_document()["fences"]
    assert fences["landcover_valid_classes"] == sorted(LANDCOVER_VALID_CLASSES)
    assert fences["landcover_fenced_classes"] == sorted(LANDCOVER_FENCED_CLASSES)
    assert fences["model_parameters_safe"] == sorted(MODEL_PARAMETERS_SAFE)
    assert fences["model_parameters_blocked"] == sorted(MODEL_PARAMETERS_BLOCKED)
    assert fences["meteorology_variables_safe"] == sorted(METEOROLOGY_VARIABLES_SAFE)
    assert fences["meteorology_variables_blocked"] == sorted(
        METEOROLOGY_VARIABLES_BLOCKED
    )


def test_property_schemas_come_from_spec_tables() -> None:
    document = capability_document()
    schemas = {entry["id"]: entry["property_schema"] for entry in document["adapters"]}

    vegetation = schemas["vegetation_geometry"]
    height_low, height_high = TREE_HEIGHT_RANGE_M
    diameter_low, diameter_high = TREE_CANOPY_DIAMETER_RANGE_M
    assert vegetation["properties"]["height_m"]["minimum"] == height_low
    assert vegetation["properties"]["height_m"]["maximum"] == height_high
    assert vegetation["properties"]["canopy_radius_m"]["minimum"] == diameter_low / 2
    assert vegetation["properties"]["canopy_radius_m"]["maximum"] == diameter_high / 2
    assert vegetation["identity_property"] == "tree_id"
    assert vegetation["rejected_properties"] == ["transmissivity"]
    # trunk_ratio's disclosed default is TreeSpec's own dataclass default.
    defaults = {
        spec.name: spec.default for spec in dataclasses.fields(TreeSpec)
    }
    assert (
        vegetation["properties"]["trunk_ratio"]["default"] == defaults["trunk_ratio"]
    )
    # trunk_ratio's bounds themselves are derived (TRUNK_RATIO_RANGE) —
    # pinned to the owning validator in
    # test_trunk_ratio_bounds_derive_from_the_tree_validator below.
    low, high = TRUNK_RATIO_RANGE
    assert vegetation["properties"]["trunk_ratio"]["minimum"] == low
    assert vegetation["properties"]["trunk_ratio"]["maximum"] == high

    building = schemas["building_geometry"]
    assert building["properties"]["height_m"]["minimum"] == HEIGHT_MIN_M
    assert building["properties"]["height_m"]["exclusive_minimum"] is True
    assert building["properties"]["height_m"]["maximum"] == HEIGHT_MAX_M
    assert building["properties"]["footprint_m"]["min_items"] == MIN_FOOTPRINT_VERTICES
    assert building["identity_property"] == "building_id"

    landcover = schemas["landcover_surface"]
    assert landcover["valid_classes"] == sorted(LANDCOVER_VALID_CLASSES)
    assert landcover["fenced_classes"] == sorted(LANDCOVER_FENCED_CLASSES)

    meteorology = schemas["meteorological_forcing"]
    assert set(meteorology["properties"]) == {spec.variable for spec in VARIABLE_SPECS}
    for spec in VARIABLE_SPECS:
        entry = meteorology["properties"][spec.variable]
        assert entry["units"] == spec.units
        assert entry["minimum"] == spec.low
        assert entry["maximum"] == spec.high
        assert entry["bounds_basis"] == spec.bounds_basis
    assert meteorology["blocked_variables"] == sorted(METEOROLOGY_VARIABLES_BLOCKED)

    parameters = schemas["model_receptor_parameters"]
    assert set(parameters["properties"]) == {spec.name for spec in PARAMETER_SPECS}
    by_name = {spec.name: spec for spec in PARAMETER_SPECS}
    for name, entry in parameters["properties"].items():
        spec = by_name[name]
        assert entry["type"] == spec.kind
        assert entry.get("minimum") == spec.low
        assert entry.get("maximum") == spec.high
        assert entry["default"] == spec.default
        assert entry["default_source"] == spec.default_source
        assert entry.get("exclusive_minimum", False) == spec.low_exclusive
    assert parameters["blocked_parameters"] == sorted(MODEL_PARAMETERS_BLOCKED)


def test_trunk_ratio_bounds_derive_from_the_tree_validator() -> None:
    """u-d1b nit a: the document's trunk_ratio bounds are IMPORTED from
    trees.py (TRUNK_RATIO_RANGE), never restated — and the range itself
    is pinned to what validate_tree_spec actually enforces, so neither
    side can drift from the other silently."""
    document = capability_document()
    vegetation = {
        entry["id"]: entry for entry in document["adapters"]
    }["vegetation_geometry"]
    entry = vegetation["property_schema"]["properties"]["trunk_ratio"]
    low, high = TRUNK_RATIO_RANGE
    assert entry["minimum"] == low
    assert entry["maximum"] == high
    assert entry["exclusive_maximum"] is True  # the half-open rendering

    # The owning validator enforces exactly that half-open interval.
    base = {
        "tree_id": "t",
        "x_m": 0.0,
        "y_m": 0.0,
        "height_m": 5.0,
        "canopy_radius_m": 2.0,
    }
    assert validate_tree_spec(TreeSpec(**base, trunk_ratio=low)) == TreeSpec(
        **base, trunk_ratio=low
    )
    with pytest.raises(ValueError, match="trunk_ratio"):
        validate_tree_spec(TreeSpec(**base, trunk_ratio=high))
    with pytest.raises(ValueError, match="trunk_ratio"):
        validate_tree_spec(TreeSpec(**base, trunk_ratio=high + 1.0))


def test_view_only_section_answers_uedit_007() -> None:
    document = capability_document()
    view = document["view_only"]
    view_metadata = [
        metadata
        for metadata in builtin_adapter_metadata()
        if metadata.status is AdapterStatus.VIEW_ONLY
    ]
    assert view_metadata, "the builtin registry must carry the output_view adapter"
    expected_layers = sorted(
        {layer for metadata in view_metadata for layer in metadata.producible_incremental}
    )
    expected_operations = sorted(
        {op for metadata in view_metadata for op in metadata.operations}
    )
    assert view["producible_layers"] == expected_layers
    assert view["operations"] == expected_operations
    assert view["zero_scientific_jobs"] is True
    assert producible_view_layers() == tuple(expected_layers)
    assert "wbgt" not in view["producible_layers"]


def test_locality_modes_derived_from_scope() -> None:
    assert locality_modes_for_scope("windows") == ("local", "full")
    assert locality_modes_for_scope("directional_windows") == ("local", "full")
    assert locality_modes_for_scope("conservative_windows_or_full") == ("local", "full")
    assert locality_modes_for_scope("full") == ("full",)
    assert locality_modes_for_scope("full_downstream_only") == ("full",)
    assert locality_modes_for_scope("full_downstream_stage_specific") == ("full",)
    assert locality_modes_for_scope("none") == ()
    # Unknown scopes degrade conservatively to full-only.
    assert locality_modes_for_scope("something_new") == ("full",)
    document = capability_document()
    entries = {entry["id"]: entry for entry in document["adapters"]}
    assert entries["vegetation_geometry"]["locality_modes"] == ["local", "full"]
    assert entries["terrain_dem"]["locality_modes"] == ["full"]
    assert entries["output_view"]["locality_modes"] == []
    assert document["locality_modes"] == ["local", "full"]
    assert document["schema_version"] == CAPABILITY_SCHEMA_VERSION
