# SPDX-License-Identifier: GPL-3.0-only
"""Capability document for the universal editing engine (UEDIT-007).

Pure, importable, and free of any server dependency: this module composes
ONE answer to "what can this build edit, within which fences, producing
which outputs?" from the sources that already own those facts —

* adapter metadata: :class:`~solweig_gpu.incremental.edit_registry.AdapterRegistry`
  (groups, statuses, operations, scopes, previews, class/parameter fences);
* the dependency graph: :class:`~solweig_gpu.incremental.edit_graph.default_edit_graph`
  (the per-family dirty-node closure);
* the adapter spec tables: the validated property domains declared next to
  each adapter's validation code (``PARAMETER_SPECS``, ``VARIABLE_SPECS``,
  vegetation/building property constants).

Nothing here re-states a fence or a bound by hand: every value is read from
its owning table at call time, so a registry or spec change flows into the
capability document without a second edit here. The only mapping this module
adds is *which* spec table serves *which* adapter id — wiring, not data.

The server surface (``GET /api/v1/capabilities`` and the view-only
scenario views endpoint) serves :func:`capability_document` verbatim;
``docs/incremental_design_tool/universal_editing/frontend_spec.md`` makes
that retrieval the frontend's single discovery point: "The frontend
retrieves capability metadata from the API. It should not hard-code the
complete tool set."
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .adapters import register_default_adapters
from .adapters.building import HEIGHT_MAX_M, HEIGHT_MIN_M, MIN_FOOTPRINT_VERTICES
from .adapters.building import EDITABLE_PROPERTIES as BUILDING_EDITABLE_PROPERTIES
from .adapters.building import IDENTITY_PROPERTY as BUILDING_IDENTITY_PROPERTY
from .adapters.met_time import VARIABLE_SPECS
from .adapters.model_parameters import PARAMETER_SPECS
from .adapters.vegetation import (
    EDITABLE_PROPERTIES as VEGETATION_EDITABLE_PROPERTIES,
)
from .adapters.vegetation import IDENTITY_PROPERTY as VEGETATION_IDENTITY_PROPERTY
from .edit_graph import EditGraph, default_edit_graph
from .edit_registry import (
    LANDCOVER_FENCED_CLASSES,
    LANDCOVER_VALID_CLASSES,
    METEOROLOGY_VARIABLES_BLOCKED,
    METEOROLOGY_VARIABLES_SAFE,
    MODEL_PARAMETERS_BLOCKED,
    MODEL_PARAMETERS_SAFE,
    VEGETATION_REJECTED_PROPERTIES,
    AdapterMetadata,
    AdapterRegistry,
    AdapterStatus,
    builtin_registry,
)
from .geometry import TreeSpec
from .trees import TREE_CANOPY_DIAMETER_RANGE_M, TREE_HEIGHT_RANGE_M
from .trees import TRUNK_RATIO_RANGE

__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "LOCALITY_MODES",
    "capability_document",
    "full_registry",
    "locality_modes_for_scope",
    "producible_view_layers",
]

#: Bump when the document's shape (not its data) changes. Data changes ride
#: the registry/schema versions of their owning tables.
CAPABILITY_SCHEMA_VERSION = 1

#: The locality vocabulary the engine implements today. Windowed families
#: degrade to full via their conservative policy; full-only families never
#: promise local (``full_only_initially`` is exactly that disclosure).
LOCALITY_MODES = ("local", "full")

#: Scope name -> advertised locality modes. Derived from the metadata's
#: ``nominal_spatial_scope`` vocabulary, never hand-listed per adapter.
_SCOPE_LOCALITY_MODES: dict[str, tuple[str, ...]] = {
    "windows": ("local", "full"),
    "directional_windows": ("local", "full"),
    "conservative_windows_or_full": ("local", "full"),
    "full": ("full",),
    "full_downstream_only": ("full",),
    "full_downstream_stage_specific": ("full",),
    "none": (),
}


def locality_modes_for_scope(nominal_spatial_scope: str) -> tuple[str, ...]:
    """Advertised locality modes for one ``nominal_spatial_scope`` value.

    Unknown scope names default to full-only (the conservative answer: an
    unlisted scope has no validated local path to promise).
    """
    return _SCOPE_LOCALITY_MODES.get(nominal_spatial_scope, ("full",))


def full_registry(
    registry: AdapterRegistry | None = None,
    graph: EditGraph | None = None,
) -> AdapterRegistry:
    """A registry with every implemented adapter bound.

    Mirrors the executor's own wiring (``register_default_adapters`` over
    ``builtin_registry``) minus the site-configured vegetation hot-swap,
    which changes no metadata. ``registry.get_adapter`` then succeeds for
    exactly the families the executor can execute end-to-end — the same
    fact ``PlanExecutor`` relies on — so the capability document's
    ``integrated`` flag can never drift from what the engine really runs.
    """
    target = registry if registry is not None else builtin_registry(graph)
    return register_default_adapters(target)


# ---------------------------------------------------------------------------
# Property schemas (wired to the adapter-owned spec tables)
# ---------------------------------------------------------------------------


def _tree_default(field_name: str) -> Any:
    """A TreeSpec field's declared default (the preset's starting value)."""
    import dataclasses

    for spec in dataclasses.fields(TreeSpec):
        if spec.name == field_name:
            return spec.default
    raise KeyError(field_name)


def _vegetation_property_schema() -> dict[str, Any]:
    height_low, height_high = TREE_HEIGHT_RANGE_M
    diameter_low, diameter_high = TREE_CANOPY_DIAMETER_RANGE_M
    ratio_low, ratio_high = TRUNK_RATIO_RANGE
    properties: dict[str, Any] = {
        "x_m": {
            "type": "number",
            "units": "m",
            "bounds": "site extent (world coordinates; validated per edit)",
        },
        "y_m": {
            "type": "number",
            "units": "m",
            "bounds": "site extent (world coordinates; validated per edit)",
        },
        "height_m": {
            "type": "number",
            "units": "m",
            "minimum": height_low,
            "maximum": height_high,
        },
        "canopy_radius_m": {
            "type": "number",
            "units": "m",
            "minimum": diameter_low / 2.0,
            "maximum": diameter_high / 2.0,
        },
        # TRUNK_RATIO_RANGE is the half-open [low, high) domain enforced
        # by trees.validate_tree_spec; exclusive_maximum is its JSON
        # Schema rendering, not an independent bound.
        "trunk_ratio": {
            "type": "number",
            "units": "ratio",
            "minimum": ratio_low,
            "maximum": ratio_high,
            "exclusive_maximum": True,
            "default": _tree_default("trunk_ratio"),
        },
    }
    # Canonical order comes from the adapter's own closed vocabulary.
    ordered = {name: properties[name] for name in VEGETATION_EDITABLE_PROPERTIES}
    return {
        "identity_property": VEGETATION_IDENTITY_PROPERTY,
        "properties": ordered,
        "rejected_properties": sorted(VEGETATION_REJECTED_PROPERTIES),
        "rejection_reason": (
            "transmissivity is validated by the data model but inert in the "
            "physics (global transVeg = 0.03, utci_process.py:649); edit it "
            "through model_receptor_parameters.transVeg instead"
        ),
    }


def _building_property_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "footprint_m": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}, "minItems": 2},
            "min_items": MIN_FOOTPRINT_VERTICES,
            "units": "m (site CRS polygon ring, unclosed)",
            "description": (
                f"ring of (x_m, y_m) vertices; at least {MIN_FOOTPRINT_VERTICES} "
                "distinct vertices, closed by the rasterizer"
            ),
        },
        "height_m": {
            "type": "number",
            "units": "m",
            "minimum": HEIGHT_MIN_M,
            "exclusive_minimum": True,
            "maximum": HEIGHT_MAX_M,
        },
    }
    ordered = {name: properties[name] for name in BUILDING_EDITABLE_PROPERTIES}
    return {
        "identity_property": BUILDING_IDENTITY_PROPERTY,
        "properties": ordered,
    }


def _landcover_property_schema(metadata: AdapterMetadata) -> dict[str, Any]:
    return {
        "valid_classes": sorted(metadata.valid_classes or LANDCOVER_VALID_CLASSES),
        "fenced_classes": sorted(LANDCOVER_FENCED_CLASSES),
        "fence_reason": (
            "water (class 7) is fenced: the physics tests lc_grid == 3, never "
            "7 — the Twater branch is dead upstream, so no adapter may claim "
            "water semantics it cannot deliver"
        ),
        "class_semantics": (
            "class codes index the land-cover parameter table (albedo, "
            "emissivity, Psun); painting re-labels cells, it never invents "
            "new materials"
        ),
    }


def _meteorology_property_schema() -> dict[str, Any]:
    return {
        "identity_property": None,
        "properties": {
            spec.variable: {
                "type": "number",
                "units": spec.units,
                "minimum": spec.low,
                "maximum": spec.high,
                "bounds_basis": spec.bounds_basis,
                **({"uncertain": spec.uncertain} if spec.uncertain else {}),
            }
            for spec in VARIABLE_SPECS
        },
        "blocked_variables": sorted(METEOROLOGY_VARIABLES_BLOCKED),
        "time_support": (
            "update_time_row edits one time index; update_range edits a "
            "range; time columns themselves are solar-cache-pinned and "
            "editable only through selected_date_time"
        ),
    }


def _model_parameters_property_schema() -> dict[str, Any]:
    return {
        "identity_property": None,
        "properties": {
            spec.name: {
                "type": spec.kind,
                **(
                    {"minimum": spec.low, "maximum": spec.high}
                    if spec.low is not None or spec.high is not None
                    else {}
                ),
                **({"exclusive_minimum": True} if spec.low_exclusive else {}),
                "default": spec.default,
                "default_source": spec.default_source,
                "bounds_basis": spec.bounds_basis,
                **({"uncertain": spec.uncertain} if spec.uncertain else {}),
            }
            for spec in PARAMETER_SPECS
        },
        "blocked_parameters": sorted(MODEL_PARAMETERS_BLOCKED),
    }


#: Adapter id -> the spec-table reader serving its property schema. This is
#: WIRING ONLY: each provider reads its adapter's validated table (or its
#: registry metadata), so bounds/fences stay single-sourced. An adapter not
#: listed here has no property vocabulary to disclose (raster patches,
#: view selections, or not-yet-implemented families).
_PROPERTY_SCHEMA_PROVIDERS: dict[
    str, Callable[[AdapterMetadata], dict[str, Any] | None]
] = {
    "vegetation_geometry": lambda _meta: _vegetation_property_schema(),
    "building_geometry": lambda _meta: _building_property_schema(),
    "landcover_surface": _landcover_property_schema,
    "meteorological_forcing": lambda _meta: _meteorology_property_schema(),
    "model_receptor_parameters": lambda _meta: _model_parameters_property_schema(),
}


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def _adapter_entry(
    metadata: AdapterMetadata,
    *,
    registry: AdapterRegistry,
    graph: EditGraph,
) -> dict[str, Any]:
    # ``downstream`` includes the source nodes themselves (the dirty set of
    # a source change is its full consumer closure).
    dirty = graph.downstream(metadata.source_nodes)
    try:
        registry.get_adapter(metadata.id)
        integrated = True
    except Exception:  # noqa: BLE001 — metadata-only registration is a state, not an error path
        integrated = False
    provider = _PROPERTY_SCHEMA_PROVIDERS.get(metadata.id)
    entry: dict[str, Any] = {
        "id": metadata.id,
        "group": metadata.group,
        "status": metadata.status.value,
        "source_nodes": list(metadata.source_nodes),
        "operations": list(metadata.operations),
        "nominal_spatial_scope": metadata.nominal_spatial_scope,
        "locality_modes": list(locality_modes_for_scope(metadata.nominal_spatial_scope)),
        "temporal_scope": metadata.temporal_scope,
        "preview": metadata.preview,
        "validation_fixtures": list(metadata.validation_fixtures),
        "integrated": integrated,
        "dirty_nodes": sorted(dirty),
        "property_schema": provider(metadata) if provider is not None else None,
    }
    if metadata.notes:
        entry["notes"] = metadata.notes
    if metadata.producible_incremental:
        entry["producible_incremental"] = list(metadata.producible_incremental)
    return entry


def _view_only_section(
    registry: AdapterRegistry, graph: EditGraph
) -> dict[str, Any]:
    view_adapters = [
        metadata
        for metadata in registry.all_metadata()
        if metadata.status is AdapterStatus.VIEW_ONLY
    ]
    layers: set[str] = set()
    operations: set[str] = set()
    for metadata in view_adapters:
        layers.update(metadata.producible_incremental)
        operations.update(metadata.operations)
    if not view_adapters:
        raise RuntimeError(
            "no view_only adapter registered: the registry bootstraps "
            "output_view, so this is a registration-order bug"
        )
    return {
        "adapter_ids": [metadata.id for metadata in view_adapters],
        "operations": sorted(operations),
        "producible_layers": sorted(layers),
        "zero_scientific_jobs": True,
        "basis": (
            "UEDIT-007: view operations select among already-published "
            "layers; they are answered from capability metadata plus the "
            "published result manifest and never enqueue a solver job"
        ),
    }


def capability_document(
    registry: AdapterRegistry | None = None,
    *,
    graph: EditGraph | None = None,
) -> dict[str, Any]:
    """The complete, self-describing edit-capability document.

    ``registry`` defaults to a freshly built registry with every
    implemented adapter bound (see :func:`full_registry`), so the
    ``integrated`` flags answer "can the engine execute this family
    end-to-end today" — the same fact the executor checks when it refuses
    a not-yet-integrated source node.
    """
    active_graph = graph if graph is not None else default_edit_graph()
    active_registry = (
        registry if registry is not None else full_registry(graph=active_graph)
    )
    entries = [
        _adapter_entry(metadata, registry=active_registry, graph=active_graph)
        for metadata in active_registry.all_metadata()
    ]
    families: dict[str, dict[str, Any]] = {}
    for metadata in active_registry.all_metadata():
        for node_id in metadata.source_nodes:
            family = families.setdefault(
                node_id,
                {
                    "dirty_nodes": sorted(active_graph.downstream((node_id,))),
                    "adapters": [],
                },
            )
            family["adapters"].append(metadata.id)
    for family in families.values():
        family["adapters"].sort()
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "adapters": entries,
        "edit_families": dict(sorted(families.items())),
        "fences": {
            "landcover_valid_classes": sorted(LANDCOVER_VALID_CLASSES),
            "landcover_fenced_classes": sorted(LANDCOVER_FENCED_CLASSES),
            "vegetation_rejected_properties": sorted(VEGETATION_REJECTED_PROPERTIES),
            "model_parameters_safe": sorted(MODEL_PARAMETERS_SAFE),
            "model_parameters_blocked": sorted(MODEL_PARAMETERS_BLOCKED),
            "meteorology_variables_safe": sorted(METEOROLOGY_VARIABLES_SAFE),
            "meteorology_variables_blocked": sorted(METEOROLOGY_VARIABLES_BLOCKED),
        },
        "locality_modes": list(LOCALITY_MODES),
        "view_only": _view_only_section(active_registry, active_graph),
    }


def producible_view_layers(
    registry: AdapterRegistry | None = None,
    *,
    graph: EditGraph | None = None,
) -> tuple[str, ...]:
    """Layers a view-only operation may select (sorted, from metadata)."""
    document = capability_document(registry, graph=graph)
    return tuple(document["view_only"]["producible_layers"])
