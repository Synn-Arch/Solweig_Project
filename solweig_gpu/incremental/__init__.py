# SPDX-License-Identifier: GPL-3.0-only
"""Experimental primitives for a CPU incremental SOLWEIG design tool.

These APIs define invalidation geometry, packed visibility storage, edit
coalescing, editable-tree rasterization with spatial queries, and a
validated memory-mapped baseline site cache. They do not yet replace the
scientific SOLWEIG kernels.
"""

from .bitmask import (
    PACKED_STACK_NAMES,
    SHADOW_STACK_POLARITY,
    PackedVisibility,
    PackedVisibility2Bit,
    bytes_per_pixel,
    bytes_per_pixel_2bit,
    create_packed_memmap,
    iter_patch_blocks,
    pack_visibility,
    pack_visibility_2bit,
    packed_nbytes,
    set_patch_window,
    set_patch_window_2bit,
    spatial_window_view,
    spatial_window_view_2bit,
    unpack_patch,
    unpack_patch_2bit,
    unpack_visibility,
    unpack_visibility_2bit,
    weighted_sum_from_packed,
    weighted_sum_from_packed_2bit,
)
from .cache import (
    CacheValidationError,
    SiteCache,
)
from .cache_builder import build_site_cache
from .edits import (
    CoalescedEditBatch,
    EditOperation,
    TreeEdit,
    coalesce_tree_edits,
    dirty_windows_for_batch,
)
from .geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    RecomputeMode,
    SunPosition,
    TreeSpec,
    choose_recompute_mode,
    dirty_window_for_edit,
    estimate_svf_radius_m,
    merge_windows,
    shadow_vector_m,
    tree_influence_bounds_m,
    window_fraction,
)
from .io import (
    copy_window,
    read_window,
    read_window_like,
    window_bounds_world,
    write_window,
)
from .manifest import (
    CACHE_ARRAY_LAYOUT,
    CACHE_MODEL_VERSION,
    CACHE_SCHEMA_VERSION,
    ArrayEntry,
    AuxSourceEntry,
    ManifestError,
    ManifestMismatchError,
    MissingManifestKeyError,
    RasterSourceEntry,
    SiteManifest,
    SolarGeometrySpec,
    load_manifest,
)
from .spatial_index import TreeSpatialIndex, WorldRect
from .trees import (
    TreeLayer,
    validate_tree_preset,
    validate_tree_spec,
)

__all__ = [
    "ArrayEntry",
    "AuxSourceEntry",
    "CACHE_ARRAY_LAYOUT",
    "CACHE_MODEL_VERSION",
    "CACHE_SCHEMA_VERSION",
    "CacheValidationError",
    "CoalescedEditBatch",
    "EditOperation",
    "InfluenceConfig",
    "ManifestError",
    "ManifestMismatchError",
    "MissingManifestKeyError",
    "PACKED_STACK_NAMES",
    "PackedVisibility",
    "PackedVisibility2Bit",
    "RasterGrid",
    "RasterSourceEntry",
    "RasterWindow",
    "RecomputeMode",
    "SHADOW_STACK_POLARITY",
    "SiteCache",
    "SiteManifest",
    "SolarGeometrySpec",
    "SunPosition",
    "TreeEdit",
    "TreeLayer",
    "TreeSpatialIndex",
    "TreeSpec",
    "WorldRect",
    "build_site_cache",
    "bytes_per_pixel",
    "bytes_per_pixel_2bit",
    "choose_recompute_mode",
    "coalesce_tree_edits",
    "copy_window",
    "create_packed_memmap",
    "dirty_window_for_edit",
    "dirty_windows_for_batch",
    "estimate_svf_radius_m",
    "iter_patch_blocks",
    "load_manifest",
    "merge_windows",
    "pack_visibility",
    "pack_visibility_2bit",
    "packed_nbytes",
    "read_window",
    "read_window_like",
    "set_patch_window",
    "set_patch_window_2bit",
    "shadow_vector_m",
    "spatial_window_view",
    "spatial_window_view_2bit",
    "tree_influence_bounds_m",
    "unpack_patch",
    "unpack_patch_2bit",
    "unpack_visibility",
    "unpack_visibility_2bit",
    "validate_tree_preset",
    "validate_tree_spec",
    "weighted_sum_from_packed",
    "weighted_sum_from_packed_2bit",
    "window_bounds_world",
    "window_fraction",
    "write_window",
]

# Phase 5 additions (exact CPU window worker). Appended only; the imports
# above are unchanged.
from .result import (
    PATCH_SCHEMA_VERSION,
    PatchChecksumError,
    PatchError,
    ResultPatch,
    discard_staging,
    load_patch,
    new_job_id,
    publish_staged_patch,
    stage_patch,
)
from .solver import (
    GVF_MARCH_METERS,
    HALO_SAFETY_PIXELS,
    FullSceneTensors,
    SiteForcing,
    SolverInputError,
    StaleSvfError,
    compose_full_scene_tensors,
    gvf_march_pixels,
    load_site_forcing,
    read_window_for_write_window,
    required_halo_pixels,
    run_full_tile,
    solve_window,
    window_svf_bundle,
)
from .worker import ExactWorker, JobOutcome, SupersededJobError, WorkerError

# Phase 5 star-export names (append-only).
__all__ += [
    "GVF_MARCH_METERS",
    "HALO_SAFETY_PIXELS",
    "FullSceneTensors",
    "JobOutcome",
    "PATCH_SCHEMA_VERSION",
    "PatchChecksumError",
    "PatchError",
    "ResultPatch",
    "SiteForcing",
    "SolverInputError",
    "StaleSvfError",
    "SupersededJobError",
    "WorkerError",
    "ExactWorker",
    "compose_full_scene_tensors",
    "discard_staging",
    "gvf_march_pixels",
    "load_patch",
    "load_site_forcing",
    "new_job_id",
    "publish_staged_patch",
    "read_window_for_write_window",
    "required_halo_pixels",
    "run_full_tile",
    "solve_window",
    "stage_patch",
    "window_svf_bundle",
]

# Wave U-B additions (universal adapter-driven editing engine core):
# typed edit deltas/plan records, the versioned dependency graph, the
# adapter registry with enforced lead rulings, and the topological
# planner. Appended only; the imports above are unchanged.
from .edit_types import (
    BuildingMassingDelta,
    EditCommand,
    EditStateError,
    ForcingChange,
    ForcingDelta,
    FrozenMapping,
    ImpactPlan,
    LandCoverPaintDelta,
    LandCoverPaintPatch,
    ModelParameterChange,
    ModelParameterDelta,
    NodeImpact,
    ObjectStateChange,
    OutputSelectionDelta,
    PreviewDescriptor,
    ScenarioTransaction,
    SiteContext,
    SourceDelta,
    SourceDeltaError,
    SpatialScope,
    TemporalScope,
    TransactionError,
    ValidatedEdit,
    VegetationObjectDelta,
    coalesce_source_deltas,
    freeze_mapping,
)
from .edit_graph import (
    GRAPH_SCHEMA_VERSION,
    EditGraph,
    EditGraphError,
    GraphNode,
    NodeKind,
    SceneGraphState,
    default_edit_graph,
)
from .edit_registry import (
    LANDCOVER_FENCED_CLASSES,
    LANDCOVER_VALID_CLASSES,
    MODEL_PARAMETERS_BLOCKED,
    MODEL_PARAMETERS_SAFE,
    REGISTRY_SCHEMA_VERSION,
    VEGETATION_REJECTED_PROPERTIES,
    AdapterMetadata,
    AdapterRegistry,
    AdapterRegistryError,
    AdapterSchemaError,
    AdapterStatus,
    EditAdapter,
    builtin_adapter_metadata,
    builtin_registry,
)
from .planner import (
    NODE_EXECUTION_COSTS,
    ConservativeSafetyPolicy,
    EditPlanner,
    LocalityDecision,
    NodeExecutionCost,
    PlanningError,
    SafetyPolicy,
)


# Wave U-B star-export names (append-only).
__all__ += [
    "AdapterMetadata",
    "AdapterRegistry",
    "AdapterRegistryError",
    "AdapterSchemaError",
    "AdapterStatus",
    "BuildingMassingDelta",
    "ConservativeSafetyPolicy",
    "EditAdapter",
    "EditCommand",
    "EditGraph",
    "EditGraphError",
    "EditPlanner",
    "EditStateError",
    "ForcingChange",
    "ForcingDelta",
    "FrozenMapping",
    "GRAPH_SCHEMA_VERSION",
    "GraphNode",
    "ImpactPlan",
    "LANDCOVER_FENCED_CLASSES",
    "LANDCOVER_VALID_CLASSES",
    "LandCoverPaintDelta",
    "LandCoverPaintPatch",
    "LocalityDecision",
    "MODEL_PARAMETERS_BLOCKED",
    "MODEL_PARAMETERS_SAFE",
    "ModelParameterChange",
    "ModelParameterDelta",
    "NODE_EXECUTION_COSTS",
    "NodeExecutionCost",
    "NodeImpact",
    "NodeKind",
    "ObjectStateChange",
    "OutputSelectionDelta",
    "PlanningError",
    "PreviewDescriptor",
    "REGISTRY_SCHEMA_VERSION",
    "ScenarioTransaction",
    "SafetyPolicy",
    "SceneGraphState",
    "SiteContext",
    "SourceDelta",
    "SourceDeltaError",
    "SpatialScope",
    "TemporalScope",
    "TransactionError",
    "VEGETATION_REJECTED_PROPERTIES",
    "ValidatedEdit",
    "VegetationObjectDelta",
    "builtin_adapter_metadata",
    "builtin_registry",
    "coalesce_source_deltas",
    "default_edit_graph",
    "freeze_mapping",
]

# Wave U-B-veg additions (first concrete EditAdapter: vegetation geometry).
# Appended only; the imports above are unchanged.
from .adapters import (
    ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION,
    EDITABLE_PROPERTIES,
    IDENTITY_PROPERTY,
    VegetationGeometryAdapter,
    coalesced_batch_from_deltas,
    register_default_adapters,
    tree_edits_from_deltas,
)

# Wave U-B-veg star-export names (append-only).
__all__ += [
    "ADAPTER_ID",
    "ADAPTER_SCHEMA_VERSION",
    "EDITABLE_PROPERTIES",
    "IDENTITY_PROPERTY",
    "VegetationGeometryAdapter",
    "coalesced_batch_from_deltas",
    "register_default_adapters",
    "tree_edits_from_deltas",
]

# Wave U-B-params additions (model_receptor_parameters EditAdapter).
# Appended only; the imports above are unchanged. ``ADAPTER_ID`` and
# ``ADAPTER_SCHEMA_VERSION`` stay vegetation's (they were taken first), so
# this adapter's constants are re-exported under MODEL_PARAMETERS_-prefixed
# names.
from .adapters import (
    MODEL_PARAMETERS_ADAPTER_ID,
    MODEL_PARAMETERS_ADAPTER_SCHEMA_VERSION,
    PARAMETER_DEFAULTS,
    PARAMETER_SPECS,
    PLUMBING_CLASSIFICATION,
    ModelReceptorParametersAdapter,
    ParameterSpec,
    before_values_from_deltas,
    kernel_arguments_from_deltas,
    register_model_parameters_adapter,
)

# Wave U-B-params star-export names (append-only).
__all__ += [
    "MODEL_PARAMETERS_ADAPTER_ID",
    "MODEL_PARAMETERS_ADAPTER_SCHEMA_VERSION",
    "PARAMETER_DEFAULTS",
    "PARAMETER_SPECS",
    "PLUMBING_CLASSIFICATION",
    "ModelReceptorParametersAdapter",
    "ParameterSpec",
    "before_values_from_deltas",
    "kernel_arguments_from_deltas",
    "register_model_parameters_adapter",
]

# Wave U-B-met additions (meteorological_forcing EditAdapter). Appended
# only; the imports above are unchanged. ``ADAPTER_ID`` and
# ``ADAPTER_SCHEMA_VERSION`` stay vegetation's, so this adapter's constants
# are re-exported under METEOROLOGY_-prefixed names.
from .edit_registry import (
    METEOROLOGY_SOURCE_NODE,
    METEOROLOGY_VARIABLES_BLOCKED,
    METEOROLOGY_VARIABLES_SAFE,
)
from .adapters import (
    MET_COLUMN_COUNT,
    METEOROLOGY_ADAPTER_ID,
    METEOROLOGY_ADAPTER_SCHEMA_VERSION,
    VARIABLE_SPECS,
    ForcingOverlay,
    ForcingVariableSpec,
    MeteorologicalForcingAdapter,
    overlay_from_deltas,
    register_meteorological_forcing_adapter,
)

# Wave U-B-met star-export names (append-only).
__all__ += [
    "MET_COLUMN_COUNT",
    "METEOROLOGY_ADAPTER_ID",
    "METEOROLOGY_ADAPTER_SCHEMA_VERSION",
    "METEOROLOGY_SOURCE_NODE",
    "METEOROLOGY_VARIABLES_BLOCKED",
    "METEOROLOGY_VARIABLES_SAFE",
    "VARIABLE_SPECS",
    "ForcingOverlay",
    "ForcingVariableSpec",
    "MeteorologicalForcingAdapter",
    "overlay_from_deltas",
    "register_meteorological_forcing_adapter",
]

# Wave U-B-lc additions (landcover_surface EditAdapter). Appended only;
# the imports above are unchanged. ``ADAPTER_ID`` and
# ``ADAPTER_SCHEMA_VERSION`` stay vegetation's, so this adapter's constants
# are re-exported under LANDCOVER_-prefixed names.
from .edit_registry import LANDCOVER_SOURCE_NODE
from .adapters import (
    FENCED_CLASSES,
    LANDCOVER_ADAPTER_ID,
    LANDCOVER_ADAPTER_SCHEMA_VERSION,
    PAINTABLE_CLASSES,
    LandCoverOverlay,
    LandCoverSurfaceAdapter,
    landcover_overlay_from_deltas,
    register_landcover_surface_adapter,
)

# Wave U-B-lc star-export names (append-only).
__all__ += [
    "FENCED_CLASSES",
    "LANDCOVER_ADAPTER_ID",
    "LANDCOVER_ADAPTER_SCHEMA_VERSION",
    "LANDCOVER_SOURCE_NODE",
    "PAINTABLE_CLASSES",
    "LandCoverOverlay",
    "LandCoverSurfaceAdapter",
    "landcover_overlay_from_deltas",
    "register_landcover_surface_adapter",
]

# Wave U-B-bld additions (building_geometry EditAdapter, full-only).
# Appended only; the imports above are unchanged. ``ADAPTER_ID`` and
# ``ADAPTER_SCHEMA_VERSION`` stay vegetation's, so this adapter's constants
# are re-exported under BUILDING_-prefixed names.
from .adapters import (
    BUILDING_ADAPTER_ID,
    BUILDING_ADAPTER_SCHEMA_VERSION,
    BUILDING_EDITABLE_PROPERTIES,
    BUILDING_IDENTITY_PROPERTY,
    BUILDING_SOURCE_NODE,
    BuildingMassingAdapter,
    BuildingSpec,
    HEIGHT_MAX_M,
    HEIGHT_MIN_M,
    MIN_FOOTPRINT_VERTICES,
    MassingEdit,
    massing_edits_from_deltas,
    register_building_massing_adapter,
)

# Wave U-B-bld star-export names (append-only).
__all__ += [
    "BUILDING_ADAPTER_ID",
    "BUILDING_ADAPTER_SCHEMA_VERSION",
    "BUILDING_EDITABLE_PROPERTIES",
    "BUILDING_IDENTITY_PROPERTY",
    "BUILDING_SOURCE_NODE",
    "BuildingMassingAdapter",
    "BuildingSpec",
    "HEIGHT_MAX_M",
    "HEIGHT_MIN_M",
    "MIN_FOOTPRINT_VERTICES",
    "MassingEdit",
    "massing_edits_from_deltas",
    "register_building_massing_adapter",
]

# Wave U-C5 additions (building rasterizer + full regeneration chain).
# Appended only; the imports above are unchanged.
from .buildings import (
    WALL_LIMIT_M,
    BuildingLayer,
    BuildingRasterResult,
    MassingRasterRecord,
)
from .regenerate import (
    STAGE_BUILDING_DSM,
    STAGE_CACHE,
    STAGE_FULL_TILE,
    STAGE_SITE,
    STAGE_SVF,
    STAGE_TREES,
    STAGE_WALLS_ASPECT,
    BuildingRegenerationResult,
    RegenerationError,
    rebuild_scenario_cache,
    regenerate_building_batch,
    regenerate_svf,
    regenerate_walls_aspect,
    run_scenario_full_tile,
    stage_scenario_site,
    write_scenario_building_dsm,
    write_scenario_trees,
)

# Wave U-C5 star-export names (append-only).
__all__ += [
    "STAGE_BUILDING_DSM",
    "STAGE_CACHE",
    "STAGE_FULL_TILE",
    "STAGE_SITE",
    "STAGE_SVF",
    "STAGE_TREES",
    "STAGE_WALLS_ASPECT",
    "WALL_LIMIT_M",
    "BuildingLayer",
    "BuildingRasterResult",
    "BuildingRegenerationResult",
    "MassingRasterRecord",
    "RegenerationError",
    "rebuild_scenario_cache",
    "regenerate_building_batch",
    "regenerate_svf",
    "regenerate_walls_aspect",
    "run_scenario_full_tile",
    "stage_scenario_site",
    "write_scenario_building_dsm",
    "write_scenario_trees",
]

# Wave U-C1 additions (temporal result store, keyed (node_id, time_index)).
# Appended only; every import above is unchanged.
from .store import (
    StaleResultError,
    StoreError,
    TemporalResultEntry,
    TemporalResultKey,
    TemporalResultStore,
)

# Wave U-C1 star-export names (append-only).
__all__ += [
    "StaleResultError",
    "StoreError",
    "TemporalResultEntry",
    "TemporalResultKey",
    "TemporalResultStore",
]

# Wave U-C1 additions (plan executor: ImpactPlan -> real recomputes ->
# published results, absorbing engine LOWs L2/L3/L5 and M2/M3 residuals).
# Appended only; every import above is unchanged.
from .executor import (
    DEFAULT_TEMPORAL_STEPS,
    EXECUTABLE_SOURCE_NODES,
    ExecutedPlan,
    ExecutorError,
    NotIntegratedError,
    PlanExecutor,
    VARIABLE_TO_RESULT_NODE,
    detect_wind_coefficients,
    stage_read_window,
)
from .planner import fence_requested_times

# Wave U-C1 star-export names (append-only).
__all__ += [
    "DEFAULT_TEMPORAL_STEPS",
    "EXECUTABLE_SOURCE_NODES",
    "ExecutedPlan",
    "ExecutorError",
    "NotIntegratedError",
    "PlanExecutor",
    "VARIABLE_TO_RESULT_NODE",
    "detect_wind_coefficients",
    "fence_requested_times",
    "stage_read_window",
]

# Wave U-D1 additions (capability document + scenario state persistence).
# Appended only; every import above is unchanged.
from .capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    LOCALITY_MODES,
    capability_document,
    full_registry,
    locality_modes_for_scope,
    producible_view_layers,
)
from .scenario_state import (
    SCENARIO_STATE_SCHEMA_VERSION,
    ScenarioStateError,
    ScenarioStateSnapshot,
    read_snapshot,
    restore_into_executor,
    snapshot_from_executor,
    write_snapshot,
)

# Wave U-D1 star-export names (append-only).
__all__ += [
    "CAPABILITY_SCHEMA_VERSION",
    "LOCALITY_MODES",
    "SCENARIO_STATE_SCHEMA_VERSION",
    "ScenarioStateError",
    "ScenarioStateSnapshot",
    "capability_document",
    "full_registry",
    "locality_modes_for_scope",
    "producible_view_layers",
    "read_snapshot",
    "restore_into_executor",
    "snapshot_from_executor",
    "write_snapshot",
]

# Wave U-D1 store reader-contract re-exports (u-d0 added CoverageStatus /
# WindowCoverage to store.py's __all__; surfaced here per the append-only
# convention — nothing above changes).
from .store import CoverageStatus, WindowCoverage

# Wave U-D1 star-export names (append-only).
__all__ += [
    "CoverageStatus",
    "WindowCoverage",
]
