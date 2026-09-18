# SPDX-License-Identifier: GPL-3.0-only
"""Concrete :class:`~solweig_gpu.incremental.edit_registry.EditAdapter` implementations.

Each module in this package implements the adapter protocol for one editable
source family by *wrapping* the existing incremental machinery rather than
duplicating it. Registration is explicit and idempotent: call
:func:`register_default_adapters` with a registry (or ``None`` for a fresh
builtin one) to bind every implemented adapter, exactly like the
``registry.register(VegetationAdapter(...))`` sequence in
``docs/incremental_design_tool/universal_editing/edit_adapter_contract.md``.
"""

from .vegetation import (
    ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION,
    EDITABLE_PROPERTIES,
    IDENTITY_PROPERTY,
    VegetationGeometryAdapter,
    coalesced_batch_from_deltas,
    register_default_adapters,
    tree_edits_from_deltas,
)

__all__ = [
    "ADAPTER_ID",
    "ADAPTER_SCHEMA_VERSION",
    "EDITABLE_PROPERTIES",
    "IDENTITY_PROPERTY",
    "VegetationGeometryAdapter",
    "coalesced_batch_from_deltas",
    "register_default_adapters",
    "tree_edits_from_deltas",
]

# Wave U-B-params additions (model_receptor_parameters adapter). Appended
# only; the vegetation imports and exports above are unchanged. The generic
# constant names stay vegetation's (they were taken first); this adapter
# exports its own under MODEL_PARAMETERS_-prefixed names.
from .model_parameters import (
    PARAMETER_DEFAULTS,
    PARAMETER_SPECS,
    PLUMBING_CLASSIFICATION,
    ModelReceptorParametersAdapter,
    ParameterSpec,
    before_values_from_deltas,
    kernel_arguments_from_deltas,
    register_model_parameters_adapter,
)
from .model_parameters import (
    ADAPTER_ID as MODEL_PARAMETERS_ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION as MODEL_PARAMETERS_ADAPTER_SCHEMA_VERSION,
)

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

# Wave U-B-met additions (meteorological_forcing adapter). Appended only;
# the vegetation and model-parameter imports/exports above are unchanged.
# The generic constant names stay vegetation's; this adapter exports its
# id/schema under METEOROLOGY_-prefixed names.
from .met_time import (
    MET_COLUMN_COUNT,
    VARIABLE_SPECS,
    ForcingOverlay,
    ForcingVariableSpec,
    MeteorologicalForcingAdapter,
    overlay_from_deltas,
    register_meteorological_forcing_adapter,
)
from .met_time import (
    ADAPTER_ID as METEOROLOGY_ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION as METEOROLOGY_ADAPTER_SCHEMA_VERSION,
)

__all__ += [
    "MET_COLUMN_COUNT",
    "METEOROLOGY_ADAPTER_ID",
    "METEOROLOGY_ADAPTER_SCHEMA_VERSION",
    "VARIABLE_SPECS",
    "ForcingOverlay",
    "ForcingVariableSpec",
    "MeteorologicalForcingAdapter",
    "overlay_from_deltas",
    "register_meteorological_forcing_adapter",
]

# Wave U-B-bld additions (building_geometry adapter, full-only). Appended
# only; the vegetation, model-parameter, and meteorology imports/exports
# above are unchanged. The generic constant names stay vegetation's; this
# adapter exports its id/schema under BUILDING_-prefixed names.
from .building import (
    BUILDING_SOURCE_NODE,
    EDITABLE_PROPERTIES as BUILDING_EDITABLE_PROPERTIES,
    HEIGHT_MAX_M,
    HEIGHT_MIN_M,
    IDENTITY_PROPERTY as BUILDING_IDENTITY_PROPERTY,
    MIN_FOOTPRINT_VERTICES,
    BuildingMassingAdapter,
    BuildingSpec,
    MassingEdit,
    massing_edits_from_deltas,
    register_building_massing_adapter,
)
from .building import (
    ADAPTER_ID as BUILDING_ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION as BUILDING_ADAPTER_SCHEMA_VERSION,
)

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

# Extend the vegetation helper additively (its own definition is untouched):
# registering "every implemented adapter" now binds both concrete adapters.
from ..edit_registry import AdapterRegistry

_register_default_adapters = register_default_adapters


def register_default_adapters(
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register every implemented adapter (idempotent) and return the registry.

    Capability listing for the U-D API is then just
    ``registry.all_metadata()``; concrete adapters resolve through
    ``registry.get_adapter``. Composes the vegetation helper with
    :func:`register_model_parameters_adapter`,
    :func:`register_meteorological_forcing_adapter`, and
    :func:`register_building_massing_adapter` so no module's
    registration code is duplicated or restructured.
    """
    target = _register_default_adapters(registry)
    target = register_model_parameters_adapter(target)
    return register_meteorological_forcing_adapter(target)


# Wave U-B-lc additions (landcover_surface adapter). Appended only; the
# vegetation, model-parameter, and met adapter imports/exports above are
# unchanged. The generic constant names stay vegetation's, so this adapter
# exports its id/schema under LANDCOVER_-prefixed names.
from .landcover import (
    FENCED_CLASSES,
    PAINTABLE_CLASSES,
    LandCoverOverlay,
    LandCoverSurfaceAdapter,
    landcover_overlay_from_deltas,
    register_landcover_surface_adapter,
)
from .landcover import (
    ADAPTER_ID as LANDCOVER_ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION as LANDCOVER_ADAPTER_SCHEMA_VERSION,
)

__all__ += [
    "FENCED_CLASSES",
    "LANDCOVER_ADAPTER_ID",
    "LANDCOVER_ADAPTER_SCHEMA_VERSION",
    "PAINTABLE_CLASSES",
    "LandCoverOverlay",
    "LandCoverSurfaceAdapter",
    "landcover_overlay_from_deltas",
    "register_landcover_surface_adapter",
]

# Extend the composed default registration additively: the definition
# above (vegetation + parameters + met) is untouched; "every implemented
# adapter" now also binds the land-cover adapter.
_register_default_adapters_with_met = register_default_adapters


def register_default_adapters(
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register every implemented adapter (idempotent) and return the registry.

    Composes the earlier helper chain (vegetation, model parameters,
    meteorological forcing) with :func:`register_landcover_surface_adapter`
    so no module's registration code is duplicated or restructured.
    """
    return register_landcover_surface_adapter(
        _register_default_adapters_with_met(registry)
    )


# Wave U-B-bld registration composition (imports/exports already appended
# above): extend the composed default registration additively once more —
# the landcover composition above is untouched; "every implemented
# adapter" now also binds the building massing adapter.
_register_default_adapters_with_lc = register_default_adapters


def register_default_adapters(  # noqa: F811 — additive wave composition
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register every implemented adapter (idempotent) and return the registry.

    Composes the full wave chain (vegetation, model parameters,
    meteorological forcing, landcover) with
    :func:`register_building_massing_adapter` so no module's registration
    code is duplicated or restructured.
    """
    return register_building_massing_adapter(
        _register_default_adapters_with_lc(registry)
    )
