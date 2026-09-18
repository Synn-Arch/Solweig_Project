# SPDX-License-Identifier: GPL-3.0-only
"""Adapter registry for the universal editing engine.

The registry owns adapter *metadata* (mirroring
``docs/incremental_design_tool/universal_editing/edit_registry.yaml``, as
reconciled by the U-A audit) and enforces the lead rulings **in code**, not
just in docs:

- **Land-cover water fence.** Valid land-cover classes are exactly
  ``{1, 2, 5, 6}``; water (7) is FENCED. Registering a land-cover adapter
  that exposes class 7 (or any class outside the valid set) is an error,
  and :meth:`AdapterRegistry.validate_edit_state` rejects any edit state
  that paints a fenced class (lead ruling 2026-09-02: the physics tests
  ``lc_grid == 3``, never 7 — the Twater branch is dead upstream, so an
  adapter must not claim water semantics it cannot deliver).

- **Vegetation transmissivity rejection.** The ``vegetation_geometry``
  property schema must REJECT ``transmissivity``: it is validated by the
  data model but inert in the physics (global ``transVeg = 0.03``,
  ``utci_process.py:649``). Registering a vegetation adapter without this
  rejection declared, or submitting an edit state carrying the property,
  is an error.

- **Model-parameter fences.** ``model_receptor_parameters`` accepts only
  the safe set (kernel args or loop-local constants) and must REJECT the
  blocked set (``patch_option`` SVF-cache-coupled; ``scale``/``location``/
  ``utc`` cache-pinned; ``walllimit`` preprocessing; ``onlyglobal`` a
  module constant whose exposure would activate dead met columns 21/22;
  ``albedo_g`` redundant with the land-cover table).

- **Meteorology variable fence.** ``meteorological_forcing`` deltas may
  only carry the physics-effective, non-time-column forcing vocabulary
  (:data:`METEOROLOGY_VARIABLES_SAFE`); time columns are solar-cache-pinned,
  ``diffuse_radiation``/``direct_radiation`` are dead while
  ``onlyglobal = 1``, and ``wind_direction`` is inert while the incremental
  path passes no wind-coefficient rasters (u-b-met, 2026-09-02).

- **Wind-coefficient fence.** Any adapter that mutates
  ``wind_coefficients`` must be registered with status
  ``scientific_extension`` (UEDIT-010: dynamic wind is disclosed and
  independently validated, never silent).

The builtin metadata entries mirror the YAML one-to-one; a drift-guard
test asserts parity so docs and code cannot diverge. Concrete adapter
*implementations* arrive in later packets via :meth:`AdapterRegistry.register`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from .edit_graph import EditGraph, default_edit_graph
from .edit_types import (
    EditCommand,
    ForcingDelta,
    FrozenMapping,
    ImpactPlan,
    LandCoverPaintDelta,
    ModelParameterDelta,
    ObjectStateChange,
    OutputSelectionDelta,
    PreviewDescriptor,
    ScenarioTransaction,
    SiteContext,
    SourceDelta,
    ValidatedEdit,
    VegetationObjectDelta,
    BuildingMassingDelta,
    freeze_mapping,
)

__all__ = [
    "AdapterMetadata",
    "AdapterRegistry",
    "AdapterRegistryError",
    "AdapterSchemaError",
    "AdapterStatus",
    "EditAdapter",
    "REGISTRY_SCHEMA_VERSION",
    "builtin_adapter_metadata",
    "builtin_registry",
    # Declarative rulings (importable by tests and later adapter packets).
    "LANDCOVER_FENCED_CLASSES",
    "LANDCOVER_VALID_CLASSES",
    "MET_VARIABLE_AFFINITY",
    "MET_VARIABLE_AFFINITY_EVIDENCE",
    "MET_UTCI_ONLY_VARIABLES",
    "METEOROLOGY_VARIABLES_BLOCKED",
    "METEOROLOGY_VARIABLES_SAFE",
    "METEOROLOGY_SOURCE_NODE",
    "MODEL_PARAMETERS_AFFINITY",
    "MODEL_PARAMETERS_AFFINITY_EVIDENCE",
    "MODEL_PARAMETERS_BLOCKED",
    "MODEL_PARAMETERS_SAFE",
    "VEGETATION_REJECTED_PROPERTIES",
    "WIND_SOURCE_NODE",
]

#: Bump when the builtin metadata set or a validation rule changes.
REGISTRY_SCHEMA_VERSION = 1


class AdapterRegistryError(RuntimeError):
    """Registration or lookup failure (duplicate id, unknown node/adapter)."""


class AdapterSchemaError(AdapterRegistryError):
    """An edit state violates an adapter's validated property schema."""


class AdapterStatus(str, Enum):
    """Capability status vocabulary (mirrors ``edit_registry.yaml``)."""

    NATIVE_INPUT = "native_input"
    ADAPTER_REQUIRED = "adapter_required"
    FULL_ONLY_INITIALLY = "full_only_initially"
    SCIENTIFIC_EXTENSION = "scientific_extension"
    VIEW_ONLY = "view_only"


# ---------------------------------------------------------------------------
# Declarative lead rulings (enforced on register and at state validation)
# ---------------------------------------------------------------------------

#: Source node whose valid paint classes are fenced (lead ruling 2026-09-02).
LANDCOVER_SOURCE_NODE = "landcover"
LANDCOVER_VALID_CLASSES = frozenset({1, 2, 5, 6})
LANDCOVER_FENCED_CLASSES = frozenset({7})

#: Vegetation source node and the property that must be rejected (inert in
#: physics: global transVeg, utci_process.py:649).
VEGETATION_SOURCE_NODE = "vegetation_dsm"
VEGETATION_REJECTED_PROPERTIES = frozenset({"transmissivity"})

#: Model-parameter source node: safe (kernel args / loop-local) vs blocked
#: (cache-coupled, preprocessing-coupled, or dead-branch-activating).
MODEL_PARAMETERS_SOURCE_NODE = "model_parameters"
MODEL_PARAMETERS_SAFE = frozenset(
    {
        "albedo_b",
        "ewall",
        "absK",
        "absL",
        "Fside",
        "Fup",
        "Fcyl",
        "cyl",
        "height",
        "transVeg",
        "firstdayleaf",
        "lastdayleaf",
        "elvis",
        "anisotropic_sky",
    }
)
MODEL_PARAMETERS_BLOCKED = frozenset(
    {
        "patch_option",
        "scale",
        "location",
        "utc",
        "walllimit",
        "onlyglobal",
        "albedo_g",
    }
)

#: Meteorology source node and the per-variable forcing fence (u-b-met,
#: 2026-09-02). The safe vocabulary is the registry entry's fixture
#: vocabulary (air_temperature, humidity, radiation, wind_speed_direction)
#: realized against the U-A audit's §(c) effective-set table: wind_direction
#: is fenced OUT as inert (the incremental path passes no wind-coefficient
#: rasters, solver.py:1090-1091/:1179), while pressure and uhii are added
#: because the audit tables them as physics-effective. Every other met
#: column is either unused at runtime or excluded below.
METEOROLOGY_SOURCE_NODE = "meteorology"
METEOROLOGY_VARIABLES_SAFE = frozenset(
    {
        "air_temperature",  # col 11 Td/Ta (utci_process.py:674)
        "humidity",  # col 10 RH (:675)
        "radiation",  # col 14 Kdn/radG (:676)
        "wind_speed",  # col 9 Wind/Ws (:680)
        "pressure",  # col 12 press/P (:679)
        "uhii",  # col 24 uhii (:688, Ta planes :874/:877)
    }
)
#: Specifically named hazards, each with its coupling reason: time columns
#: (met cols 0-3) drive the solar series pinned in the site cache, so a time
#: edit requires solar regeneration (the selected_date_time adapter's job);
#: Kdiff/Kdir are recomputed from radG while onlyglobal = 1 (solweig.py:
#: 2035-2039) and would be silent no-ops; wind_direction selects
#: wind-coefficient rasters the incremental path never loads.
METEOROLOGY_VARIABLES_BLOCKED = frozenset(
    {
        "year",  # col 0 iy
        "day_of_year",  # col 1 id
        "hour",  # col 2 it
        "minute",  # col 3 imin
        "diffuse_radiation",  # col 21 Kdiff (dead under onlyglobal = 1)
        "direct_radiation",  # col 22 Kdir (dead under onlyglobal = 1)
        "wind_direction",  # col 23 Wd (inert without wind coefficients)
    }
)

#: Wind source node: fenced behind scientific_extension adapters (UEDIT-010).
WIND_SOURCE_NODE = "wind_coefficients"

#: Meteorology per-variable solver affinity (r3a Step 0, 2026-09-04) — the
#: code-proven split behind the utci-only fast path. Every safe variable is
#: classified by tracing it through the ACTUAL solver code:
#:
#: * ``utci_only`` — enters ONLY the comfort/human calculation at time t
#:   (no radiation term, no shadow, no surface thermal state, no geometry).
#:   ``wind_speed`` feeds just the va10m plane (utci_process.py:693 read;
#:   :880-884 clamp(ones*Ws[i], 0.15) — the incremental paths pass no wind
#:   coefficients, :577-578); ``uhii`` offsets just the Ta comfort plane
#:   (:700-703 read; :887/:890; the radiation physics receives plain
#:   ``Ta[i]`` WITHOUT uhii at :838, and the wbgt/water/cylinder branches
#:   are dead or uhii-free, :705-711/:793/:800-801).
#: * ``radiation_affecting`` — touches at least one term feeding
#:   tmrt/radiation/surface temperature (solweig.py consumption sites
#:   cited per variable below), so it keeps today's full closure.
#:
#: A wrong split here is silently-wrong science; the evidence map below is
#: the audit trail and the drift guard pins the vocabulary.
MET_VARIABLE_AFFINITY: dict[str, str] = {
    "air_temperature": "radiation_affecting",
    "humidity": "radiation_affecting",
    "pressure": "radiation_affecting",
    "radiation": "radiation_affecting",
    "uhii": "utci_only",
    "wind_speed": "utci_only",
}
#: The provable utci-only subset (empty would mean NO cheap exact lever).
MET_UTCI_ONLY_VARIABLES = frozenset(
    {v for v, affinity in MET_VARIABLE_AFFINITY.items() if affinity == "utci_only"}
)
#: file:line evidence for every affinity claim (consumer sites, not prose).
MET_VARIABLE_AFFINITY_EVIDENCE: dict[str, str] = {
    "air_temperature": (
        "solweig.py:2042-2046 (EA/ESAT vapour pressure), :2051/:2056 (Lup), "
        ":2059 (Tgsubst/g), :2109-2117 (TgTemp), :2131-2133 (nocturnal "
        "Klowering), :2255-2274 (Lcyl); utci_process.py:791-793 (Twater), "
        ":800-801 (CI)"
    ),
    "humidity": (
        "solweig.py:2042-2046 (EA/ESAT), :2051/:2056 (Lup via vapour term), "
        ":2059 (Tgsubst)"
    ),
    "pressure": "solweig.py:869-877 (Trpg optical air mass), :2051/:2056 (Lup)",
    "radiation": (
        "solweig.py:2051/:2056 (Kup/Lup scaling), :2059 (Tgsubst), "
        ":2111-2117 (TgTemp), :2171-2179 (Klowering day-cycle)"
    ),
    "uhii": (
        "utci_process.py:700-703 (col 24 read), :887/:890 (Ta_mat = "
        "zeros + Ta[i] + uhii[i] — comfort plane ONLY; Solweig_2022a_calc "
        "receives plain Ta[i] at :838), :917 (utci_calculator), "
        ":933-935 (ta plane), :705-711 (wbgt branch dead, save_wbgt=False)"
    ),
    "wind_speed": (
        "utci_process.py:693 (Ws col 9 read), :577-578 (windcoeff=ones "
        "when no coefficient rasters), :880-884 (va10m_mat = "
        "clamp(windcoeff*Ws[i], 0.15)), :917 (utci_calculator), "
        ":936-938 (wind plane); no Ws/wind token in solweig.py physics"
    ),
}
def _validate_met_affinity(
    affinity: Mapping[str, str], evidence: Mapping[str, str]
) -> None:
    """Refuse an affinity table that drifted from the proven contract.

    Key drift (an unclassified or invented variable), VALUE drift (a
    typo like ``"utci-only"`` silently classifies the variable into NO
    bucket — the planner then keeps the conservative closure but the
    table is no longer the code-proven split it claims to be), and a
    missing evidence entry are all hard import-time errors: a wrong
    split here is silently-wrong science. Kept callable so the guards
    are directly testable, not just import-time noise.
    """
    if set(affinity) != set(METEOROLOGY_VARIABLES_SAFE):
        raise AdapterRegistryError(
            "MET_VARIABLE_AFFINITY drifted from METEOROLOGY_VARIABLES_SAFE "
            f"({sorted(set(affinity) ^ METEOROLOGY_VARIABLES_SAFE)}); "
            "classify every new safe variable before it becomes editable"
        )
    allowed = {"utci_only", "radiation_affecting"}
    unknown = {
        variable: value
        for variable, value in affinity.items()
        if value not in allowed
    }
    if unknown:
        raise AdapterRegistryError(
            f"unknown solver affinity value(s) {sorted(unknown.items())}; "
            "each meteorology affinity must be exactly 'utci_only' or "
            "'radiation_affecting' (a value typo classifies the variable "
            "into no bucket and would silently void the proof)"
        )
    if set(evidence) != set(affinity):
        raise AdapterRegistryError(
            "MET_VARIABLE_AFFINITY_EVIDENCE must carry file:line evidence "
            "for every safe meteorology variable"
        )


_validate_met_affinity(MET_VARIABLE_AFFINITY, MET_VARIABLE_AFFINITY_EVIDENCE)

#: Model-parameter solver affinity (r3b Step 0, 2026-09-04) — the
#: code-proven answer to "is any safe model parameter a comfort-only
#: lever?" (R0's ranking (b) hoped a receptor-only subset was). Every safe
#: parameter is ``radiation_affecting``: the human-body radiant load Sstr
#: consumes the person-geometry form factors (Fside/Fup/Fcyl/cyl branch
#: selection, solweig.py:2287-2296) and the optical coefficients
#: (absK/absL), and the published tmrt raster derives from Sstr
#: (Tmrt = (Sstr/(absL*SBC))**0.25 - 273.2, solweig.py:2299) — so a
#: parameter batch always moves tmrt and the r3a comfort-only recompute
#: shape is inadmissible. The remaining parameters reach radiation through
#: shadow reach (height), emissivity (elvis), the anisotropic diffuse
#: loops (anisotropic_sky), vegetation transmissivity/leaf cycle
#: (transVeg/firstdayleaf/lastdayleaf), or wall albedo/long-wave
#: (albedo_b/ewall). Evidence per parameter below; a drift guard refuses
#: an unclassified or typo-valued entry at import time.
MODEL_PARAMETERS_AFFINITY: dict[str, str] = {
    parameter: "radiation_affecting" for parameter in sorted(MODEL_PARAMETERS_SAFE)
}
#: file:line evidence for every affinity claim (consumer sites, not prose).
MODEL_PARAMETERS_AFFINITY_EVIDENCE: dict[str, str] = {
    "absK": "solweig.py:2287/:2292/:2296 (Sstr short-wave coefficient); read at utci_process.py:677",
    "absL": (
        "solweig.py:2287-2296 (Sstr long-wave coefficient) and :2299 "
        "(Tmrt = (Sstr/(absL*SBC))**0.25 - 273.2); read at utci_process.py:678"
    ),
    "albedo_b": (
        "solweig.py:213-215 (weightsumalbwall += tempbub * albedo_b — wall "
        "reflection weighting feeding the Kside terms Sstr consumes); "
        "utci_process.py:837 pass-through"
    ),
    "anisotropic_sky": (
        "solweig.py:684-688 (anisotropic diffuse loop gate); read at "
        "utci_process.py:671, passed at :840"
    ),
    "cyl": (
        "utci_process.py:682 (read), :838 (passed); solweig.py:2287/:2292/"
        ":2296 (selects the Sstr cylinder branch)"
    ),
    "elvis": (
        "solweig.py:2046 (esky = ... + elvis — sky emissivity additive term "
        "-> Ldown); read at utci_process.py:683, passed at :838"
    ),
    "Fcyl": "solweig.py:2287-2296 (Sstr Fcyl form factor); read at utci_process.py:681",
    "firstdayleaf": (
        "utci_process.py:713-724 (leaf_bool window -> leafon -> vegdem/psi)"
    ),
    "Fside": "solweig.py:2287-2296 (Sstr Fside form factor); read at utci_process.py:679",
    "Fup": "solweig.py:2287-2296 (Sstr Fup form factor); read at utci_process.py:680",
    "height": (
        "utci_process.py:729-735 (first/second wall-height walk reach), "
        ":817 (shadowingfunction_wallheight_23 call; imported :786) — "
        "moves time_shadow"
    ),
    "lastdayleaf": (
        "utci_process.py:713-724 (leaf_bool window -> leafon -> vegdem/psi)"
    ),
    "transVeg": (
        "utci_process.py:662 (read), :726 (psi = leafon * transVeg), "
        ":743 (svfbuveg), :747 (diffsh)"
    ),
    "ewall": (
        "solweig.py:132 (Lwall = SBC * ewall * (Tgwall + Ta + 273.15)**4 - "
        "... wall long-wave term); read at utci_process.py:676, passed at :837"
    ),
}


def _validate_model_parameter_affinity(
    affinity: Mapping[str, str], evidence: Mapping[str, str]
) -> None:
    """Refuse an affinity table that drifted from the proven contract.

    Same discipline as :func:`_validate_met_affinity`: key drift (a safe
    parameter added without classification, or an invented entry), value
    drift (a typo like ``"radiation-only"`` classifies the parameter into
    NO bucket), and missing evidence are hard import-time errors — a wrong
    split here would let a later packet plan a comfort-only recompute the
    physics refutes.
    """
    if set(affinity) != set(MODEL_PARAMETERS_SAFE):
        raise AdapterRegistryError(
            "MODEL_PARAMETERS_AFFINITY drifted from MODEL_PARAMETERS_SAFE "
            f"({sorted(set(affinity) ^ MODEL_PARAMETERS_SAFE)}); classify "
            "every new safe parameter before it becomes editable"
        )
    allowed = {"radiation_affecting"}
    unknown = {
        parameter: value
        for parameter, value in affinity.items()
        if value not in allowed
    }
    if unknown:
        raise AdapterRegistryError(
            f"unknown solver affinity value(s) {sorted(unknown.items())}; "
            "each model-parameter affinity must be exactly "
            "'radiation_affecting' (a value typo classifies the parameter "
            "into no bucket and would silently void the proof)"
        )
    if set(evidence) != set(affinity):
        raise AdapterRegistryError(
            "MODEL_PARAMETERS_AFFINITY_EVIDENCE must carry file:line "
            "evidence for every safe model parameter"
        )


_validate_model_parameter_affinity(
    MODEL_PARAMETERS_AFFINITY, MODEL_PARAMETERS_AFFINITY_EVIDENCE
)

#: View node: only view-only adapters may own it.
VIEW_SOURCE_NODE = "output_selection"


@dataclass(frozen=True, slots=True)
class AdapterMetadata:
    """One adapter's registry entry, mirroring ``edit_registry.yaml``.

    Extension fields beyond the YAML (engine-enforced rulings, all optional
    so YAML parity stays literal): ``rejected_properties`` — edit-state
    properties the adapter's schema must reject (e.g. vegetation
    transmissivity).
    """

    id: str
    group: str
    status: AdapterStatus
    source_nodes: tuple[str, ...]
    operations: tuple[str, ...]
    nominal_spatial_scope: str
    temporal_scope: str
    preview: str
    validation_fixtures: tuple[str, ...]
    notes: str = ""
    valid_classes: frozenset[int] = frozenset()
    parameters_safe: frozenset[str] = frozenset()
    parameters_blocked: frozenset[str] = frozenset()
    producible_incremental: tuple[str, ...] = ()
    rejected_properties: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name, value in (
            ("id", self.id),
            ("group", self.group),
            ("nominal_spatial_scope", self.nominal_spatial_scope),
            ("temporal_scope", self.temporal_scope),
            ("preview", self.preview),
        ):
            if not isinstance(value, str) or not value:
                raise AdapterRegistryError(f"{name} must be a non-empty string")
        if not isinstance(self.status, AdapterStatus):
            raise AdapterRegistryError("status must be an AdapterStatus")
        sources = tuple(self.source_nodes)
        operations = tuple(self.operations)
        fixtures = tuple(self.validation_fixtures)
        producible = tuple(self.producible_incremental)
        if not sources or not all(isinstance(item, str) and item for item in sources):
            raise AdapterRegistryError("source_nodes must be non-empty node ids")
        if not operations or not all(
            isinstance(item, str) and item for item in operations
        ):
            raise AdapterRegistryError("operations must be non-empty strings")
        if not all(isinstance(item, str) for item in fixtures + producible):
            raise AdapterRegistryError(
                "validation_fixtures/producible_incremental must be strings"
            )
        if not isinstance(self.notes, str):
            raise AdapterRegistryError("notes must be a string")
        classes = frozenset(self.valid_classes)
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in classes
        ):
            raise AdapterRegistryError("valid_classes must be non-negative integers")
        safe = frozenset(self.parameters_safe)
        blocked = frozenset(self.parameters_blocked)
        if safe & blocked:
            overlap = ", ".join(sorted(safe & blocked))
            raise AdapterRegistryError(
                f"parameters_safe and parameters_blocked overlap: {overlap}"
            )
        object.__setattr__(self, "source_nodes", sources)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "validation_fixtures", fixtures)
        object.__setattr__(self, "producible_incremental", producible)
        object.__setattr__(self, "valid_classes", classes)
        object.__setattr__(self, "parameters_safe", safe)
        object.__setattr__(self, "parameters_blocked", blocked)
        object.__setattr__(self, "rejected_properties", frozenset(self.rejected_properties))

    # -- derived facts -------------------------------------------------------

    @property
    def is_view_only(self) -> bool:
        return self.status is AdapterStatus.VIEW_ONLY

    @property
    def is_full_spatial(self) -> bool:
        """True when the nominal spatial strategy is whole-tile downstream."""
        return self.nominal_spatial_scope.startswith("full")

    def allows_operation(self, operation: str) -> bool:
        return operation in self.operations


# ---------------------------------------------------------------------------
# EditAdapter protocol (contract, verbatim)
# ---------------------------------------------------------------------------


@runtime_checkable
class EditAdapter(Protocol):
    """Protocol every edit adapter implements (edit-adapter contract).

    ``adapter_id``/``schema_version`` are plain attributes; the five
    methods convert a validated command into typed deltas and impact
    records. The registry also accepts metadata-only registrations (the
    builtin state) — :meth:`AdapterRegistry.get_adapter` raises until a
    real adapter object is registered for the id.
    """

    adapter_id: str
    schema_version: int

    def validate(self, command: EditCommand, context: SiteContext) -> ValidatedEdit: ...

    def preview_descriptor(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> PreviewDescriptor: ...

    def source_delta(self, edit: ValidatedEdit, context: SiteContext) -> SourceDelta: ...

    def impact_plan(self, edit: ValidatedEdit, context: SiteContext) -> ImpactPlan: ...

    def apply_source_delta(
        self, delta: SourceDelta, transaction: ScenarioTransaction
    ) -> None: ...

    def validation_fixtures(self) -> tuple[str, ...]: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AdapterRegistry:
    """Metadata + adapter store with ruling enforcement on registration.

    Enforcement summary (all raise :class:`AdapterRegistryError`):

    - duplicate adapter ids are rejected;
    - every ``source_nodes`` entry must resolve in the dependency graph
      (UEDIT-002);
    - land-cover adapters must declare ``valid_classes`` as a non-empty
      subset of ``{1, 2, 5, 6}`` and never include water (7);
    - vegetation adapters must reject the ``transmissivity`` property;
    - model-parameter adapters must block the seven coupled parameters and
      cover the safe set;
    - wind-coefficient adapters must be ``scientific_extension``;
    - ``output_selection`` adapters must be ``view_only``.
    """

    def __init__(self, graph: EditGraph | None = None) -> None:
        self._graph = graph if graph is not None else default_edit_graph()
        self._metadata: dict[str, AdapterMetadata] = {}
        self._adapters: dict[str, EditAdapter] = {}

    # -- registration --------------------------------------------------------

    def register_metadata(self, metadata: AdapterMetadata) -> AdapterMetadata:
        """Validate and store metadata without an adapter implementation."""
        self._validate_metadata(metadata)
        self._metadata[metadata.id] = metadata
        return metadata

    def register(
        self,
        adapter: EditAdapter,
        metadata: AdapterMetadata | None = None,
    ) -> AdapterMetadata:
        """Register an adapter implementation (and its metadata).

        ``metadata`` defaults to the adapter's ``metadata`` attribute when
        omitted. The adapter id must equal the metadata id; overwriting an
        existing *implementation* for the same id is allowed (hot swap for
        later packets) but a conflicting metadata redefinition is not.
        """
        if metadata is None:
            metadata = getattr(adapter, "metadata", None)
        if not isinstance(metadata, AdapterMetadata):
            raise AdapterRegistryError(
                "adapter registration requires AdapterMetadata "
                "(passed explicitly or as the adapter's 'metadata' attribute)"
            )
        adapter_id = getattr(adapter, "adapter_id", None)
        if adapter_id != metadata.id:
            raise AdapterRegistryError(
                f"adapter_id {adapter_id!r} does not match metadata id "
                f"{metadata.id!r}"
            )
        self._validate_metadata(metadata)
        self._metadata[metadata.id] = metadata
        self._adapters[metadata.id] = adapter
        return metadata

    def _validate_metadata(self, metadata: AdapterMetadata) -> None:
        existing = self._metadata.get(metadata.id)
        if existing is not None:
            if existing != metadata:
                raise AdapterRegistryError(
                    f"adapter id {metadata.id!r} is already registered with "
                    "different metadata; duplicate/conflicting ids are "
                    "rejected"
                )
            # Identical metadata: idempotent re-registration (adapter
            # implementation hot-swap in a later packet). Structural
            # checks still run below for defence in depth.
        for node_id in metadata.source_nodes:
            if not self._graph.contains(node_id):
                raise AdapterRegistryError(
                    f"adapter {metadata.id!r} references unknown dependency "
                    f"node {node_id!r} (UEDIT-002: adapters must resolve "
                    "valid nodes)"
                )
        sources = set(metadata.source_nodes)
        if LANDCOVER_SOURCE_NODE in sources:
            if not metadata.valid_classes:
                raise AdapterRegistryError(
                    "land-cover adapters must declare valid_classes "
                    f"(subset of {sorted(LANDCOVER_VALID_CLASSES)})"
                )
            fenced = metadata.valid_classes & LANDCOVER_FENCED_CLASSES
            if fenced:
                raise AdapterRegistryError(
                    "land-cover class 7 (water) is fenced in v1: the oracle "
                    "tests lc_grid == 3, never 7, so water semantics cannot "
                    "be delivered (lead ruling 2026-09-02)"
                )
            invalid = metadata.valid_classes - LANDCOVER_VALID_CLASSES
            if invalid:
                raise AdapterRegistryError(
                    "land-cover valid classes must be a subset of "
                    f"{sorted(LANDCOVER_VALID_CLASSES)}; got {sorted(invalid)}"
                )
        # The transmissivity rejection applies to adapters that expose a
        # user-facing vegetation property schema. scientific_extension
        # adapters (dynamic_wind_from_geometry also lists vegetation_dsm)
        # expose no object properties — they are fenced separately below.
        if (
            VEGETATION_SOURCE_NODE in sources
            and metadata.status is not AdapterStatus.SCIENTIFIC_EXTENSION
            and (
                metadata.rejected_properties & VEGETATION_REJECTED_PROPERTIES
            )
            != VEGETATION_REJECTED_PROPERTIES
        ):
            raise AdapterRegistryError(
                "vegetation adapters must reject the 'transmissivity' "
                "property in their edit-state schema: it is inert in the "
                "physics (global transVeg = 0.03, utci_process.py:649; "
                "lead ruling 2026-09-02)"
            )
        if MODEL_PARAMETERS_SOURCE_NODE in sources:
            missing_blocked = MODEL_PARAMETERS_BLOCKED - metadata.parameters_blocked
            if missing_blocked:
                raise AdapterRegistryError(
                    "model-parameter adapters must block "
                    f"{sorted(MODEL_PARAMETERS_BLOCKED)}; missing "
                    f"{sorted(missing_blocked)}"
                )
            missing_safe = MODEL_PARAMETERS_SAFE - metadata.parameters_safe
            if missing_safe:
                raise AdapterRegistryError(
                    "model-parameter adapters must accept the safe set "
                    f"{sorted(MODEL_PARAMETERS_SAFE)}; missing "
                    f"{sorted(missing_safe)}"
                )
        if WIND_SOURCE_NODE in sources and metadata.status is not (
            AdapterStatus.SCIENTIFIC_EXTENSION
        ):
            raise AdapterRegistryError(
                "adapters mutating wind_coefficients must be registered as "
                "scientific_extension: dynamic wind requires separate "
                "scientific validation (UEDIT-010; never silent)"
            )
        if VIEW_SOURCE_NODE in sources and metadata.status is not AdapterStatus.VIEW_ONLY:
            raise AdapterRegistryError(
                "adapters owning output_selection must be view_only "
                "(UEDIT-007: view operations enqueue zero scientific jobs)"
            )
        if metadata.status is AdapterStatus.VIEW_ONLY and VIEW_SOURCE_NODE not in sources:
            raise AdapterRegistryError(
                "view_only adapters must own output_selection as their source"
            )

    # -- lookup ---------------------------------------------------------------

    def get_metadata(self, adapter_id: str) -> AdapterMetadata:
        try:
            return self._metadata[adapter_id]
        except KeyError:
            raise AdapterRegistryError(
                f"unknown adapter {adapter_id!r} (UEDIT-001: every enabled "
                "editor must have a registered adapter)"
            ) from None

    def get_adapter(self, adapter_id: str) -> EditAdapter:
        self.get_metadata(adapter_id)
        try:
            return self._adapters[adapter_id]
        except KeyError:
            raise AdapterRegistryError(
                f"adapter {adapter_id!r} has registered metadata but no "
                "implementation yet"
            ) from None

    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._metadata))

    def all_metadata(self) -> tuple[AdapterMetadata, ...]:
        """All entries sorted by id (deterministic capability listing)."""
        return tuple(self._metadata[adapter_id] for adapter_id in self.adapter_ids())

    # -- validation services ---------------------------------------------------

    def validate_operation(self, adapter_id: str, operation: str) -> None:
        """Raise unless ``operation`` is declared by ``adapter_id``."""
        metadata = self.get_metadata(adapter_id)
        if not metadata.allows_operation(operation):
            raise AdapterRegistryError(
                f"adapter {adapter_id!r} does not support operation "
                f"{operation!r}; allowed: {list(metadata.operations)}"
            )

    def validate_edit_state(self, adapter_id: str, state: Mapping[str, Any]) -> None:
        """Check one edit state against the adapter's enforced schema.

        Enforced rules (conservative; adapters may add family-specific
        checks on top):

        - ``rejected_properties`` must not appear as keys, compared
          case-insensitively (``transmissivity``, ``Transmissivity``,
          ``TRANSMISSIVITY`` are all the fenced property — the adapter layer's
          closed vocabulary is the first line of defense; this is the
          engine-side backstop, u-b-reviewer NEW-3);
        - model-parameter states may only carry safe keys — blocked keys
          are rejected explicitly, unknown keys are rejected too (an
          unlisted parameter has no validated units/bounds);
        - land-cover states carry ``classes`` (a class code or sequence of
          codes); each must be in the adapter's ``valid_classes``, with
          the fenced water class called out by name;
        - meteorology states carry ``values`` (a mapping of forcing
          variable to value); blocked and unknown variables are rejected
          there, mirroring the delta-payload fence in ``validate_delta``
          (u-b-met, 2026-09-02).
        """
        metadata = self.get_metadata(adapter_id)
        frozen_state = freeze_mapping(state) or FrozenMapping(())
        rejected_lowercase = {
            prop.lower() for prop in metadata.rejected_properties
        }
        rejected = sorted(
            key
            for key in frozen_state
            if key.lower() in rejected_lowercase
        )
        if rejected:
            detail = ""
            if "transmissivity" in rejected_lowercase:
                detail = (
                    ": transmissivity is inert in the physics (global "
                    "transVeg = 0.03, utci_process.py:649; lead ruling "
                    "2026-09-02)"
                )
            raise AdapterSchemaError(
                f"adapter {adapter_id!r} rejects edit-state properties "
                f"{rejected}{detail}"
            )
        if MODEL_PARAMETERS_SOURCE_NODE in metadata.source_nodes:
            keys = set(frozen_state)
            blocked = sorted(keys & metadata.parameters_blocked)
            if blocked:
                raise AdapterSchemaError(
                    f"adapter {adapter_id!r} blocks parameters {blocked}: "
                    "they are cache-coupled, preprocessing-coupled, or "
                    "activate dead code paths (lead ruling 2026-09-02)"
                )
            unknown = sorted(keys - metadata.parameters_safe)
            if unknown:
                raise AdapterSchemaError(
                    f"adapter {adapter_id!r} does not define parameters "
                    f"{unknown}; only {sorted(metadata.parameters_safe)} "
                    "are editable"
                )
        if LANDCOVER_SOURCE_NODE in metadata.source_nodes:
            classes = frozen_state.get("classes")
            if classes is not None:
                codes = (
                    classes
                    if isinstance(classes, (tuple, list, frozenset, set))
                    else (classes,)
                )
                water = [code for code in codes if code in LANDCOVER_FENCED_CLASSES]
                if water:
                    raise AdapterSchemaError(
                        "land-cover class 7 (water) is fenced in v1 and "
                        "cannot be painted (lead ruling 2026-09-02)"
                    )
                invalid = sorted(
                    repr(code)
                    for code in codes
                    if not (
                        isinstance(code, int)
                        and not isinstance(code, bool)
                        and code in metadata.valid_classes
                    )
                )
                if invalid:
                    raise AdapterSchemaError(
                        f"land-cover classes {invalid} are not valid for "
                        f"adapter {adapter_id!r}; valid: "
                        f"{sorted(metadata.valid_classes)}"
                    )
        if METEOROLOGY_SOURCE_NODE in metadata.source_nodes:
            # Meteorology variable fence on the *command* states (the
            # delta-payload fence lives in validate_delta; u-b-met,
            # 2026-09-02): a forcing state's ``values`` mapping may only
            # carry the physics-effective vocabulary, so a bypass-built
            # command carrying blocked/unknown variables is rejected even
            # before an adapter turns it into a delta.
            values = frozen_state.get("values")
            if values is not None:
                if not isinstance(values, Mapping):
                    raise AdapterSchemaError(
                        f"adapter {adapter_id!r} state field 'values' must "
                        "be a mapping of forcing variable to value"
                    )
                keys = set(values)
                blocked = sorted(keys & METEOROLOGY_VARIABLES_BLOCKED)
                if blocked:
                    raise AdapterSchemaError(
                        f"adapter {adapter_id!r} blocks forcing variables "
                        f"{blocked}: time columns are pinned to the cached "
                        "solar series, Kdiff/Kdir are dead while onlyglobal "
                        "= 1, and wind_direction is inert without wind "
                        "coefficients (u-b-met, 2026-09-02)"
                    )
                unknown = sorted(keys - METEOROLOGY_VARIABLES_SAFE)
                if unknown:
                    raise AdapterSchemaError(
                        f"adapter {adapter_id!r} does not define forcing "
                        f"variables {unknown}; only "
                        f"{sorted(METEOROLOGY_VARIABLES_SAFE)} are editable"
                    )

    def validate_delta(self, adapter_id: str, delta: SourceDelta) -> None:
        """Check one typed source delta against the enforced schema.

        ``validate_edit_state`` guards the *command* states; this guards the
        *delta payload* the executor will actually consume — the review that
        introduced it demonstrated bypasses where fenced content rode
        ``ObjectStateChange.after`` / ``LandCoverPaintPatch.after_classes`` /
        ``ModelParameterChange.name`` into plans. The planner calls this for
        every edit in a batch, so no production path skips it.

        Enforced rules (conservative; unmapped delta families are checked
        generically where a ruling applies, skipped otherwise):

        - object-carrying deltas (vegetation, building): every object's
          ``before``/``after`` mappings must pass ``validate_edit_state``
          (rejected properties, safe/blocked parameters, paint classes);
        - land-cover paint deltas: every patch's ``after_classes`` (negative
          "unspecified" codes aside) must be paintable — water (7) is named
          out explicitly;
        - model-parameter deltas: every parameter name goes through the
          safe/blocked/unknown checks;
        - output-selection deltas: requested layers must be within the
          adapter's ``producible_incremental`` set (``wbgt`` is not in it
          and can never be selected).
        """
        metadata = self.get_metadata(adapter_id)
        if isinstance(delta, (VegetationObjectDelta, BuildingMassingDelta)):
            for change in delta.objects:
                if not isinstance(change, ObjectStateChange):
                    continue
                for role, state in (
                    ("before", change.before),
                    ("after", change.after),
                ):
                    if state is None:
                        continue
                    try:
                        self.validate_edit_state(adapter_id, dict(state))
                    except AdapterSchemaError as error:
                        raise AdapterSchemaError(
                            f"object {change.object_id!r} {role}-state: {error}"
                        ) from error
        elif isinstance(delta, LandCoverPaintDelta):
            for patch in delta.patches:
                painted = sorted(
                    {
                        code
                        for code in patch.after_classes
                        if isinstance(code, int)
                        and not isinstance(code, bool)
                        and code >= 0
                    }
                )
                if not painted:
                    continue
                water = [
                    code for code in painted if code in LANDCOVER_FENCED_CLASSES
                ]
                if water:
                    raise AdapterSchemaError(
                        "land-cover paint carries fenced class 7 (water): "
                        "the oracle never applies Twater to class 7 "
                        "(lc_grid == 3 dead branch), so water semantics "
                        "cannot be delivered (lead ruling 2026-09-02)"
                    )
                invalid = [
                    code
                    for code in painted
                    if code not in metadata.valid_classes
                ]
                if invalid:
                    raise AdapterSchemaError(
                        f"land-cover paint carries invalid classes {invalid}; "
                        f"valid: {sorted(metadata.valid_classes)}"
                    )
        elif isinstance(delta, ModelParameterDelta):
            for change in delta.parameters:
                self.validate_edit_state(adapter_id, {change.name: change.after_value})
        elif isinstance(delta, OutputSelectionDelta):
            if metadata.producible_incremental:
                unproducible = sorted(
                    layer
                    for layer in delta.after_layers
                    if layer not in metadata.producible_incremental
                )
                if unproducible:
                    raise AdapterSchemaError(
                        f"output layers {unproducible} are not producible "
                        f"incrementally; producible: "
                        f"{list(metadata.producible_incremental)}"
                    )
        elif isinstance(delta, ForcingDelta):
            # Per-variable fence for adapters owning the meteorology source
            # (u-b-met, 2026-09-02): the delta payload may only carry the
            # physics-effective, non-time-column vocabulary. Blocked names
            # are reported with their coupling reason (solar-cache-pinned
            # time columns, onlyglobal-dead Kdiff/Kdir, inert
            # wind_direction); anything else outside the safe vocabulary is
            # an undefined forcing variable. Other ForcingDelta families
            # (selected_date_time) declare no vocabulary yet; their fence
            # lands with the time adapter packet.
            if METEOROLOGY_SOURCE_NODE in metadata.source_nodes:
                variables = {change.variable for change in delta.changes}
                blocked = sorted(variables & METEOROLOGY_VARIABLES_BLOCKED)
                if blocked:
                    raise AdapterSchemaError(
                        f"adapter {adapter_id!r} blocks forcing variables "
                        f"{blocked}: time columns are pinned to the cached "
                        "solar series, Kdiff/Kdir are dead while onlyglobal "
                        "= 1, and wind_direction is inert without wind "
                        "coefficients (u-b-met, 2026-09-02)"
                    )
                unknown = sorted(variables - METEOROLOGY_VARIABLES_SAFE)
                if unknown:
                    raise AdapterSchemaError(
                        f"adapter {adapter_id!r} does not define forcing "
                        f"variables {unknown}; only "
                        f"{sorted(METEOROLOGY_VARIABLES_SAFE)} are editable"
                    )
            return


# ---------------------------------------------------------------------------
# Builtin metadata (mirrors edit_registry.yaml; drift-guard test enforces
# parity — do not edit one without the other)
# ---------------------------------------------------------------------------

_VEGETATION_NOTES = (
    "Per-tree transmissivity is validated but NOT consumed (global "
    "transVeg=0.03, utci_process.py:649) — transmissivity is EXCLUDED from the "
    "v1 property schema; inert until transVeg becomes per-tree/runtime (lead "
    "ruling 2026-09-02). trunk_ratio IS consumed (trees.py:152 → vegdem2 "
    "trunk zone = trunk_ratio * height_m). Spatial-coordinate disclosure "
    "(u-e2 L5): x_m/y_m outside the site extent PASS validation (the "
    "schema bounds geometry, not geography) and rasterize to zero cells — "
    "the edit commits with NO scene effect."
)

_DYNAMIC_WIND_NOTES = (
    "Parity gap: oracle applies WindCoeff rasters, incremental path assumes "
    "coeff=1 (solver.py:1238-1239, :1414-1415 pass windcoeff=None). Until a "
    "wind-coefficient adapter exists, the incremental system MUST detect "
    "wind-coefficient rasters and refuse/fall back — fenced, never silent "
    "(lead ruling 2026-09-02)."
)

BUILTIN_ADAPTER_METADATA: tuple[AdapterMetadata, ...] = (
    AdapterMetadata(
        id="vegetation_geometry",
        group="geometry",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("vegetation_dsm",),
        operations=("add", "move", "update", "delete"),
        nominal_spatial_scope="directional_windows",
        temporal_scope="replay_if_stateful",
        preview="vegetation_geometry_and_shadow",
        validation_fixtures=("isolated", "overlap", "low_sun", "boundary", "delete"),
        notes=_VEGETATION_NOTES,
        rejected_properties=frozenset({"transmissivity"}),
    ),
    AdapterMetadata(
        id="building_geometry",
        group="geometry",
        status=AdapterStatus.FULL_ONLY_INITIALLY,
        source_nodes=("building_dsm",),
        operations=("add", "move", "update", "delete", "raster_patch"),
        nominal_spatial_scope="conservative_windows_or_full",
        temporal_scope="replay",
        preview="massing_and_shadow",
        validation_fixtures=("height", "footprint", "wall_aspect", "low_sun", "boundary"),
    ),
    AdapterMetadata(
        id="terrain_dem",
        group="geometry",
        status=AdapterStatus.FULL_ONLY_INITIALLY,
        source_nodes=("dem",),
        operations=("raster_patch", "sculpt", "reset"),
        nominal_spatial_scope="full",
        temporal_scope="replay",
        preview="terrain_mesh",
        validation_fixtures=("grading", "relative_height", "boundary"),
    ),
    AdapterMetadata(
        id="landcover_surface",
        group="surface",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("landcover",),
        operations=("paint", "polygon_assign", "reset"),
        valid_classes=frozenset({1, 2, 5, 6}),
        nominal_spatial_scope="windows",
        temporal_scope="replay_if_stateful",
        preview="surface_material",
        validation_fixtures=("class_replace", "mixed_mask", "reset"),
    ),
    AdapterMetadata(
        id="meteorological_forcing",
        group="environment",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("meteorology",),
        operations=("update_time_row", "update_range", "preset", "reset"),
        nominal_spatial_scope="full_downstream_only",
        temporal_scope="changed_and_dependent_times",
        preview="environment_status",
        validation_fixtures=(
            "air_temperature",
            "humidity",
            "radiation",
            "wind_speed_direction",
        ),
    ),
    AdapterMetadata(
        id="selected_date_time",
        group="environment",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("selected_date_time",),
        operations=("select",),
        nominal_spatial_scope="full_downstream_only",
        temporal_scope="selected_or_replay",
        preview="solar_time_and_shadow",
        validation_fixtures=("morning", "noon", "low_sun", "date_change"),
    ),
    AdapterMetadata(
        id="model_receptor_parameters",
        group="advanced",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("model_parameters",),
        operations=("update", "reset"),
        parameters_safe=MODEL_PARAMETERS_SAFE,
        parameters_blocked=MODEL_PARAMETERS_BLOCKED,
        nominal_spatial_scope="full_downstream_stage_specific",
        temporal_scope="affected_times",
        preview="parameter_scope_disclosure",
        validation_fixtures=("baseline_restore", "lower_bound", "upper_bound"),
    ),
    AdapterMetadata(
        id="dynamic_wind_from_geometry",
        group="environment",
        status=AdapterStatus.SCIENTIFIC_EXTENSION,
        source_nodes=("building_dsm", "vegetation_dsm"),
        operations=("rebuild",),
        nominal_spatial_scope="full",
        temporal_scope="all",
        preview="unsupported_disclosure",
        validation_fixtures=(),
        notes=_DYNAMIC_WIND_NOTES,
    ),
    AdapterMetadata(
        id="output_view",
        group="analysis",
        status=AdapterStatus.VIEW_ONLY,
        source_nodes=("output_selection",),
        operations=("select_layer", "compare", "legend", "cached_time"),
        producible_incremental=(
            "utci",
            "tmrt",
            "shadow",
            "kup",
            "kdown",
            "lup",
            "ldown",
            "ta",
            "wind",
        ),
        nominal_spatial_scope="none",
        temporal_scope="none",
        preview="immediate",
        validation_fixtures=("zero_solver_job",),
    ),
)


def builtin_adapter_metadata() -> tuple[AdapterMetadata, ...]:
    """The builtin metadata entries (defensive copy of the tuple)."""
    return BUILTIN_ADAPTER_METADATA


def builtin_registry(graph: EditGraph | None = None) -> AdapterRegistry:
    """Fresh registry bootstrapped with the builtin metadata (no adapters)."""
    registry = AdapterRegistry(graph)
    for metadata in BUILTIN_ADAPTER_METADATA:
        registry.register_metadata(metadata)
    return registry
