# Edit-adapter contract

## Purpose

An edit adapter converts a user-facing operation into a validated scientific change and an invalidation plan. The central worker must not contain `if tree`, `if building`, `if landcover` chains scattered across the solver. Adapters register capabilities behind one protocol.

## Core data types

```python
@dataclass(frozen=True)
class EditCommand:
    edit_id: str
    scenario_id: str
    base_scene_revision: int
    adapter_id: str
    operation: str
    old_state: Mapping[str, Any] | None
    new_state: Mapping[str, Any] | None
    requested_outputs: tuple[str, ...]
    requested_times: tuple[int, ...] | None

@dataclass(frozen=True)
class NodeImpact:
    node_id: str
    spatial_scope: SpatialScope       # none | windows | full
    read_windows: tuple[RasterWindow, ...]
    write_windows: tuple[RasterWindow, ...]
    temporal_scope: TemporalScope     # one | range | replay | all
    time_start: int | None
    time_stop: int | None
    reason: str

@dataclass(frozen=True)
class ImpactPlan:
    changed_sources: tuple[str, ...]
    node_impacts: tuple[NodeImpact, ...]
    reusable_nodes: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    estimated_memory_bytes: int
    estimated_work_units: float
```

## Adapter protocol

```python
class EditAdapter(Protocol):
    adapter_id: str
    schema_version: int

    def validate(self, command: EditCommand, context: SiteContext) -> ValidatedEdit: ...

    def preview_descriptor(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> PreviewDescriptor: ...

    def source_delta(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> SourceDelta: ...

    def impact_plan(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ImpactPlan: ...

    def apply_source_delta(
        self, delta: SourceDelta, transaction: ScenarioTransaction
    ) -> None: ...

    def validation_fixtures(self) -> tuple[str, ...]: ...
```

## Required adapter metadata

Each adapter registry entry defines:

- stable adapter ID and schema version;
- source nodes it can mutate;
- allowed operations;
- JSON schema or equivalent validation;
- units, bounds, nodata rules, and coordinate semantics;
- preview behavior and explicit preview limitations;
- downstream dependency roots;
- spatial impact strategy;
- temporal replay strategy;
- cache nodes known reusable;
- full-fallback conditions;
- scientific fixtures and tolerances;
- current capability status.

## Adapter examples

### Vegetation object adapter

- user state: geometry, height, canopy/trunk representation, transmissivity;
- source delta: vegetation raster window plus metadata;
- impact: old and new geometry influence, directional shade, sky-view radius;
- exact stages: vegetation visibility through comfort;
- wind-field response: excluded unless a separate adapter/model is enabled.

### Building massing adapter

- user state: footprint and height;
- source delta: Building DSM patch;
- impact: old/new massing plus wall and visibility reach;
- exact stages: walls/aspect and all downstream geometry/radiation/comfort nodes;
- fallback: full tile until local wall/visibility equivalence passes validation.

### Land-cover brush adapter

- user state: polygon or raster mask and valid class code;
- source delta: local land-cover patch;
- impact: edited cells plus numerical halo;
- exact stages: surface/radiation/temporal state through comfort;
- visibility caches remain valid.

### Forcing adapter

- user state: audited met variables for one or more times;
- source delta: forcing rows or scenario forcing overlay;
- impact: generally full spatial scope downstream of forcing;
- expensive geometry/SVF caches remain valid;
- temporal replay begins at the earliest changed or state-dependent time.

## Registration

```python
registry.register(VegetationAdapter(...))
registry.register(BuildingMassingAdapter(...))
registry.register(LandCoverAdapter(...))
registry.register(MeteorologyAdapter(...))
```

The frontend retrieves capability metadata from the API. It should not hard-code the complete tool set.

## Versioning

An adapter schema change invalidates incompatible edit events and cached source overlays. Every result records adapter versions involved in its change set. Replaying old scenarios requires migration or explicit rejection.
