# SPDX-License-Identifier: GPL-3.0-only
"""Typed edit deltas and impact-plan records for the universal editing engine.

This module is the data-model half of the edit-adapter contract
(``docs/incremental_design_tool/universal_editing/edit_adapter_contract.md``).
The contract's core records — :class:`EditCommand`, :class:`NodeImpact`,
:class:`ImpactPlan` — are implemented *verbatim* (same field names, same
order, same types). Everything else extends the contract; nothing renames
it.

Design notes
------------

**Frozen, hashable, deterministic.** Every record is a frozen dataclass.
User-supplied ``Mapping`` payloads are normalized at construction time into
an immutable, hashable ``Mapping`` (:class:`FrozenMapping`) with recursively
frozen values (lists become tuples, nested mappings become frozen). Two
records built from equal inputs therefore compare equal, hash equal, and
reproduce equal — the property the deterministic-planner invariant
("identical batch, bitwise-identical plan") relies on.

**Typed source deltas.** Each editable source family carries a delta
subclass holding before/after payloads or window references, never raw
mutable arrays. Deltas are value objects: applying them is the adapter's
job (:meth:`EditAdapter.apply_source_delta`) against a
:class:`ScenarioTransaction`, so application is transactional — staged
deltas are invisible to the baseline until the transaction commits, and a
rollback discards them without touching any published state.

**Mixed-scope ruling (U-C, binding).** A plan carries *per-node* write-scope
records — exactly the ``node_impacts`` tuple, exposed in the ruling's
literal ``[(node_id, spatial_scope, write_windows)]`` shape by
:attr:`ImpactPlan.write_scope_records` — plus one transport bounding box
(:attr:`ImpactPlan.transport_window`, an extension field). A patch
containing any FULL node is full *for that node's products only*; coexisting
local nodes keep their windows. There is deliberately no single plan-level
spatial scope.

Conservative choices (contract ambiguities resolved the safe way):

- ``TemporalScope`` has no NONE member (the contract fixes the enum), so
  time-invariant geometry stages carry ``TemporalScope.ALL`` with
  ``time_start``/``time_stop`` unset — read as "no temporal restriction",
  never as "every timestep must be recomputed from scratch".
- Stateful stages (``surface_thermal_state``) always replay from timestep 0
  when dirty: accumulation history cannot be summarized safely.
- ``NodeImpact`` construction enforces that RANGE carries both bounds and
  ONE/REPLAY carry a start; missing temporal bounds are an error at record
  construction, not at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .geometry import RasterGrid, RasterWindow

__all__ = [
    "EditCommand",
    "EditStateError",
    "FrozenMapping",
    "ImpactPlan",
    "NodeImpact",
    "PreviewDescriptor",
    "ScenarioTransaction",
    "SiteContext",
    "SourceDelta",
    "SourceDeltaError",
    "SpatialScope",
    "TemporalScope",
    "TransactionError",
    "ValidatedEdit",
    "coalesce_source_deltas",
    "freeze_mapping",
    "identical_value_clause",
    # Typed delta families.
    "BuildingMassingDelta",
    "ForcingChange",
    "ForcingDelta",
    "LandCoverPaintDelta",
    "LandCoverPaintPatch",
    "ModelParameterChange",
    "ModelParameterDelta",
    "ObjectStateChange",
    "OutputSelectionDelta",
    "VegetationObjectDelta",
]


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


class SpatialScope(str, Enum):
    """Per-node spatial write scope (contract enum: none | windows | full)."""

    NONE = "none"
    WINDOWS = "windows"
    FULL = "full"


class TemporalScope(str, Enum):
    """Per-node temporal scope (contract enum: one | range | replay | all)."""

    ONE = "one"
    RANGE = "range"
    REPLAY = "replay"
    ALL = "all"


# ---------------------------------------------------------------------------
# Immutable, hashable mappings
# ---------------------------------------------------------------------------


class FrozenMapping(Mapping[str, Any]):
    """Immutable ``Mapping`` with structural hashing and a stable repr.

    Key order is normalized (sorted), so equal mappings hash and print
    identically regardless of construction order.
    """

    __slots__ = ("_items", "_map")

    def __init__(self, items: Sequence[tuple[str, Any]]) -> None:
        # Normalize in the constructor (U-C L3): deduplicate (last write
        # wins, matching ``dict`` semantics) and sort by key, so equal
        # mappings hash and compare equal regardless of construction order.
        normalized = dict(items)
        object.__setattr__(
            self, "_items", tuple(sorted(normalized.items(), key=lambda kv: kv[0]))
        )
        object.__setattr__(self, "_map", MappingProxyType(normalized))

    def __getitem__(self, key: str) -> Any:
        return self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMapping):
            return self._items == other._items
        if isinstance(other, Mapping):
            return dict(self._items) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        inner = ", ".join(f"{key!r}: {value!r}" for key, value in self._items)
        return "frozen({" + inner + "})"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    return value


def freeze_mapping(mapping: Mapping[str, Any] | None) -> FrozenMapping | None:
    """Return an immutable, hashable copy of ``mapping`` (or ``None``)."""
    if mapping is None:
        return None
    items: list[tuple[str, Any]] = []
    for key in sorted(mapping):
        if not isinstance(key, str):
            raise TypeError(f"mapping keys must be strings, got {key!r}")
        items.append((key, _freeze_value(mapping[key])))
    return FrozenMapping(items)


class EditStateError(ValueError):
    """A record failed structural validation (bad ids, bounds, or scopes).

    Also the typed refusal for an identical-value resubmit (U-D item d):
    a command whose declared ``old_state`` equals its ``new_state`` would
    churn dirty windows for a physically identical scene, so every adapter
    refuses it with a reason naming the property and both values via
    :func:`identical_value_clause`.
    """


class SourceDeltaError(ValueError):
    """A typed delta failed structural validation."""


def identical_value_clause(
    equals: Iterable[tuple[str, Any, Any]],
) -> str:
    """Format the ``name: before -> after`` list of an identical-value no-op.

    The shared reason-text builder for the U-D identical-value resubmit
    refusal: each item is ``(property, old_value, new_value)`` with
    ``old_value == new_value`` (that equality IS the refusal), so the
    message names the property and BOTH values without implying a change.
    """
    return ", ".join(
        f"{name}: {before!r} -> {after!r}" for name, before, after in equals
    )


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise EditStateError(f"{name} must be a non-empty string")


# ---------------------------------------------------------------------------
# Contract core records (verbatim field sets)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EditCommand:
    """One user-facing, versioned edit request (contract record, verbatim).

    ``old_state``/``new_state`` are normalized to frozen mappings;
    ``requested_outputs``/``requested_times`` to tuples. The record is
    hashable when payload values are hashable scalars/tuples/mappings.
    """

    edit_id: str
    scenario_id: str
    base_scene_revision: int
    adapter_id: str
    operation: str
    old_state: Mapping[str, Any] | None
    new_state: Mapping[str, Any] | None
    requested_outputs: tuple[str, ...]
    requested_times: tuple[int, ...] | None

    def __post_init__(self) -> None:
        for name, value in (
            ("edit_id", self.edit_id),
            ("scenario_id", self.scenario_id),
            ("adapter_id", self.adapter_id),
            ("operation", self.operation),
        ):
            _require_non_empty(name, value)
        if isinstance(self.base_scene_revision, bool) or not isinstance(
            self.base_scene_revision, int
        ):
            raise EditStateError("base_scene_revision must be an integer")
        if self.base_scene_revision < 0:
            raise EditStateError("base_scene_revision must be non-negative")
        outputs = tuple(self.requested_outputs)
        if not all(isinstance(item, str) and item for item in outputs):
            raise EditStateError("requested_outputs must be non-empty strings")
        if self.requested_times is None:
            times = None
        else:
            times = tuple(self.requested_times)
            if not all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and item >= 0
                for item in times
            ):
                raise EditStateError(
                    "requested_times must be non-negative integers or None"
                )
        object.__setattr__(self, "requested_outputs", outputs)
        object.__setattr__(self, "requested_times", times)
        object.__setattr__(self, "old_state", freeze_mapping(self.old_state))
        object.__setattr__(self, "new_state", freeze_mapping(self.new_state))


@dataclass(frozen=True, slots=True)
class NodeImpact:
    """Per-node impact record (contract record, verbatim).

    Write-scope semantics follow the U-C mixed-scope ruling: this record is
    the per-node write scope. ``spatial_scope=FULL`` means "this node's
    products are full tile" and must carry no windows; a coexisting local
    node keeps its own windows in the same plan.
    """

    node_id: str
    spatial_scope: SpatialScope
    read_windows: tuple[RasterWindow, ...]
    write_windows: tuple[RasterWindow, ...]
    temporal_scope: TemporalScope
    time_start: int | None
    time_stop: int | None
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty("node_id", self.node_id)
        if not isinstance(self.spatial_scope, SpatialScope):
            raise EditStateError("spatial_scope must be a SpatialScope")
        if not isinstance(self.temporal_scope, TemporalScope):
            raise EditStateError("temporal_scope must be a TemporalScope")
        if not isinstance(self.reason, str) or not self.reason:
            raise EditStateError("reason must be a non-empty string")
        read = tuple(self.read_windows)
        write = tuple(self.write_windows)
        if not all(isinstance(w, RasterWindow) for w in read + write):
            raise EditStateError("read/write windows must be RasterWindow values")
        if self.spatial_scope is SpatialScope.FULL and (read or write):
            raise EditStateError("a FULL node impact must not carry windows")
        if self.spatial_scope is SpatialScope.WINDOWS and not write:
            raise EditStateError("a WINDOWS node impact must carry write windows")
        if self.temporal_scope is TemporalScope.RANGE and (
            self.time_start is None or self.time_stop is None
        ):
            raise EditStateError("temporal RANGE requires time_start and time_stop")
        if self.temporal_scope in (TemporalScope.ONE, TemporalScope.REPLAY):
            if self.time_start is None:
                raise EditStateError(
                    f"temporal {self.temporal_scope.value} requires time_start"
                )
        for name, value in (
            ("time_start", self.time_start),
            ("time_stop", self.time_stop),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise EditStateError(f"{name} must be a non-negative integer or None")
        if (
            self.time_start is not None
            and self.time_stop is not None
            and self.time_stop < self.time_start
        ):
            raise EditStateError("time_stop must be >= time_start")
        object.__setattr__(self, "read_windows", read)
        object.__setattr__(self, "write_windows", write)


@dataclass(frozen=True, slots=True)
class ImpactPlan:
    """One topologically ordered execution plan (contract record, verbatim).

    Extension fields (U-C mixed-scope ruling; appended with defaults so the
    contract constructor stays source-compatible):

    - ``scene_revision``: the revision this plan was planned against —
      publication layers compare it before applying (UEDIT-006 shape).
    - ``transport_window``: single bounding box over every local write
      window, or ``None`` when the plan has no windowed writes.

    ``node_impacts`` is ordered by the dependency graph's deterministic
    topological order and doubles as the per-node write-scope record set.
    """

    changed_sources: tuple[str, ...]
    node_impacts: tuple[NodeImpact, ...]
    reusable_nodes: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    estimated_memory_bytes: int
    estimated_work_units: float
    # -- extensions (U-C mixed-scope ruling); defaults keep the contract
    #    constructor signature source-compatible.
    scene_revision: int | None = None
    transport_window: RasterWindow | None = None

    def __post_init__(self) -> None:
        sources = tuple(self.changed_sources)
        impacts = tuple(self.node_impacts)
        reusable = tuple(self.reusable_nodes)
        reasons = tuple(self.fallback_reasons)
        if not all(isinstance(item, str) and item for item in sources + reusable):
            raise EditStateError("changed_sources/reusable_nodes must be node ids")
        if not all(isinstance(item, NodeImpact) for item in impacts):
            raise EditStateError("node_impacts must contain NodeImpact records")
        if not all(isinstance(item, str) for item in reasons):
            raise EditStateError("fallback_reasons must be strings")
        if isinstance(self.estimated_memory_bytes, bool) or not isinstance(
            self.estimated_memory_bytes, int
        ):
            raise EditStateError("estimated_memory_bytes must be an integer")
        if not isinstance(self.estimated_work_units, (int, float)):
            raise EditStateError("estimated_work_units must be numeric")
        if self.estimated_memory_bytes < 0 or self.estimated_work_units < 0:
            raise EditStateError("estimates must be non-negative")
        if self.scene_revision is not None and (
            isinstance(self.scene_revision, bool)
            or not isinstance(self.scene_revision, int)
            or self.scene_revision < 0
        ):
            raise EditStateError("scene_revision must be a non-negative integer")
        if self.transport_window is not None and not isinstance(
            self.transport_window, RasterWindow
        ):
            raise EditStateError("transport_window must be a RasterWindow or None")
        object.__setattr__(self, "changed_sources", sources)
        object.__setattr__(self, "node_impacts", impacts)
        object.__setattr__(self, "reusable_nodes", reusable)
        object.__setattr__(self, "fallback_reasons", reasons)
        object.__setattr__(
            self, "estimated_work_units", float(self.estimated_work_units)
        )

    @property
    def write_scope_records(
        self,
    ) -> tuple[tuple[str, SpatialScope, tuple[RasterWindow, ...]], ...]:
        """U-C ruling literal shape: ``[(node_id, spatial_scope, write_windows)]``."""
        return tuple(
            (impact.node_id, impact.spatial_scope, impact.write_windows)
            for impact in self.node_impacts
        )


# ---------------------------------------------------------------------------
# Context, preview, validation, transaction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SiteContext:
    """Site identity and revision that adapters validate against.

    The contract names this type but does not define it; this is the
    engine's minimal definition (identity + grid + revision + time support).
    """

    site_id: str
    grid: RasterGrid
    scene_revision: int
    available_times: tuple[int, ...] = ()
    has_wind_coefficients: bool = False

    def __post_init__(self) -> None:
        _require_non_empty("site_id", self.site_id)
        if not isinstance(self.grid, RasterGrid):
            raise EditStateError("grid must be a RasterGrid")
        if isinstance(self.scene_revision, bool) or not isinstance(
            self.scene_revision, int
        ):
            raise EditStateError("scene_revision must be an integer")
        if self.scene_revision < 0:
            raise EditStateError("scene_revision must be non-negative")
        if not isinstance(self.has_wind_coefficients, bool):
            raise EditStateError("has_wind_coefficients must be a boolean")
        object.__setattr__(self, "available_times", tuple(self.available_times))

    # ``has_wind_coefficients`` is the wind-parity-gap fence carrier (lead
    # ruling 2026-09-02): the oracle applies WindCoeff rasters while the
    # incremental path assumes coeff=1 (solver.py:1090-1091, :1179). The
    # server/cache layer sets this from the site manifest; the planner
    # refuses to plan for such a site until a wind-coefficient adapter
    # passes independent validation (UEDIT-010; never silent).


@dataclass(frozen=True, slots=True)
class PreviewDescriptor:
    """Frontend preview behaviour declared by an adapter for one edit.

    ``kind`` mirrors the registry's ``preview`` vocabulary;
    ``limitations`` carries mandatory disclosures (e.g. fixed wind field).
    """

    kind: str
    immediate: bool
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("kind", self.kind)
        if not isinstance(self.immediate, bool):
            raise EditStateError("immediate must be a boolean")
        limits = tuple(self.limitations)
        if not all(isinstance(item, str) and item for item in limits):
            raise EditStateError("limitations must be non-empty strings")
        object.__setattr__(self, "limitations", limits)


@dataclass(frozen=True, slots=True)
class ValidatedEdit:
    """Output of ``EditAdapter.validate``: a command plus its typed delta.

    The planner consumes these; it never re-derives payloads from raw
    commands (single validation pass), but it *does* re-check registry
    facts (adapter known, operation allowed, source node owned) so an
    unknown adapter or operation can never reach planning.
    """

    command: EditCommand
    adapter_id: str
    schema_version: int
    source_node_id: str
    delta: SourceDelta

    def __post_init__(self) -> None:
        if not isinstance(self.command, EditCommand):
            raise EditStateError("command must be an EditCommand")
        if self.adapter_id != self.command.adapter_id:
            raise EditStateError("adapter_id must match command.adapter_id")
        _require_non_empty("adapter_id", self.adapter_id)
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise EditStateError("schema_version must be an integer")
        if self.schema_version < 1:
            raise EditStateError("schema_version must be >= 1")
        _require_non_empty("source_node_id", self.source_node_id)
        if not isinstance(self.delta, SourceDelta):
            raise EditStateError("delta must be a SourceDelta")
        if self.delta.source_node_id != self.source_node_id:
            raise EditStateError("delta.source_node_id must match source_node_id")
        if self.delta.adapter_id != self.adapter_id:
            raise EditStateError("delta.adapter_id must match adapter_id")


class TransactionError(RuntimeError):
    """A closed transaction was used, or commit/rollback was misused."""


class ScenarioTransaction:
    """Rollback-safe staging target for ``apply_source_delta``.

    Adapters *stage* typed deltas here; nothing touches the baseline until
    the caller commits, and ``rollback()`` discards everything staged. The
    later execution packets bind commit to atomic publication (scene
    revision bump + patch publish); this class owns only the staging
    discipline, so it stays free of I/O and wall-clock state.
    """

    __slots__ = ("_scenario_id", "_staged", "_state")

    def __init__(self, *, scenario_id: str) -> None:
        _require_non_empty("scenario_id", scenario_id)
        self._scenario_id = scenario_id
        self._staged: list[SourceDelta] = []
        self._state = "open"

    @property
    def scenario_id(self) -> str:
        return self._scenario_id

    @property
    def state(self) -> str:
        """``open``, ``committed``, or ``rolled_back``."""
        return self._state

    def staged_deltas(self) -> tuple[SourceDelta, ...]:
        """Staged deltas in arrival order (never re-ordered)."""
        return tuple(self._staged)

    def stage(self, delta: SourceDelta) -> None:
        if self._state != "open":
            raise TransactionError(f"transaction is {self._state}, not open")
        if not isinstance(delta, SourceDelta):
            raise EditStateError("only SourceDelta values can be staged")
        self._staged.append(delta)

    def commit(self) -> tuple[SourceDelta, ...]:
        """Close the transaction and return the staged deltas for publication."""
        if self._state != "open":
            raise TransactionError(f"transaction is {self._state}, not open")
        self._state = "committed"
        return tuple(self._staged)

    def rollback(self) -> None:
        """Discard staged deltas; the baseline is untouched by construction."""
        if self._state == "committed":
            raise TransactionError("a committed transaction cannot be rolled back")
        self._staged = []
        self._state = "rolled_back"


# ---------------------------------------------------------------------------
# Typed source deltas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceDelta:
    """Base class for typed, immutable, hashable source-change records.

    Subclasses carry before/after payloads (or window references) for one
    source family. ``spatial_windows`` reports the windows the *source*
    content changes touch (adapter-computed, conservative); forcing/parameter
    families report ``()`` because their spatial effect is downstream-only.
    """

    source_node_id: str
    adapter_id: str

    def __post_init__(self) -> None:
        _require_non_empty("source_node_id", self.source_node_id)
        _require_non_empty("adapter_id", self.adapter_id)

    @property
    def spatial_windows(self) -> tuple[RasterWindow, ...]:
        return ()

    @property
    def is_noop(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ObjectStateChange:
    """Before/after state of one editable object (tree, building, ...).

    ``before``/``after`` are frozen mappings of adapter-defined properties;
    ``None`` marks a not-yet-existing (add) or removed (delete) object.
    """

    object_id: str
    before: Mapping[str, Any] | None
    after: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        _require_non_empty("object_id", self.object_id)
        if self.before is None and self.after is None:
            raise SourceDeltaError("an object change needs a before or after state")
        object.__setattr__(self, "before", freeze_mapping(self.before))
        object.__setattr__(self, "after", freeze_mapping(self.after))


def _validate_object_delta_fields(
    objects: Sequence[ObjectStateChange], windows: Sequence[RasterWindow]
) -> None:
    if not all(isinstance(item, ObjectStateChange) for item in objects):
        raise SourceDeltaError("objects must be ObjectStateChange records")
    if not all(isinstance(item, RasterWindow) for item in windows):
        raise SourceDeltaError("windows must be RasterWindow values")


@dataclass(frozen=True, slots=True)
class VegetationObjectDelta(SourceDelta):
    """Add/move/update/delete of vegetation objects (per-tree model).

    ``windows`` are the adapter-computed conservative influence windows
    (old union new geometry), e.g. from
    :func:`solweig_gpu.incremental.geometry.dirty_window_for_edit`.
    """

    objects: tuple[ObjectStateChange, ...] = ()
    windows: tuple[RasterWindow, ...] = ()

    def __post_init__(self) -> None:
        SourceDelta.__post_init__(self)
        _validate_object_delta_fields(self.objects, self.windows)
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "windows", tuple(self.windows))

    @property
    def is_noop(self) -> bool:
        return not self.objects

    @property
    def spatial_windows(self) -> tuple[RasterWindow, ...]:
        return self.windows


@dataclass(frozen=True, slots=True)
class BuildingMassingDelta(SourceDelta):
    """Footprint/height massing change for buildings (Building DSM patch)."""

    objects: tuple[ObjectStateChange, ...] = ()
    windows: tuple[RasterWindow, ...] = ()

    def __post_init__(self) -> None:
        SourceDelta.__post_init__(self)
        _validate_object_delta_fields(self.objects, self.windows)
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "windows", tuple(self.windows))

    @property
    def is_noop(self) -> bool:
        return not self.objects

    @property
    def spatial_windows(self) -> tuple[RasterWindow, ...]:
        return self.windows


@dataclass(frozen=True, slots=True)
class LandCoverPaintPatch:
    """One paint stroke: class codes before/after inside a window.

    ``before_classes``/``after_classes`` are row-major flattened tuples of
    length ``window.area``. ``-1`` marks cells outside the actual mask when
    a caller models partial coverage; the engine treats any negative code
    as "unchanged/unspecified" and never paints it.
    """

    window: RasterWindow
    before_classes: tuple[int, ...]
    after_classes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.window, RasterWindow):
            raise SourceDeltaError("window must be a RasterWindow")
        before = tuple(self.before_classes)
        after = tuple(self.after_classes)
        for name, codes in (("before_classes", before), ("after_classes", after)):
            if not all(
                isinstance(code, int) and not isinstance(code, bool) for code in codes
            ):
                raise SourceDeltaError(f"{name} must contain integers")
        area = self.window.area
        if len(before) != area or len(after) != area:
            raise SourceDeltaError(
                "class codes must be row-major flattened with length window.area "
                f"({area}); got before={len(before)}, after={len(after)})"
            )
        object.__setattr__(self, "before_classes", before)
        object.__setattr__(self, "after_classes", after)


@dataclass(frozen=True, slots=True)
class LandCoverPaintDelta(SourceDelta):
    """Land-cover class painting: ordered patches, last-write-wins per cell.

    Coalescing concatenates patches in arrival order (the pipeline's
    ordered last-write-wins semantics); ``spatial_windows`` are the patch
    windows in order.
    """

    patches: tuple[LandCoverPaintPatch, ...] = ()

    def __post_init__(self) -> None:
        SourceDelta.__post_init__(self)
        if not all(isinstance(item, LandCoverPaintPatch) for item in self.patches):
            raise SourceDeltaError("patches must be LandCoverPaintPatch records")
        object.__setattr__(self, "patches", tuple(self.patches))

    @property
    def is_noop(self) -> bool:
        return not self.patches

    @property
    def spatial_windows(self) -> tuple[RasterWindow, ...]:
        return tuple(patch.window for patch in self.patches)


@dataclass(frozen=True, slots=True)
class ForcingChange:
    """One forcing variable's before/after value at one time index.

    ``time_index=None`` marks a whole-series change (reset/preset); the
    value may be a scalar or a mapping (structured rows — frozen on init).
    """

    variable: str
    time_index: int | None
    before_value: Any
    after_value: Any

    def __post_init__(self) -> None:
        _require_non_empty("variable", self.variable)
        if self.time_index is not None and (
            isinstance(self.time_index, bool)
            or not isinstance(self.time_index, int)
            or self.time_index < 0
        ):
            raise SourceDeltaError(
                "time_index must be a non-negative integer or None"
            )
        object.__setattr__(self, "before_value", _freeze_value(self.before_value))
        object.__setattr__(self, "after_value", _freeze_value(self.after_value))


@dataclass(frozen=True, slots=True)
class ForcingDelta(SourceDelta):
    """Meteorological forcing change (per variable/time, last-write-wins)."""

    changes: tuple[ForcingChange, ...] = ()

    def __post_init__(self) -> None:
        SourceDelta.__post_init__(self)
        if not all(isinstance(item, ForcingChange) for item in self.changes):
            raise SourceDeltaError("changes must be ForcingChange records")
        object.__setattr__(self, "changes", tuple(self.changes))

    @property
    def is_noop(self) -> bool:
        return not self.changes

    @property
    def time_indices(self) -> tuple[int, ...]:
        """Explicit changed time indices, sorted unique (empty = whole series)."""
        return tuple(
            sorted(
                {
                    change.time_index
                    for change in self.changes
                    if change.time_index is not None
                }
            )
        )


@dataclass(frozen=True, slots=True)
class ModelParameterChange:
    """One model/receptor parameter's before/after value (frozen on init)."""

    name: str
    before_value: Any
    after_value: Any

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        object.__setattr__(self, "before_value", _freeze_value(self.before_value))
        object.__setattr__(self, "after_value", _freeze_value(self.after_value))


@dataclass(frozen=True, slots=True)
class ModelParameterDelta(SourceDelta):
    """Model/receptor parameter change (registry-validated safe set only)."""

    parameters: tuple[ModelParameterChange, ...] = ()

    def __post_init__(self) -> None:
        SourceDelta.__post_init__(self)
        if not all(
            isinstance(item, ModelParameterChange) for item in self.parameters
        ):
            raise SourceDeltaError("parameters must be ModelParameterChange records")
        object.__setattr__(self, "parameters", tuple(self.parameters))

    @property
    def is_noop(self) -> bool:
        return not self.parameters


@dataclass(frozen=True, slots=True)
class OutputSelectionDelta(SourceDelta):
    """View-layer selection change: before/after requested output layers."""

    before_layers: tuple[str, ...] = ()
    after_layers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        SourceDelta.__post_init__(self)
        for name, layers in (
            ("before_layers", self.before_layers),
            ("after_layers", self.after_layers),
        ):
            frozen_layers = tuple(layers)
            if not all(isinstance(layer, str) and layer for layer in frozen_layers):
                raise SourceDeltaError(f"{name} must be non-empty strings")
            object.__setattr__(self, name, frozen_layers)

    @property
    def is_noop(self) -> bool:
        return self.before_layers == self.after_layers


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------


def _coalesce_object_changes(
    changes: Iterable[ObjectStateChange],
) -> tuple[ObjectStateChange, ...]:
    """First before-state, final after-state, per object; no-ops dropped.

    Implements the pipeline coalescing rules: add -> move -> update keeps
    the first old and final new state; add then delete cancels; a change
    whose final state equals the first old state is a no-op and disappears.
    """
    # Track (first_before, last_after) pairs as plain values: an add+delete
    # chain produces a (None, None) intermediate that must drop out without
    # ever forming an invalid ObjectStateChange record.
    merged: dict[str, tuple[Any, Any]] = {}
    for change in changes:
        previous = merged.get(change.object_id)
        if previous is not None and previous[1] != change.before:
            raise SourceDeltaError(
                f"non-contiguous change chain for object {change.object_id!r}: "
                "before-state does not match the preceding after-state"
            )
        first_before = change.before if previous is None else previous[0]
        merged[change.object_id] = (first_before, change.after)
    kept = [
        ObjectStateChange(object_id, first_before, last_after)
        for object_id, (first_before, last_after) in sorted(merged.items())
        if not (first_before is None and last_after is None)
        and first_before != last_after
    ]
    return tuple(kept)


def _coalesce_forcing(changes: Iterable[ForcingChange]) -> tuple[ForcingChange, ...]:
    """First before / last after value per (variable, time_index)."""
    merged: dict[tuple[str, int | None], ForcingChange] = {}
    for change in changes:
        key = (change.variable, change.time_index)
        previous = merged.get(key)
        first_before = (
            change.before_value if previous is None else previous.before_value
        )
        merged[key] = ForcingChange(
            change.variable, change.time_index, first_before, change.after_value
        )
    kept = [
        change
        for change in merged.values()
        if change.before_value != change.after_value
    ]
    return tuple(sorted(kept, key=lambda change: (change.variable, change.time_index)))


def _coalesce_parameters(
    changes: Iterable[ModelParameterChange],
) -> tuple[ModelParameterChange, ...]:
    """First before / last after value per parameter name."""
    merged: dict[str, ModelParameterChange] = {}
    for change in changes:
        previous = merged.get(change.name)
        first_before = (
            change.before_value if previous is None else previous.before_value
        )
        merged[change.name] = ModelParameterChange(
            change.name, first_before, change.after_value
        )
    kept = [
        change
        for change in merged.values()
        if change.before_value != change.after_value
    ]
    return tuple(sorted(kept, key=lambda change: change.name))


def coalesce_source_deltas(deltas: Sequence[SourceDelta]) -> SourceDelta:
    """Coalesce deltas of one family/node/adapter in arrival order.

    Rules (pipeline coalescing section): object deltas keep each object's
    first before-state and final after-state (add+delete cancels, identity
    changes drop); keyed deltas (forcing, parameters) keep the first before
    and last after value per key; paint deltas concatenate patches
    (last-write-wins at apply time); output selection keeps first before /
    last after layers. The returned delta is always of the input family.
    """
    if not deltas:
        raise SourceDeltaError("deltas must be non-empty")
    first = deltas[0]
    for delta in deltas:
        if type(delta) is not type(first):
            raise SourceDeltaError(
                "cannot coalesce deltas of different families: "
                f"{type(first).__name__} vs {type(delta).__name__}"
            )
        if delta.source_node_id != first.source_node_id:
            raise SourceDeltaError(
                "cannot coalesce deltas for different source nodes: "
                f"{first.source_node_id!r} vs {delta.source_node_id!r}"
            )
        if delta.adapter_id != first.adapter_id:
            raise SourceDeltaError(
                "cannot coalesce deltas from different adapters: "
                f"{first.adapter_id!r} vs {delta.adapter_id!r}"
            )

    if isinstance(first, (VegetationObjectDelta, BuildingMassingDelta)):
        objects: list[ObjectStateChange] = []
        for delta in deltas:
            objects.extend(delta.objects)
        coalesced_objects = _coalesce_object_changes(objects)
        # U-C L5: windows of input deltas whose object ids ALL canceled in
        # the coalesced chain (add+delete) no longer contribute dirty
        # windows. Attribution is only safe per whole delta: a delta with
        # mixed surviving/canceled objects keeps all of its windows
        # (conservative-only), as does a delta with no objects.
        surviving_ids = {change.object_id for change in coalesced_objects}
        windows: list[RasterWindow] = []
        for delta in deltas:
            delta_ids = {change.object_id for change in delta.objects}
            if delta_ids and delta_ids.isdisjoint(surviving_ids):
                continue  # every object in this delta canceled
            windows.extend(delta.windows)
        return type(first)(
            source_node_id=first.source_node_id,
            adapter_id=first.adapter_id,
            objects=coalesced_objects,
            windows=tuple(windows),
        )
    if isinstance(first, LandCoverPaintDelta):
        patches = tuple(patch for delta in deltas for patch in delta.patches)
        return LandCoverPaintDelta(
            source_node_id=first.source_node_id,
            adapter_id=first.adapter_id,
            patches=patches,
        )
    if isinstance(first, ForcingDelta):
        changes: list[ForcingChange] = []
        for delta in deltas:
            changes.extend(delta.changes)
        return ForcingDelta(
            source_node_id=first.source_node_id,
            adapter_id=first.adapter_id,
            changes=_coalesce_forcing(changes),
        )
    if isinstance(first, ModelParameterDelta):
        parameters: list[ModelParameterChange] = []
        for delta in deltas:
            parameters.extend(delta.parameters)
        return ModelParameterDelta(
            source_node_id=first.source_node_id,
            adapter_id=first.adapter_id,
            parameters=_coalesce_parameters(parameters),
        )
    if isinstance(first, OutputSelectionDelta):
        return OutputSelectionDelta(
            source_node_id=first.source_node_id,
            adapter_id=first.adapter_id,
            before_layers=deltas[0].before_layers,
            after_layers=deltas[-1].after_layers,
        )
    raise SourceDeltaError(
        f"no coalescing rule for delta type {type(first).__name__}"
    )
