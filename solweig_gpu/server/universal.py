# SPDX-License-Identifier: GPL-3.0-only
"""Universal edit transport: HTTP family edits onto the executor (u-d4).

Four integrated edit families (``meteorological_forcing``,
``landcover_surface``, ``building_geometry``, ``model_receptor_parameters``)
had engine support end-to-end — adapter, planner, executor — but no HTTP
transport: the tree ``POST /edits`` contract cannot carry them, so the
frontend's ``composeFamilyEdit`` could only raise ``FamilyTransportError``.
This module is the missing wiring, in two halves:

**Mapping (this module).** A :class:`~solweig_gpu.server.models.UniversalEditItem`
is shaped into an :class:`~solweig_gpu.incremental.edit_types.EditCommand` —
per-family state assembly ONLY. The mapping moves fields around; it never
decides validity. Every domain, fence, and schema check runs inside the
adapter registry and its adapters (``registry.validate_edit_state``,
``adapter.validate``), which this module calls and whose refusals it
relays verbatim as typed 4xx envelopes. Nothing about a family's science
is restated here.

**Dispatch (``executor_bridge``).** Scenarios that carry family state —
any surviving family event in the ledger OR the store's durable
``carries_family_edits`` flag (u-e1 F1) — route onto the
:class:`PlanExecutor` path; tree-only scenarios keep the legacy
ExactWorker solver unchanged.

Vegetation stays on the tree transport (``tree-edits-v1``) BY DESIGN and is
refused here with ``unsupported_transport``: the universal vegetation
grammar cannot carry ``transmissivity`` (the adapter rejects the key), and
tree authority (``scenario_trees``, quotas, reset semantics) lives in the
tree contract. ``output_view`` items are answered view-only through the
same published-result machinery as ``POST /views`` — zero jobs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from solweig_gpu.incremental.capabilities import full_registry
from solweig_gpu.incremental.edit_registry import AdapterRegistryError, AdapterSchemaError
from solweig_gpu.incremental.edit_types import EditCommand, EditStateError, SiteContext
from solweig_gpu.incremental.executor import detect_wind_coefficients
from solweig_gpu.incremental.geometry import RasterGrid
from solweig_gpu.server.jobs import SiteRegistry
from solweig_gpu.server.models import ApiError, UniversalEditItem

#: Adapter ids with a dedicated non-universal transport.
TREE_ONLY_ADAPTERS = frozenset({"vegetation_geometry"})

#: The view-only family id (never a job; answered from published results).
VIEW_ADAPTER_ID = "output_view"

#: Fallback command scope mirrors the API's default result variables.
DEFAULT_REQUESTED_OUTPUTS = ("utci", "tmrt")


def json_time_index(value: Any, *, field: str) -> int:
    """Strict JSON-integer parse for a values-level ``time_index`` (u-e2 M2).

    The transport previously ran ``int(values.pop("time_index"))`` BEFORE
    the met adapter's timestep fence saw the original value: ``5.9``
    silently became an edit at timestep 5, ``True`` became timestep 1,
    ``"07"`` became timestep 7. A wrong-timestep edit is wrong science,
    so anything that is not a JSON integer is refused typed — the adapter
    validator (its own fence) stays the second line of defense.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(
            "invalid_request",
            f"{field} must be a JSON integer timestep index, got "
            f"{type(value).__name__} {value!r}; the transport refuses to "
            "coerce floats, booleans, strings, or lists onto a timestep",
            field=field,
        )
    return int(value)


# ---------------------------------------------------------------------------
# Registry + site context (validation is registry/adapter-owned)
# ---------------------------------------------------------------------------


_ADAPTER_REGISTRY = None


def family_registry_adapter(adapter_id: str):
    """Resolve one adapter through the process-wide full registry.

    ``full_registry`` binds exactly the adapters the executor can execute
    end-to-end; an id it cannot resolve is genuinely not integrated, which
    is precisely what ``unsupported_transport`` reports. The registry is
    process-global (adapters are stateless validators), so one build is
    shared by the route and the job replay path.
    """
    global _ADAPTER_REGISTRY
    if _ADAPTER_REGISTRY is None:
        _ADAPTER_REGISTRY = full_registry()
    return _ADAPTER_REGISTRY.get_adapter(adapter_id)


def site_context(
    app_state,
    sites: SiteRegistry,
    site_id: str,
    scene_revision: int,
) -> SiteContext:
    """The adapters' :class:`SiteContext` for HTTP-side validation.

    Grid and times come from the site registry; the wind-coefficient fence
    fact comes from the executor's own ``detect_wind_coefficients``
    (stat()-based, cached per site so per-request validation stays cheap).
    """
    geometry = sites.geometry(site_id)
    config = sites.config(site_id)
    tile_key = str(sites.cache(site_id).tile_key)
    cache: dict[tuple[str, str], bool] = getattr(app_state, "universal_wind_cache", None)
    if cache is None:
        cache = {}
        app_state.universal_wind_cache = cache
    key = (site_id, tile_key)
    if key not in cache:
        cache[key] = detect_wind_coefficients(
            config.site_dir or config.cache_dir, tile_key
        )
    return SiteContext(
        site_id=site_id,
        grid=RasterGrid(
            rows=int(geometry["rows"]),
            cols=int(geometry["cols"]),
            pixel_size_m=float(geometry["pixel_size_m"]),
            origin_x_m=float(geometry["origin_x_m"]),
            origin_y_m=float(geometry["origin_y_m"]),
        ),
        scene_revision=int(scene_revision),
        available_times=tuple(range(int(geometry["time_steps"]))),
        has_wind_coefficients=cache[key],
    )


# ---------------------------------------------------------------------------
# Item -> EditCommand mapping (transport wiring, NOT validation)
# ---------------------------------------------------------------------------


def _require_window(item: UniversalEditItem, index: int) -> dict[str, Any]:
    target = item.target
    if not isinstance(target, Mapping):
        raise ApiError(
            "invalid_request",
            "a land-cover paint needs a window target "
            "({row_start, row_stop, col_start, col_stop})",
            field=f"edits[{index}].target",
        )
    return dict(target)


def _building_id(item: UniversalEditItem, index: int) -> str:
    if not isinstance(item.target, str) or not item.target:
        raise ApiError(
            "invalid_request",
            "a building edit needs the building id as its target",
            field=f"edits[{index}].target",
        )
    return item.target


def command_from_item(
    item: UniversalEditItem,
    *,
    scenario_id: str,
    scene_revision: int,
    edit_id: str,
    index: int = 0,
    requested_outputs: Sequence[str] = DEFAULT_REQUESTED_OUTPUTS,
    requested_times: tuple[int, ...] | None = None,
) -> EditCommand | None:
    """Assemble one universal item into an :class:`EditCommand`.

    Returns ``None`` for ``output_view`` items (they are answered by the
    view machinery, never executed). Raises :class:`ApiError` only for
    transport-shape problems (a missing window/id target) — everything
    else is decided by the adapter validators when
    :func:`validate_command` (or the executor) runs.
    """
    common = dict(
        edit_id=edit_id,
        scenario_id=scenario_id,
        base_scene_revision=int(scene_revision),
        adapter_id=item.adapter,
        requested_outputs=tuple(requested_outputs),
        requested_times=requested_times,
    )
    if item.adapter == "meteorological_forcing":
        values = dict(item.values)
        index_field = f"edits[{index}].values.time_index"
        time_index = item.time_index
        if "time_index" in values:
            # u-e2 M2/M3: strict parse, and never two competing sources.
            if time_index is not None:
                raise ApiError(
                    "invalid_request",
                    "time_index appears both at the item level and inside "
                    "values; which timestep the edit lands on would be "
                    "ambiguous — send exactly one",
                    field=index_field,
                    details={"adapter": item.adapter, "operation": item.operation},
                )
            time_index = json_time_index(
                values.pop("time_index"), field=index_field
            )
        if "time_start" in values and "time_stop" in values:
            if time_index is not None:
                # u-e2 M3: the range grammar has no row slot — a time_index
                # riding along was silently dropped at base.
                raise ApiError(
                    "invalid_request",
                    "a range edit (time_start/time_stop) cannot also carry "
                    "time_index; the range grammar would silently drop the "
                    "row edit — send a range or a single row, not both",
                    field=index_field,
                    details={"adapter": item.adapter, "operation": item.operation},
                )
            new_state: dict[str, Any] = {
                "time_start": values.pop("time_start"),
                "time_stop": values.pop("time_stop"),
                "values": values.pop("values", values),
            }
        else:
            new_state = {"values": values}
            if time_index is not None:
                new_state["time_index"] = time_index
        return EditCommand(
            operation=item.operation,
            old_state=None,
            new_state=new_state,
            **common,
        )
    if item.adapter == "landcover_surface":
        if item.operation == "paint":
            window = _require_window(item, index)
            unknown = sorted(set(item.values) - {"class"})
            if unknown:
                # u-e2 M3: the paint grammar is exactly values={"class"}.
                # At base any other key was silently dropped — a typo'd
                # class field painted the DEFAULT class instead of
                # refusing.
                raise ApiError(
                    "invalid_request",
                    f"landcover paint values carry unknown key(s) {unknown}; "
                    "the paint grammar is exactly values={'class': <code>} "
                    "(the window travels as the target)",
                    field=f"edits[{index}].values",
                    details={"adapter": item.adapter, "operation": item.operation},
                )
            return EditCommand(
                operation="paint",
                old_state=None,
                new_state={"window": window, "classes": item.values.get("class")},
                **common,
            )
        return EditCommand(
            operation=item.operation,
            old_state=None,
            new_state=dict(item.values),
            **common,
        )
    if item.adapter == "building_geometry":
        building_id = _building_id(item, index)
        # u-e2 M3: the target is authoritative. values/old_values could
        # silently re-aim the edit onto another building's state payload.
        for source, source_field in (
            (item.values, "values"),
            (item.old_values, "old_values"),
        ):
            if source and source.get("building_id") not in (None, building_id):
                raise ApiError(
                    "invalid_request",
                    f"building_id {source['building_id']!r} inside "
                    f"{source_field} disagrees with the edit target "
                    f"{building_id!r}; the target is authoritative — "
                    "values cannot re-aim the edit at another building",
                    field=f"edits[{index}].{source_field}",
                    details={"adapter": item.adapter, "operation": item.operation},
                )
        old_state = (
            {"building_id": building_id, **dict(item.old_values)}
            if item.old_values is not None
            else None
        )
        if item.operation == "delete":
            return EditCommand(
                operation="delete", old_state=old_state, new_state=None, **common
            )
        return EditCommand(
            operation=item.operation,
            old_state=old_state,
            new_state={"building_id": building_id, **dict(item.values)},
            **common,
        )
    if item.adapter == "model_receptor_parameters":
        if item.operation == "reset":
            return EditCommand(
                operation="reset",
                old_state=dict(item.values) if item.values else None,
                new_state=None,
                **common,
            )
        # u-e2 M3: relay the operation VERBATIM. At base any non-reset
        # op was silently rewritten to "update", so a typo ("updaet")
        # or an undeclared op never surfaced — the registry now refuses
        # it typed (validate_command relays AdapterRegistryError).
        return EditCommand(
            operation=item.operation,
            old_state=None,
            new_state=dict(item.values),
            **common,
        )
    # Every other family shares the pass-through state grammar; its
    # adapter's validators accept or refuse the payload.
    return EditCommand(
        operation=item.operation,
        old_state=dict(item.old_values) if item.old_values is not None else None,
        new_state=dict(item.values) if item.values else None,
        **common,
    )


def item_from_event(event: Mapping[str, Any]) -> UniversalEditItem:
    """Rebuild the transport item from a stored family event (replay)."""
    payload = dict(event.get("family") or {})
    return UniversalEditItem(
        adapter=str(payload["adapter"]),
        operation=str(payload["operation"]),
        values=dict(payload.get("values") or {}),
        target=payload.get("target"),
        time_index=payload.get("time_index"),
        old_values=payload.get("old_values"),
    )


# ---------------------------------------------------------------------------
# Validation (single source of truth: registry + adapters)
# ---------------------------------------------------------------------------


def validate_command(
    command: EditCommand, context: SiteContext, *, index: int = 0
) -> None:
    """Run the registry's own validation for one command.

    Raises :class:`ApiError` (``invalid_edit_state``, 422) relaying the
    validator's message verbatim — fence names, domains, and grammar come
    from the adapter, never restated here. Both refusal vocabularies of
    ``adapter.validate`` relay: :class:`EditStateError` (grammar / stale
    revision) and :class:`AdapterSchemaError` (schema + fence rulings,
    e.g. land-cover's fenced class 7 water) — the fence must reach the
    client as a typed 4xx naming the fence, never a 500. A third
    vocabulary relays as ``invalid_request`` (400): plain
    :class:`AdapterRegistryError` — a registry-level refusal, most
    commonly an operation the adapter never declared (u-e2 M1).
    """
    try:
        adapter = family_registry_adapter(command.adapter_id)
        adapter.validate(command, context)
    except (EditStateError, AdapterSchemaError) as error:
        raise ApiError(
            "invalid_edit_state",
            str(error),
            field=f"edits[{index}].values",
            details={"adapter": command.adapter_id, "operation": command.operation},
        ) from error
    except AdapterRegistryError as error:
        # u-e2 M1: a registry-level refusal — most commonly an undeclared
        # operation ("updaet", "grow", "sculpt") — must relay as a typed
        # 4xx naming the operation. At base it escaped the route's catch
        # vocabulary entirely and surfaced as an HTTP 500.
        # (AdapterSchemaError subclasses AdapterRegistryError, so the
        # clause above must stay first to keep schema rulings at 422.)
        raise ApiError(
            "invalid_request",
            str(error),
            field=f"edits[{index}].operation",
            details={"adapter": command.adapter_id, "operation": command.operation},
        ) from error


def check_transportable(item: UniversalEditItem, index: int) -> None:
    """Refuse payloads this transport cannot carry, before validation.

    Typed, named refusals only — no silent re-routing:

    * ``vegetation_geometry`` → the tree transport (``tree-edits-v1``);
    * unknown / not-yet-integrated adapter ids → ``unsupported_transport``
      (the capability document is the authority on what is integrated).
    """
    if item.adapter in TREE_ONLY_ADAPTERS:
        raise ApiError(
            "unsupported_transport",
            f"adapter {item.adapter!r} travels on the tree edit transport "
            "(POST /edits, tree-edits-v1); the universal transport does not "
            "duplicate tree authority",
            field=f"edits[{index}].adapter",
        )
    try:
        family_registry_adapter(item.adapter)
    except Exception as error:
        raise ApiError(
            "unsupported_transport",
            f"adapter {item.adapter!r} has no integrated execution path "
            f"(see GET /api/v1/capabilities): {error}",
            field=f"edits[{index}].adapter",
        ) from error
