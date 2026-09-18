# SPDX-License-Identifier: GPL-3.0-only
"""Pydantic request models and the error envelope for the design-tool API.

Field names follow ``docs/incremental_design_tool/api_contract.md`` exactly.
The tree object uses portable site-normalized UV coordinates; conversion to
projected world coordinates happens server-side through the site manifest
(see :mod:`solweig_gpu.server.app` routes and
:func:`solweig_gpu.server.store.uv_to_world`).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ApiError",
    "ERROR_STATUS",
    "SHARED_SCENARIO_ID_RE",
    "CreateScenarioRequest",
    "EditItem",
    "EditRequest",
    "ExportRequest",
    "RealtimeOperationItem",
    "RealtimeOperationsRequest",
    "RequestedResult",
    "ResetRequest",
    "TreeObject",
    "TREE_PRESETS",
    "UniversalEditItem",
    "UniversalEditRequest",
    "ViewRequest",
]


#: Scenario-id convention for the shared world. The ONE definition both
#: enforcement sites use — ``create_app``'s direct-caller validation and
#: the env entrypoint's ``SOLWEIG_SHARED_SCENARIO_ID`` validator — so the
#: twin copies cannot drift apart. The id doubles as a path segment under
#: the state root, so it must stay a safe single path component.
SHARED_SCENARIO_ID_RE = re.compile(r"scn_[a-z0-9][a-z0-9_-]{0,63}")


#: Server-side tree component presets. The client cannot invent component
#: types; preset defaults are validated together with explicit field values.
TREE_PRESETS: dict[str, dict[str, Any]] = {
    "broad_canopy": {
        "label": "Broad canopy",
        "height_m": 18.0,
        "canopy_diameter_m": 11.0,
        "trunk_ratio": 0.25,
        "transmissivity": 0.03,
        "phenology": "deciduous",
    },
}


#: HTTP status for every contract error code.
ERROR_STATUS: dict[str, int] = {
    "invalid_request": 400,
    "scenario_not_found": 404,
    "scene_version_conflict": 409,
    "invalid_tree_geometry": 422,
    "site_limit_exceeded": 403,
    "job_not_found": 404,
    "job_failed": 500,
    "result_not_ready": 404,
    "result_not_found": 404,
    "cache_version_mismatch": 409,
    "rate_limited": 429,
    "idempotency_key_reused": 409,
    "payload_too_large": 413,
    "not_acceptable": 406,
    "site_identity_mismatch": 409,
    "view_not_available": 409,
    # Universal (family) edit transport (u-d4): the adapter registry and
    # its adapters are the single source of validation truth; the server
    # maps payloads onto EditCommands and relays their refusals verbatim.
    "unsupported_transport": 400,
    "invalid_edit_state": 422,
    "invalid_family_batch": 400,
    # Realtime collaborative operation plane (R1): transport-level shape
    # checks only — family PAYLOAD validation stays with the executor
    # bridge and is never restated here.
    "unknown_source_family": 400,
    "unknown_operation_verb": 400,
    "invalid_operation_payload": 400,
    "operation_id_reused": 409,
    "operation_not_found": 404,
    "internal_error": 500,
}


class ApiError(Exception):
    """Raise anywhere in the server to emit a contract error envelope."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.field = field
        self.status = int(status if status is not None else ERROR_STATUS.get(code, 400))
        self.details = dict(details or {})
        super().__init__(f"{code}: {message}")

    def envelope(self, request_id: str | None = None) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        error.update(self.details)
        if request_id:
            error["request_id"] = request_id
        return {"error": error}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TreeObject(_StrictModel):
    """Portable tree object in site-normalized UV coordinates.

    ``u``/``v`` bounds and preset dimension ranges are validated by the
    scenario routes (producing ``invalid_tree_geometry`` with a field path),
    not by pydantic, so clients get the contract's error code.

    Note: ``trunk_ratio`` domain is ``[0, 1)`` — a ratio of 1 would raise
    the trunk zone to the canopy top, which ``solweig_gpu.incremental``
    (the authoritative validator) rejects. ``data_model.md`` documents the
    same half-open interval (reconciled 2026-09-02, p9-rel evidence note).
    The server follows the library.
    """

    tree_id: str = Field(min_length=1, max_length=64)
    component_type: str = "broad_canopy"
    u: float
    v: float
    height_m: float
    canopy_diameter_m: float
    trunk_ratio: float = 0.25
    transmissivity: float = 0.03
    phenology: str = "deciduous"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateScenarioRequest(_StrictModel):
    site_id: str = Field(min_length=1, max_length=128)
    name: str = Field(default="", max_length=256)
    initial_state: Literal["baseline"] = "baseline"


class RequestedResult(_StrictModel):
    time_indices: list[int] | None = None
    variables: list[str] | None = None
    refine_full_day: bool = False


class EditItem(_StrictModel):
    operation: Literal["add", "move", "update", "delete"]
    tree: TreeObject | None = None
    tree_id: str | None = Field(default=None, min_length=1, max_length=64)


class EditRequest(_StrictModel):
    base_scene_version: int = Field(ge=0)
    edits: list[EditItem] = Field(min_length=1, max_length=64)
    requested_result: RequestedResult | None = None


class ResetRequest(_StrictModel):
    base_scene_version: int | None = Field(default=None, ge=0)


class ExportRequest(_StrictModel):
    scene_version: int = Field(ge=0)
    format: Literal["cog", "geotiff", "npz"]
    variables: list[str] | None = None
    time_indices: list[int] | None = None


class UniversalEditItem(_StrictModel):
    """One family edit in the universal transport (u-d4).

    The shape mirrors the frontend's ``composeFamilyEdit`` payload exactly
    (``family_edits.mjs``): ``adapter`` is the registry adapter id and the
    ONLY thing the server interprets; ``values`` (and ``old_values``, a
    declared claim about current state where an adapter requires one) are
    passed to the adapter's own validators verbatim — the server never
    restates a domain, fence, or schema. ``target`` carries the edit's
    subject where the adapter grammar separates it from the values: a
    land-cover paint window, a building id, or a view layer name.
    """

    adapter: str = Field(min_length=1, max_length=64)
    operation: str = Field(min_length=1, max_length=32)
    values: dict[str, Any] = Field(default_factory=dict)
    target: Any = None
    time_index: int | None = Field(default=None, ge=0, strict=True)
    old_values: dict[str, Any] | None = None


class UniversalEditRequest(_StrictModel):
    """A batch of universal family edits under the edits contract.

    Same idempotency/If-Match/versioning discipline as the tree
    ``EditRequest``; batches advance the scene version by exactly one and
    enqueue exactly one job. A batch whose only item is an ``output_view``
    operation is answered view-only (zero jobs).
    """

    base_scene_version: int = Field(ge=0)
    edits: list[UniversalEditItem] = Field(min_length=1, max_length=64)
    requested_result: RequestedResult | None = None


class ViewRequest(_StrictModel):
    """A view-only operation (UEDIT-007: zero scientific jobs).

    ``operation`` and the valid ``layer`` vocabulary are validated against
    the capability document (the ``output_view`` adapter's registered
    operations and ``producible_incremental`` layers) — never a hardcoded
    server-side list. The response is composed from already-published
    results; no solver job is ever enqueued.
    """

    operation: Literal["select_layer", "compare", "legend", "cached_time"]
    layer: str = Field(min_length=1, max_length=32)
    time_index: int = Field(default=0, ge=0, strict=True)
    compare_layer: str | None = Field(default=None, min_length=1, max_length=32)
    scene_version: int | None = Field(default=None, ge=0)


class RealtimeOperationItem(_StrictModel):
    """One collaborative operation in the realtime transport (R1).

    ``operation_id`` is the idempotency key (globally unique across the
    operation log). ``base_revision`` is ADVISORY — the server records it
    and may log divergence, but never rejects on it (multi-writer
    acceptance; collaborative_state.md). ``source_family``/``verb`` arrive
    as free strings and are checked against the frozen
    ``realtime.types`` vocabularies by the route so unknown values get a
    TYPED envelope (pydantic Literals would collapse them into a generic
    validation error). ``payload`` is stored verbatim; family-specific
    validation stays with the executor bridge.
    """

    operation_id: str = Field(min_length=1, max_length=128)
    client_sequence: int | None = Field(default=None, ge=0)
    base_revision: int | None = Field(default=None, ge=0)
    source_family: str = Field(min_length=1, max_length=64)
    entity_id: str | None = Field(default=None, min_length=1, max_length=128)
    verb: str = Field(min_length=1, max_length=32)
    payload: dict[str, Any] = Field(default_factory=dict)


class RealtimeOperationsRequest(_StrictModel):
    """A multi-writer batch of collaborative operations (1..128 items)."""

    actor_id: str = Field(min_length=1, max_length=128)
    operations: list[RealtimeOperationItem] = Field(min_length=1, max_length=128)
