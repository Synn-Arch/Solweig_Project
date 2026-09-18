# SPDX-License-Identifier: GPL-3.0-only
"""Scenario state persistence for :class:`PlanExecutor` (U-D intake item a).

The executor's scenario state is otherwise memory-only and dies with the
process: the accumulated forcing/land-cover/model-parameter overlays, the
accumulated building-massing fold (u-d4c), the tree-layer edit log with
its publication watermarks, the scene-graph revision, and the in-memory
``(node, time)`` result-store index. This module snapshots ALL of it into
a scenario directory and restores a freshly constructed executor — in a
fresh process — so the resumed scenario keeps producing results bitwise
identical to the never-restarted sequence.

Restore discipline (the two hazards this module exists to not reintroduce):

* **u-c1b pairing.** The worker's publication watermarks and the
  executor's accumulated overlays are two views of one fact ("what the
  last PUBLISHED batch ran under"). Saving asserts they are EQUAL (a
  contradictory snapshot is refused, never written); restoring adopts
  them through :meth:`ExactWorker.adopt_published_state` — the hook for
  publications made OUTSIDE ``ExactWorker.run`` — which advances the
  published watermarks WITHOUT staging anything. ``stage_*`` would leave
  the overlays forever-pending and force every later job full-tile; the
  adopt path is what "restoring must NOT re-arm published overlays"
  means mechanically.
* **Store single-writer + revision discipline.** A fresh process builds a
  fresh store (fresh writer token). The snapshot's published entries are
  replayed through :meth:`TemporalResultStore.publish` grouped by their
  original ``(scene_revision, job_id)`` batches in ascending revision
  order — the same sequence the original publishes used — so the store's
  staleness guards and R2 window retention are reconstructed exactly, not
  bypassed.

Files in the scenario directory:

* ``scenario-state.json`` — the snapshot (the commit marker; written
  last, atomically via ``os.replace``);
* ``landcover-resolved.npy`` — the accumulated land-cover resolve memo
  (u-c6b L5), present only when the scenario ever painted.

Crash-window pairing (u-d1b review HIGH): a re-save over an existing
directory writes the memo first and the JSON marker last, so a crash
between the two ``os.replace`` calls can leave JSON v1 + memo v2 on
disk. The JSON document therefore carries a ``sha256`` of the memo
bytes it was written against, and every load re-hashes the memo file
and refuses the pair on mismatch — a torn directory is a typed
``ScenarioStateError``, never a silent half-state (which would feed a
foreign memo to the value-diff machinery and intercept a lost paint as
a whole-batch no-op). The hash-in-JSON design was chosen over
per-snapshot memo filenames: it needs no orphan cleanup, tolerates the
memo path staying stable, and additionally detects any later
corruption/tampering of the memo bytes between save and load.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .adapters.building import BuildingSpec, MassingEdit
from .adapters.landcover import LandCoverOverlay
from .adapters.met_time import ForcingOverlay
from .edit_graph import SceneGraphState
from .edit_types import (
    ForcingChange,
    FrozenMapping,
    LandCoverPaintPatch,
)
from .edits import TreeEdit
from .geometry import RasterWindow
from .trees import validate_tree_spec
from .geometry import TreeSpec
from .store import TemporalResultEntry
from .worker import PublicationWatermarks

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids an import cycle
    from .executor import PlanExecutor

__all__ = [
    "SCENARIO_STATE_SCHEMA_VERSION",
    "ScenarioStateError",
    "ScenarioStateSnapshot",
    "read_snapshot",
    "restore_into_executor",
    "snapshot_from_executor",
    "write_snapshot",
]

#: v2 (u-d1b review): the document gained ``results_root`` and
#: ``landcover_resolved_sha256`` (the memo-pairing hash). v1 documents
#: predate the pairing hash and can never be pair-verified, so they are
#: refused rather than loaded unverifiable.
#:
#: v3 (u-d4c): the document gained ``massing_edits`` — the executor's
#: ACCUMULATED building fold. A v2 snapshot taken mid-session after a
#: published building edit is not just incomplete, it is AMBIGUOUS: it
#: cannot prove the session carried no published buildings, and
#: restoring it would silently arm the baseline-bound worker routing
#: that drops them (the exact defect u-d4c fixes). v2 documents are
#: therefore refused like v1 — the refuse-unverifiable discipline, not
#: a lossy default.
SCENARIO_STATE_SCHEMA_VERSION = 3
STATE_FILENAME = "scenario-state.json"
RESOLVED_FILENAME = "landcover-resolved.npy"


class ScenarioStateError(RuntimeError):
    """Snapshot save/restore failure (identity mismatch, corrupt file,
    contradictory executor state)."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _memo_pair_error(path: Path, detail: str) -> ScenarioStateError:
    return ScenarioStateError(
        "scenario state memo/JSON pair is inconsistent "
        f"({detail}); the directory was likely torn by a crash between "
        f"the two atomic writes — refusing to restore {path.parent}"
    )


# ---------------------------------------------------------------------------
# Serialization helpers (typed records <-> JSON dicts)
# ---------------------------------------------------------------------------


def _spec_to_dict(spec: TreeSpec) -> dict[str, Any]:
    return {
        "tree_id": spec.tree_id,
        "x_m": float(spec.x_m),
        "y_m": float(spec.y_m),
        "height_m": float(spec.height_m),
        "canopy_radius_m": float(spec.canopy_radius_m),
        "trunk_ratio": float(spec.trunk_ratio),
        "transmissivity": float(spec.transmissivity),
    }


def _spec_from_dict(data: Mapping[str, Any]) -> TreeSpec:
    spec = TreeSpec(
        tree_id=str(data["tree_id"]),
        x_m=float(data["x_m"]),
        y_m=float(data["y_m"]),
        height_m=float(data["height_m"]),
        canopy_radius_m=float(data["canopy_radius_m"]),
        trunk_ratio=float(data["trunk_ratio"]),
        transmissivity=float(data["transmissivity"]),
    )
    # u-e1 L1: a snapshot is untrusted input — the SAME schema every live
    # edit enforces must gate the restore (TreeSpec.__post_init__ alone
    # accepts trunk_ratio on the closed [0, 1] while live edits enforce
    # the half-open [0, 1)). Without this, a tampered snapshot surfaced
    # as a RAW ValueError from TreeLayer.add_tree mid-replay (partially
    # applied state) instead of a typed whole-restore refusal.
    try:
        validate_tree_spec(spec)
    except ValueError as error:
        raise ScenarioStateError(
            f"snapshot tree {spec.tree_id!r} fails the live tree schema: "
            f"{error} (the snapshot is tampered or from an incompatible "
            "writer; refusing the whole restore)"
        ) from error
    return spec


def _value_to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _value_to_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise ScenarioStateError(
        f"overlay value {value!r} of type {type(value).__name__} is not "
        "serializable scenario state"
    )


def _value_from_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(
            [(str(key), _value_from_json(item)) for key, item in value.items()]
        )
    return value


def _forcing_overlay_to_dict(overlay: ForcingOverlay | None) -> list[dict[str, Any]] | None:
    if overlay is None:
        return None
    return [
        {
            "variable": change.variable,
            "time_index": change.time_index,
            "before_value": _value_to_json(change.before_value),
            "after_value": _value_to_json(change.after_value),
        }
        for change in overlay.changes
    ]


def _forcing_overlay_from_dict(data: Sequence[Mapping[str, Any]] | None) -> ForcingOverlay | None:
    if data is None:
        return None
    changes = tuple(
        ForcingChange(
            variable=str(item["variable"]),
            time_index=(
                int(item["time_index"]) if item["time_index"] is not None else None
            ),
            before_value=_value_from_json(item["before_value"]),
            after_value=_value_from_json(item["after_value"]),
        )
        for item in data
    )
    return ForcingOverlay(changes=changes)


def _window_to_dict(window: RasterWindow) -> dict[str, int]:
    return {
        "row_start": int(window.row_start),
        "row_stop": int(window.row_stop),
        "col_start": int(window.col_start),
        "col_stop": int(window.col_stop),
    }


def _window_from_dict(data: Mapping[str, int]) -> RasterWindow:
    return RasterWindow(
        row_start=int(data["row_start"]),
        row_stop=int(data["row_stop"]),
        col_start=int(data["col_start"]),
        col_stop=int(data["col_stop"]),
    )


def _landcover_overlay_to_dict(
    overlay: LandCoverOverlay | None,
) -> list[dict[str, Any]] | None:
    if overlay is None:
        return None
    return [
        {
            "window": _window_to_dict(patch.window),
            "before_classes": list(patch.before_classes),
            "after_classes": list(patch.after_classes),
        }
        for patch in overlay.patches
    ]


def _landcover_overlay_from_dict(
    data: Sequence[Mapping[str, Any]] | None,
) -> LandCoverOverlay | None:
    if data is None:
        return None
    patches = tuple(
        LandCoverPaintPatch(
            window=_window_from_dict(item["window"]),
            before_classes=tuple(int(code) for code in item["before_classes"]),
            after_classes=tuple(int(code) for code in item["after_classes"]),
        )
        for item in data
    )
    return LandCoverOverlay(patches=patches)


def _parameters_to_dict(parameters: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if parameters is None:
        return None
    return {str(name): _value_to_json(value) for name, value in sorted(parameters.items())}


def _building_spec_to_dict(spec: BuildingSpec) -> dict[str, Any]:
    return {
        "building_id": spec.building_id,
        "footprint_m": [
            [float(x), float(y)] for x, y in spec.footprint_m
        ],
        "height_m": float(spec.height_m),
    }


def _building_spec_from_dict(data: Mapping[str, Any]) -> BuildingSpec:
    return BuildingSpec(
        building_id=str(data["building_id"]),
        footprint_m=tuple(
            (float(vertex[0]), float(vertex[1])) for vertex in data["footprint_m"]
        ),
        height_m=float(data["height_m"]),
    )


def _massing_edit_to_dict(edit: MassingEdit) -> dict[str, Any]:
    return {
        "building_id": edit.building_id,
        "before": _building_spec_to_dict(edit.before) if edit.before else None,
        "after": _building_spec_to_dict(edit.after) if edit.after else None,
    }


def _massing_edit_from_dict(data: Mapping[str, Any]) -> MassingEdit:
    # MassingEdit's constructor re-validates both specs (the seam
    # discipline), so a tampered or corrupt fold entry is refused here,
    # never inside the raster write.
    return MassingEdit(
        building_id=str(data["building_id"]),
        before=(
            _building_spec_from_dict(data["before"]) if data["before"] else None
        ),
        after=_building_spec_from_dict(data["after"]) if data["after"] else None,
    )


def _grid_identity(executor: PlanExecutor) -> dict[str, Any]:
    cache = executor.cache
    return {
        "site_id": cache.site_id,
        "tile_key": cache.tile_key,
        "rows": int(cache.rows),
        "cols": int(cache.cols),
        "pixel_size_m": float(cache.pixel_size_m),
        "origin_x_m": float(cache.manifest.origin_x_m),
        "origin_y_m": float(cache.manifest.origin_y_m),
        "time_steps": int(cache.time_steps),
        "cache_schema_version": int(cache.manifest.cache_schema_version),
    }


# ---------------------------------------------------------------------------
# Snapshot record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioStateSnapshot:
    """The complete, serializable scenario state of one executor.

    ``watermarks`` mirrors the worker's publication watermarks (what the
    last published batch ran under); ``applied_*`` mirror the executor's
    accumulated state — at every quiescent point these are equal pairs,
    and :func:`snapshot_from_executor` refuses to snapshot them unequal
    (the u-c1b pairing hazard).
    """

    schema_version: int = SCENARIO_STATE_SCHEMA_VERSION
    scenario_id: str = ""
    site_identity: dict[str, Any] = field(default_factory=dict)
    selected_date_str: str = ""
    influence_config: dict[str, Any] = field(default_factory=dict)
    results_root: str = ""
    scene_revision: int = 0
    node_versions: dict[str, int] = field(default_factory=dict)
    tree_edits: tuple[dict[str, Any], ...] = ()
    acked_sequence: int = 0
    forcing_overlay: list[dict[str, Any]] | None = None
    landcover_overlay: list[dict[str, Any]] | None = None
    model_parameters: dict[str, Any] | None = None
    #: The executor's accumulated building fold (u-d4c), serialized as an
    #: ORDERED list — one entry per building id in fold order. It must be
    #: a list, not a dict: the document is written with ``sort_keys=True``
    #: (which orders KEYS, never list elements), and the fold's insertion
    #: order is part of the scenario's replay identity.
    massing_edits: tuple[dict[str, Any], ...] = ()
    landcover_resolved: str | None = None
    #: sha256 (hex) of the memo file bytes the JSON document was written
    #: against. Set by :func:`write_snapshot` when it writes the memo;
    #: verified by :func:`read_snapshot` and re-verified at restore load.
    landcover_resolved_sha256: str | None = None
    store_entries: tuple[dict[str, Any], ...] = ()
    requested_variables: tuple[str, ...] = ()
    saved_at: str = ""


def snapshot_from_executor(executor: PlanExecutor) -> ScenarioStateSnapshot:
    """Capture a quiescent executor's complete scenario state.

    Call between batches (after :meth:`PlanExecutor.execute` returned).
    The pairing assertion below is the save-time half of the u-c1b
    discipline: the worker's published overlays and the executor's
    accumulated overlays must describe the same published batch, and a
    snapshot of contradictory state is refused outright — restoring it
    would re-arm published overlays or silently revert accumulated edits.
    """
    watermarks = executor._worker.publication_watermarks
    applied_forcing = executor._applied_forcing_overlay
    applied_landcover = executor._applied_landcover_overlay
    applied_parameters = executor._applied_model_parameters
    pairing_problems: list[str] = []
    if watermarks.forcing_overlay != applied_forcing:
        pairing_problems.append("forcing overlay")
    if watermarks.landcover_overlay != applied_landcover:
        pairing_problems.append("land-cover overlay")
    if dict(watermarks.model_parameters or {}) != dict(applied_parameters or {}):
        pairing_problems.append("model parameters")
    if pairing_problems:
        raise ScenarioStateError(
            "refusing to snapshot contradictory executor state: the "
            "worker's published watermarks and the executor's accumulated "
            f"state disagree on {pairing_problems} (u-c1b pairing; save "
            "only between batches, after execute() returned)"
        )

    store_entries: list[dict[str, Any]] = []
    for key in sorted(executor.store):
        for entry in executor.store.window_entries(*key):
            store_entries.append(
                {
                    "node_id": entry.node_id,
                    "time_index": int(entry.time_index),
                    "scene_revision": int(entry.scene_revision),
                    "job_id": entry.job_id,
                    "mode": entry.mode,
                    "write_window": _window_to_dict(entry.write_window),
                    "patch_path": str(entry.patch_path) if entry.patch_path else None,
                }
            )

    import datetime

    return ScenarioStateSnapshot(
        scenario_id=executor._scenario_id,
        site_identity=_grid_identity(executor),
        selected_date_str=executor.selected_date_str,
        influence_config={
            str(key): _value_to_json(value)
            for key, value in sorted(asdict(executor._influence_config).items())
        },
        results_root=str(executor.results_root),
        scene_revision=int(executor.scene_revision),
        node_versions={
            str(node): int(version)
            for node, version in executor.state.node_versions.items()
        },
        tree_edits=tuple(
            {
                "sequence": int(edit.sequence),
                "tree_id": edit.tree_id,
                "old_tree": _spec_to_dict(edit.old_tree) if edit.old_tree else None,
                "new_tree": _spec_to_dict(edit.new_tree) if edit.new_tree else None,
            }
            for edit in executor.layer.edits
        ),
        acked_sequence=int(watermarks.acked_sequence),
        forcing_overlay=_forcing_overlay_to_dict(applied_forcing),
        landcover_overlay=_landcover_overlay_to_dict(applied_landcover),
        model_parameters=_parameters_to_dict(applied_parameters),
        massing_edits=tuple(
            _massing_edit_to_dict(edit)
            for edit in executor._applied_massing_edits.values()
        ),
        landcover_resolved=RESOLVED_FILENAME
        if executor._applied_landcover_resolved is not None
        else None,
        store_entries=tuple(store_entries),
        requested_variables=tuple(executor._worker.requested_variables),
        saved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# File I/O (atomic: the JSON document is the commit marker)
# ---------------------------------------------------------------------------


def write_snapshot(
    snapshot: ScenarioStateSnapshot,
    directory: str | Path,
    *,
    landcover_resolved: np.ndarray | None = None,
) -> Path:
    """Write ``scenario-state.json`` (+ the resolve memo) atomically.

    ``landcover_resolved`` is the executor's accumulated resolve memo; it
    must be passed exactly when ``snapshot.landcover_resolved`` names it.
    The memo lands first via ``os.replace``; the JSON document last — a
    reader that sees the document sees a complete directory. Because a
    crash between the two replaces can leave the OLD document beside the
    NEW memo, the document records the sha256 of the memo bytes it was
    written against (``landcover_resolved_sha256``); loaders refuse a
    pair whose hash does not match (see :func:`read_snapshot`).
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    memo_sha256: str | None = None
    if snapshot.landcover_resolved is not None:
        if landcover_resolved is None:
            raise ScenarioStateError(
                "snapshot declares a land-cover resolve memo but none was "
                "supplied to write"
            )
        memo = np.asarray(landcover_resolved)
        tmp_memo = target / (RESOLVED_FILENAME + ".tmp.npy")
        np.save(tmp_memo, memo, allow_pickle=False)
        os.replace(tmp_memo, target / RESOLVED_FILENAME)
        memo_sha256 = _file_sha256(target / RESOLVED_FILENAME)
    elif landcover_resolved is not None:
        raise ScenarioStateError(
            "a land-cover resolve memo was supplied to write but the "
            "snapshot does not declare one — the document would deny a "
            "memo the directory carries"
        )
    document = {
        "schema_version": snapshot.schema_version,
        "scenario_id": snapshot.scenario_id,
        "site_identity": snapshot.site_identity,
        "selected_date_str": snapshot.selected_date_str,
        "influence_config": snapshot.influence_config,
        "results_root": snapshot.results_root,
        "scene_revision": snapshot.scene_revision,
        "node_versions": snapshot.node_versions,
        "tree_edits": list(snapshot.tree_edits),
        "acked_sequence": snapshot.acked_sequence,
        "forcing_overlay": snapshot.forcing_overlay,
        "landcover_overlay": snapshot.landcover_overlay,
        "model_parameters": snapshot.model_parameters,
        "massing_edits": list(snapshot.massing_edits),
        "landcover_resolved": snapshot.landcover_resolved,
        "landcover_resolved_sha256": memo_sha256,
        "store_entries": list(snapshot.store_entries),
        "requested_variables": list(snapshot.requested_variables),
        "saved_at": snapshot.saved_at,
    }
    tmp = target / (STATE_FILENAME + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, target / STATE_FILENAME)
    return target / STATE_FILENAME


def read_snapshot(directory: str | Path) -> ScenarioStateSnapshot:
    """Read and structurally validate a snapshot directory.

    Missing document keys raise the module's :class:`ScenarioStateError`
    (typed, naming the key) — never a bare ``KeyError``. When the
    document declares a resolve memo, the memo file must exist AND hash
    to the ``landcover_resolved_sha256`` recorded in the document; a
    mismatch is the torn-directory signature (JSON v1 + memo v2 after a
    crash between the two atomic writes) and is refused here, before any
    executor is touched.
    """
    path = Path(directory) / STATE_FILENAME
    if not path.is_file():
        raise ScenarioStateError(f"no {STATE_FILENAME} under {Path(directory)}")
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ScenarioStateError(f"corrupt scenario state {path}: {error}") from None

    def require(key: str) -> Any:
        if key not in document:
            raise ScenarioStateError(
                f"scenario state {path} is missing required key {key!r}"
            )
        return document[key]

    version = int(document.get("schema_version", 0))
    if version != SCENARIO_STATE_SCHEMA_VERSION:
        raise ScenarioStateError(
            f"scenario state schema {version} under {path} is not the "
            f"supported version {SCENARIO_STATE_SCHEMA_VERSION}; a RESET "
            "edit heals the scenario (a reset the ledger covers at or "
            "after the snapshot's coverage bypasses the restore and "
            "rebuilds fresh from the ledger)"
        )
    landcover_resolved = document.get("landcover_resolved")
    memo_sha256 = document.get("landcover_resolved_sha256")
    if landcover_resolved is not None:
        memo_path = Path(directory) / str(landcover_resolved)
        if not memo_path.is_file():
            raise _memo_pair_error(path, f"missing memo {memo_path}")
        if memo_sha256 is None:
            raise _memo_pair_error(
                path,
                "the document names a memo but records no "
                "landcover_resolved_sha256 for it",
            )
        if _file_sha256(memo_path) != memo_sha256:
            raise _memo_pair_error(
                path,
                f"memo {memo_path.name} does not match the sha256 the "
                "document was written against",
            )
    return ScenarioStateSnapshot(
        schema_version=version,
        scenario_id=str(require("scenario_id")),
        site_identity=dict(require("site_identity")),
        selected_date_str=str(require("selected_date_str")),
        influence_config=dict(require("influence_config")),
        results_root=str(require("results_root")),
        scene_revision=int(require("scene_revision")),
        node_versions={str(k): int(v) for k, v in require("node_versions").items()},
        tree_edits=tuple(dict(item) for item in require("tree_edits")),
        acked_sequence=int(require("acked_sequence")),
        forcing_overlay=require("forcing_overlay"),
        landcover_overlay=require("landcover_overlay"),
        model_parameters=require("model_parameters"),
        massing_edits=tuple(dict(item) for item in require("massing_edits")),
        landcover_resolved=landcover_resolved,
        landcover_resolved_sha256=memo_sha256 if landcover_resolved else None,
        store_entries=tuple(dict(item) for item in require("store_entries")),
        requested_variables=tuple(str(name) for name in require("requested_variables")),
        saved_at=str(document.get("saved_at", "")),
    )


# ---------------------------------------------------------------------------
# Operator rebase (explicit, verified — never silent)
# ---------------------------------------------------------------------------


def rebase_results_root(
    snapshot: ScenarioStateSnapshot,
    new_root: str | Path,
    *,
    allow_root_move: bool = False,
    snapshot_directory: str | Path | None = None,
) -> ScenarioStateSnapshot:
    """Re-point a snapshot at a relocated results root (operator rebase).

    ``restore_into_executor`` REFUSES a snapshot whose ``results_root``
    diverges from the executor's — by design, because store entries pin
    absolute patch paths under the save-time root and silently re-pointing
    them could index another deployment's bytes. This function is the
    explicit, auditable way out for a deployment that physically moved the
    results directory (new machine, new mount point): it rewrites the
    snapshot's ``results_root``, translates every store entry's
    ``patch_path`` from the old root to the new one, and then VERIFIES the
    moved deployment before anything is restored:

    * the new root must exist;
    * every translated patch path must exist on disk (patches are stored
      as per-job directories under the results root);
    * entry paths that did not live under the old root must still exist
      unchanged;
    * when the snapshot declares a land-cover resolve memo, the memo must
      exist in ``snapshot_directory`` and still hash to the recorded
      ``landcover_resolved_sha256``.

    Any failure raises :class:`ScenarioStateError` naming the offending
    paths and leaves the on-disk document untouched (the rewrite is a
    single atomic ``os.replace`` that only happens after every check
    passed). ``allow_root_move`` is the explicit operator opt-in — the
    CLI-equivalent of ``--allow-root-move``; omitting it is a typed
    refusal, never a silent relocation.

    When ``snapshot_directory`` is given the updated document is written
    back there atomically; without it the function is a pure verified
    transform returning the new snapshot (the caller persists it).
    """
    if not allow_root_move:
        raise ScenarioStateError(
            "rebase_results_root requires the explicit operator opt-in "
            "allow_root_move=True (the moved-deployment rebase is never "
            "silent: restoring a re-pointed snapshot without verification "
            "could index patches from another deployment)"
        )
    old_root = Path(snapshot.results_root)
    target_root = Path(new_root)
    if not target_root.is_dir():
        raise ScenarioStateError(
            f"rebase target results root {target_root} does not exist or is "
            "not a directory (move the results tree first, then rebase)"
        )

    directory = Path(snapshot_directory) if snapshot_directory is not None else None
    if snapshot.landcover_resolved is not None and directory is None:
        raise ScenarioStateError(
            "the snapshot declares a land-cover resolve memo; "
            "rebase_results_root needs snapshot_directory to re-verify the "
            "memo hash before rewriting the document"
        )
    if directory is not None and snapshot.landcover_resolved is not None:
        memo_path = directory / str(snapshot.landcover_resolved)
        if not memo_path.is_file():
            raise ScenarioStateError(
                f"rebase refused: resolve memo {memo_path} is missing from "
                "the snapshot directory"
            )
        if _file_sha256(memo_path) != snapshot.landcover_resolved_sha256:
            raise ScenarioStateError(
                f"rebase refused: resolve memo {memo_path.name} no longer "
                "hashes to the landcover_resolved_sha256 the document was "
                "written against"
            )

    resolved_old = old_root.resolve()
    translated: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in snapshot.store_entries:
        item = dict(entry)
        raw = item.get("patch_path")
        if raw:
            path = Path(str(raw))
            try:
                relative = path.resolve().relative_to(resolved_old)
            except ValueError:
                relative = None  # pinned outside the results root; keep as-is
            if relative is not None:
                path = target_root / relative
                item["patch_path"] = str(path)
            if not path.exists():
                missing.append(str(path))
        translated.append(item)
    if missing:
        shown = ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
        raise ScenarioStateError(
            f"rebase refused: {len(missing)} store-entry patch path(s) do "
            f"not exist under {target_root} ({shown}) — the moved results "
            "tree is incomplete or the target root is wrong"
        )

    rebased = ScenarioStateSnapshot(
        **{
            **_snapshot_fields(snapshot),
            "results_root": str(target_root),
            "store_entries": tuple(translated),
        }
    )
    if directory is not None:
        path = directory / STATE_FILENAME
        if not path.is_file():
            raise ScenarioStateError(f"no {STATE_FILENAME} under {directory}")
        document = json.loads(path.read_text())
        document["results_root"] = rebased.results_root
        document["store_entries"] = list(rebased.store_entries)
        tmp = directory / (STATE_FILENAME + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    return rebased


def _snapshot_fields(snapshot: ScenarioStateSnapshot) -> dict[str, Any]:
    """Dataclass fields of ``snapshot`` as a constructor-ready mapping."""
    return {field.name: getattr(snapshot, field.name) for field in fields(snapshot)}


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def _verify_identity(executor: PlanExecutor, snapshot: ScenarioStateSnapshot) -> None:
    problems: list[str] = []
    if snapshot.scenario_id != executor._scenario_id:
        problems.append(
            f"scenario {snapshot.scenario_id!r} != executor scenario "
            f"{executor._scenario_id!r}"
        )
    # The store entries carry absolute patch paths under the save-time
    # results root; restoring under a DIFFERENT root would point the
    # rebuilt store index at another deployment's patches (or nothing).
    # Refuse — never silently relocate: an operator who moved the root
    # must move it back (or re-point the executor) so the pinned paths
    # resolve to the exact bytes the snapshot published. Path.resolve()
    # normalizes symlinks so an aliased root is NOT a divergence.
    if Path(snapshot.results_root).resolve() != executor.results_root.resolve():
        problems.append(
            f"results_root {snapshot.results_root!r} != executor results "
            f"root {str(executor.results_root)!r} (the snapshot's store "
            "entries pin absolute patch paths under the save-time root)"
        )
    current = _grid_identity(executor)
    for key in sorted(set(current) | set(snapshot.site_identity)):
        expected = snapshot.site_identity.get(key)
        if expected is not None and current.get(key) != expected:
            problems.append(f"{key}: snapshot {expected!r} != site {current.get(key)!r}")
    if snapshot.selected_date_str != executor.selected_date_str:
        problems.append(
            f"selected_date_str {snapshot.selected_date_str!r} != executor "
            f"{executor.selected_date_str!r}"
        )
    influence = {
        str(key): _value_to_json(value)
        for key, value in sorted(asdict(executor._influence_config).items())
    }
    if influence != snapshot.influence_config:
        problems.append(
            f"influence config {snapshot.influence_config!r} != executor "
            f"{influence!r} (locality policy is part of the scenario's "
            "computational identity)"
        )
    if problems:
        raise ScenarioStateError(
            "scenario state does not match this executor: " + "; ".join(problems)
        )


def _replayed_acked_sequence(
    snapshot: ScenarioStateSnapshot,
) -> int:
    """Map the original acked watermark onto the replayed layer's sequences.

    Replaying the log through the layer's public mutators renumbers every
    mutation 1..M in order (undo records included), while the ORIGINAL
    sequences were the layer's own counter at save time. Both numberings
    order the same records, so the acked watermark maps positionally: the
    restored acked sequence is the replayed sequence of the LAST original
    edit at or below the watermark (0 when none is).
    """
    acked = 0
    for position, item in enumerate(snapshot.tree_edits, start=1):
        if int(item["sequence"]) <= snapshot.acked_sequence:
            acked = position
    return acked


def _replay_tree_log(executor: PlanExecutor, snapshot: ScenarioStateSnapshot) -> None:
    """Rebuild the layer's tree state by replaying the serialized log.

    The log is the layer's own verbatim record (published replays AND
    rollback undos), so replaying it from a fresh baseline layer through
    the executor's atomic preflight reproduces the same tree state and
    the same pending/consumed split.
    """
    edits = [
        TreeEdit(
            scenario_id=executor._scenario_id,
            sequence=int(item["sequence"]),
            tree_id=str(item["tree_id"]),
            old_tree=_spec_from_dict(item["old_tree"]) if item["old_tree"] else None,
            new_tree=_spec_from_dict(item["new_tree"]) if item["new_tree"] else None,
        )
        for item in snapshot.tree_edits
    ]
    executor._replay_tree_edits(edits)


def _restore_store_index(executor: PlanExecutor, snapshot: ScenarioStateSnapshot) -> None:
    """Rebuild the published-entry index through the store's own publish.

    Entries are grouped by their original ``(scene_revision, job_id)``
    publication batch and replayed in ascending revision order — the
    original publications' relative order — so the same-revision conflict
    guards and the R2 non-intersecting-window retention resolve exactly
    as they did live. Identical duplicates inside one group (an
    idempotent republish) are collapsed; a re-restore onto the store it
    came from replays byte-identical content and is a store-accepted
    no-op.
    """
    groups: dict[tuple[int, str], dict[tuple[Any, ...], dict[str, Any]]] = {}
    for item in snapshot.store_entries:
        key = (int(item["scene_revision"]), str(item["job_id"]))
        content = (
            item["node_id"],
            int(item["time_index"]),
            json.dumps(item["write_window"], sort_keys=True),
            str(item["patch_path"]),
            item["mode"],
        )
        groups.setdefault(key, {})[content] = item
    if not groups:
        return
    entries_by_batch: list[tuple[tuple[int, str], list[TemporalResultEntry]]] = []
    for key in sorted(groups):
        entries = [
            TemporalResultEntry(
                node_id=str(item["node_id"]),
                time_index=int(item["time_index"]),
                scene_revision=int(item["scene_revision"]),
                job_id=str(item["job_id"]),
                mode=str(item["mode"]),
                write_window=_window_from_dict(item["write_window"]),
                patch_path=Path(item["patch_path"]) if item["patch_path"] else None,
            )
            for item in groups[key].values()
        ]
        entries_by_batch.append((key, entries))
    for _key, entries in entries_by_batch:
        executor.store.publish(entries, writer=executor._store_writer_id)


def restore_into_executor(
    executor: PlanExecutor,
    snapshot: ScenarioStateSnapshot,
    *,
    directory: str | Path,
) -> None:
    """Restore a snapshot into a freshly constructed executor.

    Identity (scenario, site cache, date, influence policy, results
    root) is verified FIRST; any mismatch refuses the whole restore.
    The layer log, scene
    state, worker watermarks (via ``adopt_published_state``), the worker's
    between-batch steady state (the SAME overlays staged as published —
    staged == published arms nothing pending), accumulated executor
    overlays, resolve memo, and the store index are then rebuilt in
    dependency order.
    """
    _verify_identity(executor, snapshot)

    _replay_tree_log(executor, snapshot)
    executor._state = SceneGraphState(
        graph=executor._graph,
        scene_revision=snapshot.scene_revision,
        node_versions=snapshot.node_versions,
    )

    forcing_overlay = _forcing_overlay_from_dict(snapshot.forcing_overlay)
    landcover_overlay = _landcover_overlay_from_dict(snapshot.landcover_overlay)
    model_parameters = (
        {
            str(name): _value_from_json(value)
            for name, value in dict(snapshot.model_parameters or {}).items()
        }
        if snapshot.model_parameters is not None
        else None
    )
    # u-c1b: adopt (published-outside-run semantics). Staging something
    # DIFFERENT from the published watermark would mark it pending and
    # force every later job full-tile — the exact hazard the adopt hook
    # exists for.
    executor._worker.adopt_published_state(
        PublicationWatermarks(
            acked_sequence=_replayed_acked_sequence(snapshot),
            forcing_overlay=forcing_overlay,
            landcover_overlay=landcover_overlay,
            model_parameters=model_parameters,
        )
    )
    # Post-publish steady state (u-d1 resume fidelity): a worker that just
    # published a batch holds the published overlays STAGED as well —
    # ``execute_plan`` re-stages on every dispatch, and ``worker.forcing()``
    # materializes from the STAGED overlay, so a restored worker without
    # this would answer baseline forcing until the next batch. Staging the
    # SAME overlays the adopt above published arms nothing: the worker's
    # pending detection is ``staged != published`` (and the explicit
    # presence/dirty-override flags, all off here), so staged == published
    # means exactly "nothing pending" — the same state a never-restarted
    # twin sits in between batches.
    executor._worker.stage_forcing_overlay(
        forcing_overlay, batch_pending=False
    )
    executor._worker.stage_landcover_overlay(
        landcover_overlay, dirty_windows=()
    )
    executor._worker.stage_model_parameters(model_parameters)
    executor._worker.requested_variables = tuple(snapshot.requested_variables)

    executor._applied_forcing_overlay = forcing_overlay
    executor._applied_landcover_overlay = landcover_overlay
    executor._applied_model_parameters = (
        dict(model_parameters) if model_parameters is not None else None
    )
    # u-d4c: restore the accumulated building fold. The ordered list
    # rebuilds the fold dict in its saved order, and MassingEdit's
    # constructor re-validates every entry on this seam (the same
    # discipline ``massing_edits_from_deltas`` applies to live batches).
    # With the fold restored, every later batch routes the regeneration
    # chain with the full fold — a restored session keeps its published
    # buildings instead of silently reverting to the baseline arrays.
    executor._applied_massing_edits = {
        edit.building_id: edit
        for edit in (
            _massing_edit_from_dict(item) for item in snapshot.massing_edits
        )
    }
    executor._staged_landcover_value_diff = ()
    executor._staged_landcover_resolved = None

    if snapshot.landcover_resolved is None:
        executor._applied_landcover_resolved = None
    else:
        path = Path(directory) / snapshot.landcover_resolved
        if not path.is_file():
            raise ScenarioStateError(
                f"snapshot references missing land-cover resolve memo {path}"
            )
        # Re-verify the pairing hash at load time (u-d1b): read_snapshot
        # checked it, but the memo bytes may have changed since — a memo
        # that does not belong to this document must never be fed to the
        # executor as resolved_old.
        if snapshot.landcover_resolved_sha256 is None:
            raise _memo_pair_error(
                path,
                "the snapshot names a memo but records no "
                "landcover_resolved_sha256 for it (only documents written "
                "by write_snapshot carry a verifiable pair)",
            )
        if _file_sha256(path) != snapshot.landcover_resolved_sha256:
            raise _memo_pair_error(
                path,
                f"memo {path.name} no longer matches the sha256 recorded "
                "in the document (changed since read)",
            )
        resolved = np.load(path, allow_pickle=False)
        if resolved.shape != (executor.cache.rows, executor.cache.cols):
            raise ScenarioStateError(
                f"land-cover resolve memo shape {resolved.shape} does not "
                f"match the site grid "
                f"({executor.cache.rows}, {executor.cache.cols})"
            )
        if resolved.dtype.kind not in "iu":
            raise ScenarioStateError(
                f"land-cover resolve memo dtype {resolved.dtype} is not an "
                "integer class grid"
            )
        executor._applied_landcover_resolved = resolved

    _restore_store_index(executor, snapshot)
