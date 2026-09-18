# SPDX-License-Identifier: GPL-3.0-only
"""Edit coalescing helpers for an incremental design-tool worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from .geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    SunPosition,
    TreeSpec,
    dirty_window_for_edit,
    merge_windows,
)


class EditOperation(str, Enum):
    ADD = "add"
    MOVE = "move"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class TreeEdit:
    """One committed UI edit, ordered by a monotonically increasing sequence."""

    scenario_id: str
    sequence: int
    tree_id: str
    old_tree: TreeSpec | None
    new_tree: TreeSpec | None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not self.tree_id:
            raise ValueError("tree_id must be non-empty")
        if self.old_tree is None and self.new_tree is None:
            raise ValueError("an edit must contain an old or new tree")
        for tree in (self.old_tree, self.new_tree):
            if tree is not None and tree.tree_id != self.tree_id:
                raise ValueError("tree_id must match old_tree and new_tree")

    @property
    def operation(self) -> EditOperation:
        if self.old_tree is None:
            return EditOperation.ADD
        if self.new_tree is None:
            return EditOperation.DELETE
        moved = (self.old_tree.x_m, self.old_tree.y_m) != (
            self.new_tree.x_m,
            self.new_tree.y_m,
        )
        return EditOperation.MOVE if moved else EditOperation.UPDATE


@dataclass(frozen=True, slots=True)
class CoalescedEditBatch:
    """Latest semantically equivalent edit set for one scenario."""

    scenario_id: str
    first_sequence: int
    last_sequence: int
    edits: tuple[TreeEdit, ...]


def coalesce_tree_edits(edits: Iterable[TreeEdit]) -> CoalescedEditBatch:
    """Collapse repeated edits to the same tree while preserving old-state removal.

    Examples
    --------
    * add -> move -> resize becomes one add at the final state;
    * move -> move becomes one move from the first old state to the final state;
    * add -> delete before analysis becomes a no-op and is removed.
    """
    ordered = sorted(edits, key=lambda edit: edit.sequence)
    if not ordered:
        raise ValueError("edits must be non-empty")
    scenario_ids = {edit.scenario_id for edit in ordered}
    if len(scenario_ids) != 1:
        raise ValueError("all edits in a batch must target the same scenario")
    sequences = [edit.sequence for edit in ordered]
    if len(sequences) != len(set(sequences)):
        raise ValueError("edit sequences must be unique within a batch")

    by_tree: dict[str, tuple[TreeSpec | None, TreeSpec | None, int, int]] = {}
    for edit in ordered:
        if edit.tree_id not in by_tree:
            by_tree[edit.tree_id] = (
                edit.old_tree,
                edit.new_tree,
                edit.sequence,
                edit.sequence,
            )
        else:
            first_old, previous_new, first_sequence, _ = by_tree[edit.tree_id]
            if edit.old_tree != previous_new:
                raise ValueError(
                    f"non-contiguous edit chain for tree {edit.tree_id!r}: "
                    "old_tree does not match the preceding new_tree"
                )
            by_tree[edit.tree_id] = (
                first_old,
                edit.new_tree,
                first_sequence,
                edit.sequence,
            )

    coalesced: list[TreeEdit] = []
    for tree_id, (first_old, last_new, _, last_sequence) in by_tree.items():
        if first_old is None and last_new is None:
            continue
        if first_old == last_new:
            continue
        coalesced.append(
            TreeEdit(
                scenario_id=ordered[0].scenario_id,
                sequence=last_sequence,
                tree_id=tree_id,
                old_tree=first_old,
                new_tree=last_new,
            )
        )

    coalesced.sort(key=lambda edit: edit.sequence)
    return CoalescedEditBatch(
        scenario_id=ordered[0].scenario_id,
        first_sequence=ordered[0].sequence,
        last_sequence=ordered[-1].sequence,
        edits=tuple(coalesced),
    )


def dirty_windows_for_batch(
    batch: CoalescedEditBatch,
    *,
    grid: RasterGrid,
    sun_positions: Sequence[SunPosition],
    config: InfluenceConfig = InfluenceConfig(),
    merge_gap_pixels: int = 8,
    elevation_offset_for_tree: Callable[[TreeSpec], float] | None = None,
) -> list[RasterWindow]:
    """Calculate and merge conservative invalidation windows for a batch.

    ``elevation_offset_for_tree`` optionally supplies a per-tree local relief
    (see :func:`solweig_gpu.incremental.geometry.local_elevation_offset_m`);
    without it every tree uses ``config.elevation_offset_m``.
    """
    windows = [
        dirty_window_for_edit(
            grid,
            old_tree=edit.old_tree,
            new_tree=edit.new_tree,
            sun_positions=sun_positions,
            config=config,
            elevation_offset_for_tree=elevation_offset_for_tree,
        )
        for edit in batch.edits
    ]
    return merge_windows(windows, gap_pixels=merge_gap_pixels)
