from solweig_gpu.incremental.edits import (
    EditOperation,
    TreeEdit,
    coalesce_tree_edits,
    dirty_windows_for_batch,
)
from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    RasterGrid,
    SunPosition,
    TreeSpec,
)


def tree(tree_id: str, x: float, y: float, height: float = 10.0) -> TreeSpec:
    return TreeSpec(tree_id, x, y, height, canopy_radius_m=3.0)


def test_add_move_update_coalesces_to_final_add() -> None:
    added = tree("a", 10.0, 10.0)
    moved = tree("a", 20.0, 20.0)
    resized = tree("a", 20.0, 20.0, height=14.0)
    batch = coalesce_tree_edits(
        [
            TreeEdit("s", 1, "a", None, added),
            TreeEdit("s", 2, "a", added, moved),
            TreeEdit("s", 3, "a", moved, resized),
        ]
    )
    assert len(batch.edits) == 1
    assert batch.edits[0].operation is EditOperation.ADD
    assert batch.edits[0].old_tree is None
    assert batch.edits[0].new_tree == resized


def test_add_then_delete_is_removed_as_noop() -> None:
    added = tree("a", 10.0, 10.0)
    batch = coalesce_tree_edits(
        [
            TreeEdit("s", 1, "a", None, added),
            TreeEdit("s", 2, "a", added, None),
        ]
    )
    assert batch.edits == ()


def test_move_preserves_first_old_and_last_new_state() -> None:
    start = tree("a", 10.0, 10.0)
    middle = tree("a", 20.0, 20.0)
    end = tree("a", 30.0, 30.0)
    batch = coalesce_tree_edits(
        [
            TreeEdit("s", 1, "a", start, middle),
            TreeEdit("s", 2, "a", middle, end),
        ]
    )
    assert batch.edits[0].operation is EditOperation.MOVE
    assert batch.edits[0].old_tree == start
    assert batch.edits[0].new_tree == end


def test_batch_windows_merge_nearby_edits() -> None:
    batch = coalesce_tree_edits(
        [
            TreeEdit("s", 1, "a", None, tree("a", 100.0, 900.0)),
            TreeEdit("s", 2, "b", None, tree("b", 120.0, 880.0)),
        ]
    )
    windows = dirty_windows_for_batch(
        batch,
        grid=RasterGrid(500, 500, 2.0, origin_y_m=1000.0),
        sun_positions=[SunPosition(45.0, 180.0)],
        config=InfluenceConfig(
            lowest_sky_patch_altitude_deg=45.0,
            safety_margin_m=0.0,
            block_size_pixels=1,
        ),
        merge_gap_pixels=10,
    )
    assert len(windows) == 1


def test_rejects_duplicate_sequences() -> None:
    import pytest

    with pytest.raises(ValueError, match="unique"):
        coalesce_tree_edits(
            [
                TreeEdit("s", 1, "a", None, tree("a", 10.0, 10.0)),
                TreeEdit("s", 1, "b", None, tree("b", 20.0, 20.0)),
            ]
        )


def test_rejects_non_contiguous_edit_chain() -> None:
    import pytest

    first = tree("a", 10.0, 10.0)
    first_result = tree("a", 20.0, 20.0)
    unrelated_old = tree("a", 40.0, 40.0)
    final = tree("a", 50.0, 50.0)
    with pytest.raises(ValueError, match="non-contiguous"):
        coalesce_tree_edits(
            [
                TreeEdit("s", 1, "a", first, first_result),
                TreeEdit("s", 2, "a", unrelated_old, final),
            ]
        )
