import math

import pytest

from solweig_gpu.incremental.geometry import (
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
)


def tree(tree_id: str, x: float, y: float, height: float = 10.0) -> TreeSpec:
    return TreeSpec(tree_id, x, y, height, canopy_radius_m=4.0)


def test_svf_radius_matches_geometry() -> None:
    radius = estimate_svf_radius_m(10.0, 45.0)
    assert radius == pytest.approx(10.0)


def test_shadow_vector_uses_opposite_solar_azimuth() -> None:
    east, north = shadow_vector_m(
        10.0,
        SunPosition(altitude_deg=45.0, azimuth_deg=90.0),
    )
    assert east == pytest.approx(-10.0)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_shadow_vector_caps_low_sun_length() -> None:
    east, north = shadow_vector_m(
        20.0,
        SunPosition(altitude_deg=0.2, azimuth_deg=0.0),
        maximum_length_m=100.0,
    )
    assert math.hypot(east, north) == pytest.approx(100.0)


def test_elevation_offset_extends_influence_bounds() -> None:
    # On sloped terrain the canopy's reach is measured from the lowest
    # target surface: height + relief, not the bare height.
    config_flat = InfluenceConfig(
        lowest_sky_patch_altitude_deg=45.0,
        minimum_direct_sun_altitude_deg=45.0,
        safety_margin_m=0.0,
    )
    config_sloped = InfluenceConfig(
        lowest_sky_patch_altitude_deg=45.0,
        minimum_direct_sun_altitude_deg=45.0,
        safety_margin_m=0.0,
        elevation_offset_m=30.0,
    )
    suns = [SunPosition(45.0, 0.0)]
    x0, y0, x1, y1 = tree_influence_bounds_m(tree("t", 0.0, 0.0), suns, config_flat)
    sx0, sy0, sx1, sy1 = tree_influence_bounds_m(tree("t", 0.0, 0.0), suns, config_sloped)
    # 10 m tree at 45 degrees -> ~10 m; +30 m relief -> ~40 m reach.
    assert (x1 - x0) == pytest.approx(2 * 10.0, rel=0.02)
    assert (sx1 - sx0) == pytest.approx(2 * 40.0, rel=0.02)
    assert sx1 > x1 and sy0 < y0

    with pytest.raises(ValueError):
        InfluenceConfig(elevation_offset_m=-1.0)


def test_move_invalidates_old_and_new_locations() -> None:
    grid = RasterGrid(rows=500, cols=500, pixel_size_m=2.0, origin_y_m=1000.0)
    config = InfluenceConfig(
        lowest_sky_patch_altitude_deg=45.0,
        safety_margin_m=0.0,
        block_size_pixels=1,
    )
    window = dirty_window_for_edit(
        grid,
        old_tree=tree("t", 100.0, 900.0),
        new_tree=tree("t", 500.0, 500.0),
        sun_positions=[SunPosition(45.0, 180.0)],
        config=config,
    )
    assert window.col_start <= 45
    assert window.col_stop >= 255
    assert window.row_start <= 45
    assert window.row_stop >= 255


def test_window_is_block_aligned_and_clamped() -> None:
    grid = RasterGrid(rows=100, cols=100, pixel_size_m=2.0, origin_y_m=200.0)
    config = InfluenceConfig(
        lowest_sky_patch_altitude_deg=60.0,
        safety_margin_m=0.0,
        block_size_pixels=16,
    )
    window = dirty_window_for_edit(
        grid,
        old_tree=None,
        new_tree=tree("edge", 2.0, 198.0, height=5.0),
        sun_positions=[SunPosition(45.0, 180.0)],
        config=config,
    )
    assert window.row_start == 0
    assert window.col_start == 0
    assert window.row_stop % 16 == 0
    assert window.col_stop % 16 == 0


def test_choose_recompute_mode_uses_area_fraction() -> None:
    grid = RasterGrid(rows=100, cols=100, pixel_size_m=1.0)
    assert choose_recompute_mode(RasterWindow(0, 10, 0, 10), grid) is RecomputeMode.LOCAL
    assert choose_recompute_mode(RasterWindow(0, 60, 0, 60), grid) is RecomputeMode.FULL


def test_merge_windows_handles_transitive_overlap() -> None:
    windows = [
        RasterWindow(0, 10, 0, 10),
        RasterWindow(9, 20, 9, 20),
        RasterWindow(19, 30, 19, 30),
        RasterWindow(60, 70, 60, 70),
    ]
    merged = merge_windows(windows)
    assert merged == [RasterWindow(0, 30, 0, 30), RasterWindow(60, 70, 60, 70)]


def test_rejects_non_finite_geometry_values() -> None:
    import pytest

    with pytest.raises(ValueError, match="finite"):
        SunPosition(float("nan"), 180.0)
    with pytest.raises(ValueError, match="finite"):
        TreeSpec("bad", 0.0, 0.0, float("inf"), canopy_radius_m=2.0)
    with pytest.raises(ValueError, match="finite"):
        RasterGrid(rows=10, cols=10, pixel_size_m=float("nan"))


def test_raster_window_rejects_fractional_indices() -> None:
    import pytest

    with pytest.raises(TypeError, match="integer"):
        RasterWindow(0, 10.5, 0, 10)  # type: ignore[arg-type]
