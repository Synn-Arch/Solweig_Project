# SPDX-License-Identifier: GPL-3.0-only
"""R4 Phase A: persistent vegetation SVF occluder state + corridor re-march.

Covers the four BINDING conditions from the R4 design addenda
(docs/incremental_design_tool/realtime_collaboration/design/
r4-veg-svf-occlusion-state.md, ADDENDUM + ADDENDUM 2):

1. clamp-regime fallback — an edit batch falls back to the full replay iff
   the effective march amplitude changed AND the clamp
   (``bound > scene_amaxvalue``) is active on the smaller-amplitude scene,
   evaluated on FULL-TILE composed pre/post scenes; amplitude-constant
   transitions must NOT trigger it;
2. the per-patch re-march set is ``corridor_p (union of pre/post
   global-effective-amplitude march offsets) U C`` with ``C`` computed on
   composed surface diffs, and ``bush == 0`` asserted on pre AND post;
3. marches run on W2 ``_patch_march_window`` reach-expanded windows — never
   the raw corridor bbox. DEVIATION from the letter (ADDENDUM 3, adjudicated
   compliant-as-implemented): the march amplitude is the POST scene's
   full-tile effective amplitude, not the window crop — the fold consumes
   VALUES, and the windowed letter is value-divergent in one-step regimes;
4. after every corridor re-march (and before baseline packing) the float
   vegsh/vbsh planes must be within {0, 1} — on violation the batch falls
   back (the reviewer's vbsh==2.0 counterexample is reproduced and pinned).
   Regime level (review HIGH-1): a multi-step -> one-step amplitude
   transition refuses the batch, and a baseline already one-step anywhere
   refuses the pack — the 2.0 hazard is per-REGIME, not per-window.

Parity spine: every state-path result is compared BITWISE against an
independent full-tile replay twin (a fresh ``_recompute_veg_svf_window``
per comparison) and, on the real-cache fine site, against the solver
bundle / ``solve_window`` outputs with the state path disabled.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow, TreeSpec
from solweig_gpu.incremental.solver import (
    FullSceneTensors,
    StaleSvfError,
    compose_full_scene_tensors,
    effective_march_amplitude,
    read_window_for_write_window,
    window_svf_bundle,
    _patch_march_reach_pixels,
    _patch_march_window,
    _recompute_veg_svf_window,
    _sky_patch_geometry,
)
from solweig_gpu.incremental.trees import TreeLayer, rasterize_tree_patch
from solweig_gpu.shadow import shadow as shadow_fn

from tests.test_incremental_worker import (  # noqa: F401  (fixtures/helpers)
    SVF_BUNDLE_INDEX,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
    DATE_STR,
)

PATCHES, _RINGS = _sky_patch_geometry(2)
N_PATCHES = len(PATCHES)


# ---------------------------------------------------------------------------
# Synthetic stub-site plumbing (solver-level, no GDAL) — proofs' patterns
# ---------------------------------------------------------------------------


class _StubManifest:
    origin_x_m = 0.0

    def __init__(self, rows: int, pixel: float, sha: str) -> None:
        self.origin_y_m = float(rows * pixel)
        self.manifest_sha256 = sha


class StubCache:
    """Cache protocol the solver/state modules need, backed by arrays."""

    def __init__(
        self,
        rows: int,
        cols: int,
        pixel: float,
        building: np.ndarray,
        dem: np.ndarray,
        tree_base: np.ndarray,
        *,
        svf_patches: dict[str, np.ndarray] | None = None,
        sha: str = "stub-sha-1",
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.pixel_size_m = pixel
        self.building_dsm = building
        self.dem = dem
        self.tree_base = tree_base
        self.manifest = _StubManifest(rows, pixel, sha)
        self._svf_patches = svf_patches or {}
        self.site_id = "stub-site"
        self.tile_key = "0_0"
        self.model_version = "stub-model"
        self.patch_count = N_PATCHES

    @property
    def svf_patches(self) -> dict[str, np.ndarray]:
        return self._svf_patches

    def metadata(self) -> dict:
        return {
            "site_id": self.site_id,
            "tile_key": self.tile_key,
            "model_version": self.model_version,
            "manifest_sha256": self.manifest.manifest_sha256,
        }


def make_stub_site(
    rows: int,
    cols: int,
    pixel: float,
    building: np.ndarray,
    canopy: np.ndarray,
    *,
    dem: np.ndarray | None = None,
    sha: str = "stub-sha-1",
) -> tuple[StubCache, RasterGrid, TreeLayer]:
    if dem is None:
        dem = np.zeros((rows, cols), dtype=np.float32)
    cache = StubCache(
        rows, cols, pixel,
        building.astype(np.float32), dem.astype(np.float32),
        canopy.astype(np.float32), sha=sha,
    )
    grid = RasterGrid(rows, cols, pixel, 0.0, rows * pixel)
    layer = TreeLayer(cache.tree_base, grid, scenario_id="r4a")
    return cache, grid, layer


def scene_of(cache: StubCache, layer: TreeLayer) -> FullSceneTensors:
    return compose_full_scene_tensors(cache, layer)


def full_bits(cache: StubCache, scene: FullSceneTensors):
    """INDEPENDENT twin: fresh full-tile replay bits for this scene."""
    full = RasterWindow(0, cache.rows, 0, cache.cols)
    _scalars, vegsh, vbsh, _svftotal = _recompute_veg_svf_window(
        cache, scene, full, torch.zeros((cache.rows, cache.cols)),
        accumulate_window=full,
    )
    return vegsh.numpy(), vbsh.numpy()


def replay_cubes_as_cache_patches(cache: StubCache, scene: FullSceneTensors) -> dict:
    """Materialise the P2-style float32 patch cubes for a scene (twin run)."""
    vegsh, vbsh = full_bits(cache, scene)
    return {"vegshadowmat": vegsh, "vbshmat": vbsh, "shadowmat": np.zeros_like(vegsh)}


def effective_amplitude(scene: FullSceneTensors) -> float:
    return float(
        effective_march_amplitude(
            scene.a, scene.vegdsm, scene.vegdsm2, scene_amaxvalue=scene.amaxvalue
        )
    )


def relative_bound(scene: FullSceneTensors) -> float:
    return float(
        torch.maximum(
            torch.maximum(scene.a.max(), scene.vegdsm.max()), scene.vegdsm2.max()
        )
        - torch.min(scene.a)
    )


def pos(row: int, col: int, rows: int, pixel: float) -> tuple[float, float]:
    """World position of a cell centre (grid origin at (0, rows*pixel))."""
    return (col + 0.5) * pixel, (rows - row - 0.5) * pixel


def build_main_tile(*, rows: int = 60, pixel: float = 2.0) -> tuple[StubCache, RasterGrid, TreeLayer]:
    """Proofs' 60x60 main tile: 12 m NW block, 4 m SE block, four base trees."""
    dem = np.zeros((rows, rows), dtype=np.float32)
    building = dem.copy()
    building[4:12, 4:12] = 12.0
    building[40:50, 38:48] = 4.0
    grid = RasterGrid(rows, rows, pixel, 0.0, rows * pixel)
    canopy = np.zeros((rows, rows), dtype=np.float32)
    for tree in (
        TreeSpec("base1", *pos(30, 30, rows, pixel), 6.0, 3.0),
        TreeSpec("base2", *pos(12, 45, rows, pixel), 8.0, 3.5),
        TreeSpec("base3", *pos(44, 14, rows, pixel), 5.0, 2.5),
        TreeSpec("base4", *pos(50, 50, rows, pixel), 7.0, 2.0),
    ):
        canopy_patch, _trunk = rasterize_tree_patch(tree, grid, grid.full_window)
        np.maximum(canopy, canopy_patch, out=canopy)
    return make_stub_site(rows, rows, pixel, building, canopy)


def state_bits(state, patch_index: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        state.bit_plane("vegsh", patch_index),
        state.bit_plane("vbsh", patch_index),
    )


def assert_state_matches_replay(state, cache: StubCache, scene: FullSceneTensors) -> dict:
    """Bitwise twin assertion over every patch + non-vacuity counters."""
    vegsh_twin, vbsh_twin = full_bits(cache, scene)
    mismatches = 0
    for p in range(N_PATCHES):
        state_vegsh, state_vbsh = state_bits(state, p)
        assert np.array_equal(state_vegsh, vegsh_twin[:, :, p] != 0), f"vegsh patch {p}"
        assert np.array_equal(state_vbsh, vbsh_twin[:, :, p] != 0), f"vbsh patch {p}"
        mismatches += int((state_vegsh != (vegsh_twin[:, :, p] != 0)).sum())
        mismatches += int((state_vbsh != (vbsh_twin[:, :, p] != 0)).sum())
    return {
        "mismatches": mismatches,
        "twin_vegsh_cells": int((vegsh_twin != 0).sum()),
    }


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


class TestModuleContract:
    def test_public_surface(self) -> None:
        from solweig_gpu.incremental import veg_svf_state

        for name in (
            "VegOcclusionFallback",
            "VegOcclusionKey",
            "VegOcclusionState",
            "VegOcclusionStore",
            "SCHEMA_VERSION",
            "patch_geometry_id",
            "state_key_for",
            "build_baseline_state",
            "apply_edit_batch",
            "clamped_amplitude_change_reason",
            "march_offsets",
            "corridor_mask",
        ):
            assert hasattr(veg_svf_state, name), name

    def test_fallback_error_is_typed_with_reason(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import VegOcclusionFallback

        error = VegOcclusionFallback("clamp")
        assert isinstance(error, RuntimeError)
        assert error.reason == "clamp"


# ---------------------------------------------------------------------------
# Condition 2/3 machinery: march offsets + corridor masks
# ---------------------------------------------------------------------------


class TestMarchOffsetsAndCorridor:
    def test_corridor_mask_shifts_changed_cells_backwards(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import corridor_mask

        changed = np.zeros((5, 6), dtype=bool)
        changed[1, 2] = True
        # target t reads occluder t + (dx, dy); one offset (2, -1) marks
        # cells t with changed[t + (2, -1)]; the only preimage of (1, 2)
        # is (-1, 3) which is out of bounds -> empty corridor
        corridor = corridor_mask(changed, ((2, -1),))
        assert not corridor.any()

    def test_corridor_mask_marks_preimages(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import corridor_mask

        changed = np.zeros((6, 6), dtype=bool)
        changed[3, 3] = True
        corridor = corridor_mask(changed, ((2, 1), (1, 0), (0, 0)))
        assert corridor[1, 2]  # 3-2, 3-1
        assert corridor[2, 3]  # 3-1, 3-0
        assert corridor[3, 3]  # the 0-offset reads the target itself
        assert corridor.sum() == 3

    def test_offsets_execute_while_amplitude_bounds_dz(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import march_offsets

        altitude, azimuth = PATCHES[0][0], PATCHES[0][1]
        offsets = march_offsets(azimuth, altitude, 12.0, 0.5, 60, 60)
        assert len(offsets) >= 1
        # monotone in the amplitude: more amplitude, more (or equal) steps
        longer = march_offsets(azimuth, altitude, 24.0, 0.5, 60, 60)
        assert set(offsets) <= set(longer)
        assert len(longer) >= len(offsets)

    def test_offsets_match_shadow_executed_steps(self) -> None:
        """Offsets replicate shadow()'s while-loop exactly (first-step shift).

        shadow() shifts (dx, dy) at step 1 with dz_1 = ds * tan(alt)/scale;
        the loop stops when amax < dz. For a single-step regime the only
        offset must be the step-1 shift of a direct march probe.
        """
        from solweig_gpu.incremental.veg_svf_state import march_offsets

        altitude, azimuth = PATCHES[145][0], PATCHES[145][1]  # 78 deg ring
        offsets = march_offsets(azimuth, altitude, 12.0, 0.25, 60, 60)
        assert len(offsets) == 1
        dx, dy = offsets[0]
        degrees = torch.pi / 180.0
        az = float(azimuth)
        if az == 0.0:
            az = 1e-12
        az_rad = az * degrees
        alt_rad = float(altitude) * degrees
        ds = abs(1.0 / math.cos(az_rad)) if not (
            math.pi / 4 <= az_rad < 3 * math.pi / 4
            or 5 * math.pi / 4 <= az_rad < 7 * math.pi / 4
        ) else abs(1.0 / math.sin(az_rad))
        assert dz_steps(ds, alt_rad, 0.25, 1) > 12.0  # exactly one step runs
        # the shift direction follows shadow()'s branch structure
        assert abs(dx) == 1 or abs(dy) == 1


def dz_steps(ds: float, alt_rad: float, scale: float, index: int) -> float:
    return ds * index * (math.tan(alt_rad) / scale)


# ---------------------------------------------------------------------------
# Condition 1: clamp-regime guard predicate
# ---------------------------------------------------------------------------


class TestClampRegimeGuard:
    @staticmethod
    def _reason(scene_pre, scene_post):
        from solweig_gpu.incremental.veg_svf_state import clamped_amplitude_change_reason

        return clamped_amplitude_change_reason(scene_pre, scene_post)

    def _clamped_scene_pair(self):
        """Pre scene clamped (crown overhangs the 4 m block), plus a raiser."""
        cache, grid, layer = build_main_tile()
        overhang = TreeSpec("over", *pos(44, 43, 60, 2.0), 11.0, 4.0)
        layer.add_tree(overhang)  # vegdsm 15 > abs 12 -> clamp active
        scene_pre = scene_of(cache, layer)
        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 2.0), 18.0, 3.0))
        scene_post = scene_of(cache, layer)
        return cache, grid, layer, scene_pre, scene_post

    def test_raiser_on_clamped_scene_fires(self) -> None:
        _cache, _grid, _layer, scene_pre, scene_post = self._clamped_scene_pair()
        assert relative_bound(scene_pre) > float(scene_pre.amaxvalue)  # clamped
        assert effective_amplitude(scene_pre) != effective_amplitude(scene_post)
        reason = self._reason(scene_pre, scene_post)
        assert reason is not None and "clamp" in reason

    def test_dropper_on_clamped_after_scene_fires(self) -> None:
        # smaller-amplitude scene is the POST scene and it is clamped
        cache, grid, layer = build_main_tile()
        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 2.0), 18.0, 3.0))
        scene_pre = scene_of(cache, layer)  # unclamped (18 == bound)
        overhang = TreeSpec("over", *pos(44, 43, 60, 2.0), 11.0, 4.0)
        layer.add_tree(overhang)
        layer.delete_tree("tall")  # dropper 18 -> 12, clamp re-engages
        scene_post = scene_of(cache, layer)
        assert effective_amplitude(scene_pre) > effective_amplitude(scene_post)
        assert relative_bound(scene_post) > float(scene_post.amaxvalue)
        reason = self._reason(scene_pre, scene_post)
        assert reason is not None and "clamp" in reason

    def test_amplitude_constant_clamped_scene_does_not_fire(self) -> None:
        cache, grid, layer = build_main_tile()
        overhang = TreeSpec("over", *pos(44, 43, 60, 2.0), 11.0, 4.0)
        layer.add_tree(overhang)
        scene_pre = scene_of(cache, layer)
        assert relative_bound(scene_pre) > float(scene_pre.amaxvalue)  # clamped
        layer.add_tree(TreeSpec("plain", *pos(25, 25, 60, 2.0), 5.0, 2.0))
        scene_post = scene_of(cache, layer)
        assert relative_bound(scene_post) > float(scene_post.amaxvalue)  # still clamped
        assert effective_amplitude(scene_pre) == effective_amplitude(scene_post)
        assert self._reason(scene_pre, scene_post) is None

    def test_amplitude_change_unclamped_smaller_scene_does_not_fire(self) -> None:
        cache, grid, layer = build_main_tile()
        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 2.0), 18.0, 3.0))
        scene_pre = scene_of(cache, layer)
        layer.update_tree("tall", height_m=20.0)
        scene_post = scene_of(cache, layer)
        assert effective_amplitude(scene_pre) != effective_amplitude(scene_post)
        # both scenes unclamped (bound == abs on each)
        for scene in (scene_pre, scene_post):
            assert relative_bound(scene) <= float(scene.amaxvalue)
        assert self._reason(scene_pre, scene_post) is None

    def test_boundary_bound_equals_abs_does_not_fire(self) -> None:
        # strict >: bound == abs on the smaller scene must NOT fire
        cache, grid, layer = build_main_tile()
        # pre: tall 20 m tree (abs 20, bound 20); post: resize to 18 m
        # (abs 12 -> wait: keep the tall tree at 18 and add a co-dominant 18)
        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 2.0), 20.0, 3.0))
        scene_pre = scene_of(cache, layer)
        layer.update_tree("tall", height_m=18.0)
        layer.add_tree(TreeSpec("twin", *pos(35, 35, 60, 2.0), 18.0, 3.0))
        scene_post = scene_of(cache, layer)
        # the smaller scene has bound == abs exactly (18 == 18)
        smaller = (
            scene_pre
            if effective_amplitude(scene_pre) <= effective_amplitude(scene_post)
            else scene_post
        )
        assert relative_bound(smaller) == float(smaller.amaxvalue)
        assert effective_amplitude(scene_pre) != effective_amplitude(scene_post)
        assert self._reason(scene_pre, scene_post) is None


# ---------------------------------------------------------------------------
# Conditions 1+2+3 end to end: corridor re-march is bitwise vs the twin
# ---------------------------------------------------------------------------

class TestCorridorRemarchBitwise:
    @staticmethod
    def _baseline_state(cache, layer):
        from solweig_gpu.incremental.veg_svf_state import build_baseline_state

        scene = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        state = build_baseline_state(cache)
        # V0-style pin at this site: baseline bits == fresh replay bits
        report = assert_state_matches_replay(state, cache, scene)
        assert report["mismatches"] == 0
        assert report["twin_vegsh_cells"] > 0
        return scene, state

    def test_add_move_resize_delete_sequence_bitwise(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import apply_edit_batch

        cache, grid, layer = build_main_tile()
        scene, state = self._baseline_state(cache, layer)

        edits = [
            ("add", lambda: layer.add_tree(TreeSpec("t0", *pos(10, 50, 60, 2.0), 9.0, 3.0))),
            ("move", lambda: layer.move_tree(
                "t0", x_m=pos(50, 10, 60, 2.0)[0], y_m=pos(50, 10, 60, 2.0)[1]
            )),
            ("resize", lambda: layer.update_tree("t0", height_m=11.0, canopy_radius_m=4.0)),
            ("delete", lambda: layer.delete_tree("t0")),
        ]
        total_flips = 0
        total_remarched = 0
        for name, mutate in edits:
            mutate()
            scene = scene_of(cache, layer)
            state, telemetry = apply_edit_batch(state, cache, scene)
            assert telemetry["fallback_reason"] is None
            report = assert_state_matches_replay(state, cache, scene)
            assert report["mismatches"] == 0, name
            total_flips += telemetry["vegsh_flips"] + telemetry["vbsh_flips"]
            total_remarched += telemetry["patches_remarched"]
        # non-vacuity: the sequence re-marched patches and flipped bits
        assert total_remarched > 0
        assert total_flips > 0

    def test_repeat_edits_and_overlap_bitwise(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import apply_edit_batch

        cache, grid, layer = build_main_tile()
        _scene, state = self._baseline_state(cache, layer)
        # taller than base1's 6 m at the same cell, so the max-composed
        # canopy actually changes (a strictly-lower tree would be a no-op)
        tree = TreeSpec("r", *pos(30, 30, 60, 2.0), 8.0, 3.0)
        layer.add_tree(tree)
        scene = scene_of(cache, layer)
        state, telemetry = apply_edit_batch(state, cache, scene)
        assert telemetry["vegsh_flips"] + telemetry["vbsh_flips"] > 0
        assert_state_matches_replay(state, cache, scene)
        # edit -> undo -> re-edit (repeat-edit drift fence)
        layer.delete_tree("r")
        scene = scene_of(cache, layer)
        state, _ = apply_edit_batch(state, cache, scene)
        assert_state_matches_replay(state, cache, scene)
        layer.add_tree(tree)
        scene = scene_of(cache, layer)
        state, _ = apply_edit_batch(state, cache, scene)
        assert_state_matches_replay(state, cache, scene)

    def test_amplitude_raiser_and_dropper_bitwise_no_clamp(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import apply_edit_batch

        cache, grid, layer = build_main_tile()
        scene, state = self._baseline_state(cache, layer)
        assert effective_amplitude(scene) == 12.0
        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 2.0), 18.0, 3.0))
        scene = scene_of(cache, layer)
        assert effective_amplitude(scene) == 18.0  # raiser, unclamped
        state, telemetry = apply_edit_batch(state, cache, scene)
        assert telemetry["fallback_reason"] is None
        assert telemetry["vegsh_flips"] > 0
        assert_state_matches_replay(state, cache, scene)
        layer.update_tree("tall", height_m=20.0)
        scene = scene_of(cache, layer)
        state, _ = apply_edit_batch(state, cache, scene)
        assert_state_matches_replay(state, cache, scene)
        layer.delete_tree("tall")  # dropper 20 -> 12, unclamped smaller scene
        scene = scene_of(cache, layer)
        state, telemetry = apply_edit_batch(state, cache, scene)
        assert telemetry["fallback_reason"] is None
        assert telemetry["vegsh_flips"] > 0
        assert_state_matches_replay(state, cache, scene)

    def test_corner_tree_boundary_clamping_bitwise(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import apply_edit_batch

        cache, grid, layer = build_main_tile()
        _scene, state = self._baseline_state(cache, layer)
        layer.add_tree(TreeSpec("ne", *pos(1, 58, 60, 2.0), 9.0, 3.0))
        layer.add_tree(TreeSpec("sw", *pos(58, 1, 60, 2.0), 6.0, 2.5))
        scene = scene_of(cache, layer)
        state, _ = apply_edit_batch(state, cache, scene)
        assert_state_matches_replay(state, cache, scene)

    def test_lagged_batch_catches_up_bitwise(self) -> None:
        """A refused batch leaves the state behind; the next batch catches up."""
        from solweig_gpu.incremental.veg_svf_state import apply_edit_batch

        cache, grid, layer = build_main_tile()
        scene, state = self._baseline_state(cache, layer)
        # two edits applied as ONE lagged batch (state stays at baseline)
        layer.add_tree(TreeSpec("a", *pos(10, 50, 60, 2.0), 9.0, 3.0))
        layer.add_tree(TreeSpec("b", *pos(50, 10, 60, 2.0), 7.0, 2.0))
        scene = scene_of(cache, layer)
        state, telemetry = apply_edit_batch(state, cache, scene)
        assert telemetry["fallback_reason"] is None
        assert_state_matches_replay(state, cache, scene)

    def test_clamped_raiser_batch_falls_back(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            apply_edit_batch,
            build_baseline_state,
        )

        cache, grid, layer = build_main_tile()
        _scene, state = self._baseline_state(cache, layer)
        # batch 1: overhang (11 m crown on the 4 m block) — amplitude stays
        # 12 -> 12 (clamped but constant, Case A): the state ADVANCES onto
        # the clamped scene
        layer.add_tree(TreeSpec("over", *pos(44, 43, 60, 2.0), 11.0, 4.0))
        scene_clamped = scene_of(cache, layer)
        state, _telemetry = apply_edit_batch(state, cache, scene_clamped)
        # batch 2: the raiser on a clamped smaller scene -> Case C refusal
        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 2.0), 18.0, 3.0))
        scene_post = scene_of(cache, layer)
        with pytest.raises(VegOcclusionFallback) as excinfo:
            apply_edit_batch(state, cache, scene_post)
        assert "clamp" in str(excinfo.value)
        # the state itself was NOT advanced (still describes the pre scene)
        assert np.array_equal(state.canopy_scene, scene_clamped.canopy.numpy())

    def test_surface_equal_no_op_batch_keeps_bits(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import apply_edit_batch

        cache, grid, layer = build_main_tile()
        scene, state = self._baseline_state(cache, layer)
        state, telemetry = apply_edit_batch(state, cache, scene)
        assert telemetry["patches_remarched"] == 0
        assert telemetry["vegsh_flips"] == 0
        assert_state_matches_replay(state, cache, scene)


# ---------------------------------------------------------------------------
# Condition 2: bush precondition
# ---------------------------------------------------------------------------


class TestBushPrecondition:
    def test_nonzero_bush_falls_back(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            apply_edit_batch,
            build_baseline_state,
        )

        rows = 40
        dem = np.zeros((rows, rows), dtype=np.float32)
        # vegdem2 = 0.25*canopy + dem == 0 under a 6 m crown -> bush != 0
        dem[20:28, 20:28] = -1.5
        building = np.zeros((rows, rows), dtype=np.float32)
        building[4:10, 4:10] = 12.0
        cache, grid, layer = make_stub_site(
            rows, rows, 2.0, building,
            np.zeros((rows, rows), dtype=np.float32), dem=dem,
        )
        scene = scene_of(cache, layer)
        assert not bool(scene.bush.any())  # pre scene is bush-free
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        state = build_baseline_state(cache)
        layer.add_tree(TreeSpec("b", *pos(24, 24, rows, 2.0), 6.0, 3.0))
        scene_post = scene_of(cache, layer)
        assert bool(scene_post.bush.any()) or bool(scene.bush.any())
        with pytest.raises(VegOcclusionFallback) as excinfo:
            apply_edit_batch(state, cache, scene_post)
        assert "bush" in str(excinfo.value)

    def test_corridor_mask_union_includes_c_itself(self) -> None:
        """Condition 2: the re-march set is corridor U C (C is load-bearing).

        Two pinned facts on a real edit batch. (1) SUFFICIENCY: every flip
        of the independent full-tile twin lies inside corridor U C for
        every patch. (2) NON-VACUITY of the union term: for some patch the
        ray corridor alone does NOT cover C, so dropping ``U C`` would
        shrink the mandated re-march set. (Flip-site-level hits strictly
        inside C and strictly outside the corridor were NOT reproducible
        at global march amplitude on 2 m stub tiles — the union stays the
        binding closure, verified here at the set level.)
        """
        from solweig_gpu.incremental.veg_svf_state import (
            apply_edit_batch,
            build_baseline_state,
            corridor_mask,
            march_offsets,
        )

        cache, grid, layer = build_main_tile()
        scene = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        state = build_baseline_state(cache)
        # two big crowns on bare ground (baked-raster canopy: no named trees
        # to move), far from every block and base tree
        layer.add_tree(TreeSpec("mv", *pos(20, 33, 60, 2.0), 9.0, 6.0))
        layer.add_tree(TreeSpec("mv2", *pos(45, 20, 60, 2.0), 7.0, 6.0))
        scene_post = scene_of(cache, layer)
        vegsh_pre, vbsh_pre = full_bits(cache, scene)
        vegsh_post, vbsh_post = full_bits(cache, scene_post)
        c_mask = (
            (scene.vegdsm.numpy() != scene_post.vegdsm.numpy())
            | (scene.vegdsm2.numpy() != scene_post.vegdsm2.numpy())
        )
        assert c_mask.any()
        amp = max(effective_amplitude(scene), effective_amplitude(scene_post))
        total_flips = 0
        c_outside_corridor_patches = 0
        for p in range(N_PATCHES):
            altitude, azimuth = PATCHES[p][0], PATCHES[p][1]
            offsets = march_offsets(azimuth, altitude, amp, 1.0 / 2.0, 60, 60)
            corridor = corridor_mask(c_mask, offsets)
            flips = (vegsh_pre[:, :, p] != vegsh_post[:, :, p]) | (
                vbsh_pre[:, :, p] != vbsh_post[:, :, p]
            )
            total_flips += int(flips.sum())
            # (1) sufficiency: no flip may escape corridor U C
            closure = corridor | c_mask
            assert not (flips & ~closure).any(), f"patch {p} flip outside closure"
            # (2) count patches whose corridor alone misses C cells
            if (c_mask & ~corridor).any():
                c_outside_corridor_patches += 1
        assert total_flips > 0
        assert c_outside_corridor_patches > 0, "union term U C is load-bearing"
        # and the state path (corridor U C) reproduces the twin bitwise
        state, telemetry = apply_edit_batch(state, cache, scene_post)
        assert telemetry["fallback_reason"] is None
        assert_state_matches_replay(state, cache, scene_post)


# ---------------------------------------------------------------------------
# Condition 3: reach-expanded windows, never the raw corridor bbox
# ---------------------------------------------------------------------------


class TestReachExpandedWindows:
    def test_raw_bbox_diverges_where_expanded_window_matches(self) -> None:
        """The 19/59 claim: raw corridor bboxes diverge; the state matches."""
        from solweig_gpu.incremental.veg_svf_state import (
            apply_edit_batch,
            build_baseline_state,
            corridor_mask,
            march_offsets,
        )

        cache, grid, layer = build_main_tile()
        scene = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        state = build_baseline_state(cache)
        layer.add_tree(TreeSpec("demo", *pos(30, 30, 60, 2.0), 10.0, 3.0))
        scene_post = scene_of(cache, layer)
        vegsh_twin, vbsh_twin = full_bits(cache, scene_post)

        c_mask = (scene.vegdsm.numpy() != scene_post.vegdsm.numpy()) | (
            scene.vegdsm2.numpy() != scene_post.vegdsm2.numpy()
        )
        amp = max(effective_amplitude(scene), effective_amplitude(scene_post))
        scale = 0.5
        full = RasterWindow(0, 60, 0, 60)
        raw_diverging = 0
        expanded_checked = 0
        for index in range(0, 31):  # the 6 deg ring (worst offenders)
            altitude, azimuth = PATCHES[index][0], PATCHES[index][1]
            offsets = march_offsets(azimuth, altitude, amp, scale, 60, 60)
            closure = corridor_mask(c_mask, offsets) | c_mask
            if not closure.any():
                continue
            rs, cs = np.nonzero(closure)
            bbox = RasterWindow(int(rs.min()), int(rs.max()) + 1, int(cs.min()), int(cs.max()) + 1)
            reach = _patch_march_reach_pixels(amp, scale, float(altitude))
            march_window = _patch_march_window(
                bbox, full, azimuth_deg=float(azimuth), reach_pixels=reach
            )
            assert (march_window.height, march_window.width) != (0, 0)

            def crop(t, w):
                return t[w.row_start : w.row_stop, w.col_start : w.col_stop]

            for label, window in (("raw", bbox), ("expanded", march_window)):
                a_w = crop(scene_post.a, window)
                eff_w = effective_march_amplitude(
                    a_w,
                    crop(scene_post.vegdsm, window),
                    crop(scene_post.vegdsm2, window),
                    scene_amaxvalue=scene_post.amaxvalue,
                )
                _sh, vegsh_w, vbsh_w = shadow_fn(
                    eff_w, a_w,
                    crop(scene_post.vegdsm, window),
                    crop(scene_post.vegdsm2, window),
                    crop(scene_post.bush, window),
                    azimuth, altitude, scale,
                )
                vegsh_w = vegsh_w.numpy()
                vbsh_w = vbsh_w.numpy()
                sl = (
                    slice(bbox.row_start - window.row_start, bbox.row_stop - window.row_start),
                    slice(bbox.col_start - window.col_start, bbox.col_stop - window.col_start),
                )
                ref_vegsh = vegsh_twin[bbox.row_start : bbox.row_stop, bbox.col_start : bbox.col_stop, index]
                ref_vbsh = vbsh_twin[bbox.row_start : bbox.row_stop, bbox.col_start : bbox.col_stop, index]
                bad = int((vegsh_w[sl] != ref_vegsh).sum()) + int((vbsh_w[sl] != ref_vbsh).sum())
                if label == "raw" and bad:
                    raw_diverging += 1
                if label == "expanded":
                    expanded_checked += 1
                    assert bad == 0, f"expanded window diverged at patch {index}"
        # the state path is bitwise (the real assertion) and the raw bbox
        # demonstrably diverges on at least one 6-degree patch
        state, telemetry = apply_edit_batch(state, cache, scene_post)
        assert telemetry["fallback_reason"] is None
        assert_state_matches_replay(state, cache, scene_post)
        assert raw_diverging > 0, "expected the raw bbox to diverge (condition 3)"


# ---------------------------------------------------------------------------
# Condition 4: the vbsh {0,1} fence (reviewer counterexample reproduced)
# ---------------------------------------------------------------------------


def build_reviewer_counterexample_scene():
    """60x60, 4 m pixels, 12 m building, 6 m/8 m crowns: vbsh==2.0 at patch 148."""
    rows = 60
    pixel = 4.0
    dem = np.zeros((rows, rows), dtype=np.float32)
    building = dem.copy()
    building[4:12, 4:12] = 12.0
    building[40:50, 38:48] = 4.0
    grid = RasterGrid(rows, rows, pixel, 0.0, rows * pixel)
    canopy = np.zeros((rows, rows), dtype=np.float32)
    # sub-cell crown centres, exactly the probe coordinates that reproduce
    # the reviewer's 9 non-binary patches (patch 148 included)
    for tree in (
        TreeSpec("c1", 30.5 * pixel, (rows - 30.5) * pixel, 6.0, 3.0),
        TreeSpec("c2", 45.0 * pixel, (rows - 45.0) * pixel, 8.0, 3.0),
    ):
        canopy_patch, _trunk = rasterize_tree_patch(tree, grid, grid.full_window)
        np.maximum(canopy, canopy_patch, out=canopy)
    return make_stub_site(rows, rows, pixel, building, canopy)


class TestVbshBinaryFence:
    def test_reviewer_counterexample_exists_at_patch_148(self) -> None:
        """The march genuinely produces vbsh==2.0 (non-vacuity of the fence)."""
        cache, grid, layer = build_reviewer_counterexample_scene()
        scene = scene_of(cache, layer)
        eff = effective_amplitude(scene)
        altitude, azimuth = PATCHES[148][0], PATCHES[148][1]
        _sh, vegsh, vbsh = shadow_fn(
            torch.tensor(eff), scene.a, scene.vegdsm, scene.vegdsm2, scene.bush,
            azimuth, altitude, 1.0 / 4.0,
        )
        assert float(vbsh.max()) == 2.0
        assert int((vbsh == 2.0).sum()) > 0

    def test_baseline_packing_refuses_non_binary_loudly(self) -> None:
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            build_baseline_state,
        )

        cache, grid, layer = build_reviewer_counterexample_scene()
        # the P2-style cubes for this scene carry the 2.0 values
        scene = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        with pytest.raises(VegOcclusionFallback) as excinfo:
            build_baseline_state(cache)
        assert "binary" in str(excinfo.value) or "vbsh" in str(excinfo.value)

    def test_pack_refuses_non_binary_values(self) -> None:
        from solweig_gpu.incremental.bitmask import pack_visibility

        plane = np.zeros((4, 4, 3), dtype=np.float32)
        plane[1, 1, 0] = 2.0
        with pytest.raises(ValueError):
            pack_visibility(plane)

    def test_corridor_remarch_fence_fires_on_new_counterexample(self) -> None:
        """A clean 4 m site + the counterexample crowns: the refusal is loud.

        Post-HIGH-1 this scene is refused EARLIER, at pack: 12 m amplitude
        at 4 m pixels already puts the 78-deg ring in the one-step regime
        (dz_1 = 18.8 m > 12 m), and any scene whose re-march can produce
        2.0 values is itself one-step somewhere — structurally unpackable.
        Two pins: (1) the pack-time regime refusal keeps the routing loud;
        (2) the per-window value fence itself still fires on non-binary
        planes (unit level, _fence_binary_planes).
        """
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            _fence_binary_planes,
            build_baseline_state,
        )

        rows = 60
        pixel = 4.0
        dem = np.zeros((rows, rows), dtype=np.float32)
        building = dem.copy()
        building[4:12, 4:12] = 12.0
        grid = RasterGrid(rows, rows, pixel, 0.0, rows * pixel)
        cache, grid, layer = make_stub_site(
            rows, rows, pixel, building, np.zeros((rows, rows), dtype=np.float32)
        )
        scene = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        # (1) pack-time regime refusal (the 2.0-producing batches this test
        # originally exercised at apply time are precluded here outright)
        with pytest.raises(VegOcclusionFallback, match="one-step"):
            build_baseline_state(cache)
        # (2) the per-window value fence still fires on 2.0-class planes
        vegsh = torch.zeros((4, 4, 1))
        vbsh = torch.zeros((4, 4, 1))
        vbsh[1, 1, 0] = 2.0
        with pytest.raises(VegOcclusionFallback, match="non-binary vbsh"):
            _fence_binary_planes(vegsh, vbsh, context="unit pin")


# ---------------------------------------------------------------------------
# Fold + bundle seam parity on a real 2 m-pixel cache (state path ON)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regime-level condition-4 fence (review HIGH-1): the vbsh==2.0 hazard is
# per-REGIME, not per-window — a multi-step -> one-step amplitude transition
# (unclamped dropper, Case B, condition 1 correctly silent) leaves oracle
# vbsh==2.0 cells at UNCHANGED trees outside every re-marched window.
# ---------------------------------------------------------------------------


class TestRegimeTransitionFence:
    def _transition_tile(self):
        """Reviewer repro: counterexample tile + a 27 m tree to delete.

        Pre scene (27 m effective amplitude): every patch multi-step, cubes
        binary, packable. Deleting the tall tree drops the effective
        amplitude to 12 m — one-step regime at the 78-deg ring — and the
        unchanged 6/8 m crowns (>= 15 px away) then produce oracle
        vbsh==2.0 cells NO corridor window ever covers.

        T is BAKED into the baseline canopy (a layer-added tree cannot be
        deleted — baked rasters carry no named trees — and an unbaked T
        leaves the packed baseline at 12 m, which the HIGH-1 pack fence
        refuses). Returns (cache, grid, layer, canopy_without_T).
        """
        cache, grid, layer = build_reviewer_counterexample_scene()
        canopy_pre = cache.tree_base.copy()  # crowns only (12 m scene)
        patch, _trunk = rasterize_tree_patch(
            TreeSpec("T", 45.0 * 4.0, (60 - 10.5) * 4.0, 27.0, 3.0),
            grid,
            grid.full_window,
        )
        canopy_with_t = np.maximum(canopy_pre, patch).astype(np.float32)
        cache2, grid2, layer2 = make_stub_site(
            60, 60, 4.0, cache.building_dsm, canopy_with_t, dem=cache.dem
        )
        return cache2, grid2, layer2, canopy_pre

    def test_multi_to_one_step_dropper_refuses_batch(self) -> None:
        from solweig_gpu.incremental.solver import _compose_scene_from_canopy
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            apply_edit_batch,
            build_baseline_state,
            corridor_mask,
            march_offsets,
        )

        cache, grid, layer, canopy_pre = self._transition_tile()
        scene_pre = scene_of(cache, layer)
        assert effective_amplitude(scene_pre) == 27.0
        vegsh_pre, vbsh_pre = full_bits(cache, scene_pre)
        assert not (vbsh_pre == 2.0).any()  # multi-step baseline is binary
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene_pre))
        state = build_baseline_state(cache)
        state, tel = apply_edit_batch(state, cache, scene_pre)
        assert tel["fallback_reason"] is None

        scene_post = _compose_scene_from_canopy(cache, canopy_pre)  # T gone
        assert effective_amplitude(scene_post) == 12.0
        # the oracle post scene genuinely carries the one-step 2.0 values at
        # the unchanged crowns (severity documentation: this is REAL)
        _veg_post, vbsh_post = full_bits(cache, scene_post)
        two = vbsh_post == 2.0
        assert two.any()
        one_step_patches = sorted(set(np.nonzero(two)[2].tolist()))
        # ...and at least one 2.0 patch has 2.0 cells OUTSIDE the corridor
        # closure, so the per-window fence CANNOT catch them (the regime
        # fence below is the only guard)
        C = (scene_pre.vegdsm.numpy() != scene_post.vegdsm.numpy()) | (
            scene_pre.vegdsm2.numpy() != scene_post.vegdsm2.numpy()
        )
        amp = max(effective_amplitude(scene_pre), effective_amplitude(scene_post))
        outside = 0
        for p in one_step_patches:
            altitude, azimuth = PATCHES[p][0], PATCHES[p][1]
            offsets = march_offsets(azimuth, altitude, amp, 0.25, 60, 60)
            closure = corridor_mask(C, offsets)
            closure |= C
            outside += int((two[:, :, p] & ~closure).sum())
        assert outside > 0

        with pytest.raises(VegOcclusionFallback, match="one-step"):
            apply_edit_batch(state, cache, scene_post)
        # the refusal structurally precedes mutation — the state is not
        # advanced onto the refused scene (review T-nit-2)
        assert np.array_equal(state.canopy_scene, scene_pre.canopy.numpy())

    def test_baseline_pack_refuses_one_step_regime(self) -> None:
        """A baseline already in the one-step regime anywhere refuses pack.

        8 m amplitude at 4 m pixels: dz_1 at the 78-deg ring is 18.8 m, so
        the baseline march is one-step there. The cubes are all-zero binary
        (no canopy at all), so the VALUE fence alone would pass — the
        regime check refuses before any 2.0 can appear in a later batch.
        """
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            build_baseline_state,
        )

        rows = 60
        building = np.zeros((rows, rows), dtype=np.float32)
        building[4:12, 4:12] = 8.0
        cache, grid, layer = make_stub_site(
            rows, rows, 4.0, building, np.zeros((rows, rows), dtype=np.float32)
        )
        scene = scene_of(cache, layer)
        assert effective_amplitude(scene) == 8.0
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        with pytest.raises(VegOcclusionFallback, match="one-step"):
            build_baseline_state(cache)

    def test_growth_and_constant_amplitude_stay_applied(self) -> None:
        """The regime fence must not over-refuse: growth and constant apply."""
        from solweig_gpu.incremental.veg_svf_state import (
            apply_edit_batch,
            build_baseline_state,
        )

        cache, grid, layer, _canopy_pre = self._transition_tile()
        scene_pre = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene_pre))
        state = build_baseline_state(cache)
        state, _ = apply_edit_batch(state, cache, scene_pre)  # no-op pack-in
        # constant: another tree below the 27 m ceiling
        layer.add_tree(TreeSpec("k", *pos(40, 40, 60, 4.0), 20.0, 3.0))
        scene_post = scene_of(cache, layer)
        assert effective_amplitude(scene_post) == 27.0
        state, tel = apply_edit_batch(state, cache, scene_post)
        assert tel["fallback_reason"] is None
        assert_state_matches_replay(state, cache, scene_post)
        # growth: a taller tree raises the amplitude — regimes only widen
        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 4.0), 35.0, 3.0))
        scene_tall = scene_of(cache, layer)
        assert effective_amplitude(scene_tall) == 35.0
        state, tel = apply_edit_batch(state, cache, scene_tall)
        assert tel["fallback_reason"] is None
        assert_state_matches_replay(state, cache, scene_tall)


# ---------------------------------------------------------------------------
# Non-square tiles: march_offsets takes (sizex=rows, sizey=cols) like
# shadow() (review MEDIUM-1: the state layer passed them transposed, losing
# corridor steps on non-square grids).
# ---------------------------------------------------------------------------


class TestNonSquareMarchGeometry:
    def test_offsets_bind_rows_first(self) -> None:
        """Spec pin: |dx| is bounded by sizex (rows), |dy| by sizey (cols)."""
        from solweig_gpu.incremental.veg_svf_state import march_offsets

        # west march (az 0): dx grows — the loop checks the PREVIOUS shift,
        # so the last executed step reaches |dx| == sizex and stops there
        offsets = march_offsets(PATCHES[0][1], PATCHES[0][0], 10_000.0, 0.25, 5, 1000)
        assert offsets, "step 1 always executes"
        assert [dx for dx, _dy in offsets] == [-1, -2, -3, -4, -5]
        # diagonal march near az 90: dy grows — must stop at sizey
        diagonal = next(
            i for i, t in enumerate(PATCHES) if abs(float(t[1]) - 90.0) < 15.0
        )
        offsets = march_offsets(
            PATCHES[diagonal][1], PATCHES[diagonal][0], 10_000.0, 0.25, 1000, 7
        )
        assert all(abs(dy) <= 7 for _dx, dy in offsets)
        assert len(offsets) == 7

    def test_non_square_tile_state_matches_replay(self) -> None:
        """40x80 @4 m: the corridor must cover shadow()'s executed steps.

        A 27 m tree near the west edge; 6-deg diagonal steps reach ~64
        cells — beyond the transposed bound, so corridor cells far east of
        the canopy were missed and their bits went stale.
        """
        from solweig_gpu.incremental.veg_svf_state import (
            apply_edit_batch,
            build_baseline_state,
        )

        rows, cols, pixel = 40, 80, 4.0
        building = np.zeros((rows, cols), dtype=np.float32)
        # 28 m baseline: packable at 4 m pixels (the largest first-step dz
        # over the actual patches is 23.18 m — the steepest diagonal 78-deg
        # patch; 12/20 m baselines are one-step somewhere and the HIGH-1
        # pack fence refuses them)
        building[4:12, 4:12] = 28.0
        cache, grid, layer = make_stub_site(
            rows, cols, pixel, building, np.zeros((rows, cols), dtype=np.float32)
        )
        scene = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        state = build_baseline_state(cache)
        layer.add_tree(TreeSpec("T", 5.5 * pixel, (rows - 10.5) * pixel, 27.0, 3.0))
        scene_post = scene_of(cache, layer)
        state, tel = apply_edit_batch(state, cache, scene_post)
        assert tel["fallback_reason"] is None
        assert tel["vegsh_flips"] + tel["vbsh_flips"] > 0
        assert_state_matches_replay(state, cache, scene_post)


FINE_ROWS = FINE_COLS = 96
FINE_PIXEL = 2.0
FINE_ORIGIN = (520000.0, 3760000.0)
FINE_TREES = (
    TreeSpec("f1", FINE_ORIGIN[0] + 24.5 * FINE_PIXEL, FINE_ORIGIN[1] - 24.5 * FINE_PIXEL, 3.0, 2.0),
    TreeSpec("f2", FINE_ORIGIN[0] + 70.5 * FINE_PIXEL, FINE_ORIGIN[1] - 66.5 * FINE_PIXEL, 2.5, 2.0),
)


class FineSite:
    """Real prepared site + cache at 2 m pixels (fence-clean by construction)."""


@pytest.fixture(scope="module")
def fine_site(tmp_path_factory):
    root = tmp_path_factory.mktemp("fine_site")
    grid, site = _make_prepared_site(
        root / "site",
        rows=FINE_ROWS, cols=FINE_COLS, pixel=FINE_PIXEL,
        origin=FINE_ORIGIN, epsg=32616, base_trees=FINE_TREES, met_hours=range(24),
    )
    _compute_baseline_svf(site)
    met_path = site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
    cache = _build_cache(site, root / "cache", met_path=met_path, site_id="fine")
    return SimpleNamespaceShim(cache=cache, grid=grid, site=site, met_path=met_path)


class SimpleNamespaceShim:
    def __init__(self, **kw):
        self.__dict__.update(kw)


READ_WINDOW = RasterWindow(20, 70, 20, 70)
# strictly inside READ_WINDOW (the bundle refuses an escaping cube window)
CUBE_WINDOW = RasterWindow(30, 60, 30, 65)
# full-tile read extent: at an arbitrary sub-window the replay's marches
# clamp at the read edge (not halo-exact), so bundle twins compare at the
# extent where both paths are full-tile exact (solve_window twins below use
# the exact-influence read window instead)
FULL_WINDOW = RasterWindow(0, FINE_ROWS, 0, FINE_COLS)


class TestBundleFoldParity:
    def test_baseline_pack_matches_replay_on_real_cache(self, fine_site) -> None:
        from solweig_gpu.incremental.veg_svf_state import build_baseline_state

        cache = fine_site.cache
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        scene = compose_full_scene_tensors(cache, layer)
        state = build_baseline_state(cache)
        vegsh_twin, vbsh_twin = full_bits(cache, scene)
        for p in range(N_PATCHES):
            assert np.array_equal(
                state.bit_plane("vegsh", p), vegsh_twin[:, :, p] != 0
            ), f"vegsh patch {p}"
            assert np.array_equal(
                state.bit_plane("vbsh", p), vbsh_twin[:, :, p] != 0
            ), f"vbsh patch {p}"
        assert (vegsh_twin != 0).sum() > 0

    def test_fold_anchors_p2_svf_calculator(self, fine_site) -> None:
        """Oracle anchor for the fold constants (review NIT-1).

        Every other test in this file compares shared-fold-against-shared-
        fold, so mutating _fold_veg_svf_from_planes (transmissivity 0.03,
        the 3.0459e-4 floor, the >1 clamps) broke nothing. This test pins
        the veg_changed=True replay bundle against the P2 cache's
        svf_calculator rasters on the unedited baseline scene — an
        INDEPENDENT producer — so a touched constant now diverges.
        """
        cache = fine_site.cache
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        scene = compose_full_scene_tensors(cache, layer)
        bundle = window_svf_bundle(
            cache, scene, FULL_WINDOW, veg_changed=True, cube_window=FULL_WINDOW,
        )
        # bundle layout mirrors svf_calculator's return order
        checked = {
            1: "svfaveg",
            9: "svfSaveg",
            11: "svfveg",
            13: "svfWaveg",
            18: "svftotal",
        }
        for position, name in checked.items():
            oracle = np.array(cache.svf[name])
            assert np.array_equal(
                bundle[position].numpy(), oracle
            ), f"fold diverged from the P2 svf_calculator raster {name!r}"

    def test_state_bundle_bitwise_vs_replay(self, fine_site) -> None:
        from solweig_gpu.incremental.veg_svf_state import (
            apply_edit_batch,
            build_baseline_state,
        )

        cache = fine_site.cache
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        layer.add_tree(TreeSpec("t1", *(FINE_ORIGIN[0] + 48.5 * FINE_PIXEL,
                                        FINE_ORIGIN[1] - 48.5 * FINE_PIXEL), 6.0, 5.0))
        scene = compose_full_scene_tensors(cache, layer)
        state, telemetry = apply_edit_batch(build_baseline_state(cache), cache, scene)
        assert telemetry["fallback_reason"] is None
        assert telemetry["patches_remarched"] > 0
        assert telemetry["vegsh_flips"] > 0

        with_state = window_svf_bundle(
            cache, scene, FULL_WINDOW, veg_changed=True,
            cube_window=CUBE_WINDOW, veg_state=state,
        )
        replay = window_svf_bundle(
            cache, scene, FULL_WINDOW, veg_changed=True, cube_window=CUBE_WINDOW,
        )
        sl = (
            slice(CUBE_WINDOW.row_start - FULL_WINDOW.row_start,
                  CUBE_WINDOW.row_stop - FULL_WINDOW.row_start),
            slice(CUBE_WINDOW.col_start - FULL_WINDOW.col_start,
                  CUBE_WINDOW.col_stop - FULL_WINDOW.col_start),
        )
        for name in (
            "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
            "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
            "svftotal",
        ):
            position = SVF_BUNDLE_INDEX[name]
            assert np.array_equal(
                with_state[position].numpy(), replay[position].numpy(), equal_nan=True
            ), f"scalar {name} diverged"
        for name in ("vegshmat", "vbshvegshmat"):
            position = SVF_BUNDLE_INDEX[name]
            assert with_state[position].shape == replay[position].shape
            assert np.array_equal(
                with_state[position].numpy(), replay[position].numpy()
            ), f"cube {name} diverged"
        # padding outside the cube window is exactly zero (bundle contract)
        padded = with_state[SVF_BUNDLE_INDEX["svfveg"]].numpy().copy()
        padded[sl] = 0.0
        assert not padded.any()

    def test_seam_refuses_stale_scene_loudly(self, fine_site) -> None:
        from solweig_gpu.incremental.veg_svf_state import build_baseline_state

        cache = fine_site.cache
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        scene = compose_full_scene_tensors(cache, layer)
        state = build_baseline_state(cache)
        layer.add_tree(TreeSpec("t2", *(FINE_ORIGIN[0] + 30.5 * FINE_PIXEL,
                                        FINE_ORIGIN[1] - 30.5 * FINE_PIXEL), 5.0, 3.0))
        edited = compose_full_scene_tensors(cache, layer)
        with pytest.raises(StaleSvfError):
            window_svf_bundle(
                cache, edited, READ_WINDOW, veg_changed=True,
                cube_window=CUBE_WINDOW, veg_state=state,
            )

    def test_seam_refuses_wrong_key_loudly(self, fine_site) -> None:
        from solweig_gpu.incremental import veg_svf_state

        cache = fine_site.cache
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        scene = compose_full_scene_tensors(cache, layer)
        state = veg_svf_state.build_baseline_state(cache)
        other = fine_site.cache
        tampered = SimpleNamespaceShim(
            site_id="other-site",
            tile_key=other.tile_key,
            model_version=other.model_version,
            manifest=SimpleNamespaceShim(manifest_sha256="deadbeef"),
            metadata=lambda: {
                "site_id": "other-site",
                "tile_key": other.tile_key,
                "manifest_sha256": "deadbeef",
            },
        )
        with pytest.raises(StaleSvfError):
            state.validate_against(tampered, scene)


# ---------------------------------------------------------------------------
# solve_window end-to-end twin (state path vs replay path)
# ---------------------------------------------------------------------------


class TestSolveWindowTwin:
    def test_solve_window_bitwise_with_state(self, fine_site) -> None:
        from solweig_gpu.incremental.solver import load_site_forcing, solve_window
        from solweig_gpu.incremental.veg_svf_state import (
            apply_edit_batch,
            build_baseline_state,
        )

        cache = fine_site.cache
        forcing = load_site_forcing(
            cache, site_dir=fine_site.site, selected_date_str=DATE_STR
        )
        write_window = RasterWindow(36, 60, 40, 70)
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        layer.add_tree(TreeSpec("t1", *(FINE_ORIGIN[0] + 48.5 * FINE_PIXEL,
                                        FINE_ORIGIN[1] - 48.5 * FINE_PIXEL), 6.0, 5.0))
        scene = compose_full_scene_tensors(cache, layer)
        state, _ = apply_edit_batch(build_baseline_state(cache), cache, scene)
        read_window = read_window_for_write_window(
            write_window, cache, scene, forcing
        )

        common = dict(
            read_window=read_window,
            write_window=write_window,
            forcing=forcing,
            requested_variables=("utci", "tmrt", "shadow"),
            veg_changed=True,
            time_stop=1,
            scene=scene,
        )
        with_state = solve_window(cache, layer, veg_state=state, **common)
        without = solve_window(cache, layer, **common)
        for name, arrays in with_state.items():
            assert np.array_equal(arrays, without[name], equal_nan=True), name


# ---------------------------------------------------------------------------
# Store lifecycle: load-or-build, catch-up, commit seam, persistence
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    @staticmethod
    def _store(cache, root):
        from solweig_gpu.incremental.veg_svf_state import VegOcclusionStore

        return VegOcclusionStore(cache, state_root=root)

    def test_prepare_builds_baseline_and_applies_batch(self, fine_site, tmp_path) -> None:
        cache = fine_site.cache
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        store = self._store(cache, tmp_path / "veg_state")
        scene = compose_full_scene_tensors(cache, layer)
        state, telemetry = store.prepare(scene)
        assert state is not None
        assert telemetry["baseline_packed"] is True
        assert telemetry["fallback_reason"] is None
        layer.add_tree(TreeSpec("t1", *(FINE_ORIGIN[0] + 48.5 * FINE_PIXEL,
                                        FINE_ORIGIN[1] - 48.5 * FINE_PIXEL), 6.0, 5.0))
        scene = compose_full_scene_tensors(cache, layer)
        state, telemetry = store.prepare(scene)
        assert telemetry["fallback_reason"] is None
        assert telemetry["patches_remarched"] > 0
        assert telemetry["baseline_packed"] is False
        # committed state survives as the in-memory current state
        store.commit(state)
        again, telemetry = store.prepare(scene)
        assert telemetry["baseline_packed"] is False
        assert telemetry["patches_remarched"] == 0  # no catch-up needed
        assert again is not None

    def test_prepare_routes_fallback_without_raising(self, fine_site, tmp_path) -> None:
        cache = fine_site.cache
        store = self._store(cache, tmp_path / "veg_state")
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        scene = compose_full_scene_tensors(cache, layer)
        _state, _tel = store.prepare(scene)
        # force the clamp-regime refusal: craft a scene the guard rejects by
        # tampering the store's current state scene (lagged clamped raiser)
        store._current = None  # rebuild from baseline on the next prepare
        layer.add_tree(
            TreeSpec("over", FINE_ORIGIN[0] + 11.5 * FINE_PIXEL,
                     FINE_ORIGIN[1] - 13.5 * FINE_PIXEL, 8.0, 6.0)
        )  # overhangs the 12 m NW block -> vegdsm 20 > abs 12 -> clamp
        scene_clamped = compose_full_scene_tensors(cache, layer)
        state_clamped, _tel = store.prepare(scene_clamped)
        assert state_clamped is not None
        # publication semantics: the clamped scene only becomes the state's
        # pre scene once committed (an uncommitted advance is discarded)
        store.commit(state_clamped)
        layer.add_tree(TreeSpec("tall", *(FINE_ORIGIN[0] + 50.5 * FINE_PIXEL,
                                          FINE_ORIGIN[1] - 50.5 * FINE_PIXEL), 18.0, 3.0))
        scene_raiser = compose_full_scene_tensors(cache, layer)
        state, telemetry = store.prepare(scene_raiser)
        assert state is None
        assert telemetry["fallback_reason"] is not None
        assert "clamp" in telemetry["fallback_reason"]

    def test_persisted_state_round_trip_and_checksums(self, fine_site, tmp_path) -> None:
        from solweig_gpu.incremental.veg_svf_state import VegOcclusionStateError

        cache = fine_site.cache
        root = tmp_path / "veg_state"
        store = self._store(cache, root)
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        scene = compose_full_scene_tensors(cache, layer)
        state, _ = store.prepare(scene)
        store.commit(state)
        files = sorted(p.name for p in root.rglob("*") if p.is_file())
        assert files
        fresh = self._store(cache, root)
        loaded, telemetry = fresh.prepare(scene)
        assert loaded is not None
        assert telemetry["baseline_packed"] is False
        # corrupt the persisted bytes -> typed refusal -> rebuild path
        for path in root.rglob("vegshadowmat_packed.npy"):
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 0xFF
            path.write_bytes(bytes(data))
        fresh2 = self._store(cache, root)
        state2, telemetry2 = fresh2.prepare(scene)
        # corruption must never serve wrong bits: either rebuild (state) or
        # refuse (None) — both are safe; a returned state must match replay
        if state2 is not None:
            vegsh_twin, _ = full_bits(cache, scene)
            for p in (0, 76, 152):
                assert np.array_equal(state2.bit_plane("vegsh", p), vegsh_twin[:, :, p] != 0)

    def test_key_rebuild_on_cache_change(self, fine_site, tmp_path) -> None:
        cache = fine_site.cache
        root = tmp_path / "veg_state"
        store = self._store(cache, root)
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="fine")
        scene = compose_full_scene_tensors(cache, layer)
        state, _ = store.prepare(scene)
        store.commit(state)
        # a "new cache" (different manifest sha) must not reuse the state
        tampered = SimpleNamespaceShim(
            site_id=cache.site_id,
            tile_key=cache.tile_key,
            model_version=cache.model_version,
            manifest=SimpleNamespaceShim(manifest_sha256="rebuild-me"),
            metadata=lambda: {
                "site_id": cache.site_id,
                "tile_key": cache.tile_key,
                "manifest_sha256": "rebuild-me",
            },
            rows=cache.rows, cols=cache.cols, pixel_size_m=cache.pixel_size_m,
            building_dsm=cache.building_dsm, dem=cache.dem, tree_base=cache.tree_base,
            patch_count=cache.patch_count,
        )
        tampered.svf_patches = cache.svf_patches
        from solweig_gpu.incremental.veg_svf_state import VegOcclusionStore

        store2 = VegOcclusionStore(tampered, state_root=root)
        state2, telemetry = store2.prepare(scene)
        assert state2 is not None
        assert telemetry["baseline_packed"] is True  # rebuilt, not reused


# ---------------------------------------------------------------------------
# Worker integration (state ON for veg edits; refusals route the replay)
# ---------------------------------------------------------------------------


class TestWorkerIntegration:
    @staticmethod
    def _worker(fine_site, tmp_path):
        from solweig_gpu.incremental.worker import ExactWorker
        from tests.test_incremental_worker import LOCAL_ALWAYS

        cache = fine_site.cache
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="w")
        return ExactWorker(
            cache, layer,
            site_dir=fine_site.site,
            results_root=tmp_path / "results",
            selected_date_str=DATE_STR,
            influence_config=LOCAL_ALWAYS,
        )

    def test_worker_veg_job_uses_state_and_matches_twin(self, fine_site, tmp_path) -> None:
        from solweig_gpu.incremental.result import load_patch

        worker = self._worker(fine_site, tmp_path)
        twin = self._worker(fine_site, tmp_path / "twin")
        # off-centre (clear of the NW block) so the influence window stays a
        # strict sub-window of the grid — a centre edit routes the job FULL
        x, y = FINE_ORIGIN[0] + 25.5 * FINE_PIXEL, FINE_ORIGIN[1] - 25.5 * FINE_PIXEL
        worker.layer.add_tree(TreeSpec("t1", x, y, 6.0, 5.0))
        twin.layer.add_tree(TreeSpec("t1", x, y, 6.0, 5.0))
        outcome = worker.run(time_stop=1)
        assert outcome.status == "published"
        assert outcome.mode == "local"
        timings = outcome.diagnostics.get("stage_timings", {})
        assert "svf_seconds" in timings
        veg_diag = outcome.diagnostics.get("veg_state")
        assert veg_diag is not None
        assert veg_diag.get("fallback_reason") is None
        assert veg_diag.get("patches_remarched", 0) > 0
        # twin with the state machinery disabled: bitwise-identical patches
        twin._veg_occlusion_enabled = False
        outcome_twin = twin.run(time_stop=1)
        assert outcome_twin.status == "published"
        assert outcome_twin.mode == "local"
        assert len(outcome.patch_paths) == len(outcome_twin.patch_paths) > 0
        for patch_a, patch_b in zip(outcome.patch_paths, outcome_twin.patch_paths):
            loaded_a = load_patch(Path(patch_a))
            loaded_b = load_patch(Path(patch_b))
            assert loaded_a.variables == loaded_b.variables
            assert loaded_a.write_window == loaded_b.write_window
            for name in loaded_a.variables:
                assert np.array_equal(
                    loaded_a.arrays[name], loaded_b.arrays[name], equal_nan=True
                ), name

    def test_worker_non_binary_cache_routes_replay(self, fine_site, tmp_path, monkeypatch) -> None:
        """A cache whose patch cubes are not binary disables the state path."""
        from solweig_gpu.incremental.cache import SiteCache
        from solweig_gpu.incremental.worker import ExactWorker
        from tests.test_incremental_worker import LOCAL_ALWAYS

        cache = fine_site.cache
        original = dict(cache.svf_patches)
        corrupted = np.array(original["vegshadowmat"])
        corrupted[0, 0, 0] = 2.0  # the vbsh==2.0-class value, on the veg cube
        tampered = dict(original)
        tampered["vegshadowmat"] = corrupted
        # the property returns a fresh mapping per call; patch it at class
        # level so the worker's reads see the corrupted cubes
        monkeypatch.setattr(
            SiteCache, "svf_patches", property(lambda self: tampered)
        )
        layer = TreeLayer(cache.tree_base, fine_site.grid, scenario_id="nb")
        worker = ExactWorker(
            cache, layer, site_dir=fine_site.site,
            results_root=tmp_path / "results",
            selected_date_str=DATE_STR, influence_config=LOCAL_ALWAYS,
        )
        layer.add_tree(TreeSpec("t1", *(FINE_ORIGIN[0] + 25.5 * FINE_PIXEL,
                                        FINE_ORIGIN[1] - 25.5 * FINE_PIXEL), 6.0, 3.0))
        outcome = worker.run(time_stop=1)
        assert outcome.status == "published"
        assert outcome.mode == "local"
        veg_diag = outcome.diagnostics.get("veg_state")
        assert veg_diag is not None
        assert veg_diag.get("fallback_reason") is not None
        assert "binary" in str(veg_diag["fallback_reason"])


# ---------------------------------------------------------------------------
# R4 Phase B (V4): the op-SEQUENCE proof gate over the occluder state.
#
# The harness (docs/.../design/proofs/v4_opsequence_gate.py) drives seeded
# batches B1..Bn (adds/moves/resizes/deletes/replaces/multi-op, interleaved
# amplitude epochs, scene-level trees so dominants are deletable) through
# apply_edit_batch and proves per sequence: (1) soundness -- every accepted
# state folds BITWISE to a fresh from-scratch oracle twin (scalars +
# svftotal equal_nan, vegsh/vbsh planes; oracle alphabets pinned); (2)
# composition invariance -- B1+...+Bn as ONE direct batch folds bitwise the
# oracle too; (3) refusal necessity -- the guard-less hypothetical
# continuation (real apply_edit_batch, advisory guards monkeypatched off)
# adjudicates every typed refusal: necessary (continuation diverges /
# structural guard still fires), conservative (continuation bitwise the
# oracle yet the guard fired INSIDE its mandated predicate -- design-
# mandated conservatism, ADDENDUM 2 conditions 1 and 4, review HIGH-1,
# counted but NOT a defect), or false_refusal (continuation bitwise the
# oracle AND the guard fired OUTSIDE its predicate = a guard misfire = a
# real defect); (4) serialization soundness -- store commit -> fresh store
# -> load preserves packed bytes/checksums bitwise, including across a
# refusal boundary. Sweep profiles: smoke/suite/thorough (CLI default
# thorough; R4B_SWEEP_SCALE overrides). Honest non-coverage: <= ~10
# trees/scene, tiles <= 96x96, dem >= 0 (bush-free), and the comparison
# runs through the shared canonical fold at full-tile accumulate extent
# (the solver-seam itself is pinned by the fine-site classes above).
# ---------------------------------------------------------------------------


_V4_GATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs" / "incremental_design_tool" / "realtime_collaboration"
    / "design" / "proofs" / "v4_opsequence_gate.py"
)


def _load_v4_gate():
    """Load the proof harness by path (it is a CLI tool, not a package)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("v4_opsequence_gate", _V4_GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses resolve cls.__module__ through sys.modules
    # BEFORE exec_module finishes -- register first or import crashes.
    sys.modules["v4_opsequence_gate"] = module
    spec.loader.exec_module(module)
    return module


class TestV4OpSequenceGate:
    def test_refusal_reason_classification_real_strings(self) -> None:
        """classify_refusal pinned against the REAL guard reason vocabulary.

        The value-fence reason ends "(one-step regime)" and the clamp reason
        starts "clamp-regime ..." -- the discriminators must resolve fence
        and clamp BEFORE the regime match (r4b review item 1).
        """
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            _fence_binary_planes,
            _regime_transition_reason,
        )

        v4 = _load_v4_gate()
        vbsh = torch.zeros((4, 4, 1))
        vbsh[1, 1, 0] = 2.0
        with pytest.raises(VegOcclusionFallback) as excinfo:
            _fence_binary_planes(torch.zeros((4, 4, 1)), vbsh, context="probe")
        assert v4.classify_refusal(excinfo.value.reason) == "fence"
        regime = _regime_transition_reason(27.0, 12.0, 0.25)
        assert regime is not None
        assert v4.classify_refusal(regime) == "regime"
        assert (
            v4.classify_refusal(
                "clamp-regime amplitude change: effective march amplitude "
                "12.0 -> 18.0 with the smaller-amplitude scene clamped"
            )
            == "clamp"
        )

    def test_all_sites_baseline_packable(self) -> None:
        """Every sweep site clears the HIGH-1 pack fence (multi-step base)."""
        v4 = _load_v4_gate()
        assert len(v4.SITES) == 7
        for name, spec in v4.SITES.items():
            cache, _grid = v4.build_site(spec, v4.initial_forest(spec))
            assert cache.rows == spec.rows
            assert cache.cols == spec.cols

    def test_clamp_case_c_refusal_necessary_and_predicate_held(self) -> None:
        """The pinned NECESSARY refusal: clamped raiser (Case C).

        Condition 1 fires (amplitude changed + smaller scene clamped --
        predicate verified by INDEPENDENT recomputation, not the guard's
        own code), and with the advisory guards disabled the continuation
        FOLD diverges from the oracle: the refusal prevented real silent
        corruption (the corridor-sufficiency proof's violated case).
        """
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            apply_edit_batch,
            build_baseline_state,
        )

        v4 = _load_v4_gate()
        cache, grid, layer = build_main_tile()
        scene0 = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene0))
        state = build_baseline_state(cache)

        layer.add_tree(TreeSpec("over", *pos(44, 43, 60, 2.0), 11.0, 4.0))
        scene_clamped = scene_of(cache, layer)
        state, telemetry = apply_edit_batch(state, cache, scene_clamped)
        assert telemetry["fallback_reason"] is None

        layer.add_tree(TreeSpec("tall", *pos(20, 20, 60, 2.0), 18.0, 3.0))
        scene_post = scene_of(cache, layer)
        with pytest.raises(VegOcclusionFallback) as excinfo:
            apply_edit_batch(state, cache, scene_post)
        assert "clamp" in str(excinfo.value)

        assert v4._refusal_predicate_held("clamp", scene_clamped, scene_post, 0.5) is True

        with v4.guards_disabled():
            cont_state, _tel = apply_edit_batch(state, cache, scene_post)
        orb_vegsh, orb_vbsh = full_bits(cache, scene_post)
        outcome, absorbed = v4._continuation_outcome(
            cont_state, cache, scene_post, orb_vegsh, orb_vbsh,
            v4.fold_planes(orb_vegsh, orb_vbsh, scene_post, 60, 60),
        )
        assert outcome == "fold_diverges"
        assert not absorbed

    def test_regime_refusal_conservative_not_false(self, tmp_path) -> None:
        """Regime-fence refusals that fold bitwise clean are CONSERVATIVE.

        The 27 m-dominant dropper genuinely narrows 9 patches multi-step ->
        one-step (predicate held, ADDENDUM 2 condition 4 / review HIGH-1)
        while the oracle stays binary in this instance -- the design
        refuses on the POTENTIAL (the only way to know is the replay the
        refusal routes to). Counted as conservative, NOT a false refusal:
        false_refusals == 0 is the gate. Also pins the crash/restart idiom
        ACROSS the refusal boundary (commit -> fresh store -> recover the
        committed pre-refusal state bitwise).
        """
        v4 = _load_v4_gate()
        case = v4.run_case(
            "regime4m", seed=20260904, n_batches=3,
            store_root=tmp_path / "store", stop_before=3,
        )
        refused = [s for s in case.steps if s.outcome == "refused"]
        assert len(refused) == 2
        assert all(s.refusal_class == "regime" for s in refused)
        assert all(s.refusal_verdict == "conservative" for s in refused)
        assert all(s.predicate_held for s in refused)
        assert all(s.oracle_binary for s in refused)
        assert case.summary()["false_refusals"] == 0
        assert case.defects == []
        assert case.roundtrip == "ok"

    def test_composition_invariance_direct_equals_incremental(self, tmp_path) -> None:
        """B1+...+Bn composed vs per-batch incremental: both bitwise oracle."""
        v4 = _load_v4_gate()
        case = v4.run_case(
            "main2m", seed=20260904, n_batches=6, store_root=tmp_path / "store",
        )
        assert case.invariance == "ok"
        assert case.roundtrip == "ok"
        assert case.defects == []
        assert any(s.outcome == "applied" for s in case.steps)

    def test_v4_suite_profile_sweep_gate(self) -> None:
        """The randomized seeded property sweep (suite profile, 14 cases).

        Gate: ZERO false refusals and ZERO defects of any kind (silent
        divergence, alphabet breach, refusal advanced state, serialization
        drift, invariance break, guard misfire). Non-vacuity: refusals of
        BOTH design families observed (clamp Case C and regime HIGH-1),
        every case's mid-sequence store round trip bit-exact, no invariance
        MISMATCH.
        """
        v4 = _load_v4_gate()
        result = v4.run_sweep("suite", verbose=False)
        totals = result["totals"]
        assert totals["cases"] == 14
        assert totals["steps"] >= 60
        assert totals["applied"] > 0
        assert totals["refused"] > 0
        assert totals["false_refusals"] == 0
        assert totals["conservative_refusals"] > 0
        assert totals["defects"] == []
        assert totals["absorbed_divergences"] >= 0
        assert set(totals["refusal_classes_observed"]) == {"clamp", "regime"}
        assert set(totals["conservative_by_class"]) <= {"clamp", "regime"}
        allowed = {"ok", "direct_refused", "incremental_refused", "both_refused"}
        for case in result["cases"]:
            assert case["false_refusals"] == 0
            assert case["invariance"] in allowed
            assert case["roundtrip"] == "ok"
        assert totals["roundtrip_ok"] == totals["cases"]
