# SPDX-License-Identifier: GPL-3.0-only
"""T19b gates: the production SVF-march lane router (march_router).

The router is the routing seam between the production call sites
(``_recompute_veg_svf_window``'s 153 per-patch marches and
``apply_edit_batch``'s corridor re-marches) and the registered march lanes:
the exact Numba march (T04 kernel + R3 general step tables) on the default
lane, the legacy torch kernel on refusal or the ``torch`` flag. These gates
pin the four contract legs:

* **registry discipline** — the lane vocabulary is explicit; an unknown
  flag value is refused LOUDLY, never guessed into a lane;
* **raw-bit parity** — on the routed lane the returned planes are raw-bit
  equal to ``solweig_gpu.shadow.shadow``'s across the variant grid (angles,
  scales, amplitude ladder, negative DEM, vegetation-free, canopy-over-
  building, non-square, strided inputs, NaN a-planes), including the
  non-binary one-step ``vbsh == 2.0`` regime the packing fences rely on;
* **typed refusal -> fallback** — scenes outside the Numba domain (bush >
  0, wrong dtype) fall back to the torch kernel with the ORIGINAL
  arguments, bit-identically, recorded in the lane stats — never a silent
  approximation;
* **production-path parity** — the real bundle (``window_svf_bundle``)
  and the real state path (``apply_edit_batch``) produce bitwise-identical
  results under the routed lane and the forced torch lane.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from solweig_gpu.incremental import march_router
from solweig_gpu.incremental.geometry import TreeSpec
from solweig_gpu.incremental.march_router import (
    LANE_NUMBA_MARCH,
    LANE_TORCH_SHADOW,
    MARCH_LANES,
    SVF_MARCH_LANE_ENV_FLAG,
    MarchLaneError,
)
from solweig_gpu.shadow import shadow as shadow_fn

from tests.test_incremental_perf_wave2 import (  # noqa: F401  (helpers)
    WRITE_WINDOW,
    _bundle,
    _edited_scene,
)
from tests.test_incremental_veg_occlusion import (
    PATCHES,
    build_main_tile,
    build_reviewer_counterexample_scene,
    effective_amplitude,
    pos,
    replay_cubes_as_cache_patches,
    scene_of,
)
from tests.test_incremental_worker import (  # noqa: F401  (fixture)
    SVF_BUNDLE_INDEX,
    science_site,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic planes + the two call arms
# ---------------------------------------------------------------------------


def _scene(
    rows: int,
    cols: int,
    *,
    seed: int = 7,
    negative_dem: bool = False,
    no_vegetation: bool = False,
    overlap: bool = False,
):
    """Deterministic synthetic torch float32 planes (T03 generator family)."""
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    if negative_dem:
        dem -= np.float32(12.0)
    a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    a[3 : min(9, rows), 4 : min(12, cols)] += np.float32(18.0)
    if no_vegetation:
        canopy = np.zeros((rows, cols), dtype=np.float32)
    else:
        canopy = rng.uniform(0.0, 6.0, (rows, cols)).astype(np.float32)
        canopy[canopy < np.float32(3.0)] = np.float32(0.0)
        canopy[min(10, rows) : min(20, rows), min(30, cols) : min(60, cols)] = (
            np.float32(5.5)
        )
        if overlap:
            canopy[min(2, rows) : min(11, rows), min(6, cols) : min(16, cols)] = (
                np.float32(25.0)
            )
    vegdem = canopy + dem
    vegdem2 = canopy * np.float32(0.25) + dem
    bush = np.zeros((rows, cols), dtype=np.float32)
    to_t = lambda arr: torch.from_numpy(  # noqa: E731
        np.ascontiguousarray(arr).copy()
    )
    return to_t(a), to_t(vegdem), to_t(vegdem2), to_t(bush)


def _torch_ref(amp, a, vegdem, vegdem2, bush, az, alt, scale):
    """The production call convention: 0-dim f32 tensors + python scale."""
    return shadow_fn(
        torch.tensor(np.float32(amp)),
        a,
        vegdem,
        vegdem2,
        bush,
        torch.tensor(np.float32(az)),
        torch.tensor(np.float32(alt)),
        scale,
    )


def _routed(amp, a, vegdem, vegdem2, bush, az, alt, scale, **kwargs):
    return march_router.shadow_march(
        torch.tensor(np.float32(amp)),
        a,
        vegdem,
        vegdem2,
        bush,
        torch.tensor(np.float32(az)),
        torch.tensor(np.float32(alt)),
        scale,
        torch_shadow=shadow_fn,
        **kwargs,
    )


def _bits(plane) -> np.ndarray:
    """Raw bits of one f32 plane (NaN-safe equality witness)."""
    array = plane.numpy() if isinstance(plane, torch.Tensor) else plane
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _assert_bits_equal(routed, reference, label: str) -> None:
    for name, got, want in zip(("sh", "vegsh", "vbshvegsh"), routed, reference):
        assert _bits(got).shape == _bits(want).shape, (label, name)
        assert np.array_equal(_bits(got), _bits(want)), (
            f"{label}: routed {name} bits diverge from the torch kernel "
            f"({int((_bits(got) != _bits(want)).sum())} cells)"
        )


@pytest.fixture(autouse=True)
def _clean_lane(monkeypatch):
    """Every test starts from the default lane and zeroed counters."""
    monkeypatch.delenv(SVF_MARCH_LANE_ENV_FLAG, raising=False)
    march_router.reset_lane_stats()
    yield
    march_router.reset_lane_stats()


# ---------------------------------------------------------------------------
# Leg 1: registry discipline
# ---------------------------------------------------------------------------


class TestLaneResolution:
    def test_default_lane_is_numba(self, monkeypatch) -> None:
        monkeypatch.delenv(SVF_MARCH_LANE_ENV_FLAG, raising=False)
        assert march_router.resolve_march_lane() == LANE_NUMBA_MARCH

    def test_flag_spellings(self, monkeypatch) -> None:
        for value, want in (
            ("", LANE_NUMBA_MARCH),
            ("numba", LANE_NUMBA_MARCH),
            (LANE_NUMBA_MARCH, LANE_NUMBA_MARCH),
            ("NUMBA", LANE_NUMBA_MARCH),
            ("torch", LANE_TORCH_SHADOW),
            (LANE_TORCH_SHADOW, LANE_TORCH_SHADOW),
            (" Torch ", LANE_TORCH_SHADOW),
        ):
            monkeypatch.setenv(SVF_MARCH_LANE_ENV_FLAG, value)
            assert march_router.resolve_march_lane() == want, value

    def test_unknown_lane_refused_loudly(self, monkeypatch) -> None:
        monkeypatch.setenv(SVF_MARCH_LANE_ENV_FLAG, "cuda")
        with pytest.raises(MarchLaneError) as caught:
            march_router.resolve_march_lane()
        message = str(caught.value)
        assert SVF_MARCH_LANE_ENV_FLAG in message
        assert "'cuda'" in message

    def test_unknown_lane_never_reaches_a_march(self, monkeypatch) -> None:
        monkeypatch.setenv(SVF_MARCH_LANE_ENV_FLAG, "numba-march")
        a, vegdem, vegdem2, bush = _scene(8, 8)
        with pytest.raises(MarchLaneError):
            march_router.shadow_march(
                torch.tensor(np.float32(4.0)), a, vegdem, vegdem2, bush,
                torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)),
                0.5, torch_shadow=shadow_fn,
            )
        stats = march_router.lane_stats()
        assert stats["numba_routed"] == 0
        assert stats["torch_forced"] == 0

    def test_registered_vocabulary_is_closed(self) -> None:
        assert set(MARCH_LANES) == {LANE_TORCH_SHADOW, LANE_NUMBA_MARCH}


# ---------------------------------------------------------------------------
# Leg 2: raw-bit parity on the routed lane
# ---------------------------------------------------------------------------


class TestRoutedBitParity:
    @pytest.mark.parametrize("azimuth", (0.0, 37.0, 225.0, 312.0))
    @pytest.mark.parametrize("altitude", (6.0, 42.0, 78.0))
    @pytest.mark.parametrize("scale", (0.5, 2.0))
    def test_variant_geometry_grid(self, azimuth, altitude, scale) -> None:
        a, vegdem, vegdem2, bush = _scene(24, 32)
        amp = 8.0
        reference = _torch_ref(
            amp, a, vegdem, vegdem2, bush, azimuth, altitude, scale
        )
        routed = _routed(
            amp, a, vegdem, vegdem2, bush, azimuth, altitude, scale
        )
        _assert_bits_equal(
            routed, reference, f"az={azimuth} alt={altitude} scale={scale}"
        )
        assert march_router.lane_stats()["numba_routed"] == 1

    @pytest.mark.parametrize("amplitude", (0.5, 2.0, 8.0, 30.0))
    def test_amplitude_ladder(self, amplitude) -> None:
        a, vegdem, vegdem2, bush = _scene(20, 20, seed=11)
        reference = _torch_ref(
            amplitude, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5
        )
        routed = _routed(
            amplitude, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5
        )
        _assert_bits_equal(routed, reference, f"amp={amplitude}")

    def test_negative_dem(self) -> None:
        a, vegdem, vegdem2, bush = _scene(20, 20, seed=13, negative_dem=True)
        reference = _torch_ref(12.0, a, vegdem, vegdem2, bush, 225.0, 6.0, 0.5)
        routed = _routed(12.0, a, vegdem, vegdem2, bush, 225.0, 6.0, 0.5)
        _assert_bits_equal(routed, reference, "negative dem")

    def test_vegetation_free_scene(self) -> None:
        a, vegdem, vegdem2, bush = _scene(20, 20, seed=17, no_vegetation=True)
        reference = _torch_ref(6.0, a, vegdem, vegdem2, bush, 90.0, 42.0, 0.5)
        routed = _routed(6.0, a, vegdem, vegdem2, bush, 90.0, 42.0, 0.5)
        _assert_bits_equal(routed, reference, "no vegetation")

    def test_canopy_overlapping_building(self) -> None:
        a, vegdem, vegdem2, bush = _scene(20, 20, seed=19, overlap=True)
        reference = _torch_ref(10.0, a, vegdem, vegdem2, bush, 312.0, 78.0, 0.5)
        routed = _routed(10.0, a, vegdem, vegdem2, bush, 312.0, 78.0, 0.5)
        _assert_bits_equal(routed, reference, "overlap")

    def test_non_square_grid(self) -> None:
        a, vegdem, vegdem2, bush = _scene(16, 40, seed=23)
        reference = _torch_ref(8.0, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5)
        routed = _routed(8.0, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5)
        _assert_bits_equal(routed, reference, "16x40")

    def test_strided_input_views(self) -> None:
        """Production hands in window SLICES; strided input must not bite."""
        a, vegdem, vegdem2, bush = _scene(40, 48, seed=29)
        sl = (slice(5, 35), slice(7, 41))
        reference = _torch_ref(
            8.0, a[sl], vegdem[sl], vegdem2[sl], bush[sl], 225.0, 42.0, 0.5
        )
        routed = _routed(
            8.0, a[sl], vegdem[sl], vegdem2[sl], bush[sl], 225.0, 42.0, 0.5
        )
        _assert_bits_equal(routed, reference, "strided slice")

    def test_nan_a_plane(self) -> None:
        """NaN a-planes disable the suffix exit per batch; bits still equal."""
        a, vegdem, vegdem2, bush = _scene(20, 20, seed=31)
        a = a.clone()
        a[8, 9] = float("nan")
        reference = _torch_ref(8.0, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5)
        routed = _routed(8.0, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5)
        _assert_bits_equal(routed, reference, "nan a-plane")

    def test_one_step_regime_preserves_vbsh_two(self) -> None:
        """The router must NOT flatten the non-binary vbsh==2.0 regime."""
        cache, _grid, layer = build_reviewer_counterexample_scene()
        scene = scene_of(cache, layer)
        eff = effective_amplitude(scene)
        altitude, azimuth = PATCHES[148][0], PATCHES[148][1]
        reference = shadow_fn(
            torch.tensor(np.float32(eff)),
            scene.a, scene.vegdsm, scene.vegdsm2, scene.bush,
            azimuth, altitude, 1.0 / 4.0,
        )
        routed = march_router.shadow_march(
            torch.tensor(np.float32(eff)),
            scene.a, scene.vegdsm, scene.vegdsm2, scene.bush,
            azimuth, altitude, 1.0 / 4.0,
            amplitude_policy="effective_windowed",
            torch_shadow=shadow_fn,
        )
        _assert_bits_equal(routed, reference, "patch 148 counterexample")
        assert float(routed[2].max()) == 2.0, (
            "the routed lane flattened vbsh==2.0 to {0,1} — the veg-state "
            "packing fences would silently accept a wrong regime"
        )


# ---------------------------------------------------------------------------
# Leg 3: typed refusal -> torch fallback, never approximation
# ---------------------------------------------------------------------------


class TestRefusalFallback:
    def test_bush_scene_falls_back_bit_identically(self) -> None:
        a, vegdem, vegdem2, bush = _scene(20, 20, seed=37)
        bush = bush.clone()
        bush[10, 10] = 1.5
        reference = _torch_ref(8.0, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5)
        march_router.reset_lane_stats()
        routed = _routed(8.0, a, vegdem, vegdem2, bush, 37.0, 42.0, 0.5)
        _assert_bits_equal(routed, reference, "bush>0 fallback")
        stats = march_router.lane_stats()
        assert stats["numba_routed"] == 0
        assert stats["numba_refused"] == 1
        assert stats["last_refusal"] is not None
        assert "bush" in str(stats["last_refusal"])

    def test_bush_refusal_telemetry_mapping(self) -> None:
        a, vegdem, vegdem2, bush = _scene(12, 12, seed=41)
        bush = bush.clone()
        bush[4, 4] = 2.0
        telemetry: dict = {}
        march_router.shadow_march(
            torch.tensor(np.float32(4.0)), a, vegdem, vegdem2, bush,
            torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)),
            0.5, torch_shadow=shadow_fn, telemetry=telemetry,
        )
        assert telemetry["lane"] == LANE_TORCH_SHADOW
        assert telemetry["numba_refused"] == 1
        assert "bush" in telemetry["refusal"]

    def test_wrong_dtype_falls_back(self) -> None:
        a, vegdem, vegdem2, bush = _scene(12, 12, seed=43)
        a64 = a.double()
        reference = shadow_fn(
            torch.tensor(np.float32(4.0)), a64, vegdem, vegdem2, bush,
            torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)), 0.5,
        )
        routed = march_router.shadow_march(
            torch.tensor(np.float32(4.0)), a64, vegdem, vegdem2, bush,
            torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)),
            0.5, torch_shadow=shadow_fn,
        )
        for got, want in zip(routed, reference):
            assert torch.equal(got, want)
        assert march_router.lane_stats()["numba_refused"] == 1

    def test_routed_telemetry_mapping(self) -> None:
        a, vegdem, vegdem2, bush = _scene(12, 12, seed=47)
        telemetry: dict = {}
        march_router.shadow_march(
            torch.tensor(np.float32(4.0)), a, vegdem, vegdem2, bush,
            torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)),
            0.5, torch_shadow=shadow_fn, telemetry=telemetry,
        )
        assert telemetry["lane"] == LANE_NUMBA_MARCH
        assert telemetry["numba_routed"] == 1


# ---------------------------------------------------------------------------
# Leg 4a: the forced torch lane is byte-identical + witnessable
# ---------------------------------------------------------------------------


class TestTorchLane:
    def test_flag_forces_torch_kernel_bits(self, monkeypatch) -> None:
        monkeypatch.setenv(SVF_MARCH_LANE_ENV_FLAG, "torch")
        a, vegdem, vegdem2, bush = _scene(20, 20, seed=53)
        reference = _torch_ref(8.0, a, vegdem, vegdem2, bush, 225.0, 6.0, 0.5)
        march_router.reset_lane_stats()
        routed = _routed(8.0, a, vegdem, vegdem2, bush, 225.0, 6.0, 0.5)
        _assert_bits_equal(routed, reference, "torch lane")
        stats = march_router.lane_stats()
        assert stats["torch_forced"] == 1
        assert stats["numba_routed"] == 0

    def test_torch_shadow_callable_witnesses_only_its_lane(self) -> None:
        """The caller's torch binding is the FALLBACK callable: called on
        the torch lane and on refusal, never on a routed bush-free march —
        this is what keeps solver-level shadow_fn spies meaningful."""
        a, vegdem, vegdem2, bush = _scene(16, 16, seed=59)
        calls = []

        def witness(*args):
            calls.append(1)
            return shadow_fn(*args)

        march_router.shadow_march(
            torch.tensor(np.float32(4.0)), a, vegdem, vegdem2, bush,
            torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)),
            0.5, torch_shadow=witness,
        )
        assert calls == [], "routed lane must not call the torch fallback"

        march_router.shadow_march(
            torch.tensor(np.float32(4.0)), a, vegdem, vegdem2, bush,
            torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)),
            0.5, torch_shadow=witness,
            telemetry={},
        )
        assert calls == [], "telemetry must not change the lane"

        bush_bad = bush.clone()
        bush_bad[6, 6] = 1.0
        march_router.shadow_march(
            torch.tensor(np.float32(4.0)), a, vegdem, vegdem2, bush_bad,
            torch.tensor(np.float32(37.0)), torch.tensor(np.float32(42.0)),
            0.5, torch_shadow=witness,
        )
        assert len(calls) == 1, "refusal must run the torch fallback"


# ---------------------------------------------------------------------------
# Step-table memoization (table identity is the cache contract)
# ---------------------------------------------------------------------------


class TestTableMemoization:
    def test_same_geometry_shares_one_table(self) -> None:
        first = march_router._table_for(37.0, 42.0, 0.5, 24, 32, 8.0,
                                        "effective_windowed")
        second = march_router._table_for(37.0, 42.0, 0.5, 24, 32, 8.0,
                                         "effective_windowed")
        assert first is second

    def test_f64_spellings_rounding_to_same_f32_share_one_table(self) -> None:
        first = march_router._table_for(37.0, 42.0, 0.5, 24, 32, 8.0,
                                        "effective_windowed")
        # 37.0 + 1e-6 rounds to the SAME f32 as 37.0 — the kernel would see
        # one angle, so the cache must see one key (bits, not spellings)
        second = march_router._table_for(
            37.0 + 1e-6, 42.0, 0.5, 24, 32, 8.0, "effective_windowed"
        )
        assert first is second

    def test_distinct_amplitude_bits_get_distinct_tables(self) -> None:
        first = march_router._table_for(37.0, 42.0, 0.5, 24, 32, 8.0,
                                        "effective_windowed")
        second = march_router._table_for(37.0, 42.0, 0.5, 24, 32, 8.5,
                                         "effective_windowed")
        assert first is not second

    def test_unknown_policy_refused_loudly(self) -> None:
        with pytest.raises(MarchLaneError):
            march_router._table_for(37.0, 42.0, 0.5, 24, 32, 8.0, "wild-guess")


# ---------------------------------------------------------------------------
# Leg 4b: production-path parity — the real bundle, both lanes
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestScienceSiteBundleParity:
    def test_bundle_bitwise_across_lanes(self, science_site, monkeypatch) -> None:
        """The REAL 153-patch replay through window_svf_bundle produces
        byte-identical bundles on the routed lane and the forced torch lane
        (T04 pinned the kernel bits; this WITNESSES them on the production
        path, end to end)."""
        scene = _edited_scene(science_site)

        monkeypatch.delenv(SVF_MARCH_LANE_ENV_FLAG, raising=False)
        march_router.reset_lane_stats()
        routed = _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)
        routed_stats = march_router.lane_stats()
        assert routed_stats["numba_routed"] >= 153, (
            "the production replay did not run through the numba lane"
        )
        assert routed_stats["numba_refused"] == 0

        monkeypatch.setenv(SVF_MARCH_LANE_ENV_FLAG, "torch")
        march_router.reset_lane_stats()
        torched = _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)
        torched_stats = march_router.lane_stats()
        assert torched_stats["torch_forced"] >= 153
        assert torched_stats["numba_routed"] == 0

        assert len(routed) == len(torched)
        for name, index in SVF_BUNDLE_INDEX.items():
            got = routed[index]
            want = torched[index]
            if isinstance(got, torch.Tensor):
                assert torch.equal(got, want), name
            else:
                assert got == want, name

    def test_routed_bundle_stage_counts_reach_the_timing_channel(self) -> None:
        """solve_window lifts the lane counter deltas into these timing
        fields; the jobs seam must carry them into job metrics when the
        solver reports them (float-typed counts, not seconds)."""
        from solweig_gpu.server.jobs import stage_timing_metrics

        lifted = stage_timing_metrics(
            {"svf_seconds": 1.25, "svf_march_routed": 153.0,
             "svf_march_refused": 0.0}
        )
        assert lifted["svf_march_routed"] == 153.0
        assert lifted["svf_march_refused"] == 0.0
        # legacy paths without the fields still omit them
        assert stage_timing_metrics({"svf_seconds": 1.0}) == {
            "svf_seconds": 1.0
        }


# ---------------------------------------------------------------------------
# Leg 4c: production-path parity — the real veg-state corridor, both lanes
# ---------------------------------------------------------------------------


class TestApplyEditBatchParity:
    @staticmethod
    def _prepared():
        from solweig_gpu.incremental.veg_svf_state import build_baseline_state

        cache, grid, layer = build_main_tile()
        scene = scene_of(cache, layer)
        cache.svf_patches.update(replay_cubes_as_cache_patches(cache, scene))
        return cache, layer, build_baseline_state(cache)

    def test_corridor_state_bitwise_across_lanes(self, monkeypatch) -> None:
        from solweig_gpu.incremental.veg_svf_state import apply_edit_batch

        # torch-lane arm
        cache, layer, state = self._prepared()
        layer.add_tree(TreeSpec("t0", *pos(10, 50, 60, 2.0), 9.0, 3.0))
        scene_post = scene_of(cache, layer)
        monkeypatch.setenv(SVF_MARCH_LANE_ENV_FLAG, "torch")
        march_router.reset_lane_stats()
        state_torch, telemetry_torch = apply_edit_batch(state, cache, scene_post)
        stats_torch = march_router.lane_stats()
        assert stats_torch["torch_forced"] > 0
        assert telemetry_torch["march_routed"] == 0

        # routed-lane arm from an identical fresh baseline
        cache2, layer2, state2 = self._prepared()
        layer2.add_tree(TreeSpec("t0", *pos(10, 50, 60, 2.0), 9.0, 3.0))
        scene_post2 = scene_of(cache2, layer2)
        monkeypatch.delenv(SVF_MARCH_LANE_ENV_FLAG, raising=False)
        march_router.reset_lane_stats()
        state_numba, telemetry_numba = apply_edit_batch(
            state2, cache2, scene_post2
        )
        stats_numba = march_router.lane_stats()
        assert stats_numba["numba_routed"] == telemetry_numba["march_routed"] > 0
        assert telemetry_numba["march_refused"] == 0

        # PackedVisibility wraps uint8 cubes (.data); compare the bytes
        # directly (the dataclass eq chokes on the ndarray field).
        assert np.array_equal(
            state_numba.vegsh_packed.data, state_torch.vegsh_packed.data
        ), "vegsh packed bits diverge across march lanes"
        assert np.array_equal(
            state_numba.vbsh_packed.data, state_torch.vbsh_packed.data
        ), "vbsh packed bits diverge across march lanes"

    def test_bush_batch_refusal_refuses_the_corridor(self) -> None:
        """A bushy post scene refuses the CORRIDOR path itself (condition 2)
        — routing never overrides a higher-level refusal."""
        from solweig_gpu.incremental.veg_svf_state import (
            VegOcclusionFallback,
            apply_edit_batch,
        )

        cache, layer, state = self._prepared()
        layer.add_tree(TreeSpec("t0", *pos(10, 50, 60, 2.0), 9.0, 3.0))
        scene_post = scene_of(cache, layer)
        # clone-then-mutate: never write into shared scene tensors
        scene_post.bush = scene_post.bush.clone()
        scene_post.bush[5, 5] = 1.0
        with pytest.raises(VegOcclusionFallback):
            apply_edit_batch(state, cache, scene_post)
