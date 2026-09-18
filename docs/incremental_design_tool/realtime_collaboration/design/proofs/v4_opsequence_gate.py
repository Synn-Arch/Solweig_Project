# SPDX-License-Identifier: GPL-3.0-only
"""R4 Phase B (V4) -- op-sequence proof gate for the veg SVF occluder state.

Scope (dispatch mandate, authoritative over the design's thinner V4 bullet,
which names the not-yet-built phase-B fused kernel): r4a's suite proves
SINGLE-batch correctness. V4 proves the STATE MACHINE over arbitrary op
SEQUENCES. For any sequence of accepted edit batches B1..Bn (adds / deletes /
moves / resizes / replaces / growth transitions, multi-op batches, interleaved
amplitude epochs), the packed occluder state after Bi applied incrementally
must fold to the SAME svf planes and veg planes as folding the composed scene
at Bi from scratch (state-transition soundness), and the typed refusals
(``VegOcclusionFallback``: clamp / regime / value fence) must fire exactly
when the from-scratch fold would diverge (refusal completeness: no silent
divergence, no false refusal).

Four properties, all checked per sequence:

1. soundness (every accepted step): incremental packed-state fold vs a
   from-scratch oracle twin -- bitwise (``equal_nan``) on the ten veg
   scalars + ``svftotal`` and bitwise on the vegsh/vbsh planes; oracle value
   alphabets pinned (vegsh in {0,1}, vbsh in {0,1,2}); a state-path SUCCESS
   alongside oracle ``vbsh == 2.0`` is a missing-refusal defect (alphabet
   breach).
2. composition invariance (final scene): the per-batch incremental path's
   fold vs ONE direct batch applied at the final scene (the B1+...+Bn
   composition) -- both bitwise the oracle fold. A path whose PRE scene
   trips a guard is recorded (guard outcomes are per-pre-scene and allowed
   to differ); whatever path did apply is still compared to the oracle.
3. refusal necessity (every refusal): the guard-less hypothetical
   continuation (the REAL ``apply_edit_batch`` with the three advisory
   guards disabled by monkeypatch -- no physics is duplicated) is compared
   to the oracle, and the firing guard is checked against its own mandated
   predicate. Verdicts:
     - ``necessary``: the continuation's planes diverge from the oracle
       (the refusal prevented silent corruption), OR a structural guard
       (bush) still fires with the advisory ones off;
     - ``conservative``: the continuation is bitwise the oracle yet the
       guard fired strictly inside its mandated predicate. This is
       DESIGN-MANDATED conservatism, not a defect: ADDENDUM 2 condition 4
       (regime level, review HIGH-1) refuses every multi-step -> one-step
       narrowing because oracle ``vbsh == 2.0`` CAN appear at unchanged
       cells, and condition 1 refuses every Case C amplitude change for
       the same reason. Conservative refusals are COUNTED and reported
       split by guard class (a nonzero clamp-conservative count would
       empirically qualify the Case C proof's universality and is surfaced
       for lead adjudication -- the wave still follows the binding
       condition). Regime-conservative refusals additionally record
       whether the oracle truly stayed binary (pure conservatism) or not.
     - ``false_refusal`` (DEFECT): the continuation is bitwise the oracle
       AND the guard fired OUTSIDE its mandated predicate (e.g. the clamp
       guard on an amplitude-constant pair -- forbidden by condition 1),
       or the value fence fired on planes that turn out binary.
   Planes differ but folds equal is recorded as an absorbed divergence
   (the bits were genuinely unrepresentable; the fences did their job).
4. serialization soundness (mid-sequence): ``VegOcclusionStore.commit`` ->
   fresh store -> ``prepare`` preserves the packed bytes, canopy snapshot,
   chunk versions and checksums BITWISE.

Everything is seeded and deterministic (``numpy.random.default_rng``; no
wall-clock, no unseeded RNG in any comparison). Sweep size is profiled:
``smoke`` / ``suite`` (pytest default) / ``thorough`` (CLI default); the
``R4B_SWEEP_SCALE`` environment variable overrides the profile selection.

Honest non-coverage (the sweep does NOT reach): extreme tree counts (baked +
layer trees stay <= ~10 per scene), tiles above 96x96, negative DEM (bush
scenes -- the bush guard's necessity is structural and pinned by the r4a
suite, not re-proven here), and the fine-site solver seam
(``window_svf_bundle``) which the r4a fine-site tests already pin: this
harness compares through the shared canonical fold at full-tile accumulate
extent with a zero building window, whose contribution is identical on both
sides of the differential.

Evidence: ``python docs/incremental_design_tool/realtime_collaboration/
design/proofs/v4_opsequence_gate.py`` writes
``/tmp/r4b_proof/v4_opsequence_gate.json`` (+ stdout log).
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_gpu.incremental.geometry import (  # noqa: E402
    RasterGrid,
    RasterWindow,
    TreeSpec,
)
from solweig_gpu.incremental.solver import (  # noqa: E402
    FullSceneTensors,
    _compose_scene_from_canopy,
    _fold_veg_svf_from_planes,
    _recompute_veg_svf_window,
    _sky_patch_geometry,
    effective_march_amplitude,
)
from solweig_gpu.incremental.trees import rasterize_tree_patch  # noqa: E402
from solweig_gpu.incremental import veg_svf_state as vmod  # noqa: E402
from solweig_gpu.incremental.veg_svf_state import (  # noqa: E402
    VegOcclusionFallback,
    VegOcclusionStore,
    apply_edit_batch,
    build_baseline_state,
    one_step_patch_indices,
)

PATCHES, _RINGS = _sky_patch_geometry(2)
N_PATCHES = len(PATCHES)

SCALAR_NAMES = (
    "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
    "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
)

_BASE_SEED = 20260904


# ---------------------------------------------------------------------------
# Stub site plumbing (self-contained; mirrors the r4a suite's StubCache)
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
        sha: str = "stub-sha-v4",
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.pixel_size_m = pixel
        self.building_dsm = building
        self.dem = dem
        self.tree_base = tree_base
        self.manifest = _StubManifest(rows, pixel, sha)
        self._svf_patches = svf_patches or {}
        self.site_id = "stub-site-v4"
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


@dataclass(frozen=True)
class SiteSpec:
    """A synthetic tile: buildings, optional DEM relief, baked base trees.

    The maximum block height sets the baseline effective march amplitude and
    MUST clear the tile's largest first-step dz (the HIGH-1 pack fence refuses
    one-step baselines) -- ``build_site`` asserts that, so a mis-modeled spec
    fails loudly instead of silently shrinking coverage. Scales deliberately
    mix power-of-two (0.5, 1.0, 4.0 m px) and non-power-of-two (0.4, 1/3)
    values -- the dz_1 association arithmetic (review NEW-LOW) only bites off
    the power-of-two grid.
    """

    name: str
    rows: int
    cols: int
    pixel: float
    blocks: tuple[tuple[int, int, int, int, float], ...]
    base_trees: tuple[tuple[int, int, float, float], ...]
    dem_slope: float = 0.0  # dem[r, c] = dem_slope * (r + c) >= 0 (bush-free)

    @property
    def scale(self) -> float:
        return 1.0 / self.pixel

    @property
    def block_max(self) -> float:
        return max(block[4] for block in self.blocks)


@dataclass
class VTree:
    """A scene-level tree (the op vocabulary edits THESE, not a TreeLayer).

    The state machine's contract is scene-level: ``apply_edit_batch`` takes
    any composed pre/post pair and recomposes the pre scene from the state's
    canopy snapshot. Driving it through composed canopy rasters therefore
    exercises exactly what the store/worker rely on -- and, unlike a
    TreeLayer, it lets a batch DELETE or resize the amplitude-dominant tree,
    which is the only way the regime fence (HIGH-1) is reachable.
    """

    tree_id: str
    row: float
    col: float
    height: float
    radius: float


def canopy_of(trees: list[VTree], grid: RasterGrid, rows: int, cols: int) -> np.ndarray:
    """Max-combined canopy raster (trees.py's own rasterize primitive)."""
    canopy = np.zeros((rows, cols), dtype=np.float32)
    for tree in trees:
        patch, _trunk = rasterize_tree_patch(
            TreeSpec(tree.tree_id, *cell_xy(tree, grid), tree.height, tree.radius),
            grid,
            grid.full_window,
        )
        np.maximum(canopy, patch, out=canopy)
    return canopy


def cell_xy(tree: VTree, grid: RasterGrid) -> tuple[float, float]:
    return (tree.col + 0.5) * grid.pixel_size_m, (
        grid.rows - tree.row - 0.5
    ) * grid.pixel_size_m


def build_site(
    spec: SiteSpec, initial_trees: list[VTree]
) -> tuple[StubCache, RasterGrid]:
    """Stub cache whose tree_base IS the initial forest (baseline scene)."""
    rows, cols, pixel = spec.rows, spec.cols, spec.pixel
    dem = (
        np.fromfunction(lambda r, c: spec.dem_slope * (r + c), (rows, cols))
        .astype(np.float32)
    )
    building = np.zeros((rows, cols), dtype=np.float32)
    for r0, r1, c0, c1, height in spec.blocks:
        building[r0:r1, c0:c1] = height
    grid = RasterGrid(rows, cols, pixel, 0.0, rows * pixel)
    canopy = canopy_of(initial_trees, grid, rows, cols)
    cache = StubCache(rows, cols, pixel, building, dem, canopy,
                      sha=f"stub-sha-v4-{spec.name}")

    scene = _compose_scene_from_canopy(cache, canopy)
    amp = amplitude_of(scene)
    one_step = one_step_patch_indices(spec.scale, amp)
    assert not one_step, (
        f"site {spec.name}: baseline amplitude {amp:.3f} m is one-step at "
        f"{len(one_step)} patch(es) (largest first-step dz "
        f"{float(vmod._patch_first_step_dz(spec.scale).max()):.3f} m) -- the "
        "HIGH-1 pack fence would refuse it; fix the SiteSpec"
    )
    return cache, grid


def scene_of(cache: StubCache, trees: list[VTree], grid: RasterGrid) -> FullSceneTensors:
    return _compose_scene_from_canopy(
        cache, canopy_of(trees, grid, cache.rows, cache.cols)
    )


def initial_forest(spec: SiteSpec) -> list[VTree]:
    """The deterministic baseline forest (ids s0..sn) every case starts at."""
    return [
        VTree(f"s{i}", float(row), float(col), height, radius)
        for i, (row, col, height, radius) in enumerate(spec.base_trees)
    ]


def amplitude_of(scene: FullSceneTensors) -> float:
    return float(
        effective_march_amplitude(
            scene.a, scene.vegdsm, scene.vegdsm2, scene_amaxvalue=scene.amaxvalue
        )
    )


def oracle_bits(cache: StubCache, scene: FullSceneTensors) -> tuple[np.ndarray, np.ndarray]:
    """INDEPENDENT twin: a fresh from-scratch full-tile replay (float32)."""
    full = RasterWindow(0, cache.rows, 0, cache.cols)
    _scalars, vegsh, vbsh, _svftotal = _recompute_veg_svf_window(
        cache, scene, full, torch.zeros((cache.rows, cache.cols)),
        accumulate_window=full,
    )
    return vegsh.numpy(), vbsh.numpy()


def baseline_cube_patches(vegsh: np.ndarray, vbsh: np.ndarray) -> dict[str, np.ndarray]:
    """P2-style patch cubes for ``build_baseline_state`` (V0-pinned inputs)."""
    return {
        "vegshadowmat": vegsh,
        "vbshmat": vbsh,
        "shadowmat": np.zeros_like(vegsh),
    }


def fold_planes(
    vegsh: np.ndarray, vbsh: np.ndarray, scene: FullSceneTensors, rows: int, cols: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The solver's ONE canonical fold over per-patch planes (zero building
    window: its contribution is identical on both sides of the differential)."""
    scalars, _v, _b, svftotal = _fold_veg_svf_from_planes(
        vegshmat=torch.from_numpy(np.ascontiguousarray(vegsh, dtype=np.float32)),
        vbshvegshmat=torch.from_numpy(np.ascontiguousarray(vbsh, dtype=np.float32)),
        vegdem2_read=scene.vegdsm2,
        svf_building_window=torch.zeros((rows, cols)),
        acc_r0=0, acc_r1=rows, acc_c0=0, acc_c1=cols,
        rows=rows, cols=cols,
    )
    return scalars, svftotal


def state_planes(state, cache: StubCache) -> tuple[np.ndarray, np.ndarray]:
    full = RasterWindow(0, cache.rows, 0, cache.cols)
    vegsh, vbsh = state.unpack_planes(full)
    return vegsh.numpy(), vbsh.numpy()


def fold_mismatches(
    left: tuple[dict[str, torch.Tensor], torch.Tensor],
    right: tuple[dict[str, torch.Tensor], torch.Tensor],
) -> list[str]:
    """Names of scalars not BITWISE equal (equal_nan)."""
    bad = [
        name for name in SCALAR_NAMES
        if not np.array_equal(
            left[0][name].numpy(), right[0][name].numpy(), equal_nan=True
        )
    ]
    if not np.array_equal(left[1].numpy(), right[1].numpy(), equal_nan=True):
        bad.append("svftotal")
    return bad


# ---------------------------------------------------------------------------
# Guard-disabling hypotheticals (monkeypatch -- the real physics, no copy)
# ---------------------------------------------------------------------------

_ADVISORY_GUARDS: dict[str, tuple[str, Callable[[], None]]] = {
    "clamp": ("clamped_amplitude_change_reason", lambda: setattr(
        vmod, "clamped_amplitude_change_reason", lambda pre, post: None)),
    "regime": ("_regime_transition_reason", lambda: setattr(
        vmod, "_regime_transition_reason", lambda amp_pre, amp_post, scale: None)),
    "fence": ("_fence_binary_planes", lambda: setattr(
        vmod, "_fence_binary_planes", lambda *args, **kwargs: None)),
}


@contextlib.contextmanager
def guards_disabled(which: tuple[str, ...] = ("clamp", "regime", "fence")):
    """Disable the named advisory guards inside ``veg_svf_state``.

    The bush precondition is NOT disableable here (structural: the corridor
    closure does not model bush occluders); generated scenes keep dem >= 0,
    where bush is provably zero.
    """
    saved = {}
    try:
        for name in which:
            attr, disable = _ADVISORY_GUARDS[name]
            saved[attr] = getattr(vmod, attr)
            disable()
        yield
    finally:
        for attr, value in saved.items():
            setattr(vmod, attr, value)


def classify_refusal(reason: str) -> str:
    """Classify by the REAL reason vocabulary of veg_svf_state guards.

    Order matters and is verified against the actual strings: the clamp
    reason ("clamp-regime amplitude change ...") contains BOTH "clamp" and
    "regime" -- clamp must win first; the value-fence reason ("non-binary
    vbsh plane ... (one-step regime) ...") contains "one-step" -- the fence
    discriminator "non-binary" must beat the regime match.
    """
    if "clamp" in reason:
        return "clamp"
    if "non-binary" in reason:
        return "fence"
    if "one-step" in reason or "regime" in reason:
        return "regime"
    if "bush" in reason:
        return "bush"
    return "other"


# ---------------------------------------------------------------------------
# Seeded op-sequence generation
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    verbs: list[str]
    outcome: str  # "applied" | "refused" | "noop"
    refusal_class: str | None = None
    #: "necessary" | "conservative" | "false_refusal" (refused steps only)
    refusal_verdict: str | None = None
    #: did the firing guard hold its own mandated predicate (independently
    #: recomputed)?  A violation is a guard MISFIRE -- a real defect.
    predicate_held: bool | None = None
    #: oracle vbsh strictly binary at the refused scene?
    oracle_binary: bool | None = None
    absorbed_divergence: bool = False
    vegsh_flips: int = 0
    vbsh_flips: int = 0
    detail: str = ""


@dataclass
class CaseReport:
    site: str
    seed: int
    n_batches: int
    steps: list[StepRecord] = field(default_factory=list)
    defects: list[dict[str, Any]] = field(default_factory=list)
    invariance: str = "not_run"
    roundtrip: str = "not_run"

    def summary(self) -> dict[str, Any]:
        refused = [s for s in self.steps if s.outcome == "refused"]
        return {
            "site": self.site,
            "seed": self.seed,
            "n_batches": self.n_batches,
            "steps": len(self.steps),
            "applied": sum(s.outcome in ("applied", "noop") for s in self.steps),
            "refused": len(refused),
            "refusal_classes": sorted(
                {s.refusal_class for s in refused if s.refusal_class}
            ),
            "false_refusals": sum(s.refusal_verdict == "false_refusal" for s in refused),
            "necessary_refusals": sum(s.refusal_verdict == "necessary" for s in refused),
            "conservative_refusals": sum(
                s.refusal_verdict == "conservative" for s in refused
            ),
            "conservative_by_class": {
                cls: sum(
                    s.refusal_verdict == "conservative" and s.refusal_class == cls
                    for s in refused
                )
                for cls in sorted({s.refusal_class for s in refused if s.refusal_class})
            },
            "absorbed_divergences": sum(s.absorbed_divergence for s in refused),
            "invariance": self.invariance,
            "roundtrip": self.roundtrip,
            "defects": self.defects,
        }


class SequenceGenerator:
    """Seeded batch generator over a scene-level tree list.

    Batches are generated LAZILY (each draw depends on the tree list the
    deterministic op history produced), so a seed fully determines the
    sequence. Verbs mix adds (open ground + on-building clamp-active
    placements), moves, resizes, deletes, replaces and multi-op batches;
    the height sampler deliberately straddles amplitude boundaries: the
    current amplitude (amp-constant transitions), the dz_1 regime
    thresholds of the tile's scale (multi-step -> one-step droppers), and
    the clamp boundary (open-ground dominants carry bound == abs exactly).
    Deletes and resizes reach EVERY tree -- including the baseline
    amplitude driver, which is what makes the regime fence reachable.
    """

    def __init__(self, spec: SiteSpec, trees: list[VTree],
                 rng: np.random.Generator) -> None:
        self.spec = spec
        self.trees = trees
        self.rng = rng
        self.counter = 0
        dz1 = vmod._patch_first_step_dz(spec.scale)
        values = np.sort(dz1[dz1 > 0.0])
        self.dz1_thresholds: list[float] = (
            [float(values[0]), float(np.median(values)), float(values[-1])]
            if values.size else []
        )

    def _next_id(self) -> str:
        self.counter += 1
        return f"t{self.counter}"

    def _ids(self) -> list[str]:
        return sorted(tree.tree_id for tree in self.trees)

    def _tree(self, tree_id: str) -> VTree:
        return next(tree for tree in self.trees if tree.tree_id == tree_id)

    def _open_cell(self, margin: int = 2) -> tuple[int, int]:
        boxes = [(b[0], b[1], b[2], b[3]) for b in self.spec.blocks]
        for _attempt in range(50):
            row = int(self.rng.integers(margin, self.spec.rows - margin))
            col = int(self.rng.integers(margin, self.spec.cols - margin))
            if any(r0 - 1 <= row < r1 + 1 and c0 - 1 <= col < c1 + 1
                   for r0, r1, c0, c1 in boxes):
                continue
            return row, col
        return self.spec.rows // 2, self.spec.cols // 2

    def _block_cell(self) -> tuple[int, int, float]:
        r0, r1, c0, c1, height = self.spec.blocks[
            int(self.rng.integers(len(self.spec.blocks)))
        ]
        return int(self.rng.integers(r0, r1)), int(self.rng.integers(c0, c1)), height

    def _height(self, current_amp: float) -> float:
        u = self.rng.random()
        if u < 0.35:  # below the current amplitude: amp-constant transitions
            return round(float(self.rng.uniform(3.0, max(3.5, current_amp))), 2)
        if u < 0.65:  # raiser
            return round(
                float(self.rng.uniform(current_amp + 0.5, current_amp * 1.6 + 1.0)), 2
            )
        # boundary straddles: exact dz_1 thresholds / exact current amplitude
        target = float(self.rng.choice(self.dz1_thresholds + [current_amp]))
        return round(max(2.0, target + float(self.rng.choice([-0.25, 0.0, 0.25]))), 2)

    def apply_next_batch(self, current_amp: float) -> list[str]:
        """Draw one batch and apply it to the tree list; returns descriptions."""
        roll = self.rng.random()
        ids = self._ids()
        if roll < 0.10 and ids:
            return self._multi(current_amp)
        if roll < 0.24:
            return self._add(current_amp, on_building=True)
        if roll < 0.42:
            return self._add(current_amp, on_building=False)
        if roll < 0.54 and ids:
            return self._move()
        if roll < 0.68 and ids:
            return self._resize(current_amp)
        if roll < 0.80 and ids:
            return self._resize_dominant(current_amp)
        if roll < 0.92 and ids:
            return self._delete()
        if ids:
            return self._replace(current_amp)
        return self._add(current_amp, on_building=False)

    # -- verbs (each returns descriptions; the mutation is already applied) -

    def _add(self, current_amp: float, *, on_building: bool) -> list[str]:
        tree_id = self._next_id()
        if on_building:
            row, col, block_h = self._block_cell()
            height = round(float(self.rng.uniform(3.0, block_h + 6.0)), 2)
        else:
            row, col = self._open_cell()
            height = self._height(current_amp)
        radius = round(float(self.rng.uniform(1.5, 4.0)), 2)
        self.trees.append(
            VTree(tree_id, float(row), float(col), height, radius)
        )
        return [f"add({tree_id}@r{row}c{col},h={height},bld={on_building})"]

    def _move(self) -> list[str]:
        tree_id = str(self.rng.choice(self._ids()))
        row, col = self._open_cell()
        tree = self._tree(tree_id)
        tree.row, tree.col = float(row), float(col)
        return [f"move({tree_id}->r{row}c{col})"]

    def _resize(self, current_amp: float) -> list[str]:
        tree_id = str(self.rng.choice(self._ids()))
        height = self._height(current_amp)
        radius = round(float(self.rng.uniform(1.5, 4.0)), 2)
        tree = self._tree(tree_id)
        tree.height, tree.radius = height, radius
        return [f"resize({tree_id},h={height},r={radius})"]

    def _dominant_ids(self) -> list[str]:
        return [
            tree.tree_id for tree in self.trees
            if tree.height > self.spec.block_max
        ]

    def _resize_dominant(self, current_amp: float) -> list[str]:
        """Dropper/raiser through a DOMINANT tree (the amplitude lever)."""
        dominants = self._dominant_ids()
        if not dominants:
            return self._add(current_amp, on_building=False)
        tree_id = str(self.rng.choice(dominants))
        target = float(
            self.rng.choice(self.dz1_thresholds + [current_amp, self.spec.block_max])
        )
        target = round(max(2.0, target + float(self.rng.choice([-0.25, 0.0, 0.25]))), 2)
        tree = self._tree(tree_id)
        tree.height = target
        return [f"resize_dominant({tree_id},h={target})"]

    def _delete(self) -> list[str]:
        ids = self._ids()
        # dropper bias: half the deletes target a dominant tree, so
        # amplitude DROPPERS (incl. regime-narrowing ones on the regime
        # sites) occur at a useful rate
        dominants = self._dominant_ids()
        if dominants and self.rng.random() < 0.5:
            tree_id = str(self.rng.choice(dominants))
        else:
            tree_id = str(self.rng.choice(ids))
        self.trees = [tree for tree in self.trees if tree.tree_id != tree_id]
        return [f"delete({tree_id})"]

    def _replace(self, current_amp: float) -> list[str]:
        """Delete one tree + add a new one at the same cell, ONE batch."""
        tree_id = str(self.rng.choice(self._ids()))
        new_id = self._next_id()
        old = self._tree(tree_id)
        height = self._height(current_amp)
        radius = round(float(self.rng.uniform(1.5, 4.0)), 2)
        self.trees = [tree for tree in self.trees if tree.tree_id != tree_id]
        self.trees.append(VTree(new_id, old.row, old.col, height, radius))
        return [f"delete({tree_id})", f"add({new_id}@same_cell,h={height})"]

    def _multi(self, current_amp: float) -> list[str]:
        verbs: list[str] = []
        for _ in range(int(self.rng.integers(2, 4))):
            verbs.extend(self.apply_next_batch(current_amp))
        return verbs


# ---------------------------------------------------------------------------
# The sequence driver (properties 1-4)
# ---------------------------------------------------------------------------


def _state_bytes(state) -> dict[str, Any]:
    # snapshots (copies): the refusal non-advance pin must compare the
    # pre-call bytes against the post-call state, not an aliased array
    return {
        "vegsh": state.vegsh_packed.data.copy(),
        "vbsh": state.vbsh_packed.data.copy(),
        "canopy": state.canopy_scene.copy(),
        "versions": state.chunk_versions.copy(),
        "checksums": json.dumps(state.chunk_checksums, sort_keys=True),
    }


def _state_bytes_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        np.array_equal(a["vegsh"], b["vegsh"])
        and np.array_equal(a["vbsh"], b["vbsh"])
        and np.array_equal(a["canopy"], b["canopy"])
        and np.array_equal(a["versions"], b["versions"])
        and a["checksums"] == b["checksums"]
    )


def _defect(kind: str, case: CaseReport, step: int, detail: str) -> dict[str, Any]:
    entry = {
        "kind": kind,
        "site": case.site,
        "seed": case.seed,
        "step": step,
        "detail": detail,
        "repro": (
            f"v4_gate.run_case({case.site!r}, seed={case.seed}, "
            f"n_batches={case.n_batches}, stop_before={step + 1})"
        ),
    }
    case.defects.append(entry)
    return entry


def _check_oracle_alphabets(
    case: CaseReport, step: int, vegsh: np.ndarray, vbsh: np.ndarray
) -> None:
    """Oracle-side value alphabets: vegsh in {0,1}, vbsh in {0,1,2}."""
    if not np.isin(vegsh, (0.0, 1.0)).all():
        _defect("oracle_alphabet_vegsh", case, step, f"vegsh values {np.unique(vegsh)}")
    if not np.isin(vbsh, (0.0, 1.0, 2.0)).all():
        _defect("oracle_alphabet_vbsh", case, step, f"vbsh values {np.unique(vbsh)}")


def _refusal_predicate_held(
    reason_class: str,
    scene_pre: FullSceneTensors,
    scene_post: FullSceneTensors,
    scale: float,
) -> bool | None:
    """Recompute the firing guard's OWN mandated predicate from primitives.

    Independent of the guard code paths (which could carry the exact bug
    this check hunts -- e.g. ADDENDUM 2's MAJOR: the clamp guard missing
    its amplitude-changed conjunct, firing on every edit of a steady-state
    clamped site).  ``None`` = no recomputable predicate (fence: fires iff
    the re-marched planes were non-binary, self-certifying; bush: a
    structural scene property; other: unclassified reason strings).

    - clamp (condition 1 / Case C): A_eff(pre) != A_eff(post) AND
      ``bound > amaxvalue`` on the SMALLER-amplitude scene, with
      ``bound = max(a.max, vegdsm.max, vegdsm2.max) - a.min()``.
    - regime (condition 4 / HIGH-1): A_eff dropped AND some patch's dz_1
      moved from multi-step (<= A_pre) to one-step (> A_post).
    """
    amp_pre = amplitude_of(scene_pre)
    amp_post = amplitude_of(scene_post)
    if reason_class == "clamp":
        if amp_pre == amp_post:
            return False  # fires on an amplitude-CONSTANT pair: misfire
        smaller = scene_pre if amp_pre < amp_post else scene_post
        bound = float(
            torch.maximum(
                torch.maximum(smaller.a.max(), smaller.vegdsm.max()),
                smaller.vegdsm2.max(),
            )
            - torch.min(smaller.a)
        )
        return bool(bound > float(smaller.amaxvalue))
    if reason_class == "regime":
        if amp_post >= amp_pre:
            return False  # regimes widened/held: no narrowing to refuse
        pre_one = set(one_step_patch_indices(scale, amp_pre))
        post_one = set(one_step_patch_indices(scale, amp_post))
        return bool(post_one - pre_one)
    return None


def _continuation_outcome(
    cont_state, cache: StubCache, scene: FullSceneTensors,
    orb_vegsh: np.ndarray, orb_vbsh: np.ndarray, oracle_fold,
) -> tuple[str, bool]:
    """("output_matches" | "bits_diverge" | "fold_diverges", absorbed) for a
    guard-less continuation.

    ``output_matches`` needs BOTH the packed bits (vs oracle != 0) and the
    canonical FOLD to be bitwise the oracle's: an oracle vbsh==2.0 cell
    packs to bit 1, so the != 0 comparison alone would "match" while the
    fold diverges (the state folds 1.0 where the oracle folds 2.0) -- the
    fold comparison is what makes unrepresentable bits strictly necessary.
    ``absorbed``: bits differ but the fold is bitwise equal (clamps ate the
    difference -- the fences did their job, not a false refusal)."""
    cont_vegsh, cont_vbsh = state_planes(cont_state, cache)
    bits = (cont_vegsh, cont_vbsh)
    orb_bits = (orb_vegsh != 0, orb_vbsh != 0)
    planes_match = all(
        np.array_equal(a, b) for a, b in zip(bits, orb_bits)
    )
    fold_bad = fold_mismatches(
        fold_planes(
            cont_vegsh.astype(np.float32), cont_vbsh.astype(np.float32),
            scene, cache.rows, cache.cols,
        ),
        oracle_fold,
    )
    if planes_match and not fold_bad:
        return "output_matches", False
    if not fold_bad:
        return "bits_diverge", True  # absorbed divergence
    return ("fold_diverges", False)


def run_case(
    site_name: str,
    seed: int,
    n_batches: int,
    *,
    store_root: str | Path | None = None,
    stop_before: int | None = None,
    verbose: bool = False,
) -> CaseReport:
    """Run one seeded op sequence; returns the case report.

    ``stop_before`` truncates the sequence (defect reproduction: re-run the
    pinned prefix that constructs the failure).
    """
    spec = SITES[site_name]
    rng = np.random.default_rng(seed)
    case = CaseReport(site=site_name, seed=seed, n_batches=n_batches)
    initial_trees = initial_forest(spec)
    cache, grid = build_site(spec, initial_trees)
    trees = list(initial_trees)  # the generator mutates this list

    scene = scene_of(cache, trees, grid)
    veg0, vb0 = oracle_bits(cache, scene)
    _check_oracle_alphabets(case, -1, veg0, vb0)
    cache.svf_patches.update(baseline_cube_patches(veg0, vb0))
    state = build_baseline_state(cache)
    baseline_state = build_baseline_state(cache)  # for the direct invariance path

    store = VegOcclusionStore(cache, state_root=store_root) if store_root else None
    roundtrip_step = n_batches // 2 if store is not None else -1
    generator = SequenceGenerator(spec, trees, rng)

    for step in range(n_batches):
        if stop_before is not None and step >= stop_before:
            break
        verbs = generator.apply_next_batch(amplitude_of(scene))
        scene = scene_of(cache, generator.trees, grid)
        record = StepRecord(step=step, verbs=verbs, outcome="applied")

        orb_vegsh, orb_vbsh = oracle_bits(cache, scene)
        _check_oracle_alphabets(case, step, orb_vegsh, orb_vbsh)
        oracle_fold = fold_planes(orb_vegsh, orb_vbsh, scene, cache.rows, cache.cols)

        before = _state_bytes(state)
        try:
            state, telemetry = apply_edit_batch(state, cache, scene)
        except VegOcclusionFallback as refusal:
            reason_class = classify_refusal(refusal.reason)
            record.outcome = "refused"
            record.refusal_class = reason_class
            record.predicate_held = _refusal_predicate_held(
                reason_class,
                vmod._compose_scene_from_canopy(cache, state.canopy_scene),
                scene,
                1.0 / float(cache.pixel_size_m),
            )
            record.oracle_binary = not bool((orb_vbsh == 2.0).any())
            if not _state_bytes_equal(before, _state_bytes(state)):
                _defect("refusal_advanced_state", case, step, refusal.reason)
            # Property 3: necessity via the guard-less hypothetical.
            cont_class = "none"
            with guards_disabled():
                try:
                    cont_state, _tel = apply_edit_batch(state, cache, scene)
                except VegOcclusionFallback as cont_refusal:
                    cont_state = None
                    # bush is the only structural guard left enabled; any
                    # OTHER surviving refusal means a 5th guard appeared or
                    # the monkeypatch broke -- never silently "necessary"
                    cont_class = classify_refusal(cont_refusal.reason)
            if cont_state is None:
                record.refusal_verdict = "necessary"
                if cont_class != "bush":
                    _defect(
                        "harness_integrity", case, step,
                        f"continuation refused by a NON-structural guard "
                        f"({cont_class}) with all advisory guards disabled: "
                        f"{cont_refusal.reason}",
                    )
                record.detail = f"guarded even with all advisory guards off ({cont_class})"
            else:
                outcome, absorbed = _continuation_outcome(
                    cont_state, cache, scene, orb_vegsh, orb_vbsh, oracle_fold
                )
                record.absorbed_divergence = absorbed
                if outcome == "output_matches":
                    if record.predicate_held is False:
                        record.refusal_verdict = "false_refusal"
                        _defect(
                            "false_refusal", case, step,
                            f"guard-less continuation is bitwise the oracle "
                            f"AND the {reason_class} guard fired OUTSIDE its "
                            f"mandated predicate (misfire): {refusal.reason}",
                        )
                    else:
                        record.refusal_verdict = "conservative"
                        record.detail = (
                            "conservative: continuation is bitwise the oracle "
                            f"but the {reason_class} guard fired inside its "
                            "mandated predicate (design-mandated: ADDENDUM 2 "
                            "condition " + (
                                "1/Case C" if reason_class == "clamp"
                                else "4/HIGH-1" if reason_class == "regime"
                                else "4 value fence"
                            ) + f"); oracle vbsh 2.0 present: "
                            f"{not record.oracle_binary}"
                        )
                else:
                    record.refusal_verdict = "necessary"
                    record.detail = (
                        "necessary: continuation planes differ"
                        + (" (fold-equal: absorbed by clamps)" if absorbed else "")
                    )
                if record.predicate_held is False and outcome != "output_matches":
                    _defect(
                        "refusal_outside_predicate", case, step,
                        f"{reason_class} guard fired outside its mandated "
                        f"predicate (continuation {outcome}): {refusal.reason}",
                    )
            if verbose:
                print(f"    [step {step}] REFUSED({reason_class}): "
                      f"{refusal.reason[:90]}...")
            # Property 4 across a REFUSAL boundary: the crash/restart idiom
            # must also hold when the process dies right after the last
            # successful commit -- a fresh store recovers the committed
            # (pre-refusal) state and re-derives the same no-op.
            if store is not None and step == roundtrip_step:
                store.commit(state)
                fresh = VegOcclusionStore(cache, state_root=store_root)
                pre_scene = vmod._compose_scene_from_canopy(
                    cache, state.canopy_scene
                )
                loaded, _tel = fresh.prepare(pre_scene)
                if loaded is None:
                    _defect("serialization_drift", case, step,
                            "fresh store could not recover the committed "
                            "state across a refusal boundary")
                elif not _state_bytes_equal(
                    _state_bytes(loaded), _state_bytes(state)
                ):
                    _defect("serialization_drift", case, step,
                            "loaded bytes differ after refusal-boundary "
                            "round trip")
                else:
                    case.roundtrip = "ok"
            case.steps.append(record)
            continue

        if telemetry["patches_remarched"] == 0:
            record.outcome = "noop"

        # Property 1: soundness of the accepted state.
        if (orb_vbsh == 2.0).any():
            _defect(
                "alphabet_breach", case, step,
                "state path ACCEPTED a batch whose oracle vbsh reaches 2.0 "
                "(one-step regime) -- a typed refusal was mandatory",
            )
        st_vegsh, st_vbsh = state_planes(state, cache)
        if not np.array_equal(st_vegsh, orb_vegsh != 0):
            _defect("silent_divergence_vegsh", case, step,
                    "state vegsh planes != oracle")
        if not np.array_equal(st_vbsh, orb_vbsh != 0):
            _defect("silent_divergence_vbsh", case, step,
                    "state vbsh planes != oracle")
        bad = fold_mismatches(
            fold_planes(st_vegsh.astype(np.float32), st_vbsh.astype(np.float32),
                        scene, cache.rows, cache.cols),
            oracle_fold,
        )
        if bad:
            _defect("silent_divergence_fold", case, step,
                    f"state fold != oracle fold, scalars {bad}")
        record.vegsh_flips = int(telemetry["vegsh_flips"])
        record.vbsh_flips = int(telemetry["vbsh_flips"])
        if verbose:
            print(
                f"    [step {step}] {'noop' if record.outcome == 'noop' else 'applied'} "
                f"{'+'.join(v.split('(')[0] for v in verbs)}: remarched "
                f"{telemetry['patches_remarched']}, flips "
                f"{record.vegsh_flips}/{record.vbsh_flips}"
            )

        # Property 4: serialization round trip mid-sequence.
        if store is not None and step == roundtrip_step:
            store.commit(state)
            fresh = VegOcclusionStore(cache, state_root=store_root)
            loaded, _tel = fresh.prepare(scene)
            if loaded is None:
                _defect("serialization_drift", case, step,
                        "fresh store refused its own commit")
            elif not _state_bytes_equal(_state_bytes(loaded), _state_bytes(state)):
                _defect("serialization_drift", case, step,
                        "loaded bytes differ after round trip")
            else:
                case.roundtrip = "ok"

        case.steps.append(record)

    # Property 2: composition invariance at the final scene (B1+...+Bn as
    # ONE direct batch from the fresh baseline vs the incremental path's
    # state caught up to the final scene).
    if stop_before is None and n_batches > 0:
        case.invariance = _check_invariance(
            case, cache, baseline_state, state, scene, verbose=verbose
        )
    return case


def _check_invariance(
    case: CaseReport,
    cache: StubCache,
    baseline_state,
    incremental_state,
    scene: FullSceneTensors,
    *,
    verbose: bool = False,
) -> str:
    orb_vegsh, orb_vbsh = oracle_bits(cache, scene)
    oracle_fold = fold_planes(orb_vegsh, orb_vbsh, scene, cache.rows, cache.cols)
    verdict = "ok"

    def check(label: str, applied_state) -> None:
        nonlocal verdict
        vegsh_p, vbsh_p = state_planes(applied_state, cache)
        if not (np.array_equal(vegsh_p, orb_vegsh != 0)
                and np.array_equal(vbsh_p, orb_vbsh != 0)):
            _defect("invariance_break", case, case.n_batches,
                    f"{label} planes != oracle at the final scene")
            verdict = "MISMATCH"
            return
        bad = fold_mismatches(
            fold_planes(vegsh_p.astype(np.float32), vbsh_p.astype(np.float32),
                        scene, cache.rows, cache.cols),
            oracle_fold,
        )
        if bad:
            _defect("invariance_break", case, case.n_batches,
                    f"{label} fold != oracle fold, scalars {bad}")
            verdict = "MISMATCH"

    try:
        direct, _tel = apply_edit_batch(baseline_state, cache, scene)
    except VegOcclusionFallback as refusal:
        verdict = "direct_refused"
        if verbose:
            print(f"    [invariance] direct path refused "
                  f"({classify_refusal(refusal.reason)})")
    else:
        check("direct", direct)

    # the incremental side must describe the FINAL scene: catch up if the
    # last batch was refused (production semantics -- lagged batches apply)
    if not np.array_equal(incremental_state.canopy_scene, scene.canopy.numpy()):
        try:
            incremental_state, _tel = apply_edit_batch(
                incremental_state, cache, scene
            )
        except VegOcclusionFallback as refusal:
            verdict = (
                "both_refused" if verdict == "direct_refused"
                else "incremental_refused"
            )
            if verbose:
                print(f"    [invariance] incremental catch-up refused "
                      f"({classify_refusal(refusal.reason)})")
            return verdict
    check("incremental", incremental_state)
    return verdict


# ---------------------------------------------------------------------------
# Site specs + profiles
# ---------------------------------------------------------------------------


SITES: dict[str, SiteSpec] = {
    # dz1_max per scale: 0.5 -> 11.59 m, 0.25 -> 23.18 m, 0.4 -> 14.49 m,
    # 1.0 -> 5.79 m, 1/3 -> 17.38 m. Two site FAMILIES, both needed:
    # (a) block_max > dz1_max: the packable-baseline constraint pins the
    #     amplitude floor above every dz_1, so the regime fence is
    #     structurally unreachable there (a dropper lands on block_max);
    #     these tiles exercise clamp Case A/B/C and the fold spine.
    # (b) regime sites: block_max < dz1_max < baseline amplitude (a TALL
    #     baked dominant carries the baseline) -- droppers (delete/resize of
    #     the dominant) narrow multi-step patches to one-step and the
    #     regime fence MUST refuse (review HIGH-1's exact hazard).
    "main2m": SiteSpec(
        name="main2m", rows=60, cols=60, pixel=2.0,
        blocks=((4, 12, 4, 12, 12.0), (40, 50, 38, 48, 4.0)),
        base_trees=((30, 30, 6.0, 3.0), (12, 45, 8.0, 3.5),
                    (44, 14, 5.0, 2.5), (50, 50, 7.0, 2.0)),
    ),
    "nonsquare4m": SiteSpec(
        name="nonsquare4m", rows=40, cols=80, pixel=4.0,
        blocks=((4, 12, 4, 12, 28.0),),
        base_trees=((10, 60, 20.0, 3.0), (30, 20, 18.0, 4.0), (20, 40, 24.0, 3.0)),
    ),
    "scale04_2p5m": SiteSpec(
        name="scale04_2p5m", rows=48, cols=48, pixel=2.5,
        blocks=((4, 12, 4, 12, 16.0),),
        base_trees=((24, 24, 12.0, 3.0), (10, 36, 10.0, 4.0), (38, 10, 9.0, 3.0)),
    ),
    "scale1_1m_relief": SiteSpec(
        name="scale1_1m_relief", rows=44, cols=36, pixel=1.0, dem_slope=0.03,
        blocks=((4, 12, 4, 12, 8.0),),
        base_trees=((20, 18, 6.0, 2.0), (8, 28, 5.0, 3.0), (34, 6, 7.0, 2.5)),
    ),
    "coarse3m": SiteSpec(
        name="coarse3m", rows=40, cols=40, pixel=3.0,
        blocks=((4, 12, 4, 12, 20.0),),
        base_trees=((24, 24, 15.0, 2.0), (10, 32, 12.0, 2.0), (30, 10, 10.0, 2.0)),
    ),
    # regime sites: baked dominants carry the baseline ABOVE dz1_max
    "regime4m": SiteSpec(
        name="regime4m", rows=60, cols=60, pixel=4.0,
        blocks=((4, 12, 4, 12, 12.0), (40, 50, 38, 48, 4.0)),
        # 27 m dominant at (45, 10) [the reviewer-repro construction] +
        # low crowns; baseline amp 27 > dz1_max 23.18, packable
        base_trees=((45, 10, 27.0, 3.0), (30, 30, 6.0, 3.0), (12, 45, 8.0, 3.0)),
    ),
    "regime3m": SiteSpec(
        name="regime3m", rows=40, cols=40, pixel=3.0,
        blocks=((4, 12, 4, 12, 10.0),),
        # baseline amp 20 > dz1_max 17.38; dropping the dominant -> 10 m
        base_trees=((24, 24, 20.0, 2.0), (10, 32, 8.0, 2.0), (30, 10, 6.0, 2.0)),
    ),
}

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"sequences": {name: 1 for name in SITES}, "batches": (3, 5)},
    "suite": {"sequences": {name: 2 for name in SITES}, "batches": (4, 8)},
    "thorough": {"sequences": {name: 6 for name in SITES}, "batches": (6, 12)},
}

_ROOT_COUNTER = itertools.count()


def roundtrip_root(site: str, seed: int) -> str:
    """A FRESH store root per case: leftover revisions of an earlier case on
    the same site key must never be loaded by a later one."""
    return tempfile.mkdtemp(prefix=f"r4b_v4_{site}_{seed}_{next(_ROOT_COUNTER)}_")


def run_sweep(profile: str, *, verbose: bool = True) -> dict[str, Any]:
    """Run the profile's randomized sweep; returns the evidence dict."""
    plan = PROFILES[profile]
    cases: list[CaseReport] = []
    started = time.perf_counter()
    for site_name, n_sequences in plan["sequences"].items():
        for i in range(n_sequences):
            seed = _BASE_SEED + i
            lo, hi = plan["batches"]
            n_batches = int(lo + (hi - lo) * (i % 3) / 2)
            case = run_case(
                site_name, seed, n_batches,
                store_root=roundtrip_root(site_name, seed),
                verbose=verbose,
            )
            cases.append(case)
            if verbose:
                s = case.summary()
                print(
                    f"[case] {site_name} seed={seed} steps={s['steps']} "
                    f"applied={s['applied']} refused={s['refused']} "
                    f"classes={s['refusal_classes']} "
                    f"false_refusals={s['false_refusals']} "
                    f"conservative={s['conservative_refusals']} "
                    f"invariance={s['invariance']} "
                    f"roundtrip={s['roundtrip']} defects={len(s['defects'])}"
                )
    elapsed = time.perf_counter() - started
    totals = {
        "profile": profile,
        "base_seed": _BASE_SEED,
        "cases": len(cases),
        "sites": len(plan["sequences"]),
        "steps": sum(len(c.steps) for c in cases),
        "applied": sum(c.summary()["applied"] for c in cases),
        "refused": sum(c.summary()["refused"] for c in cases),
        "false_refusals": sum(c.summary()["false_refusals"] for c in cases),
        "necessary_refusals": sum(c.summary()["necessary_refusals"] for c in cases),
        "conservative_refusals": sum(
            c.summary()["conservative_refusals"] for c in cases
        ),
        "conservative_by_class": {
            cls: sum(c.summary()["conservative_by_class"].get(cls, 0) for c in cases)
            for cls in sorted(
                {
                    cls
                    for c in cases
                    for cls in c.summary()["conservative_by_class"]
                }
            )
        },
        "absorbed_divergences": sum(c.summary()["absorbed_divergences"] for c in cases),
        "invariance_ok": sum(c.invariance.startswith("ok") or c.invariance == "ok" for c in cases),
        "invariance_direct_refused": sum(c.invariance == "direct_refused" for c in cases),
        "invariance_incremental_refused": sum(
            c.invariance == "incremental_refused" for c in cases
        ),
        "invariance_both_refused": sum(c.invariance == "both_refused" for c in cases),
        "roundtrip_ok": sum(c.roundtrip == "ok" for c in cases),
        "refusal_classes_observed": sorted(
            {cls for c in cases for cls in c.summary()["refusal_classes"]}
        ),
        "defects": [d for case in cases for d in case.defects],
        "wall_seconds": round(elapsed, 1),
    }
    return {"totals": totals, "cases": [c.summary() for c in cases]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=("smoke", "suite", "thorough"),
                        default=os.environ.get("R4B_SWEEP_SCALE", "thorough"))
    parser.add_argument("--out", default="/tmp/r4b_proof/v4_opsequence_gate.json")
    args = parser.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result = run_sweep(args.profile)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1, default=str)
    totals = result["totals"]
    print(
        f"=== V4 op-sequence gate: profile={totals['profile']} "
        f"cases={totals['cases']} steps={totals['steps']} "
        f"applied={totals['applied']} refused={totals['refused']} "
        f"false_refusals={totals['false_refusals']} "
        f"conservative={totals['conservative_refusals']} "
        f"defects={len(totals['defects'])} ({totals['wall_seconds']}s)"
    )
    print(f"wrote {args.out}")
    return 1 if totals["defects"] or totals["false_refusals"] else 0


if __name__ == "__main__":
    sys.exit(main())
