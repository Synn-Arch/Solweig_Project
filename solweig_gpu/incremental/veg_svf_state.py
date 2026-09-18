# SPDX-License-Identifier: GPL-3.0-only
"""Persistent vegetation SVF occluder state (R4 Phase A).

Design: ``docs/incremental_design_tool/realtime_collaboration/design/
r4-veg-svf-occlusion-state.md`` (+ ADDENDUM / ADDENDUM 2 / ADDENDUM 3,
binding).

The vegetation terms of the SVF bundle are BINARY per-(cell, patch)
classifications (``vegshmat``/``vbshmat``); every float scalar the physics
consumes is an order-dependent re-fold of those bits. So the persistent
state is the BITS (packed, ~4.77 MiB/stack at 500x500) plus the canopy
snapshot they describe — never float accumulators. An edit batch
recomputes bits only along per-patch CORRIDORS (rays through the changed
composed-surface cells) on W2 reach-expanded march windows, and the
scalars are re-folded through the solver's single canonical fold
(:func:`solweig_gpu.incremental.solver._fold_veg_svf_from_planes`) —
bitwise-exact by construction.

Perf framing (review LOW-1, stated honestly): the corridor path saves
against the FULL-TILE replay only. At 96x96 @2 m a corridor apply
(~0.38-0.45 s, all 153 patches re-marched at this amplitude) beat the
full-tile replay (~0.51-0.59 s) but was ~1.3x SLOWER than the
deployed-shape W2 windowed replay (read 66x78, accumulate 24x30,
~0.32-0.34 s) — the small tile keeps every patch's closure non-empty
and the W2 replay's windows are already tiny. The corridor shape pays
off as tiles grow (per-patch march windows scale with the closure bbox,
not the read window); the path stays correctness-first: every refusal
routes the replay.

Binding conditions (ADDENDUM 2), every refusal typed:

1. clamp-regime fallback — :func:`clamped_amplitude_change_reason`:
   fallback iff ``A_eff(S) != A_eff(S')`` AND
   ``bound > scene_amaxvalue`` on the smaller-``A_eff`` scene, with
   ``bound = max(a.max, vegdsm.max, vegdsm2.max) - a.min()`` evaluated on
   the FULL-TILE composed pre/post scenes with
   :func:`~solweig_gpu.incremental.solver.effective_march_amplitude`'s
   exact expression; amplitude-constant transitions never fall back;
2. per-patch re-march set = ``corridor_p`` (union of pre/post
   global-effective-amplitude march offsets) ``U C``, ``C`` from composed
   surface diffs (``vegdsm``/``vegdsm2``); ``bush == 0`` asserted on pre
   AND post scenes, else fallback;
3. marches run on ``_patch_march_window`` reach-expanded windows — NEVER
   the raw corridor bbox (raw bboxes diverge; the W2 windows reproduce
   the replay bitwise). DEVIATION from the letter (ADDENDUM 3,
   adjudicated compliant-as-implemented): the march amplitude is the
   POST scene's full-tile effective amplitude, not the march-window
   crop. The windowed letter is bit-equal at closure cells but
   value-divergent in one-step regimes (vbsh 2.0 vs 1.0), which would
   both trip the value fence spuriously and mask real 2.0s; the fold
   consumes VALUES, so the corridor march replicates the replay's
   amplitude exactly (a superset of steps is safe: corridor one-step
   implies replay one-step for any read window);
4. after every corridor re-march (and before baseline packing) the float
   ``vegsh``/``vbsh`` planes must be within {0, 1}; on violation the batch
   falls back to the full replay (the reviewer's ``vbsh == 2.0``
   counterexample lives exactly here). REGIME level (review HIGH-1):
   a multi-step -> one-step amplitude transition refuses the batch even
   though the clamp guard (condition 1) is silent — oracle 2.0 cells can
   appear at UNCHANGED trees outside every corridor closure — and a
   baseline already one-step anywhere refuses the pack
   (:func:`_regime_transition_reason`, :func:`one_step_patch_indices`).

Every refusal raises :class:`VegOcclusionFallback` (a ``RuntimeError``):
the caller routes the batch to today's proven full replay. The solver
seam (``window_svf_bundle(veg_state=...)``) additionally validates the
state against the cache/scene and raises
:class:`~solweig_gpu.incremental.solver.StaleSvfError` on breach — loud,
never silent corruption.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from solweig_gpu.incremental.bitmask import (
    PackedVisibility,
    pack_visibility,
    set_patch_window,
    spatial_window_view,
    unpack_patch,
    unpack_visibility,
)
from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.solver import (
    FullSceneTensors,
    StaleSvfError,
    _compose_scene_from_canopy,
    _patch_march_reach_pixels,
    _patch_march_window,
    _sky_patch_geometry,
    effective_march_amplitude,
)
from solweig_gpu.shadow import shadow as shadow_fn
from solweig_gpu.incremental import march_router

#: Bumped whenever the on-disk layout or the packed-bit semantics change.
SCHEMA_VERSION = 1

_PATCH_OPTION = 2
_CHUNK_CELLS = 64

_STACKS = ("vegsh", "vbsh")
_DENSE_STACK_NAMES = {"vegsh": "vegshadowmat", "vbsh": "vbshmat"}


class VegOcclusionFallback(RuntimeError):
    """Typed refusal: this batch must route the full-tile replay.

    Raised by every guard/fence above. Never caught inside this module's
    public API (``apply_edit_batch``/``build_baseline_state``); the
    :class:`VegOcclusionStore` routing layer catches it and reports the
    reason through telemetry.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class VegOcclusionStateError(RuntimeError):
    """Persisted state failed to load/validate (corruption, schema, key).

    Load-side counterpart of :class:`VegOcclusionFallback`: the store
    discards the state and rebuilds from the cache baseline; wrong bits
    are never served.
    """


# ---------------------------------------------------------------------------
# Key + patch geometry identity
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def patch_geometry_id() -> str:
    """Stable id of the memoized sky-patch geometry (R5 key member)."""
    patches, _rings = _sky_patch_geometry(_PATCH_OPTION)
    digest = hashlib.sha256()
    for altitude, azimuth, ring in patches:
        digest.update(
            f"{float(altitude)!r}|{float(azimuth)!r}|{int(ring)}\n".encode()
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class VegOcclusionKey:
    """R5 versioned-dependency-cache key shape (design section 2)."""

    site_id: str
    tile_key: str
    cache_manifest_sha256: str
    scene_revision: int
    patch_geometry_id: str
    schema_version: int = SCHEMA_VERSION


def state_key_for(cache: Any, scene_revision: int) -> VegOcclusionKey:
    metadata = cache.metadata()
    return VegOcclusionKey(
        site_id=str(metadata["site_id"]),
        tile_key=str(metadata["tile_key"]),
        cache_manifest_sha256=str(metadata["manifest_sha256"]),
        scene_revision=int(scene_revision),
        patch_geometry_id=patch_geometry_id(),
        schema_version=SCHEMA_VERSION,
    )


def _key_matches_cache(key: VegOcclusionKey, cache: Any) -> bool:
    expected = state_key_for(cache, key.scene_revision)
    identity = (
        "site_id",
        "tile_key",
        "cache_manifest_sha256",
        "patch_geometry_id",
        "schema_version",
    )
    return all(getattr(key, name) == getattr(expected, name) for name in identity)


# ---------------------------------------------------------------------------
# March geometry (shadow()'s exact executed steps) + corridors
# ---------------------------------------------------------------------------


def march_offsets(
    azimuth,
    altitude,
    amaxvalue: float,
    scale: float,
    sizex: int,
    sizey: int,
):
    """Per-step (dx, dy) replicating shadow()'s while-loop bit for bit.

    ``azimuth``/``altitude`` are the 0-dim float32 tensors the replay
    passes to ``shadow()``; ``amaxvalue`` the float amplitude of the
    while-stop. Returns the (dx, dy) integer shifts of every EXECUTED
    step (proofs/corridor_bruteforce.py, verified against the march).
    """
    degrees = torch.pi / 180.0
    az = azimuth
    if float(az) == 0.0:
        az = az * 0.0 + 1e-12
    az = az * degrees
    alt = altitude * degrees
    dx = torch.tensor(0.0, dtype=torch.float32)
    dy = torch.tensor(0.0, dtype=torch.float32)
    dz = torch.tensor(0.0, dtype=torch.float32)
    amax = torch.tensor(float(amaxvalue), dtype=torch.float32)

    pibyfour = torch.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = torch.sin(az)
    cosazimuth = torch.cos(az)
    tanazimuth = torch.tan(az)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1.0 / sinazimuth))
    dscos = torch.abs((1.0 / cosazimuth))
    tanaltitudebyscale = torch.tan(alt) / scale

    index = 1.0
    offsets = []
    while bool(amax >= dz) and bool(torch.abs(dx) < sizex) and bool(
        torch.abs(dy) < sizey
    ):
        if bool(pibyfour <= az < threetimespibyfour) or bool(
            fivetimespibyfour <= az < seventimespibyfour
        ):
            dy = signsinazimuth * index
            dx = -1.0 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos
        dz = ds * index * tanaltitudebyscale
        offsets.append((int(dx), int(dy)))
        index += 1.0
    return offsets


def corridor_mask(C: np.ndarray, offsets) -> np.ndarray:
    """{t : t + (dx_s, dy_s) in C for some executed step s}.

    The target ``t`` reads occluders at ``t + (dx_s, dy_s)`` per step;
    out-of-bounds reads are dropped (they contribute nothing).
    """
    shape = C.shape
    out = np.zeros(shape, dtype=bool)
    for dx, dy in offsets:
        r0, r1 = max(0, -dx), shape[0] - max(0, dx)
        c0, c1 = max(0, -dy), shape[1] - max(0, dy)
        if r0 >= r1 or c0 >= c1:
            continue
        out[r0:r1, c0:c1] |= C[r0 + dx : r1 + dx, c0 + dy : c1 + dy]
    return out


def _patch_first_step_dz(scale: float) -> np.ndarray:
    """dz of the FIRST march step per patch, association-matched arithmetic.

    Step 1 always executes (dz starts at 0), so a patch at amplitude ``A``
    is in the ONE-STEP regime iff ``dz_1 > A`` — the regime whose vbsh can
    reach 2.0 (the step-1 raise zeroes the accumulator after its only
    add). Pure arithmetic: the same branch/ds expressions as
    :func:`march_offsets`, and the SAME association for dz
    (``ds * index * (tan(alt)/scale)`` with ``tan/scale`` folded first —
    bitwise-equal to the naive ``ds*tan(alt)/scale`` only at power-of-two
    scales, review NEW-LOW); the 90-degree zenith patch is
    excluded (no march). Reviewer-verified regime split at 4 m pixels:
    one-step iff dz_1 > A lands exactly on the 66/78-degree rings.
    """
    patches, _rings = _sky_patch_geometry(_PATCH_OPTION)
    degrees = torch.pi / 180.0
    pibyfour = torch.pi / 4.0
    out = np.zeros(len(patches), dtype=np.float32)
    for i, patch in enumerate(patches):
        altitude, azimuth = patch[0], patch[1]
        if float(altitude) >= 90.0:
            out[i] = -1.0  # sentinel: the zenith patch never marches
            continue
        az = azimuth
        if float(az) == 0.0:
            az = az * 0.0 + 1e-12
        az = az * degrees
        alt = altitude * degrees
        sinazimuth = torch.sin(az)
        cosazimuth = torch.cos(az)
        if bool(pibyfour <= az < 3.0 * pibyfour) or bool(
            5.0 * pibyfour <= az < 7.0 * pibyfour
        ):
            ds = torch.abs(1.0 / sinazimuth)
        else:
            ds = torch.abs(1.0 / cosazimuth)
        # association-matched to the march: dz = ds * index * tbs with
        # tbs = tan(alt)/scale computed FIRST (review NEW-LOW) — the naive
        # ds*tan(alt)/scale is 1 ulp off at non-power-of-two scales
        tanaltitudebyscale = torch.tan(alt) / scale
        out[i] = float(ds * 1.0 * tanaltitudebyscale)
    return out


def one_step_patch_indices(scale: float, amplitude: float) -> list[int]:
    """Patch indices whose march is ONE-STEP at ``amplitude`` (dz_1 > A)."""
    dz1 = _patch_first_step_dz(scale)
    return [int(i) for i in np.nonzero(dz1 > float(amplitude))[0]]


def _regime_transition_reason(
    amp_pre: float, amp_post: float, scale: float
) -> str | None:
    """Condition 4, regime level: multi-step -> one-step anywhere refuses.

    The per-window binary fence cannot see a regime narrowing: vbsh==2.0
    appears at UNCHANGED cells outside every corridor closure (the state's
    bits there are stale and the fold silently diverges from the oracle —
    review HIGH-1). One-step -> multi-step (amplitude growth) only WIDENS
    regimes and stays safe; amplitude-constant transitions change nothing.
    """
    if amp_post >= amp_pre:
        return None  # regimes only widen (or hold): no new one-step patch
    dz1 = _patch_first_step_dz(scale)
    narrowed = np.nonzero((dz1 > float(amp_post)) & (dz1 <= float(amp_pre)))[0]
    if narrowed.size == 0:
        return None
    worst = float(dz1[narrowed].max())
    return (
        f"march regime narrowed to one-step at {int(narrowed.size)} patch(es) "
        f"(first-step dz up to {worst:.3f} m > post effective amplitude "
        f"{amp_post:.3f} m; pre amplitude {amp_pre:.3f} m was multi-step "
        "there): unchanged cells outside every corridor closure can carry "
        "oracle vbsh==2.0 the packed bits cannot represent — full replay "
        "required for this batch"
    )


# ---------------------------------------------------------------------------
# Condition 1: clamp-regime guard predicate
# ---------------------------------------------------------------------------


def _scene_amplitude(scene: FullSceneTensors) -> float:
    return float(
        effective_march_amplitude(
            scene.a, scene.vegdsm, scene.vegdsm2, scene_amaxvalue=scene.amaxvalue
        )
    )


def _scene_relative_bound(scene: FullSceneTensors) -> float:
    bound = torch.maximum(
        torch.maximum(scene.a.max(), scene.vegdsm.max()), scene.vegdsm2.max()
    )
    return float(bound - torch.min(scene.a))


def clamped_amplitude_change_reason(
    scene_pre: FullSceneTensors, scene_post: FullSceneTensors
) -> str | None:
    """Condition 1 predicate on FULL-TILE composed scenes.

    Returns the refusal reason iff the effective march amplitude changed
    AND the clamp (``bound > scene_amaxvalue``) is active on the
    smaller-amplitude scene — the corridor-sufficiency proof's Case C.
    Returns ``None`` for amplitude-constant transitions (Case A, proven
    safe) and amplitude changes whose smaller scene is unclamped
    (Case B, proven safe; ``bound == abs`` boundary included, strict >).
    """
    amp_pre = _scene_amplitude(scene_pre)
    amp_post = _scene_amplitude(scene_post)
    if amp_pre == amp_post:
        return None
    smaller = scene_pre if amp_pre < amp_post else scene_post
    smaller_amp = min(amp_pre, amp_post)
    bound = _scene_relative_bound(smaller)
    abs_value = float(smaller.amaxvalue)
    if bound > abs_value:
        return (
            "clamp-regime amplitude change: effective march amplitude "
            f"{amp_pre!r} -> {amp_post!r} with the smaller-amplitude scene "
            f"clamped (bound {bound!r} > scene_amaxvalue {abs_value!r} at "
            f"effective amplitude {smaller_amp!r}); corridor sufficiency is "
            "Case C — full replay required for this batch"
        )
    return None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _chunk_grid_index(rows: int, cols: int) -> tuple[int, int]:
    return (rows + _CHUNK_CELLS - 1) // _CHUNK_CELLS, (
        cols + _CHUNK_CELLS - 1
    ) // _CHUNK_CELLS


def _chunk_slices(rows: int, cols: int):
    for r0 in range(0, rows, _CHUNK_CELLS):
        for c0 in range(0, cols, _CHUNK_CELLS):
            yield (
                slice(r0, min(r0 + _CHUNK_CELLS, rows)),
                slice(c0, min(c0 + _CHUNK_CELLS, cols)),
            )


def _chunks_overlapping(window: np.ndarray | bool, rows: int, cols: int):
    """Chunk (row_slice, col_slice) list overlapping a boolean mask."""
    hits = []
    for chunk in _chunk_slices(rows, cols):
        if window[chunk[0], chunk[1]].any():
            hits.append(chunk)
    return hits


@dataclass
class VegOcclusionState:
    """Packed per-(cell, patch) vegetation occlusion bits for one scene.

    ``canopy_scene`` is the clamped canopy raster the bits describe; the
    pre scene of a later batch is recomposed from it
    (:func:`apply_edit_batch` works on any pre/post pair, so lagged
    batches catch up). Chunk versions/checksums fingerprint the packed
    bytes per 64x64 chunk so corruption is detected at load, never
    served.
    """

    key: VegOcclusionKey
    vegsh_packed: PackedVisibility
    vbsh_packed: PackedVisibility
    canopy_scene: np.ndarray
    chunk_versions: np.ndarray
    chunk_checksums: dict[str, list[str]]
    rows: int
    cols: int
    #: Seconds spent packing the baseline (telemetry only; 0.0 otherwise).
    pack_seconds: float = 0.0

    @property
    def patch_count(self) -> int:
        return self.vegsh_packed.patch_count

    def _packed(self, stack: str) -> PackedVisibility:
        if stack == "vegsh":
            return self.vegsh_packed
        if stack == "vbsh":
            return self.vbsh_packed
        raise ValueError(f"unknown occlusion stack {stack!r}")

    def bit_plane(self, stack: str, patch_index: int) -> np.ndarray:
        """One boolean (rows, cols) plane (``True`` == stored 1)."""
        return unpack_patch(self._packed(stack), patch_index)

    def bit_window(self, stack: str, window: RasterWindow) -> np.ndarray:
        packed = self._packed(stack)
        view = spatial_window_view(
            packed,
            slice(window.row_start, window.row_stop),
            slice(window.col_start, window.col_stop),
        )
        return unpack_visibility(view)

    def unpack_planes(
        self, window: RasterWindow
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Float32 (h, w, 153) cubes at ``window`` — values identical to
        the replay's ``vegshmat``/``vbshvegshmat`` (solver.py stores the
        marched 0.0/1.0 floats; the packed bits round-trip exactly)."""
        planes = []
        for stack in _STACKS:
            bits = self.bit_window(stack, window)
            planes.append(torch.from_numpy(bits.astype(np.float32)))
        return planes[0], planes[1]

    def validate_against(self, cache: Any, scene: FullSceneTensors) -> None:
        """Raise :class:`StaleSvfError` unless this state describes exactly
        this cache and this composed scene (solver-seam invariant)."""
        if not _key_matches_cache(self.key, cache):
            raise StaleSvfError(
                "vegetation occluder state key does not match the cache "
                f"({self.key!r}); rebuild or fall back to the full replay"
            )
        if self.rows != cache.rows or self.cols != cache.cols:
            raise StaleSvfError(
                f"occluder state grid {self.rows}x{self.cols} does not "
                f"match the cache grid {cache.rows}x{cache.cols}"
            )
        canopy = scene.canopy.numpy()
        if not np.array_equal(self.canopy_scene, canopy):
            raise StaleSvfError(
                "vegetation occluder state describes a different canopy "
                "than the composed scene; apply the edit batch (or fall "
                "back to the full replay)"
            )


def _chunk_checksums_of(
    vegsh: PackedVisibility, vbsh: PackedVisibility
) -> dict[str, list[str]]:
    rows, cols = vegsh.spatial_shape
    checksums: dict[str, list[str]] = {}
    for stack, packed in (("vegsh", vegsh), ("vbsh", vbsh)):
        digests = []
        for row_slice, col_slice in _chunk_slices(rows, cols):
            block = packed.data[row_slice, col_slice, :]
            digests.append(
                hashlib.sha256(np.ascontiguousarray(block)).hexdigest()
            )
        checksums[stack] = digests
    return checksums


def _fence_binary_planes(
    vegsh: torch.Tensor, vbsh: torch.Tensor, *, context: str
) -> None:
    """Condition 4: float planes must be within {0, 1} (exact membership)."""
    for name, plane in (("vegsh", vegsh), ("vbsh", vbsh)):
        values = plane.numpy()
        bad = ~((values == 0.0) | (values == 1.0))
        if bad.any():
            count = int(bad.sum())
            raise VegOcclusionFallback(
                f"non-binary {name} plane after {context}: {count} cells "
                f"outside {{0, 1}} (max {float(values.max())!r}) — the march "
                "produced a vbsh==2.0-class result (one-step regime); full "
                "replay required for this batch"
            )


def build_baseline_state(cache: Any) -> VegOcclusionState:
    """Pack the baseline occlusion bits from the site's SVF patch cubes.

    The P2 cache cubes were marched with the oracle's absolute
    ``amaxvalue``; V0 (proof ADDENDUM item 1) pinned those bits identical
    to a fresh effective-amplitude full-tile replay in BOTH regimes, so
    the packed baseline is the replay's baseline. Condition 4 fences the
    cubes before packing (pack_visibility refuses non-binary loudly as
    defense in depth).
    """
    started = time.perf_counter()
    patches = cache.svf_patches
    rows, cols = int(cache.rows), int(cache.cols)
    packed_stacks = []
    for stack in _STACKS:
        dense = np.asarray(patches[_DENSE_STACK_NAMES[stack]])
        if dense.shape != (rows, cols, len(_sky_patch_geometry(_PATCH_OPTION)[0])):
            raise VegOcclusionFallback(
                f"{_DENSE_STACK_NAMES[stack]} cube shape {dense.shape} does "
                f"not match the site grid {rows}x{cols} with 153 patches"
            )
        bad = ~((dense == 0.0) | (dense == 1.0))
        if bad.any():
            raise VegOcclusionFallback(
                f"non-binary {stack} baseline cubes: {int(bad.sum())} cells "
                f"outside {{0, 1}} (max {float(dense.max())!r}) — this site's "
                "baseline march hits the vbsh==2.0 one-step regime; the "
                "packed-state path is refused, full replay stays the path"
            )
        try:
            packed_stacks.append(pack_visibility(dense))
        except ValueError as error:
            raise VegOcclusionFallback(
                f"packing refused for {stack} baseline cubes: {error}"
            ) from error
    vegsh_packed, vbsh_packed = packed_stacks

    # Condition 4, regime level (review HIGH-1): refuse the pack when the
    # baseline amplitude already leaves ANY patch in the one-step regime —
    # even all-binary cubes are only latently safe there, and a later batch
    # (or an edit far from every crown) can surface vbsh==2.0 at unchanged
    # cells no corridor window covers. Runs after the value fence so a
    # genuinely corrupt cube reports its values, not the regime.
    baseline_scene = _compose_scene_from_canopy(
        cache,
        np.asarray(cache.tree_base, dtype=np.float32),
    )
    baseline_amp = _scene_amplitude(baseline_scene)
    one_step = one_step_patch_indices(1.0 / float(cache.pixel_size_m), baseline_amp)
    if one_step:
        raise VegOcclusionFallback(
            f"baseline march is one-step at {len(one_step)} patch(es) "
            f"(first-step dz above the baseline effective amplitude "
            f"{baseline_amp:.3f} m; e.g. patch {one_step[0]}): this site's "
            "regime can produce oracle vbsh==2.0 the packed bits cannot "
            "represent — the packed-state path is refused, full replay "
            "stays the path"
        )

    canopy = np.asarray(cache.tree_base, dtype=np.float32).copy()
    canopy[canopy < 0.0] = 0.0

    chunk_rows, chunk_cols = _chunk_grid_index(rows, cols)
    return VegOcclusionState(
        key=state_key_for(cache, scene_revision=0),
        vegsh_packed=vegsh_packed,
        vbsh_packed=vbsh_packed,
        canopy_scene=canopy,
        chunk_versions=np.ones((chunk_rows, chunk_cols), dtype=np.uint32),
        chunk_checksums=_chunk_checksums_of(vegsh_packed, vbsh_packed),
        rows=rows,
        cols=cols,
        pack_seconds=time.perf_counter() - started,
    )


def apply_edit_batch(
    state: VegOcclusionState, cache: Any, scene_post: FullSceneTensors
) -> tuple[VegOcclusionState, dict[str, Any]]:
    """Advance the packed bits across one vegetation edit batch.

    ``scene_post`` is the composed scene AFTER the batch; the pre scene is
    recomposed from the state's canopy snapshot (any pre/post pair works,
    so refused or lagged batches catch up on a later call). Guards and
    fences per the four binding conditions; every refusal raises
    :class:`VegOcclusionFallback` WITHOUT advancing the state.
    """
    started = time.perf_counter()
    telemetry: dict[str, Any] = {
        "baseline_packed": False,
        "patches_remarched": 0,
        "vegsh_flips": 0,
        "vbsh_flips": 0,
        "corridor_seconds": 0.0,
        "pack_seconds": 0.0,
        "fallback_reason": None,
    }

    scene_pre = _compose_scene_from_canopy(cache, state.canopy_scene)

    # Condition 2 precondition: bush must vanish on pre AND post scenes.
    for label, scene in (("pre", scene_pre), ("post", scene_post)):
        if bool(scene.bush.any()):
            raise VegOcclusionFallback(
                f"bush cells present on the {label} scene "
                f"({int((scene.bush != 0).sum())} cells); the corridor "
                "closure does not model bush occluders — full replay "
                "required for this batch"
            )

    # Condition 1: clamp-regime amplitude guard (Case C).
    reason = clamped_amplitude_change_reason(scene_pre, scene_post)
    if reason is not None:
        raise VegOcclusionFallback(reason)

    # Condition 4, regime level (review HIGH-1): a multi-step -> one-step
    # amplitude transition (unclamped Case B — the clamp guard above is
    # correctly silent) can surface oracle vbsh==2.0 at UNCHANGED cells
    # outside every corridor closure; the per-window fence below cannot
    # see them. Refuse the batch before any marching.
    amp_pre = _scene_amplitude(scene_pre)
    amp_post = _scene_amplitude(scene_post)
    scale = 1.0 / float(cache.pixel_size_m)
    reason = _regime_transition_reason(amp_pre, amp_post, scale)
    if reason is not None:
        raise VegOcclusionFallback(reason)

    canopy_post = scene_post.canopy.numpy()
    if np.array_equal(state.canopy_scene, canopy_post):
        return state, telemetry

    # Condition 2: C on composed surface diffs, closure = corridor U C.
    C = (scene_pre.vegdsm.numpy() != scene_post.vegdsm.numpy()) | (
        scene_pre.vegdsm2.numpy() != scene_post.vegdsm2.numpy()
    )

    march_amplitude = max(amp_pre, amp_post)
    rows, cols = state.rows, state.cols
    full = RasterWindow(0, rows, 0, cols)

    vegsh_copy = state.vegsh_packed.data.copy()
    vbsh_copy = state.vbsh_packed.data.copy()
    vegsh_new = PackedVisibility(
        data=vegsh_copy, patch_count=state.vegsh_packed.patch_count
    )
    vbsh_new = PackedVisibility(
        data=vbsh_copy, patch_count=state.vbsh_packed.patch_count
    )

    patches, _rings = _sky_patch_geometry(_PATCH_OPTION)
    # The reference the packed bits must reproduce is the FULL-TILE replay
    # (``_recompute_veg_svf_window`` at ``window = full tile``), whose march
    # amplitude is the scene-wide effective amplitude of the POST scene.
    # Marching with a window-LOCAL amplitude is bit-equal at closure cells
    # but VALUE-divergent in one-step regimes (vbsh 2.0 vs 1.0) — the fence
    # must see the replay's exact values, so the amplitude is the scene's,
    # while the reach expansion stays the union bound max(eff_pre, eff_post).
    eff_post_scene = effective_march_amplitude(
        scene_post.a,
        scene_post.vegdsm,
        scene_post.vegdsm2,
        scene_amaxvalue=scene_post.amaxvalue,
    )
    touched = np.zeros((rows, cols), dtype=bool)
    remarched = 0
    vegsh_flips = 0
    vbsh_flips = 0
    # T19b: lane counters are PROCESS-level, so the per-batch attribution is
    # a before/after DELTA — a concurrent solve's marches are never counted
    # here (same discipline as the solver's stage deltas).
    _lanes_before = march_router.lane_stats()
    for index in range(len(patches)):
        altitude, azimuth = patches[index][0], patches[index][1]
        # shadow()'s convention: sizex = a.shape[0] (rows), sizey = cols —
        # transposing them loses corridor steps on non-square tiles
        # (review MEDIUM-1)
        offsets = march_offsets(
            azimuth, altitude, march_amplitude, scale, rows, cols
        )
        corridor = corridor_mask(C, offsets)
        closure = corridor
        closure |= C
        if not closure.any():
            continue  # untouched patch: cost nothing
        remarched += 1
        rs, cs = np.nonzero(closure)
        bbox = RasterWindow(
            int(rs.min()), int(rs.max()) + 1, int(cs.min()), int(cs.max()) + 1
        )
        # Condition 3: W2 reach-expanded window, never the raw bbox.
        reach = _patch_march_reach_pixels(march_amplitude, scale, float(altitude))
        march_window = _patch_march_window(
            bbox, full, azimuth_deg=float(azimuth), reach_pixels=reach
        )
        r_slice = slice(
            march_window.row_start, march_window.row_stop
        )
        c_slice = slice(
            march_window.col_start, march_window.col_stop
        )
        a_w = scene_post.a[r_slice, c_slice]
        vegdsm_w = scene_post.vegdsm[r_slice, c_slice]
        vegdsm2_w = scene_post.vegdsm2[r_slice, c_slice]
        bush_w = scene_post.bush[r_slice, c_slice]
        # T19b: the corridor re-march runs through the registered lane
        # router (numba march on this window's own shape/amplitude, torch
        # byte-identically on refusal). The torch callable is handed in
        # from THIS module's binding so spies on ``shadow_fn`` here keep
        # witnessing the fallback lane.
        _sh, vegsh_w, vbsh_w = march_router.shadow_march(
            eff_post_scene,
            a_w,
            vegdsm_w,
            vegdsm2_w,
            bush_w,
            azimuth,
            altitude,
            scale,
            amplitude_policy="effective_windowed",
            torch_shadow=shadow_fn,
        )
        # Condition 4: fence every re-marched plane.
        _fence_binary_planes(
            vegsh_w, vbsh_w, context=f"corridor re-march of patch {index}"
        )

        inside_r = slice(
            bbox.row_start - march_window.row_start,
            bbox.row_stop - march_window.row_start,
        )
        inside_c = slice(
            bbox.col_start - march_window.col_start,
            bbox.col_stop - march_window.col_start,
        )
        veg_new = vegsh_w[inside_r, inside_c].numpy() != 0
        vb_new = vbsh_w[inside_r, inside_c].numpy() != 0

        row_slice = slice(bbox.row_start, bbox.row_stop)
        col_slice = slice(bbox.col_start, bbox.col_stop)
        veg_old = unpack_patch(state.vegsh_packed, index)[row_slice, col_slice]
        vb_old = unpack_patch(state.vbsh_packed, index)[row_slice, col_slice]
        keep = ~closure[row_slice, col_slice]
        veg_values = np.where(keep, veg_old, veg_new)
        vb_values = np.where(keep, vb_old, vb_new)
        vegsh_flips += int((veg_values != veg_old).sum())
        vbsh_flips += int((vb_values != vb_old).sum())
        set_patch_window(
            vegsh_new,
            patch_index=index,
            row_slice=row_slice,
            col_slice=col_slice,
            values=veg_values,
        )
        set_patch_window(
            vbsh_new,
            patch_index=index,
            row_slice=row_slice,
            col_slice=col_slice,
            values=vb_values,
        )
        touched |= closure

    versions = state.chunk_versions.copy()
    for row_slice, col_slice in _chunks_overlapping(touched | C, rows, cols):
        r = row_slice.start // _CHUNK_CELLS
        c = col_slice.start // _CHUNK_CELLS
        versions[r, c] = np.uint32(int(versions[r, c]) + 1)

    new_state = VegOcclusionState(
        key=replace(state.key, scene_revision=state.key.scene_revision + 1),
        vegsh_packed=vegsh_new,
        vbsh_packed=vbsh_new,
        canopy_scene=canopy_post.copy(),
        chunk_versions=versions,
        chunk_checksums=_chunk_checksums_of(vegsh_new, vbsh_new),
        rows=rows,
        cols=cols,
    )
    _lanes_after = march_router.lane_stats()
    _routed = int(_lanes_after["numba_routed"] - _lanes_before["numba_routed"])
    _refused = int(_lanes_after["numba_refused"] - _lanes_before["numba_refused"])
    telemetry.update(
        patches_remarched=remarched,
        vegsh_flips=vegsh_flips,
        vbsh_flips=vbsh_flips,
        corridor_seconds=time.perf_counter() - started,
        pack_seconds=state.pack_seconds,
        # T19b routing attribution for this batch (see the delta note above);
        # the refusal reason is the process-level last one, only meaningful
        # when this batch's own delta says a refusal happened.
        march_routed=_routed,
        march_refused=_refused,
        march_refusal=str(_lanes_after["last_refusal"]) if _refused else None,
    )
    return new_state, telemetry


# ---------------------------------------------------------------------------
# Store: load-or-build, catch-up apply, commit at publication
# ---------------------------------------------------------------------------


class VegOcclusionStore:
    """Per-scenario lifecycle for the occlusion state.

    ``prepare(scene)`` returns a state validated for the LIVE scene (or
    ``None`` plus a telemetry reason when a binding condition refuses —
    the batch then routes today's full replay). ``commit(state)`` is the
    exact-publication seam: it advances the in-memory current state and,
    when a ``state_root`` is configured, persists the revision atomically
    (staged directory + ``os.rename``), so a kill mid-persist leaves the
    previous revision intact. The publication layer (post-r2a wiring)
    calls ``commit`` exactly when the scenario's revision goes live.
    """

    def __init__(self, cache: Any, *, state_root: str | Path | None = None) -> None:
        self._cache = cache
        self._root = Path(state_root) if state_root is not None else None
        self._current: VegOcclusionState | None = None
        self._loaded = False

    # -- public API -------------------------------------------------------

    def prepare(self, scene: FullSceneTensors) -> tuple[VegOcclusionState | None, dict]:
        telemetry: dict[str, Any] = {
            "baseline_packed": False,
            "patches_remarched": 0,
            "vegsh_flips": 0,
            "vbsh_flips": 0,
            "corridor_seconds": 0.0,
            "pack_seconds": 0.0,
            "fallback_reason": None,
        }
        try:
            if not self._loaded:
                self._current = self._load_latest()
                self._loaded = True
            if self._current is not None and not _key_matches_cache(
                self._current.key, self._cache
            ):
                # New cache manifest (e.g. building edit regenerated it):
                # the state is void, rebuild from the new baseline.
                self._current = None
            if self._current is None:
                self._current = build_baseline_state(self._cache)
                telemetry["baseline_packed"] = True
                telemetry["pack_seconds"] = float(
                    getattr(self._current, "pack_seconds", 0.0)
                )
            state = self._current
            if np.array_equal(state.canopy_scene, scene.canopy.numpy()):
                return state, telemetry
            packed_baseline = bool(telemetry["baseline_packed"])
            state, applied = apply_edit_batch(state, self._cache, scene)
            telemetry.update(applied)
            # a baseline pack earlier in THIS prepare stays reported even
            # when a catch-up batch follows it
            telemetry["baseline_packed"] = packed_baseline
            return state, telemetry
        except VegOcclusionFallback as refusal:
            telemetry["fallback_reason"] = refusal.reason
            return None, telemetry

    def commit(self, state: VegOcclusionState) -> None:
        """Publish ``state`` as the current revision (atomic persist)."""
        self._current = state
        if self._root is not None:
            self._persist(state)

    # -- persistence ------------------------------------------------------

    def _directory(self) -> Path:
        key = state_key_for(self._cache, scene_revision=0)
        return (
            self._root
            / f"{key.site_id}_{key.tile_key}"
            / key.cache_manifest_sha256
            / f"schema_v{SCHEMA_VERSION}"
        )

    def _persist(self, state: VegOcclusionState) -> None:
        final = (
            self._directory() / f"rev-{state.key.scene_revision:06d}"
        )
        if final.exists():
            shutil.rmtree(final)
        staging = final.with_name(final.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        np.save(staging / "vegsh_packed.npy", state.vegsh_packed.data)
        np.save(staging / "vbsh_packed.npy", state.vbsh_packed.data)
        np.save(staging / "canopy_scene.npy", state.canopy_scene)
        np.save(staging / "chunk_versions.npy", state.chunk_versions)
        (staging / "state.json").write_text(
            json.dumps(
                {
                    "key": {
                        "site_id": state.key.site_id,
                        "tile_key": state.key.tile_key,
                        "cache_manifest_sha256": state.key.cache_manifest_sha256,
                        "scene_revision": state.key.scene_revision,
                        "patch_geometry_id": state.key.patch_geometry_id,
                        "schema_version": state.key.schema_version,
                    },
                    "rows": state.rows,
                    "cols": state.cols,
                    "patch_count": state.patch_count,
                    "chunk_checksums": state.chunk_checksums,
                },
                sort_keys=True,
            )
        )
        os.rename(staging, final)
        self._prune(final.parent)

    def _prune(self, parent: Path, keep: int = 2) -> None:
        revisions = sorted(
            (p for p in parent.glob("rev-*") if p.is_dir()),
            key=lambda p: p.name,
        )
        for stale in revisions[:-keep]:
            shutil.rmtree(stale, ignore_errors=True)

    def _load_latest(self) -> VegOcclusionState | None:
        if self._root is None:
            return None
        base = self._directory()
        if not base.is_dir():
            return None
        revisions = sorted(
            (p for p in base.glob("rev-*") if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        for revision_dir in revisions:
            try:
                state = self._load_revision(revision_dir)
            except VegOcclusionStateError:
                continue  # corrupted/stale revision: never serve wrong bits
            if _key_matches_cache(state.key, self._cache):
                return state
            return None  # a key mismatch voids the lineage: rebuild
        return None

    def _load_revision(self, revision_dir: Path) -> VegOcclusionState:
        try:
            meta = json.loads((revision_dir / "state.json").read_text())
            vegsh_data = np.load(revision_dir / "vegsh_packed.npy")
            vbsh_data = np.load(revision_dir / "vbsh_packed.npy")
            canopy = np.load(revision_dir / "canopy_scene.npy")
            versions = np.load(revision_dir / "chunk_versions.npy")
        except (OSError, ValueError) as error:
            raise VegOcclusionStateError(
                f"unreadable occluder state revision {revision_dir}: {error}"
            ) from error
        key_fields = meta.get("key", {})
        key = VegOcclusionKey(
            site_id=str(key_fields["site_id"]),
            tile_key=str(key_fields["tile_key"]),
            cache_manifest_sha256=str(key_fields["cache_manifest_sha256"]),
            scene_revision=int(key_fields["scene_revision"]),
            patch_geometry_id=str(key_fields["patch_geometry_id"]),
            schema_version=int(key_fields.get("schema_version", -1)),
        )
        if key.schema_version != SCHEMA_VERSION:
            raise VegOcclusionStateError(
                f"state schema {key.schema_version} != {SCHEMA_VERSION}"
            )
        vegsh = PackedVisibility(data=vegsh_data, patch_count=int(meta["patch_count"]))
        vbsh = PackedVisibility(data=vbsh_data, patch_count=int(meta["patch_count"]))
        expected: dict[str, list[str]] = {
            k: list(v) for k, v in meta.get("chunk_checksums", {}).items()
        }
        actual = _chunk_checksums_of(vegsh, vbsh)
        if expected != actual:
            raise VegOcclusionStateError(
                f"occluder state checksum mismatch in {revision_dir}"
            )
        return VegOcclusionState(
            key=key,
            vegsh_packed=vegsh,
            vbsh_packed=vbsh,
            canopy_scene=np.asarray(canopy, dtype=np.float32),
            chunk_versions=versions,
            chunk_checksums=actual,
            rows=int(meta["rows"]),
            cols=int(meta["cols"]),
        )
