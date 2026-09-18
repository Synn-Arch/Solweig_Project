# SPDX-License-Identifier: GPL-3.0-only
"""Sparse work builder for the incremental exact march (DESIGN.ko.md
7.3/7.4/7.5/7.7, TASKS T05).

Given the pre/post composed march scenes and the executed T03 step tables
(BOTH scenes' — the amplitude can change across an edit), this module
computes, per sky patch, the corridor-closure work set

    D_p = C  U  {t : t + d in C for some d in (D_pre(p) U D_post(p))}

where ``C`` is the composed-source diff (DESIGN 7.3: cells where the march
INPUTS ``a``/``vegdsm``/``vegdsm2``/``bush`` actually differ after exact
recomposition) and ``d`` ranges over the EXECUTED step offsets of the patch
in the pre AND the post scene. ``C`` itself is always included — the
target-local first-step trunk gate (``vegdem2[t] > a[t]``) makes an edited
cell's own output changeable with no corridor read at all.

Guards (DESIGN 7.4): the R4 refusal conditions are preserved as pure-numpy
predicates and re-checked here so the builder never emits a work list whose
splice would diverge from the oracle:

* nonzero bush on either scene (pre or post);
* clamp-regime amplitude change (Case C): amplitude changed AND the
  smaller-amplitude scene is clamped (``bound > scene_amaxvalue``, STRICT
  ``>``, the amplitude-change conjunct kept exactly);
* one-step regime: any patch with an executed count <= 1 on EITHER scene
  (the packed-bit state cannot represent ``vbsh == 2.0``; this subsumes R4's
  multi->one-step narrowing refusal AND the baseline one-step fence);
* state/cache/profile identity mismatch or unknown dependency fields.

Torch-free by construction (solweig_core contract): the compose mirror
reproduces ``_compose_scene_from_canopy`` bit-for-bit with numpy float32
ops, and the amplitude mirror reproduces ``effective_march_amplitude``
bit-for-bit (both pinned by regression tests against the originals).

Implementation choice (pure numpy + stdlib, no numba): this is a work-list
BUILDER, not a kernel — the corridor is one vectorized shifted-OR slice per
DISTINCT offset and the rest is boolean reductions, so JIT compile latency
(~0.2 s cold) would exceed the build itself on every realistic edit. T06
may revisit if measurement shows the builder hot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from solweig_core import step_tables as st

#: Work-list schema version (bump on any layout or semantics change).
SPARSE_WORK_SCHEMA_VERSION = 1

#: Chunked-bitset chunk edge in cells (64x64 = 4096 cells per chunk).
CHUNK_CELLS = 64
#: Bytes per chunk: 64 rows x uint64 per row.
CHUNK_BYTES = CHUNK_CELLS * 8
#: Bytes per row-run record (row, c0, c1_inclusive as int32).
ROW_RUN_BYTES_PER_RUN = 12
#: Route a patch to the dense kernel when its closure covers more than this
#: fraction of the tile. Chosen as "half the tile": a sparse iteration over
#: >50% of cells pays indirection per cell and loses to the dense sweep.
#: Fixed a priori from the representation-cost model (row runs cover a row
#: prefix work only while spans are short), NOT tuned against site data.
ROUTE_DENSE_PAIR_FRACTION = 0.5

#: Descriptor fields that fully identify the packed-state provenance a work
#: list may be spliced into. Unknown or missing fields refuse the build.
IDENTITY_FIELDS = (
    "schema_version",
    "site_id",
    "tile_key",
    "cache_manifest_sha256",
    "patch_geometry_id",
    "rows",
    "cols",
    "chunk_checksums_sha256",
)


class SparseWorkFallback(RuntimeError):
    """A preserved guard refused this edit batch: full replay/fallback."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Composed march inputs (numpy mirror of _compose_scene_from_canopy)
# ---------------------------------------------------------------------------


@dataclass
class MarchInputs:
    """The exact float32 planes the march kernels consume, plus provenance.

    Same member names/order as ``FullSceneTensors`` so guards read the same
    fields the torch-expressed originals do. ``vegdsm``/``vegdsm2`` are the
    vegetation planes the SVF march consumes (passed as the kernels'
    ``vegdem``/``vegdem2`` arguments by the original call site).
    """

    a: np.ndarray
    canopy: np.ndarray
    dem: np.ndarray
    vegdem: np.ndarray
    vegdem2: np.ndarray
    bush: np.ndarray
    vegdsm: np.ndarray
    vegdsm2: np.ndarray
    amaxvalue: np.float32


def _as_scene_plane(array: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"{name} must be numpy.ndarray, got {type(array).__name__}"
        )
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must be float32, got {array.dtype.str}")
    return array


def compose_march_inputs(
    a: np.ndarray, dem: np.ndarray, canopy: np.ndarray
) -> MarchInputs:
    """Bit-exact numpy mirror of ``_compose_scene_from_canopy``.

    Every expression is the torch one with scalar operands replaced by
    ``np.float32`` (exact same IEEE ops; ``0.25`` is exactly representable,
    ``temp1 < 0`` and ``vegdsm == a`` are value comparisons that keep -0.0
    and propagate NaN exactly like the torch versions).
    """
    a = _as_scene_plane(a, "a")
    dem = _as_scene_plane(dem, "dem")
    canopy = _as_scene_plane(canopy, "canopy")
    if not (a.shape == dem.shape == canopy.shape):
        raise ValueError(
            f"scene shape mismatch: a {a.shape}, dem {dem.shape}, "
            f"canopy {canopy.shape}"
        )
    temp1 = np.ascontiguousarray(canopy).copy()
    temp1[temp1 < np.float32(0.0)] = np.float32(0.0)
    vegdem = temp1 + dem
    vegdem2 = np.add(temp1 * np.float32(0.25), dem)
    # torch.logical_not on a float tensor is (value == 0); bool*float is the
    # same f32 multiply in numpy.
    bush = np.logical_not(vegdem2 * vegdem).astype(np.float32) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = np.float32(0.0)
    vegdsm2 = temp1 * np.float32(0.25) + a
    vegdsm2[vegdsm2 == a] = np.float32(0.0)
    amaxvalue = np.maximum(np.float32(a.max()), np.float32(vegdem.max()))
    return MarchInputs(
        a=np.ascontiguousarray(a).copy(),
        canopy=temp1,
        dem=np.ascontiguousarray(dem).copy(),
        vegdem=vegdem,
        vegdem2=vegdem2,
        bush=bush,
        vegdsm=vegdsm,
        vegdsm2=vegdsm2,
        amaxvalue=np.float32(amaxvalue),
    )


def scene_relative_bound(inputs: MarchInputs) -> np.float32:
    """``max(a, vegdsm, vegdsm2) - min(a)`` (the clamp bound, f32 bits)."""
    top = np.maximum(
        np.maximum(
            np.float32(inputs.a.max()),
            np.float32(inputs.vegdsm.max()),
        ),
        np.float32(inputs.vegdsm2.max()),
    )
    return np.float32(top - np.float32(inputs.a.min()))


def effective_amplitude(inputs: MarchInputs) -> np.float32:
    """Bit-exact numpy mirror of ``effective_march_amplitude``.

    ``clamp(bound, 0, scene_amaxvalue)`` in f32 scalar ops, identical
    operand order to the torch expression.
    """
    bound = scene_relative_bound(inputs)
    bound = np.maximum(bound, np.float32(0.0))
    return np.minimum(bound, np.float32(inputs.amaxvalue))


# ---------------------------------------------------------------------------
# C: composed-surface source diff (DESIGN 7.3)
# ---------------------------------------------------------------------------


def composed_change_mask(
    pre: MarchInputs,
    post: MarchInputs,
    *,
    restrict: np.ndarray | None = None,
    verify_restrict: bool = False,
) -> np.ndarray:
    """Cells where the composed MARCH INPUTS differ between the scenes.

    Equality policy (stated once, applied everywhere): IEEE float32 VALUE
    equality. ``+0.0 == -0.0`` (safe: the march consumers only use ``+``,
    ``*``, ``max`` and ``>`` where the zero sign is interchangeable), and
    any NaN operand counts as CHANGED (``x != x``), a conservative superset
    — byte equality would flag the sign of zero; value equality never
    under-reports.

    Only the four planes the march reads (``a``, ``vegdsm``, ``vegdsm2``,
    ``bush``) participate: ``vegdem`` differences with identical march
    inputs cannot change any output bit. Byte-diffing the whole composed
    scene is allowed as a conservative superset but not done, so the
    reported closure is the exact one.

    ``restrict`` narrows the diff to the composed C_candidate (union of old
    and new object footprints); with ``verify_restrict`` the FULL diff is
    also computed and a candidate that misses real changes raises.
    """
    changed = (
        (pre.a != post.a)
        | (pre.vegdsm != post.vegdsm)
        | (pre.vegdsm2 != post.vegdsm2)
        | (pre.bush != post.bush)
    )
    if restrict is not None:
        if restrict.shape != changed.shape or restrict.dtype != np.dtype(bool):
            raise ValueError(
                f"restrict must be bool of shape {changed.shape}, got "
                f"{restrict.dtype} {restrict.shape}"
            )
        if verify_restrict and bool((changed & ~restrict).any()):
            missed = int((changed & ~restrict).sum())
            raise ValueError(
                f"C_candidate restrict mask missed {missed} changed cells — "
                "the candidate footprint union is not a superset of the "
                "composed diff"
            )
        changed = changed & restrict
    return changed


# ---------------------------------------------------------------------------
# Corridor + closure (DESIGN 7.3)
# ---------------------------------------------------------------------------


def _normalize_offsets(offsets: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    return sorted({(int(dx), int(dy)) for dx, dy in offsets})


def corridor_mask(C: np.ndarray, offsets) -> np.ndarray:
    """``{t : t + (dx, dy) in C}`` for the given step offsets.

    Same clipped-shift semantics as R4's ``corridor_mask`` (targets read
    sources at ``t + (dx, dy)`` per step; out-of-bounds reads drop — never
    wrap). Offsets are deduplicated and sorted first (determinism).
    """
    C = np.asarray(C, dtype=bool)
    rows, cols = C.shape
    out = np.zeros((rows, cols), dtype=bool)
    for dx, dy in _normalize_offsets(offsets):
        r0, r1 = max(0, -dx), rows - max(0, dx)
        c0, c1 = max(0, -dy), cols - max(0, dy)
        if r0 >= r1 or c0 >= c1:
            continue
        out[r0:r1, c0:c1] |= C[r0 + dx : r1 + dx, c0 + dy : c1 + dy]
    return out


def closure_mask(
    C: np.ndarray, offsets_pre, offsets_post
) -> np.ndarray:
    """``D_p = C U corridor(C, pre offsets) U corridor(C, post offsets)``.

    Both scenes' EXECUTED offsets are unioned: an amplitude change across
    the edit makes the two step sets differ, and a target whose output
    changes may read C through EITHER march's steps.
    """
    out = np.asarray(C, dtype=bool).copy()
    out |= corridor_mask(C, offsets_pre)
    out |= corridor_mask(C, offsets_post)
    return out


def table_offsets(table: st.StepTable) -> list[tuple[int, int]]:
    """The executed step offsets of a table (deduped, sorted)."""
    return _normalize_offsets(zip(table.dx.tolist(), table.dy.tolist()))


# ---------------------------------------------------------------------------
# Preserved guards (DESIGN 7.4; both directions pinned by regression tests)
# ---------------------------------------------------------------------------


def bush_guard(pre: MarchInputs, post: MarchInputs) -> str | None:
    """Nonzero bush on either scene refuses (T04 domain is bush == 0)."""
    if bool((pre.bush != 0).any()) or bool((post.bush != 0).any()):
        return (
            "nonzero bush plane (pre or post): the bush blocks use "
            "marched-extent reduction predicates that break per-target "
            "independence — full replay required for this batch"
        )
    return None


def clamped_amplitude_change_guard(
    pre: MarchInputs, post: MarchInputs
) -> str | None:
    """Case C refusal: amplitude changed AND the SMALLER scene is clamped.

    Conjuncts kept exactly as in ``clamped_amplitude_change_reason``:
    amplitude-constant -> safe (None); amplitude changed -> look at the
    smaller-amplitude scene only; refuse iff ``bound > scene_amaxvalue``
    with STRICT ``>`` (``bound == abs`` stays sparse).
    """
    amp_pre = float(effective_amplitude(pre))
    amp_post = float(effective_amplitude(post))
    if amp_pre == amp_post:
        return None
    smaller = pre if amp_pre < amp_post else post
    smaller_amp = min(amp_pre, amp_post)
    bound = float(scene_relative_bound(smaller))
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


def one_step_regime_guard(
    tables_pre: Mapping[int, st.StepTable],
    tables_post: Mapping[int, st.StepTable],
) -> str | None:
    """Refuse any one-step (count <= 1) patch on EITHER scene.

    Superset of R4's two conditions, stated as one representability fence:
    the packed binary state cannot hold a ``vbsh == 2.0`` cell. It subsumes

    * multi-step -> one-step NARROWING (R4 condition 4: unchanged cells
      outside every closure can carry oracle vbsh == 2.0), and
    * a one-step patch in the BASELINE state (the R4 fence: such cells are
      already mis-representable, growth direction included).
    """
    for label, tables in (("pre", tables_pre), ("post", tables_post)):
        for idx in sorted(tables):
            if int(tables[idx].count) <= 1:
                return (
                    f"one-step march regime: patch {idx} executed "
                    f"{int(tables[idx].count)} step(s) on the {label} scene "
                    "(vbsh can reach 2.0, which the packed-bit state cannot "
                    "represent) — full replay required for this batch"
                )
    return None


def state_identity_guard(
    identity_pre: Mapping[str, Any] | None,
    identity_post: Mapping[str, Any] | None,
) -> str | None:
    """Refuse on any state/cache/profile identity mismatch.

    Descriptors are plain dicts over exactly :data:`IDENTITY_FIELDS`; a
    missing field, an unknown extra field (a dependency this builder was
    not reviewed against) or any value mismatch refuses. ``None``
    descriptors skip the check (caller-asserted provenance).
    """
    if identity_pre is None and identity_post is None:
        return None
    fields = set(IDENTITY_FIELDS)
    for label, descriptor in (("pre", identity_pre), ("post", identity_post)):
        if descriptor is None:
            continue
        keys = set(descriptor)
        if keys != fields:
            missing = sorted(fields - keys)
            unknown = sorted(keys - fields)
            return (
                f"state identity descriptor ({label}) field mismatch: "
                f"missing {missing}, unknown {unknown} — unknown dependency, "
                "full replay required for this batch"
            )
        if int(descriptor["schema_version"]) != SPARSE_WORK_SCHEMA_VERSION:
            return (
                f"state identity descriptor ({label}) schema_version "
                f"{descriptor['schema_version']!r} != "
                f"{SPARSE_WORK_SCHEMA_VERSION} — unknown schema, full replay "
                "required for this batch"
            )
    if identity_pre is not None and identity_post is not None:
        for name in IDENTITY_FIELDS:
            if identity_pre[name] != identity_post[name]:
                return (
                    f"state identity mismatch on {name!r} — scene/cache/"
                    "profile provenance differs, full replay required for "
                    "this batch"
                )
    return None


# ---------------------------------------------------------------------------
# Sparse representations (DESIGN 7.5)
# ---------------------------------------------------------------------------


def row_runs_from_mask(mask: np.ndarray) -> np.ndarray:
    """Maximal sorted row runs ``(row, c0, c1_inclusive)`` as (n, 3) int32.

    Deterministic row-major order; runs within a row are maximal (they never
    touch or overlap) because each run is one contiguous True span.
    """
    mask = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int, int]] = []
    for row in range(mask.shape[0]):
        line = mask[row]
        if not line.any():
            continue
        padded = np.concatenate(([False], line, [False]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        runs.extend(
            (row, int(s), int(e - 1)) for s, e in zip(starts, stops)
        )
    if not runs:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(runs, dtype=np.int32)


def mask_from_row_runs(
    runs: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Inverse of :func:`row_runs_from_mask` (validating)."""
    runs = np.asarray(runs, dtype=np.int64).reshape(-1, 3)
    rows, cols = int(shape[0]), int(shape[1])
    out = np.zeros((rows, cols), dtype=bool)
    for row, c0, c1 in runs.tolist():
        if not (0 <= row < rows and 0 <= c0 <= c1 < cols):
            raise ValueError(
                f"row run {(row, c0, c1)} outside shape {(rows, cols)}"
            )
        out[row, c0 : c1 + 1] = True
    return out


@dataclass(frozen=True)
class ChunkedBitset:
    """64x64-chunk bitset of one mask (rows/cols carried for inversion).

    Chunk ``(ci, cj)`` covers rows ``[ci*64, ci*64+64)`` and columns
    ``[cj*64, cj*64+64)``; word ``k`` of the chunk is its row ``k`` with
    column ``cj*64 + b`` in bit ``b`` (little-endian bit order). Partial
    chunks at the grid edge keep out-of-grid bits zero.
    """

    rows: int
    cols: int
    chunk_keys: tuple[tuple[int, int], ...]
    words: np.ndarray  # (n_chunks, CHUNK_CELLS) uint64

    def chunk_count(self) -> int:
        return len(self.chunk_keys)


def chunk_bitset_from_mask(mask: np.ndarray) -> ChunkedBitset:
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    keys: list[tuple[int, int]] = []
    words: list[np.ndarray] = []
    for ci in range(0, rows, CHUNK_CELLS):
        for cj in range(0, cols, CHUNK_CELLS):
            chunk = mask[
                ci : min(ci + CHUNK_CELLS, rows),
                cj : min(cj + CHUNK_CELLS, cols),
            ]
            if not chunk.any():
                continue
            keys.append((ci // CHUNK_CELLS, cj // CHUNK_CELLS))
            packed = np.zeros(CHUNK_CELLS, dtype=np.uint64)
            local = np.zeros((CHUNK_CELLS, CHUNK_CELLS), dtype=bool)
            local[: chunk.shape[0], : chunk.shape[1]] = chunk
            for k in range(CHUNK_CELLS):
                row_bits = np.packbits(local[k], bitorder="little")
                packed[k] = np.uint64(
                    int.from_bytes(row_bits.tobytes()[:8], "little")
                )
            words.append(packed)
    return ChunkedBitset(
        rows=int(rows),
        cols=int(cols),
        chunk_keys=tuple(keys),
        words=(
            np.zeros((0, CHUNK_CELLS), dtype=np.uint64)
            if not words
            else np.vstack(words)
        ),
    )


def mask_from_chunk_bitset(bitset: ChunkedBitset) -> np.ndarray:
    out = np.zeros((bitset.rows, bitset.cols), dtype=bool)
    for n, (ci, cj) in enumerate(bitset.chunk_keys):
        r0 = ci * CHUNK_CELLS
        c0 = cj * CHUNK_CELLS
        for k in range(CHUNK_CELLS):
            word = int(bitset.words[n, k])
            if word == 0:
                continue
            bits = np.frombuffer(
                word.to_bytes(8, "little"), dtype=np.uint8
            )
            line = np.unpackbits(bits, bitorder="little")[:CHUNK_CELLS]
            r = r0 + k
            if r >= bitset.rows:
                break
            width = min(CHUNK_CELLS, bitset.cols - c0)
            out[r, c0 : c0 + width] = line[:width].astype(bool)
    return out


# ---------------------------------------------------------------------------
# The build (DESIGN 7.3 + 7.4 + 7.5)
# ---------------------------------------------------------------------------


@dataclass
class SparseWorkBuild:
    """One built work list: per-patch closures in BOTH representations."""

    rows: int
    cols: int
    C: np.ndarray
    row_runs: dict[int, np.ndarray]
    bitsets: dict[int, ChunkedBitset]
    pair_count: int
    row_run_count: int
    estimated_bytes_row_runs: int
    estimated_bytes_bitsets: int
    dense_equivalent_bytes: int
    pair_steps_sparse: int
    pair_steps_dense: int
    route_dense_patches: list[int]
    empty_work: bool
    table_changed_with_empty_C: bool
    table_digests_pre: dict[int, str]
    table_digests_post: dict[int, str]
    build_seconds: float
    _masks: dict[int, np.ndarray] | None = None

    def masks(self) -> dict[int, np.ndarray]:
        if self._masks is None:
            raise RuntimeError(
                "masks were not kept (build_sparse_work(keep_masks=False))"
            )
        return self._masks

    @property
    def route_dense(self) -> bool:
        return bool(self.route_dense_patches)

    def to_dict(self) -> dict[str, Any]:
        """Deterministic summary (no timings) — byte-stable across rebuilds."""
        return {
            "schema_version": SPARSE_WORK_SCHEMA_VERSION,
            "rows": self.rows,
            "cols": self.cols,
            "pair_count": int(self.pair_count),
            "row_run_count": int(self.row_run_count),
            "empty_work": bool(self.empty_work),
            "table_changed_with_empty_C": bool(self.table_changed_with_empty_C),
            "estimated_bytes_row_runs": int(self.estimated_bytes_row_runs),
            "estimated_bytes_bitsets": int(self.estimated_bytes_bitsets),
            "dense_equivalent_bytes": int(self.dense_equivalent_bytes),
            "pair_steps_sparse": int(self.pair_steps_sparse),
            "pair_steps_dense": int(self.pair_steps_dense),
            "route_dense_patches": list(self.route_dense_patches),
            "patches": {
                int(idx): {
                    "cells": int(
                        mask_from_row_runs(
                            self.row_runs[idx], (self.rows, self.cols)
                        ).sum()
                    ),
                    "row_runs": int(self.row_runs[idx].shape[0]),
                    "chunks": int(self.bitsets[idx].chunk_count()),
                }
                for idx in sorted(self.row_runs)
            },
        }


def _validate_tables(
    tables_pre: Mapping[int, st.StepTable],
    tables_post: Mapping[int, st.StepTable],
    shape: tuple[int, int],
) -> None:
    if set(tables_pre) != set(tables_post):
        raise ValueError(
            f"table keys differ: pre {sorted(tables_pre)} vs post "
            f"{sorted(tables_post)} — one scene is missing a patch march"
        )
    for label, tables in (("pre", tables_pre), ("post", tables_post)):
        for idx in sorted(tables):
            table = tables[idx]
            if table.key.kernel_variant != st.KERNEL_SVF_SHADOW:
                raise ValueError(
                    f"table {label}[{idx}] kernel_variant "
                    f"{table.key.kernel_variant!r} != 'svf_shadow' (the "
                    "sparse closure is only proven for the SVF march)"
                )
            if (
                table.key.logical_rows,
                table.key.logical_cols,
            ) != tuple(shape):
                raise ValueError(
                    f"table {label}[{idx}] logical shape "
                    f"({table.key.logical_rows}, {table.key.logical_cols}) "
                    f"does not match the scene shape {shape} — the executed "
                    "step set is only valid for that shape"
                )


def build_sparse_work(
    pre: MarchInputs,
    post: MarchInputs,
    tables_pre: Mapping[int, st.StepTable],
    tables_post: Mapping[int, st.StepTable],
    *,
    identity_pre: Mapping[str, Any] | None = None,
    identity_post: Mapping[str, Any] | None = None,
    candidate_mask: np.ndarray | None = None,
    verify_candidate: bool = False,
    keep_masks: bool = False,
) -> SparseWorkBuild:
    """Build the per-patch corridor-closure work list for one edit batch.

    Guard order (identity -> bush -> clamp -> regime), then the composed
    diff, then the closure per patch. The empty-work shortcut fires ONLY
    when the changed-cell raster is empty AND every per-patch executed
    table is content-identical (same stop context); an empty C with
    CHANGED tables yields the honest full-tile work list, flagged
    ``table_changed_with_empty_C``.
    """
    started = time.perf_counter()
    shape = (int(pre.a.shape[0]), int(pre.a.shape[1]))
    if (int(post.a.shape[0]), int(post.a.shape[1])) != shape:
        raise ValueError(
            f"scene shapes differ: pre {shape} vs post "
            f"{(int(post.a.shape[0]), int(post.a.shape[1]))}"
        )
    _validate_tables(tables_pre, tables_post, shape)

    reason = state_identity_guard(identity_pre, identity_post)
    if reason is not None:
        raise SparseWorkFallback(f"state identity: {reason}")
    reason = bush_guard(pre, post)
    if reason is not None:
        raise SparseWorkFallback(reason)
    reason = clamped_amplitude_change_guard(pre, post)
    if reason is not None:
        raise SparseWorkFallback(reason)
    reason = one_step_regime_guard(tables_pre, tables_post)
    if reason is not None:
        raise SparseWorkFallback(reason)

    C = composed_change_mask(
        pre,
        post,
        restrict=candidate_mask,
        verify_restrict=verify_candidate,
    )
    digests_pre = {
        idx: tables_pre[idx].content_digest() for idx in sorted(tables_pre)
    }
    digests_post = {
        idx: tables_post[idx].content_digest() for idx in sorted(tables_post)
    }
    tables_identical = all(
        digests_pre[idx] == digests_post[idx] for idx in digests_pre
    )

    row_runs: dict[int, np.ndarray] = {}
    bitsets: dict[int, ChunkedBitset] = {}
    masks: dict[int, np.ndarray] | None = {} if keep_masks else None
    route_dense_patches: list[int] = []
    pair_count = 0
    row_run_count = 0
    chunk_total = 0
    pair_steps_sparse = 0
    pair_steps_dense = 0
    empty_work = not bool(C.any())
    table_changed_with_empty_C = empty_work and not tables_identical

    if empty_work and tables_identical:
        return SparseWorkBuild(
            rows=shape[0],
            cols=shape[1],
            C=C,
            row_runs={},
            bitsets={},
            pair_count=0,
            row_run_count=0,
            estimated_bytes_row_runs=0,
            estimated_bytes_bitsets=0,
            dense_equivalent_bytes=0,
            pair_steps_sparse=0,
            pair_steps_dense=0,
            route_dense_patches=[],
            empty_work=True,
            table_changed_with_empty_C=False,
            table_digests_pre=digests_pre,
            table_digests_post=digests_post,
            build_seconds=time.perf_counter() - started,
            _masks=masks,
        )

    full_tile = None
    for idx in sorted(tables_pre):
        if empty_work:
            # C is empty but the stop context changed: every cell may read
            # the changed step set — the honest closure is the full tile.
            if full_tile is None:
                full_tile = np.ones(shape, dtype=bool)
            D = full_tile
        else:
            D = closure_mask(
                C, table_offsets(tables_pre[idx]), table_offsets(tables_post[idx])
            )
            if not bool(D.any()):
                continue  # no work for this patch; omit it entirely
        runs = row_runs_from_mask(D)
        row_runs[idx] = runs
        bitsets[idx] = chunk_bitset_from_mask(D)
        if masks is not None:
            masks[idx] = D
        cells = int(D.sum())
        pair_count += cells
        row_run_count += int(runs.shape[0])
        chunk_total += bitsets[idx].chunk_count()
        pair_steps_sparse += cells * int(tables_post[idx].count)
        pair_steps_dense += shape[0] * shape[1] * int(tables_post[idx].count)
        if cells > ROUTE_DENSE_PAIR_FRACTION * shape[0] * shape[1]:
            route_dense_patches.append(int(idx))

    n_patches = len(row_runs)
    build = SparseWorkBuild(
        rows=shape[0],
        cols=shape[1],
        C=C,
        row_runs=row_runs,
        bitsets=bitsets,
        pair_count=pair_count,
        row_run_count=row_run_count,
        estimated_bytes_row_runs=(
            ROW_RUN_BYTES_PER_RUN * row_run_count + 4 * n_patches
        ),
        estimated_bytes_bitsets=CHUNK_BYTES * chunk_total + 4 * n_patches,
        dense_equivalent_bytes=sum(
            (shape[0] * shape[1] + 7) // 8 for _ in row_runs
        ),
        pair_steps_sparse=pair_steps_sparse,
        pair_steps_dense=pair_steps_dense,
        route_dense_patches=route_dense_patches,
        empty_work=False,
        table_changed_with_empty_C=table_changed_with_empty_C,
        table_digests_pre=digests_pre,
        table_digests_post=digests_post,
        build_seconds=time.perf_counter() - started,
        _masks=masks,
    )
    return build
