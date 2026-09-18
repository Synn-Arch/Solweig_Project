# SPDX-License-Identifier: GPL-3.0-only
"""Temporal checkpoints: full solver state captured at revision boundaries.

Design: ``docs/incremental_design_tool/realtime_collaboration/design/
r5-temporal-checkpoints-and-fast-serve.md`` section G2.0 (binding).

A checkpoint bounds how far a later replay (G2.1 sparse replay, G2.2 crash
restart) must rewind on the time axis. It is captured AFTER a batch fully
publishes at a scene revision — never on a wall clock, never by a
superseded/failed job — and contains:

1. ``scene_revision`` — the revision the checkpoint describes;
2. a per-``(kind, t)`` coverage manifest referencing the published patch
   files (the store's window entries, patch paths included, payloads
   never copied — the result store stays the source of truth for bytes);
3. the surface thermal state the SOLWEIG time loop threads k -> k+1,
   with ``next_step``: the state is AFTER completing timesteps
   ``0..next_step-1`` and a replay resumes AT ``next_step``;
4. an INPUT FINGERPRINT — digests of the inputs that DETERMINE the state
   (:data:`FINGERPRINT_KEYS`) — the applicability key. Storage and
   rotation stay revision-addressed, but a checkpoint may only warm-start
   a replay whose CURRENT inputs match its fingerprint: revisions can
   skip under mixed direct-edit + realtime flows, and a stale-scene
   checkpoint silently changes the science
   (``solweig_gpu/incremental/solver.py:1580-1608``). A mismatch is a
   typed refusal + cold replay + a fallback reason for telemetry —
   :func:`fingerprint_mismatch_reason` /
   :func:`select_replay_checkpoint` NEVER raise on mismatch and never
   guess;
5. sha256 checksums over every persisted byte (per-tensor digests plus a
   digest over the JSON payload, fingerprint included), validated at
   load.

The thermal state inventory (verified against the solver code, R5b
briefs): the ground/wall temperature maps ``Tgmap1``, ``Tgmap1E``,
``Tgmap1S``, ``Tgmap1W``, ``Tgmap1N`` and the surface temperature
``TgOut1`` — exactly the six float32 planes allocated at
``solweig_gpu/utci_process.py:614-619`` and mutated only by
``TsWaveDelay_2015a`` (``solweig_gpu/solweig.py:451-488``, called from
``Solweig_2022a_calc`` at ``solweig.py:2155-2160``) — plus the carried
scalars ``CI``/``firstdaytime``/``timeadd`` and ``Twater`` (the water
temperature: ``utci_process.py:728`` initializes it, and midnight rows
under land cover recompute ``Twater`` and ``CI``,
``utci_process.py:790-805`` — omitting them breaks mid-day warm
starts). ``timestepdec`` is DERIVED per step, never carried. EXTENT
policy: the chain is elementwise (no neighbor reads) and the solver
allocates it at WRITE-window extent (``utci_process.py:599``), so only
FULL-TILE states are composable — checkpoints persist full-tile planes
only, every plane must describe the same grid (write-fenced), and
windowed consumers slice from the full-tile checkpoint. Warm start only
ever helps ACROSS jobs/revisions (each write window runs the full
series for its own planes), so there is no intra-job state sharing.
``next_step=None`` marks a coverage-only checkpoint (the executor write
hook today — the solver
currently replays cold from timestep 0, ``solweig_gpu/incremental/
solver.py:1630-1633``, and the causal-prefix property at
``solver.py:1651-1654`` — state chains only carry forward — is what makes
a mid-history checkpoint a valid replay entry point once G2.1 wires the
warm start).

Bitwise discipline (the r4a/r4b gate): tensors are float32-exact,
persisted with :func:`numpy.save` on UNMODIFIED contiguous arrays — no
dtype narrowing (float64/int inputs are refused loudly at write), no
reordering. Checksums run over the raw bytes, so NaN payloads and signed
zeros roundtrip byte-exact; equality PROOFS must compare ``tobytes()``
(``==`` is not NaN-safe). float32 ``torch.Tensor`` inputs are accepted at
write — ``.detach().cpu().numpy()`` shares the bits (no dtype change, no
reorder) — and load returns numpy arrays that convert back via
``torch.from_numpy`` bit-for-bit.

Load policy (documented choice): :func:`load_checkpoint` REJECTS any
torn/corrupt checkpoint with the typed :class:`CheckpointError`
(truncated tensor bytes, flipped checksum, flipped payload digest,
corrupt JSON, wrong schema); :func:`load_latest_checkpoint` walks the
revisions newest-first, skips every rejected one, and returns ``None``
when nothing validates — the CALLER decides the fallback (the previous
valid checkpoint, or cold replay from revision 0). Wrong state is never
served.

Rotation: after every successful write, keep the last ``keep``
revisions (default 3) PLUS the revision-0 base, always. Checkpoints are
REPRODUCE aids, never a source of truth ahead of the op log: a failed
checkpoint write must never fail a published batch.

T25 spatial/COW extension (schema 3, DESIGN.ko.md §571-§579): a v3
checkpoint additionally carries the ``spatial_cow`` block — per-chunk
composed-scene/landcover digests on the :data:`CHUNK_CELLS` grid plus the
march globals' f32 bits. :func:`plan_chunk_cow_warm_start` turns a whole-
tile fingerprint mismatch into a per-chunk clean/dirty decision
(reach-dilation + amplitude ANY-crossing fences), and
:func:`compose_chunk_cow_state` splices checkpoint bytes on clean chunks
with a cold sub-window replay on dirty chunks — bit-identical to a fresh
full rebuild by the elementwise-chain argument. v2 checkpoints load
unchanged (``spatial_cow=None``: chunk COW refused with a typed reason,
whole-tile policy intact).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .geometry import RasterWindow
from .store import TemporalResultStore

__all__ = [
    "CHUNK_CELLS",
    "DEFAULT_KEEP",
    "FINGERPRINT_KEYS",
    "LOADABLE_SCHEMA_VERSIONS",
    "SCHEMA_VERSION",
    "CheckpointCoverageEntry",
    "CheckpointError",
    "CheckpointRecord",
    "ChunkCowPlan",
    "capture_spatial_cow",
    "compose_chunk_cow_state",
    "coverage_checkpoint",
    "coverage_manifest_from_store",
    "fingerprint_mismatch_reason",
    "list_checkpoints",
    "load_checkpoint",
    "load_latest_checkpoint",
    "plan_chunk_cow_warm_start",
    "prune_checkpoints",
    "select_replay_checkpoint",
    "thermal_checkpoint",
    "thermal_fingerprint",
    "write_checkpoint",
]

#: Bumped whenever the on-disk layout or the checksum scheme changes.
#: v3 adds the optional ``spatial_cow`` payload block (T25 chunk digests,
#: march globals); every earlier key is byte-identical, so v2 checkpoints
#: remain loadable — see :data:`LOADABLE_SCHEMA_VERSIONS`.
SCHEMA_VERSION = 3

#: Schema versions :func:`load_checkpoint` accepts. v2 records load with
#: ``spatial_cow=None`` (chunk COW refused at plan time by a typed reason,
#: whole-tile warm start untouched) — a behavior-preserving migration: the
#: T10 deferral asked for a schema bump + migration test, never a rewrite.
LOADABLE_SCHEMA_VERSIONS = (2, 3)

#: Chunk edge in pixels for the T25 spatial/COW digest grid. Mirrors
#: ``veg_svf_state._CHUNK_CELLS`` (the SVF store's own chunk grid); the two
#: are pinned equal by a test rather than an import so the checkpoint layer
#: stays importable without the SVF state machinery.
CHUNK_CELLS = 64

#: Rotation window: how many recent revisions survive a prune (the
#: revision-0 base is kept in addition, always).
DEFAULT_KEEP = 3

#: The canonical cross-timestep thermal tensors (utci_process.py:614-619):
#: ground + four wall-direction temperature maps and the surface
#: temperature — the only planes TsWaveDelay_2015a mutates. The format
#: accepts any named float32 full-tile map; G2.1 captures these names
#: when it wires the warm start.
THERMAL_TENSOR_NAMES = (
    "Tgmap1",
    "Tgmap1E",
    "Tgmap1S",
    "Tgmap1W",
    "Tgmap1N",
    "TgOut1",
)

#: The canonical carried scalars (R5b payload, verified against the
#: code): the clearness index and day-cycle flags plus the water
#: temperature — ``CI``/``Twater`` are recomputed at midnight rows
#: (utci_process.py:790-805), so a warm start mid-day needs them.
THERMAL_SCALAR_NAMES = ("CI", "firstdaytime", "timeadd", "Twater")

#: The input digests that DETERMINE the thermal state — the checkpoint
#: applicability key (NOT (workspace, revision): revisions can skip
#: under mixed direct-edit + realtime). ``met_prefix`` covers forcing
#: rows ``0..next_step-1`` (exactly the rows the state consumed).
FINGERPRINT_KEYS = (
    "composed_scene",
    "resolved_landcover",
    "model_parameters",
    "met_prefix",
)

#: ``rev-000123`` revision directories (the staging suffix ``.tmp`` never
#: matches — a kill mid-persist leaves staging invisible to listings).
_REVISION_DIR = re.compile(r"^rev-(\d{6})$")

#: Tensor names become file names: keep them to safe filename characters
#: (no traversal, no separators).
_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class CheckpointError(RuntimeError):
    """A checkpoint was refused: invalid input, or torn/corrupt on disk.

    Write side: structurally invalid records (non-float32 tensors, bad
    names, bad revisions) — the precision fence refuses loudly instead of
    narrowing silently. Load side: any torn or corrupt signature
    (truncated bytes, checksum mismatch, digest mismatch, unreadable
    JSON, wrong schema) — never served, the caller falls back.
    """


@dataclass(frozen=True, slots=True)
class CheckpointCoverageEntry:
    """One manifest row: a published ``(node, time)`` window entry.

    A faithful, JSON-serializable projection of
    :class:`~solweig_gpu.incremental.store.TemporalResultEntry` — the
    manifest REFERENCES the patch file (``patch_path``), it never copies
    the payload.
    """

    node_id: str
    time_index: int
    scene_revision: int
    job_id: str
    mode: str
    write_window: RasterWindow
    patch_path: Path | None


@dataclass
class CheckpointRecord:
    """One temporal checkpoint: revision, coverage, thermal state.

    ``thermal_tensors`` are named float32 (rows, cols) maps — full-tile,
    all one grid (write-fenced); torch float32 tensors accepted at write
    (the bridge shares bits). ``next_step``: the thermal state is after
    completing timesteps ``0..next_step-1``; a replay resumes AT
    ``next_step``. ``input_fingerprint`` carries the
    :data:`FINGERPRINT_KEYS` digests; a record with empty tensors,
    ``next_step=None`` and no fingerprint is a coverage-only checkpoint:
    a publication boundary marker with no warm state (what the executor
    write hook produces today — G2.1 computes the digests where it
    captures the state).
    """

    scene_revision: int
    coverage_manifest: tuple[CheckpointCoverageEntry, ...]
    thermal_tensors: dict[str, Any] = field(default_factory=dict)
    thermal_scalars: dict[str, float] = field(default_factory=dict)
    thermal_series: dict[str, tuple[float, ...]] = field(default_factory=dict)
    next_step: int | None = None
    input_fingerprint: dict[str, str] | None = None
    #: T25 spatial/COW applicability block (v3): per-chunk composed-scene
    #: and landcover digests plus the march globals (amplitude/scale bits).
    #: ``None`` on v2 records and coverage-only checkpoints — the chunk COW
    #: planner refuses those with a typed reason, never a guess.
    spatial_cow: Mapping[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Manifest + record builders
# ---------------------------------------------------------------------------


def coverage_manifest_from_store(
    store: TemporalResultStore,
) -> tuple[CheckpointCoverageEntry, ...]:
    """Snapshot every recorded window entry as a manifest tuple.

    Iterates the store's sorted keys and each key's window entries in
    their recorded (deterministic window) order, so two identical store
    states produce identical manifests.
    """
    manifest: list[CheckpointCoverageEntry] = []
    for node_id, time_index in store:
        for entry in store.window_entries(node_id, time_index):
            manifest.append(
                CheckpointCoverageEntry(
                    node_id=entry.node_id,
                    time_index=int(entry.time_index),
                    scene_revision=int(entry.scene_revision),
                    job_id=entry.job_id,
                    mode=entry.mode,
                    write_window=entry.write_window,
                    patch_path=None
                    if entry.patch_path is None
                    else Path(entry.patch_path),
                )
            )
    return tuple(manifest)


def coverage_checkpoint(
    *, scene_revision: int, store: TemporalResultStore
) -> CheckpointRecord:
    """A coverage-only checkpoint: revision + manifest, no warm state."""
    return CheckpointRecord(
        scene_revision=int(scene_revision),
        coverage_manifest=coverage_manifest_from_store(store),
    )


def thermal_fingerprint(
    *,
    building_dsm: np.ndarray,
    canopy: np.ndarray,
    dem: np.ndarray,
    resolved_landcover: np.ndarray | None,
    model_parameters: Mapping[str, Any] | None,
    met_prefix: np.ndarray,
) -> dict[str, str]:
    """Digest the inputs that DETERMINE a thermal state (G2.1 item 3).

    ONE spelling for both sides of the applicability key: the capture
    side (the worker, over the inputs the published run consumed) and the
    consumer side (G2.1 item 4, over the CURRENT inputs) — any drift
    between the two spellings would silently refuse or, worse, silently
    accept a stale state.

    ``composed_scene`` covers the three SOURCE rasters of the composed
    scene (building DSM, canopy, DEM): every vegetation or massing edit
    changes at least one, and the derived tensors (vegdsm family, SVF
    terms, march amplitude) are deterministic functions of these three.
    ``met_prefix`` is the forcing rows ``0..next_step-1`` — exactly the
    rows the captured state consumed (an edit BEYOND the prefix leaves
    this digest unchanged: the state never read that row, so a warm start
    remains sound — the boundary the tests pin). ``None`` and ``{}``
    model parameters digest identically (``run_full_tile`` treats both as
    the legacy path); a site without a land-cover raster carries a stable
    ``absent`` spelling.
    """
    def _digest(*chunks: bytes) -> str:
        hasher = hashlib.sha256()
        for chunk in chunks:
            hasher.update(chunk)
        return hasher.hexdigest()

    params = (
        {}
        if not model_parameters
        else {str(k): repr(v) for k, v in sorted(model_parameters.items())}
    )
    return {
        "composed_scene": _digest(
            np.ascontiguousarray(building_dsm, dtype=np.float32).tobytes(),
            np.ascontiguousarray(canopy, dtype=np.float32).tobytes(),
            np.ascontiguousarray(dem, dtype=np.float32).tobytes(),
        ),
        "resolved_landcover": _digest(
            b"absent"
            if resolved_landcover is None
            else np.ascontiguousarray(
                resolved_landcover, dtype=np.float32
            ).tobytes()
        ),
        "model_parameters": _digest(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
        "met_prefix": _digest(
            np.ascontiguousarray(met_prefix, dtype=np.float64).tobytes()
        ),
    }


def thermal_checkpoint(
    *,
    scene_revision: int,
    store: TemporalResultStore,
    final_state: Mapping[str, Any],
    input_fingerprint: Mapping[str, str],
) -> CheckpointRecord:
    """A full warm-state checkpoint from a captured ``final_state``.

    ``final_state`` is the vocabulary ``run_utci_window`` returns under
    ``return_final_state`` (G2.1 item 2): the six planes, the carried
    scalars, and ``next_step``. ``Twater`` is omitted while not yet
    established (the cold run's pre-midnight ``[]`` — absent on load
    means the warm start resumes it as not-established, never as a
    guess). The coverage manifest is snapshotted here so the record is
    complete at the publication boundary that writes it.
    """
    missing = [name for name in THERMAL_TENSOR_NAMES if name not in final_state]
    if missing:
        raise CheckpointError(
            f"final_state is missing the thermal plane(s) {missing}"
        )
    for name in ("CI", "firstdaytime", "timeadd", "next_step"):
        if name not in final_state:
            raise CheckpointError(
                f"final_state is missing the carried scalar {name!r}"
            )
    if "Twater" in final_state and final_state["Twater"] is not None:
        scalar_names = ("CI", "firstdaytime", "timeadd", "Twater")
    else:
        scalar_names = ("CI", "firstdaytime", "timeadd")
    return CheckpointRecord(
        scene_revision=int(scene_revision),
        coverage_manifest=coverage_manifest_from_store(store),
        thermal_tensors={
            name: final_state[name] for name in THERMAL_TENSOR_NAMES
        },
        thermal_scalars={name: float(final_state[name]) for name in scalar_names},
        next_step=int(final_state["next_step"]),
        input_fingerprint=dict(input_fingerprint),
    )


# ---------------------------------------------------------------------------
# T25: spatial/COW warm start (DESIGN.ko.md 11/§571-§579, TASKS T10 deferral)
#
# Whole-tile warm start refuses on ANY geometry edit — the r5 fingerprint is
# all-or-nothing. The thermal chain that produced the checkpoint planes is
# ELEMENTWISE (no neighbor reads, checkpoints.py header), but its per-step
# INPUTS are not: a cell's radiation at step t is folded from the SVF/shadow
# bundle, whose marches read occluders up to the march reach R pixels
# upwind. So cell x's thermal bytes at step k are a deterministic function
# of (met rows 0..k-1, carried scalars, scene cells within Chebyshev
# distance R of x, landcover at x, march globals) — and a geometry edit
# leaves them bit-identical whenever that whole input set is unchanged.
# Chunk COW exploits exactly that margin: chunks whose DILATED input
# neighborhood is digest-identical copy their planes from the checkpoint
# (copy-on-write: the bytes are only recomputed when their inputs moved),
# every other chunk replays COLD from step 0 over its sub-window — the
# elementwise chain makes a windowed run bit-equal to the full-tile run on
# that window, so the splice is bit-exact by construction, not by
# measurement.
#
# Fences (each has a RED test):
#   * dilation — a chunk is clean only if every chunk within Chebyshev
#     chunk-distance D = reach // cells + 1 matches (under-covering D
#     reuses cells whose march window crossed the edit: stale bytes);
#   * amplitude ANY-crossing — the march globals (amaxvalue = max of the
#     scene heights, scale) enter as f32 BITS. A changed step policy can
#     flip march outputs at cells whose raster neighborhood is unchanged
#     (the T22 regime-widening template), so ANY bit difference marks ALL
#     chunks dirty — there is no per-chunk amplitude argument;
#   * landcover is consumed per-cell (albedo/emissivity/TgK at the cell
#     itself; Twater itself is met-only), so its digest needs no dilation;
#   * met_prefix/model_parameters are global: a mismatch invalidates the
#     carried scalars themselves, and the scalars cannot be chunk-mixed —
#     full refusal, no COW at all;
#   * HARD contract (TASKS T10 / card): applicability is DIGEST-ONLY.
#     Scene revisions, SVF-store chunk state, veg_svf_state lanes never
#     enter this planner — thermal COW decides on its own digests and can
#     neither import nor leak SVF-state reuse decisions.
# ---------------------------------------------------------------------------


def _f32_bits_hex(value: float) -> str:
    """Exact-bit spelling of one march scalar (f32 bits as 8 hex chars)."""
    return struct.pack("<f", float(value)).hex()


def _chunk_index_grid(rows: int, cols: int, chunk_cells: int):
    """Yield ``((cr, cc), (row_slice, col_slice))`` over a chunk grid."""
    for cr, r0 in enumerate(range(0, rows, chunk_cells)):
        for cc, c0 in enumerate(range(0, cols, chunk_cells)):
            yield (cr, cc), (
                slice(r0, min(r0 + chunk_cells, rows)),
                slice(c0, min(c0 + chunk_cells, cols)),
            )


def chunk_scene_digests(
    *,
    building_dsm: np.ndarray,
    canopy: np.ndarray,
    dem: np.ndarray,
    resolved_landcover: np.ndarray | None,
    chunk_cells: int = CHUNK_CELLS,
) -> dict[str, Any]:
    """Per-chunk digests of the composed-scene rasters (ONE spelling).

    The capture side (checkpoint write) and the consume side (the COW
    planner over the CURRENT rasters) both call THIS function — any drift
    between two spellings would silently accept stale chunks. Each chunk's
    ``scene`` digest covers the three source rasters' f32 bytes at the
    chunk slice, concatenated in ``thermal_fingerprint``'s raster order;
    ``landcover`` digests the resolved land-cover the same way (stable
    ``absent`` spelling when the site has none).
    """
    if isinstance(chunk_cells, bool) or not isinstance(chunk_cells, int) or chunk_cells < 1:
        raise CheckpointError(f"chunk_cells must be a positive int, got {chunk_cells!r}")
    arrays = [
        np.ascontiguousarray(x, dtype=np.float32)
        for x in (building_dsm, canopy, dem)
    ]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or len(shapes.pop()) != 2:
        raise CheckpointError(
            "scene rasters must share one (rows, cols) shape for chunk "
            f"digestion, got {[array.shape for array in arrays]}"
        )
    rows, cols = arrays[0].shape
    scene: dict[str, str] = {}
    landcover: dict[str, str] | None = None if resolved_landcover is None else {}
    lc_array = (
        None
        if resolved_landcover is None
        else np.ascontiguousarray(resolved_landcover, dtype=np.float32)
    )
    if lc_array is not None and lc_array.shape != (rows, cols):
        raise CheckpointError(
            f"resolved_landcover shape {lc_array.shape} does not match the "
            f"scene rasters {(rows, cols)}"
        )
    for (cr, cc), (rs, cs) in _chunk_index_grid(rows, cols, chunk_cells):
        hasher = hashlib.sha256()
        for array in arrays:
            hasher.update(array[rs, cs].tobytes())
        scene[f"{cr},{cc}"] = hasher.hexdigest()
        if lc_array is not None:
            landcover[f"{cr},{cc}"] = hashlib.sha256(
                lc_array[rs, cs].tobytes()
            ).hexdigest()
    return {
        "chunk_cells": int(chunk_cells),
        "chunk_rows": (rows + chunk_cells - 1) // chunk_cells,
        "chunk_cols": (cols + chunk_cells - 1) // chunk_cells,
        "rows": int(rows),
        "cols": int(cols),
        "scene": scene,
        "landcover": landcover,
    }


def capture_spatial_cow(
    *,
    building_dsm: np.ndarray,
    canopy: np.ndarray,
    dem: np.ndarray,
    resolved_landcover: np.ndarray | None,
    march_amplitude: float,
    amaxvalue: float,
    scale: float,
    chunk_cells: int = CHUNK_CELLS,
) -> dict[str, Any]:
    """Build the ``spatial_cow`` record block at checkpoint capture time.

    The march globals use the solver's OWN spellings so the fence can never
    drift from the reach math: ``march_amplitude`` is the effective
    amplitude ``amaxvalue - min(a)`` (:func:`solver.required_halo_pixels`)
    and ``amaxvalue`` the global stop ``max(a.max(), vegdem.max())``
    (both f32 bits compared — a 1-ulp difference is a different step
    policy); ``scale`` is cells per meter (``1 / pixel_size_m``).
    """
    digests = chunk_scene_digests(
        building_dsm=building_dsm,
        canopy=canopy,
        dem=dem,
        resolved_landcover=resolved_landcover,
        chunk_cells=chunk_cells,
    )
    digests["march_amplitude_bits"] = _f32_bits_hex(march_amplitude)
    digests["amaxvalue_bits"] = _f32_bits_hex(amaxvalue)
    digests["scale_bits"] = _f32_bits_hex(scale)
    return digests


@dataclass(frozen=True, slots=True)
class ChunkCowPlan:
    """Chunk COW applicability decision + telemetry (§579 warm-hit shape).

    ``applicable=False`` carries a single ``refusal_reason`` (a global
    fence fired — no chunk may be reused); otherwise ``clean[cr, cc]``
    marks the chunks whose planes copy from the checkpoint and
    ``dirty_chunks`` the ones that must replay cold over ``dirty_window``
    (their bounding cell window). Counters feed the §579 telemetry: the
    access-width metric is ``cells_replayed_fraction`` (the share of the
    tile the time loop must recompute), ``plane_bytes_cow_reused`` the
    bytes the checkpoint serves without recomputation, and
    ``page_table_bytes`` the ledgered cost of the digest map itself
    against the memory budget.
    """

    applicable: bool
    chunk_grid: tuple[int, int]
    chunk_cells: int
    clean: np.ndarray
    dirty_chunks: tuple[tuple[int, int], ...]
    dirty_window: RasterWindow | None
    refusal_reason: str | None
    chunks_total: int
    chunks_cow_reused: int
    chunks_replayed: int
    cells_total: int
    cells_replayed: int
    cells_replayed_fraction: float
    plane_bytes_cow_reused: int
    page_table_bytes: int
    dirty_causes: dict[str, int]


def plan_chunk_cow_warm_start(
    record: CheckpointRecord,
    *,
    building_dsm: np.ndarray,
    canopy: np.ndarray,
    dem: np.ndarray,
    resolved_landcover: np.ndarray | None,
    model_parameters: Mapping[str, Any] | None,
    met_prefix: np.ndarray,
    march_amplitude: float,
    amaxvalue: float,
    scale: float,
    reach_pixels: int,
    chunk_cells: int = CHUNK_CELLS,
) -> ChunkCowPlan:
    """Decide which checkpoint chunks may warm the CURRENT inputs.

    NEVER raises on inapplicability and never guesses: a global fence
    (schema, shape, march globals, met prefix, model parameters) returns
    an all-dirty plan with ``refusal_reason`` set; the per-chunk pass then
    marks dirty every chunk whose reach-dilated scene digest, or whose own
    landcover digest, moved. The march globals use the capture spelling
    (effective amplitude ``amaxvalue - min(a)``, the global stop
    ``amaxvalue``, cells-per-meter scale). ``reach_pixels`` must be an
    UPPER bound on the march reach (per-axis pixel offset) over all sky
    patches — e.g. max over patches of the solver's
    ``_patch_march_reach_pixels`` at the CURRENT amplitude; the planner
    only uses the bound, never derives it, so its honesty stays at the
    caller.
    """
    def _all_dirty(reason: str) -> ChunkCowPlan:
        rows = cols = 0
        if record.thermal_tensors:
            first = next(iter(record.thermal_tensors.values()))
            rows, cols = np.asarray(first).shape
        chunk_rows = (rows + chunk_cells - 1) // chunk_cells
        chunk_cols = (cols + chunk_cells - 1) // chunk_cells
        return ChunkCowPlan(
            applicable=False,
            chunk_grid=(chunk_rows, chunk_cols),
            chunk_cells=int(chunk_cells),
            clean=np.zeros((chunk_rows, chunk_cols), dtype=bool),
            dirty_chunks=(),
            dirty_window=None,
            refusal_reason=reason,
            chunks_total=chunk_rows * chunk_cols,
            chunks_cow_reused=0,
            chunks_replayed=chunk_rows * chunk_cols,
            cells_total=rows * cols,
            cells_replayed=rows * cols,
            cells_replayed_fraction=1.0,
            plane_bytes_cow_reused=0,
            page_table_bytes=0,
            dirty_causes={"global": chunk_rows * chunk_cols},
        )

    if isinstance(reach_pixels, bool) or not isinstance(reach_pixels, int) or reach_pixels < 0:
        raise CheckpointError(
            f"reach_pixels must be a non-negative int, got {reach_pixels!r}"
        )
    if record.spatial_cow is None:
        return _all_dirty(
            f"checkpoint rev-{record.scene_revision:06d} carries no spatial "
            f"digests (schema {record.schema_version} or coverage-only); "
            "chunk COW refused, whole-tile fingerprint policy applies"
        )
    if not record.thermal_tensors:
        return _all_dirty("checkpoint carries no thermal planes to reuse")
    first = next(iter(record.thermal_tensors.values()))
    shape = tuple(np.asarray(first).shape)
    current = chunk_scene_digests(
        building_dsm=building_dsm,
        canopy=canopy,
        dem=dem,
        resolved_landcover=resolved_landcover,
        chunk_cells=chunk_cells,
    )
    if (current["rows"], current["cols"]) != shape:
        return _all_dirty(
            f"grid shape moved: checkpoint planes {shape} vs current scene "
            f"{(current['rows'], current['cols'])}"
        )
    stored = record.spatial_cow
    if int(stored["chunk_cells"]) != int(chunk_cells):
        return _all_dirty(
            f"chunk grid moved: checkpoint digested at "
            f"{stored['chunk_cells']}-cell chunks, planner asked for {chunk_cells}"
        )
    # March globals as BITS (the T22 ANY-crossing template): a step-policy
    # change can flip march outputs where every raster byte is unchanged,
    # so there is no per-chunk amnesty — any difference dirties all.
    if str(stored["march_amplitude_bits"]) != _f32_bits_hex(march_amplitude):
        return _all_dirty(
            "march amplitude bits moved "
            f"({stored['march_amplitude_bits']} -> {_f32_bits_hex(march_amplitude)}); "
            "step policy changed scene-wide, all chunks replay"
        )
    if str(stored["amaxvalue_bits"]) != _f32_bits_hex(amaxvalue):
        return _all_dirty(
            "global amaxvalue bits moved "
            f"({stored['amaxvalue_bits']} -> {_f32_bits_hex(amaxvalue)}); "
            "the march stop changed scene-wide, all chunks replay"
        )
    if str(stored["scale_bits"]) != _f32_bits_hex(scale):
        return _all_dirty(
            f"scale bits moved ({stored['scale_bits']} -> {_f32_bits_hex(scale)})"
        )
    # Global keys decide the carried SCALARS: a met-prefix or parameter
    # difference invalidates CI/firstdaytime/timeadd/Twater themselves, and
    # scalars cannot be chunk-mixed — refuse outright (HARD contract: the
    # thermal layer never borrows warmth across a scalar boundary).
    if record.input_fingerprint is None:
        return _all_dirty("checkpoint carries no input fingerprint")
    current_fingerprint = thermal_fingerprint(
        building_dsm=building_dsm,
        canopy=canopy,
        dem=dem,
        resolved_landcover=resolved_landcover,
        model_parameters=model_parameters,
        met_prefix=met_prefix,
    )
    for key in ("model_parameters", "met_prefix"):
        if record.input_fingerprint.get(key) != current_fingerprint[key]:
            return _all_dirty(
                f"global input digest moved at {key!r}: the carried scalars "
                "are invalid, no chunk may warm-start"
            )

    chunk_rows = int(current["chunk_rows"])
    chunk_cols = int(current["chunk_cols"])
    scene_changed = np.zeros((chunk_rows, chunk_cols), dtype=bool)
    lc_changed = np.zeros((chunk_rows, chunk_cols), dtype=bool)
    stored_scene = stored["scene"]
    stored_lc = stored["landcover"]
    for cr in range(chunk_rows):
        for cc in range(chunk_cols):
            key = f"{cr},{cc}"
            if stored_scene.get(key) != current["scene"][key]:
                scene_changed[cr, cc] = True
            elif (stored_lc or {}).get(key) != (current["landcover"] or {}).get(key):
                # Landcover is consumed per-cell (albedo/emissivity/TgK at
                # the cell itself; Twater is met-only), so its edits dirty
                # their own chunk WITHOUT reach dilation.
                lc_changed[cr, cc] = True
    # Reach dilation over SCENE changes only: a cell reads occluders up to
    # reach_pixels per axis, so chunk c's inputs include chunks at
    # Chebyshev chunk-distance d <= reach//cells + 1 (chunks at distance d
    # are at least (d-1)*cells pixels apart; covering d up to that bound
    # covers every read).
    dilation = reach_pixels // int(chunk_cells) + 1
    dirty = lc_changed.copy()
    if scene_changed.any():
        for cr in range(chunk_rows):
            for cc in range(chunk_cols):
                if not scene_changed[cr, cc]:
                    continue
                r0, r1 = max(0, cr - dilation), min(chunk_rows, cr + dilation + 1)
                c0, c1 = max(0, cc - dilation), min(chunk_cols, cc + dilation + 1)
                dirty[r0:r1, c0:c1] = True
    rows, cols = int(current["rows"]), int(current["cols"])
    dirty_chunks = tuple(
        (cr, cc)
        for cr in range(chunk_rows)
        for cc in range(chunk_cols)
        if dirty[cr, cc]
    )
    dirty_window = None
    if dirty_chunks:
        r_lo = min(cr for cr, _ in dirty_chunks) * int(chunk_cells)
        r_hi = min(rows, (max(cr for cr, _ in dirty_chunks) + 1) * int(chunk_cells))
        c_lo = min(cc for _, cc in dirty_chunks) * int(chunk_cells)
        c_hi = min(cols, (max(cc for _, cc in dirty_chunks) + 1) * int(chunk_cells))
        dirty_window = RasterWindow(r_lo, r_hi, c_lo, c_hi)
    cells_dirty = 0
    for (cr, cc), (rs, cs) in _chunk_index_grid(rows, cols, int(chunk_cells)):
        if dirty[cr, cc]:
            cells_dirty += (rs.stop - rs.start) * (cs.stop - cs.start)
    cells_total = rows * cols
    reused_cells = cells_total - cells_dirty
    n_planes = len(record.thermal_tensors)
    page_table_bytes = len(
        json.dumps(dict(stored), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    causes: dict[str, int] = {}
    if scene_changed.any():
        causes["scene_digest"] = int(scene_changed.sum())
    if lc_changed.any():
        causes["landcover_digest"] = int(lc_changed.sum())
    if dirty.any() and not causes:
        causes["dilation"] = int(dirty.sum())
    return ChunkCowPlan(
        applicable=True,
        chunk_grid=(chunk_rows, chunk_cols),
        chunk_cells=int(chunk_cells),
        clean=~dirty,
        dirty_chunks=dirty_chunks,
        dirty_window=dirty_window,
        refusal_reason=None,
        chunks_total=chunk_rows * chunk_cols,
        chunks_cow_reused=int((~dirty).sum()),
        chunks_replayed=int(dirty.sum()),
        cells_total=cells_total,
        cells_replayed=cells_dirty,
        cells_replayed_fraction=cells_dirty / cells_total,
        plane_bytes_cow_reused=reused_cells * 4 * n_planes,
        page_table_bytes=page_table_bytes,
        dirty_causes=causes or {"clean": 0},
    )


def compose_chunk_cow_state(
    record: CheckpointRecord,
    plan: ChunkCowPlan,
    dirty_replay_tensors: Mapping[str, Any] | None = None,
    *,
    dirty_replay_scalars: Mapping[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Splice the warm planes: checkpoint bytes on clean chunks, cold
    sub-window replay bytes on dirty chunks.

    Bit-exactness argument: the splice only COPIES bytes (slice reads and
    writes, no arithmetic), every cell is written exactly once (clean and
    dirty partition the grid), and both sources are bit-exact by their own
    gates — checkpoint planes are checksum-validated f32, the replay is a
    fresh solve whose windowed planes equal the full-tile run's crop on
    that window (elementwise chain). The composed state is therefore
    bit-identical to a full fresh rebuild at ``next_step`` whenever the
    plan's fences held; the oracle test proves that end-to-end.

    ``dirty_replay_tensors`` must cover every thermal plane name the
    record carries, at exactly ``plan.dirty_window``'s extent (the replay
    sub-run's own final-state crop). ``dirty_replay_scalars``, when
    given, must equal the record's carried scalars bit-for-bit — the
    scalars are met-only, so a fresh replay recomputes identical ones; a
    difference is a soundness bug somewhere else and is refused loudly,
    never averaged or guessed.
    """
    if not plan.applicable:
        raise CheckpointError(
            f"chunk COW compose refused: plan is not applicable "
            f"({plan.refusal_reason})"
        )
    names = sorted(record.thermal_tensors)
    if not names:
        raise CheckpointError("checkpoint carries no thermal planes to compose")
    first = np.asarray(record.thermal_tensors[names[0]])
    rows, cols = first.shape[0], first.shape[1]
    if plan.dirty_chunks and dirty_replay_tensors is None:
        raise CheckpointError(
            f"plan marks {len(plan.dirty_chunks)} chunk(s) dirty but no "
            "dirty_replay_tensors were provided — dirty chunks must REPLAY "
            "cold, never zero-init (their state at next_step is evolved)"
        )
    replay: dict[str, np.ndarray] = {}
    if dirty_replay_tensors is not None:
        window = plan.dirty_window
        expected = (window.row_stop - window.row_start, window.col_stop - window.col_start)
        for name in names:
            if name not in dirty_replay_tensors:
                raise CheckpointError(
                    f"dirty replay is missing the thermal plane {name!r}"
                )
            plane = dirty_replay_tensors[name]
            if isinstance(plane, torch.Tensor):
                plane = plane.detach().cpu().numpy()
            plane = np.asarray(plane)
            if plane.dtype != np.float32 or plane.ndim != 2:
                raise CheckpointError(
                    f"dirty replay plane {name!r} must be float32 "
                    f"(rows, cols), got dtype {plane.dtype!r} ndim {plane.ndim}"
                )
            if plane.shape != expected:
                raise CheckpointError(
                    f"dirty replay plane {name!r} shape {plane.shape} does "
                    f"not match the plan's dirty window extent {expected}"
                )
            replay[name] = plane
    planes: dict[str, np.ndarray] = {}
    for name in names:
        source = np.asarray(record.thermal_tensors[name])
        if source.dtype != np.float32 or source.shape != (rows, cols):
            raise CheckpointError(
                f"checkpoint plane {name!r} must be float32 {(rows, cols)}, "
                f"got dtype {source.dtype!r} shape {source.shape}"
            )
        composed = np.empty((rows, cols), dtype=np.float32)
        window = plan.dirty_window
        for (cr, cc), (rs, cs) in _chunk_index_grid(rows, cols, plan.chunk_cells):
            if plan.clean[cr, cc]:
                composed[rs, cs] = source[rs, cs]
            else:
                assert window is not None  # dirty chunk exists ⇒ window set
                local_rs = slice(rs.start - window.row_start, rs.stop - window.row_start)
                local_cs = slice(cs.start - window.col_start, cs.stop - window.col_start)
                composed[rs, cs] = replay[name][local_rs, local_cs]
        planes[name] = composed
    scalars = {str(k): float(v) for k, v in record.thermal_scalars.items()}
    if dirty_replay_scalars is not None:
        for name, value in scalars.items():
            replayed = dirty_replay_scalars.get(name)
            if replayed is None or float(replayed) != value:
                raise CheckpointError(
                    f"carried scalar {name!r} diverged between checkpoint "
                    f"({value!r}) and the fresh dirty-window replay "
                    f"({replayed!r}) — the scalars are met-only by the "
                    "soundness argument; this divergence is a bug, refusing"
                )
        missing = sorted(set(dirty_replay_scalars) - set(scalars))
        if missing:
            raise CheckpointError(
                f"dirty replay carries unknown scalar(s) {missing}"
            )
    return planes, scalars


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def _tensor_checksum(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


def _canonical_payload_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_payload_json(payload).encode("utf-8")
    ).hexdigest()


def _manifest_to_json(
    manifest: Sequence[CheckpointCoverageEntry],
) -> list[dict[str, Any]]:
    return [
        {
            "node_id": entry.node_id,
            "time_index": int(entry.time_index),
            "scene_revision": int(entry.scene_revision),
            "job_id": entry.job_id,
            "mode": entry.mode,
            "write_window": [
                int(entry.write_window.row_start),
                int(entry.write_window.row_stop),
                int(entry.write_window.col_start),
                int(entry.write_window.col_stop),
            ],
            "patch_path": None
            if entry.patch_path is None
            else str(entry.patch_path),
        }
        for entry in manifest
    ]


# ---------------------------------------------------------------------------
# Validation (the write-side fence)
# ---------------------------------------------------------------------------


def _as_float32_map(name: str, value: Any) -> np.ndarray:
    """Coerce one thermal tensor to a float32 numpy map, bits untouched.

    float32 ``torch.Tensor`` inputs bridge via ``.detach().cpu().numpy()``
    (shares memory — no dtype change, no reorder). Any other dtype is
    REFUSED loudly: never narrow precision silently.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        raise CheckpointError(
            f"thermal tensor {name!r} must be a numpy array or torch "
            f"tensor, got {type(value).__name__}"
        )
    if value.dtype != np.float32:
        raise CheckpointError(
            f"thermal tensor {name!r} must be float32, got "
            f"{value.dtype!r} — checkpoint tensors are float32-exact; "
            "narrowing/widening silently is refused"
        )
    if value.ndim != 2:
        raise CheckpointError(
            f"thermal tensor {name!r} must be a (rows, cols) map, got "
            f"ndim={value.ndim}"
        )
    return value


def _validate_spatial_cow(
    spatial_cow: Mapping[str, Any] | None,
    tensor_shape: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """Write-side fence for the T25 ``spatial_cow`` block."""
    if spatial_cow is None:
        return None
    if not isinstance(spatial_cow, Mapping):
        raise CheckpointError(
            "spatial_cow must be a mapping (capture_spatial_cow's output), "
            f"got {type(spatial_cow).__name__}"
        )
    try:
        cells = int(spatial_cow["chunk_cells"])
        chunk_rows = int(spatial_cow["chunk_rows"])
        chunk_cols = int(spatial_cow["chunk_cols"])
        rows = int(spatial_cow["rows"])
        cols = int(spatial_cow["cols"])
        scene = spatial_cow["scene"]
        landcover = spatial_cow["landcover"]
        amplitude_bits = str(spatial_cow["march_amplitude_bits"])
        amax_bits = str(spatial_cow["amaxvalue_bits"])
        scale_bits = str(spatial_cow["scale_bits"])
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(
            f"spatial_cow block is malformed: {error}"
        ) from error
    if cells < 1 or rows < 1 or cols < 1 or chunk_rows < 1 or chunk_cols < 1:
        raise CheckpointError(
            f"spatial_cow grid fields must be positive, got cells={cells}, "
            f"rows={rows}, cols={cols}, chunk grid=({chunk_rows}, {chunk_cols})"
        )
    if (chunk_rows, chunk_cols) != (
        (rows + cells - 1) // cells,
        (cols + cells - 1) // cells,
    ):
        raise CheckpointError(
            f"spatial_cow chunk grid ({chunk_rows}, {chunk_cols}) does not "
            f"match {rows}x{cols} at {cells}-cell chunks"
        )
    if tensor_shape is not None and tensor_shape != (rows, cols):
        raise CheckpointError(
            f"spatial_cow grid {rows}x{cols} does not match the thermal "
            f"planes {tensor_shape}"
        )
    if tensor_shape is None:
        raise CheckpointError(
            "spatial_cow requires thermal tensors: a coverage-only "
            "checkpoint has no planes to digest"
        )
    hex_digest = re.compile(r"^[0-9a-f]{64}$")
    for field_name, mapping in (("scene", scene), ("landcover", landcover)):
        if field_name == "landcover" and mapping is None:
            continue  # stable "absent" spelling: a site without a raster
        if not isinstance(mapping, dict):
            raise CheckpointError(
                f"spatial_cow[{field_name!r}] must be a chunk-index -> digest "
                f"mapping, got {type(mapping).__name__}"
            )
        if len(mapping) != chunk_rows * chunk_cols:
            raise CheckpointError(
                f"spatial_cow[{field_name!r}] carries {len(mapping)} digest(s) "
                f"for a {chunk_rows}x{chunk_cols} chunk grid"
            )
        for key, digest in mapping.items():
            if not isinstance(key, str) or not re.match(r"^\d+,\d+$", key):
                raise CheckpointError(
                    f"spatial_cow[{field_name!r}] key {key!r} is not a "
                    "'<row>,<col>' chunk index"
                )
            if not isinstance(digest, str) or not hex_digest.match(digest):
                raise CheckpointError(
                    f"spatial_cow[{field_name!r}][{key!r}] is not a sha256 "
                    "hex digest"
                )
    bits = re.compile(r"^[0-9a-f]{8}$")
    for field_name, value in (
        ("march_amplitude_bits", amplitude_bits),
        ("amaxvalue_bits", amax_bits),
        ("scale_bits", scale_bits),
    ):
        if not bits.match(value):
            raise CheckpointError(
                f"spatial_cow[{field_name!r}] must be 8 f32-bit hex chars, "
                f"got {value!r}"
            )
    return {
        "chunk_cells": cells,
        "chunk_rows": chunk_rows,
        "chunk_cols": chunk_cols,
        "rows": rows,
        "cols": cols,
        "scene": {str(k): str(v) for k, v in scene.items()},
        "landcover": None
        if landcover is None
        else {str(k): str(v) for k, v in landcover.items()},
        "march_amplitude_bits": amplitude_bits,
        "amaxvalue_bits": amax_bits,
        "scale_bits": scale_bits,
    }


def _validate_record(
    record: CheckpointRecord,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, tuple[float, ...]],
    dict[str, Any] | None,
]:
    if isinstance(record.scene_revision, bool) or not isinstance(
        record.scene_revision, int
    ) or record.scene_revision < 0:
        raise CheckpointError(
            f"scene_revision must be a non-negative int, got "
            f"{record.scene_revision!r}"
        )
    if record.next_step is not None and (
        isinstance(record.next_step, bool)
        or not isinstance(record.next_step, int)
        or record.next_step < 0
    ):
        raise CheckpointError(
            f"next_step must be a non-negative int or None, got "
            f"{record.next_step!r}"
        )
    if record.input_fingerprint is not None:
        if not isinstance(record.input_fingerprint, Mapping):
            raise CheckpointError(
                "input_fingerprint must be a mapping of digest names to "
                "hex strings, or None"
            )
        for key, digest in record.input_fingerprint.items():
            if key not in FINGERPRINT_KEYS:
                raise CheckpointError(
                    f"input_fingerprint key {key!r} is not one of "
                    f"{FINGERPRINT_KEYS}"
                )
            if not isinstance(digest, str):
                raise CheckpointError(
                    f"input_fingerprint[{key!r}] must be a hex digest "
                    f"string, got {type(digest).__name__}"
                )
    tensors: dict[str, np.ndarray] = {}
    for name, value in record.thermal_tensors.items():
        if not isinstance(name, str) or not _TENSOR_NAME.match(name):
            raise CheckpointError(
                f"thermal tensor names must match {_TENSOR_NAME.pattern!r} "
                f"(they become file names), got {name!r}"
            )
        tensors[name] = _as_float32_map(name, value)
    if tensors:
        # FULL-TILE fence: the thermal chain is elementwise, so every
        # plane must describe the SAME grid or a windowed consumer would
        # slice a patchwork. Refuse the write, never persist the mismatch.
        shapes = {array.shape for array in tensors.values()}
        if len(shapes) != 1:
            raise CheckpointError(
                f"thermal tensors must all share one full-tile shape, got "
                f"{sorted(shapes)} — windowed consumers slice from the "
                "full-tile checkpoint"
            )
    scalars: dict[str, float] = {}
    for name, value in record.thermal_scalars.items():
        if not isinstance(name, str) or not _TENSOR_NAME.match(name):
            raise CheckpointError(f"bad thermal scalar name {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CheckpointError(
                f"thermal scalar {name!r} must be a float, got "
                f"{type(value).__name__}"
            )
        scalars[name] = float(value)
    series: dict[str, tuple[float, ...]] = {}
    for name, values in record.thermal_series.items():
        if not isinstance(name, str) or not _TENSOR_NAME.match(name):
            raise CheckpointError(f"bad thermal series name {name!r}")
        coerced = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CheckpointError(
                    f"thermal series {name!r} must contain floats, got "
                    f"{type(value).__name__}"
                )
            coerced.append(float(value))
        series[name] = tuple(coerced)
    tensor_shape = (
        next(iter(tensors.values())).shape if tensors else None
    )
    spatial = _validate_spatial_cow(record.spatial_cow, tensor_shape)
    return tensors, scalars, series, spatial


# ---------------------------------------------------------------------------
# Write (atomic) + rotation
# ---------------------------------------------------------------------------


def write_checkpoint(
    record: CheckpointRecord,
    root: str | Path,
    *,
    keep: int = DEFAULT_KEEP,
) -> Path:
    """Persist ``record`` atomically and rotate; return the revision dir.

    Same idiom as ``VegOcclusionStore._persist``: stage the full revision
    directory under ``rev-XXXXXX.tmp`` then :func:`os.rename` it into
    place, so a kill mid-persist leaves the previous revision intact and
    never exposes a partial checkpoint. Rotation keeps the last ``keep``
    revisions plus the revision-0 base.
    """
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise CheckpointError(f"keep must be a positive int, got {keep!r}")
    tensors, scalars, series, spatial = _validate_record(record)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"rev-{record.scene_revision:06d}"
    staging = final.with_name(final.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    payload = {
        "scene_revision": int(record.scene_revision),
        "next_step": None if record.next_step is None else int(record.next_step),
        "input_fingerprint": None
        if record.input_fingerprint is None
        else {key: str(record.input_fingerprint[key]) for key in FINGERPRINT_KEYS},
        "coverage_manifest": _manifest_to_json(record.coverage_manifest),
        "thermal_scalars": scalars,
        "thermal_series": {name: list(values) for name, values in series.items()},
        "spatial_cow": spatial,
    }
    try:
        tensor_dir = staging / "tensors"
        tensor_dir.mkdir()
        checksums = {}
        for name in sorted(tensors):
            array = np.ascontiguousarray(tensors[name])
            np.save(tensor_dir / f"{name}.npy", array)
            checksums[name] = _tensor_checksum(array)
        # The checksum map lives INSIDE the payload (G2.1 hardening): the
        # payload digest therefore covers every persisted-plane digest too,
        # so tampering the map in metadata rejects at the digest layer,
        # before any tensor bytes are read. Byte-level tampering of the
        # .npy files is still caught by the per-tensor checksum compare.
        payload["thermal_tensor_checksums"] = checksums
        (staging / "checkpoint.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "payload": payload,
                    "payload_sha256": _payload_digest(payload),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        # Remove a same-revision predecessor only once the replacement is
        # fully staged — a failure above leaves the old revision intact.
        if final.exists():
            shutil.rmtree(final)
        os.rename(staging, final)
    except Exception:
        # Never leave a half-written staging directory behind on failure;
        # the previous revision (if any) is untouched by construction.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    prune_checkpoints(root, keep=keep)
    return final


def list_checkpoints(root: str | Path) -> tuple[int, ...]:
    """Sorted scene revisions present under ``root`` (staging excluded)."""
    root = Path(root)
    if not root.is_dir():
        return ()
    revisions = []
    for path in root.iterdir():
        match = _REVISION_DIR.match(path.name)
        if match is not None and path.is_dir():
            revisions.append(int(match.group(1)))
    return tuple(sorted(revisions))


def prune_checkpoints(root: str | Path, *, keep: int = DEFAULT_KEEP) -> tuple[Path, ...]:
    """Delete all but the last ``keep`` revisions plus the rev-0 base.

    Returns the deleted revision directories, oldest first. The revision-0
    base survives every prune (a replay always has a floor to fall back
    to), and it never double-counts against ``keep`` when it is already
    inside the newest window.
    """
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise CheckpointError(f"keep must be a positive int, got {keep!r}")
    root = Path(root)
    revisions = list_checkpoints(root)
    survivors = set(revisions[-keep:])
    survivors.add(0)
    deleted = tuple(
        root / f"rev-{revision:06d}"
        for revision in revisions
        if revision not in survivors
    )
    for stale in deleted:
        shutil.rmtree(stale, ignore_errors=True)
    return deleted


# ---------------------------------------------------------------------------
# Load (checksum-validated, typed rejection)
# ---------------------------------------------------------------------------


def load_checkpoint(revision_dir: str | Path) -> CheckpointRecord:
    """Load and validate one checkpoint directory.

    Raises :class:`CheckpointError` on ANY torn/corrupt signature —
    unreadable metadata, wrong schema, payload digest mismatch, missing/
    unreadable tensor files, dtype drift, per-tensor checksum mismatch.
    The caller decides the fallback (see :func:`load_latest_checkpoint`).
    """
    revision_dir = Path(revision_dir)
    try:
        meta = json.loads((revision_dir / "checkpoint.json").read_text())
    except (OSError, ValueError) as error:
        raise CheckpointError(
            f"unreadable checkpoint metadata in {revision_dir}: {error}"
        ) from error
    if not isinstance(meta, dict):
        raise CheckpointError(f"checkpoint metadata is not an object: {revision_dir}")
    schema = meta.get("schema_version")
    if schema not in LOADABLE_SCHEMA_VERSIONS:
        raise CheckpointError(
            f"checkpoint schema {schema!r} not in {LOADABLE_SCHEMA_VERSIONS} "
            f"in {revision_dir}"
        )
    payload = meta.get("payload")
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint payload missing in {revision_dir}")
    actual_digest = _payload_digest(payload)
    if meta.get("payload_sha256") != actual_digest:
        raise CheckpointError(
            f"checkpoint payload digest mismatch in {revision_dir}: "
            f"recorded {meta.get('payload_sha256')!r} != recomputed "
            f"{actual_digest!r} — torn or tampered checkpoint, refusing"
        )

    manifest: list[CheckpointCoverageEntry] = []
    raw_manifest = payload.get("coverage_manifest", [])
    if not isinstance(raw_manifest, list):
        raise CheckpointError("coverage_manifest must be a list")
    try:
        for raw in raw_manifest:
            window = raw["write_window"]
            manifest.append(
                CheckpointCoverageEntry(
                    node_id=str(raw["node_id"]),
                    time_index=int(raw["time_index"]),
                    scene_revision=int(raw["scene_revision"]),
                    job_id=str(raw["job_id"]),
                    mode=str(raw["mode"]),
                    write_window=RasterWindow(
                        int(window[0]), int(window[1]), int(window[2]), int(window[3])
                    ),
                    patch_path=None
                    if raw.get("patch_path") is None
                    else Path(raw["patch_path"]),
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(
            f"malformed coverage manifest row in {revision_dir}: {error}"
        ) from error

    scalars: dict[str, float] = {}
    raw_scalars = payload.get("thermal_scalars", {})
    if not isinstance(raw_scalars, dict):
        raise CheckpointError("thermal_scalars must be an object")
    try:
        scalars = {str(k): float(v) for k, v in raw_scalars.items()}
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"malformed thermal scalars: {error}") from error

    series: dict[str, tuple[float, ...]] = {}
    raw_series = payload.get("thermal_series", {})
    if not isinstance(raw_series, dict):
        raise CheckpointError("thermal_series must be an object")
    try:
        series = {
            str(k): tuple(float(v) for v in values)
            for k, values in raw_series.items()
        }
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"malformed thermal series: {error}") from error

    checksums = payload.get("thermal_tensor_checksums", {})
    if not isinstance(checksums, dict):
        raise CheckpointError("thermal_tensor_checksums must be an object")
    tensors: dict[str, np.ndarray] = {}
    for name in sorted(checksums):
        if not isinstance(name, str) or not _TENSOR_NAME.match(name):
            raise CheckpointError(
                f"thermal tensor name {name!r} is not a safe file name"
            )
        path = revision_dir / "tensors" / f"{name}.npy"
        try:
            array = np.load(path)
        except (OSError, ValueError, EOFError) as error:
            raise CheckpointError(
                f"unreadable thermal tensor {name!r} in {revision_dir}: "
                f"{error}"
            ) from error
        if array.dtype != np.float32 or array.ndim != 2:
            raise CheckpointError(
                f"thermal tensor {name!r} must be a float32 (rows, cols) "
                f"map, got dtype {array.dtype!r} ndim {array.ndim}"
            )
        actual = _tensor_checksum(array)
        if actual != checksums[name]:
            raise CheckpointError(
                f"thermal tensor {name!r} checksum mismatch in "
                f"{revision_dir}: recorded {checksums[name]!r} != "
                f"recomputed {actual!r} — torn or tampered checkpoint, "
                "refusing"
            )
        tensors[name] = array

    raw_revision = payload.get("scene_revision")
    if isinstance(raw_revision, bool) or not isinstance(raw_revision, int):
        raise CheckpointError(
            f"payload scene_revision must be an int, got {raw_revision!r}"
        )
    raw_next = payload.get("next_step")
    if raw_next is not None and (
        isinstance(raw_next, bool) or not isinstance(raw_next, int) or raw_next < 0
    ):
        raise CheckpointError(
            f"payload next_step must be a non-negative int or null, got "
            f"{raw_next!r}"
        )
    raw_fingerprint = payload.get("input_fingerprint")
    if raw_fingerprint is None:
        fingerprint: dict[str, str] | None = None
    elif isinstance(raw_fingerprint, dict):
        fingerprint = {str(k): str(v) for k, v in raw_fingerprint.items()}
        unknown = sorted(set(fingerprint) - set(FINGERPRINT_KEYS))
        if unknown:
            raise CheckpointError(
                f"unknown input_fingerprint keys {unknown}; expected "
                f"{list(FINGERPRINT_KEYS)}"
            )
    else:
        raise CheckpointError(
            "payload input_fingerprint must be an object or null, got "
            f"{type(raw_fingerprint).__name__}"
        )
    raw_spatial = payload.get("spatial_cow")
    if raw_spatial is not None and not isinstance(raw_spatial, dict):
        raise CheckpointError(
            "payload spatial_cow must be an object or null, got "
            f"{type(raw_spatial).__name__}"
        )
    tensor_shape = (
        next(iter(tensors.values())).shape if tensors else None
    )
    spatial = _validate_spatial_cow(raw_spatial, tensor_shape)
    return CheckpointRecord(
        scene_revision=int(raw_revision),
        coverage_manifest=tuple(manifest),
        thermal_tensors=tensors,
        thermal_scalars=scalars,
        thermal_series=series,
        next_step=None if raw_next is None else int(raw_next),
        input_fingerprint=fingerprint,
        spatial_cow=spatial,
        schema_version=int(schema),
    )


def load_latest_checkpoint(root: str | Path) -> CheckpointRecord | None:
    """Newest VALID checkpoint under ``root``, or ``None``.

    Walks the revisions newest-first and skips every checkpoint that
    fails validation — the crash-restart fallback: a torn newest revision
    falls back to the previous one; when nothing validates the caller
    replays cold from revision 0.
    """
    revisions = list_checkpoints(root)
    root = Path(root)
    for revision in reversed(revisions):
        try:
            return load_checkpoint(root / f"rev-{revision:06d}")
        except CheckpointError:
            continue
    return None


# ---------------------------------------------------------------------------
# Applicability: the input fingerprint (R5b supplement)
# ---------------------------------------------------------------------------


def fingerprint_mismatch_reason(
    record: CheckpointRecord,
    expected_fingerprint: Mapping[str, str],
) -> str | None:
    """Why ``record`` may NOT warm-start the current inputs, or ``None``.

    ``None`` means every :data:`FINGERPRINT_KEYS` digest recorded on the
    checkpoint equals the current inputs' digest — applicable. Any
    mismatch returns a human-readable reason for the fallback_reason
    telemetry: this function NEVER raises and NEVER guesses; the caller
    replays cold with the reason attached.
    """
    if record.input_fingerprint is None:
        return (
            f"checkpoint rev-{record.scene_revision:06d} carries no input "
            "fingerprint (coverage-only); warm start refused"
        )
    for key in FINGERPRINT_KEYS:
        actual = record.input_fingerprint.get(key)
        wanted = expected_fingerprint.get(key)
        if actual != wanted:
            return (
                f"input fingerprint mismatch at {key!r}: checkpoint "
                f"rev-{record.scene_revision:06d} recorded {actual!r}, "
                f"current inputs digest {wanted!r}"
            )
    return None


def select_replay_checkpoint(
    root: str | Path,
    expected_fingerprint: Mapping[str, str],
) -> tuple[CheckpointRecord | None, str | None]:
    """Pick the warm-start entry point for a replay, or report cold.

    Walks the revisions newest-first, skipping torn/corrupt checkpoints
    (typed rejection consumed here, never surfaced), and returns the
    first record whose input fingerprint matches the CURRENT inputs.
    Fingerprint — not revision — decides: an older checkpoint under
    matching inputs beats a newer one under stale inputs.

    Returns ``(record, None)`` on a match (``record.next_step`` is the
    resume point) or ``(None, reason)`` for a cold replay, where
    ``reason`` reflects the newest revision's outcome (fingerprint
    mismatch, unreadable checkpoint, or plain absence) — the
    fallback_reason telemetry. This function never raises on mismatch
    and never guesses.
    """
    root = Path(root)
    revisions = list_checkpoints(root)
    if not revisions:
        return None, f"no checkpoint under {root} — cold replay from step 0"
    newest_reason: str | None = None
    for revision in reversed(revisions):
        try:
            record = load_checkpoint(root / f"rev-{revision:06d}")
        except CheckpointError as error:
            if newest_reason is None:
                newest_reason = (
                    f"newest readable-looking checkpoint rev-{revision:06d} "
                    f"is torn/corrupt ({error}); cold replay"
                )
            continue
        reason = fingerprint_mismatch_reason(record, expected_fingerprint)
        if reason is None:
            return record, None
        if newest_reason is None:
            newest_reason = reason
    return None, newest_reason
