# SPDX-License-Identifier: GPL-3.0-only
"""T26 whole-cell scalar bundle reuse harness (DESIGN §474 / §8.3).

The candidate: a cell's COMPLETED fold bundle (the 10 clamped veg
accumulators, + svftotal) is a pure function of that cell's input bits,
so a cell whose input bits all match a previously-folded state can replay
the bundle instead of re-running the 153-patch annulus recurrence. This
is NOT the forbidden float delta update (no ``old_sum - old_term +
new_term`` anywhere): the replayed values are the fold's own bits.

Bundle-state enumeration (completeness proof — T21-style):

=====================  ==========================  =======================
item                   kind                        consumed by
=====================  ==========================  =======================
vegsh packed bits      20 B per cell               v = bit -> {0.,1.}
vbsh packed bits       20 B per cell               b = bit -> {0.,1.}
vegdem2[i,j]           f32 plane                   ONLY the ==0.0 test
                                                    (last correction;
                                                    -0.0 shares class,
                                                    NaN excluded)
svf_building[i,j]      f32 plane                   svftotal only
TRANS / W_ISO /        module frozen constants     whole recurrence
W_ANISO / PATCH_RING                               (namespace stamp)
AZIMUTH / directions
=====================  ==========================  =======================

The fold kernel (:func:`svf_fold._fold_kernel`) reads ONLY index [i, j]
of its plane/packed operands and re-initializes every accumulator per
cell — there is NO cross-cell, cross-edit, or thread-carried state.
The bundle is therefore MEMORYLESS-pure, and two cells with identical
input bits MUST have bit-identical bundles (the cache exploits exactly
that; asserted as a gate, not assumed).

TRANS scope pin (C1): ``TRANS`` is a module-level frozen bit pattern
(``svf_fold.py`` _TRANS_U32 -> TRANS), hoisted once to
``one_minus_trans`` and consumed only in the svftotal line. Its scope
is the whole process (fenced by :func:`svf_fold.frozen_tables_digest`,
whose digest includes the trans bits, and ``FOLD_CONSTANTS_SHA256``).
It is NOT a 5th per-cell fingerprint field — but the cache namespace
stamp carries its bits, so any regime where trans varies is a stamp
mismatch and refuses (mutation M6's teeth).

Fingerprints (both bit-exact):

* variant **A** (strict §474 full bundle): veg 20 B + vbsh 20 B +
  vegdem2 f32 bits + svf_building f32 bits; 11-float bundle replayed.
* variant **B** (live candidate): veg 20 B + vbsh 20 B + the
  vegdem2-zero predicate bit; the 10 accumulators are replayed and
  svftotal is recomputed fresh (``svf_building - (1 - svfveg) *
  (1 - trans)`` — 2 flops, IEEE f32, bit-identical to the kernel's
  scalar chain; pinned by gate).

Keys are the raw fingerprint BYTES (dict on ``bytes``) — equality is
exact byte equality, so a "hash collision" is unconstructible by
construction; the defensive compare is the lookup itself (C2 accounts
the full hit path, lookup included, never a hash-free fantasy row).

Discipline: harness-only — :func:`svf_fold.fold_svf` is imported, never
modified. Dev-only ablation; production wiring is out of card scope.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from solweig_core import bitplanes as bp
from solweig_core.numba_cpu import svf_fold as sf

ART = Path.home() / "Workspace" / "solweig_ultrafast_artifacts" / "t26"

#: Per-cell fingerprint widths (bytes).
FP_B_BYTES = 41   # variant B: 20 veg + 20 vbsh + 1 predicate byte
FP_A_BYTES = 48   # variant A: 20 + 20 + 4 vegdem2 + 4 svf_building

#: Bundle value widths (floats).
BUNDLE_B_FLOATS = 10   # accumulators only; svftotal recomputed
BUNDLE_A_FLOATS = 11   # includes the fold's own svftotal

FOLD_PLANES = sf.FOLD_OUTPUT_NAMES  # 11 names, svftotal last


def namespace_stamp() -> str:
    """Validity stamp for every cache entry: the fold's global-constant
    scope (frozen tables incl. the trans bits + direction masks)."""
    h = hashlib.sha256()
    h.update(sf.frozen_tables_digest().encode())
    h.update(sf.FOLD_CONSTANTS_SHA256.encode())
    for d in (sf.DIRECTION_E, sf.DIRECTION_S, sf.DIRECTION_W,
              sf.DIRECTION_N):
        h.update(np.ascontiguousarray(d).tobytes())
    return h.hexdigest()


class CacheFenceError(RuntimeError):
    """The cache's constant-scope stamp no longer matches the live fold
    namespace (frozen tables / trans bits / directions changed). Serving
    would silently fabricate bundles — refuse loudly."""


@dataclass
class BundleCache:
    """Cross-generation whole-cell bundle cache (exact byte keys)."""

    variant: str = "B"                       # "A" or "B"
    stamp: str = field(default_factory=namespace_stamp)
    store: dict = field(default_factory=dict)  # fp bytes -> np (n_flt,) f32
    stats: dict = field(default_factory=lambda: {
        "hits": 0, "misses": 0, "inserts": 0, "refusals": 0,
    })

    def check_fence(self) -> None:
        if self.stamp != namespace_stamp():
            self.stats["refusals"] += 1
            raise CacheFenceError(
                "bundle cache namespace stamp mismatch (frozen tables / "
                "trans bits / directions changed since insertion)")

    def __len__(self) -> int:
        return len(self.store)

    def approx_bytes(self) -> int:
        """Key bytes + value bytes + dict-entry overhead estimate."""
        if self.variant == "B":
            k, v = FP_B_BYTES, BUNDLE_B_FLOATS * 4
        else:
            k, v = FP_A_BYTES, BUNDLE_A_FLOATS * 4
        return len(self.store) * (k + v + 120)


def _fingerprint_rows(veg: bp.PackedBits, vbsh: bp.PackedBits,
                      vegdem2: np.ndarray, svf_building: np.ndarray,
                      cells: np.ndarray, variant: str) -> np.ndarray:
    """(n_cells, width) uint8 fingerprint matrix, row per cell.

    vegdem2 enters variant B as the ==0.0 predicate BYTE (0 or 1) and
    variant A as its raw f32 bits (little-endian). svf_building enters
    variant A as raw f32 bits.
    """
    n = cells.shape[0]
    vegd = veg.data.reshape(-1, veg.data.shape[-1])[cells[:, 0] * veg.cols
                                                   + cells[:, 1]]
    vbshd = vbsh.data.reshape(-1, vbsh.data.shape[-1])[
        cells[:, 0] * vbsh.cols + cells[:, 1]]
    if variant == "B":
        pred = (vegdem2[cells[:, 0], cells[:, 1]] == np.float32(0.0)
                ).astype(np.uint8)[:, None]
        return np.concatenate([vegd, vbshd, pred], axis=1)
    v2 = vegdem2[cells[:, 0], cells[:, 1]].view(np.uint8).reshape(-1, 4)
    sv = svf_building[cells[:, 0], cells[:, 1]].view(
        np.uint8).reshape(-1, 4)
    return np.concatenate([vegd, vbshd, v2, sv], axis=1)


def _svftotal_recompute(svf_building: np.ndarray, svfveg: np.ndarray
                        ) -> np.ndarray:
    """Bit-exact svftotal outside the kernel: the kernel's scalar f32
    chain ``svf_building - (1.0 - svfveg) * (1.0 - TRANS)`` elementwise
    (numpy f32 elementwise ops are the same IEEE ops; no FMA)."""
    one = np.float32(1.0)
    omt = np.float32(one - sf.TRANS)
    return svf_building - (one - svfveg) * omt


def cached_fold(veg_packed, vbsh_packed, vegdem2, svf_building,
                cache: BundleCache, *,
                cell_mask=None, outputs=None, n_threads: int = 1,
                min_cells: int = sf.SERIAL_MIN_CELLS):
    """fold_svf with §474 whole-cell bundle reuse.

    Cells whose fingerprint bytes are already in ``cache`` are replayed
    (variant B: 10 accumulators + fresh svftotal; variant A: all 11);
    the rest fold through :func:`svf_fold.fold_svf` restricted to the
    miss ``cell_mask`` and are inserted. Unmasked cells are bit-identical
    to ``outputs`` (the base planes), as in the direct fold.

    Returns (SvfFoldResult, timing dict). The timing dict carries the
    C2 shapes: fp build, hit-path (lookup+assemble, the verify-inclusive
    number), miss fold wall, insert cost — raw-vs-verified arithmetic is
    computed from these by the caller.
    """
    if cache.variant not in ("A", "B"):
        raise ValueError(f"unknown cache variant {cache.variant!r}")
    cache.check_fence()
    veg = sf._as_packed(veg_packed, "vegsh_packed")
    vbsh = sf._as_packed(vbsh_packed, "vbsh_packed")
    rows, cols = veg.rows, veg.cols
    v2c = sf._as_plane(vegdem2, "vegdem2", (rows, cols))
    svfc = sf._as_plane(svf_building, "svf_building", (rows, cols))

    if cell_mask is None:
        mask = np.ones((rows, cols), dtype=bool)
    else:
        mask = np.ascontiguousarray(cell_mask).astype(bool, copy=True)
    cells = np.argwhere(mask)
    n_fl = (BUNDLE_B_FLOATS if cache.variant == "B" else BUNDLE_A_FLOATS)

    t0 = time.perf_counter()
    fps = _fingerprint_rows(veg, vbsh, v2c, svfc, cells, cache.variant)
    t_fp = time.perf_counter() - t0

    t0 = time.perf_counter()
    hit_idx, miss_idx = [], []
    got = np.empty((cells.shape[0], n_fl), dtype=np.float32)
    for i in range(cells.shape[0]):
        b = fps[i].tobytes()
        val = cache.store.get(b)
        if val is None:
            miss_idx.append(i)
        else:
            got[i] = val
            hit_idx.append(i)
    t_hit = time.perf_counter() - t0

    t0 = time.perf_counter()
    if miss_idx:
        miss_mask = np.zeros((rows, cols), dtype=bool)
        mc = cells[np.asarray(miss_idx)]
        miss_mask[mc[:, 0], mc[:, 1]] = True
        res = sf.fold_svf(veg, vbsh, v2c, svfc, cell_mask=miss_mask,
                          outputs=outputs, n_threads=n_threads,
                          min_cells=min_cells)
        stacked = np.stack([getattr(res, n) for n in FOLD_PLANES[:n_fl]],
                           axis=-1)[mc[:, 0], mc[:, 1]]
        got[np.asarray(miss_idx)] = stacked
    else:
        res = None
    t_fold = time.perf_counter() - t0

    # assemble output planes: base for untouched, fold for misses (already
    # written by fold_svf via outputs=), cache for hits
    t0 = time.perf_counter()
    if outputs is not None:
        planes = [np.array(outputs[n], dtype=np.float32, order="C")
                  for n in FOLD_PLANES]
    else:
        planes = [np.zeros((rows, cols), np.float32) for _ in FOLD_PLANES]
    if hit_idx:
        hc = cells[np.asarray(hit_idx)]
        for k in range(n_fl):
            planes[k][hc[:, 0], hc[:, 1]] = got[np.asarray(hit_idx), k]
        if cache.variant == "B":
            st = _svftotal_recompute(svfc[hc[:, 0], hc[:, 1]],
                                     planes[0][hc[:, 0], hc[:, 1]])
            planes[10][hc[:, 0], hc[:, 1]] = st
    if miss_idx:
        mc = cells[np.asarray(miss_idx)]
        for k in range(len(FOLD_PLANES)):
            planes[k][mc[:, 0], mc[:, 1]] = getattr(res, FOLD_PLANES[k])[
                mc[:, 0], mc[:, 1]]
    t_asm = time.perf_counter() - t0

    # insert miss bundles (exact fp bytes -> bundle)
    t0 = time.perf_counter()
    if miss_idx:
        for i in miss_idx:
            cache.store[fps[i].tobytes()] = got[i]
        cache.stats["inserts"] += len(miss_idx)
    t_ins = time.perf_counter() - t0

    cache.stats["hits"] += len(hit_idx)
    cache.stats["misses"] += len(miss_idx)

    timing = {
        "n_cells": int(cells.shape[0]),
        "n_hits": len(hit_idx),
        "n_misses": len(miss_idx),
        "fp_build_ms": t_fp * 1e3,
        "hit_lookup_ms": t_hit * 1e3,      # C2 verify-inclusive hit path
        "miss_fold_ms": t_fold * 1e3,
        "assemble_ms": t_asm * 1e3,
        "insert_ms": t_ins * 1e3,
        "variant": cache.variant,
    }

    class _Res:
        pass

    out = _Res()
    for k, n in enumerate(FOLD_PLANES):
        setattr(out, n, planes[k])
    out.stats = {"cached": True, **timing,
                 "route": f"cached-{cache.variant}"}
    return out, timing
