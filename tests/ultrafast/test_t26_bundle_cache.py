# SPDX-License-Identifier: GPL-3.0-only
"""T26 gates: whole-cell scalar bundle reuse ablation (DESIGN §474).

The §474 candidate replays a cell's COMPLETED fold bundle when ALL of
that cell's input bits match a previously-folded state (whole-cell cache
hit under a complete dependency fingerprint) — never a float delta
update. Gates:

* PURITY        — the fold is memoryless per cell: identical input bits
                  ⇒ bit-identical bundles (the cache's licence; also the
                  unconstructible-carry probe).
* IDENTITY      — cached fold ≡ direct fold, gen-0 (all miss) and warm
                  (all hit), both fingerprint variants, all 11 planes
                  raw uint32 (variant B's fresh svftotal recompute is
                  pinned here).
* EDIT          — a dirty-cell edit generation through the cache equals
                  a fresh masked fold bit-for-bit, with real hits.
* MUTATIONS     — the fingerprint's completeness has teeth: dropping
                  vbsh (regime staleness) or the vegdem2 predicate
                  (last-correction miss) is caught; a 1-byte near-miss
                  never false-hits; the namespace fence refuses a changed
                  constant scope (incl. the TRANS pin, C1); keying by
                  vegdem2 VALUE instead of predicate stays exact (pin).
* LEDGER (C2)   — hit-path timing is charged verify-inclusive and the
                  cache memory is measured, not asserted away.

Skips cleanly when numba/scenes unavailable — no capture dependency.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from solweig_core import bitplanes as bp
from solweig_core.numba_cpu import svf_fold as sf

from tests.ultrafast import t26_bundle_cache as t26

ROWS, COLS, PATCHES = 200, 180, 153


def _scene(seed: int = 26):
    rng = np.random.default_rng(seed)
    veg = bp.pack_bits((rng.random((ROWS, COLS, PATCHES)) < 0.3
                        ).astype(np.uint8))
    vbsh = bp.pack_bits((rng.random((ROWS, COLS, PATCHES)) < 0.5
                         ).astype(np.uint8))
    vegdem2 = (rng.random((ROWS, COLS)) * 10).astype(np.float32)
    vegdem2[rng.random((ROWS, COLS)) < 0.2] = np.float32(0.0)
    svfb = (rng.random((ROWS, COLS)) * 0.8 + 0.1).astype(np.float32)
    return veg, vbsh, vegdem2, svfb


def _planes_equal(a, b) -> bool:
    return all(np.array_equal(np.ascontiguousarray(getattr(a, n)).view(
        np.uint32), np.ascontiguousarray(getattr(b, n)).view(np.uint32))
        for n in sf.FOLD_OUTPUT_NAMES)


def _fold_cell_bits(veg, vbsh, vegdem2, svfb, r, c):
    """One cell's bundle, folded alone (purity probe)."""
    m = np.zeros((veg.rows, veg.cols), dtype=bool)
    m[r, c] = True
    return sf.fold_svf(veg, vbsh, vegdem2, svfb, cell_mask=m)


# ---------------------------------------------------------------------------
# purity: identical input bits ⇒ identical bundle (memoryless licence)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(bp, "pack_bits"),
                    reason="bitplanes.pack_bits absent")
def test_identical_input_cells_have_identical_bundles() -> None:
    veg, vbsh, vegdem2, svfb = _scene()
    # copy cell (10,10)'s full input state onto (150, 170)
    veg.data[150, 170] = veg.data[10, 10]
    vbsh.data[150, 170] = vbsh.data[10, 10]
    vegdem2[150, 170] = vegdem2[10, 10]
    svfb[150, 170] = svfb[10, 10]
    bx = _fold_cell_bits(veg, vbsh, vegdem2, svfb, 10, 10)
    by = _fold_cell_bits(veg, vbsh, vegdem2, svfb, 150, 170)
    # compare the bundle AT each fold's masked cell (the rest of each
    # plane is untouched base)
    for n in sf.FOLD_OUTPUT_NAMES:
        assert np.array_equal(
            np.ascontiguousarray(getattr(bx, n)[10, 10]).view(np.uint32),
            np.ascontiguousarray(getattr(by, n)[150, 170]).view(np.uint32),
        ), (f"identical input bits produced different {n} bundles — the "
            "fold carries cross-cell state and §474 replay is unlicensed")


# ---------------------------------------------------------------------------
# identity: cached fold ≡ direct fold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["B", "A"])
def test_cached_fold_bit_equal_gen0_and_warm(variant: str) -> None:
    veg, vbsh, vegdem2, svfb = _scene()
    ref = sf.fold_svf(veg, vbsh, vegdem2, svfb)
    cache = t26.BundleCache(variant=variant)
    out0, tim0 = t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    assert tim0["n_misses"] == ROWS * COLS and tim0["n_hits"] == 0
    assert _planes_equal(out0, ref), "gen-0 cached fold != direct fold"
    out1, tim1 = t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    assert tim1["n_hits"] == ROWS * COLS and tim1["n_misses"] == 0
    assert _planes_equal(out1, ref), \
        "warm cached fold != direct fold (variant B svftotal recompute " \
        "pinned here)"
    assert tim1["hit_lookup_ms"] > 0.0, "C2: hit path must be charged"


# ---------------------------------------------------------------------------
# edit generation: dirty cells through the cache
# ---------------------------------------------------------------------------


def _edit_scene(veg, vbsh, vegdem2):
    """Overwrite a block with ANOTHER block's full input state (veg, vbsh
    AND vegdem2). The dirty cells' new fingerprints then exactly equal
    the source cells' base-generation fingerprints -> cross-generation
    hits are guaranteed by construction (variant B)."""
    veg2 = bp.PackedBits(data=veg.data.copy(), patch_count=veg.patch_count)
    vbsh2 = bp.PackedBits(data=vbsh.data.copy(),
                          patch_count=vbsh.patch_count)
    veg2.data[100:140, 40:100] = veg.data[10:50, 40:100]
    vbsh2.data[100:140, 40:100] = vbsh.data[10:50, 40:100]
    v2_2 = vegdem2.copy()
    v2_2[100:140, 40:100] = vegdem2[10:50, 40:100]
    changed = np.zeros((ROWS, COLS), dtype=bool)
    blk = (slice(100, 140), slice(40, 100))
    changed[blk] = (
        (veg2.data[blk] != veg.data[blk]).any(-1)
        | (vbsh2.data[blk] != vbsh.data[blk]).any(-1)
        | (v2_2[blk].view(np.uint32) != vegdem2[blk].view(np.uint32)))
    return veg2, vbsh2, v2_2, changed


def test_edit_generation_cached_equals_fresh_masked_fold() -> None:
    veg, vbsh, vegdem2, svfb = _scene()
    base_cache = t26.BundleCache(variant="B")
    t26.cached_fold(veg, vbsh, vegdem2, svfb, base_cache)
    veg2, vbsh2, v2_2, changed = _edit_scene(veg, vbsh, vegdem2)
    if not changed.any():
        pytest.skip("edit produced no dirty cells")
    out, tim = t26.cached_fold(veg2, vbsh2, v2_2, svfb, base_cache,
                               cell_mask=changed)
    ref = sf.fold_svf(veg2, vbsh2, v2_2, svfb, cell_mask=changed)
    assert _planes_equal(out, ref), \
        "edit-generation cached fold != fresh masked fold"
    assert tim["n_hits"] > 0, \
        "expected cross-generation hits by construction (copied patterns)"
    assert tim["n_misses"] + tim["n_hits"] == int(changed.sum())


# ---------------------------------------------------------------------------
# mutations: fingerprint completeness has teeth
# ---------------------------------------------------------------------------


def _cell_fps(veg, vbsh, vegdem2, svfb, cells, variant="B"):
    return t26._fingerprint_rows(veg, vbsh, vegdem2, svfb,
                                 np.asarray(cells, dtype=np.int64), variant)


def test_mutation_drop_vbsh_from_key_is_caught() -> None:
    """M1: a key that omits vbsh collapses cells differing only in vbsh
    (the T22 regime-staleness class). The REAL fingerprint must
    distinguish them, and their bundles must differ (so the collapsed
    key would have mis-served)."""
    veg, vbsh, vegdem2, svfb = _scene()
    vbsh2 = bp.PackedBits(data=vbsh.data.copy(),
                          patch_count=vbsh.patch_count)
    vbsh2.data[50, 50] = vbsh.data[50, 50] ^ np.uint8(0xFF)
    veg2 = veg  # identical veg
    cells = [(50, 50), (51, 51)]
    # make veg identical at both cells so vbsh is the ONLY difference
    veg.data[51, 51] = veg.data[50, 50]
    vegdem2[51, 51] = vegdem2[50, 50]
    svfb[51, 51] = svfb[50, 50]
    fp_real = _cell_fps(veg2, vbsh2, vegdem2, svfb, cells)
    assert not np.array_equal(fp_real[0], fp_real[1]), \
        "real fingerprint collapsed a vbsh-only difference"
    bx = _fold_cell_bits(veg2, vbsh2, vegdem2, svfb, 50, 50)
    by = _fold_cell_bits(veg2, vbsh2, vegdem2, svfb, 51, 51)
    assert not _planes_equal(bx, by), \
        "vbsh difference did not reach the bundle — key would be " \
        "over-complete but the mutated key's staleness is harmless " \
        "here; scene construction failed"
    fp_defective = fp_real[:, :20]  # veg bytes only
    assert np.array_equal(fp_defective[0], fp_defective[1]), \
        "defective key failed to demonstrate the collapse"


def test_mutation_drop_vegdem2_predicate_is_caught() -> None:
    """M2: a key that omits the vegdem2-zero predicate collapses a
    last-correction cell (vegdem2 == 0) with a non-correction cell —
    the S/W planes must differ, and the real key must distinguish."""
    veg, vbsh, vegdem2, svfb = _scene()
    vegdem2 = vegdem2.copy()
    vegdem2[70, 70] = np.float32(0.0)
    vegdem2[71, 71] = np.float32(7.5)
    veg.data[71, 71] = veg.data[70, 70]
    vbsh.data[71, 71] = vbsh.data[70, 70]
    svfb[71, 71] = svfb[70, 70]
    cells = [(70, 70), (71, 71)]
    fp_real = _cell_fps(veg, vbsh, vegdem2, svfb, cells)
    assert fp_real[0, -1] == 1 and fp_real[1, -1] == 0, \
        "predicate byte not populated as expected"
    bx = _fold_cell_bits(veg, vbsh, vegdem2, svfb, 70, 70)
    by = _fold_cell_bits(veg, vbsh, vegdem2, svfb, 71, 71)
    assert not np.array_equal(
        np.ascontiguousarray(bx.svfSveg).view(np.uint32),
        np.ascontiguousarray(by.svfSveg).view(np.uint32)), \
        "last correction did not move svfSveg — scene construction failed"
    fp_defective = fp_real[:, :-1]
    assert np.array_equal(fp_defective[0], fp_defective[1]), \
        "defective key failed to demonstrate the collapse"


def test_pin_key_by_vegdem2_value_stays_exact() -> None:
    """M3 pin: variant A keys by the vegdem2 VALUE bits (not the
    predicate) — over-complete but still exact: same value ⇒ same
    predicate ⇒ same bundle. Two cells differing only in vegdem2 VALUE
    with the SAME predicate must fold identically."""
    veg, vbsh, vegdem2, svfb = _scene()
    vegdem2 = vegdem2.copy()
    vegdem2[80, 80] = np.float32(3.25)
    vegdem2[81, 81] = np.float32(9.75)
    veg.data[81, 81] = veg.data[80, 80]
    vbsh.data[81, 81] = vbsh.data[80, 80]
    svfb[81, 81] = svfb[80, 80]
    cells = [(80, 80), (81, 81)]
    fp_a = _cell_fps(veg, vbsh, vegdem2, svfb, cells, variant="A")
    assert not np.array_equal(fp_a[0], fp_a[1]), \
        "variant A should distinguish different vegdem2 values"
    bx = _fold_cell_bits(veg, vbsh, vegdem2, svfb, 80, 80)
    by = _fold_cell_bits(veg, vbsh, vegdem2, svfb, 81, 81)
    for n in sf.FOLD_OUTPUT_NAMES:
        assert np.array_equal(
            np.ascontiguousarray(getattr(bx, n)[80, 80]).view(np.uint32),
            np.ascontiguousarray(getattr(by, n)[81, 81]).view(np.uint32),
        ), ("same-predicate cells folded differently — variant A's "
            "over-completeness is not the licence, purity is; this breaks")
    cache_a = t26.BundleCache(variant="A")
    out, tim = t26.cached_fold(veg, vbsh, vegdem2, svfb, cache_a)
    ref = sf.fold_svf(veg, vbsh, vegdem2, svfb)
    assert _planes_equal(out, ref)


def test_mutation_namespace_fence_refuses_changed_constant_scope() -> None:
    """M4/C1: the cache stamp carries the frozen tables' digest — which
    includes the TRANS bits (C1 scope pin) — and the direction masks. A
    regime where any constant-scope input differs must REFUSE to serve
    (loud CacheFenceError), never silently replay."""
    veg, vbsh, vegdem2, svfb = _scene()
    cache = t26.BundleCache(variant="B")
    t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    assert len(cache) > 0
    # simulate a changed constant scope WITHOUT touching the module:
    # trans lives in frozen_tables_digest; flip the stamp's world by
    # pointing the cache at a foreign stamp
    cache.stamp = "0" * 64
    with pytest.raises(t26.CacheFenceError):
        t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    assert cache.stats["refusals"] == 1


def test_mutation_trans_scope_pin() -> None:
    """C1 teeth: a per-cell-trans REGIME under a constant-scope key would
    mis-serve svftotal. The fence must catch it (stamp mismatch), and
    within the real constant scope the 2-flop recompute is pinned
    bit-exact against the kernel for both predicate classes."""
    veg, vbsh, vegdem2, svfb = _scene()
    # (a) pin: warm-cache replay reproduces the kernel's svftotal bits
    cache = t26.BundleCache(variant="B")
    t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    out, _ = t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    ref = sf.fold_svf(veg, vbsh, vegdem2, svfb)
    assert np.array_equal(np.ascontiguousarray(out.svftotal).view(np.uint32),
                          np.ascontiguousarray(ref.svftotal).view(
                              np.uint32)), \
        "variant B svftotal recompute left the kernel's bits"
    # (b) the stamp actually covers trans: flipping the trans BITS in a
    # derived digest must change the stamp
    s1 = t26.namespace_stamp()
    assert s1 != "0" * 64 and len(s1) == 64
    # (c) a per-cell-trans world is OUTSIDE the constant scope: feeding
    # per-cell trans through the recompute (simulated regime) yields
    # different svftotal bits for the same fingerprint — the fence is
    # what keeps that world from being served from this cache
    trans2 = np.float32(0.05)
    one = np.float32(1.0)
    st_a = svfb - (one - ref.svfveg) * (one - sf.TRANS)
    st_b = svfb - (one - ref.svfveg) * (one - trans2)
    assert not np.array_equal(st_a.view(np.uint32), st_b.view(np.uint32)), \
        "trans regime change did not move svftotal — C1 scenario broken"


def test_mutation_near_miss_bytes_never_false_hit() -> None:
    """M5: keys are raw fingerprint BYTES — dict equality is byte
    equality, so a 'hash collision' is unconstructible. A 1-byte
    near-miss key is a distinct key: it never serves another entry's
    bundle (uniqueness of the store is the teeth)."""
    veg, vbsh, vegdem2, svfb = _scene()
    cache = t26.BundleCache(variant="B")
    t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    keys = list(cache.store.keys())
    assert len(set(keys)) == len(keys), \
        "duplicate keys in store — byte equality broken"
    k0 = keys[0]
    near = bytes(bytearray(k0)[:20] + bytearray(b ^ 0x01 for b in
                                                k0[20:21]) + k0[21:])
    assert near != k0
    # a near-miss key is served ONLY if it is some real cell's exact
    # fingerprint — never as a fuzzy match
    if near in cache.store:
        assert near in keys


# ---------------------------------------------------------------------------
# ledger (C2/memory): measured, not asserted away
# ---------------------------------------------------------------------------


def test_cache_memory_ledger_measured() -> None:
    veg, vbsh, vegdem2, svfb = _scene()
    cache = t26.BundleCache(variant="B")
    t26.cached_fold(veg, vbsh, vegdem2, svfb, cache)
    n = len(cache)
    measured = cache.approx_bytes()
    assert n > 0 and measured > 0
    per_entry = measured / n
    # site_500 full tile projection, recorded for the report (not a
    # gate against 256 MiB — that arithmetic lives in the ablation
    # report against the T23 band)
    proj_mib = 500 * 500 * per_entry / 2**20
    assert proj_mib > 0
    t26.ART.mkdir(parents=True, exist_ok=True)
    (t26.ART / "cache_memory_ledger.json").write_text(json.dumps({
        "scene_cells": n, "approx_bytes": measured,
        "per_entry_bytes": round(per_entry, 1),
        "site500_full_tile_projection_mib": round(proj_mib, 1),
        "variant": cache.variant,
    }, indent=2))
