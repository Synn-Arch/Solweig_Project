# SPDX-License-Identifier: GPL-3.0-only
"""T22 gates: 2-bit extended vbsh encoding + wider regime proof (DESIGN 8.1/§432).

Protocol (hypothesis -> exact equivalence -> counterexample search ->
independent oracle -> measured ablation), one falsifiable hypothesis per
section:

* **H1 DOMAIN (A)** — every march variant emits vegsh in {0, 1} and vbsh in
  {0, 1, 2} as EXACT float32 integers. Sharper: ``vbsh == 2`` iff the march
  is ONE-STEP (table count == 1: the only accumulated step is s == 0, whose
  contribution the first-step exception DISCARDS via ``vbshvegsh.zero_()``)
  and the final vegsh is 1. Hence one-step patches emit vbsh in {1, 2}
  (never 0) and multi-step patches emit vbsh in {0, 1} (never 2): per patch
  vbsh is BINARY — {0,1} or {1,2} — decided by regime alone.
* **H2 EQUIVALENCE (B/C)** — the 2-bit codec round-trips the {0,1,2} domain
  exactly (typed refusal elsewhere), and ``decode(pack(march planes))`` ==
  the march planes bit-for-bit, so the ONE production fold
  (:func:`solweig_gpu.incremental.solver._fold_veg_svf_from_planes`)
  returns bit-identical scalars from f32 planes, 2-bit-unpacked planes and
  regime-remapped 1-bit planes (``value = bit + one_step_bias``). The
  ``if vbsh > zero`` threshold and the vegsh subtraction live INSIDE the
  march final graph (producer side, march.py:161-168); consumers never
  re-threshold, so decode-value equality is the whole obligation.
* **H3 ENGAGEMENT/INVALIDATION (D)** — 2-bit extends the packed-state path
  to one-step baselines (today ``pack_visibility`` refuses vbsh==2.0
  loudly), but DESIGN §432's warning is demonstrated concretely: amplitude
  WIDENING at a one-step patch flips the domain {1,2} -> {0,1} at unchanged
  cells outside every corridor closure, and today's
  ``_regime_transition_reason`` only refuses NARROWING — an extended state
  needs an ANY-crossing fence before it may engage.
* **H4 COST (E)** — naive 2-bit grows the vbsh stack 20 -> 39 bytes/pixel
  at 153 patches; the regime-remapped 1-bit encoding holds the same domain
  at UNCHANGED bytes. Timings are PROVISIONAL dev-window numbers (single
  process, repeats 3) — authority numbers stay with the lead's window.

Counterexample searches (D): corridor value collapse (``!= 0`` packing,
the silent corruption the binary fence prevents), regime-widening
staleness, and a decode-layout mutation. Logical mutations live in-test;
source-level sed mutations are out of T22 scope (codec is additive).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from test_march_dense import (  # noqa: E402
    march_table_for,
    run_original_svf,
    synthetic_scene,
)
from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu.march import march_svf_shadow  # noqa: E402
from solweig_gpu.incremental.bitmask import (  # noqa: E402
    PackedVisibility2Bit,
    bytes_per_pixel,
    bytes_per_pixel_2bit,
    pack_visibility,
    pack_visibility_2bit,
    set_patch_window_2bit,
    spatial_window_view_2bit,
    unpack_patch_2bit,
    unpack_visibility_2bit,
    weighted_sum_from_packed_2bit,
)
from solweig_gpu.incremental.solver import (  # noqa: E402
    _fold_veg_svf_from_planes,
    _sky_patch_geometry,
)
from solweig_gpu.incremental.veg_svf_state import (  # noqa: E402
    _regime_transition_reason,
    one_step_patch_indices,
)

#: Witness geometry: 4 m pixels (scale 0.25) put the 66/78-degree rings in
#: the one-step regime at amplitude 4.0 (dz_1 = tan(alt)/0.25 > 4.0) while
#: the 6/42-degree rings stay multi-step — a MIXED-regime scene. The
#: all-multi arm needs amplitude above the DIAGONAL 78-degree first step
#: (ds ~ sqrt(2) pixels -> dz_1 ~ 26.4 m), hence 60.0 — at 20.0 the az
#: 215.7 patch is still one-step and emits vbsh == 2.0.
SCALE = 0.25
AMP_ONE_STEP_MIXED = 4.0
AMP_MULTI_STEP = 60.0
ROWS, COLS = 40, 80
PATCH_COUNT = 153


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sky_patches() -> tuple:
    patches, _rings = _sky_patch_geometry(2)
    assert len(patches) == PATCH_COUNT
    return patches


def _scene_np(seed: int = 13):
    a, vegdem, vegdem2, bush = synthetic_scene(ROWS, COLS, seed=seed)
    return (
        a.numpy().copy(),
        vegdem.numpy().copy(),
        vegdem2.numpy().copy(),
        bush.numpy().copy(),
    )


_SWEEP_CACHE: dict[tuple, dict] = {}


def _sweep(seed: int, amp: float) -> dict:
    """March all 153 patches at ``amp`` (memoized) — cubes + table counts."""
    key = (seed, amp)
    cached = _SWEEP_CACHE.get(key)
    if cached is not None:
        return cached
    a, vegdem, vegdem2, bush = _scene_np(seed)
    patches = _sky_patches()
    sh_planes, veg_planes, vb_planes, counts = [], [], [], []
    for altitude, azimuth, _ring in patches:
        table, _trace = march_table_for(
            st.KERNEL_SVF_SHADOW,
            float(azimuth),
            float(altitude),
            SCALE,
            ROWS,
            COLS,
            amp,
        )
        sh, vegsh, vbsh = march_svf_shadow(table, a, vegdem, vegdem2, bush)
        sh_planes.append(sh)
        veg_planes.append(vegsh)
        vb_planes.append(vbsh)
        counts.append(int(table.count))
    cached = {
        "vegsh": np.stack(veg_planes, axis=-1),
        "vbsh": np.stack(vb_planes, axis=-1),
        "sh": np.stack(sh_planes, axis=-1),
        "counts": counts,
        "patches": patches,
        "scene": (a, vegdem, vegdem2, bush),
    }
    _SWEEP_CACHE[key] = cached
    return cached


def _fold(vegsh_cube: np.ndarray, vbsh_cube: np.ndarray, vegdem2: np.ndarray):
    """The production fold, straight from f32 (or decoded) planes."""
    return _fold_veg_svf_from_planes(
        vegshmat=torch.from_numpy(np.ascontiguousarray(vegsh_cube)),
        vbshvegshmat=torch.from_numpy(np.ascontiguousarray(vbsh_cube)),
        vegdem2_read=torch.from_numpy(vegdem2),
        svf_building_window=torch.full(
            (vegsh_cube.shape[0], vegsh_cube.shape[1]), 0.7
        ),
        acc_r0=0,
        acc_r1=vegsh_cube.shape[0],
        acc_c0=0,
        acc_c1=vegsh_cube.shape[1],
        rows=vegsh_cube.shape[0],
        cols=vegsh_cube.shape[1],
    )


def _fold_outputs_equal(first, second) -> bool:
    """Bit equality on every scalar tensor + svftotal of two fold calls."""
    scalars_a, _veg_a, _vb_a, svftotal_a = first
    scalars_b, _veg_b, _vb_b, svftotal_b = second
    if set(scalars_a) != set(scalars_b):
        return False
    if not torch.equal(svftotal_a, svftotal_b):
        return False
    return all(
        torch.equal(scalars_a[name], scalars_b[name]) for name in scalars_a
    )


def _decode_2bit_f32(packed: PackedVisibility2Bit) -> np.ndarray:
    """The state-path decode: 2-bit values -> exact f32 planes."""
    return unpack_visibility_2bit(packed).astype(np.float32)


def _regime_bias(patch_index: int, one_step: set[int]) -> int:
    """H1 corollary: one-step patches hold vbsh in {1,2} -> store vbsh-1."""
    return 1 if patch_index in one_step else 0


def _pack_remapped_1bit(vbsh_cube: np.ndarray, one_step: set[int]):
    """Regime-remapped 1-bit encoding: bias out, pack binary, remember bias.

    Not a production codec — the T22 ablation arm that holds the {0,1,2}
    domain at UNCHANGED 1-bit bytes (per-patch vbsh is binary by H1).
    """
    values = vbsh_cube.astype(np.int8)
    for p in one_step:
        values[:, :, p] -= 1
    assert np.all((values == 0) | (values == 1)), "remap domain broken"
    return pack_visibility(values.astype(bool))


def _decode_remapped_1bit(packed, one_step: set[int]) -> np.ndarray:
    from solweig_gpu.incremental.bitmask import unpack_visibility

    bits = unpack_visibility(packed)  # (rows, cols, 153) bool
    values = bits.astype(np.int8)
    bias = np.array(
        [_regime_bias(p, one_step) for p in range(PATCH_COUNT)], dtype=np.int8
    )
    return (values + bias).astype(np.float32)


# ---------------------------------------------------------------------------
# A. H1 domain gates
# ---------------------------------------------------------------------------


class TestADomain:
    def test_one_step_witness_domain_is_12(self):
        """count == 1 patches emit vbsh in {1, 2} — never 0, never beyond 2."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        one_step_tables = [
            p for p, count in zip(sweep["patches"], sweep["counts"])
            if count == 1 and float(p[0]) < 90.0
        ]
        assert one_step_tables, "fixture has no one-step patch (too weak)"
        checked = 0
        for index, (altitude, azimuth, _ring) in enumerate(sweep["patches"]):
            if sweep["counts"][index] != 1 or float(altitude) >= 90.0:
                continue
            vbsh = sweep["vbsh"][:, :, index]
            vegsh = sweep["vegsh"][:, :, index]
            assert np.all((vbsh == 1.0) | (vbsh == 2.0)), (
                f"one-step patch {index} (alt {altitude}) emitted vbsh "
                f"outside {{1, 2}}: {np.unique(vbsh)}"
            )
            assert np.all((vegsh == 0.0) | (vegsh == 1.0))
            checked += 1
        assert checked == len(one_step_tables)

    def test_multi_step_domain_is_01(self):
        """count >= 2 patches emit vbsh in {0, 1} — 2.0 is impossible."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        checked = 0
        for index, count in enumerate(sweep["counts"]):
            if count < 2:
                continue
            vbsh = sweep["vbsh"][:, :, index]
            assert np.all((vbsh == 0.0) | (vbsh == 1.0)), (
                f"multi-step patch {index} (count {count}) emitted vbsh "
                f"outside {{0, 1}}: {np.unique(vbsh)}"
            )
            checked += 1
        assert checked > 0

    def test_all_multi_amplitude_never_emits_2(self):
        sweep = _sweep(13, AMP_MULTI_STEP)
        assert all(c >= 2 for c in sweep["counts"])
        assert not np.any(sweep["vbsh"] == 2.0)
        assert np.all((sweep["vbsh"] == 0.0) | (sweep["vbsh"] == 1.0))

    def test_fence_predicate_matches_executed_tables(self):
        """one_step_patch_indices (the fence's regime test) agrees with the
        ACTUAL executed step count for every non-zenith patch."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        fence_set = set(
            one_step_patch_indices(SCALE, AMP_ONE_STEP_MIXED)
        )
        for index, (altitude, _azimuth, _ring) in enumerate(sweep["patches"]):
            if float(altitude) >= 90.0:
                continue  # zenith: fence sentinel -1 (never one-step)
            table_one_step = sweep["counts"][index] == 1
            assert table_one_step == (index in fence_set), (
                f"patch {index} (alt {altitude}): table count "
                f"{sweep['counts'][index]} vs fence membership "
                f"{index in fence_set}"
            )

    def test_zenith_patch_domain_recorded(self):
        """Zenith (fence sentinel) still satisfies the global {0,1,2} domain."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        for index, (altitude, _az, _ring) in enumerate(sweep["patches"]):
            if float(altitude) >= 90.0:
                vbsh = sweep["vbsh"][:, :, index]
                assert np.all(
                    (vbsh == 0.0) | (vbsh == 1.0) | (vbsh == 2.0)
                ), f"zenith patch {index} domain: {np.unique(vbsh)}"

    def test_wallheight_variant_never_emits_2(self):
        """wallheight_23 has no first-step vbsh reset, so the 2.0 path is
        svf_shadow-specific: its vbsh stays in {0,1} even one-step."""
        from solweig_core.numba_cpu.march import march_wallheight23
        from test_march_dense import run_original_wallheight

        a, vegdem, vegdem2, bush = synthetic_scene(ROWS, COLS, seed=13)
        for alt, amp in ((78.0, AMP_ONE_STEP_MIXED), (78.0, AMP_MULTI_STEP)):
            table, _trace = march_table_for(
                st.KERNEL_WALLHEIGHT_23, 0.0, alt, SCALE, ROWS, COLS, amp
            )
            _vegsh, _sh, vbsh = march_wallheight23(
                table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
            )
            want_v, want_s, want_vb = run_original_wallheight(
                0.0, alt, SCALE, amp, a, vegdem, vegdem2, bush
            )
            assert np.array_equal(
                np.ascontiguousarray(vbsh), want_vb.numpy()
            ), f"torch oracle disagrees at alt {alt} amp {amp}"
            assert np.all((vbsh == 0.0) | (vbsh == 1.0)), (
                f"wallheight vbsh escaped {{0,1}} at alt {alt} amp {amp}: "
                f"{np.unique(vbsh)}"
            )

    def test_sparse_variant_one_step_domain(self):
        """The sparse runner reproduces the dense one-step domain (the
        runner has no packed-state fence; T-sparse pins the bits)."""
        from solweig_core import sparse_work as sw
        from solweig_core.numba_cpu.sparse_march import march_svf_shadow_sparse

        a, vegdem, vegdem2, bush = _scene_np(13)
        table, _trace = march_table_for(
            st.KERNEL_SVF_SHADOW, 0.0, 78.0, SCALE, ROWS, COLS,
            AMP_ONE_STEP_MIXED,
        )
        assert table.count == 1
        runs = sw.row_runs_from_mask(np.ones((ROWS, COLS), dtype=bool))
        sparse = march_svf_shadow_sparse(
            table, a, vegdem, vegdem2, bush, runs
        )
        vbsh = sparse.vbshvegsh
        assert np.all((vbsh == 1.0) | (vbsh == 2.0)), (
            f"sparse one-step vbsh domain: {np.unique(vbsh)}"
        )

    def test_nan_scene_domain_still_integer(self):
        """NaN in the a-plane never breaks the integer output domain (NaN
        only threatens the suffix-exit theorem, not the value domain)."""
        a, vegdem, vegdem2, bush = _scene_np(13)
        a[5, 7] = np.nan
        a[20, 40] = np.nan
        clean = _scene_np(13)
        table, _trace = march_table_for(
            st.KERNEL_SVF_SHADOW, 133.0, 42.0, SCALE, ROWS, COLS,
            AMP_MULTI_STEP,
        )
        _sh, _vegsh, vbsh = march_svf_shadow(table, a, vegdem, vegdem2, bush)
        assert np.all(
            (vbsh == 0.0) | (vbsh == 1.0) | (vbsh == 2.0)
        ), f"NaN scene vbsh domain: {np.unique(vbsh)}"
        want = run_original_svf(
            133.0, 42.0, SCALE, AMP_MULTI_STEP,
            torch.from_numpy(a),
            torch.from_numpy(vegdem),
            torch.from_numpy(vegdem2),
            torch.from_numpy(bush),
        )
        assert np.array_equal(
            np.ascontiguousarray(vbsh), want[2].numpy()
        ), "torch oracle disagrees on the NaN scene"
        assert not np.array_equal(a, clean[0])


# ---------------------------------------------------------------------------
# B. H2 codec gates
# ---------------------------------------------------------------------------


class TestBCodec:
    def test_roundtrip_and_bytes(self):
        rng = np.random.default_rng(22)
        values = rng.integers(0, 3, (7, 9, PATCH_COUNT), dtype=np.uint8)
        packed = pack_visibility_2bit(values)
        assert isinstance(packed, PackedVisibility2Bit)
        assert packed.data.shape == (7, 9, bytes_per_pixel_2bit(PATCH_COUNT))
        assert packed.nbytes == 7 * 9 * bytes_per_pixel_2bit(PATCH_COUNT)
        assert np.array_equal(unpack_visibility_2bit(packed), values)

    def test_bytes_per_pixel_table(self):
        assert bytes_per_pixel_2bit(1) == 1
        assert bytes_per_pixel_2bit(4) == 1
        assert bytes_per_pixel_2bit(5) == 2
        assert bytes_per_pixel_2bit(PATCH_COUNT) == 39  # ceil(306/8)
        assert bytes_per_pixel(PATCH_COUNT) == 20  # the 1-bit baseline

    def test_typed_refusal_outside_ternary(self):
        rng = np.random.default_rng(3)
        base = rng.integers(0, 3, (4, 4, 8), dtype=np.uint8)
        for bad in (3, 7, 255):
            values = base.copy()
            values[1, 1, 2] = bad
            with pytest.raises(ValueError, match="ternary"):
                pack_visibility_2bit(values)
        for bad in (-1, 0.5, float("nan")):
            values = base.astype(np.float32)
            values[1, 1, 2] = bad
            with pytest.raises(ValueError, match="ternary"):
                pack_visibility_2bit(values)

    def test_constructor_validation(self):
        with pytest.raises(TypeError):
            PackedVisibility2Bit(
                data=np.zeros((2, 2, 39), dtype=np.float32), patch_count=153
            )
        with pytest.raises(ValueError):
            PackedVisibility2Bit(
                data=np.zeros((2, 2, 20), dtype=np.uint8), patch_count=153
            )

    def test_layout_pinned_by_hand(self):
        """Freeze the byte layout: patches p0..p3 at shifts 0/2/4/6."""
        values = np.array([[[0, 1, 2, 0, 2, 1]]], dtype=np.uint8)
        packed = pack_visibility_2bit(values)
        assert packed.data.shape == (1, 1, 2)
        assert packed.data[0, 0, 0] == np.uint8(0 | 1 << 2 | 2 << 4 | 0 << 6)
        assert packed.data[0, 0, 1] == np.uint8(2 | 1 << 2)

    def test_set_patch_window_matches_repack(self):
        rng = np.random.default_rng(4)
        values = rng.integers(0, 3, (6, 6, 12), dtype=np.uint8)
        packed = pack_visibility_2bit(values)
        new_values = rng.integers(0, 3, (2, 3), dtype=np.uint8)
        set_patch_window_2bit(
            packed,
            patch_index=5,
            row_slice=slice(1, 3),
            col_slice=slice(2, 5),
            values=new_values,
        )
        expected = values.copy()
        expected[1:3, 2:5, 5] = new_values
        assert np.array_equal(unpack_visibility_2bit(packed), expected)

    def test_patch_and_window_views(self):
        rng = np.random.default_rng(5)
        values = rng.integers(0, 3, (5, 5, PATCH_COUNT), dtype=np.uint8)
        packed = pack_visibility_2bit(values)
        for index in (0, 1, 62, 152):
            assert np.array_equal(
                unpack_patch_2bit(packed, index), values[:, :, index]
            )
        view = spatial_window_view_2bit(packed, slice(1, 4), slice(2, 5))
        assert np.array_equal(
            unpack_visibility_2bit(view), values[1:4, 2:5, :]
        )
        with pytest.raises(IndexError):
            unpack_patch_2bit(packed, PATCH_COUNT)

    def test_weighted_sum_exact_order(self):
        rng = np.random.default_rng(6)
        values = rng.integers(0, 3, (3, 4, 16), dtype=np.uint8)
        weights = rng.random(16).astype(np.float32)
        packed = pack_visibility_2bit(values)
        got = weighted_sum_from_packed_2bit(packed, weights)
        reference = np.zeros((3, 4), dtype=np.float32)
        for p, weight in enumerate(weights):
            reference += values[:, :, p].astype(np.float32) * weight
        assert np.array_equal(got, reference)


# ---------------------------------------------------------------------------
# C. H2 fold equivalence — the oracle differential
# ---------------------------------------------------------------------------


class TestCFoldEquivalence:
    def test_mixed_regime_scene(self):
        """The engagement gap + full decode equivalence, one scene."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        vegsh_cube = sweep["vegsh"]
        vbsh_cube = sweep["vbsh"]
        one_step = set(one_step_patch_indices(SCALE, AMP_ONE_STEP_MIXED))
        assert one_step, "fixture lost its one-step patches"
        assert np.any(vbsh_cube == 2.0), "fixture produced no vbsh==2.0 cell"

        # Today's 1-bit codec refuses the 2.0 domain loudly (engagement gap).
        with pytest.raises(ValueError, match="binary"):
            pack_visibility(vbsh_cube)

        # 2-bit: pack, decode, fold — bit-identical to the f32 reference.
        packed_2bit = pack_visibility_2bit(vbsh_cube)
        decoded = _decode_2bit_f32(packed_2bit)
        assert np.array_equal(decoded, vbsh_cube)
        reference = _fold(vegsh_cube, vbsh_cube, sweep["scene"][2])
        from_2bit = _fold(vegsh_cube, decoded, sweep["scene"][2])
        assert _fold_outputs_equal(reference, from_2bit)

        # Regime-remapped 1-bit: same domain at unchanged bytes.
        packed_remap = _pack_remapped_1bit(vbsh_cube, one_step)
        decoded_remap = _decode_remapped_1bit(packed_remap, one_step)
        assert np.array_equal(decoded_remap, vbsh_cube)
        from_remap = _fold(vegsh_cube, decoded_remap, sweep["scene"][2])
        assert _fold_outputs_equal(reference, from_remap)

    def test_binary_scene_all_three_encodings(self):
        """All-multi-step scene: 1-bit, 2-bit and remap agree bit-for-bit —
        the 2-bit codec never disturbs the existing binary domain."""
        from solweig_gpu.incremental.bitmask import unpack_visibility

        sweep = _sweep(13, AMP_MULTI_STEP)
        vegsh_cube = sweep["vegsh"]
        vbsh_cube = sweep["vbsh"]
        reference = _fold(vegsh_cube, vbsh_cube, sweep["scene"][2])

        packed_1bit = pack_visibility(vbsh_cube)
        from_1bit = _fold(
            vegsh_cube,
            unpack_visibility(packed_1bit).astype(np.float32),
            sweep["scene"][2],
        )
        assert _fold_outputs_equal(reference, from_1bit)

        packed_2bit = pack_visibility_2bit(vbsh_cube)
        from_2bit = _fold(
            vegsh_cube, _decode_2bit_f32(packed_2bit), sweep["scene"][2]
        )
        assert _fold_outputs_equal(reference, from_2bit)

        empty_one_step: set[int] = set()
        packed_remap = _pack_remapped_1bit(vbsh_cube, empty_one_step)
        from_remap = _fold(
            vegsh_cube,
            _decode_remapped_1bit(packed_remap, empty_one_step),
            sweep["scene"][2],
        )
        assert _fold_outputs_equal(reference, from_remap)

    def test_torch_oracle_sampled_patches(self):
        """Independence from a numba-only bug: sampled patches (including a
        one-step patch with 2.0 cells) match the torch oracle bit-for-bit."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        a, vegdem, vegdem2, bush = sweep["scene"]
        one_step = [
            i for i, count in enumerate(sweep["counts"]) if count == 1
        ]
        sampled = {0, 40, 80, 120, 152} | {one_step[0], one_step[-1]}
        a_t = torch.from_numpy(a)
        vegdem_t = torch.from_numpy(vegdem)
        vegdem2_t = torch.from_numpy(vegdem2)
        bush_t = torch.from_numpy(bush)
        for index in sorted(sampled):
            altitude, azimuth, _ring = sweep["patches"][index]
            _sh, vegsh_w, vbsh_w = run_original_svf(
                float(azimuth), float(altitude), SCALE, AMP_ONE_STEP_MIXED,
                a_t, vegdem_t, vegdem2_t, bush_t,
            )
            assert np.array_equal(
                vegsh_w.numpy(), sweep["vegsh"][:, :, index]
            ), f"vegsh patch {index}"
            assert np.array_equal(
                vbsh_w.numpy(), sweep["vbsh"][:, :, index]
            ), f"vbsh patch {index}"


# ---------------------------------------------------------------------------
# D. H3 counterexample gates
# ---------------------------------------------------------------------------


class TestDCounterexamples:
    def test_corridor_value_collapse_diverges(self):
        """The corridor's ``!= 0`` bool coercion (veg_svf_state.py:783)
        collapses 2.0 -> 1 silently; the fold DIVERGES from the reference.
        This is the silent corruption the binary fence prevents and the
        reason a 2-bit lane must convert VALUES, never booleans."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        vbsh_cube = sweep["vbsh"]
        assert np.any(vbsh_cube == 2.0)
        collapsed = (vbsh_cube != 0).astype(np.float32)
        reference = _fold(sweep["vegsh"], vbsh_cube, sweep["scene"][2])
        corrupted = _fold(sweep["vegsh"], collapsed, sweep["scene"][2])
        assert not _fold_outputs_equal(reference, corrupted)

    def test_regime_widening_stales_unchanged_cells(self):
        """Amplitude WIDENING (one-step -> multi at a patch) flips vbsh at
        UNCHANGED cells (2 -> 1 / 1 -> 0): an extended 2-bit state packed at
        the old amplitude diverges from the new-amplitude oracle, and
        today's narrowing-only fence stays SILENT (the H3 gap)."""
        low = _sweep(13, AMP_ONE_STEP_MIXED)
        high = _sweep(13, AMP_MULTI_STEP)
        one_step = [
            i for i, count in enumerate(low["counts"]) if count == 1
        ]
        assert one_step
        flipped = False
        for index in one_step:
            old = low["vbsh"][:, :, index]
            new = high["vbsh"][:, :, index]
            if not np.array_equal(old, new):
                flipped = True
                break
        assert flipped, "widening changed no unchanged-cell value (weak)"

        # Today's fence: widening (amp 4 -> 20) is allowed — returns None.
        assert (
            _regime_transition_reason(
                AMP_ONE_STEP_MIXED, AMP_MULTI_STEP, SCALE
            )
            is None
        )

        # The stale extended state folds differently from the new oracle.
        stale_packed = pack_visibility_2bit(low["vbsh"])
        stale = _decode_2bit_f32(stale_packed)
        oracle_new = _fold(high["vegsh"], high["vbsh"], high["scene"][2])
        stale_fold = _fold(high["vegsh"], stale, high["scene"][2])
        assert not _fold_outputs_equal(oracle_new, stale_fold)

    def test_decode_layout_mutation_breaks_roundtrip(self):
        """A shifted decode (the layout pin made load-bearing): mutating the
        bit shift corrupts the round-trip and the fold equality."""
        sweep = _sweep(13, AMP_ONE_STEP_MIXED)
        vbsh_cube = sweep["vbsh"]
        packed = pack_visibility_2bit(vbsh_cube)
        grouped = packed.data[..., :, np.newaxis] >> np.array(
            [1, 3, 5, 7], dtype=np.uint8
        )
        misdecoded = (grouped & np.uint8(3)).reshape(
            packed.data.shape[:-1] + (-1,)
        )[..., : PATCH_COUNT]
        assert not np.array_equal(misdecoded, vbsh_cube)


# ---------------------------------------------------------------------------
# E. H4 cost ablation (PROVISIONAL dev-window numbers)
# ---------------------------------------------------------------------------


class TestECostAblation:
    def test_state_bytes_arithmetic(self):
        """Byte cost is arithmetic, not timing: vbsh stack 20 -> 39 B/px."""
        veg_1bit = bytes_per_pixel(PATCH_COUNT)
        vb_1bit = bytes_per_pixel(PATCH_COUNT)
        vb_2bit = bytes_per_pixel_2bit(PATCH_COUNT)
        assert (veg_1bit, vb_1bit) == (20, 20)
        assert vb_2bit == 39
        # Per 500x500 state: 1-bit total 20 MiB, 2-bit total 28.13 MiB,
        # remapped 1-bit total 20 MiB (unchanged); f32 planes 292.97 MiB.
        pixels = 500 * 500
        print(
            f"\n[T22 bytes] pixels={pixels} "
            f"1bit_state={(veg_1bit + vb_1bit) * pixels / 2**20:.2f}MiB "
            f"2bit_state={(veg_1bit + vb_2bit) * pixels / 2**20:.2f}MiB "
            f"remap_state={(veg_1bit + vb_1bit) * pixels / 2**20:.2f}MiB "
            f"f32_planes={2 * 153 * 4 * pixels / 2**20:.2f}MiB"
        )

    def test_codec_and_fold_timings_provisional(self):
        """PROVISIONAL (single process, warmup 1, repeats 3): pack / decode
        / fold timings at the 500x500x153 production shape. No assert on
        ratios — magnitude evidence for the T29 verdict row only."""
        rng = np.random.default_rng(7)
        rows = cols = 500
        vbsh = rng.integers(0, 3, (rows, cols, PATCH_COUNT), dtype=np.uint8)
        vegsh = rng.integers(0, 2, (rows, cols, PATCH_COUNT), dtype=np.uint8)
        from solweig_gpu.incremental.bitmask import unpack_visibility

        one_step = set(one_step_patch_indices(SCALE, AMP_ONE_STEP_MIXED))
        remap_values = vbsh.astype(np.int8)
        for p in one_step:
            remap_values[:, :, p] -= 1

        def timed(label, fn, repeats: int = 3):
            fn()  # warmup
            best = float("inf")
            for _ in range(repeats):
                started = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - started)
            print(f"[T22 timing] {label}: {best * 1000:.1f} ms")
            return best

        packed_1bit_cache = pack_visibility(vegsh.astype(bool))
        packed_2bit_cache = pack_visibility_2bit(vbsh)

        t_pack_1bit = timed(
            "pack 1-bit bool (binary-shaped)",
            lambda: pack_visibility(vegsh),
        )
        t_pack_2bit = timed(
            "pack 2-bit ternary (vbsh)", lambda: pack_visibility_2bit(vbsh)
        )
        t_pack_remap = timed(
            "pack remapped 1-bit (bias subtract + pack)",
            lambda: pack_visibility(remap_values.astype(bool)),
        )
        t_decode_1bit = timed(
            "decode 1-bit -> f32 cube",
            lambda: unpack_visibility(packed_1bit_cache).astype(np.float32),
        )
        t_decode_2bit = timed(
            "decode 2-bit -> f32 cube",
            lambda: unpack_visibility_2bit(packed_2bit_cache).astype(
                np.float32
            ),
        )

        # Fold cost on IDENTICAL f32 cubes: independent of the source
        # encoding by construction (the fold never sees the codec).
        veg_f32 = vegsh.astype(np.float32)
        vb_f32 = vbsh.astype(np.float32)
        vegdem2 = np.zeros((rows, cols), dtype=np.float32)
        t_fold = timed(
            "fold _fold_veg_svf_from_planes (500x500x153)",
            lambda: _fold(veg_f32, vb_f32, vegdem2),
            repeats=1,
        )
        print(
            f"[T22 timing] ratios: pack2/pack1="
            f"{t_pack_2bit / t_pack_1bit:.2f} "
            f"pack_remap/pack1={t_pack_remap / t_pack_1bit:.2f} "
            f"decode2/decode1={t_decode_2bit / t_decode_1bit:.2f} "
            f"fold_vs_decode2={t_fold / t_decode_2bit:.1f}x"
        )
