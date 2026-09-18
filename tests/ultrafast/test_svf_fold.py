# SPDX-License-Identifier: GPL-3.0-only
"""T07 gates: canonical SVF fold over the packed bit state (TASKS T07).

Fold REFERENCE is the ORIGINAL ``solweig_gpu.shadow.svf_calculator`` —
these tests call the oracle and compare RAW float32 bits through
``compare_plane_bits`` (signed zero + NaN payload; no allclose, no ULP
tolerance). The candidate consumes PACKED input directly
(:mod:`solweig_core.bitplanes`) and reproduces the original
patch-major/annulus-minor recurrence EXACTLY:

* per patch p (ring-major, azimuth-minor, the original enumeration order),
  per annulus a: ``svfveg += w_iso * vegsh_p``, ``svfaveg += w_iso *
  vbsh_p``, then the E/S/W/N aniso pairs — same operation order, no
  pre-summed annulus weights (M4), no reordered patch reduction (M5).
* frozen constants: annulus weights / azimuths are CAPTURED uint32 bit
  patterns of the original torch computation (DESIGN 6.3 — torch-vs-numpy
  transcendental 1-ULP hazards are never recomputed); ``last`` = 3.0459e-004
  applied where ``vegdem2 == 0.0`` (-0.0 included, NaN excluded), the
  building-only ``svfS/svfW`` tail constants, ``trans = 0.03``, and
  ``SVFtotal = svf - (1 - svfveg) * (1 - trans)`` (M6).

Gates: (1) oracle raw equality — synthetic vegetated/unvegetated/NaN/±0
scenes + real site_500; (2) downstream one hop (``svfbuveg`` exactly as
``solweig_gpu/utci_process.py:867`` computes it); (4) affected-chunk-only
output (masked cells folded, everything else bit-identical to the caller's
base planes); threading 1/2/4/8 raw equality; one-step-regime packed
refusal (typed, never bool-collapsed); purity (torch-free module).

Downstream coverage is deliberately NARROW: the svfbuveg line only. The
rest of the UTCI graph (asvf, diffsh, Tmrt) is T08 scope.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from bitwise_harness import compare_plane_bits  # noqa: E402
from solweig_core import bitplanes as bp  # noqa: E402
from solweig_core import sparse_work as sw  # noqa: E402
from solweig_core.numba_cpu import svf_fold  # noqa: E402

torch = pytest.importorskip("torch")

from solweig_gpu.shadow import (  # noqa: E402
    annulus_weight,
    create_patches,
    svf_calculator,
)

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
F32 = np.float32
P = 153

#: Named outputs of the fold, in the result-NamedTuple order.
FOLD_OUTPUTS = (
    "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
    "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
    "svftotal",
)


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


# ---------------------------------------------------------------------------
# Scene builders + oracle call (patterns validated in artifacts/t07/proto)
# ---------------------------------------------------------------------------


def synth_scene(rows, cols, seed, canopy_kind="veg"):
    rng = np.random.default_rng(seed)
    dem = rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
    a = dem + rng.uniform(0.0, 4.0, (rows, cols)).astype(np.float32)
    a[4:10, 6:16] += np.float32(8.0)
    canopy = np.zeros((rows, cols), dtype=np.float32)
    if canopy_kind == "veg":
        canopy[8:20, 10:30] = np.float32(9.0)
        canopy[30:38, 40:60] = np.float32(14.0)
    return sw.compose_march_inputs(a, dem, canopy)


def nan_scene(rows, cols, seed):
    """NaN in the building raster; -0.0 (last applies) and NaN (last does
    NOT apply) in vegdem2."""
    inputs = synth_scene(rows, cols, seed, "veg")
    inputs.a[5, 5] = np.float32("nan")
    inputs.vegdsm2[2, 3] = np.float32(-0.0)
    inputs.vegdsm2[9, 70] = np.float32(-0.0)
    inputs.vegdsm2[1, 1] = np.float32("nan")
    return inputs


def oracle_svf(inputs, scale=0.5):
    """One ORIGINAL svf_calculator call, numpy outputs."""
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    ret = svf_calculator(
        patch_option=2,
        amaxvalue=torch.tensor(np.float32(inputs.amaxvalue)),
        a=t(inputs.a),
        vegdem=t(inputs.vegdsm),
        vegdem2=t(inputs.vegdsm2),
        bush=t(inputs.bush),
        scale=float(scale),
    )
    names = (
        "svf", "svfaveg", "svfE", "svfEaveg", "svfEveg", "svfN",
        "svfNaveg", "svfNveg", "svfS", "svfSaveg", "svfSveg", "svfveg",
        "svfW", "svfWaveg", "svfWveg", "vegshmat", "vbshvegshmat",
        "shmat", "SVFtotal",
    )
    return dict(zip(names, [x.numpy() for x in ret]))


def nonbinary_count(vegsh, vbsh):
    bad = ~((vegsh == 0) | (vegsh == 1)) | ~((vbsh == 0) | (vbsh == 1))
    return int(bad.sum())


def fold_from_oracle(oracle, inputs, **kwargs):
    packed_veg = bp.pack_bits(oracle["vegshmat"])
    packed_vbsh = bp.pack_bits(oracle["vbshvegshmat"])
    return svf_fold.fold_svf(
        packed_veg, packed_vbsh, inputs.vegdsm2, oracle["svf"], **kwargs
    )


def assert_fold_matches_oracle(result, oracle, label):
    """Gate-1 comparison: all 11 fold outputs vs the original, raw bits."""
    mismatch_total = 0
    for name in FOLD_OUTPUTS:
        ref = oracle["SVFtotal"] if name == "svftotal" else oracle[name]
        report = compare_plane_bits(ref, getattr(result, name))
        assert report["status"] == "PASS", (
            f"{label}/{name}: {report['status']} "
            f"({report.get('mismatch_count')} mismatches, "
            f"{report.get('examples')})"
        )
        mismatch_total += int(report["mismatch_count"])
    assert mismatch_total == 0, f"{label}: {mismatch_total} raw-bit mismatches"
    return mismatch_total


# ---------------------------------------------------------------------------
# Frozen constants pin vs LIVE original computation (M6 designated killer)
# ---------------------------------------------------------------------------


class TestFrozenConstantsPin:
    """The module's frozen tables must be the ORIGINAL torch bit patterns —
    recomputed LIVE here, never trusted from the module."""

    def _live_tables(self):
        device = torch.device("cpu")
        (
            _skyvaultalt, _skyvaultazi, annulino, skyvaultaltint,
            aziinterval, _skyvaultaziint, azistart,
        ) = create_patches(2)
        skyvaultaziint = torch.tensor(
            [360 / patches for patches in aziinterval], device=device
        )
        iazimuth = torch.zeros((1, int(torch.sum(aziinterval).item())), device=device)
        index = 0
        for j in range(skyvaultaltint.shape[0]):
            for k in range(int(360 / skyvaultaziint[j])):
                iazimuth[0, index] = k * skyvaultaziint[j] + azistart[j]
                if iazimuth[0, index] > 360.0:
                    iazimuth[0, index] = iazimuth[0, index] - 360.0
                index += 1
        aziintervalaniso = torch.ceil(aziinterval / 2.0)
        w_iso = np.zeros((8, 12), dtype=np.float32)
        w_aniso = np.zeros((8, 12), dtype=np.float32)
        az = np.zeros(P, dtype=np.float32)
        ring = np.zeros(P, dtype=np.int32)
        index = 0
        for i in range(8):
            slot = 0
            for k in range(int(annulino[i]) + 1, int(annulino[i + 1]) + 1):
                w_iso[i, slot] = annulus_weight(k, aziinterval[i], device).item()
                w_aniso[i, slot] = annulus_weight(
                    k, aziintervalaniso[i], device
                ).item()
                slot += 1
            for _j in range(int(aziinterval[i].int())):
                az[index] = iazimuth[0, index].item()
                ring[index] = i
                index += 1
        assert index == P
        return w_iso, w_aniso, az, ring

    def test_weights_and_azimuths_are_original_bits(self):
        w_iso, w_aniso, az, ring = self._live_tables()
        assert np.array_equal(
            bits(svf_fold.W_ISO), bits(w_iso)
        ), "frozen w_iso is not the original torch bit pattern"
        assert np.array_equal(
            bits(svf_fold.W_ANISO), bits(w_aniso)
        ), "frozen w_aniso is not the original torch bit pattern"
        assert np.array_equal(
            bits(svf_fold.PATCH_AZIMUTH), bits(az)
        ), "frozen patch azimuths are not the original bit patterns"
        assert np.array_equal(svf_fold.PATCH_RING, ring)
        assert list(svf_fold.N_ANNULUS) == [12, 12, 12, 12, 12, 12, 12, 6]
        # padded slots (annuli beyond the ring's count) are exact +0.0
        for i, n in enumerate(svf_fold.N_ANNULUS):
            assert int((bits(svf_fold.W_ISO[i, n:]) != 0).sum()) == 0
            assert int((bits(svf_fold.W_ANISO[i, n:]) != 0).sum()) == 0

    def test_last_trans_constants(self):
        # 3.0459e-004 and 0.03 exactly as the source literals round
        assert int(bits(np.array([svf_fold.LAST_CONST]))[0]) == int(
            bits(np.array([F32(3.0459e-004)]))[0]
        )
        assert int(bits(np.array([svf_fold.TRANS]))[0]) == int(
            bits(np.array([F32(0.03)]))[0]
        )
        # and the actual original bit patterns (frozen hex)
        assert int(bits(np.array([svf_fold.LAST_CONST]))[0]) == 0x399FB161
        assert int(bits(np.array([svf_fold.TRANS]))[0]) == 0x3CF5C28F

    def test_direction_predicates_partition(self):
        """Each patch belongs to exactly two directions; the predicates use
        only comparisons (T01-clean) against the frozen azimuth bits."""
        az = svf_fold.PATCH_AZIMUTH
        expect_e = (az >= F32(0.0)) & (az < F32(180.0))
        expect_s = (az >= F32(90.0)) & (az < F32(270.0))
        expect_w = (az >= F32(180.0)) & (az < F32(360.0))
        expect_n = (az >= F32(270.0)) | (az < F32(90.0))
        assert np.array_equal(svf_fold.DIRECTION_E, expect_e)
        assert np.array_equal(svf_fold.DIRECTION_S, expect_s)
        assert np.array_equal(svf_fold.DIRECTION_W, expect_w)
        assert np.array_equal(svf_fold.DIRECTION_N, expect_n)
        counts = (
            svf_fold.DIRECTION_E.astype(np.int32)
            + svf_fold.DIRECTION_S.astype(np.int32)
            + svf_fold.DIRECTION_W.astype(np.int32)
            + svf_fold.DIRECTION_N.astype(np.int32)
        )
        assert int(counts.min()) == 2 and int(counts.max()) == 2

    def test_digest_self_consistent(self):
        digest = svf_fold.frozen_tables_digest()
        assert digest == svf_fold.FOLD_CONSTANTS_SHA256
        assert len(digest) == 64
        # deterministic: same tables, same digest, every call
        assert svf_fold.frozen_tables_digest() == digest


# ---------------------------------------------------------------------------
# Gate 1: oracle raw equality — synthetic scenes
# ---------------------------------------------------------------------------


class TestFoldOracleSynthetic:
    """M4/M5 designated killer: any pre-summed annulus weight or reordered
    patch reduction changes float32 accumulation and fails here."""

    def test_vegetated_scene(self):
        inputs = synth_scene(40, 80, 7, "veg")
        oracle = oracle_svf(inputs)
        assert nonbinary_count(oracle["vegshmat"], oracle["vbshvegshmat"]) == 0
        result = fold_from_oracle(oracle, inputs)
        assert_fold_matches_oracle(result, oracle, "veg 40x80")

    def test_unvegetated_scene(self):
        inputs = synth_scene(40, 80, 11, "noveg")
        oracle = oracle_svf(inputs)
        assert nonbinary_count(oracle["vegshmat"], oracle["vbshvegshmat"]) == 0
        result = fold_from_oracle(oracle, inputs)
        assert_fold_matches_oracle(result, oracle, "noveg 40x80")

    def test_non_square_and_negative_dem(self):
        inputs = synth_scene(64, 48, 17, "veg")
        inputs.dem -= np.float32(20.0)
        inputs = sw.compose_march_inputs(
            inputs.a, inputs.dem, inputs.canopy
        )
        oracle = oracle_svf(inputs)
        result = fold_from_oracle(oracle, inputs)
        assert_fold_matches_oracle(result, oracle, "64x48 negdem")

    def test_nan_and_signed_zero_scene(self):
        """NaN in a; -0.0 (last applies) and NaN (last does not) in vegdem2."""
        inputs = nan_scene(40, 80, 19)
        oracle = oracle_svf(inputs)
        assert nonbinary_count(oracle["vegshmat"], oracle["vbshvegshmat"]) == 0
        # -0.0 cells took the last correction on the oracle side
        assert oracle["svfSveg"][2, 3] != np.float32(0.0)
        result = fold_from_oracle(oracle, inputs)
        assert_fold_matches_oracle(result, oracle, "nan/-0 40x80")

    def test_clamped_regime(self):
        """Dense tall canopy -> many directional veg sums above 1 -> the
        [x > 1.] = 1. clamps actually engage on both sides."""
        rows, cols = 36, 44
        rng = np.random.default_rng(23)
        dem = rng.uniform(0.0, 0.5, (rows, cols)).astype(np.float32)
        a = dem + rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
        canopy = np.full((rows, cols), np.float32(30.0), dtype=np.float32)
        canopy[0, :] = np.float32(0.0)
        inputs = sw.compose_march_inputs(a, dem, canopy)
        oracle = oracle_svf(inputs)
        clamped = sum(
            int((oracle[name] == np.float32(1.0)).sum()) for name in FOLD_OUTPUTS[:10]
        )
        assert clamped > 0, "fixture never reached the > 1 clamp"
        result = fold_from_oracle(oracle, inputs)
        assert_fold_matches_oracle(result, oracle, "clamp 36x44")


# ---------------------------------------------------------------------------
# One-step regime: packed refusal is TYPED, never a silent bool collapse
# ---------------------------------------------------------------------------


class TestOneStepRegimeRefusal:
    def test_vbsh_two_refused(self):
        inputs = synth_scene(64, 64, 13, "veg")
        oracle = oracle_svf(inputs, scale=0.25)
        bad = nonbinary_count(oracle["vegshmat"], oracle["vbshvegshmat"])
        assert bad > 0, "fixture is not a one-step regime scene"
        assert float(oracle["vbshvegshmat"].max()) == 2.0
        with pytest.raises(ValueError, match="binary"):
            bp.pack_bits(oracle["vbshvegshmat"])

    def test_fold_refuses_non_packed_inputs(self):
        inputs = synth_scene(16, 16, 3, "veg")
        oracle = oracle_svf(inputs)
        with pytest.raises((TypeError, ValueError)):
            svf_fold.fold_svf(
                oracle["vegshmat"],  # raw cube, not packed
                bp.pack_bits(oracle["vbshvegshmat"]),
                inputs.vegdsm2,
                oracle["svf"],
            )

    def test_fold_refuses_wrong_patch_count(self):
        inputs = synth_scene(16, 16, 3, "veg")
        oracle = oracle_svf(inputs)
        with pytest.raises((ValueError,)):
            svf_fold.fold_svf(
                bp.pack_bits(oracle["vegshmat"][:, :, :64]),
                bp.pack_bits(oracle["vbshvegshmat"]),
                inputs.vegdsm2,
                oracle["svf"],
            )

    def test_fold_refuses_shape_or_dtype_mismatch(self):
        inputs = synth_scene(16, 16, 3, "veg")
        oracle = oracle_svf(inputs)
        packed_veg = bp.pack_bits(oracle["vegshmat"])
        packed_vbsh = bp.pack_bits(oracle["vbshvegshmat"])
        with pytest.raises(ValueError):
            svf_fold.fold_svf(
                packed_veg, packed_vbsh, inputs.vegdsm2[:, :8], oracle["svf"]
            )
        with pytest.raises(ValueError):
            svf_fold.fold_svf(
                packed_veg, packed_vbsh,
                inputs.vegdsm2.astype(np.float64), oracle["svf"],
            )


# ---------------------------------------------------------------------------
# Synthetic packed scenes (no oracle march cost) for threading / routing
# ---------------------------------------------------------------------------


def packed_scene(rows, cols, seed, density=0.35):
    rng = np.random.default_rng(seed)
    vegsh = (rng.random((rows, cols, P)) < density).astype(np.float32)
    vbsh = (rng.random((rows, cols, P)) < density).astype(np.float32)
    vegdem2 = rng.uniform(0.0, 5.0, (rows, cols)).astype(np.float32)
    vegdem2[rng.random((rows, cols)) < 0.4] = np.float32(0.0)  # last engages
    svf_building = rng.uniform(0.4, 1.0, (rows, cols)).astype(np.float32)
    return {
        "veg": bp.pack_bits(vegsh),
        "vbsh": bp.pack_bits(vbsh),
        "vegdem2": vegdem2,
        "svf": svf_building,
    }


class TestThreading:
    BIG = (192, 384)

    def test_thread_counts_raw_equal(self):
        rows, cols = self.BIG
        scene = packed_scene(rows, cols, 29)
        assert rows * cols >= svf_fold.SERIAL_MIN_CELLS
        base = svf_fold.fold_svf(
            scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
            n_threads=1,
        )
        assert base.stats["used_threads"] == 1
        for threads in (2, 4, 8):
            par = svf_fold.fold_svf(
                scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
                n_threads=threads,
            )
            assert par.stats["used_threads"] >= 2
            for name in FOLD_OUTPUTS:
                a = getattr(base, name)
                b = getattr(par, name)
                assert np.array_equal(bits(a), bits(b)), (
                    f"{threads} threads: {name} diverged"
                )

    def test_small_scene_routes_serial(self):
        scene = packed_scene(16, 16, 31)
        result = svf_fold.fold_svf(
            scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
            n_threads=8,
        )
        assert 16 * 16 < svf_fold.SERIAL_MIN_CELLS
        assert result.stats["used_threads"] == 1

    def test_row_partition_is_exact(self):
        rows, cols = self.BIG
        scene = packed_scene(rows, cols, 29)
        result = svf_fold.fold_svf(
            scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
            n_threads=4,
        )
        covered = 0
        seen_rows = 0
        for lo, hi in result.stats["chunks"]:
            assert 0 <= lo < hi <= rows
            assert lo == covered, "chunk plan is not contiguous"
            covered = hi
            seen_rows += hi - lo
        assert covered == rows and seen_rows == rows

    def test_deterministic_repeat(self):
        scene = packed_scene(96, 128, 37)
        args = (scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"])
        r1 = svf_fold.fold_svf(*args, n_threads=4)
        r2 = svf_fold.fold_svf(*args, n_threads=4)
        for name in FOLD_OUTPUTS:
            assert np.array_equal(bits(getattr(r1, name)), bits(getattr(r2, name)))


# ---------------------------------------------------------------------------
# Gate 4: affected-chunk-only output
# ---------------------------------------------------------------------------


class TestAffectedOnly:
    def test_masked_cells_match_full_fold_and_base_untouched(self):
        rows, cols = 96, 128
        scene = packed_scene(rows, cols, 41)
        full = svf_fold.fold_svf(
            scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"]
        )
        rng = np.random.default_rng(43)
        mask = rng.random((rows, cols)) < 0.15
        assert int(mask.sum()) > 0
        # base planes: arbitrary prior state (NOT zeros — must survive)
        base = {
            name: rng.uniform(-3.0, 3.0, (rows, cols)).astype(np.float32)
            for name in FOLD_OUTPUTS
        }
        result = svf_fold.fold_svf(
            scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
            cell_mask=mask, outputs=base,
        )
        assert result.stats["masked_cells"] == int(mask.sum())
        for name in FOLD_OUTPUTS:
            got = getattr(result, name)
            # masked cells == the full fold, raw bits
            assert np.array_equal(bits(got)[mask], bits(getattr(full, name))[mask]), (
                f"{name}: masked cell diverged from the full fold"
            )
            # every other cell == the caller's base, raw bits
            assert np.array_equal(bits(got)[~mask], bits(base[name])[~mask]), (
                f"{name}: unmasked cell was touched"
            )

    def test_mask_chunk_alignment_semantics(self):
        """A mask whose affected cells live inside two specific chunks:
        outside chunks the base survives bit-for-bit (the packed-update
        contract at chunk granularity)."""
        rows, cols = 128, 128
        scene = packed_scene(rows, cols, 47)
        mask = np.zeros((rows, cols), dtype=bool)
        mask[64:100, 0:40] = True  # chunks (1, 0) and part of (1, 0) only
        rng = np.random.default_rng(53)
        base = {
            name: rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
            for name in FOLD_OUTPUTS
        }
        result = svf_fold.fold_svf(
            scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
            cell_mask=mask, outputs=base,
        )
        affected_chunks = bp.chunks_overlapping_mask(mask)
        assert affected_chunks == [(1, 0)]
        untouched = np.ones((rows, cols), dtype=bool)
        for cr, cc in affected_chunks:
            (r0, r1), (c0, c1) = bp.chunk_window(rows, cols, cr, cc)
            untouched[r0:r1, c0:c1] = False
        for name in FOLD_OUTPUTS:
            got = getattr(result, name)
            assert np.array_equal(bits(got)[untouched], bits(base[name])[untouched])

    def test_empty_mask_zero_work(self):
        scene = packed_scene(40, 80, 59)
        rng = np.random.default_rng(61)
        base = {
            name: rng.uniform(-1.0, 1.0, (40, 80)).astype(np.float32)
            for name in FOLD_OUTPUTS
        }
        result = svf_fold.fold_svf(
            scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
            cell_mask=np.zeros((40, 80), dtype=bool), outputs=base,
        )
        assert result.stats["masked_cells"] == 0
        for name in FOLD_OUTPUTS:
            assert np.array_equal(
                bits(getattr(result, name)), bits(base[name])
            ), f"{name}: empty mask must return the base untouched"

    def test_outputs_validation(self):
        scene = packed_scene(16, 16, 67)
        with pytest.raises(ValueError):
            svf_fold.fold_svf(
                scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
                cell_mask=np.zeros((8, 8), dtype=bool),
            )
        bad_base = {"svfveg": np.zeros((8, 8), dtype=np.float32)}
        with pytest.raises(ValueError):
            svf_fold.fold_svf(
                scene["veg"], scene["vbsh"], scene["vegdem2"], scene["svf"],
                outputs=bad_base,
            )


# ---------------------------------------------------------------------------
# Gate 2: downstream one hop — svfbuveg exactly as utci_process computes it
# ---------------------------------------------------------------------------


class TestDownstreamSvfbuveg:
    def test_svfbuveg_raw_bits(self):
        inputs = synth_scene(40, 80, 7, "veg")
        oracle = oracle_svf(inputs)
        result = fold_from_oracle(oracle, inputs)
        # reference: solweig_gpu/utci_process.py:867 semantics, in torch,
        # exactly as the consumer line executes it
        trans_veg = 3.0 / 100.0  # utci_process.py:774 default
        svf_t = torch.from_numpy(
            np.ascontiguousarray(oracle["svf"], dtype=np.float32)
        )
        svfveg_t = torch.from_numpy(
            np.ascontiguousarray(oracle["svfveg"], dtype=np.float32)
        )
        ref = svf_t - (1.0 - svfveg_t) * (1.0 - trans_veg)
        # candidate: my fold's svfveg through the same formula in numpy
        one = np.float32(1.0)
        trans_f32 = np.float32(1.0 - trans_veg)
        cand = oracle["svf"] - (one - result.svfveg) * trans_f32
        report = compare_plane_bits(ref.numpy(), cand)
        assert report["status"] == "PASS", report
        assert int(report["mismatch_count"]) == 0
        # and my svftotal IS this quantity with the svf_calculator constant
        report2 = compare_plane_bits(oracle["SVFtotal"], result.svftotal)
        assert report2["status"] == "PASS", report2


# ---------------------------------------------------------------------------
# Gate 1 site leg: real site_500 through the ORIGINAL svf_calculator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def site_oracle():
    if not SITE_500_CACHE.is_dir():
        pytest.skip(f"site cache {SITE_500_CACHE} not present")
    from solweig_gpu.incremental.cache import SiteCache

    cache = SiteCache.load(SITE_500_CACHE)
    canopy = np.ascontiguousarray(
        np.asarray(cache.tree_base, dtype=np.float32)
    )
    a = np.array(cache.building_dsm, dtype=np.float32)
    dem = np.array(cache.dem, dtype=np.float32)
    inputs = sw.compose_march_inputs(a, dem, canopy)
    oracle = oracle_svf(inputs, scale=1.0 / float(cache.pixel_size_m))
    return cache, inputs, oracle


class TestFoldOracleSite:
    def test_site_500_all_outputs(self, site_oracle):
        cache, inputs, oracle = site_oracle
        bad = nonbinary_count(oracle["vegshmat"], oracle["vbshvegshmat"])
        assert bad == 0, (
            f"site_500 vegsh/vbsh has {bad} non-binary cells — packed "
            "representation is inapplicable to this scene (honest BLOCKED)"
        )
        result = fold_from_oracle(oracle, inputs)
        assert_fold_matches_oracle(result, oracle, "site_500")

    def test_site_500_affected_only_matches_full(self, site_oracle):
        cache, inputs, oracle = site_oracle
        result = fold_from_oracle(oracle, inputs)
        rng = np.random.default_rng(71)
        mask = rng.random(oracle["svf"].shape) < 0.1
        base = {
            name: np.full(oracle["svf"].shape, np.float32(-7.0), dtype=np.float32)
            for name in FOLD_OUTPUTS
        }
        partial = fold_from_oracle(
            oracle, inputs, cell_mask=mask, outputs=base
        )
        assert partial.stats["masked_cells"] == int(mask.sum())
        for name in FOLD_OUTPUTS:
            got = getattr(partial, name)
            assert np.array_equal(
                bits(got)[mask], bits(getattr(result, name))[mask]
            )
            assert int((bits(got)[~mask] != bits(base[name])[~mask]).sum()) == 0


# ---------------------------------------------------------------------------
# Purity (gate 6)
# ---------------------------------------------------------------------------


class TestPurity:
    def test_source_torch_free(self):
        for rel in (
            "solweig_core/bitplanes.py",
            "solweig_core/numba_cpu/svf_fold.py",
        ):
            source = (REPO_ROOT / rel).read_text()
            assert "import torch" not in source, rel
            assert "from torch" not in source, rel

    def test_fastmath_absent(self):
        source = (
            REPO_ROOT / "solweig_core" / "numba_cpu" / "svf_fold.py"
        ).read_text()
        assert "fastmath=True" not in source
        assert "parallel=True" not in source
        assert "prange" not in source

    def test_subprocess_torch_free_fold(self):
        code = (
            "import sys; "
            "assert 'torch' not in sys.modules; "
            "import numpy as np; "
            "from solweig_core import bitplanes as bp; "
            "from solweig_core.numba_cpu import svf_fold; "
            "assert 'torch' not in sys.modules; "
            "rng = np.random.default_rng(1); "
            "veg = (rng.random((8, 8, 153)) < 0.3).astype(np.float32); "
            "vb = (rng.random((8, 8, 153)) < 0.3).astype(np.float32); "
            "v2 = rng.uniform(0, 3, (8, 8)).astype(np.float32); "
            "svf = rng.uniform(0.4, 1, (8, 8)).astype(np.float32); "
            "r = svf_fold.fold_svf(bp.pack_bits(veg), bp.pack_bits(vb), v2, svf); "
            "assert r.svfveg.shape == (8, 8) and r.svfveg.dtype == np.float32; "
            "assert r.stats['masked_cells'] == 64; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
