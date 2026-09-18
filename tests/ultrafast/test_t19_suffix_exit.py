# SPDX-License-Identifier: GPL-3.0-only
"""T19 gates: post-onset suffix exit (T27 verdict C6) bit-invariance.

Gate structure (T17-card protocol: hypothesis -> exact equivalence argument
-> counterexample search -> independent test):

* **Witness gates** — the two designated RED witnesses of the T27 proof
  obligations, built as minimal scenes:

  - the naive ``sh == 1``-only escape counterexample (onset on step 0 +
    the first-step vegetation exception leaving ``vegsh == 1`` AFTER
    suppression + ``count >= 2``): the correct exit must NOT fire at the
    end of step 0 (the ``vegsh == zero`` conjunct), so exit == full march.
    A naive mutation flips that cell's ``vbsh`` 1 -> 2 and this gate goes
    RED (demonstrated by sed mutation in artifacts/t19/mutations/).
  - the NaN guard witness (onset on step 0, a NaN source cell later in
    the chain): torch-max ABSORBS the NaN, ``sh`` falls back to 0 and the
    suffix is NOT 0-contribution — the guard must disable the exit so
    exit == full march. Removing the guard makes this gate RED.
  - one-step (``count == 1``) and zero-step tables, negative-DEM OOB
    phantom onset (``ta == 0 > a_t`` at out-of-bounds steps).

* **Differential bit gate (A1)** — ``suffix_exit=True`` vs
  ``suffix_exit=False`` raw-uint32-identical on: every one of the 74
  frozen T03 traces, the variant grid (azimuth-0 neighbourhood, altitude
  ladder, scales 0.5/1.0/2.0, negative DSM, no-veg, no-building, overlap),
  the site_500 scene (6-degree ring in full + a strided subset of the
  remaining patches + the absolute-amplitude high-altitude witnesses),
  and the sparse runner at 1/2/4/8 threads vs the dense kernel (T06
  discipline).

Both settings produce identical bits BY THEOREM (the exit only skips
provably 0-contribution steps), so every gate below compares the two
numba settings directly; equality of the ``suffix_exit=False`` setting
with the original torch kernels is already pinned by the T04 gates, and
transfers through this differential.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Sibling-module import (trace_exporter lives in this directory): the path
# insertion must be local to THIS file — a bare import only works when
# another test module happened to seed sys.path first (order-dependent,
# fails under isolated execution).
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu.march import (  # noqa: E402
    march_svf_shadow,
    march_wallheight23,
)
from solweig_core.numba_cpu.sparse_march import (  # noqa: E402
    march_svf_shadow_sparse,
)
from solweig_core.numba_cpu.step_table_gen import (  # noqa: E402
    build_table_general,
)

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
FROZEN_TRACES = (
    REPO_ROOT.parent / "solweig_ultrafast_artifacts" / "t03" / "traces"
)
FROZEN_MANIFEST = (
    REPO_ROOT.parent / "solweig_ultrafast_artifacts" / "t03"
    / "trace_manifest.json"
)

# A small grid is enough: the witnesses are per-target-cell state machines.
WITNESS_ROWS = 12
WITNESS_COLS = 12
WITNESS_TARGET = (5, 5)
#: low altitude + scale 1 -> dz_0 ~= 0.21 m, amplitude 30 -> count >= 100
#: (far beyond the 2 steps the witnesses need).
WITNESS_AMP = 30.0


def _bits_f32(hexbits: str) -> np.float32:
    return np.uint32(st.bits_hex_to_u32(hexbits)).view(np.float32)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_exit_vs_full(got, want, label) -> list[str]:
    """Raw-uint32 comparison of the three march planes (signed zero and
    NaN payloads included — the T19 contract is raw bits, not values)."""
    failures = []
    for name, g, w in zip(("sh", "vegsh", "vbshvegsh"), got, want):
        gb = (
            np.ascontiguousarray(g, dtype=np.float32).view(np.uint32).ravel()
        )
        wb = (
            np.ascontiguousarray(w, dtype=np.float32).view(np.uint32).ravel()
        )
        if not np.array_equal(gb, wb):
            bad = np.nonzero(gb != wb)[0]
            failures.append(
                f"{label} plane {name}: {bad.size} mismatched elements, "
                f"first idx {int(bad[0])} got "
                f"{int(gb[bad[0]]):#010x} want {int(wb[bad[0]]):#010x}"
            )
    return failures


def naive_witness_scene(nan_source: bool = False):
    """The designated counterexample scene (T27 §1.7 / §3.3 witness (i)).

    Target (5, 5) with ``a_t == 0``; the step-0 shifted source carries
    ``a == 5`` (onset: ``ta = 5 - dz_0 > 0``) and ``vegdem == 5.1`` (the
    first-step exception's ``fv = 0.1 in (0, dz_0)`` sets ``vegsh = 1``
    AFTER suppression); the target's ``vegdem2 == 0.5 > a_t`` keeps the
    trunk gate open. With ``count >= 2`` the CORRECT exit must run step 1
    (whose suppression zeroes the exception's ``vegsh``) before breaking;
    the naive ``sh == 1``-only escape breaks at step 0 and flips the
    cell's ``vbsh`` from 1 to 2.

    ``nan_source=True`` additionally poisons the step-2 source cell of
    the a-plane (the NaN-guard witness): onset has already fired, then
    ``f`` absorbs the NaN and ``sh`` falls back to 0 — the suffix is not
    0-contribution, so the exit must be disabled for the whole batch.
    """
    rows, cols = WITNESS_ROWS, WITNESS_COLS
    a = np.zeros((rows, cols), dtype=np.float32)
    vegdem = np.zeros((rows, cols), dtype=np.float32)
    vegdem2 = np.zeros((rows, cols), dtype=np.float32)
    bush = np.zeros((rows, cols), dtype=np.float32)
    i, j = WITNESS_TARGET
    vegdem2[i, j] = np.float32(0.5)      # trunk gate open at the target
    table = build_table_general(
        st.KERNEL_SVF_SHADOW, 45.0, 12.0, 1.0, rows, cols, WITNESS_AMP
    )
    assert table.count >= 2, "witness needs a multi-step march"
    si = i + int(table.dx[0])
    sj = j + int(table.dy[0])
    a[si, sj] = np.float32(5.0)          # onset at step 0
    vegdem[si, sj] = np.float32(5.1)     # fv = 0.1 < dz_0 ~= 0.21
    if nan_source:
        assert table.count >= 3, "NaN witness needs a step-2 source cell"
        ni = i + int(table.dx[2])
        nj = j + int(table.dy[2])
        assert (ni, nj) != (si, sj), "step-2 cell must differ from step 0"
        a[ni, nj] = np.float32("nan")
    return table, a, vegdem, vegdem2, bush


# ---------------------------------------------------------------------------
# Witness gates
# ---------------------------------------------------------------------------


class TestSuffixExitWitnesses:
    def test_naive_counterexample_scene_exit_equals_full(self):
        """Onset-on-step-0 + exception vegsh=1 + count>=2: the correct
        exit defers to step 1, so bits equal the full march (the naive
        sh-only mutation makes THIS test fail with vbsh 1 -> 2)."""
        table, a, vegdem, vegdem2, bush = naive_witness_scene()
        got = march_svf_shadow(table, a, vegdem, vegdem2, bush)
        want = march_svf_shadow(
            table, a, vegdem, vegdem2, bush, suffix_exit=False
        )
        failures = compare_exit_vs_full(got, want, "naive-witness")
        assert not failures, "\n".join(failures)
        # the witness cell actually sits in the regime the counterexample
        # describes: full march ends vegsh=1 / vbsh=1 (NOT the naive 0/2)
        i, j = WITNESS_TARGET
        assert float(want[1][i, j]) == 1.0 and float(want[2][i, j]) == 1.0, (
            f"witness cell out of regime: vegsh={float(want[1][i, j])} "
            f"vbsh={float(want[2][i, j])}"
        )

    def test_nan_scene_guard_disables_exit(self):
        """a-plane NaN after onset: the guard must disable the exit for
        the whole batch, so bits equal the full march (a guard-removal
        mutation makes THIS test fail)."""
        table, a, vegdem, vegdem2, bush = naive_witness_scene(
            nan_source=True
        )
        got = march_svf_shadow(table, a, vegdem, vegdem2, bush)
        want = march_svf_shadow(
            table, a, vegdem, vegdem2, bush, suffix_exit=False
        )
        failures = compare_exit_vs_full(got, want, "nan-witness")
        assert not failures, "\n".join(failures)
        # the hazard is real on this scene: sh falls back to 0 after the
        # NaN is absorbed, so the full-march final sh is 1 — an unguarded
        # exit-at-onset would publish 0 there.
        i, j = WITNESS_TARGET
        assert float(want[0][i, j]) == 1.0, "NaN witness lost its hazard"

    def test_nan_scene_sparse_runner_matches_dense(self):
        table, a, vegdem, vegdem2, bush = naive_witness_scene(
            nan_source=True
        )
        rows, cols = a.shape
        runs = np.asarray(
            [[r, 0, cols - 1] for r in range(rows)], dtype=np.int32
        )
        dense = march_svf_shadow(table, a, vegdem, vegdem2, bush)
        for n_threads in (1, 2):
            sparse = march_svf_shadow_sparse(
                table, a, vegdem, vegdem2, bush, runs, n_threads=n_threads
            )
            failures = compare_exit_vs_full(
                (sparse.sh, sparse.vegsh, sparse.vbshvegsh),
                dense,
                f"sparse NaN t={n_threads}",
            )
            assert not failures, "\n".join(failures)

    def test_one_step_regime_exit_equals_full(self):
        """count == 0 and count == 1 (the vbsh == 2.0 one-step regime):
        the exit cannot fire before the step(s) complete; bits equal the
        full march."""
        rows, cols = 12, 12
        rng = np.random.default_rng(11)
        a = rng.uniform(-2.0, 6.0, (rows, cols)).astype(np.float32)
        vegdem = a + rng.uniform(0.0, 2.0, (rows, cols)).astype(np.float32)
        vegdem2 = a + rng.uniform(0.0, 0.5, (rows, cols)).astype(np.float32)
        bush = np.zeros((rows, cols), dtype=np.float32)

        # count == 0 (negative amplitude): outputs come purely from the
        # init state — built directly, as the T04 zero-step gate does
        # (the previous-state while always executes step 1 for positive
        # amplitudes, so no positive amplitude produces a zero-step svf
        # table).
        table0 = st.StepTable(
            key=st.StepTableKey(
                semantics_profile="canonical_cpu_v1",
                kernel_variant=st.KERNEL_SVF_SHADOW,
                source_geometry_hash="0" * 64,
                angle_input_bits=(
                    st.f32_bits_hex(37.0), st.f32_bits_hex(35.0)
                ),
                scale_bits=st.f32_bits_hex(0.5),
                logical_rows=rows,
                logical_cols=cols,
                amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
                executed_amplitude_bits=st.f32_bits_hex(-1.0),
                boundary_policy_id=st.BOUNDARY_SHIFT_WINDOW_V1,
            ),
            count=0,
            dx=np.zeros(0, dtype=np.int32),
            dy=np.zeros(0, dtype=np.int32),
            dz_bits=np.zeros(0, dtype=np.uint32),
            stop_reason=st.STOP_AMPLITUDE,
            branch_id=np.zeros(0, dtype=np.uint8),
            previous_dz_bits=np.zeros(0, dtype=np.uint32),
        )
        got = march_svf_shadow(table0, a, vegdem, vegdem2, bush)
        want = march_svf_shadow(
            table0, a, vegdem, vegdem2, bush, suffix_exit=False
        )
        assert not compare_exit_vs_full(got, want, "zero-step")

        # count == 1: the previous-state while always executes step 1
        # (dz_prev == 0 at entry), so the one-step regime is any positive
        # amplitude BELOW dz_1 (the step-2 entry check amax >= dz_1 fails).
        t_probe = build_table_general(
            st.KERNEL_SVF_SHADOW, 45.0, 12.0, 1.0, rows, cols, 30.0
        )
        dzs = (
            np.asarray(t_probe.dz_bits, dtype=np.uint32)
            .view(np.float32)
            .astype(np.float64)
        )
        one_step_amp = float(dzs[0]) * 0.5
        table1 = build_table_general(
            st.KERNEL_SVF_SHADOW, 45.0, 12.0, 1.0, rows, cols, one_step_amp
        )
        assert table1.count == 1, "one-step fixture drifted"
        got = march_svf_shadow(table1, a, vegdem, vegdem2, bush)
        want = march_svf_shadow(
            table1, a, vegdem, vegdem2, bush, suffix_exit=False
        )
        assert not compare_exit_vs_full(got, want, "one-step")

    def test_negative_dem_oob_phantom_onset(self):
        """Negative-DEM targets take onset from the OOB zero phantom
        (``ta == 0 > a_t``): the exit must still equal the full march."""
        rows, cols = 24, 24
        rng = np.random.default_rng(23)
        dem = np.float32(-12.0) * np.ones((rows, cols), dtype=np.float32)
        a = dem + rng.uniform(0.0, 3.0, (rows, cols)).astype(np.float32)
        vegdem = a + rng.uniform(0.0, 4.0, (rows, cols)).astype(np.float32)
        vegdem2 = a + rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
        bush = np.zeros((rows, cols), dtype=np.float32)
        for alt in (12.0, 42.0, 78.0):
            table = build_table_general(
                st.KERNEL_SVF_SHADOW, 133.0, alt, 0.5, rows, cols, 25.0
            )
            got = march_svf_shadow(table, a, vegdem, vegdem2, bush)
            want = march_svf_shadow(
                table, a, vegdem, vegdem2, bush, suffix_exit=False
            )
            assert not compare_exit_vs_full(got, want, f"negdem alt={alt}")


# ---------------------------------------------------------------------------
# Differential bit gate: frozen 74 + variant grid
# ---------------------------------------------------------------------------


def synthetic_scene(rows, cols, *, seed=7, negative_dem=False,
                    no_vegetation=False, no_building=False, overlap=False):
    """Deterministic synthetic planes (same generator family as the T03/T04
    replay tests — numpy here, torch-free, since the gate is numba-only)."""
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    if negative_dem:
        dem -= np.float32(12.0)
    a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    if not no_building:
        a[3: min(9, rows), 4: min(12, cols)] += np.float32(18.0)
    if no_vegetation:
        canopy = np.zeros((rows, cols), dtype=np.float32)
    else:
        canopy = rng.uniform(0.0, 6.0, (rows, cols)).astype(np.float32)
        canopy[canopy < np.float32(3.0)] = np.float32(0.0)
        canopy[min(10, rows): min(20, rows), min(30, cols): min(60, cols)] = (
            np.float32(5.5)
        )
    if overlap:
        canopy[min(2, rows): min(11, rows), min(6, cols): min(16, cols)] = (
            np.float32(25.0)
        )
    vegdem = canopy + dem
    vegdem2 = canopy * np.float32(0.25) + dem
    bush = np.zeros((rows, cols), dtype=np.float32)
    return a, vegdem, vegdem2, bush


@pytest.fixture(scope="module")
def frozen_tables():
    if not FROZEN_TRACES.is_dir() or not FROZEN_MANIFEST.is_file():
        pytest.skip("frozen T03 artifact traces not present")
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    tables = []
    for entry in manifest["entries"]:
        path = FROZEN_TRACES / f"{entry['trace_id']}.json"
        assert _sha256_file(path) == entry["trace_sha256"], path.name
        trace = json.loads(path.read_text())
        table = st.build_table_from_trace(trace)
        assert table.content_digest() == entry["content_sha256"], path.name
        tables.append(table)
    assert len(tables) == manifest["trace_count"] == 74
    return tables


class TestFrozen74Differential:
    def test_all_frozen_traces_exit_equals_full(self, frozen_tables):
        failures = []
        checked = {"svf_shadow": 0, "wallheight_23": 0}
        for table in frozen_tables:
            key = table.key
            rows, cols = key.logical_rows, key.logical_cols
            a, vegdem, vegdem2, bush = synthetic_scene(rows, cols, seed=7)
            march = (
                march_svf_shadow
                if key.kernel_variant == st.KERNEL_SVF_SHADOW
                else march_wallheight23
            )
            got = march(table, a, vegdem, vegdem2, bush)
            want = march(table, a, vegdem, vegdem2, bush, suffix_exit=False)
            failures.extend(
                compare_exit_vs_full(got, want, f"{table.trace_id}")
            )
            checked[key.kernel_variant] += 1
        assert not failures, "\n".join(failures[:6])
        assert checked == {"svf_shadow": 37, "wallheight_23": 37}


class TestVariantGridDifferential:
    CASES = [
        (0.0, 78.0, 2.0, 20.0, {}),
        (37.0, 6.0, 2.0, 15.0, {}),
        (225.0, 42.0, 0.5, 20.0, {"no_vegetation": True}),
        (181.0, 42.0, 0.5, 20.0, {"no_building": True}),
        (133.0, 78.0, 1.0, 25.0, {"overlap": True}),
        (312.0, 6.0, 0.5, 30.0, {"negative_dem": True}),
        (90.0, 90.0, 0.5, 60.0, {"negative_dem": True}),
        (1e-12, 35.0, 1.0, 12.0, {}),
        (359.5, 55.0, 1.0, 18.0, {}),
    ]

    @pytest.mark.parametrize("kernel", st.KERNEL_VARIANTS)
    @pytest.mark.parametrize(
        "az, alt, scale, amp, kwargs", CASES, ids=lambda v: str(v)
    )
    def test_variant_case(self, kernel, az, alt, scale, amp, kwargs):
        rows, cols = 48, 36
        table = build_table_general(
            kernel, az, alt, scale, rows, cols, amp,
            amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
        )
        a, vegdem, vegdem2, bush = synthetic_scene(
            rows, cols, seed=13, **kwargs
        )
        march = (
            march_svf_shadow
            if kernel == st.KERNEL_SVF_SHADOW
            else march_wallheight23
        )
        got = march(table, a, vegdem, vegdem2, bush)
        want = march(table, a, vegdem, vegdem2, bush, suffix_exit=False)
        assert not compare_exit_vs_full(
            got, want, f"{kernel} az={az} alt={alt}"
        )


# ---------------------------------------------------------------------------
# Differential bit gate: site_500 (A1) + threads (T06 discipline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def site_scene():
    if not SITE_500_CACHE.is_dir():
        pytest.skip(f"site cache {SITE_500_CACHE} not present")
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.solver import _compose_scene_from_canopy
    from trace_exporter import capture_amplitude_policy

    cache = SiteCache.load(SITE_500_CACHE)
    canopy = np.asarray(cache.tree_base, dtype=np.float32)
    scene = _compose_scene_from_canopy(cache, canopy)
    scale = 1.0 / float(cache.pixel_size_m)
    policy = capture_amplitude_policy(
        scene.a, scene.vegdsm, scene.vegdsm2, scene.vegdem, scale=scale
    )
    assert float(scene.bush.max()) == 0.0, "site_500 bush domain changed"
    return scene, policy, scale


class TestSiteSceneDifferential:
    def test_site_patches_exit_equals_full(self, site_scene):
        from solweig_gpu.incremental.solver import _sky_patch_geometry

        scene, policy, scale = site_scene
        a = scene.a.numpy()
        vegdem = scene.vegdem.numpy()
        vegdem2 = scene.vegdem2.numpy()
        bush = scene.bush.numpy()
        rows, cols = a.shape
        eff = float(_bits_f32(policy["effective_bits"]))
        abs_ = float(_bits_f32(policy["oracle_absolute_bits"]))
        escalated = set(policy["r6_banded_patch_indices"])
        patches, _rings = _sky_patch_geometry(2)
        failures = []
        checked = ring6 = 0
        for index, (altitude, azimuth, _ring) in enumerate(patches):
            is_ring6 = float(altitude) == 6.0
            # full coverage of the 6-degree ring (the R6-heavy long chains)
            # plus a strided subset of every other ring; escalated patches
            # (none on site_500 today) are always covered.
            if not (is_ring6 or index % 7 == 0 or index in escalated):
                continue
            amp, pid = (
                (abs_, st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED)
                if index in escalated
                else (eff, st.AMPLITUDE_EFFECTIVE_WINDOWED)
            )
            table = build_table_general(
                st.KERNEL_SVF_SHADOW,
                float(azimuth),
                float(altitude),
                scale,
                rows,
                cols,
                amp,
                amplitude_policy_id=pid,
            )
            got = march_svf_shadow(table, a, vegdem, vegdem2, bush)
            want = march_svf_shadow(
                table, a, vegdem, vegdem2, bush, suffix_exit=False
            )
            failures.extend(
                compare_exit_vs_full(
                    got, want, f"site patch {index} alt={float(altitude)}"
                )
            )
            checked += 1
            ring6 += int(is_ring6)
        assert not failures, "\n".join(failures[:4])
        assert checked >= 40, f"only {checked} site patches covered"
        assert ring6 >= 20, f"only {ring6} six-degree patches covered"

    def test_site_absolute_amplitude_witnesses(self, site_scene):
        """The escalated-AMPLITUDE marching regime (the oracle's absolute
        stop on the high-altitude patches): exit == full there too."""
        from solweig_gpu.incremental.solver import _sky_patch_geometry

        scene, policy, scale = site_scene
        a = scene.a.numpy()
        vegdem = scene.vegdem.numpy()
        vegdem2 = scene.vegdem2.numpy()
        bush = scene.bush.numpy()
        rows, cols = a.shape
        abs_ = float(_bits_f32(policy["oracle_absolute_bits"]))
        patches, _rings = _sky_patch_geometry(2)
        failures = []
        for altitude, azimuth, _ring in patches:
            if float(altitude) not in (78.0, 90.0):
                continue
            table = build_table_general(
                st.KERNEL_SVF_SHADOW,
                float(azimuth),
                float(altitude),
                scale,
                rows,
                cols,
                abs_,
                amplitude_policy_id=(
                    st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED
                ),
            )
            got = march_svf_shadow(table, a, vegdem, vegdem2, bush)
            want = march_svf_shadow(
                table, a, vegdem, vegdem2, bush, suffix_exit=False
            )
            failures.extend(
                compare_exit_vs_full(
                    got, want, f"site absolute-amp alt={float(altitude)}"
                )
            )
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
    def test_sparse_threads_equal_dense_exit(self, site_scene, n_threads):
        """T06 discipline: the sparse runner with the suffix exit, at any
        thread count, stays raw-bit-equal to the dense exit march (and
        the dense full march through the gates above)."""
        from solweig_gpu.incremental.solver import _sky_patch_geometry

        scene, policy, scale = site_scene
        a = scene.a.numpy()
        vegdem = scene.vegdem.numpy()
        vegdem2 = scene.vegdem2.numpy()
        bush = scene.bush.numpy()
        rows, cols = a.shape
        eff = float(_bits_f32(policy["effective_bits"]))
        patches, _rings = _sky_patch_geometry(2)
        # two long-chain 6-degree patches and one mid-altitude patch
        picked = [p for p in patches if float(p[0]) == 6.0][:2]
        picked += [p for p in patches if float(p[0]) == 42.0][:1]
        runs = np.asarray(
            [[r, 0, cols - 1] for r in range(rows)], dtype=np.int32
        )
        for altitude, azimuth, _ring in picked:
            table = build_table_general(
                st.KERNEL_SVF_SHADOW,
                float(azimuth),
                float(altitude),
                scale,
                rows,
                cols,
                eff,
                amplitude_policy_id=st.AMPLITUDE_EFFECTIVE_WINDOWED,
            )
            dense = march_svf_shadow(table, a, vegdem, vegdem2, bush)
            sparse = march_svf_shadow_sparse(
                table,
                a,
                vegdem,
                vegdem2,
                bush,
                runs,
                n_threads=n_threads,
                min_pairs=1,
            )
            failures = compare_exit_vs_full(
                (sparse.sh, sparse.vegsh, sparse.vbshvegsh),
                dense,
                f"threads={n_threads} alt={float(altitude)}",
            )
            assert not failures, "\n".join(failures)
