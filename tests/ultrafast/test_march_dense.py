# SPDX-License-Identifier: GPL-3.0-only
"""T04 gates: dense serial Numba march kernel (DESIGN.ko.md 7.6, 12.1).

Gate structure (TASKS.ko.md T04 + lead contract):

* **RED first** — this file was written and run BEFORE
  ``solweig_core/numba_cpu/march.py`` existed; the recorded first failure is
  the collection ``ModuleNotFoundError`` (see artifacts/t04/red_first_run.txt).
* **Bit gate** — for every one of the 74 frozen T03 traces, the ORIGINAL
  kernels (:func:`solweig_gpu.shadow.shadow` /
  :func:`solweig_gpu.solweig.shadowingfunction_wallheight_23`) and the Numba
  march consume identical synthetic scenes and every output plane compares
  equal through ``bitwise_harness.compare_plane_bits`` (raw bits, signed
  zero and NaN payload included).
* **Scene gate** — site_500 composed scene: SVF shadow planes for the
  SUBSET_ALTITUDES + R6-escalated patches equal the original ``shadow()``
  outputs bit-for-bit on the full 500x500 domain.
* **Variant gates** — the contract's minimum coverage beyond the frozen 74:
  azimuth 0 neighbourhood, altitude 6/42/78/90, non-square shapes, scale
  0.5/1.0/2.0, no-vegetation, no-building, overlap max-composition, negative
  DSM, amplitude ladder, the one-step ``vbsh == 2.0`` witness, kernel
  boundary refusals (bush domain, variant mismatch, shape mismatch, dtype),
  strided-input equality and the zero-step table.
* **Mutation gate** — at least three source-level mutations of march.py
  (OOB -inf, dropped first-step vb reset, swapped accumulation order,
  shifted trunk gate) each make designated tests fail; applied by sed and
  recorded in artifacts/t04/mutations/, NOT in this file.
* **Purity gate** — march.py contains no ``import torch`` and no
  ``fastmath=True``; the module imports and runs torch-free in a subprocess.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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

from trace_exporter import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    capture_trace,
)
from bitwise_harness import compare_plane_bits  # noqa: E402
from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu.march import (  # noqa: E402
    march_svf_shadow,
    march_wallheight23,
)
from solweig_gpu.shadow import shadow as shadow_fn  # noqa: E402
from solweig_gpu.solweig import (  # noqa: E402
    shadowingfunction_wallheight_23 as wallheight23_fn,
)

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
FROZEN_TRACES = DEFAULT_ARTIFACT_ROOT / "traces"
FROZEN_MANIFEST = DEFAULT_ARTIFACT_ROOT / "trace_manifest.json"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bits_f32(value_bits: str) -> float:
    return float(st.u32_bits_to_f32(st.bits_hex_to_u32(value_bits)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_scene(
    rows: int,
    cols: int,
    *,
    seed: int = 7,
    negative_dem: bool = False,
    no_vegetation: bool = False,
    no_building: bool = False,
    overlap: bool = False,
):
    """Deterministic synthetic scene (torch float32 planes).

    Same generator family as the T03 replay tests: DEM + building block +
    sparse canopy; variants pin the contract's minimum coverage (negative
    DSM, no-vegetation, no-building, canopy overlapping the building).
    """
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    if negative_dem:
        dem -= np.float32(12.0)
    if no_building:
        a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    else:
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
            # canopy straddling the building corner, part taller than it
            canopy[min(2, rows) : min(11, rows), min(6, cols) : min(16, cols)] = (
                np.float32(25.0)
            )
    vegdem = canopy + dem
    vegdem2 = canopy * np.float32(0.25) + dem
    bush = np.zeros((rows, cols), dtype=np.float32)
    to_t = lambda arr: torch.from_numpy(np.ascontiguousarray(arr).copy())  # noqa: E731
    return to_t(a), to_t(vegdem), to_t(vegdem2), to_t(bush)


def run_original_svf(az, alt, scale, amp, a, vegdem, vegdem2, bush):
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


def run_original_wallheight(az, alt, scale, amp, a, vegdem, vegdem2, bush):
    zeros = torch.zeros(a.shape)
    out = wallheight23_fn(
        a,
        vegdem,
        vegdem2,
        float(np.float32(az)),
        float(np.float32(alt)),
        scale,
        float(np.float32(amp)),
        bush,
        zeros,
        zeros,
    )
    return out[0], out[1], out[2]  # vegsh, sh, vbshvegsh


def compare_triple(got, want, label):
    """compare_plane_bits on the three march planes; returns failure text."""
    failures = []
    for name, g, w in zip(("sh", "vegsh", "vbshvegsh"), got, want):
        report = compare_plane_bits(
            np.ascontiguousarray(w.detach().cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(g, dtype=np.float32),
        )
        if report["status"] != "PASS":
            failures.append(
                f"{label} plane {name}: {report['mismatch_count']} mismatched "
                f"elements, first={report['first_mismatches'][:2]}"
            )
    return failures


def march_table_for(kernel, az, alt, scale, rows, cols, amp):
    trace = capture_trace(
        kernel,
        az,
        alt,
        scale,
        rows,
        cols,
        amp,
        amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
        verify_probes=False,
    )
    return st.build_table_from_trace(trace), trace


#: (azimuth, altitude, scale, amplitude, scene kwargs) — the variant grid
#: completing the frozen 74: 42-degree altitude, scale 2.0, no-vegetation,
#: no-building, overlap max-composition, negative DSM, zenith.
VARIANT_CASES = [
    (0.0, 78.0, 2.0, 20.0, {}),
    (37.0, 6.0, 2.0, 15.0, {}),
    (225.0, 42.0, 0.5, 20.0, {"no_vegetation": True}),
    (181.0, 42.0, 0.5, 20.0, {"no_building": True}),
    (133.0, 78.0, 1.0, 25.0, {"overlap": True}),
    (312.0, 6.0, 0.5, 30.0, {"negative_dem": True}),
    (90.0, 90.0, 0.5, 60.0, {"negative_dem": True}),
]


@pytest.fixture(scope="module")
def frozen_tables():
    """All 74 frozen T03 traces, sha256-verified against the manifest."""
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


@pytest.fixture(scope="module")
def site_scene():
    """site_500 composed scene + amplitude policy (original code path)."""
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


# ---------------------------------------------------------------------------
# Gate b: bit gate over all 74 frozen traces
# ---------------------------------------------------------------------------


class TestBitGateFrozen74:
    def test_all_frozen_traces_raw_bit_equal(self, frozen_tables):
        failures = []
        checked = {"svf_shadow": 0, "wallheight_23": 0}
        for table in frozen_tables:
            key = table.key
            az = _bits_f32(key.angle_input_bits[0])
            alt = _bits_f32(key.angle_input_bits[1])
            amp = _bits_f32(key.executed_amplitude_bits)
            scale = float(np.float32(_bits_f32(key.scale_bits)))
            # scale_repr is the exact f64 actually fed to torch by T03; fall
            # back to the f32 value when reading frozen traces that lack it
            # is NOT acceptable — reconstruct from bits only when identical.
            trace_path = FROZEN_TRACES / f"{table.trace_id}.json"
            scale_repr = float(json.loads(trace_path.read_text())["scale_repr"])
            assert np.float32(scale_repr).view(np.uint32) == np.uint32(
                st.bits_hex_to_u32(key.scale_bits)
            )
            rows, cols = key.logical_rows, key.logical_cols
            a, vegdem, vegdem2, bush = synthetic_scene(rows, cols, seed=7)
            if key.kernel_variant == st.KERNEL_SVF_SHADOW:
                want = run_original_svf(
                    az, alt, scale_repr, amp, a, vegdem, vegdem2, bush
                )
                got = march_svf_shadow(
                    table,
                    a.numpy(),
                    vegdem.numpy(),
                    vegdem2.numpy(),
                    bush.numpy(),
                )
            else:
                want = run_original_wallheight(
                    az, alt, scale_repr, amp, a, vegdem, vegdem2, bush
                )
                got = march_wallheight23(
                    table,
                    a.numpy(),
                    vegdem.numpy(),
                    vegdem2.numpy(),
                    bush.numpy(),
                )
            label = f"{table.trace_id}"
            failures.extend(compare_triple(got, want, label))
            checked[key.kernel_variant] += 1
        assert not failures, "\n".join(failures[:6])
        assert checked == {"svf_shadow": 37, "wallheight_23": 37}

    def test_bit_gate_coverage_minimums(self, frozen_tables):
        """The frozen set actually exercises the contract's minimum grid."""
        def az_bits(t):
            return st.bits_hex_to_u32(t.key.angle_input_bits[0])

        zero = int(np.float32(0.0).view(np.uint32))
        below = int(
            np.nextafter(np.float32(0.0), np.float32(-np.inf)).view(np.uint32)
        )
        above = int(
            np.nextafter(np.float32(0.0), np.float32(np.inf)).view(np.uint32)
        )
        for kernel in (st.KERNEL_SVF_SHADOW, st.KERNEL_WALLHEIGHT_23):
            subset = [t for t in frozen_tables if t.key.kernel_variant == kernel]
            assert {az_bits(t) for t in subset} >= {below, zero, above}, (
                f"{kernel}: azimuth-0 nextafter neighbours missing"
            )
            alts = {
                _bits_f32(t.key.angle_input_bits[1]) for t in subset
            }
            # the frozen grid pins 6/78/90; 42 degrees comes from the
            # variant gates below (frozen fixtures use 35 for mid sun)
            assert {6.0, 78.0, 90.0} <= alts, f"{kernel}: altitudes"
            scales = {
                float(np.float32(_bits_f32(t.key.scale_bits))) for t in subset
            }
            assert {0.5, 1.0} <= scales, f"{kernel}: scales"
            shapes = {(t.key.logical_rows, t.key.logical_cols) for t in subset}
            assert any(r != c for r, c in shapes), f"{kernel}: non-square"
            assert (5, 1000) in shapes, f"{kernel}: extreme aspect fixture"
        # the variant gates complete the contract's minimum grid: 42-degree
        # altitude, scale 2.0, no-veg, no-building, overlap, negative DEM
        variant_alts = {case[1] for case in VARIANT_CASES}
        assert 42.0 in variant_alts
        variant_scales = {case[2] for case in VARIANT_CASES}
        assert 2.0 in variant_scales


# ---------------------------------------------------------------------------
# Variant gates: explicit minimum coverage beyond the frozen 74
# ---------------------------------------------------------------------------


class TestSceneVariantGates:
    # the SVF grid is the shared VARIANT_CASES; the wall-height grid swaps
    # azimuths so the two kernels' differing branch constants (python-float
    # vs f32-tensor pi multiples) are both exercised away from flat spots
    SVF_CASES = VARIANT_CASES

    WH_CASES = [
        (0.0, 78.0, 2.0, 20.0, {}),
        (45.0, 6.0, 2.0, 15.0, {}),
        (225.0, 42.0, 0.5, 20.0, {"no_vegetation": True}),
        (181.0, 42.0, 0.5, 20.0, {"no_building": True}),
        (133.0, 78.0, 1.0, 25.0, {"overlap": True}),
        (200.0, 6.0, 0.5, 30.0, {"negative_dem": True}),
        (270.0, 90.0, 0.5, 60.0, {"negative_dem": True}),
    ]

    @pytest.mark.parametrize("az,alt,scale,amp,scene_kw", SVF_CASES)
    def test_svf_variant_scenes(self, az, alt, scale, amp, scene_kw):
        a, vegdem, vegdem2, bush = synthetic_scene(40, 80, **scene_kw)
        table, _ = march_table_for(
            st.KERNEL_SVF_SHADOW, az, alt, scale, 40, 80, amp
        )
        got = march_svf_shadow(
            table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
        )
        want = run_original_svf(az, alt, scale, amp, a, vegdem, vegdem2, bush)
        failures = compare_triple(got, want, f"svf({az},{alt},{scale})")
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize("az,alt,scale,amp,scene_kw", WH_CASES)
    def test_wallheight_variant_scenes(self, az, alt, scale, amp, scene_kw):
        a, vegdem, vegdem2, bush = synthetic_scene(40, 80, **scene_kw)
        table, _ = march_table_for(
            st.KERNEL_WALLHEIGHT_23, az, alt, scale, 40, 80, amp
        )
        got = march_wallheight23(
            table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
        )
        want = run_original_wallheight(
            az, alt, scale, amp, a, vegdem, vegdem2, bush
        )
        failures = compare_triple(got, want, f"wh({az},{alt},{scale})")
        assert not failures, "\n".join(failures)

    def test_amplitude_ladder_sweep(self):
        a, vegdem, vegdem2, bush = synthetic_scene(40, 80, seed=11)
        az, alt, scale = 61.0, 30.0, 0.5
        for amp in (0.3, 1.5, 3.0, 7.7, 15.0, 31.0, 60.0):
            table, _ = march_table_for(
                st.KERNEL_SVF_SHADOW, az, alt, scale, 40, 80, amp
            )
            got = march_svf_shadow(
                table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
            )
            want = run_original_svf(az, alt, scale, amp, a, vegdem, vegdem2, bush)
            failures = compare_triple(got, want, f"sweep amp={amp}")
            assert not failures, "\n".join(failures)

    def test_one_step_vbsh_2_witness(self):
        """One-step regime + surviving first-step vegsh => vbsh == 2.0.

        The first step sets vegsh = 1 (canopy above ground by less than dz),
        the target-local trunk gate passes, then vbshvegsh.zero_() resets the
        accumulator — final ``1 - (0 - 1)`` is exactly float32 2.0. The
        accumulator must stay a float; bool-packing cannot represent this.
        """
        a, vegdem, vegdem2, bush = synthetic_scene(40, 80, seed=13)
        # canopy 5.5 m on flat ground; dz_1 = tan(78 deg)/0.25 ~ 18.8 > 5.5
        table, _ = march_table_for(
            st.KERNEL_SVF_SHADOW, 0.0, 78.0, 0.25, 40, 80, 4.0
        )
        assert table.count == 1, "fixture is not one-step"
        got = march_svf_shadow(
            table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
        )
        want = run_original_svf(0.0, 78.0, 0.25, 4.0, a, vegdem, vegdem2, bush)
        failures = compare_triple(got, want, "one-step")
        assert not failures, "\n".join(failures)
        bits2 = 0x40000000
        for plane_w, plane_g, name in zip(want, got, ("sh", "vegsh", "vbsh")):
            w_bits = plane_w.detach().cpu().numpy().view(np.uint32)
            g_bits = np.ascontiguousarray(plane_g).view(np.uint32)
            assert np.array_equal(w_bits == bits2, g_bits == bits2), name
        assert np.any(
            want[2].detach().cpu().numpy().view(np.uint32) == bits2
        ), "witness scene produced no vbsh == 2.0 cell (fixture too weak)"

    def test_extreme_aspect_shape(self):
        a, vegdem, vegdem2, bush = synthetic_scene(5, 1000, seed=17)
        for kernel, march_fn, orig in (
            (
                st.KERNEL_SVF_SHADOW,
                march_svf_shadow,
                run_original_svf,
            ),
            (
                st.KERNEL_WALLHEIGHT_23,
                march_wallheight23,
                run_original_wallheight,
            ),
        ):
            table, _ = march_table_for(kernel, 100.0, 35.0, 0.5, 5, 1000, 1e6)
            got = march_fn(
                table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
            )
            want = orig(100.0, 35.0, 0.5, 1e6, a, vegdem, vegdem2, bush)
            failures = compare_triple(got, want, f"5x1000 {kernel}")
            assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Gate c: site_500 real scene
# ---------------------------------------------------------------------------


class TestSiteSceneGate:
    SUBSET_ALTITUDES = {6.0, 42.0, 78.0, 90.0}

    def test_site_subset_altitudes_plus_escalated(self, site_scene):
        from solweig_gpu.incremental.solver import _sky_patch_geometry

        scene, policy, scale = site_scene
        rows = int(scene.a.shape[0])
        cols = int(scene.a.shape[1])
        assert (rows, cols) == (500, 500)
        a = scene.a
        vegdem = scene.vegdem
        vegdem2 = scene.vegdem2
        bush = scene.bush
        eff = _bits_f32(policy["effective_bits"])
        abs_ = _bits_f32(policy["oracle_absolute_bits"])
        escalated = set(policy["r6_banded_patch_indices"])
        patches, _rings = _sky_patch_geometry(2)
        failures = []
        checked = escalated_hits = 0
        for index, (altitude, azimuth, _ring) in enumerate(patches):
            if float(altitude) not in self.SUBSET_ALTITUDES and index not in escalated:
                continue
            amp, pid = (
                (abs_, st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED)
                if index in escalated
                else (eff, st.AMPLITUDE_EFFECTIVE_WINDOWED)
            )
            table, _trace = march_table_for(
                st.KERNEL_SVF_SHADOW,
                float(azimuth),
                float(altitude),
                scale,
                rows,
                cols,
                amp,
            )
            got = march_svf_shadow(
                table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
            )
            want = run_original_svf(
                float(azimuth), float(altitude), scale, amp, a, vegdem, vegdem2, bush
            )
            failures.extend(
                compare_triple(
                    got, want, f"site patch {index} alt={float(altitude)}"
                )
            )
            checked += 1
            escalated_hits += int(index in escalated)
        assert not failures, "\n".join(failures[:4])
        assert checked >= 20, f"only {checked} site patches covered"
        if escalated_hits == 0:
            # site_500's ORIGINAL R6 selector escalates nothing: no patch
            # first-step dz lands in (A_eff, A_abs] (78-degree dz_1 ~ 13.3 m
            # < 57.85; the zenith's tan is negative) — pinned by the empty
            # r6_banded_patch_indices above. Exercise the escalated-AMPLITUDE
            # marching regime directly: the same geometry marched at the
            # absolute scene stop must still equal the original bit-for-bit.
            assert policy["r6_banded_patch_indices"] == []
            for altitude, azimuth, _ring in patches:
                if float(altitude) not in (78.0, 90.0):
                    continue
                table, _trace = march_table_for(
                    st.KERNEL_SVF_SHADOW,
                    float(azimuth),
                    float(altitude),
                    scale,
                    rows,
                    cols,
                    abs_,
                )
                got = march_svf_shadow(
                    table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
                )
                want = run_original_svf(
                    float(azimuth), float(altitude), scale, abs_, a, vegdem, vegdem2, bush
                )
                failures_abs = compare_triple(
                    got, want, f"site absolute-amp alt={float(altitude)}"
                )
                assert not failures_abs, "\n".join(failures_abs)
                escalated_hits += 1
            assert escalated_hits >= 2, "absolute-amplitude witnesses missing"


# ---------------------------------------------------------------------------
# Kernel boundary contract (refusals, strides, zero-step)
# ---------------------------------------------------------------------------


class TestKernelContract:
    def _base(self, rows=16, cols=32):
        return synthetic_scene(rows, cols, seed=5)

    def test_bush_guard_refuses_nonzero_bush(self):
        a, vegdem, vegdem2, bush = self._base()
        bush = bush.clone()
        bush[2, 3] = 2.0
        table, _ = march_table_for(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 0.5, 16, 32, 20.0
        )
        with pytest.raises(ValueError, match="bush"):
            march_svf_shadow(
                table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
            )
        table_wh, _ = march_table_for(
            st.KERNEL_WALLHEIGHT_23, 37.0, 35.0, 0.5, 16, 32, 20.0
        )
        with pytest.raises(ValueError, match="bush"):
            march_wallheight23(
                table_wh, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
            )

    def test_negative_bush_supported(self):
        """bush.max() <= 0 keeps the original on the bush-free path."""
        a, vegdem, vegdem2, bush = self._base()
        bush = torch.full_like(bush, -1.0)
        table, _ = march_table_for(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 0.5, 16, 32, 20.0
        )
        got = march_svf_shadow(
            table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
        )
        want = run_original_svf(37.0, 35.0, 0.5, 20.0, a, vegdem, vegdem2, bush)
        assert not compare_triple(got, want, "negative bush")

    def test_variant_mismatch_refused(self):
        a, vegdem, vegdem2, bush = self._base()
        table, _ = march_table_for(
            st.KERNEL_WALLHEIGHT_23, 37.0, 35.0, 0.5, 16, 32, 20.0
        )
        with pytest.raises(ValueError, match="variant"):
            march_svf_shadow(
                table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
            )

    def test_shape_mismatch_refused(self):
        a, vegdem, vegdem2, bush = self._base()
        table, _ = march_table_for(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 0.5, 16, 32, 20.0
        )
        with pytest.raises(ValueError, match="shape"):
            march_svf_shadow(
                table,
                a.numpy()[:8],
                vegdem.numpy()[:8],
                vegdem2.numpy()[:8],
                bush.numpy()[:8],
            )

    def test_wrong_dtype_refused(self):
        a, vegdem, vegdem2, bush = self._base()
        table, _ = march_table_for(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 0.5, 16, 32, 20.0
        )
        with pytest.raises(ValueError, match="float32"):
            march_svf_shadow(
                table,
                a.numpy().astype(np.float64),
                vegdem.numpy(),
                vegdem2.numpy(),
                bush.numpy(),
            )

    def test_strided_input_bit_equal(self):
        """Strided (non C-contiguous) views must not change one output bit."""
        a, vegdem, vegdem2, bush = synthetic_scene(80, 160, seed=19)
        table, _ = march_table_for(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 0.5, 40, 80, 20.0
        )
        a_sub = a.numpy()[:40, :80]            # strides (640, 4): not C-contig
        assert not a_sub.flags["C_CONTIGUOUS"]
        got_strided = march_svf_shadow(
            table,
            a_sub,
            vegdem.numpy()[:40, :80],
            vegdem2.numpy()[:40, :80],
            bush.numpy()[:40, :80],
        )
        got_dense = march_svf_shadow(
            table,
            np.ascontiguousarray(a_sub),
            np.ascontiguousarray(vegdem.numpy()[:40, :80]),
            np.ascontiguousarray(vegdem2.numpy()[:40, :80]),
            np.ascontiguousarray(bush.numpy()[:40, :80]),
        )
        for g_s, g_d in zip(got_strided, got_dense):
            report = compare_plane_bits(
                np.ascontiguousarray(g_d), np.ascontiguousarray(g_s)
            )
            assert report["status"] == "PASS"

    def test_zero_step_table(self):
        """count == 0 (amaxvalue < 0): outputs come purely from init state.

        The table is built directly (count-0 column layout) — T03's
        ``capture_trace`` pads ``previous_dz_bits`` for empty step lists, a
        frozen-exporter quirk outside this task's writable files.
        """
        a, vegdem, vegdem2, bush = self._base()
        table = st.StepTable(
            key=st.StepTableKey(
                semantics_profile="canonical_cpu_v1",
                kernel_variant=st.KERNEL_SVF_SHADOW,
                source_geometry_hash="0" * 64,
                angle_input_bits=(st.f32_bits_hex(37.0), st.f32_bits_hex(35.0)),
                scale_bits=st.f32_bits_hex(0.5),
                logical_rows=16,
                logical_cols=32,
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
        want = run_original_svf(37.0, 35.0, 0.5, -1.0, a, vegdem, vegdem2, bush)
        got = march_svf_shadow(
            table, a.numpy(), vegdem.numpy(), vegdem2.numpy(), bush.numpy()
        )
        want = run_original_svf(37.0, 35.0, 0.5, -1.0, a, vegdem, vegdem2, bush)
        assert not compare_triple(got, want, "zero-step")


# ---------------------------------------------------------------------------
# Gate e: kernel purity
# ---------------------------------------------------------------------------


class TestPurity:
    def test_no_torch_and_no_fastmath_true(self):
        source = (
            REPO_ROOT / "solweig_core" / "numba_cpu" / "march.py"
        ).read_text()
        assert "import torch" not in source
        assert "from torch" not in source
        assert "fastmath=True" not in source
        assert "parallel=True" not in source
        assert "prange" not in source

    def test_module_torch_free_subprocess(self):
        code = (
            "import sys; "
            "import numpy as np; "
            "from solweig_core import step_tables as st; "
            "from solweig_core.numba_cpu.march import march_svf_shadow; "
            "assert 'torch' not in sys.modules; "
            "table = st.build_table_from_trace({"
            "'schema_version': 1, "
            "'key': {'semantics_profile': 'canonical_cpu_v1', "
            "'kernel_variant': 'svf_shadow', "
            "'source_geometry_hash': 'x'*64, "
            "'angle_input_bits': ['0x42140000','0x420c0000'], "
            "'scale_bits': '0x3f000000', "
            "'logical_rows': 4, 'logical_cols': 8, "
            "'amplitude_policy_id': 'synthetic_probe', "
            "'executed_amplitude_bits': '0x41a00000', "
            "'boundary_policy_id': 'shadow_shift_window_v1'}, "
            "'count': 0, 'dx': [], 'dy': [], 'dz_bits': [], "
            "'stop_reason': 'amplitude', "
            "'azimuth_zero_substituted': False, 'trace_id': 'purity'}); "
            "a = np.zeros((4, 8), dtype=np.float32); "
            "out = march_svf_shadow(table, a, a.copy(), a.copy(), a.copy()); "
            "assert out[0].shape == (4, 8) and out[0].dtype == np.float32"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
