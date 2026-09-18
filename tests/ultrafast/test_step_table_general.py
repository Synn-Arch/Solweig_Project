# SPDX-License-Identifier: GPL-3.0-only
"""T14 Change A (R3) gates: the GENERAL torch-free step-table producer.

``solweig_core.numba_cpu.step_table_gen.build_table_general`` reproduces,
without torch, the executed march step sequences of BOTH kernel variants
for ARBITRARY (azimuth, altitude, scale, rows, cols, amplitude) — the
"unseen angles" seam the T03 producer explicitly did not claim
(``build_table_from_trace`` is trace-derived only).

Gate structure (one bounded change, red-first):

* **Torch-freedom** — the producer imports and builds tables in a fresh
  interpreter with ``torch`` importable but NOT imported (the dev-env
  trap direction; a transitive import would land in ``sys.modules``).
* **Frozen-trace byte gate** — every one of the 74 T03 fixture traces
  (captured from the ORIGINAL torch code path) is reproduced bit-exactly:
  count, dx, dy, dz_bits, stop_reason, branch_id.
* **Unseen-angle differential (dev env)** — against the ORIGINAL
  ``march_offsets`` (svf) and the T03-verified torch engine
  ``trace_exporter._generate_steps`` (both variants) over an adversarial
  grid: dense azimuths, nextafter neighbours of every cardinal/diagonal,
  non-power-of-two scales, non-square grids, one-step regimes, the
  zenith, amplitude ladders — dx/dy/count equality and raw dz-bit
  equality, plus the nextafter amplitude-ladder stop probe on unseen
  amplitude-limited marches.
* **Mutations** — five value-sensitive producer mutations, each caught
  by a designated gate (frozen byte gate or the unseen differential).

The producer's pinned ``SOURCE_GEOMETRY_HASH`` must equal the live hash
of the defining oracle sources (drift gate: editing them without
re-pinning is a hard failure, matching T03's key-invalidation design).
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

from solweig_core import step_tables as st  # noqa: E402
from solweig_core.numba_cpu import step_table_gen as gen  # noqa: E402

try:  # dev-env only references (the runtime never imports these)
    import torch  # noqa: F401

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="dev env (torch)")

if HAS_TORCH:
    from trace_exporter import (  # noqa: E402
        _generate_steps,
        capture_fixture_traces,
        source_geometry_hash,
    )
    from solweig_gpu.incremental.veg_svf_state import march_offsets  # noqa: E402


F32 = np.float32


def _case_of(trace: dict) -> dict:
    """Producer kwargs equivalent to one captured fixture trace."""
    key = trace["key"]
    return dict(
        kernel_variant=key["kernel_variant"],
        azimuth_deg=float(st.u32_bits_to_f32(st.bits_hex_to_u32(key["angle_input_bits"][0]))),
        altitude_deg=float(st.u32_bits_to_f32(st.bits_hex_to_u32(key["angle_input_bits"][1]))),
        scale=float(trace["scale_repr"]),
        rows=int(key["logical_rows"]),
        cols=int(key["logical_cols"]),
        amplitude=float(st.u32_bits_to_f32(st.bits_hex_to_u32(key["executed_amplitude_bits"]))),
        amplitude_policy_id=key["amplitude_policy_id"],
        source_geometry_hash=key["source_geometry_hash"],
    )


# ---------------------------------------------------------------------------
# Torch-freedom (RED witness #1 of the T14 card, producer-scoped)
# ---------------------------------------------------------------------------


class TestTorchFreedom:
    def test_imports_and_builds_without_torch(self):
        """Fresh interpreter, torch importable but never imported: the
        producer module imports, builds a table, and torch stays out of
        sys.modules (a transitive torch import would appear there)."""
        code = (
            "import sys\n"
            "from solweig_core.numba_cpu import step_table_gen as gen\n"
            "t = gen.build_table_general(\n"
            "    gen.KERNEL_SVF_SHADOW if hasattr(gen, 'KERNEL_SVF_SHADOW')"
            " else 'svf_shadow',\n"
            "    37.0, 35.0, 0.5, 40, 80, 20.0)\n"
            "assert t.count > 0\n"
            "assert 'torch' not in sys.modules, 'torch imported transitively'\n"
            "assert not any(m.startswith('solweig_gpu') for m in sys.modules)\n"
            "print('OK', t.count, t.content_digest())\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("OK"), result.stdout

    def test_module_dependency_surface(self):
        """The producer module pulls nothing beyond the torch-free core
        stack (numpy + solweig_core.numba_cpu family)."""
        code = (
            "import sys\n"
            "from solweig_core.numba_cpu import step_table_gen\n"
            "tops = {m.split('.')[0] for m in sys.modules}\n"
            "assert 'torch' not in tops\n"
            "assert 'solweig_gpu' not in tops\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Frozen-trace byte gate (74 fixtures, both variants)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_traces():
    if not HAS_TORCH:
        pytest.skip("fixture capture needs the dev env (torch)")
    return capture_fixture_traces()


@pytest.mark.skipif(not HAS_TORCH, reason="dev env (torch)")
class TestFrozenTraceByteGate:
    def test_every_fixture_reproduced_bit_exact(self, fixture_traces):
        failures = []
        for trace in fixture_traces:
            table = gen.build_table_general(**_case_of(trace))
            reference = st.build_table_from_trace(trace)
            report = st.compare_step_tables(reference, table)
            if not report["all_equal"] or report["mismatch_count"]:
                failures.append(
                    (trace["trace_id"], report["first_mismatches"][:2])
                )
            else:
                assert table.stop_reason == trace["stop_reason"]
                assert list(table.branch_id) == list(trace["branch_id"])
                assert bool(table.azimuth_zero_substituted) == bool(
                    trace["azimuth_zero_substituted"]
                )
        assert not failures, f"{len(failures)} fixtures diverge: {failures[:3]}"

    def test_fixture_count_and_variant_coverage(self, fixture_traces):
        variants = {t["key"]["kernel_variant"] for t in fixture_traces}
        assert variants == {st.KERNEL_SVF_SHADOW, st.KERNEL_WALLHEIGHT_23}
        assert len(fixture_traces) >= 74
        overshoot = sum(
            1
            for t in fixture_traces
            if st.u32_bits_to_f32(st.bits_hex_to_u32(t["dz_bits"][-1]))
            > st.u32_bits_to_f32(
                st.bits_hex_to_u32(t["key"]["executed_amplitude_bits"])
            )
        )
        assert overshoot >= 10, "overshoot-step coverage collapsed"

    def test_wallheight_self_step_structure(self, fixture_traces):
        for trace in fixture_traces:
            if trace["key"]["kernel_variant"] != st.KERNEL_WALLHEIGHT_23:
                continue
            table = gen.build_table_general(**_case_of(trace))
            assert int(table.dx[0]) == 0 and int(table.dy[0]) == 0
            # the structural self step: dz == (ds*0)*tbs == ±0.0 (the sign
            # is the tbs sign — the zenith's negative tan yields -0.0,
            # bits 0x80000000, exactly as the torch engine produces)
            assert int(table.dz_bits[0]) & 0x7FFFFFFF == 0

    def test_default_hash_is_pinned_constant(self, fixture_traces):
        """Producer default hash == the live oracle-source hash (nothing
        has drifted since pinning)."""
        live, _sources = source_geometry_hash()
        assert gen.SOURCE_GEOMETRY_HASH == live

    def test_digest_is_pure_function_of_payload(self, fixture_traces):
        for trace in fixture_traces[:10]:
            one = gen.build_table_general(**_case_of(trace))
            two = gen.build_table_general(**_case_of(trace))
            assert one.content_digest() == two.content_digest()
            assert one.key == two.key


# ---------------------------------------------------------------------------
# Unseen-angle differential (dev env): producer vs ORIGINAL torch code
# ---------------------------------------------------------------------------


def _unseen_grid() -> list[dict]:
    """Adversarial unseen (az, alt, scale, rows, cols, amp) tuples."""
    cases: list[dict] = []

    def add(az, alt, scale, rows, cols, amp):
        cases.append(
            dict(
                azimuth_deg=float(F32(az)),
                altitude_deg=float(F32(alt)),
                scale=scale,
                rows=rows,
                cols=cols,
                amplitude=float(F32(amp)),
            )
        )

    # structured edges: nextafter windows of every cardinal + diagonal
    for deg in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0):
        v = F32(deg)
        for nudge in (-1, 0, 1):
            az = float(np.nextafter(v, F32(np.inf))) if nudge > 0 else (
                float(np.nextafter(v, F32(-np.inf))) if nudge < 0 else float(v)
            )
            add(az, 35.0, 0.5, 37, 53, 20.0)
            add(az, 78.0, 0.4, 37, 53, 20.0)
    # near-miss quadrants around the diagonals (branch flip zones)
    for deg in (45.0, 135.0, 225.0, 315.0):
        add(deg - 1e-4, 35.0, 1.0 / 3.0, 64, 32, 40.0)
        add(deg + 1e-4, 35.0, 1.0 / 3.0, 64, 32, 40.0)
        add(deg - 1e-6, 6.0, 2.0, 5, 1000, 1e6)
        add(deg + 1e-6, 6.0, 2.0, 5, 1000, 1e6)
    # one-step / low-sun / zenith regimes
    add(37.0, 89.9, 0.25, 40, 80, 4.0)
    add(181.3, 78.0, 0.5, 40, 80, 0.5)
    add(271.7, 90.0, 0.5, 40, 80, 1e6)
    add(90.0, 90.0, 0.5, 40, 80, 1e6)
    # seeded random unseen angles (reproducible)
    rng = np.random.default_rng(20260907)
    for _ in range(160):
        add(
            float(rng.uniform(0.0, 360.0)),
            float(rng.uniform(0.5, 90.0)),
            float(rng.choice([0.25, 1.0 / 3.0, 0.4, 0.5, 0.8, 1.0, 2.0])),
            int(rng.integers(3, 96)),
            int(rng.integers(3, 96)),
            float(rng.choice([0.5, 3.3, 12.0, 40.0, 250.0])),
        )
    return cases


_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        _GRID = _unseen_grid()
    return _GRID


@needs_torch
class TestUnseenAngleDifferential:
    def test_svf_matches_march_offsets(self):
        """dx/dy/count equality vs the ORIGINAL march_offsets replica
        (itself proven bit-for-bit against shadow()'s while-loop)."""
        checked = 0
        for case in _grid():
            table = gen.build_table_general(
                st.KERNEL_SVF_SHADOW,
                case["azimuth_deg"],
                case["altitude_deg"],
                case["scale"],
                case["rows"],
                case["cols"],
                case["amplitude"],
                amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
            )
            original = march_offsets(
                torch.tensor(F32(case["azimuth_deg"])),
                torch.tensor(F32(case["altitude_deg"])),
                case["amplitude"],
                case["scale"],
                case["rows"],
                case["cols"],
            )
            mine = [(int(x), int(y)) for x, y in zip(table.dx, table.dy)]
            want = [(int(dx), int(dy)) for dx, dy in original]
            assert mine == want, (
                f"svf offsets diverge at {case}: {mine[:5]} vs {want[:5]}"
            )
            checked += 1
        assert checked >= 200

    def test_matches_torch_engine_both_variants(self):
        """Full step-tuple equality (index, dx, dy, dz BITS, branch) vs the
        T03-verified torch engine, BOTH kernel flavors."""
        checked = 0
        for case in _grid():
            for variant in st.KERNEL_VARIANTS:
                steps, _sub = _generate_steps(
                    variant,
                    case["azimuth_deg"],
                    case["altitude_deg"],
                    case["scale"],
                    case["rows"],
                    case["cols"],
                    case["amplitude"],
                )
                table = gen.build_table_general(
                    variant,
                    case["azimuth_deg"],
                    case["altitude_deg"],
                    case["scale"],
                    case["rows"],
                    case["cols"],
                    case["amplitude"],
                    amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
                )
                assert table.count == len(steps), (
                    f"{variant} count diverge at {case}: "
                    f"{table.count} vs {len(steps)}"
                )
                for i, (index, dx, dy, dz, branch) in enumerate(steps):
                    assert int(table.dx[i]) == dx, (case, variant, i, "dx")
                    assert int(table.dy[i]) == dy, (case, variant, i, "dy")
                    assert (
                        int(table.dz_bits[i])
                        == F32(dz).view(np.uint32).item()
                    ), (case, variant, i, "dz bits",
                        hex(int(table.dz_bits[i])),
                        hex(F32(dz).view(np.uint32).item()))
                    assert int(table.branch_id[i]) == branch, (
                        case, variant, i, "branch"
                    )
                checked += 1
        assert checked >= 400

    def test_amplitude_ladder_nextafter_probe(self):
        """Unseen amplitude-limited svf marches: every dz_k is pinned by the
        ORIGINAL stop behaviour — count(dz_k) >= k+1 AND
        count(nextafter(dz_k, -inf)) <= k (march_offsets, oversized bounds).
        """
        probed = 0
        for case in _grid()[:: 7]:  # bounded runtime; every 7th case
            table = gen.build_table_general(
                st.KERNEL_SVF_SHADOW,
                case["azimuth_deg"],
                case["altitude_deg"],
                case["scale"],
                case["rows"],
                case["cols"],
                case["amplitude"],
                amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
            )
            if table.count == 0:
                continue
            if abs(int(table.dx[-1])) >= case["rows"] or abs(
                int(table.dy[-1])
            ) >= case["cols"]:
                continue  # boundary-limited: the ladder is not observable
            def count_at(amp: float) -> int:
                return len(
                    march_offsets(
                        torch.tensor(F32(case["azimuth_deg"])),
                        torch.tensor(F32(case["altitude_deg"])),
                        amp,
                        case["scale"],
                        1 << 20,
                        1 << 20,
                    )
                )
            for k in range(1, min(table.count, 5) + 1):
                candidate = F32(0).view(np.uint32)
                candidate = table.dz_bits[k - 1]
                cand_f = float(np.uint32(candidate).view(np.float32))
                below = float(
                    np.nextafter(
                        np.uint32(candidate).view(np.float32),
                        F32(-np.inf),
                    )
                )
                n_at = count_at(cand_f)
                n_below = count_at(below)
                assert n_at >= k + 1 and n_below <= k, (
                    f"dz_{k} bits {hex(int(candidate))} not pinned by the "
                    f"original stop at {case}: counts {n_at}/{n_below}"
                )
                probed += 1
        assert probed >= 20, f"only {probed} ladder probes ran"

    def test_substitution_flag_semantics(self):
        for case in _grid()[:60]:
            table = gen.build_table_general(
                st.KERNEL_SVF_SHADOW,
                case["azimuth_deg"],
                case["altitude_deg"],
                case["scale"],
                case["rows"],
                case["cols"],
                case["amplitude"],
            )
            expect = F32(case["azimuth_deg"]) == F32(0.0)
            assert bool(table.azimuth_zero_substituted) == bool(expect
                ), case
            wh = gen.build_table_general(
                st.KERNEL_WALLHEIGHT_23,
                case["azimuth_deg"],
                case["altitude_deg"],
                case["scale"],
                case["rows"],
                case["cols"],
                case["amplitude"],
            )
            assert wh.azimuth_zero_substituted is False

    def test_validation_refusals(self):
        with pytest.raises(ValueError):
            gen.build_table_general(
                st.KERNEL_SVF_SHADOW, 37.0, 0.0, 0.5, 40, 80, 20.0
            )
        with pytest.raises(ValueError):
            gen.build_table_general(
                st.KERNEL_SVF_SHADOW, 37.0, -5.0, 0.5, 40, 80, 20.0
            )
        with pytest.raises(ValueError):
            gen.build_table_general("not_a_kernel", 37.0, 35.0, 0.5, 40, 80, 20.0)


# ---------------------------------------------------------------------------
# Producer mutations (each caught by a designated gate)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="dev env (torch)")
class TestProducerMutations:
    """Five value-sensitive mutations of the producer, each proving a
    designated gate bites (frozen byte gate unless stated)."""

    def test_mutation_svf_branch_constant_flavor(self, fixture_traces, monkeypatch):
        """MUTATION: svf 5*pi/4 bound computed in the WALLHEIGHT flavor
        (f32 tensor arithmetic, 0x407B53D2) instead of the svf python-float
        flavor (0x407B53D1) -> the az=225 sector flips -> caught."""
        monkeypatch.setattr(
            gen,
            "SVF_FIVEPIBYFOUR",
            F32(F32(np.pi / 4.0) * F32(5.0)),
        )
        caught = 0
        for trace in fixture_traces:
            az = st.u32_bits_to_f32(
                st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
            )
            if float(az) != 225.0 or trace["key"]["kernel_variant"] != st.KERNEL_SVF_SHADOW:
                continue
            table = gen.build_table_general(**_case_of(trace))
            reference = st.build_table_from_trace(trace)
            report = st.compare_step_tables(reference, table)
            assert not report["all_equal"], (
                "f32-flavor svf branch constant survived at az=225"
            )
            caught += 1
        assert caught >= 1, "no az=225 svf fixture to catch the mutation"

    def test_mutation_wallheight_index0(self, fixture_traces, monkeypatch):
        """MUTATION: wallheight march starts at index 1 (svf convention)
        -> the structural (0, 0) self step disappears -> the StepTable
        structural fence refuses loudly (designated killer) — a mutated
        table can never be constructed."""
        monkeypatch.setattr(gen, "WALLHEIGHT_INDEX0", 1.0)
        kills = 0
        for trace in fixture_traces:
            if trace["key"]["kernel_variant"] != st.KERNEL_WALLHEIGHT_23:
                continue
            with pytest.raises(ValueError, match="self step"):
                gen.build_table_general(**_case_of(trace))
            kills += 1
        assert kills >= 1

    def test_mutation_dz_reassociation(self, fixture_traces, monkeypatch):
        """MUTATION: dz computed as (ds*index*tan(alt))/scale instead of
        (ds*index)*(tan(alt)/scale) -> 1-ulp dz drift at non-power-of-two
        scales -> caught by the frozen byte gate."""
        monkeypatch.setattr(gen, "DZ_ASSOCIATION", "scale_last")
        differing = 0
        for trace in fixture_traces:
            if trace["key"]["kernel_variant"] != st.KERNEL_SVF_SHADOW:
                continue
            scale = float(trace["scale_repr"])
            if scale in (0.5, 1.0, 2.0, 4.0, 0.25):
                continue  # power-of-two scales cannot separate the forms
            table = gen.build_table_general(**_case_of(trace))
            reference = st.build_table_from_trace(trace)
            report = st.compare_step_tables(reference, table)
            if list(table.dz_bits) != list(reference.dz_bits):
                differing += 1
                assert not report["all_equal"], "re-association survived"
        assert differing >= 1, "no non-power-of-two svf fixture separated"

    def test_mutation_degree_conversion_f64_once(self, monkeypatch):
        """MUTATION: degree->radian conversion computed in f64 and rounded
        once, instead of the kernels' f32*f32(pi/180) — diverges by 1 ulp
        on ~8% of angles -> caught by the UNSEEN-angle differential vs the
        torch engine (designated killer)."""
        monkeypatch.setattr(gen, "ANGLE_CONVERSION", "f64_once")
        caught = 0
        for case in _grid():
            steps, _sub = _generate_steps(
                st.KERNEL_SVF_SHADOW,
                case["azimuth_deg"],
                case["altitude_deg"],
                case["scale"],
                case["rows"],
                case["cols"],
                case["amplitude"],
            )
            table = gen.build_table_general(
                st.KERNEL_SVF_SHADOW,
                case["azimuth_deg"],
                case["altitude_deg"],
                case["scale"],
                case["rows"],
                case["cols"],
                case["amplitude"],
            )
            if table.count == len(steps) and list(table.dz_bits) == [
                F32(s[3]).view(np.uint32).item() for s in steps
            ] and list(table.dx) == [s[1] for s in steps]:
                continue
            caught += 1
            assert caught <= 400  # divergence expected, not a storm
        assert caught >= 1, "f64-once conversion survived every unseen angle"

    def test_azimuth_zero_substitution_is_f32_invisible(self, fixture_traces, monkeypatch):
        """DOCUMENTED LOCAL FLATNESS (mirrors the T03 cardinal-window
        finding): at azimuth exactly 0 the march takes the COS branch,
        where f32 rounding erases the substitution epsilon — dscos rounds
        to 1.0 and rint(k*tan(tiny)) is 0 for every k the amplitude
        admits. So NO value-sensitive mutation exists on AZ_SUBSTITUTE;
        the observable is the azimuth_zero_substituted flag itself, which
        must stay faithful (asserted here)."""
        monkeypatch.setattr(gen, "AZ_SUBSTITUTE", 1e-6)
        flagged = 0
        for trace in fixture_traces:
            az = float(
                st.u32_bits_to_f32(
                    st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
                )
            )
            if az != 0.0 or trace["key"]["kernel_variant"] != st.KERNEL_SVF_SHADOW:
                continue
            table = gen.build_table_general(**_case_of(trace))
            reference = st.build_table_from_trace(trace)
            report = st.compare_step_tables(reference, table)
            assert report["all_equal"], (
                "epsilon 1e-6 unexpectedly changed the az=0 march — the "
                "documented flatness broke; update this witness"
            )
            assert table.azimuth_zero_substituted is True
            flagged += 1
        assert flagged >= 1, "no az=0 svf fixture to pin the flag"

    def test_mutation_drop_abs_on_round(self, fixture_traces, monkeypatch):
        """MUTATION: |round(...)| loses its abs -> dy sign flips wherever
        round() returns a negative offset (tan < 0 quadrants: az in
        (90, 180) U (270, 360)) -> caught at the 135/315-diagonal and
        random-quadrant fixtures."""
        monkeypatch.setattr(gen, "ABS_ON_ROUND", False)
        caught = 0
        for trace in fixture_traces:
            if trace["key"]["kernel_variant"] != st.KERNEL_SVF_SHADOW:
                continue
            az = float(
                st.u32_bits_to_f32(
                    st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
                )
            )
            if not (90.0 < az < 180.0 or 270.0 < az < 360.0):
                continue
            table = gen.build_table_general(**_case_of(trace))
            reference = st.build_table_from_trace(trace)
            if list(table.dy) == list(reference.dy):
                continue  # e.g. sin-branch marches where round() never signs
            report = st.compare_step_tables(reference, table)
            assert not report["all_equal"], "abs-drop mutation survived"
            caught += 1
        assert caught >= 1, "no negative-tan quadrant fixture caught it"
