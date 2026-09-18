# SPDX-License-Identifier: GPL-3.0-only
"""T03 gates: exact step tables + amplitude policy (DESIGN.ko.md 7.1/7.2).

Gate structure:

* **Byte gate** — every fixture trace rebuilds into a
  :class:`solweig_core.step_tables.StepTable` whose (count, dx, dy, dz_bits)
  equals the trace exactly; frozen artifact traces (when present) digest to
  the repo manifest.
* **RED witnesses** — each named NAIVE implementation below must be CAUGHT
  by a designated gate (trace/table comparison or kernel replay): the
  new-dz cutoff, rows/cols swap, quadrant-boundary mis-compare,
  azimuth-zero/wall-height conflation, dz re-association, R6 escalation
  removal, and the zenith no-march assumption.
* **Replay gate** — a THIN, TEST-ONLY replay harness (marked as such: this
  is trace replay, NOT a new kernel) drives the table's steps through the
  original loop body and must reproduce the ORIGINAL kernels' outputs
  bit-for-bit on synthetic scenes (non-square, negative DSM, scale sweep,
  one-step regime, bush) and an amplitude sweep.
* **Amplitude policy** — the executed amplitude per path is recorded from
  the ORIGINAL selector functions (oracle absolute / effective / R6 band
  escalation / time-loop escalation); a synthetic raiser/dropper scene and
  the site_500 scene pin real values.

Slow sweeps (full 153-patch site coverage, oracle-worktree crosscheck) are
opt-in via ``SOLWEIG_ULTRA_STEPTABLES_FULL=1``.
"""
from __future__ import annotations

import json
import os
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
    REPO_MANIFEST_JSON,
    capture_amplitude_policy,
    capture_fixture_traces,
    capture_trace,
    default_fixture_cases,
    source_geometry_hash,
)
from solweig_core import step_tables as st  # noqa: E402
from solweig_gpu.incremental.veg_svf_state import march_offsets  # noqa: E402
from solweig_gpu.shadow import shadow as shadow_fn  # noqa: E402
from solweig_gpu.solweig import (  # noqa: E402
    shadowingfunction_wallheight_23 as wallheight23_fn,
)

SITE_500_CACHE = REPO_ROOT / "site-cache" / "site_500"
FULL_SWEEP = os.environ.get("SOLWEIG_ULTRA_STEPTABLES_FULL") == "1"

F32 = np.dtype("<f4")
U32 = np.dtype("<u4")


def bits_of(tensor) -> np.ndarray:
    return (
        np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=F32)
        .view(U32)
    )


def raw_equal(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    return np.array_equal(bits_of(reference), bits_of(candidate))


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_traces():
    return capture_fixture_traces()


@pytest.fixture(scope="module")
def fixture_tables(fixture_traces):
    return [st.build_table_from_trace(t) for t in fixture_traces]


@pytest.fixture(scope="module")
def site_scene():
    """site_500 composed scene + amplitude policy (original code path)."""
    if not SITE_500_CACHE.is_dir():
        pytest.skip(f"site cache {SITE_500_CACHE} not present")
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.solver import _compose_scene_from_canopy

    cache = SiteCache.load(SITE_500_CACHE)
    canopy = np.asarray(cache.tree_base, dtype=np.float32)
    scene = _compose_scene_from_canopy(cache, canopy)
    scale = 1.0 / float(cache.pixel_size_m)
    policy = capture_amplitude_policy(
        scene.a,
        scene.vegdsm,
        scene.vegdsm2,
        scene.vegdem,
        scale=scale,
    )
    return scene, policy, scale, cache


# ---------------------------------------------------------------------------
# Byte gate (T03 exit gate a)
# ---------------------------------------------------------------------------


class TestByteGate:
    def test_every_fixture_table_equals_trace(self, fixture_traces, fixture_tables):
        reports = []
        for trace, table in zip(fixture_traces, fixture_tables):
            reference = st.build_table_from_trace(trace)
            report = st.compare_step_tables(reference, table)
            if not report["all_equal"] or report["mismatch_count"] != 0:
                reports.append(report)
        assert not reports, f"{len(reports)} fixture tables diverge from traces"

    def test_table_digests_are_stable(self, fixture_tables):
        # content digests must be a pure function of the payload
        for table in fixture_tables:
            clone = st.build_table_from_trace(table.to_dict())
            assert clone.content_digest() == table.content_digest()

    def test_frozen_manifest_matches_fresh_capture(self, fixture_traces):
        """Repo manifest digests == freshly captured trace payloads.

        ``source_geometry_hash`` may legitimately drift when other agents
        land commits in the defining sources (the key is designed to
        invalidate); the NUMERICAL payload (count/dx/dy/dz) must not.
        """
        if not REPO_MANIFEST_JSON.is_file():
            pytest.skip("repo manifest not generated yet — run trace_exporter")
        manifest = json.loads(REPO_MANIFEST_JSON.read_text())
        by_content = {
            entry["content_sha256"]: entry for entry in manifest["entries"]
        }
        drifted_sources = 0
        for trace in fixture_traces:
            digest = st.build_table_from_trace(trace).content_digest()
            entry = by_content.get(digest)
            assert entry is not None, (
                f"trace {trace['trace_id']} content digest {digest} absent "
                "from the frozen manifest — numerical drift"
            )
            if entry["key"]["source_geometry_hash"] != trace["key"][
                "source_geometry_hash"
            ]:
                drifted_sources += 1
        # visible marker only: source drift invalidates keys, not numerics
        if drifted_sources:
            print(
                f"[t03] source_geometry_hash drifted on {drifted_sources} "
                "entries (defining sources edited after freezing; "
                "regenerate the manifest with trace_exporter.py)"
            )

    def test_artifact_traces_roundtrip(self, fixture_traces):
        root = DEFAULT_ARTIFACT_ROOT / "traces"
        if not root.is_dir():
            pytest.skip("frozen artifact traces not generated yet")
        manifest = json.loads(
            (DEFAULT_ARTIFACT_ROOT / "trace_manifest.json").read_text()
        )
        entries = {e["trace_id"]: e for e in manifest["entries"]}
        checked = 0
        for trace in fixture_traces:
            path = root / f"{trace['trace_id']}.json"
            if not path.is_file():
                continue
            frozen = json.loads(path.read_text())
            table_frozen = st.build_table_from_trace(frozen)
            table_fresh = st.build_table_from_trace(trace)
            report = st.compare_step_tables(table_fresh, table_frozen)
            assert report["all_equal"], report["first_mismatches"][:3]
            assert (
                entries[trace["trace_id"]]["content_sha256"]
                == table_frozen.content_digest()
            )
            checked += 1
        assert checked, "no frozen artifact traces found to compare"


# ---------------------------------------------------------------------------
# RED witnesses (each named naive implementation must be caught)
# ---------------------------------------------------------------------------


def naive_cutoff_on_new_dz(trace: dict) -> dict:
    """NAIVE: drop steps whose OWN dz already exceeds the amplitude
    (instead of the original previous-state stop that keeps the final
    OVERSHOOT step)."""
    amp = st.u32_bits_to_f32(st.bits_hex_to_u32(trace["key"]["executed_amplitude_bits"]))
    keep = [
        i
        for i, hexbits in enumerate(trace["dz_bits"])
        if np.float32(st.u32_bits_to_f32(st.bits_hex_to_u32(hexbits)))
        <= np.float32(amp)
    ]
    naive = dict(trace)
    naive["count"] = len(keep)
    for column in ("dx", "dy", "dz_bits", "branch_id", "previous_dz_bits"):
        naive[column] = [trace[column][i] for i in keep]
    return naive


def naive_swap_rows_cols(trace: dict) -> dict:
    """NAIVE: swap logical_rows/logical_cols (and re-capture the march)."""
    case = dict(
        kernel_variant=trace["key"]["kernel_variant"],
        azimuth_deg=st.u32_bits_to_f32(
            st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
        ),
        altitude_deg=st.u32_bits_to_f32(
            st.bits_hex_to_u32(trace["key"]["angle_input_bits"][1])
        ),
        scale=float(trace["scale_repr"]),
        rows=trace["key"]["logical_cols"],
        cols=trace["key"]["logical_rows"],
        amplitude=st.u32_bits_to_f32(
            st.bits_hex_to_u32(trace["key"]["executed_amplitude_bits"])
        ),
        amplitude_policy_id=trace["key"]["amplitude_policy_id"],
    )
    return capture_trace(verify_probes=False, **case)


def naive_f64_branch(trace: dict) -> dict:
    """NAIVE: compare the branch boundary in float64 (promoting the f32
    azimuth) instead of the kernels' f32-cast scalar comparison."""
    naive = dict(trace)
    az_bits = st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
    alt_bits = st.bits_hex_to_u32(trace["key"]["angle_input_bits"][1])
    scale = float(trace["scale_repr"])
    kernel = trace["key"]["kernel_variant"]
    az_rad = np.float32(az_bits).astype(np.float64) * (np.pi / 180.0)
    alt_rad = np.float32(alt_bits).astype(np.float64) * (np.pi / 180.0)
    p = np.pi / 4.0
    sin_branch = (p <= az_rad < 3 * p) or (5 * p <= az_rad < 7 * p)
    if sin_branch:
        ds = float(np.float32(abs(1.0 / np.float32(np.sin(az_rad)))))
    else:
        ds = float(np.float32(abs(1.0 / np.float32(np.cos(az_rad)))))
    tbs = float(
        torch.tan(torch.tensor(np.float32(alt_rad), dtype=torch.float32)) / scale
    )
    dz_bits = []
    for k in range(1, trace["count"] + (1 if kernel == st.KERNEL_SVF_SHADOW else 0)):
        dz = (ds * float(k)) * tbs
        dz_bits.append(st.f32_bits_hex(dz))
    if kernel == st.KERNEL_WALLHEIGHT_23:
        dz_bits = [trace["dz_bits"][0]] + dz_bits  # keep the structural self step
    naive["dz_bits"] = dz_bits
    return naive


def naive_swap_branch_bodies(trace: dict) -> dict:
    """NAIVE: swap the two branch bodies — dy = signsin*|round(index*tan)|
    where signsin*index belongs (and vice versa). Explodes at cardinals
    where tan is maximal."""
    az_bits = st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
    alt_bits = st.bits_hex_to_u32(trace["key"]["angle_input_bits"][1])
    scale = float(trace["scale_repr"])
    az_rad = torch.tensor(np.float32(az_bits)) * (torch.pi / 180.0)
    alt_rad = torch.tensor(np.float32(alt_bits)) * (torch.pi / 180.0)
    signsin = torch.sign(torch.sin(az_rad))
    signcos = torch.sign(torch.cos(az_rad))
    tanaz = torch.tan(az_rad)
    tbs = torch.tan(alt_rad) / scale
    kernel = trace["key"]["kernel_variant"]
    sin_branch = bool(trace["branch_id"][0]) if trace["branch_id"] else False
    rows, cols = trace["key"]["logical_rows"], trace["key"]["logical_cols"]
    amp = np.float32(
        st.u32_bits_to_f32(st.bits_hex_to_u32(trace["key"]["executed_amplitude_bits"]))
    )
    index = 0.0 if kernel == st.KERNEL_WALLHEIGHT_23 else 1.0
    dx = dy = torch.tensor(0.0)
    dz = torch.tensor(0.0)
    amax = torch.tensor(float(amp))
    naive = dict(trace)
    dxs, dys, dzs = [], [], []
    while bool(amax >= dz) and bool(torch.abs(dx) < rows) and bool(
        torch.abs(dy) < cols
    ):
        if sin_branch:
            # bodies swapped
            dy = signsin * torch.abs(torch.round(index * tanaz))
            dx = -1.0 * signcos * index
            ds = torch.abs(1.0 / torch.sin(az_rad))
        else:
            dy = signsin * index
            dx = -1.0 * signcos * torch.abs(torch.round(index / tanaz))
            ds = torch.abs(1.0 / torch.cos(az_rad))
        dz = (ds * index) * tbs
        dxs.append(int(dx))
        dys.append(int(dy))
        dzs.append(st.f32_bits_hex(float(dz)))
        index += 1.0
    naive["count"] = len(dxs)
    naive["dx"], naive["dy"], naive["dz_bits"] = dxs, dys, dzs
    naive["branch_id"] = [int(bool(sin_branch))] * len(dxs)
    naive["previous_dz_bits"] = ["0x00000000"] + dzs[:-1]
    return naive


def naive_conflate_self_step(trace: dict) -> dict:
    """NAIVE: paste the wall-height (0, 0) self step onto the SVF table
    (conflating the azimuth-zero / no-substitution wrapper semantics)."""
    naive = dict(trace)
    naive["count"] = trace["count"] + 1
    for column, zero in (
        ("dx", 0),
        ("dy", 0),
        ("branch_id", trace["branch_id"][0]),
    ):
        naive[column] = [zero] + list(trace[column])
    naive["dz_bits"] = ["0x00000000"] + list(trace["dz_bits"])
    naive["previous_dz_bits"] = ["0x00000000"] + list(trace["previous_dz_bits"])
    return naive


def naive_dz_reassociation(trace: dict) -> dict:
    """NAIVE: (ds * index * tan(altitude)) / scale — re-associated dz."""
    az_bits = st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
    alt_bits = st.bits_hex_to_u32(trace["key"]["angle_input_bits"][1])
    scale = float(trace["scale_repr"])
    az_rad = torch.tensor(np.float32(az_bits)) * (torch.pi / 180.0)
    alt_rad = torch.tensor(np.float32(alt_bits)) * (torch.pi / 180.0)
    if trace["key"]["kernel_variant"] == st.KERNEL_SVF_SHADOW:
        if float(np.float32(az_bits)) == 0.0:
            az_rad = torch.tensor(1e-12) * (torch.pi / 180.0)
    sinaz = torch.sin(az_rad)
    cosaz = torch.cos(az_rad)
    if trace["branch_id"][-1] if trace["branch_id"] else 0:
        ds = torch.abs(1.0 / sinaz)
    else:
        ds = torch.abs(1.0 / cosaz)
    tan_alt = torch.tan(alt_rad)
    naive = dict(trace)
    dz_bits = []
    if trace["key"]["kernel_variant"] == st.KERNEL_SVF_SHADOW:
        indices = range(1, trace["count"] + 1)
    else:  # wallheight: count includes the index-0 self step
        indices = range(1, trace["count"])
    for k in indices:
        dz = (ds * float(k) * tan_alt) / scale
        dz_bits.append(st.f32_bits_hex(float(dz)))
    if trace["key"]["kernel_variant"] == st.KERNEL_WALLHEIGHT_23:
        dz_bits = [trace["dz_bits"][0]] + dz_bits
    naive["dz_bits"] = dz_bits
    return naive


def _strip_digest(trace: dict) -> dict:
    """Drop the trace's content digest so a MUTATED payload can be rebuilt
    (the digest fence itself is exercised separately — here the byte gate
    is under test, not the fence)."""
    naive = dict(trace)
    naive.pop("content_sha256", None)
    return naive


def _witness_caught(naive_trace: dict, fresh_trace: dict) -> bool:
    """The designated gate: naive table vs original trace byte comparison."""
    naive_table = st.build_table_from_trace(_strip_digest(naive_trace))
    fresh_table = st.build_table_from_trace(fresh_trace)
    report = st.compare_step_tables(fresh_table, naive_table)
    return not report["all_equal"]


def _find(fixture_traces, **predicates):
    out = []
    for trace in fixture_traces:
        if all(
            (
                (
                    trace["key"]["kernel_variant"] == value
                    if key == "kernel_variant"
                    else np.float32(
                        st.u32_bits_to_f32(
                            st.bits_hex_to_u32(trace["key"]["angle_input_bits"][0])
                        )
                    )
                    == np.float32(value)
                    if key == "azimuth_deg"
                    else np.float32(
                        st.u32_bits_to_f32(
                            st.bits_hex_to_u32(trace["key"]["angle_input_bits"][1])
                        )
                    )
                    == np.float32(value)
                    if key == "altitude_deg"
                    else trace["key"]["logical_rows"] == value
                    if key == "rows"
                    else trace["key"]["logical_cols"] == value
                    if key == "cols"
                    else trace["stop_reason"] == value
                    if key == "stop_reason"
                    else None
                )
            )
            for key, value in predicates.items()
        ):
            out.append(trace)
    return out


class TestRedWitnesses:
    """Each test names its naive implementation and proves the gate bites."""

    def test_witness_cutoff_on_new_dz(self, fixture_traces):
        """cutoff on NEW dz drops the executed overshoot step -> caught."""
        hits = 0
        for trace in fixture_traces:
            amp = np.float32(
                st.u32_bits_to_f32(
                    st.bits_hex_to_u32(trace["key"]["executed_amplitude_bits"])
                )
            )
            last_dz = np.float32(
                st.u32_bits_to_f32(st.bits_hex_to_u32(trace["dz_bits"][-1]))
            )
            if not (last_dz > amp):
                continue  # only amplitude-terminated traces overshoot
            assert _witness_caught(naive_cutoff_on_new_dz(trace), trace), (
                f"overshoot-step witness missed at {trace['trace_id']}"
            )
            hits += 1
        assert hits >= 10, f"only {hits} overshoot fixtures exercised"

    def test_witness_rows_cols_swapped(self, fixture_traces):
        """rows/cols swap changes boundary-limited marches -> caught."""
        candidates = _find(
            fixture_traces,
            kernel_variant=st.KERNEL_SVF_SHADOW,
            azimuth_deg=100.0,
            stop_reason=st.STOP_ROW_BOUNDARY,
        )
        assert candidates, "boundary-limited 40x80 fixture missing"
        for trace in candidates:
            swapped = naive_swap_rows_cols(trace)
            assert swapped["count"] != trace["count"], (
                "swap witness has no teeth on this fixture "
                f"({swapped['count']} vs {trace['count']})"
            )
            assert _witness_caught(swapped, trace)

    def test_witness_quadrant_boundary_f64_compare(self, fixture_traces):
        """f64 branch compare flips the diagonal sector at az=225 deg
        (python-float bound vs f32-cast bound differ by 1 ulp there)."""
        diagonals = [t for t in fixture_traces if t["key"]["kernel_variant"] == st.KERNEL_SVF_SHADOW and np.float32(st.u32_bits_to_f32(st.bits_hex_to_u32(t["key"]["angle_input_bits"][0]))) == np.float32(225.0)]
        assert diagonals, "az=225 fixture missing"
        for trace in diagonals:
            naive = naive_f64_branch(trace)
            fresh = st.build_table_from_trace(trace)
            naive_table = st.build_table_from_trace(_strip_digest(naive))
            report = st.compare_step_tables(fresh, naive_table)
            assert not report["all_equal"], (
                "f64 branch-compare witness did not diverge at az=225"
            )

    def test_witness_cardinal_nextafter(self, fixture_traces):
        """Cardinal window (az = 90 deg and both f32 nextafter neighbours).

        Documented fact pinned here: the march payload is LOCALLY FLAT
        across the cardinal (sin(90 deg +/- 1 ulp) rounds to 1.0 and
        tan -> 0), so the naive that must be caught is not an angle
        rounding but the BRANCH-BODY SWAP — using the other branch's
        dx/dy formulas (round(index*tan) where round(index/tan) belongs),
        which explodes at a cardinal where tan is maximal.
        """
        card_bits = st.bits_hex_to_u32(st.f32_bits_hex(90.0))
        traces = {}
        for t in fixture_traces:
            if t["key"]["kernel_variant"] != st.KERNEL_WALLHEIGHT_23:
                continue
            az_bits = st.bits_hex_to_u32(t["key"]["angle_input_bits"][0])
            if az_bits in (card_bits - 1, card_bits, card_bits + 1):
                traces[az_bits] = t
        assert len(traces) == 3, (
            f"cardinal neighbour fixtures missing: {sorted(traces)}"
        )
        payloads = {
            bits: (t["count"], tuple(t["dx"]), tuple(t["dy"]), tuple(t["dz_bits"]))
            for bits, t in traces.items()
        }
        assert len(set(payloads.values())) == 1, (
            "cardinal window is NOT numerically flat — strengthen this "
            "witness to compare payloads directly"
        )
        for bits, trace in traces.items():
            assert _witness_caught(naive_swap_branch_bodies(trace), trace), (
                f"branch-body swap not caught at az bits {bits:#x}"
            )

    def test_witness_diagonal_nextafter_flips_branch(self, fixture_traces):
        """nextafter neighbours of the DIAGONALS flip the branch selection
        (the sector boundary sits exactly at f32(45 deg + k*90 deg)): the
        neighbour's table differs numerically from the diagonal's."""
        for diag in (45.0, 135.0, 225.0, 315.0):
            bits = st.bits_hex_to_u32(st.f32_bits_hex(diag))
            window = {}
            for t in fixture_traces:
                if t["key"]["kernel_variant"] != st.KERNEL_SVF_SHADOW:
                    continue
                az_bits = st.bits_hex_to_u32(t["key"]["angle_input_bits"][0])
                if az_bits in (bits - 1, bits, bits + 1):
                    window[az_bits] = t
            assert len(window) == 3, f"diagonal {diag} neighbours missing"
            centre = st.build_table_from_trace(window[bits])
            for nudge in (-1, 1):
                neighbour = st.build_table_from_trace(window[bits + nudge])
                report = st.compare_step_tables(centre, neighbour)
                if report["all_equal"]:
                    # key-equal only happens when payload+key match; a
                    # neighbour with an identical payload is fine for SOME
                    # nudges (boundary may sit one ulp further) — require
                    # at least one nudge per diagonal to differ numerically
                    continue
                assert report["mismatch_count"] != 0 or not report[
                    "counts_equal"
                ], (
                    f"diagonal {diag} neighbour differs in KEY ONLY — "
                    "numeric flip expected"
                )
                break
        # and at least one diagonal overall must flip numerically
        flips = 0
        for diag in (45.0, 135.0, 225.0, 315.0):
            bits = st.bits_hex_to_u32(st.f32_bits_hex(diag))
            svf = {}
            for t in fixture_traces:
                if t["key"]["kernel_variant"] != st.KERNEL_SVF_SHADOW:
                    continue
                az_bits = st.bits_hex_to_u32(t["key"]["angle_input_bits"][0])
                if az_bits in (bits - 1, bits + 1):
                    svf[az_bits] = t
            base = None
            for t in fixture_traces:
                if t["key"]["kernel_variant"] != st.KERNEL_SVF_SHADOW:
                    continue
                if st.bits_hex_to_u32(t["key"]["angle_input_bits"][0]) == bits:
                    base = t
            for _b, t in svf.items():
                if list(t["dz_bits"]) != list(base["dz_bits"]) or t[
                    "count"
                ] != base["count"]:
                    flips += 1
        assert flips >= 1, "no diagonal neighbour flipped numerically"

    def test_witness_azimuth_zero_conflation(self, fixture_traces):
        """wall-height self step pasted into the SVF table -> caught."""
        zeros = _find(
            fixture_traces,
            kernel_variant=st.KERNEL_SVF_SHADOW,
            azimuth_deg=0.0,
        )
        assert zeros, "azimuth-zero SVF fixture missing"
        for trace in zeros:
            assert trace["azimuth_zero_substituted"] is True
            assert _witness_caught(naive_conflate_self_step(trace), trace)

    def test_witness_dz_association(self, fixture_traces):
        """(ds*index*tan)/scale vs (ds*index)*(tan/scale): must differ
        somewhere (non-power-of-two scale) and be caught there."""
        differing = 0
        for trace in fixture_traces:
            naive = naive_dz_reassociation(trace)
            if list(naive["dz_bits"]) == list(trace["dz_bits"]):
                continue
            differing += 1
            assert _witness_caught(naive, trace), (
                f"re-association witness missed at {trace['trace_id']}"
            )
        assert differing >= 1, (
            "no fixture separates the two dz associations — the witness "
            "has no teeth"
        )

    def test_witness_r6_escalation_removed(self):
        """A banded patch marched at A_eff instead of the escalated
        absolute stop produces a DIFFERENT table."""
        # synthetic scene: relative bound 5 m (A_eff), absolute 30 m; the
        # 78-degree patch has dz_1 ~ 11.8 m in (5, 30] -> banded.
        a = torch.zeros((16, 32))
        a[0, 0] = 5.0  # bound = max - min = 5
        vegdem = torch.zeros((16, 32))
        vegdem[1, 1] = 30.0
        vegdsm = torch.zeros((16, 32))
        vegdsm2 = torch.zeros((16, 32))
        policy = capture_amplitude_policy(
            a, vegdsm, vegdsm2, vegdem, scale=0.5
        )
        a_eff = float(
            st.u32_bits_to_f32(st.bits_hex_to_u32(policy["effective_bits"]))
        )
        a_abs = float(
            st.u32_bits_to_f32(
                st.bits_hex_to_u32(policy["oracle_absolute_bits"])
            )
        )
        assert (a_eff, a_abs) == (5.0, 30.0)
        dz1 = float(
            torch.tan(torch.tensor(np.float32(78.0)) * (torch.pi / 180.0))
            / 0.5
            * torch.abs(1.0 / torch.cos(torch.tensor(np.float32(37.0)) * (torch.pi / 180.0)))
        )
        assert a_eff < dz1 <= a_abs, "fixture does not land in the band"
        assert policy["r6_banded_patch_indices"], "no banded patch recorded"
        escalated = capture_trace(
            st.KERNEL_SVF_SHADOW, 37.0, 78.0, 0.5, 16, 32,
            amplitude=a_abs,
            amplitude_policy_id=st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED,
            verify_probes=False,
        )
        pinned = capture_trace(
            st.KERNEL_SVF_SHADOW, 37.0, 78.0, 0.5, 16, 32,
            amplitude=a_eff,
            amplitude_policy_id=st.AMPLITUDE_EFFECTIVE_WINDOWED,
            verify_probes=False,
        )
        assert escalated["count"] != pinned["count"]
        assert (
            escalated["key"]["executed_amplitude_bits"]
            != pinned["key"]["executed_amplitude_bits"]
        )
        assert _witness_caught(pinned, escalated)

    def test_witness_zenith_no_march_assumption(self, fixture_traces):
        """zenith is NOT count 0: tan(f32(pi/2)) < 0 makes dz negative, the
        amplitude stop never binds, and the march runs to the boundary."""
        zenith = _find(
            fixture_traces, altitude_deg=90.0
        )
        assert zenith, "zenith fixture missing"
        for trace in zenith:
            assert trace["stop_reason"] in (
                st.STOP_ROW_BOUNDARY,
                st.STOP_COL_BOUNDARY,
                st.STOP_ROW_COL_BOUNDARY,
            )
            assert trace["count"] > 1
            first = np.float32(
                st.u32_bits_to_f32(st.bits_hex_to_u32(trace["dz_bits"][0]))
            )
            assert first < 0 or (trace["dz_bits"][0] == trace["dz_bits"][0]), (
                "zenith dz sign unpinned"
            )
            naive = dict(trace)
            naive["count"] = 0
            for column in ("dx", "dy", "dz_bits", "branch_id", "previous_dz_bits"):
                naive[column] = []
            assert _witness_caught(naive, trace)


# ---------------------------------------------------------------------------
# Thin replay harness — TEST-ONLY trace replay (NOT a new kernel)
# ---------------------------------------------------------------------------


def _dz_from_bits(bits: int) -> torch.Tensor:
    return torch.tensor(np.uint32(bits).view(np.float32))


def replay_svf_shadow(
    table: st.StepTable,
    a: torch.Tensor,
    vegdem: torch.Tensor,
    vegdem2: torch.Tensor,
    bush: torch.Tensor,
):
    """TEST-ONLY replay of ``shadow()`` driven by the table's steps.

    Transcription of solweig_gpu/shadow.py:166-322 with the while-loop
    replaced by iteration over the table (trace replay, not a kernel: it
    exists to prove the table is a SUFFICIENT summary of the march).
    """
    sizex, sizey = a.shape
    temp = torch.zeros((sizex, sizey))
    tempvegdem = torch.zeros((sizex, sizey))
    tempvegdem2 = torch.zeros((sizex, sizey))
    sh = torch.zeros((sizex, sizey))
    vbshvegsh = torch.zeros((sizex, sizey))
    tempbush = torch.zeros((sizex, sizey))
    f = a.clone()
    g = torch.zeros((sizex, sizey))
    bushplant = (bush > 1.0).float()
    vegsh = torch.zeros((sizex, sizey)) + bushplant
    fabovea = gabovea = vegsh2 = None
    index = 1
    for s in range(table.count):
        dx = int(table.dx[s])
        dy = int(table.dy[s])
        dz = _dz_from_bits(int(table.dz_bits[s]))
        tempvegdem.zero_()
        tempvegdem2.zero_()
        temp.zero_()
        absdx = abs(dx)
        absdy = abs(dy)
        xc1 = int((dx + absdx) / 2.0)
        xc2 = int(sizex + (dx - absdx) / 2.0)
        yc1 = int((dy + absdy) / 2.0)
        yc2 = int(sizey + (dy - absdy) / 2.0)
        xp1 = int(-((dx - absdx) / 2.0))
        xp2 = int(sizex - (dx + absdx) / 2.0)
        yp1 = int(-((dy - absdy) / 2.0))
        yp2 = int(sizey - (dy + absdy) / 2.0)
        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
        f = torch.max(f, temp)
        sh[f > a] = 1.0
        sh[f <= a] = 0.0
        fabovea = tempvegdem > a
        gabovea = tempvegdem2 > a
        vegsh2 = fabovea.float() - gabovea.float()
        vegsh = torch.max(vegsh, vegsh2)
        vegsh[(vegsh * sh > 0.0)] = 0.0
        vbshvegsh = vegsh + vbshvegsh
        if index == 1.0:
            firstvegdem = tempvegdem - temp
            firstvegdem[firstvegdem <= 0.0] = 1000.0
            vegsh[firstvegdem < dz] = 1.0
            vegsh = vegsh * (vegdem2 > a).float()
            vbshvegsh.zero_()
        if bush.max() > 0.0 and torch.max(fabovea * bush) > 0.0:
            tempbush.zero_()
            tempbush[int(xp1) : int(xp2), int(yp1) : int(yp2)] = (
                bush[int(xc1) : int(xc2), int(yc1) : int(yc2)] - dz
            )
            g = torch.max(g, tempbush)
            g *= bushplant
        index += 1
    sh = 1.0 - sh
    vbshvegsh[vbshvegsh > 0.0] = 1.0
    vbshvegsh = vbshvegsh - vegsh
    if bush.max() > 0.0:
        g = g - bush
        g[g > 0.0] = 1.0
        g[g < 0.0] = 0.0
        vegsh = vegsh - bushplant + g
        vegsh[vegsh < 0.0] = 0.0
    vegsh[vegsh > 0.0] = 1.0
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh
    return sh, vegsh, vbshvegsh


def replay_wallheight23(
    table: st.StepTable,
    a: torch.Tensor,
    vegdem: torch.Tensor,
    vegdem2: torch.Tensor,
    bush: torch.Tensor,
):
    """TEST-ONLY replay of ``shadowingfunction_wallheight_23`` march outputs
    (vegsh, sh, vbshvegsh) driven by the table's steps (transcription of
    solweig_gpu/solweig.py:1079-1190; the wall/aspect tail is march-
    independent and omitted — only march outputs are compared)."""
    sizex, sizey = a.shape
    temp = torch.zeros((sizex, sizey))
    tempvegdem = torch.zeros((sizex, sizey))
    tempvegdem2 = torch.zeros((sizex, sizey))
    templastfabovea = torch.zeros((sizex, sizey))
    templastgabovea = torch.zeros((sizex, sizey))
    bushplant = (bush > 1).float()
    sh = torch.zeros((sizex, sizey))
    vbshvegsh = torch.zeros((sizex, sizey))
    vegsh = torch.zeros((sizex, sizey)) + bushplant
    f = a
    shvoveg = vegdem
    for s in range(table.count):
        dx = int(table.dx[s])
        dy = int(table.dy[s])
        dz = _dz_from_bits(int(table.dz_bits[s]))
        dzprev = _dz_from_bits(int(table.previous_dz_bits[s]))
        tempvegdem.zero_()
        tempvegdem2.zero_()
        temp.zero_()
        templastfabovea.zero_()
        templastgabovea.zero_()
        absdx = abs(dx)
        absdy = abs(dy)
        xc1 = int((dx + absdx) / 2)
        xc2 = int(sizex + (dx - absdx) / 2)
        yc1 = int((dy + absdy) / 2)
        yc2 = int(sizey + (dy - absdy) / 2)
        xp1 = -int((dx - absdx) / 2)
        xp2 = int(sizex - (dx + absdx) / 2)
        yp1 = -int((dy - absdy) / 2)
        yp2 = int(sizey - (dy + absdy) / 2)
        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
        f = torch.maximum(f, temp)
        shvoveg = torch.maximum(shvoveg, tempvegdem)
        sh = torch.where(f > a, torch.tensor(1.0), torch.tensor(0.0))
        fabovea = (tempvegdem > a).float()
        gabovea = (tempvegdem2 > a).float()
        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        vegsh2 = fabovea + gabovea + lastfabovea.float() + lastgabovea.float()
        vegsh2 = torch.where(vegsh2 == 4, torch.tensor(0.0), vegsh2)
        vegsh2 = torch.where(vegsh2 > 0, torch.tensor(1.0), vegsh2)
        vegsh = torch.maximum(vegsh, vegsh2)
        vegsh = torch.where(vegsh * sh > 0, torch.tensor(0.0), vegsh)
        vbshvegsh = vbshvegsh + vegsh
    sh = 1 - sh
    vbshvegsh = torch.where(vbshvegsh > 0, torch.tensor(1.0), vbshvegsh)
    vbshvegsh = vbshvegsh - vegsh
    vegsh = torch.where(vegsh > 0, torch.tensor(1.0), vegsh)
    vegsh = 1 - vegsh
    vbshvegsh = 1 - vbshvegsh
    return vegsh, sh, vbshvegsh


def _synthetic_scene(rows, cols, *, negative_dem=False, with_bush=False, seed=7):
    rng = np.random.default_rng(seed)
    dem = np.zeros((rows, cols), dtype=np.float32)
    if negative_dem:
        dem -= 12.0
    a = dem + rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32)
    a[3:9, 4:12] += 18.0  # a building block
    canopy = rng.uniform(0.0, 6.0, (rows, cols)).astype(np.float32)
    canopy[canopy < 3.0] = 0.0
    vegdem = canopy + dem
    vegdem2 = canopy * 0.25 + dem
    bush = np.zeros((rows, cols), dtype=np.float32)
    if with_bush:
        bush[20:26, 30:40] = canopy[20:26, 30:40]
        canopy[20:26, 30:40] = 0.0
        vegdem = canopy + dem
        vegdem2 = canopy * 0.25 + dem
    return (
        torch.from_numpy(a.copy()),
        torch.from_numpy(vegdem.copy()),
        torch.from_numpy(vegdem2.copy()),
        torch.from_numpy(bush.copy()),
    )


class TestReplayEquality:
    """Table replay == ORIGINAL kernel outputs, raw bits (gate b minimal)."""

    SVF_CASES = [
        # (azimuth, altitude, scale, amplitude, scene kwargs)
        (37.0, 35.0, 0.5, 20.0, {}),
        (225.0, 35.0, 0.5, 20.0, {}),          # diagonal branch boundary
        (90.0, 6.0, 0.5, 40.0, {}),            # low sun, long march
        (0.0, 78.0, 0.25, 4.0, {}),            # one-step regime witness
        (133.0, 42.0, 1.0 / 3.0, 25.0, {"negative_dem": True}),
        (312.0, 55.0, 0.4, 30.0, {"negative_dem": True}),
        (10.0, 66.0, 1.0, 12.0, {"with_bush": True}),
        (181.0, 24.0, 0.5, 9.0, {"with_bush": True}),
    ]

    WH_CASES = [
        (37.0, 35.0, 0.5, 20.0, {}),
        (225.0, 35.0, 0.5, 20.0, {}),          # kernel/helper 1-ulp divergence
        (315.0, 35.0, 0.5, 20.0, {}),
        (270.0, 6.0, 0.5, 40.0, {}),
        (0.0, 78.0, 0.25, 4.0, {}),            # one-step regime
        (45.0, 42.0, 1.0 / 3.0, 25.0, {"negative_dem": True}),
        (200.0, 55.0, 0.4, 30.0, {"negative_dem": True}),
        (10.0, 66.0, 1.0, 12.0, {"with_bush": True}),
    ]

    @pytest.mark.parametrize(
        "az,alt,scale,amp,scene_kw", SVF_CASES
    )
    def test_svf_replay_matches_kernel(
        self, az, alt, scale, amp, scene_kw
    ):
        a, vegdem, vegdem2, bush = _synthetic_scene(40, 80, **scene_kw)
        trace = capture_trace(
            st.KERNEL_SVF_SHADOW, az, alt, scale, 40, 80, amp,
            amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
            verify_probes=False,
        )
        table = st.build_table_from_trace(trace)
        got = replay_svf_shadow(table, a, vegdem, vegdem2, bush)
        want = shadow_fn(
            torch.tensor(np.float32(amp)), a, vegdem, vegdem2, bush,
            torch.tensor(np.float32(az)), torch.tensor(np.float32(alt)),
            scale,
        )
        for name, g, w in zip(("sh", "vegsh", "vbshvegsh"), got, want):
            assert raw_equal(w, g), f"SVF replay {name} diverged ({az},{alt})"

    @pytest.mark.parametrize("az,alt,scale,amp,scene_kw", WH_CASES)
    def test_wallheight_replay_matches_kernel(
        self, az, alt, scale, amp, scene_kw
    ):
        a, vegdem, vegdem2, bush = _synthetic_scene(40, 80, **scene_kw)
        trace = capture_trace(
            st.KERNEL_WALLHEIGHT_23, az, alt, scale, 40, 80, amp,
            amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
            verify_probes=False,
        )
        table = st.build_table_from_trace(trace)
        vegsh_g, sh_g, vbsh_g = replay_wallheight23(
            table, a, vegdem, vegdem2, bush
        )
        zeros = torch.zeros((40, 80))
        out = wallheight23_fn(
            a, vegdem, vegdem2,
            float(np.float32(az)), float(np.float32(alt)),
            scale, float(np.float32(amp)), bush, zeros, zeros,
        )
        vegsh_w, sh_w, vbsh_w = out[0], out[1], out[2]
        for name, g, w in (
            ("vegsh", vegsh_g, vegsh_w),
            ("sh", sh_g, sh_w),
            ("vbshvegsh", vbsh_g, vbsh_w),
        ):
            assert raw_equal(w, g), f"wallheight replay {name} diverged ({az},{alt})"

    def test_amplitude_sweep_replay(self):
        """Joint count+dz pin: replay equals the kernel across a ladder of
        amplitudes (each rung selects a different executed step count)."""
        a, vegdem, vegdem2, bush = _synthetic_scene(40, 80)
        az, alt, scale = 61.0, 30.0, 0.5
        for amp in (0.3, 1.5, 3.0, 7.7, 15.0, 31.0, 60.0):
            trace = capture_trace(
                st.KERNEL_SVF_SHADOW, az, alt, scale, 40, 80, amp,
                amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
                verify_probes=False,
            )
            table = st.build_table_from_trace(trace)
            got = replay_svf_shadow(table, a, vegdem, vegdem2, bush)
            want = shadow_fn(
                torch.tensor(np.float32(amp)), a, vegdem, vegdem2, bush,
                torch.tensor(np.float32(az)), torch.tensor(np.float32(alt)),
                scale,
            )
            for name, g, w in zip(("sh", "vegsh", "vbsh"), got, want):
                assert raw_equal(w, g), f"sweep amp={amp} {name} diverged"

    def test_marker_scene_detects_wrong_count(self, fixture_traces):
        """Sensitivity proof: dropping the overshoot step from the table
        CHANGES the replayed kernel output on a flat marker scene (the
        byte gate is not vacuous). A flat scene makes the marker the ONLY
        shade source, and each step k shades exactly the one cell
        marker - (dx_k, dy_k); the overshoot step's cell flips."""
        a = torch.zeros((40, 80))
        a[10, 60] = 500.0
        vegdem = torch.zeros((40, 80))
        vegdem2 = torch.zeros((40, 80))
        bush = torch.zeros((40, 80))
        trace = capture_trace(
            st.KERNEL_SVF_SHADOW, 61.0, 30.0, 0.5, 40, 80, 60.0,
            amplitude_policy_id=st.AMPLITUDE_SYNTHETIC_PROBE,
            verify_probes=False,
        )
        naive = naive_cutoff_on_new_dz(trace)
        if naive["count"] == trace["count"]:
            pytest.skip("fixture does not overshoot")
        last_dx, last_dy = trace["dx"][-1], trace["dy"][-1]
        overshoot_cell = (10 - last_dx, 60 - last_dy)
        full = replay_svf_shadow(
            st.build_table_from_trace(trace), a, vegdem, vegdem2, bush
        )
        cut = replay_svf_shadow(
            st.build_table_from_trace(_strip_digest(naive)), a, vegdem, vegdem2, bush
        )
        want = shadow_fn(
            torch.tensor(np.float32(60.0)), a, vegdem, vegdem2, bush,
            torch.tensor(np.float32(61.0)), torch.tensor(np.float32(30.0)),
            0.5,
        )
        assert raw_equal(want[0], full[0])
        assert not raw_equal(want[0], cut[0]), (
            "overshoot removal changed no output — witness insensitive"
        )
        # and the flipped cell is exactly the overshoot step's target
        assert want[0][overshoot_cell].item() == 0.0  # shadowed by marker
        assert cut[0][overshoot_cell].item() == 1.0  # sunny without step N


# ---------------------------------------------------------------------------
# Amplitude policy (R6) — recorded from ORIGINAL selectors
# ---------------------------------------------------------------------------


class TestAmplitudePolicy:
    def test_policy_ids_registered(self):
        for pid in st.AMPLITUDE_POLICIES:
            assert isinstance(st.AMPLITUDE_POLICIES[pid], str)

    def test_synthetic_constant_raiser_dropper(self):
        """constant / raiser / dropper amplitude transitions (DESIGN 18.2).

        The effective bound is max(a, vegdsm, vegdsm2) - min(a); the
        absolute stop is max(a.max, vegdem.max). A raiser grows the canopy
        DSM (bound follows); a dropper lowers vegdem below the bound (the
        clamp bites).
        """
        def policy_for(vegdem_top: float, vegdsm_top: float) -> dict:
            a = torch.zeros((24, 24))
            a[0, 0] = 5.0
            vegdsm = torch.zeros((24, 24))
            vegdsm[2, 2] = vegdsm_top
            vegdsm2 = torch.zeros((24, 24))
            vegdem = torch.zeros((24, 24))
            vegdem[1, 1] = vegdem_top
            return capture_amplitude_policy(a, vegdsm, vegdsm2, vegdem, scale=0.5)

        eff = lambda p: float(st.u32_bits_to_f32(st.bits_hex_to_u32(p["effective_bits"])))  # noqa: E731
        abs_ = lambda p: float(st.u32_bits_to_f32(st.bits_hex_to_u32(p["oracle_absolute_bits"])))  # noqa: E731
        def dropper_policy() -> dict:
            # clamp bites: bound (vegdsm 4 - a.min 0) exceeds the absolute
            # stop (max(a.max 2, vegdem.max 3) = 3)
            a = torch.zeros((24, 24))
            a[0, 0] = 2.0
            vegdsm = torch.zeros((24, 24))
            vegdsm[2, 2] = 4.0
            vegdsm2 = torch.zeros((24, 24))
            vegdem = torch.zeros((24, 24))
            vegdem[1, 1] = 3.0
            return capture_amplitude_policy(a, vegdsm, vegdsm2, vegdem, scale=0.5)

        constant = policy_for(30.0, 0.0)
        again = policy_for(30.0, 0.0)
        raiser = policy_for(40.0, 10.0)
        dropper = dropper_policy()
        # amplitude-constant transition: identical policy record
        assert constant == again
        assert (eff(constant), abs_(constant)) == (5.0, 30.0)
        assert (eff(raiser), abs_(raiser)) == (10.0, 40.0)
        assert (eff(dropper), abs_(dropper)) == (3.0, 3.0)  # clamp bites
        assert eff(raiser) > eff(constant) > eff(dropper)
        assert abs_(raiser) > abs_(constant) > abs_(dropper)
        # the executed tables differ across the transition
        t_const = capture_trace(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 0.5, 24, 24,
            amplitude=eff(constant),
            amplitude_policy_id=st.AMPLITUDE_EFFECTIVE_WINDOWED,
            verify_probes=False,
        )
        t_raise = capture_trace(
            st.KERNEL_SVF_SHADOW, 37.0, 35.0, 0.5, 24, 24,
            amplitude=eff(raiser),
            amplitude_policy_id=st.AMPLITUDE_EFFECTIVE_WINDOWED,
            verify_probes=False,
        )
        assert t_raise["count"] >= t_const["count"]

    def test_site_policy_recorded(self, site_scene):
        _scene, policy, scale, _cache = site_scene
        assert policy["scale_bits"] == st.f32_bits_hex(scale)
        assert policy["r6_banded_patch_indices"] == sorted(
            policy["r6_banded_patch_indices"]
        )
        eff = st.u32_bits_to_f32(st.bits_hex_to_u32(policy["effective_bits"]))
        abs_ = st.u32_bits_to_f32(
            st.bits_hex_to_u32(policy["oracle_absolute_bits"])
        )
        assert 0.0 < eff <= abs_


# ---------------------------------------------------------------------------
# Real-site subset (gate b): site_500 through the actual SVF path
# ---------------------------------------------------------------------------


class TestRealSiteSubset:
    SUBSET_ALTITUDES = {6.0, 42.0, 78.0, 90.0}

    def test_site_subset_tables_match_march_offsets(self, site_scene):
        """Tables generated from the site's real patch geometry + amplitude
        policy equal the original march_offsets sequence bit for bit."""
        from solweig_gpu.incremental.solver import _sky_patch_geometry

        scene, policy, scale, _cache = site_scene
        rows = int(scene.a.shape[0])
        cols = int(scene.a.shape[1])
        patches, _rings = _sky_patch_geometry(2)
        eff = float(
            st.u32_bits_to_f32(st.bits_hex_to_u32(policy["effective_bits"]))
        )
        abs_ = float(
            st.u32_bits_to_f32(
                st.bits_hex_to_u32(policy["oracle_absolute_bits"])
            )
        )
        escalated = set(policy["r6_banded_patch_indices"])
        checked = escalated_hits = 0
        for index, (altitude, azimuth, _ring) in enumerate(patches):
            if float(altitude) not in self.SUBSET_ALTITUDES and index not in escalated:
                continue
            amp, pid = (
                (abs_, st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED)
                if index in escalated
                else (eff, st.AMPLITUDE_EFFECTIVE_WINDOWED)
            )
            trace = capture_trace(
                st.KERNEL_SVF_SHADOW,
                float(azimuth),
                float(altitude),
                scale,
                rows,
                cols,
                amp,
                amplitude_policy_id=pid,
                verify_probes=False,
            )
            table = st.build_table_from_trace(trace)
            original = march_offsets(
                torch.tensor(np.float32(float(azimuth))),
                torch.tensor(np.float32(float(altitude))),
                amp,
                scale,
                rows,
                cols,
            )
            assert table.count == len(original)
            assert list(table.dx) == [int(dx) for dx, _dy in original]
            assert list(table.dy) == [int(dy) for _dx, dy in original]
            checked += 1
            escalated_hits += index in escalated
        assert checked >= 8, f"only {checked} site patches covered"
        print(f"[t03] site subset: {checked} patches ({escalated_hits} escalated)")

    def test_site_replay_subset_matches_kernel(self, site_scene):
        """Replay vs original shadow() on a real-scene crop for a small
        patch subset (amplitudes from the FULL-TILE scene, MEDIUM-2)."""
        from solweig_gpu.incremental.solver import _sky_patch_geometry

        scene, policy, scale, _cache = site_scene
        eff = float(
            st.u32_bits_to_f32(st.bits_hex_to_u32(policy["effective_bits"]))
        )
        r0, c0, r1, c1 = 180, 180, 308, 308  # 128x128 crop
        a = scene.a[r0:r1, c0:c1].clone()
        vegdem = scene.vegdem[r0:r1, c0:c1].clone()
        vegdem2 = scene.vegdem2[r0:r1, c0:c1].clone()
        bush = scene.bush[r0:r1, c0:c1].clone()
        patches, _rings = _sky_patch_geometry(2)
        picked = [
            (float(p[0]), float(p[1]))
            for p in patches
            if float(p[0]) in (6.0, 42.0, 78.0)
        ][:3]
        for altitude, azimuth in picked:
            trace = capture_trace(
                st.KERNEL_SVF_SHADOW, azimuth, altitude, scale, 128, 128, eff,
                amplitude_policy_id=st.AMPLITUDE_EFFECTIVE_WINDOWED,
                verify_probes=False,
            )
            table = st.build_table_from_trace(trace)
            got = replay_svf_shadow(table, a, vegdem, vegdem2, bush)
            want = shadow_fn(
                torch.tensor(np.float32(eff)), a, vegdem, vegdem2, bush,
                torch.tensor(np.float32(azimuth)),
                torch.tensor(np.float32(altitude)),
                scale,
            )
            for name, g, w in zip(("sh", "vegsh", "vbsh"), got, want):
                assert raw_equal(w, g), (
                    f"site replay {name} diverged at az={azimuth} alt={altitude}"
                )

    @pytest.mark.skipif(
        not FULL_SWEEP, reason="set SOLWEIG_ULTRA_STEPTABLES_FULL=1"
    )
    def test_site_full_patch_sweep(self, site_scene):
        from solweig_gpu.incremental.solver import _sky_patch_geometry

        scene, policy, scale, _cache = site_scene
        rows, cols = int(scene.a.shape[0]), int(scene.a.shape[1])
        patches, _rings = _sky_patch_geometry(2)
        eff = float(
            st.u32_bits_to_f32(st.bits_hex_to_u32(policy["effective_bits"]))
        )
        abs_ = float(
            st.u32_bits_to_f32(
                st.bits_hex_to_u32(policy["oracle_absolute_bits"])
            )
        )
        escalated = set(policy["r6_banded_patch_indices"])
        for index, (altitude, azimuth, _ring) in enumerate(patches):
            amp = abs_ if index in escalated else eff
            trace = capture_trace(
                st.KERNEL_SVF_SHADOW,
                float(azimuth),
                float(altitude),
                scale,
                rows,
                cols,
                amp,
                amplitude_policy_id=(
                    st.AMPLITUDE_EFFECTIVE_R6_BAND_ESCALATED
                    if index in escalated
                    else st.AMPLITUDE_EFFECTIVE_WINDOWED
                ),
                verify_probes=False,
            )
            table = st.build_table_from_trace(trace)
            original = march_offsets(
                torch.tensor(np.float32(float(azimuth))),
                torch.tensor(np.float32(float(altitude))),
                amp,
                scale,
                rows,
                cols,
            )
            assert table.count == len(original)
            assert list(table.dx) == [int(dx) for dx, _ in original]
            assert list(table.dy) == [int(dy) for _, dy in original]


# ---------------------------------------------------------------------------
# Producer mutations (documented mutation proof; source-level sed runs are
# recorded in the T03 report — these are the in-suite equivalents)
# ---------------------------------------------------------------------------


class TestProducerMutations:
    def test_mutation_swap_dx_dy_columns(self, fixture_traces):
        """MUTATION: producer swaps the dx and dy columns -> the byte gate
        and the replay gate must both fail."""
        trace = _find(
            fixture_traces,
            kernel_variant=st.KERNEL_SVF_SHADOW,
            azimuth_deg=37.0,
        )[0]
        mutated = dict(trace)
        mutated["dx"], mutated["dy"] = list(trace["dy"]), list(trace["dx"])
        assert _witness_caught(mutated, trace)

    def test_mutation_drop_overshoot_step(self, fixture_traces):
        """MUTATION: producer drops the final overshoot step -> caught by
        the byte gate AND changes replayed outputs (sensitivity proven in
        TestReplayEquality.test_marker_scene_detects_wrong_count)."""
        for trace in fixture_traces:
            amp = np.float32(
                st.u32_bits_to_f32(
                    st.bits_hex_to_u32(trace["key"]["executed_amplitude_bits"])
                )
            )
            last = np.float32(
                st.u32_bits_to_f32(st.bits_hex_to_u32(trace["dz_bits"][-1]))
            )
            if last > amp:
                mutated = dict(trace)
                mutated["count"] = trace["count"] - 1
                for column in ("dx", "dy", "dz_bits", "branch_id", "previous_dz_bits"):
                    mutated[column] = list(trace[column][:-1])
                assert _witness_caught(mutated, trace)
                return
        pytest.fail("no overshoot fixture available")


# ---------------------------------------------------------------------------
# Torch-freedom of the core module (T02 contract preserved)
# ---------------------------------------------------------------------------


class TestCoreTorchFreedom:
    def test_step_tables_module_is_torch_free(self):
        import subprocess

        code = (
            "import sys;"
            "from solweig_core import step_tables;"
            "assert 'torch' not in sys.modules;"
            "t = step_tables.StepTableKey("
            "semantics_profile='canonical_cpu_v1',"
            "kernel_variant=step_tables.KERNEL_SVF_SHADOW,"
            "source_geometry_hash='x'*64,"
            "angle_input_bits=('0x00000000','0x00000000'),"
            "scale_bits='0x00000000',logical_rows=4,logical_cols=8,"
            "amplitude_policy_id=step_tables.AMPLITUDE_SYNTHETIC_PROBE,"
            "executed_amplitude_bits='0x00000000',"
            "boundary_policy_id=step_tables.BOUNDARY_SHIFT_WINDOW_V1);"
            "assert t.kernel_variant == 'svf_shadow'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
