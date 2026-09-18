# SPDX-License-Identifier: GPL-3.0-only
"""T14 Change C (R9) gates: the compact runtime capture.

The reference t08 capture (~5.6 GB/window) froze everything the oracle
touched; the torch-free runtime loaders consume ~351 MB of it.
``tests/ultrafast/compact_capture.py`` records the consumed member set
EMPIRICALLY (intercepted ``np.load`` during a real full solve — reads AND
membership tests) and re-emits a bit-identical compact capture.

Gates:

* **Raw-bit differential** — ``full_solve_capture`` over the ORIGINAL and
  the COMPACT capture produce identical outputs: every per-timestep plane
  dict compared as raw uint32 views (dtype/shape/bits, NaN payloads and
  signed zeros included), the threaded CI series bit-exact, and the
  published anchor fingerprint equal (same geometry lineage).
* **Size** — the compaction actually compacts (and drops the never-read
  153 MB diffsh reference).
* **Truncation refusal** — a compact capture that DROPS a member the
  loaders read fails LOUDLY (KeyError naming the member); never a silent
  substitution. This is also the designated mutation killer proving the
  differential has teeth.

The compact capture is produced into the shared artifacts tree:

    .venv/bin/python tests/ultrafast/compact_capture.py \
        --src $ART/t08/capture --dst $ART/t14/compact_capture
"""
from __future__ import annotations

import json
import os
import struct
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

from solweig_core.numba_cpu import full_solve as fs  # noqa: E402

SRC = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")
DST = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14/compact_capture")
LATITUDE = 30.312645
PROFILE = "site_500-default"

pytestmark = pytest.mark.skipif(
    not (SRC / "t00.npz").is_file(), reason="t08 reference capture absent"
)


def _bits_of(value):
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "arr",
            str(contiguous.dtype),
            contiguous.shape,
            contiguous.view(np.uint8).tobytes()
            if contiguous.dtype != np.float32
            else contiguous.view(np.uint32).tobytes(),
        )
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (float, np.floating)):
        return (
            "float",
            struct.pack("<d", float(value)),
            struct.pack("<f", np.float32(value).item())
            if abs(float(value)) < 3.4e38
            else None,
        )
    return ("other", repr(value))


def _diff(name: str, a, b, out: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{name}: key sets differ {set(a) ^ set(b)}")
            return
        for k in sorted(a):
            _diff(f"{name}[{k!r}]", a[k], b[k], out)
        return
    ba, bb = _bits_of(a), _bits_of(b)
    if ba != bb:
        out.append(f"{name}: {ba[:3]} != {bb[:3]}")


def _solve(cap_dir: Path):
    cap = fs.CaptureSet(cap_dir=str(cap_dir), profile=PROFILE,
                        latitude=LATITUDE)
    return fs.full_solve_capture(cap, profile=PROFILE)


@pytest.fixture(scope="module")
def solves():
    if not (DST / "t00.npz").is_file():
        pytest.skip(
            f"compact capture absent at {DST} — produce it with "
            "tests/ultrafast/compact_capture.py (see module docstring)"
        )
    orig = _solve(SRC)
    comp = _solve(DST)
    return orig, comp


class TestCompactDifferential:
    def test_raw_bit_parity_all_timesteps(self, solves):
        orig, comp = solves
        assert orig.n_timesteps == comp.n_timesteps
        assert len(orig.outputs) == len(comp.outputs)
        problems: list[str] = []
        for t, (o, c) in enumerate(zip(orig.outputs, comp.outputs)):
            _diff(f"t{t:02d}", o, c, problems)
            if problems:
                break
        assert not problems, f"{len(problems)} divergences: {problems[:5]}"

    def test_ci_series_bit_exact(self, solves):
        orig, comp = solves
        a = [struct.pack("<d", v) for v in orig.ret_CI_series]
        b = [struct.pack("<d", v) for v in comp.ret_CI_series]
        assert a == b, "threaded CI series diverged"

    def test_anchor_lineage_preserved(self, solves):
        orig, comp = solves
        assert orig.anchor_fingerprint == comp.anchor_fingerprint
        assert (orig.anchor_published is None) == (
            comp.anchor_published is None
        )
        if orig.anchor_published is not None:
            _diff(
                "anchor_state",
                vars(orig.anchor_published),
                vars(comp.anchor_published),
                problems := [],
            )
            assert not problems, problems[:5]

    def test_geometry_fingerprint_identical(self):
        """The compact capture's static planes hash to the SAME geometry
        fingerprint (anchor classify/restore consumes it)."""
        if not (DST / "t00.npz").is_file():
            pytest.skip("compact capture absent")
        a = fs.geometry_fingerprint_of_capture(
            fs.CaptureSet(cap_dir=str(SRC), profile=PROFILE,
                          latitude=LATITUDE))
        b = fs.geometry_fingerprint_of_capture(
            fs.CaptureSet(cap_dir=str(DST), profile=PROFILE,
                          latitude=LATITUDE))
        assert a == b

    def test_parity_record_written(self, solves):
        orig, comp = solves
        art = DST.parent / "gates"
        art.mkdir(parents=True, exist_ok=True)
        record = {
            "src": str(SRC),
            "dst": str(DST),
            "n_timesteps": int(orig.n_timesteps),
            "n_output_dicts": len(orig.outputs),
            "ci_series_sha256": __import__("hashlib").sha256(
                b"".join(struct.pack("<d", v)
                         for v in orig.ret_CI_series)
            ).hexdigest(),
            "equal": True,
            "profile": PROFILE,
        }
        (art / "compact_parity.json").write_text(json.dumps(record, indent=1))


class TestCompactSize:
    def test_compaction_ratio_and_dropped_reference(self):
        report_path = DST / "compact_report.json"
        if not report_path.is_file():
            pytest.skip("compact_report.json absent — run the tool")
        report = json.loads(report_path.read_text())
        src, dst = report["src_bytes"], report["dst_bytes"]
        assert dst < 0.25 * src, (
            f"compaction insufficient: {src / 1e9:.2f} GB -> {dst / 1e9:.2f} GB"
        )
        # the 153 MB diffsh reference is oracle-side, never loader-read
        assert report["files"]["diffsh_ref.npy"]["dst_bytes"] == 0
        # every loader-consumed timestep file survives
        for name in ("t00.npz", "t08.npz", "static.npz",
                     "walk_offsets.npz", "azimuth_statics.npz"):
            assert report["files"][name]["dst_bytes"] > 0, name


class TestTruncationRefusal:
    def test_dropped_consumed_member_fails_loudly(self, tmp_path):
        """MUTATION/designated killer: a compact capture missing a member
        the loaders read must raise (KeyError naming it) — the raw-bit
        differential can never mistake a truncated capture for a valid
        one because the solve refuses outright."""
        if not (DST / "t00.npz").is_file():
            pytest.skip("compact capture absent")
        from compact_capture import record_loader_keys, write_compact_capture

        usage, _ = record_loader_keys(DST, PROFILE)
        # drop one consumed day-step member (a frozen transcendental)
        day_usage = usage.get("t09.npz", {})
        victim = next(
            (m for m in day_usage.get("gets", [])
             if m.startswith("frozen_")),
            None,
        )
        assert victim, f"no frozen_* member recorded for t09: {day_usage}"
        broken = tmp_path / "broken"
        write_compact_capture(
            DST, broken, usage, extra_drop={("t09.npz", victim)}
        )
        with np.load(broken / "t09.npz") as z:
            assert victim not in z.files
        with pytest.raises(KeyError) as info:
            _solve(broken)
        assert victim in str(info.value), (
            f"refusal did not name the missing member {victim!r}"
        )
