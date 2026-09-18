# SPDX-License-Identifier: GPL-3.0-only
"""R9 compact runtime capture: dynamic key enumeration + re-emission.

The t08 reference capture is ~5.4 GB per site window because it froze
EVERYTHING the oracle touched (including verification-only planes and the
153 MB diffsh reference cube the runtime never reads). The runtime
loaders (``met_recompute``/``radiation``/``full_solve``) consume a small
subset of the members. This tool records that subset EMPIRICALLY — by
intercepting every ``np.load`` member access and ``in z.files`` membership
test during a REAL full solve — and re-emits a compact capture containing
exactly those members, bit-identical.

Refusal contract: a compact capture that DROPS a member the loaders
actually read must fail loudly at solve time (never silently substitute);
the differential gate in ``test_compact_capture.py`` proves full_solve
(orig) == full_solve (compact) on raw bits AND that the dropped-key
mutation is caught.

Usage (dev env):

    .venv/bin/python tests/ultrafast/compact_capture.py \
        --src $ART/t08/capture --dst $ART/t14/compact_capture
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: the t08 reference capture's site constants (test_full_solve.py pins
#: the same values; the compact capture inherits them verbatim).
LATITUDE = 30.312645
PROFILE = "site_500-default"


class _RecordingFiles(list):
    """``z.files`` view that records membership tests."""

    def __init__(self, names, store):
        super().__init__(names)
        self._store = store

    def __contains__(self, key):
        self._store["contains"].add(str(key))
        return super().__contains__(key)


class _RecordingNpz:
    """Proxy over an ``NpzFile`` recording every member read."""

    def __init__(self, npz, store):
        self._npz = npz
        self._store = store

    def __getitem__(self, key):
        self._store["gets"].add(str(key))
        return self._npz[key]

    def __contains__(self, key):
        # loaders test membership on the NpzFile itself (``"k" in z``);
        # recording here keeps the True-branch keys in the compact set
        self._store["contains"].add(str(key))
        return str(key) in self._npz.files

    def __iter__(self):
        return iter(self._npz.files)

    @property
    def files(self):
        return _RecordingFiles(self._npz.files, self._store)

    def __getattr__(self, name):
        return getattr(self._npz, name)


def record_loader_keys(cap_dir: Path, profile: str = "canonical_cpu_v1") -> dict:
    """Run one identity full solve with ``np.load`` intercepted; return
    ``{relative filename: {"gets": [...], "contains": [...]}}`` plus
    top-level file usage (``__npy_direct__`` for bare .npy loads)."""
    from solweig_core.numba_cpu import full_solve

    usage: dict[str, dict[str, set]] = {}
    real_load = np.load

    def recording_load(file, *args, **kwargs):
        path = Path(file)
        if path.suffix not in (".npz", ".npy"):
            return real_load(file, *args, **kwargs)
        resolved = path.resolve()
        root = cap_dir.resolve()
        rel = (
            str(resolved.relative_to(root))
            if resolved.is_relative_to(root)
            else str(resolved)
        )
        store = usage.setdefault(rel, {"gets": set(), "contains": set()})
        loaded = real_load(file, *args, **kwargs)
        if path.suffix == ".npy":
            # bare arrays are read wholesale (no member granularity)
            store["gets"].add("__whole_file__")
            return loaded
        return _RecordingNpz(loaded, store)

    cap_set = full_solve.CaptureSet(
        cap_dir=str(cap_dir), profile=profile, latitude=LATITUDE
    )
    np.load = recording_load
    try:
        result = full_solve.full_solve_capture(cap_set, profile=profile)
    finally:
        np.load = real_load
    return {
        rel: {"gets": sorted(v["gets"]), "contains": sorted(v["contains"])}
        for rel, v in usage.items()
    }, result


def _compact_members_for(path: Path, file_usage: dict) -> list[str]:
    """Members that must be re-emitted: every member READ plus every
    membership test that returned True (the True branch consumed it)."""
    with np.load(path) as z:
        present = set(z.files)
    gets = set(file_usage.get("gets", []))
    contains_true = set(file_usage.get("contains", [])) & present
    return sorted(gets | contains_true)


def write_compact_capture(
    src: Path,
    dst: Path,
    usage: dict,
    *,
    extra_drop: set[str] = frozenset(),
) -> dict:
    """Re-emit the capture with only loader-consumed members.

    ``extra_drop`` exists ONLY for the mutation witness (proving a dropped
    consumed member is caught); production compaction passes nothing.
    Returns a size report. Non-npz files (manifest.json) are copied.
    Bare ``.npy`` files are copied only when the whole-file read fired.
    """
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    report = {"files": {}, "src_bytes": 0, "dst_bytes": 0}
    for path in sorted(src.iterdir()):
        rel = path.name
        target = dst / rel
        report["src_bytes"] += path.stat().st_size
        if path.suffix == ".npz":
            file_usage = usage.get(rel, {"gets": [], "contains": []})
            members = _compact_members_for(path, file_usage)
            if extra_drop:
                members = [m for m in members if (rel, m) not in extra_drop]
            with np.load(path) as z:
                payload = {m: z[m] for m in members}
            np.savez(dst / rel, **payload)
            report["files"][rel] = {
                "members": members,
                "src_bytes": path.stat().st_size,
                "dst_bytes": target.stat().st_size,
            }
        elif path.suffix == ".npy":
            whole = "__whole_file__" in usage.get(rel, {}).get("gets", [])
            if whole:
                shutil.copyfile(path, target)
                report["files"][rel] = {"members": ["__whole_file__"],
                                        "src_bytes": path.stat().st_size,
                                        "dst_bytes": target.stat().st_size}
            else:
                report["files"][rel] = {
                    "members": [], "src_bytes": path.stat().st_size,
                    "dst_bytes": 0,
                    "note": "never read by the runtime loaders — dropped",
                }
        else:
            shutil.copyfile(path, target)
            report["files"][rel] = {"members": ["__file__"],
                                    "src_bytes": path.stat().st_size,
                                    "dst_bytes": target.stat().st_size}
        report["dst_bytes"] += target.stat().st_size if target.exists() else 0
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--profile", default="canonical_cpu_v1")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    usage, _result = record_loader_keys(args.src, args.profile)
    record_s = time.perf_counter() - started
    report = write_compact_capture(args.src, args.dst, usage)
    report["record_seconds"] = record_s
    report["usage"] = usage
    (args.dst / "compact_report.json").write_text(json.dumps(report, indent=1))
    print(
        f"compact capture: {report['src_bytes'] / 1e9:.3f} GB -> "
        f"{report['dst_bytes'] / 1e9:.3f} GB at {args.dst} "
        f"(recording {record_s:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
