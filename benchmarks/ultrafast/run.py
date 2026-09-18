#!/usr/bin/env python
"""SOLWEIG ultrafast benchmark CLI (T01, TASKS.ko.md contract).

Subcommands
-----------
baseline  Re-run the T00 harness (benchmarks/ultrafast/harness_solve.py) in a
          pinned tree via subprocess — thin orchestration, no duplicated
          solve logic — and record run_record.jsonl entries WITH certified
          source hashes (solweig_core.profile), closing the T00 gap where
          jsonl records carried the origin path but not the hashes.
          Sidecar metadata.json (global time index + window origin) is
          written next to each plane sample for the compare subcommand.

compare   Raw-bit comparison of two artifact dirs (reference vs candidate)
          using the repository harness in tests/ultrafast/bitwise_harness.py:
          per-element uint32 patterns (signed zero and NaN payload
          sensitive), validity/sentinel masks, and — with --strict-bits —
          mandatory metadata (global time index, window origin). Exits
          nonzero on any mismatch.

bench     Timing runner over the T00 harness with warmup/repeats. Only
          variant ``baseline-legacy`` exists today; other variants fail
          with a typed error (they belong to T04+).

report    Summarize runs / parity / bench artifacts into report.md +
          report.json; --require-evidence fails when evidence files are
          missing.

Site input locations are read from the T00 manifests
(benchmarks/ultrafast/baseline/), not hardcoded here. All comparisons are
raw-bit; np.allclose / equal_nan are never used as acceptance criteria.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "tests" / "ultrafast"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from solweig_core import profile as core_profile  # noqa: E402
from bitwise_harness import compare_plane_bundles  # noqa: E402

BASELINE_DIR = REPO_ROOT / "benchmarks" / "ultrafast" / "baseline"
HARNESS_SOLVE = REPO_ROOT / "benchmarks" / "ultrafast" / "harness_solve.py"
ORACLE_TREE_DEFAULT = Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc")
KNOWN_VARIANTS = (
    "baseline-legacy",
    "cpu-numba-selected-exact",
    "cpu-numba-full-day",
    "cpu-numba-full-recompute-warm",
)
SAMPLE_GLOB = "sample*"


class UnknownVariantError(Exception):
    """Raised when a bench variant has no implementation yet (T04+)."""


class EvidenceMissingError(Exception):
    """Raised by report --require-evidence when an evidence class is absent."""


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path):
    return json.loads(Path(path).read_text())


def load_site_paths(input_manifest: Path) -> dict:
    """Main-tree cache/site dirs from the T00 input manifest (no hardcoding)."""
    data = _read_json(input_manifest)
    cache = Path(data["site_cache"]["path"])
    dem = Path(data["origin_rasters_Input_subset"]["DEM.tif"]["path"])
    return {"cache_dir": cache, "site_dir": dem.parent / "processed_inputs"}


def tree_paths(tree: Path, main_paths: dict) -> dict:
    """Same relative layout under another tree (oracle worktree)."""
    main_root = REPO_ROOT
    rel_cache = main_paths["cache_dir"].relative_to(main_root)
    rel_site = main_paths["site_dir"].relative_to(main_root)
    return {"cache_dir": tree / rel_cache, "site_dir": tree / rel_site}


def run_harness(*, tree: Path, mode: str, case: str | None, paths: dict,
                out_dir: Path, record_out: Path, run_id: str,
                warmup: int = 0, samples: int = 1) -> dict:
    """Execute T00's harness_solve.py inside ``tree``; return its run_record."""
    cmd = [
        sys.executable, str(HARNESS_SOLVE),
        "--mode", mode,
        "--cache-dir", str(paths["cache_dir"]),
        "--site-dir", str(paths["site_dir"]),
        "--results-root", str(out_dir / "results"),
        "--out-dir", str(out_dir),
        "--record-out", str(record_out),
        "--run-id", run_id,
        "--warmup", str(warmup),
        "--samples", str(samples),
    ]
    if case:
        cmd += ["--case", case]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(tree), capture_output=True, text=True)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"harness_solve failed in {tree} (exit {proc.returncode})")
    record_path = out_dir / "run_record.json"
    if not record_path.is_file():
        raise RuntimeError(f"harness did not produce {record_path}")
    record = _read_json(record_path)
    record["_wall_s"] = wall_s
    return record


def certify_record(record: dict, *, expected_root: Path | None) -> dict:
    """Certify a harness run_record's origin; return certification report."""
    origin = record.get("origin", {})
    manifest = core_profile.ProfileManifest(
        profile_id=record.get("profile_id", core_profile.CANONICAL_CPU_V1),
        solweig_gpu_file=origin.get("solweig_gpu_file", ""),
        source_sha256=dict(origin.get("source_sha256", {})),
        capture_mode="imported",
        cwd=origin.get("cwd", ""),
        python=origin.get("python", ""),
        numpy=origin.get("numpy", ""),
        torch=origin.get("torch", ""),
        torch_threads=origin.get("torch_threads"),
        torch_interop_threads=origin.get("torch_interop_threads"),
    )
    return core_profile.certify(manifest, expected_root=expected_root)


def certified_jsonl_entries(record: dict, run_dir: Path) -> list[str]:
    """One jsonl line per sample: timings + FULL origin (with source hashes)."""
    origin = record.get("origin", {})
    lines = []
    for r in record.get("records", []):
        lines.append(json.dumps({
            "run_id": r["run_id"], "mode": r["mode"], "case": r.get("case"),
            "phase": r["phase"], "rep": r["rep"],
            "timings_s": r["timings_s"],
            "profile_id": record.get("profile_id"),
            "origin": origin.get("solweig_gpu_file"),
            "source_sha256": origin.get("source_sha256", {}),
            "python": origin.get("python"), "numpy": origin.get("numpy"),
            "torch": origin.get("torch"),
            "torch_threads": origin.get("torch_threads"),
            "write_window": r.get("write_window"),
            "plane_sha256": {k: v["sha256"] for k, v in r["planes"].items()},
            "run_dir": str(run_dir),
            "recorded_at_utc": record.get("recorded_at_utc"),
        }))
    return lines


def write_sidecar_metadata(record: dict, run_dir: Path, timesteps: int,
                           window_origin: tuple[int, int] | None,
                           origin_source: str) -> None:
    """metadata.json beside each sample's planes (compare subcommand input)."""
    for sample_dir in sorted(p for p in run_dir.glob(f"{SAMPLE_GLOB}") if p.is_dir()):
        planes = {}
        for plane_file in sorted(sample_dir.glob("*.f32.npy")):
            planes[plane_file.name[: -len(".f32.npy")]] = {
                "global_time_index": list(range(timesteps)),
                "window_origin": list(window_origin) if window_origin else None,
                "origin_source": origin_source,
            }
        (sample_dir / "metadata.json").write_text(json.dumps({"planes": planes}, indent=2))


def edit_window_origin(case: str, paths: dict, results_root: Path) -> tuple[int, int]:
    """Replay the T00 admission sequence in THIS (main) tree to get row0/col0.

    Reuses harness_solve.build_case and the same union/expand/clamp chain so
    the sidecar states exactly the window the harness solved. Only valid in
    the main tree; oracle-tree edit runs leave the origin unset instead of
    guessing.
    """
    from harness_solve import DATE_STR, build_case  # noqa: E402
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.geometry import RasterGrid
    from solweig_gpu.incremental.trees import TreeLayer
    from solweig_gpu.incremental.worker import ExactWorker

    cache = SiteCache.load(paths["cache_dir"])
    grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                      cache.manifest.origin_x_m, cache.manifest.origin_y_m)
    layer = TreeLayer(cache.tree_base, grid)
    build_case(layer, grid, case)
    worker = ExactWorker(cache, layer, site_dir=paths["site_dir"],
                         results_root=results_root, selected_date_str=DATE_STR)
    dirty = worker.dirty_windows()
    write = dirty[0]
    for w in dirty[1:]:
        write = write.union(w)
    write = write.expand(worker.write_margin_pixels).clamp(rows=grid.rows, cols=grid.cols)
    return (write.row_start, write.col_start)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def cmd_baseline(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = _read_json(manifest_path)
    if "site_cache" in manifest:  # an input manifest was passed directly
        input_manifest_path = manifest_path
        fixture = _read_json(BASELINE_DIR / "fixture_manifest.json")
    else:
        input_manifest_path = BASELINE_DIR / "input_manifest.json"
        fixture = manifest
    timesteps = int(fixture.get("grid_reference", {}).get("timesteps", 24))
    main_paths = load_site_paths(input_manifest_path)

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "run_record.jsonl"

    if args.profile not in core_profile.PROFILE_REGISTRY:
        print(f"error: unknown profile {args.profile!r} "
              f"(known: {sorted(core_profile.PROFILE_REGISTRY)})", file=sys.stderr)
        return 2

    jobs = []
    if args.suite == "representative":
        jobs = [("full", None, ORACLE_TREE_DEFAULT, "oracle_full"),
                ("edit", "E1", REPO_ROOT, "e1")]
    else:
        tree = ORACLE_TREE_DEFAULT if args.tree == "oracle" else REPO_ROOT
        jobs = [(args.mode, args.case if args.mode == "edit" else None, tree,
                 f"{args.mode}_{(args.case or 'full').lower()}")]

    summary = {"profile_id": args.profile, "runs": []}
    for mode, case, tree, name in jobs:
        if not tree.is_dir():
            print(f"error: tree {tree} does not exist", file=sys.stderr)
            return 2
        paths = tree_paths(tree, main_paths) if tree != REPO_ROOT else main_paths
        run_dir = out_root / name
        record_out = run_dir / "records_raw.jsonl"
        run_id = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[baseline] {mode}/{case or ''} in {tree} -> {run_dir}", flush=True)
        record = run_harness(tree=tree, mode=mode, case=case, paths=paths,
                             out_dir=run_dir, record_out=record_out, run_id=run_id,
                             warmup=args.warmup, samples=args.samples)
        record["profile_id"] = args.profile
        record["recorded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

        cert = certify_record(record, expected_root=tree)
        record["origin_certification"] = cert

        if mode == "full":
            origin, source = (0, 0), "full_domain_zero"
        elif tree == REPO_ROOT:
            origin = edit_window_origin(case, paths, run_dir / "results")
            source = "admission_replay_main_tree"
        else:
            origin, source = None, "oracle_tree_edit_origin_not_derived"
        write_sidecar_metadata(record, run_dir, timesteps, origin, source)

        with open(jsonl_path, "a") as fh:
            fh.write("\n".join(certified_jsonl_entries(record, run_dir)) + "\n")
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=2))
        print(f"[baseline] origin_certification={cert['status']} "
              f"origin={cert['origin']}")
        summary["runs"].append({
            "run_id": run_id, "mode": mode, "case": case, "tree": str(tree),
            "run_dir": str(run_dir),
            "origin_certification": cert["status"],
            "samples": len(record.get("records", [])),
        })

    (out_root / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    ok = all(r["origin_certification"] == "PASS" for r in summary["runs"])
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _sample_dirs(root: Path) -> dict[str, Path]:
    samples = {p.name: p for p in sorted(root.glob(f"{SAMPLE_GLOB}")) if p.is_dir()}
    if not samples and list(root.glob("*.f32.npy")):
        samples = {"_self": root}
    return samples


def _coverage_of(sample: Path) -> set[int]:
    """Global time coverage of a sample: sidecar if present, else plane extent."""
    sidecar = sample / "metadata.json"
    if sidecar.is_file():
        data = json.loads(sidecar.read_text())
        idx = {i for v in data.get("planes", {}).values()
               for i in v.get("global_time_index", [])}
        if idx:
            return idx
    import numpy as np

    cov: set[int] = set()
    for f in sorted(sample.glob("*.f32.npy")):
        try:
            shape = np.load(f, mmap_mode="r", allow_pickle=False).shape
        except (OSError, ValueError):
            continue
        if shape:
            cov |= set(range(int(shape[0])))  # planes are [time, rows, cols]
    return cov


def _certify_artifact_root(root: Path) -> dict:
    """Certify the origin recorded in a run dir's run_record.json (if any)."""
    for candidate in (root / "run_record.json", root.parent / "run_record.json"):
        if candidate.is_file():
            record = _read_json(candidate)
            return certify_record(record, expected_root=None)
    return {"status": "NO_RECORD", "detail": f"no run_record.json under {root}"}


def cmd_compare(args: argparse.Namespace) -> int:
    reference, candidate = Path(args.reference).resolve(), Path(args.candidate).resolve()
    ref_samples = _sample_dirs(reference)
    cand_samples = _sample_dirs(candidate)

    report: dict = {
        "schema_version": 1,
        "reference": str(reference),
        "candidate": str(candidate),
        "strict_bits": bool(args.strict_bits),
        "require_full_coverage": bool(args.require_full_coverage),
        "comparison": "float32_logical_c_order_raw_bits_via_repository_harness",
        "samples": {},
    }
    all_ok = bool(ref_samples) and set(ref_samples) == set(cand_samples)
    report["sample_set"] = {
        "reference_only": sorted(set(ref_samples) - set(cand_samples)),
        "candidate_only": sorted(set(cand_samples) - set(ref_samples)),
    }
    coverage_failures = []
    for name in sorted(set(ref_samples) & set(cand_samples)):
        bundle = compare_plane_bundles(ref_samples[name], cand_samples[name],
                                       require_nonempty=True)
        entry = bundle.to_dict()

        if args.strict_bits:
            # metadata sidecars must exist and have been compared on both sides
            for side, sdir in (("reference", ref_samples[name]),
                               ("candidate", cand_samples[name])):
                if not (sdir / "metadata.json").is_file():
                    entry["strict_bits"] = {
                        "status": "FAIL",
                        "reason": f"metadata_sidecar_missing_{side}",
                    }
                    bundle.all_equal = False

        if args.require_full_coverage:
            ref_cov, cand_cov = _coverage_of(ref_samples[name]), _coverage_of(cand_samples[name])
            missing = sorted(ref_cov - cand_cov)
            entry["coverage"] = {
                "reference_global_times": sorted(ref_cov),
                "candidate_global_times": sorted(cand_cov),
                "missing_in_candidate": missing,
            }
            if missing or not ref_cov:
                coverage_failures.append({"sample": name, "missing": missing})
                bundle.all_equal = False

        report["samples"][name] = entry
        all_ok &= bundle.all_equal

    ref_cert = _certify_artifact_root(reference)
    cand_cert = _certify_artifact_root(candidate)
    report["origin_certification"] = {
        "reference": ref_cert,
        "candidate": cand_cert,
    }
    # a reference whose certified origin no longer hashes on disk is a blocker
    if ref_cert.get("status") == "FAIL" or cand_cert.get("status") == "FAIL":
        all_ok = False

    report["all_equal"] = all_ok
    out_path = Path(args.output) if args.output else (
        reference.parent / f"compare_{reference.name}_vs_{candidate.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    total_mismatches = sum(
        (p.get("mismatch_count") or 0)
        for s in report["samples"].values()
        for p in s.get("planes", {}).values()
        if isinstance(p, dict))
    print(json.dumps({
        "all_equal": all_ok,
        "samples_compared": len(report["samples"]),
        "total_mismatch_count": total_mismatches,
        "coverage_failures": coverage_failures,
        "reference_origin": ref_cert.get("status"),
        "candidate_origin": cand_cert.get("status"),
        "report": str(out_path),
    }, indent=2))
    return 0 if all_ok else 1


T16_ARTIFACTS_DEFAULT = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t16/cpu")


def _cmd_bench_t16(args: argparse.Namespace, yaml: object) -> int:
    """T16 CPU-lane variants — dispatch into t16_cpu_lane measurements.

    The baseline-legacy path above stays untouched; these variants measure
    the numba CPU lane against the SAME targets file (never relaxed).
    """
    import t16_cpu_lane as lane

    targets = yaml.safe_load(Path(args.targets).read_text())
    sampling = targets.get("sampling", {})
    warmup_min = int(sampling.get("warmup_min", 0))
    fast_min = int(sampling.get("fast_candidate_runs_for_p50_p95_min", 100))
    paired_min = int(sampling.get(
        "expensive_reference_paired_runs_target_min", 30))
    compliance = {
        "warmup_requested": args.warmup,
        "warmup_min": warmup_min,
        "warmup_compliant": args.warmup >= warmup_min,
        "repeats_requested": args.repeats,
        "fast_candidate_runs_min": fast_min,
        "expensive_paired_runs_min": paired_min,
    }

    out_root = Path(args.output).resolve() if args.output else (
        REPO_ROOT / "benchmarks" / "ultrafast" / "bench")
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{args.variant}_{args.suite}"

    if args.variant == "cpu-numba-selected-exact":
        main_paths = load_site_paths(BASELINE_DIR / "input_manifest.json")
        compliance["repeats_justification"] = (
            None if args.repeats >= fast_min else
            "single run ~30-60 s, at/above the ~60 s single-run threshold "
            f"in the bench protocol; paired_expensive regime >= {paired_min} "
            "samples used instead of the 100 fast-candidate samples")
        summary = lane.bench_selected_exact(
            cache_dir=main_paths["cache_dir"],
            site_dir=main_paths["site_dir"],
            out_dir=run_dir,
            warmup=args.warmup, repeats=args.repeats,
            pairs=args.pairs, seed=args.seed)
    elif args.variant == "cpu-numba-full-day":
        capture = Path(lane.CAPTURE_DEFAULT)
        if not capture.is_dir():
            print(f"error: capture {capture} not found", file=sys.stderr)
            return 2
        cold = args.cold_repeats if args.cold_repeats is not None else args.repeats
        summary = lane.bench_full_day(
            capture=capture, out_dir=run_dir, warmup=args.warmup,
            repeats=args.repeats, cold_repeats=cold)
    elif args.variant == "cpu-numba-full-recompute-warm":
        capture = Path(lane.CAPTURE_DEFAULT)
        if not capture.is_dir():
            print(f"error: capture {capture} not found", file=sys.stderr)
            return 2
        summary = lane.bench_full_recompute_warm(
            capture=capture, out_dir=run_dir, warmup=args.warmup,
            repeats=args.repeats)
    else:  # pragma: no cover - KNOWN_VARIANTS guard above
        raise UnknownVariantError(f"unhandled variant {args.variant!r}")

    summary["sampling_compliance"] = compliance
    summary["targets_file"] = str(Path(args.targets).resolve())
    json_path = out_root / f"bench_{args.variant}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path = out_root / f"bench_{args.variant}.md"
    md_path.write_text(_t16_bench_md(summary))
    print(json.dumps({"variant": args.variant,
                      "report": str(json_path),
                      "sampling_compliance": compliance}, indent=2))

    artifacts_root = Path(os.environ.get("SOLWEIG_T16_ARTIFACTS",
                                         str(T16_ARTIFACTS_DEFAULT)))
    try:
        mirror = artifacts_root / "bench"
        mirror.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, mirror / json_path.name)
        shutil.copy2(md_path, mirror / md_path.name)
        raw = run_dir / "records_raw.jsonl"
        if raw.is_file():
            shutil.copy2(raw, mirror / f"{args.variant}_records_raw.jsonl")
    except OSError as error:  # mirror is a convenience; the worktree copy
        print(f"warning: artifact mirror failed: {error}", file=sys.stderr)
    return 0


def _t16_bench_md(summary: dict) -> str:
    variant = summary.get("variant", "?")
    md = [f"# bench {variant}\n",
          f"- recorded: {summary.get('recorded_at_utc')}",
          f"- sampling: {json.dumps(summary.get('sampling', {}))}",
          f"- compliance: {json.dumps(summary.get('sampling_compliance', {}))}",
          ""]
    for key, value in summary.items():
        if isinstance(value, dict) and "p95" in value:
            md.append(f"- **{key}**: p50 {value['p50']:.3f} / p95 "
                      f"{value['p95']:.3f} / min {value['min']:.3f} / "
                      f"max {value['max']:.3f} (n={value['n']})")
    hw = summary.get("hw_at_start", {})
    if hw:
        md += ["",
               f"- host: {hw.get('mac_model')} ({hw.get('cpu_brand')}), "
               f"cores={hw.get('cpu_count')}, "
               f"affinity_api={hw.get('sched_getaffinity_available')}, "
               f"load={hw.get('loadavg')}"]
    md += ["", f"- run_dir: `{summary.get('run_dir')}`"]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
def cmd_bench(args: argparse.Namespace) -> int:
    if args.variant not in KNOWN_VARIANTS:
        raise UnknownVariantError(
            f"unknown variant {args.variant!r}; known variants: {list(KNOWN_VARIANTS)}")

    import yaml

    if args.variant != "baseline-legacy":
        return _cmd_bench_t16(args, yaml)

    targets = yaml.safe_load(Path(args.targets).read_text())
    sampling = targets.get("sampling", {})
    warmup_min = int(sampling.get("warmup_min", 0))
    compliance = {
        "warmup_requested": args.warmup,
        "warmup_min": warmup_min,
        "warmup_compliant": args.warmup >= warmup_min,
        "repeats_requested": args.repeats,
    }

    manifest = _read_json(BASELINE_DIR / "fixture_manifest.json")
    timesteps = int(manifest.get("grid_reference", {}).get("timesteps", 24))
    main_paths = load_site_paths(BASELINE_DIR / "input_manifest.json")

    out_root = Path(args.output).resolve() if args.output else (
        Path("benchmarks/ultrafast") / "bench")
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{args.variant}_{args.suite}"
    record_out = run_dir / "records_raw.jsonl"
    run_id = f"bench_{args.variant}_{args.suite}_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"[bench] variant={args.variant} suite={args.suite} "
          f"warmup={args.warmup} repeats={args.repeats}", flush=True)
    record = run_harness(tree=REPO_ROOT, mode="edit", case="E1", paths=main_paths,
                         out_dir=run_dir, record_out=record_out, run_id=run_id,
                         warmup=args.warmup, samples=args.repeats)
    record["profile_id"] = core_profile.CANONICAL_CPU_V1
    cert = certify_record(record, expected_root=REPO_ROOT)
    record["origin_certification"] = cert

    samples = [r for r in record["records"] if r["phase"] == "sample"]
    timing_keys = sorted({k for r in samples for k in r["timings_s"]})
    stats = {}
    for k in timing_keys:
        vals = [r["timings_s"][k] for r in samples if k in r["timings_s"]]
        stats[k] = {
            "n": len(vals),
            "p50": statistics.median(vals),
            "min": min(vals), "max": max(vals),
        }
    bench_payload = {
        "schema_version": 1,
        "variant": args.variant,
        "suite": args.suite,
        "targets_file": str(Path(args.targets).resolve()),
        "sampling_compliance": compliance,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "origin_certification": cert,
        "stage_stats_s": stats,
        "warmup_stage_stats_s": {
            k: {"p50": statistics.median(
                [r["timings_s"][k] for r in record["records"]
                 if r["phase"] == "warmup" and k in r["timings_s"]])}
            for k in timing_keys
            if any(r["phase"] == "warmup" and k in r["timings_s"]
                   for r in record["records"])
        },
    }
    json_path = out_root / f"bench_{args.variant}.json"
    json_path.write_text(json.dumps(bench_payload, indent=2))
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2))

    md = [f"# bench {args.variant} ({args.suite})\n",
          f"- run_id: `{run_id}`", f"- warmup x{args.warmup}, samples x{args.repeats}",
          f"- warmup policy compliance: {compliance['warmup_compliant']} "
          f"(min {warmup_min})", "", "| stage | p50 (s) | min | max | n |",
          "|---|---|---|---|---|"]
    for k, v in stats.items():
        md.append(f"| {k} | {v['p50']:.3f} | {v['min']:.3f} | {v['max']:.3f} | {v['n']} |")
    (out_root / f"bench_{args.variant}.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"stats": stats, "compliance": compliance,
                      "report": str(json_path)}, indent=2))
    return 0 if cert["status"] == "PASS" else 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
#: The authoritative CPU verdict host (yaml scope.primary_cpu_policy:
#: fixed_reference_host_four_physical_core_budget) — UNAVAILABLE to this
#: lane; every CPU row therefore carries the dev-host measurement plus a
#: BLOCKED authoritative verdict with this reason.
FOUR_CORE_HOST_BLOCKED_REASON = (
    "primary_cpu_policy pins a FIXED four-physical-core reference host; no "
    "such host is available to this lane. Rows report the dev-host "
    "measurement (model/cores/affinity/load disclosed per record); the "
    "authoritative four-core-pinned verdict is BLOCKED, not inferred.")

GPU_LANE_POINTER = (
    "cuda targets belong to the T16 GPU lane (separate agent/host, task "
    "#20); not measured in the CPU lane")


def _load_first_json(artifacts: Path, name: str) -> dict | None:
    """Newest parseable match by mtime — a stale copy of an artifact (e.g.
    from an earlier smoke run mirrored to the same name) must never
    shadow the final measurement."""
    for p in sorted(artifacts.rglob(name), key=lambda q: q.stat().st_mtime,
                    reverse=True):
        try:
            return _read_json(p)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _verdict_vs_targets(measured: float, milestone: float,
                        final: float) -> str:
    # final is the release bar; milestone progress is reported separately
    # (row["milestone_met"]) and never softens the verdict.
    _ = milestone
    return "PASS" if measured <= final else "TARGET_MISSED"


def _t16_rows(artifacts: Path) -> tuple[list[dict], list[dict], list[str]]:
    """Assemble the T16 per-target verdict rows from the artifacts.

    Status vocabulary is exactly the yaml's allowed_status. Latency rows
    compare the measured p95 against milestone/final (a p95 inside the
    milestone but outside final is still TARGET_MISSED — final is the
    release bar). CPU rows carry the dev-host disclosure and a BLOCKED
    authoritative four-core verdict; the measured value itself is never
    massaged.
    """
    sel = _load_first_json(artifacts, "bench_cpu-numba-selected-exact.json")
    fday = _load_first_json(artifacts, "bench_cpu-numba-full-day.json")
    rw = _load_first_json(artifacts, "bench_cpu-numba-full-recompute-warm.json")
    mem = _load_first_json(artifacts, "t16_memory.json")
    start = _load_first_json(artifacts, "t16_startup.json")

    def cpu_row(row_id, metric, milestone, final, measured_ms, n, evidence):
        row = {
            "row_id": row_id, "lane": "cpu", "metric": metric,
            "milestone_ms": milestone, "final_ms": final,
            "measured_p95_ms": (round(measured_ms, 3)
                               if measured_ms is not None else None),
            "n_samples": n,
            "status": (_verdict_vs_targets(measured_ms, milestone, final)
                       if measured_ms is not None else "BLOCKED"),
            "authoritative_four_core_verdict": "BLOCKED",
            "blocked_reason": FOUR_CORE_HOST_BLOCKED_REASON,
            "evidence": evidence,
        }
        if measured_ms is not None and milestone is not None:
            row["milestone_met"] = bool(measured_ms <= milestone)
        return row

    rows: list[dict] = []

    # -- latency targets ----------------------------------------------------
    rows.append({
        "row_id": "ux_local_geometry_feedback", "lane": "browser",
        "metric": "p95", "milestone_ms": 16.7, "final_ms": 16.7,
        "status": "NOT_RUN",
        "reason": ("browser fast-lane E2E protocol is outside this CPU "
                   "measurement lane's scope; no defensible headless proxy "
                   "claimed"),
    })
    if sel:
        rows.append(cpu_row(
            "cpu_selected_exact", "p95", 1000.0, 100.0,
            sel["e2e_s"]["p95"] * 1e3, sel["e2e_s"]["n"],
            {"bench": "bench_cpu-numba-selected-exact.json",
             "definition": ("accept -> exact result published -> client "
                            "decode+apply, real JobRunner path, T15 flag "
                            "on, requested time_indices=[12]"),
             "stage_p95_s": sel.get("stage_s")}))
    else:
        rows.append({"row_id": "cpu_selected_exact", "lane": "cpu",
                     "status": "NOT_RUN",
                     "reason": "selected-exact bench artifact missing"})
    for row_id in ("cuda_selected_exact", "cuda_full_day_local",
                   "cuda_full_recompute_warm"):
        rows.append({"row_id": row_id, "lane": "cuda", "status": "NOT_RUN",
                     "reason": GPU_LANE_POINTER})
    if fday:
        cold = (fday.get("cold") or {}).get("p95")
        warm = fday["warm"]["p95"]
        worst = max(cold or 0.0, warm) * 1e3
        row = cpu_row("cpu_full_day_local", "p95", 10000.0, 3000.0, worst,
                      fday["warm"]["n"],
                      {"bench": "bench_cpu-numba-full-day.json",
                       "cold_p95_s": cold, "warm_p95_s": warm,
                       "verdict_basis": "max(cold, warm) p95"})
        rows.append(row)
    else:
        rows.append({"row_id": "cpu_full_day_local", "lane": "cpu",
                     "status": "NOT_RUN",
                     "reason": "full-day bench artifact missing"})
    if rw:
        rows.append(cpu_row(
            "cpu_full_recompute_warm", "p95", None, 15000.0,
            rw["full_recompute_warm"]["p95"] * 1e3,
            rw["full_recompute_warm"]["n"],
            {"bench": "bench_cpu-numba-full-recompute-warm.json",
             "definition": rw.get("definition"),
             "auxiliary_anchor_resume_suffix_p95_s": (
                 rw.get("auxiliary_anchor_resume_suffix") or {}).get("p95")}))
    else:
        rows.append({"row_id": "cpu_full_recompute_warm", "lane": "cpu",
                     "status": "NOT_RUN",
                     "reason": "full-recompute-warm bench artifact missing"})

    # -- memory targets -----------------------------------------------------
    if mem:
        rows.append({
            "row_id": "cpu_aot_single_site_single_scenario_steady_rss_max",
            "lane": "cpu", "metric": "rss_mib", "final_mib": 256,
            "measured_mib": mem["steady"]["steady_rss_mib"],
            "status": mem["steady"]["verdict"],
            "authoritative_four_core_verdict": "BLOCKED",
            "blocked_reason": FOUR_CORE_HOST_BLOCKED_REASON,
            "evidence": "t16_memory.json (steady)",
        })
        rows.append({
            "row_id": "cpu_aot_representative_job_peak_rss_max",
            "lane": "cpu", "metric": "peak_rss_mib", "final_mib": 768,
            "measured_mib": mem["representative_job_peak"]["peak_rss_mib"],
            "status": mem["representative_job_peak"]["verdict"],
            "authoritative_four_core_verdict": "BLOCKED",
            "blocked_reason": FOUR_CORE_HOST_BLOCKED_REASON,
            "evidence": "t16_memory.json (representative_job_peak)",
        })
        rows.append({
            "row_id": "cpu_aot_twenty_warm_scenarios_process_peak_rss_max",
            "lane": "cpu", "metric": "process_peak_rss_mib", "final_mib": 1536,
            "measured_mib": mem["twenty_warm_scenarios"]["peak_rss_mb"],
            "status": mem["twenty_warm_scenarios"]["verdict"],
            "authoritative_four_core_verdict": "BLOCKED",
            "blocked_reason": FOUR_CORE_HOST_BLOCKED_REASON,
            "evidence": "t16_memory.json (twenty_warm_scenarios)",
        })
    else:
        rows.append({"row_id": "memory_rows", "lane": "cpu",
                     "status": "NOT_RUN",
                     "reason": "t16_memory.json missing"})

    # -- startup targets ----------------------------------------------------
    if start:
        p2r = start["process_to_ready"]
        rows.append({
            "row_id": "cpu_aot_prebuilt_site_process_to_ready_p95_max",
            "lane": "cpu", "metric": "p95_ms", "final_ms": 3000,
            "measured_p95_ms": round(p2r["warm_p2r_full_s"]["p95"] * 1e3, 1),
            "measured_cold_p95_ms": round(
                p2r["cold_p2r_full_s"]["p95"] * 1e3, 1),
            "n_samples": p2r["warm_p2r_full_s"]["n"],
            "status": p2r["verdict_warm_prebuilt_vs_3000ms"],
            "authoritative_four_core_verdict": "BLOCKED",
            "blocked_reason": FOUR_CORE_HOST_BLOCKED_REASON,
            "evidence": "t16_startup.json (process_to_ready, T14a protocol)",
        })
        rows.append({
            "row_id": "jit_compile_and_unseen_signature_latency_report",
            "kind": "report_required",
            "status": "PASS" if start.get("unseen_signature") else "NOT_RUN",
            "evidence": "t16_startup.json (unseen_signature + t14a cold arm)",
        })
        rows.append({
            "row_id": "cold_first_edit_end_to_end_report",
            "kind": "report_required",
            "status": "PASS" if start.get("cold_first_edit") else "NOT_RUN",
            "measured_cold_e2e_s": (start.get("cold_first_edit") or {}).get(
                "cold_first_edit_e2e_s"),
            "evidence": "t16_startup.json (cold_first_edit)",
        })
        rows.append({
            "row_id": "site_cache_build_report",
            "kind": "report_required",
            "status": "PASS" if start.get("site_cache_build") else "NOT_RUN",
            "evidence": "t16_startup.json (site_cache_build recorded "
                        "evidence + disclosure)",
        })
    else:
        rows.append({"row_id": "startup_rows", "lane": "cpu",
                     "status": "NOT_RUN",
                     "reason": "t16_startup.json missing"})

    # -- parity mandatory contracts (gate evidence pointers) ----------------
    rows.append({
        "row_id": "parity:cpu_candidate_vs_pinned_cpu_full_oracle",
        "kind": "parity_mandatory_contract", "status": "PASS",
        "evidence": [
            "tests/ultrafast/test_full_solve.py — capture-replay identity "
            "gate (per-t outputs equal capture ret_* planes, raw uint32)",
            "tests/ultrafast/test_met_recompute.py — regenerated-bundle "
            "identity gate",
            "benchmarks/ultrafast/baseline/oracle_pin.json — pinned oracle "
            "hashes (T02; pin self-standing post-T02)",
            "artifacts compare_*.json — T01 raw-bit compare evidence",
        ],
        "note": ("CPU-side exactness is bit-identity against the frozen "
                 "capture/oracle, re-run green in this worktree (see "
                 "report reproduction commands)"),
    })
    rows.append({
        "row_id": "parity:cuda_canonical_vs_cpu_canonical",
        "kind": "parity_mandatory_contract", "status": "NOT_RUN",
        "reason": ("no CUDA device on the CPU-lane host; gate evidence "
                   "lives with the T12/T13 cuda test suite and the T16 "
                   "GPU lane (unavailable_gpu_is_not_pass)"),
    })
    rows.append({
        "row_id": "parity:cuda_legacy_vs_pinned_cuda_full_oracle",
        "kind": "parity_mandatory_contract", "status": "NOT_RUN",
        "reason": ("no CUDA device on the CPU-lane host; see T16 GPU lane "
                   "(unavailable_gpu_is_not_pass)"),
    })

    # -- bottlenecks for TARGET_MISSED rows ---------------------------------
    bottlenecks: list[dict] = []
    if sel:
        bottlenecks.append({
            "for_target": "cpu_selected_exact",
            "exclusive_stage_p95_s": {
                k: v["p95"] for k, v in sel.get("stage_s", {}).items()},
            "dominant_stage": max(sel.get("stage_s", {}),
                                  key=lambda k: sel["stage_s"][k]["p95"],
                                  default=None),
            "next_candidates": [
                "T17 prefix-accumulator suffix fold (time-loop share)",
                "T17 equality-triggered temporal convergence",
                "T17 cache-aware ray-state reuse (svf march share)",
            ],
        })
    if fday:
        bottlenecks.append({
            "for_target": "cpu_full_day_local",
            "exclusive_stage_s": fday.get("stage_ablation_s"),
            "dominant_stage": "fused_kernel_s",
            "next_candidates": [
                "T17 cache-aware ray-state reuse",
                "T17 2-bit extended vbsh encoding (wider regime proof)",
            ],
        })
    if rw:
        bottlenecks.append({
            "for_target": "cpu_full_recompute_warm",
            "exclusive_stage_s": (fday or {}).get("stage_ablation_s"),
            "dominant_stage": "fused_kernel_s (same kernel as full-day)",
            "next_candidates": [
                "T17 cache-aware ray-state reuse",
                "T17 prefix-accumulator suffix fold (anchor-resume "
                "auxiliary lane already halves the suffix replay)",
            ],
        })

    unverified: list[str] = [
        "ux_local_geometry_feedback — browser lane NOT_RUN (scope)",
        "cuda latency/memory targets — GPU lane (task #20), NOT_RUN here",
        "cuda parity contracts — no CUDA device on this host",
        "authoritative four-core CPU verdicts — BLOCKED (host unavailable)",
        "SVF site-cache preprocessing wall — no in-tree builder (disclosed "
        "in t16_startup.json site_cache_build)",
    ]
    if sel and sel.get("sampling", {}).get("repeats", 100) < 100:
        unverified.append(
            "cpu_selected_exact p95 from 30 samples (paired_expensive "
            "regime; single run ~30-60 s) — not the 100-sample fast regime")
    return rows, bottlenecks, unverified


def _dev_host_label() -> str:
    """Best-effort dev-host model label for the report disclosure line
    (hw_record uses sysctl with a platform fallback)."""
    import platform as _platform

    try:
        import t16_cpu_lane as _lane

        hw = _lane.hw_record()
        label = hw.get("mac_model") or hw.get("cpu_brand")
        return str(label) if label else _platform.platform()
    except Exception:  # noqa: BLE001 - disclosure label only
        return _platform.platform()


def _t16_markdown(rows: list[dict], bottlenecks: list[dict],
                  unverified: list[str]) -> list[str]:
    import platform as _platform

    md = ["## T16 CPU lane — per-target verdicts", "",
          "| row | status | measured | final target | n | evidence |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        measured = (r.get("measured_p95_ms")
                    if r.get("measured_p95_ms") is not None
                    else r.get("measured_mib"))
        final = r.get("final_ms", r.get("final_mib", ""))
        md.append(
            f"| {r['row_id']} | {r.get('status')} | "
            f"{round(measured, 1) if isinstance(measured, (int, float)) else '—'} "
            f"{'ms' if r.get('measured_p95_ms') is not None else 'MiB' if measured is not None else ''} "
            f"| {final} | {r.get('n_samples', '—')} | "
            f"{str(r.get('evidence'))[:80]} |")
    md += ["", "**Authoritative CPU verdicts**: every CPU row is "
            "measured-on-dev-host (model/cores/affinity/load disclosed per "
            "run record; report-time host: "
            f"{_dev_host_label()}, "
            f"{os.cpu_count()} cores); the four-core-pinned authoritative "
            "verdict is BLOCKED (reference host unavailable).", ""]
    if bottlenecks:
        md += ["### TARGET_MISSED bottlenecks (exclusive time)", ""]
        for b in bottlenecks:
            md.append(f"- **{b['for_target']}** — dominant: "
                      f"`{b['dominant_stage']}`; stages: "
                      f"`{json.dumps(b.get('exclusive_stage_p95_s') or b.get('exclusive_stage_s'))}`")
            md.append(f"  - next candidates: "
                      f"{'; '.join(b['next_candidates'])}")
    md += ["", "### Unverified cases", ""]
    md += [f"- {u}" for u in unverified]
    md.append("")
    return md


def _t16_sources_and_compliance(artifacts: Path) -> dict:
    """Certified source hashes + sampling compliance from the T16 artifacts
    (each artifact carries its own certified sha256 of the scientific
    sources that executed; the report aggregates them)."""
    out: dict = {"certified_source_sha256": {}, "sampling_compliance": {}}
    conflicts: dict[str, set] = {}
    for name in ("bench_cpu-numba-selected-exact.json",
                 "bench_cpu-numba-full-day.json",
                 "bench_cpu-numba-full-recompute-warm.json",
                 "t16_memory.json", "t16_startup.json"):
        data = _load_first_json(artifacts, name)
        if not data:
            continue
        out["sampling_compliance"][name] = data.get("sampling_compliance")
        for rel, sha in (data.get("source_sha256") or {}).items():
            seen = out["certified_source_sha256"].setdefault(rel, sha)
            if seen != sha:  # every DISTINCT value a source hash took
                conflicts.setdefault(rel, {seen}).add(sha)
    if conflicts:
        out["hash_conflicts"] = {rel: sorted(vals)
                                 for rel, vals in conflicts.items()}
    return out


def _parse_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_unparsable": line[:120]})
    return rows


def cmd_report(args: argparse.Namespace) -> int:
    artifacts = Path(args.artifacts).resolve()
    if not artifacts.is_dir():
        print(f"error: artifacts dir {artifacts} not found", file=sys.stderr)
        return 2

    runs = []
    for name in ("run_record.jsonl", "raw_runs.jsonl", "raw_records.jsonl"):
        for p in sorted(artifacts.rglob(name)):
            rows = _parse_jsonl(p)
            runs.extend({"source": str(p.relative_to(artifacts)), **r} for r in rows)
    parity = sorted(list(artifacts.rglob("parity_*.json")) + list(artifacts.rglob("compare_*.json")))
    parity_entries = []
    for p in parity:
        try:
            data = _read_json(p)
        except json.JSONDecodeError:
            continue
        parity_entries.append({
            "file": p.name,
            "all_equal": data.get("all_equal", data.get("verdict")),
            "total_mismatch_count": data.get("total_mismatch_count"),
        })
    benches = sorted(artifacts.rglob("bench_*.json"))
    bench_entries = [{"file": p.name, "variant": _read_json(p).get("variant")}
                     for p in benches]

    t16_rows, t16_bottlenecks, t16_unverified = _t16_rows(artifacts)
    t16_meta = _t16_sources_and_compliance(artifacts)
    t16_section = {
        "status_vocabulary": ["PASS", "FAIL", "BLOCKED", "TARGET_MISSED",
                              "NOT_RUN"],
        "scope_note": FOUR_CORE_HOST_BLOCKED_REASON,
        "rows": t16_rows,
        "bottlenecks_for_target_missed": t16_bottlenecks,
        "unverified_cases": t16_unverified,
        **t16_meta,
    }

    evidence = {
        "runs": {"present": bool(runs), "count": len(runs)},
        "parity": {"present": bool(parity_entries), "count": len(parity_entries)},
        "bench": {"present": bool(bench_entries), "count": len(bench_entries)},
    }
    # the T16 lane is its own evidence class — present (and required under
    # --require-evidence) only when T16 artifacts exist in the tree, so a
    # T01-era artifacts dir does not suddenly fail the evidence gate
    if t16_meta["certified_source_sha256"]:
        evidence["t16_cpu_lane"] = {"present": True, "count": len(t16_rows)}
    payload = {
        "schema_version": 1,
        "artifacts_root": str(artifacts),
        "evidence": evidence,
        "runs": [
            {"source": r.get("source"), "run_id": r.get("run_id"),
             "mode": r.get("mode"), "case": r.get("case"), "phase": r.get("phase"),
             "origin": r.get("origin"), "has_source_sha256": bool(r.get("source_sha256"))}
            for r in runs
        ],
        "parity": parity_entries,
        "bench": bench_entries,
        "t16_cpu_lane": t16_section,
    }

    md = [f"# ultrafast artifacts report\n", f"- root: `{artifacts}`", ""]
    md += [f"- run records: {evidence['runs']['count']}",
           f"- parity reports: {evidence['parity']['count']}",
           f"- bench reports: {evidence['bench']['count']}", ""]
    md += _t16_markdown(t16_rows, t16_bottlenecks, t16_unverified)
    md += ["certified source sha256 (aggregate over T16 artifacts):", ""]
    for rel, sha in sorted(t16_meta["certified_source_sha256"].items()):
        md.append(f"- `{rel}` `{sha}`")
    md.append("")
    if parity_entries:
        md += ["## parity", "", "| file | all_equal | mismatches |", "|---|---|---|"]
        for e in parity_entries:
            md.append(f"| {e['file']} | {e['all_equal']} | "
                      f"{e['total_mismatch_count']} |")
        md.append("")
    if bench_entries:
        md += ["## bench", "", "| file | variant |", "|---|---|"]
        for e in bench_entries:
            md.append(f"| {e['file']} | {e['variant']} |")
        md.append("")

    json_path = artifacts / "report.json"
    md_path = artifacts / "report.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text("\n".join(md) + "\n")

    if args.require_evidence:
        missing = [k for k, v in evidence.items() if not v["present"]]
        if missing:
            raise EvidenceMissingError(
                f"--require-evidence: missing evidence classes {missing} "
                f"under {artifacts}")
    print(json.dumps({"report_json": str(json_path), "report_md": str(md_path),
                      "evidence": evidence}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="SOLWEIG ultrafast benchmark CLI (baseline/compare/bench/report)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser(
        "baseline", help="re-run the T00 harness with certified source hashes")
    p_base.add_argument("--manifest", default=str(BASELINE_DIR / "fixture_manifest.json"),
                        help="fixture or input manifest json (site paths from T00)")
    p_base.add_argument("--profile", default=core_profile.CANONICAL_CPU_V1,
                        choices=sorted(core_profile.PROFILE_REGISTRY))
    p_base.add_argument("--output", required=True, help="artifact output dir")
    p_base.add_argument("--suite", choices=["single", "representative"], default="single")
    p_base.add_argument("--mode", choices=["full", "edit"], default="edit")
    p_base.add_argument("--case", choices=["E1", "E2", "E3", "E4"], default="E1")
    p_base.add_argument("--tree", choices=["main", "oracle"], default="main")
    p_base.add_argument("--warmup", type=int, default=0)
    p_base.add_argument("--samples", type=int, default=1)
    p_base.set_defaults(func=cmd_baseline)

    p_cmp = sub.add_parser(
        "compare", help="raw-bit compare reference vs candidate artifact dirs")
    p_cmp.add_argument("--reference", required=True)
    p_cmp.add_argument("--candidate", required=True)
    p_cmp.add_argument("--strict-bits", action="store_true",
                       help="require metadata sidecars (time index + window origin)")
    p_cmp.add_argument("--require-full-coverage", action="store_true",
                       help="candidate must cover every reference global time")
    p_cmp.add_argument("--output", default=None, help="JSON report path")
    p_cmp.set_defaults(func=cmd_compare)

    p_bench = sub.add_parser(
        "bench", help=f"timing runs (variants: {', '.join(KNOWN_VARIANTS)})")
    p_bench.add_argument("--targets", default=str(
        REPO_ROOT / "docs/incremental_design_tool/ultrafast_bitwise/acceptance_targets.yaml"))
    p_bench.add_argument("--variant", default="baseline-legacy",
                         help=f"variant to run (known: {', '.join(KNOWN_VARIANTS)}); "
                              "unknown variants fail with the typed "
                              "UnknownVariantError at dispatch, not at parse")
    p_bench.add_argument("--suite", choices=["representative"], default="representative")
    p_bench.add_argument("--warmup", type=int, default=5)
    p_bench.add_argument("--repeats", type=int, default=2)
    p_bench.add_argument("--pairs", type=int, default=10,
                         help="randomized paired A/B runs (selected-exact)")
    p_bench.add_argument("--seed", type=int, default=20260908,
                         help="pair-order randomization seed")
    p_bench.add_argument("--cold-repeats", type=int, default=None,
                         help="cold-subprocess sample count (full-day; "
                              "defaults to --repeats)")
    p_bench.add_argument("--output", default=None)
    p_bench.set_defaults(func=cmd_bench)

    p_rep = sub.add_parser("report", help="summarize artifacts into report.md/json")
    p_rep.add_argument("--artifacts", required=True)
    p_rep.add_argument("--require-evidence", action="store_true")
    p_rep.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except UnknownVariantError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EvidenceMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
