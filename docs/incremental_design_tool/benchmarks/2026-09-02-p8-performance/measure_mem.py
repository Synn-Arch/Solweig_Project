"""P8.4 MEM-001/MEM-002 + IO-001: worker RSS and client patch size (site_500).

Protocol (cpu_optimization.md): resource.getrusage for peak RSS (ru_maxrss),
ps(1) sampling for current RSS. Runs the typical one-tree job (10 m / r=4 at
the open S01 cell) through the real ExactWorker (solve + patch publication)
ten times with distinct positions, then:

* MEM-001: peak RSS across the whole run;
* MEM-002: steady current RSS after the jobs (cache warm), a leak check
  (RSS drift between jobs 5 and 10), and a fresh-process rerun for
  restart-equivalence;
* IO-001: zstd + identity byte size of the published typical patch.
"""
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path("/Users/alansynn/Workspace/solweig/.claude/worktrees/p8-perf")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "tests"))

CACHE_DIR = Path("/tmp/p5sci/cache_alt3")
SITE_DIR = Path("/Users/alansynn/Workspace/solweig/Input_subset/processed_inputs")
DATE_STR = "2009-08-11"
JOBS = 10


def current_rss_bytes() -> int:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip()) * 1024  # ps rss is KiB


def peak_rss_bytes() -> int:
    # ru_maxrss is bytes on macOS, KiB on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def run_jobs(report: dict) -> list[float]:
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.geometry import RasterGrid, TreeSpec
    from solweig_gpu.incremental.result import load_patch
    from solweig_gpu.incremental.trees import TreeLayer
    from solweig_gpu.incremental.worker import ExactWorker

    t0 = time.perf_counter()
    cache = SiteCache.load(CACHE_DIR)
    report["rss_after_cache_load_mb"] = round(current_rss_bytes() / 1e6, 1)

    grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                      cache.manifest.origin_x_m, cache.manifest.origin_y_m)
    y = grid.origin_y_m - 166.5 * grid.pixel_size_m

    results_root = Path("/tmp/p8perf/mem_results")
    job_rss: list[float] = []
    first_patch_dir: Path | None = None
    for i in range(JOBS):
        # A fresh layer per job with a distinct position: successive edits,
        # nothing no-ops, every job does the full local solve + publish.
        layer = TreeLayer(cache.tree_base, grid)
        layer.add_tree(TreeSpec(
            f"t{i}", grid.origin_x_m + (102.5 + i) * grid.pixel_size_m, y, 10.0, 4.0))
        worker = ExactWorker(cache, layer, site_dir=SITE_DIR,
                             results_root=results_root, selected_date_str=DATE_STR)
        outcome = worker.run(job_id=f"memjob-{i}")
        assert outcome.status == "published", outcome
        rss = current_rss_bytes()
        job_rss.append(rss / 1e6)
        if i == 0:
            report["first_job_seconds_incl_cache_load"] = round(time.perf_counter() - t0, 1)
            report["first_job_mode"] = outcome.mode
            first_patch_dir = Path(outcome.patch_paths[0])
    report["rss_per_job_mb"] = [round(v, 1) for v in job_rss]
    report["rss_drift_jobs5_to_10_mb"] = round(job_rss[-1] - job_rss[4], 1)

    # IO-001: byte size of the published typical patch on the wire.
    from solweig_gpu.server import patch_codec

    patch = load_patch(first_patch_dir)
    variables = [name for name in patch.variables if name in patch_codec.SUPPORTED_VARIABLES]
    _meta, payload, _checksum = patch_codec.encode_payload(patch.arrays, variables)
    identity_bytes = sum(int(arr.nbytes) for arr in patch.arrays.values())
    report["patch_variables"] = variables
    report["patch_shape_t_hw"] = [int(d) for d in patch.arrays[variables[0]].shape]
    report["io_patch_zstd_mb"] = round(len(payload) / 1e6, 3)
    report["io_patch_identity_mb"] = round(identity_bytes / 1e6, 3)
    report["gate_IO_001_lt_5mb"] = len(payload) < 5_000_000
    return job_rss


def main() -> None:
    report: dict = {
        "machine": "Apple M1 Pro 10-core/16GB, py3.14, torch 2.13 CPU",
        "site": "site_500 500x500@2m", "date": DATE_STR, "jobs": JOBS,
    }
    run_jobs(report)

    import gc

    gc.collect()
    time.sleep(2.0)
    report["steady_idle_rss_mb"] = round(current_rss_bytes() / 1e6, 1)
    report["peak_rss_mb"] = round(peak_rss_bytes() / 1e6, 1)
    report["gate_MEM_001_peak_lt_3000mb"] = report["peak_rss_mb"] < 3000.0
    report["gate_MEM_002_idle_lt_1500mb"] = report["steady_idle_rss_mb"] < 1500.0
    report["leak_bounded_abs_drift_lt_150mb"] = abs(report["rss_drift_jobs5_to_10_mb"]) < 150.0

    # Restart check: a fresh process must reach the same steady numbers
    # (nothing accumulates across restarts; state lives on disk).
    code = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "from solweig_gpu.incremental.cache import SiteCache\n"
        "cache = SiteCache.load(%r)\n"
        "import json, subprocess, os\n"
        "rss = int(subprocess.run(['ps','-o','rss=','-p',str(os.getpid())],"
        "capture_output=True,text=True).stdout) * 1024\n"
        "print(json.dumps({'fresh_cache_load_rss_mb': round(rss/1e6, 1)}))\n"
    ) % (str(WORKTREE), str(WORKTREE / "tests"), str(CACHE_DIR))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    report["restart_fresh_cache_load"] = json.loads(out.stdout.strip().splitlines()[-1])

    print(json.dumps(report, indent=2))
    Path("/tmp/p8perf/mem_io_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
