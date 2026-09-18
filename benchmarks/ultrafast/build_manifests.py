"""T00 manifest builder: environment, inputs, fixtures -> benchmarks/ultrafast/baseline/.

Records digests and read-only paths only; rasters are never copied into the
repository tree. Run from any cwd with the main-repo venv python:

    python benchmarks/ultrafast/build_manifests.py --repo-root /Users/alansynn/Workspace/solweig
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path, repo_root: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sysctl(name: str) -> str:
    out = subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def build_environment(venv_python: Path) -> dict:
    probe = subprocess.run(
        [str(venv_python), "-c", (
            "import json, numpy, torch, numba, sys;"
            "cfg = numpy.show_config(mode='dicts') if hasattr(numpy,'show_config') else {};"
            "print(json.dumps({'python': sys.version.split()[0],"
            "'numpy': numpy.__version__, 'torch': torch.__version__,"
            "'numba': numba.__version__,"
            "'torch_threads': torch.get_num_threads(),"
            "'torch_interop_threads': torch.get_num_interop_threads(),"
            "'cuda_available': torch.cuda.is_available()}))"
        )],
        capture_output=True, text=True, check=True,
    )
    versions = json.loads(probe.stdout.strip().splitlines()[-1])
    blas = "unknown"
    cfg_probe = subprocess.run(
        [str(venv_python), "-c", "import numpy; numpy.show_config()"],
        capture_output=True, text=True,
    )
    for line in cfg_probe.stdout.splitlines():
        if "name:" in line and "accelerate" in line:
            blas = "accelerate"
            break
        if "name:" in line and "openblas" in line.lower():
            blas = line.split("name:")[1].strip()
            break
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_purpose": "T00 baseline host (not the acceptance 4-core pinned reference)",
        "cpu_model": sysctl("machdep.cpu.brand_string"),
        "physical_cores": int(sysctl("hw.physicalcpu")),
        "logical_cores": int(sysctl("hw.logicalcpu")),
        "memory_bytes": int(sysctl("hw.memsize")),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.mac_ver()[0],
        },
        "blas": blas,
        "venv_python": str(venv_python),
        "packages": versions,
    }


def build_inputs(repo_root: Path) -> dict:
    subset = repo_root / "Input_subset"
    cache_dir = repo_root / "site-cache" / "site_500"
    rasters = {}
    for name in sorted(p.name for p in subset.glob("*.tif")):
        rasters[name] = file_entry(subset / name, repo_root)
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    return {
        "policy": (
            "read-only private data; digests recorded in-repo, rasters stay "
            "outside version control at their original paths"
        ),
        "origin_rasters_Input_subset": rasters,
        "site_cache": {
            "path": str(cache_dir),
            "manifest_sha256": sha256_file(cache_manifest_path),
            "arrays": cache_manifest.get("arrays", {}),
            "component_bytes": {
                d.name: sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                for d in sorted(cache_dir.iterdir()) if d.is_dir()
            },
        },
        "oracle_worktree_inputs": {
            "root": "/Users/alansynn/Workspace/solweig_oracle_e0d19fc",
            "pinned_commit": "e0d19fc98220a333ea312efe4604780803bbdd17",
            "site_cache_manifest_sha256": sha256_file(
                Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc/site-cache/site_500/manifest.json")
            ),
            "venv": "symlink -> /Users/alansynn/Workspace/solweig/.venv (editable install of main tree; import origin pinned by harness sys.path.insert of cwd)",
        },
    }


FIXTURES = {
    "representative_tree_edit": {
        "description": "single 10 m tree (r=4) at open cell S01 (row 166, col 102), site_500, selected_date 2009-08-11",
        "base": "site-cache/site_500 cold cache (no warm-start state)",
        "cell": [166, 102],
        "height_m": 10.0,
        "canopy_radius_m": 4.0,
    },
    "E1": "add t1 h10 r4 at (166,102)",
    "E2": "add t1 at (166,102) then move to (166,127) — 50 m east; pending batch carries add+move",
    "E3": "add t1 h10 r4 at (166,102) then height replace to 15.0 m",
    "E4": "add t1 at (166,102) and t2 (h10 r4) at (166,407) — 610 m east (305 px * 2 m)",
    "synthetic_nonsquare_40x80": {
        "generator": "benchmarks/ultrafast/make_synth_fixtures.py --shape 40 80",
        "description": "40x80 non-square synthetic scene (rows=40, cols=80), flat DEM + one building block + one tree; arrays land in the artifact root, never the repo tree",
        "status": "GENERATOR_DEFINED_TRACES_PENDING_T01",
    },
    "synthetic_one_step_4m": {
        "generator": "benchmarks/ultrafast/make_synth_fixtures.py --shape 32 48 --pixel-size 4.0 --one-step",
        "description": "4 m pixel one-time-step witness fixture for single-step march trace capture",
        "status": "GENERATOR_DEFINED_TRACES_PENDING_T01",
    },
    "met_warm_start_sequence": {
        "description": "one met warm-start sequence (E5-class time-prefix recompute)",
        "status": "NOT_RUN_T00_DEFERRED",
        "note": "E5 was not in the T00 E1-E4 run set; deferred to T01 harness work",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/Users/alansynn/Workspace/solweig")
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    out_dir = repo / "benchmarks" / "ultrafast" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_environment(repo / ".venv" / "bin" / "python")
    (out_dir / "reference_environment.json").write_text(json.dumps(env, indent=2))

    inputs = build_inputs(repo)
    (out_dir / "input_manifest.json").write_text(json.dumps(inputs, indent=2))

    (out_dir / "fixture_manifest.json").write_text(json.dumps({
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid_reference": {
            "rows": 500, "cols": 500, "pixel_size_m": 2.0,
            "date": "2009-08-11", "timesteps": 24,
            "requested_variables": ["utci", "tmrt", "shadow"],
        },
        "cases": FIXTURES,
    }, indent=2))
    print("wrote reference_environment.json, input_manifest.json, fixture_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
