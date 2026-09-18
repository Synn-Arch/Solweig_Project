# SPDX-License-Identifier: GPL-3.0-only
"""T12 — source-level mutation matrix runner (TASKS T12; RED witnesses).

Orchestrates FROM THE LOCAL WORKTREE, applies mutations ONLY on the GPU
host's repo overlay, and never leaves the overlay mutated:

  pristine backup (local, BEFORE anything)  ->  push mutated source
  -> rebuild + designated killer pytest MUST FAIL
  -> restore pristine source (rsync + sha256 verify)
  -> purge the variant build dir (the stamp cache would otherwise reuse
     the mutated libswcuda.so: stamps are per-digest but the lib path is
     fixed per variant)
  -> sanity pytest on the SAME test file MUST PASS
  -> evidence JSON + logs under the task artifacts root.

Gate: refuses to touch the remote unless SW_CUDA_MUTATIONS=1 (or
--apply); default is a dry run that still verifies every old-string is
unique in the pristine worktree file.

Usage (local):
  SW_CUDA_MUTATIONS=1 python tests/ultrafast/run_cuda_mutations.py
  SW_CUDA_MUTATIONS=1 python tests/ultrafast/run_cuda_mutations.py \
      --only mut_c_expf_approx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

REMOTE_HOST = "cura"
REMOTE_OVERLAY = "~/solweig_ultrafast/work/t12/repo_overlay"
REMOTE_PY = "~/solweig_ultrafast/py"
GPU_ORDINAL = "0"  # pinned for this run; recorded in every evidence file

ART_ROOT = (Path.home() / "Workspace" / "solweig_ultrafast_artifacts" /
            "t12" / "mutations")

PYTEST_ARGS = ["-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider"]


# ---------------------------------------------------------------------------
# the matrix: every mutation is a single surgical source edit whose ONLY
# effect is the RED witness it stands for, plus the designated killer test
# ---------------------------------------------------------------------------
MUTATIONS = [
    {
        "name": "mut_a_fold_patch_reorder",
        "witness": 5,
        "witness_desc": "FP reduction reordering (reversed patch reduction "
                        "of the SVF fold accumulators — RED-designated in "
                        "sw_fold.cu's module contract)",
        "file": "native/cuda/src/sw_fold.cu",
        "old": "    for (int p = 0; p < SW_FOLD_PATCHES; ++p) {",
        "new": "    for (int p = SW_FOLD_PATCHES - 1; p >= 0; --p) {"
               "  // MUTATION a",
        "killer": "tests/ultrafast/test_cuda_fold.py",
    },
    {
        "name": "mut_b_utci_schedule_leak",
        "witness": 7,
        "witness_desc": "compacted-context/schedule dependence: the libm "
                        "lane class leaks threadIdx into a value that must "
                        "depend only on (inputs, k, n)",
        "file": "native/cuda/src/sw_utci.cu",
        "old": "    bool use_libm = sw_is_libm_lane(k, n, T, grain);",
        "new": "    bool use_libm = sw_is_libm_lane(k, n, T, grain) ||\n"
               "                     (threadIdx.x < 8);  // MUTATION b",
        "killer": "tests/ultrafast/test_cuda_utci.py",
    },
    {
        "name": "mut_c_expf_approx",
        "witness": 3,
        "witness_desc": "approximate intrinsic: __expf replaces the SLEEF "
                        "xexpf port at the head of sw_sleef_expf",
        "file": "native/cuda/include/sw_strict_math.cuh",
        "old": "__device__ __forceinline__ float sw_sleef_expf(float d) {\n"
               "    const float R_LN2F",
        "new": "__device__ __forceinline__ float sw_sleef_expf(float d) {\n"
               "    return __expf(d);  // MUTATION c: approximate intrinsic\n"
               "    const float R_LN2F",
        "killer": "tests/ultrafast/test_cuda_primitives.py",
    },
    {
        "name": "mut_d_rad_fma_contract",
        "witness": 1,
        "witness_desc": "FMA contraction: explicit __fmaf_rn in the "
                        "TsWaveDelay blend (bypasses --fmad=false by "
                        "construction — the point of the witness)",
        "file": "native/cuda/src/sw_radiation.cu",
        "old": "    float omw = 1.0f - w1;\n"
               "    return src * omw + m_in * w1;",
        "new": "    float omw = 1.0f - w1;\n"
               "    return __fmaf_rn(src, omw, m_in * w1);  // MUTATION d",
        "killer": "tests/ultrafast/test_cuda_radiation.py",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ssh(cmd: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", REMOTE_HOST, cmd], capture_output=True, text=True,
        timeout=timeout,
    )


def remote_sha(rel_path: str) -> str | None:
    proc = ssh(
        f"sha256sum {REMOTE_OVERLAY}/{rel_path} 2>/dev/null", timeout=60)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.split()[0]


def rsync_push(local: Path, rel_path: str) -> None:
    proc = subprocess.run(
        ["rsync", "-a", str(local), f"{REMOTE_HOST}:{REMOTE_OVERLAY}/{rel_path}"],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rsync push failed: {proc.stderr}")


def gpu_idle(max_util: int = 5, max_mib: int = 1000) -> tuple[bool, str]:
    proc = ssh(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
        "--format=csv,noheader,nounits", timeout=60)
    if proc.returncode != 0:
        return False, f"nvidia-smi failed: {proc.stderr.strip()}"
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0] == GPU_ORDINAL:
            util, mem = int(parts[1]), int(parts[2])
            ok = util <= max_util and mem <= max_mib
            return ok, f"gpu{GPU_ORDINAL}: util={util}% mem={mem}MiB"
    return False, f"gpu {GPU_ORDINAL} not found in nvidia-smi output"


def wait_gpu_idle(attempts: int = 12, delay: float = 10.0) -> str:
    for _ in range(attempts):
        ok, status = gpu_idle()
        if ok:
            return status
        print(f"    [wait] {status} — retrying in {delay:.0f}s")
        time.sleep(delay)
    raise RuntimeError(f"GPU {GPU_ORDINAL} never went idle: {status}")


def run_remote_pytest(test_rel: str, log_path: Path,
                      timeout: int = 3600) -> tuple[int, str]:
    """Run the designated pytest on the overlay with SW_REQUIRE_CUDA=1.

    The mutated/rebuilt library is produced inside this same run (host.py
    rebuilds when the source digest changes). Returns (rc, output).
    """
    status = wait_gpu_idle()
    cmd = (
        f"cd {REMOTE_OVERLAY} && SW_REQUIRE_CUDA=1 "
        f"CUDA_VISIBLE_DEVICES={GPU_ORDINAL} "
        f"{REMOTE_PY} {' '.join(PYTEST_ARGS)} {test_rel} 2>&1"
    )
    t0 = time.time()
    proc = ssh(cmd, timeout=timeout)
    dt = time.time() - t0
    out = (f"$ ssh {REMOTE_HOST} '{cmd}'\n"
           f"# gpu idle check: {status}\n"
           f"# exit={proc.returncode} elapsed={dt:.1f}s\n\n"
           f"{proc.stdout}\n{proc.stderr}")
    log_path.write_text(out)
    return proc.returncode, out


def purge_remote_build() -> None:
    """Remove the canonical variant build dir (lib + all per-digest stamps).

    Mandatory before the post-restore sanity run: the stamp cache is keyed
    by input digest but the library path is fixed, so a stamp hit after
    restore would silently reuse the MUTATED libswcuda.so.
    """
    proc = ssh(f"rm -rf {REMOTE_OVERLAY}/native/cuda/build/canonical",
               timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to purge remote build dir: {proc.stderr}")


def prepare_backups() -> None:
    """Local pristine copies (copy BEFORE any mutation), sha256 recorded."""
    ART_ROOT.mkdir(parents=True, exist_ok=True)
    for m in MUTATIONS:
        d = ART_ROOT / m["name"] / "pristine"
        d.mkdir(parents=True, exist_ok=True)
        src = REPO / m["file"]
        dest = d / Path(m["file"]).name
        if not dest.is_file():
            dest.write_bytes(src.read_bytes())
        pristine = (REPO / m["file"]).read_bytes()
        backed = dest.read_bytes()
        if backed != pristine:
            # the worktree file changed since a previous run — refresh and
            # note it (the backup must mirror the pristine worktree exactly)
            dest.write_bytes(pristine)
        m["pristine_sha256"] = sha256_bytes(pristine)


def check_edit_sites() -> None:
    """Dry-run safety: every old-string occurs EXACTLY once in the pristine
    worktree file, and the edit produces a changed file."""
    for m in MUTATIONS:
        text = (REPO / m["file"]).read_text()
        n = text.count(m["old"])
        assert n == 1, (
            f"{m['name']}: old-string occurs {n}x in {m['file']} "
            f"(must be exactly 1)"
        )
        assert m["new"] != m["old"], f"{m['name']}: no-op edit"


def run_mutation(m: dict, apply: bool) -> dict:
    print(f"\n=== {m['name']} (witness {m['witness']}: "
          f"{m['witness_desc']})")
    ev_dir = ART_ROOT / m["name"]
    ev_dir.mkdir(parents=True, exist_ok=True)
    pristine = (REPO / m["file"]).read_bytes()
    pristine_sha = sha256_bytes(pristine)
    mutated = pristine.decode().replace(m["old"], m["new"]).encode()
    mutated_sha = sha256_bytes(mutated)
    (ev_dir / "pristine" / Path(m["file"]).name).write_bytes(pristine)

    evidence = {
        "name": m["name"],
        "witness": m["witness"],
        "witness_desc": m["witness_desc"],
        "file": m["file"],
        "killer_test": m["killer"],
        "gpu_ordinal": GPU_ORDINAL,
        "pristine_sha256": pristine_sha,
        "mutated_sha256": mutated_sha,
        "remote_pre_sha256": None,
        "remote_restored_sha256": None,
        "killed": None,
        "killer_rc": None,
        "sanity_rc": None,
        "sanity_pass": None,
        "restored_bits_match": None,
    }

    if not apply:
        print(f"    [dry-run] would mutate {m['file']} "
              f"({len(pristine)} -> {len(mutated)} bytes), "
              f"killer={m['killer']}")
        (ev_dir / "mutation_dryrun.json").write_text(
            json.dumps(evidence, indent=2))
        return evidence

    rsha = remote_sha(m["file"])
    evidence["remote_pre_sha256"] = rsha
    if rsha != pristine_sha:
        raise RuntimeError(
            f"{m['name']}: remote {m['file']} sha {rsha} != worktree "
            f"{pristine_sha} — overlay out of sync; sync before mutating"
        )

    mutated_tmp = ev_dir / (Path(m["file"]).name + ".mutated")
    mutated_tmp.write_bytes(mutated)

    try:
        # push mutation -> killer MUST fail
        rsync_push(mutated_tmp, m["file"])
        got = remote_sha(m["file"])
        assert got == mutated_sha, f"push verify failed: {got}"
        print(f"    mutated pushed (sha {mutated_sha[:16]}…)")

        rc, _ = run_remote_pytest(
            m["killer"], ev_dir / "killer_run.log")
        evidence["killer_rc"] = rc
        evidence["killed"] = rc != 0
        print(f"    killer {m['killer']}: exit={rc} "
              f"-> {'KILLED' if rc != 0 else 'SURVIVED (!!)'}")
    finally:
        # restore pristine no matter what
        pristine_tmp = ev_dir / (Path(m["file"]).name + ".pristine_restore")
        pristine_tmp.write_bytes(pristine)
        rsync_push(pristine_tmp, m["file"])
        got = remote_sha(m["file"])
        evidence["remote_restored_sha256"] = got
        evidence["restored_bits_match"] = (got == pristine_sha)
        print(f"    restored (sha match: {got == pristine_sha})")
        mutated_tmp.unlink(missing_ok=True)
        pristine_tmp.unlink(missing_ok=True)

    if not evidence["restored_bits_match"]:
        raise RuntimeError(
            f"{m['name']}: restore failed — remote bits differ from pristine"
        )

    # sanity on the SAME test file, on a purged build dir (forces a real
    # rebuild from the pristine sources — no stale mutated .so reuse)
    purge_remote_build()
    rc, _ = run_remote_pytest(m["killer"], ev_dir / "sanity_run.log")
    evidence["sanity_rc"] = rc
    evidence["sanity_pass"] = rc == 0
    print(f"    sanity {m['killer']}: exit={rc} "
          f"-> {'PASS' if rc == 0 else 'FAIL (!!)'}")

    (ev_dir / "mutation.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=None,
                        help="run only these mutation names (repeatable)")
    parser.add_argument("--apply", action="store_true",
                        help="actually touch the remote (else dry-run)")
    args = parser.parse_args()

    apply_ = args.apply or os.environ.get("SW_CUDA_MUTATIONS") == "1"
    check_edit_sites()
    prepare_backups()
    print(f"artifacts root: {ART_ROOT}")
    print(f"mode: {'APPLY (remote mutations enabled)' if apply_ else 'DRY RUN'}")

    sel = [m for m in MUTATIONS
           if args.only is None or m["name"] in args.only]
    unknown = (set(args.only or []) - {m["name"] for m in MUTATIONS})
    if unknown:
        parser.error(f"unknown mutation(s): {sorted(unknown)}")

    results = [run_mutation(m, apply_) for m in sel]

    summary = {
        "mode": "apply" if apply_ else "dry-run",
        "remote_host": REMOTE_HOST,
        "remote_overlay": REMOTE_OVERLAY,
        "gpu_ordinal": GPU_ORDINAL,
        "mutations": results,
        "all_killed": all(r["killed"] for r in results if r["killed"] is not None),
        "all_sane": all(r["sanity_pass"] for r in results
                        if r["sanity_pass"] is not None),
    }
    (ART_ROOT / "mutation_summary.json").write_text(json.dumps(summary,
                                                               indent=2))
    print("\n=== summary")
    for r in results:
        print(f"  {r['name']}: killed={r['killed']} "
              f"sanity_pass={r['sanity_pass']} "
              f"restored={r['restored_bits_match']}")
    print(f"  -> {ART_ROOT / 'mutation_summary.json'}")
    if apply_:
        ok = summary["all_killed"] and summary["all_sane"] and all(
            r["restored_bits_match"] for r in results)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
