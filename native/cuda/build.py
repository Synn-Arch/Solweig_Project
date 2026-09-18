# SPDX-License-Identifier: GPL-3.0-only
"""nvcc build driver for the ultrafast CUDA kernels (TASKS T12).

Torch-free: build configuration depends only on the filesystem, nvcc and
the environment. Builds ``build/<variant>/libswcuda.so`` from every
``src/*.cu`` (plus the generated UTCI poly header under ``include/``).

Variants (DESIGN.ko.md 6.2 — the RED-witness build mutations):

===================  ====================================================
variant              nvcc flags
===================  ====================================================
canonical            --fmad=false --ftz=false --prec-div=true
                     --prec-sqrt=true   (the certified profile)
mut_fmad             canonical + --fmad=true        (contraction witness)
mut_ftz              canonical + --ftz=true         (subnormal witness)
mut_fastmath         canonical + --use_fast_math    (approx intrinsics)
mut_expf             canonical + -DSW_MUT_APPROX_EXPF  (__expf swap)
mut_libm_log         canonical + -DSW_MUT_LIBM_LOGF   (device libm log)
legacy_cuda_v1       --fmad=true (aten-default-like; UNCERTIFIED —
                     characterization only, never a parity target)
===================  ====================================================

The effective flag string is baked into the object (``-DSW_BUILD_FLAG_STRING``)
and reported through the C ABI (``sw_build_flags``) so a test can assert it
loaded the variant it asked for.

Caching: the build directory is keyed by the sha256 of (sources, headers,
flags, arch); a rebuilt-on-demand model with no timestamp guessing.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
SRC_DIR = PKG_DIR / "src"
INCLUDE_DIR = PKG_DIR / "include"
BUILD_ROOT = PKG_DIR / "build"

DEFAULT_ARCH = os.environ.get("SW_CUDA_ARCH", "sm_86")

CANONICAL_FLAGS = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
]

#: variant -> extra nvcc flags (canonical base is always included except
#: where a variant explicitly overrides a canonical flag)
VARIANTS: dict[str, list[str]] = {
    "canonical": [],
    # RED witness 1: FMA contraction enabled
    "mut_fmad": ["--fmad=true"],
    # RED witness 2: FTZ difference
    "mut_ftz": ["--ftz=true"],
    # RED witness 3: approximate intrinsics (__expf etc. compiled from
    # plain expf calls under fast-math; mut_expf swaps explicitly)
    "mut_fastmath": ["--use_fast_math"],
    # RED witness 3 (explicit swap): __expf in place of the SLEEF port
    "mut_expf": ["-DSW_MUT_APPROX_EXPF"],
    # RED witness 4: CPU/GPU library discrepancy on transcendentals
    "mut_libm_log": ["-DSW_MUT_LIBM_LOGF"],
    # legacy profile characterization build (UNCERTIFIED)
    "legacy_cuda_v1": ["--fmad=true"],
}

#: builds whose flag string the host must refuse to certify as canonical
UNCERTIFIED_VARIANTS = frozenset({"legacy_cuda_v1"}) | {
    name for name in VARIANTS if name.startswith("mut_")
}


def find_nvcc() -> str:
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            nvcc = str(candidate)
    if not nvcc:
        raise FileNotFoundError(
            "nvcc not found (set NVCC or CUDA_HOME, or extend PATH)"
        )
    return nvcc


def variant_flags(variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown build variant {variant!r}; known: {sorted(VARIANTS)}"
        )
    flags = list(CANONICAL_FLAGS)
    for flag in VARIANTS[variant]:
        if flag.startswith("--fmad=") or flag.startswith("--ftz="):
            flags = [f for f in flags
                     if not f.startswith(flag.split("=")[0] + "=")]
        flags.append(flag)
    return flags


def _hash_inputs(sources: list[Path], flags: list[str], arch: str) -> str:
    h = hashlib.sha256()
    h.update(arch.encode())
    h.update("|".join(flags).encode())
    for src in sorted(sources):
        h.update(str(src.relative_to(PKG_DIR)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(src.read_bytes()).digest())
    for hdr in sorted(INCLUDE_DIR.glob("*.cuh")):
        h.update(str(hdr.relative_to(PKG_DIR)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(hdr.read_bytes()).digest())
    return h.hexdigest()


def build(
    variant: str = "canonical",
    *,
    arch: str | None = None,
    verbose: bool = False,
    force: bool = False,
    log_dir: Path | None = None,
) -> Path:
    """Compile ``build/<variant>/libswcuda.so``; returns its path.

    ``log_dir`` (default ``<variant build dir>``) receives build.log —
    the caller archives it under the task artifacts root.
    """
    arch = arch or DEFAULT_ARCH
    flags = variant_flags(variant)
    sources = sorted(SRC_DIR.glob("*.cu"))
    if not sources:
        raise FileNotFoundError(f"no .cu sources under {SRC_DIR}")

    digest = _hash_inputs(sources, flags, arch)
    out_dir = BUILD_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = out_dir / "libswcuda.so"
    stamp = out_dir / f"{digest}.stamp"
    if lib.is_file() and stamp.is_file() and not force:
        return lib

    nvcc = find_nvcc()
    cmd = [
        nvcc,
        f"-arch={arch}",
        *flags,
        "-O3",
        "-std=c++17",
        "--compiler-options",
        "-fPIC",
        "-shared",
        f'-DSW_BUILD_FLAG_STRING="{arch} {" ".join(flags)}"',
        f"-I{INCLUDE_DIR}",
        *[str(s) for s in sources],
        "-o",
        str(lib),
    ]
    log_path = (log_dir or out_dir) / "build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PKG_DIR)
    )
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\n"
        f"exit={proc.returncode}\n\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )
    if verbose:
        print(f"[build:{variant}] {' '.join(cmd)}", file=sys.stderr)
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvcc failed for variant {variant} (exit {proc.returncode}); "
            f"see {log_path}"
        )
    stamp.write_text(digest)
    return lib


def library_path(variant: str = "canonical") -> Path:
    return BUILD_ROOT / variant / "libswcuda.so"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="canonical",
                        choices=sorted(VARIANTS))
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    out = build(args.variant, arch=args.arch, force=args.force,
                verbose=args.verbose)
    print(out)
