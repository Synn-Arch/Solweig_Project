#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate include/sw_utci_poly_generated.cuh from the CPU canonical UTCI
polynomial source (ultrafast T12).

SINGLE SOURCE OF TRUTH: ``solweig_core/numba_cpu/utci.py::
POLY_EXPRESSION_SOURCE`` — the 211-statement block generated from the
oracle AST (T09). This tool transpiles it to CUDA C++:

* ``F32(<decimal literal>)``  -> exact hex float literal (float.hex() of
  the float32 value — decimal parsing can never disagree between hosts)
* ``nadd/nsub/nmul/ndiv(``    -> ``sw_nadd/sw_nsub/sw_nmul/sw_ndiv(``
* ``_torch_pow_scalar(``      -> ``sw_torch_pow_scalar(``

The generated header is COMMITTED; tests/ultrafast/test_cuda_utci.py
re-runs this generator into a temp file and asserts byte equality, so
the committed artifact can never drift from the CPU canonical source.

No numba import: the statement block is read out of utci.py via ``ast``
(text-only), so the generator runs in any torch/numba-free environment.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
UTCI_PY = REPO_ROOT / "solweig_core" / "numba_cpu" / "utci.py"
OUT_HEADER = REPO_ROOT / "native" / "cuda" / "include" / "sw_utci_poly_generated.cuh"

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RE_NEG_F32 = re.compile(rf"\(-F32\(({_NUM})\)\)")
_RE_F32 = re.compile(rf"\bF32\(({_NUM})\)")
_RE_STMT = re.compile(r"^acc = nadd\(acc, (?:nmul\(.*\)|F32\(.+\)|_torch_pow_scalar\(.*\))\)$")

_HEADER = """\
// GENERATED FILE — DO NOT EDIT.
// Produced by native/cuda/tools/gen_utci_cuda.py from
// solweig_core/numba_cpu/utci.py POLY_EXPRESSION_SOURCE (T09 oracle-AST
// statement block, the CPU canonical UTCI polynomial). Regenerate with:
//   python native/cuda/tools/gen_utci_cuda.py
// tests/ultrafast/test_cuda_utci.py asserts byte equality with a fresh
// regeneration, so this file cannot drift from the CPU source.
#ifndef SW_UTCI_POLY_GENERATED_CUH_
#define SW_UTCI_POLY_GENERATED_CUH_

#include "sw_strict_math.cuh"

// The 211-statement left-associated accumulation chain. Statement order,
// operand order and every rounding site mirror POLY_EXPRESSION_SOURCE
// exactly; only the primitive names and literal spellings differ.
__device__ __forceinline__ float sw_utci_poly_element(
        float dtm, float ta, float va, float pa, bool use_libm) {
"""

_FOOTER = """\
    return acc;
}

#endif  // SW_UTCI_POLY_GENERATED_CUH_
"""


def f32_hex(literal: str) -> str:
    """Exact C hex-float literal for the float32 nearest ``literal``."""
    import numpy as np

    value = np.float32(literal)
    f = float(value)
    if f == 0.0:
        sign = "-" if str(literal).lstrip().startswith("-") else ""
        return f"{sign}0x0p+0f"
    return f.hex() + "f"


def extract_poly_source(text: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "POLY_EXPRESSION_SOURCE":
                    if not isinstance(node.value, ast.Constant):
                        raise ValueError("POLY_EXPRESSION_SOURCE is not a plain string")
                    return node.value.value
    raise ValueError("POLY_EXPRESSION_SOURCE assignment not found in utci.py")


def transform_line(line: str) -> str:
    line = _RE_NEG_F32.sub(lambda m: f"(-{f32_hex(m.group(1))})", line)
    line = _RE_F32.sub(lambda m: f32_hex(m.group(1)), line)
    for py, cu in (("nmul", "sw_nmul"), ("nsub", "sw_nsub"),
                   ("ndiv", "sw_ndiv"), ("nadd", "sw_nadd")):
        line = re.sub(rf"\b{py}\(", f"{cu}(", line)
    line = line.replace("_torch_pow_scalar(", "sw_torch_pow_scalar(")
    return line + ";"


def generate() -> str:
    source_block = extract_poly_source(UTCI_PY.read_text())
    out = [_HEADER]
    n_stmt = 0
    for raw in source_block.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if n_stmt == 0:
            if line != "acc = ta":
                raise ValueError(f"expected initializer 'acc = ta', got {line!r}")
            out.append("    float acc = ta;")
            n_stmt += 1
            continue
        if not _RE_STMT.match(line):
            raise ValueError(f"unexpected POLY statement shape: {line!r}")
        out.append("    " + transform_line(line))
        n_stmt += 1
    if n_stmt != 211:
        raise ValueError(f"expected 211 poly statements, got {n_stmt}")
    out.append(_FOOTER)
    return "".join(l + "\n" for l in out)


def main(argv: list[str]) -> int:
    emit = generate()
    if len(argv) > 1 and argv[1] == "--check":
        current = OUT_HEADER.read_text()
        if current != emit:
            print(
                f"STALE: {OUT_HEADER} differs from a fresh regeneration",
                file=sys.stderr,
            )
            return 1
        print("OK: generated header matches POLY_EXPRESSION_SOURCE")
        return 0
    OUT_HEADER.parent.mkdir(parents=True, exist_ok=True)
    OUT_HEADER.write_text(emit)
    print(f"wrote {OUT_HEADER} ({len(emit)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
