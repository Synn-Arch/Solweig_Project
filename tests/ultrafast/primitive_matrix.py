"""Primitive characterization matrix (DESIGN.ko.md 10.2, T01).

Runs the arithmetic primitives the SOLWEIG exact paths depend on across
execution contexts and records raw-bit agreement between backends:

* operators: add/sub/mul/div, comparisons (eq/lt), maximum/minimum, round,
  pow (integer and float exponents), log/exp, sin/cos/tan, sqrt;
* contexts: scalar tensor (0-d), broadcast scalar, dense tensor, compacted
  (masked-gather) tensor;
* shapes: vector/tail boundaries 0,1,2,7,8,9,15,16,17,31,32,33,63,64,65 and
  the real E1 extent (248, 220);
* values: domain boundaries, nextafter neighbors, signed zeros, subnormals,
  the -999 validity sentinel, NaN/inf where the source accepts them.

Backends available now: torch-CPU (same venv as the pinned oracle) and
numpy. Every comparison is raw-bit (uint32 view / bool mismatch count);
ULP distance is reported diagnostically only. The output feeds T09's UTCI
compatibility planning: which (operator, context, shape) tuples cannot be
swapped between torch and numpy, and where dense vs compacted dispatch
changes bits within a single backend.

Runnable as a script to emit the frozen JSON artifacts:
    python tests/ultrafast/primitive_matrix.py --full \
        --matrix-out benchmarks/ultrafast/baseline/primitive_matrix.json \
        --profile-out benchmarks/ultrafast/baseline/numeric_profile.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
U32 = np.dtype("<u4")
E1_EXTENT_SHAPE = (248, 220)  # real E1 write-window extent (site_500, 2 m)
SHAPES_FAST: tuple[tuple[int, ...], ...] = (
    (0,), (1,), (2,), (7,), (8,), (9,), (15,), (16,), (17,), (33,), E1_EXTENT_SHAPE,
)
SHAPES_FULL: tuple[tuple[int, ...], ...] = (
    (0,), (1,), (2,), (7,), (8,), (9,), (15,), (16,), (17,),
    (31,), (32,), (33,), (63,), (64,), (65,), E1_EXTENT_SHAPE,
)
SENTINEL = np.float32(-999.0)


# ---------------------------------------------------------------------------
# Value pool
# ---------------------------------------------------------------------------
def special_values() -> np.ndarray:
    f32 = np.float32
    fi = np.finfo(np.float32)
    one = f32(1.0)
    return np.array([
        f32(0.0), f32(-0.0),                                   # signed zeros
        one, np.nextafter(one, f32(2.0)), np.nextafter(one, f32(0.0)),  # 1-ULP nbrs
        -one, np.nextafter(-one, f32(-2.0)),
        f32(0.5), f32(-0.5), f32(2.0), f32(3.0), f32(-2.5),
        f32(np.pi), f32(-np.pi / 2),                           # trig boundaries
        fi.max, fi.min,                                        # +-FLT_MAX
        fi.tiny, -fi.tiny,                                     # min normal
        fi.smallest_subnormal, -fi.smallest_subnormal,         # subnormals
        f32(1e-20), f32(-1e-20),
        f32(55.7), f32(-18.4), f32(293.15), f32(311.9),        # UTCI/Tmrt range
        SENTINEL,                                              # validity sentinel
        f32(np.inf), f32(-np.inf),
        f32("nan"),
        np.array(0x7FC00000, dtype=U32).view("<f4"),           # quiet NaN
        np.array(0xFFC00000, dtype=U32).view("<f4"),           # -quiet NaN
        np.array(0x7F800001, dtype=U32).view("<f4"),           # signaling NaN
    ], dtype="<f4")


def build_array(shape: tuple[int, ...], seed: int = 20090811) -> np.ndarray:
    """Deterministic float32 array tiling the special pool, then LCG noise."""
    n = int(np.prod(shape)) if shape else 1
    pool = special_values()
    if n == 0:
        return np.zeros(shape, dtype="<f4")
    tiles = (n + pool.size - 1) // pool.size
    arr = np.tile(pool, tiles)[:n].copy()
    # perturb the non-special tail so lanes are not all identical
    rng = np.random.default_rng(seed)
    tail_mask = np.zeros(n, dtype=bool)
    tail_mask[pool.size:] = True
    with np.errstate(all="ignore"):
        arr[tail_mask] += rng.standard_normal(int(tail_mask.sum())).astype("<f4")
    return arr.reshape(shape)


VALID_MASK_THRESHOLD = np.float32(-900.0)  # keeps -999 sentinel invalid


def compacted_views(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(valid_mask, compacted_values) following the <= -999 validity contract."""
    mask = arr > VALID_MASK_THRESHOLD
    return mask, arr[mask]


# ---------------------------------------------------------------------------
# Operators (torch implementations imported lazily)
# ---------------------------------------------------------------------------
def _torch():
    import torch

    return torch


def _op_table() -> dict[str, dict[str, Any]]:
    """name -> {kind, torch, numpy, broadcast_scalar}."""
    t = _torch()
    table: dict[str, dict[str, Any]] = {
        "add":  {"kind": "binary", "torch": lambda a, b: a + b,      "numpy": lambda a, b: a + b,      "bcast": 2.5},
        "sub":  {"kind": "binary", "torch": lambda a, b: a - b,      "numpy": lambda a, b: a - b,      "bcast": 1.25},
        "mul":  {"kind": "binary", "torch": lambda a, b: a * b,      "numpy": lambda a, b: a * b,      "bcast": 0.5},
        "div":  {"kind": "binary", "torch": lambda a, b: a / b,      "numpy": lambda a, b: a / b,      "bcast": 3.0},
        "pow_int":   {"kind": "binary", "torch": lambda a, b: t.pow(a, 2), "numpy": lambda a, b: np.power(a, 2), "bcast": None},
        "pow_float": {"kind": "binary", "torch": lambda a, b: t.pow(a, b), "numpy": lambda a, b: np.power(a, b), "bcast": 2.5},
        "maximum": {"kind": "binary", "torch": t.maximum,  "numpy": np.maximum,  "bcast": 0.5,
                    "torch_scalar_wrap": True},
        "minimum": {"kind": "binary", "torch": t.minimum,  "numpy": np.minimum,  "bcast": -0.5,
                    "torch_scalar_wrap": True},
        "eq": {"kind": "binary", "torch": lambda a, b: a == b, "numpy": lambda a, b: a == b, "bcast": 1.0},
        "lt": {"kind": "binary", "torch": lambda a, b: a < b,  "numpy": lambda a, b: a < b,  "bcast": 0.0},
        "round": {"kind": "unary", "torch": t.round, "numpy": np.round, "bcast": None},
        "log":   {"kind": "unary", "torch": t.log,   "numpy": np.log,   "bcast": None},
        "exp":   {"kind": "unary", "torch": t.exp,   "numpy": np.exp,   "bcast": None},
        "sin":   {"kind": "unary", "torch": t.sin,   "numpy": np.sin,   "bcast": None},
        "cos":   {"kind": "unary", "torch": t.cos,   "numpy": np.cos,   "bcast": None},
        "tan":   {"kind": "unary", "torch": t.tan,   "numpy": np.tan,   "bcast": None},
        "sqrt":  {"kind": "unary", "torch": t.sqrt,  "numpy": np.sqrt,  "bcast": None},
    }
    return table


def _ulp_label(ref_bits: int, cand_bits: int) -> str:
    r = np.array([ref_bits], dtype=U32).view("<f4")[0]
    c = np.array([cand_bits], dtype=U32).view("<f4")[0]
    if np.isnan(r) and np.isnan(c):
        return "nan_payload_or_sign"
    if np.isnan(r) or np.isnan(c):
        return "nan_vs_non_nan"
    if r == 0.0 and c == 0.0:
        return "signed_zero"
    if np.isinf(r) or np.isinf(c):
        return "infinity"
    ri = np.array([ref_bits], dtype=U32).view("<i4").astype(np.int64)[0]
    ci = np.array([cand_bits], dtype=U32).view("<i4").astype(np.int64)[0]
    key_r = -ri - 1 if ri < 0 else ri + (1 << 31)
    key_c = -ci - 1 if ci < 0 else ci + (1 << 31)
    d = abs(int(key_r) - int(key_c))
    return f"{d}_ulp" if d <= 16 else "gt_16_ulp"


# ---------------------------------------------------------------------------
# Context runners
# ---------------------------------------------------------------------------
def _second_operand(a: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic tensor-tensor second operand from the same pool."""
    b = np.roll(build_array(a.shape, seed + 7), 3).copy()
    b[b == SENTINEL] = np.float32(1.5)  # keep the second operand valid-domain
    return b


def _bits(x) -> np.ndarray:
    """Result -> comparable bit plane (uint32 view for f32, uint8 for bool)."""
    a = _to_numpy(x)
    if a.dtype == np.bool_:
        return a.view(np.uint8)
    if a.dtype.str == "<f4":
        return a.view(U32)
    raise TypeError(f"unexpected result dtype {a.dtype.str}")


def _dtype_str(x) -> str:
    return str(_to_numpy(x).dtype)


def _to_numpy(x) -> np.ndarray:
    t = _torch()
    if isinstance(x, t.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x))


def run_context(op_name: str, op: dict, context: str, shape: tuple[int, ...],
                seed: int = 20090811) -> list[dict[str, Any]]:
    """Execute one (op, context, shape) cell; return comparison records.

    Records produced per cell:
      * torch_vs_numpy — same context, identical input bits, cross-backend;
      * {torch,numpy}_dense_vs_compacted — within-backend dispatch check:
        op applied to the masked-gather subset vs the same positions of the
        dense result (the compaction-order sensitivity DESIGN 10.2 warns of);
      * torch_vs_torch_repeat — same-backend determinism control.
    """
    records: list[dict[str, Any]] = []
    a = build_array(shape, seed)

    def annotate(rec: dict) -> dict:
        for m in rec.get("first_mismatches", []):
            m["classification"] = _ulp_label(
                int(m["reference_bits"], 16), int(m["candidate_bits"], 16))
        return rec

    def new_record(pair: str, ref, cand, *, context_tag: str | None = None,
                   elements: int | None = None, extra: dict | None = None) -> dict:
        ref_bits, cand_bits = _bits(ref).reshape(-1), _bits(cand).reshape(-1)
        if ref_bits.shape != cand_bits.shape:
            rec = {"mismatch_count": None, "reason": "shape_mismatch",
                   "first_mismatches": []}
        elif ref_bits.dtype != cand_bits.dtype:
            rec = {"mismatch_count": None, "reason": "dtype_mismatch",
                   "first_mismatches": []}
        else:
            diff = ref_bits != cand_bits
            n = int(np.count_nonzero(diff))
            ms = []
            for idx in (np.flatnonzero(diff)[:4] if n else []):
                ms.append({
                    "flat_index": int(idx),
                    "reference_bits": f"0x{int(ref_bits[idx]):08x}",
                    "candidate_bits": f"0x{int(cand_bits[idx]):08x}",
                })
            rec = {"mismatch_count": n, "first_mismatches": ms}
        out = {
            "op": op_name, "context": context_tag or context,
            "shape": list(shape), "pair": pair,
            "elements": a.size if elements is None else elements,
            "output_dtypes": [_dtype_str(ref), _dtype_str(cand)],
            **rec,
        }
        if extra:
            out.update(extra)
        return annotate(out)

    def apply(backend: str, x: np.ndarray, *, gathered: bool, second: np.ndarray | None,
              scalar: float | None):
        fn = op["torch"] if backend == "torch" else op["numpy"]
        if op["kind"] == "unary":
            arg = torch_from(x) if backend == "torch" else x
            return fn(arg)
        if op_name == "pow_int":
            arg = torch_from(x) if backend == "torch" else x
            return fn(arg, 2)
        if scalar is not None:
            arg = torch_from(x) if backend == "torch" else x
            if backend == "torch" and op.get("torch_scalar_wrap"):
                # torch.maximum/minimum reject python scalars; the broadcast
                # context is realized as a 0-d float32 tensor (recorded in
                # the op table so the JSON stays honest about the adaptation)
                return fn(arg, _torch().tensor(scalar, dtype=_torch().float32))
            return fn(arg, scalar)
        return fn(torch_from(x) if backend == "torch" else x,
                  torch_from(second) if backend == "torch" else second)

    if context == "scalar_tensor":
        pool = special_values()
        for v in pool:
            at, an = _torch().tensor(v, dtype=_torch().float32), np.float32(v)
            wt, wn = _torch().tensor(np.float32(0.5), dtype=_torch().float32), np.float32(0.5)
            try:
                with np.errstate(all="ignore"):
                    if op["kind"] == "unary":
                        rt, rn = op["torch"](at), op["numpy"](an)
                    else:
                        rt, rn = op["torch"](at, wt), op["numpy"](an, wn)
                records.append(new_record(
                    "torch_vs_numpy", rt, rn, context_tag="scalar_tensor",
                    elements=1,
                    extra={"scalar_value_bits": f"0x{int(np.float32(v).view(U32)):08x}"}))
            except (RuntimeError, ValueError) as exc:
                records.append({"op": op_name, "context": "scalar_tensor",
                                "shape": [], "pair": "torch_vs_numpy",
                                "error": str(exc)[:160]})
        return records

    b = None if op["kind"] == "unary" else (
        None if op_name == "pow_int" else _second_operand(a, seed))
    scalar = op["bcast"] if context == "broadcast_scalar" else None
    mask = (a > VALID_MASK_THRESHOLD).reshape(-1)

    with np.errstate(all="ignore"):
        dense_t = apply("torch", a, gathered=False, second=b, scalar=scalar)
        dense_n = apply("numpy", a, gathered=False, second=b, scalar=scalar)
        records.append(new_record("torch_vs_numpy", dense_t, dense_n))

        # within-backend dense vs compacted dispatch
        compact_a = a.reshape(-1)[mask]
        for backend, dense in (("torch", dense_t), ("numpy", dense_n)):
            second_c = None
            if b is not None:
                second_c = b.reshape(-1)[mask]
            compact_out = apply(backend, compact_a, gathered=True,
                                second=second_c, scalar=scalar)
            gathered_bits = _bits(compact_out)
            dense_on_valid = _bits(dense).reshape(-1)[mask]
            if (gathered_bits.shape != dense_on_valid.shape
                    or gathered_bits.dtype != dense_on_valid.dtype):
                records.append({
                    "op": op_name, "context": f"{context}::compacted",
                    "shape": list(shape), "pair": f"{backend}_dense_vs_compacted",
                    "elements": int(mask.sum()), "mismatch_count": None,
                    "reason": "shape_or_dtype_mismatch", "first_mismatches": []})
                continue
            diff = gathered_bits != dense_on_valid
            n = int(np.count_nonzero(diff))
            rec = {
                "op": op_name, "context": f"{context}::compacted",
                "shape": list(shape), "pair": f"{backend}_dense_vs_compacted",
                "elements": int(mask.sum()), "mismatch_count": n,
                "first_mismatches": [],
            }
            for idx in (np.flatnonzero(diff)[:4] if n else []):
                rb, cb = int(gathered_bits[idx]), int(dense_on_valid[idx])
                rec["first_mismatches"].append({
                    "compacted_index": int(idx),
                    "reference_bits": f"0x{rb:08x}",
                    "candidate_bits": f"0x{cb:08x}",
                    "classification": _ulp_label(rb, cb)})
            records.append(annotate(rec))

        # same-backend determinism control
        dense_t2 = apply("torch", a, gathered=False, second=b, scalar=scalar)
        records.append(new_record("torch_vs_torch_repeat", dense_t, dense_t2))

    return records


def torch_from(a: np.ndarray):
    t = _torch()
    return t.from_numpy(np.ascontiguousarray(a).copy())


# ---------------------------------------------------------------------------
# Matrix driver
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class MatrixResult:
    records: list[dict[str, Any]]
    environment: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        cmp_records = [r for r in self.records if "error" not in r]
        by_pair: dict[str, dict[str, Any]] = {}
        for r in cmp_records:
            key = (r["pair"], r["op"])
            slot = by_pair.setdefault(
                f'{r["pair"]}|{r["op"]}', {"cells": 0, "mismatched_cells": 0,
                                           "total_mismatch_elements": 0,
                                           "first_mismatch": None})
            slot["cells"] += 1
            mc = r.get("mismatch_count") or 0
            if mc:
                slot["mismatched_cells"] += 1
                slot["total_mismatch_elements"] += mc
                if slot["first_mismatch"] is None:
                    slot["first_mismatch"] = {
                        "context": r["context"], "shape": r["shape"],
                        "detail": r["first_mismatches"][:1]}
        return {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": self.environment,
            "cell_count": len(cmp_records),
            "error_cell_count": len(self.records) - len(cmp_records),
            "pairs": by_pair,
        }


def probe_environment() -> dict[str, Any]:
    """Frozen numeric-profile facts: versions, build ids, threads, ISA, promotion."""
    t = _torch()
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numpy_build": np.__config__.show(mode="dicts") if hasattr(np.__config__, "show") else None,
        "torch": t.__version__,
        "torch_build": str(t.__config__.parallel_info())[:500],
        "torch_file": str(Path(t.__file__).resolve()),
        "torch_threads": int(t.get_num_threads()),
        "torch_interop_threads": int(t.get_num_interop_threads()),
        "platform": f"{platform.system()}/{platform.machine()}",
        "platform_version": platform.version(),
        "cpu_brand": _cpu_brand(),
        "f32_dtype_str": np.dtype("<f4").str,
    }
    env["dtype_promotion_probes"] = promotion_probes()
    env["rounding_probes"] = rounding_probes()
    return env


def _cpu_brand() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5).stdout.strip()
        return platform.processor() or "unknown"
    except Exception:
        return "unknown"


def promotion_probes() -> dict[str, str]:
    t = _torch()
    a = np.ones(4, dtype="<f4")
    at = t.from_numpy(a.copy())
    probes: dict[str, str] = {}

    def np_dtype(expr) -> str:
        try:
            return str(expr().dtype)
        except Exception as exc:  # noqa: BLE001
            return f"error:{type(exc).__name__}"

    def t_dtype(expr) -> str:
        try:
            return str(expr().dtype)
        except Exception as exc:  # noqa: BLE001
            return f"error:{type(exc).__name__}"

    probes["np_f32_array_plus_python_float"] = np_dtype(lambda: a + 2.5)
    probes["np_f32_array_plus_python_int"] = np_dtype(lambda: a + 2)
    probes["np_f32_array_plus_np_float32"] = np_dtype(lambda: a + np.float32(2.5))
    probes["np_f32_array_plus_np_float64"] = np_dtype(lambda: a + np.float64(2.5))
    probes["torch_f32_tensor_plus_python_float"] = t_dtype(lambda: at + 2.5)
    probes["torch_f32_tensor_plus_python_int"] = t_dtype(lambda: at + 2)
    probes["torch_f32_tensor_plus_f64_scalar_tensor"] = t_dtype(
        lambda: at + t.tensor(2.5, dtype=t.float64))
    probes["torch_f32_tensor_plus_f64_1d_tensor"] = t_dtype(
        lambda: at + t.ones(4, dtype=t.float64))
    probes["torch_f32_tensor_plus_np_float32"] = t_dtype(lambda: at + np.float32(2.5))
    probes["torch_f32_tensor_plus_np_float64"] = t_dtype(lambda: at + np.float64(2.5))
    probes["np_f32_scalar_log"] = np_dtype(lambda: np.log(np.float32(2.0)))
    probes["torch_scalar_pow_int"] = t_dtype(lambda: t.pow(at, 2))
    probes["torch_pow_float_exponent"] = t_dtype(lambda: t.pow(at, at))
    probes["np_power_f32_float_exp"] = np_dtype(lambda: np.power(a, np.float32(2.5)))
    return probes


def rounding_probes() -> dict[str, list[float]]:
    t = _torch()
    vals = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5], dtype="<f4")
    return {
        "numpy_round": [float(x) for x in np.round(vals)],
        "torch_round": [float(x) for x in t.round(torch_from(vals))],
    }


def run_matrix(shapes: tuple[tuple[int, ...], ...] = SHAPES_FAST,
               ops: tuple[str, ...] | None = None,
               contexts: tuple[str, ...] = ("scalar_tensor", "broadcast_scalar",
                                            "dense")) -> MatrixResult:
    """Execute the (op, context, shape) matrix; compacted dispatch checks are
    emitted inside the array-context cells as ``{backend}_dense_vs_compacted``
    records."""
    table = _op_table()
    records: list[dict[str, Any]] = []
    for name in (ops or tuple(table)):
        op = table[name]
        if "scalar_tensor" in contexts:
            records.extend(run_context(name, op, "scalar_tensor", ()))
        for context in ("broadcast_scalar", "dense"):
            if context not in contexts:
                continue
            for shape in shapes:
                records.extend(run_context(name, op, context, shape))
    return MatrixResult(records=records, environment=probe_environment())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help=f"full shape sweep (default: fast subset)")
    ap.add_argument("--matrix-out", type=Path, required=True)
    ap.add_argument("--profile-out", type=Path, required=True)
    args = ap.parse_args()

    shapes = SHAPES_FULL if args.full else SHAPES_FAST
    result = run_matrix(shapes=shapes)
    summary = result.summary()
    payload = {**summary, "records": result.records}
    args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_out.write_text(json.dumps(payload, indent=2))
    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_out.write_text(json.dumps(
        {"schema_version": 1,
         "generated_at_utc": datetime.now(timezone.utc).isoformat(),
         "environment": result.environment}, indent=2))

    # console digest
    mm = [(k, v) for k, v in summary["pairs"].items() if v["mismatched_cells"]]
    print(f"cells={summary['cell_count']} error_cells={summary['error_cell_count']} "
          f"mismatching_pair_ops={len(mm)}")
    for key, v in sorted(mm):
        print(f"  MISMATCH {key}: cells={v['mismatched_cells']}/{v['cells']} "
              f"elements={v['total_mismatch_elements']} first={v['first_mismatch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
