# SPDX-License-Identifier: GPL-3.0-only
"""T12 GATE 1 — CUDA primitive layer bit-equality (TASKS T12, DESIGN 6.2).

The strict arithmetic layer (native/cuda/include/sw_strict_math.cuh) must
reproduce the CPU canonical primitive BITS (solweig_core/numba_cpu/
math_compat.py — the frozen T09 reference) on the committed pin lattices:

* SLEEF xexpf / xlogf_u1 / xpowf device ports, raw uint32 equality,
  signed zero and NaN payload included (no allclose, no tolerance);
* the opmath pow (scalar-tail lanes) via the exact integer-mantissa form;
* plain IEEE add/sub/mul/div and the contraction witness (a*b + c as two
  ops) — clean under the strict build, flipped under --fmad=true;
* torch elementwise maximum and the first-NaN-operand wrappers;
* the torch chunk-layout lane predicate as a pure (i, n, T, grain) table.

RED witnesses covered here (as pristine-build PASS + mutation FAIL runs
under tests/ultrafast/run_cuda_mutations.py):
  #1 FMA contraction, #2 FTZ, #3 approximate intrinsics, #4 CPU/GPU
  library discrepancy, #7 schedule dependence (is_libm_lane table).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from sw_cuda_harness import (  # noqa: E402
    compare_bits,
    f32_from_hex,
    load_host_module,
    load_pins,
    require_canonical_runtime,
)


@pytest.fixture(scope="module")
def rt():
    return require_canonical_runtime()


@pytest.fixture(scope="module")
def pins():
    return load_pins()


@pytest.fixture(scope="module")
def host():
    return load_host_module()


# ---------------------------------------------------------------------------
# build identity
# ---------------------------------------------------------------------------


class TestBuildIdentity:
    def test_strict_flags_reported(self, rt):
        for flag in ("--fmad=false", "--ftz=false", "--prec-div=true",
                     "--prec-sqrt=true"):
            assert flag in rt.build_flags, rt.build_flags
        assert "fast_math" not in rt.build_flags

    def test_device_visible(self, rt):
        name = rt.device_name()
        assert name and "sm_" in name, name

    def test_purity_torch_free(self, host):
        """native/cuda runtime is torch-free (import boundary proof)."""
        source = (Path(host.__file__).parent / "host.py").read_text()
        assert "import torch" not in source
        assert "from torch" not in source
        build_src = (Path(host.__file__).parent / "build.py").read_text()
        assert "import torch" not in build_src


# ---------------------------------------------------------------------------
# SLEEF ports
# ---------------------------------------------------------------------------


class TestSleefPorts:
    def test_expf_pins(self, rt, host, pins):
        s = pins["exp"]
        x = f32_from_hex(s["x"])
        got = host.run_exp(rt, x)
        msg, n = compare_bits(f32_from_hex(s["y"]), got, "exp")
        assert msg == "PASS", msg

    def test_logf_pins(self, rt, host, pins):
        s = pins["log"]
        got = host.run_log(rt, f32_from_hex(s["x"]))
        msg, n = compare_bits(f32_from_hex(s["y"]), got, "log")
        assert msg == "PASS", msg

    def test_powf_pins(self, rt, host, pins):
        s = pins["powf"]
        got = host.run_powf(rt, f32_from_hex(s["x"]), f32_from_hex(s["e"]))
        msg, n = compare_bits(f32_from_hex(s["y"]), got, "powf")
        assert msg == "PASS", msg

    @pytest.mark.parametrize("e", [4, 5, 6])
    def test_opmath_pow_pins(self, rt, host, pins, e):
        s = pins[f"opmath_pow{e}"]
        got = host.run_opmath_pow(rt, f32_from_hex(s["x"]), e)
        msg, n = compare_bits(f32_from_hex(s["y"]), got, f"opmath_pow{e}")
        assert msg == "PASS", msg


# ---------------------------------------------------------------------------
# plain IEEE ops + contraction witness
# ---------------------------------------------------------------------------


class TestStrictArithmetic:
    def test_arith_pins(self, rt, host, pins):
        s = pins["arith"]
        a = f32_from_hex(s["a"])
        b = f32_from_hex(s["b"])
        add, sub, mul, dv = host.run_arith(rt, a, b)
        for got, key, label in ((add, "add", "a+b"), (sub, "sub", "a-b"),
                                (mul, "mul", "a*b"), (dv, "div", "a/b")):
            msg, _ = compare_bits(f32_from_hex(s[key]), got, label)
            assert msg == "PASS", msg

    def test_contraction_witness_clean(self, rt, host, pins):
        """a*b + c as two separate ops stays UN-fused under the strict
        build (RED witness 1: --fmad=true flips these bits)."""
        s = pins["arith"]
        got = host.run_contract(rt, f32_from_hex(s["a"]), f32_from_hex(s["b"]),
                                f32_from_hex(s["c"]))
        msg, _ = compare_bits(f32_from_hex(s["contract"]), got, "a*b+c")
        assert msg == "PASS", msg

    def test_ftz_witness_subnormals_survive(self, rt, host, pins):
        """Subnormal operands/results keep their bits under --ftz=false
        (RED witness 2: --ftz=true flushes them)."""
        a = np.array([1e-40, 1e-42, 8.5e-44], dtype=np.float32)
        b = np.array([1e-30, 1.0, 3.0], dtype=np.float32)
        add, _sub, mul, _dv = host.run_arith(rt, a, b)
        want_add = (a + b).astype(np.float32)
        want_mul = (a * b).astype(np.float32)
        assert np.array_equal(add.view(np.uint32), want_add.view(np.uint32))
        assert np.array_equal(mul.view(np.uint32), want_mul.view(np.uint32))
        # at least one genuinely subnormal result bit pattern present
        sub_bits = want_mul.view(np.uint32)
        assert np.any((sub_bits & np.float32(np.inf).view(np.uint32)) == 0), (
            "fixture lost all subnormal results — witness too weak"
        )


# ---------------------------------------------------------------------------
# torch semantics wrappers
# ---------------------------------------------------------------------------


class TestTorchSemantics:
    def test_maximum_pins(self, rt, host, pins):
        s = pins["maximum"]
        got = host.run_maximum(rt, f32_from_hex(s["a"]), f32_from_hex(s["b"]))
        msg, _ = compare_bits(f32_from_hex(s["y"]), got, "maximum")
        assert msg == "PASS", msg

    def test_nadd_nmul_pins(self, rt, host, pins):
        for key, fn in (("nadd", "run_narith"), ("nmul", "run_narith")):
            s = pins[key]
            add, _sub, mul, _dv = host.run_narith(
                rt, f32_from_hex(s["a"]), f32_from_hex(s["b"]))
            got = add if key == "nadd" else mul
            msg, _ = compare_bits(f32_from_hex(s["y"]), got, key)
            assert msg == "PASS", msg

    def test_is_libm_lane_table(self, rt, host, pins):
        """The chunk-layout predicate is a pure (i, n, T, grain) table —
        identical on device for every query (RED witness 7)."""
        s = pins["is_libm_lane"]
        i_arr = np.array(s["i"], dtype=np.int64)
        n_arr = np.array(s["n"], dtype=np.int64)
        T_arr = np.array(s["T"], dtype=np.int32)
        g_arr = np.array(s["grain"], dtype=np.int32)
        got = host.run_is_libm_lane(rt, i_arr, n_arr, T_arr, g_arr)
        want = np.array(s["y"], dtype=np.int32)
        bad = int((got != want).sum())
        assert bad == 0, (
            f"is_libm_lane: {bad}/{want.size} mismatches, first "
            f"{np.nonzero(got != want)[0][:3]}"
        )
