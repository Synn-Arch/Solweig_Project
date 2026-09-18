# SPDX-License-Identifier: GPL-3.0-only
"""T12 — CUDA profile separation + runtime purity (TASKS T12, DESIGN 3.1).

Two profiles may coexist in one process but NEVER mix:

* ``canonical_cuda_v1`` — the strict build, the ONLY profile allowed to
  claim canonical parity (assert_canonical_strict);
* ``legacy_cuda_v1`` — the aten-default-like characterization build
  (``--fmad=true``), UNCERTIFIED: loading it must refuse certification,
  and its buffers must be refused by the canonical runtime
  (RuntimeMismatch) so no legacy bit can ever contaminate a canonical
  cache or claim parity.

Purity: the native/cuda runtime is torch-free — statically (no torch
import/include anywhere under native/cuda) and dynamically (the host
module loads, builds and launches a kernel in a subprocess where the
torch import is poisoned).

RED witness 6 (packed-byte write race — the class, at this surface):
the T12 kernels only ever READ the packed bitplane cubes (const
__restrict__) and write disjoint per-lane outputs; the in-surface
witness is launch-interleaving determinism — two stages issued
back-to-back on one runtime without an intervening sync reproduce
their serial bits exactly, in either order. Any hidden shared
write/race buffer between kernels shows up as a bit flip here.
"""
from __future__ import annotations

import json
import subprocess
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
    require_canonical_runtime,
    require_runtime,
)

NATIVE_CUDA = ULTRA_DIR.parents[1] / "native" / "cuda"


@pytest.fixture(scope="module")
def host():
    return load_host_module()


@pytest.fixture(scope="module")
def rt():
    return require_canonical_runtime()


# ---------------------------------------------------------------------------
# profile separation
# ---------------------------------------------------------------------------


class TestProfileSeparation:
    def test_unknown_profile_id_refused(self, host):
        with pytest.raises(ValueError, match="unknown profile_id"):
            host.CudaRuntime("mut_fmad")

    def test_two_profiles_distinct_libraries_and_flags(self, host):
        rt_can = require_runtime(host.CANONICAL_CUDA_V1)
        rt_leg = require_runtime(host.LEGACY_CUDA_V1)
        assert rt_can is not rt_leg
        assert rt_can.lib_path != rt_leg.lib_path
        assert "--fmad=false" in rt_can.build_flags
        assert "--fmad=true" in rt_leg.build_flags
        assert "--fmad=false" not in rt_leg.build_flags

    def test_runtime_cache_is_per_profile(self, host):
        # skip discipline: building needs nvcc (no CUDA locally -> skip,
        # hard fail under SW_REQUIRE_CUDA=1). require_runtime delegates to
        # host.get_runtime's cache, so the warm-up is the same namespace.
        require_runtime(host.CANONICAL_CUDA_V1)
        a = host.get_runtime(host.CANONICAL_CUDA_V1)
        b = host.get_runtime(host.CANONICAL_CUDA_V1)
        c = host.get_runtime(host.LEGACY_CUDA_V1)
        assert a is b and a is not c

    def test_legacy_refuses_canonical_certification(self, host):
        rt_leg = require_runtime(host.LEGACY_CUDA_V1)
        with pytest.raises(host.RuntimeMismatch, match="not the canonical"):
            rt_leg.assert_canonical_strict()

    def test_canonical_passes_strict_assertion(self, host):
        rt = require_canonical_runtime()
        rt.assert_canonical_strict()  # must not raise

    def test_buffers_refuse_cross_profile_use(self, host):
        rt_can = require_runtime(host.CANONICAL_CUDA_V1)
        rt_leg = require_runtime(host.LEGACY_CUDA_V1)
        dev = rt_can.alloc((4,), np.float32)
        with pytest.raises(host.RuntimeMismatch, match="never share"):
            dev._owner(rt_leg)  # noqa: SLF001 — the contract under test
        dev_leg = rt_leg.alloc((4,), np.float32)
        with pytest.raises(host.RuntimeMismatch):
            dev_leg._owner(rt_can)


# ---------------------------------------------------------------------------
# purity (torch-free runtime)
# ---------------------------------------------------------------------------


class TestRuntimePurity:
    def test_static_no_torch_in_native_cuda_tree(self):
        """The RUNTIME surface (host.py, build.py, every .cu/.cuh) must be
        torch-free. tools/gen_*.py are reference generators — torch is
        allowed there by contract, but only at function scope (never
        imported when the module itself is imported for its spec tables).
        """
        offenders = []
        for p in (NATIVE_CUDA / "host.py", NATIVE_CUDA / "build.py"):
            if "import torch" in p.read_text():
                offenders.append(str(p))
        for p in list((NATIVE_CUDA / "src").glob("*.cu")) + list(
                (NATIVE_CUDA / "include").glob("*.cuh")):
            text = p.read_text()
            if "#include <torch" in text or "ATen" in text or \
                    "c10::" in text:
                offenders.append(str(p))
        for p in (NATIVE_CUDA / "tools").glob("gen_*.py"):
            text = p.read_text()
            if "import torch" in text.split("def ")[0]:
                offenders.append(f"{p} (module-scope torch import)")
        assert not offenders, f"torch leakage in native/cuda: {offenders}"

    def test_runtime_loads_and_launches_with_torch_poisoned(self, host):
        """Subprocess poisons the torch import (find spec -> raise), then
        loads host.py by path, opens the canonical runtime and launches a
        kernel. Torch anywhere on the runtime path = ImportError = fail."""
        # skip discipline: the probe builds/launches on a real device —
        # without CUDA it would only re-report "nvcc not found" as a
        # failure, not a purity verdict (hard fail under SW_REQUIRE_CUDA=1).
        require_runtime(host.CANONICAL_CUDA_V1)
        code = """
import sys
import importlib.abc


class _Poison(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is poisoned for this purity probe")
        return None


sys.meta_path.insert(0, _Poison())
sys.path.insert(0, sys.argv[1])

import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "sw_cuda_host_purity", str(Path(sys.argv[1]) / "host.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rt = mod.get_runtime(mod.CANONICAL_CUDA_V1)
rt.assert_canonical_strict()
y = mod.run_arith(rt, np.array([1.0], dtype=np.float32),
                  np.array([2.0], dtype=np.float32))
add, sub, mul, dv = y
assert int(np.asarray(add).view(np.uint32)[0]) == 0x40400000, add  # 3.0
assert int(np.asarray(sub).view(np.uint32)[0]) == 0xBF800000, sub  # -1
assert int(np.asarray(mul).view(np.uint32)[0]) == 0x40000000, mul  # 2.0
assert int(np.asarray(dv).view(np.uint32)[0]) == 0x3F000000, dv    # 0.5
print("PURITY_OK")
"""
        code = "import numpy as np\n" + code
        proc = subprocess.run(
            [sys.executable, "-c", code, str(NATIVE_CUDA)],
            capture_output=True, text=True, timeout=600,
        )
        assert "PURITY_OK" in proc.stdout, (
            f"torch-poisoned runtime probe failed:\n{proc.stdout}\n"
            f"{proc.stderr}"
        )
        assert proc.returncode == 0


# ---------------------------------------------------------------------------
# RED witness 6 stand-in: launch-interleaving determinism (no hidden
# shared/racy write path between kernels; packed cubes are read-only)
# ---------------------------------------------------------------------------


class TestWitness6LaunchInterleaving:
    def test_fold_and_utci_back_to_back_match_serial(self, rt, host):
        pins_path = NATIVE_CUDA / "data" / "fold_pins.json"
        if not pins_path.is_file():
            pytest.skip(f"fold pins not generated ({pins_path})")
        pins = json.loads(pins_path.read_text())

        sys.path.insert(0, str(ULTRA_DIR.parents[1]))
        rng = np.random.default_rng(0xACC)
        ta, rh, tmrt, va = (
            rng.uniform(-5.0, 38.0, (64, 64)),
            rng.uniform(20.0, 95.0, (64, 64)),
            rng.uniform(10.0, 75.0, (64, 64)),
            rng.uniform(0.15, 12.0, (64, 64)),
        )
        utci_in = tuple(np.ascontiguousarray(a, dtype=np.float32)
                        for a in (ta, rh, tmrt, va))
        utci_serial = host.run_utci_dense(rt, *utci_in)

        t = pins["tables"]
        tables = {
            "w_iso": f32_from_hex(t["w_iso"]).reshape(8, 12),
            "w_aniso": f32_from_hex(t["w_aniso"]).reshape(8, 12),
            "ring": np.array(t["ring"], dtype=np.int32),
            "na": np.array(t["na"], dtype=np.int32),
            "dir_e": np.array(t["dir_e"], dtype=np.int8),
            "dir_s": np.array(t["dir_s"], dtype=np.int8),
            "dir_w": np.array(t["dir_w"], dtype=np.int8),
            "dir_n": np.array(t["dir_n"], dtype=np.int8),
            "last_const": f32_from_hex([t["last_bits"]])[0],
            "one_minus_trans": f32_from_hex([t["one_minus_trans_bits"]])[0],
        }
        entry = pins["cases"][0]
        shape = (entry["rows"], entry["cols"])

        def run_fold_once():
            return host.run_fold(
                rt,
                _u8(entry["veg_bytes"]).reshape((*shape, 20)),
                _u8(entry["vbsh_bytes"]).reshape((*shape, 20)),
                f32_from_hex(entry["vegdem2"]).reshape(shape),
                f32_from_hex(entry["svf_building"]).reshape(shape),
                tables)

        fold_serial = run_fold_once()

        # interleave WITHOUT intermediate syncs: fold -> utci -> fold -> utci
        interleaved = []
        for _ in range(2):
            interleaved.append(run_fold_once())
            interleaved.append(host.run_utci_dense(rt, *utci_in))
        for i, got in enumerate(interleaved):
            want = fold_serial if i % 2 == 0 else utci_serial
            label = f"interleave[{i}]:fold" if i % 2 == 0 else \
                f"interleave[{i}]:utci"
            msg, _ = compare_bits(want, got, label)
            assert msg == "PASS", msg

    def test_kernel_results_are_reproducible_per_runtime(self, rt, host):
        x = np.linspace(0.1, 30.0, 4096, dtype=np.float32)
        a = host.run_exp(rt, x)
        b = host.run_exp(rt, x)
        msg, _ = compare_bits(a, b, "expf-rerun")
        assert msg == "PASS", msg


def _u8(hex_list):
    return np.array([int(h, 16) for h in hex_list], dtype=np.uint8)
