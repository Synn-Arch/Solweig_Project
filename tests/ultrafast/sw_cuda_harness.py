# SPDX-License-Identifier: GPL-3.0-only
"""Shared plumbing for the CUDA gate tests (TASKS T12).

Loads the torch-free ctypes host (native/cuda/host.py) by file path
(no package install), pins the canonical profile's strict build, loads
the committed primitive pins, and provides raw-bit comparison with
first-mismatch diagnostics.

Skip discipline (binding contract): no GPU / no nvcc / no library means
``pytest.skip`` on a local machine, but a HARD failure when
``SW_REQUIRE_CUDA=1`` — the GPU host never silently skips a gate.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"
if str(NATIVE_CUDA) not in sys.path:
    sys.path.insert(0, str(NATIVE_CUDA))

_host_module = None


def load_host_module():
    """import native/cuda/host.py under a unique module name."""
    global _host_module
    if _host_module is None:
        spec = importlib.util.spec_from_file_location(
            "sw_cuda_host", NATIVE_CUDA / "host.py")
        assert spec is not None and spec.loader is not None
        _host_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_host_module)
    return _host_module


def require_canonical_runtime():
    """Canonical-strict runtime or pytest.skip/fail per SW_REQUIRE_CUDA."""
    host = load_host_module()
    try:
        rt = host.require_runtime(host.CANONICAL_CUDA_V1)
    except host.SwCudaUnavailable as exc:
        if host.cuda_required():
            pytest.fail(f"SW_REQUIRE_CUDA=1 but canonical runtime failed: {exc}")
        pytest.skip(f"no CUDA: {exc}")
    if rt is None:
        pytest.skip("no CUDA runtime available")
    rt.assert_canonical_strict()
    return rt


def require_runtime(profile_id: str):
    host = load_host_module()
    try:
        rt = host.require_runtime(profile_id)
    except host.SwCudaUnavailable as exc:
        if host.cuda_required():
            pytest.fail(f"SW_REQUIRE_CUDA=1 but {profile_id} failed: {exc}")
        pytest.skip(f"no CUDA: {exc}")
    if rt is None:
        pytest.skip(f"no CUDA runtime for {profile_id}")
    return rt


def skip_local_only(reason: str):
    host = load_host_module()
    if host.cuda_required():
        pytest.fail(f"SW_REQUIRE_CUDA=1: {reason}")
    pytest.skip(reason)


_pins = None


def load_pins() -> dict:
    global _pins
    if _pins is None:
        path = NATIVE_CUDA / "data" / "primitive_pins.json"
        if not path.is_file():
            skip_local_only(f"primitive pins not generated ({path})")
        _pins = json.loads(path.read_text())["sets"]
    return _pins


def f32_from_hex(hex_list) -> np.ndarray:
    bits = np.array([int(h, 16) for h in hex_list], dtype=np.uint32)
    return bits.view(np.float32)


def compare_bits(expected_f32: np.ndarray, got_f32: np.ndarray, label: str):
    """Raw uint32 comparison — signed zero and NaN payload distinct."""
    want = np.ascontiguousarray(expected_f32, dtype=np.float32).reshape(-1).view(np.uint32)
    got = np.ascontiguousarray(got_f32, dtype=np.float32).reshape(-1).view(np.uint32)
    if want.shape != got.shape:
        return (f"{label}: shape mismatch want {want.shape} got {got.shape}",
                0)
    bad = want != got
    n = int(bad.sum())
    if n == 0:
        return ("PASS", 0)
    idx = np.nonzero(bad)[0][:3]
    details = "; ".join(
        f"[{i}] want 0x{int(want[i]):08x}({np.float32(want[i])!r}) "
        f"got 0x{int(got[i]):08x}({np.float32(got[i])!r})"
        for i in idx
    )
    return (f"{label}: {n}/{want.size} mismatched bits, first: {details}", n)
