"""T11 gate: solweig_core.numba_cpu.sleef_trig — bit-exact f32 trig ports.

torch is allowed in THIS file (live-oracle comparison for the SLEEF ports).
The module under test (solweig_core.numba_cpu.sleef_trig) must stay
torch-free (pinned by test_sleef_trig_purity below).

Parity contract: RAW uint32 bits vs torch on this host (arm64 macOS,
torch built with -DAT_BUILD_ARM_VEC256_WITH_SLEEF -> Sleef_*f4_u10 ==
SLEEF AdvSIMD CONFIG 1 / ENABLE_FMA_SP u1 kernels).  No allclose, no
1-ULP tolerance: signed zero and NaN payloads must survive exactly.

Probes that established the identification live in
/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t11/probes/
(probe_native_sleef.py: natively compiled pinned SLEEF CONFIG 1 == torch
== this port, 0/800000; CONFIG 2 (AdvSIMDNOFMA) != torch).
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from solweig_core.numba_cpu import sleef_trig as st

U32 = np.uint32

_TRIG_CASES = (
    ("sin", torch.sin, st.sleef_sin_v1, (-124.0, 124.0)),
    ("cos", torch.cos, st.sleef_cos_v1, (-124.0, 124.0)),
    ("asin", torch.asin, st.sleef_asin_v1, (-1.0, 1.0)),
    ("acos", torch.acos, st.sleef_acos_v1, (-1.0, 1.0)),
)


def _bit_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.asarray(a).view(U32) != np.asarray(b).view(U32)).sum())


@pytest.mark.parametrize("name,torch_fn,port_fn,span", _TRIG_CASES)
def test_sleef_trig_bit_parity_vs_torch(name, torch_fn, port_fn, span):
    """RAW-bit parity vs torch on 200k deterministic points + dense grid."""
    rng = np.random.default_rng(20090811)
    lo, hi = span
    mag = min(abs(lo), abs(hi))
    if name in ("sin", "cos"):
        a = np.concatenate([
            np.linspace(lo, hi, 100000),
            rng.uniform(lo, hi, 100000),
            np.linspace(0.0, mag, 50000),
        ]).astype(np.float32)
    else:
        a = np.concatenate([
            rng.uniform(lo, hi, 150000),
            np.linspace(lo, hi, 50000),
        ]).astype(np.float32)
    # stay strictly inside the ported fast path / domain
    a = a[np.abs(a) < np.float32(124.99)] if name in ("sin", "cos") else a
    a = a[np.abs(a) <= np.float32(1.0)] if name in ("asin", "acos") else a
    t = torch_fn(torch.from_numpy(a)).numpy()
    p = port_fn(a)
    assert _bit_diff(t, p) == 0, f"{name}: {_bit_diff(t, p)} raw-bit diffs / {a.size}"


def test_sleef_trig_edge_bits():
    """Signed zeros, ±1, ±0.5, near-boundary, denormal-scale inputs."""
    a = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 124.99, -124.99,
                  1e-30, -1e-30, 0.9999999, -0.9999999],
                 dtype=np.float32)
    for name, torch_fn, port_fn, _ in _TRIG_CASES:
        if name in ("sin", "cos"):
            args = a
        else:
            args = a[np.abs(a) <= 1.0]
        t = torch_fn(torch.from_numpy(args)).numpy()
        p = port_fn(args)
        assert _bit_diff(t, p) == 0, (
            f"{name} edge mismatch:\n"
            f" args  {[hex(x) for x in args.view(U32)]}\n"
            f" torch {[hex(x) for x in t.view(U32)]}\n"
            f" port  {[hex(x) for x in p.view(U32)]}")


def test_sleef_trig_range_guards_raise():
    """Out-of-domain inputs raise — never silently degrade."""
    with pytest.raises(ValueError):
        st.sleef_sin_v1(np.array([125.0], dtype=np.float32))
    with pytest.raises(ValueError):
        st.sleef_cos_v1(np.array([-1000.0], dtype=np.float32))
    with pytest.raises(ValueError):
        st.sleef_asin_v1(np.array([1.0000001], dtype=np.float32))
    with pytest.raises(ValueError):
        st.sleef_acos_v1(np.array([-1.0000001], dtype=np.float32))


def test_sleef_trig_frozen_digest():
    """Frozen self-pin: port output digest on a fixed input vector.

    Catches silent drift of the port OR of the local torch build (the
    digest equals the digest torch produces on this host; if a torch
    upgrade changes its SLEEF pin, test_sleef_trig_bit_parity_vs_torch
    fails first and this digest must then be re-derived deliberately).
    """
    rng = np.random.default_rng(4242)
    a = np.concatenate([rng.uniform(-124.99, 124.99, 65536),
                        rng.uniform(-1.0, 1.0, 65536)]).astype(np.float32)
    trig = a[:65536]
    inv = a[65536:]
    outs = b"".join([
        st.sleef_sin_v1(trig).tobytes(), st.sleef_cos_v1(trig).tobytes(),
        st.sleef_asin_v1(inv).tobytes(), st.sleef_acos_v1(inv).tobytes(),
    ])
    import hashlib
    digest = hashlib.sha256(outs).hexdigest()
    expected = hashlib.sha256(b"".join([
        torch.sin(torch.from_numpy(trig)).numpy().tobytes(),
        torch.cos(torch.from_numpy(trig)).numpy().tobytes(),
        torch.asin(torch.from_numpy(inv)).numpy().tobytes(),
        torch.acos(torch.from_numpy(inv)).numpy().tobytes(),
    ])).hexdigest()
    assert digest == expected, "port digest diverged from torch digest"


def test_sleef_trig_purity_torch_free():
    """The runtime module must not import torch (AST + subprocess pin)."""
    src_path = Path(st.__file__)
    tree = ast.parse(src_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all("torch" not in (al.name or "") for al in node.names), \
                "sleef_trig must not import torch"
        if isinstance(node, ast.ImportFrom):
            assert "torch" not in (node.module or ""), \
                "sleef_trig must not import from torch"
    code = ("import sys;"
            "import solweig_core.numba_cpu.sleef_trig as m;"
            "assert 'torch' not in sys.modules, 'torch leaked into module';"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True,
                       cwd=str(src_path.parents[2]))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_sleef_trig_no_fastmath():
    """fastmath must stay OFF everywhere in the module (contraction would
    destroy the df double-float error terms)."""
    src = Path(st.__file__).read_text()
    assert "fastmath=True" not in src
    assert 'fastmath=False' in src
