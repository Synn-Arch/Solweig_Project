# SPDX-License-Identifier: GPL-3.0-only
"""T09 UTCI compatibility kernel tests (RED first).

Raw-bit parity vs the ORIGINAL solweig_gpu UTCI path (calculate_utci.py
utci_calculator + utci_process recompute_utci_steps): uint32 bit
comparison of every lane, signed zero + NaN payload included, no
allclose/equal_nan acceptance.

torch is allowed in THIS file (live-oracle parity; T08 test_radiation
precedent); the runtime modules under test (solweig_core.numba_cpu.utci,
.math_compat) must stay torch-free (pinned by test_purity_torch_free).

RED-first: neither runtime module existed at task start (first
import failed; development probes p6/p7 under
artifacts/t09/probes/ drove the kernel to 0 mismatch before this file
was written). The M1-M7 mutation cycles (artifacts/t09/mutations/)
re-derive the RED direction: each designated killer below FAILS under
its mutation and passes on pristine code.

Site/capture-dependent tests skip when the t08 capture bundle or the
pinned oracle tree is absent.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from solweig_core.numba_cpu import math_compat, utci  # noqa: E402

ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts")
CAP = ART / "t08/capture"
ORACLE_TREE = Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc")
SITE = ORACLE_TREE / "site-cache/site_500"
ORACLE_SRC = REPO_ROOT / "solweig_gpu/calculate_utci.py"

F32 = np.float32

#: RED-witness anchors (mutations/run_mutations.py M1-M7); each must stay
#: present EXACTLY once so the mutation cycles stay reproducible. M7
#: anchors math_compat.py; M1-M6 anchor utci.py.
MUTATION_ANCHORS_UTCI = {
    "M1_horner_rewrite": "acc = ta\nacc = nadd(acc, F32(0.607562052))",
    "M3_f64_promotion": "acc = nadd(acc, F32(0.607562052))",
    "M2_new_fma_contraction": "es = nadd(es, nmul(G3, tk))",
    "M4_pow_to_multiply": (
        "es = nadd(es, nmul(G6, _torch_pow_scalar(tk, F32(4.0), use_libm)))"
    ),
    "M5_nan_added_to_mask": (
        "or va[r, c] <= NEG999 or tmrt[r, c] <= NEG999):\n                n += 1"
    ),
    "M6_broadcast_context_flip": "ehpa = ndiv(nmul(es, rh), RH_DIV)",
}
MUTATION_ANCHORS_MATH = {
    "M7_odd_tail_mismatch": "return i >= e - (L % 8)",
}

#: frozen SLEEF-vs-opmath-f64-pow divergent input bits (p9 sweep; torch's
#: scalar-tail pow is pow(f64, f64) single-rounded to f32 — T09 P9). Each
#: bit also diverges from the multiply chain, so the same bits kill M4
#: (pow->multiply) and M7 (tail misclassification).
DIVERGENT_BITS = {
    "ta": {4: 0xC223D5B8, 5: 0xC1F05599, 6: 0x423DA6E2},
    "va": {4: 0x410D5DD7, 5: 0x409A03DA, 6: 0x40186D2B},
    "dtm": {4: 0x41890F4C, 5: 0x408A0CD4, 6: 0x4254E075},
    "tk": {4: 0x4388846C, 5: 0x43A3B5FE, 6: 0x43838DC9},
    "pa": {4: 0x44894A13, 5: 0x4307E2F9, 6: 0x438636C2},
}

#: M6 witness (p8_m6_split_witness.json hit 1): f32-plane context
#: 0x41c269d4 vs python-float context 0x41c269cd at these exact inputs
M6_WITNESS = {
    "ta": 0x41C80000, "rh": 0x42A10000, "va": 0x3F000000, "tmrt": 0x41700000,
}

#: baseline plane content digest (sha256 of the (24,500,500) f32 payload)
BASELINE_UTCI_SHA16 = "7d002c29f84e173c"


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def f32bits(u32) -> np.float32:
    a = np.empty(1, dtype=np.float32)
    a.view(np.uint32)[0] = np.uint32(u32)
    return a[0]


def _nudge(x, k) -> np.float32:
    """x moved k ulps (k may be negative), staying finite."""
    u = int(x.view(np.uint32)) + k
    assert 0 < u < 0xFFFFFFFF, "nudge left the finite range"
    a = np.empty(1, dtype=np.float32)
    a.view(np.uint32)[0] = np.uint32(u)
    return a[0]


def _exact_add_witness(addend, want) -> np.float32:
    """x with F32(x + addend) bit-equal to ``want`` (tk = ta + 273.15).
    The sum's rounding interval spans ~512 ulps of x, so the search walks
    x both directions over the full interval."""
    x = F32(np.float64(want) - np.float64(addend))
    for k in range(0, 1025):
        for cand in ((x,) if k == 0 else (_nudge(x, k), _nudge(x, -k))):
            if F32(cand + addend).view(np.uint32) == want.view(np.uint32):
                return cand
    raise AssertionError("no ta reproduces the tk witness bit")


def capture_present() -> bool:
    return (CAP / "manifest.json").is_file() and SITE.is_dir()


def _mismatch(a, b) -> int:
    return int(np.count_nonzero(bits(a) != bits(b)))


# ---------------------------------------------------------------------------
# module / purity / anchors
# ---------------------------------------------------------------------------

def test_module_imports() -> None:
    """RED-first: the T09 runtime modules exist and are importable."""
    assert utci is not None and math_compat is not None


def test_purity_torch_free() -> None:
    """Runtime modules must not import torch (AST + subprocess pin)."""
    for mod in (utci, math_compat):
        tree = ast.parse(Path(mod.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "torch" not in name, f"torch import: {name}"
    code = (
        "import sys; sys.path.insert(0, {root!r}); "
        "import numpy as np; "
        "from solweig_core.numba_cpu import utci; "
        "rng = np.random.default_rng(1); "
        "ta = rng.uniform(-20, 40, (16, 16)).astype(np.float32); "
        "rh = rng.uniform(10, 100, (16, 16)).astype(np.float32); "
        "va = rng.uniform(0.2, 9, (16, 16)).astype(np.float32); "
        "tm = (ta + 10).astype(np.float32); "
        "d = utci.utci_calculator_dense(ta, rh, tm, va); "
        "tv, rv, mv, xv = utci.compact_valid(ta, rh, tm, va); "
        "s = utci.utci_calculator_sparse(tv, rv, mv, xv); "
        "assert d.sum() == d.sum() and s.sum() == s.sum(); "
        "assert 'torch' not in sys.modules, 'torch leaked into runtime'; "
        "assert not any(m.startswith('solweig_gpu') for m in sys.modules)"
    ).format(root=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_mutation_anchors_unique() -> None:
    """Each M1-M7 anchor appears exactly once in its runtime module."""
    src_utci = Path(utci.__file__).read_text()
    for name, anchor in MUTATION_ANCHORS_UTCI.items():
        assert src_utci.count(anchor) == 1, \
            f"{name}: anchor count {src_utci.count(anchor)}"
    src_math = Path(math_compat.__file__).read_text()
    for name, anchor in MUTATION_ANCHORS_MATH.items():
        assert src_math.count(anchor) == 1, \
            f"{name}: anchor count {src_math.count(anchor)}"


# ---------------------------------------------------------------------------
# manifest / hybrid status
# ---------------------------------------------------------------------------

def test_manifest_and_hybrid_status() -> None:
    man = utci.get_expression_manifest()
    assert man["schema"] == "utci-expression-manifest/1"
    assert "ea3bd118eb15f9bc9320bc7becdc2a4a0f78a6f6" in man["oracle"]
    assert set(man["expressions"]) >= {
        "utci.invalid_mask", "utci.tk", "utci.es.log", "utci.es.pow_4",
        "utci.es.chain_add", "utci.ehPa", "utci.D_Tmrt", "utci.Pa",
        "utci.poly.sum", "utci.poly.pow_4_5_6", "utci.scatter",
        "utci.outer_mask",
    }
    for key, entry in man["expressions"].items():
        assert entry["resolution"] == "clean", f"{key} not clean"
    assert man["layout_model"]["params"]["torch_threads"] == 8
    assert man["layout_model"]["params"]["grain"] == 32768
    assert "left_for_T10" in man

    hybrid, table = math_compat.get_hybrid_status()
    assert hybrid["torch_free"] is True
    assert hybrid["fallback_primitives"] == {}
    for prim, res in table.items():
        assert res.startswith("clean"), f"{prim}: {res}"


# ---------------------------------------------------------------------------
# AST structural equality — POLY_EXPRESSION_SOURCE vs the oracle AST
# ---------------------------------------------------------------------------

NAME_MAP = {"D_Tmrt": "dtm", "Ta": "ta", "va": "va", "Pa": "pa"}


def _flatten_add(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_add(node.left) + [node.right]
    return [node]


def _normalize(node):
    """Comparable tuple tree: unwrap F32()/-F32()/nadd/nmul/pow routing."""
    if isinstance(node, ast.Call):
        fid = getattr(node.func, "id", "")
        if fid == "F32" and isinstance(node.args[0], ast.Constant):
            return ("const", node.args[0].value)
        if fid == "_torch_pow_scalar":
            base, expo, flag = (_normalize(a) for a in node.args)
            assert base[0] == "var" and expo[0] == "const" \
                and flag == ("var", "use_libm")
            return ("pow", base, expo)
        if fid in ("nadd", "nmul"):
            op = "+" if fid == "nadd" else "*"
            return (op, _normalize(node.args[0]), _normalize(node.args[1]))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _normalize(node.operand)
        assert inner[0] == "const"
        return ("const", -inner[1])
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            b, e = _normalize(node.left), node.right.value
            if e == 2:
                return ("*", b, b)
            if e == 3:
                return ("*", ("*", b, b), b)
            assert e in (4, 5, 6)
            return ("pow", b, ("const", float(e)))
        op = {ast.Mult: "*", ast.Add: "+"}[type(node.op)]
        return (op, _normalize(node.left), _normalize(node.right))
    if isinstance(node, ast.Constant):
        return ("const", node.value)
    if isinstance(node, ast.Name):
        return ("var", NAME_MAP.get(node.id, node.id))
    raise AssertionError(ast.dump(node))


def test_ast_structural_equality() -> None:
    """The committed POLY_EXPRESSION_SOURCE normalizes to the oracle
    utci_polynomial AST — original association preserved (M1 anchor)."""
    tree = ast.parse(ORACLE_SRC.read_text())
    poly = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "utci_polynomial")
    assign = next(n for n in poly.body if isinstance(n, ast.Assign))
    oracle_digest = json.dumps(_normalize(assign.value), sort_keys=True)

    lines = [ln for ln in utci.POLY_EXPRESSION_SOURCE.splitlines() if ln]
    assert len(lines) == 211, len(lines)
    acc = _normalize(ast.parse(lines[0], mode="exec").body[0].value)
    for ln in lines[1:]:
        rhs = ast.parse(ln, mode="exec").body[0].value
        norm = _normalize(rhs)
        assert norm[0] == "+" and norm[1] == ("var", "acc"), ln
        acc = ("+", acc, norm[2])
    assert json.dumps(acc, sort_keys=True) == oracle_digest


# ---------------------------------------------------------------------------
# synthetic parity — dense + sparse, all RED-witness inputs embedded
# ---------------------------------------------------------------------------

SCENES = {
    "temperate": (-20, 32, 20, 95, 0.5, 10, -15, 45),
    "extreme": (-45, 48, 2, 100, 0.15, 17, -40, 70),
    "humid": (18, 40, 60, 100, 0.3, 6, 0, 35),
    "arid": (25, 47, 2, 30, 1, 12, 5, 60),
    "windy": (-10, 30, 20, 90, 6, 18, -10, 30),
    "calm": (-5, 35, 30, 95, 0.15, 0.6, -5, 40),
}


def _build_scene(name, seed):
    ta0, ta1, rh0, rh1, va0, va1, d0, d1 = SCENES[name]
    rng = np.random.default_rng(seed)
    rows, cols = 137, 149
    ta = rng.uniform(ta0, ta1, (rows, cols)).astype(np.float32)
    rh = rng.uniform(rh0, rh1, (rows, cols)).astype(np.float32)
    va = rng.uniform(va0, va1, (rows, cols)).astype(np.float32)
    tmrt = (ta + rng.uniform(d0, d1, (rows, cols))).astype(np.float32)
    return _embed_witnesses(ta, rh, va, tmrt)


def _embed_witnesses(ta, rh, va, tmrt):
    """Specials + frozen divergent bits + M6 witness at body ordinals."""
    ta[0, 0] = np.nan                      # qNaN 0x7fc00000 — stays VALID
    rh[1, 1] = np.nan
    ta[2, 2] = F32(-999.0)                 # invalid lanes
    ta[3, 3] = F32(-0.0)                   # signed zero — valid
    rh[4, 4] = F32(-999.0)
    va[5, 5] = F32(-999.0)
    tmrt[6, 6] = F32(-999.0)
    rh[7, 7] = f32bits(0xFFFFFFFF)         # full-payload NaN
    # M6 witness at (0, 1): ordinal 1 (a body lane in every scene)
    ta[0, 1] = f32bits(M6_WITNESS["ta"])
    rh[0, 1] = f32bits(M6_WITNESS["rh"])
    va[0, 1] = f32bits(M6_WITNESS["va"])
    tmrt[0, 1] = f32bits(M6_WITNESS["tmrt"])
    r = 9
    for _, b in DIVERGENT_BITS["ta"].items():
        ta[r, 3] = f32bits(b); r += 1
    for _, b in DIVERGENT_BITS["va"].items():
        va[r, 5] = f32bits(b); r += 1
    for _, b in DIVERGENT_BITS["dtm"].items():
        # dtm = tmrt - ta must bit-equal the frozen bit: ta := 0.0 makes
        # the subtraction exact for any tmrt (a random ta's ulp can make
        # the wanted dtm bit unreachable)
        ta[r, 7] = F32(0.0)
        tmrt[r, 7] = f32bits(b); r += 1
    for _, b in DIVERGENT_BITS["tk"].items():
        ta[r, 9] = _exact_add_witness(F32(273.15), f32bits(b)); r += 1
    for _, b in DIVERGENT_BITS["pa"].items():
        va[r, 11] = f32bits(b); r += 1
    return (np.ascontiguousarray(ta), np.ascontiguousarray(rh),
            np.ascontiguousarray(va), np.ascontiguousarray(tmrt))


def _torch_oracle(ta, rh, tmrt, va):
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(8)
    from solweig_gpu.calculate_utci import utci_calculator
    return utci_calculator(torch.tensor(ta), torch.tensor(rh),
                           torch.tensor(tmrt), torch.tensor(va)).numpy()


def test_dense_parity_synthetic_scenes() -> None:
    """Dense kernel == torch oracle, raw bits, on all six scenes with
    NaN payloads, -999 invalids, signed zero, the frozen
    SLEEF-vs-opmath-f64-pow divergent bits and the M6 broadcast-context
    witness embedded (M1-M6 designated killer)."""
    for i, name in enumerate(SCENES):
        ta, rh, va, tmrt = _build_scene(name, 4242 + i)
        oracle = _torch_oracle(ta, rh, tmrt, va)
        mine = utci.utci_calculator_dense(ta, rh, tmrt, va, torch_threads=8)
        d = _mismatch(mine, oracle)
        assert d == 0, f"{name}: {d} mismatching lanes"


def test_sparse_bitidentity_and_odd_tails() -> None:
    """Sparse variant: (a) bit-identical per valid element to dense on the
    compacted synthetic scene; (b) direct raw-bit parity vs torch on odd
    extents including the all-scalar small-n path and tail lanes seeded
    with divergent bits (M7 designated killer)."""
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(8)
    from solweig_gpu.calculate_utci import utci_calculator

    ta, rh, va, tmrt = _build_scene("extreme", 777)
    dense = utci.utci_calculator_dense(ta, rh, tmrt, va, torch_threads=8)
    tv, rv, mv, xv = utci.compact_valid(ta, rh, tmrt, va)
    sparse = utci.utci_calculator_sparse(tv, rv, mv, xv, torch_threads=8)
    valid_flat = dense != F32(-999.0)
    assert _mismatch(sparse, dense.reshape(-1)[valid_flat.reshape(-1)]) == 0

    rng = np.random.default_rng(99)
    for n in (209421, 8191, 13, 7, 1):
        ta_v = rng.uniform(-30.0, 45.0, n).astype(np.float32)
        rh_v = rng.uniform(5.0, 100.0, n).astype(np.float32)
        va_v = rng.uniform(0.15, 15.0, n).astype(np.float32)
        tm_v = (ta_v + rng.uniform(-30.0, 60.0, n)).astype(np.float32)
        # divergent bits across the extent, dense in the final 8 lanes so
        # the per-chunk scalar tail is always seeded
        for pos in list(range(0, n, max(1, n // 37))) + list(range(max(0, n - 8), n)):
            va_v[pos] = f32bits(DIVERGENT_BITS["va"][5])
            ta_v[pos] = f32bits(DIVERGENT_BITS["ta"][4])
        ta_v = np.ascontiguousarray(ta_v); rh_v = np.ascontiguousarray(rh_v)
        va_v = np.ascontiguousarray(va_v); tm_v = np.ascontiguousarray(tm_v)
        oracle = utci_calculator(torch.tensor(ta_v), torch.tensor(rh_v),
                                 torch.tensor(tm_v), torch.tensor(va_v)).numpy()
        mine = utci.utci_calculator_sparse(ta_v, rh_v, tm_v, va_v,
                                           torch_threads=8)
        d = _mismatch(mine, oracle)
        assert d == 0, f"n={n}: {d} mismatching lanes"


def test_tail_lane_libm_witness_site_extent() -> None:
    """Site_500-shaped extent (n=209422, teff=7, C=29918, tail lanes
    209420-209421): the per-chunk opmath-f64 pow tail must classify
    exactly as torch does — divergent bits at the tail lanes and a body
    control lane."""
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(8)
    from solweig_gpu.calculate_utci import utci_calculator

    n = 209422
    rng = np.random.default_rng(5)
    ta_v = np.full(n, F32(21.3)).astype(np.float32)
    rh_v = rng.uniform(30.0, 80.0, n).astype(np.float32)
    va_v = np.full(n, F32(3.1)).astype(np.float32)
    tm_v = (ta_v + 12.7).astype(np.float32)
    for pos in (0, 100000, 209419, 209420, 209421):
        va_v[pos] = f32bits(DIVERGENT_BITS["va"][5])
        ta_v[pos] = f32bits(DIVERGENT_BITS["ta"][6])
    ta_v = np.ascontiguousarray(ta_v); rh_v = np.ascontiguousarray(rh_v)
    va_v = np.ascontiguousarray(va_v); tm_v = np.ascontiguousarray(tm_v)
    oracle = utci_calculator(torch.tensor(ta_v), torch.tensor(rh_v),
                             torch.tensor(tm_v), torch.tensor(va_v)).numpy()
    mine = utci.utci_calculator_sparse(ta_v, rh_v, tm_v, va_v,
                                       torch_threads=8)
    assert _mismatch(mine, oracle) == 0


# ---------------------------------------------------------------------------
# primitive gates (math_compat)
# ---------------------------------------------------------------------------

def test_exact_pow_forms_lattice() -> None:
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(8)
    rng = np.random.default_rng(31)
    ta = np.arange(-50.0, 55.0005, 0.005, dtype=np.float64)
    bases = np.unique(np.concatenate([
        (ta + 273.15).astype(np.float32), rng.uniform(200.0, 340.0, 4000),
        rng.uniform(0.01, 70.0, 4000),
        np.array([np.nan, 0.0, -0.0, 1.0, -1.0, np.inf, -np.inf],
                 dtype=np.float32),
    ]).astype(np.float32))
    for e in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        t = torch.pow(torch.tensor(bases), e).numpy()
        m = math_compat.torch_pow_scalar_vector(
            np.ascontiguousarray(bases), F32(e), 8, 32768)
        assert _mismatch(m, t) == 0, f"e={e}"


def test_layout_model_sizes() -> None:
    """_is_libm_lane == torch's observed scalar-tail lane set on 15 sizes
    (all lanes seeded with a SLEEF-vs-opmath-f64 divergent pow(x,5)
    input; M7 designated killer)."""
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(8)
    from numba import njit

    @njit(cache=False, fastmath=False, error_model="numpy")
    def _sleef_vec(xs, e):
        out = np.empty(xs.size, dtype=np.float32)
        for i in range(xs.size):
            out[i] = math_compat.sleef_powf(xs[i], e)
        return out

    x = f32bits(DIVERGENT_BITS["va"][5])
    for n in (1, 3, 7, 8, 9, 13, 100, 8191, 32768, 32769, 65537,
              100003, 209422, 250000, 1048576):
        xs = np.full(n, x, dtype=np.float32)
        t = torch.pow(torch.tensor(xs), 5.0).numpy()
        s = _sleef_vec(xs, F32(5.0))
        observed = set(np.nonzero(t.view(np.uint32) != s.view(np.uint32))[0].tolist())
        model = {i for i in range(n)
                 if math_compat._is_libm_lane(i, n, 8, 32768)}
        assert observed == model, f"n={n}: extra={observed - model} missing={model - observed}"


def test_exp_log_primitives_domain() -> None:
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(8)
    from numba import njit

    @njit(cache=False, fastmath=False, error_model="numpy")
    def _exp_vec(xs):
        out = np.empty(xs.size, dtype=np.float32)
        for i in range(xs.size):
            out[i] = math_compat.sleef_expf(xs[i])
        return out

    @njit(cache=False, fastmath=False, error_model="numpy")
    def _log_vec(xs):
        out = np.empty(xs.size, dtype=np.float32)
        for i in range(xs.size):
            out[i] = math_compat.sleef_logf_u1(xs[i])
        return out

    rng = np.random.default_rng(11)
    tk = np.unique(np.concatenate([
        np.arange(203.15, 333.15, 0.001, dtype=np.float64).astype(np.float32),
        rng.uniform(150.0, 350.0, 20000),
        np.array([np.nan, 0.0, -0.0, 1.0, np.inf, -np.inf], dtype=np.float32),
    ]).astype(np.float32))
    es_in = np.concatenate([tk - F32(273.15),
                            rng.uniform(-103.0, 100.0, 20000)
                            .astype(np.float32)])
    t_e = torch.exp(torch.tensor(es_in)).numpy()
    t_l = torch.log(torch.tensor(tk)).numpy()
    assert _mismatch(_exp_vec(np.ascontiguousarray(es_in)), t_e) == 0
    assert _mismatch(_log_vec(np.ascontiguousarray(tk)), t_l) == 0


# ---------------------------------------------------------------------------
# site_500 chain gate (skips without the capture bundle)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not capture_present(), reason="t08 capture/site absent")
def test_site_500_chain() -> None:
    """Full chain from T08 fused outputs: mine == live torch recompute ==
    published baseline plane at t in {0, 5, 12, 23} (raw bits)."""
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(8)
    from solweig_gpu.utci_process import recompute_utci_steps

    met = np.loadtxt(SITE / "metfiles/metfile_0_0.txt", skiprows=1,
                     dtype=np.float64)
    bd = np.load(SITE / "static/building_dsm.f32.npy")
    dem = np.load(SITE / "static/dem.f32.npy")
    buildings = utci.buildings_from_dsm(bd, dem)
    n_valid = int(np.count_nonzero(buildings == 1))
    assert n_valid == 209422

    base = np.load(SITE / "baseline_results/utci.f32.npy")
    base_tmrt = np.load(SITE / "baseline_results/tmrt.f32.npy")
    sha = hashlib.sha256(base.tobytes()).hexdigest()[:16]
    assert sha == BASELINE_UTCI_SHA16, f"baseline digest drifted: {sha}"

    ts_list = (0, 5, 12, 23)
    planes = {}
    for ts in ts_list:
        planes[ts] = np.ascontiguousarray(
            np.load(CAP / f"t{ts:02d}.npz")["ret_Tmrt"], dtype=np.float32)
        assert _mismatch(planes[ts], base_tmrt[ts]) == 0, f"t{ts} tmrt"

    oracle = recompute_utci_steps(met, list(ts_list), planes, bd, dem)
    for j, ts in enumerate(ts_list):
        mine = utci.apply_buildings_mask(
            utci.utci_met_step(planes[ts], met[ts, 11], met[ts, 10],
                               met[ts, 9], uhii_i=0.0, torch_threads=8),
            buildings)
        assert _mismatch(mine, oracle[j]) == 0, f"t{ts} vs live oracle"
        assert _mismatch(mine, base[ts]) == 0, f"t{ts} vs baseline"
