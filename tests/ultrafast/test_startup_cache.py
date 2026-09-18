# SPDX-License-Identifier: GPL-3.0-only
"""T14a startup-cache gates — the UTCI family's numba disk cache (R8 fix).

Before T14a every @njit kernel in ``solweig_core/numba_cpu/utci.py`` and
``math_compat.py`` was ``cache=False``: the polynomial kernel was exec'd
from ``POLY_EXPRESSION_SOURCE`` (``co_filename='<utci_poly_element>'``),
and numba raises ``RuntimeError: ... no locator available`` for a cache
request on such a function, so the whole family recompiled in every
process (audit: utci dense 4.5-7.4 s per process vs the 3000 ms
process-to-ready budget). T14a materializes the polynomial into the real
module ``_utci_poly_gen.py`` (byte-exact, import-time drift gate) and
flips the family to ``cache=True``.

These gates pin the mechanism, not a wall clock:

- WARM-START (test_warm_start_cache_artifacts): a subprocess calling the
  public UTCI entries leaves a numba cache index for EVERY family member
  (utci kernels, math_compat primitives, the generated polynomial
  module), a second subprocess against the same cache dir replays them
  (no new indexes, cold-vs-warm first-call ratio recorded to the t14a
  artifacts dir — never wall-asserted), and neither run emits a numba
  ``Cannot cache`` / ``no locator`` refusal.
- CROSS-PROCESS BIT PARITY (test_cross_process_bit_parity): the same
  computation in a cold-cache-dir subprocess (first compile), a
  warm-cache-dir subprocess (disk-cache replay) and this process
  (in-process reference) yields byte-identical sha256 digests, on a
  scene seeded with NaN payloads, -999 invalids, signed zeros and the
  frozen SLEEF-vs-opmath-f64-pow divergent bits (both lane classes).
- CACHE DETERMINISM / DRIFT: two fresh cache dirs over the same source
  produce the same index inventory and digests (determinism); an edited
  generated module makes ``utci`` REFUSE to import before any cache
  artifact can be served (drift gate — designated killer for a bypassed
  gate); an edited kernel source with a pre-warmed cache produces the
  edited bits, never the stale artifact's (numba mtime+size staleness).
- PURITY: the torch-free stack's third-party import set in a fresh
  subprocess stays within {numpy, numba, llvmlite, scipy} + numba's own
  transitives (coverage, yaml), with no torch.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts")
ART_T14A = ART / "t14a"

#: every @njit kernel of the UTCI family that the payload below reaches.
#: A cache flag silently reverted to cache=False removes the member's
#: .nbi from the cache dir and fails the warm-start gate (T14a mutation
#: MT3 designated killer).
EXPECTED_CACHE_MEMBERS = {
    "utci._utci_dense_kernel",
    "utci._utci_sparse_kernel",
    "utci._utci_element",
    "_utci_poly_gen._utci_poly_element",
    "math_compat.sleef_expf",
    "math_compat.sleef_logf_u1",
    "math_compat.sleef_powf",
    "math_compat._logkf",
    "math_compat._expkf",
    "math_compat._vldexp",
    "math_compat._is_libm_lane",
    "math_compat._torch_pow_scalar",
    "math_compat.nadd",
    "math_compat.nsub",
    "math_compat.nmul",
    "math_compat.ndiv",
}

#: frozen SLEEF-vs-opmath-f64-pow divergent input bits (T09 P9; same
#: witnesses as tests/ultrafast/test_utci.py DIVERGENT_BITS) so the
#: parity digest is sensitive to lane-class routing, not just values.
_DIVERGENT_U32 = (0xC223D5B8, 0x410D5DD7, 0x409A03DA, 0x41890F0C)

#: the subprocess payload. Deterministic scene builder + digest emitter;
#: ``run()`` imports solweig_core from ``$T14A_ROOT`` so the same text
#: serves the real tree, copied trees and the in-process reference.
_PAYLOAD = '''
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np


def _f32bits(u32):
    a = np.empty(1, dtype=np.float32)
    a.view(np.uint32)[0] = np.uint32(u32)
    return a[0]


def build_inputs():
    rng = np.random.default_rng(20260908)
    rows, cols = 96, 128
    ta = rng.uniform(-10, 40, (rows, cols)).astype(np.float32)
    rh = rng.uniform(0, 100, (rows, cols)).astype(np.float32)
    tm = rng.uniform(-10, 70, (rows, cols)).astype(np.float32)
    va = rng.uniform(0, 17, (rows, cols)).astype(np.float32)
    ta[0, 0] = np.nan                       # qNaN payload stays a valid lane
    rh[1, 1] = np.float32(-0.0)             # signed zero
    ta[2, 2] = np.float32(-999.0)           # invalid lanes
    va[3, 3] = np.float32(-999.0)
    tm[4, 4] = np.float32(-999.0)
    rh[5, 5] = np.float32(np.nan)           # NaN in a different operand slot
    for j, u in enumerate(_DIVERGENT_U32):
        ta[6, 6 + j] = _f32bits(u)
        va[7, 6 + j] = _f32bits(u)
    return ta, rh, tm, va


def _dig(a):
    return hashlib.sha256(
        np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


def run():
    sys.path.insert(0, os.environ["T14A_ROOT"])
    t0 = time.perf_counter()
    from solweig_core.numba_cpu import utci
    t_imp = time.perf_counter() - t0
    ta, rh, tm, va = build_inputs()
    out = {"timings": {"import_s": t_imp}, "warnings": [], "digests": {}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        dense = utci.utci_calculator_dense(ta, rh, tm, va)
        out["timings"]["dense_first_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        tv, rv, mv, xv = utci.compact_valid(ta, rh, tm, va)
        tv[8188] = _f32bits(_DIVERGENT_U32[1])
        tv[8190] = _f32bits(_DIVERGENT_U32[2])
        sp = utci.utci_calculator_sparse(tv[:8191], rv[:8191], mv[:8191],
                                         xv[:8191])
        out["timings"]["sparse_first_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        met = utci.utci_met_step(tm, 21.3, 55.0, 3.1)
        out["timings"]["met_first_s"] = time.perf_counter() - t0
        out["warnings"] = [str(wi.message) for wi in w]
    out["digests"] = {"dense": _dig(dense), "sparse8191": _dig(sp),
                      "met": _dig(met)}
    return out


if __name__ == "__main__":
    print("@@RESULT@@" + json.dumps(run()))
'''.replace("_DIVERGENT_U32", repr(_DIVERGENT_U32))


def _run_payload(cache_dir: Path, root: Path, payload_file: Path | None = None):
    """Run the payload in a fresh subprocess with an isolated numba cache.

    Returns (parsed dict, stderr). Fails hard on nonzero exit so a gate
    never mistakes an import error for a pass.
    """
    env = dict(os.environ)
    env["T14A_ROOT"] = str(root)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    if payload_file is None:
        payload_file = ART_T14A / "gates" / "_startup_payload.py"
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    payload_file.write_text(_PAYLOAD)
    proc = subprocess.run(
        [sys.executable, str(payload_file)], capture_output=True, text=True,
        env=env)
    assert proc.returncode == 0, (
        f"payload failed (root={root}, cache={cache_dir}):\n{proc.stderr}")
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT@@")]
    assert marker, f"no result marker in stdout:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(marker[0][len("@@RESULT@@"):]), proc.stderr


def _cache_index_names(cache_dir: Path) -> set[str]:
    """`module.function` names of every .nbi index under a cache dir
    (lineno and python-version suffixes stripped)."""
    names = set()
    for p in cache_dir.rglob("*.nbi"):
        stem = p.name.split(".py3")[0]           # drop .pyXY[abiflags].nbi
        names.add(__import__("re").sub(r"-\d+$", "", stem))  # drop lineno
    return names


# ---------------------------------------------------------------------------
# W1: warm-start gate — cache artifacts exist and are replayed
# ---------------------------------------------------------------------------

def test_warm_start_cache_artifacts(tmp_path) -> None:
    """Every family member leaves a disk-cache index; a second process
    against the same cache dir reuses them (no new indexes, no cache
    refusal warnings, cold/warm first-call ratio recorded to artifacts)."""
    cache_dir = tmp_path / "cache"
    cold, stderr_cold = _run_payload(cache_dir, REPO_ROOT)
    got = _cache_index_names(cache_dir)
    missing = EXPECTED_CACHE_MEMBERS - got
    assert not missing, f"no numba cache index for: {sorted(missing)}"
    for err in ("Cannot cache", "no locator available"):
        assert err not in stderr_cold, f"numba cache refusal: {err}"

    before = _cache_index_names(cache_dir)
    n_nbc_before = len(list(cache_dir.rglob("*.nbc")))
    warm, stderr_warm = _run_payload(cache_dir, REPO_ROOT)
    after = _cache_index_names(cache_dir)
    assert EXPECTED_CACHE_MEMBERS <= after
    assert after == before, f"warm run created new cache indexes: {after - before}"
    assert len(list(cache_dir.rglob("*.nbc"))) == n_nbc_before, (
        "warm run compiled new overloads (cache not replayed)")
    for err in ("Cannot cache", "no locator available"):
        assert err not in stderr_warm, f"numba cache refusal: {err}"

    record = {
        "ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "cold_timings_s": cold["timings"],
        "warm_timings_s": warm["timings"],
        "cache_members": sorted(after),
        "numba_cache_dir_env": True,
    }
    if ART_T14A.is_dir():
        out = ART_T14A / "gates" / "warm_start_timings.jsonl"
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
    # mechanism assertion (not a wall-clock target): replay must beat
    # first-compile by a wide factor; observed >10x, asserted >2x.
    assert warm["timings"]["dense_first_s"] < \
        cold["timings"]["dense_first_s"] / 2, (cold["timings"], warm["timings"])


# ---------------------------------------------------------------------------
# W2: cross-process bit parity — first-compile == disk-cache replay == local
# ---------------------------------------------------------------------------

def test_cross_process_bit_parity(tmp_path) -> None:
    """Cold-cache subprocess, warm-cache subprocess and the in-process
    reference produce byte-identical UTCI outputs (sha256 over raw f32
    payload, NaN payloads / signed zeros / divergent pow bits included)."""
    cache_dir = tmp_path / "parity"
    cold, _ = _run_payload(cache_dir, REPO_ROOT)
    warm, _ = _run_payload(cache_dir, REPO_ROOT)
    ns: dict = {}
    exec(compile(_PAYLOAD, "<t14a_payload>", "exec"), ns)
    os.environ["T14A_ROOT"] = str(REPO_ROOT)
    local = ns["run"]()

    assert cold["digests"] == warm["digests"], (
        "disk-cache replay changed the bits:\n"
        f"cold={cold['digests']}\nwarm={warm['digests']}")
    assert cold["digests"] == local["digests"], (
        "subprocess vs in-process divergence:\n"
        f"cold={cold['digests']}\nlocal={local['digests']}")
    if ART_T14A.is_dir():
        out = ART_T14A / "gates" / "cross_process_parity.json"
        out.write_text(json.dumps({
            "digests_cold": cold["digests"],
            "digests_warm": warm["digests"],
            "digests_inprocess": local["digests"],
            "equal": True,
        }, indent=2) + "\n")


# ---------------------------------------------------------------------------
# W3: cache determinism and staleness
# ---------------------------------------------------------------------------

def test_cache_determinism_two_fresh_dirs(tmp_path) -> None:
    """Same source + two fresh cache dirs -> identical index inventory and
    identical output digests (the cache is deterministic in what it
    produces and in what it serves)."""
    d1, _ = _run_payload(tmp_path / "c1", REPO_ROOT)
    d2, _ = _run_payload(tmp_path / "c2", REPO_ROOT)
    assert d1["digests"] == d2["digests"]
    n1 = _cache_index_names(tmp_path / "c1")
    n2 = _cache_index_names(tmp_path / "c2")
    assert n1 == n2, f"cache inventory diverged: {n1 ^ n2}"
    assert EXPECTED_CACHE_MEMBERS <= n1


def _copy_tree(tmp_path, name="solweig_core"):
    dest = tmp_path / name
    shutil.copytree(REPO_ROOT / "solweig_core", dest,
                    ignore=shutil.ignore_patterns("__pycache__"))
    return dest.parent


def test_drift_gate_edited_generated_module(tmp_path) -> None:
    """Editing _utci_poly_gen.py (a copied tree) must make utci.py REFUSE
    to import — the drift gate fires before any numba cache artifact,
    stale or not, can be served (T14a mutation MT1/MT2 designated
    killer; numba's own staleness check is mtime+size, not content)."""
    root = _copy_tree(tmp_path)
    gen = root / "solweig_core" / "numba_cpu" / "_utci_poly_gen.py"
    text = gen.read_text()
    assert "F32(0.607562052)" in text
    gen.write_text(text.replace("F32(0.607562052)", "F32(0.607562053)", 1))
    env = dict(os.environ)
    env["T14A_ROOT"] = str(root)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "cache")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import os, sys; sys.path.insert(0, os.environ['T14A_ROOT']);\n"
         "from solweig_core.numba_cpu import utci"],
        capture_output=True, text=True, env=env)
    assert proc.returncode != 0, "drifted _utci_poly_gen.py imported cleanly"
    assert "drift" in proc.stderr, proc.stderr
    assert "0.607562053" not in proc.stdout


def test_no_stale_cache_hit_for_edited_source(tmp_path) -> None:
    """numba staleness pin: pre-warm a cache dir with a pristine copy,
    then edit a kernel constant (TK_OFF) in the copy — the edited run
    must produce the EDITED bits (== a fresh-cache run of the edited
    tree), never the stale artifact's pristine bits."""
    root = _copy_tree(tmp_path)
    utci_py = root / "solweig_core" / "numba_cpu" / "utci.py"
    cache = tmp_path / "cache"

    pristine, _ = _run_payload(cache, root)
    text = utci_py.read_text()
    anchor = "TK_OFF = F32(273.15)"
    assert anchor in text
    utci_py.write_text(text.replace(anchor, "TK_OFF = F32(274.0)", 1))

    warm_stale, _ = _run_payload(cache, root)          # warm, pre-edit artifacts
    fresh_edited, _ = _run_payload(tmp_path / "c2", root)  # true recompile
    assert warm_stale["digests"] != pristine["digests"], (
        "edited source served the STALE cache artifact's bits")
    assert warm_stale["digests"] == fresh_edited["digests"], (
        "warm stale-cache run diverged from a true recompile of the "
        "edited source")


# ---------------------------------------------------------------------------
# W4: purity of the torch-free stack after the caching change
# ---------------------------------------------------------------------------

def test_purity_closed_import_set(tmp_path) -> None:
    """Fresh subprocess: the torch-free stack (step_tables, sparse_work,
    bitplanes, numba_cpu family incl. the generated module) pulls no
    torch and adds NO third-party top beyond the declared toolchain
    {numpy, numba, llvmlite, scipy} (+ their transitives, e.g. coverage/
    yaml via numba, cython runtime via scipy wheels). Measured as the
    DELTA over a control subprocess that imports the toolchain alone, so
    toolchain-internal noise is not miscounted as a stack dependency."""
    def _tops(body: str):
        code = body + (
            "\ntops = sorted({m.split('.')[0] for m in sys.modules}\n"
            "               - set(sys.stdlib_module_names))\n"
            "print('TOPS', tops)\n"
        )
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = str(tmp_path / "cache")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("TOPS ")]
        assert line, proc.stdout
        return set(eval(line[0][len("TOPS "):]))

    control = _tops("import sys; import numpy, numba, llvmlite, scipy")
    stack = _tops(
        "import os, sys; sys.path.insert(0, %r);\n"
        "import importlib\n"
        "for m in ['solweig_core.step_tables', 'solweig_core.sparse_work',\n"
        "          'solweig_core.bitplanes', 'solweig_core.numba_cpu.march',\n"
        "          'solweig_core.numba_cpu.sparse_march',\n"
        "          'solweig_core.numba_cpu.svf_fold',\n"
        "          'solweig_core.numba_cpu.radiation',\n"
        "          'solweig_core.numba_cpu.utci',\n"
        "          'solweig_core.numba_cpu.math_compat',\n"
        "          'solweig_core.numba_cpu._utci_poly_gen']:\n"
        "    importlib.import_module(m)\n"
        "assert 'torch' not in sys.modules\n"
        "assert not any(m.startswith('solweig_gpu') for m in sys.modules)\n"
        % str(REPO_ROOT)
    )
    delta = stack - control - {"solweig_core"}
    assert not delta, f"third-party tops beyond the toolchain: {delta}"
