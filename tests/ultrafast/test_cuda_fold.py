# SPDX-License-Identifier: GPL-3.0-only
"""T12 GATE 3 — CUDA SVF fold bit-equality (TASKS T12, DESIGN 3.4/6.3).

The CUDA fold (native/cuda/src/sw_fold.cu, one thread per target cell
replaying the patch-major/annulus-minor float32 recurrence over the T07
packed bit state) must reproduce the CANONICAL CPU fold BITS
(solweig_core/numba_cpu/svf_fold.py — the frozen T07 reference, bit-equal
to the original svf_calculator fold) on:

* the full synthetic pin set — density sweep, degenerate cubes,
  special-value planes (``last`` predicate on +0.0/-0.0/NaN vegdem2,
  NaN/±0/sub-unit/super-unit svf_building) — full-plane raw uint32
  equality on all 11 outputs (signed zero + NaN payload, no tolerance);
* the masked route — masked cells equal the full fold, unmasked cells
  bit-identical to the caller's base planes (affected-chunk-only
  contract).

RED witness 5 (FP reduction reordering): the raw-bit pins are the
designated killer — any pre-summed annulus weight, reordered patch
reduction, or contracted multiply-add changes float32 bits that these
comparations pin (proven live by the mutation matrix, which builds a
reordered fold and expects these tests to fail).

The frozen tables are DATA: the pins record the module digest AND an
independent digest recomputed from the serialized arrays; the tests
recompute both, so any single-bit table mutation fails here before any
kernel runs.
"""
from __future__ import annotations

import hashlib
import json
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
)

NATIVE_CUDA = ULTRA_DIR.parents[1] / "native" / "cuda"

FOLD_OUTPUTS = (
    "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
    "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
    "svftotal",
)


@pytest.fixture(scope="module")
def rt():
    return require_canonical_runtime()


@pytest.fixture(scope="module")
def host():
    return load_host_module()


@pytest.fixture(scope="module")
def pins():
    path = NATIVE_CUDA / "data" / "fold_pins.json"
    if not path.is_file():
        pytest.skip(f"fold pins not generated ({path})")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def tables(pins):
    t = pins["tables"]
    return {
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


def u8_from_hex(hex_list) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_list], dtype=np.uint8)


def run_case(host, rt, entry, tables):
    shape = (entry["rows"], entry["cols"])
    nbp = 20
    veg = u8_from_hex(entry["veg_bytes"]).reshape((*shape, nbp))
    vbsh = u8_from_hex(entry["vbsh_bytes"]).reshape((*shape, nbp))
    vegdem2 = f32_from_hex(entry["vegdem2"]).reshape(shape)
    svf = f32_from_hex(entry["svf_building"]).reshape(shape)
    kwargs = {}
    if "mask" in entry:
        kwargs["cell_mask"] = np.array(entry["mask"], dtype=np.uint8).reshape(
            shape)
        kwargs["out_base"] = f32_from_hex(entry["out_base"]).reshape(
            (*shape, 11))
    return host.run_fold(rt, veg, vbsh, vegdem2, svf, tables, **kwargs)


# ---------------------------------------------------------------------------
# frozen tables are DATA (digest cross-tie, killed before any kernel runs)
# ---------------------------------------------------------------------------


class TestFoldTablesAreFrozenData:
    def test_serialized_tables_digest(self, pins):
        t = pins["tables"]
        parts = [
            str(t["n_patches"]),
            ",".join(str(x) for x in t["na"]),
            ",".join(f"{int(h, 16):08x}" for h in t["w_iso"]),
            ",".join(f"{int(h, 16):08x}" for h in t["w_aniso"]),
            ",".join(f"{int(h, 16):08x}" for h in t["azimuth"]),
            ",".join(str(x) for x in t["ring"]),
            t["last_bits"],
            t["trans_bits"],
        ]
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        assert digest == t["fold_constants_sha256_serialized"], (
            "serialized tables no longer digest to their recorded value"
        )
        assert t["fold_constants_sha256_serialized"] == \
            t["fold_constants_sha256_module"] == \
            "0a13e80a61b1cae69344c2517f45df6db6675bc2578b91cb909d2673c57519a8"

    def test_direction_predicates_partition(self, pins):
        t = pins["tables"]
        counts = (np.array(t["dir_e"], dtype=np.int32)
                  + np.array(t["dir_s"], dtype=np.int32)
                  + np.array(t["dir_w"], dtype=np.int32)
                  + np.array(t["dir_n"], dtype=np.int32))
        assert len(counts) == 153
        assert counts.min() == 2 and counts.max() == 2, (
            "each patch must belong to exactly two directions"
        )


# ---------------------------------------------------------------------------
# full-plane raw-bit equality (RED witness 5 designated killer)
# ---------------------------------------------------------------------------


class TestFoldRawBitEquality:
    def test_all_cases_11_outputs_raw_bit_equal(self, rt, host, pins, tables):
        failures = []
        for entry in pins["cases"]:
            got = run_case(host, rt, entry, tables)
            for k, name in enumerate(FOLD_OUTPUTS):
                want = f32_from_hex(entry[name]).reshape(got.shape[:2])
                msg, _ = compare_bits(want, got[:, :, k],
                                      f"{entry['name']}:{name}")
                if msg != "PASS":
                    failures.append(msg)
        assert not failures, "\n".join(failures[:6])

    def test_pin_set_covers_witnesses(self, pins):
        names = {c["name"] for c in pins["cases"]}
        # last-predicate lanes (zero vegdem2), NaN lanes, masked route
        assert any("special" in n for n in names), \
            "no special-value case pinned"
        assert "masked_affine" in names, "no masked case pinned"
        assert any(n.startswith("rand_p") for n in names), \
            "no density-sweep case pinned"
        # the masked case must actually mask a strict subset
        entry = next(c for c in pins["cases"] if c["name"] == "masked_affine")
        mask = np.array(entry["mask"], dtype=bool)
        assert 0 < int(mask.sum()) < mask.size

    def test_masked_unmasked_cells_keep_base_bits(self, rt, host, pins,
                                                  tables):
        """Affected-chunk-only: outside the mask the 11 outputs stay
        bit-identical to the caller's base planes (incl. NaN/±0 lanes)."""
        entry = next(c for c in pins["cases"] if c["name"] == "masked_affine")
        shape = (entry["rows"], entry["cols"])
        mask = np.array(entry["mask"], dtype=bool).reshape(shape)
        base = f32_from_hex(entry["out_base"]).reshape((*shape, 11))
        got = run_case(host, rt, entry, tables)
        want = base[~mask].view(np.uint32)
        have = got[~mask].view(np.uint32)
        assert np.array_equal(want, have), (
            f"{int((want != have).any(axis=1).sum())} unmasked cells changed"
        )
        # and the masked cells equal the pinned full-fold values
        for k, name in enumerate(FOLD_OUTPUTS):
            want_m = f32_from_hex(entry[name]).reshape(shape)
            msg, _ = compare_bits(want_m[mask], got[mask][:, k],
                                  f"masked:{name}")
            assert msg == "PASS", msg
