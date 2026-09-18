# SPDX-License-Identifier: GPL-3.0-only
"""ISA-mismatch cache probe (T14 Change D witness vehicle).

Run with ``NUMBA_CACHE_DIR`` (and optionally ``NUMBA_CPU_NAME``) set;
prints ``@@RESULT@@ <sha256>`` of the computed bits and the sha256 of
every cache artifact afterwards. The packaging test drives it three
times (cold / warm-same-CPU / different-CPU-name) to witness:

* same CPU name, warm cache -> artifacts byte-identical (pure hit);
* different CPU name (a stale cross-ISA cache directory) -> artifacts
  REWRITTEN (the ISA key misses; numba recompiles instead of reusing
  foreign-ISA objects) while the computed bits stay identical.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def probe(x):
    return x + np.float32(1.0) - np.float32(0.5) * np.float32(2.0)


def _main() -> int:
    value = probe(np.float32(2.0))
    bits = np.float32(value).view(np.uint32).tobytes()
    result = hashlib.sha256(bits).hexdigest()
    cache = Path(sys.argv[1])
    print(f"@@RESULT@@ {result}")
    for f in sorted(cache.rglob("*")):
        if f.is_file():
            print(f.name, hashlib.sha256(f.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
