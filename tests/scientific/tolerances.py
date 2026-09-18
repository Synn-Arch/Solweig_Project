# SPDX-License-Identifier: GPL-3.0-only
"""Tolerance constants for the T3 scientific differential gates.

Source: docs/incremental_design_tool/agent/validation_pipeline.md and the
incremental algorithm spec. The exact worker is expected to match the
full-tile oracle far below these bounds (identical arithmetic ordering is
the design goal); these constants are the outer acceptance envelope.

``OUTSIDE_WINDOW_EXACT`` is not a tolerance: outside the write window the
candidate must be bit-identical to stored baseline results.
"""

from __future__ import annotations

# Continuous variables (degrees C). Differential errors are expected to be
# ~1e-5 (reordering-free replay); these bound acceptable numeric drift.
UTCI_MAE_MAX = 0.02
UTCI_P99_MAX = 0.10
UTCI_MAX_ABS = 0.25
TMRT_MAE_MAX = 0.05
TMRT_P99_MAX = 0.20
TMRT_MAX_ABS = 1.0

# Binary shadow planes: fraction of cells allowed to disagree inside the
# write window (target is 0 for the exact worker).
SHADOW_MISMATCH_FRACTION_MAX = 0.001

# Exact equality is required outside the write window, on the boundary-ring
# cells the seam behaviour is reported but bounded by the inside tolerances.
OUTSIDE_WINDOW_EXACT = True

# Boundary ring width (cells) measured inward from the write-window edge.
BOUNDARY_RING_CELLS = (0, 2)

# Distance bands (cells inward from the window edge) used to localise any
# seam-dependent error growth.
DISTANCE_BANDS = ((0, 2), (3, 8), (9, 16))
