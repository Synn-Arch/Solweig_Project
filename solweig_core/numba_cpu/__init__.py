# SPDX-License-Identifier: GPL-3.0-only
"""Torch-free Numba CPU kernels (DESIGN.ko.md 5.2/12.1, TASKS T04+).

``solweig_core.numba_cpu.march`` owns the dense serial shadow march that
consumes T03 step tables. The package never imports torch; numpy + numba
only.
"""
from solweig_core.numba_cpu.march import march_svf_shadow, march_wallheight23

__all__ = ["march_svf_shadow", "march_wallheight23"]
