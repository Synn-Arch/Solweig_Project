# SPDX-License-Identifier: GPL-3.0-only
"""Shared configuration for the T3 scientific differential suite.

Marker ``scientific`` is registered in pytest.ini. Heavy differential runs
(expensive oracle replay) are selected with ``-m scientific``; the metrics
unit tests in this package run in the default fast suite.
"""

from __future__ import annotations
