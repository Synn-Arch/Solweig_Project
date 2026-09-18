# SPDX-License-Identifier: GPL-3.0-only
"""Executor building-chain vocabulary fence tests (U-D1, intake item c).

The executor's building chain reads ``worker.requested_variables``
directly; the fence added in U-D1 makes a future widening of the patch
transport vocabulary FAIL LOUDLY at the executor instead of silently
asking the regeneration chain for variables it cannot produce.
"""

from __future__ import annotations

import inspect

import pytest

from solweig_gpu.incremental.executor import (
    _BUILDING_CHAIN_VARIABLES,
    ExecutorError,
    PlanExecutor,
)
from solweig_gpu.incremental.regenerate import regenerate_building_batch
from solweig_gpu.incremental.result import SUPPORTED_VARIABLES


def _chain_default_vocabulary() -> frozenset[str]:
    """The regeneration chain's own declared default vocabulary."""
    default = inspect.signature(regenerate_building_batch).parameters[
        "requested_variables"
    ].default
    return frozenset(default)


def test_fence_constant_is_the_chain_signature_default() -> None:
    """The fence is DERIVED from the chain, never a second list."""
    assert _BUILDING_CHAIN_VARIABLES == _chain_default_vocabulary()
    assert _BUILDING_CHAIN_VARIABLES == {"utci", "tmrt", "shadow"}


def test_transport_vocabulary_cannot_outgrow_the_chain() -> None:
    """Drift guard: if the patch transport widens (SUPPORTED_VARIABLES
    gains a variable) without widening ``regenerate_building_batch``, the
    executor fence will start refusing building batches — this test
    fails first, naming the real fix (widen the chain, and the fence
    follows automatically from the signature)."""
    assert set(SUPPORTED_VARIABLES) <= _BUILDING_CHAIN_VARIABLES


def test_widened_vocabulary_fails_loudly_at_the_executor(tmp_path) -> None:
    """A widened requested set is refused BEFORE any chain work runs."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_incremental_scenario_state import _make_executor

    executor: PlanExecutor = _make_executor(tmp_path)
    executor._worker.requested_variables = ("utci", "tmrt", "shadow", "kup")

    with pytest.raises(ExecutorError, match="cannot produce requested variables") as info:
        executor._regenerate_building_chain(
            (),
            target_revision=executor.scene_revision + 1,
            forcing_overlay=None,
            landcover_overlay=None,
        )
    assert "kup" in str(info.value)
    assert "regenerate_building_batch" in str(info.value)


def test_documented_vocabulary_passes_the_fence(tmp_path) -> None:
    """The exact documented vocabulary is inside the fence (the chain
    runs; with no edits it degenerates, so we only prove the fence lets
    the call THROUGH it by checking a non-fence failure mode)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_incremental_scenario_state import _make_executor

    executor: PlanExecutor = _make_executor(tmp_path)
    executor._worker.requested_variables = ("shadow", "tmrt", "utci")

    # Empty edits: the fence passes, the chain itself rejects the empty
    # batch with its own (different) error — proving the vocabulary check
    # did not fire.
    with pytest.raises(Exception) as info:
        executor._regenerate_building_chain(
            (),
            target_revision=executor.scene_revision + 1,
            forcing_overlay=None,
            landcover_overlay=None,
        )
    assert "cannot produce requested variables" not in str(info.value)
