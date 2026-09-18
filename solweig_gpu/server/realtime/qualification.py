# SPDX-License-Identifier: GPL-3.0-only
"""The structural qualification contract for ``fast_qualified`` results (R7).

A fast kernel may publish ``fast_qualified`` ONLY when its payload carries a
COMPLETE anchored qualifier. This module is the kernel-agnostic validator the
fast lane applies structurally (scheduler.py): a ``fast_qualified`` result
whose payload has ANY defect below is a kernel defect, and the lane downgrades
it to ``visual_pending`` loudly (typed telemetry + payload disclosure) —
"never reduce precision silently" enforced by structure, not convention
(realtime_contract.yaml scientific fence ``no_silent_precision_reduction``).

The schema mirrors ``realtime_contract.yaml`` ``result_classes`` requires
(``qualification_domain``, ``error_evidence``, ``exact_base_revision``,
``reconciliation``) and ``compensation_and_exactness.md`` ("Do not fabricate
a per-pixel confidence interval unless it has empirical calibration"):

.. code-block:: python

    payload = {
        "qualifier": {
            "model_version": str,        # non-empty; the model version
            "form": str,                 # non-empty; how to read the delta
            "domain": {
                "families": [str, ...],  # non-empty; families covered
                "variables": [str, ...], # non-empty; planes covered
                "site_scope": str,       # where the error bound was measured
            },
            "error_evidence": {
                "metrics": {var: {...}}, # non-empty; MEASURED holdout error
                "n_holdout": int >= 1,   # held-out edit count behind metrics
                "evidence": str,         # non-empty; evidence pointer
                "calibration_run_id": str,
            },
            "reconciliation": str,       # non-empty; how exact catch-up works
        },
        "exact_base_revision": int >= 0, # the ANCHOR
        "base_planes": ...,              # WHAT base was used, never implicit
    }

The validator checks STRUCTURE only (presence, types, non-emptiness). The
honesty of the numbers is the calibration harness's job — measured error on
HELD-OUT solves, never fitted-on-train numbers. ``site_scope`` and the
compensated kernel's ``domain.variant`` are disclosure fields (carried,
not structurally required): the required core is families + variables,
the error evidence, the anchor, and the base planes.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "QUALIFIER_DEFECTS",
    "qualifier_defects",
]


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_str_seq(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return False
    return len(value) > 0 and all(_nonempty_str(item) for item in value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


#: The bounded defect vocabulary (telemetry label values; never free-form).
QUALIFIER_DEFECTS = (
    "missing_qualifier",
    "invalid_qualifier",
    "missing_model_version",
    "missing_form",
    "missing_domain",
    "missing_domain_variables",
    "missing_error_evidence",
    "invalid_error_evidence",
    "missing_reconciliation",
    "missing_anchor",
    "invalid_anchor",
    "missing_base_planes",
)


def qualifier_defects(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Structural defects of a ``fast_qualified`` payload (empty = valid).

    Deterministic order (check order); every value is drawn from the bounded
    :data:`QUALIFIER_DEFECTS` vocabulary.
    """
    defects: list[str] = []

    qualifier = payload.get("qualifier")
    if qualifier is None:
        return ("missing_qualifier",)
    if not isinstance(qualifier, Mapping):
        return ("invalid_qualifier",)

    if not _nonempty_str(qualifier.get("model_version")):
        defects.append("missing_model_version")
    if not _nonempty_str(qualifier.get("form")):
        defects.append("missing_form")

    domain = qualifier.get("domain")
    if not isinstance(domain, Mapping):
        defects.append("missing_domain")
    else:
        if not _nonempty_str_seq(domain.get("families")):
            defects.append("missing_domain")
        if not _nonempty_str_seq(domain.get("variables")):
            defects.append("missing_domain_variables")

    evidence = qualifier.get("error_evidence")
    if not isinstance(evidence, Mapping):
        defects.append("missing_error_evidence")
    else:
        metrics = evidence.get("metrics")
        n_holdout = evidence.get("n_holdout")
        if (
            not isinstance(metrics, Mapping)
            or len(metrics) == 0
            or not _is_int(n_holdout)
            or n_holdout < 1
            or not _nonempty_str(evidence.get("evidence"))
        ):
            defects.append("invalid_error_evidence")

    if not _nonempty_str(qualifier.get("reconciliation")):
        defects.append("missing_reconciliation")

    anchor = payload.get("exact_base_revision")
    if not _is_int(anchor):
        defects.append("missing_anchor")
    elif anchor < 0:
        defects.append("invalid_anchor")

    base_planes = payload.get("base_planes")
    if isinstance(base_planes, str):
        if not _nonempty_str(base_planes):
            defects.append("missing_base_planes")
    elif isinstance(base_planes, Mapping):
        if len(base_planes) == 0:
            defects.append("missing_base_planes")
    else:
        defects.append("missing_base_planes")

    return tuple(defects)
