# SPDX-License-Identifier: GPL-3.0-only
"""``python -m solweig_gpu.server`` — environment-driven boot entrypoint.

Maps the deployment ``SOLWEIG_*`` environment variables onto
:func:`solweig_gpu.server.app.create_app` keyword arguments and runs the
resulting application under uvicorn. This is the single-process API entry
of the two-process research deployment (API + worker thread in this
process, static frontend + same-origin proxy served separately by
``studio/serve.py``).

Recognised variables (everything else starting with ``SOLWEIG_`` produces a
stderr warning and is ignored):

===================================  =======================================
Variable                             Meaning
===================================  =======================================
``SOLWEIG_STATE_ROOT``               State directory (default ``./state``).
``SOLWEIG_SITE_ID``                  Required; comma-separated site ids.
``SOLWEIG_CACHE_ROOT``               Required parent of ``<site_id>/`` caches.
``SOLWEIG_SITE_DIR_<ID>``            Optional per-site ``site_dir``.
``SOLWEIG_SELECTED_DATE_STR_<ID>``   Optional per-site simulation date.
``SOLWEIG_HOST``                     uvicorn host (default ``0.0.0.0``).
``SOLWEIG_PORT``                     uvicorn port (default ``8000``).
``SOLWEIG_MAX_TREES_PER_SCENARIO``   ``max_trees_per_scenario``.
``SOLWEIG_COALESCE_MS``              ``coalescing_window_ms``.
``SOLWEIG_RT_ADAPTIVE_COALESCE``     T24 (§710) adaptive debounce: shrink
                                    the coalescing wait to the 50 ms
                                    contract floor for isolated jobs.
                                    Unset keeps the fixed window.
``SOLWEIG_REQUESTS_PER_MINUTE``      ``requests_per_minute_per_ip``.
``SOLWEIG_EDITS_PER_MINUTE``         ``edits_per_minute``.
``SOLWEIG_SHARED_SCENARIO_ID``       Shared-world scenario id (default
                                    ``scn_shared_world``). Always wired:
                                    every visitor joins this ONE workspace
                                    instead of minting a per-session
                                    scenario.
===================================  =======================================

``<ID>`` is the site id upper-cased with every non-alphanumeric character
replaced by an underscore (``campus-1km-v1`` → ``CAMPUS_1KM_V1``).

Invalid values refuse to boot with a clear message and exit code 2; unset
optional variables keep ``create_app``'s defaults. The heavy solver stack
is imported only after the configuration validates.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from solweig_gpu.server.models import SHARED_SCENARIO_ID_RE

__all__ = ["BootConfigError", "app_from_env", "main"]

#: Site-independent variables this entrypoint understands.
KNOWN_ENV_VARS = frozenset(
    {
        "SOLWEIG_STATE_ROOT",
        "SOLWEIG_SITE_ID",
        "SOLWEIG_CACHE_ROOT",
        "SOLWEIG_HOST",
        "SOLWEIG_PORT",
        "SOLWEIG_MAX_TREES_PER_SCENARIO",
        "SOLWEIG_COALESCE_MS",
        "SOLWEIG_RT_ADAPTIVE_COALESCE",
        "SOLWEIG_REQUESTS_PER_MINUTE",
        "SOLWEIG_EDITS_PER_MINUTE",
        "SOLWEIG_SHARED_SCENARIO_ID",
    }
)

#: Per-site variable prefixes (``<PREFIX><SITE_ID_WITH_UNDERSCORES>``).
SITE_DIR_PREFIX = "SOLWEIG_SITE_DIR_"
SELECTED_DATE_PREFIX = "SOLWEIG_SELECTED_DATE_STR_"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_STATE_ROOT = "./state"

#: The shared world every visitor joins (single-canvas deployment). The id
#: follows the minted ``scn_<segment>`` convention and doubles as a path
#: segment under the state root (``scenarios/<id>/``), so it must stay a
#: safe single path component.
DEFAULT_SHARED_SCENARIO_ID = "scn_shared_world"


class BootConfigError(Exception):
    """Invalid ``SOLWEIG_*`` configuration; boot refuses with exit code 2."""


def site_env_suffix(site_id: str) -> str:
    """The env-var suffix for ``site_id`` (``campus-1km-v1`` → ``CAMPUS_1KM_V1``)."""
    return re.sub(r"[^A-Za-z0-9]", "_", site_id).upper()


def _raw(environ: Mapping[str, str], name: str) -> str | None:
    """The variable's value, refusing empty strings (unset keeps the default)."""
    value = environ.get(name)
    if value is None:
        return None
    if not value.strip():
        raise BootConfigError(
            f"{name} is empty; either unset it or give it a valid value"
        )
    return value


def _int_value(environ: Mapping[str, str], name: str) -> int | None:
    raw = _raw(environ, name)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        raise BootConfigError(f"{name} must be an integer, got {raw!r}") from None


def _nonnegative_int(environ: Mapping[str, str], name: str) -> int | None:
    value = _int_value(environ, name)
    if value is not None and value < 0:
        raise BootConfigError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
    return value


def _milliseconds(environ: Mapping[str, str], name: str) -> float | None:
    raw = _raw(environ, name)
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        raise BootConfigError(
            f"{name} must be a number of milliseconds, got {raw!r}"
        ) from None
    if not math.isfinite(value) or value < 0:
        raise BootConfigError(
            f"{name} must be a finite non-negative number of milliseconds, got {raw!r}"
        )
    return value


def _shared_scenario_id(environ: Mapping[str, str]) -> str:
    """The shared-world scenario id (default ``scn_shared_world``).

    Always returns an id — the shared world is not optional for this
    entrypoint — but an operator-supplied value must follow the scenario-id
    convention (``scn_`` prefix, safe single path segment) or boot refuses.
    """
    raw = _raw(environ, "SOLWEIG_SHARED_SCENARIO_ID")
    if raw is None:
        return DEFAULT_SHARED_SCENARIO_ID
    value = raw.strip()
    if not SHARED_SCENARIO_ID_RE.fullmatch(value):
        raise BootConfigError(
            f"SOLWEIG_SHARED_SCENARIO_ID must be a scenario id like "
            f"{DEFAULT_SHARED_SCENARIO_ID!r} (scn_ prefix, then lowercase "
            f"letters, digits, underscores or hyphens), got {raw!r}"
        )
    return value


def _warn(message: str, stderr: TextIO) -> None:
    print(f"warning: {message}", file=stderr)


def _check_unknown_vars(
    environ: Mapping[str, str], site_suffixes: set[str], stderr: TextIO
) -> None:
    """Warn (never fail) about ``SOLWEIG_*`` variables nothing will read."""
    for key in sorted(environ):
        if not isinstance(key, str) or not key.startswith("SOLWEIG_"):
            continue
        if key in KNOWN_ENV_VARS:
            continue
        if key.startswith(SITE_DIR_PREFIX):
            suffix = key[len(SITE_DIR_PREFIX) :]
            if suffix in site_suffixes:
                continue
            _warn(
                f"ignoring {key}: suffix {suffix!r} does not match any "
                f"SOLWEIG_SITE_ID entry",
                stderr,
            )
            continue
        if key.startswith(SELECTED_DATE_PREFIX):
            suffix = key[len(SELECTED_DATE_PREFIX) :]
            if suffix in site_suffixes:
                continue
            _warn(
                f"ignoring {key}: suffix {suffix!r} does not match any "
                f"SOLWEIG_SITE_ID entry",
                stderr,
            )
            continue
        _warn(f"ignoring unknown SOLWEIG environment variable {key!r}", stderr)


def _sites(
    environ: Mapping[str, str], site_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Build the ``create_app(sites=...)`` mapping (``None`` values omitted)."""
    cache_root_raw = _raw(environ, "SOLWEIG_CACHE_ROOT")
    if cache_root_raw is None:
        raise BootConfigError(
            "SOLWEIG_CACHE_ROOT is required when SOLWEIG_SITE_ID is set: point it "
            "at the parent directory holding one <site_id>/ cache per site"
        )
    cache_root = Path(cache_root_raw)
    sites: dict[str, dict[str, Any]] = {}
    for site_id in site_ids:
        # A site id must be a single safe path SEGMENT: an absolute
        # component ("/etc") or a traversal component ("..", "a/../..")
        # would resolve the cache_dir OUTSIDE SOLWEIG_CACHE_ROOT (pathlib
        # joins absolute components by replacement), silently serving a
        # directory the operator never named.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", site_id):
            raise BootConfigError(
                f"SOLWEIG_SITE_ID entry {site_id!r} is not a safe site id: use "
                "letters, digits, dots, underscores or hyphens, starting with "
                "a letter or digit (an absolute or '..' component would "
                "resolve the cache directory outside SOLWEIG_CACHE_ROOT)"
            )
        entry: dict[str, Any] = {"cache_dir": cache_root / site_id}
        suffix = site_env_suffix(site_id)
        site_dir = _raw(environ, SITE_DIR_PREFIX + suffix)
        if site_dir is not None:
            entry["site_dir"] = Path(site_dir)
        selected_date = _raw(environ, SELECTED_DATE_PREFIX + suffix)
        if selected_date is not None:
            entry["selected_date_str"] = selected_date
        sites[site_id] = entry
    return sites


def _site_ids(environ: Mapping[str, str]) -> list[str]:
    raw = environ.get("SOLWEIG_SITE_ID")
    if raw is None or not raw.strip():
        raise BootConfigError(
            "SOLWEIG_SITE_ID is not set: the two-process research deployment needs "
            "at least one site to serve edits from. Set SOLWEIG_SITE_ID "
            "(comma-separated for multiple sites) and SOLWEIG_CACHE_ROOT."
        )
    site_ids = [part.strip() for part in raw.split(",")]
    if not all(site_ids):
        raise BootConfigError(
            f"SOLWEIG_SITE_ID contains an empty site id: {raw!r}"
        )
    return site_ids


def app_from_env(
    environ: Mapping[str, str],
    *,
    app_factory: Callable[..., Any] | None = None,
    stderr: TextIO | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the API app from ``SOLWEIG_*`` environment variables.

    Returns ``(app, uvicorn_kwargs)`` where ``uvicorn_kwargs`` holds the
    ``host``/``port`` for :func:`uvicorn.run`. ``app_factory`` defaults to
    :func:`solweig_gpu.server.app.create_app` (imported only after the
    configuration validates); tests inject a recording fake instead.

    Raises :class:`BootConfigError` naming the variable and value for every
    invalid setting; unknown ``SOLWEIG_*`` variables only warn on ``stderr``.
    """
    if stderr is None:
        stderr = sys.stderr

    site_ids = _site_ids(environ)
    site_suffixes = {site_env_suffix(site_id) for site_id in site_ids}
    _check_unknown_vars(environ, site_suffixes, stderr)

    app_kwargs: dict[str, Any] = {
        "state_root": Path(
            _raw(environ, "SOLWEIG_STATE_ROOT") or DEFAULT_STATE_ROOT
        ),
        "sites": _sites(environ, site_ids),
        # Always on for this entrypoint: the shared world is created at boot
        # so every visitor joins one workspace (id overridable per deploy).
        "shared_scenario_id": _shared_scenario_id(environ),
    }

    max_trees = _int_value(environ, "SOLWEIG_MAX_TREES_PER_SCENARIO")
    if max_trees is not None:
        if max_trees < 1:
            raise BootConfigError(
                "SOLWEIG_MAX_TREES_PER_SCENARIO must be a positive integer "
                f"(at least 1), got {max_trees!r}"
            )
        app_kwargs["max_trees_per_scenario"] = max_trees

    coalesce_ms = _milliseconds(environ, "SOLWEIG_COALESCE_MS")
    if coalesce_ms is not None:
        app_kwargs["coalescing_window_ms"] = coalesce_ms

    edits_per_minute = _nonnegative_int(environ, "SOLWEIG_EDITS_PER_MINUTE")
    if edits_per_minute is not None:
        app_kwargs["edits_per_minute"] = edits_per_minute

    requests_per_minute = _nonnegative_int(environ, "SOLWEIG_REQUESTS_PER_MINUTE")
    if requests_per_minute is not None:
        app_kwargs["requests_per_minute_per_ip"] = requests_per_minute

    host = _raw(environ, "SOLWEIG_HOST") or DEFAULT_HOST
    port = _int_value(environ, "SOLWEIG_PORT")
    if port is None:
        port = DEFAULT_PORT
    elif not 0 <= port <= 65535:
        raise BootConfigError(
            f"SOLWEIG_PORT must be an integer between 0 and 65535, got {port!r}"
        )

    if app_factory is None:
        from solweig_gpu.server.app import create_app as app_factory

    app = app_factory(**app_kwargs)
    return app, {"host": host, "port": port}


def main() -> int:
    """Boot the API from the process environment; exit 2 on bad configuration."""
    try:
        app, uvicorn_kwargs = app_from_env(os.environ)
    except BootConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    import uvicorn

    uvicorn.run(app, **uvicorn_kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
