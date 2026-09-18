# SPDX-License-Identifier: GPL-3.0-only
"""Environment-driven boot entrypoint tests (``python -m solweig_gpu.server``).

The unit tests exercise the pure builder ``app_from_env`` with a recording
fake app factory, so no server (and no solver stack) is booted: they pin the
exact env-var -> ``create_app`` kwarg mapping, the defaults, multi-site
parsing, per-site overrides, and the fail-fast (exit 2) error paths.

The ``deployment``-marked test boots the REAL module in a subprocess against
a synthetic site cache on an ephemeral port and drives /health/live,
/health/ready, and /api/v1/capabilities over real HTTP.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="server extra (fastapi) not installed")
pytest.importorskip("uvicorn", reason="server extra (uvicorn) not installed")

from solweig_gpu.server.__main__ import BootConfigError, app_from_env, main  # noqa: E402

SITE_ID = "test-site"

#: A sentinel distinct from any FastAPI instance: the fake factory returns it
#: so tests can assert the builder handed THEIR app object back.
SENTINEL_APP = object()


def make_factory(calls: list[dict[str, Any]]):
    """An app factory that records kwargs and returns a sentinel."""

    def factory(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SENTINEL_APP

    return factory


def base_env(**overrides: str) -> dict[str, str]:
    """A minimal valid environment (one site, no optional knobs)."""
    env = {
        "SOLWEIG_SITE_ID": SITE_ID,
        "SOLWEIG_CACHE_ROOT": "/srv/site-cache",
    }
    env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# Mapping: env -> create_app kwargs + uvicorn kwargs
# ---------------------------------------------------------------------------


class TestEnvMapping:
    def test_missing_site_id_is_fatal(self) -> None:
        calls: list[dict[str, Any]] = []
        with pytest.raises(BootConfigError) as excinfo:
            app_from_env({}, app_factory=make_factory(calls))
        assert "SOLWEIG_SITE_ID" in str(excinfo.value)
        # The refusal must explain why a site is mandatory.
        assert "site" in str(excinfo.value).lower()
        assert calls == [], "no app may be built when configuration is invalid"

    def test_missing_cache_root_is_fatal(self) -> None:
        calls: list[dict[str, Any]] = []
        with pytest.raises(BootConfigError) as excinfo:
            app_from_env({"SOLWEIG_SITE_ID": SITE_ID}, app_factory=make_factory(calls))
        message = str(excinfo.value)
        assert "SOLWEIG_SITE_ID" in message and "SOLWEIG_CACHE_ROOT" in message
        assert calls == []

    def test_defaults(self) -> None:
        calls: list[dict[str, Any]] = []
        app, uvicorn_kwargs = app_from_env(base_env(), app_factory=make_factory(calls))
        assert app is SENTINEL_APP
        # Only the required kwargs plus the always-on shared world are passed
        # when no optional env var is set: create_app's own defaults
        # (coalescing 500 ms, 200 trees, rate limits 30/20 per minute) remain
        # the single source of truth.
        assert calls == [
            {
                "state_root": Path("./state"),
                "sites": {SITE_ID: {"cache_dir": Path("/srv/site-cache") / SITE_ID}},
                "shared_scenario_id": "scn_shared_world",
            }
        ]
        assert uvicorn_kwargs == {"host": "0.0.0.0", "port": 8000}

    def test_full_mapping(self) -> None:
        calls: list[dict[str, Any]] = []
        app, uvicorn_kwargs = app_from_env(
            base_env(
                SOLWEIG_STATE_ROOT="/var/lib/solweig",
                SOLWEIG_HOST="127.0.0.1",
                SOLWEIG_PORT="9443",
                SOLWEIG_MAX_TREES_PER_SCENARIO="75",
                SOLWEIG_COALESCE_MS="250",
                SOLWEIG_EDITS_PER_MINUTE="10",
                SOLWEIG_REQUESTS_PER_MINUTE="60",
                SOLWEIG_SHARED_SCENARIO_ID="scn_demo_world",
            ),
            app_factory=make_factory(calls),
        )
        assert app is SENTINEL_APP
        assert calls == [
            {
                "state_root": Path("/var/lib/solweig"),
                "sites": {SITE_ID: {"cache_dir": Path("/srv/site-cache") / SITE_ID}},
                "max_trees_per_scenario": 75,
                "coalescing_window_ms": 250.0,
                "edits_per_minute": 10,
                "requests_per_minute_per_ip": 60,
                "shared_scenario_id": "scn_demo_world",
            }
        ]
        assert uvicorn_kwargs == {"host": "127.0.0.1", "port": 9443}

    def test_state_root_empty_is_fatal(self) -> None:
        calls: list[dict[str, Any]] = []
        with pytest.raises(BootConfigError) as excinfo:
            app_from_env(
                base_env(SOLWEIG_STATE_ROOT=""), app_factory=make_factory(calls)
            )
        assert "SOLWEIG_STATE_ROOT" in str(excinfo.value)
        assert calls == []


# ---------------------------------------------------------------------------
# Sites: multi-site lists and per-site overrides
# ---------------------------------------------------------------------------


class TestSiteParsing:
    def test_comma_separated_multi_site(self) -> None:
        calls: list[dict[str, Any]] = []
        app_from_env(
            {
                "SOLWEIG_SITE_ID": "alpha, beta-gamma ,delta",
                "SOLWEIG_CACHE_ROOT": "/srv/cache",
            },
            app_factory=make_factory(calls),
        )
        sites = calls[0]["sites"]
        assert set(sites) == {"alpha", "beta-gamma", "delta"}
        assert sites["beta-gamma"]["cache_dir"] == Path("/srv/cache/beta-gamma")

    def test_empty_site_segment_is_fatal(self) -> None:
        calls: list[dict[str, Any]] = []
        with pytest.raises(BootConfigError) as excinfo:
            app_from_env(
                {
                    "SOLWEIG_SITE_ID": "alpha,,delta",
                    "SOLWEIG_CACHE_ROOT": "/srv/cache",
                },
                app_factory=make_factory(calls),
            )
        assert "SOLWEIG_SITE_ID" in str(excinfo.value)
        assert calls == []

    def test_per_site_overrides_use_underscored_names(self) -> None:
        """Hyphenated site ids map to underscored upper-case env suffixes."""
        calls: list[dict[str, Any]] = []
        app_from_env(
            {
                "SOLWEIG_SITE_ID": "campus-1km-v1,plain",
                "SOLWEIG_CACHE_ROOT": "/srv/cache",
                "SOLWEIG_SITE_DIR_CAMPUS_1KM_V1": "/srv/site-dirs/campus",
                "SOLWEIG_SELECTED_DATE_STR_PLAIN": "2015-07-01",
            },
            app_factory=make_factory(calls),
        )
        sites = calls[0]["sites"]
        # Override present where the env var exists...
        assert sites["campus-1km-v1"] == {
            "cache_dir": Path("/srv/cache/campus-1km-v1"),
            "site_dir": Path("/srv/site-dirs/campus"),
        }
        assert sites["plain"] == {
            "cache_dir": Path("/srv/cache/plain"),
            "selected_date_str": "2015-07-01",
        }

    def test_empty_per_site_override_is_fatal(self) -> None:
        calls: list[dict[str, Any]] = []
        with pytest.raises(BootConfigError) as excinfo:
            app_from_env(
                base_env(SOLWEIG_SITE_DIR_TEST_SITE=""),
                app_factory=make_factory(calls),
            )
        assert "SOLWEIG_SITE_DIR_TEST_SITE" in str(excinfo.value)
        assert calls == []

    @pytest.mark.parametrize(
        "site_id",
        [
            "/etc",  # absolute component: Path("/srv") / "/etc" == Path("/etc")
            "a/../../b",  # traversal above CACHE_ROOT
            "..",
            ".",
            "a b",  # whitespace inside a segment
            "a\nb",
        ],
    )
    def test_unsafe_site_id_is_fatal(self, site_id: str) -> None:
        """u-dep1-verify NIT-1: a site id with an absolute component or a
        traversal component would RESOLVE the cache_dir outside
        SOLWEIG_CACHE_ROOT (pathlib joins absolute components by
        replacement) — the boot must refuse instead of silently serving a
        cache directory the operator never named."""
        calls: list[dict[str, Any]] = []
        with pytest.raises(BootConfigError) as excinfo:
            app_from_env(
                {
                    "SOLWEIG_SITE_ID": site_id,
                    "SOLWEIG_CACHE_ROOT": "/srv/cache",
                },
                app_factory=make_factory(calls),
            )
        assert "SOLWEIG_SITE_ID" in str(excinfo.value)
        # The message carries the id repr'd, so control characters
        # (newline) appear escaped rather than raw.
        assert repr(site_id) in str(excinfo.value)
        assert calls == []


# ---------------------------------------------------------------------------
# Fail-fast on invalid values; warn (never fail) on unknown vars
# ---------------------------------------------------------------------------


class TestValueValidation:
    @pytest.mark.parametrize(
        ("envar", "value"),
        [
            ("SOLWEIG_PORT", "not-a-port"),
            ("SOLWEIG_PORT", "70000"),
            ("SOLWEIG_PORT", "-1"),
            ("SOLWEIG_MAX_TREES_PER_SCENARIO", "many"),
            ("SOLWEIG_MAX_TREES_PER_SCENARIO", "0"),
            ("SOLWEIG_COALESCE_MS", "soon"),
            ("SOLWEIG_COALESCE_MS", "-5"),
            ("SOLWEIG_EDITS_PER_MINUTE", "fast"),
            ("SOLWEIG_EDITS_PER_MINUTE", "-1"),
            ("SOLWEIG_REQUESTS_PER_MINUTE", "unlimited"),
            ("SOLWEIG_SHARED_SCENARIO_ID", "shared_world"),
            ("SOLWEIG_SHARED_SCENARIO_ID", "scn_"),
            ("SOLWEIG_SHARED_SCENARIO_ID", "SCN_SHARED_WORLD"),
            ("SOLWEIG_SHARED_SCENARIO_ID", "scn_shared world"),
        ],
    )
    def test_invalid_values_fail_fast(self, envar: str, value: str) -> None:
        calls: list[dict[str, Any]] = []
        with pytest.raises(BootConfigError) as excinfo:
            app_from_env(
                base_env(**{envar: value}), app_factory=make_factory(calls)
            )
        # The error must name the variable and the offending value.
        message = str(excinfo.value)
        assert envar in message
        assert value in message
        assert calls == []

    def test_zero_rate_limit_is_an_integer_not_a_keyword(self) -> None:
        """0 passes through as the integer 0 (create_app's falsy semantics
        disable the limiter); 'none' is NOT a recognized disable keyword."""
        calls: list[dict[str, Any]] = []
        app_from_env(
            base_env(SOLWEIG_EDITS_PER_MINUTE="0"),
            app_factory=make_factory(calls),
        )
        assert calls[0]["edits_per_minute"] == 0

        with pytest.raises(BootConfigError):
            app_from_env(
                base_env(SOLWEIG_EDITS_PER_MINUTE="none"),
                app_factory=make_factory(calls),
            )

    def test_unknown_solweig_var_warns_but_boots(self, capsys: pytest.CaptureFixture) -> None:
        """Legacy doc-only variables (e.g. SOLWEIG_FULL_RECOMPUTE_FRACTION)
        produce a stderr warning and are otherwise ignored."""
        calls: list[dict[str, Any]] = []
        app, _ = app_from_env(
            base_env(SOLWEIG_FULL_RECOMPUTE_FRACTION="0.30"),
            app_factory=make_factory(calls),
        )
        assert app is SENTINEL_APP and len(calls) == 1
        stderr = capsys.readouterr().err
        assert "SOLWEIG_FULL_RECOMPUTE_FRACTION" in stderr
        assert "unknown" in stderr.lower()

    def test_override_for_unlisted_site_warns_but_boots(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        calls: list[dict[str, Any]] = []
        app_from_env(
            base_env(SOLWEIG_SITE_DIR_GHOST_SITE="/srv/ghost"),
            app_factory=make_factory(calls),
        )
        assert len(calls) == 1
        assert "sites" in calls[0]
        stderr = capsys.readouterr().err
        assert "SOLWEIG_SITE_DIR_GHOST_SITE" in stderr
        assert "SOLWEIG_SITE_ID" in stderr


class TestSharedScenarioIdConvention:
    def test_one_regex_definition_backs_both_entrypoints(self) -> None:
        """The shared-world id convention lives in ONE place (the models
        layer both enforcement sites sit on): ``create_app``'s
        direct-caller validation and the env validator must share the
        same compiled pattern, not twin copies that can drift apart."""
        from solweig_gpu.server import __main__ as server_main
        from solweig_gpu.server import app as server_app
        from solweig_gpu.server import models as server_models

        canonical = server_models.SHARED_SCENARIO_ID_RE
        assert canonical is server_app.SHARED_SCENARIO_ID_RE
        assert canonical is server_main.SHARED_SCENARIO_ID_RE


# ---------------------------------------------------------------------------
# main(): exit-code contract (no server booted)
# ---------------------------------------------------------------------------


class TestMainExitCodes:
    def test_invalid_config_exits_2_with_message(
        self,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(os, "environ", {})
        assert main() == 2
        stderr = capsys.readouterr().err
        assert "SOLWEIG_SITE_ID" in stderr

    def test_valid_config_reaches_uvicorn(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorded: dict[str, Any] = {}

        def fake_run(app: Any, **kwargs: Any) -> None:
            recorded["app"] = app
            recorded.update(kwargs)

        # Real create_app with a hermetic state root (no ./state pollution).
        monkeypatch.setattr(
            os, "environ", base_env(SOLWEIG_STATE_ROOT=str(tmp_path / "state"))
        )
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", fake_run)
        assert main() == 0
        assert recorded["host"] == "0.0.0.0"
        assert recorded["port"] == 8000
        # main() hands uvicorn a REAL create_app application, not a stub.
        from fastapi import FastAPI

        assert isinstance(recorded["app"], FastAPI)


# ---------------------------------------------------------------------------
# Real boot: the module runs a live server over HTTP (marked ``deployment``)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(base_url: str, path: str) -> tuple[int, Any]:
    request = urllib.request.Request(base_url + path)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # pragma: no cover - diagnostics
        return error.code, error.read().decode("utf-8", "replace")


@pytest.mark.deployment
def test_module_boots_and_serves_over_http(tmp_path: Path) -> None:
    """Boot ``python -m solweig_gpu.server`` for real and drive the ops +
    capability surface over HTTP against a synthetic site cache."""
    from tests.test_server_api import SITE_ID as CACHE_SITE_ID, make_site_cache

    assert CACHE_SITE_ID == SITE_ID
    cache_root = tmp_path / "cache"
    make_site_cache(tmp_path)  # creates <tmp>/cache/<SITE_ID>

    port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "SOLWEIG_STATE_ROOT": str(tmp_path / "state"),
            "SOLWEIG_SITE_ID": SITE_ID,
            "SOLWEIG_CACHE_ROOT": str(cache_root),
            "SOLWEIG_HOST": "127.0.0.1",
            "SOLWEIG_PORT": str(port),
        }
    )
    repo_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "solweig_gpu.server"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        # First boot imports the solver stack (torch et al.): allow minutes.
        deadline = time.monotonic() + 180.0
        live: tuple[int, Any] | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read().decode("utf-8", "replace")
                raise AssertionError(f"server exited early (rc={process.returncode}):\n{stderr}")
            try:
                live = _get(base_url, "/health/live")
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        assert live is not None, "server never answered /health/live"
        assert live[0] == 200

        status, ready = _get(base_url, "/health/ready")
        assert status == 200, f"/health/ready not ready: {ready}"

        status, capabilities = _get(base_url, "/api/v1/capabilities")
        assert status == 200
        assert isinstance(capabilities, dict)
        assert capabilities.get("adapters"), capabilities.keys()
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
            process.wait(timeout=10)
