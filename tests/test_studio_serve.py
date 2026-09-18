# SPDX-License-Identifier: GPL-3.0-only
"""serve.py routing tests: root-URL studio, injected connected defaults,
legacy-path 301s, and the /docs mount.

The handler runs in-process (``ThreadingHTTPServer`` on an ephemeral port)
exactly as ``serve.py`` main wires it, so the dispatch order (proxy → legacy
redirect → injected index → static) is exercised over real HTTP without
touching the network beyond loopback.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
STUDIO_DIR = ROOT / "studio"

APP_SCRIPT_MARKER = '<script type="module" src="./app.mjs"></script>'
API_LINE = 'globalThis.SOLWEIG_API_BASE="/api"'
SITE_LINE = 'globalThis.SOLWEIG_SITE_ID="site_500"'


def _load_serve_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("studio_serve", STUDIO_DIR / "serve.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVE = _load_serve_module()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn auto-follow off so 301 Location headers are observable."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        return None


OPENER = urllib.request.build_opener(_NoRedirect)


class _EchoUpstream(http.server.BaseHTTPRequestHandler):
    """Fixed-JSON upstream for proxy checks: echoes path + forwarded Accept."""

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps(
            {"path": self.path, "accept": self.headers.get("Accept")}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Solweig-Test", "passthrough")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # keep pytest output clean


@contextlib.contextmanager
def studio_server(api_origin: str | None = None, site_id: str | None = None) -> Iterator[str]:
    """Run serve.py's handler on an ephemeral port; yield the base URL."""
    handler = SERVE.DemoHandler
    handler.api_origin = api_origin
    handler.site_id = site_id
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        functools.partial(handler, directory=str(STUDIO_DIR)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        handler.api_origin = None
        handler.site_id = None


def _get(base: str, path: str) -> tuple[int, str, dict[str, str]]:
    """GET without redirect-following; HTTP redirects surface as their status."""
    try:
        with OPENER.open(base + path, timeout=5) as response:
            return response.status, response.read().decode("utf-8"), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace"), dict(error.headers)


def test_offline_root_serves_plain_index() -> None:
    with studio_server(api_origin=None) as base:
        status, body, _ = _get(base, "/")
        assert status == 200
        assert "SOLWEIG_API_BASE" not in body
        assert "SOLWEIG_SITE_ID" not in body


def test_connected_root_injects_defaults_before_app_module() -> None:
    # A dummy unreachable origin is enough: index injection happens before
    # any upstream contact (only /api/* would dial upstream).
    with studio_server(api_origin="http://127.0.0.1:1", site_id="site_500") as base:
        status, body, _ = _get(base, "/")
        assert status == 200
        assert API_LINE in body
        assert SITE_LINE in body
        marker_at = body.find(APP_SCRIPT_MARKER)
        assert marker_at >= 0
        assert body.find(API_LINE) < marker_at
        assert body.find(SITE_LINE) < marker_at


def test_connected_root_without_site_omits_site_line() -> None:
    with studio_server(api_origin="http://127.0.0.1:1", site_id=None) as base:
        status, body, _ = _get(base, "/index.html")
        assert status == 200
        assert API_LINE in body
        assert "SOLWEIG_SITE_ID" not in body


def test_legacy_prefix_redirects_with_query_preserved() -> None:
    with studio_server(api_origin="http://127.0.0.1:1", site_id="site_500") as base:
        status, _, headers = _get(base, "/examples/incremental_design_tool/?api=/api")
        assert status == 301
        assert headers.get("Location") == "/?api=/api"


def test_legacy_prefix_redirects_deep_path() -> None:
    with studio_server() as base:
        status, _, headers = _get(base, "/examples/incremental_design_tool/app.mjs")
        assert status == 301
        assert headers.get("Location") == "/app.mjs"


def test_docs_mount_serves_repository_docs() -> None:
    with studio_server() as base:
        status, body, _ = _get(base, "/docs/incremental_design_tool/index.md")
        assert status == 200
        assert "SOLWEIG" in body
        # Only regular files serve: directory URLs must not list contents.
        for directory_path in ("/docs/", "/docs/incremental_design_tool/"):
            status_dir, _, _ = _get(base, directory_path)
            assert status_dir == 404, f"{directory_path} must not serve a listing"


def test_app_module_serves_javascript_mime_type() -> None:
    with studio_server() as base:
        status, _, headers = _get(base, "/app.mjs")
        assert status == 200
        assert headers.get("Content-type") == "text/javascript"


def test_traversal_outside_studio_is_contained() -> None:
    with studio_server() as base:
        for path in ("/../etc/passwd", "/docs/../../pyproject.toml"):
            status, body, _ = _get(base, path)
            assert status in (301, 403, 404), f"{path} leaked with status {status}"
            assert "PASSWD" not in body and "requires-python" not in body


def test_legacy_prefix_near_miss_serves_404() -> None:
    # A longer path sharing the prefix stem must NOT redirect — the redirect
    # matches the exact prefix or prefix + "/".
    with studio_server() as base:
        status, _, _ = _get(base, "/examples/incremental_design_toolish")
        assert status == 404


def test_api_proxy_forwards_through_new_dispatch() -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    try:
        origin = f"http://127.0.0.1:{upstream.server_address[1]}"
        with studio_server(api_origin=origin) as base:
            request = urllib.request.Request(
                base + "/api/v1/ping", headers={"Accept": "application/json"}
            )
            with OPENER.open(request, timeout=5) as response:
                assert response.status == 200
                assert response.headers.get("X-Solweig-Test") == "passthrough"
                payload = json.loads(response.read().decode("utf-8"))
        assert payload["path"] == "/api/v1/ping"
        assert payload["accept"] == "application/json"
    finally:
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
