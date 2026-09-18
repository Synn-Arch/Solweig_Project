#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Serve the SOLWEIG Studio frontend from the repository root URL.

The static root is this directory (``studio/``), so ``/`` is the studio.
With ``--api <base-url>``, requests under ``/api/`` are reverse-proxied to the
SOLWEIG-GPU HTTP API so the browser talks to it same-origin (the API server
does not send CORS headers). Binary patch payloads are streamed through
unchanged. In that connected mode the served ``index.html`` carries an
injected script that presets ``SOLWEIG_API_BASE``/``SOLWEIG_SITE_ID``, so the
demo needs no query parameters; ``?api=`` (empty) still forces offline mode.

Bookmarks from the pre-restructure layout keep working: the old
``/examples/incremental_design_tool/...`` prefix 301-redirects to the same
path at the root (query string preserved). ``/docs/...`` serves the
repository's ``docs/`` tree so the in-app spec link resolves without exposing
the whole repository.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

STUDIO_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDIO_DIR.parent
DOCS_ROOT = REPO_ROOT / "docs"
LEGACY_PREFIX = "/examples/incremental_design_tool"
# Injection anchor: the first app.mjs script tag in index.html.
APP_SCRIPT_MARKER = b'<script type="module" src="./app.mjs"></script>'


class DemoHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".webp": "image/webp",
    }

    # Set by main() when --api is passed: upstream base, e.g. "http://127.0.0.1:8123".
    api_origin: str | None = None
    # Set by main() from SOLWEIG_SITE_ID (first entry): the demo's default site.
    site_id: str | None = None

    def end_headers(self) -> None:
        # Static responses always revalidate: the studio is a fast-moving
        # research prototype, and a heuristic browser cache once masked a
        # deployed app.mjs fix (stale module kept request paths doubled).
        # Proxied /api responses are pass-through and stay untouched.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self._maybe_proxy("GET"):
            return
        if self._redirect_legacy():
            return
        if self._maybe_serve_index("GET"):
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self._maybe_proxy("HEAD"):
            return
        if self._redirect_legacy():
            return
        if self._maybe_serve_index("HEAD"):
            return
        super().do_HEAD()

    def do_POST(self) -> None:  # noqa: N802
        if self._maybe_proxy("POST"):
            return
        self.send_error(http.client.METHOD_NOT_ALLOWED)

    def translate_path(self, path: str) -> str:
        # /docs/<x> maps to the repository's docs tree (the studio's spec
        # link points there), serving REGULAR FILES ONLY — no auto-generated
        # directory listings. A path that resolves outside docs/, or to a
        # directory, falls back to the normal studio-rooted translation,
        # whose own traversal guard (it drops '..' components) answers the
        # attempt with a 404 (nothing under studio/docs/ exists).
        parts = urlsplit(path)
        if parts.path.startswith("/docs/"):
            relative = unquote(parts.path[len("/docs/") :])
            candidate = (DOCS_ROOT / relative).resolve()
            if candidate.is_relative_to(DOCS_ROOT) and candidate.is_file():
                return str(candidate)
        return super().translate_path(path)

    def _redirect_legacy(self) -> bool:
        """301 the pre-restructure URL prefix to the root. Returns True if handled."""
        parts = urlsplit(self.path)
        if parts.path != LEGACY_PREFIX and not parts.path.startswith(LEGACY_PREFIX + "/"):
            return False
        rest = parts.path[len(LEGACY_PREFIX) :]
        if not rest.startswith("/"):
            rest = "/" + rest
        location = rest
        if parts.query:
            # Old bookmarks carry overrides (?api=, ?site=): keep them
            # honored on the redirect target.
            location += "?" + parts.query
        self.send_response(http.client.MOVED_PERMANENTLY)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def _maybe_serve_index(self, method: str) -> bool:
        """Serve index.html with connected-mode defaults injected.

        Only in connected mode (--api given) and only for the studio root:
        a classic script right before the app.mjs module tag presets
        SOLWEIG_API_BASE/SOLWEIG_SITE_ID, so a bare URL boots connected
        without query parameters (classic scripts run before deferred
        modules). Returns False to fall back to plain static serving when
        the injection anchor is missing — the page still works, the caller
        just passes explicit ?api=/api&site=<id> as before.
        """
        if self.api_origin is None:
            return False
        if urlsplit(self.path).path not in ("/", "/index.html"):
            return False
        try:
            page = (STUDIO_DIR / "index.html").read_bytes()
            marker_at = page.find(APP_SCRIPT_MARKER)
        except OSError:
            return False
        if marker_at < 0:
            return False
        site_line = b""
        if self.site_id:
            # Operator-set static config, validated once at startup (main
            # refuses characters that would break out of the script tag).
            site_line = f'globalThis.SOLWEIG_SITE_ID="{self.site_id}";'.encode()
        injection = b'<script>globalThis.SOLWEIG_API_BASE="/api";' + site_line + b"</script>\n"
        body = page[:marker_at] + injection + page[marker_at:]
        self.send_response(http.client.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if method == "GET":
            self.wfile.write(body)
        return True

    def _upstream_headers(self) -> dict[str, str]:
        """Forward request headers that matter to the API contract.

        Only headers actually present are sent (no empty-string values), and
        Authorization is passed through when the browser supplied one.
        """
        headers = {}
        for name in ("Content-Type", "Idempotency-Key", "If-Match", "Accept", "Authorization"):
            value = self.headers.get(name)
            if value:
                headers[name] = value
        return headers

    def _maybe_proxy(self, method: str) -> bool:
        """Forward /api/* to the upstream API server. Returns True if handled."""
        if self.api_origin is None or not self.path.startswith("/api/"):
            return False
        upstream = urlsplit(self.api_origin)
        # Set BEFORE the try: a cold upstream (connection refused) raises in
        # connection.request before any header is written, and the except
        # branch must find the flag bound (an unbound name there turned every
        # proxied GET into a 500 during API warm-up on the demo host).
        headers_committed = False
        try:
            body = None
            length_header = self.headers.get("Content-Length")
            if length_header is not None:
                body = self.rfile.read(int(length_header))
            connection = http.client.HTTPConnection(
                upstream.hostname,
                upstream.port or 80,
                timeout=60,
            )
            upstream_path = self.path
            if upstream.path and upstream.path != "/":
                upstream_path = upstream.path.rstrip("/") + self.path
            connection.request(
                method,
                upstream_path,
                body=body,
                headers=self._upstream_headers(),
            )
            response = connection.getresponse()
            # SSE responses must stream chunk-by-chunk: buffering the whole
            # body (the old response.read()) waits for an EOF that a live
            # event stream never sends, so the browser sees a hung request
            # and the realtime subscription fails through the proxy.
            is_event_stream = (response.getheader("content-type") or "").startswith(
                "text/event-stream"
            )
            # Once end_headers() runs, the status line + headers are on the
            # wire: a later send_error would append a SECOND status line into
            # what the browser already parsed as a body (corrupted stream).
            self.send_response(response.status)
            # Content-Encoding is dropped deliberately: http.client already
            # decoded the upstream body, so the bytes we re-send are plain
            # and must not carry a stale gzip/zstd label.
            hop_by_hop = {"connection", "transfer-encoding", "content-encoding", "content-length", "server", "date"}
            for name, value in response.getheaders():
                if name.lower() not in hop_by_hop:
                    self.send_header(name, value)
            if is_event_stream:
                # No Content-Length: HTTP/1.0 close-framing while chunks stream.
                self.end_headers()
                headers_committed = True
                if method != "HEAD":
                    while True:
                        # read1: at most ONE read on the raw socket, so a
                        # partial SSE frame is forwarded immediately instead
                        # of buffering until 64 KiB or EOF (which a live
                        # event stream never reaches).
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            else:
                payload = response.read()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                headers_committed = True
                if method != "HEAD":
                    self.wfile.write(payload)
            connection.close()
        except (OSError, http.client.HTTPException) as error:
            if headers_committed:
                # Mid-stream upstream failure: log + close only. The abrupt
                # close (HTTP/1.0 close-framing) is the honest signal — the
                # browser's EventSource reconnects on its own schedule.
                self.log_error("API upstream failed mid-response: %s", error)
                self.close_connection = True
            else:
                self.send_error(http.client.BAD_GATEWAY, f"API upstream unreachable: {error}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--api",
        default=None,
        metavar="BASE_URL",
        help="Reverse-proxy /api/* to the SOLWEIG-GPU API (e.g. http://127.0.0.1:8123) "
        "so the browser talks to it same-origin.",
    )
    arguments = parser.parse_args()
    if arguments.api:
        DemoHandler.api_origin = arguments.api
    # The API exports SOLWEIG_SITE_ID (comma-separated for multi-site); the
    # studio's connected-mode default is the first entry. Query parameters
    # still override the injected default. The id is interpolated verbatim
    # into served HTML, so unsafe characters are a startup error, not an
    # injection-time assert (which `python -O` would strip).
    site_id = os.environ.get("SOLWEIG_SITE_ID", "").split(",")[0] or None
    if site_id and any(character in site_id for character in '"<>\\\n'):
        raise SystemExit(
            f"invalid SOLWEIG_SITE_ID entry {site_id!r}: the studio injects it "
            "into served HTML, so quotes, angle brackets, backslashes and "
            "newlines are not allowed"
        )
    DemoHandler.site_id = site_id
    handler = lambda *args, **kwargs: DemoHandler(  # noqa: E731
        *args,
        directory=str(STUDIO_DIR),
        **kwargs,
    )
    server = http.server.ThreadingHTTPServer((arguments.host, arguments.port), handler)
    api_note = f" -> API {arguments.api}" if arguments.api else ""
    print(f"Serving SOLWEIG Studio at http://{arguments.host}:{arguments.port}/{api_note}")
    server.serve_forever()
