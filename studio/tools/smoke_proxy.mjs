// SPDX-License-Identifier: GPL-3.0-only
//
// Smoke test for serve.py --api: POST /edits through the proxy (headers +
// ETag passthrough), binary payload passthrough, and static page serving.
// Usage: node tools/smoke_proxy.mjs

import http from "node:http";
import { spawn } from "node:child_process";
import assert from "node:assert/strict";

const upstreamSeen = [];
const upstream = http.createServer((req, res) => {
  let body = Buffer.alloc(0);
  req.on("data", (chunk) => {
    body = Buffer.concat([body, chunk]);
  });
  req.on("end", () => {
    upstreamSeen.push({
      method: req.method,
      path: req.url,
      idempotencyKey: req.headers["idempotency-key"] ?? null,
      ifMatch: req.headers["if-match"] ?? null,
      authorization: req.headers.authorization ?? null,
      allHeaders: req.headers,
      bodyLength: body.length,
    });
    if (req.url.includes("/payload")) {
      res.writeHead(200, {
        "content-type": "application/vnd.solweig.patch+identity",
        etag: '"sha256-abc"',
        "x-solweig-scene-version": "3",
      });
      res.end(Buffer.from([1, 2, 3, 4, 5, 6, 7, 8]));
      return;
    }
    res.writeHead(202, { "content-type": "application/json", etag: '"scene-version-1"' });
    res.end(JSON.stringify({ ok: true, path: req.url }));
  });
});

await new Promise((resolve) => upstream.listen(8998, "127.0.0.1", resolve));
const server = spawn(
  "python3",
  ["serve.py", "--port", "8999", "--api", "http://127.0.0.1:8998"],
  { stdio: ["ignore", "pipe", "pipe"], cwd: new URL("..", import.meta.url).pathname },
);
await new Promise((resolve) => setTimeout(resolve, 800));

try {
  const edit = await fetch("http://127.0.0.1:8999/api/v1/scenarios/s1/edits", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "Idempotency-Key": "k-123",
      "If-Match": '"scene-version-0"',
      Accept: "application/json",
      Authorization: "Bearer demo-token",
    },
    body: JSON.stringify({ base_scene_version: 0, edits: [] }),
  });
  const editBody = await edit.json();
  assert.equal(edit.status, 202);
  assert.equal(edit.headers.get("etag"), '"scene-version-1"');
  assert.equal(editBody.path, "/api/v1/scenarios/s1/edits");

  const payload = await fetch("http://127.0.0.1:8999/api/v1/scenarios/s1/results/3/payload");
  const bytes = Buffer.from(await payload.arrayBuffer());
  assert.equal(payload.status, 200);
  assert.equal(bytes.length, 8);
  assert.deepEqual([...bytes], [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(payload.headers.get("content-type"), "application/vnd.solweig.patch+identity");
  assert.equal(payload.headers.get("x-solweig-scene-version"), "3");

  const page = await fetch("http://127.0.0.1:8999/");
  assert.equal(page.status, 200);
  assert.equal(page.headers.get("content-type"), "text/html; charset=utf-8");
  const pageText = await page.text();
  assert.match(pageText, /globalThis\.SOLWEIG_API_BASE="\/api"/, "connected mode injects the API default");

  const legacy = await fetch("http://127.0.0.1:8999/examples/incremental_design_tool/index.html", {
    redirect: "manual",
  });
  assert.equal(legacy.status, 301, "legacy path redirects instead of 404ing");
  assert.equal(legacy.headers.get("location"), "/index.html");

  assert.equal(upstreamSeen.length, 2);
  const proxiedEdit = upstreamSeen[0];
  assert.equal(proxiedEdit.method, "POST");
  assert.equal(proxiedEdit.path, "/api/v1/scenarios/s1/edits");
  assert.equal(proxiedEdit.idempotencyKey, "k-123", "Idempotency-Key must survive the proxy");
  assert.equal(proxiedEdit.ifMatch, '"scene-version-0"', "If-Match must survive the proxy");
  assert.equal(proxiedEdit.authorization, "Bearer demo-token", "Authorization must pass through");
  const emptyHeaders = Object.entries(proxiedEdit.allHeaders).filter(
    ([, value]) => value === "",
  );
  assert.deepEqual(emptyHeaders, [], "no empty-string header values may be forwarded");

  console.log("smoke_proxy: OK (edit 202 + etag + auth, payload 8 bytes exact, headers forwarded, injected index served, legacy 301)");
} finally {
  server.kill("SIGTERM");
  upstream.close();
}
