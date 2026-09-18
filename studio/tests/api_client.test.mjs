// SPDX-License-Identifier: GPL-3.0-only
//
// Adapter-contract tests: a fake fetch serves canned contract responses from
// docs/incremental_design_tool/api_contract.md; no server is involved.

import test from "node:test";
import assert from "node:assert/strict";

import {
  ApiClient,
  ApiClientError,
  PATCH_MEDIA_TYPE_IDENTITY,
  PATCH_MEDIA_TYPE_ZSTD,
  parseRetryAfterMs,
  resolveApiBase,
  zstdStreamSupported,
} from "../api_client.mjs";

const SCENARIO = "scn_test_01";
const SITE_ROWS = 8;
const SITE_COLS = 8;

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function jsonResponse(status, body, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

/**
 * Build a contract manifest + payload pair. Values are provided per variable
 * already flattened in wire order (variable-major, time, row, column).
 */
async function buildResult({
  sceneVersion,
  window = { row_start: 0, row_stop: SITE_ROWS, col_start: 0, col_stop: SITE_COLS },
  timeIndices = [0],
  variables = ["utci"],
  values,
  schemaVersion = 1,
  compression = "identity",
  limitations = ["Tree-induced local wind-field changes are not recomputed."],
}) {
  const rows = window.row_stop - window.row_start;
  const cols = window.col_stop - window.col_start;
  const chunks = [];
  const variableEntries = [];
  for (const name of variables) {
    variableEntries.push({
      name,
      dtype: "float32",
      shape: [timeIndices.length, rows, cols],
      nodata: "nan",
    });
    chunks.push(new Uint8Array(values[name].buffer.slice(0)));
  }
  const payloadBytes = new Uint8Array(chunks.reduce((a, c) => a + c.byteLength, 0));
  let offset = 0;
  for (const chunk of chunks) {
    payloadBytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const hex = await sha256Hex(payloadBytes);
  const manifest = {
    schema_version: schemaVersion,
    scenario_id: SCENARIO,
    scene_version: sceneVersion,
    exact: true,
    model_version: "solweig-gpu-2.0.0+incremental.1",
    site_cache_version: "campus-1km-v1:cache-1",
    window,
    time_indices: timeIndices,
    variables: variableEntries,
    payload_url: `/api/v1/scenarios/${SCENARIO}/results/${sceneVersion}/payload`,
    compression,
    checksum: `sha256:${hex}`,
    metrics: { mean_utci_delta_c: -0.84, peak_utci_delta_c: -3.42, improved_area_m2: 8940 },
    limitations,
  };
  return { manifest, payloadBytes, etag: `"sha256-${hex}"` };
}

function planeValue(time, row, col, salt = 0) {
  return 1000 * time + 10 * row + col + 0.5 + salt;
}

function makeVariable(variables, timeCount, rows, cols, salt = 0) {
  const out = {};
  for (const name of variables) {
    const array = new Float32Array(timeCount * rows * cols);
    let index = 0;
    for (let time = 0; time < timeCount; time += 1) {
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          array[index] = planeValue(time, row, col, salt);
          index += 1;
        }
      }
    }
    out[name] = array;
  }
  return out;
}

test("createScenario sends idempotency key and contract body", async () => {
  const calls = [];
  const client = new ApiClient({
    baseUrl: "http://api.test",
    sessionId: "sess-1",
    fetch: async (url, init) => {
      calls.push({ url, init });
      return jsonResponse(201, {
        scenario_id: SCENARIO,
        site_id: "campus-1km-v1",
        scene_version: 0,
        exact_result_version: 0,
        status: "exact",
        trees: [],
        result_manifest_url: `/api/v1/scenarios/${SCENARIO}/results/0`,
      });
    },
  });
  const created = await client.createScenario({ siteId: "campus-1km-v1", name: "demo" });
  assert.equal(created.scenario_id, SCENARIO);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://api.test/api/v1/scenarios");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["Idempotency-Key"], "sess-1:create");
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    site_id: "campus-1km-v1",
    name: "demo",
    initial_state: "baseline",
  });
});

test("commitEdits maps camelCase to contract fields and sends If-Match + key", async () => {
  const calls = [];
  const client = new ApiClient({
    fetch: async (url, init) => {
      calls.push({ url, init });
      return jsonResponse(
        202,
        {
          scenario_id: SCENARIO,
          scene_version: 18,
          exact_result_version: 17,
          status: "refining",
          job_id: "job_1",
          coalescing_window_ms: 500,
          status_url: "/api/v1/jobs/job_1",
        },
        { ETag: '"scene-version-18"' },
      );
    },
  });
  const body = await client.commitEdits(SCENARIO, {
    baseSceneVersion: 17,
    edits: [{ operation: "add", tree: { tree_id: "t1", component_type: "broad_canopy" } }],
    requestedResult: { timeIndices: [12], variables: ["utci"] },
    idempotencyKey: "browser-7-edit-42",
  });
  assert.equal(body.scene_version, 18);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `/api/v1/scenarios/${SCENARIO}/edits`);
  assert.equal(calls[0].init.headers["Idempotency-Key"], "browser-7-edit-42");
  assert.equal(calls[0].init.headers["If-Match"], '"scene-version-17"');
  const sent = JSON.parse(calls[0].init.body);
  assert.deepEqual(sent, {
    base_scene_version: 17,
    edits: [{ operation: "add", tree: { tree_id: "t1", component_type: "broad_canopy" } }],
    requested_result: { time_indices: [12], variables: ["utci"] },
  });
});

test("commitUniversalEdits posts family items to the universal endpoint with the same preconditions", async () => {
  const calls = [];
  const client = new ApiClient({
    fetch: async (url, init) => {
      calls.push({ url, init });
      return jsonResponse(
        202,
        {
          scenario_id: SCENARIO,
          scene_version: 6,
          exact_result_version: 5,
          status: "refining",
          job_id: "job_6",
          transport: "universal-edits-v1",
          coalescing_window_ms: 500,
          status_url: "/api/v1/jobs/job_6",
        },
        { ETag: '"scene-version-6"' },
      );
    },
  });
  const item = {
    adapter: "meteorological_forcing",
    operation: "update_time_row",
    values: { air_temperature: 31.5 },
    target: null,
    time_index: 12,
    old_values: null,
  };
  const body = await client.commitUniversalEdits(SCENARIO, {
    baseSceneVersion: 5,
    edits: [item],
    requestedResult: { timeIndices: [12] },
    idempotencyKey: "browser-7-edit-9",
  });
  assert.equal(body.transport, "universal-edits-v1");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `/api/v1/scenarios/${SCENARIO}/edits/universal`);
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["Idempotency-Key"], "browser-7-edit-9");
  assert.equal(calls[0].init.headers["If-Match"], '"scene-version-5"');
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    base_scene_version: 5,
    edits: [item],
    requested_result: { time_indices: [12] },
  });
});

test("error envelopes become ApiClientError with code, field, and details", async () => {
  const client = new ApiClient({
    fetch: async () =>
      jsonResponse(409, {
        error: {
          code: "scene_version_conflict",
          message: "The scenario changed after the client's base version.",
          current_scene_version: 19,
          scenario_url: `/api/v1/scenarios/${SCENARIO}`,
        },
      }),
  });
  await assert.rejects(
    () =>
      client.commitEdits(SCENARIO, {
        baseSceneVersion: 17,
        edits: [{ operation: "add", tree_id_only: false, tree: { tree_id: "t1" } }],
        idempotencyKey: "k",
      }),
    (error) => {
      assert.ok(error instanceof ApiClientError);
      assert.equal(error.code, "scene_version_conflict");
      assert.equal(error.status, 409);
      assert.equal(error.details.current_scene_version, 19);
      assert.equal(error.details.scenario_url, `/api/v1/scenarios/${SCENARIO}`);
      return true;
    },
  );

  const geometryClient = new ApiClient({
    fetch: async () =>
      jsonResponse(422, {
        error: {
          code: "invalid_tree_geometry",
          message: "canopy_diameter_m must be between 1 and 30",
          field: "edits[0].tree.canopy_diameter_m",
          request_id: "req_1",
        },
      }),
  });
  await assert.rejects(
    () =>
      geometryClient.commitEdits(SCENARIO, {
        baseSceneVersion: 1,
        edits: [{ operation: "add", tree: { tree_id: "t1" } }],
        idempotencyKey: "k",
      }),
    (error) => {
      assert.equal(error.code, "invalid_tree_geometry");
      assert.equal(error.field, "edits[0].tree.canopy_diameter_m");
      assert.equal(error.status, 422);
      return true;
    },
  );
});

test("Retry-After headers surface as details.retry_after_ms (body wins, junk ignored)", async () => {
  // The transport-level IP gate signals its wait ONLY through the header:
  // its JSON envelope carries no retry field, so the client must mine the
  // header or lose the hint entirely.
  const headerOnly = new ApiClient({
    fetch: async () =>
      jsonResponse(
        429,
        { error: { code: "rate_limited", message: "more than 600 requests per minute" } },
        { "Retry-After": "4" },
      ),
  });
  await assert.rejects(
    () => headerOnly.getScenario(SCENARIO),
    (error) => {
      assert.equal(error.code, "rate_limited");
      assert.equal(error.details.retry_after_ms, 4000);
      return true;
    },
  );

  // Float delay-seconds parse too (defence against a future server shape).
  assert.equal(parseRetryAfterMs("1.5"), 1500);
  assert.equal(parseRetryAfterMs("0"), 0);
  assert.equal(parseRetryAfterMs(null), null);
  assert.equal(parseRetryAfterMs(""), null);
  assert.equal(parseRetryAfterMs("Wed, 21 Oct 2015 07:28:00 GMT"), null, "HTTP-date form is not honored");
  assert.equal(parseRetryAfterMs("soon"), null);

  // Body-level hints keep precedence when both exist.
  const bodyWins = new ApiClient({
    fetch: async () =>
      jsonResponse(
        429,
        { error: { code: "rate_limited", message: "workspace burst", retry_after_ms: 250 } },
        { "Retry-After": "9" },
      ),
  });
  await assert.rejects(
    () => bodyWins.getScenario(SCENARIO),
    (error) => {
      assert.equal(error.details.retry_after_ms, 250);
      return true;
    },
  );
});

test("fetch rejection surfaces as network_error", async () => {
  const client = new ApiClient({
    fetch: async () => {
      throw new TypeError("fetch failed");
    },
  });
  await assert.rejects(
    () => client.getScenario(SCENARIO),
    (error) => error.code === "network_error",
  );
});

test("runtime lacks zstd DecompressionStream in this node (evidence gate)", () => {
  // Documents the platform fact behind the identity fallback decision: if
  // this ever flips to true on the CI node, the zstd decode test below
  // becomes the end-to-end path instead of the stub.
  assert.equal(typeof globalThis.DecompressionStream, "function");
  assert.equal(zstdStreamSupported(), false, "node should not expose zstd streams yet");
});

test("payload Accept negotiates identity when zstd is unavailable", async () => {
  const calls = [];
  const result = await buildResult({ sceneVersion: 2, values: makeVariable(["utci"], 3, 8, 8) });
  const client = new ApiClient({
    fetch: async (url, init) => {
      calls.push({ url, init });
      return new Response(result.payloadBytes, {
        status: 200,
        headers: {
          "content-type": PATCH_MEDIA_TYPE_IDENTITY,
          etag: result.etag,
          "x-solweig-scene-version": "2",
          "x-solweig-schema-version": "1",
        },
      });
    },
  });
  const payload = await client.getResultPayload(result.manifest.payload_url);
  assert.deepEqual(payload.bytes, result.payloadBytes);
  assert.equal(payload.etag, result.etag);
  assert.equal(payload.sceneVersion, 2);
  assert.equal(calls[0].init.headers.Accept, PATCH_MEDIA_TYPE_IDENTITY);
});

test("payload Accept prefers zstd when the runtime supports it", async () => {
  const calls = [];
  const client = new ApiClient({
    decompressionStream: makePassthroughZstdStream(),
    fetch: async (url, init) => {
      calls.push({ url, init });
      return new Response(new Uint8Array(4), { status: 200 });
    },
  });
  await client.getResultPayload("/payload");
  assert.equal(
    calls[0].init.headers.Accept,
    `${PATCH_MEDIA_TYPE_ZSTD}, ${PATCH_MEDIA_TYPE_IDENTITY};q=0.5`,
  );
});

test("decodePatch verifies checksum and splits variable-major wire order", async () => {
  const variables = ["utci", "tmrt"];
  const values = {
    utci: makeVariable(["utci"], 2, 8, 8).utci,
    tmrt: makeVariable(["tmrt"], 2, 8, 8, 5000).tmrt,
  };
  const result = await buildResult({
    sceneVersion: 4,
    timeIndices: [5, 6],
    variables,
    values,
  });
  const client = new ApiClient({ fetch: async () => new Response(null) });
  const decoded = await client.decodePatch(result.manifest, {
    bytes: result.payloadBytes,
    contentType: PATCH_MEDIA_TYPE_IDENTITY,
    etag: result.etag,
  });
  assert.equal(decoded.manifest.scene_version, 4);
  assert.deepEqual(decoded.timeIndices, [5, 6]);
  assert.deepEqual(decoded.window, { rowStart: 0, rowStop: 8, colStart: 0, colStop: 8 });

  const { utci, tmrt } = decoded.variables;
  assert.equal(utci.length, 2 * 8 * 8);
  assert.equal(tmrt.length, 2 * 8 * 8);
  // Little-endian float32 wire values survive the round trip exactly.
  assert.equal(utci[0 * 64 + 3 * 8 + 7], planeValue(0, 3, 7));
  assert.equal(utci[1 * 64 + 0 * 8 + 0], planeValue(1, 0, 0));
  assert.equal(tmrt[1 * 64 + 2 * 8 + 2], planeValue(1, 2, 2, 5000));
});

test("decodePatch handles a sub-window patch with matching checksum", async () => {
  const window = { row_start: 2, row_stop: 6, col_start: 3, col_stop: 7 };
  const rows = 4;
  const cols = 4;
  const values = makeVariable(["utci"], 1, rows, cols, 77);
  const result = await buildResult({
    sceneVersion: 7,
    window,
    timeIndices: [12],
    variables: ["utci"],
    values,
  });
  const client = new ApiClient({ fetch: async () => new Response(null) });
  const decoded = await client.decodePatch(result.manifest, {
    bytes: result.payloadBytes,
    contentType: PATCH_MEDIA_TYPE_IDENTITY,
    etag: result.etag,
  });
  assert.deepEqual(decoded.window, { rowStart: 2, rowStop: 6, colStart: 3, colStop: 7 });
  assert.equal(decoded.variables.utci.length, rows * cols);
  assert.equal(decoded.variables.utci[2 * 4 + 3], planeValue(0, 2, 3, 77));
});

test("decodePatch rejects tampered payloads, bad schemas, and bad lengths", async () => {
  const client = new ApiClient({ fetch: async () => new Response(null) });

  const good = await buildResult({ sceneVersion: 1, values: makeVariable(["utci"], 1, 8, 8) });
  const tampered = good.payloadBytes.slice();
  tampered[0] ^= 0xff;
  await assert.rejects(
    () => client.decodePatch(good.manifest, { bytes: tampered, etag: good.etag }),
    (error) => error.code === "payload_checksum_mismatch",
  );

  const wrongEtag = await buildResult({ sceneVersion: 1, values: makeVariable(["utci"], 1, 8, 8) });
  await assert.rejects(
    () =>
      client.decodePatch(wrongEtag.manifest, {
        bytes: wrongEtag.payloadBytes,
        etag: '"sha256-deadbeef"',
      }),
    (error) => error.code === "payload_checksum_mismatch",
  );

  const badSchema = await buildResult({
    sceneVersion: 1,
    schemaVersion: 99,
    values: makeVariable(["utci"], 1, 8, 8),
  });
  await assert.rejects(
    () => client.decodePatch(badSchema.manifest, { bytes: badSchema.payloadBytes }),
    (error) => error.code === "payload_schema_mismatch",
  );

  const truncated = await buildResult({ sceneVersion: 1, values: makeVariable(["utci"], 1, 8, 8) });
  const shortManifest = {
    ...truncated.manifest,
    variables: truncated.manifest.variables.map((entry) => ({ ...entry, shape: [2, 8, 8] })),
  };
  await assert.rejects(
    () => client.decodePatch(shortManifest, { bytes: truncated.payloadBytes }),
    (error) => error.code === "payload_length_mismatch",
  );

  const windowMismatch = await buildResult({
    sceneVersion: 1,
    values: makeVariable(["utci"], 1, 8, 8),
  });
  const mismatchManifest = {
    ...windowMismatch.manifest,
    window: { row_start: 0, row_stop: 4, col_start: 0, col_stop: 8 },
  };
  await assert.rejects(
    () => client.decodePatch(mismatchManifest, { bytes: windowMismatch.payloadBytes }),
    (error) => error.code === "payload_shape_mismatch",
  );
});

test("decodePatch decodes the zstd branch through a stub DecompressionStream", async () => {
  // The stub "decompresses" by passing bytes through, so the fixture payload
  // is raw bytes labeled zstd; this exercises the zstd code path (format
  // check, stream plumbing, post-decompress validation) end to end.
  const values = makeVariable(["utci"], 1, 8, 8, 9);
  const result = await buildResult({
    sceneVersion: 3,
    timeIndices: [9],
    compression: "zstd",
    values,
  });
  const client = new ApiClient({
    decompressionStream: makePassthroughZstdStream(),
    fetch: async () => new Response(null),
  });
  const decoded = await client.decodePatch(result.manifest, {
    bytes: result.payloadBytes,
    contentType: PATCH_MEDIA_TYPE_ZSTD,
    etag: result.etag,
  });
  assert.equal(decoded.variables.utci[4 * 8 + 4], planeValue(0, 4, 4, 9));
});

test("decodePatch fails clearly when zstd is served but unsupported", async () => {
  const values = makeVariable(["utci"], 1, 8, 8);
  const result = await buildResult({
    sceneVersion: 3,
    compression: "zstd",
    values,
  });
  const client = new ApiClient({ fetch: async () => new Response(null) });
  await assert.rejects(
    () => client.decodePatch(result.manifest, { bytes: result.payloadBytes, etag: result.etag }),
    (error) => {
      assert.equal(error.code, "unsupported_payload_compression");
      assert.match(error.message, /identity/);
      return true;
    },
  );
});

/** A DecompressionStream stub whose "zstd" is a byte passthrough. */
function makePassthroughZstdStream() {
  return class StubZstdDecompressionStream {
    constructor(format) {
      if (format !== "zstd") throw new TypeError(`unsupported format ${format}`);
      return new TransformStream({
        transform(chunk, controller) {
          controller.enqueue(chunk);
        },
      });
    }
  };
}

test("resolveApiBase de-doubles a same-origin proxy prefix", () => {
  // All client paths already carry the /api/v1 prefix, so the documented
  // same-origin proxy base "?api=/api" must resolve to "" (paths ride the
  // proxy unchanged) while STILL reporting connected mode.
  assert.deepEqual(resolveApiBase("/api"), { baseUrl: "", connected: true });
  assert.deepEqual(resolveApiBase("/api/"), { baseUrl: "", connected: true });
  assert.deepEqual(resolveApiBase(undefined), { baseUrl: "", connected: false });
  assert.deepEqual(resolveApiBase(""), { baseUrl: "", connected: false });
  // Absolute origins are untouched (CORS-enabled servers keep working).
  assert.deepEqual(resolveApiBase("http://127.0.0.1:8000"), {
    baseUrl: "http://127.0.0.1:8000",
    connected: true,
  });
  // An absolute origin that happens to end in /api keeps its scheme intact:
  // the strip only applies to same-origin relative bases.
  assert.deepEqual(resolveApiBase("http://127.0.0.1:8000/api"), {
    baseUrl: "http://127.0.0.1:8000",
    connected: true,
  });
});

test("decodePatch accepts an identity-served payload behind a zstd manifest", async () => {
  // Production shape (2026-09-04 deployment): the server honors
  // Accept: ...+identity and serves RAW planes, while the manifest still
  // carries the zstd-encoding checksum. The X-SOLWEIG-Checksum header
  // carries the digest of the bytes actually served — that is the only
  // checksum the client may verify the downloaded bytes against.
  const values = makeVariable(["utci"], 1, 8, 8);
  const result = await buildResult({ sceneVersion: 2, values });
  const servedDigest = await sha256Hex(result.payloadBytes);
  const manifest = {
    ...result.manifest,
    compression: "zstd",
    checksum: "sha256:" + "0".repeat(64), // zstd-encoding digest, NOT these bytes
  };
  const client = new ApiClient({ fetch: async () => new Response(null) });
  const decoded = await client.decodePatch(manifest, {
    bytes: result.payloadBytes,
    contentType: PATCH_MEDIA_TYPE_IDENTITY,
    etag: `"sha256-${servedDigest}"`,
    checksum: `sha256:${servedDigest}`,
  });
  assert.equal(decoded.variables.utci.length, 8 * 8);
  assert.equal(decoded.variables.utci[3 * 8 + 7], planeValue(0, 3, 7));
});

test("decodePatch still refuses tampered bytes when the header checksum disagrees", async () => {
  const values = makeVariable(["utci"], 1, 8, 8);
  const result = await buildResult({ sceneVersion: 2, values });
  const tampered = result.payloadBytes.slice();
  tampered[0] ^= 0xff;
  const client = new ApiClient({ fetch: async () => new Response(null) });
  await assert.rejects(
    client.decodePatch(
      { ...result.manifest, compression: "zstd", checksum: "sha256:" + "0".repeat(64) },
      {
        bytes: tampered,
        contentType: PATCH_MEDIA_TYPE_IDENTITY,
        checksum: "sha256:" + (await sha256Hex(result.payloadBytes)),
      },
    ),
    (error) => {
      assert.equal(error.code, "payload_checksum_mismatch");
      return true;
    },
  );
});
