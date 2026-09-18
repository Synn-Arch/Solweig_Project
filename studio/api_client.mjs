// SPDX-License-Identifier: GPL-3.0-only
//
// API adapter seam for the SOLWEIG incremental design tool.
//
// This module is the ONLY place in the frontend that talks HTTP. It wraps the
// contract documented in `docs/incremental_design_tool/api_contract.md`:
//
//   POST /api/v1/scenarios                        (Idempotency-Key)
//   GET  /api/v1/capabilities                     (capability document)
//   GET  /api/v1/scenarios/{id}
//   POST /api/v1/scenarios/{id}/edits             (Idempotency-Key, If-Match)
//   POST /api/v1/scenarios/{id}/edits/universal   (Idempotency-Key, If-Match)
//   POST /api/v1/scenarios/{id}/reset             (Idempotency-Key, If-Match)
//   POST /api/v1/scenarios/{id}/views             (view-only, zero jobs)
//   GET  /api/v1/jobs/{job_id}                    (or any status_url)
//   GET  .../results/{scene_version}              (manifest)
//   GET  .../results/{scene_version}/payload      (binary patch)
//
// `fetch` is injected through the constructor so node tests can serve canned
// contract responses without a server, and the browser passes the real fetch.
//
// Payload compression is negotiated through the `Accept` header. The manifest
// `checksum` is SHA-256 over the *served* (possibly still compressed) bytes,
// so the client verifies it before decompressing; the response `ETag` carries
// the same digest (`"sha256-<hex>"`).

export const PATCH_SCHEMA_VERSION = 1;
export const PATCH_MEDIA_TYPE_ZSTD = "application/vnd.solweig.patch+zstd";
export const PATCH_MEDIA_TYPE_IDENTITY = "application/vnd.solweig.patch+identity";

export class ApiClientError extends Error {
  constructor({ code, message, field = null, status = 0, details = {}, url = null }) {
    super(message || code);
    this.name = "ApiClientError";
    this.code = code;
    this.message = message || code;
    this.field = field;
    this.status = Number(status) || 0;
    this.details = details ?? {};
    this.url = url;
  }
}

function networkError(url, cause) {
  return new ApiClientError({
    code: "network_error",
    message: `request to ${url} failed: ${cause?.message ?? cause}`,
    url,
  });
}

/**
 * Parse a `Retry-After` header into milliseconds. Only the delay-seconds
 * form is honored (int or float — the transport-level IP gate sends whole
 * seconds); HTTP-date forms and junk return null so callers keep their own
 * fallbacks. Body-level hints win over the header at the call site.
 */
export function parseRetryAfterMs(headerValue) {
  const raw = String(headerValue ?? "").trim();
  if (raw === "") return null;
  const seconds = Number(raw);
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  return seconds * 1000;
}

/** Detect whether the runtime can decode zstd streams natively. */
export function zstdStreamSupported(DecompressionStreamImpl = globalThis.DecompressionStream) {
  if (typeof DecompressionStreamImpl !== "function") return false;
  try {
    new DecompressionStreamImpl("zstd");
    return true;
  } catch {
    return false;
  }
}

/**
 * Resolve the "?api=" query parameter into the ApiClient base URL.
 *
 * Every client path already carries the `/api/v1` prefix, so a same-origin
 * reverse-proxy base like `/api` (the README contract for the serve.py proxy)
 * must NOT be prepended to those paths — it would request `/api/api/v1/...`
 * and 404. The trailing `/api` segment is therefore stripped from relative
 * bases, while absolute origins are preserved (minus their own trailing
 * `/api`, which the same de-doubling rule covers) so CORS-enabled servers
 * keep working unchanged.
 *
 * @param {string|null|undefined} apiParam raw "?api=" value (or undefined).
 * @returns {{baseUrl: string, connected: boolean}} connected is true for any
 *   non-empty parameter, even when the strip reduced the base to "".
 */
export function resolveApiBase(apiParam) {
  const raw = String(apiParam ?? "").trim();
  if (raw === "") return { baseUrl: "", connected: false };
  let baseUrl = raw.replace(/\/+$/, "");
  baseUrl = baseUrl.replace(/\/api$/i, "");
  return { baseUrl, connected: true };
}

export function littleEndianPlatform() {
  const probe = new Uint8Array([1, 0, 0, 0]);
  return new Uint32Array(probe.buffer)[0] === 1;
}

const IS_LITTLE_ENDIAN = littleEndianPlatform();

function readFloat32LittleEndian(buffer, byteOffset, count) {
  if (IS_LITTLE_ENDIAN) {
    return new Float32Array(buffer, byteOffset, count);
  }
  const view = new DataView(buffer, byteOffset, count * 4);
  const out = new Float32Array(count);
  for (let index = 0; index < count; index += 1) out[index] = view.getFloat32(index * 4, true);
  return out;
}

async function sha256Hex(bytes, cryptoImpl) {
  const subtle = cryptoImpl?.subtle;
  if (!subtle) {
    throw new ApiClientError({
      code: "unsupported_runtime",
      message: "WebCrypto (crypto.subtle) is required to verify result checksums",
    });
  }
  const digest = await subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export class ApiClient {
  /**
   * @param {object} options
   * @param {string} options.baseUrl API origin ("" for same-origin). No trailing slash.
   * @param {Function} [options.fetch] fetch implementation (tests inject a fake).
   * @param {object} [options.crypto] WebCrypto implementation.
   * @param {Function} [options.decompressionStream] DecompressionStream constructor.
   * @param {string} [options.sessionId] stable session id for idempotency keys.
   */
  constructor({
    baseUrl = "",
    fetch: fetchImpl = globalThis.fetch?.bind(globalThis),
    crypto: cryptoImpl = globalThis.crypto,
    decompressionStream = globalThis.DecompressionStream,
    sessionId = null,
  } = {}) {
    if (typeof fetchImpl !== "function") {
      throw new TypeError("ApiClient requires a fetch implementation");
    }
    this.fetch = fetchImpl;
    this.crypto = cryptoImpl;
    this.decompressionStream = decompressionStream;
    this.baseUrl = String(baseUrl ?? "").replace(/\/+$/, "");
    this.sessionId = sessionId || this.#generateSessionId();
    this.editCounter = 0;
  }

  #generateSessionId() {
    // Prefer a cryptographically random id (L-3); Math.random is only a
    // last-resort fallback for exotic runtimes without WebCrypto.
    if (typeof this.crypto?.randomUUID === "function") {
      return `studio-${this.crypto.randomUUID()}`;
    }
    return `studio-${Math.random().toString(36).slice(2, 10)}`;
  }

  url(path) {
    if (/^https?:\/\//i.test(path)) return path;
    if (this.baseUrl === "") return path;
    return `${this.baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
  }

  nextIdempotencyKey(scope = "edit") {
    if (scope === "edit") {
      this.editCounter += 1;
      return `${this.sessionId}:edit-${this.editCounter}`;
    }
    return `${this.sessionId}:${scope}`;
  }

  async #request(path, { method = "GET", body = null, headers = {} } = {}) {
    const url = this.url(path);
    let response;
    try {
      response = await this.fetch(url, {
        method,
        headers: {
          ...(body !== null ? { "Content-Type": "application/json" } : {}),
          ...headers,
        },
        ...(body !== null ? { body: JSON.stringify(body) } : {}),
      });
    } catch (cause) {
      throw networkError(url, cause);
    }
    if (!response.ok) {
      throw await this.#errorFromResponse(response, url);
    }
    return response;
  }

  async #errorFromResponse(response, url) {
    let code = `http_${response.status}`;
    let message = `${response.status} ${response.statusText ?? ""}`.trim();
    let field = null;
    let details = {};
    try {
      const parsed = await response.json();
      if (parsed?.error) {
        code = parsed.error.code ?? code;
        message = parsed.error.message ?? message;
        field = parsed.error.field ?? null;
        const { code: _c, message: _m, field: _f, request_id: _r, ...rest } = parsed.error;
        details = rest;
      }
    } catch {
      // Non-JSON error body: keep the HTTP-level code and message.
    }
    // The transport-level IP rate gate (server middleware) signals its wait
    // ONLY through the header — its JSON envelope has no retry field — so
    // surface it as details.retry_after_ms. Body fields keep precedence.
    const retryAfterMs = parseRetryAfterMs(response.headers?.get("retry-after"));
    if (retryAfterMs !== null && details.retry_after_ms === undefined) {
      details.retry_after_ms = retryAfterMs;
    }
    return new ApiClientError({ code, message, field, status: response.status, details, url });
  }

  // -- capabilities ----------------------------------------------------------

  /**
   * The engine's capability document (U-D2 discovery point). The frontend's
   * edit UI is generated from this document; it must never carry a hardcoded
   * fallback vocabulary. The short alias /capabilities exists for bootstrap;
   * the versioned path is preferred and the alias is only consulted when the
   * versioned one is absent (404), matching routes_capabilities.py.
   */
  async getCapabilities() {
    let response;
    try {
      response = await this.#request("/api/v1/capabilities");
    } catch (error) {
      if (error instanceof ApiClientError && error.status === 404) {
        response = await this.#request("/capabilities");
      } else {
        throw error;
      }
    }
    return await response.json();
  }

  // -- view-only operations (UEDIT-007: zero scientific jobs) ----------------

  /**
   * Run one view-only operation against the scenario's published result.
   *
   * Never enqueues a solver job; refusals arrive as typed errors whose
   * details carry `published_layers` (view_not_available 409) or nothing
   * (result_not_ready 404 / scenario_not_found 404 / invalid_request 400).
   *
   * @param {string} scenarioId
   * @param {object} payload
   * @param {string} payload.operation select_layer|compare|legend|cached_time
   * @param {string} payload.layer
   * @param {number} [payload.timeIndex]
   * @param {string|null} [payload.compareLayer]
   * @param {number|null} [payload.sceneVersion] defaults server-side to the
   *        scenario's latest published exact result
   */
  async createView(scenarioId, {
    operation,
    layer,
    timeIndex = 0,
    compareLayer = null,
    sceneVersion = null,
  }) {
    const body = { operation, layer, time_index: Number(timeIndex) };
    if (compareLayer !== null && compareLayer !== undefined && compareLayer !== "") {
      body.compare_layer = compareLayer;
    }
    if (sceneVersion !== null && sceneVersion !== undefined) {
      body.scene_version = Number(sceneVersion);
    }
    const response = await this.#request(
      `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/views`,
      { method: "POST", body },
    );
    return await response.json();
  }

  // -- scenarios -----------------------------------------------------------

  async createScenario({ siteId, name = "", idempotencyKey = null }) {
    const key = idempotencyKey ?? this.nextIdempotencyKey("create");
    const response = await this.#request("/api/v1/scenarios", {
      method: "POST",
      body: { site_id: siteId, name, initial_state: "baseline" },
      headers: { "Idempotency-Key": key },
    });
    return await response.json();
  }

  /**
   * @param {string} scenarioIdOrUrl scenario id, or an absolute path/URL as
   *        returned by error envelopes (`details.scenario_url`) or job bodies
   */
  async getScenario(scenarioIdOrUrl) {
    const asString = String(scenarioIdOrUrl);
    const path = asString.startsWith("/")
      ? asString
      : `/api/v1/scenarios/${encodeURIComponent(asString)}`;
    const response = await this.#request(path);
    return await response.json();
  }

  /**
   * Commit one batch of edits. Exactly one POST is issued per call.
   *
   * @param {object} payload
   * @param {number} payload.baseSceneVersion last known scene_version
   * @param {Array} payload.edits contract edit items ({operation, tree?, tree_id?})
   * @param {object|null} payload.requestedResult {time_indices?, variables?, refine_full_day?}
   * @param {string} payload.idempotencyKey unique per logical commit
   */
  async commitEdits(scenarioId, {
    baseSceneVersion,
    edits,
    requestedResult = null,
    idempotencyKey,
  }) {
    const body = {
      base_scene_version: Number(baseSceneVersion),
      edits,
    };
    if (requestedResult) {
      body.requested_result = {
        ...(requestedResult.timeIndices ? { time_indices: requestedResult.timeIndices } : {}),
        ...(requestedResult.variables ? { variables: requestedResult.variables } : {}),
        ...(requestedResult.refineFullDay !== undefined
          ? { refine_full_day: Boolean(requestedResult.refineFullDay) }
          : {}),
      };
    }
    const response = await this.#request(
      `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/edits`,
      {
        method: "POST",
        body,
        headers: {
          "Idempotency-Key": idempotencyKey,
          "If-Match": `"scene-version-${Number(baseSceneVersion)}"`,
        },
      },
    );
    return await response.json();
  }

  /**
   * Commit one batch of universal (family) edits. Same contract machinery
   * as commitEdits — Idempotency-Key, If-Match scene-version precondition,
   * requested_result assembly, 202 + job body — but the items are universal
   * edit items ({adapter, operation, values, target, time_index,
   * old_values}) and the response body additionally carries `transport:
   * "universal-edits-v1"`. Family edits ride the same job queue,
   * coalescing, and idempotency as tree edits.
   *
   * @param {object} payload same shape as commitEdits, with universal items
   */
  async commitUniversalEdits(scenarioId, {
    baseSceneVersion,
    edits,
    requestedResult = null,
    idempotencyKey,
  }) {
    const body = {
      base_scene_version: Number(baseSceneVersion),
      edits,
    };
    if (requestedResult) {
      body.requested_result = {
        ...(requestedResult.timeIndices ? { time_indices: requestedResult.timeIndices } : {}),
        ...(requestedResult.variables ? { variables: requestedResult.variables } : {}),
        ...(requestedResult.refineFullDay !== undefined
          ? { refine_full_day: Boolean(requestedResult.refineFullDay) }
          : {}),
      };
    }
    const response = await this.#request(
      `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/edits/universal`,
      {
        method: "POST",
        body,
        headers: {
          "Idempotency-Key": idempotencyKey,
          "If-Match": `"scene-version-${Number(baseSceneVersion)}"`,
        },
      },
    );
    return await response.json();
  }

  async resetScenario(scenarioId, { baseSceneVersion, idempotencyKey }) {
    const headers = { "Idempotency-Key": idempotencyKey };
    if (baseSceneVersion !== undefined && baseSceneVersion !== null) {
      headers["If-Match"] = `"scene-version-${Number(baseSceneVersion)}"`;
    }
    const response = await this.#request(
      `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/reset`,
      { method: "POST", body: {}, headers },
    );
    return await response.json();
  }

  // -- jobs ----------------------------------------------------------------

  async getJob(statusUrlOrJobId) {
    const path = statusUrlOrJobId.startsWith("/")
      ? statusUrlOrJobId
      : `/api/v1/jobs/${statusUrlOrJobId}`;
    const response = await this.#request(path);
    return await response.json();
  }

  // -- results -------------------------------------------------------------

  async getResultManifest(manifestUrl) {
    const response = await this.#request(manifestUrl);
    return await response.json();
  }

  /** Accept header for binary payloads, based on runtime zstd capability. */
  payloadAccept() {
    return zstdStreamSupported(this.decompressionStream)
      ? `${PATCH_MEDIA_TYPE_ZSTD}, ${PATCH_MEDIA_TYPE_IDENTITY};q=0.5`
      : `${PATCH_MEDIA_TYPE_IDENTITY}`;
  }

  /**
   * Download a binary result payload with compression negotiation.
   *
   * @returns {{bytes: Uint8Array, contentType: string, etag: string|null,
   *            sceneVersion: number|null, checksum: string|null}}
   */
  async getResultPayload(payloadUrl) {
    const url = this.url(payloadUrl);
    let response;
    try {
      response = await this.fetch(url, {
        method: "GET",
        headers: { Accept: this.payloadAccept() },
      });
    } catch (cause) {
      throw networkError(url, cause);
    }
    if (!response.ok) {
      throw await this.#errorFromResponse(response, url);
    }
    const buffer = await response.arrayBuffer();
    return {
      bytes: new Uint8Array(buffer),
      contentType: (response.headers?.get("content-type") ?? "").split(";")[0].trim(),
      etag: response.headers?.get("etag") ?? null,
      sceneVersion: Number(response.headers?.get("x-solweig-scene-version")) || null,
      checksum: response.headers?.get("x-solweig-checksum") ?? null,
    };
  }

  // -- patch decoding ------------------------------------------------------

  async #decompressPayload(manifest, payload) {
    // The SERVED content type wins over the manifest's compression field:
    // under Accept negotiation the server may deliver identity planes behind
    // a zstd-declaring manifest (and vice versa on a zstd-capable client).
    const servedType = String(payload.contentType ?? "");
    let compression;
    if (servedType === PATCH_MEDIA_TYPE_IDENTITY) compression = "identity";
    else if (servedType === PATCH_MEDIA_TYPE_ZSTD) compression = "zstd";
    else compression = String(manifest.compression ?? "zstd");
    const bytes = payload.bytes;
    if (compression === "identity") return bytes;
    if (compression !== "zstd") {
      throw new ApiClientError({
        code: "unsupported_payload_compression",
        message: `manifest declares unsupported compression ${JSON.stringify(compression)}`,
      });
    }
    if (!zstdStreamSupported(this.decompressionStream)) {
      throw new ApiClientError({
        code: "unsupported_payload_compression",
        message:
          "the server served a zstd payload but this browser cannot decode zstd streams " +
          "(DecompressionStream('zstd') unavailable); connect a server that honors " +
          `Accept: ${PATCH_MEDIA_TYPE_IDENTITY}`,
      });
    }
    const stream = new Blob([bytes]).stream().pipeThrough(
      new this.decompressionStream("zstd"),
    );
    const buffer = await new Response(stream).arrayBuffer();
    return new Uint8Array(buffer);
  }

  #expectedNbytes(variables) {
    let total = 0;
    for (const entry of variables) {
      const dtype = String(entry.dtype ?? "float32");
      if (dtype !== "float32") {
        throw new ApiClientError({
          code: "payload_dtype_mismatch",
          message: `variable ${entry.name} declares dtype ${dtype}, expected float32`,
        });
      }
      const shape = entry.shape ?? [];
      if (shape.length !== 3 || shape.some((v) => !Number.isInteger(v) || v < 0)) {
        throw new ApiClientError({
          code: "payload_shape_mismatch",
          message: `variable ${entry.name} shape must be (time, rows, cols), got ${JSON.stringify(shape)}`,
        });
      }
      total += shape.reduce((a, b) => a * b, 1) * 4;
    }
    return total;
  }

  /**
   * Verify and decode a result payload against its manifest.
   *
   * Order follows `data_model.md` and `solweig_gpu/server/patch_codec.py`:
   * C-order, little-endian, variable-major, then time, then row, then column.
   * The checksum (and ETag, when present) is verified over the served bytes
   * exactly as downloaded, before decompression.
   *
   * @returns {{variables: Object<string, Float32Array>, window: object,
   *            timeIndices: number[], manifest: object}}
   */
  async decodePatch(manifest, payload) {
    if (Number(manifest.schema_version) !== PATCH_SCHEMA_VERSION) {
      throw new ApiClientError({
        code: "payload_schema_mismatch",
        message: `patch schema version ${manifest.schema_version} is not supported (expected ${PATCH_SCHEMA_VERSION})`,
      });
    }
    const variables = manifest.variables ?? [];
    if (!Array.isArray(variables) || variables.length === 0) {
      throw new ApiClientError({
        code: "payload_shape_mismatch",
        message: "manifest declares no variables",
      });
    }

    const servedDigest = await sha256Hex(payload.bytes, this.crypto);
    // The manifest checksum covers the zstd encoding, but the server honors
    // Accept negotiation and may serve identity planes instead. The
    // X-SOLWEIG-Checksum response header carries the digest of the bytes
    // ACTUALLY served, so it is the only checksum these bytes may be
    // verified against; the manifest value remains the fallback for
    // transports (and tests) that supply no header.
    const checksum = String(
      payload.checksum ?? manifest.checksum ?? "",
    );
    if (!checksum.startsWith("sha256:")) {
      throw new ApiClientError({
        code: "payload_checksum_mismatch",
        message: "no sha256 checksum for the served payload (header or manifest)",
      });
    }
    if (servedDigest !== checksum.slice("sha256:".length)) {
      throw new ApiClientError({
        code: "payload_checksum_mismatch",
        message: `payload checksum mismatch: manifest ${checksum}, downloaded sha256:${servedDigest}`,
      });
    }
    if (payload.etag && payload.etag !== `"sha256-${servedDigest}"`) {
      throw new ApiClientError({
        code: "payload_checksum_mismatch",
        message: `payload ETag ${payload.etag} does not match the downloaded bytes`,
      });
    }

    const raw = await this.#decompressPayload(manifest, payload);
    const total = this.#expectedNbytes(variables);
    if (raw.byteLength !== total) {
      throw new ApiClientError({
        code: "payload_length_mismatch",
        message: `decompressed payload is ${raw.byteLength} bytes but the manifest implies ${total}`,
      });
    }

    const window = manifest.window ?? {};
    const height = Number(window.row_stop) - Number(window.row_start);
    const width = Number(window.col_stop) - Number(window.col_start);
    const decoded = {};
    let offset = 0;
    for (const entry of variables) {
      const [time, rows, cols] = entry.shape;
      const count = time * rows * cols;
      if (rows !== height || cols !== width) {
        throw new ApiClientError({
          code: "payload_shape_mismatch",
          message: `variable ${entry.name} shape ${entry.shape} does not match the manifest window ${height}×${width}`,
        });
      }
      decoded[entry.name] = readFloat32LittleEndian(
        raw.buffer,
        raw.byteOffset + offset,
        count,
      ).slice();
      offset += count * 4;
    }
    return {
      variables: decoded,
      window: {
        rowStart: Number(window.row_start),
        rowStop: Number(window.row_stop),
        colStart: Number(window.col_start),
        colStop: Number(window.col_stop),
      },
      timeIndices: (manifest.time_indices ?? []).map((t) => Number(t)),
      manifest,
    };
  }
}
