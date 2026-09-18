// SPDX-License-Identifier: GPL-3.0-only
//
// Realtime collaboration client for the SOLWEIG design tool (additive module,
// in adoption — the studio UI does not use it yet).
//
// DOM-free, Node-testable counterpart to `api_client.mjs` for the
// collaborative operation plane described in
// `docs/incremental_design_tool/realtime_collaboration/`:
//
//   POST /api/v1/workspaces/{id}/operations            (idempotent submit)
//   GET  /api/v1/workspaces/{id}/operations?since_server_sequence=N  (catch-up)
//   GET  /api/v1/workspaces/{id}/events                (EventSource stream)
//
// `fetch`, the EventSource factory, timers, and the clock are injected, so
// node tests drive everything with canned responses and a manual clock; the
// app passes the real seams. The client never gates a submit on
// `base_revision` — it is advisory metadata only; the server's reducer owns
// ordering and conflict resolution.
//
// Server events consumed (emitted by the realtime plane):
//
//   canonical_revision {workspace_revision, epoch_id, operations: [...],
//                       fast_revision, exact_revision[, snapshot: true]}
//   fast_revision      {fast_revision, workspace_revision, result_class,
//                       exact_base_revision}
//   exact_revision     {exact_revision, workspace_revision}
//   heartbeat          {missed_events?: N}
//
// Fast-lane consumption contract (r2c):
//
// * The server guarantees `canonical_revision` for a revision reaches the
//   subscriber BEFORE the matching `fast_revision` frame — the client relies
//   on that for the initial render, but a reconnect may MISS fast frames
//   entirely (fast payloads are in-memory + SSE only): a `visual_pending`
//   result with no follow-up fast frame is a legitimate terminal state until
//   the next epoch. The client never waits for one.
// * `result_class` is `fast_exact` (the fast result is authoritative for its
//   revision) or `visual_pending` (a placeholder — the exact result is still
//   owed). `fast_qualified` is reserved server-side; any unknown future
//   class displays conservatively as visual_pending (`effectiveClass`).
// * Supersession fence: a fast result is rendered only for the canonical
//   revision it approximates (`targetRevision`). Once a newer canonical
//   revision is rendered, the fast result is `superseded` and never
//   authoritative again — the client never renders a fast result behind the
//   last rendered canonical revision.
// * A lagged subscriber's heartbeat carries `missed_events: N` (broadcast.py
//   drops the OLDEST frames when its queue overflows); the client answers it
//   with the same catch-up GET the reconnect path uses.
// * Admission rejects BEFORE durable acceptance: 429 (workspace burst) and
//   503 `fast_lane_unavailable` (server-wide) surface as typed
//   `RealtimeAdmissionError` (`kind: "burst" | "server_saturated"` with
//   `retry_after_ms` + `advice`) instead of being retried blind. Because an
//   idempotent RETRY of an already-durable batch can also draw a 429 (r2b
//   F7), the client reconciles through the catch-up GET before surfacing a
//   rejection on a retry attempt — a batch the server already accepted
//   resolves through the canonical fence instead of rejecting.
//
// Zero-loss fences: submit promises resolve only when the operation appears
// in a canonical_revision (unresolved acks older than the stall deadline
// surface through `onStall`), and a reconnect buffers live events until a
// catch-up GET has closed the event gap — operations are applied exactly once
// per `operation_id` regardless of how often they are re-carried.

import { parseRetryAfterMs } from "./api_client.mjs";

export const REALTIME_SUBMIT_RETRY_BASE_MS = 1000;
export const REALTIME_MAX_SUBMIT_RETRIES = 3;
export const REALTIME_RECONNECT_BASE_MS = 1000;
export const REALTIME_MAX_RECONNECT_RETRIES = 3;
export const REALTIME_STALL_TIMEOUT_MS = 10_000;

export const REALTIME_EVENT_TYPES = Object.freeze([
  "canonical_revision",
  "fast_revision",
  "exact_revision",
  "exact_progress",
  "heartbeat",
]);

export const RESULT_CLASSES = Object.freeze([
  "fast_exact",
  "fast_qualified",
  "visual_pending",
  "exact_reconciled",
]);

export class RealtimeClientError extends Error {
  constructor({ code, message, status = 0, details = {}, url = null }) {
    super(message || code);
    this.name = "RealtimeClientError";
    this.code = code;
    this.message = message || code;
    this.status = Number(status) || 0;
    this.details = details ?? {};
    this.url = url;
  }
}

/**
 * Typed admission rejection (BEFORE durable acceptance — nothing was
 * appended): `kind: "burst"` (429, the workspace's own burst — the client
 * can fix it) or `kind: "server_saturated"` (503 fast_lane_unavailable —
 * retry later). `retryAfterMs` is the server's retry hint, `advice` its
 * remediation list. Surfaced by `submitOperations`; NEVER auto-retried.
 */
export class RealtimeAdmissionError extends RealtimeClientError {
  constructor({ kind, retryAfterMs = null, advice = [], code, message, status = 0, details = {}, url = null }) {
    super({ code, message, status, details, url });
    this.name = "RealtimeAdmissionError";
    this.kind = kind;
    this.retryAfterMs = Number.isFinite(Number(retryAfterMs)) ? Number(retryAfterMs) : null;
    this.advice = Object.freeze([...advice]);
  }
}

function networkError(url, cause) {
  return new RealtimeClientError({
    code: "network_error",
    message: `request to ${url} failed: ${cause?.message ?? cause}`,
    url,
  });
}

export class RealtimeClient {
  #nowFn;
  #clientSequence = 0;
  #batchCounter = 0;
  #workspaces = new Map();

  /**
   * @param {object} options
   * @param {string} [options.baseUrl] API origin ("" for same-origin).
   * @param {Function} [options.fetch] fetch implementation (tests inject).
   * @param {Function} [options.eventSourceFactory] `(url) => EventSource-like`
   *        ({addEventListener, close}). Real EventSource unavailable in Node;
   *        this factory is the seam where the app connects it.
   * @param {string} [options.actorId] stable actor id (default: generated uuid).
   * @param {object} [options.crypto] WebCrypto (for the default actor uuid).
   * @param {object} [options.timers] injectable setTimeout/clearTimeout.
   * @param {Function} [options.now] injectable monotonic clock (ms).
   * @param {number} [options.stallTimeoutMs] ack-stall deadline.
   * @param {Function} [options.onStatus] `({phase, workspaceId, ...}) => void`.
   * @param {Function} [options.onStall] `({workspaceId, operationId,
   *        serverSequence, ageMs}) => void` for unresolved acks past the deadline.
   */
  constructor({
    baseUrl = "",
    fetch: fetchImpl = globalThis.fetch?.bind(globalThis),
    eventSourceFactory =
      typeof globalThis.EventSource === "function"
        ? (url) => new globalThis.EventSource(url)
        : null,
    actorId = null,
    crypto: cryptoImpl = globalThis.crypto,
    timers = null,
    now = null,
    submitRetryBaseMs = REALTIME_SUBMIT_RETRY_BASE_MS,
    maxSubmitRetries = REALTIME_MAX_SUBMIT_RETRIES,
    reconnectBaseMs = REALTIME_RECONNECT_BASE_MS,
    maxReconnectRetries = REALTIME_MAX_RECONNECT_RETRIES,
    stallTimeoutMs = REALTIME_STALL_TIMEOUT_MS,
    onStatus = () => {},
    onStall = () => {},
  } = {}) {
    if (typeof fetchImpl !== "function") {
      throw new TypeError("RealtimeClient requires a fetch implementation");
    }
    this.fetch = fetchImpl;
    this.eventSourceFactory = eventSourceFactory;
    this.crypto = cryptoImpl;
    this.baseUrl = String(baseUrl ?? "").replace(/\/+$/, "");
    this.actorId = actorId || this.#generateActorId();
    this.submitRetryBaseMs = submitRetryBaseMs;
    this.maxSubmitRetries = maxSubmitRetries;
    this.reconnectBaseMs = reconnectBaseMs;
    this.maxReconnectRetries = maxReconnectRetries;
    this.stallTimeoutMs = stallTimeoutMs;
    this.onStatus = onStatus;
    this.onStall = onStall;
    this.timers = timers ?? {
      setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
      clearTimeout: (id) => globalThis.clearTimeout(id),
    };
    this.#nowFn = now ?? (() => globalThis.performance?.now?.() ?? Date.now());
    this.#clientSequence = 0;
    this.#batchCounter = 0;
    this.#workspaces = new Map();
  }

  #generateActorId() {
    if (typeof this.crypto?.randomUUID === "function") {
      return this.crypto.randomUUID();
    }
    return `actor-${Math.random().toString(36).slice(2, 10)}`;
  }

  #nowMs() {
    return Number(this.#nowFn()) || 0;
  }

  url(path) {
    if (/^https?:\/\//i.test(path)) return path;
    if (this.baseUrl === "") return path;
    return `${this.baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
  }

  #status(workspace, phase, detail = {}) {
    this.onStatus({ phase, workspaceId: workspace.id, ...detail });
  }

  // -- shared HTTP -----------------------------------------------------------

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
    if (typeof response.json !== "function") return null;
    return await response.json();
  }

  async #errorFromResponse(response, url) {
    let code = `http_${response.status}`;
    let message = `${response.status} ${response.statusText ?? ""}`.trim();
    let details = {};
    try {
      const parsed = await response.json();
      if (parsed?.error) {
        code = parsed.error.code ?? code;
        message = parsed.error.message ?? message;
        const { code: _c, message: _m, request_id: _r, ...rest } = parsed.error;
        details = rest;
      }
    } catch {
      // Non-JSON error body: keep the HTTP-level code and message.
    }
    // The transport-level IP rate gate (server middleware) signals its wait
    // ONLY through the header — its JSON envelope has no retry field — so
    // surface it as details.retry_after_ms for #asAdmissionRejection. Body
    // fields (the workspace burst limiter's own hint) keep precedence.
    const retryAfterMs = parseRetryAfterMs(response.headers?.get("retry-after"));
    if (retryAfterMs !== null && details.retry_after_ms === undefined) {
      details.retry_after_ms = retryAfterMs;
    }
    return new RealtimeClientError({ code, message, status: response.status, details, url });
  }

  /** Transient transport failures (network drop, 5xx, 429) are retryable. */
  #isTransient(error) {
    if (!(error instanceof RealtimeClientError)) return false;
    return (
      error.code === "network_error" ||
      error.code === "rate_limited" ||
      (error.status ?? 0) >= 500
    );
  }

  // -- workspace state -------------------------------------------------------

  #makeWorkspaceState(workspaceId) {
    const workspace = {
      id: workspaceId,
      subscribed: false,
      state: "idle", // idle | live | buffering | reconnecting | catching_up | closed | failed
      handlers: {},
      source: null,
      sourceBroken: false,
      gapPending: false,
      bufferedEvents: [],
      reconnectCycle: false,
      reconnectAttempts: 0,
      reconnectTimer: null,
      appliedOperationIds: new Set(),
      pendingAcks: new Map(),
      revisions: { workspaceRevision: 0, fastRevision: 0, exactRevision: 0 },
      // Highest canonical revision actually rendered (canonical_revision
      // frames, exact_revision frames, and the catch-up GET body — the
      // sources the supersession fence trusts). Fast frames never advance it.
      lastCanonicalRevision: 0,
      lastServerSequence: 0,
      resultClass: null,
      lastExactEventAt: this.#nowMs(),
      lastHeartbeatAt: null,
    };
    workspace.handle = {
      workspaceId,
      close: () => this.unsubscribeWorkspace(workspaceId),
      get state() {
        return workspace.state;
      },
    };
    return workspace;
  }

  #workspaceState(workspaceId) {
    let workspace = this.#workspaces.get(workspaceId);
    if (!workspace) {
      workspace = this.#makeWorkspaceState(workspaceId);
      this.#workspaces.set(workspaceId, workspace);
    }
    return workspace;
  }

  /** Revision triple for a workspace (or null when unknown). */
  revisionsFor(workspaceId) {
    const workspace = this.#workspaces.get(workspaceId);
    if (!workspace) return null;
    return { ...workspace.revisions };
  }

  /** Connection bookkeeping for a workspace (or null when unknown). */
  subscriptionState(workspaceId) {
    const workspace = this.#workspaces.get(workspaceId);
    if (!workspace) return null;
    return {
      state: workspace.state,
      lastServerSequence: workspace.lastServerSequence,
      lastHeartbeatAt: workspace.lastHeartbeatAt,
      appliedOperationCount: workspace.appliedOperationIds.size,
      pendingAckCount: workspace.pendingAcks.size,
    };
  }

  /**
   * One-shot journal fetch for a subscribed workspace: apply every operation
   * past the client's watermark exactly once (deduped by operation_id against
   * live SSE frames). The subscriber's SCENE rebuild rides this — the SSE
   * snapshot frame is deliberately empty of history, so a client joining a
   * workspace that already holds operations calls this right after
   * subscribing to converge its scene, not just its revisions. Failure is
   * the caller's to surface (the stream itself stays live).
   */
  async catchUpWorkspace(workspaceId) {
    const workspace = this.#workspaces.get(workspaceId);
    if (!workspace) {
      throw new RealtimeClientError({
        code: "not_subscribed",
        message: `catchUpWorkspace: ${workspaceId} is not subscribed`,
      });
    }
    await this.#catchUp(workspace);
  }

  /**
   * Result-class view for the status badge (`realtime_badge.mjs`):
   * `{class, effectiveClass, authoritative, superseded, targetRevision,
   * fastRevision, exactRevision, exactBaseRevision, ageMs}`.
   *
   * `class` is the raw wire class (preserved for forward compatibility);
   * `effectiveClass` is the conservative display class — `fast_exact` only
   * when a NON-superseded fast result is authoritative for its revision,
   * `exact_reconciled` only when the exact result still covers the rendered
   * canonical revision, otherwise `visual_pending` (reserved classes like
   * `fast_qualified` and unknown future classes land here too). `ageMs`
   * counts ms since the last `exact_revision` event while the display class
   * is not `exact_reconciled`. Null before any class-bearing event.
   */
  classFor(workspaceId) {
    const workspace = this.#workspaces.get(workspaceId);
    if (!workspace || !workspace.resultClass) return null;
    const record = workspace.resultClass;
    const superseded = workspace.lastCanonicalRevision > Number(record.targetRevision ?? 0);
    const effectiveClass = !superseded && record.class === "fast_exact"
      ? "fast_exact"
      : !superseded && record.class === "exact_reconciled"
        ? "exact_reconciled"
        : "visual_pending";
    const ageMs =
      effectiveClass === "exact_reconciled"
        ? 0
        : Math.max(0, this.#nowMs() - workspace.lastExactEventAt);
    return {
      class: record.class,
      effectiveClass,
      authoritative: effectiveClass !== "visual_pending",
      superseded,
      targetRevision: record.targetRevision ?? null,
      fastRevision: workspace.revisions.fastRevision,
      exactRevision: workspace.revisions.exactRevision,
      exactBaseRevision: record.exactBaseRevision,
      ageMs,
    };
  }

  // -- operation submission --------------------------------------------------

  /**
   * Submit a batch of operations. `baseRevision` is ADVISORY: it rides the
   * wire but never gates the submit locally — the server reducer resolves
   * ordering and conflicts.
   *
   * The returned promise resolves — per operation — only when that operation
   * appears in a canonical_revision event (zero-loss client fence), with
   * `{operationId, serverSequence, epochId, duplicate}`. Network failures
   * re-POST the identical idempotent body (1s/2s/4s backoff, three retries);
   * a duplicate ack resolves as success carrying the ORIGINAL server_sequence.
   *
   * @param {Array} operations `{sourceFamily, entityId, verb, payload,
   *        operationId?, baseRevision?}` items.
   * @param {object} [options] `{baseRevision}` for the whole batch.
   * @returns {Promise<Array<object>>} one ack result per submitted operation.
   */
  async submitOperations(workspaceId, operations, { baseRevision = null } = {}) {
    if (!Array.isArray(operations) || operations.length === 0) {
      throw new RealtimeClientError({
        code: "invalid_request",
        message: "submitOperations requires a non-empty operations array",
      });
    }
    const workspace = this.#workspaceState(workspaceId);
    const items = operations.map((operation) => {
      const clientSequence = (this.#clientSequence += 1);
      const operationId = String(
        operation?.operationId ?? `${this.actorId}:op-${clientSequence}`,
      );
      const base =
        operation?.baseRevision !== undefined && operation?.baseRevision !== null
          ? Number(operation.baseRevision)
          : baseRevision !== null && baseRevision !== undefined
            ? Number(baseRevision)
            : null;
      return {
        operationId,
        clientSequence,
        wire: {
          operation_id: operationId,
          client_sequence: clientSequence,
          base_revision: base,
          source_family: operation?.sourceFamily ?? operation?.source_family ?? null,
          entity_id: operation?.entityId ?? operation?.entity_id ?? null,
          verb: operation?.verb ?? null,
          payload: operation?.payload ?? {},
        },
      };
    });

    const body = { actor_id: this.actorId, operations: items.map((item) => item.wire) };
    const idempotencyKey = `${this.actorId}:batch-${(this.#batchCounter += 1)}`;

    // Register the ack fence before the POST so a canonical_revision that
    // races the HTTP response still resolves the submit.
    const ackPromises = items.map((item) => {
      // A resubmit of the SAME operation_id (held-row retry, stall
      // resends) overwrites the fence entry — settle the orphaned one as a
      // duplicate first or its submit promise never resolves (leak; the
      // failure path only reaches the newest entry).
      const previous = workspace.pendingAcks.get(item.operationId);
      if (previous && !previous.settled) {
        previous.settled = true;
        if (previous.timer !== null) this.timers.clearTimeout(previous.timer);
        workspace.pendingAcks.delete(item.operationId);
        previous.resolve({
          operationId: item.operationId,
          serverSequence: null,
          epochId: null,
          duplicate: true,
        });
      }
      const promise = new Promise((resolve, reject) => {
        workspace.pendingAcks.set(item.operationId, {
          resolve,
          reject,
          settled: false,
          serverSequence: null,
          epochId: null,
          duplicate: false,
          submittedAt: null,
          timer: null,
        });
      });
      // A failed batch rejects each pending ack, but until Promise.all
      // consumes them the caller holds only the batch-level promise — guard
      // the bare per-op rejections against surfacing as unhandled.
      promise.catch(() => {});
      return promise;
    });

    let result;
    try {
      result = await this.#postOperationsWithRetries(
        workspace,
        body,
        idempotencyKey,
        items.map((item) => item.operationId),
      );
    } catch (error) {
      for (const item of items) this.#failPendingAck(workspace, item.operationId, error);
      throw error;
    }

    if (result.reconciled) {
      // The batch turned out to be already durable (r2b F7): the reconcile
      // GET applied the operations, which resolved the pending acks through
      // the normal fence. Nothing to register from the (absent) HTTP acks.
      return await Promise.all(ackPromises);
    }

    const acks = Array.isArray(result.response)
      ? result.response
      : result.response?.acks ?? result.response?.operations ?? [];
    for (const ack of acks) {
      const operationId = String(ack?.operation_id ?? "");
      const entry = workspace.pendingAcks.get(operationId);
      if (!entry || entry.settled) continue;
      entry.serverSequence = Number(ack.server_sequence) || null;
      entry.epochId = ack.epoch_id ?? null;
      entry.duplicate = Boolean(ack.duplicate);
      entry.submittedAt = this.#nowMs();
      if (typeof this.onStall === "function" && this.stallTimeoutMs > 0) {
        entry.timer = this.timers.setTimeout(() => {
          if (entry.settled) return;
          // Fence only: surface the unresolved ack; the promise itself stays
          // resolvable when the operation eventually lands.
          this.onStall({
            workspaceId,
            operationId,
            serverSequence: entry.serverSequence,
            ageMs: this.#nowMs() - entry.submittedAt,
          });
        }, this.stallTimeoutMs);
      }
    }
    return await Promise.all(ackPromises);
  }

  #failPendingAck(workspace, operationId, error) {
    const entry = workspace.pendingAcks.get(operationId);
    if (!entry || entry.settled) return;
    entry.settled = true;
    if (entry.timer !== null) this.timers.clearTimeout(entry.timer);
    workspace.pendingAcks.delete(operationId);
    entry.reject(error);
  }

  /** Admission rejections surface instead of retrying: 429 burst, 503 saturated. */
  #isAdmissionRejection(error) {
    return (
      error instanceof RealtimeClientError &&
      (error.status === 429 ||
        (error.status === 503 && error.code === "fast_lane_unavailable"))
    );
  }

  #asAdmissionRejection(error) {
    return new RealtimeAdmissionError({
      kind: error.status === 503 ? "server_saturated" : "burst",
      retryAfterMs: error.details?.retry_after_ms,
      advice: Array.isArray(error.details?.advice) ? error.details.advice : [],
      code: error.code,
      message: error.message,
      status: error.status,
      details: error.details,
      url: error.url,
    });
  }

  /**
   * r2b F7 reconciliation: a 429 on an idempotent RETRY may mean the batch
   * is ALREADY durable (the first attempt's response was lost and admission
   * now rejects the replay). Catch up from the watermark; if any operation
   * of the batch appears, the batch landed — the GET has applied it and the
   * canonical fence resolves the pending acks. Returns false when the catch
   * up itself fails or nothing is durable (surface the rejection).
   */
  async #reconcileDurableBatch(workspace, operationIds) {
    try {
      await this.#catchUp(workspace);
    } catch {
      return false;
    }
    return operationIds.some((operationId) =>
      workspace.appliedOperationIds.has(operationId),
    );
  }

  async #postOperationsWithRetries(workspace, body, idempotencyKey, operationIds) {
    const path = `/api/v1/workspaces/${encodeURIComponent(workspace.id)}/operations`;
    let attempt = 0;
    for (;;) {
      try {
        const response = await this.#request(path, {
          method: "POST",
          body,
          headers: { "Idempotency-Key": idempotencyKey },
        });
        return { response };
      } catch (error) {
        if (this.#isAdmissionRejection(error)) {
          // Only a RETRY can carry a durable batch (the first attempt's
          // rejection happened before any acceptance, so there is nothing
          // to reconcile).
          if (attempt > 0 && (await this.#reconcileDurableBatch(workspace, operationIds))) {
            return { reconciled: true };
          }
          throw this.#asAdmissionRejection(error);
        }
        if (attempt < this.maxSubmitRetries && this.#isTransient(error)) {
          const backoffMs = this.submitRetryBaseMs * 2 ** attempt;
          await new Promise((resolve) => this.timers.setTimeout(resolve, backoffMs));
          attempt += 1;
          continue;
        }
        throw error;
      }
    }
  }

  // -- subscription ----------------------------------------------------------

  /**
   * Subscribe to a workspace's realtime event stream.
   *
   * @param {string} workspaceId
   * @param {object} [handlers]
   * @param {Function} [handlers.onOperation] `(operation, {origin})` — fired
   *        EXACTLY once per unique operation_id, in apply order
   *        (`origin` is "event" or "catch_up").
   * @param {Function} [handlers.onCanonicalRevision] raw canonical events.
   * @param {Function} [handlers.onFastRevision] raw fast events.
   * @param {Function} [handlers.onExactRevision] raw exact events.
   * @param {Function} [handlers.onExactProgress] additive job-transition
   *        telemetry `{job_id, status, target_revision, eta_seconds,
   *        eta_basis[, queue_position]}` — the countdown plane for
   *        server-minted exact-lane jobs whose ids ride no operation.
   * @param {Function} [handlers.onHeartbeat] raw heartbeat events.
   * @param {Function} [handlers.onRoster] live actor ids from the heartbeat
   *        roster (presence truth; absent on older servers).
   * @returns {{workspaceId, close(): void, state: string}} subscription handle.
   */
  subscribeWorkspace(workspaceId, handlers = {}) {
    const workspace = this.#workspaceState(workspaceId);
    if (workspace.subscribed) return workspace.handle;
    workspace.handlers = {
      onOperation: () => {},
      onCanonicalRevision: () => {},
      onFastRevision: () => {},
      onExactRevision: () => {},
      onExactProgress: () => {},
      onHeartbeat: () => {},
      onRoster: () => {},
      ...handlers,
    };
    workspace.subscribed = true;
    this.#status(workspace, "connecting");
    this.#openStream(workspace);
    workspace.state = "live";
    this.#status(workspace, "live");
    return workspace.handle;
  }

  unsubscribeWorkspace(workspaceId) {
    const workspace = this.#workspaces.get(workspaceId);
    if (!workspace) return;
    workspace.subscribed = false;
    this.#closeSource(workspace);
    if (workspace.reconnectTimer !== null) {
      this.timers.clearTimeout(workspace.reconnectTimer);
      workspace.reconnectTimer = null;
    }
    workspace.state = "closed";
    this.#status(workspace, "closed");
  }

  #openStream(workspace) {
    if (typeof this.eventSourceFactory !== "function") {
      workspace.state = "failed";
      throw new RealtimeClientError({
        code: "event_source_unavailable",
        message:
          "RealtimeClient requires an EventSource factory (inject one for tests; " +
          "the app passes the browser EventSource)",
      });
    }
    // Presence identity rides the subscribe URL: the server's heartbeat
    // roster only lists actors that registered one (?actor_id=, sanitized
    // server-side), so omitting it would keep every client's live tier
    // permanently empty and demote all presence to "+N earlier".
    const params = new URLSearchParams({ actor_id: this.actorId });
    const url = this.url(
      `/api/v1/workspaces/${encodeURIComponent(workspace.id)}/events?${params}`,
    );
    const source = this.eventSourceFactory(url);
    workspace.source = source;
    workspace.sourceBroken = false;
    for (const type of [...REALTIME_EVENT_TYPES, "error"]) {
      source.addEventListener?.(type, (event) => this.#onStreamEvent(workspace, type, event));
    }
  }

  #closeSource(workspace) {
    workspace.source?.close?.();
    workspace.source = null;
  }

  #onStreamEvent(workspace, type, event) {
    if (!workspace.subscribed) return;
    if (type === "error") {
      this.#onStreamError(workspace);
      return;
    }
    const parsed = this.#parseEventData(event?.data);
    if (workspace.gapPending) {
      // Zero-loss: during a reconnect gap nothing applies until the catch-up
      // GET has closed the hole; events are buffered and flushed in order.
      workspace.bufferedEvents.push({ type, event: parsed ?? {} });
      return;
    }
    this.#deliverEvent(workspace, type, parsed ?? {});
  }

  #parseEventData(raw) {
    if (raw && typeof raw === "object") return raw;
    if (typeof raw === "string" && raw !== "") {
      try {
        return JSON.parse(raw);
      } catch {
        return null;
      }
    }
    return null;
  }

  #deliverEvent(workspace, type, event) {
    if (type === "canonical_revision") this.#applyCanonicalEvent(workspace, event);
    else if (type === "fast_revision") this.#applyFastEvent(workspace, event);
    else if (type === "exact_revision") this.#applyExactEvent(workspace, event);
    else if (type === "exact_progress") {
      // Additive countdown telemetry (broadcast_exact_progress): passed
      // through verbatim — state derivation (deadline monotonicity, terminal
      // retirement) is the app shell's business, not the transport's.
      workspace.handlers.onExactProgress?.(event);
    } else if (type === "heartbeat") {
      workspace.lastHeartbeatAt = this.#nowMs();
      // Live-actor roster (broadcast.py heartbeat): presence truth. Unknown
      // field on older servers → no call, no break.
      if (Array.isArray(event?.roster)) {
        workspace.handlers.onRoster?.(event.roster.map(String).filter(Boolean));
      }
      workspace.handlers.onHeartbeat?.(event);
      // Lagged-recovery notice (broadcast.py): the hub drops a lagging
      // subscriber's OLDEST frames and flags the count on the next
      // heartbeat. Answer it with the same catch-up the reconnect path uses.
      const missed = Number(event?.missed_events ?? 0);
      if (Number.isFinite(missed) && missed > 0) {
        this.#recoverMissedEvents(workspace, missed);
      }
    }
  }

  // -- missed_events recovery -------------------------------------------------

  /**
   * Close the gap a `missed_events` heartbeat revealed. The stream itself is
   * still healthy, so unlike the reconnect path nothing is re-opened: the
   * gap fence goes up synchronously (later frames buffer), the catch-up GET
   * replays the dropped operations, then buffered frames flush in order.
   */
  #recoverMissedEvents(workspace, missedCount) {
    if (workspace.gapPending || workspace.reconnectCycle) return;
    workspace.gapPending = true;
    workspace.bufferedEvents = [];
    workspace.state = "catching_up";
    this.#status(workspace, "catching_up", { missedEvents: missedCount });
    this.#runMissedEventsRecovery(workspace).catch((cause) => {
      workspace.gapPending = false;
      workspace.state = "failed";
      this.#status(workspace, "catch_up_failed", { error: this.#errorPayload(cause) });
    });
  }

  async #runMissedEventsRecovery(workspace) {
    let catchUpError = null;
    try {
      await this.#catchUp(workspace);
    } catch (error) {
      catchUpError = error;
    }
    if (!workspace.subscribed || workspace.reconnectCycle) {
      // Closed, or the reconnect cycle took the gap over mid-recovery (it
      // resets the buffer and re-fetches everything itself).
      workspace.gapPending = false;
      return;
    }
    if (catchUpError !== null) {
      // The GET failed: drop the buffer (nothing in it was applied and the
      // watermark is untouched) and hand recovery to the reconnect cycle,
      // whose fresh subscription also delivers a corrective snapshot frame.
      this.#closeSource(workspace);
      workspace.gapPending = false;
      workspace.bufferedEvents = [];
      this.#onStreamError(workspace);
      return;
    }
    const buffered = workspace.bufferedEvents;
    workspace.bufferedEvents = [];
    // Clear the fence BEFORE the flush: a flushed missed_events heartbeat
    // (the queue can overflow again mid-recovery) must be able to trigger
    // the next cycle.
    workspace.gapPending = false;
    for (const { type, event } of buffered) {
      this.#deliverEvent(workspace, type, event);
    }
    // Only the OWNING cycle returns the badge to live: a flushed heartbeat
    // may have started the next cycle (gapPending re-armed synchronously),
    // and that cycle's own tail transitions when its catch-up lands.
    if (workspace.subscribed && workspace.state === "catching_up" && !workspace.gapPending) {
      workspace.state = "live";
      this.#status(workspace, "live");
    }
  }

  #applyCanonicalEvent(workspace, event) {
    const revisions = workspace.revisions;
    revisions.workspaceRevision = Math.max(
      revisions.workspaceRevision,
      Number(event.workspace_revision ?? 0),
    );
    workspace.lastCanonicalRevision = Math.max(
      workspace.lastCanonicalRevision,
      Number(event.workspace_revision ?? 0),
    );
    if (event.fast_revision !== undefined && event.fast_revision !== null) {
      revisions.fastRevision = Math.max(revisions.fastRevision, Number(event.fast_revision));
    }
    if (event.exact_revision !== undefined && event.exact_revision !== null) {
      revisions.exactRevision = Math.max(revisions.exactRevision, Number(event.exact_revision));
    }
    this.#reconcileIfCaughtUp(workspace);
    const operations = Array.isArray(event.operations) ? event.operations : [];
    for (const rawOperation of operations) {
      this.#applyOperation(workspace, rawOperation, "event");
    }
    workspace.handlers.onCanonicalRevision?.(event);
  }

  #applyFastEvent(workspace, event) {
    const revisions = workspace.revisions;
    const fastRevision = Number(event.fast_revision ?? 0);
    revisions.fastRevision = Math.max(revisions.fastRevision, fastRevision);
    if (event.workspace_revision !== undefined && event.workspace_revision !== null) {
      revisions.workspaceRevision = Math.max(
        revisions.workspaceRevision,
        Number(event.workspace_revision),
      );
    }
    // A fast result only ever replaces an equal-or-newer one: a stale frame
    // (lagged delivery, reconnect flush) must never walk the class record
    // back, and a re-carried frame is an idempotent overwrite.
    const currentSequence = workspace.resultClass?.fastSequence ?? Number.NEGATIVE_INFINITY;
    if (
      workspace.resultClass === null ||
      fastRevision > currentSequence ||
      (fastRevision === currentSequence && workspace.resultClass.class !== "exact_reconciled")
    ) {
      workspace.resultClass = {
        class: String(event.result_class ?? "visual_pending"),
        // Reserved (fast_qualified) and unknown future classes display
        // conservatively as visual_pending via classFor's effectiveClass.
        exactBaseRevision:
          event.exact_base_revision !== undefined && event.exact_base_revision !== null
            ? Number(event.exact_base_revision)
            : null,
        targetRevision:
          event.workspace_revision !== undefined && event.workspace_revision !== null
            ? Number(event.workspace_revision)
            : revisions.workspaceRevision,
        fastSequence: fastRevision,
      };
    }
    workspace.handlers.onFastRevision?.(event);
  }

  #applyExactEvent(workspace, event) {
    const revisions = workspace.revisions;
    revisions.exactRevision = Math.max(revisions.exactRevision, Number(event.exact_revision ?? 0));
    if (event.workspace_revision !== undefined && event.workspace_revision !== null) {
      revisions.workspaceRevision = Math.max(
        revisions.workspaceRevision,
        Number(event.workspace_revision),
      );
    }
    // The exact plane only covers canonical revisions — the exact frame is
    // itself evidence the canonical revision was published.
    workspace.lastCanonicalRevision = Math.max(
      workspace.lastCanonicalRevision,
      Number(event.workspace_revision ?? 0),
    );
    workspace.lastExactEventAt = this.#nowMs();
    this.#reconcileIfCaughtUp(workspace);
    workspace.handlers.onExactRevision?.(event);
  }

  /** An exact result never claims a class above the workspace it covers. */
  #reconcileIfCaughtUp(workspace) {
    const { workspaceRevision, exactRevision } = workspace.revisions;
    if (
      workspace.resultClass?.class !== "exact_reconciled" &&
      exactRevision > 0 &&
      exactRevision >= workspaceRevision
    ) {
      workspace.resultClass = {
        class: "exact_reconciled",
        exactBaseRevision: exactRevision,
        targetRevision: exactRevision,
        fastSequence: workspace.revisions.fastRevision,
      };
    }
  }

  /**
   * Apply one operation exactly once: dedupe by operation_id, advance the
   * catch-up watermark from any carried server_sequence, and resolve the
   * pending submit ack when this is the operation's canonical appearance.
   */
  #applyOperation(workspace, rawOperation, origin) {
    const operationId = String(rawOperation?.operation_id ?? "");
    if (operationId === "" || workspace.appliedOperationIds.has(operationId)) return false;
    workspace.appliedOperationIds.add(operationId);
    const serverSequence = Number(rawOperation.server_sequence);
    const hasServerSequence = Number.isFinite(serverSequence) && serverSequence > 0;
    if (hasServerSequence) {
      workspace.lastServerSequence = Math.max(workspace.lastServerSequence, serverSequence);
    }
    const entry = workspace.pendingAcks.get(operationId);
    if (entry && !entry.settled) {
      entry.settled = true;
      if (entry.timer !== null) this.timers.clearTimeout(entry.timer);
      workspace.pendingAcks.delete(operationId);
      entry.resolve({
        operationId,
        serverSequence:
          entry.serverSequence ?? (hasServerSequence ? serverSequence : null),
        epochId: entry.epochId,
        duplicate: entry.duplicate,
      });
    }
    workspace.handlers.onOperation?.(
      {
        operationId,
        actorId: rawOperation.actor_id ?? null,
        sourceFamily: rawOperation.source_family ?? null,
        entityId: rawOperation.entity_id ?? null,
        verb: rawOperation.verb ?? null,
        payload: rawOperation.compact_payload ?? rawOperation.payload ?? null,
        serverSequence: hasServerSequence ? serverSequence : null,
      },
      { origin },
    );
    return true;
  }

  // -- reconnect with catch-up -----------------------------------------------

  #onStreamError(workspace) {
    if (workspace.reconnectCycle) {
      // The replacement stream broke while its gap was still open; the active
      // cycle picks this up after the catch-up attempt settles.
      workspace.sourceBroken = true;
      this.#closeSource(workspace);
      return;
    }
    workspace.reconnectCycle = true;
    this.#closeSource(workspace);
    workspace.state = "reconnecting";
    this.#runReconnectCycle(workspace).catch((cause) => {
      workspace.reconnectCycle = false;
      workspace.state = "failed";
      this.#status(workspace, "reconnect_failed", { error: this.#errorPayload(cause) });
    });
  }

  async #runReconnectCycle(workspace) {
    while (workspace.subscribed) {
      workspace.reconnectAttempts += 1;
      if (workspace.reconnectAttempts > this.maxReconnectRetries) {
        workspace.reconnectCycle = false;
        workspace.state = "failed";
        this.#status(workspace, "reconnect_failed", {
          attempts: workspace.reconnectAttempts - 1,
        });
        return;
      }
      const backoffMs = this.reconnectBaseMs * 2 ** (workspace.reconnectAttempts - 1);
      this.#status(workspace, "reconnecting", {
        attempt: workspace.reconnectAttempts,
        maxRetries: this.maxReconnectRetries,
        backoffMs,
      });
      await new Promise((resolve) => {
        workspace.reconnectTimer = this.timers.setTimeout(() => {
          workspace.reconnectTimer = null;
          resolve();
        }, backoffMs);
      });
      if (!workspace.subscribed) return;

      // Open the replacement stream FIRST (buffering), then fetch the gap:
      // everything from the GET onward is covered by the stream, everything
      // before it by the catch-up response — nothing can fall in between.
      workspace.gapPending = true;
      workspace.bufferedEvents = [];
      this.#openStream(workspace);
      workspace.state = "buffering";
      let catchUpError = null;
      try {
        await this.#catchUp(workspace);
      } catch (error) {
        if (!this.#isTransient(error)) {
          workspace.gapPending = false;
          this.#closeSource(workspace);
          workspace.reconnectCycle = false;
          workspace.state = "failed";
          this.#status(workspace, "reconnect_failed", { error: this.#errorPayload(error) });
          return;
        }
        catchUpError = error;
      }
      if (!workspace.subscribed) {
        workspace.gapPending = false;
        return;
      }
      const buffered = workspace.bufferedEvents;
      workspace.bufferedEvents = [];
      for (const { type, event } of buffered) {
        this.#deliverEvent(workspace, type, event);
      }
      if (catchUpError !== null || workspace.sourceBroken) {
        // Transient catch-up failure or the young stream died again: drop
        // this attempt and back off again (bounded by the retry budget).
        this.#closeSource(workspace);
        workspace.gapPending = false;
        workspace.state = "reconnecting";
        continue;
      }
      workspace.gapPending = false;
      workspace.state = "live";
      workspace.reconnectAttempts = 0;
      workspace.reconnectCycle = false;
      this.#status(workspace, "live");
      return;
    }
  }

  /**
   * Close the event gap after a reconnect: re-read every operation the client
   * has not APPLIED yet (the watermark only advances on applied operations,
   * so in-flight acks are never skipped) and apply it in server order. The
   * operation_id dedupe makes re-carried operations harmless.
   */
  async #catchUp(workspace) {
    const path =
      `/api/v1/workspaces/${encodeURIComponent(workspace.id)}/operations` +
      `?since_server_sequence=${workspace.lastServerSequence}`;
    const response = await this.#request(path);
    const body = Array.isArray(response) ? { operations: response } : (response ?? {});
    const operations = Array.isArray(body.operations) ? [...body.operations] : [];
    operations.sort(
      (a, b) => Number(a?.server_sequence ?? 0) - Number(b?.server_sequence ?? 0),
    );
    for (const rawOperation of operations) {
      this.#applyOperation(workspace, rawOperation, "catch_up");
    }
    const revisions = workspace.revisions;
    if (body.workspace_revision !== undefined && body.workspace_revision !== null) {
      revisions.workspaceRevision = Math.max(
        revisions.workspaceRevision,
        Number(body.workspace_revision),
      );
      // The GET body's workspace_revision is the canonical scene_version —
      // it advances the supersession fence like a canonical frame would.
      workspace.lastCanonicalRevision = Math.max(
        workspace.lastCanonicalRevision,
        Number(body.workspace_revision),
      );
    }
    if (body.fast_revision !== undefined && body.fast_revision !== null) {
      revisions.fastRevision = Math.max(revisions.fastRevision, Number(body.fast_revision));
    }
    if (body.exact_revision !== undefined && body.exact_revision !== null) {
      revisions.exactRevision = Math.max(revisions.exactRevision, Number(body.exact_revision));
    }
  }

  #errorPayload(error) {
    if (error instanceof RealtimeClientError) {
      return { code: error.code, message: error.message };
    }
    return { code: "unexpected_error", message: String(error?.message ?? error) };
  }
}
