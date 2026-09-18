// SPDX-License-Identifier: GPL-3.0-only
//
// Exact-analysis session: the DOM-free state machine that drives the API
// contract on top of `ApiClient`.
//
// Responsibilities (the P7 acceptance gates):
//
//   UI-001  one POST /edits per committed interaction (mutations are chained
//           through a queue so base_scene_version is never guessed)
//   UI-003  stale rejection: every result is validated against the awaited
//           scene version before it may touch the texture; out-of-order poll
//           completions are dropped by a poll generation guard
//   UI-004  failures never mutate design state; the last attempt stays
//           retryable
//   UI-005  the manifest's model scope (model_version, site_cache_version,
//           limitations) is surfaced through callbacks
//
// The session holds only versions and job bookkeeping. Trees belong to the
// app; the session receives contract-shaped edit items and reports
// authoritative state back through callbacks.

import { ApiClientError } from "./api_client.mjs";
import { CapabilityError, parseCapabilityDocument } from "./capabilities.mjs";

export const TERMINAL_JOB_STATUSES = new Set(["complete", "superseded", "failed", "cancelled"]);

/**
 * Default single-client conflict policy (frontend_spec.md + api_contract.md):
 * reload authoritative state, then reapply the attempted edit exactly once if
 * it is still meaningful against the server's state.
 *
 * Tree edits consult the authoritative tree list (add only when absent,
 * move/update/delete only when present). Universal family edits carry no
 * client-side state to consult: the reapply happens exactly once and the
 * server's registry validators make the final ruling (e.g. a duplicate
 * building add is a typed edit_rejected, surfaced through onValidation).
 */
export function defaultConflictPolicy({ operation, treeId, serverTreeIds, item = null }) {
  if (item && typeof item === "object" && typeof item.adapter === "string") {
    return true; // universal family edit: server fences are the authority
  }
  const exists = serverTreeIds.has(treeId);
  if (operation === "add") return !exists;
  return exists; // move / update / delete need the tree server-side
}

export class ExactSession {
  #pollGeneration = 0;
  #pollTimer = null;
  #mutationQueue = Promise.resolve();
  #ready = false;
  #connectBacklog = [];
  /** Source nodes of the in-flight commit (echoed on the applied status). */
  #activeSourceNodes = null;
  /**
   * @param {object} options
   * @param {import("./api_client.mjs").ApiClient} options.client
   * @param {object} [options.callbacks]
   * @param {number} [options.pollIntervalMs]
   * @param {object} [options.timers] injectable setTimeout/clearTimeout
   * @param {Function} [options.conflictPolicy]
   */
  constructor({
    client,
    callbacks = {},
    // 1 s job-status cadence: solves run 40 s+, so sub-second polling only
    // burns the shared transport budget (through the studio proxy every
    // browser shares one IP rate bucket) without catching completions sooner.
    pollIntervalMs = 1000,
    timers = null,
    conflictPolicy = defaultConflictPolicy,
  }) {
    this.client = client;
    this.callbacks = {
      onStatus: () => {},
      onBaseline: () => {},
      onPatchApplied: () => {},
      onScope: () => {},
      onValidation: () => {},
      onAuthoritative: () => {},
      onCapabilities: () => {},
      onSiteMismatch: () => {},
      ...callbacks,
    };
    this.pollIntervalMs = pollIntervalMs;
    /** Transient GET /jobs failures (network, 5xx, 429) retry with backoff. */
    this.maxPollRetries = 5;
    this.maxPollBackoffMs = 5000;
    /**
     * Transient connect failures (network, 5xx, 429) reconnect with bounded
     * exponential backoff — 1 s, 2 s, 4 s (R-1). 4xx errors are permanent.
     */
    this.connectRetryBaseMs = 1000;
    this.maxConnectRetries = 3;
    /**
     * Cold-boot wake ladder (scale-to-zero): transport-class connect failures
     * (network error, 5xx — never 4xx, which are permanent) keep retrying for
     * ~90 s total, so a sleeping analysis machine can finish waking instead of
     * dropping the visitor onto the dead-end failure screen mid-wake.
     */
    this.connectWakeBudgetMs = 90000;
    this.connectWakeCapMs = 5000;
    this.timers = timers ?? {
      setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
      clearTimeout: (id) => globalThis.clearTimeout(id),
    };
    this.conflictPolicy = conflictPolicy;
    this.provideRecompute = null;

    this.scenarioId = null;
    this.siteId = null;
    this.sceneVersion = 0;
    this.exactResultVersion = 0;
    /** Pinned site grid placement from the scenario detail (or null). */
    this.siteGeometry = null;
    /** scene_version of the newest requested-but-unapplied result (null = exact). */
    this.awaitingVersion = null;
    this.grid = null; // {rows, cols} learned from the baseline manifest
    this.scope = null; // {modelVersion, siteCacheVersion, limitations}
    /** Parsed capability document (fetched at session start, never hardcoded). */
    this.capabilities = null;
    /**
     * Set when the server refuses an edit/reset with 409
     * site_identity_mismatch: {pinnedIdentity, currentIdentity, message}.
     * While set, further commits fail fast locally — the scenario is pinned
     * to a site this deployment no longer serves, so retrying would loop.
     */
    this.siteMismatch = null;
    this.lastError = null;

    this.#pollGeneration = 0;
    this.#pollTimer = null;
    this.#mutationQueue = Promise.resolve();
    this.lastAttempt = null;
  }

  get connected() {
    return this.scenarioId !== null;
  }

  isExactCurrent() {
    return this.awaitingVersion === null;
  }

  #status(phase, detail = {}) {
    this.callbacks.onStatus({ phase, ...detail });
  }

  // -----------------------------------------------------------------------
  // Connect: create scenario, load authoritative state, load baseline result
  // -----------------------------------------------------------------------

  async connect(
    { siteId, name = "SOLWEIG Studio session", scenarioId = null } = {},
    retryCount = 0,
    mintedScenarioId = null,
    wakeElapsedMs = 0,
    createKey = null
  ) {
    this.#status("connecting");
    // A fresh connect creates a fresh scenario with a fresh site pin, so any
    // prior mismatch state is obsolete (recovery path for site_mismatch).
    this.siteMismatch = null;
    try {
      if (scenarioId === null && mintedScenarioId === null) {
        // Shared world — the ONE decision point for join-vs-mint: a visitor
        // without an explicit ``?scenario=`` joins the server's default
        // workspace when the capability document publishes one. Only an
        // older server without the pointer mints a private scenario.
        // Capabilities load here so the mint path and the join branch below
        // share the same fetch; retries that carry a scenario or minted id
        // skip this block, and the `capabilities === null` guards keep one
        // attempt from fetching the document twice.
        await this.#loadCapabilities();
        scenarioId = this.capabilities?.defaultWorkspaceId ?? null;
      }
      if (scenarioId) {
        // Join an existing workspace (``?scenario=`` link or the server's
        // shared world): the collaborative plane is per-scenario, so a
        // second editor adopts the SAME scenario instead of minting a
        // private one. No create — the scenario is past creation by
        // construction.
        const existing = await this.client.getScenario(scenarioId);
        this.scenarioId = existing.scenario_id ?? scenarioId;
        this.siteId = existing.site_id ?? siteId;
        if (this.capabilities === null) await this.#loadCapabilities();
        const scenario = await this.client.getScenario(this.scenarioId);
        this.#adoptScenario(scenario);
        const manifestUrl =
          scenario.result_manifest_url ??
          `/api/v1/scenarios/${this.scenarioId}/results/${this.exactResultVersion}`;
        let result = null;
        try {
          result = await this.#fetchResult(manifestUrl, this.exactResultVersion);
        } catch (error) {
          // A missing baseline (result_not_ready) alongside an active job is
          // the boot race below; anything else fails the join as before.
          if (
            !(error instanceof ApiClientError && error.code === "result_not_ready") ||
            !existing.active_job_id
          ) {
            throw error;
          }
        }
        if (result === null && existing.active_job_id) {
          // Boot race on a site without stored baseline outputs: the shared
          // world exists but its baseline job is still running. Wait for the
          // scenario's active job (same ladder as the mint path), then read
          // the baseline once more before giving up on the join.
          await this.#awaitJob(null, existing.active_job_id);
          result = await this.#fetchResult(manifestUrl, this.exactResultVersion);
        }
        if (result === null) {
          throw new ApiClientError({
            code: "scenario_not_found",
            message: `workspace ${scenarioId} has no readable baseline result`,
          });
        }
        this.grid = { rows: result.window.rowStop, cols: result.window.colStop };
        this.awaitingVersion = null;
        this.callbacks.onBaseline(result);
        this.#ready = true;
        this.#status("ready", { sceneVersion: this.sceneVersion });
        this.#drainConnectBacklog();
        return result;
      }
      let baselineStatusUrl = null;
      let baselineJobId = null;
      if (mintedScenarioId === null) {
        // One idempotency key per connect chain: a create whose response is
        // lost re-POSTs as a REPLAY (the server re-serves the stored
        // response) instead of minting a second scenario per retry.
        createKey ??= this.client.nextIdempotencyKey("create");
        const created = await this.client.createScenario({ siteId, name, idempotencyKey: createKey });
        this.scenarioId = created.scenario_id;
        this.siteId = created.site_id ?? siteId;
        baselineStatusUrl = created.status_url ?? null;
        baselineJobId = created.job_id ?? null;
      } else {
        // Retry of an attempt that already minted (R-1): rejoin that scenario
        // instead of POSTing another create — a re-mint per retry multiplies
        // requests exactly when the transport is already strained, and every
        // fresh scenario would wait on its own baseline job again.
        this.scenarioId = mintedScenarioId;
        this.siteId = siteId;
      }

      // Session start = discover capabilities (U-D2): the edit UI is built
      // from the document, so a missing/unusable document fails the connect
      // visibly — there is deliberately NO hardcoded fallback vocabulary.
      // (Already loaded when the shared-world resolution above ran.)
      if (this.capabilities === null) await this.#loadCapabilities();

      // Sites without stored baseline outputs schedule a baseline job; wait for
      // it so the first result fetch cannot 404. On a mint-retry the create
      // response (and its status_url) is gone — the scenario detail's
      // active_job_id is the standing equivalent for the still-pending job.
      if (mintedScenarioId !== null) {
        const pending = await this.client.getScenario(this.scenarioId);
        baselineJobId = pending.active_job_id ?? null;
      }
      if (baselineStatusUrl !== null || baselineJobId !== null) {
        await this.#awaitJob(baselineStatusUrl, baselineJobId);
      }

      const scenario = await this.client.getScenario(this.scenarioId);
      this.#adoptScenario(scenario);

      const manifestUrl =
        scenario.result_manifest_url ??
        `/api/v1/scenarios/${this.scenarioId}/results/${this.exactResultVersion}`;
      let result = await this.#fetchResult(manifestUrl, this.exactResultVersion);
      if (result === null) {
        // A reset in the connecting window invalidated the fetch (R-2). No
        // edits can have been posted before ready, so the server scene is
        // still the pristine baseline: read it again on the current
        // generation instead of failing the connect.
        result = await this.#fetchResult(manifestUrl, this.exactResultVersion);
        if (result === null) {
          throw new ApiClientError({
            code: "connect_interrupted",
            message: "the session was reset while connecting",
          });
        }
      }
      this.grid = { rows: result.window.rowStop, cols: result.window.colStop };
      this.awaitingVersion = null;
      this.callbacks.onBaseline(result);
      this.#ready = true;
      this.#status("ready", { sceneVersion: this.sceneVersion });
      // Replay any edits the user committed while connecting (M-3): the
      // server is now authoritative and each carries a fresh base version.
      this.#drainConnectBacklog();
      return result;
    } catch (error) {
      const wakeBudgetLeft = wakeElapsedMs < this.connectWakeBudgetMs;
      if (this.#isTransportClass(error) && wakeBudgetLeft) {
        // Cold boot (the analysis machine wakes from scale-to-zero in ~40-60 s):
        // keep retrying well past the short ladder while the wake budget lasts,
        // announcing the wake so the UI can say the server is starting rather
        // than counting retries. Held edits stay in the backlog (M-3) and only
        // release if every attempt fails.
        const carried = scenarioId ?? this.scenarioId; // null when nothing landed yet
        this.scenarioId = null; // never claim connected from a failed attempt
        this.#status("reconnecting", {
          retry: retryCount + 1,
          waking: true,
          error: this.#errorPayload(error),
        });
        const backoffMs = Math.min(
          this.connectRetryBaseMs * 2 ** retryCount,
          this.connectWakeCapMs,
        );
        await new Promise((resolve) => this.timers.setTimeout(resolve, backoffMs));
        if (scenarioId) {
          return await this.connect(
            { siteId, name, scenarioId },
            retryCount + 1,
            null,
            wakeElapsedMs + backoffMs,
            createKey,
          );
        }
        return await this.connect(
          { siteId, name },
          retryCount + 1,
          carried,
          wakeElapsedMs + backoffMs,
          createKey,
        );
      }
      if (retryCount < this.maxConnectRetries && this.#isTransientError(error)) {
        // Transient connect failure (R-1): reconnect with bounded
        // exponential backoff. Held edits stay in the backlog and replay in
        // order when a later attempt lands; they are only released if every
        // attempt fails. A scenario this attempt already minted (or adopted
        // on a join) is carried into the retry: a create-path retry that
        // re-POSTs replays with the same idempotency key instead of minting
        // a fresh workspace. The wake budget is threaded through untouched:
        // an admission-control pause neither resets nor inflates it.
        const carried = scenarioId ?? this.scenarioId; // null when the mint POST itself failed
        this.scenarioId = null; // never claim connected from a failed attempt
        this.#status("reconnecting", {
          retry: retryCount + 1,
          maxRetries: this.maxConnectRetries,
          error: this.#errorPayload(error),
        });
        const backoffMs = this.connectRetryBaseMs * 2 ** retryCount;
        await new Promise((resolve) => this.timers.setTimeout(resolve, backoffMs));
        if (scenarioId) {
          return await this.connect(
            { siteId, name, scenarioId },
            retryCount + 1,
            null,
            wakeElapsedMs,
            createKey,
          );
        }
        return await this.connect({ siteId, name }, retryCount + 1, carried, wakeElapsedMs, createKey);
      }
      if (
        error instanceof ApiClientError &&
        error.code === "rate_limited" &&
        wakeBudgetLeft
      ) {
        // A 429 that arrives once the short ladder is spent, while the
        // server may still be waking: admission control paused us, but it
        // must not kill a waking connect. Fall back to the capped wake
        // cadence (announced as waking) — the budget keeps running down, so
        // this can never loop forever.
        const carried = scenarioId ?? this.scenarioId;
        this.scenarioId = null;
        this.#status("reconnecting", {
          retry: retryCount + 1,
          waking: true,
          error: this.#errorPayload(error),
        });
        const backoffMs = Math.min(
          this.connectRetryBaseMs * 2 ** retryCount,
          this.connectWakeCapMs,
        );
        await new Promise((resolve) => this.timers.setTimeout(resolve, backoffMs));
        if (scenarioId) {
          return await this.connect(
            { siteId, name, scenarioId },
            retryCount + 1,
            null,
            wakeElapsedMs + backoffMs,
            createKey,
          );
        }
        return await this.connect(
          { siteId, name },
          retryCount + 1,
          carried,
          wakeElapsedMs + backoffMs,
          createKey,
        );
      }
      // Never POST edits against a half-connected session; release the
      // backlog without sending (the UI already surfaces the connect failure).
      this.#releaseConnectBacklog();
      throw error;
    }
  }

  async #loadCapabilities() {
    let document;
    try {
      document = await this.client.getCapabilities();
    } catch (error) {
      // A transport-class drop (cold boot) rides the wake ladder as itself;
      // wrapping it would hide the transport class from the connect catch.
      if (this.#isTransportClass(error)) throw error;
      throw new ApiClientError({
        code: "capability_document_unavailable",
        message:
          `the server did not serve the capability document ` +
          `(${error.code ?? "error"}: ${error.message}); this frontend has no ` +
          `hardcoded edit vocabulary to fall back to`,
        status: error.status ?? 0,
      });
    }
    try {
      this.capabilities = parseCapabilityDocument(document);
    } catch (error) {
      if (error instanceof CapabilityError) {
        throw new ApiClientError({
          code: error.code,
          message: `capability document rejected: ${error.message}`,
        });
      }
      throw error;
    }
    this.callbacks.onCapabilities(this.capabilities);
  }

  #enqueueConnectBacklog(payload) {
    return new Promise((resolve) => {
      this.#connectBacklog.push({ payload, resolve });
    });
  }

  #drainConnectBacklog() {
    for (const { payload, resolve } of this.#connectBacklog.splice(0)) {
      const run = () => this.#commitEditsUnchecked(payload);
      const result = this.#mutationQueue.then(run, run);
      this.#mutationQueue = result.then(
        () => {},
        () => {},
      );
      resolve(result);
    }
  }

  #releaseConnectBacklog() {
    for (const { resolve } of this.#connectBacklog.splice(0)) resolve(null);
  }

  #adoptScenario(scenario) {
    this.sceneVersion = Number(scenario.scene_version ?? 0);
    this.exactResultVersion = Number(scenario.exact_result_version ?? 0);
    // The pinned site's grid placement (origin + pixel size): realtime
    // vegetation ops carry world metres, and this is the only surface that
    // exposes where the raster sits. Absent on older servers — callers
    // keep their span-local fallback frame.
    this.siteGeometry = scenario.site_geometry ?? null;
    if (scenario.model_scope) {
      this.scope = {
        modelVersion: scenario.model_scope.model_version ?? "",
        siteCacheVersion: scenario.model_scope.site_cache_version ?? "",
        limitations: [...(scenario.model_scope.limitations ?? [])],
      };
      this.callbacks.onScope(this.scope);
    }
    this.callbacks.onAuthoritative({
      sceneVersion: this.sceneVersion,
      exactResultVersion: this.exactResultVersion,
      trees: scenario.trees ?? [],
      reason: "adopt",
    });
  }

  // -----------------------------------------------------------------------
  // Committed interactions (UI-001)
  // -----------------------------------------------------------------------

  /**
   * Commit one interaction: exactly one POST /edits or /edits/universal (plus
   * at most one contract-mandated recovery POST after a 409
   * scene_version_conflict).
   *
   * @param {object} payload
   * @param {Array} payload.edits contract edit items (tree or universal)
   * @param {object|null} payload.requested {timeIndices, variables, refineFullDay}
   * @param {string} [payload.label] human-readable reason
   * @param {string} [payload.transport] "tree-edits-v1" (default) or
   *        "universal-edits-v1" (family edits ride the same job machinery)
   * @param {string[]} [payload.sourceNodes] capability-document source nodes
   *        the edit writes (e.g. ["vegetation_dsm"]); echoed on status events
   *        so the UI can classify changed/reused/recomputed stages (UEDIT-009)
   */
  commitEdits({
    edits,
    requested = null,
    label = "edit",
    transport = "tree-edits-v1",
    sourceNodes = null,
  } = {}) {
    if (this.siteMismatch) {
      // The scenario is pinned to a site identity this deployment no longer
      // serves; fail fast instead of posting into a guaranteed 409 loop.
      this.#emitSiteMismatch("commit blocked: reconnect with a fresh scenario first");
      return Promise.resolve(null);
    }
    if (!this.#ready) {
      // Connecting: hold the edit and replay it in order once the baseline is
      // loaded, so it cannot be lost or posted against a stale session (M-3).
      return this.#enqueueConnectBacklog({ edits, requested, label, transport, sourceNodes });
    }
    const run = () =>
      this.#commitEditsUnchecked({ edits, requested, label, transport, sourceNodes });
    // Chain mutations so a second commit learns the base_scene_version of the
    // first instead of racing it into a self-inflicted 409.
    const result = this.#mutationQueue.then(run, run);
    this.#mutationQueue = result.then(
      () => {},
      () => {},
    );
    return result;
  }

  async #commitEditsUnchecked({ edits, requested, label, transport, sourceNodes }) {
    const attempt = {
      edits,
      requested,
      label,
      transport: transport === "universal-edits-v1" ? transport : "tree-edits-v1",
      sourceNodes: Array.isArray(sourceNodes) ? sourceNodes.map(String) : null,
      baseSceneVersion: this.sceneVersion,
      idempotencyKey: this.client.nextIdempotencyKey("edit"),
      accepted: false,
      retriedAfterConflict: false,
      retriedAfterRateLimit: false,
    };
    return await this.#runAttempt(attempt);
  }

  async #runAttempt(attempt) {
    if (!this.connected) {
      this.#handleCommitError(
        new ApiClientError({ code: "not_connected", message: "session is not connected" }),
        attempt,
      );
      return null;
    }
    this.lastAttempt = attempt;
    try {
      return await this.#postEdits(attempt);
    } catch (error) {
      this.#handleCommitError(error, attempt);
      return null;
    }
  }

  async #postEdits(attempt) {
    const common = {
      baseSceneVersion: attempt.baseSceneVersion,
      edits: attempt.edits,
      requestedResult: attempt.requested,
      idempotencyKey: attempt.idempotencyKey,
    };
    const body =
      attempt.transport === "universal-edits-v1"
        ? await this.client.commitUniversalEdits(this.scenarioId, common)
        : await this.client.commitEdits(this.scenarioId, common);
    attempt.accepted = true;
    this.#activeSourceNodes = attempt.sourceNodes;
    this.sceneVersion = Number(body.scene_version);
    this.awaitingVersion = this.sceneVersion;
    this.lastError = null;
    this.#startPoll(body.status_url, body.job_id, this.sceneVersion);
    this.#status("queued", {
      jobId: body.job_id,
      sceneVersion: this.sceneVersion,
      coalescingWindowMs: body.coalescing_window_ms,
      label: attempt.label,
      sourceNodes: attempt.sourceNodes,
    });
    return body;
  }

  #handleCommitError(error, attempt) {
    if (error instanceof ApiClientError && error.code === "site_identity_mismatch") {
      this.#noteSiteMismatch(error);
      return;
    }
    if (error instanceof ApiClientError && error.code === "scene_version_conflict") {
      this.#recoverFromConflict(attempt, error).catch((cause) => {
        this.lastError = cause;
        this.#status("failed", { error: this.#errorPayload(cause) });
      });
      return;
    }
    if (
      error instanceof ApiClientError &&
      error.code === "rate_limited" &&
      !attempt.retriedAfterRateLimit
    ) {
      // The shared transport bucket (every browser behind the studio proxy
      // is one client IP) can 429 an otherwise-valid POST; bounce it once
      // off the server's hint instead of banner-ing immediately.
      this.#retryAfterRateLimit(attempt, error).catch((cause) => {
        this.lastError = cause;
        this.#status("failed", { error: this.#errorPayload(cause) });
      });
      return;
    }
    if (
      error instanceof ApiClientError &&
      error.status >= 400 &&
      error.status < 500 &&
      error.code !== "network_error" &&
      error.code !== "rate_limited"
    ) {
      // Client error (e.g. invalid_tree_geometry): the server rejected the
      // edit; design state is untouched and a retry cannot succeed unchanged.
      this.lastError = error;
      this.callbacks.onValidation({
        code: error.code,
        message: error.message,
        field: error.field,
      });
      return;
    }
    // Network / server error: keep the attempt retryable (UI-004).
    this.lastError = error;
    this.#status("failed", { error: this.#errorPayload(error) });
  }

  #emitSiteMismatch(message) {
    const detail = {
      message,
      pinnedIdentity: this.siteMismatch?.pinnedIdentity ?? null,
      currentIdentity: this.siteMismatch?.currentIdentity ?? null,
    };
    this.callbacks.onSiteMismatch(detail);
    this.#status("site_mismatch", detail);
  }

  /**
   * U-D intake item g: the live site geometry no longer matches the
   * scenario's pinned identity. This is permanent for the scenario — record
   * pinned vs current, kill every pending poll/result, and surface the
   * mismatch so all further mutations (commit, retry, reset) fail fast
   * locally instead of posting into a guaranteed 409 loop (no zombie state).
   * Recovery is a fresh scenario via a new connect.
   */
  #noteSiteMismatch(error) {
    this.siteMismatch = {
      pinnedIdentity: error.details.pinned_identity ?? null,
      currentIdentity: error.details.current_identity ?? null,
      message: error.message,
    };
    this.lastError = error;
    this.#invalidatePolls();
    this.awaitingVersion = null;
    this.#emitSiteMismatch(error.message);
  }

  /**
   * 409 recovery (api_contract.md): reload authoritative state, then reapply
   * or discard the attempted edit exactly once.
   */
  async #recoverFromConflict(attempt, error) {
    this.#status("conflict", { currentSceneVersion: error.details.current_scene_version });
    const scenarioUrl =
      error.details.scenario_url ?? `/api/v1/scenarios/${encodeURIComponent(this.scenarioId)}`;
    const scenario = await this.client.getScenario(scenarioUrl);
    this.#adoptScenario(scenario);

    const firstEdit = attempt.edits[0] ?? {};
    const treeId = firstEdit.tree?.tree_id ?? firstEdit.tree_id ?? null;
    const reapply = this.conflictPolicy({
      operation: firstEdit.operation,
      treeId,
      serverTreeIds: new Set((scenario.trees ?? []).map((tree) => tree.tree_id)),
      authoritative: scenario,
      item: firstEdit,
    });

    if (!reapply || attempt.retriedAfterConflict) {
      this.#status("conflict_discarded", {
        reason: !reapply
          ? "the server state no longer contains the edited tree"
          : "recovery already retried once",
      });
      return;
    }

    attempt.retriedAfterConflict = true;
    attempt.baseSceneVersion = this.sceneVersion;
    attempt.idempotencyKey = this.client.nextIdempotencyKey("edit");
    await this.#postEdits(attempt);
  }

  /**
   * One automatic retry of a rate-limited commit: wait out the server's
   * Retry-After hint (details.retry_after_ms; 2 s fallback when the server
   * sent none) and re-send. The Idempotency-Key is unchanged, so the re-POST
   * is a replay and can never double-apply; the retry flag bounds this to
   * exactly one bounce — a second rate_limited surfaces like any failure.
   */
  async #retryAfterRateLimit(attempt, error) {
    attempt.retriedAfterRateLimit = true;
    const hintMs = Number(error.details?.retry_after_ms);
    const waitMs = Number.isFinite(hintMs) && hintMs > 0 ? hintMs : 2000;
    await new Promise((resolve) => this.timers.setTimeout(resolve, waitMs));
    await this.#postEdits(attempt);
  }

  /**
   * Re-request analysis without changing the design (a no-op `update` edit):
   * used for solar-time changes, "Run again", and retry after a failed job.
   */
  requestRecompute({ tree, requested = null, label = "recompute" } = {}) {
    if (!tree) return Promise.resolve(null);
    return this.commitEdits({
      edits: [{ operation: "update", tree }],
      requested,
      label,
    });
  }

  /**
   * Run one view-only operation (UEDIT-007). Views are answered from the
   * published result — zero scientific jobs — so they bypass the mutation
   * queue entirely and never touch the poll generation.
   *
   * @param {object} payload {operation, layer, timeIndex?, compareLayer?, sceneVersion?}
   * @returns {Promise<object>} the view response body
   * @throws {ApiClientError} typed refusals: view_not_available (409, details
   *         carry published_layers), result_not_ready / scenario_not_found
   *         (404), invalid_request (400, e.g. an off-vocabulary layer)
   */
  async createView(payload = {}) {
    if (!this.connected) {
      throw new ApiClientError({
        code: "not_connected",
        message: "view operations require a connected session",
      });
    }
    return await this.client.createView(this.scenarioId, {
      operation: payload.operation,
      layer: payload.layer,
      timeIndex: payload.timeIndex ?? 0,
      compareLayer: payload.compareLayer ?? null,
      sceneVersion:
        payload.sceneVersion !== undefined && payload.sceneVersion !== null
          ? payload.sceneVersion
          : this.exactResultVersion,
    });
  }

  /**
   * Retry affordance (UI-004). An attempt that never received 202 is re-sent
   * with the same idempotency key; otherwise the app-level recompute provider
   * produces a fresh analysis request. Blocked under a site mismatch exactly
   * like commitEdits: an unacknowledged attempt must never be re-POSTed into
   * the guaranteed 409 (no zombie POST).
   */
  retryLast() {
    if (this.siteMismatch) {
      this.#emitSiteMismatch("retry blocked: reconnect with a fresh scenario first");
      return Promise.resolve(null);
    }
    if (!this.#ready) return Promise.resolve(null);
    if (this.lastAttempt && !this.lastAttempt.accepted) {
      // Re-send the never-acknowledged attempt with its original idempotency
      // key: safe to repeat, and a stale base version resolves through the
      // normal 409 recovery path.
      const attempt = this.lastAttempt;
      const run = () => this.#runAttempt(attempt);
      const result = this.#mutationQueue.then(run, run);
      this.#mutationQueue = result.then(
        () => {},
        () => {},
      );
      return result;
    }
    if (typeof this.provideRecompute === "function") {
      const request = this.provideRecompute();
      if (request) return this.requestRecompute(request);
    }
    return Promise.resolve(null);
  }

  /** Reset to the exact baseline (a versioned mutation like any other). */
  async reset({ label = "reset" } = {}) {
    if (this.siteMismatch) {
      // The pinned identity is stale for the whole scenario; reset cannot
      // repin it (only a fresh scenario can). Refuse locally — no zombie POST.
      throw new ApiClientError({
        code: "site_identity_mismatch",
        message:
          `reset refused: the scenario's pinned site identity no longer matches ` +
          `the live site; start a new scenario`,
      });
    }
    if (!this.#ready) {
      // Connecting (or the connect failed): held edits are local-only design
      // state — nothing was ever POSTed — and the server scene is still the
      // pristine baseline, so dropping them IS the reset. Without this they
      // would replay onto the baseline once the connection lands and apply
      // stale positions (R-2).
      const held = this.#connectBacklog.length;
      this.#releaseConnectBacklog();
      // Anything the in-flight connect is still fetching predates the reset;
      // invalidate it so it cannot announce around the reset (M-4 machinery).
      this.#invalidatePolls();
      if (held > 0) {
        this.#status("held_edits_discarded", { count: held, label });
      }
      return null;
    }
    // Any poll still in flight belongs to the pre-reset scene; invalidate it
    // or its completion would spuriously announce an applied result (M-4).
    this.#invalidatePolls();
    this.awaitingVersion = null;
    let body;
    try {
      body = await this.client.resetScenario(this.scenarioId, {
        baseSceneVersion: this.sceneVersion,
        idempotencyKey: this.client.nextIdempotencyKey("reset"),
      });
    } catch (error) {
      if (error instanceof ApiClientError && error.code === "site_identity_mismatch") {
        // The mismatch surfaced with the reset itself (not on an earlier
        // commit): record it exactly like a refused commit so every later
        // mutation fails fast and the banner renders, then rethrow so the
        // caller can undo any optimistic local clearing — the server scene
        // kept its trees, so the local design must stay visible too.
        this.#noteSiteMismatch(error);
      }
      throw error;
    }
    this.sceneVersion = Number(body.scene_version);
    this.exactResultVersion = Number(body.exact_result_version);
    const manifestUrl =
      body.result_manifest_url ??
      `/api/v1/scenarios/${this.scenarioId}/results/${this.exactResultVersion}`;
    const result = await this.#fetchResult(manifestUrl, this.sceneVersion);
    this.awaitingVersion = null;
    this.callbacks.onBaseline(result);
    this.callbacks.onAuthoritative({
      sceneVersion: this.sceneVersion,
      exactResultVersion: this.exactResultVersion,
      trees: [],
      reason: "reset",
    });
    this.#status("ready", { sceneVersion: this.sceneVersion, label });
    return result;
  }

  // -----------------------------------------------------------------------
  // Job polling with stale rejection (UI-003)
  // -----------------------------------------------------------------------

  /** Invalidate every in-flight poll: its result is superseded by design. */
  #invalidatePolls() {
    this.#pollGeneration += 1;
    if (this.#pollTimer !== null) {
      this.timers.clearTimeout(this.#pollTimer);
      this.#pollTimer = null;
    }
  }

  #startPoll(statusUrl, jobId, targetSceneVersion) {
    this.#invalidatePolls();
    const generation = this.#pollGeneration;
    this.#schedulePoll(statusUrl, jobId, targetSceneVersion, generation, 0);
  }

  #schedulePoll(statusUrl, jobId, targetSceneVersion, generation, delayMs, failureCount = 0) {
    if (this.#pollTimer !== null) this.timers.clearTimeout(this.#pollTimer);
    this.#pollTimer = this.timers.setTimeout(() => {
      this.#pollTimer = null;
      this.#poll(statusUrl, jobId, targetSceneVersion, generation).catch((error) => {
        if (generation !== this.#pollGeneration) return;
        // Transient transport failures (network drop, 5xx, 429) retry with
        // backoff instead of permanently killing the poll loop (L-4).
        if (failureCount < this.maxPollRetries && this.#isTransientError(error)) {
          const backoff = Math.min(
            this.pollIntervalMs * 2 ** (failureCount + 1),
            this.maxPollBackoffMs,
          );
          this.#schedulePoll(
            statusUrl,
            jobId,
            targetSceneVersion,
            generation,
            backoff,
            failureCount + 1,
          );
          return;
        }
        this.lastError = error;
        this.#status("failed", { error: this.#errorPayload(error) });
      });
    }, delayMs);
  }

  /** Transient transport failures (network drop, 5xx, 429) are retryable. */
  #isTransientError(error) {
    if (!(error instanceof ApiClientError)) return false;
    return (
      error.code === "network_error" ||
      error.code === "rate_limited" ||
      (error.status ?? 0) >= 500
    );
  }

  /**
   * Transport-class failure (network drop, 5xx — NOT 4xx, NOT 429): the
   * server may simply be asleep. These ride the cold-boot wake ladder.
   */
  #isTransportClass(error) {
    if (!(error instanceof ApiClientError)) return false;
    return error.code === "network_error" || (error.status ?? 0) >= 500;
  }

  async #poll(statusUrl, jobId, targetSceneVersion, generation) {
    if (generation !== this.#pollGeneration) return; // invalidated while scheduled
    const job = await this.client.getJob(statusUrl ?? jobId);
    if (generation !== this.#pollGeneration) return; // a newer job owns the result

    const status = job.status;
    if (status === "queued") {
      this.#status("queued", {
        jobId: job.job_id,
        queuePosition: job.queue_position,
        targetSceneVersion,
        mode: job.mode ?? null,
        // Honest ETA (server a83ac9f): queued bodies carry one only for
        // exact-lane reconcile jobs; null stays null — never a guess.
        etaSeconds: job.eta_seconds ?? null,
        etaBasis: job.eta_basis ?? null,
      });
      this.#schedulePoll(statusUrl, jobId, targetSceneVersion, generation, this.pollIntervalMs);
      return;
    }
    if (status === "running") {
      this.#status("running", {
        jobId: job.job_id,
        stage: job.stage,
        progress: job.progress,
        window: job.window,
        targetSceneVersion,
        // Recompute scope (full-tile vs windowed/local): surfaced mid-run so
        // the status line can be honest about a minutes-long full recompute.
        mode: job.mode ?? null,
        // Honest ETA: "history" = this scenario's own p80, "static" = the
        // disclosed fallback; null means not yet known early in a run.
        etaSeconds: job.eta_seconds ?? null,
        etaBasis: job.eta_basis ?? null,
      });
      this.#schedulePoll(statusUrl, jobId, targetSceneVersion, generation, this.pollIntervalMs);
      return;
    }

    if (status === "superseded") {
      // A newer edit owns the result; drop silently per contract.
      this.#activeSourceNodes = null;
      this.#status("superseded", { jobId: job.job_id, targetSceneVersion });
      return;
    }
    if (status === "failed" || status === "cancelled") {
      // Design state is untouched; keep the retry affordance (UI-004).
      this.#activeSourceNodes = null;
      this.lastError = new ApiClientError({
        code: job.error?.code ?? (status === "cancelled" ? "job_cancelled" : "job_failed"),
        message: job.error?.message ?? `job ended with status ${status}`,
      });
      this.#status("failed", { error: this.#errorPayload(this.lastError), jobId: job.job_id });
      return;
    }
    if (status !== "complete") return;

    if (Number(job.target_scene_version) < Number(this.awaitingVersion)) {
      // Out-of-order completion for an older scene version: ignore.
      return;
    }
    if (!job.result_manifest_url) return;

    // Executed impact plan (u-d4): executor-path jobs serve the per-node
    // stages the server actually routed; tree-only jobs carry none and the
    // UI keeps deriving from the document.
    const impactPlan = job.impact_plan ?? null;
    const result = await this.#fetchResult(job.result_manifest_url, this.awaitingVersion, generation);
    if (result === null) return; // stale or superseded while downloading
    // The decode awaits could have spanned a newer commit or a reset; only
    // the generation that started this poll may still announce it (M-4).
    if (generation !== this.#pollGeneration) return;
    this.exactResultVersion = result.manifest.scene_version;
    if (this.awaitingVersion === result.manifest.scene_version) {
      this.awaitingVersion = null;
    }
    const sourceNodes = this.#activeSourceNodes;
    this.#activeSourceNodes = null;
    this.callbacks.onPatchApplied({ ...result, jobId: job.job_id, impactPlan });
    this.#status("applied", {
      sceneVersion: result.manifest.scene_version,
      jobId: job.job_id,
      metrics: result.manifest.metrics,
      sourceNodes,
      impactPlan,
    });
  }

  /**
   * Download + validate one result. Returns null when the result turned stale
   * while downloading (UI-003) so callers drop it silently.
   */
  async #fetchResult(manifestUrl, awaitedVersion, generation = this.#pollGeneration) {
    const manifest = await this.client.getResultManifest(manifestUrl);
    if (generation !== this.#pollGeneration) return null;
    // Stale rejection: never apply a result older than the awaited version.
    if (Number(manifest.scene_version) !== Number(awaitedVersion)) return null;
    const payload = await this.client.getResultPayload(manifest.payload_url);
    if (generation !== this.#pollGeneration) return null;
    const decoded = await this.client.decodePatch(manifest, payload);
    // decodePatch itself awaits (checksum + decompression stream); the world
    // may have moved on again, so re-check before handing the result back.
    if (generation !== this.#pollGeneration) return null;
    if (Number(decoded.manifest.scene_version) !== Number(awaitedVersion)) return null;
    return decoded;
  }

  /**
   * Poll a job to a terminal state without applying anything (baseline job).
   * Transient GET failures (network, 5xx, 429) retry with the same backoff
   * ladder as the poll loop instead of failing the whole connect: a cold-site
   * baseline solve runs for minutes and a single 429 (shared transport
   * bucket) must not kill the boot.
   */
  async #awaitJob(statusUrl, jobId) {
    let failures = 0;
    for (;;) {
      let job;
      try {
        job = await this.client.getJob(statusUrl ?? jobId);
      } catch (error) {
        if (failures < this.maxPollRetries && this.#isTransientError(error)) {
          const backoff = Math.min(
            this.pollIntervalMs * 2 ** (failures + 1),
            this.maxPollBackoffMs,
          );
          await new Promise((resolve) => this.timers.setTimeout(resolve, backoff));
          failures += 1;
          continue;
        }
        throw error;
      }
      failures = 0; // a healthy poll re-arms the full retry budget
      if (TERMINAL_JOB_STATUSES.has(job.status)) {
        if (job.status === "failed") {
          throw new ApiClientError({
            code: job.error?.code ?? "job_failed",
            message: job.error?.message ?? "baseline job failed",
          });
        }
        return job;
      }
      await new Promise((resolve) => this.timers.setTimeout(resolve, this.pollIntervalMs));
    }
  }

  #errorPayload(error) {
    if (error instanceof ApiClientError) {
      // retry_after_ms rides only rate_limited payloads — the cooldown frame
      // needs the server's own wait hint; every other failure ignores it.
      const retryAfterMs = Number(error.details?.retry_after_ms);
      return {
        code: error.code,
        message: error.message,
        field: error.field,
        ...(Number.isFinite(retryAfterMs) && retryAfterMs > 0 ? { retryAfterMs } : {}),
      };
    }
    return { code: "unexpected_error", message: String(error?.message ?? error) };
  }
}
