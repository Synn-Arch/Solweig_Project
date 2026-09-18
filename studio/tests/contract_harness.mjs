// SPDX-License-Identifier: GPL-3.0-only
//
// Shared in-memory contract-server harness for ExactSession tests (moved
// verbatim from exact_session.test.mjs when U-D2 added view/mismatch suites,
// so several test files can share one fake server without re-running each
// other's tests through cross-imports).
//
// Canned scenario/edit/job/result/capabilities responses driven by a manual
// clock. No live server is involved.

export const SCENARIO = "scn_test_01";
export const ROWS = 8;
export const COLS = 8;
export const TIME_STEPS = 3;
export const FULL_WINDOW = { row_start: 0, row_stop: ROWS, col_start: 0, col_stop: COLS };

export async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function makePlane(salt) {
  const plane = new Float32Array(TIME_STEPS * ROWS * COLS);
  let index = 0;
  for (let time = 0; time < TIME_STEPS; time += 1) {
    for (let row = 0; row < ROWS; row += 1) {
      for (let col = 0; col < COLS; col += 1) {
        plane[index] = 40 + salt + time * 0.01 - row * 0.001 - col * 0.001;
        index += 1;
      }
    }
  }
  return plane;
}

export async function encodeResult({
  sceneVersion,
  window = FULL_WINDOW,
  timeIndices = [0, 1, 2],
  values,
  limitations,
}) {
  const rows = window.row_stop - window.row_start;
  const cols = window.col_stop - window.col_start;
  const raw = new Uint8Array(values.buffer.slice(0));
  const hex = await sha256Hex(raw);
  return {
    manifest: {
      schema_version: 1,
      scenario_id: SCENARIO,
      scene_version: sceneVersion,
      exact: true,
      model_version: "solweig-gpu-2.0.0+incremental.1",
      site_cache_version: "campus-1km-v1:cache-1",
      window,
      time_indices: timeIndices,
      variables: [
        { name: "utci", dtype: "float32", shape: [timeIndices.length, rows, cols], nodata: "nan" },
      ],
      payload_url: `/api/v1/scenarios/${SCENARIO}/results/${sceneVersion}/payload`,
      compression: "identity",
      checksum: `sha256:${hex}`,
      metrics: {
        duration_ms: 1200,
        mean_utci_delta_c: -0.84,
        peak_utci_delta_c: -3.42,
        improved_area_m2: 8940,
        window_fraction: (rows * cols) / (ROWS * COLS),
      },
      limitations:
        limitations ??
        [
          "Tree-induced local wind-field changes are not recomputed.",
          "Results are exact within the published window; cells outside it are unchanged from the baseline scene.",
        ],
    },
    payloadBytes: raw,
    etag: `"sha256-${hex}"`,
  };
}

export function apiTree(treeId, { u = 0.5, v = 0.5 } = {}) {
  return {
    tree_id: treeId,
    component_type: "broad_canopy",
    u,
    v,
    height_m: 18,
    canopy_diameter_m: 11,
    trunk_ratio: 0.25,
    transmissivity: 0.03,
    phenology: "deciduous",
    metadata: { label: `Tree ${treeId}` },
  };
}

export function addEdit(treeId) {
  return [{ operation: "add", tree: apiTree(treeId) }];
}

/**
 * Minimal in-memory implementation of the API contract. Test cases hook
 * `onEdit`, `onManifest`, and job scripts to produce conflicts, failures,
 * and stale results.
 */
export class FakeContractServer {
  constructor({ jobScript = ["queued", "running", "complete"] } = {}) {
    this.calls = [];
    this.sceneVersion = 0;
    this.exactResultVersion = 0;
    this.trees = [];
    this.appliedEdits = 0;
    this.jobs = new Map();
    this.results = new Map();
    this.stored = new Map(); // idempotency key -> {status, body}
    /** Drops the create RESPONSE after applying it (network loss shape). */
    this.loseResponseCreates = 0;
    this.jobScript = jobScript;
    this.onEdit = null;
    this.onUniversalEdit = null; // async (server, body, key) => Response | null
    this.onManifest = null;
    this.onReset = null; // async (body, idempotencyKey) => Response | null
    this.loseResponseEdits = 0;
    this.scenarioGate = null; // Promise awaited before GET /scenarios responds
    this.jobGetFailures = 0; // transient GET /jobs failures to simulate
    this.payloadGateFor = null; // (sceneVersion) => Promise|null to delay payloads
    // Universal-edit transport (u-d4): the last accepted batch, and a hook
    // to serve an executed impact_plan on the completing job body.
    this.universalEdits = [];
    this.impactPlanFor = null; // (sceneVersion) => plan object | null
    // HARNESS ADAPTATION (U-D2, reported): connect() now fetches the capability
    // document, so the fake contract server serves the checked-in snapshot of
    // the real server document (assets/capabilities_snapshot.json).
    this.capabilitiesStatus = 200;
    this.capabilitiesDocument = null; // defaults to the checked-in snapshot
    // View-only operation (UEDIT-007) canned responses/refusals.
    this.viewHandler = null; // async (body) => Response | null
  }

  async seedBaseline() {
    const result = await encodeResult({ sceneVersion: 0, values: makePlane(0) });
    this.results.set(0, result);
    return result;
  }

  #json(status, body, headers = {}) {
    return new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json", ...headers },
    });
  }

  /**
   * Universal edit batch (POST /edits/universal): same job/versioning
   * machinery as tree edits — one scene-version bump, one job, an
   * idempotent-replay record — plus `transport: "universal-edits-v1"` and
   * an optional executed impact_plan served on the completing job body.
   */
  async #applyUniversalEdits(body, idempotencyKey) {
    this.universalEdits.push(body);
    this.sceneVersion += 1;
    const jobId = `job_${this.sceneVersion}`;
    const impactPlan = this.impactPlanFor ? this.impactPlanFor(this.sceneVersion) : null;
    this.jobs.set(jobId, {
      job_id: jobId,
      scenario_id: SCENARIO,
      target_scene_version: this.sceneVersion,
      script: [...this.jobScript],
      step: 0,
      impactPlan,
    });
    const window = { row_start: 2, row_stop: 6, col_start: 3, col_stop: 7 };
    const values = new Float32Array(1 * 4 * 4).map(
      (_, index) => 35 + this.sceneVersion + index * 0.01,
    );
    const result = await encodeResult({
      sceneVersion: this.sceneVersion,
      window,
      timeIndices: [12],
      values,
    });
    this.results.set(this.sceneVersion, result);
    const responseBody = {
      scenario_id: SCENARIO,
      scene_version: this.sceneVersion,
      exact_result_version: this.exactResultVersion,
      status: "refining",
      job_id: jobId,
      transport: "universal-edits-v1",
      coalescing_window_ms: 500,
      status_url: `/api/v1/jobs/${jobId}`,
    };
    const response = {
      status: 202,
      body: responseBody,
      headers: { ETag: `"scene-version-${this.sceneVersion}"` },
    };
    this.stored.set(idempotencyKey, response);
    return response;
  }

  async #applyEdits(body, idempotencyKey) {
    this.appliedEdits += 1;
    for (const item of body.edits) {
      if (item.operation === "delete") {
        this.trees = this.trees.filter((tree) => tree.tree_id !== item.tree_id);
      } else {
        this.trees = [...this.trees.filter((tree) => tree.tree_id !== item.tree.tree_id), item.tree];
      }
    }
    this.sceneVersion += 1;
    const jobId = `job_${this.sceneVersion}`;
    this.jobs.set(jobId, {
      job_id: jobId,
      scenario_id: SCENARIO,
      target_scene_version: this.sceneVersion,
      script: [...this.jobScript],
      step: 0,
    });
    const window = { row_start: 2, row_stop: 6, col_start: 3, col_stop: 7 };
    const values = new Float32Array(1 * 4 * 4).map(
      (_, index) => 35 + this.sceneVersion + index * 0.01,
    );
    const result = await encodeResult({
      sceneVersion: this.sceneVersion,
      window,
      timeIndices: [12],
      values,
    });
    this.results.set(this.sceneVersion, result);
    const responseBody = {
      scenario_id: SCENARIO,
      scene_version: this.sceneVersion,
      exact_result_version: this.exactResultVersion,
      status: "refining",
      job_id: jobId,
      coalescing_window_ms: 500,
      status_url: `/api/v1/jobs/${jobId}`,
    };
    const response = {
      status: 202,
      body: responseBody,
      headers: { ETag: `"scene-version-${this.sceneVersion}"` },
    };
    this.stored.set(idempotencyKey, response);
    return response;
  }

  fetchImpl = async (url, init = {}) => {
    const method = init.method ?? "GET";
    const path = new URL(url, "http://server.test").pathname;
    this.calls.push({ method, path, headers: init.headers ?? {}, body: init.body ?? null });
    const idempotencyKey = init.headers?.["Idempotency-Key"] ?? null;

    if (method === "GET" && (path === "/api/v1/capabilities" || path === "/capabilities")) {
      if (this.capabilitiesStatus !== 200) {
        return this.#json(this.capabilitiesStatus, {
          error: { code: "not_found", message: "capabilities unavailable" },
        });
      }
      const document = this.capabilitiesDocument ?? (await importCapabilitiesSnapshot());
      return this.#json(200, document);
    }

    if (method === "POST" && path === "/api/v1/scenarios") {
      const body = {
        scenario_id: SCENARIO,
        site_id: "campus-1km-v1",
        scene_version: 0,
        exact_result_version: 0,
        status: "exact",
        trees: [],
        result_manifest_url: `/api/v1/scenarios/${SCENARIO}/results/0`,
      };
      // Idempotent replay, same contract as the edits routes: a re-POST with
      // a known key re-serves the stored response instead of minting again.
      if (idempotencyKey && this.stored.has(idempotencyKey)) {
        const stored = this.stored.get(idempotencyKey);
        return this.#json(stored.status, stored.body, stored.headers);
      }
      if (idempotencyKey) this.stored.set(idempotencyKey, { status: 201, body, headers: {} });
      if (this.loseResponseCreates > 0) {
        this.loseResponseCreates -= 1;
        throw new TypeError("simulated drop after the create was applied");
      }
      return this.#json(201, body);
    }

    if (method === "POST" && path === `/api/v1/scenarios/${SCENARIO}/views`) {
      if (this.viewHandler) {
        const override = await this.viewHandler(JSON.parse(init.body ?? "{}"));
        if (override) return override;
      }
      return this.#json(200, {
        operation: "select_layer",
        scenario_id: SCENARIO,
        scene_version: this.sceneVersion,
        job_enqueued: false,
        zero_scientific_jobs: true,
        result_manifest_url: `/api/v1/scenarios/${SCENARIO}/results/${this.exactResultVersion}`,
        result_payload_url: `/api/v1/scenarios/${SCENARIO}/results/${this.exactResultVersion}/payload`,
        layer: { name: "utci", dtype: "float32", shape: [3, ROWS, COLS], nodata: "nan" },
        legend: { min: 18.2, max: 41.5, mean: 27.9, count: 64 },
        cached_time_indices: [0, 1, 2],
      });
    }

    if (method === "GET" && path === `/api/v1/scenarios/${SCENARIO}`) {
      if (this.scenarioGate) await this.scenarioGate;
      const active = [...this.jobs.values()].find(
        (job) => job.step < job.script.length && job.script[job.step] !== "complete",
      );
      return this.#json(200, {
        scenario_id: SCENARIO,
        site_id: "campus-1km-v1",
        scene_version: this.sceneVersion,
        exact_result_version: this.exactResultVersion,
        status: this.sceneVersion === this.exactResultVersion ? "exact" : "refining",
        trees: this.trees,
        active_job_id: active?.job_id ?? null,
        result_manifest_url: `/api/v1/scenarios/${SCENARIO}/results/${this.exactResultVersion}`,
        model_scope: {
          model_version: "solweig-gpu-2.0.0+incremental.1",
          site_cache_version: "campus-1km-v1:cache-1",
          limitations: ["Tree-induced local wind-field changes are not recomputed."],
        },
      });
    }

    if (method === "POST" && path === `/api/v1/scenarios/${SCENARIO}/edits`) {
      const stored = this.stored.get(idempotencyKey);
      if (
        stored &&
        stored.body &&
        JSON.stringify(JSON.parse(init.body)) ===
          JSON.stringify(stored.requestBody ?? JSON.parse(init.body))
      ) {
        // idempotent replay of an acknowledged mutation
        return this.#json(stored.status, stored.body, stored.headers);
      }
      if (this.onEdit) {
        const override = await this.onEdit(this, JSON.parse(init.body), idempotencyKey);
        if (override) return override;
      }
      const response = await this.#applyEdits(JSON.parse(init.body), idempotencyKey);
      response.requestBody = JSON.parse(init.body);
      if (this.loseResponseEdits > 0) {
        this.loseResponseEdits -= 1;
        throw new TypeError("simulated connection drop after the server applied the edit");
      }
      return this.#json(response.status, response.body, response.headers);
    }

    if (method === "POST" && path === `/api/v1/scenarios/${SCENARIO}/edits/universal`) {
      const stored = this.stored.get(idempotencyKey);
      if (
        stored &&
        stored.body &&
        JSON.stringify(JSON.parse(init.body)) ===
          JSON.stringify(stored.requestBody ?? JSON.parse(init.body))
      ) {
        // idempotent replay of an acknowledged universal mutation
        return this.#json(stored.status, stored.body, stored.headers);
      }
      if (this.onUniversalEdit) {
        const override = await this.onUniversalEdit(this, JSON.parse(init.body), idempotencyKey);
        if (override) return override;
      }
      const response = await this.#applyUniversalEdits(JSON.parse(init.body), idempotencyKey);
      response.requestBody = JSON.parse(init.body);
      if (this.loseResponseEdits > 0) {
        this.loseResponseEdits -= 1;
        throw new TypeError(
          "simulated connection drop after the server applied the universal edit",
        );
      }
      return this.#json(response.status, response.body, response.headers);
    }

    if (method === "POST" && path === `/api/v1/scenarios/${SCENARIO}/reset`) {
      if (this.onReset) {
        const override = await this.onReset(JSON.parse(init.body ?? "{}"), idempotencyKey);
        if (override) return override;
      }
      this.sceneVersion += 1;
      this.exactResultVersion = this.sceneVersion;
      this.trees = [];
      const result = await encodeResult({ sceneVersion: this.sceneVersion, values: makePlane(0) });
      this.results.set(this.sceneVersion, result);
      return this.#json(
        200,
        {
          scenario_id: SCENARIO,
          scene_version: this.sceneVersion,
          exact_result_version: this.exactResultVersion,
          status: "exact",
          trees: [],
          result_manifest_url: `/api/v1/scenarios/${SCENARIO}/results/${this.exactResultVersion}`,
        },
        { ETag: `"scene-version-${this.sceneVersion}"` },
      );
    }

    const jobMatch = path.match(/^\/api\/v1\/jobs\/(.+)$/);
    if (method === "GET" && jobMatch) {
      if (this.jobGetFailures > 0) {
        this.jobGetFailures -= 1;
        throw new TypeError("simulated ECONNRESET while polling");
      }
      const job = this.jobs.get(jobMatch[1]);
      if (!job) {
        return this.#json(404, { error: { code: "job_not_found", message: "no such job" } });
      }
      const status = job.script[Math.min(job.step, job.script.length - 1)];
      const body = {
        job_id: job.job_id,
        scenario_id: SCENARIO,
        target_scene_version: job.target_scene_version,
        status,
      };
      if (status === "queued") {
        body.queue_position = 1;
        body.progress = null;
        // Honest ETA (server a83ac9f): QUEUED bodies carry eta only for
        // exact-lane reconcile jobs; ordinary queued jobs are (null, null).
        body.eta_seconds = job.etaSeconds ?? null;
        body.eta_basis = job.etaBasis ?? null;
      } else if (status === "running") {
        body.stage = "time_loop";
        body.progress = { completed_time_steps: 9, total_time_steps: 24 };
        body.mode = job.mode ?? "local";
        body.window = { row_start: 2, row_stop: 6, col_start: 3, col_stop: 7 };
        // Honest ETA while running: null until the mode is known early in a
        // run; "history" once this scenario's own p80 lands, "static" as the
        // disclosed full-tile fallback.
        body.eta_seconds = job.etaSeconds ?? null;
        body.eta_basis = job.etaBasis ?? null;
      } else if (status === "complete") {
        body.result_manifest_url = `/api/v1/scenarios/${SCENARIO}/results/${job.target_scene_version}`;
        body.metrics = { duration_ms: 1200 };
        // Executed impact plan (u-d4): served on terminal job bodies only.
        if (job.impactPlan) body.impact_plan = job.impactPlan;
        this.exactResultVersion = job.target_scene_version;
      } else if (status === "failed") {
        body.error = { code: "job_failed", message: "worker exploded" };
      }
      job.step += 1;
      return this.#json(200, body);
    }

    const manifestMatch = path.match(/^\/api\/v1\/scenarios\/[^/]+\/results\/(\d+)$/);
    if (method === "GET" && manifestMatch) {
      const record = this.results.get(Number(manifestMatch[1]));
      if (!record) {
        return this.#json(404, { error: { code: "result_not_ready", message: "not published" } });
      }
      if (this.onManifest) {
        const override = await this.onManifest(Number(manifestMatch[1]), record);
        if (override) return this.#json(200, override);
      }
      return this.#json(200, record.manifest);
    }

    const payloadMatch = path.match(/^\/api\/v1\/scenarios\/[^/]+\/results\/(\d+)\/payload$/);
    if (method === "GET" && payloadMatch) {
      const record = this.results.get(Number(payloadMatch[1]));
      if (!record) {
        return this.#json(404, { error: { code: "result_not_ready", message: "not published" } });
      }
      if (this.payloadGateFor) {
        const gate = this.payloadGateFor(Number(payloadMatch[1]));
        if (gate) await gate;
      }
      return new Response(record.payloadBytes, {
        status: 200,
        headers: {
          "content-type": "application/vnd.solweig.patch+identity",
          etag: record.etag,
          "x-solweig-scene-version": String(record.manifest.scene_version),
          "x-solweig-schema-version": "1",
        },
      });
    }

    return this.#json(404, { error: { code: "not_found", message: path } });
  };

  editPosts() {
    return this.calls.filter((call) => call.path.endsWith("/edits") && call.method === "POST");
  }

  universalEditPosts() {
    return this.calls.filter(
      (call) => call.path.endsWith("/edits/universal") && call.method === "POST",
    );
  }
}

export function makeFakeClock() {
  const tasks = [];
  let nextId = 1;
  async function settle() {
    for (let round = 0; round < 12; round += 1) {
      await new Promise((resolve) => setImmediate(resolve));
    }
  }
  return {
    timers: {
      setTimeout(fn, _ms) {
        const task = { id: nextId, fn };
        nextId += 1;
        tasks.push(task);
        return task.id;
      },
      clearTimeout(id) {
        const index = tasks.findIndex((task) => task.id === id);
        if (index >= 0) tasks.splice(index, 1);
      },
    },
    pendingCount: () => tasks.length,
    async tick(count = 1) {
      for (let index = 0; index < count; index += 1) {
        const task = tasks.shift();
        if (!task) break;
        task.fn();
        await settle();
      }
      await settle();
    },
    settle,
  };
}

let snapshotPromise = null;
function importCapabilitiesSnapshot() {
  snapshotPromise ??= import("../assets/capabilities_snapshot.json", {
    with: { type: "json" },
  }).then((module) => module.default);
  return snapshotPromise;
}
