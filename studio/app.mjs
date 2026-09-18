// SPDX-License-Identifier: GPL-3.0-only
//
// SOLWEIG Studio application shell.
//
// Two analysis paths coexist (docs/incremental_design_tool/adr/0002):
//
//   * Local preview worker (`solver_worker.mjs`) — immediate, non-scientific,
//     always client-side. It drives ghost trees, drag previews, and the
//     approximate thermal overlay shown while the exact result is stale.
//   * Exact session (`exact_session.mjs` + `api_client.mjs`) — activated with
//     `?api=<base-url>`. Committed interactions send exactly one POST /edits,
//     poll the job, and apply the validated patch through the renderer's
//     texSubImage2D region path. Without `?api=` the studio runs fully
//     offline exactly as before.
//
// Acceptance gates covered here: UI-001 (one edit per committed interaction),
// UI-002 (debounced sliders), UI-003 (stale results ignored), UI-004 (failure
// preserves design state), UI-005 (model scope disclosed).
//
// U-D2 (capability-driven UI): at session start the app fetches the capability
// document (GET /api/v1/capabilities) and BUILDS the edit UI from it — editor
// list, property bounds, class fences, variable lists, and view layers. The
// only checked-in vocabulary is the offline demo's document snapshot
// (assets/capabilities_snapshot.json, generated from the server); connected
// mode never falls back to it. UEDIT-009 renders the changed/reused/
// recomputed stage classification per commit; UEDIT-007 view-only operations
// run against the published result with a zero-jobs affordance; a 409
// site_identity_mismatch fails visibly with pinned vs current identity.

import { ApiClient, ApiClientError, resolveApiBase } from "./api_client.mjs";
import { ExactSession } from "./exact_session.mjs";
import { RealtimeClient, RealtimeAdmissionError } from "./realtime_client.mjs";
import {
  buildingOperation,
  frameFromGeometry,
  frameFromSpans,
  frameResidualOk,
  operationSummary,
  presenceFromOperations,
  solveFrameFromSamples,
  treeOperation,
  uvToWorld,
  worldToUv,
  worldTree,
} from "./op_builder.mjs";
import {
  applyRealtimeBadge,
  collabSubscriptionAlive,
  owedDurationText,
  realtimeBadgeState,
} from "./realtime_badge.mjs";
import {
  adapterPresentation,
  impactPanelModel,
  parseCapabilityDocument,
  propertyControls,
  viewErrorModel,
  viewPanelModel,
  viewResultModel,
} from "./capabilities.mjs";
import {
  DraftValidationError,
  EDIT_TRANSPORTS,
  FamilyTransportError,
  composeFamilyEdit,
  editTransport,
} from "./family_edits.mjs";
import {
  BUILDING_PRESETS,
  TREE_PRESETS,
  accumulateDirtyRect,
  apiObjectToTree,
  apiValuesToBuilding,
  buildingToWorld,
  canvasToUv,
  cloneBuilding,
  cloneTree,
  computeExactMetrics,
  createPropertyEditor,
  dirtyRectForBuildingEdit,
  dirtyRectForEdit,
  distancePointToSegment,
  editItemForChange,
  exactBadgeForConnection,
  findTreeAtCanvasPoint,
  formatInteger,
  formatSignedTemperature,
  gridWindowToRect,
  isCurrentWorkerMessage,
  isUvInsideSite,
  makeBuilding,
  makeTree,
  rectToGridWindow,
  uvToCanvas,
  treeToApiObject,
  unionRects,
  worldToUv as sceneWorldToUv,
} from "./model.mjs";
import { SceneRenderer } from "./renderer.mjs";
import {
  applyExactPatch,
  applyPatch,
  cubeTimePlane,
  decodeAnalysisFixture,
} from "./solver_kernel.mjs";

// ---------------------------------------------------------------------------
// Honest-ETA status copy (server a83ac9f): QUEUED/RUNNING job bodies carry
// eta_seconds (int | null) and eta_basis ("history" | "static" | null).
// null means honestly unknown — these builders never invent a number. They
// are pure and live OUTSIDE the DOM guard below so the job-status tests can
// pin the exact user-facing strings.
// ---------------------------------------------------------------------------

/** "about 80 s" below 90 s (rounded to the nearest 10), otherwise minutes. */
export function formatEtaWait(seconds) {
  if (seconds < 90) return `about ${Math.round(seconds / 10) * 10} s`;
  return `about ${Math.max(1, Math.round(seconds / 60))} min`;
}

/**
 * The ETA clause for a running exact job, or null when the server sent no
 * estimate (mode not yet known early in a run) — the caller then renders no
 * clause at all. "history" is this scenario's own past runs; anything else
 * the server discloses reads as the deployment fallback.
 */
function etaClause(etaSeconds, etaBasis) {
  if (!Number.isFinite(etaSeconds) || etaSeconds <= 0) return null;
  if (etaBasis === "history") {
    return `${formatEtaWait(etaSeconds)} left (from past runs)`;
  }
  return `typically ${formatEtaWait(etaSeconds)}`;
}

/**
 * The SSE exact_progress event (server wave ①) folded into a small view
 * state. Pure so the tests can pin the countdown contract:
 *
 * - A TERMINAL status retires the countdown for THAT job (null) — an
 *   unrelated job's terminal event must not clear another job's view.
 * - The deadline is `now + eta_seconds` and is recomputed ONLY when the eta
 *   itself changed. Re-emitted identical etas (dispatch → mode-known often
 *   restate the same estimate) therefore keep the original deadline, so the
 *   displayed remaining time is monotone non-increasing between genuine eta
 *   revisions — it never jolts back up.
 * - No eta yet (early run, mode unknown) → deadlineMs null; the caller
 *   renders presence of the job without inventing a number.
 */
const TERMINAL_JOB_STATUSES = new Set([
  "complete",
  "failed",
  "superseded",
  "cancelled",
  "no-op",
]);

export function exactProgressState(prev, event, nowMs) {
  const jobId = event?.job_id !== undefined && event?.job_id !== null
    ? String(event.job_id)
    : "";
  if (!jobId) return prev ?? null;
  const status = String(event.status ?? "");
  if (TERMINAL_JOB_STATUSES.has(status)) {
    return prev && prev.jobId === jobId ? null : prev ?? null;
  }
  const etaSeconds = Number(event.eta_seconds);
  const eta = Number.isFinite(etaSeconds) && etaSeconds > 0 ? etaSeconds : null;
  const sameJob = prev !== null && prev.jobId === jobId;
  const deadlineMs = sameJob && eta !== null && prev.etaSeconds === eta
    ? prev.deadlineMs
    : eta !== null
      ? nowMs + eta * 1000
      : null;
  const queuePosition = Number(event.queue_position);
  return {
    jobId,
    status,
    targetRevision: Number(event.target_revision) || 0,
    etaSeconds: eta,
    etaBasis: typeof event.eta_basis === "string" ? event.eta_basis : null,
    queuePosition: Number.isFinite(queuePosition) ? queuePosition : null,
    deadlineMs,
  };
}

/**
 * The countdown clause for the class-badge age span, or null when there is
 * nothing to show. Past the deadline the countdown NEVER goes negative or
 * silently vanishes — it says the estimate was exceeded (honest), with the
 * basis distinction preserved.
 */
export function exactProgressClause(progress, nowMs) {
  if (!progress || !progress.deadlineMs) return null;
  const remainingS = (progress.deadlineMs - nowMs) / 1000;
  if (remainingS <= 0) {
    return progress.etaBasis === "history"
      ? "taking longer than past runs"
      : "taking longer than usual";
  }
  const basis = progress.etaBasis === "history" ? " (from past runs)" : "";
  return `${formatEtaWait(remainingS)} left${basis}`;
}

/**
 * Split the session's observed roster into server-confirmed live actors and
 * "earlier this session" ones. Pure; the heartbeat roster (server truth)
 * leads, first-seen ordinals stay stable via actorDisplayName.
 */
export function rosterView(observedRoster, liveRoster) {
  const observed = (observedRoster ?? []).map(String);
  const live = (liveRoster ?? []).map(String).filter(Boolean);
  const liveSet = new Set(live);
  return {
    live,
    earlier: observed.filter((id) => !liveSet.has(id)),
  };
}

/** The #jobStatusDetail line for a RUNNING exact job (detail = poll status). */
export function exactRunningStatusText({ mode, stage, progress, etaSeconds, etaBasis } = {}) {
  const clause = etaClause(etaSeconds, etaBasis);
  // Full-tile recomputes run for minutes: promise that instead of implying a
  // seconds-long window patch. No invented progress — the duration promise
  // (and, when the server has one, its honest ETA) is the signal.
  if (mode === "full") {
    // One duration clause, never two stacked dashes: with the server's eta
    // the estimate REPLACES the generic promise; without it ("a few
    // minutes") the promise is all we honestly have.
    if (!clause) return "Recalculating the whole map — a few minutes";
    return `Recalculating the whole map — ${clause}`;
  }
  const progressText = progress
    ? `${progress.completed_time_steps}/${progress.total_time_steps} steps`
    : "starting up";
  const base = `Updating the map · ${progressText}`;
  return clause ? `${base} · ${clause}` : base;
}

/** The #jobStatusDetail line for a QUEUED exact job (detail = poll status). */
export function exactQueuedStatusText({
  queuePosition,
  sceneVersion,
  etaSeconds,
  etaBasis,
} = {}) {
  // Queued bodies carry an eta only for exact-lane reconcile jobs — a full
  // recompute queued behind the current run: name that wait plainly, with
  // the same basis disclosure the running line uses ("history" vs the
  // deployment fallback). The scene version is accepted for signature
  // compatibility but renders only where it belongs: tooltips.
  if (Number.isFinite(etaSeconds) && etaSeconds > 0) {
    const wait =
      etaBasis === "history"
        ? `${formatEtaWait(etaSeconds)} (from past runs)`
        : `typically ${formatEtaWait(etaSeconds)}`;
    return `Waiting its turn — ${wait}`;
  }
  return queuePosition ? `Queued — position ${queuePosition}` : "Queued";
}

// ---------------------------------------------------------------------------
// Catch-up copy (a visitor joining a settled shared world, or a reconnect).
// The local session never ran an exact job on this path, so every status
// surface must speak from the workspace's revision triple instead of the boot
// defaults ("No exact job yet", "receipt pending", ...). Same rule as the ETA
// builders: pure, node-testable, and never inventing a number — the client
// has no timestamp for a settled result, so no "settled Ns" is fabricated.
// ---------------------------------------------------------------------------

const RESULT_CLASS_WORDS = {
  fast_exact: { word: "Fast precise", symbol: "●", rank: 2 },
  fast_qualified: { word: "Close estimate", symbol: "◐", rank: 1 },
  visual_pending: { word: "Preview", symbol: "◌", rank: 0 },
  exact_reconciled: { word: "Precise", symbol: "■", rank: 3 },
};

// The owed Verify copy: plain words only — the revision gap lives in
// tooltips (rail triple, badge title), never in these visible lines.
const verifyOwedLampLine = () => `Preview shown — the precise map is being prepared.`;
const verifyOwedDetailLine = () => `Preview · precise map pending`;

/**
 * The owed Verify strings. ONE source for both the live path
 * (renderVerifyOwed, driven by fast frames) and the catch-up path (driven by
 * the revision triple) — every surface echoes the same words. The revision
 * triple stays in the signature (callers pass it); the copy itself carries
 * no revision numbers.
 */
export function verifyOwedTexts() {
  return {
    lamp: verifyOwedLampLine(),
    detail: verifyOwedDetailLine(),
  };
}

/**
 * The #jobStatusDetail line for a caught-up session with no local exact job:
 * the workspace-level exact state from the revision triple. Settled mirrors
 * markVerifySettled's "Precise map · ready"; owed reuses the verify-owed
 * line; a pristine world (W0) whose verified baseline landed is precise
 * as-is. Returns null while nothing is known — the caller keeps its current
 * honest line.
 */
export function catchUpExactStatusText(revisions, { hasBaselineExact = false } = {}) {
  const { workspaceRevision, fastRevision, exactRevision } = revisions ?? {};
  const workspace = Number(workspaceRevision ?? 0);
  const exact = Number(exactRevision ?? 0);
  if (workspace > 0) {
    return exact >= workspace ? `Precise map · ready` : verifyOwedDetailLine();
  }
  return hasBaselineExact ? "Precise map · ready" : null;
}

/**
 * Honest five-stage bar for a caught-up session: the replayed operations are
 * already receipted and canonical (the journal IS the receipt), the fast lane
 * published R{F}, and Verify is settled or owed. Returns null for a pristine
 * world (W0 — no receipts to claim) or unknown revisions; the caller leaves
 * the bar's current state alone.
 */
export function catchUpPipelineLamps(revisions) {
  const { workspaceRevision, fastRevision, exactRevision } = revisions ?? {};
  const workspace = Number(workspaceRevision ?? 0);
  const fast = Number(fastRevision ?? 0);
  const exact = Number(exactRevision ?? 0);
  if (!(workspace > 0)) return null;
  const owed = exact < workspace;
  const smalls = {
    stamp: `receipted · R${workspace}`,
    share: `applied · R${workspace}`,
    verify: owed ? verifyOwedLampLine() : `precise · R${exact}`,
  };
  if (fast > 0) smalls.paint = `fast · R${fast}`;
  return {
    // Owed settles the bar at Paint (Verify carries the owed ring instead of
    // a done check); settled completes it through Verify.
    focus: owed ? "paint" : "verify",
    verifyOwed: owed,
    smalls,
  };
}

/**
 * Result-class badge override for a pristine shared world: W0 means zero
 * edits, so the verified baseline result IS the precise picture — "Precise",
 * never "Preview". Any class-bearing event or non-pristine world returns
 * null (the class machinery owns the badge).
 */
export function catchUpBadgeView(revisions, { hasBaselineExact = false } = {}) {
  const workspace = Number(revisions?.workspaceRevision ?? 0);
  if (workspace !== 0 || !hasBaselineExact) return null;
  return { word: "Precise", glyph: "■", state: "exact_reconciled" };
}

/**
 * The #exactBadge class echo. Catch-up fix: with no class-bearing frame but
 * KNOWN revisions, "Connecting" overstates transport (the subscription is
 * live; the pixels are a preview) — echo Preview instead. The pristine
 * baseline-exact catch-up echoes Exact like any settled truth.
 */
export function exactBadgeEcho({
  rawClass = null,
  superseded = false,
  revisionsKnown = false,
  catchUpExact = false,
} = {}) {
  if (catchUpExact) return { text: "Exact", state: "exact" };
  if (!rawClass) {
    return revisionsKnown
      ? { text: "Preview", state: "preview" }
      : { text: "Connecting", state: "connecting" };
  }
  if (superseded || !RESULT_CLASS_WORDS[rawClass] || rawClass === "visual_pending") {
    return { text: "Preview", state: "preview" };
  }
  if (rawClass === "fast_qualified") return { text: "Estimate", state: "preview" };
  return { text: "Exact", state: "exact" };
}

/**
 * The rail's OFFLINE note is the offline-mode class lamp: a connected-mode
 * session (connecting or connected) must never render the OFFLINE word. A
 * failed connect genuinely falls back to the local preview — the note shows.
 */
export function offlineRailNoteHidden(connectionKind) {
  return connectionKind === "connecting" || connectionKind === "connected";
}

/**
 * Boot transport-pill phase: connected mode honestly says "Connecting" (the
 * subscription does not exist yet); only offline mode owns the "Offline" word.
 */
export function bootTransportPhase(connected) {
  return connected ? "connecting" : "offline";
}

/**
 * The rail's owed chip: fully hidden (not "+0") unless the exact lane
 * genuinely trails the workspace.
 */
export function railOwedVisibility(revisions) {
  const { workspaceRevision, exactRevision } = revisions ?? {};
  const gap =
    Number.isFinite(exactRevision) && Number.isFinite(workspaceRevision)
      ? workspaceRevision - exactRevision
      : 0;
  return {
    hidden: !(gap > 0),
    text: gap > 0 ? `updating +${gap}` : null,
  };
}

// ---------------------------------------------------------------------------
// Wave-3 realtime UX builders (pure; the DOM guard below only renders these).
// Same testable-seam rule as the ETA block: everything the live surfaces say
// or count is decided here, pinned by tests/realtime_ux.test.mjs.
// ---------------------------------------------------------------------------

/**
 * Per-ledger-row five-stage strip (Stamp → Share → Paint → Verify; Sketch is
 * done the moment a row exists — the gesture was sketched locally before it
 * was submitted). States: "done" / "active" / "failed" / "owed".
 *
 * Honesty rule: Verify stays "owed" FOREVER — the client never observes a
 * row-level exact settle (the exact lane reconciles the workspace, not one
 * operation), so the strip must never claim a verify it cannot know. Paint
 * is the last stage a row can honestly reach ("paintedRevision" set by the
 * fast lane's markRowsPainted).
 */
export function ledgerStageDots(row = {}) {
  const state = row.state ?? null;
  const painted = row.paintedRevision !== null && row.paintedRevision !== undefined;
  const stamp =
    state === "refused"
      ? "failed"
      : state === "sending"
        ? "active"
        : state === "applied"
          ? "done"
          : "owed"; // held + unknown: nothing is in flight, nothing is owed-yet-done
  return [
    { stage: "sketch", state: "done" },
    { stage: "stamp", state: stamp },
    { stage: "share", state: state === "applied" ? "done" : "owed" },
    { stage: "paint", state: state === "applied" && painted ? "done" : "owed" },
    { stage: "verify", state: "owed" },
  ];
}

/**
 * The single quotable string a receipt click copies:
 * `{operation_id} · seq {n} · epoch {id?}` — segments drop (never render
 * empty) when the row has not learned them yet; the operation id is the one
 * durable segment (held rows re-POST the same id).
 */
export function receiptQuoteText({ operationId = null, serverSequence = null, epochId = null } = {}) {
  const parts = [];
  if (operationId) parts.push(String(operationId));
  if (serverSequence !== null && serverSequence !== undefined) {
    parts.push(`seq ${serverSequence}`);
  }
  if (epochId) parts.push(`epoch ${epochId}`);
  return parts.join(" · ");
}

/**
 * One-line degradation-ladder key for the result-class badge tooltip, so the
 * symbol vocabulary (■ ● ◐ ◌) is discoverable exactly where it is used
 * rather than only in a distant legend. Order = most trustworthy first.
 */
export function ladderLegendLine() {
  return "Picture quality — ■ Precise result  ● Fast precise  ◐ Close estimate  ◌ Preview";
}

/**
 * Transport pill phase → [dataset state, label] (D3: transport ≠ trust).
 * Plain data so the pure label builder below can share it with the DOM.
 */
const TRANSPORT_PILL_STATES = {
  connecting: ["connecting", "Connecting"],
  live: ["live", "Live"],
  reconnecting: ["reconnecting", "Reconnecting"],
  reconnect_failed: ["failed", "Reconnect failed"],
  buffering: ["catching_up", "Catching up"],
  catching_up: ["catching_up", "Catching up"],
  // §4 catch_up_failed: same surface as reconnecting — one flicker, no
  // separate alarm for a recovery handed to the reconnect cycle.
  catch_up_failed: ["reconnecting", "Reconnecting"],
  closed: ["offline", "Offline"],
  failed: ["failed", "Failed"],
};

/**
 * Transport pill label with reconnect-budget honesty: the realtime client's
 * bounded reconnect cycle carries `{attempt, maxRetries}` on its
 * "reconnecting" status, and the pill must spend them — "Reconnecting"
 * forever is a lie of omission about a bounded budget. Null-safe: counts
 * unknown (or a non-reconnecting phase) render the plain label.
 */
export function transportPillLabel(phase, { attempt = null, maxAttempts = null } = {}) {
  const [, label] = TRANSPORT_PILL_STATES[phase] ?? TRANSPORT_PILL_STATES.closed;
  const n = Number(attempt);
  const m = Number(maxAttempts);
  const counted =
    label === "Reconnecting" &&
    Number.isInteger(n) &&
    n >= 1 &&
    Number.isInteger(m) &&
    m >= 1;
  return counted ? `${label} · attempt ${n} of ${m}` : label;
}

/** The rate-limit cooldown reassurance clause (append-once, see below). */
export const RATE_LIMIT_REASSURANCE = "(capacity refills automatically — nothing was lost)";

/**
 * Idempotent reassurance for rate_limited copy: the cooldown can re-announce
 * (poll retries, held-row countdowns), and the clause must never double —
 * append only when it is not already present.
 */
export function withCapacityReassurance(text) {
  const base = String(text ?? "").trimEnd();
  if (base.includes(RATE_LIMIT_REASSURANCE)) return base;
  return base ? `${base} ${RATE_LIMIT_REASSURANCE}` : RATE_LIMIT_REASSURANCE;
}

/**
 * The legacy recompute anchor gate: a tree the exact funnel can actually
 * address, or null. Realtime-native trees (added/adopted through the
 * operation plane) are UNKNOWN to the legacy tree store — a recompute
 * anchored on one 404s. When EVERY tree is realtime-native there is no
 * addressable anchor and the caller must skip the server call entirely (the
 * local sketch plus the realtime fast/exact lanes own the follow-up) rather
 * than mint a job that can only fail.
 */
export function treeForLegacyRecompute(trees, selectedTreeId) {
  const list = Array.isArray(trees) ? trees : [];
  const selected = list.find((tree) => tree?.id === selectedTreeId) ?? null;
  if (selected && !selected.realtimeNative) return selected;
  return list.find((tree) => !tree.realtimeNative) ?? null;
}

// ---------------------------------------------------------------------------
// Building gesture → wire mapping (pure; the DOM below only wires these).
//
// Buildings ride the realtime operation plane ONLY — there is no legacy
// funnel for them — so the gesture vocabulary (add/delete/move/update, the
// same words commitDesignEdit derives for trees) must map exactly onto the
// plane's frozen verb set. Two rules are load-bearing:
//
//   * the wire has NO "update" verb (types.OperationVerb): a property edit
//     is a "replace" carrying the whole values object, mirroring how
//     realtimeTreeEnvelope maps a local "update" onto "replace";
//   * the reducer's `move` fold keeps position fields only
//     (_POSITION_FIELDS["building_geometry"] == ("footprint_m",)): a move
//     carrying a simultaneous height change would silently DROP the height,
//     so a move+height edit is promoted to a replace.
// ---------------------------------------------------------------------------

/** Ring-bounds center of a footprint ({ Infinity } when the ring is empty). */
export function footprintCenterM(footprintM) {
  let minXM = Infinity;
  let minYM = Infinity;
  let maxXM = -Infinity;
  let maxYM = -Infinity;
  for (const [xM, yM] of footprintM ?? []) {
    minXM = Math.min(minXM, xM);
    maxXM = Math.max(maxXM, xM);
    minYM = Math.min(minYM, yM);
    maxYM = Math.max(maxYM, yM);
  }
  return { xM: (minXM + maxXM) / 2, yM: (minYM + maxYM) / 2 };
}

/**
 * The local operation word for a building edit: add / delete / move /
 * update. A MOVE is a translated ring (center moved); size or height
 * changes with a steady center are updates — the same distinction
 * commitDesignEdit draws for trees via u/v equality.
 */
export function buildingOperationWord(oldBuilding, newBuilding) {
  if (!oldBuilding) return "add";
  if (!newBuilding) return "delete";
  const oldCenter = footprintCenterM(oldBuilding.footprintM);
  const newCenter = footprintCenterM(newBuilding.footprintM);
  const moved =
    Math.abs(oldCenter.xM - newCenter.xM) > 1e-9 ||
    Math.abs(oldCenter.yM - newCenter.yM) > 1e-9;
  return moved ? "move" : "update";
}

/** Local operation word → the plane's frozen verb (see the block comment). */
export function buildingWireVerb(operation, oldBuilding, newBuilding) {
  if (operation === "add") return "add";
  if (operation === "delete") return "delete";
  if (operation === "move") {
    const heightChanged =
      oldBuilding &&
      newBuilding &&
      Number(oldBuilding.heightM) !== Number(newBuilding.heightM);
    return heightChanged ? "replace" : "move";
  }
  return "replace";
}

/**
 * Wire `values` for one building verb: the footprint ring ALWAYS rides
 * whole (it is the authoritative geometry — u/v are not wire values for
 * buildings), height rides whenever the fold stores it. `footprint_m` is
 * in LOCAL scene metres here; the submit path owns the frame conversion
 * (buildingFootprintThroughFrame).
 */
export function buildingWireValues(verb, building) {
  if (!building || verb === "delete") return {};
  if (verb === "move") return { footprint_m: building.footprintM };
  return { footprint_m: building.footprintM, height_m: building.heightM };
}

/**
 * Local scene metres → the server's world CRS: local ring corner → scene
 * UV (model.worldToUv) → wire metres (op_builder.uvToWorld with the live
 * frame). Identity when the frame is the span-local default; a pure offset
 * when the server disclosed a UTM origin.
 */
export function buildingFootprintThroughFrame(footprintM, scene, frame) {
  return (footprintM ?? []).map(([xM, yM]) => {
    const local = sceneWorldToUv(xM, yM, scene);
    const world = uvToWorld(local.u, local.v, frame);
    return [world.x_m, world.y_m];
  });
}

/** Inverse of {@link buildingFootprintThroughFrame} (wire → local metres). */
export function buildingFootprintFromFrame(wireFootprint, scene, frame) {
  return (wireFootprint ?? []).map(([xM, yM]) => {
    const uv = worldToUv(xM, yM, frame);
    const local = buildingToWorld(uv, scene);
    return [local.xM, local.yM];
  });
}

// Frame calibration samples: the wire position of an operation pairs with
// the local position of the SAME object only when the operation did not
// RELOCATE that object. A moved op pairs the pre-edit local corner with the
// post-edit wire corner — offset by the move delta — and one bogus sample
// can lock a permanently wrong local↔wire frame (solveFrameFromSamples
// exact-fits the first non-degenerate pair it finds). The correspondence
// rule below refuses exactly those samples: the wire position, read back
// through the CURRENT frame hypothesis, must already agree with the local
// position before the pair is trusted.

/** Tolerance, in scene metres, for "this operation did not move it". */
export const FRAME_SAMPLE_TOLERANCE_M = 0.5;

/**
 * A calibration sample from a vegetation op, or null when the op moved the
 * tree (wire x_m/y_m no longer describes the local (u, v)).
 */
export function vegetationFrameSample(localTree, values, frame, scene) {
  const xM = Number(values?.x_m);
  const yM = Number(values?.y_m);
  if (!Number.isFinite(xM) || !Number.isFinite(yM)) return null;
  const hypothesis = worldToUv(xM, yM, frame);
  const tolUv =
    FRAME_SAMPLE_TOLERANCE_M / Math.min(scene.widthMeters, scene.heightMeters);
  if (
    Math.abs(hypothesis.u - localTree.u) > tolUv ||
    Math.abs(hypothesis.v - localTree.v) > tolUv
  ) {
    return null;
  }
  return { u: localTree.u, v: localTree.v, x_m: xM, y_m: yM };
}

/**
 * A calibration sample from a building op, or null when the op moved or
 * resized the ring (only an in-place op — e.g. a height-only replace —
 * keeps corner 0 meaning the same physical point on both sides).
 */
export function buildingFrameSample(localBuilding, wireFootprint, frame, scene) {
  const wireCorner = Array.isArray(wireFootprint?.[0]) ? wireFootprint[0] : null;
  if (
    !wireCorner ||
    !Number.isFinite(Number(wireCorner[0])) ||
    !Number.isFinite(Number(wireCorner[1]))
  ) {
    return null;
  }
  const hypothesis = buildingFootprintFromFrame(wireFootprint, scene, frame);
  const localRing = localBuilding.footprintM;
  if (hypothesis.length !== localRing.length) return null;
  const unmoved = hypothesis.every(([xM, yM], index) => {
    const [localXM, localYM] = localRing[index];
    return (
      Math.abs(xM - localXM) <= FRAME_SAMPLE_TOLERANCE_M &&
      Math.abs(yM - localYM) <= FRAME_SAMPLE_TOLERANCE_M
    );
  });
  if (!unmoved) return null;
  const cornerUv = sceneWorldToUv(localRing[0][0], localRing[0][1], scene);
  return {
    u: cornerUv.u,
    v: cornerUv.v,
    x_m: Number(wireCorner[0]),
    y_m: Number(wireCorner[1]),
  };
}

/**
 * Whether the DISPLAYED hour's exact plane is verified for the current
 * workspace revision — the same basis rule renderTimelineFreshness applies
 * to its filled dots (the newest exact patch covered the hour AND the exact
 * lane has caught up). Facts in, boolean out; the slider's skip disclosure
 * must not claim "owed" for a plane that is already verified.
 */
export function exactHourVerified({
  displayedHour = null,
  coveredHours = null,
  exactRevision = 0,
  workspaceRevision = 0,
} = {}) {
  const hour = Number(displayedHour);
  if (!Number.isInteger(hour) || hour < 0) return false;
  const covered =
    coveredHours instanceof Set ? coveredHours : new Set(Array.from(coveredHours ?? []));
  const exact = Number(exactRevision ?? 0);
  const workspace = Number(workspaceRevision ?? 0);
  const exactCurrent = exact > 0 && exact >= workspace;
  return exactCurrent && covered.has(hour);
}

/**
 * The slider skip disclosure (realtime plane owns gestures, no legacy anchor,
 * no time-operation transport): says exactly what happened — the hour view
 * moved, and either this hour is already verified or the exact refresh is
 * owed and verifies asynchronously (the established Verify-owed vocabulary).
 */
export function solarTimeSkipAnnounce(time, hourVerified) {
  return hourVerified
    ? `Solar time changed to ${time} — this hour's precise picture is already verified.`
    : `Solar time changed to ${time} — this hour's precise picture is being prepared.`;
}

// The shell below is DOM-driven. Guarding the boot keeps this module
// importable from node smoke tests (`node --input-type=module -e
// 'import("./app.mjs")'`) without jsdom; everything runs only in a browser.
if (typeof document !== "undefined") {
  const elements = {
  sceneCard: document.querySelector("#sceneCard"),
  glCanvas: document.querySelector("#glCanvas"),
  overlayCanvas: document.querySelector("#overlayCanvas"),
  componentList: document.querySelector("#componentList"),
  buildingList: document.querySelector("#buildingList"),
  connectionPill: document.querySelector("#connectionPill"),
  connectionPillText: document.querySelector("#connectionPillText"),
  workerPill: document.querySelector("#workerPill"),
  workerPillText: document.querySelector("#workerPillText"),
  realtimePill: document.querySelector("#realtimePill"),
  realtimePillText: document.querySelector("#realtimePillText"),
  analysisToast: document.querySelector("#analysisToast"),
  analysisToastTitle: document.querySelector("#analysisToastTitle"),
  analysisToastDetail: document.querySelector("#analysisToastDetail"),
  analysisDuration: document.querySelector("#analysisDuration"),
  liveStatus: document.querySelector("#liveStatus"),
  failureBanner: document.querySelector("#failureBanner"),
  failureMessage: document.querySelector("#failureMessage"),
  retryButton: document.querySelector("#retryButton"),
  compareButton: document.querySelector("#compareButton"),
  resetButton: document.querySelector("#resetButton"),
  exportButton: document.querySelector("#exportButton"),
  timeSlider: document.querySelector("#timeSlider"),
  timeValue: document.querySelector("#timeValue"),
  sceneTimeLabel: document.querySelector("#sceneTimeLabel"),
  heatLayerToggle: document.querySelector("#heatLayerToggle"),
  shadowLayerToggle: document.querySelector("#shadowLayerToggle"),
  roiLayerToggle: document.querySelector("#roiLayerToggle"),
  selectedTreeName: document.querySelector("#selectedTreeName"),
  selectionProperties: document.querySelector("#selectionProperties"),
  rejectedPropertiesNote: document.querySelector("#rejectedPropertiesNote"),
  deleteTreeButton: document.querySelector("#deleteTreeButton"),
  rerunButton: document.querySelector("#rerunButton"),
  meanDeltaMetric: document.querySelector("#meanDeltaMetric"),
  meanDeltaCaption: document.querySelector("#meanDeltaCaption"),
  peakDeltaMetric: document.querySelector("#peakDeltaMetric"),
  areaMetric: document.querySelector("#areaMetric"),
  roiMetric: document.querySelector("#roiMetric"),
  roiCaption: document.querySelector("#roiCaption"),
  sceneVersion: document.querySelector("#sceneVersion"),
  dirtyWindowStep: document.querySelector("#dirtyWindowStep"),
  computeStep: document.querySelector("#computeStep"),
  jobSteps: document.querySelector("#jobSteps"),
  jobStatusDetail: document.querySelector("#jobStatusDetail"),
  modelVersionValue: document.querySelector("#modelVersionValue"),
  siteCacheVersionValue: document.querySelector("#siteCacheVersionValue"),
  modelScopeList: document.querySelector("#modelScopeList"),
  exactBadge: document.querySelector("#exactBadge"),
  // U-D2 capability-driven UI
  designTabButton: document.querySelector("#designTabButton"),
  layersTabButton: document.querySelector("#layersTabButton"),
  designTab: document.querySelector("#designTab"),
  viewsTab: document.querySelector("#viewsTab"),
  capabilityBadge: document.querySelector("#capabilityBadge"),
  editFamiliesRoot: document.querySelector("#editFamiliesRoot"),
  viewLayerSelect: document.querySelector("#viewLayerSelect"),
  viewOperationSelect: document.querySelector("#viewOperationSelect"),
  viewCompareSelect: document.querySelector("#viewCompareSelect"),
  viewCompareField: document.querySelector("#viewCompareField"),
  viewTimeIndex: document.querySelector("#viewTimeIndex"),
  viewRunButton: document.querySelector("#viewRunButton"),
  viewResult: document.querySelector("#viewResult"),
  viewZeroBadge: document.querySelector("#viewZeroBadge"),
  viewBasisCopy: document.querySelector("#viewBasisCopy"),
  impactPanel: document.querySelector("#impactPanel"),
  siteMismatchBanner: document.querySelector("#siteMismatchBanner"),
  siteMismatchMessage: document.querySelector("#siteMismatchMessage"),
  siteMismatchIdentities: document.querySelector("#siteMismatchIdentities"),
  reconnectButton: document.querySelector("#reconnectButton"),
  // Realtime-plane surfaces (ux_redesign/interaction_simulations.md §2/§5).
  // Every one is OPTIONAL: queried defensively — a shell without the id
  // degrades silently (feature off), never throws.
  resultClassBadge: document.querySelector("#resultClassBadge"),
  transportPill: document.querySelector("#transportPill"),
  transportPillText: document.querySelector("#transportPillText"),
  presenceBar: document.querySelector("#presenceBar"),
  metronomeRail: document.querySelector("#metronomeRail"),
  epochLed: document.querySelector("#epochLed"),
  railWorkspace: document.querySelector("#railWorkspace"),
  railFast: document.querySelector("#railFast"),
  railExact: document.querySelector("#railExact"),
  railTriple: document.querySelector("#railTriple"),
  railOwed: document.querySelector("#railOwed"),
  railOfflineNote: document.querySelector("#railOfflineNote"),
  railActors: document.querySelector("#railActors"),
  railMode: document.querySelector("#railMode"),
  heldCount: document.querySelector("#heldCount"),
  activityToggle: document.querySelector("#activityToggle"),
  activityList: document.querySelector("#activityList"),
  activityRows: document.querySelector("#activityRows"),
  activityMeta: document.querySelector("#activityMeta"),
  timelineFreshness: document.querySelector("#timelineFreshness"),
  qualifierPanel: document.querySelector("#qualifierPanel"),
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const urlParams = new URLSearchParams(globalThis.location?.search ?? "");
const apiParam = urlParams.get("api") ?? globalThis.SOLWEIG_API_BASE ?? "";
// "?api=/api" (the serve.py proxy contract) de-doubles to a same-origin base:
// client paths already carry /api/v1, so the raw prefix must not be prepended.
const { baseUrl: apiBaseUrl, connected: connectedMode } = resolveApiBase(apiParam);
// "||" not "??": an EMPTY ?site= falls through to the injected default.
const siteId = urlParams.get("site") || globalThis.SOLWEIG_SITE_ID || "campus-1km-v1";

const fixture = await fetch("./assets/baseline.json").then((response) => {
  if (!response.ok) throw new Error(`Failed to load baseline fixture: ${response.status}`);
  return response.json();
});
const scene = {
  ...fixture.scene,
  gridWidth: fixture.analysis.gridWidth,
  gridHeight: fixture.analysis.gridHeight,
};
const decoded = decodeAnalysisFixture(fixture.analysis);
const baseline = decoded.baseline;
const current = baseline.slice();

const renderer = new SceneRenderer(elements.glCanvas, elements.overlayCanvas, scene);
await renderer.setBaseImage("./assets/site_base.webp");

const state = {
  hour: 12,
  trees: [],
  selectedTreeId: null,
  buildings: [],
  selectedBuildingId: null,
  current,
  dirtyRect: null,
  pendingDirtyRect: null,
  ghostTree: null,
  ghostBuilding: null,
  compareBaseline: false,
  showHeat: true,
  showShadows: true,
  showDirtyRegion: true,
  version: 0,
  workerReady: false,
  latestJobId: 0,
  activeJobId: null,
  dragPreset: null,
  dragBuildingPreset: null,
  treeOrdinal: 0,
  buildingOrdinal: 0,
  pointerDrag: null,
  timeTimer: null,
  // Connected-mode exact layer (server grid, all time steps of one variable).
  exact: null, // {cube, rows, cols, timeSteps, displayedHour}
  previewStale: false,
  serverDirtyRect: null,
  // Measured wall time of completed precise (exact) updates this session —
  // the only duration the workspace-owed badge age line may disclose
  // (owedDurationText); never a client-side guess.
  lastExactDurationMs: null,
  exactDurationCount: 0,
  // U-D2: parsed capability document + generated-editor registries.
  capabilities: null,
  selectionControls: [], // {control, input, output, binding} for the tree editor
  treeAdapterSourceNodes: null,
  buildingControls: null, // {root, controls: [{key, input, output}]} for buildings
};

// ---------------------------------------------------------------------------
// Capability-driven UI (U-D2)
//
// Everything below is GENERATED from the parsed capability document: the
// editor list, every property bound, the class vocabulary, the view-layer
// list, and the dependency-stage classification. The only constants here are
// presentation labels and unit wiring (schema property name → local tree
// attribute), never physics vocabulary.
// ---------------------------------------------------------------------------

/** The adapter whose edits ride the current tree-edit contract (wiring). */
const TREE_EDITOR_TRANSPORT = "tree-edits-v1";

/**
 * Schema property → local tree attribute wiring (unit conversion only; the
 * bounds shown in the UI always come from the document's property_schema).
 */
const TREE_PROPERTY_BINDINGS = {
  height_m: {
    read: (tree) => tree.heightM,
    write: (tree, value) => {
      tree.heightM = Number(value);
    },
  },
  canopy_radius_m: {
    // The schema speaks radius; the tree model keeps diameter.
    read: (tree) => tree.canopyDiameterM / 2,
    write: (tree, value) => {
      tree.canopyDiameterM = Number(value) * 2;
    },
  },
  trunk_ratio: {
    // Every local tree carries trunkRatio (makeTree/apiObjectToTree fill it
    // from the presets / server contract) — no restated schema default here.
    read: (tree) => tree.trunkRatio,
    write: (tree, value) => {
      tree.trunkRatio = Number(value);
    },
  },
};

function treeEditorAdapter() {
  return (
    state.capabilities?.editableAdapters.find(
      (adapter) => EDIT_TRANSPORTS[adapter.id] === TREE_EDITOR_TRANSPORT,
    ) ?? null
  );
}

/** Wire one parsed capability document into every generated panel. */
function setCapabilities(capabilities) {
  state.capabilities = capabilities;
  elements.capabilityBadge.textContent =
    `${capabilities.editableAdapters.length} editors · ` +
    `${capabilities.disclosedAdapters.length} read-only`;
  elements.capabilityBadge.dataset.state = "ok";
  buildSelectionProperties();
  buildEditFamilies();
  buildViewsPanel();
  syncSelectionPanel();
}

/**
 * Fail visibly when no usable document exists. There is deliberately NO
 * hardcoded fallback vocabulary: an empty palette states the truth.
 */
function setCapabilitiesError(error) {
  elements.capabilityBadge.textContent = "document unavailable";
  elements.capabilityBadge.dataset.state = "error";
  elements.editFamiliesRoot.replaceChildren(
    Object.assign(document.createElement("p"), {
      className: "family-error",
      textContent:
        `The capability document could not be loaded (${error.code ?? "error"}: ` +
        `${error.message ?? error}). No editing tools are shown rather than ` +
        `a wrong list. ${
          connectedMode ? "Reconnect once the server serves its editor document." : "Restore assets/capabilities_snapshot.json."
        }`,
    }),
  );
  elements.viewLayerSelect.replaceChildren();
  elements.viewOperationSelect.replaceChildren();
  elements.viewResult.replaceChildren(
    Object.assign(document.createElement("p"), {
      className: "view-empty",
      textContent: "View operations need the capability document.",
    }),
  );
}

function propertyRow({ label, valueText, min, max, step, kind, id }) {
  const row = document.createElement("div");
  row.className = "property-row";
  const head = document.createElement("div");
  const labelElement = document.createElement("label");
  labelElement.textContent = label;
  labelElement.htmlFor = id;
  const output = document.createElement("output");
  output.htmlFor = id;
  output.value = valueText;
  head.append(labelElement, output);
  const input = document.createElement("input");
  input.id = id;
  if (kind === "toggle") {
    input.type = "checkbox";
    input.checked = Boolean(min);
  } else {
    input.type = kind === "range" ? "range" : "number";
    if (min !== null && min !== undefined) input.min = String(min);
    if (max !== null && max !== undefined) input.max = String(max);
    input.step = String(step ?? "any");
    input.value = String(valueText);
  }
  row.append(head, input);
  return { row, input, output };
}

/** Build the selected-object property editors from the tree editor's schema. */
function buildSelectionProperties() {
  const root = elements.selectionProperties;
  root.replaceChildren();
  state.selectionControls = [];
  state.treeAdapterSourceNodes = null;
  elements.rejectedPropertiesNote.hidden = true;

  const adapter = treeEditorAdapter();
  let rejected = [];
  if (!adapter) {
    root.append(
      Object.assign(document.createElement("p"), {
        className: "family-hint",
        textContent:
          "The capability document lists no tree editor for this build, so no " +
          "tree property editors are available (position editing stays on the map).",
      }),
    );
  } else {
    state.treeAdapterSourceNodes = [...adapter.sourceNodes];
    const controlsResult = propertyControls(adapter, state.capabilities);
    rejected = controlsResult.rejected;

    for (const control of controlsResult.controls) {
      const binding = TREE_PROPERTY_BINDINGS[control.name] ?? null;
      const id = `selection-${control.name}`;
      if (!binding) {
        // Positional/geometry properties (x_m/y_m): placement happens by
        // dragging on the map. The document's prose bounds ("site extent,
        // world coordinates") are code-level facts — printing them as panel
        // rows is metadata, not user information.
        continue;
      }
      const valueText =
        control.default !== null && control.default !== undefined
          ? String(control.default)
          : String(control.min ?? 0);
      const { row, input, output } = propertyRow({
        label: control.label,
        valueText,
        min: control.kind === "toggle" ? control.default : control.min,
        max: control.max,
        step: control.step,
        kind: control.kind,
        id,
      });
      input.disabled = true;
      input.title = [control.basis, control.uncertain].filter(Boolean).join(" — ") || control.label;
      input.addEventListener("input", () => {
        const value = control.kind === "toggle" ? input.checked : Number(input.value);
        updateSelectedTreeProperty((tree) => binding.write(tree, value));
      });
      root.append(row);
      state.selectionControls.push({ control, input, output, binding, row });
    }
  }

  buildBuildingPropertyControls(root);

  if (rejected.length > 0) {
    // The schema's rejection reasons cite physics internals (transVeg,
    // utci_process.py) — honest ledger material, panel noise. The panel
    // keeps the verdict; the document keeps the why.
    elements.rejectedPropertiesNote.hidden = false;
    elements.rejectedPropertiesNote.textContent =
      `${rejected.map((entry) => entry.label).join(", ")} — not editable here.`;
  }
}

/**
 * Building property editors: height (2–100 m) and width/depth (2–40 m)
 * sliders that regenerate the axis-aligned footprint ring around its
 * current center. Unlike the tree controls these bounds are studio-side
 * presentation bounds — the wire contract's own clamp (height (0, 1000],
 * ring ≥ 3 distinct vertices) stays the server's business.
 */
function buildBuildingPropertyControls(root) {
  const container = document.createElement("div");
  container.hidden = true;
  const controls = [];
  const definitions = [
    {
      key: "heightM",
      label: "Height",
      min: 2,
      max: 100,
      write: (building, value) => {
        building.heightM = Number(value);
      },
    },
    {
      key: "widthM",
      label: "Width",
      min: 2,
      max: 40,
      write: (building, value) => {
        regenerateBuildingFootprint(building, Number(value), building.depthM);
      },
    },
    {
      key: "depthM",
      label: "Depth",
      min: 2,
      max: 40,
      write: (building, value) => {
        regenerateBuildingFootprint(building, building.widthM, Number(value));
      },
    },
  ];
  for (const definition of definitions) {
    const { row, input, output } = propertyRow({
      label: definition.label,
      valueText: String(definition.min),
      min: definition.min,
      max: definition.max,
      step: 0.5,
      kind: "range",
      id: `building-${definition.key}`,
    });
    input.disabled = true;
    input.title = `${definition.label} in metres — resizing regenerates the footprint around its center`;
    input.addEventListener("input", () => {
      updateSelectedBuildingProperty((building) => definition.write(building, input.value));
    });
    container.append(row);
    controls.push({ key: definition.key, input, output });
  }
  root.append(container);
  state.buildingControls = { root: container, controls };
}

// -- edit family editors ----------------------------------------------------

/**
 * Plain-words rows for the "Universal editors" section. Each names ONE thing
 * a first-year student can picture; the capability document's own facts
 * (status, operations, locality, temporal scope) ride the row's tooltip so
 * the raw disclosure stays one hover away.
 */
const FAMILY_ROWS = Object.freeze([
  { adapterId: "vegetation_geometry", label: "Trees — add, move, resize" },
  { adapterId: "building_geometry", label: "Buildings — add, move, resize" },
  { adapterId: "landcover_surface", label: "Ground surface — paint materials" },
  { adapterId: "meteorological_forcing", label: "Weather — adjust conditions" },
  { adapterId: "selected_date_time", label: "Date & time — choose the hour" },
]);

function buildEditFamilies() {
  const root = elements.editFamiliesRoot;
  root.replaceChildren();
  const adapterById = new Map(
    [
      ...state.capabilities.editableAdapters,
      ...state.capabilities.disclosedAdapters,
    ].map((adapter) => [adapter.id, adapter]),
  );
  for (const row of FAMILY_ROWS) {
    const adapter = adapterById.get(row.adapterId);
    // The buildings row names an INTERACTIVE surface (the library cards +
    // canvas editing); claiming it against a server whose document does not
    // grant the family would be a lie, so the row is gated on the adapter
    // being editable. The other rows describe the studio's own affordances
    // and stay (their truth lives in the document tooltips).
    if (row.adapterId === "building_geometry" && !(adapter && adapter.editable)) continue;
    const item = document.createElement("p");
    item.className = "family-note";
    item.textContent = row.label;
    if (adapter) {
      const presentation = adapterPresentation(adapter);
      item.title = `${adapter.id} · ${presentation.statusLabel} · ${presentation.metaLine}`;
    }
    root.append(item);
  }
  root.append(
    Object.assign(document.createElement("p"), {
      className: "family-hint",
      textContent: "The server confirms which edits are available.",
    }),
  );
  // Working form editors for any OTHER integrated family stay functional
  // below the plain rows (the tree editor is the map itself — row one).
  // Buildings are skipped: their dedicated library cards + selection
  // editors already cover the family, and a second raw JSON form for the
  // same fields is duplicate surface, not extra power.
  for (const adapter of state.capabilities.editableAdapters) {
    if (adapter.id === treeEditorAdapter()?.id) continue;
    if (adapter.id === "building_geometry") continue;
    root.append(buildFamilyCard(adapter));
  }
}

function familySummary(adapter, presentation) {
  // Collapsed card line: a plain title and one honest word ("Editable" /
  // "Read-only"). The capability document's own vocabulary (operations,
  // locality, temporal scope) stays one hover away on the summary tooltip.
  const summary = document.createElement("summary");
  summary.title = `${adapter.id} · ${presentation.statusLabel} · ${presentation.metaLine}`;
  const title = document.createElement("strong");
  title.textContent = presentation.title;
  const pill = document.createElement("span");
  pill.className = "status-pill";
  pill.dataset.tone = presentation.statusTone;
  pill.textContent = presentation.enabled ? "Editable" : "Read-only";
  summary.append(title, pill);
  return summary;
}

function boundsLine(control) {
  const low = control.exclusiveMinimum ? `> ${control.min}` : `≥ ${control.min}`;
  const high =
    control.max === null || control.max === undefined
      ? "no upper bound"
      : control.exclusiveMaximum
        ? `< ${control.max}`
        : `≤ ${control.max}`;
  return `${control.label}: ${low} … ${high}${control.units ? ` ${control.units}` : ""}`;
}

function buildFamilyCard(adapter) {
  const presentation = adapterPresentation(adapter);
  const card = document.createElement("details");
  card.className = "family-card";
  card.append(familySummary(adapter, presentation));
  const body = document.createElement("div");
  body.className = "family-body";
  card.append(body);

  const isTreeEditor = adapter.id === treeEditorAdapter()?.id;
  const { controls, classPicker, rejected, blocked, timeSupport } = propertyControls(
    adapter,
    state.capabilities,
  );

  if (isTreeEditor) {
    body.append(
      Object.assign(document.createElement("p"), {
        className: "family-hint",
        textContent:
          "Trees are placed and edited directly on the map (drag to move, select " +
          "to edit); the bounds below come from the document's property schema.",
      }),
    );
    const list = document.createElement("p");
    list.className = "family-note";
    list.textContent = controls.map((control) => boundsLine(control)).join(" · ");
    body.append(list);
    if (rejected.length > 0) {
      body.append(
        Object.assign(document.createElement("p"), {
          className: "family-note",
          textContent: `Rejected by the schema: ${rejected
            .map((entry) => `${entry.label} (${entry.reason})`)
            .join("; ")}`,
        }),
      );
    }
    card.append(body);
    return card;
  }

  // Form-based family editor: operation + generated property controls. The
  // transport decides where the commit lands (tree vs universal endpoint);
  // `touched` tracks the properties the user actually entered so a universal
  // edit stays partial (untouched properties keep their scene values instead
  // of being reverted to schema defaults).
  const transport = editTransport(adapter.id);
  const isUniversal = transport === "universal-edits-v1";
  const draft = {
    values: {},
    touched: {},
    operation: adapter.operations[0],
    identity: "",
    oldValues: null,
    window: null,
  };
  const errorLine = document.createElement("p");
  errorLine.className = "family-error";
  errorLine.hidden = true;

  if (adapter.operations.length > 0) {
    const operationField = document.createElement("label");
    operationField.className = "view-field";
    operationField.append(Object.assign(document.createElement("span"), { textContent: "Operation" }));
    const select = document.createElement("select");
    select.className = "select-input";
    for (const operation of adapter.operations) {
      const option = document.createElement("option");
      option.value = operation;
      option.textContent = operation;
      select.append(option);
    }
    select.addEventListener("change", () => {
      draft.operation = select.value;
    });
    operationField.append(select);
    body.append(operationField);
  }

  const identityProperty = adapter.propertySchema?.identity_property ?? null;
  if (identityProperty) {
    const field = document.createElement("label");
    field.className = "view-field";
    field.append(Object.assign(document.createElement("span"), { textContent: `${identityProperty} (target)` }));
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = identityProperty;
    input.addEventListener("input", () => {
      draft.identity = input.value;
    });
    field.append(input);
    body.append(field);
  }

  for (const control of controls) {
    if (control.kind === "toggle") {
      const row = document.createElement("div");
      row.className = "checkbox-row";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = `family-${adapter.id}-${control.name}`;
      input.checked = Boolean(control.default);
      draft.values[control.name] = input.checked;
      input.addEventListener("change", () => {
        draft.values[control.name] = input.checked;
        draft.touched[control.name] = true;
      });
      const label = document.createElement("label");
      label.htmlFor = input.id;
      label.textContent = control.label;
      row.append(input, label);
      body.append(row);
      continue;
    }
    if (control.kind === "polygon") {
      const field = document.createElement("label");
      field.className = "view-field";
      field.append(
        Object.assign(document.createElement("span"), {
          textContent: `${control.label} (ring of [x, y] vertices)`,
        }),
      );
      const input = document.createElement("input");
      input.type = "text";
      input.className = "select-input";
      input.placeholder = `[[x1, y1], …] — at least ${control.minItems} vertices`;
      input.addEventListener("input", () => {
        draft.values[control.name] = input.value;
        draft.touched[control.name] = true;
      });
      field.append(input);
      body.append(field);
      continue;
    }
    const initial =
      control.default !== null && control.default !== undefined
        ? control.default
        : (control.min ?? 0);
    const { row, input } = propertyRow({
      label: control.label,
      valueText: String(initial),
      min: control.min,
      max: control.max,
      step: control.step,
      kind: control.kind,
      id: `family-${adapter.id}-${control.name}`,
    });
    draft.values[control.name] = Number(initial);
    input.addEventListener("input", () => {
      draft.values[control.name] = input.value;
      draft.touched[control.name] = true;
    });
    if (control.basis || control.uncertain) {
      input.title = [control.basis, control.uncertain].filter(Boolean).join(" — ");
    }
    body.append(row);
  }

  if (classPicker) {
    const chips = document.createElement("div");
    chips.className = "class-chips";
    chips.setAttribute("role", "group");
    chips.setAttribute("aria-label", "paint class");
    let selectedClass = classPicker.options[0];
    for (const code of classPicker.options) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "class-chip";
      chip.textContent = String(code);
      chip.setAttribute("aria-pressed", String(code === selectedClass));
      chip.addEventListener("click", () => {
        selectedClass = code;
        for (const other of chips.children) {
          other.setAttribute("aria-pressed", String(other === chip));
        }
      });
      chips.append(chip);
    }
    body.append(chips);
    draft.values.class = selectedClass;
    if (classPicker.fenced.length > 0) {
      body.append(
        Object.assign(document.createElement("p"), {
          className: "family-note",
          textContent:
            `Fenced classes ${classPicker.fenced.join(", ")} are excluded by the ` +
            `document (no adapter may claim semantics it cannot deliver).`,
        }),
      );
    }
  }

  if (timeSupport) {
    const field = document.createElement("label");
    field.className = "view-field";
    field.append(Object.assign(document.createElement("span"), { textContent: "Time index" }));
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = "1";
    input.value = "0";
    input.addEventListener("input", () => {
      draft.timeIndex = Number(input.value);
    });
    field.append(input);
    body.append(field);
    body.append(
      Object.assign(document.createElement("p"), {
        className: "family-note",
        textContent: timeSupport,
      }),
    );
  }

  if (isUniversal && adapter.id === "landcover_surface") {
    // Paint addressing: the capability document carries the class vocabulary
    // but no window grammar (the raster window is transport addressing, not
    // physics vocabulary), so the UI supplies its own row/col window inputs.
    const windowField = document.createElement("div");
    windowField.className = "view-field";
    windowField.append(
      Object.assign(document.createElement("span"), {
        textContent: "Paint window (row start/stop · col start/stop)",
      }),
    );
    const windowInputs = {};
    for (const key of ["row_start", "row_stop", "col_start", "col_stop"]) {
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.step = "1";
      input.placeholder = key;
      input.className = "select-input";
      windowInputs[key] = input;
      input.addEventListener("input", () => {
        const window = {};
        for (const [name, field] of Object.entries(windowInputs)) {
          window[name] = field.value === "" ? null : Number(field.value);
        }
        draft.window = window;
      });
      windowField.append(input);
    }
    body.append(windowField);
    body.append(
      Object.assign(document.createElement("p"), {
        className: "family-note",
        textContent:
          "The window addresses the cells a paint re-labels (stop bounds exclusive, " +
          "raster row/col). Operations that need no window ignore it.",
      }),
    );
  }

  if (isUniversal && identityProperty) {
    // Building-style families address an entity by id; move/update/delete
    // also carry its CURRENT values so the adapter can validate the
    // transition (old_values is user-declared state, validated server-side).
    const field = document.createElement("label");
    field.className = "view-field";
    field.append(
      Object.assign(document.createElement("span"), {
        textContent: `Current values (JSON — for move/update/delete)`,
      }),
    );
    const input = document.createElement("textarea");
    input.rows = 2;
    input.className = "select-input";
    input.placeholder = `{"height_m": 12, "footprint_m": [[x, y], …]} — the entity's current values`;
    input.addEventListener("input", () => {
      draft.oldValuesText = input.value;
    });
    field.append(input);
    body.append(field);
  }

  if (blocked.length > 0) {
    body.append(
      Object.assign(document.createElement("p"), {
        className: "family-note",
        textContent: `Blocked by the document: ${blocked.join(", ")}.`,
      }),
    );
  }

  const commitButton = document.createElement("button");
  commitButton.type = "button";
  commitButton.className = "primary-button view-run";
  commitButton.textContent = "Validate & commit";
  commitButton.addEventListener("click", () => {
    errorLine.hidden = true;
    // Polygon controls arrive as text; parse to the ring array the contract
    // will expect (bounds/min_items stay the schema's business).
    const properties = { ...draft.values };
    for (const control of controls) {
      if (control.kind === "polygon" && typeof properties[control.name] === "string") {
        try {
          properties[control.name] = JSON.parse(properties[control.name]);
        } catch {
          errorLine.hidden = false;
          errorLine.textContent = `${control.label}: paste a JSON array like [[510, 320], [520, 325], …]`;
          return;
        }
      }
    }
    // old_values is declared current state (user-pasted), validated by the
    // server's adapter — the UI only refuses text it cannot parse.
    let oldValues = null;
    if (isUniversal && typeof draft.oldValuesText === "string" && draft.oldValuesText.trim()) {
      try {
        oldValues = JSON.parse(draft.oldValuesText);
      } catch {
        errorLine.hidden = false;
        errorLine.textContent = "old_values: paste a JSON object (or leave it empty)";
        return;
      }
      if (!oldValues || typeof oldValues !== "object" || Array.isArray(oldValues)) {
        errorLine.hidden = false;
        errorLine.textContent = "old_values: paste a JSON object (or leave it empty)";
        return;
      }
    }
    // Paint addressing: a complete window becomes the item target; anything
    // else falls back to the identity field (building-style ids).
    let target = draft.identity || null;
    if (
      isUniversal &&
      adapter.id === "landcover_surface" &&
      draft.operation === "paint" &&
      draft.window
    ) {
      const window = draft.window;
      const keys = ["row_start", "row_stop", "col_start", "col_stop"];
      if (keys.some((key) => window[key] === null || !Number.isFinite(Number(window[key])))) {
        errorLine.hidden = false;
        errorLine.textContent = "paint window: fill all four row/col bounds";
        return;
      }
      target = Object.fromEntries(keys.map((key) => [key, Number(window[key])]));
    }
    let items;
    try {
      items = composeFamilyEdit(state.capabilities, adapter.id, {
        operation: draft.operation,
        properties,
        target,
        timeIndex: draft.timeIndex ?? null,
        oldValues,
        touchedProperties: Object.keys(draft.touched),
      });
    } catch (error) {
      if (error instanceof DraftValidationError) {
        errorLine.hidden = false;
        errorLine.textContent = error.message;
        return;
      }
      if (error instanceof FamilyTransportError) {
        showTransportPending(body, error, presentation.title);
        announce(
          `${presentation.title} edits are validated but this server build has no ` +
            `transport for them yet.`,
        );
        return;
      }
      throw error;
    }
    if (!connectedMode) {
      errorLine.hidden = false;
      errorLine.textContent =
        "Connect an exact server (?api=<base-url>) to commit this family.";
      return;
    }
    session?.commitEdits({
      edits: items,
      transport: transport ?? "tree-edits-v1",
      sourceNodes: [...adapter.sourceNodes],
      requested: requestedResultForCurrentHour(),
      label: `${presentation.title} ${draft.operation}`,
    });
  });
  body.append(errorLine, commitButton);
  return card;
}

function showTransportPending(body, error, title) {
  body.querySelector(".family-payload")?.remove();
  body.querySelector(".transport-note")?.remove();
  const note = document.createElement("p");
  note.className = "family-note transport-note";
  note.textContent =
    `${title}: this server build has no transport wired for ` +
    `${error.adapterId} edits yet. The validated payload below is exactly ` +
    `the universal item this frontend will send once a transport lands; ` +
    `nothing was submitted.`;
  const payload = document.createElement("pre");
  payload.className = "family-payload";
  payload.textContent = JSON.stringify(error.payload, null, 1);
  body.append(note, payload);
}

// -- view-only panel (UEDIT-007) ---------------------------------------------

const VIEW_OPERATION_LABELS = {
  select_layer: "Select layer",
  compare: "Compare two layers",
  legend: "Legend statistics",
  cached_time: "Cached time steps",
};

function buildViewsPanel() {
  const model = viewPanelModel(state.capabilities);
  elements.viewBasisCopy.textContent =
    model.basis ||
    "View operations select among already-published layers; they never enqueue a solver job.";
  elements.viewOperationSelect.replaceChildren(
    ...model.operations.map((operation) => {
      const option = document.createElement("option");
      option.value = operation;
      option.textContent = VIEW_OPERATION_LABELS[operation] ?? operation;
      return option;
    }),
  );
  const layerOptions = model.layers.map((layer) => {
    const option = document.createElement("option");
    option.value = layer;
    option.textContent = layer;
    return option;
  });
  elements.viewLayerSelect.replaceChildren(...layerOptions);
  elements.viewCompareSelect.replaceChildren(
    ...model.layers.map((layer) => {
      const option = document.createElement("option");
      option.value = layer;
      option.textContent = layer;
      return option;
    }),
  );
  elements.viewCompareField.hidden =
    elements.viewOperationSelect.value !== "compare";
  elements.viewZeroBadge.dataset.state = "idle";
  elements.viewZeroBadge.textContent = "0 jobs";
  elements.viewResult.replaceChildren(
    Object.assign(document.createElement("p"), {
      className: "view-empty",
      textContent: "Run a view to inspect a published layer.",
    }),
  );
}

async function runView() {
  const operation = elements.viewOperationSelect.value;
  const layer = elements.viewLayerSelect.value;
  if (!operation || !layer) {
    renderViewError({ code: "not_ready", message: "the layers list is still loading" });
    return;
  }
  if (!connectedMode || !session?.connected) {
    renderViewError({
      code: "not_connected",
      message: "view operations need a connected exact session (?api=<base-url>)",
    });
    return;
  }
  const timeIndex = Number(elements.viewTimeIndex.value) || 0;
  elements.viewZeroBadge.dataset.state = "idle";
  try {
    const response = await session.createView({
      operation,
      layer,
      timeIndex,
      compareLayer: operation === "compare" ? elements.viewCompareSelect.value : null,
    });
    renderViewResult(viewResultModel(response));
    announce(`View ${operation} on ${layer} answered without a solver job.`);
  } catch (error) {
    renderViewError(viewErrorModel(error));
  }
}

function viewRow(labelText, value) {
  const row = document.createElement("div");
  row.className = "view-row";
  row.append(
    Object.assign(document.createElement("span"), { textContent: labelText }),
    Object.assign(document.createElement("code"), { textContent: String(value) }),
  );
  return row;
}

function layerCard(layerMeta, heading) {
  const card = document.createElement("div");
  card.className = "view-card";
  card.append(Object.assign(document.createElement("strong"), { textContent: heading }));
  if (layerMeta?.name) card.append(viewRow("layer", layerMeta.name));
  if (layerMeta?.dtype) card.append(viewRow("dtype", layerMeta.dtype));
  if (layerMeta?.shape) card.append(viewRow("shape", layerMeta.shape.join(" × ")));
  if (layerMeta?.nodata !== undefined && layerMeta?.nodata !== null) {
    card.append(viewRow("nodata", layerMeta.nodata));
  }
  return card;
}

function renderViewResult(model) {
  if (model.zeroJob) {
    elements.viewZeroBadge.dataset.state = "free";
    elements.viewZeroBadge.textContent = model.zeroJob;
  } else {
    elements.viewZeroBadge.dataset.state = "idle";
    elements.viewZeroBadge.textContent = "—";
  }
  const result = document.createElement("div");
  result.className = "view-result-grid";
  const head = document.createElement("div");
  head.className = "view-card zero";
  head.append(
    Object.assign(document.createElement("strong"), {
      textContent: `${model.operation} · scene v${model.sceneVersion ?? "?"}`,
    }),
  );
  head.append(viewRow("time index", model.timeIndex ?? 0));
  result.append(head);

  if (model.operation === "compare") {
    result.append(layerCard(model.layer, "layer"), layerCard(model.compareLayer, "compare with"));
  } else if (model.operation === "legend" && model.legend) {
    const legendCard = document.createElement("div");
    legendCard.className = "view-card";
    legendCard.append(Object.assign(document.createElement("strong"), { textContent: "legend" }));
    for (const [key, value] of Object.entries(model.legend)) {
      legendCard.append(viewRow(key, value));
    }
    result.append(legendCard);
  } else if (model.operation === "cached_time") {
    const chipCard = document.createElement("div");
    chipCard.className = "view-card";
    chipCard.append(Object.assign(document.createElement("strong"), { textContent: "cached time steps" }));
    const chips = document.createElement("div");
    chips.className = "time-chips";
    for (const index of model.cachedTimeIndices ?? []) {
      const chip = document.createElement("span");
      chip.className = "time-chip";
      if (index === model.timeIndex) chip.style.borderColor = "rgba(114,224,178,.6)";
      chip.textContent = String(index);
      chips.append(chip);
    }
    chipCard.append(chips);
    result.append(chipCard);
  } else {
    result.append(layerCard(model.layer, "layer"));
  }
  if (model.resultManifestUrl) result.append(viewRow("manifest", model.resultManifestUrl));
  elements.viewResult.replaceChildren(...result.children);
}

function renderViewError(model) {
  const card = document.createElement("div");
  card.className = "view-card view-error";
  card.append(
    Object.assign(document.createElement("strong"), { textContent: model.code }),
    Object.assign(document.createElement("p"), {
      className: "family-hint",
      textContent: model.message,
    }),
  );
  if (model.publishedLayers) {
    card.append(
      Object.assign(document.createElement("p"), {
        className: "family-note",
        textContent: `Producible by the engine but not published at this scene version. Published layers: ${model.publishedLayers.join(", ")}.`,
      }),
    );
  }
  if (model.cachedTimeIndices) {
    card.append(
      Object.assign(document.createElement("p"), {
        className: "family-note",
        textContent: `Cached time steps: ${model.cachedTimeIndices.join(", ")}.`,
      }),
    );
  }
  elements.viewResult.replaceChildren(card);
}

// -- dependency impact panel (UEDIT-009) --------------------------------------

function stageChipRow(label, nodes, stage) {
  const row = document.createElement("div");
  row.className = "impact-row";
  row.append(Object.assign(document.createElement("span"), { textContent: label }));
  const chips = document.createElement("div");
  chips.className = "impact-nodes";
  for (const node of nodes) {
    const chip = document.createElement("span");
    chip.className = "stage-chip";
    chip.dataset.stage = stage;
    chip.textContent = node;
    chips.append(chip);
  }
  row.append(chips);
  return row;
}

function renderImpact(sourceNodes, metrics = null, impactPlan = null) {
  if (!state.capabilities) return;
  let model;
  try {
    model = impactPanelModel(state.capabilities, sourceNodes, metrics, impactPlan);
  } catch (error) {
    elements.impactPanel.replaceChildren(
      Object.assign(document.createElement("p"), {
        className: "family-error",
        textContent: `Impact unavailable: ${error.message}`,
      }),
    );
    return;
  }
  if (model.recomputeOnly) {
    // A job that writes no source node (time-only recompute, manual rerun,
    // retry) must not leave the PREVIOUS edit's stage classification on
    // screen through this newer job: report the recompute itself.
    const note = document.createElement("p");
    note.className = "impact-scope";
    const emphasis = document.createElement("em");
    emphasis.textContent = "Recompute only — this request changes no model input.";
    note.append(emphasis);
    if (model.scope?.scopeText) {
      note.append(
        Object.assign(document.createElement("small"), {
          textContent: `Scope: ${model.scope.scopeText}`,
        }),
      );
    }
    elements.impactPanel.replaceChildren(note);
    return;
  }
  const panel = document.createElement("div");
  panel.className = "impact-panel";
  panel.append(
    stageChipRow("Changed", model.changed, "changed"),
    stageChipRow("Recomputed", model.recomputed, "recomputed"),
    stageChipRow("Reused", model.reused, "reused"),
  );
  if (model.serverPlan && Array.isArray(model.perNode)) {
    // Server-served executed plan (u-d4): the per-node `why` lines are the
    // planner's own routing reasons, not a derived closure.
    const reasons = model.perNode.filter((entry) => entry.why);
    if (reasons.length > 0) {
      const list = document.createElement("ul");
      list.className = "impact-reasons";
      for (const entry of reasons.slice(0, 6)) {
        const item = document.createElement("li");
        item.textContent = `${entry.node}: ${entry.why}`;
        list.append(item);
      }
      panel.append(list);
    }
  }
  const scope = document.createElement("p");
  scope.className = "impact-scope";
  if (model.scope?.scopeText) {
    scope.innerHTML = "";
    const emphasis = document.createElement("em");
    emphasis.textContent = `Scope: ${model.scope.scopeText}`;
    scope.append(emphasis);
    if (model.serverPlan && model.routing?.mode) {
      scope.append(
        Object.assign(document.createElement("small"), {
          textContent: ` · plan routing: ${model.routing.mode} (server-executed)`,
        }),
      );
    }
    if (model.scope.fallbackReason) {
      scope.append(
        Object.assign(document.createElement("small"), {
          textContent: `fallback: ${model.scope.fallbackReason}`,
        }),
      );
    }
  } else {
    scope.textContent = "Scope: precise figures pending…";
  }
  panel.append(scope);
  elements.impactPanel.replaceChildren(...panel.children);
}

// -- site-identity mismatch (fail visibly, never a zombie state) --------------

function handleSiteMismatch({ message, pinnedIdentity, currentIdentity }) {
  elements.siteMismatchBanner.dataset.state = "visible";
  elements.siteMismatchMessage.textContent =
    `${message} The scenario is pinned to a site identity this deployment no ` +
    `longer serves, so edits and resets are refused until you start a ` +
    `fresh scenario. Your design stays visible below.`;
  renderIdentityDiff(pinnedIdentity, currentIdentity);
  setConnectionState("failed", "Site identity mismatch");
  setWorkerState("ready", "Session blocked", "site identity mismatch — reconnect to continue");
  announce("Site identity mismatch. Start a fresh scenario to continue.");
}

function renderIdentityDiff(pinnedIdentity, currentIdentity) {
  const rows = document.createDocumentFragment();
  const keys = [
    ...new Set([
      ...Object.keys(pinnedIdentity ?? {}),
      ...Object.keys(currentIdentity ?? {}),
    ]),
  ].sort();
  if (keys.length === 0) {
    elements.siteMismatchIdentities.replaceChildren(
      Object.assign(document.createElement("p"), {
        className: "family-hint",
        textContent: "The server did not include the pinned/current identities.",
      }),
    );
    return;
  }
  for (const key of keys) {
    const pinned = pinnedIdentity?.[key];
    const current = currentIdentity?.[key];
    const row = document.createElement("div");
    row.dataset.diff = String(JSON.stringify(pinned) !== JSON.stringify(current));
    const keyElement = document.createElement("span");
    keyElement.className = "id-key";
    keyElement.textContent = key;
    const pinnedElement = document.createElement("span");
    pinnedElement.className = "id-value is-pinned";
    pinnedElement.textContent = `pinned: ${JSON.stringify(pinned) ?? "—"}`;
    const currentElement = document.createElement("span");
    currentElement.className = "id-value is-current";
    currentElement.textContent = `current: ${JSON.stringify(current) ?? "—"}`;
    row.append(keyElement, pinnedElement, currentElement);
    rows.append(row);
  }
  elements.siteMismatchIdentities.replaceChildren(rows);
}

elements.reconnectButton.addEventListener("click", () => {
  elements.siteMismatchBanner.dataset.state = "hidden";
  clearFailure();
  // A fresh scenario carries a fresh site pin; the baseline lands through the
  // normal connect path and onAuthoritative adopts the server's empty scene.
  connectToServer();
});

// -- sidebar tabs --------------------------------------------------------------

function switchTab(tab) {
  const design = tab === "design";
  elements.designTabButton.classList.toggle("is-active", design);
  elements.layersTabButton.classList.toggle("is-active", !design);
  elements.designTabButton.setAttribute("aria-selected", String(design));
  elements.layersTabButton.setAttribute("aria-selected", String(!design));
  elements.designTab.hidden = !design;
  elements.viewsTab.hidden = design;
}
elements.designTabButton.addEventListener("click", () => switchTab("design"));
elements.layersTabButton.addEventListener("click", () => switchTab("layers"));
elements.viewRunButton.addEventListener("click", () => {
  runView();
});

// ---------------------------------------------------------------------------
// Local preview worker (non-scientific; always client-side)
// ---------------------------------------------------------------------------

const worker = new Worker(new URL("./solver_worker.mjs", import.meta.url), { type: "module" });
worker.postMessage(
  {
    type: "init",
    baseline: baseline.buffer.slice(0),
    buildingMask: decoded.buildingMask.buffer.slice(0),
    scene,
  },
);

function canvasCoordinates(event) {
  const rect = elements.overlayCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (elements.overlayCanvas.width / rect.width),
    y: (event.clientY - rect.top) * (elements.overlayCanvas.height / rect.height),
  }
}

function getSelectedTree() {
  return state.trees.find((tree) => tree.id === state.selectedTreeId) ?? null;
}

function getSelectedBuilding() {
  return state.buildings.find((building) => building.id === state.selectedBuildingId) ?? null;
}

function announce(message) {
  if (!elements.liveStatus) return;
  // Compare-guard (same pattern as setJobStatusDetail): the waking/retry
  // cycles re-announce identical copy every attempt, and a repeated DOM
  // write to the aria-live region is BOTH flicker and screen-reader noise
  // — the line only changes (and only re-speaks) when the sentence does.
  if (elements.liveStatus.textContent === message) return;
  elements.liveStatus.textContent = message;
}

function setConnectionState(kind, text) {
  if (!elements.connectionPill) return;
  // Anti-flicker (same compare-guard pattern as setJobStatusDetail): the
  // waking-cycle and poll paths re-render the same state constantly; only a
  // CHANGED class or text touches the DOM.
  if (elements.connectionPill.dataset.state !== kind) elements.connectionPill.dataset.state = kind;
  if (elements.connectionPillText.textContent !== text) {
    elements.connectionPillText.textContent = text;
  }
  // The rail's offline note is the OFFLINE-mode class lamp: a session that is
  // connecting or connected must never render the OFFLINE word (catch-up
  // honesty — the join window IS collaborative, not "no collaboration").
  // Only a failed connect (the app falls back to the local preview) and the
  // offline mode itself own the note.
  if (elements.railOfflineNote) {
    const hidden = offlineRailNoteHidden(kind);
    if (elements.railOfflineNote.hidden !== hidden) elements.railOfflineNote.hidden = hidden;
  }
  // The badge must follow the mode (L-6): offline preview must never claim
  // "Exact", and a connecting session has no verified result yet either, so
  // it shows a neutral label until the first verified baseline lands (R-4).
  // With a live collaborative workspace the badge stops being a CONNECTION
  // echo entirely (I-29/L-6: it must never claim a class it cannot support):
  // renderClassBadge renders it from the same classFor() view, so a
  // "connected" pill can no longer pin it to "Exact" forever.
  if (collab.workspaceId) return;
  const badge = exactBadgeForConnection(kind);
  if (elements.exactBadge.textContent !== badge.text) elements.exactBadge.textContent = badge.text;
  if (elements.exactBadge.dataset.state !== badge.state) {
    elements.exactBadge.dataset.state = badge.state;
  }
}

function setWorkerState(kind, title, detail, durationSeconds = null) {
  const running = kind === "running";
  elements.workerPill.dataset.state = running ? "running" : "ready";
  elements.workerPillText.textContent = running ? "Updating" : "Ready";
  elements.analysisToast.dataset.state = running ? "running" : "ready";
  elements.analysisToastTitle.textContent = title;
  elements.analysisToastDetail.textContent = detail;
  if (durationSeconds !== null) {
    elements.analysisDuration.textContent = `${durationSeconds.toFixed(2)} s`;
  }
}

/** "14:05" — the HH:MM stamp vocabulary for the last-applied-exact mark. */
function formatClockStamp(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * The "last map update HH:MM" marker (goal ①): rides the worker pill and the
 * analysis toast as a title, so it answers "when did the pixels last come
 * from the server?" without claiming anything about a run in flight.
 */
function renderLastExactStamp() {
  if (!rt.lastExactAtMs) return;
  const stamp = `Last map update ${formatClockStamp(rt.lastExactAtMs)}`;
  if (elements.workerPill) elements.workerPill.title = stamp;
  if (elements.analysisToast) elements.analysisToast.title = stamp;
}

/**
 * 1 s re-render of the countdown-bearing surfaces while an eta deadline is
 * live (goal ①: the remaining time visibly DECREASES). Timer lives only
 * while there is something to count; every other surface it touches is
 * compare-guarded, so idle ticks are cheap no-op DOM writes at worst.
 */
function syncExactProgressTicker() {
  const wantTimer = Boolean(rt.exactProgress?.deadlineMs);
  if (wantTimer && !rt.exactProgressTimer) {
    rt.exactProgressTimer = setInterval(() => {
      renderClassBadge();
      if (!rt.exactProgress?.deadlineMs) {
        clearInterval(rt.exactProgressTimer);
        rt.exactProgressTimer = 0;
      }
    }, 1000);
  } else if (!wantTimer && rt.exactProgressTimer) {
    clearInterval(rt.exactProgressTimer);
    rt.exactProgressTimer = 0;
    renderClassBadge();
  }
}

/**
 * #jobStatusDetail write with an anti-flicker guard: the ~1 s poll re-renders
 * the same line every tick, so the DOM is only touched when the rendered
 * text actually changed (and a shell without the node degrades silently,
 * same defensive-query pattern as the other detail writes).
 */
function setJobStatusDetail(text) {
  if (!elements.jobStatusDetail) return;
  if (elements.jobStatusDetail.textContent === text) return;
  elements.jobStatusDetail.textContent = text;
}

function showFailure(message) {
  elements.failureBanner.dataset.state = "visible";
  elements.failureMessage.textContent = message;
}

function clearFailure() {
  elements.failureBanner.dataset.state = "hidden";
}

// ---------------------------------------------------------------------------
// Five-stage pipeline (Sketch → Stamp → Share → Paint → Verify)
//
// #jobSteps is relabeled IN PLACE (index.html is owned elsewhere): the four
// legacy <li>s are rebuilt into the five realtime stages, and the two live
// detail nodes (#dirtyWindowStep, #computeStep) are RE-HOMED, not replaced,
// so their element references stay valid. Stages transition on the realtime
// events (submit → stamp, canonical ack → share, fast → paint, exact →
// verify); the legacy exact-session path maps onto the same list
// (queued → share, running → verify, applied → verify done).
// ---------------------------------------------------------------------------

const PIPELINE_STAGES = Object.freeze([
  { key: "sketch", label: "Sketch", hint: "local preview kernel" },
  { key: "stamp", label: "Stamp", hint: "sending · receipt pending" },
  { key: "share", label: "Share", hint: "canonical order pending" },
  { key: "paint", label: "Paint", hint: "fast result pending" },
  { key: "verify", label: "Verify", hint: "exact result pending" },
]);

const pipelineSmalls = new Map();

const VERIFY_STAGE_INDEX = PIPELINE_STAGES.findIndex((stage) => stage.key === "verify");
const STAMP_STAGE_INDEX = PIPELINE_STAGES.findIndex((stage) => stage.key === "stamp");

function buildPipelineStages() {
  if (!elements.jobSteps) return;
  const items = [];
  for (const stage of PIPELINE_STAGES) {
    const item = document.createElement("li");
    const badge = document.createElement("span");
    badge.textContent = String(PIPELINE_STAGES.indexOf(stage) + 1);
    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = stage.label;
    const small = document.createElement("small");
    small.textContent = stage.hint;
    body.append(title, small);
    item.append(badge, body);
    // Re-home the live detail nodes the rest of the shell still writes to.
    if (stage.key === "paint" && elements.dirtyWindowStep) {
      body.append(elements.dirtyWindowStep);
    }
    if (stage.key === "verify" && elements.computeStep) {
      body.append(elements.computeStep);
    }
    pipelineSmalls.set(stage.key, small);
    items.push(item);
  }
  elements.jobSteps.replaceChildren(...items);
}

/**
 * Mark the pipeline at one stage: earlier stages complete, later stages go
 * idle. A new gesture restarts the bar at its own first stage — the bar
 * describes the CURRENT gesture's five steps, not a global high-water mark.
 */
function setPipelineStage(key, mode = "running") {
  if (!elements.jobSteps) return;
  const index = PIPELINE_STAGES.findIndex((stage) => stage.key === key);
  if (index < 0) return;
  [...elements.jobSteps.children].forEach((item, position) => {
    item.classList.toggle("is-running", mode === "running" && position === index);
    item.classList.toggle(
      "is-complete",
      position < index || (mode === "done" && position === index),
    );
    // A stage transition re-describes the bar: owed/failed lamps from the
    // previous gesture must not survive into the new one.
    item.classList.remove("is-owed", "is-failed");
  });
  elements.jobSteps.dataset.stage = key;
  elements.jobSteps.dataset.mode = mode;
}

function updateStageDetails({ window: detailWindow, durationMs } = {}) {
  if (detailWindow && elements.dirtyWindowStep) {
    const width = detailWindow.colStop - detailWindow.colStart;
    const height = detailWindow.rowStop - detailWindow.rowStart;
    elements.dirtyWindowStep.textContent = `${width} × ${height} cells + halo`;
  }
  if (durationMs !== undefined && elements.computeStep) {
    elements.computeStep.textContent = `sketch · ${(durationMs / 1000).toFixed(2)} s`;
  }
}

function pipelineSmall(key) {
  return pipelineSmalls.get(key) ?? null;
}

/**
 * Legacy exact-session status → pipeline stages (compatibility for the
 * non-realtime edit path): queued → Share running, running → Verify
 * running, applied → Verify done. Offline preview completes Sketch only
 * (a browser sketch is never a verified result — it does not claim to be).
 */
function updateJobSteps(activeIndex = null, details = {}) {
  updateStageDetails(details);
  if (activeIndex === null) {
    setPipelineStage(connectedMode ? "verify" : "sketch", "done");
    return;
  }
  if (activeIndex <= 1) setPipelineStage("share", "running");
  else setPipelineStage("verify", "running");
}

function render() {
  const comparing = state.compareBaseline;
  if (!connectedMode) {
    const values = comparing ? baseline : state.current;
    renderer.updateHeatmap(values, scene.gridWidth, scene.gridHeight);
  }
  renderer.renderBase({
    showHeat: state.showHeat,
    heatOpacity: comparing ? 0.48 : 0.64,
    // Preview delta texture only while the exact result is stale.
    showPreview: connectedMode && state.previewStale && !comparing,
    previewOpacity: 0.42,
  });
  renderer.renderOverlay({
    trees: comparing ? [] : state.trees,
    selectedTreeId: comparing ? null : state.selectedTreeId,
    dirtyRect: comparing ? null : (state.serverDirtyRect ?? state.dirtyRect),
    hour: state.hour,
    showShadows: state.showShadows && !comparing,
    showDirtyRegion: state.showDirtyRegion && !comparing,
    ghostTree: comparing ? null : state.ghostTree,
    showPreviewMarker: connectedMode && state.previewStale && !comparing,
    // The renderer takes buildings as a plain list (no dedicated ghost
    // parameter): the drag preview rides the same list as a full-opacity
    // member with a non-selectable "ghost" id.
    buildings: comparing
      ? []
      : state.ghostBuilding
        ? [...state.buildings, state.ghostBuilding]
        : state.buildings,
    selectedBuildingId: comparing ? null : state.selectedBuildingId,
  });
}

/** Full re-upload of the displayed exact plane (baseline load / hour switch). */
function refreshExactTexture() {
  if (!state.exact) return;
  const plane = cubeTimePlane(
    state.exact.cube,
    state.exact.displayedHour,
    state.exact.cols,
    state.exact.rows,
  );
  renderer.updateHeatmap(plane, state.exact.cols, state.exact.rows);
  renderer.heatValues = plane; // mirror for region updates
}

function syncSelectionPanel() {
  const selected = getSelectedTree();
  const selectedBuilding = getSelectedBuilding();
  // Selection is exclusive across families: tree controls stay enabled only
  // for a tree selection, building controls only for a building selection,
  // and the shared delete button follows whichever object is selected. The
  // tree rows also HIDE on a building selection — a column of dead sliders
  // for the other family is noise, not information.
  for (const { input, row } of state.selectionControls) {
    input.disabled = !selected;
    if (row) row.hidden = Boolean(selectedBuilding);
  }
  if (state.buildingControls) {
    state.buildingControls.root.hidden = !selectedBuilding;
    for (const { input } of state.buildingControls.controls) {
      input.disabled = !selectedBuilding;
    }
  }
  elements.deleteTreeButton.disabled = !selected && !selectedBuilding;
  if (!selected && selectedBuilding) {
    elements.selectedTreeName.textContent = selectedBuilding.label;
    elements.deleteTreeButton.setAttribute("aria-label", "Delete selected building");
    for (const { key, input, output } of state.buildingControls?.controls ?? []) {
      input.value = String(selectedBuilding[key]);
      if (output) output.value = `${Number(selectedBuilding[key]).toFixed(1)} m`;
    }
    return;
  }
  elements.deleteTreeButton.setAttribute("aria-label", "Delete selected tree");
  if (!selected) {
    elements.selectedTreeName.textContent = "No tree selected";
    return;
  }
  elements.selectedTreeName.textContent = selected.label;
  for (const { control, input, output, binding } of state.selectionControls) {
    if (!binding) continue;
    const value = binding.read(selected);
    if (control.kind === "toggle") {
      input.checked = Boolean(value);
      if (output) output.value = value ? "on" : "off";
    } else {
      input.value = String(value);
      if (output) {
        output.value =
          control.units && ["m", "ratio"].includes(control.units)
            ? `${Number(value).toFixed(control.units === "m" ? 1 : 2)} ${control.units}`
            : String(Math.round(Number(value) * 100) / 100);
      }
    }
  }
}

function updateMetrics(metrics, window) {
  const fraction =
    ((window.colStop - window.colStart) * (window.rowStop - window.rowStart)) /
    (scene.gridWidth * scene.gridHeight);
  const originalCells = Math.round(fraction * 500 * 500);
  elements.meanDeltaMetric.textContent = formatSignedTemperature(metrics.meanDelta);
  elements.peakDeltaMetric.textContent = formatSignedTemperature(metrics.peakDelta);
  elements.areaMetric.textContent = `${formatInteger(metrics.improvedAreaM2)} m²`;
  elements.roiMetric.textContent = `${(fraction * 100).toFixed(1)}%`;
  elements.roiCaption.textContent = `${formatInteger(originalCells)} / 250,000 cells`;
  elements.sceneVersion.textContent = `v${state.version}`;
}

/**
 * Metrics from a server result manifest (region + time stated, UI-005).
 *
 * `decodedWindow` is the DECODED camelCase window — the raw manifest's
 * `window` is snake_case and must never be read here directly (M-2).
 *
 * `trackDuration` (default true) records the measured duration of a
 * completed EDIT update on state (lastExactDurationMs/exactDurationCount)
 * so the owed badge can later say "usually about 40 s". The baseline load
 * passes false — its full-solve duration says nothing about how long an
 * edit's precise update takes.
 */
function updateExactMetrics(manifest, decodedWindow, { trackDuration = true } = {}) {
  const grid = state.exact
    ? { rows: state.exact.rows, cols: state.exact.cols }
    : { rows: 500, cols: 500 };
  const derived = computeExactMetrics(manifest, decodedWindow, grid);
  if (trackDuration && Number.isFinite(derived.durationMs) && derived.durationMs > 0) {
    state.lastExactDurationMs = derived.durationMs;
    state.exactDurationCount += 1;
  }
  if (derived.meanDeltaC !== undefined) {
    elements.meanDeltaMetric.textContent = formatSignedTemperature(derived.meanDeltaC);
  }
  if (derived.peakDeltaC !== undefined) {
    elements.peakDeltaMetric.textContent = formatSignedTemperature(derived.peakDeltaC);
  }
  if (derived.improvedAreaM2 !== undefined) {
    elements.areaMetric.textContent = `${formatInteger(derived.improvedAreaM2)} m²`;
  }
  if (derived.fraction !== null) {
    elements.roiMetric.textContent = `${(derived.fraction * 100).toFixed(1)}%`;
  }
  if (derived.windowCells !== null && derived.totalCells !== null) {
    elements.roiCaption.textContent =
      `${formatInteger(derived.windowCells)} / ${formatInteger(derived.totalCells)} cells`;
  }
  elements.sceneVersion.textContent = `v${derived.sceneVersion}`;
  if (derived.durationMs !== undefined) {
    elements.computeStep.textContent = `precise · ${(derived.durationMs / 1000).toFixed(2)} s`;
  }
  if (derived.window) {
    elements.meanDeltaCaption.textContent =
      `window rows ${derived.window.rowStart}–${derived.window.rowStop}, ` +
      `cols ${derived.window.colStart}–${derived.window.colStop}`;
  }
}

function fullDirtyRectForCurrentTrees() {
  if (state.trees.length === 0) return { minU: 0, minV: 0, maxU: 1, maxV: 1 };
  return unionRects(
    state.trees.map((tree) => dirtyRectForEdit(null, tree, state.hour, scene)),
  );
}

function scheduleAnalysis(rect, reason) {
  // Any job that becomes stale is ignored. Its dirty region therefore remains
  // unapplied and must be included in the next submitted job. Accumulating the
  // regions here prevents rapid edits on different trees from losing updates.
  state.pendingDirtyRect = accumulateDirtyRect(state.pendingDirtyRect, rect, 0.3);
  const normalizedRect = state.pendingDirtyRect;
  state.dirtyRect = normalizedRect;

  if (!state.workerReady) {
    setWorkerState("running", "Waiting for preview worker", reason);
    render();
    return;
  }

  const window = rectToGridWindow(normalizedRect, scene.gridWidth, scene.gridHeight, 4);
  const jobId = ++state.latestJobId;
  state.activeJobId = jobId;
  // Stage 1 — Sketch: the local preview kernel repaints immediately; every
  // later stage is server work that never blocks the sketch.
  setPipelineStage("sketch", "running");
  updateStageDetails({ window });
  setWorkerState("running", connectedMode ? "Preview updated — precise map queued" : "Updating thermal analysis", reason);
  render();

  worker.postMessage({
    type: "analyze",
    jobId,
    revision: state.version,
    trees: state.trees.map((tree) => ({ ...tree })),
    window,
    hour: state.hour,
    minimumDurationMs: 560,
  });
}

worker.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "ready") {
    state.workerReady = true;
    setWorkerState("ready", "Analysis ready", "Offline scene loaded", 0);
    if (!connectedMode) {
      scheduleAnalysis(fullDirtyRectForCurrentTrees(), "Initializing proposal state");
    }
    return;
  }
  if (message.type === "error") {
    if (!isCurrentWorkerMessage(message, state.latestJobId, state.version)) return;
    setWorkerState("ready", "Preview failed", message.error, 0);
    state.activeJobId = null;
    return;
  }
  if (
    message.type !== "result" ||
    !isCurrentWorkerMessage(message, state.latestJobId, state.version)
  ) return;

  const patch = new Float32Array(message.patch);
  applyPatch(state.current, patch, message.window, scene.gridWidth);
  state.activeJobId = null;
  if (connectedMode) {
    // The preview output is a non-scientific delta layer drawn on top of the
    // exact texture until the exact patch lands (H-1: upload it or the user
    // gets zero approximate feedback while the server refines).
    state.pendingDirtyRect = null;
    renderer.updatePreviewHeatmap(state.current, scene.gridWidth, scene.gridHeight);
    render();
    return;
  }
  state.pendingDirtyRect = null;
  updateMetrics(message.metrics, message.window);
  updateJobSteps(null, { window: message.window, durationMs: message.durationMs });
  setWorkerState(
    "ready",
    "Analysis updated",
    `${formatInteger(message.metrics.affectedCells)} analysis cells patched`,
    message.durationMs / 1000,
  );
  render();
});

// ---------------------------------------------------------------------------
// Exact session (connected mode)
// ---------------------------------------------------------------------------

let session = null;

function requestedResultForCurrentHour() {
  return { timeIndices: [state.hour], variables: ["utci"] };
}

function firstTreeForRecompute() {
  // Legacy recompute anchor gate (pure model in treeForLegacyRecompute, pinned
  // by tests): realtime-native trees are unknown to the legacy funnel — the
  // POST 404s against a tree it never knew. When EVERY tree is realtime-native
  // there is no addressable anchor: callers get null and must SKIP the server
  // call (the sketch plus the realtime fast/exact lanes own the follow-up)
  // instead of minting a job that can only fail.
  return treeForLegacyRecompute(state.trees, state.selectedTreeId);
}

function connectToServer() {
  const client = new ApiClient({ baseUrl: apiBaseUrl });
  session = new ExactSession({
    client,
    callbacks: {
      onStatus: handleSessionStatus,
      onCapabilities: (capabilities) => setCapabilities(capabilities),
      onSiteMismatch: handleSiteMismatch,
      onBaseline: handleBaselineResult,
      onPatchApplied: handlePatchApplied,
      onScope: handleScope,
      onValidation: handleValidation,
      onAuthoritative: handleAuthoritative,
    },
  });
  session.provideRecompute = () => {
    const tree = firstTreeForRecompute();
    return tree ? { tree: treeToApiObject(tree), requested: requestedResultForCurrentHour() } : null;
  };

  setConnectionState("connecting", "Connecting to the server");
  // ``?scenario=<id>`` joins an explicit workspace (the private-sandbox
  // hatch). Without it the session joins the server's ONE shared world via
  // the capability document's default_workspace_id — the join-vs-mint
  // decision lives in ExactSession.connect (single source of truth).
  const joinScenarioId = urlParams.get("scenario");
  session
    .connect({
      siteId,
      name: "SOLWEIG Studio session",
      scenarioId: joinScenarioId,
    })
    .catch((error) => {
      setConnectionState("failed", "Server unreachable");
      if (error.code === "capability_document_unavailable") {
        // Fail visibly: connected mode NEVER falls back to the bundled demo
        // snapshot — an unavailable document means no editors, stated as such.
        setCapabilitiesError(error);
      }
      showFailure(`Could not connect to the server: ${error.message}`);
      announce("The server is unreachable. Your design stays usable as a preview.");
    });
}

// -- Collaborative plane: telemetry + the interactive submit path ------------
//
// The studio's EDIT plane for design gestures in connected mode IS the
// realtime operation plane (scenario id == workspace id): committed gestures
// submit exactly ONE operation per interaction (UI-001) through
// RealtimeClient.submitOperations, reconcile via SSE, and drive the
// five-stage pipeline. The exact session keeps the baseline/result machinery
// (poll, decode, texture patch) and stays the FALLBACK edit plane whenever
// the realtime subscription is not alive — exactly one plane per gesture, so
// a gesture can never double-apply (legacy funnel + native fold).
//
// The revision triple + result-class badge remain telemetry (r2c) with the
// split surfaces from the ux_redesign: #transportPill answers "am I hearing
// the server?", #resultClassBadge answers "can I trust the pixels?" (D3).
const collab = { client: null, subscription: null, workspaceId: null };

// The actor id shared by the realtime client (POST actor_id) and the
// operation-id minting pattern "{actor_id}:op-{client_sequence}".
const realtimeActorId = `studio-${Math.random().toString(36).slice(2, 10)}`;

// Realtime editing bookkeeping (all UI surfaces read from here).
const rt = {
  enabled: false, // true once a live subscription owns the submit plane
  frame: frameFromSpans({ spanX: scene.widthMeters, spanY: scene.heightMeters }),
  frameSolved: false, // true once observed ops recovered the server's frame
  frameSamples: [],
  rows: new Map(), // operationId -> ledger row record
  rowOrder: [], // insertion order, bounded
  roster: [], // observed remote actor ids (presence, first-seen order)
  heldRows: 0,
  lastClassRank: -1,
  lastQualifier: null, // {qualifier, event} of the latest fast_qualified frame
  qualifierOpen: false,
  coveredExactTimes: new Set(), // timesteps carried by the newest exact patch
  lastAppliedExactResult: 0, // exact_revision of the last applied realtime result
  liveRoster: [], // server-confirmed live actors (heartbeat roster field)
  exactProgress: null, // exact_progress view (see exactProgressState)
  exactProgressTimer: 0, // 1 s countdown ticker while an eta is known
  lastExactAtMs: 0, // when this session last APPLIED an exact result
  catchUpBuffer: 0,
  catchUpFlush: null,
  stallChecked: new Set(), // operationIds already receipt-checked (I-26)
};

const RT_MAX_LEDGER_ROWS = 40;

function renderCollabBadge() {
  if (!elements.realtimePill) return;
  applyRealtimeBadge(
    elements.realtimePill,
    elements.realtimePillText,
    realtimeBadgeState(
      collab.workspaceId ? collab.client.revisionsFor(collab.workspaceId) : null,
      collab.workspaceId ? collab.client.classFor(collab.workspaceId) : null,
      collab.subscription?.state ?? "idle",
    ),
  );
  renderMetronomeRail();
}

// -- metronome rail (W·F·E triple, epoch led, actors, mode) ------------------

const railEpoch = { lastWorkspaceRevision: null };

/**
 * One renderer for the whole rail, called from every collab-badge render
 * (each of which already rides an SSE event: operation, canonical, fast,
 * exact, heartbeat). All elements optional — a shell without the rail ids
 * degrades silently.
 */
function renderMetronomeRail() {
  if (!elements.metronomeRail) return;
  const revisions =
    collab.workspaceId && collab.client
      ? collab.client.revisionsFor(collab.workspaceId)
      : null;
  const live = realtimePlaneReady();
  elements.metronomeRail.dataset.mode = live ? "live" : "local";

  if (elements.railWorkspace) {
    elements.railWorkspace.textContent = revisions ? String(revisions.workspaceRevision ?? "—") : "—";
  }
  if (elements.railFast) {
    elements.railFast.textContent = revisions ? String(revisions.fastRevision ?? "—") : "—";
  }
  if (elements.railExact) {
    elements.railExact.textContent = revisions ? String(revisions.exactRevision ?? "—") : "—";
  }
  if (elements.railTriple) {
    const owed =
      revisions &&
      Number.isFinite(revisions.exactRevision) &&
      Number.isFinite(revisions.workspaceRevision) &&
      revisions.exactRevision < revisions.workspaceRevision;
    elements.railTriple.classList.toggle("is-owed", Boolean(owed));
  }
  if (elements.railOwed) {
    // Pure decision (pinned by tests): the chip hides entirely at delta 0 —
    // "exact owed +0" would claim owed work that does not exist.
    const owed = railOwedVisibility(revisions);
    if (elements.railOwed.hidden !== owed.hidden) elements.railOwed.hidden = owed.hidden;
    if (owed.text !== null && elements.railOwed.textContent !== owed.text) {
      elements.railOwed.textContent = owed.text;
    }
  }
  if (elements.epochLed) {
    // Epoch heartbeat: flash the led once per observed workspace revision.
    const w = revisions?.workspaceRevision ?? null;
    if (w !== null && w !== railEpoch.lastWorkspaceRevision) {
      if (railEpoch.lastWorkspaceRevision !== null) {
        elements.epochLed.classList.remove("is-receiving");
        void elements.epochLed.offsetWidth; // restart the one-shot animation
        elements.epochLed.classList.add("is-receiving");
      }
      railEpoch.lastWorkspaceRevision = w;
    }
    elements.epochLed.classList.toggle("is-offline", !live);
  }
  if (elements.railActors) {
    const view = rosterView(rt.roster, liveOthers());
    elements.railActors.hidden = view.live.length === 0 && view.earlier.length === 0;
    if (!elements.railActors.hidden) {
      const names = view.live.map(actorDisplayName).join(" · ");
      const suffix = view.earlier.length > 0 ? ` · +${view.earlier.length} earlier` : "";
      elements.railActors.textContent = `${names}${suffix}`;
      elements.railActors.title =
        `Live now: ${view.live.join(", ") || "nobody else"}. ` +
        `Earlier this session: ${view.earlier.join(", ") || "nobody"}.`;
    }
  }
  if (elements.railMode) {
    elements.railMode.textContent = live ? "LIVE" : "LOCAL";
  }
}

/** True when the realtime plane owns gesture submission right now. */
function realtimePlaneReady() {
  return Boolean(
    rt.enabled && collab.client && collab.workspaceId && collabSubscriptionAlive(collab.subscription),
  );
}

/** Advisory base_revision: the last observed workspace_revision, or null. */
function lastObservedWorkspaceRevision() {
  if (!collab.client || !collab.workspaceId) return null;
  return collab.client.revisionsFor(collab.workspaceId)?.workspaceRevision ?? null;
}

// -- transport pill (D3: transport ≠ trust) -----------------------------------

function renderTransportPill(phase, counts = null) {
  if (!elements.transportPill) return;
  // Compare-guarded writes (setJobStatusDetail pattern): every SSE status
  // re-renders this pill; only a CHANGED state or label touches the DOM.
  // Label copy (including the reconnect budget line) is the pure builder's
  // call — pinned by tests/realtime_ux.test.mjs.
  const [state] = TRANSPORT_PILL_STATES[phase] ?? TRANSPORT_PILL_STATES.closed;
  const label = transportPillLabel(phase, counts ?? {});
  if (elements.transportPill.dataset.state !== state) elements.transportPill.dataset.state = state;
  // The pill ships `hidden` (connected-only widget): the first render on a
  // live subscription plane is what reveals it. Text goes to the inner span
  // so the pulse-dot markup survives.
  elements.transportPill.hidden = false;
  const textElement = elements.transportPillText ?? elements.transportPill;
  if (textElement.textContent !== label) textElement.textContent = label;
}

// -- result-class badge (D1/D2/D4/D5: one trust home, words + symbol) ---------

const GLOSSARY_SENTENCES = {
  fast_exact: "Real physics, answered in about a second.",
  fast_qualified:
    "Physics with a stated margin of error, shown while the precise map is still computing. Open the badge for the error evidence.",
  visual_pending:
    "A placeholder on top of the last verified picture. The precise map is being prepared.",
  exact_reconciled: "The precise result is verified and up to date.",
};

function renderClassBadge() {
  const badge = elements.resultClassBadge;
  if (!badge) return;
  // The shell owns the structured children (glyph / word / owed / stale /
  // age spans + the #qualTrigger button): write INTO them — a
  // badge.textContent write would destroy the trigger and the marker spans.
  const glyph = badge.querySelector(".class-glyph");
  const word = badge.querySelector(".class-word");
  const owed = badge.querySelector(".class-owed");
  const staleSpan = badge.querySelector(".class-stale");
  const age = badge.querySelector(".class-age");
  const qualTrigger = badge.querySelector("#qualTrigger");
  if (!connectedMode || !collab.workspaceId) {
    badge.dataset.state = "visual_pending";
    if (glyph) glyph.textContent = "◌";
    if (word) word.textContent = "Preview";
    if (owed) owed.hidden = true;
    if (staleSpan) staleSpan.hidden = true;
    if (age) age.hidden = true;
    if (qualTrigger) qualTrigger.hidden = true;
    badge.title = "The instant browser-only repaint. A sketch is never a scientific result.";
    return;
  }
  const view = collab.client.classFor(collab.workspaceId);
  const rawClass = view?.class ?? null;
  const revisions = collab.client.revisionsFor(collab.workspaceId);
  // Catch-up override (pristine shared world): W0 with a landed, verified
  // baseline IS the exact picture — no class-bearing frame will ever arrive
  // to say so, because nothing was edited. Without this the badge would read
  // "Preview (owed)" on a world that owes nothing.
  const catchUpExactView = catchUpBadgeView(revisions, { hasBaselineExact: Boolean(state.exact) });
  const known = catchUpExactView
    ? {
        word: catchUpExactView.word,
        symbol: catchUpExactView.glyph,
        rank: RESULT_CLASS_WORDS.exact_reconciled.rank,
      }
    : (RESULT_CLASS_WORDS[rawClass] ?? RESULT_CLASS_WORDS.visual_pending);
  // Unknown future classes display conservatively as Preview (effectiveClass
  // rule) with the raw wire class visible in the tooltip — never a blank or
  // over-claiming badge.
  const superseded = Boolean(view?.superseded);
  const stateWord = catchUpExactView
    ? catchUpExactView.state
    : known === RESULT_CLASS_WORDS.visual_pending && rawClass && !RESULT_CLASS_WORDS[rawClass]
      ? "visual_pending"
      : (rawClass ?? "visual_pending");
  if (badge.dataset.state !== stateWord) badge.dataset.state = stateWord;
  if (glyph && glyph.textContent !== known.symbol) glyph.textContent = known.symbol;
  if (word && word.textContent !== known.word) word.textContent = known.word;
  if (owed) owed.hidden = rawClass !== "visual_pending";
  if (staleSpan) staleSpan.hidden = !superseded;
  const ageMs = Number(view?.ageMs ?? 0);
  if (age) {
    // Priority: a live exact_progress countdown (server ① wave) when the
    // estimate is known; else the legacy elapsed-age line. Both stay inside
    // the honest-vocabulary rule — the eta text comes only from the server.
    const clause = owedDurationText({
      lastExactDurationMs: state.lastExactDurationMs,
      measuredCount: state.exactDurationCount,
    });
    const countdown = exactProgressClause(rt.exactProgress, Date.now());
    if (countdown) {
      age.hidden = false;
      age.textContent = `updating · ${countdown}${clause ? ` · ${clause}` : ""}`;
    } else {
      age.hidden = !(ageMs > 10_000);
      if (ageMs > 10_000) {
        // "updating · 12 s · usually about 50 s" — the age this update has been
        // running, plus the session's MEASURED history when it has one (never
        // an invented number; owedDurationText returns null without one).
        age.textContent = `updating · ${Math.round(ageMs / 1000)} s${clause ? ` · ${clause}` : ""}`;
      }
    }
  }
  if (qualTrigger) {
    qualTrigger.hidden = rawClass !== "fast_qualified";
    qualTrigger.setAttribute("aria-expanded", String(rt.qualifierOpen));
  }
  // I-29/L-6: #exactBadge is a CLASS echo, never a connection word — a
  // "connected" pill must not pin "Exact" onto a picture that is still a
  // preview. Same trio vocabulary as the offline path: exact (Viridian),
  // preview (Ochre dashed), connecting (hollow, only until the first
  // class-bearing frame). Superseded and unknown wire classes display
  // conservatively as Preview. Catch-up: once the revision triple is KNOWN,
  // "Connecting" would overstate transport (the subscription is live and the
  // pixels are honestly a preview), and a pristine baseline-exact world
  // echoes Exact. Writes are compare-guarded (anti-flicker).
  if (elements.exactBadge) {
    const echo = exactBadgeEcho({
      rawClass,
      superseded,
      revisionsKnown:
        Number(revisions?.workspaceRevision ?? 0) > 0 || Number(revisions?.fastRevision ?? 0) > 0,
      catchUpExact: Boolean(catchUpExactView),
    });
    if (elements.exactBadge.textContent !== echo.text) {
      elements.exactBadge.textContent = echo.text;
    }
    if (elements.exactBadge.dataset.state !== echo.state) {
      elements.exactBadge.dataset.state = echo.state;
    }
  }
  const tooltipLines = [
    GLOSSARY_SENTENCES[
      catchUpExactView
        ? "exact_reconciled"
        : rawClass && RESULT_CLASS_WORDS[rawClass]
          ? rawClass
          : "visual_pending"
    ],
  ];
  if (!RESULT_CLASS_WORDS[rawClass] && rawClass) {
    tooltipLines.push(`wire class: ${rawClass} (displayed conservatively)`);
  }
  if (superseded) {
    tooltipLines.push(
      `This picture describes R${view.targetRevision ?? "?"}; the scene is at R${lastObservedWorkspaceRevision() ?? "?"}. Refreshing.`,
    );
  }
  if (ageMs > 10_000) {
    tooltipLines.push(`updating · ${Math.round(ageMs / 1000)} s`);
  }
  if (revisions) {
    tooltipLines.push(
      `W${revisions.workspaceRevision} F${revisions.fastRevision} E${revisions.exactRevision}`,
    );
  }
  // Ladder key (last line, a footer): the symbol vocabulary is used HERE, so
  // it is discovered HERE — ■ ● ◐ ◌ need no external legend to decode.
  tooltipLines.push(ladderLegendLine());
  badge.title = tooltipLines.join("\n");
  // Degradation ladder: animate only UPWARD transitions (I-20 edge) — the
  // reassuring one to celebrate; downgrades ride the badge text + ledger.
  const rank = known.rank;
  if (rank > rt.lastClassRank) {
    badge.dataset.advancing = "true";
    setTimeout(() => {
      badge.dataset.advancing = "false";
    }, 900);
  }
  rt.lastClassRank = rank;
}

// -- qualifier panel (§5 outline; opens from any "~", D5) ---------------------

function qualifierLine(parent, label, value) {
  const row = document.createElement("div");
  row.className = "qual-row";
  row.append(
    Object.assign(document.createElement("span"), { className: "qual-key", textContent: label }),
    Object.assign(document.createElement("span"), { className: "qual-value", textContent: value }),
  );
  parent.append(row);
}

function renderQualifierPanel() {
  const panel = elements.qualifierPanel;
  if (!panel) return;
  if (!rt.qualifierOpen || !rt.lastQualifier) {
    panel.hidden = true;
    return;
  }
  const { qualifier, event } = rt.lastQualifier;
  const revisions = collab.client?.revisionsFor(collab.workspaceId) ?? {};
  const exactBase = Number(qualifier.exact_base_revision ?? event?.exact_base_revision ?? 0);
  const workspace = Number(revisions.workspaceRevision ?? event?.workspace_revision ?? 0);
  panel.replaceChildren();
  // 1. Header — class chip + anchors.
  const header = document.createElement("h3");
  header.textContent =
    `Close estimate for revision R${workspace} · fast R${revisions.fastRevision ?? "?"} · ` +
    `built on precise R${exactBase}`;
  panel.append(header);
  // 2. Plain meaning — the glossary sentence, verbatim.
  panel.append(
    Object.assign(document.createElement("p"), {
      className: "qual-meaning",
      textContent: GLOSSARY_SENTENCES.fast_qualified,
    }),
  );
  // 3-5. Error evidence, coverage, limitations (server fields, verbatim).
  const evidence = qualifier.error_evidence ?? {};
  if (Object.keys(evidence).length > 0) {
    const evidenceBlock = document.createElement("div");
    evidenceBlock.append(Object.assign(document.createElement("h4"), { textContent: "Error evidence" }));
    for (const [key, value] of Object.entries(evidence)) {
      qualifierLine(evidenceBlock, key, typeof value === "object" ? JSON.stringify(value) : String(value));
    }
    panel.append(evidenceBlock);
  }
  const coverage = qualifier.coverage ?? {};
  if (Object.keys(coverage).length > 0) {
    const coverageBlock = document.createElement("div");
    coverageBlock.append(Object.assign(document.createElement("h4"), { textContent: "Coverage" }));
    for (const [key, value] of Object.entries(coverage)) {
      qualifierLine(coverageBlock, key, typeof value === "object" ? JSON.stringify(value) : String(value));
    }
    panel.append(coverageBlock);
  }
  const limitations = [...(qualifier.limitations ?? [])];
  if (limitations.length > 0) {
    const limitationBlock = document.createElement("div");
    limitationBlock.append(Object.assign(document.createElement("h4"), { textContent: "Limitations" }));
    const list = document.createElement("ul");
    for (const text of limitations) {
      list.append(Object.assign(document.createElement("li"), { textContent: String(text) }));
    }
    limitationBlock.append(list);
    panel.append(limitationBlock);
  }
  // 6. Expected catch-up.
  panel.append(
    Object.assign(document.createElement("p"), {
      className: "qual-row",
      textContent: `The precise map is catching up — revision R${exactBase} → R${workspace}; it arrives on its own.`,
    }),
  );
  // 7. Anchor line.
  panel.append(
    Object.assign(document.createElement("p"), {
      className: "qual-row",
      textContent:
        `built on the verified precise result for R${exactBase}; the difference R${exactBase}→R${workspace} ` +
        `is what is being estimated.`,
    }),
  );
  // 8. Link — Model scope card.
  qualifierLine(
    panel,
    "model",
    `${elements.modelVersionValue?.textContent ?? "?"} · site cache ${elements.siteCacheVersionValue?.textContent ?? "?"} — see the Model scope card`,
  );
  panel.hidden = false;
}

function noteFastQualifier(event) {
  const qualifier = event?.payload?.qualifier ?? null;
  if (event?.result_class === "fast_qualified" && qualifier) {
    rt.lastQualifier = { qualifier, event };
  } else {
    rt.lastQualifier = null;
    if (rt.qualifierOpen) {
      rt.qualifierOpen = false;
      renderQualifierPanel();
    }
  }
}

// -- presence (observed actors; honest, never a guaranteed roster) ------------

/**
 * Friendly ordinal for a remote actor ("Editor 1", "Editor 2"): stable for
 * the whole session because the roster is first-seen ordered and never
 * reordered. Raw actor ids ride the element title instead of the visible
 * line — a truncated id is noise to a designer, an ordinal is not.
 */
function actorDisplayName(actorId) {
  const index = rt.roster.indexOf(actorId);
  return index >= 0 ? `Editor ${index + 1}` : "Editor";
}

function observeActor(actorId) {
  const roster = presenceFromOperations([...rt.roster, actorId], realtimeActorId);
  if (roster.length !== rt.roster.length) {
    rt.roster = roster;
    renderPresenceBar();
  }
}

// The live tier means "who ELSE is here": the server roster honestly
// includes the local actor (we subscribe with our own id), but rendering
// yourself as a collaborator is noise. Filter at the render seam;
// rt.liveRoster stays raw server truth.
function liveOthers() {
  return rt.liveRoster.filter((actorId) => actorId !== realtimeActorId);
}

function renderPresenceBar() {
  const bar = elements.presenceBar;
  if (!bar) return;
  bar.dataset.count = String(rt.roster.length);
  // Connected-only widget: revealed by the first render on the realtime plane.
  bar.hidden = false;
  // Three tiers, server-truth first: heartbeat-roster actors are LIVE now;
  // actors observed editing earlier this session but absent from the roster
  // are "+N earlier" (their connection closed — the roster self-corrects
  // within one 5 s heartbeat tick, so no client-side TTL guesswork).
  const view = rosterView(rt.roster, liveOthers());
  const liveNames = view.live.map(actorDisplayName).join(" · ");
  const earlierCount = view.earlier.length;
  const earlierClause = earlierCount > 0 ? ` · +${earlierCount} earlier` : "";
  bar.textContent =
    view.live.length > 0
      ? `with ${liveNames}${earlierClause}`
      : earlierCount > 0
        ? `solo session${earlierClause}`
        : "solo session";
  bar.title =
    `Live now: ${view.live.join(", ") || "nobody else"}. ` +
    `Earlier this session: ${view.earlier.join(", ") || "nobody"}. ` +
    `Live list is the server's heartbeat roster; "earlier" are people whose ` +
    `edits this session saw but whose connection has since closed.`;
}

function renderHeldCount() {
  const chip = elements.heldCount;
  if (!chip) return;
  chip.textContent = `${rt.heldRows} held`;
  renderActivitySummary();
  chip.hidden = rt.heldRows === 0;
  chip.dataset.tone = rt.heldRows > 2 ? "saturated" : "held";
}

// -- timeline freshness (per-timestep basis dots) ------------------------------

const timelineFreshnessKey = { last: null };

function renderTimelineFreshness() {
  const strip = elements.timelineFreshness;
  if (!strip || !state.exact) return;
  const revisions = collab.client && collab.workspaceId
    ? collab.client.revisionsFor(collab.workspaceId) ?? {}
    : {};
  const workspace = Number(revisions.workspaceRevision ?? 0);
  const exact = Number(revisions.exactRevision ?? 0);
  const exactCurrent = exact > 0 && exact >= workspace;
  // Rebuild only when the picture can change (verify-review F11): every SSE
  // event calls this renderer, and rewriting up to 64 identical dots each
  // time is pure DOM churn.
  const key = `${workspace}:${exact}:${state.exact.displayedHour}:${[
    ...rt.coveredExactTimes,
  ].join(",")}`;
  if (key === timelineFreshnessKey.last) return;
  timelineFreshnessKey.last = key;
  const steps = Math.min(state.exact.timeSteps, 64);
  const dots = [];
  for (let timestep = 0; timestep < steps; timestep += 1) {
    // The .fresh-dot vocabulary (styles.css §timeline): is-exact / is-cached /
    // is-hollow carry the basis, is-active rings the hour you are looking at.
    // The static shells ship the same vocabulary, so JS and markup agree.
    const covered = rt.coveredExactTimes.has(timestep);
    const basis = exactCurrent && covered ? "exact" : exact > 0 ? "cached" : "never";
    const dot = document.createElement("span");
    dot.className =
      basis === "exact"
        ? "fresh-dot is-exact"
        : basis === "cached"
          ? "fresh-dot is-cached"
          : "fresh-dot is-hollow";
    if (timestep === state.exact.displayedHour) dot.classList.add("is-active");
    dot.dataset.fresh = basis;
    dot.dataset.hour = String(timestep);
    dot.title =
      basis === "exact"
        ? `hour ${timestep}: exact for the current revision R${workspace}`
        : basis === "cached"
          ? `hour ${timestep}: cached exact from an older revision (last exact R${exact})`
          : `hour ${timestep}: never computed exactly`;
    dots.push(dot);
  }
  strip.replaceChildren(...dots);
}

// -- activity ledger (rows with receipts; one row per operation_id) -----------

const RT_ROW_STATE_LABELS = {
  sending: "sending",
  held: "held",
  applied: "applied",
  refused: "refused",
  gap: "caught up",
};

function ledgerRowElement(row) {
  const item = document.createElement("li");
  item.className = "rt-ledger-row";
  item.dataset.state = row.state;
  if (row.remote) item.classList.add("is-remote");
  if (row.pulse) item.classList.add("pulse-remote");
  const head = document.createElement("div");
  head.className = "rt-row-head";
  const summary = document.createElement("span");
  summary.className = "rt-row-summary";
  summary.textContent = row.remote
    ? `${actorDisplayName(row.actorId)} · ${row.label}`
    : row.label;
  if (row.remote && row.actorId) summary.title = `actor ${row.actorId}`;
  const stateChip = document.createElement("span");
  stateChip.className = "rt-row-state";
  stateChip.textContent =
    row.state === "held" && row.heldKind === "network"
      ? "Held — offline"
      : row.state === "applied" && row.appliedRevision !== null
        ? `applied · R${row.appliedRevision}`
        : RT_ROW_STATE_LABELS[row.state] ?? row.state;
  // Per-row five-stage micro strip (the big pipeline bar speaks for the
  // SESSION; this speaks for the one operation): the pure ledgerStageDots
  // model, rendered as compact .rt-stage-dot spans. The dots' own tooltip
  // spells the strip out (title on the group, not per dot — one hover).
  const dotsModel = ledgerStageDots(row);
  const dots = document.createElement("span");
  dots.className = "rt-stage-dots";
  dots.title = dotsModel
    .map(({ stage, state }) => `${stage[0].toUpperCase()}${stage.slice(1)} ${state}`)
    .join(" · ");
  for (const { stage, state } of dotsModel) {
    const dot = document.createElement("span");
    dot.className = "rt-stage-dot";
    dot.dataset.stage = stage;
    dot.dataset.state = state;
    dots.append(dot);
  }
  head.append(summary, stateChip, dots);
  if (row.serverSequence !== null && row.serverSequence !== undefined) {
    const receipt = document.createElement("code");
    receipt.className = "rt-row-receipt";
    receipt.textContent = `seq ${row.serverSequence}`;
    receipt.title =
      `Receipt ${row.operationId}${row.epochId ? ` · epoch ${row.epochId}` : ""} — ` +
      `quote it and anyone can find that exact change. Click to copy.`;
    wireReceiptCopy(receipt, row);
    head.append(receipt);
  } else if (row.operationId) {
    const receipt = document.createElement("code");
    receipt.className = "rt-row-receipt";
    receipt.textContent = row.operationId;
    receipt.title = "Receipt id (no server sequence yet). Click to copy.";
    wireReceiptCopy(receipt, row);
    head.append(receipt);
  }
  item.append(head);
  if (row.baseDivergent) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "rt-advisory-chip";
    chip.textContent =
      `based on R${row.baseDivergent.from} — server merged it into R${row.baseDivergent.to}`;
    chip.title =
      "You edited an older view of the scene. Your edit still applied; the server ordered it " +
      "after everyone else's. Nothing was overwritten.";
    chip.addEventListener("click", () => {
      announce(chip.title);
    });
    item.append(chip);
  }
  const notes = document.createElement("ul");
  notes.className = "rt-row-notes";
  for (const note of row.notes) {
    notes.append(Object.assign(document.createElement("li"), { textContent: note }));
  }
  if (row.state === "held") {
    const heldNote = document.createElement("li");
    heldNote.className = "rt-held-note";
    notes.append(heldNote);
    row.heldNoteEl = heldNote;
  }
  if (notes.children.length > 0) item.append(notes);
  return item;
}

/**
 * Click-to-receipt (the one row action): copying the full quotable receipt
 * (`{operation_id} · seq {n} · epoch {id?}` — receiptQuoteText). Clipboard
 * first; announce() is the fallback when the clipboard API is unavailable or
 * denied, with copy that does NOT claim a copy that didn't happen. The row
 * flashes once so the click always has a visible consequence. Clicking
 * anywhere else on the row does nothing — the handler lives on the receipt
 * element only (stopPropagation keeps it that way).
 */
function wireReceiptCopy(receipt, row) {
  receipt.classList.add("rt-receipt-copy");
  receipt.tabIndex = 0;
  receipt.setAttribute("role", "button");
  const copyReceipt = async () => {
    const quote = receiptQuoteText({
      operationId: row.operationId,
      serverSequence: row.serverSequence ?? null,
      epochId: row.epochId ?? null,
    });
    let copied = false;
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(quote);
        copied = true;
      } catch {
        copied = false; // denied/insecure context — fall through to announce
      }
    }
    announce(copied ? `Receipt copied: ${quote}` : `Receipt: ${quote}`);
    const item = receipt.closest(".rt-ledger-row");
    if (!item) return;
    item.classList.add("is-copied");
    setTimeout(() => item.classList.remove("is-copied"), 900);
  };
  receipt.addEventListener("click", (event) => {
    event.stopPropagation();
    copyReceipt();
  });
  receipt.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    event.stopPropagation();
    copyReceipt();
  });
}

function renderLedgerRow(row) {
  if (!elements.activityRows) return;
  const previous = row.el;
  row.el = ledgerRowElement(row);
  if (previous?.isConnected) previous.replaceWith(row.el);
  else elements.activityRows.prepend(row.el);
  trimLedgerRows();
  renderActivitySummary();
}

function renderActivitySummary() {
  const summary = document.querySelector("#activitySummary");
  if (!summary) return;
  // Held rows stay in rowOrder (trimLedgerRows keeps them), so rowOrder is
  // the single count of live rows — adding heldRows would double-count.
  const total = rt.rowOrder.length;
  const applied = [...rt.rows.values()].filter((r) => r.state === "applied").length;
  if (total === 0) {
    summary.textContent = "No edits yet";
    return;
  }
  const parts = [`${total} edit${total === 1 ? "" : "s"}`];
  if (applied > 0) parts.push(`${applied} applied`);
  if (rt.heldRows > 0) parts.push(`${rt.heldRows} held`);
  summary.textContent = parts.join(" · ");
}

function trimLedgerRows() {
  if (!elements.activityList) return;
  while (rt.rowOrder.length > RT_MAX_LEDGER_ROWS) {
    const oldestId = rt.rowOrder.shift();
    const oldest = rt.rows.get(oldestId);
    if (!oldest) continue;
    if (oldest.state === "held") {
      // Held rows never trim (their 100 ms countdown timer and Map entry
      // must survive until release) — put the id BACK and stop this pass:
      // dropping it from rowOrder here orphans the row forever.
      rt.rowOrder.unshift(oldestId);
      return;
    }
    oldest.el?.remove();
    rt.rows.delete(oldestId);
  }
}

function createLedgerRow({ operationId, envelope = null, label, remote = false, actorId = null }) {
  const row = {
    operationId,
    envelope,
    label,
    remote,
    actorId,
    state: remote ? "applied" : "sending",
    serverSequence: null,
    epochId: null,
    appliedRevision: null,
    baseDivergent: null,
    heldKind: null,
    notes: [],
    heldNoteEl: null,
    heldTimer: null,
    pulse: false,
    el: null,
  };
  rt.rows.set(operationId, row);
  rt.rowOrder.push(operationId);
  return row;
}

function flushCatchUpMarker() {
  rt.catchUpFlush = null;
  const count = rt.catchUpBuffer;
  rt.catchUpBuffer = 0;
  if (!count || !elements.activityRows) return;
  const marker = document.createElement("li");
  marker.className = "rt-ledger-row is-gap";
  marker.dataset.state = "gap";
  marker.textContent = `⋯ caught up ${count} operation${count === 1 ? "" : "s"}`;
  elements.activityRows.prepend(marker);
}

function noteCatchUpOperation() {
  rt.catchUpBuffer += 1;
  if (rt.catchUpFlush !== null) return;
  rt.catchUpFlush = setTimeout(flushCatchUpMarker, 60);
}

// -- submit path (Stamp → Share), held flow, stall receipt lookup --------------

function toClientOperation(envelope) {
  return {
    sourceFamily: envelope.source_family,
    entityId: envelope.entity_id,
    verb: envelope.verb,
    payload: envelope.payload,
    operationId: envelope.operation_id,
    baseRevision: envelope.base_revision,
  };
}

async function submitEnvelope(row) {
  setPipelineStage("stamp", "running");
  try {
    const acks = await collab.client.submitOperations(collab.workspaceId, [
      toClientOperation(row.envelope),
    ]);
    onGestureAcked(row, acks[0]);
  } catch (error) {
    if (error instanceof RealtimeAdmissionError) {
      holdRow(row, error);
      return;
    }
    if (error?.code === "operation_id_reused") {
      releaseHeldRow(row);
      row.state = "refused";
      row.notes.push("Refused — id reused: the original operation is unchanged; nothing was double-applied.");
      renderLedgerRow(row);
      settleRefusedPipeline();
      return;
    }
    if (error?.code === "network_error" || error?.name === "NetworkError") {
      // Network-exhausted submits hold with the original id (I-29): the next
      // epoch's resubmit is idempotent, so nothing can double-apply.
      holdRow(row, {
        kind: "network",
        retryAfterMs: 2000,
        advice: [],
      });
      return;
    }
    releaseHeldRow(row);
    row.state = "refused";
    row.notes.push(
      `Your edit could not be sent (${error?.code ?? "error"}): ${error?.message ?? error}. ` +
        `Your design is preserved — nothing was lost.`,
    );
    renderLedgerRow(row);
    settleRefusedPipeline();
    showFailure(
      `Your edit could not be sent (${error?.code ?? "error"}): ${error?.message ?? error}. ` +
        `Your design is preserved — nothing was lost.`,
    );
  }
}

/**
 * A refused gesture ends the pipeline at Stamp — otherwise the bar stays
 * "Stamp running" forever for a submit that already terminally failed
 * (verify-review F9). Terminal, not failed-looking: the row + banner carry
 * the failure; the bar just stops claiming work is in flight.
 */
function settleRefusedPipeline() {
  setPipelineStage("stamp", "done");
  const stampSmall = pipelineSmall("stamp");
  if (stampSmall) stampSmall.textContent = "refused · not sent";
  if (!elements.jobSteps || STAMP_STAGE_INDEX < 0) return;
  const stampItem = elements.jobSteps.children[STAMP_STAGE_INDEX];
  if (!stampItem) return;
  // is-complete's green check would celebrate a refusal: the stamp lamp
  // carries the Brick failure ring instead.
  stampItem.classList.remove("is-complete");
  stampItem.classList.add("is-failed");
}

function onGestureAcked(row, ack) {
  releaseHeldRow(row);
  row.serverSequence = ack?.serverSequence ?? null;
  row.epochId = ack?.epochId ?? null;
  if (ack?.duplicate) {
    row.notes.push(`duplicate ack — original seq ${ack.serverSequence}`);
  }
  const revision = lastObservedWorkspaceRevision();
  row.state = "applied";
  row.appliedRevision = revision;
  // base_divergent advisory (I-23): the submitted base_revision predates the
  // revision the server merged the gesture into. Advisory only — never a
  // rejection, never a blocking flow.
  const base = row.envelope?.base_revision;
  if (base !== null && base !== undefined && revision !== null && Number(base) < Number(revision)) {
    row.baseDivergent = { from: Number(base), to: Number(revision) };
  }
  rt.stallChecked.delete(row.operationId);
  renderLedgerRow(row);
  // The submit promise resolves at the canonical fence: Stamp + Share are
  // both settled here (the POST ack and the canonical appearance are one
  // observable transition client-side).
  setPipelineStage("share", "done");
  const stampSmall = pipelineSmall("stamp");
  if (stampSmall && row.serverSequence !== null) stampSmall.textContent = `receipt seq ${row.serverSequence}`;
  const shareSmall = pipelineSmall("share");
  if (shareSmall && revision !== null) shareSmall.textContent = `applied · R${revision}`;
  renderCollabBadge();
}

/**
 * Held flow (I-24/I-25): admission rejections are capacity facts with a
 * countdown, never errors. The row auto-resubmits the SAME operation_id
 * when the countdown reaches zero (idempotent — even a slightly-early
 * retry is safe).
 */
function holdRow(row, error) {
  const kind =
    error.kind === "server_saturated"
      ? "server_saturated"
      : error.kind === "network"
        ? "network"
        : "burst";
  const retryMs = Number.isFinite(Number(error.retryAfterMs)) && error.retryAfterMs > 0
    ? Number(error.retryAfterMs)
    : kind === "server_saturated"
      ? 30_000
      : kind === "network"
        ? 2_000
        : 1_500;
  if (row.state !== "held") {
    rt.heldRows += 1;
    renderHeldCount();
  }
  row.state = "held";
  row.heldKind = kind;
  const heldUntil = Date.now() + retryMs;
  const copy =
    kind === "server_saturated"
      ? `Server busy: new analysis work is paused fleet-wide. Held edits resubmit when capacity returns (retry hint ${(retryMs / 1000).toFixed(0)} s). Local sketches keep working.`
      : kind === "network"
        ? `Connection lost. Edits are held with their operation ids and resubmit in order when it returns.`
        : `Held: this workspace is editing faster than the analysis budget. Your edits are safe — keep working.`;
  if (error.advice?.length > 0) {
    row.notes.push(`Tip: ${error.advice.join(" · ")}`);
  }
  renderLedgerRow(row); // (re)creates the held note element
  const note = row.heldNoteEl;
  if (!note) return;
  note.dataset.kind = kind;
  note.dataset.copy = copy;
  if (row.heldTimer !== null) clearInterval(row.heldTimer);
  row.heldTimer = setInterval(() => {
    const remaining = heldUntil - Date.now();
    if (remaining <= 0) {
      // Never fire into a dead plane: the resubmit would POST against a
      // torn-down subscription. The idempotent id makes waiting free — the
      // next tick after the plane is live again sends the identical
      // operation. (Workspace switches clear held timers entirely in
      // startCollabTelemetry's re-establishment branch.)
      if (!realtimePlaneReady()) return;
      clearInterval(row.heldTimer);
      row.heldTimer = null;
      note.textContent = "resubmitting…";
      submitEnvelope(row);
      return;
    }
    const seconds = (remaining / 1000).toFixed(1);
    note.textContent =
      kind === "server_saturated"
        ? `${copy} (retry hint ${seconds} s)`
        : kind === "network"
          ? `${copy} Retrying in ${seconds} s.`
          : `${copy} Resubmitting in ${seconds} s.`;
  }, 100);
}

function releaseHeldRow(row) {
  if (row.state === "held") {
    rt.heldRows = Math.max(0, rt.heldRows - 1);
    renderHeldCount();
  }
  if (row.heldTimer !== null) {
    clearInterval(row.heldTimer);
    row.heldTimer = null;
  }
}

/** I-26: quiet receipt lookup when the ack fence stalls at 10 s. */
async function checkOperationReceipt(operationId) {
  if (!session?.client || !collab.workspaceId) throw new Error("no session");
  const path =
    `/api/v1/workspaces/${encodeURIComponent(collab.workspaceId)}/operations/` +
    encodeURIComponent(operationId);
  const record = await session.client.getResultManifest(path);
  return {
    serverSequence: Number(record?.server_sequence ?? 0) || null,
    epochId: record?.epoch_id ?? null,
  };
}

function handleOperationStall({ operationId }) {
  if (!realtimePlaneReady() || rt.stallChecked.has(operationId)) return;
  rt.stallChecked.add(operationId);
  const row = rt.rows.get(operationId);
  if (!row || row.state !== "sending") return;
  row.notes.push("No receipt yet — checking");
  renderLedgerRow(row);
  checkOperationReceipt(operationId)
    .then((receipt) => {
      if (!receipt.serverSequence) throw new Error("no receipt fields");
      row.notes.push("receipt found by lookup");
      onGestureAcked(row, {
        operationId,
        serverSequence: receipt.serverSequence,
        epochId: receipt.epochId,
        duplicate: false,
      });
    })
    .catch(() => {
      // 404 operation_not_found: the POST never reached durability —
      // resubmit the identical id, never a fresh one.
      row.notes.push(
        "The server has no record of this edit yet — resending the same receipt id.",
      );
      renderLedgerRow(row);
      submitEnvelope(row);
    });
}

// -- gesture → envelope (vegetation in the adapter's world-metre schema) -------

function realtimeTreeEnvelope(operation, oldTree, newTree) {
  const baseRevision = lastObservedWorkspaceRevision();
  if (operation === "add") {
    return treeOperation(realtimeActorId, "add", {
      tree_id: newTree.id,
      ...worldTree(newTree, rt.frame),
    }, baseRevision);
  }
  if (operation === "delete") {
    return treeOperation(realtimeActorId, "delete", { tree_id: oldTree.id }, baseRevision);
  }
  if (operation === "move") {
    // A drag that commits while a property debounce is pending carries BOTH
    // changes (rebase folds the property delta into the pending tree): a
    // position-only move payload would drop the property change, and the
    // deferred commit's old-tree diff would then see only the move. When
    // non-positional fields differ too, send ONE replace with the full
    // adapter schema — exactly one operation per gesture either way.
    const propsChanged =
      oldTree !== null &&
      oldTree !== undefined &&
      (Number(oldTree.heightM) !== Number(newTree.heightM) ||
        Number(oldTree.canopyDiameterM) !== Number(newTree.canopyDiameterM) ||
        Number(oldTree.trunkRatio ?? 0.25) !== Number(newTree.trunkRatio ?? 0.25));
    if (propsChanged) {
      return treeOperation(realtimeActorId, "replace", {
        tree_id: newTree.id,
        ...worldTree(newTree, rt.frame),
      }, baseRevision);
    }
    const world = worldTree(newTree, rt.frame);
    return treeOperation(realtimeActorId, "move", {
      tree_id: newTree.id,
      x_m: world.x_m,
      y_m: world.y_m,
    }, baseRevision);
  }
  return treeOperation(realtimeActorId, "replace", {
    tree_id: newTree.id,
    ...worldTree(newTree, rt.frame),
  }, baseRevision);
}

function submitTreeGestureRealtime(operation, oldTree, newTree, reason) {
  const envelope = realtimeTreeEnvelope(operation, oldTree, newTree);
  if (operation === "add" && newTree) newTree.realtimeNative = true;
  const row = createLedgerRow({
    operationId: envelope.operation_id,
    envelope,
    label: `${reason} · ${operationSummary(envelope, newTree?.label ?? oldTree?.label ?? null)}`,
  });
  renderLedgerRow(row);
  submitEnvelope(row);
}

// -- remote operations: adoption, calibration, presence, pulses ----------------

function adoptRemoteVegetationOperation(operation) {
  const payload = operation?.payload ?? {};
  const values =
    payload.values && typeof payload.values === "object" && !Array.isArray(payload.values)
      ? payload.values
      : payload;
  const treeId = operation.entity_id;
  if (!treeId) return;
  const index = state.trees.findIndex((tree) => tree.id === treeId);
  const existing = index >= 0 ? state.trees[index] : null;
  const xM = Number(values.x_m);
  const yM = Number(values.y_m);
  const u = Number.isFinite(xM) && Number.isFinite(yM) ? worldToUv(xM, yM, rt.frame).u : Number(values.u);
  const v = Number.isFinite(xM) && Number.isFinite(yM) ? worldToUv(xM, yM, rt.frame).v : Number(values.v);
  if (operation.verb === "delete") {
    if (existing) {
      state.trees.splice(index, 1);
      if (state.selectedTreeId === treeId) state.selectedTreeId = state.trees.at(-1)?.id ?? null;
      syncSelectionPanel();
      render();
    }
    return;
  }
  if (!Number.isFinite(u) || !Number.isFinite(v)) return;
  const apiTree = {
    tree_id: treeId,
    u,
    v,
    height_m: Number(values.height_m ?? existing?.heightM ?? 12),
    canopy_diameter_m:
      values.canopy_radius_m !== undefined
        ? Number(values.canopy_radius_m) * 2
        : (values.canopy_diameter_m ?? existing?.canopyDiameterM ?? 8),
    trunk_ratio: Number(values.trunk_ratio ?? existing?.trunkRatio ?? 0.25),
  };
  const mapped = apiObjectToTree(apiTree, state.treeOrdinal + 1);
  mapped.realtimeNative = true;
  if (existing) {
    state.trees[index] = { ...mapped, label: existing.label, preset: existing.preset, crown: existing.crown };
  } else {
    state.treeOrdinal += 1;
    state.trees.push(mapped);
  }
  syncSelectionPanel();
  render();
}

/**
 * Adopt a remote `building_geometry` operation (echoes of other actors'
 * add/move/replace/delete). Wire corners arrive in the server's world CRS
 * and fold back to local scene metres through the live frame
 * (buildingFootprintFromFrame); presentation (label/preset) survives an
 * adoption over an existing building, mirroring the vegetation adoption.
 */
function adoptRemoteBuildingOperation(operation) {
  const payload = operation?.payload ?? {};
  const values =
    payload.values && typeof payload.values === "object" && !Array.isArray(payload.values)
      ? payload.values
      : payload;
  const buildingId = operation.entity_id;
  if (!buildingId) return;
  const index = state.buildings.findIndex((building) => building.id === buildingId);
  const existing = index >= 0 ? state.buildings[index] : null;
  if (operation.verb === "delete") {
    if (existing) {
      state.buildings.splice(index, 1);
      if (state.selectedBuildingId === buildingId) {
        state.selectedBuildingId = state.buildings.at(-1)?.id ?? null;
      }
      syncSelectionPanel();
      render();
    }
    return;
  }
  const wireFootprint = Array.isArray(values.footprint_m) ? values.footprint_m : null;
  if (!wireFootprint || wireFootprint.length < 3) return; // malformed: audited server-side
  const localFootprint = buildingFootprintFromFrame(wireFootprint, scene, rt.frame);
  const adopted = apiValuesToBuilding(
    {
      footprint_m: localFootprint,
      height_m: Number(values.height_m ?? existing?.heightM ?? 10),
    },
    existing,
  );
  adopted.id = buildingId;
  adopted.realtimeNative = true;
  // Recenter placement u/v from the adopted ring (apiValuesToBuilding leaves
  // u/v untouched by design — the scene lives here, not in the values).
  const center = footprintCenterM(adopted.footprintM);
  const centerUv = sceneWorldToUv(center.xM, center.yM, scene);
  adopted.u = centerUv.u;
  adopted.v = centerUv.v;
  if (existing) {
    state.buildings[index] = adopted;
  } else {
    // A remote add carries geometry only: name it by the measured ring
    // (preset-matched when the dimensions fit one, else a plain ordinal) —
    // the renderer's palette falls back gracefully on an unknown preset.
    // The suffix derives from the CURRENT world size, not a session counter:
    // journal replay re-runs this branch, and a counter would inflate the
    // label on every replay ("Building 05" for the second building).
    const suffix = String(state.buildings.length + 1).padStart(2, "0");
    const matched = Object.entries(BUILDING_PRESETS).find(
      ([, preset]) =>
        Math.abs(preset.widthM - adopted.widthM) <= 0.5 &&
        Math.abs(preset.depthM - adopted.depthM) <= 0.5 &&
        Math.abs(preset.heightM - adopted.heightM) <= 0.5,
    );
    adopted.label = matched ? `${matched[1].label} ${suffix}` : `Building ${suffix}`;
    adopted.preset = matched ? matched[0] : null;
    state.buildings.push(adopted);
  }
  syncSelectionPanel();
  render();
}

function noteRemoteOperation(rawOperation, origin) {
  // The client normalizes wire records to camelCase ({operationId, actorId,
  // sourceFamily, ...}); accept either shape so this stays shape-stable.
  const operation = {
    operation_id: rawOperation?.operationId ?? rawOperation?.operation_id ?? null,
    actor_id: rawOperation?.actorId ?? rawOperation?.actor_id ?? null,
    source_family: rawOperation?.sourceFamily ?? rawOperation?.source_family ?? null,
    entity_id: rawOperation?.entityId ?? rawOperation?.entity_id ?? null,
    verb: rawOperation?.verb ?? null,
    payload: rawOperation?.payload ?? null,
    server_sequence: rawOperation?.serverSequence ?? rawOperation?.server_sequence ?? null,
  };
  const actorId = operation.actor_id;
  if (actorId) observeActor(actorId);
  const own = actorId !== null && actorId === realtimeActorId;
  if (operation.source_family === "vegetation_geometry") {
    // Calibration: a funnelled op carries true world x_m/y_m for a tree the
    // client knows at (u, v) — two distinct positions recover the frame.
    // Only UNMOVED ops calibrate (see the correspondence rule above the
    // samplers); the candidate frame must fit the WHOLE sample set before
    // it locks.
    const values = operation.payload?.values ?? {};
    const local = own ? null : state.trees.find((tree) => tree.id === operation.entity_id);
    const sample =
      local && !rt.frameSolved ? vegetationFrameSample(local, values, rt.frame, scene) : null;
    if (sample) {
      rt.frameSamples.push(sample);
      const solved = solveFrameFromSamples(rt.frameSamples);
      if (solved && frameResidualOk(rt.frameSamples, solved)) {
        rt.frame = solved;
        rt.frameSolved = true;
      } else if (solved) {
        // A candidate that some sample contradicts is not trusted: drop the
        // violators so calibration converges on the consistent subset.
        rt.frameSamples = rt.frameSamples.filter((entry) =>
          frameResidualOk([entry], solved),
        );
      }
    }
    // Own native ops are already applied locally at gesture-commit time —
    // adopting the echo could clobber a newer local drag. Remote ops (and
    // legacy-funnelled history) adopt: the collaborative state is truth.
    if (!own) adoptRemoteVegetationOperation(operation);
  }
  if (operation.source_family === "building_geometry") {
    // Same calibration treatment, gated the same way: only an op whose ring
    // did NOT move (a height-only replace, say) is a true world sample for
    // the local ring — a move's wire corner describes the NEW position and
    // would poison the frame (see the correspondence rule above).
    const values = operation.payload?.values ?? {};
    const local = own ? null : state.buildings.find((b) => b.id === operation.entity_id);
    const sample =
      local && !rt.frameSolved
        ? buildingFrameSample(local, values.footprint_m, rt.frame, scene)
        : null;
    if (sample) {
      rt.frameSamples.push(sample);
      const solved = solveFrameFromSamples(rt.frameSamples);
      if (solved && frameResidualOk(rt.frameSamples, solved)) {
        rt.frame = solved;
        rt.frameSolved = true;
      } else if (solved) {
        rt.frameSamples = rt.frameSamples.filter((entry) =>
          frameResidualOk([entry], solved),
        );
      }
    }
    if (!own) adoptRemoteBuildingOperation(operation);
  }
  if (own) return; // own op: the ledger row already exists
  const entityLabel =
    state.trees.find((tree) => tree.id === operation.entity_id)?.label ??
    state.buildings.find((building) => building.id === operation.entity_id)?.label ??
    null;
  const row = createLedgerRow({
    operationId: operation.operation_id,
    label: operationSummary(operation, entityLabel),
    remote: true,
    actorId,
  });
  row.pulse = true;
  row.serverSequence = Number(operation.server_sequence ?? 0) || null;
  row.epochId = null; // per-op epoch rides only the submit ack, not the stream
  row.appliedRevision = null;
  if (origin === "catch_up") {
    // Gap recovery compresses remote ops into one marker line (I-27).
    row.el?.remove();
    rt.rows.delete(operation.operation_id);
    rt.rowOrder = rt.rowOrder.filter((id) => id !== operation.operation_id);
    noteCatchUpOperation();
    return;
  }
  renderLedgerRow(row);
}

// -- fast / exact events: paint + verify stages, timeline, exact bridge --------

function markRowsPainted(revision) {
  let painted = false;
  for (const row of rt.rows.values()) {
    if (row.state === "applied" && row.paintedRevision === undefined && revision !== null) {
      row.paintedRevision = revision;
      painted = true;
      // The row's stage strip has a new paint:done dot to show (the state
      // chip itself does not change on paint — this re-render is why the
      // strip can be honest per-row).
      renderLedgerRow(row);
    }
  }
  if (painted) {
    setPipelineStage("paint", "done");
    const paintSmall = pipelineSmall("paint");
    if (paintSmall) paintSmall.textContent = `fast · R${revision}`;
  }
  // A published fast frame with the exact lane still trailing means Verify is
  // OWED, not idle — an idle grey lamp would understate promised work.
  renderVerifyOwed();
}

/**
 * Verify-owed lamp: exact_revision < workspace_revision after a fast frame
 * marks the Verify stage owed (steady Ochre dashed ring — verify verifies
 * asynchronously, nothing is polled, so there is no pulse). Runs after every
 * fast event; idempotent, and clears itself the moment the exact lane lands.
 */
function renderVerifyOwed() {
  if (!elements.jobSteps || VERIFY_STAGE_INDEX < 0) return;
  const revisions =
    collab.client && collab.workspaceId
      ? collab.client.revisionsFor(collab.workspaceId)
      : null;
  const exact = Number(revisions?.exactRevision ?? 0);
  const workspace = Number(revisions?.workspaceRevision ?? 0);
  const owed =
    workspace > 0 && Number.isFinite(exact) && Number.isFinite(workspace) && exact < workspace;
  const item = elements.jobSteps.children[VERIFY_STAGE_INDEX];
  if (!item) return;
  item.classList.toggle("is-owed", owed);
  if (!owed) return;
  // Same strings the catch-up path renders (verifyOwedTexts is the ONE
  // source — badge word = class word across surfaces).
  const texts = verifyOwedTexts(revisions ?? {});
  const verifySmall = pipelineSmall("verify");
  if (verifySmall && verifySmall.textContent !== texts.lamp) verifySmall.textContent = texts.lamp;
  setJobStatusDetail(texts.detail);
}

/** Verify settled: the exact frame landed at R{revision} — clear the owed lamp. */
function markVerifySettled(revision) {
  const verifySmall = pipelineSmall("verify");
  if (verifySmall) verifySmall.textContent = `precise · R${revision}`;
  setJobStatusDetail("Precise map · ready");
}

// True while THIS session's own exact job owns the #jobStatusDetail line
// (queued/running poll rewrites every ~1 s): catch-up copy must never stomp
// a live local job's status.
let localExactJobActive = false;

/**
 * Catch-up renderer: a visitor joining a settled shared world (or a
 * reconnect) has NO local exact job, yet the workspace's exact state is
 * known from the revision triple — settle the status line and the five-stage
 * bar from that truth instead of the boot defaults ("No exact job yet",
 * "receipt pending", classless lamps). Skipped while a live gesture or a
 * local exact job owns those surfaces (the bar's running mode gates it).
 */
function renderCatchUpState() {
  if (!collab.client || !collab.workspaceId) return;
  // A running bar belongs to the current gesture (or the legacy exact path):
  // catch-up settles only an idle/done bar.
  if (elements.jobSteps && elements.jobSteps.dataset.mode === "running") return;
  const revisions = collab.client.revisionsFor(collab.workspaceId);
  if (!revisions) return;
  const lamps = catchUpPipelineLamps(revisions);
  if (lamps) {
    setPipelineStage(lamps.focus, "done");
    for (const [key, text] of Object.entries(lamps.smalls)) {
      const small = pipelineSmall(key);
      if (small && small.textContent !== text) small.textContent = text;
    }
    if (lamps.verifyOwed) {
      // The owed lamp + the owed detail line, same as the live fast-frame
      // path (renderVerifyOwed writes both).
      renderVerifyOwed();
      return;
    }
  }
  if (localExactJobActive) return;
  const detail = catchUpExactStatusText(revisions, { hasBaselineExact: Boolean(state.exact) });
  if (detail !== null) setJobStatusDetail(detail);
}

/**
 * Verify bridge: the realtime exact lane publishes results the exact session
 * never polled (no legacy job ran). Fetch the published result at the exact
 * revision and reuse the session's decode + region-patch machinery.
 */
async function maybeFetchExactResult(exactRevision) {
  const revision = Number(exactRevision ?? 0);
  if (!revision || !state.exact || revision <= rt.lastAppliedExactResult) return;
  if (!session?.client || !collab.workspaceId) return;
  const manifestPath =
    `/api/v1/scenarios/${encodeURIComponent(collab.workspaceId)}/results/${revision}`;
  try {
    const manifest = await session.client.getResultManifest(manifestPath);
    if (Number(manifest.scene_version) !== revision) return; // keyed elsewhere — skip quietly
    const payload = await session.client.getResultPayload(manifest.payload_url);
    const decoded = await session.client.decodePatch(manifest, payload);
    rt.lastAppliedExactResult = revision;
    rt.lastExactAtMs = Date.now();
    renderLastExactStamp();
    rt.coveredExactTimes = new Set((decoded.timeIndices ?? []).map(Number));
    handlePatchApplied(decoded);
    setPipelineStage("verify", "done");
    markVerifySettled(revision);
  } catch {
    // No published result at that revision key (yet): the verify stage and
    // the badge settle from the exact_revision event alone.
  }
}

// -- subscription wiring --------------------------------------------------------

function startCollabTelemetry(scenarioId) {
  if (!elements.realtimePill || !scenarioId) return;
  // Same scenario with a subscription that still owns its recovery (live,
  // reconnecting, catching_up, ...): nothing to do. A TERMINAL subscription
  // (failed — reconnect budget exhausted — or closed) is torn down and
  // re-established (r2c-review F1).
  if (collab.workspaceId === scenarioId && collabSubscriptionAlive(collab.subscription)) {
    rt.enabled = true;
    syncBuildingControlsEnabled();
    return;
  }
  collab.subscription?.close();
  rt.enabled = false;
  // Full realtime-plane reset (verify-review F4/F10): a re-established
  // subscription is a NEW workspace view — stale ledger rows, held timers,
  // stall checks, roster, and especially lastAppliedExactResult (a
  // carried-over high-water mark would suppress the verify bridge's
  // maybeFetchExactResult on the new scenario) must all start clean.
  for (const row of rt.rows.values()) {
    if (row.heldTimer !== null) clearInterval(row.heldTimer);
    row.el?.remove();
  }
  rt.rows.clear();
  rt.rowOrder = [];
  rt.roster = [];
  rt.stallChecked.clear();
  rt.lastAppliedExactResult = 0;
  rt.lastClassRank = -1;
  rt.lastQualifier = null;
  rt.qualifierOpen = false;
  rt.catchUpBuffer = 0;
  if (rt.catchUpFlush !== null) {
    clearTimeout(rt.catchUpFlush);
    rt.catchUpFlush = null;
  }
  railEpoch.lastWorkspaceRevision = null;
  timelineFreshnessKey.last = null; // force one rebuild for the new view
  rt.heldRows = 0;
  renderHeldCount();
  if (elements.activityRows) elements.activityRows.replaceChildren();
  if (elements.activityMeta) {
    elements.activityMeta.textContent = "live · shared session";
  }
  collab.client = new RealtimeClient({
    baseUrl: apiBaseUrl,
    actorId: realtimeActorId,
    onStatus: (status) => {
      renderCollabBadge();
      // The reconnecting status carries the bounded budget's counts
      // ({attempt, maxRetries}); every other phase ignores them (the pure
      // label builder decides when they are spendable).
      renderTransportPill(status?.phase, {
        attempt: status?.attempt ?? null,
        maxAttempts: status?.maxRetries ?? null,
      });
      if (status?.phase === "live") rt.enabled = true;
      if (status?.phase === "failed" || status?.phase === "closed") rt.enabled = false;
      // Building gestures exist only on the live plane: its availability
      // flips the Add controls' disabled state.
      syncBuildingControlsEnabled();
      if (status?.phase === "catching_up") {
        announce("Live updates paused — catching up from the server.");
      }
    },
    onStall: handleOperationStall,
  });
  collab.workspaceId = scenarioId;
  // subscribeWorkspace fires its synchronous "connecting"/"live" statuses
  // DURING the call — the onStatus sync ran while collab.subscription was
  // still the previous (null) handle, so realtimePlaneReady() read false and
  // the building Add controls stayed disabled on a quiet world (no later
  // phase transition to re-run them). Re-sync now that the handle exists.
  collab.subscription = collab.client.subscribeWorkspace(scenarioId, {
    onOperation: (operation, meta) => {
      noteRemoteOperation(operation, meta?.origin);
      renderCollabBadge();
    },
    onCanonicalRevision: () => {
      renderCollabBadge();
      renderClassBadge();
      renderTimelineFreshness();
      // The subscribe snapshot arrives as a canonical frame: it carries the
      // world's revision triple (but no class-bearing fast/exact frame), so
      // the catch-up surfaces settle here for a visitor joining a settled or
      // owed world.
      renderCatchUpState();
    },
    onFastRevision: (event) => {
      renderCollabBadge();
      renderClassBadge();
      noteFastQualifier(event);
      if (rt.qualifierOpen) renderQualifierPanel();
      const revision = Number(event?.workspace_revision ?? 0);
      if (revision > 0) markRowsPainted(revision);
      renderTimelineFreshness();
      if (event?.latency_ms !== undefined && event?.result_class) {
        const paintSmall = pipelineSmall("paint");
        if (paintSmall) {
          paintSmall.textContent = `fast · ${(Number(event.latency_ms) / 1000).toFixed(2)} s`;
        }
      }
    },
    onExactRevision: (event) => {
      renderCollabBadge();
      renderClassBadge();
      renderTimelineFreshness();
      const revision = Number(event?.exact_revision ?? 0);
      if (revision > 0) {
        setPipelineStage("verify", "done");
        markVerifySettled(revision);
        maybeFetchExactResult(revision);
      }
    },
    onHeartbeat: () => renderCollabBadge(),
    onRoster: (roster) => {
      // Heartbeat roster (server ③ wave): the live set, delivered every 5 s.
      // Fold new live actors into the first-seen order so ordinals stay
      // stable, then re-render the presence surfaces from the split view.
      rt.liveRoster = Array.isArray(roster)
        ? roster.map(String).filter(Boolean)
        : [];
      for (const actorId of rt.liveRoster) observeActor(actorId);
      renderPresenceBar();
      renderCollabBadge();
    },
    onExactProgress: (event) => {
      // exact_progress (server ① wave): mint/dispatch/mode-known keep the
      // countdown alive, terminal transitions retire it. The monotone
      // deadline rule lives in exactProgressState; here only render.
      rt.exactProgress = exactProgressState(rt.exactProgress, event, Date.now());
      syncExactProgressTicker();
      renderClassBadge();
      renderCollabBadge();
    },
  });
  renderCollabBadge();
  renderTransportPill(collab.subscription?.state ?? "connecting");
  renderClassBadge();
  renderPresenceBar();
  renderTimelineFreshness();
  syncBuildingControlsEnabled();
  // Late-join scene rebuild: the SSE snapshot frame is empty of history by
  // design, so a client joining a workspace that already holds operations
  // (a ?scenario= join, a reload, a reconnect after sleep) replays the
  // journal from its watermark. Deduped by operation_id against the live
  // frames that raced the GET — each op applies exactly once. A failure
  // leaves the stream live; the next gap/reconnect recovery retries it.
  collab.client
    .catchUpWorkspace(scenarioId)
    // The GET body carries the revision triple even when no canonical frame
    // raced it — settle the catch-up surfaces from whichever landed last.
    // The plane is fully established only after this resolves, so this is
    // also the deterministic point where the building Add controls un-disable
    // on a world that never emits another phase transition.
    .then(() => {
      renderCatchUpState();
      syncBuildingControlsEnabled();
    })
    .catch((error) => console.warn("catch-up failed:", error?.message ?? error));
  // Wire the static toggles exactly once: a re-established subscription must
  // not stack a second click handler (double-toggle).
  if (elements.activityToggle && !elements.activityToggle.dataset.rtWired) {
    elements.activityToggle.dataset.rtWired = "true";
    elements.activityToggle.addEventListener("click", () => {
      if (!elements.activityList) return;
      elements.activityList.hidden = !elements.activityList.hidden;
      elements.activityToggle.setAttribute(
        "aria-expanded",
        String(!elements.activityList.hidden),
      );
    });
  }
  if (elements.resultClassBadge && !elements.resultClassBadge.dataset.rtWired) {
    elements.resultClassBadge.dataset.rtWired = "true";
    elements.resultClassBadge.addEventListener("click", () => {
      if (!elements.qualifierPanel || !rt.lastQualifier) return;
      rt.qualifierOpen = !rt.qualifierOpen;
      renderQualifierPanel();
    });
  }
}

function handleSessionStatus({ phase, ...detail }) {
  switch (phase) {
    case "connecting":
      setConnectionState("connecting", "Connecting to the server");
      break;
    case "reconnecting":
      if (detail.waking) {
        // Cold boot (scale-to-zero wake, ~40-60 s): the machine is starting,
        // not broken — cycle the wording instead of counting retries.
        setConnectionState(
          "connecting",
          detail.retry % 2 === 1
            ? "Waking the analysis server — up to a minute"
            : "Still waking the analysis server — up to a minute",
        );
        announce("Waking the analysis server. This can take up to a minute.");
        break;
      }
      // Transient connect failure (R-1): retry with bounded backoff.
      setConnectionState(
        "connecting",
        `Server unreachable · retry ${detail.retry} of ${detail.maxRetries ?? session?.maxConnectRetries ?? 3}`,
      );
      announce("The server is unreachable. Retrying to connect.");
      break;
    case "ready":
      setConnectionState("connected", `Connected · ${session?.siteId ?? siteId}`);
      startCollabTelemetry(session?.scenarioId ?? null);
      clearFailure();
      announce("Precise updates ready.");
      break;
    case "queued":
      localExactJobActive = true;
      updateJobSteps(1, {});
      // The builder keeps sceneVersion in its signature (tooltips may spend
      // it); the visible line renders only the wait or the queue position.
      setJobStatusDetail(exactQueuedStatusText(detail));
      // UEDIT-009: classify the dirty-node closure as soon as the edit is
      // accepted; the server-executed plan and exact scope figures arrive
      // with the applied job.
      renderImpact(detail.sourceNodes ?? null, null);
      announce("Precise update queued.");
      break;
    case "running": {
      localExactJobActive = true;
      updateJobSteps(2, { window: detail.window });
      if (detail.window && state.exact) {
        // The server's window is authoritative for the dirty-region outline.
        state.serverDirtyRect = gridWindowToRect(
          detail.window,
          state.exact.rows,
          state.exact.cols,
        );
      }
      // Status copy (full-recompute honesty + the server's honest ETA when
      // it has one) lives in the pure builder above, pinned by tests.
      setJobStatusDetail(exactRunningStatusText(detail));
      if (detail.mode === "full") {
        setWorkerState("running", "Recalculating the whole map", "takes a few minutes");
        announce("Recalculating the whole map. This takes a few minutes.");
      } else {
        // The raw stage name ("time_loop") stays in the job strip's technical
        // detail; the announce speaks plain words (job_eta pins stage names
        // out of visible copy).
        setWorkerState("running", "Precise update running", "updating the map");
        announce("Precise update running — the map is being updated.");
      }
      render();
      break;
    }
    case "applied":
      localExactJobActive = false;
      setJobStatusDetail("Precise map updated");
      updateJobSteps(null, { durationMs: detail.metrics?.duration_ms });
      // UEDIT-009: final stage classification + exact recompute scope. When
      // the job served its executed impact_plan, the server's per-node stages
      // replace the derived closure classification.
      renderImpact(detail.sourceNodes ?? null, detail.metrics ?? null, detail.impactPlan ?? null);
      setWorkerState(
        "ready",
        "Precise map updated",
        "precise result applied",
        (detail.metrics?.duration_ms ?? 0) / 1000,
      );
      clearFailure();
      announce("Precise map updated.");
      break;
    case "superseded":
      localExactJobActive = false;
      setJobStatusDetail("Superseded by a newer edit");
      break;
    case "conflict":
      announce("The scenario changed on the server; reloading authoritative state.");
      break;
    case "conflict_discarded":
      showFailure(`The scenario changed on the server; edit discarded (${detail.reason}).`);
      announce("The scenario changed on the server; edit discarded.");
      break;
    case "held_edits_discarded":
      // Reset during the connecting window dropped held edits (R-2); nothing
      // failed, so announce rather than raising the failure banner.
      announce("Edits held while connecting were discarded by the reset.");
      break;
    case "site_mismatch":
      // Banner + identity diff are rendered by the dedicated onSiteMismatch
      // callback (handleSiteMismatch); this switch only needs to not fall
      // through to the failure banner.
      break;
    case "failed":
      // UI-004: preview and design state remain; offer retry.
      localExactJobActive = false;
      setConnectionState("connected", "Connected · last update failed");
      if (detail.error?.code === "rate_limited") {
        // §4: rate_limited is a COOLDOWN frame, not a failure — the budget
        // refills on a timer and nothing was lost. Wait comes from the
        // server's retry_after_ms (the 2 s fallback matches the session's
        // single automatic retry). "Precise update failed" stays reserved
        // for genuine solver failures below.
        const waitSeconds = Math.max(
          1,
          Math.round(Number(detail.error?.retryAfterMs ?? 2000) / 1000),
        );
        setWorkerState(
          "ready",
          "Scenario rate limit reached",
          `next edit in ${waitSeconds} s. Your design is preserved.`,
        );
        elements.analysisToast.dataset.state = "cooldown";
        // The reassurance clause is idempotent (append-once): this announce
        // can repeat across poll retries and must never double it.
        announce(
          withCapacityReassurance(`Scenario rate limit reached — next edit in ${waitSeconds} s.`),
        );
        break;
      }
      setWorkerState("ready", "Precise update failed", detail.error?.message ?? "unknown error");
      showFailure(
        `Precise update failed (${detail.error?.code ?? "error"}): ` +
          `${detail.error?.message ?? ""} Your design is preserved; you can retry.`,
      );
      announce("Precise update failed. Your design is preserved.");
      break;
    default:
      break;
  }
}

function handleBaselineResult(result) {
  const utci = result.variables.utci;
  if (!utci) {
    showFailure("The server baseline result has no UTCI variable.");
    return;
  }
  const [timeSteps, rows, cols] = result.manifest.variables
    .find((entry) => entry.name === "utci")
    .shape;
  state.exact = {
    cube: utci,
    rows,
    cols,
    timeSteps,
    displayedHour: Math.min(Math.max(state.hour, 0), timeSteps - 1),
  };
  // The baseline result covers every timestep exactly (timeline basis dots).
  rt.coveredExactTimes = new Set(Array.from({ length: timeSteps }, (_, index) => index));
  renderTimelineFreshness();
  state.version = result.manifest.scene_version;
  state.previewStale = false;
  state.serverDirtyRect = null;
  renderer.clearPreviewHeatmap();
  updateExactMetrics(result.manifest, result.window, { trackDuration: false });
  refreshExactTexture();
  render();
}

function handlePatchApplied(result) {
  // The session already validated manifest.scene_version against the awaited
  // version and verified the payload checksum (UI-003). Re-check against the
  // session's known scene version before anything touches the texture.
  if (!state.exact) return;
  const manifest = result.manifest;
  if (Number(manifest.scene_version) < Number(session.sceneVersion)) return;
  const utci = result.variables.utci;
  if (!utci) return;

  const writtenTimeStep = applyExactPatch(state.exact.cube, utci, {
    window: result.window,
    timeIndices: result.timeIndices,
    hour: state.hour,
    timeSteps: state.exact.timeSteps,
    gridWidth: state.exact.cols,
    gridHeight: state.exact.rows,
  });

  if (writtenTimeStep === state.exact.displayedHour) {
    // Upload only the changed sub-rectangle (frontend_spec.md patch rules).
    const width = result.window.colStop - result.window.colStart;
    const height = result.window.rowStop - result.window.rowStart;
    const planeSize = width * height;
    const position = result.timeIndices.findIndex((t) => Number(t) === Number(writtenTimeStep));
    const region = utci.subarray(position * planeSize, (position + 1) * planeSize);
    renderer.updateHeatRegion(region, result.window);
  }

  state.version = manifest.scene_version;
  state.previewStale = false;
  state.pendingDirtyRect = null;
  state.serverDirtyRect = null;
  renderer.clearPreviewHeatmap();
  updateExactMetrics(manifest, result.window);
  render();
}

function handleScope(scope) {
  elements.modelVersionValue.textContent = scope.modelVersion || "—";
  elements.siteCacheVersionValue.textContent = scope.siteCacheVersion || "—";
  elements.modelScopeList.replaceChildren(
    ...scope.limitations.map((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      return item;
    }),
  );
}

/** Server contract field name → schema property name (unit wiring only). */
const CONTRACT_FIELD_ALIASES = {
  canopy_diameter_m: "canopy_radius_m",
  trunk_zone_ratio: "trunk_ratio",
};

function controlForContractField(field) {
  const match = /^edits\[\d+\]\.tree\.([A-Za-z0-9_]+)$/.exec(field ?? "");
  if (!match) return null;
  const leaf = match[1];
  const propertyName = CONTRACT_FIELD_ALIASES[leaf] ?? leaf;
  return (
    state.selectionControls.find(
      (entry) => entry.control.name === propertyName && entry.input,
    ) ?? null
  );
}

function handleValidation({ code, message, field }) {
  // 4xx such as invalid_tree_geometry: surface the field error, keep the
  // design state exactly as the user left it. The offending input is looked
  // up in the GENERATED control registry — no hardcoded field map.
  showFailure(
    `The server rejected ${field ? String(field) : "the edit"}: ${message}. ` +
      `Adjust the highlighted control.`,
  );
  announce(`Edit rejected: ${message}`);
  const entry = controlForContractField(field);
  if (entry) {
    entry.input.classList.add("is-invalid");
    setTimeout(() => entry.input.classList.remove("is-invalid"), 2200);
  }
}

function handleAuthoritative({ trees, sceneVersion, reason }) {
  // Adopt the pinned site's grid placement as the authoritative uv→world
  // frame the moment the server discloses it: realtime vegetation ops carry
  // world metres, and a span-local frame (x_m in [0, span]) puts every op
  // outside real site rasters (UTM origins) — the exact lane then refuses
  // the op for producing no dirty window. Marked solved so calibration
  // samples can never clobber the server's own geometry.
  const geometry = session?.siteGeometry;
  if (
    geometry &&
    geometry.origin_x_m != null &&
    geometry.origin_y_m != null
  ) {
    const frame = frameFromGeometry({
      originX: Number(geometry.origin_x_m),
      originY: Number(geometry.origin_y_m),
      cols: Number(geometry.cols),
      rows: Number(geometry.rows),
      pixelSize: Number(geometry.pixel_size_m),
    });
    // All four finite and BOTH spans positive — a null origin (Number(null)
    // === 0) or NaN y must not adopt a wrong frame with frameSolved=true,
    // which would permanently block the calibration fallback.
    if (
      Number.isFinite(frame.originX) &&
      Number.isFinite(frame.originY) &&
      frame.spanX > 0 &&
      frame.spanY > 0
    ) {
      rt.frame = frame;
      rt.frameSolved = true;
    }
  }
  // Adopt the server's tree list (conflict recovery / reset). Trees that also
  // exist locally keep their local presentation attributes.
  const byId = new Map(state.trees.map((tree) => [tree.id, tree]));
  state.trees = trees.map((apiTree, index) => {
    const local = byId.get(apiTree.tree_id);
    const mapped = apiObjectToTree(apiTree, index + 1);
    return local ? { ...mapped, label: local.label, preset: local.preset, crown: local.crown } : mapped;
  });
  state.treeOrdinal = Math.max(state.treeOrdinal, state.trees.length);
  if (!state.trees.some((tree) => tree.id === state.selectedTreeId)) {
    state.selectedTreeId = state.trees.at(-1)?.id ?? null;
  }
  // The authoritative payload carries no building list (buildings live only
  // in the realtime canonical state): a reset wipes the local rings to match
  // the reset world — an "adopt" reconnect keeps them (the journal replay
  // dedupes against them and would otherwise lose already-applied ops).
  if (reason === "reset") {
    state.buildings = [];
    state.selectedBuildingId = null;
    buildingPropertyEditor.cancel();
  }
  state.version = sceneVersion;
  syncSelectionPanel();
  render();
  if (reason === "reset") {
    state.previewStale = false;
    renderer.clearPreviewHeatmap();
  }
}

// ---------------------------------------------------------------------------
// Committed interactions
// ---------------------------------------------------------------------------

function commitDesignEdit(oldTree, newTree, reason) {
  let operation;
  if (!oldTree) operation = "add";
  else if (!newTree) operation = "delete";
  else {
    const moved = oldTree.u !== newTree.u || oldTree.v !== newTree.v;
    operation = moved ? "move" : "update";
  }
  const rect = dirtyRectForEdit(oldTree, newTree, state.hour, scene);
  if (connectedMode) {
    // Immediate, non-scientific preview feedback always runs (Sketch stage),
    // then the committed interaction rides EXACTLY ONE edit plane (UI-001):
    // the realtime operation plane when a live subscription owns submits,
    // the legacy exact session otherwise — never both, so a gesture cannot
    // double-apply (legacy funnel + native fold).
    state.previewStale = true;
    scheduleAnalysis(rect, reason);
    if (realtimePlaneReady()) {
      submitTreeGestureRealtime(operation, oldTree, newTree, reason);
      return;
    }
    session?.commitEdits({
      edits: [editItemForChange(operation, oldTree, newTree)],
      requested: requestedResultForCurrentHour(),
      label: reason,
      sourceNodes: state.treeAdapterSourceNodes,
    });
    return;
  }
  state.version += 1;
  scheduleAnalysis(rect, reason);
}

function addTree(presetName, u, v) {
  state.treeOrdinal += 1;
  const tree = makeTree(presetName, u, v, state.treeOrdinal);
  state.trees.push(tree);
  state.selectedTreeId = tree.id;
  syncSelectionPanel();
  commitDesignEdit(null, tree, `${tree.label} added`);
}

function removeSelectedTree() {
  const selected = getSelectedTree();
  if (!selected) return;
  const oldTree = cloneTree(selected);
  state.trees = state.trees.filter((tree) => tree.id !== selected.id);
  state.selectedTreeId = state.trees.at(-1)?.id ?? null;
  propertyEditor.cancel();
  syncSelectionPanel();
  commitDesignEdit(oldTree, null, `${oldTree.label} removed`);
}

// Property editing: preview is debounced (150 ms trailing) and the commit is
// debounced inside the spec's 400–600 ms window (UI-002).
const propertyEditor = createPropertyEditor({
  preview: (oldTree, newTree) => {
    scheduleAnalysis(dirtyRectForEdit(oldTree, newTree, state.hour, scene), "Property preview");
  },
  commit: (oldTree, newTree) => {
    if (newTree) commitDesignEdit(oldTree, cloneTree(newTree), `${newTree.label} updated`);
  },
});

function updateSelectedTreeProperty(mutate) {
  const selected = getSelectedTree();
  if (!selected) return;
  mutate(selected);
  syncSelectionPanel();
  // The visual tree updates immediately; only the thermal recompute and the
  // server edit are debounced.
  render();
  propertyEditor.input(selected);
}

// ---------------------------------------------------------------------------
// Buildings (realtime-only family)
//
// Buildings mirror the tree gesture paths — add/drag/nudge/property/delete,
// local preview first, exactly one committed operation per gesture — with
// ONE structural difference: there is no legacy edit funnel for them. Every
// building gesture rides the realtime operation plane; when that plane is
// not live the entry points refuse (never silently drop) and the Add
// controls disable with a "Needs the live server" tooltip.
// ---------------------------------------------------------------------------

/**
 * Regenerate the axis-aligned footprint ring around its CURRENT center for
 * new width/depth (a size edit never moves the building).
 */
function regenerateBuildingFootprint(building, widthM, depthM) {
  const center = footprintCenterM(building.footprintM);
  const halfWidthM = Number(widthM) / 2;
  const halfDepthM = Number(depthM) / 2;
  building.widthM = Number(widthM);
  building.depthM = Number(depthM);
  building.footprintM = [
    [center.xM - halfWidthM, center.yM - halfDepthM],
    [center.xM + halfWidthM, center.yM - halfDepthM],
    [center.xM + halfWidthM, center.yM + halfDepthM],
    [center.xM - halfWidthM, center.yM + halfDepthM],
  ];
}

/** Translate a building (ring + placement u/v) to a new site-UV center. */
function translateBuildingToUv(building, u, v) {
  const next = buildingToWorld({ u, v }, scene);
  const previous = buildingToWorld(building, scene);
  const dxM = next.xM - previous.xM;
  const dyM = next.yM - previous.yM;
  if (dxM === 0 && dyM === 0) return;
  building.footprintM = building.footprintM.map(([xM, yM]) => [xM + dxM, yM + dyM]);
  building.u = u;
  building.v = v;
}

/**
 * Hit-test a canvas point against footprint rings: projected-corner bbox
 * prefilter, then a ray-cast point-in-polygon pass, with
 * distancePointToSegment over the ring edges catching near-miss grabs the
 * ray cast would refuse. Smallest bbox wins so a courtyard pavilion inside
 * a block's ring stays selectable.
 */
function findBuildingAtCanvasPoint(buildings, x, y, scene, edgeTolerancePx = 6) {
  let best = null;
  let bestArea = Infinity;
  for (const building of buildings) {
    const corners = building.footprintM.map(([xM, yM]) => {
      const uv = sceneWorldToUv(xM, yM, scene);
      return uvToCanvas(uv.u, uv.v, scene);
    });
    const minX = Math.min(...corners.map((point) => point.x));
    const maxX = Math.max(...corners.map((point) => point.x));
    const minY = Math.min(...corners.map((point) => point.y));
    const maxY = Math.max(...corners.map((point) => point.y));
    if (x < minX || x > maxX || y < minY || y > maxY) continue;
    let crossings = 0;
    let onEdge = false;
    for (let index = 0; index < corners.length; index += 1) {
      const a = corners[index];
      const b = corners[(index + 1) % corners.length];
      if (distancePointToSegment(x, y, a.x, a.y, b.x, b.y) <= edgeTolerancePx) {
        onEdge = true;
        break;
      }
      if (a.y <= y !== b.y <= y) {
        const t = (y - a.y) / (b.y - a.y);
        if (x < a.x + t * (b.x - a.x)) crossings += 1;
      }
    }
    if (!onEdge && crossings % 2 === 0) continue;
    const area = (maxX - minX) * (maxY - minY);
    if (area < bestArea) {
      best = building;
      bestArea = area;
    }
  }
  return best;
}

function commitBuildingEdit(oldBuilding, newBuilding, reason) {
  const operation = buildingOperationWord(oldBuilding, newBuilding);
  const rect = dirtyRectForBuildingEdit(oldBuilding, newBuilding, state.hour, scene);
  if (!realtimePlaneReady()) {
    // No legacy funnel exists for buildings: a gesture arriving while the
    // plane is down keeps its local sketch (design preserved, like a refused
    // tree submit) but is NOT silently dropped — the announce says it was
    // not shared.
    announce(
      "Buildings need the live server — this change was not shared. Reconnect and repeat it.",
    );
    scheduleAnalysis(rect, reason);
    return;
  }
  // Same Sketch-first discipline as trees: previewStale flags the picture
  // as a preview until the precise job lands (the fast lane does not
  // compensate building edits, so Preview is the honest intermediate).
  state.previewStale = true;
  scheduleAnalysis(rect, reason);
  submitBuildingGestureRealtime(operation, oldBuilding, newBuilding, reason);
}

function submitBuildingGestureRealtime(operation, oldBuilding, newBuilding, reason) {
  const building = newBuilding ?? oldBuilding;
  const verb = buildingWireVerb(operation, oldBuilding, newBuilding);
  const values = buildingWireValues(verb, building);
  if (values.footprint_m) {
    // Local scene metres → the server's world CRS (the same frame the
    // vegetation submit path speaks through worldTree).
    values.footprint_m = buildingFootprintThroughFrame(values.footprint_m, scene, rt.frame);
  }
  const envelope = buildingOperation(
    realtimeActorId,
    verb,
    building.id,
    values,
    lastObservedWorkspaceRevision(),
  );
  if (operation === "add" && newBuilding) newBuilding.realtimeNative = true;
  const row = createLedgerRow({
    operationId: envelope.operation_id,
    envelope,
    label: `${reason} · ${operationSummary(envelope, building.label)}`,
  });
  renderLedgerRow(row);
  submitEnvelope(row);
}

function addBuilding(presetName, u, v) {
  state.buildingOrdinal += 1;
  const building = makeBuilding(presetName, u, v, state.buildingOrdinal, scene);
  state.buildings.push(building);
  // Selection is exclusive across families.
  state.selectedTreeId = null;
  state.selectedBuildingId = building.id;
  syncSelectionPanel();
  commitBuildingEdit(null, building, `${building.label} added`);
}

function addBuildingAtSceneCenter(presetName) {
  if (!BUILDING_PRESETS[presetName]) return;
  if (!realtimePlaneReady()) {
    announce("Buildings need the live server — connect before adding them.");
    return;
  }
  const jitter = ((state.buildingOrdinal % 5) - 2) * 0.02;
  addBuilding(presetName, 0.5 + jitter, 0.36 + jitter);
}

function removeSelectedBuilding() {
  const selected = getSelectedBuilding();
  if (!selected) return;
  const oldBuilding = cloneBuilding(selected);
  state.buildings = state.buildings.filter((building) => building.id !== selected.id);
  state.selectedBuildingId = state.buildings.at(-1)?.id ?? null;
  buildingPropertyEditor.cancel();
  syncSelectionPanel();
  commitBuildingEdit(oldBuilding, null, `${oldBuilding.label} removed`);
}

/**
 * Building property debounce (UI-002, same 150/480 ms discipline as the
 * tree editor). model.mjs's createPropertyEditor cannot be reused verbatim:
 * it deep-copies through cloneTree, whose shallow spread would ALIAS the
 * footprint corner arrays between the pending copy and the live building —
 * a vertex edit on one would leak into the other. cloneBuilding is the
 * exact deep copy for buildings.
 */
function createBuildingPropertyEditor({
  preview,
  commit,
  previewDelayMs = 150,
  commitDelayMs = 480,
} = {}) {
  let editStartBuilding = null;
  let pendingBuilding = null;
  let previewTimer = null;
  let commitTimer = null;

  function flush() {
    if (pendingBuilding === null) return;
    const oldBuilding = editStartBuilding;
    const newBuilding = pendingBuilding;
    editStartBuilding = null;
    pendingBuilding = null;
    commit(oldBuilding, newBuilding);
  }

  return {
    input(currentBuilding) {
      if (pendingBuilding === null) {
        editStartBuilding = currentBuilding ? cloneBuilding(currentBuilding) : null;
      }
      pendingBuilding = currentBuilding ? cloneBuilding(currentBuilding) : null;
      if (previewTimer !== null) clearTimeout(previewTimer);
      previewTimer = setTimeout(() => {
        previewTimer = null;
        preview(editStartBuilding, pendingBuilding);
      }, previewDelayMs);
      if (commitTimer !== null) clearTimeout(commitTimer);
      commitTimer = setTimeout(() => {
        commitTimer = null;
        flush();
      }, commitDelayMs);
    },
    /** Rebase a pending round onto a post-drag building (M-1 parity). */
    rebase(building) {
      if (pendingBuilding === null) return;
      pendingBuilding = building ? cloneBuilding(building) : null;
    },
    cancel() {
      if (previewTimer !== null) clearTimeout(previewTimer);
      if (commitTimer !== null) clearTimeout(commitTimer);
      previewTimer = null;
      commitTimer = null;
      editStartBuilding = null;
      pendingBuilding = null;
    },
  };
}

const buildingPropertyEditor = createBuildingPropertyEditor({
  preview: (oldBuilding, newBuilding) => {
    scheduleAnalysis(
      dirtyRectForBuildingEdit(oldBuilding, newBuilding, state.hour, scene),
      "Property preview",
    );
  },
  commit: (oldBuilding, newBuilding) => {
    if (newBuilding) {
      commitBuildingEdit(oldBuilding, cloneBuilding(newBuilding), `${newBuilding.label} updated`);
    }
  },
});

function updateSelectedBuildingProperty(mutate) {
  const selected = getSelectedBuilding();
  if (!selected) return;
  mutate(selected);
  syncSelectionPanel();
  render();
  buildingPropertyEditor.input(selected);
}

for (const card of elements.componentList.querySelectorAll("[data-preset]")) {
  card.addEventListener("dragstart", (event) => {
    state.dragPreset = card.dataset.preset;
    event.dataTransfer.effectAllowed = "copy";
    event.dataTransfer.setData("text/plain", state.dragPreset);
    elements.sceneCard.classList.add("is-dragging");
  });
  card.addEventListener("dragend", () => {
    state.dragPreset = null;
    state.ghostTree = null;
    elements.sceneCard.classList.remove("is-dragging");
    render();
  });
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      addTreeAtSceneCenter(card.dataset.preset);
    }
  });
}

// Keyboard placement (frontend_spec.md accessibility): every draggable
// component supports an Add command that places it near the scene focus.
for (const button of elements.componentList.querySelectorAll("[data-add-preset]")) {
  button.addEventListener("click", () => addTreeAtSceneCenter(button.dataset.addPreset));
}

function addTreeAtSceneCenter(presetName) {
  if (!TREE_PRESETS[presetName]) return;
  const jitter = ((state.treeOrdinal % 5) - 2) * 0.02;
  addTree(presetName, 0.5 + jitter, 0.36 + jitter);
}

/**
 * Buildings are realtime-only: while no live subscription owns the submit
 * plane the Add buttons disable and the cards stop dragging, each with the
 * same one-line reason as a tooltip. Called at wiring time and on every
 * rt.enabled flip.
 */
function syncBuildingControlsEnabled() {
  if (!elements.buildingList) return;
  // Two gates: the live plane AND the capability document granting the
  // building family. A server that refuses building_geometry would reject
  // every submit — the controls must not invite the attempt. Unknown
  // capabilities (document not yet loaded) keep the plane gate only.
  const familyGranted =
    state.capabilities?.editableAdapters?.some(
      (adapter) => adapter.id === "building_geometry",
    ) ?? true;
  const ready = realtimePlaneReady() && familyGranted;
  const reason = realtimePlaneReady()
    ? ""
    : "Needs the live server";
  for (const button of elements.buildingList.querySelectorAll("[data-add-building]")) {
    button.disabled = !ready;
    button.title = reason;
  }
  for (const button of elements.buildingList.querySelectorAll("[data-add-building]")) {
    button.disabled = !ready;
    button.title = reason;
  }
  for (const card of elements.buildingList.querySelectorAll("[data-building]")) {
    card.draggable = ready;
    card.title = reason;
    card.setAttribute("aria-disabled", String(!ready));
  }
}

if (elements.buildingList) {
  for (const card of elements.buildingList.querySelectorAll("[data-building]")) {
    card.addEventListener("dragstart", (event) => {
      state.dragBuildingPreset = card.dataset.building;
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("text/plain", state.dragBuildingPreset);
      elements.sceneCard.classList.add("is-dragging");
    });
    card.addEventListener("dragend", () => {
      state.dragBuildingPreset = null;
      state.ghostBuilding = null;
      elements.sceneCard.classList.remove("is-dragging");
      render();
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        addBuildingAtSceneCenter(card.dataset.building);
      }
    });
  }
  for (const button of elements.buildingList.querySelectorAll("[data-add-building]")) {
    button.addEventListener("click", () => addBuildingAtSceneCenter(button.dataset.addBuilding));
  }
}
syncBuildingControlsEnabled();

elements.sceneCard.addEventListener("dragover", (event) => {
  event.preventDefault();
  const point = canvasCoordinates(event);
  const uv = canvasToUv(point.x, point.y, scene);
  const presetName = state.dragPreset || event.dataTransfer.getData("text/plain");
  if (TREE_PRESETS[presetName]) {
    if (!isUvInsideSite(uv)) {
      state.ghostTree = null;
    } else {
      state.ghostTree = {
        ...makeTree(presetName, uv.u, uv.v, state.treeOrdinal + 1),
        id: "ghost",
        label: TREE_PRESETS[presetName].label,
      };
    }
    render();
    return;
  }
  const buildingPreset = state.dragBuildingPreset || event.dataTransfer.getData("text/plain");
  if (!BUILDING_PRESETS[buildingPreset]) return;
  if (!isUvInsideSite(uv)) {
    state.ghostBuilding = null;
  } else {
    // Ghost preview rides the renderer's buildings list (it has no dedicated
    // ghost-building parameter): a full-opacity member with a non-matching
    // id, so it never takes the selection ring.
    state.ghostBuilding = {
      ...makeBuilding(buildingPreset, uv.u, uv.v, state.buildingOrdinal + 1, scene),
      id: "ghost",
      label: BUILDING_PRESETS[buildingPreset].label,
    };
  }
  render();
});

elements.sceneCard.addEventListener("drop", (event) => {
  event.preventDefault();
  const point = canvasCoordinates(event);
  const uv = canvasToUv(point.x, point.y, scene);
  const presetName = state.dragPreset || event.dataTransfer.getData("text/plain");
  const buildingPreset = state.dragBuildingPreset || event.dataTransfer.getData("text/plain");
  state.ghostTree = null;
  state.ghostBuilding = null;
  state.dragPreset = null;
  state.dragBuildingPreset = null;
  elements.sceneCard.classList.remove("is-dragging");
  if (TREE_PRESETS[presetName] && isUvInsideSite(uv)) {
    addTree(presetName, uv.u, uv.v);
    return;
  }
  if (BUILDING_PRESETS[buildingPreset]) {
    if (!isUvInsideSite(uv)) return; // outside the site: same silent refuse as trees
    if (!realtimePlaneReady()) {
      announce("Buildings need the live server — this drop was not shared. Reconnect and drop again.");
      return;
    }
    addBuilding(buildingPreset, uv.u, uv.v);
  }
});

elements.overlayCanvas.addEventListener("pointerdown", (event) => {
  if (state.compareBaseline) return;
  const point = canvasCoordinates(event);
  // Trees hit-test first (they draw above buildings), then footprints.
  const hit = findTreeAtCanvasPoint(state.trees, point.x, point.y, scene, 33);
  if (!hit) {
    const buildingHit = findBuildingAtCanvasPoint(state.buildings, point.x, point.y, scene);
    if (!buildingHit) {
      state.selectedTreeId = null;
      state.selectedBuildingId = null;
      syncSelectionPanel();
      render();
      return;
    }
    state.selectedTreeId = null;
    state.selectedBuildingId = buildingHit.id;
    state.pointerDrag = {
      pointerId: event.pointerId,
      kind: "building",
      buildingId: buildingHit.id,
      oldBuilding: cloneBuilding(buildingHit),
    };
    syncSelectionPanel();
    render();
    try {
      elements.overlayCanvas.setPointerCapture(event.pointerId);
    } catch {
      // Capture unavailable — selection stands, drag tracking degrades gracefully.
    }
    return;
  }
  state.selectedTreeId = hit.id;
  state.selectedBuildingId = null;
  state.pointerDrag = {
    pointerId: event.pointerId,
    kind: "tree",
    treeId: hit.id,
    oldTree: cloneTree(hit),
  };
  syncSelectionPanel();
  render();
  // Pointer capture is a drag-tracking enhancement, never a gate: on a capture
  // failure (synthetic events carry no active pointer; a stale pointer id on
  // real hardware does the same) the throw must not skip the selection sync
  // above — the drag still tracks by pointerId on subsequent moves.
  try {
    elements.overlayCanvas.setPointerCapture(event.pointerId);
  } catch {
    // Capture unavailable — selection stands, drag tracking degrades gracefully.
  }
});

elements.overlayCanvas.addEventListener("pointermove", (event) => {
  if (!state.pointerDrag || state.pointerDrag.pointerId !== event.pointerId) return;
  const point = canvasCoordinates(event);
  const uv = canvasToUv(point.x, point.y, scene);
  if (!isUvInsideSite(uv, 0.02)) return;
  const clampedU = Math.min(1, Math.max(0, uv.u));
  const clampedV = Math.min(1, Math.max(0, uv.v));
  if (state.pointerDrag.kind === "building") {
    const building = state.buildings.find(
      (candidate) => candidate.id === state.pointerDrag.buildingId,
    );
    if (!building) return;
    translateBuildingToUv(building, clampedU, clampedV);
    render();
    return;
  }
  const tree = state.trees.find((candidate) => candidate.id === state.pointerDrag.treeId);
  if (!tree) return;
  tree.u = clampedU;
  tree.v = clampedV;
  render();
});

function finishPointerDrag(event) {
  if (!state.pointerDrag || state.pointerDrag.pointerId !== event.pointerId) return;
  const drag = state.pointerDrag;
  state.pointerDrag = null;
  if (elements.overlayCanvas.hasPointerCapture(event.pointerId)) {
    elements.overlayCanvas.releasePointerCapture(event.pointerId);
  }
  if (drag.kind === "building") {
    const building = state.buildings.find((candidate) => candidate.id === drag.buildingId);
    const oldBuilding = drag.oldBuilding;
    if (!building) return;
    if (building.u !== oldBuilding.u || building.v !== oldBuilding.v) {
      // A property debounce may be pending for this building; rebase it onto
      // the dragged position or its late commit would snap it back (M-1).
      buildingPropertyEditor.rebase(building);
      commitBuildingEdit(oldBuilding, cloneBuilding(building), `${building.label} moved`);
    }
    return;
  }
  const tree = state.trees.find((candidate) => candidate.id === drag.treeId);
  const oldTree = drag.oldTree;
  if (tree && (tree.u !== oldTree.u || tree.v !== oldTree.v)) {
    // A property debounce may be pending for this tree; rebase it onto the
    // dragged position or its late commit would snap the tree back (M-1).
    propertyEditor.rebase(tree);
    commitDesignEdit(oldTree, cloneTree(tree), `${tree.label} moved`);
  }
}

elements.overlayCanvas.addEventListener("pointerup", finishPointerDrag);

/**
 * D7 (deliberate behavior change): a CANCELLED drag (pointercancel, Escape)
 * RESTORES the pre-drag position and sends NOTHING — where this shell
 * previously committed a cancel like a drop. The pre-drag snapshot in
 * state.pointerDrag is the restore source; the preview kernel and
 * any pending property debounce rebase onto the restored position.
 */
function cancelPointerDrag(event) {
  if (!state.pointerDrag || state.pointerDrag.pointerId !== event.pointerId) return;
  const drag = state.pointerDrag;
  state.pointerDrag = null;
  if (elements.overlayCanvas.hasPointerCapture(event.pointerId)) {
    elements.overlayCanvas.releasePointerCapture(event.pointerId);
  }
  if (drag.kind === "building") {
    const building = state.buildings.find((candidate) => candidate.id === drag.buildingId);
    const oldBuilding = drag.oldBuilding;
    if (!building) return;
    if (building.u !== oldBuilding.u || building.v !== oldBuilding.v) {
      building.u = oldBuilding.u;
      building.v = oldBuilding.v;
      building.footprintM = oldBuilding.footprintM.map(([xM, yM]) => [xM, yM]);
      buildingPropertyEditor.rebase(building);
      render();
      announce("Drag cancelled — position restored. Nothing was sent.");
    }
    return;
  }
  const tree = state.trees.find((candidate) => candidate.id === drag.treeId);
  const oldTree = drag.oldTree;
  if (tree && (tree.u !== oldTree.u || tree.v !== oldTree.v)) {
    tree.u = oldTree.u;
    tree.v = oldTree.v;
    propertyEditor.rebase(tree);
    render();
    announce("Drag cancelled — position restored. Nothing was sent.");
  }
}

elements.overlayCanvas.addEventListener("pointercancel", cancelPointerDrag);

// Keyboard nudge: arrow keys move the selected tree or building; Shift
// multiplies the step. Commits are debounced exactly like slider edits.
elements.overlayCanvas.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.pointerDrag) {
    cancelPointerDrag({ pointerId: state.pointerDrag.pointerId });
    return;
  }
  const deltas = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, -1],
    ArrowDown: [0, 1],
  };
  const delta = deltas[event.key];
  if (!delta) return;
  const selected = getSelectedTree();
  if (selected) {
    event.preventDefault();
    const step = (event.shiftKey ? 0.02 : 0.004);
    selected.u = Math.min(1, Math.max(0, selected.u + delta[0] * step));
    selected.v = Math.min(1, Math.max(0, selected.v + delta[1] * step));
    render();
    propertyEditor.input(selected);
    return;
  }
  const selectedBuilding = getSelectedBuilding();
  if (!selectedBuilding) return;
  event.preventDefault();
  const step = (event.shiftKey ? 0.02 : 0.004);
  translateBuildingToUv(
    selectedBuilding,
    Math.min(1, Math.max(0, selectedBuilding.u + delta[0] * step)),
    Math.min(1, Math.max(0, selectedBuilding.v + delta[1] * step)),
  );
  render();
  buildingPropertyEditor.input(selectedBuilding);
});

// Property editors are generated per capability document in
// buildSelectionProperties(); their input listeners attach there.

// The delete button serves whichever family owns the (exclusive) selection.
elements.deleteTreeButton.addEventListener("click", () => {
  if (getSelectedBuilding()) {
    removeSelectedBuilding();
    return;
  }
  removeSelectedTree();
});
elements.retryButton.addEventListener("click", () => {
  clearFailure();
  session?.retryLast();
});
elements.rerunButton.addEventListener("click", () => {
  if (connectedMode) {
    const tree = firstTreeForRecompute();
    if (tree) {
      session.requestRecompute({
        tree: treeToApiObject(tree),
        requested: requestedResultForCurrentHour(),
        label: "Manual refinement requested",
      });
    }
    return;
  }
  scheduleAnalysis(fullDirtyRectForCurrentTrees(), "Manual refinement requested");
});

elements.timeSlider.addEventListener("input", () => {
  state.hour = Number(elements.timeSlider.value);
  const time = `${String(state.hour).padStart(2, "0")}:00`;
  elements.timeValue.textContent = time;
  elements.sceneTimeLabel.textContent = `${time} · Summer design day`;
  if (connectedMode && state.exact) {
    // Swap the displayed exact plane immediately (no server round-trip);
    // exact refinement for the new hour is debounced below.
    const displayed = Math.min(Math.max(state.hour, 0), state.exact.timeSteps - 1);
    if (displayed !== state.exact.displayedHour) {
      state.exact.displayedHour = displayed;
      refreshExactTexture();
    }
  }
  render();
  clearTimeout(state.timeTimer);
  state.timeTimer = setTimeout(() => {
    if (connectedMode) {
      // The sketch always refreshes (the local preview kernel answers the new
      // hour immediately); the server round-trip below is the gated half.
      state.previewStale = true;
      scheduleAnalysis(fullDirtyRectForCurrentTrees(), `Solar time changed to ${time}`);
      const tree = firstTreeForRecompute();
      if (!tree) {
        // No time-operation transport exists on the realtime plane (the
        // slider hour is a RESULT view, not a state write — the
        // selected_date_time family carries a date, op_builder.mjs has no
        // hour verb), and every tree is realtime-native so the legacy funnel
        // cannot anchor either: keep the already-swapped verified plane +
        // sketch, let the realtime lanes own the exact refresh, and say so.
        if (realtimePlaneReady()) {
          const revisions =
            collab.client && collab.workspaceId
              ? collab.client.revisionsFor(collab.workspaceId)
              : null;
          announce(
            solarTimeSkipAnnounce(
              time,
              exactHourVerified({
                displayedHour: state.exact?.displayedHour ?? state.hour,
                coveredHours: rt.coveredExactTimes,
                exactRevision: revisions?.exactRevision,
                workspaceRevision: revisions?.workspaceRevision,
              }),
            ),
          );
        }
        return;
      }
      // Time-only changes ride a no-op update edit carrying the requested
      // hour (the edits array requires at least one edit item).
      session.requestRecompute({
        tree: treeToApiObject(tree),
        requested: requestedResultForCurrentHour(),
        label: `Solar time changed to ${time}`,
      });
    } else {
      state.version += 1;
      scheduleAnalysis(fullDirtyRectForCurrentTrees(), `Solar time changed to ${time}`);
    }
  }, 350);
});

elements.compareButton.addEventListener("click", () => {
  state.compareBaseline = !state.compareBaseline;
  elements.compareButton.setAttribute("aria-pressed", String(state.compareBaseline));
  render();
});
elements.heatLayerToggle.addEventListener("change", () => {
  state.showHeat = elements.heatLayerToggle.checked;
  render();
});
elements.shadowLayerToggle.addEventListener("change", () => {
  state.showShadows = elements.shadowLayerToggle.checked;
  render();
});
elements.roiLayerToggle.addEventListener("change", () => {
  state.showDirtyRegion = elements.roiLayerToggle.checked;
  render();
});

elements.resetButton.addEventListener("click", () => {
  if (connectedMode && session?.siteMismatch) {
    // The reset is guaranteed to be refused (409) — refuse BEFORE clearing
    // any local design state. The mismatch banner promises the design stays
    // visible; wiping it locally while the server scene keeps its trees would
    // break exactly that promise.
    showFailure(
      "Reset refused: the scenario's pinned site identity no longer matches the live site. Start a fresh scenario.",
    );
    announce("Reset refused under site identity mismatch. Reconnect to continue.");
    return;
  }
  const previousTrees = state.trees.map(cloneTree);
  const previousSelectedTreeId = state.selectedTreeId;
  const previousBuildings = state.buildings.map(cloneBuilding);
  const previousSelectedBuildingId = state.selectedBuildingId;
  state.trees = [];
  state.selectedTreeId = null;
  state.buildings = [];
  state.selectedBuildingId = null;
  state.current.set(baseline);
  state.pendingDirtyRect = null;
  propertyEditor.cancel();
  buildingPropertyEditor.cancel();
  state.dirtyRect = previousTrees.length ? { minU: 0, minV: 0, maxU: 1, maxV: 1 } : null;
  state.serverDirtyRect = null;
  syncSelectionPanel();
  if (connectedMode && session?.connected) {
    session.reset({ label: "Scenario reset to baseline" }).catch((error) => {
      if (error instanceof ApiClientError && error.code === "site_identity_mismatch") {
        // The refusal arrived with the POST (the mismatch was unknown until
        // now): restore the pre-reset design so it stays visible — the server
        // scene kept its trees. The session already recorded the mismatch and
        // the banner rendered through onSiteMismatch.
        state.trees = previousTrees;
        state.selectedTreeId =
          previousTrees.some((tree) => tree.id === previousSelectedTreeId)
            ? previousSelectedTreeId
            : previousTrees.at(-1)?.id ?? null;
        state.buildings = previousBuildings;
        state.selectedBuildingId =
          previousBuildings.some((building) => building.id === previousSelectedBuildingId)
            ? previousSelectedBuildingId
            : previousBuildings.at(-1)?.id ?? null;
        state.dirtyRect = null;
        state.previewStale = true;
        syncSelectionPanel();
        render();
        scheduleAnalysis(fullDirtyRectForCurrentTrees(), "Reset refused — design restored");
        return;
      }
      showFailure(`Reset failed: ${error.message}`);
    });
    return;
  }
  if (connectedMode && session) {
    // Still connecting (or the last connect failed): the server scene has no
    // edits yet, so the reset's whole job is dropping the held-edit backlog
    // so stale edits cannot replay onto the baseline once a connection lands
    // (R-2). The session clears the backlog and skips the server round-trip.
    session.reset({ label: "Scenario reset to baseline" });
  }
  state.version += 1;
  scheduleAnalysis({ minU: 0, minV: 0, maxU: 1, maxV: 1 }, "Scenario reset to baseline");
});

elements.exportButton.addEventListener("click", async () => {
  const output = document.createElement("canvas");
  output.width = elements.glCanvas.width;
  output.height = elements.glCanvas.height;
  const context = output.getContext("2d");
  context.drawImage(elements.glCanvas, 0, 0);
  context.drawImage(elements.overlayCanvas, 0, 0);
  const blob = await new Promise((resolve) => output.toBlob(resolve, "image/png"));
  if (!blob) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `solweig-studio-v${state.version}.png`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

function updateScopeCardForOffline() {
  elements.exactBadge.textContent = "Preview";
  elements.exactBadge.dataset.state = "preview";
  elements.modelVersionValue.textContent = "browser preview kernel";
  elements.siteCacheVersionValue.textContent = "local fixture";
  elements.modelScopeList.replaceChildren(
    ...[
      "The browser preview kernel is not a scientific SOLWEIG result.",
      "Connect an exact server with ?api=<base-url> for validated results.",
    ].map((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      return item;
    }),
  );
}

// The five-stage pipeline list is built from the shell in BOTH modes (the
// legacy exact path and offline previews map onto the same stages), and the
// result-class badge starts honest: offline/unconnected is Preview ◌.
buildPipelineStages();
renderClassBadge();
// Connected mode honestly says "Connecting" from the first frame — the
// OFFLINE word belongs to offline mode only (never while a server session is
// being established; the join window can last a scale-to-zero wake minute).
renderTransportPill(bootTransportPhase(connectedMode));

if (connectedMode) {
  // Connected mode starts from the server's authoritative baseline scene; the
  // three-tree starter proposal below is the offline demo's teaching fixture.
  setConnectionState("connecting", "Connecting to the server");
  syncSelectionPanel();
  render();
  connectToServer();
} else {
  setConnectionState("local", "Local preview mode");
  updateScopeCardForOffline();
  // Offline demo: the SAME code path runs against a bundled snapshot of the
  // server's capability document (see tools/README note in this directory).
  // Connected mode never uses this file — there is no fallback vocabulary.
  fetch("./assets/capabilities_snapshot.json")
    .then((response) => {
      if (!response.ok) throw new Error(`snapshot HTTP ${response.status}`);
      return response.json();
    })
    .then((document_) => setCapabilities(parseCapabilityDocument(document_)))
    .catch((error) =>
      setCapabilitiesError({
        code: "snapshot_unavailable",
        message: error.message,
      }),
    );
  // Start with a plausible proposal so the prototype communicates before the
  // first interaction. All three objects remain editable and removable.
  const starterTrees = [
    ["shade", 0.515, 0.305],
    ["broad", 0.575, 0.345],
    ["shade", 0.455, 0.365],
  ];
  for (const [preset, u, v] of starterTrees) {
    state.treeOrdinal += 1;
    state.trees.push(makeTree(preset, u, v, state.treeOrdinal));
  }
  state.selectedTreeId = state.trees[1].id;
  state.version = 1;
  state.dirtyRect = fullDirtyRectForCurrentTrees();
  syncSelectionPanel();
  render();
}

// ---------------------------------------------------------------------------
// E2E inspection hook (orca browser tests). Read-only getters over live
// module state — no behavior, no UI. Undefined when disconnected.
// ---------------------------------------------------------------------------
globalThis.__solweigStudio = {
  get scenarioId() {
    return session?.scenarioId ?? null;
  },
  get actorId() {
    return collab.client?.actorId ?? null;
  },
  get workspaceId() {
    return collab.workspaceId ?? null;
  },
  get revisions() {
    return collab.client && collab.workspaceId
      ? collab.client.revisionsFor(collab.workspaceId)
      : null;
  },
  get connectionState() {
    return elements.connectionPill?.dataset.state ?? null;
  },
  // Wave-3 surfaces (read-only, same rule): the announce line, the transport
  // pill's state+label pair, the class badge tooltip, and the ledger rows'
  // stage-strip/receipt wiring.
  get liveStatusText() {
    return elements.liveStatus?.textContent ?? null;
  },
  get transportPill() {
    if (!elements.transportPill) return null;
    return {
      state: elements.transportPill.dataset.state ?? null,
      text:
        elements.transportPillText?.textContent ??
        elements.transportPill.textContent ??
        null,
    };
  },
  get classBadgeTitle() {
    return elements.resultClassBadge?.title ?? null;
  },
  get ledgerStageStrips() {
    const rows = elements.activityRows?.querySelectorAll(".rt-ledger-row") ?? [];
    return [...rows].map((item) => ({
      state: item.dataset.state ?? null,
      dots: [...item.querySelectorAll(".rt-stage-dot")].map(
        (dot) => `${dot.dataset.stage}:${dot.dataset.state}`,
      ),
      receipt: item.querySelector(".rt-row-receipt")?.textContent ?? null,
      receiptCopyable: Boolean(item.querySelector(".rt-receipt-copy")),
    }));
  },
  get treeCount() {
    return state.trees.length;
  },
  get treeIds() {
    return state.trees.map((tree) => tree.id);
  },
  // Building surfaces (same read-only rule) for the collaborative-building
  // E2E loop: count, ids, and ring-center canvas targets.
  get buildingCount() {
    return state.buildings.length;
  },
  get buildingIds() {
    return state.buildings.map((building) => building.id);
  },
  get selectedBuildingId() {
    return state.selectedBuildingId;
  },
  get buildingTargets() {
    return state.buildings.map((building) => {
      const center = footprintCenterM(building.footprintM);
      const uv = sceneWorldToUv(center.xM, center.yM, scene);
      const point = uvToCanvas(uv.u, uv.v, scene);
      return {
        id: building.id,
        fx: point.x / elements.overlayCanvas.width,
        fy: point.y / elements.overlayCanvas.height,
      };
    });
  },
  // Canvas-space hit targets (fraction of canvas width/height) for E2E
  // pointer synthesis: the perspective uvToCanvas warp means proportional
  // guesses miss, these never do.
  get treeTargets() {
    return state.trees.map((tree) => {
      const point = uvToCanvas(tree.u, tree.v, scene);
      return {
        id: tree.id,
        fx: point.x / elements.overlayCanvas.width,
        fy: point.y / elements.overlayCanvas.height,
      };
    });
  },
};
}
