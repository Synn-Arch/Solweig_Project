// SPDX-License-Identifier: GPL-3.0-only
//
// Design-gesture -> realtime operation envelope builder (pure module).
//
// This module owns the CLIENT half of the collaborative operation plane's
// payload grammar (docs/incremental_design_tool/realtime_collaboration/
// collaborative_state.md + ux_redesign/interaction_simulations.md I-19).
// Design gestures — tree add/replace/move/delete, land-cover paints,
// meteorology / model-parameter writes, output-view and date-time
// selections — map onto the wire envelope POST /api/v1/workspaces/{id}/
// operations accepts:
//
//     {operation_id, client_sequence, base_revision, source_family,
//      entity_id, verb, payload}
//
// The vocabulary mirrors the server's frozen contracts exactly:
//
//   * families/verbs — solweig_gpu/server/realtime/types.py
//     (SourceFamily, OperationVerb); transport shape —
//     routes_realtime.validate_operation_shape (non-empty payload object);
//   * payload grammar — solweig_gpu/server/realtime/reducer.py: field
//     values under payload["values"] or flat, land-cover paint carries
//     {window, class}, scalar writes carry time_index / time_start /
//     time_stop at the payload top level, identity keys never need to ride
//     inside values (the entity id is the state key);
//   * vegetation VALUE vocabulary — the adapter's edit-state schema
//     (x_m, y_m, height_m, canopy_radius_m, trunk_ratio in WORLD metres).
//     `treeOperation` spreads the tree object it is given verbatim; the
//     caller decides the vocabulary. The app passes the adapter schema
//     (see `worldTree` below), because the reducer's canonical object
//     state feeds the executor bridge verbatim and the vegetation adapter
//     rejects unknown fields (u, v, canopy_diameter_m, transmissivity...).
//     World-metre positions need the site grid geometry no endpoint
//     exposes, so the UV -> world frame is a first-class concept here
//     (`frameFromSpans` / `uvToWorld` / `solveFrameFromSamples`).
//
// base_revision is advisory metadata: stamped verbatim from the gesture's
// composition context, never validated or coerced here (the server flags
// divergence with the advisory base_divergent ack field; advisory fields
// are excluded from the operation fingerprint, so an idempotent resubmit
// may carry a fresh client_sequence/base_revision with the SAME
// operation_id — store.OperationFingerprintConflict contract).
//
// Everything here is plain-data-in / plain-data-out: no DOM, no fetch, no
// clock — node --test covers it directly (tests/realtime_submit.test.mjs).

// ---------------------------------------------------------------------------
// Frozen vocabularies (server mirrors)
// ---------------------------------------------------------------------------

/** The 7 families the realtime plane carries (types.SourceFamily). */
export const SOURCE_FAMILIES = new Set([
  "building_geometry",
  "vegetation_geometry",
  "landcover_surface",
  "meteorological_forcing",
  "model_receptor_parameters",
  "selected_date_time",
  "output_view",
]);

/** The 7 operation verbs (types.OperationVerb). */
export const OPERATION_VERBS = new Set([
  "add",
  "replace",
  "move",
  "delete",
  "paint",
  "set",
  "select",
]);

/**
 * Legacy universal-adapter operation -> realtime verb (jobs.py
 * `_LEGACY_VERBS` verbatim). Unmappable operations are absent — the
 * converter refuses rather than mis-verb'ing.
 */
export const LEGACY_VERB_MAP = Object.freeze({
  add: "add",
  delete: "delete",
  remove: "delete",
  move: "move",
  paint: "paint",
  select: "select",
  update: "replace",
  update_time_row: "set",
  update_range: "set",
  preset: "set",
  reset: "set",
  set: "set",
});

/** Vegetation position fields the reducer's `move` fold accepts. */
export const VEGETATION_POSITION_FIELDS = Object.freeze(["u", "v", "x_m", "y_m"]);

/** `trunk_ratio` default when a tree model carries none (server contract). */
export const DEFAULT_TRUNK_RATIO = 0.25;

/** Typed refusal for a malformed envelope (mirrors the server's 4xx codes). */
export class OperationShapeError extends Error {
  constructor({ code = "invalid_operation_payload", message, field = null } = {}) {
    super(message ?? "malformed operation envelope");
    this.name = "OperationShapeError";
    this.code = code;
    this.field = field;
  }
}

// ---------------------------------------------------------------------------
// Actor session (operation-id minting)
// ---------------------------------------------------------------------------

/** Per-actor sequence counters (module state; actors count independently). */
const actorSequences = new Map();

/** Next client_sequence for one actor — monotonically increasing from 1. */
export function nextClientSequence(actorId) {
  const key = String(actorId);
  const next = (actorSequences.get(key) ?? 0) + 1;
  actorSequences.set(key, next);
  return next;
}

/** The deterministic operation id for one actor's sequence number. */
export function mintOperationId(actorId, seq) {
  return `${actorId}:op-${seq}`;
}

function envelope(actorId, sourceFamily, entityId, verb, payload, baseRevision) {
  const clientSequence = nextClientSequence(actorId);
  return {
    operation_id: mintOperationId(actorId, clientSequence),
    client_sequence: clientSequence,
    base_revision: baseRevision,
    source_family: sourceFamily,
    entity_id: entityId,
    verb,
    payload,
  };
}

/**
 * One actor's minting session over this module (`createActorSession` ties
 * the counter, the id pattern, and a label together for UI surfaces that
 * quote "your" ids — e.g. held rows re-POSTing the SAME id).
 */
export function createActorSession(actorId) {
  const actor = String(actorId ?? "");
  if (!actor) {
    throw new OperationShapeError({
      code: "invalid_request",
      message: "createActorSession requires a non-empty actorId",
      field: "actor_id",
    });
  }
  return {
    actorId: actor,
    nextClientSequence: () => nextClientSequence(actor),
    mintOperationId: (seq) => mintOperationId(actor, seq),
  };
}

// ---------------------------------------------------------------------------
// UV <-> world frames (vegetation world-metre positions)
// ---------------------------------------------------------------------------

/** Frame from explicit metre spans (origin + the site's metre extents). */
export function frameFromSpans({ originX = 0, originY = 0, spanX = 1, spanY = 1 } = {}) {
  return Object.freeze({
    originX: Number(originX),
    originY: Number(originY),
    spanX: Number(spanX),
    spanY: Number(spanY),
  });
}

/** Frame from site grid geometry (`cols * pixelSize` are the metre spans). */
export function frameFromGeometry({
  originX = 0,
  originY = 0,
  cols = 1,
  rows = 1,
  pixelSize = 1,
} = {}) {
  return frameFromSpans({
    originX,
    originY,
    spanX: Number(cols) * Number(pixelSize),
    spanY: Number(rows) * Number(pixelSize),
  });
}

/** Normalized UV -> world metres (store `uv_to_world` semantics). */
export function uvToWorld(u, v, frame) {
  const f = frame ?? frameFromSpans({});
  return {
    x_m: f.originX + Number(u) * f.spanX,
    y_m: f.originY - Number(v) * f.spanY,
  };
}

/** World metres -> normalized UV (inverse of {@link uvToWorld}). */
export function worldToUv(xM, yM, frame) {
  const f = frame ?? frameFromSpans({});
  return {
    u: (Number(xM) - f.originX) / f.spanX,
    v: (f.originY - Number(yM)) / f.spanY,
  };
}

/**
 * Solve the uv->world affine frame from observed operations.
 *
 * `samples` are `{u, v, x_m, y_m}` records — the client's `(u, v)` for a
 * tree whose legacy-funnelled operation carried true world metres (the
 * funnel converts UV via the site manifest; no endpoint exposes the
 * geometry, but every funnelled op IS two equations). Each axis needs two
 * samples at DISTINCT normalized coordinates. The solved frame matches the
 * TRANSFORM convention exactly: `x = originX + u·spanX` but
 * `y = originY − v·spanY` (world y grows north while v grows south), so
 * the v axis solves with a sign flip — solving the raw slope form would
 * hand back a negative spanY that `worldToUv` then flips twice.
 *
 * A degenerate sample pair (same coordinate, or a zero/infinitesimal
 * solved span) is SKIPPED, not fatal: one bad observation must not stall
 * calibration while good samples keep arriving. Returns a frozen frame,
 * or `null` when no valid pair remains on some axis (caller keeps its
 * fallback frame).
 */
export function solveFrameFromSamples(samples) {
  if (!Array.isArray(samples)) return null;
  const usable = samples.filter(
    (sample) =>
      sample !== null &&
      typeof sample === "object" &&
      [sample.u, sample.v, sample.x_m, sample.y_m].every((value) =>
        Number.isFinite(Number(value)),
      ),
  );
  // `sign` is +1 for the x form (b = o + a·s) and −1 for the y form
  // (b = o − a·s): span = sign · slope, origin = b − sign · a · span.
  const solve = (aKey, bKey, sign) => {
    for (let i = 0; i < usable.length; i += 1) {
      for (let j = i + 1; j < usable.length; j += 1) {
        const a = Number(usable[i][aKey]);
        const b = Number(usable[j][aKey]);
        const delta = b - a;
        if (Math.abs(delta) < 1e-9) continue; // same coordinate: no slope
        const span = sign * (Number(usable[j][bKey]) - Number(usable[i][bKey])) / delta;
        if (!Number.isFinite(span) || Math.abs(span) < 1e-9) continue; // degenerate pair
        const origin = Number(usable[i][bKey]) - sign * a * span;
        return { span, origin };
      }
    }
    return null;
  };
  const x = solve("u", "x_m", 1);
  const y = solve("v", "y_m", -1);
  if (!x || !y) return null;
  return Object.freeze({
    originX: x.origin,
    originY: y.origin,
    spanX: x.span,
    spanY: y.span,
  });
}

/**
 * Whether every observed sample maps through `frame` to its wire metres
 * within `toleranceM`. A candidate frame is only worth LOCKING when the
 * whole sample set agrees — `solveFrameFromSamples` exact-fits the pair it
 * picks, so a pair-level solve says nothing about the samples it ignored.
 */
export function frameResidualOk(samples, frame, toleranceM = 1) {
  if (!frame || !Array.isArray(samples) || samples.length === 0) return false;
  return samples.every((sample) => {
    const mapped = uvToWorld(sample.u, sample.v, frame);
    return (
      Math.abs(mapped.x_m - Number(sample.x_m)) <= toleranceM &&
      Math.abs(mapped.y_m - Number(sample.y_m)) <= toleranceM
    );
  });
}

/**
 * The adapter-schema tree object `treeOperation` should carry: world-metre
 * positions from `(u, v)` + the frame, radius from diameter, and the
 * server's trunk_ratio default. Identity rides as entity_id, so it is not
 * duplicated inside the payload.
 */
export function worldTree(tree, frame) {
  if (!tree || typeof tree !== "object") {
    throw new OperationShapeError({
      code: "invalid_request",
      message: "worldTree requires a tree object",
      field: "tree",
    });
  }
  const world = uvToWorld(tree.u, tree.v, frame);
  return {
    x_m: world.x_m,
    y_m: world.y_m,
    height_m: Number(tree.heightM),
    canopy_radius_m: Number(tree.canopyDiameterM) / 2,
    trunk_ratio: Number(tree.trunkRatio ?? DEFAULT_TRUNK_RATIO),
  };
}

// ---------------------------------------------------------------------------
// Family builders (the frozen gesture -> envelope contract)
// ---------------------------------------------------------------------------

function treeIdOf(tree) {
  return tree?.tree_id !== undefined && tree?.tree_id !== null ? String(tree.tree_id) : null;
}

/**
 * vegetation_geometry: add/replace/move/delete of one tree object.
 *
 * The payload is the tree's field mapping, spread verbatim — flat payload
 * IS the field mapping for the reducer's object fold. Pass the adapter
 * schema ({x_m, y_m, height_m, canopy_radius_m, trunk_ratio} — see
 * `worldTree`) for native realtime ops; the legacy API tree (u/v +
 * component fields) is accepted for wire-shape fidelity but its fields
 * would not survive the adapter's exact-lane validation.
 */
export function treeOperation(actorId, verb, tree, baseRevision) {
  return envelope(actorId, "vegetation_geometry", treeIdOf(tree), verb, { ...tree }, baseRevision);
}

/** landcover_surface: paint one class value across a half-open window. */
export function landcoverPaintOperation(actorId, window, classValue, baseRevision) {
  return envelope(
    actorId,
    "landcover_surface",
    null,
    "paint",
    { window: { ...window }, class: classValue },
    baseRevision,
  );
}

/** meteorological_forcing: set one variable at one time index. */
export function meteorologySetOperation(actorId, variable, timeIndex, value, baseRevision) {
  return envelope(actorId, "meteorological_forcing", String(variable), "set", {
    values: { [String(variable)]: value },
    time_index: timeIndex,
  }, baseRevision);
}

/** model_receptor_parameters: set one time-invariant receptor parameter. */
export function parameterSetOperation(actorId, name, value, baseRevision) {
  return envelope(actorId, "model_receptor_parameters", String(name), "set", {
    values: { [String(name)]: value },
  }, baseRevision);
}

/** output_view: select one output for the workspace's shared view. */
export function viewSelectOperation(actorId, outputName, baseRevision) {
  return envelope(actorId, "output_view", String(outputName), "select", {
    values: { output: String(outputName) },
  }, baseRevision);
}

/** selected_date_time: set the shared selected date/time (no entity). */
export function dateOperation(actorId, dateStr, baseRevision) {
  return envelope(actorId, "selected_date_time", null, "set", {
    values: { date: dateStr },
  }, baseRevision);
}

/**
 * building_geometry: add/replace/move/delete of one building object
 * (`values` = the state fields; identity rides as entity_id).
 */
export function buildingOperation(actorId, verb, buildingId, values, baseRevision) {
  const fields = { ...(values ?? {}) };
  delete fields.building_id;
  return envelope(actorId, "building_geometry", String(buildingId), verb, {
    values: fields,
  }, baseRevision);
}

// ---------------------------------------------------------------------------
// Universal family items -> envelopes (reuses composeFamilyEdit output)
// ---------------------------------------------------------------------------

/**
 * Convert universal edit items (`composeFamilyEdit` output:
 * `{adapter, operation, values, target, time_index, old_values}`) into
 * realtime wire envelopes.
 *
 * Adapter id -> source family, legacy operation -> verb via
 * `LEGACY_VERB_MAP` (the server's own funnel map), string targets ->
 * entity_id, paint window targets ride as the payload window. Unmappable
 * operations refuse with a typed error (never silently mis-verb'd).
 */
export function operationsFromFamilyItems(items, actorId) {
  if (!Array.isArray(items) || items.length === 0) {
    throw new OperationShapeError({
      code: "invalid_request",
      message: "operationsFromFamilyItems requires a non-empty items array",
      field: "items",
    });
  }
  return items.map((item) => {
    const family = item.adapter;
    if (!SOURCE_FAMILIES.has(family)) {
      throw new OperationShapeError({
        code: "unknown_source_family",
        message:
          `adapter ${String(family)} is not a realtime source family ` +
          `(${[...SOURCE_FAMILIES].join(", ")})`,
        field: "adapter",
      });
    }
    const verb = LEGACY_VERB_MAP[item.operation];
    if (!verb) {
      throw new OperationShapeError({
        code: "unknown_operation_verb",
        message:
          `operation ${String(item.operation)} has no realtime verb mapping ` +
          `(refusing rather than mis-verb'ing)`,
        field: "operation",
      });
    }
    const values = { ...(item.values ?? {}) };
    // Identity keys never travel inside values (the entity id is the key).
    delete values.tree_id;
    delete values.building_id;
    const target = item.target;
    const entityId = typeof target === "string" && target ? target : null;
    const payload = { values };
    if (item.time_index !== null && item.time_index !== undefined) {
      payload.time_index = Number(item.time_index);
    }
    if (target !== null && typeof target === "object" && !Array.isArray(target)) {
      if ("row_start" in target) payload.window = { ...target };
      else payload.target = { ...target };
    }
    return envelope(actorId, family, entityId, verb, payload, null);
  });
}

// ---------------------------------------------------------------------------
// Ledger / presence helpers (pure; shared by the app shell and tests)
// ---------------------------------------------------------------------------

/**
 * Presence roster from observed operations: unique actor ids EXCLUDING the
 * local actor, insertion-ordered by first observation.
 */
export function presenceFromOperations(actorIds, localActorId) {
  const seen = new Set();
  const roster = [];
  for (const actorId of actorIds ?? []) {
    if (!actorId || actorId === localActorId || seen.has(actorId)) continue;
    seen.add(actorId);
    roster.push(actorId);
  }
  return roster;
}

/**
 * Short human label for a ledger row: "move · vegetation · Broad canopy 03".
 * `entityLabel` (optional friendly label from the local tree store) wins over
 * the raw entity_id when one exists.
 */
export function operationSummary(wireEnvelope, entityLabel = null) {
  const value = wireEnvelope ?? {};
  if (!SOURCE_FAMILIES.has(value.source_family) || !OPERATION_VERBS.has(value.verb)) {
    return "operation";
  }
  const family = String(value.source_family).replace(/_geometry$/, "").replace(/_/g, " ");
  const entity = entityLabel || (value.entity_id ? String(value.entity_id) : "");
  return `${value.verb} · ${family}${entity ? ` · ${entity}` : ""}`;
}
