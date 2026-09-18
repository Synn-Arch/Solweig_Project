// SPDX-License-Identifier: GPL-3.0-only
//
// Capability-document model for the universal editing frontend (U-D2).
//
// The frontend "retrieves capability metadata from the API. It should not
// hard-code the complete tool set" (docs/incremental_design_tool/
// universal_editing/frontend_spec.md). This module is the single reader of
// that document: every editor list, property bound, class vocabulary,
// variable list, view layer, and dependency-stage classification below is
// DERIVED from the document — no physics vocabulary lives here.
//
// Pure and DOM-free: `app.mjs` renders what these functions produce, and the
// node test suite drives them with fixture documents whose shape is copied
// from `solweig_gpu/server/routes_capabilities.py` reality
// (`assets/capabilities_snapshot.json` is a verbatim server snapshot).

export const SUPPORTED_CAPABILITY_SCHEMA_VERSION = 1;

export class CapabilityError extends Error {
  constructor(message, { code = "invalid_capability_document" } = {}) {
    super(message);
    this.name = "CapabilityError";
    this.code = code;
  }
}

/** Human label for an adapter id ("meteorological_forcing" → "Meteorological forcing"). */
export function labelFromId(id) {
  const text = String(id).replace(/[_-]+/g, " ").trim();
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : String(id);
}

function labelFromProperty(name) {
  return labelFromId(name);
}

function requireArray(value, where) {
  if (!Array.isArray(value)) {
    throw new CapabilityError(`${where} must be an array, got ${typeof value}`);
  }
  return value.map((entry) => String(entry));
}

function finiteOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

// ---------------------------------------------------------------------------
// Document parsing
// ---------------------------------------------------------------------------

function parseAdapter(entry, viewOperations) {
  if (!entry || typeof entry !== "object") {
    throw new CapabilityError("adapters[] entries must be objects");
  }
  if (typeof entry.id !== "string" || entry.id.length === 0) {
    throw new CapabilityError("adapter entry is missing a string id");
  }
  const operations = requireArray(entry.operations, `adapter ${entry.id} operations`);
  const adapter = {
    id: entry.id,
    group: String(entry.group ?? ""),
    status: String(entry.status ?? ""),
    sourceNodes: requireArray(entry.source_nodes, `adapter ${entry.id} source_nodes`),
    operations,
    nominalSpatialScope: String(entry.nominal_spatial_scope ?? ""),
    localityModes: requireArray(entry.locality_modes, `adapter ${entry.id} locality_modes`),
    temporalScope: String(entry.temporal_scope ?? ""),
    preview: String(entry.preview ?? ""),
    validationFixtures: requireArray(
      entry.validation_fixtures,
      `adapter ${entry.id} validation_fixtures`,
    ),
    integrated: entry.integrated === true,
    dirtyNodes: requireArray(entry.dirty_nodes, `adapter ${entry.id} dirty_nodes`),
    propertySchema: entry.property_schema ?? null,
    notes: typeof entry.notes === "string" ? entry.notes : null,
    producibleIncremental: Array.isArray(entry.producible_incremental)
      ? entry.producible_incremental.map(String)
      : [],
  };
  // An adapter is an EDITOR when the engine executes it end-to-end AND it
  // does something beyond selecting among published results. Both facts come
  // from the document (integrated flag + the view_only operations vocabulary).
  adapter.editable =
    adapter.integrated && adapter.operations.some((op) => !viewOperations.includes(op));
  return adapter;
}

/**
 * Parse and validate a capability document.
 *
 * @param {object} document body served by GET /api/v1/capabilities
 * @returns {object} normalized capabilities model (see fields below)
 */
export function parseCapabilityDocument(document) {
  if (!document || typeof document !== "object") {
    throw new CapabilityError("capability document must be an object");
  }
  const schemaVersion = Number(document.schema_version);
  if (!Number.isInteger(schemaVersion)) {
    throw new CapabilityError("capability document is missing an integer schema_version");
  }
  if (schemaVersion > SUPPORTED_CAPABILITY_SCHEMA_VERSION) {
    throw new CapabilityError(
      `capability document schema_version ${schemaVersion} is newer than this frontend ` +
        `supports (${SUPPORTED_CAPABILITY_SCHEMA_VERSION}); upgrade the studio`,
      { code: "capability_schema_unsupported" },
    );
  }
  if (schemaVersion < 1) {
    // Non-positive versions are malformed, not "old": versions count from 1.
    throw new CapabilityError(
      `capability document schema_version ${schemaVersion} is not a valid version ` +
        `(supported: 1…${SUPPORTED_CAPABILITY_SCHEMA_VERSION})`,
      { code: "capability_schema_unsupported" },
    );
  }
  if (!Array.isArray(document.adapters) || document.adapters.length === 0) {
    throw new CapabilityError("capability document has no adapters[]");
  }
  const viewOnly = document.view_only ?? {};
  const viewOperations = requireArray(viewOnly.operations, "view_only.operations");
  const adapters = document.adapters.map((entry) => parseAdapter(entry, viewOperations));
  const adapterById = new Map(adapters.map((adapter) => [adapter.id, adapter]));

  const familiesRaw = document.edit_families ?? {};
  if (typeof familiesRaw !== "object" || Array.isArray(familiesRaw)) {
    throw new CapabilityError("edit_families must be an object keyed by source node");
  }
  const editFamilies = new Map();
  for (const [sourceNode, family] of Object.entries(familiesRaw)) {
    editFamilies.set(sourceNode, {
      sourceNode,
      dirtyNodes: requireArray(family?.dirty_nodes, `edit_families.${sourceNode}.dirty_nodes`),
      adapters: requireArray(family?.adapters, `edit_families.${sourceNode}.adapters`),
    });
  }

  const viewAdapterIds = requireArray(viewOnly.adapter_ids, "view_only.adapter_ids");

  // The full dependency-node vocabulary: every family source plus every node
  // any family can dirty. This is what "reused" is measured against.
  const nodeSet = new Set(editFamilies.keys());
  for (const family of editFamilies.values()) {
    for (const node of family.dirtyNodes) nodeSet.add(node);
  }

  return {
    schemaVersion,
    adapters,
    adapterById,
    editFamilies,
    /**
     * The server's shared-workspace pointer (deployment state, not engine
     * vocabulary): the one workspace every visitor joins by default. Null
     * on older servers, which keeps the mint-a-private-scenario fallback.
     */
    defaultWorkspaceId:
      typeof document.default_workspace_id === "string" && document.default_workspace_id
        ? document.default_workspace_id
        : null,
    fences: document.fences ?? {},
    localityModes: requireArray(document.locality_modes, "locality_modes"),
    viewOnly: {
      adapterIds: viewAdapterIds,
      operations: [...viewOperations].sort(),
      producibleLayers: requireArray(viewOnly.producible_layers, "view_only.producible_layers"),
      zeroScientificJobs: viewOnly.zero_scientific_jobs === true,
      basis: typeof viewOnly.basis === "string" ? viewOnly.basis : "",
    },
    nodes: [...nodeSet].sort(),
    /** Editors: integrated adapters with operations beyond view (document order). */
    editableAdapters: adapters.filter((adapter) => adapter.editable),
    /** The view_only adapter entries (they are never editors). */
    viewAdapters: adapters.filter((adapter) => viewAdapterIds.includes(adapter.id)),
    /**
     * Non-integrated, non-view adapters: rendered disabled with their status
     * and disclosure text (UEDIT-010 surfaces the dynamic-wind parity note).
     */
    disclosedAdapters: adapters.filter(
      (adapter) => !adapter.integrated && !viewAdapterIds.includes(adapter.id),
    ),
  };
}

// ---------------------------------------------------------------------------
// Presentation (labels only — no data)
// ---------------------------------------------------------------------------

/** Display strings for the registry's status vocabulary. Unknown → raw status. */
const STATUS_LABELS = {
  adapter_required: "Adapter required",
  full_only_initially: "Full-tile recompute only (initially)",
  scientific_extension: "Scientific extension — fenced",
  view_only: "View only — zero solver jobs",
};

/**
 * How one adapter is presented. `enabled` follows the document's integrated
 * flag; labels are chrome, every fact (scope, operations, disclosure) is the
 * document's own text.
 */
export function adapterPresentation(adapter) {
  const label = STATUS_LABELS[adapter.status] ?? adapter.status;
  return {
    id: adapter.id,
    title: labelFromId(adapter.id),
    group: adapter.group,
    enabled: adapter.editable === true,
    statusKey: adapter.status,
    statusLabel: adapter.editable ? `${label} · integrated` : label,
    statusTone: adapter.editable
      ? "ok"
      : adapter.status === "scientific_extension"
        ? "fenced"
        : "pending",
    disclosure: adapter.notes,
    preview: adapter.preview,
    metaLine:
      `${adapter.operations.join(", ")} · ` +
      (adapter.localityModes.length
        ? `${adapter.localityModes.join("/")} locality`
        : "no spatial scope") +
      ` · ${adapter.temporalScope || "n/a"}`,
  };
}

// ---------------------------------------------------------------------------
// Property controls (sliders / pickers generated from property_schema)
// ---------------------------------------------------------------------------

function numericControl(name, spec) {
  const min = finiteOrNull(spec.minimum);
  const max = finiteOrNull(spec.maximum);
  const isInteger = spec.type === "int";
  const bothBounds = min !== null && max !== null && max > min;
  return {
    kind: bothBounds ? "range" : "number",
    name,
    label: labelFromProperty(name),
    units: typeof spec.units === "string" ? spec.units : null,
    min,
    max,
    exclusiveMinimum: spec.exclusive_minimum === true,
    exclusiveMaximum: spec.exclusive_maximum === true,
    default: spec.default !== undefined ? spec.default : null,
    // Range inputs need a step; "any" keeps float schemas honest, ints step 1.
    step: isInteger ? "1" : "any",
    integer: isInteger,
    boundsText: typeof spec.bounds === "string" ? spec.bounds : null,
    basis: typeof spec.bounds_basis === "string" ? spec.bounds_basis : null,
    uncertain: typeof spec.uncertain === "string" ? spec.uncertain : null,
    description: typeof spec.description === "string" ? spec.description : null,
  };
}

function controlFor(name, spec) {
  if (!spec || typeof spec !== "object") {
    throw new CapabilityError(`property ${name} has no spec object`);
  }
  switch (spec.type) {
    case "number":
    case "float":
    case "int":
      return numericControl(name, spec);
    case "bool":
      return {
        kind: "toggle",
        name,
        label: labelFromProperty(name),
        default: spec.default !== undefined ? Boolean(spec.default) : false,
        basis: typeof spec.bounds_basis === "string" ? spec.bounds_basis : null,
        uncertain: typeof spec.uncertain === "string" ? spec.uncertain : null,
      };
    case "array":
      return {
        kind: "polygon",
        name,
        label: labelFromProperty(name),
        minItems: Number(spec.min_items ?? spec.items?.minItems ?? 0) || 0,
        units: typeof spec.units === "string" ? spec.units : null,
        description: typeof spec.description === "string" ? spec.description : null,
      };
    default:
      throw new CapabilityError(
        `property ${name} declares unsupported type ${JSON.stringify(spec.type)} ` +
          `(capability schema ${SUPPORTED_CAPABILITY_SCHEMA_VERSION})`,
      );
  }
}

/**
 * Generate the property-editor descriptors for one adapter, purely from its
 * property_schema (plus the document fences as a fallback source for class
 * vocabularies when an adapter carries schema-less fence data).
 *
 * @returns {{identityProperty: string|null, controls: Array, rejected: Array,
 *            blocked: string[], classPicker: object|null, timeSupport: string|null,
 *            fenceReason: string|null, classSemantics: string|null}}
 */
export function propertyControls(adapter, capabilities = null) {
  const schema = adapter?.propertySchema ?? null;
  const empty = {
    identityProperty: null,
    controls: [],
    rejected: [],
    blocked: [],
    classPicker: null,
    timeSupport: null,
    fenceReason: null,
    classSemantics: null,
  };
  if (!schema || typeof schema !== "object") return empty;

  const result = { ...empty };

  if (schema.identity_property !== undefined) {
    result.identityProperty = schema.identity_property;
  }
  if (typeof schema.time_support === "string") result.timeSupport = schema.time_support;

  // Land-cover style schema: a class vocabulary instead of named properties.
  if (Array.isArray(schema.valid_classes) || Array.isArray(schema.fenced_classes)) {
    result.classPicker = classPickerFrom(schema, capabilities, adapter);
    if (typeof schema.fence_reason === "string") result.fenceReason = schema.fence_reason;
    if (typeof schema.class_semantics === "string") result.classSemantics = schema.class_semantics;
    return result;
  }

  if (schema.properties && typeof schema.properties === "object") {
    result.controls = Object.entries(schema.properties).map(([name, spec]) =>
      controlFor(name, spec),
    );
  }
  if (Array.isArray(schema.rejected_properties)) {
    const reason =
      typeof schema.rejection_reason === "string"
        ? schema.rejection_reason
        : "rejected by the adapter's property schema";
    result.rejected = schema.rejected_properties.map((name) => ({
      name: String(name),
      label: labelFromProperty(name),
      reason,
    }));
  }
  if (Array.isArray(schema.blocked_variables)) {
    result.blocked = schema.blocked_variables.map(String);
  }
  if (Array.isArray(schema.blocked_parameters)) {
    result.blocked = schema.blocked_parameters.map(String);
  }
  return result;
}

/** Paintable classes = valid − fenced, from the schema first, fences second. */
function classPickerFrom(schema, capabilities, adapter) {
  const valid = (schema.valid_classes ?? capabilities?.fences?.landcover_valid_classes ?? []).map(
    Number,
  );
  const fenced = (
    schema.fenced_classes ?? capabilities?.fences?.landcover_fenced_classes ?? []
  ).map(Number);
  const fencedSet = new Set(fenced);
  return {
    kind: "classes",
    options: [...new Set(valid)].filter((code) => !fencedSet.has(code)).sort((a, b) => a - b),
    fenced: [...new Set(fenced)].sort((a, b) => a - b),
    sourceAdapter: adapter?.id ?? null,
  };
}

/**
 * Schema default of one property on one adapter's property_schema.
 *
 * Wiring lookup for contract passthroughs the EDITED adapter's own schema
 * rejects (e.g. the vegetation schema fences `transmissivity` and its
 * rejection reason points at `model_receptor_parameters.transVeg`): the
 * default is read from the document, never restated as a local literal.
 *
 * @returns {*} the schema's declared default, or null when the document
 *          declares neither the adapter, the property, nor a default.
 */
export function adapterPropertyDefault(capabilities, adapterId, propertyName) {
  const spec =
    capabilities?.adapterById?.get(adapterId)?.propertySchema?.properties?.[propertyName];
  return spec && spec.default !== undefined ? spec.default : null;
}

/**
 * Validate a draft {property → value} against generated controls.
 *
 * Unknown draft properties are REJECTED, never passed through: the universal
 * edit payload must carry exactly the schema's vocabulary. For class-picker
 * families pass `{ classPicker }` so the draft's paint class is validated
 * against the document's valid-minus-fenced vocabulary too — the UI chips
 * constrain the pointer, but the payload itself must be impossible to poison
 * with a fenced or off-vocabulary class.
 *
 * @param {Array} controls generated controls (property_schema derived)
 * @param {object} draft raw {property → value} from the editor
 * @param {object} [options]
 * @param {object|null} [options.classPicker=null] from propertyControls()
 * @returns {{values: object, errors: Array<{property: string, message: string}>}}
 *          `values` holds ONLY schema-known properties with defaults filled.
 */
export function validateDraft(controls, draft = {}, { classPicker = null } = {}) {
  const values = {};
  const errors = [];
  const known = new Set(controls.map((control) => control.name));
  if (classPicker) known.add("class");
  for (const [name, raw] of Object.entries(draft ?? {})) {
    if (!known.has(name)) {
      errors.push({
        property: name,
        message: "is not a property this adapter's schema declares",
      });
      continue;
    }
    values[name] = raw;
  }
  for (const control of controls) {
    const raw = values[control.name];
    if (raw === undefined || raw === null || raw === "") {
      if (control.default !== null && control.default !== undefined) {
        values[control.name] = control.default;
        continue;
      }
      errors.push({ property: control.name, message: "a value is required" });
      continue;
    }
    if (control.kind === "toggle") {
      if (typeof raw !== "boolean") {
        errors.push({ property: control.name, message: "must be true or false" });
      }
      continue;
    }
    if (control.kind === "range" || control.kind === "number") {
      const value = Number(raw);
      if (!Number.isFinite(value)) {
        errors.push({ property: control.name, message: `${JSON.stringify(raw)} is not a number` });
        continue;
      }
      if (control.integer && !Number.isInteger(value)) {
        errors.push({ property: control.name, message: `${value} must be a whole number` });
        continue;
      }
      if (control.min !== null) {
        const violates =
          control.exclusiveMinimum ? value <= control.min : value < control.min;
        if (violates) {
          errors.push({
            property: control.name,
            message: `${value} is below the ${control.exclusiveMinimum ? "exclusive " : ""}minimum ${control.min}`,
          });
          continue;
        }
      }
      if (control.max !== null) {
        const violates =
          control.exclusiveMaximum ? value >= control.max : value > control.max;
        if (violates) {
          errors.push({
            property: control.name,
            message: `${value} is above the ${control.exclusiveMaximum ? "exclusive " : ""}maximum ${control.max}`,
          });
        }
      }
    }
  }
  if (classPicker) {
    const raw = values.class;
    if (raw !== undefined && raw !== null && raw !== "") {
      const code = Number(raw);
      if (classPicker.fenced.includes(code)) {
        errors.push({
          property: "class",
          message: `class ${raw} is fenced by the document (fenced: ${classPicker.fenced.join(", ")})`,
        });
      } else if (!classPicker.options.includes(code)) {
        errors.push({
          property: "class",
          message: `class ${raw} is not in the document's vocabulary (${classPicker.options.join(", ")})`,
        });
      } else {
        values.class = code;
      }
    }
  }
  return { values, errors };
}

// ---------------------------------------------------------------------------
// UEDIT-009 — changed / reused / recomputed stage classification
// ---------------------------------------------------------------------------

/**
 * Classify the dependency-graph nodes for one committed edit.
 *
 * The document's `edit_families` carries each source node's dirty-node
 * closure; the classification is derived, not asserted:
 *
 *   changed    — the source node(s) the edit wrote (family inputs)
 *   recomputed — the rest of that family's dirty closure (downstream stages)
 *   reused     — every other node in the document's vocabulary ( untouched )
 *
 * This derivation is the FALLBACK. Executor-path jobs serve the executed
 * impact plan (`impact_plan` in the job body, plan schema v1); when present,
 * `impactPanelModel` classifies from the server's per-node stages instead —
 * the document's declared closure can only approximate what the planner
 * actually routed (windowed vs full, reused vs recomputed).
 */
export function stageClassification(capabilities, sourceNodes) {
  const edited = [...new Set((sourceNodes ?? []).map(String))];
  if (edited.length === 0) {
    throw new CapabilityError("stageClassification requires at least one source node");
  }
  const unknown = edited.filter((node) => !capabilities.editFamilies.has(node));
  if (unknown.length > 0) {
    throw new CapabilityError(
      `unknown edit family ${unknown.join(", ")}; the document lists: ` +
        `${[...capabilities.editFamilies.keys()].sort().join(", ")}`,
    );
  }
  const dirtySet = new Set();
  for (const node of edited) {
    for (const dirty of capabilities.editFamilies.get(node).dirtyNodes) dirtySet.add(dirty);
  }
  const changedSet = new Set(edited);
  const recomputedSet = new Set([...dirtySet].filter((node) => !changedSet.has(node)));
  const reusedSet = new Set(capabilities.nodes.filter((node) => !dirtySet.has(node)));
  return {
    changed: [...changedSet].sort(),
    recomputed: [...recomputedSet].sort(),
    reused: [...reusedSet].sort(),
  };
}

/** Realized scope line from a completed job's metrics (mode + window share). */
export function scopeFromMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") return null;
  const mode = typeof metrics.mode === "string" ? metrics.mode : "";
  const fraction = finiteOrNull(metrics.window_fraction);
  let scopeText = null;
  if (mode === "no-op") {
    scopeText = "no recompute — the published result was re-served";
  } else if (fraction !== null) {
    scopeText = `${(fraction * 100).toFixed(1)}% of the tile${mode ? ` · mode ${mode}` : ""}`;
  } else if (mode) {
    scopeText = `mode ${mode}`;
  }
  return {
    scopeText,
    mode: mode || null,
    windowFraction: fraction,
    fallbackReason:
      typeof metrics.fallback_reason === "string" && metrics.fallback_reason
        ? metrics.fallback_reason
        : null,
  };
}

/**
 * Classify from the server-served executed impact plan (plan schema v1):
 * per-node `{node, stage: changed|reused|recomputed, why, spatial_scope}`.
 *
 * The plan is authoritative but defensive: a malformed plan returns null and
 * the caller falls back to the document-derived classification. Nodes the
 * plan does not list are reused (the plan reports what it routed, not the
 * whole vocabulary).
 *
 * @returns {object|null} {changed, recomputed, reused, perNode, routing,
 *                         estimates, realized, mode, jobId, sceneRevision}
 */
export function planClassification(capabilities, plan) {
  if (!plan || typeof plan !== "object" || !Array.isArray(plan.nodes)) return null;
  const changed = [];
  const recomputed = [];
  const reused = [];
  const perNode = [];
  for (const entry of plan.nodes) {
    if (!entry || typeof entry.node !== "string") return null;
    const stage =
      entry.stage === "changed" || entry.stage === "recomputed" ? entry.stage : "reused";
    if (stage === "changed") changed.push(entry.node);
    else if (stage === "recomputed") recomputed.push(entry.node);
    else reused.push(entry.node);
    perNode.push({
      node: entry.node,
      stage,
      why: typeof entry.why === "string" && entry.why ? entry.why : null,
      spatialScope:
        typeof entry.spatial_scope === "string" && entry.spatial_scope
          ? entry.spatial_scope
          : null,
    });
  }
  // The plan lists the nodes it routed; every other node in the document's
  // vocabulary was reused by construction.
  const listed = new Set([...changed, ...recomputed, ...reused]);
  for (const node of capabilities.nodes) {
    if (!listed.has(node)) reused.push(node);
  }
  const routing =
    plan.routing && typeof plan.routing === "object" ? { ...plan.routing } : null;
  return {
    changed: [...changed].sort(),
    recomputed: [...recomputed].sort(),
    reused: [...reused].sort(),
    perNode,
    routing,
    estimates: plan.estimates && typeof plan.estimates === "object" ? { ...plan.estimates } : null,
    realized: plan.realized && typeof plan.realized === "object" ? { ...plan.realized } : null,
    mode: typeof plan.mode === "string" ? plan.mode : null,
    jobId: typeof plan.job_id === "string" ? plan.job_id : null,
    sceneRevision: Number.isFinite(Number(plan.scene_revision))
      ? Number(plan.scene_revision)
      : null,
  };
}

/**
 * Full UEDIT-009 impact-panel model: node classification plus the realized
 * scope line.
 *
 * When the completed job served its executed `impact_plan`, the server's
 * per-node stages win (`serverPlan: true`, with per-node why/routing detail).
 * Otherwise the classification is derived from the document's declared
 * closures for the edited families.
 *
 * A job with neither a plan nor source nodes (time-only recompute, manual
 * rerun, retry) gets a distinct `recomputeOnly` model instead of throwing:
 * the panel must report the recompute rather than silently keeping the
 * previous edit's classification on screen through the newer job.
 */
export function impactPanelModel(capabilities, sourceNodes, metrics = null, impactPlan = null) {
  const scope = scopeFromMetrics(metrics);
  const plan = planClassification(capabilities, impactPlan);
  if (plan) {
    return { recomputeOnly: false, serverPlan: true, ...plan, scope };
  }
  if (!Array.isArray(sourceNodes) || sourceNodes.length === 0) {
    return {
      recomputeOnly: true,
      serverPlan: false,
      changed: [],
      recomputed: [],
      reused: [],
      scope,
    };
  }
  return {
    recomputeOnly: false,
    serverPlan: false,
    ...stageClassification(capabilities, sourceNodes),
    scope,
  };
}

// ---------------------------------------------------------------------------
// View-only panel models (UEDIT-007 / UEDIT-009 frontend side)
// ---------------------------------------------------------------------------

/** The layer/operation vocabulary for the view panel, straight from view_only. */
export function viewPanelModel(capabilities) {
  const { viewOnly } = capabilities;
  return {
    operations: [...viewOnly.operations],
    layers: [...viewOnly.producibleLayers],
    zeroScientificJobs: viewOnly.zeroScientificJobs,
    basis: viewOnly.basis,
  };
}

/**
 * Render model for one successful view response. The zero-job affordance
 * quotes the SERVER's flags — the UI never asserts "free" on its own.
 */
export function viewResultModel(response) {
  if (!response || typeof response !== "object") return null;
  const model = {
    operation: response.operation ?? null,
    sceneVersion: response.scene_version ?? null,
    layer: response.layer ?? null,
    compareLayer: response.compare_layer ?? null,
    timeIndex: response.time_index ?? null,
    cachedTimeIndices: Array.isArray(response.cached_time_indices)
      ? response.cached_time_indices.map(Number)
      : null,
    legend: response.legend ?? null,
    resultManifestUrl: response.result_manifest_url ?? null,
    resultPayloadUrl: response.result_payload_url ?? null,
    zeroJob:
      response.zero_scientific_jobs === true && response.job_enqueued === false
        ? "0 solver jobs — answered from the published result"
        : null,
  };
  return model;
}

/** Render model for a refused view request (typed envelope details kept). */
export function viewErrorModel(error) {
  return {
    code: error?.code ?? "view_error",
    message: error?.message ?? String(error),
    publishedLayers: Array.isArray(error?.details?.published_layers)
      ? error.details.published_layers.map(String)
      : null,
    cachedTimeIndices: Array.isArray(error?.details?.cached_time_indices)
      ? error.details.cached_time_indices.map(Number)
      : null,
  };
}
