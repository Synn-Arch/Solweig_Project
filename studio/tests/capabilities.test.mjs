// SPDX-License-Identifier: GPL-3.0-only
//
// Capability-document parsing and the UI models GENERATED from it (U-D2).
//
// The primary fixture is the checked-in snapshot of the real server document
// (assets/capabilities_snapshot.json); synthetic documents cover edge shapes.
// No live server is involved.

import test from "node:test";
import assert from "node:assert/strict";

import snapshot from "../assets/capabilities_snapshot.json" with { type: "json" };
import {
  CapabilityError,
  SUPPORTED_CAPABILITY_SCHEMA_VERSION,
  adapterPresentation,
  adapterPropertyDefault,
  impactPanelModel,
  parseCapabilityDocument,
  propertyControls,
  scopeFromMetrics,
  stageClassification,
  validateDraft,
  viewErrorModel,
  viewPanelModel,
  viewResultModel,
} from "../capabilities.mjs";

function parse(document = snapshot) {
  return parseCapabilityDocument(document);
}

// ---------------------------------------------------------------------------
// Document shape → editor lists (nothing is hardcoded in the frontend)

test("the shared-world pointer is carried through, null when the server omits it", () => {
  // Deployment state, not engine vocabulary: a server with a shared world
  // publishes default_workspace_id so visitors join it; the checked-in
  // snapshot (older server shape) must parse with a null pointer.
  const capabilities = parse();
  assert.equal(capabilities.defaultWorkspaceId, null);
  const shared = parseCapabilityDocument({
    ...snapshot,
    default_workspace_id: "scn_shared_world",
  });
  assert.equal(shared.defaultWorkspaceId, "scn_shared_world");
});

test("editable adapters are exactly the integrated ones with non-view operations", () => {
  const capabilities = parse();
  assert.deepEqual(
    capabilities.editableAdapters.map((adapter) => adapter.id).sort(),
    [
      "building_geometry",
      "landcover_surface",
      "meteorological_forcing",
      "model_receptor_parameters",
      "vegetation_geometry",
    ],
  );
  for (const adapter of capabilities.editableAdapters) {
    assert.equal(adapter.integrated, true, `${adapter.id} must be integrated`);
    assert.ok(
      adapter.operations.some((operation) => !capabilities.viewOnly.operations.includes(operation)),
      `${adapter.id} must operate beyond view selection`,
    );
  }
});

test("the view_only adapter is never an editor but is listed as a view adapter", () => {
  const capabilities = parse();
  assert.deepEqual(capabilities.viewAdapters.map((adapter) => adapter.id), ["output_view"]);
  assert.ok(!capabilities.editableAdapters.some((adapter) => adapter.id === "output_view"));
});

test("non-integrated adapters are disclosed with their document status", () => {
  const capabilities = parse();
  assert.deepEqual(
    capabilities.disclosedAdapters.map((adapter) => adapter.id),
    ["dynamic_wind_from_geometry", "selected_date_time", "terrain_dem"],
  );
  // UEDIT-010 parity note must survive verbatim — the frontend never
  // paraphrases a scientific disclosure.
  const wind = capabilities.disclosedAdapters.find(
    (adapter) => adapter.id === "dynamic_wind_from_geometry",
  );
  assert.match(wind.notes, /WindCoeff/i);
  assert.match(wind.notes, /coeff=1/);
});

test("parse rejects documents the frontend cannot honour", () => {
  assert.throws(() => parseCapabilityDocument(null), CapabilityError);
  assert.throws(() => parseCapabilityDocument({}), /schema_version/);
  assert.throws(
    () => parseCapabilityDocument({ schema_version: 2, adapters: snapshot.adapters }),
    /newer than this frontend supports/,
  );
  assert.throws(
    () => parseCapabilityDocument({ schema_version: 1, adapters: [] }),
    /no adapters/,
  );
  // Non-positive versions are malformed, not merely "older": versions count
  // from 1, so 0/negative must never pass as supported.
  assert.throws(
    () => parseCapabilityDocument({ schema_version: 0, adapters: snapshot.adapters }),
    CapabilityError,
  );
  assert.throws(
    () => parseCapabilityDocument({ schema_version: -3, adapters: snapshot.adapters }),
    /not a valid version/,
  );
  assert.throws(
    () =>
      parseCapabilityDocument({
        schema_version: 1,
        adapters: snapshot.adapters,
        view_only: snapshot.view_only,
        edit_families: [],
      }),
    /edit_families must be an object/,
  );
});

// ---------------------------------------------------------------------------
// Property controls generated from property_schema / fences

test("vegetation controls carry the document's bounds and units", () => {
  const capabilities = parse();
  const adapter = capabilities.adapterById.get("vegetation_geometry");
  const { controls, rejected, identityProperty } = propertyControls(adapter, capabilities);
  assert.equal(identityProperty, "tree_id");

  const height = controls.find((control) => control.name === "height_m");
  assert.equal(height.kind, "range");
  assert.equal(height.min, 3);
  assert.equal(height.max, 40);
  assert.equal(height.units, "m");

  const radius = controls.find((control) => control.name === "canopy_radius_m");
  assert.equal(radius.min, 0.5);
  assert.equal(radius.max, 15);

  const trunk = controls.find((control) => control.name === "trunk_ratio");
  assert.equal(trunk.exclusiveMaximum, true);
  assert.equal(trunk.default, 0.25);

  // Positional properties are real controls (map-driven in the UI, but the
  // schema still declares them — with prose bounds, not numbers).
  assert.ok(controls.some((control) => control.name === "x_m"));

  // The schema's rejection list, with the schema's own reason.
  assert.deepEqual(rejected.map((entry) => entry.name), ["transmissivity"]);
  assert.match(rejected[0].reason, /inert in the physics/);
});

test("landcover class picker is valid minus fenced, straight from the schema", () => {
  const capabilities = parse();
  const adapter = capabilities.adapterById.get("landcover_surface");
  const { classPicker, fenceReason } = propertyControls(adapter, capabilities);
  assert.deepEqual(classPicker.options, [1, 2, 5, 6]);
  assert.deepEqual(classPicker.fenced, [7]);
  assert.match(fenceReason, /fenced/);
});

test("landcover classes can also come from document fences when the schema is partial", () => {
  const capabilities = parse();
  const fencedOnly = {
    ...capabilities.adapterById.get("landcover_surface"),
    // Schema declares the fence but not the valid vocabulary: the document-
    // level fences supply it (single source of truth either way).
    propertySchema: { fenced_classes: [7], fence_reason: "water is fenced" },
  };
  const { classPicker } = propertyControls(fencedOnly, capabilities);
  assert.deepEqual(classPicker.options, [1, 2, 5, 6]);
  assert.deepEqual(classPicker.fenced, [7]);
});

test("meteorology exposes blocked variables and time support from the schema", () => {
  const capabilities = parse();
  const adapter = capabilities.adapterById.get("meteorological_forcing");
  const { blocked, timeSupport } = propertyControls(adapter, capabilities);
  assert.ok(blocked.includes("wind_direction"));
  assert.ok(blocked.includes("hour"));
  assert.match(timeSupport, /time/i);
});

test("draft validation enforces the generated bounds", () => {
  const capabilities = parse();
  const adapter = capabilities.adapterById.get("vegetation_geometry");
  const { controls } = propertyControls(adapter, capabilities);

  // In-bounds draft fills defaults for absent properties (position comes from
  // the map in the UI; the draft supplies it).
  const ok = validateDraft(controls, { x_m: 510, y_m: 320, height_m: 12, canopy_radius_m: 4 });
  assert.deepEqual(ok.errors, []);
  assert.equal(ok.values.trunk_ratio, 0.25); // default from the schema

  // Out-of-bounds and exclusive-maximum violations are rejected.
  const bad = validateDraft(controls, {
    x_m: 510,
    y_m: 320,
    height_m: 2,
    canopy_radius_m: 99,
    trunk_ratio: 1,
  });
  assert.equal(bad.errors.length, 3);
  assert.ok(bad.errors.some((error) => error.property === "height_m"));
  assert.ok(
    bad.errors.some((error) => error.property === "trunk_ratio" && /exclusive/.test(error.message)),
  );

  // Unknown draft properties are rejected and never forwarded: the universal
  // edit payload must carry exactly the schema's vocabulary.
  const poisoned = validateDraft(controls, {
    x_m: 510,
    y_m: 320,
    height_m: 12,
    canopy_radius_m: 4,
    trunk_ratio: 0.25,
    transmissivity: 0.5, // the schema explicitly REJECTS this property
  });
  assert.ok(
    poisoned.errors.some((error) => error.property === "transmissivity" && /schema declares/.test(error.message)),
  );
  assert.ok(!("transmissivity" in poisoned.values), "unknown keys must not survive into values");
});

test("adapterPropertyDefault reads declared schema defaults, missing sources yield null", () => {
  const capabilities = parse();
  // The single document-declared source for the vegetation passthrough.
  assert.equal(adapterPropertyDefault(capabilities, "model_receptor_parameters", "transVeg"), 0.03);
  assert.equal(adapterPropertyDefault(capabilities, "vegetation_geometry", "trunk_ratio"), 0.25);
  // Unknown adapter / property / document degrade to null — never a guess.
  assert.equal(adapterPropertyDefault(capabilities, "no_such_adapter", "transVeg"), null);
  assert.equal(
    adapterPropertyDefault(capabilities, "model_receptor_parameters", "no_such_property"),
    null,
  );
  assert.equal(adapterPropertyDefault(null, "model_receptor_parameters", "transVeg"), null);
});

// ---------------------------------------------------------------------------
// UEDIT-009 stage classification (derived from edit_families closures)

test("stage classification splits nodes into changed / recomputed / reused", () => {
  const capabilities = parse();
  const stages = stageClassification(capabilities, ["vegetation_dsm"]);
  assert.deepEqual(stages.changed, ["vegetation_dsm"]);
  // Downstream of the vegetation family closure.
  assert.ok(stages.recomputed.includes("vegetation_visibility"));
  assert.ok(stages.recomputed.includes("utci"));
  // Untouched families' nodes are reused as-is.
  assert.ok(stages.reused.includes("building_dsm"));
  assert.ok(stages.reused.includes("meteorology"));
  // Partition property: every node lands in exactly one bucket.
  const all = [...stages.changed, ...stages.recomputed, ...stages.reused].sort();
  assert.deepEqual(all, [...new Set(all)].sort());
  assert.equal(all.length, capabilities.nodes.length);
});

test("multi-family edits union their closures", () => {
  const capabilities = parse();
  const stages = stageClassification(capabilities, ["vegetation_dsm", "landcover"]);
  assert.deepEqual(stages.changed, ["landcover", "vegetation_dsm"]);
  assert.ok(!stages.recomputed.includes("landcover"));
  assert.ok(stages.recomputed.includes("surface_thermal_state"));
  assert.ok(!stages.reused.includes("vegetation_visibility"));
});

test("unknown families are an error, never a silent guess", () => {
  const capabilities = parse();
  assert.throws(() => stageClassification(capabilities, ["not_a_node"]), CapabilityError);
  assert.throws(() => stageClassification(capabilities, []), CapabilityError);
});

test("scope line quotes the job metrics, including the no-op case", () => {
  assert.equal(scopeFromMetrics(null), null);
  assert.equal(
    scopeFromMetrics({ mode: "no-op", window_fraction: 0 }).scopeText,
    "no recompute — the published result was re-served",
  );
  const windowed = scopeFromMetrics({ mode: "windowed", window_fraction: 0.25 });
  assert.equal(windowed.scopeText, "25.0% of the tile · mode windowed");
  assert.equal(windowed.fallbackReason, null);
  const fallback = scopeFromMetrics({
    mode: "full",
    window_fraction: 1,
    fallback_reason: "window below threshold",
  });
  assert.equal(fallback.fallbackReason, "window below threshold");
  assert.equal(scopeFromMetrics({ mode: "full" }).scopeText, "mode full");
});

test("impact panel model combines classification and scope", () => {
  const capabilities = parse();
  const model = impactPanelModel(capabilities, ["meteorology"], {
    mode: "windowed",
    window_fraction: 0.125,
  });
  assert.equal(model.recomputeOnly, false);
  assert.deepEqual(model.changed, ["meteorology"]);
  assert.equal(model.scope.scopeText, "12.5% of the tile · mode windowed");
});

test("jobs without source nodes report recompute-only, never a stale classification", () => {
  const capabilities = parse();
  // A time-only recompute / manual rerun / retry carries no source nodes: the
  // model must say so instead of letting an older edit's classification ride
  // through the newer job.
  for (const sourceNodes of [null, undefined, []]) {
    const model = impactPanelModel(capabilities, sourceNodes, {
      mode: "windowed",
      window_fraction: 0.5,
    });
    assert.equal(model.recomputeOnly, true, `sourceNodes=${JSON.stringify(sourceNodes)}`);
    assert.deepEqual(model.changed, []);
    assert.deepEqual(model.recomputed, []);
    assert.deepEqual(model.reused, []);
    assert.equal(model.scope.scopeText, "50.0% of the tile · mode windowed");
  }
  // The classification itself still throws on garbage — unknown families stay
  // loud, only the no-source case is a legitimate recompute.
  assert.throws(() => impactPanelModel(capabilities, ["not_a_node"]), CapabilityError);
});

// ---------------------------------------------------------------------------
// Server-served executed plans (u-d4): executor-path jobs publish the
// per-node stages they actually routed; the plan wins over the derived
// closure, and malformed plans fall back to the derivation.
// ---------------------------------------------------------------------------

test("an executed impact plan classifies nodes the server actually routed", () => {
  const capabilities = parse();
  const plan = {
    schema_version: 1,
    status: "executed",
    mode: "local",
    scene_revision: 4,
    job_id: "job_7",
    nodes: [
      { node: "landcover", stage: "changed", why: "paint re-labeled 64 cells", spatial_scope: "window" },
      { node: "radiation", stage: "recomputed", why: "downstream of landcover" },
      { node: "tmrt", stage: "reused", why: "patch cache hit" },
    ],
    routing: { mode: "windowed", write_windows: 1 },
    estimates: { work_units: 9 },
    realized: { patch_count: 1 },
  };
  const model = impactPanelModel(capabilities, ["landcover"], null, plan);
  assert.equal(model.serverPlan, true);
  assert.deepEqual(model.changed, ["landcover"]);
  assert.deepEqual(model.recomputed, ["radiation"]);
  assert.ok(model.reused.includes("tmrt"));
  // Nodes the plan never listed are reused by construction.
  assert.ok(model.reused.includes("meteorology"));
  assert.ok(model.reused.includes("vegetation_dsm"));
  // Per-node detail + routing ride along for the panel's why-lines.
  const landcover = model.perNode.find((entry) => entry.node === "landcover");
  assert.equal(landcover.why, "paint re-labeled 64 cells");
  assert.equal(landcover.spatialScope, "window");
  assert.equal(model.routing.mode, "windowed");
  assert.equal(model.realized.patch_count, 1);
  assert.equal(model.jobId, "job_7");
  assert.equal(model.sceneRevision, 4);
});

test("malformed plans fall back to the document-derived classification", () => {
  const capabilities = parse();
  for (const bad of [
    null,
    {},
    { nodes: "nope" },
    { nodes: [{ stage: "changed" }] }, // missing node name
    { nodes: [42] },
  ]) {
    const model = impactPanelModel(capabilities, ["meteorology"], null, bad);
    assert.equal(model.serverPlan, false, `plan=${JSON.stringify(bad)}`);
    assert.deepEqual(model.changed, ["meteorology"]);
  }
  // Unknown plan stage vocabulary is contained, never propagated raw: the
  // node reads as reused (never an unclassified chip), and nothing else
  // changes.
  const model = impactPanelModel(capabilities, ["meteorology"], null, {
    nodes: [{ node: "meteorology", stage: "teleported" }],
  });
  assert.equal(model.serverPlan, true);
  assert.deepEqual(model.changed, []);
  assert.ok(model.reused.includes("meteorology"));
});

test("a plan wins even when the commit carried no source nodes", () => {
  // A no-source-node job that still served a plan (e.g. a family replay)
  // classifies from the plan rather than reporting recompute-only.
  const capabilities = parse();
  const model = impactPanelModel(capabilities, [], null, {
    nodes: [{ node: "model_parameters", stage: "changed", why: "reset" }],
  });
  assert.equal(model.recomputeOnly, false);
  assert.deepEqual(model.changed, ["model_parameters"]);
});

// ---------------------------------------------------------------------------
// View-only panel models (UEDIT-007)

test("view panel vocabulary comes from view_only, operations sorted", () => {
  const model = viewPanelModel(parse());
  assert.deepEqual(model.operations, ["cached_time", "compare", "legend", "select_layer"]);
  assert.deepEqual(model.layers, [
    "kdown",
    "kup",
    "ldown",
    "lup",
    "shadow",
    "ta",
    "tmrt",
    "utci",
    "wind",
  ]);
  assert.equal(model.zeroScientificJobs, true);
  assert.match(model.basis, /published/i);
});

test("view result model asserts zero jobs only when the server says so", () => {
  const free = viewResultModel({
    operation: "legend",
    scene_version: 4,
    job_enqueued: false,
    zero_scientific_jobs: true,
    legend: { min: 18.2, max: 41.5 },
  });
  assert.match(free.zeroJob, /0 solver jobs/);
  const cautious = viewResultModel({ operation: "legend", job_enqueued: true });
  assert.equal(cautious.zeroJob, null);
});

test("view error model keeps the refusal's typed details", () => {
  // ApiClient flattens the server's error envelope, so details arrive as
  // error.details.{published_layers, cached_time_indices}.
  const model = viewErrorModel({
    code: "view_not_available",
    message: "layer not published",
    details: { published_layers: ["utci", "tmrt"] },
  });
  assert.deepEqual(model.publishedLayers, ["utci", "tmrt"]);
  const cached = viewErrorModel({
    code: "view_not_available",
    message: "time step not cached",
    details: { cached_time_indices: [6, 12, 18] },
  });
  assert.deepEqual(cached.cachedTimeIndices, [6, 12, 18]);
});

// ---------------------------------------------------------------------------
// Presentation

test("adapter presentation maps known statuses and passes unknown ones through", () => {
  const capabilities = parse();
  const wind = adapterPresentation(
    capabilities.adapterById.get("dynamic_wind_from_geometry"),
  );
  assert.equal(wind.statusTone, "fenced");
  assert.equal(wind.enabled, false);

  const unknown = adapterPresentation({
    id: "future_adapter",
    group: "",
    status: "some_new_status",
    sourceNodes: [],
    operations: [],
    nominalSpatialScope: "",
    localityModes: [],
    temporalScope: "",
    preview: "",
    validationFixtures: [],
    integrated: false,
    dirtyNodes: [],
    propertySchema: null,
    notes: null,
    producibleIncremental: [],
    editable: false,
  });
  // Unknown vocabulary is shown raw, never guessed at.
  assert.equal(unknown.statusLabel, "some_new_status");
});

test("snapshot declares the supported schema version", () => {
  assert.equal(snapshot.schema_version, SUPPORTED_CAPABILITY_SCHEMA_VERSION);
});
