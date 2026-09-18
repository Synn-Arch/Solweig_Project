// SPDX-License-Identifier: GPL-3.0-only
//
// Family edit composition (U-D2 transports, u-d4 universal wiring): drafts
// validated against the capability document, tree transport conversion, the
// universal items the four integrated families now send, and the
// transport-disclosure error for adapters this build cannot carry yet.

import test from "node:test";
import assert from "node:assert/strict";

import snapshot from "../assets/capabilities_snapshot.json" with { type: "json" };
import { parseCapabilityDocument } from "../capabilities.mjs";
import {
  DraftValidationError,
  EDIT_TRANSPORTS,
  FamilyTransportError,
  composeFamilyEdit,
  editTransport,
} from "../family_edits.mjs";

const capabilities = parseCapabilityDocument(snapshot);

function vegetationDraft(overrides = {}) {
  return {
    operation: "update",
    treeId: "tree-7",
    u: 0.51,
    v: 0.32,
    properties: { x_m: 510, y_m: 320, height_m: 14, canopy_radius_m: 4.5, trunk_ratio: 0.3 },
    ...overrides,
  };
}

test("transport table wires the tree contract and the four universal families", () => {
  assert.deepEqual(EDIT_TRANSPORTS, {
    vegetation_geometry: "tree-edits-v1",
    meteorological_forcing: "universal-edits-v1",
    landcover_surface: "universal-edits-v1",
    building_geometry: "universal-edits-v1",
    model_receptor_parameters: "universal-edits-v1",
  });
  // Every integrated editor has a transport; editTransport is the lookup the
  // app uses to pick the commit endpoint.
  for (const adapter of capabilities.editableAdapters) {
    assert.ok(editTransport(adapter.id), `${adapter.id} needs a transport entry`);
  }
  assert.equal(editTransport("terrain_dem"), null);
});

test("vegetation drafts convert schema radius to contract diameter", () => {
  const [item] = composeFamilyEdit(capabilities, "vegetation_geometry", vegetationDraft());
  assert.equal(item.operation, "update");
  assert.equal(item.tree.tree_id, "tree-7");
  assert.equal(item.tree.height_m, 14);
  assert.equal(item.tree.canopy_diameter_m, 9); // radius 4.5 × 2
  assert.equal(item.tree.trunk_ratio, 0.3);
  // transmissivity rides the contract as a fixed passthrough (the schema
  // rejects it as an editable property); the generated editor never offers it.
  assert.equal(item.tree.transmissivity, 0.03);
});

test("vegetation defaults come from the schema, not from a local list", () => {
  const properties = { ...vegetationDraft().properties, trunk_ratio: undefined };
  const [item] = composeFamilyEdit(capabilities, "vegetation_geometry", {
    ...vegetationDraft(),
    properties,
  });
  assert.equal(item.tree.trunk_ratio, 0.25); // schema default via validateDraft
});

test("vegetation delete needs only the id", () => {
  const [item] = composeFamilyEdit(capabilities, "vegetation_geometry", {
    operation: "delete",
    treeId: "tree-7",
  });
  assert.deepEqual(item, { operation: "delete", tree_id: "tree-7" });
});

test("out-of-bounds drafts are rejected with the property named", () => {
  assert.throws(
    () =>
      composeFamilyEdit(capabilities, "vegetation_geometry", {
        ...vegetationDraft(),
        properties: { ...vegetationDraft().properties, height_m: 120 },
      }),
    (error) => {
      assert.ok(error instanceof DraftValidationError);
      assert.ok(error.errors.some((entry) => entry.property === "height_m"));
      return true;
    },
  );
});

test("unknown adapters are a draft error, not a transport error", () => {
  assert.throws(
    () => composeFamilyEdit(capabilities, "no_such_adapter", {}),
    DraftValidationError,
  );
});

// -- the four universal families: exact POST /edits/universal items ---------

test("parameter drafts compose a universal item carrying only touched properties", () => {
  // touchedProperties keeps a partial update PARTIAL: untouched parameters
  // stay at their scene values instead of being reverted to schema defaults.
  const [item] = composeFamilyEdit(capabilities, "model_receptor_parameters", {
    operation: "update",
    properties: { absK: 0.75, albedo_b: 0.2, ewall: 0.9 },
    touchedProperties: ["absK"],
  });
  assert.deepEqual(item, {
    adapter: "model_receptor_parameters",
    operation: "update",
    values: { absK: 0.75 },
    target: null,
    time_index: null,
    old_values: null,
  });
});

test("parameter drafts without touch-tracking send the full validated set", () => {
  // Drafts that carry no touchedProperties (programmatic callers) send the
  // schema-validated draft with defaults filled in — schema vocabulary only.
  const [item] = composeFamilyEdit(capabilities, "model_receptor_parameters", {
    operation: "update",
    properties: { absK: 0.75 },
  });
  assert.equal(item.values.absK, 0.75);
  assert.equal(item.values.cyl, true); // bool control default
  assert.equal(item.values.transVeg, 0.03);
  // Exactly the document's parameter vocabulary, nothing else.
  const schemaProperties = Object.keys(
    capabilities.adapterById.get("model_receptor_parameters").propertySchema.properties,
  ).sort();
  assert.deepEqual(Object.keys(item.values).sort(), schemaProperties);
});

test("a universal draft that touches nothing is a draft error", () => {
  assert.throws(
    () =>
      composeFamilyEdit(capabilities, "model_receptor_parameters", {
        operation: "update",
        properties: { absK: 0.7 },
        touchedProperties: [],
      }),
    (error) =>
      error instanceof DraftValidationError &&
      error.errors.some((entry) => entry.property === "values"),
  );
});

test("unknown draft properties are rejected before any payload is composed", () => {
  // The universal item must never carry vocabulary the schema does not
  // declare — a typo or a fenced property stops at validation, not at the
  // server endpoint.
  assert.throws(
    () =>
      composeFamilyEdit(capabilities, "model_receptor_parameters", {
        operation: "update",
        properties: { absK: 0.75, absorp_th: 0.4 }, // absorp_th: not in the schema
      }),
    (error) =>
      error instanceof DraftValidationError &&
      error.errors.some((entry) => entry.property === "absorp_th"),
  );
});

test("meteorology drafts require the schema's full variable set, then compose the item", () => {
  // Partial drafts are rejected with every missing variable named — the
  // document's vocabulary, not a local list (a met row update is whole-row
  // by contract, so the schema deliberately declares no defaults).
  assert.throws(
    () =>
      composeFamilyEdit(capabilities, "meteorological_forcing", {
        operation: "update_time_row",
        properties: { air_temperature: 31.5 },
        touchedProperties: ["air_temperature"],
      }),
    (error) =>
      error instanceof DraftValidationError &&
      ["humidity", "radiation", "wind_speed", "pressure", "uhii"].every((name) =>
        error.errors.some((entry) => entry.property === name),
      ),
  );

  const fullProperties = {
    air_temperature: 31.5,
    humidity: 14,
    radiation: 800,
    wind_speed: 2.4,
    pressure: 1013,
    uhii: 0,
  };
  const [item] = composeFamilyEdit(capabilities, "meteorological_forcing", {
    operation: "update_time_row",
    properties: fullProperties,
    touchedProperties: Object.keys(fullProperties),
    timeIndex: 12,
  });
  assert.deepEqual(item, {
    adapter: "meteorological_forcing",
    operation: "update_time_row",
    values: fullProperties,
    target: null,
    time_index: 12,
    old_values: null,
  });
});

test("landcover paint composes a window-targeted universal item", () => {
  const [item] = composeFamilyEdit(capabilities, "landcover_surface", {
    operation: "paint",
    properties: { class: 5 },
    touchedProperties: [],
    target: { row_start: 40, row_stop: 48, col_start: 48, col_stop: 56 },
  });
  // The paint class is always carried (a class-picker family's whole payload
  // is the class), even with nothing else touched.
  assert.deepEqual(item, {
    adapter: "landcover_surface",
    operation: "paint",
    values: { class: 5 },
    target: { row_start: 40, row_stop: 48, col_start: 48, col_stop: 56 },
    time_index: null,
    old_values: null,
  });
});

test("fenced and off-vocabulary paint classes are draft errors", () => {
  // Class 7 is FENCED by the document (dead water branch upstream): it must
  // be rejected here so no transport endpoint can ever receive it.
  assert.throws(
    () =>
      composeFamilyEdit(capabilities, "landcover_surface", {
        operation: "paint",
        properties: { class: 7 },
      }),
    (error) => {
      assert.ok(error instanceof DraftValidationError);
      assert.ok(
        error.errors.some(
          (entry) => entry.property === "class" && /fenced by the document/.test(entry.message),
        ),
      );
      return true;
    },
  );
  // Off-vocabulary classes are rejected with the valid vocabulary named.
  assert.throws(
    () =>
      composeFamilyEdit(capabilities, "landcover_surface", {
        operation: "paint",
        properties: { class: 99 },
      }),
    (error) =>
      error instanceof DraftValidationError &&
      error.errors.some(
        (entry) => entry.property === "class" && /not in the document's vocabulary/.test(entry.message),
      ),
  );
});

test("building add/move/delete compose id-targeted universal items", () => {
  const footprint = [
    [1120, 1856],
    [1144, 1856],
    [1144, 1880],
    [1120, 1880],
  ];
  const [add] = composeFamilyEdit(capabilities, "building_geometry", {
    operation: "add",
    properties: { footprint_m: footprint, height_m: 12 },
    touchedProperties: ["footprint_m", "height_m"],
    target: "b-42",
  });
  assert.deepEqual(add, {
    adapter: "building_geometry",
    operation: "add",
    values: { footprint_m: footprint, height_m: 12 },
    target: "b-42",
    time_index: null,
    old_values: null,
  });

  // A move declares the entity's current values (old_values) alongside the
  // new state — user-declared, validated by the server's adapter.
  const [move] = composeFamilyEdit(capabilities, "building_geometry", {
    operation: "move",
    properties: { footprint_m: footprint, height_m: 15 },
    touchedProperties: ["footprint_m", "height_m"],
    target: "b-42",
    oldValues: { footprint_m: footprint, height_m: 12 },
  });
  assert.deepEqual(move.old_values, { footprint_m: footprint, height_m: 12 });
  assert.equal(move.operation, "move");

  // Delete addresses the identity only.
  const [remove] = composeFamilyEdit(capabilities, "building_geometry", {
    operation: "delete",
    target: "b-42",
    oldValues: { footprint_m: footprint, height_m: 12 },
  });
  assert.deepEqual(remove.values, {});
  assert.deepEqual(remove.old_values, { footprint_m: footprint, height_m: 12 });
  assert.equal(remove.target, "b-42");
});

test("a building draft missing required geometry is a draft error", () => {
  // No schema defaults exist for footprint/height, so an untouched add is
  // refused here — no accidental default-height buildings.
  assert.throws(
    () =>
      composeFamilyEdit(capabilities, "building_geometry", {
        operation: "add",
        properties: {},
        touchedProperties: [],
        target: "b-42",
      }),
    (error) =>
      error instanceof DraftValidationError &&
      error.errors.some((entry) => entry.property === "footprint_m"),
  );
});

test("adapters without a transport entry disclose the universal item, verbatim", () => {
  // The snapshot marks no such editor (the five editors all have transports),
  // so prove the disclosure on a doctored document: terrain_dem marked
  // integrated would be editable but still untransported in this build.
  const localCapabilities = parseCapabilityDocument(snapshot);
  const terrain = localCapabilities.adapterById.get("terrain_dem");
  terrain.integrated = true;
  terrain.editable = true;
  localCapabilities.editableAdapters.push(terrain);
  assert.throws(
    () =>
      composeFamilyEdit(localCapabilities, "terrain_dem", {
        operation: "raise",
        properties: {},
        touchedProperties: [],
      }),
    (error) => {
      assert.ok(error instanceof FamilyTransportError);
      assert.equal(error.adapterId, "terrain_dem");
      // The payload is the exact universal item grammar — no invented keys.
      assert.deepEqual(Object.keys(error.payload).sort(), [
        "adapter",
        "old_values",
        "operation",
        "target",
        "time_index",
        "values",
      ]);
      return true;
    },
  );
});

test("tree-transport transmissivity is read from the document, never restated", () => {
  // The vegetation schema rejects transmissivity as editable and points at
  // model_receptor_parameters.transVeg; the passthrough value is THAT
  // schema's declared default — prove it by moving the document's default.
  const localCapabilities = parseCapabilityDocument(snapshot);
  const transVeg =
    localCapabilities.adapterById.get("model_receptor_parameters").propertySchema.properties
      .transVeg;
  const original = transVeg.default;
  transVeg.default = 0.11;
  try {
    const [item] = composeFamilyEdit(
      localCapabilities,
      "vegetation_geometry",
      vegetationDraft(),
    );
    assert.equal(item.tree.transmissivity, 0.11, "must follow the document's default");
  } finally {
    transVeg.default = original;
  }
  // Against the real snapshot the derived value is the document's 0.03.
  const [snapshotItem] = composeFamilyEdit(capabilities, "vegetation_geometry", vegetationDraft());
  assert.equal(snapshotItem.tree.transmissivity, 0.03);
});
