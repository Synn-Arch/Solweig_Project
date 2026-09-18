// SPDX-License-Identifier: GPL-3.0-only
//
// Building gesture → wire-flow gates: the pure mapping chain the DOM shell
// wires for every building gesture —
//
//   makeBuilding (local ring, scene metres)
//     → buildingOperationWord  (add/delete/move/update, the same words
//       commitDesignEdit derives for trees)
//     → buildingWireVerb       (the plane's frozen verb set: add/delete/
//       move/replace — the wire has NO "update", and a move carrying a
//       height change promotes to replace because the reducer's move fold
//       keeps footprint_m only)
//     → buildingWireValues     (footprint_m always whole; height_m whenever
//       the fold stores it)
//     → buildingOperation      (op_builder envelope: identity rides as
//       entity_id, building_id never travels inside values)
//     → buildingFootprintThroughFrame / FromFrame (local scene metres ↔
//       the server's world CRS)
//
// All builders are pure and live outside app.mjs's DOM guard; no server,
// no DOM, no clock is involved.

import test from "node:test";
import assert from "node:assert/strict";

import {
  BUILDING_PRESETS,
  apiValuesToBuilding,
  buildingToWorld,
  cloneBuilding,
  makeBuilding,
} from "../model.mjs";
import {
  OPERATION_VERBS,
  buildingOperation,
  frameFromGeometry,
  frameFromSpans,
} from "../op_builder.mjs";
import {
  buildingFootprintFromFrame,
  buildingFootprintThroughFrame,
  buildingOperationWord,
  buildingWireValues,
  buildingWireVerb,
  footprintCenterM,
} from "../app.mjs";

// The scene shape the building model reads (metre spans for UV↔world).
const SCENE = { widthMeters: 500, heightMeters: 500 };

/** A moved copy of a building at a new UV center (same id, label, ring shape). */
function movedTo(building, u, v) {
  const center = buildingToWorld({ u, v }, SCENE);
  const halfWidthM = building.widthM / 2;
  const halfDepthM = building.depthM / 2;
  return {
    ...building,
    u,
    v,
    footprintM: [
      [center.xM - halfWidthM, center.yM - halfDepthM],
      [center.xM + halfWidthM, center.yM - halfDepthM],
      [center.xM + halfWidthM, center.yM + halfDepthM],
      [center.xM - halfWidthM, center.yM + halfDepthM],
    ],
  };
}

// ---------------------------------------------------------------------------
// Envelope shape — buildingOperation (op_builder)
// ---------------------------------------------------------------------------

test("an add envelope rides building_geometry with entity_id identity and no building_id inside values", () => {
  const building = makeBuilding("pavilion", 0.5, 0.5, 1, SCENE);
  const envelope = buildingOperation(
    "studio-a1",
    "add",
    building.id,
    { footprint_m: building.footprintM, height_m: building.heightM },
    4,
  );
  assert.equal(envelope.source_family, "building_geometry");
  assert.equal(envelope.verb, "add");
  assert.equal(typeof envelope.entity_id, "string");
  assert.equal(envelope.entity_id, building.id);
  assert.deepEqual(envelope.payload.values, {
    footprint_m: building.footprintM,
    height_m: building.heightM,
  });
  assert.equal("building_id" in envelope.payload.values, false);
});

test("buildingOperation strips a building_id that re-aims the identity", () => {
  const envelope = buildingOperation(
    "studio-a1",
    "replace",
    "building-target",
    { building_id: "building-other", height_m: 12 },
    null,
  );
  assert.equal(envelope.payload.values.building_id, undefined);
  assert.deepEqual(envelope.payload.values, { height_m: 12 });
});

test("a delete envelope keeps a non-empty payload with empty values", () => {
  const envelope = buildingOperation("studio-a1", "delete", "building-x", {}, null);
  // The route requires a non-empty payload object; {values: {}} passes shape
  // validation while the reducer's delete fold stores nothing.
  assert.deepEqual(envelope.payload, { values: {} });
  assert.equal(envelope.verb, "delete");
});

// ---------------------------------------------------------------------------
// Gesture words — buildingOperationWord
// ---------------------------------------------------------------------------

test("add and delete classify by object presence, exactly like trees", () => {
  const building = makeBuilding("tower", 0.4, 0.4, 1, SCENE);
  assert.equal(buildingOperationWord(null, building), "add");
  assert.equal(buildingOperationWord(building, null), "delete");
});

test("a translated ring is a move; a steady-center resize or height edit is an update", () => {
  const building = makeBuilding("block", 0.25, 0.25, 1, SCENE);
  assert.equal(buildingOperationWord(building, movedTo(building, 0.5, 0.5)), "move");
  const taller = { ...building, heightM: building.heightM + 6 };
  assert.equal(buildingOperationWord(building, taller), "update");
  // A width edit regenerates the ring AROUND the same center (the property
  // panel's regenerateBuildingFootprint discipline), so it stays an update.
  const center = footprintCenterM(building.footprintM);
  const halfWidthM = (building.widthM + 4) / 2;
  const halfDepthM = building.depthM / 2;
  const wider = {
    ...building,
    widthM: building.widthM + 4,
    footprintM: [
      [center.xM - halfWidthM, center.yM - halfDepthM],
      [center.xM + halfWidthM, center.yM - halfDepthM],
      [center.xM + halfWidthM, center.yM + halfDepthM],
      [center.xM - halfWidthM, center.yM + halfDepthM],
    ],
  };
  assert.equal(buildingOperationWord(building, wider), "update");
});

test("footprintCenterM reports the ring-bounds midpoint", () => {
  assert.deepEqual(footprintCenterM([[0, 0], [10, 0], [10, 20], [0, 20]]), { xM: 5, yM: 10 });
});

// ---------------------------------------------------------------------------
// Wire verbs — buildingWireVerb (the plane's frozen vocabulary)
// ---------------------------------------------------------------------------

test("the wire vocabulary has no update verb — a property edit maps to replace", () => {
  assert.equal(OPERATION_VERBS.has("update"), false);
  assert.equal(OPERATION_VERBS.has("replace"), true);
});

test("add and delete gesture words pass through unchanged", () => {
  assert.equal(buildingWireVerb("add", null, makeBuilding("pavilion", 0.5, 0.5, 1, SCENE)), "add");
  assert.equal(buildingWireVerb("delete", makeBuilding("pavilion", 0.5, 0.5, 1, SCENE), null), "delete");
});

test("a nudge or drag maps to move; a height edit maps to replace", () => {
  const building = makeBuilding("block", 0.3, 0.3, 1, SCENE);
  const moved = movedTo(building, 0.31, 0.3);
  assert.equal(buildingWireVerb("move", building, moved), "move");
  const taller = { ...building, heightM: building.heightM + 6 };
  assert.equal(buildingWireVerb("update", building, taller), "replace");
});

test("a move carrying a simultaneous height change promotes to replace — the move fold keeps footprint_m only", () => {
  const building = makeBuilding("tower", 0.3, 0.3, 1, SCENE);
  const draggedAndResized = { ...movedTo(building, 0.4, 0.4), heightM: building.heightM + 5 };
  // reducer.py _POSITION_FIELDS["building_geometry"] == ("footprint_m",): a
  // "move" would silently DROP the height write.
  assert.equal(buildingWireVerb("move", building, draggedAndResized), "replace");
});

// ---------------------------------------------------------------------------
// Wire values — buildingWireValues
// ---------------------------------------------------------------------------

test("an add/replace carries the whole ring plus height; a move carries the ring only", () => {
  const building = makeBuilding("pavilion", 0.5, 0.5, 1, SCENE);
  assert.deepEqual(buildingWireValues("add", building), {
    footprint_m: building.footprintM,
    height_m: building.heightM,
  });
  assert.deepEqual(buildingWireValues("replace", building), {
    footprint_m: building.footprintM,
    height_m: building.heightM,
  });
  assert.deepEqual(buildingWireValues("move", building), { footprint_m: building.footprintM });
});

test("a height edit's values carry the changed height_m and the authoritative ring", () => {
  const building = makeBuilding("block", 0.5, 0.5, 1, SCENE);
  const taller = { ...building, heightM: 22 };
  const values = buildingWireValues(buildingWireVerb("update", building, taller), taller);
  assert.equal(values.height_m, 22);
  assert.deepEqual(values.footprint_m, building.footprintM);
});

test("a nudge-move's values carry a footprint_m that actually changed", () => {
  const building = makeBuilding("block", 0.3, 0.3, 1, SCENE);
  const moved = movedTo(building, 0.6, 0.6);
  const values = buildingWireValues(buildingWireVerb("move", building, moved), moved);
  assert.deepEqual(Object.keys(values), ["footprint_m"]);
  assert.notEqual(
    JSON.stringify(values.footprint_m),
    JSON.stringify(building.footprintM),
    "the ring moved with the gesture",
  );
});

test("a delete carries no values", () => {
  const building = makeBuilding("pavilion", 0.5, 0.5, 1, SCENE);
  assert.deepEqual(buildingWireValues("delete", building), {});
});

// ---------------------------------------------------------------------------
// Frame conversion — local scene metres ↔ server world CRS
// ---------------------------------------------------------------------------

test("the span-local default frame round-trips footprints exactly", () => {
  // Binary-exact UV (0.5/0.25) so the round-trip through UV is exactly
  // representable — the assertion is strict deep-equality by design. The
  // app's default rt.frame (originY 0) is span-LOCAL, not local-identity:
  // wire y rides originY − v·span while the model's y is north-up
  // ((1 − v)·H) — the same convention worldTree uses for trees, so
  // submit and adopt stay consistent even before calibration solves.
  const building = makeBuilding("tower", 0.5, 0.25, 1, SCENE);
  const frame = frameFromSpans({ spanX: SCENE.widthMeters, spanY: SCENE.heightMeters });
  const wire = buildingFootprintThroughFrame(building.footprintM, SCENE, frame);
  assert.deepEqual(buildingFootprintFromFrame(wire, SCENE, frame), building.footprintM);
  // The true local-identity frame carries originY = spanY (north-up):
  const identity = frameFromSpans({
    originY: SCENE.heightMeters,
    spanX: SCENE.widthMeters,
    spanY: SCENE.heightMeters,
  });
  assert.deepEqual(buildingFootprintThroughFrame(building.footprintM, SCENE, identity), building.footprintM);
});

test("a UTM-origin frame shifts every corner; the inverse restores the local ring", () => {
  const building = makeBuilding("block", 0.75, 0.25, 1, SCENE);
  const frame = frameFromGeometry({
    originX: 600000,
    originY: 4000000,
    cols: 500,
    rows: 500,
    pixelSize: 1,
  });
  const wire = buildingFootprintThroughFrame(building.footprintM, SCENE, frame);
  for (const [xM] of wire) {
    assert.ok(xM >= 600000, "x moved into the server's CRS");
  }
  assert.deepEqual(buildingFootprintFromFrame(wire, SCENE, frame), building.footprintM);
});

test("a full submit round-trip: gesture word → verb → values → envelope", () => {
  const building = makeBuilding("pavilion", 0.5, 0.36, 1, SCENE);
  const operation = buildingOperationWord(null, building);
  const verb = buildingWireVerb(operation, null, building);
  const values = buildingWireValues(verb, building);
  const envelope = buildingOperation("studio-e2e", verb, building.id, values, 7);
  assert.equal(operation, "add");
  assert.equal(envelope.verb, "add");
  assert.equal(envelope.source_family, "building_geometry");
  assert.deepEqual(envelope.payload.values, {
    footprint_m: building.footprintM,
    height_m: BUILDING_PRESETS.pavilion.heightM,
  });
});

// ---------------------------------------------------------------------------
// Model round-trips the adoption path relies on
// ---------------------------------------------------------------------------

test("cloneBuilding deep-copies the ring — vertex edits never alias across copies", () => {
  const building = makeBuilding("block", 0.5, 0.5, 1, SCENE);
  const copy = cloneBuilding(building);
  copy.footprintM[0][0] += 25;
  assert.equal(building.footprintM[0][0] + 25, copy.footprintM[0][0]);
});

test("apiValuesToBuilding re-measures width/depth from the adopted ring", () => {
  const existing = makeBuilding("pavilion", 0.5, 0.5, 1, SCENE);
  const adopted = apiValuesToBuilding(
    { footprint_m: movedTo(existing, 0.7, 0.7).footprintM, height_m: 18 },
    existing,
  );
  assert.equal(adopted.heightM, 18);
  assert.equal(adopted.widthM, existing.widthM);
  assert.equal(adopted.depthM, existing.depthM);
  assert.equal(adopted.label, existing.label, "presentation survives an adoption");
});
