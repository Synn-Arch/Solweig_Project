// SPDX-License-Identifier: GPL-3.0-only

export const TREE_PRESETS = Object.freeze({
  shade: Object.freeze({
    label: "Shade tree",
    heightM: 12,
    canopyDiameterM: 7,
    trunkRatio: 0.25,
    transmissivity: 0.03,
    crown: "round",
  }),
  broad: Object.freeze({
    label: "Broad canopy",
    heightM: 18,
    canopyDiameterM: 11,
    trunkRatio: 0.25,
    transmissivity: 0.03,
    crown: "broad",
  }),
  evergreen: Object.freeze({
    label: "Evergreen",
    heightM: 10,
    canopyDiameterM: 5,
    trunkRatio: 0.25,
    transmissivity: 0.08,
    crown: "conical",
  }),
});

export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function makeTree(presetName, u, v, ordinal = 1) {
  const preset = TREE_PRESETS[presetName];
  if (!preset) {
    throw new Error(`Unknown tree preset: ${presetName}`);
  }
  const suffix = String(ordinal).padStart(2, "0");
  return {
    // The millisecond stamp alone collides across ACTORS (two editors adding
    // in the same ms + same ordinal mint one shared id, and the operation
    // plane upserts by tree_id — the second add silently replaces the first).
    // The random tail scopes ids to the actor's own gesture.
    id: `tree-${Date.now().toString(36)}-${suffix}-${Math.random().toString(36).slice(2, 6)}`,
    label: `${preset.label} ${suffix}`,
    preset: presetName,
    u: clamp(u, 0, 1),
    v: clamp(v, 0, 1),
    heightM: preset.heightM,
    canopyDiameterM: preset.canopyDiameterM,
    // Every preset declares both (guarded by the test suite) — no restated
    // schema defaults here to drift from the capability document / contract.
    trunkRatio: preset.trunkRatio,
    transmissivity: preset.transmissivity,
    crown: preset.crown,
  };
}

export function applyHomography(matrix, x, y) {
  if (!Array.isArray(matrix) || matrix.length !== 3) {
    throw new TypeError("matrix must be a 3×3 array");
  }
  const denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
  if (Math.abs(denominator) < 1e-12) {
    throw new RangeError("homography maps the point to infinity");
  }
  return {
    x: (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
    y: (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
  };
}

export function uvToCanvas(u, v, scene) {
  return applyHomography(scene.homography, u, v);
}

export function canvasToUv(x, y, scene) {
  const point = applyHomography(scene.inverseHomography, x, y);
  return { u: point.x, v: point.y };
}

export function isUvInsideSite(point, tolerance = 0) {
  return (
    point.u >= -tolerance &&
    point.u <= 1 + tolerance &&
    point.v >= -tolerance &&
    point.v <= 1 + tolerance
  );
}

export function sunForHour(hour) {
  const clampedHour = clamp(Number(hour), 6, 18);
  const phase = ((clampedHour - 6) / 12) * Math.PI;
  const altitudeDeg = Math.max(5, 66 * Math.sin(phase));
  const azimuthDeg = 90 + (clampedHour - 6) * 15;
  return { altitudeDeg, azimuthDeg };
}

export function shadowVectorMeters(tree, hour, maximumLengthM = 300) {
  const sun = sunForHour(hour);
  const altitude = Math.max(5, sun.altitudeDeg) * (Math.PI / 180);
  const azimuth = (sun.azimuthDeg % 360) * (Math.PI / 180);
  const lengthM = Math.min(tree.heightM / Math.tan(altitude), maximumLengthM);
  return {
    eastM: -lengthM * Math.sin(azimuth),
    northM: -lengthM * Math.cos(azimuth),
    lengthM,
  };
}

export function treeToWorld(tree, scene) {
  return {
    xM: tree.u * scene.widthMeters,
    yM: (1 - tree.v) * scene.heightMeters,
  };
}

export function worldToUv(xM, yM, scene) {
  return {
    u: xM / scene.widthMeters,
    v: 1 - yM / scene.heightMeters,
  };
}

export function influenceRectForTree(
  tree,
  hour,
  scene,
  {
    lowestSkyPatchAltitudeDeg = 6,
    maximumRadiusM = 300,
    safetyMarginM = 6,
  } = {},
) {
  const position = treeToWorld(tree, scene);
  const canopyRadiusM = tree.canopyDiameterM / 2;
  const svfRadiusM = Math.min(
    tree.heightM / Math.tan((lowestSkyPatchAltitudeDeg * Math.PI) / 180),
    maximumRadiusM,
  );
  const baseRadiusM = Math.max(canopyRadiusM, svfRadiusM) + safetyMarginM;
  const shadow = shadowVectorMeters(tree, hour, maximumRadiusM);
  const corridorRadiusM = canopyRadiusM + safetyMarginM;
  const endX = position.xM + shadow.eastM;
  const endY = position.yM + shadow.northM;

  const minXM = Math.min(
    position.xM - baseRadiusM,
    position.xM - corridorRadiusM,
    endX - corridorRadiusM,
  );
  const maxXM = Math.max(
    position.xM + baseRadiusM,
    position.xM + corridorRadiusM,
    endX + corridorRadiusM,
  );
  const minYM = Math.min(
    position.yM - baseRadiusM,
    position.yM - corridorRadiusM,
    endY - corridorRadiusM,
  );
  const maxYM = Math.max(
    position.yM + baseRadiusM,
    position.yM + corridorRadiusM,
    endY + corridorRadiusM,
  );

  return clampRect({
    minU: minXM / scene.widthMeters,
    maxU: maxXM / scene.widthMeters,
    minV: 1 - maxYM / scene.heightMeters,
    maxV: 1 - minYM / scene.heightMeters,
  });
}

export function unionRects(rectangles) {
  const present = rectangles.filter(Boolean);
  if (present.length === 0) {
    return null;
  }
  return clampRect({
    minU: Math.min(...present.map((rect) => rect.minU)),
    minV: Math.min(...present.map((rect) => rect.minV)),
    maxU: Math.max(...present.map((rect) => rect.maxU)),
    maxV: Math.max(...present.map((rect) => rect.maxV)),
  });
}

export function clampRect(rect) {
  const minU = clamp(Math.min(rect.minU, rect.maxU), 0, 1);
  const maxU = clamp(Math.max(rect.minU, rect.maxU), 0, 1);
  const minV = clamp(Math.min(rect.minV, rect.maxV), 0, 1);
  const maxV = clamp(Math.max(rect.minV, rect.maxV), 0, 1);
  return { minU, minV, maxU, maxV };
}

export function dirtyRectForEdit(oldTree, newTree, hour, scene, options = {}) {
  if (!oldTree && !newTree) {
    throw new Error("oldTree or newTree is required");
  }
  return unionRects([
    oldTree ? influenceRectForTree(oldTree, hour, scene, options) : null,
    newTree ? influenceRectForTree(newTree, hour, scene, options) : null,
  ]);
}

export function accumulateDirtyRect(
  pendingRect,
  incomingRect,
  fullRecomputeFraction = 0.3,
) {
  if (!(fullRecomputeFraction > 0 && fullRecomputeFraction <= 1)) {
    throw new RangeError("fullRecomputeFraction must be in (0, 1]");
  }
  const combined = unionRects([pendingRect, incomingRect]);
  if (!combined) return null;
  return rectAreaFraction(combined) >= fullRecomputeFraction
    ? { minU: 0, minV: 0, maxU: 1, maxV: 1 }
    : combined;
}

export function rectAreaFraction(rect) {
  if (!rect) return 0;
  return Math.max(0, rect.maxU - rect.minU) * Math.max(0, rect.maxV - rect.minV);
}

export function rectToGridWindow(rect, gridWidth, gridHeight, blockSize = 4) {
  if (!rect) {
    throw new Error("rect is required");
  }
  if (gridWidth <= 0 || gridHeight <= 0 || blockSize <= 0) {
    throw new RangeError("grid dimensions and block size must be positive");
  }
  const alignDown = (value) => Math.floor(value / blockSize) * blockSize;
  const alignUp = (value) => Math.ceil(value / blockSize) * blockSize;
  return {
    colStart: clamp(alignDown(Math.floor(rect.minU * gridWidth)), 0, gridWidth),
    colStop: clamp(alignUp(Math.ceil(rect.maxU * gridWidth)), 0, gridWidth),
    rowStart: clamp(alignDown(Math.floor(rect.minV * gridHeight)), 0, gridHeight),
    rowStop: clamp(alignUp(Math.ceil(rect.maxV * gridHeight)), 0, gridHeight),
  };
}

export function unionGridWindows(windows, gridWidth, gridHeight) {
  const present = windows.filter(Boolean);
  if (present.length === 0) return null;
  return {
    colStart: clamp(Math.min(...present.map((w) => w.colStart)), 0, gridWidth),
    colStop: clamp(Math.max(...present.map((w) => w.colStop)), 0, gridWidth),
    rowStart: clamp(Math.min(...present.map((w) => w.rowStart)), 0, gridHeight),
    rowStop: clamp(Math.max(...present.map((w) => w.rowStop)), 0, gridHeight),
  };
}

export function gridWindowArea(window) {
  if (!window) return 0;
  return Math.max(0, window.colStop - window.colStart) * Math.max(0, window.rowStop - window.rowStart);
}

export function distancePointToSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const lengthSquared = abx * abx + aby * aby;
  if (lengthSquared === 0) {
    return Math.hypot(px - ax, py - ay);
  }
  const t = clamp(((px - ax) * abx + (py - ay) * aby) / lengthSquared, 0, 1);
  const x = ax + t * abx;
  const y = ay + t * aby;
  return Math.hypot(px - x, py - y);
}

export function findTreeAtCanvasPoint(trees, x, y, scene, hitRadiusPx = 26) {
  let best = null;
  let bestDistance = Infinity;
  for (const tree of trees) {
    const point = uvToCanvas(tree.u, tree.v, scene);
    const distance = Math.hypot(point.x - x, point.y - y);
    if (distance <= hitRadiusPx && distance < bestDistance) {
      best = tree;
      bestDistance = distance;
    }
  }
  return best;
}

export function cloneTree(tree) {
  return tree ? { ...tree } : null;
}

export function isCurrentWorkerMessage(message, latestJobId, currentRevision) {
  return (
    Number(message?.jobId) === Number(latestJobId) &&
    Number(message?.revision) === Number(currentRevision)
  );
}

export function formatSignedTemperature(value) {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(1)}°C`;
}

export function formatInteger(value) {
  return Math.round(value).toLocaleString("en-US");
}

// ---------------------------------------------------------------------------
// API tree-object mapping (docs/incremental_design_tool/data_model.md)
// ---------------------------------------------------------------------------

const API_CROWN_BY_PRESET = { shade: "round", broad: "broad", evergreen: "conical" };
const PRESET_BY_API_CROWN = { conical: "evergreen", broad: "broad" };

/**
 * Convert an internal design tree into the contract `TreeObject`.
 *
 * The server model is strict (`extra="forbid"`): only contract fields are
 * sent; presentation-only attributes (label, preset/crown style) travel in
 * the free-form `metadata` object so a reload can restore them.
 */
export function treeToApiObject(tree) {
  if (!tree) return null;
  return {
    tree_id: tree.id,
    component_type: "broad_canopy",
    u: Number(tree.u),
    v: Number(tree.v),
    height_m: Number(tree.heightM),
    canopy_diameter_m: Number(tree.canopyDiameterM),
    // Schema property trunk_ratio ([0,1), schema-declared default): every
    // local tree carries it (makeTree/apiObjectToTree fill it from the
    // presets / the server contract), so no restated default lives here. A
    // malformed tree surfaces as a server-side 422, not a silent local value.
    trunk_ratio: Number(tree.trunkRatio),
    transmissivity: Number(tree.transmissivity),
    phenology: "deciduous",
    metadata: {
      label: tree.label,
      preset: tree.preset,
    },
  };
}

/** Inverse of {@link treeToApiObject} (presentation attributes best-effort). */
export function apiObjectToTree(apiTree, ordinal = 1) {
  if (!apiTree) return null;
  const preset = apiTree.metadata?.preset ?? PRESET_BY_API_CROWN.broad;
  const known = TREE_PRESETS[preset] ? preset : Object.keys(TREE_PRESETS)[0];
  const suffix = String(ordinal).padStart(2, "0");
  return {
    id: apiTree.tree_id,
    label: apiTree.metadata?.label ?? `${TREE_PRESETS[known].label} ${suffix}`,
    preset: known,
    u: Number(apiTree.u),
    v: Number(apiTree.v),
    heightM: Number(apiTree.height_m),
    canopyDiameterM: Number(apiTree.canopy_diameter_m),
    // The server's TreeObject always serializes trunk_ratio/transmissivity
    // (its model declares defaults), so no client-side restatement is needed.
    trunkRatio: Number(apiTree.trunk_ratio),
    transmissivity: Number(apiTree.transmissivity),
    crown: API_CROWN_BY_PRESET[known] ?? "round",
  };
}

/** Build a contract edit item from an internal design-tree change. */
export function editItemForChange(operation, oldTree, newTree) {
  if (operation === "delete") {
    return { operation: "delete", tree_id: oldTree.id };
  }
  return { operation, tree: treeToApiObject(newTree) };
}

// ---------------------------------------------------------------------------
// Building components (realtime family `building_geometry`)
// ---------------------------------------------------------------------------
//
// Buildings are the second design-object family after trees. The realtime
// contract differs from the tree object API in one shape decision: identity
// rides as `entity_id` (= building_id) on the operation, while `values`
// carries ONLY geometry — `footprint_m` (site-CRS world meters, unclosed
// ring, >= 3 distinct vertices) and `height_m` in (0, 1000]. The engine
// paints DEM + height in the footprint cells, so the ring — not a center
// point — is the authoritative geometry, and it is minted in world meters
// at placement time.

export const BUILDING_PRESETS = Object.freeze({
  pavilion: Object.freeze({
    label: "Pavilion",
    widthM: 8,
    depthM: 8,
    heightM: 4,
  }),
  block: Object.freeze({
    label: "Residential block",
    widthM: 16,
    depthM: 10,
    heightM: 15,
  }),
  tower: Object.freeze({
    label: "Tower",
    widthM: 12,
    depthM: 12,
    heightM: 40,
  }),
});

/**
 * Mint a design building from a preset, centered at the placed UV point.
 *
 * The footprint ring is derived once, here, in site-CRS world meters — the
 * coordinate system both the engine and the wire speak — so downstream
 * consumers (dirty rects, wire values) read geometry instead of re-deriving
 * it from u/v, which would drift after a remote adoption reshapes the ring.
 */
export function makeBuilding(presetName, u, v, ordinal = 1, scene) {
  const preset = BUILDING_PRESETS[presetName];
  if (!preset) {
    throw new Error(`Unknown building preset: ${presetName}`);
  }
  const suffix = String(ordinal).padStart(2, "0");
  const clampedU = clamp(u, 0, 1);
  const clampedV = clamp(v, 0, 1);
  const center = buildingToWorld({ u: clampedU, v: clampedV }, scene);
  const halfWidthM = preset.widthM / 2;
  const halfDepthM = preset.depthM / 2;
  return {
    // Same collision rationale as makeTree: the millisecond stamp alone
    // collides across ACTORS (the operation plane upserts by entity_id, so
    // the second add silently replaces the first); the random tail scopes
    // ids to the actor's own gesture.
    id: `building-${Date.now().toString(36)}-${suffix}-${Math.random().toString(36).slice(2, 6)}`,
    label: `${preset.label} ${suffix}`,
    preset: presetName,
    u: clampedU,
    v: clampedV,
    // v1 footprints are axis-aligned rectangles (rotation is not a wire
    // value yet); widthM/depthM are kept beside the ring so property editing
    // can offer sliders without re-measuring vertices.
    widthM: preset.widthM,
    depthM: preset.depthM,
    heightM: preset.heightM,
    // Counter-clockwise, UNCLOSED ring (the first vertex is not repeated —
    // the wire format expects an open vertex list) of the four corners.
    footprintM: [
      [center.xM - halfWidthM, center.yM - halfDepthM],
      [center.xM + halfWidthM, center.yM - halfDepthM],
      [center.xM + halfWidthM, center.yM + halfDepthM],
      [center.xM - halfWidthM, center.yM + halfDepthM],
    ],
  };
}

/** World-meter center of a building (same UV convention as trees: y north-up). */
export function buildingToWorld(building, scene) {
  return {
    xM: building.u * scene.widthMeters,
    yM: (1 - building.v) * scene.heightMeters,
  };
}

/** Axis-aligned world-meter bounds of a footprint ring ({ Infinity } if empty). */
function buildingFootprintBoundsM(building) {
  let minXM = Infinity;
  let minYM = Infinity;
  let maxXM = -Infinity;
  let maxYM = -Infinity;
  for (const [xM, yM] of building.footprintM) {
    minXM = Math.min(minXM, xM);
    maxXM = Math.max(maxXM, xM);
    minYM = Math.min(minYM, yM);
    maxYM = Math.max(maxYM, yM);
  }
  return { minXM, minYM, maxXM, maxYM };
}

/**
 * Site-UV rect covering every cell a building can change at `hour`.
 *
 * Union of the footprint rect grown by the sky-obstruction margin (a wall
 * blocks low sky patches out to height / tan(altitude), the rect playing
 * the role the canopy circle plays for trees) and the shadow corridor: the
 * displaced footprint is the convex hull of its four displaced corners, so
 * folding each corner's shadow endpoint in (min/max) is exact and lets the
 * FAR corner along the shadow direction extend the corridor conservatively.
 */
export function influenceRectForBuilding(
  building,
  hour,
  scene,
  {
    lowestSkyPatchAltitudeDeg = 6,
    maximumRadiusM = 300,
    safetyMarginM = 6,
  } = {},
) {
  const bounds = buildingFootprintBoundsM(building);
  const obstructionMarginM =
    Math.min(
      building.heightM / Math.tan((lowestSkyPatchAltitudeDeg * Math.PI) / 180),
      maximumRadiusM,
    ) + safetyMarginM;
  let minXM = bounds.minXM - obstructionMarginM;
  let minYM = bounds.minYM - obstructionMarginM;
  let maxXM = bounds.maxXM + obstructionMarginM;
  let maxYM = bounds.maxYM + obstructionMarginM;

  // shadowVectorMeters reads heightM only, so the building passes as-is.
  const shadow = shadowVectorMeters(building, hour, maximumRadiusM);
  for (const [cornerXM, cornerYM] of building.footprintM) {
    minXM = Math.min(minXM, cornerXM + shadow.eastM - safetyMarginM);
    minYM = Math.min(minYM, cornerYM + shadow.northM - safetyMarginM);
    maxXM = Math.max(maxXM, cornerXM + shadow.eastM + safetyMarginM);
    maxYM = Math.max(maxYM, cornerYM + shadow.northM + safetyMarginM);
  }

  return clampRect({
    minU: minXM / scene.widthMeters,
    maxU: maxXM / scene.widthMeters,
    minV: 1 - maxYM / scene.heightMeters,
    maxV: 1 - minYM / scene.heightMeters,
  });
}

/**
 * Dirty rect for a building edit — union of the old and new influence
 * regions (add = {null, new}, delete = {old, null}, move/update = both).
 *
 * Named separately from the tree {@link dirtyRectForEdit}: that export is
 * already owned by the tree family, and dispatch-by-shape here would
 * entangle the two influence functions.
 */
export function dirtyRectForBuildingEdit(oldBuilding, newBuilding, hour, scene, options = {}) {
  if (!oldBuilding && !newBuilding) {
    throw new Error("oldBuilding or newBuilding is required");
  }
  return unionRects([
    oldBuilding ? influenceRectForBuilding(oldBuilding, hour, scene, options) : null,
    newBuilding ? influenceRectForBuilding(newBuilding, hour, scene, options) : null,
  ]);
}

/**
 * Deep-copy a building. The footprint ring holds mutable corner arrays — a
 * shallow spread would alias them between copies, letting a vertex edit on
 * one leak into the other. Every other field is a scalar, so shallow is
 * exact for the rest.
 */
export function cloneBuilding(building) {
  return building
    ? { ...building, footprintM: building.footprintM.map(([xM, yM]) => [xM, yM]) }
    : null;
}

/**
 * Convert an internal building into the realtime `values` payload for the
 * `building_geometry` family (verbs add/move/update/delete; identity rides
 * as entity_id on the operation, not here).
 *
 * The values carry geometry verbatim — footprint vertices are already
 * site-CRS world meters and heights plain meter scalars, so nothing is
 * rounded or converted. A building_id inside values would be stripped or
 * refused as a re-aimed id, so it is never included.
 */
export function buildingValuesToApi(building) {
  if (!building) return null;
  return {
    footprint_m: building.footprintM,
    height_m: building.heightM,
  };
}

/**
 * Inverse of {@link buildingValuesToApi}: fold a remote actor's wire
 * `values` back into a local building (adoption of add/update echoes).
 *
 * Wire vertices are already world meters in the site CRS — no UV
 * conversion. Placement u/v is left untouched: re-deriving it from the ring
 * needs the scene, which the values payload does not carry, so recentering
 * stays with the caller that holds one. widthM/depthM are re-measured from
 * the ring's bounds so a remote reshape cannot leave stale dimensions
 * beside new vertices.
 */
export function apiValuesToBuilding(values, existing = null) {
  if (!values) return existing ? cloneBuilding(existing) : null;
  const footprintM = values.footprint_m.map(([xM, yM]) => [Number(xM), Number(yM)]);
  const bounds = buildingFootprintBoundsM({ footprintM });
  return {
    ...existing,
    heightM: Number(values.height_m),
    footprintM,
    widthM: bounds.maxXM - bounds.minXM,
    depthM: bounds.maxYM - bounds.minYM,
  };
}

/**
 * Pick the manifest time index closest to a requested solar hour.
 * Baseline results cover every time step; edit results cover what was asked.
 */
export function selectTimeIndex(timeIndices, hour) {
  if (!Array.isArray(timeIndices) || timeIndices.length === 0) return null;
  let best = timeIndices[0];
  let bestDistance = Math.abs(Number(timeIndices[0]) - Number(hour));
  for (const candidate of timeIndices.slice(1)) {
    const distance = Math.abs(Number(candidate) - Number(hour));
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return Number(best);
}

/**
 * Array position (not value) of the time index closest to `hour`.
 * A patch's value array is laid out (time, rows, cols) with `time` following
 * the manifest `time_indices` order, so callers slicing a patch need the
 * position while callers labeling the display need the value.
 */
export function nearestTimeIndexPosition(timeIndices, hour) {
  if (!Array.isArray(timeIndices) || timeIndices.length === 0) return -1;
  let best = 0;
  let bestDistance = Math.abs(Number(timeIndices[0]) - Number(hour));
  for (let index = 1; index < timeIndices.length; index += 1) {
    const distance = Math.abs(Number(timeIndices[index]) - Number(hour));
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best;
}

/** True when a patch covering `timeIndices` can update the displayed hour. */
export function timeIndicesCoverHour(timeIndices, hour) {
  return selectTimeIndex(timeIndices, hour) !== null;
}

/**
 * Convert a server raster window (half-open grid bounds) into a site UV rect
 * so the dirty-region overlay can draw the authoritative recompute region.
 */
export function gridWindowToRect(window, rows, cols) {
  return {
    minU: window.colStart / cols,
    maxU: window.colStop / cols,
    minV: window.rowStart / rows,
    maxV: window.rowStop / rows,
  };
}

// ---------------------------------------------------------------------------
// Debounced property editing (frontend_spec.md "Property editing")
// ---------------------------------------------------------------------------

/**
 * Coalesce rapid property-slider input into exactly one preview recompute and
 * one committed edit.
 *
 * Spec: "input event -> update visual tree immediately; no input for 400 to
 * 600 ms -> commit one replacement tree. The client stores the tree state at
 * the beginning of the debounce interval so the old influence region is not
 * lost."
 *
 * Timers are injectable so node tests drive the clock manually.
 */
export function createPropertyEditor({
  preview,
  commit,
  previewDelayMs = 150,
  commitDelayMs = 480,
  setTimeoutFn = globalThis.setTimeout?.bind(globalThis),
  clearTimeoutFn = globalThis.clearTimeout?.bind(globalThis),
} = {}) {
  if (typeof preview !== "function" || typeof commit !== "function") {
    throw new TypeError("createPropertyEditor requires preview and commit functions");
  }
  let editStartTree = null;
  let pendingTree = null;
  let previewTimer = null;
  let commitTimer = null;

  function flush() {
    if (pendingTree === null) return;
    const oldTree = editStartTree;
    const newTree = pendingTree;
    editStartTree = null;
    pendingTree = null;
    commit(oldTree, newTree);
  }

  return {
    input(currentTree) {
      // The tree state at the START of the debounce interval is kept so the
      // old influence region is still known when the edit finally commits.
      if (pendingTree === null) editStartTree = currentTree ? cloneTree(currentTree) : null;
      pendingTree = currentTree ? cloneTree(currentTree) : null;
      if (previewTimer !== null) clearTimeoutFn(previewTimer);
      previewTimer = setTimeoutFn(() => {
        previewTimer = null;
        preview(editStartTree, pendingTree);
      }, previewDelayMs);
      if (commitTimer !== null) clearTimeoutFn(commitTimer);
      commitTimer = setTimeoutFn(() => {
        commitTimer = null;
        flush();
      }, commitDelayMs);
    },
    /**
     * Rebase a pending debounce round onto `tree` without restarting the
     * timers. Called when a pointer drag commits mid-debounce: the pending
     * property edit must carry the tree's NEW position, or its 400-600 ms
     * commit would send the pre-drag u/v and snap the tree back.
     */
    rebase(tree) {
      if (pendingTree === null) return;
      pendingTree = tree ? cloneTree(tree) : null;
    },
    /** Drop a pending commit (e.g. the tree was deleted mid-debounce). */
    cancel() {
      if (previewTimer !== null) clearTimeoutFn(previewTimer);
      if (commitTimer !== null) clearTimeoutFn(commitTimer);
      previewTimer = null;
      commitTimer = null;
      editStartTree = null;
      pendingTree = null;
    },
  };
}

/**
 * Pure metric derivation for a server result (UI-005 card).
 *
 * `manifest` uses the contract's snake_case (`metrics.window_fraction`,
 * `window.row_start`), while the DECODED patch window is camelCase — callers
 * must pass the decoded window, never the manifest's raw one.
 *
 * @param {object} manifest raw server manifest (snake_case)
 * @param {object|null} decodedWindow camelCase {rowStart,rowStop,colStart,colStop}
 * @param {{rows: number, cols: number}} grid exact grid shape
 */
export function computeExactMetrics(manifest, decodedWindow, grid) {
  const metrics = manifest?.metrics ?? {};
  const totalCells = grid && grid.rows && grid.cols ? grid.rows * grid.cols : null;
  let windowCells = null;
  if (
    decodedWindow &&
    Number.isFinite(decodedWindow.rowStart) &&
    Number.isFinite(decodedWindow.rowStop) &&
    Number.isFinite(decodedWindow.colStart) &&
    Number.isFinite(decodedWindow.colStop)
  ) {
    windowCells =
      (decodedWindow.rowStop - decodedWindow.rowStart) *
      (decodedWindow.colStop - decodedWindow.colStart);
  }
  const fraction =
    metrics.window_fraction !== undefined
      ? Number(metrics.window_fraction)
      : totalCells !== null && windowCells !== null
        ? windowCells / totalCells
        : null;
  return {
    meanDeltaC: metrics.mean_utci_delta_c,
    peakDeltaC: metrics.peak_utci_delta_c,
    improvedAreaM2: metrics.improved_area_m2,
    durationMs: metrics.duration_ms,
    fraction,
    windowCells,
    totalCells,
    window: decodedWindow ?? null,
    sceneVersion: manifest?.scene_version,
  };
}

/**
 * Model-scope badge for a connection state (L-6/R-4): "Exact" is claimed only
 * once a verified server result exists ("connected" — the baseline has landed
 * and its checksum was verified). While connecting nothing is verified yet, so
 * the badge stays neutral; offline and failed-connect states stay honest
 * about showing a preview.
 *
 * @param {"connecting"|"connected"|"failed"|"local"} kind connection pill state
 * @returns {{text: string, state: string}}
 */
export function exactBadgeForConnection(kind) {
  if (kind === "connected") return { text: "Exact", state: "exact" };
  if (kind === "connecting") return { text: "Connecting", state: "connecting" };
  return { text: "Preview", state: "preview" };
}
