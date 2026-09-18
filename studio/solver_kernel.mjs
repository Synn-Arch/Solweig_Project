// SPDX-License-Identifier: GPL-3.0-only

import {
  distancePointToSegment,
  gridWindowArea,
  nearestTimeIndexPosition,
  shadowVectorMeters,
  treeToWorld,
} from "./model.mjs";

export function decodeAnalysisFixture(analysis) {
  const expectedLength = analysis.gridWidth * analysis.gridHeight;
  if (analysis.utci.length !== expectedLength || analysis.buildingMask.length !== expectedLength) {
    throw new Error("analysis fixture dimensions do not match flattened arrays");
  }
  const baseline = new Float32Array(expectedLength);
  const buildingMask = new Uint8Array(expectedLength);
  for (let index = 0; index < expectedLength; index += 1) {
    const value = analysis.utci[index];
    baseline[index] = value === null ? Number.NaN : Number(value);
    buildingMask[index] = analysis.buildingMask[index] ? 1 : 0;
  }
  return { baseline, buildingMask };
}

function treeCoolingDelta(xM, yM, tree, hour, scene) {
  const origin = treeToWorld(tree, scene);
  const shadow = shadowVectorMeters(tree, hour);
  const endX = origin.xM + shadow.eastM;
  const endY = origin.yM + shadow.northM;
  const canopyRadiusM = Math.max(1.5, tree.canopyDiameterM / 2);
  const distanceToShadow = distancePointToSegment(
    xM,
    yM,
    origin.xM,
    origin.yM,
    endX,
    endY,
  );
  const distanceToTrunk = Math.hypot(xM - origin.xM, yM - origin.yM);

  const shadowWidth = canopyRadiusM * 1.18;
  const directShade = Math.max(0, 1 - distanceToShadow / shadowWidth);
  const crownCooling = Math.exp(-Math.pow(distanceToTrunk / (canopyRadiusM * 1.35), 2));
  const diffuseReachM = Math.min(tree.heightM / Math.tan((6 * Math.PI) / 180), 180);
  const diffuse = Math.exp(-distanceToTrunk / Math.max(18, diffuseReachM * 0.4));
  const opacity = 1 - tree.transmissivity;

  return opacity * (3.25 * directShade + 1.35 * crownCooling + 0.42 * diffuse);
}

export function computeIncrementalPatch({
  baseline,
  buildingMask,
  trees,
  window,
  scene,
  hour,
}) {
  const width = scene.gridWidth;
  const height = scene.gridHeight;
  if (baseline.length !== width * height || buildingMask.length !== width * height) {
    throw new Error("baseline arrays do not match the analysis grid");
  }

  const patchWidth = window.colStop - window.colStart;
  const patchHeight = window.rowStop - window.rowStart;
  if (patchWidth <= 0 || patchHeight <= 0) {
    throw new Error("incremental window must have positive area");
  }

  const patch = new Float32Array(patchWidth * patchHeight);
  let patchIndex = 0;
  let deltaSum = 0;
  let validCells = 0;
  let improvedCells = 0;
  let peakCooling = 0;

  for (let row = window.rowStart; row < window.rowStop; row += 1) {
    const v = (row + 0.5) / height;
    const yM = (1 - v) * scene.heightMeters;
    for (let col = window.colStart; col < window.colStop; col += 1) {
      const index = row * width + col;
      const baselineValue = baseline[index];
      if (buildingMask[index] || !Number.isFinite(baselineValue)) {
        patch[patchIndex] = Number.NaN;
        patchIndex += 1;
        continue;
      }

      const u = (col + 0.5) / width;
      const xM = u * scene.widthMeters;
      let coolingPotential = 0;
      for (const tree of trees) {
        coolingPotential += treeCoolingDelta(xM, yM, tree, hour, scene);
      }

      // Saturation prevents overlapping trees from producing unbounded cooling.
      const cooling = 5.2 * (1 - Math.exp(-coolingPotential / 5.2));
      const proposalValue = baselineValue - cooling;
      patch[patchIndex] = proposalValue;
      patchIndex += 1;

      const delta = proposalValue - baselineValue;
      deltaSum += delta;
      validCells += 1;
      if (delta <= -1) improvedCells += 1;
      peakCooling = Math.min(peakCooling, delta);
    }
  }

  const sourcePixelAreaM2 =
    (scene.widthMeters / width) * (scene.heightMeters / height);
  return {
    patch,
    window: { ...window },
    metrics: {
      meanDelta: validCells ? deltaSum / validCells : 0,
      peakDelta: peakCooling,
      improvedAreaM2: improvedCells * sourcePixelAreaM2,
      affectedCells: gridWindowArea(window),
      validCells,
    },
  };
}

export function applyPatch(target, patch, window, gridWidth) {
  const patchWidth = window.colStop - window.colStart;
  const patchHeight = window.rowStop - window.rowStart;
  if (patch.length !== patchWidth * patchHeight) {
    throw new Error("patch length does not match its window");
  }
  let patchIndex = 0;
  for (let row = window.rowStart; row < window.rowStop; row += 1) {
    const targetStart = row * gridWidth + window.colStart;
    target.set(patch.subarray(patchIndex, patchIndex + patchWidth), targetStart);
    patchIndex += patchWidth;
  }
  return target;
}

/**
 * Apply one decoded server result variable into a full time cube.
 *
 * The cube is laid out `(timeSteps, gridHeight, gridWidth)` — the same
 * variable-major/time/row/column order the server encodes — and the patch
 * window is half-open (`[rowStart, rowStop) × [colStart, colStop)`).
 *
 * @param {Float32Array} cube full-site cube for one variable
 * @param {Float32Array} values decoded patch variable, `(time, rows, cols)`
 * @param {object} options
 * @param {object} options.window decoded half-open window bounds
 * @param {number[]} options.timeIndices manifest time indices of `values`
 * @param {number} options.hour displayed/requested solar hour
 * @param {number} options.timeSteps cube time depth
 * @param {number} options.gridWidth cube grid width (columns)
 * @param {number} options.gridHeight cube grid height (rows)
 * @returns {number|null} the absolute time step written, or null when the
 *   patch does not cover a usable time for `hour`
 */
export function applyExactPatch(
  cube,
  values,
  { window, timeIndices, hour, timeSteps, gridWidth, gridHeight },
) {
  const position = nearestTimeIndexPosition(timeIndices, hour);
  if (position < 0) return null;
  const timeStep = Number(timeIndices[position]);
  if (!Number.isInteger(timeStep) || timeStep < 0 || timeStep >= timeSteps) return null;

  const width = window.colStop - window.colStart;
  const height = window.rowStop - window.rowStart;
  if (
    window.rowStart < 0 ||
    window.rowStop > gridHeight ||
    window.colStart < 0 ||
    window.colStop > gridWidth
  ) {
    throw new Error("patch window lies outside the site grid");
  }
  const planeSize = width * height;
  const slice = values.subarray(position * planeSize, (position + 1) * planeSize);
  if (slice.length !== width * height) {
    throw new Error("decoded patch slice does not match its window");
  }

  let sourceIndex = 0;
  for (let row = window.rowStart; row < window.rowStop; row += 1) {
    const targetStart = (timeStep * gridHeight + row) * gridWidth + window.colStart;
    cube.set(slice.subarray(sourceIndex, sourceIndex + width), targetStart);
    sourceIndex += width;
  }
  return timeStep;
}

/** View one absolute time step of a full cube as a flat row-major plane. */
export function cubeTimePlane(cube, timeStep, gridWidth, gridHeight) {
  const planeSize = gridWidth * gridHeight;
  const start = timeStep * planeSize;
  if (start + planeSize > cube.length) {
    throw new RangeError("time step lies outside the cube");
  }
  return cube.subarray(start, start + planeSize);
}
