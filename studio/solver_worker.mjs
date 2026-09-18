// SPDX-License-Identifier: GPL-3.0-only

import { computeIncrementalPatch } from "./solver_kernel.mjs";

let baseline = null;
let buildingMask = null;
let scene = null;

self.addEventListener("message", async (event) => {
  const message = event.data;
  if (message.type === "init") {
    baseline = new Float32Array(message.baseline);
    buildingMask = new Uint8Array(message.buildingMask);
    scene = message.scene;
    self.postMessage({ type: "ready" });
    return;
  }

  if (message.type !== "analyze") return;
  if (!baseline || !buildingMask || !scene) {
    self.postMessage({
      type: "error",
      jobId: message.jobId,
      revision: message.revision,
      error: "worker was not initialized",
    });
    return;
  }

  const started = performance.now();
  try {
    const result = computeIncrementalPatch({
      baseline,
      buildingMask,
      trees: message.trees,
      window: message.window,
      scene,
      hour: message.hour,
    });

    // Keep the prototype's state transition visible. Production code removes
    // this floor and reports the actual server-side elapsed time.
    const elapsed = performance.now() - started;
    const minimumDurationMs = Number(message.minimumDurationMs ?? 520);
    if (elapsed < minimumDurationMs) {
      await new Promise((resolve) => setTimeout(resolve, minimumDurationMs - elapsed));
    }
    const durationMs = performance.now() - started;
    self.postMessage(
      {
        type: "result",
        jobId: message.jobId,
        revision: message.revision,
        window: result.window,
        metrics: result.metrics,
        durationMs,
        patch: result.patch.buffer,
      },
      [result.patch.buffer],
    );
  } catch (error) {
    self.postMessage({
      type: "error",
      jobId: message.jobId,
      revision: message.revision,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
