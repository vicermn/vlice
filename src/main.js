import { initWasm } from "./wasmLoader.js";
import { initCamera } from "./camera.js";
import { createDetector } from "./detector.js";
import { detectWasmFeatures } from "./wasmFeatureDetect.js";

const CONF_THRESHOLD = 0.25;

let detector;

document.addEventListener("DOMContentLoaded", async () => {
  const features = await detectWasmFeatures();
  console.log("WASM Features:", features);

  await initWasm();

  detector = createDetector({
    confidenceThreshold: CONF_THRESHOLD,
  });

  await initCamera((ctx, canvas) => {
    detector.processFrame(ctx, canvas);
  });
});
