const FLOATS_PER_DETECTION = 5;
const MAX_DETECTIONS = 20;
const MAX_PLATE_LEN = 8;
const PLATE_SLOT_SIZE = MAX_PLATE_LEN + 1;

export function createDetector({ confidenceThreshold }) {
  let dst;
  let resultBuffer;
  let plateBuffer;
  const detectedPlates = new Set();
  initMemory();

  function initMemory() {
    dst = _malloc(320 * 320 * 4);

    const resultArray = new Float32Array(FLOATS_PER_DETECTION * MAX_DETECTIONS);
    resultBuffer = _malloc(resultArray.byteLength);
    HEAPF32.set(resultArray, resultBuffer / Float32Array.BYTES_PER_ELEMENT);

    const plateArray = new Uint8Array(MAX_DETECTIONS * PLATE_SLOT_SIZE);
    plateBuffer = _malloc(plateArray.byteLength);
    HEAPU8.set(plateArray, plateBuffer);
  }

  function processFrame(ctx, canvas) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    HEAPU8.set(imageData.data, dst);

    _process_frame(
      dst,
      canvas.width,
      canvas.height,
      confidenceThreshold,
      resultBuffer,
      plateBuffer,
    );

    renderDetections(ctx);
  }

  function renderDetections(ctx) {
    const detections = HEAPF32.subarray(
      resultBuffer / Float32Array.BYTES_PER_ELEMENT,
      resultBuffer / Float32Array.BYTES_PER_ELEMENT +
        FLOATS_PER_DETECTION * MAX_DETECTIONS,
    );

    const plateHeap = HEAPU8.subarray(
      plateBuffer,
      plateBuffer + MAX_DETECTIONS * PLATE_SLOT_SIZE,
    );

    for (let i = 0; i < MAX_DETECTIONS; i++) {
      const base = i * 5;
      const conf = detections[base + 0];
      const x = detections[base + 1];
      const y = detections[base + 2];
      const w = detections[base + 3];
      const h = detections[base + 4];

      ctx.lineWidth = 4;
      ctx.strokeStyle = getColor(conf);
      ctx.strokeRect(x, y, w, h);

      const rawPlate = decodePlate(plateHeap, i);
      if (!rawPlate) continue;
      // Normalize:
      // 1. Uppercase
      // 2. Remove non-alphanumeric characters
      // 3. Trim whitespace
      const normalizedPlate = rawPlate
        .toUpperCase()
        .replace(/[^A-Z0-9]/g, "")
        .trim();
      if (normalizedPlate.length < 6) continue;
      // ✅ Only show plates that exist in your CSV set
      // ✅ Only append if it's new
      if (!detectedPlates.has(normalizedPlate)) {
        detectedPlates.add(normalizedPlate);
        addPlateToList(normalizedPlate);
      }
    }
  }

  function decodePlate(heap, index) {
    const start = index * PLATE_SLOT_SIZE;
    const bytes = heap.slice(start, start + PLATE_SLOT_SIZE);
    return new TextDecoder().decode(bytes).replace(/\0.*$/, "");
  }

  function getColor(conf) {
    if (conf >= 0.6) return "rgba(34,197,94,1)";
    if (conf >= 0.4) return "rgba(255,165,0,1)";
    return "rgba(239,68,68,1)";
  }

  function addPlateToList(text) {
    const li = document.createElement("li");
    li.className =
      "p-4 rounded-xl bg-slate-800/60 border border-white/10 backdrop-blur-md";

    li.innerHTML = `
    <div class="flex items-center justify-between">
      <span class="font-mono text-lg tracking-widest text-cyan-400">
        ${text}
      </span>
    </div>
  `;

    document.getElementById("plateList").prepend(li);
  }

  return { processFrame };
}
