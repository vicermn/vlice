const FLOATS_PER_DETECTION = 5;
const MAX_DETECTIONS = 20;
const MAX_PLATE_LEN = 8;
const PLATE_SLOT_SIZE = MAX_PLATE_LEN + 1;

export function createDetector({ confidenceThreshold }) {
  let dst;
  let resultBuffer;
  let plateBuffer;
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

    ctx.font = "16px Inter, Arial, sans-serif";
    ctx.textBaseline = "middle";
    for (let i = 0; i < MAX_DETECTIONS; i++) {
      const base = i * 5;
      const conf = detections[base + 0];
      const x = detections[base + 1];
      const y = detections[base + 2];
      const w = detections[base + 3];
      const h = detections[base + 4];

      if (conf <= 0) continue;

      const rawPlate = decodePlate(plateHeap, i);
      if (!rawPlate) continue;

      const normalizedPlate = rawPlate
        .toUpperCase()
        .replace(/[^A-Z0-9]/g, "")
        .trim();

      if (normalizedPlate.length < 6) continue;

      const color = getColor(conf);

      drawModernBox(ctx, x, y, w, h, normalizedPlate, color);
    }
  }

  function drawModernBox(ctx, x, y, w, h, text, color) {
    const radius = 14;

    // === BOX BACKGROUND (glass body) ===
    ctx.save();

    ctx.beginPath();
    roundedRect(ctx, x, y, w, h, radius);

    // Frosted glass fill (darkened for better contrast)
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)"; // Darker background for improved readability
    ctx.fill();

    // Soft gradient border
    const gradient = ctx.createLinearGradient(x, y, x + w, y + h);
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, "rgba(255,255,255,0.6)");

    ctx.lineWidth = 2.5;
    ctx.strokeStyle = gradient;
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.stroke();

    ctx.restore();

    // === GLASS LABEL ===
    drawGlassLabel(ctx, x, y, text, color);
  }

  function drawGlassLabel(ctx, x, y, text, color) {
    const paddingX = 14;
    const paddingY = 8;
    const radius = 10;

    ctx.font = "600 14px Inter, sans-serif";
    const textWidth = ctx.measureText(text).width;

    const labelWidth = textWidth + paddingX * 2;
    const labelHeight = 30;

    const labelX = x;
    const labelY = y - labelHeight - 8;

    ctx.save();

    // Glass background (darkened for better contrast)
    ctx.beginPath();
    roundedRect(ctx, labelX, labelY, labelWidth, labelHeight, radius);

    ctx.fillStyle = "rgba(0, 0, 0, 0.4)"; // Darker glass background
    ctx.fill();

    // Thin glossy border
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.stroke();

    // Soft glow
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;

    // Text
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, labelX + paddingX, labelY + labelHeight / 2);

    ctx.restore();
  }

  function roundedRect(ctx, x, y, w, h, r) {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
  }

  function getColor(conf) {
    if (conf >= 0.6) return "rgba(34,197,94,1)";
    if (conf >= 0.4) return "rgba(255,165,0,1)";
    return "rgba(239,68,68,1)";
  }

  function getColor(conf) {
    if (conf >= 0.6) return "rgba(34,197,94,1)";
    if (conf >= 0.4) return "rgba(255,165,0,1)";
    return "rgba(239,68,68,1)";
  }

  function decodePlate(heap, index) {
    const start = index * PLATE_SLOT_SIZE;
    const bytes = heap.slice(start, start + PLATE_SLOT_SIZE);
    return new TextDecoder().decode(bytes).replace(/\0.*$/, "");
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
