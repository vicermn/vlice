const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");

let session = null;
let labels = null;
let frameCount = 0;

const INPUT_SIZE = 640;
const OCR_SIZE = { w: 150, h: 50 };

const CONF_THRESHOLD = 0.6;
const IOU_THRESHOLD = 0.45;

// --- Tesseract OCR ---
let ocrWorker = null;
async function initOCR() {
  ocrWorker = await Tesseract.createWorker();
  await ocrWorker.setParameters({
    tessedit_char_whitelist: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
  });
}

// --- Reusable canvases ---
const offscreen = document.createElement("canvas");
offscreen.width = INPUT_SIZE;
offscreen.height = INPUT_SIZE;
const offctx = offscreen.getContext("2d");

const plateCanvas = document.createElement("canvas");
plateCanvas.width = OCR_SIZE.w;
plateCanvas.height = OCR_SIZE.h;
const plateCtx = plateCanvas.getContext("2d");

// --- Plate cache ---
const seenPlates = new Map();

// --- Helpers ---
function showMessage(msg) {
  console.warn(msg);
  ctx.fillStyle = "red";
  ctx.font = "18px monospace";
  ctx.fillText(msg, 10, 30);
}

// ===============================
// ✅ NEW: Letterbox (YOLO-correct)
// ===============================
function letterbox(video, size) {
  const scale = Math.min(size / video.videoWidth, size / video.videoHeight);
  const newW = Math.round(video.videoWidth * scale);
  const newH = Math.round(video.videoHeight * scale);

  offctx.fillStyle = "black";
  offctx.fillRect(0, 0, size, size);

  const dx = Math.floor((size - newW) / 2);
  const dy = Math.floor((size - newH) / 2);

  offctx.drawImage(video, dx, dy, newW, newH);

  return { scale, dx, dy };
}

startBtn.onclick = async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Starting camera...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    startBtn.remove();

    session = await ort.InferenceSession.create("model/plate.onnx", {
      executionProviders: ["wasm"],
    });

    await initOCR();
    loop();
  } catch (e) {
    console.error(e);
    showMessage("Camera or ONNX error");
  }
};

// --- Main loop ---
const INFERENCE_EVERY = 5;
const OCR_EVERY = 15;

async function loop() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (session && frameCount % INFERENCE_EVERY === 0) {
    // ===============================
    // ✅ FIXED preprocessing
    // ===============================
    const { scale, dx, dy } = letterbox(video, INPUT_SIZE);
    const img = offctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

    // CHW tensor (YOLO expects this)
    const data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    let r = 0;
    let g = INPUT_SIZE * INPUT_SIZE;
    let b = g * 2;

    for (let i = 0; i < img.length; i += 4) {
      data[r++] = img[i] / 255;
      data[g++] = img[i + 1] / 255;
      data[b++] = img[i + 2] / 255;
    }

    const input = new ort.Tensor("float32", data, [
      1,
      3,
      INPUT_SIZE,
      INPUT_SIZE,
    ]);

    try {
      const outputs = await session.run({ images: input });

      const detections = postprocess(outputs, scale, dx, dy).filter(
        (b) => b.score > 0.7 && b.w > 10 && b.h > 10,
      );
      console.log(detections);
      for (const d of detections) {
        drawBox(d);

        if (frameCount % OCR_EVERY === 0) {
          const key = `${Math.round(d.x)}_${Math.round(d.y)}_${Math.round(d.w)}_${Math.round(d.h)}`;
          if (!seenPlates.has(key)) {
            ocrPlate(d, key);
          }
        }
      }
    } catch (e) {
      console.error(e);
      showMessage("Inference error");
    }
  }

  frameCount++;
  requestAnimationFrame(loop);
}

// ===============================
// ✅ FIXED YOLOv8 postprocess
// ===============================
function postprocess(output, scale, dx, dy) {
  const tensor = output[Object.keys(output)[0]];
  const data = tensor.data;
  const num = tensor.dims[2]; // usually 8400
  const boxes = [];

  for (let i = 0; i < num; i++) {
    const cx = data[i];
    const cy = data[i + num];
    const w = data[i + num * 2];
    const h = data[i + num * 3];
    const obj = data[i + num * 4];
    const cls = data[i + num * 5]; // single-class model

    const score = obj * cls;
    if (score < CONF_THRESHOLD) continue;

    const x = (cx - w / 2 - dx) / scale;
    const y = (cy - h / 2 - dy) / scale;

    boxes.push({
      x: x * canvas.width,
      y: y * canvas.height,
      w: (w / scale) * canvas.width,
      h: (h / scale) * canvas.height,
      score,
    });
  }

  return nms(boxes);
}

// ===============================
// ✅ NMS (required for sanity)
// ===============================
function nms(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const selected = [];

  for (const box of boxes) {
    let keep = true;
    for (const s of selected) {
      if (iou(box, s) > IOU_THRESHOLD) {
        keep = false;
        break;
      }
    }
    if (keep) selected.push(box);
  }
  return selected;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union ? inter / union : 0;
}

// --- Draw bounding box ---
function drawBox(b) {
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  ctx.strokeRect(b.x, b.y, b.w, b.h);
}

// --- OCR ---
async function ocrPlate(b, key) {
  plateCtx.drawImage(video, b.x, b.y, b.w, b.h, 0, 0, OCR_SIZE.w, OCR_SIZE.h);

  try {
    const {
      data: { text },
    } = await ocrWorker.recognize(plateCanvas);
    const trimmed = text.trim();
    if (trimmed) {
      seenPlates.set(key, trimmed);
      ctx.fillStyle = "lime";
      ctx.font = "18px monospace";
      ctx.fillText(trimmed, b.x, b.y - 5);
    }
  } catch (e) {
    console.error("OCR error:", e);
  }
}
