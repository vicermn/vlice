// OCR.js

const OCR_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const PAD_CHAR = "_";

function preprocessPlate(img, x1, y1, x2, y2) {
  const plateCanvas = document.createElement("canvas");
  plateCanvas.width = 128; // model width
  plateCanvas.height = 64; // model height

  const ctx = plateCanvas.getContext("2d");

  const plateW = x2 - x1;
  const plateH = y2 - y1;

  // Optional: add margin around plate
  const marginX = Math.round(plateW * 0.1);
  const marginY = Math.round(plateH * 0.1);
  x1 = Math.max(0, x1 - marginX);
  y1 = Math.max(0, y1 - marginY);
  x2 = Math.min(img.width, x2 + marginX);
  y2 = Math.min(img.height, y2 + marginY);

  ctx.drawImage(
    img,
    x1,
    y1,
    x2 - x1,
    y2 - y1,
    0,
    0,
    plateCanvas.width,
    plateCanvas.height,
  );

  return plateCanvas;
}

// Function to normalize and clean the plate text
function normalizePlateText(raw) {
  return raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
}

function formatLicense(text) {
  console.log(text);
  const t = normalizePlateText(text);
  if (t.length >= 7) return t.slice(0, 7);
  if (t.length >= 6) return t.slice(0, 6);
  return t;
}

// Converts a canvas image to a tensor suitable for OCR
function tensorFromPlate(canvas) {
  const W = canvas.width; // 128
  const H = canvas.height; // 64
  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, W, H).data;

  // NHWC uint8
  const data = new Uint8Array(H * W * 3);
  let ptr = 0;
  for (let i = 0; i < imgData.length; i += 4) {
    data[ptr++] = imgData[i]; // R
    data[ptr++] = imgData[i + 1]; // G
    data[ptr++] = imgData[i + 2]; // B
  }

  return new ort.Tensor("uint8", data, [1, H, W, 3]); // NHWC
}

// Decode the logits from OCR output
function ctcDecode(logits) {
  const T = logits.length / (OCR_CHARS.length + 1); // +1 for padding
  let last = -1,
    text = "";
  for (let t = 0; t < T; t++) {
    let max = -Infinity,
      idx = 0;
    for (let c = 0; c < OCR_CHARS.length + 1; c++) {
      const v = logits[t * (OCR_CHARS.length + 1) + c];
      if (v > max) {
        max = v;
        idx = c;
      }
    }
    // ignore padding
    if (idx !== last && idx < OCR_CHARS.length) {
      text += OCR_CHARS[idx];
    }
    last = idx;
  }
  return text;
}

// Decode the plate from the output slots of OCR
function decodePlateSlots(logits) {
  const numSlots = 9; // from YAML
  const numClasses = OCR_CHARS.length + 1; // +1 for pad '_'
  let plate = "";

  for (let i = 0; i < numSlots; i++) {
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const v = logits[i * numClasses + c];
      if (v > maxVal) {
        maxVal = v;
        maxIdx = c;
      }
    }
    if (maxIdx < OCR_CHARS.length) {
      plate += OCR_CHARS[maxIdx];
    }
  }

  return plate;
}

// Recognize the license plate text from the plate canvas
async function recognizePlate(plateCanvas, ocrSession) {
  // Only simple grayscale variants are needed
  const variants = [plateCanvas];

  const results = [];
  for (const v of variants) {
    const input = tensorFromPlate(v);
    const out = await ocrSession.run({ input: input });
    input.dispose();
    const logits = out[Object.keys(out)[0]].data;
    if (out) {
      for (const key in out) {
        out[key]?.dispose?.();
      }
    }
    results.push(normalizePlateText(decodePlateSlots(logits)));
  }

  // Voting is overkill for this model, but kept for robustness
  return formatLicense(results[0]);
}

export {
  recognizePlate,
  normalizePlateText,
  formatLicense,
  tensorFromPlate,
  preprocessPlate,
};
