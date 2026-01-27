document.addEventListener("DOMContentLoaded", async () => {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    let yoloSession = null;
    let ocrSession = null;

    const CONF_THRESHOLD = 0.3;
    const MODEL_SIZE = 640;

    // Character set from fast-plate-ocr YAML
    const OCR_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const PAD_CHAR = "_";
    const audioContext = new (
    window.AudioContext || window.webkitAudioContext
    )();
    const oscillator = audioContext.createOscillator();
    oscillator.type = "sine"; // Choose the type of wave (sine, square, etc.)
    oscillator.frequency.setValueAtTime(440, audioContext.currentTime); // Frequency in Hz (440 Hz is the A note)
    oscillator.connect(audioContext.destination);
    // Global array for CSV column data
    let plates = new Set();

    /* =========================
    PLATE TEXT NORMALIZATION
    ========================= */

    function normalizePlateText(raw) {
    return raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
    }

    function formatLicense(text) {
    const t = normalizePlateText(text);
    if (t.length >= 7) return t.slice(0, 7);
    if (t.length >= 6) return t.slice(0, 6);
    return t;
    }

    /* =========================
    INIT
    ========================= */

    async function init() {
    yoloSession = await ort.InferenceSession.create(
        "model/hotness.onnx",
        { executionProviders: ["wasm"] },
    );
    console.log("YOLO loaded");

    ocrSession = await ort.InferenceSession.create("ocr/cct.onnx", {
        executionProviders: ["wasm"],
    });
    console.log(ocrSession.inputNames);

    console.log("fast-plate-ocr loaded");
    }

    /* =========================
    YOLO FUNCTIONS — UNTOUCHED
    ========================= */

    function imageToTensor(img, size = MODEL_SIZE) {
    const off = document.createElement("canvas");
    off.width = off.height = size;
    const ctxOff = off.getContext("2d");
    ctxOff.fillStyle = "black";
    ctxOff.fillRect(0, 0, size, size);

    const scale = Math.min(size / img.width, size / img.height);
    const newW = img.width * scale;
    const newH = img.height * scale;
    const dx = (size - newW) / 2;
    const dy = (size - newH) / 2;
    ctxOff.drawImage(img, dx, dy, newW, newH);

    const imgData = ctxOff.getImageData(0, 0, size, size).data;
    const data = new Float32Array(3 * size * size);
    let r = 0,
        g = size * size,
        b = 2 * size * size;
    for (let i = 0; i < imgData.length; i += 4) {
        data[r++] = imgData[i] / 255;
        data[g++] = imgData[i + 1] / 255;
        data[b++] = imgData[i + 2] / 255;
    }
    return {
        tensor: new ort.Tensor("float32", data, [1, 3, size, size]),
        scale,
        dx,
        dy,
    };
    }

    function mapBoxes(boxes, scale, dx, dy, imgW, imgH) {
    const results = [];
    for (let i = 0; i < boxes.length / 6; i++) {
        const o = i * 6;
        let x1 = boxes[o],
        y1 = boxes[o + 1],
        x2 = boxes[o + 2],
        y2 = boxes[o + 3];
        const score = boxes[o + 4];
        if (!score || score < CONF_THRESHOLD) continue;

        x1 = Math.max(0, (x1 - dx) / scale);
        y1 = Math.max(0, (y1 - dy) / scale);
        x2 = Math.min(imgW, (x2 - dx) / scale);
        y2 = Math.min(imgH, (y2 - dy) / scale);
        results.push({ x1, y1, x2, y2, score });
    }
    return results;
    }

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

    /* =========================
    OCR HELPERS — UPDATED FOR fast-plate-ocr
    ========================= */

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

    function ctcDecode(logits) {
    const T = logits.length / (OCR_CHARS.length + 1); // +1 for padding
    let last = -1,
        text = "";
    for (let t = 0; t < T; t++) {
        let max = -Infinity,
        idx = 0;
        for (let c = 0; c < OCR_CHARS.length + 1; c++) {
        // include pad
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

    function decodePlateSlots(logits) {
    // logits shape: [num_slots * num_classes], flattened
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

    async function recognizePlate(plateCanvas) {
    // Only simple grayscale variants are needed
    const variants = [plateCanvas];

    const results = [];
    for (const v of variants) {
        const input = tensorFromPlate(v);
        const out = await ocrSession.run({ input: input });
        const logits = out[Object.keys(out)[0]].data;
        results.push(normalizePlateText(decodePlateSlots(logits)));
    }

    // Voting is overkill for this model, but kept for robustness
    return formatLicense(results[0]);
    }

    /* =========================
    MAIN DETECTION LOOP
    ========================= */

    async function detectImage(img) {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const { tensor, scale, dx, dy } = imageToTensor(img, MODEL_SIZE);
    const output = await yoloSession.run({ images: tensor });
    const boxes = mapBoxes(
        output.output0.data,
        scale,
        dx,
        dy,
        img.width,
        img.height,
    );

    // Clear existing plate text list
    const plateList = document.getElementById("plateList");
    plateList.innerHTML = "";

    for (const b of boxes) {
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

        const plateCanvas = preprocessPlate(img, b.x1, b.y1, b.x2, b.y2);
        const plateText = await recognizePlate(plateCanvas);
        if (plates.has(plateText)) {
        // Add detected plate text to the list
        const listItem = document.createElement("li");
        listItem.textContent = plateText;
        listItem.classList.add(
            "text-lg",
            "font-semibold",
            "text-red-600",
            "bg-white",
            "p-2",
            "rounded-md",
            "shadow-md",
            "hover:bg-red-50",
        );

        plateList.appendChild(listItem);
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 1); // Play for 1 second
        }
    }
    }

    // File upload listener for image
    document
    .getElementById("imgUpload")
    .addEventListener("change", async (e) => {
        if (!yoloSession || !ocrSession) await init();
        const file = e.target.files[0];
        if (!file) return;
        const img = new Image();
        img.onload = () => detectImage(img);
        img.src = URL.createObjectURL(file);
    });

    // File upload listener for CSV
    document
    .getElementById("csvUpload")
    .addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            const csvData = event.target.result;
            const rows = csvData.split("\n");

            // Extract the header row to find the column index
            const header = rows[0].split(",").map((col) => col.trim()); // Assuming first row is the header
            const targetColumnName = "vehicle_license";
            const targetColumnIndex = header.indexOf(targetColumnName);

            if (targetColumnIndex === -1) {
            console.error(`Column "${targetColumnName}" not found.`);
            return;
            }

            // Extract data from the target column (ignoring the header row)
            plates = new Set(
            rows.slice(1).map((row) => {
                const columns = row.split(",").map((col) => col.trim());
                return columns[targetColumnIndex]; // Extract the value from the target column
            }),
            );

            console.log(plates); // View the contents of the array
        };
        reader.readAsText(file);
        }
    });
});
