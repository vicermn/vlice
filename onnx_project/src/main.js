// Main.js
import { imageToTensor, mapBoxes } from "./yolo.js";
import { preprocessPlate, recognizePlate } from "./ocr.js";

document.addEventListener("DOMContentLoaded", async () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  let yoloSession = null;
  let ocrSession = null;

  const MODEL_SIZE = 320;
  let plates = new Set();

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();

  let lastFrameTime = 0;
  const frameInterval = 1000 / 10; // 10 FPS = 1000ms / 10

  /* =========================
     INIT
  ========================= */
  async function init() {
    yoloSession = await ort.InferenceSession.create("src/model/yolo320.onnx", {
      executionProviders: ["wasm"],
    });
    console.log("YOLO loaded");

    ocrSession = await ort.InferenceSession.create("src/model/cct.onnx", {
      executionProviders: ["wasm"],
    });
    console.log("fast-plate-ocr loaded");

    startWebcam();
  }

  /* =========================
     START WEBCAM
  ========================= */
  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: 640, // Lower resolution for performance
          height: 480,
        },
        audio: false,
      });
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        detectLoop();
      };
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  }

  /* =========================
     DETECTION LOOP
  ========================= */
  async function detectLoop() {
    const now = performance.now();
    const elapsedTime = now - lastFrameTime;

    if (elapsedTime < frameInterval) {
      // If the elapsed time is less than the interval (10 FPS), skip this frame
      requestAnimationFrame(detectLoop);
      return;
    }

    // Update last frame time
    lastFrameTime = now;

    // Draw video frame onto canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Use an offscreen canvas for processing
    let offCanvas = document.createElement("canvas");
    offCanvas.width = video.videoWidth;
    offCanvas.height = video.videoHeight;
    const offCtx = offCanvas.getContext("2d");
    offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);

    // Convert the frame to tensor
    const { tensor, scale, dx, dy } = imageToTensor(offCanvas, MODEL_SIZE);
    let output;
    let boxes;
    try {
      // Run the model
      output = await yoloSession.run({ images: tensor });

      // Process the output (boxes)
      boxes = mapBoxes(
        output.output0.data,
        scale,
        dx,
        dy,
        canvas.width,
        canvas.height,
      );
    } finally {
      // Always cleanup
      tensor.dispose();
      if (output) {
        for (const key in output) {
          output[key]?.dispose?.();
        }
      }
    }

    // Clear canvas overlay and redraw video
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Prepare the list of plates (if any)
    const plateList = document.getElementById("plateList");
    plateList.innerHTML = "";

    for (const b of boxes) {
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

      let plateCanvas = preprocessPlate(offCanvas, b.x1, b.y1, b.x2, b.y2);

      const plateText = await recognizePlate(plateCanvas, ocrSession);

      plateCanvas.width = 0;
      plateCanvas.height = 0;
      plateCanvas = null; // Explicitly release the canvas reference
      if (plates.has(plateText)) {
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

        playBeep();
      }
    }
    offCanvas.width = 0;
    offCanvas.height = 0;
    offCanvas = null; // Explicitly release the canvas reference
    // Request the next frame after processing
    requestAnimationFrame(detectLoop);
  }

  function playBeep() {
    const osc = audioContext.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(440, audioContext.currentTime);
    osc.connect(audioContext.destination);
    osc.start();
    osc.stop(audioContext.currentTime + 0.5);
  }

  /* =========================
     CSV UPLOAD
  ========================= */
  document.getElementById("csvUpload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (event) {
        const csvData = event.target.result;
        const rows = csvData.split("\n");

        const header = rows[0].split(",").map((col) => col.trim());
        const targetColumnName = "vehicle_license";
        const targetColumnIndex = header.indexOf(targetColumnName);

        if (targetColumnIndex === -1) {
          console.error(`Column "${targetColumnName}" not found.`);
          return;
        }

        plates = new Set(
          rows.slice(1).map((row) => {
            const columns = row.split(",").map((col) => col.trim());
            return columns[targetColumnIndex];
          }),
        );

        console.log(plates);
      };
      reader.readAsText(file);
    }
  });

  // Initialize sessions
  await init();
});
