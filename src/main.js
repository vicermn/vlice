// Main.js
import { imageToTensor, mapBoxes } from "./yolo.js"; // Import OCR functions
import { preprocessPlate, recognizePlate } from "./ocr.js"; // Import OCR functions

document.addEventListener("DOMContentLoaded", async () => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  let yoloSession = null;
  let ocrSession = null;

  const MODEL_SIZE = 640;
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const oscillator = audioContext.createOscillator();
  oscillator.type = "sine"; // Choose the type of wave (sine, square, etc.)
  oscillator.frequency.setValueAtTime(440, audioContext.currentTime); // Frequency in Hz (440 Hz is the A note)
  oscillator.connect(audioContext.destination);
  let plates = new Set();

  /* =========================
    INIT
    ========================= */
  async function init() {
    yoloSession = await ort.InferenceSession.create("src/model/yolo.onnx", {
      executionProviders: ["wasm"],
    });
    console.log("YOLO loaded");

    ocrSession = await ort.InferenceSession.create("src/model//cct.onnx", {
      executionProviders: ["wasm"],
    });
    console.log("fast-plate-ocr loaded");
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
      const plateText = await recognizePlate(plateCanvas, ocrSession);
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
  document.getElementById("imgUpload").addEventListener("change", async (e) => {
    if (!yoloSession || !ocrSession) await init();
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => detectImage(img);
    img.src = URL.createObjectURL(file);
  });

  // File upload listener for CSV
  document.getElementById("csvUpload").addEventListener("change", async (e) => {
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
