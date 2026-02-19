import { preprocessPlate, recognizePlate } from "./ocr.js";

(function () {
  document.addEventListener("DOMContentLoaded", initializeDetection);
  let plates = new Set();
  const Module = (window.Module = window.Module || {});
  const CONF_THRESHOLD = 0.25;
  let has_simd, has_threads;
  let wasmModuleLoaded = false;
  let isStreaming = false;
  let ocrSession = null;
  function initializeDetection() {
    const audioContext = new (
      window.AudioContext || window.webkitAudioContext
    )();

    // Handle WASM module initialization
    Module.onRuntimeInitialized = async function () {
      wasmModuleLoaded = true;
      _yolo_init();
      initializeStream(); // Directly initialize the stream once WASM is ready
      ocrSession = await ort.InferenceSession.create("static/model/cct.onnx", {
        executionProviders: ["wasm"],
      });
    };

    // Detect SIMD and threads support
    wasmFeatureDetect.simd().then((simdSupported) => {
      has_simd = simdSupported;
      console.log("SIMD supported:", has_simd);

      wasmFeatureDetect.threads().then((threadsSupported) => {
        has_threads = threadsSupported;
        console.log("Threads supported:", threadsSupported);
        loadYoloModel();
      });
    });

    // Load YOLO model files using the new function
    function loadYoloModel() {
      const yolo_module_name = "static/model/yolo";
      const yolowasm = `${yolo_module_name}.wasm`;
      const yolojs = `${yolo_module_name}.js`;

      // Locate the model files
      Module.locateFile = (path) => {
        const modelDir = yolo_module_name.substring(
          0,
          yolo_module_name.lastIndexOf("/") + 1,
        );
        return modelDir + path;
      };

      loadWasmAndScript(yolowasm, yolojs);
    }

    // Refactored WASM and JS loading function
    function loadWasmAndScript(wasmUrl, jsUrl) {
      fetch(wasmUrl)
        .then((response) => response.arrayBuffer())
        .then((buffer) => {
          Module.wasmBinary = buffer;
          const script = document.createElement("script");
          script.src = jsUrl;
          script.onload = () => {
            console.log(`Script loaded: ${jsUrl}`);
          };
          document.body.appendChild(script);
        })
        .catch((error) =>
          console.error("Error loading WASM or script:", error),
        );
    }

    // Initialize camera stream
    let dst = null,
      resultarray = null,
      resultbuffer = null;

    function initializeStream() {
      const video = document.getElementById("video");
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      const constraints = { audio: false, video: { width: 320, height: 320 } };

      navigator.mediaDevices
        .getUserMedia(constraints)
        .then((mediaStream) => {
          video.srcObject = mediaStream;
          video.onloadedmetadata = () => video.play();
          toggleCameraModal(true);
        })
        .catch((err) => {
          console.log(err.message);
          toggleCameraModal(false);
        });

      video.addEventListener("canplay", () => {
        if (!isStreaming) {
          canvas.setAttribute("width", 320);
          canvas.setAttribute("height", 320);
          isStreaming = true;
        }
      });

      video.addEventListener("play", () => {
        // Call malloc only once when the video starts playing
        if (!dst && wasmModuleLoaded) {
          mallocAndCallSFilter();
        }
      });
    }

    // Allocate memory only once during initialization
    function mallocAndCallSFilter() {
      dst = _malloc(320 * 320 * 4); // Allocate memory for image buffer
      resultarray = new Float32Array(6 * 20); // Initialize result array for detections
      resultbuffer = _malloc(6 * 20 * Float32Array.BYTES_PER_ELEMENT); // Allocate memory for result buffer

      HEAPF32.set(resultarray, resultbuffer / Float32Array.BYTES_PER_ELEMENT);
      sFilter();
    }

    // Process frames
    function sFilter() {
      const video = document.getElementById("video");
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");

      if (video.paused || video.ended) return;
      ctx.fillRect(0, 0, 320, 320);
      ctx.drawImage(video, 0, 0, 320, 320);
      ncnn_yolo(ctx, canvas);

      window.requestAnimationFrame(sFilter);
    }

    function playBeep() {
      const osc = audioContext.createOscillator();
      osc.type = "sine";
      osc.frequency.setValueAtTime(440, audioContext.currentTime);
      osc.connect(audioContext.destination);
      osc.start();
      osc.stop(audioContext.currentTime + 0.5);
    }

    // YOLO object detection
    async function ncnn_yolo(ctx, canvas) {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      HEAPU8.set(imageData.data, dst); // Copy image data to the allocated memory

      _process_frame(
        dst,
        canvas.width,
        canvas.height,
        CONF_THRESHOLD,
        resultbuffer,
      );

      const qaqarray = HEAPF32.subarray(
        resultbuffer / Float32Array.BYTES_PER_ELEMENT,
        resultbuffer / Float32Array.BYTES_PER_ELEMENT + 6 * 20,
      );

      const plateList = document.getElementById("plateList");
      plateList.innerHTML = "";

      // Draw bounding boxes for detected objects
      for (let i = 0; i < 20; i++) {
        const [label, prob, bbox_x, bbox_y, bbox_w, bbox_h] = qaqarray.slice(
          i * 6,
          i * 6 + 6,
        );

        if (label < 0 || prob < CONF_THRESHOLD) continue;

        // Determine the color of the bounding box based on label and confidence
        let colorBase;
        if (prob >= 0.6) {
          colorBase = "rgba(34, 197, 94, 1)"; // Green for prob >= 60%
        } else if (prob >= 0.4) {
          colorBase = "rgba(255, 165, 0, 1)"; // Orange for 40% <= prob < 60%
        } else {
          colorBase = "rgba(239, 68, 68, 1)"; // Red for prob < 40%
        }

        // Draw only the border with the color based on the confidence
        ctx.lineWidth = 4;
        ctx.strokeStyle = colorBase;
        ctx.strokeRect(bbox_x, bbox_y, bbox_w, bbox_h);

        // If the object is a license plate (assuming label for plates is a specific value, e.g., '2')
        if (prob >= CONF_THRESHOLD) {
          // Preprocess the plate region and recognize the plate
          let plateCanvas = preprocessPlate(
            canvas,
            bbox_x,
            bbox_y,
            bbox_x + bbox_w,
            bbox_y + bbox_h,
          );

          // Wait for the plate text to be recognized
          const plateText = await recognizePlate(plateCanvas, ocrSession);

          // Release the canvas reference
          plateCanvas.width = 0;
          plateCanvas.height = 0;
          plateCanvas = null;

          // If the plate text is valid and hasn't been processed before, display it
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

            // Play a beep sound (assumed to be a function you have)
            playBeep();
          }
        }
      }
    }

    // Toggle camera error modal
    function toggleCameraModal(success) {
      const modal = document.getElementById("camera-error-modal");
      if (!modal) return;
      modal.classList.toggle("hidden", success);
    }
  }
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
})();

!(function (e, a) {
  "object" == typeof exports && "undefined" != typeof module
    ? (module.exports = a())
    : "function" == typeof define && define.amd
      ? define(a)
      : ((e = e || self).wasmFeatureDetect = a());
})(this, function () {
  "use strict";
  return {
    bigInt: () =>
      (async (e) => {
        try {
          return (
            (await WebAssembly.instantiate(e)).instance.exports.b(BigInt(0)) ===
            BigInt(0)
          );
        } catch (e) {
          return !1;
        }
      })(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 126, 1, 126, 3, 2, 1, 0,
          7, 5, 1, 1, 98, 0, 0, 10, 6, 1, 4, 0, 32, 0, 11,
        ]),
      ),
    bulkMemory: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 3, 1,
          0, 1, 10, 14, 1, 12, 0, 65, 0, 65, 0, 65, 0, 252, 10, 0, 0, 11,
        ]),
      ),
    exceptions: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 9, 1,
          7, 0, 6, 64, 7, 26, 11, 11,
        ]),
      ),
    multiValue: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 0, 2, 127, 127, 3, 2, 1, 0,
          10, 8, 1, 6, 0, 65, 0, 65, 0, 11,
        ]),
      ),
    mutableGlobals: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 2, 8, 1, 1, 97, 1, 98, 3, 127, 1, 6, 6,
          1, 127, 1, 65, 0, 11, 7, 5, 1, 1, 97, 3, 1,
        ]),
      ),
    referenceTypes: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 7, 1,
          5, 0, 208, 112, 26, 11,
        ]),
      ),
    saturatedFloatToInt: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 12, 1,
          10, 0, 67, 0, 0, 0, 0, 252, 0, 26, 11,
        ]),
      ),
    signExtensions: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 8, 1,
          6, 0, 65, 0, 192, 26, 11,
        ]),
      ),
    simd: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 9, 1,
          7, 0, 65, 0, 253, 15, 26, 11,
        ]),
      ),
    tailCall: async () =>
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 6, 1,
          4, 0, 18, 0, 11,
        ]),
      ),
    threads: () =>
      (async (e) => {
        try {
          return (
            new ("undefined" != typeof MessageChannel
              ? MessageChannel
              : await import("worker_threads").then(
                  (e) => e.MessageChannel,
                ))().port1.postMessage(new SharedArrayBuffer(1)),
            WebAssembly.validate(e)
          );
        } catch (e) {
          return !1;
        }
      })(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 4, 1,
          3, 1, 1, 10, 11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11,
        ]),
      ),
  };
});
