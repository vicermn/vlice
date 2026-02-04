# License Plate Detection + OCR

This project uses **YOLOv8** for license plate detection and **fast-plate-ocr** for reading the plate text in the browser via ONNX.

## How to Run

1. Open this project folder in **VS Code**.
2. Make sure you have the **Live Server** extension installed.
3. Open the `index.html` file in VS Code.
4. Right-click on `index.html` and select **"Open with Live Server"**.
5. Your browser will open the page.

## Using the App

1. Click the **"Choose File"** button to upload an image of a car with a visible license plate.
2. The app will automatically:
   - Detect the license plate using YOLOv8.
   - Recognize the plate text using fast-plate-ocr.
3. The detected plate will be highlighted with a rectangle and the recognized text will appear above it.

## Notes

- The OCR model supports Latin-alphabet plates and outputs up to **9 characters**.
- Make sure the plate is clearly visible for best results.
- The model files (`hotness.onnx` and `cct.onnx`) must be in the `model/` and `ocr/` folders respectively.

## TODO

- ✅ Reduce YOLO input to 416×416 (or even 320×320 if plates are big enough).
- [ ] Process fewer frames (10–15 fps).
- [ ] Dispose tensors immediately after use.
- [ ] Reuse canvases instead of creating new ones every frame.
- [ ] Use WASM backend, not WebGL/WebGPU.
- [ ] Optional: crop the image to ROI to save even more memory.
