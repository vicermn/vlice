# License Plate Detection + OCR

This project now leverages **ncnn** for the YOLOv8 portion, providing improved **speed** and **memory management**. The previous YOLOv8 ONNX model has been retired and moved to the `onnx_project` folder for archival purposes. The new **ncnn-based YOLOv8** implementation enhances both performance and resource efficiency, making it a better choice for web applications.

## How to Run

1. Open this project folder in **VS Code**.
2. Ensure you have the **Live Server** extension installed.
3. Open the `index.html` file in VS Code.
4. Right-click on `index.html` and select **"Open with Live Server"**.
5. Your browser will open the page.

## Using the App

1. Click the **"Choose File"** button to upload an image of a car with a visible license plate.
2. The app will automatically:
   - Detect the license plate using **YOLOv8 with ncnn**.
   - Recognize the plate text using **fast-plate-ocr**.
3. The detected plate will be highlighted with a rectangle, and the recognized text will appear above it.

## Notes

- The OCR model supports **Latin-alphabet plates** and outputs up to **9 characters**.
- For best results, ensure the plate is clearly visible.

## TODO

- Explore **OCR model** compatible with ncnn and specifically trained for **license plate recognition**. We can either train one ourselves or look for an existing model.
