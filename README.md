# License Plate Detection + OCR

This project now leverages **ncnn** for the YOLOv8 portion, providing improved **speed** and **memory management**. The previous YOLOv8 ONNX model and fast has been retired and moved to the `onnx_project` folder for archival purposes. The new **ncnn-based YOLOv8** implementation enhances both performance and resource efficiency, making it a better choice for web applications.

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
   - Recognize the plate text using **PaddleOCRv5**.
3. The detected plate will be highlighted with a rectangle, and the recognized text will appear above it.

## Notes

- The OCR model supports **Latin-alphabet plates** and outputs up to **8 characters**.
- For best results, ensure the plate is clearly visible.

It looks like the sandbox file link failed on upload. I regenerated the file. Try this download instead:

**Download the Markdown file:**
[setup_ncnn_yolo_paddleocr.md](sandbox:/mnt/data/setup_ncnn_yolo_paddleocr.md)

If it still fails, you can also copy the full Markdown directly below and save it as
`setup_ncnn_yolo_paddleocr.md`.

---

# ncnn Setup and Model Conversion Guide

## Overview

This guide explains how to:

1. Build **ncnn** from source.
2. Convert **YOLOv8** models to ncnn.
3. Convert **PaddleOCR** models to ncnn.

The **ncnn setup is shared** by both workflows. After ncnn is built, the process splits depending on the model you want to run.

---

# 1. Install and Build ncnn

First, clone and build the official ncnn repository.

```bash
git clone https://github.com/Tencent/ncnn
cd ncnn
```

## Build ncnn

```bash
mkdir build
cd build

cmake ..
make -j$(nproc)

sudo make install
```

This installs tools including:

- `onnx2ncnn`
- `pnnx`

These are required for converting models.

For platform-specific instructions see the official repository:

[https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn)

---

# 2. YOLOv8 → ncnn Workflow

After ncnn is built, follow these steps to convert YOLOv8.

## Step 1 — Export YOLOv8 to ONNX

Export your trained YOLOv8 model using Ultralytics:

```bash
yolo export model=best.pt format=onnx
```

This generates:

```
best.onnx
```

---

## Step 2 — Simplify the ONNX Model

Install the ONNX simplifier if needed:

```bash
pip install onnxsim
```

Then simplify the model:

```bash
python3 -m onnxsim best.onnx best-sim.onnx
```

---

## Step 3 — Convert ONNX → ncnn

Use the ncnn conversion tool:

```bash
onnx2ncnn best-sim.onnx best.param best.bin
```

Outputs:

```
best.param
best.bin
```

These are the model files used by ncnn for inference.

---

## Step 4 — Debugging Conversion Failures

Common debugging steps:

- Verify the ONNX model:

```python
import onnx
onnx.checker.check_model("best.onnx")
```

- Ensure your **ncnn version is recent**.
- Inspect logs for **unsupported operators**.
- Simplify the ONNX model again with `onnxsim`.

Notes:

- YOLO versions **past v8 can be problematic**.
- **v11** may work but can require extra fixes.
- **v12 currently has many compatibility issues** with ncnn.

Example implementations can be found in the ncnn examples:

[https://github.com/Tencent/ncnn/tree/master/examples](https://github.com/Tencent/ncnn/tree/master/examples)

---

# 3. PaddleOCR → ncnn Workflow

PaddleOCR conversion follows the same process used in the official example:

[https://github.com/Tencent/ncnn/blob/master/examples/ppocrv5.cpp](https://github.com/Tencent/ncnn/blob/master/examples/ppocrv5.cpp)

Unlike YOLO, **PaddleOCR does not require building ncnn to export the model**, but **ncnn is required for inference**.

---

## Step 1 — Install Dependencies

```bash
pip install paddlepaddle==3.0.0
pip install paddleocr==3.0.0
```

Install the ONNX conversion tool:

```bash
paddlex --install paddle2onnx
```

---

## Step 2 — Test PaddleOCR

Run a quick OCR test:

```bash
paddleocr ocr -i test.png
```

This also downloads the official models to:

```
~/.paddlex/official_models/
```

---

## Step 3 — Convert Paddle Model → ONNX

Example conversion:

```bash
paddlex --paddle2onnx \
  --paddle_model_dir ~/.paddlex/official_models/PP-OCRv5_mobile_rec \
  --onnx_model_dir PP-OCRv5_mobile_rec
```

This produces:

```
PP-OCRv5_mobile_rec.onnx
```

---

## Step 4 — Convert ONNX → ncnn (using pnnx)

Use the `pnnx` tool from ncnn:

```bash
pnnx PP-OCRv5_mobile_rec.onnx \
  inputshape=[1,3,48,160] \
  inputshape2=[1,3,48,256]
```

This generates:

```
PP-OCRv5_mobile_rec.ncnn.param
PP-OCRv5_mobile_rec.ncnn.bin
```

These files can now be used for **ncnn inference**.

---

# Summary

### Shared Step

Build **ncnn** once.

### Workflow Split

**YOLOv8**

```
YOLOv8 → ONNX → onnxsim → onnx2ncnn → ncnn inference
```

**PaddleOCR**

```
Paddle Model → paddle2onnx → pnnx → ncnn inference
```

Good luck debugging if issues arise—ncnn's conversion process can be tricky, especially if layers in the model are not fully supported.

For more information, refer to the official ncnn GitHub repository: [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn).

## TODO

- Add Camera toggle
- Fine Tune Paddlev5 model with license plates
