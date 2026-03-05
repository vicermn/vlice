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

## Setting Up ncnn

To use **ncnn** for YOLOv8, you need to set up the **ncnn** library and convert the YOLOv8 ONNX model to ncnn format. Follow the steps below to get started.

### Prerequisites

1. Install **CMake** (version 3.10 or above).
2. Install **Python** (version 3.6 or above).
3. Install **ONNX** and **ONNX-Simplifier**.

### Step-by-Step Instructions

1. **Clone ncnn Repository**  
   First, clone the official ncnn repository from GitHub:

   ```bash
   git clone https://github.com/Tencent/ncnn
   cd ncnn
   ```

2. **Build ncnn**
   Follow the build instructions in the ncnn repository to compile it for your system. You can use CMake for the build process:

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   sudo make install
   ```

   You can find more detailed instructions for different platforms (Linux, macOS, Windows) in the [ncnn README](https://github.com/Tencent/ncnn).

3. **Prepare YOLOv8 ONNX Model**
   Export your YOLOv8 model to ONNX format using the existing tools in the `onnx_tools` folder.

4. **Simplify the ONNX Model**
   Before converting the ONNX model to ncnn format, it's recommended to run the **onnx-simplifier** to simplify the model:

   ```bash
   python3 -m onnxsim <input_model.onnx> <output_model.onnx>
   ```

5. **Convert ONNX to ncnn**
   Once your model is simplified, use the **onnx2ncnn** tool to convert the ONNX model to ncnn format. This can be done via the following command:

   ```bash
   onnx2ncnn <input_model.onnx> <output_model.param> <output_model.bin>
   ```

   - `input_model.onnx`: Your simplified YOLOv8 ONNX model.
   - `output_model.param`: The output param file.
   - `output_model.bin`: The output bin file.

6. **Debugging Failures**
   If you encounter any issues during the conversion, the debugging process might involve:
   - Verifying the ONNX model's integrity using tools like `onnx.checker.check_model`.
   - Checking the ncnn and ONNX model versions for compatibility.
   - Reviewing the ncnn logs for errors related to unsupported layers or operations.

   Good luck debugging if issues arise—ncnn's conversion process can be tricky, especially if layers in the model are not fully supported.

For more information, refer to the official ncnn GitHub repository: [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn).

## TODO

- Add Camera toggle
- Fine Tune Paddlev5 model with license plates
