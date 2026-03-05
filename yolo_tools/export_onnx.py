from ultralytics import YOLO

# Load the trained model
model = YOLO('outputs/yolo8/weights/best.pt')

# Export the model to ONNX
model.export(format='onnx', opset=12, simplify=True, dynamic=False, batch=1, imgsz=320, nms=False)  # You can change imgsz according to your input resolution


# Load the exported ONNX model
onnx_model = YOLO("outputs/yolo8/weights/best.onnx")
# Run inference
results = onnx_model("./cn_car.webp")
results[0].show()