from ultralytics import YOLO

# Load the trained model
model = YOLO('outputs/yolo_custom_model/weights/best.pt')

# Export the model to ONNX
model.export(format='tfjs', imgsz=640, nms=True, batch=1, simplify=True)  # You can change imgsz according to your input resolution


# Load the exported ONNX model
onnx_model = YOLO("outputs/yolo_custom_model/weights/best.tfjs")

# Run inference
results = onnx_model("./cn_car.webp")
results[0].show()