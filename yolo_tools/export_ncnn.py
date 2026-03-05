from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO('outputs/yolo8/weights/best.pt')

# Export to NCNN format
model.export(format='ncnn', imgsz=640, simplify=True, dynamic=False, opset=11)  # creates '/yolo26n_ncnn_model'

model = YOLO("outputs/yolo8/weights/best_ncnn_model")
results = model("./mnplate.webp")  # Use first Vulkan device
results[0].show()