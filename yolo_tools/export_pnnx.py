from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO('outputs/yolo8/weights/best.pt')

# Export to NCNN format
model.export(format='pytorch', opset=11, dynamic=False, nms=False, imgsz=640)  # creates '/yolo26n_ncnn_model'

model = YOLO("outputs/yolo8/weights/best_ncnn_model")
results = model("./cn_car.webp")  # Use first Vulkan device
results[0].show()