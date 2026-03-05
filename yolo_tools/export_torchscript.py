from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO('outputs/yolo11/yolo_11_custom_model/weights/best.pt')

# Export to NCNN format
model.export(format="torchscript")  # creates '/yolo26n_ncnn_model'

model = YOLO("outputs/yolo11/yolo_11_custom_model/weights/best.torchscript")
results = model("./cn_car.webp")  # Use first Vulkan device
results[0].show()