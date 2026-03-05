import onnx
from onnx import version_converter

# Load your original model
model_path = 'outputs/yolo_custom_model/weights/best.onnx'
model = onnx.load(model_path)
print(f"Loaded Opset Version: {model.opset_import[0].version}")
opset_ver = 17
# Convert the model to Opset version 21
model_converted = version_converter.convert_version(model, opset_ver)

# Save the converted model
onnx.save(model_converted, "yolov26.onnx")

model_path = f"yolov26.onnx"
model = onnx.load(model_path)
print(f"Saved Opset Version: {model.opset_import[0].version}")