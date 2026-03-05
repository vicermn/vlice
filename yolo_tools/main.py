from ultralytics import YOLO

# Load your pre-trained model or start from scratch
# model = YOLO('yolo26n.pt')  # Load pre-trained YOLOv6 model or your custom model

# # Train the model
# model.train(
#     data='dataset/yolo26/data.yaml',  # Custom dataset YAML
#     epochs=50,  # Number of epochs
#     batch=16,  # Corrected argument for batch size
#     imgsz=640,  # Image size
#     project='outputs',  # Output directory
#     name='yolo_custom_model',  # Model name
#     device='0'  # Set device (use '0' for GPU or 'cpu')
# )

# model = YOLO('yolo11n.pt')  # Load pre-trained YOLOv6 model or your custom model

# # Train the model
# model.train(
#     data='dataset/yolo11/data.yaml',  # Custom dataset YAML
#     epochs=50,  # Number of epochs
#     batch=16,  # Corrected argument for batch size
#     imgsz=640,  # Image size
#     project='outputs/yolo11',  # Output directory
#     name='yolo_11_custom_model',  # Model name
#     device='0'  # Set device (use '0' for GPU or 'cpu')
# )

model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv6 model or your custom model

# Train the model
model.train(
    data='dataset/yolo8/data.yaml',  # Custom dataset YAML
    epochs=50,  # Number of epochs
    batch=16,  # Corrected argument for batch size
    imgsz=640,  # Image size
    project='outputs/yolo8',  # Output directory
    name='yolo_8_custom_model',  # Model name
    device='0'  # Set device (use '0' for GPU or 'cpu')
)
