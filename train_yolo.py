from ultralytics import YOLO
import torch

DATASET = "stamp_data.yaml"
MODEL = "yolov8n.pt"

EPOCHS = 50
IMAGE_SIZE = 416
BATCH_SIZE = 16

if torch.cuda.is_available():
    device = 0
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA GPU not available. Using CPU.")

model = YOLO(MODEL)

model.train(
    data=DATASET,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=device,
    project="runs/detect",
    name="yolov8_stamp",
    pretrained=True,
    patience=20,
    workers=4,
    verbose=True
)

print("Training completed.")
print("Best model: runs/detect/yolov8_stamp/weights/best.pt")
