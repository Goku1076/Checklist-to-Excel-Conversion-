from ultralytics import YOLO
import cv2

MODEL_PATH = "runs/detect/yolov8_stamp/weights/best.pt"
IMAGE_PATH = "checksheet_image.jpg"
CONFIDENCE = 0.40

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE,
    save=True,
    verbose=False
)

result = results[0]

if result.boxes is not None and len(result.boxes) > 0:
    print("STAMP DETECTED")

    for box in result.boxes:
        confidence = float(box.conf[0])
        print(f"Confidence: {confidence:.2f}")
else:
    print("NO STAMP DETECTED")

annotated = result.plot()

cv2.imshow("Faint Stamp Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
