import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO

CHECKSHEET_PATH = "20250615.jpg"
YOLO_MODEL_PATH = "runs/detect/yolov8_stamp/weights/best.pt"
SCALE_PERCENT = 20
YOLO_CONFIDENCE = 0.40

DEBUG_DIR = "debug_regions"
OUTPUT_IMAGE = "output_checksheet_with_boxes.jpg"
CSV_PATH = "stamp_detection_results.csv"

REGIONS = {
    "exterior_lh": (275, 600, 700, 750),
    "exterior_rh": (275, 750, 700, 900),
    "exterior_final": (175, 1300, 700, 1575),
    "interior_lh": (175, 2100, 425, 2870),
    "interior_rh": (425, 2100, 675, 2870),
    "interior_final": (175, 2880, 700, 3200)
}

COLORS = {
    "exterior_lh": (128, 0, 128),
    "exterior_rh": (0, 255, 0),
    "exterior_final": (255, 0, 0),
    "interior_lh": (128, 0, 255),
    "interior_rh": (147, 20, 255),
    "interior_final": (0, 0, 0)
}

os.makedirs(DEBUG_DIR, exist_ok=True)

model = YOLO(YOLO_MODEL_PATH)


def scale_box(box):
    return tuple(int(v * SCALE_PERCENT / 100) for v in box)


def hough_circle_detection(image, label):
    if image is None or image.size == 0:
        return "NG"

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=5,
            maxRadius=30
        )

        output = image.copy()

        if circles is None:
            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"{label}_hough_NG.jpg"),
                output
            )
            return "NG"

        circles = np.uint16(np.around(circles))

        for x, y, r in circles[0]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

        cv2.imwrite(
            os.path.join(DEBUG_DIR, f"{label}_hough_OK.jpg"),
            output
        )

        return "OK"

    except Exception as e:
        print(f"Hough error in {label}: {e}")
        return "NG"


def yolo_detection(image, label):
    if image is None or image.size == 0:
        return "NG"

    try:
        results = model.predict(
            source=image,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

        result = results[0]
        output = image.copy()

        if result.boxes is None or len(result.boxes) == 0:
            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"{label}_yolo_NG.jpg"),
                output
            )
            return "NG"

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"STAMP {confidence:.2f}",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        cv2.imwrite(
            os.path.join(DEBUG_DIR, f"{label}_yolo_OK.jpg"),
            output
        )

        return "OK"

    except Exception as e:
        print(f"YOLO error in {label}: {e}")
        return "NG"


def detect_stamp(image, label):
    result = hough_circle_detection(image, label)

    if result == "OK":
        return "OK"

    return yolo_detection(image, label)


image = cv2.imread(CHECKSHEET_PATH)

if image is None:
    raise FileNotFoundError(CHECKSHEET_PATH)

height, width = image.shape[:2]

new_width = int(width * SCALE_PERCENT / 100)
new_height = int(height * SCALE_PERCENT / 100)

image = cv2.resize(image, (new_width, new_height))

regions = {
    name: scale_box(box)
    for name, box in REGIONS.items()
}

results = []

for label, (x1, y1, x2, y2) in regions.items():

    x1 = max(0, min(x1, new_width))
    x2 = max(0, min(x2, new_width))
    y1 = max(0, min(y1, new_height))
    y2 = max(0, min(y2, new_height))

    if x2 <= x1 or y2 <= y1:
        results.append({
            "Section": label.upper(),
            "Status": "SKIPPED"
        })
        continue

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        results.append({
            "Section": label.upper(),
            "Status": "SKIPPED"
        })
        continue

    status = detect_stamp(roi, label)

    results.append({
        "Section": label.upper(),
        "Status": status
    })

    color = COLORS.get(label, (0, 0, 255))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        image,
        f"{label.upper()} : {status}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

cv2.imwrite(OUTPUT_IMAGE, image)

cv2.imshow("Stamp Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

df = pd.DataFrame(results)
df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if os.path.exists(CSV_PATH):
    df.to_csv(CSV_PATH, mode="a", header=False, index=False)
else:
    df.to_csv(CSV_PATH, index=False)

for result in results:
    print(f"{result['Section']:20} : {result['Status']}")

print(f"CSV: {CSV_PATH}")
print(f"Output: {OUTPUT_IMAGE}")
