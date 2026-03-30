import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# =========================
# Load models
# =========================

# Load your trained YOLOv8 license plate detector
yolo = YOLO("ALPR final\\code files\\models\\license_plate_best.pt")  # <-- update path if needed

# PaddleOCR (NO deprecated args)
ocr = PaddleOCR(lang="en")

print("✅ YOLO and PaddleOCR loaded successfully")

# =========================
# Image / Video input
# =========================

VIDEO_PATH = "ALPR final\\test videos\\pexels-george-morina-5222550 3 (2160p).mp4"   # or image path
OUTPUT_PATH = "alpr\\paddle_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =========================
# Helper: preprocess plate
# =========================
def preprocess_plate(crop):
    crop = cv2.resize(crop, (160, 48))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return crop

# =========================
# Inference loop
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, conf=0.4)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = preprocess_plate(crop)

            # PaddleOCR inference
            ocr_result = ocr.ocr(crop)

            plate_text = ""
            plate_conf = 0.0

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    plate_text += line[1][0]
                    plate_conf = max(plate_conf, line[1][1])

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{plate_text} ({plate_conf:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            print("Plate:", plate_text, "Conf:", plate_conf)

    writer.write(frame)

cap.release()
writer.release()

print("✅ Inference completed")
print("📁 Output saved to:", OUTPUT_PATH)
