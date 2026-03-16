import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import time
import csv
from collections import defaultdict, deque

# =========================
# Load models
# =========================
model = YOLO(r"C:\Users\Anika Sharma\Downloads\alpr\models\license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# =========================
# Plate regex (UK style)
# =========================
plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

# =========================
# OCR correction
# =========================
def correct_plate_format(text):
    num2alpha = {'0':'O','1':'I','2':'Z','3':'E','4':'A','5':'S','6':'G','7':'T','8':'B','9':'P'}
    alpha2num = {'O':'0','I':'1','Z':'2','E':'3','A':'4','S':'5','G':'6','T':'7','B':'8','P':'9'}

    text = text.upper().replace(" ", "")
    if len(text) != 7:
        return ""

    out = []
    for i, ch in enumerate(text):
        if i < 2 or i >= 4:
            out.append(num2alpha.get(ch, ch))
        else:
            out.append(alpha2num.get(ch, ch))
    return "".join(out)

# =========================
# OCR pipeline
# =========================
def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.resize(thr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        res = reader.readtext(thr, detail=0,
                              allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if res:
            cand = correct_plate_format(res[0])
            if plate_pattern.match(cand):
                return cand
    except:
        pass
    return ""

# =========================
# OCR stabilization
# =========================
plate_hist = defaultdict(lambda: deque(maxlen=8))
plate_final = {}

def stable_plate(track_id, new_text):
    if new_text:
        plate_hist[track_id].append(new_text)
        plate_final[track_id] = max(
            set(plate_hist[track_id]),
            key=plate_hist[track_id].count
        )
    return plate_final.get(track_id, "")

# =========================
# Confidence coloring
# =========================
def conf_color(c):
    if c >= 0.75: return (0,255,0)
    if c >= 0.5:  return (0,255,255)
    return (0,0,255)

# =========================
# Video I/O
# =========================
cap = cv2.VideoCapture("pexels-george-morina-5222550 3 (2160p).mp4")
out = cv2.VideoWriter(
    "output_with_metrics.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(3)), int(cap.get(4)))
)

# =========================
# Performance logging
# =========================
log = open("performance_log.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["frame", "fps", "det_ms", "ocr_ms", "total_ms"])

prev_time = time.time()
frame_id = 0
CONF_THRESH = 0.3
locked_boxes = {}

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    det_start = time.time()
    results = model.track(
        frame,
        conf=CONF_THRESH,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )
    det_ms = (time.time() - det_start) * 1000

    ocr_times = []

    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        for box, tid in zip(r.boxes, r.boxes.id):
            track_id = int(tid.item())
            conf = float(box.conf.item())

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            ocr_start = time.time()
            text = recognize_plate(crop)
            ocr_times.append((time.time() - ocr_start) * 1000)

            stable_text = stable_plate(track_id, text)

            # Box locking
            if track_id in locked_boxes:
                x1,y1,x2,y2 = locked_boxes[track_id]
            elif stable_text:
                locked_boxes[track_id] = (x1,y1,x2,y2)

            color = conf_color(conf)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            if stable_text:
                pos = (x1, y1-20 if y1>60 else y2+55)
                cv2.putText(frame, stable_text, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,0), 7)
                cv2.putText(frame, stable_text, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)

    total_ms = (time.time() - frame_start) * 1000
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    avg_ocr = np.mean(ocr_times) if ocr_times else 0

    # Overlay metrics
    cv2.putText(frame, f"FPS: {fps:.2f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    cv2.putText(frame, f"Det: {det_ms:.1f} ms", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    cv2.putText(frame, f"OCR: {avg_ocr:.1f} ms", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    cv2.putText(frame, f"Total: {total_ms:.1f} ms", (20,160),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

    writer.writerow([frame_id, fps, det_ms, avg_ocr, total_ms])
    frame_id += 1
    out.write(frame)

cap.release()
out.release()
log.close()

print("✅ Video + performance_log.csv generated")
