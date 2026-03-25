"""
ANPR — 3-Stage Cascaded Pipeline
Saves detections (frame crop + metadata) to MongoDB and local disk.
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from inference import get_model
from ultralytics import YOLO
import supervision as sv
from pymongo import MongoClient
from logger import get_logger

log = get_logger("pipeline")

load_dotenv()

API_KEY        = os.getenv("ROBOFLOW_API_KEY")
STAGE_2_ID     = os.getenv("STAGE_2_ID", "licenceplate-w75l3/1")
STAGE_3_ID     = os.getenv("STAGE_3_ID", "anpr-e7qws/4")
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 10))
MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB       = os.getenv("MONGO_DB", "anpr")
DETECTIONS_DIR = os.getenv("DETECTIONS_DIR", "detections")

if not API_KEY:
    raise EnvironmentError("ROBOFLOW_API_KEY is not set.")

os.makedirs(DETECTIONS_DIR, exist_ok=True)

# ── MongoDB ───────────────────────────────────────────────────────────────────
mongo   = MongoClient(MONGO_URI)
db      = mongo[MONGO_DB]
col     = db["detections"]

# ── COCO vehicle classes ──────────────────────────────────────────────────────
VEHICLE_CLASSES = {2, 3, 5, 7}

CLASS_MAP = {
    "0": "-", "1": "0", "2": "1", "3": "2", "4": "3",
    "5": "4", "6": "5", "7": "6", "8": "7", "9": "8", "-": "9",
}

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading models...")
stage1 = YOLO("yolov8n.pt")
stage2 = get_model(model_id=STAGE_2_ID, api_key=API_KEY)
stage3 = get_model(model_id=STAGE_3_ID, api_key=API_KEY)
log.info("Models loaded — stage1=yolov8n, stage2=%s, stage3=%s", STAGE_2_ID, STAGE_3_ID)
print("Models ready.")
tracker         = sv.ByteTrack()
box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

plate_cache: dict[int, str] = {}


def clamp_box(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def stage1_detect(frame):
    results = stage1(frame, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(results)
    if len(dets) == 0:
        return dets
    return dets[np.isin(dets.class_id, list(VEHICLE_CLASSES))]


def stage2_locate(frame, vehicle_box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(vehicle_box, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    results = stage2.infer(crop)[0]
    dets = sv.Detections.from_inference(results)
    if len(dets) == 0:
        return None, None
    best = np.argmax(dets.confidence)
    px1, py1, px2, py2 = clamp_box(dets.xyxy[best], x2 - x1, y2 - y1)
    plate_box = np.array([x1 + px1, y1 + py1, x1 + px2, y1 + py2])
    plate_crop = frame[y1 + py1:y1 + py2, x1 + px1:x1 + px2]
    return plate_box, plate_crop


def stage3_ocr(frame, plate_box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(plate_box, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    results = stage3.infer(crop)[0]
    dets = sv.Detections.from_inference(results)
    if len(dets) == 0:
        return ""
    x_centers = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2
    order = np.argsort(x_centers)
    chars = [CLASS_MAP.get(dets.data["class_name"][i], dets.data["class_name"][i]) for i in order]
    return "".join(chars)


def save_detection(frame, vehicle_box, plate_crop, plate_text, tracker_id, source, frame_idx):
    """Save cropped vehicle frame to disk and record in MongoDB."""
    ts        = datetime.now(timezone.utc)
    filename  = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_id{tracker_id}.jpg"
    filepath  = os.path.join(DETECTIONS_DIR, filename)

    # Save vehicle crop
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(vehicle_box, w, h)
    vehicle_crop = frame[y1:y2, x1:x2]
    cv2.imwrite(filepath, vehicle_crop)

    # Save plate crop separately if available
    plate_filename = None
    if plate_crop is not None and plate_crop.size > 0:
        plate_filename = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_id{tracker_id}_plate.jpg"
        cv2.imwrite(os.path.join(DETECTIONS_DIR, plate_filename), plate_crop)

    doc = {
        "timestamp":       ts,
        "source":          str(source),
        "frame_index":     frame_idx,
        "tracker_id":      int(tracker_id),
        "plate_text":      plate_text,
        "vehicle_crop":    filename,
        "plate_crop":      plate_filename,
        "bbox":            [int(x1), int(y1), int(x2), int(y2)],
        "workflow_result": None,   # filled later via viewer
    }
    result = col.insert_one(doc)
    log.info("Saved detection — tracker_id=%s plate=%s file=%s", tracker_id, plate_text, filename)
    return str(result.inserted_id)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        raise RuntimeError(f"Cannot open source: {source}")

    log.info("Pipeline started — source=%s frame_interval=%s", source, FRAME_INTERVAL)

    print(f"Pipeline running on: {source} (every {FRAME_INTERVAL} frames) — press 'q' to quit.")

    frame_count   = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_INTERVAL == 0:
            log.debug("Processing frame %d", frame_count)
            dets = stage1_detect(frame)

            if len(dets) > 0:
                dets = tracker.update_with_detections(dets)
                log.debug("Frame %d — %d vehicle(s) detected", frame_count, len(dets))

                for box, tid in zip(dets.xyxy, dets.tracker_id):
                    plate_box, plate_crop = stage2_locate(frame, box)
                    if plate_box is None:
                        log.debug("Frame %d tracker_id=%s — no plate found", frame_count, tid)
                        continue
                    text = stage3_ocr(frame, plate_box)
                    if text:
                        plate_cache[tid] = text
                        log.info("Frame %d tracker_id=%s — plate read: %s", frame_count, tid, text)

                    save_detection(frame, box, plate_crop, plate_cache.get(tid, ""),
                                   tid, source, frame_count)

                labels = [f"ID:{tid}  {plate_cache.get(tid, '')}" for tid in dets.tracker_id]
                last_annotated = box_annotator.annotate(scene=frame.copy(), detections=dets)
                last_annotated = label_annotator.annotate(scene=last_annotated, detections=dets, labels=labels)
            else:
                log.debug("Frame %d — no vehicles detected", frame_count)
                last_annotated = frame.copy()

        display = last_annotated if last_annotated is not None else frame
        cv2.imshow("ANPR — Pipeline", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Pipeline stopped — total frames processed: %d", frame_count)
    mongo.close()


if __name__ == "__main__":
    main()
