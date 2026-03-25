"""
Extract the top-N highest-confidence vehicle detections from a video.
Saves cropped frames to disk and records them in MongoDB for the viewer.

Usage:
    python3 extract_top_frames.py <video_path>
"""

import os
import sys
import cv2
import heapq
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from ultralytics import YOLO
from pymongo import MongoClient
from logger import get_logger

load_dotenv()

MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB       = os.getenv("MONGO_DB", "anpr")
DETECTIONS_DIR = os.getenv("DETECTIONS_DIR", "detections")
TOP_N          = int(os.getenv("TOP_N_FRAMES", 10))

# COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

log   = get_logger("extract")
mongo = MongoClient(MONGO_URI)
col   = mongo[MONGO_DB]["detections"]

os.makedirs(DETECTIONS_DIR, exist_ok=True)


def scan_video(video_path: str) -> list[dict]:
    """
    Scan every frame, run Stage 1, keep a min-heap of the top-N detections
    by confidence. Returns list of top entries sorted best-first.
    Each entry: {frame_idx, confidence, frame, box}
    """
    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Scanning %d frames in %s for top-%d detections", total, video_path, TOP_N)

    heap        = []   # min-heap: (confidence, frame_idx, frame, box)
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, verbose=False)[0]
        boxes   = results.boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(boxes.conf[i].item())
            box  = boxes.xyxy[i].cpu().numpy()

            entry = (conf, frame_idx, frame.copy(), box)

            if len(heap) < TOP_N:
                heapq.heappush(heap, entry)
            elif conf > heap[0][0]:
                heapq.heapreplace(heap, entry)

        if frame_idx % 100 == 0:
            log.info("  ...frame %d / %d", frame_idx, total)

    cap.release()
    log.info("Scan complete — %d candidate frames collected", len(heap))
    return sorted(heap, key=lambda x: x[0], reverse=True)


def save_top_frames(video_path: str, top: list[dict]):
    source = os.path.basename(video_path)
    saved  = 0

    for conf, frame_idx, frame, box in top:
        h, w = frame.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))
        crop = frame[y1:y2, x1:x2]

        ts       = datetime.now(timezone.utc)
        filename = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_frame{frame_idx}.jpg"
        filepath = os.path.join(DETECTIONS_DIR, filename)
        cv2.imwrite(filepath, crop)

        doc = {
            "timestamp":       ts,
            "source":          source,
            "frame_index":     frame_idx,
            "tracker_id":      None,
            "plate_text":      "",
            "vehicle_crop":    filename,
            "plate_crop":      None,
            "bbox":            [x1, y1, x2, y2],
            "confidence":      round(conf, 4),
            "workflow_result": None,
        }
        col.insert_one(doc)
        log.info("Saved frame %d — conf=%.3f file=%s", frame_idx, conf, filename)
        saved += 1

    log.info("Done — %d frames saved to '%s' and recorded in MongoDB", saved, DETECTIONS_DIR)
    print(f"\nExtracted {saved} top frames from '{source}'. Open viewer.py to inspect them.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_top_frames.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    top = scan_video(video_path)
    save_top_frames(video_path, top)
    mongo.close()


if __name__ == "__main__":
    main()
