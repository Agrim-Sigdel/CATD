"""
ANPR Viewer — PyQt5 UI to browse MongoDB detections and run local pipeline inference.
"""

import os
import sys
import cv2
import logging
import numpy as np
from datetime import timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from inference import get_model
from logger import get_logger

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel, QTextEdit,
    QHeaderView, QMessageBox, QSplitter, QFrame, QFileDialog, QProgressBar,
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

log = get_logger("viewer")

load_dotenv()

MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB       = os.getenv("MONGO_DB", "anpr")
DETECTIONS_DIR = os.getenv("DETECTIONS_DIR", "detections")
API_KEY        = os.getenv("ROBOFLOW_API_KEY")
STAGE_2_ID     = os.getenv("STAGE_2_ID", "licenceplate-w75l3/1")
STAGE_3_ID     = os.getenv("STAGE_3_ID", "anpr-e7qws/4")

CLASS_MAP = {
    "0": "-", "1": "0", "2": "1", "3": "2", "4": "3",
    "5": "4", "6": "5", "7": "6", "8": "7", "9": "8", "-": "9",
}

mongo = MongoClient(MONGO_URI)
col   = mongo[MONGO_DB]["detections"]

# Load Stage 2 & 3 once at startup
log.info("Loading Stage 2 & 3 models...")
stage2 = get_model(model_id=STAGE_2_ID, api_key=API_KEY)
stage3 = get_model(model_id=STAGE_3_ID, api_key=API_KEY)
log.info("Models ready.")


def _clamp(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def run_pipeline_on_image(image_path: str) -> tuple[str, str]:
    """
    Run Stage 2 (plate localisation) + Stage 3 (OCR) on a saved vehicle crop.
    Returns (plate_text, plate_crop_filename or "").
    """
    import supervision as sv

    frame = cv2.imread(image_path)
    if frame is None:
        return "", ""

    h, w = frame.shape[:2]

    # Stage 2 — locate plate
    res2 = stage2.infer(frame)[0]
    dets2 = sv.Detections.from_inference(res2)
    if len(dets2) == 0:
        return "", ""

    best = np.argmax(dets2.confidence)
    px1, py1, px2, py2 = _clamp(dets2.xyxy[best], w, h)
    plate_crop = frame[py1:py2, px1:px2]
    if plate_crop.size == 0:
        return "", ""

    # Save plate crop
    base = os.path.splitext(os.path.basename(image_path))[0]
    plate_fn   = f"{base}_plate.jpg"
    plate_path = os.path.join(DETECTIONS_DIR, plate_fn)
    cv2.imwrite(plate_path, plate_crop)

    # Stage 3 — OCR
    res3  = stage3.infer(plate_crop)[0]
    dets3 = sv.Detections.from_inference(res3)
    if len(dets3) == 0:
        return "", plate_fn

    x_centers = (dets3.xyxy[:, 0] + dets3.xyxy[:, 2]) / 2
    order = np.argsort(x_centers)
    chars = [CLASS_MAP.get(dets3.data["class_name"][i], dets3.data["class_name"][i]) for i in order]
    return "".join(chars), plate_fn


# ── Qt log handler ────────────────────────────────────────────────────────────
class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record):
        self._signal.emit(self.format(record), record.levelno)


# ── Background worker: run Stage 2+3 on a single crop ────────────────────────
class PipelineWorker(QThread):
    done  = pyqtSignal(str, str, str)  # (plate_text, plate_crop_fn, record_id_str)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, record_id):
        super().__init__()
        self.image_path = image_path
        self.record_id  = record_id

    def run(self):
        try:
            plate_text, plate_fn = run_pipeline_on_image(self.image_path)
            col.update_one({"_id": self.record_id}, {"$set": {
                "plate_text":  plate_text,
                "plate_crop":  plate_fn or None,
                "workflow_result": {"stage2_stage3": "local", "plate": plate_text},
            }})
            self.done.emit(plate_text, plate_fn, str(self.record_id))
        except Exception as e:
            self.error.emit(str(e))


# ── Background worker for video extraction ───────────────────────────────────
class VideoExtractWorker(QThread):
    progress = pyqtSignal(int, int)   # (current_frame, total_frames)
    done     = pyqtSignal(int)        # saved count
    error    = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            import heapq
            from ultralytics import YOLO
            from supervision import ByteTrack, Detections
            from datetime import datetime, timezone as tz

            STAGE1_CONF  = float(os.getenv("STAGE1_CONFIDENCE", 0.25))
            ZONE_EDGE    = float(os.getenv("ZONE_EDGE_PCT", 10)) / 100.0
            N_EDGE       = int(os.getenv("ZONE_FRAMES_EDGE", 5))
            N_MID        = int(os.getenv("ZONE_FRAMES_MID", 5))
            VEHICLE_CLASSES = {2, 3, 5, 7}

            model   = YOLO("yolov8n.pt")
            tracker = ByteTrack()
            cap     = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Cannot open: {self.video_path}")
                return

            total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            edge_cut  = int(total * ZONE_EDGE)   # e.g. 10% boundary

            # per_vehicle per zone: {tid: {"start": heap, "mid": heap, "end": heap}}
            zones: dict[int, dict[str, list]] = {}
            frame_idx = 0

            log.info("Scanning %d frames | conf≥%.2f | zones edge=%d%% N_edge=%d N_mid=%d",
                     total, STAGE1_CONF, int(ZONE_EDGE * 100), N_EDGE, N_MID)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                self.progress.emit(frame_idx, total)

                # Determine zone
                if frame_idx <= edge_cut:
                    zone_key, n = "start", N_EDGE
                elif frame_idx >= total - edge_cut:
                    zone_key, n = "end", N_EDGE
                else:
                    zone_key, n = "mid", N_MID

                results = model(frame, verbose=False, conf=STAGE1_CONF)[0]
                dets    = Detections.from_ultralytics(results)
                if len(dets) > 0:
                    dets = dets[np.isin(dets.class_id, list(VEHICLE_CLASSES))]
                if len(dets) == 0:
                    continue

                dets = tracker.update_with_detections(dets)

                for i in range(len(dets)):
                    tid   = int(dets.tracker_id[i])
                    conf  = float(dets.confidence[i])
                    box   = dets.xyxy[i]
                    entry = (conf, frame_idx, frame.copy(), box)

                    vehicle_zones = zones.setdefault(tid, {"start": [], "mid": [], "end": []})
                    heap = vehicle_zones[zone_key]
                    if len(heap) < n:
                        heapq.heappush(heap, entry)
                    elif conf > heap[0][0]:
                        heapq.heapreplace(heap, entry)

            cap.release()
            log.info("Scan done — %d unique vehicles across 3 zones", len(zones))

            source = os.path.basename(self.video_path)
            saved  = 0

            for tid, vehicle_zones in zones.items():
                for zone_key, heap in vehicle_zones.items():
                    for conf, fidx, frame, box in sorted(heap, key=lambda x: x[0], reverse=True):
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
                        crop = frame[y1:y2, x1:x2]
                        ts   = datetime.now(tz.utc)
                        fn   = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_tid{tid}_{zone_key}_f{fidx}.jpg"
                        cv2.imwrite(os.path.join(DETECTIONS_DIR, fn), crop)
                        col.insert_one({
                            "timestamp":       ts,
                            "source":          source,
                            "frame_index":     fidx,
                            "tracker_id":      tid,
                            "zone":            zone_key,
                            "plate_text":      "",
                            "vehicle_crop":    fn,
                            "plate_crop":      None,
                            "bbox":            [x1, y1, x2, y2],
                            "confidence":      round(conf, 4),
                            "workflow_result": None,
                        })
                        log.info("Vehicle %d | zone=%s | frame %d | conf=%.3f", tid, zone_key, fidx, conf)
                        saved += 1

            self.done.emit(saved)
        except Exception as e:
            self.error.emit(str(e))


# ── Main window ───────────────────────────────────────────────────────────────
class ANPRViewer(QMainWindow):
    log_signal = pyqtSignal(str, int)   # (message, levelno)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANPR Detection Viewer")
        self.resize(1200, 780)
        self.records       = []
        self._worker       = None
        self._video_worker = None
        self._batch_queue  = []
        self._batch_total  = 0
        self._batch_done   = 0

        # attach Qt log handler to root logger so all modules feed the panel
        qt_handler = QtLogHandler(self.log_signal)
        logging.getLogger().addHandler(qt_handler)
        self.log_signal.connect(self._append_log)

        self._build_ui()
        self._load_records()
        log.info("Viewer started")

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4; }
            QTableWidget { background: #2a2a3e; gridline-color: #3a3a5e;
                           selection-background-color: #4a4a8e; border: none; }
            QHeaderView::section { background: #3a3a5e; color: #cdd6f4;
                                   padding: 6px; border: none; }
            QPushButton { background: #4a4a8e; color: white; border-radius: 6px;
                          padding: 7px 16px; font-size: 13px; }
            QPushButton:hover { background: #6a6aae; }
            QPushButton:disabled { background: #333355; color: #666688; }
            QTextEdit { background: #2a2a3e; border: 1px solid #3a3a5e;
                        border-radius: 4px; font-family: Courier; font-size: 11px; }
            QLabel#plate { color: #00ff99; font-size: 26px; font-weight: bold;
                           font-family: Courier; }
            QLabel#section { color: #89b4fa; font-size: 12px; font-weight: bold; }
        """)

        splitter = QSplitter(Qt.Horizontal)
        # panels added below after log panel is built

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 4, 8)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Tracker ID", "Zone", "Confidence", "Plate Text", "Source"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().selectionChanged.connect(self._on_select)
        lv.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_refresh  = QPushButton("Refresh")
        self.btn_upload   = QPushButton("Upload Video")
        self.btn_workflow = QPushButton("Send Selected")
        self.btn_batch    = QPushButton("Send All Selected")
        self.btn_delete   = QPushButton("Delete")
        self.btn_refresh.clicked.connect(self._load_records)
        self.btn_upload.clicked.connect(self._upload_video)
        self.btn_workflow.clicked.connect(self._send_to_workflow)
        self.btn_batch.clicked.connect(self._send_batch)
        self.btn_delete.clicked.connect(self._delete_record)
        for b in (self.btn_refresh, self.btn_upload, self.btn_workflow, self.btn_batch, self.btn_delete):
            btn_row.addWidget(b)
        lv.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background:#2a2a3e; border-radius:4px; color:white; text-align:center; }"
            "QProgressBar::chunk { background:#4a4a8e; border-radius:4px; }"
        )
        lv.addWidget(self.progress_bar)

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setContentsMargins(4, 8, 8, 8)
        rv.setSpacing(8)

        def section(title):
            l = QLabel(title)
            l.setObjectName("section")
            return l

        rv.addWidget(section("Vehicle Crop"))
        self.vehicle_img = QLabel(alignment=Qt.AlignCenter)
        self.vehicle_img.setFixedHeight(220)
        self.vehicle_img.setStyleSheet("background:#2a2a3e; border-radius:4px;")
        rv.addWidget(self.vehicle_img)

        rv.addWidget(section("Plate Crop"))
        self.plate_img = QLabel(alignment=Qt.AlignCenter)
        self.plate_img.setFixedHeight(90)
        self.plate_img.setStyleSheet("background:#2a2a3e; border-radius:4px;")
        rv.addWidget(self.plate_img)

        rv.addWidget(section("Plate Text"))
        self.plate_text_lbl = QLabel("—", alignment=Qt.AlignCenter)
        self.plate_text_lbl.setObjectName("plate")
        rv.addWidget(self.plate_text_lbl)

        rv.addWidget(section("Workflow Result"))
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("Not yet sent to workflow.")
        rv.addWidget(self.result_box)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([660, 540])

        # ── Bottom log panel ──────────────────────────────────────────────────
        outer = QWidget()
        ov    = QVBoxLayout(outer)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(0)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(left)
        top_splitter.addWidget(right)
        top_splitter.setSizes([660, 540])

        log_label = QLabel("  Live Log")
        log_label.setObjectName("section")
        log_label.setFixedHeight(22)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(140)
        self.log_box.setStyleSheet(
            "background:#0d0d1a; color:#a6e3a1; font-family:Courier; font-size:11px;"
            "border-top: 1px solid #3a3a5e;"
        )

        btn_clear = QPushButton("Clear Log")
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self.log_box.clear)

        log_header = QHBoxLayout()
        log_header.addWidget(log_label)
        log_header.addStretch()
        log_header.addWidget(btn_clear)

        ov.addWidget(top_splitter, stretch=1)
        ov.addLayout(log_header)
        ov.addWidget(self.log_box)

        self.setCentralWidget(outer)

    def _append_log(self, message: str, levelno: int):
        colors = {
            logging.DEBUG:    "#6c7086",
            logging.INFO:     "#a6e3a1",
            logging.WARNING:  "#f9e2af",
            logging.ERROR:    "#f38ba8",
            logging.CRITICAL: "#ff0000",
        }
        color = colors.get(levelno, "#cdd6f4")
        self.log_box.append(f'<span style="color:{color}">{message}</span>')
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    def _load_records(self):
        self.table.setRowCount(0)
        self.records = list(col.find().sort("timestamp", -1).limit(200))
        for r in self.records:
            ts = r["timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col_idx, val in enumerate([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                str(r.get("tracker_id", "")),
                r.get("zone", ""),
                f"{r.get('confidence', 0):.3f}",
                r.get("plate_text", ""),
                os.path.basename(str(r.get("source", ""))),
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col_idx, item)

    def _selected_record(self):
        rows = self.table.selectedItems()
        if not rows:
            return None
        row = self.table.currentRow()
        return self.records[row] if row < len(self.records) else None

    def _selected_records(self):
        rows = sorted(set(i.row() for i in self.table.selectedItems()))
        return [self.records[r] for r in rows if r < len(self.records)]

    def _on_select(self):
        rec = self._selected_record()
        if not rec:
            return
        self._show_image(self.vehicle_img, rec.get("vehicle_crop"), (480, 210))
        self._show_image(self.plate_img,   rec.get("plate_crop"),   (480, 80))
        self.plate_text_lbl.setText(rec.get("plate_text") or "—")
        wf = rec.get("workflow_result")
        self.result_box.setPlainText(str(wf) if wf else "Not yet sent to workflow.")

    def _show_image(self, label, filename, max_size):
        if not filename:
            label.setText("No image")
            return
        path = os.path.join(DETECTIONS_DIR, filename)
        if not os.path.exists(path):
            label.setText("File missing")
            return
        pix = QPixmap(path).scaled(
            max_size[0], max_size[1],
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.flv *.webm *.ts *.mts *.m2ts *.3gp *.ogv *.mpg *.mpeg *.mxf *.rm *.rmvb *.divx *.asf);;All Files (*)"
        )
        if not path:
            return
        top_n = int(os.getenv("TOP_N_FRAMES", 10))
        log.info("Video upload started — file=%s top_n=%d", os.path.basename(path), top_n)
        self.btn_upload.setEnabled(False)
        self.btn_upload.setText("Processing...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self._video_worker = VideoExtractWorker(path)
        self._video_worker.progress.connect(self._on_extract_progress)
        self._video_worker.done.connect(self._on_extract_done)
        self._video_worker.error.connect(self._on_extract_error)
        self._video_worker.start()

    def _on_extract_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"Frame {current} / {total}")

    def _on_extract_done(self, saved: int):
        log.info("Video extraction complete — %d frames saved", saved)
        self.btn_upload.setEnabled(True)
        self.btn_upload.setText("Upload Video")
        self.progress_bar.setVisible(False)
        self._load_records()
        QMessageBox.information(self, "Done", f"Extracted {saved} top frames.\nReady to send to workflow.")

    def _on_extract_error(self, msg: str):
        log.error("Video extraction error: %s", msg)
        self.btn_upload.setEnabled(True)
        self.btn_upload.setText("Upload Video")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Extraction error", msg)

    def _send_batch(self):
        recs = self._selected_records()
        if not recs:
            QMessageBox.warning(self, "No selection", "Select one or more rows first.")
            return
        log.info("Batch send — %d detections queued", len(recs))
        self.btn_batch.setEnabled(False)
        self.btn_batch.setText(f"Sending 0/{len(recs)}...")
        self._batch_queue   = list(recs)
        self._batch_total   = len(recs)
        self._batch_done    = 0
        self._run_next_batch()

    def _run_next_batch(self):
        if not self._batch_queue:
            self.btn_batch.setEnabled(True)
            self.btn_batch.setText("Send All Selected")
            log.info("Batch complete — %d sent", self._batch_done)
            self._load_records()
            return
        rec  = self._batch_queue.pop(0)
        path = os.path.join(DETECTIONS_DIR, rec.get("vehicle_crop", ""))
        if not os.path.exists(path):
            log.warning("Skipping missing file: %s", path)
            self._batch_done += 1
            self._run_next_batch()
            return
        self._worker = PipelineWorker(path, rec["_id"])
        self._worker.done.connect(self._on_batch_item_done)
        self._worker.error.connect(self._on_batch_item_error)
        self._worker.start()

    def _on_batch_item_done(self, plate_text: str, _fn: str, _rid: str):
        self._batch_done += 1
        remaining = self._batch_total - self._batch_done
        self.btn_batch.setText(f"Sending {self._batch_done}/{self._batch_total}...")
        log.info("Batch item %d/%d — plate=%s", self._batch_done, self._batch_total, plate_text)
        self._run_next_batch()

    def _on_batch_item_error(self, msg):
        self._batch_done += 1
        log.error("Batch item error: %s", msg)
        self._run_next_batch()

    def _send_to_workflow(self):
        rec = self._selected_record()
        if not rec:
            QMessageBox.warning(self, "No selection", "Select a detection first.")
            return
        path = os.path.join(DETECTIONS_DIR, rec.get("vehicle_crop", ""))
        if not os.path.exists(path):
            QMessageBox.critical(self, "Missing file", f"Image not found:\n{path}")
            return

        self.btn_workflow.setEnabled(False)
        self.btn_workflow.setText("Running...")
        self.result_box.setPlainText("Running pipeline...")
        log.info("Running pipeline — file=%s tracker_id=%s", rec.get("vehicle_crop"), rec.get("tracker_id"))

        self._worker = PipelineWorker(path, rec["_id"])
        self._worker.done.connect(self._on_pipeline_done)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_pipeline_done(self, plate_text: str, plate_fn: str, _rid: str):
        log.info("Pipeline result — plate=%s", plate_text)
        self.plate_text_lbl.setText(plate_text or "—")
        if plate_fn:
            self._show_image(self.plate_img, plate_fn, (480, 80))
        self.result_box.setPlainText(f"Plate: {plate_text}" if plate_text else "No plate detected.")
        self.btn_workflow.setEnabled(True)
        self.btn_workflow.setText("Send Selected")
        self._load_records()

    def _on_pipeline_error(self, msg: str):
        log.error("Pipeline error: %s", msg)
        self.result_box.setPlainText(f"Error: {msg}")
        self.btn_workflow.setEnabled(True)
        self.btn_workflow.setText("Send Selected")
        QMessageBox.critical(self, "Pipeline error", msg)

    def _delete_record(self):
        rec = self._selected_record()
        if not rec:
            return
        if QMessageBox.question(self, "Delete", "Delete this detection?") != QMessageBox.Yes:
            return
        for key in ("vehicle_crop", "plate_crop"):
            fn = rec.get(key)
            if fn:
                p = os.path.join(DETECTIONS_DIR, fn)
                if os.path.exists(p):
                    os.remove(p)
        col.delete_one({"_id": rec["_id"]})
        log.info("Deleted detection — tracker_id=%s plate=%s", rec.get("tracker_id"), rec.get("plate_text"))
        self._load_records()
        self.vehicle_img.clear()
        self.plate_img.clear()
        self.plate_text_lbl.setText("—")
        self.result_box.clear()

    def closeEvent(self, event):
        mongo.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 11))
    win = ANPRViewer()
    win.show()
    sys.exit(app.exec_())
