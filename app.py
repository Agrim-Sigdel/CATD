"""
ANPR — Roboflow Hosted Workflow via REST API (frame-by-frame)
Captures webcam locally, sends each frame to the workflow, displays results.
"""

import os
import cv2
import base64
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from logger import get_logger

log = get_logger("app")

load_dotenv()

API_KEY     = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE   = "anpr-dswdq"
WORKFLOW_ID = "custom-workflow-4"

if not API_KEY:
    raise EnvironmentError("ROBOFLOW_API_KEY is not set.")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)


def encode(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode("utf-8")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    print("Workflow pipeline running — press 'q' to quit.")
    log.info("app.py started — workspace=%s workflow=%s", WORKSPACE, WORKFLOW_ID)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image2": encode(frame)},
                use_cache=True,
            )
            output = result[0] if isinstance(result, list) else result

            plate_chars = output.get("Text Extraction", [])
            plate_text  = "".join(plate_chars) if isinstance(plate_chars, list) else str(plate_chars)

            if plate_text:
                log.info("Workflow plate detected: %s", plate_text)
                print(f"Plate: {plate_text}")
                cv2.putText(frame, plate_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        except Exception as e:
            log.error("Workflow error: %s", e)
            cv2.putText(frame, f"Error: {e}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("ANPR — Workflow", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
