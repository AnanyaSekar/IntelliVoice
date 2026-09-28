from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

# =========================
# FLASK APP
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# =========================
# LOAD YOLO MODEL
# =========================

print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")
print("YOLO11n loaded successfully!")

important_objects = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus"
}

# =========================
# SERVE FRONTEND
# =========================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    # Serve files such as style.css and script.js
    allowed_files = {"index.html", "style.css", "script.js"}

    if filename in allowed_files:
        return send_from_directory(BASE_DIR, filename)

    return jsonify({"error": "File not found"}), 404


# =========================
# BACKEND STATUS
# =========================

@app.route("/api/status")
def status():
    return jsonify({
        "status": "ready",
        "message": "IntelliVoice AI backend is ready"
    })


# =========================
# OBJECT DETECTION
# =========================

@app.route("/api/detect", methods=["POST"])
def detect():

    file = request.files.get("frame")

    if file is None:
        return jsonify({
            "error": "No camera frame received"
        }), 400

    image_bytes = file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "error": "Could not read camera frame"
        }), 400

    results = model(frame, verbose=False)

    detections = []
    frame_width = frame.shape[1]

    for result in results:
        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            if object_name not in important_objects:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center_x = (x1 + x2) / 2

            if center_x < frame_width / 3:
                position = "left"
            elif center_x < 2 * frame_width / 3:
                position = "center"
            else:
                position = "right"

            # Estimate the fraction of the frame
            # occupied by the detected object.
            box_width = x2 - x1
            box_height = y2 - y1

            area = box_width * box_height
            frame_area = frame.shape[1] * frame.shape[0]

            size_ratio = area / frame_area

            detections.append({
                "object": object_name,
                "confidence": round(confidence, 2),
                "position": position,
                "size_ratio": round(size_ratio, 4),
                "box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

    return jsonify({
        "detections": detections
    })


# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    print()
    print("======================================")
    print("       INTELLIVOICE AI ASSISTANT")
    print("======================================")
    print("Open http://127.0.0.1:5000 in your browser")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )