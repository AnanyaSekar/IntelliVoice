#!/usr/bin/env python3
"""
Pretrained YOLO Test Script for IntelliVoice
Loads a lightweight pretrained YOLO model (yolov8n.pt or yolov11n.pt) and runs real-time
object detection on the webcam feed, displaying bounding boxes, labels, and confidence.
"""

import sys
import time
import cv2

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"[ERROR] Required packages not found: {e}", file=sys.stderr)
    print("Please make sure you have activated the 'intellivoice' environment and installed the requirements.", file=sys.stderr)
    sys.exit(1)

def main():
    print("IntelliVoice: Starting Pretrained YOLO Test...")

    # Select device: use Apple Silicon GPU (MPS) if available, else CPU
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("[INFO] Apple Silicon GPU acceleration (MPS) detected. Using MPS.")
    else:
        print("[INFO] MPS not available. Using CPU for inference.")

    # Load a lightweight pretrained model
    # yolov8n.pt is 6.2MB, yolov11n.pt is 5.3MB. Let's use yolov8n.pt as a reliable standard.
    model_name = "yolov8n.pt"
    print(f"Loading pretrained model '{model_name}' on {device}...")
    try:
        model = YOLO(model_name).to(device)
        print("[SUCCESS] YOLO model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}", file=sys.stderr)
        sys.exit(1)

    print("Attempting to open camera index 0...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open camera index 0.", file=sys.stderr)
        print("Please check that your webcam is connected and camera permissions are granted.", file=sys.stderr)
        sys.exit(1)

    print("[SUCCESS] Camera initialized successfully!")
    print("Running real-time object detection. Press 'q' or 'Q' to exit.")

    # Variables to track FPS
    prev_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame. Exiting loop...")
                break

            # Run YOLO inference
            # stream=True processes generator for memory efficiency
            # verbose=False reduces terminal noise
            results = model(frame, stream=True, verbose=False)

            # Draw detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    # x1, y1, x2, y2 represent the bounding box corners
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get confidence
                    conf = float(box.conf[0])

                    # Get class index and class name
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    # Draw rectangle
                    color = (0, 255, 0)  # Green for general detections
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw text label (class_name and confidence)
                    label = f"{class_name} {conf:.2f}"
                    tf = max(1, int(1.5)) # font thickness
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),  # Red text
                        2,
                        lineType=cv2.LINE_AA
                    )

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),  # Blue text
                2,
                lineType=cv2.LINE_AA
            )

            # Display window
            cv2.imshow("IntelliVoice - YOLO Test", frame)

            # Exit on Q key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exit key pressed. Closing...")
                break

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}", file=sys.stderr)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. OpenCV windows closed. YOLO test finished.")

if __name__ == "__main__":
    main()
