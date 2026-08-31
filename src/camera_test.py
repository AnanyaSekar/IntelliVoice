#!/usr/bin/env python3
"""
Camera Test Script for IntelliVoice
Verifies that the camera can be initialized and displays live video using OpenCV.
"""

import sys
import cv2

def main():
    print("IntelliVoice: Starting Camera Test...")
    print("Attempting to open camera index 0...")

    # Initialize the camera (index 0 is usually the default built-in webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera was initialized successfully
    if not cap.isOpened():
        print("[ERROR] Could not open camera index 0.", file=sys.stderr)
        print("Please check that:", file=sys.stderr)
        print("  1. A webcam is connected to your machine.", file=sys.stderr)
        print("  2. The terminal or IDE has permission to access the camera.", file=sys.stderr)
        print("  3. No other application is currently using the camera.", file=sys.stderr)
        sys.exit(1)

    print("[SUCCESS] Camera initialized successfully!")
    print("Press 'q' or 'Q' in the video window to exit.")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame read was not successful
            if not ret:
                print("[WARNING] Failed to grab frame. Exiting video loop...")
                break

            # Display the resulting frame
            cv2.imshow("IntelliVoice - Camera Test", frame)

            # Wait for 'q' key to exit
            # Wait for 1 ms per iteration to process window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exit key pressed. Closing...")
                break

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}", file=sys.stderr)

    finally:
        # When everything done, release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. OpenCV windows closed. Camera test finished.")

if __name__ == "__main__":
    main()
