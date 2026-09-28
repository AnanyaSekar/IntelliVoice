from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded!")
print("Starting camera...")
print("Press Q to stop.")

results = model(
    source=0,
    show=True,
    stream=True
)

for result in results:
    pass