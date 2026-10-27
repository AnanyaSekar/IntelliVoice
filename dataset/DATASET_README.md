# IntelliVoice Custom Dataset Documentation

This document describes the dataset strategy, class definitions, collection methodologies, and split conventions for the **IntelliVoice** project.

---

## 1. Dataset Purpose

The **IntelliVoice** dataset is designed to train a custom computer-vision model (YOLO) tailored for visually impaired navigation in public transport hubs and urban streets. Unlike general object-detection datasets, this dataset focuses specifically on:
- Identifying transit vehicles (`bus`) to facilitate boarding.
- Detecting road features for safety (`crosswalk`, `traffic_light`).
- Identifying immediate physical obstacles and paths (`person`, `pole`, `stairs`, `pothole`, `door`, `bench`).

---

## 2. Target Classes and Definitions

The dataset uses a fixed schema of 12 classes:

| ID | Class Name | Category | Description / Annotation Guidelines |
|----|------------|----------|-------------------------------------|
| 0 | `bus` | Transit | Any public transit bus or shuttle. Bounding boxes must cover the entire bus. (Route numbers are not split into distinct YOLO classes; they will be processed via downstream OCR). |
| 1 | `car` | Road Safety | Motorized four-wheeled passenger vehicles. |
| 2 | `motorcycle` | Road Safety | Motorized two-wheelers including scooters and motorbikes. |
| 3 | `bicycle` | Road Safety | Non-motorized bicycles. |
| 4 | `traffic_light` | Road Safety | Pedestrian or vehicle traffic signals (annotate the housing enclosing the lights). |
| 5 | `crosswalk` | Road Safety | Painted pedestrian zebra crossings. Bounding boxes should cover the visible crosswalk region on the road surface. |
| 6 | `person` | Obstacle | Pedestrians, commuters, or drivers in transit. |
| 7 | `pole` | Obstacle | Utility poles, street light poles, sign posts, and vertical columns. |
| 8 | `stairs` | Navigation | Outdoor or public indoor stairs/steps (bounding boxes cover the active stairs area). |
| 9 | `pothole` | Obstacle | Depressions or damage on the road surface that present a tripping/stepping hazard. |
| 10| `door` | Navigation | Transit station doors, bus doors, or building entrance doors. |
| 11| `bench` | Navigation | Public benches or seating at bus stops/shelters. |

---

## 3. Data Collection Strategy

### A. Image Diversity Guidelines
To ensure robustness, data collection must prioritize a wide variety of capture conditions:
* **Lighting**: Day, dusk, evening (low light), and direct sunlight.
* **Camera Angle**: First-person perspective (simulating smart glasses or chest-mounted cameras).
* **Distance**: Far-range (for advance warning) and close-range (immediate collision path).
* **Occlusions**: Objects partially covered by pedestrians, other vehicles, or poles.
* **Motion**: Typical slight motion blur from walking.

### B. Image Targets (Phase 2 Roadmap)
- **Transit vehicles (`bus`)**: Target 500–1000 images, capturing diverse bus styles, side views, front views, and route display panels.
- **Common obstacles (`person`, `pole`, `bench`, `door`)**: Target 500–1000 images per class.
- **Traffic/Navigation (`traffic_light`, `crosswalk`, `stairs`, `pothole`)**: Target 300–700 images per class.

### C. Privacy & Ethics
- Avoid collecting private or sensitive personal information.
- Bounding boxes do not require face landmarks; faces should ideally be blurred or captured in public, crowded settings where individual identity is not targeted or identifiable.

---

## 4. Annotation Guidelines
- **Tooling**: We recommend using **Roboflow** for group collaboration, annotation management, and export.
- **Bounding Boxes**:
  - Must tightly enclose the object of interest.
  - Avoid including excessive background space.
  - If an object is partially occluded, draw the bounding box only around the visible part, or estimate the bounds if the occlusion is minimal (<20%).

---

## 5. Dataset Structure

The dataset follows the standard YOLO configuration:

```
dataset/
├── data.yaml            # Relative dataset paths and names mapping
├── raw/                 # Unsorted raw imagery/videos before sorting or splitting
├── train/
│   ├── images/          # 70% of images for training
│   └── labels/          # YOLO annotation files (.txt)
├── val/
│   ├── images/          # 20% of images for validation
│   └── labels/          # YOLO annotation files (.txt)
└── test/
    ├── images/          # 10% of images for testing and final evaluation
    └── labels/          # YOLO annotation files (.txt)
```

### Split Strategy to Prevent Data Leakage
- When extracting frames from video recordings, frames from the same video sequence must reside **exclusively in the same split** (e.g. all frames of video `rec_001` must go to `train`, not split across `train` and `val`). This prevents near-identical sequential frames from artificially inflating validation/testing accuracy.

---

## 6. Dataset Quality Checks
Before concluding Phase 2, the dataset must pass the quality validation suite:
- **Annotation Integrity**: Ensure no missing labels or orphaned text files, and confirm classes are within index 0–11.
- **Image Quality**: Detect and flag extremely dark (average luminance < 30) or blurry (Laplacian variance < 100) images.
- **Class Balance**: Generate a distribution chart to ensure no class is severely underrepresented (e.g., <50 instances).
- **Leakage check**: Confirm that identical or highly similar images do not overlap between splits.
