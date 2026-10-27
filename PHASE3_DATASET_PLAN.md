# IntelliVoice Phase 3 — Dataset Acquisition and Preparation Plan

---

## 1. Executive Overview

This document defines the formal dataset acquisition, preparation, annotation, and licensing plan for **IntelliVoice** (Phase 3). 

The primary objective of Phase 3 is to construct a balanced, high-quality, non-redundant object detection dataset formatted for YOLO, without altering the core 12-class schema or violating third-party data licenses.

---

## 2. Architectural Pipeline & Route Recognition Strategy

### **CRITICAL ARCHITECTURAL RULE: Standardized Class Mapping**
IntelliVoice does **NOT** train individual YOLO classes for specific bus route numbers (e.g., `bus_21G`, `bus_12B`, `bus_5A`). Creating separate object detection classes for every route number is unscalable, brittle, and architecturally flawed.

Instead, route identification is decoupled into a modular two-stage vision pipeline:

```
+-------------------------------------------------------------+
|                  YOLO Object Detection                      |
|                  (Detects Class 0: bus)                     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 Bus / Display Region Crop                   |
|       (Isolates destination signboards & LED panels)        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Optical Character Recognition (OCR)     |
|          (Extracts route text string, e.g., "21G")          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                Context & Journey Engine                     |
|    (Validates extracted route against user transit itinerary)|
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Priority Engine                         |
|     (Ranks hazard/arrival priority based on spatial context)|
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Voice Alert Manager                     |
|       (Generates concise TTS audio: "Bus 21G arriving")     |
+-------------------------------------------------------------+
```

### **Bus-Related Data Categorization**
To support this pipeline, bus data collection in Phase 3 explicitly categorizes images across four distinct operational contexts:
1. **Whole Bus Detection**: Whole-vehicle bounding boxes under varied lighting, distances, and angles.
2. **Route Display / Destination Boards**: Close-up and medium-range shots of LED displays and printed signboards for OCR testing.
3. **Bus-Stop Environment**: Shelters, boarding queues, and arrival bays.
4. **Door & Entry Steps**: Bus entrance doors (`door`) and boarding steps (`stairs`).

---

## 3. Target Classes & Dataset Acquisition Matrix

The dataset addresses 12 initial target classes. The acquisition strategy pairs existing open-source public benchmark datasets with targeted custom collection for visually impaired navigation scenarios.

### **Class Specification Table**

| ID | Class Name | Category | Recommended Source | Type | Selection Justification | Initial Images | BBox Available | Conversion Required | License / Source | Local Storage | GitHub Dist. | Split Strategy |
|:--:|:-----------|:---------|:-------------------|:----:|:------------------------|:--------------:|:--------------:|:-------------------:|:-----------------|:-------------:|:------------:|:---------------|
| 0 | `bus` | Transit | COCO 2017 + Open Images V7 + Custom Capture | Hybrid | Baseline YOLO weights from COCO; custom captures add local Indian transit buses & display signboards. | 800 | Yes (Public) / No (Custom) | COCO JSON / CSV $\rightarrow$ YOLO txt; Custom manual annotation. | CC BY 4.0 / Custom | Yes (`/dataset/external` & `/dataset/raw`) | Custom Only (Download script for Public) | 70/20/10 (Grouped by video session) |
| 1 | `car` | Road Safety | COCO 2017 / BDD100K | Public | High diversity of passenger vehicles across weather, night scenes, and street perspectives. | 600 | Yes | COCO JSON $\rightarrow$ YOLO txt | CC BY 4.0 / BSD 3-Clause | Yes | No (Fetch script) | 70/20/10 |
| 2 | `motorcycle` | Road Safety | COCO 2017 / Indian Driving Dataset (IDD) | Public | Captures two-wheelers in high-density urban traffic and mixed pedestrian zones. | 500 | Yes | COCO JSON / IDD XML $\rightarrow$ YOLO txt | CC BY 4.0 / CC BY-NC-SA 4.0 | Yes | No (Fetch script) | 70/20/10 |
| 3 | `bicycle` | Road Safety | COCO 2017 / Open Images V7 | Public | Detects parked and active bicycles on sidewalks and near bus stop approaches. | 400 | Yes | COCO JSON $\rightarrow$ YOLO txt | CC BY 4.0 | Yes | No (Fetch script) | 70/20/10 |
| 4 | `traffic_light` | Road Safety | COCO 2017 / Bosch Small Traffic Lights | Public | Standardized detection of vehicular and pedestrian traffic signals at intersections. | 500 | Yes | COCO JSON / Pascal VOC XML $\rightarrow$ YOLO txt | CC BY 4.0 / MIT | Yes | No (Fetch script) | 70/20/10 |
| 5 | `crosswalk` | Road Safety | Roboflow Universe Public Crosswalk + Custom Capture | Hybrid | Essential for street crossing guidance; custom capture ensures eye-level perspective at street corners. | 500 | Yes (Roboflow) / No (Custom) | Roboflow YOLO format; Custom manual annotation. | CC BY 4.0 / Custom | Yes | Custom Only | 70/20/10 (Grouped by location) |
| 6 | `person` | Obstacle | COCO 2017 / Open Images V7 | Public | Rich diversity of ages, poses, clothing, and crowded transit environments without privacy issues. | 700 | Yes | COCO JSON $\rightarrow$ YOLO txt | CC BY 4.0 | Yes | No (Fetch script) | 70/20/10 |
| 7 | `pole` | Obstacle | Mapillary Vistas / Cityscapes + Custom Capture | Hybrid | Vertical poles (utility poles, bus sign posts) are critical head/body collision hazards for visually impaired pedestrians. | 450 | Partial (Segmentation) | Mask Polygon Envelope $\rightarrow$ Bounding Box YOLO txt | Mapillary Research / Custom | Yes | Custom Only | 70/20/10 |
| 8 | `stairs` | Navigation | MCube Stairs Dataset + Custom Capture | Hybrid | Critical navigation feature at transit station entrances, foot overbridges, and bus terminal platforms. | 400 | Yes (Public) / No (Custom) | VOC XML / Roboflow $\rightarrow$ YOLO txt | CC BY 4.0 / Custom | Yes | Custom Only | 70/20/10 (Grouped by staircase) |
| 9 | `pothole` | Obstacle | Roboflow Dedicated Pothole Dataset / Pothole600 | Public | Dedicated dataset avoids dangerous field collection on active roadways while providing thousands of annotated surface defects. | 500 | Yes | VOC XML / YOLO txt | Public Domain / CC BY 4.0 | Yes | No (Fetch script) | 70/20/10 |
| 10 | `door` | Navigation | NYU Depth V2 / SUN RGB-D + Custom Capture | Hybrid | Doors indicate entry/exit points for buses, transit offices, and station facilities. | 400 | Partial | 2D BBox extraction $\rightarrow$ YOLO txt | MIT / Custom | Yes | Custom Only | 70/20/10 |
| 11 | `bench` | Navigation | COCO 2017 / Open Images V7 | Public | Identifies public seating at bus stops and transit terminals for resting/waiting points. | 400 | Yes | COCO JSON $\rightarrow$ YOLO txt | CC BY 4.0 | Yes | No (Fetch script) | 70/20/10 |

---

## 4. Dataset Novelty Strategy

The core scientific and engineering novelty of **IntelliVoice** does **NOT** rely on claiming to create a brand-new object detection dataset from scratch. 

Instead, the novelty lies in the **Context-Aware Multimodal System Architecture** designed for visually impaired public transportation navigation. The dataset is built explicitly to fuel a multi-stage assistance pipeline:

1. **Spatial Object Detection (YOLO)**: Real-time bounding box extraction for safety-critical hazards and navigation landmarks.
2. **Text & Route Parsing (OCR)**: Downstream OCR execution on cropped bus signboards and station signage.
3. **Monocular Depth & Distance Estimation**: Estimating spatial distance to detected hazards (`pothole`, `stairs`, `person`).
4. **Context & Journey State Machine**: Tracking navigation state (e.g., "Waiting at Bus Stop", "Boarding Bus", "Crossing Street").
5. **Dynamic Threat & Priority Engine**: Filtering non-critical detections to prevent cognitive overload.
6. **Adaptive Alert Suppression**: Suppressing repetitive alerts for stationary objects while highlighting immediate motion hazards.
7. **Natural Voice Feedback (TTS)**: Translating prioritized detection events into concise auditory cues.

---

## 5. Dataset Licensing & Repository Rules

To ensure strict compliance with open-source licenses and maintain a clean Git repository, the project adheres to the following rules:

1. **No Binary Bloat in Git**: Raw third-party dataset images and large zip archives must **NEVER** be committed to the Git repository.
2. **Local External Storage**: All downloaded public benchmark datasets must reside in an uncommitted local directory: `dataset/external/` (listed in `.gitignore`).
3. **Reproducible Acquisition Scripts**: Public data acquisition must be driven by automated Python scripts (`src/download_public_data.py`) that fetch images via official APIs or direct links.
4. **License Audit Log**: Every external image source must be recorded in `dataset/LICENSE_AUDIT.md` detailing the original author, source URL, license type (e.g., CC BY 4.0, MIT), and modification status.
5. **Redistribution Policy**:
   - **Public Datasets**: Redistributed via acquisition scripts and configuration files, not image binaries.
   - **Custom IntelliVoice Captures**: Allowed in the repository under CC BY 4.0 or stored in release assets, provided no private facial data is included.

---

## 6. Initial Dataset Target & Quality Criteria

Rather than prematurely scaling to tens of thousands of images, the initial Phase 3 target focuses on a high-quality dataset of approximately **6,250 total instances** across **~3,500 curated images**.

### **Environmental Diversity Matrix**
Data selection and custom collection must strictly satisfy variation across:
* **Distance**: Far-range (advance warning), Mid-range (path planning), Close-range (immediate collision path).
* **Viewpoint**: Pedestrian eye-level, waist-level (chest camera simulation), tilted angles.
* **Lighting**: Broad daylight, overcast, dusk, artificial night lighting at bus shelters.
* **Occlusion**: Partial occlusions (up to 40%) by pedestrians, traffic, or street infrastructure.
* **Geographic Relevance**: Indian urban road scenes, public bus models (e.g., BMTC, BEST, DTC), and local street infrastructure.

---

## 7. Phase 3 Step Breakdown (Phase 3A — Phase 3I)

Phase 3 implementation is divided into 9 sequential, verifiable milestones:

```
+-------------------------------------------------------------------+
| Phase 3A: Dataset Source Selection & License Verification         |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3B: Automated Dataset Acquisition (Public Benchmarks)       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3C: Format Conversion & Class Index Harmonization           |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3D: Custom IntelliVoice Image Collection (Gaps & Context)   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3E: Roboflow Bounding Box Annotation (Custom Captures)      |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3F: Dataset Merge & Schema Harmonization                    |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3G: Automated Dataset Quality Validation                    |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3H: Sequence-Grouped Train / Val / Test Split               |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3I: Dataset Freeze & Version Tagging (Pre-Training Baseline)|
+-------------------------------------------------------------------+
```

### **Detailed Step Descriptions**
* **Phase 3A — Dataset Source Selection**: Finalize URLs, APIs, and license audits for public benchmark subsets (COCO, Open Images, IDD, Roboflow).
* **Phase 3B — Dataset Acquisition**: Execute headless download scripts to pull required public images into `dataset/external/`.
* **Phase 3C — Dataset Conversion**: Convert COCO JSON, VOC XML, and Open Images CSV annotations into standardized 12-class YOLO `.txt` files.
* **Phase 3D — Custom Image Collection**: Conduct targeted smartphone image/video capture for local bus stops, Indian transit buses, stairs, doors, and crosswalks.
* **Phase 3E — Annotation**: Upload custom captures to Roboflow and draw tight bounding box annotations for targeted classes.
* **Phase 3F — Dataset Merge**: Merge converted public subsets and annotated custom captures into `dataset/images` and `dataset/labels`.
* **Phase 3G — Quality Validation**: Execute `dataset_manager.py` to check for missing labels, out-of-bounds coordinates, dark/blurry images, and class distribution imbalances.
* **Phase 3H — Final Split**: Perform a 70% Train / 20% Val / 10% Test split using sequence-grouping to ensure zero data leakage between video frames.
* **Phase 3I — Dataset Freeze**: Create a read-only dataset manifest and freeze `dataset/data.yaml` prior to Phase 4 model training.

---

## 8. Source & Licensing Risk Analysis

| Class Name | Primary Source | License Type | Risk / Uncertainty Assessment | Mitigation Plan |
|:-----------|:---------------+:-------------|:------------------------------|:----------------|
| `pole` | Mapillary Vistas / Cityscapes | Mapillary Research / Non-Commercial | Mapillary research license restricts commercial redistribution of raw imagery. | Extract bounding box labels and use custom-collected images for any publicly distributed dataset subset. |
| `motorcycle` | Indian Driving Dataset (IDD) | CC BY-NC-SA 4.0 | Non-commercial restriction (NC) prevents commercial usage of raw IDD images. | Use COCO 2017 motorcycle subset for base dataset; use IDD strictly for academic/local validation testing. |
| `stairs` | MCube / Roboflow Public | Varied Roboflow Universe Licenses | Roboflow community datasets often have inconsistent annotation quality or loose bounding boxes. | Perform automated filtering via `dataset_manager.py` to remove bounding boxes exceeding image boundaries or with zero width/height. |
| `door` | NYU Depth V2 / SUN RGB-D | Custom Academic / BSD | SUN RGB-D focuses primarily on indoor home/office environments rather than outdoor transit stations. | Supplement public indoor door datasets with custom smartphone captures of bus doors and transit station entrances. |
