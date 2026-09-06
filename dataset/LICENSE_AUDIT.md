# IntelliVoice Phase 3A — Dataset License Audit

This document presents the legal, licensing, attribution, and redistribution audit for all potential external datasets evaluated for the **IntelliVoice** project.

---

## 1. External Dataset License Audits

### 1. COCO 2017 (Common Objects in Context)
* **Official Website/Repository**: [https://cocodataset.org](https://cocodataset.org)
* **Exact Subset/Task**: 2017 Train/Val Object Detection Annotations
* **Classes Extracted**: `bus`, `car`, `motorcycle`, `bicycle`, `traffic_light`, `person`, `bench`
* **Annotation Format**: COCO JSON format (`bbox: [x, y, width, height]`)
* **Official License/Terms**:
  * **Annotations**: Creative Commons Attribution 4.0 International (CC BY 4.0) by COCO Consortium.
  * **Images**: Images are collected from Flickr. The COCO consortium does not own copyright of images. Users must comply with individual Flickr user licenses and terms of use.
* **Local Download Permitted**: Yes (via `cocodataset.org` or FiftyOne API).
* **Image Redistribution**: No (do not commit Flickr image binaries to GitHub; distribute acquisition script).
* **Commercial Use**: Permitted for annotations (CC BY 4.0); image usage subject to Flickr fair use / individual licenses.
* **Attribution Required**: Yes (Cite Lin et al., COCO).
* **Registration Required**: No.
* **IntelliVoice Usage Decision**: **APPROVED** (Annotations CC BY 4.0; images fetched locally via script).

---

### 2. Open Images V7 (Google)
* **Official Website/Repository**: [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
* **Exact Subset/Task**: Train/Validation V7 Object Detection Bounding Boxes
* **Classes Extracted**: `bus`, `car`, `motorcycle`, `bicycle`, `traffic_light`, `person`, `pole` (`/m/0166ls`), `door` (`/m/02dgv`), `bench` (`/m/01mzpv`)
* **Annotation Format**: CSV (`ImageID, LabelName, XMin, XMax, YMin, YMax`)
* **Official License/Terms**:
  * **Annotations**: Creative Commons Attribution 4.0 International (CC BY 4.0) by Google LLC.
  * **Images**: Images are licensed under Creative Commons Attribution 2.0 / 4.0 / Public Domain as specified per-image in the dataset metadata.
* **Local Download Permitted**: Yes (via FiftyOne or official AWS/GCP download scripts).
* **Image Redistribution**: No (do not commit image binaries to Git; distribute acquisition scripts).
* **Commercial Use**: Permitted (subject to individual image CC BY licenses).
* **Attribution Required**: Yes (Google Open Images V7 attribution).
* **Registration Required**: No.
* **IntelliVoice Usage Decision**: **APPROVED** (Primary public source for `pole` and `door` without registration or non-commercial restrictions).

---

### 3. BDD100K (Berkeley DeepDrive)
* **Official Website/Repository**: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) / UC Berkeley
* **Exact Subset/Task**: 100K Image Detection Set
* **Classes Extracted**: `car`, `bus`, `motorcycle`, `bicycle`, `traffic_light`, `person`
* **Annotation Format**: BDD JSON format
* **Official License/Terms**:
  * **Code & Annotations**: BSD 3-Clause License.
  * **Image Data**: Berkeley DeepDrive License Agreement (Strictly for Academic/Educational/Non-Commercial Research). Commercial use requires an explicit corporate sponsorship/license from UC Berkeley/BAIR.
* **Local Download Permitted**: Yes (Requires user account creation & agreement).
* **Image Redistribution**: No.
* **Commercial Use**: Restricted (Non-commercial research only).
* **Attribution Required**: Yes (Cite Yu et al., BDD100K).
* **Registration Required**: Yes (Academic/Research account login).
* **IntelliVoice Usage Decision**: **APPROVED FOR LOCAL/ACADEMIC USE ONLY** (Kept as secondary validation set; not included in primary pipeline due to non-commercial EULA).

---

### 4. Indian Driving Dataset (IDD)
* **Official Website/Repository**: [http://idd.insirelab.org](http://idd.insirelab.org) / IIIT Hyderabad & Intel
* **Exact Subset/Task**: IDD Detection (10k images, 15 categories)
* **Classes Extracted**: `bus`, `motorcycle`, `car`, `person`, `rider`
* **Annotation Format**: Pascal VOC XML
* **Official License/Terms**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
* **Local Download Permitted**: Yes (Requires registration and approval via IDD portal).
* **Image Redistribution**: Restricted (ShareAlike under NC terms; cannot redistribute commercially).
* **Commercial Use**: No (NC restriction).
* **Attribution Required**: Yes (IIIT Hyderabad / Intel AutoNifty citation).
* **Registration Required**: Yes (User portal login required).
* **IntelliVoice Usage Decision**: **APPROVED FOR LOCAL/ACADEMIC USE ONLY** (Extremely valuable for Indian traffic domain adaptation, but restricted by CC BY-NC-SA 4.0).

---

### 5. Bosch Small Traffic Lights Dataset (BSTLD)
* **Official Website/Repository**: [https://github.com/bosch-ros-pkg/bstld](https://github.com/bosch-ros-pkg/bstld) / Heidelberg University
* **Exact Subset/Task**: Bosch Small Traffic Lights Detection Set
* **Classes Extracted**: `traffic_light`
* **Annotation Format**: YAML annotations
* **Official License/Terms**: MIT License for code and annotations. Images hosted for research evaluation under Bosch dataset agreement.
* **Local Download Permitted**: Yes.
* **Image Redistribution**: No (distribute download link/script).
* **Commercial Use**: Restricted for raw images; MIT for annotations.
* **Attribution Required**: Yes (Bosch / Heidelberg University).
* **Registration Required**: No.
* **IntelliVoice Usage Decision**: **APPROVED FOR LOCAL/ACADEMIC USE ONLY** (Secondary backup for small traffic light detection).

---

### 6. Mapillary Vistas Dataset
* **Official Website/Repository**: [https://www.mapillary.com/datasets/vistas](https://www.mapillary.com/datasets/vistas)
* **Exact Subset/Task**: Commercial / Research Street Scene Instance Segmentation
* **Classes Evaluated**: `pole`
* **Annotation Format**: PNG Segmentation Masks
* **Official License/Terms**: Mapillary Commercial EULA or Mapillary Research License (Strictly non-commercial research).
* **Local Download Permitted**: Yes (Requires account approval).
* **Image Redistribution**: No.
* **Commercial Use**: No (requires expensive commercial agreement).
* **Attribution Required**: Yes.
* **Registration Required**: Yes.
* **IntelliVoice Usage Decision**: **REJECTED** (Unnecessary: `pole` annotations are available in Open Images V7 and COCO under CC BY 4.0 without Mapillary's restrictive EULA or registration requirements).

---

### 7. Cityscapes Dataset
* **Official Website/Repository**: [https://www.cityscapes-dataset.com](https://www.cityscapes-dataset.com) / Daimler AG & TU Darmstadt
* **Exact Subset/Task**: Cityscapes fine annotations (gtFine)
* **Classes Evaluated**: `pole`, `person`, `rider`, `car`, `bus`
* **Annotation Format**: JSON Polygon / PNG Masks
* **Official License/Terms**: Cityscapes License Agreement (Free for non-commercial research and education; academic email registration mandatory).
* **Local Download Permitted**: Yes (Requires academic login).
* **Image Redistribution**: No.
* **Commercial Use**: No.
* **Attribution Required**: Yes (Cordts et al., Cityscapes).
* **Registration Required**: Yes (Academic domain verification required).
* **IntelliVoice Usage Decision**: **REJECTED** (Unnecessary: Adds complex registration and non-commercial academic constraints for classes already covered by COCO and Open Images V7).

---

### 8. Roboflow Universe — Pothole Detection Dataset
* **Official Website/Repository**: [https://universe.roboflow.com/roboflow-100/pothole-ij7w2](https://universe.roboflow.com/roboflow-100/pothole-ij7w2) (Roboflow 100 Benchmark)
* **Exact Subset/Task**: Pothole Detection v1 (YOLOv8 export)
* **Classes Extracted**: `pothole`
* **Annotation Format**: YOLO txt format (`class x_center y_center width height`)
* **Official License/Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0) displayed on Roboflow Universe dataset page.
* **Local Download Permitted**: Yes (via Roboflow API key / direct zip download).
* **Image Redistribution**: Permitted with attribution (CC BY 4.0).
* **Commercial Use**: Permitted (CC BY 4.0).
* **Attribution Required**: Yes (Roboflow 100 / Pothole Dataset author).
* **Registration Required**: No (Public Roboflow API download).
* **IntelliVoice Usage Decision**: **APPROVED** (Primary public source for `pothole`).

---

### 9. Roboflow Universe — Crosswalk Detection Dataset
* **Official Website/Repository**: [https://universe.roboflow.com/crosswalk-detection/crosswalk-yolov8](https://universe.roboflow.com/crosswalk-detection/crosswalk-yolov8) (Version 1)
* **Exact Subset/Task**: Crosswalk Dataset v1
* **Classes Extracted**: `crosswalk`
* **Annotation Format**: YOLO txt format
* **Official License/Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Local Download Permitted**: Yes (via Roboflow API).
* **Image Redistribution**: Permitted with attribution (CC BY 4.0).
* **Commercial Use**: Permitted (CC BY 4.0).
* **Attribution Required**: Yes.
* **Registration Required**: No.
* **IntelliVoice Usage Decision**: **APPROVED** (Primary public source for `crosswalk`, to be supplemented by custom collection).

---

### 10. Roboflow Universe — Stairs Detection Dataset
* **Official Website/Repository**: [https://universe.roboflow.com/stairs-detection/stairs-v2](https://universe.roboflow.com/stairs-detection/stairs-v2) (Version 2)
* **Exact Subset/Task**: Stairs Detection Set v2
* **Classes Extracted**: `stairs`
* **Annotation Format**: YOLO txt format
* **Official License/Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Local Download Permitted**: Yes (via Roboflow API).
* **Image Redistribution**: Permitted with attribution (CC BY 4.0).
* **Commercial Use**: Permitted (CC BY 4.0).
* **Attribution Required**: Yes.
* **Registration Required**: No.
* **IntelliVoice Usage Decision**: **APPROVED** (Primary public source for `stairs`, to be supplemented by custom collection).

---

### 11. NYU Depth V2 / SUN RGB-D
* **Official Website/Repository**: [https://rgbd.cs.princeton.edu](https://rgbd.cs.princeton.edu) / Princeton University
* **Exact Subset/Task**: SUN RGB-D 2D Bounding Boxes
* **Classes Evaluated**: `door`
* **Annotation Format**: 3D & 2D Polygons / MatLab `.mat` files
* **Official License/Terms**: MIT License / Custom Academic License.
* **Local Download Permitted**: Yes.
* **Image Redistribution**: Restricted.
* **Commercial Use**: Academic focus.
* **Registration Required**: No.
* **IntelliVoice Usage Decision**: **REJECTED** (Unnecessary: SUN RGB-D focuses on indoor home/office furniture. Open Images V7 provides outdoor/building `door` bounding boxes under CC BY 4.0 with far simpler acquisition and superior domain relevance).

---

## 2. Source Simplification & Optimization Summary

To avoid unnecessary legal licensing complications and account walls, the dataset acquisition pipeline has been streamlined:

1. **Replaced Mapillary Vistas & Cityscapes**: Extracted `pole` from **Open Images V7** and **COCO 2017** instead. This eliminates the need for proprietary Mapillary EULAs or academic email registration walls.
2. **Replaced SUN RGB-D / NYU Depth V2**: Extracted `door` from **Open Images V7** (`/m/02dgv`) and custom captures instead of dealing with 3D indoor RGB-D MATLAB formats.
3. **Isolated Non-Commercial Datasets**: Restricted **BDD100K** and **IDD** strictly to local academic evaluation subsets, ensuring the core pipeline relies on permissive CC BY 4.0 / Open Data sources.

---

## 3. Phase 3A Decision Table

| Class | Final Source(s) | Status | Reason |
|:------|:----------------|:-------|:-------|
| 0: `bus` | COCO 2017 + Open Images V7 + Custom Capture | **APPROVED** | Permissive CC BY 4.0 licensing; custom captures provide Indian bus models & signboards for OCR. |
| 1: `car` | COCO 2017 + Open Images V7 | **APPROVED** | Fully permissive CC BY 4.0 sources; no registration required. |
| 2: `motorcycle` | COCO 2017 + Open Images V7 (IDD for local val) | **APPROVED** | COCO/Open Images provide main permissive training set; IDD kept strictly for local Indian domain validation. |
| 3: `bicycle` | COCO 2017 + Open Images V7 | **APPROVED** | Standardized open dataset classes with zero licensing complications. |
| 4: `traffic_light` | COCO 2017 + Open Images V7 | **APPROVED** | Permissive CC BY 4.0 sources with abundant variations. |
| 5: `crosswalk` | Roboflow Crosswalk v1 + Custom Capture | **APPROVED** | Roboflow v1 is explicitly licensed CC BY 4.0; custom captures provide pedestrian perspective. |
| 6: `person` | COCO 2017 + Open Images V7 | **APPROVED** | Massive open dataset coverage without privacy compliance risks. |
| 7: `pole` | Open Images V7 + COCO 2017 + Custom Capture | **APPROVED** | Replaced Mapillary/Cityscapes to avoid non-commercial research EULAs and registration walls. |
| 8: `stairs` | Roboflow Stairs v2 + Custom Capture | **APPROVED** | Roboflow v2 is CC BY 4.0; custom captures provide transit station staircases. |
| 9: `pothole` | Roboflow Pothole (RF100) | **APPROVED** | Dedicated pothole detection benchmark under CC BY 4.0; eliminates hazardous field collection. |
| 10: `door` | Open Images V7 + Custom Capture | **APPROVED** | Replaced SUN RGB-D; Open Images V7 provides outdoor building/transit door annotations under CC BY 4.0. |
| 11: `bench` | COCO 2017 + Open Images V7 | **APPROVED** | Standard open dataset classes under CC BY 4.0. |

---

## 4. Approved Acquisition List for Phase 3B

The following verified dataset sources are **APPROVED** to proceed to automated script acquisition in Phase 3B:

1. **COCO 2017 Train/Val Subsets** (via official COCO API / FiftyOne) — *CC BY 4.0 Annotations*
2. **Open Images V7 Subsets** (via Open Images V7 AWS/FiftyOne API) — *CC BY 4.0 Annotations*
3. **Roboflow Universe Pothole Dataset (`pothole-ij7w2`)** — *CC BY 4.0*
4. **Roboflow Universe Crosswalk Dataset (`crosswalk-yolov8`)** — *CC BY 4.0*
5. **Roboflow Universe Stairs Dataset (`stairs-v2`)** — *CC BY 4.0*
6. **IntelliVoice Custom Captures** (Bus, Destination Display, Stairs, Crosswalks, Doors) — *Proprietary / CC BY 4.0*

*(Note: IDD and BDD100K are flagged for optional local academic validation scripts only, and will not be bundled in the primary open repository).*
