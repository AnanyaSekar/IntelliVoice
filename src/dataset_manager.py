#!/usr/bin/env python3
"""
IntelliVoice Dataset Manager
Provides utilities for dataset splitting (70/20/10), quality checks, and visualization.
Ensures zero data leakage by grouping sequence/video frames.
"""

import os
import sys
import glob
import shlex
import hashlib
import argparse
import yaml
from pathlib import Path

# Try importing dependencies; handle gracefully if not installed yet
try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"[WARNING] Missing dependency: {e}")
    print("Some functionality in dataset_manager.py (image checks, visualizations) requires packages from requirements.txt.")

# Target Classes
CLASSES = [
    "bus", "car", "motorcycle", "bicycle", "traffic_light", "crosswalk",
    "person", "pole", "stairs", "pothole", "door", "bench"
]

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_sequence_key(filename):
    """
    Extracts a prefix key to identify frames belonging to the same video/sequence.
    E.g., 'video1_frame0021.jpg' -> 'video1'
    E.g., 'bus_route_21_img_04.png' -> 'bus_route_21'
    Uses underscore splitting to extract the base prefix.
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) > 1:
        # Check if the last part is numeric (common for frame numbers)
        if parts[-1].isdigit() or (parts[-1].startswith('frame') and parts[-1][5:].isdigit()):
            return "_".join(parts[:-1])
        # Default: strip last component
        return "_".join(parts[:-1])
    return stem

def calculate_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

class DatasetManager:
    def __init__(self, dataset_dir="./dataset", outputs_dir="./outputs"):
        self.dataset_dir = Path(dataset_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.dataset_dir / "raw"
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Splits images and labels from dataset/images and dataset/labels
        into train, val, and test splits (70/20/10).
        Ensures sequence-level groupings to prevent data leakage.
        """
        print("\n--- Running Dataset Split Utility ---")

        # Find all images
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(self.images_dir, ext)))
            images.extend(glob.glob(os.path.join(self.images_dir, ext.upper())))

        if not images:
            print(f"[ERROR] No images found in '{self.images_dir}'. Please place annotated images there first.")
            return False

        # Group images by sequence key to avoid leakage
        sequences = {}
        for img_path in images:
            img_path = Path(img_path)
            seq_key = get_sequence_key(img_path.name)
            if seq_key not in sequences:
                sequences[seq_key] = []
            sequences[seq_key].append(img_path)

        # Shuffle the sequences deterministically for consistency
        seq_keys = sorted(list(sequences.keys()))
        import random
        random.seed(42)
        random.shuffle(seq_keys)

        # Count total images
        total_images = len(images)
        print(f"Found {total_images} total images across {len(seq_keys)} distinct sequences.")

        # Allocate sequences to splits based on ratio
        train_keys, val_keys, test_keys = [], [], []
        curr_train_count = 0
        curr_val_count = 0

        target_train = total_images * train_ratio
        target_val = total_images * val_ratio

        for key in seq_keys:
            seq_images = sequences[key]
            count = len(seq_images)

            if curr_train_count < target_train:
                train_keys.append(key)
                curr_train_count += count
            elif curr_val_count < target_val:
                val_keys.append(key)
                curr_val_count += count
            else:
                test_keys.append(key)

        # Copy/Move files helper
        import shutil
        splits = {
            "train": train_keys,
            "val": val_keys,
            "test": test_keys
        }

        for split_name, keys in splits.items():
            split_img_dir = self.dataset_dir / split_name / "images"
            split_lbl_dir = self.dataset_dir / split_name / "labels"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            print(f"Populating '{split_name}' split with {len(keys)} sequences...")
            copied_count = 0

            for key in keys:
                for img_path in sequences[key]:
                    # Determine label path
                    lbl_name = img_path.stem + ".txt"
                    lbl_path = self.labels_dir / lbl_name

                    # Target paths
                    target_img = split_img_dir / img_path.name
                    target_lbl = split_lbl_dir / lbl_name

                    # Copy image
                    shutil.copy2(img_path, target_img)

                    # Copy label if exists, else write empty file (background frame)
                    if lbl_path.exists():
                        shutil.copy2(lbl_path, target_lbl)
                    else:
                        target_lbl.touch()

                    copied_count += 1

            print(f"[SUCCESS] '{split_name}' split complete: {copied_count} images copied.")

        return True

    def validate_dataset(self):
        """
        Validates dataset structure, annotations, image quality, and check leakage.
        """
        print("\n--- Running Dataset Validation Suite ---")
        issues = []
        class_counts = {i: 0 for i in range(len(CLASSES))}
        duplicate_hashes = {}
        total_blurry = 0
        total_dark = 0
        total_bright = 0
        image_splits = {"train": set(), "val": set(), "test": set()}

        splits_to_check = ["train", "val", "test"]

        # 1. Structural Checks
        for split in splits_to_check:
            img_path = self.dataset_dir / split / "images"
            lbl_path = self.dataset_dir / split / "labels"

            if not img_path.exists() or not lbl_path.exists():
                print(f"[WARNING] Split directory '{split}' or its folders do not exist.")
                continue

            # Load files
            images = sorted(list(img_path.glob("*.*")))
            labels = sorted(list(lbl_path.glob("*.txt")))

            print(f"\nChecking split '{split}': {len(images)} images, {len(labels)} label files.")

            # Record file stems to check train-test leakage later
            for img in images:
                image_splits[split].add(img.name)

            # Mapping for quick lookup
            label_stems = {lbl.stem: lbl for lbl in labels}

            # Check individual files
            for img in images:
                stem = img.stem

                # 2. Check for missing labels
                if stem not in label_stems:
                    issues.append(f"[{split.upper()}] Missing label file for image: {img.name}")
                    continue

                lbl = label_stems[stem]

                # 3. Check for duplicates
                try:
                    h = calculate_md5(img)
                    if h in duplicate_hashes:
                        issues.append(f"[{split.upper()}] Duplicate image detected: {img.name} is identical to {duplicate_hashes[h]}")
                    else:
                        duplicate_hashes[h] = img.name
                except Exception as e:
                    issues.append(f"[{split.upper()}] Error calculating hash for {img.name}: {e}")

                # 4. Check image exposure and blurriness using OpenCV
                try:
                    frame = cv2.imread(str(img))
                    if frame is None:
                        issues.append(f"[{split.upper()}] Image failed to load: {img.name}")
                        continue

                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Blurriness check (Laplacian variance)
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if lap_var < 100:
                        total_blurry += 1
                        # Note as dynamic warning but not strict block error
                        # print(f"  [Blurry Frame] {img.name} (Variance: {lap_var:.1f})")

                    # Exposure check
                    avg_brightness = np.mean(gray)
                    if avg_brightness < 30:
                        total_dark += 1
                    elif avg_brightness > 225:
                        total_bright += 1

                except Exception as e:
                    issues.append(f"[{split.upper()}] Failed quality check for {img.name}: {e}")

                # 5. Check Annotation values inside label files
                try:
                    with open(lbl, 'r') as lf:
                        lines = lf.readlines()

                    for idx, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            issues.append(f"[{split.upper()}] Invalid YOLO format in {lbl.name} line {idx+1}: '{line}'")
                            continue

                        class_id_str = parts[0]
                        if not class_id_str.isdigit():
                            issues.append(f"[{split.upper()}] Non-integer class index in {lbl.name} line {idx+1}: '{class_id_str}'")
                            continue

                        class_id = int(class_id_str)
                        if class_id < 0 or class_id >= len(CLASSES):
                            issues.append(f"[{split.upper()}] Out-of-bounds class index in {lbl.name} line {idx+1}: {class_id}")
                            continue

                        # Check bounding box coordinates (YOLO coordinates must be between 0 and 1)
                        coords = [float(p) for p in parts[1:]]
                        for c_idx, c in enumerate(coords):
                            if c < 0.0 or c > 1.0:
                                issues.append(f"[{split.upper()}] Bounding box out of range (0-1) in {lbl.name} line {idx+1}: {c}")

                        class_counts[class_id] += 1

                except Exception as e:
                    issues.append(f"[{split.upper()}] Failed to parse label {lbl.name}: {e}")

        # 6. Check for Leakage between splits
        leak_tr_val = image_splits["train"].intersection(image_splits["val"])
        leak_tr_ts = image_splits["train"].intersection(image_splits["test"])
        leak_val_ts = image_splits["val"].intersection(image_splits["test"])

        if leak_tr_val:
            issues.append(f"[LEAKAGE] Images overlapping between Train and Val splits: {leak_tr_val}")
        if leak_tr_ts:
            issues.append(f"[LEAKAGE] Images overlapping between Train and Test splits: {leak_tr_ts}")
        if leak_val_ts:
            issues.append(f"[LEAKAGE] Images overlapping between Val and Test splits: {leak_val_ts}")

        # 7. Print Quality Validation Summary
        print("\n--- Validation Summary ---")
        print(f"Total blurry images detected (Laplacian Var < 100): {total_blurry}")
        print(f"Total dark images detected (Luminance < 30): {total_dark}")
        print(f"Total bright images detected (Luminance > 225): {total_bright}")

        print("\nClass counts in dataset annotations:")
        for cid, count in class_counts.items():
            print(f"  Class {cid:2d} ({CLASSES[cid]:15s}): {count:5d} instances")

        # Flag class imbalance
        imbalanced = []
        for cid, count in class_counts.items():
            if count == 0:
                imbalanced.append(CLASSES[cid])
        if imbalanced:
            print(f"\n[WARNING] Imbalanced dataset! The following classes have zero annotations: {imbalanced}")

        if issues:
            print(f"\n[FAILURE] Found {len(issues)} data quality issues:")
            for issue in issues[:15]:
                print(f"  - {issue}")
            if len(issues) > 15:
                print(f"  ... and {len(issues) - 15} more issues.")
            return False
        else:
            print("\n[SUCCESS] Dataset structure & quality verification passed. No critical errors found.")
            return True

    def generate_visualizations(self):
        """
        Generates plots of class distribution and save them.
        Also tries to draw sample annotations if split images exist.
        """
        print("\n--- Generating Dataset Visualizations ---")
        
        # 1. Gather Class Instance Counts
        class_counts = {name: 0 for name in CLASSES}
        for split in ["train", "val", "test"]:
            lbl_dir = self.dataset_dir / split / "labels"
            if not lbl_dir.exists():
                continue
            for lbl in lbl_dir.glob("*.txt"):
                try:
                    with open(lbl, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5 and parts[0].isdigit():
                                cid = int(parts[0])
                                if 0 <= cid < len(CLASSES):
                                    class_counts[CLASSES[cid]] += 1
                except Exception:
                    pass

        # Save class distribution plot
        try:
            plt.figure(figsize=(10, 5))
            names = list(class_counts.keys())
            counts = list(class_counts.values())
            
            plt.bar(names, counts, color='teal')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Number of Instances")
            plt.title("IntelliVoice Custom Dataset - Class Distribution")
            plt.tight_layout()
            
            dist_plot = self.outputs_dir / "class_distribution.png"
            plt.savefig(dist_plot)
            plt.close()
            print(f"[SUCCESS] Saved class distribution plot to: {dist_plot}")
        except Exception as e:
            print(f"[ERROR] Failed to save class distribution plot: {e}")

        # 2. Draw Sample Annotations Collage
        self._visualize_annotated_samples()

    def _visualize_annotated_samples(self):
        """
        Finds up to 4 images with annotations and saves a visual composite showing the boxes.
        """
        annotated_images = []
        for split in ["train", "val", "test"]:
            img_dir = self.dataset_dir / split / "images"
            lbl_dir = self.dataset_dir / split / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue

            for img_path in img_dir.glob("*.*"):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists() and lbl_path.stat().st_size > 0:
                    annotated_images.append((img_path, lbl_path))
                if len(annotated_images) >= 4:
                    break
            if len(annotated_images) >= 4:
                break

        if not annotated_images:
            print("[INFO] No annotated images found to draw. Visualization skipped.")
            return

        print(f"Drawing bounding boxes on {len(annotated_images)} sample images...")
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for idx, (img_path, lbl_path) in enumerate(annotated_images):
            try:
                # Load using OpenCV
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape

                # Read labels
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        cx, cy, bw, bh = [float(x) for x in parts[1:]]

                        # Convert from normalized YOLO to pixel coords
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)

                        # Draw box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # Draw label
                        label_text = CLASSES[cid]
                        cv2.putText(img, label_text, (x1, max(y1 - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                axes[idx].imshow(img)
                axes[idx].set_title(f"{img_path.name} ({split})")
                axes[idx].axis('off')

            except Exception as e:
                print(f"[ERROR] Failed drawing boxes on {img_path.name}: {e}")

        # Hide any unused subplots
        for idx in range(len(annotated_images), 4):
            axes[idx].axis('off')

        samples_plot = self.outputs_dir / "annotated_samples.png"
        try:
            plt.tight_layout()
            plt.savefig(samples_plot)
            plt.close()
            print(f"[SUCCESS] Saved annotated samples visualization to: {samples_plot}")
        except Exception as e:
            print(f"[ERROR] Failed to save samples plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="IntelliVoice Custom Dataset Manager")
    parser.add_argument("--action", type=str, choices=["split", "validate", "visualize", "all"], default="validate",
                        help="Action to perform: split, validate, visualize, or all (split, validate, and visualize)")
    parser.add_argument("--dataset-dir", type=str, default="./dataset", help="Path to dataset directory")
    parser.add_argument("--outputs-dir", type=str, default="./outputs", help="Path to save visual outputs/plots")
    
    args = parser.parse_args()
    
    manager = DatasetManager(dataset_dir=args.dataset_dir, outputs_dir=args.outputs_dir)
    
    if args.action == "split":
        manager.split_dataset()
    elif args.action == "validate":
        manager.validate_dataset()
    elif args.action == "visualize":
        manager.generate_visualizations()
    elif args.action == "all":
        # Validate raw folder, split, then validate final splits and generate plots
        split_success = manager.split_dataset()
        if split_success:
            manager.validate_dataset()
            manager.generate_visualizations()
        else:
            print("[ERROR] Split failed, skipping validation and visualization.")

if __name__ == "__main__":
    main()
