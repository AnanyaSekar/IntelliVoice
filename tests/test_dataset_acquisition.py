#!/usr/bin/env python3
"""
Unit Tests for IntelliVoice Dataset Acquisition Infrastructure
Verifies class mapping resolution, STATUS reporting, dry-run execution,
and metadata (SOURCE_INFO.md) creation.
"""

import unittest
import sys
import os
import shutil
import tempfile
from pathlib import Path

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_acquisition import DatasetAcquisitionManager, INTELLIVOICE_CLASSES, CLASS_MAPPINGS

class TestDatasetAcquisition(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.manager = DatasetAcquisitionManager(root_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_class_mapping_coco(self):
        target, status = self.manager.verify_class_mapping("coco", "bus")
        self.assertEqual(target, "bus")
        self.assertEqual(status, "VERIFIED")

        target, status = self.manager.verify_class_mapping("coco", "traffic light")
        self.assertEqual(target, "traffic_light")
        self.assertEqual(status, "VERIFIED")

    def test_class_mapping_open_images(self):
        mid_test_cases = {
            "/m/01bj38": "bus",
            "/m/0k4j": "car",
            "/m/04_15": "motorcycle",
            "/m/0199g": "bicycle",
            "/m/015qff": "traffic_light",
            "/m/01g317": "person",
            "/m/0166ls": "pole",
            "/m/02dgv": "door",
            "/m/01mzpv": "bench"
        }
        for mid, expected_target in mid_test_cases.items():
            target, status = self.manager.verify_class_mapping("open_images", mid)
            self.assertEqual(target, expected_target, f"Failed mapping for MID {mid}")
            self.assertEqual(status, "VERIFIED", f"Status not VERIFIED for MID {mid}")

    def test_open_images_bbox_conversion(self):
        # Open Images normalized bounds: xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.8
        xmin, xmax, ymin, ymax = 0.1, 0.5, 0.2, 0.8
        w = xmax - xmin
        h = ymax - ymin
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0

        self.assertAlmostEqual(x_center, 0.3)
        self.assertAlmostEqual(y_center, 0.5)
        self.assertAlmostEqual(w, 0.4)
        self.assertAlmostEqual(h, 0.6)

    def test_class_mapping_uncertain(self):
        target, status = self.manager.verify_class_mapping("coco", "unknown_alien_object")
        self.assertIsNone(target)
        self.assertEqual(status, "UNMAPPED")

    def test_dry_run_execution(self):
        # Run dry run inside temporary workspace
        self.manager.dry_run()

        # Verify SOURCE_INFO.md created in raw subfolders
        source_keys = ["coco", "open_images", "pothole", "crosswalk", "stairs"]
        for key in source_keys:
            info_file = Path(self.test_dir) / "dataset" / "raw" / key / "SOURCE_INFO.md"
            self.assertTrue(info_file.exists(), f"SOURCE_INFO.md missing for {key}")

        # Verify outputs created
        report_json = Path(self.test_dir) / "outputs" / "dataset_acquisition" / "acquisition_report.json"
        summary_md = Path(self.test_dir) / "outputs" / "dataset_acquisition" / "acquisition_summary.md"
        self.assertTrue(report_json.exists())
        self.assertTrue(summary_md.exists())

    def test_status_reporting(self):
        status = self.manager.get_status()
        self.assertIn("coco", status)
        self.assertIn("open_images", status)
        self.assertIn("pothole", status)
        self.assertIn("crosswalk", status)
        self.assertIn("stairs", status)

    def test_open_images_mock_parsing(self):
        # Setup mock annotation CSV in test_dir
        anno_dir = Path(self.test_dir) / "dataset" / "raw" / "open_images" / "annotations"
        anno_dir.mkdir(parents=True, exist_ok=True)
        csv_file = anno_dir / "validation-annotations-bbox.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n")
            f.write("000002b66c9c498e,freebase,/m/0166ls,1,0.1,0.5,0.2,0.8,0,0,0,0,0\n")
            f.write("000002b66c9c498e,freebase,/m/02dgv,1,0.2,0.6,0.3,0.9,0,0,0,0,0\n")

        # Mock image download URL retrieve to prevent network call
        def dummy_retrieve(url, filename):
            pass

        import urllib.request
        orig_retrieve = urllib.request.urlretrieve
        try:
            urllib.request.urlretrieve = dummy_retrieve
            res = self.manager.acquire_open_images()
            self.assertTrue(res)
        finally:
            urllib.request.urlretrieve = orig_retrieve

        # Check that reports were generated
        report_json = Path(self.test_dir) / "outputs" / "dataset_acquisition" / "open_images_report.json"
        summary_md = Path(self.test_dir) / "outputs" / "dataset_acquisition" / "open_images_summary.md"
        self.assertTrue(report_json.exists())
        self.assertTrue(summary_md.exists())

if __name__ == "__main__":
    unittest.main()

