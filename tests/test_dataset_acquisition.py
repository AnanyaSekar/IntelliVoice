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
        target, status = self.manager.verify_class_mapping("open_images", "/m/0166ls")
        self.assertEqual(target, "pole")
        self.assertEqual(status, "VERIFIED")

        target, status = self.manager.verify_class_mapping("open_images", "/m/02dgv")
        self.assertEqual(target, "door")
        self.assertEqual(status, "VERIFIED")

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

if __name__ == "__main__":
    unittest.main()
