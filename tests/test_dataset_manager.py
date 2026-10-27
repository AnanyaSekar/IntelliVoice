#!/usr/bin/env python3
"""
Unit Tests for IntelliVoice Dataset Manager
Verifies that the sequence grouping and splitting logic behaves as expected.
"""

import unittest
import sys
from pathlib import Path

# Add src/ to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_manager import get_sequence_key

class TestDatasetManager(unittest.TestCase):
    def test_get_sequence_key_standard(self):
        # Frame extraction from video
        self.assertEqual(get_sequence_key("video1_frame0021.jpg"), "video1")
        self.assertEqual(get_sequence_key("bus_route_21G_04.png"), "bus_route_21G")
        
    def test_get_sequence_key_no_underscore(self):
        # Plain files
        self.assertEqual(get_sequence_key("bus.jpg"), "bus")
        self.assertEqual(get_sequence_key("car1.png"), "car1")
        
    def test_get_sequence_key_non_numeric_last(self):
        # Check that if the last component is not numeric or "frameX", it still groups reasonably
        self.assertEqual(get_sequence_key("pothole_street_view.jpg"), "pothole_street")

if __name__ == "__main__":
    unittest.main()
