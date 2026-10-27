"""Checks for synchronization, class semantics, image decoding and run leakage."""

import unittest
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
from types import SimpleNamespace

from orda_sim.dataset import collection_group, decode_image, nearest_stamp, normalized_box, split_runs, visible_traffic_lamp


class DatasetTests(unittest.TestCase):
    def test_split_has_no_run_overlap(self):
        result = split_runs(["a", "b", "c", "d"], 0.25, 29)
        self.assertEqual(set(result), {"a", "b", "c", "d"})
        self.assertEqual(list(result.values()).count("val"), 1)
        self.assertEqual(result, split_runs(["d", "c", "b", "a"], 0.25, 29))
        with self.assertRaises(ValueError):
            split_runs(["a"], .2, 0)

    def test_missing_label_packet_is_not_a_negative(self):
        self.assertIsNone(nearest_stamp([100, 200], 150, 10))
        self.assertIsNone(nearest_stamp([], 100, 10))
        self.assertEqual(nearest_stamp([100, 200], 199, 10), 200)

    def test_repeated_seed_keeps_same_split_group(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            for path in (first, second):
                path.mkdir()
                (path / "collection.json").write_text(json.dumps({"automatic_views": True, "seed": 29}))
            self.assertEqual(collection_group(first), collection_group(second))

    def test_clipped_box_stays_normalized(self):
        box = {"class_id": 3, "xyxy": [-10, 20, 90, 60]}
        result = normalized_box(box, 100, 100, 8)
        self.assertEqual(result, [3, .45, .4, .9, .4])
        self.assertIsNone(normalized_box({"class_id": 1, "xyxy": [-90, 0, 10, 10]}, 100, 100, 8))
        with self.assertRaises(ValueError):
            normalized_box({"class_id": 8, "xyxy": [0, 0, 10, 10]}, 100, 100, 8)

    def test_rgb_row_padding(self):
        message = SimpleNamespace(encoding="rgb8", height=2, width=1, step=4,
                                  data=bytes([255, 0, 0, 99, 0, 255, 0, 99]))
        image = decode_image(message)
        self.assertEqual(image.tolist(), [[[0, 0, 255]], [[0, 255, 0]]])

    def test_signal_state_needs_visible_lamp_not_just_housing(self):
        settings = {"hue_ranges": {"traffic_red": [[0, 12], [168, 179]]},
                    "dark_value_max": 70, "minimum_dark_fraction": .75,
                    "saturation_min": 100, "value_min": 60,
                    "minimum_color_fraction": .015, "minimum_blob_fill": .45}
        panel = np.full((128, 56, 3), 20, np.uint8)
        self.assertFalse(visible_traffic_lamp(panel, "traffic_red", settings))
        cv2.circle(panel, (28, 22), 10, (20, 20, 220), -1)
        self.assertTrue(visible_traffic_lamp(panel, "traffic_red", settings))
        panel[:80, :] = (0, 200, 240)  # Foreground sign hides the active red lamp.
        self.assertFalse(visible_traffic_lamp(panel, "traffic_red", settings))


if __name__ == "__main__":
    unittest.main()
