#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("mcap_vio_to_euroc.py")
SPEC = importlib.util.spec_from_file_location("mcap_vio_to_euroc", SCRIPT_PATH)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class CameraSynchronizationTest(unittest.TestCase):

    def test_matches_streams_after_temporary_frame_misalignment(self):
        period = 33_000_000
        reference = [index * period for index in range(7)]
        skipped_then_recovered = [
            0,
            period,
            3 * period,
            4 * period,
            5 * period,
            5 * period + 1_000,
            6 * period,
        ]

        statistics = CONVERTER.validate_camera_sync(
            [reference, skipped_then_recovered, reference, reference], 10.0
        )

        self.assertEqual(statistics["matched_groups"], 6)
        self.assertEqual(statistics["dropped_frames"], [1, 1, 1, 1])
        self.assertEqual(statistics["max_matched_skew_ns"], 0)


if __name__ == "__main__":
    unittest.main()
