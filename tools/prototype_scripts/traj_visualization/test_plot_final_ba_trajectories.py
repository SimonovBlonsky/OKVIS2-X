#!/usr/bin/env python3

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("plot_final_ba_trajectories.py")
SPEC = importlib.util.spec_from_file_location("plot_final_ba_trajectories", SCRIPT_PATH)
VISUALIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VISUALIZER
SPEC.loader.exec_module(VISUALIZER)


class TrajectoryVisualizationScriptTest(unittest.TestCase):

    def test_visualization_script_exists(self):
        self.assertTrue(SCRIPT_PATH.is_file())

    def test_loads_okvis_columns_and_ignores_extra_values(self):
        contents = (
            "timestamp,p_WS_W_x,p_WS_W_y,p_WS_W_z,q_WS_x,q_WS_y,q_WS_z,q_WS_w,"
            "SID\n"
            "1000000000,1,2,3,0.1,0.2,0.3,0.4,7,99\n"
            "2000000000,4,5,6,0.5,0.6,0.7,0.8,8,100\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            path.write_text(contents)

            trajectory = VISUALIZER.load_okvis_trajectory(path)

        self.assertEqual(trajectory.timestamps.tolist(), [1.0, 2.0])
        self.assertEqual(
            trajectory.positions_xyz.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        self.assertEqual(
            trajectory.orientations_quat_wxyz.tolist(),
            [[0.4, 0.1, 0.2, 0.3], [0.8, 0.5, 0.6, 0.7]],
        )

    def test_calculates_three_dimensional_loop_metrics(self):
        positions = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]

        metrics = VISUALIZER.calculate_metrics(positions)

        expected_distance = 5.0 + math.sqrt(26.0)
        self.assertAlmostEqual(metrics.origin_error_m, 1.0)
        self.assertAlmostEqual(metrics.total_distance_m, expected_distance)
        self.assertAlmostEqual(metrics.error_percent, 100.0 / expected_distance)

    def test_origin_zoom_limits_include_start_and_end_with_minimum_span(self):
        positions = [[1.0, 2.0, 0.0], [20.0, 30.0, 0.0], [1.01, 1.99, 0.0]]

        x_limits, y_limits = VISUALIZER.origin_zoom_limits(positions)

        self.assertLess(x_limits[0], 1.0)
        self.assertGreater(x_limits[1], 1.01)
        self.assertLess(y_limits[0], 1.99)
        self.assertGreater(y_limits[1], 2.0)
        self.assertGreaterEqual(x_limits[1] - x_limits[0], 0.06)
        self.assertAlmostEqual(
            x_limits[1] - x_limits[0], y_limits[1] - y_limits[0]
        )

    def test_origin_zoom_inset_is_positioned_in_lower_left(self):
        left, bottom, width, height = VISUALIZER.ORIGIN_INSET_BOUNDS

        self.assertLess(left, 0.1)
        self.assertLess(bottom, 0.1)
        self.assertLess(left + width, 0.5)
        self.assertLess(bottom + height, 0.5)
        self.assertEqual((width, height), (0.30, 0.30))

    def test_discovers_device_and_sequence_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for device_dir, sequence in (
                ("0729_VIO_EGO4", "20260729-175405"),
                ("0729_VIO_EGO2", "20260729-172830"),
                ("0729_VIO_EGO2", "20260729-171621"),
            ):
                result = root / device_dir / sequence / "results"
                result.mkdir(parents=True)
                (result / "okvis2-slam-calib-final-ba_trajectory.csv").touch()

            inputs = VISUALIZER.discover_trajectories(root)

        self.assertEqual(
            [(item.device, item.sequence) for item in inputs],
            [
                ("EGO2", "20260729-171621"),
                ("EGO2", "20260729-172830"),
                ("EGO4", "20260729-175405"),
            ],
        )

    def test_generates_individual_and_combined_png_files(self):
        contents = (
            "timestamp,p_WS_W_x,p_WS_W_y,p_WS_W_z,q_WS_x,q_WS_y,q_WS_z,q_WS_w\n"
            "1000000000,0,0,0,0,0,0,1\n"
            "2000000000,1,1,0,0,0,0,1\n"
            "3000000000,0,0,0.1,0,0,0,1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            for device_dir, sequence in (
                ("0729_VIO_EGO2", "sequence-a"),
                ("0729_VIO_EGO4", "sequence-b"),
            ):
                result = root / device_dir / sequence / "results"
                result.mkdir(parents=True)
                (result / "okvis2-slam-calib-final-ba_trajectory.csv").write_text(
                    contents
                )

            generated = VISUALIZER.generate_plots(root, output, dpi=72)

            self.assertEqual(
                [path.name for path in generated],
                [
                    "EGO2_sequence-a_trajectory.png",
                    "EGO4_sequence-b_trajectory.png",
                    "all_trajectories.png",
                ],
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in generated))


if __name__ == "__main__":
    unittest.main()
