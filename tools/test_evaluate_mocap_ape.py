#!/usr/bin/env python3

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import evaluate_mocap_ape as evaluator


class EvaluateMocapApePlotTest(unittest.TestCase):
    def test_parse_args_accepts_plot_path(self):
        args = evaluator.parse_args(
            ["mocap.log", "result", "--plot", "trajectory.png"]
        )

        self.assertEqual(args.plot, Path("trajectory.png"))

    def test_metrics_use_associated_gt_3d_distance(self):
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
        )
        estimate = reference + np.asarray(
            [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.0, 0.004, 0.0]]
        )

        metrics = evaluator.compute_plot_metrics(reference, estimate)

        expected_rmse_m = math.sqrt((0.003**2 + 0.004**2) / 3.0)
        self.assertAlmostEqual(metrics.gt_distance_m, 7.0)
        self.assertAlmostEqual(metrics.ape_rmse_mm, expected_rmse_m * 1000.0)
        self.assertAlmostEqual(
            metrics.error_percentage, 100.0 * expected_rmse_m / 7.0
        )

    def test_annotation_contains_requested_metrics(self):
        metrics = evaluator.PlotMetrics(
            gt_distance_m=7.0,
            ape_rmse_mm=2.886751,
            error_percentage=0.0412393,
        )

        annotation = evaluator.format_metrics_annotation(metrics)

        self.assertIn("总运动里程: 7.000 m", annotation)
        self.assertIn("APE RMSE: 2.887 mm", annotation)
        self.assertIn("误差百分比: 0.0412%", annotation)

    def test_plot_writes_nonempty_png(self):
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.0, 1.0, 0.2]]
        )
        estimate = reference + np.asarray(
            [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.0, 0.003, 0.0]]
        )
        metrics = evaluator.compute_plot_metrics(reference, estimate)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "trajectory.png"
            evaluator.save_trajectory_plot(
                output, reference, estimate, metrics
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
