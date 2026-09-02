#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("analyze_multiday.py")
SPEC = importlib.util.spec_from_file_location("analyze_multiday", SCRIPT_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class AnalyzeMultidayTest(unittest.TestCase):

    def test_default_range_includes_20260806(self):
        self.assertEqual(
            ANALYSIS.DEFAULT_DAYS,
            ("20260803", "20260804", "20260805", "20260806"),
        )
        self.assertEqual(ANALYSIS.DEFAULT_EXPECTED_SEQUENCES, 24)
        self.assertIn("20260806", ANALYSIS.DAY_COLORS)
        self.assertIn("20260806", ANALYSIS.DAY_MARKERS)

    def test_analysis_period_label_describes_selected_days(self):
        self.assertEqual(
            ANALYSIS.analysis_period_label(
                ("20260806", "20260803", "20260805", "20260804")
            ),
            "Four-day (20260803-20260806)",
        )

    def test_orientation_excitation_accumulates_relative_rotation(self):
        half_sqrt = np.sqrt(0.5)
        quaternions_wxyz = np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
            [0.0, 0.0, 0.0, 1.0],
        ])

        metrics = ANALYSIS.orientation_excitation_metrics(
            np.asarray([0.0, 1.0, 2.0]), quaternions_wxyz
        )

        self.assertAlmostEqual(metrics["orientation_path_rad"], np.pi)

    def test_alarm_rows_debounce_persistent_events_and_normalize_duration(self):
        timestamps = np.asarray([0.00, 0.03, 0.06, 0.20, 0.30, 0.36, 0.42, 1.0])
        angular_speed = np.asarray([4.0, 4.0, 4.0, 0.0, 4.0, 4.0, 4.0, 0.0])
        sequence_rows = [{
            "group": "test",
            "sequence": "seq-a",
            "analysis_duration_s": 150.0,
            "fixed_lever_ape_median_mm": 12.0,
        }]
        contexts = {
            "seq-a": {
                "motion": {
                    "timestamps": timestamps,
                    "angular_speed": angular_speed,
                },
                "imu_timeseries": {
                    "timestamps": timestamps,
                    "gyro_norm": angular_speed,
                },
            }
        }

        rows = ANALYSIS.alarm_threshold_rows(
            sequence_rows,
            contexts,
            thresholds_radps=np.asarray([3.0]),
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source"] for row in rows}, {"mocap", "imu"})
        self.assertTrue(all(row["event_count"] == 1 for row in rows))
        self.assertTrue(all(row["events_per_5min"] == 2.0 for row in rows))
        self.assertTrue(all(row["ape_over_10mm"] for row in rows))

    def test_timer_max_count_uses_largest_cumulative_value(self):
        text = "\n".join(
            (
                "2.07 attempt loop closure        3\t00:00:00.1",
                "unrelated log line",
                "2.07 attempt loop closure        11\t00:00:00.2",
                "2.07 attempt loop closure        8\t00:00:00.2",
            )
        )

        count = ANALYSIS.timer_max_count(
            text, "2.07 attempt loop closure"
        )

        self.assertEqual(count, 11)

    def test_summarizes_failure_chain_across_runs(self):
        sequence_rows = [
            {
                "day": "20260803",
                "sequence": "target",
                "analysis_duration_s": 120.0,
                "corrected_ape_median_mm": 50.0,
            }
        ]
        run_rows = [
            {
                "sequence": "target",
                "landmark_time_span_median_s": 0.4,
                "observations_per_landmark": 5.0,
                "distinct_states_per_landmark_mean": 3.2,
                "ransac_large_reprojection_count": 150,
                "ransac_fail_count": 80,
                "uninitialised_landmark_ransac_count": 75,
                "loop_descriptor_match_count": 350,
                "loop_attempt_count": 300,
                "loop_accepted_count": 12,
                "dropped_camera_correspondence_count": 20,
            },
            {
                "sequence": "target",
                "landmark_time_span_median_s": 0.8,
                "observations_per_landmark": 4.6,
                "distinct_states_per_landmark_mean": 3.0,
                "ransac_large_reprojection_count": 250,
                "ransac_fail_count": 120,
                "uninitialised_landmark_ransac_count": 115,
                "loop_descriptor_match_count": 550,
                "loop_attempt_count": 500,
                "loop_accepted_count": 8,
                "dropped_camera_correspondence_count": 24,
            },
        ]

        rows = ANALYSIS.summarize_failure_chain_runs(sequence_rows, run_rows)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["landmark_time_span_median_s"], 0.6)
        self.assertAlmostEqual(rows[0]["ransac_fail_count"], 100.0)
        self.assertAlmostEqual(rows[0]["loop_attempts_per_min"], 200.0)
        self.assertAlmostEqual(rows[0]["loop_rejection_fraction"], 0.975)

    def test_world_displacement_error_is_gauge_aligned_increment_error(self):
        evaluation = SimpleNamespace(
            timestamps=np.arange(5.0),
            reference_positions=np.column_stack(
                (np.arange(5.0), np.zeros(5), np.zeros(5))
            ),
            estimate_positions=np.column_stack(
                (2.0 * np.arange(5.0), np.zeros(5), np.zeros(5))
            ),
        )

        elapsed, errors = ANALYSIS.world_displacement_error_series(
            evaluation, delta_s=1.0
        )

        np.testing.assert_allclose(elapsed, np.arange(4.0))
        np.testing.assert_allclose(errors, np.ones(4))

    def test_plots_failure_chain_evidence(self):
        sequence_rows = [
            {
                "day": "20260803",
                "sequence": "control",
                "corrected_ape_median_mm": 5.0,
                "landmark_time_span_median_s": 50.0,
                "ransac_fail_count": 5.0,
                "loop_attempts_per_min": 2.0,
                "loop_rejection_fraction": 0.1,
            },
            {
                "day": "20260803",
                "sequence": "target",
                "corrected_ape_median_mm": 50.0,
                "landmark_time_span_median_s": 0.6,
                "ransac_fail_count": 100.0,
                "loop_attempts_per_min": 200.0,
                "loop_rejection_fraction": 0.98,
            },
        ]
        stage_rows = [
            {"run": "run1", "stage": "online", "ape_rmse_mm": 40.0},
            {"run": "run1", "stage": "final", "ape_rmse_mm": 50.0},
            {"run": "run1", "stage": "final-ba", "ape_rmse_mm": 45.0},
            {"run": "run2", "stage": "online", "ape_rmse_mm": 60.0},
            {"run": "run2", "stage": "final", "ape_rmse_mm": 70.0},
            {"run": "run2", "stage": "final-ba", "ape_rmse_mm": 65.0},
        ]
        timeline_rows = [
            {
                "elapsed_s": 90.0 + index,
                "angular_speed_radps": 1.0 + 0.1 * index,
                "run1_5s_displacement_error_mm": 20.0 + index,
                "run2_5s_displacement_error_mm": 30.0 + index,
            }
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure_chain.png"

            ANALYSIS.plot_failure_chain_evidence(
                output,
                sequence_rows,
                stage_rows,
                timeline_rows,
                camera_gap_elapsed_s=[93.8, 94.5],
                failure_drop_window_s=(93.8, 94.5),
                target_sequence="target",
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_discovers_flat_and_grouped_day_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            datasets = root / "data"
            layouts = (
                ("20260803", None, "20260803-100000", "unclassified"),
                ("20260805", "slow", "20260805-100000", "slow"),
            )
            for day, group_dir, sequence, _ in layouts:
                sequence_dir = results / day
                if group_dir:
                    sequence_dir /= group_dir
                sequence_dir /= sequence
                sequence_dir.mkdir(parents=True)
                (sequence_dir / f"mocap_{sequence}.log").touch()
                for run in ("run1", "run2"):
                    run_dir = sequence_dir / run
                    run_dir.mkdir()
                    (run_dir / ANALYSIS.FINAL_BA_FILE).touch()
                dataset = datasets / day
                if day == "20260805":
                    dataset /= "morning"
                (dataset / f"{sequence}_euroc").mkdir(parents=True)

            discovered = ANALYSIS.discover_sequences(
                results, datasets, days=("20260803", "20260805")
            )

        self.assertEqual(
            [
                (item.day, item.group, item.sequence, len(item.run_dirs))
                for item in discovered
            ],
            [
                ("20260803", "unclassified", "20260803-100000", 2),
                ("20260805", "slow", "20260805-100000", 2),
            ],
        )

    def test_alarm_classification_summary_counts_rejections_and_misses(self):
        sequence_rows = [
            {"sequence": "safe-accepted", "corrected_ape_median_mm": 7.0},
            {"sequence": "safe-rejected", "corrected_ape_median_mm": 8.0},
            {"sequence": "bad-rejected", "corrected_ape_median_mm": 30.0},
            {"sequence": "bad-accepted", "corrected_ape_median_mm": 50.0},
        ]
        alarm_rows = [
            {
                "sequence": sequence,
                "source": "imu",
                "threshold_radps": 3.0,
                "events_per_5min": events,
            }
            for sequence, events in (
                ("safe-accepted", 1.0),
                ("safe-rejected", 5.0),
                ("bad-rejected", 10.0),
                ("bad-accepted", 2.0),
            )
        ]

        summary = ANALYSIS.alarm_classification_summary(
            sequence_rows,
            alarm_rows,
            source="imu",
            threshold_radps=3.0,
        )

        self.assertEqual(summary["true_positive"], 1)
        self.assertEqual(summary["true_negative"], 1)
        self.assertEqual(summary["false_positive"], 1)
        self.assertEqual(summary["false_negative"], 1)

    def test_correlation_comparison_reports_subset_and_combined(self):
        rows = [
            {"day": "20260804", "metric": 0.0, "corrected_ape_median_mm": 1.0},
            {"day": "20260804", "metric": 1.0, "corrected_ape_median_mm": 2.0},
            {"day": "20260805", "metric": 2.0, "corrected_ape_median_mm": 3.0},
            {"day": "20260805", "metric": 3.0, "corrected_ape_median_mm": 4.0},
            {"day": "20260805", "metric": 4.0, "corrected_ape_median_mm": 5.0},
        ]

        comparison = ANALYSIS.correlation_comparison(rows, "metric")

        self.assertAlmostEqual(comparison["all_rho"], 1.0)
        self.assertAlmostEqual(comparison["day_20260805_rho"], 1.0)
        self.assertEqual(comparison["all_sequences"], 5)
        self.assertEqual(comparison["day_20260805_sequences"], 3)

    def test_plots_three_multiday_diagnostics(self):
        sequence_rows = []
        for index, (day, ape) in enumerate(
            (("20260803", 7.0), ("20260804", 3.0), ("20260805", 30.0))
        ):
            sequence_rows.append(
                {
                    "day": day,
                    "sequence": f"{day}-00000{index}",
                    "corrected_ape_median_mm": ape,
                    "motion_angular_speed_radps_p95": 0.5 + index,
                    "orientation_path_rad": 10.0 + 20.0 * index,
                    "translation_path_m": 2.0 + 10.0 * index,
                    "translation_per_orientation_m_per_rad": 0.1 + 0.03 * index,
                }
            )
        alarm_rows = []
        for source in ("mocap", "imu"):
            for threshold in np.arange(2.0, 4.01, 0.5):
                for index, sequence_row in enumerate(sequence_rows):
                    alarm_rows.append(
                        {
                            "day": sequence_row["day"],
                            "sequence": sequence_row["sequence"],
                            "source": source,
                            "threshold_radps": float(threshold),
                            "events_per_5min": max(
                                0.0, (index + 1) * 8.0 - 4.0 * threshold
                            ),
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            ANALYSIS.plot_multiday_diagnostics(
                output, sequence_rows, alarm_rows, candidate_threshold_radps=3.0
            )

            for name in (
                "01_multiday_motion_excitation_vs_ape.png",
                "02_multiday_alarm_count_vs_ape_at_3radps.png",
                "03_multiday_alarm_threshold_tradeoff.png",
            ):
                path = output / name
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
