#!/usr/bin/env python3

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from tools.accuracy_analysis.scripts import analyze_vio_causal_diagnostics as analysis


class CausalDiagnosticsAnalysisTest(unittest.TestCase):

    def test_trapezoidal_integral_supports_installed_numpy(self):
        self.assertAlmostEqual(
            analysis._trapezoidal_integral(
                np.asarray([0.0, 2.0, 2.0]),
                np.asarray([0.0, 1.0, 2.0]),
            ),
            3.0,
        )

    def test_ransac_contract_requires_complete_pose_and_outcome_columns(self):
        required = analysis.REQUIRED_COLUMNS["vio_diag_ransac.csv"]
        for column in (
            "data_association_start_tx",
            "data_association_start_qz",
            "pre_invocation_tx",
            "pre_invocation_qz",
            "gp3p_model_tx",
            "gp3p_model_qz",
            "model_computed",
            "threshold_success",
            "returned_success",
            "start_to_model_rotation_rad",
            "pre_invocation_to_model_translation_m",
        ):
            self.assertIn(column, required)

    def test_frame_contract_requires_visual_removal_counts(self):
        required = analysis.REQUIRED_COLUMNS["vio_diag_frame.csv"]
        for reason in range(4):
            self.assertIn(f"observations_removed_reason_{reason}", required)

    def test_detect_angular_events_merges_short_gap_and_integrates_angle(self):
        timestamps = np.arange(0.0, 1.01, 0.05)
        speed = np.zeros_like(timestamps)
        speed[(timestamps >= 0.10) & (timestamps <= 0.20)] = 4.0
        speed[(timestamps >= 0.40) & (timestamps <= 0.50)] = 5.0

        events = analysis.detect_angular_events(timestamps, speed)

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start_s, 0.10)
        self.assertAlmostEqual(events[0].end_s, 0.50)
        self.assertAlmostEqual(events[0].peak_radps, 5.0)
        self.assertGreater(events[0].integrated_angle_rad, 0.0)

    def test_detect_angular_events_rejects_short_event_and_separates_long_gap(self):
        timestamps = np.asarray(
            [0.0, 0.02, 0.04, 0.10, 0.20, 0.30, 0.50, 0.60]
        )
        speed = np.asarray([0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 4.0])

        events = analysis.detect_angular_events(timestamps, speed)

        self.assertEqual([(event.start_s, event.end_s) for event in events], [
            (0.10, 0.20),
            (0.50, 0.60),
        ])

    def test_image_proxies_distinguish_uniform_sharp_and_blurred_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uniform = np.full((64, 64), 127, dtype=np.uint8)
            checker = (
                ((np.indices((64, 64)) // 4).sum(axis=0) % 2) * 255
            ).astype(np.uint8)
            uniform_path = root / "uniform.png"
            sharp_path = root / "sharp.png"
            blurred_path = root / "blurred.png"
            Image.fromarray(uniform).save(uniform_path)
            sharp_image = Image.fromarray(checker)
            sharp_image.save(sharp_path)
            sharp_image.filter(ImageFilter.GaussianBlur(radius=2.0)).save(
                blurred_path
            )

            uniform_stats = analysis.compute_image_statistics(uniform_path)
            sharp_stats = analysis.compute_image_statistics(sharp_path)
            blurred_stats = analysis.compute_image_statistics(blurred_path)

        self.assertEqual(uniform_stats["image_laplacian_variance"], 0.0)
        self.assertGreater(
            sharp_stats["image_laplacian_variance"],
            blurred_stats["image_laplacian_variance"],
        )
        self.assertGreater(
            sharp_stats["image_gradient_median"],
            blurred_stats["image_gradient_median"],
        )

    def test_robust_onset_and_recovery_require_consecutive_frames(self):
        timestamps = np.arange(12, dtype=float) * 0.1
        values = np.asarray([10, 10, 10, 10, 13, 13, 13, 13, 10, 10, 10, 10])

        onset = analysis.detect_robust_onset(
            timestamps, values, baseline_values=values[:4],
            harmful_direction=1, epsilon=1.0,
        )
        recovery = analysis.detect_robust_recovery(
            timestamps, values, baseline_values=values[:4],
            search_start_s=0.7, epsilon=1.0,
        )

        self.assertAlmostEqual(onset, 0.4)
        self.assertIsNone(recovery)

        longer_timestamps = np.arange(14, dtype=float) * 0.1
        longer_values = np.r_[values, 10, 10]
        self.assertAlmostEqual(
            analysis.detect_robust_recovery(
                longer_timestamps,
                longer_values,
                baseline_values=values[:4],
                search_start_s=0.7,
                epsilon=1.0,
            ),
            0.8,
        )

    def test_control_matching_uses_raw_translation_and_rotation(self):
        event = {
            "sequence": "seq-a",
            "run": "run1",
            "start_s": 20.0,
            "end_s": 21.0,
            "active_initialised_landmarks": 100.0,
            "accepted_map_matches": 50.0,
            "mocap_body_translation_m": 0.002,
            "mocap_body_rotation_rad": 0.001,
            "image_laplacian_variance": 20.0,
            "keypoints_total": 200.0,
        }
        candidates = [
            {
                **event,
                "candidate_id": "same",
                "start_s": 40.0,
                "end_s": 41.0,
                "peak_angular_speed_radps": 0.5,
            },
            {
                **event,
                "candidate_id": "other-run",
                "run": "run2",
                "start_s": 40.0,
                "end_s": 41.0,
                "peak_angular_speed_radps": 0.5,
            },
            {
                **event,
                "candidate_id": "bad-rotation",
                "start_s": 50.0,
                "end_s": 51.0,
                "peak_angular_speed_radps": 0.5,
                "mocap_body_rotation_rad": 0.5,
            },
        ]

        selected = analysis.select_matched_controls(
            event, candidates,
            angular_events=[analysis.AngularEvent(20.0, 21.0, 4.0, 1.0)],
        )

        self.assertEqual([row["candidate_id"] for row in selected], ["same"])

    def test_loader_rejects_missing_required_ransac_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._write_minimal_run(root)
            ransac = run_root / "diagnostics" / "vio_diag_ransac.csv"
            with ransac.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = [
                    field for field in reader.fieldnames
                    if field != "gp3p_model_qz"
                ]
            with ransac.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(fields)

            with self.assertRaisesRegex(ValueError, "gp3p_model_qz"):
                analysis.load_diagnostic_run(root, "seq-a", "run1")

    def test_typed_rows_accepts_semantically_empty_metrics(self):
        distribution_columns = {
            "best_map_descriptor_distance_p10_cam0",
            "best_map_descriptor_distance_median_cam0",
            "best_map_descriptor_distance_p90_cam0",
            "baseline_m_p10",
            "baseline_m_median",
            "ray_angle_rad_p10",
            "ray_angle_rad_median",
            "inlier_grid_fraction_cam0",
        }
        required = {"schema_version", "timestamp_ns", "frame_id"} | (
            distribution_columns
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.csv"
            fields = sorted(required)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "schema_version": 1,
                    "timestamp_ns": 1_000_000_000,
                    "frame_id": 1,
                })

            rows = analysis._typed_rows(path, required)

        self.assertEqual(len(rows), 1)
        for column in distribution_columns:
            self.assertIsNone(rows[0][column])

    def test_loader_does_not_materialize_landmark_event_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._write_minimal_run(root)
            path = run_root / "diagnostics" / "vio_diag_landmark_events.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                fields = next(csv.reader(stream))
            row = {field: 0 for field in fields}
            row.update({
                "schema_version": 1,
                "event_sequence": 1,
                "event_timestamp_ns": 1_000_000_000,
                "event_frame_id": 7,
                "graph_role": "realtime",
                "event_type": "observation_removed",
                "reason": "gp3p_outlier",
            })
            with path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)

            loaded = analysis.load_diagnostic_run(root, "seq-a", "run1")

        self.assertEqual(loaded.landmark_events, [])

    def test_aggregate_frame_metrics_joins_gp3p_geometry_and_lifecycle(self):
        diagnostic_run = analysis.DiagnosticRun(
            sequence="seq-a",
            run="run1",
            root=Path("."),
            manifest={},
            metadata={"camera_count": "1"},
            frames=[{
                "frame_id": 7,
                "timestamp_ns": 1_000_000_000,
                "keypoints_cam0": 100,
                "grid_fraction_cam0": 0.5,
                "hull_fraction_cam0": 0.4,
                "projected_eligible_cam0": 80,
                "descriptor_comparisons_cam0": 70,
                "descriptor_candidates_below_threshold_cam0": 60,
                "accepted_initialised_cam0": 20,
                "accepted_uninitialised_cam0": 5,
                "best_map_descriptor_distance_median_cam0": 21.0,
                "accepted_descriptor_distance_median": 18.0,
                "predicted_reprojection_error_px_median": 3.0,
                "active_initialised_landmarks": 90,
                "active_uninitialised_landmarks": 10,
                "landmark_births": 2,
                "observations_added": 25,
                "observations_removed_reason_0": 2,
                "observations_removed_reason_1": 3,
                "observations_removed_reason_2": 5,
                "observations_removed_reason_3": 7,
            }],
            triangulation=[{
                "frame_id": 7,
                "source": "temporal_motion_stereo",
                "attempts": 10,
                "parallel": 4,
                "initialisable": 3,
                "ray_angle_rad_p10": 0.01,
                "baseline_m_p10": 0.02,
            }],
            initialisation=[],
            ransac=[{
                "frame_id": 7,
                "status": "threshold_rejected",
                "inlier_ratio": 0.4,
                "start_to_model_rotation_rad": 0.2,
                "start_to_model_translation_m": 0.3,
                "pre_invocation_to_model_rotation_rad": 0.1,
                "pre_invocation_to_model_translation_m": 0.2,
            }],
            landmark_events=[],
        )

        rows = analysis.aggregate_frame_metrics(diagnostic_run)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["accepted_map_matches"], 25)
        self.assertEqual(rows[0]["gp3p_failure_count"], 1)
        self.assertEqual(rows[0]["visual_observation_removals"], 17)
        self.assertAlmostEqual(rows[0]["temporal_parallel_fraction"], 0.4)
        self.assertAlmostEqual(rows[0]["temporal_ray_angle_p10_rad"], 0.01)

    def test_gp3p_onset_scan_includes_failures_during_angular_event(self):
        rows = [
            {"time_s": time, "gp3p_failure_count": value}
            for time, value in zip(
                np.arange(-1.0, 1.1, 0.1),
                [0] * 10 + [1] + [0] * 10,
            )
        ]
        event = analysis.AngularEvent(0.0, 0.5, 5.0, 1.0)

        onset = analysis.gp3p_failure_onset(rows, event)

        self.assertAlmostEqual(onset, 0.0)

    def test_gp3p_onset_does_not_scan_late_recovery_window(self):
        rows = [
            {"time_s": time, "gp3p_failure_count": int(time >= 1.1)}
            for time in np.arange(-1.0, 2.1, 0.1)
        ]
        event = analysis.AngularEvent(0.0, 0.5, 5.0, 1.0)

        self.assertIsNone(analysis.gp3p_failure_onset(rows, event))

    def test_mediation_model_reports_insufficient_rows(self):
        result = analysis.fit_mediation_model(
            [{"sequence": "a", "angular_integral": 1.0}],
            mediator="mediator_delta",
            outcome="gp3p_outcome",
        )
        self.assertEqual(result["status"], "insufficient_model_rows")

    def test_event_id_includes_experiment_identity(self):
        run = analysis.DiagnosticRun(
            sequence="seq-a",
            run="run1",
            root=Path("."),
            manifest={"experiment_id": "seq-a-imu-offset-p5ms"},
            metadata={"camera_count": "1"},
            frames=[],
            triangulation=[],
            initialisation=[],
            ransac=[],
            landmark_events=[],
        )
        events, _ = analysis.build_event_metrics(
            run,
            [],
            [analysis.AngularEvent(1.0, 1.1, 4.0, 0.4)],
        )
        self.assertEqual(
            events[0]["event_id"],
            "seq-a-imu-offset-p5ms-run1-event000",
        )

    def test_camera_matching_applies_image_delay_before_nearest_join(self):
        raw = np.asarray([1_000_000_000, 1_100_000_000], dtype=np.int64)
        frames = np.asarray([975_130_000, 1_075_130_000], dtype=np.int64)

        matches = analysis.match_corrected_camera_frames(
            frames, raw, image_delay_s=0.02487
        )

        np.testing.assert_array_equal(matches, [0, 1])

    def test_imu_metrics_use_previous_to_current_half_open_interval(self):
        frame_ns = np.asarray(
            [1_000_000_000, 1_100_000_000, 1_200_000_000], dtype=np.int64
        )
        imu_ns = np.asarray(
            [1_000_000_000, 1_050_000_000, 1_100_000_000, 1_150_000_000],
            dtype=np.int64,
        )
        gyro = np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )

        metrics = analysis.compute_frame_interval_imu_metrics(
            frame_ns, imu_ns, gyro
        )

        self.assertIsNone(metrics[0]["imu_gyro_max_radps"])
        self.assertEqual(metrics[1]["imu_sample_count"], 2)
        self.assertAlmostEqual(metrics[1]["imu_gyro_max_radps"], 1.0)
        self.assertEqual(metrics[2]["imu_sample_count"], 2)
        self.assertAlmostEqual(metrics[2]["imu_gyro_max_radps"], 2.0)

    def test_sensor_root_supports_root_and_nested_mav0_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root_layout = root / "root-layout"
            nested_layout = root / "nested-layout"
            (root_layout / "cam0").mkdir(parents=True)
            (root_layout / "imu0").mkdir()
            (nested_layout / "mav0" / "cam0").mkdir(parents=True)
            (nested_layout / "mav0" / "imu0").mkdir()

            self.assertEqual(analysis._resolve_sensor_root(root_layout), root_layout)
            self.assertEqual(
                analysis._resolve_sensor_root(nested_layout), nested_layout / "mav0"
            )

    def test_event_outcomes_are_paired_against_matched_control_windows(self):
        run = analysis.DiagnosticRun(
            sequence="seq-a",
            run="run1",
            root=Path("."),
            manifest={},
            metadata={"camera_count": "1"},
            frames=[],
            triangulation=[],
            initialisation=[],
            ransac=[],
            landmark_events=[],
        )
        rows = []
        for time_s in np.arange(0.0, 55.0, 0.5):
            row = {
                "sequence": "seq-a",
                "run": "run1",
                "time_s": float(time_s),
                "timestamp_ns": int(time_s * 1e9),
                "active_initialised_landmarks": 100.0,
                "accepted_map_matches": 50.0,
                "mocap_body_translation_m": 0.01,
                "mocap_body_rotation_rad": 0.02,
                "image_laplacian_variance": 20.0,
                "keypoints_total": 200.0,
                "gp3p_failure_count": 0.0,
                "gp3p_invocations": 0.0,
            }
            if 20.0 <= time_s <= 21.5:
                row["gp3p_failure_count"] = 3.0
                row["gp3p_invocations"] = 4.0
            elif 40.0 <= time_s <= 41.5:
                row["gp3p_failure_count"] = 1.0
                row["gp3p_invocations"] = 4.0
            if 21.5 < time_s <= 23.0:
                row["gp3p_failure_count"] = 2.0
                row["gp3p_invocations"] = 4.0
            elif 41.5 < time_s <= 43.0:
                row["gp3p_failure_count"] = 1.0
                row["gp3p_invocations"] = 4.0
            if 23.0 <= time_s <= 26.0:
                row["active_initialised_landmarks"] = 70.0
            elif 43.0 <= time_s <= 46.0:
                row["active_initialised_landmarks"] = 90.0
            rows.append(row)

        with patch.object(
            analysis,
            "select_matched_controls",
            return_value=[{"start_s": 40.0, "end_s": 41.0}],
        ):
            event_rows, _ = analysis.build_event_metrics(
                run,
                rows,
                [analysis.AngularEvent(20.0, 21.0, 5.0, 1.0)],
            )

        event = event_rows[0]
        self.assertAlmostEqual(event["gp3p_outcome"], 0.75)
        self.assertAlmostEqual(event["gp3p_control_outcome"], 0.25)
        self.assertAlmostEqual(event["gp3p_outcome_paired"], 0.50)
        self.assertAlmostEqual(event["gp3p_post_outcome"], 0.50)
        self.assertAlmostEqual(event["gp3p_post_control_outcome"], 0.25)
        self.assertAlmostEqual(event["gp3p_post_outcome_paired"], 0.25)
        self.assertAlmostEqual(event["map_support_outcome"], -30.0)
        self.assertAlmostEqual(event["map_support_control_outcome"], -10.0)
        self.assertAlmostEqual(event["map_support_outcome_paired"], -20.0)

    def test_mediation_models_use_paired_outcomes(self):
        captured = []

        def capture(rows, *, mediator, outcome, samples):
            if rows:
                captured.append((outcome, rows[0][outcome]))
            return {"status": "captured"}

        event = {
            "sequence": "seq-a",
            "control_status": "matched",
            "angular_integral": 1.0,
            "accepted_map_matches_paired_delta": -2.0,
            "gp3p_outcome": 0.9,
            "gp3p_outcome_paired": 0.4,
            "map_support_outcome": -30.0,
            "map_support_outcome_paired": -20.0,
            "pre_map_support": 100.0,
            "mocap_translation": 0.01,
            "mocap_rotation": 0.5,
            "pre_image_sharpness": 20.0,
        }
        with patch.object(analysis, "bootstrap_mediation_model", side_effect=capture):
            analysis.build_mediation_rows([event], bootstrap_samples=5)

        self.assertIn(("gp3p_outcome", 0.4), captured)
        self.assertIn(("map_support_outcome", -20.0), captured)

    def test_analyze_discovers_manifest_sequence_under_experiment_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_minimal_run(
                root,
                result_id="seq-a-imu-offset-p5ms",
                sequence="seq-a",
                experiment_id="seq-a-imu-offset-p5ms",
                intervention="imu_time_offset_ns",
                intervention_value="5000000",
            )
            output = root / "output"
            analysis.analyze_runs(argparse.Namespace(
                diagnostics_root=root,
                data_root=root / "unused-data",
                output=output,
                sequences=["seq-a"],
                bootstrap_samples=5,
            ))

            with (
                output / "tables" / "causal_diagnostics_coverage.csv"
            ).open(newline="", encoding="utf-8") as stream:
                coverage = list(csv.DictReader(stream))
            self.assertEqual(len(coverage), 1)
            self.assertEqual(
                coverage[0]["experiment_id"], "seq-a-imu-offset-p5ms"
            )
            self.assertEqual(coverage[0]["sequence"], "seq-a")
            self.assertEqual(coverage[0]["intervention"], "imu_time_offset_ns")
            expected = {
                "causal_frame_metrics.csv",
                "causal_diagnostics_coverage.csv",
                "causal_event_metrics.csv",
                "impulse_mediator_recovery.csv",
                "causal_mediation_models.csv",
            }
            self.assertEqual(
                {path.name for path in (output / "tables").iterdir()}, expected
            )
            self.assertGreater(
                (
                    output
                    / "figures"
                    / "impulse_mediator_timeline.png"
                ).stat().st_size,
                0,
            )

    def _write_minimal_run(
        self,
        root: Path,
        *,
        result_id: str = "seq-a",
        sequence: str = "seq-a",
        experiment_id: str | None = None,
        intervention: str = "baseline",
        intervention_value: str = "none",
    ) -> Path:
        run_root = root / result_id / "run1"
        diagnostics = run_root / "diagnostics"
        diagnostics.mkdir(parents=True)
        (diagnostics / ".vio_diagnostics.complete").touch()
        config = root / "config.yaml"
        config.write_text("image_delay: 0.02487\n", encoding="utf-8")
        mocap = root / "mocap.log"
        mocap.touch()
        (run_root / "run_manifest.json").write_text(
            json.dumps({
                "experiment_id": experiment_id or sequence,
                "sequence": sequence,
                "run": "run1",
                "intervention": intervention,
                "intervention_value": intervention_value,
                "mocap_path": str(mocap),
                "config_path": str(config),
                "image_delay_s": 0.02487,
            }),
            encoding="utf-8",
        )
        with (diagnostics / "vio_diag_metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["schema_version", "key", "value"])
            writer.writerow([1, "camera_count", "1"])
            writer.writerow([1, "run_complete", "true"])
            writer.writerow([1, "writer_failed", "false"])
        for filename, required in analysis.REQUIRED_COLUMNS.items():
            with (diagnostics / filename).open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                csv.writer(stream).writerow(sorted(required))
        return run_root


if __name__ == "__main__":
    unittest.main()
