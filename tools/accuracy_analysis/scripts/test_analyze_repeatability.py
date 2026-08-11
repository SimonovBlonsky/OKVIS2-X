#!/usr/bin/env python3

import math
import tempfile
import unittest
import warnings
from dataclasses import FrozenInstanceError, fields
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import analyze_repeatability as analysis


class AnalysisNumericsTest(unittest.TestCase):

    def test_write_csv_allows_header_only_output_with_explicit_fieldnames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"

            analysis.write_csv(path, [], fieldnames=["sequence", "camera"])

            self.assertEqual(path.read_text(encoding="utf-8"), "sequence,camera\n")

    def test_write_csv_rejects_empty_rows_without_fieldnames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"

            with self.assertRaisesRegex(ValueError, "empty CSV"):
                analysis.write_csv(path, [])

    def test_camera_gap_event_fields_match_generated_event_schema(self):
        timestamps = np.asarray([0.0, 0.01, 0.03, 0.04])
        filenames = np.asarray(["0.png", "1.png", "2.png", "3.png"])

        with patch.object(
            analysis,
            "load_camera_index",
            return_value=(timestamps, filenames),
        ):
            events = analysis.camera_gap_events(
                Path("/dataset"), "control-a", start=0.0, image_delay=0.0
            )

        self.assertTrue(events)
        self.assertEqual(tuple(events[0]), analysis.CAMERA_GAP_EVENT_FIELDS)

    def test_grouped_bar_layout_centers_three_groups_within_category(self):
        width, offsets = analysis.grouped_bar_layout(3)

        self.assertGreater(width, 0.0)
        self.assertEqual(len(offsets), 3)
        self.assertAlmostEqual(offsets[0], -offsets[2])
        self.assertAlmostEqual(offsets[1], 0.0)
        for offset in offsets:
            with self.subTest(offset=offset):
                self.assertGreaterEqual(offset - width / 2.0, -0.4)
                self.assertLessEqual(offset + width / 2.0, 0.4)

    def test_grouped_bar_layout_rejects_nonpositive_group_count(self):
        for group_count in (0, -1):
            with self.subTest(group_count=group_count):
                with self.assertRaises(ValueError):
                    analysis.grouped_bar_layout(group_count)

    def test_sequence_spec_matches_approved_immutable_contract(self):
        self.assertEqual(
            tuple(field.name for field in fields(analysis.SequenceSpec)),
            ("name", "role", "dataset", "result_dir", "mocap", "color"),
        )
        spec = analysis.SequenceSpec(
            name="control-a",
            role="control",
            dataset=Path("/datasets/control-a"),
            result_dir=Path("/results/control-a"),
            mocap=Path("/mocap/control-a.log"),
            color="#123456",
        )

        with self.assertRaises(FrozenInstanceError):
            spec.role = "target"

    def test_default_inputs_include_second_control_and_correct_target_mocap(self):
        self.assertEqual(analysis.CONTROL_SEQUENCE_184027, "20260803-184027")
        self.assertEqual(
            analysis.CONTROL_DATASET_184027,
            Path("/home/chenguyuan/data/20260803/20260803-184027_euroc"),
        )
        self.assertEqual(
            analysis.DEFAULT_RESULTS_ROOT.parent / analysis.CONTROL_SEQUENCE_184027,
            Path(
                "/home/chenguyuan/code/okvis_ws/src/OKVIS2-X/"
                "workspace/ego2_results/20260803-184027"
            ),
        )
        self.assertEqual(
            analysis.DEFAULT_MOCAP,
            Path(
                "/home/chenguyuan/data/20260803/mocap_ego2_20260803/"
                "mocap_20260803_184540.log"
            ),
        )

    def test_build_sequence_specs_uses_all_cli_override_paths(self):
        arguments = SimpleNamespace(
            results_root=Path("/sentinel/results/20260803-184537"),
            control_dataset_183537=Path("/sentinel/data/control-183537"),
            control_dataset_184027=Path("/sentinel/data/control-184027"),
            dataset=Path("/sentinel/data/target"),
            mocap=Path("/sentinel/mocap/target.log"),
        )

        with patch.object(
            analysis,
            "unique_mocap_log",
            side_effect=lambda result_dir: result_dir
            / f"mocap_{result_dir.name}.log",
        ):
            specs = analysis.build_sequence_specs(arguments)

        self.assertEqual(
            [spec.result_dir for spec in specs],
            [
                Path("/sentinel/results/20260803-183537"),
                Path("/sentinel/results/20260803-184027"),
                Path("/sentinel/results/20260803-184537/bak3"),
            ],
        )
        self.assertEqual(
            [spec.dataset for spec in specs],
            [
                arguments.control_dataset_183537,
                arguments.control_dataset_184027,
                arguments.dataset,
            ],
        )
        self.assertEqual(specs[0].mocap.parent, specs[0].result_dir)
        self.assertEqual(specs[1].mocap.parent, specs[1].result_dir)
        self.assertEqual(specs[2].mocap, arguments.mocap)
        self.assertEqual([spec.role for spec in specs], ["control", "control", "target"])

    def test_parse_arguments_accepts_both_control_dataset_overrides(self):
        with patch(
            "sys.argv",
            [
                "analyze_repeatability.py",
                "--control-dataset-183537",
                "/sentinel/data/control-183537",
                "--control-dataset-184027",
                "/sentinel/data/control-184027",
            ],
        ):
            arguments = analysis.parse_arguments()

        self.assertEqual(
            arguments.control_dataset_183537,
            Path("/sentinel/data/control-183537"),
        )
        self.assertEqual(
            arguments.control_dataset_184027,
            Path("/sentinel/data/control-184027"),
        )

    def test_reference_dataset_cli_alias_targets_first_control(self):
        with patch(
            "sys.argv",
            [
                "analyze_repeatability.py",
                "--reference-dataset",
                "/sentinel/data/control-183537",
            ],
        ):
            arguments = analysis.parse_arguments()

        self.assertEqual(
            arguments.control_dataset_183537,
            Path("/sentinel/data/control-183537"),
        )

    def test_unique_mocap_log_requires_exactly_one_result_log(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            with self.assertRaisesRegex(ValueError, "found 0"):
                analysis.unique_mocap_log(result_dir)

            expected = result_dir / "mocap_first.log"
            expected.touch()
            self.assertEqual(analysis.unique_mocap_log(result_dir), expected)

            (result_dir / "mocap_second.log").touch()
            with self.assertRaisesRegex(ValueError, "found 2"):
                analysis.unique_mocap_log(result_dir)

    def test_analyze_sequence_spec_attaches_role_color_and_spec(self):
        spec = analysis.SequenceSpec(
            name="control-a",
            role="control",
            dataset=Path("/datasets/control-a"),
            result_dir=Path("/results/control-a"),
            mocap=Path("/mocap/control-a.log"),
            color="#123456",
        )
        analyzed = {"sequence": spec.name, "summary": {}}

        with patch.object(
            analysis, "analyze_sequence", return_value=analyzed
        ) as analyze_sequence:
            context = analysis.analyze_sequence_spec(
                spec, samples_per_camera=17, image_delay=0.003
            )

        analyze_sequence.assert_called_once_with(
            spec.name,
            spec.dataset,
            spec.result_dir,
            spec.mocap,
            17,
            0.003,
        )
        self.assertEqual(context["role"], "control")
        self.assertEqual(context["color"], "#123456")
        self.assertIs(context["spec"], spec)

    def test_reference_analysis_summarizes_each_control_spec(self):
        specs = [
            analysis.SequenceSpec(
                name=name,
                role="control",
                dataset=Path(f"/datasets/{name}"),
                result_dir=Path(f"/results/{name}"),
                mocap=Path(f"/mocap/{name}.log"),
                color=color,
            )
            for name, color in (("control-a", "#123456"), ("control-b", "#654321"))
        ]
        evaluations = [
            SimpleNamespace(timestamps=np.asarray([1.0, 2.0])),
            SimpleNamespace(timestamps=np.asarray([3.0, 4.0])),
        ]

        with (
            patch.object(analysis, "load_okvis_trajectory") as load_trajectory,
            patch.object(analysis, "load_mocap_trajectory") as load_mocap,
            patch.object(analysis, "evaluate_ape", side_effect=evaluations),
            patch.object(analysis, "_motion_from_mocap", return_value={}),
            patch.object(
                analysis,
                "summarize_motion",
                return_value={
                    "source": "mocap",
                    "samples": 2,
                    "duration_s": 1.0,
                    "linear_speed_mps_median": 0.5,
                },
            ),
            patch.object(
                analysis,
                "summarize_stage",
                side_effect=lambda name, stage, trajectory, evaluation: {
                    "run": name,
                    "stage": stage,
                    "ape_rmse_m": 0.001,
                },
            ),
        ):
            rows = analysis._reference_analysis(specs)

        self.assertEqual([row["sequence"] for row in rows], ["control-a", "control-b"])
        self.assertTrue(all(row["mocap_linear_speed_mps_median"] == 0.5 for row in rows))
        self.assertEqual(
            [call.args[0] for call in load_trajectory.call_args_list],
            [spec.result_dir / analysis.STAGE_FILES["final-ba"] for spec in specs],
        )
        self.assertEqual(
            [call.args[0] for call in load_mocap.call_args_list],
            [spec.mocap for spec in specs],
        )

    def test_partition_sequence_contexts_is_independent_of_input_order(self):
        contexts = [
            {"sequence": "control-a", "role": "control"},
            {"sequence": "control-b", "role": "control"},
            {"sequence": "target", "role": "target"},
        ]

        for ordering in permutations(contexts):
            with self.subTest(order=[context["sequence"] for context in ordering]):
                controls, target = analysis.partition_sequence_contexts(list(ordering))

                self.assertEqual(
                    {context["sequence"] for context in controls},
                    {"control-a", "control-b"},
                )
                self.assertEqual(target["sequence"], "target")

    def test_partition_sequence_contexts_rejects_missing_controls(self):
        with self.assertRaisesRegex(ValueError, "control"):
            analysis.partition_sequence_contexts(
                [{"sequence": "target", "role": "target"}]
            )

    def test_partition_sequence_contexts_rejects_missing_target(self):
        contexts = [
            {"sequence": "control-a", "role": "control"},
            {"sequence": "control-b", "role": "control"},
        ]

        with self.assertRaisesRegex(ValueError, "target"):
            analysis.partition_sequence_contexts(contexts)

    def test_partition_sequence_contexts_rejects_multiple_targets(self):
        contexts = [
            {"sequence": "control", "role": "control"},
            {"sequence": "target-a", "role": "target"},
            {"sequence": "target-b", "role": "target"},
        ]

        with self.assertRaisesRegex(ValueError, "target"):
            analysis.partition_sequence_contexts(contexts)

    def test_partition_sequence_contexts_rejects_unknown_role(self):
        contexts = [
            {"sequence": "control", "role": "control"},
            {"sequence": "control-typo", "role": "controle"},
            {"sequence": "target", "role": "target"},
        ]

        with self.assertRaises(ValueError) as caught:
            analysis.partition_sequence_contexts(contexts)

        self.assertIn("control-typo", str(caught.exception))
        self.assertIn("controle", str(caught.exception))

    def test_control_envelope_status_uses_closed_control_interval(self):
        controls = [3.0, 1.0, 2.0]

        self.assertEqual(analysis.control_envelope_status(controls, 0.5), "below")
        self.assertEqual(analysis.control_envelope_status(controls, 1.0), "within")
        self.assertEqual(analysis.control_envelope_status(controls, 2.0), "within")
        self.assertEqual(analysis.control_envelope_status(controls, 3.0), "within")
        self.assertEqual(analysis.control_envelope_status(controls, 4.0), "above")

    def test_control_envelope_status_rejects_empty_or_nonfinite_values(self):
        invalid_cases = (
            ([], 1.0),
            ([1.0, float("nan")], 1.0),
            ([1.0, 2.0], float("inf")),
        )

        for controls, target in invalid_cases:
            with self.subTest(controls=controls, target=target):
                with self.assertRaisesRegex(ValueError, "finite.*control|target"):
                    analysis.control_envelope_status(controls, target)

    def test_control_target_metric_rows_are_order_invariant_and_do_not_duplicate_target(self):
        metric_names = (
            "ape_rmse_m",
            "observations_per_landmark",
            "distinct_states_per_landmark_mean",
            "landmark_time_span_median_s",
        )
        contexts = [
            {
                "sequence": "20260803-184537/bak3",
                "role": "target",
                "color": "#target-context",
                "summary": dict.fromkeys(metric_names, 999.0),
            },
            {
                "sequence": "20260803-184027",
                "role": "control",
                "color": "#control-b",
                "summary": {
                    "ape_rmse_m": 0.008,
                    "observations_per_landmark": 9.0,
                    "distinct_states_per_landmark_mean": 6.0,
                    "landmark_time_span_median_s": 140.0,
                },
            },
            {
                "sequence": "20260803-183537",
                "role": "control",
                "color": "#control-a",
                "summary": {
                    "ape_rmse_m": 0.007,
                    "observations_per_landmark": 8.5,
                    "distinct_states_per_landmark_mean": 5.5,
                    "landmark_time_span_median_s": 60.0,
                },
            },
        ]
        run_rows = [
            {
                "run": run,
                "ape_rmse_m": 0.01 * index,
                "observations_per_landmark": 4.0 + index,
                "distinct_states_per_landmark_mean": 3.0 + index / 10.0,
                "landmark_time_span_median_s": 0.25 * index,
            }
            for index, run in enumerate(("bak4", "bak2", "bak1", "bak3"), 1)
        ]

        baseline = analysis.control_target_metric_rows(contexts, run_rows)
        for ordering in permutations(contexts):
            with self.subTest(order=[context["sequence"] for context in ordering]):
                self.assertEqual(
                    analysis.control_target_metric_rows(list(ordering), run_rows),
                    baseline,
                )

        self.assertEqual(
            [row["label"] for row in baseline],
            [
                "control\n183537",
                "control\n184027",
                "target\nbak1",
                "target\nbak2",
                "target\nbak3",
                "target\nbak4",
            ],
        )
        self.assertEqual(sum(row["name"] == "bak3" for row in baseline), 1)
        self.assertEqual([row["kind"] for row in baseline], ["control"] * 2 + ["target"] * 4)
        self.assertEqual(baseline[0]["marker"], "D")
        self.assertEqual(baseline[0]["color"], "#control-a")
        self.assertEqual(baseline[2]["color"], analysis.RUN_COLORS["bak1"])
        self.assertAlmostEqual(baseline[0]["ape_rmse_mm"], 7.0)

    def test_control_target_metric_rows_require_two_controls_and_bak1_through_bak4(self):
        summary = {
            "ape_rmse_m": 0.01,
            "observations_per_landmark": 5.0,
            "distinct_states_per_landmark_mean": 3.0,
            "landmark_time_span_median_s": 0.5,
        }
        contexts = [
            {"sequence": "control-a", "role": "control", "color": "a", "summary": summary},
            {"sequence": "control-b", "role": "control", "color": "b", "summary": summary},
            {"sequence": "target", "role": "target", "color": "t", "summary": summary},
        ]
        runs = [{"run": name, **summary} for name in ("bak1", "bak2", "bak3", "bak4")]

        with self.assertRaisesRegex(ValueError, "exactly two controls"):
            analysis.control_target_metric_rows(contexts[1:], runs)
        with self.assertRaisesRegex(ValueError, "bak1.*bak4"):
            analysis.control_target_metric_rows(
                contexts, runs[:-1] + [{"run": "bak5", **summary}]
            )

    def test_plot_control_envelope_and_target_runs_writes_decodable_figure(self):
        summary = {
            "ape_rmse_m": 0.01,
            "observations_per_landmark": 5.0,
            "distinct_states_per_landmark_mean": 3.0,
            "landmark_time_span_median_s": 0.5,
        }
        contexts = [
            {"sequence": "control-a", "role": "control", "color": "#123456", "summary": summary},
            {"sequence": "control-b", "role": "control", "color": "#654321", "summary": summary},
            {"sequence": "target", "role": "target", "color": "#abcdef", "summary": summary},
        ]
        runs = [{"run": name, **summary} for name in ("bak1", "bak2", "bak3", "bak4")]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            analysis._plot_control_envelope_and_target_runs(output, contexts, runs)
            figure_path = output / "15_control_envelope_and_target_runs.png"

            self.assertGreater(figure_path.stat().st_size, 0)
            from PIL import Image

            with Image.open(figure_path) as figure:
                figure.verify()
                self.assertGreater(figure.width, 0)
                self.assertGreater(figure.height, 0)

    def test_sync_percentage_axis_limits_validate_and_pad_all_values(self):
        values = [88.48, 97.25, 99.90]

        lower, upper = analysis.sync_percentage_axis_limits(values)

        self.assertGreaterEqual(lower, 0.0)
        self.assertLess(lower, min(values))
        self.assertGreater(upper, max(values))
        for invalid_values in (
            [],
            [float("nan")],
            [float("inf")],
            [-0.01, 99.0],
            [88.0, 100.01],
        ):
            with self.subTest(values=invalid_values):
                with self.assertRaisesRegex(ValueError, "finite|0.*100|non-empty"):
                    analysis.sync_percentage_axis_limits(invalid_values)

    def test_plot_sequence_sensor_timing_keeps_all_sync_bars_and_labels_visible(self):
        contexts = []
        for index, (exact_sync, one_to_one_sync) in enumerate(
            ((0.8848, 0.9710), (0.9750, 0.9920), (0.9950, 0.9990))
        ):
            contexts.append(
                {
                    "sequence": f"sequence-{index}",
                    "color": ("#147d92", "#7357a5", "#b3261e")[index],
                    "imu_timeseries": {
                        "timestamps": np.asarray([0.0, 0.005, 0.010, 0.015]),
                        "gyro_norm": np.asarray([0.1, 0.2, 0.3, 0.4]),
                    },
                    "camera_rows": [
                        {"camera": f"cam{camera}", "gap_count_over_1_5x": camera}
                        for camera in range(4)
                    ],
                    "camera_aggregate": {
                        "exact_sync_fraction": exact_sync,
                        "one_to_one_sync_fraction": one_to_one_sync,
                    },
                }
            )
        captured_figures = []

        with patch.object(
            analysis,
            "_save_figure",
            side_effect=lambda figure, _path: captured_figures.append(figure),
        ):
            analysis._plot_sequence_sensor_timing(Path("/unused"), contexts)

        self.assertEqual(len(captured_figures), 1)
        figure = captured_figures[0]
        sync_axis = figure.axes[3]
        lower, upper = sync_axis.get_ylim()
        bar_tops = [bar.get_y() + bar.get_height() for bar in sync_axis.patches]
        exact_bar_tops = bar_tops[::2]
        self.assertEqual(len(exact_bar_tops), 3)
        self.assertEqual(
            [round(value, 2) for value in exact_bar_tops],
            [88.48, 97.50, 99.50],
        )
        self.assertTrue(all(lower <= value <= upper for value in bar_tops))

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis_bounds = sync_axis.get_window_extent(renderer)
        self.assertEqual(len(sync_axis.texts), 6)
        for label in sync_axis.texts:
            with self.subTest(label=label.get_text()):
                self.assertGreaterEqual(label.xy[1], lower)
                self.assertLessEqual(label.xy[1], upper)
                label_bounds = label.get_window_extent(renderer)
                self.assertGreaterEqual(label_bounds.y0, axis_bounds.y0)
                self.assertLessEqual(label_bounds.y1, axis_bounds.y1)
        analysis.plt.close(figure)

    def test_plot_sequence_image_quality_omits_top_left_repeated_legend(self):
        contexts = []
        for context_index, (sequence, color) in enumerate(
            (("control-a", "#123456"), ("control-b", "#654321"))
        ):
            camera_rows = []
            quality_rows = []
            for camera_index in range(4):
                camera = f"cam{camera_index}"
                camera_rows.append(
                    {
                        "camera": camera,
                        "sharpness_median": 768.0 + 10.0 * context_index + camera_index,
                        "sharpness_p5": 400.0 + camera_index,
                        "fraction_below_control_p5": 0.05 * (camera_index + 1),
                        "intensity_mean_median": 100.0 + camera_index,
                        "intensity_std_median": 20.0 + camera_index,
                    }
                )
                for angular_speed, sharpness in (
                    (0.25, 800.0),
                    (0.75, 700.0),
                    (1.50, 600.0),
                    (2.50, 500.0),
                ):
                    quality_rows.append(
                        {
                            "camera": camera,
                            "mocap_angular_speed_radps": angular_speed,
                            "laplacian_variance": sharpness + context_index,
                        }
                    )
            contexts.append(
                {
                    "sequence": sequence,
                    "color": color,
                    "camera_rows": camera_rows,
                    "quality_rows": quality_rows,
                }
            )
        captured_figures = []

        with patch.object(
            analysis,
            "_save_figure",
            side_effect=lambda figure, _path: captured_figures.append(figure),
        ):
            analysis._plot_sequence_image_quality(Path("/unused"), contexts)

        self.assertEqual(len(captured_figures), 1)
        self.assertIsNone(captured_figures[0].axes[0].get_legend())
        self.assertIsNotNone(captured_figures[0].axes[1].get_legend())
        analysis.plt.close(captured_figures[0])

    def test_control_metric_axis_scale_validates_values_and_preserves_zero_span(self):
        self.assertEqual(
            analysis.control_metric_axis_scale("ape_rmse_mm", [1.0, 2.0]),
            "log",
        )
        self.assertEqual(
            analysis.control_metric_axis_scale(
                "landmark_time_span_median_s", [0.0, 1.0]
            ),
            "linear",
        )
        self.assertEqual(
            analysis.control_metric_axis_scale(
                "landmark_time_span_median_s", [0.5, 1.0]
            ),
            "log",
        )
        self.assertEqual(
            analysis.control_metric_axis_scale(
                "observations_per_landmark", [0.0, 1.0]
            ),
            "linear",
        )
        for field, values, message in (
            ("ape_rmse_mm", [0.0, 1.0], "positive"),
            ("ape_rmse_mm", [-1.0, 1.0], "positive"),
            ("landmark_time_span_median_s", [-1.0, 1.0], "nonnegative"),
            ("observations_per_landmark", [float("nan"), 1.0], "finite"),
            ("distinct_states_per_landmark_mean", [1.0, float("inf")], "finite"),
        ):
            with self.subTest(field=field, values=values):
                with self.assertRaisesRegex(ValueError, message):
                    analysis.control_metric_axis_scale(field, values)

    def test_plot_control_envelope_keeps_zero_span_point_and_annotation_visible(self):
        summary = {
            "ape_rmse_m": 0.01,
            "observations_per_landmark": 5.0,
            "distinct_states_per_landmark_mean": 3.0,
            "landmark_time_span_median_s": 0.0,
        }
        contexts = [
            {"sequence": "control-a", "role": "control", "color": "#123456", "summary": summary},
            {"sequence": "control-b", "role": "control", "color": "#654321", "summary": summary},
            {"sequence": "target", "role": "target", "color": "#abcdef", "summary": summary},
        ]
        runs = [{"run": name, **summary} for name in ("bak1", "bak2", "bak3", "bak4")]
        captured_figures = []

        with patch.object(
            analysis,
            "_save_figure",
            side_effect=lambda figure, _path: captured_figures.append(figure),
        ):
            analysis._plot_control_envelope_and_target_runs(
                Path("/unused"), contexts, runs
            )

        self.assertEqual(len(captured_figures), 1)
        span_axis = captured_figures[0].axes[3]
        self.assertEqual(span_axis.get_yscale(), "linear")
        self.assertTrue(
            any(text.get_text() == "0.000" and text.get_visible() for text in span_axis.texts)
        )
        self.assertTrue(
            any(
                np.any(np.isclose(collection.get_offsets()[:, 1], 0.0))
                for collection in span_axis.collections
                if len(collection.get_offsets())
            )
        )
        analysis.plt.close(captured_figures[0])

    def test_repeatability_statistics_reports_ranges_fold_and_descriptive_correlations(self):
        run_rows = []
        for index, (observations_per_landmark, distinct_states, span) in enumerate(
            ((4.0, 1.0, 1.0), (3.0, 2.0, 3.0), (2.0, 3.0, 2.0), (1.0, 4.0, 4.0)),
            1,
        ):
            run_rows.append(
                {
                    "run": f"bak{index}",
                    "ape_rmse_m": float(index),
                    "states": 50 + index,
                    "landmarks": 100 + 10 * index,
                    "observations": 500 + 100 * index,
                    "observations_per_landmark": observations_per_landmark,
                    "distinct_states_per_landmark_mean": distinct_states,
                    "landmark_time_span_median_s": span,
                }
            )
        pairwise_rows = [
            {"aligned_rmse_m": 0.4},
            {"aligned_rmse_m": 0.1},
            {"aligned_rmse_m": 0.3},
        ]

        statistics = analysis.repeatability_statistics(run_rows, pairwise_rows)

        self.assertEqual(statistics["n_runs"], 4)
        self.assertEqual(statistics["ape_rmse_min_m"], 1.0)
        self.assertEqual(statistics["ape_rmse_max_m"], 4.0)
        self.assertEqual(statistics["ape_rmse_fold"], 4.0)
        self.assertEqual(statistics["pairwise_aligned_rmse_min_m"], 0.1)
        self.assertEqual(statistics["pairwise_aligned_rmse_max_m"], 0.4)
        self.assertEqual((statistics["states_min"], statistics["states_max"]), (51.0, 54.0))
        self.assertEqual((statistics["landmarks_min"], statistics["landmarks_max"]), (110.0, 140.0))
        self.assertEqual((statistics["observations_min"], statistics["observations_max"]), (600.0, 900.0))
        self.assertEqual(
            (
                statistics["observations_per_landmark_min"],
                statistics["observations_per_landmark_max"],
            ),
            (1.0, 4.0),
        )
        self.assertEqual(
            (
                statistics["distinct_states_per_landmark_mean_min"],
                statistics["distinct_states_per_landmark_mean_max"],
            ),
            (1.0, 4.0),
        )
        self.assertEqual(
            (
                statistics["landmark_time_span_median_min_s"],
                statistics["landmark_time_span_median_max_s"],
            ),
            (1.0, 4.0),
        )
        self.assertAlmostEqual(statistics["ape_observations_per_landmark_spearman"], -1.0)
        self.assertAlmostEqual(
            statistics["ape_distinct_states_per_landmark_mean_spearman"], 1.0
        )
        self.assertAlmostEqual(
            statistics["ape_landmark_time_span_median_spearman"], 0.8
        )
        self.assertIs(statistics["descriptive_only"], True)

    def test_repeatability_statistics_rejects_incomplete_or_nonfinite_inputs(self):
        valid_run = {
            "run": "bak1",
            "ape_rmse_m": 1.0,
            "states": 1,
            "landmarks": 1,
            "observations": 1,
            "observations_per_landmark": 1.0,
            "distinct_states_per_landmark_mean": 1.0,
            "landmark_time_span_median_s": 1.0,
        }
        second_run = {**valid_run, "run": "bak2", "ape_rmse_m": 2.0}

        with self.assertRaisesRegex(ValueError, "at least two runs"):
            analysis.repeatability_statistics([valid_run], [{"aligned_rmse_m": 0.1}])
        with self.assertRaisesRegex(ValueError, "pairwise"):
            analysis.repeatability_statistics([valid_run, second_run], [])
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.repeatability_statistics(
                [{**valid_run, "ape_rmse_m": float("nan")}, second_run],
                [{"aligned_rmse_m": 0.1}],
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.repeatability_statistics(
                [valid_run, second_run], [{"aligned_rmse_m": float("inf")}]
            )

    def test_render_sequence_comparison_section_is_dynamic_and_order_invariant(self):
        def make_context(
            sequence, role, color, ape, normalized, angular_p50, observations
        ):
            summary = {
                "sequence": sequence,
                "ape_rmse_m": ape,
                "ape_over_distance_percent": normalized,
                "motion_linear_speed_mps_median": 0.2 + ape,
                "motion_linear_speed_mps_p95": 0.5 + ape,
                "motion_angular_speed_radps_median": angular_p50,
                "motion_angular_speed_radps_p95": 1.5 + ape,
                "motion_linear_above_0_5_duration_s": 2.0 + ape,
                "motion_linear_above_0_5_fraction": 0.02 + ape,
                "motion_linear_above_0_7_duration_s": 1.0 + ape,
                "motion_angular_above_1_0_duration_s": 3.0 + ape,
                "motion_angular_above_1_0_fraction": 0.03 + ape,
                "motion_angular_above_2_0_duration_s": 1.5 + ape,
                "motion_angular_above_2_0_fraction": 0.01 + ape,
                "imu_samples": 1000,
                "imu_max_interval_ms": 5.5,
                "imu_gap_count_over_7_5ms": 0,
                "imu_frozen_six_axis_count": 0,
                "imu_gyro_saturation_count": 0,
                "imu_accel_saturation_count": 0,
                "camera_one_to_one_sync_fraction": 0.99,
                "mocap_tracked_fraction": 1.0,
                "mocap_tracking_loss_records": 0,
                "mocap_mean_error_median_m": 0.0002,
                "states": 20 + int(ape * 100),
                "landmarks": 100,
                "observations": observations * 100,
                "observations_per_landmark": observations,
                "distinct_states_per_landmark_mean": observations / 2.0,
                "landmark_time_span_median_s": observations * 2.0,
                "keypoints_per_camera_frame": 350.0,
            }
            return {
                "sequence": sequence,
                "role": role,
                "color": color,
                "summary": summary,
            }

        contexts = [
            make_context("target-sequence", "target", "#target", 0.04, 0.30, 0.60, 6.0),
            make_context("control-b", "control", "#control-b", 0.02, 0.20, 0.70, 7.0),
            make_context("control-a", "control", "#control-a", 0.01, 0.10, 0.50, 5.0),
        ]
        camera_rows = []
        low_fractions = {
            "control-a": (0.10, 0.20),
            "control-b": (0.15, 0.25),
            "target-sequence": (0.30, 0.20),
        }
        for sequence, fractions in low_fractions.items():
            for camera, fraction in zip(("left-eye", "right-eye"), fractions):
                camera_rows.append(
                    {
                        "sequence": sequence,
                        "camera": camera,
                        "csv_frames": 100,
                        "missing_images": 0,
                        "gap_count_over_1_5x": 1,
                        "sharpness_samples": 10,
                        "sharpness_median": 500.0,
                        "sharpness_p5": 400.0,
                        "control_sharpness_p5": 410.0,
                        "fraction_below_control_p5": fraction,
                    }
                )
        run_rows = [
            {
                "run": run,
                "ape_rmse_m": ape,
                "states": states,
                "landmarks": landmarks,
                "observations": observations,
                "observations_per_landmark": persistence,
                "distinct_states_per_landmark_mean": distinct,
                "landmark_time_span_median_s": span,
            }
            for run, ape, states, landmarks, observations, persistence, distinct, span in (
                ("bak1", 0.4, 11, 110, 440, 4.0, 2.0, 0.50),
                ("bak2", 0.2, 12, 120, 600, 5.0, 3.0, 0.25),
                ("bak3", 0.1, 13, 130, 780, 6.0, 4.0, 0.75),
                ("bak4", 0.3, 14, 140, 420, 3.0, 1.0, 1.00),
            )
        ]
        pairwise_rows = [
            {"aligned_rmse_m": 0.05},
            {"aligned_rmse_m": 0.20},
        ]

        baseline = analysis.render_sequence_comparison_section(
            contexts, camera_rows, run_rows, pairwise_rows
        )
        for ordering in permutations(contexts):
            with self.subTest(order=[context["sequence"] for context in ordering]):
                self.assertEqual(
                    analysis.render_sequence_comparison_section(
                        list(ordering), camera_rows, run_rows, pairwise_rows
                    ),
                    baseline,
                )

        for sequence in ("control-a", "control-b", "target-sequence"):
            self.assertIn(sequence, baseline)
        self.assertIn("| control-a | 4.00x | 3.00x |", baseline)
        self.assertIn("| control-b | 2.00x | 1.50x |", baseline)
        self.assertIn("left-eye", baseline)
        self.assertIn("right-eye", baseline)
        self.assertIn("raw samples pooled across all controls", baseline)
        self.assertIn("angular speed p50 [rad/s] | 0.500-0.700 | 0.600 | within", baseline)
        distinctive, within = baseline.split("Target-distinctive metrics", 1)[1].split(
            "Metrics within the closed control envelope", 1
        )
        self.assertNotIn("angular speed p50", distinctive)
        self.assertIn("angular speed p50", within)
        for metric in (
            "angular speed p95",
            "angular time >1 rad/s",
            "angular time >2 rad/s",
            "linear time >0.5 m/s",
            "IMU gaps >7.5 ms",
            "camera one-to-one sync",
            "mocap tracked",
            "observations / landmark",
            "mean distinct states / landmark",
            "median landmark span",
            "states",
            "keypoints / camera frame",
        ):
            self.assertIn(metric, baseline)
        self.assertIn("n=4", baseline)
        self.assertIn("bak3", baseline)
        self.assertIn("cannot establish significance or causality", baseline)
        self.assertIn("cannot quantitatively separate or prove a single cause", baseline)
        self.assertIn("does not prove absolute physical synchronization", baseline)
        self.assertNotIn("fraction_below_reference", baseline)
        self.assertNotIn("reference versus target", baseline)
        self.assertNotIn("cam1-cam3", baseline)

        target_summary = next(
            context["summary"]
            for context in contexts
            if context["role"] == "target"
        )
        target_summary["camera_one_to_one_sync_fraction"] = 1.0
        above_sync_report = analysis.render_sequence_comparison_section(
            contexts, camera_rows, run_rows, pairwise_rows
        )
        sync_interpretation = (
            "Above-envelope camera one-to-one synchronization is favorable or neutral "
            "and is not evidence of worse acquisition timing."
        )
        self.assertIn(sync_interpretation, above_sync_report)
        target_summary["camera_one_to_one_sync_fraction"] = 0.98
        below_sync_report = analysis.render_sequence_comparison_section(
            contexts, camera_rows, run_rows, pairwise_rows
        )
        self.assertNotIn(sync_interpretation, below_sync_report)

        constant_run_rows = [
            {
                **row,
                "observations_per_landmark": 5.0,
                "distinct_states_per_landmark_mean": 3.0,
                "landmark_time_span_median_s": 0.5,
            }
            for row in run_rows
        ]
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            constant_report = analysis.render_sequence_comparison_section(
                contexts, camera_rows, constant_run_rows, pairwise_rows
            )
        constant_input_warnings = [
            str(warning.message)
            for warning in caught_warnings
            if "constant" in str(warning.message).lower()
        ]
        self.assertEqual(constant_input_warnings, [])
        self.assertIn("undefined (constant input)", constant_report)
        self.assertNotIn(
            "median-span ordering is not monotonic with APE", constant_report
        )

        single_control_contexts = [
            context
            for context in contexts
            if context["role"] == "target" or context["sequence"] == "control-a"
        ]
        single_control_camera_rows = [
            row
            for row in camera_rows
            if row["sequence"] in {"control-a", "target-sequence"}
        ]
        single_control_report = analysis.render_sequence_comparison_section(
            single_control_contexts,
            single_control_camera_rows,
            run_rows,
            pairwise_rows,
        )
        self.assertIn("## Sequence control-envelope comparison", single_control_report)
        self.assertIn("raw samples pooled across all controls", single_control_report)
        for count_specific_text in (
            "Three-sequence",
            "both controls",
            "two per-control",
        ):
            self.assertNotIn(count_specific_text, single_control_report)

    def test_write_report_summarizes_both_control_ranges_without_single_control_prose(self):
        stage_rows = []
        run_rows = []
        for index, run in enumerate(("bak1", "bak2", "bak3", "bak4"), 1):
            for stage in analysis.STAGE_FILES:
                stage_rows.append(
                    {
                        "run": run,
                        "stage": stage,
                        "ape_rmse_m": 0.01 * index,
                        "mocap_path_m": 40.0,
                        "ape_over_distance_percent": 0.1 * index,
                        "poses": 100,
                    }
                )
            run_rows.append(
                {
                    "run": run,
                    "ape_rmse_m": 0.01 * index,
                    "states": 10 + index,
                    "landmarks": 100 + index,
                    "observations": 500 + index,
                    "observations_per_landmark": 5.0 - index / 10.0,
                    "distinct_states_per_landmark_mean": 3.0 - index / 10.0,
                    "landmark_time_span_median_s": 0.25 * index,
                    "ape_angular_speed_spearman": 0.1 * index,
                }
            )
        control_rows = [
            {
                "sequence": "control-a",
                "mocap_path_m": 30.0,
                "ape_rmse_m": 0.01,
                "ape_over_distance_percent": 0.1,
                "mocap_linear_speed_mps_median": 0.2,
                "mocap_angular_speed_radps_median": 0.5,
            },
            {
                "sequence": "control-b",
                "mocap_path_m": 35.0,
                "ape_rmse_m": 0.02,
                "ape_over_distance_percent": 0.2,
                "mocap_linear_speed_mps_median": 0.3,
                "mocap_angular_speed_radps_median": 0.7,
            },
        ]
        pairwise_rows = [
            {"aligned_rmse_m": 0.05},
            {"aligned_rmse_m": 0.10},
        ]
        imu_summary = {
            "samples": 1000,
            "duration_s": 5.0,
            "median_interval_ms": 5.0,
            "max_interval_ms": 5.5,
            "gap_count_over_10ms": 0,
            "gyro_saturation_count": 0,
            "accel_saturation_count": 0,
        }
        camera_rows = [
            {
                "camera": "left-eye",
                "csv_frames": 100,
                "missing_images": 0,
                "max_interval_ms": 34.0,
                "sharpness_median": 500.0,
                "sharpness_p5": 400.0,
            }
        ]
        camera_aggregate = {
            "camera0_trajectory_offset_ms": 25.0,
            "common_exact_timestamps": 100,
            "exact_sync_fraction": 1.0,
            "sharpness_linear_speed_spearman": -0.1,
            "sharpness_angular_speed_spearman": -0.2,
        }
        motion_summary = {
            "linear_speed_mps_median": 0.25,
            "linear_speed_mps_p95": 0.5,
            "linear_speed_mps_max": 1.0,
            "angular_speed_radps_median": 0.6,
            "angular_speed_radps_p95": 1.8,
            "angular_speed_radps_max": 4.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "REPORT.md"
            analysis._write_report(
                path,
                stage_rows,
                run_rows,
                control_rows,
                pairwise_rows,
                imu_summary,
                camera_rows,
                camera_aggregate,
                0.025,
                motion_summary,
                {"ape_max_m": 10.0},
            )
            report = path.read_text(encoding="utf-8")

        self.assertIn("control-a", report)
        self.assertIn("control-b", report)
        self.assertIn("Control APE RMSE spans **10.000-20.000 mm**", report)
        self.assertIn("normalized APE spans **0.1000-0.2000%**", report)
        self.assertIn("## Control sequences", report)
        self.assertIn("## Reproduce", report)
        self.assertIn(
            "cd workspace/ego2_results/20260803-184537/analysis", report
        )
        self.assertIn(
            "conda run -n okvis2x python analyze_repeatability.py", report
        )
        self.assertIn(
            "conda run -n okvis2x python -m unittest -v "
            "test_analyze_repeatability.py",
            report,
        )
        self.assertIn("system Python does not provide `evo`", report)
        expected_facts = {
            "within-run correlation": (
                "Across the four runs, the absolute within-run Spearman correlation "
                "between per-pose final-BA translation error and temporally interpolated "
                "mocap angular speed ranges from 0.100 to 0.400."
            ),
            "final-BA passes and threads": (
                "The source hard-codes a 100-iteration limit for each of the two "
                "final-BA optimisation passes. Its thread count is the sum of configured "
                "realtime and full-graph thread counts, 12 (8+4) in the inspected YAML."
            ),
            "isolation conclusion": (
                "The first variant that collapses both graph-count spread and APE spread "
                "implicates that change as a contributor; confirm by re-enabling it and "
                "with randomized/factorial follow-up runs."
            ),
        }
        for fact, expected in expected_facts.items():
            with self.subTest(fact=fact):
                self.assertIn(expected, report)
        for stale_claim in (
            "correlation between final-BA APE and mocap angular speed",
            "identifies the dominant mechanism",
        ):
            with self.subTest(stale_claim=stale_claim):
                self.assertNotIn(stale_claim, report)
        self.assertNotIn("Target/reference", report)
        self.assertNotIn("motion- and scale-matched", report)

    def test_pooled_control_sharpness_thresholds_exclude_target_per_camera(self):
        contexts = [
            {
                "sequence": "target",
                "role": "target",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": -1000.0},
                    {"camera": "cam1", "laplacian_variance": -10000.0},
                ],
            },
            {
                "sequence": "control-a",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 10.0},
                    {"camera": "cam0", "laplacian_variance": 20.0},
                    {"camera": "cam1", "laplacian_variance": 100.0},
                    {"camera": "cam1", "laplacian_variance": 200.0},
                ],
            },
            {
                "sequence": "control-b",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 30.0},
                    {"camera": "cam0", "laplacian_variance": 40.0},
                    {"camera": "cam1", "laplacian_variance": 300.0},
                    {"camera": "cam1", "laplacian_variance": 400.0},
                ],
            },
        ]

        thresholds = analysis.pooled_control_sharpness_thresholds(contexts)

        self.assertEqual(set(thresholds), {"cam0", "cam1"})
        self.assertAlmostEqual(thresholds["cam0"], 11.5)
        self.assertAlmostEqual(thresholds["cam1"], 115.0)

    def test_pooled_control_thresholds_reject_empty_or_nonfinite_samples(self):
        invalid_cases = (
            ([], r"control-b.*no sharpness samples"),
            (
                [
                    {"camera": "cam0", "laplacian_variance": 30.0},
                    {"camera": "cam0", "laplacian_variance": float("nan")},
                ],
                r"control-b.*cam0.*non-finite",
            ),
        )
        for quality_rows, message in invalid_cases:
            contexts = [
                {"sequence": "target", "role": "target", "quality_rows": []},
                {
                    "sequence": "control-a",
                    "role": "control",
                    "quality_rows": [
                        {"camera": "cam0", "laplacian_variance": 10.0},
                        {"camera": "cam0", "laplacian_variance": 20.0},
                    ],
                },
                {
                    "sequence": "control-b",
                    "role": "control",
                    "quality_rows": quality_rows,
                },
            ]

            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    analysis.pooled_control_sharpness_thresholds(contexts)

    def test_pooled_control_thresholds_reject_different_camera_sets(self):
        contexts = [
            {"sequence": "target", "role": "target", "quality_rows": []},
            {
                "sequence": "control-a",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 10.0},
                    {"camera": "cam1", "laplacian_variance": 100.0},
                ],
            },
            {
                "sequence": "control-b",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 30.0},
                ],
            },
        ]

        with self.assertRaisesRegex(ValueError, r"control-b.*cam1"):
            analysis.pooled_control_sharpness_thresholds(contexts)

    def test_pooled_control_thresholds_reject_unequal_camera_sample_counts(self):
        contexts = [
            {"sequence": "target", "role": "target", "quality_rows": []},
            {
                "sequence": "control-a",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 10.0},
                    {"camera": "cam0", "laplacian_variance": 20.0},
                ],
            },
            {
                "sequence": "control-b",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 30.0},
                ],
            },
        ]

        with self.assertRaisesRegex(ValueError, r"control-b.*cam0.*1.*2"):
            analysis.pooled_control_sharpness_thresholds(contexts)

    def test_finalize_quality_comparison_uses_pooled_control_threshold_fields(self):
        contexts = [
            {
                "sequence": "target",
                "role": "target",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": -1000.0},
                    {"camera": "cam0", "laplacian_variance": 1000.0},
                ],
                "camera_rows": [{"camera": "cam0"}],
                "summary": {},
            },
            {
                "sequence": "control-a",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 10.0},
                    {"camera": "cam0", "laplacian_variance": 20.0},
                ],
                "camera_rows": [{"camera": "cam0"}],
                "summary": {},
            },
            {
                "sequence": "control-b",
                "role": "control",
                "quality_rows": [
                    {"camera": "cam0", "laplacian_variance": 30.0},
                    {"camera": "cam0", "laplacian_variance": 40.0},
                ],
                "camera_rows": [{"camera": "cam0"}],
                "summary": {},
            },
        ]

        camera_rows = analysis.finalize_quality_comparison(contexts)

        self.assertEqual(len(camera_rows), 3)
        for row in camera_rows:
            self.assertAlmostEqual(row["control_sharpness_p5"], 11.5)
            self.assertNotIn("reference_sharpness_p5", row)
            self.assertNotIn("fraction_below_reference_p5", row)
            self.assertNotIn("samples_below_reference_p5", row)
        self.assertEqual(camera_rows[0]["samples_below_control_p5"], 1)
        self.assertAlmostEqual(camera_rows[0]["fraction_below_control_p5"], 0.5)
        for context in contexts:
            self.assertIn(
                "image_fraction_below_control_camera_p5", context["summary"]
            )
            self.assertNotIn(
                "image_fraction_below_reference_camera_p5", context["summary"]
            )

    def test_threshold_statistics_weight_duration_and_merge_consecutive_samples(self):
        values = np.asarray([0.2, 0.8, 0.9, 0.1])
        durations = np.asarray([1.0, 2.0, 3.0, 4.0])

        statistics = analysis.threshold_statistics(values, durations, threshold=0.7)

        self.assertAlmostEqual(statistics["duration_s"], 5.0)
        self.assertAlmostEqual(statistics["fraction"], 0.5)
        self.assertAlmostEqual(statistics["longest_s"], 5.0)
        self.assertEqual(statistics["event_count"], 1)

    def test_motion_summary_includes_four_rad_per_second_exposure(self):
        motion = {
            "timestamps": np.arange(5.0),
            "durations": np.ones(5),
            "linear_speed": np.zeros(5),
            "angular_speed": np.asarray([0.5, 1.5, 2.5, 3.5, 4.5]),
        }

        summary = analysis.summarize_motion(motion)

        self.assertAlmostEqual(summary["angular_above_4_0_fraction"], 0.2)
        self.assertAlmostEqual(summary["angular_above_4_0_duration_s"], 1.0)

    def test_motion_threshold_rows_include_four_rad_per_second(self):
        motion = {
            "durations": np.ones(2),
            "linear_speed": np.zeros(2),
            "angular_speed": np.asarray([0.0, 5.0]),
        }

        rows = analysis.motion_threshold_rows("sequence", motion)
        angular_thresholds = [
            row["threshold"] for row in rows if row["metric"] == "angular_speed"
        ]

        self.assertEqual(angular_thresholds, [1.0, 2.0, 3.0, 4.0])

    def test_camera_sync_metrics_include_exact_and_tolerant_matching(self):
        indices = {
            "cam0": np.asarray([0.0, 1.0, 2.0]),
            "cam1": np.asarray([0.0, 1.0002, 2.0]),
            "cam2": np.asarray([0.0, 1.0, 2.02]),
            "cam3": np.asarray([0.0, 1.0, 2.0]),
        }

        metrics, skews = analysis.camera_sync_metrics(indices)

        self.assertEqual(metrics["common_exact_timestamps"], 1)
        self.assertAlmostEqual(metrics["exact_sync_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["within_1ms_fraction"], 2.0 / 3.0)
        self.assertEqual(metrics["one_to_one_sync_groups"], 2)
        self.assertAlmostEqual(metrics["one_to_one_sync_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["nearest_skew_max_ms"], 20.0)
        np.testing.assert_allclose(skews * 1000.0, [0.0, 0.2, 20.0])

    def test_camera_timestamp_correction_subtracts_configured_image_delay(self):
        corrected = analysis.correct_camera_timestamps(
            np.asarray([10.0, 11.0]), image_delay=0.025
        )

        np.testing.assert_allclose(corrected, [9.975, 10.975])

    def test_image_quality_metrics_report_exposure_contrast_and_clipping(self):
        pixels = np.asarray([[0, 10], [250, 255]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            from PIL import Image

            Image.fromarray(pixels).save(path)

            metrics = analysis.image_quality_metrics(path)

        self.assertAlmostEqual(metrics["intensity_mean"], 128.75)
        self.assertAlmostEqual(metrics["intensity_std"], float(np.std(pixels)))
        self.assertAlmostEqual(metrics["dark_clip_fraction"], 0.25)
        self.assertAlmostEqual(metrics["bright_clip_fraction"], 0.25)
        self.assertGreater(metrics["laplacian_variance"], 0.0)

    def test_mean_frame_difference_detects_intensity_change(self):
        first = np.asarray([[0, 10], [20, 30]], dtype=np.uint8)
        second = first + 10
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            from PIL import Image

            Image.fromarray(first).save(first_path)
            Image.fromarray(second).save(second_path)

            difference = analysis.mean_frame_difference(first_path, second_path)

        self.assertAlmostEqual(difference, 10.0)

    def test_prefix_alignment_exposes_drift_after_the_fit_window(self):
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        target = (rotation @ source.T).T + np.asarray([4.0, -2.0, 1.5])
        target[-1] += np.asarray([1.0, 0.0, 0.0])

        _, errors = analysis.prefix_align_and_errors(
            source, target, np.arange(5.0), alignment_duration=3.0
        )

        np.testing.assert_allclose(errors[:4], 0.0, atol=1e-12)
        self.assertAlmostEqual(errors[-1], 1.0)

    def test_okvis_reader_keeps_first_duplicate_timestamp(self):
        contents = (
            "timestamp,p_WS_W_x,p_WS_W_y,p_WS_W_z,q_WS_x,q_WS_y,q_WS_z,q_WS_w,"
            "v_WS_W_x,v_WS_W_y,v_WS_W_z\n"
            "1000000000,0,0,0,0,0,0,1,1,0,0\n"
            "2000000000,1,0,0,0,0,0,1,2,0,0\n"
            "2000000000,99,0,0,0,0,0,1,99,0,0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            path.write_text(contents, encoding="utf-8")

            trajectory = analysis.load_okvis_trajectory(path)

        self.assertEqual(trajectory.timestamps.tolist(), [1.0, 2.0])
        self.assertEqual(trajectory.positions[-1].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(trajectory.velocities[-1].tolist(), [2.0, 0.0, 0.0])

    def test_imu_reader_converts_nanosecond_timestamps(self):
        contents = (
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
            "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
            "a_RS_S_z [m s^-2]\n"
            "1000000000,1,2,3,4,5,6\n"
            "1005000000,7,8,9,10,11,12\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text(contents, encoding="utf-8")

            imu = analysis.load_imu(path)

        np.testing.assert_allclose(imu.timestamps, [1.0, 1.005])
        np.testing.assert_allclose(imu.gyroscope[1], [7.0, 8.0, 9.0])
        np.testing.assert_allclose(imu.accelerometer[0], [4.0, 5.0, 6.0])

    def test_mocap_integrity_counts_tracking_loss_and_quality(self):
        contents = (
            "time: 1.000 frame_id: 7 latency: 10.0\n"
            "Base rigid body id: 15; tracked: 1; mean_error: 0.001\n"
            "time: 1.010 frame_id: 8 latency: 12.0\n"
            "Base rigid body id: 15; tracked: 0; mean_error: 0.002\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mocap.log"
            path.write_text(contents, encoding="utf-8")

            summary = analysis.analyze_mocap_integrity(path)

        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["tracked_records"], 1)
        self.assertEqual(summary["tracking_loss_records"], 1)
        self.assertAlmostEqual(summary["mean_error_median_m"], 0.0015)
        self.assertAlmostEqual(summary["latency_median_ms"], 11.0)

    def test_descriptive_statistics_include_requested_percentiles(self):
        values = np.arange(1.0, 101.0)

        statistics = analysis.descriptive_statistics(values)

        self.assertEqual(statistics["count"], 100)
        self.assertAlmostEqual(statistics["mean"], 50.5)
        self.assertAlmostEqual(statistics["median"], 50.5)
        self.assertAlmostEqual(statistics["p95"], 95.05)
        self.assertAlmostEqual(statistics["max"], 100.0)

    def test_path_length_sums_three_dimensional_steps(self):
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]
        )

        self.assertAlmostEqual(analysis.path_length(positions), 17.0)

    def test_linear_speed_uses_each_timestamp_interval(self):
        timestamps = np.asarray([10.0, 12.0, 15.0])
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [6.0, 8.0, 0.0], [6.0, 8.0, 12.0]]
        )

        speed = analysis.linear_speed(timestamps, positions)

        np.testing.assert_allclose(speed, [5.0, 4.0])

    def test_bounded_interpolation_marks_queries_outside_source_range(self):
        values = analysis.bounded_interpolate(
            np.asarray([-0.1, 0.5, 1.1]),
            np.asarray([0.0, 1.0]),
            np.asarray([10.0, 20.0]),
        )

        self.assertTrue(np.isnan(values[0]))
        self.assertAlmostEqual(values[1], 15.0)
        self.assertTrue(np.isnan(values[2]))

    def test_angular_speed_uses_shortest_quaternion_rotation(self):
        timestamps = np.asarray([0.0, 1.0, 2.0])
        half = math.sqrt(0.5)
        quaternions_wxyz = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [half, 0.0, 0.0, half], [0.0, 0.0, 0.0, 1.0]]
        )

        speed = analysis.angular_speed(timestamps, quaternions_wxyz)

        np.testing.assert_allclose(speed, [math.pi / 2.0, math.pi / 2.0])

    def test_rigid_alignment_removes_rotation_and_translation(self):
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        target = (rotation @ source.T).T + np.asarray([4.0, -2.0, 1.5])

        aligned, errors = analysis.rigid_align_and_errors(source, target)

        np.testing.assert_allclose(aligned, target, atol=1e-12)
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)

    def test_first_sustained_crossing_requires_consecutive_samples(self):
        values = np.asarray([0.0, 2.0, 0.0, 2.0, 2.0, 2.0, 0.0])

        index = analysis.first_sustained_crossing(values, threshold=1.0, samples=3)

        self.assertEqual(index, 3)

    def test_g2o_record_counter_counts_each_record_type(self):
        contents = "\n".join(
            [
                "VERTEX_SE3:QUAT_TIME 1 values",
                "FRAME 1 0 values",
                "FRAME 1 1 values",
                "VERTEX_TRACKXYZ 7 values",
                "EDGE_OBS values",
                "EDGE_OBS values",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.g2o"
            path.write_text(contents, encoding="utf-8")

            counts = analysis.count_g2o_records(path)

        self.assertEqual(counts["VERTEX_SE3:QUAT_TIME"], 1)
        self.assertEqual(counts["FRAME"], 2)
        self.assertEqual(counts["VERTEX_TRACKXYZ"], 1)
        self.assertEqual(counts["EDGE_OBS"], 2)

    def test_map_track_statistics_use_distinct_states_and_time_span(self):
        contents = "\n".join(
            [
                "VERTEX_SE3:QUAT_TIME 1 values 1000000000",
                "VERTEX_SE3:QUAT_TIME 2 values 2000000000",
                "VERTEX_SE3:QUAT_TIME 3 values 4000000000",
                "VERTEX_TRACKXYZ 10 values",
                "VERTEX_TRACKXYZ 11 values",
                "EDGE_OBS 1 0 5 10 values",
                "EDGE_OBS 1 1 8 10 values",
                "EDGE_OBS 2 0 7 10 values",
                "EDGE_OBS 3 0 9 11 values",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.g2o"
            path.write_text(contents, encoding="utf-8")

            statistics = analysis.map_track_statistics(path)

        self.assertAlmostEqual(statistics["distinct_states_per_landmark_mean"], 1.5)
        self.assertAlmostEqual(statistics["distinct_states_per_landmark_median"], 1.5)
        self.assertAlmostEqual(statistics["landmark_time_span_median_s"], 0.5)
        self.assertAlmostEqual(statistics["single_state_landmark_fraction"], 0.5)

    def test_landmark_quality_statistics_parse_final_quality(self):
        contents = "\n".join(
            [
                "VERTEX_TRACKXYZ 1 0 0 1 0.0005",
                "VERTEX_TRACKXYZ 2 0 0 1 0.0100",
                "VERTEX_TRACKXYZ 3 0 0 1 0.0500",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.g2o"
            path.write_text(contents, encoding="utf-8")

            statistics = analysis.landmark_quality_statistics(path)

        self.assertEqual(statistics["quality_count"], 3)
        self.assertAlmostEqual(statistics["quality_median"], 0.01)
        self.assertAlmostEqual(statistics["quality_p90"], 0.042)
        self.assertAlmostEqual(statistics["quality_p95"], 0.046)
        self.assertAlmostEqual(
            statistics["quality_fraction_above_0p001"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            statistics["quality_initialized_fraction"], 1.0 / 3.0
        )

    def test_landmark_quality_statistics_require_valid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_path = root / "empty.g2o"
            empty_path.write_text("FRAME 1 0 values\n", encoding="utf-8")
            malformed_path = root / "malformed.g2o"
            malformed_path.write_text(
                "VERTEX_TRACKXYZ 1 0 0 1 invalid\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "no VERTEX_TRACKXYZ quality"):
                analysis.landmark_quality_statistics(empty_path)
            with self.assertRaisesRegex(
                ValueError, r"malformed\.g2o:1: malformed VERTEX_TRACKXYZ quality"
            ):
                analysis.landmark_quality_statistics(malformed_path)

    def test_target_run_map_summary_persists_full_topology_and_imu_counts(self):
        contents = "\n".join(
            [
                "VERTEX_SE3:QUAT_TIME 1 values 1000000000",
                "VERTEX_SE3:QUAT_TIME 2 values 2000000000",
                "VERTEX_SE3:QUAT_TIME 3 values 4000000000",
                "FRAME 1 0 values",
                "FRAME 1 1 values",
                "VERTEX_TRACKXYZ 10 0 0 1 0.01",
                "VERTEX_TRACKXYZ 11 0 0 1 0.05",
                "EDGE_OBS 1 0 5 10 values",
                "EDGE_OBS 1 1 8 10 values",
                "EDGE_OBS 2 0 7 10 values",
                "EDGE_OBS 3 0 9 11 values",
                "FRAME:KEYPOINT values",
                "FRAME:KEYPOINT values",
                "FRAME:KEYPOINT values",
                "EDGE_IMU values",
                "EDGE_IMU:MEASUREMENTS values",
                "EDGE_IMU:MEASUREMENTS values",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            (result_dir / "okvis2-slam-calib-final_map.g2o").write_text(
                contents, encoding="utf-8"
            )

            row = analysis.target_run_map_summary(
                result_dir, {"run": "bak1", "ape_rmse_m": 0.02}
            )

        self.assertEqual(row["run"], "bak1")
        self.assertEqual(row["states"], 3)
        self.assertEqual(row["keypoints"], 3)
        self.assertAlmostEqual(row["observations_per_landmark"], 2.0)
        self.assertAlmostEqual(row["distinct_states_per_landmark_mean"], 1.5)
        self.assertAlmostEqual(row["distinct_states_per_landmark_median"], 1.5)
        self.assertAlmostEqual(row["distinct_states_per_landmark_p95"], 1.95)
        self.assertEqual(row["distinct_states_per_landmark_max"], 2.0)
        self.assertAlmostEqual(row["landmark_time_span_median_s"], 0.5)
        self.assertAlmostEqual(row["landmark_time_span_p95_s"], 0.95)
        self.assertAlmostEqual(row["landmark_time_span_max_s"], 1.0)
        self.assertAlmostEqual(row["single_state_landmark_fraction"], 0.5)
        self.assertEqual(row["quality_count"], 2)
        self.assertAlmostEqual(row["quality_median"], 0.03)
        self.assertAlmostEqual(row["quality_initialized_fraction"], 0.5)
        self.assertEqual(row["imu_edges"], 1)
        self.assertEqual(row["imu_measurements"], 2)


if __name__ == "__main__":
    unittest.main()
