#!/usr/bin/env python3

import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


SCRIPT_PATH = Path(__file__).with_name("analyze_cross_sample_diagnostics.py")
SPEC = importlib.util.spec_from_file_location(
    "analyze_cross_sample_diagnostics", SCRIPT_PATH
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


class CrossSampleDiagnosticsTest(unittest.TestCase):

    def test_main_guard_is_the_last_top_level_statement(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        guards = [
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ]

        self.assertEqual(guards, [len(tree.body) - 1])

    def test_validate_unique_keys_accepts_expected_cardinality(self):
        rows = [
            {"sequence": "a", "run": "run1"},
            {"sequence": "a", "run": "run2"},
        ]

        ANALYSIS.validate_unique_keys(
            rows, ("sequence", "run"), expected=2
        )

    def test_validate_unique_keys_reports_duplicate_and_wrong_count(self):
        duplicate = [
            {"sequence": "a", "run": "run1"},
            {"sequence": "a", "run": "run1"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate.*a.*run1"):
            ANALYSIS.validate_unique_keys(
                duplicate, ("sequence", "run"), expected=2
            )
        with self.assertRaisesRegex(ValueError, "expected 2.*found 1"):
            ANALYSIS.validate_unique_keys(
                duplicate[:1], ("sequence", "run"), expected=2
            )

    def test_cohort_memberships_exclude_only_declared_subsets(self):
        rows = [{"sequence": f"seq-{index:02d}"} for index in range(18)]
        rows += [
            {"sequence": "20260806-175103"},
            {"sequence": "20260806-175304"},
            {"sequence": "20260806-175539"},
            {"sequence": "20260805-122310"},
            {"sequence": "20260805-123231"},
            {"sequence": "20260805-123752"},
        ]

        cohorts = ANALYSIS.cohort_memberships(rows)

        self.assertEqual(len(cohorts["all"]), 24)
        self.assertEqual(len(cohorts["without_impulse"]), 21)
        self.assertEqual(len(cohorts["without_mocap_correction"]), 21)
        self.assertEqual(len(cohorts["natural_uncorrected_subset"]), 18)
        self.assertNotIn(
            "20260806-175304",
            {row["sequence"] for row in cohorts["without_impulse"]},
        )
        self.assertNotIn(
            "20260805-122310",
            {row["sequence"] for row in cohorts["without_mocap_correction"]},
        )

    def test_cliffs_delta_reports_separation_ties_and_reversal(self):
        self.assertEqual(ANALYSIS.cliffs_delta([4, 5, 6], [1, 2, 3]), 1.0)
        self.assertEqual(ANALYSIS.cliffs_delta([1, 2], [1, 2]), 0.0)
        self.assertEqual(ANALYSIS.cliffs_delta([1, 2, 3], [4, 5, 6]), -1.0)

    def test_association_strength_uses_documented_boundaries(self):
        self.assertEqual(ANALYSIS.association_strength(0.60), "strong")
        self.assertEqual(ANALYSIS.association_strength(-0.35), "moderate")
        self.assertEqual(ANALYSIS.association_strength(0.20), "weak")
        self.assertEqual(
            ANALYSIS.association_strength(0.199), "currently_not_supported"
        )
        self.assertEqual(
            ANALYSIS.association_strength(0.80, effect="cliffs_delta"),
            "strong",
        )
        self.assertEqual(
            ANALYSIS.association_strength(0.474, effect="cliffs_delta"),
            "moderate",
        )
        self.assertEqual(
            ANALYSIS.association_strength(0.147, effect="cliffs_delta"),
            "weak",
        )

    def test_population_grade_cannot_exceed_full_or_sensitivity_grade(self):
        self.assertEqual(
            ANALYSIS.population_support_grade(
                full_strength="moderate",
                sensitivity_strengths=["strong", "strong", "strong"],
                coverage_fraction=1.0,
                direction_consistent=True,
            ),
            "moderate",
        )
        self.assertEqual(
            ANALYSIS.population_support_grade(
                full_strength="strong",
                sensitivity_strengths=["strong", "moderate", "strong"],
                coverage_fraction=1.0,
                direction_consistent=True,
            ),
            "moderate",
        )

    def test_population_grade_is_capped_when_coverage_or_direction_is_weak(self):
        self.assertEqual(
            ANALYSIS.population_support_grade(
                full_strength="strong",
                sensitivity_strengths=["strong", "strong", "strong"],
                coverage_fraction=0.75,
                direction_consistent=True,
            ),
            "weak",
        )
        self.assertEqual(
            ANALYSIS.population_support_grade(
                full_strength="strong",
                sensitivity_strengths=["strong", "strong", "strong"],
                coverage_fraction=1.0,
                direction_consistent=False,
            ),
            "weak",
        )

    def test_outcome_labels_use_joint_fragmentation_and_scale_rules(self):
        row = {
            "corrected_ape_median_mm": 10.001,
            "ransac_fail_per_min": 15.0,
            "landmark_time_span_median_s": 3.0,
            "sim3_improvement_pct": 25.0,
            "scale_estimate_to_mocap": 0.90,
        }

        labelled = ANALYSIS.apply_outcome_labels(row)

        self.assertTrue(labelled["ape_over_10mm"])
        self.assertTrue(labelled["visual_fragmentation"])
        self.assertTrue(labelled["scale_instability"])

    def test_outcome_labels_do_not_promote_single_symptoms(self):
        base = {
            "corrected_ape_median_mm": 10.0,
            "ransac_fail_per_min": 14.99,
            "landmark_time_span_median_s": 2.0,
            "sim3_improvement_pct": 24.99,
            "scale_estimate_to_mocap": 0.50,
        }
        labelled = ANALYSIS.apply_outcome_labels(base)
        self.assertFalse(labelled["ape_over_10mm"])
        self.assertFalse(labelled["visual_fragmentation"])
        self.assertFalse(labelled["scale_instability"])

        long_tracks = ANALYSIS.apply_outcome_labels(
            {**base, "ransac_fail_per_min": 20.0, "landmark_time_span_median_s": 3.01}
        )
        self.assertFalse(long_tracks["visual_fragmentation"])

    def test_strict_join_preserves_order_and_rejects_missing_or_duplicate_keys(self):
        left = [
            {"sequence": "b", "ape": 2.0},
            {"sequence": "a", "ape": 1.0},
        ]
        right = [
            {"sequence": "a", "metric": 10.0},
            {"sequence": "b", "metric": 20.0},
        ]

        joined = ANALYSIS.strict_join(left, right, ("sequence",))

        self.assertEqual([row["sequence"] for row in joined], ["b", "a"])
        self.assertEqual([row["metric"] for row in joined], [20.0, 10.0])

        with self.assertRaisesRegex(ValueError, "missing.*b"):
            ANALYSIS.strict_join(left, right[:1], ("sequence",))
        with self.assertRaisesRegex(ValueError, "duplicate.*a"):
            ANALYSIS.strict_join(left, [right[0], right[0]], ("sequence",))

    def test_collect_run_diagnostics_normalizes_log_counts_by_sequence_duration(self):
        map_contents = "\n".join(
            [
                "VERTEX_SE3:QUAT_TIME 1 values 1000000000",
                "VERTEX_SE3:QUAT_TIME 2 values 2000000000",
                "FRAME 1 0 values",
                "VERTEX_TRACKXYZ 10 0 0 1 0.05",
                "EDGE_OBS 1 0 0 0 10 values",
                "EDGE_OBS 2 0 0 0 10 values",
                "FRAME:KEYPOINT values",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = []
            for sequence in ("seq-a", "seq-b"):
                run_dirs = []
                for run in ("run1", "run2"):
                    run_dir = root / sequence / run
                    run_dir.mkdir(parents=True)
                    (run_dir / "okvis2-slam-calib-final_map.g2o").write_text(
                        map_contents, encoding="utf-8"
                    )
                    (run_dir / "okvis.log").write_text(
                        "RANSAC FAIL\nRANSAC FAIL\nlarge reprojection error\n",
                        encoding="utf-8",
                    )
                    run_dirs.append(run_dir)
                specs.append(
                    SimpleNamespace(
                        day="20260803",
                        sequence=sequence,
                        run_dirs=tuple(run_dirs),
                    )
                )

            rows = ANALYSIS.collect_run_diagnostics(
                specs, {"seq-a": 60.0, "seq-b": 120.0}
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["sequence"], row["run"]) for row in rows},
            {
                ("seq-a", "run1"),
                ("seq-a", "run2"),
                ("seq-b", "run1"),
                ("seq-b", "run2"),
            },
        )
        self.assertEqual(rows[0]["quality_count"], 1)
        self.assertAlmostEqual(rows[0]["ransac_fail_per_min"], 2.0)
        self.assertAlmostEqual(rows[2]["ransac_fail_per_min"], 1.0)

    def test_similarity_alignment_recovers_scale_without_hiding_se3_error(self):
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        estimate = reference * 2.0 + np.asarray([4.0, -3.0, 1.0])

        aligned, scale, errors = ANALYSIS.similarity_align_and_errors(
            estimate, reference
        )

        self.assertAlmostEqual(scale, 0.5)
        np.testing.assert_allclose(aligned, reference, atol=1e-12)
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)

    def test_aggregate_run_rows_requires_two_runs_and_uses_medians(self):
        rows = [
            {
                "day": "20260803",
                "sequence": "a",
                "run": "run1",
                "ransac_fail_per_min": 2.0,
                "landmark_time_span_median_s": 4.0,
                "run_log_available": True,
            },
            {
                "day": "20260803",
                "sequence": "a",
                "run": "run2",
                "ransac_fail_per_min": 4.0,
                "landmark_time_span_median_s": 2.0,
                "run_log_available": False,
            },
        ]

        aggregate = ANALYSIS.aggregate_run_rows(rows, expected_runs=2)

        self.assertEqual(len(aggregate), 1)
        self.assertAlmostEqual(aggregate[0]["ransac_fail_per_min"], 3.0)
        self.assertAlmostEqual(
            aggregate[0]["landmark_time_span_median_s"], 3.0
        )
        self.assertAlmostEqual(aggregate[0]["run_log_coverage"], 0.5)

        with self.assertRaisesRegex(ValueError, "a: expected 2 runs, found 1"):
            ANALYSIS.aggregate_run_rows(rows[:1], expected_runs=2)

    def test_collect_image_diagnostics_uses_uniform_sampling_and_keys_rows(self):
        spec = SimpleNamespace(
            day="20260803",
            sequence="seq-a",
            dataset=Path("/dataset"),
            mocap=Path("/mocap.log"),
        )
        baseline = {
            "seq-a": {"analysis_start_s": 10.0, "analysis_end_s": 20.0}
        }
        motion = {
            "timestamps": np.asarray([10.0, 20.0]),
            "linear_speed": np.asarray([0.1, 0.2]),
            "angular_speed": np.asarray([0.3, 0.4]),
        }
        camera_rows = [{"camera": "cam0", "sharpness_samples": 80}]
        sample_rows = [{"camera": "cam0", "laplacian_variance": 12.0}]
        aggregate = {"exact_sync_fraction": 1.0}
        mocap_integrity = {"tracked_fraction": 1.0}
        reference = ANALYSIS.repeatability.Trajectory(
            timestamps=np.asarray([10.0, 20.0]),
            positions=np.zeros((2, 3)),
            quaternions_wxyz=np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            velocities=np.full((2, 3), np.nan),
        )

        with patch.object(
            ANALYSIS.repeatability,
            "load_mocap_trajectory",
            return_value=reference,
        ), patch.object(
            ANALYSIS.repeatability, "_motion_from_mocap", return_value=motion
        ), patch.object(
            ANALYSIS.repeatability,
            "analyze_cameras",
            return_value=(camera_rows, sample_rows, aggregate),
        ) as analyze_cameras, patch.object(
            ANALYSIS.repeatability,
            "analyze_mocap_integrity",
            return_value=mocap_integrity,
        ):
            result = ANALYSIS.collect_image_diagnostics(
                [spec], baseline, image_delay_s=0.025, samples_per_camera=80
            )

        per_camera, per_sample, camera_aggregate, mocap_rows = result
        analyze_cameras.assert_called_once_with(
            spec.dataset,
            10.0,
            motion["timestamps"],
            motion["linear_speed"],
            motion["angular_speed"],
            80,
            0.025,
        )
        self.assertEqual(per_camera[0]["sequence"], "seq-a")
        self.assertEqual(per_sample[0]["day"], "20260803")
        self.assertEqual(camera_aggregate[0]["exact_sync_fraction"], 1.0)
        self.assertEqual(mocap_rows[0]["tracked_fraction"], 1.0)

    def test_compute_correlation_rows_emits_every_cohort_and_effect_size(self):
        rows = []
        ordinary = [f"seq-{index:02d}" for index in range(18)]
        sequences = ordinary + sorted(
            ANALYSIS.IMPULSE_SEQUENCES | ANALYSIS.MOCAP_CORRECTED_SEQUENCES
        )
        for index, sequence in enumerate(sequences):
            rows.append(
                {
                    "sequence": sequence,
                    "corrected_ape_median_mm": float(index + 1),
                    "ape_over_10mm": index >= 10,
                    "visual_fragmentation": index >= 18,
                    "metric": float(index),
                }
            )
        factor = {
            "factor": "test_metric",
            "label_zh": "测试指标",
            "metric": "metric",
            "expected_direction": 1,
            "role": "candidate_trigger",
            "is_proxy": False,
        }

        result = ANALYSIS.compute_correlation_rows(rows, [factor])

        self.assertEqual(len(result), 4)
        self.assertEqual(
            {row["cohort"] for row in result},
            {
                "all",
                "without_impulse",
                "without_mocap_correction",
                "natural_uncorrected_subset",
            },
        )
        full = next(row for row in result if row["cohort"] == "all")
        self.assertEqual(full["available_sequences"], 24)
        self.assertAlmostEqual(full["spearman_rho"], 1.0)
        self.assertAlmostEqual(full["ape_over_10mm_cliffs_delta"], 1.0)

    def test_evidence_synthesis_keeps_unsupported_factors_and_caps_proxies(self):
        factors = [
            {
                "factor": "direct",
                "label_zh": "直接指标",
                "metric": "direct_metric",
                "expected_direction": 1,
                "role": "downstream_state",
                "is_proxy": False,
            },
            {
                "factor": "proxy",
                "label_zh": "间接代理",
                "metric": "proxy_metric",
                "expected_direction": 1,
                "role": "candidate_trigger",
                "is_proxy": True,
            },
            {
                "factor": "unsupported",
                "label_zh": "无支持指标",
                "metric": "unsupported_metric",
                "expected_direction": 1,
                "role": "candidate_trigger",
                "is_proxy": False,
            },
        ]
        rows = []
        ordinary = [f"seq-{index:02d}" for index in range(18)]
        sequences = ordinary + sorted(
            ANALYSIS.IMPULSE_SEQUENCES | ANALYSIS.MOCAP_CORRECTED_SEQUENCES
        )
        unsupported_values = [0.0, 1.0] * 12
        for index, sequence in enumerate(sequences):
            rows.append(
                {
                    "sequence": sequence,
                    "corrected_ape_median_mm": float(index + 1),
                    "ape_over_10mm": index >= 10,
                    "visual_fragmentation": index >= 18,
                    "direct_metric": float(index),
                    "proxy_metric": float(index),
                    "unsupported_metric": unsupported_values[index],
                }
            )

        correlations = ANALYSIS.compute_correlation_rows(rows, factors)
        evidence = ANALYSIS.synthesize_evidence(correlations, factors)

        self.assertEqual({row["factor"] for row in evidence}, {"direct", "proxy", "unsupported"})
        by_factor = {row["factor"]: row for row in evidence}
        self.assertEqual(by_factor["direct"]["support_level"], "strong")
        self.assertEqual(by_factor["proxy"]["support_level"], "weak")
        self.assertEqual(
            by_factor["unsupported"]["support_level"],
            "currently_not_supported",
        )

    def test_csv_roundtrip_preserves_text_keys_and_numeric_types(self):
        rows = [
            {
                "day": "20260803",
                "sequence": "20260803-183537",
                "run": "run1",
                "value": 1.25,
                "count": 3,
                "available": True,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            ANALYSIS.write_csv_rows(path, rows)

            loaded = ANALYSIS.read_csv_rows(path)

        self.assertEqual(loaded[0]["day"], "20260803")
        self.assertEqual(loaded[0]["sequence"], "20260803-183537")
        self.assertEqual(loaded[0]["run"], "run1")
        self.assertEqual(loaded[0]["value"], 1.25)
        self.assertEqual(loaded[0]["count"], 3)
        self.assertIs(loaded[0]["available"], True)

    def test_summarize_image_diagnostics_uses_all_samples_and_camera_coverage(self):
        camera_rows = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "camera": camera,
                "missing_images": missing,
                "sharpness_samples": 2,
                "max_interval_ms": 31.0 + missing,
            }
            for camera, missing in (("cam0", 0), ("cam1", 1))
        ]
        sample_rows = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "camera": "cam0",
                "laplacian_variance": value,
                "intensity_std": 5.0 + value,
                "dark_clip_fraction": 0.01,
                "bright_clip_fraction": 0.02,
                "previous_frame_mae": value / 10.0,
            }
            for value in (10.0, 20.0, 30.0, 40.0)
        ]
        aggregate_rows = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "exact_sync_fraction": 0.9,
            }
        ]
        mocap_rows = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "tracked_fraction": 0.99,
            }
        ]

        result = ANALYSIS.summarize_image_diagnostics(
            camera_rows,
            sample_rows,
            aggregate_rows,
            mocap_rows,
            expected_cameras=2,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["image_sample_count"], 4)
        self.assertAlmostEqual(result[0]["laplacian_variance_p5"], 11.5)
        self.assertAlmostEqual(result[0]["laplacian_variance_p10"], 13.0)
        self.assertAlmostEqual(result[0]["laplacian_variance_median"], 25.0)
        self.assertEqual(result[0]["camera_missing_images"], 1)
        self.assertEqual(result[0]["exact_sync_fraction"], 0.9)
        self.assertEqual(result[0]["mocap_tracked_fraction"], 0.99)

    def test_compute_sim3_run_rows_uses_corrected_evaluation(self):
        spec = SimpleNamespace(
            day="20260805",
            sequence="20260805-122310",
            mocap=Path("/mocap.log"),
            run_dirs=(Path("/result/run1"), Path("/result/run2")),
        )
        reference_positions = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        estimate_positions = reference_positions * 2.0
        evaluation = SimpleNamespace(
            reference_positions=reference_positions,
            reference_quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            estimate_positions=estimate_positions,
        )
        corrected = SimpleNamespace(
            reference_positions=reference_positions,
            estimate_positions=estimate_positions,
            rmse_m=0.5,
        )

        with patch.object(
            ANALYSIS.repeatability, "load_mocap_trajectory", return_value=object()
        ), patch.object(
            ANALYSIS.repeatability, "load_okvis_trajectory", return_value=object()
        ), patch.object(
            ANALYSIS.repeatability, "evaluate_ape", return_value=evaluation
        ), patch.object(
            ANALYSIS.day_analysis,
            "session_fixed_lever",
            return_value=np.asarray([0.1, 0.0, 0.0]),
        ), patch.object(
            ANALYSIS.day_analysis, "apply_effective_lever", return_value=corrected
        ) as apply_lever:
            rows = ANALYSIS.compute_sim3_run_rows([spec])

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["scale_estimate_to_mocap"], 0.5)
        self.assertAlmostEqual(rows[0]["se3_rmse_mm"], 500.0)
        self.assertAlmostEqual(rows[0]["sim3_rmse_mm"], 0.0, places=9)
        apply_lever.assert_called()

    def test_render_population_figures_uses_semantic_names_and_coverage(self):
        sequence_rows = []
        run_rows = []
        for index in range(24):
            sequence = f"20260803-{index:06d}"
            fragmented = index >= 18
            sequence_rows.append(
                {
                    "sequence": sequence,
                    "corrected_ape_median_mm": 1.0 + index**2,
                    "corrected_ape_run1_mm": 1.0 + index**2 * 0.9,
                    "corrected_ape_run2_mm": 1.0 + index**2 * 1.1,
                    "motion_angular_speed_radps_p95": 0.1 + index * 0.05,
                    "motion_angular_speed_radps_p99": 0.2 + index * 0.1,
                    "motion_angular_speed_radps_max": 0.5 + index * 0.2,
                    "motion_angular_above_3_0_fraction": index / 100.0,
                    "motion_angular_above_3_0_event_count": index,
                    "angular_events_per_5min": index / 2.0,
                    "ransac_fail_per_min": 1.0 if not fragmented else 30.0,
                    "landmark_time_span_median_s": 20.0 if not fragmented else 1.0,
                    "loop_attempts_per_min": 2.0 + index,
                    "loop_rejection_fraction": index / 24.0,
                    "quality_initialized_fraction": 0.8 - index / 40.0,
                    "observations_per_landmark": 6.0 - index / 10.0,
                    "laplacian_variance_p5": 500.0 - index * 10.0,
                    "intensity_std_median": 40.0 - index / 2.0,
                    "camera_missing_images": 0,
                    "mocap_tracked_fraction": 1.0,
                    "baseline_over_rotation_p10_cm_per_rad": 10.0 - index / 5.0,
                    "frac_rotation_gt_0p25_baseline_lt_5cm_pct": index,
                    "stereo_landmark_fraction": 0.4 - index / 100.0,
                    "sim3_improvement_pct": index * 2.0,
                    "scale_estimate_to_mocap": 1.0 - index / 100.0,
                    "ape_over_10mm": index >= 4,
                    "visual_fragmentation": fragmented,
                    "scale_instability": index >= 20,
                }
            )
            for run_index in (1, 2):
                run_rows.append(
                    {
                        "sequence": sequence,
                        "run": f"run{run_index}",
                        "quality_initialized_fraction": 0.8 - index / 40.0,
                        "quality_median": 0.1 - index / 500.0,
                        "landmark_time_span_median_s": 20.0 if not fragmented else 1.0,
                    }
                )
        factors = [
            {
                "factor": "angular_p95",
                "label_zh": "角速度 p95",
                "metric": "motion_angular_speed_radps_p95",
                "expected_direction": 1,
                "role": "candidate_trigger",
                "is_proxy": False,
            }
        ]
        correlations = ANALYSIS.compute_correlation_rows(sequence_rows, factors)
        evidence = ANALYSIS.synthesize_evidence(correlations, factors)
        impulse_rows = [
            {
                "sequence": sequence,
                "event_index": 1,
                "event_start_s": 10.0,
                "event_end_s": 11.0,
                "peak_time_s": 10.5,
                "peak_radps": 5.0,
                "first_sustained_10cm_run1_s": 10.7,
                "first_sustained_10cm_run2_s": 10.8,
            }
            for sequence in sorted(ANALYSIS.IMPULSE_SEQUENCES)
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            captured = {}
            add_manifest_figure = ANALYSIS._add_manifest_figure

            def capture_figure(
                figure, output_path, filename, claim, sequence_count, manifest
            ):
                if filename == "tracking_landmark_failure_state.png":
                    captured["tracking_x_limits"] = [
                        axis.get_xlim() for axis in figure.axes
                    ]
                if filename == "angular_impulse_timeline.png":
                    legend = figure.axes[0].get_legend()
                    captured["impulse_legend"] = (
                        []
                        if legend is None
                        else [text.get_text() for text in legend.get_texts()]
                    )
                if filename == "candidate_factor_evidence.png":
                    legend = figure.axes[0].get_legend()
                    captured["evidence_legend"] = (
                        []
                        if legend is None
                        else [text.get_text() for text in legend.get_texts()]
                    )
                add_manifest_figure(
                    figure,
                    output_path,
                    filename,
                    claim,
                    sequence_count,
                    manifest,
                )

            with patch.object(
                ANALYSIS, "_add_manifest_figure", side_effect=capture_figure
            ):
                manifest = ANALYSIS.render_population_figures(
                    output,
                    sequence_rows,
                    run_rows,
                    correlations,
                    evidence,
                    impulse_rows,
                )

            names = {Path(row["path"]).name for row in manifest}
            self.assertIn("ape_angular_velocity_relationships.png", names)
            for row in manifest:
                name = Path(row["path"]).name
                self.assertFalse(name[0].isdigit())
                image = np.asarray(Image.open(output / name), dtype=float)
                self.assertGreater(float(np.var(image)), 0.0)
                if row["claim"] == "angular_impulse_timing":
                    self.assertEqual(row["sequence_count"], 3)
                else:
                    self.assertEqual(row["sequence_count"], 24)
            self.assertTrue(
                all(x_limits[0] >= 0.0 for x_limits in captured["tracking_x_limits"])
            )
            self.assertEqual(
                captured["impulse_legend"],
                [
                    ">3 rad/s event window",
                    "angular-speed peak",
                    "run1 sustained 10 cm drift",
                    "run2 sustained 10 cm drift",
                ],
            )
            self.assertEqual(
                captured["evidence_legend"],
                ["strong", "moderate", "weak", "currently not supported"],
            )

    def test_default_factors_retain_all_candidate_and_downstream_families(self):
        factors = {row["factor"] for row in ANALYSIS.DEFAULT_FACTORS}
        self.assertTrue(
            {
                "angular_speed_p95",
                "angular_speed_p99",
                "angular_speed_max",
                "high_angular_fraction",
                "angular_event_frequency",
                "translation_per_orientation",
                "high_rotation_low_translation",
                "image_edge_content_p5",
                "image_contrast",
                "ransac_fail_rate",
                "loop_attempt_rate",
                "landmark_span",
                "observations_per_landmark",
                "initialized_quality_fraction",
                "stereo_landmark_fraction",
                "camera_missing_images",
                "mocap_tracking_fraction",
                "sim3_improvement",
            }.issubset(factors)
        )

    def test_build_unified_rows_joins_runs_aggregates_and_applies_labels(self):
        baseline_sequence = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "run_count": 2,
                "analysis_duration_s": 60.0,
                "corrected_ape_median_mm": 20.0,
                "motion_angular_above_3_0_event_count": 2,
            }
        ]
        baseline_runs = [
            {"day": "20260803", "sequence": "seq-a", "run": f"run{index}"}
            for index in (1, 2)
        ]
        run_diagnostics = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "run": f"run{index}",
                "run_log_available": True,
                "ransac_fail_per_min": 20.0 + index,
                "landmark_time_span_median_s": 1.0,
                "quality_initialized_fraction": 0.1,
                "observations_per_landmark": 3.0,
            }
            for index in (1, 2)
        ]
        sim3_rows = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "run": f"run{index}",
                "sim3_improvement_pct": 30.0,
                "scale_estimate_to_mocap": 0.5,
            }
            for index in (1, 2)
        ]
        stereo_rows = [
            {
                "sequence": "seq-a",
                "run": f"run{index}",
                "stereo_landmark_fraction": 0.3,
            }
            for index in (1, 2)
        ]
        observability_rows = [
            {
                "sequence": "seq-a",
                "baseline_over_rotation_p10_cm_per_rad": 1.0,
            }
        ]
        image_rows = [
            {
                "day": "20260803",
                "sequence": "seq-a",
                "laplacian_variance_p5": 100.0,
            }
        ]

        sequence_rows, run_rows = ANALYSIS.build_unified_rows(
            baseline_sequence,
            baseline_runs,
            run_diagnostics,
            sim3_rows,
            stereo_rows,
            observability_rows,
            image_rows,
            expected_sequences=1,
            expected_runs=2,
        )

        self.assertEqual(len(run_rows), 2)
        self.assertEqual(len(sequence_rows), 1)
        row = sequence_rows[0]
        self.assertAlmostEqual(row["ransac_fail_per_min"], 21.5)
        self.assertAlmostEqual(row["angular_events_per_5min"], 10.0)
        self.assertTrue(row["ape_over_10mm"])
        self.assertTrue(row["visual_fragmentation"])
        self.assertTrue(row["scale_instability"])

    def test_cli_forwards_paths_days_and_image_sampling(self):
        arguments = [
            "--results-root",
            "/results",
            "--data-root",
            "/data",
            "--output",
            "/output",
            "--days",
            "20260803",
            "20260804",
            "--samples-per-camera",
            "40",
            "--causal-diagnostics-root",
            "/causal",
        ]
        with patch.object(
            ANALYSIS, "run_cross_sample_analysis", return_value={}
        ) as run_analysis:
            result = ANALYSIS.main(arguments)

        self.assertEqual(result, 0)
        run_analysis.assert_called_once_with(
            Path("/results"),
            Path("/data"),
            Path("/output"),
            days=("20260803", "20260804"),
            expected_sequences=None,
            samples_per_camera=40,
            causal_diagnostics_root=Path("/causal"),
        )

    def test_causal_artifact_rows_require_complete_output_set(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tables = output / "tables"
            figures = output / "figures"
            tables.mkdir()
            figures.mkdir()
            for filename in ANALYSIS.CAUSAL_TABLES:
                (tables / filename).write_text("header\n", encoding="utf-8")
            for filename in ANALYSIS.CAUSAL_FIGURES:
                (figures / filename).write_bytes(b"png")
            for filename in ANALYSIS.CAUSAL_REPORTS:
                (output / filename).write_text("# report\n", encoding="utf-8")

            rows = ANALYSIS.causal_artifact_rows(output, sequence_count=24)

            self.assertEqual(
                len(rows),
                len(ANALYSIS.CAUSAL_TABLES)
                + len(ANALYSIS.CAUSAL_FIGURES)
                + len(ANALYSIS.CAUSAL_REPORTS),
            )
            (tables / ANALYSIS.CAUSAL_TABLES[0]).unlink()
            with self.assertRaisesRegex(ValueError, "incomplete causal"):
                ANALYSIS.causal_artifact_rows(output, sequence_count=24)

    def test_causal_evidence_section_covers_all_paths_and_caps_proxies(self):
        event_rows = [{
            "sequence": "20260806-175103",
            "intervention": "baseline",
            "keypoints_total_delta": -1.0,
            "keypoints_total_onset_s": 0.1,
            "gp3p_onset_s": 0.3,
            "rotation_only_minus_relative_pose_inlier_ratio_delta": 0.2,
            "rotation_only_minus_relative_pose_inlier_ratio_onset_s": 0.1,
            "temporal_ray_angle_p10_rad_delta": -0.5,
            "temporal_ray_angle_p10_rad_onset_s": 0.2,
            "gp3p_start_to_model_rotation_rad_delta": 0.4,
            "gp3p_start_to_model_rotation_rad_onset_s": 0.2,
            "visual_observation_removals_delta": 1.0,
            "visual_observation_removals_onset_s": 0.4,
            "active_initialised_landmarks_onset_s": 0.6,
        }]
        model_rows = [{
            "family": family,
            "status": "exploratory_small_n",
            "spearman": 0.7,
        } for family in (
            "feature_availability",
            "triangulation_geometry",
            "prediction_consistency",
            "map_feedback",
        )]
        recovery_rows = [
            {"sequence": "20260806-175103"},
            {"sequence": "20260806-175304"},
        ]

        rows = ANALYSIS.build_causal_hypothesis_evidence(
            event_rows, model_rows, recovery_rows
        )
        markdown = ANALYSIS.render_causal_evidence_section(rows)

        self.assertEqual(
            [row["path"] for row in rows],
            ["H1", "H2_initialisation", "H2_runtime", "H3", "H4"],
        )
        for row in rows:
            self.assertEqual(
                set((
                    "direct_measurement",
                    "temporal_precedence",
                    "dose_relation",
                    "recovery_175103_contrast",
                    "controlled_intervention",
                    "support_level",
                    "limitation",
                )) - row.keys(),
                set(),
            )
            self.assertNotEqual(row["support_level"], "strong")
        self.assertIn("H2 初始化路径", markdown)
        self.assertIn("H2 运行期 3D-2D 路径", markdown)
        self.assertIn("直接测量", markdown)
        self.assertIn("局限性", markdown)

    def test_valid_timing_intervention_is_path_specific_without_bypassing_recovery(self):
        event_rows = [{
            "sequence": "20260806-175103",
            "intervention": "imu_time_offset_ns",
            "intervention_valid": "true",
            "gp3p_onset_s": 0.3,
            "keypoints_total_delta": -1.0,
            "keypoints_total_onset_s": 0.1,
            "rotation_only_minus_relative_pose_inlier_ratio_delta": 0.2,
            "rotation_only_minus_relative_pose_inlier_ratio_onset_s": 0.1,
            "temporal_ray_angle_p10_rad_delta": -0.5,
            "temporal_ray_angle_p10_rad_onset_s": 0.2,
            "gp3p_start_to_model_rotation_rad_delta": 0.4,
            "gp3p_start_to_model_rotation_rad_onset_s": 0.2,
            "visual_observation_removals_delta": 1.0,
            "visual_observation_removals_onset_s": 0.4,
            "active_initialised_landmarks_onset_s": 0.6,
        }]
        model_rows = [{
            "family": family,
            "status": "ok",
            "spearman": 0.7,
        } for family in (
            "feature_availability",
            "triangulation_geometry",
            "prediction_consistency",
            "map_feedback",
        )]
        recovery_rows = [
            {"sequence": "20260806-175103"},
            {"sequence": "20260806-175304"},
        ]

        rows = ANALYSIS.build_causal_hypothesis_evidence(
            event_rows, model_rows, recovery_rows
        )
        by_path = {row["path"]: row for row in rows}

        self.assertEqual(
            by_path["H3"]["controlled_intervention"], "validated"
        )
        self.assertEqual(by_path["H3"]["recovery_175103_contrast"], "not_available")
        for path in ("H1", "H2_initialisation", "H2_runtime", "H3", "H4"):
            self.assertNotEqual(by_path[path]["support_level"], "strong")

    def test_recovery_contrast_requires_target_recovery_and_both_failures(self):
        def event(sequence, recovery):
            return {
                "sequence": sequence,
                "intervention": "baseline",
                "gp3p_onset_s": 0.3,
                "gp3p_start_to_model_rotation_rad_delta": 0.4,
                "gp3p_start_to_model_rotation_rad_onset_s": 0.2,
                "gp3p_start_to_model_rotation_rad_recovery_s": recovery,
            }

        rows = ANALYSIS.build_causal_hypothesis_evidence(
            [
                event("20260806-175103", None),
                event("20260806-175304", None),
                event("20260806-175539", None),
            ],
            [{
                "family": "prediction_consistency",
                "status": "ok",
                "spearman": 0.7,
            }],
            [
                {"sequence": "20260806-175103"},
                {"sequence": "20260806-175304"},
                {"sequence": "20260806-175539"},
            ],
        )
        h3 = {row["path"]: row for row in rows}["H3"]

        self.assertEqual(h3["recovery_175103_contrast"], "not_supported")
        self.assertNotIn(h3["support_level"], {"moderate", "strong"})

    def test_strong_support_requires_specificity_and_replication_flags(self):
        def event(sequence, recovery, **extra):
            return {
                "sequence": sequence,
                "intervention": "imu_time_offset_ns",
                "intervention_valid": "true",
                "gp3p_onset_s": 0.3,
                "gp3p_start_to_model_rotation_rad_delta": 0.4,
                "gp3p_start_to_model_rotation_rad_onset_s": 0.2,
                "gp3p_start_to_model_rotation_rad_recovery_s": recovery,
                **extra,
            }

        event_rows = [
            event("20260806-175103", 2.0),
            event("20260806-175304", None),
            event("20260806-175539", None),
        ]
        model_rows = [{
            "family": "prediction_consistency",
            "status": "ok",
            "spearman": 0.7,
        }]
        recovery_rows = [
            {"sequence": "20260806-175103"},
            {"sequence": "20260806-175304"},
            {"sequence": "20260806-175539"},
        ]

        before = ANALYSIS.build_causal_hypothesis_evidence(
            event_rows, model_rows, recovery_rows
        )
        before_h3 = {row["path"]: row for row in before}["H3"]
        event_rows[0]["specificity_valid"] = "true"
        event_rows[0]["replication_valid"] = "true"
        after = ANALYSIS.build_causal_hypothesis_evidence(
            event_rows, model_rows, recovery_rows
        )
        after_h3 = {row["path"]: row for row in after}["H3"]

        self.assertEqual(before_h3["support_level"], "moderate")
        self.assertEqual(before_h3["specificity"], "not_validated")
        self.assertEqual(before_h3["replication"], "not_validated")
        self.assertEqual(after_h3["support_level"], "strong")
        self.assertEqual(after_h3["specificity"], "validated")
        self.assertEqual(after_h3["replication"], "validated")


if __name__ == "__main__":
    unittest.main()
