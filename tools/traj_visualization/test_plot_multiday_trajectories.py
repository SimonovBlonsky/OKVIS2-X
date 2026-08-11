#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


SCRIPT_PATH = Path(__file__).with_name("plot_multiday_trajectories.py")
SPEC = importlib.util.spec_from_file_location("plot_multiday_trajectories", SCRIPT_PATH)
VISUALIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VISUALIZER
SPEC.loader.exec_module(VISUALIZER)


TRAJECTORY = "okvis2-slam-calib-final-ba_trajectory.csv"


class MultidayTrajectoryVisualizationTest(unittest.TestCase):

    def test_discovers_flat_and_grouped_sequences_with_two_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            data = root / "data"
            layouts = (
                ("20260803", (), "20260803-120000"),
                ("20260805", ("fast",), "20260805-130000"),
            )
            for day, groups, sequence in layouts:
                sequence_dir = results / day
                for group in groups:
                    sequence_dir /= group
                sequence_dir /= sequence
                sequence_dir.mkdir(parents=True)
                (sequence_dir / "mocap_test.log").write_text("mocap\n", encoding="utf-8")
                for run in ("run1", "run2"):
                    run_dir = sequence_dir / run
                    run_dir.mkdir()
                    (run_dir / TRAJECTORY).write_text("trajectory\n", encoding="utf-8")
                (data / day / f"{sequence}_euroc").mkdir(parents=True)

            specs = VISUALIZER.discover_multiday_sequences(
                results,
                data,
                days=("20260803", "20260805"),
            )

        self.assertEqual(
            [(spec.day, spec.group, spec.sequence) for spec in specs],
            [
                ("20260803", "unclassified", "20260803-120000"),
                ("20260805", "fast", "20260805-130000"),
            ],
        )
        self.assertTrue(all(len(spec.run_dirs) == 2 for spec in specs))

    def test_discovery_rejects_duplicate_mocap_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = "20260803-120000"
            sequence_dir = root / "results" / "20260803" / sequence
            sequence_dir.mkdir(parents=True)
            for name in ("mocap_a.log", "mocap_b.log"):
                (sequence_dir / name).write_text("mocap\n", encoding="utf-8")
            run_dir = sequence_dir / "run1"
            run_dir.mkdir()
            (run_dir / TRAJECTORY).write_text("trajectory\n", encoding="utf-8")
            (root / "data" / "20260803" / f"{sequence}_euroc").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "expected one mocap log"):
                VISUALIZER.discover_multiday_sequences(
                    root / "results",
                    root / "data",
                    days=("20260803",),
                )

    def test_render_pages_cover_each_sequence_once_without_fixed_page_count(self):
        sequences = []
        reference = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]])
        for index in range(5):
            runs = (
                VISUALIZER.AlignedRun(
                    "run1", reference + index, reference, 1.0 + index
                ),
                VISUALIZER.AlignedRun(
                    "run2", reference + index + 0.1, reference, 1.2 + index
                ),
            )
            sequences.append(
                VISUALIZER.AlignedSequence(
                    "20260803", f"20260803-{index:06d}", runs
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = VISUALIZER.render_trajectory_pages(
                output, sequences, max_panels_per_page=3, dpi=72
            )

            covered = []
            for row in manifest:
                path = Path(row["path"])
                image = np.asarray(Image.open(path), dtype=float)
                self.assertGreater(float(np.var(image)), 0.0)
                covered.extend(row["sequences"].split(";"))

        self.assertEqual(len(manifest), 2)
        self.assertEqual(sorted(covered), sorted(item.sequence for item in sequences))
        self.assertEqual(len(covered), len(set(covered)))
        self.assertTrue(all(not Path(row["path"]).name[0].isdigit() for row in manifest))

    def test_load_aligned_sequence_applies_the_confirmed_reference_correction(self):
        spec = SimpleNamespace(
            day="20260805",
            sequence="20260805-122310",
            mocap=Path("/mocap.log"),
            run_dirs=(Path("/results/run1"), Path("/results/run2")),
        )
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]
        )
        estimate = reference + 0.1
        evaluation = SimpleNamespace(
            reference_positions=reference,
            reference_quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            estimate_positions=estimate,
        )
        corrected = SimpleNamespace(
            reference_positions=reference + 1.0,
            estimate_positions=estimate + 1.0,
            rmse_m=0.012,
        )

        with patch.object(
            VISUALIZER.repeatability, "load_mocap_trajectory", return_value=object()
        ), patch.object(
            VISUALIZER.repeatability, "load_okvis_trajectory", return_value=object()
        ), patch.object(
            VISUALIZER.repeatability, "evaluate_ape", return_value=evaluation
        ), patch.object(
            VISUALIZER.day_analysis,
            "session_fixed_lever",
            return_value=np.asarray([0.1, 0.0, 0.0]),
        ), patch.object(
            VISUALIZER.day_analysis, "apply_effective_lever", return_value=corrected
        ) as apply_lever:
            item = VISUALIZER.load_aligned_sequence(spec)

        self.assertEqual(item.sequence, "20260805-122310")
        self.assertEqual([run.name for run in item.runs], ["run1", "run2"])
        self.assertAlmostEqual(item.runs[0].ape_rmse_mm, 12.0)
        np.testing.assert_allclose(item.runs[0].reference_positions, reference + 1.0)
        self.assertEqual(apply_lever.call_count, 2)

    def test_generate_multiday_pages_validates_population_and_writes_manifest(self):
        specs = [
            SimpleNamespace(day="20260803", sequence=f"seq-{index:02d}")
            for index in range(2)
        ]
        aligned = [
            VISUALIZER.AlignedSequence(spec.day, spec.sequence, tuple())
            for spec in specs
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            VISUALIZER, "discover_multiday_sequences", return_value=specs
        ), patch.object(
            VISUALIZER, "load_aligned_sequence", side_effect=aligned
        ), patch.object(
            VISUALIZER,
            "render_trajectory_pages",
            return_value=[
                {
                    "path": str(Path(directory) / "trajectory_overview_part_01.png"),
                    "kind": "figure",
                    "claim": "trajectory_population_overview",
                    "sequence_count": 2,
                    "sequences": "seq-00;seq-01",
                }
            ],
        ):
            manifest = VISUALIZER.generate_multiday_trajectory_pages(
                Path("/results"),
                Path("/data"),
                Path(directory),
                days=("20260803",),
                expected_sequences=2,
            )

            self.assertEqual(len(manifest), 1)
            self.assertTrue((Path(directory) / "trajectory_manifest.csv").is_file())

            with self.assertRaisesRegex(ValueError, "expected 3 sequences, found 2"):
                VISUALIZER.generate_multiday_trajectory_pages(
                    Path("/results"),
                    Path("/data"),
                    Path(directory),
                    days=("20260803",),
                    expected_sequences=3,
                )

    def test_cli_forwards_paths_days_and_render_options(self):
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
            "--max-panels-per-page",
            "4",
            "--dpi",
            "90",
        ]
        with patch.object(
            VISUALIZER, "generate_multiday_trajectory_pages", return_value=[]
        ) as generate:
            result = VISUALIZER.main(arguments)

        self.assertEqual(result, 0)
        generate.assert_called_once_with(
            Path("/results"),
            Path("/data"),
            Path("/output"),
            days=("20260803", "20260804"),
            expected_sequences=None,
            max_panels_per_page=4,
            dpi=90,
        )


if __name__ == "__main__":
    unittest.main()
