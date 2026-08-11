import argparse
import csv
import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.accuracy_analysis.scripts import run_vio_diagnostics as runner


class ReplayRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "okvis_app_synchronous"
        self.binary.touch()
        self.config = self.root / "config.yaml"
        self.config.write_text("image_delay: 0.02487\n", encoding="utf-8")
        self.data_root = self.root / "data"
        self.reference = self.root / "reference"
        self.results = self.root / "results"
        self.sequence = "20260806-175103"
        (self.data_root / "20260806" / f"{self.sequence}_euroc").mkdir(
            parents=True
        )
        mocap_dir = self.reference / "20260806" / "group" / self.sequence
        mocap_dir.mkdir(parents=True)
        (mocap_dir / "mocap_20260806_175053.log").touch()

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self):
        return argparse.Namespace(
            binary=self.binary,
            config=self.config,
            data_root=self.data_root,
            reference_results_root=self.reference,
            results_root=self.results,
            sequences=[self.sequence],
            dataset_manifest=None,
            resume_all=False,
            repeats=1,
            jobs=1,
            dry_run=False,
            skip_complete=False,
        )

    def mark_completed(self, replay):
        replay.diagnostics_dir.mkdir(parents=True)
        for name in runner.DIAGNOSTIC_FILES:
            (replay.diagnostics_dir / name).touch()
        (replay.diagnostics_dir / "vio_diag_metadata.csv").write_text(
            "schema_version,key,value\n1,run_complete,true\n",
            encoding="utf-8",
        )
        (replay.diagnostics_dir / ".vio_diagnostics.complete").touch()
        (replay.run_dir / "okvis2-slam-calib_trajectory.csv").touch()
        (replay.run_dir / "okvis2-slam-calib-final-ba_trajectory.csv").touch()

    def write_dataset_manifest(self, dataset=None):
        manifest = self.root / "datasets.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "experiment_id",
                    "dataset",
                    "sequence",
                    "intervention",
                    "intervention_value",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "experiment_id": "imu-plus-5ms",
                    "dataset": str(
                        dataset
                        or self.data_root
                        / "20260806"
                        / f"{self.sequence}_euroc"
                    ),
                    "sequence": self.sequence,
                    "intervention": "imu_time_offset_ns",
                    "intervention_value": "5000000",
                }
            )
        return manifest

    def command_arguments(self, *extra):
        return [
            "--binary",
            str(self.binary),
            "--config",
            str(self.config),
            "--data-root",
            str(self.data_root),
            "--reference-results-root",
            str(self.reference),
            "--results-root",
            str(self.results),
            *extra,
        ]

    def test_prepare_builds_exact_command_without_creating_output(self):
        replay = runner.prepare_replays(self.arguments())[0]
        self.assertEqual(
            replay.command,
            (
                str(self.binary.resolve()),
                str(self.config.resolve()),
                str((self.data_root / "20260806" / f"{self.sequence}_euroc").resolve()),
                str((self.results / self.sequence / "run1").resolve()),
            ),
        )
        self.assertFalse(replay.run_dir.exists())

    def test_refuses_nonempty_output(self):
        run_dir = self.results / self.sequence / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "existing").touch()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            runner.prepare_replays(self.arguments())

    def test_skip_complete_omits_valid_completed_output(self):
        replay = runner.prepare_replays(self.arguments())[0]
        self.mark_completed(replay)
        arguments = self.arguments()
        arguments.skip_complete = True
        self.assertEqual(runner.prepare_replays(arguments), [])

    def test_skip_complete_still_refuses_partial_output(self):
        run_dir = self.results / self.sequence / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "run.log").touch()
        arguments = self.arguments()
        arguments.skip_complete = True
        with self.assertRaisesRegex(ValueError, "not a valid completed run"):
            runner.prepare_replays(arguments)

    def test_resume_all_discovers_nested_datasets_and_only_returns_pending_runs(self):
        nested_sequence = "20260805-123231"
        nested_dataset = (
            self.data_root
            / "20260805"
            / "morning"
            / f"{nested_sequence}_euroc"
        )
        nested_dataset.mkdir(parents=True)
        nested_mocap = (
            self.reference
            / "20260805"
            / "slow"
            / nested_sequence
            / "mocap_20260805_123232.log"
        )
        nested_mocap.parent.mkdir(parents=True)
        nested_mocap.touch()

        arguments = self.arguments()
        arguments.sequences = None
        arguments.resume_all = True
        arguments.repeats = 2
        initial = runner.prepare_replays(arguments)
        completed = [
            replay for replay in initial if replay.sequence == self.sequence
        ]
        for replay in completed:
            self.mark_completed(replay)

        with patch("builtins.print") as output:
            pending = runner.prepare_replays(arguments)

        self.assertEqual(
            [(replay.sequence, replay.run_name) for replay in pending],
            [(nested_sequence, "run1"), (nested_sequence, "run2")],
        )
        self.assertTrue(all(replay.dataset == nested_dataset.resolve() for replay in pending))
        output.assert_any_call(
            "resume summary: sequences=2, completed_runs=2, pending_runs=2"
        )

    def test_resume_all_only_schedules_missing_repeat(self):
        arguments = self.arguments()
        arguments.sequences = None
        arguments.resume_all = True
        arguments.repeats = 2
        replays = runner.prepare_replays(arguments)
        self.mark_completed(replays[0])

        pending = runner.prepare_replays(arguments)

        self.assertEqual(
            [(replay.sequence, replay.run_name) for replay in pending],
            [(self.sequence, "run2")],
        )

    def test_resume_all_rejects_partial_output(self):
        run_dir = self.results / self.sequence / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "run.log").touch()
        arguments = self.arguments()
        arguments.sequences = None
        arguments.resume_all = True
        with self.assertRaisesRegex(ValueError, "not a valid completed run"):
            runner.prepare_replays(arguments)

    def test_resume_all_rejects_duplicate_dataset_directories(self):
        duplicate = (
            self.data_root / "20260806" / "nested" / f"{self.sequence}_euroc"
        )
        duplicate.mkdir(parents=True)
        arguments = self.arguments()
        arguments.sequences = None
        arguments.resume_all = True
        with self.assertRaisesRegex(ValueError, "expected exactly one dataset"):
            runner.prepare_replays(arguments)

    def test_rejects_missing_binary(self):
        arguments = self.arguments()
        arguments.binary = self.root / "missing-binary"
        with self.assertRaisesRegex(ValueError, "missing binary"):
            runner.prepare_replays(arguments)

    def test_rejects_missing_config(self):
        arguments = self.arguments()
        arguments.config = self.root / "missing-config.yaml"
        with self.assertRaisesRegex(ValueError, "missing config"):
            runner.prepare_replays(arguments)

    def test_rejects_missing_dataset(self):
        arguments = self.arguments()
        arguments.sequences = ["20260806-000000"]
        with self.assertRaisesRegex(ValueError, "missing dataset"):
            runner.prepare_replays(arguments)

    def test_requires_exactly_one_mocap(self):
        duplicate = (
            self.reference
            / "20260806"
            / "other"
            / self.sequence
            / "mocap_20260806_175054.log"
        )
        duplicate.parent.mkdir(parents=True)
        duplicate.touch()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.prepare_replays(self.arguments())

    def test_run_sets_diagnostic_environment_and_rejects_nonzero_exit(self):
        replay = runner.prepare_replays(self.arguments())[0]
        with patch("subprocess.run") as process:
            process.return_value.returncode = 7
            with self.assertRaisesRegex(RuntimeError, "exited with 7"):
                runner.run_replay(replay, self.config, "build")
        environment = process.call_args.kwargs["env"]
        self.assertEqual(environment["OKVIS_DIAGNOSTICS_DIR"], str(replay.diagnostics_dir))
        self.assertEqual(
            environment["OKVIS_DIAGNOSTICS_RUN_ID"],
            f"{self.sequence}-run1",
        )
        self.assertEqual(environment["OKVIS_DIAGNOSTICS_BUILD_ID"], "build")
        self.assertEqual(environment["QT_QPA_PLATFORM"], "offscreen")

    def test_validate_requires_run_complete(self):
        replay = runner.prepare_replays(self.arguments())[0]
        replay.diagnostics_dir.mkdir(parents=True)
        for name in runner.DIAGNOSTIC_FILES:
            (replay.diagnostics_dir / name).touch()
        with (replay.diagnostics_dir / "vio_diag_metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["schema_version", "key", "value"])
            writer.writerow([1, "run_complete", "false"])
        (replay.diagnostics_dir / ".vio_diagnostics.complete").touch()
        (replay.run_dir / "okvis2-slam-calib_trajectory.csv").touch()
        (replay.run_dir / "okvis2-slam-calib-final-ba_trajectory.csv").touch()
        with self.assertRaisesRegex(ValueError, "run_complete"):
            runner.validate_completed_run(replay)

    def test_validate_rejects_writer_failure(self):
        replay = runner.prepare_replays(self.arguments())[0]
        replay.diagnostics_dir.mkdir(parents=True)
        for name in runner.DIAGNOSTIC_FILES:
            (replay.diagnostics_dir / name).touch()
        (replay.diagnostics_dir / "vio_diag_metadata.csv").write_text(
            "schema_version,key,value\n"
            "1,run_complete,true\n"
            "1,writer_failed,true\n",
            encoding="utf-8",
        )
        (replay.diagnostics_dir / ".vio_diagnostics.complete").touch()
        (replay.run_dir / "okvis2-slam-calib_trajectory.csv").touch()
        (replay.run_dir / "okvis2-slam-calib-final-ba_trajectory.csv").touch()
        with self.assertRaisesRegex(ValueError, "writer reported failure"):
            runner.validate_completed_run(replay)

    def test_validate_does_not_treat_final_ba_as_online_trajectory(self):
        replay = runner.prepare_replays(self.arguments())[0]
        replay.diagnostics_dir.mkdir(parents=True)
        for name in runner.DIAGNOSTIC_FILES:
            (replay.diagnostics_dir / name).touch()
        (replay.diagnostics_dir / "vio_diag_metadata.csv").write_text(
            "schema_version,key,value\n1,run_complete,true\n",
            encoding="utf-8",
        )
        (replay.diagnostics_dir / ".vio_diagnostics.complete").touch()
        (replay.run_dir / "okvis2-slam-calib-final-ba_trajectory.csv").touch()
        with self.assertRaisesRegex(ValueError, "online trajectory"):
            runner.validate_completed_run(replay)

    def test_dry_run_prints_commands_without_starting_replays(self):
        with patch.object(runner, "run_replay") as run_replay, patch(
            "builtins.print"
        ) as output:
            result = runner.main(
                self.command_arguments(
                    "--sequences", self.sequence, "--repeats", "1", "--dry-run"
                )
            )
        self.assertEqual(result, 0)
        run_replay.assert_not_called()
        output.assert_called_once_with(
            " ".join(runner.prepare_replays(self.arguments())[0].command)
        )
        self.assertFalse(self.results.exists())

    def test_success_writes_manifest_and_prints_completion(self):
        arguments = self.arguments()
        arguments.sequences = None
        arguments.dataset_manifest = self.write_dataset_manifest()
        replay = runner.prepare_replays(arguments)[0]

        def completed_process(*unused_args, **unused_kwargs):
            replay.diagnostics_dir.mkdir(parents=True)
            for name in runner.DIAGNOSTIC_FILES:
                (replay.diagnostics_dir / name).touch()
            (replay.diagnostics_dir / "vio_diag_metadata.csv").write_text(
                "schema_version,key,value\n1,run_complete,true\n",
                encoding="utf-8",
            )
            (replay.diagnostics_dir / ".vio_diagnostics.complete").touch()
            (replay.run_dir / "okvis2-slam-calib_trajectory.csv").touch()
            (
                replay.run_dir / "okvis2-slam-calib-final-ba_trajectory.csv"
            ).touch()
            return SimpleNamespace(returncode=0)

        with patch("subprocess.run", side_effect=completed_process) as process, patch(
            "builtins.print"
        ) as output:
            runner.run_replay(replay, self.config, "test-build")

        manifest_path = replay.run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["experiment_id"], "imu-plus-5ms")
        self.assertEqual(manifest["dataset"], str(replay.dataset))
        self.assertEqual(manifest["sequence"], self.sequence)
        self.assertEqual(manifest["intervention"], "imu_time_offset_ns")
        self.assertEqual(manifest["intervention_value"], "5000000")
        self.assertEqual(manifest["return_code"], 0)
        self.assertIn("diagnostics/.vio_diagnostics.complete", manifest["produced_files"])
        self.assertEqual(
            process.call_args.kwargs["env"]["OKVIS_DIAGNOSTICS_RUN_ID"],
            "imu-plus-5ms-run1",
        )
        output.assert_called_once_with(
            f"completed {replay.experiment_id}/{replay.run_name}: {manifest_path}"
        )

    def test_dataset_manifest_controls_dataset_and_output_identity(self):
        arguments = self.arguments()
        arguments.sequences = None
        arguments.dataset_manifest = self.write_dataset_manifest()
        replay = runner.prepare_replays(arguments)[0]
        self.assertEqual(replay.experiment_id, "imu-plus-5ms")
        self.assertEqual(replay.intervention, "imu_time_offset_ns")
        self.assertEqual(replay.intervention_value, "5000000")
        self.assertEqual(
            replay.run_dir,
            (self.results / "imu-plus-5ms" / "run1").resolve(),
        )

    def test_sequences_and_dataset_manifest_are_mutually_exclusive(self):
        manifest = self.write_dataset_manifest()
        parser = runner.build_argument_parser()
        errors = io.StringIO()
        with patch("sys.stderr", errors), self.assertRaises(SystemExit):
            parser.parse_args(
                self.command_arguments(
                    "--sequences", self.sequence,
                    "--dataset-manifest", str(manifest),
                )
            )
        self.assertIn("not allowed with argument", errors.getvalue())

    def test_dataset_manifest_rejects_duplicate_experiment_ids(self):
        manifest = self.write_dataset_manifest()
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(
                "imu-plus-5ms,/unused,20260806-000000,"
                "imu_time_offset_ns,5000000\n"
            )
        arguments = self.arguments()
        arguments.sequences = None
        arguments.dataset_manifest = manifest
        with self.assertRaisesRegex(ValueError, "duplicate experiment_id"):
            runner.prepare_replays(arguments)

    def test_dataset_manifest_rejects_empty_required_field(self):
        manifest = self.write_dataset_manifest()
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text(
            text.replace("imu_time_offset_ns", ""), encoding="utf-8"
        )
        arguments = self.arguments()
        arguments.sequences = None
        arguments.dataset_manifest = manifest
        with self.assertRaisesRegex(ValueError, "empty intervention"):
            runner.prepare_replays(arguments)


if __name__ == "__main__":
    unittest.main()
