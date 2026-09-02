import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.accuracy_analysis.scripts import prepare_imu_time_offset_variants as variants


class ImuTimeOffsetVariantTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "20260806-175103_euroc"
        self.sensor_root = self.source
        for name in ("cam0", "cam1", "lidar0"):
            directory = self.sensor_root / name
            directory.mkdir(parents=True)
            (directory / "sensor.yaml").write_text(
                f"sensor: {name}\n", encoding="utf-8"
            )
        self.imu0 = self.sensor_root / "imu0"
        self.imu0.mkdir()
        (self.imu0 / "sensor.yaml").write_text(
            "sensor: imu0\n", encoding="utf-8"
        )
        self.imu_csv = self.imu0 / "data.csv"
        self.imu_csv.write_text(
            "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n"
            "100000000,1,2,3,4,5,6\n"
            "200000000,7,8,9,10,11,12\n",
            encoding="utf-8",
        )
        (self.source / "dataset-note.txt").write_text(
            "source remains immutable\n", encoding="utf-8"
        )
        self.target = self.root / "variants" / "imu-plus-25ns"

    def tearDown(self):
        self.temporary.cleanup()

    def create_variant(self, offset_ns=25):
        return variants.create_imu_time_offset_variant(
            source_dataset=self.source,
            target_dataset=self.target,
            offset_ns=offset_ns,
            experiment_id="imu-plus-25ns",
            sequence="20260806-175103",
        )

    def test_shifts_only_imu_timestamps_and_symlinks_other_content(self):
        source_imu = self.imu_csv.read_bytes()
        manifest = self.create_variant()

        self.assertEqual(self.imu_csv.read_bytes(), source_imu)
        for name in ("cam0", "cam1", "lidar0"):
            link = self.target / name
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (self.sensor_root / name).resolve())
        root_link = self.target / "dataset-note.txt"
        self.assertTrue(root_link.is_symlink())
        self.assertEqual(root_link.resolve(), (self.source / root_link.name).resolve())

        target_imu = self.target / "imu0"
        self.assertTrue(target_imu.is_dir())
        self.assertFalse(target_imu.is_symlink())
        self.assertFalse((target_imu / "data.csv").is_symlink())
        self.assertTrue((target_imu / "sensor.yaml").is_symlink())
        self.assertEqual(
            (target_imu / "data.csv").read_text(encoding="utf-8"),
            "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n"
            "100000025,1,2,3,4,5,6\n"
            "200000025,7,8,9,10,11,12\n",
        )

        manifest_path = self.target / "variant_manifest.json"
        self.assertEqual(manifest_path, manifest)
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(recorded["experiment_id"], "imu-plus-25ns")
        self.assertEqual(recorded["dataset"], str(self.target.resolve()))
        self.assertEqual(recorded["source_dataset"], str(self.source.resolve()))
        self.assertEqual(recorded["sequence"], "20260806-175103")
        self.assertEqual(recorded["intervention"], "imu_time_offset_ns")
        self.assertEqual(recorded["intervention_value"], "25")
        self.assertEqual(recorded["row_count"], 2)

    def test_rejects_existing_target(self):
        self.target.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "target already exists"):
            self.create_variant()

    def test_rejects_broken_symlink_target(self):
        self.target.parent.mkdir(parents=True)
        self.target.symlink_to(self.root / "absent-target", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "target already exists"):
            self.create_variant()

    def test_rejects_missing_required_directory(self):
        missing = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "missing source dataset"):
            variants.create_imu_time_offset_variant(
                missing, self.target, 25, "missing", "20260806-175103"
            )
        self.imu0.rename(self.sensor_root / "renamed-imu0")
        with self.assertRaisesRegex(ValueError, "missing IMU directory"):
            self.create_variant()

    def test_plan_rejects_missing_stereo_camera_directory(self):
        for camera in ("cam0", "cam1"):
            with self.subTest(camera=camera):
                original = self.sensor_root / camera
                renamed = self.sensor_root / f"renamed-{camera}"
                original.rename(renamed)
                try:
                    with self.assertRaisesRegex(ValueError, camera):
                        variants.plan_variants(
                            self.source, self.root / "planned", [0]
                        )
                finally:
                    renamed.rename(original)

    def test_create_rejects_missing_stereo_camera_directory(self):
        for camera in ("cam0", "cam1"):
            with self.subTest(camera=camera):
                self.target = self.root / "variants" / f"missing-{camera}"
                original = self.sensor_root / camera
                renamed = self.sensor_root / f"renamed-{camera}"
                original.rename(renamed)
                try:
                    with self.assertRaisesRegex(ValueError, camera):
                        self.create_variant()
                    self.assertFalse(self.target.exists())
                finally:
                    renamed.rename(original)

    def test_preserves_nested_mav0_layout(self):
        nested_source = self.root / "nested_euroc"
        nested_sensor_root = nested_source / "mav0"
        nested_cam = nested_sensor_root / "cam0"
        nested_cam1 = nested_sensor_root / "cam1"
        nested_imu = nested_sensor_root / "imu0"
        nested_cam.mkdir(parents=True)
        nested_cam1.mkdir()
        nested_imu.mkdir()
        (nested_imu / "data.csv").write_text(
            "#timestamp [ns],w_x\n100,1\n", encoding="utf-8"
        )
        nested_target = self.root / "nested-variant"
        variants.create_imu_time_offset_variant(
            nested_source, nested_target, 10, "nested", "nested"
        )
        self.assertTrue((nested_target / "mav0" / "cam0").is_symlink())
        self.assertEqual(
            (nested_target / "mav0" / "imu0" / "data.csv").read_text(
                encoding="utf-8"
            ),
            "#timestamp [ns],w_x\n110,1\n",
        )

    def test_batch_cli_creates_deterministic_variants_and_dataset_manifest(self):
        output_root = self.root / "batch"
        with patch("builtins.print"):
            result = variants.main(
                [
                    "--source-dataset", str(self.source),
                    "--output-root", str(output_root),
                    "--offsets-ms", "-1", "0", "2",
                ]
            )
        self.assertEqual(result, 0)
        expected = (
            ("m1ms", "-1000000"),
            ("0ms", "0"),
            ("p2ms", "2000000"),
        )
        for token, offset_ns in expected:
            target = output_root / f"{self.source.name}.imu-offset-{token}"
            self.assertTrue(target.is_dir())
            manifest = json.loads(
                (target / "variant_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["experiment_id"],
                f"20260806-175103-imu-offset-{token}",
            )
            self.assertEqual(manifest["intervention_value"], offset_ns)

        with (output_root / "dataset_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["experiment_id"] for row in rows],
            [
                "20260806-175103-imu-offset-m1ms",
                "20260806-175103-imu-offset-0ms",
                "20260806-175103-imu-offset-p2ms",
            ],
        )

    def test_batch_cli_dry_run_does_not_create_output(self):
        output_root = self.root / "dry-run"
        with patch("builtins.print") as output:
            result = variants.main(
                [
                    "--source-dataset", str(self.source),
                    "--output-root", str(output_root),
                    "--offsets-ms", "0",
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(output_root.exists())
        output.assert_called_once_with(
            "would create 20260806-175103-imu-offset-0ms: "
            f"{output_root / (self.source.name + '.imu-offset-0ms')} (0 ns)"
        )

    def test_rejects_bad_imu_row_without_leaving_target(self):
        self.imu_csv.write_text(
            "#timestamp [ns],w_x\nnot-an-integer,1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "invalid timestamp") as raised:
            self.create_variant()
        self.assertFalse(self.target.exists())
        temporary = list(
            self.target.parent.glob(f".{self.target.name}.tmp-*")
        )
        self.assertEqual(len(temporary), 1)
        self.assertIn(str(temporary[0]), str(raised.exception))

    def test_rejects_negative_shifted_timestamp_without_leaving_target(self):
        with self.assertRaisesRegex(ValueError, "negative timestamp") as raised:
            self.create_variant(offset_ns=-100000001)
        self.assertFalse(self.target.exists())
        temporary = list(
            self.target.parent.glob(f".{self.target.name}.tmp-*")
        )
        self.assertEqual(len(temporary), 1)
        self.assertIn(str(temporary[0]), str(raised.exception))

    def test_publish_fsync_failure_rolls_target_back_to_temporary(self):
        real_fsync_directory = variants.fsync_directory

        def fail_after_publish(path):
            path = Path(path)
            if path == self.target.parent and self.target.exists():
                raise OSError("simulated post-publish fsync failure")
            real_fsync_directory(path)

        with patch.object(
            variants, "fsync_directory", side_effect=fail_after_publish
        ):
            with self.assertRaisesRegex(
                OSError, "incomplete temporary variant retained"
            ) as raised:
                self.create_variant()

        self.assertFalse(self.target.exists())
        temporary = list(
            self.target.parent.glob(f".{self.target.name}.tmp-*")
        )
        self.assertEqual(len(temporary), 1)
        self.assertTrue((temporary[0] / "variant_manifest.json").is_file())
        self.assertIn(str(temporary[0]), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
