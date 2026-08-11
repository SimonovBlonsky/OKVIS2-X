#!/usr/bin/env python3
"""Contract tests for the causal-chain report generator."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("generate_causal_chain_report.py")
TABLES = REPO_ROOT / "workspace/ego2_results/202608_week1_analysis/tables"
DIAGNOSTICS = REPO_ROOT / "workspace/ego2_results/202608_causal_diagnostics"
IMAGE_DELAY = REPO_ROOT / "workspace/ego2_results/202608_image_delay_experiments"

EXPECTED_FIGURES = {
    "01_time_error_amplification.png",
    "02_impulse_matching_degradation.png",
    "03_gp3p_fragmentation_chain.png",
    "04_recovery_contrast.png",
    "05_fragmentation_and_loop_response.png",
    "06_observability_and_scale.png",
    "07_end_to_end_evidence_matrix.png",
    "08_image_delay_intervention.png",
    "09_population_angular_failure_chain.png",
}


class CausalChainReportTest(unittest.TestCase):
    def test_generator_writes_complete_report_with_delay_interventions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--diagnostics-root",
                    str(DIAGNOSTICS),
                    "--tables-root",
                    str(TABLES),
                    "--image-delay-root",
                    str(IMAGE_DELAY),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("高角速度/大量旋转到视觉碎片化与高 APE", report)
            self.assertIn("尚未证明", report)
            self.assertIn("不支持作为独立诱因", report)
            self.assertIn("24 个序列", report)
            self.assertIn("10.743 mm", report)
            self.assertIn("13.374 mm", report)
            self.assertIn("148.168 mm", report)
            self.assertIn("不支持“整台设备统一改成 39.25 ms”", report)
            self.assertIn("预测重投影误差上升 `26/26`", report)
            self.assertIn("198 个同 run 高角事件", report)

            figure_names = {path.name for path in (output / "figures").glob("*.png")}
            self.assertEqual(figure_names, EXPECTED_FIGURES)
            for figure in (output / "figures").glob("*.png"):
                self.assertGreater(figure.stat().st_size, 20_000, figure.name)

            with (output / "tables/evidence_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                evidence = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(evidence), 12)
            levels = {row["evidence_level"] for row in evidence}
            self.assertTrue(
                {"强支持", "中等支持", "尚未证明", "不支持作为独立诱因"}
                <= levels
            )
            self.assertTrue(all(row["source"] for row in evidence))

            with (output / "tables/image_delay_intervention_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                interventions = list(csv.DictReader(stream))
            self.assertEqual(len(interventions), 14)
            self.assertEqual(
                {row["sequence"] for row in interventions},
                {"20260803-183537", "20260803-184027", "20260803-184537"},
            )


if __name__ == "__main__":
    unittest.main()
