#!/usr/bin/env python3
"""Render corrected run1/run2/mocap trajectory pages for EGO2 multiday results."""

import argparse
import os
import csv
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "okvis_multiday_traj_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "okvis_multiday_traj_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.accuracy_analysis.scripts import analyze_multiday
from tools.accuracy_analysis.scripts import mocap_reference_correction as day_analysis
from tools.accuracy_analysis.scripts import analyze_repeatability as repeatability


@dataclass(frozen=True)
class AlignedRun:
    name: str
    estimate_positions: np.ndarray
    reference_positions: np.ndarray
    ape_rmse_mm: float


@dataclass(frozen=True)
class AlignedSequence:
    day: str
    sequence: str
    runs: tuple[AlignedRun, ...]


def discover_multiday_sequences(
    results_root: Path,
    data_root: Path,
    *,
    days: tuple[str, ...] = analyze_multiday.DEFAULT_DAYS,
):
    return analyze_multiday.discover_sequences(
        Path(results_root), Path(data_root), days=days
    )


def load_aligned_sequence(spec) -> AlignedSequence:
    if len(spec.run_dirs) != 2:
        raise ValueError(
            f"{spec.sequence}: expected two final-BA runs, found {len(spec.run_dirs)}"
        )
    reference = repeatability.load_mocap_trajectory(spec.mocap)
    lever = day_analysis.session_fixed_lever(
        spec.sequence, day_analysis.FIXED_DIAGNOSTIC_LEVER_M
    )
    runs = []
    for run_dir_value in sorted(spec.run_dirs):
        run_dir = Path(run_dir_value)
        trajectory = repeatability.load_okvis_trajectory(
            run_dir / day_analysis.FINAL_BA_FILE
        )
        evaluation = repeatability.evaluate_ape(
            reference, trajectory, max_diff=0.01
        )
        corrected = day_analysis.apply_effective_lever(
            evaluation.reference_positions,
            evaluation.reference_quaternions_wxyz,
            evaluation.estimate_positions,
            lever,
        )
        runs.append(
            AlignedRun(
                run_dir.name,
                corrected.estimate_positions,
                corrected.reference_positions,
                float(corrected.rmse_m * 1000.0),
            )
        )
    return AlignedSequence(spec.day, spec.sequence, tuple(runs))


def _xy(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or len(positions) < 2 or positions.shape[1] not in {2, 3}:
        raise ValueError("trajectory positions must be an N x 2 or N x 3 array")
    if not np.isfinite(positions).all():
        raise ValueError("trajectory positions must be finite")
    return positions[:, :2]


def _draw_sequence(axis: plt.Axes, item: AlignedSequence) -> None:
    if len(item.runs) != 2:
        raise ValueError(f"{item.sequence}: expected two aligned runs")
    reference = _xy(item.runs[0].reference_positions)
    axis.plot(
        reference[:, 0],
        reference[:, 1],
        color="#202124",
        linewidth=2.0,
        label="mocap",
    )
    styles = (("#147d64", "run1"), ("#b3261e", "run2"))
    for run, (color, label) in zip(item.runs, styles):
        estimate = _xy(run.estimate_positions)
        axis.plot(
            estimate[:, 0],
            estimate[:, 1],
            color=color,
            linewidth=1.3,
            alpha=0.9,
            label=f"{label}: {run.ape_rmse_mm:.1f} mm",
        )
        axis.scatter(
            estimate[-1, 0], estimate[-1, 1], marker="x", s=28, color=color
        )
    axis.scatter(
        reference[0, 0],
        reference[0, 1],
        marker="o",
        s=28,
        color="#2457a6",
        zorder=4,
    )
    axis.set_title(item.sequence, fontsize=10)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, color="#d9dde0", linewidth=0.55, alpha=0.8)
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=7, loc="best")


def render_trajectory_pages(
    output: Path,
    sequences: list[AlignedSequence],
    *,
    max_panels_per_page: int = 6,
    dpi: int = 180,
) -> list[dict]:
    if not sequences:
        raise ValueError("cannot render empty trajectory sequence list")
    if max_panels_per_page <= 0:
        raise ValueError("max_panels_per_page must be positive")
    ordered = sorted(sequences, key=lambda item: (item.day, item.sequence))
    sequence_names = [item.sequence for item in ordered]
    if len(sequence_names) != len(set(sequence_names)):
        raise ValueError("duplicate trajectory sequence")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for page_index, start in enumerate(
        range(0, len(ordered), max_panels_per_page), 1
    ):
        page_items = ordered[start : start + max_panels_per_page]
        columns = min(3, len(page_items))
        rows = int(np.ceil(len(page_items) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.6 * columns, 4.8 * rows),
            squeeze=False,
        )
        for axis, item in zip(axes.flat, page_items):
            _draw_sequence(axis, item)
        for axis in axes.flat[len(page_items) :]:
            axis.set_axis_off()
        figure.suptitle(
            f"Final-BA trajectories: run1 / run2 / mocap (part {page_index})",
            fontsize=15,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = output / f"trajectory_overview_part_{page_index:02d}.png"
        figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(figure)
        manifest.append(
            {
                "path": str(path),
                "kind": "figure",
                "claim": "trajectory_population_overview",
                "sequence_count": len(page_items),
                "sequences": ";".join(item.sequence for item in page_items),
            }
        )
    return manifest


def generate_multiday_trajectory_pages(
    results_root: Path,
    data_root: Path,
    output: Path,
    *,
    days: tuple[str, ...] = analyze_multiday.DEFAULT_DAYS,
    expected_sequences: int | None = analyze_multiday.DEFAULT_EXPECTED_SEQUENCES,
    max_panels_per_page: int = 6,
    dpi: int = 180,
) -> list[dict]:
    specs = discover_multiday_sequences(results_root, data_root, days=days)
    if expected_sequences is not None and len(specs) != expected_sequences:
        raise ValueError(
            f"expected {expected_sequences} sequences, found {len(specs)}"
        )
    aligned = [load_aligned_sequence(spec) for spec in specs]
    output = Path(output)
    manifest = render_trajectory_pages(
        output,
        aligned,
        max_panels_per_page=max_panels_per_page,
        dpi=dpi,
    )
    manifest_path = output / "trajectory_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=analyze_multiday.DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--data-root", type=Path, default=analyze_multiday.DEFAULT_DATA_ROOT
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=analyze_multiday.DEFAULT_OUTPUT / "figures",
    )
    parser.add_argument("--days", nargs="+", default=list(analyze_multiday.DEFAULT_DAYS))
    parser.add_argument("--max-panels-per-page", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    days = tuple(arguments.days)
    expected = (
        analyze_multiday.DEFAULT_EXPECTED_SEQUENCES
        if days == analyze_multiday.DEFAULT_DAYS
        else None
    )
    manifest = generate_multiday_trajectory_pages(
        arguments.results_root,
        arguments.data_root,
        arguments.output,
        days=days,
        expected_sequences=expected,
        max_panels_per_page=arguments.max_panels_per_page,
        dpi=arguments.dpi,
    )
    print(
        f"Wrote {len(manifest)} trajectory overview pages to {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
