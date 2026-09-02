#!/usr/bin/env python3

import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "okvis_traj_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "okvis_traj_cache")
)

import numpy as np
from evo.core.trajectory import PoseTrajectory3D


CSV_FIELDS = (
    "timestamp",
    "p_WS_W_x",
    "p_WS_W_y",
    "p_WS_W_z",
    "q_WS_x",
    "q_WS_y",
    "q_WS_z",
    "q_WS_w",
)
ORIGIN_INSET_BOUNDS = (0.04, 0.07, 0.30, 0.30)


@dataclass(frozen=True)
class TrajectoryInput:
    device: str
    sequence: str
    path: Path


@dataclass(frozen=True)
class TrajectoryMetrics:
    origin_error_m: float
    total_distance_m: float
    error_percent: float


@dataclass(frozen=True)
class LoadedTrajectory:
    source: TrajectoryInput
    trajectory: PoseTrajectory3D
    metrics: TrajectoryMetrics


def discover_trajectories(root: Path) -> list[TrajectoryInput]:
    inputs = []
    for path in root.glob("*/*/results/*final-ba_trajectory.csv"):
        relative = path.relative_to(root)
        device_match = re.search(r"EGO\d+", relative.parts[0], re.IGNORECASE)
        if device_match is None:
            continue
        inputs.append(
            TrajectoryInput(device_match.group(0).upper(), relative.parts[1], path)
        )
    return sorted(
        inputs,
        key=lambda item: (int(item.device.removeprefix("EGO")), item.sequence),
    )


def load_okvis_trajectory(path: Path) -> PoseTrajectory3D:
    rows = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, skipinitialspace=True)
        missing = [field for field in CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing CSV fields: {', '.join(missing)}")
        for row_number, row in enumerate(reader, 2):
            try:
                rows.append([float(row[field]) for field in CSV_FIELDS])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: invalid trajectory value") from error
    if len(rows) < 2:
        raise ValueError(f"{path}: trajectory contains fewer than two poses")

    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: trajectory contains non-finite values")
    timestamps = values[:, 0] / 1e9
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")

    quaternions_wxyz = values[:, [7, 4, 5, 6]]
    return PoseTrajectory3D(values[:, 1:4], quaternions_wxyz, timestamps)


def calculate_metrics(positions: np.ndarray) -> TrajectoryMetrics:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an N x 3 array with at least two poses")
    total_distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    origin_error = float(np.linalg.norm(positions[-1] - positions[0]))
    error_percent = 100.0 * origin_error / total_distance if total_distance else 0.0
    return TrajectoryMetrics(origin_error, total_distance, error_percent)


def origin_zoom_limits(
    positions: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an N x 3 array with at least two poses")
    start = positions[0]
    end = positions[-1]
    center = (start[:2] + end[:2]) / 2.0
    half_span = max(
        float(np.max(np.abs(end[:2] - start[:2]))) * 1.5,
        float(np.linalg.norm(end - start)) * 1.5,
        0.03,
    )
    return (
        (float(center[0] - half_span), float(center[0] + half_span)),
        (float(center[1] - half_span), float(center[1] + half_span)),
    )


def _load_plot_modules():
    from evo.tools.settings import SETTINGS

    SETTINGS.plot_backend = "Agg"
    from evo.tools import plot
    import matplotlib.pyplot as plt

    return plot, plt


def _draw_origin_inset(ax, item: LoadedTrajectory, evo_plot, color: str) -> None:
    positions = item.trajectory.positions_xyz
    inset = ax.inset_axes(ORIGIN_INSET_BOUNDS)
    evo_plot.traj(
        inset,
        evo_plot.PlotMode.xy,
        item.trajectory,
        color=color,
        alpha=0.95,
    )
    inset.scatter(
        positions[0, 0],
        positions[0, 1],
        s=82,
        marker="o",
        color="#17823b",
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )
    inset.scatter(
        positions[-1, 0],
        positions[-1, 1],
        s=62,
        marker="X",
        color="#c83e3e",
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
    )
    inset.plot(
        [positions[0, 0], positions[-1, 0]],
        [positions[0, 1], positions[-1, 1]],
        linestyle=":",
        linewidth=1.3,
        color="#5f6368",
        alpha=0.9,
        zorder=3,
    )
    x_limits, y_limits = origin_zoom_limits(positions)
    inset.set_xlim(x_limits)
    inset.set_ylim(y_limits)
    inset.set_aspect("equal", adjustable="box")
    inset.set_title("Start / End zoom", fontsize=8, pad=3)
    inset.tick_params(axis="both", labelsize=6)
    inset.grid(True, color="#d9dde0", linewidth=0.55, alpha=0.9)
    inset.set_facecolor("#fbfcfd")
    for spine in inset.spines.values():
        spine.set_edgecolor("#5f6368")
        spine.set_linewidth(1.0)
    ax.indicate_inset_zoom(inset, edgecolor="#5f6368", alpha=0.65)


def _draw_trajectory(ax, item: LoadedTrajectory, evo_plot) -> None:
    colors = {"EGO2": "#147d92", "EGO4": "#d46a1f"}
    color = colors.get(item.source.device, "#3f6b45")
    positions = item.trajectory.positions_xyz

    evo_plot.traj(
        ax,
        evo_plot.PlotMode.xy,
        item.trajectory,
        color=color,
        label="Trajectory",
        alpha=0.95,
    )
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        s=52,
        marker="o",
        color="#17823b",
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="Start",
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        s=58,
        marker="X",
        color="#c83e3e",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="End",
    )
    ax.plot(
        [positions[0, 0], positions[-1, 0]],
        [positions[0, 1], positions[-1, 1]],
        linestyle=":",
        linewidth=1.2,
        color="#5f6368",
        alpha=0.8,
        zorder=3,
    )
    ax.set_title(f"{item.source.device} | {item.source.sequence}", fontsize=12)
    ax.text(
        0.02,
        0.98,
        (
            f"Origin error: {item.metrics.origin_error_m:.3f} m\n"
            f"Total distance: {item.metrics.total_distance_m:.3f} m\n"
            f"Error percentage: {item.metrics.error_percent:.3f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#aeb4b9",
            "alpha": 0.9,
        },
        zorder=10,
    )
    _draw_origin_inset(ax, item, evo_plot, color)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#d9dde0", linewidth=0.7, alpha=0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=True)


def generate_plots(input_dir: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    inputs = discover_trajectories(input_dir)
    if not inputs:
        raise ValueError(f"no final-BA trajectory CSV files found under {input_dir}")

    loaded = []
    for source in inputs:
        trajectory = load_okvis_trajectory(source.path)
        loaded.append(
            LoadedTrajectory(
                source,
                trajectory,
                calculate_metrics(trajectory.positions_xyz),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    evo_plot, plt = _load_plot_modules()
    generated = []

    for item in loaded:
        figure = plt.figure(figsize=(7.2, 6.2))
        axis = evo_plot.prepare_axis(figure, evo_plot.PlotMode.xy)
        _draw_trajectory(axis, item, evo_plot)
        path = output_dir / (
            f"{item.source.device}_{item.source.sequence}_trajectory.png"
        )
        figure.savefig(path, dpi=dpi, facecolor="white")
        plt.close(figure)
        generated.append(path)

    devices = sorted(
        {item.source.device for item in loaded},
        key=lambda value: int(value.removeprefix("EGO")),
    )
    columns = max(
        sum(item.source.device == device for item in loaded) for device in devices
    )
    rows = len(devices)
    overview = plt.figure(figsize=(6.0 * columns, 5.0 * rows))
    plot_index = 1
    for device in devices:
        device_items = [item for item in loaded if item.source.device == device]
        for item in device_items:
            subplot_arg = rows * 100 + columns * 10 + plot_index
            axis = evo_plot.prepare_axis(
                overview, evo_plot.PlotMode.xy, subplot_arg=subplot_arg
            )
            _draw_trajectory(axis, item, evo_plot)
            plot_index += 1
        for _ in range(columns - len(device_items)):
            axis = overview.add_subplot(rows, columns, plot_index)
            axis.set_axis_off()
            plot_index += 1
    overview.suptitle("Final BA Trajectories (evo, XY projection)", fontsize=17)
    combined_path = output_dir / "all_trajectories.png"
    overview.savefig(combined_path, dpi=dpi, facecolor="white")
    plt.close(overview)
    generated.append(combined_path)

    for item in loaded:
        print(
            f"{item.source.device} {item.source.sequence}: "
            f"origin_error={item.metrics.origin_error_m:.6f} m, "
            f"total_distance={item.metrics.total_distance_m:.6f} m, "
            f"error_percentage={item.metrics.error_percent:.6f}%"
        )
    return generated


def parse_arguments() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    default_input = repository_root / "workspace" / "final_ba_trajectory_files"
    parser = argparse.ArgumentParser(
        description="Plot OKVIS final-BA trajectories with evo and loop metrics."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    input_dir = arguments.input_dir.resolve()
    output_dir = (
        arguments.output_dir.resolve()
        if arguments.output_dir
        else input_dir / "trajectory_plots"
    )
    generated = generate_plots(input_dir, output_dir, arguments.dpi)
    print(f"Wrote {len(generated)} images to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
