#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMBER_TOKEN = rf"({NUMBER})(?![A-Za-z0-9_.+-])"
TIME_RE = re.compile(rf"\btime:\s*{NUMBER_TOKEN}")
TRACKED_RE = re.compile(r"\btracked:\s*([01])(?=$|\s|[,;])")
VALUE_SEPARATOR = r"(?:\s+|\s*,\s*)"
POSE_RE = re.compile(
    rf"\bPose:\s*rpy=\s*{NUMBER_TOKEN}{VALUE_SEPARATOR}{NUMBER_TOKEN}"
    rf"{VALUE_SEPARATOR}{NUMBER_TOKEN}\s+xyz=\s*{NUMBER_TOKEN}{VALUE_SEPARATOR}"
    rf"{NUMBER_TOKEN}{VALUE_SEPARATOR}{NUMBER_TOKEN}"
)
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
LOCAL_FINAL_BA_RE = re.compile(
    r"^okvis2-(?:vio|slam)(?:-calib)?-final-ba_trajectory\.csv$"
)


@dataclass(frozen=True)
class Pose:
    timestamp: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class PlotMetrics:
    gt_distance_m: float
    ape_rmse_mm: float
    error_percentage: float


class EvaluationError(RuntimeError):
    pass


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def parse_mocap(path: Path) -> list[Pose]:
    poses = []
    pending_time = None
    pending_tracked = None
    last_line_number = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            last_line_number = line_number
            time_match = TIME_RE.search(line)
            if "time:" in line and not time_match:
                raise EvaluationError(f"{path}:{line_number}: malformed time value")
            if time_match:
                if pending_time is not None:
                    raise EvaluationError(
                        f"{path}:{line_number}: incomplete mocap record"
                    )
                pending_time = (float(time_match.group(1)), line_number)

            tracked_match = TRACKED_RE.search(line)
            if "tracked:" in line and not tracked_match:
                raise EvaluationError(f"{path}:{line_number}: malformed tracked value")
            if tracked_match:
                if pending_time is None:
                    raise EvaluationError(
                        f"{path}:{line_number}: tracked state without time"
                    )
                if pending_tracked is not None:
                    raise EvaluationError(
                        f"{path}:{line_number}: repeated tracked state in mocap record"
                    )
                pending_tracked = tracked_match.group(1) == "1"

            if "Pose:" not in line:
                continue
            pose_match = POSE_RE.search(line)
            if pending_time is None or pending_tracked is None or not pose_match:
                raise EvaluationError(
                    f"{path}:{line_number}: malformed or incomplete mocap record"
                )
            values = [float(value) for value in pose_match.groups()]
            if pending_tracked:
                poses.append(
                    Pose(
                        pending_time[0],
                        tuple(values[3:6]),
                        rpy_to_quaternion(*values[:3]),
                    )
                )
            pending_time = None
            pending_tracked = None

    if pending_time is not None:
        raise EvaluationError(
            f"{path}:{last_line_number or pending_time[1]}: incomplete mocap record"
        )
    return poses


def parse_okvis_csv(path: Path) -> list[Pose]:
    poses = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, skipinitialspace=True)
        missing = [field for field in CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise EvaluationError(f"{path}:1: missing CSV fields: {', '.join(missing)}")
        for row_number, row in enumerate(reader, 2):
            try:
                values = [float(row[field]) for field in CSV_FIELDS]
            except (TypeError, ValueError) as error:
                raise EvaluationError(
                    f"{path}:{row_number}: invalid numeric value"
                ) from error
            poses.append(
                Pose(values[0] / 1e9, tuple(values[1:4]), tuple(values[4:8]))
            )
    return poses


def validate_poses(poses: list[Pose], label: str = "trajectory") -> None:
    if len(poses) < 2:
        raise EvaluationError(f"{label} has fewer than two valid poses")
    previous_timestamp = -math.inf
    for pose in poses:
        values = (pose.timestamp, *pose.position, *pose.quaternion)
        if not all(math.isfinite(value) for value in values):
            raise EvaluationError(f"{label} contains non-finite values")
        if pose.timestamp <= previous_timestamp:
            raise EvaluationError(f"{label} timestamps are not strictly increasing")
        previous_timestamp = pose.timestamp


def write_tum(path: Path, poses: list[Pose]) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("# timestamp tx ty tz qx qy qz qw\n")
        for pose in poses:
            values = (pose.timestamp, *pose.position, *pose.quaternion)
            output.write(" ".join(str(value) for value in values) + "\n")


def compute_plot_metrics(reference_positions, estimate_positions) -> PlotMetrics:
    try:
        import numpy as np
    except ImportError as error:
        raise EvaluationError("plotting requires numpy") from error

    reference = np.asarray(reference_positions, dtype=float)
    estimate = np.asarray(estimate_positions, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1:] != (3,)
        or estimate.shape != reference.shape
        or len(reference) < 2
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(estimate))
    ):
        raise EvaluationError(
            "aligned GT and SLAM positions must be matching finite Nx3 arrays"
        )

    gt_distance_m = float(
        np.sum(np.linalg.norm(np.diff(reference, axis=0), axis=1))
    )
    if gt_distance_m <= 0.0:
        raise EvaluationError("associated GT trajectory has zero movement distance")
    ape_rmse_m = float(
        np.sqrt(np.mean(np.sum((estimate - reference) ** 2, axis=1)))
    )
    return PlotMetrics(
        gt_distance_m=gt_distance_m,
        ape_rmse_mm=ape_rmse_m * 1000.0,
        error_percentage=100.0 * ape_rmse_m / gt_distance_m,
    )


def format_metrics_annotation(metrics: PlotMetrics) -> str:
    return (
        f"总运动里程: {metrics.gt_distance_m:.3f} m\n"
        f"APE RMSE: {metrics.ape_rmse_mm:.3f} mm\n"
        f"误差百分比: {metrics.error_percentage:.4f}%"
    )


def align_associated_positions(
    reference: list[Pose], estimate: list[Pose], max_diff: float
):
    try:
        import numpy as np
        from evo.core import sync
        from evo.core.trajectory import PoseTrajectory3D
    except ImportError as error:
        raise EvaluationError(
            "plotting requires evo and numpy; run in the okvis2x conda environment"
        ) from error

    def to_evo(poses: list[Pose]) -> PoseTrajectory3D:
        return PoseTrajectory3D(
            positions_xyz=np.asarray([pose.position for pose in poses], dtype=float),
            orientations_quat_wxyz=np.asarray(
                [
                    (
                        pose.quaternion[3],
                        pose.quaternion[0],
                        pose.quaternion[1],
                        pose.quaternion[2],
                    )
                    for pose in poses
                ],
                dtype=float,
            ),
            timestamps=np.asarray([pose.timestamp for pose in poses], dtype=float),
        )

    reference_evo, estimate_evo = sync.associate_trajectories(
        to_evo(reference),
        to_evo(estimate),
        max_diff=max_diff,
        first_name="mocap",
        snd_name="SLAM",
    )
    estimate_evo.align(reference_evo, correct_scale=False)
    return (
        reference_evo.positions_xyz.copy(),
        estimate_evo.positions_xyz.copy(),
    )


def save_trajectory_plot(
    output: Path, reference_positions, estimate_positions, metrics: PlotMetrics
) -> None:
    cache_root = Path(tempfile.gettempdir()) / "okvis2x-plot-cache"
    matplotlib_cache = cache_root / "matplotlib"
    font_cache = cache_root / "fontconfig"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise EvaluationError("plotting requires matplotlib and numpy") from error

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = np.asarray(reference_positions, dtype=float)
    estimate = np.asarray(estimate_positions, dtype=float)

    font_name = "DejaVu Sans"
    for candidate in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ):
        try:
            font_path = font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        break

    with plt.rc_context(
        {"font.family": "sans-serif", "font.sans-serif": [font_name],
         "axes.unicode_minus": False}
    ):
        figure = plt.figure(figsize=(9.0, 8.0), constrained_layout=True)
        grid = figure.add_gridspec(2, 1, height_ratios=(0.20, 0.80))
        annotation_axis = figure.add_subplot(grid[0])
        trajectory_axis = figure.add_subplot(grid[1])

        annotation_axis.axis("off")
        annotation_axis.set_title("Mocap GT 与 SLAM 估计轨迹", fontsize=16)
        annotation_axis.text(
            0.0,
            0.48,
            format_metrics_annotation(metrics),
            ha="left",
            va="center",
            fontsize=11,
            linespacing=1.5,
        )

        trajectory_axis.plot(
            reference[:, 0], reference[:, 1],
            color="#202124", linewidth=2.2, label="GT mocap 轨迹",
        )
        trajectory_axis.plot(
            estimate[:, 0], estimate[:, 1],
            color="#D1495B", linewidth=1.6, label="SLAM 估计轨迹",
        )
        trajectory_axis.scatter(
            reference[0, 0], reference[0, 1],
            marker="o", s=42, color="#2A9D8F", label="起点", zorder=4,
        )
        trajectory_axis.scatter(
            reference[-1, 0], reference[-1, 1],
            marker="s", s=42, color="#E9C46A", edgecolor="#202124",
            linewidth=0.6, label="终点", zorder=4,
        )
        trajectory_axis.set_xlabel("X (m)")
        trajectory_axis.set_ylabel("Y (m)")
        trajectory_axis.set_aspect("equal", adjustable="datalim")
        trajectory_axis.grid(True, color="#DADCE0", linewidth=0.8, alpha=0.8)
        trajectory_axis.legend(loc="best", frameon=False)

        figure.savefig(output, dpi=180, format="png")
        plt.close(figure)


def resolve_estimate(value: Path) -> Path:
    value = value.expanduser()
    if value.is_file():
        return value
    if not value.is_dir():
        raise EvaluationError(f"estimate path does not exist: {value}")

    matches = sorted(
        item
        for item in value.iterdir()
        if item.is_file() and LOCAL_FINAL_BA_RE.fullmatch(item.name)
    )
    if not matches:
        raise EvaluationError(f"no local final-BA trajectory found in {value}")
    if len(matches) > 1:
        raise EvaluationError(
            "multiple local final-BA trajectories found: "
            + ", ".join(str(item) for item in matches)
        )
    return matches[0]


def validate_overlap(reference: list[Pose], estimate: list[Pose]) -> None:
    start = max(reference[0].timestamp, estimate[0].timestamp)
    end = min(reference[-1].timestamp, estimate[-1].timestamp)
    if start >= end:
        raise EvaluationError("mocap and OKVIS trajectories do not overlap in time")


def build_evo_command(
    executable: str,
    reference_tum: Path,
    estimate_tum: Path,
    max_diff: float,
    save_results: Path | None,
) -> list[str]:
    command = [
        executable,
        "tum",
        str(reference_tum),
        str(estimate_tum),
        "--align",
        "--t_max_diff",
        str(max_diff),
    ]
    if save_results is not None:
        command.extend(["--save_results", str(save_results)])
    return command


def find_evo_ape() -> str:
    executable = shutil.which("evo_ape")
    if executable is None:
        raise EvaluationError(
            "evo_ape is not installed in the current environment; for okvis2x run: "
            "conda run -n okvis2x python -m pip install evo"
        )
    return executable


def run_evo(
    reference_tum: Path,
    estimate_tum: Path,
    max_diff: float,
    save_results: Path | None,
) -> int:
    command = build_evo_command(
        find_evo_ape(), reference_tum, estimate_tum, max_diff, save_results
    )
    return subprocess.run(command, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OKVIS final-BA trajectory against mocap with evo APE "
            "and rigid SE(3) alignment."
        )
    )
    parser.add_argument("mocap_log", type=Path, help="OptiTrack mocap.log path")
    parser.add_argument(
        "estimate",
        type=Path,
        help="OKVIS final-BA CSV path or result directory",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        default=0.01,
        help="maximum timestamp association difference in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--tum-dir",
        type=Path,
        help="keep converted mocap.tum and okvis.tum in this directory",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        help="save evo results to this ZIP file",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help=(
            "save an XY trajectory PNG with aligned mocap GT, SLAM estimate, "
            "GT distance, APE RMSE and normalized error percentage"
        ),
    )
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> int:
    if not math.isfinite(args.max_diff) or args.max_diff <= 0.0:
        raise EvaluationError("--max-diff must be a finite positive number")

    mocap_path = args.mocap_log.expanduser()
    estimate_path = resolve_estimate(args.estimate)
    reference = parse_mocap(mocap_path)
    estimate = parse_okvis_csv(estimate_path)
    validate_poses(reference, "mocap trajectory")
    validate_poses(estimate, "OKVIS trajectory")
    validate_overlap(reference, estimate)

    print(f"Estimate: {estimate_path}")
    print(f"Valid poses: mocap={len(reference)}, okvis={len(estimate)}")

    def evaluate_in(directory: Path) -> int:
        reference_tum = directory / "mocap.tum"
        estimate_tum = directory / "okvis.tum"
        write_tum(reference_tum, reference)
        write_tum(estimate_tum, estimate)
        return_code = run_evo(
            reference_tum, estimate_tum, args.max_diff, args.save_results
        )
        if return_code != 0 or args.plot is None:
            return return_code

        reference_positions, estimate_positions = align_associated_positions(
            reference, estimate, args.max_diff
        )
        metrics = compute_plot_metrics(reference_positions, estimate_positions)
        plot_path = args.plot.expanduser()
        save_trajectory_plot(
            plot_path, reference_positions, estimate_positions, metrics
        )
        print(f"GT distance: {metrics.gt_distance_m:.3f} m")
        print(f"APE RMSE: {metrics.ape_rmse_mm:.3f} mm")
        print(f"Error percentage: {metrics.error_percentage:.4f}%")
        print(f"Trajectory plot: {plot_path}")
        return 0

    if args.tum_dir is not None:
        tum_dir = args.tum_dir.expanduser()
        tum_dir.mkdir(parents=True, exist_ok=True)
        return evaluate_in(tum_dir)
    with tempfile.TemporaryDirectory(prefix="okvis-ape-") as directory:
        return evaluate_in(Path(directory))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return evaluate(args)
    except (EvaluationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
