#!/usr/bin/env python3
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ego5_outdoor_origin_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ego5_outdoor_origin_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
RESULT_ROOT = Path(
    os.environ.get(
        "EGO5_OUTDOOR_RESULTS_ROOT",
        str(REPOSITORY / "workspace/ego5_results/outdoor_0818_with_calib_0817"),
    )
)
OUTPUT_IMAGE = RESULT_ROOT / "ego5_outdoor_0818_origin_zoom_all.png"
OUTPUT_CSV = RESULT_ROOT / "ego5_outdoor_0818_origin_metrics.csv"
CSV_FIELDS = ("p_WS_W_x", "p_WS_W_y", "p_WS_W_z")


def configure_fonts() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_name = (
        font_manager.FontProperties(fname=font_path).get_name()
        if font_path.is_file()
        else "DejaVu Sans"
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def load_positions(path: Path) -> np.ndarray:
    positions = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, skipinitialspace=True)
        missing = [field for field in CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing CSV fields: {', '.join(missing)}")
        for line_number, row in enumerate(reader, 2):
            try:
                positions.append([float(row[field]) for field in CSV_FIELDS])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid position") from error
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise ValueError(f"{path}: fewer than two valid 3D positions")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite position")
    return values


def calculate_metrics(positions: np.ndarray) -> tuple[float, float, float]:
    total_distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    origin_error = float(np.linalg.norm(positions[-1] - positions[0]))
    error_percentage = 100.0 * origin_error / total_distance if total_distance else 0.0
    return origin_error, total_distance, error_percentage


def origin_zoom_limits(positions: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    start = positions[0, :2]
    end = positions[-1, :2]
    center = (start + end) / 2.0
    half_span = max(float(np.max(np.abs(end - start))) * 1.75, 0.025)
    return (
        (float(center[0] - half_span), float(center[0] + half_span)),
        (float(center[1] - half_span), float(center[1] + half_span)),
    )


def draw_start_end(axis, positions: np.ndarray, *, marker_scale: float = 1.0) -> None:
    axis.scatter(
        positions[0, 0],
        positions[0, 1],
        s=72 * marker_scale,
        marker="o",
        color="#17823b",
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
        label="起点（原点）",
    )
    axis.scatter(
        positions[-1, 0],
        positions[-1, 1],
        s=76 * marker_scale,
        marker="X",
        color="#c83e3e",
        edgecolors="white",
        linewidths=0.9,
        zorder=6,
        label="终点",
    )
    axis.plot(
        [positions[0, 0], positions[-1, 0]],
        [positions[0, 1], positions[-1, 1]],
        linestyle=":",
        linewidth=1.4,
        color="#5f6368",
        zorder=4,
    )


def draw_trajectory(axis, sequence: str, positions: np.ndarray) -> tuple[float, float, float]:
    positions = positions - positions[0]
    metrics = calculate_metrics(positions)
    origin_error, total_distance, error_percentage = metrics

    axis.plot(
        positions[:, 0],
        positions[:, 1],
        color="#216b80",
        linewidth=1.45,
        alpha=0.95,
        label="Final-BA 轨迹",
    )
    draw_start_end(axis, positions)
    axis.set_title(sequence, fontsize=14, fontweight="bold")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, color="#d9dde0", linewidth=0.7, alpha=0.8)
    axis.legend(loc="lower right", frameon=True, fontsize=8)
    axis.text(
        0.02,
        0.98,
        f"原点误差（3D）: {origin_error:.4f} m\n"
        f"总运动里程（3D）: {total_distance:.3f} m\n"
        f"原点误差百分比: {error_percentage:.4f}%",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#aeb4b9",
            "alpha": 0.93,
        },
        zorder=10,
    )

    inset = axis.inset_axes((0.04, 0.07, 0.34, 0.34))
    inset.plot(positions[:, 0], positions[:, 1], color="#216b80", linewidth=1.2)
    draw_start_end(inset, positions, marker_scale=0.72)
    x_limits, y_limits = origin_zoom_limits(positions)
    inset.set_xlim(x_limits)
    inset.set_ylim(y_limits)
    inset.set_aspect("equal", adjustable="box")
    inset.set_title("原点放大", fontsize=8, pad=3)
    inset.tick_params(axis="both", labelsize=6)
    inset.grid(True, color="#d9dde0", linewidth=0.55, alpha=0.9)
    inset.set_facecolor("#fbfcfd")
    for spine in inset.spines.values():
        spine.set_edgecolor("#5f6368")
        spine.set_linewidth(1.0)
    axis.indicate_inset_zoom(inset, edgecolor="#5f6368", alpha=0.6)
    return metrics


def main() -> int:
    configure_fonts()
    inputs = sorted(RESULT_ROOT.glob("*/run1/*final-ba_trajectory.csv"))
    if len(inputs) != 3:
        raise RuntimeError(f"expected three final-BA trajectories, found {len(inputs)}")

    figure, axes = plt.subplots(1, 3, figsize=(20.5, 7.0))
    rows = []
    for axis, path in zip(axes, inputs):
        sequence = path.parent.parent.name
        positions = load_positions(path)
        origin_error, total_distance, error_percentage = draw_trajectory(
            axis, sequence, positions
        )
        rows.append(
            (sequence, path, len(positions), origin_error, total_distance, error_percentage)
        )

    figure.suptitle(
        "EGO5 Outdoor 2026-08-18 Final-BA 轨迹与原点闭环误差",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "原点误差 = 终点与起点的三维 L2 距离；原点误差百分比 = 原点误差 / 三维轨迹总运动里程 × 100%。",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.015, 0.055, 0.985, 0.93), w_pad=2.2)
    figure.savefig(OUTPUT_IMAGE, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "sequence",
                "trajectory_csv",
                "poses",
                "origin_error_m_3d",
                "total_distance_m_3d",
                "origin_error_percentage",
            ]
        )
        writer.writerows(rows)

    for sequence, _, poses, origin_error, total_distance, percentage in rows:
        print(
            f"{sequence}: poses={poses}, origin_error={origin_error:.6f} m, "
            f"distance={total_distance:.6f} m, percentage={percentage:.6f}%"
        )
    print(f"Image: {OUTPUT_IMAGE}")
    print(f"Data: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
