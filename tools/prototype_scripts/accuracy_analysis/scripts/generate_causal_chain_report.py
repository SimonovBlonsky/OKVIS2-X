#!/usr/bin/env python3
"""Generate the baseline-only VIO causal-chain report and its figures."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-causal-chain-report")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-causal-chain-report")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

try:
    from . import refresh_population_causal_outputs as population_refresh
except ImportError:
    import refresh_population_causal_outputs as population_refresh


IMPULSE_SEQUENCES = (
    "20260806-175103",
    "20260806-175304",
    "20260806-175539",
)
SEQUENCE_COLOURS = {
    "20260806-175103": "#16843b",
    "20260806-175304": "#b3261e",
    "20260806-175539": "#7357a5",
}
TABLE_SOURCES = {
    "events": "causal_event_metrics.csv",
    "frames": "causal_frame_metrics.csv",
    "recovery": "impulse_mediator_recovery.csv",
    "runs": "cross_sample_run_metrics.csv",
    "sequences": "cross_sample_sequence_metrics.csv",
    "evidence": "cross_sample_evidence.csv",
    "sim3": "observability_sim3_by_run.csv",
    "failure_windows": "observability_failure_windows.csv",
    "stages": "20260803-184537_failure_chain_stages.csv",
    "population_chain": "population_angular_failure_chain.csv",
    "paired_effects": "population_angular_event_paired_effects.csv",
}

# These four reason-specific counts were already extracted from the baseline
# structured lifecycle logs. Reading the multi-gigabyte event files again would
# make a report-only command unnecessarily expensive.
REMOVAL_COUNTS = {
    ("20260806-175304", "run1"): (2734, 14464),
    ("20260806-175304", "run2"): (2805, 14374),
    ("20260806-175539", "run1"): (1638, 17498),
    ("20260806-175539", "run2"): (1824, 16685),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-root", required=True, type=Path,
        help="Baseline structured-diagnostics root.",
    )
    parser.add_argument(
        "--tables-root", required=True, type=Path,
        help="Existing derived evidence tables.",
    )
    parser.add_argument(
        "--image-delay-root", required=True, type=Path,
        help="Completed image-delay intervention experiment root.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required table: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_impulse_frame_rows(path: Path) -> list[dict[str, str]]:
    """Read only the three anti-impact sequences from the 181 MB frame table."""
    if not path.is_file():
        raise FileNotFoundError(f"missing required table: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            row for row in csv.DictReader(stream)
            if row.get("sequence") in IMPULSE_SEQUENCES
        ]


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return math.nan
    return float(value)


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required experiment file: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_ape_rmse_mm(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"missing required APE result: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "rmse":
            return float(fields[1]) * 1000.0
    raise ValueError(f"missing rmse in {path}")


def parse_archived_184537_rows(readme: Path) -> list[dict[str, object]]:
    if not readme.is_file():
        raise FileNotFoundError(f"missing image-delay README: {readme}")
    parsed: list[dict[str, object]] = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 4 or not fields[1].endswith(" mm"):
            continue
        label, ape_text, ransac_text, large_text = fields
        if not ape_text[:-3].replace(".", "", 1).isdigit():
            continue
        if label.startswith("24.869740 ms 基线 run"):
            run = label.rsplit(" ", 1)[-1]
            configuration = "baseline_24p869740ms"
            delay_ms = 24.869740
            extrinsics = "online"
            evaluation_control = "false"
        elif label.startswith("40 ms run"):
            run = label.rsplit(" ", 1)[-1]
            configuration = "delay_40ms"
            delay_ms = 40.0
            extrinsics = "online"
            evaluation_control = "false"
        elif "固定外参" in label:
            run = "run1"
            configuration = "baseline_delay_fixed_extrinsics"
            delay_ms = 24.869740
            extrinsics = "fixed"
            evaluation_control = "false"
        elif "只平移输出时间戳" in label:
            run = "run1"
            configuration = "timestamp_shift_only"
            delay_ms = 24.869740
            extrinsics = "online"
            evaluation_control = "true"
        else:
            continue
        parsed.append(
            {
                "sequence": "20260803-184537",
                "run": run,
                "configuration": configuration,
                "delay_ms": delay_ms,
                "extrinsics": extrinsics,
                "evaluation_control": evaluation_control,
                "ape_rmse_mm": float(ape_text[:-3]),
                "ransac_fail_count": "" if ransac_text == "N/A" else int(ransac_text),
                "large_reprojection_count": "" if large_text == "N/A" else int(large_text),
                "source": "202608_image_delay_experiments/README.md",
            }
        )
    if len(parsed) != 6:
        raise ValueError(f"expected 6 archived 184537 rows in {readme}, found {len(parsed)}")
    return parsed


def load_image_delay_interventions(
    image_delay_root: Path,
    run_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    baseline_lookup = {
        (row["sequence"], row["run"]): row for row in run_rows
        if row["sequence"] in {"20260803-183537", "20260803-184027"}
    }
    output: list[dict[str, object]] = []
    for sequence in ("20260803-183537", "20260803-184027"):
        for run in ("run1", "run2"):
            baseline = baseline_lookup[(sequence, run)]
            output.append(
                {
                    "sequence": sequence,
                    "run": run,
                    "configuration": "baseline_24p869740ms",
                    "delay_ms": 24.869740,
                    "extrinsics": "online",
                    "evaluation_control": "false",
                    "ape_rmse_mm": number(baseline, "corrected_ape_rmse_mm"),
                    "ransac_fail_count": int(number(baseline, "ransac_fail_count")),
                    "large_reprojection_count": int(number(baseline, "ransac_large_reprojection_count")),
                    "source": "cross_sample_run_metrics.csv",
                }
            )
            run_directory = image_delay_root / sequence / "delay_39p25ms" / run
            manifest = parse_key_value_file(run_directory / "run_manifest.txt")
            if manifest.get("return_code") != "0":
                raise ValueError(f"incomplete image-delay run: {run_directory}")
            if not math.isclose(float(manifest["image_delay_s"]), 0.03925, abs_tol=1e-12):
                raise ValueError(f"unexpected image_delay in {run_directory}")
            failures = parse_key_value_file(run_directory / "frontend_failure_counts.txt")
            output.append(
                {
                    "sequence": sequence,
                    "run": run,
                    "configuration": "delay_39p25ms",
                    "delay_ms": 39.25,
                    "extrinsics": "online",
                    "evaluation_control": "false",
                    "ape_rmse_mm": parse_ape_rmse_mm(run_directory / "ape.txt"),
                    "ransac_fail_count": int(failures["ransac_fail_count"]),
                    "large_reprojection_count": int(failures["large_reprojection_count"]),
                    "source": str((run_directory / "ape.txt").relative_to(image_delay_root.parent)),
                }
            )
    output.extend(parse_archived_184537_rows(image_delay_root / "README.md"))
    return output


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def configure_plot_style() -> None:
    noto_directory = Path("/usr/share/fonts/opentype/noto")
    noto_regular = noto_directory / "NotoSansCJK-Regular.ttc"
    if noto_regular.is_file():
        for font_path in noto_directory.glob("NotoSansCJK-*.ttc"):
            font_manager.fontManager.addfont(str(font_path))
        cjk_family = font_manager.FontProperties(fname=str(noto_regular)).get_name()
    else:
        cjk_family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cjk_family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfa",
            "axes.edgecolor": "#555555",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def panel_note(
    axis: plt.Axes,
    text: str,
    *,
    width: int | None = None,
    outside: bool = False,
) -> None:
    if width is not None:
        text = "\n".join(
            textwrap.fill(line, width=width) for line in text.splitlines()
        )
    axis.text(
        0.01,
        1.02 if outside else 0.99,
        text,
        transform=axis.transAxes,
        va="bottom" if outside else "top",
        ha="left",
        fontsize=8.3,
        linespacing=1.35,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#b8b8b8"},
        zorder=20,
    )


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def strongest_events(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["sequence"], row["run"])
        if key not in selected or number(row, "angular_integral") > number(
            selected[key], "angular_integral"
        ):
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def percent_change(row: dict[str, str], metric: str) -> float:
    baseline = number(row, f"{metric}_baseline")
    during = number(row, f"{metric}_mediator")
    return (during / baseline - 1.0) * 100.0


def plot_time_error_amplification(output: Path) -> None:
    residual_delay_s = 0.015130260
    focal_length_px = 311.0
    angular_speed = np.linspace(0.0, 5.0, 300)
    rotation_error_deg = np.degrees(angular_speed * residual_delay_s)
    pixel_error = focal_length_px * angular_speed * residual_delay_s

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    axis = axes[0]
    axis.plot(angular_speed, rotation_error_deg, color="#27647b", linewidth=2.5)
    local_rate = 4.08
    theoretical_deg = math.degrees(local_rate * residual_delay_s)
    axis.scatter(
        [local_rate], [theoretical_deg], s=80, marker="D", color="#d18b00",
        label=f"理论姿态差 {theoretical_deg:.2f}°",
    )
    axis.scatter(
        [local_rate, local_rate], [3.36, 3.43], s=75, marker="x",
        linewidth=2.5, color="#b3261e", label="两次 GP3P 视觉修正 3.36°/3.43°",
    )
    axis.set_xlabel("角速度 |ω| (rad/s)")
    axis.set_ylabel("15.13 ms 残余时差对应的姿态误差 (deg)")
    axis.set_xlim(0, 5)
    axis.set_ylim(0, 4.7)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8.5)
    panel_note(
        axis,
        "x：图像时刻附近的角速度；y：|ω|×|δt| 预测的姿态相位误差。\n"
        "规律：角速度越高，同一时间误差造成的姿态错位越大；首次密集失败处理论量与视觉修正量闭合。",
    )

    axis = axes[1]
    axis.plot(angular_speed, pixel_error, color="#27647b", linewidth=2.5)
    thresholds = (
        (4.0, "优化后 observation 删除阈值 4 px", "#b3261e"),
        (4.87, "GP3P 触发阈值约 4.87 px", "#d18b00"),
        (21.7, "地图匹配搜索范围约 21.7 px", "#16843b"),
    )
    for value, label, colour in thresholds:
        axis.axhline(value, color=colour, linestyle="--", linewidth=1.6, label=label)
    for label, rate in (("p95", 1.895), ("p99", 2.629), ("max", 4.476)):
        value = focal_length_px * rate * residual_delay_s
        axis.scatter(rate, value, marker="o", s=55, color="#27647b")
        axis.annotate(f"{label}: {value:.1f} px", (rate, value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("角速度 |ω| (rad/s)")
    axis.set_ylabel("近光轴预测投影偏移 f|ω||δt| (px)")
    axis.set_xlim(0, 5)
    axis.set_ylim(0, 25)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8.2)
    panel_note(
        axis,
        "x：角速度；y：由残余时差换算的近光轴像素偏移。\n"
        "规律：候选仍可能落在 21.7 px 搜索区内，但几何误差会先跨过 4.87 px GP3P 和 4 px 删除门限。",
    )
    figure.suptitle("时间误差被高角速度放大：从姿态相位差到 OKVIS2 离散门限", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output)


def plot_impulse_matching_degradation(
    rows: Sequence[dict[str, str]], output: Path
) -> None:
    selected = strongest_events(
        [row for row in rows if row.get("sequence") in IMPULSE_SEQUENCES]
    )
    metrics = (
        ("keypoints_total", "特征点总数", "特征数接近不变。"),
        ("image_laplacian_variance", "Laplacian 清晰度代理", "冲击期清晰度下降。"),
        ("best_descriptor_distance_median", "最佳地图 descriptor 距离", "descriptor 距离明显增大。"),
        ("accepted_map_matches", "接受的既有地图匹配", "既有地图匹配减少约四分之一。"),
        ("predicted_reprojection_error_px_median", "预测重投影误差中位数", "几何误差增幅远大于特征数。"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    x = np.arange(len(selected))
    labels = [f"{row['sequence'][-6:]}\n{row['run']}" for row in selected]
    colours = [SEQUENCE_COLOURS[row["sequence"]] for row in selected]
    for axis, (metric, title, rule) in zip(axes.flat, metrics):
        values = [percent_change(row, metric) for row in selected]
        axis.bar(x, values, color=colours, alpha=0.88)
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel("事件前基线到冲击窗口的变化 (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        panel_note(
            axis,
            f"x：序列与重放；y：冲击期相对事件前的变化。\n规律：{rule}",
            width=21,
        )
    axis = axes.flat[-1]
    axis.axis("off")
    axis.text(
        0.02,
        0.96,
        "颜色说明\n"
        "绿色：175103（冲击后恢复）\n"
        "红色：175304（未恢复并尺度崩溃）\n"
        "紫色：175539（中间状态）\n\n"
        "合并判断\n"
        "特征点仅 -2.1% 至 +0.3%，但清晰度下降 26%-51%，descriptor 距离恶化 93%-198%，"
        "地图匹配下降 23%-26%，预测重投影误差上升 173%-443%。\n\n"
        "这支持‘特征仍在，但既有 3D 地图对应不再满足外观与几何一致性’，而不是单纯特征饥饿。",
        transform=axis.transAxes,
        va="top",
        fontsize=11,
        linespacing=1.6,
        bbox={"facecolor": "#f4f4f2", "edgecolor": "#aaaaaa", "boxstyle": "round,pad=0.6"},
    )
    figure.suptitle("高角速度冲击首先破坏地图匹配一致性，而非特征数量", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, output)


def plot_population_angular_failure_chain(
    run_rows: Sequence[dict[str, str]],
    paired_rows: Sequence[dict[str, str]],
    output: Path,
) -> None:
    converted_runs = [
        {field: population_refresh._coerce(value) for field, value in row.items()}
        for row in run_rows
    ]
    converted_effects = [
        {field: population_refresh._coerce(value) for field, value in row.items()}
        for row in paired_rows
    ]
    population_refresh.plot_population_chain(
        converted_runs, converted_effects, output
    )


def plot_gp3p_fragmentation_chain(
    run_rows: Sequence[dict[str, str]], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    axis = axes[0]
    x_values = [number(row, "ransac_fail_count") for row in run_rows]
    y_values = [number(row, "uninitialised_landmark_ransac_count") for row in run_rows]
    ape = [number(row, "corrected_ape_rmse_mm") for row in run_rows]
    colours = ["#b3261e" if value > 10.0 else "#16843b" for value in ape]
    axis.scatter(x_values, y_values, c=colours, alpha=0.8, edgecolors="white", linewidths=0.5)
    maximum = max(x_values + y_values) * 1.05
    axis.plot([0, maximum], [0, maximum], linestyle="--", color="#555555", label="retry 次数 = primary FAIL 次数")
    axis.set_xlabel("primary GP3P RANSAC FAIL 次数")
    axis.set_ylabel("启用未初始化 landmark 的 retry 次数")
    axis.set_xlim(-2, maximum)
    axis.set_ylim(-2, maximum)
    axis.set_xscale("symlog", linthresh=10, linscale=1.0)
    axis.set_yscale("symlog", linthresh=10, linscale=1.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8.5)
    panel_note(
        axis,
        "x：每次重放中已初始化地图的 primary FAIL；y：随后启用未初始化/新生 landmark 的 RANSAC。\n"
        "规律：点接近 y=x，说明 retry 是 primary FAIL 的直接代码分支，而非独立随机事件。",
    )

    axis = axes[1]
    labels = [f"{sequence[-6:]}\n{run}" for sequence, run in REMOVAL_COUNTS]
    gp3p = [counts[0] for counts in REMOVAL_COUNTS.values()]
    post = [counts[1] for counts in REMOVAL_COUNTS.values()]
    positions = np.arange(len(labels))
    width = 0.36
    axis.bar(positions - width / 2, gp3p, width, label="GP3P outlier 删除", color="#d18b00")
    axis.bar(positions + width / 2, post, width, label="优化后 >4 px 删除", color="#b3261e")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("按原因记录的 observation 删除次数")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25, which="both")
    axis.legend(loc="lower right", fontsize=8.5)
    panel_note(
        axis,
        "x：两条明显退化序列的两次重放；y：结构化日志中的 observation 删除次数（对数轴）。\n"
        "规律：优化后 >4 px 删除量远高于 GP3P 当场剔除，说明视觉约束流失主要在后续优化检查中扩大。",
    )
    figure.suptitle("primary GP3P 失败后进入 retry 与 observation 删除级联", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output)


def binned_series(
    rows: Iterable[dict[str, str]], field: str, bin_width: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    bins: defaultdict[int, list[float]] = defaultdict(list)
    for row in rows:
        if not finite(row.get(field)) or not finite(row.get("time_from_event_start_s")):
            continue
        time_s = number(row, "time_from_event_start_s")
        index = math.floor(time_s / bin_width)
        bins[index].append(number(row, field))
    indices = sorted(bins)
    return (
        np.asarray([(index + 0.5) * bin_width for index in indices]),
        np.asarray([statistics.median(bins[index]) for index in indices]),
    )


def plot_recovery_contrast(
    event_rows: Sequence[dict[str, str]],
    frame_rows: Sequence[dict[str, str]],
    sequence_rows: Sequence[dict[str, str]],
    output: Path,
) -> None:
    chosen = {(row["sequence"], row["run"]): row for row in strongest_events(event_rows)}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for sequence in IMPULSE_SEQUENCES:
        selected_frames = []
        for row in frame_rows:
            if row["sequence"] != sequence:
                continue
            key = (row["sequence"], row["run"])
            event = chosen.get(key)
            if event is None or row.get("event_id") != f"event{int(float(event['event_index'])):03d}":
                continue
            selected_frames.append(row)
        times, support = binned_series(selected_frames, "active_initialised_landmarks")
        pre = support[(times >= -4.5) & (times < 0)]
        baseline = statistics.median(pre) if len(pre) else 1.0
        axes[0].plot(times, support / baseline * 100.0, color=SEQUENCE_COLOURS[sequence], linewidth=2.2, label=sequence[-6:])
        times, removals = binned_series(selected_frames, "visual_observation_removals")
        axes[1].plot(times, removals, color=SEQUENCE_COLOURS[sequence], linewidth=2.2, label=sequence[-6:])
    for axis in axes[:2]:
        axis.axvspan(0.0, 0.9, color="#d18b00", alpha=0.13, label="角速度冲击")
        axis.axvline(0.0, color="#555555", linewidth=0.8)
        axis.set_xlim(-5, 10)
        axis.grid(alpha=0.25)
    axes[0].axhline(100, color="#777777", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("相对冲击开始时间 (s)")
    axes[0].set_ylabel("活跃已初始化 landmark（事件前=100%）")
    axes[0].legend(loc="lower right", fontsize=8.5)
    panel_note(
        axes[0],
        "x：相对最强冲击的时间；y：既有已初始化地图支撑相对事件前水平。\n"
        "规律：恢复能力不同；175103 能重新建立支撑，175304 长期处于弱旧地图状态。",
    )
    axes[1].set_xlabel("相对冲击开始时间 (s)")
    axes[1].set_ylabel("每帧优化后 observation 删除数（0.25 s 中位数）")
    panel_note(
        axes[1],
        "x：相对冲击时间；y：每帧删除的视觉观测。\n"
        "规律：删除峰值若不能快速回落，会持续侵蚀 landmark 观测和旧地图支撑。",
    )

    sequence_by_name = {row["sequence"]: row for row in sequence_rows}
    ape_values = [number(sequence_by_name[sequence], "corrected_ape_rmse_mm") for sequence in IMPULSE_SEQUENCES]
    axes[2].bar(
        np.arange(3), ape_values,
        color=[SEQUENCE_COLOURS[sequence] for sequence in IMPULSE_SEQUENCES],
    )
    axes[2].set_xticks(np.arange(3), [sequence[-6:] for sequence in IMPULSE_SEQUENCES])
    axes[2].set_yscale("log")
    axes[2].set_ylim(min(ape_values) * 0.65, max(ape_values) * 2.8)
    axes[2].set_xlabel("抗冲击序列")
    axes[2].set_ylabel("final-BA APE RMSE (mm，对数轴)")
    axes[2].grid(axis="y", alpha=0.25, which="both")
    for index, value in enumerate(ape_values):
        if value == max(ape_values):
            axes[2].text(
                index, value * 0.72, f"{value:.1f} mm", ha="center",
                fontsize=8.5, color="white",
            )
        else:
            axes[2].text(index, value * 1.15, f"{value:.1f} mm", ha="center", fontsize=8.5)
    panel_note(
        axes[2],
        "x：三条冲击序列；y：两次重放汇总 APE。\n"
        "规律：峰值角速度不是充分条件；旧地图支撑能否恢复决定误差是回落、残留还是尺度崩溃。",
    )
    figure.suptitle("相同类型冲击后的恢复分叉：175103 恢复，175304 未恢复，175539 居中", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output)


def evidence_lookup(rows: Sequence[dict[str, str]], factor: str) -> dict[str, str]:
    for row in rows:
        if row.get("factor") == factor:
            return row
    raise KeyError(f"missing evidence factor: {factor}")


def plot_fragmentation_and_loop_response(
    sequence_rows: Sequence[dict[str, str]],
    evidence_rows: Sequence[dict[str, str]],
    output: Path,
) -> None:
    panels = (
        ("ransac_fail_per_min", "GP3P RANSAC FAIL 次数/min", "ransac_fail_rate", "越频繁失败，APE 越高。"),
        ("loop_attempts_per_min", "回环尝试次数/min", "loop_attempt_rate", "回环激增与高 APE 同现，是碎片化后的响应。"),
        ("landmark_time_span_median_s", "landmark 中位存活时间 (s)", "landmark_span", "寿命越短，APE 越高。"),
        ("observations_per_landmark", "每个 landmark 的观测数", "observations_per_landmark", "长期观测支撑越少，APE 倾向越高。"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, (metric, xlabel, factor, rule) in zip(axes.flat, panels):
        evidence = evidence_lookup(evidence_rows, factor)
        rho = number(evidence, "all_spearman_rho")
        for row in sequence_rows:
            sequence = row["sequence"]
            impulse = sequence in IMPULSE_SEQUENCES
            value = number(row, metric)
            ape = number(row, "corrected_ape_rmse_mm")
            axis.scatter(
                value, ape, marker="^" if impulse else "o", s=62 if impulse else 42,
                color=SEQUENCE_COLOURS.get(sequence, "#27647b"), alpha=0.82,
                edgecolors="white", linewidths=0.45,
            )
            if sequence in {"20260803-184537", "20260806-175304", "20260806-175539"}:
                axis.annotate(sequence[-6:], (value, ape), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("APE RMSE (mm，对数轴)")
        axis.set_yscale("log")
        if metric == "ransac_fail_per_min":
            axis.set_xscale("symlog", linthresh=5, linscale=1.4)
            axis.set_xlim(0, max(number(row, metric) for row in sequence_rows) * 1.12)
        elif metric == "loop_attempts_per_min":
            axis.set_xscale("symlog", linthresh=10, linscale=1.4)
            axis.set_xlim(0, max(number(row, metric) for row in sequence_rows) * 1.12)
        axis.grid(alpha=0.24, which="both")
        direction = "同向" if rho > 0 else "反向"
        panel_note(
            axis,
            f"x：{xlabel}；y：跨序列 APE RMSE。\n规律：{rule}\n"
            f"Spearman 相关系数={rho:+.3f}（{direction}单调关系；绝对值越接近 1，关系越强）。",
            width=68,
            outside=True,
        )
    figure.text(
        0.5,
        0.015,
        "视觉碎片化定义：已初始化地图 GP3P 频繁失败 + 反复依赖新生/未初始化 landmark + observation 大量删除 + "
        "landmark 高频 birth/death、寿命短 + 活跃旧地图支撑下降。回环尝试激增是响应量，不是独立根因。",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "#f4f4f2", "edgecolor": "#aaaaaa", "boxstyle": "round,pad=0.5"},
    )
    figure.suptitle("24 个序列的视觉碎片化状态、回环响应与 APE", fontsize=15)
    figure.tight_layout(rect=(0, 0.055, 1, 0.91), h_pad=6.0)
    save_figure(figure, output)


def plot_observability_and_scale(
    sequence_rows: Sequence[dict[str, str]],
    run_rows: Sequence[dict[str, str]],
    evidence_rows: Sequence[dict[str, str]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    evidence = evidence_lookup(evidence_rows, "high_rotation_low_translation")
    rho = number(evidence, "all_spearman_rho")
    axis = axes[0]
    for row in sequence_rows:
        sequence = row["sequence"]
        x_value = number(row, "frac_rotation_gt_0p25_baseline_lt_5cm_pct")
        y_value = number(row, "corrected_ape_rmse_mm")
        axis.scatter(
            x_value, y_value, marker="^" if sequence in IMPULSE_SEQUENCES else "o",
            s=62 if sequence in IMPULSE_SEQUENCES else 42,
            color=SEQUENCE_COLOURS.get(sequence, "#27647b"), alpha=0.82,
            edgecolors="white", linewidths=0.45,
        )
        if sequence in {"20260803-184537", "20260806-175304", "20260806-175539"}:
            axis.annotate(sequence[-6:], (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("0.25 s 内旋转较大且平移 baseline <5 cm 的时间比例 (%)")
    axis.set_ylabel("APE RMSE (mm，对数轴)")
    axis.set_yscale("log")
    axis.grid(alpha=0.24, which="both")
    panel_note(
        axis,
        "x：高旋转/低平移窗口比例，是弱 temporal parallax 的运动代理；y：APE。\n"
        f"规律：相关系数={rho:+.3f}，与放大作用相容；但代理会受错误位姿制造的假视差影响，不能证明它是首次触发器。",
    )

    axis = axes[1]
    all_values = []
    for row in run_rows:
        se3 = number(row, "se3_rmse_mm")
        sim3 = number(row, "sim3_rmse_mm")
        if not (math.isfinite(se3) and math.isfinite(sim3) and se3 > 0 and sim3 > 0):
            continue
        all_values.extend((se3, sim3))
        sequence = row["sequence"]
        improvement = number(row, "sim3_improvement_pct")
        colour = "#b3261e" if improvement > 90 else SEQUENCE_COLOURS.get(sequence, "#27647b")
        axis.scatter(se3, sim3, color=colour, alpha=0.82, s=48, edgecolors="white", linewidths=0.45)
        if sequence == "20260806-175304":
            offset = (6, 7) if row["run"] == "run1" else (-82, -13)
            axis.annotate(
                f"175304/{row['run']}", (se3, sim3), xytext=offset,
                textcoords="offset points", fontsize=8,
            )
    low = min(all_values) * 0.75
    high = max(all_values) * 1.35
    axis.plot([low, high], [low, high], linestyle="--", color="#555555", label="SE(3)=Sim(3)：无尺度改善")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("SE(3) 对齐 APE RMSE (mm)")
    axis.set_ylabel("允许尺度校正的 Sim(3) APE RMSE (mm)")
    axis.grid(alpha=0.24, which="both")
    axis.legend(loc="lower right", fontsize=8.5)
    panel_note(
        axis,
        "x：保留估计尺度后的误差；y：额外校正尺度后的误差。\n"
        "规律：175304 从数十米降到约 100 mm，强证明确有尺度失稳；这只证明结果含尺度分量，不单独证明弱视差的因果来源。",
    )
    figure.suptitle("弱 temporal parallax 与尺度失稳：支持为放大器，不支持为唯一首因", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output)


def paired_delay_rows(
    rows: Sequence[dict[str, object]], sequence: str
) -> list[tuple[dict[str, object], dict[str, object]]]:
    changed_configuration = (
        "delay_40ms" if sequence == "20260803-184537" else "delay_39p25ms"
    )
    pairs = []
    for run in ("run1", "run2"):
        baseline = next(
            row for row in rows
            if row["sequence"] == sequence
            and row["run"] == run
            and row["configuration"] == "baseline_24p869740ms"
        )
        changed = next(
            row for row in rows
            if row["sequence"] == sequence
            and row["run"] == run
            and row["configuration"] == changed_configuration
        )
        pairs.append((baseline, changed))
    return pairs


def plot_image_delay_intervention(
    rows: Sequence[dict[str, object]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    metrics = (
        ("ape_rmse_mm", "APE RMSE (mm，对数轴)", "184537 校正后显著下降；39.25 ms 对另外两条序列反而恶化。"),
        ("ransac_fail_count", "primary RANSAC FAIL 次数（对数轴）", "184537 的 FAIL 下降一个数量级；两条反例序列明显增加。"),
        ("large_reprojection_count", "大重投影误差触发次数（对数轴）", "前端触发量与 APE 同方向响应，连接上游 delay 与视觉失效。"),
    )
    for axis, (metric, ylabel, rule) in zip(axes.flat[:3], metrics):
        for sequence in (
            "20260803-183537", "20260803-184027", "20260803-184537"
        ):
            colour = {
                "20260803-183537": "#27647b",
                "20260803-184027": "#d18b00",
                "20260803-184537": "#b3261e",
            }[sequence]
            for run_index, (baseline, changed) in enumerate(
                paired_delay_rows(rows, sequence)
            ):
                x_values = [0.0, 1.0]
                y_values = [float(baseline[metric]), float(changed[metric])]
                axis.plot(
                    x_values, y_values,
                    marker="o" if run_index == 0 else "s",
                    color=colour,
                    linewidth=2.0,
                    alpha=0.95 if run_index == 0 else 0.65,
                    label=sequence[-6:] if run_index == 0 else None,
                )
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.9)
        axis.set_xticks(
            [0.0, 1.0],
            ["24.87 ms\n基线", "39.25 ms（183537/184027）\n40.00 ms（184537）"],
        )
        axis.set_xlabel("固定 image_delay 配置")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="best", fontsize=8.2, ncol=3)
        panel_note(
            axis,
            f"x：固定 image_delay；y：{ylabel}。\n规律：{rule}",
            width=66,
            outside=True,
        )

    axis = axes.flat[3]
    controls = [
        ("基线\nrun1", "baseline_24p869740ms", "run1", "#777777"),
        ("仅平移\n评价时间戳", "timestamp_shift_only", "run1", "#d18b00"),
        ("固定外参\nrun1", "baseline_delay_fixed_extrinsics", "run1", "#7357a5"),
        ("40 ms\nrun1", "delay_40ms", "run1", "#16843b"),
        ("基线\nrun2", "baseline_24p869740ms", "run2", "#777777"),
        ("40 ms\nrun2", "delay_40ms", "run2", "#16843b"),
    ]
    values = []
    for _, configuration, run, _ in controls:
        row = next(
            item for item in rows
            if item["sequence"] == "20260803-184537"
            and item["configuration"] == configuration
            and item["run"] == run
        )
        values.append(float(row["ape_rmse_mm"]))
    positions = np.arange(len(controls))
    axis.bar(positions, values, color=[item[3] for item in controls])
    axis.set_xticks(positions, [item[0] for item in controls], fontsize=8.2)
    axis.set_ylabel("184537 APE RMSE (mm，对数轴)")
    axis.set_xlabel("控制实验配置")
    axis.set_yscale("log")
    axis.set_ylim(min(values) * 0.65, max(values) * 2.0)
    axis.grid(axis="y", alpha=0.25, which="both")
    for index, value in enumerate(values):
        axis.text(index, value * 1.10, f"{value:.3f}", ha="center", fontsize=8)
    panel_note(
        axis,
        "x：184537 的 delay/外参/评价控制；y：APE。\n"
        "规律：仅改 40 ms 才稳定改善；固定外参和只平移评价时间戳均不能复现改善。",
        width=66,
        outside=True,
    )
    figure.text(
        0.5, 0.012,
        "圆点为 run1，方块为 run2。39.25/40 ms 是序列级干预值，不是设备统一硬件延迟。",
        ha="center", fontsize=9.5,
        bbox={"facecolor": "#f4f4f2", "edgecolor": "#aaaaaa", "boxstyle": "round,pad=0.45"},
    )
    figure.suptitle("image_delay 受控干预：184537 改善与跨序列反例同时成立", fontsize=15)
    figure.tight_layout(rect=(0, 0.045, 1, 0.91), h_pad=6.2)
    save_figure(figure, output)


def evidence_rows() -> list[dict[str, str]]:
    return [
        {
            "link": "高角速度 × 时间误差 -> 姿态相位不一致",
            "evidence_level": "强支持",
            "evidence": "3.54° 理论差与 3.36°/3.43° 视觉修正闭合；184537 仅改 delay 后 APE、FAIL 和大重投影触发同步下降。",
            "limitation": "39.25 ms 会恶化另外两条序列，校正值不能跨序列通用。",
            "source": "analysis.md；image_delay_intervention_summary.csv",
            "scores": "3,3,3,3,3",
        },
        {
            "link": "时间误差的具体上游来源",
            "evidence_level": "尚未证明",
            "evidence": "多相机最佳有效延迟一致、分段值变化，且三个序列对同一 delay 干预方向不同。",
            "limitation": "仍无法区分 clock skew、曝光时刻语义、转换链路或低激励扫描偏差。",
            "source": "analysis.md；image_delay_intervention_summary.csv",
            "scores": "2,2,3,2,1",
        },
        {
            "link": "姿态不一致 -> 既有地图投影/匹配退化",
            "evidence_level": "强支持",
            "evidence": "26/26 高角 run 中预测重投影上升且地图匹配下降；198 个匹配事件中对应比例为 92.4%/68.2%。",
            "limitation": "有效时间误差、清晰度与场景内容仍有共同运动混杂。",
            "source": "population_angular_failure_chain.csv；population_angular_event_paired_effects.csv",
            "scores": "3,3,3,1,3",
        },
        {
            "link": "曝光期像移/模糊参与失配",
            "evidence_level": "中等支持",
            "evidence": "26/26 高角 run 清晰度下降；198 个匹配事件中 91.9% 下降，角位移剂量 rho=-0.393。",
            "limitation": "缺少曝光时间和曝光期角位移；不能作为唯一首因。",
            "source": "population_angular_failure_chain.csv；population_angular_event_paired_effects.csv",
            "scores": "2,3,3,1,2",
        },
        {
            "link": "重投影误差跨阈值 -> primary GP3P",
            "evidence_level": "强支持",
            "evidence": "源码阈值约 4.87 px；378 个事件中 224 个在高角窗口或随后 0.5 s 出现 GP3P FAIL。",
            "limitation": "多个健康 run 预测误差同样上升但未失败，跨阈值还取决于原地图支撑和误差余量。",
            "source": "Frontend.cpp；vio_diag_ransac.csv",
            "scores": "3,3,3,1,3",
        },
        {
            "link": "primary FAIL -> 未初始化 landmark retry",
            "evidence_level": "强支持",
            "evidence": "跨运行 FAIL 与 retry 计数接近一一对应，源码为直接失败分支。",
            "limitation": "retry 成功只保证局部位姿，不保证长期地图约束。",
            "source": "cross_sample_run_metrics.csv；Frontend.cpp",
            "scores": "3,3,3,3,3",
        },
        {
            "link": "预测/优化残差增大 -> >4 px observation 删除",
            "evidence_level": "强支持",
            "evidence": "26/26 高角 run 的 removal 增加，198 个匹配事件中 84.3% 增加；源码按 4 px 剔除。",
            "limitation": "计数体现优化检查结果，不单独定位每个残差的来源。",
            "source": "population_angular_failure_chain.csv；causal_frame_metrics.csv；Frontend.cpp",
            "scores": "3,3,3,1,3",
        },
        {
            "link": "observation 丢失 -> cleanup/短寿命/旧地图支撑下降",
            "evidence_level": "强支持",
            "evidence": "源码清理无观测 landmark；跨样本短寿命与高 FAIL/APE 强同现，active support 在匹配事件中 67.2% 朝下降。",
            "limitation": "support 常在高角窗口后才累积下降；本次全量刷新不读取超大 landmark event 正文。",
            "source": "causal_frame_metrics.csv；cross_sample_evidence.csv；ViSlamBackend.cpp",
            "scores": "3,3,3,3,3",
        },
        {
            "link": "landmark 短寿命/quality 差是独立诱因",
            "evidence_level": "不支持作为独立诱因",
            "evidence": "时间顺序显示二者主要位于 GP3P/删除之后；quality 通过比例与 APE 支持弱。",
            "limitation": "它们仍可作为反馈放大器。",
            "source": "causal_frame_metrics.csv；cross_sample_evidence.csv",
            "scores": "1,0,1,0,1",
        },
        {
            "link": "持续视觉碎片化 -> 回环候选/尝试激增",
            "evidence_level": "中等支持",
            "evidence": "回环尝试率与 APE 相关系数 +0.894，并与其他碎片化状态同现。",
            "limitation": "尚缺候选来源和事件级先后日志；不作为独立根因。",
            "source": "cross_sample_evidence.csv",
            "scores": "2,1,3,3,2",
        },
        {
            "link": "旧地图支撑重建 -> 冲击后恢复",
            "evidence_level": "强支持",
            "evidence": "175103 是自然反例：大峰值后支撑恢复且 APE 约 12 mm；175304 未恢复并崩溃。",
            "limitation": "自然对照不是受控干预。",
            "source": "causal_frame_metrics.csv；cross_sample_sequence_metrics.csv",
            "scores": "3,3,3,1,2",
        },
        {
            "link": "弱 temporal parallax -> 深度/尺度误差放大",
            "evidence_level": "中等支持",
            "evidence": "高旋转低平移代理与 APE 同向；175304 的 Sim(3) 改善约 99.6%-99.8%。",
            "limitation": "仅 61.6% 匹配事件 ray angle 下降，initialisable 下降只有 43.4%；不是统一首次中介。",
            "source": "cross_sample_sequence_metrics.csv；observability_sim3_by_run.csv",
            "scores": "2,2,3,3,2",
        },
        {
            "link": "错误位姿制造假视差及状态相互吸收残差",
            "evidence_level": "尚未证明",
            "evidence": "灾难性窗口中表观 baseline 上升而有效三角化率下降，与该机制相容。",
            "limitation": "缺少逐 landmark 深度方差和状态残差归因消融。",
            "source": "causal_event_metrics.csv；observability_failure_windows.csv",
            "scores": "2,2,2,0,2",
        },
        {
            "link": "final-BA 无法恢复已丢失/错误视觉约束",
            "evidence_level": "中等支持",
            "evidence": "184537 两次运行的 online/final/final-BA APE 均维持高位。",
            "limitation": "只直接验证了 184537，不能外推所有失败模式。",
            "source": "20260803-184537_failure_chain_stages.csv",
            "scores": "3,2,3,0,2",
        },
        {
            "link": "持续视觉碎片化 -> 高 APE RMSE",
            "evidence_level": "强支持",
            "evidence": "FAIL 率/寿命与 APE 相关系数为 +0.928/-0.935；184537 delay 干预使 FAIL 与 APE 同步下降。",
            "limitation": "delay 重放未输出完整 lifecycle CSV，尚未直接验证其中的删除量和 landmark 寿命同步恢复。",
            "source": "cross_sample_evidence.csv；image_delay_intervention_summary.csv",
            "scores": "3,3,3,3,3",
        },
    ]


def write_evidence_table(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("link", "evidence_level", "evidence", "limitation", "source")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_intervention_table(
    path: Path, rows: Sequence[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "sequence", "run", "configuration", "delay_ms", "extrinsics",
        "evaluation_control", "ape_rmse_mm", "ransac_fail_count",
        "large_reprojection_count", "source",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_evidence_matrix(rows: Sequence[dict[str, str]], output: Path) -> None:
    score_matrix = np.asarray(
        [[int(value) for value in row["scores"].split(",")] for row in rows],
        dtype=float,
    )
    labels = [f"{index + 1}. {row['link']}\n[{row['evidence_level']}]" for index, row in enumerate(rows)]
    columns = ("直接量测", "时间顺序", "重复/反例", "跨样本", "源码/物理")
    cmap = ListedColormap(["#e7e7e5", "#f2c879", "#78b7c5", "#247a4b"])
    figure, axis = plt.subplots(figsize=(13.8, 11.8))
    image = axis.imshow(score_matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5)
    axis.set_xticks(np.arange(len(columns)), columns)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8.4)
    axis.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    for row_index in range(score_matrix.shape[0]):
        for column_index in range(score_matrix.shape[1]):
            score = int(score_matrix[row_index, column_index])
            axis.text(column_index, row_index, str(score), ha="center", va="center", color="white" if score == 3 else "#202020", fontsize=9)
    axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    colourbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.03, ticks=[0, 1, 2, 3])
    colourbar.ax.set_yticklabels(["无/反证", "有限", "中等", "强"])
    axis.set_title(
        "证据维度评分（0-3）：评分表示对该行命题的支持，不代表整条链路已由单一实验证明",
        pad=18,
    )
    figure.suptitle("高角速度/大量旋转 -> 视觉碎片化 -> 高 APE：端到端证据矩阵", fontsize=15, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(figure, output)


def markdown_table(rows: Sequence[dict[str, str]]) -> str:
    lines = ["| 链路 | 证据等级 | 直接依据 | 证据边界 |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['link']} | **{row['evidence_level']}** | {row['evidence']} | {row['limitation']} |"
        )
    return "\n".join(lines)


def intervention_markdown_table(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "| 序列/重放 | 基线 delay | 基线 APE | 干预 delay | 干预 APE | APE 比值 | RANSAC FAIL（基线→干预） | 大重投影触发（基线→干预） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sequence in ("20260803-183537", "20260803-184027", "20260803-184537"):
        for baseline, changed in paired_delay_rows(rows, sequence):
            ratio = float(changed["ape_rmse_mm"]) / float(baseline["ape_rmse_mm"])
            lines.append(
                f"| `{sequence}/{baseline['run']}` | {float(baseline['delay_ms']):.3f} ms | "
                f"{float(baseline['ape_rmse_mm']):.3f} mm | {float(changed['delay_ms']):.3f} ms | "
                f"{float(changed['ape_rmse_mm']):.3f} mm | {ratio:.3f}× | "
                f"{int(baseline['ransac_fail_count'])}→{int(changed['ransac_fail_count'])} | "
                f"{int(baseline['large_reprojection_count'])}→{int(changed['large_reprojection_count'])} |"
            )
    return "\n".join(lines)


def write_report(
    path: Path,
    diagnostics_root: Path,
    sequence_rows: Sequence[dict[str, str]],
    evidence: Sequence[dict[str, str]],
    interventions: Sequence[dict[str, object]],
) -> None:
    diagnostic_sequences = sorted(
        child.name for child in diagnostics_root.iterdir()
        if child.is_dir() and child.name.startswith("202608")
    )
    cross_sample_count = len({row["sequence"] for row in sequence_rows})
    report = f"""# 高角速度/大量旋转到视觉碎片化与高 APE 的因果链报告

## 结论先行

现有结果已把链路推进到“**24 序列/48 重放全覆盖 + 物理量闭合 + 受控 delay 干预 + 同 run 匹配事件 + 恢复反例**”的程度，但仍不是每个中间箭头均由受控干预直接测量。最稳妥的总体结论是：

> 26 个存在 `>3 rad/s` 事件的重放均表现为图像清晰度下降、预测重投影误差上升、accepted map matches 下降和 observation removal 增加。这表明高角运动普遍制造前端压力；但健康重放未必跨过 GP3P 失败门限。`184537` 只修改固定 image_delay 后，APE、RANSAC FAIL 和大重投影误差触发同步下降，证明有效时间错位是该序列的主要可控上游触发器。跨过门限后，系统反复 retry、删除 observation 和清理 landmark，旧地图支撑若不能恢复才演化为持续视觉碎片化和高 APE。

但三条序列对 `184537` 拟合值附近的高 delay 设置响应方向不同：`184537` 使用 `40 ms` 显著改善，而 `39.25 ms` 显著破坏 `183537`、并恶化 `184027`。因此实验支持“**序列相关或序列内变化的有效时间错位 + OKVIS2 单一固定 delay 模型不足**”，不支持“整台设备统一改成 39.25 ms”。

## 1. 数据范围与证据口径

- `{len(diagnostic_sequences)}` 个序列、48 次重放均具有完整结构化 VIO 诊断输出；跨样本汇总覆盖 **{cross_sample_count} 个序列**。
- 全量诊断包含 378 个 `>3 rad/s` 事件，其中 198 个获得同 run 低角匹配对照；26 次重放包含高角事件。
- `20260805-122310 / 123231 / 123752` 使用不同 mocap 刚体。APE、线速度、平移路径和弱视差运动代理均已应用固定杠杆臂 `[-0.1195, -0.0035, 0.1563] m` 修正；平移修正不改变角速度。
- 分析器不会物化 `vio_diag_landmark_events.csv` 正文；全样本刷新从现有帧级汇总表按 run 流式处理。
- 三条抗冲击序列 `175103 / 175304 / 175539` 均是“长时间动作幅度较小，随后短时间快速转动”的实验；每条重放两次。
- `184537` 是持续大量旋转且同日 APE 明显偏高的样本，以 `183537 / 184027` 为自然对照。
- image_delay 干预覆盖 `184537` 的基线/40 ms/固定外参/评价时间戳控制，以及 `183537/184027` 各两次 39.25 ms 反例重放。
- 证据优先级：受控干预 > 同事件时间顺序 > 重复运行 > 恢复反例 > 跨序列相关性。Spearman 相关系数只描述单调关系，不能单独证明因果。

“视觉碎片化”在本文中不是人工二值标签，而是以下联合运行期状态：**已初始化地图的 GP3P 频繁失败、反复依赖未初始化/新生 landmark retry、优化后 observation 大量删除、landmark 高频 birth/death 与短寿命、active initialized map support 下降**。回环候选/尝试激增是该状态的下游响应，不作为独立根因。

## 2. 高角速度如何把时间误差放大到 OKVIS2 门限

![时间误差放大与门限闭合](figures/01_time_error_amplification.png)

OKVIS2 当前使用单一固定 `image_delay=24.869740 ms`，图像时间为 `raw timestamp - image_delay`，随后 IMU 被积分到这个修正后的图像时刻。基线对齐扫描显示，`184537` 全序列及四相机的最佳有效延迟约为 `39.25 / 41.75 / 41.50 / 39.25 ms`，失败窗口和分段估计还会变化；它只能说明存在序列相关、可能漂移的**有效时间误差**，尚不能定位 clock skew、曝光时刻语义或转换链路中的具体来源。

在首次密集失败附近，局部角速度约 `4.08 rad/s`、有效残余时差约 `15.13 ms`，一阶模型给出 `3.54°` 姿态差；两次重放中 GP3P 模型相对 IMU 数据关联起始位姿的修正分别为 `3.36°` 和 `3.43°`。量级和方向闭合，强支持图像与 IMU 处于同一旋转的不同时间相位。

源码给出两个不同门限：地图匹配投影搜索约 `3+f×0.06≈21.7 px`，平均重投影误差超过 `3+f×0.006≈4.87 px` 才触发 primary GP3P；优化后单 observation 超过 `4 px` 会被删除。因此 descriptor 候选“还能找到”与几何一致性“已经失败”可以同时成立。

## 3. image_delay 受控干预闭合上游触发链路

![image_delay 受控干预](figures/08_image_delay_intervention.png)

{intervention_markdown_table(interventions)}

`184537` 的两次成组重放只把 image_delay 从 `24.869740 ms` 改为 `40 ms`：APE 分别从 `103.168→10.743 mm` 和 `56.342→13.374 mm`；RANSAC FAIL 从 `142→7` 和 `94→8`；大重投影触发从 `215→19` 和 `278→19`。三类量在两次重放中同方向、跨数量级改善，使“时间错位是 `184537` 主要上游触发器”从机制支持升级为**强干预证据**。

两个替代解释没有复现这种改善：保留 `24.869740 ms` 但固定外参得到 `126.552 mm` APE、`135` 次 FAIL 和 `211` 次大重投影触发；不重放 VIO、只把基线输出时间戳平移 `-15.130260 ms` 后 APE 为 `103.396 mm`，与基线 run1 的 `103.168 mm` 基本相同。因此改善不是在线外参优化或 APE 评价时间关联制造的假象。

跨序列反例同样重要。把 `183537` 改为 `39.25 ms` 后，两次 APE 从 `7.633/6.653 mm` 恶化为 `148.168/100.296 mm`，FAIL 从 `21/18` 增至 `165/195`；`184027` 的 APE 从 `7.604/8.126 mm` 恶化为 `12.799/20.732 mm`，FAIL 从 `6/5` 增至 `90/72`。这证明 `39.25 ms` 只是 `184537` 的有效校正附近值，不是设备固定硬件延迟；也说明低旋转或不同激励下的“扫描最优 delay”不能不经重放验证就直接作为配置。

机器可读干预结果见 [`tables/image_delay_intervention_summary.csv`](tables/image_delay_intervention_summary.csv)。需要保留的证据边界是：40 ms 重放保存了 APE 和前端 FAIL/大重投影计数，但没有生成完整 lifecycle CSV，所以尚不能直接声称 observation 删除量和 landmark 寿命也已由该干预恢复。

## 4. 全样本高角速度到前端失效链

![全样本高角失稳链](figures/09_population_angular_failure_chain.png)

在 13 个含高角事件的序列、26 次重放中，高角区间相对各自事件前 baseline 的方向一致性为：预测重投影误差上升 `26/26`、accepted map matches 下降 `26/26`、observation removal 增加 `26/26`、Laplacian 清晰度下降 `26/26`。GP3P 失败帧率的中位数则从 `0%` 增至 `8.49%`，失败 run 与健康 run 明显分叉。

198 个同 run 高角事件与低角匹配对照进一步给出：清晰度下降 `91.9%`、预测重投影误差上升 `92.4%`、observation removal 增加 `84.3%`、地图匹配下降 `68.2%`。角位移剂量与清晰度变化的相关系数为 `-0.393 (p<1e-8)`，与地图匹配变化为 `-0.220 (p=0.0019)`。因此当前最稳的共同中介是“**清晰度/边缘内容下降 + 投影预测误差上升 -> 既有地图匹配削弱**”。

但 GP3P 并非所有 run 都失败：`183537 / 184027 / 174220 / 173716 / 174511` 等重放也经历上述压力，失败帧率仍为 0。是否跨过门限取决于原有地图支撑、有效时间误差、预测误差余量和恢复能力。高角速度是风险输入，不是脱离这些条件后的充分原因。

事件起点判据修正后，378 个事件中 224 个能在高角窗口或随后 0.5 秒内定位首次 GP3P 失败；其中只有 48 个同时检测到 accepted map matches 的 robust onset 不晚于首次失败。故报告不再声称每个单事件的精确 onset 顺序均已闭合，而以全 run 方向一致性和匹配事件效应作为主证据。

### 4.1 抗冲击实验的局部放大观察

![冲击期匹配退化](figures/02_impulse_matching_degradation.png)

三条抗冲击序列保留为机制放大观察，而不是总体结论的数据主体。其最强事件的两次重放中，特征点总数只变化 `-2.1%` 至 `+0.3%`，Laplacian 清晰度代理下降 `26%-51%`，最佳地图 descriptor 距离恶化 `93%-198%`，accepted map matches 下降 `23%-26%`，predicted reprojection error 中位数上升 `173%-443%`。

因此数据不支持“高速转动首先导致特征完全检测不到”。更符合数据的是：曝光期像移/模糊与预测姿态错位共同削弱既有地图对应，且**几何一致性恶化远强于特征数量变化**。由于日志缺少曝光时间和曝光期积分转角，模糊只评为中等支持的辅助因素，不能写成唯一首因。

## 5. primary GP3P 之后的碎片化级联

![GP3P retry 与删除级联](figures/03_gp3p_fragmentation_chain.png)

跨运行统计中，primary GP3P FAIL 与“启用未初始化 landmark 的 retry”计数几乎一一对应；源码也明确把 retry 放在 primary 失败分支。retry 可以给出局部位姿，但它依赖的是尚未形成长期地图约束的新生/未初始化 landmark。

`175304 / 175539` 的 reason-specific 日志进一步显示，优化后 `>4 px` 的 observation 删除量远高于 GP3P 当场 outlier 删除。失去 observation 的 landmark 随后由 `unobserved_landmark_cleanup` 清理。因此当前可观察到的顺序是：

```text
预测/匹配几何不一致
  -> primary non-central GP3P FAIL
  -> 未初始化/新生 landmark retry
  -> 优化后 >4 px observation 大量删除
  -> landmark 失去观测并 cleanup
  -> 高频 birth/death、短寿命、旧地图支撑下降
```

这也是为什么 landmark 寿命短不能单独写成上游诱因：在结构化日志中，它主要出现在 GP3P、重投影删除和 cleanup 之后。

## 6. 恢复分叉决定是否演化为高 APE

![冲击后的恢复分叉](figures/04_recovery_contrast.png)

`175103` 是关键反例：它具有很高的瞬时角速度，但旧地图支撑能重新建立，两次重放 APE 约 `12 mm`。`175304` 在强冲击后持续依赖短寿命 landmark，active initialized map support 下降，最终出现数十米级 SE(3) APE。`175539` 出现密集 FAIL/retry，但最终为约 `76-80 mm`，介于恢复和尺度崩溃之间。

因此不能设定“超过某个角速度就必然整轨失败”的单阈值。**角速度是误差放大输入，地图支撑能否恢复是系统从短时故障走向长期漂移的分叉点。**

## 7. 全样本碎片化状态、回环响应与 APE

![碎片化指标与 APE](figures/05_fragmentation_and_loop_response.png)

24 个序列的 Spearman 相关系数为：RANSAC FAIL 率与 APE `+0.928`，回环尝试率与 APE `+0.894`，landmark 中位寿命与 APE `-0.935`，observations/landmark 与 APE `-0.398`。正系数表示两者倾向同向变化，负系数表示反向变化；绝对值越接近 1，单调关系越强。

前三项关系很强，但因果解释不同：RANSAC FAIL 是最直接的运行期几何失败状态；landmark 寿命是删除/cleanup 的下游状态；回环尝试激增更像局部地图被切分后对旧帧外观候选的响应。当前没有逐候选来源和事件级回环先后日志，因此“碎片化导致回环候选激增”的具体箭头评为中等支持，且明确不把回环激增当作独立根因。

## 8. 弱 temporal parallax 与尺度失稳

![弱视差与尺度](figures/06_observability_and_scale.png)

高旋转/低平移窗口比例与 APE 呈同向关系，支持弱 temporal parallax 作为放大器。更直接的结果是：`175304` 的 SE(3) APE 为数十米，而允许尺度校正后的 Sim(3) APE 约 `100 mm`，说明灾难性误差主要含尺度分量。

但事件配对证据只在 `61.6%` 的事件中显示 temporal ray angle p10 朝退化方向变化，而 `initialisable_fraction` 只有 `43.4%` 朝下降方向、中位数反而上升。它们不支持“弱视差或不可初始化比例下降是所有首次 GP3P 失败的统一中介”。同时，冲击发生在初始化之后，运行期失败来自 non-central 3D-2D GP3P，不是 E 矩阵或五点法失败。错误位姿还可能制造表观假视差，因此弱视差只评为**中等支持的反馈放大器**。

## 9. 端到端证据矩阵

![端到端证据矩阵](figures/07_end_to_end_evidence_matrix.png)

{markdown_table(evidence)}

机器可读版本见 [`tables/evidence_summary.csv`](tables/evidence_summary.csv)。矩阵中的 0-3 分别表示对该行命题无支持/存在反证、有限支持、中等支持和强支持。它刻意把“尚未证明”和“不支持作为独立诱因”保留在主结论中，避免把合理机制写成既成因果事实。

## 10. 当前可成立的完整链路

```text
高角速度冲击或持续大量旋转
  + 固定/漂移的相机-IMU 有效时间误差                      [184537 含双重放强干预证据；上游来源尚未证明]
  + 曝光期像移/模糊、弱 temporal parallax                  [中等支持；辅助/放大因素]
  -> IMU 被积分到错误的图像时刻
  -> 姿态预测误差约 |ω|×|δt|                              [3.54° vs 3.36°/3.43° 闭合]
  -> 既有 3D landmark 预测投影偏离视觉观测
  -> descriptor 候选仍可能落在约 21.7 px 搜索范围
  -> 预测重投影误差上升、既有地图匹配下降                  [26/26 run 方向一致；198 匹配事件支持]
  -> 部分 run 平均误差超过约 4.87 px，触发 primary GP3P    [源码+日志强支持；健康反例未跨门限]
  -> 已初始化地图 inlier 支撑不足，primary FAIL
  -> 使用未初始化/新生 landmark retry                     [直接分支强支持]
  -> 优化后 >4 px observation 大量删除
  -> 无观测 landmark cleanup
  -> 高频 birth/death、短寿命、active initialized support 下降
     = “视觉碎片化”运行期状态                              [强支持；不是独立根因标签]
  -> 回环候选/尝试增多                                     [中等支持；下游响应]
  -> 旧地图支撑重建：RANSAC/删除回落，轨迹恢复（175103）
  -> 旧地图支撑未重建：长期依赖短寿命新 landmark
  -> 大旋转/小平移下深度和尺度约束弱                       [中等支持的放大器；非统一首次中介]
  -> 假视差、地图几何与状态相互吸收残差                    [尚未证明]
  -> 连续漂移、尺度失稳或不可恢复局部偏差
  -> final-BA 不能补回已经丢失/错误的视觉约束              [184537 中等支持]
  -> APE RMSE 升高                                         [跨样本+恢复反例+delay 干预强支持]
```

## 11. 下一步如何补强

1. 用结构化 diagnostics 再跑 `184537` 的 40 ms 组，直接验证 observation 删除、cleanup、landmark lifetime 和 active initialized support 是否随 FAIL 一起恢复。
2. 对 `184537` 做 `30/35/40/45/50 ms` 扫描，每格至少 3-5 次，并比较固定 offset 与分段/affine `t_imu=a×t_camera+b` 校正。
3. 保存 camera/IMU 的硬件时间、header stamp 与 recorded timestamp，区分固定 offset、clock skew 和曝光时刻语义。
4. 记录曝光时间和曝光期间陀螺积分，直接检验清晰度下降是否由旋转像移中介。
5. 输出逐 landmark 的真实 temporal/stereo baseline、ray angle、深度方差、birth/death 原因和使用它的 RANSAC invocation，区分真实弱视差与错误位姿制造的假视差。
6. 增加 loop candidate 的来源帧年龄、地图片段 ID 和尝试时间，验证回环激增确实发生在旧地图支撑下降之后。

## 12. 数据与源码索引

- 事件与帧级表：`workspace/ego2_results/202608_week1_analysis/tables/causal_event_metrics.csv`、`causal_frame_metrics.csv`
- 全样本角速度链表：`population_angular_failure_chain.csv`、`population_angular_event_paired_effects.csv`
- 跨样本表：`cross_sample_sequence_metrics.csv`、`cross_sample_run_metrics.csv`、`cross_sample_evidence.csv`
- 可观性表：`observability_sim3_by_run.csv`、`observability_failure_windows.csv`
- image_delay 实验：`workspace/ego2_results/202608_image_delay_experiments/README.md` 及各序列 `ape.txt`、`frontend_failure_counts.txt`、`run_manifest.txt`
- 汇总干预表：`causal_chain_report/tables/image_delay_intervention_summary.csv`
- 固定 image delay：`config/okvis2_eucm_EGO2.yaml:77`
- 图像时间修正：`okvis_multisensor_processing/src/ThreadedSlam.cpp:191`
- IMU propagation：`okvis_ceres/src/ViGraph.cpp:525`
- GP3P 触发/retry：`okvis_frontend/src/Frontend.cpp:1768`、`:1790`、`:1999`
- 地图匹配搜索范围：`okvis_frontend/src/Frontend.cpp:2054`
- 优化后 4 px 删除：`okvis_frontend/src/Frontend.cpp:3001`
- 无观测 landmark cleanup：`okvis_ceres/src/ViSlamBackend.cpp:1943`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    diagnostics_root = args.diagnostics_root.resolve()
    tables_root = args.tables_root.resolve()
    image_delay_root = args.image_delay_root.resolve()
    output = args.output.resolve()
    if not diagnostics_root.is_dir():
        raise FileNotFoundError(f"missing diagnostics root: {diagnostics_root}")
    if not image_delay_root.is_dir():
        raise FileNotFoundError(f"missing image-delay root: {image_delay_root}")
    tables = {
        key: read_rows(tables_root / name)
        for key, name in TABLE_SOURCES.items()
        if key != "frames"
    }
    tables["frames"] = read_impulse_frame_rows(
        tables_root / TABLE_SOURCES["frames"]
    )
    interventions = load_image_delay_interventions(image_delay_root, tables["runs"])
    configure_plot_style()

    figures = output / "figures"
    plot_time_error_amplification(figures / "01_time_error_amplification.png")
    plot_impulse_matching_degradation(
        tables["events"], figures / "02_impulse_matching_degradation.png"
    )
    plot_population_angular_failure_chain(
        tables["population_chain"],
        tables["paired_effects"],
        figures / "09_population_angular_failure_chain.png",
    )
    plot_gp3p_fragmentation_chain(
        tables["runs"], figures / "03_gp3p_fragmentation_chain.png"
    )
    plot_recovery_contrast(
        tables["events"], tables["frames"], tables["sequences"],
        figures / "04_recovery_contrast.png",
    )
    plot_fragmentation_and_loop_response(
        tables["sequences"], tables["evidence"],
        figures / "05_fragmentation_and_loop_response.png",
    )
    plot_observability_and_scale(
        tables["sequences"], tables["runs"], tables["evidence"],
        figures / "06_observability_and_scale.png",
    )
    evidence = evidence_rows()
    plot_evidence_matrix(evidence, figures / "07_end_to_end_evidence_matrix.png")
    plot_image_delay_intervention(
        interventions, figures / "08_image_delay_intervention.png"
    )
    write_evidence_table(output / "tables/evidence_summary.csv", evidence)
    write_intervention_table(
        output / "tables/image_delay_intervention_summary.csv", interventions
    )
    write_report(
        output / "report.md", diagnostics_root, tables["sequences"], evidence,
        interventions,
    )
    print(f"report: {output / 'report.md'}")
    print(f"figures: {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
