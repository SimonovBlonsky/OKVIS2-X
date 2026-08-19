#!/usr/bin/env python3
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ego0_0813_timing_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ego0_0813_timing_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(
    os.environ.get(
        "EGO0_TIME_OFFSET_RESULTS_ROOT",
        str(REPOSITORY / "workspace/ego0_results/20260813_imu_time_fix_validation"),
    )
)
OUTPUT_IMAGE = OUTPUT_ROOT / "ego0_0813_imu_time_offset_quantitative_analysis.png"
OUTPUT_CSV = OUTPUT_ROOT / "ego0_0813_imu_time_offset_quantitative_analysis.csv"

SEQUENCES = (
    "173040",
    "173528",
    "174018",
    "174859",
    "175451",
    "180140",
    "180940",
    "181451",
    "182322",
    "185137",
)
IMU_CORRECTION_MS = np.asarray(
    (1297.5, 1296.5, 1299.0, 1297.0, 1297.5, 1293.5, 85.0, 1300.5, 1296.5, -1.0)
)
IMU_MOCAP_BEST_RHO = np.asarray(
    (0.9707, 0.9860, 0.9691, 0.9843, 0.9854, 0.8857, 0.9138, 0.9663, 0.9741, 0.9866)
)
IMU_MOCAP_ZERO_RHO = np.asarray(
    (0.2580, -0.0045, 0.1275, -0.0244, 0.0797, 0.2385, 0.8591, 0.1240, 0.1018, 0.9866)
)

IMAGE_IMU = {
    "181451": {"zero_rho": -0.0216, "best_rho": 0.9712, "correction_ms": 1300.0},
    "185137": {"zero_rho": 0.9512, "best_rho": 0.9517, "correction_ms": 4.0},
}
IMAGE_MOCAP_BEST_MS = {"181451": 8.0, "185137": -7.0}

IMAGE_DELAY_MS = 1.962929

INTERVENTION = {
    "raw": {
        "ape_rmse_m": 2633.978588,
        "large_reprojection_error": 9609,
        "ransac_fail": 7956,
    },
    "imu_plus_1298ms": {
        "ape_rmse_m": 0.011086,
        "large_reprojection_error": 0,
        "ransac_fail": 0,
    },
}

ACCURACY_RESULTS = (
    {
        "label": "原始 181451",
        "ape_rmse_m": 2633.978588,
        "annotation": "2633.979 m",
        "kind": "failed",
    },
    {
        "label": "181451\nIMU +1298 ms",
        "ape_rmse_m": 0.011086,
        "annotation": "11.086 mm",
        "kind": "corrected",
    },
    {
        "label": "180140\nIMU +1294 ms",
        "ape_rmse_m": 0.002169788,
        "annotation": "2.170 mm",
        "kind": "corrected",
    },
    {
        "label": "182322\nIMU +1300 ms",
        "ape_rmse_m": 0.004668602,
        "annotation": "4.669 mm",
        "kind": "corrected",
    },
    {
        "label": "185137\n原始时间戳",
        "ape_rmse_m": 0.009120,
        "annotation": "9.120 mm",
        "kind": "control",
    },
)

DROP_TIMELINE_ELAPSED_S = np.asarray(
    (224.2368, 226.3702, 226.8702, 227.3702, 229.0369, 231.5369, 232.5369)
)
DROP_TIMELINE_ERROR_M = np.asarray(
    (0.005167, 3.101297, 3.758526, 4.846113, 12.915503, 38.439309, 52.975150)
)
CAMERA_MESSAGE_COUNTS = np.asarray((7425, 7773, 7773, 7773))


def configure_fonts() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    else:
        font_name = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def annotate_bars(axis, bars, formatter, *, padding=3) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            formatter(value),
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def draw_offset_panel(axis) -> None:
    colors = ["#d1495b" if abs(value) > 500.0 else "#2a9d8f" for value in IMU_CORRECTION_MS]
    bars = axis.bar(np.arange(len(SEQUENCES)), IMU_CORRECTION_MS, color=colors, width=0.72)
    axis.axhspan(-15.0, 15.0, color="#2a9d8f", alpha=0.12, label="近同步（±15 ms）")
    axis.set_ylim(-80.0, 1435.0)
    axis.set_xticks(np.arange(len(SEQUENCES)), SEQUENCES, rotation=36, ha="right")
    axis.set_ylabel("IMU 时间戳需增加的补偿量 (ms)")
    axis.set_title("A. 时间错位的普遍性：8/10 条序列的 IMU 约提前 1.3 s", loc="left")
    axis.grid(axis="y", color="#d9dde0", linewidth=0.7, alpha=0.8)
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    for bar, offset, rho in zip(bars, IMU_CORRECTION_MS, IMU_MOCAP_BEST_RHO):
        axis.annotate(
            f"{offset:+.1f}\nρ={rho:.3f}",
            (bar.get_x() + bar.get_width() / 2.0, max(offset, 0.0)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    axis.text(
        0.012,
        0.84,
        "红色：>500 ms；绿色：小偏移\n185137（正常）为 −1.0 ms",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#aeb4b9", "alpha": 0.9},
    )


def draw_correlation_panel(axis) -> None:
    axis.set_axis_off()
    axis.set_title("B. 如何识别时间错位、计算 ρ 并拟合理论补偿量", loc="left")

    correlation_axis = axis.inset_axes((0.02, 0.10, 0.41, 0.76))
    labels = tuple(IMAGE_IMU)
    positions = np.arange(len(labels))
    width = 0.34
    zero = [IMAGE_IMU[label]["zero_rho"] for label in labels]
    best = [IMAGE_IMU[label]["best_rho"] for label in labels]
    zero_bars = correlation_axis.bar(
        positions - width / 2.0,
        zero,
        width,
        color="#8c96a0",
        label="原始时间戳",
    )
    best_bars = correlation_axis.bar(
        positions + width / 2.0,
        best,
        width,
        color="#2a9d8f",
        label="最佳补偿后",
    )
    correlation_axis.axhline(0.0, color="#202124", linewidth=0.8)
    correlation_axis.set_ylim(-0.12, 1.13)
    correlation_axis.set_xticks(positions, labels)
    correlation_axis.set_ylabel("图像运动–IMU 的 Spearman ρ", fontsize=8.5)
    correlation_axis.grid(axis="y", color="#d9dde0", linewidth=0.7, alpha=0.8)
    correlation_axis.legend(loc="lower right", frameon=False, fontsize=7.5)
    annotate_bars(correlation_axis, zero_bars, lambda value: f"{value:.4f}")
    annotate_bars(correlation_axis, best_bars, lambda value: f"{value:.4f}")

    method_text = (
        "① 构造可比较的旋转运动强度\n"
        r"$q_I(t)=\|\omega_{IMU}(t)\|_2$" "；"
        r"$q_M(t)=\|\mathrm{Log}(R_M(t)^TR_M(t+\delta t))\|_2/\delta t$"
        "\n图像侧 q_C(t) 由相邻帧视觉旋转/运动强度得到。\n\n"
        "② 对候选补偿 d，把原 IMU 样本时间改为 t+d：\n"
        r"$q_I^{d}(t)=\mathrm{Interp}[q_I](t-d)$"
        "；仅保留共同时间范围。\n\n"
        "③ Spearman ρ 是两列秩的 Pearson 相关：\n"
        r"$\rho(d)=\mathrm{Cov}(\mathrm{rank}(q_{ref}),\mathrm{rank}(q_I^d))"
        r"/(\sigma_{rank(ref)}\sigma_{rank(I)})$"
        "\n它比较波形的单调一致性，不依赖两传感器量纲。\n\n"
        "④ 扫描 d 并对相关峰邻域拟合："
        r"$\hat d=\arg\max_d\rho(d)$"
        "。\n"
        "实际写入 CSV："
        r"$d_{run}\approx\hat d-d_{image}$"
        f"（image_delay={IMAGE_DELAY_MS:.6f} ms）。\n"
        "例：181451 为 1300−1.962929≈1298 ms；"
        "180140/182322 分别写入 +1294/+1300 ms。"
    )
    axis.text(
        0.47,
        0.91,
        method_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        linespacing=1.30,
        bbox={"facecolor": "#f7f9fa", "edgecolor": "#aeb4b9", "alpha": 0.96},
    )


def draw_ape_panel(axis) -> None:
    labels = [result["label"] for result in ACCURACY_RESULTS]
    values = [result["ape_rmse_m"] for result in ACCURACY_RESULTS]
    colors = {
        "failed": "#d1495b",
        "corrected": "#2a9d8f",
        "control": "#457b9d",
    }
    bars = axis.bar(
        labels,
        values,
        color=[colors[result["kind"]] for result in ACCURACY_RESULTS],
        width=0.66,
    )
    axis.set_yscale("log")
    axis.set_ylabel("SE(3) APE RMSE (m，对数轴)")
    axis.set_title("C. 只修正 IMU 时间戳：三条退化序列恢复到 2.2–11.1 mm", loc="left")
    axis.grid(axis="y", which="both", color="#d9dde0", linewidth=0.7, alpha=0.8)
    for bar, result in zip(bars, ACCURACY_RESULTS):
        axis.annotate(
            result["annotation"],
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.text(
        0.31,
        0.58,
        "181451 严格受控：相机、内外参、配置和程序不变，仅移动 IMU 时间戳\n"
        "185137 无约 1.3 s 时间错位，原始 run2 为 9.120 mm（run1 为 17.616 mm）",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#aeb4b9", "alpha": 0.9},
    )


def draw_failure_panel(axis) -> None:
    labels = ("大重投影误差", "RANSAC FAIL")
    positions = np.arange(len(labels))
    width = 0.34
    raw = (
        INTERVENTION["raw"]["large_reprojection_error"],
        INTERVENTION["raw"]["ransac_fail"],
    )
    corrected = (
        INTERVENTION["imu_plus_1298ms"]["large_reprojection_error"],
        INTERVENTION["imu_plus_1298ms"]["ransac_fail"],
    )
    raw_bars = axis.bar(positions - width / 2.0, raw, width, color="#d1495b", label="原始")
    corrected_bars = axis.bar(positions + width / 2.0, corrected, width, color="#2a9d8f", label="IMU +1.298 s")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("run.log 事件计数")
    axis.set_title("D. 181451 受控干预：时间修正后前端几何失败归零", loc="left")
    axis.grid(axis="y", color="#d9dde0", linewidth=0.7, alpha=0.8)
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    annotate_bars(axis, raw_bars, lambda value: f"{int(value)}")
    annotate_bars(axis, corrected_bars, lambda value: f"{int(value)}")
    axis.text(
        0.02,
        0.92,
        "相机 CSV/PNG、内外参、OKVIS 配置与程序均未改变",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#aeb4b9", "alpha": 0.9},
    )


def draw_frame_drop_timeline(axis) -> None:
    axis.plot(
        DROP_TIMELINE_ELAPSED_S,
        DROP_TIMELINE_ERROR_M,
        color="#d1495b",
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="按缺口前轨迹对齐后的平移误差",
    )
    axis.axvspan(224.3000, 226.3667, color="#f4a261", alpha=0.26)
    axis.axvspan(229.1000, 231.5333, color="#f4a261", alpha=0.26)
    axis.set_yscale("log")
    axis.set_xlim(223.8, 233.1)
    axis.set_ylim(0.002, 100.0)
    axis.set_xlabel("相对序列起点时间 (s)")
    axis.set_ylabel("平移误差 (m，对数轴)")
    axis.set_title("E. 180940：cam0 掉帧后，误差由毫米级瞬间跳到米级", loc="left")
    axis.grid(which="both", color="#d9dde0", linewidth=0.7, alpha=0.8)
    axis.legend(loc="lower right", frameon=False, fontsize=8.5)
    axis.text(225.33, 50.0, "cam0 缺 61 帧\n视觉中断 2.067 s", ha="center", va="top", fontsize=8)
    axis.text(230.32, 50.0, "cam0 缺 72 帧\n视觉中断 2.433 s", ha="center", va="top", fontsize=8)
    axis.text(
        0.02,
        0.08,
        "缺口前全段 APE RMSE = 4.919 mm；缺口前最后 30 s = 2.477 mm\n"
        "第一缺口恢复首帧误差 = 3.101 m，随后立即出现 tracking failure / RANSAC FAIL",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#aeb4b9", "alpha": 0.92},
    )


def draw_camera_count_panel(axis) -> None:
    labels = ("cam0", "cam1", "cam2", "cam3")
    bars = axis.bar(
        labels,
        CAMERA_MESSAGE_COUNTS,
        color=("#d1495b", "#8c96a0", "#8c96a0", "#8c96a0"),
        width=0.66,
    )
    axis.set_ylim(0, 8700)
    axis.set_ylabel("原始 MCAP 消息数")
    axis.set_title("F. 原始 MCAP 已存在严重掉帧：不是 EuRoC 转换造成", loc="left")
    axis.grid(axis="y", color="#d9dde0", linewidth=0.7, alpha=0.8)
    annotate_bars(axis, bars, lambda value: f"{int(value)}")
    axis.text(
        0.02,
        0.94,
        "cam0 比每一路其他相机少 348 帧；MCAP 与 EuRoC 计数完全一致",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#aeb4b9", "alpha": 0.92},
    )
    axis.text(
        0.02,
        0.10,
        "独立故障链：cam0 缺帧 → 其余三路因无四目对应被丢弃 → 无视觉更新\n"
        "→ 3d2d tracking lost → 连续 RANSAC FAIL → 轨迹发散",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#7f1d1d",
        bbox={"facecolor": "#fff7f5", "edgecolor": "#d1495b", "alpha": 0.95},
    )


def save_csv() -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "sequence",
                "imu_timestamp_correction_ms",
                "imu_mocap_best_rho",
                "imu_mocap_zero_rho",
            ]
        )
        for values in zip(
            SEQUENCES, IMU_CORRECTION_MS, IMU_MOCAP_BEST_RHO, IMU_MOCAP_ZERO_RHO
        ):
            writer.writerow(values)
        writer.writerow([])
        writer.writerow(["181451_control", "raw", "imu_plus_1298ms"])
        writer.writerow(
            [
                "ape_rmse_m",
                INTERVENTION["raw"]["ape_rmse_m"],
                INTERVENTION["imu_plus_1298ms"]["ape_rmse_m"],
            ]
        )
        writer.writerow(
            [
                "large_reprojection_error",
                INTERVENTION["raw"]["large_reprojection_error"],
                INTERVENTION["imu_plus_1298ms"]["large_reprojection_error"],
            ]
        )
        writer.writerow(
            [
                "ransac_fail",
                INTERVENTION["raw"]["ransac_fail"],
                INTERVENTION["imu_plus_1298ms"]["ransac_fail"],
            ]
        )
        writer.writerow([])
        writer.writerow(["ape_validation", "ape_rmse_m", "annotation", "kind"])
        for result in ACCURACY_RESULTS:
            writer.writerow(
                [
                    result["label"].replace("\n", " "),
                    result["ape_rmse_m"],
                    result["annotation"],
                    result["kind"],
                ]
            )
        writer.writerow([])
        writer.writerow(["180940_frame_drop", "value"])
        writer.writerow(["pre_gap_ape_rmse_m", 0.004918758])
        writer.writerow(["last_30s_pre_gap_ape_rmse_m", 0.002476508])
        writer.writerow(["first_gap_missing_cam0_frames", 61])
        writer.writerow(["first_gap_duration_s", 2.066667])
        writer.writerow(["first_frame_error_after_resume_m", 3.101297])
        writer.writerow(["second_gap_missing_cam0_frames", 72])
        writer.writerow(["second_gap_duration_s", 2.433333])
        writer.writerow(["cam0_mcap_messages", CAMERA_MESSAGE_COUNTS[0]])
        writer.writerow(["cam1_to_cam3_mcap_messages_each", CAMERA_MESSAGE_COUNTS[1]])
        writer.writerow([])
        writer.writerow(["method", "formula_or_value"])
        writer.writerow(["spearman", "Pearson correlation of the two rank vectors"])
        writer.writerow(["best_correction", "argmax_d rho(d)"])
        writer.writerow(["configured_image_delay_ms", IMAGE_DELAY_MS])
        writer.writerow(["runtime_imu_shift", "best raw-image correction minus image_delay"])


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(18.5, 17.0),
        gridspec_kw={"height_ratios": (1.08, 1.0, 1.0)},
    )
    draw_offset_panel(axes[0, 0])
    draw_correlation_panel(axes[0, 1])
    draw_ape_panel(axes[1, 0])
    draw_failure_panel(axes[1, 1])
    draw_frame_drop_timeline(axes[2, 0])
    draw_camera_count_panel(axes[2, 1])
    figure.suptitle(
        "EGO0 2026-08-13：IMU 时间戳提前是多数序列 SLAM 退化主因；相机掉帧可能独立触发退化",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.968,
        "时间修正后 180140 / 181451 / 182322 达到 2.170 / 11.086 / 4.669 mm；"
        "无约 1.3 s 错位的 185137 同样为毫米级",
        ha="center",
        va="top",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.017,
        "时间修正实验仅改变 imu0/data.csv 时间戳；相机数据、内外参、OKVIS 配置与程序不变。"
        "APE 均使用对应 mocap 与 SE(3) Umeyama 对齐。180940 时间线使用缺口前轨迹拟合的同一刚体变换。\n"
        "结论边界：时间错位解释多数完整序列的系统性退化；180940 证明原始采集中的 cam0 掉帧是另一项可独立导致退化的数据质量问题。",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.02, 0.065, 0.98, 0.945), h_pad=2.7, w_pad=2.4)
    figure.savefig(OUTPUT_IMAGE, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    save_csv()
    print(f"Image: {OUTPUT_IMAGE}")
    print(f"Data: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
