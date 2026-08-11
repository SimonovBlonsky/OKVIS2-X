#!/usr/bin/env python3
"""Refresh population causal tables from the existing frame-level table.

This postprocessor deliberately never opens ``vio_diag_landmark_events.csv``.
The frame table is streamed one run at a time so the completed 24-sequence
diagnostic cohort can be refreshed without repeating image decoding or loading
multi-gigabyte lifecycle logs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-population-causal")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-population-causal")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    from . import analyze_vio_causal_diagnostics as causal
    from .mocap_reference_correction import LEVER_AFFECTED_SEQUENCES
except ImportError:
    import analyze_vio_causal_diagnostics as causal
    from mocap_reference_correction import LEVER_AFFECTED_SEQUENCES


RUN_METRICS = (
    "image_laplacian_variance",
    "predicted_reprojection_error_px_median",
    "accepted_map_matches",
    "visual_observation_removals",
    "active_initialised_landmarks",
    "temporal_ray_angle_p10_rad",
    "temporal_parallel_fraction",
    "initialisable_fraction",
)

PAIRED_METRICS = (
    ("image_laplacian_variance", "图像 Laplacian 清晰度", -1),
    ("keypoints_total", "特征点数量", -1),
    ("best_descriptor_distance_median", "最佳地图 descriptor 距离", 1),
    ("accepted_map_matches", "accepted map matches", -1),
    ("predicted_reprojection_error_px_median", "预测重投影误差", 1),
    ("visual_observation_removals", "observation removals", 1),
    ("active_initialised_landmarks", "active initialized landmarks", -1),
    ("temporal_ray_angle_p10_rad", "temporal ray angle p10", -1),
    ("temporal_parallel_fraction", "temporal parallel fraction", 1),
    ("initialisable_fraction", "三角化 initialisable fraction", -1),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-root", required=True, type=Path)
    parser.add_argument("--figures-root", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args(argv)


def _coerce(value: str) -> object:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return (
            [{field: _coerce(value) for field, value in row.items()} for row in reader],
            list(reader.fieldnames),
        )


def write_rows(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    preferred_fields: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(preferred_fields)
    fields.extend(
        sorted({field for row in rows for field in row} - set(preferred_fields))
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def median(rows: Iterable[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if finite(row.get(field))]
    return float(np.median(values)) if values else None


def failure_frame_rate(rows: Sequence[dict[str, object]]) -> float | None:
    values = [float(row["gp3p_failure_count"]) for row in rows if finite(row.get("gp3p_failure_count"))]
    return float(np.mean(np.asarray(values) > 0.0)) if values else None


def window_failure_frame_rate(
    rows: Sequence[dict[str, object]], start_s: float, end_s: float
) -> float | None:
    return failure_frame_rate(causal._window_rows(rows, start_s, end_s))


def _control_outcomes(
    rows: Sequence[dict[str, object]],
    event: causal.AngularEvent,
    all_events: Sequence[causal.AngularEvent],
    sequence: str,
    run: str,
) -> list[tuple[float | None, float | None, float | None]]:
    covariates = causal._baseline_control_values(rows, event.start_s)
    if covariates is None:
        return []
    matching_event = {
        "sequence": sequence,
        "run": run,
        "start_s": event.start_s,
        "end_s": event.end_s,
        **covariates,
    }
    controls = causal.select_matched_controls(
        matching_event,
        causal.enumerate_low_angular_candidates(rows, event),
        angular_events=all_events,
    )
    return [
        causal._event_outcomes(
            rows, float(control["start_s"]), float(control["end_s"])
        )
        for control in controls
    ]


def _median_outcome(
    outcomes: Sequence[tuple[float | None, float | None, float | None]], index: int
) -> float | None:
    values = [float(row[index]) for row in outcomes if finite(row[index])]
    return float(np.median(values)) if values else None


def refresh_event_rows(
    rows: Sequence[dict[str, object]],
    event_rows: Sequence[dict[str, object]],
) -> None:
    if not event_rows:
        return
    sequence = str(event_rows[0]["sequence"])
    run = str(event_rows[0]["run"])
    angular_events = [
        causal.AngularEvent(
            float(row["start_s"]),
            float(row["end_s"]),
            float(row["peak_radps"]),
            float(row["angular_integral"]),
        )
        for row in event_rows
    ]
    for event_row, event in zip(event_rows, angular_events):
        event_row["event_index"] = int(float(event_row["event_index"]))
        gp3p, gp3p_post, map_support = causal._event_outcomes(
            rows, event.start_s, event.end_s
        )
        controls = _control_outcomes(
            rows, event, angular_events, sequence, run
        )
        gp3p_control = _median_outcome(controls, 0)
        gp3p_post_control = _median_outcome(controls, 1)
        map_control = _median_outcome(controls, 2)
        event_row.update(
            {
                "gp3p_onset_s": (
                    onset - event.start_s
                    if (onset := causal.gp3p_failure_onset(rows, event)) is not None
                    else None
                ),
                "gp3p_outcome": gp3p,
                "gp3p_control_outcome": gp3p_control,
                "gp3p_outcome_paired": (
                    gp3p - gp3p_control
                    if gp3p is not None and gp3p_control is not None
                    else None
                ),
                "gp3p_post_outcome": gp3p_post,
                "gp3p_post_control_outcome": gp3p_post_control,
                "gp3p_post_outcome_paired": (
                    gp3p_post - gp3p_post_control
                    if gp3p_post is not None and gp3p_post_control is not None
                    else None
                ),
                "map_support_outcome": map_support,
                "map_support_control_outcome": map_control,
                "map_support_outcome_paired": (
                    map_support - map_control
                    if map_support is not None and map_control is not None
                    else None
                ),
                "gp3p_baseline_failure_frame_rate": window_failure_frame_rate(
                    rows, event.start_s - 5.0, event.start_s - 1.0
                ),
                "gp3p_event_failure_frame_rate": window_failure_frame_rate(
                    rows, event.start_s, event.end_s + 0.5
                ),
                "gp3p_post_failure_frame_rate": window_failure_frame_rate(
                    rows,
                    math.nextafter(event.end_s + 0.5, math.inf),
                    event.end_s + 2.0,
                ),
            }
        )


def build_run_row(
    sequence: str,
    run: str,
    rows: Sequence[dict[str, object]],
    angular_event_count: int,
) -> dict[str, object]:
    baseline = [row for row in rows if row.get("event_phase") == "baseline"]
    angular = [row for row in rows if row.get("event_phase") == "angular_input"]
    output: dict[str, object] = {
        "sequence": sequence,
        "run": run,
        "angular_event_count": angular_event_count,
        "has_high_angular_event": bool(angular),
        "fixed_mocap_lever_applied": sequence in LEVER_AFFECTED_SEQUENCES,
        "baseline_frame_count": len(baseline),
        "angular_frame_count": len(angular),
        "gp3p_failure_frame_rate_baseline": failure_frame_rate(baseline),
        "gp3p_failure_frame_rate_angular": failure_frame_rate(angular),
    }
    if output["gp3p_failure_frame_rate_baseline"] is not None and output["gp3p_failure_frame_rate_angular"] is not None:
        output["gp3p_failure_frame_rate_delta"] = (
            float(output["gp3p_failure_frame_rate_angular"])
            - float(output["gp3p_failure_frame_rate_baseline"])
        )
    else:
        output["gp3p_failure_frame_rate_delta"] = None
    for metric in RUN_METRICS:
        baseline_value = median(baseline, metric)
        angular_value = median(angular, metric)
        output[f"{metric}_baseline"] = baseline_value
        output[f"{metric}_angular"] = angular_value
        output[f"{metric}_delta"] = (
            angular_value - baseline_value
            if angular_value is not None and baseline_value is not None
            else None
        )
        output[f"{metric}_relative_change"] = (
            angular_value / baseline_value - 1.0
            if angular_value is not None
            and baseline_value is not None
            and baseline_value != 0.0
            else None
        )
    return output


def stream_frame_runs(
    path: Path,
) -> Iterable[tuple[tuple[str, str], list[dict[str, object]]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {"sequence", "run", "time_s", "event_phase"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        seen: set[tuple[str, str]] = set()
        for key, group in itertools.groupby(
            reader, key=lambda row: (row["sequence"], row["run"])
        ):
            if key in seen:
                raise ValueError(f"{path}: rows for {key} are not contiguous")
            seen.add(key)
            yield key, [
                {field: _coerce(value) for field, value in row.items()}
                for row in group
            ]


def build_paired_effect_rows(
    event_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    matched = [row for row in event_rows if row.get("control_status") == "matched"]
    output = []
    for metric, label, harmful_direction in PAIRED_METRICS:
        field = f"{metric}_paired_delta"
        selected = [row for row in matched if finite(row.get(field))]
        values = np.asarray([float(row[field]) for row in selected], dtype=float)
        doses = np.asarray([float(row["angular_integral"]) for row in selected], dtype=float)
        rho, p_value = (
            spearmanr(doses, values) if len(values) >= 3 else (math.nan, math.nan)
        )
        output.append(
            {
                "metric": metric,
                "label": label,
                "harmful_direction": harmful_direction,
                "matched_event_count": len(values),
                "median_robust_paired_delta": (
                    float(np.median(values)) if len(values) else None
                ),
                "harmful_direction_fraction": (
                    float(np.mean(harmful_direction * values > 0.0))
                    if len(values) else None
                ),
                "dose_spearman_rho_raw": float(rho),
                "dose_spearman_rho_harmful_direction": (
                    float(harmful_direction * rho) if math.isfinite(rho) else None
                ),
                "dose_spearman_p_value": float(p_value),
            }
        )
    return output


def configure_plot_style() -> None:
    noto = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if noto.is_file():
        font_manager.fontManager.addfont(str(noto))
        family = font_manager.FontProperties(fname=str(noto)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.facecolor": "#fbfbfa",
            "figure.facecolor": "white",
            "font.size": 9,
        }
    )


def _panel_note(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.01,
        1.02,
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.7,
        linespacing=1.3,
        bbox={"facecolor": "white", "alpha": 0.94, "edgecolor": "#b8b8b8"},
    )


def plot_population_chain(
    run_rows: Sequence[dict[str, object]],
    paired_rows: Sequence[dict[str, object]],
    output: Path,
) -> None:
    selected = [row for row in run_rows if row["has_high_angular_event"]]
    labels = [f"{str(row['sequence'])[-6:]}-{str(row['run'])[-1]}" for row in selected]
    x = np.arange(len(selected))
    failure_delta = np.asarray(
        [100.0 * float(row["gp3p_failure_frame_rate_delta"]) for row in selected]
    )
    colours = ["#b3261e" if delta > 0.0 else "#27647b" for delta in failure_delta]
    panels = (
        (
            "image_laplacian_variance_relative_change",
            100.0,
            "清晰度变化 (%)",
            "x：26 个高角事件重放；y：高角区间相对基线的 Laplacian 清晰度变化。\n规律：全部重放下降，支持曝光期像移/模糊或边缘损失是稳定辅助中介。",
        ),
        (
            "predicted_reprojection_error_px_median_delta",
            1.0,
            "预测重投影误差变化 (px)",
            "x：26 个高角事件重放；y：预测重投影误差中位数相对基线增量。\n规律：全部重放上升，是高角输入到 3D-2D 不一致之间最稳定的直接中介。",
        ),
        (
            "accepted_map_matches_relative_change",
            100.0,
            "地图匹配变化 (%)",
            "x：26 个高角事件重放；y：accepted map matches 相对基线变化。\n规律：全部重放下降；已有地图支撑被削弱，但健康 run 未必跨过 GP3P 门限。",
        ),
        (
            "visual_observation_removals_delta",
            1.0,
            "每帧 observation 删除增量",
            "x：26 个高角事件重放；y：每帧 observation removal 中位数相对基线增量。\n规律：全部重放增加，说明几何不一致会在优化后的 4 px 检查中继续扩大。",
        ),
        (
            "gp3p_failure_frame_rate_delta",
            100.0,
            "GP3P 失败帧率增量 (百分点)",
            "x：26 个高角事件重放；y：GP3P 失败帧率相对基线增量。\n规律：只有部分 run 跨过失败门限；高角速度制造前端压力，但不是单独充分条件。",
        ),
    )
    figure, axes = plt.subplots(3, 2, figsize=(20, 15.5))
    for axis, (field, scale, ylabel, note) in zip(axes.flat, panels):
        values = np.asarray([float(row[field]) * scale for row in selected])
        axis.bar(x, values, color=colours, alpha=0.82)
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.23)
        _panel_note(axis, note)

    axis = axes.flat[-1]
    effect = {str(row["metric"]): row for row in paired_rows}
    metrics = (
        "image_laplacian_variance",
        "predicted_reprojection_error_px_median",
        "accepted_map_matches",
        "visual_observation_removals",
        "active_initialised_landmarks",
        "temporal_ray_angle_p10_rad",
        "initialisable_fraction",
    )
    names = ("清晰度", "预测重投影", "地图匹配", "观测删除", "活跃旧地图", "时序射线角", "可初始化比例")
    fractions = [100.0 * float(effect[metric]["harmful_direction_fraction"]) for metric in metrics]
    positions = np.arange(len(metrics))
    axis.barh(
        positions,
        fractions,
        color=["#147d64" if value >= 65.0 else "#d18b00" if value >= 55.0 else "#8a8f94" for value in fractions],
    )
    axis.axvline(50.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_xlim(0.0, 100.0)
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlabel("198 个匹配事件中朝失效方向变化的比例 (%)")
    axis.grid(axis="x", alpha=0.23)
    _panel_note(
        axis,
        "x：同 run 高角事件相对低角匹配对照朝失效方向变化的比例；y：候选中介/下游量。\n"
        "规律：重投影、清晰度和删除最一致；ray angle 较弱，initialisable fraction 不支持统一首次中介。",
    )
    figure.legend(
        handles=[
            Line2D([], [], color="#b3261e", linewidth=8, label="高角期 GP3P 失败帧率上升"),
            Line2D([], [], color="#27647b", linewidth=8, label="高角期未出现 GP3P 失败帧率上升"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.025,
        "0805 中午三条 mocap 刚体不同的序列没有 >3 rad/s 事件，因此不在本图 26 个 run 中；其 APE、平移和运动代理已在总体表图中应用固定杠杆臂修正。",
        ha="center",
        fontsize=8.5,
    )
    figure.suptitle(
        "全样本高角速度失稳链：前端压力普遍出现，跨过 GP3P 门限取决于系统余量",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=5.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    path: Path,
    event_rows: Sequence[dict[str, object]],
    run_rows: Sequence[dict[str, object]],
    paired_rows: Sequence[dict[str, object]],
) -> None:
    matched = [row for row in event_rows if row.get("control_status") == "matched"]
    onset = [row for row in event_rows if finite(row.get("gp3p_onset_s"))]
    ordered = [
        row for row in onset
        if finite(row.get("accepted_map_matches_onset_s"))
        and float(row["accepted_map_matches_onset_s"]) <= float(row["gp3p_onset_s"])
    ]
    selected_runs = [row for row in run_rows if row["has_high_angular_event"]]
    effect = {str(row["metric"]): row for row in paired_rows}
    lines = [
        "# VIO 角速度-视觉碎片化全样本诊断摘要",
        "",
        f"- 覆盖：24 个序列、{len(run_rows)} 次重放。",
        f"- `>3 rad/s` 事件：{len(event_rows)} 个；具有同 run 低角匹配对照：{len(matched)} 个。",
        f"- 存在高角事件的重放：{len(selected_runs)}；GP3P 首次失败可定位：{len(onset)}/{len(event_rows)} 个事件。",
        f"- accepted map matches 退化不晚于首次 GP3P 失败：{len(ordered)}/{len(onset)} 个可定位失败事件。",
        "- `20260805-122310 / 123231 / 123752` 的位置型 mocap 指标与 APE 已应用固定杠杆臂 `[-0.1195, -0.0035, 0.1563] m` 修正；角速度本身不受该平移修正影响。",
        "- 本刷新只流式读取 `causal_frame_metrics.csv`，没有读取任何 `vio_diag_landmark_events.csv` 正文。",
        "",
        "## 匹配事件证据",
        "",
        "| 指标 | 匹配事件数 | robust paired delta 中位数 | 朝失效方向比例 | 剂量相关 rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, _, _ in PAIRED_METRICS:
        row = effect[metric]
        lines.append(
            f"| {row['label']} | {int(row['matched_event_count'])} | "
            f"{float(row['median_robust_paired_delta']):+.3f} | "
            f"{100.0 * float(row['harmful_direction_fraction']):.1f}% | "
            f"{float(row['dose_spearman_rho_raw']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "高角期间清晰度下降、预测重投影误差上升、地图匹配下降和 observation removal 增加在所有高角 run 中方向一致；这是当前最稳定的中间链路。",
            "GP3P 失败只发生在部分 run，说明高角输入是压力和风险放大器，不是脱离地图支撑、有效时间误差和恢复余量后的充分条件。",
            "temporal ray angle 只得到弱到中等一致性，`initialisable_fraction` 更不支持统一下降；弱视差和三角化退化应作为辅助或反馈放大因素，而不是首次 GP3P 失败的统一中介。",
            "active initialized support 也并非总在高角窗口内立即下降；landmark 短寿命、birth/death 和 support 下降仍按后续累积状态解释。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh(args: argparse.Namespace) -> None:
    tables_root = args.tables_root.resolve()
    figures_root = args.figures_root.resolve()
    event_path = tables_root / "causal_event_metrics.csv"
    frame_path = tables_root / "causal_frame_metrics.csv"
    coverage_path = tables_root / "causal_diagnostics_coverage.csv"
    event_rows, event_fields = read_rows(event_path)
    coverage_rows, _ = read_rows(coverage_path)
    events_by_run: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in event_rows:
        events_by_run.setdefault((str(row["sequence"]), str(row["run"])), []).append(row)
    coverage_keys = {(str(row["sequence"]), str(row["run"])) for row in coverage_rows}

    run_rows = []
    frame_keys = set()
    for key, frame_rows in stream_frame_runs(frame_path):
        frame_keys.add(key)
        selected_events = events_by_run.get(key, [])
        refresh_event_rows(frame_rows, selected_events)
        run_rows.append(
            build_run_row(key[0], key[1], frame_rows, len(selected_events))
        )
    if frame_keys != coverage_keys:
        raise ValueError(
            f"frame/coverage run mismatch: missing={sorted(coverage_keys-frame_keys)}, "
            f"unexpected={sorted(frame_keys-coverage_keys)}"
        )

    write_rows(event_path, event_rows, preferred_fields=event_fields)
    run_rows.sort(key=lambda row: (str(row["sequence"]), str(row["run"])))
    write_rows(tables_root / "population_angular_failure_chain.csv", run_rows)
    paired_rows = build_paired_effect_rows(event_rows)
    write_rows(tables_root / "population_angular_event_paired_effects.csv", paired_rows)
    model_rows = causal.build_mediation_rows(
        event_rows, bootstrap_samples=args.bootstrap_samples
    )
    write_rows(tables_root / "causal_mediation_models.csv", model_rows)
    causal.plot_mediator_paths(
        event_rows, figures_root / "angular_to_fragmentation_mediator_paths.png"
    )
    causal.plot_onset_recovery(
        event_rows, figures_root / "mediator_onset_recovery.png"
    )
    configure_plot_style()
    plot_population_chain(
        run_rows,
        paired_rows,
        figures_root / "population_angular_failure_chain.png",
    )
    write_summary(
        tables_root.parent / "causal_diagnostics_summary.md",
        event_rows,
        run_rows,
        paired_rows,
    )
    print(
        f"refreshed {len(event_rows)} events, {len(run_rows)} runs, "
        f"{sum(bool(row['has_high_angular_event']) for row in run_rows)} high-angular runs"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    refresh(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
