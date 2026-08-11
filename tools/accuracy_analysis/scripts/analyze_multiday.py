#!/usr/bin/env python3
"""Compare motion excitation, alarm counts, and APE across EGO2 days."""

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "okvis_analysis_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "okvis_analysis_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.accuracy_analysis.scripts import mocap_reference_correction as day_analysis
from tools.accuracy_analysis.scripts import analyze_repeatability as repeatability


FINAL_BA_FILE = day_analysis.FINAL_BA_FILE
DEFAULT_DAYS = ("20260803", "20260804", "20260805", "20260806")
DEFAULT_EXPECTED_SEQUENCES = 24
DEFAULT_RESULTS_ROOT = REPOSITORY / "workspace/ego2_results"
DEFAULT_DATA_ROOT = Path("/home/chenguyuan/data")
DEFAULT_OUTPUT = (
    REPOSITORY / "workspace/ego2_results/20260803_20260805_accuracy_analysis"
)
CANDIDATE_THRESHOLD_RADPS = 3.0
MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN = 3.0
APE_THRESHOLD_MM = 10.0
FAILURE_CHAIN_TARGET = "20260803-184537"
FAILURE_CHAIN_TIMELINE_S = (85.0, 115.0)
FAILURE_CHAIN_DELTA_S = 5.0
LOOP_ATTEMPT_TIMER = "2.07 attempt loop closure"
LOOP_ACCEPTED_TIMER = "2.08 add loop closure"
DAY_COLORS = {
    "20260803": "#b3261e",
    "20260804": "#147d64",
    "20260805": "#2457a6",
    "20260806": "#8c5a00",
}
DAY_MARKERS = {
    "20260803": "s",
    "20260804": "^",
    "20260805": "o",
    "20260806": "D",
}
MOTION_METRICS = (
    ("motion_angular_speed_radps_p95", "Mocap angular speed p95 [rad/s]"),
    ("orientation_path_rad", "Cumulative orientation path [rad]"),
    ("translation_path_m", "Mocap translation path [m]"),
    (
        "translation_per_orientation_m_per_rad",
        "Translation / orientation [m/rad]",
    ),
)


def orientation_excitation_metrics(
    timestamps: np.ndarray, quaternions_wxyz: np.ndarray
) -> dict[str, float]:
    timestamps_array = np.asarray(timestamps, dtype=float)
    intervals = np.diff(timestamps_array)
    angular_speed = repeatability.angular_speed(
        timestamps_array, np.asarray(quaternions_wxyz, dtype=float)
    )
    return {
        "orientation_path_rad": float(np.sum(angular_speed * intervals))
    }


def _debounced_event_count(
    timestamps: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    minimum_duration_s: float = 0.05,
    merge_gap_s: float = 0.25,
) -> int:
    times = np.asarray(timestamps, dtype=float)
    samples = np.asarray(values, dtype=float)
    if (
        times.ndim != 1
        or samples.shape != times.shape
        or len(times) < 2
        or np.any(np.diff(times) <= 0.0)
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("alarm timestamps and values must be finite vectors")
    active = samples > threshold
    transitions = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    qualified = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if times[stop] - times[start] >= minimum_duration_s
    ]
    merged: list[tuple[int, int]] = []
    for start, stop in qualified:
        if merged and times[start] - times[merged[-1][1]] < merge_gap_s:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return len(merged)


def alarm_threshold_rows(
    sequence_rows: list[dict],
    contexts: dict[str, dict],
    *,
    thresholds_radps: np.ndarray | None = None,
) -> list[dict]:
    thresholds = np.asarray(
        thresholds_radps
        if thresholds_radps is not None
        else np.arange(2.0, 4.5001, 0.05),
        dtype=float,
    )
    if (
        thresholds.ndim != 1
        or not len(thresholds)
        or not np.all(np.isfinite(thresholds))
    ):
        raise ValueError("alarm thresholds must be a finite non-empty vector")
    output = []
    for sequence_row in sequence_rows:
        sequence = str(sequence_row["sequence"])
        if sequence not in contexts:
            raise ValueError(f"{sequence}: missing alarm context")
        duration_s = float(sequence_row["analysis_duration_s"])
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(f"{sequence}: invalid analysis duration")
        context = contexts[sequence]
        sources = {
            "mocap": (
                context["motion"]["timestamps"],
                context["motion"]["angular_speed"],
            ),
            "imu": (
                context["imu_timeseries"]["timestamps"],
                context["imu_timeseries"]["gyro_norm"],
            ),
        }
        ape_field = (
            "fixed_lever_ape_median_mm"
            if "fixed_lever_ape_median_mm" in sequence_row
            else "corrected_ape_median_mm"
        )
        ape_mm = float(sequence_row[ape_field])
        for source, (timestamps, values) in sources.items():
            for threshold in thresholds:
                event_count = _debounced_event_count(
                    timestamps, values, float(threshold)
                )
                output.append(
                    {
                        "group": sequence_row.get("group", "unclassified"),
                        "sequence": sequence,
                        "source": source,
                        "threshold_radps": float(threshold),
                        "event_count": event_count,
                        "events_per_5min": event_count * 300.0 / duration_s,
                        "ape_rmse_mm": ape_mm,
                        "ape_over_10mm": bool(ape_mm > APE_THRESHOLD_MM),
                    }
                )
    return output


@dataclass(frozen=True)
class SequenceSpec:
    day: str
    group: str
    sequence: str
    sequence_dir: Path
    run_dirs: tuple[Path, ...]
    dataset: Path
    mocap: Path


def analysis_period_label(days: tuple[str, ...] | list[str]) -> str:
    """Describe the inclusive analysis period represented by day labels."""
    selected_days = sorted(set(days))
    if not selected_days:
        raise ValueError("analysis period must contain at least one day")
    count_names = {1: "Single-day", 2: "Two-day", 3: "Three-day", 4: "Four-day"}
    count_label = count_names.get(len(selected_days), f"{len(selected_days)}-day")
    return f"{count_label} ({selected_days[0]}-{selected_days[-1]})"


def discover_sequences(
    results_root: Path,
    data_root: Path,
    *,
    days: tuple[str, ...] = DEFAULT_DAYS,
) -> list[SequenceSpec]:
    """Discover both flat day/sequence and grouped day/group/sequence layouts."""
    sequences = []
    for day in days:
        day_results = Path(results_root) / day
        day_data = Path(data_root) / day
        if not day_results.is_dir():
            raise ValueError(f"results day not found: {day_results}")
        if not day_data.is_dir():
            raise ValueError(f"dataset day not found: {day_data}")
        datasets = {
            path.name.removesuffix("_euroc"): path
            for path in day_data.rglob("*_euroc")
            if path.is_dir()
        }
        sequence_dirs = sorted(
            path
            for path in day_results.rglob("*")
            if path.is_dir() and any(path.glob("mocap_*.log"))
        )
        for sequence_dir in sequence_dirs:
            mocap_logs = sorted(sequence_dir.glob("mocap_*.log"))
            if len(mocap_logs) != 1:
                raise ValueError(
                    f"expected one mocap log in {sequence_dir}, "
                    f"found {len(mocap_logs)}"
                )
            run_dirs = tuple(
                sorted(
                    run_dir
                    for run_dir in sequence_dir.glob("run*")
                    if run_dir.is_dir()
                    and (run_dir / FINAL_BA_FILE).is_file()
                )
            )
            if not run_dirs:
                continue
            sequence = sequence_dir.name
            if sequence not in datasets:
                raise ValueError(f"dataset not found for {sequence}")
            group = (
                "unclassified"
                if sequence_dir.parent == day_results
                else sequence_dir.parent.name
            )
            sequences.append(
                SequenceSpec(
                    day=day,
                    group=group,
                    sequence=sequence,
                    sequence_dir=sequence_dir,
                    run_dirs=run_dirs,
                    dataset=datasets[sequence],
                    mocap=mocap_logs[0],
                )
            )
    return sorted(sequences, key=lambda item: (item.day, item.sequence))


def _spearman(values: np.ndarray, ape: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    ape = np.asarray(ape, dtype=float)
    valid = np.isfinite(values) & np.isfinite(ape)
    count = int(np.count_nonzero(valid))
    if count < 3 or np.ptp(values[valid]) == 0.0 or np.ptp(ape[valid]) == 0.0:
        return float("nan"), float("nan"), count
    result = spearmanr(values[valid], ape[valid])
    return float(result.statistic), float(result.pvalue), count


def correlation_comparison(rows: list[dict], metric: str) -> dict:
    if not rows:
        raise ValueError("correlation rows must not be empty")
    all_values = np.asarray([row[metric] for row in rows], dtype=float)
    all_ape = np.asarray(
        [row["corrected_ape_median_mm"] for row in rows], dtype=float
    )
    day_rows = [row for row in rows if row["day"] == "20260805"]
    day_values = np.asarray([row[metric] for row in day_rows], dtype=float)
    day_ape = np.asarray(
        [row["corrected_ape_median_mm"] for row in day_rows], dtype=float
    )
    all_rho, all_pvalue, all_count = _spearman(all_values, all_ape)
    day_rho, day_pvalue, day_count = _spearman(day_values, day_ape)
    return {
        "metric": metric,
        "all_sequences": all_count,
        "all_rho": all_rho,
        "all_pvalue": all_pvalue,
        "day_20260805_sequences": day_count,
        "day_20260805_rho": day_rho,
        "day_20260805_pvalue": day_pvalue,
        "absolute_rho_change": abs(all_rho) - abs(day_rho),
        "direction_consistent": bool(
            np.isfinite(all_rho)
            and np.isfinite(day_rho)
            and (all_rho == 0.0 or day_rho == 0.0 or np.sign(all_rho) == np.sign(day_rho))
        ),
    }


def timer_max_count(log_text: str, label: str) -> int:
    """Return the largest cumulative count printed for an OKVIS timer."""
    pattern = re.compile(re.escape(label) + r"\s+(\d+)(?:\s|$)")
    values = [int(match.group(1)) for match in pattern.finditer(log_text)]
    return max(values, default=0)


def collect_failure_chain_run_rows(specs: list[SequenceSpec]) -> list[dict]:
    rows = []
    for spec in specs:
        for run_dir in spec.run_dirs:
            topology = repeatability.map_topology_summary(run_dir)
            logs = sorted(run_dir.glob("*.log"))
            if len(logs) > 1:
                raise ValueError(
                    f"expected at most one run log in {run_dir}, found {len(logs)}"
                )
            log_text = (
                logs[0].read_text(encoding="utf-8", errors="replace")
                if logs
                else ""
            )
            rows.append(
                {
                    "day": spec.day,
                    "sequence": spec.sequence,
                    "run": run_dir.name,
                    "run_log_available": bool(logs),
                    "landmarks": topology["landmarks"],
                    "observations_per_landmark": topology[
                        "observations_per_landmark"
                    ],
                    "distinct_states_per_landmark_mean": topology[
                        "distinct_states_per_landmark_mean"
                    ],
                    "landmark_time_span_median_s": topology[
                        "landmark_time_span_median_s"
                    ],
                    "ransac_large_reprojection_count": log_text.count(
                        "large reprojection error"
                    ),
                    "ransac_fail_count": log_text.count("RANSAC FAIL"),
                    "uninitialised_landmark_ransac_count": log_text.count(
                        "Running RANSAC also with uninitialised landmarks"
                    ),
                    "loop_descriptor_match_count": timer_max_count(
                        log_text, "2.04 loop closure descriptor matching"
                    ),
                    "loop_attempt_count": timer_max_count(
                        log_text, LOOP_ATTEMPT_TIMER
                    ),
                    "loop_accepted_count": timer_max_count(
                        log_text, LOOP_ACCEPTED_TIMER
                    ),
                    "dropped_camera_correspondence_count": log_text.count(
                        "without correspondence -- dropping"
                    ),
                }
            )
    return rows


def summarize_failure_chain_runs(
    sequence_rows: list[dict], run_rows: list[dict]
) -> list[dict]:
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[row["sequence"]].append(row)
    output = []
    for sequence_row in sequence_rows:
        sequence = sequence_row["sequence"]
        runs = grouped.get(sequence, [])
        if not runs:
            raise ValueError(f"{sequence}: no failure-chain run metrics")

        def median(field: str) -> float:
            return float(np.median([float(row[field]) for row in runs]))

        attempts = median("loop_attempt_count")
        accepted = median("loop_accepted_count")
        duration_s = float(sequence_row["analysis_duration_s"])
        output.append(
            {
                "day": sequence_row["day"],
                "sequence": sequence,
                "corrected_ape_median_mm": float(
                    sequence_row["corrected_ape_median_mm"]
                ),
                "analysis_duration_s": duration_s,
                "landmark_time_span_median_s": median(
                    "landmark_time_span_median_s"
                ),
                "observations_per_landmark": median(
                    "observations_per_landmark"
                ),
                "distinct_states_per_landmark_mean": median(
                    "distinct_states_per_landmark_mean"
                ),
                "ransac_large_reprojection_count": median(
                    "ransac_large_reprojection_count"
                ),
                "ransac_fail_count": median("ransac_fail_count"),
                "uninitialised_landmark_ransac_count": median(
                    "uninitialised_landmark_ransac_count"
                ),
                "loop_descriptor_match_count": median(
                    "loop_descriptor_match_count"
                ),
                "loop_attempt_count": attempts,
                "loop_accepted_count": accepted,
                "loop_attempts_per_min": (
                    attempts / (duration_s / 60.0) if duration_s > 0.0 else float("nan")
                ),
                "loop_rejection_fraction": (
                    1.0 - accepted / attempts if attempts > 0.0 else 0.0
                ),
                "dropped_camera_correspondence_count": median(
                    "dropped_camera_correspondence_count"
                ),
            }
        )
    return output


def world_displacement_error_series(
    evaluation: repeatability.AlignedEvaluation,
    *,
    delta_s: float = FAILURE_CHAIN_DELTA_S,
    tolerance_s: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare world-frame displacement increments over a fixed time delta."""
    if delta_s <= 0.0:
        raise ValueError("delta_s must be positive")
    timestamps = np.asarray(evaluation.timestamps, dtype=float)
    following = np.searchsorted(timestamps, timestamps + delta_s)
    valid = following < len(timestamps)
    first = np.flatnonzero(valid)
    following = following[valid]
    close = np.abs(
        timestamps[following] - timestamps[first] - delta_s
    ) <= tolerance_s
    first = first[close]
    following = following[close]
    reference_delta = (
        evaluation.reference_positions[following]
        - evaluation.reference_positions[first]
    )
    estimate_delta = (
        evaluation.estimate_positions[following]
        - evaluation.estimate_positions[first]
    )
    errors = np.linalg.norm(estimate_delta - reference_delta, axis=1)
    return timestamps[first] - timestamps[0], errors


def target_stage_rows(spec: SequenceSpec) -> list[dict]:
    reference = repeatability.load_mocap_trajectory(spec.mocap)
    rows = []
    for run_dir in spec.run_dirs:
        for stage, filename in repeatability.STAGE_FILES.items():
            evaluation = repeatability.evaluate_ape(
                reference,
                repeatability.load_okvis_trajectory(run_dir / filename),
            )
            rows.append(
                {
                    "sequence": spec.sequence,
                    "run": run_dir.name,
                    "stage": stage,
                    "associated_poses": len(evaluation.timestamps),
                    "ape_rmse_mm": float(
                        np.sqrt(np.mean(evaluation.errors**2)) * 1000.0
                    ),
                }
            )
    return rows


def _drop_window_before_first_ransac_failure(
    log_text: str, evaluation_start_s: float, image_delay_s: float
) -> tuple[float, float] | None:
    lines = log_text.splitlines()
    failure_index = next(
        (index for index, line in enumerate(lines) if "RANSAC FAIL" in line),
        None,
    )
    if failure_index is None:
        return None
    timestamps_ns = []
    pattern = re.compile(r"image at t=(\d+) without correspondence -- dropping")
    for line in lines[max(0, failure_index - 32) : failure_index]:
        match = pattern.search(line)
        if match:
            timestamps_ns.append(int(match.group(1)))
    if not timestamps_ns:
        return None
    elapsed = np.asarray(timestamps_ns, dtype=float) / 1e9
    elapsed -= image_delay_s + evaluation_start_s
    return float(np.min(elapsed)), float(np.max(elapsed))


def target_timeline_evidence(
    spec: SequenceSpec,
    *,
    start_s: float = FAILURE_CHAIN_TIMELINE_S[0],
    end_s: float = FAILURE_CHAIN_TIMELINE_S[1],
    delta_s: float = FAILURE_CHAIN_DELTA_S,
    image_delay_s: float,
) -> tuple[list[dict], list[dict], tuple[float, float] | None]:
    reference = repeatability.load_mocap_trajectory(spec.mocap)
    run_series = {}
    evaluation_start_s = None
    drop_windows = []
    for run_dir in spec.run_dirs:
        evaluation = repeatability.evaluate_ape(
            reference,
            repeatability.load_okvis_trajectory(
                run_dir / repeatability.STAGE_FILES["online"]
            ),
        )
        elapsed, errors = world_displacement_error_series(
            evaluation, delta_s=delta_s
        )
        run_series[run_dir.name] = (elapsed, errors * 1000.0)
        if evaluation_start_s is None:
            evaluation_start_s = float(evaluation.timestamps[0])
        logs = sorted(run_dir.glob("*.log"))
        if logs:
            window = _drop_window_before_first_ransac_failure(
                logs[0].read_text(encoding="utf-8", errors="replace"),
                float(evaluation.timestamps[0]),
                image_delay_s,
            )
            if window is not None:
                drop_windows.append(window)
    if evaluation_start_s is None or not run_series:
        raise ValueError(f"{spec.sequence}: no target timeline data")

    base_run = sorted(run_series)[0]
    base_elapsed = run_series[base_run][0]
    mask = (base_elapsed >= start_s) & (base_elapsed <= end_s - delta_s)
    timeline_elapsed = base_elapsed[mask]
    motion_timestamps = 0.5 * (
        reference.timestamps[:-1] + reference.timestamps[1:]
    )
    angular_speed = repeatability.angular_speed(
        reference.timestamps, reference.quaternions_wxyz
    )
    angular_values = np.interp(
        evaluation_start_s + timeline_elapsed,
        motion_timestamps,
        angular_speed,
    )
    rows = []
    for index, elapsed_s in enumerate(timeline_elapsed):
        row = {
            "sequence": spec.sequence,
            "elapsed_s": float(elapsed_s),
            "angular_speed_radps": float(angular_values[index]),
        }
        for run, (elapsed, errors_mm) in run_series.items():
            row[f"{run}_5s_displacement_error_mm"] = float(
                np.interp(elapsed_s, elapsed, errors_mm)
            )
        rows.append(row)

    gap_rows = [
        row
        for row in repeatability.camera_gap_events(
            spec.dataset,
            spec.sequence,
            evaluation_start_s,
            image_delay_s,
        )
        if start_s <= float(row["elapsed_s"]) <= end_s
    ]
    failure_window = (
        (
            min(window[0] for window in drop_windows),
            max(window[1] for window in drop_windows),
        )
        if drop_windows
        else None
    )
    return rows, gap_rows, failure_window


def alarm_classification_summary(
    sequence_rows: list[dict],
    alarm_rows: list[dict],
    *,
    source: str,
    threshold_radps: float,
    ape_threshold_mm: float = APE_THRESHOLD_MM,
    maximum_acceptable_events_per_5min: float = (
        MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN
    ),
) -> dict:
    candidates = {
        row["sequence"]: row
        for row in alarm_rows
        if row["source"] == source
        and np.isclose(float(row["threshold_radps"]), threshold_radps)
    }
    if len(candidates) != len(sequence_rows):
        raise ValueError(
            f"expected {len(sequence_rows)} {source} alarm rows at "
            f"{threshold_radps:g} rad/s, found {len(candidates)}"
        )
    counts = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    for row in sequence_rows:
        inaccurate = float(row["corrected_ape_median_mm"]) > ape_threshold_mm
        rejected = (
            float(candidates[row["sequence"]]["events_per_5min"])
            > maximum_acceptable_events_per_5min
        )
        if inaccurate and rejected:
            counts["true_positive"] += 1
        elif not inaccurate and not rejected:
            counts["true_negative"] += 1
        elif not inaccurate and rejected:
            counts["false_positive"] += 1
        else:
            counts["false_negative"] += 1
    return {
        "source": source,
        "threshold_radps": float(threshold_radps),
        "ape_threshold_mm": float(ape_threshold_mm),
        "maximum_acceptable_events_per_5min": float(
            maximum_acceptable_events_per_5min
        ),
        **counts,
    }


def _threshold_tradeoff_rows(
    sequence_rows: list[dict], alarm_rows: list[dict]
) -> list[dict]:
    ape_by_sequence = {
        row["sequence"]: float(row["corrected_ape_median_mm"])
        for row in sequence_rows
    }
    output = []
    for source in sorted({row["source"] for row in alarm_rows}):
        source_rows = [row for row in alarm_rows if row["source"] == source]
        for threshold in sorted(
            {float(row["threshold_radps"]) for row in source_rows}
        ):
            subset = [
                row
                for row in source_rows
                if np.isclose(float(row["threshold_radps"]), threshold)
            ]
            safe = [
                float(row["events_per_5min"])
                for row in subset
                if ape_by_sequence[row["sequence"]] <= APE_THRESHOLD_MM
            ]
            inaccurate = [
                float(row["events_per_5min"])
                for row in subset
                if ape_by_sequence[row["sequence"]] > APE_THRESHOLD_MM
            ]
            if not safe or not inaccurate:
                raise ValueError("alarm sweep requires both APE classes")
            false_positive = sum(
                value > MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN for value in safe
            )
            false_negative = sum(
                value <= MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN
                for value in inaccurate
            )
            output.append(
                {
                    "source": source,
                    "threshold_radps": threshold,
                    "maximum_safe_events_per_5min": max(safe),
                    "minimum_inaccurate_events_per_5min": min(inaccurate),
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "strict_separation": bool(
                        max(safe) <= MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN
                        and min(inaccurate)
                        > MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN
                    ),
                    "no_false_negative": bool(
                        min(inaccurate) > MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN
                    ),
                }
            )
    return output


def common_threshold_ranges(tradeoff_rows: list[dict]) -> dict:
    sources = sorted({row["source"] for row in tradeoff_rows})
    threshold_sets = [
        {float(row["threshold_radps"]) for row in tradeoff_rows if row["source"] == source}
        for source in sources
    ]
    thresholds = sorted(set.intersection(*threshold_sets))
    one_way = []
    strict = []
    for threshold in thresholds:
        subset = [
            row
            for row in tradeoff_rows
            if np.isclose(float(row["threshold_radps"]), threshold)
        ]
        if all(bool(row["no_false_negative"]) for row in subset):
            one_way.append(threshold)
        if all(bool(row["strict_separation"]) for row in subset):
            strict.append(threshold)
    return {
        "one_way_safe_min_radps": min(one_way) if one_way else float("nan"),
        "one_way_safe_max_radps": max(one_way) if one_way else float("nan"),
        "strict_min_radps": min(strict) if strict else float("nan"),
        "strict_max_radps": max(strict) if strict else float("nan"),
        "strict_range_exists": bool(strict),
    }


def analyze_sequences(
    specs: list[SequenceSpec],
    *,
    alarm_thresholds_radps: np.ndarray | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not specs:
        raise ValueError("no sequences found")
    sequence_rows = []
    run_rows = []
    contexts = {}
    for index, spec in enumerate(specs, 1):
        reference = repeatability.load_mocap_trajectory(spec.mocap)
        raw_rmse_mm = []
        corrected_rmse_mm = []
        first_evaluation = None
        lever = day_analysis.session_fixed_lever(
            spec.sequence, day_analysis.FIXED_DIAGNOSTIC_LEVER_M
        )
        corrected_reference_positions = day_analysis.correct_reference_positions(
            reference.positions,
            reference.quaternions_wxyz,
            lever,
        )
        motion_reference = repeatability.Trajectory(
            reference.timestamps,
            corrected_reference_positions,
            reference.quaternions_wxyz,
            reference.velocities,
        )
        for run_dir in spec.run_dirs:
            trajectory = repeatability.load_okvis_trajectory(
                run_dir / FINAL_BA_FILE
            )
            evaluation = repeatability.evaluate_ape(
                reference, trajectory, max_diff=0.01
            )
            if first_evaluation is None:
                first_evaluation = evaluation
            raw_rmse_m = float(np.sqrt(np.mean(evaluation.errors**2)))
            corrected = day_analysis.apply_effective_lever(
                evaluation.reference_positions,
                evaluation.reference_quaternions_wxyz,
                evaluation.estimate_positions,
                lever,
            )
            raw_rmse_mm.append(raw_rmse_m * 1000.0)
            corrected_rmse_mm.append(corrected.rmse_m * 1000.0)
            run_rows.append(
                {
                    "day": spec.day,
                    "group": spec.group,
                    "sequence": spec.sequence,
                    "run": run_dir.name,
                    "associated_poses": len(evaluation.timestamps),
                    "raw_ape_rmse_mm": raw_rmse_m * 1000.0,
                    "corrected_ape_rmse_mm": corrected.rmse_m * 1000.0,
                    "fixed_lever_applied": bool(np.linalg.norm(lever) > 0.0),
                }
            )
        if first_evaluation is None:
            raise ValueError(f"{spec.sequence}: no final-BA evaluations")
        start = float(first_evaluation.timestamps[0])
        end = float(first_evaluation.timestamps[-1])
        motion = repeatability._motion_from_mocap(motion_reference, start, end)
        motion_summary = repeatability.summarize_motion(motion)
        pose_mask = (reference.timestamps >= start) & (reference.timestamps <= end)
        pose_times = reference.timestamps[pose_mask]
        pose_positions = corrected_reference_positions[pose_mask]
        pose_quaternions = reference.quaternions_wxyz[pose_mask]
        orientation = orientation_excitation_metrics(
            pose_times, pose_quaternions
        )
        translation_path_m = repeatability.path_length(pose_positions)
        imu = repeatability.load_imu(spec.dataset / "imu0/data.csv")
        imu_summary, imu_timeseries = repeatability.analyze_imu(imu, start, end)
        row = {
            "day": spec.day,
            "group": spec.group,
            "sequence": spec.sequence,
            "dataset": str(spec.dataset),
            "run_count": len(spec.run_dirs),
            "analysis_start_s": start,
            "analysis_end_s": end,
            "analysis_duration_s": end - start,
            "translation_path_m": translation_path_m,
            "translation_per_orientation_m_per_rad": (
                translation_path_m / orientation["orientation_path_rad"]
                if orientation["orientation_path_rad"] > 0.0
                else float("nan")
            ),
            **orientation,
            **{f"motion_{key}": value for key, value in motion_summary.items()},
            **{f"imu_{key}": value for key, value in imu_summary.items()},
            "raw_ape_run1_mm": raw_rmse_mm[0],
            "raw_ape_run2_mm": raw_rmse_mm[1] if len(raw_rmse_mm) > 1 else float("nan"),
            "raw_ape_median_mm": float(np.median(raw_rmse_mm)),
            "corrected_ape_run1_mm": corrected_rmse_mm[0],
            "corrected_ape_run2_mm": (
                corrected_rmse_mm[1]
                if len(corrected_rmse_mm) > 1
                else float("nan")
            ),
            "corrected_ape_median_mm": float(np.median(corrected_rmse_mm)),
            "fixed_lever_applied": bool(np.linalg.norm(lever) > 0.0),
            "fixed_lever_x_m": float(lever[0]),
            "fixed_lever_y_m": float(lever[1]),
            "fixed_lever_z_m": float(lever[2]),
        }
        # Keep compatibility with the single-day alarm helper.
        row["fixed_lever_ape_median_mm"] = row["corrected_ape_median_mm"]
        sequence_rows.append(row)
        contexts[spec.sequence] = {
            "motion": motion,
            "imu_timeseries": imu_timeseries,
        }
        print(
            f"[{index:02d}/{len(specs)}] {spec.sequence}: "
            f"APE {row['corrected_ape_median_mm']:.3f} mm, "
            f"mocap p95 {row['motion_angular_speed_radps_p95']:.3f} rad/s"
        )

    alarm_rows = alarm_threshold_rows(
        sequence_rows,
        contexts,
        thresholds_radps=alarm_thresholds_radps,
    )
    day_by_sequence = {row["sequence"]: row["day"] for row in sequence_rows}
    for row in alarm_rows:
        row["day"] = day_by_sequence[row["sequence"]]
    return sequence_rows, run_rows, alarm_rows


def _plot_by_day(axis: plt.Axes, rows: list[dict], x: np.ndarray, y: np.ndarray) -> None:
    for day in sorted({row["day"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["day"] == day]
        axis.scatter(
            x[indices],
            y[indices],
            color=DAY_COLORS.get(day, "#555555"),
            marker=DAY_MARKERS.get(day, "o"),
            s=64,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )


def _annotate_sequences(
    axis: plt.Axes, rows: list[dict], x: np.ndarray, y: np.ndarray
) -> None:
    for index, row in enumerate(rows):
        axis.annotate(
            row["sequence"][-6:],
            (x[index], y[index]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=DAY_COLORS.get(row["day"], "#444444"),
        )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_failure_chain_evidence(
    path: Path,
    sequence_rows: list[dict],
    stage_rows: list[dict],
    timeline_rows: list[dict],
    *,
    camera_gap_elapsed_s: list[float],
    failure_drop_window_s: tuple[float, float] | None,
    target_sequence: str = FAILURE_CHAIN_TARGET,
) -> None:
    if not sequence_rows or not stage_rows or not timeline_rows:
        raise ValueError("failure-chain plot inputs must not be empty")
    targets = [row for row in sequence_rows if row["sequence"] == target_sequence]
    if len(targets) != 1:
        raise ValueError(
            f"expected one target row for {target_sequence}, found {len(targets)}"
        )

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(17, 12))
    target_color = "#c62828"
    control_color = "#7a7f87"

    ape = np.asarray(
        [row["corrected_ape_median_mm"] for row in sequence_rows], dtype=float
    )
    spans = np.asarray(
        [row["landmark_time_span_median_s"] for row in sequence_rows], dtype=float
    )
    ransac_fail = np.asarray(
        [row["ransac_fail_count"] for row in sequence_rows], dtype=float
    )
    target_ids = np.asarray(
        [row["sequence"] == target_sequence for row in sequence_rows], dtype=bool
    )
    color_values = np.log10(1.0 + ransac_fail)
    norm = Normalize(
        vmin=float(np.min(color_values)),
        vmax=float(max(np.max(color_values), np.min(color_values) + 1e-6)),
    )

    axis = axes[0, 0]
    scatter = axis.scatter(
        spans[~target_ids],
        ape[~target_ids],
        c=color_values[~target_ids],
        cmap="viridis",
        norm=norm,
        s=75,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    axis.scatter(
        spans[target_ids],
        ape[target_ids],
        c=color_values[target_ids],
        cmap="viridis",
        norm=norm,
        marker="*",
        s=280,
        edgecolors=target_color,
        linewidths=1.8,
        zorder=5,
    )
    axis.axhline(APE_THRESHOLD_MM, color=target_color, linestyle="--", linewidth=1.1)
    axis.axvline(1.02, color="#ef6c00", linestyle="--", linewidth=1.1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="Landmark 中位持续时间 [s, log]",
        ylabel="APE RMSE [mm, log]",
        title="A. 高 APE 与 landmark 短寿命形成同一失效簇",
    )
    axis.grid(alpha=0.24, which="both")
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("log10(1 + RANSAC FAIL 次数)")

    attempts_per_min = np.asarray(
        [row["loop_attempts_per_min"] for row in sequence_rows], dtype=float
    )
    rejection_percent = np.asarray(
        [100.0 * row["loop_rejection_fraction"] for row in sequence_rows],
        dtype=float,
    )
    axis = axes[0, 1]
    rejection_norm = Normalize(vmin=0.0, vmax=100.0)
    loop_scatter = axis.scatter(
        attempts_per_min[~target_ids],
        ape[~target_ids],
        c=rejection_percent[~target_ids],
        cmap="plasma",
        norm=rejection_norm,
        s=75,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    axis.scatter(
        attempts_per_min[target_ids],
        ape[target_ids],
        c=rejection_percent[target_ids],
        cmap="plasma",
        norm=rejection_norm,
        marker="*",
        s=280,
        edgecolors=target_color,
        linewidths=1.8,
        zorder=5,
    )
    axis.axhline(APE_THRESHOLD_MM, color=target_color, linestyle="--", linewidth=1.1)
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_yscale("log")
    axis.set_xlim(
        left=0.0,
        right=max(1.0, float(np.max(attempts_per_min)) * 2.2),
    )
    axis.set(
        xlabel="回环尝试次数 / min [symlog]",
        ylabel="APE RMSE [mm, log]",
        title="B. 跟踪碎片化后，回环候选与拒绝率同时爆炸",
    )
    axis.grid(alpha=0.24, which="both")
    colorbar = figure.colorbar(loop_scatter, ax=axis, pad=0.01)
    colorbar.set_label("回环尝试拒绝率 [%]")

    for axis_to_annotate, x_values in (
        (axes[0, 0], spans),
        (axes[0, 1], attempts_per_min),
    ):
        for index, row in enumerate(sequence_rows):
            if ape[index] <= APE_THRESHOLD_MM and row["sequence"] != target_sequence:
                continue
            right_aligned = x_values[index] >= float(
                np.percentile(x_values, 90)
            )
            axis_to_annotate.annotate(
                row["sequence"][-6:],
                (x_values[index], ape[index]),
                xytext=((-5 if right_aligned else 5), 4),
                textcoords="offset points",
                fontsize=7.5,
                ha=("right" if right_aligned else "left"),
                color=(
                    target_color
                    if row["sequence"] == target_sequence
                    else "#333333"
                ),
            )

    axis = axes[1, 0]
    stage_order = list(repeatability.STAGE_FILES)
    stage_labels = {"online": "online", "final": "final", "final-ba": "final-BA"}
    for run, color, marker in (
        ("run1", "#1565c0", "o"),
        ("run2", "#ef6c00", "s"),
    ):
        values = {
            row["stage"]: float(row["ape_rmse_mm"])
            for row in stage_rows
            if row["run"] == run
        }
        if set(values) != set(stage_order):
            raise ValueError(f"{run}: incomplete stage metrics")
        axis.plot(
            np.arange(len(stage_order)),
            [values[stage] for stage in stage_order],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=7,
            label=run,
        )
    axis.axhline(APE_THRESHOLD_MM, color=target_color, linestyle="--", linewidth=1.1)
    axis.set_xticks(
        np.arange(len(stage_order)),
        [stage_labels[stage] for stage in stage_order],
    )
    axis.set(
        ylabel="APE RMSE [mm]",
        title="C. 后端继续改变误差，但 online 阶段已经异常",
    )
    axis.grid(alpha=0.24)
    axis.legend(loc="best")

    axis = axes[1, 1]
    elapsed = np.asarray([row["elapsed_s"] for row in timeline_rows], dtype=float)
    for run, color in (("run1", "#1565c0"), ("run2", "#ef6c00")):
        field = f"{run}_5s_displacement_error_mm"
        if field not in timeline_rows[0]:
            continue
        axis.plot(
            elapsed,
            [row[field] for row in timeline_rows],
            color=color,
            linewidth=1.6,
            label=f"{run} 5 s 位移增量误差",
        )
    for index, gap_s in enumerate(camera_gap_elapsed_s):
        axis.axvline(
            gap_s,
            color="#0288d1",
            alpha=0.35,
            linewidth=0.9,
            label="相机 66.67 ms gap" if index == 0 else None,
        )
    if failure_drop_window_s is not None:
        axis.axvspan(
            failure_drop_window_s[0],
            failure_drop_window_s[1],
            color="#d32f2f",
            alpha=0.16,
            label="首次 RANSAC FAIL 前的配对丢帧簇",
        )
    axis.set(
        xlabel="序列时间 [s]",
        ylabel="5 s 位移增量误差 [mm]",
        title="D. 局部配对中断、旋转激励与轨迹变形的时间关系",
    )
    axis.grid(alpha=0.22)
    motion_axis = axis.twinx()
    motion_axis.plot(
        elapsed,
        [row["angular_speed_radps"] for row in timeline_rows],
        color="#424242",
        linewidth=0.8,
        alpha=0.55,
        label="mocap 角速度",
    )
    motion_axis.set_ylabel("mocap 角速度 [rad/s]", color="#424242")
    handles, labels = axis.get_legend_handles_labels()
    motion_handles, motion_labels = motion_axis.get_legend_handles_labels()
    axis.legend(handles + motion_handles, labels + motion_labels, loc="upper left", fontsize=7.5)

    target = targets[0]
    figure.suptitle(
        f"{target_sequence} 视觉跟踪失效证据链\n"
        f"APE {target['corrected_ape_median_mm']:.1f} mm | "
        f"landmark 中位寿命 {target['landmark_time_span_median_s']:.3f} s | "
        f"RANSAC FAIL {target['ransac_fail_count']:.1f} | "
        f"回环拒绝率 {100.0 * target['loop_rejection_fraction']:.1f}%",
        fontsize=15,
    )
    _save_figure(figure, Path(path))


def plot_multiday_diagnostics(
    output: Path,
    sequence_rows: list[dict],
    alarm_rows: list[dict],
    *,
    candidate_threshold_radps: float = CANDIDATE_THRESHOLD_RADPS,
) -> None:
    output = Path(output)
    period_label = analysis_period_label(
        [row["day"] for row in sequence_rows]
    )
    day_handles = [
        Line2D(
            [0],
            [0],
            marker=DAY_MARKERS.get(day, "o"),
            color="none",
            markerfacecolor=DAY_COLORS.get(day, "#555555"),
            markeredgecolor="white",
            markersize=8,
            label=day,
        )
        for day in sorted({row["day"] for row in sequence_rows})
    ]
    ape = np.asarray(
        [row["corrected_ape_median_mm"] for row in sequence_rows], dtype=float
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, (field, xlabel) in zip(axes.ravel(), MOTION_METRICS):
        values = np.asarray([row[field] for row in sequence_rows], dtype=float)
        comparison = correlation_comparison(sequence_rows, field)
        _plot_by_day(axis, sequence_rows, values, ape)
        _annotate_sequences(axis, sequence_rows, values, ape)
        axis.axhline(APE_THRESHOLD_MM, color="#d93025", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set(
            xlabel=xlabel,
            ylabel="Corrected APE RMSE [mm, log]",
            title=(
                f"Spearman rho: 0805 {comparison['day_20260805_rho']:+.2f} "
                f"-> all {comparison['all_rho']:+.2f} "
                f"(p={comparison['all_pvalue']:.3g})"
            ),
        )
        axis.margins(x=0.09)
        axis.grid(alpha=0.26, which="both")
    axes[0, 0].legend(handles=day_handles, loc="best", fontsize=8)
    figure.suptitle(
        f"{period_label} motion excitation versus APE\n"
        "Confirmed mocap rigid-body configuration correction is applied only "
        "to the three affected 0805 noon sequences",
        fontsize=14,
    )
    _save_figure(figure, output / "01_multiday_motion_excitation_vs_ape.png")

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, source in zip(axes, ("mocap", "imu")):
        candidate_rows = {
            row["sequence"]: row
            for row in alarm_rows
            if row["source"] == source
            and np.isclose(
                float(row["threshold_radps"]), candidate_threshold_radps
            )
        }
        counts = np.asarray(
            [candidate_rows[row["sequence"]]["events_per_5min"] for row in sequence_rows],
            dtype=float,
        )
        summary = alarm_classification_summary(
            sequence_rows,
            alarm_rows,
            source=source,
            threshold_radps=candidate_threshold_radps,
        )
        _plot_by_day(axis, sequence_rows, counts, ape)
        _annotate_sequences(axis, sequence_rows, counts, ape)
        axis.axvline(
            MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN,
            color="#d93025",
            linestyle="--",
            linewidth=1.0,
        )
        axis.axhline(APE_THRESHOLD_MM, color="#d93025", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set(
            xlabel="Debounced alarms per 5 min",
            title=(
                f"{source.upper()}: TP {summary['true_positive']}, "
                f"FN {summary['false_negative']}, "
                f"FP {summary['false_positive']}, TN {summary['true_negative']}"
            ),
        )
        axis.margins(x=0.10)
        axis.grid(alpha=0.26, which="both")
    axes[0].set_ylabel("Corrected APE RMSE [mm, log]")
    axes[0].legend(handles=day_handles, loc="best", fontsize=8)
    figure.suptitle(
        f"N={candidate_threshold_radps:.1f} rad/s alarm count versus APE "
        "(50 ms persistence, 250 ms merge gap)",
        fontsize=14,
    )
    _save_figure(figure, output / "02_multiday_alarm_count_vs_ape_at_3radps.png")

    tradeoff_rows = _threshold_tradeoff_rows(sequence_rows, alarm_rows)
    ranges = common_threshold_ranges(tradeoff_rows)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="col")
    for column, source in enumerate(("mocap", "imu")):
        rows = [row for row in tradeoff_rows if row["source"] == source]
        thresholds = np.asarray([row["threshold_radps"] for row in rows])
        max_safe = np.asarray([row["maximum_safe_events_per_5min"] for row in rows])
        min_bad = np.asarray([row["minimum_inaccurate_events_per_5min"] for row in rows])
        false_positive = np.asarray([row["false_positive"] for row in rows])
        false_negative = np.asarray([row["false_negative"] for row in rows])
        top = axes[0, column]
        top.plot(thresholds, max_safe, color="#147d64", label="max APE <= 10 mm")
        top.plot(thresholds, min_bad, color="#b3261e", label="min APE > 10 mm")
        top.axhline(
            MAXIMUM_ACCEPTABLE_EVENTS_PER_5MIN,
            color="#202124",
            linestyle="--",
            linewidth=1.0,
            label="3 alarms / 5 min",
        )
        top.axvline(candidate_threshold_radps, color="#777777", linestyle=":")
        top.set_yscale("symlog", linthresh=1.0)
        top.set_ylim(bottom=0.0)
        top.set(title=source.upper(), ylabel="Alarm envelope [events / 5 min]")
        top.grid(alpha=0.26, which="both")
        bottom = axes[1, column]
        bottom.plot(thresholds, false_positive, color="#d97706", label="false positive")
        bottom.plot(thresholds, false_negative, color="#7c3aed", label="false negative")
        bottom.axvline(candidate_threshold_radps, color="#777777", linestyle=":")
        bottom.set(
            xlabel="Alarm threshold N [rad/s]",
            ylabel="Misclassified sequences",
        )
        bottom.set_ylim(bottom=-0.15)
        bottom.grid(alpha=0.26)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[1, 0].legend(loc="best", fontsize=8)
    strict_text = (
        f"{ranges['strict_min_radps']:.2f}-{ranges['strict_max_radps']:.2f} rad/s"
        if ranges["strict_range_exists"]
        else "none"
    )
    figure.suptitle(
        f"{period_label} alarm-threshold trade-off\n"
        f"Common no-false-negative range: "
        f"{ranges['one_way_safe_min_radps']:.2f}-"
        f"{ranges['one_way_safe_max_radps']:.2f} rad/s; "
        f"strict perfect-separation range: {strict_text}",
        fontsize=14,
    )
    _save_figure(figure, output / "03_multiday_alarm_threshold_tradeoff.png")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    known = set()
    for row in rows:
        for field in row:
            if field not in known:
                fields.append(field)
                known.add(field)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", nargs="+", default=list(DEFAULT_DAYS))
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    specs = discover_sequences(
        arguments.results_root,
        arguments.data_root,
        days=tuple(arguments.days),
    )
    sequence_rows, run_rows, alarm_rows = analyze_sequences(specs)
    expected_sequences = (
        DEFAULT_EXPECTED_SEQUENCES
        if tuple(arguments.days) == DEFAULT_DAYS
        else None
    )
    if expected_sequences is not None and len(sequence_rows) != expected_sequences:
        raise ValueError(
            f"expected {expected_sequences} sequences, found {len(sequence_rows)}"
        )
    correlations = [
        correlation_comparison(sequence_rows, field) for field, _ in MOTION_METRICS
    ]
    tradeoff_rows = _threshold_tradeoff_rows(sequence_rows, alarm_rows)
    ranges = common_threshold_ranges(tradeoff_rows)
    classifications = [
        {
            **alarm_classification_summary(
                sequence_rows,
                alarm_rows,
                source=source,
                threshold_radps=CANDIDATE_THRESHOLD_RADPS,
            ),
            **ranges,
        }
        for source in ("mocap", "imu")
    ]
    failure_run_rows = collect_failure_chain_run_rows(specs)
    failure_sequence_rows = summarize_failure_chain_runs(
        sequence_rows, failure_run_rows
    )
    tables = arguments.output / "tables"
    figures = arguments.output / "figures"
    _write_csv(tables / "multiday_sequence_metrics.csv", sequence_rows)
    _write_csv(tables / "multiday_run_metrics.csv", run_rows)
    _write_csv(tables / "multiday_alarm_threshold_sweep.csv", alarm_rows)
    _write_csv(tables / "multiday_correlations.csv", correlations)
    _write_csv(tables / "multiday_alarm_tradeoff.csv", tradeoff_rows)
    _write_csv(
        tables / "multiday_alarm_classification_at_3radps.csv",
        classifications,
    )
    _write_csv(
        tables / "multiday_failure_chain_run_metrics.csv",
        failure_run_rows,
    )
    _write_csv(
        tables / "multiday_failure_chain_sequence_metrics.csv",
        failure_sequence_rows,
    )
    plot_multiday_diagnostics(
        figures,
        sequence_rows,
        alarm_rows,
        candidate_threshold_radps=CANDIDATE_THRESHOLD_RADPS,
    )
    target_specs = [
        spec for spec in specs if spec.sequence == FAILURE_CHAIN_TARGET
    ]
    if target_specs:
        if len(target_specs) != 1:
            raise ValueError(
                f"expected one {FAILURE_CHAIN_TARGET} spec, found {len(target_specs)}"
            )
        target_spec = target_specs[0]
        image_delay_s = repeatability.parse_image_delay(
            REPOSITORY / "config/okvis2_eucm_EGO2.yaml"
        )
        stage_rows = target_stage_rows(target_spec)
        timeline_rows, gap_rows, failure_window = target_timeline_evidence(
            target_spec,
            image_delay_s=image_delay_s,
        )
        _write_csv(
            tables / f"{FAILURE_CHAIN_TARGET}_failure_chain_stages.csv",
            stage_rows,
        )
        _write_csv(
            tables / f"{FAILURE_CHAIN_TARGET}_failure_chain_timeline.csv",
            timeline_rows,
        )
        _write_csv(
            tables / f"{FAILURE_CHAIN_TARGET}_failure_chain_camera_gaps.csv",
            gap_rows,
        )
        plot_failure_chain_evidence(
            figures / f"04_{FAILURE_CHAIN_TARGET}_failure_chain.png",
            failure_sequence_rows,
            stage_rows,
            timeline_rows,
            camera_gap_elapsed_s=sorted(
                {float(row["elapsed_s"]) for row in gap_rows}
            ),
            failure_drop_window_s=failure_window,
            target_sequence=FAILURE_CHAIN_TARGET,
        )
    print(f"Analysis outputs written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
