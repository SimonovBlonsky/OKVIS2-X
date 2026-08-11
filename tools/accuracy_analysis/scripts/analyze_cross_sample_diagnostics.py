#!/usr/bin/env python3
"""Build unified cross-sample diagnostics for the 20260803-20260806 EGO2 runs."""

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "okvis_cross_sample_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "okvis_cross_sample_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.accuracy_analysis.scripts import mocap_reference_correction as day_analysis
from tools.accuracy_analysis.scripts import analyze_multiday
from tools.accuracy_analysis.scripts import analyze_repeatability as repeatability
from tools.accuracy_analysis.scripts import analyze_vio_causal_diagnostics as causal_analysis
from tools.accuracy_analysis.scripts import refresh_population_causal_outputs as population_refresh
from tools.accuracy_analysis.scripts import generate_causal_chain_report as causal_report


IMPULSE_SEQUENCES = frozenset(
    {
        "20260806-175103",
        "20260806-175304",
        "20260806-175539",
    }
)
MOCAP_CORRECTED_SEQUENCES = frozenset(
    {
        "20260805-122310",
        "20260805-123231",
        "20260805-123752",
    }
)
APE_THRESHOLD_MM = 10.0
FRAGMENTATION_RANSAC_FAIL_PER_MIN = 15.0
FRAGMENTATION_LANDMARK_SPAN_S = 3.0
SCALE_IMPROVEMENT_THRESHOLD_PCT = 25.0
SCALE_DEVIATION_THRESHOLD = 0.10
SUPPORT_ORDER = {
    "currently_not_supported": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}
LOOP_DESCRIPTOR_TIMER = "2.04 loop closure descriptor matching"
LOOP_ATTEMPT_TIMER = "2.07 attempt loop closure"
LOOP_ACCEPTED_TIMER = "2.08 add loop closure"
TEXT_CSV_FIELDS = frozenset(
    {
        "day",
        "group",
        "sequence",
        "run",
        "stage",
        "source",
        "dataset",
        "camera",
        "factor",
        "label_zh",
        "metric",
        "role",
        "cohort",
        "support_level",
        "evidence_outcome",
        "stereo_pairs",
    }
)
CAUSAL_TABLES = (
    "causal_frame_metrics.csv",
    "causal_diagnostics_coverage.csv",
    "causal_event_metrics.csv",
    "impulse_mediator_recovery.csv",
    "causal_mediation_models.csv",
    "causal_hypothesis_evidence.csv",
    "population_angular_failure_chain.csv",
    "population_angular_event_paired_effects.csv",
)
CAUSAL_FIGURES = (
    "impulse_mediator_timeline.png",
    "angular_to_fragmentation_mediator_paths.png",
    "mediator_onset_recovery.png",
    "population_angular_failure_chain.png",
)
CAUSAL_REPORTS = (
    "causal_diagnostics_summary.md",
    "causal_mediator_evidence.md",
)
DEFAULT_FACTORS = (
    {
        "factor": "angular_speed_p95",
        "label_zh": "角速度 p95",
        "label_en": "Angular speed p95",
        "metric": "motion_angular_speed_radps_p95",
        "expected_direction": 1,
        "role": "candidate_trigger",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "accepted-feature parallax and exposure-time motion",
    },
    {
        "factor": "angular_speed_p99",
        "label_zh": "角速度 p99",
        "label_en": "Angular speed p99",
        "metric": "motion_angular_speed_radps_p99",
        "expected_direction": 1,
        "role": "candidate_trigger",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "accepted-feature parallax and exposure-time motion",
    },
    {
        "factor": "angular_speed_max",
        "label_zh": "最大角速度",
        "label_en": "Maximum angular speed",
        "metric": "motion_angular_speed_radps_max",
        "expected_direction": 1,
        "role": "candidate_trigger",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "event-aligned feature survival and innovation",
    },
    {
        "factor": "high_angular_fraction",
        "label_zh": "角速度大于 3 rad/s 时间比例",
        "label_en": "Time fraction above 3 rad/s",
        "metric": "motion_angular_above_3_0_fraction",
        "expected_direction": 1,
        "role": "candidate_trigger",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "event-aligned feature survival and innovation",
    },
    {
        "factor": "angular_event_frequency",
        "label_zh": "高角速度事件频率",
        "label_en": "High-angular event frequency",
        "metric": "angular_events_per_5min",
        "expected_direction": 1,
        "role": "candidate_trigger",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "event-aligned feature survival and innovation",
    },
    {
        "factor": "translation_per_orientation",
        "label_zh": "单位转角平移路径",
        "label_en": "Translation per orientation",
        "metric": "translation_per_orientation_m_per_rad",
        "expected_direction": -1,
        "role": "candidate_trigger",
        "is_proxy": True,
        "evidence_outcome": "ape",
        "measurement_needed": "camera-ray parallax of accepted temporal matches",
    },
    {
        "factor": "high_rotation_low_translation",
        "label_zh": "高旋转低平移暴露",
        "label_en": "High-rotation/low-translation exposure",
        "metric": "frac_rotation_gt_0p25_baseline_lt_5cm_pct",
        "expected_direction": 1,
        "role": "candidate_trigger",
        "is_proxy": True,
        "evidence_outcome": "ape",
        "measurement_needed": "camera-ray parallax and isParallel timing",
    },
    {
        "factor": "image_edge_content_p5",
        "label_zh": "图像边缘/清晰度代理 p5",
        "label_en": "Image edge-content proxy p5",
        "metric": "laplacian_variance_p5",
        "expected_direction": -1,
        "role": "candidate_trigger",
        "is_proxy": True,
        "evidence_outcome": "ape",
        "measurement_needed": "separate optical blur, scene texture and exposure",
    },
    {
        "factor": "image_contrast",
        "label_zh": "图像强度标准差",
        "label_en": "Image intensity standard deviation",
        "metric": "intensity_std_median",
        "expected_direction": -1,
        "role": "candidate_trigger",
        "is_proxy": True,
        "evidence_outcome": "ape",
        "measurement_needed": "accepted-feature count and spatial distribution",
    },
    {
        "factor": "ransac_fail_rate",
        "label_zh": "GP3P RANSAC FAIL 率",
        "label_en": "GP3P RANSAC FAIL rate",
        "metric": "ransac_fail_per_min",
        "expected_direction": 1,
        "role": "downstream_state",
        "is_proxy": False,
        "evidence_outcome": "visual_fragmentation",
        "measurement_needed": "per-frame inlier and innovation logs for causal timing",
    },
    {
        "factor": "loop_attempt_rate",
        "label_zh": "回环尝试率",
        "label_en": "Loop attempt rate",
        "metric": "loop_attempts_per_min",
        "expected_direction": 1,
        "role": "downstream_state",
        "is_proxy": False,
        "evidence_outcome": "visual_fragmentation",
        "measurement_needed": "map-fragment identity and candidate provenance",
    },
    {
        "factor": "landmark_span",
        "label_zh": "Landmark 中位存活时间",
        "label_en": "Landmark median span",
        "metric": "landmark_time_span_median_s",
        "expected_direction": -1,
        "role": "downstream_state",
        "is_proxy": False,
        "evidence_outcome": "visual_fragmentation",
        "measurement_needed": "online track birth/death reasons",
    },
    {
        "factor": "observations_per_landmark",
        "label_zh": "每 Landmark 观测数",
        "label_en": "Observations per landmark",
        "metric": "observations_per_landmark",
        "expected_direction": -1,
        "role": "downstream_state",
        "is_proxy": False,
        "evidence_outcome": "visual_fragmentation",
        "measurement_needed": "online observation rejection reasons",
    },
    {
        "factor": "initialized_quality_fraction",
        "label_zh": "初始化质量通过比例",
        "label_en": "Initialized-quality fraction",
        "metric": "quality_initialized_fraction",
        "expected_direction": -1,
        "role": "downstream_state",
        "is_proxy": False,
        "evidence_outcome": "visual_fragmentation",
        "measurement_needed": "quality evolution at landmark creation time",
    },
    {
        "factor": "stereo_landmark_fraction",
        "label_zh": "立体 Landmark 比例",
        "label_en": "Stereo-landmark fraction",
        "metric": "stereo_landmark_fraction",
        "expected_direction": -1,
        "role": "candidate_trigger",
        "is_proxy": True,
        "evidence_outcome": "ape",
        "measurement_needed": "local stereo support at failure onset",
    },
    {
        "factor": "camera_missing_images",
        "label_zh": "索引图像缺失数",
        "label_en": "Missing indexed images",
        "metric": "camera_missing_images",
        "expected_direction": 1,
        "role": "candidate_trigger",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "none when complete coverage is verified",
    },
    {
        "factor": "mocap_tracking_fraction",
        "label_zh": "动捕跟踪比例",
        "label_en": "Mocap tracked fraction",
        "metric": "mocap_tracked_fraction",
        "expected_direction": -1,
        "role": "evaluation_integrity",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "none when tracking and interval integrity are complete",
    },
    {
        "factor": "sim3_improvement",
        "label_zh": "Sim(3) 相对改善",
        "label_en": "Sim(3) APE improvement",
        "metric": "sim3_improvement_pct",
        "expected_direction": 1,
        "role": "downstream_state",
        "is_proxy": False,
        "evidence_outcome": "ape",
        "measurement_needed": "time-varying local scale state",
    },
)


def validate_unique_keys(
    rows: list[dict], fields: tuple[str, ...], *, expected: int | None = None
) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate keys for {fields}: {duplicates}")
    if expected is not None and len(rows) != expected:
        raise ValueError(
            f"expected {expected} rows for {fields}, found {len(rows)}"
        )


def _coerce_csv_value(field: str, value: str):
    if field in TEXT_CSV_FIELDS:
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return ""
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def read_csv_rows(path: Path) -> list[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = [
            {field: _coerce_csv_value(field, value) for field, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path = Path(path)
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


def strict_join(
    left_rows: list[dict],
    right_rows: list[dict],
    keys: tuple[str, ...],
    *,
    right_prefix: str = "",
    required: bool = True,
) -> list[dict]:
    validate_unique_keys(left_rows, keys)
    validate_unique_keys(right_rows, keys)
    right_by_key = {
        tuple(row[key] for key in keys): row for row in right_rows
    }
    left_keys = {tuple(row[key] for key in keys) for row in left_rows}
    right_keys = set(right_by_key)
    missing = sorted(left_keys - right_keys)
    unexpected = sorted(right_keys - left_keys)
    if required and missing:
        raise ValueError(f"missing right rows for {keys}: {missing}")
    if required and unexpected:
        raise ValueError(f"unexpected right rows for {keys}: {unexpected}")
    output = []
    for left in left_rows:
        key = tuple(left[field] for field in keys)
        joined = dict(left)
        right = right_by_key.get(key)
        if right is None:
            joined[f"{right_prefix}available"] = False
            output.append(joined)
            continue
        for field, value in right.items():
            if field in keys:
                continue
            output_field = f"{right_prefix}{field}"
            if output_field in joined and joined[output_field] != value:
                raise ValueError(
                    f"conflicting field {output_field} for {keys}={key}"
                )
            joined[output_field] = value
        if not required:
            joined[f"{right_prefix}available"] = True
        output.append(joined)
    return output


def _timer_max_count(log_text: str, label: str) -> int:
    pattern = re.compile(re.escape(label) + r"\s+(\d+)(?:\s|$)")
    return max((int(match.group(1)) for match in pattern.finditer(log_text)), default=0)


def collect_run_diagnostics(
    specs: list, duration_by_sequence: dict[str, float]
) -> list[dict]:
    rows = []
    for spec in specs:
        if spec.sequence not in duration_by_sequence:
            raise ValueError(f"{spec.sequence}: missing evaluation duration")
        duration_s = float(duration_by_sequence[spec.sequence])
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(f"{spec.sequence}: invalid evaluation duration {duration_s}")
        duration_min = duration_s / 60.0
        for run_dir_value in spec.run_dirs:
            run_dir = Path(run_dir_value)
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
            ransac_fail_count = log_text.count("RANSAC FAIL")
            reprojection_count = log_text.count("large reprojection error")
            uninitialised_count = log_text.count(
                "Running RANSAC also with uninitialised landmarks"
            )
            descriptor_matches = _timer_max_count(
                log_text, LOOP_DESCRIPTOR_TIMER
            )
            loop_attempts = _timer_max_count(log_text, LOOP_ATTEMPT_TIMER)
            loop_accepted = _timer_max_count(log_text, LOOP_ACCEPTED_TIMER)
            rows.append(
                {
                    "day": spec.day,
                    "sequence": spec.sequence,
                    "run": run_dir.name,
                    "analysis_duration_s": duration_s,
                    "run_log_available": bool(logs),
                    **topology,
                    "landmark_creation_per_min": topology["landmarks"]
                    / duration_min,
                    "ransac_fail_count": ransac_fail_count,
                    "ransac_fail_per_min": ransac_fail_count / duration_min,
                    "ransac_large_reprojection_count": reprojection_count,
                    "ransac_large_reprojection_per_min": reprojection_count
                    / duration_min,
                    "uninitialised_landmark_ransac_count": uninitialised_count,
                    "uninitialised_landmark_ransac_per_min": uninitialised_count
                    / duration_min,
                    "loop_descriptor_match_count": descriptor_matches,
                    "loop_attempt_count": loop_attempts,
                    "loop_accepted_count": loop_accepted,
                    "loop_attempts_per_min": loop_attempts / duration_min,
                    "loop_rejection_fraction": (
                        1.0 - loop_accepted / loop_attempts
                        if loop_attempts
                        else 0.0
                    ),
                    "dropped_camera_correspondence_count": log_text.count(
                        "without correspondence -- dropping"
                    ),
                }
            )
    return sorted(rows, key=lambda row: (row["sequence"], row["run"]))


def similarity_align_and_errors(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 3
    ):
        raise ValueError("source and target must be matching N x 3 arrays")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target must contain finite values")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0.0:
        reflection[-1] = -1.0
    rotation = vt.T @ np.diag(reflection) @ u.T
    denominator = float(np.sum(source_zero**2))
    if denominator <= 0.0:
        raise ValueError("source positions have zero spatial variance")
    scale = float(np.dot(singular_values, reflection) / denominator)
    translation = target_center - scale * (rotation @ source_center)
    aligned = (scale * (rotation @ source.T)).T + translation
    errors = np.linalg.norm(aligned - target, axis=1)
    return aligned, scale, errors


def aggregate_run_rows(
    rows: list[dict], *, expected_runs: int = 2
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    output = []
    for sequence in sorted(grouped):
        run_rows = sorted(grouped[sequence], key=lambda row: row["run"])
        if len(run_rows) != expected_runs:
            raise ValueError(
                f"{sequence}: expected {expected_runs} runs, found {len(run_rows)}"
            )
        days = {row["day"] for row in run_rows}
        if len(days) != 1:
            raise ValueError(f"{sequence}: inconsistent day values {sorted(days)}")
        aggregate = {
            "day": run_rows[0]["day"],
            "sequence": sequence,
            "run_count": len(run_rows),
            "run_log_coverage": float(
                np.mean([bool(row.get("run_log_available")) for row in run_rows])
            ),
        }
        fields = sorted(set().union(*(row.keys() for row in run_rows)))
        for field in fields:
            if field in {"day", "sequence", "run", "run_log_available"}:
                continue
            values = [row.get(field) for row in run_rows]
            if not all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in values
            ):
                continue
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            aggregate[field] = (
                float(np.median(finite)) if len(finite) else float("nan")
            )
        output.append(aggregate)
    return output


def _rename_stereo_conflicts(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        renamed = {}
        for field, value in row.items():
            if field == "landmarks":
                renamed["stereo_landmarks"] = value
            else:
                renamed[field] = value
        output.append(renamed)
    return output


def build_unified_rows(
    baseline_sequence_rows: list[dict],
    baseline_run_rows: list[dict],
    run_diagnostic_rows: list[dict],
    sim3_rows: list[dict],
    stereo_rows: list[dict],
    observability_rows: list[dict],
    image_rows: list[dict],
    *,
    expected_sequences: int = 24,
    expected_runs: int = 48,
) -> tuple[list[dict], list[dict]]:
    validate_unique_keys(
        baseline_sequence_rows, ("sequence",), expected=expected_sequences
    )
    for rows in (baseline_run_rows, run_diagnostic_rows, sim3_rows, stereo_rows):
        validate_unique_keys(rows, ("sequence", "run"), expected=expected_runs)
    validate_unique_keys(
        observability_rows, ("sequence",), expected=expected_sequences
    )
    validate_unique_keys(image_rows, ("sequence",), expected=expected_sequences)

    run_rows = strict_join(
        baseline_run_rows, run_diagnostic_rows, ("sequence", "run")
    )
    run_rows = strict_join(run_rows, sim3_rows, ("sequence", "run"))
    run_rows = strict_join(
        run_rows,
        _rename_stereo_conflicts(stereo_rows),
        ("sequence", "run"),
    )
    validate_unique_keys(run_rows, ("sequence", "run"), expected=expected_runs)

    run_aggregates = aggregate_run_rows(run_rows, expected_runs=2)
    sequence_rows = strict_join(
        baseline_sequence_rows, run_aggregates, ("sequence",)
    )
    sequence_rows = strict_join(
        sequence_rows, observability_rows, ("sequence",)
    )
    sequence_rows = strict_join(sequence_rows, image_rows, ("sequence",))
    labelled = []
    for row in sequence_rows:
        duration_s = float(row["analysis_duration_s"])
        if duration_s <= 0.0:
            raise ValueError(f"{row['sequence']}: invalid analysis duration")
        enriched = dict(row)
        enriched["angular_events_per_5min"] = (
            float(row["motion_angular_above_3_0_event_count"])
            / (duration_s / 300.0)
        )
        labelled.append(apply_outcome_labels(enriched))
    validate_unique_keys(labelled, ("sequence",), expected=expected_sequences)
    return sorted(labelled, key=lambda row: row["sequence"]), sorted(
        run_rows, key=lambda row: (row["sequence"], row["run"])
    )


def collect_image_diagnostics(
    specs: list,
    baseline_by_sequence: dict[str, dict],
    *,
    image_delay_s: float,
    samples_per_camera: int = 80,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    camera_rows = []
    sample_rows = []
    aggregate_rows = []
    mocap_rows = []
    for spec in specs:
        if spec.sequence not in baseline_by_sequence:
            raise ValueError(f"{spec.sequence}: missing baseline sequence row")
        baseline = baseline_by_sequence[spec.sequence]
        start = float(baseline["analysis_start_s"])
        end = float(baseline["analysis_end_s"])
        reference = repeatability.load_mocap_trajectory(spec.mocap)
        lever = day_analysis.session_fixed_lever(
            spec.sequence, day_analysis.FIXED_DIAGNOSTIC_LEVER_M
        )
        corrected_positions = day_analysis.correct_reference_positions(
            reference.positions,
            reference.quaternions_wxyz,
            lever,
        )
        motion_reference = repeatability.Trajectory(
            reference.timestamps,
            corrected_positions,
            reference.quaternions_wxyz,
            reference.velocities,
        )
        motion = repeatability._motion_from_mocap(
            motion_reference, start, end
        )
        per_camera, per_sample, aggregate = repeatability.analyze_cameras(
            spec.dataset,
            start,
            motion["timestamps"],
            motion["linear_speed"],
            motion["angular_speed"],
            samples_per_camera,
            image_delay_s,
        )
        key = {"day": spec.day, "sequence": spec.sequence}
        camera_rows.extend([{**key, **row} for row in per_camera])
        sample_rows.extend([{**key, **row} for row in per_sample])
        aggregate_rows.append({**key, **aggregate})
        mocap_rows.append(
            {**key, **repeatability.analyze_mocap_integrity(spec.mocap)}
        )
    return camera_rows, sample_rows, aggregate_rows, mocap_rows


def _finite_values(rows: list[dict], field: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def summarize_image_diagnostics(
    camera_rows: list[dict],
    sample_rows: list[dict],
    aggregate_rows: list[dict],
    mocap_rows: list[dict],
    *,
    expected_cameras: int = 4,
) -> list[dict]:
    validate_unique_keys(aggregate_rows, ("sequence",))
    validate_unique_keys(mocap_rows, ("sequence",))
    aggregate_by_sequence = {
        row["sequence"]: row for row in aggregate_rows
    }
    mocap_by_sequence = {row["sequence"]: row for row in mocap_rows}
    cameras_by_sequence = defaultdict(list)
    samples_by_sequence = defaultdict(list)
    for row in camera_rows:
        cameras_by_sequence[row["sequence"]].append(row)
    for row in sample_rows:
        samples_by_sequence[row["sequence"]].append(row)
    sequences = sorted(set(cameras_by_sequence) | set(samples_by_sequence))
    output = []
    for sequence in sequences:
        cameras = cameras_by_sequence[sequence]
        samples = samples_by_sequence[sequence]
        if len(cameras) != expected_cameras:
            raise ValueError(
                f"{sequence}: expected {expected_cameras} cameras, found {len(cameras)}"
            )
        if not samples:
            raise ValueError(f"{sequence}: no sampled image metrics")
        if sequence not in aggregate_by_sequence:
            raise ValueError(f"{sequence}: missing camera aggregate")
        if sequence not in mocap_by_sequence:
            raise ValueError(f"{sequence}: missing mocap integrity row")
        laplacian = _finite_values(samples, "laplacian_variance")
        contrast = _finite_values(samples, "intensity_std")
        dark_clip = _finite_values(samples, "dark_clip_fraction")
        bright_clip = _finite_values(samples, "bright_clip_fraction")
        frame_mae = _finite_values(samples, "previous_frame_mae")
        if not len(laplacian):
            raise ValueError(f"{sequence}: no finite Laplacian variance samples")
        aggregate = aggregate_by_sequence[sequence]
        mocap = mocap_by_sequence[sequence]
        row = {
            "day": cameras[0]["day"],
            "sequence": sequence,
            "camera_count": len(cameras),
            "image_sample_count": len(samples),
            "camera_sharpness_sample_count": int(
                sum(int(camera["sharpness_samples"]) for camera in cameras)
            ),
            "camera_missing_images": int(
                sum(int(camera["missing_images"]) for camera in cameras)
            ),
            "camera_max_interval_ms": float(
                max(float(camera["max_interval_ms"]) for camera in cameras)
            ),
            "laplacian_variance_p5": float(np.percentile(laplacian, 5)),
            "laplacian_variance_p10": float(np.percentile(laplacian, 10)),
            "laplacian_variance_median": float(np.median(laplacian)),
            "intensity_std_median": float(np.median(contrast)),
            "dark_clip_fraction_mean": float(np.mean(dark_clip)),
            "bright_clip_fraction_mean": float(np.mean(bright_clip)),
            "previous_frame_mae_p5": float(np.percentile(frame_mae, 5)),
            "previous_frame_mae_median": float(np.median(frame_mae)),
        }
        row.update(
            {
                field: value
                for field, value in aggregate.items()
                if field not in {"day", "sequence"}
            }
        )
        row.update(
            {
                f"mocap_{field}": value
                for field, value in mocap.items()
                if field not in {"day", "sequence"}
            }
        )
        output.append(row)
    return output


def compute_sim3_run_rows(specs: list) -> list[dict]:
    rows = []
    for spec in specs:
        reference = repeatability.load_mocap_trajectory(spec.mocap)
        lever = day_analysis.session_fixed_lever(
            spec.sequence, day_analysis.FIXED_DIAGNOSTIC_LEVER_M
        )
        for run_dir_value in spec.run_dirs:
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
            _, scale, sim3_errors = similarity_align_and_errors(
                corrected.estimate_positions, corrected.reference_positions
            )
            se3_rmse_mm = float(corrected.rmse_m * 1000.0)
            sim3_rmse_mm = float(np.sqrt(np.mean(sim3_errors**2)) * 1000.0)
            rows.append(
                {
                    "day": spec.day,
                    "sequence": spec.sequence,
                    "run": run_dir.name,
                    "associated_poses": len(evaluation.reference_positions),
                    "se3_rmse_mm": se3_rmse_mm,
                    "sim3_rmse_mm": sim3_rmse_mm,
                    "sim3_improvement_pct": (
                        100.0 * (se3_rmse_mm - sim3_rmse_mm) / se3_rmse_mm
                        if se3_rmse_mm > 0.0
                        else 0.0
                    ),
                    "scale_estimate_to_mocap": scale,
                    "fixed_lever_applied": bool(np.linalg.norm(lever) > 0.0),
                }
            )
    return sorted(rows, key=lambda row: (row["sequence"], row["run"]))


def _finite_spearman(
    values: np.ndarray, outcomes: np.ndarray
) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    valid = np.isfinite(values) & np.isfinite(outcomes)
    count = int(np.count_nonzero(valid))
    if (
        count < 3
        or np.ptp(values[valid]) == 0.0
        or np.ptp(outcomes[valid]) == 0.0
    ):
        return float("nan"), float("nan"), count
    result = spearmanr(values[valid], outcomes[valid])
    return float(result.statistic), float(result.pvalue), count


def compute_correlation_rows(
    sequence_rows: list[dict], factors: list[dict]
) -> list[dict]:
    cohorts = cohort_memberships(sequence_rows)
    output = []
    for factor in factors:
        metric = factor["metric"]
        expected_direction = int(factor["expected_direction"])
        if expected_direction not in {-1, 1}:
            raise ValueError(
                f"{factor['factor']}: expected_direction must be -1 or 1"
            )
        for cohort_name, rows in cohorts.items():
            numeric = []
            for row in rows:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    value = float("nan")
                numeric.append(value)
            values = np.asarray(numeric, dtype=float)
            ape = np.asarray(
                [float(row["corrected_ape_median_mm"]) for row in rows],
                dtype=float,
            )
            rho, pvalue, available = _finite_spearman(values, ape)
            valid = np.isfinite(values)
            high_ape = np.asarray(
                [bool(row["ape_over_10mm"]) for row in rows], dtype=bool
            )
            fragmented = np.asarray(
                [bool(row["visual_fragmentation"]) for row in rows],
                dtype=bool,
            )
            ape_delta = cliffs_delta(
                values[valid & high_ape], values[valid & ~high_ape]
            )
            fragmentation_delta = cliffs_delta(
                values[valid & fragmented], values[valid & ~fragmented]
            )
            output.append(
                {
                    **factor,
                    "cohort": cohort_name,
                    "cohort_sequences": len(rows),
                    "available_sequences": available,
                    "coverage_fraction": available / len(rows) if rows else 0.0,
                    "spearman_rho": rho,
                    "spearman_pvalue": pvalue,
                    "spearman_strength": association_strength(rho),
                    "ape_over_10mm_cliffs_delta": ape_delta,
                    "ape_over_10mm_cliffs_strength": association_strength(
                        ape_delta, effect="cliffs_delta"
                    ),
                    "visual_fragmentation_cliffs_delta": fragmentation_delta,
                    "visual_fragmentation_cliffs_strength": association_strength(
                        fragmentation_delta, effect="cliffs_delta"
                    ),
                    "spearman_expected_direction": bool(
                        np.isfinite(rho) and rho * expected_direction > 0.0
                    ),
                }
            )
    return output


def synthesize_evidence(
    correlation_rows: list[dict], factors: list[dict]
) -> list[dict]:
    grouped = defaultdict(dict)
    for row in correlation_rows:
        grouped[row["factor"]][row["cohort"]] = row
    output = []
    required_cohorts = (
        "all",
        "without_impulse",
        "without_mocap_correction",
        "natural_uncorrected_subset",
    )
    for factor in factors:
        factor_id = factor["factor"]
        by_cohort = grouped.get(factor_id, {})
        missing = [name for name in required_cohorts if name not in by_cohort]
        if missing:
            raise ValueError(f"{factor_id}: missing correlation cohorts {missing}")
        outcome = factor.get("evidence_outcome", "ape")
        if outcome == "ape":
            effect_field = "spearman_rho"
            strength_field = "spearman_strength"
        elif outcome == "visual_fragmentation":
            effect_field = "visual_fragmentation_cliffs_delta"
            strength_field = "visual_fragmentation_cliffs_strength"
        elif outcome == "ape_over_10mm":
            effect_field = "ape_over_10mm_cliffs_delta"
            strength_field = "ape_over_10mm_cliffs_strength"
        else:
            raise ValueError(f"{factor_id}: unsupported evidence outcome {outcome}")
        full = by_cohort["all"]
        sensitivity = [by_cohort[name] for name in required_cohorts[1:]]
        effects = [float(full[effect_field])] + [
            float(row[effect_field]) for row in sensitivity
        ]
        expected_direction = int(factor["expected_direction"])
        direction_consistent = all(
            np.isfinite(effect) and effect * expected_direction > 0.0
            for effect in effects
        )
        grade = population_support_grade(
            full_strength=full[strength_field],
            sensitivity_strengths=[
                row[strength_field] for row in sensitivity
            ],
            coverage_fraction=float(full["coverage_fraction"]),
            direction_consistent=direction_consistent,
        )
        if factor.get("is_proxy", False):
            grade = min(grade, "weak", key=SUPPORT_ORDER.__getitem__)
        output.append(
            {
                **factor,
                "support_level": grade,
                "evidence_outcome": outcome,
                "all_available_sequences": full["available_sequences"],
                "all_coverage_fraction": full["coverage_fraction"],
                "all_spearman_rho": full["spearman_rho"],
                "all_spearman_pvalue": full["spearman_pvalue"],
                "all_ape_over_10mm_cliffs_delta": full[
                    "ape_over_10mm_cliffs_delta"
                ],
                "all_visual_fragmentation_cliffs_delta": full[
                    "visual_fragmentation_cliffs_delta"
                ],
                "sensitivity_direction_consistent": direction_consistent,
            }
        )
    return output


def _sequence_color(row: dict) -> str:
    if bool(row.get("visual_fragmentation", False)):
        return "#b3261e"
    if bool(row.get("ape_over_10mm", False)):
        return "#d97706"
    return "#147d64"


def _short_sequence(sequence: str) -> str:
    if re.fullmatch(r"\d{8}-\d{6}", sequence):
        return f"{sequence[6:8]}-{sequence[-6:]}"
    return sequence


def _finite_field(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def _add_manifest_figure(
    figure: plt.Figure,
    output: Path,
    filename: str,
    claim: str,
    sequence_count: int,
    manifest: list[dict],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    manifest.append(
        {
            "path": str(path),
            "kind": "figure",
            "claim": claim,
            "sequence_count": sequence_count,
        }
    )


def _scatter_ape(
    axis: plt.Axes,
    rows: list[dict],
    field: str,
    label: str,
    *,
    log_x: bool = False,
) -> None:
    x = _finite_field(rows, field)
    y = _finite_field(rows, "corrected_ape_median_mm")
    colors = [_sequence_color(row) for row in rows]
    axis.scatter(x, y, c=colors, s=42, edgecolors="white", linewidths=0.6)
    axis.axhline(APE_THRESHOLD_MM, color="#5f6368", linestyle="--", linewidth=1.0)
    axis.set_xlabel(label)
    axis.set_ylabel("SE(3) APE RMSE [mm]")
    axis.set_yscale("log")
    if log_x and np.all(x > 0.0):
        axis.set_xscale("log")
    axis.grid(True, color="#d9dde0", linewidth=0.6, alpha=0.8)
    for row, x_value, y_value in zip(rows, x, y):
        if bool(row.get("ape_over_10mm", False)):
            axis.annotate(
                _short_sequence(row["sequence"]),
                (x_value, y_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )


def render_population_figures(
    output: Path,
    sequence_rows: list[dict],
    run_rows: list[dict],
    correlation_rows: list[dict],
    evidence_rows: list[dict],
    impulse_rows: list[dict],
) -> list[dict]:
    if not sequence_rows:
        raise ValueError("cannot render figures without sequence rows")
    validate_unique_keys(sequence_rows, ("sequence",))
    output = Path(output)
    rows = sorted(sequence_rows, key=lambda row: row["sequence"])
    labels = [_short_sequence(row["sequence"]) for row in rows]
    indices = np.arange(len(rows))
    manifest = []

    figure, axis = plt.subplots(figsize=(15.0, 6.5))
    medians = _finite_field(rows, "corrected_ape_median_mm")
    axis.bar(indices, medians, color=[_sequence_color(row) for row in rows], alpha=0.82)
    axis.scatter(
        indices - 0.12,
        _finite_field(rows, "corrected_ape_run1_mm"),
        marker="o",
        s=25,
        color="#202124",
        label="run1",
        zorder=3,
    )
    axis.scatter(
        indices + 0.12,
        _finite_field(rows, "corrected_ape_run2_mm"),
        marker="x",
        s=30,
        color="#ffffff",
        linewidths=1.2,
        label="run2",
        zorder=3,
    )
    axis.axhline(APE_THRESHOLD_MM, color="#5f6368", linestyle="--", label="10 mm")
    axis.set_yscale("log")
    axis.set_ylabel("SE(3) APE RMSE [mm]")
    axis.set_xticks(indices, labels, rotation=60, ha="right", fontsize=8)
    axis.set_title(f"Population accuracy and run repeatability (n={len(rows)})")
    axis.grid(True, axis="y", color="#d9dde0", linewidth=0.6)
    axis.legend(ncol=3)
    _add_manifest_figure(
        figure,
        output,
        "ape_population_repeatability.png",
        "population_accuracy_repeatability",
        len(rows),
        manifest,
    )

    angular_fields = (
        ("motion_angular_speed_radps_p95", "Angular speed p95 [rad/s]"),
        ("motion_angular_speed_radps_p99", "Angular speed p99 [rad/s]"),
        ("motion_angular_speed_radps_max", "Angular speed maximum [rad/s]"),
        ("motion_angular_above_3_0_fraction", "Time fraction >3 rad/s"),
        ("angular_events_per_5min", ">3 rad/s events per 5 min"),
        ("orientation_path_rad", "Cumulative orientation path [rad]"),
    )
    available_angular = [item for item in angular_fields if item[0] in rows[0]]
    columns = 3
    plot_rows = int(np.ceil(len(available_angular) / columns))
    figure, axes = plt.subplots(
        plot_rows, columns, figsize=(15.0, 4.6 * plot_rows), squeeze=False
    )
    for axis, (field, label) in zip(axes.flat, available_angular):
        _scatter_ape(axis, rows, field, label)
    for axis in axes.flat[len(available_angular) :]:
        axis.set_axis_off()
    figure.suptitle(
        f"APE versus angular-motion statistics across all sequences (n={len(rows)})",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _add_manifest_figure(
        figure,
        output,
        "ape_angular_velocity_relationships.png",
        "ape_angular_velocity_relationships",
        len(rows),
        manifest,
    )

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    ransac = _finite_field(rows, "ransac_fail_per_min")
    lifetime = _finite_field(rows, "landmark_time_span_median_s")
    colors = [_sequence_color(row) for row in rows]
    axes[0].scatter(ransac, lifetime, c=colors, s=55, edgecolors="white", linewidths=0.7)
    axes[0].axvline(FRAGMENTATION_RANSAC_FAIL_PER_MIN, color="#5f6368", linestyle="--")
    axes[0].axhline(FRAGMENTATION_LANDMARK_SPAN_S, color="#5f6368", linestyle="--")
    axes[0].set_xscale("symlog", linthresh=0.5)
    axes[0].set_xlim(left=0.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("RANSAC FAIL rate [/min]")
    axes[0].set_ylabel("Final-map landmark median span [s]")
    axes[0].set_title("Joint fragmentation label")
    _scatter_ape(axes[1], rows, "ransac_fail_per_min", "RANSAC FAIL rate [/min]")
    axes[1].set_xscale("symlog", linthresh=0.5)
    axes[1].set_xlim(left=0.0)
    for axis in axes:
        axis.grid(True, color="#d9dde0", linewidth=0.6, alpha=0.8)
    figure.suptitle(f"Tracking fragmentation across the full population (n={len(rows)})")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _add_manifest_figure(
        figure,
        output,
        "tracking_landmark_failure_state.png",
        "tracking_fragmentation_population",
        len(rows),
        manifest,
    )

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    _scatter_ape(axes[0], rows, "loop_attempts_per_min", "Loop attempts [/min]")
    _scatter_ape(axes[1], rows, "loop_rejection_fraction", "Loop rejection fraction")
    figure.suptitle(f"Loop-closure response to map fragmentation (n={len(rows)})")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _add_manifest_figure(
        figure,
        output,
        "loop_closure_fragmentation_response.png",
        "loop_closure_population",
        len(rows),
        manifest,
    )

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.6))
    _scatter_ape(
        axes[0], rows, "quality_initialized_fraction", "Initialized quality fraction"
    )
    _scatter_ape(
        axes[1], rows, "observations_per_landmark", "Observations per landmark"
    )
    run_quality = _finite_field(run_rows, "quality_median")
    axes[2].hist(run_quality, bins=min(20, max(5, len(run_quality) // 2)), color="#357a8a")
    axes[2].set_xlabel("Final-map landmark quality median")
    axes[2].set_ylabel("Runs")
    axes[2].set_title(f"Posterior quality over {len(run_rows)} runs")
    axes[2].grid(True, axis="y", color="#d9dde0", linewidth=0.6)
    figure.suptitle("Landmark survival and posterior triangulation geometry")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _add_manifest_figure(
        figure,
        output,
        "landmark_posterior_geometry.png",
        "landmark_posterior_geometry_population",
        len(rows),
        manifest,
    )

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    _scatter_ape(axes[0, 0], rows, "laplacian_variance_p5", "Laplacian variance p5")
    _scatter_ape(axes[0, 1], rows, "intensity_std_median", "Intensity std median")
    _scatter_ape(axes[1, 0], rows, "camera_missing_images", "Missing indexed images")
    _scatter_ape(axes[1, 1], rows, "mocap_tracked_fraction", "Mocap tracked fraction")
    figure.suptitle(f"Image edge-content/sharpness and sensor proxies (n={len(rows)})")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _add_manifest_figure(
        figure,
        output,
        "image_sensor_proxies.png",
        "image_sensor_proxy_population",
        len(rows),
        manifest,
    )

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    _scatter_ape(
        axes[0, 0],
        rows,
        "baseline_over_rotation_p10_cm_per_rad",
        "0.5 s baseline/rotation p10 [cm/rad]",
    )
    _scatter_ape(
        axes[0, 1],
        rows,
        "frac_rotation_gt_0p25_baseline_lt_5cm_pct",
        "High-rotation/low-translation exposure [%]",
    )
    _scatter_ape(
        axes[1, 0], rows, "stereo_landmark_fraction", "Stereo landmark fraction"
    )
    _scatter_ape(
        axes[1, 1], rows, "sim3_improvement_pct", "Sim(3) APE improvement [%]"
    )
    figure.suptitle(f"Observability, stereo support and scale diagnostics (n={len(rows)})")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _add_manifest_figure(
        figure,
        output,
        "observability_stereo_scale.png",
        "observability_scale_population",
        len(rows),
        manifest,
    )

    figure, axis = plt.subplots(
        figsize=(11.5, max(3.8, 0.48 * len(evidence_rows) + 1.5))
    )
    effect = np.asarray([float(row["all_spearman_rho"]) for row in evidence_rows])
    y = np.arange(len(evidence_rows))
    support_colors = {
        "strong": "#147d64",
        "moderate": "#3b78b5",
        "weak": "#d97706",
        "currently_not_supported": "#8a8f94",
    }
    axis.barh(
        y,
        np.nan_to_num(effect),
        color=[support_colors[row["support_level"]] for row in evidence_rows],
    )
    axis.axvline(0.0, color="#202124", linewidth=0.8)
    axis.set_yticks(
        y, [row.get("label_en", row["factor"]) for row in evidence_rows]
    )
    axis.set_xlim(-1.0, 1.0)
    axis.set_xlabel("Spearman rho with SE(3) APE")
    axis.set_title("Candidate and downstream evidence grades")
    axis.grid(True, axis="x", color="#d9dde0", linewidth=0.6)
    axis.legend(
        handles=[
            Patch(color=support_colors["strong"], label="strong"),
            Patch(color=support_colors["moderate"], label="moderate"),
            Patch(color=support_colors["weak"], label="weak"),
            Patch(
                color=support_colors["currently_not_supported"],
                label="currently not supported",
            ),
        ],
        loc="upper left",
        title="Support",
    )
    figure.tight_layout()
    _add_manifest_figure(
        figure,
        output,
        "candidate_factor_evidence.png",
        "candidate_factor_evidence_population",
        len(rows),
        manifest,
    )

    if impulse_rows:
        impulse_sequences = sorted({row["sequence"] for row in impulse_rows})
        figure, axis = plt.subplots(figsize=(12.0, 4.8))
        y_by_sequence = {sequence: index for index, sequence in enumerate(impulse_sequences)}
        for row in impulse_rows:
            y_value = y_by_sequence[row["sequence"]]
            start = float(row["event_start_s"])
            end = float(row["event_end_s"])
            axis.barh(
                y_value,
                end - start,
                left=start,
                height=0.34,
                color="#d97706",
                alpha=0.8,
            )
            axis.scatter(float(row["peak_time_s"]), y_value, marker="^", color="#b3261e")
            for field, marker, color in (
                ("first_sustained_10cm_run1_s", "o", "#147d64"),
                ("first_sustained_10cm_run2_s", "x", "#2457a6"),
            ):
                value = float(row[field])
                if np.isfinite(value):
                    axis.scatter(value, y_value, marker=marker, color=color, s=48)
        axis.set_yticks(
            np.arange(len(impulse_sequences)),
            [_short_sequence(sequence) for sequence in impulse_sequences],
        )
        axis.set_xlabel("Elapsed time from evaluation start [s]")
        axis.set_title("Designed angular impulses and sustained 10 cm drift onset")
        axis.grid(True, axis="x", color="#d9dde0", linewidth=0.6)
        axis.legend(
            handles=[
                Patch(color="#d97706", alpha=0.8, label=">3 rad/s event window"),
                Line2D(
                    [],
                    [],
                    marker="^",
                    linestyle="none",
                    color="#b3261e",
                    label="angular-speed peak",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color="#147d64",
                    label="run1 sustained 10 cm drift",
                ),
                Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="none",
                    color="#2457a6",
                    label="run2 sustained 10 cm drift",
                ),
            ],
            loc="upper left",
            ncol=2,
            fontsize=8,
        )
        figure.tight_layout()
        _add_manifest_figure(
            figure,
            output,
            "angular_impulse_timeline.png",
            "angular_impulse_timing",
            len(impulse_sequences),
            manifest,
        )
    required_tracking_factors = {
        "ransac_fail_rate",
        "loop_attempt_rate",
        "landmark_span",
        "observations_per_landmark",
    }
    if required_tracking_factors <= {
        str(row.get("factor")) for row in evidence_rows
    }:
        tracking_path = output / "tracking_metrics_vs_ape_strong_evidence.png"
        causal_report.plot_fragmentation_and_loop_response(
            rows, evidence_rows, tracking_path
        )
        manifest.append(
            {
                "path": str(tracking_path),
                "kind": "figure",
                "claim": "direct_tracking_metrics_vs_ape",
                "sequence_count": len(rows),
            }
        )
    return manifest


def run_cross_sample_analysis(
    results_root: Path,
    data_root: Path,
    output: Path,
    *,
    days: tuple[str, ...] = analyze_multiday.DEFAULT_DAYS,
    expected_sequences: int | None = analyze_multiday.DEFAULT_EXPECTED_SEQUENCES,
    samples_per_camera: int = 80,
    causal_diagnostics_root: Path | None = None,
) -> dict[str, list[dict]]:
    results_root = Path(results_root)
    data_root = Path(data_root)
    output = Path(output)
    tables = output / "tables"
    figures = output / "figures"
    specs = analyze_multiday.discover_sequences(
        results_root, data_root, days=days
    )
    if expected_sequences is not None and len(specs) != expected_sequences:
        raise ValueError(
            f"expected {expected_sequences} sequences, found {len(specs)}"
        )
    expected_sequence_count = len(specs)
    expected_run_count = sum(len(spec.run_dirs) for spec in specs)
    if expected_run_count != expected_sequence_count * 2:
        raise ValueError(
            f"expected two runs per sequence, found {expected_run_count} runs "
            f"for {expected_sequence_count} sequences"
        )

    baseline_sequence_rows = read_csv_rows(
        tables / "multiday_sequence_metrics.csv"
    )
    baseline_run_rows = read_csv_rows(tables / "multiday_run_metrics.csv")
    validate_unique_keys(
        baseline_sequence_rows,
        ("sequence",),
        expected=expected_sequence_count,
    )
    validate_unique_keys(
        baseline_run_rows,
        ("sequence", "run"),
        expected=expected_run_count,
    )
    baseline_by_sequence = {
        row["sequence"]: row for row in baseline_sequence_rows
    }
    duration_by_sequence = {
        sequence: float(row["analysis_duration_s"])
        for sequence, row in baseline_by_sequence.items()
    }

    print(f"Collecting final-map and runtime diagnostics for {expected_run_count} runs")
    run_diagnostic_rows = collect_run_diagnostics(
        specs, duration_by_sequence
    )
    validate_unique_keys(
        run_diagnostic_rows,
        ("sequence", "run"),
        expected=expected_run_count,
    )

    print("Recomputing correction-aware SE(3)/Sim(3) diagnostics")
    sim3_rows = compute_sim3_run_rows(specs)
    validate_unique_keys(
        sim3_rows, ("sequence", "run"), expected=expected_run_count
    )

    stereo_rows = read_csv_rows(
        tables / "observability_stereo_support_by_run.csv"
    )
    observability_rows = read_csv_rows(
        tables / "observability_sequence_0p5s.csv"
    )

    print(
        f"Sampling {samples_per_camera} deterministic frames per camera "
        f"for {expected_sequence_count} sequences"
    )
    image_delay_s = repeatability.parse_image_delay(
        REPOSITORY / "config/okvis2_eucm_EGO2.yaml"
    )
    camera_rows, image_sample_rows, camera_aggregate_rows, mocap_rows = (
        collect_image_diagnostics(
            specs,
            baseline_by_sequence,
            image_delay_s=image_delay_s,
            samples_per_camera=samples_per_camera,
        )
    )
    image_sequence_rows = summarize_image_diagnostics(
        camera_rows,
        image_sample_rows,
        camera_aggregate_rows,
        mocap_rows,
    )

    sequence_rows, run_rows = build_unified_rows(
        baseline_sequence_rows,
        baseline_run_rows,
        run_diagnostic_rows,
        sim3_rows,
        stereo_rows,
        observability_rows,
        image_sequence_rows,
        expected_sequences=expected_sequence_count,
        expected_runs=expected_run_count,
    )
    correlation_rows = compute_correlation_rows(
        sequence_rows, list(DEFAULT_FACTORS)
    )
    evidence_rows = synthesize_evidence(
        correlation_rows, list(DEFAULT_FACTORS)
    )
    impulse_path = tables / "20260806_angular_impulse_events.csv"
    impulse_rows = (
        read_csv_rows(impulse_path)
        if "20260806" in days and impulse_path.is_file()
        else []
    )

    table_outputs = (
        (
            "cross_sample_sequence_metrics.csv",
            sequence_rows,
            "unified_sequence_metrics",
            expected_sequence_count,
        ),
        (
            "cross_sample_run_metrics.csv",
            run_rows,
            "unified_run_metrics",
            expected_sequence_count,
        ),
        (
            "cross_sample_correlations.csv",
            correlation_rows,
            "cohort_associations",
            expected_sequence_count,
        ),
        (
            "cross_sample_evidence.csv",
            evidence_rows,
            "graded_factor_evidence",
            expected_sequence_count,
        ),
        (
            "cross_sample_camera_metrics.csv",
            camera_rows,
            "uniform_camera_metrics",
            expected_sequence_count,
        ),
        (
            "cross_sample_image_samples.csv",
            image_sample_rows,
            "uniform_image_samples",
            expected_sequence_count,
        ),
        (
            "cross_sample_image_sequence_metrics.csv",
            image_sequence_rows,
            "image_proxy_sequence_metrics",
            expected_sequence_count,
        ),
        (
            "cross_sample_mocap_integrity.csv",
            mocap_rows,
            "mocap_integrity",
            expected_sequence_count,
        ),
        (
            "cross_sample_sim3_by_run.csv",
            sim3_rows,
            "correction_aware_scale_diagnostics",
            expected_sequence_count,
        ),
    )
    manifest = []
    for filename, rows_to_write, claim, coverage in table_outputs:
        path = tables / filename
        write_csv_rows(path, rows_to_write)
        manifest.append(
            {
                "path": f"tables/{filename}",
                "kind": "table",
                "claim": claim,
                "sequence_count": coverage,
            }
        )

    figure_manifest = render_population_figures(
        figures,
        sequence_rows,
        run_rows,
        correlation_rows,
        evidence_rows,
        impulse_rows,
    )
    manifest.extend(
        {
            **row,
            "path": f"figures/{Path(row['path']).name}",
        }
        for row in figure_manifest
    )
    if causal_diagnostics_root is not None:
        causal_analysis.analyze_runs(
            argparse.Namespace(
                diagnostics_root=Path(causal_diagnostics_root),
                data_root=data_root,
                output=output,
                sequences=[spec.sequence for spec in specs],
                bootstrap_samples=500,
            )
        )
        population_refresh.refresh(
            argparse.Namespace(
                tables_root=tables,
                figures_root=figures,
                bootstrap_samples=500,
            )
        )
        write_causal_evidence_report(output)
        manifest.extend(
            causal_artifact_rows(output, expected_sequence_count)
        )
    write_csv_rows(tables / "cross_sample_artifact_manifest.csv", manifest)
    print(
        f"Wrote {len(sequence_rows)} sequence rows, {len(run_rows)} run rows, "
        f"and {len(figure_manifest)} figures to {output}"
    )
    return {
        "sequence_rows": sequence_rows,
        "run_rows": run_rows,
        "correlation_rows": correlation_rows,
        "evidence_rows": evidence_rows,
        "manifest": manifest,
    }


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
        "--output", type=Path, default=analyze_multiday.DEFAULT_OUTPUT
    )
    parser.add_argument("--days", nargs="+", default=list(analyze_multiday.DEFAULT_DAYS))
    parser.add_argument("--samples-per-camera", type=int, default=80)
    parser.add_argument(
        "--causal-diagnostics-root",
        type=Path,
        default=None,
        help="Validated replay root containing structured VIO diagnostics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    days = tuple(arguments.days)
    expected = (
        analyze_multiday.DEFAULT_EXPECTED_SEQUENCES
        if days == analyze_multiday.DEFAULT_DAYS
        else None
    )
    run_cross_sample_analysis(
        arguments.results_root,
        arguments.data_root,
        arguments.output,
        days=days,
        expected_sequences=expected,
        samples_per_camera=arguments.samples_per_camera,
        causal_diagnostics_root=arguments.causal_diagnostics_root,
    )
    return 0


def cohort_memberships(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "all": list(rows),
        "without_impulse": [
            row for row in rows if row["sequence"] not in IMPULSE_SEQUENCES
        ],
        "without_mocap_correction": [
            row
            for row in rows
            if row["sequence"] not in MOCAP_CORRECTED_SEQUENCES
        ],
        "natural_uncorrected_subset": [
            row
            for row in rows
            if row["sequence"] not in IMPULSE_SEQUENCES
            and row["sequence"] not in MOCAP_CORRECTED_SEQUENCES
        ],
    }


def causal_artifact_rows(
    output: Path, sequence_count: int
) -> list[dict[str, object]]:
    output = Path(output)
    expected = [
        *(output / "tables" / filename for filename in CAUSAL_TABLES),
        *(output / "figures" / filename for filename in CAUSAL_FIGURES),
        *(output / filename for filename in CAUSAL_REPORTS),
    ]
    missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(
            "incomplete causal diagnostics output: "
            + ", ".join(str(path) for path in missing)
        )
    rows = [
        {
            "path": f"tables/{filename}",
            "kind": "table",
            "claim": "event_aligned_causal_mediator_evidence",
            "sequence_count": sequence_count,
        }
        for filename in CAUSAL_TABLES
    ]
    rows.extend(
        {
            "path": f"figures/{filename}",
            "kind": "figure",
            "claim": "event_aligned_causal_mediator_evidence",
            "sequence_count": sequence_count,
        }
        for filename in CAUSAL_FIGURES
    )
    rows.extend(
        {
            "path": filename,
            "kind": "report",
            "claim": "graded_causal_mediator_evidence",
            "sequence_count": sequence_count,
        }
        for filename in CAUSAL_REPORTS
    )
    return rows


def _finite_evidence_value(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _has_direct_measurement(
    event_rows: list[dict], metrics: tuple[str, ...]
) -> bool:
    suffixes = ("delta", "baseline", "mediator", "paired_delta")
    return any(
        _finite_evidence_value(row.get(f"{metric}_{suffix}"))
        for row in event_rows
        for metric in metrics
        for suffix in suffixes
    )


def _row_has_metric(row: dict, metric: str) -> bool:
    return any(
        _finite_evidence_value(row.get(f"{metric}_{suffix}"))
        for suffix in ("delta", "baseline", "mediator", "paired_delta")
    )


def _recovery_contrast(
    event_rows: list[dict],
    recovery_rows: list[dict],
    metrics: tuple[str, ...],
) -> tuple[str, bool]:
    sequences = (
        "20260806-175103",
        "20260806-175304",
        "20260806-175539",
    )
    recovery_sequences = {
        str(row.get("sequence")) for row in recovery_rows
    }
    if not set(sequences).issubset(recovery_sequences):
        return "not_available", False
    compared = False
    for metric in metrics:
        by_sequence = {
            sequence: [
                row for row in event_rows
                if str(row.get("sequence")) == sequence
                and _row_has_metric(row, metric)
            ]
            for sequence in sequences
        }
        if any(not rows for rows in by_sequence.values()):
            continue
        compared = True
        target_recovers = all(
            _finite_evidence_value(row.get(f"{metric}_recovery_s"))
            for row in by_sequence[sequences[0]]
        )
        failures_do_not_recover = all(
            not _finite_evidence_value(row.get(f"{metric}_recovery_s"))
            for sequence in sequences[1:]
            for row in by_sequence[sequence]
        )
        if target_recovers and failures_do_not_recover:
            return "supported", True
    return ("not_supported" if compared else "not_available"), False


def _temporal_precedence(
    event_rows: list[dict], metrics: tuple[str, ...], *, feedback: bool = False
) -> tuple[str, bool]:
    compared = 0
    ordered = 0
    for row in event_rows:
        gp3p = row.get("gp3p_onset_s")
        if feedback:
            removal = row.get("visual_observation_removals_onset_s")
            map_loss = row.get("active_initialised_landmarks_onset_s")
            if not all(
                _finite_evidence_value(value)
                for value in (gp3p, removal, map_loss)
            ):
                continue
            compared += 1
            ordered += float(gp3p) <= float(removal) <= float(map_loss)
            continue
        for metric in metrics:
            onset = row.get(f"{metric}_onset_s")
            if not (
                _finite_evidence_value(onset)
                and _finite_evidence_value(gp3p)
            ):
                continue
            compared += 1
            ordered += float(onset) <= float(gp3p)
    if compared == 0:
        return "not_measured", False
    label = "supported" if ordered else "not_supported"
    return f"{label}:{ordered}/{compared}", ordered > 0


def _dose_relation(
    model_rows: list[dict], families: tuple[str, ...]
) -> tuple[str, bool]:
    eligible = [
        row
        for row in model_rows
        if str(row.get("family")) in families
        and str(row.get("status")) in {"ok", "exploratory_small_n"}
        and _finite_evidence_value(row.get("spearman"))
    ]
    if not eligible:
        return "not_measured", False
    stable = sum(abs(float(row["spearman"])) >= 0.20 for row in eligible)
    label = "supported" if stable else "not_supported"
    return f"{label}:{stable}/{len(eligible)}", stable > 0


def build_causal_hypothesis_evidence(
    event_rows: list[dict],
    model_rows: list[dict],
    recovery_rows: list[dict],
) -> list[dict[str, object]]:
    specifications = (
        (
            "H1", "H1 图像与特征退化",
            ("keypoints_total", "grid_fraction_mean", "hull_fraction_mean",
             "image_laplacian_variance", "accepted_map_matches"),
            ("feature_availability", "map_matching"), False,
            ("exposure", "image_exposure", "texture", "feature_density"),
            "离线图像清晰度仍是纹理/模糊混合代理；稳定特征与匹配会反驳该路径。",
        ),
        (
            "H2_initialisation", "H2 初始化路径（2D-2D）",
            ("rotation_only_minus_relative_pose_inlier_ratio",),
            ("triangulation_geometry",), False,
            ("geometry", "translation_baseline", "camera_translation"),
            "只适用于实际执行 2D-2D 初始化的帧，不能外推解释已初始化阶段的 GP3P。",
        ),
        (
            "H2_runtime", "H2 运行期 3D-2D 路径（GP3P）",
            ("temporal_ray_angle_p10_rad", "temporal_parallel_fraction",
             "spatial_ray_angle_p10_rad", "initialisable_fraction"),
            ("triangulation_geometry",), False,
            ("geometry", "translation_baseline", "camera_translation"),
            "mocap 位移不是 accepted-feature parallax；结论以实际射线角和相机中心为准。",
        ),
        (
            "H3", "H3 预测与标定敏感性",
            ("predicted_reprojection_error_px_median",
             "gp3p_start_to_model_rotation_rad",
             "gp3p_pre_invocation_to_model_rotation_rad"),
            ("prediction_consistency",), False,
            ("imu_time_offset_ns", "timing", "calibration", "exposure_timing"),
            "观测残差不能唯一识别时延、曝光、外参或 IMU 误差，需通过受控干预区分。",
        ),
        (
            "H4", "H4 地图支撑反馈回路",
            ("visual_observation_removals", "active_initialised_landmarks",
             "landmark_births", "gp3p_inlier_ratio"),
            ("map_feedback",), True,
            ("map_support",),
            "RANSAC 失败、短 landmark 寿命和 observation removal 仍是下游状态，不能作为独立根因。",
        ),
    )
    output = []
    for (
        path, label, metrics, families, feedback, intervention_types,
        limitation,
    ) in specifications:
        direct = _has_direct_measurement(event_rows, metrics)
        temporal_label, temporal = _temporal_precedence(
            event_rows, metrics, feedback=feedback
        )
        dose_label, dose = _dose_relation(model_rows, families)
        recovery_label, recovery_supported = _recovery_contrast(
            event_rows, recovery_rows, metrics
        )
        intervention_rows = [
            row for row in event_rows
            if str(row.get("intervention", "")) in intervention_types
        ]
        intervention_valid = any(
            str(row.get("intervention_valid", "")).lower()
            in {"1", "true", "yes"}
            for row in intervention_rows
        )
        specificity_valid = any(
            str(row.get("specificity_valid", "")).lower()
            in {"1", "true", "yes"}
            for row in intervention_rows
        )
        replication_valid = any(
            str(row.get("replication_valid", "")).lower()
            in {"1", "true", "yes"}
            for row in intervention_rows
        )
        fully_validated_intervention = any(
            all(
                str(row.get(field, "")).lower() in {"1", "true", "yes"}
                for field in (
                    "intervention_valid",
                    "specificity_valid",
                    "replication_valid",
                )
            )
            for row in intervention_rows
        )
        support = "currently_not_supported"
        if direct:
            support = "weak"
        if direct and temporal and dose and recovery_supported:
            support = "moderate"
        if support == "moderate" and fully_validated_intervention:
            support = "strong"
        controlled = "not_available"
        if intervention_rows:
            controlled = (
                "validated" if intervention_valid else "available_not_validated"
            )
        output.append(
            {
                "path": path,
                "label_zh": label,
                "direct_measurement": "available" if direct else "not_available",
                "temporal_precedence": temporal_label,
                "dose_relation": dose_label,
                "recovery_175103_contrast": recovery_label,
                "controlled_intervention": controlled,
                "specificity": (
                    "validated" if specificity_valid else "not_validated"
                ),
                "replication": (
                    "validated" if replication_valid else "not_validated"
                ),
                "support_level": support,
                "limitation": limitation,
            }
        )
    return output


def render_causal_evidence_section(rows: list[dict[str, object]]) -> str:
    lines = [
        "## 仅基于结构化事件表的中介证据",
        "",
        "本表是保守的事件内证据审计，不纳入另行归档的 image_delay 受控重放，也不以 26-run 方向一致性替代事件级 onset/剂量/恢复三项同时成立。完整综合等级以 `202608_causal_diagnostics/causal_chain_report/report.md` 为准。",
        "",
        "| 路径 | 直接测量 | 时序先后 | 剂量关系 | 175103 恢复对照 | 受控干预 | 特异性 | 重复性 | 支持等级 | 局限性 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {label_zh} | {direct_measurement} | {temporal_precedence} | "
            "{dose_relation} | {recovery_175103_contrast} | "
            "{controlled_intervention} | {specificity} | {replication} | "
            "{support_level} | {limitation} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "支持等级以同序列配对事件为基础；只有本事件表内部的受控干预通过 manipulation check 后才允许评为 strong。",
            "H2 初始化路径与运行期 3D-2D GP3P 分开解释。",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_causal_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_causal_evidence_report(output: Path) -> list[dict[str, object]]:
    output = Path(output)
    tables = output / "tables"
    rows = build_causal_hypothesis_evidence(
        _read_causal_rows(tables / "causal_event_metrics.csv"),
        _read_causal_rows(tables / "causal_mediation_models.csv"),
        _read_causal_rows(tables / "impulse_mediator_recovery.csv"),
    )
    write_csv_rows(tables / "causal_hypothesis_evidence.csv", rows)
    (output / "causal_mediator_evidence.md").write_text(
        render_causal_evidence_section(rows), encoding="utf-8"
    )
    return rows


def cliffs_delta(group_a: list[float], group_b: list[float]) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    greater = np.sum(a[:, None] > b[None, :])
    lower = np.sum(a[:, None] < b[None, :])
    return float((greater - lower) / (len(a) * len(b)))


def association_strength(
    value: float, *, effect: str = "spearman"
) -> str:
    magnitude = abs(float(value))
    if not np.isfinite(magnitude):
        return "currently_not_supported"
    if effect == "spearman":
        thresholds = (0.60, 0.35, 0.20)
    elif effect == "cliffs_delta":
        thresholds = (0.80, 0.474, 0.147)
    else:
        raise ValueError(f"unsupported effect type: {effect}")
    if magnitude >= thresholds[0]:
        return "strong"
    if magnitude >= thresholds[1]:
        return "moderate"
    if magnitude >= thresholds[2]:
        return "weak"
    return "currently_not_supported"


def population_support_grade(
    *,
    full_strength: str,
    sensitivity_strengths: list[str],
    coverage_fraction: float,
    direction_consistent: bool,
) -> str:
    strengths = [full_strength, *sensitivity_strengths]
    unknown = [strength for strength in strengths if strength not in SUPPORT_ORDER]
    if unknown:
        raise ValueError(f"unsupported evidence strength: {unknown[0]}")
    grade = min(strengths, key=SUPPORT_ORDER.__getitem__)
    if coverage_fraction < 0.80 or not direction_consistent:
        grade = min(grade, "weak", key=SUPPORT_ORDER.__getitem__)
    return grade


def apply_outcome_labels(row: dict) -> dict:
    output = dict(row)
    output["ape_over_10mm"] = (
        float(row["corrected_ape_median_mm"]) > APE_THRESHOLD_MM
    )
    output["visual_fragmentation"] = (
        float(row["ransac_fail_per_min"])
        >= FRAGMENTATION_RANSAC_FAIL_PER_MIN
        and float(row["landmark_time_span_median_s"])
        <= FRAGMENTATION_LANDMARK_SPAN_S
    )
    scale_deviation = abs(float(row["scale_estimate_to_mocap"]) - 1.0)
    output["scale_instability"] = (
        float(row["sim3_improvement_pct"])
        >= SCALE_IMPROVEMENT_THRESHOLD_PCT
        and scale_deviation + 1e-12 >= SCALE_DEVIATION_THRESHOLD
    )
    output["impulse_experiment"] = row.get("sequence") in IMPULSE_SEQUENCES
    return output


if __name__ == "__main__":
    raise SystemExit(main())
