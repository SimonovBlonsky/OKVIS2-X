#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "okvis_analysis_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "okvis_analysis_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from tools.evaluate_mocap_ape import parse_mocap


DEFAULT_RESULTS_ROOT = REPOSITORY / "workspace/ego2_results/20260803-184537"
DEFAULT_DATASET = Path("/home/chenguyuan/data/20260803/20260803-184537_euroc")
DEFAULT_MOCAP = Path(
    "/home/chenguyuan/data/20260803/mocap_ego2_20260803/"
    "mocap_20260803_184540.log"
)
DEFAULT_CONFIG = REPOSITORY / "config/okvis2_eucm_EGO2.yaml"
CONTROL_SEQUENCE_183537 = "20260803-183537"
CONTROL_DATASET_183537 = Path(
    "/home/chenguyuan/data/20260803/20260803-183537_euroc"
)
CONTROL_SEQUENCE_184027 = "20260803-184027"
CONTROL_DATASET_184027 = Path(
    "/home/chenguyuan/data/20260803/20260803-184027_euroc"
)
COMPARISON_RUN = "bak3"
EXCLUDED_RUNS = {"bak0"}
STAGE_FILES = {
    "online": "okvis2-slam-calib_trajectory.csv",
    "final": "okvis2-slam-calib-final_trajectory.csv",
    "final-ba": "okvis2-slam-calib-final-ba_trajectory.csv",
}
CAMERA_GAP_EVENT_FIELDS = (
    "sequence",
    "camera",
    "raw_timestamp_s",
    "timestamp_s",
    "elapsed_s",
    "interval_ms",
    "median_interval_ms",
    "following_interval_ms",
    "immediately_followed_by_bunched_frame",
    "paired_short_interval_delay_s",
    "paired_short_interval_ms",
    "has_later_paired_short_interval",
)
RUN_COLORS = {
    "bak0": "#b3261e",
    "bak1": "#e67e22",
    "bak2": "#147d92",
    "bak3": "#16843b",
    "bak4": "#7357a5",
}


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    role: str
    dataset: Path
    result_dir: Path
    mocap: Path
    color: str


@dataclass(frozen=True)
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_wxyz: np.ndarray
    velocities: np.ndarray


@dataclass(frozen=True)
class ImuData:
    timestamps: np.ndarray
    gyroscope: np.ndarray
    accelerometer: np.ndarray


@dataclass(frozen=True)
class AlignedEvaluation:
    timestamps: np.ndarray
    reference_positions: np.ndarray
    reference_quaternions_wxyz: np.ndarray
    estimate_positions: np.ndarray
    estimate_quaternions_wxyz: np.ndarray
    errors: np.ndarray


def grouped_bar_layout(group_count: int) -> tuple[float, list[float]]:
    if group_count < 1:
        raise ValueError("group_count must be at least one")
    width = min(0.8 / group_count, 0.34)
    offsets = [
        (index - (group_count - 1) / 2.0) * width
        for index in range(group_count)
    ]
    return width, offsets


def sync_percentage_axis_limits(
    values: list[float] | np.ndarray,
) -> tuple[float, float]:
    try:
        percentages = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("sync percentages must be finite") from error
    if percentages.ndim != 1 or not len(percentages):
        raise ValueError("sync percentages must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(percentages)):
        raise ValueError("sync percentages must be finite")
    if np.any(percentages < 0.0) or np.any(percentages > 100.0):
        raise ValueError("sync percentages must be between 0 and 100")

    minimum = float(np.min(percentages))
    maximum = float(np.max(percentages))
    span = max(maximum - minimum, 1.0)
    lower = max(0.0, minimum - max(0.5, 0.08 * span))
    upper = maximum + max(0.75, 0.10 * span)
    return lower, upper


def unique_mocap_log(result_dir: Path) -> Path:
    mocap_paths = sorted(result_dir.glob("mocap_*.log"))
    if len(mocap_paths) != 1:
        raise ValueError(
            f"expected one mocap log in {result_dir}, found {len(mocap_paths)}"
        )
    return mocap_paths[0]


def build_sequence_specs(arguments: argparse.Namespace) -> list[SequenceSpec]:
    control_root = arguments.results_root.parent
    control_result_dir_183537 = control_root / CONTROL_SEQUENCE_183537
    control_result_dir_184027 = control_root / CONTROL_SEQUENCE_184027
    return [
        SequenceSpec(
            name=CONTROL_SEQUENCE_183537,
            role="control",
            dataset=arguments.control_dataset_183537,
            result_dir=control_result_dir_183537,
            mocap=unique_mocap_log(control_result_dir_183537),
            color="#147d92",
        ),
        SequenceSpec(
            name=CONTROL_SEQUENCE_184027,
            role="control",
            dataset=arguments.control_dataset_184027,
            result_dir=control_result_dir_184027,
            mocap=unique_mocap_log(control_result_dir_184027),
            color="#7357a5",
        ),
        SequenceSpec(
            name=f"20260803-184537/{COMPARISON_RUN}",
            role="target",
            dataset=arguments.dataset,
            result_dir=arguments.results_root / COMPARISON_RUN,
            mocap=arguments.mocap,
            color="#b3261e",
        ),
    ]


def partition_sequence_contexts(contexts: list[dict]) -> tuple[list[dict], dict]:
    valid_roles = {"control", "target"}
    for context in contexts:
        role = context["role"]
        if role not in valid_roles:
            raise ValueError(
                f"sequence {context['sequence']!r} has invalid role {role!r}"
            )
    controls = [context for context in contexts if context["role"] == "control"]
    targets = [context for context in contexts if context["role"] == "target"]
    if not controls:
        raise ValueError("expected at least one control sequence")
    if len(targets) != 1:
        raise ValueError(
            f"expected exactly one target sequence, found {len(targets)}"
        )
    return controls, targets[0]


def control_envelope_status(
    control_values: list[float] | np.ndarray, target_value: float
) -> str:
    try:
        controls = np.asarray(control_values, dtype=float)
        target = float(target_value)
    except (TypeError, ValueError) as error:
        raise ValueError("finite control and target values are required") from error
    if controls.ndim != 1 or not len(controls) or not np.all(np.isfinite(controls)):
        raise ValueError("finite control values are required")
    if not np.isfinite(target):
        raise ValueError("target value must be finite")
    if target < float(np.min(controls)):
        return "below"
    if target > float(np.max(controls)):
        return "above"
    return "within"


def control_target_metric_rows(
    contexts: list[dict], run_rows: list[dict]
) -> list[dict]:
    controls, _ = partition_sequence_contexts(contexts)
    if len(controls) != 2:
        raise ValueError(f"expected exactly two controls, found {len(controls)}")
    expected_runs = [f"bak{index}" for index in range(1, 5)]
    runs_by_name = {row["run"]: row for row in run_rows}
    if sorted(row["run"] for row in run_rows) != expected_runs:
        raise ValueError("expected exactly one row for each of bak1 through bak4")

    rows = []
    for context in sorted(controls, key=lambda item: item["sequence"]):
        sequence = context["sequence"]
        summary = context["summary"]
        rows.append(
            {
                "kind": "control",
                "name": sequence,
                "label": f"control\n{sequence.rsplit('-', 1)[-1]}",
                "color": context["color"],
                "marker": "D",
                "ape_rmse_mm": summary["ape_rmse_m"] * 1000.0,
                "observations_per_landmark": summary[
                    "observations_per_landmark"
                ],
                "distinct_states_per_landmark_mean": summary[
                    "distinct_states_per_landmark_mean"
                ],
                "landmark_time_span_median_s": summary[
                    "landmark_time_span_median_s"
                ],
            }
        )
    for run in expected_runs:
        summary = runs_by_name[run]
        rows.append(
            {
                "kind": "target",
                "name": run,
                "label": f"target\n{run}",
                "color": RUN_COLORS[run],
                "marker": "o",
                "ape_rmse_mm": summary["ape_rmse_m"] * 1000.0,
                "observations_per_landmark": summary[
                    "observations_per_landmark"
                ],
                "distinct_states_per_landmark_mean": summary[
                    "distinct_states_per_landmark_mean"
                ],
                "landmark_time_span_median_s": summary[
                    "landmark_time_span_median_s"
                ],
            }
        )
    return rows


def control_metric_axis_scale(field: str, values: list[float] | np.ndarray) -> str:
    try:
        numeric_values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} values must be finite") from error
    if (
        numeric_values.ndim != 1
        or not len(numeric_values)
        or not np.all(np.isfinite(numeric_values))
    ):
        raise ValueError(f"{field} values must be finite")
    if field == "ape_rmse_mm":
        if np.any(numeric_values <= 0.0):
            raise ValueError("APE values must be positive and finite")
        return "log"
    if field == "landmark_time_span_median_s":
        if np.any(numeric_values < 0.0):
            raise ValueError("landmark span values must be nonnegative and finite")
        return "log" if np.all(numeric_values > 0.0) else "linear"
    return "linear"


def pooled_control_sharpness_thresholds(contexts: list[dict]) -> dict[str, float]:
    controls, _ = partition_sequence_contexts(contexts)
    pooled_values = defaultdict(list)
    expected_cameras = None
    expected_counts = {}
    expected_sequence = None
    for context in controls:
        sequence = context["sequence"]
        values_by_camera = defaultdict(list)
        for row in context["quality_rows"]:
            values_by_camera[row["camera"]].append(row["laplacian_variance"])
        if not values_by_camera:
            raise ValueError(f"control {sequence!r} has no sharpness samples")

        cameras = set(values_by_camera)
        is_first_control = expected_cameras is None
        if is_first_control:
            expected_cameras = cameras
            expected_sequence = sequence
        elif cameras != expected_cameras:
            missing = sorted(expected_cameras - cameras)
            extra = sorted(cameras - expected_cameras)
            raise ValueError(
                f"control {sequence!r} camera set differs from control "
                f"{expected_sequence!r}: missing cameras {missing}, "
                f"extra cameras {extra}"
            )

        for camera in sorted(cameras):
            try:
                values = np.asarray(values_by_camera[camera], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"control {sequence!r} camera {camera!r} has "
                    "non-numeric sharpness samples"
                ) from error
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"control {sequence!r} camera {camera!r} has "
                    "non-finite sharpness samples"
                )
            count = len(values)
            if is_first_control:
                expected_counts[camera] = count
            elif count != expected_counts[camera]:
                raise ValueError(
                    f"control {sequence!r} camera {camera!r} has {count} samples; "
                    f"expected {expected_counts[camera]} to match control "
                    f"{expected_sequence!r}"
                )
            pooled_values[camera].extend(values.tolist())
    return {
        camera: float(np.percentile(values, 5))
        for camera, values in sorted(pooled_values.items())
    }


def path_length(positions: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must be an N x 3 array with at least two rows")
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _validated_intervals(timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.ndim != 1 or len(timestamps) != len(positions):
        raise ValueError("timestamps and poses must have the same length")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return intervals


def linear_speed(timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    intervals = _validated_intervals(timestamps, positions)
    return np.linalg.norm(np.diff(positions, axis=0), axis=1) / intervals


def angular_speed(timestamps: np.ndarray, quaternions_wxyz: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must be an N x 4 wxyz array")
    intervals = _validated_intervals(timestamps, quaternions)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("quaternions must have non-zero norm")
    quaternions = quaternions / norms[:, None]
    dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
    angles = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    return angles / intervals


def bounded_interpolate(
    query: np.ndarray, source_timestamps: np.ndarray, source_values: np.ndarray
) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    timestamps = np.asarray(source_timestamps, dtype=float)
    values = np.asarray(source_values, dtype=float)
    if timestamps.ndim != 1 or values.ndim != 1 or timestamps.shape != values.shape:
        raise ValueError("source timestamps and values must be matching vectors")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("source timestamps must be strictly increasing")
    result = np.asarray(np.interp(query, timestamps, values), dtype=float)
    return np.where(
        (query < timestamps[0]) | (query > timestamps[-1]), float("nan"), result
    )


def rigid_align_and_errors(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching N x 3 arrays")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    return aligned, np.linalg.norm(aligned - target, axis=1)


def prefix_align_and_errors(
    source: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    alignment_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching N x 3 arrays")
    if timestamps.ndim != 1 or len(timestamps) != len(source):
        raise ValueError("timestamps and positions must have the same length")
    if alignment_duration < 0.0 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps and alignment duration must be valid")
    fit = timestamps - timestamps[0] <= alignment_duration
    if np.count_nonzero(fit) < 3:
        raise ValueError("alignment window must contain at least three poses")
    source_center = source[fit].mean(axis=0)
    target_center = target[fit].mean(axis=0)
    covariance = (source[fit] - source_center).T @ (target[fit] - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    return aligned, np.linalg.norm(aligned - target, axis=1)


def first_sustained_crossing(
    values: np.ndarray, threshold: float, samples: int
) -> int | None:
    values = np.asarray(values, dtype=float)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if len(values) < samples:
        return None
    windows = np.convolve(
        (values > threshold).astype(np.int64),
        np.ones(samples, dtype=np.int64),
        mode="valid",
    )
    matches = np.flatnonzero(windows == samples)
    return int(matches[0]) if len(matches) else None


def count_g2o_records(path: Path) -> Counter:
    counts = Counter()
    with Path(path).open(encoding="utf-8", errors="replace") as source:
        for line in source:
            fields = line.split(maxsplit=1)
            if fields:
                counts[fields[0]] += 1
    return counts


def map_track_statistics(path: Path) -> dict:
    state_times = {}
    landmark_ids = set()
    landmark_states = defaultdict(set)
    with Path(path).open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                if fields[0] == "VERTEX_SE3:QUAT_TIME":
                    state_times[int(fields[1])] = float(fields[-1]) / 1e9
                elif fields[0] == "VERTEX_TRACKXYZ":
                    landmark_ids.add(int(fields[1]))
                elif fields[0] == "EDGE_OBS":
                    state_id = int(fields[1])
                    landmark_id = int(fields[4])
                    landmark_ids.add(landmark_id)
                    landmark_states[landmark_id].add(state_id)
            except (IndexError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: malformed G2O record") from error
    if not state_times or not landmark_ids:
        raise ValueError(f"{path}: no states or landmarks")
    distinct_states = []
    time_spans = []
    for landmark_id in sorted(landmark_ids):
        states = landmark_states[landmark_id]
        distinct_states.append(len(states))
        times = [state_times[state_id] for state_id in states if state_id in state_times]
        time_spans.append(max(times) - min(times) if times else 0.0)
    distinct_values = np.asarray(distinct_states, dtype=float)
    span_values = np.asarray(time_spans, dtype=float)
    return {
        "distinct_states_per_landmark_mean": float(np.mean(distinct_values)),
        "distinct_states_per_landmark_median": float(np.median(distinct_values)),
        "distinct_states_per_landmark_p95": float(np.percentile(distinct_values, 95)),
        "distinct_states_per_landmark_max": float(np.max(distinct_values)),
        "landmark_time_span_median_s": float(np.median(span_values)),
        "landmark_time_span_p95_s": float(np.percentile(span_values, 95)),
        "landmark_time_span_max_s": float(np.max(span_values)),
        "single_state_landmark_fraction": float(np.mean(distinct_values <= 1.0)),
    }


def landmark_quality_statistics(
    path: Path,
    *,
    weak_threshold: float = 0.001,
    initialization_threshold: float = 0.04,
) -> dict[str, float | int]:
    qualities = []
    with Path(path).open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if not fields or fields[0] != "VERTEX_TRACKXYZ":
                continue
            try:
                quality = float(fields[5])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed VERTEX_TRACKXYZ quality"
                ) from error
            if not np.isfinite(quality):
                raise ValueError(
                    f"{path}:{line_number}: non-finite landmark quality"
                )
            qualities.append(quality)
    if not qualities:
        raise ValueError(f"{path}: no VERTEX_TRACKXYZ quality values")
    values = np.asarray(qualities, dtype=float)
    return {
        "quality_count": int(len(values)),
        "quality_median": float(np.median(values)),
        "quality_p90": float(np.percentile(values, 90)),
        "quality_p95": float(np.percentile(values, 95)),
        "quality_fraction_above_0p001": float(
            np.mean(values > weak_threshold)
        ),
        "quality_initialized_fraction": float(
            np.mean(values >= initialization_threshold)
        ),
    }


def descriptive_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def threshold_statistics(
    values: np.ndarray, durations: np.ndarray, threshold: float
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if values.ndim != 1 or durations.shape != values.shape or not len(values):
        raise ValueError("values and durations must be matching non-empty vectors")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(durations)):
        raise ValueError("values and durations must be finite")
    if np.any(durations <= 0.0):
        raise ValueError("durations must be positive")
    active = values > threshold
    active_duration = float(np.sum(durations[active]))
    longest = 0.0
    current = 0.0
    for is_active, duration in zip(active, durations):
        if is_active:
            current += float(duration)
            longest = max(longest, current)
        else:
            current = 0.0
    starts = active & ~np.r_[False, active[:-1]]
    total_duration = float(np.sum(durations))
    return {
        "threshold": float(threshold),
        "duration_s": active_duration,
        "fraction": active_duration / total_duration,
        "longest_s": longest,
        "event_count": int(np.count_nonzero(starts)),
    }


def load_okvis_trajectory(path: Path) -> Trajectory:
    required = (
        "timestamp",
        "p_WS_W_x",
        "p_WS_W_y",
        "p_WS_W_z",
        "q_WS_x",
        "q_WS_y",
        "q_WS_z",
        "q_WS_w",
    )
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, skipinitialspace=True)
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
        has_velocity = all(
            field in (reader.fieldnames or [])
            for field in ("v_WS_W_x", "v_WS_W_y", "v_WS_W_z")
        )
        for row_number, row in enumerate(reader, 2):
            try:
                velocity = (
                    [float(row[f"v_WS_W_{axis}"]) for axis in "xyz"]
                    if has_velocity
                    else [float("nan")] * 3
                )
                rows.append(
                    [
                        float(row["timestamp"]) / 1e9,
                        *[float(row[f"p_WS_W_{axis}"]) for axis in "xyz"],
                        float(row["q_WS_w"]),
                        *[float(row[f"q_WS_{axis}"]) for axis in "xyz"],
                        *velocity,
                    ]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: invalid numeric value") from error
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than two trajectory poses")
    values = np.asarray(rows, dtype=float)
    _, first_indices = np.unique(values[:, 0], return_index=True)
    values = values[np.sort(first_indices)]
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    return Trajectory(values[:, 0], values[:, 1:4], values[:, 4:8], values[:, 8:11])


def load_imu(path: Path) -> ImuData:
    values = np.loadtxt(path, delimiter=",", comments="#", dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise ValueError(f"{path}: expected at least two EuRoC IMU rows with 7 fields")
    timestamps = values[:, 0] / 1e9
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path}: IMU timestamps are not strictly increasing")
    return ImuData(timestamps, values[:, 1:4], values[:, 4:7])


def load_mocap_trajectory(path: Path) -> Trajectory:
    poses = parse_mocap(Path(path))
    if len(poses) < 2:
        raise ValueError(f"{path}: fewer than two tracked mocap poses")
    timestamps = np.asarray([pose.timestamp for pose in poses], dtype=float)
    positions = np.asarray([pose.position for pose in poses], dtype=float)
    quaternions_xyzw = np.asarray([pose.quaternion for pose in poses], dtype=float)
    quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
    velocities = np.full_like(positions, np.nan)
    return Trajectory(timestamps, positions, quaternions_wxyz, velocities)


def analyze_mocap_integrity(path: Path) -> dict:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    time_pattern = re.compile(rf"\btime:\s*({number}).*\blatency:\s*({number})")
    tracking_pattern = re.compile(
        rf"\btracked:\s*([01]).*\bmean_error:\s*({number})"
    )
    timestamps = []
    latencies = []
    tracked = []
    mean_errors = []
    with Path(path).open(encoding="utf-8", errors="replace") as source:
        for line in source:
            time_match = time_pattern.search(line)
            if time_match:
                timestamps.append(float(time_match.group(1)))
                latencies.append(float(time_match.group(2)))
            tracking_match = tracking_pattern.search(line)
            if tracking_match:
                tracked.append(int(tracking_match.group(1)))
                mean_errors.append(float(tracking_match.group(2)))
    if len(timestamps) < 2 or len(tracked) != len(timestamps):
        raise ValueError(f"{path}: incomplete mocap integrity records")
    timestamps_array = np.asarray(timestamps, dtype=float)
    intervals = np.diff(timestamps_array)
    if np.any(intervals <= 0.0):
        raise ValueError(f"{path}: mocap timestamps are not strictly increasing")
    latency_stats = descriptive_statistics(np.asarray(latencies, dtype=float))
    error_stats = descriptive_statistics(np.asarray(mean_errors, dtype=float))
    median_interval = float(np.median(intervals))
    tracked_count = int(np.count_nonzero(tracked))
    return {
        "records": len(timestamps),
        "tracked_records": tracked_count,
        "tracking_loss_records": len(tracked) - tracked_count,
        "tracked_fraction": tracked_count / len(tracked),
        "duration_s": timestamps[-1] - timestamps[0],
        "median_interval_ms": median_interval * 1e3,
        "p99_interval_ms": float(np.percentile(intervals, 99) * 1e3),
        "max_interval_ms": float(np.max(intervals) * 1e3),
        "gap_count_over_1_5x": int(
            np.count_nonzero(intervals > 1.5 * median_interval)
        ),
        "mean_error_median_m": error_stats["median"],
        "mean_error_p95_m": error_stats["p95"],
        "mean_error_p99_m": error_stats["p99"],
        "mean_error_max_m": error_stats["max"],
        "latency_median_ms": latency_stats["median"],
        "latency_p95_ms": latency_stats["p95"],
        "latency_p99_ms": latency_stats["p99"],
        "latency_max_ms": latency_stats["max"],
    }


def to_evo_trajectory(trajectory: Trajectory) -> PoseTrajectory3D:
    return PoseTrajectory3D(
        trajectory.positions.copy(),
        trajectory.quaternions_wxyz.copy(),
        trajectory.timestamps.copy(),
    )


def evaluate_ape(
    reference: Trajectory, estimate: Trajectory, max_diff: float = 0.01
) -> AlignedEvaluation:
    reference_evo, estimate_evo = sync.associate_trajectories(
        to_evo_trajectory(reference),
        to_evo_trajectory(estimate),
        max_diff=max_diff,
        first_name="mocap",
        snd_name="OKVIS2-X",
    )
    estimate_evo.align(reference_evo, correct_scale=False)
    errors = np.linalg.norm(
        estimate_evo.positions_xyz - reference_evo.positions_xyz, axis=1
    )
    return AlignedEvaluation(
        reference_evo.timestamps.copy(),
        reference_evo.positions_xyz.copy(),
        reference_evo.orientations_quat_wxyz.copy(),
        estimate_evo.positions_xyz.copy(),
        estimate_evo.orientations_quat_wxyz.copy(),
        errors,
    )


def load_camera_index(path: Path) -> tuple[np.ndarray, list[str]]:
    timestamps = []
    filenames = []
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        for row_number, row in enumerate(reader, 1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{row_number}: expected timestamp and filename")
            timestamps.append(int(row[0]) / 1e9)
            filenames.append(row[1].strip())
    values = np.asarray(timestamps, dtype=float)
    if len(values) < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{path}: invalid camera timestamps")
    return values, filenames


def correct_camera_timestamps(
    timestamps: np.ndarray, image_delay: float
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.ndim != 1 or not np.all(np.isfinite(timestamps)):
        raise ValueError("camera timestamps must be a finite vector")
    if not np.isfinite(image_delay):
        raise ValueError("image delay must be finite")
    return timestamps - image_delay


def camera_sync_metrics(
    indices: dict[str, np.ndarray], tolerance_s: float = 0.001
) -> tuple[dict, np.ndarray]:
    if "cam0" not in indices or len(indices) < 2:
        raise ValueError("camera indices must contain cam0 and another camera")
    ordered = {}
    for name, timestamps in indices.items():
        values = np.asarray(timestamps, dtype=float)
        if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{name}: timestamps must be non-empty and increasing")
        ordered[name] = values
    timestamp_sets = {
        name: set(np.rint(values * 1e9).astype(np.int64))
        for name, values in ordered.items()
    }
    common = set.intersection(*timestamp_sets.values())
    base = ordered["cam0"]
    maximum_skews = np.zeros(len(base), dtype=float)
    for name, values in ordered.items():
        if name == "cam0":
            continue
        insertion = np.searchsorted(values, base)
        before = np.clip(insertion - 1, 0, len(values) - 1)
        after = np.clip(insertion, 0, len(values) - 1)
        nearest = np.minimum(np.abs(base - values[before]), np.abs(base - values[after]))
        maximum_skews = np.maximum(maximum_skews, nearest)
    names = sorted(ordered)
    pointers = np.zeros(len(names), dtype=np.int64)
    group_skews = []
    while all(pointers[index] < len(ordered[name]) for index, name in enumerate(names)):
        current = np.asarray(
            [ordered[name][pointers[index]] for index, name in enumerate(names)]
        )
        span = float(np.max(current) - np.min(current))
        if span <= tolerance_s:
            group_skews.append(span)
            pointers += 1
        else:
            pointers[int(np.argmin(current))] += 1
    group_skews_array = np.asarray(group_skews, dtype=float)
    metrics = {
        "common_exact_timestamps": len(common),
        "minimum_camera_frames": min(len(values) for values in ordered.values()),
        "maximum_camera_frames": max(len(values) for values in ordered.values()),
        "exact_sync_fraction": len(common) / max(len(values) for values in ordered.values()),
        "within_1ms_fraction": float(np.mean(maximum_skews <= 0.001)),
        "within_tolerance_fraction": float(np.mean(maximum_skews <= tolerance_s)),
        "nearest_skew_median_ms": float(np.median(maximum_skews) * 1e3),
        "nearest_skew_p95_ms": float(np.percentile(maximum_skews, 95) * 1e3),
        "nearest_skew_p99_ms": float(np.percentile(maximum_skews, 99) * 1e3),
        "nearest_skew_max_ms": float(np.max(maximum_skews) * 1e3),
        "sync_tolerance_ms": tolerance_s * 1e3,
        "one_to_one_sync_groups": len(group_skews),
        "one_to_one_sync_fraction": len(group_skews)
        / min(len(values) for values in ordered.values()),
        "one_to_one_skew_median_ms": float(np.median(group_skews_array) * 1e3),
        "one_to_one_skew_p95_ms": float(np.percentile(group_skews_array, 95) * 1e3),
        "one_to_one_skew_p99_ms": float(np.percentile(group_skews_array, 99) * 1e3),
        "one_to_one_skew_max_ms": float(np.max(group_skews_array) * 1e3),
    }
    return metrics, maximum_skews


def _grayscale_values(path: Path, max_width: int) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        if grayscale.width > max_width:
            height = max(1, round(grayscale.height * max_width / grayscale.width))
            grayscale = grayscale.resize((max_width, height), Image.Resampling.BILINEAR)
        return np.asarray(grayscale, dtype=np.float64)


def image_sharpness(path: Path, max_width: int = 640) -> float:
    values = _grayscale_values(path, max_width)
    return float(np.var(ndimage.laplace(values)))


def image_quality_metrics(path: Path, max_width: int = 640) -> dict[str, float]:
    values = _grayscale_values(path, max_width)
    return {
        "laplacian_variance": float(np.var(ndimage.laplace(values))),
        "intensity_mean": float(np.mean(values)),
        "intensity_std": float(np.std(values)),
        "intensity_p1": float(np.percentile(values, 1)),
        "intensity_p99": float(np.percentile(values, 99)),
        "dark_clip_fraction": float(np.mean(values < 5.0)),
        "bright_clip_fraction": float(np.mean(values > 250.0)),
    }


def mean_frame_difference(
    first_path: Path, second_path: Path, max_width: int = 640
) -> float:
    first = _grayscale_values(first_path, max_width)
    second = _grayscale_values(second_path, max_width)
    if first.shape != second.shape:
        raise ValueError("frame dimensions do not match")
    return float(np.mean(np.abs(first - second)))


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | tuple[str, ...] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot write empty CSV: {path}")
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join("---" for _ in headers) + "|")
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def percentile_fields(prefix: str, values: np.ndarray) -> dict[str, float]:
    stats = descriptive_statistics(values)
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def parse_image_delay(path: Path) -> float:
    pattern = re.compile(r"^\s*image_delay:\s*([0-9.eE+-]+)")
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            match = pattern.match(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"{path}: image_delay not found")


def summarize_stage(
    run: str,
    stage: str,
    trajectory: Trajectory,
    evaluation: AlignedEvaluation,
) -> dict:
    error_stats = descriptive_statistics(evaluation.errors)
    estimate_speed = linear_speed(trajectory.timestamps, trajectory.positions)
    estimate_angular = angular_speed(
        trajectory.timestamps, trajectory.quaternions_wxyz
    )
    steps = np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1)
    csv_velocity = np.linalg.norm(trajectory.velocities, axis=1)
    reference_distance = path_length(evaluation.reference_positions)
    maximum_step_index = int(np.argmax(steps))
    return {
        "run": run,
        "stage": stage,
        "poses": len(trajectory.timestamps),
        "associated_poses": len(evaluation.timestamps),
        "duration_s": trajectory.timestamps[-1] - trajectory.timestamps[0],
        "estimated_path_m": path_length(trajectory.positions),
        "mocap_path_m": reference_distance,
        "ape_rmse_m": float(np.sqrt(np.mean(evaluation.errors**2))),
        "ape_mean_m": error_stats["mean"],
        "ape_median_m": error_stats["median"],
        "ape_p95_m": float(np.percentile(evaluation.errors, 95)),
        "ape_p99_m": float(np.percentile(evaluation.errors, 99)),
        "ape_max_m": error_stats["max"],
        "ape_over_distance_percent": (
            100.0 * float(np.sqrt(np.mean(evaluation.errors**2))) / reference_distance
            if reference_distance
            else float("nan")
        ),
        "max_step_m": float(steps[maximum_step_index]),
        "max_step_time_s": (
            trajectory.timestamps[maximum_step_index + 1] - trajectory.timestamps[0]
        ),
        **percentile_fields("pose_speed_mps", estimate_speed),
        **percentile_fields("pose_angular_speed_radps", estimate_angular),
        **percentile_fields("csv_speed_mps", csv_velocity),
    }


def pairwise_repeatability(
    final_ba: dict[str, Trajectory], alignment_duration: float = 5.0
) -> tuple[list[dict], np.ndarray, list[str]]:
    names = sorted(final_ba)
    matrix = np.zeros((len(names), len(names)), dtype=float)
    rows = []
    for first_index, first in enumerate(names):
        for second_index in range(first_index + 1, len(names)):
            second = names[second_index]
            first_traj = final_ba[first]
            second_traj = final_ba[second]
            common, first_ids, second_ids = np.intersect1d(
                first_traj.timestamps,
                second_traj.timestamps,
                return_indices=True,
            )
            first_positions = first_traj.positions[first_ids]
            second_positions = second_traj.positions[second_ids]
            elapsed = common - common[0]
            fit = elapsed <= alignment_duration
            source_center = second_positions[fit].mean(axis=0)
            target_center = first_positions[fit].mean(axis=0)
            covariance = (
                (second_positions[fit] - source_center).T
                @ (first_positions[fit] - target_center)
            )
            u, _, vt = np.linalg.svd(covariance)
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0.0:
                vt[-1] *= -1.0
                rotation = vt.T @ u.T
            translation = target_center - rotation @ source_center
            aligned = (rotation @ second_positions.T).T + translation
            errors = np.linalg.norm(aligned - first_positions, axis=1)
            rmse = float(np.sqrt(np.mean(errors**2)))
            sustained_index = first_sustained_crossing(errors, 0.01, 30)
            matrix[first_index, second_index] = rmse
            matrix[second_index, first_index] = rmse
            rows.append(
                {
                    "run_a": first,
                    "run_b": second,
                    "common_poses": len(common),
                    "alignment_duration_s": alignment_duration,
                    "aligned_rmse_m": rmse,
                    "aligned_median_m": float(np.median(errors)),
                    "aligned_max_m": float(np.max(errors)),
                    "first_sustained_1cm_s": (
                        float(elapsed[sustained_index])
                        if sustained_index is not None
                        else float("nan")
                    ),
                }
            )
    return rows, matrix, names


def _descriptive_spearman(
    first: np.ndarray, second: np.ndarray
) -> float | None:
    if np.all(first == first[0]) or np.all(second == second[0]):
        return None
    return float(spearmanr(first, second).statistic)


def _format_descriptive_spearman(value: float | None) -> str:
    return "undefined (constant input)" if value is None else f"{value:.3f}"


def repeatability_statistics(
    run_rows: list[dict], pairwise_rows: list[dict]
) -> dict:
    if len(run_rows) < 2:
        raise ValueError("repeatability statistics require at least two runs")
    if not pairwise_rows:
        raise ValueError("repeatability statistics require non-empty pairwise rows")
    fields = (
        "ape_rmse_m",
        "states",
        "landmarks",
        "observations",
        "observations_per_landmark",
        "distinct_states_per_landmark_mean",
        "landmark_time_span_median_s",
    )
    try:
        values = {
            field: np.asarray([row[field] for row in run_rows], dtype=float)
            for field in fields
        }
        pairwise = np.asarray(
            [row["aligned_rmse_m"] for row in pairwise_rows], dtype=float
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("repeatability inputs must contain finite numeric values") from error
    if any(not np.all(np.isfinite(items)) for items in values.values()) or not np.all(
        np.isfinite(pairwise)
    ):
        raise ValueError("repeatability inputs must contain finite numeric values")
    ape = values["ape_rmse_m"]
    if np.any(ape <= 0.0):
        raise ValueError("APE values must be positive and finite")

    statistics = {
        "n_runs": len(run_rows),
        "ape_rmse_min_m": float(np.min(ape)),
        "ape_rmse_max_m": float(np.max(ape)),
        "ape_rmse_fold": float(np.max(ape) / np.min(ape)),
        "pairwise_aligned_rmse_min_m": float(np.min(pairwise)),
        "pairwise_aligned_rmse_max_m": float(np.max(pairwise)),
        "descriptive_only": True,
    }
    for field in (
        "states",
        "landmarks",
        "observations",
        "observations_per_landmark",
        "distinct_states_per_landmark_mean",
    ):
        statistics[f"{field}_min"] = float(np.min(values[field]))
        statistics[f"{field}_max"] = float(np.max(values[field]))
    span = values["landmark_time_span_median_s"]
    statistics.update(
        {
            "landmark_time_span_median_min_s": float(np.min(span)),
            "landmark_time_span_median_max_s": float(np.max(span)),
            "ape_observations_per_landmark_spearman": _descriptive_spearman(
                ape, values["observations_per_landmark"]
            ),
            "ape_distinct_states_per_landmark_mean_spearman": _descriptive_spearman(
                ape, values["distinct_states_per_landmark_mean"]
            ),
            "ape_landmark_time_span_median_spearman": _descriptive_spearman(
                ape, span
            ),
        }
    )
    return statistics


def analyze_imu(imu: ImuData, start: float, end: float) -> tuple[dict, dict]:
    mask = (imu.timestamps >= start) & (imu.timestamps <= end)
    timestamps = imu.timestamps[mask]
    gyroscope = imu.gyroscope[mask]
    accelerometer = imu.accelerometer[mask]
    intervals = np.diff(timestamps)
    gyro_norm = np.linalg.norm(gyroscope, axis=1)
    accel_norm = np.linalg.norm(accelerometer, axis=1)
    median_interval = float(np.median(intervals))
    frozen = np.all(
        np.diff(np.column_stack((gyroscope, accelerometer)), axis=0) == 0.0,
        axis=1,
    )
    summary = {
        "samples": len(timestamps),
        "duration_s": timestamps[-1] - timestamps[0],
        "mean_frequency_hz": (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]),
        "median_interval_ms": median_interval * 1e3,
        "p95_interval_ms": float(np.percentile(intervals, 95) * 1e3),
        "p99_interval_ms": float(np.percentile(intervals, 99) * 1e3),
        "interval_std_ms": float(np.std(intervals) * 1e3),
        "max_interval_ms": float(np.max(intervals) * 1e3),
        "gap_count_over_1_5x": int(np.count_nonzero(intervals > 1.5 * median_interval)),
        "gap_count_over_7_5ms": int(np.count_nonzero(intervals > 0.0075)),
        "gap_count_over_10ms": int(np.count_nonzero(intervals > 0.010)),
        "frozen_six_axis_count": int(np.count_nonzero(frozen)),
        "gyro_saturation_count": int(
            np.count_nonzero(np.any(np.abs(gyroscope) >= 7.8, axis=1))
        ),
        "accel_saturation_count": int(
            np.count_nonzero(np.any(np.abs(accelerometer) >= 176.0, axis=1))
        ),
        **percentile_fields("gyro_radps", gyro_norm),
        **percentile_fields("accel_mps2", accel_norm),
    }
    for sensor_name, values in (
        ("gyro", gyroscope),
        ("accel", accelerometer),
    ):
        for axis_index, axis in enumerate("xyz"):
            summary[f"{sensor_name}_{axis}_min"] = float(np.min(values[:, axis_index]))
            summary[f"{sensor_name}_{axis}_max"] = float(np.max(values[:, axis_index]))
    timeseries = {
        "timestamps": timestamps,
        "elapsed": timestamps - start,
        "gyro_norm": gyro_norm,
        "accel_norm": accel_norm,
    }
    return summary, timeseries


def analyze_cameras(
    dataset: Path,
    start: float,
    motion_times: np.ndarray,
    linear_motion: np.ndarray,
    angular_motion: np.ndarray,
    samples_per_camera: int,
    image_delay: float,
) -> tuple[list[dict], list[dict], dict]:
    indices = {}
    filenames = {}
    summary_rows = []
    sharpness_rows = []
    for camera_index in range(4):
        camera = f"cam{camera_index}"
        timestamps, names = load_camera_index(dataset / camera / "data.csv")
        corrected_timestamps = correct_camera_timestamps(timestamps, image_delay)
        indices[camera] = timestamps
        filenames[camera] = names
        intervals = np.diff(timestamps)
        median_interval = float(np.median(intervals))
        existing = sum(
            (dataset / camera / "data" / filename).is_file() for filename in names
        )
        sample_ids = np.unique(
            np.linspace(0, len(timestamps) - 1, min(samples_per_camera, len(timestamps)), dtype=int)
        )
        camera_quality = []
        for sample_index in sample_ids:
            image_path = dataset / camera / "data" / names[int(sample_index)]
            if not image_path.is_file():
                continue
            quality = image_quality_metrics(image_path)
            raw_timestamp = float(timestamps[int(sample_index)])
            timestamp = float(corrected_timestamps[int(sample_index)])
            linear_value = float(
                bounded_interpolate(timestamp, motion_times, linear_motion)
            )
            angular_value = float(
                bounded_interpolate(timestamp, motion_times, angular_motion)
            )
            previous_difference = float("nan")
            if sample_index > 0:
                previous_path = dataset / camera / "data" / names[int(sample_index) - 1]
                if previous_path.is_file():
                    previous_difference = mean_frame_difference(
                        previous_path, image_path
                    )
            camera_quality.append({**quality, "previous_frame_mae": previous_difference})
            sharpness_rows.append(
                {
                    "camera": camera,
                    "raw_timestamp_s": raw_timestamp,
                    "timestamp_s": timestamp,
                    "elapsed_s": timestamp - start,
                    **quality,
                    "previous_frame_mae": previous_difference,
                    "mocap_linear_speed_mps": linear_value,
                    "mocap_angular_speed_radps": angular_value,
                }
            )
        sharpness = np.asarray(
            [row["laplacian_variance"] for row in camera_quality], dtype=float
        )
        brightness = np.asarray(
            [row["intensity_mean"] for row in camera_quality], dtype=float
        )
        contrast = np.asarray(
            [row["intensity_std"] for row in camera_quality], dtype=float
        )
        frame_difference = np.asarray(
            [row["previous_frame_mae"] for row in camera_quality], dtype=float
        )
        frame_difference = frame_difference[np.isfinite(frame_difference)]
        sharpness_stats = descriptive_statistics(sharpness)
        summary_rows.append(
            {
                "camera": camera,
                "csv_frames": len(timestamps),
                "existing_images": existing,
                "missing_images": len(timestamps) - existing,
                "median_interval_ms": median_interval * 1e3,
                "max_interval_ms": float(np.max(intervals) * 1e3),
                "gap_count_over_1_5x": int(
                    np.count_nonzero(intervals > 1.5 * median_interval)
                ),
                "sharpness_samples": len(camera_quality),
                "sharpness_p1": float(np.percentile(sharpness, 1)),
                "sharpness_median": sharpness_stats["median"],
                "sharpness_p5": float(np.percentile(sharpness, 5)),
                "sharpness_p10": float(np.percentile(sharpness, 10)),
                "sharpness_p90": float(np.percentile(sharpness, 90)),
                "sharpness_min": float(np.min(sharpness)),
                "intensity_mean_median": float(np.median(brightness)),
                "intensity_std_median": float(np.median(contrast)),
                "dark_clip_fraction_mean": float(
                    np.mean([row["dark_clip_fraction"] for row in camera_quality])
                ),
                "bright_clip_fraction_mean": float(
                    np.mean([row["bright_clip_fraction"] for row in camera_quality])
                ),
                "previous_frame_mae_p5": float(np.percentile(frame_difference, 5)),
                "previous_frame_mae_median": float(np.median(frame_difference)),
                "previous_frame_mae_p95": float(np.percentile(frame_difference, 95)),
                "near_duplicate_frame_samples": int(
                    np.count_nonzero(frame_difference < 1.0)
                ),
            }
        )
    sync_metrics, _ = camera_sync_metrics(indices, tolerance_s=0.010)
    all_sharpness = np.asarray(
        [row["laplacian_variance"] for row in sharpness_rows], dtype=float
    )
    all_linear = np.asarray(
        [row["mocap_linear_speed_mps"] for row in sharpness_rows], dtype=float
    )
    all_angular = np.asarray(
        [row["mocap_angular_speed_radps"] for row in sharpness_rows], dtype=float
    )
    valid_linear = np.isfinite(all_linear)
    valid_angular = np.isfinite(all_angular)
    aggregate = {
        **sync_metrics,
        "sharpness_linear_speed_spearman": float(
            spearmanr(all_sharpness[valid_linear], all_linear[valid_linear]).statistic
        ),
        "sharpness_angular_speed_spearman": float(
            spearmanr(all_sharpness[valid_angular], all_angular[valid_angular]).statistic
        ),
    }
    return summary_rows, sharpness_rows, aggregate


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _elapsed(evaluation: AlignedEvaluation) -> np.ndarray:
    return evaluation.timestamps - evaluation.timestamps[0]


def _plot_ape_trajectories(
    output: Path,
    evaluations: dict[str, AlignedEvaluation],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    reference = next(iter(evaluations.values()))
    for name, evaluation in evaluations.items():
        color = RUN_COLORS[name]
        axes[0, 0].plot(
            evaluation.estimate_positions[:, 0],
            evaluation.estimate_positions[:, 1],
            color=color,
            linewidth=1.0,
            label=name,
        )
        axes[0, 1].plot(
            evaluation.estimate_positions[:, 0],
            evaluation.estimate_positions[:, 2],
            color=color,
            linewidth=1.0,
        )
        axes[0, 2].plot(
            evaluation.estimate_positions[:, 1],
            evaluation.estimate_positions[:, 2],
            color=color,
            linewidth=1.0,
        )
        axes[1, 0].plot(_elapsed(evaluation), evaluation.errors * 1000, color=color, linewidth=0.8)
        axes[1, 1].plot(
            _elapsed(evaluation),
            np.minimum(evaluation.errors * 1000, 500),
            color=color,
            linewidth=0.8,
        )
        axes[1, 2].plot(
            _elapsed(evaluation),
            np.cumsum(evaluation.errors**2) / np.arange(1, len(evaluation.errors) + 1),
            color=color,
            linewidth=0.8,
        )
    for axis, pair in zip(axes[0], ((0, 1), (0, 2), (1, 2))):
        axis.plot(
            reference.reference_positions[:, pair[0]],
            reference.reference_positions[:, pair[1]],
            color="#202124",
            linewidth=1.4,
            label="mocap" if pair == (0, 1) else None,
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="Aligned x-y trajectories")
    axes[0, 1].set(xlabel="x [m]", ylabel="z [m]", title="Aligned x-z trajectories")
    axes[0, 2].set(xlabel="y [m]", ylabel="z [m]", title="Aligned y-z trajectories")
    axes[1, 0].set(xlabel="Sequence time [s]", ylabel="APE [mm]", title="Translation APE (full range)")
    axes[1, 1].set(xlabel="Sequence time [s]", ylabel="APE [mm]", title="APE clipped at 500 mm")
    axes[1, 2].set(xlabel="Sequence time [s]", ylabel="Cumulative MSE [m$^2$]", title="Cumulative squared error")
    for axis in axes[1]:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle(
        "20260803-184537 final-BA: primary runs (bak0 excluded)", fontsize=14
    )
    _save_figure(figure, output / "01_ape_trajectories.png")


def _plot_stage_ape(output: Path, stage_rows: list[dict]) -> None:
    stages = list(STAGE_FILES)
    runs = sorted({row["run"] for row in stage_rows})
    values = {(row["run"], row["stage"]): row["ape_rmse_m"] * 1000 for row in stage_rows}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(runs))
    width = 0.24
    for index, stage in enumerate(stages):
        bars = axes[0].bar(
            x + (index - 1) * width,
            [values[(run, stage)] for run in runs],
            width,
            label=stage,
        )
        axes[0].bar_label(bars, fmt="%.1f", fontsize=7, rotation=90, padding=2)
    axes[0].set_xticks(x, runs)
    axes[0].set_yscale("log")
    axes[0].set(ylabel="APE RMSE [mm, log scale]", title="APE by output stage")
    axes[0].legend()
    for run in runs:
        run_rows = [row for row in stage_rows if row["run"] == run]
        axes[1].plot(
            stages,
            [next(row["ape_rmse_m"] for row in run_rows if row["stage"] == stage) * 1000 for stage in stages],
            marker="o",
            color=RUN_COLORS[run],
            label=run,
        )
    axes[1].set_yscale("log")
    axes[1].set(ylabel="APE RMSE [mm, log scale]", title="Backend stage can improve or amplify error")
    axes[1].legend(ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
    _save_figure(figure, output / "02_stage_ape.png")


def _plot_ape_and_motion(
    output: Path,
    evaluations: dict[str, AlignedEvaluation],
    motion: dict,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for name, evaluation in evaluations.items():
        axes[0].plot(_elapsed(evaluation), np.minimum(evaluation.errors * 1000, 500), color=RUN_COLORS[name], linewidth=0.8, label=name)
    axes[1].plot(motion["elapsed"], motion["linear_speed"], color="#147d92", linewidth=0.9)
    axes[2].plot(motion["elapsed"], motion["angular_speed"], color="#b3261e", linewidth=0.9)
    axes[0].set(ylabel="APE [mm]", title="Final-BA APE, clipped at 500 mm")
    axes[1].set(ylabel="Linear speed [m/s]", title="Mocap motion")
    axes[2].set(xlabel="Sequence time [s]", ylabel="Angular speed [rad/s]")
    axes[0].legend(ncol=5)
    for axis in axes:
        axis.grid(alpha=0.25)
    _save_figure(figure, output / "03_ape_and_motion.png")


def _plot_motion_dynamics(output: Path, motion: dict, imu_timeseries: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    axes[0, 0].plot(motion["elapsed"], motion["linear_speed"], color="#147d92", linewidth=0.8)
    axes[0, 1].plot(motion["elapsed"], motion["angular_speed"], color="#b3261e", linewidth=0.8)
    axes[1, 0].plot(imu_timeseries["elapsed"], imu_timeseries["accel_norm"], color="#16843b", linewidth=0.55)
    axes[1, 1].plot(imu_timeseries["elapsed"], imu_timeseries["gyro_norm"], color="#7357a5", linewidth=0.55)
    labels = (
        ("Mocap linear speed", "Speed [m/s]"),
        ("Mocap angular speed", "Rate [rad/s]"),
        ("Accelerometer norm", "Acceleration [m/s$^2$]"),
        ("Gyroscope norm", "Rate [rad/s]"),
    )
    for axis, (title, ylabel) in zip(axes.flat, labels):
        axis.set(title=title, ylabel=ylabel, xlabel="Sequence time [s]")
        axis.grid(alpha=0.25)
    _save_figure(figure, output / "04_motion_dynamics.png")


def _plot_pairwise(output: Path, matrix: np.ndarray, names: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    values = matrix * 1000
    image_handle = axis.imshow(values, cmap="YlOrRd")
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(column, row, f"{values[row, column]:.1f}", ha="center", va="center", fontsize=9)
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(names)), names)
    axis.set_title("Pairwise trajectory RMSE after first-5-s alignment [mm]")
    figure.colorbar(image_handle, ax=axis, label="RMSE [mm]")
    _save_figure(figure, output / "05_pairwise_repeatability.png")


def _plot_map_topology(output: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    runs = [row["run"] for row in rows]
    axes[0].bar(runs, [row["states"] for row in rows], color=[RUN_COLORS[name] for name in runs], label="states")
    axes[0].bar(runs, [row["camera_frames"] for row in rows], bottom=[row["states"] for row in rows], color="#9aa0a6", label="camera frames")
    axes[1].bar(runs, [row["landmarks"] for row in rows], color=[RUN_COLORS[name] for name in runs], label="landmarks")
    axes[1].bar(runs, [row["observations"] for row in rows], bottom=[row["landmarks"] for row in rows], color="#9aa0a6", label="observations")
    axes[0].set(title="Graph states and frames", ylabel="Count")
    axes[1].set(title="Landmarks and observations", ylabel="Count")
    for axis in axes:
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output / "06_map_topology.png")


def _plot_outlier_window(output: Path, evaluations: dict[str, AlignedEvaluation]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for name, evaluation in evaluations.items():
        elapsed = _elapsed(evaluation)
        mask = (elapsed >= 145) & (elapsed <= 162)
        axes[0].plot(elapsed[mask], evaluation.errors[mask] * 1000, color=RUN_COLORS[name], linewidth=1.0, label=name)
        axes[1].plot(elapsed[mask], evaluation.estimate_positions[mask, 2], color=RUN_COLORS[name], linewidth=1.0)
    axes[0].set(ylabel="APE [mm]", title="Error anomaly around 154 s")
    axes[1].set(xlabel="Sequence time [s]", ylabel="Aligned z [m]", title="Aligned vertical position")
    axes[0].legend(ncol=5)
    for axis in axes:
        axis.grid(alpha=0.25)
    _save_figure(figure, output / "07_outlier_window.png")


def _plot_image_sharpness(output: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for camera in sorted({row["camera"] for row in rows}):
        subset = [row for row in rows if row["camera"] == camera]
        axes[0].scatter([row["mocap_linear_speed_mps"] for row in subset], [row["laplacian_variance"] for row in subset], s=9, alpha=0.5, label=camera)
        axes[1].scatter([row["mocap_angular_speed_radps"] for row in subset], [row["laplacian_variance"] for row in subset], s=9, alpha=0.5, label=camera)
    axes[0].set(xlabel="Mocap linear speed [m/s]", ylabel="Laplacian variance", title="Sharpness vs linear speed")
    axes[1].set(xlabel="Mocap angular speed [rad/s]", ylabel="Laplacian variance", title="Sharpness vs angular speed")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    _save_figure(figure, output / "08_image_sharpness_motion.png")


def _plot_reference_comparison(output: Path, target_rows: list[dict], reference_rows: list[dict]) -> None:
    target = [row for row in target_rows if row["stage"] == "final-ba"]
    all_rows = target + reference_rows
    labels = [row.get("run", row.get("sequence")) for row in all_rows]
    colors = [RUN_COLORS.get(label, "#5f6368") for label in labels]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(labels, [row["ape_rmse_m"] * 1000 for row in all_rows], color=colors)
    axes[1].bar(labels, [row["ape_over_distance_percent"] for row in all_rows], color=colors)
    axes[0].set_yscale("log")
    axes[0].set(title="Absolute APE RMSE", ylabel="APE [mm, log scale]")
    axes[1].set(title="APE normalized by mocap path length", ylabel="APE / distance [%]")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25, which="both")
    _save_figure(figure, output / "09_reference_comparison.png")


def _motion_from_mocap(reference: Trajectory, start: float, end: float) -> dict:
    mask = (reference.timestamps >= start) & (reference.timestamps <= end)
    timestamps = reference.timestamps[mask]
    positions = reference.positions[mask]
    quaternions = reference.quaternions_wxyz[mask]
    durations = np.diff(timestamps)
    speed_times = 0.5 * (timestamps[:-1] + timestamps[1:])
    return {
        "timestamps": speed_times,
        "elapsed": speed_times - start,
        "durations": durations,
        "linear_speed": linear_speed(timestamps, positions),
        "angular_speed": angular_speed(timestamps, quaternions),
    }


def _reference_analysis(control_specs: list[SequenceSpec]) -> list[dict]:
    rows = []
    for spec in control_specs:
        trajectory = load_okvis_trajectory(
            spec.result_dir / STAGE_FILES["final-ba"]
        )
        reference = load_mocap_trajectory(spec.mocap)
        evaluation = evaluate_ape(reference, trajectory)
        row = summarize_stage(spec.name, "final-ba", trajectory, evaluation)
        motion = _motion_from_mocap(
            reference,
            float(evaluation.timestamps[0]),
            float(evaluation.timestamps[-1]),
        )
        motion_summary = summarize_motion(motion)
        row.update(
            {
                f"mocap_{key}": value
                for key, value in motion_summary.items()
                if key not in {"source", "samples", "duration_s"}
            }
        )
        row["sequence"] = row.pop("run")
        rows.append(row)
    return rows


def summarize_motion(motion: dict) -> dict:
    summary = {
        "source": "mocap",
        "samples": len(motion["timestamps"]),
        "duration_s": float(np.sum(motion["durations"])),
        **percentile_fields("linear_speed_mps", motion["linear_speed"]),
        **percentile_fields("angular_speed_radps", motion["angular_speed"]),
    }
    for metric, values, thresholds in (
        ("linear", motion["linear_speed"], (0.5, 0.7)),
        ("angular", motion["angular_speed"], (1.0, 2.0, 3.0, 4.0)),
    ):
        for threshold in thresholds:
            statistics = threshold_statistics(
                values, motion["durations"], threshold
            )
            token = str(threshold).replace(".", "_")
            for field in ("duration_s", "fraction", "longest_s", "event_count"):
                summary[f"{metric}_above_{token}_{field}"] = statistics[field]
    return summary


def motion_threshold_rows(sequence: str, motion: dict) -> list[dict]:
    rows = []
    for metric, unit, values, thresholds in (
        ("linear_speed", "m/s", motion["linear_speed"], (0.5, 0.7)),
        (
            "angular_speed",
            "rad/s",
            motion["angular_speed"],
            (1.0, 2.0, 3.0, 4.0),
        ),
    ):
        for threshold in thresholds:
            rows.append(
                {
                    "sequence": sequence,
                    "metric": metric,
                    "unit": unit,
                    **threshold_statistics(values, motion["durations"], threshold),
                }
            )
    return rows


def camera_gap_events(
    dataset: Path,
    sequence: str,
    start: float,
    image_delay: float,
) -> list[dict]:
    events = []
    for camera_index in range(4):
        camera = f"cam{camera_index}"
        timestamps, _ = load_camera_index(dataset / camera / "data.csv")
        intervals = np.diff(timestamps)
        median_interval = float(np.median(intervals))
        gap_interval_ids = np.flatnonzero(intervals > 1.5 * median_interval)
        short_interval_ids = np.flatnonzero(intervals < 0.5 * median_interval)
        paired_short_ids = {
            int(gap_id): int(short_id)
            for gap_id, short_id in zip(gap_interval_ids, short_interval_ids)
        }
        for interval_index in gap_interval_ids:
            raw_event_time = float(timestamps[interval_index] + median_interval)
            event_time = raw_event_time - image_delay
            next_interval = (
                float(intervals[interval_index + 1])
                if interval_index + 1 < len(intervals)
                else float("nan")
            )
            paired_short_index = paired_short_ids.get(int(interval_index))
            if paired_short_index is not None:
                paired_short_delay = float(
                    timestamps[paired_short_index] - raw_event_time
                )
                paired_short_interval = float(intervals[paired_short_index])
            else:
                paired_short_delay = float("nan")
                paired_short_interval = float("nan")
            events.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "raw_timestamp_s": raw_event_time,
                    "timestamp_s": event_time,
                    "elapsed_s": event_time - start,
                    "interval_ms": float(intervals[interval_index] * 1e3),
                    "median_interval_ms": median_interval * 1e3,
                    "following_interval_ms": next_interval * 1e3,
                    "immediately_followed_by_bunched_frame": bool(
                        np.isfinite(next_interval)
                        and next_interval < 0.5 * median_interval
                    ),
                    "paired_short_interval_delay_s": paired_short_delay,
                    "paired_short_interval_ms": paired_short_interval * 1e3,
                    "has_later_paired_short_interval": bool(
                        np.isfinite(paired_short_delay)
                        and paired_short_delay >= 0.0
                    ),
                }
            )
    return events


def map_topology_summary(result_dir: Path) -> dict:
    map_path = result_dir / "okvis2-slam-calib-final_map.g2o"
    counts = count_g2o_records(map_path)
    states = counts["VERTEX_SE3:QUAT_TIME"]
    camera_frames = counts["FRAME"]
    landmarks = counts["VERTEX_TRACKXYZ"]
    observations = counts["EDGE_OBS"]
    keypoints = counts["FRAME:KEYPOINT"]
    return {
        "states": states,
        "camera_frames": camera_frames,
        "landmarks": landmarks,
        "observations": observations,
        "keypoints": keypoints,
        "landmarks_per_state": landmarks / states,
        "observations_per_state": observations / states,
        "observations_per_landmark": observations / landmarks,
        "keypoints_per_camera_frame": keypoints / camera_frames,
        **map_track_statistics(map_path),
        **landmark_quality_statistics(map_path),
    }


def target_run_map_summary(result_dir: Path, final_row: dict) -> dict:
    map_path = result_dir / "okvis2-slam-calib-final_map.g2o"
    counts = count_g2o_records(map_path)
    return {
        **final_row,
        **map_topology_summary(result_dir),
        "imu_edges": counts["EDGE_IMU"],
        "imu_measurements": counts["EDGE_IMU:MEASUREMENTS"],
    }


def quality_bin_rows(quality_rows: list[dict]) -> list[dict]:
    rows = []
    edges = (0.0, 0.5, 1.0, 2.0, float("inf"))
    labels = ("0-0.5", "0.5-1", "1-2", ">=2")
    sequences = sorted({row["sequence"] for row in quality_rows})
    cameras = sorted({row["camera"] for row in quality_rows})
    for sequence in sequences:
        for camera in cameras:
            subset = [
                row
                for row in quality_rows
                if row["sequence"] == sequence and row["camera"] == camera
            ]
            angular = np.asarray(
                [row["mocap_angular_speed_radps"] for row in subset], dtype=float
            )
            sharpness = np.asarray(
                [row["laplacian_variance"] for row in subset], dtype=float
            )
            for lower, upper, label in zip(edges[:-1], edges[1:], labels):
                mask = (angular >= lower) & (angular < upper)
                values = sharpness[mask]
                rows.append(
                    {
                        "sequence": sequence,
                        "camera": camera,
                        "angular_speed_bin_radps": label,
                        "samples": int(np.count_nonzero(mask)),
                        "sharpness_p5": (
                            float(np.percentile(values, 5)) if len(values) else float("nan")
                        ),
                        "sharpness_median": (
                            float(np.median(values)) if len(values) else float("nan")
                        ),
                    }
                )
    return rows


def build_error_timeline(
    sequence: str,
    evaluation: AlignedEvaluation,
    motion: dict,
    quality_rows: list[dict],
    gap_events: list[dict],
    alignment_duration: float = 5.0,
) -> tuple[list[dict], dict]:
    _, prefix_errors = prefix_align_and_errors(
        evaluation.estimate_positions,
        evaluation.reference_positions,
        evaluation.timestamps,
        alignment_duration,
    )
    linear_values = np.interp(
        evaluation.timestamps, motion["timestamps"], motion["linear_speed"]
    )
    angular_values = np.interp(
        evaluation.timestamps, motion["timestamps"], motion["angular_speed"]
    )
    camera_sharpness = []
    for camera in sorted({row["camera"] for row in quality_rows}):
        subset = sorted(
            (row for row in quality_rows if row["camera"] == camera),
            key=lambda row: row["timestamp_s"],
        )
        camera_sharpness.append(
            np.interp(
                evaluation.timestamps,
                [row["timestamp_s"] for row in subset],
                [row["laplacian_variance"] for row in subset],
            )
        )
    median_sharpness = np.median(np.asarray(camera_sharpness), axis=0)
    near_camera_gap = np.zeros(len(evaluation.timestamps), dtype=bool)
    for event in gap_events:
        near_camera_gap |= np.abs(evaluation.timestamps - event["timestamp_s"]) <= 0.5
    intervals = np.diff(evaluation.timestamps)
    sample_rate = 1.0 / float(np.median(intervals))
    sustained_samples = max(1, int(round(sample_rate)))
    first_crossings = {}
    for threshold_mm in (10.0, 20.0, 50.0):
        index = first_sustained_crossing(
            prefix_errors, threshold_mm / 1000.0, sustained_samples
        )
        first_crossings[f"first_sustained_prefix_{int(threshold_mm)}mm_s"] = (
            float(evaluation.timestamps[index] - evaluation.timestamps[0])
            if index is not None
            else float("nan")
        )
    rolling_samples = sustained_samples
    rolling = np.sqrt(
        np.convolve(
            prefix_errors**2,
            np.ones(rolling_samples, dtype=float) / rolling_samples,
            mode="valid",
        )
    )
    rolling_times = evaluation.timestamps[rolling_samples - 1 :]
    lag = max(1, int(round(5.0 * sample_rate)))
    if len(rolling) > lag:
        increases = rolling[lag:] - rolling[:-lag]
        growth_index = int(np.argmax(increases)) + lag
        largest_growth_time = float(rolling_times[growth_index])
        largest_growth_mm = float(increases[growth_index - lag] * 1000.0)
        growth_pose_index = int(
            np.argmin(np.abs(evaluation.timestamps - largest_growth_time))
        )
        growth_window = (
            (evaluation.timestamps >= largest_growth_time - 5.0)
            & (evaluation.timestamps <= largest_growth_time)
        )
    else:
        largest_growth_time = float("nan")
        largest_growth_mm = float("nan")
        growth_pose_index = 0
        growth_window = np.zeros(len(evaluation.timestamps), dtype=bool)
        growth_window[0] = True
    growth_gap_events = sum(
        largest_growth_time - 5.0 <= event["timestamp_s"] <= largest_growth_time
        for event in gap_events
    ) if np.isfinite(largest_growth_time) else 0
    for row in quality_rows:
        row["global_ape_mm"] = float(
            bounded_interpolate(
                row["timestamp_s"], evaluation.timestamps, evaluation.errors
            )
            * 1000.0
        )
        row["prefix_error_mm"] = float(
            bounded_interpolate(
                row["timestamp_s"], evaluation.timestamps, prefix_errors
            )
            * 1000.0
        )
    rows = [
        {
            "sequence": sequence,
            "timestamp_s": float(timestamp),
            "elapsed_s": float(timestamp - evaluation.timestamps[0]),
            "global_ape_mm": float(global_error * 1000.0),
            "prefix_error_mm": float(prefix_error * 1000.0),
            "mocap_linear_speed_mps": float(linear_value),
            "mocap_angular_speed_radps": float(angular_value),
            "sampled_median_sharpness": float(sharpness),
            "within_0_5s_camera_gap": bool(near_gap),
        }
        for timestamp, global_error, prefix_error, linear_value, angular_value, sharpness, near_gap in zip(
            evaluation.timestamps,
            evaluation.errors,
            prefix_errors,
            linear_values,
            angular_values,
            median_sharpness,
            near_camera_gap,
        )
    ]
    summary = {
        "prefix_alignment_duration_s": alignment_duration,
        "prefix_error_rmse_m": float(np.sqrt(np.mean(prefix_errors**2))),
        "prefix_error_median_m": float(np.median(prefix_errors)),
        "prefix_error_p95_m": float(np.percentile(prefix_errors, 95)),
        "prefix_error_max_m": float(np.max(prefix_errors)),
        "ape_linear_speed_spearman": float(
            spearmanr(evaluation.errors, linear_values).statistic
        ),
        "ape_angular_speed_spearman": float(
            spearmanr(evaluation.errors, angular_values).statistic
        ),
        "ape_sharpness_spearman": float(
            spearmanr(evaluation.errors, median_sharpness).statistic
        ),
        "largest_5s_prefix_growth_elapsed_s": (
            largest_growth_time - evaluation.timestamps[0]
            if np.isfinite(largest_growth_time)
            else float("nan")
        ),
        "largest_5s_prefix_growth_mm": largest_growth_mm,
        "largest_growth_linear_speed_mps": float(linear_values[growth_pose_index]),
        "largest_growth_angular_speed_radps": float(angular_values[growth_pose_index]),
        "largest_growth_sampled_sharpness": float(
            median_sharpness[growth_pose_index]
        ),
        "largest_growth_near_camera_gap": bool(near_camera_gap[growth_pose_index]),
        "largest_growth_window_linear_speed_max_mps": float(
            np.max(linear_values[growth_window])
        ),
        "largest_growth_window_angular_speed_max_radps": float(
            np.max(angular_values[growth_window])
        ),
        "largest_growth_window_angular_speed_p95_radps": float(
            np.percentile(angular_values[growth_window], 95)
        ),
        "largest_growth_window_sharpness_min": float(
            np.min(median_sharpness[growth_window])
        ),
        "largest_growth_window_sharpness_median": float(
            np.median(median_sharpness[growth_window])
        ),
        "largest_growth_window_camera_gap_events": growth_gap_events,
        **first_crossings,
    }
    return rows, summary


def analyze_sequence(
    sequence: str,
    dataset: Path,
    result_dir: Path,
    mocap_path: Path,
    samples_per_camera: int,
    image_delay: float,
) -> dict:
    trajectory = load_okvis_trajectory(result_dir / STAGE_FILES["final-ba"])
    mocap = load_mocap_trajectory(mocap_path)
    evaluation = evaluate_ape(mocap, trajectory)
    start = float(evaluation.timestamps[0])
    end = float(evaluation.timestamps[-1])
    motion = _motion_from_mocap(mocap, start, end)
    motion_summary = summarize_motion(motion)
    imu = load_imu(dataset / "imu0" / "data.csv")
    imu_summary, imu_timeseries = analyze_imu(
        imu, float(imu.timestamps[0]), float(imu.timestamps[-1])
    )
    camera_rows, quality_rows, camera_aggregate = analyze_cameras(
        dataset,
        start,
        motion["timestamps"],
        motion["linear_speed"],
        motion["angular_speed"],
        samples_per_camera,
        image_delay,
    )
    for row in camera_rows:
        row.update({"sequence": sequence, **camera_aggregate})
    for row in quality_rows:
        row["sequence"] = sequence
    gaps = camera_gap_events(dataset, sequence, start, image_delay)
    error_rows, error_summary = build_error_timeline(
        sequence, evaluation, motion, quality_rows, gaps
    )
    map_summary = map_topology_summary(result_dir)
    mocap_summary = analyze_mocap_integrity(mocap_path)
    stage_summary = summarize_stage(sequence, "final-ba", trajectory, evaluation)
    camera0_timestamps, _ = load_camera_index(dataset / "cam0" / "data.csv")
    camera_aggregate["camera0_raw_minus_trajectory_ms"] = (
        camera0_timestamps[0] - trajectory.timestamps[0]
    ) * 1000.0
    camera_aggregate["camera0_corrected_minus_trajectory_ms"] = (
        camera0_timestamps[0] - image_delay - trajectory.timestamps[0]
    ) * 1000.0
    camera_aggregate["camera0_trajectory_offset_ms"] = camera_aggregate[
        "camera0_raw_minus_trajectory_ms"
    ]
    camera_aggregate["imu_lead_before_first_raw_camera_ms"] = (
        camera0_timestamps[0] - imu.timestamps[0]
    ) * 1000.0
    camera_aggregate["imu_tail_after_last_raw_camera_ms"] = (
        imu.timestamps[-1] - camera0_timestamps[-1]
    ) * 1000.0
    camera_aggregate["imu_lead_before_first_camera_ms"] = (
        camera0_timestamps[0] - image_delay - imu.timestamps[0]
    ) * 1000.0
    camera_aggregate["imu_tail_after_last_camera_ms"] = (
        imu.timestamps[-1] - (camera0_timestamps[-1] - image_delay)
    ) * 1000.0
    for row in camera_rows:
        row.update(
            {
                key: camera_aggregate[key]
                for key in (
                    "camera0_trajectory_offset_ms",
                    "camera0_raw_minus_trajectory_ms",
                    "camera0_corrected_minus_trajectory_ms",
                    "imu_lead_before_first_raw_camera_ms",
                    "imu_tail_after_last_raw_camera_ms",
                    "imu_lead_before_first_camera_ms",
                    "imu_tail_after_last_camera_ms",
                )
            }
        )
    summary = {
        "sequence": sequence,
        **{
            key: value
            for key, value in stage_summary.items()
            if key not in {"run", "stage"}
        },
        **map_summary,
        **{f"motion_{key}": value for key, value in motion_summary.items()},
        **{f"imu_{key}": value for key, value in imu_summary.items()},
        **{f"camera_{key}": value for key, value in camera_aggregate.items()},
        **{f"mocap_{key}": value for key, value in mocap_summary.items()},
        **error_summary,
    }
    return {
        "sequence": sequence,
        "dataset": dataset,
        "result_dir": result_dir,
        "mocap_path": mocap_path,
        "trajectory": trajectory,
        "evaluation": evaluation,
        "motion": motion,
        "motion_summary": motion_summary,
        "motion_threshold_rows": motion_threshold_rows(sequence, motion),
        "imu_summary": imu_summary,
        "imu_timeseries": imu_timeseries,
        "camera_rows": camera_rows,
        "camera_aggregate": camera_aggregate,
        "quality_rows": quality_rows,
        "gap_events": gaps,
        "error_rows": error_rows,
        "error_summary": error_summary,
        "map_summary": map_summary,
        "mocap_summary": mocap_summary,
        "summary": summary,
    }


def analyze_sequence_spec(
    spec: SequenceSpec, samples_per_camera: int, image_delay: float
) -> dict:
    context = analyze_sequence(
        spec.name,
        spec.dataset,
        spec.result_dir,
        spec.mocap,
        samples_per_camera,
        image_delay,
    )
    context.update({"role": spec.role, "color": spec.color, "spec": spec})
    return context


def finalize_quality_comparison(contexts: list[dict]) -> list[dict]:
    control_thresholds = pooled_control_sharpness_thresholds(contexts)
    all_camera_rows = []
    for context in contexts:
        all_sharpness = np.asarray(
            [row["laplacian_variance"] for row in context["quality_rows"]],
            dtype=float,
        )
        all_below = []
        for camera_row in context["camera_rows"]:
            camera = camera_row["camera"]
            values = np.asarray(
                [
                    row["laplacian_variance"]
                    for row in context["quality_rows"]
                    if row["camera"] == camera
                ],
                dtype=float,
            )
            threshold = control_thresholds[camera]
            below = values < threshold
            camera_row["control_sharpness_p5"] = threshold
            camera_row["fraction_below_control_p5"] = float(np.mean(below))
            camera_row["samples_below_control_p5"] = int(np.count_nonzero(below))
            all_below.extend(below.tolist())
            all_camera_rows.append(camera_row)
        context["summary"].update(
            {
                "image_sharpness_p5": float(np.percentile(all_sharpness, 5)),
                "image_sharpness_median": float(np.median(all_sharpness)),
                "image_fraction_below_control_camera_p5": float(
                    np.mean(all_below)
                ),
            }
        )
    return all_camera_rows


def _plot_sequence_motion_comparison(output: Path, contexts: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9))
    for context in contexts:
        sequence = context["sequence"]
        motion = context["motion"]
        progress = 100.0 * motion["elapsed"] / motion["elapsed"][-1]
        color = context["color"]
        label = sequence.replace("20260803-", "")
        axes[0, 0].plot(
            progress, motion["linear_speed"], color=color, linewidth=0.7, label=label
        )
        axes[0, 1].plot(
            progress, motion["angular_speed"], color=color, linewidth=0.7, label=label
        )
    x_linear = np.arange(2)
    x_angular = np.arange(3)
    width, offsets = grouped_bar_layout(len(contexts))
    for context, offset in zip(contexts, offsets):
        rows = context["motion_threshold_rows"]
        linear_rows = [row for row in rows if row["metric"] == "linear_speed"]
        angular_rows = [row for row in rows if row["metric"] == "angular_speed"]
        label = context["sequence"].replace("20260803-", "")
        color = context["color"]
        linear_bars = axes[1, 0].bar(
            x_linear + offset,
            [100.0 * row["fraction"] for row in linear_rows],
            width,
            color=color,
            label=label,
        )
        angular_bars = axes[1, 1].bar(
            x_angular + offset,
            [100.0 * row["fraction"] for row in angular_rows],
            width,
            color=color,
            label=label,
        )
        axes[1, 0].bar_label(linear_bars, fmt="%.2f", fontsize=8)
        axes[1, 1].bar_label(angular_bars, fmt="%.2f", fontsize=8)
    axes[0, 0].set(
        title="Mocap linear speed over normalized sequence time",
        xlabel="Sequence progress [%]",
        ylabel="Speed [m/s]",
    )
    axes[0, 1].set(
        title="Mocap angular speed over normalized sequence time",
        xlabel="Sequence progress [%]",
        ylabel="Angular speed [rad/s]",
    )
    axes[1, 0].set_xticks(x_linear, (">0.5", ">0.7"))
    axes[1, 0].set(
        title="Duration above linear-speed thresholds",
        ylabel="Evaluated duration [%]",
        xlabel="Threshold [m/s]",
    )
    axes[1, 1].set_xticks(x_angular, (">1", ">2", ">3"))
    axes[1, 1].set(
        title="Duration above angular-speed thresholds",
        ylabel="Evaluated duration [%]",
        xlabel="Threshold [rad/s]",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Matched-sequence motion comparison", fontsize=14)
    _save_figure(figure, output / "10_sequence_motion_comparison.png")


def _plot_sequence_sensor_timing(output: Path, contexts: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9))
    for context in contexts:
        sequence = context["sequence"]
        color = context["color"]
        label = sequence.replace("20260803-", "")
        imu = context["imu_timeseries"]
        intervals_ms = np.diff(imu["timestamps"]) * 1e3
        axes[0, 0].hist(
            intervals_ms,
            bins=np.linspace(4.8, 5.7, 90),
            histtype="step",
            density=True,
            linewidth=1.2,
            color=color,
            label=label,
        )
        sorted_gyro = np.sort(imu["gyro_norm"])
        axes[0, 1].plot(
            sorted_gyro,
            np.linspace(0.0, 100.0, len(sorted_gyro), endpoint=False),
            color=color,
            linewidth=1.2,
            label=label,
        )
    cameras = [f"cam{index}" for index in range(4)]
    x = np.arange(4)
    width, offsets = grouped_bar_layout(len(contexts))
    sync_labels = ("Exact", "10 ms one-to-one")
    sync_percentages = []
    for context, offset in zip(contexts, offsets):
        color = context["color"]
        label = context["sequence"].replace("20260803-", "")
        camera_rows = sorted(context["camera_rows"], key=lambda row: row["camera"])
        bars = axes[1, 0].bar(
            x + offset,
            [row["gap_count_over_1_5x"] for row in camera_rows],
            width,
            color=color,
            label=label,
        )
        axes[1, 0].bar_label(bars, fontsize=8)
        aggregate = context["camera_aggregate"]
        context_sync_percentages = [
            100.0 * aggregate["exact_sync_fraction"],
            100.0 * aggregate["one_to_one_sync_fraction"],
        ]
        sync_percentages.extend(context_sync_percentages)
        sync_bars = axes[1, 1].bar(
            np.arange(2) + offset,
            context_sync_percentages,
            width,
            color=color,
            label=label,
        )
        axes[1, 1].bar_label(sync_bars, fmt="%.2f", fontsize=8)
    axes[0, 0].set(
        title="IMU sample interval distribution",
        xlabel="Interval [ms]",
        ylabel="Density",
    )
    axes[0, 1].set(
        title="IMU gyroscope magnitude CDF",
        xlabel="Gyroscope magnitude [rad/s]",
        ylabel="Percentile [%]",
    )
    axes[1, 0].set_xticks(x, cameras)
    axes[1, 0].set(
        title="Camera long intervals (>1.5x median)",
        xlabel="Camera",
        ylabel="Count",
    )
    axes[1, 1].set_xticks(np.arange(2), sync_labels)
    axes[1, 1].set_ylim(*sync_percentage_axis_limits(sync_percentages))
    axes[1, 1].set(
        title="Four-camera timestamp association",
        ylabel="Associated frames [%]",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Sensor timing and integrity comparison", fontsize=14)
    _save_figure(figure, output / "11_sequence_sensor_timing.png")


def _plot_sequence_image_quality(output: Path, contexts: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9))
    cameras = [f"cam{index}" for index in range(4)]
    x = np.arange(4)
    width, offsets = grouped_bar_layout(len(contexts))
    for context, offset in zip(contexts, offsets):
        color = context["color"]
        label = context["sequence"].replace("20260803-", "")
        rows = sorted(context["camera_rows"], key=lambda row: row["camera"])
        median_bars = axes[0, 0].bar(
            x + offset,
            [row["sharpness_median"] for row in rows],
            width,
            color=color,
            alpha=0.82,
            label=label,
        )
        axes[0, 0].scatter(
            x + offset,
            [row["sharpness_p5"] for row in rows],
            color="#202124",
            s=18,
            zorder=3,
        )
        axes[0, 0].bar_label(median_bars, fmt="%.0f", fontsize=8)
        low_bars = axes[0, 1].bar(
            x + offset,
            [100.0 * row["fraction_below_control_p5"] for row in rows],
            width,
            color=color,
            label=label,
        )
        axes[0, 1].bar_label(low_bars, fmt="%.1f", fontsize=8)
    bin_labels = ("0-0.5", "0.5-1", "1-2", ">=2")
    bin_x = np.arange(4)
    for context, offset in zip(contexts, offsets):
        values = []
        for bin_index, (lower, upper) in enumerate(
            ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf")))
        ):
            subset = [
                row["laplacian_variance"]
                for row in context["quality_rows"]
                if lower <= row["mocap_angular_speed_radps"] < upper
            ]
            values.append(float(np.median(subset)) if subset else float("nan"))
        bars = axes[1, 0].bar(
            bin_x + offset,
            values,
            width,
            color=context["color"],
            label=context["sequence"].replace("20260803-", ""),
        )
        axes[1, 0].bar_label(bars, fmt="%.0f", fontsize=8)
    exposure_x = np.arange(2)
    for context, offset in zip(contexts, offsets):
        rows = context["camera_rows"]
        bars = axes[1, 1].bar(
            exposure_x + offset,
            [
                np.mean([row["intensity_mean_median"] for row in rows]),
                np.mean([row["intensity_std_median"] for row in rows]),
            ],
            width,
            color=context["color"],
            label=context["sequence"].replace("20260803-", ""),
        )
        axes[1, 1].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0, 0].set_xticks(x, cameras)
    axes[0, 0].set(
        title="Sampled sharpness median (black dot: p5)",
        ylabel="Laplacian variance",
    )
    axes[0, 1].set_xticks(x, cameras)
    axes[0, 1].set(
        title="Frames below the pooled control camera p5",
        ylabel="Sampled frames [%]",
    )
    axes[1, 0].set_xticks(bin_x, bin_labels)
    axes[1, 0].set(
        title="Sharpness controlled by mocap angular-speed bin",
        xlabel="Angular speed [rad/s]",
        ylabel="Median Laplacian variance",
    )
    axes[1, 1].set_xticks(exposure_x, ("Mean intensity", "Contrast std"))
    axes[1, 1].set(title="Exposure and contrast (camera mean)", ylabel="Gray level")
    for axis_index, axis in enumerate(axes.flat):
        axis.grid(alpha=0.25)
        if axis_index:
            axis.legend()
    sample_counts = [
        sum(row["camera"] == camera for row in context["quality_rows"])
        for context in contexts
        for camera in cameras
    ]
    sample_count_text = (
        str(sample_counts[0])
        if len(set(sample_counts)) == 1
        else f"{min(sample_counts)}-{max(sample_counts)}"
    )
    figure.suptitle(
        f"Image quality comparison ({sample_count_text} deterministic samples per camera)",
        fontsize=14,
    )
    _save_figure(figure, output / "12_sequence_image_quality.png")


def _plot_sequence_error_timeline(output: Path, contexts: list[dict]) -> None:
    figure, axes = plt.subplots(
        len(contexts),
        3,
        figsize=(18, max(9.0, 4.25 * len(contexts))),
        squeeze=False,
    )
    for row_index, context in enumerate(contexts):
        rows = context["error_rows"]
        elapsed = np.asarray([row["elapsed_s"] for row in rows])
        global_errors = np.asarray([row["global_ape_mm"] for row in rows])
        prefix_errors = np.asarray([row["prefix_error_mm"] for row in rows])
        axes[row_index, 0].plot(
            elapsed,
            global_errors,
            color="#147d92",
            linewidth=0.7,
            label="global SE(3) APE",
        )
        axes[row_index, 0].plot(
            elapsed,
            prefix_errors,
            color="#b3261e",
            linewidth=0.7,
            alpha=0.8,
            label="first-5-s aligned",
        )
        axes[row_index, 1].plot(
            elapsed,
            [row["mocap_angular_speed_radps"] for row in rows],
            color="#7357a5",
            linewidth=0.7,
        )
        axes[row_index, 2].plot(
            elapsed,
            [row["sampled_median_sharpness"] for row in rows],
            color="#16843b",
            linewidth=0.9,
        )
        for event in context["gap_events"]:
            if 0.0 <= event["elapsed_s"] <= elapsed[-1]:
                axes[row_index, 1].axvline(
                    event["elapsed_s"], color="#9aa0a6", alpha=0.08, linewidth=0.5
                )
        growth = context["error_summary"]["largest_5s_prefix_growth_elapsed_s"]
        for axis in axes[row_index]:
            axis.axvline(growth, color="#202124", linestyle="--", linewidth=0.8)
            axis.grid(alpha=0.25)
            axis.set_xlabel("Evaluated sequence time [s]")
        maximum_error = float(np.max(np.r_[global_errors, prefix_errors]))
        axes[row_index, 0].set_yscale("symlog", linthresh=10.0)
        axes[row_index, 0].set_ylim(0.0, maximum_error * 1.08)
        axes[row_index, 0].set_ylabel("Error [mm]")
        axes[row_index, 1].set_ylabel("Angular speed [rad/s]")
        axes[row_index, 2].set_ylabel("Laplacian variance")
        axes[row_index, 0].legend(fontsize=8)
        axes[row_index, 0].set_title(
            context["sequence"].replace("20260803-", "")
            + " trajectory error (full range, symlog)"
        )
        axes[row_index, 1].set_title("Mocap angular speed (gray: camera timing events)")
        axes[row_index, 2].set_title("Interpolated four-camera median sharpness")
    figure.suptitle(
        "Error, rotation and image quality timeline (dashed: largest 5 s error growth)",
        fontsize=14,
    )
    _save_figure(figure, output / "13_error_motion_quality_timeline.png")


def _plot_map_quality_comparison(output: Path, contexts: list[dict]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(19, 9))
    metrics = (
        ("states", "Keyframe states"),
        ("landmarks", "Landmarks"),
        ("observations", "Feature observations"),
        ("observations_per_landmark", "Observations / landmark"),
        ("distinct_states_per_landmark_mean", "Distinct states / landmark"),
        ("landmark_time_span_median_s", "Median landmark time span [s]"),
        ("keypoints_per_camera_frame", "Detected keypoints / camera frame"),
        ("ape_rmse_m", "APE RMSE [mm]"),
    )
    labels = [context["sequence"].replace("20260803-", "") for context in contexts]
    colors = [context["color"] for context in contexts]
    for axis, (field, title) in zip(axes.flat, metrics):
        values = [context["summary"][field] for context in contexts]
        if field == "ape_rmse_m":
            values = [value * 1000.0 for value in values]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.2f", fontsize=9)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Map topology and tracking efficiency", fontsize=14)
    _save_figure(figure, output / "14_map_quality_comparison.png")


def _plot_control_envelope_and_target_runs(
    output: Path, contexts: list[dict], run_rows: list[dict]
) -> None:
    rows = control_target_metric_rows(contexts, run_rows)
    figure, axes = plt.subplots(2, 2, figsize=(17, 10))
    metrics = (
        ("ape_rmse_mm", "APE RMSE [mm]"),
        ("observations_per_landmark", "Observations / landmark"),
        (
            "distinct_states_per_landmark_mean",
            "Mean distinct states / landmark",
        ),
        ("landmark_time_span_median_s", "Median landmark span [s]"),
    )
    x = np.arange(len(rows))
    labels = [row["label"] for row in rows]
    for axis, (field, title) in zip(axes.flat, metrics):
        values = np.asarray([row[field] for row in rows], dtype=float)
        axis_scale = control_metric_axis_scale(field, values)
        control_values = values[:2]
        axis.axhspan(
            float(np.min(control_values)),
            float(np.max(control_values)),
            color="#dadce0",
            alpha=0.65,
            zorder=0,
            label="control envelope",
        )
        for point_x, value, row in zip(x, values, rows):
            axis.scatter(
                point_x,
                value,
                color=row["color"],
                marker=row["marker"],
                edgecolor="#202124",
                linewidth=0.45,
                s=66,
                zorder=2,
            )
            axis.annotate(
                f"{value:.3f}",
                (point_x, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_yscale(axis_scale)
        axis.grid(axis="y", alpha=0.25, which="both")
        axis.margins(x=0.08, y=0.18)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Control envelopes and target repeatability runs", fontsize=14)
    _save_figure(figure, output / "15_control_envelope_and_target_runs.png")


def render_sequence_comparison_section(
    contexts: list[dict],
    camera_rows: list[dict],
    run_rows: list[dict],
    pairwise_rows: list[dict],
) -> str:
    controls, target = partition_sequence_contexts(contexts)
    controls = sorted(controls, key=lambda context: context["sequence"])
    ordered_contexts = controls + [target]
    summaries = [context["summary"] for context in ordered_contexts]
    cameras = sorted({row["camera"] for row in camera_rows})
    if not cameras:
        raise ValueError("sequence comparison requires camera rows")
    camera_by_key = {
        (row["sequence"], row["camera"]): row for row in camera_rows
    }
    expected_camera_keys = {
        (context["sequence"], camera)
        for context in ordered_contexts
        for camera in cameras
    }
    missing_camera_keys = sorted(expected_camera_keys - set(camera_by_key))
    if missing_camera_keys:
        raise ValueError(f"missing sequence camera rows: {missing_camera_keys}")

    overview_table = format_markdown_table(
        [
            "role",
            "sequence",
            "APE [mm]",
            "APE / path [%]",
            "linear p50/p95 [m/s]",
            "angular p50/p95 [rad/s]",
            "angular >1/>2 [s]",
        ],
        [
            [
                context["role"],
                context["sequence"],
                f"{summary['ape_rmse_m'] * 1000.0:.3f}",
                f"{summary['ape_over_distance_percent']:.4f}",
                f"{summary['motion_linear_speed_mps_median']:.3f} / {summary['motion_linear_speed_mps_p95']:.3f}",
                f"{summary['motion_angular_speed_radps_median']:.3f} / {summary['motion_angular_speed_radps_p95']:.3f}",
                f"{summary['motion_angular_above_1_0_duration_s']:.3f} / {summary['motion_angular_above_2_0_duration_s']:.3f}",
            ]
            for context, summary in zip(ordered_contexts, summaries)
        ],
    )
    ratio_table = format_markdown_table(
        [
            "control sequence",
            "target/control APE",
            "target/control normalized APE",
        ],
        [
            [
                control["sequence"],
                f"{target['summary']['ape_rmse_m'] / control['summary']['ape_rmse_m']:.2f}x",
                f"{target['summary']['ape_over_distance_percent'] / control['summary']['ape_over_distance_percent']:.2f}x",
            ]
            for control in controls
        ],
    )
    sensor_table = format_markdown_table(
        [
            "role",
            "sequence",
            "IMU samples / max dt [ms]",
            "IMU gaps / frozen / saturation",
            "camera missing / long intervals / sync [%]",
            "mocap tracked [%] / losses / median error [mm]",
        ],
        [
            [
                context["role"],
                context["sequence"],
                f"{summary['imu_samples']} / {summary['imu_max_interval_ms']:.3f}",
                f"{summary['imu_gap_count_over_7_5ms']} / {summary['imu_frozen_six_axis_count']} / {summary['imu_gyro_saturation_count'] + summary['imu_accel_saturation_count']}",
                f"{sum(camera_by_key[(context['sequence'], camera)]['missing_images'] for camera in cameras)} / {sum(camera_by_key[(context['sequence'], camera)]['gap_count_over_1_5x'] for camera in cameras)} / {summary['camera_one_to_one_sync_fraction'] * 100.0:.3f}",
                f"{summary['mocap_tracked_fraction'] * 100.0:.3f} / {summary['mocap_tracking_loss_records']} / {summary['mocap_mean_error_median_m'] * 1000.0:.3f}",
            ]
            for context, summary in zip(ordered_contexts, summaries)
        ],
    )
    image_table = format_markdown_table(
        [
            "role",
            "sequence",
            "camera",
            "sharpness median / p5",
            "pooled control p5",
            "below pooled control p5 [%]",
        ],
        [
            [
                context["role"],
                context["sequence"],
                camera,
                f"{camera_by_key[(context['sequence'], camera)]['sharpness_median']:.1f} / {camera_by_key[(context['sequence'], camera)]['sharpness_p5']:.1f}",
                f"{camera_by_key[(context['sequence'], camera)]['control_sharpness_p5']:.1f}",
                f"{camera_by_key[(context['sequence'], camera)]['fraction_below_control_p5'] * 100.0:.2f}",
            ]
            for context in ordered_contexts
            for camera in cameras
        ],
    )
    map_table = format_markdown_table(
        [
            "role",
            "sequence",
            "states",
            "landmarks",
            "observations",
            "observations / landmark",
            "mean distinct states / landmark",
            "median landmark span [s]",
            "keypoints / camera frame",
        ],
        [
            [
                context["role"],
                context["sequence"],
                str(summary["states"]),
                str(summary["landmarks"]),
                str(summary["observations"]),
                f"{summary['observations_per_landmark']:.3f}",
                f"{summary['distinct_states_per_landmark_mean']:.3f}",
                f"{summary['landmark_time_span_median_s']:.3f}",
                f"{summary['keypoints_per_camera_frame']:.1f}",
            ]
            for context, summary in zip(ordered_contexts, summaries)
        ],
    )

    metric_definitions = (
        ("APE RMSE [mm]", "ape_rmse_m", 1000.0, 3),
        ("normalized APE [%]", "ape_over_distance_percent", 1.0, 4),
        (
            "angular speed p50 [rad/s]",
            "motion_angular_speed_radps_median",
            1.0,
            3,
        ),
        (
            "angular speed p95 [rad/s]",
            "motion_angular_speed_radps_p95",
            1.0,
            3,
        ),
        (
            "angular time >1 rad/s [s]",
            "motion_angular_above_1_0_duration_s",
            1.0,
            3,
        ),
        (
            "angular time >2 rad/s [s]",
            "motion_angular_above_2_0_duration_s",
            1.0,
            3,
        ),
        (
            "linear time >0.5 m/s [s]",
            "motion_linear_above_0_5_duration_s",
            1.0,
            3,
        ),
        (
            "linear time >0.7 m/s [s]",
            "motion_linear_above_0_7_duration_s",
            1.0,
            3,
        ),
        ("IMU gaps >7.5 ms", "imu_gap_count_over_7_5ms", 1.0, 0),
        ("IMU frozen samples", "imu_frozen_six_axis_count", 1.0, 0),
        ("IMU gyro saturation", "imu_gyro_saturation_count", 1.0, 0),
        ("IMU accel saturation", "imu_accel_saturation_count", 1.0, 0),
        (
            "camera one-to-one sync [%]",
            "camera_one_to_one_sync_fraction",
            100.0,
            3,
        ),
        ("mocap tracked [%]", "mocap_tracked_fraction", 100.0, 3),
        ("mocap tracking losses", "mocap_tracking_loss_records", 1.0, 0),
        ("observations / landmark", "observations_per_landmark", 1.0, 3),
        (
            "mean distinct states / landmark",
            "distinct_states_per_landmark_mean",
            1.0,
            3,
        ),
        (
            "median landmark span [s]",
            "landmark_time_span_median_s",
            1.0,
            3,
        ),
        ("states", "states", 1.0, 0),
        ("keypoints / camera frame", "keypoints_per_camera_frame", 1.0, 1),
    )
    metric_values = []
    for label, field, scale, precision in metric_definitions:
        metric_values.append(
            (
                label,
                [control["summary"][field] * scale for control in controls],
                target["summary"][field] * scale,
                precision,
            )
        )
    metric_values.append(
        (
            "camera missing indexed images",
            [
                sum(
                    camera_by_key[(control["sequence"], camera)]["missing_images"]
                    for camera in cameras
                )
                for control in controls
            ],
            sum(
                camera_by_key[(target["sequence"], camera)]["missing_images"]
                for camera in cameras
            ),
            0,
        )
    )
    for camera in cameras:
        metric_values.append(
            (
                f"{camera} below pooled control p5 [%]",
                [
                    camera_by_key[(control["sequence"], camera)][
                        "fraction_below_control_p5"
                    ]
                    * 100.0
                    for control in controls
                ],
                camera_by_key[(target["sequence"], camera)][
                    "fraction_below_control_p5"
                ]
                * 100.0,
                2,
            )
        )

    envelope_rows = []
    outside_metrics = []
    within_metrics = []
    sync_status = None
    for label, control_values, target_value, precision in metric_values:
        status = control_envelope_status(control_values, target_value)
        number_format = f"{{:.{precision}f}}"
        envelope_rows.append(
            [
                label,
                f"{number_format.format(min(control_values))}-{number_format.format(max(control_values))}",
                number_format.format(target_value),
                status,
            ]
        )
        if status == "within":
            within_metrics.append(label)
        else:
            outside_metrics.append(f"{label} ({status})")
        if label == "camera one-to-one sync [%]":
            sync_status = status
    envelope_table = format_markdown_table(
        ["metric", "control range", "target", "target status"], envelope_rows
    )
    outside_text = ", ".join(outside_metrics) if outside_metrics else "none"
    within_text = ", ".join(within_metrics) if within_metrics else "none"
    sync_interpretation = (
        "Above-envelope camera one-to-one synchronization is favorable or neutral "
        "and is not evidence of worse acquisition timing."
        if sync_status == "above"
        else ""
    )

    repeatability = repeatability_statistics(run_rows, pairwise_rows)
    ordered_runs = sorted(run_rows, key=lambda row: row["run"])
    repeatability_table = format_markdown_table(
        [
            "run",
            "APE [mm]",
            "states",
            "landmarks",
            "observations",
            "observations / landmark",
            "mean distinct states / landmark",
            "median landmark span [s]",
        ],
        [
            [
                row["run"],
                f"{row['ape_rmse_m'] * 1000.0:.3f}",
                str(row["states"]),
                str(row["landmarks"]),
                str(row["observations"]),
                f"{row['observations_per_landmark']:.3f}",
                f"{row['distinct_states_per_landmark_mean']:.3f}",
                f"{row['landmark_time_span_median_s']:.3f}",
            ]
            for row in ordered_runs
        ],
    )
    best_run = min(run_rows, key=lambda row: row["ape_rmse_m"])
    observations_leader = max(
        run_rows, key=lambda row: row["observations_per_landmark"]
    )
    distinct_states_leader = max(
        run_rows, key=lambda row: row["distinct_states_per_landmark_mean"]
    )
    if best_run["run"] == observations_leader["run"] == distinct_states_leader["run"]:
        leader_text = (
            f"`{best_run['run']}` has the lowest APE and the highest observations per "
            "landmark and mean distinct states per landmark"
        )
    else:
        leader_text = (
            f"`{best_run['run']}` has the lowest APE; "
            f"`{observations_leader['run']}` has the highest observations per landmark; "
            f"`{distinct_states_leader['run']}` has the highest mean distinct states per landmark"
        )
    span_rho = repeatability["ape_landmark_time_span_median_spearman"]
    if span_rho is None:
        span_order_text = (
            "the median-span/APE rank correlation is undefined (constant input)"
        )
    elif np.isclose(abs(span_rho), 1.0):
        span_order_text = "the median-span ordering is monotonic with APE"
    else:
        span_order_text = "the median-span ordering is not monotonic with APE"
    persistence_correlations = ", ".join(
        _format_descriptive_spearman(repeatability[field])
        for field in (
            "ape_observations_per_landmark_spearman",
            "ape_distinct_states_per_landmark_mean_spearman",
            "ape_landmark_time_span_median_spearman",
        )
    )

    control_names = ", ".join(f"`{control['sequence']}`" for control in controls)
    return f"""
## Sequence control-envelope comparison

This section uses the control set {control_names}; the unique target is `{target['sequence']}`. All sequences use identical parsing, deterministic image sampling, evo association, and rigid alignment without scale correction.

### Result and motion

{overview_table}

{ratio_table}

### Acquisition integrity

{sensor_table}

The indexed-stream checks cover IMU gaps, frozen samples and saturation, camera-file completeness and timestamp association, and mocap tracking completeness. Absence of a systematic anomaly in these global checks does not prove absolute physical synchronization or exclude a localized timing error.

### Image quality

{image_table}

For each camera, `control_sharpness_p5` is the fifth percentile of raw samples pooled across all controls, not an average of per-control percentiles. `fraction_below_control_p5` applies that common camera threshold to each sequence. Camera names in this table are derived from the data.

### Tracking and map structure

{map_table}

Observations per landmark are `EDGE_OBS / VERTEX_TRACKXYZ`; mean distinct states count the unique `EDGE_OBS` state IDs retained for each landmark; median landmark span is the median difference between the latest and earliest retained state timestamps. These describe the final G2O graph, not online feature-track lifetime.

### Closed control-envelope classification

{envelope_table}

Target-distinctive metrics, defined strictly as below the minimum control or above the maximum control: {outside_text}.

Metrics within the closed control envelope, including equality at either boundary: {within_text}.

{sync_interpretation}

Stronger or more frequent rotation has a physically plausible path to low sharpness because angular motion causes pixel displacement during an exposure and can produce blur. Laplacian variance also depends on scene texture and viewpoint. These sequences are not frame-wise repeats, so their correlations support only that rotational stress and a low-sharpness tail co-occur and may contribute; they cannot quantitatively separate or prove a single cause.

### Four-run instability diagnosis

{repeatability_table}

Across n={repeatability['n_runs']}, final-BA APE spans **{repeatability['ape_rmse_min_m'] * 1000.0:.3f}-{repeatability['ape_rmse_max_m'] * 1000.0:.3f} mm**, a **{repeatability['ape_rmse_fold']:.2f}x** fold range. Pairwise first-5-s aligned RMSE spans **{repeatability['pairwise_aligned_rmse_min_m'] * 1000.0:.1f}-{repeatability['pairwise_aligned_rmse_max_m'] * 1000.0:.1f} mm** and does not use mocap. Final graphs span {repeatability['states_min']:.0f}-{repeatability['states_max']:.0f} states, {repeatability['landmarks_min']:.0f}-{repeatability['landmarks_max']:.0f} landmarks, and {repeatability['observations_min']:.0f}-{repeatability['observations_max']:.0f} observations. Persistence spans {repeatability['observations_per_landmark_min']:.3f}-{repeatability['observations_per_landmark_max']:.3f} observations per landmark, {repeatability['distinct_states_per_landmark_mean_min']:.3f}-{repeatability['distinct_states_per_landmark_mean_max']:.3f} mean distinct states per landmark, and {repeatability['landmark_time_span_median_min_s']:.3f}-{repeatability['landmark_time_span_median_max_s']:.3f} s median landmark span.

{leader_text}, while {span_order_text}. The n=4 Spearman coefficients of APE against observations per landmark, mean distinct states per landmark, and median span are {persistence_correlations}. They are descriptive only and cannot establish significance or causality.

All four runs process the same indexed sensor input. Their different final G2O graph structures place the run-to-run instability inside estimator execution; sensor data cannot explain the run-to-run spread. Source-visible amplifiers include OpenGV's default `randomSeed=true` path using `time(0) + clock()`, the `enforce_realtime` wall-clock optimization budget, asynchronous realtime/full-graph interaction, parallel floating-point reductions, and discrete matching or acceptance thresholds. These mechanisms can amplify small differences, but the available final files do not identify the unique first divergence.
"""


def append_sequence_comparison_report(
    path: Path,
    contexts: list[dict],
    camera_rows: list[dict],
    run_rows: list[dict],
    pairwise_rows: list[dict],
) -> None:
    section = render_sequence_comparison_section(
        contexts, camera_rows, run_rows, pairwise_rows
    )
    existing = path.read_text(encoding="utf-8")
    for marker in (
        "\n## Sequence control-envelope comparison\n",
        "\n## Three-sequence control-envelope comparison\n",
        "\n## Matched-sequence quantitative comparison\n",
    ):
        existing = existing.split(marker, 1)[0]
    path.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")


def _write_report(
    path: Path,
    stage_rows: list[dict],
    run_rows: list[dict],
    reference_rows: list[dict],
    pairwise_rows: list[dict],
    imu_summary: dict,
    camera_rows: list[dict],
    camera_aggregate: dict,
    image_delay: float,
    motion_summary: dict,
    excluded_summary: dict,
) -> None:
    finals = sorted((row for row in stage_rows if row["stage"] == "final-ba"), key=lambda row: row["run"])
    best = min(finals, key=lambda row: row["ape_rmse_m"])
    worst = max(finals, key=lambda row: row["ape_rmse_m"])
    map_states = [row["states"] for row in run_rows]
    map_landmarks = [row["landmarks"] for row in run_rows]
    controls = sorted(reference_rows, key=lambda row: row["sequence"])
    control_ape_mm = np.asarray(
        [row["ape_rmse_m"] * 1000.0 for row in controls], dtype=float
    )
    control_normalized = np.asarray(
        [row["ape_over_distance_percent"] for row in controls], dtype=float
    )
    control_linear_medians = np.asarray(
        [row["mocap_linear_speed_mps_median"] for row in controls], dtype=float
    )
    control_angular_medians = np.asarray(
        [row["mocap_angular_speed_radps_median"] for row in controls], dtype=float
    )
    control_names = " and ".join(f"`{row['sequence']}`" for row in controls)
    target_normalized = best["ape_over_distance_percent"]
    target_normalized_status = control_envelope_status(
        control_normalized, target_normalized
    )
    repeatability = repeatability_statistics(run_rows, pairwise_rows)
    stage_table = format_markdown_table(
        ["run", "online [mm]", "final [mm]", "final-BA [mm]", "distance [m]", "final-BA [%]"],
        [[
            run,
            *[f"{next(row for row in stage_rows if row['run'] == run and row['stage'] == stage)['ape_rmse_m'] * 1000:.3f}" for stage in STAGE_FILES],
            f"{next(row for row in finals if row['run'] == run)['mocap_path_m']:.3f}",
            f"{next(row for row in finals if row['run'] == run)['ape_over_distance_percent']:.4f}",
        ] for run in sorted(row["run"] for row in finals)],
    )
    control_table = format_markdown_table(
        ["sequence", "distance [m]", "APE RMSE [mm]", "APE / distance [%]"],
        [[row["sequence"], f"{row['mocap_path_m']:.3f}", f"{row['ape_rmse_m'] * 1000:.3f}", f"{row['ape_over_distance_percent']:.4f}"] for row in controls],
    )
    camera_table = format_markdown_table(
        ["camera", "frames", "missing", "max interval [ms]", "sharpness median", "sharpness p5"],
        [[row["camera"], str(row["csv_frames"]), str(row["missing_images"]), f"{row['max_interval_ms']:.3f}", f"{row['sharpness_median']:.1f}", f"{row['sharpness_p5']:.1f}"] for row in camera_rows],
    )
    motion_correlations = [abs(row["ape_angular_speed_spearman"]) for row in run_rows]
    persistence_correlations = ", ".join(
        _format_descriptive_spearman(repeatability[field])
        for field in (
            "ape_observations_per_landmark_spearman",
            "ape_distinct_states_per_landmark_mean_spearman",
            "ape_landmark_time_span_median_spearman",
        )
    )
    report = f"""# 20260803-184537 OKVIS2-X repeatability analysis

## Executive conclusion

The primary analysis excludes `bak0`, whose isolated final-BA jump reaches {excluded_summary['ape_max_m']:.3f} m and is an obvious backend outlier. The remaining four outputs are genuinely non-deterministic before evo: they contain the same {finals[0]['poses']} unique output timestamps, but the saved graphs contain {min(map_states)}-{max(map_states)} states and {min(map_landmarks)}-{max(map_landmarks)} landmarks. Because evo is a deterministic downstream evaluator, it cannot create these map differences.

Final-BA APE among `bak1`-`bak4` ranges from **{best['ape_rmse_m'] * 1000:.3f} mm ({best['run']})** to **{worst['ape_rmse_m'] * 1000:.3f} mm ({worst['run']})**, a {worst['ape_rmse_m'] / best['ape_rmse_m']:.1f}x spread. This remains far beyond numerical noise after removing the outlier.

The controls are {control_names}. Control APE RMSE spans **{np.min(control_ape_mm):.3f}-{np.max(control_ape_mm):.3f} mm**, and normalized APE spans **{np.min(control_normalized):.4f}-{np.max(control_normalized):.4f}%**. Their median mocap linear speed spans {np.min(control_linear_medians):.3f}-{np.max(control_linear_medians):.3f} m/s and median angular speed spans {np.min(control_angular_medians):.3f}-{np.max(control_angular_medians):.3f} rad/s.

The best target run has normalized error **{target_normalized:.4f}%** over a {best['mocap_path_m']:.3f} m path. It is **{target_normalized_status}** the closed control normalized-error range. Reporting both controls avoids treating a difference from one control as a target-specific property.

## Quantitative results

{stage_table}

Pairwise final-BA RMSE after aligning only the first 5 s ranges from {repeatability['pairwise_aligned_rmse_min_m'] * 1000:.1f} to {repeatability['pairwise_aligned_rmse_max_m'] * 1000:.1f} mm. This measures run-to-run trajectory divergence independently of mocap.

Across the four final graphs, states span {repeatability['states_min']:.0f}-{repeatability['states_max']:.0f}, landmarks {repeatability['landmarks_min']:.0f}-{repeatability['landmarks_max']:.0f}, and observations {repeatability['observations_min']:.0f}-{repeatability['observations_max']:.0f}. Retained-observation persistence spans {repeatability['observations_per_landmark_min']:.3f}-{repeatability['observations_per_landmark_max']:.3f} observations per landmark, {repeatability['distinct_states_per_landmark_mean_min']:.3f}-{repeatability['distinct_states_per_landmark_mean_max']:.3f} mean distinct states, and {repeatability['landmark_time_span_median_min_s']:.3f}-{repeatability['landmark_time_span_median_max_s']:.3f} s median landmark span. With n=4, their APE Spearman coefficients are {persistence_correlations}; these are descriptive only, not significance tests or causal estimates.

Across the four runs, the absolute within-run Spearman correlation between per-pose final-BA translation error and temporally interpolated mocap angular speed ranges from {min(motion_correlations):.3f} to {max(motion_correlations):.3f}. Since the same motion produces very different error histories, motion is a stressor but not a sufficient explanation for run-to-run variation.

## Motion and sensor integrity

During the evaluated interval, mocap linear speed has median {motion_summary['linear_speed_mps_median']:.3f} m/s, p95 {motion_summary['linear_speed_mps_p95']:.3f} m/s and maximum {motion_summary['linear_speed_mps_max']:.3f} m/s. Mocap angular speed has median {motion_summary['angular_speed_radps_median']:.3f} rad/s, p95 {motion_summary['angular_speed_radps_p95']:.3f} rad/s and maximum {motion_summary['angular_speed_radps_max']:.3f} rad/s. Figure 04 provides mocap and IMU motion time series; `motion_summary.csv` contains the exact statistics.

The complete IMU stream contains {imu_summary['samples']} samples over {imu_summary['duration_s']:.3f} s. Median interval is {imu_summary['median_interval_ms']:.3f} ms, maximum interval is {imu_summary['max_interval_ms']:.3f} ms, gaps over 10 ms: {imu_summary['gap_count_over_10ms']}, gyro saturation count: {imu_summary['gyro_saturation_count']}, accelerometer saturation count: {imu_summary['accel_saturation_count']}. This does not support IMU acquisition loss as the cause.

The configured `image_delay` is {image_delay * 1000:.6f} ms; the first raw camera0 timestamp leads the first output trajectory timestamp by {camera_aggregate['camera0_trajectory_offset_ms']:.6f} ms. This confirms the configured software timestamp correction was reflected in the output, not physical synchronization accuracy. Across four cameras, {camera_aggregate['common_exact_timestamps']} timestamps are exactly common, or {camera_aggregate['exact_sync_fraction'] * 100:.3f}% relative to the largest camera stream. All indexed files were checked for existence. Sharpness is evaluated on deterministic uniform samples, not every frame; Spearman correlation is {camera_aggregate['sharpness_linear_speed_spearman']:.3f} with linear speed and {camera_aggregate['sharpness_angular_speed_spearman']:.3f} with angular speed.

{camera_table}

These checks find no missing indexed image files, material IMU gaps, saturation, or corrupt sampled images. The camera indices do contain a small number of long/short interval pairs, and sharpness decreases with angular speed, so local motion-quality or timing windows may affect absolute accuracy. However, the indexed sensor stream is identical across runs and cannot explain the four-way repeatability spread; that spread is introduced by estimator execution.

## Control sequences

{control_table}

## Source-level causes of non-determinism

The configuration and source contain several mechanisms that amplify small timing and random differences:

1. OpenGV sample consensus defaults to `randomSeed=true`, which seeds from `time(0) + clock()`; `false` uses fixed seed `12345u`. The four inspected OKVIS call sites do not pass a fixed-seed choice, so their RANSAC hypotheses are source-visible random inputs.
2. With `enforce_realtime`, `ThreadedSlam.cpp` starts timing before state insertion, frontend association, GPS handling and queue work, then gives optimization `0.035 s - elapsed`; an exhausted budget is clamped to 10 ms. The callback also guarantees at least three iterations. This is a wall-clock-sensitive optimization budget, not a pure matching or whole-frame hard cutoff.
3. Realtime and full-graph optimization interact asynchronously. Atomic state and query/import gates can alter later loop candidates or graph updates depending on completion timing; the final files do not show that this specific path caused any one observed divergence.
4. The source hard-codes a 100-iteration limit for each of the two final-BA optimisation passes. Its thread count is the sum of configured realtime and full-graph thread counts, 12 (8+4) in the inspected YAML. Other Ceres paths are also multithreaded. Parallel floating-point reductions and threshold branches can amplify small differences.
5. Detection and matching can run in parallel, while keyframe overlap, DBoW probability, RANSAC inlier count and inlier-ratio checks are discrete thresholds. Small upstream differences can therefore switch later decisions.

These are mechanisms present in the inspected source, not a unique attribution. The four runs have no preserved binary/configuration/runtime manifest, so the current checkout cannot prove which build, settings, thread paths or first branch each run used. The exact first divergence cannot be reconstructed from final files alone.

## Recommended isolation experiment

Use the same dataset, executable and machine load; change one variable at a time and run each variant at least five times:

1. Set only `enforce_realtime: 0`.
2. Set `parallelise_detection: 0` and, in a separate variant, `num_matching_threads: 1`; disabling detection parallelism alone does not serialize matching.
3. Set realtime and full-graph Ceres thread counts to 1; temporarily set `do_final_ba: 0` to separate online repeatability from final BA.
4. Set only `do_loop_closures: 0` to test whether the main spread is loop-closure driven.
5. Change the Frame sample-consensus wrappers to pass `randomSeed=false` (OpenGV's fixed `12345u` path) and repeat; there is no YAML seed field.

Record logs and save maps for every run. The first variant that collapses both graph-count spread and APE spread implicates that change as a contributor; confirm by re-enabling it and with randomized/factorial follow-up runs. Do not combine all changes in the first experiment.

## Reproduce

From the repository root, run the production analysis and its unit tests in the `okvis2x` environment:

```bash
cd workspace/ego2_results/20260803-184537/analysis
conda run -n okvis2x python analyze_repeatability.py
conda run -n okvis2x python -m unittest -v test_analyze_repeatability.py
```

The `okvis2x` environment is required because the system Python does not provide `evo`.

## Artifacts and definitions

- APE: evo-style timestamp association (`max_diff=0.01 s`) followed by rigid SE(3) alignment, no scale correction.
- Path-normalized error: `100 * APE_RMSE / associated mocap path length`.
- Pairwise repeatability: common output timestamps, second trajectory rigidly aligned to the first using only the first 5 s.
- Image sharpness: variance of a Laplacian after grayscale resize to at most 640 pixels width.
- CSV files contain all values used for the report and figures.
"""
    path.write_text(report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze repeated OKVIS2-X runs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--control-dataset-183537",
        "--reference-dataset",
        dest="control_dataset_183537",
        type=Path,
        default=CONTROL_DATASET_183537,
    )
    parser.add_argument(
        "--control-dataset-184027",
        type=Path,
        default=CONTROL_DATASET_184027,
    )
    parser.add_argument("--mocap", type=Path, default=DEFAULT_MOCAP)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-samples-per-camera", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_runs = sorted(path for path in arguments.results_root.glob("bak[0-9]*") if path.is_dir())
    if len(all_runs) != 5:
        raise ValueError(f"expected five bak runs in {arguments.results_root}, found {len(all_runs)}")
    runs = [path for path in all_runs if path.name not in EXCLUDED_RUNS]

    reference = load_mocap_trajectory(arguments.mocap)
    stage_rows = []
    trajectories = {}
    evaluations = {}
    run_rows = []
    for run_dir in all_runs:
        run = run_dir.name
        for stage, filename in STAGE_FILES.items():
            trajectory = load_okvis_trajectory(run_dir / filename)
            evaluation = evaluate_ape(reference, trajectory)
            row = summarize_stage(run, stage, trajectory, evaluation)
            if run not in EXCLUDED_RUNS:
                stage_rows.append(row)
            trajectories[(run, stage)] = trajectory
            evaluations[(run, stage)] = evaluation
        final_row = summarize_stage(
            run,
            "final-ba",
            trajectories[(run, "final-ba")],
            evaluations[(run, "final-ba")],
        )
        topology_row = target_run_map_summary(run_dir, final_row)
        if run not in EXCLUDED_RUNS:
            run_rows.append(topology_row)

    excluded_summary = summarize_stage(
        "bak0",
        "final-ba",
        trajectories[("bak0", "final-ba")],
        evaluations[("bak0", "final-ba")],
    )

    final_ba = {run_dir.name: trajectories[(run_dir.name, "final-ba")] for run_dir in runs}
    final_evaluations = {run_dir.name: evaluations[(run_dir.name, "final-ba")] for run_dir in runs}
    pairwise_rows, pairwise_matrix, pairwise_names = pairwise_repeatability(final_ba)
    image_delay = parse_image_delay(arguments.config)
    sequence_specs = build_sequence_specs(arguments)
    contexts = [
        analyze_sequence_spec(
            spec,
            arguments.image_samples_per_camera,
            image_delay,
        )
        for spec in sequence_specs
    ]
    control_contexts, target_context = partition_sequence_contexts(contexts)
    comparison_camera_rows = finalize_quality_comparison(contexts)
    comparison_quality_rows = [
        row for context in contexts for row in context["quality_rows"]
    ]
    comparison_motion_rows = [
        row for context in contexts for row in context["motion_threshold_rows"]
    ]
    comparison_gap_rows = [
        row for context in contexts for row in context["gap_events"]
    ]
    comparison_error_rows = [
        row for context in contexts for row in context["error_rows"]
    ]
    comparison_quality_bins = quality_bin_rows(comparison_quality_rows)
    motion = target_context["motion"]
    motion_summary = target_context["motion_summary"]
    for row in run_rows:
        evaluation = final_evaluations[row["run"]]
        linear_values = np.interp(
            evaluation.timestamps, motion["timestamps"], motion["linear_speed"]
        )
        angular_values = np.interp(
            evaluation.timestamps, motion["timestamps"], motion["angular_speed"]
        )
        row["ape_linear_speed_spearman"] = float(
            spearmanr(evaluation.errors, linear_values).statistic
        )
        row["ape_angular_speed_spearman"] = float(
            spearmanr(evaluation.errors, angular_values).statistic
        )
    imu_summary = target_context["imu_summary"]
    imu_timeseries = target_context["imu_timeseries"]
    camera_rows = target_context["camera_rows"]
    sharpness_rows = target_context["quality_rows"]
    camera_aggregate = target_context["camera_aggregate"]
    reference_rows = _reference_analysis(
        [context["spec"] for context in control_contexts]
    )

    write_csv(output / "stage_summary.csv", stage_rows)
    write_csv(output / "run_summary.csv", run_rows)
    write_csv(output / "reference_summary.csv", reference_rows)
    write_csv(output / "pairwise_repeatability.csv", pairwise_rows)
    write_csv(output / "imu_summary.csv", [imu_summary])
    write_csv(output / "motion_summary.csv", [motion_summary])
    write_csv(output / "camera_summary.csv", camera_rows)
    write_csv(output / "image_sharpness.csv", sharpness_rows)
    write_csv(
        output / "sequence_comparison_summary.csv",
        [context["summary"] for context in contexts],
    )
    write_csv(output / "sequence_motion_thresholds.csv", comparison_motion_rows)
    write_csv(output / "sequence_camera_quality.csv", comparison_camera_rows)
    write_csv(
        output / "sequence_image_quality_samples.csv", comparison_quality_rows
    )
    write_csv(output / "sequence_image_quality_bins.csv", comparison_quality_bins)
    write_csv(
        output / "sequence_camera_gap_events.csv",
        comparison_gap_rows,
        fieldnames=CAMERA_GAP_EVENT_FIELDS,
    )
    write_csv(output / "sequence_error_timeline.csv", comparison_error_rows)
    write_csv(
        output / "sequence_imu_integrity.csv",
        [
            {"sequence": context["sequence"], **context["imu_summary"]}
            for context in contexts
        ],
    )
    write_csv(
        output / "sequence_mocap_integrity.csv",
        [
            {"sequence": context["sequence"], **context["mocap_summary"]}
            for context in contexts
        ],
    )
    write_csv(
        output / "sequence_map_quality.csv",
        [
            {"sequence": context["sequence"], **context["map_summary"]}
            for context in contexts
        ],
    )

    _plot_ape_trajectories(output, final_evaluations)
    _plot_stage_ape(output, stage_rows)
    _plot_ape_and_motion(output, final_evaluations, motion)
    _plot_motion_dynamics(output, motion, imu_timeseries)
    _plot_pairwise(output, pairwise_matrix, pairwise_names)
    _plot_map_topology(output, run_rows)
    _plot_outlier_window(output, final_evaluations)
    _plot_image_sharpness(output, sharpness_rows)
    _plot_reference_comparison(output, stage_rows, reference_rows)
    _plot_sequence_motion_comparison(output, contexts)
    _plot_sequence_sensor_timing(output, contexts)
    _plot_sequence_image_quality(output, contexts)
    _plot_sequence_error_timeline(output, contexts)
    _plot_map_quality_comparison(output, contexts)
    _plot_control_envelope_and_target_runs(output, contexts, run_rows)
    _write_report(
        output / "REPORT.md",
        stage_rows,
        run_rows,
        reference_rows,
        pairwise_rows,
        imu_summary,
        camera_rows,
        camera_aggregate,
        image_delay,
        motion_summary,
        excluded_summary,
    )
    append_sequence_comparison_report(
        output / "REPORT.md",
        contexts,
        comparison_camera_rows,
        run_rows,
        pairwise_rows,
    )
    print(f"Wrote repeatability analysis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
