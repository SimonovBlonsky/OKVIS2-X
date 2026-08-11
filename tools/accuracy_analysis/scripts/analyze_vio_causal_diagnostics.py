#!/usr/bin/env python3
"""Analyze event-aligned VIO diagnostics for angular-motion failures."""

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "okvis_causal_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "okvis_causal_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import spearmanr

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.accuracy_analysis.scripts import mocap_reference_correction
from tools.accuracy_analysis.scripts.run_vio_diagnostics import parse_image_delay
from tools.evaluate_mocap_ape import parse_mocap, parse_okvis_csv


plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "Droid Sans Fallback",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


SCHEMA_VERSION = 1
IMPULSE_SEQUENCES = frozenset(
    {"20260806-175103", "20260806-175304", "20260806-175539"}
)
VISUAL_OBSERVATION_REMOVAL_REASONS = frozenset({
    "gp3p_outlier",
    "post_optimisation_reprojection",
    "initialisation_2d2d_outlier",
    "loop_closure_reassociation",
})
MEDIATORS = {
    "feature_availability": [
        "keypoints_total",
        "grid_fraction_mean",
        "hull_fraction_mean",
        "image_laplacian_variance",
        "image_gradient_median",
        "image_intensity_stddev",
    ],
    "map_matching": [
        "projected_eligible_map_landmarks",
        "descriptor_comparisons",
        "descriptor_candidates_below_threshold",
        "accepted_map_matches",
        "best_descriptor_distance_median",
        "descriptor_distance_median",
        "predicted_reprojection_error_px_median",
    ],
    "triangulation_geometry": [
        "temporal_ray_angle_p10_rad",
        "temporal_parallel_fraction",
        "spatial_ray_angle_p10_rad",
        "initialisable_fraction",
        "mocap_body_translation_m",
        "mocap_body_rotation_rad",
        "mocap_body_translation_per_rotation_m_per_rad",
        "rotation_only_minus_relative_pose_inlier_ratio",
    ],
    "prediction_consistency": [
        "gp3p_start_to_model_rotation_rad",
        "gp3p_start_to_model_translation_m",
        "gp3p_pre_invocation_to_model_rotation_rad",
        "gp3p_pre_invocation_to_model_translation_m",
    ],
    "map_feedback": [
        "gp3p_inlier_ratio",
        "visual_observation_removals",
        "active_initialised_landmarks",
        "landmark_births",
    ],
}

POSE_COMPONENTS = ("tx", "ty", "tz", "qw", "qx", "qy", "qz")
FRAME_CAMERA_COLUMNS = (
    "keypoints_cam{camera}",
    "response_p10_cam{camera}",
    "response_median_cam{camera}",
    "response_p90_cam{camera}",
    "grid_fraction_cam{camera}",
    "hull_fraction_cam{camera}",
    "projected_eligible_cam{camera}",
    "descriptor_comparisons_cam{camera}",
    "descriptor_candidates_below_threshold_cam{camera}",
    "epipolar_rejected_cam{camera}",
    "divergent_ray_rejected_cam{camera}",
    "accepted_initialised_cam{camera}",
    "accepted_uninitialised_cam{camera}",
    "best_map_descriptor_distance_p10_cam{camera}",
    "best_map_descriptor_distance_median_cam{camera}",
    "best_map_descriptor_distance_p90_cam{camera}",
)
RANSAC_CAMERA_COLUMNS = (
    "correspondences_cam{camera}",
    "inliers_cam{camera}",
    "correspondence_grid_fraction_cam{camera}",
    "inlier_grid_fraction_cam{camera}",
)


def _trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))


def _pose_columns(prefix: str) -> set[str]:
    return {f"{prefix}_{component}" for component in POSE_COMPONENTS}


REQUIRED_COLUMNS = {
    "vio_diag_frame.csv": {
        "schema_version",
        "timestamp_ns",
        "frame_id",
        "initialised",
        "data_association_succeeded",
        "tracking_quality_below_threshold",
        "keyframe",
        "tracking_quality",
        "keypoints_cam0",
        "response_p10_cam0",
        "response_median_cam0",
        "response_p90_cam0",
        "grid_fraction_cam0",
        "hull_fraction_cam0",
        "projected_eligible_cam0",
        "descriptor_comparisons_cam0",
        "descriptor_candidates_below_threshold_cam0",
        "epipolar_rejected_cam0",
        "divergent_ray_rejected_cam0",
        "accepted_initialised_cam0",
        "accepted_uninitialised_cam0",
        "best_map_descriptor_distance_p10_cam0",
        "best_map_descriptor_distance_median_cam0",
        "best_map_descriptor_distance_p90_cam0",
        "active_initialised_landmarks",
        "active_uninitialised_landmarks",
        "landmark_births",
        "landmark_initialisations",
        "observations_added",
        "observations_removed_reason_0",
        "observations_removed_reason_1",
        "observations_removed_reason_2",
        "observations_removed_reason_3",
    },
    "vio_diag_triangulation.csv": {
        "schema_version",
        "timestamp_ns",
        "frame_id",
        "source",
        "camera0",
        "camera1",
        "attempts",
        "descriptor_candidates",
        "valid",
        "invalid",
        "parallel",
        "initialisable",
        "baseline_m_p10",
        "baseline_m_median",
        "ray_angle_rad_p10",
        "ray_angle_rad_median",
    },
    "vio_diag_initialisation.csv": {
        "schema_version",
        "timestamp_ns",
        "current_frame_id",
        "older_frame_id",
        "camera",
        "invocation",
        "correspondences",
        "rotation_model_computed",
        "rotation_inliers",
        "rotation_inlier_ratio",
        "relative_pose_model_computed",
        "relative_pose_inliers",
        "relative_pose_inlier_ratio",
        "selected_model",
        "selected_model_successful",
        "selected_inliers",
        "function_returned_success",
        "function_return_value",
    },
    "vio_diag_ransac.csv": {
        "schema_version",
        "timestamp_ns",
        "frame_id",
        "invocation",
        "primary_trigger",
        "trigger_mask",
        "status",
        "correspondences",
        "inliers",
        "outliers",
        "removed_observations",
        "inlier_ratio",
        "returned_success",
        "model_computed",
        "threshold_success",
        "data_association_start_pose_source",
        "pre_invocation_pose_source",
        "correspondences_cam0",
        "inliers_cam0",
        "correspondence_grid_fraction_cam0",
        "inlier_grid_fraction_cam0",
        "start_to_model_rotation_rad",
        "start_to_model_translation_m",
        "pre_invocation_to_model_rotation_rad",
        "pre_invocation_to_model_translation_m",
    }
    | _pose_columns("data_association_start")
    | _pose_columns("pre_invocation")
    | _pose_columns("gp3p_model"),
    "vio_diag_landmark_events.csv": {
        "schema_version",
        "event_sequence",
        "event_timestamp_ns",
        "event_frame_id",
        "subject_timestamp_ns",
        "subject_frame_id",
        "birth_timestamp_ns",
        "birth_frame_id",
        "landmark_id",
        "graph_role",
        "event_type",
        "reason",
        "initialised_before",
        "initialised_after",
        "observations_before",
        "observations_after",
        "quality",
    },
}

BOOLEAN_COLUMNS = {
    "initialised",
    "data_association_succeeded",
    "tracking_quality_below_threshold",
    "keyframe",
    "returned_success",
    "rotation_model_computed",
    "relative_pose_model_computed",
    "selected_model_successful",
    "function_returned_success",
    "model_computed",
    "threshold_success",
    "initialised_before",
    "initialised_after",
}
INTEGER_COLUMNS = {
    "schema_version",
    "frame_id",
    "landmark_id",
    "invocation",
    "event_sequence",
    "current_frame_id",
    "older_frame_id",
    "camera",
    "event_frame_id",
    "subject_frame_id",
    "birth_frame_id",
    "trigger_mask",
    "camera0",
    "camera1",
    "correspondences",
    "inliers",
    "outliers",
    "removed_observations",
    "attempts",
    "descriptor_candidates",
    "valid",
    "invalid",
    "parallel",
    "initialisable",
    "rotation_inliers",
    "relative_pose_inliers",
    "selected_inliers",
    "function_return_value",
    "observations_before",
    "observations_after",
    "active_initialised_landmarks",
    "active_uninitialised_landmarks",
    "landmark_births",
    "landmark_initialisations",
    "observations_added",
}
OPTIONAL_COLUMNS = {
    "tracking_quality",
    "best_map_descriptor_distance_p10_cam0",
    "best_map_descriptor_distance_median_cam0",
    "best_map_descriptor_distance_p90_cam0",
    "baseline_m_p10",
    "baseline_m_median",
    "ray_angle_rad_p10",
    "ray_angle_rad_median",
    "inlier_grid_fraction_cam0",
    "inlier_ratio",
    "rotation_inlier_ratio",
    "relative_pose_inlier_ratio",
    "subject_timestamp_ns",
    "subject_frame_id",
    "birth_timestamp_ns",
    "birth_frame_id",
    "quality",
    "start_to_model_rotation_rad",
    "start_to_model_translation_m",
    "pre_invocation_to_model_rotation_rad",
    "pre_invocation_to_model_translation_m",
} | _pose_columns("gp3p_model")


@dataclasses.dataclass(frozen=True)
class DiagnosticRun:
    sequence: str
    run: str
    root: Path
    manifest: dict[str, object]
    metadata: dict[str, str]
    frames: list[dict[str, object]]
    triangulation: list[dict[str, object]]
    initialisation: list[dict[str, object]]
    ransac: list[dict[str, object]]
    landmark_events: list[dict[str, object]]


@dataclasses.dataclass(frozen=True)
class AngularEvent:
    start_s: float
    end_s: float
    peak_radps: float
    integrated_angle_rad: float


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing diagnostic file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def _is_integer_column(key: str) -> bool:
    return (
        key.endswith("_ns")
        or key in INTEGER_COLUMNS
        or key.startswith(
            (
                "keypoints_cam",
                "projected_eligible_cam",
                "descriptor_comparisons_cam",
                "descriptor_candidates_below_threshold_cam",
                "epipolar_rejected_cam",
                "divergent_ray_rejected_cam",
                "accepted_initialised_cam",
                "accepted_uninitialised_cam",
                "correspondences_cam",
                "inliers_cam",
                "observations_removed_reason_",
            )
        )
    )


def _convert_scalar(key: str, value: str) -> object:
    if value == "":
        return None
    if key in BOOLEAN_COLUMNS:
        if value not in {"0", "1"}:
            raise ValueError(f"{key}: expected 0 or 1, got {value!r}")
        return value == "1"
    if _is_integer_column(key):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{key}: expected integer, got {value!r}") from error
    try:
        converted = float(value)
    except ValueError:
        return value
    if not math.isfinite(converted):
        raise ValueError(f"{key}: non-finite numeric value {value!r}")
    return converted


def _typed_rows(path: Path, required: set[str]) -> list[dict[str, object]]:
    rows = []
    for row_number, row in enumerate(_read_rows(path, required), 2):
        converted = {key: _convert_scalar(key, value) for key, value in row.items()}
        missing_values = [
            key for key in required
            if converted.get(key) is None and key not in OPTIONAL_COLUMNS
        ]
        if missing_values:
            raise ValueError(
                f"{path}:{row_number}: empty required values {sorted(missing_values)}"
            )
        rows.append(converted)
    return rows


def _validate_csv_columns(path: Path, required: set[str]) -> None:
    if not path.is_file():
        raise ValueError(f"missing diagnostic file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        fields = set(next(csv.reader(stream), []))
    missing = required - fields
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _require_unique(
    rows: Sequence[dict[str, object]], fields: tuple[str, ...], name: str
) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"{name}: duplicate keys {duplicates}")


def _validate_camera_columns(
    path: Path, camera_count: int, templates: Sequence[str]
) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        fields = set(csv.DictReader(stream).fieldnames or ())
    expected = {
        template.format(camera=camera)
        for camera in range(camera_count)
        for template in templates
    }
    missing = expected - fields
    if missing:
        raise ValueError(f"{path}: camera column count mismatch; missing {sorted(missing)}")
    prefixes = tuple(template.split("{camera}", 1)[0] for template in templates)
    for field in fields:
        for prefix in prefixes:
            if field.startswith(prefix):
                suffix = field[len(prefix):]
                if suffix.isdigit() and int(suffix) >= camera_count:
                    raise ValueError(
                        f"{path}: camera column count mismatch; unexpected {field}"
                    )


def load_diagnostic_run(root: Path, result_id: str, run: str) -> DiagnosticRun:
    run_root = Path(root) / result_id / run
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("mocap_path", "config_path", "image_delay_s"):
        if key not in manifest:
            raise ValueError(f"{manifest_path}: missing {key}")
    for key in ("mocap_path", "config_path"):
        if not Path(str(manifest[key])).is_file():
            raise ValueError(f"{manifest_path}: {key} does not exist")
    image_delay_s = float(manifest["image_delay_s"])
    if not math.isfinite(image_delay_s):
        raise ValueError(f"{manifest_path}: invalid image_delay_s")
    config_delay = parse_image_delay(Path(str(manifest["config_path"])))
    if not math.isclose(config_delay, image_delay_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{manifest_path}: image delay mismatch {image_delay_s} != {config_delay}"
        )
    sequence = str(manifest.get("sequence", result_id)).strip()
    manifest_run = str(manifest.get("run", run)).strip()
    if not sequence:
        raise ValueError(f"{manifest_path}: empty sequence")
    if manifest_run != run:
        raise ValueError(
            f"{manifest_path}: run mismatch {manifest_run!r} != {run!r}"
        )

    diagnostics = run_root / "diagnostics"
    if not (diagnostics / ".vio_diagnostics.complete").is_file():
        raise ValueError(f"{diagnostics}: missing completion sentinel")
    if (diagnostics / ".vio_diagnostics.active").exists():
        raise ValueError(f"{diagnostics}: active writer sentinel remains")
    metadata_rows = _read_rows(
        diagnostics / "vio_diag_metadata.csv",
        {"schema_version", "key", "value"},
    )
    if any(row["schema_version"] != str(SCHEMA_VERSION) for row in metadata_rows):
        raise ValueError(f"{diagnostics}: unsupported metadata schema")
    metadata = {row["key"]: row["value"] for row in metadata_rows}
    if metadata.get("run_complete") != "true":
        raise ValueError(f"{diagnostics}: run is not complete")
    if metadata.get("writer_failed", "false") == "true":
        raise ValueError(f"{diagnostics}: writer reported failure")
    try:
        camera_count = int(metadata["camera_count"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{diagnostics}: invalid camera_count") from error
    if camera_count < 1:
        raise ValueError(f"{diagnostics}: invalid camera_count")

    _validate_camera_columns(
        diagnostics / "vio_diag_frame.csv", camera_count, FRAME_CAMERA_COLUMNS
    )
    _validate_camera_columns(
        diagnostics / "vio_diag_ransac.csv", camera_count, RANSAC_CAMERA_COLUMNS
    )
    tables = {
        name: _typed_rows(diagnostics / name, required)
        for name, required in REQUIRED_COLUMNS.items()
        if name != "vio_diag_landmark_events.csv"
    }
    _validate_csv_columns(
        diagnostics / "vio_diag_landmark_events.csv",
        REQUIRED_COLUMNS["vio_diag_landmark_events.csv"],
    )
    for name, rows in tables.items():
        versions = {row["schema_version"] for row in rows}
        if versions and versions != {SCHEMA_VERSION}:
            raise ValueError(f"{name}: unsupported schema versions {versions}")

    frames = tables["vio_diag_frame.csv"]
    triangulation = tables["vio_diag_triangulation.csv"]
    initialisation = tables["vio_diag_initialisation.csv"]
    ransac = tables["vio_diag_ransac.csv"]
    _require_unique(frames, ("frame_id",), "frame")
    _require_unique(
        triangulation,
        ("frame_id", "source", "camera0", "camera1"),
        "triangulation",
    )
    _require_unique(
        initialisation,
        ("current_frame_id", "older_frame_id", "camera", "invocation"),
        "initialisation",
    )
    _require_unique(ransac, ("frame_id", "invocation"), "ransac")
    frame_timestamps = [int(row["timestamp_ns"]) for row in frames]
    if any(second <= first for first, second in zip(frame_timestamps, frame_timestamps[1:])):
        raise ValueError("frame timestamps are not strictly increasing")
    return DiagnosticRun(
        sequence=sequence,
        run=manifest_run,
        root=run_root,
        manifest=manifest,
        metadata=metadata,
        frames=frames,
        triangulation=triangulation,
        initialisation=initialisation,
        ransac=ransac,
        landmark_events=[],
    )


def detect_angular_events(
    timestamps_s: np.ndarray,
    angular_speed_radps: np.ndarray,
    threshold_radps: float = 3.0,
    minimum_duration_s: float = 0.05,
    merge_gap_s: float = 0.25,
) -> list[AngularEvent]:
    timestamps = np.asarray(timestamps_s, dtype=float)
    speed = np.asarray(angular_speed_radps, dtype=float)
    if timestamps.ndim != 1 or speed.shape != timestamps.shape:
        raise ValueError("timestamps and angular speed must be matching 1-D arrays")
    if (
        timestamps.size < 2
        or np.any(np.diff(timestamps) <= 0.0)
        or not np.all(np.isfinite(timestamps))
        or not np.all(np.isfinite(speed))
    ):
        raise ValueError("timestamps must contain at least two increasing samples")
    above = speed > threshold_radps
    transitions = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    qualified = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if timestamps[stop] - timestamps[start] >= minimum_duration_s
    ]
    merged: list[tuple[int, int]] = []
    for start, stop in qualified:
        if merged and timestamps[start] - timestamps[merged[-1][1]] < merge_gap_s:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return [
        AngularEvent(
            start_s=float(timestamps[start]),
            end_s=float(timestamps[stop]),
            peak_radps=float(np.max(speed[start : stop + 1])),
            integrated_angle_rad=_trapezoidal_integral(
                speed[start : stop + 1], timestamps[start : stop + 1]
            ),
        )
        for start, stop in merged
    ]


def compute_image_statistics(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("L")
    if image.width < 3 or image.height < 3:
        raise ValueError(f"{path}: image must be at least 3x3 pixels")
    if image.width > 640:
        height = round(image.height * 640 / image.width)
        image = image.resize((640, height), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float64)
    center = values[1:-1, 1:-1]
    laplacian = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4.0 * center
    )
    gradient_y, gradient_x = np.gradient(values)
    statistics = {
        "image_laplacian_variance": float(np.var(laplacian)),
        "image_gradient_median": float(
            np.median(np.hypot(gradient_x, gradient_y))
        ),
        "image_intensity_stddev": float(np.std(values)),
    }
    if not all(math.isfinite(value) for value in statistics.values()):
        raise ValueError(f"{path}: image statistics are not finite")
    return statistics


def _robust_baseline(
    baseline_values: Sequence[float], epsilon: float
) -> tuple[float, float]:
    values = np.asarray(baseline_values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("baseline values and positive epsilon are required")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 0.05 * abs(median), epsilon)
    return median, scale


def detect_robust_onset(
    timestamps_s: Sequence[float],
    values: Sequence[float],
    *,
    baseline_values: Sequence[float],
    harmful_direction: int,
    epsilon: float,
    threshold: float = 2.0,
    consecutive: int = 3,
) -> float | None:
    timestamps = np.asarray(timestamps_s, dtype=float)
    samples = np.asarray(values, dtype=float)
    if timestamps.ndim != 1 or samples.shape != timestamps.shape:
        raise ValueError("timestamps and values must match")
    if harmful_direction not in {-1, 1}:
        raise ValueError("harmful_direction must be -1 or 1")
    median, scale = _robust_baseline(baseline_values, epsilon)
    harmful_z = harmful_direction * (samples - median) / scale
    active = np.isfinite(harmful_z) & (harmful_z >= threshold)
    for index in range(0, len(active) - consecutive + 1):
        if np.all(active[index : index + consecutive]):
            return float(timestamps[index])
    return None


def detect_robust_recovery(
    timestamps_s: Sequence[float],
    values: Sequence[float],
    *,
    baseline_values: Sequence[float],
    search_start_s: float,
    epsilon: float,
    threshold: float = 2.0,
    consecutive: int = 5,
) -> float | None:
    timestamps = np.asarray(timestamps_s, dtype=float)
    samples = np.asarray(values, dtype=float)
    if timestamps.ndim != 1 or samples.shape != timestamps.shape:
        raise ValueError("timestamps and values must match")
    median, scale = _robust_baseline(baseline_values, epsilon)
    recovered = np.isfinite(samples) & (np.abs((samples - median) / scale) <= threshold)
    recovered &= timestamps >= search_start_s
    for index in range(0, len(recovered) - consecutive + 1):
        if np.all(recovered[index : index + consecutive]):
            return float(timestamps[index])
    return None


CONTROL_COVARIATES = (
    "active_initialised_landmarks",
    "accepted_map_matches",
    "mocap_body_translation_m",
    "mocap_body_rotation_rad",
    "image_laplacian_variance",
    "keypoints_total",
)
CONTROL_FLOORS = {
    "active_initialised_landmarks": 1.0,
    "accepted_map_matches": 1.0,
    "mocap_body_translation_m": 0.001,
    "mocap_body_rotation_rad": 0.01,
    "image_laplacian_variance": 1e-3,
    "keypoints_total": 1.0,
}


def select_matched_controls(
    event: dict[str, object],
    candidates: Iterable[dict[str, object]],
    *,
    angular_events: Sequence[AngularEvent],
    max_controls: int = 3,
) -> list[dict[str, object]]:
    if max_controls < 1:
        raise ValueError("max_controls must be positive")
    event_duration = float(event["end_s"]) - float(event["start_s"])
    eligible: list[tuple[float, float, str, dict[str, object]]] = []
    for candidate in candidates:
        if (
            candidate.get("sequence") != event.get("sequence")
            or candidate.get("run") != event.get("run")
        ):
            continue
        start = float(candidate["start_s"])
        end = float(candidate["end_s"])
        if not math.isclose(end - start, event_duration, abs_tol=1e-6):
            continue
        if float(candidate.get("peak_angular_speed_radps", math.inf)) >= 1.0:
            continue
        if any(
            start < angular.end_s + 10.0 and end > angular.start_s - 10.0
            for angular in angular_events
        ):
            continue
        distances = []
        within_caliper = True
        for field in CONTROL_COVARIATES:
            event_value = float(event[field])
            candidate_value = float(candidate[field])
            if not (math.isfinite(event_value) and math.isfinite(candidate_value)):
                within_caliper = False
                break
            scale = max(abs(event_value), CONTROL_FLOORS[field])
            tolerance = max(0.25 * abs(event_value), CONTROL_FLOORS[field])
            if abs(candidate_value - event_value) > tolerance:
                within_caliper = False
                break
            distances.append((candidate_value - event_value) / scale)
        if within_caliper:
            eligible.append(
                (
                    float(np.linalg.norm(distances)),
                    start,
                    str(candidate.get("candidate_id", "")),
                    candidate,
                )
            )
    eligible.sort(key=lambda item: item[:3])
    return [item[3] for item in eligible[:max_controls]]


def _finite_values(values: Iterable[object]) -> np.ndarray:
    numeric = []
    for value in values:
        if value is None:
            continue
        converted = float(value)
        if math.isfinite(converted):
            numeric.append(converted)
    return np.asarray(numeric, dtype=float)


def _median(values: Iterable[object]) -> float | None:
    numeric = _finite_values(values)
    return float(np.median(numeric)) if len(numeric) else None


def _maximum(values: Iterable[object]) -> float | None:
    numeric = _finite_values(values)
    return float(np.max(numeric)) if len(numeric) else None


def aggregate_frame_metrics(run: DiagnosticRun) -> list[dict[str, object]]:
    camera_count = int(run.metadata["camera_count"])
    triangulation_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    initialisation_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    ransac_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    events_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in run.triangulation:
        triangulation_by_frame[int(row["frame_id"])].append(row)
    for row in run.initialisation:
        initialisation_by_frame[int(row["current_frame_id"])].append(row)
    for row in run.ransac:
        ransac_by_frame[int(row["frame_id"])].append(row)
    for row in run.landmark_events:
        events_by_frame[int(row["event_frame_id"])].append(row)

    if not run.frames:
        return []
    first_timestamp_ns = int(run.frames[0]["timestamp_ns"])
    rows = []
    for frame in run.frames:
        frame_id = int(frame["frame_id"])
        timestamp_ns = int(frame["timestamp_ns"])
        triangulation = triangulation_by_frame.get(frame_id, [])
        initialisation = initialisation_by_frame.get(frame_id, [])
        ransac = ransac_by_frame.get(frame_id, [])
        lifecycle = [
            event for event in events_by_frame.get(frame_id, [])
            if event.get("graph_role") == "realtime"
        ]
        temporal = [
            row for row in triangulation
            if row.get("source") == "temporal_motion_stereo"
        ]
        spatial = [
            row for row in triangulation
            if row.get("source") == "spatial_stereo"
        ]
        temporal_attempts = sum(int(row.get("attempts", 0)) for row in temporal)
        temporal_parallel = sum(int(row.get("parallel", 0)) for row in temporal)
        all_attempts = sum(int(row.get("attempts", 0)) for row in triangulation)
        all_initialisable = sum(
            int(row.get("initialisable", 0)) for row in triangulation
        )
        rotation_ratios = _finite_values(
            row.get("rotation_inlier_ratio") for row in initialisation
        )
        relative_ratios = _finite_values(
            row.get("relative_pose_inlier_ratio") for row in initialisation
        )
        row = {
            "experiment_id": str(
                run.manifest.get("experiment_id", run.sequence)
            ),
            "sequence": run.sequence,
            "run": run.run,
            "intervention": str(
                run.manifest.get("intervention", "baseline")
            ),
            "intervention_value": str(
                run.manifest.get("intervention_value", "none")
            ),
            "timestamp_ns": timestamp_ns,
            "frame_id": frame_id,
            "time_s": (timestamp_ns - first_timestamp_ns) / 1e9,
            "keypoints_total": sum(
                int(frame.get(f"keypoints_cam{camera}", 0))
                for camera in range(camera_count)
            ),
            "grid_fraction_mean": _median(
                frame.get(f"grid_fraction_cam{camera}")
                for camera in range(camera_count)
            ),
            "hull_fraction_mean": _median(
                frame.get(f"hull_fraction_cam{camera}")
                for camera in range(camera_count)
            ),
            "projected_eligible_map_landmarks": sum(
                int(frame.get(f"projected_eligible_cam{camera}", 0))
                for camera in range(camera_count)
            ),
            "descriptor_comparisons": sum(
                int(frame.get(f"descriptor_comparisons_cam{camera}", 0))
                for camera in range(camera_count)
            ),
            "descriptor_candidates_below_threshold": sum(
                int(
                    frame.get(
                        f"descriptor_candidates_below_threshold_cam{camera}", 0
                    )
                )
                for camera in range(camera_count)
            ),
            "accepted_map_matches": sum(
                int(frame.get(f"accepted_initialised_cam{camera}", 0))
                + int(frame.get(f"accepted_uninitialised_cam{camera}", 0))
                for camera in range(camera_count)
            ),
            "best_descriptor_distance_median": _median(
                frame.get(f"best_map_descriptor_distance_median_cam{camera}")
                for camera in range(camera_count)
            ),
            "descriptor_distance_median": frame.get(
                "accepted_descriptor_distance_median"
            ),
            "predicted_reprojection_error_px_median": frame.get(
                "predicted_reprojection_error_px_median"
            ),
            "tracking_quality": frame.get("tracking_quality"),
            "active_initialised_landmarks": int(
                frame.get("active_initialised_landmarks", 0)
            ),
            "active_uninitialised_landmarks": int(
                frame.get("active_uninitialised_landmarks", 0)
            ),
            "landmark_births": int(frame.get("landmark_births", 0)),
            "observations_added": int(frame.get("observations_added", 0)),
            "temporal_ray_angle_p10_rad": _median(
                row.get("ray_angle_rad_p10") for row in temporal
            ),
            "temporal_baseline_p10_m": _median(
                row.get("baseline_m_p10") for row in temporal
            ),
            "temporal_parallel_fraction": (
                temporal_parallel / temporal_attempts
                if temporal_attempts else None
            ),
            "spatial_ray_angle_p10_rad": _median(
                row.get("ray_angle_rad_p10") for row in spatial
            ),
            "initialisable_fraction": (
                all_initialisable / all_attempts if all_attempts else None
            ),
            "rotation_only_minus_relative_pose_inlier_ratio": (
                float(np.max(rotation_ratios) - np.max(relative_ratios))
                if len(rotation_ratios) and len(relative_ratios) else None
            ),
            "gp3p_invocations": len(ransac),
            "gp3p_failure_count": sum(
                row.get("status")
                in {"model_computation_failed", "threshold_rejected"}
                for row in ransac
            ),
            "gp3p_inlier_ratio": _median(
                row.get("inlier_ratio") for row in ransac
            ),
            "gp3p_start_to_model_rotation_rad": _maximum(
                row.get("start_to_model_rotation_rad") for row in ransac
            ),
            "gp3p_start_to_model_translation_m": _maximum(
                row.get("start_to_model_translation_m") for row in ransac
            ),
            "gp3p_pre_invocation_to_model_rotation_rad": _maximum(
                row.get("pre_invocation_to_model_rotation_rad") for row in ransac
            ),
            "gp3p_pre_invocation_to_model_translation_m": _maximum(
                row.get("pre_invocation_to_model_translation_m") for row in ransac
            ),
            "visual_observation_removals": sum(
                event.get("event_type") == "observation_removed"
                and event.get("reason")
                in VISUAL_OBSERVATION_REMOVAL_REASONS
                for event in lifecycle
            ) + sum(
                int(frame.get(f"observations_removed_reason_{reason}", 0))
                for reason in range(4)
            ),
        }
        rows.append(row)
    return rows


def gp3p_failure_onset(
    frame_rows: Sequence[dict[str, object]], event: AngularEvent
) -> float | None:
    """Return the first GP3P failure from angular onset through early aftermath.

    ``gp3p_failure_count`` is normally a sparse binary frame-level signal.  The
    generic robust-z detector is unsuitable here: with an all-zero baseline and
    ``epsilon=1``, a real single failure only reaches z=1 and can never cross the
    default z=2 threshold.  Preserve the exact first observed failure instead;
    callers can use the baseline failure rate to distinguish a new onset from a
    continuation of pre-existing instability.
    """
    tolerance = 1e-9
    scan = [
        row for row in frame_rows
        if event.start_s - tolerance
        <= float(row["time_s"])
        <= event.end_s + 0.5 + tolerance
    ]
    for row in scan:
        value = row.get("gp3p_failure_count")
        if value is not None and math.isfinite(float(value)) and float(value) > 0.0:
            return float(row["time_s"])
    return None


MODEL_COVARIATES = (
    "pre_map_support",
    "mocap_translation",
    "mocap_rotation",
    "pre_image_sharpness",
)


def _standardize(values: np.ndarray) -> np.ndarray:
    deviation = float(np.std(values))
    if not math.isfinite(deviation) or deviation <= 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / deviation


def fit_mediation_model(
    rows: Sequence[dict[str, object]], *, mediator: str, outcome: str
) -> dict[str, object]:
    required = ("sequence", "angular_integral", mediator, outcome) + MODEL_COVARIATES
    valid = []
    for row in rows:
        try:
            numeric = [float(row[field]) for field in required[1:]]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in numeric):
            valid.append(row)
    result: dict[str, object] = {
        "status": "insufficient_model_rows",
        "rows": len(valid),
        "sequences": len({str(row["sequence"]) for row in valid}),
        "mediator": mediator,
        "outcome": outcome,
    }
    if len(valid) < 9:
        return result
    columns = {
        field: _standardize(
            np.asarray([float(row[field]) for row in valid], dtype=float)
        )
        for field in ("angular_integral", mediator, outcome) + MODEL_COVARIATES
    }
    ones = np.ones(len(valid), dtype=float)
    reduced = np.column_stack(
        [ones, columns["angular_integral"]]
        + [columns[field] for field in MODEL_COVARIATES]
    )
    full = np.column_stack(
        [ones, columns["angular_integral"], columns[mediator]]
        + [columns[field] for field in MODEL_COVARIATES]
    )
    condition = float(np.linalg.cond(full))
    result["condition_number"] = condition
    angular_rotation_rho = spearmanr(
        columns["angular_integral"], columns["mocap_rotation"]
    ).statistic
    result["angular_mocap_rotation_spearman"] = float(angular_rotation_rho)
    if not math.isfinite(condition) or condition > 1e6:
        result["status"] = "collinear_design"
        return result
    reduced_coefficients = np.linalg.lstsq(
        reduced, columns[outcome], rcond=None
    )[0]
    full_coefficients = np.linalg.lstsq(full, columns[outcome], rcond=None)[0]
    mediator_design = np.column_stack(
        [ones, columns["angular_integral"]]
        + [columns[field] for field in MODEL_COVARIATES]
    )
    mediator_coefficients = np.linalg.lstsq(
        mediator_design, columns[mediator], rcond=None
    )[0]
    c1 = float(reduced_coefficients[1])
    c_prime = float(full_coefficients[1])
    result.update(
        {
            "status": (
                "exploratory_small_n" if result["sequences"] < 6 else "ok"
            ),
            "angular_to_mediator": float(mediator_coefficients[1]),
            "angular_total": c1,
            "angular_adjusted": c_prime,
            "attenuation": c1 - c_prime,
            "mediator_to_outcome": float(full_coefficients[2]),
            "spearman": float(
                spearmanr(
                    columns["angular_integral"], columns[outcome]
                ).statistic
            ),
        }
    )
    return result


def match_corrected_camera_frames(
    frame_timestamps_ns: Sequence[int],
    raw_camera_timestamps_ns: Sequence[int],
    *,
    image_delay_s: float,
) -> np.ndarray:
    frames = np.asarray(frame_timestamps_ns, dtype=np.int64)
    raw = np.asarray(raw_camera_timestamps_ns, dtype=np.int64)
    if (
        frames.ndim != 1
        or raw.ndim != 1
        or not len(frames)
        or len(raw) < 2
        or np.any(np.diff(frames) <= 0)
        or np.any(np.diff(raw) <= 0)
        or not math.isfinite(image_delay_s)
    ):
        raise ValueError("camera and frame timestamps must be increasing vectors")
    delay_ns = int(round(image_delay_s * 1e9))
    corrected = raw - delay_ns
    insertions = np.searchsorted(corrected, frames)
    right = np.clip(insertions, 0, len(corrected) - 1)
    left = np.clip(insertions - 1, 0, len(corrected) - 1)
    choose_right = np.abs(corrected[right] - frames) < np.abs(
        corrected[left] - frames
    )
    indices = np.where(choose_right, right, left)
    camera_period_ns = int(np.median(np.diff(corrected)))
    if np.any(np.abs(corrected[indices] - frames) > camera_period_ns):
        raise ValueError("camera/frame join exceeds one camera period")
    return indices.astype(int)


def compute_frame_interval_imu_metrics(
    frame_timestamps_ns: Sequence[int],
    imu_timestamps_ns: Sequence[int],
    gyroscope_radps: np.ndarray,
    *,
    saturation_radps: float = 34.9,
) -> list[dict[str, object]]:
    frames = np.asarray(frame_timestamps_ns, dtype=np.int64)
    imu = np.asarray(imu_timestamps_ns, dtype=np.int64)
    gyro = np.asarray(gyroscope_radps, dtype=float)
    if (
        frames.ndim != 1
        or imu.ndim != 1
        or gyro.shape != (len(imu), 3)
        or not len(frames)
        or len(imu) < 2
        or np.any(np.diff(frames) <= 0)
        or np.any(np.diff(imu) <= 0)
        or not np.all(np.isfinite(gyro))
        or not math.isfinite(saturation_radps)
        or saturation_radps <= 0.0
    ):
        raise ValueError("invalid frame or IMU data")
    magnitude = np.linalg.norm(gyro, axis=1)
    output: list[dict[str, object]] = [
        {
            "imu_gyro_max_radps": None,
            "imu_gyro_mean_radps": None,
            "imu_angular_integral_rad": None,
            "imu_sample_count": 0,
            "imu_max_gap_s": None,
            "imu_saturation_count": 0,
        }
    ]
    for previous, current in zip(frames[:-1], frames[1:]):
        start = int(np.searchsorted(imu, previous, side="left"))
        stop = int(np.searchsorted(imu, current, side="left"))
        interval_magnitude = magnitude[start:stop]
        interval_timestamps = imu[start:stop]
        if not len(interval_magnitude):
            output.append(
                {
                    "imu_gyro_max_radps": None,
                    "imu_gyro_mean_radps": None,
                    "imu_angular_integral_rad": None,
                    "imu_sample_count": 0,
                    "imu_max_gap_s": None,
                    "imu_saturation_count": 0,
                }
            )
            continue
        integral = (
            _trapezoidal_integral(
                interval_magnitude,
                (interval_timestamps - interval_timestamps[0]) / 1e9,
            )
            if len(interval_magnitude) >= 2 else 0.0
        )
        gaps = np.diff(interval_timestamps) / 1e9
        output.append(
            {
                "imu_gyro_max_radps": float(np.max(interval_magnitude)),
                "imu_gyro_mean_radps": float(np.mean(interval_magnitude)),
                "imu_angular_integral_rad": integral,
                "imu_sample_count": int(len(interval_magnitude)),
                "imu_max_gap_s": float(np.max(gaps)) if len(gaps) else None,
                "imu_saturation_count": int(
                    np.count_nonzero(interval_magnitude >= saturation_radps)
                ),
            }
        )
    return output


def _load_camera_index_ns(path: Path) -> tuple[np.ndarray, list[str]]:
    timestamps = []
    filenames = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        for row_number, row in enumerate(reader, 1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{row_number}: malformed camera row")
            try:
                timestamps.append(int(row[0]))
            except ValueError as error:
                raise ValueError(
                    f"{path}:{row_number}: invalid camera timestamp"
                ) from error
            filenames.append(row[1].strip())
    values = np.asarray(timestamps, dtype=np.int64)
    if len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError(f"{path}: camera timestamps are not increasing")
    return values, filenames


def _load_imu_ns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", comments="#", dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise ValueError(f"{path}: expected EuRoC IMU rows")
    timestamps = values[:, 0].astype(np.int64)
    if np.any(np.diff(timestamps) <= 0) or not np.all(np.isfinite(values[:, 1:])):
        raise ValueError(f"{path}: invalid IMU values")
    return timestamps, values[:, 1:4]


def _resolve_dataset(run: DiagnosticRun, data_root: Path) -> Path:
    manifest_dataset = run.manifest.get("dataset_path") or run.manifest.get("dataset")
    if manifest_dataset:
        dataset = Path(str(manifest_dataset))
    else:
        day = run.sequence.split("-", 1)[0]
        dataset = Path(data_root) / day / f"{run.sequence}_euroc"
    if not dataset.is_dir():
        raise ValueError(f"missing dataset: {dataset}")
    return dataset


def _resolve_sensor_root(dataset: Path) -> Path:
    candidates = [dataset, dataset / "mav0"]
    valid = [
        candidate
        for candidate in candidates
        if (candidate / "cam0").is_dir() and (candidate / "imu0").is_dir()
    ]
    if len(valid) != 1:
        raise ValueError(
            f"{dataset}: expected exactly one root or mav0 sensor layout"
        )
    return valid[0]


def _interpolate_mocap(
    timestamps_s: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    query_s: np.ndarray,
) -> tuple[np.ndarray, Rotation, np.ndarray]:
    valid = (query_s >= timestamps_s[0]) & (query_s <= timestamps_s[-1])
    interpolated_positions = np.full((len(query_s), 3), np.nan)
    for axis in range(3):
        interpolated_positions[valid, axis] = np.interp(
            query_s[valid], timestamps_s, positions[:, axis]
        )
    rotations = Rotation.identity(len(query_s))
    if np.any(valid):
        slerp = Slerp(timestamps_s, Rotation.from_quat(quaternions_xyzw))
        valid_rotations = slerp(query_s[valid])
        quaternion_rows = rotations.as_quat()
        quaternion_rows[valid] = valid_rotations.as_quat()
        rotations = Rotation.from_quat(quaternion_rows)
    return interpolated_positions, rotations, valid


def attach_sensor_metrics(
    run: DiagnosticRun,
    frame_rows: list[dict[str, object]],
    *,
    data_root: Path,
) -> tuple[list[AngularEvent], dict[str, object]]:
    if not frame_rows:
        return [], {"frames": 0, "matched_images": 0}
    dataset = _resolve_dataset(run, data_root)
    sensor_root = _resolve_sensor_root(dataset)
    frame_ns = np.asarray([int(row["timestamp_ns"]) for row in frame_rows])
    image_delay_s = float(run.manifest["image_delay_s"])
    camera_ns, filenames = _load_camera_index_ns(
        sensor_root / "cam0" / "data.csv"
    )
    image_indices = match_corrected_camera_frames(
        frame_ns, camera_ns, image_delay_s=image_delay_s
    )
    image_cache: dict[int, dict[str, float]] = {}
    for row, image_index in zip(frame_rows, image_indices):
        index = int(image_index)
        if index not in image_cache:
            image_cache[index] = compute_image_statistics(
                sensor_root / "cam0" / "data" / filenames[index]
            )
        row.update(image_cache[index])
        row["camera_raw_timestamp_ns"] = int(camera_ns[index])
        row["camera_corrected_timestamp_ns"] = int(
            camera_ns[index] - round(image_delay_s * 1e9)
        )
        row["camera_join_error_ns"] = abs(
            int(row["camera_corrected_timestamp_ns"]) - int(row["timestamp_ns"])
        )

    imu_ns, gyroscope = _load_imu_ns(sensor_root / "imu0" / "data.csv")
    interval_metrics = compute_frame_interval_imu_metrics(
        frame_ns, imu_ns, gyroscope
    )
    for row, metrics in zip(frame_rows, interval_metrics):
        row.update(metrics)
    gyro_magnitude = np.linalg.norm(gyroscope, axis=1)
    relative_imu_s = (imu_ns - frame_ns[0]) / 1e9
    events = detect_angular_events(relative_imu_s, gyro_magnitude)

    mocap_poses = parse_mocap(Path(str(run.manifest["mocap_path"])))
    if len(mocap_poses) < 2:
        raise ValueError(f"{run.manifest['mocap_path']}: insufficient mocap poses")
    mocap_timestamps = np.asarray(
        [pose.timestamp for pose in mocap_poses], dtype=float
    )
    mocap_positions = np.asarray(
        [pose.position for pose in mocap_poses], dtype=float
    )
    mocap_quaternions = np.asarray(
        [pose.quaternion for pose in mocap_poses], dtype=float
    )
    lever = mocap_reference_correction.session_fixed_lever(
        run.sequence, mocap_reference_correction.FIXED_DIAGNOSTIC_LEVER_M
    )
    mocap_positions = mocap_reference_correction.correct_reference_positions(
        mocap_positions, mocap_quaternions[:, [3, 0, 1, 2]], lever
    )
    query_s = frame_ns / 1e9
    body_positions, body_rotations, valid = _interpolate_mocap(
        mocap_timestamps, mocap_positions, mocap_quaternions, query_s
    )
    rotation_steps = np.full(len(frame_rows), np.nan)
    translation_steps = np.full(len(frame_rows), np.nan)
    for index in range(1, len(frame_rows)):
        if not (valid[index - 1] and valid[index]):
            continue
        translation_steps[index] = np.linalg.norm(
            body_positions[index] - body_positions[index - 1]
        )
        rotation_steps[index] = (
            body_rotations[index - 1].inv() * body_rotations[index]
        ).magnitude()
    for index, row in enumerate(frame_rows):
        translation = translation_steps[index]
        rotation = rotation_steps[index]
        row["mocap_body_translation_m"] = (
            float(translation) if math.isfinite(translation) else None
        )
        row["mocap_body_rotation_rad"] = (
            float(rotation) if math.isfinite(rotation) else None
        )
        row["mocap_body_translation_per_rotation_m_per_rad"] = (
            float(translation / rotation)
            if math.isfinite(translation)
            and math.isfinite(rotation)
            and rotation >= 1e-3 else None
        )
    return events, {
        "frames": len(frame_rows),
        "matched_images": len(frame_rows),
        "unique_images_read": len(image_cache),
        "mocap_matched_frames": int(np.count_nonzero(valid)),
        "camera_join_max_ns": max(
            int(row["camera_join_error_ns"]) for row in frame_rows
        ),
    }


def _window_rows(
    rows: Sequence[dict[str, object]], start_s: float, end_s: float
) -> list[dict[str, object]]:
    return [
        row for row in rows
        if start_s <= float(row["time_s"]) <= end_s
    ]


def _window_median(
    rows: Sequence[dict[str, object]], field: str
) -> float | None:
    return _median(row.get(field) for row in rows)


def _window_sum(
    rows: Sequence[dict[str, object]], field: str
) -> float | None:
    values = _finite_values(row.get(field) for row in rows)
    return float(np.sum(values)) if len(values) else None


def _baseline_control_values(
    rows: Sequence[dict[str, object]], start_s: float
) -> dict[str, float] | None:
    baseline = _window_rows(rows, start_s - 5.0, start_s - 1.0)
    fields = {
        "active_initialised_landmarks": "active_initialised_landmarks",
        "accepted_map_matches": "accepted_map_matches",
        "mocap_body_translation_m": "mocap_body_translation_m",
        "mocap_body_rotation_rad": "mocap_body_rotation_rad",
        "image_laplacian_variance": "image_laplacian_variance",
        "keypoints_total": "keypoints_total",
    }
    values = {
        output: _window_median(baseline, source)
        for output, source in fields.items()
    }
    if any(value is None for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items()}


def enumerate_low_angular_candidates(
    rows: Sequence[dict[str, object]],
    event: AngularEvent,
) -> list[dict[str, object]]:
    if not rows:
        return []
    duration = event.end_s - event.start_s
    candidates = []
    next_start = -math.inf
    for row in rows:
        start = float(row["time_s"])
        if start < 5.0 or start < next_start:
            continue
        end = start + duration
        if end + 10.0 > float(rows[-1]["time_s"]):
            break
        window = _window_rows(rows, start, end)
        peak = _maximum(item.get("imu_gyro_max_radps") for item in window)
        covariates = _baseline_control_values(rows, start)
        if peak is None or covariates is None:
            continue
        candidates.append(
            {
                "candidate_id": f"{start:.9f}",
                "sequence": row["sequence"],
                "run": row["run"],
                "start_s": start,
                "end_s": end,
                "peak_angular_speed_radps": peak,
                **covariates,
            }
        )
        next_start = start + max(duration, 1.0)
    return candidates


HARMFUL_DIRECTIONS = defaultdict(
    lambda: 1,
    {
        "keypoints_total": -1,
        "grid_fraction_mean": -1,
        "hull_fraction_mean": -1,
        "image_laplacian_variance": -1,
        "image_gradient_median": -1,
        "image_intensity_stddev": -1,
        "projected_eligible_map_landmarks": -1,
        "descriptor_candidates_below_threshold": -1,
        "accepted_map_matches": -1,
        "temporal_ray_angle_p10_rad": -1,
        "spatial_ray_angle_p10_rad": -1,
        "initialisable_fraction": -1,
        "gp3p_inlier_ratio": -1,
        "active_initialised_landmarks": -1,
    },
)


def _metric_epsilon(metric: str) -> float:
    if any(token in metric for token in ("count", "keypoints", "landmarks", "matches", "removals")):
        return 1.0
    if "px" in metric or "descriptor" in metric:
        return 0.1
    if "fraction" in metric or "ratio" in metric:
        return 1e-3
    if "rad" in metric or "rotation" in metric:
        return 1e-4
    if metric.endswith("_m") or "translation" in metric or "baseline" in metric:
        return 1e-4
    return 1e-3


def _event_metric_change(
    rows: Sequence[dict[str, object]],
    metric: str,
    start_s: float,
    end_s: float,
) -> tuple[float | None, float | None, float | None]:
    baseline = _window_median(
        _window_rows(rows, start_s - 5.0, start_s - 1.0), metric
    )
    mediator = _window_median(
        _window_rows(rows, start_s, end_s + 0.5), metric
    )
    if baseline is None or mediator is None:
        return baseline, mediator, None
    scale_baseline = _finite_values(
        row.get(metric)
        for row in _window_rows(rows, start_s - 5.0, start_s - 1.0)
    )
    _, scale = _robust_baseline(scale_baseline, _metric_epsilon(metric))
    return baseline, mediator, (mediator - baseline) / scale


def _event_outcomes(
    rows: Sequence[dict[str, object]], start_s: float, end_s: float
) -> tuple[float | None, float | None, float | None]:
    def failure_ratio(window_start_s: float, window_end_s: float) -> float | None:
        window = _window_rows(rows, window_start_s, window_end_s)
        failures = _window_sum(window, "gp3p_failure_count")
        invocations = _window_sum(window, "gp3p_invocations")
        return (
            failures / invocations
            if failures is not None and invocations not in {None, 0.0}
            else None
        )

    # The first GP3P response commonly occurs during the angular event or in the
    # first half-second after it.  Keep a separate post-event persistence metric
    # instead of shifting the primary outcome window past the actual failures.
    gp3p_outcome = failure_ratio(start_s, end_s + 0.5)
    gp3p_post_outcome = failure_ratio(
        math.nextafter(end_s + 0.5, math.inf), end_s + 2.0
    )
    baseline_support = _window_median(
        _window_rows(rows, start_s - 5.0, start_s - 1.0),
        "active_initialised_landmarks",
    )
    map_support = _window_median(
        _window_rows(rows, end_s + 2.0, end_s + 5.0),
        "active_initialised_landmarks",
    )
    map_support_outcome = (
        map_support - baseline_support
        if map_support is not None and baseline_support is not None
        else None
    )
    return gp3p_outcome, gp3p_post_outcome, map_support_outcome


def _local_trajectory_path(run_root: Path) -> Path | None:
    candidates = [
        path for path in run_root.glob("okvis2-*_trajectory.csv")
        if "-final" not in path.name
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation, target_center - rotation @ source_center


def compute_event_local_errors(
    run: DiagnosticRun,
    frame_rows: Sequence[dict[str, object]],
    events: Sequence[AngularEvent],
) -> dict[int, np.ndarray | None]:
    trajectory_path = _local_trajectory_path(run.root)
    if trajectory_path is None or not frame_rows:
        return {index: None for index in range(len(events))}
    estimate_poses = parse_okvis_csv(trajectory_path)
    mocap_poses = parse_mocap(Path(str(run.manifest["mocap_path"])))
    if len(estimate_poses) < 2 or len(mocap_poses) < 2:
        return {index: None for index in range(len(events))}
    estimate_t = np.asarray([pose.timestamp for pose in estimate_poses])
    estimate_positions = np.asarray([pose.position for pose in estimate_poses])
    mocap_t = np.asarray([pose.timestamp for pose in mocap_poses])
    mocap_positions = np.asarray([pose.position for pose in mocap_poses])
    mocap_quaternions = np.asarray([pose.quaternion for pose in mocap_poses])
    lever = mocap_reference_correction.session_fixed_lever(
        run.sequence, mocap_reference_correction.FIXED_DIAGNOSTIC_LEVER_M
    )
    mocap_positions = mocap_reference_correction.correct_reference_positions(
        mocap_positions, mocap_quaternions[:, [3, 0, 1, 2]], lever
    )
    reference_at_estimate = np.column_stack(
        [np.interp(estimate_t, mocap_t, mocap_positions[:, axis]) for axis in range(3)]
    )
    valid_estimate = (estimate_t >= mocap_t[0]) & (estimate_t <= mocap_t[-1])
    first_frame_s = int(frame_rows[0]["timestamp_ns"]) / 1e9
    frame_t = np.asarray([int(row["timestamp_ns"]) / 1e9 for row in frame_rows])
    reference_at_frames = np.column_stack(
        [np.interp(frame_t, mocap_t, mocap_positions[:, axis]) for axis in range(3)]
    )
    estimate_at_frames = np.column_stack(
        [np.interp(frame_t, estimate_t, estimate_positions[:, axis]) for axis in range(3)]
    )
    frame_valid = (
        (frame_t >= estimate_t[0])
        & (frame_t <= estimate_t[-1])
        & (frame_t >= mocap_t[0])
        & (frame_t <= mocap_t[-1])
    )
    output: dict[int, np.ndarray | None] = {}
    for index, event in enumerate(events):
        relative_estimate_t = estimate_t - first_frame_s
        pre = (
            valid_estimate
            & (relative_estimate_t >= event.start_s - 5.0)
            & (relative_estimate_t <= event.start_s - 1.0)
        )
        if np.count_nonzero(pre) < 30:
            output[index] = None
            continue
        pre_times = estimate_t[pre]
        if pre_times[-1] - pre_times[0] < 2.0:
            output[index] = None
            continue
        rotation, translation = _rigid_transform(
            estimate_positions[pre], reference_at_estimate[pre]
        )
        aligned = (rotation @ estimate_at_frames.T).T + translation
        errors = np.full(len(frame_rows), np.nan)
        errors[frame_valid] = np.linalg.norm(
            aligned[frame_valid] - reference_at_frames[frame_valid], axis=1
        )
        output[index] = errors
    return output


def build_event_metrics(
    run: DiagnosticRun,
    frame_rows: Sequence[dict[str, object]],
    events: Sequence[AngularEvent],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    event_rows = []
    impulse_rows = []
    local_errors = compute_event_local_errors(run, frame_rows, events)
    candidates_by_event = {
        index: enumerate_low_angular_candidates(frame_rows, event)
        for index, event in enumerate(events)
    }
    for index, event in enumerate(events):
        experiment_id = str(
            run.manifest.get("experiment_id", run.sequence)
        )
        covariates = _baseline_control_values(frame_rows, event.start_s)
        if covariates is None:
            covariates = {field: math.nan for field in CONTROL_COVARIATES}
        matching_event = {
            "sequence": run.sequence,
            "run": run.run,
            "start_s": event.start_s,
            "end_s": event.end_s,
            **covariates,
        }
        controls = select_matched_controls(
            matching_event,
            candidates_by_event[index],
            angular_events=events,
        ) if all(math.isfinite(float(value)) for value in covariates.values()) else []
        baseline_window = _window_rows(
            frame_rows, event.start_s - 5.0, event.start_s - 1.0
        )
        gp3p_outcome, gp3p_post_outcome, map_support_outcome = _event_outcomes(
            frame_rows, event.start_s, event.end_s
        )
        control_outcomes = [
            _event_outcomes(
                frame_rows,
                float(control["start_s"]),
                float(control["end_s"]),
            )
            for control in controls
        ]
        gp3p_control_values = _finite_values(
            outcome[0] for outcome in control_outcomes
        )
        gp3p_post_control_values = _finite_values(
            outcome[1] for outcome in control_outcomes
        )
        map_control_values = _finite_values(
            outcome[2] for outcome in control_outcomes
        )
        gp3p_control_outcome = (
            float(np.median(gp3p_control_values))
            if len(gp3p_control_values) else None
        )
        gp3p_post_control_outcome = (
            float(np.median(gp3p_post_control_values))
            if len(gp3p_post_control_values) else None
        )
        map_support_control_outcome = (
            float(np.median(map_control_values))
            if len(map_control_values) else None
        )
        pre_map_support = _window_median(
            baseline_window, "active_initialised_landmarks"
        )
        event_row: dict[str, object] = {
            "event_id": f"{experiment_id}-{run.run}-event{index:03d}",
            "experiment_id": experiment_id,
            "sequence": run.sequence,
            "run": run.run,
            "intervention": str(
                run.manifest.get("intervention", "baseline")
            ),
            "intervention_value": str(
                run.manifest.get("intervention_value", "none")
            ),
            "event_index": index,
            "start_s": event.start_s,
            "end_s": event.end_s,
            "peak_radps": event.peak_radps,
            "angular_integral": event.integrated_angle_rad,
            "matched_control_count": len(controls),
            "control_status": "matched" if controls else "no_matched_control",
            "pre_map_support": pre_map_support,
            "pre_image_sharpness": _window_median(
                baseline_window, "image_laplacian_variance"
            ),
            "mocap_translation": _window_sum(
                _window_rows(frame_rows, event.start_s, event.end_s),
                "mocap_body_translation_m",
            ),
            "mocap_rotation": _window_sum(
                _window_rows(frame_rows, event.start_s, event.end_s),
                "mocap_body_rotation_rad",
            ),
            "gp3p_outcome": gp3p_outcome,
            "gp3p_control_outcome": gp3p_control_outcome,
            "gp3p_outcome_paired": (
                gp3p_outcome - gp3p_control_outcome
                if gp3p_outcome is not None and gp3p_control_outcome is not None
                else None
            ),
            "gp3p_post_outcome": gp3p_post_outcome,
            "gp3p_post_control_outcome": gp3p_post_control_outcome,
            "gp3p_post_outcome_paired": (
                gp3p_post_outcome - gp3p_post_control_outcome
                if gp3p_post_outcome is not None
                and gp3p_post_control_outcome is not None
                else None
            ),
            "map_support_outcome": map_support_outcome,
            "map_support_control_outcome": map_support_control_outcome,
            "map_support_outcome_paired": (
                map_support_outcome - map_support_control_outcome
                if map_support_outcome is not None
                and map_support_control_outcome is not None
                else None
            ),
            "gp3p_onset_s": None,
            "persistent_drift_onset_s": None,
        }
        gp3p_onset = gp3p_failure_onset(frame_rows, event)
        if gp3p_onset is not None:
            event_row["gp3p_onset_s"] = gp3p_onset - event.start_s
        for family, metrics in MEDIATORS.items():
            for metric in metrics:
                baseline, mediator, delta = _event_metric_change(
                    frame_rows, metric, event.start_s, event.end_s
                )
                event_row[f"{metric}_baseline"] = baseline
                event_row[f"{metric}_mediator"] = mediator
                event_row[f"{metric}_delta"] = delta
                control_deltas = []
                for control in controls:
                    _, _, control_delta = _event_metric_change(
                        frame_rows,
                        metric,
                        float(control["start_s"]),
                        float(control["end_s"]),
                    )
                    if control_delta is not None:
                        control_deltas.append(control_delta)
                event_row[f"{metric}_paired_delta"] = (
                    delta - float(np.median(control_deltas))
                    if delta is not None and control_deltas else None
                )
                baseline_values = _finite_values(
                    row.get(metric) for row in baseline_window
                )
                mediator_scan = _window_rows(
                    frame_rows, event.start_s, event.end_s + 0.5
                )
                if len(baseline_values) and mediator_scan:
                    onset = detect_robust_onset(
                        [float(row["time_s"]) for row in mediator_scan],
                        [row.get(metric, math.nan) for row in mediator_scan],
                        baseline_values=baseline_values,
                        harmful_direction=HARMFUL_DIRECTIONS[metric],
                        epsilon=_metric_epsilon(metric),
                    )
                    event_row[f"{metric}_onset_s"] = (
                        onset - event.start_s if onset is not None else None
                    )
                    recovery_scan = _window_rows(
                        frame_rows, event.end_s, event.end_s + 10.0
                    )
                    recovery = detect_robust_recovery(
                        [float(row["time_s"]) for row in recovery_scan],
                        [row.get(metric, math.nan) for row in recovery_scan],
                        baseline_values=baseline_values,
                        search_start_s=event.end_s + 5.0,
                        epsilon=_metric_epsilon(metric),
                    ) if recovery_scan else None
                    event_row[f"{metric}_recovery_s"] = (
                        recovery - event.end_s if recovery is not None else None
                    )
        errors = local_errors.get(index)
        if errors is not None:
            relative_times = np.asarray(
                [float(row["time_s"]) for row in frame_rows]
            )
            persistent = (
                (relative_times >= event.start_s)
                & (relative_times <= event.end_s + 10.0)
                & np.isfinite(errors)
                & (errors >= 0.10)
            )
            for sample_index in range(max(0, len(persistent) - 2)):
                if np.all(persistent[sample_index : sample_index + 3]):
                    event_row["persistent_drift_onset_s"] = (
                        float(relative_times[sample_index]) - event.start_s
                    )
                    break
        event_rows.append(event_row)

        if run.sequence in IMPULSE_SEQUENCES:
            recovery_row: dict[str, object] = {
                "event_id": event_row["event_id"],
                "experiment_id": event_row["experiment_id"],
                "sequence": run.sequence,
                "run": run.run,
                "intervention": event_row["intervention"],
                "intervention_value": event_row["intervention_value"],
                "peak_radps": event.peak_radps,
                "angular_integral": event.integrated_angle_rad,
                "persistent_drift_onset_s": event_row[
                    "persistent_drift_onset_s"
                ],
            }
            for offset in (1, 3, 5, 10):
                window = _window_rows(
                    frame_rows,
                    event.end_s + offset - 0.25,
                    event.end_s + offset + 0.25,
                )
                for metric in (
                    "accepted_map_matches",
                    "active_initialised_landmarks",
                    "gp3p_inlier_ratio",
                    "temporal_ray_angle_p10_rad",
                ):
                    value = _window_median(window, metric)
                    baseline = _window_median(baseline_window, metric)
                    recovery_row[f"{metric}_deficit_at_{offset}s"] = (
                        value - baseline
                        if value is not None and baseline is not None else None
                    )
            impulse_rows.append(recovery_row)
    return event_rows, impulse_rows


def annotate_event_phases(
    rows: Sequence[dict[str, object]], events: Sequence[AngularEvent]
) -> None:
    for row in rows:
        time_s = float(row["time_s"])
        row["event_id"] = None
        row["event_phase"] = None
        row["time_from_event_start_s"] = None
        for index, event in enumerate(events):
            if not event.start_s - 5.0 <= time_s <= event.end_s + 10.0:
                continue
            row["event_id"] = f"event{index:03d}"
            row["time_from_event_start_s"] = time_s - event.start_s
            if event.start_s - 5.0 <= time_s <= event.start_s - 1.0:
                row["event_phase"] = "baseline"
            elif event.start_s <= time_s <= event.end_s:
                row["event_phase"] = "angular_input"
            elif time_s <= event.end_s + 0.5:
                row["event_phase"] = "mediator"
            elif time_s <= event.end_s + 2.0:
                row["event_phase"] = "gp3p_outcome"
            elif time_s <= event.end_s + 5.0:
                row["event_phase"] = "map_outcome"
            else:
                row["event_phase"] = "late_recovery"
            break


def bootstrap_mediation_model(
    rows: Sequence[dict[str, object]],
    *,
    mediator: str,
    outcome: str,
    samples: int = 500,
    seed: int = 20260807,
) -> dict[str, object]:
    base = fit_mediation_model(rows, mediator=mediator, outcome=outcome)
    if base["status"] in {"insufficient_model_rows", "collinear_design"}:
        return base
    sequences = sorted({str(row["sequence"]) for row in rows})
    if not sequences or samples < 1:
        return base
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence"])].append(row)
    rng = np.random.default_rng(seed)
    coefficient_names = (
        "angular_to_mediator",
        "angular_total",
        "angular_adjusted",
        "attenuation",
        "mediator_to_outcome",
    )
    distributions = {name: [] for name in coefficient_names}
    for _ in range(samples):
        bootstrap_rows = []
        for draw, sequence in enumerate(rng.choice(sequences, len(sequences), replace=True)):
            for source in by_sequence[str(sequence)]:
                copied = dict(source)
                copied["sequence"] = f"draw{draw}:{sequence}"
                bootstrap_rows.append(copied)
        fitted = fit_mediation_model(
            bootstrap_rows, mediator=mediator, outcome=outcome
        )
        if fitted["status"] in {"insufficient_model_rows", "collinear_design"}:
            continue
        for name in coefficient_names:
            distributions[name].append(float(fitted[name]))
    for name, values in distributions.items():
        if values:
            lower, median, upper = np.percentile(values, [2.5, 50.0, 97.5])
            base[f"{name}_ci_low"] = float(lower)
            base[f"{name}_bootstrap_median"] = float(median)
            base[f"{name}_ci_high"] = float(upper)
            base[f"{name}_direction_stability"] = float(
                max(np.mean(np.asarray(values) >= 0.0), np.mean(np.asarray(values) <= 0.0))
            )
    base["bootstrap_samples_valid"] = min(
        (len(values) for values in distributions.values()), default=0
    )
    return base


def build_mediation_rows(
    event_rows: Sequence[dict[str, object]], *, bootstrap_samples: int
) -> list[dict[str, object]]:
    output = []
    for family, metrics in MEDIATORS.items():
        for metric in metrics:
            model_rows = []
            for event in event_rows:
                paired = event.get(f"{metric}_paired_delta")
                if paired is None or event.get("control_status") != "matched":
                    continue
                model_rows.append(
                    {
                        "sequence": event["sequence"],
                        "angular_integral": event["angular_integral"],
                        "mediator_delta": paired,
                        "gp3p_outcome": event.get("gp3p_outcome_paired"),
                        "map_support_outcome": event.get(
                            "map_support_outcome_paired"
                        ),
                        "pre_map_support": event.get("pre_map_support"),
                        "mocap_translation": event.get("mocap_translation"),
                        "mocap_rotation": event.get("mocap_rotation"),
                        "pre_image_sharpness": event.get("pre_image_sharpness"),
                    }
                )
            for outcome in ("gp3p_outcome", "map_support_outcome"):
                result = bootstrap_mediation_model(
                    model_rows,
                    mediator="mediator_delta",
                    outcome=outcome,
                    samples=bootstrap_samples,
                )
                result.update(
                    {
                        "family": family,
                        "metric": metric,
                        "outcome": outcome,
                        "matched_control_coverage": (
                            len(model_rows) / len(event_rows) if event_rows else 0.0
                        ),
                        "interpretation": (
                            "attenuation_consistent_with_mediation_not_causal_indirect_effect"
                        ),
                    }
                )
                output.append(result)
    return output


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    preferred = [
        field for field in (
            "event_id", "sequence", "run", "event_index", "timestamp_ns",
            "frame_id", "time_s", "family", "metric", "outcome", "status",
        ) if field in fields
    ]
    ordered = preferred + [field for field in fields if field not in preferred]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for field in ordered:
                value = row.get(field)
                if value is None:
                    serialized[field] = ""
                elif isinstance(value, float) and not math.isfinite(value):
                    serialized[field] = ""
                else:
                    serialized[field] = value
            writer.writerow(serialized)


def _panel_note(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.01, 0.99, text, transform=axis.transAxes, va="top", ha="left",
        fontsize=8, bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )


def plot_impulse_timeline(
    frame_rows: Sequence[dict[str, object]], output: Path
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colours = {"20260806-175103": "#16843b", "20260806-175304": "#b3261e", "20260806-175539": "#7357a5"}
    impulse = [row for row in frame_rows if row.get("sequence") in IMPULSE_SEQUENCES]
    panels = (
        ("imu_gyro_max_radps", "角速度峰值 (rad/s)", "角速度冲击是输入，不是视觉失效标签。"),
        ("gp3p_failure_count", "GP3P 失败次数/帧", "失败若紧随冲击出现，支持预测或匹配一致性先退化。"),
        ("active_initialised_landmarks", "活跃已初始化 Landmark", "持续下降表示地图视觉支撑未能恢复。"),
    )
    for axis, (field, ylabel, rule) in zip(axes, panels):
        for (sequence, run), grouped in _group_rows(impulse, ("sequence", "run")):
            values = [(float(row["time_s"]), row.get(field)) for row in grouped if row.get(field) is not None]
            if values:
                axis.plot([v[0] for v in values], [float(v[1]) for v in values], color=colours[sequence], alpha=0.75, label=f"{sequence[-6:]}/{run}")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        _panel_note(axis, f"x：序列内时间；y：{ylabel}。\n规律：{rule}")
    axes[-1].set_xlabel("序列内时间 (s)")
    if impulse:
        axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    figure.suptitle("抗冲击序列：角速度输入到视觉支撑恢复的时间线")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _group_rows(rows: Sequence[dict[str, object]], fields: tuple[str, ...]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return sorted(grouped.items())


def plot_mediator_paths(
    event_rows: Sequence[dict[str, object]], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = (
        ("accepted_map_matches_paired_delta", "匹配数配对变化", "高角速度后匹配支撑越低，GP3P 失败风险越高。"),
        ("temporal_ray_angle_p10_rad_paired_delta", "时序射线角 p10 配对变化", "射线夹角下降对应更弱的时序三角化几何。"),
    )
    for axis, (field, ylabel, rule) in zip(axes, panels):
        for row in event_rows:
            value = row.get(field)
            if value is None:
                continue
            impulse = row.get("sequence") in IMPULSE_SEQUENCES
            axis.scatter(
                float(row["angular_integral"]), float(value),
                marker="^" if impulse else "o",
                color="#b3261e" if impulse else "#147d92", alpha=0.8,
            )
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.set_xlabel("高角速度事件积分转角 (rad)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        _panel_note(axis, f"x：事件角速度积分；y：{ylabel}。\n规律：{rule}")
    figure.suptitle("角速度到视觉碎片化的候选中介路径（同序列低角速度对照）")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_onset_recovery(
    event_rows: Sequence[dict[str, object]], output: Path
) -> None:
    metrics = (
        "accepted_map_matches",
        "temporal_ray_angle_p10_rad",
        "gp3p_inlier_ratio",
        "active_initialised_landmarks",
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(metrics))
    for event_index, row in enumerate(event_rows):
        onsets = [row.get(f"{metric}_onset_s") for metric in metrics]
        for metric_index, onset in enumerate(onsets):
            if onset is None:
                continue
            axis.scatter(
                metric_index + (event_index % 5 - 2) * 0.04,
                float(onset),
                marker="^" if row.get("sequence") in IMPULSE_SEQUENCES else "o",
                color="#b3261e" if row.get("sequence") in IMPULSE_SEQUENCES else "#147d92",
                alpha=0.75,
            )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_xticks(positions, ["匹配支撑", "时序射线角", "GP3P 内点率", "活跃 Landmark"])
    axis.set_ylabel("相对角速度事件开始的退化 onset (s)")
    axis.grid(axis="y", alpha=0.25)
    _panel_note(axis, "x：中介/下游状态；y：连续三帧越过 robust 阈值的首次时间。\n规律：中介 onset 早于 GP3P/地图支撑下降才支持中间链路的时间顺序。")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_summary_report(
    path: Path,
    event_rows: Sequence[dict[str, object]],
    model_rows: Sequence[dict[str, object]],
) -> None:
    matched = sum(row.get("control_status") == "matched" for row in event_rows)
    ordered = sum(
        row.get("accepted_map_matches_onset_s") is not None
        and row.get("gp3p_onset_s") is not None
        and float(row["accepted_map_matches_onset_s"]) <= float(row["gp3p_onset_s"])
        for row in event_rows
    )
    lines = [
        "# VIO 角速度-视觉碎片化因果诊断",
        "",
        f"- 角速度事件数：{len(event_rows)}",
        f"- 获得同序列低角速度对照：{matched}/{len(event_rows)}",
        f"- 匹配支撑退化不晚于 GP3P onset：{ordered}/{len(event_rows)}",
        "",
        "## 证据口径",
        "",
        "RANSAC 失败、短 landmark 寿命和频繁 observation removal 仍按下游状态解释。",
        "本分析只把事件前后时间顺序、剂量关系和 175103 恢复反例作为中介支持；",
        "在几何、曝光、纹理和 IMU 时延干预完成前，不把 attenuation 称为因果间接效应。",
        "2D-2D rotation-only/relative-pose 只解释初始化路径，runtime GP3P 单独汇报。",
        "",
        "## 模型状态",
        "",
    ]
    status_counts = Counter(str(row.get("status")) for row in model_rows)
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_runs(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output)
    tables = output / "tables"
    figures = output / "figures"
    diagnostics_root = Path(arguments.diagnostics_root)
    requested = set(arguments.sequences or ())
    matched_requests: set[str] = set()
    all_frames = []
    all_events = []
    all_impulse = []
    coverage = []
    for result_root in sorted(
        (path for path in diagnostics_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        run_names = sorted(
            path.name for path in result_root.glob("run*") if path.is_dir()
        )
        if not run_names:
            if result_root.name in requested:
                raise ValueError(f"{result_root}: no run directories")
            continue
        for run_name in run_names:
            diagnostic_run = load_diagnostic_run(
                diagnostics_root, result_root.name, run_name
            )
            aliases = {result_root.name, diagnostic_run.sequence}
            if requested and requested.isdisjoint(aliases):
                continue
            matched_requests.update(requested.intersection(aliases))
            frame_rows = aggregate_frame_metrics(diagnostic_run)
            angular_events, run_coverage = attach_sensor_metrics(
                diagnostic_run, frame_rows, data_root=Path(arguments.data_root)
            )
            annotate_event_phases(frame_rows, angular_events)
            event_rows, impulse_rows = build_event_metrics(
                diagnostic_run, frame_rows, angular_events
            )
            all_frames.extend(frame_rows)
            all_events.extend(event_rows)
            all_impulse.extend(impulse_rows)
            coverage.append(
                {
                    "experiment_id": str(
                        diagnostic_run.manifest.get(
                            "experiment_id", diagnostic_run.sequence
                        )
                    ),
                    "sequence": diagnostic_run.sequence,
                    "run": diagnostic_run.run,
                    "intervention": str(
                        diagnostic_run.manifest.get("intervention", "baseline")
                    ),
                    "intervention_value": str(
                        diagnostic_run.manifest.get(
                            "intervention_value", "none"
                        )
                    ),
                    **run_coverage,
                    "angular_events": len(angular_events),
                }
            )
    missing = requested - matched_requests
    if missing:
        raise ValueError(
            f"diagnostic sequences or experiment ids not found: {sorted(missing)}"
        )
    if not coverage:
        raise ValueError(f"{diagnostics_root}: no diagnostic runs selected")
    model_rows = build_mediation_rows(
        all_events, bootstrap_samples=arguments.bootstrap_samples
    )
    _write_rows(tables / "causal_frame_metrics.csv", all_frames)
    _write_rows(tables / "causal_diagnostics_coverage.csv", coverage)
    _write_rows(tables / "causal_event_metrics.csv", all_events)
    _write_rows(tables / "impulse_mediator_recovery.csv", all_impulse)
    _write_rows(tables / "causal_mediation_models.csv", model_rows)
    plot_impulse_timeline(all_frames, figures / "impulse_mediator_timeline.png")
    plot_mediator_paths(all_events, figures / "angular_to_fragmentation_mediator_paths.png")
    plot_onset_recovery(all_events, figures / "mediator_onset_recovery.png")
    _write_summary_report(output / "causal_diagnostics_summary.md", all_events, model_rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze event-aligned OKVIS VIO causal diagnostics"
    )
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/home/chenguyuan/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    analyze_runs(arguments)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(2)
