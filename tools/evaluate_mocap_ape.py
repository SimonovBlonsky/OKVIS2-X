#!/usr/bin/env python3

import argparse
import csv
import math
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
        return run_evo(
            reference_tum, estimate_tum, args.max_diff, args.save_results
        )

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
