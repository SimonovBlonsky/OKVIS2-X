#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def symlink_children(source: Path, target: Path, excluded: set[str]) -> None:
    for child in source.iterdir():
        if child.name not in excluded:
            (target / child.name).symlink_to(child.resolve(), child.is_dir())


@dataclass(frozen=True)
class VariantPlan:
    offset_ms: int
    offset_ns: int
    experiment_id: str
    target_dataset: Path


def sensor_root(source_dataset: Path) -> Path:
    root_imu = source_dataset / "imu0"
    nested_imu = source_dataset / "mav0" / "imu0"
    if root_imu.is_dir():
        return source_dataset
    if nested_imu.is_dir():
        return source_dataset / "mav0"
    raise ValueError(
        f"missing IMU directory: expected {root_imu} or {nested_imu}"
    )


def validate_required_sensors(source_sensor_root: Path) -> None:
    for camera in ("cam0", "cam1"):
        path = source_sensor_root / camera
        if not path.is_dir():
            raise ValueError(f"missing {camera} directory: {path}")
    imu_csv = source_sensor_root / "imu0" / "data.csv"
    if not imu_csv.is_file():
        raise ValueError(f"missing IMU data: {imu_csv}")


def sequence_from_dataset(source_dataset: Path) -> str:
    name = source_dataset.name
    return name[:-6] if name.endswith("_euroc") else name


def offset_token(offset_ms: int) -> str:
    if offset_ms < 0:
        return f"m{abs(offset_ms)}ms"
    if offset_ms > 0:
        return f"p{offset_ms}ms"
    return "0ms"


def plan_variants(
    source_dataset: Path, output_root: Path, offsets_ms: list[int]
) -> list[VariantPlan]:
    source_dataset = Path(source_dataset).resolve()
    output_root = Path(output_root).absolute()
    if not source_dataset.is_dir():
        raise ValueError(f"missing source dataset: {source_dataset}")
    source_sensor_root = sensor_root(source_dataset)
    validate_required_sensors(source_sensor_root)
    if not offsets_ms:
        raise ValueError("at least one --offsets-ms value is required")
    if len(offsets_ms) != len(set(offsets_ms)):
        raise ValueError("duplicate --offsets-ms value")
    sequence = sequence_from_dataset(source_dataset)
    plans = []
    for offset_ms in offsets_ms:
        token = offset_token(offset_ms)
        plans.append(
            VariantPlan(
                offset_ms=offset_ms,
                offset_ns=offset_ms * 1_000_000,
                experiment_id=f"{sequence}-imu-offset-{token}",
                target_dataset=(
                    output_root / f"{source_dataset.name}.imu-offset-{token}"
                ),
            )
        )
    for plan in plans:
        if plan.target_dataset.exists() or plan.target_dataset.is_symlink():
            raise ValueError(f"target already exists: {plan.target_dataset}")
    return plans


def write_shifted_imu_csv(source: Path, target: Path, offset_ns: int) -> int:
    row_count = 0
    with source.open("r", encoding="utf-8", newline="") as input_stream, target.open(
        "x", encoding="utf-8", newline=""
    ) as output_stream:
        for line_number, line in enumerate(input_stream, start=1):
            body = line.rstrip("\r\n")
            ending = line[len(body):]
            if not body or body.lstrip().startswith("#"):
                output_stream.write(line)
                continue
            fields = body.split(",", 1)
            timestamp_text = fields[0].strip()
            if len(fields) != 2 or not re.fullmatch(r"-?[0-9]+", timestamp_text):
                raise ValueError(
                    f"{source}:{line_number}: invalid timestamp {timestamp_text!r}"
                )
            timestamp = int(timestamp_text)
            shifted = timestamp + offset_ns
            if timestamp < 0 or shifted < 0:
                raise ValueError(
                    f"{source}:{line_number}: negative timestamp after offset"
                )
            output_stream.write(f"{shifted},{fields[1]}{ending}")
            row_count += 1
        if row_count == 0:
            raise ValueError(f"{source}: no IMU data rows")
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return row_count


def write_json_fsync(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def create_imu_time_offset_variant(
    source_dataset: Path,
    target_dataset: Path,
    offset_ns: int,
    experiment_id: str,
    sequence: str,
) -> Path:
    source_dataset = Path(source_dataset).resolve()
    target_dataset = Path(target_dataset).absolute()
    if not source_dataset.is_dir():
        raise ValueError(f"missing source dataset: {source_dataset}")
    source_sensor_root = sensor_root(source_dataset)
    validate_required_sensors(source_sensor_root)
    source_imu0 = source_sensor_root / "imu0"
    source_csv = source_imu0 / "data.csv"
    if not source_csv.is_file():
        raise ValueError(f"missing IMU data: {source_csv}")
    if target_dataset.exists() or target_dataset.is_symlink():
        raise ValueError(f"target already exists: {target_dataset}")
    if target_dataset == source_dataset or source_dataset in target_dataset.parents:
        raise ValueError("target dataset must not be inside source dataset")
    if isinstance(offset_ns, bool) or not isinstance(offset_ns, int):
        raise ValueError("offset_ns must be an integer")
    for name, value in (("experiment_id", experiment_id), ("sequence", sequence)):
        if not value or not value.strip():
            raise ValueError(f"{name} must be nonempty")

    target_dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dataset.name}.tmp-", dir=target_dataset.parent
        )
    )
    published = False
    try:
        sensor_relative = source_sensor_root.relative_to(source_dataset)
        temporary_sensor_root = temporary / sensor_relative
        temporary_imu0 = temporary_sensor_root / "imu0"
        temporary_imu0.mkdir(parents=True)
        if sensor_relative == Path("."):
            symlink_children(source_dataset, temporary, {"imu0"})
        else:
            symlink_children(source_dataset, temporary, {sensor_relative.name})
            symlink_children(source_sensor_root, temporary_sensor_root, {"imu0"})
        symlink_children(source_imu0, temporary_imu0, {"data.csv"})
        row_count = write_shifted_imu_csv(
            source_csv, temporary_imu0 / "data.csv", offset_ns
        )
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id.strip(),
            "dataset": str(target_dataset),
            "source_dataset": str(source_dataset),
            "sequence": sequence.strip(),
            "intervention": "imu_time_offset_ns",
            "intervention_value": str(offset_ns),
            "sensor_layout": str(sensor_relative),
            "row_count": row_count,
        }
        write_json_fsync(temporary / "variant_manifest.json", manifest)
        fsync_directory(temporary_imu0)
        if temporary_sensor_root != temporary:
            fsync_directory(temporary_sensor_root)
        fsync_directory(temporary)
        fsync_directory(target_dataset.parent)
        if target_dataset.exists() or target_dataset.is_symlink():
            raise ValueError(f"target already exists: {target_dataset}")
        temporary.rename(target_dataset)
        published = True
        fsync_directory(target_dataset.parent)
    except (OSError, ValueError) as error:
        if published and target_dataset.exists() and not temporary.exists():
            try:
                target_dataset.rename(temporary)
            except OSError as rollback_error:
                raise type(error)(
                    f"{error}; published target rollback failed: "
                    f"{rollback_error}; inspect {target_dataset}"
                ) from error
        raise type(error)(
            f"{error}; incomplete temporary variant retained at {temporary}"
        ) from error
    return target_dataset / "variant_manifest.json"


def write_dataset_manifest(
    path: Path, plans: list[VariantPlan], sequence: str
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "experiment_id",
                "dataset",
                "sequence",
                "intervention",
                "intervention_value",
            ),
        )
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "experiment_id": plan.experiment_id,
                    "dataset": str(plan.target_dataset),
                    "sequence": sequence,
                    "intervention": "imu_time_offset_ns",
                    "intervention_value": str(plan.offset_ns),
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def create_variants(
    source_dataset: Path, output_root: Path, offsets_ms: list[int]
) -> Path:
    source_dataset = Path(source_dataset).resolve()
    output_root = Path(output_root).absolute()
    plans = plan_variants(source_dataset, output_root, offsets_ms)
    manifest_path = output_root / "dataset_manifest.csv"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError(f"target already exists: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    sequence = sequence_from_dataset(source_dataset)
    for plan in plans:
        create_imu_time_offset_variant(
            source_dataset=source_dataset,
            target_dataset=plan.target_dataset,
            offset_ns=plan.offset_ns,
            experiment_id=plan.experiment_id,
            sequence=sequence,
        )
    write_dataset_manifest(manifest_path, plans, sequence)
    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an immutable Euroc dataset variant with shifted IMU time"
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--offsets-ms", type=int, nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    plans = plan_variants(
        arguments.source_dataset, arguments.output_root, arguments.offsets_ms
    )
    if arguments.dry_run:
        for plan in plans:
            print(
                f"would create {plan.experiment_id}: {plan.target_dataset} "
                f"({plan.offset_ns} ns)"
            )
        return 0
    manifest = create_variants(
        arguments.source_dataset, arguments.output_root, arguments.offsets_ms
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
